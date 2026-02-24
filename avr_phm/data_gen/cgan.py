"""
Conditional WGAN-GP for rare fault augmentation.

Implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) using
GRU-based generator and bidirectional GRU critic for time-series
augmentation of rare fault scenarios.

Architecture:
    Generator: z(32) + condition(14) → Dense(128) → GRU(128, 2 layers)
               → TimeDistributed Dense(3) → output(batch, 100, 3)
    Critic:    sequence(100, 3) + condition(14) → Dense(64)
               → BiGRU(128, 2 layers) → Dense(64) → Dense(1)

Condition encoding (14-dim):
    - Scenario: 7-dim one-hot
    - Fault mechanism: 4-dim one-hot (none, thyristor, capacitor, terminal)
    - Severity: 3-dim one-hot (healthy, incipient, critical)

Training: WGAN-GP with n_critic=5, gradient penalty λ=10.

References:
    - Gulrajani et al. (2017). Improved Training of Wasserstein GANs.
    - Yoon et al. (2019). Time-series Generative Adversarial Networks.
"""

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


# ─── Condition Encoding Constants ────────────────────────────────────────────
SCENARIOS: list[str] = [
    "baseline", "arctic_cold", "desert_heat", "artillery_firing",
    "rough_terrain", "weapons_active", "emp_simulation",
]
FAULT_MECHANISMS: list[str] = ["none", "thyristor", "capacitor", "terminal"]
SEVERITY_LEVELS: list[str] = ["healthy", "incipient", "developing", "critical"]
CONDITION_DIM: int = len(SCENARIOS) + len(FAULT_MECHANISMS) + len(SEVERITY_LEVELS)  # 15


def encode_condition(
    scenario: str,
    fault_mechanism: str,
    severity: str,
) -> np.ndarray:
    """
    Encode scenario, fault mechanism, and severity as a one-hot condition vector.

    Purpose:
        Creates the 14-dimensional condition vector for conditional generation.

    Inputs:
        scenario: One of SCENARIOS.
        fault_mechanism: One of FAULT_MECHANISMS.
        severity: One of SEVERITY_LEVELS.

    Outputs:
        condition: np.ndarray of shape (14,) with one-hot encoding.
    """
    vec: np.ndarray = np.zeros(CONDITION_DIM, dtype=np.float32)

    # Scenario one-hot (positions 0-6)
    if scenario in SCENARIOS:
        vec[SCENARIOS.index(scenario)] = 1.0

    # Fault mechanism one-hot (positions 7-10)
    offset_fm: int = len(SCENARIOS)
    if fault_mechanism in FAULT_MECHANISMS:
        vec[offset_fm + FAULT_MECHANISMS.index(fault_mechanism)] = 1.0

    # Severity one-hot (positions 11-13)
    offset_sev: int = offset_fm + len(FAULT_MECHANISMS)
    if severity in SEVERITY_LEVELS:
        vec[offset_sev + SEVERITY_LEVELS.index(severity)] = 1.0

    return vec


class CGANGenerator(nn.Module):
    """
    GRU-based conditional generator for WGAN-GP.

    Architecture:
        Input: z ~ N(0,1) of shape (batch, latent_dim=32) +
               condition vector (14-dim)
        → Dense(128) + LayerNorm + ReLU
        → Repeat: (batch, seq_len=100, hidden=128)
        → GRU(hidden=128, num_layers=2, bidirectional=False)
        → TimeDistributed Dense(3)  # voltage, current, temperature
        → Output: (batch, seq_len=100, n_features=3)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = 128,
        seq_len: int = 100,
        n_features: int = 3,
        n_gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim
        self.seq_len: int = seq_len
        self.hidden_dim: int = hidden_dim

        # Input projection: latent + condition → hidden
        self.input_proj: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GRU for temporal generation
        self.gru: nn.GRU = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # TimeDistributed output projection
        self.output_proj: nn.Linear = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate synthetic time-series from noise and condition.

        Inputs:
            z: Latent noise tensor of shape (batch, latent_dim).
            condition: Condition vector of shape (batch, condition_dim).

        Outputs:
            Synthetic sequence of shape (batch, seq_len, n_features).
        """
        # Concatenate noise and condition
        x: torch.Tensor = torch.cat([z, condition], dim=-1)

        # Project to hidden dimension
        h: torch.Tensor = self.input_proj(x)

        # Repeat across sequence length
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Generate temporal structure
        out, _ = self.gru(h)

        # Project to output features
        output: torch.Tensor = self.output_proj(out)
        return output


class CGANCritic(nn.Module):
    """
    Bidirectional GRU-based critic (discriminator) for WGAN-GP.

    Architecture:
        Input: real/fake sequence (batch, seq_len=100, 3) +
               condition embedding
        → TimeDistributed Dense(64)
        → GRU(hidden=128, num_layers=2, bidirectional=True)
        → Flatten final hidden state
        → Concatenate with condition vector
        → Dense(64) + LeakyReLU(0.2)
        → Dense(1)  # No sigmoid — WGAN outputs unbounded score
    """

    def __init__(
        self,
        n_features: int = 3,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = 128,
        n_gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim

        # TimeDistributed input projection
        self.input_proj: nn.Linear = nn.Linear(n_features, 64)

        # Bidirectional GRU
        self.gru: nn.GRU = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Final classification head
        # BiGRU output: 2 * hidden_dim (forward + backward) from last layer
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(2 * hidden_dim + condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a sequence as real or fake (WGAN unbounded score).

        Inputs:
            sequence: Input tensor of shape (batch, seq_len, n_features).
            condition: Condition vector of shape (batch, condition_dim).

        Outputs:
            Score tensor of shape (batch, 1). Higher = more real.
        """
        # Project input features
        x: torch.Tensor = self.input_proj(sequence)

        # Bidirectional GRU
        _, h_n = self.gru(x)
        # h_n shape: (n_layers * 2, batch, hidden_dim)
        # Take the last layer's forward and backward hidden states
        h_forward: torch.Tensor = h_n[-2]  # Forward of last layer
        h_backward: torch.Tensor = h_n[-1]  # Backward of last layer
        h_combined: torch.Tensor = torch.cat(
            [h_forward, h_backward], dim=-1
        )

        # Concatenate with condition and classify
        h_cond: torch.Tensor = torch.cat(
            [h_combined, condition], dim=-1
        )
        score: torch.Tensor = self.head(h_cond)
        return score


def compute_gradient_penalty(
    critic: CGANCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute WGAN-GP gradient penalty.

    Purpose:
        Enforces the 1-Lipschitz constraint on the critic via gradient
        penalty on interpolated samples.

    Inputs:
        critic: Critic network.
        real: Real sequences (batch, seq_len, n_features).
        fake: Generated sequences (batch, seq_len, n_features).
        condition: Condition vectors (batch, condition_dim).
        device: Torch device.

    Outputs:
        Gradient penalty scalar tensor.

    Mathematical basis:
        GP = E[(||∇_x̂ D(x̂)||₂ - 1)²]
        where x̂ = ε*x_real + (1-ε)*x_fake, ε ~ U(0,1)
    """
    batch_size: int = real.size(0)
    eps: torch.Tensor = torch.rand(batch_size, 1, 1, device=device)
    interpolated: torch.Tensor = (
        eps * real + (1.0 - eps) * fake
    ).requires_grad_(True)

    scores: torch.Tensor = critic(interpolated, condition)
    gradients: torch.Tensor = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm: torch.Tensor = gradients.norm(2, dim=1)
    penalty: torch.Tensor = ((gradient_norm - 1.0) ** 2).mean()
    return penalty


def train_cgan(
    real_data: torch.Tensor,
    conditions: torch.Tensor,
    latent_dim: int = 32,
    batch_size: int = 32,
    max_epochs: int = 300,
    n_critic: int = 5,
    lr_g: float = 1e-4,
    lr_d: float = 2e-4,
    gp_weight: float = 10.0,
    checkpoint_dir: str = "outputs/checkpoints",
    device_str: str = "cpu",
) -> tuple[CGANGenerator, CGANCritic, dict[str, list[float]]]:
    """
    Train the WGAN-GP model.

    Purpose:
        Executes the full WGAN-GP training loop with gradient penalty,
        including checkpointing and loss logging.

    Inputs:
        real_data: Real sequences tensor (n_samples, seq_len, n_features).
        conditions: Condition vectors (n_samples, condition_dim).
        latent_dim: Dimension of latent noise vector.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs.
        n_critic: Number of critic updates per generator update.
        lr_g: Generator learning rate.
        lr_d: Critic (discriminator) learning rate.
        gp_weight: Gradient penalty coefficient λ (10.0 per WGAN-GP paper).
        checkpoint_dir: Directory for saving checkpoints.
        device_str: Device string ('cpu' or 'cuda').

    Outputs:
        (generator, critic, losses_dict) where losses_dict has keys
        'critic_loss' and 'generator_loss' with per-epoch values.

    Mathematical basis:
        WGAN-GP training:
            Critic loss: E[D(fake)] - E[D(real)] + λ * GP
            Generator loss: -E[D(fake)]
    """
    device: torch.device = torch.device(device_str)

    generator: CGANGenerator = CGANGenerator(
        latent_dim=latent_dim,
    ).to(device)

    critic: CGANCritic = CGANCritic().to(device)

    opt_g: torch.optim.Adam = torch.optim.Adam(
        generator.parameters(), lr=lr_g, betas=(0.0, 0.9)
    )
    opt_d: torch.optim.Adam = torch.optim.Adam(
        critic.parameters(), lr=lr_d, betas=(0.0, 0.9)
    )

    dataset: torch.utils.data.TensorDataset = (
        torch.utils.data.TensorDataset(real_data, conditions)
    )
    dataloader: torch.utils.data.DataLoader = (
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    )

    losses: dict[str, list[float]] = {
        "critic_loss": [],
        "generator_loss": [],
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check for existing checkpoint
    ckpt_path: str = os.path.join(checkpoint_dir, "cgan_latest.pt")
    start_epoch: int = 0
    if os.path.exists(ckpt_path):
        checkpoint: dict[str, Any] = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        generator.load_state_dict(checkpoint["generator_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        opt_g.load_state_dict(checkpoint["opt_g_state_dict"])
        opt_d.load_state_dict(checkpoint["opt_d_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        losses = checkpoint.get("losses", losses)
        print(f"[RESUME] Resuming cGAN training from epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs):
        epoch_c_loss: float = 0.0
        epoch_g_loss: float = 0.0
        n_batches: int = 0

        for batch_real, batch_cond in dataloader:
            batch_real = batch_real.to(device)
            batch_cond = batch_cond.to(device)
            current_batch_size: int = batch_real.size(0)

            # ─── Train Critic n_critic times ─────────────────────────────
            for _ in range(n_critic):
                z: torch.Tensor = torch.randn(
                    current_batch_size, latent_dim, device=device
                )
                fake: torch.Tensor = generator(z, batch_cond).detach()

                score_real: torch.Tensor = critic(
                    batch_real, batch_cond
                )
                score_fake: torch.Tensor = critic(fake, batch_cond)

                gp: torch.Tensor = compute_gradient_penalty(
                    critic, batch_real, fake, batch_cond, device
                )

                c_loss: torch.Tensor = (
                    score_fake.mean()
                    - score_real.mean()
                    + gp_weight * gp
                )

                opt_d.zero_grad()
                c_loss.backward()
                opt_d.step()

            # ─── Train Generator once ────────────────────────────────────
            z = torch.randn(
                current_batch_size, latent_dim, device=device
            )
            fake = generator(z, batch_cond)
            score_fake = critic(fake, batch_cond)
            g_loss: torch.Tensor = -score_fake.mean()

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_c_loss += c_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1

        # Log epoch losses
        if n_batches > 0:
            losses["critic_loss"].append(epoch_c_loss / n_batches)
            losses["generator_loss"].append(epoch_g_loss / n_batches)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"[Epoch {epoch+1}/{max_epochs}] "
                f"Critic: {losses['critic_loss'][-1]:.4f} | "
                f"Generator: {losses['generator_loss'][-1]:.4f}"
            )

        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "opt_g_state_dict": opt_g.state_dict(),
                    "opt_d_state_dict": opt_d.state_dict(),
                    "losses": losses,
                },
                ckpt_path,
            )
            print(f"[CHECKPOINT] Saved at epoch {epoch+1}")

    return generator, critic, losses


def generate_augmentation_data(
    generator: CGANGenerator,
    n_samples: int,
    condition: np.ndarray,
    latent_dim: int = 32,
    device_str: str = "cpu",
) -> np.ndarray:
    """
    Generate augmentation data using trained generator.

    Purpose:
        Produces synthetic time-series sequences conditioned on a
        specific scenario/fault/severity combination.

    Inputs:
        generator: Trained CGANGenerator.
        n_samples: Number of sequences to generate.
        condition: Condition vector of shape (condition_dim,).
        latent_dim: Latent noise dimension.
        device_str: Device string.

    Outputs:
        Synthetic sequences of shape (n_samples, seq_len, n_features).
    """
    device: torch.device = torch.device(device_str)
    generator.eval()

    with torch.no_grad():
        z: torch.Tensor = torch.randn(n_samples, latent_dim, device=device)
        cond: torch.Tensor = torch.tensor(
            condition, dtype=torch.float32, device=device
        ).unsqueeze(0).repeat(n_samples, 1)
        synthetic: torch.Tensor = generator(z, cond)

    return synthetic.cpu().numpy()


def build_augmentation_dataset(
    generator: CGANGenerator,
    n_fault_samples: int = 1000,
    latent_dim: int = 32,
    device_str: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implement master plan augmentation strategy.

    Purpose:
        Generate 3× as many fault samples as healthy, then down-sample
        to 1:4 ratio (fault:healthy) for final training set.

    Inputs:
        generator: Trained CGANGenerator.
        n_fault_samples: Base number of fault samples to generate.
        latent_dim: Latent noise dimension.
        device_str: Device string.

    Outputs:
        (augmented_sequences, augmented_conditions) arrays.
    """
    all_sequences: list[np.ndarray] = []
    all_conditions: list[np.ndarray] = []

    # Generate 3× fault samples for each mechanism × severity combo
    fault_combos: list[tuple[str, str]] = [
        (mech, sev)
        for mech in ["thyristor", "capacitor", "terminal"]
        for sev in ["incipient", "developing", "critical"]
    ]

    samples_per_combo: int = max(1, n_fault_samples // len(fault_combos))

    for mechanism, severity in fault_combos:
        for scenario in SCENARIOS:
            cond: np.ndarray = encode_condition(scenario, mechanism, severity)
            seqs: np.ndarray = generate_augmentation_data(
                generator, samples_per_combo, cond, latent_dim, device_str
            )
            all_sequences.append(seqs)
            cond_batch: np.ndarray = np.tile(cond, (len(seqs), 1))
            all_conditions.append(cond_batch)

    # Also generate healthy samples (1/3 the amount)
    healthy_per_scenario: int = max(
        1, (n_fault_samples * 3) // (4 * len(SCENARIOS))
    )
    for scenario in SCENARIOS:
        cond = encode_condition(scenario, "none", "healthy")
        seqs = generate_augmentation_data(
            generator, healthy_per_scenario, cond, latent_dim, device_str
        )
        all_sequences.append(seqs)
        cond_batch = np.tile(cond, (len(seqs), 1))
        all_conditions.append(cond_batch)

    augmented_sequences: np.ndarray = np.concatenate(all_sequences, axis=0)
    augmented_conditions: np.ndarray = np.concatenate(all_conditions, axis=0)

    print(f"[AUGMENT] Generated {len(augmented_sequences)} augmented samples")
    print(f"  Fault samples: ~{n_fault_samples * 3}")
    print(f"  Healthy samples: ~{healthy_per_scenario * len(SCENARIOS)}")

    return augmented_sequences, augmented_conditions


def run_tests() -> None:
    """Sanity checks for the WGAN-GP module."""
    # Test 1: Condition encoding produces correct shape
    cond: np.ndarray = encode_condition("baseline", "none", "healthy")
    assert cond.shape == (CONDITION_DIM,), (
        f"Condition dim should be {CONDITION_DIM}, got {cond.shape}"
    )
    assert cond.sum() == 3.0, "Exactly 3 entries should be 1.0"

    # Test 2: Generator produces correct output shape
    gen: CGANGenerator = CGANGenerator(latent_dim=32, condition_dim=CONDITION_DIM)
    z: torch.Tensor = torch.randn(4, 32)
    c: torch.Tensor = torch.zeros(4, CONDITION_DIM)
    output: torch.Tensor = gen(z, c)
    assert output.shape == (4, 100, 3), (
        f"Generator output shape should be (4, 100, 3), got {output.shape}"
    )

    # Test 3: Critic produces scalar scores
    crit: CGANCritic = CGANCritic(condition_dim=CONDITION_DIM)
    scores: torch.Tensor = crit(output.detach(), c)
    assert scores.shape == (4, 1), (
        f"Critic output should be (4, 1), got {scores.shape}"
    )

    # Test 4: Developing severity is encoded correctly
    cond_dev: np.ndarray = encode_condition("baseline", "capacitor", "developing")
    assert cond_dev.sum() == 3.0, "Should have 3 one-hot entries"

    print("[PASS] data_gen/cgan.py — all tests passed.")


if __name__ == "__main__":
    import argparse
    import glob

    import pandas as pd

    parser = argparse.ArgumentParser(description="WGAN-GP training and augmentation")
    parser.add_argument("--test", action="store_true", help="Run sanity checks only")
    parser.add_argument("--train", action="store_true", help="Train the cGAN")
    parser.add_argument("--augment", action="store_true",
                        help="Generate augmentation data using trained model")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory with raw CSV data")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                        help="Directory for synthetic output")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.train:
        # Load raw data and window it for cGAN training
        csv_files: list[str] = sorted(glob.glob(
            os.path.join(args.data_dir, "avr_data_*.csv")
        ))
        if not csv_files:
            print(f"[ERROR] No data CSVs found in {args.data_dir}")
            raise SystemExit(1)

        print(f"[LOAD] Found {len(csv_files)} data files")
        seq_len: int = 100
        all_windows: list[np.ndarray] = []
        all_conds: list[np.ndarray] = []

        for csv_path in csv_files:
            df: pd.DataFrame = pd.read_csv(csv_path)
            # Extract scenario and determine condition
            scenario_name: str = df["scenario"].iloc[0] if "scenario" in df.columns else "baseline"
            # Use "none"/"healthy" as default condition (no fault info in windows)
            cond: np.ndarray = encode_condition(scenario_name, "none", "healthy")

            # Extract V, I, T columns and create sliding windows
            cols: list[str] = ["voltage_v", "current_a", "temperature_c"]
            if not all(c in df.columns for c in cols):
                print(f"  [SKIP] {csv_path} — missing columns")
                continue

            data: np.ndarray = df[cols].values.astype(np.float32)
            # Normalize per-channel
            for ch in range(data.shape[1]):
                ch_std: float = float(np.std(data[:, ch]))
                ch_mean: float = float(np.mean(data[:, ch]))
                if ch_std > 1e-10:
                    data[:, ch] = (data[:, ch] - ch_mean) / ch_std

            n_windows: int = (len(data) - seq_len) // 10  # stride 10
            for wi in range(n_windows):
                start: int = wi * 10
                window: np.ndarray = data[start:start + seq_len]
                if len(window) == seq_len:
                    all_windows.append(window)
                    all_conds.append(cond)

            print(f"  [OK] {os.path.basename(csv_path)}: {n_windows} windows")

        real_data: torch.Tensor = torch.tensor(
            np.array(all_windows), dtype=torch.float32
        )
        conditions: torch.Tensor = torch.tensor(
            np.array(all_conds), dtype=torch.float32
        )
        print(f"[TRAIN] {len(real_data)} windows, condition_dim={CONDITION_DIM}")

        generator, critic, losses = train_cgan(
            real_data=real_data,
            conditions=conditions,
            latent_dim=32,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            device_str=args.device,
        )
        print(f"[DONE] cGAN training complete. Final critic loss: "
              f"{losses['critic_loss'][-1]:.4f}")

    elif args.augment:
        # Load trained generator and produce augmentation data
        ckpt_path: str = os.path.join(args.checkpoint_dir, "cgan_latest.pt")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] No checkpoint at {ckpt_path}. Train first with --train.")
            raise SystemExit(1)

        device: torch.device = torch.device(args.device)
        gen: CGANGenerator = CGANGenerator(
            latent_dim=32, condition_dim=CONDITION_DIM
        ).to(device)
        ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["generator_state_dict"])

        aug_seqs, aug_conds = build_augmentation_dataset(
            gen, n_fault_samples=1000, device_str=args.device
        )

        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "augmented_sequences.npy"), aug_seqs)
        np.save(os.path.join(args.output_dir, "augmented_conditions.npy"), aug_conds)
        print(f"[SAVED] Augmented data to {args.output_dir}/")

    else:
        run_tests()

