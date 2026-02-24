"""
Unified training harness for all models.

Handles:
    - wandb experiment tracking
    - Multi-seed runs [42, 123, 456, 789, 2026] (Section 16)
    - Checkpointing every 50 epochs
    - Resume from interrupted training
    - Mixed precision training (AMP)
    - Gradient clipping (max norm = 1.0)
    - Early stopping with patience = 100
"""

import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from config import load_yaml, resolve_path, set_all_seeds

# Multi-seed configuration (Section 16)
SEEDS: list[int] = [42, 123, 456, 789, 2026]


class EarlyStopping:
    """Early stopping with patience counter."""

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 1e-5,
        mode: str = "min",
    ) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.mode: str = mode
        self.counter: int = 0
        self.best_score: float | None = None
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Update and return True if should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved: bool = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_single_seed(
    model_class: type,
    model_kwargs: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Any,
    seed: int = 42,
    max_epochs: int = 2000,
    lr: float = 1e-4,
    patience: int = 100,
    checkpoint_every: int = 50,
    checkpoint_dir: str = "outputs/checkpoints",
    run_name: str = "pinn",
    use_wandb: bool = True,
    gradient_clip: float = 1.0,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """
    Train a model with a single seed.

    Purpose:
        Executes the full training loop with all bells and whistles:
        wandb logging, checkpointing, resume, AMP, gradient clipping,
        and early stopping.

    Inputs:
        model_class: Model class to instantiate.
        model_kwargs: Dict of kwargs for model constructor.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_fn: Callable(predictions, targets) → loss dict.
        seed: Random seed for this run.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        checkpoint_every: Save checkpoint every N epochs.
        checkpoint_dir: Directory for checkpoints.
        run_name: Name for this experiment run.
        use_wandb: Whether to log to wandb.
        gradient_clip: Max gradient norm for clipping.
        device_str: Device string.

    Outputs:
        Dict with final metrics, best epoch, training history.
    """
    set_all_seeds(seed)
    device: torch.device = torch.device(device_str)

    # Instantiate model
    model: nn.Module = model_class(**model_kwargs).to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=lr
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30
        )
    )

    # AMP scaler
    use_amp: bool = device_str == "cuda"
    scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=use_amp)

    # Early stopping
    early_stop: EarlyStopping = EarlyStopping(patience=patience)

    # Checkpoint resume
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path: str = os.path.join(
        checkpoint_dir, f"{run_name}_seed{seed}_latest.pt"
    )
    start_epoch: int = 0
    best_val_loss: float = float("inf")
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "lr": []
    }

    if os.path.exists(ckpt_path):
        checkpoint: dict[str, Any] = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)
        print(f"[RESUME] Training from epoch {start_epoch}")

    # Initialize wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project="avr-phm",
                name=f"{run_name}_seed{seed}",
                config={
                    "seed": seed,
                    "max_epochs": max_epochs,
                    "lr": lr,
                    "model": run_name,
                    **model_kwargs,
                },
                resume="allow",
            )
        except Exception:
            use_wandb = False

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_train_loss: float = 0.0
        n_train_batches: int = 0

        for batch in train_loader:
            x_batch: torch.Tensor = batch[0].to(device)
            y_batch: torch.Tensor = batch[1].to(device) if len(batch) > 1 else torch.zeros(1)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions: Any = model(x_batch)
                if isinstance(predictions, dict):
                    # PINN-style output
                    loss_dict: dict[str, torch.Tensor] = loss_fn(
                        predictions, {"forecast": y_batch}
                    )
                    loss: torch.Tensor = loss_dict.get(
                        "total", loss_dict.get("loss", torch.tensor(0.0))
                    )
                else:
                    loss = torch.nn.functional.mse_loss(predictions, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss: float = epoch_train_loss / max(n_train_batches, 1)

        # Validation
        model.eval()
        epoch_val_loss: float = 0.0
        n_val_batches: int = 0

        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                y_batch = batch[1].to(device) if len(batch) > 1 else torch.zeros(1)

                predictions = model(x_batch)
                if isinstance(predictions, dict):
                    loss_dict = loss_fn(predictions, {"forecast": y_batch})
                    loss = loss_dict.get("total", torch.tensor(0.0))
                else:
                    loss = torch.nn.functional.mse_loss(predictions, y_batch)

                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss: float = epoch_val_loss / max(n_val_batches, 1)

        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr: float = optimizer.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": current_lr,
                    "epoch": epoch,
                })
            except Exception:
                pass

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path: str = os.path.join(
                checkpoint_dir, f"{run_name}_seed{seed}_best.pt"
            )
            torch.save(model.state_dict(), best_path)

        # Checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, ckpt_path)

        # Print progress
        if (epoch + 1) % 50 == 0:
            print(
                f"[Epoch {epoch+1}/{max_epochs}] "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if early_stop.step(avg_val_loss):
            print(f"[EARLY STOP] at epoch {epoch+1}")
            break

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    return {
        "seed": seed,
        "best_val_loss": best_val_loss,
        "final_epoch": epoch,
        "history": history,
    }


def train_multi_seed(
    model_class: type,
    model_kwargs: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Any,
    seeds: list[int] | None = None,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """
    Train model with multiple seeds (Section 16) and aggregate results.

    Purpose:
        Runs training 5 times with different seeds and reports
        mean ± std of validation metrics.

    Inputs:
        model_class: Model class.
        model_kwargs: Model constructor kwargs.
        train_loader: Training data.
        val_loader: Validation data.
        loss_fn: Loss function.
        seeds: List of seeds. Default: [42, 123, 456, 789, 2026].
        **train_kwargs: Passed to train_single_seed.

    Outputs:
        Dict with per-seed results and aggregated statistics.
    """
    if seeds is None:
        seeds = SEEDS

    all_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        result: dict[str, Any] = train_single_seed(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            seed=seed,
            **train_kwargs,
        )
        all_results.append(result)

    # Aggregate
    val_losses: np.ndarray = np.array(
        [r["best_val_loss"] for r in all_results]
    )

    return {
        "per_seed": all_results,
        "val_loss_mean": float(np.mean(val_losses)),
        "val_loss_std": float(np.std(val_losses)),
        "seeds_used": seeds,
    }


def run_tests() -> None:
    """Sanity checks for training harness."""
    # Test 1: EarlyStopping works
    es: EarlyStopping = EarlyStopping(patience=3)
    for val in [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]:
        es.step(val)
    assert es.should_stop, "Early stopping should trigger after 3 no-improvement steps"

    # Test 2: Seed list is correct per spec
    assert SEEDS == [42, 123, 456, 789, 2026], "Seed list mismatch"

    print("[PASS] experiments/train.py — all tests passed.")


if __name__ == "__main__":
    import argparse
    import glob

    import pandas as pd

    parser = argparse.ArgumentParser(
        description="AVR-PHM training harness"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run sanity checks instead of training",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max training epochs (default: from model.yaml)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--single-seed", action="store_true",
        help="Train with only seed=42 instead of all 5 seeds",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        from config import get_device
        from features.engineer import (
            create_time_aware_splits,
            engineer_all_features,
        )
        from models.pinn import AVRPINN, compute_total_loss

        # ── Load config ───────────────────────────────────────────────
        model_cfg = load_yaml("model")
        paths_cfg = load_yaml("paths")
        device_str: str = get_device()

        max_epochs: int = args.max_epochs or model_cfg["pinn"]["max_epochs"]
        batch_size: int = model_cfg["pinn"]["batch_size"]
        lr: float = model_cfg["pinn"]["learning_rate"]
        patience: int = model_cfg["pinn"]["patience"]
        checkpoint_every: int = model_cfg["pinn"]["checkpoint_every_n_epochs"]
        window_size: int = model_cfg["data"]["window_size_samples"]
        stride: int = model_cfg["data"]["stride_samples"]

        # ── Load raw data ─────────────────────────────────────────────
        raw_dir = str(resolve_path(paths_cfg["data"]["raw_dir"]))
        csv_files = sorted(glob.glob(os.path.join(raw_dir, "avr_data_*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No data files found in {raw_dir}. "
                f"Run 'python -m data_gen.pipeline' first."
            )

        print(f"[DATA] Loading {len(csv_files)} data files from {raw_dir}")
        all_dfs: list[pd.DataFrame] = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            # Load matching fault log if it exists
            fault_path = csv_path.replace("avr_data_", "fault_log_")
            fault_df = None
            if os.path.exists(fault_path):
                fault_df = pd.read_csv(fault_path)
            featured_df = engineer_all_features(df, fault_log_df=fault_df)
            all_dfs.append(featured_df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[DATA] Combined dataset: {len(combined_df)} samples, "
              f"{len(combined_df.columns)} features")

        # ── Split ─────────────────────────────────────────────────────
        splits = create_time_aware_splits(combined_df)

        # ── Build windowed tensors ────────────────────────────────────
        # Select feature columns (exclude metadata and target columns)
        target_cols = [
            "fault_1s", "fault_5s", "fault_10s", "fault_30s",
            "fault_mechanism", "severity",
        ] + [f"voltage_next_{k}" for k in range(1, 11)]
        meta_cols = ["timestamp", "scenario", "run_id"]
        feature_cols = [
            c for c in combined_df.columns
            if c not in target_cols + meta_cols
        ]
        n_features = len(feature_cols)
        print(f"[DATA] Using {n_features} input features")

        def build_windows(
            df: pd.DataFrame,
            indices: np.ndarray,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Build sliding window tensors from DataFrame rows."""
            sub = df.iloc[indices].reset_index(drop=True)
            X_vals = sub[feature_cols].values.astype(np.float32)
            # Target: 10-step voltage forecast
            y_cols = [f"voltage_next_{k}" for k in range(1, 11)]
            Y_vals = sub[y_cols].values.astype(np.float32)

            windows_x: list[np.ndarray] = []
            windows_y: list[np.ndarray] = []
            for start in range(0, len(X_vals) - window_size, stride):
                end = start + window_size
                windows_x.append(X_vals[start:end])
                windows_y.append(Y_vals[end - 1])

            if not windows_x:
                return torch.zeros(0), torch.zeros(0)
            return (
                torch.tensor(np.array(windows_x)),
                torch.tensor(np.array(windows_y)),
            )

        print("[DATA] Building train/val sliding windows...")
        X_train, Y_train = build_windows(combined_df, splits["train"])
        X_val, Y_val = build_windows(combined_df, splits["val"])
        print(f"[DATA] Train windows: {X_train.shape}, Val windows: {X_val.shape}")

        if X_train.shape[0] == 0:
            raise RuntimeError("No training windows produced — check data.")

        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=batch_size, shuffle=False,
        )

        # ── Model ─────────────────────────────────────────────────────
        model_kwargs: dict[str, Any] = {
            "n_input_features": n_features,
            "window_size": window_size,
            "dropout_rate": 0.15,
        }

        def loss_fn(
            predictions: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            """Wrapper loss that uses MSE on voltage forecast."""
            forecast = predictions.get("forecast")
            target = targets.get("forecast")
            if forecast is not None and target is not None:
                mse = torch.nn.functional.mse_loss(forecast, target)
            else:
                mse = torch.tensor(0.0)
            return {"total": mse, "loss": mse}

        # ── Train ─────────────────────────────────────────────────────
        ckpt_dir = str(resolve_path(
            paths_cfg["outputs"]["checkpoints_dir"]
        ))
        results_dir = str(resolve_path(
            paths_cfg["outputs"]["results_dir"]
        ))
        os.makedirs(results_dir, exist_ok=True)

        train_kwargs: dict[str, Any] = {
            "max_epochs": max_epochs,
            "lr": lr,
            "patience": patience,
            "checkpoint_every": checkpoint_every,
            "checkpoint_dir": ckpt_dir,
            "run_name": "pinn",
            "use_wandb": not args.no_wandb,
            "gradient_clip": float(
                model_cfg["global"]["gradient_clipping"]
            ),
            "device_str": device_str,
        }

        if args.single_seed:
            result = train_single_seed(
                model_class=AVRPINN,
                model_kwargs=model_kwargs,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                seed=42,
                **train_kwargs,
            )
            print(f"\n[DONE] Best val loss: {result['best_val_loss']:.6f}")
        else:
            result = train_multi_seed(
                model_class=AVRPINN,
                model_kwargs=model_kwargs,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                **train_kwargs,
            )
            print(f"\n[DONE] Val loss: "
                  f"{result['val_loss_mean']:.6f} ± "
                  f"{result['val_loss_std']:.6f}")

        # Save summary
        import json
        summary_path = os.path.join(results_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                result, f, indent=2,
                default=lambda x: (
                    float(x) if isinstance(x, np.floating) else str(x)
                ),
            )
        print(f"[SAVED] Training summary to {summary_path}")
