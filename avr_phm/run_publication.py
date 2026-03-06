"""
run_publication.py — Complete Tier-1 Publication Experiment Pipeline.

Hardware target: AMD Ryzen 7 5800H, 16 GB RAM, NVIDIA RTX 3050 (4 GB VRAM)

Runs ALL experiments needed for publication:
  1. Multi-seed PINN training (5 seeds × 200 epochs, early stopping)
  2. Baseline training (Threshold, Random Forest ×5 seeds)
  3. PHM metrics on held-out & stress-combo test sets
  4. Wilcoxon signed-rank significance tests
  5. SHAP explainability analysis
  6. IEEE-format publication figures (300 DPI, PNG + PDF)

Usage:
    python run_publication.py              # Runs everything
    python run_publication.py --train      # Train PINN + baselines only
    python run_publication.py --evaluate   # Evaluate + significance only
    python run_publication.py --shap       # SHAP only
    python run_publication.py --figures    # Figures only
    python run_publication.py --all        # Same as no args

Estimated wall-clock time (Ryzen 5800H + RTX 3050):
    PINN training:  ~15 min (5 seeds × 200 epochs, GPU)
    Baselines:      ~8 min  (RF ×5 seeds × 4 horizons, 8-thread CPU)
    Evaluation:     ~6 min  (RF ×5 seeds × 4 horizons × 2 test sets)
    SHAP:           ~3 min  (TreeExplainer on 500 samples)
    Figures:        ~1 min
    Total:          ~33 min
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE CONSTANTS — Single source of truth
# ═══════════════════════════════════════════════════════════════════════════

# ── Data configuration ────────────────────────────────────────────────────
SEQ_LEN: int = 100
STRIDE: int = 10
HORIZONS: list[str] = ["fault_1s", "fault_5s", "fault_10s", "fault_30s"]

# ── Hardware profile: Ryzen 7 5800H + RTX 3050 4 GB + 16 GB DDR4 ─────────
PINN_BATCH_SIZE: int = 256
VAL_CHUNK_SIZE: int = 256
RF_NJOBS: int = -1

# ── Seed configuration ────────────────────────────────────────────────────
SEEDS: list[int] = [42, 123, 456, 789, 2026]

# ── Training loop configuration ───────────────────────────────────────────
MAX_EPOCHS: int = 200
PINN_LR_INIT: float = 1e-3
PINN_LR_MIN: float = 1e-6
PINN_LR_TMAX: int = 200
EARLY_STOP_PATIENCE: int = 20
EARLY_STOP_DELTA: float = 1e-4
GRAD_CLIP_MAX_NORM: float = 1.0
PRIMARY_HORIZON: str = "fault_10s"

# ── Paths (relative to avr_phm/) ─────────────────────────────────────────
DATA_DIR: str = "data/raw"
FEATURED_DIR: str = "data/featured"
RESULTS_DIR: str = "outputs/results"
CHECKPOINT_DIR: str = "outputs/checkpoints"
FIGURES_DIR: str = "outputs/figures"


def pick_device() -> torch.device:
    """Select the best available device: CUDA → CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[DEVICE] Using CUDA: {name} ({mem:.1f} GB)")
        return torch.device("cuda")
    print("[DEVICE] CUDA not available, using CPU")
    return torch.device("cpu")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_featured_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load featured dataset (from cache or compute from raw)."""
    featured_path = os.path.join(FEATURED_DIR, "all_featured.csv")

    if os.path.exists(featured_path):
        print("[LOAD] Loading cached featured data...")
        df = pd.read_csv(featured_path)
    else:
        print("[LOAD] Computing features from raw data...")
        from features.engineer import engineer_all_features
        csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "avr_data_*.csv")))
        if not csv_files:
            print(f"[ERROR] No data CSVs in {DATA_DIR}")
            sys.exit(1)
        dfs = []
        for f in csv_files:
            raw = pd.read_csv(f)
            fault_f = f.replace("avr_data_", "fault_log_")
            fault_df = pd.read_csv(fault_f) if os.path.exists(fault_f) else None
            featured = engineer_all_features(raw, fault_log_df=fault_df)
            dfs.append(featured)
            print(f"  [OK] {os.path.basename(f)}: {len(featured)} rows × {len(featured.columns)} cols")
        df = pd.concat(dfs, ignore_index=True)
        os.makedirs(FEATURED_DIR, exist_ok=True)
        df.to_csv(featured_path, index=False)
        print(f"[SAVED] {featured_path}")

    from features.engineer import create_time_aware_splits
    splits = create_time_aware_splits(df)
    return df, splits


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get input feature columns (exclude targets and metadata)."""
    exclude_prefixes = [
        "fault_1s", "fault_5s", "fault_10s", "fault_30s",
        "fault_mechanism", "severity", "rul_seconds",
        "voltage_next_", "timestamp", "run_id", "scenario",
    ]
    return [
        col for col in df.columns
        if not any(col.startswith(p) for p in exclude_prefixes)
    ]


def prepare_windowed_data(
    X: np.ndarray, seq_len: int = SEQ_LEN, stride: int = STRIDE
) -> np.ndarray:
    """Create sliding windows: (N, seq_len, n_features)."""
    n_windows = max(0, (len(X) - seq_len) // stride)
    n_features = X.shape[1]
    windows = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
    for i in range(n_windows):
        s = i * stride
        windows[i] = X[s: s + seq_len]
    return windows


def get_window_target_indices(n_rows: int, seq_len: int, stride: int) -> np.ndarray:
    """Return the label row index for each sliding window (Bug 09 fix).

    Window i spans rows [i*stride : i*stride + seq_len].
    The label for window i is at the last row: i*stride + seq_len - 1.
    """
    n_windows = max(0, (n_rows - seq_len) // stride)
    return np.array([i * stride + seq_len - 1 for i in range(n_windows)], dtype=np.int64)


# ─── Early Stopping (AUROC-based) ────────────────────────────────────────────

class EarlyStopping:
    """Early stopping based on validation AUROC (higher is better).

    Monitors val_auroc on PRIMARY_HORIZON. Saves the best model checkpoint
    whenever val_auroc improves by more than delta. Signals stop after
    patience consecutive epochs without sufficient improvement.
    """

    def __init__(
        self,
        patience: int,
        delta: float,
        checkpoint_path: str,
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def step(self, val_auroc: float, model: torch.nn.Module, epoch: int) -> None:
        """Call once per epoch. Saves checkpoint if improved, increments counter otherwise."""
        improvement = val_auroc - self.best_score

        if improvement > self.delta:
            self.best_score = val_auroc
            self.counter = 0
            self.best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auroc": val_auroc,
            }, self.checkpoint_path)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"    [ES] New best val_auroc={val_auroc:.6f} at epoch {epoch+1}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"    [EARLY STOP] at epoch {epoch+1} "
                          f"(no improvement for {self.patience} epochs, "
                          f"best={self.best_score:.6f} at epoch {self.best_epoch+1})")

    def load_best(self, model: torch.nn.Module) -> float:
        """Load the best saved checkpoint into model. Returns best val_auroc."""
        if not os.path.exists(self.checkpoint_path):
            print(f"    [ES] WARNING: No checkpoint at {self.checkpoint_path}")
            return self.best_score
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if self.verbose:
            print(f"    [ES] Restored best checkpoint: "
                  f"val_auroc={ckpt['val_auroc']:.6f} (epoch {ckpt['epoch']+1})")
        return float(ckpt["val_auroc"])


def compute_val_auroc(
    model: torch.nn.Module,
    val_windows: torch.Tensor,
    val_targets: torch.Tensor,
    horizon: str,
    chunk_size: int = VAL_CHUNK_SIZE,
) -> float:
    """Compute validation AUROC for a single horizon without gradients.

    Returns 0.5 if no positive labels or predictions are constant.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    preds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for vs in range(0, len(val_windows), chunk_size):
            ve = min(vs + chunk_size, len(val_windows))
            out = model(val_windows[vs:ve])
            preds_list.append(torch.sigmoid(out[horizon].squeeze(-1)).cpu())

    all_preds = torch.cat(preds_list).numpy()
    all_targets = val_targets.numpy() if isinstance(val_targets, torch.Tensor) else val_targets

    if all_targets.sum() == 0 or np.std(all_preds) < 1e-8:
        return 0.5

    try:
        return float(roc_auc_score(all_targets, all_preds))
    except ValueError:
        return 0.5


# ─── Section 1: Multi-Seed PINN Training ─────────────────────────────────────

def train_pinn_multiseed(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Train PINN across 5 seeds on GPU with early stopping."""
    from models.pinn import AVRPINN

    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    results: dict[str, Any] = {"seeds": {}, "summary": {}, "n_features": n_features}

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]

    # ─── Normalize ────────────────────────────────────────────────────
    train_X = train_df[feature_cols].values.astype(np.float32)
    val_X = val_df[feature_cols].values.astype(np.float32)

    # Replace NaN/Inf before computing stats
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=1e6, neginf=-1e6)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=1e6, neginf=-1e6)

    feat_mean = train_X.mean(axis=0)
    feat_std = train_X.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    train_X = (train_X - feat_mean) / feat_std
    val_X = (val_X - feat_mean) / feat_std

    np.save(os.path.join(RESULTS_DIR, "feat_mean.npy"), feat_mean)
    np.save(os.path.join(RESULTS_DIR, "feat_std.npy"), feat_std)

    # ─── Create sliding windows ──────────────────────────────────────
    train_windows = torch.from_numpy(prepare_windowed_data(train_X))
    val_windows = torch.from_numpy(prepare_windowed_data(val_X))

    # Targets: take the label at the *end* of each window (Bug 09 fix)
    train_label_indices = get_window_target_indices(len(train_df), SEQ_LEN, STRIDE)
    val_label_indices = get_window_target_indices(len(val_df), SEQ_LEN, STRIDE)

    # Bug 03 fix: build multi-task target dicts for all 4 horizons
    train_targets_dict = {
        h: torch.from_numpy(train_df[h].values[train_label_indices].astype(np.float32))
        for h in HORIZONS
    }
    val_targets_dict = {
        h: torch.from_numpy(val_df[h].values[val_label_indices].astype(np.float32))
        for h in HORIZONS
    }
    # Reference for class-weight computation
    train_targets_ref = train_targets_dict["fault_10s"]

    print(f"[PINN] {len(train_windows)} train / {len(val_windows)} val windows")
    print(f"[PINN] Input shape: ({SEQ_LEN}, {n_features}), device={device}")
    print(f"[PINN] Positive rate (train): {train_targets_ref.mean():.4f}")

    # Move validation data to GPU once (fits in 4GB easily)
    val_windows_gpu = val_windows.to(device)
    # Move all targets to GPU once
    train_targets_dict_gpu = {h: t.to(device) for h, t in train_targets_dict.items()}
    val_targets_dict_gpu = {h: t.to(device) for h, t in val_targets_dict.items()}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"[PINN SEED {seed}] Training up to {MAX_EPOCHS} epochs...")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ─── Instantiate model ────────────────────────────────────
        # PINN forward() expects (batch, window_size, n_features)
        # and internally does x.permute(0, 2, 1) for Conv1d.
        model = AVRPINN(n_input_features=n_features).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PINN_LR_INIT, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=PINN_LR_TMAX, eta_min=PINN_LR_MIN,
        )

        # Class-weight for imbalanced data
        n_pos = train_targets_ref.sum().item()
        n_neg = len(train_targets_ref) - n_pos
        pos_weight = n_neg / max(n_pos, 1.0)
        print(f"  Pos weight: {pos_weight:.1f} (1:{pos_weight:.0f} ratio)")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"pinn_seed{seed}_best.pt")
        stopper = EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            delta=EARLY_STOP_DELTA,
            checkpoint_path=ckpt_path,
        )
        seed_losses: dict[str, list[float]] = {"train": [], "val": []}

        t_seed_start = time.time()
        print(f"  LR: {PINN_LR_INIT} -> {PINN_LR_MIN} (cosine, T_max={PINN_LR_TMAX})")
        print(f"  Early stop: patience={EARLY_STOP_PATIENCE}, "
              f"delta={EARLY_STOP_DELTA}, criterion=val_auroc[{PRIMARY_HORIZON}]")

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(len(train_windows))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_windows), PINN_BATCH_SIZE):
                end = min(start + PINN_BATCH_SIZE, len(train_windows))
                idx = perm[start:end]

                # Input: (batch, seq_len, n_features) — PINN permutes internally
                batch_x = train_windows[idx].to(device)

                optimizer.zero_grad()
                output = model(batch_x)

                # Bug 01+03 fix: use compute_total_loss with all 4 horizons
                batch_targets = {h: train_targets_dict_gpu[h][idx] for h in HORIZONS}

                # Bug 10 fix: compute physics residuals using PyTorch finite-difference
                v_col = feature_cols.index("voltage_v") if "voltage_v" in feature_cols else 0
                i_col = feature_cols.index("current_a") if "current_a" in feature_cols else 1
                v_seq = batch_x[:, :, v_col]
                i_seq = batch_x[:, :, i_col]
                t_batch = torch.linspace(0, (SEQ_LEN - 1) * 0.1, SEQ_LEN, device=device).unsqueeze(0).expand(batch_x.shape[0], -1)
                physics_residual = model.physics_residual.compute_residuals(t_batch, v_seq, i_seq, output.get("forecast"))

                from models.pinn import compute_total_loss
                loss_dict = compute_total_loss(
                    predictions=output,
                    targets=batch_targets,
                    physics_residuals=physics_residual,
                    lambda_physics=0.1,
                    lambda_data=0.5,
                    lambda_fault=1.0,
                )
                loss = loss_dict["total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            seed_losses["train"].append(avg_train)

            # ─── Validation (loss + AUROC) ────────────────────────
            model.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                n_val_chunks = 0
                for vs in range(0, len(val_windows_gpu), VAL_CHUNK_SIZE):
                    ve = min(vs + VAL_CHUNK_SIZE, len(val_windows_gpu))
                    chunk_out = model(val_windows_gpu[vs:ve])
                    chunk_targets = {h: val_targets_dict_gpu[h][vs:ve] for h in HORIZONS}
                    dummy_residuals = torch.zeros(ve - vs, 3, device=device)
                    v_loss = compute_total_loss(
                        chunk_out, chunk_targets, dummy_residuals,
                        lambda_physics=0.0, lambda_data=0.5, lambda_fault=1.0,
                    )["total"]
                    val_loss_total += v_loss.item()
                    n_val_chunks += 1
                val_loss = val_loss_total / max(n_val_chunks, 1)

            seed_losses["val"].append(val_loss)

            # ─── AUROC-based early stopping ───────────────────────
            val_auroc = compute_val_auroc(
                model=model,
                val_windows=val_windows_gpu,
                val_targets=val_targets_dict[PRIMARY_HORIZON],
                horizon=PRIMARY_HORIZON,
                chunk_size=VAL_CHUNK_SIZE,
            )

            stopper.step(val_auroc=val_auroc, model=model, epoch=epoch)

            if (epoch + 1) % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_seed_start
                print(
                    f"  [Epoch {epoch+1:3d}/{MAX_EPOCHS}] "
                    f"train={avg_train:.6f}  val_loss={val_loss:.6f}  "
                    f"val_auroc={val_auroc:.4f}  lr={lr:.6f}  "
                    f"({elapsed:.0f}s)"
                )

            if stopper.should_stop:
                break

        # Restore best checkpoint
        best_auroc = stopper.load_best(model)
        elapsed = time.time() - t_seed_start
        results["seeds"][str(seed)] = {
            "best_val_auroc": best_auroc,
            "best_epoch": stopper.best_epoch + 1,
            "final_epoch": epoch + 1,
            "time_seconds": round(elapsed, 1),
            "train_losses": seed_losses["train"],
            "val_losses": seed_losses["val"],
        }
        print(f"  [DONE] Seed {seed}: best_val_auroc={best_auroc:.6f} "
              f"at epoch {stopper.best_epoch+1} ({elapsed:.0f}s)")

    # ─── Summary statistics ───────────────────────────────────────
    aurocs = [
        v["best_val_auroc"] for v in results["seeds"].values()
        if v["best_val_auroc"] > 0
    ]
    if aurocs:
        results["summary"] = {
            "mean_val_auroc": round(float(np.mean(aurocs)), 6),
            "std_val_auroc": round(float(np.std(aurocs)), 6),
            "n_seeds": len(aurocs),
        }
        print(f"\n[PINN SUMMARY] Val AUROC: {np.mean(aurocs):.6f} "
              f"± {np.std(aurocs):.6f} ({len(aurocs)} seeds)")

    return results


def train_lstm_multiseed(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Train pure deep learning LSTM baseline across 5 seeds on GPU."""
    from models.baseline_lstm import AVRLSTM, compute_lstm_loss

    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    results: dict[str, Any] = {"seeds": {}, "summary": {}, "n_features": n_features}

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]

    # ─── Normalize ────────────────────────────────────────────────────
    train_X = train_df[feature_cols].values.astype(np.float32)
    val_X = val_df[feature_cols].values.astype(np.float32)

    train_X = np.nan_to_num(train_X, nan=0.0, posinf=1e6, neginf=-1e6)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Use existing normalization stats (created by PINN if run first)
    mean_path = os.path.join(RESULTS_DIR, "feat_mean.npy")
    std_path = os.path.join(RESULTS_DIR, "feat_std.npy")
    if os.path.exists(mean_path):
        feat_mean = np.load(mean_path)
        feat_std = np.load(std_path)
    else:
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0

    train_X = (train_X - feat_mean) / feat_std
    val_X = (val_X - feat_mean) / feat_std

    # ─── Create sliding windows ──────────────────────────────────────
    train_windows = torch.from_numpy(prepare_windowed_data(train_X))
    val_windows = torch.from_numpy(prepare_windowed_data(val_X))

    # Targets
    train_targets_dict = {
        h: torch.from_numpy(train_df[h].values[SEQ_LEN - 1::STRIDE][:len(train_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }
    val_targets_dict = {
        h: torch.from_numpy(val_df[h].values[SEQ_LEN - 1::STRIDE][:len(val_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }

    train_targets = train_targets_dict["fault_10s"].cpu() # For class weights

    print(f"[LSTM] {len(train_windows)} train / {len(val_windows)} val windows")

    val_windows_gpu = val_windows.to(device)

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"[LSTM SEED {seed}] Training up to {MAX_EPOCHS} epochs...")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = AVRLSTM(n_input_features=n_features).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PINN_LR_INIT, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=PINN_LR_MIN,
        )

        # Class-weight
        n_pos = train_targets.sum().item()
        n_neg = len(train_targets) - n_pos
        pos_weight = n_neg / max(n_pos, 1.0)

        stopper = EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            delta=EARLY_STOP_DELTA,
            ckpt_path=os.path.join(CHECKPOINT_DIR, f"lstm_seed{seed}_best.pt"),
            seed=seed,
            n_features=n_features,
        )
        seed_losses: dict[str, list[float]] = {"train": [], "val": []}

        t_seed_start = time.time()

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(len(train_windows))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_windows), PINN_BATCH_SIZE):
                end = min(start + PINN_BATCH_SIZE, len(train_windows))
                idx = perm[start:end]

                batch_x = train_windows[idx].to(device)
                batch_targets = {h: t[idx] for h, t in train_targets_dict.items()}

                optimizer.zero_grad()
                output = model(batch_x)

                losses = compute_lstm_loss(output, batch_targets)
                loss = losses["total"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            seed_losses["train"].append(avg_train)

            # ─── Validation (loss + AUROC) ────────────────────────
            model.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                n_val_batches = 0
                for vs in range(0, len(val_windows_gpu), VAL_CHUNK_SIZE):
                    ve = min(vs + VAL_CHUNK_SIZE, len(val_windows_gpu))
                    chunk_out = model(val_windows_gpu[vs:ve])
                    chunk_targets = {h: t[vs:ve] for h, t in val_targets_dict.items()}
                    v_loss = compute_lstm_loss(chunk_out, chunk_targets)["total"]
                    val_loss_total += v_loss.item()
                    n_val_batches += 1
                val_loss = val_loss_total / max(n_val_batches, 1)

            seed_losses["val"].append(val_loss)

            # ─── AUROC-based early stopping ───────────────────────
            val_auroc = compute_val_auroc(
                model=model,
                val_windows=val_windows_gpu,
                val_targets=val_targets_dict[PRIMARY_HORIZON],
                horizon=PRIMARY_HORIZON,
                chunk_size=VAL_CHUNK_SIZE,
            )

            stopper.step(val_auroc=val_auroc, model=model, epoch=epoch)

            if (epoch + 1) % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_seed_start
                print(
                    f"  [Epoch {epoch+1:3d}/{MAX_EPOCHS}] "
                    f"train={avg_train:.6f}  val_loss={val_loss:.6f}  "
                    f"val_auroc={val_auroc:.4f}  lr={lr:.6f}  "
                    f"({elapsed:.0f}s)"
                )

            if stopper.should_stop:
                break

        # Restore best checkpoint
        best_auroc = stopper.load_best(model)
        elapsed = time.time() - t_seed_start
        results["seeds"][str(seed)] = {
            "best_val_auroc": best_auroc,
            "best_epoch": stopper.best_epoch + 1,
            "final_epoch": epoch + 1,
            "time_seconds": round(elapsed, 1),
            "train_losses": seed_losses["train"],
            "val_losses": seed_losses["val"],
        }
        print(f"  [DONE] Seed {seed}: best_val_auroc={best_auroc:.6f} "
              f"at epoch {stopper.best_epoch+1} ({elapsed:.0f}s)")

    # ─── Summary statistics ───────────────────────────────────────
    aurocs = [
        v["best_val_auroc"] for v in results["seeds"].values()
        if v["best_val_auroc"] > 0
    ]
    if aurocs:
        results["summary"] = {
            "mean_val_auroc": round(float(np.mean(aurocs)), 6),
            "std_val_auroc": round(float(np.std(aurocs)), 6),
            "n_seeds": len(aurocs),
        }
        print(f"\n[LSTM SUMMARY] Val AUROC: {np.mean(aurocs):.6f} "
              f"± {np.std(aurocs):.6f} ({len(aurocs)} seeds)")

    return results

def train_cnn_multiseed(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Train pure deep learning 1D-CNN baseline across 5 seeds on GPU."""
    from models.baseline_cnn import AVRCNN, compute_cnn_loss

    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    results: dict[str, Any] = {"seeds": {}, "summary": {}, "n_features": n_features}

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]

    train_X = train_df[feature_cols].values.astype(np.float32)
    val_X = val_df[feature_cols].values.astype(np.float32)

    train_X = np.nan_to_num(train_X, nan=0.0, posinf=1e6, neginf=-1e6)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=1e6, neginf=-1e6)

    mean_path = os.path.join(RESULTS_DIR, "feat_mean.npy")
    std_path = os.path.join(RESULTS_DIR, "feat_std.npy")
    if os.path.exists(mean_path):
        feat_mean = np.load(mean_path)
        feat_std = np.load(std_path)
    else:
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0

    train_X = (train_X - feat_mean) / feat_std
    val_X = (val_X - feat_mean) / feat_std

    train_windows = torch.from_numpy(prepare_windowed_data(train_X))
    val_windows = torch.from_numpy(prepare_windowed_data(val_X))

    train_targets_dict = {
        h: torch.from_numpy(train_df[h].values[SEQ_LEN - 1::STRIDE][:len(train_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }
    val_targets_dict = {
        h: torch.from_numpy(val_df[h].values[SEQ_LEN - 1::STRIDE][:len(val_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }

    train_targets = train_targets_dict["fault_10s"].cpu() 

    print(f"[CNN] {len(train_windows)} train / {len(val_windows)} val windows")
    val_windows_gpu = val_windows.to(device)

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"[CNN SEED {seed}] Training up to {MAX_EPOCHS} epochs...")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = AVRCNN(n_input_features=n_features).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PINN_LR_INIT, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=PINN_LR_MIN,
        )

        n_pos = train_targets.sum().item()
        n_neg = len(train_targets) - n_pos
        pos_weight = n_neg / max(n_pos, 1.0)

        stopper = EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            delta=EARLY_STOP_DELTA,
            ckpt_path=os.path.join(CHECKPOINT_DIR, f"cnn_seed{seed}_best.pt"),
            seed=seed,
            n_features=n_features,
        )
        seed_losses: dict[str, list[float]] = {"train": [], "val": []}

        t_seed_start = time.time()

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(len(train_windows))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_windows), PINN_BATCH_SIZE):
                end = min(start + PINN_BATCH_SIZE, len(train_windows))
                idx = perm[start:end]

                batch_x = train_windows[idx].to(device)
                batch_targets = {h: t[idx] for h, t in train_targets_dict.items()}

                optimizer.zero_grad()
                output = model(batch_x)

                losses = compute_cnn_loss(output, batch_targets)
                loss = losses["total"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            seed_losses["train"].append(avg_train)

            # ─── Validation (loss + AUROC) ────────────────────────
            model.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                n_val_batches = 0
                for vs in range(0, len(val_windows_gpu), VAL_CHUNK_SIZE):
                    ve = min(vs + VAL_CHUNK_SIZE, len(val_windows_gpu))
                    chunk_out = model(val_windows_gpu[vs:ve])
                    chunk_targets = {h: t[vs:ve] for h, t in val_targets_dict.items()}
                    v_loss = compute_cnn_loss(chunk_out, chunk_targets)["total"]
                    val_loss_total += v_loss.item()
                    n_val_batches += 1
                val_loss = val_loss_total / max(n_val_batches, 1)

            seed_losses["val"].append(val_loss)

            # ─── AUROC-based early stopping ───────────────────────
            val_auroc = compute_val_auroc(
                model=model,
                val_windows=val_windows_gpu,
                val_targets=val_targets_dict[PRIMARY_HORIZON],
                horizon=PRIMARY_HORIZON,
                chunk_size=VAL_CHUNK_SIZE,
            )

            stopper.step(val_auroc=val_auroc, model=model, epoch=epoch)

            if (epoch + 1) % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_seed_start
                print(
                    f"  [Epoch {epoch+1:3d}/{MAX_EPOCHS}] "
                    f"train={avg_train:.6f}  val_loss={val_loss:.6f}  "
                    f"val_auroc={val_auroc:.4f}  lr={lr:.6f}  "
                    f"({elapsed:.0f}s)"
                )

            if stopper.should_stop:
                break

        # Restore best checkpoint
        best_auroc = stopper.load_best(model)
        elapsed = time.time() - t_seed_start
        results["seeds"][str(seed)] = {
            "best_val_auroc": best_auroc,
            "best_epoch": stopper.best_epoch + 1,
            "final_epoch": epoch + 1,
            "time_seconds": round(elapsed, 1),
            "train_losses": seed_losses["train"],
            "val_losses": seed_losses["val"],
        }
        print(f"  [DONE] Seed {seed}: best_val_auroc={best_auroc:.6f} "
              f"at epoch {stopper.best_epoch+1} ({elapsed:.0f}s)")

    aurocs = [v["best_val_auroc"] for v in results["seeds"].values() if v["best_val_auroc"] > 0]
    if aurocs:
        results["summary"] = {
            "mean_val_auroc": round(float(np.mean(aurocs)), 6),
            "std_val_auroc": round(float(np.std(aurocs)), 6),
            "n_seeds": len(aurocs),
        }
        print(f"\n[CNN SUMMARY] Val AUROC: {np.mean(aurocs):.6f} ± {np.std(aurocs):.6f}")

    return results

# ─── Section 2: Baseline Training ────────────────────────────────────────────

def train_baselines(
    df: pd.DataFrame, splits: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Train Threshold and Random Forest baselines across horizons and seeds."""
    feature_cols = get_feature_columns(df)
    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]
    results: dict[str, Any] = {}

    train_X = np.nan_to_num(
        train_df[feature_cols].values.astype(np.float32), nan=0.0
    )
    val_X = np.nan_to_num(
        val_df[feature_cols].values.astype(np.float32), nan=0.0
    )

    v_col = feature_cols.index("voltage_v") if "voltage_v" in feature_cols else 0

    for horizon in HORIZONS:
        train_y = train_df[horizon].values
        val_y = val_df[horizon].values

        # ─── Threshold Baseline ──────────────────────────────────
        print(f"\n[BASELINE] Threshold detector — {horizon}")
        best_f1, best_thresh = 0.0, 0.0
        for pct in np.arange(0.85, 1.15, 0.005):
            thresh = 28.0 * pct
            preds = (val_X[:, v_col] < thresh).astype(int)
            f = f1_score(val_y, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh = f, thresh
        results[f"threshold_{horizon}"] = {"f1": best_f1, "threshold": best_thresh}
        print(f"  Threshold={best_thresh:.2f}V, F1={best_f1:.4f}")

        # ─── Random Forest (5 seeds) ────────────────────────────
        print(f"[BASELINE] Random Forest — {horizon}")
        rf_f1s, rf_aurocs = [], []
        for seed in SEEDS:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=25, random_state=seed,
                class_weight="balanced", n_jobs=RF_NJOBS,
                min_samples_leaf=5,
            )
            rf.fit(train_X, train_y)
            pred = rf.predict(val_X)
            proba = rf.predict_proba(val_X)
            # Handle edge case where only one class is seen
            if proba.shape[1] == 2:
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba[:, 0]

            rf_f1s.append(f1_score(val_y, pred, zero_division=0))
            try:
                rf_aurocs.append(roc_auc_score(val_y, proba_pos))
            except ValueError:
                rf_aurocs.append(0.5)

        results[f"rf_{horizon}"] = {
            "f1_mean": round(float(np.mean(rf_f1s)), 4),
            "f1_std": round(float(np.std(rf_f1s)), 4),
            "auroc_mean": round(float(np.mean(rf_aurocs)), 4),
            "auroc_std": round(float(np.std(rf_aurocs)), 4),
            "per_seed_f1": [round(x, 4) for x in rf_f1s],
            "per_seed_auroc": [round(x, 4) for x in rf_aurocs],
        }
        print(f"  RF F1={np.mean(rf_f1s):.4f}±{np.std(rf_f1s):.4f}  "
              f"AUROC={np.mean(rf_aurocs):.4f}±{np.std(rf_aurocs):.4f}")

    return results


# ─── Section 3: PHM Evaluation Metrics ───────────────────────────────────────

def evaluate_all_models(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """Compute full PHM metrics on both test sets."""
    from models.pinn import AVRPINN
    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    results: dict[str, Any] = {}

    train_df = df.iloc[splits["train"]]
    train_X = np.nan_to_num(
        train_df[feature_cols].values.astype(np.float32), nan=0.0
    )

    # Load normalization stats if available
    mean_path = os.path.join(RESULTS_DIR, "feat_mean.npy")
    std_path = os.path.join(RESULTS_DIR, "feat_std.npy")
    if os.path.exists(mean_path):
        feat_mean = np.load(mean_path)
        feat_std = np.load(std_path)
    else:
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0

    test_sets = [
        ("test_held_out_scenario", "held_out"),
        ("test_stress_combo", "stress_combo"),
    ]

    for split_key, result_key in test_sets:
        if split_key not in splits or len(splits[split_key]) == 0:
            continue

        results[result_key] = {}
        test_df = df.iloc[splits[split_key]]
        test_X_raw = np.nan_to_num(
            test_df[feature_cols].values.astype(np.float32), nan=0.0
        )

        for horizon in HORIZONS:
            train_y = train_df[horizon].values
            test_y = test_df[horizon].values

            if test_y.sum() == 0 or test_y.sum() == len(test_y):
                print(f"  [SKIP] {result_key}/{horizon}: no class variation")
                continue

            metrics: dict[str, Any] = {}

            # Bug 27+06 fix: compute test windows ONCE, before all model loops
            test_X_norm = (test_X_raw - feat_mean) / feat_std
            test_wins = torch.from_numpy(
                prepare_windowed_data(test_X_norm)
            ).to(device)
            # Bug 09 fix: use explicit window target indices
            win_indices = get_window_target_indices(len(test_y), SEQ_LEN, STRIDE)
            test_y_wins = test_y[win_indices]

            # Bug 08 fix: align RF evaluation to same windowed population
            test_X_aligned = test_X_raw[win_indices]
            test_y_aligned = test_y_wins

            # ─── RF metrics (5 seeds) ────────────────────────────
            rf_aurocs, rf_auprcs, rf_f1s, rf_recall1 = [], [], [], []
            for seed in SEEDS:
                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=25, random_state=seed,
                    class_weight="balanced", n_jobs=RF_NJOBS,
                    min_samples_leaf=5,
                )
                rf.fit(train_X, train_y)
                pred = rf.predict(test_X_aligned)
                proba = rf.predict_proba(test_X_aligned)
                proba_pos = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

                rf_f1s.append(f1_score(test_y_aligned, pred, zero_division=0))
                try:
                    rf_aurocs.append(roc_auc_score(test_y_aligned, proba_pos))
                except ValueError:
                    rf_aurocs.append(0.5)
                try:
                    rf_auprcs.append(average_precision_score(test_y_aligned, proba_pos))
                except ValueError:
                    rf_auprcs.append(0.0)

                # Recall at 1% FAR
                try:
                    fpr, tpr, _ = roc_curve(test_y_aligned, proba_pos)
                    rf_recall1.append(float(np.interp(0.01, fpr, tpr)))
                except Exception:
                    rf_recall1.append(0.0)

            for key, vals in [
                ("auroc", rf_aurocs), ("auprc", rf_auprcs),
                ("f1", rf_f1s), ("recall_at_1pct_far", rf_recall1),
            ]:
                metrics[f"rf_{key}_mean"] = round(float(np.mean(vals)), 4)
                metrics[f"rf_{key}_std"] = round(float(np.std(vals)), 4)

            # ─── PINN metrics (if checkpoints exist) ─────────────
            pinn_aurocs_all = []
            for seed in SEEDS:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"pinn_seed{seed}_best.pt")
                if not os.path.exists(ckpt_path):
                    continue

                model = AVRPINN(n_input_features=n_features).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                # test_wins already computed above (Bug 27+06 fix)

                if len(test_wins) == 0 or test_y_wins.sum() == 0:
                    continue

                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        ve = min(vs + 256, len(test_wins))
                        out = model(test_wins[vs:ve])
                        # Bug 04 fix: apply sigmoid at inference (model outputs logits)
                        preds_list.append(torch.sigmoid(out[horizon].squeeze(-1)).cpu())
                    pinn_proba = torch.cat(preds_list).numpy()

                try:
                    pinn_aurocs_all.append(roc_auc_score(test_y_wins, pinn_proba))
                except ValueError:
                    pass

            if pinn_aurocs_all:
                metrics["pinn_auroc_mean"] = round(float(np.mean(pinn_aurocs_all)), 4)
                metrics["pinn_auroc_std"] = round(float(np.std(pinn_aurocs_all)), 4)
                
            # ─── LSTM metrics (if checkpoints exist) ─────────────
            from models.baseline_lstm import AVRLSTM
            lstm_aurocs_all = []
            for seed in SEEDS:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"lstm_seed{seed}_best.pt")
                if not os.path.exists(ckpt_path):
                    continue

                model = AVRLSTM(n_input_features=n_features).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        ve = min(vs + 256, len(test_wins))
                        out = model(test_wins[vs:ve])
                        preds_list.append(out[horizon].squeeze(-1).cpu())
                    lstm_proba = torch.cat(preds_list).numpy()

                try:
                    lstm_aurocs_all.append(roc_auc_score(test_y_wins, lstm_proba))
                except ValueError:
                    pass
            
            if lstm_aurocs_all:
                metrics["lstm_auroc_mean"] = round(float(np.mean(lstm_aurocs_all)), 4)
                metrics["lstm_auroc_std"] = round(float(np.std(lstm_aurocs_all)), 4)
                
            # ─── 1D-CNN metrics (if checkpoints exist) ─────────────
            from models.baseline_cnn import AVRCNN
            cnn_aurocs_all = []
            for seed in SEEDS:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"cnn_seed{seed}_best.pt")
                if not os.path.exists(ckpt_path):
                    continue

                model = AVRCNN(n_input_features=n_features).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        ve = min(vs + 256, len(test_wins))
                        out = model(test_wins[vs:ve])
                        preds_list.append(out[horizon].squeeze(-1).cpu())
                    cnn_proba = torch.cat(preds_list).numpy()

                try:
                    cnn_aurocs_all.append(roc_auc_score(test_y_wins, cnn_proba))
                except ValueError:
                    pass
            
            if cnn_aurocs_all:
                metrics["cnn_auroc_mean"] = round(float(np.mean(cnn_aurocs_all)), 4)
                metrics["cnn_auroc_std"] = round(float(np.std(cnn_aurocs_all)), 4)

            results[result_key][horizon] = metrics
            auroc_str = f"AUROC={metrics['rf_auroc_mean']:.4f}±{metrics['rf_auroc_std']:.4f}"
            auprc_str = f"AUPRC={metrics['rf_auprc_mean']:.4f}"
            f1_str = f"F1={metrics['rf_f1_mean']:.4f}"
            
            pinn_str = f"  PINN={metrics['pinn_auroc_mean']:.4f}" if pinn_aurocs_all else ""
            cnn_str = f"  CNN={metrics['cnn_auroc_mean']:.4f}" if cnn_aurocs_all else ""
            lstm_str = f"  LSTM={metrics['lstm_auroc_mean']:.4f}" if lstm_aurocs_all else ""
            
            print(f"  [{result_key}/{horizon}] {auroc_str}  {f1_str}{pinn_str}{cnn_str}{lstm_str}")

    return results


# ─── Section 4: Statistical Significance ──────────────────────────────────────

def compute_significance(
    df: pd.DataFrame, splits: dict[str, np.ndarray], device: torch.device
) -> dict[str, Any]:
    """Wilcoxon signed-rank tests for Tier 1: PINN vs LSTM, LSTM vs RF, RF vs Threshold."""
    from models.pinn import AVRPINN
    from models.baseline_lstm import AVRLSTM
    
    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    results: dict[str, Any] = {}

    split_key = "test_held_out_scenario"
    if split_key not in splits or len(splits[split_key]) == 0:
        return results

    train_df = df.iloc[splits["train"]]
    test_df = df.iloc[splits[split_key]]

    train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0)
    test_X_raw = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0)
    v_col = feature_cols.index("voltage_v") if "voltage_v" in feature_cols else 0

    mean_path = os.path.join(RESULTS_DIR, "feat_mean.npy")
    std_path = os.path.join(RESULTS_DIR, "feat_std.npy")
    if os.path.exists(mean_path):
        feat_mean = np.load(mean_path)
        feat_std = np.load(std_path)
    else:
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0

    test_X_norm = (test_X_raw - feat_mean) / feat_std
    test_wins = torch.from_numpy(prepare_windowed_data(test_X_norm)).to(device)

    # Bug 08+09 fix: compute aligned indices once
    win_indices_sig = get_window_target_indices(len(test_df), SEQ_LEN, STRIDE)
    test_X_aligned_sig = test_X_raw[win_indices_sig]

    for horizon in HORIZONS:
        train_y = train_df[horizon].values
        test_y = test_df[horizon].values
        # Bug 09 fix: use explicit indices
        test_y_wins = test_y[win_indices_sig]
        
        if test_y.sum() == 0 or test_y_wins.sum() == 0:
            continue

        rf_f1s, thresh_f1s, pinn_aurocs, lstm_aurocs, cnn_aurocs = [], [], [], [], []
        rf_aurocs_sig = []
        
        for seed in SEEDS:
            # RF — evaluated on aligned population (Bug 08 fix)
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=25, random_state=seed,
                class_weight="balanced", n_jobs=RF_NJOBS,
            )
            rf.fit(train_X, train_y)
            rf_pred_aligned = rf.predict(test_X_aligned_sig)
            rf_f1s.append(f1_score(test_y_wins, rf_pred_aligned, zero_division=0))
            # RF AUROC for PINN vs RF test (Bug 28)
            rf_proba_sig = rf.predict_proba(test_X_aligned_sig)
            rf_proba_pos_sig = rf_proba_sig[:, 1] if rf_proba_sig.shape[1] == 2 else rf_proba_sig[:, 0]
            try:
                rf_aurocs_sig.append(roc_auc_score(test_y_wins, rf_proba_pos_sig))
            except ValueError:
                rf_aurocs_sig.append(0.5)

            # Threshold — also on aligned population
            thresh_pred = (test_X_aligned_sig[:, v_col] < 26.5).astype(int)
            thresh_f1s.append(f1_score(test_y_wins, thresh_pred, zero_division=0))
            
            # PINN — Bug 04 fix: apply sigmoid
            ckpt_pinn = os.path.join(CHECKPOINT_DIR, f"pinn_seed{seed}_best.pt")
            if os.path.exists(ckpt_pinn):
                model = AVRPINN(n_input_features=n_features).to(device)
                model.load_state_dict(torch.load(ckpt_pinn, map_location=device, weights_only=False)["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        preds_list.append(torch.sigmoid(model(test_wins[vs:min(vs+256, len(test_wins))])[horizon].squeeze(-1)).cpu())
                    try:
                        pinn_aurocs.append(roc_auc_score(test_y_wins, torch.cat(preds_list).numpy()))
                    except ValueError:
                        pass
                        
            # LSTM
            ckpt_lstm = os.path.join(CHECKPOINT_DIR, f"lstm_seed{seed}_best.pt")
            if os.path.exists(ckpt_lstm):
                model = AVRLSTM(n_input_features=n_features).to(device)
                model.load_state_dict(torch.load(ckpt_lstm, map_location=device, weights_only=False)["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        preds_list.append(model(test_wins[vs:min(vs+256, len(test_wins))])[horizon].squeeze(-1).cpu())
                    try:
                        lstm_aurocs.append(roc_auc_score(test_y_wins, torch.cat(preds_list).numpy()))
                    except ValueError:
                        pass
                        
            # 1D-CNN
            from models.baseline_cnn import AVRCNN
            ckpt_cnn = os.path.join(CHECKPOINT_DIR, f"cnn_seed{seed}_best.pt")
            if os.path.exists(ckpt_cnn):
                model = AVRCNN(n_input_features=n_features).to(device)
                model.load_state_dict(torch.load(ckpt_cnn, map_location=device, weights_only=False)["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    preds_list = []
                    for vs in range(0, len(test_wins), 256):
                        preds_list.append(model(test_wins[vs:min(vs+256, len(test_wins))])[horizon].squeeze(-1).cpu())
                    try:
                        cnn_aurocs.append(roc_auc_score(test_y_wins, torch.cat(preds_list).numpy()))
                    except ValueError:
                        pass

        results[horizon] = {}
        
        # Test 1A: PINN vs 1D-CNN (Tier-1 Core claim: Physics matters)
        if len(pinn_aurocs) > 1 and len(cnn_aurocs) > 1 and len(pinn_aurocs) == len(cnn_aurocs):
            diffs = [p - c for p, c in zip(pinn_aurocs, cnn_aurocs)]
            if len(set(diffs)) > 1:
                try:
                    stat, p = stats.wilcoxon(pinn_aurocs, cnn_aurocs, alternative="greater")
                    verdict = "✓ PINN>CNN (SIG)" if p < 0.05 else "✗ not sig"
                    print(f"  [{horizon}] PINN vs CNN (AUROC)  p={p:.6f} → {verdict}")
                    results[horizon]["pinn_vs_cnn"] = {"stat": float(stat), "p": float(p), "sig": p < 0.05}
                except Exception:
                    pass
        
        # Test 1B: PINN vs LSTM (RNN baseline check)
        if len(pinn_aurocs) > 1 and len(lstm_aurocs) > 1 and len(pinn_aurocs) == len(lstm_aurocs):
            diffs = [p - l for p, l in zip(pinn_aurocs, lstm_aurocs)]
            if len(set(diffs)) > 1:
                try:
                    stat, p = stats.wilcoxon(pinn_aurocs, lstm_aurocs, alternative="greater")
                    verdict = "✓ PINN>LSTM (SIG)" if p < 0.05 else "✗ not sig"
                    print(f"  [{horizon}] PINN vs LSTM (AUROC) p={p:.6f} → {verdict}")
                    results[horizon]["pinn_vs_lstm"] = {"stat": float(stat), "p": float(p), "sig": p < 0.05}
                except Exception:
                    pass

        # Test 2: RF vs Threshold (Original claim)
        diffs = [r - t for r, t in zip(rf_f1s, thresh_f1s)]
        if len(set(diffs)) > 1:
            try:
                stat, p = stats.wilcoxon(rf_f1s, thresh_f1s, alternative="greater")
                verdict = "✓ RF>Thresh (SIG)" if p < 0.05 else "✗ not sig"
                print(f"  [{horizon}] RF vs Thresh (F1)    p={p:.6f} → {verdict}")
                results[horizon]["rf_vs_thresh"] = {"stat": float(stat), "p": float(p), "sig": p < 0.05}
            except Exception:
                pass

        # Bug 28 fix: Test 1C: PINN vs RF (AUROC) — the paper's PRIMARY claim
        if len(pinn_aurocs) >= 2 and len(rf_aurocs_sig) >= 2:
            min_len = min(len(pinn_aurocs), len(rf_aurocs_sig))
            diffs_pinn_rf = [p - r for p, r in zip(pinn_aurocs[:min_len], rf_aurocs_sig[:min_len])]
            if len(set(diffs_pinn_rf)) > 1 and all(d != 0 for d in diffs_pinn_rf):
                try:
                    stat, p = stats.wilcoxon(pinn_aurocs[:min_len], rf_aurocs_sig[:min_len],
                                             alternative="greater")
                    verdict = "✓ PINN>RF (p<0.05)" if p < 0.05 else f"✗ not sig (p={p:.4f})"
                    print(f"  [{horizon}] PINN vs RF (AUROC): {verdict}")
                    results[horizon]["pinn_vs_rf"] = {
                        "pinn_mean": round(float(np.mean(pinn_aurocs)), 4),
                        "rf_mean": round(float(np.mean(rf_aurocs_sig)), 4),
                        "stat": float(stat), "p": float(p), "sig": bool(p < 0.05),
                    }
                except Exception as e:
                    print(f"  [{horizon}] PINN vs RF: Wilcoxon failed ({e})")
            else:
                # Bug 14 fix: bootstrap CI fallback (more robust at n=5)
                diffs_arr = np.array(diffs_pinn_rf)
                rng_boot = np.random.default_rng(42)
                boot_means = [
                    np.mean(rng_boot.choice(diffs_arr, len(diffs_arr), replace=True))
                    for _ in range(5000)
                ]
                ci_low = float(np.percentile(boot_means, 2.5))
                ci_high = float(np.percentile(boot_means, 97.5))
                sig = ci_low > 0.0
                verdict = "✓ PINN>RF (CI above 0)" if sig else f"✗ CI includes 0 [{ci_low:.4f}, {ci_high:.4f}]"
                print(f"  [{horizon}] PINN vs RF bootstrap: {verdict}")
                results[horizon]["pinn_vs_rf"] = {
                    "mean_diff": float(np.mean(diffs_arr)),
                    "ci_low": ci_low, "ci_high": ci_high, "sig": sig,
                }

    return results


# ─── Section 5: SHAP Explainability ──────────────────────────────────────────

def run_shap_analysis(
    df: pd.DataFrame, splits: dict[str, np.ndarray]
) -> dict[str, Any]:
    """SHAP TreeExplainer on Random Forest for feature importance ranking."""
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed. Install via: pip install shap")
        # Fallback: use RF feature_importances_ (Gini importance)
        return _run_gini_importance(df, splits)

    feature_cols = get_feature_columns(df)
    train_df = df.iloc[splits["train"]]
    test_df = df.iloc[splits["test_held_out_scenario"]]

    train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0)
    test_X = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0)

    results: dict[str, Any] = {}
    horizon = "fault_10s"
    train_y = train_df[horizon].values

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, random_state=42,
        class_weight="balanced", n_jobs=RF_NJOBS,
    )
    rf.fit(train_X, train_y)

    n_explain = min(500, len(test_X))
    print(f"[SHAP] Computing TreeExplainer on {n_explain} samples...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(test_X[:n_explain])

    # Handle binary classification output format
    if isinstance(shap_values, list) and len(shap_values) == 2:
        importance = np.abs(shap_values[1]).mean(axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            importance = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
    else:
        importance = np.abs(np.array(shap_values)).mean(axis=0)

    # Build ranking
    ranked_idx = np.argsort(-importance)  # Descending
    top_features = [
        (feature_cols[int(i)], round(float(importance[int(i)]), 6))
        for i in ranked_idx[:20]
    ]

    physics_feats = {
        "dv_dt", "di_dt", "dp_dt", "power_instantaneous_w",
        "voltage_ripple_amplitude_v", "thermal_stress_index",
        "load_impedance_ohm", "voltage_deviation_v", "voltage_within_spec",
    }

    results[horizon] = {
        "top_20_features": top_features,
        "physics_in_top_5": [f for f, _ in top_features[:5] if f in physics_feats],
        "physics_in_top_10": [f for f, _ in top_features[:10] if f in physics_feats],
    }

    print(f"\n[SHAP] Top 15 features for {horizon}:")
    print(f"  {'Rank':>4}  {'Feature':<42} {'Mean |SHAP|':>12}")
    print(f"  {'─'*4}  {'─'*42} {'─'*12}")
    for i, (feat, imp) in enumerate(top_features[:15]):
        tag = " ◆ PHYSICS" if feat in physics_feats else ""
        print(f"  {i+1:4d}  {feat:<42} {imp:12.6f}{tag}")

    n_phys_top5 = len(results[horizon]["physics_in_top_5"])
    n_phys_top10 = len(results[horizon]["physics_in_top_10"])
    print(f"\n  Physics features in top-5:  {n_phys_top5}")
    print(f"  Physics features in top-10: {n_phys_top10}")

    return results


def _run_gini_importance(
    df: pd.DataFrame, splits: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Fallback when SHAP is not installed: use Gini importance."""
    print("[FALLBACK] Using Gini importance (install shap for full analysis)")
    feature_cols = get_feature_columns(df)
    train_df = df.iloc[splits["train"]]
    train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0)

    results: dict[str, Any] = {}
    horizon = "fault_10s"
    train_y = train_df[horizon].values

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, random_state=42,
        class_weight="balanced", n_jobs=RF_NJOBS,
    )
    rf.fit(train_X, train_y)
    importance = rf.feature_importances_
    ranked_idx = np.argsort(-importance)

    physics_feats = {
        "dv_dt", "di_dt", "dp_dt", "power_instantaneous_w",
        "voltage_ripple_amplitude_v", "thermal_stress_index",
        "load_impedance_ohm", "voltage_deviation_v",
    }

    top_features = [
        (feature_cols[int(i)], round(float(importance[int(i)]), 6))
        for i in ranked_idx[:20]
    ]

    results[horizon] = {
        "method": "gini_importance",
        "top_20_features": top_features,
        "physics_in_top_5": [f for f, _ in top_features[:5] if f in physics_feats],
        "physics_in_top_10": [f for f, _ in top_features[:10] if f in physics_feats],
    }

    print(f"\n[GINI] Top 15 features for {horizon}:")
    for i, (feat, imp) in enumerate(top_features[:15]):
        tag = " ◆ PHYSICS" if feat in physics_feats else ""
        print(f"  {i+1:4d}  {feat:<42} {imp:12.6f}{tag}")

    return results


# ─── Section 6: Publication Figures ───────────────────────────────────────────

def generate_figures(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    all_results: dict[str, Any],
) -> None:
    """Generate 300 DPI IEEE-format figures (PNG + PDF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        sns.set_palette("deep")
    except ImportError:
        sns = None

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    os.makedirs(FIGURES_DIR, exist_ok=True)
    feature_cols = get_feature_columns(df)

    def savefig(fig: Any, name: str) -> None:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {name}")

    # ─── Fig 2: MIL-STD Waveform Samples ─────────────────────────────
    print("[FIG] Generating waveform samples...")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))
    scenarios = ["baseline", "arctic_cold", "desert_heat", "artillery_firing"]
    colors = ["#2196F3", "#00BCD4", "#FF5722", "#FF9800"]
    for idx, (ax, sc) in enumerate(zip(axes.flat, scenarios)):
        csv_path = os.path.join(DATA_DIR, f"avr_data_{sc}_run1.csv")
        if os.path.exists(csv_path):
            sdf = pd.read_csv(csv_path)
            t = sdf["timestamp"].values[:3000]
            v = sdf["voltage_v"].values[:3000]
            ax.plot(t, v, linewidth=0.5, color=colors[idx])
            ax.set_title(sc.replace("_", " ").title(), fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.axhline(y=28.0, color="gray", linestyle="--", alpha=0.5, label="Nominal (28V)")
            ax.axhspan(23.5, 32.5, alpha=0.04, color="green")
            ax.set_ylim(20, 36)
    plt.tight_layout()
    savefig(fig, "fig2_milstd_waveforms")

    # ─── Fig 3: PINN Training Curves ─────────────────────────────────
    pinn_data = all_results.get("pinn", {}).get("seeds", {})
    if pinn_data:
        print("[FIG] Generating PINN training curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for seed_str, sd in pinn_data.items():
            if sd.get("train_losses"):
                ax1.plot(sd["train_losses"], label=f"Seed {seed_str}", alpha=0.7)
                ax2.plot(sd["val_losses"], label=f"Seed {seed_str}", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss (BCE)")
        ax1.set_title("PINN Training Loss")
        ax1.legend()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss (BCE)")
        ax2.set_title("PINN Validation Loss")
        ax2.legend()
        plt.tight_layout()
        savefig(fig, "fig3_pinn_training_curves")

    # ─── Fig 4: ROC Curves ───────────────────────────────────────────
    print("[FIG] Generating ROC curves...")
    split_key = "test_held_out_scenario"
    if split_key in splits and len(splits[split_key]) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        test_df = df.iloc[splits[split_key]]
        train_df = df.iloc[splits["train"]]
        test_X = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0)
        train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0)
        test_y = test_df["fault_10s"].values
        train_y = train_df["fault_10s"].values

        if test_y.sum() > 0:
            # ─── Bug 31 fix: PINN ROC curve ──────────────────────
            pinn_plotted = False
            ckpt_path = os.path.join(CHECKPOINT_DIR, "pinn_seed42_best.pt")
            if os.path.exists(ckpt_path):
                try:
                    from models.pinn import AVRPINN
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    n_features = len(feature_cols)
                    model = AVRPINN(n_input_features=n_features).to(device)
                    ckpt = torch.load(
                        ckpt_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()

                    # Normalize test data using training stats
                    feat_mean = train_X.mean(axis=0)
                    feat_std = train_X.std(axis=0) + 1e-8
                    test_X_norm = (test_X - feat_mean) / feat_std
                    test_wins = torch.from_numpy(
                        prepare_windowed_data(test_X_norm)
                    ).to(device)
                    win_indices = get_window_target_indices(
                        len(test_y), SEQ_LEN, STRIDE
                    )
                    test_y_wins = test_y[win_indices]

                    if len(test_wins) > 0 and test_y_wins.sum() > 0:
                        with torch.no_grad():
                            preds = []
                            for vs in range(0, len(test_wins), 256):
                                ve = min(vs + 256, len(test_wins))
                                out = model(test_wins[vs:ve])
                                preds.append(
                                    torch.sigmoid(out["fault_10s"].squeeze(-1)).cpu()
                                )
                            pinn_proba = torch.cat(preds).numpy()
                        fpr_p, tpr_p, _ = roc_curve(test_y_wins, pinn_proba)
                        auroc_p = roc_auc_score(test_y_wins, pinn_proba)
                        ax.plot(
                            fpr_p, tpr_p, color="#4CAF50", linewidth=2.5,
                            label=f"PINN (AUC={auroc_p:.3f})",
                        )
                        pinn_plotted = True
                except Exception as e:
                    print(f"  [WARN] Could not plot PINN ROC: {e}")

            # ─── RF ROC ──────────────────────────────────────────
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=25, random_state=42,
                class_weight="balanced", n_jobs=RF_NJOBS,
            )
            rf.fit(train_X, train_y)
            rf_proba = rf.predict_proba(test_X)[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(test_y, rf_proba)
            auroc_rf = roc_auc_score(test_y, rf_proba)
            ax.plot(fpr_rf, tpr_rf, color="#2196F3", linewidth=2,
                    label=f"Random Forest (AUC={auroc_rf:.3f})")

            v_idx = feature_cols.index("voltage_v")
            fpr_th, tpr_th, _ = roc_curve(test_y, -test_X[:, v_idx])
            auroc_th = roc_auc_score(test_y, -test_X[:, v_idx])
            ax.plot(fpr_th, tpr_th, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"Threshold (AUC={auroc_th:.3f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves — Fault Prediction (τ=10s)")
            ax.legend(loc="lower right")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        plt.tight_layout()
        savefig(fig, "fig4_roc_curves")

    # ─── Fig 5: Precision-Recall Curves ──────────────────────────────
    if split_key in splits and len(splits[split_key]) > 0 and test_y.sum() > 0:
        from sklearn.metrics import precision_recall_curve as prc
        print("[FIG] Generating Precision-Recall curves...")
        fig, ax = plt.subplots(figsize=(6, 5))
        for h, color in zip(HORIZONS, ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]):
            test_yh = test_df[h].values
            train_yh = train_df[h].values
            if test_yh.sum() == 0:
                continue
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=25, random_state=42,
                class_weight="balanced", n_jobs=RF_NJOBS,
            )
            rf.fit(train_X, train_yh)
            proba_h = rf.predict_proba(test_X)[:, 1]
            prec, rec, _ = prc(test_yh, proba_h)
            ap = average_precision_score(test_yh, proba_h)
            label_h = h.replace("fault_", "τ=").replace("s", "s")
            ax.plot(rec, prec, color=color, linewidth=2, label=f"{label_h} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves by Horizon")
        ax.legend(loc="upper right")
        plt.tight_layout()
        savefig(fig, "fig5_precision_recall")

    # ─── Fig 6: SHAP / Feature Importance ────────────────────────────
    shap_data = all_results.get("shap", {}).get("fault_10s", {})
    if shap_data and "top_20_features" in shap_data:
        print("[FIG] Generating feature importance bar chart...")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        top = shap_data["top_20_features"][:15]
        names = [t[0] for t in top][::-1]
        values = [t[1] for t in top][::-1]

        physics_feats = {
            "dv_dt", "di_dt", "dp_dt", "power_instantaneous_w",
            "voltage_ripple_amplitude_v", "thermal_stress_index",
            "load_impedance_ohm", "voltage_deviation_v",
        }
        colors = ["#2196F3" if n in physics_feats else "#90A4AE" for n in names]

        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        method = shap_data.get("method", "shap")
        xlabel = "Mean |SHAP value|" if method != "gini_importance" else "Gini Importance"
        ax.set_xlabel(xlabel)
        ax.set_title(f"Feature Importance — Fault Prediction (τ=10s)")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#2196F3", label="Physics-derived"),
            Patch(facecolor="#90A4AE", label="Statistical/Rolling"),
        ], loc="lower right")
        plt.tight_layout()
        savefig(fig, "fig6_feature_importance")

    # ─── Fig 7: Fault Distribution by Scenario ──────────────────────
    if "scenario" in df.columns:
        print("[FIG] Generating fault distribution by scenario...")
        fig, ax = plt.subplots(figsize=(8.5, 4))
        scenarios_list = sorted(df["scenario"].unique())
        fault_rates = []
        for sc in scenarios_list:
            rate = df[df["scenario"] == sc]["fault_10s"].mean() * 100
            fault_rates.append(rate)
        labels = [s.replace("_", "\n") for s in scenarios_list]
        palette = ["#1976D2", "#00838F", "#C62828", "#E65100",
                    "#2E7D32", "#6A1B9A", "#455A64"][:len(labels)]
        bars = ax.bar(labels, fault_rates, color=palette)
        ax.set_ylabel("Fault Rate at τ=10s (%)")
        ax.set_title("Fault Incidence by Operational Scenario")
        for bar, rate in zip(bars, fault_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        savefig(fig, "fig7_fault_distribution")

    # ─── Fig 8: Multi-Horizon AUROC Comparison ───────────────────────
    eval_data = all_results.get("evaluation", {}).get("held_out", {})
    if eval_data:
        print("[FIG] Generating multi-horizon AUROC comparison...")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        h_labels, rf_means, rf_stds = [], [], []
        pinn_means, pinn_stds = [], []

        for h in HORIZONS:
            if h not in eval_data:
                continue
            h_labels.append(h.replace("fault_", "τ=").replace("s", "s"))
            rf_means.append(eval_data[h]["rf_auroc_mean"])
            rf_stds.append(eval_data[h]["rf_auroc_std"])
            pm = eval_data[h].get("pinn_auroc_mean", 0)
            ps = eval_data[h].get("pinn_auroc_std", 0)
            pinn_means.append(pm)
            pinn_stds.append(ps)

        x = np.arange(len(h_labels))
        width = 0.35

        ax.bar(x - width/2, rf_means, width, yerr=rf_stds, capsize=5,
               label="Random Forest", color="#2196F3", alpha=0.85)
        if any(p > 0 for p in pinn_means):
            ax.bar(x + width/2, pinn_means, width, yerr=pinn_stds, capsize=5,
                   label="PINN", color="#FF9800", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(h_labels)
        ax.set_ylabel("AUROC")
        ax.set_title("Multi-Horizon Fault Prediction Performance")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random baseline")
        ax.legend()
        plt.tight_layout()
        savefig(fig, "fig8_multi_horizon_auroc")

    print(f"\n[FIGURES] All figures saved to {FIGURES_DIR}/")


# ─── Main Orchestrator ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVR-PHM Publication Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Run everything (default)")
    parser.add_argument("--train", action="store_true", help="Train PINN + baselines")
    parser.add_argument("--evaluate", action="store_true", help="PHM metrics + significance")
    parser.add_argument("--shap", action="store_true", help="SHAP explainability")
    parser.add_argument("--figures", action="store_true", help="Publication figures")
    parser.add_argument("--force", action="store_true", help="Proceed even if VVA gate fails")
    args = parser.parse_args()

    run_all = args.all or not any([args.train, args.evaluate, args.shap, args.figures])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = pick_device()
    t_start = time.time()

    print("\n" + "=" * 70)
    print("  AVR-PHM TIER-1 PUBLICATION EXPERIMENT RUNNER")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────
    df, splits = load_featured_data()
    print(f"\n[DATA] {len(df):,} samples × {len(df.columns)} columns")
    for name, idx in splits.items():
        print(f"  {name}: {len(idx):,} samples ({100*len(idx)/len(df):.1f}%)")

    all_results: dict[str, Any] = {"hardware": {
        "device": str(device),
        "seeds": SEEDS,
        "max_epochs": MAX_EPOCHS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }}

    # ─── Bug 07 fix: Run VVA validation before training ─────────────────
    if run_all or args.train:
        try:
            from data_gen.vva import run_full_vva
            from data_gen.cgan import CGANGenerator, generate_augmentation_data

            cgan_ckpt_path = os.path.join(CHECKPOINT_DIR, "cgan_latest.pt")
            if os.path.exists(cgan_ckpt_path):
                print("\n[VVA] Running synthetic data quality gate...")
                # Build real windows from training split
                train_idx = splits.get("train", np.array([]))
                vva_cols = ["voltage_v", "current_a", "temperature_c"]
                if len(train_idx) > 0 and all(c in df.columns for c in vva_cols):
                    train_data = df.iloc[train_idx][vva_cols].values.astype(np.float32)
                    n_vva = min(500, (len(train_data) - SEQ_LEN) // STRIDE)
                    if n_vva > 50:
                        real_seqs = np.array(
                            [train_data[i * STRIDE: i * STRIDE + SEQ_LEN] for i in range(n_vva)]
                        )
                        # Load cGAN and generate synthetic sequences
                        ckpt = torch.load(
                            cgan_ckpt_path, map_location="cpu", weights_only=False
                        )
                        gen = CGANGenerator(
                            latent_dim=32, condition_dim=ckpt["generator_state_dict"][
                                "cond_proj.weight"
                            ].shape[1],
                        )
                        gen.load_state_dict(ckpt["generator_state_dict"])
                        # Generate with zeros condition (baseline healthy)
                        cond_dim = gen.cond_proj.weight.shape[1]
                        synth_seqs = generate_augmentation_data(
                            gen, n_samples=n_vva,
                            condition=np.zeros(cond_dim, dtype=np.float32),
                        )
                        vva_report = run_full_vva(real_seqs, synth_seqs)
                        acceptance = vva_report.get("acceptance", {})
                        mmd_ok = acceptance.get("mmd_pass", False)
                        prop_ok = acceptance.get("propensity_pass", False)
                        acf_ok = acceptance.get("acf_pass", False)
                        all_pass = mmd_ok and prop_ok and acf_ok
                        print(f"[VVA] MMD pass={mmd_ok}, Propensity pass={prop_ok}, "
                              f"ACF pass={acf_ok} → {'ACCEPTED' if all_pass else 'REJECTED'}")
                        if not all_pass and not args.force:
                            raise RuntimeError(
                                "[VVA] Synthetic data quality gate FAILED. "
                                "Re-train cGAN or use --force to override."
                            )
                        elif not all_pass:
                            print("[VVA] WARNING: Quality gate failed but --force is set, proceeding.")
            else:
                print("[VVA] No cGAN checkpoint found — skipping synthetic data validation.")
        except ImportError as e:
            print(f"[VVA] VVA module not available ({e}), skipping validation gate.")

    # ─── Step 1: Train ────────────────────────────────────────────
    if run_all or args.train:
        print(f"\n{'═'*70}")
        print("  STEP 1A/6: MULTI-SEED PINN TRAINING")
        print(f"{'═'*70}")
        t1 = time.time()
        all_results["pinn"] = train_pinn_multiseed(df, splits, device)
        print(f"  [TIME] {time.time()-t1:.0f}s")

        print(f"\n{'═'*70}")
        print("  STEP 1B/6: MULTI-SEED LSTM TRAINING (BASELINE)")
        print(f"{'═'*70}")
        t1b = time.time()
        all_results["lstm"] = train_lstm_multiseed(df, splits, device)
        print(f"  [TIME] {time.time()-t1b:.0f}s")
        
        print(f"\n{'═'*70}")
        print("  STEP 1C/6: MULTI-SEED 1D-CNN TRAINING (BASELINE)")
        print(f"{'═'*70}")
        t1c = time.time()
        all_results["cnn"] = train_cnn_multiseed(df, splits, device)
        print(f"  [TIME] {time.time()-t1c:.0f}s")

        print(f"\n{'═'*70}")
        print("  STEP 2/6: BASELINE ML TRAINING (RF)")
        print(f"{'═'*70}")
        t2 = time.time()
        all_results["baselines"] = train_baselines(df, splits)
        print(f"  [TIME] {time.time()-t2:.0f}s")

    # ─── Step 2: Evaluate ─────────────────────────────────────────
    if run_all or args.evaluate:
        print(f"\n{'═'*70}")
        print("  STEP 3/6: PHM EVALUATION METRICS")
        print(f"{'═'*70}")
        t3 = time.time()
        all_results["evaluation"] = evaluate_all_models(df, splits, device)
        print(f"  [TIME] {time.time()-t3:.0f}s")

        print(f"\n{'═'*70}")
        print("  STEP 4/6: STATISTICAL SIGNIFICANCE")
        print(f"{'═'*70}")
        t4 = time.time()
        all_results["significance"] = compute_significance(df, splits, device)
        print(f"  [TIME] {time.time()-t4:.0f}s")

    # ─── Step 3: SHAP ─────────────────────────────────────────────
    if run_all or args.shap:
        print(f"\n{'═'*70}")
        print("  STEP 5/6: SHAP EXPLAINABILITY")
        print(f"{'═'*70}")
        t5 = time.time()
        all_results["shap"] = run_shap_analysis(df, splits)
        print(f"  [TIME] {time.time()-t5:.0f}s")

    # ─── Step 4: Figures ──────────────────────────────────────────
    if run_all or args.figures:
        print(f"\n{'═'*70}")
        print("  STEP 6/6: PUBLICATION FIGURES")
        print(f"{'═'*70}")
        t6 = time.time()
        generate_figures(df, splits, all_results)
        print(f"  [TIME] {time.time()-t6:.0f}s")

    # ─── Save Results ─────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "publication_results.json")
    all_results["total_time_seconds"] = round(time.time() - t_start, 1)

    def np_convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} not serializable")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=np_convert)

    total = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  ✓ COMPLETE — Total time: {total/60:.1f} min ({total:.0f}s)")
    print(f"  Results:  {results_path}")
    print(f"  Figures:  {FIGURES_DIR}/")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
