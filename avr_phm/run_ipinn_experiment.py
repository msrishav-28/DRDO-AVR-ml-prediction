"""
run_ipinn_experiment.py — Standalone Execution for I-PINN (Adaptive Weighting)
Year: 2025 SOTA Concept

This script trains the AdaptiveWeightedPINN (I-PINN) from `tier_1_concepts.py`
without disturbing the main `run_publication.py` pipeline. It demonstrates
homoscedastic uncertainty weighting across the physics, data, and fault losses.

Hardware target: GPU (if available)
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from config import get_device

# Hardcoded config for independence
HORIZONS = ["fault_1s", "fault_5s", "fault_10s", "fault_30s"]
SEQ_LEN = 100
STRIDE = 10  # Match run_publication.py; STRIDE=1 causes OOM on 4GB GPU
from models.pinn import AVRPINN, compute_total_loss, AVRPhysicsResidual
from models.tier_1_concepts import AdaptiveWeightedPINN

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/featured")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs/results")

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_prefixes = [
        "fault_1s", "fault_5s", "fault_10s", "fault_30s",
        "fault_mechanism", "severity", "rul_seconds",
        "voltage_next_", "timestamp", "run_id", "scenario",
    ]
    return [
        col for col in df.columns
        if not any(col.startswith(p) for p in exclude_prefixes)
    ]

def prepare_windowed_data(X: np.ndarray, seq_len: int = SEQ_LEN, stride: int = STRIDE) -> np.ndarray:
    n_windows = max(0, (len(X) - seq_len) // stride)
    n_features = X.shape[1]
    windows = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i] = X[start : start + seq_len]
    return windows

def load_data(device: torch.device):
    """Load and normalize data, holding out a validation set."""
    featured_path = os.path.join(DATA_DIR, "all_featured.csv")
    if not os.path.exists(featured_path):
        raise FileNotFoundError(f"Featured data not found at {featured_path}")
        
    df = pd.read_csv(featured_path)
    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(0.8 * len(df))
    
    train_df = df.iloc[indices[:split_idx]]
    val_df = df.iloc[indices[split_idx:]]
    
    train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
    val_X = np.nan_to_num(val_df[feature_cols].values.astype(np.float32))
    
    feat_mean = train_X.mean(axis=0)
    feat_std = train_X.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    
    train_X = (train_X - feat_mean) / feat_std
    val_X = (val_X - feat_mean) / feat_std
    
    train_windows = torch.from_numpy(prepare_windowed_data(train_X))
    val_windows = torch.from_numpy(prepare_windowed_data(val_X))
    
    train_targets = {
        h: torch.from_numpy(train_df[h].values[SEQ_LEN - 1::STRIDE][:len(train_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }
    val_targets = {
        h: torch.from_numpy(val_df[h].values[SEQ_LEN - 1::STRIDE][:len(val_windows)].astype(np.float32)).to(device)
        for h in HORIZONS
    }
    
    return train_windows, train_targets, val_windows, val_targets, n_features

def train_ipinn():
    device = get_device()
    print(f"Loading data for I-PINN experiment on {device}...")
    train_x, train_y, val_x, val_y, n_features = load_data(device)
    
    print(f"[I-PINN] {len(train_x)} train / {len(val_x)} val windows")
    
    # Instantiate the base PINN and wrap it in the Adaptive I-PINN
    base_pinn = AVRPINN(n_input_features=n_features, dropout_rate=0.15).to(device)
    model = AdaptiveWeightedPINN(base_encoder=base_pinn, num_tasks=3).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = 512
    max_epochs = 30
    
    print("\n[TRAINING START]")
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        epoch_loss = 0.0
        
        for start in range(0, len(train_x), batch_size):
            end = min(start + batch_size, len(train_x))
            idx = perm[start:end]
            
            batch_x = train_x[idx].to(device)
            batch_y = {h: t[idx] for h, t in train_y.items()}
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            # Bug 34 fix: use model forecast output (has grad) instead of input features
            residual_calc = AVRPhysicsResidual()
            b_size, seq, _ = batch_x.shape
            t_batch = torch.linspace(0, (seq - 1) * 0.1, seq).unsqueeze(0).expand(b_size, -1).to(device)
            
            forecast = out.get("forecast", None)
            if forecast is not None and forecast.requires_grad:
                # Use forecast output for physics residual (has gradient flow)
                v_forecast = forecast[:, :seq] if forecast.shape[-1] >= seq else batch_x[:, :, 0]
                i_forecast = batch_x[:, :, 1]  # current from input (no forecast head for current)
                residuals = residual_calc.compute_residuals(t_batch, v_forecast, i_forecast, forecast)
            else:
                # Fallback: skip physics loss if forecast head not suitable
                residuals = torch.zeros(b_size, 3, device=device)
            
            # Extract the raw scalar losses
            loss_dict = compute_total_loss(out, batch_y, residuals, fault_weights={"fault_1s": 1.0, "fault_5s": 1.0, "fault_10s": 1.0, "fault_30s": 1.0})
            
            # Apply dynamic adaptive weighting
            total_loss, weights = model.compute_adaptive_loss(
                loss_dict.get("data", torch.tensor(0.0, device=device)),
                loss_dict.get("physics", torch.tensor(0.0, device=device)),
                loss_dict.get("fault", torch.tensor(0.0, device=device))
            )
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        print(f"[Epoch {epoch+1:02d}/{max_epochs}] Loss: {epoch_loss/len(train_x):.4f} | "
              f"W_Data: {weights['weight_data']:.3f}, W_Phys: {weights['weight_physics']:.3f}, W_Fault: {weights['weight_fault']:.3f}")

if __name__ == "__main__":
    train_ipinn()
