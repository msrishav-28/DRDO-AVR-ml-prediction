"""
run_digitaltwin_experiment.py — Standalone Execution for Feature-Space PINN
Year: 2025 SOTA Concept

This script trains the DigitalTwinPINN from `tier_1_concepts.py` without 
disturbing the main `run_publication.py` pipeline. It demonstrates predicting
an unobservable physical state first, then conditioning fault predictions on it.

Hardware target: CPU/GPU
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import get_device, HORIZONS, SEQ_LEN, STRIDE
from features.engineer import get_feature_columns
from experiments.evaluate import prepare_windowed_data
from models.tier_1_concepts import DigitalTwinPINN

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/featured")

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
    train_X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
    
    feat_mean = train_X.mean(axis=0)
    feat_std = train_X.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    
    train_X = (train_X - feat_mean) / feat_std
    train_windows = torch.from_numpy(prepare_windowed_data(train_X))
    
    # We only take the 10s horizon for this simplified concept demonstration
    train_targets = torch.from_numpy(
        train_df["fault_10s"].values[SEQ_LEN - 1::STRIDE][:len(train_windows)].astype(np.float32)
    ).to(device)
    
    return train_windows, train_targets, n_features

def train_digital_twin():
    device = get_device()
    print(f"Loading data for Digital Twin/Feature-Space PINN on {device}...")
    train_x, train_y, n_features = load_data(device)
    
    # We collapse the time dimension to simulate standard features for the concept network
    train_x = train_x.mean(dim=1).to(device) 
    
    model = DigitalTwinPINN(input_dim=n_features, latent_physics_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    batch_size = 512
    max_epochs = 30
    
    print("\n[TRAINING START]")
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        epoch_fault_loss = 0.0
        epoch_phys_loss = 0.0
        
        for start in range(0, len(train_x), batch_size):
            end = min(start + batch_size, len(train_x))
            idx = perm[start:end]
            
            batch_x = train_x[idx]
            batch_y = train_y[idx].unsqueeze(-1)
            
            optimizer.zero_grad()
            
            # Forward pass provides prognostics AND the unobservable physics state
            fault_logits, hidden_physics_state = model(batch_x)
            
            # prognostics loss
            l_fault = F.binary_cross_entropy_with_logits(fault_logits, batch_y)
            
            # internal physics constraint (mocked in the concept class)
            l_phys = model.physics_constraint_loss(hidden_physics_state, None)
            
            total_loss = l_fault + (0.5 * l_phys)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_fault_loss += l_fault.item()
            epoch_phys_loss += l_phys.item()
            
        print(f"[Epoch {epoch+1:02d}/{max_epochs}] Fault Loss: {epoch_fault_loss/len(train_x):.4f} | "
              f"Est. Latent Mean: {hidden_physics_state.mean().item():.3f}")

if __name__ == "__main__":
    train_digital_twin()
