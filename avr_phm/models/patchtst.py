"""
Baseline 3: PatchTST for time-series forecasting.

Wrapper around a simplified PatchTST implementation for voltage forecasting.
PatchTST segments the input time-series into patches and processes them
with a Transformer encoder.

Architecture:
    Input: (batch, seq_len=100, n_features)
    Patch embedding: patch_size=16, stride=8
    Transformer: 4 heads, 2 layers, d_model=64
    Output: (batch, forecast_horizon=10)

Reference:
    Nie et al. (2023). A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers. ICLR 2023.
"""

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


class PatchEmbedding(nn.Module):
    """Segment time-series into patches and embed."""

    def __init__(
        self,
        n_features: int = 3,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.stride: int = stride
        self.projection: nn.Linear = nn.Linear(
            n_features * patch_size, d_model
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed patches.

        Inputs:
            x: (batch, seq_len, n_features)

        Outputs:
            patches: (batch, n_patches, d_model)
        """
        batch_size, seq_len, n_feat = x.shape

        # Unfold into patches
        patches: list[torch.Tensor] = []
        for start in range(0, seq_len - self.patch_size + 1, self.stride):
            patch: torch.Tensor = x[:, start:start + self.patch_size, :]
            patches.append(patch.reshape(batch_size, -1))

        if not patches:
            # Fallback: use entire sequence as single patch
            patches = [x.reshape(batch_size, -1)[:, :self.patch_size * n_feat]]

        stacked: torch.Tensor = torch.stack(patches, dim=1)
        embedded: torch.Tensor = self.projection(stacked)
        embedded = self.layer_norm(embedded)
        return embedded


class PatchTST(nn.Module):
    """
    PatchTST model for time-series forecasting.

    Segments input into patches, processes with Transformer encoder,
    and projects to forecast horizon.
    """

    def __init__(
        self,
        n_features: int = 3,
        seq_len: int = 100,
        forecast_horizon: int = 10,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.forecast_horizon: int = forecast_horizon

        # Patch embedding
        self.patch_embed: PatchEmbedding = PatchEmbedding(
            n_features=n_features,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
        )

        # Compute number of patches
        n_patches: int = max(1, (seq_len - patch_size) // stride + 1)

        # Positional encoding
        self.pos_encoding: nn.Parameter = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer: nn.TransformerEncoderLayer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Flatten and project to forecast
        self.flatten_head: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_horizon),
        )

        # Also support fault classification head
        self.fault_head: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Inputs:
            x: (batch, seq_len, n_features)

        Outputs:
            Dict with 'forecast' and 'fault_prob' tensors.
        """
        # Embed patches
        patches: torch.Tensor = self.patch_embed(x)

        # Add positional encoding
        n_p: int = patches.shape[1]
        pos_enc: torch.Tensor = self.pos_encoding[:, :n_p, :]
        patches = patches + pos_enc

        # Transformer encoding
        encoded: torch.Tensor = self.transformer(patches)

        # Forecast
        forecast: torch.Tensor = self.flatten_head(encoded)
        fault_logit: torch.Tensor = self.fault_head(encoded)

        return {
            "forecast": forecast,
            "fault_prob": torch.sigmoid(fault_logit),
        }


def run_tests() -> None:
    """Sanity checks for PatchTST."""
    model: PatchTST = PatchTST(
        n_features=3, seq_len=100, forecast_horizon=10,
        patch_size=16, stride=8
    )
    x: torch.Tensor = torch.randn(4, 100, 3)
    out: dict[str, torch.Tensor] = model(x)
    assert out["forecast"].shape == (4, 10), (
        f"Forecast shape should be (4, 10), got {out['forecast'].shape}"
    )
    assert out["fault_prob"].shape == (4, 1), (
        f"Fault prob shape should be (4, 1), got {out['fault_prob'].shape}"
    )

    n_params: int = sum(p.numel() for p in model.parameters())
    print(f"[PASS] models/patchtst.py — all tests passed. Params: {n_params:,}")


if __name__ == "__main__":
    run_tests()
