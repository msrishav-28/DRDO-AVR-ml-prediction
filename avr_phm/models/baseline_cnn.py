"""
Standard 1D-CNN architecture for AVR fault prognostics.

This serves as a direct ABLATION baseline for the PINN. It uses the exact same
1D-CNN temporal encoder, shared representation, and classification heads as the
PINN, but removes the physics loss and specialized physics heads.
This isolates the contribution of the physical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVRCNN(nn.Module):
    """
    Multi-task 1D-CNN for AVR fault prognostics.

    Input: sliding window (batch, window_size=100, n_input_features)

    Architecture (Identical to PINN's data-driven components):
        Temporal encoder: 1D-CNN (kernel=5, filters=64, 3 layers) + GAP
        Shared: Dense(128) + Dropout(0.15) + LayerNorm + GELU
        4 task heads (fault_1s, fault_5s, fault_10s, fault_30s)
    """

    def __init__(
        self,
        n_input_features: int = 10,
        window_size: int = 100,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()

        self.window_size: int = window_size
        self.dropout_rate: float = dropout_rate

        # ─── Temporal Encoder (1D-CNN) ───────────────────────────────────
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Conv1d(n_input_features, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
        )

        # ─── Shared Representation ───────────────────────────────────────
        self.shared: nn.Sequential = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # ─── Task Heads ─────────────────────────────────────────────────
        # Heads 1-4: Binary fault warning at {1s, 5s, 10s, 30s}
        self.head_fault_1s: nn.Sequential = self._make_binary_head(dropout_rate)
        self.head_fault_5s: nn.Sequential = self._make_binary_head(dropout_rate)
        self.head_fault_10s: nn.Sequential = self._make_binary_head(dropout_rate)
        self.head_fault_30s: nn.Sequential = self._make_binary_head(dropout_rate)

    @staticmethod
    def _make_binary_head(dropout_rate: float) -> nn.Sequential:
        """Create a binary classification task head."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.

        Inputs:
            x: Input tensor of shape (batch, window_size, n_features).

        Outputs:
            Dict with keys: 'fault_1s', 'fault_5s', 'fault_10s', 'fault_30s'
        """
        # CNN expects (batch, channels, seq_len)
        x_cnn: torch.Tensor = x.permute(0, 2, 1)

        # Encode
        encoded: torch.Tensor = self.encoder(x_cnn).squeeze(-1)

        # Shared representation
        shared: torch.Tensor = self.shared(encoded)

        # Task outputs
        outputs: dict[str, torch.Tensor] = {
            "fault_1s": torch.sigmoid(self.head_fault_1s(shared)),
            "fault_5s": torch.sigmoid(self.head_fault_5s(shared)),
            "fault_10s": torch.sigmoid(self.head_fault_10s(shared)),
            "fault_30s": torch.sigmoid(self.head_fault_30s(shared)),
        }

        return outputs


def compute_cnn_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    fault_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute the standard data-driven loss for the CNN baseline.
    Identical to the PINN loss but WITHOUT the physics residual component.
    """
    from models.baseline_lstm import focal_loss  # Reuse existing FL logic
    if fault_weights is None:
        fault_weights = {
            "fault_1s": 3.0,
            "fault_5s": 2.5,
            "fault_10s": 2.0,
            "fault_30s": 1.5,
        }

    # ─── Fault Loss (weighted focal loss) ────────────────────────────────
    l_fault: torch.Tensor = torch.tensor(0.0, device=next(iter(predictions.values())).device)
    for horizon, weight in fault_weights.items():
        if horizon in predictions and horizon in targets:
            l_fault = l_fault + weight * focal_loss(
                predictions[horizon], targets[horizon]
            )

    return {
        "total": l_fault,
        "fault": l_fault,
    }


def run_tests() -> None:
    model = AVRCNN(n_input_features=10, window_size=100)
    x = torch.randn(4, 100, 10)
    out = model(x)
    assert out["fault_10s"].shape == (4, 1)

    n_params: int = sum(p.numel() for p in model.parameters())
    print(f"[PASS] models/baseline_cnn.py — params: {n_params:,}")

if __name__ == "__main__":
    run_tests()
