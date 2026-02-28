"""
Standard LSTM architecture for AVR fault prognostics.

This serves as the pure deep learning baseline (no physical constraints)
to demonstrate the added value of the Physics-Informed Neural Network (PINN).
It predicts faults across the same 4 horizons (1s, 5s, 10s, 30s) but uses
a standard Recurrent Neural Network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVRLSTM(nn.Module):
    """
    Multi-task LSTM for AVR fault prognostics.

    Input: sliding window (batch, window_size=100, n_input_features)

    Architecture:
        Temporal encoder: 2-layer LSTM (hidden_size=64)
        Shared: Dense(128) + Dropout(0.15) + LayerNorm + GELU
        4 task heads (fault_1s, fault_5s, fault_10s, fault_30s)
    """

    def __init__(
        self,
        n_input_features: int = 10,
        window_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # ─── Temporal Encoder (LSTM) ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # ─── Shared Representation ───────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # ─── Task Heads ─────────────────────────────────────────────────
        # Heads 1-4: Binary fault warning at {1s, 5s, 10s, 30s}
        self.head_fault_1s = self._make_binary_head(dropout_rate)
        self.head_fault_5s = self._make_binary_head(dropout_rate)
        self.head_fault_10s = self._make_binary_head(dropout_rate)
        self.head_fault_30s = self._make_binary_head(dropout_rate)


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
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output from the last time step
        last_step = lstm_out[:, -1, :]

        # Shared representation
        shared = self.shared(last_step)

        # Task outputs
        outputs: dict[str, torch.Tensor] = {
            "fault_1s": torch.sigmoid(self.head_fault_1s(shared)),
            "fault_5s": torch.sigmoid(self.head_fault_5s(shared)),
            "fault_10s": torch.sigmoid(self.head_fault_10s(shared)),
            "fault_30s": torch.sigmoid(self.head_fault_30s(shared)),
        }

        return outputs

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_forward_passes: int = 50,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        MC Dropout inference for uncertainty quantification.
        """
        self.train()  # Keep dropout active

        all_outputs: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for _ in range(n_forward_passes):
                outputs: dict[str, torch.Tensor] = self.forward(x)
                for key, val in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    all_outputs[key].append(val)

        results: dict[str, dict[str, torch.Tensor]] = {}
        for key, vals in all_outputs.items():
            stacked: torch.Tensor = torch.stack(vals, dim=0)
            results[key] = {
                "mean": stacked.mean(dim=0),
                "std": stacked.std(dim=0),
            }

        self.eval()
        return results

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = logits.squeeze(-1)
    
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1.0 - probs)
    focal_weight = (1.0 - pt) ** gamma

    if alpha is not None:
        alpha_t = torch.where(targets == 1, alpha[1], alpha[0])
        focal_weight = focal_weight * alpha_t

    loss = (focal_weight * bce).mean()
    return loss

def compute_lstm_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    fault_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute the standard data-driven loss for the LSTM baseline.
    Identical to the PINN loss but WITHOUT the physics residual component.
    """
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

def run_tests():
    model = AVRLSTM(n_input_features=10, window_size=100)
    x = torch.randn(4, 100, 10)
    out = model(x)
    assert out["fault_10s"].shape == (4, 1)

    n_params: int = sum(p.numel() for p in model.parameters())
    print(f"[PASS] models/baseline_lstm.py — params: {n_params:,}")

if __name__ == "__main__":
    run_tests()
