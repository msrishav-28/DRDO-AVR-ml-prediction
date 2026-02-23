"""
Baseline 2: Bidirectional GRU Recurrent Autoencoder for anomaly detection.

Architecture:
    Encoder: BiGRU [128, 64] → latent (32)
    Decoder: GRU [64, 128] → Dense(n_features)

Trained ONLY on healthy (baseline) data. Anomaly score = reconstruction error.
Threshold = mean + 2.5*std of healthy reconstruction error.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


class RecurrentAutoencoder(nn.Module):
    """Bidirectional GRU autoencoder for unsupervised anomaly detection."""

    def __init__(
        self,
        n_features: int = 3,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if encoder_hidden is None:
            encoder_hidden = [128, 64]
        if decoder_hidden is None:
            decoder_hidden = [64, 128]

        self.latent_dim: int = latent_dim

        # Encoder: BiGRU layers
        self.encoder_gru1: nn.GRU = nn.GRU(
            input_size=n_features,
            hidden_size=encoder_hidden[0],
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.encoder_gru2: nn.GRU = nn.GRU(
            input_size=encoder_hidden[0] * 2,
            hidden_size=encoder_hidden[1],
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.encoder_fc: nn.Linear = nn.Linear(
            encoder_hidden[1] * 2, latent_dim
        )

        # Decoder: GRU layers (unidirectional)
        self.decoder_fc: nn.Linear = nn.Linear(
            latent_dim, decoder_hidden[0]
        )
        self.decoder_gru1: nn.GRU = nn.GRU(
            input_size=decoder_hidden[0],
            hidden_size=decoder_hidden[0],
            batch_first=True,
            dropout=dropout,
        )
        self.decoder_gru2: nn.GRU = nn.GRU(
            input_size=decoder_hidden[0],
            hidden_size=decoder_hidden[1],
            batch_first=True,
            dropout=dropout,
        )
        self.decoder_output: nn.Linear = nn.Linear(
            decoder_hidden[1], n_features
        )

        # Threshold (set during fit on healthy data)
        self.threshold: float = 0.0
        self.mean_error: float = 0.0
        self.std_error: float = 0.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent vector."""
        out1, _ = self.encoder_gru1(x)
        out2, _ = self.encoder_gru2(out1)
        # Use last timestep
        latent: torch.Tensor = self.encoder_fc(out2[:, -1, :])
        return latent

    def decode(
        self, z: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Decode latent vector back to sequence."""
        h: torch.Tensor = torch.relu(self.decoder_fc(z))
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        out1, _ = self.decoder_gru1(h)
        out2, _ = self.decoder_gru2(out1)
        output: torch.Tensor = self.decoder_output(out2)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z: torch.Tensor = self.encode(x)
        reconstructed: torch.Tensor = self.decode(z, x.shape[1])
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error."""
        with torch.no_grad():
            x_hat: torch.Tensor = self.forward(x)
            error: torch.Tensor = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return error

    def set_threshold(
        self,
        healthy_data: torch.Tensor,
        n_sigma: float = 2.5,
    ) -> float:
        """
        Set anomaly detection threshold from healthy data errors.

        Mathematical basis:
            threshold = mean(error_healthy) + n_sigma * std(error_healthy)
        """
        errors: torch.Tensor = self.compute_anomaly_score(healthy_data)
        self.mean_error = float(errors.mean())
        self.std_error = float(errors.std())
        self.threshold = self.mean_error + n_sigma * self.std_error
        return self.threshold

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict anomalies: 1 if anomaly, 0 if normal."""
        scores: torch.Tensor = self.compute_anomaly_score(x)
        return (scores > self.threshold).cpu().numpy().astype(np.int32)


def run_tests() -> None:
    """Sanity checks for recurrent autoencoder."""
    model: RecurrentAutoencoder = RecurrentAutoencoder(n_features=3)
    x: torch.Tensor = torch.randn(4, 100, 3)
    out: torch.Tensor = model(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    scores: torch.Tensor = model.compute_anomaly_score(x)
    assert scores.shape == (4,), f"Score shape should be (4,), got {scores.shape}"

    print("[PASS] models/recurrent_ae.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
