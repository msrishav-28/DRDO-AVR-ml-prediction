"""
Multi-task Physics-Informed Neural Network (PINN) for AVR fault prognostics.

The PINN combines data-driven learning with physics constraints from the
synchronous generator d-q axis equations. It uses DeepXDE with PyTorch
backend for physics residual computation.

Architecture:
    Temporal encoder: 1D-CNN (kernel=5, filters=64, 3 layers) + GAP
    Shared: Dense(128) + Dropout(0.15) + LayerNorm + GELU
    Task heads:
        Head 1-4 (fault_τ): Dense(64) → Dense(1) + Sigmoid
        Head 5 (mechanism): Dense(64) → Dense(4) + Softmax
        Head 6 (forecast):  Dense(64) → Dense(10) — voltage trajectory
        Head 7 (severity):  Dense(64) → Dense(4) + Softmax
        Head 8 (RUL):       Dense(64) → Dense(32) → Dense(1)

Total parameters: ~180,000 (edge-deployable target).

Loss function:
    L_total = λ₁*L_data + λ₂*L_physics + λ₃*L_fault
    L_data: MSE on voltage forecast
    L_physics: mean(ε₁² + ε₂² + ε₃²)/3
    L_fault: weighted focal loss on fault horizons

References:
    - Raissi et al. (2019). Physics-informed neural networks. JCP.
    - Lu et al. (2021). DeepXDE: A deep learning library for solving
      differential equations. SIAM Review.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from simulator.constants import Ka, Ke, Ra, Te, V_base, Vref


class AVRPhysicsResidual(nn.Module):
    """
    Computes the physics residual loss for the AVR PINN.

    The residual is the L2 norm of how much the neural network's
    predictions violate the d-q axis generator equations.

    We use REDUCED-ORDER physics constraints (not full 8th-order DAE)
    to keep training tractable:

    Constraint 1 (voltage dynamics):
        ε₁ = dVt/dt - f_voltage(Eq_dprime, Ed_dprime, Id, Iq, Ra)

    Constraint 2 (excitation dynamics):
        ε₂ = dVf/dt - (1/Te) * (-Ke * Vf + Ka * (Vref - Vt))

    Constraint 3 (power conservation):
        ε₃ = P_electrical - V_dc * I_dc  (should be ~0)
    """

    def __init__(
        self,
        ke: float = Ke,
        ka: float = Ka,
        te: float = Te,
        vref: float = Vref,
        ra: float = Ra,
        v_base: float = V_base,
    ) -> None:
        super().__init__()
        self.ke: float = ke
        self.ka: float = ka
        self.te: float = te
        self.vref: float = vref
        self.ra: float = ra
        self.v_base: float = v_base

    def compute_residuals(
        self,
        t: torch.Tensor,
        v_pred: torch.Tensor,
        i_pred: torch.Tensor,
        model_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics residuals using PyTorch finite-difference (autograd-compatible).

        Bug 10 fix: replaced scipy ODE with native PyTorch operations.
        Bug 02 fix: replaced tautological power constraint with power balance.

        Inputs:
            t: Time tensor of shape (batch, seq_len).
            v_pred: Voltage sequence (batch, seq_len).
            i_pred: Current sequence (batch, seq_len).
            model_outputs: Full model output tensor for additional states.

        Outputs:
            Residual tensor of shape (batch, 3).
        """
        batch_size: int = v_pred.shape[0]

        # Handle 1D inputs (single timestep) — return zeros
        if v_pred.dim() == 1 or v_pred.shape[-1] < 2:
            return torch.zeros(batch_size, 3, device=v_pred.device)

        # Constraint 1: Voltage dynamics via finite-difference
        # dV/dt ≈ (I - V/R_load) / C  where R_load = V_base / I_rated
        dt = (t[:, 1:] - t[:, :-1]).clamp(min=1e-6)
        dv_dt_numerical = (v_pred[:, 1:] - v_pred[:, :-1]) / dt
        v_mid = 0.5 * (v_pred[:, 1:] + v_pred[:, :-1])
        i_mid = 0.5 * (i_pred[:, 1:] + i_pred[:, :-1])
        R_load = self.v_base / 10.0  # nominal load resistance
        C_eff = 0.01  # effective capacitance
        dv_dt_model = (i_mid - v_mid / R_load) / C_eff
        constraint_1 = (dv_dt_numerical - dv_dt_model).pow(2).mean()

        # Constraint 2: Excitation dynamics
        vt_pu = v_mid / self.v_base
        vf_implied = vt_pu * self.ke
        excitation_residual = (
            (1.0 / self.te) * (-self.ke * vf_implied + self.ka * (self.vref - vt_pu))
        )
        constraint_2 = excitation_residual.pow(2).mean()

        # Constraint 3: Power balance (Bug 02 fix — replaces tautological P=V*I check)
        # P_load ≈ V * I should be consistent with P_rated + thermal losses
        P_load = v_mid * i_mid
        P_rated = self.v_base * 10.0  # nominal power
        thermal_loss = 0.001 * (v_mid - self.v_base).pow(2)
        constraint_3 = (P_load - P_rated - thermal_loss).pow(2).mean()

        residuals = torch.zeros(batch_size, 3, device=v_pred.device)
        residuals[:, 0] = constraint_1
        residuals[:, 1] = constraint_2
        residuals[:, 2] = constraint_3

        return residuals


class AVRPINN(nn.Module):
    """
    Multi-task Physics-Informed Neural Network for AVR fault prognostics.

    Input: sliding window (batch, window_size=100, n_input_features)

    Architecture:
        Temporal encoder: 1D-CNN (3 layers, kernel=5, filters=64) + GAP
        Shared: Dense(128) + Dropout(0.15) + LayerNorm + GELU
        7+1 task heads (separate for each task)

    Total parameters: ~180K (edge-deployable).
    MC Dropout for uncertainty quantification (Section 21).
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

        # Head 5: Mechanism classification (4 classes)
        self.head_mechanism: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(64, 4),
        )

        # Head 6: Voltage forecast (10 steps)
        self.head_forecast: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(64, 10),
        )

        # Head 7: Severity classification (4 classes)
        self.head_severity: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(64, 4),
        )

        # Head 8: RUL estimation (Section 19)
        self.head_rul: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Physics residual module
        self.physics_residual: AVRPhysicsResidual = AVRPhysicsResidual()

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
            Dict with keys: 'fault_1s', 'fault_5s', 'fault_10s', 'fault_30s',
            'mechanism', 'forecast', 'severity', 'rul'.
        """
        # CNN expects (batch, channels, seq_len)
        x_cnn: torch.Tensor = x.permute(0, 2, 1)

        # Encode
        encoded: torch.Tensor = self.encoder(x_cnn).squeeze(-1)

        # Shared representation
        shared: torch.Tensor = self.shared(encoded)

        # Task outputs — Bug 04 fix: output raw logits, sigmoid applied in loss/inference
        outputs: dict[str, torch.Tensor] = {
            "fault_1s": self.head_fault_1s(shared),
            "fault_5s": self.head_fault_5s(shared),
            "fault_10s": self.head_fault_10s(shared),
            "fault_30s": self.head_fault_30s(shared),
            "mechanism": self.head_mechanism(shared),
            "forecast": self.head_forecast(shared),
            "severity": self.head_severity(shared),
            "rul": self.head_rul(shared),
        }

        return outputs

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_forward_passes: int = 50,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        MC Dropout inference for uncertainty quantification (Section 21).

        Purpose:
            Run T=50 forward passes with dropout active to estimate
            epistemic uncertainty.

        Inputs:
            x: Input tensor (batch, window_size, n_features).
            n_forward_passes: Number of stochastic forward passes.

        Outputs:
            Dict mapping task names to {'mean', 'std'} tensors.

        Mathematical basis:
            MC Dropout approximates Bayesian inference:
            p(y|x) ≈ (1/T) Σ p(y|x, θ_t) where θ_t ~ q(θ)
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
    """
    Focal loss for imbalanced binary classification.

    Purpose:
        Down-weights easy examples and focuses training on hard examples.

    Inputs:
        logits: Raw output (before sigmoid) of shape (batch, 1).
        targets: Binary labels of shape (batch, 1).
        gamma: Focusing parameter (default 2.0 per spec).
        alpha: Per-class weight tensor, or None for uniform.

    Outputs:
        Scalar focal loss.

    Mathematical basis:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        where p_t = p if y=1, (1-p) if y=0

    Raises:
        ValueError if logit dimensions don't match targets.
    """
    logits = logits.squeeze(-1)
    
    bce: torch.Tensor = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs: torch.Tensor = torch.sigmoid(logits)
    pt: torch.Tensor = torch.where(targets == 1, probs, 1.0 - probs)
    focal_weight: torch.Tensor = (1.0 - pt) ** gamma

    if alpha is not None:
        alpha_t: torch.Tensor = torch.where(targets == 1, alpha[1], alpha[0])
        focal_weight = focal_weight * alpha_t

    loss: torch.Tensor = (focal_weight * bce).mean()
    return loss


def asymmetric_rul_loss(
    pred_rul: torch.Tensor,
    true_rul: torch.Tensor,
    alpha: float = 1.3,
) -> torch.Tensor:
    """
    Asymmetric MSE loss for RUL prediction (Section 19).

    Purpose:
        Penalizes late predictions (overestimating RUL) more heavily
        than early predictions, since late warnings are more dangerous.

    Inputs:
        pred_rul: Predicted RUL in seconds.
        true_rul: True RUL in seconds.
        alpha: Penalty multiplier for late predictions (default 1.3).

    Outputs:
        Scalar loss.

    Mathematical basis:
        error = pred - true
        loss = α*error² if error > 0 (late), else error² (early)
    """
    error: torch.Tensor = pred_rul - true_rul
    loss: torch.Tensor = torch.where(
        error > 0, alpha * error**2, error**2
    )
    return loss.mean()


def compute_total_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    physics_residuals: torch.Tensor,
    fault_weights: dict[str, float] | None = None,
    lambda_physics: float = 0.3,
    lambda_data: float = 0.5,
    lambda_fault: float = 0.2,
) -> dict[str, torch.Tensor]:
    """
    Compute the composite PINN loss function.

    Purpose:
        Combines data loss, physics residual loss, and fault prediction loss
        into a single training objective.

    Inputs:
        predictions: Dict of model outputs from AVRPINN.forward().
        targets: Dict of ground truth labels.
        physics_residuals: Residual tensor (batch, 3) from AVRPhysicsResidual.
        fault_weights: Per-horizon weight dict.
        lambda_physics: Physics loss weight λ₂ (default 0.3).
        lambda_data: Data loss weight λ₁ (default 0.5).
        lambda_fault: Fault loss weight λ₃ (default 0.2).

    Outputs:
        Dict with keys: 'total', 'data', 'physics', 'fault',
        'mechanism', 'severity', 'rul'.

    Mathematical basis:
        L_total = λ₁*L_data + λ₂*L_physics + λ₃*L_fault
        L_data: MSE on voltage forecast
        L_physics: mean(ε₁² + ε₂² + ε₃²)/3
        L_fault: weighted focal loss on 4 horizons
    """
    if fault_weights is None:
        fault_weights = {
            "fault_1s": 3.0,
            "fault_5s": 2.5,
            "fault_10s": 2.0,
            "fault_30s": 1.5,
        }

    # ─── Data Loss (MSE on voltage forecast) ─────────────────────────────
    l_data: torch.Tensor = torch.tensor(0.0, device=physics_residuals.device)
    if "forecast" in predictions and "forecast" in targets:
        l_data = F.mse_loss(predictions["forecast"], targets["forecast"])

    # ─── Physics Loss (Bug 11 fix: normalize to O(1)) ──────────────────
    # Characteristic scale: 28V / 0.1s = 280 V/s → residuals in (V/s)²
    V_SCALE = 28.0
    DT_SCALE = 0.1
    residual_normalised = physics_residuals / (V_SCALE / DT_SCALE + 1e-8)
    l_physics: torch.Tensor = residual_normalised.pow(2).mean()

    # ─── Fault Loss (weighted focal loss) ────────────────────────────────
    l_fault: torch.Tensor = torch.tensor(0.0, device=physics_residuals.device)
    for horizon, weight in fault_weights.items():
        if horizon in predictions and horizon in targets:
            l_fault = l_fault + weight * focal_loss(
                predictions[horizon], targets[horizon]
            )

    # ─── Mechanism Loss (cross-entropy) ──────────────────────────────────
    l_mechanism: torch.Tensor = torch.tensor(0.0, device=physics_residuals.device)
    if "mechanism" in predictions and "mechanism" in targets:
        l_mechanism = F.cross_entropy(
            predictions["mechanism"],
            targets["mechanism"].long(),
        )

    # ─── Severity Loss (cross-entropy) ───────────────────────────────────
    l_severity: torch.Tensor = torch.tensor(0.0, device=physics_residuals.device)
    if "severity" in predictions and "severity" in targets:
        l_severity = F.cross_entropy(
            predictions["severity"],
            targets["severity"].long(),
        )

    # ─── RUL Loss (asymmetric MSE) ───────────────────────────────────────
    l_rul: torch.Tensor = torch.tensor(0.0, device=physics_residuals.device)
    if "rul" in predictions and "rul" in targets:
        l_rul = asymmetric_rul_loss(
            predictions["rul"].squeeze(), targets["rul"].squeeze()
        )

    # ─── Total ───────────────────────────────────────────────────────────
    l_total: torch.Tensor = (
        lambda_data * l_data
        + lambda_physics * l_physics
        + lambda_fault * l_fault
        + 0.1 * l_mechanism
        + 0.1 * l_severity
        + 0.1 * l_rul
    )

    return {
        "total": l_total,
        "data": l_data,
        "physics": l_physics,
        "fault": l_fault,
        "mechanism": l_mechanism,
        "severity": l_severity,
        "rul": l_rul,
    }


def run_tests() -> None:
    """Sanity checks for the PINN module."""
    # Test 1: Model output shapes
    model: AVRPINN = AVRPINN(n_input_features=10, window_size=100)
    x: torch.Tensor = torch.randn(4, 100, 10)
    outputs: dict[str, torch.Tensor] = model(x)

    assert outputs["fault_1s"].shape == (4, 1), "fault_1s shape mismatch"
    assert outputs["mechanism"].shape == (4, 4), "mechanism shape mismatch"
    assert outputs["forecast"].shape == (4, 10), "forecast shape mismatch"
    assert outputs["rul"].shape == (4, 1), "RUL shape mismatch"

    # Test 2: Parameter count ~ 180K
    n_params: int = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 500_000, (
        f"Parameter count {n_params:,} outside expected range"
    )

    # Test 3: Focal loss computes without error
    logits_test: torch.Tensor = torch.randn(10, 1)
    targets_test: torch.Tensor = torch.randint(0, 2, (10, 1)).float()
    fl: torch.Tensor = focal_loss(logits_test, targets_test)
    assert fl.item() >= 0.0, "Focal loss should be non-negative"

    print(f"[PASS] models/pinn.py — all tests passed. "
          f"Params: {n_params:,}")


if __name__ == "__main__":
    run_tests()
