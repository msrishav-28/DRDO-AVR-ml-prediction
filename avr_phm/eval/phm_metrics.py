"""
PHM-grade evaluation metrics.

Implements the complete evaluation framework from master plan Section 10:
    - Multi-horizon classification metrics (per horizon τ)
    - Voltage forecast regression metrics
    - Mechanism classification metrics
    - Lead time distribution analysis
    - RUL metrics (NASA scoring function, Section 19)
    - Computational complexity metrics (Section 24)

All metrics return dicts suitable for direct logging to wandb and
inclusion in publication tables.
"""

import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    horizon_name: str = "",
) -> dict[str, float]:
    """
    Compute full PHM classification metrics for a single horizon.

    Purpose:
        Calculates all metrics required for the publication comparison table.

    Inputs:
        y_true: Ground truth binary labels (n_samples,).
        y_pred: Predicted binary labels (n_samples,).
        y_proba: Predicted probabilities (n_samples,), optional.
        horizon_name: String identifier for this horizon (e.g., '1s').

    Outputs:
        Dict with all metric values, prefixed by horizon_name.

    Metrics computed:
        Precision, Recall, F1 (macro), Accuracy, AUC-ROC, AUC-PR,
        Specificity (TNR), False Alarm Rate (FAR), Missed Detection Rate (MDR).
    """
    prefix: str = f"{horizon_name}_" if horizon_name else ""
    metrics: dict[str, float] = {}

    metrics[f"{prefix}accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )
    metrics[f"{prefix}precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    metrics[f"{prefix}recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    metrics[f"{prefix}f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics[f"{prefix}f1_binary"] = float(
        f1_score(y_true, y_pred, average="binary", zero_division=0)
    )

    # Confusion matrix derived metrics
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn: int = int(cm[0, 0])
    fp: int = int(cm[0, 1])
    fn: int = int(cm[1, 0])
    tp: int = int(cm[1, 1])

    metrics[f"{prefix}specificity"] = float(
        tn / max(tn + fp, 1)
    )
    metrics[f"{prefix}false_alarm_rate"] = float(
        fp / max(fp + tn, 1)
    )
    metrics[f"{prefix}missed_detection_rate"] = float(
        fn / max(fn + tp, 1)
    )

    if y_proba is not None:
        try:
            metrics[f"{prefix}auroc"] = float(
                roc_auc_score(y_true, y_proba)
            )
        except ValueError:
            metrics[f"{prefix}auroc"] = 0.0

        try:
            metrics[f"{prefix}auprc"] = float(
                average_precision_score(y_true, y_proba)
            )
        except ValueError:
            metrics[f"{prefix}auprc"] = 0.0

    return metrics


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute voltage forecast regression metrics.

    Inputs:
        y_true: True voltage trajectory (n_samples, forecast_horizon).
        y_pred: Predicted voltage trajectory (n_samples, forecast_horizon).

    Outputs:
        Dict with MAE, RMSE, R² per step and overall.
    """
    metrics: dict[str, float] = {}

    # Overall metrics
    error: np.ndarray = y_true - y_pred
    metrics["forecast_mae"] = float(np.mean(np.abs(error)))
    metrics["forecast_rmse"] = float(np.sqrt(np.mean(error**2)))

    ss_res: float = float(np.sum(error**2))
    ss_tot: float = float(np.sum((y_true - y_true.mean()) ** 2))
    metrics["forecast_r2"] = float(
        1.0 - ss_res / max(ss_tot, 1e-10)
    )

    # Per-step metrics
    n_steps: int = y_true.shape[1] if y_true.ndim > 1 else 1
    if y_true.ndim > 1:
        for step in range(n_steps):
            step_error: np.ndarray = y_true[:, step] - y_pred[:, step]
            metrics[f"forecast_mae_step{step+1}"] = float(
                np.mean(np.abs(step_error))
            )
            metrics[f"forecast_rmse_step{step+1}"] = float(
                np.sqrt(np.mean(step_error**2))
            )

    return metrics


def compute_mechanism_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute fault mechanism classification metrics.

    Inputs:
        y_true: True mechanism labels (n_samples,) — 0=none, 1=thyristor,
                2=capacitor, 3=terminal.
        y_pred: Predicted mechanism labels (n_samples,).

    Outputs:
        Dict with per-class F1 and macro-averaged metrics.
    """
    metrics: dict[str, float] = {}

    metrics["mechanism_accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )
    metrics["mechanism_f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Per-class F1
    mechanism_names: list[str] = ["none", "thyristor", "capacitor", "terminal"]
    per_class_f1: np.ndarray = f1_score(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0
    )
    for i, name in enumerate(mechanism_names):
        if i < len(per_class_f1):
            metrics[f"mechanism_f1_{name}"] = float(per_class_f1[i])

    return metrics


def compute_lead_time_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, float]:
    """
    Analyze the lead time distribution of fault warnings.

    Purpose:
        Measures how early the model warns before actual faults occur.

    Inputs:
        y_true: True fault labels (n_samples,).
        y_pred: Predicted fault labels (n_samples,).
        timestamps: Time values for each sample (n_samples,).

    Outputs:
        Dict with lead time statistics (mean, median, std, min, max).
    """
    metrics: dict[str, float] = {}

    # Find fault onset times (first y_true=1 in each fault episode)
    fault_starts: list[int] = []
    in_fault: bool = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_fault:
            fault_starts.append(i)
            in_fault = True
        elif y_true[i] == 0:
            in_fault = False

    # For each fault, find first warning
    lead_times: list[float] = []
    for fault_idx in fault_starts:
        # Look backward for first prediction=1
        for j in range(fault_idx, -1, -1):
            if y_pred[j] == 0:
                if j + 1 <= fault_idx:
                    lead_time: float = max(0.0, float(
                        timestamps[fault_idx] - timestamps[j + 1]
                    ))
                    lead_times.append(lead_time)
                break

    if lead_times:
        lt_arr: np.ndarray = np.array(lead_times)
        metrics["lead_time_mean_s"] = float(np.mean(lt_arr))
        metrics["lead_time_median_s"] = float(np.median(lt_arr))
        metrics["lead_time_std_s"] = float(np.std(lt_arr))
        metrics["lead_time_min_s"] = float(np.min(lt_arr))
        metrics["lead_time_max_s"] = float(np.max(lt_arr))
        metrics["n_detected_faults"] = len(lead_times)
    else:
        metrics["lead_time_mean_s"] = 0.0
        metrics["lead_time_median_s"] = 0.0
        metrics["n_detected_faults"] = 0

    return metrics


def compute_rul_metrics(
    true_rul: np.ndarray,
    pred_rul: np.ndarray,
) -> dict[str, float]:
    """
    Compute RUL estimation metrics per Section 19.

    Inputs:
        true_rul: True Remaining Useful Life (n_samples,).
        pred_rul: Predicted RUL (n_samples,).

    Outputs:
        Dict with MAE, RMSE, NASA Scoring Function, α-λ accuracy.

    Mathematical basis:
        NASA scoring function:
            S = Σ exp(-d/13) - 1   if d < 0 (early prediction)
            S = Σ exp(d/10) - 1    if d ≥ 0 (late prediction)
            where d = pred - true

        α-λ accuracy:
            Percentage of predictions within α% of true RUL at evaluation
            time λ (default α=0.2, λ=0.5).
    """
    metrics: dict[str, float] = {}
    error: np.ndarray = pred_rul - true_rul

    metrics["rul_mae"] = float(np.mean(np.abs(error)))
    metrics["rul_rmse"] = float(np.sqrt(np.mean(error**2)))

    # NASA Scoring Function
    scores: np.ndarray = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )
    metrics["rul_nasa_score"] = float(np.sum(scores))

    # α-λ accuracy (α=0.2)
    alpha: float = 0.2
    within_alpha: np.ndarray = np.abs(error) <= alpha * np.abs(true_rul)
    metrics["rul_alpha_accuracy"] = float(np.mean(within_alpha))

    return metrics


def compute_computational_complexity(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """
    Measure computational complexity metrics per Section 24.

    Inputs:
        model: PyTorch model.
        input_shape: Input tensor shape (batch, seq_len, n_features).
        device: Device to measure on.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Outputs:
        Dict with n_parameters, latency_ms, throughput_samples_per_sec,
        memory_mb.
    """
    metrics: dict[str, float] = {}

    # Parameter count
    n_params: int = sum(p.numel() for p in model.parameters())
    metrics["n_parameters"] = float(n_params)
    metrics["n_parameters_trainable"] = float(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    # Latency measurement
    model.to(device)
    model.eval()
    x: torch.Tensor = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    start: float = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed: float = time.perf_counter() - start
    avg_latency_s: float = elapsed / n_runs

    metrics["latency_ms"] = avg_latency_s * 1000.0
    metrics["throughput_samples_per_sec"] = float(
        input_shape[0] / avg_latency_s
    )

    # Memory footprint (PyTorch model state)
    param_bytes: int = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    buffer_bytes: int = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    metrics["memory_mb"] = (param_bytes + buffer_bytes) / (1024 * 1024)

    return metrics


def run_tests() -> None:
    """Sanity checks for PHM metrics."""
    rng: np.random.Generator = np.random.default_rng(42)

    # Test 1: Perfect classification
    y_true: np.ndarray = np.array([0, 0, 1, 1, 0, 1])
    y_pred: np.ndarray = np.array([0, 0, 1, 1, 0, 1])
    m: dict[str, float] = compute_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0, "Perfect predictions should give accuracy=1.0"
    assert m["f1_binary"] == 1.0, "Perfect predictions should give F1=1.0"

    # Test 2: Forecast metrics
    true_f: np.ndarray = rng.normal(28, 1, (50, 10))
    pred_f: np.ndarray = true_f + rng.normal(0, 0.1, (50, 10))
    fm: dict[str, float] = compute_forecast_metrics(true_f, pred_f)
    assert fm["forecast_mae"] < 0.5, "Small noise should give low MAE"

    # Test 3: RUL NASA scoring
    true_rul: np.ndarray = np.array([100.0, 50.0, 25.0])
    pred_rul: np.ndarray = np.array([90.0, 55.0, 20.0])
    rul_m: dict[str, float] = compute_rul_metrics(true_rul, pred_rul)
    assert "rul_nasa_score" in rul_m, "Missing NASA scoring function"

    print("[PASS] eval/phm_metrics.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
