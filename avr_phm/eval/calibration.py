"""
Calibration analysis for probabilistic predictions.

Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and reliability diagram generation per master plan Section 10.

A well-calibrated model should have: P(Y=1 | predicted_prob ∈ bin_k) ≈ mean(bin_k)

References:
    - Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    - Naeini et al. (2015). Obtaining Well Calibrated Probabilities Using
      Bayesian Binning into Quantiles. AAAI.
"""

import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Purpose:
        Measures how well predicted probabilities align with actual outcomes.

    Inputs:
        y_true: Binary ground truth labels (n_samples,).
        y_proba: Predicted probabilities (n_samples,).
        n_bins: Number of equally-spaced probability bins.

    Outputs:
        Dict with 'ece', 'mce', and per-bin data for reliability diagram.

    Mathematical basis:
        ECE = Σ (n_k / N) * |acc(k) - conf(k)|
        MCE = max_k |acc(k) - conf(k)|
        where:
            acc(k) = fraction of positives in bin k
            conf(k) = mean predicted probability in bin k
            n_k = number of samples in bin k
    """
    bin_edges: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
    ece: float = 0.0
    mce: float = 0.0
    n_total: int = len(y_true)

    bin_data: list[dict[str, float]] = []

    for i in range(n_bins):
        mask: np.ndarray = (y_proba >= bin_edges[i]) & (
            y_proba < bin_edges[i + 1]
        )
        n_bin: int = int(mask.sum())

        if n_bin == 0:
            bin_data.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2.0),
                "accuracy": 0.0,
                "confidence": 0.0,
                "count": 0,
                "gap": 0.0,
            })
            continue

        bin_accuracy: float = float(y_true[mask].mean())
        bin_confidence: float = float(y_proba[mask].mean())
        gap: float = abs(bin_accuracy - bin_confidence)

        ece += (n_bin / n_total) * gap
        mce = max(mce, gap)

        bin_data.append({
            "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2.0),
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "count": n_bin,
            "gap": gap,
        })

    return {
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "bins": bin_data,
    }


def plot_reliability_diagram(
    calibration_data: dict[str, Any],
    model_name: str = "Model",
    save_path: str | None = None,
) -> None:
    """
    Generate publication-quality reliability diagram.

    Purpose:
        Visualizes model calibration as a bar chart of predicted vs actual
        probability, compared against the diagonal (perfect calibration).

    Inputs:
        calibration_data: Output from compute_ece().
        model_name: Model name for plot title.
        save_path: Path to save figure, or None to display.
    """
    bins: list[dict[str, float]] = calibration_data["bins"]
    ece: float = calibration_data["ece"]

    bin_centers: list[float] = [b["bin_center"] for b in bins]
    accuracies: list[float] = [b["accuracy"] for b in bins]
    confidences: list[float] = [b["confidence"] for b in bins]
    counts: list[int] = [int(b["count"]) for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    bar_width: float = 1.0 / len(bins) * 0.8
    ax1.bar(bin_centers, accuracies, width=bar_width, color="#2196F3",
            alpha=0.7, label="Accuracy", edgecolor="black", linewidth=0.5)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(f"{model_name} — Reliability Diagram\nECE = {ece:.4f}", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Histogram of predictions
    ax2.bar(bin_centers, counts, width=bar_width, color="#FF9800",
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


def run_tests() -> None:
    """Sanity checks for calibration module."""
    rng: np.random.Generator = np.random.default_rng(42)

    # Test 1: Perfect calibration → ECE ≈ 0
    n: int = 1000
    y_true: np.ndarray = rng.integers(0, 2, n)
    y_proba: np.ndarray = y_true.astype(np.float64)
    result: dict[str, Any] = compute_ece(y_true, y_proba)
    assert result["ece"] < 0.01, (
        f"Perfect calibration should have ECE ≈ 0, got {result['ece']:.4f}"
    )

    # Test 2: Random predictions → ECE > 0
    y_random: np.ndarray = rng.uniform(0, 1, n)
    result_random: dict[str, Any] = compute_ece(y_true, y_random)
    assert result_random["ece"] > 0.0, "Random predictions should have ECE > 0"

    print("[PASS] eval/calibration.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
