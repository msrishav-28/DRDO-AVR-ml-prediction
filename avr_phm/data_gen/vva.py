"""
Validation, Verification, and Accreditation (VVA) suite.

Implements four mandatory metrics for validating synthetic data quality,
required for publication per SAE ARP6887:
    1. Maximum Mean Discrepancy (MMD)
    2. Propensity Score Matching
    3. Train on Synthetic, Test on Real (TSTR)
    4. Autocorrelation Analysis

References:
    - SAE ARP6887: Process Standard for Development and Application of
      Analytical Methods for PHM Solutions
    - Gretton et al. (2012). A Kernel Two-Sample Test. JMLR.
"""

import random
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def compute_mmd(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
    kernel: str = "rbf",
    sigma: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy between real and synthetic distributions.

    Purpose:
        MMD is a non-parametric distance between probability distributions.
        Low MMD indicates that the synthetic data distribution closely
        matches the real data distribution.

    Inputs:
        real_sequences: Shape (n_real, seq_len, n_features).
        synthetic_sequences: Shape (n_synth, seq_len, n_features).
        kernel: Kernel function ('rbf').
        sigma: RBF kernel bandwidth.

    Outputs:
        MMD² value (float). Lower is better.

    Mathematical basis:
        MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        where k(x,y) = exp(-||x-y||²/(2σ²)) is the RBF kernel.

    Acceptance threshold: MMD < 0.05
    Publication requirement: Report mean ± std over 5 random subsamples of size 500.
    """
    # Flatten sequences to (n, seq_len * n_features)
    n_real: int = real_sequences.shape[0]
    n_synth: int = synthetic_sequences.shape[0]
    x: np.ndarray = real_sequences.reshape(n_real, -1)
    y: np.ndarray = synthetic_sequences.reshape(n_synth, -1)

    def _rbf_kernel(a: np.ndarray, b: np.ndarray, sigma_val: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        sq_dist: np.ndarray = (
            np.sum(a**2, axis=1, keepdims=True)
            + np.sum(b**2, axis=1, keepdims=True).T
            - 2.0 * a @ b.T
        )
        return np.exp(-sq_dist / (2.0 * sigma_val**2))

    kxx: np.ndarray = _rbf_kernel(x, x, sigma)
    kyy: np.ndarray = _rbf_kernel(y, y, sigma)
    kxy: np.ndarray = _rbf_kernel(x, y, sigma)

    # Unbiased MMD² estimator
    mmd_sq: float = (
        (kxx.sum() - np.trace(kxx)) / (n_real * (n_real - 1))
        + (kyy.sum() - np.trace(kyy)) / (n_synth * (n_synth - 1))
        - 2.0 * kxy.mean()
    )

    return max(mmd_sq, 0.0)


def compute_mmd_multikernel(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> dict[str, float]:
    """
    Compute MMD with multiple bandwidths and median heuristic.

    Purpose:
        Tests MMD robustness across kernel bandwidths σ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}.

    Inputs:
        real: Shape (n_real, seq_len, n_features).
        synthetic: Shape (n_synth, seq_len, n_features).

    Outputs:
        Dict with per-sigma MMD and median heuristic result.
    """
    sigmas: list[float] = [0.1, 0.5, 1.0, 5.0, 10.0]
    results: dict[str, float] = {}

    for s in sigmas:
        results[f"mmd_sigma_{s}"] = compute_mmd(real, synthetic, sigma=s)

    # Median heuristic for optimal bandwidth
    x_flat: np.ndarray = real.reshape(real.shape[0], -1)
    y_flat: np.ndarray = synthetic.reshape(synthetic.shape[0], -1)
    combined: np.ndarray = np.concatenate([x_flat, y_flat], axis=0)

    # Subsample for efficiency
    n_sub: int = min(500, len(combined))
    rng: np.random.Generator = np.random.default_rng(42)
    idx: np.ndarray = rng.choice(len(combined), n_sub, replace=False)
    sub: np.ndarray = combined[idx]

    pairwise_sq: np.ndarray = (
        np.sum(sub**2, axis=1, keepdims=True)
        + np.sum(sub**2, axis=1, keepdims=True).T
        - 2.0 * sub @ sub.T
    )
    median_dist: float = float(np.sqrt(np.median(pairwise_sq[pairwise_sq > 0])))
    results["mmd_median_heuristic"] = compute_mmd(
        real, synthetic, sigma=median_dist
    )
    results["median_bandwidth"] = median_dist

    return results


def compute_propensity_score(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Train a logistic regression to distinguish real vs synthetic data.

    Purpose:
        If a classifier cannot distinguish real from synthetic data
        (AUC ≈ 0.5), the synthetic data is high quality.

    Inputs:
        real_sequences: Shape (n_real, seq_len, n_features).
        synthetic_sequences: Shape (n_synth, seq_len, n_features).
        n_splits: Number of cross-validation folds.

    Outputs:
        Dict with 'auc_mean', 'auc_std'.

    Mathematical basis:
        Pipeline:
        1. Flatten sequences to feature vectors
        2. Extract: mean, std, min, max, autocorr(lag=1), autocorr(lag=5) per channel
        3. Label: real=1, synthetic=0
        4. 5-fold cross-validated logistic regression
        5. Compute AUC-ROC

    Warning if AUC > 0.65: synthetic quality too low for publication.
    Critical failure if AUC > 0.75: do not proceed with this checkpoint.
    """
    def _extract_summary_features(sequences: np.ndarray) -> np.ndarray:
        """Extract summary statistics per sequence."""
        n: int = sequences.shape[0]
        n_channels: int = sequences.shape[2]
        # 6 stats per channel
        features: np.ndarray = np.zeros((n, n_channels * 6))

        for ch in range(n_channels):
            data: np.ndarray = sequences[:, :, ch]
            offset: int = ch * 6
            features[:, offset] = np.mean(data, axis=1)
            features[:, offset + 1] = np.std(data, axis=1)
            features[:, offset + 2] = np.min(data, axis=1)
            features[:, offset + 3] = np.max(data, axis=1)
            # Autocorrelation at lag 1
            for i in range(n):
                if len(data[i]) > 1:
                    features[i, offset + 4] = np.corrcoef(
                        data[i, :-1], data[i, 1:]
                    )[0, 1] if np.std(data[i]) > 1e-10 else 0.0
                if len(data[i]) > 5:
                    features[i, offset + 5] = np.corrcoef(
                        data[i, :-5], data[i, 5:]
                    )[0, 1] if np.std(data[i]) > 1e-10 else 0.0

        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        return features

    real_feat: np.ndarray = _extract_summary_features(real_sequences)
    synth_feat: np.ndarray = _extract_summary_features(synthetic_sequences)

    X: np.ndarray = np.concatenate([real_feat, synth_feat], axis=0)
    y: np.ndarray = np.concatenate([
        np.ones(len(real_feat)),
        np.zeros(len(synth_feat)),
    ])

    clf: LogisticRegression = LogisticRegression(
        max_iter=1000, random_state=42
    )

    # Bug 16 fix: use MLP discriminator for nonlinear real/synthetic boundary
    from sklearn.neural_network import MLPClassifier
    clf_mlp: MLPClassifier = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )

    from sklearn.model_selection import StratifiedKFold

    skf: StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs: list[float] = []
    proba: np.ndarray = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        clf_mlp.fit(X[train_idx], y[train_idx])
        fold_proba: np.ndarray = clf_mlp.predict_proba(X[val_idx])[:, 1]
        proba[val_idx] = fold_proba
        fold_aucs.append(float(roc_auc_score(y[val_idx], fold_proba)))

    auc_mean: float = float(np.mean(fold_aucs))
    auc_std: float = float(np.std(fold_aucs))

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def evaluate_tstr(
    synthetic_train: np.ndarray,
    synthetic_labels: np.ndarray,
    real_test: np.ndarray,
    real_test_labels: np.ndarray,
    real_train: np.ndarray | None = None,
    real_train_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.

    Purpose:
        Trains a simple classifier on synthetic data only and evaluates
        on real held-out data. Compares to TRTR baseline.

    Inputs:
        synthetic_train: Synthetic training sequences flattened to features.
        synthetic_labels: Labels for synthetic data.
        real_test: Real test sequences flattened to features.
        real_test_labels: Labels for real test data.

    Outputs:
        Dict with 'tstr_f1', 'trtr_f1', 'tstr_trtr_ratio',
        'tstr_auroc', 'trtr_auroc'.

    Publication requirement: TSTR F1 ≥ 0.90 × TRTR F1.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # TSTR: Train on synthetic
    clf_tstr: RandomForestClassifier = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    clf_tstr.fit(synthetic_train, synthetic_labels)
    pred_tstr: np.ndarray = clf_tstr.predict(real_test)
    proba_tstr: np.ndarray = clf_tstr.predict_proba(real_test)
    f1_tstr: float = float(f1_score(
        real_test_labels, pred_tstr, average="macro", zero_division=0
    ))
    # TSTR AUROC
    try:
        tstr_auroc: float = float(roc_auc_score(
            real_test_labels, proba_tstr,
            multi_class="ovr", average="macro",
        ))
    except ValueError:
        tstr_auroc = 0.0

    # TRTR: Train on real, test on real (needs real training data)
    f1_trtr: float = 0.0
    trtr_auroc: float = 0.0
    if real_train is not None and real_train_labels is not None:
        clf_trtr: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        clf_trtr.fit(real_train, real_train_labels)
        pred_trtr: np.ndarray = clf_trtr.predict(real_test)
        proba_trtr: np.ndarray = clf_trtr.predict_proba(real_test)
        f1_trtr = float(f1_score(
            real_test_labels, pred_trtr, average="macro", zero_division=0
        ))
        try:
            trtr_auroc = float(roc_auc_score(
                real_test_labels, proba_trtr,
                multi_class="ovr", average="macro",
            ))
        except ValueError:
            trtr_auroc = 0.0

    ratio: float = f1_tstr / max(f1_trtr, 1e-10) if f1_trtr > 0 else 0.0

    return {
        "tstr_f1": f1_tstr,
        "trtr_f1": f1_trtr,
        "tstr_trtr_ratio": ratio,
        "tstr_auroc": tstr_auroc,
        "trtr_auroc": trtr_auroc,
    }


def compute_autocorrelation_similarity(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
    max_lag: int = 50,
) -> dict[str, float]:
    """
    Compute ACF similarity between real and synthetic data.

    Purpose:
        Verifies that the temporal correlation structure is preserved
        in synthetic data.

    Inputs:
        real_sequences: Shape (n_real, seq_len, n_features).
        synthetic_sequences: Shape (n_synth, seq_len, n_features).
        max_lag: Maximum lag for ACF computation.

    Outputs:
        Dict with per-lag Pearson correlations and overall score.

    Acceptance: Pearson correlation of ACF vectors > 0.95 across all lags.
    """
    lags: list[int] = [1, 5, 10, 20, 50]
    lags = [l for l in lags if l <= max_lag]

    def _compute_acf(sequences: np.ndarray, lag: int) -> np.ndarray:
        """Compute mean autocorrelation at given lag across sequences."""
        n_seq: int = sequences.shape[0]
        n_channels: int = sequences.shape[2]
        acf_values: np.ndarray = np.zeros((n_seq, n_channels))

        for i in range(n_seq):
            for ch in range(n_channels):
                s: np.ndarray = sequences[i, :, ch]
                if len(s) > lag and np.std(s) > 1e-10:
                    acf_values[i, ch] = np.corrcoef(
                        s[:-lag], s[lag:]
                    )[0, 1]

        return np.nan_to_num(acf_values, nan=0.0)

    results: dict[str, float] = {}

    for lag in lags:
        acf_real: np.ndarray = _compute_acf(real_sequences, lag)
        acf_synth: np.ndarray = _compute_acf(synthetic_sequences, lag)

        # Mean ACF per channel
        mean_real: np.ndarray = acf_real.mean(axis=0)
        mean_synth: np.ndarray = acf_synth.mean(axis=0)

        # Pearson correlation between ACF vectors
        if np.std(mean_real) > 1e-10 and np.std(mean_synth) > 1e-10:
            corr: float = float(np.corrcoef(mean_real, mean_synth)[0, 1])
        else:
            corr = 1.0 if np.allclose(mean_real, mean_synth) else 0.0

        results[f"acf_corr_lag{lag}"] = corr

    # Overall score: minimum across all lags
    lag_corrs: list[float] = [
        results[f"acf_corr_lag{l}"] for l in lags
    ]
    results["acf_min_correlation"] = min(lag_corrs) if lag_corrs else 0.0

    return results


def run_full_vva(
    real_sequences: np.ndarray,
    synthetic_sequences: np.ndarray,
) -> dict[str, Any]:
    """
    Run the complete VVA suite and return all metrics.

    Purpose:
        Single entry point for all four VVA metrics.

    Inputs:
        real_sequences: Shape (n_real, seq_len, n_features).
        synthetic_sequences: Shape (n_synth, seq_len, n_features).

    Outputs:
        Dict with all VVA metric results.
    """
    results: dict[str, Any] = {}

    print("[VVA] Computing MMD...")
    results["mmd"] = compute_mmd_multikernel(real_sequences, synthetic_sequences)

    print("[VVA] Computing propensity score...")
    results["propensity"] = compute_propensity_score(
        real_sequences, synthetic_sequences
    )

    print("[VVA] Computing ACF similarity...")
    results["acf"] = compute_autocorrelation_similarity(
        real_sequences, synthetic_sequences
    )

    # Check acceptance criteria — Bug 23 fix: use median-heuristic MMD
    mmd_median: float = results["mmd"].get("mmd_median_heuristic", results["mmd"].get("mmd_sigma_1.0", 1.0))
    auc: float = results["propensity"]["auc_mean"]
    acf_min: float = results["acf"]["acf_min_correlation"]

    results["acceptance"] = {
        "mmd_pass": mmd_median < 0.05,
        "propensity_pass": auc < 0.65,
        "propensity_critical": auc > 0.75,
        "acf_pass": acf_min > 0.95,
    }

    return results


def run_tests() -> None:
    """Sanity checks for VVA metrics."""
    rng: np.random.Generator = np.random.default_rng(42)

    # Test 1: MMD of identical distributions is near zero
    data: np.ndarray = rng.normal(0, 1, (100, 50, 3))
    mmd_val: float = compute_mmd(data, data, sigma=1.0)
    assert mmd_val < 0.01, (
        f"MMD of identical data should be ~0, got {mmd_val:.4f}"
    )

    # Test 2: Propensity score of identical data is near 0.5
    prop: dict[str, Any] = compute_propensity_score(data, data)
    assert 0.15 < prop["auc_mean"] < 0.85, (
        f"AUC for identical data should be ~0.5, got {prop['auc_mean']:.3f}"
    )

    print("[PASS] data_gen/vva.py — all tests passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VVA suite for synthetic data validation")
    parser.add_argument("--test", action="store_true", help="Run sanity checks only")
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        run_tests()
        print("\n[INFO] To run full VVA, call run_full_vva() with real and synthetic data arrays.")
        print("  Example: python -m data_gen.vva --test")
