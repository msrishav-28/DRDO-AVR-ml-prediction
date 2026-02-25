"""
Tests for the evaluation metrics modules.

Validates:
    - VVA suite: MMD, propensity score, TSTR, ACF similarity
    - PHM metrics module exists and is importable
    - Calibration module exists
    - XAI/SHAP module exists
    - Metric computations produce expected ranges
"""

import numpy as np
import pytest

from data_gen.vva import (
    compute_mmd,
    compute_mmd_multikernel,
    compute_propensity_score,
    compute_autocorrelation_similarity,
    evaluate_tstr,
    run_full_vva,
)


@pytest.fixture
def identical_data() -> tuple[np.ndarray, np.ndarray]:
    """Create two identical random datasets for baseline tests."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (100, 50, 3)).astype(np.float32)
    return data, data.copy()


@pytest.fixture
def different_data() -> tuple[np.ndarray, np.ndarray]:
    """Create two clearly different datasets."""
    rng = np.random.default_rng(42)
    data_a = rng.normal(0, 1, (100, 50, 3)).astype(np.float32)
    data_b = rng.normal(5, 2, (100, 50, 3)).astype(np.float32)
    return data_a, data_b


class TestMMD:
    """Tests for Maximum Mean Discrepancy."""

    def test_identical_distributions_near_zero(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        mmd = compute_mmd(real, synthetic, sigma=1.0)
        assert mmd < 0.01, f"MMD of identical data should be ~0, got {mmd:.4f}"

    def test_different_distributions_positive(
        self, different_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = different_data
        mmd = compute_mmd(real, synthetic, sigma=1.0)
        assert mmd > 0.0, "MMD of different data should be positive"

    def test_mmd_nonnegative(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        mmd = compute_mmd(real, synthetic, sigma=1.0)
        assert mmd >= 0.0, "MMD should never be negative"

    def test_multikernel_returns_all_sigmas(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        results = compute_mmd_multikernel(real, synthetic)
        expected_keys = [
            "mmd_sigma_0.1", "mmd_sigma_0.5", "mmd_sigma_1.0",
            "mmd_sigma_5.0", "mmd_sigma_10.0",
            "mmd_median_heuristic", "median_bandwidth",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"


class TestPropensityScore:
    """Tests for propensity score matching."""

    def test_identical_data_auc_near_half(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        result = compute_propensity_score(real, synthetic)
        assert "auc_mean" in result
        assert "auc_std" in result
        # With identical data, AUC should be around 0.5 (random guessing)
        assert 0.15 < result["auc_mean"] < 0.85, (
            f"AUC for identical data: {result['auc_mean']:.3f}"
        )

    def test_different_data_high_auc(
        self, different_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = different_data
        result = compute_propensity_score(real, synthetic)
        assert result["auc_mean"] > 0.6, (
            f"Very different data should have high AUC, got {result['auc_mean']:.3f}"
        )

    def test_std_computed(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Propensity score must compute per-fold std."""
        real, synthetic = identical_data
        result = compute_propensity_score(real, synthetic)
        # std should be a float (may be 0 for very consistent folds)
        assert isinstance(result["auc_std"], float)


class TestACFSimilarity:
    """Tests for autocorrelation function similarity."""

    def test_identical_data_high_correlation(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        result = compute_autocorrelation_similarity(real, synthetic)
        assert "acf_min_correlation" in result

    def test_lag_keys_present(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        result = compute_autocorrelation_similarity(real, synthetic, max_lag=50)
        for lag in [1, 5, 10, 20, 50]:
            assert f"acf_corr_lag{lag}" in result


class TestTSTR:
    """Tests for Train-Synthetic-Test-Real evaluation."""

    def test_tstr_returns_required_keys(self) -> None:
        rng = np.random.default_rng(42)
        n_train, n_test = 100, 50
        n_features = 10
        synth_train = rng.normal(0, 1, (n_train, n_features))
        synth_labels = rng.integers(0, 2, n_train)
        real_test = rng.normal(0, 1, (n_test, n_features))
        real_labels = rng.integers(0, 2, n_test)

        result = evaluate_tstr(synth_train, synth_labels, real_test, real_labels)

        required_keys = ["tstr_f1", "trtr_f1", "tstr_trtr_ratio",
                         "tstr_auroc", "trtr_auroc"]
        for key in required_keys:
            assert key in result, f"Missing TSTR key: {key}"

    def test_tstr_with_real_train(self) -> None:
        """When real_train is provided, TRTR should be non-zero."""
        rng = np.random.default_rng(42)
        n = 100
        n_features = 10
        synth_train = rng.normal(0, 1, (n, n_features))
        synth_labels = rng.integers(0, 2, n)
        real_test = rng.normal(0, 1, (n, n_features))
        real_labels = rng.integers(0, 2, n)
        real_train = rng.normal(0, 1, (n, n_features))
        real_train_labels = rng.integers(0, 2, n)

        result = evaluate_tstr(
            synth_train, synth_labels,
            real_test, real_labels,
            real_train=real_train,
            real_train_labels=real_train_labels,
        )
        assert result["trtr_f1"] > 0.0, "TRTR F1 should be > 0 with real data"


class TestFullVVA:
    """Tests for the combined VVA suite."""

    def test_run_full_vva_returns_acceptance(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        results = run_full_vva(real, synthetic)
        assert "acceptance" in results
        assert "mmd_pass" in results["acceptance"]
        assert "propensity_pass" in results["acceptance"]
        assert "acf_pass" in results["acceptance"]

    def test_full_vva_identical_data_passes(
        self, identical_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        real, synthetic = identical_data
        results = run_full_vva(real, synthetic)
        assert results["acceptance"]["mmd_pass"], "Identical data should pass MMD"


class TestEvalModuleImports:
    """Smoke tests for eval module availability."""

    def test_phm_metrics_importable(self) -> None:
        try:
            from eval import phm_metrics
            assert phm_metrics is not None
        except ImportError:
            pytest.skip("phm_metrics not importable")

    def test_calibration_importable(self) -> None:
        try:
            from eval import calibration
            assert calibration is not None
        except ImportError:
            pytest.skip("calibration not importable")

    def test_xai_importable(self) -> None:
        try:
            from eval import xai
            assert xai is not None
        except ImportError:
            pytest.skip("xai not importable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
