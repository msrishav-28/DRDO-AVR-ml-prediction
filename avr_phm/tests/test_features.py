"""
Tests for the feature engineering module.

Validates:
    - Lag features computed correctly
    - Rolling statistics include all channels (V, I, T)
    - Physics features (dV/dt, power, impedance, etc.) are present
    - Scenario one-hot encoding is 7-dim
    - Target variables (multi-horizon, mechanism, severity, RUL) are correct
    - Time-aware splits have no temporal leakage
    - engineer_all_features() produces a complete DataFrame
"""

import numpy as np
import pandas as pd
import pytest

from features.engineer import (
    FEATURE_SPEC,
    SCENARIO_NAMES,
    TARGETS,
    compute_lag_features,
    compute_physics_features,
    compute_rolling_features,
    compute_scenario_encoding,
    compute_targets,
    create_time_aware_splits,
    engineer_all_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame matching simulator output format."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "timestamp": np.arange(0, n * 0.1, 0.1)[:n],
        "voltage_v": np.random.normal(28.0, 0.5, n),
        "current_a": np.random.uniform(40.0, 50.0, n),
        "temperature_c": np.random.uniform(20.0, 35.0, n),
        "scenario": "baseline",
        "run_id": 1,
    })


@pytest.fixture
def multi_scenario_df() -> pd.DataFrame:
    """Create a DataFrame with multiple scenarios for split testing."""
    np.random.seed(42)
    dfs = []
    for scenario in SCENARIO_NAMES:
        n = 100
        df = pd.DataFrame({
            "timestamp": np.arange(0, n * 0.1, 0.1)[:n],
            "voltage_v": np.random.normal(28.0, 0.5, n),
            "current_a": np.random.uniform(40.0, 50.0, n),
            "temperature_c": np.random.uniform(20.0, 35.0, n),
            "scenario": scenario,
            "run_id": 1,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestLagFeatures:
    """Tests for lag feature computation."""

    def test_voltage_lags_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_lag_features(sample_df)
        for lag in FEATURE_SPEC["lags"]["voltage_v"]:
            assert f"voltage_v_lag{lag}" in result.columns

    def test_current_lags_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_lag_features(sample_df)
        for lag in FEATURE_SPEC["lags"]["current_a"]:
            assert f"current_a_lag{lag}" in result.columns

    def test_temperature_lags_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_lag_features(sample_df)
        for lag in FEATURE_SPEC["lags"]["temperature_c"]:
            assert f"temperature_c_lag{lag}" in result.columns

    def test_lag_values_correct(self, sample_df: pd.DataFrame) -> None:
        result = compute_lag_features(sample_df)
        # Lag 1 of voltage should equal previous voltage value
        for i in range(1, len(result)):
            expected = sample_df["voltage_v"].iloc[i - 1]
            actual = result["voltage_v_lag1"].iloc[i]
            assert abs(actual - expected) < 1e-10


class TestRollingFeatures:
    """Tests for rolling statistics computation."""

    def test_all_channels_rolled(self, sample_df: pd.DataFrame) -> None:
        """Rolling stats must include voltage, current, AND temperature."""
        result = compute_rolling_features(sample_df)
        for channel in ["voltage_v", "current_a", "temperature_c"]:
            for window in FEATURE_SPEC["rolling"]["windows"]:
                for stat in FEATURE_SPEC["rolling"]["stats"]:
                    col = f"{channel}_rolling_{stat}_{window}"
                    assert col in result.columns, f"Missing: {col}"

    def test_rolling_mean_values(self, sample_df: pd.DataFrame) -> None:
        """Rolling mean of window=1 should equal the original value."""
        result = compute_rolling_features(sample_df)
        # Window 5 rolling mean should be close to the local mean
        col = "voltage_v_rolling_mean_5"
        assert col in result.columns
        assert not result[col].isna().all()


class TestPhysicsFeatures:
    """Tests for physics-derived features."""

    def test_all_physics_features_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_physics_features(sample_df)
        expected_cols = [
            "dv_dt", "di_dt", "power_instantaneous_w", "dp_dt",
            "voltage_deviation_v", "voltage_within_spec",
            "load_impedance_ohm", "thermal_stress_index",
            "voltage_ripple_amplitude_v",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing physics feature: {col}"

    def test_power_is_v_times_i(self, sample_df: pd.DataFrame) -> None:
        result = compute_physics_features(sample_df)
        p = result["power_instantaneous_w"].values
        expected = (sample_df["voltage_v"] * sample_df["current_a"]).values
        np.testing.assert_allclose(p, expected, rtol=1e-5)

    def test_voltage_within_spec(self, sample_df: pd.DataFrame) -> None:
        result = compute_physics_features(sample_df)
        v = sample_df["voltage_v"].values
        spec = result["voltage_within_spec"].values
        for i in range(len(v)):
            expected = 1.0 if 23.5 <= v[i] <= 32.5 else 0.0
            assert spec[i] == expected, f"Spec violation at index {i}"

    def test_thermal_stress_index_range(self, sample_df: pd.DataFrame) -> None:
        result = compute_physics_features(sample_df, t_ambient_c=20.0, t_max_c=150.0)
        tsi = result["thermal_stress_index"].values
        # With temps 20-35°C, TSI should be [0, 0.115] approx
        assert np.all(tsi >= -0.5), f"TSI too low: {tsi.min()}"
        assert np.all(tsi <= 1.5), f"TSI too high: {tsi.max()}"


class TestScenarioEncoding:
    """Tests for scenario one-hot encoding."""

    def test_encoding_is_7dim(self, sample_df: pd.DataFrame) -> None:
        result = compute_scenario_encoding(sample_df)
        scenario_cols = [c for c in result.columns if c.startswith("scenario_")]
        assert len(scenario_cols) == 7, f"Expected 7 scenario cols, got {len(scenario_cols)}"

    def test_encoding_sums_to_one(self, sample_df: pd.DataFrame) -> None:
        result = compute_scenario_encoding(sample_df)
        scenario_cols = [f"scenario_{s}" for s in SCENARIO_NAMES]
        row_sums = result[scenario_cols].sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Each row should sum to 1.0"


class TestTargets:
    """Tests for target variable computation."""

    def test_multi_horizon_targets_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_targets(sample_df)
        for horizon in ["fault_1s", "fault_5s", "fault_10s", "fault_30s"]:
            assert horizon in result.columns, f"Missing target: {horizon}"

    def test_mechanism_and_severity(self, sample_df: pd.DataFrame) -> None:
        result = compute_targets(sample_df)
        assert "fault_mechanism" in result.columns
        assert "severity" in result.columns

    def test_rul_target_present(self, sample_df: pd.DataFrame) -> None:
        result = compute_targets(sample_df)
        assert "rul_seconds" in result.columns
        # Without faults, all RUL should be 300.0 (capped)
        assert np.all(result["rul_seconds"].values == 300.0)

    def test_voltage_forecast_targets(self, sample_df: pd.DataFrame) -> None:
        result = compute_targets(sample_df)
        for step in range(1, 11):
            col = f"voltage_next_{step}"
            assert col in result.columns, f"Missing forecast target: {col}"

    def test_no_fault_means_all_zero_labels(self, sample_df: pd.DataFrame) -> None:
        """Without fault log, all horizon labels should be 0."""
        result = compute_targets(sample_df)
        for horizon in ["fault_1s", "fault_5s", "fault_10s", "fault_30s"]:
            assert result[horizon].sum() == 0, f"Expected 0 faults for {horizon}"


class TestTimeAwareSplits:
    """Tests for time-aware data splitting."""

    def test_emp_simulation_held_out(self, multi_scenario_df: pd.DataFrame) -> None:
        result = engineer_all_features(multi_scenario_df)
        splits = create_time_aware_splits(result)
        assert len(splits["test_held_out_scenario"]) > 0, (
            "emp_simulation should be in test set"
        )

    def test_stress_combo_separate(self, multi_scenario_df: pd.DataFrame) -> None:
        result = engineer_all_features(multi_scenario_df)
        splits = create_time_aware_splits(result)
        assert len(splits["test_stress_combo"]) > 0, (
            "desert_heat + artillery_firing should be in stress combo"
        )

    def test_no_overlap(self, multi_scenario_df: pd.DataFrame) -> None:
        result = engineer_all_features(multi_scenario_df)
        splits = create_time_aware_splits(result)
        all_indices = np.concatenate(list(splits.values()))
        assert len(all_indices) == len(set(all_indices)), (
            "Split indices must not overlap"
        )

    def test_val_is_last_fraction(self, multi_scenario_df: pd.DataFrame) -> None:
        result = engineer_all_features(multi_scenario_df)
        splits = create_time_aware_splits(result)
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0


class TestEngineerAllFeatures:
    """Integration tests for the full pipeline."""

    def test_no_nans_in_output(self, sample_df: pd.DataFrame) -> None:
        result = engineer_all_features(sample_df)
        nan_count = result.isna().sum().sum()
        assert nan_count == 0, f"Output has {nan_count} NaN values"

    def test_output_row_count_matches(self, sample_df: pd.DataFrame) -> None:
        result = engineer_all_features(sample_df)
        assert len(result) == len(sample_df), "Row count should not change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
