"""
Feature engineering for AVR-PHM time-series data.

Implements the complete FEATURE_SPEC from master plan Section 7:
    - Raw lag features per channel (V, I, T)
    - Rolling statistics (mean, std, min, max, skew)
    - Physics-derived features (dV/dt, power, impedance, ripple, etc.)
    - Scenario one-hot encoding
    - Target variable generation (multi-horizon fault warning)
    - Time-aware data splitting (no temporal leakage)

All features are computed per window applied to windowed sequences.
Column naming includes units: voltage_v, current_a, temperature_c.
"""

import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


# ─── Feature Specification (from master plan Section 7) ──────────────────────
FEATURE_SPEC: dict[str, Any] = {
    "lags": {
        "voltage_v": [1, 2, 3, 5, 10, 20],
        "current_a": [1, 2, 3, 5, 10],
        "temperature_c": [1, 5, 10],
    },
    "rolling": {
        "windows": [5, 10, 20, 50],
        "stats": ["mean", "std", "min", "max", "skew"],
    },
    "physics": {
        "voltage_rate_of_change": "dV/dt using finite difference",
        "current_rate_of_change": "dI/dt",
        "power_instantaneous": "V(t) * I(t)",
        "power_rate_of_change": "d(P)/dt",
        "voltage_deviation_from_nominal": "V(t) - 28.0",
        "voltage_within_spec": "indicator: 1 if 23.5 <= V <= 32.5 else 0",
        "load_impedance_estimate": "V(t) / I(t)",
        "thermal_stress_index": "(T(t) - T_ambient) / (T_max - T_ambient)",
        "voltage_ripple_amplitude": "rolling_std over 5-sample window * 2*sqrt(2)",
    },
    "scenario_encoding": "7-dim one-hot",
}

TARGETS: dict[str, str] = {
    "fault_1s": "bool: any fault in next 10 samples (1s)",
    "fault_5s": "bool: any fault in next 50 samples (5s)",
    "fault_10s": "bool: any fault in next 100 samples (10s)",
    "fault_30s": "bool: any fault in next 300 samples (30s)",
    "fault_mechanism": "int: 0=none, 1=thyristor, 2=capacitor, 3=terminal",
    "voltage_next_10_steps": "array of shape (10,): V(t+1)...V(t+10)",
    "severity": "int: 0=healthy, 1=incipient, 2=developing, 3=critical",
}

SCENARIO_NAMES: list[str] = [
    "baseline", "arctic_cold", "desert_heat", "artillery_firing",
    "rough_terrain", "weapons_active", "emp_simulation",
]

MECHANISM_MAP: dict[str, int] = {
    "none": 0, "healthy": 0,
    "thyristor": 1,
    "capacitor": 2,
    "terminal": 3,
}


def compute_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute lag features for all channels per FEATURE_SPEC.

    Purpose:
        Creates time-lagged versions of voltage, current, and temperature
        signals to capture temporal patterns.

    Inputs:
        df: DataFrame with columns voltage_v, current_a, temperature_c.

    Outputs:
        DataFrame with lag feature columns appended.

    Mathematical basis:
        lag_k(x, t) = x(t - k)
    """
    result: pd.DataFrame = df.copy()

    for channel, lags in FEATURE_SPEC["lags"].items():
        if channel not in result.columns:
            continue
        for lag in lags:
            col_name: str = f"{channel}_lag{lag}"
            result[col_name] = result[channel].shift(lag)

    return result


def compute_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling statistics for voltage and current channels.

    Purpose:
        Captures local statistical properties of the signal within
        sliding windows of different sizes.

    Inputs:
        df: DataFrame with voltage_v and current_a columns.

    Outputs:
        DataFrame with rolling statistic columns appended.

    Mathematical basis:
        Per window w: mean, std, min, max, skewness of x(t-w+1:t).
    """
    result: pd.DataFrame = df.copy()
    channels: list[str] = ["voltage_v", "current_a"]

    for channel in channels:
        if channel not in result.columns:
            continue
        for window in FEATURE_SPEC["rolling"]["windows"]:
            rolling: pd.core.window.Rolling = result[channel].rolling(
                window=window, min_periods=1
            )
            for stat in FEATURE_SPEC["rolling"]["stats"]:
                col_name: str = f"{channel}_rolling_{stat}_{window}"
                if stat == "mean":
                    result[col_name] = rolling.mean()
                elif stat == "std":
                    result[col_name] = rolling.std()
                elif stat == "min":
                    result[col_name] = rolling.min()
                elif stat == "max":
                    result[col_name] = rolling.max()
                elif stat == "skew":
                    result[col_name] = rolling.skew()

    return result


def compute_physics_features(
    df: pd.DataFrame,
    dt: float = 0.1,
    nominal_voltage_v: float = 28.0,
    voltage_min_v: float = 23.5,
    voltage_max_v: float = 32.5,
    t_ambient_c: float = 25.0,
    t_max_c: float = 150.0,
) -> pd.DataFrame:
    """
    Compute physics-derived features per FEATURE_SPEC.

    Purpose:
        Creates features grounded in the physical domain model of the
        AVR system. These are the most important features for the PINN.

    Inputs:
        df: DataFrame with voltage_v, current_a, temperature_c columns.
        dt: Sampling interval in seconds (0.1s for 10Hz).
        nominal_voltage_v: Nominal bus voltage (28V).
        voltage_min_v: MIL-STD-1275E under-voltage threshold.
        voltage_max_v: MIL-STD-1275E over-voltage threshold.
        t_ambient_c: Ambient temperature for thermal stress calculation.
        t_max_c: Maximum junction temperature.

    Outputs:
        DataFrame with physics feature columns appended.

    Mathematical basis:
        dV/dt: finite difference d(voltage_v)/dt
        dI/dt: finite difference d(current_a)/dt
        P(t): V(t) * I(t)
        dP/dt: finite difference d(P)/dt
        V_dev(t): V(t) - V_nominal
        V_spec(t): 1 if V_min <= V(t) <= V_max else 0
        Z_est(t): V(t) / I(t)
        TSI(t): (T(t) - T_ambient) / (T_max - T_ambient)
        Ripple(t): rolling_std(V, 5) * 2√2
    """
    result: pd.DataFrame = df.copy()

    v: pd.Series = result["voltage_v"]
    i: pd.Series = result["current_a"]
    t: pd.Series = result["temperature_c"]

    # Rate of change features
    result["dv_dt"] = v.diff() / dt
    result["di_dt"] = i.diff() / dt

    # Power features
    result["power_instantaneous_w"] = v * i
    result["dp_dt"] = result["power_instantaneous_w"].diff() / dt

    # Voltage deviation from nominal
    result["voltage_deviation_v"] = v - nominal_voltage_v

    # Voltage within spec indicator
    result["voltage_within_spec"] = (
        (v >= voltage_min_v) & (v <= voltage_max_v)
    ).astype(np.float32)

    # Load impedance estimate
    safe_current: pd.Series = i.replace(0, 1e-6)
    result["load_impedance_ohm"] = v / safe_current

    # Thermal stress index
    delta_t_max: float = t_max_c - t_ambient_c
    if abs(delta_t_max) > 1e-6:
        result["thermal_stress_index"] = (t - t_ambient_c) / delta_t_max
    else:
        result["thermal_stress_index"] = 0.0

    # Voltage ripple amplitude
    rolling_std_5: pd.Series = v.rolling(window=5, min_periods=1).std()
    result["voltage_ripple_amplitude_v"] = rolling_std_5 * 2.0 * np.sqrt(2.0)

    return result


def compute_scenario_encoding(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add scenario one-hot encoding columns.

    Purpose:
        Creates 7-dimensional one-hot encoding of the scenario field.

    Inputs:
        df: DataFrame with 'scenario' column.

    Outputs:
        DataFrame with scenario_* columns appended.
    """
    result: pd.DataFrame = df.copy()

    for scenario in SCENARIO_NAMES:
        col_name: str = f"scenario_{scenario}"
        result[col_name] = (result["scenario"] == scenario).astype(np.float32)

    return result


def compute_targets(
    df: pd.DataFrame,
    fault_log_df: pd.DataFrame | None = None,
    sampling_rate_hz: float = 10.0,
) -> pd.DataFrame:
    """
    Generate target variables for multi-horizon fault prediction.

    Purpose:
        Creates binary fault warning labels for horizons {1s, 5s, 10s, 30s},
        fault mechanism labels, severity labels, and voltage forecast targets.

    Inputs:
        df: Feature DataFrame with voltage_v column.
        fault_log_df: Fault log DataFrame with timestamp and fault_mechanism.
        sampling_rate_hz: Data sampling rate (10Hz).

    Outputs:
        DataFrame with target columns appended.

    Mathematical basis:
        fault_τ(t) = 1 if any fault in window [t, t + τ*sampling_rate]
        voltage_next_k(t) = V(t+k) for k ∈ {1,...,10}
    """
    result: pd.DataFrame = df.copy()
    n: int = len(result)

    # ─── Multi-horizon fault warnings ────────────────────────────────────────
    # Build fault indicator from fault log
    fault_indicator: np.ndarray = np.zeros(n, dtype=np.int32)
    mechanism_indicator: np.ndarray = np.zeros(n, dtype=np.int32)
    severity_indicator: np.ndarray = np.zeros(n, dtype=np.int32)

    if fault_log_df is not None and len(fault_log_df) > 0:
        # Filter out operational transients
        actual_faults: pd.DataFrame = fault_log_df
        if "is_operational_transient" in fault_log_df.columns:
            actual_faults = fault_log_df[
                fault_log_df["is_operational_transient"] != True  # noqa: E712
            ]

        for _, fault_row in actual_faults.iterrows():
            fault_time: float = float(fault_row.get("timestamp", 0))
            fault_idx: int = int(fault_time * sampling_rate_hz)
            if 0 <= fault_idx < n:
                fault_indicator[fault_idx] = 1
                mech: str = str(fault_row.get("fault_mechanism", "none"))
                mechanism_indicator[fault_idx] = MECHANISM_MAP.get(mech, 0)

                sev_str: str = str(fault_row.get("fault_type", "healthy"))
                if sev_str == "healthy":
                    severity_indicator[fault_idx] = 0
                elif sev_str == "incipient":
                    severity_indicator[fault_idx] = 1
                elif sev_str == "developing":
                    severity_indicator[fault_idx] = 2
                elif sev_str == "critical":
                    severity_indicator[fault_idx] = 3

    # Compute forward-looking fault windows
    horizons: dict[str, int] = {
        "fault_1s": 10,
        "fault_5s": 50,
        "fault_10s": 100,
        "fault_30s": 300,
    }

    for horizon_name, horizon_samples in horizons.items():
        labels: np.ndarray = np.zeros(n, dtype=np.int32)
        # Reverse cumulative sum approach for efficiency
        cumsum: np.ndarray = np.cumsum(fault_indicator[::-1])[::-1]
        for i in range(n):
            end_idx: int = min(i + horizon_samples, n)
            if end_idx > i:
                window_faults: int = int(
                    cumsum[i] - (cumsum[end_idx] if end_idx < n else 0)
                )
                labels[i] = 1 if window_faults > 0 else 0
        result[horizon_name] = labels

    # Fault mechanism and severity
    result["fault_mechanism"] = mechanism_indicator
    result["severity"] = severity_indicator

    # ─── Voltage forecast targets ────────────────────────────────────────────
    for step in range(1, 11):
        col_name: str = f"voltage_next_{step}"
        result[col_name] = result["voltage_v"].shift(-step)

    return result


def engineer_all_features(
    df: pd.DataFrame,
    fault_log_df: pd.DataFrame | None = None,
    dt: float = 0.1,
    ambient_temp_c: float = 25.0,
) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline.

    Purpose:
        Single entry point that applies all feature transformations
        in the correct order.

    Inputs:
        df: Raw timeseries DataFrame from simulation.
        fault_log_df: Fault log DataFrame, or None.
        dt: Sampling interval (s).
        ambient_temp_c: Ambient temperature for thermal stress.

    Outputs:
        Fully-featured DataFrame with all engineered columns.
    """
    result: pd.DataFrame = df.copy()

    # Step 1: Lag features
    result = compute_lag_features(result)

    # Step 2: Rolling statistics
    result = compute_rolling_features(result)

    # Step 3: Physics-derived features
    result = compute_physics_features(
        result, dt=dt, t_ambient_c=ambient_temp_c
    )

    # Step 4: Scenario encoding
    if "scenario" in result.columns:
        result = compute_scenario_encoding(result)

    # Step 5: Target variables
    result = compute_targets(result, fault_log_df)

    # Step 6: Clean up NaNs from lags and rolling
    result = result.bfill()
    result = result.ffill()
    result = result.fillna(0.0)

    return result


def create_time_aware_splits(
    df: pd.DataFrame,
    test_scenarios: list[str] | None = None,
    val_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Create time-aware train/val/test splits with NO temporal leakage.

    Purpose:
        Implements the split strategy from master plan Section 7:
        1. Hold out emp_simulation entirely for test set
        2. Last 15% of each remaining run = validation
        3. Everything before val split = training
        4. Second test: hold out desert_heat + artillery_firing combined

    Inputs:
        df: Feature-engineered DataFrame with 'scenario', 'run_id' columns.
        test_scenarios: List of scenarios held out for testing.
            Default: ["emp_simulation"].
        val_fraction: Fraction of each run reserved for validation.

    Outputs:
        Dict with keys: 'train', 'val', 'test_held_out_scenario',
        'test_stress_combo'. Values are index arrays.

    CRITICAL: Do NOT use random shuffle. Time-series must never leak
    future into past.
    """
    if test_scenarios is None:
        test_scenarios = ["emp_simulation"]

    stress_combo_scenarios: list[str] = ["desert_heat", "artillery_firing"]

    indices_train: list[int] = []
    indices_val: list[int] = []
    indices_test_held: list[int] = []
    indices_test_stress: list[int] = []

    for (scenario, run_id), group in df.groupby(["scenario", "run_id"]):
        group_indices: np.ndarray = group.index.values
        n_group: int = len(group_indices)

        if scenario in test_scenarios:
            indices_test_held.extend(group_indices.tolist())
        elif scenario in stress_combo_scenarios:
            indices_test_stress.extend(group_indices.tolist())
        else:
            # Time-aware split: last val_fraction for validation
            val_start: int = int(n_group * (1.0 - val_fraction))
            indices_train.extend(group_indices[:val_start].tolist())
            indices_val.extend(group_indices[val_start:].tolist())

    splits: dict[str, np.ndarray] = {
        "train": np.array(indices_train, dtype=np.int64),
        "val": np.array(indices_val, dtype=np.int64),
        "test_held_out_scenario": np.array(
            indices_test_held, dtype=np.int64
        ),
        "test_stress_combo": np.array(
            indices_test_stress, dtype=np.int64
        ),
    }

    # Log split statistics
    total: int = sum(len(v) for v in splits.values())
    print(f"[SPLITS] Total: {total}")
    for split_name, split_idx in splits.items():
        print(f"  {split_name}: {len(split_idx)} samples "
              f"({100.0 * len(split_idx) / max(total, 1):.1f}%)")

    return splits


def run_tests() -> None:
    """Sanity checks for feature engineering."""
    # Test 1: Feature engineering produces expected physics columns
    test_df: pd.DataFrame = pd.DataFrame({
        "timestamp": np.arange(0, 10, 0.1),
        "voltage_v": np.random.normal(28.0, 0.5, 100),
        "current_a": np.random.uniform(40.0, 50.0, 100),
        "temperature_c": np.random.uniform(20.0, 30.0, 100),
        "scenario": "baseline",
        "run_id": 1,
    })
    result: pd.DataFrame = engineer_all_features(test_df)
    assert "dv_dt" in result.columns, "Missing dV/dt feature"
    assert "power_instantaneous_w" in result.columns, "Missing power feature"
    assert "voltage_ripple_amplitude_v" in result.columns, "Missing ripple feature"

    # Test 2: Time-aware split preserves temporal order
    test_df["scenario"] = np.where(
        np.arange(100) < 80, "baseline", "emp_simulation"
    )
    test_df["run_id"] = 1
    splits: dict[str, np.ndarray] = create_time_aware_splits(test_df)
    assert len(splits["test_held_out_scenario"]) > 0, (
        "emp_simulation should be in test set"
    )
    assert len(splits["train"]) > 0, "Training set should not be empty"

    print("[PASS] features/engineer.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
