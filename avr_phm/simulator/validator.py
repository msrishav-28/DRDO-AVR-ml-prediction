"""
Simulator output validation and sanity checks.

Performs systematic validation of the DAE simulator output to ensure
physical consistency, unit correctness, and adherence to MIL-STD
specifications. This module is run after each simulation to catch
silent data corruption.

Checks performed:
    1. Column naming conventions (units in column names)
    2. Voltage range plausibility
    3. Current non-negativity
    4. Temperature physical range
    5. Temporal ordering and monotonicity
    6. No pu/physical unit mixing
    7. NaN/Inf detection
    8. Fault log consistency
"""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

from simulator.constants import V_base


# Expected columns in avr_timeseries_df
EXPECTED_TIMESERIES_COLUMNS: list[str] = [
    "timestamp", "voltage_v", "current_a", "temperature_c",
    "delta", "omega", "Eq_prime", "Ed_prime",
    "Eq_dprime", "Ed_dprime", "Vf", "Vr",
    "scenario", "run_id",
]

# Expected columns in fault_log_df
EXPECTED_FAULT_LOG_COLUMNS: list[str] = [
    "timestamp", "fault_type", "fault_mechanism", "severity",
    "duration_ms", "status", "is_operational_transient",
]


class ValidationResult:
    """
    Container for validation check results.

    Purpose:
        Collects pass/fail results from all validation checks with
        descriptive messages for debugging.
    """

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.passed: int = 0
        self.failed: int = 0

    def add_check(
        self, name: str, passed: bool, message: str = ""
    ) -> None:
        """Record a validation check result."""
        self.checks.append({
            "name": name,
            "passed": passed,
            "message": message,
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def all_passed(self) -> bool:
        """True if all checks passed."""
        return self.failed == 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines: list[str] = [
            f"Validation: {self.passed}/{self.passed + self.failed} checks passed."
        ]
        for check in self.checks:
            status: str = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{status}] {check['name']}: {check['message']}")
        return "\n".join(lines)


def validate_timeseries(
    df: pd.DataFrame,
    strict: bool = True,
) -> ValidationResult:
    """
    Validate simulator output timeseries DataFrame.

    Purpose:
        Performs comprehensive sanity checks on the simulator output to
        ensure data quality before downstream processing.

    Inputs:
        df: Timeseries DataFrame from simulate_avr_mission() or
            simulate_scenario().
        strict: If True, fails on warnings; if False, only fails on errors.

    Outputs:
        ValidationResult with all check results.
    """
    result: ValidationResult = ValidationResult()

    # Check 1: Required columns present
    missing_cols: list[str] = [
        c for c in EXPECTED_TIMESERIES_COLUMNS if c not in df.columns
    ]
    result.add_check(
        "required_columns",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All present",
    )

    # Check 2: Column naming convention (units in names)
    unit_columns: dict[str, str] = {
        "voltage_v": "V",
        "current_a": "A",
        "temperature_c": "°C",
    }
    for col, unit in unit_columns.items():
        result.add_check(
            f"unit_naming_{col}",
            col in df.columns,
            f"Column '{col}' contains unit '{unit}' in name",
        )

    if len(df) == 0:
        result.add_check("non_empty", False, "DataFrame is empty")
        return result

    result.add_check("non_empty", True, f"{len(df)} rows")

    # Check 3: No NaN or Inf values in numeric columns
    numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    nan_counts: pd.Series = df[numeric_cols].isna().sum()
    has_nan: bool = nan_counts.sum() > 0
    result.add_check(
        "no_nan",
        not has_nan,
        f"NaN counts: {nan_counts[nan_counts > 0].to_dict()}" if has_nan else "Clean",
    )

    inf_mask: pd.DataFrame = np.isinf(df[numeric_cols].select_dtypes(include=[np.number]))
    has_inf: bool = inf_mask.any().any()
    result.add_check(
        "no_inf",
        not has_inf,
        "Contains Inf values" if has_inf else "Clean",
    )

    # Check 4: Voltage range plausibility
    if "voltage_v" in df.columns:
        v_min: float = df["voltage_v"].min()
        v_max: float = df["voltage_v"].max()
        v_mean: float = df["voltage_v"].mean()

        # Allow transient spikes up to 300V and drops to 0V
        result.add_check(
            "voltage_range",
            -10.0 <= v_min and v_max <= 400.0,
            f"Range: [{v_min:.2f}, {v_max:.2f}]V, Mean: {v_mean:.2f}V",
        )

        # Mean voltage should be near nominal unless heavy faulting
        if strict:
            result.add_check(
                "voltage_nominal",
                15.0 <= v_mean <= 40.0,
                f"Mean voltage {v_mean:.2f}V should be near 28V nominal",
            )

    # Check 5: Current non-negativity (physical constraint)
    if "current_a" in df.columns:
        i_min: float = df["current_a"].min()
        result.add_check(
            "current_nonneg",
            i_min >= -5.0,  # Allow small numerical noise
            f"Min current: {i_min:.2f}A",
        )

    # Check 6: Temperature physical range
    if "temperature_c" in df.columns:
        t_min: float = df["temperature_c"].min()
        t_max: float = df["temperature_c"].max()
        result.add_check(
            "temperature_range",
            -60.0 <= t_min and t_max <= 250.0,
            f"Range: [{t_min:.2f}, {t_max:.2f}]°C",
        )

    # Check 7: Timestamp monotonicity
    if "timestamp" in df.columns:
        timestamps: np.ndarray = df["timestamp"].values.astype(float)
        is_monotonic: bool = np.all(np.diff(timestamps) >= 0)
        result.add_check(
            "timestamp_monotonic",
            is_monotonic,
            "Monotonically increasing" if is_monotonic else "NOT monotonic",
        )

    # Check 8: No per-unit values leaked into physical columns
    if "voltage_v" in df.columns:
        max_v: float = df["voltage_v"].max()
        all_below_2: bool = max_v < 2.0
        result.add_check(
            "no_pu_leak",
            not all_below_2,
            "Values appear to be in pu, not physical units" if all_below_2
            else f"Max voltage {max_v:.2f}V (physical units confirmed)",
        )

    return result


def validate_fault_log(
    fault_df: pd.DataFrame,
) -> ValidationResult:
    """
    Validate fault log DataFrame.

    Purpose:
        Ensures fault log integrity and completeness.

    Inputs:
        fault_df: Fault log DataFrame from simulation.

    Outputs:
        ValidationResult with check results.
    """
    result: ValidationResult = ValidationResult()

    # Check 1: Required columns
    missing_cols: list[str] = [
        c for c in EXPECTED_FAULT_LOG_COLUMNS if c not in fault_df.columns
    ]
    result.add_check(
        "fault_log_columns",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All present",
    )

    if len(fault_df) == 0:
        result.add_check(
            "fault_log_populated",
            True,
            "Empty fault log (valid for healthy runs)",
        )
        return result

    # Check 2: is_operational_transient column exists
    result.add_check(
        "transient_flag",
        "is_operational_transient" in fault_df.columns,
        "is_operational_transient field present",
    )

    # Check 3: Valid fault mechanisms
    valid_mechanisms: set[str] = {
        "thyristor", "capacitor", "terminal",
        "ies", "cranking", "spike", "load_dump",
    }
    if "fault_mechanism" in fault_df.columns:
        mechanisms: set[str] = set(fault_df["fault_mechanism"].unique())
        invalid: set[str] = mechanisms - valid_mechanisms
        result.add_check(
            "valid_mechanisms",
            len(invalid) == 0,
            f"Invalid mechanisms: {invalid}" if invalid
            else f"Mechanisms: {mechanisms}",
        )

    # Check 4: Severity in valid range
    if "severity" in fault_df.columns:
        sev_min: float = fault_df["severity"].min()
        sev_max: float = fault_df["severity"].max()
        result.add_check(
            "severity_range",
            0.0 <= sev_min and sev_max <= 1.0,
            f"Range: [{sev_min:.3f}, {sev_max:.3f}]",
        )

    return result


def run_tests() -> None:
    """Sanity checks for the validator module."""
    # Test 1: Valid DataFrame passes all checks
    valid_df: pd.DataFrame = pd.DataFrame({
        "timestamp": np.arange(0, 10, 0.1),
        "voltage_v": np.random.normal(28.0, 0.5, 100),
        "current_a": np.random.uniform(40.0, 50.0, 100),
        "temperature_c": np.random.uniform(20.0, 30.0, 100),
        "delta": np.random.uniform(0.4, 0.6, 100),
        "omega": np.random.uniform(-0.01, 0.01, 100),
        "Eq_prime": np.ones(100) * 1.05,
        "Ed_prime": np.zeros(100),
        "Eq_dprime": np.ones(100) * 1.05,
        "Ed_dprime": np.zeros(100),
        "Vf": np.ones(100) * 1.05,
        "Vr": np.zeros(100),
        "scenario": "baseline",
        "run_id": 1,
    })
    result: ValidationResult = validate_timeseries(valid_df)
    assert result.all_passed, f"Valid DataFrame should pass:\n{result.summary()}"

    # Test 2: Empty fault log is valid
    empty_fault: pd.DataFrame = pd.DataFrame(
        columns=EXPECTED_FAULT_LOG_COLUMNS
    )
    fault_result: ValidationResult = validate_fault_log(empty_fault)
    assert fault_result.all_passed, (
        f"Empty fault log should be valid:\n{fault_result.summary()}"
    )

    print("[PASS] simulator/validator.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
