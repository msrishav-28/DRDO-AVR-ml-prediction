"""
Baseline 0: Rule-based threshold detector.

The simplest possible fault detector using hardcoded thresholds from
MIL-STD-1275E operating limits. Serves as the lower-bound baseline
that any learning-based method must beat.

Rules:
    V < 23.5V  → under-voltage fault
    V > 32.5V  → over-voltage fault
    |dV/dt| > 5 V/s  → slew-rate fault
    rolling_std(V, 10) > 2.0V  → ripple fault
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


class ThresholdDetector:
    """Rule-based fault detector using MIL-STD-1275E thresholds."""

    def __init__(
        self,
        v_under: float = 23.5,
        v_over: float = 32.5,
        dv_dt_max: float = 5.0,
        ripple_std_max: float = 2.0,
        ripple_window: int = 10,
    ) -> None:
        self.v_under: float = v_under
        self.v_over: float = v_over
        self.dv_dt_max: float = dv_dt_max
        self.ripple_std_max: float = ripple_std_max
        self.ripple_window: int = ripple_window

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fault flags from raw timeseries data.

        Inputs:
            df: DataFrame with voltage_v column.

        Outputs:
            Binary fault prediction array of shape (n_samples,).
        """
        v: np.ndarray = df["voltage_v"].values
        n: int = len(v)
        fault_flags: np.ndarray = np.zeros(n, dtype=np.int32)

        # Rule 1: Under-voltage
        fault_flags[v < self.v_under] = 1

        # Rule 2: Over-voltage
        fault_flags[v > self.v_over] = 1

        # Rule 3: Slew rate
        if "dv_dt" in df.columns:
            dv_dt: np.ndarray = df["dv_dt"].values
        else:
            dv_dt = np.diff(v, prepend=v[0]) * 10.0
        fault_flags[np.abs(dv_dt) > self.dv_dt_max] = 1

        # Rule 4: Ripple
        rolling_std: np.ndarray = pd.Series(v).rolling(
            window=self.ripple_window, min_periods=1
        ).std().values
        fault_flags[rolling_std > self.ripple_std_max] = 1

        return fault_flags

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions as pseudo-probabilities (0.0 or 1.0)."""
        return self.predict(df).astype(np.float64)


def run_tests() -> None:
    """Sanity checks for threshold detector."""
    test_df: pd.DataFrame = pd.DataFrame({
        "voltage_v": [28.0, 23.0, 33.0, 28.0, 28.0],
    })
    det: ThresholdDetector = ThresholdDetector()
    preds: np.ndarray = det.predict(test_df)
    assert preds[0] == 0, "28V should be normal"
    assert preds[1] == 1, "23V should trigger under-voltage"
    assert preds[2] == 1, "33V should trigger over-voltage"

    print("[PASS] models/baseline_threshold.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
