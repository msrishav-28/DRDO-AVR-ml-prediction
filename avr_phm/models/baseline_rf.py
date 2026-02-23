"""
Baseline 1: Random Forest classifier + GBM voltage forecast.

RF(500 trees, max_depth=20, class_weight='balanced') for fault classification.
GradientBoostingRegressor for voltage forecast.
This is the strongest non-deep-learning baseline.
"""

import random
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


class RFBaseline:
    """Random Forest fault classifier with GBM voltage forecaster."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 20,
        class_weight: str = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.clf: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            min_samples_leaf=5,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.forecaster: GradientBoostingRegressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
        )
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_forecast_train: np.ndarray | None = None,
        y_forecast_train: np.ndarray | None = None,
    ) -> None:
        """
        Train RF classifier and GBM forecaster.

        Inputs:
            X_train: Feature matrix (n_samples, n_features).
            y_train: Fault labels (n_samples,).
            X_forecast_train: Features for voltage forecasting.
            y_forecast_train: Voltage targets for forecasting.
        """
        self.clf.fit(X_train, y_train)

        if X_forecast_train is not None and y_forecast_train is not None:
            self.forecaster.fit(X_forecast_train, y_forecast_train)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fault labels."""
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fault probabilities."""
        return self.clf.predict_proba(X)

    def forecast_voltage(self, X: np.ndarray) -> np.ndarray:
        """Predict voltage values."""
        return self.forecaster.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        """Return RF feature importances."""
        return self.clf.feature_importances_


def run_tests() -> None:
    """Sanity checks for RF baseline."""
    rng: np.random.Generator = np.random.default_rng(42)
    X: np.ndarray = rng.normal(0, 1, (200, 10))
    y: np.ndarray = rng.integers(0, 2, 200)

    model: RFBaseline = RFBaseline(n_estimators=10, max_depth=5, n_jobs=1)
    model.fit(X, y)
    preds: np.ndarray = model.predict(X)
    assert len(preds) == 200, "Prediction count mismatch"
    assert set(preds).issubset({0, 1}), "Predictions should be binary"

    print("[PASS] models/baseline_rf.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
