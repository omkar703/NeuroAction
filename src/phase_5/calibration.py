"""Probability calibration and Brier score evaluation fitted strictly on training folds."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Fits Platt scaling calibrators strictly on out-of-fold training predictions."""

    def __init__(self):
        self._model: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def fit(self, train_scores: np.ndarray, train_labels: np.ndarray) -> "ProbabilityCalibrator":
        """Fit Platt scaling logistic regression on training predictions only."""
        X = np.asarray(train_scores).reshape(-1, 1)
        y = np.asarray(train_labels, dtype=int)
        
        if len(np.unique(y)) < 2:
            # Cannot fit if only one class exists in training fold
            self.is_fitted = False
            return self

        self._model = LogisticRegression(C=1.0, solver="lbfgs")
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_probability(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw continuous scores into calibrated probabilities."""
        if not self.is_fitted or self._model is None:
            # Fall back to raw clipped scores if uncalibrated
            return np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
        X = np.asarray(scores).reshape(-1, 1)
        return self._model.predict_proba(X)[:, 1]

    @staticmethod
    def compute_brier_score(probabilities: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute mean squared error between probabilities and binary ground truth."""
        p = np.clip(np.asarray(probabilities, dtype=np.float32), 0.0, 1.0)
        y = np.asarray(ground_truth, dtype=np.float32)
        return float(np.mean((p - y) ** 2))
