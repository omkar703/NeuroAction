"""Classical machine learning decoders for pre-movement EEG classification.

Includes:
1. Chance baseline (stratified/uniform)
2. Logistic Regression (L2 regularized with StandardScaler)
3. Linear Discriminant Analysis (LDA with shrinkage)
4. CSP + LDA
5. CSP + Logistic Regression
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.pre_eeg import CommonSpatialPattern


class ChanceBaseline(BaseEstimator, ClassifierMixin):
    """Predicts empirical class prior probabilities or 0.5 balanced chance."""

    def __init__(self, balanced: bool = True):
        self.balanced = balanced
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChanceBaseline":
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        probs = np.full((n, 2), 0.5, dtype=np.float32)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class LogisticRegressionDecoder(BaseEstimator, ClassifierMixin):
    """Standardized Logistic Regression with L2 regularization."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipe: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionDecoder":
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )),
        ])
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)


class LDADecoder(BaseEstimator, ClassifierMixin):
    """Standardized Linear Discriminant Analysis with automatic shrinkage."""

    def __init__(self, shrinkage: str = "auto"):
        self.shrinkage = shrinkage
        self.pipe: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDADecoder":
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)),
        ])
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)


class CSPLDADecoder(BaseEstimator, ClassifierMixin):
    """Common Spatial Pattern + Linear Discriminant Analysis.
    
    Expects 3D raw EEG input: (n_trials, n_samples, n_channels).
    """

    def __init__(self, n_components: int = 3, shrinkage: str = "auto"):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.csp: Optional[CommonSpatialPattern] = None
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[LinearDiscriminantAnalysis] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLDADecoder":
        self.csp = CommonSpatialPattern(n_components=self.n_components)
        X_csp = self.csp.fit_transform(X, y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_csp)
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)
        self.clf.fit(X_scaled, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_csp = self.csp.transform(X)
        X_scaled = self.scaler.transform(X_csp)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_csp = self.csp.transform(X)
        X_scaled = self.scaler.transform(X_csp)
        return self.clf.predict(X_scaled)


class CSPLogRegDecoder(BaseEstimator, ClassifierMixin):
    """Common Spatial Pattern + Logistic Regression.
    
    Expects 3D raw EEG input: (n_trials, n_samples, n_channels).
    """

    def __init__(self, n_components: int = 3, C: float = 1.0, random_state: int = 42):
        self.n_components = n_components
        self.C = C
        self.random_state = random_state
        self.csp: Optional[CommonSpatialPattern] = None
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[LogisticRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLogRegDecoder":
        self.csp = CommonSpatialPattern(n_components=self.n_components)
        X_csp = self.csp.fit_transform(X, y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_csp)
        self.clf = LogisticRegression(C=self.C, random_state=self.random_state, solver="lbfgs")
        self.clf.fit(X_scaled, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_csp = self.csp.transform(X)
        X_scaled = self.scaler.transform(X_csp)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_csp = self.csp.transform(X)
        X_scaled = self.scaler.transform(X_csp)
        return self.clf.predict(X_scaled)
