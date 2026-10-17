"""Statistical evaluation, participant-aware uncertainty, and metric computations."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


class StatisticalAnalyzer:
    """Computes primary classification, trajectory, and probabilistic verification metrics."""

    @staticmethod
    def compute_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive binary classification metrics."""
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)

        if len(yt) == 0:
            return {
                "balanced_accuracy": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auroc": 0.5,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        bal_acc = float(0.5 * (sens + spec))
        f1 = float(2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        auroc = 0.5
        if y_score is not None and len(np.unique(yt)) > 1:
            try:
                auroc = float(roc_auc_score(yt, y_score))
            except Exception:
                auroc = 0.5

        return {
            "balanced_accuracy": bal_acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "recall": sens,
            "f1_score": f1,
            "auroc": auroc,
            "confusion_matrix": cm.tolist(),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    @staticmethod
    def compute_trajectory_metrics(
        actual_endpoints: np.ndarray,
        predicted_endpoints: np.ndarray,
    ) -> Dict[str, float]:
        """Compute 3D Euclidean Cartesian trajectory error metrics."""
        p_act = np.asarray(actual_endpoints, dtype=np.float32)
        p_pred = np.asarray(predicted_endpoints, dtype=np.float32)
        
        diffs = p_act - p_pred
        euclidean_errors = np.linalg.norm(diffs, axis=1)

        rmse = float(np.sqrt(np.mean(euclidean_errors ** 2)))
        mean_err = float(np.mean(euclidean_errors))
        std_err = float(np.std(euclidean_errors))

        return {
            "rmse_mm": rmse,
            "mean_error_mm": mean_err,
            "std_error_mm": std_err,
        }

    @staticmethod
    def compute_coverage_metrics(
        total_eligible: int,
        decided_count: int,
        abstained_count: int,
    ) -> Dict[str, float]:
        """Compute coverage and abstention proportions."""
        tot = max(total_eligible, 1)
        return {
            "coverage_rate": float(decided_count / tot),
            "abstention_rate": float(abstained_count / tot),
        }

    @staticmethod
    def compute_participant_clustered_ci(
        df: pd.DataFrame,
        metric_fn: Any,
        participant_col: str = "participant_id",
        n_boot: int = 500,
        random_state: int = 42,
    ) -> Tuple[float, float]:
        """Participant-level clustered bootstrap confidence interval (95%)."""
        rng = np.random.RandomState(random_state)
        subjects = df[participant_col].unique()
        boot_values = []

        for _ in range(n_boot):
            sampled_subs = rng.choice(subjects, size=len(subjects), replace=True)
            sampled_df = pd.concat([df[df[participant_col] == s] for s in sampled_subs], ignore_index=True)
            try:
                val = metric_fn(sampled_df)
                if np.isfinite(val):
                    boot_values.append(val)
            except Exception:
                continue

        if len(boot_values) < 20:
            return (float("nan"), float("nan"))
        return (float(np.percentile(boot_values, 2.5)), float(np.percentile(boot_values, 97.5)))
