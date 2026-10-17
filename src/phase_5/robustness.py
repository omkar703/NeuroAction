"""Robustness and sensitivity analyzer evaluating performance across experimental parameters."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.phase_5.statistical_analysis import StatisticalAnalyzer


class RobustnessAnalyzer:
    """Evaluates stability and generalization across participant, geometric, and kinematic subgroups."""

    @staticmethod
    def evaluate_participant_breakdown(evaluation_df: pd.DataFrame) -> pd.DataFrame:
        """Report performance disaggregated by individual held-out participant."""
        records = []
        for sub, grp in evaluation_df.groupby("participant_id"):
            # Target failure outcome (positive class = failure, score = continuous_score)
            y_fail = (~grp["true_target_success"]).astype(int).values
            scores = grp["continuous_score"].values
            
            # Trajectory metrics
            act_ends = grp[["actual_endpoint_x", "actual_endpoint_y", "actual_endpoint_z"]].values
            pred_ends = grp[["pred_endpoint_x", "pred_endpoint_y", "pred_endpoint_z"]].values
            traj_metrics = StatisticalAnalyzer.compute_trajectory_metrics(act_ends, pred_ends)
            
            # Binary metrics for discordance/failure risk prediction
            bin_metrics = StatisticalAnalyzer.compute_binary_metrics(
                y_true=y_fail,
                y_pred=(scores >= 0.5).astype(int),
                y_score=scores,
            )
            coverage = StatisticalAnalyzer.compute_coverage_metrics(
                total_eligible=len(grp),
                decided_count=int((~grp["is_abstention"]).sum()),
                abstained_count=int(grp["is_abstention"].sum()),
            )

            records.append({
                "Participant": sub,
                "Trials": len(grp),
                "Endpoint RMSE (mm)": traj_metrics["rmse_mm"],
                "Mean Error (mm)": traj_metrics["mean_error_mm"],
                "Balanced Accuracy": bin_metrics["balanced_accuracy"],
                "Sensitivity": bin_metrics["sensitivity"],
                "Specificity": bin_metrics["specificity"],
                "AUROC": bin_metrics["auroc"],
                "Coverage Rate": coverage["coverage_rate"],
            })

        return pd.DataFrame(records)

    @staticmethod
    def evaluate_target_direction_subgroups(evaluation_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate performance stratified by physical reach direction (Left vs Right)."""
        records = []
        for direction, grp in evaluation_df.groupby("physical_hand_direction"):
            act_ends = grp[["actual_endpoint_x", "actual_endpoint_y", "actual_endpoint_z"]].values
            pred_ends = grp[["pred_endpoint_x", "pred_endpoint_y", "pred_endpoint_z"]].values
            traj_metrics = StatisticalAnalyzer.compute_trajectory_metrics(act_ends, pred_ends)

            records.append({
                "Physical Direction": direction.upper(),
                "Trials": len(grp),
                "Endpoint RMSE (mm)": traj_metrics["rmse_mm"],
                "Mean Error (mm)": traj_metrics["mean_error_mm"],
            })
        return pd.DataFrame(records)

    @staticmethod
    def evaluate_tolerance_sensitivity_grid(
        evaluation_df: pd.DataFrame,
        tolerances_mm: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Evaluate target success classification across varying geometric tolerance boundaries."""
        tols = tolerances_mm or [25.0, 50.0, 75.0, 100.0]
        records = []

        # Reference physical target coordinates based on physical reaching goal
        hand_targets = np.where(
            evaluation_df["physical_hand_direction"].values[:, None] == "left",
            np.array([[-188.4, 147.7, 44.9]]),
            np.array([[188.4, 147.8, 54.3]])
        )
        act_ends = evaluation_df[["actual_endpoint_x", "actual_endpoint_y", "actual_endpoint_z"]].values
        pred_ends = evaluation_df[["pred_endpoint_x", "pred_endpoint_y", "pred_endpoint_z"]].values

        act_dists = np.linalg.norm(act_ends - hand_targets, axis=1)
        pred_dists = np.linalg.norm(pred_ends - hand_targets, axis=1)

        for tol in tols:
            y_true_tol = (act_dists <= tol).astype(int)
            y_pred_tol = (pred_dists <= tol).astype(int)

            metrics = StatisticalAnalyzer.compute_binary_metrics(y_true_tol, y_pred_tol)
            records.append({
                "Tolerance (mm)": tol,
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Sensitivity": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "True Success Count": int(y_true_tol.sum()),
                "True Failure Count": int((1 - y_true_tol).sum()),
            })

        return pd.DataFrame(records)

    @staticmethod
    def evaluate_duration_subgroups(evaluation_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate trajectory forecast error stratified by reach movement duration."""
        # Split into Short, Medium, Long reaches if duration is present in manifest
        records = []
        # Approximate reach length from endpoint distance
        dists = np.abs(evaluation_df["actual_endpoint_x"].values)
        bins = [0, 150.0, 180.0, 500.0]
        labels = ["Short (<150 mm)", "Medium (150-180 mm)", "Long (>180 mm)"]
        cat = pd.cut(dists, bins=bins, labels=labels)

        for label, grp in evaluation_df.groupby(cat, observed=False):
            if len(grp) == 0:
                continue
            act_ends = grp[["actual_endpoint_x", "actual_endpoint_y", "actual_endpoint_z"]].values
            pred_ends = grp[["pred_endpoint_x", "pred_endpoint_y", "pred_endpoint_z"]].values
            traj = StatisticalAnalyzer.compute_trajectory_metrics(act_ends, pred_ends)
            records.append({
                "Reach Group": str(label),
                "Trials": len(grp),
                "Endpoint RMSE (mm)": traj["rmse_mm"],
                "Mean Error (mm)": traj["mean_error_mm"],
            })

        return pd.DataFrame(records)
