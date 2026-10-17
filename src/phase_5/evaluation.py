"""Master evaluation pipeline for Phase 5 prospective prediction and comparative benchmarking."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.phase_5.config import Phase5Config
from src.phase_5.fold_manager import LOSOFoldCoordinator
from src.phase_5.baseline_models import (
    PersistenceBaselineModel,
    FairConstantVelocityBaselineModel,
    NeuralTargetClassifier,
    KinematicTrajectoryForecaster,
)
from src.phase_5.fusion_model import MultimodalOutcomeFusionModel
from src.phase_5.outcome_labels import OutcomeLabelEngine
from src.phase_5.prediction_policy import PredictionPolicy, DecisionPolicyState
from src.phase_4.trial_registry import TrialRegistry
from src.phase_5.schemas import Phase5EvaluationRow, TaskMetricsSummary
from src.phase_5.statistical_analysis import StatisticalAnalyzer


class Phase5Evaluator:
    """Coordinates Leave-One-Subject-Out evaluation across all tasks and models."""

    def __init__(self, config: Optional[Phase5Config] = None):
        self.config = config or Phase5Config()
        self.coordinator = LOSOFoldCoordinator(config=self.config)
        self.registry = TrialRegistry(manifest_path=self.config.manifest_path)
        self.policy = PredictionPolicy(
            tolerance_mm=self.config.target_tolerance_mm,
            abstention_margin_mm=self.config.distance_abstention_margin_mm,
            min_confidence=self.config.confidence_threshold,
        )
        self.fusion_model = MultimodalOutcomeFusionModel(policy=self.policy)

    def run_cohort_evaluation(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run strict Leave-One-Subject-Out evaluation across all 269 cohort trials."""
        subject_folds = self.coordinator.get_subject_folds()
        eval_rows: List[Dict[str, Any]] = []

        all_persistence_errs = []
        all_cv_errs = []

        for test_sub in subject_folds:
            context = self.coordinator.get_fold_context(test_sub)
            test_samples = context["test_samples"]
            neural_adapter = context["neural_adapter"]
            kinematic_adapter = context["kinematic_adapter"]
            target_refs = context["target_references"]
            dt_train_mean_s = context["dt_train_mean_s"]
            fold_id = f"held_out_{test_sub.replace('-', '')}"

            neural_clf = NeuralTargetClassifier(neural_adapter)
            kin_forecaster = KinematicTrajectoryForecaster(kinematic_adapter)

            for s in test_samples:
                trial = self.registry.get_trial(s.trial_id)
                if trial is None:
                    continue

                # 1. Physical target reference for this reach
                # Target reference is chosen by s.target_goal (stimulus proxy)
                target_ref = target_refs[s.target_goal]

                # 2. Model Inference
                n_res = neural_clf.predict(trial)
                f_res = kin_forecaster.forecast(
                    trial=trial,
                    protocol="kinematic_onset",
                    speed_threshold_mm_s=self.config.speed_threshold_mm_s,
                    horizons_ms=self.config.horizons_ms,
                )

                # 3. Baselines
                p_pers = PersistenceBaselineModel.predict_endpoint(s.prefix_positions)
                p_cv = FairConstantVelocityBaselineModel.predict_endpoint(
                    prefix_positions=s.prefix_positions,
                    dt_train_mean_s=dt_train_mean_s,
                )
                all_persistence_errs.append(np.linalg.norm(p_pers - s.actual_endpoint))
                all_cv_errs.append(np.linalg.norm(p_cv - s.actual_endpoint))

                # 4. Multimodal Fusion Prediction
                pred = self.fusion_model.predict(
                    trial_id=s.trial_id,
                    participant_id=s.participant_id,
                    neural_result=n_res,
                    forecast_result=f_res,
                    target_reference=target_ref,
                    fold_id=fold_id,
                    training_participants=context["training_subjects"],
                )

                # 5. Retrospective Ground Truth Outcome Label
                outcome = OutcomeLabelEngine.compute_outcome(
                    trial_id=s.trial_id,
                    target_reference=target_ref,
                    actual_endpoint=s.actual_endpoint,
                    custom_tolerance_mm=self.config.target_tolerance_mm,
                )

                endpoint_err = float(np.linalg.norm(pred.pred_endpoint - s.actual_endpoint))
                pred_success = (pred.decision_policy_state == DecisionPolicyState.PREDICTED_SUCCESS)
                agreement = (pred_success == outcome.target_success)

                eval_rows.append({
                    "trial_id": s.trial_id,
                    "participant_id": s.participant_id,
                    "run_id": s.run_id,
                    "congruency": s.congruency,
                    "target_goal": s.target_goal,
                    "physical_hand_direction": s.physical_hand_direction,
                    "is_held_out": True,
                    "fold_id": fold_id,
                    "target_reference_provenance": target_ref.provenance,
                    "is_target_authoritative": target_ref.is_authoritative,
                    "actual_endpoint_x": float(s.actual_endpoint[0]),
                    "actual_endpoint_y": float(s.actual_endpoint[1]),
                    "actual_endpoint_z": float(s.actual_endpoint[2]),
                    "true_target_success": outcome.target_success,
                    "true_target_failure": outcome.target_failure,
                    "outcome_status": outcome.status,
                    "prediction_time_ms": pred.prediction_time_ms,
                    "neural_goal_pred": pred.neural_goal_pred,
                    "neural_confidence": pred.neural_confidence,
                    "pred_endpoint_x": float(pred.pred_endpoint[0]),
                    "pred_endpoint_y": float(pred.pred_endpoint[1]),
                    "pred_endpoint_z": float(pred.pred_endpoint[2]),
                    "endpoint_error_mm": endpoint_err,
                    "pred_distance_to_target_mm": pred.distance_to_target_mm,
                    "decision_policy_state": pred.decision_policy_state,
                    "is_abstention": pred.is_abstention,
                    "continuous_score": pred.continuous_score,
                    "predicted_success": pred_success,
                    "outcome_agreement": agreement,
                    "evaluation_eligibility": "ELIGIBLE_PROXY",
                })

        df = pd.DataFrame(eval_rows)

        # Summary Metrics Calculation
        ridge_errors = df["endpoint_error_mm"].values
        ridge_rmse = float(np.sqrt(np.mean(ridge_errors ** 2)))
        ridge_mean = float(np.mean(ridge_errors))
        cv_rmse = float(np.sqrt(np.mean(np.array(all_cv_errs) ** 2)))
        cv_mean = float(np.mean(all_cv_errs))
        pers_rmse = float(np.sqrt(np.mean(np.array(all_persistence_errs) ** 2)))
        pers_mean = float(np.mean(all_persistence_errs))

        # Binary condition discrimination (ground truth y = incongruent)
        y_cond = (df["congruency"] == "incongruent").astype(int).values
        # Predicted condition conflict from fusion score
        y_pred_cond = (df["continuous_score"] >= 0.5).astype(int).values
        cond_metrics = StatisticalAnalyzer.compute_binary_metrics(
            y_true=y_cond,
            y_pred=y_pred_cond,
            y_score=df["continuous_score"].values,
        )

        # Neural stimulus target decoding (Task A)
        y_neural_true = (df["target_goal"] == "right").astype(int).values
        y_neural_pred = (df["neural_goal_pred"] == "right").astype(int).values
        neural_metrics = StatisticalAnalyzer.compute_binary_metrics(y_neural_true, y_neural_pred)

        # Coverage metrics
        cov_metrics = StatisticalAnalyzer.compute_coverage_metrics(
            total_eligible=len(df),
            decided_count=int((~df["is_abstention"]).sum()),
            abstained_count=int(df["is_abstention"].sum()),
        )

        metrics_summary = {
            "total_evaluated_trials": len(df),
            "subjects_evaluated": subject_folds,
            "neural_task_a_balanced_acc": neural_metrics["balanced_accuracy"],
            "kinematic_task_b_ridge_rmse_mm": ridge_rmse,
            "kinematic_task_b_ridge_mean_mm": ridge_mean,
            "kinematic_task_b_fair_cv_rmse_mm": cv_rmse,
            "kinematic_task_b_persistence_rmse_mm": pers_rmse,
            "condition_discrimination_balanced_acc": cond_metrics["balanced_accuracy"],
            "condition_discrimination_auroc": cond_metrics["auroc"],
            "condition_discrimination_sensitivity": cond_metrics["sensitivity"],
            "condition_discrimination_specificity": cond_metrics["specificity"],
            "coverage_rate": cov_metrics["coverage_rate"],
            "abstention_rate": cov_metrics["abstention_rate"],
            "is_authoritative_ground_truth": False,
            "evidence_level": "Level 2 (Existing-Data Exploratory)",
        }

        # Export CSV artifact
        out_csv = self.config.output_dir / "phase_5_evaluation_matrix.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        return df, metrics_summary
