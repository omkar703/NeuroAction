"""Evaluation engine for continuous kinematic trajectory forecasting under LOSO."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.trajectory.schema import TrajectorySample, ForecastResult
from src.trajectory.baselines import PersistenceBaseline, ConstantVelocityBaseline
from src.trajectory.models import RidgeTrajectoryForecaster, MLPTrajectoryForecaster, GRUTrajectoryForecaster


class LOSOTrajectoryEvaluator:
    """Evaluates kinematic trajectory models under strict Leave-One-Subject-Out cross-validation."""

    HORIZONS_MS = [100, 200, 300, 500]

    def __init__(self, samples: List[TrajectorySample]):
        self.samples = samples
        self.subjects = sorted(list(set(s.participant_id for s in self.samples)))

    def _get_model(self, model_key: str, fold_idx: int):
        if model_key == "persistence":
            return PersistenceBaseline()
        elif model_key == "constant_velocity":
            return ConstantVelocityBaseline()
        elif model_key == "ridge":
            return RidgeTrajectoryForecaster(alpha=10.0)
        elif model_key == "compact_mlp":
            return MLPTrajectoryForecaster(epochs=50, lr=0.005, random_state=42 + fold_idx)
        elif model_key == "compact_gru":
            return GRUTrajectoryForecaster(epochs=50, lr=0.005, random_state=42 + fold_idx)
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

    def evaluate_all_models(
        self,
        model_keys: Optional[List[str]] = None,
        output_dir: Path = Path("reports/phase_3"),
    ) -> Dict:
        """Run full LOSO cross-validation across all models and export verified CSV artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = output_dir / "trajectory_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        keys_to_run = model_keys or ["persistence", "constant_velocity", "ridge", "compact_mlp", "compact_gru"]

        endpoint_records: List[Dict] = []
        horizon_records: List[Dict] = []
        sample_predictions: List[Dict] = []

        print(f"Beginning LOSO trajectory evaluation across {len(self.subjects)} subjects: {self.subjects}")

        # Store trial-level errors per model for win-rate comparisons
        trial_errors: Dict[str, Dict[str, float]] = {m: {} for m in keys_to_run}
        horizon_point_errors: Dict[str, Dict[int, Dict[str, float]]] = {
            m: {h: {} for h in self.HORIZONS_MS} for m in keys_to_run
        }

        for fold_idx, test_sub in enumerate(self.subjects):
            train_samples = [s for s in self.samples if s.participant_id != test_sub]
            test_samples = [s for s in self.samples if s.participant_id == test_sub]
            print(f"  [Fold {fold_idx + 1}/{len(self.subjects)}] Test Subject: {test_sub} (Train: {len(train_samples)}, Test: {len(test_samples)})")

            for m_key in keys_to_run:
                model = self._get_model(m_key, fold_idx)
                model.fit(train_samples)

                # Persist fold model for held-out validation demonstration
                if m_key == "ridge" and test_sub == "sub-01":
                    held_out_dir = Path("models/phase_3")
                    held_out_dir.mkdir(parents=True, exist_ok=True)
                    import pickle
                    with open(held_out_dir / "kinematic_forecaster_held_out_sub01.pkl", "wb") as f:
                        pickle.dump(model, f)

                for sample in test_samples:
                    # 1. Endpoint prediction
                    pred_end = model.predict_endpoint(sample)
                    actual_end = sample.actual_endpoint
                    diff_end = pred_end - actual_end
                    euc_end = float(np.linalg.norm(diff_end))

                    trial_errors[m_key][sample.trial_id] = euc_end

                    endpoint_records.append({
                        "participant_id": sample.participant_id,
                        "run_id": sample.run_id,
                        "trial_id": sample.trial_id,
                        "model_name": m_key,
                        "fold_id": test_sub,
                        "congruency": sample.congruency,
                        "target_goal": sample.target_goal,
                        "physical_hand_direction": sample.physical_hand_direction,
                        "observation_duration_ms": 200,
                        "endpoint_error_mm": euc_end,
                        "endpoint_error_x_mm": float(abs(diff_end[0])),
                        "endpoint_error_y_mm": float(abs(diff_end[1])),
                        "endpoint_error_z_mm": float(abs(diff_end[2])),
                        "predicted_end_x": float(pred_end[0]),
                        "predicted_end_y": float(pred_end[1]),
                        "predicted_end_z": float(pred_end[2]),
                        "actual_end_x": float(actual_end[0]),
                        "actual_end_y": float(actual_end[1]),
                        "actual_end_z": float(actual_end[2]),
                    })

                    # 2. Horizon trajectory forecasting
                    for h in self.HORIZONS_MS:
                        if h not in sample.horizon_trajectories:
                            continue
                        pred_traj = model.predict_horizon(sample, h)
                        actual_traj = sample.horizon_trajectories[h]
                        pred_pt = model.predict_horizon_point(sample, h)
                        actual_pt = sample.horizon_points[h]

                        pt_diff = pred_pt - actual_pt
                        pt_euc = float(np.linalg.norm(pt_diff))
                        horizon_point_errors[m_key][h][sample.trial_id] = pt_euc

                        # Multi-step trajectory error over window
                        traj_diffs = pred_traj - actual_traj
                        step_eucs = np.linalg.norm(traj_diffs, axis=1)
                        mean_traj_euc = float(np.mean(step_eucs))
                        # True 3D Euclidean RMSE over trajectory steps (ensures RMSE >= Mean)
                        rmse_traj = float(np.sqrt(np.mean(step_eucs ** 2)))

                        horizon_records.append({
                            "participant_id": sample.participant_id,
                            "run_id": sample.run_id,
                            "trial_id": sample.trial_id,
                            "model_name": m_key,
                            "fold_id": test_sub,
                            "congruency": sample.congruency,
                            "horizon_ms": h,
                            "horizon_point_error_mm": pt_euc,
                            "horizon_point_error_x_mm": float(abs(pt_diff[0])),
                            "horizon_point_error_y_mm": float(abs(pt_diff[1])),
                            "horizon_point_error_z_mm": float(abs(pt_diff[2])),
                            "trajectory_mean_error_mm": mean_traj_euc,
                            "trajectory_rmse_mm": rmse_traj,
                        })

                    # Store selected trajectory predictions for visualization demo
                    if fold_idx == 0 and len(sample_predictions) < 5 and m_key in ("persistence", "constant_velocity", "ridge"):
                        sample_predictions.append({
                            "trial_id": sample.trial_id,
                            "participant_id": sample.participant_id,
                            "model_name": m_key,
                            "prefix_positions": sample.prefix_positions.tolist(),
                            "predicted_endpoint": pred_end.tolist(),
                            "actual_endpoint": actual_end.tolist(),
                            "predicted_horizon_500": model.predict_horizon(sample, 500).tolist(),
                            "actual_horizon_500": sample.horizon_trajectories[500].tolist(),
                        })

        df_endpoints = pd.DataFrame(endpoint_records)
        df_horizons = pd.DataFrame(horizon_records)

        # Save trial-level endpoint predictions
        ep_csv = output_dir / "endpoint_predictions.csv"
        df_endpoints.to_csv(ep_csv, index=False)
        print(f"Saved {len(df_endpoints)} endpoint predictions to {ep_csv}")

        # Compute Participant-Level Metrics
        part_metrics_rows = []
        for m_key in keys_to_run:
            m_sub_df = df_endpoints[df_endpoints["model_name"] == m_key]
            for sub in self.subjects:
                s_df = m_sub_df[m_sub_df["participant_id"] == sub]
                errs = s_df["endpoint_error_mm"].values
                part_metrics_rows.append({
                    "model_name": m_key,
                    "participant_id": sub,
                    "n_trials": len(s_df),
                    "mean_endpoint_error_mm": float(np.mean(errs)),
                    "rmse_endpoint_error_mm": float(np.sqrt(np.mean(errs ** 2))),
                    "median_endpoint_error_mm": float(np.median(errs)),
                    "p90_endpoint_error_mm": float(np.percentile(errs, 90)),
                    "mae_x_mm": float(np.mean(s_df["endpoint_error_x_mm"])),
                    "mae_y_mm": float(np.mean(s_df["endpoint_error_y_mm"])),
                    "mae_z_mm": float(np.mean(s_df["endpoint_error_z_mm"])),
                    "rmse_x_mm": float(np.sqrt(np.mean(s_df["endpoint_error_x_mm"] ** 2))),
                    "rmse_y_mm": float(np.sqrt(np.mean(s_df["endpoint_error_y_mm"] ** 2))),
                    "rmse_z_mm": float(np.sqrt(np.mean(s_df["endpoint_error_z_mm"] ** 2))),
                })

        df_part_metrics = pd.DataFrame(part_metrics_rows)
        part_csv = output_dir / "participant_metrics.csv"
        df_part_metrics.to_csv(part_csv, index=False)
        print(f"Saved participant metrics to {part_csv}")

        # Compute Summary Trajectory & Endpoint Metrics (Participant Macro-Average)
        summary_rows = []
        t_crit = stats.t.ppf(0.975, df=len(self.subjects) - 1)  # 2.776 for df=4

        for m_key in keys_to_run:
            s_data = df_part_metrics[df_part_metrics["model_name"] == m_key]
            sub_means = s_data["mean_endpoint_error_mm"].values
            sub_rmses = s_data["rmse_endpoint_error_mm"].values
            sub_medians = s_data["median_endpoint_error_mm"].values
            sub_p90s = s_data["p90_endpoint_error_mm"].values

            macro_mean = float(np.mean(sub_means))
            macro_rmse = float(np.mean(sub_rmses))
            macro_sem = float(stats.sem(sub_means))
            macro_median = float(np.mean(sub_medians))
            macro_p90 = float(np.mean(sub_p90s))

            # Win rates against baselines
            win_vs_persist = 0.0
            win_vs_cv = 0.0
            if m_key not in ("persistence",):
                wins_p = [
                    1.0 if trial_errors[m_key][t_id] < trial_errors["persistence"][t_id] else 0.0
                    for t_id in trial_errors[m_key]
                ]
                win_vs_persist = float(np.mean(wins_p) * 100.0)
            if m_key not in ("constant_velocity",):
                wins_c = [
                    1.0 if trial_errors[m_key][t_id] < trial_errors["constant_velocity"][t_id] else 0.0
                    for t_id in trial_errors[m_key]
                ]
                win_vs_cv = float(np.mean(wins_c) * 100.0)

            ci_low = max(0.0, macro_mean - t_crit * macro_sem)
            ci_high = macro_mean + t_crit * macro_sem

            summary_rows.append({
                "Model": m_key.upper(),
                "Mean Endpoint Error (mm)": macro_mean,
                "RMSE (mm)": macro_rmse,
                "SEM (mm)": macro_sem,
                "Student-t 95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
                "Median Error (mm)": macro_median,
                "90th Percentile (mm)": macro_p90,
                "MAE X (mm)": float(np.mean(s_data["mae_x_mm"])),
                "MAE Y (mm)": float(np.mean(s_data["mae_y_mm"])),
                "MAE Z (mm)": float(np.mean(s_data["mae_z_mm"])),
                "RMSE X (mm)": float(np.mean(s_data["rmse_x_mm"])),
                "RMSE Y (mm)": float(np.mean(s_data["rmse_y_mm"])),
                "RMSE Z (mm)": float(np.mean(s_data["rmse_z_mm"])),
                "Win vs Persistence (%)": win_vs_persist,
                "Win vs ConstVel (%)": win_vs_cv,
            })

        df_summary = pd.DataFrame(summary_rows)
        sum_csv = output_dir / "trajectory_metrics.csv"
        df_summary.to_csv(sum_csv, index=False)
        print(f"Saved overall trajectory metrics to {sum_csv}")

        # Compute Forecast Metrics by Horizon
        horizon_rows = []
        for h in self.HORIZONS_MS:
            h_df = df_horizons[df_horizons["horizon_ms"] == h]
            for m_key in keys_to_run:
                m_h_df = h_df[h_df["model_name"] == m_key]
                
                # Participant level means
                sub_means_h = []
                sub_medians_h = []
                sub_rmses_h = []
                for sub in self.subjects:
                    sub_errs = m_h_df[m_h_df["participant_id"] == sub]["horizon_point_error_mm"].values
                    sub_means_h.append(np.mean(sub_errs))
                    sub_medians_h.append(np.median(sub_errs))
                    sub_rmses_h.append(np.sqrt(np.mean(sub_errs ** 2)))

                mean_pt = float(np.mean(sub_means_h))
                sem_pt = float(stats.sem(sub_means_h))
                median_pt = float(np.mean(sub_medians_h))
                rmse_pt = float(np.mean(sub_rmses_h))

                ci_low_h = max(0.0, mean_pt - t_crit * sem_pt)
                ci_high_h = mean_pt + t_crit * sem_pt

                # Win rate at this horizon
                win_rate_pers = 0.0
                win_rate_cv = 0.0
                if m_key != "persistence":
                    w_p = [
                        1.0 if horizon_point_errors[m_key][h][t_id] < horizon_point_errors["persistence"][h][t_id] else 0.0
                        for t_id in horizon_point_errors[m_key][h]
                    ]
                    win_rate_pers = float(np.mean(w_p) * 100.0)
                if m_key != "constant_velocity":
                    w_c = [
                        1.0 if horizon_point_errors[m_key][h][t_id] < horizon_point_errors["constant_velocity"][h][t_id] else 0.0
                        for t_id in horizon_point_errors[m_key][h]
                    ]
                    win_rate_cv = float(np.mean(w_c) * 100.0)

                horizon_rows.append({
                    "Horizon (ms)": h,
                    "Model": m_key.upper(),
                    "Mean Error (mm)": mean_pt,
                    "Point Target RMSE (mm)": rmse_pt,
                    "SEM (mm)": sem_pt,
                    "Student-t 95% CI": f"[{ci_low_h:.2f}, {ci_high_h:.2f}]",
                    "Median Error (mm)": median_pt,
                    "Trajectory Mean Error (mm)": float(m_h_df["trajectory_mean_error_mm"].mean()),
                    "Trajectory RMSE (mm)": float(m_h_df["trajectory_rmse_mm"].mean()),
                    "Win vs Persistence (%)": win_rate_pers,
                    "Win vs ConstVel (%)": win_rate_cv,
                })

        df_horizon_summary = pd.DataFrame(horizon_rows)
        horiz_csv = output_dir / "forecast_metrics_by_horizon.csv"
        df_horizon_summary.to_csv(horiz_csv, index=False)
        print(f"Saved horizon forecast metrics to {horiz_csv}")

        # Save trajectory prediction examples for demo / inspection
        sample_pred_file = pred_dir / "sample_forecast_predictions.json"
        with open(sample_pred_file, "w") as f:
            json.dump(sample_predictions, f, indent=2)
        print(f"Saved demo predictions to {sample_pred_file}")

        return {
            "df_summary": df_summary,
            "df_part_metrics": df_part_metrics,
            "df_endpoints": df_endpoints,
            "df_horizon_summary": df_horizon_summary,
        }
