"""Master execution driver for Phase 4: Neural Intention–Action Digital Twin.

Executes:
1. Batch evaluation across all 269 valid trials across all 5 subjects under dual protocols.
2. Multimodal fusion and intention–action conflict analysis.
3. Conflict detection confusion matrix and statistical inference (Sensitivity, Specificity, AUROC).
4. Deterministic demo trial simulation and structured event logging.
5. Export of reproducibility manifests and publication figures.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import copy
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.conflict_engine import ConflictEngine
from src.phase_4.coordinate_mapping import (
    TARGET_COORDINATES_MM,
    estimate_fold_target_centroids,
)
from src.phase_4.dashboard import generate_dashboard_figure
from src.phase_4.event_logger import EventLogger
from src.phase_4.kinematic_adapter import KinematicTwinAdapter
from src.phase_4.neural_adapter import NeuralIntentionAdapter
from src.phase_4.replay_engine import ReplayEngine
from src.phase_4.trial_registry import TrialRegistry


def generate_architecture_figure(output_path: Path):
    """Generate publication Figure 9: Multimodal Architecture and Conflict Engine."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.95,
        "NEUROACTION PHASE 4: MULTIMODAL DIGITAL TWIN ARCHITECTURE",
        ha="center", va="top", fontsize=15, fontweight="bold", color="#38bdf8"
    )
    ax.text(
        0.5, 0.90,
        "Multimodal Synchronization, Trajectory Forecasting & Interpretable Conflict Detection",
        ha="center", va="top", fontsize=11, color="#94a3b8"
    )

    boxes = [
        # (x, y, w, h, title, subtitle, color)
        (0.05, 0.55, 0.25, 0.28, "Phase 2 Neural Decoder", "Scalp EEG [-1000, 0] ms\n26 Channels | Combined Feats\nOutput: Target Goal (Left/Right)\nPosterior Probabilities", "#3b82f6"),
        (0.38, 0.55, 0.25, 0.28, "Phase 3 Forecaster", "3D Leap Motion Kinematics\n200 ms Observed Prefix\nRidge Regressor (alpha=10.0)\nOutput: Endpoints & Horizons", "#ec4899"),
        (0.71, 0.55, 0.25, 0.28, "Coordinate Mapping", "Leap Motion Workspace (mm)\nTarget Left: [-188, 150, 53] mm\nTarget Right: [+187, 149, 51] mm\nDeterministic Alignment", "#10b981"),
        (0.20, 0.12, 0.60, 0.32, "Phase 4 Intention–Action Conflict Engine", "Metric A: Endpoint-to-Target Distance d_target = ||p_end - p_target||2\nMetric B: Directional Disagreement Angle θ = arccos(v_pred · v_target)\nMetric C: Neural Goal Agreement II(g_pred == g_true)\nMetric D: Retrospective Horizon Tracking RMSE\nDiscrete States: ALIGNED | POTENTIAL_DISAGREEMENT | UNCERTAIN", "#8b5cf6"),
    ]

    for x, y, w, h, title, sub, col in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#0f172a", edgecolor=col, linewidth=2, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=11, fontweight="bold", color=col, transform=ax.transAxes, zorder=3)
        ax.text(x + w / 2, y + 0.04, sub, ha="center", va="bottom", fontsize=8.5, color="#e2e8f0", family="monospace", transform=ax.transAxes, zorder=3)

    # Connecting arrows
    arrowprops = dict(arrowstyle="->", color="#f59e0b", lw=2.5)
    ax.annotate("", xy=(0.35, 0.44), xytext=(0.20, 0.55), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowprops)
    ax.annotate("", xy=(0.50, 0.44), xytext=(0.50, 0.55), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowprops)
    ax.annotate("", xy=(0.65, 0.44), xytext=(0.80, 0.55), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowprops)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 9 at {output_path}")


def generate_conflict_analysis_figure(eval_df: pd.DataFrame, output_path: Path):
    """Generate publication Figure 10: Retrospective Conflict Detection ROC & Trajectory Separation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Filter Protocol B rows
    df_b = eval_df[eval_df["protocol"] == "kinematic_onset"].copy()
    congruent = df_b[df_b["congruency"] == "congruent"]
    incongruent = df_b[df_b["congruency"] == "incongruent"]

    # --- Subplot 1: Endpoint-to-Target Distance Distribution ---
    ax1 = axes[0, 0]
    ax1.set_title("A. Target-Endpoint Distance ($d_{\\mathrm{target}}$)", color="#38bdf8", fontsize=11, fontweight="bold")
    ax1.hist(congruent["d_target_mm"], bins=20, alpha=0.7, color="#10b981", label=f"Congruent (N={len(congruent)})", density=True)
    ax1.hist(incongruent["d_target_mm"], bins=20, alpha=0.7, color="#ef4444", label=f"Incongruent (N={len(incongruent)})", density=True)
    ax1.axvline(50.0, color="#f59e0b", ls="--", lw=2, label="Threshold $d = 50$ mm")
    ax1.set_xlabel("Distance to Mapped Target (mm)", color="#94a3b8")
    ax1.set_ylabel("Density", color="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    # --- Subplot 2: Directional Disagreement Angle Distribution ---
    ax2 = axes[0, 1]
    ax2.set_title("B. Directional Disagreement Angle ($\\theta_{\\mathrm{dir}}$)", color="#38bdf8", fontsize=11, fontweight="bold")
    ax2.hist(congruent["directional_angle_deg"], bins=20, alpha=0.7, color="#10b981", label="Congruent (Aligned)", density=True)
    ax2.hist(incongruent["directional_angle_deg"], bins=20, alpha=0.7, color="#ef4444", label="Incongruent (Opposite)", density=True)
    ax2.axvline(45.0, color="#f59e0b", ls="--", lw=2, label="Threshold $\\theta = 45^\\circ$")
    ax2.set_xlabel("Angle Relative to Target Vector (degrees)", color="#94a3b8")
    ax2.set_ylabel("Density", color="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    # --- Subplot 3: ROC Curve for Conflict Classification ---
    ax3 = axes[1, 0]
    ax3.set_title("C. Conflict Detection ROC Curve", color="#38bdf8", fontsize=11, fontweight="bold")
    y_true = (df_b["congruency"] == "incongruent").astype(int).values
    y_scores = df_b["c_norm"].values
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)

    ax3.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"Conflict Detector (AUROC = {auc_val:.3f})")
    ax3.plot([0, 1], [0, 1], color="#64748b", ls="--", lw=1.5, label="Chance Baseline (0.500)")
    ax3.set_xlabel("False Positive Rate (1 - Specificity)", color="#94a3b8")
    ax3.set_ylabel("True Positive Rate (Sensitivity)", color="#94a3b8")
    ax3.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    # --- Subplot 4: Spatial Endpoint Clustering in Leap Space ---
    ax4 = axes[1, 1]
    ax4.set_title("D. Predicted Endpoint Separation in Workspace", color="#38bdf8", fontsize=11, fontweight="bold")
    ax4.scatter(congruent["pred_end_x"], congruent["pred_end_z"], color="#10b981", alpha=0.6, s=40, label="Congruent Reaches")
    ax4.scatter(incongruent["pred_end_x"], incongruent["pred_end_z"], color="#ef4444", alpha=0.6, s=40, label="Incongruent Reaches")
    ax4.scatter([-188.0], [53.0], color="#3b82f6", s=140, marker="s", edgecolors="white", label="Target: LEFT")
    ax4.scatter([187.5], [51.0], color="#a855f7", s=140, marker="s", edgecolors="white", label="Target: RIGHT")
    ax4.set_xlabel("X Lateral Position (mm)", color="#94a3b8")
    ax4.set_ylabel("Z Depth Position (mm)", color="#94a3b8")
    ax4.legend(loc="lower center", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 10 at {output_path}")


def run_phase_4_pipeline(config: Optional[DigitalTwinConfig] = None):
    """Execute complete Phase 4 batch evaluation and artifact generation."""
    t0 = time.time()
    cfg = config or DigitalTwinConfig()
    print("=" * 75)
    print("NEUROACTION — PHASE 4 MASTER PIPELINE EXECUTION")
    print("Multimodal Fusion, Trajectory Simulation, Conflict Detection & Digital Twin Replay")
    print("=" * 75)

    registry = TrialRegistry(manifest_path=cfg.manifest_path)
    valid_trials = registry.list_trials(valid_only=True)
    print(f"\n[Step 1/5] Loaded Trial Registry: {len(valid_trials)} valid trials across subjects.")

    # Precompute fold-specific target centroid proxies (strictly training subjects' congruent reaches)
    print("\n[Step 2/5] Estimating fold-specific target centroid proxies under LOSO...")
    samples, _ = KinematicTwinAdapter().segmenter.load_cohort(manifest_path=cfg.manifest_path, mode="kinematic_onset")
    subjects = sorted(list(set(s.participant_id for s in samples)))
    fold_centroids = {}
    for held_out_sub in subjects:
        train_samples = [s for s in samples if s.participant_id != held_out_sub]
        fold_centroids[held_out_sub] = estimate_fold_target_centroids(train_samples)
        c_l = fold_centroids[held_out_sub]["left"]
        c_r = fold_centroids[held_out_sub]["right"]
        print(f"  Held-out {held_out_sub}: Left proxy = [{c_l[0]:.1f}, {c_l[1]:.1f}, {c_l[2]:.1f}] mm | "
              f"Right proxy = [{c_r[0]:.1f}, {c_r[1]:.1f}, {c_r[2]:.1f}] mm")

    conflict_engine = ConflictEngine(config=cfg)

    # 2. Batch Evaluation across all valid trials using strict held-out models
    print("\n[Step 3/5] Running Strict LOSO Multimodal Evaluation across all 269 trials...")
    records = []
    protocols = ["kinematic_onset", "event_locked"]

    for trial_id in valid_trials:
        trial = registry.get_trial(trial_id)
        if trial is None:
            continue
        meta = registry.get_metadata(trial_id)
        sub = trial.participant_id

        # Strictly Held-Out Model Adapters for this participant
        n_adapter = NeuralIntentionAdapter.load_held_out(sub, model_dir=cfg.model_dir_phase_2)
        k_adapter = KinematicTwinAdapter.load_held_out(sub, model_dir=cfg.model_dir_phase_3)

        # Neural intention inference (LOSO held-out)
        n_res = n_adapter.predict_trial(trial=trial, current_time_ms=0.0)

        # Fold-specific target proxy (strictly from other 4 subjects)
        pred_goal_key = n_res.predicted_goal if n_res.predicted_goal in ("left", "right") else "left"
        target_proxy = fold_centroids[sub][pred_goal_key].copy()

        # Update metadata target coordinates with fold-specific proxy
        meta_fold = copy.deepcopy(meta)
        meta_fold.target_coordinates = target_proxy

        for proto in protocols:
            k_res = k_adapter.forecast_trial(
                trial=trial,
                protocol=proto,
                speed_threshold_mm_s=cfg.speed_threshold_mm_s,
                horizons_ms=cfg.horizons_ms,
            )

            c_res = conflict_engine.evaluate_conflict(
                neural_result=n_res,
                forecast_result=k_res,
                trial_metadata=meta_fold,
                replay_time_ms=k_res.forecast_generation_time_ms,
            )

            p_end = k_res.predicted_endpoint
            a_end = k_res.actual_endpoint
            end_err = float(np.linalg.norm(p_end - a_end)) if a_end is not None else np.nan

            records.append({
                "participant_id": trial.participant_id,
                "run_id": trial.run_id,
                "trial_id": trial.trial_id,
                "congruency": trial.congruency,
                "target_goal": trial.target_goal,
                "physical_hand_direction": trial.physical_hand_direction,
                "ball_color": trial.ball_color,
                "protocol": proto,
                "is_held_out": True,
                "neural_model": n_res.model_name,
                "neural_model_provenance": f"lda_held_out_{sub.replace('-', '')}",
                "neural_goal": n_res.predicted_goal,
                "neural_conf": n_res.confidence,
                "neural_prob_left": n_res.probability_left,
                "neural_prob_right": n_res.probability_right,
                "neural_goal_match": bool(n_res.predicted_goal == trial.target_goal),
                "forecast_model": k_res.model_name,
                "kinematic_model_provenance": f"ridge_held_out_{sub.replace('-', '')}",
                "target_proxy_provenance": f"training_subs_excluding_{sub}",
                "forecast_gen_ms": k_res.forecast_generation_time_ms,
                "pred_end_x": float(p_end[0]),
                "pred_end_y": float(p_end[1]),
                "pred_end_z": float(p_end[2]),
                "actual_end_x": float(a_end[0]) if a_end is not None else np.nan,
                "actual_end_y": float(a_end[1]) if a_end is not None else np.nan,
                "actual_end_z": float(a_end[2]) if a_end is not None else np.nan,
                "endpoint_error_mm": end_err,
                "conflict_state": c_res.conflict_state,
                "is_conflict": c_res.is_conflict,
                "d_target_mm": c_res.endpoint_target_distance_mm,
                "c_norm": c_res.normalized_distance_score,
                "directional_angle_deg": c_res.directional_disagreement_deg,
                "neural_goal_agreement": c_res.neural_goal_agreement,
                "motor_forecast_agreement": c_res.motor_forecast_agreement,
            })

    eval_df = pd.DataFrame(records)
    # Save both audited_evaluation_matrix.csv and digital_twin_eval_matrix.csv
    audited_csv = cfg.output_dir / "audited_evaluation_matrix.csv"
    eval_csv = cfg.output_dir / "digital_twin_eval_matrix.csv"
    eval_df.to_csv(audited_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    print(f"Saved Audited Evaluation Matrix ({len(eval_df)} rows) to {audited_csv}")

    # 3. Conflict Detection Performance Analysis (Protocol B)
    print("\n[Step 4/5] Computing Audited Conflict Detection Metrics under LOSO...")
    df_proto_b = eval_df[eval_df["protocol"] == "kinematic_onset"].copy()
    y_true = (df_proto_b["congruency"] == "incongruent").astype(int).values
    y_pred = df_proto_b["is_conflict"].astype(int).values
    y_scores = df_proto_b["c_norm"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = recall_score(y_true, y_pred)
    spec = float(tn / (tn + fp))
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)

    # Upstream component metrics under LOSO
    y_true_goal = (df_proto_b["target_goal"] == "right").astype(int).values
    y_pred_goal = (df_proto_b["neural_goal"] == "right").astype(int).values
    neural_goal_bal_acc = balanced_accuracy_score(y_true_goal, y_pred_goal)
    kinematic_rmse = float(np.sqrt(np.mean(df_proto_b["endpoint_error_mm"] ** 2)))

    conf_matrix_df = pd.DataFrame([
        {"Metric": "True Positives (Incongruent Detected Conflict)", "Value": int(tp)},
        {"Metric": "False Positives (Congruent Flagged Conflict)", "Value": int(fp)},
        {"Metric": "True Negatives (Congruent Detected Aligned)", "Value": int(tn)},
        {"Metric": "False Negatives (Incongruent Missed Conflict)", "Value": int(fn)},
        {"Metric": "Sensitivity / Recall", "Value": round(sens, 4)},
        {"Metric": "Specificity", "Value": round(spec, 4)},
        {"Metric": "Precision", "Value": round(prec, 4)},
        {"Metric": "F1-Score", "Value": round(f1, 4)},
        {"Metric": "Balanced Accuracy", "Value": round(bal_acc, 4)},
        {"Metric": "Conflict Score AUROC", "Value": round(auroc, 4)},
        {"Metric": "Phase 2 Neural Goal Balanced Accuracy (LOSO)", "Value": round(neural_goal_bal_acc, 4)},
        {"Metric": "Phase 3 Kinematic Endpoint RMSE (LOSO mm)", "Value": round(kinematic_rmse, 2)},
    ])
    conf_csv = cfg.output_dir / "conflict_confusion_matrix.csv"
    conf_matrix_df.to_csv(conf_csv, index=False)
    print("\nAudited Statistical Summary:")
    print(conf_matrix_df.to_string(index=False))

    # Threshold Sensitivity Analysis
    sens_rows = []
    for d_th in [25.0, 50.0, 75.0, 100.0]:
        for th_deg in [30.0, 45.0, 60.0]:
            pred_flags = (
                (df_proto_b["d_target_mm"] > d_th) | (df_proto_b["directional_angle_deg"] > th_deg)
            ).astype(int)
            b_acc = balanced_accuracy_score(y_true, pred_flags)
            f_score = f1_score(y_true, pred_flags)
            sens_rows.append({
                "Distance Threshold (mm)": d_th,
                "Angle Threshold (deg)": th_deg,
                "Balanced Accuracy": round(b_acc, 4),
                "F1-Score": round(f_score, 4),
            })
    sens_df = pd.DataFrame(sens_rows)
    sens_csv = cfg.output_dir / "threshold_sensitivity_analysis.csv"
    sens_df.to_csv(sens_csv, index=False)
    print(f"\nSaved Threshold Sensitivity Analysis to {sens_csv}")

    # 4. Generate Figures & Demo Artifacts
    print("\n[Step 5/5] Generating Publication Figures & Demo Artifacts...")
    fig9_path = cfg.figures_dir / "figure_9_multimodal_architecture.png"
    generate_architecture_figure(fig9_path)

    fig10_path = cfg.figures_dir / "figure_10_conflict_detection_analysis.png"
    generate_conflict_analysis_figure(eval_df, fig10_path)

    fig11_path = cfg.figures_dir / "figure_11_demo_trial_replay.png"
    generate_dashboard_figure(trial_id=cfg.demo_trial_id, output_path=fig11_path, config=cfg)
    print(f"Generated Figure 11 at {fig11_path}")

    # Replay demo trial and export logs
    replay_engine = ReplayEngine(config=cfg, registry=registry)
    demo_frames = list(replay_engine.stream_trial(trial_id=cfg.demo_trial_id, protocol="kinematic_onset"))
    logger = EventLogger(output_dir=cfg.demo_dir)
    logger.export_event_log_csv(demo_frames)
    logger.export_trial_summary_json(demo_frames[-1])
    logger.export_reproducibility_manifest(cfg)
    print(f"Exported demo artifacts in {cfg.demo_dir}")

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"PHASE 4 MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    run_phase_4_pipeline()
