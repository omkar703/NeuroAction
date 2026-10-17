"""Master execution driver for Phase 3: Continuous 3D Kinematics and Trajectory Forecasting."""

import sys
import time
from pathlib import Path

# Ensure root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np

from src.trajectory.segmentation import TrajectorySegmenter
from src.trajectory.evaluation import LOSOTrajectoryEvaluator
from src.trajectory.models import RidgeTrajectoryForecaster
from src.trajectory.inference import KinematicTwinForecaster
from src.visualization.trajectory_plots import generate_all_phase_3_figures
from src.trajectory.demo import run_demo


def run_phase_3_pipeline(
    output_dir: Path = Path("reports/phase_3"),
    model_dir: Path = Path("models/phase_3"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("NEUROACTION — PHASE 3 MASTER PIPELINE EXECUTION")
    print("Continuous 3D Kinematics Reconstruction, Trajectory Forecasting & Kinematic Digital Twin")
    print("=" * 75)

    t0 = time.time()

    # Step 1: Load and segment cohort for both modes
    print("\n[Step 1/6] Loading and Segmenting 3D Kinematics for Both Experimental Protocols...")
    segmenter = TrajectorySegmenter()
    
    # Protocol A: Event-Locked (reaction time interval)
    samples_a, qc_a = segmenter.load_cohort(mode="event_locked")
    print(f"  [Protocol A: Event-Locked] Segmented {len(samples_a)} valid samples across {len(qc_a['subjects_evaluated'])} subjects.")
    
    # Protocol B: Kinematic-Onset-Locked (physical movement initiation, speed >= 20 mm/s)
    samples_b, qc_b = segmenter.load_cohort(mode="kinematic_onset", speed_threshold_mm_s=20.0)
    latencies = [s.onset_latency_ms for s in samples_b]
    print(f"  [Protocol B: Kinematic-Onset] Segmented {len(samples_b)} valid samples across {len(qc_b['subjects_evaluated'])} subjects.")
    print(f"    Kinematic Onset Latency: Mean={np.mean(latencies):.1f} ms, Median={np.median(latencies):.1f} ms, Min={np.min(latencies):.1f} ms, Max={np.max(latencies):.1f} ms.")

    # Step 2: Run LOSO Evaluation for Protocol A (Event-Locked)
    print("\n[Step 2/6] Running LOSO Evaluation for Protocol A (Event-Locked)...")
    dir_a = output_dir / "event_locked"
    eval_a = LOSOTrajectoryEvaluator(samples_a)
    res_a = eval_a.evaluate_all_models(
        model_keys=["persistence", "constant_velocity", "ridge", "compact_mlp", "compact_gru"],
        output_dir=dir_a,
    )
    # Also save Protocol A artifacts at root output_dir for backward compatibility
    eval_a.evaluate_all_models(
        model_keys=["persistence", "constant_velocity", "ridge", "compact_mlp", "compact_gru"],
        output_dir=output_dir,
    )

    # Step 3: Run LOSO Evaluation for Protocol B (Kinematic-Onset-Locked)
    print("\n[Step 3/6] Running LOSO Evaluation for Protocol B (Kinematic-Onset-Locked)...")
    dir_b = output_dir / "kinematic_onset"
    eval_b = LOSOTrajectoryEvaluator(samples_b)
    res_b = eval_b.evaluate_all_models(
        model_keys=["persistence", "constant_velocity", "ridge", "compact_mlp", "compact_gru"],
        output_dir=dir_b,
    )

    # Step 4: Generate Protocol Comparison Table
    print("\n[Step 4/6] Exporting Protocol Comparison Table (Event-Locked vs Kinematic-Onset)...")
    comp_rows = []
    models_to_comp = ["PERSISTENCE", "CONSTANT_VELOCITY", "RIDGE", "COMPACT_MLP", "COMPACT_GRU"]
    for m in models_to_comp:
        row_a = res_a["df_summary"][res_a["df_summary"]["Model"] == m].iloc[0]
        row_b = res_b["df_summary"][res_b["df_summary"]["Model"] == m].iloc[0]
        comp_rows.append({
            "Model": m,
            "Event_Locked_Endpoint_Mean_mm": row_a["Mean Endpoint Error (mm)"],
            "Event_Locked_Endpoint_Median_mm": row_a["Median Error (mm)"],
            "Event_Locked_Win_vs_Persist_%": row_a["Win vs Persistence (%)"],
            "Kinematic_Onset_Endpoint_Mean_mm": row_b["Mean Endpoint Error (mm)"],
            "Kinematic_Onset_Endpoint_Median_mm": row_b["Median Error (mm)"],
            "Kinematic_Onset_Win_vs_Persist_%": row_b["Win vs Persistence (%)"],
        })
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = output_dir / "kinematic_comparison_event_vs_onset.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"Saved protocol comparison table to {comp_csv}")
    print(df_comp.to_string(index=False))

    # Step 5: Fit and Persist Production Models
    print(f"\n[Step 5/6] Fitting and Persisting Production Forecasters to {model_dir}...")
    # Train on all available samples for standalone production use
    prod_model = RidgeTrajectoryForecaster(alpha=10.0)
    prod_model.fit(samples_b)
    twin = KinematicTwinForecaster(prod_model)
    twin.save(model_dir, filename="kinematic_forecaster.pkl")

    # Step 6: Generate Publication Figures & Live Demonstration
    print(f"\n[Step 6/6] Generating Publication-Grade Figures & Real-Trial Demonstration in {fig_dir}...")
    sample_preds_path = output_dir / "trajectory_predictions" / "sample_forecast_predictions.json"
    sample_preds = []
    if sample_preds_path.exists():
        with open(sample_preds_path, "r") as f:
            sample_preds = json.load(f)

    generate_all_phase_3_figures(
        samples=samples_b,
        df_summary=res_b["df_summary"],
        df_horizons=res_b["df_horizon_summary"],
        sample_preds=sample_preds,
        output_dir=fig_dir,
    )

    # Real-Trial Live Demonstration on Held-Out Sub-01
    print("\nExecuting Real-Trial Live Demonstration on Held-Out Test Trial...")
    run_demo(
        trial_id=samples_b[0].trial_id,
        horizon_ms=500,
        model_dir=model_dir,
        save_plot_path=fig_dir / "figure_5_demo_forecast.png",
    )

    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"PHASE 3 MASTER PIPELINE EXECUTION COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min).")
    print(f"Artifacts exported to: {output_dir}")
    print(f"Production models saved to: {model_dir}")
    print("=" * 75)


if __name__ == "__main__":
    run_phase_3_pipeline()
