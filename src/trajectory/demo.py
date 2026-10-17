"""Interactive / CLI demonstration script for real-trial trajectory forecasting."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.trajectory.segmentation import TrajectorySegmenter
from src.trajectory.baselines import PersistenceBaseline, ConstantVelocityBaseline
from src.trajectory.inference import KinematicTwinForecaster
from src.visualization.trajectory_plots import setup_plotting_style


def run_demo(
    trial_id: str = "sub-01_run-01_L1_tr00",
    horizon_ms: int = 500,
    model_dir: Path = Path("models/phase_3"),
    save_plot_path: Optional[Path] = Path("reports/phase_3/figures/demo_forecast_output.png"),
):
    """Load an actual recorded trial, run baselines and the saved kinematic twin, and print/plot results."""
    print("=" * 70)
    print("NEUROACTION — PHASE 3 KINEMATIC DIGITAL TWIN LIVE DEMONSTRATION")
    print("=" * 70)

    # 1. Load actual trial data
    print(f"\n[1/4] Loading real recorded test trial: {trial_id}...")
    segmenter = TrajectorySegmenter()
    samples, _ = segmenter.load_cohort()
    target_sample = next((s for s in samples if s.trial_id == trial_id), None)
    if target_sample is None:
        print(f"Trial {trial_id} not found among {len(samples)} valid segmented trials. Using first trial: {samples[0].trial_id}")
        target_sample = samples[0]

    print(f"  Participant: {target_sample.participant_id} | Condition: {target_sample.congruency}")
    print(f"  Physical Hand: {target_sample.physical_hand_direction.upper()} | Target Goal: {target_sample.target_goal.upper()}")
    print(f"  Observed 200 ms Prefix: 20 samples @ 100 Hz, start pos: {target_sample.prefix_positions[0]}")

    # 2. Load trained KinematicTwinForecaster
    held_out_filename = f"kinematic_forecaster_held_out_{target_sample.participant_id.replace('-', '')}.pkl"
    held_out_path = model_dir / held_out_filename
    if held_out_path.exists():
        print(f"\n[2/4] Loading HELD-OUT Kinematic Twin Forecaster ({held_out_filename}) from {model_dir}...")
        print(f"      (Model was trained strictly WITHOUT participant {target_sample.participant_id} — honest held-out out-of-fold demo)")
        forecaster = KinematicTwinForecaster.load(model_dir, filename=held_out_filename)
    else:
        print(f"\n[2/4] Loading trained Kinematic Twin Forecaster (production cohort model) from {model_dir}...")
        forecaster = KinematicTwinForecaster.load(model_dir)

    # Run Baselines
    persistence = PersistenceBaseline()
    const_vel = ConstantVelocityBaseline()
    const_vel.fit([s for s in samples if s.participant_id != target_sample.participant_id])

    # 3. Execute Predictions
    print(f"\n[3/4] Running inference for Horizon={horizon_ms} ms and Endpoint...")
    pred_persist_end = persistence.predict_endpoint(target_sample)
    pred_persist_h = persistence.predict_horizon(target_sample, horizon_ms)

    pred_cv_end = const_vel.predict_endpoint(target_sample)
    pred_cv_h = const_vel.predict_horizon(target_sample, horizon_ms)

    result = forecaster.forecast(
        observed_trajectory=target_sample.prefix_positions,
        sampling_rate_hz=target_sample.sampling_rate,
        forecast_horizon_ms=horizon_ms,
        trial_id=target_sample.trial_id,
    )
    pred_twin_end = result.predicted_endpoint
    pred_twin_h = result.predicted_positions
    actual_end = target_sample.actual_endpoint

    # Errors
    err_persist_end = np.linalg.norm(pred_persist_end - actual_end)
    err_cv_end = np.linalg.norm(pred_cv_end - actual_end)
    err_twin_end = np.linalg.norm(pred_twin_end - actual_end)

    print("\n--- Quantitative Prediction Summary ---")
    print(f"Actual Endpoint Position:    {actual_end} mm")
    print(f"Persistence Predicted End:   {pred_persist_end} mm (Error: {err_persist_end:.2f} mm)")
    print(f"Const-Vel Predicted End:     {pred_cv_end} mm (Error: {err_cv_end:.2f} mm)")
    print(f"Kinematic Twin Pred End:     {pred_twin_end} mm (Error: {err_twin_end:.2f} mm)")

    if horizon_ms in target_sample.horizon_trajectories:
        actual_h = target_sample.horizon_trajectories[horizon_ms]
        err_h_persist = np.mean(np.linalg.norm(pred_persist_h - actual_h, axis=1))
        err_h_cv = np.mean(np.linalg.norm(pred_cv_h - actual_h, axis=1))
        err_h_twin = np.mean(np.linalg.norm(pred_twin_h - actual_h, axis=1))
        print(f"\nHorizon +{horizon_ms} ms Trajectory Mean Error:")
        print(f"  Persistence:     {err_h_persist:.2f} mm")
        print(f"  Const Velocity:  {err_h_cv:.2f} mm")
        print(f"  Kinematic Twin:  {err_h_twin:.2f} mm (Improvement: {err_h_cv - err_h_twin:+.2f} mm)")

    # 4. Generate Visual Demonstration Plot
    print(f"\n[4/4] Rendering trajectory forecast comparison plot...")
    setup_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left: Lateral X(t)
    ax = axes[0]
    t_obs = target_sample.prefix_times
    t_fut = t_obs[-1] + np.arange(1, len(pred_twin_h) + 1) * 10.0

    ax.plot(t_obs, target_sample.prefix_positions[:, 0], color="black", lw=3, label="Observed Prefix (0-200 ms)")
    if horizon_ms in target_sample.horizon_trajectories:
        ax.plot(t_fut, target_sample.horizon_trajectories[horizon_ms][:, 0], color="#2ca02c", lw=2.5, label="Actual Future Path")
    ax.plot(t_fut, pred_persist_h[:, 0], color="#7f7f7f", linestyle=":", lw=1.8, label="Persistence")
    ax.plot(t_fut, pred_cv_h[:, 0], color="#e377c2", linestyle="-.", lw=1.8, label="Constant Velocity")
    ax.plot(t_fut, pred_twin_h[:, 0], color="#1f77b4", linestyle="--", lw=2.5, label="Kinematic Twin (Learned)")
    ax.axvline(t_obs[-1], color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Lateral Position X (mm)")
    ax.set_title("A. Lateral Trajectory Forecast Ahead of Prefix", fontweight="bold")
    ax.legend(fontsize=8, loc="best")

    # Right: Spatial Endpoint Error Bar Chart
    ax2 = axes[1]
    models = ["Persistence", "Const-Velocity", "Kinematic Twin"]
    errors = [err_persist_end, err_cv_end, err_twin_end]
    colors = ["#7f7f7f", "#e377c2", "#1f77b4"]
    bars = ax2.bar(models, errors, color=colors, width=0.5, alpha=0.85)
    ax2.set_ylabel("Endpoint Error (mm)")
    ax2.set_title("B. Final Hand Endpoint Error (mm)", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f} mm", ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"Demonstration: Forecast on Test Trial {target_sample.trial_id} ({target_sample.physical_hand_direction})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_plot_path:
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, dpi=180)
        print(f"Saved demo figure to {save_plot_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("DEMONSTRATION RUN COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kinematic Twin trajectory forecast demo on a real test trial.")
    parser.add_argument("--trial", type=str, default="sub-01_run-01_L1_tr00", help="Trial ID to run demo on")
    parser.add_argument("--horizon", type=int, default=500, help="Forecast horizon in ms (100, 200, 300, 500)")
    args = parser.parse_args()

    run_demo(trial_id=args.trial, horizon_ms=args.horizon)
