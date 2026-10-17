"""Scientific visualization and figure generation engine for Phase 3 Kinematics."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from src.trajectory.schema import TrajectorySample


def setup_plotting_style():
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
        "lines.linewidth": 1.75,
        "grid.alpha": 0.5,
    })


def plot_figure_1_coordinate_system(samples: List[TrajectorySample], output_dir: Path):
    """Figure 1: 3D Hand Trajectory Profiles and Visuomotor Coordinate Workspace."""
    setup_plotting_style()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    left_samples = [s for s in samples if s.physical_hand_direction == "left"][:12]
    right_samples = [s for s in samples if s.physical_hand_direction == "right"][:12]

    for s in left_samples:
        traj = s.full_trajectory
        ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color="#1f77b4", alpha=0.6, lw=1.5)
        ax.scatter(traj[-1, 0], traj[-1, 2], traj[-1, 1], color="#1f77b4", s=25, alpha=0.8)

    for s in right_samples:
        traj = s.full_trajectory
        ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color="#d62728", alpha=0.6, lw=1.5)
        ax.scatter(traj[-1, 0], traj[-1, 2], traj[-1, 1], color="#d62728", s=25, alpha=0.8)

    # Origin / start marker
    ax.scatter([0], [50], [150], color="black", s=70, marker="^", label="Nominal Start Area (x=0)")
    ax.plot([], [], color="#1f77b4", label="Leftward Reach Trajectory (Hand Left)")
    ax.plot([], [], color="#d62728", label="Rightward Reach Trajectory (Hand Right)")

    ax.set_xlabel("Lateral X (mm, Left < 0 < Right)", labelpad=10)
    ax.set_ylabel("Depth Z (mm)", labelpad=10)
    ax.set_zlabel("Height Y (mm)", labelpad=10)
    ax.set_title("Figure 1: 3D Reaching Kinematics Workspace (IACKD Leap Motion Tracking)", fontweight="bold")
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "figure_1_coordinate_workspace.png", dpi=200)
    plt.close()


def plot_figure_2_kinematic_dynamics(samples: List[TrajectorySample], output_dir: Path):
    """Figure 2: Velocity and Acceleration Dynamics during the Observed 200 ms Prefix."""
    setup_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    t_ms = np.arange(20) * 10.0  # 0 to 190 ms

    left_samples = [s for s in samples if s.physical_hand_direction == "left"]
    right_samples = [s for s in samples if s.physical_hand_direction == "right"]

    left_vx = np.array([s.prefix_velocities[:, 0] for s in left_samples])
    right_vx = np.array([s.prefix_velocities[:, 0] for s in right_samples])

    left_disp_x = np.array([s.prefix_positions[:, 0] - s.prefix_positions[0, 0] for s in left_samples])
    right_disp_x = np.array([s.prefix_positions[:, 0] - s.prefix_positions[0, 0] for s in right_samples])

    # Subplot A: X Displacement
    ax = axes[0, 0]
    ax.plot(t_ms, np.mean(left_disp_x, axis=0), color="#1f77b4", label="Left Reach Mean", lw=2)
    ax.fill_between(t_ms, np.percentile(left_disp_x, 25, axis=0), np.percentile(left_disp_x, 75, axis=0), color="#1f77b4", alpha=0.2)
    ax.plot(t_ms, np.mean(right_disp_x, axis=0), color="#d62728", label="Right Reach Mean", lw=2)
    ax.fill_between(t_ms, np.percentile(right_disp_x, 25, axis=0), np.percentile(right_disp_x, 75, axis=0), color="#d62728", alpha=0.2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Lateral Displacement $\Delta x$ (mm)")
    ax.set_title("A. Early Lateral Displacement Profile [0–200 ms]", fontweight="bold")
    ax.legend(loc="upper left")

    # Subplot B: X Velocity
    ax = axes[0, 1]
    ax.plot(t_ms, np.mean(left_vx, axis=0), color="#1f77b4", label="Left Velocity $v_x$", lw=2)
    ax.plot(t_ms, np.mean(right_vx, axis=0), color="#d62728", label="Right Velocity $v_x$", lw=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Lateral Velocity $v_x$ (mm/s)")
    ax.set_title("B. Velocity Emergence During Initial Prefix", fontweight="bold")
    ax.legend(loc="upper left")

    # Subplot C: 3D Speed Profile
    ax = axes[1, 0]
    speeds_l = np.linalg.norm(np.array([s.prefix_velocities for s in left_samples]), axis=2)
    speeds_r = np.linalg.norm(np.array([s.prefix_velocities for s in right_samples]), axis=2)
    ax.plot(t_ms, np.mean(speeds_l, axis=0), color="#1f77b4", label="Left Hand Speed", lw=2)
    ax.plot(t_ms, np.mean(speeds_r, axis=0), color="#d62728", label="Right Hand Speed", lw=2)
    ax.set_xlabel("Time from Movement Onset (ms)")
    ax.set_ylabel("3D Speed $\|v\|$ (mm/s)")
    ax.set_title("C. Instantaneous 3D Hand Speed", fontweight="bold")
    ax.legend(loc="upper left")

    # Subplot D: Distribution of 200 ms Displacement
    ax = axes[1, 1]
    dx200_l = left_disp_x[:, -1]
    dx200_r = right_disp_x[:, -1]
    ax.hist(dx200_l, bins=15, alpha=0.6, color="#1f77b4", label=f"Left (Mean={np.mean(dx200_l):.2f} mm)")
    ax.hist(dx200_r, bins=15, alpha=0.6, color="#d62728", label=f"Right (Mean={np.mean(dx200_r):.2f} mm)")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Displacement at 200 ms (mm)")
    ax.set_ylabel("Trial Count")
    ax.set_title("D. 200 ms Lateral Separation Distribution", fontweight="bold")
    ax.legend(loc="upper right")

    fig.suptitle("Figure 2: Observed 200 ms Movement Prefix Kinematic Signatures", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_2_prefix_dynamics.png", dpi=200)
    plt.close()


def plot_figure_3_endpoint_error_comparison(df_summary: pd.DataFrame, output_dir: Path):
    """Figure 3: Endpoint Prediction Error across Baselines and Learned Models."""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    models = df_summary["Model"].tolist()
    means = df_summary["Mean Endpoint Error (mm)"].tolist()
    sems = df_summary["SEM (mm)"].tolist()
    medians = df_summary["Median Error (mm)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, means, width, yerr=sems, capsize=4, label="Mean Error (mm) ± SEM", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width/2, medians, width, label="Median Error (mm)", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Endpoint Prediction Error (mm)")
    ax.set_title("Figure 3: Early Endpoint Prediction Error Under LOSO Validation", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "figure_3_endpoint_error_comparison.png", dpi=200)
    plt.close()


def plot_figure_4_horizon_error_curves(df_horizons: pd.DataFrame, output_dir: Path):
    """Figure 4: Multi-Horizon Forecasting Accuracy across Horizons (100, 200, 300, 500 ms)."""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    models = df_horizons["Model"].unique()
    colors = {
        "PERSISTENCE": "#7f7f7f",
        "CONSTANT_VELOCITY": "#e377c2",
        "RIDGE": "#1f77b4",
        "COMPACT_MLP": "#2ca02c",
        "COMPACT_GRU": "#d62728",
    }

    for m in models:
        sub_df = df_horizons[df_horizons["Model"] == m].sort_values("Horizon (ms)")
        h_vals = sub_df["Horizon (ms)"].values
        means = sub_df["Mean Error (mm)"].values
        sems = sub_df["SEM (mm)"].values
        col = colors.get(m, "#333333")
        ax.plot(h_vals, means, marker="o", lw=2, label=m, color=col)
        ax.fill_between(h_vals, means - sems, means + sems, color=col, alpha=0.15)

    ax.set_xlabel("Forecast Horizon Ahead of Prefix (ms)")
    ax.set_ylabel("3D Trajectory Error (mm)")
    ax.set_title("Figure 4: Kinematic Forecaster Error Across Forward Prediction Horizons", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_4_horizon_forecasting_error.png", dpi=200)
    plt.close()


def plot_figure_5_case_study(sample_pred: Dict, output_dir: Path):
    """Figure 5: Detailed 3D and Axis-by-Axis Trajectory Forecast Case Study on a Held-Out Trial."""
    setup_plotting_style()
    fig = plt.figure(figsize=(11, 7))

    prefix = np.array(sample_pred["prefix_positions"])
    actual_traj = np.array(sample_pred["actual_horizon_500"])
    pred_traj = np.array(sample_pred["predicted_horizon_500"])
    actual_end = np.array(sample_pred["actual_endpoint"])
    pred_end = np.array(sample_pred["predicted_endpoint"])

    t_obs = np.arange(len(prefix)) * 10.0
    t_fut = 200.0 + np.arange(1, len(actual_traj) + 1) * 10.0

    # 3D Subplot
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.plot(prefix[:, 0], prefix[:, 2], prefix[:, 1], color="black", lw=3, label="Observed Prefix (0–200ms)")
    ax3.plot(actual_traj[:, 0], actual_traj[:, 2], actual_traj[:, 1], color="#2ca02c", lw=2.5, label="Actual Future (+500ms)")
    ax3.plot(pred_traj[:, 0], pred_traj[:, 2], pred_traj[:, 1], color="#1f77b4", linestyle="--", lw=2.5, label="Forecast Future (+500ms)")
    ax3.scatter([actual_end[0]], [actual_end[2]], [actual_end[1]], color="#2ca02c", s=60, marker="*", label="Actual Endpoint")
    ax3.scatter([pred_end[0]], [pred_end[2]], [pred_end[1]], color="#1f77b4", s=60, marker="x", label="Predicted Endpoint")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_zlabel("Y (mm)")
    ax3.set_title("A. 3D Spatial Trajectory Forecast", fontweight="bold")
    ax3.legend(fontsize=7, loc="upper left")

    # Subplot B: Lateral X(t)
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t_obs, prefix[:, 0], color="black", lw=2.5, label="Observed (0-200 ms)")
    ax.plot(t_fut, actual_traj[:, 0], color="#2ca02c", lw=2, label="Actual Future")
    ax.plot(t_fut, pred_traj[:, 0], color="#1f77b4", linestyle="--", lw=2, label="Predicted Future")
    ax.scatter([t_fut[-1] + 1500], [actual_end[0]], color="#2ca02c", marker="*", s=50, label="Final Actual End")
    ax.scatter([t_fut[-1] + 1500], [pred_end[0]], color="#1f77b4", marker="x", s=50, label="Final Pred End")
    ax.axvline(200, color="gray", linestyle=":", alpha=0.7)
    ax.set_ylabel("Lateral X (mm)")
    ax.set_title("B. Lateral Axis $x(t)$ Development", fontweight="bold")
    ax.legend(fontsize=8, loc="best")

    # Subplot C: Height Y(t)
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_obs, prefix[:, 1], color="black", lw=2.5)
    ax.plot(t_fut, actual_traj[:, 1], color="#2ca02c", lw=2)
    ax.plot(t_fut, pred_traj[:, 1], color="#1f77b4", linestyle="--", lw=2)
    ax.axvline(200, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Height Y (mm)")
    ax.set_title("C. Vertical Axis $y(t)$ Development", fontweight="bold")

    # Subplot D: Depth Z(t)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t_obs, prefix[:, 2], color="black", lw=2.5)
    ax.plot(t_fut, actual_traj[:, 2], color="#2ca02c", lw=2)
    ax.plot(t_fut, pred_traj[:, 2], color="#1f77b4", linestyle="--", lw=2)
    ax.axvline(200, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Depth Z (mm)")
    ax.set_title("D. Depth Axis $z(t)$ Development", fontweight="bold")

    trial_id = sample_pred.get("trial_id", "test_trial")
    fig.suptitle(f"Figure 5: Trajectory Forecast Case Study on Held-Out Trial ({trial_id})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_5_forecast_case_study.png", dpi=200)
    plt.close()


def generate_all_phase_3_figures(
    samples: List[TrajectorySample],
    df_summary: pd.DataFrame,
    df_horizons: pd.DataFrame,
    sample_preds: List[Dict],
    output_dir: Path = Path("reports/phase_3/figures"),
):
    """Generate and save all publication-grade figures for Phase 3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_figure_1_coordinate_system(samples, output_dir)
    plot_figure_2_kinematic_dynamics(samples, output_dir)
    plot_figure_3_endpoint_error_comparison(df_summary, output_dir)
    plot_figure_4_horizon_error_curves(df_horizons, output_dir)
    if sample_preds:
        plot_figure_5_case_study(sample_preds[0], output_dir)
    print(f"Generated all Phase 3 figures in {output_dir}")
