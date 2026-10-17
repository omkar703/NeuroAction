from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.trial_loader import TrialLoader


def generate_all_plots(output_dir: Path = Path("reports/phase_1/figures")):
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = TrialLoader()
    
    # Load sample trials for sub-01
    trials_sub01 = loader.load_run("sub-01", "run-01_L1")
    manifest = pd.read_csv("data/manifests/iackd_manifest.csv")
    
    # Set scientific plotting style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
    })

    # ==========================================
    # Plot 1: Dataset Structure Overview
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = ["Total Participants", "Runs/Participant (Typical)", "Valid Trials (Dataset)", "Scalp EEG Channels", "Pre-EEG Freq (Hz)", "Execution Freq (Hz)"]
    values = [15, 8, 6671, 26, 200, 100]
    bars = ax.barh(categories, values, color="#2b5c8f")
    ax.set_xscale("log")
    ax.set_xlabel("Quantity / Value (Log Scale)")
    ax.set_title("Plot 1: IACKD Dataset Specifications & Multimodal Scope [Dataset-wide]")
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.15, bar.get_y() + bar.get_height()/2, f"{int(w):,}" if w > 10 else f"{w}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "plot1_dataset_overview.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 2: Trial Condition Distribution
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cond_counts = manifest["congruency"].value_counts()
    lbl_counts = manifest["label_code"].value_counts().sort_index()
    labels = ["0: red-left\n(Congruent L)", "1: red-right\n(Congruent R)", "2: yellow-left\n(Incongruent R->L)", "3: yellow-right\n(Incongruent L->R)"]
    colors = ["#3274a1", "#4192c7", "#e1812c", "#f29c48"]
    ax.bar(labels, [lbl_counts.get(i, 0) for i in range(4)], color=colors, width=0.6)
    ax.set_ylabel("Trial Count")
    ax.set_title("Plot 2: Experimental Condition & Label Balance [Cached-subset, N=160]")
    for i, c in enumerate(lbl_counts):
        ax.text(i, c + 1, str(c), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "plot2_condition_distribution.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 3: Example Pre-Movement EEG
    # ==========================================
    # Find representative left and right trials
    tr_left = next(t for t in trials_sub01 if t.physical_hand_direction == "left")
    c3_idx = tr_left.channels.index("C3")
    cz_idx = tr_left.channels.index("CZ")
    c4_idx = tr_left.channels.index("C4")
    
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    t_pre = tr_left.times_pre
    for ax, idx, ch_name, color in zip(axes, [c3_idx, cz_idx, c4_idx], ["C3 (Left Motor)", "Cz (Midline)", "C4 (Right Motor)"], ["#1f77b4", "#2ca02c", "#d62728"]):
        ax.plot(t_pre, tr_left.eeg_pre[:, idx], label=f"{ch_name}", color=color)
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.7)
        ax.set_ylabel(r"Amplitude ($\mu$V)")
        ax.legend(loc="upper left")
        ax.set_ylim(-35, 35)
    axes[-1].set_xlabel("Time Relative to Movement Onset (seconds)")
    fig.suptitle(f"Plot 3: Pre-Movement EEG Waveforms [Single Trial: Sub-01 {tr_left.trial_id}]")
    plt.tight_layout()
    plt.savefig(output_dir / "plot3_eeg_motor_channels.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 4: Example Kinematics x(t), y(t), z(t)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_ms = tr_left.times_me
    ax.plot(t_ms, tr_left.hand_x, label="x(t) [Lateral Movement]", color="#1f77b4", lw=2)
    ax.plot(t_ms, tr_left.hand_y, label="y(t) [Vertical Height]", color="#2ca02c", lw=1.5, linestyle="--")
    ax.plot(t_ms, tr_left.hand_z, label="z(t) [Depth/Longitudinal]", color="#9467bd", lw=1.5, linestyle=":")
    ax.axvline(0.0, color="gray", linestyle="--", label="Movement Onset (t=0)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Displacement (mm)")
    ax.set_title(f"Plot 4: 3D Hand Kinematics vs Time [Single Trial: Sub-01 {tr_left.trial_id}]")
    ax.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot4_kinematics_3d_time.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 5: Hand Spatial Trajectory
    # ==========================================
    tr_right = next(t for t in trials_sub01 if t.physical_hand_direction == "right")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr_left.hand_x, tr_left.hand_y, label=f"Leftward Trial ({tr_left.trial_id})", color="#1f77b4", lw=2)
    ax.plot(tr_right.hand_x, tr_right.hand_y, label=f"Rightward Trial ({tr_right.trial_id})", color="#d62728", lw=2)
    ax.scatter([tr_left.hand_x[~np.isnan(tr_left.hand_x)][0]], [tr_left.hand_y[~np.isnan(tr_left.hand_y)][0]], color="black", s=60, label="Start Position", zorder=5)
    ax.set_xlabel("Lateral Position x (mm)")
    ax.set_ylabel("Height y (mm)")
    ax.set_title("Plot 5: Spatial Hand Trajectory in Leap Motion Frame [Representative Trials: Sub-01]")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot5_hand_spatial_trajectory.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 6: Ball / Visual Trajectory
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tr_left.times_me, tr_left.ball_x_pix, label=f"Leftward Ball ({tr_left.trial_id})", color="#1f77b4", lw=2)
    ax.plot(tr_right.times_me, tr_right.ball_x_pix, label=f"Rightward Ball ({tr_right.trial_id})", color="#d62728", lw=2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Screen Ball Position x (pixels)")
    ax.set_title("Plot 6: External Virtual Action Trajectory [Representative Trials: Sub-01]")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "plot6_ball_trajectory.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 7: Multimodal Synchronization Timeline
    # ==========================================
    # Side-by-side comparison: Congruent vs Incongruent
    tr_cong = next(t for t in trials_sub01 if t.congruency == "congruent")
    tr_incong = next(t for t in trials_sub01 if t.congruency == "incongruent")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    for col_idx, (tr, title) in enumerate([(tr_cong, "Congruent Trial (Same Direction)"), (tr_incong, "Incongruent Trial (Opposite Direction)")]):
        # Top: Pre-movement EEG (C3 & C4)
        ax_top = axes[0, col_idx]
        ax_top.plot(tr.times_pre * 1000, tr.eeg_pre[:, c3_idx], label="C3", color="#1f77b4", alpha=0.8)
        ax_top.plot(tr.times_pre * 1000, tr.eeg_pre[:, c4_idx], label="C4", color="#d62728", alpha=0.8)
        ax_top.axvline(0, color="black", linestyle="--")
        ax_top.set_title(f"{title}\nTrial: {tr.trial_id} ({tr.ball_color.upper()}, Goal: {tr.target_goal})")
        ax_top.set_ylabel(r"Pre-EEG ($\mu$V)")
        ax_top.set_xlim(-1000, max(tr.times_me[-1], 2500))
        ax_top.legend(loc="upper left")
        
        # Middle: Physical Hand x(t)
        ax_mid = axes[1, col_idx]
        ax_mid.plot(tr.times_me, tr.hand_x, label="Hand x (mm)", color="#2ca02c", lw=2)
        ax_mid.axvline(0, color="black", linestyle="--")
        ax_mid.axhline(0, color="gray", linestyle=":")
        ax_mid.set_ylabel("Hand x (mm)")
        ax_mid.legend(loc="upper left")
        
        # Bottom: Virtual Ball x(t)
        ax_bot = axes[2, col_idx]
        ax_bot.plot(tr.times_me, tr.ball_x_pix, label="Ball x (px)", color="#e1812c", lw=2)
        ax_bot.axvline(0, color="black", linestyle="--")
        ax_bot.axhline(0, color="gray", linestyle=":")
        ax_bot.set_ylabel("Ball x (px)")
        ax_bot.set_xlabel("Time (ms) [0 = Movement Onset]")
        ax_bot.legend(loc="upper left")
        
    fig.suptitle("Plot 7: Multimodal Temporal Synchronization [Representative Trials: Sub-01]")
    plt.tight_layout()
    plt.savefig(output_dir / "plot7_multimodal_synchronization.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 8: Coverage / Quality Distribution
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    covs = manifest[manifest["coverage"] > 0]["coverage"]
    ax.hist(covs, bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
    ax.axvline(covs.median(), color="red", linestyle="--", label=f"Median Coverage: {covs.median():.2f}")
    ax.set_xlabel("Kinematic Coverage Ratio (cov)")
    ax.set_ylabel("Trial Count")
    ax.set_title("Plot 8: Leap Motion Kinematic Coverage Distribution [Cached-subset, N=160]")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "plot8_kinematic_coverage_distribution.png", dpi=150)
    plt.close()

    # ==========================================
    # Plot 9: Trial Duration Distribution
    # ==========================================
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    dur_cong = manifest[(manifest["congruency"] == "congruent") & (manifest["execution_samples"] > 0)]["execution_samples"] * 10  # ms
    dur_incong = manifest[(manifest["congruency"] == "incongruent") & (manifest["execution_samples"] > 0)]["execution_samples"] * 10  # ms
    bp = ax.boxplot([dur_cong, dur_incong], tick_labels=["Congruent", "Incongruent"], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3274a1')
    bp['boxes'][1].set_facecolor('#e1812c')
    ax.set_ylabel("Execution Duration (ms)")
    ax.set_title("Plot 9: Observed Increase in Execution Duration under Incongruent Mapping [Cached-subset, N=160]")
    # Annotate medians
    ax.text(1, dur_cong.median() + 80, f"Med: {dur_cong.median():.0f} ms", ha="center", fontweight="bold")
    ax.text(2, dur_incong.median() + 80, f"Med: {dur_incong.median():.0f} ms", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "plot9_trial_duration_distribution.png", dpi=150)
    plt.close()

    print(f"Successfully generated all 9 plots in {output_dir}")


if __name__ == "__main__":
    generate_all_plots()
