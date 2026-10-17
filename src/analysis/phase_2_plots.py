"""Phase 2 Scientific Figures Generator.

Generates the 10 required publication figures:
- Figure 1: Experimental relationship: Goal <-> Hand <-> Ball
- Figure 2: C3/Cz/C4 pre-movement EEG (Readiness Potential waveforms)
- Figure 3: mu/beta time-frequency & ERD/ERS analysis
- Figure 4: Goal-decoding temporal performance
- Figure 5: Motor-command decoding temporal performance
- Figure 6: Goal vs motor comparison
- Figure 7: Congruent -> Incongruent transfer (The Central Experiment)
- Figure 8: Subject-level performance distribution
- Figure 9: Goal and motor temporal commitment curves (NCP)
- Figure 10: Spatial/channel contribution analysis (All vs Motor-focused)
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style():
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


def plot_figure_1_experimental_design(output_dir: Path):
    """Figure 1: Experimental relationship between Goal, Hand, and Ball in Congruent & Incongruent."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Congruent Subplot
    ax = axes[0]
    ax.text(0.5, 0.90, "CONGRUENT MAPPING (Red Cue)", ha="center", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.annotate("Target Goal:\nLeft", xy=(0.2, 0.65), xytext=(0.2, 0.65), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#e1edf8", ec="#1f77b4", lw=1.5))
    ax.annotate("Ball Motion:\nLeft", xy=(0.2, 0.40), xytext=(0.2, 0.40), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#e1edf8", ec="#1f77b4", lw=1.5))
    ax.annotate("Physical Hand:\nLeft (Aligned)", xy=(0.2, 0.15), xytext=(0.2, 0.15), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#d5e8d4", ec="#2ca02c", lw=2))

    ax.annotate("Target Goal:\nRight", xy=(0.8, 0.65), xytext=(0.8, 0.65), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#e1edf8", ec="#1f77b4", lw=1.5))
    ax.annotate("Ball Motion:\nRight", xy=(0.8, 0.40), xytext=(0.8, 0.40), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#e1edf8", ec="#1f77b4", lw=1.5))
    ax.annotate("Physical Hand:\nRight (Aligned)", xy=(0.8, 0.15), xytext=(0.8, 0.15), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#d5e8d4", ec="#2ca02c", lw=2))

    # Arrows
    ax.annotate("", xy=(0.2, 0.50), xytext=(0.2, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"))
    ax.annotate("", xy=(0.2, 0.25), xytext=(0.2, 0.33), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2ca02c"))
    ax.annotate("", xy=(0.8, 0.50), xytext=(0.8, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"))
    ax.annotate("", xy=(0.8, 0.25), xytext=(0.8, 0.33), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2ca02c"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Incongruent Subplot
    ax = axes[1]
    ax.text(0.5, 0.90, "INCONGRUENT MAPPING (Yellow Cue)", ha="center", fontsize=12, fontweight="bold", color="#d62728")
    ax.annotate("Target Goal:\nLeft", xy=(0.2, 0.65), xytext=(0.2, 0.65), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2e9", ec="#d62728", lw=1.5))
    ax.annotate("Ball Motion:\nLeft", xy=(0.2, 0.40), xytext=(0.2, 0.40), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2e9", ec="#d62728", lw=1.5))
    ax.annotate("Physical Hand:\nRight (REVERSED)", xy=(0.8, 0.15), xytext=(0.8, 0.15), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8d7da", ec="#d62728", lw=2))

    ax.annotate("Target Goal:\nRight", xy=(0.8, 0.65), xytext=(0.8, 0.65), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2e9", ec="#d62728", lw=1.5))
    ax.annotate("Ball Motion:\nRight", xy=(0.8, 0.40), xytext=(0.8, 0.40), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2e9", ec="#d62728", lw=1.5))
    ax.annotate("Physical Hand:\nLeft (REVERSED)", xy=(0.2, 0.15), xytext=(0.2, 0.15), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8d7da", ec="#d62728", lw=2))

    # Cross Arrows
    ax.annotate("", xy=(0.2, 0.50), xytext=(0.2, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"))
    ax.annotate("", xy=(0.8, 0.50), xytext=(0.8, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"))
    ax.annotate("", xy=(0.75, 0.22), xytext=(0.25, 0.33), arrowprops=dict(arrowstyle="->", lw=2, color="#d62728", linestyle="--"))
    ax.annotate("", xy=(0.25, 0.22), xytext=(0.75, 0.33), arrowprops=dict(arrowstyle="->", lw=2, color="#d62728", linestyle="--"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Figure 1: Goal <-> Hand <-> Ball Relationship Under Visuomotor Remapping", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_1_experimental_design.png", dpi=200)
    plt.close()


def plot_figure_2_readiness_potential(rp_results: Dict, output_dir: Path):
    """Figure 2: C3/Cz/C4 pre-movement EEG waveforms across conditions."""
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.5), sharex=True)
    times = rp_results["times"] * 1000.0  # Convert to ms

    channels = rp_results["channels"]
    channel_labels = ["C3 (Left Motor)", "Cz (Supplementary Motor)", "C4 (Right Motor)"]

    for ch_idx, (ch_name, ch_label) in enumerate(zip(channels, channel_labels)):
        ax = axes[ch_idx]
        # Congruent vs Incongruent
        c_mean = rp_results["congruent"]["mean"][:, ch_idx]
        c_low = rp_results["congruent"]["ci_low"][:, ch_idx]
        c_high = rp_results["congruent"]["ci_high"][:, ch_idx]

        i_mean = rp_results["incongruent"]["mean"][:, ch_idx]
        i_low = rp_results["incongruent"]["ci_low"][:, ch_idx]
        i_high = rp_results["incongruent"]["ci_high"][:, ch_idx]

        ax.plot(times, c_mean, label="Congruent (All)", color="#1f77b4", lw=2)
        ax.fill_between(times, c_low, c_high, color="#1f77b4", alpha=0.18)

        ax.plot(times, i_mean, label="Incongruent (All)", color="#d62728", lw=2, linestyle="--")
        ax.fill_between(times, i_low, i_high, color="#d62728", alpha=0.18)

        ax.axvline(0.0, color="black", linestyle=":", alpha=0.8, label="Movement Onset (t=0)")
        ax.set_ylabel(r"Amplitude ($\mu$V)")
        ax.set_title(f"Channel {ch_label}", fontsize=11, fontweight="bold")
        if ch_idx == 0:
            ax.legend(loc="upper left", framealpha=0.9)

    axes[-1].set_xlabel("Time Relative to Movement Onset (ms)")
    fig.suptitle("Figure 2: Bereitschaftspotential / Readiness Potential Across Sensorimotor Cortices [-1000 to 0 ms]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_2_readiness_potential.png", dpi=200)
    plt.close()


def plot_figure_3_smr_erd(smr_results: Dict, output_dir: Path):
    """Figure 3: mu/beta time-frequency & ERD/ERS analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    times = smr_results["times"] * 1000.0

    # Top Left: Mu ERD on C3, Cz, C4
    ax = axes[0, 0]
    ax.plot(times, smr_results["all"]["mu_erd_mean"][:, 0], label=r"C3 ($\mu$: 8-13 Hz)", color="#1f77b4", lw=2)
    ax.plot(times, smr_results["all"]["mu_erd_mean"][:, 1], label=r"Cz ($\mu$: 8-13 Hz)", color="#2ca02c", lw=2)
    ax.plot(times, smr_results["all"]["mu_erd_mean"][:, 2], label=r"C4 ($\mu$: 8-13 Hz)", color="#d62728", lw=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(0, color="black", linestyle=":")
    ax.set_ylabel("ERD / ERS (%)")
    ax.set_title(r"A. $\mu$-Band (8–13 Hz) Power Suppression [ERD%]", fontweight="bold")
    ax.legend(loc="lower left")

    # Bottom Left: Beta ERD on C3, Cz, C4
    ax = axes[1, 0]
    ax.plot(times, smr_results["all"]["beta_erd_mean"][:, 0], label=r"C3 ($\beta$: 13-30 Hz)", color="#1f77b4", lw=2)
    ax.plot(times, smr_results["all"]["beta_erd_mean"][:, 1], label=r"Cz ($\beta$: 13-30 Hz)", color="#2ca02c", lw=2)
    ax.plot(times, smr_results["all"]["beta_erd_mean"][:, 2], label=r"C4 ($\beta$: 13-30 Hz)", color="#d62728", lw=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(0, color="black", linestyle=":")
    ax.set_xlabel("Time Relative to Movement Onset (ms)")
    ax.set_ylabel("ERD / ERS (%)")
    ax.set_title(r"B. $\beta$-Band (13–30 Hz) Power Suppression [ERD%]", fontweight="bold")
    ax.legend(loc="lower left")

    # Top Right: Lateralization Index (C3 - C4) / (C3 + C4)
    ax = axes[0, 1]
    ax.plot(times, smr_results["hand_left"]["mu_li_mean"], label=r"Left Hand ($\mu$ LI)", color="#3274a1", lw=2)
    ax.plot(times, smr_results["hand_right"]["mu_li_mean"], label=r"Right Hand ($\mu$ LI)", color="#e1812c", lw=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(0, color="black", linestyle=":")
    ax.set_ylabel("Lateralization Index")
    ax.set_title(r"C. Sensorimotor Lateralization $(C3 - C4)/(C3 + C4)$", fontweight="bold")
    ax.legend(loc="upper left")

    # Bottom Right: Spectrogram Heatmap (Cz)
    ax = axes[1, 1]
    spec = smr_results["spectrogram"]
    im = ax.pcolormesh(
        spec["times"] * 1000.0,
        spec["freqs"],
        10 * np.log10(spec["mean_power_cz"] + 1e-12),
        shading="gouraud",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.axvline(0, color="white", linestyle=":")
    ax.set_xlabel("Time Relative to Movement Onset (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("D. Time-Frequency Power Spectrum [Cz]", fontweight="bold")

    fig.suptitle("Figure 3: Sensorimotor Rhythm Dynamics, ERD Suppression & Spectral Evolution",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_3_smr_erd.png", dpi=200)
    plt.close()


def plot_figure_4_goal_temporal(temp_dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """Figure 4: Goal-decoding temporal performance curves."""
    df_all = temp_dfs["goal_all"]
    df_transfer = temp_dfs["goal_transfer"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df_all["center_ms"], df_all["balanced_accuracy"], label="Goal Decoder (All -> All)", color="#1f77b4", marker="o", lw=2)
    ax.fill_between(df_all["center_ms"], df_all["ci_low"], df_all["ci_high"], color="#1f77b4", alpha=0.2)

    ax.plot(df_transfer["center_ms"], df_transfer["balanced_accuracy"], label="Goal Transfer (Congruent -> Incongruent)",
            color="#2ca02c", marker="s", lw=2, linestyle="--")
    ax.fill_between(df_transfer["center_ms"], df_transfer["ci_low"], df_transfer["ci_high"], color="#2ca02c", alpha=0.15)

    ax.axhline(0.50, color="gray", linestyle=":", label="Theoretical Chance (0.50)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.7, label="Movement Onset (t=0)")

    ax.set_xlabel("Sliding Window Center Time (ms before onset)")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 4: Temporal Trajectory of Abstract Goal Decoding Across Pre-Movement Epoch", fontweight="bold")
    ax.set_ylim(0.40, 0.90)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_4_goal_temporal.png", dpi=200)
    plt.close()


def plot_figure_5_motor_temporal(temp_dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """Figure 5: Motor-command decoding temporal performance curves."""
    df_all = temp_dfs["motor_all"]
    df_transfer = temp_dfs["motor_transfer"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df_all["center_ms"], df_all["balanced_accuracy"], label="Motor Decoder (All -> All)", color="#d62728", marker="o", lw=2)
    ax.fill_between(df_all["center_ms"], df_all["ci_low"], df_all["ci_high"], color="#d62728", alpha=0.2)

    ax.plot(df_transfer["center_ms"], df_transfer["balanced_accuracy"], label="Motor Transfer (Congruent -> Incongruent)",
            color="#9467bd", marker="^", lw=2, linestyle="--")
    ax.fill_between(df_transfer["center_ms"], df_transfer["ci_low"], df_transfer["ci_high"], color="#9467bd", alpha=0.15)

    ax.axhline(0.50, color="gray", linestyle=":", label="Theoretical Chance (0.50)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.7, label="Movement Onset (t=0)")

    ax.set_xlabel("Sliding Window Center Time (ms before onset)")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 5: Temporal Trajectory of Physical Motor-Command Decoding Across Pre-Movement Epoch", fontweight="bold")
    ax.set_ylim(0.15, 0.85)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_5_motor_temporal.png", dpi=200)
    plt.close()


def plot_figure_6_goal_vs_motor_comparison(matrix_df: pd.DataFrame, output_dir: Path):
    """Figure 6: Goal vs Motor comparison across models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Filter for All -> All condition
    subset = matrix_df[(matrix_df["Train Condition"] == "All") & (matrix_df["Test Condition"] == "All")]
    
    models = sorted(list(subset["Model"].unique()))
    x = np.arange(len(models))
    width = 0.35

    goal_accs = [subset[(subset["Model"] == m) & (subset["Target"] == "Goal")]["Balanced Accuracy"].values[0] for m in models]
    motor_accs = [subset[(subset["Model"] == m) & (subset["Target"] == "Motor")]["Balanced Accuracy"].values[0] for m in models]

    b1 = ax.bar(x - width/2, goal_accs, width, label="Task A: Goal Decoder", color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + width/2, motor_accs, width, label="Task B: Motor Command Decoder", color="#d62728", alpha=0.9)

    ax.axhline(0.50, color="gray", linestyle="--", label="Chance Baseline (0.50)")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 6: Direct Comparison: Abstract Goal vs Physical Motor Decoding Across Models", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.40, 0.85)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_6_goal_vs_motor_comparison.png", dpi=200)
    plt.close()


def plot_figure_7_congruent_to_incongruent_transfer(matrix_df: pd.DataFrame, output_dir: Path):
    """Figure 7: Congruent -> Incongruent transfer (The Central Experiment of Phase 2)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Compare Congruent -> Congruent vs Congruent -> Incongruent
    lda_rows = matrix_df[matrix_df["Model"] == "LDA"]
    
    comparisons = [
        ("Goal: Congruent -> Congruent", "Goal", "Congruent", "Congruent"),
        ("Goal: Congruent -> Incongruent\n(Transfer Test)", "Goal", "Congruent", "Incongruent"),
        ("Motor: Congruent -> Congruent", "Motor", "Congruent", "Congruent"),
        ("Motor: Congruent -> Incongruent\n(Transfer Test)", "Motor", "Congruent", "Incongruent"),
    ]

    labels = [c[0] for c in comparisons]
    accs = []
    sems = []
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for _, target, tr_c, te_c in comparisons:
        row = lda_rows[(lda_rows["Target"] == target) & (lda_rows["Train Condition"] == tr_c) & (lda_rows["Test Condition"] == te_c)]
        accs.append(row["Balanced Accuracy"].values[0])
        sems.append(row["SEM"].values[0])

    bars = ax.bar(labels, accs, yerr=sems, capsize=5, color=colors, alpha=0.88, width=0.55)
    ax.axhline(0.50, color="gray", linestyle="--", label="Chance Baseline (0.50)")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 7: The Central Transfer Experiment: Neural Generalization Across Visuomotor Remapping", fontweight="bold")
    ax.set_ylim(0.15, 0.90)

    for bar, acc in zip(bars, accs):
        ax.annotate(f"{acc:.3f}", xy=(bar.get_x() + bar.get_width()/2, acc + 0.02),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_7_transfer_experiment.png", dpi=200)
    plt.close()


def plot_figure_8_subject_distribution(subject_results: List[Dict], output_dir: Path):
    """Figure 8: Subject-level performance distribution across LOSO folds."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    subs = [m["subject"] for m in subject_results]
    accs = [m["balanced_accuracy"] for m in subject_results]
    aucs = [m["auroc"] for m in subject_results]

    x = np.arange(len(subs))
    w = 0.35

    b1 = ax.bar(x - w/2, accs, w, label="Balanced Accuracy", color="#2b5c8f", alpha=0.85)
    b2 = ax.bar(x + w/2, aucs, w, label="AUROC", color="#e1812c", alpha=0.85)

    ax.axhline(0.50, color="gray", linestyle="--", label="Chance Baseline (0.50)")
    ax.axhline(np.mean(accs), color="#2b5c8f", linestyle=":", label=f"Mean Acc: {np.mean(accs):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(subs)
    ax.set_ylabel("Score")
    ax.set_title("Figure 8: Individual Subject Performance Distribution Under Leave-One-Subject-Out CV", fontweight="bold")
    ax.set_ylim(0.35, 1.00)

    for bar, val in zip(b1, accs):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(b2, aucs):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_8_subject_distribution.png", dpi=200)
    plt.close()


def plot_figure_9_neural_commitment_points(
    temp_dfs: Dict[str, pd.DataFrame], ncps: Dict[str, Dict], output_dir: Path
):
    """Figure 9: Goal and Motor temporal commitment curves (Neural Commitment Point identification)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    df_goal = temp_dfs["goal_all"]
    df_motor = temp_dfs["motor_all"]

    ax.plot(df_goal["center_ms"], df_goal["balanced_accuracy"], label="Goal Decoder Trajectory", color="#1f77b4", marker="o", lw=2)
    ax.fill_between(df_goal["center_ms"], df_goal["ci_low"], df_goal["ci_high"], color="#1f77b4", alpha=0.18)

    ax.plot(df_motor["center_ms"], df_motor["balanced_accuracy"], label="Motor Decoder Trajectory", color="#d62728", marker="s", lw=2)
    ax.fill_between(df_motor["center_ms"], df_motor["ci_low"], df_motor["ci_high"], color="#d62728", alpha=0.18)

    ax.axhline(0.50, color="gray", linestyle=":", label="Theoretical Chance (0.50)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.8, label="Movement Onset (t=0)")

    # Mark Goal NCP if detected
    if ncps["goal_all"]["detected"]:
        t_ncp = ncps["goal_all"]["center_ms"]
        acc_ncp = ncps["goal_all"]["balanced_accuracy"]
        ax.axvline(t_ncp, color="#1f77b4", linestyle="--", lw=1.5)
        ax.scatter([t_ncp], [acc_ncp], color="#1f77b4", s=120, zorder=5)
        ax.annotate(f"Goal NCP:\n{int(t_ncp)} ms", xy=(t_ncp, acc_ncp), xytext=(t_ncp - 120, acc_ncp + 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e1edf8", ec="#1f77b4", lw=1.2), fontweight="bold")

    # Mark Motor NCP if detected
    if ncps["motor_all"]["detected"]:
        t_ncp_m = ncps["motor_all"]["center_ms"]
        acc_ncp_m = ncps["motor_all"]["balanced_accuracy"]
        ax.axvline(t_ncp_m, color="#d62728", linestyle="--", lw=1.5)
        ax.scatter([t_ncp_m], [acc_ncp_m], color="#d62728", s=120, zorder=5)
        ax.annotate(f"Motor NCP:\n{int(t_ncp_m)} ms", xy=(t_ncp_m, acc_ncp_m), xytext=(t_ncp_m + 30, acc_ncp_m - 0.08),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#d62728", lw=1.2), fontweight="bold")

    ax.set_xlabel("Time Before Movement Onset (ms)")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 9: Neural Commitment Point (NCP) Identification & Temporal Predictability Onset", fontweight="bold")
    ax.set_ylim(0.40, 0.85)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_9_neural_commitment_points.png", dpi=200)
    plt.close()


def plot_figure_10_spatial_ablation(ablation_results: Dict[str, Dict], output_dir: Path):
    """Figure 10: Spatial/channel contribution analysis (All 26 scalp channels vs Motor-focused ablation)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    categories = ["Task A: Goal Decoder", "Task B: Motor Command Decoder"]
    all_ch_accs = [ablation_results["goal"]["all_channels_acc"], ablation_results["motor"]["all_channels_acc"]]
    motor_ch_accs = [ablation_results["goal"]["motor_channels_acc"], ablation_results["motor"]["motor_channels_acc"]]

    x = np.arange(len(categories))
    w = 0.32

    b1 = ax.bar(x - w/2, all_ch_accs, w, label="All 26 Scalp Channels", color="#2b5c8f", alpha=0.9)
    b2 = ax.bar(x + w/2, motor_ch_accs, w, label="Motor-Focused (FC/C/CP, 9 Channels)", color="#2ca02c", alpha=0.9)

    ax.axhline(0.50, color="gray", linestyle="--", label="Chance Baseline (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylabel("Balanced Accuracy (LOSO)")
    ax.set_title("Figure 10: Spatial Channel Ablation: Whole-Scalp (26) vs Motor-Focused (9) Channels", fontweight="bold")
    ax.set_ylim(0.40, 0.85)

    for bar, val in zip(b1, all_ch_accs):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(b2, motor_ch_accs):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_10_spatial_ablation.png", dpi=200)
    plt.close()


def generate_all_figures(
    rp_results: Dict,
    smr_results: Dict,
    matrix_df: pd.DataFrame,
    temp_dfs: Dict[str, pd.DataFrame],
    ncps: Dict[str, Dict],
    ablation_results: Dict[str, Dict],
    subject_results: List[Dict],
    output_dir: Path = Path("reports/phase_2/figures"),
):
    """Generate and save all 10 Phase 2 figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    plot_figure_1_experimental_design(output_dir)
    plot_figure_2_readiness_potential(rp_results, output_dir)
    plot_figure_3_smr_erd(smr_results, output_dir)
    plot_figure_4_goal_temporal(temp_dfs, output_dir)
    plot_figure_5_motor_temporal(temp_dfs, output_dir)
    plot_figure_6_goal_vs_motor_comparison(matrix_df, output_dir)
    plot_figure_7_congruent_to_incongruent_transfer(matrix_df, output_dir)
    plot_figure_8_subject_distribution(subject_results, output_dir)
    plot_figure_9_neural_commitment_points(temp_dfs, ncps, output_dir)
    plot_figure_10_spatial_ablation(ablation_results, output_dir)
    print(f"Successfully generated all 10 figures in {output_dir}")
