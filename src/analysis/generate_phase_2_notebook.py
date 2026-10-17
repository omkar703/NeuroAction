"""Creates and executes the Phase 2 walkthrough notebook: NeuroAction_Phase2_Walkthrough.ipynb."""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path


def create_phase_2_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        # Markdown Cell 1
        nbf.v4.new_markdown_cell("""# NeuroAction — Phase 2 Interactive Walkthrough
## Neural Intention Representation, Goal–Motor Dissociation & Temporal Commitment Analysis

**Dataset**: IACKD (OpenNeuro `ds006840`, Nature *Scientific Data* 2026)  
**Objective**: Determine whether pre-movement EEG encodes the abstract task goal, physical motor command, or both, and identify when this information becomes statistically decodable prior to movement onset ($-1000$ to $0$ ms).

---
### Notebook Outline
1. **Multimodal Data Ingestion & Class Balance Verification**
2. **Descriptive Neuroscience: Bereitschaftspotential / Readiness Potential (RP)**
3. **Sensorimotor Rhythms (SMR): $\\mu$ (8–13 Hz) & $\\beta$ (13–30 Hz) ERD/ERS Dynamics**
4. **Feature Engineering & Leakage-Free Common Spatial Pattern (CSP)**
5. **Master Experiment Matrix: Leave-One-Subject-Out (LOSO) Evaluation**
6. **The Central Transfer Experiment: Congruent $\\to$ Incongruent Mapping Transfer**
7. **Temporal Sliding-Window Trajectory & Neural Commitment Point (NCP)**
8. **Spatial Channel Ablation: Whole-Scalp vs. Motor Strip**
9. **Out-of-Fold Predictions & Artifact Summary**"""),

        # Code Cell 1: Environment & Setup
        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path.cwd()))

from src.data.trial_loader import TrialLoader
from src.neuroscience.readiness_potential import ReadinessPotentialAnalyzer
from src.neuroscience.sensorimotor_rhythms import SensorimotorRhythmAnalyzer
from src.features.pre_eeg import get_sliding_windows, extract_trial_features, CommonSpatialPattern
from src.models.decoders import ChanceBaseline, LogisticRegressionDecoder, LDADecoder, CSPLDADecoder
from src.models.eegnet import EEGNetClassifier
from src.models.cross_validation import ExperimentRunner

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print(f"PyTorch version: {torch.__version__}")
print("NeuroAction Phase 2 modules imported successfully.")"""),

        # Markdown Cell 2
        nbf.v4.new_markdown_cell("""## 1. Multimodal Data Ingestion & Cohort Inspection
We load all valid pre-movement trials across our Leave-One-Subject-Out benchmark cohort (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`)."""),

        # Code Cell 2
        nbf.v4.new_code_cell("""loader = TrialLoader()
manifest = pd.read_csv("data/manifests/iackd_manifest.csv")
valid_manifest = manifest[manifest["valid"]]

print(f"Total manifest trials: {len(manifest)} | Valid trials: {len(valid_manifest)}")
print("\\nTrials per participant:")
print(valid_manifest["participant_id"].value_counts().sort_index())

print("\\nClass Balance across Cohort:")
print(pd.crosstab(valid_manifest["congruency"], valid_manifest["target_goal"]))

# Load CanonicalTrial objects
trials = []
for sub in sorted(valid_manifest["participant_id"].unique()):
    sub_runs = valid_manifest[valid_manifest["participant_id"] == sub]["run_id"].unique()
    for r in sub_runs:
        trials.extend(loader.load_run(sub, r))

print(f"\\nLoaded {len(trials)} valid CanonicalTrial objects into memory.")"""),

        # Markdown Cell 3
        nbf.v4.new_markdown_cell("""## 2. Descriptive Neuroscience: Readiness Potential (RP)
We analyze the slow negative cortical motor potential ramping over $[-1000, 0]$ ms on midline (`Cz`) and lateral motor channels (`C3`, `C4`)."""),

        # Code Cell 3
        nbf.v4.new_code_cell("""rp_analyzer = ReadinessPotentialAnalyzer()
rp_res = rp_analyzer.analyze_trials(trials)

times_ms = rp_res["times"] * 1000.0

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, ch in enumerate(["C3", "CZ", "C4"]):
    ax = axes[i]
    c_mean = rp_res["congruent"]["mean"][:, i]
    c_low, c_high = rp_res["congruent"]["ci_low"][:, i], rp_res["congruent"]["ci_high"][:, i]
    i_mean = rp_res["incongruent"]["mean"][:, i]
    i_low, i_high = rp_res["incongruent"]["ci_low"][:, i], rp_res["incongruent"]["ci_high"][:, i]
    
    ax.plot(times_ms, c_mean, label="Congruent", color="#1f77b4", lw=2)
    ax.fill_between(times_ms, c_low, c_high, color="#1f77b4", alpha=0.15)
    ax.plot(times_ms, i_mean, label="Incongruent", color="#d62728", lw=2, linestyle="--")
    ax.fill_between(times_ms, i_low, i_high, color="#d62728", alpha=0.15)
    
    slope_info = rp_res["all"]["slopes"][ch]
    ax.set_title(f"Channel {ch}\\nSlope: {slope_info['slope_uV_per_s']:.2f} uV/s (p={slope_info['pvalue']:.2e})", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    if i == 0:
        ax.set_ylabel(r"Amplitude ($\\mu$V)")
        ax.legend(loc="upper left")

plt.suptitle("Bereitschaftspotential / Readiness Potential Across Sensorimotor Cortices", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()"""),

        # Markdown Cell 4
        nbf.v4.new_markdown_cell("""## 3. Sensorimotor Rhythms (SMR): $\\mu$ (8–13 Hz) & $\\beta$ (13–30 Hz) ERD/ERS
Event-Related Desynchronization (ERD) reflects cortical excitability and motor preparation. We evaluate power suppression relative to baseline $[-1000, -800]$ ms."""),

        # Code Cell 4
        nbf.v4.new_code_cell("""smr_analyzer = SensorimotorRhythmAnalyzer()
smr_res = smr_analyzer.analyze_trials(trials)

print(f"Cz mu-band ERD at movement onset: {smr_res['all']['mu_erd_mean'][-1, 1]:.2f}%")
print(f"Cz beta-band ERD at movement onset: {smr_res['all']['beta_erd_mean'][-1, 1]:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(times_ms, smr_res["all"]["mu_erd_mean"][:, 1], label="Mu (8-13 Hz)", color="#1f77b4", lw=2)
ax1.plot(times_ms, smr_res["all"]["beta_erd_mean"][:, 1], label="Beta (13-30 Hz)", color="#d62728", lw=2)
ax1.axhline(0, color="gray", linestyle=":")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Power Suppression / ERD (%)")
ax1.set_title("Midline Motor Cortex (Cz) Power Suppression", fontweight="bold")
ax1.legend()

ax2.plot(times_ms, smr_res["hand_left"]["mu_li_mean"], label="Left Hand Reaches", color="#3274a1", lw=2)
ax2.plot(times_ms, smr_res["hand_right"]["mu_li_mean"], label="Right Hand Reaches", color="#e1812c", lw=2)
ax2.axhline(0, color="gray", linestyle=":")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Lateralization Index (C3 - C4)/(C3 + C4)")
ax2.set_title("Sensorimotor Hemispheric Lateralization", fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.show()"""),

        # Markdown Cell 5
        nbf.v4.new_markdown_cell("""## 4. Master Experiment Matrix: Leave-One-Subject-Out (LOSO)
We inspect the complete 10-condition experiment matrix evaluated across all models under Leave-One-Subject-Out cross-validation."""),

        # Code Cell 5
        nbf.v4.new_code_cell("""matrix_df = pd.read_csv("reports/phase_2/experiment_matrix_loso.csv")
display_cols = ["Target", "Train Condition", "Test Condition", "Model", "Balanced Accuracy", "SEM", "95% CI", "F1", "AUROC"]
matrix_df[display_cols].head(20)"""),

        # Markdown Cell 6
        nbf.v4.new_markdown_cell("""## 5. The Central Transfer Experiment: Congruent $\\to$ Incongruent Mapping Transfer
This is the pivotal scientific test of Phase 2:
* A model trained on **Congruent** trials (where $\\text{Goal} == \\text{Hand}$) is tested on **Incongruent** trials (where $\\text{Goal} \\neq \\text{Hand}$).
* If it predicts Goal: **Goal Representation**.
* If it predicts Hand: **Motor Command Representation**."""),

        # Code Cell 6
        nbf.v4.new_code_cell("""transfer_df = matrix_df[
    (matrix_df["Model"].isin(["LDA", "EEGNET"])) &
    (matrix_df["Train Condition"] == "Congruent") &
    (matrix_df["Test Condition"] == "Incongruent")
][["Target", "Model", "Balanced Accuracy", "SEM", "95% CI", "AUROC"]]

print("THE CENTRAL TRANSFER EXPERIMENT RESULTS:")
display(transfer_df)"""),

        # Markdown Cell 7
        nbf.v4.new_markdown_cell("""## 6. Temporal Sliding Windows & Neural Commitment Point (NCP)
We examine the temporal decoding trajectories across all 17 sliding windows (from $-1000$ to $0$ ms)."""),

        # Code Cell 7
        nbf.v4.new_code_cell("""summary_json = Path("reports/phase_2/phase_2_results_summary.json")
import json
with open(summary_json) as f:
    summ = json.load(f)

print("Identified Neural Commitment Points (NCP):")
for k, v in summ["ncps"].items():
    print(f"  {k}: {v['status']}")"""),

        # Markdown Cell 8
        nbf.v4.new_markdown_cell("""## 7. Spatial Channel Ablation: Whole-Scalp (26) vs. Motor-Focused (9)
We evaluate whether removing non-motor channels (parietal, occipital, frontal) degrades goal decoding."""),

        # Code Cell 8
        nbf.v4.new_code_cell("""ablation = summ["ablation"]
print("CHANNEL ABLATION SUMMARY:")
print(f"Goal Decoder: All 26 = {ablation['goal']['all_channels_acc']:.3f} | Motor 9 = {ablation['goal']['motor_channels_acc']:.3f} (Delta = {ablation['goal']['delta']:+.3f})")
print(f"Motor Decoder: All 26 = {ablation['motor']['all_channels_acc']:.3f} | Motor 9 = {ablation['motor']['motor_channels_acc']:.3f} (Delta = {ablation['motor']['delta']:+.3f})")"""),

        # Markdown Cell 9
        nbf.v4.new_markdown_cell("""## 8. Out-of-Fold Predictions Preview
All 2,690 out-of-fold trial-level predictions exported for the Phase 4 conflict engine."""),

        # Code Cell 9
        nbf.v4.new_code_cell("""oof_df = pd.read_csv("reports/phase_2/out_of_fold_predictions.csv")
print(f"Total out-of-fold prediction records: {len(oof_df)}")
oof_df.head(10)"""),

        # Markdown Cell 10
        nbf.v4.new_markdown_cell("""## 9. Gallery of Generated Publication Figures
Displaying all 10 publication figures saved in `reports/phase_2/figures/`."""),

        # Code Cell 10
        nbf.v4.new_code_cell("""from IPython.display import Image, display

figures = sorted(list(Path("reports/phase_2/figures").glob("figure_*.png")))
print(f"Found {len(figures)} figures:")
for fig_path in figures:
    print(f"  {fig_path.name}")
    display(Image(filename=str(fig_path), width=700))""")
    ]

    nb_path = Path("NeuroAction_Phase2_Walkthrough.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {nb_path}")

    print("Executing notebook with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved {nb_path}")


if __name__ == "__main__":
    create_phase_2_notebook()
