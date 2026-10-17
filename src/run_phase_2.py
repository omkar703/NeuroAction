"""Master execution driver for Phase 2: Neural Intention Representation & Temporal Commitment.

Executes all neuroscience analyses, classical & deep ML pipelines under LOSO,
sliding-window temporal trajectories, neural commitment point detection, channel ablations,
saves out-of-fold predictions, and generates publication figures.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import pandas as pd
import numpy as np

from src.data.trial_loader import TrialLoader
from src.neuroscience.readiness_potential import ReadinessPotentialAnalyzer
from src.neuroscience.sensorimotor_rhythms import SensorimotorRhythmAnalyzer
from src.models.cross_validation import ExperimentRunner
from src.analysis.phase_2_plots import generate_all_figures


def run_phase_2_pipeline(output_dir: Path = Path("reports/phase_2")):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEUROACTION — PHASE 2 MASTER PIPELINE EXECUTION")
    print("Neural Intention Representation, Goal–Motor Dissociation & Temporal Commitment")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    print("\n[Step 1/6] Loading cached multimodal trials...")
    loader = TrialLoader()
    manifest_df = pd.read_csv("data/manifests/iackd_manifest.csv")
    valid_manifest = manifest_df[manifest_df["valid"]]

    subjects = sorted(list(valid_manifest["participant_id"].unique()))
    print(f"Loaded manifest with {len(manifest_df)} total indexed trials ({len(valid_manifest)} valid).")
    print(f"Participants in cohort ({len(subjects)} subjects): {subjects}")

    # Load trials
    trials = []
    for sub in subjects:
        sub_rows = valid_manifest[valid_manifest["participant_id"] == sub]
        runs = sorted(list(sub_rows["run_id"].unique()))
        for r in runs:
            run_trials = loader.load_run(sub, r)
            trials.extend(run_trials)
    print(f"Successfully loaded {len(trials)} valid CanonicalTrial objects in {time.time() - t0:.2f}s.")

    # 2. Descriptive Neuroscience Analyses
    print("\n[Step 2/6] Running Descriptive Neuroscience Analyses (RP & SMR ERD/ERS)...")
    rp_analyzer = ReadinessPotentialAnalyzer()
    rp_results = rp_analyzer.analyze_trials(trials)
    print("  Readiness Potential (RP) computed over [-1000, 0] ms.")
    for ch, slope_info in rp_results["all"]["slopes"].items():
        print(f"    RP Late Slope ({ch}): {slope_info['slope_uV_per_s']:.2f} uV/s (p={slope_info['pvalue']:.4f})")

    smr_analyzer = SensorimotorRhythmAnalyzer()
    smr_results = smr_analyzer.analyze_trials(trials)
    print("  Sensorimotor Rhythms (SMR) mu/beta ERD & Lateralization computed.")
    print(f"    Mean mu ERD at onset (Cz): {smr_results['all']['mu_erd_mean'][-1, 1]:.2f}%")
    print(f"    Mean beta ERD at onset (Cz): {smr_results['all']['beta_erd_mean'][-1, 1]:.2f}%")

    # 3. Model Training & Critical Experiment Matrix under LOSO
    print("\n[Step 3/6] Running Cross-Validation & Critical Experiment Matrix under LOSO...")
    runner = ExperimentRunner(trials)
    models_to_run = ["chance", "logistic_regression", "lda", "csp_lda", "eegnet"]
    matrix_df, all_oofs = runner.run_critical_experiment_matrix(model_names=models_to_run)
    print("\nCritical Experiment Matrix Results (Summary Top Rows):")
    print(matrix_df.to_string(index=False))

    # Save Experiment Matrix CSV
    matrix_csv = output_dir / "experiment_matrix_loso.csv"
    matrix_df.to_csv(matrix_csv, index=False)
    print(f"Saved Experiment Matrix to {matrix_csv}")

    # 4. Temporal Sliding-Window Decoding & Neural Commitment Points
    print("\n[Step 4/6] Running Temporal Sliding-Window Decoding across 17 pre-movement windows...")
    temp_dfs = runner.run_temporal_sliding_decoding(model_name="lda")
    ncps = runner.determine_neural_commitment_points(temp_dfs)
    print("Neural Commitment Point (NCP) Analysis:")
    for k, v in ncps.items():
        print(f"  {k}: {v['status']}")

    # 5. Spatial Channel Ablation Study
    print("\n[Step 5/6] Running Channel Ablation: Whole-Scalp (26) vs Motor-Focused (9)...")
    ablation_results = runner.run_channel_ablation(model_name="lda")
    print(f"  Goal Decoder: All 26={ablation_results['goal']['all_channels_acc']:.3f}, "
          f"Motor 9={ablation_results['goal']['motor_channels_acc']:.3f} "
          f"(Delta={ablation_results['goal']['delta']:+.3f})")
    print(f"  Motor Decoder: All 26={ablation_results['motor']['all_channels_acc']:.3f}, "
          f"Motor 9={ablation_results['motor']['motor_channels_acc']:.3f} "
          f"(Delta={ablation_results['motor']['delta']:+.3f})")

    # Save Out-of-Fold Predictions
    print("\n[Step 6/6] Saving Out-of-Fold Predictions & Generating Publication Figures...")
    oof_csv = runner.save_out_of_fold_predictions(all_oofs, output_dir / "out_of_fold_predictions.csv")

    # Extract subject metrics for best classical model (LDA)
    summ_lda_all, _ = runner.evaluate_loso(model_name="lda", target_type="goal", train_cond="all", test_cond="all")
    subject_results = summ_lda_all["per_subject"]

    # Generate all 10 Figures
    generate_all_figures(
        rp_results=rp_results,
        smr_results=smr_results,
        matrix_df=matrix_df,
        temp_dfs=temp_dfs,
        ncps=ncps,
        ablation_results=ablation_results,
        subject_results=subject_results,
        output_dir=fig_dir,
    )

    # Save summary dictionary for report generation
    summary_data = {
        "n_trials": len(trials),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "ncps": ncps,
        "ablation": ablation_results,
        "readiness_potential_slopes": {
            ch: rp_results["all"]["slopes"][ch] for ch in rp_results["all"]["slopes"]
        },
        "smr_erd_onset": {
            "mu_erd_c3": float(smr_results["all"]["mu_erd_mean"][-1, 0]),
            "mu_erd_cz": float(smr_results["all"]["mu_erd_mean"][-1, 1]),
            "mu_erd_c4": float(smr_results["all"]["mu_erd_mean"][-1, 2]),
            "beta_erd_c3": float(smr_results["all"]["beta_erd_mean"][-1, 0]),
            "beta_erd_cz": float(smr_results["all"]["beta_erd_mean"][-1, 1]),
            "beta_erd_c4": float(smr_results["all"]["beta_erd_mean"][-1, 2]),
        },
    }
    with open(output_dir / "phase_2_results_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PHASE 2 PIPELINE EXECUTION COMPLETE in {total_time:.2f}s ({total_time/60:.2f} min).")
    print(f"Artifacts saved in: {output_dir}")
    print("=" * 70)
    return summary_data, matrix_df


if __name__ == "__main__":
    run_phase_2_pipeline()
