# NeuroAction — Experimental Results & Research Artifacts

This directory consolidates the audited empirical results, model predictions, evaluation tables, performance metrics, and publication figures across all investigations of the **NeuroAction** research project.

---

## Directory Organization

```text
results/
├── README.md                                  # This master index and provenance guide
├── figures/                                   # Consolidated publication figures and scientific plots
│   ├── README.md                              # Detailed figure catalog with captions
│   ├── figure_1_experimental_design.png
│   ├── figure_2_readiness_potential.png
│   ├── figure_3_endpoint_error_comparison.png
│   ├── figure_4_horizon_forecasting_error.png
│   ├── figure_9_multimodal_architecture.png
│   ├── figure_10_conflict_detection_analysis.png
│   ├── figure_11_demo_trial_replay.png
│   ├── figure_13_comparative_benchmarks.png
│   └── ...
├── predictions/                               # Prediction CSV files with trial-level outputs
│   ├── README.md                              # Column schemas and data dictionary
│   ├── neural_goal_predictions_loso.csv
│   ├── trajectory_endpoint_predictions_loso.csv
│   └── reconciled_multimodal_evaluation_matrix.csv
├── tables/                                    # Audited evaluation matrices & cross-validation summaries
│   ├── README.md                              # Table definitions and experimental protocols
│   ├── neural_goal_decoding_experiment_matrix.csv
│   ├── forecasting_metrics_by_horizon.csv
│   ├── kinematic_comparison_event_vs_onset.csv
│   ├── participant_forecasting_metrics.csv
│   ├── trajectory_benchmarks.csv
│   ├── intention_action_evaluation_matrix.csv
│   ├── intention_action_confusion_matrix.csv
│   ├── digital_twin_evaluation_matrix.csv
│   ├── threshold_sensitivity_analysis.csv
│   └── trial_eligibility_matrix.csv
└── metrics/                                   # JSON performance metrics and summary statistics
    ├── README.md                              # Metric definitions
    └── neural_goal_decoding_summary.json
```

---

## Summary of Audited Benchmark Results

Every metric reported below is grounded directly in the audited CSV and JSON artifacts in this directory, evaluated under strict 5-fold Leave-One-Subject-Out (LOSO) cross-validation on OpenNeuro `ds006840`:

| Research Task | Evaluated Model | Empirical Metric | Baseline Comparison | Provenance Source |
| :--- | :--- | :---: | :---: | :--- |
| **3D Trajectory Forecasting** | **Ridge Regressor** (200 ms prefix) | **47.25 mm** RMSE | **−44.8%** vs. Fair Constant Velocity (85.65 mm)<br>**−73.4%** vs. Static Persistence (177.51 mm) | `results/tables/participant_forecasting_metrics.csv` |
| **Neural Goal Decoding** | **CSP + LDA** ($[-1000, 0]$ ms EEG) | **68.34%** BalAcc | Chance: 50.0% | `results/predictions/neural_goal_predictions_loso.csv` |
| **Intention–Action Discordance** | **Multimodal Conflict Engine** | **66.46%** BalAcc<br>**0.6834** AUROC | Chance: 50.0% | `results/tables/intention_action_evaluation_matrix.csv` |
| **Condition Discrimination** | **Multimodal Fusion Model** | **88.35%** BalAcc<br>**0.9395** AUROC | Condition cue reversal discrimination (congruent vs. incongruent) | `results/tables/intention_action_confusion_matrix.csv` |

---

## Scientific and Epistemic Distinctions

1. **Proxy Target Coordinates**: The OpenNeuro `ds006840` dataset does not include independently surveyed 3D physical target positions. Left ($-188, 150, 53$ mm) and Right ($+188, 149, 51$ mm) targets serve as empirical centroid proxies derived from clean reach endpoints.
2. **Experimental Condition vs. Motor Slips**: The 88.35% Balanced Accuracy and 0.9395 AUROC quantify experimental stimulus condition discrimination (congruent vs. incongruent target cues), *not* involuntary biological motor slips.
3. **Filter Disclosure**: Offline EEG preprocessing employed zero-phase non-causal bandpass filtering (0.5–40 Hz). While standard in neuroscience, real-time prospective deployments must substitute strictly causal IIR filters.
