# NeuroAction — Evaluation Tables & Cross-Validation Matrices

This directory contains aggregated evaluation metrics, cross-validation matrices, and sensitivity analyses.

---

## Evaluation Tables

### 1. `participant_forecasting_metrics.csv`
Per-participant trajectory forecasting benchmark comparing the Ridge Regressor against the Fair Constant Velocity baseline and Static Persistence baseline across 5-fold LOSO folds:

| Participant ID | Valid Trials | Persistence RMSE (mm) | Fair Constant Velocity RMSE (mm) | Ridge Forecaster RMSE (mm) | Error Reduction vs. Fair CV (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `sub-01` | 40 | 179.80 | 88.85 | **50.60** | **−43.1%** |
| `sub-02` | 52 | 180.45 | 87.05 | **47.64** | **−45.3%** |
| `sub-03` | 60 | 176.62 | 82.26 | **44.97** | **−45.3%** |
| `sub-08` | 57 | 177.30 | 85.94 | **48.97** | **−43.0%** |
| `sub-15` | 60 | 174.96 | 85.50 | **44.62** | **−47.8%** |
| **Cohort Mean** | **269** | **177.51** | **85.65** | **47.25** | **−44.8%** |

### 2. `forecasting_metrics_by_horizon.csv`
Evaluation of trajectory forecasting accuracy as a function of prefix observation horizon ($100$, $200$, $300$, and $500$ ms). Confirms that $200$ ms achieves the optimal trade-off between lead time ($> 1000$ ms) and low error ($47.25$ mm).

### 3. `intention_action_evaluation_matrix.csv`
Multimodal conflict tracking results evaluating agreement between the pre-movement neural intention prediction and overt kinematic trajectory.

### 4. `intention_action_confusion_matrix.csv`
Contingency matrix detailing true positive, true negative, false positive, and false negative counts for intention–action discordance detection under LOSO cross-validation.

### 5. `trial_eligibility_matrix.csv`
The authoritative audit of all 280 experimental trials in the OpenNeuro `ds006840` cohort, establishing that 269 trials meet strict multimodal criteria (valid pre-movement EEG, valid 200 ms prefix, and non-NaN 3D endpoint), while 11 trials are excluded due to optical sensor dropouts.
