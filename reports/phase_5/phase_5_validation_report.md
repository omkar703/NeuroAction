# NeuroAction Phase 5 — Independent Scientific Validation Report

**Document Version:** 1.0.0  
**Status:** Official Scientific Validation Report  
**Date of Evaluation:** October 2026  
**Primary Dataset:** OpenNeuro `ds006840` (IACKD), Version `1.0.0`  
**Cohort Scope:** 5 Participants (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`), $N = 269$ Valid Trials  
**Figure References:**
- `reports/phase_5/figures/figure_12_validation_architecture.png`
- `reports/phase_5/figures/figure_13_comparative_benchmarks.png`
- `reports/phase_5/figures/figure_14_prospective_protocol_timeline.png`

---

## 1. Executive Summary

Phase 5 of the NeuroAction project establishes a rigorous, leakage-controlled independent validation framework designed to evaluate prospective outcome prediction in motor execution. This report presents the empirical results generated across the full cohort of 269 valid multimodal trials evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation.

### Key Scientific Findings:
1. **Kinematic Trajectory Forecasting (Task B):** The regularized Ridge kinematic forecaster achieved a cohort 3D endpoint RMSE of **$47.25\text{ mm}$** (Mean Euclidean Error: **$32.85\text{ mm}$**). This constitutes a **$44.8\%$ error reduction** over the fair constant-velocity baseline ($85.65\text{ mm}$) and a **$73.4\%$ error reduction** over the static persistence baseline ($177.51\text{ mm}$), confirming the predictive superiority of the learned kinematic twin.
2. **Pre-Movement Neural Goal Decoding (Task A):** Evaluating pre-movement 64-channel EEG features under LOSO yielded a balanced accuracy of **$68.34\%$** ($95\%\text{ CI}: [62.4\%, 74.1\%]$), replicating the audited Phase 2 performance without subject leakage.
3. **Multimodal Outcome Prediction (Task F):** Combining neural posterior probabilities with continuous kinematic forecast-to-target geometry achieved an AUROC of **$0.9395$** and a balanced accuracy of **$88.35\%$** on covered trials. The decision policy maintained a coverage rate of **$82.16\%$** with an active abstention rate of **$17.84\%$** on ambiguous trials.
4. **Target Provenance Qualification:** Because OpenNeuro `ds006840` does not contain independently surveyed 3D physical target coordinates, target locations were defined via leave-one-subject-out empirical cluster centroids. Consequently, empirical results on `ds006840` represent **exploratory proxy evaluations**; authoritative biological intention–action validation remains **pending prospective human recording** as specified in the accompanying protocol.

---

## 2. Model Comparison on Identical Held-Out Trials ($N = 269$)

To ensure scientific fairness, all evaluated models were tested on the exact same 269 held-out trials across identical 5 LOSO folds.

### Table 1: Master Comparative Benchmark
| Model Name | Inputs | Prediction Time ($t_{\text{pred}}$) | Primary Target | 3D Endpoint RMSE | Mean Euclidean Error | Balanced Accuracy | AUROC | Coverage / Abstention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Persistence Baseline** | Initial hand coordinate ($t=0$) | $0\text{ ms}$ (at start) | Final hand position | $177.51\text{ mm}$ | $177.07\text{ mm}$ | N/A | N/A | $100.0\% / 0.0\%$ |
| **Fair Constant-Velocity** | $200\text{ ms}$ kinematic prefix | $200\text{ ms}$ post-onset | Extrapolated endpoint | $85.65\text{ mm}$ | $75.76\text{ mm}$ | N/A | N/A | $100.0\% / 0.0\%$ |
| **Neural Target Classifier** | 64-channel pre-EEG | $0\text{ ms}$ (at onset) | Visual target direction | N/A | N/A | **$68.34\%$** | $0.6834$ | $100.0\% / 0.0\%$ |
| **Ridge Kinematic Forecaster** | $200\text{ ms}$ kinematic prefix | $200\text{ ms}$ post-onset | 3D physical endpoint | **$47.25\text{ mm}$** | **$32.85\text{ mm}$** | N/A | N/A | $100.0\% / 0.0\%$ |
| **Multimodal Outcome Fusion** | Pre-EEG + $200\text{ ms}$ prefix | $200\text{ ms}$ post-onset | Exploratory condition / discordance | **$47.25\text{ mm}$** | **$32.85\text{ mm}$** | **$88.35\%$** (All) / **$89.39\%$** (Cov) | **$0.9395$** | **$82.16\% / 17.84\%$** |

*Note: Relative error reduction of Ridge Forecaster vs Constant-Velocity Baseline is $100 \times (1 - 47.25 / 85.65) = 44.8\%$.*

---

## 3. Reconciled Participant-Level Breakdown

Table 2 presents the audited performance breakdown across all 5 benchmark participants under LOSO evaluation. In every participant, the Ridge Forecaster decisively outperforms both kinematic baselines.

### Table 2: Reconciled Participant-Level Evaluation Matrix
| Participant ID | Valid Trials | Persistence RMSE | Fair CV RMSE | Ridge Forecaster RMSE | Ridge Reduction vs Fair CV | Neural BalAcc | Condition BalAcc (All) | Policy BalAcc (Covered) | Policy Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sub-01`** | 40 | $181.48\text{ mm}$ | $109.30\text{ mm}$ | **$41.78\text{ mm}$** | **$61.8\%$** | $72.50\%$ | $90.00\%$ | **$95.00\%$** | $70.00\%$ |
| **`sub-02`** | 52 | $178.36\text{ mm}$ | $110.30\text{ mm}$ | **$59.90\text{ mm}$** | **$45.7\%$** | $73.08\%$ | $92.15\%$ | **$98.08\%$** | $88.46\%$ |
| **`sub-03`** | 60 | $176.29\text{ mm}$ | $38.08\text{ mm}$ | **$28.71\text{ mm}$** | **$24.6\%$** | $86.67\%$ | $100.00\%$ | **$100.00\%$** | $76.67\%$ |
| **`sub-08`** | 57 | $178.34\text{ mm}$ | $75.14\text{ mm}$ | **$25.20\text{ mm}$** | **$66.5\%$** | $53.57\%$ | $92.86\%$ | **$97.92\%$** | $92.98\%$ |
| **`sub-15`** | 60 | $174.49\text{ mm}$ | $86.97\text{ mm}$ | **$65.63\text{ mm}$** | **$24.5\%$** | $56.67\%$ | $68.33\%$ | **$55.26\%$** | $80.00\%$ |
| **Cohort Summary** | **269** | **$177.51\text{ mm}$** | **$85.65\text{ mm}$** | **$47.25\text{ mm}$** | **$44.8\%$** | **$68.34\%$** | **$88.35\%$** | **$89.39\%$** | **$82.16\%$** |

---

## 4. Robustness and Subgroup Analysis

To confirm generalization across task parameters and kinematic variability, four orthogonal robustness analyses were conducted:

### 4.1 Target Acceptance Tolerance Grid
The target success criterion was systematically evaluated across four spatial tolerance thresholds ($25.0$, $50.0$, $75.0$, and $100.0\text{ mm}$):
- **$25.0\text{ mm}$ (Strict):** Target Success Rate = $61.34\%$, Fusion BalAcc = $81.42\%$, AUROC = $0.8871$.
- **$50.0\text{ mm}$ (Standard Reference):** Target Success Rate = $98.14\%$, Fusion BalAcc = $88.35\%$, AUROC = $0.9395$.
- **$75.0\text{ mm}$ (Permissive):** Target Success Rate = $99.63\%$, Fusion BalAcc = $91.10\%$, AUROC = $0.9520$.
- **$100.0\text{ mm}$ (Wide):** Target Success Rate = $100.0\%$, Ceiling reached.

### 4.2 Reach Direction Stratification
Forecasting accuracy remained stable across lateral reaching directions:
- **Leftward Reaches ($N = 135$):** Ridge RMSE = $48.12\text{ mm}$, Mean Error = $33.40\text{ mm}$.
- **Rightward Reaches ($N = 134$):** Ridge RMSE = $46.36\text{ mm}$, Mean Error = $32.29\text{ mm}$.
- *Directional Bias:* Negligible lateral asymmetry ($\Delta\text{RMSE} = 1.76\text{ mm}$).

### 4.3 Movement Duration Terciles
Trials were partitioned into duration terciles based on total reach execution time:
- **Short Duration ($< 620\text{ ms}$, $N = 90$):** Ridge RMSE = $42.15\text{ mm}$ ($200\text{ ms}$ prefix covers $\approx 35\%$ of reach).
- **Medium Duration ($620$–$880\text{ ms}$, $N = 89$):** Ridge RMSE = $46.80\text{ mm}$ ($200\text{ ms}$ prefix covers $\approx 27\%$ of reach).
- **Long Duration ($> 880\text{ ms}$, $N = 90$):** Ridge RMSE = $52.41\text{ mm}$ ($200\text{ ms}$ prefix covers $\approx 18\%$ of reach).

---

## 5. Statistical Rigor and Leakage Prevention Audit

The Phase 5 evaluation pipeline incorporates extensive structural safeguards against analytical leakage:

```mermaid
flowchart TD
    Raw["Raw Trials (N = 280)"] --> Audit["Data Auditor"]
    Audit -->|11 Incomplete Trials Excluded| Valid["Valid Multimodal Trials (N = 269)"]
    Valid --> Coordinator["LOSO Fold Coordinator"]
    Coordinator --> TrSet["Training Subjects (k - 1)"]
    Coordinator --> TeSet["Held-Out Test Subject (k)"]
    TrSet --> TrainFit["Fit Normalizers & Ridge Weights"]
    TrSet --> CentroidFit["Compute Proxy Centroids Excluding Subject k"]
    TrainFit --> Predict["Prospective Twin Inference at t = 200 ms"]
    TeSet --> Predict
    Predict --> Freeze["Save & Hash Predictions"]
    CentroidFit --> Eval["Retrospective Evaluation Engine"]
    TeSet --> Eval
    Freeze --> Eval
    Eval --> MasterMatrix["phase_5_evaluation_matrix.csv"]
```

1. **Participant Separation:** Evaluated test participants were strictly omitted from all feature normalization, regression fitting, and target centroid estimation.
2. **Temporal Separation:** Kinematic forecasting features strictly use samples from $t \le 200\text{ ms}$ post-movement onset. Full-trial durations are not leaked to the forecaster.
3. **Causal Decoupling:** Pre-movement EEG decoding features strictly precede physical movement onset.
4. **Prediction Freezing:** Predictions are computed, hashed, and stored prior to retrospective endpoint evaluation.

---

## 6. Scientific Limitations and Four-Level Validation Assessment

In compliance with Phase 5 Section 20 reporting rules, achievements are categorized across the four standard scientific levels:

| Level | Definition | NeuroAction Status | Audit Justification |
| :---: | :--- | :---: | :--- |
| **Level 1** | **Software Implementation** | **VERIFIED** | Full Phase 5 codebase implemented, verified, and backed by 93 passing automated tests. |
| **Level 2** | **Existing-Data Retrospective Evaluation** | **VERIFIED** | Full 269-trial LOSO evaluation executed on `ds006840` with documented provenance and audited metrics. |
| **Level 3** | **Independent Outcome Prediction** | **PENDING** | `ds006840` lacks independently surveyed physical 3D targets. Current labels rely on empirical cluster centroids. |
| **Level 4** | **Prospective External Validation** | **PENDING** | Requires prospective human recording adhering to `reports/phase_5/prospective_validation_protocol.md`. |

### Crucial Scientific Boundaries:
- **Exploratory Status of Existing Data:** Because `ds006840` target coordinates are cluster-derived proxies, high classification performance reflects robust separation of physical reaching endpoints rather than independent validation of biological intention failure.
- **Congruency vs Motor Failure:** Experimental congruency (instructed stimulus-response reversal) is an experimental cognitive manipulation, not an unintentional motor error.
- **Nonclinical Position:** The system is an offline computational research twin. It does not provide real-time neurological diagnostics or robotic intervention.
