# NeuroAction Phase 4 — Validation and Verification Audit Report

**Document ID:** `NA-P4-VAL-001`  
**Phase:** 4 — System Validation & Verification  
**Release Audit:** Final Scientific Consistency and Release Audit  
**Date:** October 2026  
**Auditor:** Independent Computational Neuroscience & BCI Verification Suite  
**Final Status:** PASSED (60 / 60 Passing Automated Tests)  
**Scientific Verification Verdict:** **Partially Supported (with audited correction)**  
*(Engineering Integration Verified | Condition Discrimination Supported under LOSO | Biological Discordance Unvalidated | Offline Demonstration Only)*

---

## 1. Scope & Verification Methodology

This validation audit independently examines the code, models, schemas, temporal integrity, data leakage safeguards, conflict detection formulas, and reproducibility artifacts of Phase 4:

**Target Package:** `src/phase_4/`  
**Master Runner:** `src/run_phase_4.py`  
**Dashboard & CLI:** `src/phase_4/dashboard.py`, `src/phase_4/cli.py`  
**Test Suite:** `tests/test_phase_4.py`, `tests/test_phase_2.py`, `tests/test_phase_3.py`, `tests/test_validation.py`  
**Benchmark Dataset:** OpenNeuro `ds006840` (IACKD: *Intention–Action Conflict Kinematics Dataset*, Nature *Scientific Data* 2026, Version 1.0.0, DOI: `10.18112/openneuro.ds006840.v1.0.0`)

---

## 2. Test Execution Summary

The complete regression test suite was executed in the project virtual environment with Python 3.10.18:

```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

### Official Pytest Execution Log:
```
============================= test session starts ==============================
platform linux -- Python 3.10.18, pytest-9.1.1, pluggy-1.6.0
rootdir: /run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement
plugins: platformdirs-4.12.3
collected 60 items

tests/test_phase_2.py .......                                            [ 11%]
tests/test_phase_3.py .............                                      [ 33%]
tests/test_phase_4.py .............................                      [ 81%]
tests/test_validation.py ...........                                     [100%]

============================== 60 passed in 10.48s =============================
```

- **Phase 2 Tests:** 7 / 7 PASSED (Decoders, LOSO cross-validation, zero data leakage, NCP detection).
- **Phase 3 Tests:** 13 / 13 PASSED (Kinematic segmentation, dual protocols, Euclidean RMSE, held-out model loading).
- **Phase 4 Tests:** 29 / 29 PASSED (Data schemas, model adapters, 15 synthetic conflict cases, temporal anti-leakage, replay repeatability, held-out model provenance across all 5 subjects, fold-specific target proxy isolation, evaluation matrix audit, leakage-free baseline duration estimation).
- **Validation Tests:** 11 / 11 PASSED (Manifest schema, OpenNeuro HDF5 reader, CanonicalTrial integrity).

---

## 3. Detailed Verification Audit by Category

### Category A: Data Schemas & Data Contracts (PASSED)
- **Contract Verification:** All dataclasses (`TrialMetadata`, `NeuralIntentionResult`, `KinematicObservation`, `TrajectoryForecast`, `ConflictAssessment`, `FusionState`, `ReplayFrame`) serialize to Python dictionary and JSON structures without truncation or data corruption.
- **Null Safety:** Missing values (e.g., tracking dropouts, unmapped targets) remain explicitly `None` rather than being imputed with misleading defaults.
- **JSON Sanitization:** Non-finite floats (NaN/Inf) are cleanly sanitized to `null` before serialization (`sanitize_for_json`), completely resolving Canvas freezing in the web dashboard.

### Category B: Neural Intention Adapter & Held-Out Provenance (PASSED)
- **Model Checkpoints:** Verified presence and independent loading of held-out model checkpoints for all 5 subjects:
  - `models/phase_2/lda_goal_decoder_held_out_sub01.pkl`
  - `models/phase_2/lda_goal_decoder_held_out_sub02.pkl`
  - `models/phase_2/lda_goal_decoder_held_out_sub03.pkl`
  - `models/phase_2/lda_goal_decoder_held_out_sub08.pkl`
  - `models/phase_2/lda_goal_decoder_held_out_sub15.pkl`
- **Zero Leakage:** In batch evaluation, each participant's trials are evaluated strictly by their corresponding held-out model with `is_held_out = True`. Full-cohort models are never used for held-out scoring.
- **Label Semantics:** Decoder predicts `target_goal` (`ball_dir`), which corresponds to the **instructed visual stimulus target location** on screen, not an unconstrained internal intention.
- **Acausal Offline Filtering:** Pipeline explicitly documents that zero-phase Butterworth bandpass filtering is acausal, precluding direct streaming deployment without causal filter conversion.

### Category C: Kinematic Trajectory Adapter & Held-Out Provenance (PASSED)
- **Model Checkpoints:** Verified presence and independent loading of held-out forecaster checkpoints for all 5 subjects:
  - `models/phase_3/kinematic_forecaster_held_out_sub01.pkl`
  - `models/phase_3/kinematic_forecaster_held_out_sub02.pkl`
  - `models/phase_3/kinematic_forecaster_held_out_sub03.pkl`
  - `models/phase_3/kinematic_forecaster_held_out_sub08.pkl`
  - `models/phase_3/kinematic_forecaster_held_out_sub15.pkl`
- **Dual Protocols:** Verified execution under both Protocol A (`event_locked`) and Protocol B (`kinematic_onset`).
- **Prefix Requirement:** Trajectory forecaster strictly enforces a 20-sample prefix ($200$ ms at 100 Hz). Incomplete prefixes result in safe `status = "UNAVAILABLE"` returns.

### Category D: Target Coordinates & Empirical Proxy Isolation (PASSED)
- **Empirical Proxy Classification:** Acknowledged that OpenNeuro `ds006840` derivative files contain no independent 3D physical coordinates; reaching centroids are empirical endpoint-derived proxies.
- **Fold-Specific Proxy Isolation:** Target centroids are computed strictly from the 4 training subjects in each fold (`estimate_fold_target_centroids`), completely excluding the evaluated participant's reaches.
- **Stability:** Across all 5 folds, proxy centroids are stable within $\pm 0.2\text{ mm}$ lateral.
- **Frame & Unit Guard:** Calling conflict analysis with mismatched units (e.g., `cm` vs `mm`) or frames (e.g., `vicon` vs `leap_motion_cartesian`) returns `INVALID_INPUT`.

### Category E: Conflict Engine Synthetic Cases (PASSED — 15 / 15 Cases)
All 15 synthetic unit test cases stipulated in the specification were evaluated and passed:

| Test Case | Condition Description | Expected State | Actual Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Case 1** | Predicted endpoint equals target ($d=0$, $\theta=0$) | `ALIGNED` | `ALIGNED`, $d=0.0$ mm, $\theta=0.0^\circ$ | **PASSED** |
| **Case 2** | Endpoint within tolerance ($d=30$ mm $\le 50$ mm) | `ALIGNED` | `ALIGNED`, $d=30.0$ mm | **PASSED** |
| **Case 3** | Endpoint exceeds tolerance ($d=80$ mm $> 50$ mm) | `POTENTIAL_DISAGREEMENT` | `POTENTIAL_DISAGREEMENT`, $d=80.0$ mm | **PASSED** |
| **Case 4** | Direction agrees with target ($\theta < 5^\circ$) | `ALIGNED` | `ALIGNED`, $\theta = 0.0^\circ$ | **PASSED** |
| **Case 5** | Direction opposes target ($\theta > 140^\circ$) | `POTENTIAL_DISAGREEMENT` | `POTENTIAL_DISAGREEMENT`, $\theta = 180.0^\circ$ | **PASSED** |
| **Case 6** | Zero/near-zero vector magnitude | `UNCERTAIN` | `UNCERTAIN` (Safe zero-division guard) | **PASSED** |
| **Case 7** | Target coordinates missing / unmapped | `UNAVAILABLE` | `UNAVAILABLE` | **PASSED** |
| **Case 8** | Coordinate frames differ | `INVALID_INPUT` | `INVALID_INPUT` | **PASSED** |
| **Case 9** | Coordinate units differ | `INVALID_INPUT` | `INVALID_INPUT` | **PASSED** |
| **Case 10** | Timestamp alignment leakage guard | Rejected | Leakage flagged ($p_{\text{time}} < d_{\text{time}}$) | **PASSED** |
| **Case 11** | Goal label is missing | `UNAVAILABLE` | `UNAVAILABLE` | **PASSED** |
| **Case 12** | Neural & kinematic trial IDs differ | `INVALID_INPUT` | `INVALID_INPUT` | **PASSED** |
| **Case 13** | Non-finite coordinates (NaN or Inf) | `INVALID_INPUT` | `INVALID_INPUT` | **PASSED** |
| **Case 14** | Missing samples / tracking dropouts | `UNAVAILABLE` | `UNAVAILABLE` | **PASSED** |
| **Case 15** | Borderline classifier confidence ($< 0.55$) | `UNCERTAIN` | `UNCERTAIN` | **PASSED** |

### Category F: Temporal Anti-Leakage & Replay Repeatability (PASSED)
- **Anti-Leakage Verification:** In test `test_temporal_anti_leakage_on_altered_future`, future trajectory samples at $t > 500$ ms were mutated to extreme values ($\pm 9999.0$ mm). Replay evaluation at $t = 200$ ms produced bit-identical past positions and conflict states (`equal_nan=True`).
- **Deterministic Repeatability:** Running the replay stream twice on trial `sub-01_run-01_L1_tr00` yielded 141 identical frames with exact timestamp matches.

---

## 4. Empirical Evaluation Matrix Audit (Strict LOSO)

The master batch run evaluated all 269 trials across subjects `sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15` under strict LOSO cross-validation:
- Total rows generated in `reports/phase_4/audited_evaluation_matrix.csv`: **538 rows** (269 trials $\times$ 2 protocols).
- Explicit provenance tracked per row (`neural_model_provenance`, `kinematic_model_provenance`, `target_proxy_provenance`, `is_held_out=True`).
- **Audited Confusion Matrix Summary (`reports/phase_4/conflict_confusion_matrix.csv`, Protocol B, N=269):**
  - **Sensitivity / Recall on Incongruent Trials:** **$73.53\%$** (100 / 136 detected)
  - **Specificity on Congruent Trials:** **$59.40\%$** (79 / 133 cleared)
  - **Precision:** **$64.94\%$** (100 / 154)
  - **F1-Score:** **$0.6897$**
  - **Condition Discrimination Balanced Accuracy:** **$66.46\%$** (closely tracks Phase 2 neural goal balanced accuracy)
  - **Conflict Score AUROC:** **$0.6834$** (computed from continuous normalized distance score $c_{\text{norm}}$)
  - **Phase 2 Neural Goal Balanced Accuracy (LOSO):** **$68.34\%$**
  - **Phase 3 Kinematic Endpoint RMSE (LOSO Cohort):** **$47.25\text{ mm}$** (vs. Fair Constant Velocity Baseline $85.65\text{ mm}$, $44.8\%$ error reduction; Persistence Baseline $177.51\text{ mm}$, $73.4\%$ error reduction)

#### Per-Participant Held-Out Kinematic Generalization (Full-Endpoint Prediction)
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) immediately following movement onset ($t_{\text{onset}}$).
- **Forecast Target:** The final physical reach endpoint $\mathbf{p}_{\text{endpoint}} = [x_{\text{end}}, y_{\text{end}}, z_{\text{end}}]^T$ at trial completion.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE across held-out test trials, with error reduction calculated consistently as:
  $$\text{Error Reduction} = 100 \times \left(1 - \frac{\text{RMSE}_{\text{Ridge}}}{\text{RMSE}_{\text{Fair CV}}}\right)$$

| Participant | Held-Out Trials | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`sub-01`** | 40 | **$41.78\text{ mm}$** | $109.30\text{ mm}$ | $181.48\text{ mm}$ | **$61.8\%$** |
| **`sub-02`** | 52 | **$59.90\text{ mm}$** | $110.30\text{ mm}$ | $178.36\text{ mm}$ | **$45.7\%$** |
| **`sub-03`** | 60 | **$28.71\text{ mm}$** | $38.08\text{ mm}$ | $176.29\text{ mm}$ | **$24.6\%$** |
| **`sub-08`** | 57 | **$25.20\text{ mm}$** | $75.14\text{ mm}$ | $178.34\text{ mm}$ | **$66.5\%$** |
| **`sub-15`** | 60 | **$65.63\text{ mm}$** | $86.97\text{ mm}$ | $174.49\text{ mm}$ | **$24.5\%$** |
| **Cohort** | **269** | **$47.25\text{ mm}$** | **$85.65\text{ mm}$** | **$177.51\text{ mm}$** | **$44.8\%$** |

#### Predefined Fixed-Horizon Comparisons ($N=269$ Held-Out Trials, Zero Duration Estimation Needed)
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) post-onset.
- **Forecast Horizon:** Fixed elapsed future time $H \in \{100, 200, 300, 500\}\text{ ms}$ post-prefix.
- **Target Definition:** Exact intermediate 3D hand position $\mathbf{p}(t_{\text{prefix\_end}} + H)$ along executed Leap Motion reach trajectory.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE and relative error reduction $100 \times (1 - \text{RMSE}_{\text{Ridge}} / \text{RMSE}_{\text{Fair CV}})$:

| Predefined Horizon | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV |
| :---: | :---: | :---: | :---: | :---: |
| **$H = 100\text{ ms}$** | **$1.44\text{ mm}$** | $3.94\text{ mm}$ | $8.82\text{ mm}$ | **$63.5\%$** |
| **$H = 200\text{ ms}$** | **$4.27\text{ mm}$** | $8.80\text{ mm}$ | $18.44\text{ mm}$ | **$51.5\%$** |
| **$H = 300\text{ ms}$** | **$7.81\text{ mm}$** | $14.08\text{ mm}$ | $28.47\text{ mm}$ | **$44.5\%$** |
| **$H = 500\text{ ms}$** | **$16.06\text{ mm}$** | $24.83\text{ mm}$ | $48.75\text{ mm}$ | **$35.3\%$** |

---

## 5. Artifact & Demonstration Verification

| Deliverable | Location | Description | Audit Status |
| :--- | :--- | :--- | :--- |
| **Scientific Audit Report** | `reports/phase_4/phase_4_scientific_audit.md` | Formal 10-point audit document | **Created / Approved** |
| **Technical Report** | `reports/phase_4/phase_4_report.md` | Full technical report updated with audited LOSO metrics | **Updated & Verified** |
| **Validation Report** | `reports/phase_4/phase_4_validation_report.md` | This document | **Updated & Verified** |
| **Audited Evaluation Matrix** | `reports/phase_4/audited_evaluation_matrix.csv` | 538 rows with full per-row provenance | **Verified (Strict LOSO)** |
| **Batch Evaluation Matrix** | `reports/phase_4/digital_twin_eval_matrix.csv` | Full batch matrix | **Verified (Strict LOSO)** |
| **Confusion Matrix** | `reports/phase_4/conflict_confusion_matrix.csv` | Summary metrics under Protocol B | **Verified** |
| **Threshold Sensitivity Grid** | `reports/phase_4/threshold_sensitivity_analysis.csv` | 12-cell parameter grid ($d \in [25, 100]$, $\theta \in [30, 60]$) | **Verified** |
| **Figure 9 (Architecture)** | `reports/phase_4/figures/figure_9_multimodal_architecture.png` | System architecture diagram | **Verified** |
| **Figure 10 (ROC / Disagreement)**| `reports/phase_4/figures/figure_10_conflict_detection_analysis.png` | Empirical conflict distribution and ROC | **Regenerated** |
| **Figure 11 (Demo Snapshot)** | `reports/phase_4/figures/figure_11_demo_trial_replay.png` | 6-panel research dashboard snapshot | **Verified** |
| **Demo Replay Event Log** | `reports/phase_4/demo/demo_replay_event_log.csv` | Step-by-step causal replay telemetry | **Verified** |
| **Demo Summary JSON** | `reports/phase_4/demo/demo_trial_summary.json` | Complete state dictionary export | **Verified** |
| **Reproducibility Manifest** | `reports/phase_4/demo/reproducibility_manifest.json` | Model artifact hashes, parameters, OS info | **Verified** |

---

## 6. Official Release Status Specification

### 6.1 What is Empirically Supported:
- Pre-movement scalp EEG decodes instructed visual target location above chance under strict LOSO (**68.34% balanced accuracy**).
- Trajectory forecaster predicts 3D Cartesian endpoints with **47.25 mm RMSE** under strict LOSO, substantially beating persistence ($177.51$ mm) and fair constant velocity ($85.65$ mm) baselines.
- Multimodal congruent-versus-incongruent condition discrimination achieves **66.46% balanced accuracy** and **0.6834 AUROC** under strict LOSO.

### 6.2 What is Software-Verified:
- 60 / 60 regression and unit tests passing across all four project phases.
- Complete causal separation in replay engine; deterministic bit-identical frames across runs.
- Interactive web dashboard (port 8050) with NaN-safe Canvas rendering.
- All model artifacts, scaler parameters, and target proxies are verified as 100% held-out per fold.

### 6.3 What Remains Exploratory:
- Conflict thresholds ($d = 50.0$ mm, $\theta = 45.0^\circ$) represent illustrative geometric boundaries.
- Normalized score $c_{\text{norm}}$ is an engineering indicator, not a calibrated biological error probability.

### 6.4 What the Dataset Does Not Establish:
- The dataset does not provide an independent ground truth for uninstructed biological intention errors or motor slips.
- The dataset does not contain independent 3D Cartesian coordinates for reaching targets (only screen pixels).

### 6.5 Additional Measurements Needed for Full Biological Discordance Validation:
- Direct motion capture calibration of 3D physical target objects.
- Experimental paradigms with uninstructed target perturbations or slip-inducing force fields to elicit spontaneous intention–action discordance.
- Causal online IIR/FIR EEG filtering evaluated in live closed-loop streaming.

---

## 7. Audit Conclusion & Final Release Verdict

### Final Scientific Verdict: **Partially Supported (with audited correction)**

**Summary Assessment:**
1. **Engineering Integration:** **VERIFIED** — Modular architecture, adapters, schemas, replay simulator, and interactive dashboard are functionally robust and fully tested.
2. **Scientific Evaluation:** **PARTIALLY SUPPORTED** — Multimodal congruent-versus-incongruent condition discrimination is validated under strict LOSO at **66.46% balanced accuracy** and **0.6834 AUROC**.
3. **Biological Intention–Action Conflict Detection:** **NOT VALIDATED** — The task paradigm evaluates instructed reversal, not spontaneous neurological intention discordance.
4. **Timing Characterization:** **OFFLINE DEMONSTRATION ONLY** — Pre-movement EEG filtering is acausal; forecaster requires 200 ms movement prefix.
