# NeuroAction Phase 4 — Independent Scientific and Statistical Audit

**Document ID:** `NA-P4-AUD-001`  
**Phase:** 4 — System Validation & Scientific Revalidation  
**Release Audit:** Final Scientific Consistency and Release Audit  
**Date:** October 2026  
**Auditor:** Independent Computational Neuroscience, BCI, and Statistical Inference Audit  
**Status:** Complete — Audited, Verified & Release-Ready  
**Final Release Verdict:** **Partially Supported (with audited correction)**  
*(Engineering Integration Verified | Condition Discrimination Partially Supported | Biological Intention–Action Discordance Unvalidated | Offline Demonstration Only)*

---

## Executive Summary & Audit Overview

This document presents the independent scientific and statistical audit of **Phase 4** of the NeuroAction initiative (*Neural Intention–Action Digital Twin*).

The primary research question addressed in Phase 4 is:
> *Can an integrated neural-kinematic digital twin detect intention–action discordance early in movement initiation, simulate deterministic trial execution without data leakage, and provide an interpretable research interface for human-in-the-loop validation?*

In the initial Phase 4 implementation report (`NA-P4-REP-001`), an empirical conflict detection performance of **95.51% balanced accuracy** and **0.9951 AUROC** was claimed across 269 trials. This audit was convened to independently verify dataset identity, label semantics, data leakage safeguards, and statistical inference.

### Key Audit Findings & Resolutions:
1. **Dataset Identity Resolved:** The underlying benchmark dataset is definitively confirmed as **OpenNeuro `ds006840`** (IACKD: *Intention–Action Conflict Kinematics Dataset*, Nature *Scientific Data* 2026, Version 1.0.0, DOI: `10.18112/openneuro.ds006840.v1.0.0`). The typographical reference to `ds003612` in preliminary audit notes has been retired.
2. **Phase 2 Label Semantics Clarified:** In the pipeline's canonical data schema, `target_goal` is derived directly as `ball_dir` (the visual screen stimulus position). In congruent trials (Red cue), stimulus position coincides with physical reach direction (`ball_dir == hand_dir`). In incongruent trials (Yellow cue), the task rule instructs stimulus-response reversal, making physical hand reach direction opposite to visual stimulus direction (`hand_dir != ball_dir`). Phase 2 EEG models decode the **instructed visual stimulus target location**, not an unconstrained, spontaneous internal intention.
3. **Resolution of Condition Conflation:** The ground-truth reference for "conflict" was defined as the experimental condition ($y = \text{incongruent}$). Consequently, the classifier evaluated **congruent-versus-incongruent experimental-condition discrimination**, not an independent biological intention–action discordance outcome.
4. **Correction of In-Sample Model Leakage:** In the initial batch execution, while participant `sub-01` was evaluated with a held-out model, participants `sub-02` through `sub-15` silently evaluated predictions from full-cohort models (`lda_goal_decoder.pkl` and `kinematic_forecaster.pkl`), creating in-sample leakage that artificially inflated conflict detection metrics to ~95.5%.
   - **Remediation:** Strictly held-out checkpoints were trained and persisted for all 5 subjects (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`) for both Phase 2 and Phase 3. Every prediction in `reports/phase_4/audited_evaluation_matrix.csv` is now verified as 100% held-out.
5. **Target Coordinates are Empirical Proxies:** The OpenNeuro `ds006840` derivatives contain no independent 3D physical coordinates for reaching targets (only 2D screen stimulus pixels `ball_x_pix`). Target centroids ($\approx \pm 188$ mm lateral) are **empirical endpoint-derived proxies**. Fold-specific target proxies are estimated strictly from training subjects in each fold ($s \ne s_{\text{held\_out}}$), completely eliminating target-coordinate leakage.
6. **Audited Strict LOSO Performance (Protocol B, $N = 269$ trials):**
   - **Phase 2 Neural Goal Balanced Accuracy (LOSO):** **68.34%** (matches Phase 2 audited findings).
   - **Phase 3 Kinematic Endpoint RMSE (LOSO):** **47.25 mm** (Fair CV Baseline = $85.65$ mm, Persistence Baseline = $177.51$ mm; forecaster achieves $44.8\%$ error reduction over CV and $73.4\%$ over persistence).
   - **Condition Sensitivity / Recall (Incongruent):** **73.53%** (100 / 136 detected)
   - **Condition Specificity (Congruent):** **59.40%** (79 / 133 cleared)
   - **Precision:** **64.94%** (100 / 154)
   - **F1-Score:** **0.6897**
   - **Condition Discrimination Balanced Accuracy:** **66.46%**
   - **Conflict Score AUROC (from continuous $c_{\text{norm}}$):** **0.6834**
7. **Observed Error Relationship:** Because the Phase 3 kinematic forecaster reliably predicts reach direction (47.25 mm RMSE vs. ~375 mm target separation), conflict classification decisions directly track the accuracy of the upstream pre-movement neural goal decoder (68.34%).

---

## 1. Resolution of Dataset Identity

### 1.1 Empirical Evidence from Repository Artifacts
A thorough audit of source files, catalogs, manifests, and data caches confirms the true dataset identity:
- **Remote Catalog (`data/manifests/openneuro_remote_catalog.json`):** Contains 1,551 lines with authoritative AWS S3 URLs:
  `https://s3.amazonaws.com/openneuro.org/ds006840/derivatives/...`
- **Benchmark Manifest (`data/manifests/iackd_manifest.csv`):** Contains trial metadata corresponding to the 15 participants of IACKD.
- **Data Caches (`data/cache/`):** Contains derivatives for `sub-01`, `sub-02`, `sub-03`, `sub-08`, and `sub-15` downloaded from `ds006840`.
- **Phase 1, 2, and 3 Reports:** Consistently identify the dataset as IACKD (OpenNeuro `ds006840`, Version 1.0.0, DOI: `10.18112/openneuro.ds006840.v1.0.0`).

### 1.2 Retiring the Erroneous Identifier
The identifier `ds003612` appeared only in exploratory review notes during the recent Phase 4 audit as an accidental typographic substitution with another OpenNeuro BCI dataset. It has been completely corrected and replaced with `ds006840` across all files.

---

## 2. Verification of Phase 2 Label Semantics

### 2.1 Provenance of Trial Labels
In `src/data/trial_loader.py` and `src/data/schema.py`, experimental semantics are derived from the raw MATLAB derivative variable `label_code`:

```python
# From src/data/trial_loader.py lines 26-29:
color = "red" if label_code in (0, 1) else "yellow"
ball_dir = "left" if label_code in (0, 2) else "right"
hand_dir = "left" if label_code in (0, 3) else "right"
congruency = "congruent" if label_code in (0, 1) else "incongruent"
```

In `CanonicalTrial`, `target_goal` is assigned `ball_dir`:
- `target_goal`: "left" or "right" (derived from visual ball stimulus location).
- `physical_hand_direction`: "left" or "right" (physical motor execution direction).
- `ball_color`: "red" (instructs direct reaching) or "yellow" (instructs stimulus-response reversal).
- `congruency`: "congruent" (red cue) or "incongruent" (yellow cue).

### 2.2 Semantic Relationship Across Conditions

| Condition | Visual Cue | `ball_dir` (`target_goal`) | Instructed Rule | `physical_hand_direction` | Relationship |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Congruent ($N=133$)** | Red | Left ($x < 0$) | Reach toward cue | Left ($x < 0$) | Goal == Physical Hand |
| **Congruent ($N=133$)** | Red | Right ($x > 0$) | Reach toward cue | Right ($x > 0$) | Goal == Physical Hand |
| **Incongruent ($N=136$)** | Yellow | Left ($x < 0$) | Reach opposite to cue | Right ($x > 0$) | Goal $\ne$ Physical Hand |
| **Incongruent ($N=136$)** | Yellow | Right ($x > 0$) | Reach opposite to cue | Left ($x < 0$) | Goal $\ne$ Physical Hand |

### 2.3 Scientific Implication for Neural Decoding
What does the Phase 2 classifier predict?
- The classifier is trained on $y = (\text{target\_goal} == \text{"right"})$.
- Because `target_goal` is operationalized as `ball_dir`, the classifier predicts the **instructed visual stimulus target location** from pre-movement scalp EEG.
- In incongruent trials, the stimulus location and the physical movement direction are opposites.
- Therefore, the decoded signal represents the spatial location of the visual target stimulus under the experimental task rules. It cannot be characterized as an unconstrained biological intention independent of the sensory cue.

---

## 3. Verification of the Leave-One-Subject-Out (LOSO) Pipeline

### 3.1 Trace of Preprocessing and Modeling Steps
To verify that evaluation on held-out participants is strictly independent and free of cross-subject leakage:

1. **Fixed Per-Trial Operations:**
   - Pre-movement EEG window slicing: strictly $[-1000, 0]$ ms relative to movement onset ($t = 0$ ms).
   - Feature extraction (`extract_trial_features`): computes channel-wise bandpowers (delta, theta, alpha, beta, gamma) and temporal statistics. This operation is computed trial-by-trial with zero inter-trial or inter-subject dependencies.
2. **Learned Transformation Parameters (`StandardScaler`):**
   - In `src/models/decoders.py`, `LDADecoder.pipe` wraps `StandardScaler` and `LinearDiscriminantAnalysis`.
   - The scaler's mean ($\mu$) and standard deviation ($\sigma$) are fitted strictly on training subjects in each fold.
   - Empirical verification confirmed distinct scaler parameters across folds:
     - `sub-01` held-out training set scaler mean sum: $10,534.57$
     - `sub-02` held-out training set scaler mean sum: $9,918.45$
     - `sub-03` held-out training set scaler mean sum: $11,544.09$
     - `sub-08` held-out training set scaler mean sum: $5,613.46$
     - `sub-15` held-out training set scaler mean sum: $10,992.56$
3. **Phase 3 Kinematic Forecaster Parameters:**
   - In `src/trajectory/models.py`, `RidgeTrajectoryForecaster` fits a scaler and multi-output Ridge regression model.
   - Scaler mean sums and Ridge coefficients differ across all 5 folds (e.g., `sub-01` scaler mean sum = $1041.19$, `sub-02` = $1066.16$, `sub-15` = $1021.19$).
4. **Fold-Specific Target Proxy Estimation:**
   - Reaching target centroids are calculated strictly from the 4 training subjects in each fold using `estimate_fold_target_centroids()`.
   - The held-out participant's reaching endpoints are completely excluded from centroid estimation.
5. **Per-Row Provenance Tracking:**
   - Every evaluated row in `reports/phase_4/audited_evaluation_matrix.csv` explicitly records:
     - `neural_model_provenance`: e.g., `lda_held_out_sub01`
     - `kinematic_model_provenance`: e.g., `ridge_held_out_sub01`
     - `target_proxy_provenance`: `training_subs_excluding_sub-01`
     - `is_held_out = True`

Zero in-sample evaluations or shared statistics cross the fold boundary.

---

## 4. Verification of Conflict-Score Meaning

### 4.1 Scoring Formula and Operational Mechanics
In `src/phase_4/conflict_engine.py`:
1. The decoded neural goal $\hat{g} \in \{\text{left}, \text{right}\}$ selects the fold-specific target proxy $\mathbf{p}_{\text{target}}(\hat{g})$.
2. The kinematic forecaster predicts hand endpoint $\hat{\mathbf{p}}_{\text{end}}$ from the first 200 ms movement prefix.
3. **Endpoint-to-Target Distance:**
   $$d_{\text{target}} = \|\hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{target}}(\hat{g})\|_2 \quad (\text{mm})$$
4. **Normalized Conflict Score:**
   $$c_{\text{norm}} = \min\left(1.0, \frac{d_{\text{target}}}{d_{\text{reference}}}\right), \quad d_{\text{reference}} = 100.0 \text{ mm}$$
5. **Directional Disagreement Angle:**
   $$\theta_{\text{dir}} = \arccos\left(\text{clip}\left(\frac{\mathbf{v}_{\text{pred}} \cdot \mathbf{v}_{\text{target}}}{\|\mathbf{v}_{\text{pred}}\| \|\mathbf{v}_{\text{target}}\| + \epsilon}, -1.0, 1.0\right)\right) \quad (^\circ)$$
6. **Discrete Conflict State:**
   - If $d_{\text{target}} > 50.0$ mm or $\theta_{\text{dir}} > 45.0^\circ \implies$ `POTENTIAL_DISAGREEMENT` (`is_conflict = True`).
   - Otherwise $\implies$ `ALIGNED` (`is_conflict = False`).

### 4.2 What the Evaluation Measures
In the benchmark evaluation, the reference label is:
$$y = \begin{cases} 1 & \text{if } \text{congruency} == \text{"incongruent"} \\ 0 & \text{if } \text{congruency} == \text{"congruent"} \end{cases}$$

Therefore, this evaluation measures **congruent-versus-incongruent experimental-condition discrimination**. It demonstrates how well a multimodal system combining EEG decoding and trajectory forecasting can determine whether a trial involved direct reaching or stimulus-response reversal.

It does **not** evaluate spontaneous, uninstructed biological intention errors (such as reaching for the wrong cup or experiencing a motor slip), because all participants in `ds006840` followed instructed experimental reversal rules.

### 4.3 Precise Description of the Observed Error Relationship
In prior un-audited notes, it was loosely claimed that conflict detection is "mathematically bounded" by neural decoding accuracy. A rigorous analysis establishes the exact empirical mechanism:

In this binary lateral reaching task, physical targets are separated by $\approx 375$ mm. The kinematic forecaster has an endpoint RMSE of 47.25 mm and correctly identifies the physical reach direction (left vs. right) in $> 96\%$ of trials.
Consequently:
- When the pre-movement neural decoder predicts $\hat{g} = g_{\text{true}}$, the conflict engine correctly classifies the condition ($d_{\text{target}} \approx 10$ mm on congruent trials $\rightarrow$ Aligned; $d_{\text{target}} \approx 375$ mm on incongruent trials $\rightarrow$ Disagreement).
- When the neural decoder predicts $\hat{g} \ne g_{\text{true}}$ (a $31.7\%$ error rate under LOSO), it maps the target proxy to the opposite side of workspace. This flips the decision: congruent trials appear discordant ($d_{\text{target}} \approx 375$ mm $\rightarrow$ False Positive), and incongruent trials appear aligned ($d_{\text{target}} \approx 10$ mm $\rightarrow$ False Negative).

Empirically, this causes the balanced accuracy of condition discrimination (**66.46%**) to directly mirror the balanced accuracy of the pre-movement neural decoder (**68.34%**) across all 5 subjects:
- `sub-01`: Conflict BalAcc = $72.50\%$, Goal BalAcc = $72.50\%$
- `sub-02`: Conflict BalAcc = $71.19\%$, Goal BalAcc = $73.08\%$
- `sub-03`: Conflict BalAcc = $85.00\%$, Goal BalAcc = $86.67\%$
- `sub-08`: Conflict BalAcc = $54.37\%$, Goal BalAcc = $53.57\%$
- `sub-15`: Conflict BalAcc = $51.67\%$, Goal BalAcc = $56.67\%$

This close correspondence is an empirical consequence of decoder error propagation in a binary choice task, not an abstract universal theorem.

---

## 5. Numerical Results and Verification

All metrics were recalculated directly from `reports/phase_4/audited_evaluation_matrix.csv` under Protocol B (Kinematic Onset, $N = 269$ trials):

### 5.1 Confusion Matrix and Classification Metrics

| Metric | Recalculated Value | Evaluation Details |
| :--- | :---: | :--- |
| **Total Benchmark Trials (Protocol B)** | **269** | 136 Incongruent, 133 Congruent |
| **True Positives ($TP$)** | **100** | Incongruent trials detected as conflict |
| **False Positives ($FP$)** | **54** | Congruent trials flagged as conflict |
| **True Negatives ($TN$)** | **79** | Congruent trials correctly detected as aligned |
| **False Negatives ($FN$)** | **36** | Incongruent trials missed as aligned |
| **Condition Sensitivity / Recall** | **73.53%** | $100 / (100 + 36) = 100 / 136$ |
| **Condition Specificity** | **59.40%** | $79 / (79 + 54) = 79 / 133$ |
| **Precision** | **64.94%** | $100 / (100 + 54) = 100 / 154$ |
| **F1-Score** | **0.6897** | $2 \cdot (0.6494 \cdot 0.7353) / (0.6494 + 0.7353)$ |
| **Condition Balanced Accuracy** | **66.46%** | $(0.7353 + 0.5940) / 2$ |
| **Conflict Score AUROC** | **0.6834** | Computed from continuous $c_{\text{norm}}$ scores |

### 5.2 Baseline Fairness Audit and Leakage-Free Comparisons

#### A. Identification of Baseline Temporal Leakage
In earlier exploratory calculations, the constant-velocity extrapolation used the test trial's complete trajectory duration:
$$\Delta t_{\text{rem}} = (\text{len}(s.\text{full\_trajectory}) - 20) \times 0.01\text{ s}$$
This calculation inadvertently accessed future sample counts of the held-out test trial that would be unavailable at forecast generation time ($t = 200$ ms).

#### B. Fair Leakage-Free Implementation Strategies
To establish an unbiased benchmark, two leakage-free strategies were evaluated across the exact same 269 held-out trials under Protocol B:

1. **Strategy 1 — Endpoint Extrapolation via Training-Fold Duration Estimator:**
   Remaining movement duration is estimated strictly from training participants in each fold:
   $$\overline{\Delta t}_{\text{rem}}^{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{s \in \text{train}} \left(\frac{\text{movement\_duration\_ms}(s) - 200.0}{1000.0}\right)$$
   For held-out trials of participant $s_{\text{test}}$, the constant velocity model extrapolates linearly using only the 200 ms prefix velocity and $\overline{\Delta t}_{\text{rem}}^{\text{train}}$:
   $$\hat{\mathbf{p}}_{\text{CV}} = \mathbf{p}(t_{\text{prefix\_end}}) + \mathbf{v}_{\text{prefix}} \times \overline{\Delta t}_{\text{rem}}^{\text{train}}$$
   Zero knowledge of the test trial's future duration or trajectory is used.

2. **Strategy 2 — Predefined Shared Horizon Evaluation (Zero Duration Estimation Needed):**
   Predictions are compared at fixed future horizons $H \in \{100, 200, 300, 500\}$ ms where extrapolation interval $\Delta t = H$ is identical and prespecified across all models without requiring duration estimation.

#### C. Quantitative Benchmark Results across Complete Cohort ($N=269$ Held-Out Trials)

| Model / Baseline | Duration Strategy | Endpoint RMSE | Mean Euclidean Error | Error Reduction by Ridge |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Forecaster (LOSO)** | Learned Multi-Output | **$47.25\text{ mm}$** | **$32.85\text{ mm}$** | — |
| **Constant Velocity (Fair)** | Training-Fold Duration ($\overline{\Delta t}_{\text{rem}}^{\text{train}}$) | **$85.65\text{ mm}$** | **$75.76\text{ mm}$** | **$44.8\%$** |
| *Constant Velocity (Historical Leaked)* | *Test Trial Duration (Acausal)* | *$82.38\text{ mm}$* | *$73.12\text{ mm}$* | *Superceded* |
| **Persistence Baseline** | No Extrapolation ($\hat{\mathbf{p}} = \mathbf{p}_{\text{prefix\_end}}$) | **$177.51\text{ mm}$** | **$177.07\text{ mm}$** | **$73.4\%$** |

#### D. Per-Participant Held-Out Kinematic Generalization (Strict LOSO Full-Endpoint Prediction)
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) immediately following movement onset ($t_{\text{onset}}$).
- **Forecast Target:** The final physical reach endpoint $\mathbf{p}_{\text{endpoint}} = [x_{\text{end}}, y_{\text{end}}, z_{\text{end}}]^T$ at trial completion.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE (and mean error) across held-out test trials, alongside relative error reduction calculated consistently as:
  $$\text{Error Reduction} = 100 \times \left(1 - \frac{\text{RMSE}_{\text{Ridge}}}{\text{RMSE}_{\text{Fair CV}}}\right)$$

| Participant | Held-Out Trials | $\overline{\Delta t}_{\text{rem}}^{\text{train}}$ | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV | Ridge Mean Error | Fair CV Mean Error | Persistence Mean Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sub-01`** | 40 | 2.47 s | **$41.78\text{ mm}$** | $109.30\text{ mm}$ | $181.48\text{ mm}$ | **$61.8\%$** | $33.95\text{ mm}$ | $103.45\text{ mm}$ | $181.09\text{ mm}$ |
| **`sub-02`** | 52 | 2.08 s | **$59.90\text{ mm}$** | $110.30\text{ mm}$ | $178.36\text{ mm}$ | **$45.7\%$** | $29.41\text{ mm}$ | $106.03\text{ mm}$ | $177.76\text{ mm}$ |
| **`sub-03`** | 60 | 2.79 s | **$28.71\text{ mm}$** | $38.08\text{ mm}$ | $176.29\text{ mm}$ | **$24.6\%$** | $24.59\text{ mm}$ | $34.08\text{ mm}$ | $175.95\text{ mm}$ |
| **`sub-08`** | 57 | 2.61 s | **$25.20\text{ mm}$** | $75.14\text{ mm}$ | $178.34\text{ mm}$ | **$66.5\%$** | $22.27\text{ mm}$ | $69.11\text{ mm}$ | $178.15\text{ mm}$ |
| **`sub-15`** | 60 | 2.80 s | **$65.63\text{ mm}$** | $86.97\text{ mm}$ | $174.49\text{ mm}$ | **$24.5\%$** | $53.42\text{ mm}$ | $79.05\text{ mm}$ | $173.89\text{ mm}$ |
| **Cohort** | **269** | **—** | **$47.25\text{ mm}$** | **$85.65\text{ mm}$** | **$177.51\text{ mm}$** | **$44.8\%$** | **$32.85\text{ mm}$** | **$75.76\text{ mm}$** | **$177.07\text{ mm}$** |

In every single evaluated participant, the Ridge forecaster achieves lower RMSE and lower mean error than both fair baselines ($\text{RMSE}_{\text{Ridge}} < \text{RMSE}_{\text{CV}} < \text{RMSE}_{\text{Persistence}}$).

#### E. Predefined Fixed-Horizon Comparisons ($N=269$ Held-Out Trials, Zero Duration Estimation Needed)
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) post-onset.
- **Forecast Horizon:** Fixed elapsed future time $H \in \{100, 200, 300, 500\}\text{ ms}$ post-prefix.
- **Target Definition:** Exact intermediate 3D hand position $\mathbf{p}(t_{\text{prefix\_end}} + H)$ along the actual executed Leap Motion reach trajectory.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE, mean error, and relative error reduction $100 \times (1 - \text{RMSE}_{\text{Ridge}} / \text{RMSE}_{\text{Fair CV}})$ across all 269 held-out trials:

| Predefined Horizon | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV | Ridge Mean Error | Fair CV Mean Error | Persistence Mean Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H = 100\text{ ms}$** | **$1.44\text{ mm}$** | $3.94\text{ mm}$ | $8.82\text{ mm}$ | **$63.5\%$** | **$1.16\text{ mm}$** | $3.19\text{ mm}$ | $7.83\text{ mm}$ |
| **$H = 200\text{ ms}$** | **$4.27\text{ mm}$** | $8.80\text{ mm}$ | $18.44\text{ mm}$ | **$51.5\%$** | **$3.46\text{ mm}$** | $7.16\text{ mm}$ | $16.44\text{ mm}$ |
| **$H = 300\text{ ms}$** | **$7.81\text{ mm}$** | $14.08\text{ mm}$ | $28.47\text{ mm}$ | **$44.5\%$** | **$6.20\text{ mm}$** | $11.61\text{ mm}$ | $25.57\text{ mm}$ |
| **$H = 500\text{ ms}$** | **$16.06\text{ mm}$** | $24.83\text{ mm}$ | $48.75\text{ mm}$ | **$35.3\%$** | **$12.71\text{ mm}$** | $21.13\text{ mm}$ | $44.55\text{ mm}$ |

### 5.3 Threshold Sensitivity Consistency
The 12-cell threshold parameter grid in `reports/phase_4/threshold_sensitivity_analysis.csv` demonstrates balanced accuracy between $66.09\%$ and $66.83\%$ across all combinations ($d \in [25, 100]$ mm, $\theta \in [30^\circ, 60^\circ]$), confirming stability.

---

## 6. Offline Timing Limitations & Real-Time Constraints

To prevent mischaracterization of system capabilities, the timing pipeline is explicitly partitioned into causal and acausal stages:

```
[ Visual Cue Presentation (t_cue) ]
              |
              v
[ Pre-Movement EEG Window: -1000 ms to 0 ms relative to onset ]
  * Offline / Acausal Zero-Phase Filtering (scipy.signal.filtfilt)
  * Feature Extraction & LDA Decoder Inference
              |
              v
[ Kinematic Movement Onset (t_onset = 0 ms, velocity >= 20 mm/s) ]
              |
              v (200 ms observed hand movement prefix)
              |
[ Forecast Generation Timestamp: t = t_onset + 200 ms ]
  * Ridge Forecaster generates predicted endpoint p_pred_end
  * Conflict Engine evaluates d_target, theta_dir, and discrete state
              |
              v
[ Movement Completion / Retrospective Touch (t_endpoint) ]
  * Ground truth touch recorded for retrospective error verification
```

### Essential Scientific Qualifications:
1. **Zero-Phase Filtering is Acausal:** The pre-movement EEG features rely on bidirectional forward-backward Butterworth filtering (`filtfilt`). This operation accesses future samples relative to each time point to achieve zero phase distortion. Therefore, the neural decoding module is strictly an **offline simulation**, not a real-time streaming pipeline.
2. **Kinematic Forecaster Requires Prefix:** The kinematic forecaster does not predict hand trajectories *before* movement onset. It requires an observed prefix of $200$ ms ($20$ samples at 100 Hz) after physical onset before generating a forecast.
3. **No Real-Time or Clinical Claims:** The digital twin is a research simulation environment for offline replay and multimodal algorithm development. It is not approved or validated for closed-loop clinical neuroprosthetics.

---

## 7. Official Release Status Specification

This section formalizes the scientific and engineering status of Phase 4:

### 7.1 What is Empirically Supported:
- Pre-movement scalp EEG decodes instructed visual target location above chance under strict LOSO cross-validation (**68.34% balanced accuracy**).
- A 200 ms kinematic movement prefix supports continuous 3D endpoint forecasting that significantly outperforms standard physical baselines under strict LOSO (**47.25 mm RMSE** vs. $177.51$ mm persistence).
- Multimodal condition discrimination achieves **66.46% balanced accuracy** and **0.6834 AUROC** under strict, leakage-free LOSO cross-validation.

### 7.2 What is Software-Verified:
- 60 / 60 automated unit and regression tests pass across the entire repository.
- Causal replay engine guarantees bit-identical simulation frames across repeated runs without access to future coordinates.
- Interactive web dashboard (port 8050) operates with robust NaN-sanitized Canvas rendering.
- All model artifacts, scaler parameters, and target proxies are verified as 100% held-out per fold.

### 7.3 What Remains Exploratory:
- Heuristic conflict thresholds ($d_{\text{target}} = 50.0$ mm, $\theta_{\text{dir}} = 45.0^\circ$) represent illustrative geometric boundaries rather than clinically tuned decision limits.
- Discordance index $c_{\text{norm}}$ is an engineering indicator, not a calibrated posterior probability of biological conflict.

### 7.4 What the Dataset Does Not Establish:
- The dataset does not provide an independent ground truth for uninstructed biological intention errors or motor slips.
- The dataset does not contain independent 3D Cartesian coordinates for reaching targets (only screen pixel locations).

### 7.5 Additional Measurements Needed for Full Biological Discordance Validation:
- Direct motion capture calibration of 3D physical target objects.
- Experimental paradigms with uninstructed target perturbations or slip-inducing force fields to elicit spontaneous intention–action discordance.
- Causal online IIR/FIR EEG filtering evaluated in live closed-loop streaming.

---

## 8. Final Scientific Verdict

### Status: **Partially Supported (with audited correction)**

**Verdict Summary:**
- **Engineering Integration:** **VERIFIED** — Modular architecture, adapters, schemas, replay simulator, and interactive dashboard are functionally robust and fully tested.
- **Scientific Evaluation:** **PARTIALLY SUPPORTED** — Multimodal congruent-versus-incongruent condition discrimination is validated under strict LOSO at **66.46% balanced accuracy** and **0.6834 AUROC**.
- **Biological Intention–Action Conflict Detection:** **NOT VALIDATED** — The task paradigm evaluates instructed reversal, not spontaneous neurological intention discordance.
- **Timing Characterization:** **OFFLINE DEMONSTRATION ONLY** — Pre-movement EEG filtering is acausal; forecaster requires 200 ms movement prefix.
