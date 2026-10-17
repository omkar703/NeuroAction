# NeuroAction Phase 4 — Final Technical Report
## Neural Intention–Action Digital Twin: Multimodal Fusion, Trajectory Simulation, Conflict Detection, and Interactive Replay

**Document ID:** `NA-P4-REP-001`  
**Phase:** 4 — Final Integration & Verification  
**Date:** October 2026  
**Status:** Verified, Validated & Complete  
**Authors:** NeuroAction BCI & Computational Neuroscience Research Team  

---

## 1. Executive Summary & Research Objectives

The primary research objective of **Phase 4** of the NeuroAction initiative is to unite pre-movement cognitive goal decoding (Phase 2) and continuous 3D limb kinematics forecasting (Phase 3) into an **interpretable, multimodal Digital Twin**.

While Phase 2 addressed *what* abstract goal the brain intends ($[-1000, 0]$ ms pre-movement scalp EEG) and Phase 3 addressed *how* the physical hand moves in 3D Cartesian workspace ($t \ge 0$ ms Leap Motion optical tracking), Phase 4 addresses the critical intersection:
> **Can an integrated neural-kinematic digital twin detect intention–action discordance early in movement initiation, simulate deterministic trial execution without data leakage, and provide an interpretable research interface for human-in-the-loop validation?**

### Key Scientific & Engineering Achievements:
1. **Multimodal Fusion Architecture (`src/phase_4/`):** Built a zero-leakage, modular package wrapping audited Phase 2 decoders (`LDADecoder`, `LogisticRegressionDecoder`) and Phase 3 forecasters (`RidgeTrajectoryForecaster`) via typed adapter contracts.
2. **Deterministic Offline Replay Engine (`src/phase_4/replay_engine.py`):** Achieved 100% causal separation between past observations and future ground truth. Replay frames are bit-identical across runs, preserving empirical timestamps without acausal temporal leakage.
3. **Interpretable 4-Metric Conflict Engine (`src/phase_4/conflict_engine.py`):** Formalized four orthogonal discordance measurements:
   - Metric A: Endpoint-to-target Euclidean distance $d_{\text{target}}$ (mm) and normalized score $C_{\text{distance}} \in [0, 1]$.
   - Metric B: Directional disagreement angle $\theta_{\text{dir}}$ ($^\circ$).
   - Metric C: Neural goal-decoding agreement with experimental instruction.
   - Metric D: Retrospective trajectory tracking RMSE against future ground truth.
4. **Audited Conflict Detection Benchmark under Strict LOSO (269 Trials across 5 Subjects):**
   - **Phase 2 Neural Goal Balanced Accuracy (LOSO):** $68.34\%$
   - **Phase 3 Kinematic Forecaster Endpoint RMSE (LOSO):** $47.25\text{ mm}$ (vs. $85.65\text{ mm}$ fair CV baseline and $177.51\text{ mm}$ persistence baseline)
   - **Condition Sensitivity (Recall on Incongruent Trials):** $73.53\%$ (100 / 136 detected)
   - **Condition Specificity (True Negative Rate on Congruent Trials):** $59.40\%$ (79 / 133 aligned)
   - **Precision:** $64.94\%$ (100 / 154)
   - **Condition Discrimination Balanced Accuracy:** $66.46\%$ (bounded by neural decoder accuracy)
   - **Conflict Score AUROC:** $0.6834$
5. **Interactive Research Dashboard (`src/phase_4/dashboard.py`):** Developed a zero-external-dependency 6-panel research dashboard using Python's standard library `http.server`, featuring interactive scrub sliders, 3D workspace projection, neural telemetry cards, conflict gauges, and event timelines.
6. **Audited Test Suite:** 60 / 60 unit and regression tests passing across all four project phases (`tests/test_phase_2.py`, `tests/test_phase_3.py`, `tests/test_phase_4.py`, `tests/test_validation.py`), including dedicated regression tests for held-out model provenance, fold-specific target proxy isolation, and leakage-free baseline duration estimation.

---

## 2. System Architecture & Multimodal Pipeline

The digital twin architecture decouples raw signal ingestion, feature extraction, model inference, and state fusion into clean modular layers:

```
[Scalp EEG (-1000 to 0 ms)] --------> [Phase 2 Neural Adapter] ----> [Decoded Goal g_hat] ----+
                                                                                              |
[3D Kinematics (0 to t_obs ms)] ----> [Phase 3 Forecaster] --------> [Predicted Endpoint p] -+---> [Multimodal Fusion Engine]
                                                                                              |              |
[Visual Cue / Task Metadata] -------> [Coordinate Mapping] --------> [Target Zone p_target] --+              v
                                                                                                   [Conflict Engine]
                                                                                                     - Metric A: d_target
                                                                                                     - Metric B: theta_dir
                                                                                                     - Metric C: Goal Match
                                                                                                     - Metric D: Retro RMSE
                                                                                                     - State: ALIGNED / CONFLICT
                                                                                                             |
                                                                                                             v
                                                                                                 [Replay Engine & Dashboard]
```

### 2.1 Component Overview

| Module | File Path | Primary Function | Upstream Dependency |
| :--- | :--- | :--- | :--- |
| **Data Contracts** | `src/phase_4/schemas.py` | Typed dataclasses for metadata, observations, forecasts, fusion states, and frames | `numpy`, `typing` |
| **Configuration** | `src/phase_4/config.py` | Hyperparameters, thresholds, target centroids, and file paths | `pathlib` |
| **Trial Registry** | `src/phase_4/trial_registry.py` | Manifest loading, caching, trial integrity validation | `data/manifests/iackd_manifest.csv` |
| **Neural Adapter** | `src/phase_4/neural_adapter.py` | Wraps Phase 2 LDA/Logistic decoders, enforces $[-1000, 0]$ ms windowing | `models/phase_2/`, `src/features/pre_eeg.py` |
| **Kinematic Adapter** | `src/phase_4/kinematic_adapter.py` | Wraps Phase 3 Ridge forecaster, segments 200 ms movement prefix | `models/phase_3/`, `src/trajectory/` |
| **Coordinate Mapping** | `src/phase_4/coordinate_mapping.py` | Mapped 3D target coordinates, frame validation, unit conversions | Empirical Leap centroids |
| **Temporal Alignment** | `src/phase_4/temporal_alignment.py` | Clock synchronization, timeline markers, anti-leakage checks | `src/data/schema.py` |
| **Conflict Engine** | `src/phase_4/conflict_engine.py` | Evaluates $d_{\text{target}}$, $\theta_{\text{dir}}$, neural match, and conflict states | `src/phase_4/schemas.py` |
| **Fusion Engine** | `src/phase_4/fusion_engine.py` | Unites multimodal outputs into single coherent `FusionState` | `src/phase_4/conflict_engine.py` |
| **Replay Engine** | `src/phase_4/replay_engine.py` | Deterministic simulation advancing causally through discrete timestamps | `src/phase_4/fusion_engine.py` |
| **Event Logger** | `src/phase_4/event_logger.py` | Exports replay logs, trial summaries, and reproducibility manifests | `json`, `pandas`, `hashlib` |
| **Dashboard** | `src/phase_4/dashboard.py` | Standalone interactive HTTP dashboard & 6-panel Matplotlib generator | `http.server`, `matplotlib` |
| **CLI Runner** | `src/phase_4/cli.py` | Command-line interface for playback, listing, and inspection | `argparse` |
| **Master Runner** | `src/run_phase_4.py` | Batch evaluation runner across all 269 trials and figure generation | `scipy`, `sklearn`, `matplotlib` |

---

## 3. Verified Coordinate Frames & Target Mapping

### 3.1 Workspace Geometry
Hand positions are recorded by a Leap Motion optical sensor in Cartesian millimeters (`mm`):
- **$X$-axis (Lateral):** Left $< 0$, Right $> 0$.
- **$Y$-axis (Elevation):** Height above table surface ($\sim 149$ mm).
- **$Z$-axis (Depth):** Forward extension towards screen ($\sim 52$ mm).

### 3.2 Empirical Endpoint-Derived Target Proxies
Because the public OpenNeuro `ds006840` derivatives do not record independent 3D Cartesian coordinates for physical reaching targets (only 2D visual screen pixel coordinates `ball_x_pix`), target coordinates are defined as **empirical endpoint-derived proxies**.

Under rigorous Leave-One-Subject-Out (LOSO) cross-validation, these proxies are estimated strictly from the 4 training subjects in each fold (`estimate_fold_target_centroids`), completely excluding the held-out subject's reaching endpoints:

$$\mathbf{p}_{\text{target, LEFT}} \approx [-188.3, 150.0, 54.5]^\top \text{ mm}$$

$$\mathbf{p}_{\text{target, RIGHT}} \approx [+187.7, 148.7, 52.8]^\top \text{ mm}$$

Centroids across all five LOSO folds demonstrate exceptional spatial stability ($\sigma < 0.2\text{ mm}$ lateral), confirming the consistency of physical endpoints across participants while strictly guarding against in-sample leakage.

### 3.3 Phase 2 Label Semantics & Task Goal Dissociation
In the canonical trial schema, `target_goal` is derived as `ball_dir` (the visual screen stimulus location) from the dataset's `label_code`:
- **Congruent Trials (N=133):** Red visual cue instructs direct reaching toward the stimulus. Decoded goal matches physical movement direction (e.g., Goal Left reaches to $[-188.2, 150.0, 53.0]$ mm; $\text{ball\_dir} == \text{hand\_dir}$).
- **Incongruent Trials (N=136):** Yellow visual cue instructs stimulus-response reversal. The visual stimulus is `ball_dir`, while physical reaching proceeds to the opposite target ($[+186.9, 149.7, 50.4]$ mm; $\text{ball\_dir} \ne \text{hand\_dir}$).
- **Decoder Target Meaning:** Phase 2 pre-movement scalp EEG decoders are trained to predict `target_goal` (`ball_dir`), meaning they predict the **instructed visual stimulus target location**, not an unconstrained internal intention.
- **Spatial Divergence:** Incongruent trials exhibit a massive lateral target separation:
  $$\Delta x = |186.9 - (-188.0)| \approx 375 \text{ mm}$$
  providing a definitive physical basis for early condition discrimination between the visual target and hand kinematics.

---

## 4. Conflict Engine Formulation & State Machine

The Conflict Engine evaluates four complementary, non-conflated measurements:

### 4.1 Measurement Formulations

1. **Measurement A — Endpoint-to-Target Distance:**
   $$d_{\text{target}} = \|\hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{target}}\|_2 \quad (\text{mm})$$
   $$C_{\text{distance}} = \min\left(1.0, \frac{d_{\text{target}}}{d_{\text{reference}}}\right), \quad d_{\text{reference}} = 100.0 \text{ mm}$$

2. **Measurement B — Directional Disagreement Angle:**
   $$\mathbf{v}_{\text{pred}} = \hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{origin}}, \quad \mathbf{v}_{\text{target}} = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{origin}}$$
   $$\theta_{\text{dir}} = \arccos\left(\text{clip}\left(\frac{\mathbf{v}_{\text{pred}} \cdot \mathbf{v}_{\text{target}}}{\|\mathbf{v}_{\text{pred}}\| \|\mathbf{v}_{\text{target}}\| + \epsilon}, -1.0, 1.0\right)\right) \quad (^\circ)$$

3. **Measurement C — Neural Goal Agreement:**
   $$\text{Agreement}_{\text{goal}} = \mathbb{I}(\hat{g} = g_{\text{true}})$$

4. **Measurement D — Retrospective Endpoint Error:**
   $$\text{Error}_{\text{retrospective}} = \|\hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{end, true}}\|_2 \quad (\text{mm})$$

### 4.2 Transparent Conflict States

```
                       [Validate Inputs & Coordinates]
                                     |
                       +-------------+-------------+
                       |                           |
                 (Non-finite,                (Valid Inputs)
                Missing Frame)                     |
                       |                           v
                [INVALID_INPUT /          [Check Zero Norm /
                  UNAVAILABLE]             Confidence < 0.55]
                                                   |
                                     +-------------+-------------+
                                     |                           |
                                   (True)                     (False)
                                     |                           |
                                [UNCERTAIN]            [d_target > 50 mm OR
                                                        theta_dir > 45 deg]
                                                                 |
                                                   +-------------+-------------+
                                                   |                           |
                                                 (True)                     (False)
                                                   |                           |
                                       [POTENTIAL_DISAGREEMENT]            [ALIGNED]
```

---

## 5. Audited Batch Evaluation Results (Strict LOSO Cross-Validation)

Batch evaluation was executed across all 269 valid trials (538 evaluations across Protocol A and Protocol B) under strict Leave-One-Subject-Out (LOSO) cross-validation, using strictly held-out Phase 2 decoders and Phase 3 forecasters for each participant alongside fold-specific target proxies:

### 5.1 Audited Experimental-Condition Discrimination Matrix (Protocol B: Kinematic-Onset-Locked)

| Evaluation Metric | Mathematical Definition | Audited Value | Scientific Meaning / Benchmark |
| :--- | :--- | :--- | :--- |
| **Cohort Size** | Valid multimodal trials | **269 trials** | 5 subjects (`sub-01` to `sub-15`) under LOSO |
| **Incongruent Condition (Reversal)** | Reaching opposite to cue | **136 trials** | Condition-level conflict reference ($y=1$) |
| **Congruent Condition (Direct Reach)** | Reaching directly to cue | **133 trials** | Condition-level aligned reference ($y=0$) |
| **True Positives ($TP$)** | Incongruent trials detected as Conflict | **100 trials** | Reversal reaches flagged discordant |
| **False Positives ($FP$)** | Congruent trials flagged as Conflict | **54 trials** | Misclassified neural goals causing false alarms |
| **True Negatives ($TN$)** | Congruent trials detected as Aligned | **79 trials** | Correctly cleared congruent reaches |
| **False Negatives ($FN$)** | Incongruent trials missed as Aligned | **36 trials** | Misclassified neural goals causing missed conflicts |
| **Sensitivity (Recall)** | $TP / (TP + FN)$ | **$73.53\%$** | $100 / 136$ incongruent trials flagged |
| **Specificity** | $TN / (TN + FP)$ | **$59.40\%$** | $79 / 133$ congruent trials cleared |
| **Precision** | $TP / (TP + FP)$ | **$64.94\%$** | $100 / 154$ conflict alerts genuine |
| **F1-Score** | $2 \cdot (\text{Prec} \cdot \text{Rec}) / (\text{Prec} + \text{Rec})$ | **$0.6897$** | Harmonic balance |
| **Condition Discrimination Balanced Accuracy** | $(\text{Sensitivity} + \text{Specificity}) / 2$ | **$66.46\%$** | Closely tracks neural goal accuracy ($68.34\%$) |
| **Conflict Score AUROC** | Computed from continuous $c_{\text{norm}}$ | **$0.6834$** | Continuous score class separability |
| **Phase 2 Goal Accuracy (LOSO)** | Balanced accuracy of EEG decoder | **$68.34\%$** | Matches Phase 2 audited findings |
| **Phase 3 Forecaster RMSE (LOSO)** | 3D Cartesian endpoint error | **$47.25\text{ mm}$** | Forecaster generalizability across cohort |

### 5.2 Observed Relationship Between Condition Discrimination and Neural Decoding
The audited condition discrimination balanced accuracy of **$66.46\%$** closely tracks the Phase 2 neural goal balanced accuracy of **$68.34\%$**. This observed relationship is governed by the empirical mechanics of the two-choice task:
1. **On Congruent Trials:** When the pre-movement neural decoder predicts the correct target ($68.3\%$ of trials), the kinematic reach aligns with the predicted target ($\theta \approx 0^\circ, d \approx 10$ mm) $\rightarrow$ `ALIGNED`. When the neural decoder misclassifies the goal ($31.7\%$ of trials), the conflict engine compares the physical reach against the opposite target centroid ($\Delta x \approx 375$ mm) $\rightarrow$ `POTENTIAL_DISAGREEMENT` (False Positive).
2. **On Incongruent Trials:** When the neural decoder predicts the correct target ($68.3\%$ of trials), the physical reach proceeds toward the opposite target ($\Delta x \approx 375$ mm) $\rightarrow$ `POTENTIAL_DISAGREEMENT` (True Positive). When the neural decoder misclassifies the target, it erroneously matches the physical hand direction $\rightarrow$ `ALIGNED` (False Negative).

Because the kinematic forecaster reliably identifies left versus right reaching directions (47.25 mm RMSE vs. 375 mm separation), the final condition classification decisions predominantly reflect whether the pre-movement neural decoder made a correct goal prediction.

### 5.3 Trajectory Forecasting Retrospective Accuracy across Entire Cohort

#### A. Full-Endpoint Prediction Comparison
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) immediately following kinematic movement onset ($t_{\text{onset}}$).
- **Forecast Target:** The final physical reach endpoint $\mathbf{p}_{\text{endpoint}} = [x_{\text{end}}, y_{\text{end}}, z_{\text{end}}]^T$ at trial completion.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE (and mean error) across $N=269$ strictly held-out trials under LOSO, with error reduction calculated consistently as:
  $$\text{Error Reduction} = 100 \times \left(1 - \frac{\text{RMSE}_{\text{Ridge}}}{\text{RMSE}_{\text{Baseline}}}\right)$$

Across all 269 valid trials evaluated under strict LOSO (each subject evaluated using their corresponding held-out Ridge forecaster checkpoint):
- **Ridge Forecaster Endpoint RMSE (LOSO Cohort):** **$47.25\text{ mm}$** (Mean Euclidean Error: **$32.85\text{ mm}$**)
- **Constant Velocity Baseline (Fair, Training-Duration Strategy):** **$85.65\text{ mm}$** (Mean: **$75.76\text{ mm}$**, Ridge achieves $44.8\%$ error reduction)
- **Persistence Baseline:** **$177.51\text{ mm}$** (Mean: **$177.07\text{ mm}$**, Ridge achieves $73.4\%$ error reduction)

##### Per-Participant Held-Out Kinematic Generalization:
| Participant | Held-Out Trials | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`sub-01`** | 40 | **$41.78\text{ mm}$** | $109.30\text{ mm}$ | $181.48\text{ mm}$ | **$61.8\%$** |
| **`sub-02`** | 52 | **$59.90\text{ mm}$** | $110.30\text{ mm}$ | $178.36\text{ mm}$ | **$45.7\%$** |
| **`sub-03`** | 60 | **$28.71\text{ mm}$** | $38.08\text{ mm}$ | $176.29\text{ mm}$ | **$24.6\%$** |
| **`sub-08`** | 57 | **$25.20\text{ mm}$** | $75.14\text{ mm}$ | $178.34\text{ mm}$ | **$66.5\%$** |
| **`sub-15`** | 60 | **$65.63\text{ mm}$** | $86.97\text{ mm}$ | $174.49\text{ mm}$ | **$24.5\%$** |
| **Cohort** | **269** | **$47.25\text{ mm}$** | **$85.65\text{ mm}$** | **$177.51\text{ mm}$** | **$44.8\%$** |

In every single evaluated participant, the Ridge forecaster achieves substantially lower RMSE and mean error than both fair baselines.

#### B. Predefined Fixed-Horizon Comparisons (Zero Duration Estimation Needed)
- **Input Prefix:** The first $200\text{ ms}$ ($20$ samples at $100\text{ Hz}$) post-onset.
- **Forecast Horizon:** Fixed elapsed future time $H \in \{100, 200, 300, 500\}\text{ ms}$ post-prefix.
- **Target Definition:** Exact intermediate 3D hand position $\mathbf{p}(t_{\text{prefix\_end}} + H)$ along the executed Leap Motion reach trajectory.
- **Evaluation Metric:** 3D Cartesian Euclidean distance RMSE and relative error reduction $100 \times (1 - \text{RMSE}_{\text{Ridge}} / \text{RMSE}_{\text{Fair CV}})$ across all 269 held-out trials:

| Predefined Horizon | Ridge RMSE | Fair CV RMSE | Persistence RMSE | Ridge Error Reduction vs Fair CV |
| :---: | :---: | :---: | :---: | :---: |
| **$H = 100\text{ ms}$** | **$1.44\text{ mm}$** | $3.94\text{ mm}$ | $8.82\text{ mm}$ | **$63.5\%$** |
| **$H = 200\text{ ms}$** | **$4.27\text{ mm}$** | $8.80\text{ mm}$ | $18.44\text{ mm}$ | **$51.5\%$** |
| **$H = 300\text{ ms}$** | **$7.81\text{ mm}$** | $14.08\text{ mm}$ | $28.47\text{ mm}$ | **$44.5\%$** |
| **$H = 500\text{ ms}$** | **$16.06\text{ mm}$** | $24.83\text{ mm}$ | $48.75\text{ mm}$ | **$35.3\%$** |

---

## 6. Interactive Dashboard & Demonstration Scenario

### 6.1 Interactive 6-Panel Research Dashboard
The dashboard provides a dedicated panel layout fulfilling all Step 8 requirements:
- **Panel A (Controls & Clock):** Subject selector, trial dropdown, forecasting protocol selector, play/pause, restart, and scrub slider.
- **Panel B (3D Twin Workspace):** Top-down Leap Motion projection ($X-Z$) rendering target zones (Left: Blue, Right: Green), past observed path (Amber), predicted future forecast (Pink dashed), predicted endpoint (Pink cross), and future ground truth (Grey dotted).
- **Panel C (Neural Intention Card):** Decoded goal, confidence score, posterior probabilities, and offline zero-phase indicator.
- **Panel D (Conflict Gauge):** Color-coded conflict state banner, endpoint-to-target distance ($d_{\text{target}}$), directional angle ($\theta_{\text{dir}}$), normalized discordance index, and natural language explanation.
- **Panel E (Temporal Timeline):** Event sequence markers for EEG start ($-1000$ ms), Go-Cue ($0$ ms), Kinematic Onset ($t_{\text{onset}}$), Prefix End ($t_{\text{prefix\_end}}$), and current simulation clock.
- **Panel F (Verification Metrics):** Retrospective endpoint tracking error, held-out model badge, and cohort baseline comparisons.

### 6.2 Demonstrating Trial `sub-01_run-01_L1_tr00`
- **Participant:** `sub-01` (Held-out validation)
- **Experimental Semantics:** Congruent, Red cue, Ball Left, Hand Left.
- **Neural Intention Output:** Decoded Goal: `LEFT` ($P(\text{left}) = 84.7\%$, Confidence: $0.847$, Temporally available at $t = 0.0$ ms).
- **Kinematic Forecast Output:** Movement onset at $t_{\text{onset}} = 360.0$ ms. Forecast available at $t = 560.0$ ms. Predicted Endpoint: $[-178.4, 151.2, 52.8]$ mm.
- **Conflict Assessment:** Endpoint distance to Left Target: $9.8$ mm (within $50$ mm tolerance); Directional angle: $1.2^\circ$. Conflict State: `ALIGNED`.
- **Retrospective Error:** Endpoint error vs actual touch: $14.2$ mm.
- **Exported Files:**
  - `reports/phase_4/demo/demo_replay_event_log.csv` (141 frames logged)
  - `reports/phase_4/demo/demo_trial_summary.json`
  - `reports/phase_4/demo/reproducibility_manifest.json`

---

## 7. Automated Test Suite & Regression Verification

The test suite incorporates 60 comprehensive unit and regression tests covering all four project phases, including dedicated regression tests for held-out model provenance, fold-specific target proxy isolation, and leakage-free baseline comparisons:

```
tests/test_phase_2.py .......                                            [ 11%]
tests/test_phase_3.py .............                                      [ 33%]
tests/test_phase_4.py .............................                      [ 81%]
tests/test_validation.py ...........                                     [100%]
============================== 60 passed in 10.48s =============================
```

### Audited Verification & Regression Tests Added:
- `test_held_out_model_provenance_all_subjects`: Verifies that held-out model checkpoints exist and load for all 5 subjects (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`) in both Phase 2 and Phase 3 without falling back to full-cohort models.
- `test_fold_target_centroids_excludes_held_out_subject`: Verifies that `estimate_fold_target_centroids` computes target centroids strictly from training subjects, completely excluding the test subject's endpoints.
- `test_audited_evaluation_matrix_provenance`: Verifies that `reports/phase_4/audited_evaluation_matrix.csv` (538 rows) contains complete provenance columns and that `is_held_out` is `True` for every row.
- `test_threshold_sensitivity_provenance`: Verifies that `reports/phase_4/threshold_sensitivity_analysis.csv` exists and documents stability across the full 12-cell threshold parameter grid.
- `test_baseline_comparison_leakage_free`: Verifies that the constant-velocity baseline estimates remaining movement duration strictly from training participants without accessing the test trial's future duration.

### Key Synthetic Verification Cases Verified:
1. **Case 1 (Exact Match):** $d=0$ mm, $\theta=0^\circ \rightarrow$ `ALIGNED`.
2. **Case 2 (Within Tolerance):** $d=30$ mm $\le 50$ mm $\rightarrow$ `ALIGNED`.
3. **Case 3 (Exceeds Tolerance):** $d=80$ mm $> 50$ mm $\rightarrow$ `POTENTIAL_DISAGREEMENT`.
4. **Case 4 (Direction Agrees):** $\theta < 5^\circ \rightarrow$ `ALIGNED`.
5. **Case 5 (Direction Opposes):** $\theta > 140^\circ \rightarrow$ `POTENTIAL_DISAGREEMENT`.
6. **Case 6 (Zero Vector Magnitude):** Safely returns `UNCERTAIN` without division-by-zero crash.
7. **Case 7 (Missing Target Coordinates):** Safely returns `UNAVAILABLE`.
8. **Case 8 (Mismatched Coordinate Frames):** Safely returns `INVALID_INPUT`.
9. **Case 9 (Mismatched Units):** Safely returns `INVALID_INPUT`.
10. **Case 10 (Temporal Leakage Guard):** Rejects predictions with input timestamps beyond declared clock.
11. **Case 11 (Missing Goal Label):** Safely returns `UNAVAILABLE`.
12. **Case 12 (Trial ID Mismatch):** Safely returns `INVALID_INPUT`.
13. **Case 13 (Non-finite Coordinates):** Rejects NaN/Inf coordinates with `INVALID_INPUT`.
14. **Case 14 (Missing Samples/Dropouts):** Safely returns `UNAVAILABLE`.
15. **Case 15 (Uncertain Classifier Confidence):** Low confidence ($< 0.55$) returns `UNCERTAIN`.
16. **Anti-Leakage Mutated Future Test:** Modifying future hand positions ($t > 500$ ms) produces bit-identical frames at $t = 200$ ms.

---

## 8. Reproducibility Instructions

### 8.1 Launching the Interactive Web Dashboard
```bash
PYTHONPATH=. .venv/bin/python src/phase_4/dashboard.py --port 8050
```
Open your browser at `http://127.0.0.1:8050`.

### 8.2 Replaying a Trial via CLI & Exporting Logs
```bash
PYTHONPATH=. .venv/bin/python src/phase_4/cli.py replay --trial-id sub-01_run-01_L1_tr00 --protocol kinematic_onset --export
```

### 8.3 Executing the Complete Batch Evaluation & Figure Generation
```bash
PYTHONPATH=. .venv/bin/python src/run_phase_4.py
```

### 8.4 Running the Test Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

---

## 9. Limitations & Future Real-Time Requirements

### 9.1 Acknowledged Limitations
1. **Offline Replay Nature:** Pre-movement scalp EEG signals were preprocessed using zero-phase forward-backward Butterworth filters. While replay enforces causal temporal gating, the neural features themselves depend on acausal filtering.
2. **Fixed Experimental Targets:** Target positions reflect two discrete screen centroids. Generalizing to arbitrary 3D reach-to-grasp targets requires continuous spatial target mapping.
3. **Engineering Score vs Clinical Probability:** The conflict score $C_{\text{distance}}$ is a normalized engineering indicator calibrated against target geometry, not a validated medical risk probability.

### 9.2 Requirements for Live Closed-Loop Real-Time Operation
To advance from offline simulation to live closed-loop neuroprosthetic intervention:
1. **Causal EEG Filtering:** Replace zero-phase filtering with real-time IIR or minimum-phase FIR filters with group-delay compensation.
2. **Real-time Onset Detection:** Implement streaming continuous movement onset detectors (e.g., CUSUM or Bayesian changepoint algorithms) operating on live Leap Motion or EMG streams.
3. **Hardware Streaming Buffer:** Implement an LSL (Lab Streaming Layer) synchronization ring buffer guaranteeing sub-20 ms latency.
4. **Safety Interlock Architecture:** Real-time robotic or prosthetic controllers must implement hard real-time safety limits preventing sudden trajectory overrides during `UNCERTAIN` or `UNAVAILABLE` states.

---

## 10. Official Release Status Specification & Verdict

### 10.1 Empirical and Software Release Status
- **Empirically Supported:**
  - Pre-movement scalp EEG decodes instructed visual target location above chance under strict LOSO (**68.34% balanced accuracy**).
  - Trajectory forecaster predicts 3D Cartesian endpoints with **47.25 mm RMSE** under strict LOSO, substantially beating persistence ($177.51$ mm) and fair constant velocity ($85.65$ mm) baselines.
  - Multimodal congruent-versus-incongruent condition discrimination achieves **66.46% balanced accuracy** and **0.6834 AUROC** under strict LOSO.
- **Software-Verified:**
  - 60 / 60 regression and unit tests passing across all four project phases.
  - Complete causal separation in replay engine; deterministic bit-identical frames across runs.
  - Interactive web dashboard (port 8050) with NaN-safe Canvas rendering.
- **Exploratory Elements:**
  - Conflict thresholds ($d = 50.0$ mm, $\theta = 45.0^\circ$) represent illustrative geometric boundaries.
  - Normalized score $c_{\text{norm}}$ is an engineering indicator, not a calibrated biological error probability.
- **Dataset Limitations (`ds006840`):**
  - Evaluates instructed stimulus-response reversal, not uninstructed spontaneous motor errors.
  - Contains no independent physical 3D coordinates for reaching targets (only screen pixels).

### 10.2 Audit Trail for Superseded Numbers
The initial claim of **95.51% balanced accuracy** and **0.9951 AUROC** was generated during preliminary batch evaluation when participants `sub-02` through `sub-15` inadvertently evaluated against full-cohort models. Training strictly held-out models for all 5 participants and enforcing fold-specific target proxies revised these metrics to the true, scientifically defensible figures of **66.46% balanced accuracy** and **0.6834 AUROC**.

### 10.3 Final Release Verdict
**Partially Supported (with audited correction)**
- Engineering integration verified.
- Experimental condition discrimination partially supported under strict LOSO.
- Biological intention–action discordance unvalidated.
- Offline demonstration only.
