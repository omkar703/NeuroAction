# Phase 5 Integration Assessment & Repository Audit
## NeuroAction — Neural Intention–Action Digital Twin

**Auditor:** NeuroAction Senior Research & Engineering Group  
**Date:** October 2026  
**Target Platform:** Repository root (`Brain_Movement/`)  
**Scope:** Systematic audit of existing Phase 1–4 codebase, data provenance, coordinate frames, model artifacts, and evaluation feasibility for Phase 5 prospective outcome prediction.

---

## 1. Verified Available Datasets and Valid Trials

- **Dataset Provenance:** OpenNeuro **`ds006840`** (IACKD: Inter-Action Coordination and Kinematic Dynamics, Version 1.0.0, Nature *Scientific Data* 2026, DOI: `10.18112/openneuro.ds006840.v1.0.0`).
  *(Note: The identifier `ds003612` in early preliminary project notes was an accidental typographical error and has been systematically excised).*
- **Primary Cohort:** 5 participants evaluated under strict Leave-One-Subject-Out (LOSO):
  - `sub-01`: 40 valid multimodal trials (Run `run-01_L1`)
  - `sub-02`: 52 valid multimodal trials (Runs `run-01_L1`, `run-01_R1`)
  - `sub-03`: 60 valid multimodal trials (Runs `run-01_L1`, `run-01_R1`)
  - `sub-08`: 57 valid multimodal trials (Runs `run-01_L1`, `run-01_R1`)
  - `sub-15`: 60 valid multimodal trials (Runs `run-01_L1`, `run-01_R1`)
  - **Total Cohort Size:** **269 valid multimodal trials** (out of 280 indexed rows in `data/manifests/iackd_manifest.csv`; 11 excluded due to Leap Motion tracking dropouts or missing pre-EEG segments).
- **Trial Balancing:**
  - Congruency: 133 Congruent trials ($49.4\%$), 136 Incongruent trials ($50.6\%$).
  - Target Goal (`ball_dir`): 135 Left ($50.2\%$), 134 Right ($49.8\%$).
  - Physical Hand Direction (`hand_dir`): 135 Left ($50.2\%$), 134 Right ($49.8\%$).

---

## 2. Actual Phase 2 Output Semantics

- **Decoded Target Variable:** `target_goal` is derived as `ball_dir` from the trial's experimental condition code (`label_code`).
  - Red cue (Congruent): $\text{ball\_dir} == \text{hand\_dir}$.
  - Yellow cue (Incongruent): $\text{hand\_dir} \neq \text{ball\_dir}$ (stimulus-response reversal rule).
- **Scientific Implication:** Phase 2 decodes the **instructed visual stimulus target location on screen**, *not* an unconstrained internal motor intention.
- **Model Checkpoints:** Linear Discriminant Analysis (LDA) with 26-channel whole-scalp CSP spatial filters.
  - Held-out model checkpoints exist for all 5 subjects: `lda_goal_decoder_held_out_sub01.pkl` through `lda_goal_decoder_held_out_sub15.pkl`.
- **Output Capabilities:** Posterior class probabilities $[P(\text{left}), P(\text{right})]$, binary goal classification $\in \{\text{left}, \text{right}\}$, and classification confidence score $\max(P(\text{left}), P(\text{right}))$.
- **Temporal Nature:** Evaluated on the $[-1000, 0]\text{ ms}$ pre-movement EEG window relative to visual go-cue ($0\text{ ms}$). However, the underlying derivative was preprocessed using offline zero-phase forward-backward filtering (`filtfilt`), meaning Phase 2 must strictly be classified as **offline/acausal**.

---

## 3. Actual Phase 3 Input and Output Timing

- **Sensory Modality:** Leap Motion optical tracker ($100\text{ Hz}$ nominal, millimetre 3D Cartesian coordinates).
- **Movement Initiation Discovery:** Visual go-cue ($t = 0\text{ ms}$) is preceded by a visuomotor reaction time latency of $370.2 \pm 184.6\text{ ms}$ (median $360.0\text{ ms}$). During the initial $0$–$200\text{ ms}$ post-cue, the hand displaces by only $0.83 \pm 0.88\text{ ms}$ ($7.2\text{ mm/s}$).
- **Evaluated Protocols:**
  - **Protocol A (Event-Locked):** Observation prefix $t \in [0, 200]\text{ ms}$ post-cue (decodes reaction-time postural drift).
  - **Protocol B (Kinematic-Onset):** Movement onset defined when speed crosses $\ge 20\text{ mm/s}$ ($t_{\text{onset}}$). Observation prefix spans $t \in [t_{\text{onset}}, t_{\text{onset}} + 200]\text{ ms}$ ($20$ samples).
- **Forecast Outputs:**
  - Future trajectory slices at fixed horizons $H \in \{100, 200, 300, 500\}\text{ ms}$ post-prefix.
  - Final resting 3D endpoint $\mathbf{p}_{\text{endpoint}} = [x_{\text{end}}, y_{\text{end}}, z_{\text{end}}]^T$ in Leap Motion coordinates.
- **Accuracy Benchmark (Protocol B, Strict LOSO):**
  - Ridge Forecaster Endpoint RMSE: **$47.25\text{ mm}$** (Mean Euclidean Error: **$32.85\text{ mm}$**).
  - Fair Constant Velocity Baseline: **$85.65\text{ mm}$** ($44.8\%$ error reduction).
  - Persistence Baseline: **$177.51\text{ mm}$** ($73.4\%$ error reduction).

---

## 4. Availability of Independent Physical Target Coordinates

- **Critical Audit Finding:** OpenNeuro `ds006840` **DOES NOT CONTAIN** independent, calibrated 3D physical coordinates for the physical reaching targets.
  - The experiment recorded 2D screen pixel coordinates for the visual virtual ball, but did not track or calibrate the physical Leap Motion 3D workspace boundaries against external physical target markers.
- **Target Proxies Used in Phase 4:**
  - Target centroids are empirical endpoint-derived cluster centers:
    - Left Target Centroid: $\mathbf{c}_{\text{left}} \approx [-188.4, 147.7, 44.9]\text{ mm}$
    - Right Target Centroid: $\mathbf{c}_{\text{right}} \approx [+188.4, 147.8, 54.3]\text{ mm}$
  - In Phase 4, fold-specific target proxies were estimated strictly from the 4 training participants (`estimate_fold_target_centroids()`) to prevent test-subject leakage.
- **Phase 5 Mandate:** These centroids remain **exploratory empirical proxies**. They must never be represented as authoritative, independently measured physical ground truth.

---

## 5. Availability of Independently Defined Target-Success or Failure Labels

- **Current Status in `ds006840`:** The original dataset does **NOT** contain independent annotations of reaching task success or motor execution errors.
  - In Phase 4, the ground truth evaluated was $y = (\text{congruency} == \text{"incongruent"})$.
  - This measures **experimental-condition discrimination** (identifying whether the trial followed a direct reach rule or a stimulus-reversal rule), **NOT** biological intention–action conflict or motor failure.
  - A successful incongruent reach is a correct, rule-following action, not a motor error.
- **Phase 5 Mandate:** Independent target-reaching success ($y_{\text{success}} \in \{0, 1\}$) must be formally defined as a measurable geometric criterion relative to an independently specified target region. Because independent physical targets do not exist in `ds006840`, empirical target-success evaluations on `ds006840` are strictly **exploratory proxy evaluations**, while authoritative validation requires a prospective study design.

---

## 6. Model Artifacts Suitable for Held-Out Evaluation

All required model checkpoints are persisted and verified on disk:
- **Phase 2 Neural Decoders (`models/phase_2/`):**
  - `lda_goal_decoder_held_out_sub01.pkl`
  - `lda_goal_decoder_held_out_sub02.pkl`
  - `lda_goal_decoder_held_out_sub03.pkl`
  - `lda_goal_decoder_held_out_sub08.pkl`
  - `lda_goal_decoder_held_out_sub15.pkl`
- **Phase 3 Kinematic Forecasters (`models/phase_3/`):**
  - `kinematic_forecaster_held_out_sub01.pkl`
  - `kinematic_forecaster_held_out_sub02.pkl`
  - `kinematic_forecaster_held_out_sub03.pkl`
  - `kinematic_forecaster_held_out_sub08.pkl`
  - `kinematic_forecaster_held_out_sub15.pkl`

Each checkpoint was trained strictly excluding the corresponding test participant, ensuring genuine out-of-fold generalization.

---

## 7. Existing Leakage Protections

1. **Participant Separation:** LOSO cross-validation is strictly maintained. Scaler normalization parameters, CSP filters, Ridge regressors, and target proxies are fit exclusively on training participants.
2. **Kinematic Anti-Leakage:**
   - Trajectory forecaster consumes only the first $200\text{ ms}$ post-onset. Future samples are strictly hidden.
   - Constant-velocity baseline uses training-fold duration estimation ($\overline{\Delta t}_{\text{rem}}^{\text{train}}$), completely blind to test trial movement duration.
3. **Causal Simulation Separation:** In `ReplayEngine`, `actual_future_path` and `actual_endpoint` are marked `None` during causal simulation and are populated only in retrospective mode.

---

## 8. Required Data Not Available in the Repository

The following measurements are required for authoritative biological validation but are **absent** from OpenNeuro `ds006840`:
1. **Calibrated Physical 3D Target Coordinates:** Independent optical motion-capture coordinates of physical target zones.
2. **Explicit Motor Error / Slip Annotations:** Independent ground truth for uninstructed motor execution errors, trajectory deviations, or target misses.
3. **Target Perturbations:** Experimental trials featuring target displacement or force-field perturbations during movement execution.
4. **Causal Online EEG Stream:** Pre-movement EEG sampled and filtered via causal real-time IIR/FIR filters without acausal zero-phase lookahead.

---

## 9. Claims That Can Legitimately Be Evaluated Now (Level 1 & Level 2)

1. **Software Implementation (Level 1):** Complete, modular Phase 5 architecture implementing data contracts, eligibility audits, baseline models, fusion classifiers, decision policies with abstention, external dataset ingestion, and automated testing.
2. **Kinematic Generalization (Level 2):** Retrospective evaluation of held-out 3D endpoint forecasting ($47.25\text{ mm}$ RMSE) and multi-horizon accuracy across 269 trials.
3. **Exploratory Target-Reaching Prediction (Level 2):** Evaluation of whether early kinematic forecasts predict landing within fold-specific target proxy centroids ($d \le 50.0\text{ mm}$).
4. **Condition Discrimination Baseline (Level 2):** Clarified assessment showing condition classification achieves $66.46\%$ balanced accuracy and $0.6834$ AUROC, mathematically tied to Phase 2 neural goal decoding ($68.34\%$).

---

## 10. Claims That Require Prospective Human Recordings (Level 3 & Level 4)

1. **Independent Target-Success Outcome Prediction (Level 3):** Cannot be authoritatively validated without independent physical target coordinates.
2. **Biological Intention–Action Discordance Detection (Level 3/4):** Cannot be claimed on `ds006840` because the dataset contains instructed rule reversals, not spontaneous neurological intention errors.
3. **Prospective Real-Time BCI Deployment (Level 4):** Requires online causal EEG filtering and live closed-loop streaming.

---

## Audit Conclusion & Phase 5 Execution Roadmap

The Phase 5 framework will be built with absolute scientific honesty:
- Implement the complete, extensible evaluation engine and data contracts in `src/phase_5/`.
- Provide an independent eligibility audit classifying `ds006840` trials as eligible for exploratory proxy evaluation while marking authoritative independent ground-truth validation as **pending**.
- Specify a comprehensive prospective validation protocol (`prospective_validation_protocol.md`) and external validation ingestion interface.
- Maintain full compatibility with earlier phases and ensure all 60 existing tests continue to pass.
