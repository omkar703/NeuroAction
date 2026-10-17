# NeuroAction Phase 4 — System Integration & Interface Assessment

**Document ID:** `NA-P4-INT-001`  
**Phase:** 4 — Neural Intention–Action Digital Twin  
**Date:** October 2026  
**Status:** Complete & Verified  

---

## 1. Executive Summary

This integration assessment establishes the technical and scientific foundation for **Phase 4** of the NeuroAction project: the multimodal fusion, kinematic trajectory forecasting, intention–action conflict detection, and deterministic trial replay engine.

Phase 4 integrates the outputs of two audited prior phases:
1. **Phase 2 (Pre-movement Scalp EEG Intention Decoding):** Decodes abstract intended target goals ($[-1000, 0]$ ms window) and dissociates cognitive goal representations from physical motor commands under Leave-One-Subject-Out (LOSO) cross-validation.
2. **Phase 3 (Continuous 3D Kinematics Reconstruction & Trajectory Forecasting):** Segments 3D Leap Motion hand kinematics, extracts 200 ms movement prefixes, and forecasts forward trajectories ($+100, +200, +300, +500$ ms) and final reaching endpoints under dual protocols (Protocol A: Event-Locked vs Protocol B: Kinematic-Onset-Locked).

This document audits all interfaces, coordinate systems, temporal references, model checkpoints, data contracts, and potential sources of error or data leakage prior to runtime implementation.

---

## 2. Interface Audit Matrix

| System Component | Upstream Origin | Data Contract / Class | Output Dimension & Units | Provenance & Validation |
| :--- | :--- | :--- | :--- | :--- |
| **Multimodal Manifest** | `data/manifests/iackd_manifest.csv` | `pd.DataFrame` (280 indexed rows) | 269 valid trials across 5 subjects (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`) | Verified in `tests/test_validation.py` |
| **Canonical Trial Loader** | `src/data/trial_loader.py` | `CanonicalTrial` | Pre-EEG: `(200, 26)`, 200 Hz; ME-Kinematics: `(T, 3)`, 100 Hz (`mm`) | Verified in `tests/test_validation.py` |
| **Phase 2 Neural Decoder** | `src/models/decoders.py` | `LDADecoder`, `LogisticRegressionDecoder` | Class: `left` (0) vs `right` (1); Posterior Proba: `(2,)` float32 | LOSO evaluated in `reports/phase_2/experiment_matrix_loso.csv` |
| **Phase 2 Checkpoints** | `models/phase_2/` | Pickle (`.pkl`) | `lda_goal_decoder.pkl` (Cohort), `lda_goal_decoder_held_out_sub01.pkl` | Strictly fitted without held-out test fold |
| **Phase 3 Forecaster** | `src/trajectory/inference.py` | `KinematicTwinForecaster`, `ForecastResult` | `predicted_endpoint`: `(3,)` (`mm`); `predicted_positions`: `(H, 3)` (`mm`) | Evaluated in `reports/phase_3/phase_3_eval_matrix.csv` |
| **Phase 3 Checkpoints** | `models/phase_3/` | Pickle (`.pkl`) | `kinematic_forecaster.pkl`, `kinematic_forecaster_held_out_sub01.pkl` | Verified in `tests/test_phase_3.py` |
| **Coordinate Space** | Leap Motion Optical Sensor | Workspace Cartesian Coordinates | $X \in [-300, +300]$ mm (Lateral), $Y \in [50, 300]$ mm (Height), $Z \in [-100, 200]$ mm (Depth) | Verified across 269 trials |

---

## 3. Detailed Interface Specifications

### 3.1 Trial Identification and Registry
- **Unique Trial Key:** String format `{participant_id}_{run_id}_tr{trial_index:02d}` (e.g., `sub-01_run-01_L1_tr00`).
- **Participant Pool:** 5 subjects (`sub-01`: 40 trials, `sub-02`: 52 trials, `sub-03`: 60 trials, `sub-08`: 57 trials, `sub-15`: 60 trials).
- **Condition Breakdown:** 133 Congruent trials (Color: Red, Hand matches Goal) and 136 Incongruent trials (Color: Yellow, Hand opposes Goal).
- **Trial Validity Constraints:** `trial.valid == True`, Leap Motion kinematics present and non-empty, prefix window free of NaNs.

### 3.2 Temporal References and Clock Synchronization
The experimental trial contains multiple distinct physical and computational timestamps that must not be conflated:
1. **$t_{\text{EEG}} \in [-1000, 0]$ ms:** Scalp EEG pre-movement recording interval.
2. **$t = 0.0$ ms (Go-Cue Event Marker):** Visual stimulus appears on screen (ball appears red or yellow, moving left or right).
3. **$t_{\text{onset}}$ (Physical Movement Onset):** The moment physical hand speed crosses the velocity threshold ($\ge 20.0$ mm/s). Empirical cohort distribution:
   - Mean Latency: $370.2 \pm 112.4$ ms
   - Median Latency: $360.0$ ms
   - Range: $[80.0, 780.0]$ ms
4. **$t_{\text{prefix\_end}} = t_{\text{onset}} + 200$ ms:** The end of the 20-sample kinematic observation window. This is the **earliest causal timestamp at which Phase 3 forecasting can occur** under Protocol B.
5. **$t_{\text{end}}$ (Movement Termination):** The end of recorded hand motion ($t_{\text{end}} \approx 1000 - 1800$ ms).

```
Timeline:
|--- Pre-EEG [-1000, 0] ms ---|--- Reaction Latency ---|--- 200 ms Prefix ---|-------- Future Horizons --------|--- Endpoint ---|
^ -1000 ms                    ^ 0 ms (Go-Cue)          ^ t_onset (370 ms)    ^ t_prefix_end (570 ms)            ^ t_end
                              [Neural Prediction Ready]                      [Trajectory Forecast Generated]
```

### 3.3 Temporal Validity and Causal Separation Rules
To prevent data leakage during trial replay:
- **Rule 1 (Neural Availability):** The full pre-movement neural intention prediction covers $[-1000, 0]$ ms and is stamped as available at $t = 0.0$ ms. It must never be rendered as available at $t < 0.0$ ms during causal replay.
- **Rule 2 (Kinematic Availability):** Trajectory forecasts require a 20-sample prefix. Under Protocol A, forecasts become available at $t = 200.0$ ms; under Protocol B, forecasts become available at $t = t_{\text{onset}} + 200.0$ ms. Forecasts must never be rendered before their prefix completion timestamp.
- **Rule 3 (Strict Anti-Leakage):** Modification or presence of future ground-truth kinematic samples ($t > t_{\text{replay}}$) must have zero effect on predictions generated at $t_{\text{replay}}$.
- **Rule 4 (Offline vs Online Labeling):** Because pre-movement EEG filtering utilizes zero-phase forward-backward Butterworth filters (offline), replay is designated as **Deterministic Offline Replay** rather than real-time streaming BCI.

---

## 4. Coordinate Frames and Target Mapping

### 4.1 Physical Workspace Frame
The Leap Motion optical tracking sensor records Cartesian hand positions in millimeters (`mm`) relative to its optical baseline:
- **$X$-axis:** Lateral axis (Left $< 0$ mm, Right $> 0$ mm).
- **$Y$-axis:** Vertical elevation ($Y > 0$ mm above table, nominal resting height $\sim 149$ mm).
- **$Z$-axis:** Depth / forward extension ($Z > 0$ mm away from user, nominal depth $\sim 52$ mm).

### 4.2 Empirical Target Centroid Mapping
Analysis of the 269 valid experimental trials reveals two discrete, tightly clustered physical reaching targets:

$$\mathbf{p}_{\text{target, LEFT}} = [-188.0, 150.0, 53.0]^\top \text{ mm}$$

$$\mathbf{p}_{\text{target, RIGHT}} = [+187.5, 149.0, 51.0]^\top \text{ mm}$$

Empirical verification:
- On Congruent trials: Goal `left` reaches to $[-188.2, 150.0, 53.0]$ mm; Goal `right` reaches to $[+187.7, 148.7, 50.9]$ mm.
- On Incongruent trials: Goal `left` reaches to $[+186.9, 149.7, 50.4]$ mm (stimulus-response reversal); Goal `right` reaches to $[-187.9, 149.4, 53.1]$ mm.

Therefore, the discrete cognitive target goal $\in \{\text{"left"}, \text{"right"}\}$ maps deterministically to 3D spatial target centroids in the Leap Motion coordinate system:

$$\mathcal{M}(\text{"left"}) = \mathbf{p}_{\text{target, LEFT}}, \quad \mathcal{M}(\text{"right"}) = \mathbf{p}_{\text{target, RIGHT}}$$

---

## 5. Conflict Measurement Formulations

Phase 4 introduces an interpretable, multimodal Intention–Action Conflict Engine with four orthogonal conflict metrics:

### 5.1 Measurement A: Predicted Endpoint-to-Target Distance ($d_{\text{target}}$)
Given the decoded neural goal $\hat{g} \in \{\text{"left"}, \text{"right"}\}$ with mapped 3D target coordinates $\mathbf{p}_{\text{target}} = \mathcal{M}(\hat{g})$ and the kinematic forecaster's predicted hand endpoint $\hat{\mathbf{p}}_{\text{end}} \in \mathbb{R}^3$:

$$d_{\text{target}} = \|\hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{target}}\|_2 \quad (\text{in mm})$$

A normalized continuous conflict index $C_{\text{distance}} \in [0, 1]$ is defined using an empirical reference threshold $d_{\text{ref}} = 100.0$ mm (half the lateral target separation $\approx 187$ mm):

$$C_{\text{distance}} = \min\left(1.0, \frac{d_{\text{target}}}{d_{\text{ref}}}\right)$$

### 5.2 Measurement B: Directional Disagreement Angle ($\theta_{\text{dir}}$)
Comparing the intended movement vector $\mathbf{v}_{\text{target}} = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{obs}}(0)$ and the predicted movement vector $\hat{\mathbf{v}}_{\text{pred}} = \hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{obs}}(0)$:

$$\cos \theta = \frac{\hat{\mathbf{v}}_{\text{pred}} \cdot \mathbf{v}_{\text{target}}}{\|\hat{\mathbf{v}}_{\text{pred}}\|_2 \|\mathbf{v}_{\text{target}}\|_2 + \epsilon}$$

$$\theta_{\text{dir}} = \arccos(\text{clip}(\cos \theta, -1.0, 1.0)) \quad (\text{in degrees})$$

### 5.3 Measurement C: Neural Goal–Decoded Agreement
Assesses whether the decoded intention $\hat{g}$ matches the experimental instructed goal $g_{\text{true}}$:

$$\text{Agreement}_{\text{goal}} = \mathbb{I}(\hat{g} = g_{\text{true}})$$

Reported separately from physical trajectory conflict to prevent confounding cognitive decoding accuracy with motor execution reversal.

### 5.4 Measurement D: Retrospective Forecast Error
Retrospective tracking error against future ground truth $\mathbf{p}_{\text{end}}$:

$$\text{Error}_{\text{endpoint}} = \|\hat{\mathbf{p}}_{\text{end}} - \mathbf{p}_{\text{end}}\|_2 \quad (\text{in mm})$$

### 5.5 Discrete Conflict States
The conflict engine categorizes each evaluation into transparent states:
- `ALIGNED`: $d_{\text{target}} \le 50.0$ mm and $\theta_{\text{dir}} \le 45.0^\circ$.
- `POTENTIAL_DISAGREEMENT`: $d_{\text{target}} > 50.0$ mm or $\theta_{\text{dir}} > 45.0^\circ$.
- `UNCERTAIN`: Input signal confidence $< 0.55$ or high trajectory variance.
- `UNAVAILABLE`: Missing kinematic prefix, unmapped target, or missing sensor stream.
- `INVALID_INPUT`: NaN/inf coordinates, malformed trial, or corrupt timestamps.

---

## 6. Implementation Architecture & Next Steps

The Phase 4 software suite will reside in `src/phase_4/`:

```
src/phase_4/
├── __init__.py               # Package root
├── config.py                 # Dataclasses & hyperparameter configurations
├── schemas.py                # Typed dataclasses for multimodal twin contracts
├── trial_registry.py         # Multi-subject trial cache, manifest validation, QC
├── neural_adapter.py         # Phase 2 model loader, feature extractor, predictor
├── kinematic_adapter.py      # Phase 3 forecaster loader (with held-out sub-01 support)
├── coordinate_mapping.py     # 3D spatial target zones and coordinate transforms
├── temporal_alignment.py     # Clock synchronization & causality validator
├── conflict_engine.py        # 4-metric conflict analyzer and state classifier
├── fusion_engine.py          # Multimodal state fusion engine
├── replay_engine.py          # Deterministic causal trial step simulator
├── event_logger.py           # Structured event logging and export utilities
├── dashboard.py              # Zero-dependency interactive web dashboard (http.server)
└── cli.py                    # Command-line interface for twin playback & evaluation
```

All 31 existing tests across `test_phase_2.py`, `test_phase_3.py`, and `test_validation.py` remain the non-negotiable baseline. Phase 4 will introduce comprehensive unit and regression tests in `tests/test_phase_4.py`.
