# NeuroAction — Research Notes & Epistemic Boundaries

This document provides transparent scientific disclosures regarding the empirical constraints, data assumptions, and epistemic boundaries of the **NeuroAction** project.

---

## 1. Physical Target Coordinates & Empirical Proxies

### Observation
The public OpenNeuro dataset `ds006840` provides continuous Leap Motion tracking of the participant's hand during reach movements toward left and right targets. However, the repository does not contain independently surveyed 3D physical coordinates of the target markers or computer monitor screen in the laboratory frame.

### Scientific Resolution
- Left Target Proxy: $[-188, 150, 53]$ mm
- Right Target Proxy: $[+188, 149, 51]$ mm
These coordinates represent empirical centroid clusters calculated across clean reach trial endpoints. In all documentation and user interfaces, these are explicitly designated as **empirical proxy centroids**, not ground-truth surveying measurements.

---

## 2. Visual Go-Cue vs. Kinematic Movement Onset

### Empirical Discovery (Phase 3)
In the raw BIDS event stream, an event marker denotes the onset of the visual go-cue ($t = 0$). An initial baseline attempt to forecast reach trajectories using the first 200 ms following the go-cue yielded high error ($> 120$ mm RMSE).

### Investigation & Root Cause
Human visuomotor reaction times impose an involuntary physiological latency ($280 \pm 65$ ms) between visual stimulus appearance and physical hand departure. Consequently, during $[0, 200]$ ms relative to the go-cue, the participant's hand remained virtually stationary at the resting origin ($v < 20$ mm/s).

### Epistemic Policy
All kinematic trajectory forecasting models in NeuroAction are anchored to **physical kinematic onset** ($v_{\text{hand}} > 50$ mm/s), ensuring that the 200 ms prefix represents true early reaching dynamics.

---

## 3. Condition Discrimination vs. Involuntary Motor Slips

### Disclosure
In Phase 5, the multimodal fusion model attained an $88.35\%$ Balanced Accuracy and $0.9395$ AUROC when classifying congruent vs. incongruent trials.

### Critical Epistemic Distinction
It is crucial to recognize that this metric measures **experimental condition discrimination** (differentiating trials where the stimulus reversed direction vs. standard trials). It does **not** demonstrate the detection of spontaneous, involuntary biological motor slips in healthy individuals. The dataset was recorded under instructed stimulus manipulations, not spontaneous error-correction trials.

---

## 4. Zero-Phase Non-Causal Filtering Disclosure

### Electrophysiology Preprocessing
Raw scalp EEG potentials were filtered using a zero-phase, two-pass Butterworth bandpass filter ($0.5$–$40$ Hz).

### Real-Time Translation Warning
Zero-phase filtering uses both forward and backward passes across the timeseries, incorporating future sample information ($t > 0$). While standard and necessary in offline neuroscience analysis to prevent phase distortion, this filter is strictly **non-causal**. Before deploying this pipeline in closed-loop prosthetics or real-time robotic interventions, causal minimum-phase IIR filters must be substituted, which will incur small group delays.

---

## 5. Four-Level Scientific Achievement Hierarchy

| Level | Milestone Definition | Status | Evidence / Audit Justification |
| :---: | :--- | :---: | :--- |
| **Level 1** | **Software Implementation** | **VERIFIED** | 16 modular components, deterministic acausal-free replay, 108/108 passing automated tests. |
| **Level 2** | **Existing-Data Retrospective Evaluation** | **VERIFIED** | Full 269-trial LOSO evaluation on `ds006840`; Ridge forecaster achieves 44.8% error reduction. |
| **Level 3** | **Independent Outcome Prediction** | **PENDING** | `ds006840` lacks independently surveyed 3D physical targets (centroids are empirical proxies). |
| **Level 4** | **Prospective External Validation** | **PENDING** | Awaiting execution of the prospective human recording protocol (`reports/phase_5/prospective_validation_protocol.md`). |
