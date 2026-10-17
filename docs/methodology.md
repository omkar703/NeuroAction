# NeuroAction — Scientific Methodology & Mathematical Formulations

This document provides a comprehensive mathematical and methodological specification of the **NeuroAction** neural intention decoding and 3D trajectory forecasting pipeline.

---

## 1. Experimental Reach Paradigm & Coordinate Space

The research investigates a planar-reaching task captured concurrently via 26-channel scalp EEG (200 Hz) and Leap Motion optical tracking (100 Hz).

### Cartesian Workspace Geometry
- **Origin** ($0, 150, 50$ mm): The resting hand position above the Leap Motion optical sensor.
- **Left Target Proxy** ($-188, 150, 53$ mm): Lateral reach to the left hemisphere.
- **Right Target Proxy** ($+188, 149, 51$ mm): Lateral reach to the right hemisphere.
- **Coordinate axes**:
  - $X$: Lateral displacement ($\text{Left} < 0 < \text{Right}$)
  - $Y$: Height above sensor ($\sim 150$ mm)
  - $Z$: Forward depth ($\sim 50$ mm)

---

## 2. Neural Intention Decoding (Task A)

Pre-movement cognitive intentions are decoded from scalp EEG during the preparatory period prior to overt movement onset ($[-1000, 0]$ ms).

### Signal Conditioning
Raw scalp potentials $S \in \mathbb{R}^{C \times T}$ ($C = 26$ channels, $T = 200$ samples) undergo zero-phase bandpass filtering isolating sensorimotor mu ($\mu$: 8–12 Hz) and beta ($\beta$: 13–30 Hz) rhythm desynchronisation:

$$\tilde{S}_{\mu} = \mathcal{B}_{\mu}(S), \quad \tilde{S}_{\beta} = \mathcal{B}_{\beta}(S)$$

### Common Spatial Pattern (CSP) Decomposition
Common Spatial Patterns design spatial filters $W \in \mathbb{R}^{C \times K}$ that maximize the variance for one condition while minimizing it for the other:

$$\max_{W} \frac{W^T \Sigma_{\text{left}} W}{W^T \Sigma_{\text{right}} W}$$

where $\Sigma_{\text{left}}$ and $\Sigma_{\text{right}}$ are class-conditional spatial covariance matrices:

$$\Sigma_c = \frac{1}{N_c} \sum_{i=1}^{N_c} \frac{S_i S_i^T}{\text{Tr}(S_i S_i^T)}$$

The feature vector $f \in \mathbb{R}^{2K}$ is extracted as log-variance of projected signals:

$$f_k = \log \left( \frac{\text{var}(w_k^T S)}{\sum_{j=1}^{2K} \text{var}(w_j^T S)} \right)$$

### Linear Discriminant Analysis (LDA)
The projected CSP features are classified via Linear Discriminant Analysis:

$$y_{\text{goal}} = \text{sign}\left( w_{\text{lda}}^T f + b \right)$$

yielding posterior class probability $P(\text{Left} \mid S) = \sigma(w_{\text{lda}}^T f + b)$.

---

## 3. Kinematic Trajectory Forecasting (Task B)

Kinematic trajectory forecasting extrapolates continuous 3D limb coordinates $(x(t), y(t), z(t))$ to predict the final reach endpoint $(x_{\text{end}}, y_{\text{end}}, z_{\text{end}})$ from an early observation prefix of duration $T_{\text{prefix}} = 200$ ms ($K = 20$ samples).

### Kinematic Onset vs. Visual Go-Cue
A critical scientific finding in Phase 3 was that human reaction time delays physical hand departure by $280 \pm 65$ ms following the visual go-cue. Evaluating prefixes relative to visual go-cue resulted in evaluating stationary resting hands. NeuroAction therefore anchors prefix windows strictly to **physical kinematic onset** ($v_{\text{hand}} > 50$ mm/s).

### Ridge Regression Forecaster
Given the prefix state vector stacking 3D positions, velocities, and accelerations:

$$X_{\text{prefix}} = \begin{bmatrix} p(t_1) & v(t_1) & a(t_1) & \cdots & p(t_K) & v(t_K) & a(t_K) \end{bmatrix}^T$$

Ridge regression solves the regularized least-squares objective:

$$\hat{W}_{\text{ridge}} = \arg\min_W \| Y_{\text{endpoint}} - X W \|_2^2 + \lambda \| W \|_2^2 = (X^T X + \lambda I)^{-1} X^T Y_{\text{endpoint}}$$

with $L_2$ regularization parameter $\lambda = 1.0$.

### Baseline Comparators
1. **Static Persistence Baseline**: Assumes no further movement after prefix:
   $$\hat{p}_{\text{end}} = p(T_{\text{prefix}})$$
2. **Fair Constant Velocity Baseline**: Extrapolates position linearly using instantaneous velocity at prefix termination:
   $$\hat{p}_{\text{end}} = p(T_{\text{prefix}}) + v(T_{\text{prefix}}) \cdot \bar{T}_{\text{remaining}}$$

---

## 4. Cross-Validation Protocol: Leave-One-Subject-Out (LOSO)

To prevent data leakage and guarantee that models generalize across distinct human brains and reach kinematics:
- All models undergo 5-fold Leave-One-Subject-Out (LOSO) evaluation.
- In each fold $k \in \{1, \dots, 5\}$, four participants form the training set ($N_{\text{train}} \approx 215$ trials), and the fifth participant forms the strictly held-out test set ($N_{\text{test}} \approx 54$ trials).
- No test-participant EEG statistics, normalizations, or kinematic patterns contaminate feature extraction or model fitting.

---

## 5. Intention–Action Congruence & Conflict Engine

The Phase 4 fusion engine evaluates cross-modal agreement between the pre-movement neural intention prediction $\hat{g}_{\text{neural}} \in \{\text{Left}, \text{Right}\}$ and early kinematic trajectory $\hat{p}_{\text{forecast}}$:

$$\Delta_{\text{congruence}} = \text{sign}(\hat{x}_{\text{pred}}) \cdot \text{sign}(x_{\text{target}}(\hat{g}_{\text{neural}}))$$

Discordance is flagged when $\Delta_{\text{congruence}} < 0$ with neural decoder confidence exceeding decision threshold $\tau_{\text{conf}} \ge 0.65$.
