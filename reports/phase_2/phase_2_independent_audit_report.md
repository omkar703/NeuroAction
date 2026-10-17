# Independent Scientific and Statistical Audit: NeuroAction Phase 2

**Project:** NeuroAction — Neural Intention–Action Digital Twin  
**Subproject Audited:** Phase 2: Neural Intention Representation, Goal–Motor Dissociation & Temporal Commitment Analysis  
**Dataset:** IACKD (OpenNeuro `ds006840`, Version `1.0.0`, Nature *Scientific Data* 2026)  
**Auditor:** Independent Reviewer (Specialization: Computational Neuroscience, BCI, Electrophysiology, ML Verification)  
**Date of Audit:** October 2026  
**Primary Deliverable:** `reports/phase_2/phase_2_independent_audit_report.md`  

---

## A. Executive Verdict

**Verdict:** **Promising but Requiring Corrections**

### Summary of Verdict Rationale
Phase 2 establishes a well-engineered, reproducible computational pipeline with rigorous zero-leakage cross-validation, proper software architecture, and verified data provenance. The empirical observation that decoders trained on congruent visuomotor trials generalize to incongruent trials by predicting the spatial target goal rather than the physical hand direction is robust and computationally verified.

However, several critical and major scientific, statistical, and interpretative issues prevent Phase 2 from being defensible as originally reported:
1. **Mathematical Tautology Presented as Dual Discovery:** The reported $70.36\%$ goal accuracy versus $29.64\%$ motor accuracy under Congruent $\to$ Incongruent transfer does **not** represent two independent scientific findings. Because goal and motor labels are exact logical complements on incongruent trials ($y_{\text{motor}} \equiv 1 - y_{\text{goal}}$), any classifier tested on this condition will mathematically exhibit $\text{Accuracy}_{\text{motor}} \equiv 1.0 - \text{Accuracy}_{\text{goal}}$. Below-chance motor accuracy is a necessary algebraic consequence of goal transfer, not empirical proof that the motor cortex lacks motor commands.
2. **Left-Censored "Neural Commitment Point" (−900 ms):** The reported −900 ms Neural Commitment Point is an artifact of epoch boundaries and stimulus timing. In the IACKD paradigm, the visual cue appears at $t = -1000$ ms. Because decoding accuracy is already $\sim 69\%$ in the very first analyzed window ($[-1000, -800]$ ms, center $-900$ ms), the pipeline merely detects the earliest window available in the dataset. It does not observe a physiological transition from chance to commitment. Moreover, whole-scalp decoding in this early window is heavily confounded by visual evoked potentials (VEPs).
3. **Statistical Degrees of Freedom and Confidence Band Underestimation:** Across the Leave-One-Subject-Out (LOSO) evaluations ($N = 5$ subjects), uncertainty intervals were computed using a standard normal critical value ($Z = 1.96$) rather than Student's $t$ with $df = 4$ ($t_{0.025, 4} = 2.776$). This systematically underestimated confidence intervals and uncertainty bands by **$41.6\%$**.
4. **Pseudo-Replication in Electrophysiological Hypothesis Testing:** Readiness Potential late slope significance was computed by regressing the 100 correlated time samples of the **grand-average waveform** against time, treating temporal sample points as independent observations ($df = 98$, yielding inflated $p < 10^{-6}$), rather than testing slope variability across participants ($df = 4$).
5. **Restricted Cohort Generalization:** Although the IACKD dataset contains 15 subjects ($6,671$ valid trials), the empirical modeling was executed on a 5-subject benchmark cohort ($269$ trials). Language in summary reports occasionally conflates dataset-wide properties with cohort-level modeling results.

With the corrections and tempered interpretations specified in this audit, the core findings provide a solid empirical basis for progressing to Phase 3.

---

## B. Key Findings Table

| ID | Severity | Finding | Evidence | Consequence | Required Correction |
|:---|:---:|:---|:---|:---|:---|
| **F-01** | **Critical** | **Coupled Tautological Transfer Metrics:** $70.36\%$ goal vs $29.64\%$ motor accuracy presented as dual independent findings. | On incongruent trials, $y_{\text{motor}} = 1 - y_{\text{goal}}$. Models trained on congruent trials are identical. | Misleadingly suggests two separate physiological discoveries; misinterprets algebraic inversion ($1 - p$) as active neural suppression of motor commands. | Clarify that motor transfer accuracy is mathematically coupled ($\text{Acc}_{\text{motor}} = 1 - \text{Acc}_{\text{goal}}$). Base motor non-decodability claims on the `Motor: All -> All` orthogonal condition ($52.7\%$), not transfer inversion. |
| **F-02** | **Critical** | **Left-Censored Neural Commitment Point (−900 ms):** Claim that intention forms at $-900$ ms overstates the evidence. | `premove` onset occurs at $-1.0$ s (stimulus cue onset); Window 0 is $[-1000, -800]$ ms. Accuracy is already $69.2\%$ in Window 0. | Misidentifies stimulus-locked visual processing as the physiological onset of motor commitment; ignores pre-stimulus baseline. | Rename to *"Earliest Evaluated Pre-Movement Information"*; document left-censoring; acknowledge that stimulus onset occurs at $-1000$ ms. |
| **F-03** | **Major** | **Underestimated Uncertainty Intervals ($Z = 1.96$ vs $t = 2.776$):** 95% CIs assume $N \to \infty$ rather than $N = 5$. | `src/models/cross_validation.py` line 243: `mean_acc - 1.96 * sem_acc`. | Confidence intervals are $41.6\%$ narrower than statistically valid for $df = 4$, overstating statistical certainty. | Recompute all LOSO 95% CIs using Student's $t$ critical value ($t_{0.025, 4} = 2.776$). Update tables and figure shaded bands. |
| **F-04** | **Major** | **Sample Size Pseudo-Replication in RP Slopes:** Regression on averaged ERP waveform time points. | `src/neuroscience/readiness_potential.py` lines 88–94: `stats.linregress(t_late, y_late)` where `y_late` is the grand-average ERP. | Degrees of freedom inflated to $N_{\text{time}} - 2 = 98$; yields artificially tiny $p = 3.97 \times 10^{-7}$ by ignoring inter-subject variance. | Test slopes at the participant level ($df = 4$, $t = -3.83, p = 0.0186$) or trial level via linear mixed-effects modeling. |
| **F-05** | **Major** | **Visual Evoked Potential (VEP) Confounding:** Abstract intention decoding conflated with cue sensory processing. | Channel ablation drops $-9.75\%$ when excluding parieto-occipital electrodes; Window 0 covers $0$–$200$ ms post-visual cue. | Early decoding reflects sensory processing of red/yellow cues and extrapersonal target coordinates, not pure motor intention. | Acknowledge that pre-movement EEG reflects a compound of visual cue perception, visuospatial attention, and motor preparation. |
| **F-06** | **Moderate** | **Incomplete Prediction Artifacts:** `out_of_fold_predictions.csv` lacks temporal window and transfer condition records. | `src/models/cross_validation.py` lines 294–295 filters OOF saves strictly to `tr_c == "all" and te_c == "all"` and `window == None`. | Figures 4, 5, 7, and 9 cannot be regenerated solely from the prediction CSV without rerunning cross-validation. | Serialize out-of-fold trial predictions for all condition pairs and temporal sliding windows into separate or unified CSV artifacts. |
| **F-07** | **Moderate** | **Post-Cue SMR Baseline Window:** ERD/ERS computed relative to $[-1.0, -0.8]$ s post-stimulus. | `src/neuroscience/sensorimotor_rhythms.py` line 49: `baseline_window = (-1.0, -0.8)`. | Baseline contains stimulus-evoked potentials and early cortical activation rather than true resting EEG. | Document that the baseline is an intra-epoch reference rather than a resting pre-stimulus baseline. |
| **F-08** | **Minor** | **Zero-Phase Acausal Filtering Caveat:** Official BIDS derivative used zero-phase FIR filtering. | `reports/phase_2/phase_2_report.md` Section 4 correctly identifies this; needs reinforcement in abstract/conclusions. | Minor temporal leakage across movement onset ($t = 0$) prevents interpreting temporal curves as causal streaming performance. | Reiterate prominently that temporal curves reflect offline information bounds, requiring causal filtering for Phase 4 deployment. |

---

## C. Independently Verified Metrics

All models were evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation across the 5 benchmark participants (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`, total $N = 269$ valid trials). 

### 1. Primary Classification Benchmarks (All $\to$ All Mapping, $[-1000, 0]$ ms Epoch)

| Model | Target | Reported Balanced Acc | Recalculated Unweighted Mean (LOSO) | Recalculated Pooled Trial Acc | Reported SEM | True Student-$t$ 95% CI ($df=4$) | Reported Normal CI ($Z=1.96$) | AUROC (LOSO) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chance** | Goal | $50.00\%$ | **$50.00\%$** | $50.00\%$ | $0.0000$ | $[0.500, 0.500]$ | $[0.500, 0.500]$ | $0.500$ | **Verified** |
| **Chance** | Motor | $50.00\%$ | **$50.00\%$** | $50.00\%$ | $0.0000$ | $[0.500, 0.500]$ | $[0.500, 0.500]$ | $0.500$ | **Verified** |
| **Logistic Reg.** | Goal | $62.23\%$ | **$62.23\%$** | $62.37\%$ | $0.0369$ | $[0.520, 0.725]$ | $[0.550, 0.695]$ | $0.702$ | **Verified** |
| **Logistic Reg.** | Motor | $50.32\%$ | **$50.32\%$** | $50.21\%$ | $0.0173$ | $[0.455, 0.551]$ | $[0.469, 0.537]$ | $0.522$ | **Verified** |
| **LDA** | Goal | $68.50\%$ | **$68.50\%$** | $68.34\%$ | $0.0604$ | $[0.517, 0.853]$ | $[0.567, 0.803]$ | $0.756$ | **Verified** |
| **LDA** | Motor | $52.72\%$ | **$52.72\%$** | $52.43\%$ | $0.0104$ | $[0.498, 0.556]$ | $[0.507, 0.548]$ | $0.505$ | **Verified** |
| **CSP + LDA** | Goal | $50.28\%$ | **$50.28\%$** | $50.90\%$ | $0.0221$ | $[0.441, 0.564]$ | $[0.459, 0.546]$ | $0.470$ | **Verified** |
| **CSP + LDA** | Motor | $48.76\%$ | **$48.76\%$** | $48.65\%$ | $0.0198$ | $[0.433, 0.543]$ | $[0.449, 0.526]$ | $0.510$ | **Verified** |
| **EEGNet** | Goal | $70.38\%$ | **$70.38\%$** | $70.61\%$ | $0.0568$ | $[0.546, 0.861]$ | $[0.593, 0.815]$ | $0.786$ | **Verified** |
| **EEGNet** | Motor | $51.11\%$ | **$51.11\%$** | $50.94\%$ | $0.0185$ | $[0.460, 0.562]$ | $[0.475, 0.547]$ | $0.521$ | **Verified** |

### 2. The Central Transfer Experiment (Congruent $\to$ Incongruent Mapping)

| Model | Target Type | Reported Balanced Acc | Recalculated LOSO Mean | Algebraic Complement ($1 - \text{Acc}$) | Reported SEM | True Student-$t$ 95% CI | AUROC | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LDA** | Goal Transfer | $70.36\%$ | **$70.36\%$** | $1.0 - 0.2964 = 0.7036$ | $0.0539$ | $[0.554, 0.853]$ | $0.758$ | **Verified (Coupled)** |
| **LDA** | Motor Transfer | $29.64\%$ | **$29.64\%$** | $1.0 - 0.7036 = 0.2964$ | $0.0539$ | $[0.147, 0.446]$ | $0.242$ | **Verified (Coupled)** |
| **EEGNet** | Goal Transfer | $70.55\%$ | **$70.55\%$** | $1.0 - 0.2945 = 0.7055$ | $0.0535$ | $[0.557, 0.854]$ | $0.776$ | **Verified (Coupled)** |
| **EEGNet** | Motor Transfer | $29.45\%$ | **$29.45\%$** | $1.0 - 0.7055 = 0.2945$ | $0.0535$ | $[0.146, 0.443]$ | $0.224$ | **Verified (Coupled)** |
| **LogReg** | Goal Transfer | $63.74\%$ | **$63.74\%$** | $1.0 - 0.3626 = 0.6374$ | $0.0477$ | $[0.505, 0.770]$ | $0.677$ | **Verified (Coupled)** |
| **LogReg** | Motor Transfer | $36.26\%$ | **$36.26\%$** | $1.0 - 0.6374 = 0.3626$ | $0.0477$ | $[0.230, 0.495]$ | $0.323$ | **Verified (Coupled)** |

### 3. Metric Aggregation Audit
* **Method Employed:** Unweighted arithmetic mean across participants (LOSO fold macro-average).
* **Implications:** This gives equal weight ($20\%$) to each participant regardless of trial count (`sub-01`: 40, `sub-02`: 52, `sub-03`: 60, `sub-08`: 57, `sub-15`: 60). As verified above, the unweighted LOSO mean ($68.50\%$) matches the reported numbers identically, whereas the trial-level pooled accuracy is $68.34\%$. Unweighted averaging is appropriate for inter-subject BCI evaluation, but requires Student's $t$ scaling for uncertainty estimation.

---

## D. Central Transfer Experiment Audit

### 1. Label Mapping Semantics
In `src/data/trial_loader.py` (lines 18–30) and `src/data/schema.py`:
* **Congruent Trials (Codes 0 & 1):** $\text{Target Goal} \equiv \text{Physical Hand Direction}$.
* **Incongruent Trials (Codes 2 & 3):** $\text{Target Goal} \neq \text{Physical Hand Direction}$ ($\text{Goal} = \text{Left} \iff \text{Hand} = \text{Right}$; $\text{Goal} = \text{Right} \iff \text{Hand} = \text{Left}$).

### 2. Leakage and LOSO Integrity
Tracing the pipeline execution in `src/models/cross_validation.py` lines 167–190:
* The test subject is strictly masked out from training data (`train_mask = ~test_mask`).
* Standard scalers, shrinkage LDA covariance estimation, and EEGNet parameter updates are performed exclusively on `X_tr` (the remaining 4 subjects).
* Spatial CSP filters (`src/features/pre_eeg.py` line 148) are fitted strictly on the training fold.
* **Verdict:** LOSO partitioning has **zero data leakage** across subjects.

### 3. Mathematical Reciprocity Law (Tautological Coupling)
Consider a classifier $f: \mathcal{X} \to \{0, 1\}$ trained on congruent trials.
Because $y_{\text{goal}} = y_{\text{motor}}$ on all congruent trials, training on $y_{\text{goal}}$ produces the **identical mathematical model** as training on $y_{\text{motor}}$.

When evaluating this model on incongruent test trials where labels are balanced ($N_0 = N_1 = N/2$):
For every trial $i$:
$$y_{\text{motor}}^{(i)} = 1 - y_{\text{goal}}^{(i)}$$
A prediction $\hat{y}^{(i)} \in \{0, 1\}$ is correct for the goal if $\hat{y}^{(i)} = y_{\text{goal}}^{(i)}$. It is correct for the motor command if $\hat{y}^{(i)} = y_{\text{motor}}^{(i)} = 1 - y_{\text{goal}}^{(i)} \iff \hat{y}^{(i)} \neq y_{\text{goal}}^{(i)}$.
Consequently:
$$\text{Accuracy}_{\text{motor}} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} = y_{\text{motor}}^{(i)}) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} \neq y_{\text{goal}}^{(i)}) = 1.0 - \text{Accuracy}_{\text{goal}}$$
Likewise, for continuous probability outputs $P(\text{right})$:
$$\text{AUROC}_{\text{motor}} = 1.0 - \text{AUROC}_{\text{goal}}$$

### 4. Strongest Defensible Scientific Conclusion
* **Defensible Claim:** When humans plan movements under visuomotor remapping, preparatory scalp EEG signals ($[-1000, 0]$ ms) consistently track the intended spatial target goal rather than the required physical hand direction. A classifier trained on normal (congruent) reaching transfers to reversed reaching by predicting the goal ($70.36\%$).
* **Overstated / Indefensible Claim:** Claiming that $29.64\%$ motor accuracy proves that the motor cortex does not prepare physical commands is a logical error. The $29.64\%$ score is simply the complement of $70.36\%$. The actual evidence that motor command direction is not stably decodable from pre-movement EEG comes from the **orthogonal mixed condition** (`Motor: All -> All`), where decoders fail to exceed chance ($52.72\% \pm 1.04\%$, AUROC $0.505$).

---

## E. Audit of Figure 4 and Figure 5

```
       FIGURE 4 (Goal Temporal)                 FIGURE 5 (Motor Temporal)
   1.0 ┌─────────────────────────┐          1.0 ┌─────────────────────────┐
       │   All->All (~68-71%)    │              │                         │
   0.8 │──o──o──o──o──o──o──o──o─│          0.8 │                         │
       │                         │              │                         │
   0.6 │.........................│          0.6 │                         │
   0.5 ├─────────────────────────┤          0.5 ├──o──o──o──o──o──o──o──o─┤ All->All (~50%)
   0.4 │   Chance = 0.50         │          0.4 │.........................│ Chance = 0.50
       │                         │          0.3 │--^--^--^--^--^--^--^--^-│ Transfer (~29%)
   0.2 └─────────────────────────┘          0.2 └─────────────────────────┘
     -1000        -500          0             -1000        -500          0
             Time (ms)                                 Time (ms)
```

### 1. Specification of Temporal Curves
* **Window Count & Parameters:** 17 overlapping temporal windows of duration $200$ ms (40 samples @ 200 Hz) with stride $50$ ms (10 samples).
* **Epoch Range:** Window 0 spans $[-1000, -800]$ ms (center $-900$ ms); Window 16 spans $[-200, 0]$ ms (center $-100$ ms).
* **Model Used:** Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.

### 2. Metric and Band Verification
* **Metric Plotted:** Unweighted macro-average balanced accuracy across 5 LOSO folds.
* **Shaded Uncertainty Bands:** Calculated as $\text{Mean} \pm 1.96 \times \text{SEM}$.
* **Audit Finding on Uncertainty Bands:** 
  The shaded regions represent standard error of the mean across the 5 subjects, but scaled by $1.96$. For $N = 5$ ($df = 4$), the true $95\%$ confidence interval requires scaling by $t_{0.025, 4} = 2.776$. Thus, the shaded visual envelope in Figure 4 and Figure 5 is **$29.4\%$ narrower than a true $95\%$ confidence band**, visually exaggerating precision.

### 3. Detailed Trajectory Analysis

#### Figure 4 (Goal Decoding Trajectory)
* Window 0 ($-900$ ms): $69.17\%$ (SEM: $0.0601$, Student-$t$ $95\%$ CI: $[0.525, 0.858]$).
* Window 4 ($-700$ ms): $71.58\%$ (SEM: $0.0366$, Student-$t$ $95\%$ CI: $[0.614, 0.817]$).
* Window 8 ($-500$ ms): $70.90\%$ (SEM: $0.0576$, Student-$t$ $95\%$ CI: $[0.549, 0.869]$).
* Window 16 ($-100$ ms): $62.53\%$ (SEM: $0.0451$, Student-$t$ $95\%$ CI: $[0.500, 0.750]$).
* **Observation:** The curve is flat and elevated above chance across the **entire** epoch. It does not ramp up from baseline.

#### Figure 5 (Motor Command Trajectory)
* Window 0 ($-900$ ms): $45.44\%$ (SEM: $0.0313$, Student-$t$ $95\%$ CI: $[0.368, 0.541]$).
* Window 4 ($-700$ ms): $55.29\%$ (SEM: $0.0212$, Student-$t$ $95\%$ CI: $[0.494, 0.612]$).
* Window 7 ($-550$ ms): $56.23\%$ (SEM: $0.0364$, Student-$t$ $95\%$ CI: $[0.461, 0.663]$).
* Window 16 ($-100$ ms): $49.37\%$ (SEM: $0.0231$, Student-$t$ $95\%$ CI: $[0.430, 0.558]$).
* **Observation:** The All $\to$ All motor curve (red line) hovers within random fluctuation distance of $50\%$. The transfer curve (purple line) is permanently inverted ($\sim 29\%–35\%$).

### 4. What the Plots Actually Demonstrate
* **Demonstrated:** Extrapersonal goal information is continuously present and decodable across the full $[-1000, 0]$ ms epoch. Physical motor command direction is not stably decodable prior to movement onset on mixed datasets.
* **NOT Demonstrated:** The plots do **not** demonstrate the physiological *emergence* or *ignition* of motor intention. Because accuracy starts at $69.2\%$ at the left edge of the plot, the onset of goal decoding occurred prior to the recorded epoch window.

---

## F. Audit of the Claimed −900 ms Neural Commitment Point (NCP)

### 1. Operational Definition in the Code
In `src/models/cross_validation.py` lines 360–410:
An NCP is declared at the first temporal window $i$ where:
$$\text{ci\_low}_i > 0.50 \quad \text{AND} \quad \text{ci\_low}_{i+1} > 0.50$$
where $\text{ci\_low} = \text{mean\_acc} - 1.96 \times \text{SEM}$.

### 2. Flaws in the NCP Implementation
1. **Left-Censored Trivial Detection:**
   At Window 0 ($-900$ ms), `mean_acc` is $0.6917$, `sem` is $0.0601$, and $\text{ci\_low} = 0.5739 > 0.50$.
   At Window 1 ($-850$ ms), `mean_acc` is $0.7110$, `sem` is $0.0276$, and $\text{ci\_low} = 0.6568 > 0.50$.
   The code terminates immediately on loop index $i = 0$!
   The value −900 ms is selected **solely because it is the earliest window tested in the dataset**.
2. **Normal Approximation with Small Sample Size:**
   Using $1.96$ assumes an infinite sample size. For $df = 4$, the Student-$t$ factor is $2.776$.
   Evaluating Window 0 with Student-$t$: $\text{ci\_low}_t = 0.6917 - 2.776 \times 0.0601 = 0.5248 > 0.50$.
   Window 0 remains statistically above chance under Student-$t$, but the safety margin above chance is only $2.48\%$, not $7.39\%$.
3. **No Baseline Comparison:**
   In neuroscience (e.g., Libet readiness potential studies, Schultze-Kraft et al. point-of-no-return), a commitment point is defined as the time when an uncommitted baseline transitions into a committed state. Here, there is no pre-stimulus baseline in `pre_eeg`.

### 3. Independent Hypothesis Testing Across Temporal Windows
To test whether Window 0 and subsequent windows are genuinely significant when properly calibrated, we performed an independent one-sample Student's $t$-test against chance ($0.50$) across the 5 subjects for all 17 windows, applying Benjamini-Hochberg False Discovery Rate (FDR) correction:

| Window | Center Time (ms) | Mean Acc | SEM | $t$-statistic ($df=4$) | Uncorrected $p$-value | FDR $q$-value | Statistically Significant ($q < 0.05$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **−900.0** | **0.6917** | **0.0601** | **3.190** | **0.0166** | **0.0238** | **Yes (Above Chance)** |
| 1 | −850.0 | 0.7110 | 0.0276 | 7.636 | 0.0008 | 0.0078 | Yes |
| 2 | −800.0 | 0.6931 | 0.0503 | 3.838 | 0.0092 | 0.0204 | Yes |
| 3 | −750.0 | 0.6651 | 0.0270 | 6.120 | 0.0018 | 0.0088 | Yes |
| 4 | −700.0 | 0.7158 | 0.0366 | 5.897 | 0.0021 | 0.0088 | Yes |
| 5 | −650.0 | 0.6634 | 0.0573 | 2.853 | 0.0231 | 0.0265 | Yes |
| 6 | −600.0 | 0.6980 | 0.0389 | 5.096 | 0.0035 | 0.0119 | Yes |
| 7 | −550.0 | 0.6542 | 0.0543 | 2.841 | 0.0234 | 0.0265 | Yes |
| 8 | −500.0 | 0.7090 | 0.0576 | 3.628 | 0.0111 | 0.0204 | Yes |
| 9 | −450.0 | 0.6856 | 0.0638 | 2.912 | 0.0218 | 0.0265 | Yes |
| 10 | −400.0 | 0.6569 | 0.0494 | 3.177 | 0.0168 | 0.0238 | Yes |
| 11 | −350.0 | 0.7094 | 0.0286 | 7.331 | 0.0009 | 0.0078 | Yes |
| 12 | −300.0 | 0.6585 | 0.0349 | 4.537 | 0.0053 | 0.0149 | Yes |
| 13 | −250.0 | 0.6137 | 0.0321 | 3.540 | 0.0120 | 0.0204 | Yes |
| 14 | −200.0 | 0.6235 | 0.0344 | 3.588 | 0.0115 | 0.0204 | Yes |
| 15 | −150.0 | 0.5924 | 0.0406 | 2.277 | 0.0426 | 0.0426 | Yes |
| 16 | −100.0 | 0.6253 | 0.0451 | 2.780 | 0.0249 | 0.0265 | Yes |

### 4. Audit Conclusion on the NCP
While every pre-movement window is statistically above chance even after strict FDR correction ($q < 0.05$), claiming a **"Neural Commitment Point at −900 ms"** is scientifically unsupportable. It implies that $-900$ ms is when commitment *began*, whereas the statistical test confirms that intention-related information was **already fully present at the earliest available moment of the recording**.

---

## G. Preprocessing, Temporal Leakage & Visual Cue Confounding

### 1. Provenance and Zero-Phase Filtering
* The pre-movement derivative (`pre_eeg`) was processed by the IACKD dataset authors using a zero-phase (forward-backward) FIR bandpass filter ($0.1–45$ Hz) and $50$ Hz notch filter.
* **Mathematical Risk:** Zero-phase filtering is acausal. The filter output at time $t$ contains linear weighted terms from times $t + \tau$. Consequently, neural signals occurring *after* movement onset ($t > 0$, such as motor execution or sensory reafference) can bleed backwards across the $t = 0$ boundary into pre-movement samples within the filter impulse response length ($\sim 50–100$ ms).
* **Impact on Results:** This leakage primarily threatens samples near $t = 0$ (Windows 15 and 16). It does not explain the high decoding accuracy observed at $-900$ to $-500$ ms.

### 2. Experimental Timeline and Visual Cue Confounding
Inspection of `dataset/sub-01/eeg/sub-01_task-ihc_acq-left_run-01_events.tsv` reveals the exact trial structure:
* `premove` event onset: $t_{\text{stim}} = \text{cue presentation}$.
* `premove` duration: exactly $1.000$ second.
* `execution` event onset: $t_{\text{stim}} + 1.000$ s (movement start).

```
 ───┼─────────────────────────────────────────────┼──────────────────────► Time
    │◄─────────── 1.0 s Pre-Movement ────────────►│
 Stimulus Cue                                  Movement Onset
 (Red/Yellow ball)                                (t = 0)
    │                                             │
    ▼                                             ▼
 Visual Evoked Potentials                  Motor Execution
 (P100, N1, Visuomotor Mapping)            (Hand Kinematics)
 Window 0: [-1000, -800] ms
```

**Critical Scientific Consequence:**
In this paradigm, $t = -1000$ ms is the **onset of visual stimulation**.
* Window 0 ($[-1000, -800]$ ms, center $-900$ ms) corresponds to the **first 200 ms following stimulus onset**.
* During this window, primary visual cortex (`O1, Oz, O2`), extrastriate cortex, and parietal association areas exhibit sharp visual evoked potentials (P100, N170) encoding stimulus color and spatial hemifield.
* Therefore, the high classification accuracy in Window 0 ($69.2\%$) is driven substantially by **visuoperceptual stimulus processing**, not an endogenous, internally generated motor intention.

---

## H. Dataset Scope, Sample Size & Generalization

### 1. Reconciling Dataset Counts
* **Total Dataset Population:** 15 enrolled subjects ($s01$–$s15$), 120 task runs, 7,040 recorded trials, 6,671 valid multimodal trials.
* **Phase 2 Benchmark Cohort:** 5 subjects (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`), 280 indexed trials.
* **Rejected Trials in Cohort:** 11 trials dropped:
  * `sub-08`: 3 trials (0-based `[7, 8, 27]`, 1-based `[8, 9, 28]`) dropped due to Leap Motion hand tracking failure.
  * `sub-02`: 8 trials dropped due to execution recording misalignment.
* **Analyzed Valid Trials:** **269 valid trials**.

### 2. Small-N Subject Variance and `sub-08` Performance
Performance across the 5 subjects reveals marked heterogeneity:

| Participant | Valid Trials | Goal Balanced Acc (LDA) | Motor Balanced Acc (LDA) | Goal Transfer Acc (Cong $\to$ Incong) |
|:---|:---:|:---:|:---:|:---:|
| `sub-01` | 40 | $72.50\%$ | $52.50\%$ | $73.33\%$ |
| `sub-02` | 52 | $73.08\%$ | $49.63\%$ | $76.92\%$ |
| `sub-03` | 60 | $86.67\%$ | $55.00\%$ | $86.67\%$ |
| `sub-08` | 57 | **$53.57\%$** | $51.48\%$ | **$53.57\%$** |
| `sub-15` | 60 | $56.67\%$ | $55.00\%$ | $61.29\%$ |
| **Cohort Mean** | **269** | **$68.50\%$** | **$52.72\%$** | **$70.36\%$** |

* `sub-03` achieves exceptional goal decodability ($86.67\%$).
* `sub-08` is barely above chance ($53.57\%$).
* With $N = 5$, degrees of freedom is $df = 4$. True $95\%$ confidence intervals must use $t = 2.776$, widening the Goal LDA interval to $[51.7\%, 85.3\%]$.

### 3. Hierarchy of Generalization Claims

```
   LEVEL 1: Held-out participant in 5-subject cohort        [EMPIRICALLY VERIFIED]
   LEVEL 2: Generalization to remaining 10 IACKD subjects   [HIGH PLAUSIBILITY, UNTESTED]
   LEVEL 3: Generalization to unseen sessions / days        [UNKNOWN (Single-Session Data)]
   LEVEL 4: Generalization to motor-impaired clinical users [UNKNOWN (Healthy Cohort Only)]
   LEVEL 5: Real-time causal streaming deployment           [NOT DEMONSTRATED (Acausal Filter)]
```

The report should explicitly bound its conclusions to Level 1 and avoid implying that Level 4 or Level 5 has been demonstrated.

---

## I. Other Neuroscientific Claims Audit

### 1. Bereitschaftspotential / Readiness Potential (RP)
* **Reported Finding:** Late RP slope on Cz is $-3.77\ \mu\text{V/s}$ ($p < 10^{-15}$).
* **Audit Finding:**
  The negative deflection prior to movement onset on sensorimotor electrodes is real and matches classical electrophysiology (Kornhuber & Deecke).
  However, the reported $p$-value was derived from `stats.linregress` over the 100 time points of the **mean waveform** ($df = 98$), which commits pseudo-replication.
  * Correct subject-level one-sample $t$-test across the 5 subjects ($df = 4$):
    Slopes: $[-2.61, -6.12, -3.76, -5.61, -0.75]\ \mu\text{V/s}$.
    Mean = $-3.77\ \mu\text{V/s}$, $t = -3.83$, **$p = 0.0186$**.
  * The effect remains statistically significant, but the true $p$-value is $0.019$, not $10^{-15}$.

### 2. Sensorimotor Rhythms (SMR) $\mu$ and $\beta$ ERD/ERS
* **Reported Finding:** Pre-movement $\mu$ and $\beta$ power suppression (ERD) peaking near movement onset.
* **Audit Finding:**
  Power envelopes were correctly extracted using Hilbert transforms.
  However, the baseline window ($[-1.0, -0.8]$ s) is post-cue. It measures power suppression relative to the stimulus presentation period rather than true rest.

### 3. Whole-Scalp vs. Motor Strip Channel Ablation
* **Reported Finding:** Restricting electrodes to 9 motor channels (`FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4`) reduces Goal LDA accuracy by $-9.75\%$ (from $68.50\%$ to $58.75\%$).
* **Audit Finding:**
  The code and result are verified.
  However, the interpretation that this "proves parieto-occipital cortical localization" oversteps what scalp EEG can claim without source modeling (e.g. beamforming/sLORETA). Due to volume conduction and reference effects, channel-level degradation indicates distributed scalp topography, but does not prove localized cortical origin.

---

## J. Reproducibility & Actionable Recommendations for Phase 3

### 1. Reproducibility Assessment
The Phase 2 pipeline is exceptionally well-structured and reproducible:
* Environment: Python 3.10 with PyTorch, Scikit-learn, Scipy, and H5py.
* Automated Tests: All 18 automated tests in `tests/test_validation.py` and `tests/test_phase_2.py` pass without error.
* Code Cleanliness: Functions are modular, docstrings are clear, and random seeds are consistently seeded across folds.

### 2. Actionable Recommendations for Phase 2 Revisions & Phase 3 Transition

#### Immediate Phase 2 Documentation Corrections
1. **Revise the Central Transfer Claim:** Clearly state that the $70.36\%$ goal vs $29.64\%$ motor accuracy on transfer is an algebraically coupled outcome of evaluating a single classifier against complementary labels. Frame the evidence for lack of preparatory motor decoding around the orthogonal `Motor: All -> All` baseline ($52.7\%$).
2. **Re-label the −900 ms Point:** Replace "Neural Commitment Point at −900 ms" with "Earliest Evaluated Pre-Movement Goal Decodability". Acknowledge stimulus cue presentation at $-1000$ ms.
3. **Correct Uncertainty Scaling:** Replace $Z = 1.96$ with Student-$t$ ($t = 2.776$) in all 95% CI tables and plot shaded regions.
4. **Correct the RP Slope $p$-Value:** Update the reported $p$-value from $p < 10^{-15}$ to the subject-level value $p = 0.0186$.

#### Methodological Requirements for Phase 3 (Kinematics & Forecaster)
1. **Continuous 3D Trajectory Formulation:** Utilize the Leap Motion kinematics stream ($100$ Hz) already aligned in `CanonicalTrial` (`hand_x`, `hand_y`, `hand_z`). Build autoregressive trajectory forecasters (e.g. LSTM, Kalman Filter, or MLP) using the early execution window ($0$ to $200$ ms).
2. **Expand the LOSO Evaluation Cohort:** Use `OpenNeuroClient` to download and cache the remaining 10 subjects from OpenNeuro S3, expanding the benchmark from $N = 5$ to the full $N = 15$ dataset to achieve higher statistical power ($df = 14$).
3. **Transition to Causal Filtering for the Real-Time Twin:** Implement single-pass minimum-phase / causal IIR filtering for any EEG decoding intended to feed the live Phase 4 conflict engine.
4. **Serialize Full Out-of-Fold Predictions:** Ensure that future cross-validation runs export trial-level predictions for all condition pairs and temporal windows into CSV artifacts.

---

*End of Independent Audit Report.*
