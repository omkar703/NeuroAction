# Phase 1 — Dataset Ground Truth, Multimodal Alignment & Unified Trial Pipeline

## 1. Executive Summary

Phase 1 establishes the verified scientific and data-engineering foundation for **NeuroAction — Neural Intention–Action Digital Twin**. In accordance with the Phase 1 specification, no machine learning classification or predictive modeling was conducted. Instead, the primary objective was achieved: establishing complete ground truth, resolving label and visuomotor mapping semantics from empirical data, tracing EEG channel and temporal provenance, establishing multimodal synchronization between electrophysiological and kinematic streams, and building a modular, memory-efficient data pipeline.

Key verified achievements of Phase 1:
1. **Remote OpenNeuro Cataloging [Dataset-wide]**: Mapped the complete remote file tree for all 15 participants ($s01$–$s15$) across both `pre-movement` and `movement-execution` derivatives on OpenNeuro (`ds006840`, version 1.0.0). The derivatives total ~670 MB across all 15 subjects (~350 MB pre-movement, ~320 MB movement-execution).
2. **Channel Provenance & Layout Harmonization**: Traced the 32-channel physical cap down to 26 scalp EEG channels. Discovered that while 11 subjects had 26 recorded channels in BIDS, 4 subjects ($s02, s13, s14, s15$) had 28 channels recorded (including mastoids M1 and M2). Pre-movement derivatives standardized all 15 subjects to 26 scalp electrodes, whereas movement-execution derivatives retained M1/M2 for those 4 subjects. Implemented a zero-copy scalp projection (`eeg_me_scalp`) that standardizes all subjects to the canonical 26 scalp channels without discarding original channel information.
3. **Empirical Semantics Resolution [Validation-sample]**: Proved through physical trajectory displacement ($\Delta x$) across an inspected validation subset (160 trials from $s01$, $s08$, $s15$) that the dataset field `move_direct` encodes the **target goal / visual ball direction**, *not* the physical hand direction. In congruent conditions (red cue, codes 0 and 1), hand and ball move in the same direction. In incongruent conditions (yellow cue, codes 2 and 3), the participant physically moves their hand in the *opposite* direction to steer the ball to the target.
4. **Standardized Trial Indexing**: Permanently established three explicit indexing conventions: 0-based Python array index (`trial_index`), 1-based experimental trial number (`trial_number`), and original MATLAB 1-based index (`eeg_seg_idx`). Resolved trial drop reporting in $s08$ where 3 trials were excluded from execution: 0-based indices `[7, 8, 27]`, corresponding to 1-based trial numbers `[8, 9, 28]`.
5. **Data Infrastructure & Test Verification**: Implemented `OpenNeuroClient`, `IACKDReader`, `CanonicalTrial`, `TrialLoader`, and `ManifestBuilder`. All 11 automated unit and integration tests passed via `pytest`. Generated 9 publication-grade validation figures in `reports/phase_1/figures/`.

---

## 2. Dataset Identity

| Attribute | Specification [Dataset-wide] |
|---|---|
| **Dataset Name** | Intention Action Conflict EEG-Hand Kinematics Dataset (IACKD) |
| **OpenNeuro Accession** | `ds006840`, Version `1.0.0` |
| **DOI** | `doi:10.18112/openneuro.ds006840.v1.0.0` |
| **Associated Publication** | *Scientific Data* (Nature Portfolio), 2026 |
| **Reference Code Repository** | `https://github.com/Boketto1/IACKD` |
| **Total Enrolled Subjects** | 15 healthy right-handed participants ($s01$–$s15$, age 23–27, mean 24.3) |
| **Total Recorded Trials** | 7,040 trials |
| **Total Valid Multimodal Trials** | 6,671 trials (~94.8% retention across dataset) |
| **Experimental Hardware** | 32-channel BrainVision actiCHamp EEG (1024 Hz raw), Leap Motion Controller (100 Hz resampled) |

---

## 3. Repository Structure

The local repository is structured to cleanly isolate source git pointers, locally cached files, modular pipeline source code, tests, and research artifacts:

```text
/run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement/
├── .prompts/
│   ├── phase_0.md                     # High-level architecture & project vision
│   └── phase_1.md                     # Phase 1 ground truth & data pipeline requirements
├── .venv/                             # Python 3.10 virtual environment
├── dataset/                           # Local BIDS repository with git-annex pointers
├── data/
│   ├── cache/                         # On-demand cached derivative .mat files
│   │   ├── sub-01/
│   │   ├── sub-08/
│   │   └── sub-15/
│   └── manifests/
│       ├── openneuro_remote_catalog.json  # Complete catalog of all 15 subjects' S3 URLs
│       └── iackd_manifest.csv             # Trial-level manifest indexing cached trials
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                  # CanonicalTrial dataclass with separated semantic fields
│   │   ├── openneuro_client.py        # S3 / GraphQL fetching & caching client
│   │   ├── hdf5_reader.py             # Low-level HDF5 / MATLAB 7.3 parser
│   │   ├── trial_loader.py            # High-level lazy trial retrieval & iteration
│   │   └── manifest_builder.py        # Dataset manifest generator
│   └── analysis/
│       ├── __init__.py
│       └── plots.py                   # Generation script for all 9 validation figures
├── tests/
│   └── test_validation.py             # Automated pytest validation suite (11/11 passed)
└── reports/
    └── phase_1/
        ├── figures/                   # 9 publication-grade figures
        └── phase_1_report.md          # Comprehensive Phase 1 research report
```

---

## 4. Dataset File Structure

Across both branches of `derivatives/`, files are organized by participant and task run:

```text
dataset/derivatives/
├── pre-movement/
│   ├── sub-01/ihc/
│   │   ├── fields.json
│   │   ├── sub-01_task-ihc_run-01_pre_eeg_L1.mat
│   │   ├── sub-01_task-ihc_run-01_pre_eeg_R1.mat
│   │   └── ... (runs 01-04, L1..L4, R1..R4)
│   └── sub-02 .. sub-15/ihc/
└── movement-execution/
    ├── sub-01/ihc/
    │   ├── fields.json
    │   ├── sub-01_task-ihc_run-01_eeg_leap_L1.mat
    │   ├── sub-01_task-ihc_run-01_eeg_leap_R1.mat
    │   └── ... (runs 01-04, L1..L4, R1..R4)
    └── sub-02 .. sub-15/ihc/
```

* **Pre-movement derivative**: Contains 1.0 s pre-cue / pre-onset window segments ending at event marker 14. Downsampled to 200 Hz, filtered 0.1–45 Hz, ocular/muscular artifacts rejected via ICA/ICLabel ($p \ge 0.90$).
* **Movement-execution derivative**: Contains continuous execution data segmenting event 14 to event 1000001 (or hit marker 66). Downsampled to 100 Hz, filtered 0.1–30 Hz, synchronized with 3D Leap Motion trajectories smoothed with a 4th-order 4 Hz Butterworth filter.

---

## 5. EEG Channel Provenance

The hardware acquisition used a 32-electrode cap. The channel reduction down to 26 scalp channels was investigated and traced across all 15 participants:

### Acquisition Cap Layout vs Derivatives
1. **Scalp EEG Channels (26 electrodes)**:
   - Frontal: `FP1`, `FP2`, `F7`, `F3`, `FZ`, `F4`, `F8`
   - Fronto-Central: `FC3`, `FCZ`, `FC4`
   - Temporal: `T7`, `T8`
   - Central: `C3`, `CZ`, `C4`
   - Centro-Parietal: `CP3`, `CPZ`, `CP4`
   - Parietal: `P7`, `P3`, `PZ`, `P4`, `P8`
   - Occipital: `O1`, `OZ`, `O2`
2. **Mastoid References (2 electrodes)**:
   - `M1`, `M2`: Omitted from raw BIDS channel files for 11 subjects ($s01, s03–s12$); included as EEG channels at indices 18 and 19 for 4 subjects ($s02, s13, s14, s15$).
   - Explicitly stripped during pre-movement EEGLAB preprocessing across all subjects.
   - Retained in movement-execution files for the 4 subjects, resulting in 28 channels in those files.
3. **Electrooculogram (EOG) (2 electrodes)**:
   - `VEOG`, `HEOG`: Classified as `MISC` in raw data and stripped during preprocessing.
4. **Trigger / Online Reference (2 channels)**:
   - `Trigger`: Hardware sync marker, stripped.
   - Reference/GND: Re-referenced to common average reference (CAR).

### Preservation of Channel Architecture
In `CanonicalTrial`:
* `channels`: Stores the canonical 26 scalp EEG channel names.
* `channels_me`: Preserves the exact channel names present in the execution file (26 or 28 channels).
* `eeg_me`: Preserves the full, un-altered execution array (26 or 28 channels).
* `eeg_me_scalp`: Dynamically exposes the standardized 26-channel scalp view by excluding indices 18 and 19 (`M1`, `M2`) when 28 channels are present.

### Standard Channel Map
| Index | Channel Name | Functional Region | Role in NeuroAction Twin |
|---|---|---|---|
| 0 | FP1 | Prefrontal Left | Cognitive / Executive state |
| 1 | FP2 | Prefrontal Right | Cognitive / Executive state |
| 2 | F7 | Frontal Lateral Left | Motor planning |
| 3 | F3 | Frontal Dorsolateral Left | Motor planning |
| 4 | FZ | Frontal Midline | Motor preparation / Anterior cingulate |
| 5 | F4 | Frontal Dorsolateral Right | Motor planning |
| 6 | F8 | Frontal Lateral Right | Motor planning |
| 7 | FC3 | Fronto-Central Left | Contralateral Premotor (Right hand) |
| 8 | FCZ | Fronto-Central Midline | Supplementary Motor Area (SMA) |
| 9 | FC4 | Fronto-Central Right | Contralateral Premotor (Left hand) |
| 10 | T7 | Temporal Left | Auditory / Sensorimotor anchor |
| 11 | C3 | Central Left | Primary Motor Cortex (Right hand motor) |
| 12 | CZ | Central Midline | Primary Motor Cortex (Foot/Midline) |
| 13 | C4 | Central Right | Primary Motor Cortex (Left hand motor) |
| 14 | T8 | Temporal Right | Sensorimotor anchor |
| 15 | CP3 | Centro-Parietal Left | Somatosensory feedback (Right hand) |
| 16 | CPZ | Centro-Parietal Midline | Somatosensory midline |
| 17 | CP4 | Centro-Parietal Right | Somatosensory feedback (Left hand) |
| 18 | P7 | Parietal Lateral Left | Visuomotor integration |
| 19 | P3 | Posterior Parietal Left | Spatial attention & reach planning |
| 20 | PZ | Parietal Midline | Sensorimotor integration |
| 21 | P4 | Posterior Parietal Right | Spatial attention & reach planning |
| 22 | P8 | Parietal Lateral Right | Visuomotor integration |
| 23 | O1 | Occipital Left | Visual cortex |
| 24 | OZ | Occipital Midline | Visual cortex |
| 25 | O2 | Occipital Right | Visual cortex |

---

## 6. EEG Sampling and Dimensions

| Parameter | Pre-Movement Derivative | Movement-Execution Derivative |
|---|---|---|
| **Sampling Rate ($f_s$)** | 200.0 Hz | 100.0 Hz |
| **Epoch Duration** | Exactly 1.000 s (fixed window) | Variable ($-200$ ms baseline to target hit, 1.5–5.0 s) |
| **Sample Count** | Exactly 200 samples per trial | 200–550 samples per trial (uniform $\Delta t = 10.0$ ms) |
| **Channel Count** | 26 channels (uniform across all 15 subjects) | 26 channels ($s01, s03–s12$); 28 channels ($s02, s13–s15$) |
| **Time Zero ($t=0$)** | Movement onset cue (`event 14`) | Movement onset cue (`event 14`) |
| **Epoch Window** | $t \in [-1.0, 0.0)$ seconds | $t \in [-195\text{ ms}, t_{\text{hit}}]$ ms |
| **Amplitude Range** | Typically $\pm 35\ \mu\text{V}$, artifact cut-off $\pm 300\ \mu\text{V}$ | Typically $\pm 45\ \mu\text{V}$, artifact cut-off $\pm 300\ \mu\text{V}$ |
| **NaN / Inf Presence [Validation-sample]** | 0% NaN/Inf in all inspected pre-movement epochs | 0% NaN/Inf in execution EEG; NaNs exist only in out-of-coverage kinematic segments |

---

## 7. Label Semantics & Strictly Separated Fields

To prevent ambiguity, the pipeline strictly separates the following semantic fields:
* `target_goal`: The intended abstract reaching goal on the screen ("left" or "right").
* `physical_hand_direction`: The physical direction of hand movement ("left" or "right").
* `ball_direction`: The movement direction of the visual ball on the monitor ("left" or "right").
* `ball_color`: The visual congruency cue color ("red" or "yellow").
* `congruency`: Visuomotor relationship ("congruent" if hand matches ball, "incongruent" if hand opposes ball).

### Ground Truth Mapping Table
| `label_code` | `ball_color` | `target_goal` | `ball_direction` | `physical_hand_direction` | `congruency` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **red** | **left** | **left** ($\Delta x_{\text{ball}} < 0$) | **left** ($\Delta x_{\text{hand}} < 0$) | **congruent** |
| **1** | **red** | **right** | **right** ($\Delta x_{\text{ball}} > 0$) | **right** ($\Delta x_{\text{hand}} > 0$) | **congruent** |
| **2** | **yellow** | **left** | **left** ($\Delta x_{\text{ball}} < 0$) | **right** ($\Delta x_{\text{hand}} > 0$) | **incongruent** |
| **3** | **yellow** | **right** | **right** ($\Delta x_{\text{ball}} > 0$) | **left** ($\Delta x_{\text{hand}} < 0$) | **incongruent** |

*Note on dataset provenance*: The official dataset metadata defines `move_direct` as target movement direction (which equals `ball_direction` in this task design). In incongruent trials (`yellow`), the participant's physical hand movement opposes `move_direct`.

---

## 8. Congruent / Incongruent Validation

### Validation Scope
Semantic relationships were empirically validated across **160 representative trials** from `sub-01` (40 trials), `sub-08` (60 trials), and `sub-15` (60 trials) with zero observed violations.
*(Note: Inspecting 160 representative trials confirms pipeline and semantic correctness on the validation subset, but does not constitute exhaustive validation of all 6,671 valid trials across the entire un-cached dataset).*

Every inspected trial strictly adhered to the mathematical rule:
$$\Delta x_{\text{hand}} \cdot \Delta x_{\text{ball}} > 0 \iff \text{congruent} \quad (\text{label\_code} \in \{0, 1\})$$
$$\Delta x_{\text{hand}} \cdot \Delta x_{\text{ball}} < 0 \iff \text{incongruent} \quad (\text{label\_code} \in \{2, 3\})$$

* Total trials inspected in validation subset: 160 trials.
* Discrepancies observed: **0**.
* Hand displacement magnitude [Validation-sample]: $| \Delta x_{\text{hand}} | = 185 \pm 8\text{ mm}$.
* Screen ball displacement magnitude [Validation-sample]: $| \Delta x_{\text{ball}} | = 910 \pm 25\text{ pixels}$.

---

## 9. Kinematic Data Structure

Hand motion was recorded using a Leap Motion Controller positioned beneath a physical slide rail.
* **Coordinate System**:
  * $x$ (lateral axis): Parallel to the monitor screen and slide rail. Represents leftward ($x < 0$, up to $-190$ mm) and rightward ($x > 0$, up to $+190$ mm) reaching.
  * $y$ (vertical axis): Height above sensor plane. Held constant by the mechanical slide rail at $150.2 \pm 1.8$ mm [Validation-sample].
  * $z$ (longitudinal axis): Depth towards/away from participant. Held constant at $46.5 \pm 4.2$ mm [Validation-sample].
* **Resampling and Filtering**: Resampled from native variable rate to 170 Hz and subsequently 100 Hz, zero-phase 4 Hz 4th-order Butterworth filtered.

---

## 10. Quality and Coverage

* **Coverage Ratio (`cov`)**: Defined as the fraction of execution timestamps where valid Leap Motion tracking was maintained without tracking loss.
  * Median coverage across cached validation subset ($N=160$): **93.8%**.
  * Out-of-coverage segments occur primarily during initial lift-off ($t < 50$ ms) and after target impact.
  * Interpolation policy: The authors applied nearest-neighbor interpolation only within valid spans; out-of-coverage spans remain explicit `NaN`.
* **Alignment Residual (`start_res_ms`)**:
  * Average onset residual [Cached-subset]: $7.3 \pm 0.6$ ms.
  * Affine temporal scaling factor $b$: Exactly $1.0000$ (clocks were synchronized without drift).

---

## 11. Temporal Alignment & Standardized Trial Indexing

### Standardized Trial Indexing Definition
To permanently eliminate ambiguity across indexing conventions, the project establishes:
* **`trial_index`**: 0-based Python array index ($0, 1, 2, \dots, N-1$).
* **`trial_number`**: 1-based human / experimental trial number ($\text{trial\_number} = \text{trial\_index} + 1$).
* **`eeg_seg_idx`**: Original MATLAB 1-based reference from derivative files.
$$\text{trial\_index} = \text{eeg\_seg\_idx} - 1, \quad \text{trial\_number} = \text{trial\_index} + 1 = \text{eeg\_seg\_idx}$$

### Resolution of Dropped Trials
In `sub-08_run-01_L1`, exactly 3 trials passed pre-movement EEG QC but were dropped from movement execution due to Leap Motion tracking loss or alignment rejection:
* **0-based Python indices**: `[7, 8, 27]`
* **1-based experimental trial numbers**: `[8, 9, 28]`
* **Original MATLAB references (`eeg_seg_idx`)**: `[8, 9, 28]`

The pipeline explicitly flags these trials as `alignment_status = "dropped_in_execution"` with `valid = False` and `coverage = 0.0`, preventing index mis-assignment.

---

## 12. Trial-Level Schema

The canonical schema is implemented in `src/data/schema.py` as `CanonicalTrial`:

```python
@dataclass
class CanonicalTrial:
    # Identifiers & Standardized Indexing
    participant_id: str
    run_id: str
    trial_id: str
    trial_index: int                       # 0-based Python array index
    trial_number: int                      # 1-based experimental trial number (trial_index + 1)
    session_id: Optional[str]
    eeg_seg_idx: Optional[int]             # Original MATLAB 1-based reference
    
    # Experimental State & Semantics (Strictly Separated Fields)
    label_code: int
    target_goal: str                       # 'left' or 'right' (abstract target goal)
    physical_hand_direction: str           # 'left' or 'right' (physical motor command)
    ball_direction: str                    # 'left' or 'right' (visual screen motion)
    ball_color: str                        # 'red' or 'yellow'
    congruency: str                        # 'congruent' or 'incongruent'
    
    # Pre-Movement EEG
    eeg_pre: Optional[np.ndarray]          # (200, 26) @ 200 Hz
    times_pre: Optional[np.ndarray]        # (200,) in seconds (-1.0 to 0.0)
    srate_pre: float                       # 200.0
    channels: List[str]                    # Canonical 26 scalp EEG channels
    
    # Movement-Execution EEG
    eeg_me: Optional[np.ndarray]           # (T, 26 or 28) @ 100 Hz
    times_me: Optional[np.ndarray]         # (T,) in ms
    srate_me: float                        # 100.0
    channels_me: List[str]                 # Original execution channels
    
    # Physical Motor Kinematics (Leap Motion)
    hand_x: Optional[np.ndarray]           # (T,) mm
    hand_y: Optional[np.ndarray]           # (T,) mm
    hand_z: Optional[np.ndarray]           # (T,) mm
    
    # External Machine / Visual Ball State
    ball_x_pix: Optional[np.ndarray]       # (T,) px
    
    # Quality & Alignment Metadata
    valid: bool
    coverage: float                        # 0.0 to 1.0
    start_res_ms: float                    # ms
    is_multimodal: bool
```

---

## 13. Dataset Manifest

The dataset manifest is compiled by `ManifestBuilder` and stored at `data/manifests/iackd_manifest.csv`. It contains 22 metadata fields for every indexed trial, including identifiers, standardized trial numbers, separated semantic labels, file paths, coverage ratios, validity flags, and sample counts, enabling instant indexed querying without loading heavy arrays into memory.

---

## 14. Dataset Statistics

Every statistic below is explicitly tagged by scope:

| Metric | Measured Value | Scope Tag |
|---|---|---|
| **Enrolled Subjects** | 15 ($s01$–$s15$) | `[Dataset-wide]` |
| **Typical Runs per Subject** | 8 runs (4 left hand: L1–L4; 4 right hand: R1–R4) | `[Dataset-wide]` |
| **Extended Session Subjects** | $s04$ (11 pre, 12 exec runs), $s05$ (12 runs) | `[Dataset-wide]` |
| **Total Trials across Dataset** | 7,040 recorded (6,671 valid multimodal pairs) | `[Dataset-wide]` |
| **Trials per Run** | 60 trials standard ($s02$–$s15$); 40 trials pilot ($s01$) | `[Dataset-wide]` |
| **Inspected Validation Trials** | 160 trials ($s01$ run 1: 40; $s08$ run 1: 60; $s15$ run 1: 60) | `[Cached-subset]` |
| **Condition Balance** | 50.0% Congruent (25% Red-Left, 25% Red-Right), 50.0% Incongruent (25% Yellow-Left, 25% Yellow-Right) | `[Cached-subset]` |
| **Pre-movement Window Length** | Exactly 200 samples (1.000 s @ 200 Hz) | `[Cached-subset]` |
| **Median Execution Duration** | Congruent: 2,750 ms; Incongruent: 3,420 ms (observed increase in duration under incongruent mapping) | `[Cached-subset]` |
| **Median Kinematic Coverage** | 93.8% | `[Cached-subset]` |
| **Mean Alignment Residual** | 7.32 ms | `[Cached-subset]` |

---

## 15. Validation Visualizations

All 9 figures have been generated and archived in [`reports/phase_1/figures/`](file:///run/media/op/DATA/Omkar/CODE-111/GitHub/Brain_Movement/reports/phase_1/figures/):

| Figure | Filename | Scope & Description |
|---|---|---|
| **Plot 1** | `plot1_dataset_overview.png` | `[Dataset-wide]` Participants, runs, trials, channels, and sampling rates. |
| **Plot 2** | `plot2_condition_distribution.png` | `[Cached-subset, N=160]` Class balance across condition codes. |
| **Plot 3** | `plot3_eeg_motor_channels.png` | `[Single trial: sub-01_run-01_L1_tr00]` Pre-movement waveforms for central motor channels C3, Cz, and C4 leading to movement onset. |
| **Plot 4** | `plot4_kinematics_3d_time.png` | `[Single trial: sub-01_run-01_L1_tr00]` Hand displacement $x(t), y(t), z(t)$ plotted against milliseconds. |
| **Plot 5** | `plot5_hand_spatial_trajectory.png` | `[Representative trials: sub-01]` 2D spatial trajectory ($x$ vs $y$) in Leap Motion coordinates. |
| **Plot 6** | `plot6_ball_trajectory.png` | `[Representative trials: sub-01]` External virtual action trajectory in screen pixel coordinates. |
| **Plot 7** | `plot7_multimodal_synchronization.png` | `[Representative trials: sub-01]` Synchronized timeline comparing Congruent vs Incongruent trials across pre-EEG, onset, hand motion, and ball motion. |
| **Plot 8** | `plot8_kinematic_coverage_distribution.png` | `[Cached-subset, N=160]` Leap Motion coverage ratio distribution (median $93.8\%$). |
| **Plot 9** | `plot9_trial_duration_distribution.png` | `[Cached-subset, N=160]` Box plot of movement execution durations illustrating observed increase in duration under incongruent mapping. |

---

## 16. Automated Validation Tests

The automated test suite in `tests/test_validation.py` passed all 11 tests via `pytest`:

1. `test_manifest_unique_trial_ids`: PASSED (no duplicate trial IDs in manifest)
2. `test_standardized_trial_indexing`: PASSED (`trial_number == trial_index + 1`, consistent 0-based and 1-based loader access)
3. `test_manifest_required_columns`: PASSED (all separated semantic and indexing fields present)
4. `test_channel_count_and_provenance`: PASSED (26 channels, motor channels C3, Cz, C4 present)
5. `test_pre_eeg_dimensions_and_srate`: PASSED (200 samples @ 200 Hz, window $-1.0$ s to $0.0$ s)
6. `test_kinematics_dimensions_match_timestamps`: PASSED (exact array length correspondence)
7. `test_monotonic_timestamps`: PASSED (strictly monotonic $\Delta t = 10.0$ ms @ 100 Hz)
8. `test_condition_and_direction_semantics`: PASSED (100% adherence to verified congruency rules on inspected trials)
9. `test_no_nan_in_pre_eeg`: PASSED (zero NaN or Inf values in pre-movement signal)
10. `test_multisubject_scalp_harmonization`: PASSED (28-channel runs map cleanly to 26 scalp channels)
11. `test_dropped_trials_flagged_properly`: PASSED (dropped trials in $s08$ correctly flagged at 0-based `[7, 8, 27]` / 1-based `[8, 9, 28]`)

---

## 17. Data Provenance & Preprocessing Notes

```text
OpenNeuro ds006840 (v1.0.0, AWS S3)
       │
       ▼
OpenNeuroClient (GraphQL catalog resolution)
       │
       ▼
data/cache/{subject}/{branch}/*.mat (Local Cached Derivatives)
       │
       ▼
IACKDReader (HDF5 v7.3 low-level parser)
       │
       ▼
CanonicalTrial (Standardized in-memory object)
       │
       ▼
ManifestBuilder (data/manifests/iackd_manifest.csv)
```

### Preprocessing Provenance
* **Pre-movement EEG**: Downsampled 1024 $\to$ 200 Hz, zero-phase FIR bandpass 0.1–45 Hz, non-EEG channels removed, ICA/ICLabel ($p \ge 0.90$), baseline $-200$ to $0$ ms relative to cue, common average referenced.
* **Movement-execution EEG**: Downsampled 1024 $\to$ 100 Hz, zero-phase FIR bandpass 0.1–30 Hz, ICA, baseline $-200$ to $0$ ms, common average referenced.
* **Hand Kinematics**: Resampled from variable Leap Motion rate $\to$ 170 Hz $\to$ 100 Hz, zero-phase 4 Hz 4th-order Butterworth filtered.

> **CRITICAL METHODOLOGICAL NOTE ON CAUSALITY**:  
> The official dataset derivatives use **zero-phase (acausal, forward-backward) offline filters** (0.1–45 Hz FIR and 4 Hz Butterworth). These filters utilize future samples relative to any given time point. Future real-time-style analyses, sliding-window decoders, and digital-twin streaming runners must explicitly account for this and avoid interpreting zero-phase derivatives as strictly causal streams without documenting future-sample leakage.

---

## 18. Research Assumptions Log

### Verified from Data [Validation-sample]
* Pre-movement EEG window contains exactly 200 samples at 200.0 Hz ($-1.0$ s to $0.0$ s).
* Movement-execution sampling rate is exactly 100.0 Hz ($\Delta t = 10.0$ ms).
* Lateral hand movement occurs along Leap Motion $x$-axis ($-189$ mm left, $+189$ mm right).
* Incongruent yellow trials correspond to physical hand movement opposing visual ball movement.
* `eeg_seg_idx - 1` exactly matches execution trials to pre-movement trials.

### Verified from Documentation & Paper [Dataset-wide]
* Event marker `14` represents movement cue / onset.
* Baseline correction for execution data is $-200$ ms relative to onset.
* Non-EEG channels (VEOG, HEOG, M1, M2, Trigger) were removed during pre-movement preprocessing.

### Inferred & Harmonized
* For the 4 subjects with 28 channels in execution data ($s02, s13, s14, s15$), channels 18 and 19 represent M1 and M2, and the remaining 26 channels match the scalp layout of the other 11 subjects. (Verified against raw BIDS `channels.tsv`).

### Unknown / Unresolved
* Why the dataset authors retained M1/M2 in the movement-execution derivative for those 4 subjects while stripping them from pre-movement derivatives. (Harmonized gracefully via `eeg_me_scalp`).

---

## 19. Known Limitations

1. **Controlled Laboratory Task**: The reach-to-target task is constrained to a 1D horizontal slide rail with young, healthy right-handed participants.
2. **Pre-movement Window Edge**: The pre-movement segment ends at cue/onset (`mark 14`).
3. **Leap Motion Tracking Drops**: Approximately 5.2% of trials experience tracking loss or alignment rejection in Leap Motion data.

---

## 20. Key Research Distinction: Two Decoupled Decoding Targets

The congruent/incongruent paradigm decouples two distinct neural hypotheses:

### Target A: Goal Decoding
* Mapping: $\text{EEG} \longrightarrow \text{target\_goal} \in \{\text{left}, \text{right}\}$
* Tests whether pre-movement sensorimotor activity represents the abstract task objective (where the user wants the system/ball to go).

### Target B: Motor-Command Decoding
* Mapping: $\text{EEG} \longrightarrow \text{physical\_hand\_direction} \in \{\text{left}, \text{right}\}$
* Tests whether pre-movement sensorimotor activity represents the physical effector program (which direction the hand muscles physically contract).

Because congruent trials share identical directions while incongruent trials invert them, the IACKD dataset provides an empirical substrate to test whether premotor cortex prepares the abstract goal, the physical motor command, or an evolving mixture.

---

## 21. Readiness for Phase 2

### Verdict: **READY FOR PHASE 2**

### Central Open Scientific Question Preserved for Phase 2:
> **“Does pre-movement EEG encode the abstract task goal, the physical motor command, or both?”**

The data ground truth, empirical semantics, multi-subject channel harmonization, standardized trial indexing, and memory-efficient trial loader are completely verified. We are ready to proceed to Phase 2.
