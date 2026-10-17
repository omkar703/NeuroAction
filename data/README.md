# NeuroAction — Data Directory & Dataset Provenance

This directory houses the trial manifests, preprocessed derivative cache, and external validation templates for the **NeuroAction** project.

---

## Directory Organization

```text
data/
├── README.md                                  # This documentation file
├── manifests/                                 # Canonical trial manifests
│   ├── iackd_manifest.csv                     # 280-trial audited manifest with validity flags
│   └── openneuro_remote_catalog.json          # Complete OpenNeuro ds006840 remote file manifest
├── cache/                                     # Fast-loading derivative .mat matrices
│   ├── sub-01/                                # Segmented EEG + Leap Motion kinematics (40 valid trials)
│   ├── sub-02/                                # 52 valid trials
│   ├── sub-03/                                # 60 valid trials
│   ├── sub-08/                                # 57 valid trials
│   └── sub-15/                                # 60 valid trials
└── sample_external_validation_synthetic.csv   # Prospective validation schema test data
```

---

## Dataset Provenance: OpenNeuro ds006840

The research data originates from the publicly accessible OpenNeuro repository:
- **Dataset Name**: *IACKD: Intention Action Conflict EEG-Hand Kinematics Dataset*
- **Persistent DOI**: `doi:10.18112/openneuro.ds006840.v1.0.0`
- **Citation**: Cai et al., *Scientific Data* (2024).
- **License**: Creative Commons CC0 1.0 Universal (Public Domain Dedication).
- **Format**: Brain Imaging Data Structure (BIDS) v1.7.0.

### Benchmark Cohort ($N = 5$)
While the raw BIDS dataset contains 15 participants, 5 participants have complete, uncorrupted concurrent 26-channel scalp EEG and 3D optical tracking:
- `sub-01` (40 valid reach trials)
- `sub-02` (52 valid reach trials)
- `sub-03` (60 valid reach trials)
- `sub-08` (57 valid reach trials)
- `sub-15` (60 valid reach trials)
- **Total Valid Evaluated Cohort**: **269 valid reach trials** (out of 280 total recorded trials; 11 trials excluded due to optical sensor occlusion or dropout).

---

## Raw BIDS Dataset vs. Preprocessed Cache

- **`dataset/` (Root Directory)**: Contains the raw BIDS hierarchy with BrainVision `.eeg`, `.vhdr`, `.vmrk` electrophysiology files and raw optical tracking logs.
- **`data/cache/`**: Contains pre-extracted HDF5/MATLAB `.mat` structures where 26 scalp EEG channels have been zero-phase bandpassed to $[0.5, 40]$ Hz at 200 Hz, and Leap Motion hand positions ($X, Y, Z$) have been aligned to movement onset at 50/100 Hz.

---

## Data Dictionary: iackd_manifest.csv

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `participant_id` | `str` | Subject identifier (`sub-01`, `sub-02`, etc.) |
| `run_id` | `str` | Experimental run (`run-01`, `run-02`) |
| `trial_index` | `int` | Zero-indexed trial number ($0$–$39$) |
| `trial_id` | `str` | Unique trial key (`sub-01_run-01_L1_tr00`) |
| `condition` | `str` | Task condition: `congruent` (60%), `incongruent` (20%), or `free_choice` (20%) |
| `target_direction` | `str` | Instructed physical reach direction: `left` or `right` |
| `pre_eeg_available` | `bool` | True if $[-1000, 0]$ ms pre-movement EEG is present and artifact-free |
| `kinematics_available` | `bool` | True if continuous Leap Motion tracking was logged |
| `valid` | `bool` | Master eligibility flag ($1$ if trial contains valid prefix and non-NaN endpoint) |
