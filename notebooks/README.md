# NeuroAction — Research Walkthrough Notebooks

This directory contains the sequential, publication-grade Jupyter research walkthrough notebooks for the **NeuroAction** neural intention–action digital twin. 

All notebooks are pre-executed with cell outputs, tables, and visualization plots preserved. They can also be re-executed deterministically against the local OpenNeuro `ds006840` dataset and preprocessed derivatives.

---

## Notebook Index and Research Workflow

The notebooks follow the five sequential scientific investigations of the NeuroAction project:

| Order | Notebook Filename | Research Investigation & Focus | Primary Inputs | Key Generated Outputs / Figures |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`01_data_exploration_and_alignment.ipynb`](./01_data_exploration_and_alignment.ipynb) | **Data Foundations & Multimodal Alignment**<br>BIDS dataset ingestion, 26 scalp EEG channels, Leap Motion 3D optical tracking, and temporal alignment. | `dataset/` (BIDS)<br>`data/manifests/` | 9 exploratory alignment figures (`plot1`–`plot9`), trial duration distributions, multimodal sync checks. |
| **02** | [`02_neural_goal_decoding.ipynb`](./02_neural_goal_decoding.ipynb) | **Neural Intention Decoding & Goal Dissociation**<br>Sensorimotor rhythm (SMR) mu (8–12 Hz) & beta (13–30 Hz) event-related desynchronisation (ERD), Common Spatial Patterns (CSP), and Linear Discriminant Analysis (LDA) goal classification. | `data/cache/` (.mat)<br>`models/phase_2/` | 10 publication figures (`figure_1`–`figure_10`), readiness potentials (RP), ERD time-frequency maps, LOSO out-of-fold predictions. |
| **03** | [`03_kinematic_trajectory_forecasting.ipynb`](./03_kinematic_trajectory_forecasting.ipynb) | **Continuous 3D Kinematics & Trajectory Forecasting**<br>Leap Motion coordinate frame transformation, kinematic onset discovery (vs. visual go-cue), 200 ms prefix polynomial and Ridge regression forecasting. | `data/cache/` (.mat)<br>`models/phase_3/` | 7 trajectory figures (`figure_1`–`figure_5`), 3D coordinate workspaces, horizon error curves (100–500 ms), endpoint error comparisons. |
| **04** | [`04_multimodal_digital_twin_replay.ipynb`](./04_multimodal_digital_twin_replay.ipynb) | **Multimodal Digital Twin Fusion & Conflict Tracking**<br>Deterministic causal replay engine, cross-modal congruence tracking, intention–action discordance detection, and 6-panel dashboard. | `models/phase_2/`<br>`models/phase_3/`<br>`data/cache/` | 5 system figures (`figure_9`–`figure_11`), conflict ROC analysis (0.6834 AUROC), replay timeseries, sub-03 case study. |
| **05** | [`05_independent_validation_audit.ipynb`](./05_independent_validation_audit.ipynb) | **Independent Scientific Validation & Prospective Protocol**<br>Complete 280-trial eligibility audit (269 valid), 5-fold LOSO cross-validation, baseline comparisons, prospective human recording protocol. | `data/manifests/`<br>`reports/phase_4/` | 3 audit figures (`figure_12`–`figure_14`), comparative benchmark bar charts, protocol timelines, final evaluation matrix. |

---

## Environment Setup and Launching Jupyter

### 1. Activate Environment

The project relies on Python 3.10 with PyTorch, MNE, scikit-learn, SciPy, NumPy, and Pandas.

```bash
# In the repository root:
source .venv/bin/activate
```

### 2. Launch Jupyter

You can launch Jupyter from either the repository root or directly within the `notebooks/` directory:

```bash
# Option A: From repository root (recommended)
jupyter lab

# Option B: Classic notebook server
jupyter notebook notebooks/
```

### 3. Path Resolution Architecture

Each notebook includes an automated root-detection header in its initial code cell:

```python
import os
import sys
from pathlib import Path

# Ensures repository root is on sys.path and set as current working directory
PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
```

This guarantees that whether a notebook kernel is spawned from `.` or `notebooks/`, all relative paths (`data/`, `models/`, `results/`, `reports/`) resolve deterministically without path errors.

---

## Dependencies Between Notebooks

1. **Notebook 01** can run independently on the raw BIDS dataset and manifests.
2. **Notebook 02** consumes preprocessed derivative `.mat` files in `data/cache/` (or generates them) and loads trained LDA models from `models/phase_2/`.
3. **Notebook 03** uses `data/cache/` and `models/phase_3/` to evaluate trajectory forecasting across prefixes from 100 ms to 500 ms.
4. **Notebook 04** fuses the trained neural decoder and kinematic forecaster into the multimodal digital-twin replay loop.
5. **Notebook 05** audits the combined evaluation matrices and executes the 5-fold Leave-One-Subject-Out benchmark comparisons.

---

## Computational Requirements

- **Memory**: $\ge 8$ GB RAM recommended for loading multi-channel EEG epochs and optical tracking arrays.
- **Disk**: $\sim 500$ MB for cached derivative matrices (`data/cache/`).
- **GPU**: Optional; all models (CSP, LDA, Ridge regression, Polynomial extrapolators) execute on CPU in $< 60$ seconds.
