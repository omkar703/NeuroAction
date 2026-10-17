# NeuroAction — Research Figures Catalog

This directory contains publication-ready figures generated across the scientific investigations of the **NeuroAction** project.

---

## Figure Catalog by Investigation Stage

### 1. Exploratory Data Analysis & Multimodal Alignment
- **`plot1_dataset_overview.png`**: Breakdown of OpenNeuro `ds006840` sessions, runs, and trials across participants.
- **`plot2_condition_distribution.png`**: Distribution of trial conditions: Congruent (60%), Incongruent (20%), and Free Choice (20%).
- **`plot3_eeg_motor_channels.png`**: Raw 26-channel scalp EEG waveforms centered over primary motor cortex ($C3, CZ, C4$).
- **`plot4_kinematics_3d_time.png`**: Leap Motion Cartesian position timeseries ($X, Y, Z$) showing onset, acceleration, and deceleration.
- **`plot5_hand_spatial_trajectory.png`**: 3D spatial reach trajectories of the hand moving from origin to left/right targets.
- **`plot6_ball_trajectory.png`**: Kinematic trajectory of the visual stimulus object across trials.
- **`plot7_multimodal_synchronization.png`**: Synchronization alignment between 200 Hz EEG sampling and 50 Hz optical kinematic tracking.
- **`plot8_kinematic_coverage_distribution.png`**: Spatial coverage and boundary envelope of Leap Motion optical workspace.
- **`plot9_trial_duration_distribution.png`**: Histogram of trial durations from movement onset to target attainment (median $\sim 1200$ ms).

### 2. Neural Intention Decoding & Goal Dissociation
- **`figure_1_experimental_design.png`**: Schematic of experimental paradigm, trial timing (fixation, preparation, go-cue), and EEG/kinematic timelines.
- **`figure_2_readiness_potential.png`**: Grand-average Readiness Potential (RP / *Bereitschaftspotential*) showing negative slow cortical drift preceding movement onset across motor channels ($C3, CZ, C4$).
- **`figure_3_smr_erd.png`**: Time-frequency spectrogram displaying sensorimotor rhythm (SMR) event-related desynchronisation (ERD) in the mu (8–12 Hz) and beta (13–30 Hz) bands.
- **`figure_4_goal_temporal.png`**: Temporal evolution of neural goal decoding accuracy in sliding pre-movement windows ($[-1000, 0]$ ms).
- **`figure_5_motor_temporal.png`**: Temporal profile of motor execution decoding showing rapid separation following kinematic onset.
- **`figure_6_goal_vs_motor_comparison.png`**: Direct comparison of pre-movement goal intention vs. post-onset motor execution dynamics.
- **`figure_7_transfer_experiment.png`**: Cross-participant generalization matrix under 5-fold Leave-One-Subject-Out (LOSO) evaluation.
- **`figure_8_subject_distribution.png`**: Per-participant performance variation across the 5 benchmark subjects (`sub-01`, `sub-02`, `sub-03`, `sub-08`, `sub-15`).
- **`figure_9_neural_commitment_points.png`**: Identified neural commitment thresholds where pre-movement intention becomes irreversible.
- **`figure_10_spatial_ablation.png`**: Scalp channel ablation study isolating top motor/premotor electrodes ($FC3, CZ, C3, C4, CP3$).

### 3. Continuous 3D Kinematics & Trajectory Forecasting
- **`figure_1_coordinate_workspace.png`**: 3D Leap Motion Cartesian coordinate system ($X$: Lateral, $Y$: Vertical, $Z$: Depth) with empirical target centroids.
- **`figure_2_prefix_dynamics.png`**: Kinematic prefix dynamics comparing 100 ms, 200 ms, 300 ms, and 500 ms observation windows.
- **`figure_3_endpoint_error_comparison.png`**: Bar chart comparing 3D endpoint RMSE across Ridge Regressor (47.25 mm), Fair CV (85.65 mm), and Persistence (177.51 mm).
- **`figure_4_horizon_forecasting_error.png`**: Trajectory RMSE as a function of prefix duration, demonstrating diminishing returns beyond 200 ms.
- **`figure_5_forecast_case_study.png`**: Multi-panel case study demonstrating extrapolated trajectory cone and true hand path.
- **`demo_forecast_output.png`**: Detailed endpoint forecast visualization for canonical reach trial `sub-01_run-01_L1_tr00`.

### 4. Multimodal Digital Twin Fusion & Conflict Tracking
- **`figure_9_multimodal_architecture.png`**: Architectural schematic of the real-time causal fusion engine, temporal alignment buffer, and conflict detector.
- **`figure_10_conflict_detection_analysis.png`**: Empirical conflict detection score distributions, threshold sensitivity, and ROC curve (0.6834 AUROC).
- **`figure_11_demo_trial_replay.png`**: 6-panel snapshot of the digital twin during causal replay of trial `sub-01_run-01_R1_tr01`.
- **`figure_11_dashboard_snapshot.png`**: Full-screen view of the interactive HTML replay dashboard.
- **`figure_11_sub03_snapshot.png`**: Causal replay snapshot for participant `sub-03` demonstrating robust trajectory tracking under high trial jitter.

### 5. Independent Scientific Validation
- **`figure_12_validation_architecture.png`**: Flowchart of the independent validation protocol, separating retrospection from prospective recording.
- **`figure_13_comparative_benchmarks.png`**: Comparative benchmark summary comparing Ridge, Polynomial, Constant Velocity, and Persistence models.
- **`figure_14_prospective_protocol_timeline.png`**: Chronological protocol timeline for prospective human subject validation with physical target optotrak calibration.
