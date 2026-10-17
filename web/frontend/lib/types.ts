export interface TrialMeta {
  trial_id: string;
  participant_id: string;
  congruency: 'congruent' | 'incongruent' | string;
  target_goal: 'left' | 'right' | string;
  physical_hand_direction: 'left' | 'right' | string;
  ball_color?: string;
  [key: string]: unknown;
}

export interface NeuralResult {
  predicted_goal: 'left' | 'right' | string;
  probability_left: number;
  probability_right: number;
  confidence: number;
  prediction_time_ms: number;
  model_name: string;
  status: string;
  [key: string]: unknown;
}

export interface ConflictFrame {
  time_ms: number;
  state: 'congruent' | 'diverging' | 'conflict' | 'resolved' | string;
  dist_mm: number;
  norm_dist: number;
  explanation: string;
  [key: string]: unknown;
}

export interface TrialData {
  trial_id: string;
  participant_id: string;
  times_ms: number[];
  hand_x: number[];
  hand_y: number[];
  hand_z: number[];
  predicted_endpoint: [number, number, number] | number[] | null;
  forecast_generation_time_ms?: number;
  horizon_path?: [number, number, number][] | number[][] | { x: number; y: number; z: number }[];
  neural?: NeuralResult;
  conflict_timeseries?: ConflictFrame[];
  metadata?: TrialMeta | Record<string, unknown>;
  congruency?: 'congruent' | 'incongruent' | string;
  target_goal?: 'left' | 'right' | string;
  physical_hand_direction?: 'left' | 'right' | string;
  ball_color?: string;
  data_source?: string;
  [key: string]: unknown;
}

export interface EEGSignal {
  channel: string;
  available_channels: string[];
  times_ms: number[];
  amplitude_uv: number[];
  processing: string;
  srate_hz: number;
  trial_id: string;
  [key: string]: unknown;
}

export interface CohortMetrics {
  ridge_endpoint_rmse_mm: number;
  fair_cv_rmse_mm: number;
  persistence_rmse_mm: number;
  ridge_vs_cv_reduction_pct: number;
  phase4_balanced_accuracy_pct: number;
  phase4_auroc: number;
  phase5_cond_disc_balanced_accuracy_pct: number;
  phase5_auroc: number;
  phase5_note: string;
  n_valid_trials: number;
  n_participants: number;
  [key: string]: unknown;
}

export interface ParticipantMetrics {
  participant_id: string;
  ridge_endpoint_rmse_mm?: number;
  fair_cv_rmse_mm?: number;
  persistence_rmse_mm?: number;
  n_trials?: number;
  accuracy_pct?: number;
  [key: string]: unknown;
}

export interface ModelInfo {
  model_name: string;
  type?: string;
  parameters?: Record<string, unknown>;
  version?: string;
  training_protocol?: string;
  features?: string[];
  [key: string]: unknown;
}

export interface LDACoefficients {
  participant_id?: string;
  model_name?: string;
  channels?: string[];
  features?: string[];
  coefficients?: number[];
  weights?: number[];
  intercept?: number;
  interpretation?: string;
  data_source?: string;
  [key: string]: unknown;
}

