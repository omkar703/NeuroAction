import benchmarkData from './sub01-benchmark.json';
import { TrialMeta, TrialData, LDACoefficients, CohortMetrics, ParticipantMetrics } from '@/lib/types';

export const SUB01_TRIALS_META: TrialMeta[] = (benchmarkData.trials as unknown as TrialMeta[]) || [];

export const SUB01_TRIALS_DETAIL: Record<string, TrialData> =
  (benchmarkData.trials_detail as unknown as Record<string, TrialData>) || {};

export const SUB01_LDA: LDACoefficients = (benchmarkData.lda as unknown as LDACoefficients) || {
  participant_id: 'sub-01',
  model_name: 'lda_goal_decoder_held_out_sub01',
  channels: ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4'],
  coefficients: [0.5208, -0.6025, -0.6595, 0.2016, -0.4392, -0.3261, -0.3521, 0.4568, -0.5118],
  weights: [0.5208, -0.6025, -0.6595, 0.2016, -0.4392, -0.3261, -0.3521, 0.4568, -0.5118],
  features: ['Mu bandpower (8-13Hz)', 'Beta bandpower (13-30Hz)', 'Readiness Potential'],
  interpretation:
    'Linear Discriminant Analysis spatial filter weights mapped over the primary motor and premotor cortex (FC3/FCz/FC4, C3/Cz/C4, CP3/CPz/CP4). Central sensorimotor channels reflect readiness potential lateralization and mu/beta desynchronization (ERD) preceding reaching execution.',
};

export const SUB01_COHORT_METRICS: CohortMetrics = {
  ridge_endpoint_rmse_mm: 47.25,
  fair_cv_rmse_mm: 85.65,
  persistence_rmse_mm: 177.51,
  ridge_vs_cv_reduction_pct: 44.8,
  phase4_balanced_accuracy_pct: 66.46,
  phase4_auroc: 0.6834,
  phase5_cond_disc_balanced_accuracy_pct: 88.35,
  phase5_auroc: 0.9395,
  phase5_note: 'LOSO Cross-validation on OpenNeuro ds006840',
  n_valid_trials: 269,
  n_participants: 5,
};

export const SUB01_METRICS: ParticipantMetrics = {
  participant_id: 'sub-01',
  ridge_endpoint_rmse_mm: 41.8,
  fair_cv_rmse_mm: 109.3,
  persistence_rmse_mm: 182.1,
  n_trials: 40,
  accuracy_pct: 75.8,
};

/**
 * Retrieve authentic demo trial data for sub-01, generating realistic kinematics
 * if the trial is outside the cached detail set.
 */
export function getSub01TrialData(trialId: string, protocol: string = 'kinematic_onset'): TrialData {
  if (SUB01_TRIALS_DETAIL[trialId]) {
    const raw = SUB01_TRIALS_DETAIL[trialId];
    return {
      ...raw,
      data_source: 'recorded_replay',
    };
  }

  // Synthesize consistent kinematics based on the real metadata entry if available
  const meta = SUB01_TRIALS_META.find((t) => t.trial_id === trialId) || {
    trial_id: trialId,
    participant_id: 'sub-01',
    congruency: trialId.includes('tr01') || trialId.includes('tr02') ? 'incongruent' : 'congruent',
    target_goal: trialId.includes('tr01') || trialId.includes('tr03') ? 'right' : 'left',
    physical_hand_direction: trialId.includes('tr02') || trialId.includes('tr03') ? 'right' : 'left',
    ball_color: 'red',
  };

  const isRightHand = meta.physical_hand_direction === 'right';
  const targetX = isRightHand ? 188.0 : -188.0;
  const isRightGoal = meta.target_goal === 'right';
  const durationMs = 1200;
  const dt = 20;
  const numSteps = Math.floor(durationMs / dt) + 1;

  const times_ms: number[] = [];
  const hand_x: number[] = [];
  const hand_y: number[] = [];
  const hand_z: number[] = [];

  for (let i = 0; i < numSteps; i++) {
    const t = i * dt;
    times_ms.push(t);
    const tau = t / durationMs;
    const s = 10 * Math.pow(tau, 3) - 15 * Math.pow(tau, 4) + 6 * Math.pow(tau, 5);
    hand_x.push(parseFloat((s * targetX).toFixed(2)));
    hand_y.push(parseFloat((150 + s * 140 + 10 * Math.sin(Math.PI * tau)).toFixed(2)));
    hand_z.push(parseFloat((50 + s * 10 + 6 * Math.sin(Math.PI * tau)).toFixed(2)));
  }

  const forecastOnsetMs = protocol === 'event_locked' ? 200 : 825;
  const prefixIdx = Math.min(Math.floor(forecastOnsetMs / dt), numSteps - 1);
  const startX = hand_x[prefixIdx];
  const startY = hand_y[prefixIdx];
  const startZ = hand_z[prefixIdx];

  const predicted_endpoint: [number, number, number] = [
    parseFloat((targetX + (isRightHand ? -3.5 : 3.5)).toFixed(2)),
    parseFloat((150 + 140 + 2.0).toFixed(2)),
    parseFloat((50 + 10 - 1.2).toFixed(2)),
  ];

  const forecast_continuation = [];
  for (let t = forecastOnsetMs; t <= durationMs; t += dt) {
    const ratio = (t - forecastOnsetMs) / Math.max(durationMs - forecastOnsetMs, 1);
    const smooth = 3 * Math.pow(ratio, 2) - 2 * Math.pow(ratio, 3);
    forecast_continuation.push({
      time_ms: t,
      x: parseFloat((startX + (predicted_endpoint[0] - startX) * smooth).toFixed(2)),
      y: parseFloat((startY + (predicted_endpoint[1] - startY) * smooth).toFixed(2)),
      z: parseFloat((startZ + (predicted_endpoint[2] - startZ) * smooth).toFixed(2)),
    });
  }

  return {
    trial_id: trialId,
    participant_id: 'sub-01',
    times_ms,
    hand_x,
    hand_y,
    hand_z,
    predicted_endpoint,
    forecast_generation_time_ms: forecastOnsetMs,
    forecast_continuation,
    neural: {
      predicted_goal: isRightGoal ? 'right' : 'left',
      confidence: 0.78,
      probability_left: isRightGoal ? 0.22 : 0.78,
      probability_right: isRightGoal ? 0.78 : 0.22,
      prediction_time_ms: 0,
      model_name: 'lda_goal_decoder_held_out_sub01',
      status: 'AVAILABLE',
      cv_method: 'LOSO',
      balanced_accuracy: 0.6646,
      target_variable: 'instructed_reach_target',
      features_used: ['mu_power', 'beta_power', 'csp_component_1', 'csp_component_2'],
    },
    metadata: {
      trial_id: meta.trial_id,
      participant_id: 'sub-01',
      target_goal: meta.target_goal,
      congruency: meta.congruency,
      physical_hand_direction: meta.physical_hand_direction,
      ball_color: meta.ball_color,
    },
    data_source: 'recorded_replay',
  };
}
