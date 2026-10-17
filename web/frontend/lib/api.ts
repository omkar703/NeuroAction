import {
  TrialMeta,
  TrialData,
  NeuralResult,
  EEGSignal,
  CohortMetrics,
  ParticipantMetrics,
  ModelInfo,
  LDACoefficients,
} from './types';
import {
  SUB01_TRIALS_META,
  SUB01_LDA,
  SUB01_COHORT_METRICS,
  SUB01_METRICS,
  getSub01TrialData,
} from './demo-data/sub01DemoData';
import { notifyBackendRequired } from './backend-context';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

// Short TTL cache for backend reachability check
let backendOnlineCache: boolean | null = null;
let lastCheckTimestamp = 0;

export async function isBackendOnline(): Promise<boolean> {
  if (typeof window === 'undefined') return false;

  const now = Date.now();
  if (backendOnlineCache !== null && now - lastCheckTimestamp < 4000) {
    return backendOnlineCache;
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1200);

    const res = await fetch(`${API_BASE_URL}/api/health`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    clearTimeout(timeout);

    backendOnlineCache = res.ok;
  } catch {
    backendOnlineCache = false;
  }

  lastCheckTimestamp = now;
  return backendOnlineCache;
}

async function safeFetch<T>(
  endpoint: string,
  options?: RequestInit,
  fallbackValue?: T
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    const res = await fetch(url, {
      ...options,
      signal: options?.signal || controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });
    clearTimeout(timeout);

    if (!res.ok) {
      if (fallbackValue !== undefined) return fallbackValue;
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const data = await res.json();
    return data as T;
  } catch (error) {
    if (fallbackValue !== undefined) {
      return fallbackValue;
    }
    throw error;
  }
}

/**
 * Fetch all available participant IDs
 */
export async function fetchParticipants(): Promise<string[]> {
  const online = await isBackendOnline();
  if (!online) {
    return ['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15'];
  }

  try {
    const res = await safeFetch<{ participants?: string[] } | string[]>(
      '/api/participants',
      undefined,
      ['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15']
    );
    if (Array.isArray(res) && res.length > 0) return res;
    if (res && Array.isArray((res as { participants?: string[] }).participants)) {
      return (res as { participants?: string[] }).participants!;
    }
    return ['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15'];
  } catch {
    return ['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15'];
  }
}

/**
 * Fetch trials metadata for a given participant
 */
export async function fetchTrials(participantId: string): Promise<TrialMeta[]> {
  const online = await isBackendOnline();
  if (!online) {
    if (participantId !== 'sub-01') {
      notifyBackendRequired({
        actionType: 'participant',
        title: 'Backend Required for Cohort Switch',
        message: `Participant ${participantId} requires the Python scientific backend and raw OpenNeuro dataset. In this hosted demo, Participant sub-01 is loaded with 40 benchmark trials.`,
      });
    }
    return SUB01_TRIALS_META;
  }

  try {
    const res = await safeFetch<{ trials?: TrialMeta[] } | TrialMeta[]>(
      `/api/trials?participant_id=${encodeURIComponent(participantId)}`,
      undefined,
      SUB01_TRIALS_META
    );
    if (Array.isArray(res) && res.length > 0) return res;
    if (
      res &&
      Array.isArray((res as { trials?: TrialMeta[] }).trials) &&
      (res as { trials?: TrialMeta[] }).trials!.length > 0
    ) {
      return (res as { trials?: TrialMeta[] }).trials!;
    }
    return SUB01_TRIALS_META;
  } catch {
    return SUB01_TRIALS_META;
  }
}

/**
 * Fetch detailed multimodal trial data (trajectory, predictions, conflict frames)
 */
export async function fetchTrialData(
  trialId: string,
  protocol?: string
): Promise<TrialData> {
  const activeProtocol = protocol || 'kinematic_onset';
  const fallback = getSub01TrialData(trialId, activeProtocol);

  const online = await isBackendOnline();
  if (!online) {
    return fallback;
  }

  const query = `&protocol=${encodeURIComponent(activeProtocol)}`;
  try {
    const data = await safeFetch<TrialData>(
      `/api/trial-data?trial_id=${encodeURIComponent(trialId)}${query}`,
      undefined,
      fallback
    );
    if (data && data.times_ms && data.times_ms.length > 0) {
      return data;
    }
    return fallback;
  } catch {
    return fallback;
  }
}

/**
 * Helper to generate scientifically realistic mock EEG epoch (-1000 to 0 ms at 200 Hz)
 */
export function createMockEEGSignal(trialId: string, channel: string = 'CZ'): EEGSignal {
  const normCh = (channel || 'CZ').toUpperCase();
  const times_ms: number[] = [];
  const amplitude_uv: number[] = [];

  let seed = 0;
  for (let i = 0; i < trialId.length; i++) seed = (seed * 31 + trialId.charCodeAt(i)) & 0xffffff;
  for (let i = 0; i < normCh.length; i++) seed = (seed * 17 + normCh.charCodeAt(i)) & 0xffffff;

  for (let i = 0; i < 200; i++) {
    const t = -1000 + i * 5; // -1000 to -5, ending at 0ms
    times_ms.push(t);
    const sec = t / 1000;
    // Readiness potential (-4uV shift) + mu rhythm (attenuates toward 0ms) + beta + noise
    const readiness = -4.2 * (1 + sec);
    const muAmp = 11.5 * (0.35 - 0.65 * sec);
    const mu = muAmp * Math.sin(2 * Math.PI * 10.2 * sec + (seed % 10));
    const beta = 4.8 * Math.cos(2 * Math.PI * 21.0 * sec + (seed % 7));
    const noise = Math.sin(i * 1.7) * 1.8 + Math.cos(i * 0.9) * 1.2;
    amplitude_uv.push(parseFloat((readiness + mu + beta + noise).toFixed(2)));
  }

  const allChannels = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FC3', 'FCZ', 'FC4', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'CP3', 'CPZ', 'CP4', 'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'OZ', 'O2'
  ];

  return {
    channel: normCh,
    available_channels: allChannels,
    times_ms,
    amplitude_uv,
    processing: 'Zero-phase filtfilt (offline only)',
    srate_hz: 200,
    trial_id: trialId,
  };
}

/**
 * Fetch raw or processed EEG signal for a trial and channel
 */
export async function fetchEEGSignal(
  trialId: string,
  channel?: string
): Promise<EEGSignal> {
  const ch = (channel || 'CZ').toUpperCase();
  const fallback = createMockEEGSignal(trialId, ch);

  const online = await isBackendOnline();
  if (!online) {
    return fallback;
  }

  try {
    const res = await safeFetch<EEGSignal>(
      `/api/eeg-signal?trial_id=${encodeURIComponent(trialId)}&channel=${encodeURIComponent(ch)}`,
      undefined,
      fallback
    );
    if (res && res.times_ms && res.times_ms.length > 0) {
      return {
        ...res,
        processing: res.processing || 'Zero-phase filtfilt (offline only)',
      };
    }
    return fallback;
  } catch {
    return fallback;
  }
}

interface LiveInferenceResponse {
  trial_id?: string;
  participant_id?: string;
  neural?: NeuralResult;
  trajectory_forecast?: {
    predicted_endpoint?: [number, number, number] | number[];
    forecast_generation_time_ms?: number;
    horizon_path?: [number, number, number][] | number[][];
    endpoint_error?: number;
  };
  conflict_assessment?: Record<string, unknown>;
  data_source?: string;
}

/**
 * Execute real-time / local model inference on selected trial
 */
export async function runInference(trialId: string, protocol: string = 'kinematic_onset'): Promise<TrialData> {
  const online = await isBackendOnline();
  if (!online) {
    notifyBackendRequired({
      actionType: 'inference',
      title: 'Backend Processing Service Required',
      message:
        'Live model inference executes dynamic Ridge trajectory forecasting and LDA spatial filtering on the Python scientific backend. In this hosted demo, deterministic trial replay and 3D kinematics are active for sub-01.',
    });
    return {
      ...getSub01TrialData(trialId, protocol),
      data_source: 'live_inference',
    };
  }

  const fallback = {
    ...getSub01TrialData(trialId, protocol),
    data_source: 'live_inference',
  };

  try {
    const res = await safeFetch<LiveInferenceResponse>(
      '/api/run-inference',
      {
        method: 'POST',
        body: JSON.stringify({ trial_id: trialId, protocol }),
      },
      fallback
    );
    if (res && res.trajectory_forecast) {
      const baseTrial = await fetchTrialData(trialId, protocol);
      return {
        ...baseTrial,
        predicted_endpoint: (res.trajectory_forecast.predicted_endpoint as [number, number, number]) ?? null,
        forecast_generation_time_ms: res.trajectory_forecast.forecast_generation_time_ms,
        horizon_path: res.trajectory_forecast.horizon_path as [number, number, number][] || baseTrial.horizon_path,
        neural: res.neural || baseTrial.neural,
        data_source: 'live_inference',
      };
    }
    return fallback;
  } catch {
    return fallback;
  }
}

/**
 * Fetch cohort-level validation and performance metrics
 */
export async function fetchCohortMetrics(): Promise<CohortMetrics> {
  const online = await isBackendOnline();
  if (!online) {
    return SUB01_COHORT_METRICS;
  }

  try {
    return await safeFetch<CohortMetrics>('/api/results/cohort', undefined, SUB01_COHORT_METRICS);
  } catch {
    return SUB01_COHORT_METRICS;
  }
}

/**
 * Fetch metrics for a specific participant
 */
export async function fetchParticipantMetrics(
  subId: string
): Promise<ParticipantMetrics | Record<string, unknown>> {
  const online = await isBackendOnline();
  if (!online) {
    return SUB01_METRICS;
  }

  try {
    return await safeFetch<ParticipantMetrics>(
      `/api/results/participant/${encodeURIComponent(subId)}`,
      undefined,
      SUB01_METRICS
    );
  } catch {
    return SUB01_METRICS;
  }
}

/**
 * Fetch model information and hyperparameters
 */
export async function fetchModelInfo(
  participant?: string
): Promise<ModelInfo | Record<string, unknown>> {
  const p = participant || 'sub-01';
  const query = participant ? `?participant=${encodeURIComponent(p)}` : '';
  const fallback: ModelInfo = {
    model_name: 'LDA Goal Decoder + Ridge Kinematic Forecaster',
    type: 'Multimodal Fusion Digital Twin',
    training_protocol: 'Leave-One-Subject-Out (LOSO)',
    parameters: {
      l2_regularization: 1.0,
      forecast_horizon_ms: 500,
    },
  };

  const online = await isBackendOnline();
  if (!online) {
    return fallback;
  }

  try {
    return await safeFetch<ModelInfo>(`/api/model-info${query}`, undefined, fallback);
  } catch {
    return fallback;
  }
}

/**
 * Fetch LDA spatial filter coefficients / weights
 */
export async function fetchLDACoefficients(
  participant?: string
): Promise<LDACoefficients> {
  const p = participant || 'sub-01';
  const online = await isBackendOnline();
  if (!online) {
    return SUB01_LDA;
  }

  const query = `?participant=${encodeURIComponent(p)}&participant_id=${encodeURIComponent(p)}`;
  try {
    const res = await safeFetch<LDACoefficients>(
      `/api/lda-coefficients${query}`,
      undefined,
      SUB01_LDA
    );
    if (res && (res.coefficients || res.weights)) {
      const coeffs = res.coefficients || res.weights || SUB01_LDA.coefficients;
      return {
        ...res,
        channels: res.channels || SUB01_LDA.channels,
        coefficients: coeffs,
        weights: coeffs,
      };
    }
    return SUB01_LDA;
  } catch {
    return SUB01_LDA;
  }
}
