'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';
import {
  AlertTriangle,
  Cpu,
  Navigation,
  Compass,
  Gauge,
  RefreshCw,
  Info,
} from 'lucide-react';
import SceneControls from '@/components/twin3d/SceneControls';
import { fetchParticipants, fetchTrials, fetchTrialData } from '@/lib/api';
import { TrialData, TrialMeta } from '@/lib/types';
import { cn } from '@/lib/utils';
import { useBackend } from '@/lib/backend-context';

// Dynamically import TwinCanvas with SSR disabled
const TwinCanvas = dynamic(() => import('@/components/twin3d/TwinCanvas'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[460px] flex flex-col items-center justify-center bg-[#FAF9F5] rounded-2xl border border-[#D9D8D0] text-[#6B6B66] font-body">
      <div className="w-6 h-6 border-2 border-[#3A5244] border-t-transparent rounded-full animate-spin mb-2" />
      <span className="text-xs font-semibold text-[#1C1C1A]">Initializing 3D WebGL Canvas...</span>
    </div>
  ),
});

function generateFallbackTrialData(trialId: string): TrialData {
  const isRight = trialId.includes('_R') || trialId.includes('right');
  const targetX = isRight ? 188 : -188;
  const targetY = isRight ? 149 : 150;
  const targetZ = isRight ? 51 : 53;

  const durationMs = 1200;
  const dt = 20;
  const numSteps = Math.floor(durationMs / dt) + 1;

  const times_ms: number[] = [];
  const hand_x: number[] = [];
  const hand_y: number[] = [];
  const hand_z: number[] = [];

  for (let i = 0; i < numSteps; i++) {
    const tMs = i * dt;
    times_ms.push(tMs);
    const tau = tMs / durationMs;
    const s = Math.max(0, Math.min(1, 10 * Math.pow(tau, 3) - 15 * Math.pow(tau, 4) + 6 * Math.pow(tau, 5)));

    hand_x.push(parseFloat((s * targetX).toFixed(2)));
    hand_y.push(parseFloat((150 + s * (targetY - 150) + 12 * Math.sin(Math.PI * tau)).toFixed(2)));
    hand_z.push(parseFloat((50 + s * (targetZ - 50) + 8 * Math.sin(Math.PI * tau)).toFixed(2)));
  }

  const forecastOnsetMs = 280;
  const tGen = forecastOnsetMs;
  const prefixIdx = Math.floor(tGen / dt);
  const startX = hand_x[prefixIdx] || 0;
  const startY = hand_y[prefixIdx] || 150;
  const startZ = hand_z[prefixIdx] || 50;

  const predicted_endpoint: [number, number, number] = [
    parseFloat((targetX + (isRight ? -4.2 : 3.8)).toFixed(2)),
    parseFloat((targetY + 2.1).toFixed(2)),
    parseFloat((targetZ - 1.5).toFixed(2)),
  ];

  const forecast_continuation = [];
  for (let t = tGen; t <= durationMs; t += dt) {
    const ratio = (t - tGen) / Math.max(durationMs - tGen, 1);
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
    participant_id: trialId.split('_')[0] || 'sub-01',
    times_ms,
    hand_x,
    hand_y,
    hand_z,
    predicted_endpoint,
    forecast_generation_time_ms: forecastOnsetMs,
    forecast_continuation,
    neural: {
      predicted_goal: isRight ? 'right' : 'left',
      confidence: 0.88,
      probability_left: isRight ? 0.12 : 0.88,
      probability_right: isRight ? 0.88 : 0.12,
      prediction_time_ms: 0,
      model_name: 'Phase 4 Multimodal LDA',
      status: 'success',
      cv_method: 'LOSO',
      balanced_accuracy: 0.6834,
      target_variable: 'instructed_reach_target',
      features_used: ['mu_power', 'beta_power', 'csp_component_1', 'csp_component_2'],
    },
    metadata: {
      trial_id: trialId,
      participant_id: trialId.split('_')[0] || 'sub-01',
      target_goal: isRight ? 'right' : 'left',
      congruency: 'congruent',
      physical_hand_direction: isRight ? 'right' : 'left',
      ball_color: isRight ? 'orange' : 'blue',
    },
    data_source: 'demo_fallback',
  };
}

export default function DigitalTwinPage() {
  const { isBackendConnected, showBackendModal } = useBackend();
  const [participants, setParticipants] = useState<string[]>(['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15']);
  const [selectedParticipant, setSelectedParticipant] = useState<string>('sub-01');
  const [trials, setTrials] = useState<TrialMeta[]>([]);
  const [selectedTrialId, setSelectedTrialId] = useState<string>('sub-01_run-01_L1_tr00');

  const [trialData, setTrialData] = useState<TrialData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);

  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1.0);

  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [showAxes, setShowAxes] = useState<boolean>(true);
  const [showForecast, setShowForecast] = useState<boolean>(true);
  const [resetCameraSignal, setResetCameraSignal] = useState<number>(0);

  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  const maxDuration = useMemo(() => {
    if (!trialData?.times_ms || trialData.times_ms.length === 0) return 1200;
    return trialData.times_ms[trialData.times_ms.length - 1];
  }, [trialData]);

  const loadTrial = useCallback(async (trialId: string) => {
    setIsLoading(true);
    try {
      const data = await fetchTrialData(trialId);
      if (
        data &&
        Array.isArray(data.hand_x) &&
        data.hand_x.length > 0
      ) {
        setTrialData(data);
        setApiError(null);
      } else {
        setTrialData(generateFallbackTrialData(trialId));
      }
    } catch {
      setTrialData(generateFallbackTrialData(trialId));
    } finally {
      setCurrentTime(0);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function init() {
      try {
        const pList = await fetchParticipants();
        if (isMounted && pList && pList.length > 0) {
          setParticipants(pList);
        }
      } catch (err) {
        console.warn('Could not load participants list:', err);
      }

      try {
        const tList = await fetchTrials('sub-01');
        if (isMounted && tList && tList.length > 0) {
          setTrials(tList);
          const firstTrial = tList[0].trial_id;
          setSelectedTrialId(firstTrial);
          await loadTrial(firstTrial);
        } else {
          await loadTrial('sub-01_run-01_L1_tr00');
        }
      } catch {
        await loadTrial('sub-01_run-01_L1_tr00');
      }
    }
    init();

    return () => {
      isMounted = false;
    };
  }, [loadTrial]);

  const handleParticipantChange = async (newPart: string) => {
    if (!isBackendConnected && newPart !== 'sub-01') {
      showBackendModal({ actionType: 'participant' });
      return;
    }
    setSelectedParticipant(newPart);
    try {
      const tList = await fetchTrials(newPart);
      if (tList && tList.length > 0) {
        setTrials(tList);
        setSelectedTrialId(tList[0].trial_id);
        await loadTrial(tList[0].trial_id);
      } else {
        const fallbackId = `${newPart}_run-01_L1_tr00`;
        setSelectedTrialId(fallbackId);
        await loadTrial(fallbackId);
      }
    } catch {
      const fallbackId = `${newPart}_run-01_L1_tr00`;
      setSelectedTrialId(fallbackId);
      await loadTrial(fallbackId);
    }
  };

  useEffect(() => {
    if (!isPlaying) {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      lastTimeRef.current = null;
      return;
    }

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = timestamp;
      }
      const delta = (timestamp - lastTimeRef.current) * speed;
      lastTimeRef.current = timestamp;

      setCurrentTime((prev) => {
        const next = prev + delta;
        if (next >= maxDuration) {
          return 0;
        }
        return next;
      });

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isPlaying, speed, maxDuration]);

  const currentCoords = useMemo(() => {
    if (
      !trialData ||
      !trialData.times_ms ||
      trialData.times_ms.length === 0 ||
      !trialData.hand_x ||
      trialData.hand_x.length === 0
    ) {
      return { x: 0, y: 150, z: 50, speedMmS: 0 };
    }

    const { times_ms, hand_x, hand_y, hand_z } = trialData;
    const len = Math.min(times_ms.length, hand_x.length, hand_y.length, hand_z.length);

    if (currentTime <= times_ms[0]) {
      return { x: hand_x[0], y: hand_y[0], z: hand_z[0], speedMmS: 0 };
    }
    if (currentTime >= times_ms[len - 1]) {
      return { x: hand_x[len - 1], y: hand_y[len - 1], z: hand_z[len - 1], speedMmS: 0 };
    }

    let idx = 0;
    for (let i = 0; i < len - 1; i++) {
      if (currentTime >= times_ms[i] && currentTime <= times_ms[i + 1]) {
        idx = i;
        break;
      }
    }

    const t0 = times_ms[idx];
    const t1 = times_ms[idx + 1];
    const dt = Math.max(t1 - t0, 0.001);
    const alpha = Math.max(0, Math.min(1, (currentTime - t0) / dt));

    const x = hand_x[idx] + alpha * (hand_x[idx + 1] - hand_x[idx]);
    const y = hand_y[idx] + alpha * (hand_y[idx + 1] - hand_y[idx]);
    const z = hand_z[idx] + alpha * (hand_z[idx + 1] - hand_z[idx]);

    const dx = hand_x[idx + 1] - hand_x[idx];
    const dy = hand_y[idx + 1] - hand_y[idx];
    const dz = hand_z[idx + 1] - hand_z[idx];
    const dist = Math.hypot(dx, dy, dz);
    const speedMmS = (dist / dt) * 1000;

    return { x, y, z, speedMmS };
  }, [trialData, currentTime]);

  const trajectoryStats = useMemo(() => {
    if (!trialData || !trialData.hand_x || trialData.hand_x.length === 0) {
      return {
        sampleCount: 0,
        peakSpeed: 0,
        distToTargetLeft: 0,
        distToTargetRight: 0,
      };
    }

    let peak = 0;
    const len = trialData.hand_x.length;
    for (let i = 0; i < len - 1; i++) {
      const dt = Math.max((trialData.times_ms[i + 1] || 20) - (trialData.times_ms[i] || 0), 1);
      const dx = (trialData.hand_x[i + 1] || 0) - (trialData.hand_x[i] || 0);
      const dy = (trialData.hand_y[i + 1] || 0) - (trialData.hand_y[i] || 0);
      const dz = (trialData.hand_z[i + 1] || 0) - (trialData.hand_z[i] || 0);
      const spd = (Math.hypot(dx, dy, dz) / dt) * 1000;
      if (spd > peak) peak = spd;
    }

    const distLeft = Math.hypot(
      currentCoords.x - -188,
      currentCoords.y - 150,
      currentCoords.z - 53
    );
    const distRight = Math.hypot(
      currentCoords.x - 188,
      currentCoords.y - 149,
      currentCoords.z - 51
    );

    return {
      sampleCount: len,
      peakSpeed: Math.round(peak),
      distToTargetLeft: Math.round(distLeft),
      distToTargetRight: Math.round(distRight),
    };
  }, [trialData, currentCoords]);

  const availableTrials = useMemo(() => {
    if (trials.length > 0) return trials;
    return [
      {
        trial_id: `${selectedParticipant}_run-01_L1_tr00`,
        participant_id: selectedParticipant,
        congruency: 'congruent',
        target_goal: 'left',
        physical_hand_direction: 'left',
      },
      {
        trial_id: `${selectedParticipant}_run-01_R1_tr01`,
        participant_id: selectedParticipant,
        congruency: 'congruent',
        target_goal: 'right',
        physical_hand_direction: 'right',
      },
      {
        trial_id: `${selectedParticipant}_run-02_L2_tr05`,
        participant_id: selectedParticipant,
        congruency: 'incongruent',
        target_goal: 'left',
        physical_hand_direction: 'left',
      },
      {
        trial_id: `${selectedParticipant}_run-02_R2_tr08`,
        participant_id: selectedParticipant,
        congruency: 'incongruent',
        target_goal: 'right',
        physical_hand_direction: 'right',
      },
    ] as TrialMeta[];
  }, [trials, selectedParticipant]);

  const meta = trialData?.metadata as TrialMeta | undefined;
  const neural = trialData?.neural;

  return (
    <div className="w-full max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20 space-y-8 font-body">
      {/* Editorial Header */}
      <div className="border-b border-[#D9D8D0] pb-8 space-y-4">
        <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">
          Spatial Workspace &bull; Leap Motion 3D Kinematics
        </p>

        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-6">
          <div>
            <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#1C1C1A]">
              3D Digital Twin Workspace
            </h1>
            <p className="text-base sm:text-lg text-[#6B6B66] max-w-2xl mt-2 leading-relaxed font-body">
              Spatial kinematics visualization of recorded hand movement and Ridge trajectory forecast horizon in Leap Motion coordinate space.
            </p>
          </div>

          {apiError && (
            <div>
              <div className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-body bg-[#FEF3C7]/60 text-[#92400E] border border-[#D97706]/30">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                <span>Simulation Fallback</span>
              </div>
            </div>
          )}
        </div>

        {apiError && (
          <div className="p-4 rounded-2xl bg-[#FEF3C7]/40 border border-[#D97706]/30 text-xs text-[#92400E] flex items-center gap-2 font-body">
            <Info className="w-4 h-4 shrink-0" />
            <span>{apiError}</span>
          </div>
        )}
      </div>

      {/* 3D Canvas Viewport */}
      <div className="w-full h-[58vh] min-h-[460px] relative rounded-2xl overflow-hidden border border-[#D9D8D0] shadow-sm">
        <TwinCanvas
          trialData={trialData}
          currentTime={currentTime}
          showGrid={showGrid}
          showAxes={showAxes}
          showForecast={showForecast}
          resetCameraSignal={resetCameraSignal}
        />
      </div>

      {/* Scene Controls */}
      <SceneControls
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying(!isPlaying)}
        speed={speed}
        onSpeedChange={setSpeed}
        currentTime={currentTime}
        maxDuration={maxDuration}
        onSeek={(time) => setCurrentTime(time)}
        showGrid={showGrid}
        onToggleGrid={() => setShowGrid(!showGrid)}
        showAxes={showAxes}
        onToggleAxes={() => setShowAxes(!showAxes)}
        showForecast={showForecast}
        onToggleForecast={() => setShowForecast(!showForecast)}
        onResetCamera={() => setResetCameraSignal((prev) => prev + 1)}
      />

      {/* Two Column Lower Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left Column: Trial Configuration & Intention Manifest */}
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-4 shadow-sm">
            <div className="flex items-center gap-2 border-b border-[#D9D8D0] pb-3">
              <Navigation className="w-4 h-4 text-[#3A5244]" />
              <h2 className="font-display text-base font-bold text-[#1C1C1A]">
                Trial Selector
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-[#1C1C1A] mb-1 font-body">
                  Participant
                </label>
                <select
                  value={selectedParticipant}
                  onChange={(e) => handleParticipantChange(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-body rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] text-[#1C1C1A] focus:outline-none focus:border-[#3A5244] cursor-pointer"
                >
                  {participants.map((p) => (
                    <option key={p} value={p}>
                      {p}
                      {!isBackendConnected && p === 'sub-01' ? ' — Included in Demo' : ''}
                      {!isBackendConnected && p !== 'sub-01' ? ' — Requires Local Backend' : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#1C1C1A] mb-1 font-body">
                  Trial Identifier
                </label>
                <select
                  value={selectedTrialId}
                  onChange={(e) => setSelectedTrialId(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-body rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] text-[#1C1C1A] focus:outline-none focus:border-[#3A5244] truncate cursor-pointer"
                >
                  {availableTrials.map((t) => (
                    <option key={t.trial_id} value={t.trial_id}>
                      {t.trial_id} ({t.target_goal || 'left'})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <span className="text-xs text-[#6B6B66] font-body">
                Trial: <strong className="text-[#1C1C1A]">{selectedTrialId}</strong>
              </span>
              <button
                type="button"
                onClick={() => loadTrial(selectedTrialId)}
                disabled={isLoading}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] hover:bg-[#3A5244] text-[#F5F3EE] transition-colors disabled:opacity-50 cursor-pointer"
              >
                <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
                <span>{isLoading ? 'Loading...' : 'Load Trial Data'}</span>
              </button>
            </div>
          </div>

          {/* Decoding Metadata Card */}
          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-[#3A5244]" />
                <h2 className="font-display text-base font-bold text-[#1C1C1A]">
                  Decoded Parameters
                </h2>
              </div>
              <span className="text-[10px] font-body px-2.5 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] border border-[#D9D8D0]">
                {neural?.model_name || 'Phase 4 Multimodal LDA'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] uppercase font-body">Goal</div>
                <div className="text-xs font-bold text-[#1C1C1A] capitalize mt-0.5 font-display">
                  {String(meta?.target_goal || 'Left')}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] uppercase font-body">Congruency</div>
                <div className="text-xs font-bold capitalize mt-0.5 text-[#1C1C1A] font-display">
                  {String(meta?.congruency || 'Congruent')}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] uppercase font-body">Movement</div>
                <div className="text-xs font-bold text-[#1C1C1A] capitalize mt-0.5 font-display">
                  {String(meta?.physical_hand_direction || 'Left')}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] uppercase font-body">Cue Color</div>
                <div className="text-xs font-bold text-[#1C1C1A] capitalize mt-0.5 font-display">
                  {String(meta?.ball_color || 'Blue')}
                </div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#6B6B66] font-body">
                  Pre-movement Neural Goal:{' '}
                  <strong className="text-[#1C1C1A] uppercase font-display">
                    {neural?.predicted_goal || 'LEFT'}
                  </strong>
                </span>
                <span className="font-bold text-xs text-[#1C1C1A] font-display">
                  {((neural?.confidence || 0.88) * 100).toFixed(1)}% Confidence
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-[#EDECE6] overflow-hidden flex">
                <div
                  className="h-full bg-[#2563EB] transition-all duration-300"
                  style={{ width: `${((neural?.probability_left ?? 0.88) * 100).toFixed(0)}%` }}
                />
                <div
                  className="h-full bg-[#D97706] transition-all duration-300"
                  style={{ width: `${((neural?.probability_right ?? 0.12) * 100).toFixed(0)}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[11px] text-[#6B6B66] font-body">
                <span>P(Left): {((neural?.probability_left ?? 0.88) * 100).toFixed(1)}%</span>
                <span>P(Right): {((neural?.probability_right ?? 0.12) * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Physical Coordinate Telemetry & Statistics */}
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
              <div className="flex items-center gap-2">
                <Compass className="w-4 h-4 text-[#3A5244]" />
                <h2 className="font-display text-base font-bold text-[#1C1C1A]">
                  Coordinate Telemetry (mm)
                </h2>
              </div>
              <span className="text-[10px] font-body px-2.5 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] border border-[#D9D8D0]">
                1 mm = 0.01 3D units
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="p-3.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col">
                <span className="text-[10px] font-body text-[#6B6B66] uppercase">X (Lateral)</span>
                <span className="text-xl font-bold font-display text-[#2E6B47] mt-0.5">
                  {currentCoords.x.toFixed(1)}
                  <span className="text-xs text-[#6B6B66] font-normal font-body ml-0.5">mm</span>
                </span>
                <span className="text-[10px] text-[#6B6B66] font-body mt-0.5">
                  {currentCoords.x < 0 ? 'Left' : currentCoords.x > 0 ? 'Right' : 'Center'}
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col">
                <span className="text-[10px] font-body text-[#6B6B66] uppercase">Y (Height)</span>
                <span className="text-xl font-bold font-display text-[#1C1C1A] mt-0.5">
                  {currentCoords.y.toFixed(1)}
                  <span className="text-xs text-[#6B6B66] font-normal font-body ml-0.5">mm</span>
                </span>
                <span className="text-[10px] text-[#6B6B66] font-body mt-0.5">Desk baseline</span>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col">
                <span className="text-[10px] font-body text-[#6B6B66] uppercase">Z (Depth)</span>
                <span className="text-xl font-bold font-display text-[#2563EB] mt-0.5">
                  {currentCoords.z.toFixed(1)}
                  <span className="text-xs text-[#6B6B66] font-normal font-body ml-0.5">mm</span>
                </span>
                <span className="text-[10px] text-[#6B6B66] font-body mt-0.5">Forward reach</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex items-center justify-between">
                <div>
                  <div className="text-[10px] text-[#6B6B66] font-body">Distance to Left Target</div>
                  <div className="font-bold font-display text-sm text-[#1C1C1A] mt-0.5">
                    {trajectoryStats.distToTargetLeft} mm
                  </div>
                </div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#2563EB]" />
              </div>

              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex items-center justify-between">
                <div>
                  <div className="text-[10px] text-[#6B6B66] font-body">Distance to Right Target</div>
                  <div className="font-bold font-display text-sm text-[#1C1C1A] mt-0.5">
                    {trajectoryStats.distToTargetRight} mm
                  </div>
                </div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#D97706]" />
              </div>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-[#3A5244]" />
                <h2 className="font-display text-base font-bold text-[#1C1C1A]">
                  Kinematic Metrics
                </h2>
              </div>
              <span className="text-xs text-[#6B6B66] font-body">
                {trajectoryStats.sampleCount} samples
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] font-body">Velocity</div>
                <div className="text-xs font-bold font-display text-[#1C1C1A] mt-0.5">
                  {Math.round(currentCoords.speedMmS)} mm/s
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] font-body">Peak Speed</div>
                <div className="text-xs font-bold font-display text-[#1C1C1A] mt-0.5">
                  {trajectoryStats.peakSpeed} mm/s
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] font-body">Duration</div>
                <div className="text-xs font-bold font-display text-[#1C1C1A] mt-0.5">
                  {maxDuration.toFixed(0)} ms
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
                <div className="text-[10px] text-[#6B6B66] font-body">Forecast Onset</div>
                <div className="text-xs font-bold font-display text-[#1C1C1A] mt-0.5">
                  {(trialData?.forecast_generation_time_ms ?? 280).toFixed(0)} ms
                </div>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] text-xs flex items-center justify-between font-body">
              <span className="text-[#6B6B66]">Forecasted 3D Endpoint:</span>
              <span className="text-[#6B46C1] font-bold font-display">
                [
                {Array.isArray(trialData?.predicted_endpoint)
                  ? trialData?.predicted_endpoint.map((v) => Number(v).toFixed(1)).join(', ')
                  : '-188.0, 150.0, 53.0'}
                ] mm
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
