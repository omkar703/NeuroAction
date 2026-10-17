'use client';

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Radio, Crosshair, ArrowRight, Brain } from 'lucide-react';
import { TrialData } from '@/lib/types';

// Dynamic import of TwinCanvas with SSR disabled
const TwinCanvas = dynamic(() => import('@/components/twin3d/TwinCanvas'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[220px] flex flex-col items-center justify-center bg-[#F5F8F5] rounded-xl text-[#6B6B66] font-body">
      <div className="w-5 h-5 border-2 border-[#3A5244] border-t-transparent rounded-full animate-spin mb-1.5" />
      <span className="text-[11px] font-semibold text-[#1C1C1A]">Loading 3D Workspace...</span>
    </div>
  ),
});

// Generate canonical left-reach trial data for hero demonstration
function createHeroTrialData(): TrialData {
  const durationMs = 1200;
  const dt = 20;
  const numSteps = Math.floor(durationMs / dt) + 1;

  const times_ms: number[] = [];
  const hand_x: number[] = [];
  const hand_y: number[] = [];
  const hand_z: number[] = [];

  const startX = 0;
  const startY = 150;
  const startZ = 50;

  const targetX = -188;
  const targetY = 150;
  const targetZ = 53;

  for (let i = 0; i < numSteps; i++) {
    const t = i * dt;
    times_ms.push(t);

    const ratio = t / durationMs;
    // Minimum jerk smooth reach trajectory profile
    const s = 10 * Math.pow(ratio, 3) - 15 * Math.pow(ratio, 4) + 6 * Math.pow(ratio, 5);

    hand_x.push(parseFloat((startX + (targetX - startX) * s).toFixed(2)));
    hand_y.push(parseFloat((startY + (targetY - startY) * s).toFixed(2)));
    hand_z.push(parseFloat((startZ + (targetZ - startZ) * s).toFixed(2)));
  }

  const prefixIdx = Math.min(10, numSteps - 1); // 200 ms
  const prefixT = times_ms[prefixIdx];

  const forecast_continuation = [];
  for (let t = prefixT; t <= durationMs; t += dt) {
    const ratio = (t - prefixT) / (durationMs - prefixT);
    const smooth = 3 * Math.pow(ratio, 2) - 2 * Math.pow(ratio, 3);
    forecast_continuation.push({
      time_ms: t,
      x: parseFloat((hand_x[prefixIdx] + (targetX - hand_x[prefixIdx]) * smooth).toFixed(2)),
      y: parseFloat((hand_y[prefixIdx] + (targetY - hand_y[prefixIdx]) * smooth).toFixed(2)),
      z: parseFloat((hand_z[prefixIdx] + (targetZ - hand_z[prefixIdx]) * smooth).toFixed(2)),
    });
  }

  return {
    trial_id: 'sub-01_run-01_L1_tr00',
    participant_id: 'sub-01',
    times_ms,
    hand_x,
    hand_y,
    hand_z,
    predicted_endpoint: [targetX, targetY, targetZ],
    forecast_generation_time_ms: 200,
    forecast_continuation,
    neural: {
      predicted_goal: 'left',
      confidence: 0.884,
      probability_left: 0.884,
      probability_right: 0.116,
      prediction_time_ms: 0,
      model_name: 'Phase 4 Multimodal LDA',
      status: 'success',
      cv_method: 'LOSO',
      balanced_accuracy: 0.6646,
      target_variable: 'instructed_reach_target',
      features_used: ['mu_power', 'beta_power', 'csp_component_1', 'csp_component_2'],
    },
    metadata: {
      trial_id: 'sub-01_run-01_L1_tr00',
      participant_id: 'sub-01',
      target_goal: 'left',
      congruency: 'congruent',
      physical_hand_direction: 'left',
      ball_color: 'blue',
    },
    data_source: 'hero_live_demonstration',
  };
}

const HERO_TRIAL = createHeroTrialData();

const EEG_CHANNELS = [
  { name: 'CZ', color: '#3A5244', freq: 10, amp: 14 },
  { name: 'C3', color: '#2563EB', freq: 13, amp: 16 },
  { name: 'C4', color: '#B45309', freq: 15, amp: 12 },
];

export default function HeroMultimodalViz() {
  const [currentTime, setCurrentTime] = useState<number>(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  // Auto-looping trajectory clock (0 ms to 1400 ms)
  useEffect(() => {
    function tick(now: number) {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = now;
      }
      const elapsed = (now - lastTimeRef.current) * 1.0; // 1.0x playback speed
      lastTimeRef.current = now;

      setCurrentTime((prev) => {
        const next = prev + elapsed;
        if (next > 1400) {
          return 0; // seamless loop
        }
        return next;
      });

      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Draw 2D EEG waves on left canvas synchronized with playback
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, W, H);

    // Subtle Grid
    ctx.strokeStyle = '#F0EFEA';
    ctx.lineWidth = 1;
    for (let x = 40; x < W; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    const laneH = H / EEG_CHANNELS.length;
    for (let i = 1; i < EEG_CHANNELS.length; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * laneH);
      ctx.lineTo(W, i * laneH);
      ctx.stroke();
    }

    // Sweep cursor line (synchronized with currentTime)
    // currentTime loops 0 -> 1400. We map it to sweep across the EEG window
    const sweepProgress = (currentTime % 1400) / 1400;
    const sweepX = 40 + sweepProgress * (W - 40);

    ctx.strokeStyle = 'rgba(58, 82, 68, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(sweepX, 0);
    ctx.lineTo(sweepX, H);
    ctx.stroke();

    // Waveforms
    EEG_CHANNELS.forEach((ch, idx) => {
      const centerY = idx * laneH + laneH / 2;

      // Label
      ctx.fillStyle = ch.color;
      ctx.font = 'bold 10px "Open Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(ch.name, 32, centerY);

      // Dot
      ctx.beginPath();
      ctx.arc(36, centerY, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Wave path
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.6;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      const waveW = W - 42;
      for (let px = 0; px <= waveW; px += 2) {
        const t = (px / waveW) * 2 * Math.PI;
        // Dynamic desynchronization as time approaches 0 ms
        const desyncFactor = 1 - (px / waveW) * 0.35;
        const phaseOffset = idx * 1.4 + currentTime * 0.003;
        const waveY =
          centerY +
          Math.sin(t * ch.freq * 0.4 + phaseOffset) * ch.amp * desyncFactor +
          Math.sin(t * ch.freq * 0.8 + phaseOffset * 0.7) * (ch.amp * 0.35);

        if (px === 0) {
          ctx.moveTo(42 + px, waveY);
        } else {
          ctx.lineTo(42 + px, waveY);
        }
      }
      ctx.stroke();
    });
  }, [currentTime]);

  const isPrefix = currentTime <= 200;

  return (
    <div className="w-full bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-sm overflow-hidden flex flex-col font-body">
      {/* Top Header Strip: Neural Input -> Decoded Intention -> 3D Kinematics */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-[#FAF9F5] border-b border-[#D9D8D0] gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#EAEFE9] text-[#3A5244] text-[10px] font-semibold">
            <Radio className="w-2.5 h-2.5 animate-pulse" />
            <span>EEG [-1000, 0] ms</span>
          </span>
          <ArrowRight className="w-3 h-3 text-[#AEADA4]" />
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#EFF6FF] text-[#2563EB] text-[10px] font-semibold">
            <Brain className="w-2.5 h-2.5" />
            <span>Goal: Left (88%)</span>
          </span>
          <ArrowRight className="w-3 h-3 text-[#AEADA4]" />
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#EDECE6] text-[#1C1C1A] text-[10px] font-semibold">
            <Crosshair className="w-2.5 h-2.5 text-[#3A5244]" />
            <span>3D Forecast (+200 ms)</span>
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#6B6B66]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244] animate-ping" />
          <span>t = {Math.min(currentTime, 1200).toFixed(0)} ms</span>
        </div>
      </div>

      {/* Main Split Grid: Left = EEG Signals, Right = 3D Workspace */}
      <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-[#D9D8D0] h-[340px] sm:h-[370px]">
        {/* Left Pane: Pre-movement EEG Oscilloscope */}
        <div className="relative flex flex-col p-3 bg-[#FFFFFF]">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244]" />
              <span className="text-[11px] font-bold text-[#1C1C1A]">Cortical Readiness</span>
            </div>
            <span className="text-[9px] text-[#6B6B66] font-mono">8–30 Hz Sensorimotor μ/β</span>
          </div>

          <div className="flex-1 w-full relative rounded-lg border border-[#EDECE6] overflow-hidden bg-[#FFFFFF]">
            <canvas ref={canvasRef} className="w-full h-full" />
            {/* Time labels overlay */}
            <div className="absolute bottom-1 left-10 right-2 flex justify-between text-[8px] font-mono text-[#AEADA4] select-none pointer-events-none">
              <span>-1000 ms</span>
              <span>-500 ms</span>
              <span className="text-[#3A5244] font-bold">0 ms Onset</span>
            </div>
          </div>

          <div className="mt-2 flex items-center justify-between text-[10px] text-[#6B6B66]">
            <span>Contralateral motor desync</span>
            <span className="text-[#3A5244] font-semibold">CSP Spatial Filter</span>
          </div>
        </div>

        {/* Right Pane: Compact 3D Digital Twin Kinematics */}
        <div className="relative flex flex-col p-3 bg-[#F5F8F5]">
          <div className="flex items-center justify-between mb-2 z-10">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2563EB]" />
              <span className="text-[11px] font-bold text-[#1C1C1A]">3D Kinematic Extrapolation</span>
            </div>
            <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full bg-[#FFFFFF]/90 text-[#3A5244] border border-[#D9D8D0]">
              {isPrefix ? '200 ms Prefix' : 'Ridge Horizon'}
            </span>
          </div>

          <div className="flex-1 w-full relative rounded-xl border border-[#D9D8D0] overflow-hidden bg-[#F5F8F5]">
            <TwinCanvas
              trialData={HERO_TRIAL}
              currentTime={Math.min(currentTime, 1200)}
              showGrid={true}
              showAxes={false}
              showForecast={true}
              resetCameraSignal={0}
            />
          </div>

          <div className="mt-2 flex items-center justify-between text-[10px] text-[#6B6B66] z-10">
            <span>Leap Motion optical reach</span>
            <span className="font-semibold text-[#1C1C1A]">Target: (−188, 150, 53) mm</span>
          </div>
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div className="px-4 py-2 bg-[#FAF9F5] border-t border-[#D9D8D0] flex flex-wrap items-center justify-between text-[11px] text-[#6B6B66]">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[#1C1C1A]">Multimodal Causality:</span>
          <span>
            {currentTime <= 200
              ? 'Cortical intention decoded → physical hand begins reaching prefix'
              : 'Ridge regression extrapolates hand coordinates to left target endpoint (47.25 mm RMSE)'}
          </span>
        </div>
        <div className="hidden sm:flex items-center gap-1 text-[10px] text-[#3A5244] font-semibold">
          <span>Continuous Live Loop</span>
        </div>
      </div>
    </div>
  );
}
