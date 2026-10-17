'use client';

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowRight, Radio } from 'lucide-react';
import { TrialData } from '@/lib/types';

// Dynamically import Canvas and HeroWorkspace3D with SSR disabled
const HeroCanvas = dynamic(
  () =>
    import('@react-three/fiber').then((mod) => {
      const Canvas = mod.Canvas;
      return function CanvasWrapper({
        trialData,
        currentTime,
      }: {
        trialData: TrialData;
        currentTime: number;
      }) {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const HeroWorkspace3D = require('@/components/twin3d/HeroWorkspace3D').default;
        return (
          <Canvas
            camera={{ position: [0.6, 2.7, 3.8], fov: 42, near: 0.1, far: 50 }}
            gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
            className="w-full h-full"
          >
            <HeroWorkspace3D trialData={trialData} currentTime={currentTime} />
          </Canvas>
        );
      };
    }),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-[#F5F8F5]/80">
        <div className="w-8 h-8 border-2 border-[#3A5244] border-t-transparent rounded-full animate-spin" />
      </div>
    ),
  }
);

// Canonical right-reach trial data
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

  const targetX = 188;
  const targetY = 149;
  const targetZ = 51;

  for (let i = 0; i < numSteps; i++) {
    const t = i * dt;
    times_ms.push(t);
    const ratio = t / durationMs;
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
    trial_id: 'sub-01_run-01_R1_tr01',
    participant_id: 'sub-01',
    times_ms,
    hand_x,
    hand_y,
    hand_z,
    predicted_endpoint: [targetX, targetY, targetZ],
    forecast_generation_time_ms: 200,
    forecast_continuation,
    neural: {
      predicted_goal: 'right',
      confidence: 0.884,
      probability_left: 0.116,
      probability_right: 0.884,
      prediction_time_ms: 0,
      model_name: 'Phase 4 Multimodal LDA',
      status: 'success',
      cv_method: 'LOSO',
      balanced_accuracy: 0.6646,
      target_variable: 'instructed_reach_target',
      features_used: ['mu_power', 'beta_power', 'csp_component_1', 'csp_component_2'],
    },
    metadata: {
      trial_id: 'sub-01_run-01_R1_tr01',
      participant_id: 'sub-01',
      target_goal: 'right',
      congruency: 'congruent',
      physical_hand_direction: 'right',
      ball_color: 'orange',
    },
    data_source: 'hero_live_demonstration',
  };
}

const HERO_TRIAL = createHeroTrialData();

const EEG_CHANNELS = [
  { name: 'CZ', color: '#3A5244', freq: 10, amp: 14 },
  { name: 'C3', color: '#2563EB', freq: 13, amp: 16 },
  { name: 'C4', color: '#B45309', freq: 15, amp: 13 },
];

export default function HeroWideStage() {
  const [currentTime, setCurrentTime] = useState<number>(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  // Synchronous auto-looping replay clock (0 to 1400 ms)
  useEffect(() => {
    function tick(now: number) {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = now;
      }
      const elapsed = (now - lastTimeRef.current) * 0.25; // 0.25x gentle slow-motion replay speed
      lastTimeRef.current = now;

      setCurrentTime((prev) => {
        const next = prev + elapsed;
        if (next > 1500) return 0;
        return next;
      });

      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  // 2D EEG Oscilloscope rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    // Subtle grid
    ctx.strokeStyle = '#EDECE6';
    ctx.lineWidth = 1;
    const laneH = H / EEG_CHANNELS.length;
    for (let i = 1; i < EEG_CHANNELS.length; i++) {
      ctx.beginPath();
      ctx.moveTo(32, i * laneH);
      ctx.lineTo(W, i * laneH);
      ctx.stroke();
    }

    // Sweep cursor
    const sweepProgress = (currentTime % 1500) / 1500;
    const sweepX = 36 + sweepProgress * (W - 36);

    ctx.strokeStyle = 'rgba(58, 82, 68, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(sweepX, 0);
    ctx.lineTo(sweepX, H);
    ctx.stroke();

    // Wave paths
    EEG_CHANNELS.forEach((ch, idx) => {
      const centerY = idx * laneH + laneH / 2;

      // Label
      ctx.fillStyle = ch.color;
      ctx.font = 'bold 9px "Open Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(ch.name, 26, centerY);

      // Dot
      ctx.beginPath();
      ctx.arc(30, centerY, 2, 0, Math.PI * 2);
      ctx.fill();

      // Waveform
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      const waveW = W - 36;
      for (let px = 0; px <= waveW; px += 2) {
        const t = (px / waveW) * 2 * Math.PI;
        const desync = 1 - (px / waveW) * 0.35;
        const phase = idx * 1.5 + currentTime * 0.003;
        const y =
          centerY +
          Math.sin(t * ch.freq * 0.4 + phase) * ch.amp * desync +
          Math.sin(t * ch.freq * 0.8 + phase * 0.7) * (ch.amp * 0.3);

        if (px === 0) ctx.moveTo(36 + px, y);
        else ctx.lineTo(36 + px, y);
      }
      ctx.stroke();
    });
  }, [currentTime]);

  const activeTimeMs = Math.min(currentTime, 1200);

  return (
    <section className="relative w-full min-h-[calc(100vh-4rem)] flex items-center overflow-hidden bg-[#F5F3EE] border-b border-[#D9D8D0]">
      {/* ── 1. BACKGROUND WIDE 3D WORKSPACE (Spanning Full Hero Width) ── */}
      <div className="absolute inset-0 w-full h-full pointer-events-auto">
        <HeroCanvas trialData={HERO_TRIAL} currentTime={activeTimeMs} />
      </div>

      {/* ── 2. BLUR & FROSTED GRADIENT MASK (Soft Left Overlay for Text Legibility) ── */}
      <div
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-full lg:w-[62%] pointer-events-none bg-gradient-to-r from-[#F5F3EE] via-[#F5F3EE]/92 to-[#F5F3EE]/10 lg:to-transparent backdrop-blur-[1.5px]"
      />

      {/* ── 3. FLOATING SYNCHRONIZED NEURAL STREAM CARD (Top-Right of Hero) ── */}
      <div className="hidden md:flex absolute top-6 right-6 lg:right-10 z-20 pointer-events-auto">
        <div className="w-[420px] lg:w-[440px] bg-[#FFFFFF]/92 backdrop-blur-md border border-[#D9D8D0] rounded-2xl p-4 shadow-sm space-y-3 font-body">
          {/* Card Header */}
          <div className="flex items-center justify-between pb-2.5 border-b border-[#D9D8D0]">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-[#3A5244] animate-pulse" />
              <span className="text-xs font-bold text-[#1C1C1A]">
                Sensorimotor Stream
              </span>
            </div>
            <span className="text-[11px] font-mono font-semibold px-2.5 py-0.5 rounded-full bg-[#EDECE6] text-[#1C1C1A]">
              t = {activeTimeMs.toFixed(0)} ms
            </span>
          </div>

          {/* Mini Waveform Oscilloscope */}
          <div className="h-[115px] w-full rounded-xl bg-[#FAF9F5] border border-[#EDECE6] overflow-hidden relative">
            <canvas ref={canvasRef} className="w-full h-full" />
            <div className="absolute bottom-1.5 right-2.5 text-[9px] font-mono text-[#AEADA4]">
              μ/β desync (8–30 Hz)
            </div>
          </div>
        </div>
      </div>

      {/* ── 4. FOREGROUND HERO CONTENT (Positioned Prominently on Top) ── */}
      <div className="relative z-10 w-full max-w-6xl mx-auto px-6 lg:px-10 py-12 sm:py-16 lg:py-20 pointer-events-none">
        <div className="max-w-xl pointer-events-auto">
          {/* Top Overline */}
          <p className="text-xs tracking-[0.3em] uppercase text-[#6B6B66] mb-4 font-body">
            NEURAL INTENTION–ACTION RESEARCH
          </p>

          {/* Main Headline */}
          <h1 className="font-display text-4xl sm:text-5xl xl:text-6xl font-bold leading-[1.06] tracking-tight text-[#1C1C1A]">
            From Neural Signals<br />
            <span className="text-[#3A5244]">to Predicted</span><br />
            Movement.
          </h1>

          {/* Subtitle Description */}
          <p className="mt-6 text-base sm:text-lg text-[#6B6B66] leading-[1.8] font-body">
            NeuroAction decodes pre-movement scalp EEG goal intentions
            and forecasts 3D limb trajectories — 200&nbsp;ms before full
            reach completion.
          </p>

          {/* Action CTAs */}
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <Link
              href="/science"
              className="inline-flex items-center gap-2.5 px-6 py-3 bg-[#1C1C1A] text-[#F5F3EE] text-sm font-semibold tracking-wide rounded-full hover:bg-[#3A5244] transition-colors duration-200 font-body shadow-xs"
            >
              Explore the Science <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/model-lab"
              className="inline-flex items-center gap-2 px-6 py-3 border border-[#D9D8D0] bg-[#FFFFFF]/80 backdrop-blur-md text-[#1C1C1A] text-sm font-body tracking-wide rounded-full hover:bg-[#EDECE6] transition-colors duration-200 shadow-xs"
            >
              Open Model Lab
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
