'use client';

import React, { useMemo } from 'react';
import { Brain, ArrowRight, Crosshair, Radio } from 'lucide-react';
import { TrialData } from '@/lib/types';

interface TwinSignalFlowBannerProps {
  trialData: TrialData | null;
  currentTime: number;
  isPlaying: boolean;
}

export default function TwinSignalFlowBanner({
  trialData,
  currentTime,
  isPlaying,
}: TwinSignalFlowBannerProps) {
  const isRight = trialData?.metadata?.target_goal === 'right' || trialData?.neural?.predicted_goal === 'right';
  const targetColor = isRight ? '#B45309' : '#2563EB';
  const targetBg = isRight ? '#FEF3C7' : '#EFF6FF';
  const targetLabel = isRight ? 'Right Target (+188 mm)' : 'Left Target (−188 mm)';
  const confidence = trialData?.neural?.confidence
    ? (trialData.neural.confidence * 100).toFixed(1)
    : '88.0';

  // Calculate current hand coordinates
  const currentHandPos = useMemo(() => {
    if (!trialData?.times_ms || !trialData.hand_x || trialData.hand_x.length === 0) {
      return { x: 0, y: 150, z: 50, speed: 0 };
    }
    const times = trialData.times_ms;
    let idx = times.findIndex((t) => t >= currentTime);
    if (idx === -1) idx = times.length - 1;
    const x = trialData.hand_x[idx] ?? 0;
    const y = trialData.hand_y[idx] ?? 150;
    const z = trialData.hand_z[idx] ?? 50;

    // Approximate instantaneous velocity
    let speed = 0;
    if (idx > 0 && times[idx] !== times[idx - 1]) {
      const dt = (times[idx] - times[idx - 1]) / 1000;
      const dx = x - (trialData.hand_x[idx - 1] ?? x);
      const dy = y - (trialData.hand_y[idx - 1] ?? y);
      const dz = z - (trialData.hand_z[idx - 1] ?? z);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      speed = Math.round(dist / Math.max(dt, 0.001));
    }
    return { x, y, z, speed };
  }, [trialData, currentTime]);

  const isPrefix = currentTime <= 200;

  return (
    <div className="w-full bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-4 sm:p-5 shadow-xs font-body transition-all">
      {/* Top Header Label */}
      <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0] mb-3.5">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#3A5244] animate-pulse" />
          <span className="text-[11px] font-bold tracking-[0.25em] uppercase text-[#6B6B66]">
            Neural–Kinematic Causality Stream
          </span>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs text-[#6B6B66]">
          <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-[#EDECE6] font-semibold text-[#1C1C1A]">
            t = {currentTime.toFixed(0)} ms
          </span>
          <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-[#EAEFE9] text-[#3A5244] font-semibold">
            {isPlaying ? 'Replay Active' : 'Paused'}
          </span>
        </div>
      </div>

      {/* 3-Stage Signal Flow Grid */}
      <div className="grid grid-cols-1 md:grid-cols-11 gap-3 items-center">
        {/* Stage 1: Pre-movement Scalp EEG (Cols 1-3) */}
        <div className={`md:col-span-3 p-3 rounded-xl border transition-all ${
          isPrefix ? 'bg-[#FAF9F5] border-[#3A5244]/40 shadow-xs ring-1 ring-[#3A5244]/20' : 'bg-[#FAF9F5] border-[#D9D8D0]'
        }`}>
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-[#3A5244]" />
              <span className="text-xs font-bold text-[#1C1C1A]">1. Cortical Scalp EEG</span>
            </div>
            <span className="text-[9px] font-semibold px-2 py-0.2 rounded-full bg-[#EAEFE9] text-[#3A5244]">
              [-1000, 0] ms
            </span>
          </div>

          <p className="text-[11px] text-[#6B6B66] leading-snug mb-2">
            Sensorimotor μ (8–12 Hz) &amp; β (13–30 Hz) rhythm desynchronisation.
          </p>

          <div className="flex items-center justify-between text-[10px] text-[#6B6B66] pt-1.5 border-t border-[#D9D8D0]/60">
            <span>Channels: <strong className="text-[#1C1C1A]">C3, CZ, C4</strong></span>
            <span className="inline-flex items-center gap-1 text-[#3A5244] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244] animate-ping" />
              Active
            </span>
          </div>
        </div>

        {/* Conduit 1: Neural Decoding Vector (Col 4) */}
        <div className="md:col-span-1 flex flex-col items-center justify-center py-1">
          <div className="flex md:flex-col items-center gap-1 text-[#3A5244]">
            <span className="text-[9px] font-semibold tracking-wider uppercase text-[#AEADA4] hidden md:block">CSP</span>
            <div className="w-7 h-7 rounded-full bg-[#EDECE6] border border-[#D9D8D0] flex items-center justify-center shadow-xs">
              <ArrowRight className="w-3.5 h-3.5 text-[#3A5244]" />
            </div>
            <span className="text-[9px] font-semibold tracking-wider uppercase text-[#AEADA4] hidden md:block">LDA</span>
          </div>
        </div>

        {/* Stage 2: Decoded Goal Intention (Cols 5-7) */}
        <div className="md:col-span-3 p-3 rounded-xl border bg-[#FAF9F5] border-[#D9D8D0] transition-all">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5 text-[#2563EB]" />
              <span className="text-xs font-bold text-[#1C1C1A]">2. Decoded Intention</span>
            </div>
            <span
              className="text-[10px] font-bold px-2.5 py-0.5 rounded-full border shadow-2xs"
              style={{ backgroundColor: targetBg, color: targetColor, borderColor: `${targetColor}40` }}
            >
              {targetLabel}
            </span>
          </div>

          <p className="text-[11px] text-[#6B6B66] leading-snug mb-2">
            Spatial filter projects cortical preparatory state to target hemisphere.
          </p>

          <div className="flex items-center justify-between text-[10px] text-[#6B6B66] pt-1.5 border-t border-[#D9D8D0]/60">
            <span>Confidence: <strong className="text-[#1C1C1A]">{confidence}%</strong></span>
            <span className="text-[#3A5244] font-semibold">Pre-movement Resolved</span>
          </div>
        </div>

        {/* Conduit 2: Motor Execution Vector (Col 8) */}
        <div className="md:col-span-1 flex flex-col items-center justify-center py-1">
          <div className="flex md:flex-col items-center gap-1 text-[#3A5244]">
            <span className="text-[9px] font-semibold tracking-wider uppercase text-[#AEADA4] hidden md:block">Ridge</span>
            <div className="w-7 h-7 rounded-full bg-[#EDECE6] border border-[#D9D8D0] flex items-center justify-center shadow-xs">
              <ArrowRight className="w-3.5 h-3.5 text-[#3A5244]" />
            </div>
            <span className="text-[9px] font-semibold tracking-wider uppercase text-[#AEADA4] hidden md:block">+200ms</span>
          </div>
        </div>

        {/* Stage 3: 3D Kinematic Actuation (Cols 9-11) */}
        <div className={`md:col-span-3 p-3 rounded-xl border transition-all ${
          !isPrefix ? 'bg-[#FAF9F5] border-[#3A5244]/40 shadow-xs ring-1 ring-[#3A5244]/20' : 'bg-[#FAF9F5] border-[#D9D8D0]'
        }`}>
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Crosshair className="w-3.5 h-3.5 text-[#3A5244]" />
              <span className="text-xs font-bold text-[#1C1C1A]">3. Physical Movement</span>
            </div>
            <span className="text-[9px] font-semibold px-2 py-0.2 rounded-full bg-[#EDECE6] text-[#1C1C1A]">
              {isPrefix ? 'Prefix Window' : 'Trajectory Forecast'}
            </span>
          </div>

          <p className="text-[11px] text-[#6B6B66] leading-snug mb-2">
            Leap Motion coordinates actuated in 3D workspace toward decoded goal.
          </p>

          <div className="flex items-center justify-between text-[10px] text-[#6B6B66] pt-1.5 border-t border-[#D9D8D0]/60">
            <span>
              Pos: <strong className="text-[#1C1C1A]">{currentHandPos.x.toFixed(0)}, {currentHandPos.y.toFixed(0)}, {currentHandPos.z.toFixed(0)} mm</strong>
            </span>
            <span className="font-semibold text-[#1C1C1A]">{currentHandPos.speed} mm/s</span>
          </div>
        </div>
      </div>
    </div>
  );
}
