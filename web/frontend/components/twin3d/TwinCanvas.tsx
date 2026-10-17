'use client';

import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import Workspace3D from './Workspace3D';
import { TrialData } from '@/lib/types';
import { AlertCircle, MousePointer } from 'lucide-react';

export interface TwinCanvasProps {
  trialData: TrialData | null;
  currentTime: number;
  showGrid: boolean;
  showAxes: boolean;
  showForecast: boolean;
  resetCameraSignal: number;
}

function Fallback2DView({
  trialData,
  currentTime,
}: {
  trialData: TrialData | null;
  currentTime: number;
}) {
  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-8 text-center bg-[#FAF9F6] rounded border border-[#DCDDD5]">
      <div className="w-10 h-10 rounded-full bg-[#FEF3C7] border border-[#D97706]/30 flex items-center justify-center text-[#92400E] mb-3">
        <AlertCircle className="w-5 h-5" />
      </div>
      <h3 className="font-serif text-base font-semibold text-[#252722] mb-1">
        WebGL Hardware Acceleration Unavailable
      </h3>
      <p className="max-w-md text-xs text-[#6F716B] mb-4 leading-relaxed">
        Your browser or display environment does not support WebGL 3D rendering.
      </p>

      {trialData && (
        <div className="w-full max-w-sm p-3 rounded bg-[#FFFFFF] border border-[#DCDDD5] text-left font-mono text-xs space-y-1 text-[#252722]">
          <div className="text-[#6F716B]">Trial: {trialData.trial_id}</div>
          <div className="text-[#6F716B]">Time: {currentTime.toFixed(0)} ms</div>
          <div>Goal: {String(trialData.metadata?.target_goal || 'left')}</div>
          <div>Endpoint: {JSON.stringify(trialData.predicted_endpoint || [-188, 150, 53])}</div>
        </div>
      )}
    </div>
  );
}

export default function TwinCanvas({
  trialData,
  currentTime,
  showGrid,
  showAxes,
  showForecast,
  resetCameraSignal,
}: TwinCanvasProps) {
  const [hasWebGL] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true;
    try {
      const canvas = document.createElement('canvas');
      return Boolean(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch {
      return false;
    }
  });

  if (hasWebGL === false) {
    return <Fallback2DView trialData={trialData} currentTime={currentTime} />;
  }

  return (
    <div className="relative w-full h-full min-h-[460px] bg-[#F5F8F5] rounded-2xl border border-[#D9D8D0] overflow-hidden select-none shadow-sm">
      <Canvas
        camera={{ position: [0, 3, 4], fov: 46, near: 0.1, far: 50 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        className="w-full h-full"
      >
        <Workspace3D
          trialData={trialData}
          currentTime={currentTime}
          showGrid={showGrid}
          showAxes={showAxes}
          showForecast={showForecast}
          resetCameraSignal={resetCameraSignal}
        />
      </Canvas>

      {/* Floating Info Overlays */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10 text-xs">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#FFFFFF]/90 backdrop-blur-md text-[#1C1C1A] border border-[#D9D8D0] shadow-xs font-body text-[11px] font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244]" />
            <span>Leap Motion Space (1 mm = 0.01 units)</span>
          </span>
          {trialData?.trial_id && (
            <span className="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-full bg-[#FFFFFF]/90 backdrop-blur-md text-[#6B6B66] border border-[#D9D8D0] shadow-xs font-body text-[11px]">
              {trialData.trial_id}
            </span>
          )}
        </div>

        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#FFFFFF]/90 backdrop-blur-md text-[#6B6B66] border border-[#D9D8D0] shadow-xs text-[11px] font-body">
          <MousePointer className="w-3 h-3 text-[#3A5244]" />
          <span>Rotate: Left Drag &bull; Pan: Right Drag &bull; Zoom: Scroll</span>
        </div>
      </div>
    </div>
  );
}
