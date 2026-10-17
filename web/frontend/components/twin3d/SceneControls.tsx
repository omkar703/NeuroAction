'use client';

import React, { useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Grid,
  Compass,
  Sparkles,
  Info,
  ChevronDown,
  ChevronUp,
  FastForward,
  SkipBack,
  SkipForward,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SceneControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  currentTime: number;
  maxDuration: number;
  onSeek: (timeMs: number) => void;
  showGrid: boolean;
  onToggleGrid: () => void;
  showAxes: boolean;
  onToggleAxes: () => void;
  showForecast: boolean;
  onToggleForecast: () => void;
  onResetCamera: () => void;
}

const SPEED_OPTIONS = [0.5, 1.0, 2.0];

export default function SceneControls({
  isPlaying,
  onTogglePlay,
  speed,
  onSpeedChange,
  currentTime,
  maxDuration,
  onSeek,
  showGrid,
  onToggleGrid,
  showAxes,
  onToggleAxes,
  showForecast,
  onToggleForecast,
  onResetCamera,
}: SceneControlsProps) {
  const [isNoteOpen, setIsNoteOpen] = useState(false);

  const durationSafe = Math.max(maxDuration, 100);

  return (
    <div className="flex flex-col gap-3.5 p-4 rounded bg-[#FAF9F6] border border-[#DCDDD5]">
      {/* Upper Bar: Playback Controls & Scrubber */}
      <div className="flex flex-col gap-2.5">
        {/* Timeline Scrubber & Timestamps */}
        <div className="flex items-center justify-between text-xs font-mono text-[#6F716B]">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#455D4A]" />
            <span className="font-semibold text-[#252722]">{(currentTime / 1000).toFixed(2)}s</span>
            <span>({currentTime.toFixed(0)} ms)</span>
          </div>
          <div>
            {(durationSafe / 1000).toFixed(2)}s ({durationSafe.toFixed(0)} ms)
          </div>
        </div>

        {/* Slider input */}
        <div className="relative flex items-center w-full">
          <input
            type="range"
            min={0}
            max={durationSafe}
            step={10}
            value={currentTime}
            onChange={(e) => onSeek(parseFloat(e.target.value))}
            aria-label="Timeline scrubber"
            className="w-full h-1.5 rounded appearance-none cursor-pointer bg-[#EAE9E2] accent-[#455D4A] focus:outline-none"
          />
        </div>

        {/* Buttons Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          {/* Left Group */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={onTogglePlay}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-[#455D4A] hover:bg-[#364A3B] text-white transition-colors cursor-pointer"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Play</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={() => onSeek(0)}
              title="Jump to start"
              className="p-1.5 rounded bg-[#F0EFE9] hover:bg-[#EAE9E2] text-[#252722] border border-[#DCDDD5] transition-colors"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={() => onSeek(durationSafe)}
              title="Jump to endpoint"
              className="p-1.5 rounded bg-[#F0EFE9] hover:bg-[#EAE9E2] text-[#252722] border border-[#DCDDD5] transition-colors"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center gap-1">
            <FastForward className="w-3.5 h-3.5 text-[#6F716B] mr-0.5" />
            {SPEED_OPTIONS.map((rate) => (
              <button
                key={rate}
                type="button"
                onClick={() => onSpeedChange(rate)}
                className={cn(
                  'px-2 py-0.5 text-xs font-mono rounded transition-colors',
                  speed === rate
                    ? 'bg-[#455D4A] text-white font-medium'
                    : 'bg-[#F0EFE9] text-[#6F716B] hover:text-[#252722] border border-[#DCDDD5]'
                )}
              >
                {rate}x
              </button>
            ))}
          </div>

          {/* View Toggles & Reset Camera */}
          <div className="flex items-center gap-1.5 text-xs">
            <button
              type="button"
              onClick={onToggleGrid}
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded border transition-colors',
                showGrid
                  ? 'bg-[#F0EFE9] text-[#252722] border-[#455D4A] font-medium'
                  : 'bg-[#FAF9F6] text-[#6F716B] border-[#DCDDD5]'
              )}
            >
              <Grid className="w-3 h-3" />
              <span>Grid</span>
            </button>

            <button
              type="button"
              onClick={onToggleAxes}
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded border transition-colors',
                showAxes
                  ? 'bg-[#F0EFE9] text-[#252722] border-[#455D4A] font-medium'
                  : 'bg-[#FAF9F6] text-[#6F716B] border-[#DCDDD5]'
              )}
            >
              <Compass className="w-3 h-3" />
              <span>Axes</span>
            </button>

            <button
              type="button"
              onClick={onToggleForecast}
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded border transition-colors',
                showForecast
                  ? 'bg-[#F0EFE9] text-[#252722] border-[#455D4A] font-medium'
                  : 'bg-[#FAF9F6] text-[#6F716B] border-[#DCDDD5]'
              )}
            >
              <Sparkles className="w-3 h-3 text-[#455D4A]" />
              <span>Forecast</span>
            </button>

            <button
              type="button"
              onClick={onResetCamera}
              title="Reset 3D camera"
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#F0EFE9] hover:bg-[#EAE9E2] text-[#252722] border border-[#DCDDD5] transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset View</span>
            </button>
          </div>
        </div>
      </div>

      {/* Coordinate Note Disclosure */}
      <div className="pt-2 border-t border-[#DCDDD5]">
        <button
          type="button"
          onClick={() => setIsNoteOpen(!isNoteOpen)}
          className="flex items-center justify-between w-full text-left text-xs text-[#6F716B] hover:text-[#252722] transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-[#455D4A]" />
            <span className="font-medium text-[#252722]">
              Coordinate Visualization Reference
            </span>
          </div>
          {isNoteOpen ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </button>

        {isNoteOpen && (
          <div className="mt-2 p-3 rounded bg-[#F0EFE9] border border-[#DCDDD5] text-xs text-[#6F716B] space-y-1 leading-relaxed">
            <p className="font-medium text-[#252722]">
              Visualizes recorded Leap Motion optical telemetry in desk-centric coordinates (mm).
            </p>
            <p className="text-[11px]">
              Three.js scene units: 1 mm = 0.01 units. Left Target: (-188, 150, 53) mm. Right Target: (188, 149, 51) mm.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
