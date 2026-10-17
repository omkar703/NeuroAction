'use client';

import React, { useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Info,
  Clock,
  HelpCircle,
} from 'lucide-react';

export interface PlaybackControlsProps {
  currentTime: number;
  maxTime: number;
  minTime?: number;
  stepMs?: number;
  isPlaying: boolean;
  playbackSpeed: number;
  onPlayPause: () => void;
  onStepForward: () => void;
  onRestart: () => void;
  onSpeedChange: (speed: number) => void;
  onSeek: (time: number) => void;
  disabled?: boolean;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

export default function PlaybackControls({
  currentTime,
  maxTime,
  minTime = 0,
  stepMs = 10,
  isPlaying,
  playbackSpeed,
  onPlayPause,
  onStepForward,
  onRestart,
  onSpeedChange,
  onSeek,
  disabled = false,
}: PlaybackControlsProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-5 space-y-4 shadow-sm font-body">
      {/* Top Row: Replay Indicator + Current Time Display */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        {/* Offline Replay Badge with Info Tooltip */}
        <div className="relative flex items-center gap-1.5">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#EDECE6] text-[#1C1C1A] border border-[#D9D8D0] text-xs font-body font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244]" />
            <span>Deterministic Replay</span>
            <button
              type="button"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              onClick={() => setShowTooltip((prev) => !prev)}
              aria-label="Replay Information"
              className="text-[#6B6B66] hover:text-[#1C1C1A] focus:outline-none ml-0.5 cursor-pointer"
            >
              <Info className="w-3 h-3" />
            </button>
          </div>

          {/* Info Tooltip Bubble */}
          {showTooltip && (
            <div className="absolute left-0 top-full mt-2 w-72 p-4 bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-xl z-50 text-xs text-[#6B6B66] leading-relaxed font-body">
              <div className="font-bold text-[#1C1C1A] mb-1 flex items-center gap-1 font-display">
                <HelpCircle className="w-3.5 h-3.5 text-[#3A5244]" />
                <span>Deterministic Scientific Replay</span>
              </div>
              <p>
                Visualizes real recorded kinematics and EEG from OpenNeuro <strong className="text-[#1C1C1A]">ds006840</strong> replayed through the trained Ridge forecaster and LDA decoder. It is not a live streaming acquisition.
              </p>
            </div>
          )}
        </div>

        {/* Time Counter */}
        <div className="flex items-center gap-2 bg-[#EDECE6] border border-[#D9D8D0] px-3 py-1 rounded-full text-xs font-body text-[#1C1C1A]">
          <Clock className="w-3.5 h-3.5 text-[#6B6B66]" />
          <div>
            <span className="font-bold font-display">{Math.round(currentTime)}</span>
            <span className="text-[#6B6B66]"> / {Math.round(maxTime)} ms</span>
          </div>
        </div>
      </div>

      {/* Progress Track / Slider */}
      <div className="relative flex items-center w-full">
        <input
          type="range"
          min={minTime}
          max={maxTime}
          step={stepMs}
          value={currentTime}
          disabled={disabled}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="w-full h-1.5 bg-[#EDECE6] rounded-full appearance-none cursor-pointer accent-[#3A5244] disabled:opacity-40"
        />
      </div>

      {/* Bottom Controls Row: Play/Pause, Step, Restart, Speed */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            type="button"
            onClick={onPlayPause}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] hover:bg-[#3A5244] text-[#F5F3EE] transition-colors disabled:opacity-40 cursor-pointer"
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

          {/* Step Forward */}
          <button
            type="button"
            onClick={onStepForward}
            disabled={disabled || isPlaying}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full text-xs font-semibold font-body bg-[#FAF9F5] hover:bg-[#EDECE6] border border-[#D9D8D0] text-[#1C1C1A] transition-colors disabled:opacity-40 cursor-pointer"
            title="Step forward +10ms"
          >
            <StepForward className="w-3.5 h-3.5" />
            <span>Step</span>
          </button>

          {/* Restart */}
          <button
            type="button"
            onClick={onRestart}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full text-xs font-semibold font-body bg-[#FAF9F5] hover:bg-[#EDECE6] border border-[#D9D8D0] text-[#1C1C1A] transition-colors disabled:opacity-40 cursor-pointer"
            title="Restart playback to t=0ms"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1 text-xs font-body">
          <span className="text-[11px] text-[#AEADA4] mr-1 uppercase font-semibold">Speed:</span>
          {SPEED_OPTIONS.map((rate) => (
            <button
              key={rate}
              type="button"
              disabled={disabled}
              onClick={() => onSpeedChange(rate)}
              className={`px-2.5 py-1 rounded-full text-[11px] font-semibold transition-colors cursor-pointer ${
                playbackSpeed === rate
                  ? 'bg-[#1C1C1A] text-[#F5F3EE]'
                  : 'bg-[#FAF9F5] text-[#6B6B66] hover:text-[#1C1C1A] border border-[#D9D8D0]'
              }`}
            >
              {rate}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
