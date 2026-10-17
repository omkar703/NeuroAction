'use client';

import React from 'react';
import {
  Waves,
  Sliders,
  Cpu,
  Target,
  ShieldAlert,
  CheckCircle2,
  Loader2,
  Clock,
  AlertCircle,
} from 'lucide-react';

export type PipelineStageStatus = 'idle' | 'running' | 'done' | 'error';

export interface PipelineVizProps {
  isLoading?: boolean;
  isLoaded?: boolean;
  isLiveInference?: boolean;
  activeStage?: number; // 1 to 5
  stageStatuses?: {
    input?: PipelineStageStatus;
    preprocessing?: PipelineStageStatus;
    inference?: PipelineStageStatus;
    prediction?: PipelineStageStatus;
    evaluation?: PipelineStageStatus;
  };
}

interface StageDefinition {
  id: string;
  step: string;
  label: string;
  shortDesc: string;
  icon: React.ComponentType<{ className?: string }>;
}

const STAGES: StageDefinition[] = [
  {
    id: 'input',
    step: '01',
    label: 'EEG & Kinematics Input',
    shortDesc: 'Pre-movement EEG [-1000, 0] ms & 200 ms kinematic prefix',
    icon: Waves,
  },
  {
    id: 'preprocessing',
    step: '02',
    label: 'Signal Conditioning',
    shortDesc: 'Zero-phase bandpass, spatial CSP decomposition & calibration',
    icon: Sliders,
  },
  {
    id: 'inference',
    step: '03',
    label: 'Model Inference',
    shortDesc: 'LOSO Ridge Trajectory Forecaster & LDA Intention Decoder',
    icon: Cpu,
  },
  {
    id: 'prediction',
    step: '04',
    label: 'Trajectory Forecast',
    shortDesc: 'Predicted 3D endpoint coordinate and decoded target goal',
    icon: Target,
  },
  {
    id: 'evaluation',
    step: '05',
    label: 'Congruence & Error',
    shortDesc: 'Spatial divergence tracking and Euclidean endpoint RMSE',
    icon: ShieldAlert,
  },
];

export default function PipelineViz({
  isLoading = false,
  isLoaded = false,
  isLiveInference = false,
  activeStage,
  stageStatuses,
}: PipelineVizProps) {
  const getStageStatus = (stageIdx: number, stageId: string): PipelineStageStatus => {
    if (stageStatuses && (stageStatuses as Record<string, PipelineStageStatus>)[stageId]) {
      return (stageStatuses as Record<string, PipelineStageStatus>)[stageId];
    }

    if (isLoading) {
      if (stageIdx <= 2) return 'running';
      return 'idle';
    }

    if (isLoaded) {
      return 'done';
    }

    if (activeStage !== undefined) {
      if (stageIdx + 1 < activeStage) return 'done';
      if (stageIdx + 1 === activeStage) return 'running';
      return 'idle';
    }

    return 'idle';
  };

  const renderStatusBadge = (status: PipelineStageStatus) => {
    switch (status) {
      case 'running':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-body px-2 py-0.5 rounded-full bg-[#EDECE6] text-[#2563EB] border border-[#D9D8D0]">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>RUNNING</span>
          </span>
        );
      case 'done':
        return (
          <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded-full bg-[#E4EDE6] text-[#3A5244] border border-[#3A5244]/30">
            <CheckCircle2 className="w-2.5 h-2.5" />
          </span>
        );
      case 'error':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-body px-2 py-0.5 rounded-full bg-[#FEF3C7] text-[#92400E] border border-[#D97706]/30">
            <AlertCircle className="w-2.5 h-2.5" />
            <span>ERROR</span>
          </span>
        );
      case 'idle':
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-body px-2 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] border border-[#D9D8D0]">
            <Clock className="w-2.5 h-2.5" />
            <span>IDLE</span>
          </span>
        );
    }
  };

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-6 space-y-4 shadow-sm font-body">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h2 className="font-display text-sm font-bold text-[#1C1C1A]">
              Processing Pipeline
            </h2>
            <p className="text-[11px] text-[#6B6B66] font-body">Multimodal execution stages</p>
          </div>
        </div>

        <span className="text-[10px] font-body px-2.5 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] border border-[#D9D8D0]">
          {isLiveInference ? 'Live Mode' : 'Deterministic Replay'}
        </span>
      </div>

      {/* Vertical Pipeline Stages */}
      <div className="space-y-2.5">
        {STAGES.map((stage, idx) => {
          const status = getStageStatus(idx, stage.id);
          const Icon = stage.icon;
          const isDone = status === 'done';
          const isRunning = status === 'running';

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-3 p-3 rounded-xl border transition-colors font-body ${
                isRunning
                  ? 'bg-[#FAF9F5] border-[#3A5244] shadow-sm'
                  : isDone
                  ? 'bg-[#FAF9F5] border-[#D9D8D0]'
                  : 'bg-[#EDECE6]/40 border-[#D9D8D0]'
              }`}
            >
              <div
                className={`w-7 h-7 rounded-full border flex items-center justify-center shrink-0 mt-0.5 ${
                  isDone
                    ? 'border-[#3A5244]/30 bg-[#E4EDE6] text-[#3A5244]'
                    : isRunning
                    ? 'border-[#3A5244] bg-[#FFFFFF] text-[#3A5244]'
                    : 'border-[#D9D8D0] bg-[#EDECE6] text-[#6B6B66]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-1 mb-0.5">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-body font-bold text-[#AEADA4]">
                      {stage.step}
                    </span>
                    <h3 className="text-xs font-bold text-[#1C1C1A] truncate font-display">
                      {stage.label}
                    </h3>
                  </div>
                  <div>{renderStatusBadge(status)}</div>
                </div>

                <p className="text-[11px] text-[#6B6B66] leading-tight font-body">
                  {stage.shortDesc}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
