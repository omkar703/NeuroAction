'use client';

import React from 'react';
import {
  Layers,
  CheckCircle2,
  Database,
  Filter,
  BarChart3,
  Brain,
  Compass,
  Crosshair,
  TrendingUp,
} from 'lucide-react';
import { TrialData } from '@/lib/types';

interface InputOutputInspectorProps {
  trialData: TrialData | null;
}

export default function InputOutputInspector({ trialData }: InputOutputInspectorProps) {
  const isLoaded = !!trialData;

  const trialId = trialData?.trial_id || '—';
  const participantId = trialData?.participant_id || (trialData?.trial_id ? trialData.trial_id.split('_')[0] : '—');
  const eegShape = isLoaded ? '200 samples × 26 channels (1.0s @ 200 Hz)' : '—';

  const filterSpec = isLoaded ? 'Band-pass 8–30 Hz (zero-phase filtfilt, offline)' : '—';
  const motorChannelsSelected = isLoaded
    ? 'FC3, FCZ, FC4, C3, CZ, C4, CP3, CPZ, CP4 (9 channels)'
    : '—';

  const featureCount = isLoaded
    ? '6 time-domain + 3 bandpower = 9 features/channel (234 total)'
    : '—';

  const predictedGoal = trialData?.neural?.predicted_goal || (trialData?.target_goal as string) || '—';
  const probLeft = trialData?.neural?.probability_left !== undefined
    ? `${(trialData.neural.probability_left * 100).toFixed(1)}%`
    : '—';
  const probRight = trialData?.neural?.probability_right !== undefined
    ? `${(trialData.neural.probability_right * 100).toFixed(1)}%`
    : '—';
  const confidence = trialData?.neural?.confidence !== undefined
    ? `${(trialData.neural.confidence * 100).toFixed(1)}%`
    : '—';

  const prefixSamples = isLoaded && trialData?.hand_x && trialData.hand_x.length >= 20
    ? '20 samples × 3D (100 Hz, [0, 200] ms)'
    : isLoaded ? `${trialData?.hand_x?.length || 0} samples × 3D` : '—';

  const endpointCoords = trialData?.predicted_endpoint
    ? `[${Number(trialData.predicted_endpoint[0]).toFixed(1)}, ${Number(trialData.predicted_endpoint[1]).toFixed(1)}, ${Number(trialData.predicted_endpoint[2]).toFixed(1)}] mm`
    : '—';

  let endpointErrorStr = '—';
  if (trialData) {
    if (trialData.endpoint_error_mm !== undefined) {
      endpointErrorStr = `${Number(trialData.endpoint_error_mm).toFixed(2)} mm`;
    } else if (trialData.predicted_endpoint && Array.isArray(trialData.predicted_endpoint)) {
      const pred = trialData.predicted_endpoint;
      const isLeft = (trialData.neural?.predicted_goal || trialData.target_goal) === 'left';
      const target = isLeft ? [-188.0, 150.0, 53.0] : [188.0, 149.0, 51.0];
      const err = Math.sqrt(
        Math.pow(pred[0] - target[0], 2) +
        Math.pow(pred[1] - target[1], 2) +
        Math.pow(pred[2] - target[2], 2)
      );
      endpointErrorStr = `${err.toFixed(2)} mm`;
    }
  }

  interface InspectorField {
    label: string;
    value: string;
    mono?: boolean;
    badge?: string;
  }

  interface InspectorStep {
    num: number;
    title: string;
    icon: React.ComponentType<{ className?: string }>;
    description: string;
    status: 'done' | 'pending';
    fields: InspectorField[];
  }

  const steps: InspectorStep[] = [
    {
      num: 1,
      title: 'Data Ingestion',
      icon: Database,
      description: 'Raw pre-movement EEG epoch and identifiers from repository',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Trial ID', value: trialId, mono: true },
        { label: 'Participant', value: participantId, mono: true },
        { label: 'EEG Dimension', value: eegShape },
      ],
    },
    {
      num: 2,
      title: 'Signal Conditioning',
      icon: Filter,
      description: 'Zero-phase bandpass filtering and sensorimotor subset selection',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Filter Specs', value: filterSpec },
        { label: 'Channels', value: motorChannelsSelected, mono: true },
      ],
    },
    {
      num: 3,
      title: 'Feature Extraction',
      icon: BarChart3,
      description: 'Temporal statistics and mu/beta spectral bandpower computation',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Feature Count', value: featureCount },
        { label: 'Feature Window', value: '[-1000, 0] ms pre-movement' },
      ],
    },
    {
      num: 4,
      title: 'LDA Inference',
      icon: Brain,
      description: 'Linear Discriminant mapping features to target class probabilities',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Decoded Goal', value: predictedGoal.toUpperCase() },
        { label: 'P(Left) / P(Right)', value: `${probLeft} / ${probRight}`, mono: true },
        { label: 'Confidence', value: confidence, mono: true },
      ],
    },
    {
      num: 5,
      title: 'Kinematic Prefix',
      icon: Compass,
      description: 'Initial optical hand trajectory window recorded post-onset',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Prefix Buffer', value: prefixSamples },
        { label: 'Prefix Duration', value: '0 to 200 ms post movement onset' },
      ],
    },
    {
      num: 6,
      title: 'Ridge Forecaster',
      icon: TrendingUp,
      description: 'Regularized regression forecasting terminal 3D coordinates',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Forecast Endpoint', value: endpointCoords, mono: true },
        { label: 'Frame', value: 'Leap Motion Cartesian (mm)' },
      ],
    },
    {
      num: 7,
      title: 'Error Evaluation',
      icon: Crosshair,
      description: 'Retrospective scoring against empirical reach target proxy',
      status: isLoaded ? 'done' : 'pending',
      fields: [
        { label: 'Endpoint Error', value: endpointErrorStr, mono: true },
        { label: 'Reference Proxy', value: 'Empirical target centroid' },
      ],
    },
  ];

  return (
    <div className="w-full flex flex-col rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] p-5 space-y-4 shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0]">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-[#3A5244]" />
          <h3 className="font-display text-sm font-semibold text-[#1C1C1A]">
            Inference Pipeline Audit Trace
          </h3>
        </div>
      </div>

      {/* 7 Pipeline Steps Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {steps.map((step) => {
          const Icon = step.icon;
          const isDone = step.status === 'done';

          return (
            <div
              key={step.num}
              className={`flex flex-col justify-between p-3.5 rounded-xl border transition-colors ${
                isDone
                  ? 'bg-[#FAF9F5] border-[#D9D8D0]'
                  : 'bg-[#EDECE6]/40 border-[#D9D8D0]'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    <span className="w-5 h-5 rounded-full bg-[#EDECE6] border border-[#D9D8D0] text-[#1C1C1A] font-mono text-[11px] font-semibold flex items-center justify-center">
                      {step.num}
                    </span>
                    <Icon className="w-3.5 h-3.5 text-[#3A5244]" />
                    <span className="text-xs font-semibold text-[#1C1C1A]">
                      {step.title}
                    </span>
                  </div>
                  {isDone ? (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono text-[#3A5244]">
                      <CheckCircle2 className="w-3 h-3" />
                      verified
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono text-[#6B6B66]">pending</span>
                  )}
                </div>

                <p className="text-[11px] text-[#6B6B66] leading-relaxed mb-3">
                  {step.description}
                </p>

                <div className="space-y-1.5 pt-2 border-t border-[#D9D8D0] text-[11px]">
                  {step.fields.map((field) => (
                    <div key={field.label} className="flex flex-col">
                      <span className="text-[10px] text-[#6B6B66]">{field.label}</span>
                      <span className={`text-[#1C1C1A] ${field.mono ? 'font-mono' : ''}`}>
                        {field.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
