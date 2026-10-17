'use client';

import React, { useState } from 'react';
import {
  FileText,
  Copy,
  Check,
  CircleDot,
  ArrowLeft,
  ArrowRight,
  Brain,
  Database,
  Radio,
} from 'lucide-react';
import { NeuralResult } from '@/lib/types';

export interface MetadataCardProps {
  trialId: string;
  participantId: string;
  congruency?: string;
  targetGoal?: string;
  physicalHandDirection?: string;
  ballColor?: string;
  neural?: NeuralResult;
  dataSource?: string;
  forecastGenerationTimeMs?: number;
}

export default function MetadataCard({
  trialId,
  participantId,
  congruency = 'congruent',
  targetGoal = 'left',
  physicalHandDirection = 'left',
  ballColor = 'red',
  neural,
  dataSource = 'recorded_replay',
  forecastGenerationTimeMs,
}: MetadataCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyId = () => {
    if (!trialId) return;
    navigator.clipboard?.writeText(trialId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isCongruent = congruency.toLowerCase() === 'congruent';
  const isLive = dataSource === 'live_inference';

  const probLeft = neural?.probability_left !== undefined ? neural.probability_left : (neural?.predicted_goal === 'left' ? 0.84 : 0.16);
  const probRight = neural?.probability_right !== undefined ? neural.probability_right : (neural?.predicted_goal === 'right' ? 0.84 : 0.16);
  const confidencePct = neural?.confidence !== undefined
    ? (neural.confidence * 100).toFixed(1)
    : (Math.max(probLeft, probRight) * 100).toFixed(1);
  const predictedGoal = neural?.predicted_goal || (probLeft >= probRight ? 'left' : 'right');

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-6 shadow-sm font-body space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D9D8D0] pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-[#EDECE6] text-[#3A5244] flex items-center justify-center shrink-0">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-display text-base font-bold text-[#1C1C1A]">
              Trial Manifest &amp; Provenance
            </h2>
            <p className="text-xs text-[#6B6B66] font-body">
              Audit metadata, experimental conditions, and pre-movement neural intention classification
            </p>
          </div>
        </div>

        {/* Data Source & Latency Badges */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          {forecastGenerationTimeMs !== undefined && (
            <span className="text-[11px] font-body text-[#6B6B66] bg-[#EDECE6] border border-[#D9D8D0] px-3 py-1 rounded-full font-semibold">
              Latency: {forecastGenerationTimeMs} ms
            </span>
          )}
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-body bg-[#EDECE6] text-[#1C1C1A] border border-[#D9D8D0] font-semibold">
            {isLive ? (
              <>
                <Radio className="w-3.5 h-3.5 text-[#3A5244]" />
                <span>Live Inference</span>
              </>
            ) : (
              <>
                <Database className="w-3.5 h-3.5 text-[#6B6B66]" />
                <span>Recorded Replay</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Wide 3-Column Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-stretch">

        {/* Column 1: Trial Identity & Origin (3 of 12 cols) */}
        <div className="md:col-span-12 lg:col-span-4 p-4 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col justify-between space-y-3">
          <div>
            <span className="text-[10px] font-body uppercase tracking-[0.2em] text-[#AEADA4] font-semibold block mb-2">
              Trial Identification
            </span>
            <div className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0]">
              <span className="font-display text-xs text-[#1C1C1A] font-bold truncate" title={trialId}>
                {trialId || 'sub-01_run-01_L1_tr00'}
              </span>
              <button
                onClick={handleCopyId}
                className="p-1.5 rounded-full bg-[#EDECE6] hover:bg-[#D9D8D0] text-[#6B6B66] hover:text-[#1C1C1A] transition-colors shrink-0 cursor-pointer"
                title="Copy Trial ID"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-[#3A5244]" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs pt-1">
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] block font-body">Subject</span>
              <span className="font-display font-bold text-[#1C1C1A] text-xs mt-0.5 block">
                {participantId || 'sub-01'}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] block font-body">Evaluation</span>
              <span className="font-display font-bold text-[#3A5244] text-xs mt-0.5 block">
                LOSO Fold
              </span>
            </div>
          </div>
        </div>

        {/* Column 2: Experimental Conditions 4-Item Grid (4 of 12 cols) */}
        <div className="md:col-span-12 lg:col-span-4 p-4 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col justify-between space-y-3">
          <span className="text-[10px] font-body uppercase tracking-[0.2em] text-[#AEADA4] font-semibold block">
            Experimental Protocol Conditions
          </span>

          <div className="grid grid-cols-2 gap-2 text-xs flex-1">
            {/* Protocol Condition */}
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col justify-between">
              <span className="text-[10px] uppercase font-body text-[#AEADA4] font-semibold">
                Congruence
              </span>
              <div className="mt-1 font-bold font-display capitalize text-[#1C1C1A] flex items-center justify-between">
                <span className="flex items-center gap-1 text-xs">
                  <CircleDot className={`w-3 h-3 ${isCongruent ? 'text-[#3A5244]' : 'text-[#B45309]'}`} />
                  <span>{congruency}</span>
                </span>
                <span className={`text-[9px] font-body px-1.5 py-0.5 rounded-full ${
                  isCongruent ? 'bg-[#3A5244]/10 text-[#3A5244]' : 'bg-[#B45309]/10 text-[#B45309]'
                }`}>
                  {isCongruent ? 'Aligned' : 'Conflict'}
                </span>
              </div>
            </div>

            {/* Stimulus Color */}
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col justify-between">
              <span className="text-[10px] uppercase font-body text-[#AEADA4] font-semibold">
                Stimulus Cue
              </span>
              <span className="font-bold font-display capitalize text-[#1C1C1A] text-xs mt-1">
                {ballColor} Ball
              </span>
            </div>

            {/* Instructed Target */}
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col justify-between">
              <span className="text-[10px] uppercase font-body text-[#AEADA4] font-semibold">
                Instructed Target
              </span>
              <div className="flex items-center gap-1 mt-1 font-bold font-display text-xs text-[#2563EB]">
                {targetGoal === 'left' ? (
                  <>
                    <ArrowLeft className="w-3 h-3" />
                    <span>Left Target</span>
                  </>
                ) : (
                  <>
                    <span>Right Target</span>
                    <ArrowRight className="w-3 h-3" />
                  </>
                )}
              </div>
            </div>

            {/* Kinematic Direction */}
            <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col justify-between">
              <span className="text-[10px] uppercase font-body text-[#AEADA4] font-semibold">
                Limb Motion
              </span>
              <div className="flex items-center gap-1 mt-1 font-bold font-display text-xs text-[#2E6B47]">
                {physicalHandDirection === 'left' ? (
                  <>
                    <ArrowLeft className="w-3 h-3" />
                    <span>Left Reach</span>
                  </>
                ) : (
                  <>
                    <span>Right Reach</span>
                    <ArrowRight className="w-3 h-3" />
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Column 3: Pre-Movement Neural Intention Decoder (5 of 12 cols) */}
        <div className="md:col-span-12 lg:col-span-4 p-4 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs font-bold text-[#1C1C1A] font-display">
              <Brain className="w-4 h-4 text-[#3A5244]" />
              <span>Pre-Movement Neural Decoder</span>
            </div>
            <span className="text-[10px] font-body text-[#6B6B66] px-2.5 py-0.5 rounded-full bg-[#EDECE6] border border-[#D9D8D0] font-semibold">
              LDA Classifier
            </span>
          </div>

          {/* Goal & Confidence */}
          <div className="p-3 rounded-lg bg-[#FFFFFF] border border-[#D9D8D0] space-y-2">
            <div className="flex items-center justify-between text-xs font-body">
              <div className="flex items-center gap-1.5">
                <span className="text-[#6B6B66]">Decoded Intention:</span>
                <span className="font-bold font-display text-[#1C1C1A] capitalize flex items-center gap-1">
                  {predictedGoal === 'left' ? (
                    <>
                      <ArrowLeft className="w-3 h-3 text-[#2563EB]" />
                      <span>Left</span>
                    </>
                  ) : (
                    <>
                      <span>Right</span>
                      <ArrowRight className="w-3 h-3 text-[#2563EB]" />
                    </>
                  )}
                </span>
              </div>

              <div>
                <span className="text-[#6B6B66] text-[11px]">Confidence: </span>
                <span className="font-bold font-display text-[#1C1C1A]">{confidencePct}%</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2 rounded-full bg-[#EDECE6] overflow-hidden">
              <div
                className="h-full bg-[#3A5244] rounded-full transition-all duration-300"
                style={{ width: `${confidencePct}%` }}
              />
            </div>

            {/* Posterior Probabilities */}
            <div className="flex justify-between text-[11px] font-body text-[#6B6B66] pt-0.5">
              <span>P(Left): {(probLeft * 100).toFixed(1)}%</span>
              <span className="font-semibold text-[#1C1C1A]">t = 0 ms onset</span>
              <span>P(Right): {(probRight * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
