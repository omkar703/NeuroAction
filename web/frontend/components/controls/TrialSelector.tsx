'use client';

import React, { useEffect, useState } from 'react';
import {
  ArrowRight,
  ArrowLeft,
  CircleDot,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react';
import { fetchParticipants, fetchTrials } from '@/lib/api';
import { TrialMeta } from '@/lib/types';
import { useBackend } from '@/lib/backend-context';

export interface TrialSelectorProps {
  selectedParticipant: string;
  selectedTrial: string;
  selectedProtocol: string;
  onSelectParticipant: (participantId: string) => void;
  onSelectTrial: (trialId: string) => void;
  onSelectProtocol: (protocol: string) => void;
  onLoadTrial: () => void;
  onRunLiveInference?: () => void;
  isLoading?: boolean;
  metadata?: {
    congruency?: string;
    target_goal?: string;
    physical_hand_direction?: string;
    ball_color?: string;
    trial_id?: string;
    participant_id?: string;
  } | null;
}

export default function TrialSelector({
  selectedParticipant,
  selectedTrial,
  selectedProtocol,
  onSelectParticipant,
  onSelectTrial,
  onSelectProtocol,
  onLoadTrial,
  onRunLiveInference,
  isLoading = false,
  metadata,
}: TrialSelectorProps) {
  const { isBackendConnected, showBackendModal } = useBackend();
  const [participants, setParticipants] = useState<string[]>(['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15']);
  const [trialsList, setTrialsList] = useState<TrialMeta[]>([]);
  const [isFetchingTrials, setIsFetchingTrials] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function loadParticipants() {
      try {
        const list = await fetchParticipants();
        if (isMounted && list && list.length > 0) {
          setParticipants(list);
          if (!list.includes(selectedParticipant)) {
            onSelectParticipant(list[0]);
          }
        }
      } catch (err) {
        console.warn('Could not load participants list:', err);
      }
    }
    loadParticipants();
    return () => {
      isMounted = false;
    };
  }, [selectedParticipant, onSelectParticipant]);

  useEffect(() => {
    let isMounted = true;
    async function loadTrialsForParticipant() {
      if (!selectedParticipant) return;
      setIsFetchingTrials(true);
      try {
        const trials = await fetchTrials(selectedParticipant);
        if (isMounted) {
          setTrialsList(trials);
          if (trials.length > 0 && !trials.some((t) => t.trial_id === selectedTrial)) {
            onSelectTrial(trials[0].trial_id);
          }
        }
      } catch (err) {
        console.warn('Could not load trials list:', err);
      } finally {
        if (isMounted) {
          setIsFetchingTrials(false);
        }
      }
    }
    loadTrialsForParticipant();
    return () => {
      isMounted = false;
    };
  }, [selectedParticipant, onSelectTrial, selectedTrial]);

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-6 space-y-4 shadow-sm font-body">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h2 className="font-display text-sm font-bold text-[#1C1C1A]">
              Trial Configuration
            </h2>
            <p className="text-[11px] text-[#6B6B66] font-body">Dataset selection and protocol parameters</p>
          </div>
        </div>
        {!isBackendConnected ? (
          <span className="text-[10px] font-body px-2.5 py-0.5 rounded-full bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A]">
            Demo · sub-01
          </span>
        ) : (
          <span className="text-[10px] font-body px-2.5 py-0.5 rounded-full bg-[#ECFDF5] text-[#065F46] border border-[#A7F3D0]">
            Full Cohort · ds006840
          </span>
        )}
      </div>

      {/* Selectors Grid */}
      <div className="space-y-3">
        {/* Participant Selector */}
        <div>
          <label className="block text-xs font-semibold text-[#1C1C1A] mb-1 flex items-center justify-between font-body">
            <span>Participant (LOSO Split)</span>
            <span className="text-[10px] text-[#AEADA4] font-body">Held-Out Subject</span>
          </label>
          <select
            value={selectedParticipant}
            onChange={(e) => {
              const pid = e.target.value;
              if (!isBackendConnected && pid !== 'sub-01') {
                showBackendModal({ actionType: 'participant' });
                return;
              }
              onSelectParticipant(pid);
            }}
            disabled={isLoading}
            className="w-full bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl px-3 py-2 text-xs text-[#1C1C1A] font-body focus:outline-none focus:border-[#3A5244] disabled:opacity-50 cursor-pointer"
          >
            {participants.map((pid) => (
              <option key={pid} value={pid}>
                {pid} (Subject {pid.replace('sub-', '')})
                {!isBackendConnected && pid === 'sub-01' ? ' — Included in Demo' : ''}
                {!isBackendConnected && pid !== 'sub-01' ? ' — Requires Local Backend' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Trial Selector */}
        <div>
          <label className="block text-xs font-semibold text-[#1C1C1A] mb-1 flex items-center justify-between font-body">
            <span>Trial Identifier</span>
            {isFetchingTrials ? (
              <span className="text-[10px] text-[#6B6B66] flex items-center gap-1 font-body">
                <Loader2 className="w-3 h-3 animate-spin" /> Loading...
              </span>
            ) : (
              <span className="text-[10px] text-[#AEADA4] font-body">
                {trialsList.length} indexed
              </span>
            )}
          </label>
          <select
            value={selectedTrial}
            onChange={(e) => onSelectTrial(e.target.value)}
            disabled={isLoading || isFetchingTrials}
            className="w-full bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl px-3 py-2 text-xs text-[#1C1C1A] font-body focus:outline-none focus:border-[#3A5244] disabled:opacity-50 cursor-pointer"
          >
            {trialsList.length > 0 ? (
              trialsList.map((trial) => (
                <option key={trial.trial_id} value={trial.trial_id}>
                  {trial.trial_id} ({trial.congruency}, {trial.target_goal})
                </option>
              ))
            ) : (
              <option value={selectedTrial}>
                {selectedTrial || 'sub-01_run-01_L1_tr00'}
              </option>
            )}
          </select>
        </div>

        {/* Protocol Selector */}
        <div>
          <label className="block text-xs font-semibold text-[#1C1C1A] mb-1 flex items-center justify-between font-body">
            <span>Forecasting Protocol</span>
            <span className="text-[10px] text-[#AEADA4] font-body">Prefix Window</span>
          </label>
          <select
            value={selectedProtocol}
            onChange={(e) => onSelectProtocol(e.target.value)}
            disabled={isLoading}
            className="w-full bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl px-3 py-2 text-xs text-[#1C1C1A] font-body focus:outline-none focus:border-[#3A5244] disabled:opacity-50 cursor-pointer"
          >
            <option value="kinematic_onset">
              Kinematic Onset (T_gen = 825ms, velocity-thresholded)
            </option>
            <option value="event_locked">
              Event Locked (T_gen = 205ms, fixed go-cue lock)
            </option>
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-2 flex flex-col sm:flex-row gap-2">
        <button
          onClick={onLoadTrial}
          disabled={isLoading}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] hover:bg-[#3A5244] text-[#F5F3EE] transition-colors disabled:opacity-50 cursor-pointer"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Loading Trial...</span>
            </>
          ) : (
            <>
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Load &amp; Replay</span>
            </>
          )}
        </button>

        {onRunLiveInference && (
          <button
            onClick={onRunLiveInference}
            disabled={isLoading}
            className="inline-flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-full text-xs font-semibold font-body border border-[#D9D8D0] hover:bg-[#EDECE6] text-[#1C1C1A] transition-colors disabled:opacity-50 cursor-pointer"
            title="Execute prospective inference using saved models"
          >
            <Sparkles className="w-3.5 h-3.5 text-[#3A5244]" />
            <span>Run Inference</span>
          </button>
        )}
      </div>

      {/* Loaded Metadata Summary */}
      {metadata && (
        <div className="pt-3 border-t border-[#D9D8D0] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-[#AEADA4] font-body font-semibold">
              Trial Parameters
            </span>
            <span className="text-[10px] text-[#6B6B66] font-body">
              {metadata.participant_id}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {/* Condition */}
            <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] font-body block font-medium">
                Condition
              </span>
              <span className="font-semibold text-[#1C1C1A] capitalize flex items-center gap-1 mt-0.5 font-display">
                <CircleDot className="w-3 h-3 text-[#3A5244]" />
                {metadata.congruency || 'congruent'}
              </span>
            </div>

            {/* Target Goal */}
            <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] font-body block font-medium">
                Target Goal
              </span>
              <div className="flex items-center gap-1 mt-0.5 font-semibold text-[#1C1C1A] font-display">
                {metadata.target_goal === 'left' ? (
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
              </div>
            </div>

            {/* Ball Color */}
            <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] font-body block font-medium">
                Cue Color
              </span>
              <span className="font-semibold text-[#1C1C1A] capitalize block mt-0.5 font-display">
                {metadata.ball_color || 'Standard'}
              </span>
            </div>

            {/* Hand Direction */}
            <div className="p-2.5 rounded-xl bg-[#FAF9F5] border border-[#D9D8D0]">
              <span className="text-[10px] uppercase text-[#AEADA4] font-body block font-medium">
                Hand Direction
              </span>
              <div className="flex items-center gap-1 mt-0.5 font-semibold text-[#1C1C1A] font-display">
                {metadata.physical_hand_direction === 'left' ? (
                  <>
                    <ArrowLeft className="w-3 h-3 text-[#2E6B47]" />
                    <span>Left</span>
                  </>
                ) : (
                  <>
                    <span>Right</span>
                    <ArrowRight className="w-3 h-3 text-[#2E6B47]" />
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
