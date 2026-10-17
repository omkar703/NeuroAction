'use client';

import React, { useState, useEffect } from 'react';
import { ArrowRight } from 'lucide-react';
import Link from 'next/link';
import EEGExplorer from '@/components/signals/EEGExplorer';
import LDAInspector from '@/components/signals/LDAInspector';
import InputOutputInspector from '@/components/pipeline/InputOutputInspector';
import {
  fetchParticipants,
  fetchTrials,
  fetchTrialData,
} from '@/lib/api';
import { TrialMeta, TrialData } from '@/lib/types';
import { useBackend } from '@/lib/backend-context';

export default function ExplainabilityPage() {
  const { isBackendConnected, showBackendModal } = useBackend();
  const [participants, setParticipants] = useState<string[]>(['sub-01', 'sub-02', 'sub-03', 'sub-08', 'sub-15']);
  const [selectedParticipant, setSelectedParticipant] = useState<string>('sub-01');
  const [trials, setTrials] = useState<TrialMeta[]>([]);
  const [selectedTrialId, setSelectedTrialId] = useState<string>('sub-01_run-01_L1_tr00');
  const [trialData, setTrialData] = useState<TrialData | null>(null);
  const [selectedChannel, setSelectedChannel] = useState<string>('CZ');

  useEffect(() => {
    let isCancelled = false;
    fetchParticipants()
      .then((pList) => {
        if (!isCancelled && pList && pList.length > 0) {
          setParticipants(pList);
        }
      })
      .catch((err) => console.warn('Failed to load participants list:', err));
    return () => {
      isCancelled = true;
    };
  }, []);

  useEffect(() => {
    let isCancelled = false;
    fetchTrials(selectedParticipant)
      .then((tList) => {
        if (!isCancelled && tList && tList.length > 0) {
          setTrials(tList);
          setSelectedTrialId((prev) => {
            const match = tList.find((t) => t.trial_id === prev);
            return match ? match.trial_id : tList[0].trial_id;
          });
        }
      })
      .catch((err) => console.warn('Failed to load trials for participant:', err));
    return () => {
      isCancelled = true;
    };
  }, [selectedParticipant]);

  useEffect(() => {
    let isCancelled = false;
    if (!selectedTrialId) return;

    fetchTrialData(selectedTrialId)
      .then((data) => {
        if (!isCancelled) {
          setTrialData(data);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          console.error('Failed to load trial data:', err);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [selectedTrialId]);

  const handleParticipantChange = (pId: string) => {
    if (!isBackendConnected && pId !== 'sub-01') {
      showBackendModal({ actionType: 'participant' });
      return;
    }
    setSelectedParticipant(pId);
  };

  const handleTrialChange = (tId: string) => {
    setSelectedTrialId(tId);
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20 space-y-10 font-body">
      {/* Editorial Header */}
      <div className="border-b border-[#D9D8D0] pb-8 space-y-3">
        <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">
          Model Interpretability &bull; Epistemic Audit
        </p>

        <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#1C1C1A]">
          Signal Explainability &amp; Feature Attribution
        </h1>

        <p className="max-w-3xl text-[#6B6B66] text-base sm:text-lg leading-relaxed font-body">
          Inspect the transformation chain from raw pre-movement scalp EEG epochs to spatial LDA projection weights, kinematic forecast horizons, and epistemic boundaries.
        </p>
      </div>

      {/* Trial Selector Control Bar */}
      <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-xs font-body">
          <div className="flex items-center gap-2">
            <label htmlFor="participant-dropdown" className="font-semibold text-[#1C1C1A]">
              Participant:
            </label>
            <select
              id="participant-dropdown"
              value={selectedParticipant}
              onChange={(e) => handleParticipantChange(e.target.value)}
              className="bg-[#FAF9F5] text-[#1C1C1A] font-body text-xs rounded-xl px-3 py-1.5 border border-[#D9D8D0] focus:outline-none focus:border-[#3A5244] cursor-pointer"
            >
              {participants.map((p) => (
                <option key={p} value={p}>
                  {p.toUpperCase()}
                  {!isBackendConnected && p === 'sub-01' ? ' — Demo Dataset' : ''}
                  {!isBackendConnected && p !== 'sub-01' ? ' — Requires Local Backend' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="trial-dropdown" className="font-semibold text-[#1C1C1A]">
              Active Trial:
            </label>
            <select
              id="trial-dropdown"
              value={selectedTrialId}
              onChange={(e) => handleTrialChange(e.target.value)}
              className="bg-[#FAF9F5] text-[#1C1C1A] font-body text-xs rounded-xl px-3 py-1.5 border border-[#D9D8D0] focus:outline-none focus:border-[#3A5244] max-w-[240px] truncate cursor-pointer"
            >
              {trials.length > 0 ? (
                trials.map((t) => (
                  <option key={t.trial_id} value={t.trial_id}>
                    {t.trial_id} ({t.congruency})
                  </option>
                ))
              ) : (
                <option value={selectedTrialId}>{selectedTrialId}</option>
              )}
            </select>
          </div>

          {trialData && (
            <div className="hidden lg:flex items-center gap-2 text-xs font-body text-[#6B6B66]">
              <span className="px-3 py-1 rounded-full bg-[#EDECE6] border border-[#D9D8D0]">
                Cue: <strong className="text-[#1C1C1A]">{trialData.target_goal || trialData.neural?.predicted_goal || 'left'}</strong>
              </span>
              <span className="px-3 py-1 rounded-full bg-[#EDECE6] border border-[#D9D8D0]">
                Condition: <strong className="text-[#1C1C1A]">{trialData.congruency || 'congruent'}</strong>
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs">
          <Link
            href={`/digital-twin?trial=${encodeURIComponent(selectedTrialId)}`}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full font-semibold font-body border border-[#D9D8D0] text-[#1C1C1A] hover:bg-[#EDECE6] transition-colors cursor-pointer"
          >
            <span>View in 3D Twin</span>
            <ArrowRight className="w-3.5 h-3.5 text-[#3A5244]" />
          </Link>
        </div>
      </div>

      {/* Two-Column Layout: EEGExplorer (Left) + LDAInspector (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        <EEGExplorer
          trialId={selectedTrialId}
          selectedChannel={selectedChannel}
          onChannelChange={(ch) => setSelectedChannel(ch)}
        />

        <LDAInspector participantId={selectedParticipant} />
      </div>

      {/* Full-Width Section: InputOutputInspector */}
      <InputOutputInspector trialData={trialData} />
    </div>
  );
}
