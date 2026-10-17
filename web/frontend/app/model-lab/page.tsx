'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import {
  Brain,
  ArrowRight,
  AlertTriangle,
} from 'lucide-react';
import { fetchTrialData, runInference } from '@/lib/api';
import { TrialData } from '@/lib/types';
import { useBackend } from '@/lib/backend-context';
import TrialSelector from '@/components/controls/TrialSelector';
import PlaybackControls from '@/components/controls/PlaybackControls';
import PipelineViz from '@/components/pipeline/PipelineViz';
import TrajectoryChart from '@/components/trajectory/TrajectoryChart';
import MetadataCard from '@/components/signals/MetadataCard';
import ConflictTimeline from '@/components/signals/ConflictTimeline';

export default function ModelLabPage() {
  const { isBackendConnected, showBackendModal } = useBackend();
  const [selectedParticipant, setSelectedParticipant] = useState<string>('sub-01');
  const [selectedTrial, setSelectedTrial] = useState<string>('sub-01_run-01_L1_tr00');
  const [selectedProtocol, setSelectedProtocol] = useState<string>('kinematic_onset');

  const [trialData, setTrialData] = useState<TrialData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const maxTime = trialData && trialData.times_ms && trialData.times_ms.length > 0
    ? trialData.times_ms[trialData.times_ms.length - 1]
    : 1200;

  const handleLoadTrial = useCallback(async () => {
    if (!selectedTrial) return;
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);

    try {
      const data = await fetchTrialData(selectedTrial, selectedProtocol);
      setTrialData(data);
      setCurrentTime(0);
    } catch (err: unknown) {
      console.error('Failed to load trial:', err);
      setError('Could not retrieve trial data from server. Ensure backend is running.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedTrial, selectedProtocol]);

  const handleRunLiveInference = useCallback(async () => {
    if (!selectedTrial) return;
    if (!isBackendConnected) {
      showBackendModal({ actionType: 'inference' });
    }
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);

    try {
      const liveData = await runInference(selectedTrial, selectedProtocol);
      setTrialData(liveData);
      setCurrentTime(0);
    } catch (err: unknown) {
      console.error('Live inference failed:', err);
      setError('Live inference request failed. Falling back to cached trial replay.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedTrial, selectedProtocol, isBackendConnected, showBackendModal]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const data = await fetchTrialData(selectedTrial, selectedProtocol);
        if (active) {
          setTrialData(data);
          setCurrentTime(0);
        }
      } catch (err: unknown) {
        if (active) {
          console.error('Failed to load initial trial:', err);
          setError('Backend offline. Demonstrating offline trial models.');
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [selectedTrial, selectedProtocol]);

  useEffect(() => {
    if (isPlaying) {
      const tickMs = 25;
      const deltaMs = tickMs * playbackSpeed;

      timerRef.current = setInterval(() => {
        setCurrentTime((prev) => {
          const next = prev + deltaMs;
          if (next >= maxTime) {
            setIsPlaying(false);
            return maxTime;
          }
          return next;
        });
      }, tickMs);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, playbackSpeed, maxTime]);

  const handlePlayPause = useCallback(() => {
    if (currentTime >= maxTime) {
      setCurrentTime(0);
    }
    setIsPlaying((prev) => !prev);
  }, [currentTime, maxTime]);

  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime((prev) => Math.min(maxTime, prev + 10));
  }, [maxTime]);

  const handleRestart = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, []);

  const handleSeek = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleSpeedChange = useCallback((speed: number) => {
    setPlaybackSpeed(speed);
  }, []);

  const metadataSummary = trialData
    ? {
        congruency: (trialData.congruency || (trialData.metadata as Record<string, unknown>)?.congruency) as string | undefined,
        target_goal: (trialData.target_goal || (trialData.metadata as Record<string, unknown>)?.target_goal) as string | undefined,
        physical_hand_direction: (trialData.physical_hand_direction ||
          (trialData.metadata as Record<string, unknown>)?.physical_hand_direction) as string | undefined,
        ball_color: (trialData.ball_color || (trialData.metadata as Record<string, unknown>)?.ball_color) as string | undefined,
        trial_id: trialData.trial_id,
        participant_id: trialData.participant_id,
      }
    : null;

  return (
    <div className="w-full max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20 space-y-10 font-body">
      {/* Header */}
      <div className="border-b border-[#D9D8D0] pb-8 space-y-4">
        <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">
          Laboratory Replay &bull; OpenNeuro ds006840
        </p>
        <div>
          <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#1C1C1A]">
            Model Lab &amp; Empirical Replay
          </h1>
          <p className="text-base sm:text-lg text-[#6B6B66] max-w-2xl mt-2 leading-relaxed font-body">
            Inspect synchronous trial recordings, evaluate trained Ridge regression extrapolations, and trace intention–action discordance.
          </p>
        </div>

        {error && (
          <div className="p-4 rounded-2xl bg-[#FEF3C7]/40 border border-[#D97706]/30 flex items-center gap-2 text-[#92400E] text-xs font-body">
            <AlertTriangle className="w-4 h-4 shrink-0 text-[#D97706]" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* 2-Column Laboratory Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Trial Controls & Pipeline Manifest (4 of 12 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <TrialSelector
            selectedParticipant={selectedParticipant}
            selectedTrial={selectedTrial}
            selectedProtocol={selectedProtocol}
            onSelectParticipant={setSelectedParticipant}
            onSelectTrial={setSelectedTrial}
            onSelectProtocol={setSelectedProtocol}
            onLoadTrial={handleLoadTrial}
            onRunLiveInference={handleRunLiveInference}
            isLoading={isLoading}
            metadata={metadataSummary}
          />

          <PipelineViz
            isLoading={isLoading}
            isLoaded={!!trialData}
            isLiveInference={trialData?.data_source === 'live_inference'}
          />
        </div>

        {/* Right Column: Playback & Trajectory Inspection (8 of 12 cols) */}
        <div className="lg:col-span-8 space-y-6">
          <PlaybackControls
            currentTime={currentTime}
            maxTime={maxTime}
            minTime={0}
            stepMs={10}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            onPlayPause={handlePlayPause}
            onStepForward={handleStepForward}
            onRestart={handleRestart}
            onSpeedChange={handleSpeedChange}
            onSeek={handleSeek}
            disabled={isLoading || !trialData}
          />

          <TrajectoryChart
            times_ms={trialData?.times_ms || []}
            hand_x={trialData?.hand_x || []}
            hand_y={trialData?.hand_y || []}
            hand_z={trialData?.hand_z || []}
            predicted_endpoint={trialData?.predicted_endpoint || null}
            forecast_generation_time_ms={trialData?.forecast_generation_time_ms || 825}
            currentTime={currentTime}
            title={`Kinematic Trajectory & Horizon Forecast — ${selectedTrial}`}
          />

          <ConflictTimeline
            conflictTimeseries={trialData?.conflict_timeseries || []}
            currentTime={currentTime}
            onSeek={handleSeek}
            title="Intention–Action Divergence Timeline"
          />

          {/* Quick link to raw EEG explorer */}
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs shadow-sm">
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-[#1C1C1A] font-display">
                <Brain className="w-3.5 h-3.5 text-[#3A5244]" />
                <span>Pre-Movement EEG Inspection</span>
              </div>
              <p className="text-[#6B6B66] font-body">
                View 26-channel raw potentials and CSP filter weights for {selectedTrial}.
              </p>
            </div>
            <Link
              href={`/explainability?trial_id=${encodeURIComponent(selectedTrial)}`}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold font-body border border-[#D9D8D0] text-[#1C1C1A] hover:bg-[#EDECE6] transition-colors self-start sm:self-center shrink-0 cursor-pointer"
            >
              <span>Explainability Explorer</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#3A5244]" />
            </Link>
          </div>
        </div>
      </div>

      {/* Full-Width Trial Manifest & Provenance Section */}
      {trialData && (
        <div className="w-full">
          <MetadataCard
            trialId={trialData.trial_id}
            participantId={trialData.participant_id}
            congruency={metadataSummary?.congruency}
            targetGoal={metadataSummary?.target_goal}
            physicalHandDirection={metadataSummary?.physical_hand_direction}
            ballColor={metadataSummary?.ball_color}
            neural={trialData.neural}
            dataSource={trialData.data_source}
            forecastGenerationTimeMs={trialData.forecast_generation_time_ms}
          />
        </div>
      )}
    </div>
  );
}
