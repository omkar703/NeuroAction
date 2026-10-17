'use client';

import React, { useState, useEffect, useSyncExternalStore } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
} from 'recharts';
import { Activity, AlertTriangle, RotateCcw, Loader2 } from 'lucide-react';
import { fetchEEGSignal } from '@/lib/api';
import { EEGSignal } from '@/lib/types';

interface EEGExplorerProps {
  trialId: string;
  channelOptions?: string[];
  selectedChannel?: string;
  onChannelChange?: (ch: string) => void;
}

const DEFAULT_CHANNELS = [
  'CZ', 'C3', 'C4',
  'FCZ', 'FC3', 'FC4',
  'CPZ', 'CP3', 'CP4',
  'FZ', 'F3', 'F4', 'F7', 'F8',
  'PZ', 'P3', 'P4', 'P7', 'P8',
  'OZ', 'O1', 'O2',
  'FP1', 'FP2', 'T7', 'T8',
];

export default function EEGExplorer({
  trialId,
  channelOptions = DEFAULT_CHANNELS,
  selectedChannel,
  onChannelChange,
}: EEGExplorerProps) {
  const [internalChannel, setInternalChannel] = useState<string>('CZ');
  const currentChannel = selectedChannel ?? internalChannel;

  const [signalData, setSignalData] = useState<EEGSignal | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [xDomain, setXDomain] = useState<[number, number] | null>(null);

  const isMounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  useEffect(() => {
    let isCancelled = false;

    fetchEEGSignal(trialId, currentChannel)
      .then((data) => {
        if (!isCancelled) {
          setSignalData(data);
          setError(null);
          setLoading(false);
          setXDomain(null);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          console.error('Failed to load EEG signal:', err);
          setError('Unable to load EEG signal for this channel.');
          setLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [trialId, currentChannel]);

  const handleChannelSelect = (ch: string) => {
    setInternalChannel(ch);
    if (onChannelChange) {
      onChannelChange(ch);
    }
  };

  const chartData = signalData?.times_ms?.map((time, idx) => ({
    time,
    amplitude: signalData.amplitude_uv?.[idx] !== undefined ? signalData.amplitude_uv[idx] : 0,
  })) || [];

  const handleZoom = () => {
    if (refAreaLeft === null || refAreaRight === null || refAreaLeft === refAreaRight) {
      setRefAreaLeft(null);
      setRefAreaRight(null);
      return;
    }
    const [start, end] = refAreaLeft < refAreaRight
      ? [refAreaLeft, refAreaRight]
      : [refAreaRight, refAreaLeft];

    if (end - start >= 50) {
      setXDomain([start, end]);
    }
    setRefAreaLeft(null);
    setRefAreaRight(null);
  };

  const handleResetZoom = () => {
    setXDomain(null);
    setRefAreaLeft(null);
    setRefAreaRight(null);
  };

  const availableChannelsList = signalData?.available_channels?.length
    ? signalData.available_channels
    : channelOptions;

  return (
    <div className="w-full flex flex-col rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] p-5 space-y-4 shadow-xs">
      {/* Component Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#D9D8D0]">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h3 className="font-display text-sm font-semibold text-[#1C1C1A] flex items-center gap-2">
              EEG Signal Inspector
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] border border-[#D9D8D0]">
                [-1000, 0] ms
              </span>
            </h3>
            <p className="text-[11px] text-[#6B6B66]">
              Pre-movement sensorimotor potentials at electrode {currentChannel}
            </p>
          </div>
        </div>

        {/* Channel Selector & Reset Zoom */}
        <div className="flex items-center gap-2 text-xs">
          {xDomain && (
            <button
              onClick={handleResetZoom}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs bg-[#EDECE6] hover:bg-[#E2E0D8] text-[#1C1C1A] border border-[#D9D8D0] transition-colors"
              title="Reset Zoom Domain"
            >
              <RotateCcw className="w-3 h-3 text-[#3A5244]" />
              <span>Reset Zoom</span>
            </button>
          )}

          <div className="flex items-center gap-1.5">
            <label htmlFor="channel-select" className="text-xs text-[#6B6B66]">
              Channel:
            </label>
            <select
              id="channel-select"
              value={currentChannel}
              onChange={(e) => handleChannelSelect(e.target.value)}
              className="bg-[#FFFFFF] text-[#1C1C1A] font-mono text-xs rounded-xl px-2.5 py-1 border border-[#D9D8D0] focus:outline-none focus:border-[#3A5244]"
            >
              {availableChannelsList.map((ch) => (
                <option key={ch} value={ch}>
                  {ch}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      <div className="relative w-full h-64 bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl p-2 select-none">
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-[#FFFFFF]/80 backdrop-blur-xs">
            <Loader2 className="w-5 h-5 text-[#3A5244] animate-spin mb-1.5" />
            <span className="text-xs text-[#6B6B66]">Loading EEG waveform...</span>
          </div>
        )}

        {error ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-center p-4 text-xs text-[#6B6B66]">
            <AlertTriangle className="w-6 h-6 text-[#B45309] mb-1.5" />
            <p className="font-medium text-[#1C1C1A]">{error}</p>
          </div>
        ) : isMounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
              onMouseDown={(e) => e && e.activeLabel && setRefAreaLeft(Number(e.activeLabel))}
              onMouseMove={(e) => refAreaLeft !== null && e && e.activeLabel && setRefAreaRight(Number(e.activeLabel))}
              onMouseUp={handleZoom}
            >
              <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" opacity={0.8} />

              <XAxis
                dataKey="time"
                stroke="#D9D8D0"
                tick={{ fill: '#6B6B66', fontSize: 11 }}
                domain={xDomain || [-1000, 0]}
                type="number"
                unit=" ms"
              />

              <YAxis
                stroke="#D9D8D0"
                tick={{ fill: '#6B6B66', fontSize: 11 }}
                unit=" μV"
                domain={['auto', 'auto']}
              />

              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload as { time: number; amplitude: number };
                    return (
                      <div className="p-2.5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-xl shadow-md text-xs font-mono text-[#1C1C1A]">
                        <div className="text-[#6B6B66]">t = {d.time} ms</div>
                        <div className="text-[#3A5244] font-semibold">
                          Voltage: {Number(d.amplitude).toFixed(2)} μV
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />

              <Line
                type="monotone"
                dataKey="amplitude"
                stroke="#3A5244"
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />

              {refAreaLeft !== null && refAreaRight !== null && (
                <ReferenceArea
                  x1={refAreaLeft}
                  x2={refAreaRight}
                  strokeOpacity={0.3}
                  fill="#3A5244"
                  fillOpacity={0.15}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[#D9D8D0] text-xs text-[#6B6B66]">
        <span>Drag over region to zoom in</span>
        <span className="font-mono text-[11px]">Sampling: 200 Hz &bull; 8–30 Hz Bandpass</span>
      </div>
    </div>
  );
}
