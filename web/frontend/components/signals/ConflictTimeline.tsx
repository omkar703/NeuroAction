'use client';

import React, { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import {
  ShieldAlert,
  ShieldCheck,
  Clock,
  AlertTriangle,
  Info,
} from 'lucide-react';
import { ConflictFrame } from '@/lib/types';

export interface ConflictTimelineProps {
  conflictTimeseries?: ConflictFrame[];
  currentTime?: number;
  onSeek?: (timeMs: number) => void;
  title?: string;
}

interface TimelineDatum {
  time: number;
  norm_dist: number;
  state: string;
  dist_mm: number;
  explanation: string;
  aligned_score?: number | null;
  conflict_score?: number | null;
  unavailable_score?: number | null;
}

export default function ConflictTimeline({
  conflictTimeseries = [],
  currentTime = 0,
  onSeek,
  title = 'Causal Intention-Action Conflict Timeline',
}: ConflictTimelineProps) {
  const [hoveredPoint, setHoveredPoint] = useState<TimelineDatum | null>(null);

  const chartData = useMemo<TimelineDatum[]>(() => {
    if (!conflictTimeseries || conflictTimeseries.length === 0) return [];

    return conflictTimeseries.map((frame) => {
      const state = (frame.state || 'UNAVAILABLE').toUpperCase();
      const normDist = frame.norm_dist !== undefined && frame.norm_dist !== null
        ? Math.min(1, Math.max(0, frame.norm_dist))
        : 0;

      const isUnavailable = state === 'UNAVAILABLE';
      const isConflict = state === 'POTENTIAL_DISAGREEMENT' || state === 'CONFLICT' || state === 'DIVERGING';

      return {
        time: frame.time_ms,
        norm_dist: normDist,
        state,
        dist_mm: frame.dist_mm || 0,
        explanation: frame.explanation || (isUnavailable ? 'Model not yet temporally available (before 200ms prefix)' : ''),
        aligned_score: !isUnavailable && !isConflict ? normDist : null,
        conflict_score: isConflict ? normDist : null,
        unavailable_score: isUnavailable ? 0 : null,
      };
    });
  }, [conflictTimeseries]);

  const currentFrame = useMemo<TimelineDatum | null>(() => {
    if (chartData.length === 0) return null;
    let closest = chartData[0];
    let minDiff = Math.abs(closest.time - currentTime);

    for (const d of chartData) {
      const diff = Math.abs(d.time - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = d;
      }
    }
    return closest;
  }, [chartData, currentTime]);

  const activeDisplay = hoveredPoint || currentFrame;
  const maxTime = chartData.length > 0 ? chartData[chartData.length - 1].time : 1000;

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-5 space-y-4 shadow-xs">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#D9D8D0] pb-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h2 className="font-display text-sm font-semibold text-[#1C1C1A]">{title}</h2>
            <p className="text-[11px] text-[#6B6B66]">
              Normalized cross-modal distance score (c_norm &in; [0, 1]) over time
            </p>
          </div>
        </div>

        {/* Current State Badge */}
        {activeDisplay && (
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium border ${
                activeDisplay.state === 'ALIGNED'
                  ? 'bg-[#EAEFE9] text-[#3A5244] border-[#3A5244]/20'
                  : activeDisplay.state === 'POTENTIAL_DISAGREEMENT' || activeDisplay.state === 'CONFLICT'
                  ? 'bg-[#FEF3C7] text-[#92400E] border-[#D97706]/30'
                  : 'bg-[#EDECE6] text-[#6B6B66] border-[#D9D8D0]'
              }`}
            >
              {activeDisplay.state === 'ALIGNED' ? (
                <ShieldCheck className="w-3.5 h-3.5" />
              ) : activeDisplay.state === 'POTENTIAL_DISAGREEMENT' || activeDisplay.state === 'CONFLICT' ? (
                <AlertTriangle className="w-3.5 h-3.5" />
              ) : (
                <Clock className="w-3.5 h-3.5" />
              )}
              <span>{activeDisplay.state}</span>
            </span>

            <span className="text-xs font-mono text-[#1C1C1A] bg-[#EDECE6] border border-[#D9D8D0] px-2.5 py-0.5 rounded-full">
              Score: <strong className="font-semibold">{activeDisplay.norm_dist.toFixed(2)}</strong>
            </span>
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className="w-full h-[220px] bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl p-2 relative">
        {chartData.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-[#6B6B66]">
            <ShieldAlert className="w-8 h-8 mb-2 animate-pulse text-[#D9D8D0]" />
            <p className="text-xs">No conflict timeseries available</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 25, left: -15, bottom: 5 }}
              onClick={(e) => {
                const eventData = e as unknown as { activePayload?: Array<{ payload: { time: number } }> };
                if (eventData?.activePayload && eventData.activePayload.length > 0 && onSeek) {
                  onSeek(eventData.activePayload[0].payload.time);
                }
              }}
              onMouseMove={(e) => {
                const eventData = e as unknown as { activePayload?: Array<{ payload: TimelineDatum }> };
                if (eventData?.activePayload && eventData.activePayload.length > 0) {
                  setHoveredPoint(eventData.activePayload[0].payload);
                }
              }}
              onMouseLeave={() => setHoveredPoint(null)}
            >
              <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" opacity={0.8} />

              <XAxis
                dataKey="time"
                stroke="#D9D8D0"
                tick={{ fill: '#6B6B66', fontSize: 11 }}
                tickFormatter={(val) => `${val}ms`}
                domain={[0, maxTime]}
              />

              <YAxis
                stroke="#D9D8D0"
                tick={{ fill: '#6B6B66', fontSize: 11 }}
                domain={[0, 1.05]}
                ticks={[0, 0.25, 0.5, 0.75, 1.0]}
                tickFormatter={(val) => val.toFixed(1)}
              />

              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload || !payload.length) return null;
                  const data = payload[0].payload as TimelineDatum;

                  const isUnavail = data.state === 'UNAVAILABLE';
                  const isConflict = data.state === 'POTENTIAL_DISAGREEMENT' || data.state === 'CONFLICT';

                  return (
                    <div className="p-2.5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-xl shadow-md text-xs space-y-1 max-w-xs text-[#1C1C1A]">
                      <div className="flex items-center justify-between gap-4 font-mono border-b border-[#D9D8D0] pb-1">
                        <span className="font-semibold">Time: {data.time} ms</span>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-mono ${
                            isUnavail
                              ? 'bg-[#EDECE6] text-[#6B6B66]'
                              : isConflict
                              ? 'bg-[#FEF3C7] text-[#92400E]'
                              : 'bg-[#EAEFE9] text-[#3A5244]'
                          }`}
                        >
                          {data.state}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 font-mono pt-0.5">
                        <div className="text-[#6B6B66]">
                          <span>c_norm: </span>
                          <span className="font-semibold text-[#1C1C1A]">{data.norm_dist.toFixed(2)}</span>
                        </div>
                        <div className="text-[#6B6B66]">
                          <span>d_target: </span>
                          <span className="font-semibold text-[#1C1C1A]">{data.dist_mm.toFixed(0)} mm</span>
                        </div>
                      </div>

                      <p className="text-[11px] text-[#6B6B66] italic pt-1 border-t border-[#D9D8D0] leading-snug">
                        {isUnavail
                          ? 'Prior to 200 ms kinematic prefix'
                          : data.explanation || 'Instantaneous alignment'}
                      </p>
                    </div>
                  );
                }}
              />

              {/* Threshold Line at 0.50 */}
              <ReferenceLine
                y={0.5}
                stroke="#D97706"
                strokeDasharray="3 3"
                strokeWidth={1.5}
                label={{
                  value: 'Disagreement Threshold (0.50)',
                  fill: '#92400E',
                  position: 'insideTopRight',
                  fontSize: 10,
                  offset: 6,
                }}
              />

              {/* Current Playback Line */}
              <ReferenceLine
                x={currentTime}
                stroke="#3A5244"
                strokeWidth={1.5}
                label={{
                  value: `${Math.round(currentTime)}ms`,
                  fill: '#1C1C1A',
                  position: 'top',
                  fontSize: 10,
                }}
              />

              {/* Aligned Segment */}
              <Line
                type="monotone"
                dataKey="aligned_score"
                name="Aligned Score"
                stroke="#3A5244"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />

              {/* Conflict Segment */}
              <Line
                type="monotone"
                dataKey="conflict_score"
                name="Divergence Score"
                stroke="#B45309"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Footer Notes */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-[#D9D8D0] text-xs">
        <div className="flex items-center gap-4 text-[#1C1C1A]">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#3A5244] rounded-full" />
            <span className="text-[11px]">Congruent (Aligned)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#B45309] rounded-full" />
            <span className="text-[11px]">Incongruent (Discordance)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 border-t border-dashed border-[#D97706]" />
            <span className="text-[11px]">Threshold (0.50)</span>
          </div>
        </div>

        <div className="text-[11px] text-[#6B6B66] flex items-center gap-1 font-mono">
          <Info className="w-3 h-3" />
          <span>Click anywhere along timeline to seek</span>
        </div>
      </div>
    </div>
  );
}
