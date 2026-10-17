'use client';

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
} from 'recharts';
import { Activity, Compass, Info } from 'lucide-react';

export interface TrajectoryChartProps {
  times_ms: number[];
  hand_x: number[];
  hand_y: number[];
  hand_z: number[];
  predicted_endpoint: [number, number, number] | number[] | null;
  forecast_generation_time_ms?: number;
  currentTime?: number;
  title?: string;
}

interface ChartDatum {
  time: number;
  actual_x: number;
  forecast_x?: number | null;
  hand_y: number;
  hand_z: number;
}

export default function TrajectoryChart({
  times_ms = [],
  hand_x = [],
  hand_y = [],
  hand_z = [],
  predicted_endpoint = null,
  forecast_generation_time_ms = 825,
  currentTime = 0,
  title = '3D Kinematic Hand X Trajectory & Horizon Forecast',
}: TrajectoryChartProps) {
  const chartData = useMemo<ChartDatum[]>(() => {
    if (!times_ms || times_ms.length === 0) return [];

    const maxT = times_ms[times_ms.length - 1];
    const tGen = forecast_generation_time_ms ?? 825;
    const predX = predicted_endpoint ? predicted_endpoint[0] : null;

    let tGenIdx = times_ms.findIndex((t) => t >= tGen);
    if (tGenIdx === -1) tGenIdx = times_ms.length - 1;
    const startX = hand_x[tGenIdx] !== undefined ? hand_x[tGenIdx] : (hand_x[0] || 0);

    return times_ms.map((t, idx) => {
      const actualX = hand_x[idx] !== undefined ? hand_x[idx] : 0;
      let forecastX: number | null = null;

      if (predX !== null && t >= tGen) {
        if (maxT > tGen) {
          const ratio = (t - tGen) / (maxT - tGen);
          forecastX = parseFloat((startX + (predX - startX) * ratio).toFixed(2));
        } else {
          forecastX = predX;
        }
      }

      return {
        time: t,
        actual_x: actualX,
        forecast_x: forecastX,
        hand_y: hand_y[idx] !== undefined ? hand_y[idx] : 0,
        hand_z: hand_z[idx] !== undefined ? hand_z[idx] : 0,
      };
    });
  }, [times_ms, hand_x, hand_y, hand_z, predicted_endpoint, forecast_generation_time_ms]);

  const currentValues = useMemo(() => {
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

  const maxTime = times_ms.length > 0 ? times_ms[times_ms.length - 1] : 1000;
  const predX = predicted_endpoint ? predicted_endpoint[0] : null;

  return (
    <div className="bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl p-6 space-y-4 shadow-sm font-body">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#D9D8D0] pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h2 className="font-display text-sm font-bold text-[#1C1C1A]">{title}</h2>
            <p className="text-[11px] text-[#6B6B66] font-body">
              Recorded hand path (green) vs. Ridge forecast extrapolation (violet)
            </p>
          </div>
        </div>

        {/* Current coordinates preview badge */}
        {currentValues && (
          <div className="flex items-center gap-3 bg-[#EDECE6] border border-[#D9D8D0] px-3 py-1 rounded-full text-xs font-body text-[#1C1C1A]">
            <div>
              <span className="text-[#6B6B66]">X: </span>
              <span className="font-bold font-display text-[#2E6B47]">{currentValues.actual_x.toFixed(1)}</span>
              <span className="text-[#6B6B66] text-[10px]"> mm</span>
            </div>
            <div>
              <span className="text-[#6B6B66]">Y: </span>
              <span className="font-semibold">{currentValues.hand_y.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-[#6B6B66]">Z: </span>
              <span className="font-semibold">{currentValues.hand_z.toFixed(1)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className="w-full h-[320px] bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl p-3 relative">
        {chartData.length === 0 ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-[#6B6B66] font-body">
            <Compass className="w-8 h-8 mb-2 animate-pulse text-[#AEADA4]" />
            <p className="text-xs">No trajectory data loaded</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 15, right: 25, left: -10, bottom: 5 }}
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
                tickFormatter={(val) => `${val}mm`}
              />

              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload || !payload.length) return null;
                  const data = payload[0].payload as ChartDatum;
                  const isPostGen = data.time >= forecast_generation_time_ms;

                  return (
                    <div className="p-3 bg-[#FFFFFF] border border-[#D9D8D0] rounded-xl shadow-md text-xs font-body space-y-1 text-[#1C1C1A]">
                      <div className="flex items-center justify-between gap-4 border-b border-[#D9D8D0] pb-1">
                        <span className="font-bold font-display">t = {data.time} ms</span>
                        <span className="text-[10px] text-[#AEADA4] font-semibold uppercase">
                          {isPostGen ? 'FORECAST' : 'PREFIX'}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 pt-0.5">
                        <div className="text-[#2E6B47]">
                          <span className="text-[#6B6B66]">Observed X: </span>
                          <span className="font-bold font-display">{data.actual_x.toFixed(1)} mm</span>
                        </div>
                        {data.forecast_x !== null && data.forecast_x !== undefined && (
                          <div className="text-[#6B46C1]">
                            <span className="text-[#6B6B66]">Forecast X: </span>
                            <span className="font-bold font-display">{data.forecast_x.toFixed(1)} mm</span>
                          </div>
                        )}
                        <div className="text-[#6B6B66]">
                          <span>Y: {data.hand_y.toFixed(1)} mm</span>
                        </div>
                        <div className="text-[#6B6B66]">
                          <span>Z: {data.hand_z.toFixed(1)} mm</span>
                        </div>
                      </div>
                    </div>
                  );
                }}
              />

              {/* Forecast Generation Cutoff */}
              <ReferenceLine
                x={forecast_generation_time_ms}
                stroke="#B45309"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                label={{
                  value: `Forecast cutoff (${forecast_generation_time_ms}ms)`,
                  fill: '#92400E',
                  position: 'insideTopLeft',
                  fontSize: 10,
                  offset: 8,
                }}
              />

              {/* Real-time Playback Scrub Cursor */}
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

              {/* Predicted Endpoint Terminal Dot */}
              {predX !== null && (
                <ReferenceDot
                  x={maxTime}
                  y={predX}
                  r={4}
                  fill="#6B46C1"
                  stroke="#FFFFFF"
                  strokeWidth={1.5}
                  label={{
                    value: `Pred: ${predX.toFixed(0)}mm`,
                    fill: '#6B46C1',
                    position: 'left',
                    fontSize: 10,
                    offset: 8,
                  }}
                />
              )}

              {/* Observed Trajectory */}
              <Line
                type="monotone"
                dataKey="actual_x"
                name="Observed Hand X"
                stroke="#2E6B47"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />

              {/* Forecast Continuation */}
              <Line
                type="monotone"
                dataKey="forecast_x"
                name="Ridge Forecast"
                stroke="#6B46C1"
                strokeWidth={1.8}
                strokeDasharray="4 4"
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Legend & Notes */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-[#D9D8D0] text-xs font-body">
        <div className="flex flex-wrap items-center gap-4 text-[#1C1C1A]">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#2E6B47] rounded-full" />
            <span className="text-[11px] font-medium">Observed Path (X)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 border-t border-dashed border-[#6B46C1]" />
            <span className="text-[11px] font-medium">Ridge Forecast</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#6B46C1]" />
            <span className="text-[11px] font-medium">Forecasted Endpoint</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#3A5244]" />
            <span className="text-[11px] font-medium">Time Cursor</span>
          </div>
        </div>

        <div className="text-[11px] text-[#6B6B66] flex items-center gap-1 font-body">
          <Info className="w-3 h-3" />
          <span>Leap Motion mm coordinates</span>
        </div>
      </div>
    </div>
  );
}
