'use client';

import React, { useState, useEffect, useMemo, useSyncExternalStore } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Brain, AlertCircle, Loader2 } from 'lucide-react';
import { fetchLDACoefficients } from '@/lib/api';
import { LDACoefficients } from '@/lib/types';

interface LDAInspectorProps {
  participantId?: string;
}

interface ChannelWeightItem {
  channel: string;
  weight: number;
  absWeight: number;
}

const MOTOR_CHANNELS = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4'];

export default function LDAInspector({ participantId = 'cohort' }: LDAInspectorProps) {
  const [ldaData, setLdaData] = useState<LDACoefficients | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const isMounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  useEffect(() => {
    let isCancelled = false;

    fetchLDACoefficients(participantId)
      .then((res) => {
        if (!isCancelled) {
          setLdaData(res);
          setError(null);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          console.error('Failed to load LDA weights:', err);
          setError('Unable to load LDA spatial weights.');
          setLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [participantId]);

  const chartData: ChannelWeightItem[] = useMemo(() => {
    if (!ldaData) {
      return MOTOR_CHANNELS.map((ch) => ({
        channel: ch,
        weight: 0,
        absWeight: 0,
      }));
    }

    const channels = ldaData.channels || MOTOR_CHANNELS;
    const rawWeights = ldaData.coefficients || ldaData.weights || [];

    const weightMap: Record<string, number> = {};
    channels.forEach((ch, idx) => {
      if (rawWeights[idx] !== undefined) {
        weightMap[ch.toUpperCase()] = rawWeights[idx];
      }
    });

    return MOTOR_CHANNELS.map((ch) => {
      const w = weightMap[ch.toUpperCase()] ?? (weightMap[ch] !== undefined ? weightMap[ch] : 0);
      return {
        channel: ch,
        weight: parseFloat(w.toFixed(4)),
        absWeight: parseFloat(Math.abs(w).toFixed(4)),
      };
    });
  }, [ldaData]);

  return (
    <div className="w-full flex flex-col rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] p-5 space-y-4 shadow-xs">
      {/* Component Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#D9D8D0]">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-[#3A5244]" />
          <div>
            <h3 className="font-display text-sm font-semibold text-[#1C1C1A]">
              LDA Spatial Decision Coefficients
            </h3>
            <p className="text-[11px] text-[#6B6B66]">
              Subject context:{' '}
              <span className="font-mono text-[#1C1C1A] uppercase font-medium">
                {participantId}
              </span>
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#3A5244]" />
            <span className="text-[#6B6B66]">Right Goal (+)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#2563EB]" />
            <span className="text-[#6B6B66]">Left Goal (-)</span>
          </div>
        </div>
      </div>

      {/* Main Chart Container */}
      <div className="relative w-full h-64 bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl p-2 select-none">
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-[#FFFFFF]/80 backdrop-blur-xs">
            <Loader2 className="w-5 h-5 text-[#3A5244] animate-spin mb-1.5" />
            <span className="text-xs text-[#6B6B66]">Loading LDA weights...</span>
          </div>
        )}

        {error ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-center p-4 text-xs text-[#6B6B66]">
            <AlertCircle className="w-6 h-6 text-[#B45309] mb-1.5" />
            <p className="font-medium text-[#1C1C1A]">{error}</p>
          </div>
        ) : !isMounted ? null : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 8, right: 24, left: 12, bottom: 8 }}
            >
              <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" horizontal={false} />
              <XAxis
                type="number"
                stroke="#D9D8D0"
                tick={{ fill: '#6B6B66', fontSize: 11 }}
                domain={['auto', 'auto']}
              />
              <YAxis
                type="category"
                dataKey="channel"
                stroke="#D9D8D0"
                tick={{ fill: '#1C1C1A', fontSize: 11, fontWeight: 500 }}
                width={42}
              />
              <ReferenceLine x={0} stroke="#D9D8D0" strokeWidth={1} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload as ChannelWeightItem;
                    const isPos = item.weight >= 0;
                    return (
                      <div className="bg-[#FFFFFF] border border-[#D9D8D0] p-2.5 rounded-xl shadow-md text-xs font-mono text-[#1C1C1A]">
                        <div className="text-[#6B6B66] mb-0.5">
                          Channel: <span className="font-semibold">{item.channel}</span>
                        </div>
                        <div className="font-semibold">
                          Coefficient: {item.weight > 0 ? `+${item.weight}` : item.weight}
                        </div>
                        <div className="text-[10px] text-[#6B6B66] mt-0.5">
                          Bias: {isPos ? 'Right target (+)' : 'Left target (-)'}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="weight" radius={[3, 3, 3, 3]} isAnimationActive={false}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.weight >= 0 ? '#3A5244' : '#2563EB'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Explanatory Note */}
      <div className="p-3.5 rounded-xl bg-[#EDECE6] border border-[#D9D8D0] text-xs text-[#6B6B66] leading-relaxed">
        <strong className="text-[#1C1C1A]">Interpretation: </strong>
        Weights represent each motor electrode&apos;s contribution to the decision hyperplane separating left vs. right reach targets. Coefficients reflect statistical separability across sensorimotor channels, not anatomical source localization.
      </div>
    </div>
  );
}
