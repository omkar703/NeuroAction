'use client';

import React, { useSyncExternalStore } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  CheckCircle2,
  Clock,
  Activity,
} from 'lucide-react';

const rmseBenchmarkData = [
  {
    name: 'Ridge Regression',
    shortName: 'Ridge',
    rmse: 47.25,
    fill: '#2E6B47',
    description: 'Regularized Trajectory Forecaster',
    reduction: '44.8% error reduction vs. Fair CV',
  },
  {
    name: 'Fair Constant Velocity',
    shortName: 'Fair CV',
    rmse: 85.65,
    fill: '#2563EB',
    description: 'Kinematic Linear Extrapolation Baseline',
    reduction: 'Linear kinematic baseline',
  },
  {
    name: 'Persistence (Zero-V)',
    shortName: 'Persistence',
    rmse: 177.51,
    fill: '#94A3B8',
    description: 'Static Zero-Velocity Baseline',
    reduction: 'Zero velocity static baseline',
  },
];

const participantTableData = [
  {
    participant: 'sub-01',
    validTrials: 40,
    ridgeRmse: 41.8,
    fairCvRmse: 109.3,
    reductionPct: 61.8,
    handNote: 'Right-dominant reach profile',
  },
  {
    participant: 'sub-02',
    validTrials: 52,
    ridgeRmse: 59.9,
    fairCvRmse: 110.3,
    reductionPct: 45.7,
    handNote: 'High kinematic variance',
  },
  {
    participant: 'sub-03',
    validTrials: 60,
    ridgeRmse: 28.7,
    fairCvRmse: 38.1,
    reductionPct: 24.6,
    handNote: 'Lowest overall Cartesian error',
  },
  {
    participant: 'sub-08',
    validTrials: 57,
    ridgeRmse: 25.2,
    fairCvRmse: 75.1,
    reductionPct: 66.5,
    handNote: 'Highest relative gain (+66.5%)',
  },
  {
    participant: 'sub-15',
    validTrials: 60,
    ridgeRmse: 65.6,
    fairCvRmse: 87.0,
    reductionPct: 24.5,
    handNote: 'Pronounced lateral curve profile',
  },
];

interface TooltipPayloadItem {
  name: string;
  value: number;
  payload: {
    name: string;
    rmse: number;
    description: string;
    reduction: string;
    fill: string;
  };
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
}

function CustomBarTooltip({ active, payload }: CustomTooltipProps) {
  if (active && payload && payload.length > 0) {
    const item = payload[0].payload;
    return (
      <div className="rounded-xl bg-[#FFFFFF] border border-[#D9D8D0] p-3 shadow-md text-xs font-body text-[#1C1C1A]">
        <p className="font-bold font-display">{item.name}</p>
        <p className="text-sm font-bold text-[#1C1C1A] mt-0.5 font-display">
          {item.rmse.toFixed(2)}{' '}
          <span className="text-xs font-normal text-[#6B6B66] font-body">mm RMSE</span>
        </p>
        <p className="text-[11px] text-[#6B6B66] mt-0.5 font-body">{item.description}</p>
        <div className="mt-1 pt-1 border-t border-[#D9D8D0] text-[11px] font-semibold text-[#3A5244] font-body">
          {item.reduction}
        </div>
      </div>
    );
  }
  return null;
}

export default function EvidencePage() {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  return (
    <div className="w-full max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20 space-y-16 font-body">
      {/* Editorial Header */}
      <section className="space-y-4 border-b border-[#D9D8D0] pb-8">
        <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">
          Empirical Evidence &bull; 5-Fold LOSO Evaluation
        </p>

        <div className="space-y-2 max-w-3xl">
          <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#1C1C1A]">
            Results, Benchmarks &amp; Scientific Evidence
          </h1>
          <p className="text-base sm:text-lg text-[#6B6B66] leading-relaxed font-body">
            Quantitative benchmarks from retrospective Leave-One-Subject-Out evaluation, participant-level error distributions, condition discrimination outcomes, and scientific boundary definitions.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2 text-xs font-body text-[#6B6B66]">
          <span>Cohort: 269 Valid Reach Trials (5 Participants)</span>
          <span>&bull;</span>
          <span>11 Excluded Trials</span>
          <span>&bull;</span>
          <span>Level 2: Exploratory</span>
        </div>
      </section>

      {/* 1. Cohort Endpoint Error Comparison */}
      <section className="space-y-6">
        <div>
          <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">Trajectory Benchmarks</p>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
            Kinematic Trajectory Forecasting Benchmarks
          </h2>
          <p className="mt-1 text-sm text-[#6B6B66] font-body">
            Mean 3D Euclidean endpoint error across 269 valid reach trials (200 ms prefix window).
          </p>
        </div>

        {/* 4 Stat Boxes */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-sm">
            <span className="text-[11px] uppercase tracking-wider text-[#AEADA4] font-body block">
              Ridge (Proposed)
            </span>
            <div className="font-display text-3xl font-bold text-[#2E6B47] mt-1">
              47.25 <span className="text-sm font-semibold text-[#6B6B66] font-body">mm</span>
            </div>
            <p className="text-xs text-[#6B6B66] mt-1 font-body">
              -44.8% error vs. Fair CV
            </p>
          </div>

          <div className="p-5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-sm">
            <span className="text-[11px] uppercase tracking-wider text-[#AEADA4] font-body block">
              Fair Constant Velocity
            </span>
            <div className="font-display text-3xl font-bold text-[#1C1C1A] mt-1">
              85.65 <span className="text-sm font-semibold text-[#6B6B66] font-body">mm</span>
            </div>
            <p className="text-xs text-[#6B6B66] mt-1 font-body">
              Linear prefix extrapolation
            </p>
          </div>

          <div className="p-5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-sm">
            <span className="text-[11px] uppercase tracking-wider text-[#AEADA4] font-body block">
              Persistence (Zero-V)
            </span>
            <div className="font-display text-3xl font-bold text-[#6B6B66] mt-1">
              177.51 <span className="text-sm font-semibold text-[#6B6B66] font-body">mm</span>
            </div>
            <p className="text-xs text-[#6B6B66] mt-1 font-body">
              Static reference baseline
            </p>
          </div>

          <div className="p-5 bg-[#FFFFFF] border border-[#D9D8D0] rounded-2xl shadow-sm">
            <span className="text-[11px] uppercase tracking-wider text-[#AEADA4] font-body block">
              Relative Gain
            </span>
            <div className="font-display text-3xl font-bold text-[#3A5244] mt-1">
              44.8%
            </div>
            <p className="text-xs text-[#6B6B66] mt-1 font-body">
              (85.65 - 47.25) / 85.65
            </p>
          </div>
        </div>

        {/* Recharts Bar Chart */}
        <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-[#D9D8D0] pb-3">
            <div>
              <h3 className="font-display text-base font-bold text-[#1C1C1A]">
                3D Cartesian Endpoint Error Comparison (RMSE, mm)
              </h3>
              <p className="text-xs text-[#6B6B66] font-body">
                Lower is better. Computed across all N = 269 valid trials under 5-fold LOSO.
              </p>
            </div>
            <div className="hidden sm:flex items-center gap-3 text-xs font-body text-[#6B6B66]">
              <span className="flex items-center gap-1.5 font-medium">
                <span className="w-2.5 h-2.5 rounded-full bg-[#2E6B47]" /> Ridge
              </span>
              <span className="flex items-center gap-1.5 font-medium">
                <span className="w-2.5 h-2.5 rounded-full bg-[#2563EB]" /> Fair CV
              </span>
              <span className="flex items-center gap-1.5 font-medium">
                <span className="w-2.5 h-2.5 rounded-full bg-[#94A3B8]" /> Persistence
              </span>
            </div>
          </div>

          <div className="w-full h-64 bg-[#FAF9F5] border border-[#D9D8D0] rounded-xl p-3">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={rmseBenchmarkData}
                  margin={{ top: 16, right: 30, left: 0, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#EDECE6" vertical={false} />
                  <XAxis
                    dataKey="shortName"
                    stroke="#D9D8D0"
                    tick={{ fill: '#1C1C1A', fontSize: 12, fontWeight: 600 }}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#D9D8D0"
                    tick={{ fill: '#6B6B66', fontSize: 11 }}
                    tickLine={false}
                    unit=" mm"
                    domain={[0, 200]}
                  />
                  <Tooltip content={<CustomBarTooltip />} />
                  <Bar
                    dataKey="rmse"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={56}
                    isAnimationActive={false}
                  >
                    {rmseBenchmarkData.map((entry) => (
                      <Cell key={`cell-${entry.name}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : null}
          </div>

          <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs text-[#6B6B66] font-body">
            <span className="font-semibold text-[#1C1C1A]">
              Ridge achieves 44.8% lower endpoint error than the fair constant-velocity baseline
            </span>
            <span className="text-[11px]">
              Evaluated across 269 valid trials (N = 5 participants)
            </span>
          </div>
        </div>
      </section>

      {/* 2. Participant Breakdown Table */}
      <section className="space-y-4">
        <div>
          <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">Cohort Evaluation</p>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
            Participant-Level Cross-Validation Folds
          </h2>
          <p className="mt-1 text-sm text-[#6B6B66] font-body">
            Leave-One-Subject-Out (LOSO) performance across each held-out participant.
          </p>
        </div>

        <div className="border border-[#D9D8D0] rounded-2xl overflow-hidden bg-[#FFFFFF] shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-[#1C1C1A] font-body">
              <thead className="bg-[#FAF9F5] text-xs font-semibold uppercase tracking-wider text-[#6B6B66] border-b border-[#D9D8D0]">
                <tr>
                  <th scope="col" className="px-5 py-3.5">Participant</th>
                  <th scope="col" className="px-5 py-3.5 text-center">Valid Trials</th>
                  <th scope="col" className="px-5 py-3.5 text-right">Ridge RMSE</th>
                  <th scope="col" className="px-5 py-3.5 text-right">Fair CV RMSE</th>
                  <th scope="col" className="px-5 py-3.5 text-right">Error Reduction</th>
                  <th scope="col" className="px-5 py-3.5 hidden md:table-cell">Kinematic Profile</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#D9D8D0]">
                {participantTableData.map((row) => (
                  <tr key={row.participant} className="hover:bg-[#FAF9F5] transition-colors">
                    <td className="px-5 py-3.5 font-bold text-[#1C1C1A] font-display">
                      {row.participant}
                    </td>
                    <td className="px-5 py-3.5 text-center text-[#6B6B66]">
                      {row.validTrials}
                    </td>
                    <td className="px-5 py-3.5 text-right font-bold text-[#2E6B47] font-display">
                      {row.ridgeRmse.toFixed(1)} mm
                    </td>
                    <td className="px-5 py-3.5 text-right text-[#2563EB] font-display font-semibold">
                      {row.fairCvRmse.toFixed(1)} mm
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#E4EDE6] text-[#3A5244] border border-[#3A5244]/30">
                        +{row.reductionPct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-5 py-3.5 hidden md:table-cell text-xs text-[#6B6B66]">
                      {row.handNote}
                    </td>
                  </tr>
                ))}
                {/* Cohort Mean Summary Row */}
                <tr className="bg-[#EDECE6] font-bold border-t border-[#D9D8D0]">
                  <td className="px-5 py-3.5 text-[#1C1C1A] uppercase tracking-wider font-display">
                    Cohort Mean
                  </td>
                  <td className="px-5 py-3.5 text-center text-[#1C1C1A]">
                    269
                  </td>
                  <td className="px-5 py-3.5 text-right text-[#2E6B47] font-display">
                    47.25 mm
                  </td>
                  <td className="px-5 py-3.5 text-right text-[#2563EB] font-display">
                    85.65 mm
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#E4EDE6] text-[#3A5244] border border-[#3A5244]/30">
                      +44.8%
                    </span>
                  </td>
                  <td className="px-5 py-3.5 hidden md:table-cell text-xs text-[#6B6B66] font-normal">
                    Weighted average across 5 benchmark folds
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 3. Dataset & Experimental Architecture */}
      <section className="space-y-4">
        <div>
          <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">Technical Specs</p>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
            Dataset &amp; Experimental Architecture
          </h2>
          <p className="mt-1 text-sm text-[#6B6B66] font-body">
            Specifications of the OpenNeuro ds006840 benchmark and signal processing pipeline.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs text-[#6B6B66]">
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">OpenNeuro ds006840 (IACKD)</strong>
            <span className="font-body">5 healthy participants (sub-01, sub-02, sub-03, sub-08, sub-15) performing reaching motions.</span>
          </div>
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">Scalp EEG (26 Channels, 200 Hz)</strong>
            <span className="font-body">Analysis window: [-1000, 0] ms pre-onset capturing sensorimotor mu and beta rhythm desynchronization.</span>
          </div>
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">3D Optical Kinematics (100 Hz)</strong>
            <span className="font-body">Leap Motion sensor recording 3D hand path. Initial 200 ms prefix used for trajectory forecasting.</span>
          </div>
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">Model Architectures</strong>
            <span className="font-body">LDA with spatial CSP for neural goal decoding; regularized Ridge regression for 3D trajectory forecasting.</span>
          </div>
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">Cross-Validation</strong>
            <span className="font-body">Strict 5-fold Leave-One-Subject-Out protocol ensuring zero data leakage between training and testing folds.</span>
          </div>
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1.5 shadow-sm">
            <strong className="text-[#1C1C1A] block font-display font-bold text-xs">Trial Exclusions</strong>
            <span className="font-body">11 trials excluded (from 280) due to optical sensor dropouts or prefix &lt; 100 ms, leaving 269 valid trials.</span>
          </div>
        </div>
      </section>

      {/* 5. Three-Part Honest Scientific Summary */}
      <section className="space-y-4">
        <div>
          <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">Epistemic Status</p>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
            Three-Part Honest Scientific Summary
          </h2>
          <p className="mt-1 text-sm text-[#6B6B66] font-body">
            Demarcation of verified software, retrospective findings, and pending validation milestones.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
            <div className="flex items-center gap-1.5 font-bold font-display text-sm text-[#1C1C1A] border-b border-[#D9D8D0] pb-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#3A5244]" />
              <span>1. Implemented</span>
            </div>
            <p className="text-[#6B6B66] leading-relaxed font-body">
              Full offline pipeline (Phases 2–5), local ML model execution, LOSO evaluation, deterministic trial replay, and 3D WebGL visualization.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-[#3A5244] font-body">
              Status: Verified Software Pipeline
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
            <div className="flex items-center gap-1.5 font-bold font-display text-sm text-[#1C1C1A] border-b border-[#D9D8D0] pb-2.5">
              <Activity className="w-4 h-4 text-[#2563EB]" />
              <span>2. Demonstrated</span>
            </div>
            <p className="text-[#6B6B66] leading-relaxed font-body">
              Ridge trajectory RMSE 47.25 mm (44.8% error reduction vs. Fair CV), Phase 4 condition discrimination 66.46%, pre-movement EEG goal decoding.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-[#2563EB] font-body">
              Status: Retrospective Existing-Data Evidence
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
            <div className="flex items-center gap-1.5 font-bold font-display text-sm text-[#1C1C1A] border-b border-[#D9D8D0] pb-2.5">
              <Clock className="w-4 h-4 text-[#6B6B66]" />
              <span>3. Pending</span>
            </div>
            <p className="text-[#6B6B66] leading-relaxed font-body">
              Independent biological outcome prediction, prospective external hospital cohort validation, real-time causal closed-loop acquisition.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-[#6B6B66] font-body">
              Status: Pending Clinical Validation
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
