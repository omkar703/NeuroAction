'use client';

import React, { useState, lazy, Suspense } from 'react';
import Link from 'next/link';
import {
  ChevronRight, Radio, Brain,
  Crosshair, GitCompare, TrendingDown, Database, Zap,
} from 'lucide-react';

const HeroWideStage = lazy(() => import('@/components/viz/HeroWideStage'));

/* ─── Data ───────────────────────────────────────────────────────────────── */
const pipelineStages = [
  {
    num: '01', title: 'Pre-Movement Scalp EEG', shortTitle: 'EEG Acquisition',
    role: 'Signal Conditioning', icon: Radio, tag: 'Neural Input',
    spec: '26 Channels · 200 Hz', window: '[-1000, 0] ms pre-movement window',
    summary: 'Raw potentials over motor and premotor cortex are filtered to isolate sensorimotor mu (8–12 Hz) and beta (13–30 Hz) rhythm desynchronization.',
  },
  {
    num: '02', title: 'Spatial Filtering & Goal Decoding', shortTitle: 'Goal Decoding',
    role: 'Pattern Classification', icon: Brain, tag: 'Neural Decoder',
    spec: 'Common Spatial Patterns + LDA', window: 'Left vs. Right target classification',
    summary: 'Decomposes multi-channel EEG into spatial filters that maximise variance between reach directions prior to overt movement onset.',
  },
  {
    num: '03', title: 'Kinematic Trajectory Forecasting', shortTitle: '3D Forecasting',
    role: 'Trajectory Extrapolation', icon: Crosshair, tag: 'Kinematic Model',
    spec: 'Ridge Regression (47.25 mm RMSE)', window: '200 ms prefix (20 kinematic samples)',
    summary: 'Extrapolates 3D hand coordinates to predict final reach endpoint from the initial 200 ms of Leap Motion optical kinematics.',
  },
  {
    num: '04', title: 'Intention–Action Congruence', shortTitle: 'Congruence Audit',
    role: 'Cross-Modal Assessment', icon: GitCompare, tag: 'Validation',
    spec: '66.46% BalAcc · 0.6834 AUROC', window: '5-fold LOSO cross-validation',
    summary: 'Monitors spatial consistency between decoded neural goal and overt movement trajectory under Leave-One-Subject-Out evaluation.',
  },
];

const empiricalMetrics = [
  { value: '269',   unit: '',    label: 'Valid Reach Trials',       sub: '5-participant LOSO cohort',              icon: Database,   fg: '#3A5244', bg: 'rgba(58,82,68,0.08)'     },
  { value: '47.25', unit: ' mm', label: 'Endpoint RMSE',           sub: '−44.8% vs. kinematic baseline',          icon: TrendingDown, fg: '#2E6B47', bg: 'rgba(46,107,71,0.08)'    },
  { value: '66.46', unit: '%',   label: 'Goal Balanced Accuracy',  sub: '0.6834 AUROC · 5-fold LOSO',             icon: Brain,      fg: '#2563EB', bg: 'rgba(37,99,235,0.08)'    },
  { value: '200',   unit: ' ms', label: 'Forecast Lead Time',      sub: 'Prefix window before reach completion',  icon: Zap,        fg: '#6B46C1', bg: 'rgba(107,70,193,0.08)'   },
];

const exploreLinks = [
  { title: 'Scientific Foundations',  href: '/science',        description: 'Sensorimotor rhythm desynchronisation, reach paradigm, and trajectory theory.' },
  { title: 'Model Lab & Benchmarks',  href: '/model-lab',      description: 'Ridge regression hyperparameters, LOSO fold results, and live trial replay.' },
  { title: '3D Digital Twin',         href: '/digital-twin',   description: 'WebGL workspace showing reach kinematics, forecast cones, and decoded intention.' },
  { title: 'Signal Explainability',   href: '/explainability', description: '26-channel scalp maps, CSP eigenvalue distributions, and feature importance.' },
  { title: 'Evidence & Verification', href: '/evidence',       description: 'Per-participant metrics, data provenance audit, and scientific boundaries.' },
];

/* ─── Component ──────────────────────────────────────────────────────────── */
export default function OverviewPage() {
  const [activeIdx, setActiveIdx] = useState(0);
  const stage = pipelineStages[activeIdx];
  const ActiveIcon = stage.icon;

  return (
    <div className="w-full">

      {/* ═══════════════════════════════════════════════════
          HERO — split layout: text left / EEG animation right
      ════════════════════════════════════════════════════ */}
      {/* ═══════════════════════════════════════════════════
          HERO — Wide 3D Stage with Frosted Blur & Foreground Copy
      ════════════════════════════════════════════════════ */}
      <Suspense
        fallback={
          <section className="relative w-full min-h-[calc(100vh-4rem)] bg-[#F5F3EE] border-b border-[#D9D8D0] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-[#3A5244] border-t-transparent rounded-full animate-spin" />
          </section>
        }
      >
        <HeroWideStage />
      </Suspense>

      {/* ═══════════════════════════════════════════════════
          ABSTRACT
      ════════════════════════════════════════════════════ */}
      <section className="w-full border-b border-[#D9D8D0] bg-[#FAF9F5]">
        <div className="max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-10 items-start">
            <div className="md:col-span-4">
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A] leading-snug">
                The Research<br />Question
              </h2>
            </div>
            <div className="md:col-span-8 space-y-5">
              <p className="text-base text-[#1C1C1A] leading-[1.85] font-body">
                In assistive neuroprosthetics and human–robot collaboration, anticipating reach goals
                before motion concludes can substantially reduce interface latency. Non-invasive scalp
                EEG suffers from high trial-to-trial variability and low SNR, making pre-movement
                decoding exceptionally challenging.
              </p>
              <p className="text-base text-[#6B6B66] leading-[1.85] font-body">
                NeuroAction investigates whether sensorimotor rhythm desynchronisation
                (mu 8–12 Hz; beta 13–30 Hz) in the 1,000&nbsp;ms preceding movement onset
                encodes reliable directional information, and whether early kinematic observations
                combined with Ridge regularisation can forecast complete 3D reach endpoints.
              </p>
              <div className="pl-5 border-l-2 border-[#3A5244]">
                <p className="text-sm text-[#1C1C1A] leading-relaxed font-body">
                  <strong className="font-semibold">Core finding:</strong>{' '}
                  Ridge trajectory forecasting achieved 47.25&nbsp;mm endpoint RMSE — a
                  44.8% reduction versus the constant-velocity baseline (85.65&nbsp;mm)
                  across 5 participants under LOSO cross-validation.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════
          PIPELINE — interactive stage explorer
      ════════════════════════════════════════════════════ */}
      <section className="w-full border-b border-[#D9D8D0] bg-[#F5F3EE]">
        <div className="max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20">

          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-12">
            <div>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
                Methodological Architecture
              </h2>
            </div>
            <p className="text-sm text-[#6B6B66] italic font-body self-end">Stage {activeIdx + 1} of 4</p>
          </div>

          {/* Stage tabs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            {pipelineStages.map((s, idx) => {
              const Icon = s.icon;
              const active = activeIdx === idx;
              return (
                <button
                  key={s.num}
                  type="button"
                  onClick={() => setActiveIdx(idx)}
                  className={`group text-left rounded-2xl border p-4 transition-all duration-200 cursor-pointer ${
                    active
                      ? 'bg-[#1C1C1A] border-[#1C1C1A] text-[#F5F3EE] shadow-lg'
                      : 'bg-[#FFFFFF] border-[#D9D8D0] hover:border-[#AEADA4] hover:shadow-md'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      active ? 'bg-[#3A5244] text-white' : 'bg-[#EDECE6] text-[#1C1C1A] group-hover:bg-[#D9D8D0]'
                    }`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className={`text-[10px] font-semibold tracking-[0.2em] uppercase font-body ${
                      active ? 'text-[#9DB5A2]' : 'text-[#AEADA4]'
                    }`}>{s.num}</span>
                  </div>
                  <div className={`text-[10px] uppercase tracking-wider mb-1 font-body ${active ? 'text-[#9DB5A2]' : 'text-[#AEADA4]'}`}>
                    {s.role}
                  </div>
                  <div className={`text-sm font-bold font-display leading-snug ${active ? 'text-white' : 'text-[#1C1C1A]'}`}>
                    {s.shortTitle}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Stage detail panel */}
          <div className="rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] shadow-sm overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#D9D8D0] bg-[#FAF9F5]">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-[#1C1C1A] text-[#F5F3EE] flex items-center justify-center">
                  <ActiveIcon className="w-4 h-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold tracking-[0.15em] uppercase text-[#3A5244] font-body">
                      Stage {stage.num}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#EDECE6] text-[#6B6B66] font-body border border-[#D9D8D0]">
                      {stage.tag}
                    </span>
                  </div>
                  <h3 className="font-display text-lg font-bold text-[#1C1C1A] leading-snug mt-0.5">
                    {stage.title}
                  </h3>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={activeIdx === 0}
                  onClick={() => setActiveIdx(p => Math.max(0, p - 1))}
                  className="px-4 py-1.5 rounded-full text-xs font-semibold font-body border border-[#D9D8D0] text-[#6B6B66] hover:bg-[#EDECE6] disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
                >
                  ← Prev
                </button>
                <button
                  type="button"
                  disabled={activeIdx === pipelineStages.length - 1}
                  onClick={() => setActiveIdx(p => Math.min(pipelineStages.length - 1, p + 1))}
                  className="px-4 py-1.5 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] text-[#F5F3EE] hover:bg-[#3A5244] disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
                >
                  Next →
                </button>
              </div>
            </div>
            <div className="px-6 py-6">
              <p className="text-base text-[#6B6B66] leading-[1.8] mb-6 max-w-2xl font-body">{stage.summary}</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[{label: 'Specification', value: stage.spec}, {label: 'Observation Window', value: stage.window}].map(f => (
                  <div key={f.label} className="rounded-xl bg-[#F5F3EE] border border-[#D9D8D0] px-5 py-4">
                    <p className="text-[10px] uppercase tracking-[0.2em] text-[#AEADA4] mb-1.5 font-body">{f.label}</p>
                    <p className="text-sm font-bold text-[#1C1C1A] leading-snug font-display">{f.value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════
          METRICS
      ════════════════════════════════════════════════════ */}
      <section className="w-full border-b border-[#D9D8D0] bg-[#FAF9F5]">
        <div className="max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20">
          <div className="mb-12">
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">Key Findings</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-px bg-[#D9D8D0] rounded-2xl overflow-hidden border border-[#D9D8D0]">
            {empiricalMetrics.map((m, i) => {
              const Icon = m.icon;
              return (
                <div key={m.label} className="group bg-[#FFFFFF] px-6 py-8 flex flex-col justify-between hover:bg-[#F5F3EE] transition-colors duration-200 relative overflow-hidden">
                  <div
                    aria-hidden
                    className="absolute -top-6 -right-6 w-24 h-24 rounded-full blur-2xl opacity-50 group-hover:opacity-80 transition-opacity"
                    style={{ background: m.bg }}
                  />
                  <div className="flex items-start justify-between mb-8">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: m.bg, color: m.fg }}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-3xl font-bold opacity-[0.06] font-display" style={{ color: m.fg }}>0{i + 1}</span>
                  </div>
                  <div>
                    <div className="text-4xl sm:text-5xl font-black tracking-tight leading-none mb-1 font-display" style={{ color: m.fg }}>
                      {m.value}<span className="text-2xl font-bold">{m.unit}</span>
                    </div>
                    <div className="text-sm font-bold text-[#1C1C1A] mt-3 font-display">{m.label}</div>
                    <div className="text-xs text-[#6B6B66] mt-1 leading-snug font-body italic">{m.sub}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════
          LIMITATIONS
      ════════════════════════════════════════════════════ */}
      <section className="w-full border-b border-[#D9D8D0] bg-[#F5F3EE]">
        <div className="max-w-6xl mx-auto px-6 lg:px-10 py-14">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
            <div className="md:col-span-3">
              <h2 className="font-display text-xl font-bold text-[#1C1C1A]">Limitations</h2>
            </div>
            <div className="md:col-span-9 grid grid-cols-1 sm:grid-cols-2 gap-5">
              {[
                { title: 'Retrospective Analysis', body: 'All evaluations use the open-access OpenNeuro ds006840 dataset. Leap Motion tracking drops excluded 11 trials from kinematic benchmarks.' },
                { title: 'Condition Discrimination', body: 'High classification accuracy reflects stimulus-response congruency rules, not real-time detection of spontaneous biological motor errors.' },
              ].map(c => (
                <div key={c.title} className="rounded-2xl border border-[#D9D8D0] bg-[#FFFFFF] p-5">
                  <h3 className="text-sm font-bold text-[#1C1C1A] mb-2 font-display">{c.title}</h3>
                  <p className="text-xs text-[#6B6B66] leading-relaxed font-body">{c.body}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════
          EXPLORE LINKS
      ════════════════════════════════════════════════════ */}
      <section className="w-full bg-[#FAF9F5]">
        <div className="max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20">
          <div className="mb-10">
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">Project Resources</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {exploreLinks.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className="group rounded-2xl border border-[#D9D8D0] bg-[#FFFFFF] p-6 hover:bg-[#F5F3EE] hover:border-[#AEADA4] hover:shadow-md transition-all duration-200 flex flex-col justify-between min-h-[140px]"
              >
                <div>
                  <h3 className="font-display text-base font-bold text-[#1C1C1A] group-hover:text-[#3A5244] transition-colors mb-2">{item.title}</h3>
                  <p className="text-xs text-[#6B6B66] leading-relaxed font-body">{item.description}</p>
                </div>
                <div className="mt-4 inline-flex items-center text-xs font-semibold text-[#3A5244] group-hover:translate-x-1 transition-transform font-body">
                  View <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
