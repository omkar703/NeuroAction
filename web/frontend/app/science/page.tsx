'use client';

import React, { useState, useEffect, useRef, useMemo, useSyncExternalStore } from 'react';
import Link from 'next/link';
import { useInView } from 'framer-motion';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Cell,
  Legend,
} from 'recharts';
import {
  Activity,
  TrendingUp,
  BarChart2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Info,
} from 'lucide-react';

const STAGES = [
  { id: 'stage-1', num: 1, title: 'The EEG Recording', short: 'EEG Recording' },
  { id: 'stage-2', num: 2, title: 'Neural Goal Decoding', short: 'Goal Decoding' },
  { id: 'stage-3', num: 3, title: 'Hand Movement', short: 'Hand Movement' },
  { id: 'stage-4', num: 4, title: 'Trajectory Forecasting', short: 'Forecasting' },
  { id: 'stage-5', num: 5, title: 'Model Comparison', short: 'Model Comparison' },
  { id: 'stage-6', num: 6, title: 'Limitations & Interpretation', short: 'Limitations' },
];

export default function SciencePage() {
  const [activeStage, setActiveStage] = useState<number>(1);
  const isMounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  const scrollToStage = (stageNum: number) => {
    const el = document.getElementById(`stage-${stageNum}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setActiveStage(stageNum);
    }
  };

  const handlePrev = () => {
    if (activeStage > 1) {
      scrollToStage(activeStage - 1);
    }
  };

  const handleNext = () => {
    if (activeStage < STAGES.length) {
      scrollToStage(activeStage + 1);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-6 lg:px-10 py-16 sm:py-20 space-y-12 font-body">
      {/* Editorial Header */}
      <div className="border-b border-[#D9D8D0] pb-8 space-y-3">
        <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-2 font-body">
          Methodological Architecture &bull; 6 Pedagogical Stages
        </p>
        <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#1C1C1A]">
          The Science of Neural Trajectory Forecasting
        </h1>
        <p className="text-base sm:text-lg text-[#6B6B66] max-w-3xl leading-relaxed font-body">
          Electrophysiological mechanisms, mathematical decoders, kinematic extrapolation, and empirical validation boundaries.
        </p>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
        {/* Sticky Sidebar Progress */}
        <aside className="lg:col-span-3 sticky top-20 z-20 space-y-4">
          <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0] text-xs font-body">
              <span className="font-semibold uppercase tracking-wider text-[#6B6B66]">
                Stages
              </span>
              <span className="text-[#3A5244] font-bold font-display">
                {activeStage} / {STAGES.length}
              </span>
            </div>

            {/* Stage links */}
            <div className="mt-3 space-y-1">
              {STAGES.map((s) => {
                const isActive = activeStage === s.num;
                return (
                  <button
                    key={s.id}
                    onClick={() => scrollToStage(s.num)}
                    className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-left text-xs transition-colors cursor-pointer font-body ${
                      isActive
                        ? 'bg-[#EDECE6] text-[#1C1C1A] font-semibold border-l-2 border-[#3A5244]'
                        : 'text-[#6B6B66] hover:text-[#1C1C1A] hover:bg-[#EDECE6]/50'
                    }`}
                  >
                    <span className="font-display text-[11px] text-[#AEADA4] w-4 font-bold">
                      {s.num}.
                    </span>
                    <span className="truncate">{s.short}</span>
                  </button>
                );
              })}
            </div>

            {/* Prev / Next Stage Buttons */}
            <div className="mt-4 pt-3 border-t border-[#D9D8D0] flex items-center justify-between gap-2">
              <button
                onClick={handlePrev}
                disabled={activeStage <= 1}
                className="flex-1 inline-flex items-center justify-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold font-body border border-[#D9D8D0] text-[#6B6B66] hover:bg-[#EDECE6] disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Prev</span>
              </button>
              <button
                onClick={handleNext}
                disabled={activeStage >= STAGES.length}
                className="flex-1 inline-flex items-center justify-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] text-[#F5F3EE] hover:bg-[#3A5244] disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                <span>Next</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Quick Link Card */}
          <div className="p-5 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0] text-xs text-[#6B6B66] space-y-2">
            <span className="font-bold text-[#1C1C1A] block font-display">Raw Signal Explorer</span>
            <p className="text-xs leading-relaxed font-body">
              Inspect multi-channel voltage traces and spatial LDA coefficients directly.
            </p>
            <Link
              href="/explainability"
              className="inline-flex items-center gap-1 text-[#3A5244] hover:text-[#2D4036] font-semibold pt-1 font-body"
            >
              <span>Explainability Suite</span>
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </aside>

        {/* Right Column: 6 Sections */}
        <main className="lg:col-span-9 space-y-14 pb-16">
          {/* STAGE 1 */}
          <StageSection
            stageNum={1}
            onInView={() => setActiveStage(1)}
            title="Stage 1 — The EEG Recording"
            subtitle="Scalp electrophysiology preceding motor onset"
          >
            <Stage1Content isMounted={isMounted} />
          </StageSection>

          {/* STAGE 2 */}
          <StageSection
            stageNum={2}
            onInView={() => setActiveStage(2)}
            title="Stage 2 — Neural Goal Decoding"
            subtitle="Linear Discriminant Analysis mapping pre-movement EEG to instructed direction"
          >
            <Stage2Content />
          </StageSection>

          {/* STAGE 3 */}
          <StageSection
            stageNum={3}
            onInView={() => setActiveStage(3)}
            title="Stage 3 — Hand Movement"
            subtitle="High-speed optical tracking of reach kinematics and congruency conditions"
          >
            <Stage3Content />
          </StageSection>

          {/* STAGE 4 */}
          <StageSection
            stageNum={4}
            onInView={() => setActiveStage(4)}
            title="Stage 4 — Trajectory Forecasting"
            subtitle="Early kinematic prefix extrapolation to predicted 3D reach endpoints"
          >
            <Stage4Content isMounted={isMounted} />
          </StageSection>

          {/* STAGE 5 */}
          <StageSection
            stageNum={5}
            onInView={() => setActiveStage(5)}
            title="Stage 5 — Model Comparison"
            subtitle="Leave-One-Subject-Out (LOSO) benchmark evaluation across kinematic baselines"
          >
            <Stage5Content isMounted={isMounted} />
          </StageSection>

          {/* STAGE 6 */}
          <StageSection
            stageNum={6}
            onInView={() => setActiveStage(6)}
            title="Stage 6 — Limitations & Interpretation"
            subtitle="Scientific claims audit and boundary taxonomy"
          >
            <Stage6Content />
          </StageSection>
        </main>
      </div>
    </div>
  );
}

// Stage Section Wrapper
interface StageSectionProps {
  stageNum: number;
  title: string;
  subtitle: string;
  onInView: () => void;
  children: React.ReactNode;
}

function StageSection({ stageNum, title, subtitle, onInView, children }: StageSectionProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { amount: 0.3, once: false });

  useEffect(() => {
    if (inView) {
      onInView();
    }
  }, [inView, onInView]);

  return (
    <div id={`stage-${stageNum}`} ref={ref} className="scroll-mt-20">
      <section className="rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] p-6 sm:p-8 space-y-6 shadow-sm">
        <div className="pb-4 border-b border-[#D9D8D0] space-y-1">
          <p className="text-[11px] tracking-[0.3em] uppercase text-[#AEADA4] mb-1 font-body">
            Stage 0{stageNum}
          </p>
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#1C1C1A]">
            {title}
          </h2>
          <p className="text-xs text-[#6B6B66] font-body">
            {subtitle}
          </p>
        </div>

        {children}
      </section>
    </div>
  );
}

// STAGE 1 CONTENT
const MOTOR_ELECTRODES = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4'];

function Stage1Content({ isMounted }: { isMounted: boolean }) {
  const schematicData = useMemo(() => {
    const data = [];
    for (let i = 0; i <= 100; i++) {
      const t = -1000 + i * 10;
      const sec = t / 1000;
      const readiness = -5.0 * (1 + sec);
      const mu = 12 * (0.4 - 0.6 * sec) * Math.sin(2 * Math.PI * 10 * sec);
      const beta = 4.5 * Math.cos(2 * Math.PI * 22 * sec);
      const uv = parseFloat((readiness + mu + beta).toFixed(2));
      data.push({ time_ms: t, amplitude_uv: uv });
    }
    return data;
  }, []);

  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed text-[#1C1C1A]">
        Scalp electroencephalography records cortical potentials during the{' '}
        <span className="font-semibold text-[#1C1C1A]">[-1000, 0] ms</span> pre-movement window.
        As a participant prepares a goal-directed reach, sensorimotor rhythms exhibit
        event-related desynchronization (ERD) in the mu (8–12 Hz) and beta (13–30 Hz) bands alongside
        a slow negative Bereitschaftspotential (readiness potential).
      </p>

      {/* Schematic EEG Trace */}
      <div className="p-5 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0] space-y-3">
        <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0]">
          <span className="text-xs font-bold text-[#1C1C1A] uppercase tracking-wider flex items-center gap-1.5 font-display">
            <Activity className="w-3.5 h-3.5 text-[#3A5244]" />
            Illustrative Pre-Movement EEG Waveform (Cz)
          </span>
          <span className="text-[11px] text-[#6B6B66] font-body">
            Pedagogical simulation (-1000 to 0 ms)
          </span>
        </div>

        <div className="w-full h-52 select-none">
          {isMounted ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={schematicData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="time_ms"
                  type="number"
                  domain={[-1000, 0]}
                  unit=" ms"
                  stroke="#D9D8D0"
                  tick={{ fill: '#6B6B66', fontSize: 11 }}
                />
                <YAxis
                  unit=" μV"
                  stroke="#D9D8D0"
                  tick={{ fill: '#6B6B66', fontSize: 11 }}
                  domain={[-25, 25]}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-[#FFFFFF] border border-[#D9D8D0] p-3 rounded-xl text-xs shadow-md font-body text-[#1C1C1A]">
                          <div className="text-[#6B6B66]">Time: {label} ms</div>
                          <div className="font-semibold text-[#2563EB]">
                            Amplitude: {payload[0].value} μV
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="amplitude_uv"
                  stroke="#2563EB"
                  strokeWidth={1.8}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-[#6B6B66]">
              Loading EEG schematic...
            </div>
          )}
        </div>
      </div>

      {/* Motor Channel Array Badges */}
      <div className="space-y-2">
        <span className="text-xs font-bold text-[#1C1C1A] block font-display">
          Primary Sensorimotor Electrodes (9 Channels Over Motor Cortex):
        </span>
        <div className="flex flex-wrap gap-2">
          {MOTOR_ELECTRODES.map((ch) => (
            <span
              key={ch}
              className="px-3 py-1 rounded-full bg-[#EDECE6] text-[#1C1C1A] border border-[#D9D8D0] font-body text-xs font-semibold"
            >
              {ch}
            </span>
          ))}
        </div>
      </div>

      {/* Key Facts Box */}
      <div className="p-5 rounded-2xl bg-[#EDECE6] border border-[#D9D8D0] space-y-2">
        <div className="text-xs font-bold text-[#1C1C1A] uppercase tracking-wider flex items-center gap-1.5 font-display">
          <Info className="w-3.5 h-3.5 text-[#3A5244]" />
          <span>Protocol Specifications (OpenNeuro ds006840)</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#6B6B66] font-body">
          <div><strong className="text-[#1C1C1A]">Sampling rate:</strong> 200 Hz continuous digitization</div>
          <div><strong className="text-[#1C1C1A]">Analysis window:</strong> [-1000, 0] ms relative to movement onset</div>
          <div><strong className="text-[#1C1C1A]">Electrode layout:</strong> 26 scalp electrodes (standard 10–20 array)</div>
          <div><strong className="text-[#1C1C1A]">Behavioral paradigm:</strong> Reach-to-touch with congruent/incongruent cues</div>
        </div>
      </div>
    </div>
  );
}

// STAGE 2 CONTENT
function Stage2Content() {
  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed">
        A Linear Discriminant Analysis (LDA) decoder maps EEG features extracted from sensorimotor channels to an intended target direction. Features include time-domain statistics and spectral power in the mu (8–12 Hz) and beta (13–30 Hz) frequency bands.
      </p>

      {/* Diagram: [EEG Features] -> [LDA Decoder] -> [Predicted Goal] */}
      <div className="p-6 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0]">
        <span className="text-xs font-bold text-[#6B6B66] uppercase tracking-wider block mb-4 font-display">
          Classification Pipeline
        </span>
        <div className="flex flex-col md:flex-row items-center justify-center gap-3 text-center">
          <div className="w-full md:w-auto p-4 rounded-xl bg-[#FFFFFF] border border-[#D9D8D0] text-[#1C1C1A] text-xs font-bold font-display shadow-sm">
            EEG Feature Matrix
            <div className="text-[10px] text-[#6B6B66] font-body mt-0.5 font-normal">
              234 spatio-spectral features
            </div>
          </div>

          <div className="text-[#AEADA4] font-bold text-base hidden md:block">→</div>
          <div className="text-[#AEADA4] font-bold text-xs md:hidden">↓</div>

          <div className="w-full md:w-auto p-4 rounded-xl bg-[#FFFFFF] border border-[#D9D8D0] text-[#1C1C1A] text-xs font-bold font-display shadow-sm">
            LDA Classifier
            <div className="text-[10px] text-[#6B6B66] font-body mt-0.5 font-normal">
              Linear Discriminant + Shrinkage
            </div>
          </div>

          <div className="text-[#AEADA4] font-bold text-base hidden md:block">→</div>
          <div className="text-[#AEADA4] font-bold text-xs md:hidden">↓</div>

          <div className="w-full md:w-auto p-4 rounded-xl bg-[#E4EDE6] border border-[#3A5244]/30 text-[#3A5244] text-xs font-bold font-display shadow-sm">
            Target Prediction
            <div className="text-[10px] text-[#3A5244] font-body mt-0.5 font-normal">
              Left vs. Right (Binary)
            </div>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1">
          <span className="text-xs text-[#6B6B66] font-body">Target Variable Decoded:</span>
          <p className="text-xs text-[#1C1C1A] leading-relaxed font-body">
            The decoder predicts the experimental visual cue target direction prior to movement onset.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-1">
          <span className="text-xs text-[#6B6B66] font-body">Cross-Subject Generalization:</span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-[#1C1C1A] font-display">66.46%</span>
            <span className="text-xs text-[#6B6B66] font-body">5-fold LOSO balanced accuracy (AUROC: 0.6834)</span>
          </div>
        </div>
      </div>

      {/* Boundary Callout */}
      <div className="p-5 rounded-2xl bg-[#FEF3C7]/40 border border-[#D97706]/30 text-xs text-[#92400E] space-y-1 font-body">
        <div className="flex items-center gap-1.5 font-bold font-display">
          <AlertTriangle className="w-3.5 h-3.5 text-[#D97706] shrink-0" />
          <span>Decoded Visual Cue ≠ Measured Biological Intention</span>
        </div>
        <p className="leading-relaxed">
          The LDA model was trained on trial cue labels presented on screen. While neural activations correlate with instructed direction, internal willful agency cannot be inferred from scalp EEG classification accuracy alone.
        </p>
      </div>
    </div>
  );
}

// STAGE 3 CONTENT
function Stage3Content() {
  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed">
        Following the visual cue, participants reach to one of two target locations. Spatial kinematics are tracked continuously with a Leap Motion optical sensor at 100 Hz.
      </p>

      {/* Top-Down Workspace Schematic */}
      <div className="p-5 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0] flex flex-col items-center">
        <span className="text-xs font-bold text-[#6B6B66] uppercase tracking-wider mb-3 self-start font-display">
          Top-Down Workspace Schematic (Leap Motion Coordinates)
        </span>

        <svg viewBox="0 0 500 220" className="w-full max-w-lg h-auto">
          {/* Workspace boundary */}
          <rect x="20" y="20" width="460" height="180" rx="16" fill="#FFFFFF" stroke="#D9D8D0" strokeWidth="1" strokeDasharray="3 3" />

          {/* Left Target */}
          <circle cx="100" cy="70" r="22" fill="#E4EDE6" stroke="#3A5244" strokeWidth="1.5" />
          <circle cx="100" cy="70" r="5" fill="#3A5244" />
          <text x="100" y="106" fill="#1C1C1A" fontSize="11" textAnchor="middle" fontWeight="700">
            Left Target
          </text>
          <text x="100" y="120" fill="#6B6B66" fontSize="9" textAnchor="middle">
            [-188, 150, 53] mm
          </text>

          {/* Right Target */}
          <circle cx="400" cy="70" r="22" fill="#E4EDE6" stroke="#3A5244" strokeWidth="1.5" />
          <circle cx="400" cy="70" r="5" fill="#3A5244" />
          <text x="400" y="106" fill="#1C1C1A" fontSize="11" textAnchor="middle" fontWeight="700">
            Right Target
          </text>
          <text x="400" y="120" fill="#6B6B66" fontSize="9" textAnchor="middle">
            [+188, 149, 51] mm
          </text>

          {/* Hand Start */}
          <circle cx="250" cy="165" r="16" fill="#EDECE6" stroke="#AEADA4" strokeWidth="1.5" />
          <circle cx="250" cy="165" r="4" fill="#1C1C1A" />
          <text x="250" y="195" fill="#1C1C1A" fontSize="11" textAnchor="middle" fontWeight="700">
            Hand Start [0, 0, 0]
          </text>

          {/* Curved trajectory path to Left target */}
          <path
            d="M 240 152 C 220 110, 150 85, 124 73"
            fill="none"
            stroke="#2E6B47"
            strokeWidth="2"
            strokeDasharray="4 3"
          />
          <text x="175" y="132" fill="#2E6B47" fontSize="10" fontStyle="italic">
            Reach Path
          </text>
        </svg>
      </div>

      {/* Congruent vs Incongruent Description */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[#1C1C1A] font-display">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#3A5244]" />
            <span>Congruent Reach Trials</span>
          </div>
          <p className="text-xs text-[#6B6B66] leading-relaxed font-body">
            Target stimulus direction equals instructed reach direction. The participant reaches directly toward the cued target with direct sensorimotor coupling.
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[#1C1C1A] font-display">
            <AlertTriangle className="w-3.5 h-3.5 text-[#B45309]" />
            <span>Incongruent Reach Trials</span>
          </div>
          <p className="text-xs text-[#6B6B66] leading-relaxed font-body">
            Target stimulus direction is opposite to instructed reach direction (reversal rule). Demands cognitive inhibition and spatial rule remapping.
          </p>
        </div>
      </div>
    </div>
  );
}

// STAGE 4 CONTENT
function Stage4Content({ isMounted }: { isMounted: boolean }) {
  const trajectoryData = useMemo(() => {
    const data = [];
    for (let i = 0; i <= 20; i++) {
      const t = i * 10;
      const progress = t / 800;
      const x = -180 * Math.sin(progress * Math.PI * 0.5);
      data.push({
        time_ms: t,
        observed_x: parseFloat(x.toFixed(1)),
        forecast_x: parseFloat(x.toFixed(1)),
      });
    }
    for (let i = 21; i <= 80; i++) {
      const t = i * 10;
      const progress = t / 800;
      const x = -180 * Math.sin(progress * Math.PI * 0.5);
      data.push({
        time_ms: t,
        observed_x: null,
        forecast_x: parseFloat(x.toFixed(1)),
      });
    }
    return data;
  }, []);

  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed">
        The kinematic forecaster takes the initial 200 ms of hand movement (20 samples at 100 Hz) and predicts the full 3D reach endpoint. Regularized Ridge regression models extrapolate velocity curvature to estimate the terminal coordinate before physical completion.
      </p>

      {/* Trajectory Forecast Chart */}
      <div className="p-5 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0] space-y-3">
        <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0]">
          <span className="text-xs font-bold text-[#1C1C1A] uppercase tracking-wider flex items-center gap-1.5 font-display">
            <TrendingUp className="w-3.5 h-3.5 text-[#3A5244]" />
            Kinematic Extrapolation (X Displacement)
          </span>
          <span className="text-[11px] text-[#6B6B66] font-body">
            Prefix cutoff: t = 200 ms
          </span>
        </div>

        <div className="w-full h-52 select-none">
          {isMounted ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="time_ms"
                  unit=" ms"
                  stroke="#D9D8D0"
                  tick={{ fill: '#6B6B66', fontSize: 11 }}
                />
                <YAxis
                  unit=" mm"
                  stroke="#D9D8D0"
                  tick={{ fill: '#6B6B66', fontSize: 11 }}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-[#FFFFFF] border border-[#D9D8D0] p-3 rounded-xl text-xs shadow-md font-body text-[#1C1C1A]">
                          <div className="text-[#6B6B66]">Time: {label} ms</div>
                          {payload[0] && payload[0].value !== null && (
                            <div className="text-[#2E6B47] font-semibold">
                              {payload[0].name}: {payload[0].value} mm
                            </div>
                          )}
                          {payload[1] && payload[1].value !== null && (
                            <div className="text-[#6B46C1] font-semibold">
                              {payload[1].name}: {payload[1].value} mm
                            </div>
                          )}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ paddingBottom: 6 }}
                  formatter={(val) => <span className="text-xs text-[#1C1C1A] font-body">{val}</span>}
                />
                <Line
                  type="monotone"
                  dataKey="observed_x"
                  stroke="#2E6B47"
                  strokeWidth={2}
                  dot={false}
                  name="Observed Prefix (0–200 ms)"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast_x"
                  stroke="#6B46C1"
                  strokeWidth={1.8}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Forecast Continuation (200–800 ms)"
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-[#6B6B66]">
              Loading trajectory chart...
            </div>
          )}
        </div>
      </div>

      {/* Forecaster Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-4 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0]">
          <span className="text-[11px] text-[#6B6B66] block font-body">Input Prefix</span>
          <span className="text-base font-bold text-[#1C1C1A] font-display">20 samples</span>
          <p className="text-[11px] text-[#6B6B66] mt-0.5 font-body">200 ms post-movement onset</p>
        </div>

        <div className="p-4 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0]">
          <span className="text-[11px] text-[#6B6B66] block font-body">Forecaster Output</span>
          <span className="text-base font-bold text-[#1C1C1A] font-display">3D Endpoint</span>
          <p className="text-[11px] text-[#6B6B66] mt-0.5 font-body">[x, y, z] Cartesian coordinates</p>
        </div>

        <div className="p-4 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0]">
          <span className="text-[11px] text-[#6B6B66] block font-body">Ridge RMSE vs. Fair CV</span>
          <span className="text-base font-bold text-[#3A5244] font-display">
            47.25 mm vs. 85.65 mm
          </span>
          <p className="text-[11px] text-[#6B6B66] mt-0.5 font-body">44.8% error reduction</p>
        </div>
      </div>
    </div>
  );
}

// STAGE 5 CONTENT
function Stage5Content({ isMounted }: { isMounted: boolean }) {
  const comparisonData = [
    { name: 'Persistence', rmse_mm: 177.51, color: '#94A3B8' },
    { name: 'Fair CV', rmse_mm: 85.65, color: '#2563EB' },
    { name: 'Ridge', rmse_mm: 47.25, color: '#2E6B47' },
  ];

  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed">
        The Ridge forecaster is evaluated against two baselines under strict Leave-One-Subject-Out cross-validation: a static persistence baseline (zero velocity) and a fair constant-velocity baseline matching the same 200 ms prefix information.
      </p>

      {/* Bar Chart */}
      <div className="p-5 rounded-2xl bg-[#FAF9F5] border border-[#D9D8D0] space-y-3">
        <div className="flex items-center justify-between pb-3 border-b border-[#D9D8D0]">
          <span className="text-xs font-bold text-[#1C1C1A] uppercase tracking-wider flex items-center gap-1.5 font-display">
            <BarChart2 className="w-3.5 h-3.5 text-[#3A5244]" />
            Cohort 3D Endpoint Error (RMSE, mm)
          </span>
          <span className="text-[11px] text-[#6B6B66] font-body">
            Lower error indicates higher accuracy
          </span>
        </div>

        <div className="w-full h-56 select-none">
          {isMounted ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparisonData}
                margin={{ top: 16, right: 24, left: 0, bottom: 8 }}
              >
                <CartesianGrid stroke="#EDECE6" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="name"
                  stroke="#D9D8D0"
                  tick={{ fill: '#1C1C1A', fontSize: 12, fontWeight: 600 }}
                />
                <YAxis
                  unit=" mm"
                  stroke="#D9D8D0"
                  tick={{ fill: '#6B6B66', fontSize: 11 }}
                  domain={[0, 200]}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const item = payload[0].payload;
                      return (
                        <div className="bg-[#FFFFFF] border border-[#D9D8D0] p-3 rounded-xl shadow-md text-xs font-body text-[#1C1C1A]">
                          <div className="text-[#6B6B66] font-medium">{item.name}</div>
                          <div className="text-[#1C1C1A] font-bold font-display mt-0.5">
                            Endpoint RMSE: {item.rmse_mm} mm
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="rmse_mm" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-[#6B6B66]">
              Loading benchmark chart...
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#6B6B66] font-body">
        <div className="p-4 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0]">
          <strong className="text-[#1C1C1A] block mb-0.5 font-display">Metric Definition:</strong>
          Root-mean-square error measuring 3D Cartesian Euclidean distance between predicted and true reach endpoints across all valid trials.
        </div>
        <div className="p-4 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0]">
          <strong className="text-[#1C1C1A] block mb-0.5 font-display">Relative Improvement:</strong>
          Ridge achieves a 44.8% error reduction relative to Fair Constant Velocity and a 73.4% reduction relative to static Persistence.
        </div>
      </div>
    </div>
  );
}

// STAGE 6 CONTENT
function Stage6Content() {
  return (
    <div className="space-y-6 text-sm text-[#1C1C1A] font-body">
      <p className="leading-relaxed">
        To maintain scientific integrity, findings are interpreted through a three-tier epistemic audit taxonomy separating verified empirical results from exploratory hypotheses and unproven claims.
      </p>

      {/* Structured Limitations Taxonomy */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Tier 1 */}
        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
          <div className="flex items-center gap-1.5 pb-2.5 border-b border-[#D9D8D0]">
            <CheckCircle2 className="w-4 h-4 text-[#3A5244]" />
            <h3 className="font-bold text-xs text-[#1C1C1A] uppercase tracking-wider font-display">
              1. Established
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-[#6B6B66] leading-relaxed font-body">
            <li className="flex items-start gap-1.5">
              <span className="text-[#3A5244] font-bold">&bull;</span>
              <span>End-to-end multimodal pipeline software works correctly</span>
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-[#3A5244] font-bold">&bull;</span>
              <span>Ridge forecaster reliably outperforms both kinematic baselines</span>
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-[#3A5244] font-bold">&bull;</span>
              <span>Phase 4 condition discrimination: 66.46% BalAcc (LOSO)</span>
            </li>
          </ul>
        </div>

        {/* Tier 2 */}
        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
          <div className="flex items-center gap-1.5 pb-2.5 border-b border-[#D9D8D0]">
            <AlertTriangle className="w-4 h-4 text-[#B45309]" />
            <h3 className="font-bold text-xs text-[#1C1C1A] uppercase tracking-wider font-display">
              2. Exploratory
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-[#6B6B66] leading-relaxed font-body">
            <li className="flex items-start gap-1.5">
              <span className="text-[#B45309] font-bold">&bull;</span>
              <span>Phase 5 88.35% discrimination reflects stimulus-response congruency rules</span>
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-[#B45309] font-bold">&bull;</span>
              <span>Target centroids are empirical reach proxies, not independently surveyed targets</span>
            </li>
          </ul>
        </div>

        {/* Tier 3 */}
        <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#D9D8D0] space-y-3 shadow-sm">
          <div className="flex items-center gap-1.5 pb-2.5 border-b border-[#D9D8D0]">
            <XCircle className="w-4 h-4 text-[#6B6B66]" />
            <h3 className="font-bold text-xs text-[#1C1C1A] uppercase tracking-wider font-display">
              3. Pending
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-[#6B6B66] leading-relaxed font-body">
            <li className="flex items-start gap-1.5">
              <span className="text-[#6B6B66] font-bold">&bull;</span>
              <span>Independent biological motor error detection</span>
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-[#6B6B66] font-bold">&bull;</span>
              <span>Prospective multi-site external validation</span>
            </li>
            <li className="flex items-start gap-1.5">
              <span className="text-[#6B6B66] font-bold">&bull;</span>
              <span>Real-time causal closed-loop inference</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Button to Model Lab */}
      <div className="pt-4 flex flex-col sm:flex-row items-baseline justify-between gap-3 border-t border-[#D9D8D0]">
        <span className="text-xs text-[#6B6B66] font-body">
          Explore canonical test trial <code className="font-body font-semibold text-[#1C1C1A] bg-[#EDECE6] px-2 py-0.5 rounded-full border border-[#D9D8D0]">sub-01_run-01_L1_tr00</code> in the Model Lab:
        </span>
        <Link
          href="/model-lab?trial=sub-01_run-01_L1_tr00"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-semibold font-body bg-[#1C1C1A] text-[#F5F3EE] hover:bg-[#3A5244] transition-colors"
        >
          <span>Open in Model Lab</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    </div>
  );
}
