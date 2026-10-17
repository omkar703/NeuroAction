'use client';

import React, { useEffect, useRef } from 'react';

const CHANNELS = [
  { name: 'CZ',  color: '#3A5244', dotColor: '#3A5244', freqs: [10, 20, 5],  amps: [20, 7, 3]  },
  { name: 'FC3', color: '#2563EB', dotColor: '#2563EB', freqs: [13, 26, 6],  amps: [15, 6, 3]  },
  { name: 'C3',  color: '#6B46C1', dotColor: '#6B46C1', freqs: [8,  16, 4],  amps: [24, 9, 4]  },
  { name: 'CP4', color: '#B45309', dotColor: '#B45309', freqs: [15, 30, 7],  amps: [14, 5, 2]  },
];

const LANE_H  = 78;   // px per channel lane
const LABEL_W = 54;   // px for left channel labels
const PAD_B   = 26;   // px bottom for time axis

function eegAt(channel: typeof CHANNELS[0], t: number, spikePhase: number): number {
  const base = channel.freqs.reduce((acc, f, i) => {
    return acc + channel.amps[i] * Math.sin(2 * Math.PI * f * t + i);
  }, 0);

  const spikeTrigger = Math.abs(Math.sin(2 * Math.PI * 0.3 * spikePhase));
  const spike = spikeTrigger > 0.97 ? 35 * (1 - spikeTrigger / 0.97 * (1 - spikeTrigger) * 30) : 0;
  return base + spike * 0.45;
}

export default function NeuralSignalViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr   = window.devicePixelRatio || 1;
    let frameId = 0;
    let t       = 0;
    const speed = 0.0035;

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      canvas!.width  = rect.width  * dpr;
      canvas!.height = rect.height * dpr;
    }
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function draw() {
      const ctx = canvas!.getContext('2d');
      if (!ctx) return;
      const W = canvas!.width;
      const H = canvas!.height;

      ctx.scale(dpr, dpr);
      const Wl = W / dpr;
      const Hl = H / dpr;

      // ── Clean Theme-Matched Surface ───────────────
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, Wl, Hl);

      // ── Subtle Background Grid ────────────────────
      ctx.strokeStyle = '#EDECE6';
      ctx.lineWidth   = 1;
      const gridX = 64;
      const gridY = LANE_H;
      for (let x = LABEL_W; x < Wl; x += gridX) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, Hl - PAD_B);
        ctx.stroke();
      }
      for (let i = 0; i <= CHANNELS.length; i++) {
        const y = i * gridY;
        ctx.beginPath();
        ctx.moveTo(LABEL_W, y);
        ctx.lineTo(Wl, y);
        ctx.stroke();
      }

      // ── Channel Labels ────────────────────────────
      ctx.textAlign    = 'right';
      ctx.textBaseline = 'middle';
      CHANNELS.forEach((ch, i) => {
        const cy = i * LANE_H + LANE_H / 2;

        // Subtle indicator dot
        ctx.fillStyle = ch.dotColor;
        ctx.beginPath();
        ctx.arc(LABEL_W - 14, cy, 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#1C1C1A';
        ctx.font      = 'bold 11px "Open Sans", sans-serif';
        ctx.fillText(ch.name, LABEL_W - 22, cy);
      });

      // ── Waveforms ─────────────────────────────────
      const plotW = Wl - LABEL_W;
      CHANNELS.forEach((ch, i) => {
        const cy = i * LANE_H + LANE_H / 2;

        ctx.strokeStyle = ch.color;
        ctx.lineWidth   = 1.75;
        ctx.beginPath();

        for (let px = 0; px <= plotW; px++) {
          const tSample = t - (plotW - px) / plotW;
          const y = cy - eegAt(ch, tSample, tSample);
          if (px === 0) ctx.moveTo(LABEL_W + px, y);
          else          ctx.lineTo(LABEL_W + px, y);
        }
        ctx.stroke();
      });

      // ── Soft Scanning Cursor Line ─────────────────
      const scanX    = Wl - 2;
      const scanGrad = ctx.createLinearGradient(Wl - 32, 0, scanX, 0);
      scanGrad.addColorStop(0, 'rgba(58, 82, 68, 0)');
      scanGrad.addColorStop(1, 'rgba(58, 82, 68, 0.08)');
      ctx.fillStyle = scanGrad;
      ctx.fillRect(Wl - 32, 0, 32, Hl - PAD_B);

      ctx.strokeStyle = '#3A5244';
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      ctx.moveTo(scanX, 0);
      ctx.lineTo(scanX, Hl - PAD_B);
      ctx.stroke();

      // ── Time Axis Labels ──────────────────────────
      ctx.fillStyle    = '#6B6B66';
      ctx.font         = '500 10px "Open Sans", sans-serif';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      const msSteps = ['-1000', '-750', '-500', '-250', '0 ms'];
      msSteps.forEach((ms, idx) => {
        const x = LABEL_W + (idx / (msSteps.length - 1)) * plotW;
        ctx.fillText(ms, x, Hl - PAD_B / 2);
      });

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      t += speed;
      frameId = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      cancelAnimationFrame(frameId);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-[340px] rounded-2xl overflow-hidden bg-[#FFFFFF] border border-[#D9D8D0] shadow-sm">
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        aria-label="Pre-movement EEG signal channels"
      />
    </div>
  );
}
