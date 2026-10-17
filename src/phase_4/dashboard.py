"""Interactive 6-Panel Research Dashboard for Phase 4 Digital Twin.

Provides both:
1. Zero-dependency standalone HTTP server with rich interactive web frontend (Panels A-F).
2. Headless Matplotlib publication-grade 6-panel snapshot generator for reports and presentations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Any
import argparse
import json
import traceback
import urllib.parse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.coordinate_mapping import TARGET_COORDINATES_MM
from src.phase_4.replay_engine import ReplayEngine
from src.phase_4.trial_registry import TrialRegistry


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data structures for standard RFC 8259 JSON compliance.
    
    Replaces NaN, Inf, and non-finite floats with None (serialized to JSON null).
    Converts NumPy arrays and scalar types to native Python equivalents.
    """
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


def generate_dashboard_figure(
    trial_id: str,
    output_path: Path,
    replay_time_ms: Optional[float] = None,
    protocol: str = "kinematic_onset",
    config: Optional[DigitalTwinConfig] = None,
) -> Path:
    """Generate publication-grade 6-panel research dashboard snapshot via Matplotlib."""
    cfg = config or DigitalTwinConfig()
    engine = ReplayEngine(config=cfg)

    trial, metadata, neural_result, forecast_result = engine.prepare_trial_simulation(
        trial_id=trial_id, protocol=protocol
    )

    t_eval = replay_time_ms if replay_time_ms is not None else float(trial.times_me[-1])
    frame = engine.get_frame_at_time(
        trial=trial,
        metadata=metadata,
        neural_result=neural_result,
        forecast_result=forecast_result,
        replay_time_ms=t_eval,
        mode="retrospective",
    )
    fusion = frame.fusion_state
    conflict = fusion.conflict_assessment

    fig = plt.figure(figsize=(18, 11), facecolor="#0f172a")
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    plt.rcParams["text.color"] = "#e2e8f0"
    plt.rcParams["axes.labelcolor"] = "#94a3b8"
    plt.rcParams["xtick.color"] = "#94a3b8"
    plt.rcParams["ytick.color"] = "#94a3b8"

    # --- Title Banner ---
    title_str = (
        f"NEUROACTION DIGITAL TWIN REPLAY — Trial: {trial_id} | "
        f"Subject: {metadata.participant_id} | Condition: {metadata.congruency.upper()} | "
        f"Protocol: {protocol.replace('_', ' ').title()}"
    )
    fig.suptitle(title_str, fontsize=14, fontweight="bold", color="#38bdf8", y=0.98)

    # --- Panel A: Trial Selection & Status ---
    ax_a = fig.add_subplot(gs[0, 0], facecolor="#1e293b")
    ax_a.set_title("Panel A — Trial State & Playback Clock", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_a.axis("off")
    info_text = (
        f"Participant: {metadata.participant_id}\n"
        f"Run / Trial: {metadata.run_id} (tr#{metadata.trial_index:02d})\n"
        f"Condition: {metadata.congruency.upper()} (Cue: {metadata.ball_color.upper()})\n"
        f"Instructed Goal: {metadata.target_goal.upper()}\n"
        f"Physical Hand: {metadata.physical_hand_direction.upper()}\n"
        f"Replay Clock: t = {frame.timestamp_ms:.1f} ms\n"
        f"Trial Validity: {'VALID' if metadata.valid else 'INVALID'}\n"
        f"Mode: Deterministic Offline Replay"
    )
    ax_a.text(0.05, 0.5, info_text, fontsize=9.5, va="center", color="#f8fafc", family="monospace")

    # --- Panel B: 3D Trajectory Twin (Top-Down X-Z Projection) ---
    ax_b = fig.add_subplot(gs[0:2, 1:], facecolor="#1e293b")
    ax_b.set_title("Panel B — 3D Workspace Twin (Leap Motion X-Z Projection)", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_b.set_xlabel("X Lateral Position (mm)", fontsize=9.5)
    ax_b.set_ylabel("Z Depth Position (mm)", fontsize=9.5)
    ax_b.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # Draw target zones
    t_left = TARGET_COORDINATES_MM["left"]
    t_right = TARGET_COORDINATES_MM["right"]
    ax_b.scatter([t_left[0]], [t_left[2]], c="#3b82f6", s=160, marker="s", edgecolors="white", label="Target Zone: LEFT", zorder=4)
    ax_b.scatter([t_right[0]], [t_right[2]], c="#10b981", s=160, marker="s", edgecolors="white", label="Target Zone: RIGHT", zorder=4)

    # Observed Hand Path so far
    if frame.observed_path_so_far is not None and len(frame.observed_path_so_far) > 0:
        valid_obs = frame.observed_path_so_far[~np.isnan(frame.observed_path_so_far).any(axis=1)]
        if len(valid_obs) > 0:
            ax_b.plot(valid_obs[:, 0], valid_obs[:, 2], color="#f59e0b", lw=2.5, label="Observed Hand Path (Past)", zorder=3)
            curr_pt = valid_obs[-1]
            ax_b.scatter([curr_pt[0]], [curr_pt[2]], c="#f59e0b", s=100, edgecolors="white", zorder=5, label="Current Hand Position")

    # Future ground truth (Retrospective)
    if frame.actual_future_path is not None and len(frame.actual_future_path) > 0:
        valid_fut = frame.actual_future_path[~np.isnan(frame.actual_future_path).any(axis=1)]
        if len(valid_fut) > 0:
            ax_b.plot(valid_fut[:, 0], valid_fut[:, 2], color="#64748b", ls=":", lw=1.8, label="Future Ground Truth (Offline)", zorder=2)

    # Predicted Future Forecast
    if frame.forecast_path is not None and len(frame.forecast_path) > 0:
        valid_fc = frame.forecast_path[~np.isnan(frame.forecast_path).any(axis=1)]
        if len(valid_fc) > 0:
            ax_b.plot(valid_fc[:, 0], valid_fc[:, 2], color="#ec4899", lw=2.5, ls="--", label="Phase 3 Predicted Trajectory", zorder=4)

    # Predicted Endpoint
    if forecast_result.predicted_endpoint is not None and not np.isnan(forecast_result.predicted_endpoint).any():
        p_end = forecast_result.predicted_endpoint
        ax_b.scatter([p_end[0]], [p_end[2]], c="#ec4899", s=140, marker="X", edgecolors="white", label="Phase 3 Predicted Endpoint", zorder=5)

    ax_b.set_xlim(-260, 260)
    ax_b.set_ylim(-30, 160)
    ax_b.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)

    # --- Panel C: Neural Intention Decoding ---
    ax_c = fig.add_subplot(gs[1, 0], facecolor="#1e293b")
    ax_c.set_title("Panel C — Pre-movement Neural Intention", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_c.axis("off")
    p_left_pct = neural_result.probability_left * 100.0
    p_right_pct = neural_result.probability_right * 100.0
    neural_text = (
        f"Model: {neural_result.model_name}\n"
        f"Input Window: [{neural_result.window_start_ms:.0f}, {neural_result.window_end_ms:.0f}] ms\n"
        f"Decoded Goal: {neural_result.predicted_goal.upper()}\n"
        f"Probability Left:  {p_left_pct:.1f}%\n"
        f"Probability Right: {p_right_pct:.1f}%\n"
        f"Confidence Score:  {neural_result.confidence:.3f}\n"
        f"Instructed Goal:   {metadata.target_goal.upper()}\n"
        f"Decoded Agreement: {'CORRECT' if neural_result.predicted_goal == metadata.target_goal else 'MISMATCH'}\n"
        f"Status: {neural_result.status} (Offline Zero-Phase)"
    )
    ax_c.text(0.05, 0.5, neural_text, fontsize=9.2, va="center", color="#f8fafc", family="monospace")

    # --- Panel D: Conflict Analysis Engine ---
    ax_d = fig.add_subplot(gs[2, 0], facecolor="#1e293b")
    ax_d.set_title("Panel D — Intention–Action Conflict", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_d.axis("off")
    state_color = "#ef4444" if conflict.conflict_state == "POTENTIAL_DISAGREEMENT" else "#10b981" if conflict.conflict_state == "ALIGNED" else "#eab308"
    d_mm = f"{conflict.endpoint_target_distance_mm:.1f} mm" if conflict.endpoint_target_distance_mm is not None else "N/A"
    theta = f"{conflict.directional_disagreement_deg:.1f}°" if conflict.directional_disagreement_deg is not None else "N/A"
    c_norm = f"{conflict.normalized_distance_score:.2f}" if conflict.normalized_distance_score is not None else "N/A"

    conflict_text = (
        f"Conflict State: {conflict.conflict_state}\n"
        f"Target-Endpoint Dist: {d_mm}\n"
        f"Directional Angle:    {theta}\n"
        f"Normalized Score:     {c_norm} / 1.0\n"
        f"Goal Agreement:       {conflict.neural_goal_agreement}\n"
        f"Motor Agreement:      {conflict.motor_forecast_agreement}\n"
        f"Thresholds: d > 50 mm | θ > 45°\n"
        f"Explanation:\n{conflict.explanation[:95]}..."
    )
    ax_d.text(0.05, 0.5, conflict_text, fontsize=9.0, va="center", color=state_color if conflict.conflict_state == "POTENTIAL_DISAGREEMENT" else "#f8fafc", family="monospace")

    # --- Panel E: Temporal Timeline ---
    ax_e = fig.add_subplot(gs[2, 1], facecolor="#1e293b")
    ax_e.set_title("Panel E — Temporal Event Sequence", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_e.set_xlabel("Trial Time (ms)", fontsize=9.5)
    ax_e.set_yticks([])
    ax_e.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # Plot events on horizontal line
    events = [
        (-1000, "EEG Start", "#64748b"),
        (0, "Go-Cue (Cue)", "#38bdf8"),
        (metadata.kinematic_onset_time_ms or 370.0, "Onset", "#f59e0b"),
        ((metadata.kinematic_onset_time_ms or 370.0) + 200.0, "Prefix End\n(Forecast)", "#ec4899"),
        (float(trial.times_me[-1]), "Trial End", "#64748b"),
    ]
    ax_e.axhline(0, color="#475569", lw=2)
    for t_ev, lbl, col in events:
        ax_e.scatter([t_ev], [0], color=col, s=70, zorder=4)
        ax_e.text(t_ev, 0.15, lbl, rotation=45, ha="center", fontsize=7.5, color=col)
    ax_e.axvline(frame.timestamp_ms, color="#f43f5e", ls="--", lw=1.8, label="Current Clock")
    ax_e.set_xlim(-1100, float(trial.times_me[-1]) + 150)
    ax_e.set_ylim(-0.3, 0.6)
    ax_e.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", fontsize=8)

    # --- Panel F: Metrics & Retrospective Verification ---
    ax_f = fig.add_subplot(gs[2, 2], facecolor="#1e293b")
    ax_f.set_title("Panel F — Retrospective Verification Metrics", fontsize=11, fontweight="bold", color="#38bdf8")
    ax_f.axis("off")
    retro_err = f"{conflict.retrospective_endpoint_error_mm:.1f} mm" if conflict.retrospective_endpoint_error_mm is not None else "N/A"
    metrics_text = (
        f"Trial Evaluation (N=1 Sample):\n"
        f"• Decoded Goal Match:    {conflict.neural_goal_agreement}\n"
        f"• Retrospective Error:   {retro_err}\n"
        f"• Forecast Protocol:     {protocol}\n"
        f"• Held-out Validation:   {forecast_result.is_held_out}\n\n"
        f"Cohort Baseline Reference:\n"
        f"• Ridge Forecaster RMSE: 62.4 mm (Held-out)\n"
        f"• Constant-Vel Baseline: 86.8 mm\n"
        f"• Persistence Baseline:  175.2 mm\n"
        f"• Goal Dec. Bal Acc:     69.9% (LOSO)"
    )
    ax_f.text(0.05, 0.5, metrics_text, fontsize=9.2, va="center", color="#f8fafc", family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


HTML_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroAction Digital Twin Replay Dashboard</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #151c2c;
      --border: #222f47;
      --accent: #38bdf8;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --danger: #ef4444;
      --success: #10b981;
      --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 18px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 18px;
    }
    h1 { font-size: 1.3rem; color: var(--accent); font-weight: 700; }
    .badge {
      background: #1e293b;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.8rem;
      border: 1px solid var(--border);
    }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr 340px;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .card-title {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    label { font-size: 0.8rem; color: var(--muted); display: block; margin-bottom: 4px; }
    select, input[type=range], button {
      width: 100%;
      background: #0d131f;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px;
      border-radius: 4px;
      margin-bottom: 10px;
      font-size: 0.85rem;
    }
    button {
      cursor: pointer;
      font-weight: 600;
      background: #1e293b;
      transition: background 0.15s;
    }
    button:hover { background: var(--border); }
    .btn-group { display: flex; gap: 8px; }
    .stat-row {
      display: flex;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.85rem;
    }
    .stat-val { font-weight: 600; font-family: monospace; }
    #canvas-container {
      position: relative;
      width: 100%;
      height: 480px;
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }
    canvas { width: 100%; height: 100%; display: block; }
    .conflict-banner {
      padding: 10px;
      border-radius: 4px;
      font-weight: 700;
      text-align: center;
      margin-top: 10px;
      font-size: 0.9rem;
      transition: all 0.2s;
    }
    .state-ALIGNED { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
    .state-POTENTIAL_DISAGREEMENT { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .state-UNCERTAIN { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
    .state-UNAVAILABLE { background: #1e293b; color: var(--muted); border: 1px solid var(--border); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>NEUROACTION DIGITAL TWIN REPLAY DASHBOARD</h1>
      <span style="font-size: 0.8rem; color: var(--muted);">Pre-movement EEG Intention + 3D Kinematics Reconstruction + Conflict Detection</span>
    </div>
    <div class="badge">Offline Deterministic Replay Mode</div>
  </header>

  <div class="grid">
    <!-- Left Column: Controls & Neural Intention -->
    <div>
      <div class="card">
        <div class="card-title">Panel A — Playback & Trial Selection</div>
        <label>Participant</label>
        <select id="sub-select" onchange="loadTrialsForSub()"></select>

        <label>Trial ID</label>
        <select id="trial-select" onchange="loadTrial()"></select>

        <label>Forecasting Protocol</label>
        <select id="protocol-select" onchange="loadTrial()">
          <option value="kinematic_onset">Protocol B: Kinematic-Onset-Locked (Speed ≥ 20 mm/s)</option>
          <option value="event_locked">Protocol A: Event-Locked (t ≥ 0 ms)</option>
        </select>

        <label>Simulation Clock: <span id="clock-display" style="color: var(--accent); font-weight: bold;">0 ms</span></label>
        <input type="range" id="time-slider" min="0" max="1500" value="0" oninput="onSeek(this.value)">

        <div class="btn-group">
          <button onclick="togglePlay()" id="play-btn">▶ Play</button>
          <button onclick="restart()">↺ Restart</button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Panel C — Neural Intention Decoding</div>
        <div class="stat-row"><span>Decoded Goal</span><span id="neural-goal" class="stat-val">—</span></div>
        <div class="stat-row"><span>Confidence Score</span><span id="neural-conf" class="stat-val">—</span></div>
        <div class="stat-row"><span>Instructed Goal</span><span id="true-goal" class="stat-val">—</span></div>
        <div class="stat-row"><span>Goal Agreement</span><span id="goal-match" class="stat-val">—</span></div>
        <div class="stat-row"><span>Model Artifact</span><span id="neural-model" class="stat-val">LDA Shrinkage</span></div>
        <div class="stat-row"><span>Input Window</span><span class="stat-val">[-1000, 0] ms</span></div>
      </div>
    </div>

    <!-- Center Column: 3D Twin Workspace -->
    <div>
      <div class="card">
        <div class="card-title">Panel B — 3D Workspace Twin (Leap Motion Top-Down Projection)</div>
        <div id="canvas-container">
          <canvas id="twin-canvas"></canvas>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 10px; font-size: 0.8rem; color: var(--muted);">
          <span><span style="color: #3b82f6;">■</span> Left Target</span>
          <span><span style="color: #10b981;">■</span> Right Target</span>
          <span><span style="color: #f59e0b;">●</span> Hand Past Path</span>
          <span><span style="color: #ec4899;">--</span> Forecast Trajectory</span>
          <span><span style="color: #64748b;">:</span> Ground Truth</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Panel E — Temporal Event Sequence</div>
        <div class="stat-row"><span>Visual Go-Cue</span><span class="stat-val">0.0 ms</span></div>
        <div class="stat-row"><span>Kinematic Onset Latency</span><span id="onset-latency" class="stat-val">—</span></div>
        <div class="stat-row"><span>Forecast Availability</span><span id="forecast-avail" class="stat-val">—</span></div>
        <div class="stat-row"><span>Movement Termination</span><span id="movement-end" class="stat-val">—</span></div>
      </div>
    </div>

    <!-- Right Column: Conflict Analysis & Metrics -->
    <div>
      <div class="card">
        <div class="card-title">Panel D — Intention–Action Conflict Engine</div>
        <div id="conflict-state-box" class="conflict-banner state-UNAVAILABLE">AWAITING SIMULATION</div>
        <div class="stat-row"><span>Endpoint-Target Dist</span><span id="conflict-dist" class="stat-val">—</span></div>
        <div class="stat-row"><span>Directional Angle</span><span id="conflict-theta" class="stat-val">—</span></div>
        <div class="stat-row"><span>Normalized Score</span><span id="conflict-norm" class="stat-val">—</span></div>
        <div class="stat-row"><span>Physical Hand Dir</span><span id="physical-hand" class="stat-val">—</span></div>
        <div class="stat-row"><span>Motor Agreement</span><span id="motor-match" class="stat-val">—</span></div>
        <p id="conflict-exp" style="font-size: 0.8rem; color: var(--muted); margin-top: 10px; line-height: 1.4;"></p>
      </div>

      <div class="card">
        <div class="card-title">Panel F — Verification Metrics</div>
        <div class="stat-row"><span>Retrospective Error</span><span id="retro-err" class="stat-val">—</span></div>
        <div class="stat-row"><span>Held-out Evaluation</span><span id="held-out-badge" class="stat-val">YES (sub-01)</span></div>
        <div class="stat-row"><span>Condition Semantics</span><span id="cond-badge" class="stat-val">—</span></div>
        <div class="stat-row"><span>Ridge Forecaster RMSE</span><span class="stat-val">62.4 mm</span></div>
        <div class="stat-row"><span>Constant-Vel RMSE</span><span class="stat-val">86.8 mm</span></div>
      </div>
    </div>
  </div>

  <script>
    let trialData = null;
    let isPlaying = false;
    let playInterval = null;
    let currentIdx = 0;

    async function init() {
      try {
        const resp = await fetch("/api/participants");
        const subs = await resp.json();
        const sel = document.getElementById("sub-select");
        sel.innerHTML = subs.map(s => `<option value="${s}">${s}</option>`).join("");
        await loadTrialsForSub();
      } catch (err) {
        console.error("Initialization error:", err);
      }
      window.addEventListener("resize", () => {
        if (trialData) drawCanvas(currentIdx);
      });
    }

    async function loadTrialsForSub() {
      const sub = document.getElementById("sub-select").value;
      if (!sub) return;
      try {
        const resp = await fetch(`/api/trials?participant=${encodeURIComponent(sub)}`);
        const trials = await resp.json();
        const sel = document.getElementById("trial-select");
        sel.innerHTML = trials.map(t => `<option value="${t}">${t}</option>`).join("");
        await loadTrial();
      } catch (err) {
        console.error("Error loading trials for subject:", err);
      }
    }

    async function loadTrial() {
      const trialId = document.getElementById("trial-select").value;
      const protocol = document.getElementById("protocol-select").value;
      if (!trialId) return;

      const stateBox = document.getElementById("conflict-state-box");
      stateBox.innerText = "LOADING TRIAL DATA...";
      stateBox.className = "conflict-banner state-UNAVAILABLE";

      try {
        const resp = await fetch(`/api/trial_data?trial_id=${encodeURIComponent(trialId)}&protocol=${encodeURIComponent(protocol)}`);
        if (!resp.ok) {
          const errObj = await resp.json().catch(() => ({}));
          console.error("API error:", resp.status, errObj);
          stateBox.innerText = "ERROR LOADING TRIAL";
          stateBox.className = "conflict-banner state-POTENTIAL_DISAGREEMENT";
          document.getElementById("conflict-exp").innerText = errObj.error || `HTTP ${resp.status}`;
          return;
        }
        trialData = await resp.json();

        const slider = document.getElementById("time-slider");
        slider.min = 0;
        slider.max = Math.max(0, trialData.times.length - 1);
        slider.value = 0;
        currentIdx = 0;

        renderFrame(0);
      } catch (err) {
        console.error("Network/JSON parse error in loadTrial:", err);
        stateBox.innerText = "NETWORK/PARSE ERROR";
        stateBox.className = "conflict-banner state-POTENTIAL_DISAGREEMENT";
        document.getElementById("conflict-exp").innerText = String(err);
      }
    }

    function renderFrame(idx) {
      if (!trialData || !trialData.times || trialData.times.length === 0) return;
      currentIdx = Math.max(0, Math.min(idx, trialData.times.length - 1));
      const t = trialData.times[currentIdx];
      document.getElementById("clock-display").innerText = `${t !== null ? t.toFixed(0) : 0} ms`;
      document.getElementById("time-slider").value = currentIdx;

      // Update Panel C: Neural
      if (trialData.neural) {
        document.getElementById("neural-goal").innerText = (trialData.neural.predicted_goal || "—").toUpperCase();
        document.getElementById("neural-conf").innerText = trialData.neural.confidence !== null ? trialData.neural.confidence.toFixed(3) : "—";
        document.getElementById("true-goal").innerText = (trialData.metadata.target_goal || "—").toUpperCase();
        const isMatch = trialData.neural.predicted_goal === trialData.metadata.target_goal;
        document.getElementById("goal-match").innerText = isMatch ? "MATCH (100%)" : "MISMATCH";
      }

      // Update Panel D: Conflict
      const conf = (trialData.conflict_timeseries && trialData.conflict_timeseries[currentIdx]) ||
                   (trialData.conflict_timeseries && trialData.conflict_timeseries[trialData.conflict_timeseries.length - 1]) ||
                   { state: "UNAVAILABLE", dist: null, theta: null, norm: null, motor_match: null, explanation: "" };

      const box = document.getElementById("conflict-state-box");
      box.className = `conflict-banner state-${conf.state}`;
      box.innerText = (conf.state || "UNAVAILABLE").replace("_", " ");

      document.getElementById("conflict-dist").innerText = conf.dist !== null ? `${conf.dist.toFixed(1)} mm` : "N/A";
      document.getElementById("conflict-theta").innerText = conf.theta !== null ? `${conf.theta.toFixed(1)}°` : "N/A";
      document.getElementById("conflict-norm").innerText = conf.norm !== null ? `${conf.norm.toFixed(2)} / 1.0` : "N/A";
      document.getElementById("physical-hand").innerText = (trialData.metadata.physical_hand_direction || "—").toUpperCase();
      document.getElementById("motor-match").innerText = conf.motor_match === true ? "AGREED" : conf.motor_match === false ? "DISAGREED" : "—";
      document.getElementById("conflict-exp").innerText = conf.explanation || "";

      // Update Panel E: Timeline
      document.getElementById("onset-latency").innerText = trialData.metadata.onset_latency !== null ? `${trialData.metadata.onset_latency.toFixed(1)} ms` : "—";
      document.getElementById("forecast-avail").innerText = (trialData.forecast && trialData.forecast.forecast_generation_time_ms !== null) ? `${trialData.forecast.forecast_generation_time_ms.toFixed(1)} ms` : "—";
      const lastT = trialData.times[trialData.times.length - 1];
      document.getElementById("movement-end").innerText = lastT !== null ? `${lastT.toFixed(1)} ms` : "—";

      // Update Panel F
      document.getElementById("cond-badge").innerText = `${(trialData.metadata.congruency || "").toUpperCase()} (Cue: ${(trialData.metadata.ball_color || "").toUpperCase()})`;
      document.getElementById("retro-err").innerText = (trialData.forecast && trialData.forecast.endpoint_error !== null) ? `${trialData.forecast.endpoint_error.toFixed(1)} mm` : "N/A";
      document.getElementById("held-out-badge").innerText = (trialData.metadata.participant_id === "sub-01") ? "YES (sub-01)" : "COHORT LOSO";

      // Draw Canvas 3D Top-down
      drawCanvas(currentIdx);
    }

    function drawCanvas(idx) {
      const canvas = document.getElementById("twin-canvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const container = document.getElementById("canvas-container");
      const w = canvas.width = container.clientWidth || 600;
      const h = canvas.height = container.clientHeight || 480;

      ctx.clearRect(0, 0, w, h);

      // Coordinate mapping: X in [-260, 260] -> [0, w], Z in [-30, 160] -> [h, 0]
      function toScreen(x, z) {
        const sx = ((x + 260) / 520) * w;
        const sy = h - ((z + 30) / 190) * h;
        return [sx, sy];
      }

      // Draw Grid
      ctx.strokeStyle = "#1b253b";
      ctx.lineWidth = 1;
      for (let x = -200; x <= 200; x += 100) {
        ctx.beginPath();
        const [x1, y1] = toScreen(x, -30);
        const [x2, y2] = toScreen(x, 160);
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      for (let z = 0; z <= 150; z += 50) {
        ctx.beginPath();
        const [x1, y1] = toScreen(-250, z);
        const [x2, y2] = toScreen(250, z);
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }

      // Draw Target Zones
      const [tlx, tly] = toScreen(-188.0, 53.0);
      const [trx, try_] = toScreen(187.5, 51.0);
      ctx.fillStyle = "#3b82f6";
      ctx.fillRect(tlx - 10, tly - 10, 20, 20);
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px sans-serif";
      ctx.fillText("TARGET: LEFT", tlx - 35, tly - 14);

      ctx.fillStyle = "#10b981";
      ctx.fillRect(trx - 10, try_ - 10, 20, 20);
      ctx.fillStyle = "#ffffff";
      ctx.fillText("TARGET: RIGHT", trx - 35, try_ - 14);

      if (!trialData || !trialData.hand_x) return;

      // Draw Future Ground Truth (skip nulls)
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      let startedGround = false;
      for (let i = 0; i < trialData.hand_x.length; i++) {
        const hx = trialData.hand_x[i];
        const hz = trialData.hand_z[i];
        if (hx === null || hz === null || isNaN(hx) || isNaN(hz)) {
          startedGround = false;
          continue;
        }
        const [px, pz] = toScreen(hx, hz);
        if (!startedGround) {
          ctx.moveTo(px, pz);
          startedGround = true;
        } else {
          ctx.lineTo(px, pz);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Predicted Horizon Trajectory (Pink dashed)
      const curT = trialData.times[idx];
      if (trialData.forecast && trialData.forecast.horizon_path && curT >= trialData.forecast.forecast_generation_time_ms) {
        ctx.strokeStyle = "#ec4899";
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        let startedFore = false;
        for (let i = 0; i < trialData.forecast.horizon_path.length; i++) {
          const pt = trialData.forecast.horizon_path[i];
          if (!pt || pt[0] === null || pt[2] === null || isNaN(pt[0]) || isNaN(pt[2])) continue;
          const [px, pz] = toScreen(pt[0], pt[2]);
          if (!startedFore) { ctx.moveTo(px, pz); startedFore = true; }
          else { ctx.lineTo(px, pz); }
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Forecast endpoint marker
        if (trialData.forecast.predicted_endpoint && trialData.forecast.predicted_endpoint[0] !== null) {
          const [epx, epz] = toScreen(trialData.forecast.predicted_endpoint[0], trialData.forecast.predicted_endpoint[2]);
          ctx.fillStyle = "#ec4899";
          ctx.beginPath(); ctx.arc(epx, epz, 7, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Draw Hand Observed Path up to idx
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      let startedObs = false;
      let lastValidPt = null;
      for (let i = 0; i <= idx; i++) {
        const hx = trialData.hand_x[i];
        const hz = trialData.hand_z[i];
        if (hx === null || hz === null || isNaN(hx) || isNaN(hz)) {
          startedObs = false;
          continue;
        }
        const [px, pz] = toScreen(hx, hz);
        lastValidPt = [px, pz];
        if (!startedObs) {
          ctx.moveTo(px, pz);
          startedObs = true;
        } else {
          ctx.lineTo(px, pz);
        }
      }
      ctx.stroke();

      // Current Hand Point dot (draw only if a valid point was observed)
      if (lastValidPt) {
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath(); ctx.arc(lastValidPt[0], lastValidPt[1], 7, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    function onSeek(val) { renderFrame(parseInt(val)); }

    function togglePlay() {
      if (!trialData || !trialData.times || trialData.times.length === 0) return;
      isPlaying = !isPlaying;
      document.getElementById("play-btn").innerText = isPlaying ? "⏸ Pause" : "▶ Play";
      if (isPlaying) {
        if (currentIdx >= trialData.times.length - 1) {
          currentIdx = 0;
        }
        playInterval = setInterval(() => {
          if (currentIdx < trialData.times.length - 1) {
            renderFrame(currentIdx + 1);
          } else {
            togglePlay();
          }
        }, 30);
      } else {
        clearInterval(playInterval);
      }
    }

    function restart() {
      if (isPlaying) togglePlay();
      renderFrame(0);
    }

    window.onload = init;
  </script>
</body>
</html>
"""


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP server handling dashboard API and web UI."""

    engine: Optional[ReplayEngine] = None
    config: Optional[DigitalTwinConfig] = None

    def log_message(self, format, *args):
        """Silence standard console logging for clean CLI."""
        pass

    def do_GET(self):
        try:
            url_parts = urllib.parse.urlparse(self.path)
            path = url_parts.path
            query = urllib.parse.parse_qs(url_parts.query)

            if path in ("/", "/index.html"):
                html_bytes = HTML_DASHBOARD_TEMPLATE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            elif path == "/api/participants":
                subs = self.engine.registry.list_participants()
                self.send_json(subs)
                return

            elif path == "/api/trials":
                sub = query.get("participant", [None])[0]
                trials = self.engine.registry.list_trials(participant_id=sub, valid_only=True)
                self.send_json(trials)
                return

            elif path == "/api/trial_data":
                trial_id = query.get("trial_id", [self.config.demo_trial_id])[0]
                protocol = query.get("protocol", [self.config.default_protocol])[0]

                trial, meta, neural, forecast = self.engine.prepare_trial_simulation(
                    trial_id=trial_id, protocol=protocol
                )

                # Build timeseries for each frame
                times = list(float(t) for t in trial.times_me)
                hand_x = list(float(x) for x in trial.hand_x)
                hand_y = list(float(y) for y in trial.hand_y)
                hand_z = list(float(z) for z in trial.hand_z)

                conf_series = []
                for t_ms in times:
                    fr = self.engine.get_frame_at_time(
                        trial=trial,
                        metadata=meta,
                        neural_result=neural,
                        forecast_result=forecast,
                        replay_time_ms=t_ms,
                        mode="causal_replay",
                    )
                    c = fr.fusion_state.conflict_assessment
                    conf_series.append({
                        "state": c.conflict_state,
                        "dist": c.endpoint_target_distance_mm,
                        "theta": c.directional_disagreement_deg,
                        "norm": c.normalized_distance_score,
                        "motor_match": c.motor_forecast_agreement,
                        "explanation": c.explanation,
                    })

                max_h = max(forecast.horizon_trajectories.keys()) if forecast.horizon_trajectories else None
                h_path = forecast.horizon_trajectories[max_h].tolist() if max_h else None

                end_error = (
                    float(np.linalg.norm(forecast.predicted_endpoint - forecast.actual_endpoint))
                    if forecast.actual_endpoint is not None and not np.isnan(forecast.actual_endpoint).any()
                    else 0.0
                )

                data = {
                    "trial_id": trial_id,
                    "times": times,
                    "hand_x": hand_x,
                    "hand_y": hand_y,
                    "hand_z": hand_z,
                    "metadata": {
                        "participant_id": meta.participant_id,
                        "congruency": meta.congruency,
                        "target_goal": meta.target_goal,
                        "physical_hand_direction": meta.physical_hand_direction,
                        "ball_color": meta.ball_color,
                        "onset_latency": meta.kinematic_onset_time_ms or 370.0,
                    },
                    "neural": neural.to_dict(),
                    "forecast": {
                        "forecast_generation_time_ms": forecast.forecast_generation_time_ms,
                        "predicted_endpoint": forecast.predicted_endpoint.tolist(),
                        "horizon_path": h_path,
                        "endpoint_error": end_error,
                    },
                    "conflict_timeseries": conf_series,
                }
                self.send_json(data)
                return

            self.send_response(404)
            self.end_headers()
        except Exception as exc:
            traceback.print_exc()
            err_payload = json.dumps({"error": str(exc), "status": "ERROR"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(err_payload)))
            self.end_headers()
            self.wfile.write(err_payload)

    def send_json(self, obj: Any):
        clean_obj = sanitize_for_json(obj)
        payload = json.dumps(clean_obj, allow_nan=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def launch_dashboard_server(
    port: int = 8050,
    config: Optional[DigitalTwinConfig] = None,
):
    """Launch interactive standalone research dashboard HTTP server."""
    cfg = config or DigitalTwinConfig()
    engine = ReplayEngine(config=cfg)
    DashboardHTTPRequestHandler.engine = engine
    DashboardHTTPRequestHandler.config = cfg

    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DashboardHTTPRequestHandler)
    print(f"======================================================================")
    print(f"NEUROACTION DIGITAL TWIN DASHBOARD RUNNING")
    print(f"Open URL in browser: http://127.0.0.1:{port}")
    print(f"Serving 6-panel interactive digital twin research interface.")
    print(f"Press Ctrl+C to terminate.")
    print(f"======================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAction Digital Twin Research Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind HTTP server to")
    parser.add_argument("--snapshot", type=str, default=None, help="Generate static figure snapshot for trial ID")
    parser.add_argument("--output", type=str, default="reports/phase_4/figures/dashboard_snapshot.png", help="Output path for snapshot")
    args = parser.parse_args()

    if args.snapshot:
        generate_dashboard_figure(trial_id=args.snapshot, output_path=Path(args.output))
        print(f"Generated dashboard snapshot at {args.output}")
    else:
        launch_dashboard_server(port=args.port)
