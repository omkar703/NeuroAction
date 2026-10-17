"""Command-line interface (CLI) for Phase 4 Digital Twin replay and analysis."""

import argparse
from pathlib import Path
import sys

from src.phase_4.config import DigitalTwinConfig
from src.phase_4.dashboard import launch_dashboard_server, generate_dashboard_figure
from src.phase_4.event_logger import EventLogger
from src.phase_4.replay_engine import ReplayEngine
from src.phase_4.trial_registry import TrialRegistry


def main():
    parser = argparse.ArgumentParser(
        prog="neuroaction-twin",
        description="NeuroAction Phase 4 — Neural Intention–Action Digital Twin CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Command 1: list-trials
    p_list = subparsers.add_parser("list-trials", help="List available valid trials")
    p_list.add_argument("--subject", type=str, default=None, help="Filter by participant ID (e.g. sub-01)")

    # Command 2: replay
    p_replay = subparsers.add_parser("replay", help="Run deterministic replay of a single trial")
    p_replay.add_argument("--trial-id", type=str, default="sub-01_run-01_L1_tr00", help="Trial ID to replay")
    p_replay.add_argument("--protocol", type=str, default="kinematic_onset", choices=["kinematic_onset", "event_locked"])
    p_replay.add_argument("--export", action="store_true", help="Export summary and event log CSV")

    # Command 3: dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch interactive 6-panel research dashboard")
    p_dash.add_argument("--port", type=int, default=8050, help="Port to bind dashboard server to")

    # Command 4: snapshot
    p_snap = subparsers.add_parser("snapshot", help="Export static publication-grade 6-panel figure")
    p_snap.add_argument("--trial-id", type=str, default="sub-01_run-01_L1_tr00", help="Trial ID to capture")
    p_snap.add_argument("--output", type=str, default="reports/phase_4/figures/figure_11_dashboard_snapshot.png")

    args = parser.parse_args()

    cfg = DigitalTwinConfig()
    registry = TrialRegistry(manifest_path=cfg.manifest_path)

    if args.command == "list-trials":
        trials = registry.list_trials(participant_id=args.subject, valid_only=True)
        print(f"Total valid trials matching query: {len(trials)}")
        for t in trials[:20]:
            print(f"  {t}")
        if len(trials) > 20:
            print(f"  ... and {len(trials) - 20} more.")

    elif args.command == "replay":
        print(f"Running deterministic replay for trial: {args.trial_id} (Protocol: {args.protocol})")
        engine = ReplayEngine(config=cfg, registry=registry)
        frames = list(engine.stream_trial(trial_id=args.trial_id, protocol=args.protocol))
        print(f"Successfully simulated {len(frames)} frames.")
        final_frame = frames[-1]
        c = final_frame.fusion_state.conflict_assessment
        print(f"Final State:")
        print(f"  Replay Clock: {final_frame.timestamp_ms:.1f} ms")
        print(f"  Decoded Goal: {final_frame.fusion_state.neural_result.predicted_goal.upper()}")
        print(f"  Conflict State: {c.conflict_state}")
        print(f"  Endpoint-Target Distance: {c.endpoint_target_distance_mm:.1f} mm")
        print(f"  Directional Angle: {c.directional_disagreement_deg:.1f}°")

        if args.export:
            logger = EventLogger(output_dir=cfg.demo_dir)
            p_log = logger.export_event_log_csv(frames)
            p_sum = logger.export_trial_summary_json(final_frame)
            p_man = logger.export_reproducibility_manifest(cfg)
            print(f"Exported artifacts:")
            print(f"  Event Log: {p_log}")
            print(f"  Trial Summary: {p_sum}")
            print(f"  Manifest: {p_man}")

    elif args.command == "dashboard":
        launch_dashboard_server(port=args.port, config=cfg)

    elif args.command == "snapshot":
        generate_dashboard_figure(trial_id=args.trial_id, output_path=Path(args.output), config=cfg)
        print(f"Snapshot saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
