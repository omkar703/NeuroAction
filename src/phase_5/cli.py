"""Command-line interface for Phase 5 validation pipeline."""

import argparse
import json
from pathlib import Path
import sys

from src.phase_5.config import Phase5Config
from src.phase_5.data_audit import DataAuditor
from src.phase_5.evaluation import Phase5Evaluator
from src.phase_5.external_validation import ExternalValidationManager
from src.phase_5.robustness import RobustnessAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroAction Phase 5 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Audit command
    subparsers.add_parser("audit", help="Audit trial eligibility across tasks")

    # Evaluate command
    subparsers.add_parser("evaluate", help="Run full Leave-One-Subject-Out evaluation")

    # Robustness command
    subparsers.add_parser("robustness", help="Run parameter and subgroup robustness analysis")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and validate external dataset")
    ingest_parser.add_argument("--csv", type=str, required=True, help="Path to external CSV dataset")

    # Generate template command
    template_parser = subparsers.add_parser("generate-template", help="Generate synthetic external validation template")
    template_parser.add_argument("--output", type=str, default="data/sample_external_validation_synthetic.csv")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Phase5Config()

    if args.command == "audit":
        auditor = DataAuditor(config=config)
        summary = auditor.get_summary_statistics()
        print("=" * 70)
        print("NEUROACTION PHASE 5 DATA & ELIGIBILITY AUDIT")
        print("=" * 70)
        for k, v in summary.items():
            print(f"  {k:40s}: {v}")
        df = auditor.build_eligibility_dataframe()
        out_path = config.output_dir / "eligibility_matrix.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved eligibility matrix to {out_path}")

    elif args.command == "evaluate":
        evaluator = Phase5Evaluator(config=config)
        print("Running Phase 5 LOSO Cohort Evaluation across 269 trials...")
        df, summary = evaluator.run_cohort_evaluation()
        print("=" * 70)
        print("PHASE 5 LOSO EVALUATION SUMMARY")
        print("=" * 70)
        for k, v in summary.items():
            print(f"  {k:40s}: {v}")
        print(f"\nEvaluation matrix saved to {config.output_dir / 'phase_5_evaluation_matrix.csv'}")

    elif args.command == "robustness":
        matrix_path = config.output_dir / "phase_5_evaluation_matrix.csv"
        if not matrix_path.exists():
            evaluator = Phase5Evaluator(config=config)
            df, _ = evaluator.run_cohort_evaluation()
        else:
            import pandas as pd
            df = pd.read_csv(matrix_path)

        print("=" * 70)
        print("ROBUSTNESS ANALYSIS: PARTICIPANT BREAKDOWN")
        print("=" * 70)
        part_df = RobustnessAnalyzer.evaluate_participant_breakdown(df)
        print(part_df.to_string(index=False))

        print("\n" + "=" * 70)
        print("ROBUSTNESS ANALYSIS: TARGET TOLERANCE GRID")
        print("=" * 70)
        tol_df = RobustnessAnalyzer.evaluate_tolerance_sensitivity_grid(df)
        print(tol_df.to_string(index=False))

    elif args.command == "ingest":
        csv_path = Path(args.csv)
        valid, report = ExternalValidationManager.validate_dataset_file(csv_path)
        print("=" * 70)
        print("EXTERNAL DATASET INGESTION VALIDATION REPORT")
        print("=" * 70)
        print(f"  Validation Status: {'SUCCESS' if valid else 'FAILED'}")
        for k, v in report.items():
            print(f"  {k:25s}: {v}")
        if not valid:
            sys.exit(1)

    elif args.command == "generate-template":
        out_path = Path(args.output)
        path = ExternalValidationManager.generate_synthetic_template(out_path)
        print(f"Generated synthetic external validation template at: {path}")


if __name__ == "__main__":
    main()
