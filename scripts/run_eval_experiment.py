"""
Script to execute the controlled behavioral improvement experiment (Card 30)
and generate docs/behavioral-improvement-experiment.md.
"""

import sys
import argparse
from pathlib import Path

from evals.runner import (
    run_controlled_experiment,
    format_experiment_markdown_report,
    run_live_prompt_experiment,
)
from src.agent.system_prompt import SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


REPORT_PATH = REPO_ROOT / "docs" / "behavioral-improvement-experiment.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="run against the configured LLM"
    )
    parser.add_argument("--baseline-prompt-file", type=Path)
    parser.add_argument("--revised-prompt-file", type=Path)
    args = parser.parse_args()

    if args.live:
        if not args.baseline_prompt_file:
            parser.error("--live requires --baseline-prompt-file")
        baseline_prompt = args.baseline_prompt_file.read_text(encoding="utf-8")
        revised_prompt = (
            args.revised_prompt_file.read_text(encoding="utf-8")
            if args.revised_prompt_file
            else SYSTEM_PROMPT
        )
        baseline, revised = run_live_prompt_experiment(baseline_prompt, revised_prompt)
        print("Live prompt experiment (same tool registry and 10-case set)")
        for label, summary in (("Baseline", baseline), ("Revised", revised)):
            print(
                f"{label}: accuracy={summary.decision_accuracy_pct}%, "
                f"trajectory={summary.trajectory_pass_rate_pct}%, "
                f"avg_iterations={summary.avg_iterations}, "
                f"tokens={summary.total_tokens}, "
                f"cost=${summary.total_estimated_cost_usd:.6f}"
            )
        return

    print("Running controlled behavioral experiment (Card 30)...")
    exp = run_controlled_experiment()

    print(f"Git Commit SHA: {exp.commit_sha}")
    print(f"Provider/Model: {exp.provider} / {exp.model}")
    print("\n--- BASELINE METRICS ---")
    print(f"Decision Accuracy:     {exp.baseline.decision_accuracy_pct}%")
    print(f"Trajectory Pass Rate:  {exp.baseline.trajectory_pass_rate_pct}%")
    print(f"Total Iterations:      {exp.baseline.total_iterations}")
    print(f"Total Tokens:          {exp.baseline.total_tokens}")
    print(f"Estimated Cost USD:    ${exp.baseline.total_estimated_cost_usd:.6f}")

    print("\n--- REVISED METRICS ---")
    print(f"Decision Accuracy:     {exp.revised.decision_accuracy_pct}%")
    print(f"Trajectory Pass Rate:  {exp.revised.trajectory_pass_rate_pct}%")
    print(f"Total Iterations:      {exp.revised.total_iterations}")
    print(f"Total Tokens:          {exp.revised.total_tokens}")
    print(f"Estimated Cost USD:    ${exp.revised.total_estimated_cost_usd:.6f}")

    report_markdown = format_experiment_markdown_report(exp)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_markdown, encoding="utf-8")
    print(f"\nReport successfully saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
