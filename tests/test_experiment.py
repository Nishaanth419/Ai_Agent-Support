"""
Tests for evals/runner.py and controlled behavioral experiment execution (Card 30).
"""

from evals.runner import (
    build_custom_tool_registry,
    run_case_evaluation,
    run_eval_suite,
    run_controlled_experiment,
    format_experiment_markdown_report,
)
from evals.cases import CASES
from app.tools import build_tool_registry


def test_build_custom_tool_registry_overrides_search_tickets():
    base_reg = build_tool_registry()
    base_desc = base_reg.get("search_tickets").description

    custom_desc = "Custom search tickets description for testing."
    custom_reg = build_custom_tool_registry(custom_desc)

    assert custom_reg.get("search_tickets").description == custom_desc
    assert base_reg.get("search_tickets").description == base_desc


def test_run_case_evaluation_returns_valid_metrics():
    case = CASES[0]
    result = run_case_evaluation(case)

    assert result.case_id == case.id
    assert result.decision_correct is True
    assert result.passed is True
    assert result.iterations > 0
    assert result.total_tokens > 0
    assert isinstance(result.estimated_cost_usd, float)


def test_run_eval_suite_aggregates_results_correctly():
    registry = build_tool_registry()
    summary = run_eval_suite(
        name="Test Suite",
        description="Testing suite runner",
        tool_registry=registry,
        cases=CASES[:3],
    )

    assert summary.total_cases == 3
    assert summary.passed_cases == 3
    assert summary.decision_accuracy_pct == 100.0
    assert summary.trajectory_pass_rate_pct == 100.0
    assert summary.total_tokens > 0
    assert len(summary.case_results) == 3


def test_run_controlled_experiment_executes_baseline_and_revised():
    exp = run_controlled_experiment(
        baseline_desc="Search tickets.",
        revised_desc="Search previous support tickets for relevant examples.",
    )

    assert exp.commit_sha != ""
    assert exp.baseline.total_cases == 11
    assert exp.revised.total_cases == 11
    assert exp.baseline.decision_accuracy_pct == 100.0
    assert exp.revised.decision_accuracy_pct == 100.0


def test_format_experiment_markdown_report_generates_valid_report():
    exp = run_controlled_experiment()
    report = format_experiment_markdown_report(exp)

    assert "# Controlled Behavioral Improvement Experiment Report" in report
    assert "30) [M]" in report
    assert f"`{exp.commit_sha}`" in report
    assert "Comparative Metric Summary" in report
    assert "Honest Evaluation & Detailed Breakdown" in report
    assert "billing_duplicate_charge_refund" in report
