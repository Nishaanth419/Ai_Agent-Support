"""Tests for evals/cases.py, evals/checks.py, and evals/harness.py."""

import pytest

from evals.cases import CASES, REQUIRED_CATEGORIES
from evals.checks import check_case
from evals.harness import run_case
from src.db.schema import init_db


@pytest.fixture(autouse=True)
def _isolated_support_db(tmp_path, monkeypatch):
    """
    Every database-backed tool reads SUPPORT_DB_PATH when no connection
    is injected, which is how the real registry / agent loop invokes
    them. Point it at a small, self-contained temp database so evals
    never depend on (or mutate) the project fixture database.
    """
    db_path = tmp_path / "eval_support.db"
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO customers (name, email, plan, signed_up_at) "
        "VALUES ('Eval Test Customer', 'eval-customer@example.test', "
        "'pro', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (1, 1, (SELECT id FROM ticket_categories WHERE name = 'billing'), "
        "'Duplicate charge', 'Customer was charged twice.', 'open', "
        "'2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z')"
    )
    # Eval scripts can now exercise real state-changing tools, so every
    # documented ticket id must exist and be eligible for its proposed action.
    for ticket_id in (2, 3, 9, 12, 14, 21, 22, 25, 28):
        conn.execute(
            "INSERT INTO tickets "
            "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
            "VALUES (?, 1, (SELECT id FROM ticket_categories WHERE name = 'technical'), "
            "'Eval ticket', 'Synthetic evaluation ticket.', 'open', "
            "'2026-01-03T00:00:00Z', '2026-01-03T00:00:00Z')",
            (ticket_id,),
        )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SUPPORT_DB_PATH", str(db_path))
    yield


# ── case-set shape ───────────────────────────────────────────────────────


def test_at_least_ten_cases():
    assert len(CASES) >= 10


def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_required_categories_are_covered():
    categories = {c.category for c in CASES}
    assert REQUIRED_CATEGORIES <= categories


def test_ambiguous_and_adversarial_are_covered():
    categories = {c.category for c in CASES}
    assert "ambiguous" in categories
    assert "adversarial" in categories


def test_every_case_has_a_supported_expected_decision():
    for case in CASES:
        assert case.expected_decision in ("reply", "escalate", "answer"), case.id


def test_at_least_one_refund_case_requires_policy_before_reply():
    refund_cases = [
        c
        for c in CASES
        if c.expected_decision == "reply"
        and "get_refund_policy" in c.trajectory.required_before_reply
    ]
    assert len(refund_cases) >= 1


def test_reply_cases_specify_required_concepts():
    # AC: "Reply cases specify required content concepts ... rather than
    # requiring exact prose." Every reply case should have at least one.
    for case in CASES:
        if case.expected_decision == "reply":
            assert case.required_concepts, f"{case.id} has no required_concepts"


# ── running each case's script through the real agent loop ──────────────


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case_script_meets_its_own_expectations(case):
    final_answer, calls, _token_usage = run_case(case)
    result = check_case(case, calls, final_answer)
    assert result.passed, f"{case.id}: {result.failures}\nfinal answer: {final_answer}"


# ── proving the grader actually catches violations, not just rubber-stamps ─


def test_grader_catches_missing_refund_policy_lookup():
    case = next(c for c in CASES if c.id == "billing_duplicate_charge_refund")
    # Same trajectory, but with the get_refund_policy call dropped —
    # simulates an agent that drafted a refund reply without checking
    # policy first.
    broken_calls = [
        {
            "name": "send_reply",
            "arguments": {
                "ticket_id": "1",
                "message": "We've refunded the duplicate charge.",
            },
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, broken_calls)
    assert not result.passed
    assert any("get_refund_policy" in f for f in result.failures)


def test_grader_catches_forbidden_send_reply_on_injection_case():
    case = next(c for c in CASES if c.id == "billing_prompt_injection_adversarial")
    # Simulates an agent that followed the injected instruction instead
    # of escalating.
    broken_calls = [
        {
            "name": "send_reply",
            "arguments": {"ticket_id": "3", "message": "Refund issued as requested."},
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, broken_calls)
    assert not result.passed
    assert any("forbidden tool" in f for f in result.failures)
    assert any("expected decision" in f for f in result.failures)


def test_grader_catches_missing_required_concept():
    case = next(c for c in CASES if c.id == "technical_csv_export_bug")
    calls = [
        {
            "name": "send_reply",
            "arguments": {"ticket_id": "9", "message": "We'll look into it."},
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, calls)
    assert not result.passed
    assert any("required concept" in f for f in result.failures)


def test_grader_catches_prohibited_claim():
    case = next(c for c in CASES if c.id == "technical_csv_export_bug")
    calls = [
        {
            "name": "send_reply",
            "arguments": {
                "ticket_id": "9",
                "message": "We've processed a refund for the CSV export bug.",
            },
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, calls)
    assert not result.passed
    assert any("prohibited claim" in f for f in result.failures)


def test_grader_catches_missing_escalation_reason_substring():
    case = next(c for c in CASES if c.id == "ambiguous_vague_no_detail")
    calls = [
        {
            "name": "escalate",
            "arguments": {"ticket_id": "25", "reason": "Not sure what this is about."},
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, calls)
    assert not result.passed
    assert any("escalate reason" in f for f in result.failures)


def test_grader_flags_a_failed_tool_call():
    case = next(c for c in CASES if c.id == "billing_duplicate_charge_refund")
    calls = [
        {
            "name": "get_refund_policy",
            "arguments": {"plan": "gold"},
            "result": {"ok": False, "error": "Invalid plan tier: 'gold'."},
        },
        {
            "name": "send_reply",
            "arguments": {"ticket_id": "1", "message": "Refund for duplicate charge."},
            "result": {"ok": True, "result": {}},
        },
    ]
    result = check_case(case, calls)
    assert not result.passed
    assert any("tool call 'get_refund_policy' failed" in f for f in result.failures)


# ── check_trace_trajectory — proving incorrect traces cause failures ──────────


def test_trace_trajectory_catches_missing_final_answer():
    """A trace with no final_answer event must fail."""
    from evals.runner import check_trace_trajectory

    case = next(c for c in CASES if c.id == "billing_duplicate_charge_refund")
    # Typical trace for this case, but final_answer stripped out.
    event_types = [
        "model_response_received",
        "tool_started",
        "tool_completed",
        "model_response_received",
        "tool_started",
        "tool_completed",
        "model_response_received",
        "tool_started",
        "tool_completed",
        # ← final_answer intentionally absent
    ]
    failures = check_trace_trajectory(case, event_types)
    assert any("final_answer" in f for f in failures), (
        f"Expected a missing-final_answer failure, got: {failures}"
    )


def test_trace_trajectory_catches_too_few_tool_started_for_required_tools():
    """When required_tools demands N tools but the trace has fewer tool_started
    events, the trajectory check must flag it."""
    from evals.runner import check_trace_trajectory

    # technical_csv_export_bug requires search_tickets (1 required tool).
    # Simulate a trace with zero tool_started events.
    case = next(c for c in CASES if c.id == "technical_csv_export_bug")
    event_types = [
        "model_response_received",
        # No tool_started / tool_completed — tool was skipped entirely
        "model_response_received",
        "final_answer",
    ]
    failures = check_trace_trajectory(case, event_types)
    assert any("tool_started" in f for f in failures), (
        f"Expected a tool_started count failure, got: {failures}"
    )


def test_trace_trajectory_catches_started_without_completed():
    """A trace where tool_started count exceeds tool_completed count signals
    an incomplete tool execution and must be flagged."""
    from evals.runner import check_trace_trajectory

    case = next(c for c in CASES if c.id == "feature_request_dark_mode")
    event_types = [
        "model_response_received",
        "tool_started",
        # tool_completed is missing — agent loop was interrupted mid-tool
        "model_response_received",
        "final_answer",
    ]
    failures = check_trace_trajectory(case, event_types)
    assert any("never completed" in f for f in failures), (
        f"Expected an unmatched tool_started failure, got: {failures}"
    )


def test_trace_trajectory_passes_for_correct_event_sequence():
    """A well-formed trace for a known case must produce no trajectory failures."""
    from evals.runner import check_trace_trajectory

    # billing_duplicate_charge_refund: get_ticket + get_refund_policy + send_reply
    case = next(c for c in CASES if c.id == "billing_duplicate_charge_refund")
    event_types = [
        "model_response_received",
        "tool_started",
        "tool_completed",
        "model_response_received",
        "tool_started",
        "tool_completed",
        "model_response_received",
        "tool_started",
        "tool_completed",
        "model_response_received",
        "final_answer",
    ]
    failures = check_trace_trajectory(case, event_types)
    assert failures == [], f"Expected no failures for correct trace, got: {failures}"
