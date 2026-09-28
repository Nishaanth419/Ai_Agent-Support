"""Tests for the success-and-impossible-goal termination demonstration.

Both scenarios run entirely with the fake provider (LLM_PROVIDER=fake set at
module import time below) so no live API key, database, or network call is
required.  All six tests can run in CI with a bare ``python -m pytest``.

Scenarios covered
-----------------
Scenario A — Happy-path completion
    A scripted two-turn client calls get_ticket on turn 1, then returns a
    terminal text answer on turn 2.  Expected outcome: ``terminal_reason``
    is ``"completed"`` and the function returns a non-empty answer string.

Scenario B — Impossible goal / iteration-cap termination
    A FakeModelClient always requests a tool so the loop never produces a
    terminal text response.  With ``max_iterations=3`` the loop raises
    ``IterationLimitExceededError`` and writes ``terminal_reason =
    "execution_limit"`` to the trace.  The run must stop without hanging,
    without a recursion overflow, and without an unhandled exception of any
    other type.
"""

from __future__ import annotations

import json
import os

# Force the fake provider before any project import.
os.environ["LLM_PROVIDER"] = "fake"

import pytest

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.exceptions import IterationLimitExceededError
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient, ScriptedModelClient

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

TICKET_ID = "T-DEMO-001"
TICKET_CONTEXT = (
    "Customer: Ananya Rao | Plan: pro\n"
    "Subject: Incorrect charge\n"
    "Body: I was billed $99 but my plan is $49."
)
SYSTEM_PROMPT = "You are a support triage agent. Use tools then reply."


def _seed_messages(system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please triage ticket {TICKET_ID}:\n\n{TICKET_CONTEXT}",
        },
    ]


# Per-call token usage reported by the fake model — deterministic.
_USAGE = TokenUsage(input_tokens=15, output_tokens=10, total_tokens=25, reported=True)

_TOOL_REQ = ToolRequest(
    id="call-1", name="get_ticket", arguments={"ticket_id": TICKET_ID}
)

IMPOSSIBLE_CAP = 3  # low enough to trigger the cap quickly


def _make_registry() -> ToolRegistry:
    """Minimal read-only registry used by all tests."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_ticket",
            description="Fetches ticket details.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=lambda ticket_id: {
                "ticket_id": ticket_id,
                "status": "open",
                "category": "billing",
            },
        )
    )
    return registry


def _scripted_client() -> ScriptedModelClient:
    """Two-turn script: tool call then terminal answer."""
    return ScriptedModelClient(
        [
            NormalizedResponse(tool_requests=[_TOOL_REQ], usage=_USAGE),
            NormalizedResponse(
                text="Triage: billing overcharge. Recommend $50 credit.",
                usage=_USAGE,
            ),
        ]
    )


def _impossible_client() -> FakeModelClient:
    """Always requests a tool — the loop can never terminate naturally."""
    return FakeModelClient(
        canned_response=NormalizedResponse(
            text="Still checking...",
            tool_requests=[_TOOL_REQ],
            usage=_USAGE,
        )
    )


# ---------------------------------------------------------------------------
# Scenario A — Successful happy-path completion
# ---------------------------------------------------------------------------


def test_success_run_completes_and_returns_answer(tmp_path):
    """Happy-path: run_agent_turn returns a non-empty answer and 'completed' trace."""
    trace = tmp_path / "runs.jsonl"

    answer, usage, _messages = run_agent_turn(
        _seed_messages(),
        client=_scripted_client(),
        registry=_make_registry(),
        trace_path=trace,
        run_id="test-success",
    )

    # Return value checks
    assert isinstance(answer, str)
    assert len(answer) > 0, "Expected a non-empty triage answer"
    assert usage.total_tokens > 0, "Expected non-zero cumulative tokens"

    # Trace record checks
    record = json.loads(trace.read_text().strip())
    assert record["run_id"] == "test-success"
    assert record["terminal_reason"] == "completed"
    assert record["usage"]["total_tokens"] == usage.total_tokens


def test_success_run_diagnostics_structure(tmp_path):
    """Diagnostic events have expected fields and the final reason is 'completed'."""
    diagnostics: list[dict] = []

    run_agent_turn(
        _seed_messages(),
        client=_scripted_client(),
        registry=_make_registry(),
        trace_path=tmp_path / "runs.jsonl",
        verbose_diagnostics=True,
        diagnostic_sink=diagnostics.append,
    )

    model_calls = [d for d in diagnostics if d["event"] == "model_call"]
    terminations = [d for d in diagnostics if d["event"] == "termination"]

    # There must be at least one model-call event and exactly one termination.
    assert len(model_calls) >= 1, "Expected at least one model_call diagnostic"
    assert len(terminations) == 1, "Expected exactly one termination diagnostic"

    # Every model_call event must carry the required diagnostic fields.
    required_fields = {
        "event",
        "iteration",
        "message_count",
        "call_tokens",
        "cumulative_tokens",
        "requested_tool_name",
        "requested_tool_names",
        "terminal",
    }
    for mc in model_calls:
        missing = required_fields - mc.keys()
        assert not missing, f"model_call diagnostic missing fields: {missing}"
        assert mc["call_tokens"] > 0
        assert mc["cumulative_tokens"] > 0
        assert mc["message_count"] >= 2  # at least system + user

    # The termination event must report "completed".
    term = terminations[0]
    assert term["terminal_reason"] == "completed"
    assert term["cumulative_tokens"] > 0


# ---------------------------------------------------------------------------
# Scenario B — Impossible goal / iteration-cap termination
# ---------------------------------------------------------------------------


def test_impossible_run_raises_iteration_limit(tmp_path):
    """IterationLimitExceededError is raised; message names the cap."""
    with pytest.raises(IterationLimitExceededError) as exc_info:
        run_agent_turn(
            _seed_messages(),
            client=_impossible_client(),
            registry=_make_registry(),
            max_iterations=IMPOSSIBLE_CAP,
            max_total_tokens=100_000,
            max_calls_per_tool=100,
            trace_path=tmp_path / "runs.jsonl",
        )

    error_msg = str(exc_info.value)
    assert "Maximum iteration limit reached" in error_msg
    assert f"cap is {IMPOSSIBLE_CAP}" in error_msg


def test_impossible_run_stops_without_hanging(tmp_path):
    """Run terminates cleanly and trace shows 'execution_limit' with correct iteration count."""
    trace = tmp_path / "runs.jsonl"

    with pytest.raises(IterationLimitExceededError):
        run_agent_turn(
            _seed_messages(),
            client=_impossible_client(),
            registry=_make_registry(),
            max_iterations=IMPOSSIBLE_CAP,
            max_total_tokens=100_000,
            max_calls_per_tool=100,
            trace_path=trace,
            run_id="test-impossible",
        )

    record = json.loads(trace.read_text().strip())
    assert record["run_id"] == "test-impossible"
    assert record["terminal_reason"] == "execution_limit"

    # The agent must have completed exactly IMPOSSIBLE_CAP model-call iterations
    # before the cap was hit.
    iteration_limit_event = next(
        (e for e in record["events"] if e["type"] == "iteration_limit_hit"), None
    )
    assert iteration_limit_event is not None, "Expected 'iteration_limit_hit' event"
    assert iteration_limit_event["completed_iterations"] == IMPOSSIBLE_CAP
    assert iteration_limit_event["max_iterations"] == IMPOSSIBLE_CAP

    # Verify the trace records the exact configuration that was enforced,
    # so reviewers can confirm which caps were active without checking secrets.
    cfg = record["configuration"]
    assert cfg["max_iterations"] == IMPOSSIBLE_CAP
    # max_total_tokens contains the substring "token" which the tracer's
    # sensitive-key regex matches, so it is intentionally redacted in the
    # trace.  Assert the redacted marker rather than the raw integer to
    # confirm the security rule is applied consistently.
    assert cfg["max_total_tokens"] == "[REDACTED]"
    assert cfg["max_calls_per_tool"] == 100
    # Credentials must never appear in trace — not even as a key name.
    assert "api_key" not in cfg


def test_impossible_run_diagnostics_show_every_iteration(tmp_path):
    """Diagnostics include one model_call per iteration and a correct termination event."""
    diagnostics: list[dict] = []

    with pytest.raises(IterationLimitExceededError):
        run_agent_turn(
            _seed_messages(),
            client=_impossible_client(),
            registry=_make_registry(),
            max_iterations=IMPOSSIBLE_CAP,
            max_total_tokens=100_000,
            max_calls_per_tool=100,
            trace_path=tmp_path / "runs.jsonl",
            verbose_diagnostics=True,
            diagnostic_sink=diagnostics.append,
        )

    model_calls = [d for d in diagnostics if d["event"] == "model_call"]
    terminations = [d for d in diagnostics if d["event"] == "termination"]

    # One model_call diagnostic per completed iteration.
    assert len(model_calls) == IMPOSSIBLE_CAP, (
        f"Expected {IMPOSSIBLE_CAP} model_call events, got {len(model_calls)}"
    )

    # Iterations are numbered 1..N in order.
    for i, mc in enumerate(model_calls, start=1):
        assert mc["iteration"] == i, f"Expected iteration {i}, got {mc['iteration']}"

    # Exactly one termination event.
    assert len(terminations) == 1
    assert terminations[0]["terminal_reason"] == "execution_limit"
    assert terminations[0]["iteration"] == IMPOSSIBLE_CAP + 1


def test_both_runs_emit_token_and_message_diagnostics(tmp_path):
    """Both scenarios produce call_tokens, cumulative_tokens, and message_count."""

    def _collect(client, **extra) -> list[dict]:
        diags: list[dict] = []
        try:
            run_agent_turn(
                _seed_messages(),
                client=client,
                registry=_make_registry(),
                trace_path=tmp_path / f"runs-{id(client)}.jsonl",
                verbose_diagnostics=True,
                diagnostic_sink=diags.append,
                **extra,
            )
        except IterationLimitExceededError:
            pass
        return diags

    scenario_a_diags = _collect(_scripted_client())
    scenario_b_diags = _collect(
        _impossible_client(),
        max_iterations=IMPOSSIBLE_CAP,
        max_total_tokens=100_000,
        max_calls_per_tool=100,
    )

    for label, diags in [("A", scenario_a_diags), ("B", scenario_b_diags)]:
        model_calls = [d for d in diags if d["event"] == "model_call"]
        assert model_calls, f"Scenario {label}: no model_call diagnostics"
        for mc in model_calls:
            assert mc["call_tokens"] > 0, f"Scenario {label}: call_tokens must be > 0"
            assert mc["cumulative_tokens"] >= mc["call_tokens"], (
                f"Scenario {label}: cumulative_tokens must be >= call_tokens"
            )
            assert mc["message_count"] >= 2, (
                f"Scenario {label}: message_count must be >= 2 (system + user)"
            )
