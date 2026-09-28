"""
Drives an EvalCase through the real agent loop with a scripted, offline
model client, and records every tool dispatch so evals/checks.py has
something to grade.

Uses the real tool registry (app.tools.build_tool_registry()) — only the
model is faked. get_refund_policy is a pure lookup and send_reply /
escalate are stubs today. All database-backed tools use SUPPORT_DB_PATH
(or the real project support.db if it is unset); tests/test_eval_cases.py
points it at an isolated temp database for the test session.
"""

import os
from pathlib import Path
from typing import Any

from app.approval import ApprovalDecision
from app.tools import build_tool_registry
from app.agent import run_agent_turn
from src.agent.system_prompt import SYSTEM_PROMPT
from src.llm.base import TokenUsage
from src.llm.fake import ScriptedModelClient

from evals.cases import EvalCase


class _EvalApproval:
    """Fake approval gate used during eval runs.

    Approval policy (documented for evaluators): always approves reply
    proposals without prompting stdin.  escalate no longer requests human
    approval, so this fake only needs to cover send_reply.  This keeps every
    eval case non-interactive and reproducible — the eval suite tests *agent*
    decision quality, not human-in-the-loop flow.
    """

    def request_approval(self, ticket_id: str, message: str) -> ApprovalDecision:
        """Approve a scripted reply proposal in the isolated eval database."""
        return ApprovalDecision(approved=True)


class RecordingToolRegistry:
    """
    Wraps a ToolRegistry and logs every dispatched call, in order, so
    trajectory checks work independent of run_agent_turn()'s own message
    history (the grader deliberately ignores that history even though
    run_agent_turn() returns it). Duck-types the two methods
    run_agent_turn() actually calls on a registry — definitions() and
    dispatch() — rather than subclassing ToolRegistry.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def definitions(self):
        return self._inner.definitions()

    def dispatch(self, name: str, arguments):
        result = self._inner.dispatch(name, arguments)
        self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result


def run_case(
    case: EvalCase,
    trace_path: str | Path | None = None,
) -> tuple[str, list[dict[str, Any]], TokenUsage]:
    """Run one EvalCase's script through the real agent loop.

    Returns ``(final_answer, recorded_tool_calls, token_usage)``.

    ``trace_path`` — optional destination for the per-case agent trace.
    When *None* the tracer defaults to ``traces/agent_runs.jsonl``.  The
    runner passes a per-case temp path so eval traces stay isolated from
    the live trace stream.
    """
    client = ScriptedModelClient(list(case.script))
    registry = RecordingToolRegistry(build_tool_registry(_EvalApproval()))

    kwargs: dict[str, Any] = {}
    if trace_path is not None:
        kwargs["trace_path"] = trace_path
    # ScriptedModelClient raises if generate() is called more times than the
    # case script provides for; the LLM compaction-summary polish call is an
    # extra, unscripted call, so it's off here regardless of the live default.
    kwargs["summarize_dropped_history"] = False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please triage ticket {case.ticket_id}:\n\n{case.description}",
        },
    ]
    final_answer, token_usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        **kwargs,
    )
    return final_answer, registry.calls, token_usage


def run_case_isolated(
    case: EvalCase,
    db_path: str | Path,
    trace_path: str | Path | None = None,
) -> tuple[str, list[dict[str, Any]], TokenUsage]:
    """Run one EvalCase with a caller-supplied pre-seeded database path.

    Sets ``SUPPORT_DB_PATH`` in the current process environment for the
    duration of the call, then restores the previous value.  This gives
    the runner control over DB lifetime while using the same
    ``run_case()`` machinery.
    """
    previous = os.environ.get("SUPPORT_DB_PATH")
    os.environ["SUPPORT_DB_PATH"] = str(db_path)
    try:
        return run_case(case, trace_path=trace_path)
    finally:
        if previous is None:
            os.environ.pop("SUPPORT_DB_PATH", None)
        else:
            os.environ["SUPPORT_DB_PATH"] = previous
