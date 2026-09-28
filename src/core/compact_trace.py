"""Compact, glanceable console rendering of an agent turn as it runs.

This is the default view a customer-care operator sees (``LiveTraceView`` in
``src/core/live_trace.py`` is the opt-in, full-detail alternative reached via
``run_agent.py --verbose``). Same hook interface, same call sites in
``app/agent.py`` -- this class just renders far less per step: one line for
the tool and the handful of arguments an operator actually needs to
recognise the call (e.g. which ticket, which category), then one line for
the outcome. No per-loop model/token bookkeeping, no raw tool result JSON --
an operator reads the ticket-relevant facts in the agent's final answer, not
in the tool's raw payload.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from src.core.trace_format import display, format_duration, summarize_result

# Which arguments are worth showing per tool call, in display order. A tool
# left out of this map (or an argument left out of its list) is simply not
# shown -- e.g. run_python_analysis's `code`/`data` bodies are omitted since
# a snippet doesn't summarize into a few key=value pairs.
TOOL_KEY_PARAMS: dict[str, list[str]] = {
    "get_ticket": ["ticket_id"],
    "search_tickets": ["query", "category"],
    "get_customer_history": ["customer_id"],
    "get_refund_policy": ["book_category"],
    "list_tickets": ["customer_id", "status", "category"],
    "send_reply": ["ticket_id"],
    "escalate": ["ticket_id", "reason"],
    "create_ticket": ["customer_id", "category", "subject"],
}

_MAX_VALUE_LENGTH = 60


def _format_key_params(name: str, arguments: dict[str, Any] | None) -> str:
    keys = TOOL_KEY_PARAMS.get(name)
    if not keys or not arguments:
        return ""

    parts = []
    for key in keys:
        if key not in arguments:
            continue
        value = display(arguments[key])
        if len(value) > _MAX_VALUE_LENGTH:
            value = value[: _MAX_VALUE_LENGTH - 1] + "…"
        parts.append(f"{key}={value}")
    return ", ".join(parts)


class CompactTraceView:
    """Prints one short call line + one short outcome line per tool call."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._out = stream or sys.stdout
        # Args for the tools requested in the current model_call, consumed
        # in order as tool_started fires for each -- see app/agent.py, which
        # calls tool_started/tool_completed for response.tool_requests in
        # the same order it just handed model_call.
        self._pending_args: list[dict[str, Any]] = []

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def turn_started(self) -> None:
        pass

    def model_call(
        self,
        *,
        iteration: int,
        duration_ms: float | None,
        usage: dict[str, Any] | None,
        requested_tools: list[dict[str, Any]],
    ) -> None:
        self._pending_args = [tool.get("arguments") or {} for tool in requested_tools]

    def tool_started(self, *, name: str) -> None:
        arguments = self._pending_args.pop(0) if self._pending_args else {}
        formatted = _format_key_params(name, arguments)
        self._print(f"  → {name}({formatted})")

    def tool_completed(
        self,
        *,
        name: str,
        duration_ms: float | None,
        success: bool,
        result: Any,
    ) -> None:
        if success:
            self._print(f"    ✓ {format_duration(duration_ms)}")
        else:
            self._print(f"    ✗ {summarize_result(result, max_length=120)}")

    def compaction(
        self,
        *,
        before: int,
        after: int,
        dropped: int,
        llm_summary_applied: bool = False,
    ) -> None:
        if dropped:
            self._print(f"  ⚠ condensed older context ({dropped} messages dropped)")

    def turn_finished(
        self, *, terminal_reason: str, usage: dict[str, Any] | None
    ) -> None:
        pass

    def run_failed(self, *, error_type: str, message: str) -> None:
        self._print(f"  ✗ {error_type}: {message}")
