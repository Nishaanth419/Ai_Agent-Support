"""Live, human-readable console rendering of an agent turn as it runs.

``scripts/trace_viewer.py`` renders a finished run tree from
``traces/agent_runs.jsonl`` after the fact. ``LiveTraceView`` prints the same
kind of detail -- which tools were called, with what arguments, how long
each step took, what each tool returned -- to the terminal as the run
happens, so a caller (an interactive REPL, a demo script) doesn't have to
re-run the trace viewer to see what the agent just did.

``run_agent_turn`` (see ``app/agent.py``) calls these hooks at the same
points where it already calls ``RunTracer``; this view holds no state about
limits or termination logic itself, it only formats what it's told.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from src.core.trace_format import (
    display,
    format_duration,
    format_json_block,
    format_tokens,
)


class LiveTraceView:
    """Prints one readable line (or small block) per agent-loop step."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._out = stream or sys.stdout

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def turn_started(self) -> None:
        self._print()
        self._print("┌─ Agent trace " + "─" * 50)

    def model_call(
        self,
        *,
        iteration: int,
        duration_ms: float | None,
        usage: dict[str, Any] | None,
        requested_tools: list[dict[str, Any]],
    ) -> None:
        self._print(
            f"│ Loop {iteration} · model call "
            f"— {format_duration(duration_ms)}, tokens {format_tokens(usage)}"
        )

        if not requested_tools:
            self._print("│   → no tool calls; finalizing answer")
            return

        for tool in requested_tools:
            name = display(tool.get("name"))
            arguments = display(tool.get("validated_arguments", tool.get("arguments")))
            validation_error = tool.get("validation_error")
            status = (
                f"INVALID ARGS: {validation_error}" if validation_error else "requested"
            )
            self._print(f"│   → tool: {name}  args: {arguments}  [{status}]")

    def tool_started(self, *, name: str) -> None:
        self._print(f"│   ⏳ running {name} ...")

    def tool_completed(
        self,
        *,
        name: str,
        duration_ms: float | None,
        success: bool,
        result: Any,
    ) -> None:
        icon = "✓" if success else "✗"
        self._print(f"│   {icon} {name} — {format_duration(duration_ms)}")
        self._print("│     Result:")
        for line in format_json_block(result, line_prefix="│       ").splitlines():
            self._print(line)

    def compaction(
        self,
        *,
        before: int,
        after: int,
        dropped: int,
        llm_summary_applied: bool = False,
    ) -> None:
        summary_note = (
            " -- condensed into an LLM-written summary"
            if llm_summary_applied
            else " -- condensed into a summary"
            if dropped
            else ""
        )
        self._print(
            f"│ ⚠ history compacted: {before} → {after} messages "
            f"({dropped} dropped{summary_note})"
        )

    def turn_finished(
        self, *, terminal_reason: str, usage: dict[str, Any] | None
    ) -> None:
        self._print(f"└─ Terminal: {terminal_reason} | tokens {format_tokens(usage)}")

    def run_failed(self, *, error_type: str, message: str) -> None:
        self._print(f"└─ ✗ {error_type}: {message}")
