"""Shared, human-readable formatting for agent-loop trace data.

Both the offline viewer (``scripts/trace_viewer.py``, which reads a finished
run back out of ``traces/agent_runs.jsonl``) and the live console view
(``src/core/live_trace.py``, which prints the same kind of data as a run
happens) format the same underlying shapes: token usage dicts, tool
arguments/results, and durations. Keeping that formatting in one place means
a run watched live and the same run re-inspected later read identically.
"""

from __future__ import annotations

import json
from typing import Any


def display(value: Any, fallback: str = "unknown") -> str:
    """Return a compact human-readable value."""
    if value is None or value == "":
        return fallback

    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)

    return text


def summarize_result(result: Any, max_length: int | None = 160) -> str:
    """Return a summary of a tool result, optionally bounded in length."""
    if result is None:
        return "unknown"

    if isinstance(result, dict):
        if result.get("ok") is False:
            error = result.get("error", "tool failed")
            return f"ERROR: {error}"

        nested = result.get("result")
        if isinstance(nested, dict):
            text = ", ".join(f"{key}={value}" for key, value in nested.items())
        elif nested is not None:
            text = str(nested)
        else:
            text = ", ".join(f"{key}={value}" for key, value in result.items())
    else:
        text = str(result)

    text = " ".join(text.split())

    if max_length is not None and len(text) > max_length:
        return text[: max_length - 1] + "…"

    return text


def format_duration(milliseconds: Any) -> str:
    """Format duration values without inventing missing data."""
    if milliseconds is None:
        return "unknown"

    try:
        value = float(milliseconds)
    except (TypeError, ValueError):
        return "unknown"

    if value < 1000:
        return f"{value:.0f} ms"

    return f"{value / 1000:.3f} s"


def format_json_block(value: Any, line_prefix: str = "") -> str:
    """Return a pretty-printed, multi-line, indented rendering of ``value``.

    Unlike ``summarize_result``, this never truncates or flattens to one
    line -- it's for callers that want to show a tool's full result exactly
    as returned, just laid out so nested objects/arrays are readable.
    ``line_prefix`` is prepended to every line, so this can be embedded
    inside an already-indented block of output.
    """
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = str(value)

    if not line_prefix:
        return text

    return "\n".join(f"{line_prefix}{line}" for line in text.splitlines())


def format_tokens(usage: dict[str, Any] | None) -> str:
    """Format provider-reported token counts."""
    if not isinstance(usage, dict):
        return "input=unknown output=unknown total=unknown"

    return (
        f"input={display(usage.get('input_tokens'))} "
        f"output={display(usage.get('output_tokens'))} "
        f"total={display(usage.get('total_tokens'))}"
    )
