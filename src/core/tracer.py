"""Structured, append-only audit traces for agent runs.

The default trace stream is ``traces/agent_runs.jsonl``.  Each line is a complete
JSON object, so a failed process never invalidates records written by earlier runs.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any


_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|cookie|password|secret|token|credential)", re.I
)
_USAGE_KEYS = {"input_tokens", "output_tokens", "total_tokens"}
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def redact(value: Any, key: str | None = None) -> Any:
    """Return JSON-safe data without credentials or obvious contact details."""
    if key and key.lower() not in _USAGE_KEYS and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): redact(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _EMAIL.sub("[REDACTED_EMAIL]", value)
    return value


class RunTracer:
    """Collect one run record and atomically append it to a JSONL trace stream."""

    def __init__(
        self,
        path: str | Path,
        configuration: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.started_at = datetime.now(UTC)
        self._started_monotonic = monotonic()
        self.record: dict[str, Any] = {
            "run_id": run_id or str(uuid.uuid4()),
            "configuration": redact(configuration),
            "started_at": self.started_at.isoformat(),
            "ended_at": None,
            "terminal_reason": None,
            "duration_ms": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "iterations": [],
            "events": [],
        }
        self._finalized = False

    def elapsed_ms(self) -> int:
        return round((monotonic() - self._started_monotonic) * 1000)

    def add_iteration(
        self,
        *,
        number: int,
        iteration_type: str = "model_call",
        model_duration_ms: int | None = None,
        usage: dict[str, int] | None = None,
        cumulative_usage: dict[str, int] | None = None,
        requested_tools: list[dict[str, Any]] | None = None,
        executed_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add one phase to the trace timeline.

        ``number`` is the agent-loop iteration. The model-call phase and its
        resulting tool-execution phase deliberately share that number.
        """
        if iteration_type not in {"model_call", "tool_execution"}:
            raise ValueError(f"Unsupported iteration type: {iteration_type}")

        entry: dict[str, Any] = {
            "iteration_type": iteration_type,
            "number": number,
            "elapsed_ms": self.elapsed_ms(),
        }
        if iteration_type == "model_call":
            entry.update(
                {
                    "model_duration_ms": model_duration_ms,
                    "usage": usage or {},
                    "cumulative_usage": cumulative_usage or {},
                    "requested_tools": requested_tools or [],
                }
            )
        else:
            entry["executed_tools"] = executed_tools or []
        self.record["iterations"].append(redact(entry))

    def add_event(self, event_type: str, **details: Any) -> None:
        self.record["events"].append(
            redact({"type": event_type, "elapsed_ms": self.elapsed_ms(), **details})
        )

    def finalize(
        self, terminal_reason: str, usage: dict[str, int], **details: Any
    ) -> None:
        if self._finalized:
            return
        self.record["ended_at"] = datetime.now(UTC).isoformat()
        self.record["duration_ms"] = self.elapsed_ms()
        self.record["terminal_reason"] = terminal_reason
        self.record["usage"] = redact(usage)
        if details:
            self.record["terminal_details"] = redact(details)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND plus one os.write keeps separate runs from overwriting each other.
        payload = (
            json.dumps(self.record, ensure_ascii=False, sort_keys=True, default=str)
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        self._finalized = True
