"""Run an offline, deterministic context-compaction demonstration.

Run with ``python -m scripts.demo_context_compaction``. The scripted client
makes three read-only tool calls then returns a final answer. A deliberately
small message threshold forces compaction before the third call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Set this before importing project configuration so the scenario stays offline.
os.environ["LLM_PROVIDER"] = "fake"

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest

SYSTEM_PROMPT = "Stable system instruction: triage safely."
TICKET_ID = "COMPACT-1"
TICKET_CONTEXT = "Stable original ticket: reproduce the compaction path."
COMPACTION_THRESHOLD = 4
TRACE_PATH = Path("traces/demo_context_compaction.jsonl")
CALL_USAGE = TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5, reported=True)


class RecordingScriptedClient(ModelClient):
    """A fixed four-call client that retains each context it was sent."""

    def __init__(self) -> None:
        self.call_count = 0
        self.message_snapshots: list[list[dict[str, Any]]] = []

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> NormalizedResponse:
        self.message_snapshots.append([dict(message) for message in messages])
        self.call_count += 1
        if self.call_count <= 3:
            return NormalizedResponse(
                tool_requests=[
                    ToolRequest(
                        id=f"lookup-{self.call_count}",
                        name="lookup",
                        arguments={"ticket_id": TICKET_ID},
                    )
                ],
                usage=CALL_USAGE,
            )
        return NormalizedResponse(text="Compaction run completed.", usage=CALL_USAGE)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            description="Reads deterministic ticket evidence.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=lambda ticket_id: {"ticket_id": ticket_id, "status": "open"},
        )
    )
    return registry


def main() -> int:
    """Execute the scenario and print reviewable compaction evidence."""
    client = RecordingScriptedClient()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please triage ticket {TICKET_ID}:\n\n{TICKET_CONTEXT}",
        },
    ]
    answer, usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=_registry(),
        compaction_threshold=COMPACTION_THRESHOLD,
        trace_path=TRACE_PATH,
    )
    record = json.loads(TRACE_PATH.read_text(encoding="utf-8").splitlines()[-1])
    compactions = [
        event for event in record["events"] if event["type"] == "history_compacted"
    ]
    post_compaction_context = client.message_snapshots[2]
    post_compaction_iterations = [
        item["number"]
        for item in record["iterations"]
        if item["iteration_type"] == "model_call" and item["number"] > 2
    ]

    print("Context compaction demonstration (offline; scripted fake model)")
    print(f"Threshold: {COMPACTION_THRESHOLD} messages")
    for event in compactions:
        print(
            "Compaction: "
            f"{event['messages_before']} -> {event['messages_after']} messages "
            f"(dropped {event['dropped_messages']}); "
            f"tokens={event['cumulative_usage']['total_tokens']}; "
            f"tool_calls={event['tool_call_counts']}"
        )
    print(
        "Stable context retained: "
        f"system={post_compaction_context[0]['content'] == SYSTEM_PROMPT}, "
        f"ticket={TICKET_CONTEXT in post_compaction_context[1]['content']}"
    )
    print(
        f"Valid model iterations after first compaction: {post_compaction_iterations}"
    )
    print(f"Final answer: {answer}")
    print(f"Final cumulative tokens: {usage.total_tokens}; lookup calls: 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
