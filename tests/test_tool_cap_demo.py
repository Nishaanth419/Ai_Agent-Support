"""Tests for the per-tool call cap demonstration."""

import pytest

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.exceptions import ToolCallLimitExceededError
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient


def test_repeated_read_tool_stops_at_per_tool_cap():
    """Repeated calls to one read tool stop exactly at the configured cap."""
    tool_executed_count = 0
    state_changing_calls = 0

    def get_ticket(ticket_id: str):
        nonlocal tool_executed_count
        tool_executed_count += 1
        return {
            "status": "ok",
            "ticket_id": ticket_id,
        }

    def send_reply(ticket_id: str, message: str):
        nonlocal state_changing_calls
        state_changing_calls += 1
        return {
            "status": "sent",
            "ticket_id": ticket_id,
            "message": message,
        }

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="get_ticket",
            description="Fetches ticket details.",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                },
                "required": ["ticket_id"],
            },
            function=get_ticket,
        )
    )

    registry.register(
        ToolDefinition(
            name="send_reply",
            description="Sends a customer reply.",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["ticket_id", "message"],
            },
            function=send_reply,
        )
    )

    # The fake model always requests the same read-only tool.
    repeated_tool_response = NormalizedResponse(
        text="Checking ticket...",
        tool_requests=[
            ToolRequest(
                id="call_1",
                name="get_ticket",
                arguments={"ticket_id": "T101"},
            )
        ],
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            reported=True,
        ),
    )

    client = FakeModelClient(canned_response=repeated_tool_response)

    cap = 2

    messages = [
        {"role": "system", "content": "Test system prompt"},
        {
            "role": "user",
            "content": "Please triage ticket T101:\n\nTest ticket for tool-cap demonstration.",
        },
    ]

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=10,
            max_total_tokens=1000,
            max_calls_per_tool=cap,
        )

    error_message = str(exc_info.value)

    # The first two calls execute.
    assert tool_executed_count == cap

    # The prohibited third call must not dispatch the tool.
    assert tool_executed_count != cap + 1

    # No state-changing tool is invoked during termination.
    assert state_changing_calls == 0

    # The termination explains which tool exceeded the cap.
    assert "get_ticket" in error_message
    assert f"observed {cap + 1} calls" in error_message
    assert f"cap is {cap}" in error_message
