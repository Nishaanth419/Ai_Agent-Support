"""Demonstrate deterministic per-tool call-cap enforcement."""

import os

os.environ["LLM_PROVIDER"] = "fake"

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.exceptions import ToolCallLimitExceededError
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient

CAP = 2


def get_ticket(ticket_id: str):
    """Read-only fake ticket tool."""
    print(f"Executing get_ticket call for {ticket_id}")
    return {
        "status": "ok",
        "ticket_id": ticket_id,
    }


def send_reply(ticket_id: str, message: str):
    """State-changing tool included to prove it is never invoked."""
    print("ERROR: send_reply should not be invoked.")
    return {
        "status": "sent",
        "ticket_id": ticket_id,
        "message": message,
    }


def build_registry() -> ToolRegistry:
    """Build the registry used by the demonstration."""
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

    return registry


def main() -> int:
    """Run the deterministic tool-cap demonstration."""
    response = NormalizedResponse(
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

    client = FakeModelClient(canned_response=response)
    registry = build_registry()

    print("Per-tool call cap demonstration")
    print("--------------------------------")
    print("Fake model repeatedly requests: get_ticket")
    print(f"Configured per-tool cap: {CAP}")
    print()

    try:
        messages = [
            {"role": "system", "content": "Test system prompt"},
            {
                "role": "user",
                "content": "Please triage ticket T101:\n\nDeterministic tool-cap demonstration.",
            },
        ]
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=10,
            max_total_tokens=1000,
            max_calls_per_tool=CAP,
        )
    except ToolCallLimitExceededError as exc:
        print("Termination:")
        print(exc)
        print()
        print("The repeated read-tool call was blocked before dispatch.")
        print("No state-changing tool was invoked.")
        return 0

    print("ERROR: the run completed without reaching the tool-call cap.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
