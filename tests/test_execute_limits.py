import pytest

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.exceptions import (
    IterationLimitExceededError,
    TokenAccountingError,
    TokenLimitExceededError,
    ToolCallLimitExceededError,
)
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient


# Helper mock tool
def dummy_tool_func(ticket_id: str):
    return {"status": "ok", "ticket_id": ticket_id}


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="get_ticket",
            description="Fetches ticket details.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=dummy_tool_func,
        )
    )
    return reg


def test_iteration_limit_exceeded(registry):
    # Fake client always requests a tool, so loop never completes naturally
    response_with_tool = NormalizedResponse(
        text="Looking up ticket...",
        tool_requests=[
            ToolRequest(id="call_1", name="get_ticket", arguments={"ticket_id": "T101"})
        ],
        usage=TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, reported=True
        ),
    )
    client = FakeModelClient(canned_response=response_with_tool)

    messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Please triage ticket T101:\n\nTest ticket"},
    ]
    with pytest.raises(IterationLimitExceededError) as exc_info:
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=3,
            max_total_tokens=1000,
            max_calls_per_tool=10,
        )

    assert "Maximum iteration limit reached" in str(exc_info.value)
    assert "cap is 3" in str(exc_info.value)


def test_token_limit_exceeded(registry):
    # Each call returns 60 tokens. Max tokens is 100. Call 2 should exceed limit (120 > 100).
    response_high_tokens = NormalizedResponse(
        text="Looking up ticket...",
        tool_requests=[
            ToolRequest(id="call_1", name="get_ticket", arguments={"ticket_id": "T101"})
        ],
        usage=TokenUsage(
            input_tokens=30, output_tokens=30, total_tokens=60, reported=True
        ),
    )
    client = FakeModelClient(canned_response=response_high_tokens)

    messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Please triage ticket T101:\n\nTest ticket"},
    ]
    with pytest.raises(TokenLimitExceededError) as exc_info:
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=10,
            max_total_tokens=100,
            max_calls_per_tool=10,
        )

    assert "Maximum token limit reached" in str(exc_info.value)
    assert "observed 120 total tokens, cap is 100" in str(exc_info.value)


def test_unreported_token_metadata(registry):
    # Response has reported=False
    response_unreported = NormalizedResponse(
        text="Some response",
        tool_requests=[],
        usage=TokenUsage(
            input_tokens=0, output_tokens=0, total_tokens=0, reported=False
        ),
    )
    client = FakeModelClient(canned_response=response_unreported)

    messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Please triage ticket T101:\n\nTest ticket"},
    ]
    with pytest.raises(TokenAccountingError) as exc_info:
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=5,
            max_total_tokens=1000,
            max_calls_per_tool=5,
        )

    assert "LLM provider failed to return exact token usage metadata" in str(
        exc_info.value
    )


def test_per_tool_call_limit_exceeded(registry):
    tool_executed_count = 0

    def counting_tool(ticket_id: str):
        nonlocal tool_executed_count
        tool_executed_count += 1
        return {"status": "ok"}

    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="get_ticket",
            description="Fetches ticket details.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=counting_tool,
        )
    )

    # Fake client requests get_ticket repeatedly
    response_with_tool = NormalizedResponse(
        text="Checking ticket...",
        tool_requests=[
            ToolRequest(id="call_1", name="get_ticket", arguments={"ticket_id": "T101"})
        ],
        usage=TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, reported=True
        ),
    )
    client = FakeModelClient(canned_response=response_with_tool)

    messages = [
        {"role": "system", "content": "Test prompt"},
        {"role": "user", "content": "Please triage ticket T101:\n\nTest ticket"},
    ]
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        run_agent_turn(
            messages,
            client=client,
            registry=reg,
            max_iterations=10,
            max_total_tokens=1000,
            max_calls_per_tool=2,
        )

    assert "Maximum call limit reached for tool 'get_ticket'" in str(exc_info.value)
    assert "cap is 2" in str(exc_info.value)
    # Crucial check: Tool was executed exactly 2 times and stopped BEFORE executing the 3rd time
    assert tool_executed_count == 2


def test_normal_execution_within_limits(registry):
    response_final = NormalizedResponse(
        text="Ticket triage complete. Resolved issue.",
        tool_requests=[],
        usage=TokenUsage(
            input_tokens=15, output_tokens=15, total_tokens=30, reported=True
        ),
    )
    client = FakeModelClient(canned_response=response_final)

    messages = [
        {"role": "system", "content": "Test prompt"},
        {"role": "user", "content": "Please triage ticket T101:\n\nTest ticket"},
    ]
    answer, usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        max_iterations=5,
        max_total_tokens=1000,
        max_calls_per_tool=5,
    )

    assert answer == "Ticket triage complete. Resolved issue."
    assert usage.total_tokens == 30
