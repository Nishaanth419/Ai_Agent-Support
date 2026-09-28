import json

import pytest

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.exceptions import TokenLimitExceededError
from src.core.tracer import RunTracer
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient, ScriptedModelClient


def _seed_messages(
    ticket_id: str, ticket_context: str, system_prompt: str
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please triage ticket {ticket_id}:\n\n{ticket_context}",
        },
    ]


def registry() -> ToolRegistry:
    value = ToolRegistry()
    value.register(
        ToolDefinition(
            name="lookup",
            description="Looks up a ticket.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=lambda ticket_id: {"ticket_id": ticket_id, "status": "open"},
        )
    )
    return value


def test_trace_is_jsonl_and_captures_tool_activity(tmp_path):
    path = tmp_path / "runs.jsonl"
    response = NormalizedResponse(
        text="done",
        tool_requests=[
            ToolRequest(id="call-1", name="lookup", arguments={"ticket_id": "T1"})
        ],
        usage=TokenUsage(
            input_tokens=2, output_tokens=3, total_tokens=5, reported=True
        ),
    )
    client = FakeModelClient(canned_response=response)

    with pytest.raises(TokenLimitExceededError):
        run_agent_turn(
            _seed_messages("T1", "private@example.com", "prompt"),
            client=client,
            registry=registry(),
            max_total_tokens=5,
            trace_path=path,
            run_id="run-1",
        )

    record = json.loads(path.read_text().strip())
    assert record["run_id"] == "run-1"
    assert record["terminal_reason"] == "execution_limit"
    assert record["usage"]["total_tokens"] == 10
    model_iteration, tool_iteration = record["iterations"][:2]
    assert model_iteration["iteration_type"] == "model_call"
    assert model_iteration["model_duration_ms"] >= 0
    assert model_iteration["requested_tools"][0]["validated_arguments"] == {
        "ticket_id": "T1"
    }
    assert tool_iteration["iteration_type"] == "tool_execution"
    assert model_iteration["number"] == tool_iteration["number"] == 1
    tool = tool_iteration["executed_tools"][0]
    assert tool["duration_ms"] >= 0
    assert tool["success"] is True
    assert tool["result"]["ok"] is True
    assert any(event["type"] == "execution_limit" for event in record["events"])
    assert any(event["type"] == "model_response_received" for event in record["events"])
    assert any(event["type"] == "tool_started" for event in record["events"])
    assert any(event["type"] == "tool_completed" for event in record["events"])


def test_trace_captures_failed_tool_execution(tmp_path):
    path = tmp_path / "runs.jsonl"
    failing_registry = ToolRegistry()
    failing_registry.register(
        ToolDefinition(
            name="lookup",
            description="Looks up a ticket.",
            parameters={"type": "object", "properties": {}},
            function=lambda: (_ for _ in ()).throw(RuntimeError("service unavailable")),
        )
    )
    response = NormalizedResponse(
        text="trying lookup",
        tool_requests=[ToolRequest(id="call-1", name="lookup", arguments={})],
        usage=TokenUsage(
            input_tokens=1, output_tokens=1, total_tokens=2, reported=True
        ),
    )

    with pytest.raises(TokenLimitExceededError):
        run_agent_turn(
            _seed_messages("T1", "context", "prompt"),
            client=FakeModelClient(canned_response=response),
            registry=failing_registry,
            max_total_tokens=2,
            trace_path=path,
        )

    record = json.loads(path.read_text().strip())
    tool = record["iterations"][1]["executed_tools"][0]
    assert tool["success"] is False
    assert "RuntimeError: service unavailable" in tool["error"]
    completed = next(
        event for event in record["events"] if event["type"] == "tool_completed"
    )
    assert completed["success"] is False


def test_tracer_appends_and_redacts_sensitive_values(tmp_path):
    path = tmp_path / "runs.jsonl"
    for run_id in ("one", "two"):
        tracer = RunTracer(path, {"api_key": "not-for-trace"}, run_id=run_id)
        tracer.add_event(
            "human_approval", outcome="approved", email="person@example.com"
        )
        tracer.add_event("compaction", reason="context_limit")
        tracer.finalize(
            "completed", {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["run_id"] for record in records] == ["one", "two"]
    assert records[0]["configuration"]["api_key"] == "[REDACTED]"
    assert records[0]["events"][0]["email"] == "[REDACTED_EMAIL]"
    assert records[0]["usage"]["total_tokens"] == 3


def test_verbose_diagnostics_are_structured_optional_and_secret_free(tmp_path):
    diagnostics = []
    answer, usage, _messages = run_agent_turn(
        _seed_messages(
            "T1", "GEMINI_API_KEY=should-not-appear", "secret system prompt"
        ),
        client=FakeModelClient(
            canned_response=NormalizedResponse(
                text="done",
                usage=TokenUsage(
                    input_tokens=2, output_tokens=3, total_tokens=5, reported=True
                ),
            )
        ),
        registry=registry(),
        trace_path=tmp_path / "runs.jsonl",
        verbose_diagnostics=True,
        diagnostic_sink=diagnostics.append,
    )

    assert answer == "done"
    assert usage.total_tokens == 5
    assert diagnostics == [
        {
            "event": "model_call",
            "iteration": 1,
            "message_count": 2,
            "call_tokens": 5,
            "cumulative_tokens": 5,
            "requested_tool_name": None,
            "requested_tool_names": [],
            "terminal": True,
        },
        {
            "event": "termination",
            "terminal_reason": "completed",
            "iteration": 1,
            "cumulative_tokens": 5,
        },
    ]
    assert "should-not-appear" not in repr(diagnostics)


def test_diagnostic_sink_is_silent_when_diagnostics_are_disabled(tmp_path):
    diagnostics = []
    run_agent_turn(
        _seed_messages("T1", "context", "prompt"),
        client=FakeModelClient(),
        registry=registry(),
        trace_path=tmp_path / "runs.jsonl",
        diagnostic_sink=diagnostics.append,
    )

    assert diagnostics == []


def test_verbose_diagnostics_include_requested_tool_name(tmp_path):
    diagnostics = []
    run_agent_turn(
        _seed_messages("T1", "context", "prompt"),
        client=ScriptedModelClient(
            [
                NormalizedResponse(
                    tool_requests=[
                        ToolRequest(
                            id="call-1", name="lookup", arguments={"ticket_id": "T1"}
                        )
                    ],
                    usage=TokenUsage(total_tokens=2, reported=True),
                ),
                NormalizedResponse(
                    text="done", usage=TokenUsage(total_tokens=3, reported=True)
                ),
            ]
        ),
        registry=registry(),
        trace_path=tmp_path / "runs.jsonl",
        verbose_diagnostics=True,
        diagnostic_sink=diagnostics.append,
    )

    assert diagnostics[0]["requested_tool_name"] == "lookup"
    assert diagnostics[0]["requested_tool_names"] == ["lookup"]
    assert diagnostics[0]["terminal"] is False
    assert diagnostics[1]["message_count"] == 4
    assert diagnostics[1]["call_tokens"] == 3
    assert diagnostics[1]["cumulative_tokens"] == 5


def test_verbose_diagnostics_handle_multiple_tool_requests(tmp_path):
    diagnostics = []
    trace_path = tmp_path / "runs.jsonl"
    reg = registry()
    reg.register(
        ToolDefinition(
            name="customer_info",
            description="Fetches customer details.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
            function=lambda customer_id: {"customer_id": customer_id, "tier": "gold"},
        )
    )

    run_agent_turn(
        _seed_messages("T1", "context", "prompt"),
        client=ScriptedModelClient(
            [
                NormalizedResponse(
                    tool_requests=[
                        ToolRequest(
                            id="call-1", name="lookup", arguments={"ticket_id": "T1"}
                        ),
                        ToolRequest(
                            id="call-2",
                            name="customer_info",
                            arguments={"customer_id": "C1"},
                        ),
                    ],
                    usage=TokenUsage(total_tokens=4, reported=True),
                ),
                NormalizedResponse(
                    text="done", usage=TokenUsage(total_tokens=3, reported=True)
                ),
            ]
        ),
        registry=reg,
        trace_path=trace_path,
        verbose_diagnostics=True,
        diagnostic_sink=diagnostics.append,
    )

    # First iteration requested multiple tools: requested_tool_name points to the primary/first tool
    assert diagnostics[0]["event"] == "model_call"
    assert diagnostics[0]["requested_tool_name"] == "lookup"
    assert diagnostics[0]["requested_tool_names"] == ["lookup", "customer_info"]
    assert diagnostics[0]["terminal"] is False
    trace = json.loads(trace_path.read_text().strip())
    first_model_phase, first_tool_phase = trace["iterations"][:2]
    assert first_model_phase["number"] == first_tool_phase["number"] == 1
    assert len(first_tool_phase["executed_tools"]) == 2
    # Next iteration terminates the loop
    assert diagnostics[1]["event"] == "model_call"
    assert diagnostics[1]["requested_tool_name"] is None
    assert diagnostics[1]["requested_tool_names"] == []
    assert diagnostics[1]["terminal"] is True
    # Termination event
    assert diagnostics[2]["event"] == "termination"
    assert diagnostics[2]["terminal_reason"] == "completed"
