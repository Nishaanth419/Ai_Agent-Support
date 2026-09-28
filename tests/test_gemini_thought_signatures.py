"""Thought-signature round-trip tests for stateless Gemini tool calls."""

import copy
from types import SimpleNamespace

from google.genai import types

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest
from src.llm.gemini import GeminiClient

SIGNATURE = b"opaque-gemini-thought-signature"


class CapturingModels:
    def __init__(self, response):
        self.response = response

    def generate_content(self, **kwargs):
        return self.response


class SequenceClient(ModelClient):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.message_snapshots = []

    def generate(self, messages, tools=None):
        self.message_snapshots.append(copy.deepcopy(messages))
        return next(self.responses)


def test_gemini_response_captures_function_call_thought_signature():
    response = SimpleNamespace(
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                finish_reason=None,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_ticket", args={"ticket_id": "2"}
                            ),
                            thought_signature=SIGNATURE,
                        )
                    ],
                ),
            )
        ],
    )
    client = GeminiClient.__new__(GeminiClient)
    client.client = SimpleNamespace(models=CapturingModels(response))
    client.model = "test-model"

    normalized = client.generate(
        messages=[{"role": "user", "content": "Retrieve ticket 2"}]
    )

    assert normalized.tool_requests[0].provider_metadata == {
        "thought_signature": SIGNATURE
    }


def test_gemini_conversion_resends_thought_signature_unchanged():
    client = GeminiClient.__new__(GeminiClient)
    messages = [
        {
            "role": "model",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "get_ticket",
                    "arguments": {"ticket_id": "2"},
                    "provider_metadata": {"thought_signature": SIGNATURE},
                }
            ],
        }
    ]

    _, converted = client._convert_messages(messages)
    part = converted[0].parts[0]

    assert part.function_call.name == "get_ticket"
    assert part.function_call.args == {"ticket_id": "2"}
    assert part.thought_signature == SIGNATURE


def test_agent_history_carries_opaque_provider_metadata_to_next_model_call(tmp_path):
    client = SequenceClient(
        [
            NormalizedResponse(
                tool_requests=[
                    ToolRequest(
                        id="call-1",
                        name="lookup",
                        arguments={"ticket_id": "2"},
                        provider_metadata={"thought_signature": SIGNATURE},
                    )
                ],
                usage=TokenUsage(total_tokens=10, reported=True),
            ),
            NormalizedResponse(
                text="Done",
                usage=TokenUsage(total_tokens=10, reported=True),
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            description="Retrieve a ticket.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=lambda ticket_id: {"ticket_id": ticket_id},
        )
    )

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Please triage ticket 2:\n\nTicket context"},
    ]
    answer, _, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        trace_path=tmp_path / "trace.jsonl",
    )

    replayed_call = client.message_snapshots[1][2]["tool_calls"][0]
    assert answer == "Done"
    assert replayed_call["provider_metadata"] == {"thought_signature": SIGNATURE}
