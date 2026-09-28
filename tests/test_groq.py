import json
from types import SimpleNamespace

import httpx
import pytest
from groq import APIStatusError

from src.core.exceptions import ProviderAPIError
from src.llm.groq import GroqClient


def _tool_use_failed_error() -> APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return APIStatusError(
        "tool call arguments not valid JSON",
        response=response,
        body={
            "error": {
                "message": "Failed to parse tool call arguments as JSON",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
            }
        },
    )


class ChatCompletions:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.call_count = 0

    def create(self, **request):
        self.call_count += 1
        self.request = request
        if callable(self.response):
            return self.response(self.call_count)
        return self.response


def _client_with_response(response):
    completions = ChatCompletions(response)
    client = GroqClient.__new__(GroqClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.model = "test-model"
    return client, completions


def test_groq_converts_agent_history_and_tool_schemas():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="get_ticket",
                                arguments='{"ticket_id":"2"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    client, completions = _client_with_response(response)

    normalized = client.generate(
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Get ticket 2"},
            {
                "role": "model",
                "tool_calls": [
                    {
                        "id": "previous-call",
                        "name": "lookup",
                        "arguments": {"id": "1"},
                        "provider_metadata": {"thought_signature": b"gemini-only"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "previous-call",
                "name": "lookup",
                "content": '{"ok":true}',
            },
        ],
        tools=[
            {
                "name": "get_ticket",
                "description": "Get one ticket.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                    "additionalProperties": False,
                },
            }
        ],
    )

    previous_call = completions.request["messages"][2]["tool_calls"][0]
    assert previous_call == {
        "id": "previous-call",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"id": "1"})},
    }
    assert completions.request["messages"][3]["tool_call_id"] == "previous-call"
    assert (
        completions.request["tools"][0]["function"]["parameters"][
            "additionalProperties"
        ]
        is False
    )
    assert normalized.tool_requests[0].id == "call-1"
    assert normalized.tool_requests[0].arguments == {"ticket_id": "2"}
    assert normalized.stop_reason == "tool_calls"
    assert completions.request["parallel_tool_calls"] is True
    assert normalized.usage.input_tokens == 20
    assert normalized.usage.output_tokens == 5
    assert normalized.usage.total_tokens == 25
    assert normalized.usage.reported is True


def test_groq_rejects_malformed_tool_arguments():
    response = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="get_ticket", arguments="not-json"
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    client, _ = _client_with_response(response)

    with pytest.raises(ProviderAPIError, match="invalid JSON arguments"):
        client.generate(messages=[{"role": "user", "content": "Get a ticket"}])


def test_groq_preserves_multiple_tool_calls_from_one_response():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="get_ticket",
                                arguments='{"ticket_id":"2"}',
                            ),
                        ),
                        SimpleNamespace(
                            id="call-2",
                            function=SimpleNamespace(
                                name="search_tickets",
                                arguments=(
                                    '{"query":"invoice seat count",'
                                    '"category":"billing"}'
                                ),
                            ),
                        ),
                    ],
                ),
            )
        ],
    )
    client, completions = _client_with_response(response)

    normalized = client.generate(
        messages=[{"role": "user", "content": "Investigate ticket 2"}],
        tools=[],
    )

    assert [request.name for request in normalized.tool_requests] == [
        "get_ticket",
        "search_tickets",
    ]
    assert normalized.tool_requests[1].arguments == {
        "query": "invoice seat count",
        "category": "billing",
    }
    assert completions.request["parallel_tool_calls"] is True


def test_groq_retries_and_recovers_from_tool_use_failed():
    success = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="get_ticket", arguments='{"ticket_id":"2"}'
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    def flaky(call_count):
        if call_count == 1:
            raise _tool_use_failed_error()
        return success

    client, completions = _client_with_response(flaky)

    normalized = client.generate(messages=[{"role": "user", "content": "Get ticket 2"}])

    assert completions.call_count == 2
    assert normalized.tool_requests[0].arguments == {"ticket_id": "2"}


def test_groq_raises_after_exhausting_tool_use_failed_retries():
    def always_fails(_call_count):
        raise _tool_use_failed_error()

    client, completions = _client_with_response(always_fails)

    with pytest.raises(ProviderAPIError, match="Groq API error"):
        client.generate(messages=[{"role": "user", "content": "Get ticket 2"}])

    from src.core.config import Config

    assert completions.call_count == Config.GROQ_TOOL_USE_FAILED_MAX_RETRIES + 1
