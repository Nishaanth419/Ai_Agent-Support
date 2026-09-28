"""
End-to-end coverage for run_agent_turn's multi-turn chat contract: a
caller (the run_agent.py REPL, in production) keeps a `messages` list
across turns, and each call to run_agent_turn appends to and returns
that same list so the next turn can build on it.
"""

import json

import pytest

from app import tools
from app.agent import run_agent_turn
from app.tools import build_tool_registry
from src.agent.system_prompt import SYSTEM_PROMPT
from src.db.schema import init_db
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import ScriptedModelClient

NOW = "2026-09-01T10:00:00Z"


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (1, 'Ada Lovelace', 'ada@example.com', 'pro', ?)",
        (NOW,),
    )
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (2, 'Grace Hopper', 'grace@example.com', 'free', ?)",
        (NOW,),
    )
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (1, 1, NULL, 'Refund for Salt and Starlight', 'Body one', 'open', ?, ?)",
        (NOW, NOW),
    )
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (2, 2, NULL, 'Where is my order?', 'Body two', 'open', ?, ?)",
        (NOW, NOW),
    )
    conn.commit()
    yield conn
    conn.close()


def _text_only(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        text=text,
        stop_reason="STOP",
        usage=TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, reported=True
        ),
    )


def _tool_call(name: str, call_id: str, **arguments) -> NormalizedResponse:
    return NormalizedResponse(
        text=None,
        tool_requests=[ToolRequest(id=call_id, name=name, arguments=arguments)],
        stop_reason="TOOL_CALLS",
        usage=TokenUsage(
            input_tokens=10, output_tokens=10, total_tokens=20, reported=True
        ),
    )


def test_conversation_carries_messages_across_turns_for_two_different_tickets(database):
    registry = build_tool_registry()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1: operator asks about ticket 1.
    messages.append({"role": "user", "content": "Look into ticket 1 for me."})
    client_1 = ScriptedModelClient(
        [
            _tool_call("get_ticket", "call_1", ticket_id="1"),
            _text_only("Ticket 1 is an open refund request for Salt and Starlight."),
        ]
    )
    answer_1, usage_1, messages = run_agent_turn(
        messages, client=client_1, registry=registry
    )

    assert "Salt and Starlight" in answer_1
    assert usage_1.total_tokens == 40
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "Look into ticket 1 for me."}

    tool_result_1 = json.loads(
        next(m for m in messages if m.get("role") == "tool")["content"]
    )
    assert tool_result_1["result"]["ticket"]["id"] == 1

    # Turn 2: operator pivots to a different ticket entirely, in the same
    # conversation. The prompt's "work from the most recently specified
    # ticket" guidance is what a live model would use here; this scripted
    # client just proves the plumbing carries the new turn correctly.
    messages.append({"role": "user", "content": "Now check ticket 2 as well."})
    client_2 = ScriptedModelClient(
        [
            _tool_call("get_ticket", "call_2", ticket_id="2"),
            _text_only("Ticket 2 is an open order-status question."),
        ]
    )
    answer_2, usage_2, messages = run_agent_turn(
        messages, client=client_2, registry=registry
    )

    assert "order-status" in answer_2
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    tool_results = [
        json.loads(m["content"]) for m in messages if m.get("role") == "tool"
    ]
    seen_ticket_ids = {r["result"]["ticket"]["id"] for r in tool_results}
    assert seen_ticket_ids == {1, 2}

    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    assert user_turns == [
        "Look into ticket 1 for me.",
        "Now check ticket 2 as well.",
    ]
