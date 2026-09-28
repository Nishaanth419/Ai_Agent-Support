import json
from app.agent import (
    _COMPACTION_FACTS_KEY,
    _build_summary_message,
    _describe_tool_call,
    _dedupe_preserve_order,
    _extract_turn_facts,
    _polish_compaction_summary,
    compact_history,
)
from pathlib import Path
from app.agent import run_agent_turn
from app.tools import build_tool_registry
from src.core.config import Config
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest


def test_compaction_below_threshold():
    """Proves non-trigger behavior below the threshold."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ticket"},
        {"role": "model", "content": "thinking"},
    ]
    # Threshold is 5, we have 3. Should not trigger.
    new_messages, dropped = compact_history(messages, threshold=5)

    assert dropped == 0
    assert len(new_messages) == 3
    assert new_messages == messages


def test_compaction_preserves_prefix_and_latest():
    """Proves system prompt, original ticket, and the latest evidence are retained."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ticket"},
        {"role": "model", "tool_calls": [{"id": "1", "name": "get_ticket"}]},
        {"role": "tool", "content": '{"ok": true}'},
        {"role": "model", "tool_calls": [{"id": "2", "name": "search_tickets"}]},
        {"role": "tool", "content": '{"ok": true}'},
    ]

    # Target size: max(4, 4-4) = 4.
    # Current len: 6.
    # It must drop the middle turn (messages[2] and [3]) to get closer to 4,
    # replacing it with one summary message rather than deleting it outright.
    new_messages, dropped = compact_history(messages, threshold=4)

    assert dropped == 2
    assert len(new_messages) == 5
    # Prefix retained
    assert new_messages[0]["role"] == "system"
    assert new_messages[1]["role"] == "user"
    # The dropped turn survives as a fact in the summary, not silently gone
    assert new_messages[2]["role"] == "model"
    assert "get_ticket" in new_messages[2]["content"]
    # Latest turn retained
    assert new_messages[3]["tool_calls"][0]["name"] == "search_tickets"


def test_compaction_preserves_human_rejections():
    """Proves that a turn containing a rejected human approval is never dropped."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ticket"},
        # Turn 1: A rejected reply (Length: 2)
        {"role": "model", "tool_calls": [{"id": "1", "name": "send_reply"}]},
        {
            "role": "tool",
            "content": json.dumps({"status": "rejected", "human_feedback": "No"}),
        },
        # Turn 2: A normal lookup (Length: 2)
        {"role": "model", "tool_calls": [{"id": "2", "name": "get_ticket"}]},
        {"role": "tool", "content": '{"ok": true}'},
        # Turn 3: Latest action (Length: 2)
        {"role": "model", "tool_calls": [{"id": "3", "name": "get_policy"}]},
        {"role": "tool", "content": '{"ok": true}'},
    ]

    new_messages, dropped = compact_history(messages, threshold=3)

    # 1. Verify exact counts (the dropped turn adds one summary message back)
    assert dropped == 2
    assert len(new_messages) == 7

    # 2. Verify Prefix is untouched
    assert new_messages[0]["role"] == "system"
    assert new_messages[1]["role"] == "user"

    # 3. Verify the dropped get_ticket turn survives as a summary fact
    assert new_messages[2]["role"] == "model"
    assert "get_ticket" in new_messages[2]["content"]

    # 4. Verify Turn 1 (Rejection) survived and is in the next slot
    assert new_messages[3]["tool_calls"][0]["name"] == "send_reply"
    assert "rejected" in new_messages[4]["content"]

    # 5. Verify Turn 3 (get_policy) is the preserved last turn
    assert new_messages[5]["tool_calls"][0]["name"] == "get_policy"
    assert new_messages[6]["content"] == '{"ok": true}'


def test_compaction_protocol_validity():
    """Proves that a model message and its parallel tool responses are never separated."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ticket"},
        # Turn 1: One model turn with two parallel tool calls (Length: 3)
        {"role": "model", "tool_calls": [{"id": "1"}, {"id": "2"}]},
        {"role": "tool", "content": "res1"},
        {"role": "tool", "content": "res2"},
        # Turn 2: Latest turn (Length: 1)
        {"role": "model", "content": "done"},
    ]
    # Total messages = 6.

    # Target size: 4. Current len: 6.
    # Drops Turn 1 (length 3), replaced by one summary message.
    new_messages, dropped = compact_history(messages, threshold=4)

    assert dropped == 3
    assert len(new_messages) == 4
    assert new_messages[-1]["content"] == "done"


def test_compaction_buffer_behavior():
    """Verifies that compaction targets threshold - 4 to prevent looping compaction."""
    # Prefix (2) + 6 turns of 2 msgs = 14 messages total
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ticket"},
    ]
    for i in range(6):
        messages.extend(
            [
                {"role": "model", "content": f"turn{i}"},
                {"role": "tool", "content": f"result{i}"},
            ]
        )

    # Threshold 12. Target size = (12 - 4) = 8.
    # To get from 14 to 8, it must drop 3 turns (6 messages), replaced by one
    # summary message carrying their facts forward.
    new_messages, dropped = compact_history(messages, threshold=12)

    assert len(new_messages) == 9
    assert dropped == 6


def test_compaction_preserves_current_user_question_glued_to_earlier_turn():
    """Proves a follow-up question never gets dropped when it has no `model`
    message of its own to anchor a turn, and instead landed on the tail of
    an earlier, otherwise-droppable turn.

    Regression for a real observed failure: after two rounds of compaction,
    the user's actual pending question ("how many tickets does customer 3
    have") was silently deleted while data fetched for it survived orphaned,
    and the agent fell back to re-answering the oldest still-visible
    question (the seed prompt) instead.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "seed question about customer 7 vs 5"},
        # Turn: answered and irrelevant now -- droppable.
        {"role": "model", "tool_calls": [{"id": "1", "name": "create_ticket"}]},
        {"role": "tool", "content": '{"ok": true}'},
        # Turn: also answered -- BUT the next user question got glued to its
        # tail because it has no `model` message to start its own turn.
        {"role": "model", "tool_calls": [{"id": "2", "name": "escalate"}]},
        {"role": "tool", "content": '{"ok": true, "result": {"escalated": true}}'},
        {"role": "user", "content": "how many tickets does customer 3 have?"},
        # Turns from working the *new* question -- data fetched for it.
        {"role": "model", "tool_calls": [{"id": "3", "name": "list_tickets"}]},
        {"role": "tool", "content": '{"ok": true, "result": {"count": 5}}'},
        {"role": "model", "tool_calls": [{"id": "4", "name": "run_python_analysis"}]},
        {"role": "tool", "content": '{"ok": true, "result": {"succeeded": false}}'},
    ]

    new_messages, dropped = compact_history(messages, threshold=8)

    contents = [str(m.get("content", "")) for m in new_messages]
    assert any("customer 3" in c for c in contents), (
        "the live follow-up question must survive compaction even though it "
        "isn't in the structurally-last turn"
    )
    # Sanity: something was actually dropped (old, answered turns).
    assert dropped > 0


def test_run_agent_turn_triggers_compaction(tmp_path: Path):
    """Proves that compaction is natively triggered inside the agent loop and writes to the trace."""
    registry = build_tool_registry()

    # We create a custom Fake Client that loops 3 times to build up message history
    class LoopingFakeClient(ModelClient):
        def __init__(self):
            self.call_count = 0
            self.message_snapshots = []

        def generate(self, messages, tools=None) -> NormalizedResponse:
            self.message_snapshots.append([dict(message) for message in messages])
            self.call_count += 1
            if self.call_count >= 4:
                return NormalizedResponse(
                    text="All done.", usage=TokenUsage(total_tokens=10, reported=True)
                )
            # Request a tool to keep the loop going and build history
            return NormalizedResponse(
                tool_requests=[
                    ToolRequest(
                        id=f"t{self.call_count}",
                        name="get_ticket",
                        arguments={"ticket_id": "1"},
                    )
                ],
                usage=TokenUsage(total_tokens=10, reported=True),
            )

    trace_file = tmp_path / "trace.jsonl"

    # Run with a deliberately tiny threshold (4) to force compaction early
    client = LoopingFakeClient()
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Please triage ticket 1:\n\nTest"},
    ]
    final_answer, usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        compaction_threshold=4,
        trace_path=trace_file,
        # This test is about the compaction trigger/trace mechanics, not the
        # LLM summary polish (covered separately) -- LoopingFakeClient's
        # scripted call_count would otherwise be thrown off by the extra call.
        summarize_dropped_history=False,
    )

    assert final_answer == "All done."
    assert client.call_count == 4  # A terminal model call follows compaction.
    # The third call follows the first compaction. Its stable prefix is verbatim.
    assert client.message_snapshots[2][0] == {"role": "system", "content": "Sys"}
    assert client.message_snapshots[2][1] == {
        "role": "user",
        "content": "Please triage ticket 1:\n\nTest",
    }
    assert usage.total_tokens == 40

    # Read the trace file to verify the event was emitted correctly
    with open(trace_file, "r") as f:
        trace_data = json.loads(f.read())

    events = trace_data.get("events", [])
    compaction_events = [e for e in events if e.get("type") == "history_compacted"]

    assert len(compaction_events) > 0
    first_compaction = compaction_events[0]
    assert first_compaction["messages_before"] == 6
    # +1 vs. the raw 4 kept messages: the dropped turn becomes a summary
    # message instead of being deleted outright.
    assert first_compaction["messages_after"] == 5
    assert first_compaction["dropped_messages"] == 2
    assert first_compaction["llm_summary_applied"] is False
    assert first_compaction["system_prompt_preserved"] is True
    assert first_compaction["seed_user_message_preserved"] is True
    # At compaction no accounting is reset; later usage/calls are legitimate work.
    assert first_compaction["cumulative_usage"]["total_tokens"] == 20
    assert first_compaction["tool_call_counts"] == {"get_ticket": 2}
    model_calls = [
        item
        for item in trace_data["iterations"]
        if item["iteration_type"] == "model_call"
    ]
    assert model_calls[2]["number"] == 3
    assert model_calls[2]["cumulative_usage"]["total_tokens"] == 30


def test_compaction_default_threshold_triggers_at_25th_message():
    """Regression guard for the explicit ask: compaction should trigger once
    the history reaches 25 messages, not the old default of 12."""
    assert Config.COMPACTION_THRESHOLD == 24

    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i == 0 else "model", "content": f"m{i}"} for i in range(23)
    ]
    assert len(messages) == 24
    _new_messages, dropped = compact_history(
        messages, threshold=Config.COMPACTION_THRESHOLD
    )
    assert dropped == 0  # 24 messages must NOT trigger compaction yet

    messages.append({"role": "model", "content": "m24"})
    assert len(messages) == 25
    _new_messages, dropped = compact_history(
        messages, threshold=Config.COMPACTION_THRESHOLD
    )
    assert dropped > 0  # the 25th message must trigger it


def test_describe_tool_call_preserves_human_rejection_feedback_verbatim():
    """The exact fact that mattered in a real incident: a rejected send_reply
    carries the human reviewer's own words forward, unparaphrased, and flags
    that the reply still needs to be revised and retried."""
    tool_message = {
        "role": "tool",
        "tool_call_id": "1",
        "name": "send_reply",
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "status": "rejected",
                    "human_feedback": "write it more formally",
                    "ticket_id": 50,
                },
            }
        ),
    }
    fact = _describe_tool_call(
        "send_reply", {"ticket_id": "50", "message": "hi"}, tool_message
    )
    assert "REJECTED" in fact
    assert "write it more formally" in fact
    assert "50" in fact
    assert "revised and retried" in fact


def test_describe_tool_call_reports_get_ticket_structural_fields():
    tool_message = {
        "role": "tool",
        "tool_call_id": "1",
        "name": "get_ticket",
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "ticket": {
                        "id": 50,
                        "status": "open",
                        "category": "other",
                        "subject": "Need help",
                    }
                },
            }
        ),
    }
    fact = _describe_tool_call("get_ticket", {"ticket_id": "50"}, tool_message)
    assert "status=open" in fact
    assert "category=other" in fact
    assert "Need help" in fact


def test_extract_turn_facts_covers_tool_calls_and_operator_text():
    turn = [
        {"role": "user", "content": "resolve ticket 50"},
        {
            "role": "model",
            "tool_calls": [
                {"id": "1", "name": "get_ticket", "arguments": {"ticket_id": "50"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "get_ticket",
            "content": json.dumps(
                {
                    "ok": True,
                    "result": {
                        "ticket": {
                            "id": 50,
                            "status": "open",
                            "category": "other",
                            "subject": "Need help",
                        }
                    },
                }
            ),
        },
    ]
    facts = _extract_turn_facts(turn)
    assert any("Operator said: resolve ticket 50" == f for f in facts)
    assert any("get_ticket" in f and "status=open" in f for f in facts)


def test_build_summary_message_is_model_role_and_carries_structured_facts():
    message = _build_summary_message(["fact one", "fact two"], max_facts=40)

    assert message["role"] == "model"
    assert "fact one" in message["content"]
    assert "fact two" in message["content"]
    assert message[_COMPACTION_FACTS_KEY] == ["fact one", "fact two"]


def test_build_summary_message_caps_and_dedupes_facts():
    facts = [f"fact {i}" for i in range(5)] + ["fact 4"]  # trailing duplicate
    message = _build_summary_message(facts, max_facts=3)

    # Deduped first, then capped to the most recent max_facts.
    assert message[_COMPACTION_FACTS_KEY] == ["fact 2", "fact 3", "fact 4"]


def test_dedupe_preserve_order():
    assert _dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_compaction_merges_repeated_passes_into_one_summary_message():
    """A long-running conversation compacts more than once. The second pass
    must merge into the existing summary rather than stacking a second one,
    and must not lose the facts the first pass already captured."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "seed"},
        {
            "role": "model",
            "tool_calls": [
                {"id": "1", "name": "get_ticket", "arguments": {"ticket_id": "1"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "get_ticket",
            "content": '{"ok": true}',
        },
        {
            "role": "model",
            "tool_calls": [
                {"id": "2", "name": "get_ticket", "arguments": {"ticket_id": "2"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "name": "get_ticket",
            "content": '{"ok": true}',
        },
        {"role": "model", "content": "final answer"},
    ]

    first_pass, dropped_1 = compact_history(messages, threshold=4)
    summaries = [m for m in first_pass if _COMPACTION_FACTS_KEY in m]
    assert len(summaries) == 1
    assert dropped_1 > 0

    # Grow the history again and compact a second time.
    second_round_messages = first_pass + [
        {
            "role": "model",
            "tool_calls": [
                {"id": "3", "name": "get_ticket", "arguments": {"ticket_id": "3"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "3",
            "name": "get_ticket",
            "content": '{"ok": true}',
        },
        {
            "role": "model",
            "tool_calls": [
                {"id": "4", "name": "get_ticket", "arguments": {"ticket_id": "4"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "4",
            "name": "get_ticket",
            "content": '{"ok": true}',
        },
        {"role": "model", "content": "second final answer"},
    ]
    second_pass, dropped_2 = compact_history(second_round_messages, threshold=4)

    summaries = [m for m in second_pass if _COMPACTION_FACTS_KEY in m]
    assert len(summaries) == 1, "must merge into one summary, never stack a second"
    assert dropped_2 > 0
    # Facts captured by the first pass must still be present after the second.
    first_pass_facts = set(first_pass[2][_COMPACTION_FACTS_KEY])
    assert first_pass_facts.issubset(set(summaries[0][_COMPACTION_FACTS_KEY]))


def test_polish_compaction_summary_uses_llm_and_tracks_tokens():
    class FakeSummarizerClient(ModelClient):
        def generate(self, messages, tools=None):
            return NormalizedResponse(
                text="Ticket 50 is open; ticket 78 is open.",
                usage=TokenUsage(
                    input_tokens=5, output_tokens=3, total_tokens=8, reported=True
                ),
            )

    summary_message = _build_summary_message(
        ["get_ticket(50): status=open", "get_ticket(78): status=open"], max_facts=40
    )
    cumulative_tokens = TokenUsage()

    applied = _polish_compaction_summary(
        summary_message, FakeSummarizerClient(), cumulative_tokens
    )

    assert applied is True
    assert "Ticket 50 is open" in summary_message["content"]
    assert cumulative_tokens.total_tokens == 8


def test_polish_compaction_summary_falls_back_to_structured_facts_on_failure():
    class FailingClient(ModelClient):
        def generate(self, messages, tools=None):
            raise RuntimeError("provider is down")

    facts = ["get_ticket(50): status=open"]
    summary_message = _build_summary_message(facts, max_facts=40)
    original_content = summary_message["content"]
    cumulative_tokens = TokenUsage()

    applied = _polish_compaction_summary(
        summary_message, FailingClient(), cumulative_tokens
    )

    assert applied is False
    # Structured, already-correct content is untouched by the failed call.
    assert summary_message["content"] == original_content
    assert cumulative_tokens.total_tokens == 0
