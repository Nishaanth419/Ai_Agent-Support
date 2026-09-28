"""Tests for immediate, non-gated escalation handoff."""

import copy

import pytest

from app import tools
from app.agent import run_agent_turn
from src.db.schema import init_db
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest

NOW = "2026-09-01T10:00:00Z"


class SequenceClient(ModelClient):
    """Return preplanned model responses and retain their message history."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.snapshots = []

    def generate(self, messages, tools=None):
        self.snapshots.append(copy.deepcopy(messages))
        return next(self.responses)


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    conn.execute(
        "INSERT INTO customers VALUES (1, 'Ada', 'ada@example.com', 'pro', ?)", (NOW,)
    )
    for ticket_id, status in ((7, "open"), (8, "closed")):
        conn.execute(
            "INSERT INTO tickets (id, customer_id, category_id, subject, body, status, created_at, updated_at) "
            "VALUES (?, 1, NULL, 'Help', 'Body', ?, ?, ?)",
            (ticket_id, status, NOW, NOW),
        )
    conn.commit()
    yield conn
    conn.close()


def test_escalate_is_validated_state_changing_and_uses_a_fixed_queue():
    tool = tools.build_tool_registry().get("escalate")
    assert tool.state_changing is True
    assert tool.parameters["required"] == ["ticket_id", "reason"]
    assert tool.parameters["properties"]["reason"]["pattern"] == "\\S"
    assert tools.DEFAULT_HUMAN_ESCALATION_QUEUE in tools.HUMAN_ESCALATION_QUEUES


def test_registry_rejects_missing_or_blank_escalation_fields():
    registry = tools.build_tool_registry()
    missing = registry.dispatch("escalate", {"ticket_id": "7"})
    blank = registry.dispatch("escalate", {"ticket_id": "7", "reason": " "})
    assert missing["error"] == "Missing required argument: reason"
    assert blank["ok"] is False


def test_confirmed_escalation_atomically_records_handoff(database):
    result = tools.escalate("7", "Possible data exposure", conn=database)
    assert result["escalated"] is True
    ticket = database.execute(
        "SELECT status, updated_at FROM tickets WHERE id = 7"
    ).fetchone()
    escalation = database.execute(
        "SELECT reason, assigned_to, escalated_at FROM escalations WHERE ticket_id = 7"
    ).fetchone()
    assert ticket["status"] == "escalated"
    assert ticket["updated_at"] == escalation["escalated_at"]
    assert tuple(escalation[:2]) == ("Possible data exposure", "human-support")


def test_escalation_transaction_rolls_back_if_ticket_update_fails(database):
    """An update failure after the insert must leave no partial escalation row."""
    database.execute(
        "CREATE TRIGGER reject_escalated_ticket_update "
        "BEFORE UPDATE OF status ON tickets "
        "WHEN NEW.status = 'escalated' "
        "BEGIN SELECT RAISE(ABORT, 'forced update failure'); END"
    )
    database.commit()

    result = tools.escalate("7", "Possible data exposure", conn=database)

    assert result["status"] == "error"
    assert (
        database.execute("SELECT status FROM tickets WHERE id = 7").fetchone()[0]
        == "open"
    )
    escalation_count = database.execute(
        "SELECT COUNT(*) FROM escalations WHERE ticket_id = 7"
    ).fetchone()[0]
    assert escalation_count == 0


@pytest.mark.parametrize(
    "ticket_id, reason, fragment",
    [
        ("999", "Need review", "No ticket found"),
        ("8", "Need review", "already closed"),
        ("7", "", "non-empty"),
    ],
)
def test_invalid_ticket_or_reason_returns_readable_error(
    database, ticket_id, reason, fragment
):
    result = tools.escalate(ticket_id, reason, conn=database)
    assert result["status"] == "error"
    assert fragment in result["message"]


def test_duplicate_escalation_is_rejected_without_second_record(database):
    assert tools.escalate("7", "First reason", conn=database)["escalated"]
    duplicate = tools.escalate("7", "Second reason", conn=database)
    assert duplicate["status"] == "error"
    assert "open escalation" in duplicate["message"]
    assert (
        database.execute(
            "SELECT COUNT(*) FROM escalations WHERE ticket_id = 7"
        ).fetchone()[0]
        == 1
    )


def test_confirmed_escalation_ends_the_agent_run(database):
    client = SequenceClient(
        [
            NormalizedResponse(
                tool_requests=[
                    ToolRequest(
                        id="escalate-1",
                        name="escalate",
                        arguments={"ticket_id": "7", "reason": "Unsafe request"},
                    )
                ],
                usage=TokenUsage(total_tokens=10, reported=True),
            )
        ]
    )
    messages = [
        {"role": "system", "content": "Prompt"},
        {"role": "user", "content": "Please triage ticket 7:\n\nBody"},
    ]
    answer, _, _messages = run_agent_turn(messages, client, tools.build_tool_registry())
    assert answer == "Escalation created and assigned to the human queue."
    assert len(client.snapshots) == 1
