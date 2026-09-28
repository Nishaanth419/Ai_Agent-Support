"""Approval, persistence, validation, and loop tests for send_reply."""

import copy
import json

import pytest
from qdrant_client import QdrantClient

from app import tools
from app.agent import run_agent_turn
from app.approval import ApprovalDecision, TerminalApprovalInterface
from src.db.schema import init_db
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest
from src.search.embedder import FakeEmbedder
from src.search.vector_store import TicketVectorStore

NOW = "2026-09-01T10:00:00Z"


class FakeApprovalInterface:
    def __init__(self, approved, feedback="", before_decision=None):
        self.decision = ApprovalDecision(approved=approved, feedback=feedback)
        self.before_decision = before_decision
        self.calls = []

    def request_approval(self, ticket_id, message):
        self.calls.append((ticket_id, message))
        if self.before_decision:
            self.before_decision()
        return self.decision


class SequenceModelClient(ModelClient):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.message_snapshots = []

    def generate(self, messages, tools=None):
        self.message_snapshots.append(copy.deepcopy(messages))
        return next(self.responses)


@pytest.fixture
def embedder():
    return FakeEmbedder(dimension=32)


@pytest.fixture
def store(embedder):
    return TicketVectorStore(
        QdrantClient(location=":memory:"), "test", embedder.dimension
    )


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (1, 'Ada', 'ada@example.com', 'pro', ?)",
        (NOW,),
    )
    for ticket_id, status in (
        (7, "open"),
        (8, "pending"),
        (9, "closed"),
        (10, "resolved"),
    ):
        conn.execute(
            "INSERT INTO tickets "
            "(id, customer_id, subject, body, status, created_at, updated_at) "
            "VALUES (?, 1, 'Help', 'Ticket body', ?, ?, ?)",
            (ticket_id, status, NOW, NOW),
        )
    conn.commit()
    yield conn
    conn.close()


def test_send_reply_is_registered_as_state_changing_with_validated_arguments():
    tool = tools.build_tool_registry().get("send_reply")

    assert tool is not None
    assert tool.state_changing is True
    assert tool.parameters["required"] == ["ticket_id", "message"]
    assert tool.parameters["properties"]["ticket_id"]["pattern"]
    assert tool.parameters["properties"]["message"]["maxLength"] == 4000


def test_terminal_interface_displays_exact_proposal_and_requires_explicit_decision():
    answers = iter(["yes", "reject", "Please make it less formal."])
    prompts = []
    output = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    interface = TerminalApprovalInterface(input_fn=fake_input, output_fn=output.append)
    decision = interface.request_approval("7", "Hello, Ada.\nYour issue is fixed.")

    assert output[:3] == [
        "Ticket ID: 7",
        "Proposed reply:",
        "Hello, Ada.\nYour issue is fixed.",
    ]
    assert "Please enter exactly 'approve' or 'reject'." in output
    assert prompts == [
        "Decision [approve/reject]: ",
        "Decision [approve/reject]: ",
        "Feedback for the agent (optional): ",
    ]
    assert decision == ApprovalDecision(False, "Please make it less formal.")


def test_nothing_is_persisted_before_human_approval(database):
    def assert_preapproval_state():
        assert database.execute("SELECT COUNT(*) FROM replies").fetchone()[0] == 0
        ticket = database.execute(
            "SELECT status, updated_at FROM tickets WHERE id = 7"
        ).fetchone()
        assert tuple(ticket) == ("open", NOW)

    proposal = tools.send_reply("7", "Exact proposed reply", conn=database)
    assert proposal["status"] == "proposed"
    assert_preapproval_state()

    approval = FakeApprovalInterface(True, before_decision=assert_preapproval_state)
    registry = tools.build_tool_registry(approval)
    result = registry.dispatch(
        "send_reply", {"ticket_id": "7", "message": "Exact proposed reply"}
    )

    assert result["result"]["sent"] is True
    assert approval.calls == [("7", "Exact proposed reply")]


def test_approval_atomically_writes_once_and_resolves_ticket(database):
    approval = FakeApprovalInterface(True)
    registry = tools.build_tool_registry(approval)

    first = registry.dispatch(
        "send_reply", {"ticket_id": "7", "message": "Approved reply"}
    )
    second = registry.dispatch(
        "send_reply", {"ticket_id": "7", "message": "Duplicate reply"}
    )

    assert first["ok"] is True
    assert first["result"]["status"] == "approved"
    assert first["result"]["sent"] is True
    assert second["result"]["status"] == "error"
    assert "already resolved" in second["result"]["message"]
    assert approval.calls == [("7", "Approved reply")]

    replies = database.execute(
        "SELECT body, status, approved_at FROM replies WHERE ticket_id = 7"
    ).fetchall()
    ticket = database.execute(
        "SELECT status, updated_at FROM tickets WHERE id = 7"
    ).fetchone()
    assert len(replies) == 1
    assert replies[0]["body"] == "Approved reply"
    assert replies[0]["status"] == "approved"
    assert replies[0]["approved_at"] is not None
    assert ticket["status"] == "resolved"
    assert ticket["updated_at"] == replies[0]["approved_at"]


def test_rejection_returns_feedback_without_send_state_mutation(database):
    approval = FakeApprovalInterface(False, "Add the troubleshooting steps.")
    registry = tools.build_tool_registry(approval)

    result = registry.dispatch(
        "send_reply", {"ticket_id": "7", "message": "Draft reply"}
    )

    assert result == {
        "ok": True,
        "result": {
            "ticket_id": 7,
            "status": "rejected",
            "sent": False,
            "human_feedback": "Add the troubleshooting steps.",
            "message": "Reply rejected by human reviewer. Revise before retrying.",
        },
    }
    ticket = database.execute(
        "SELECT status, updated_at FROM tickets WHERE id = 7"
    ).fetchone()
    assert tuple(ticket) == ("open", NOW)
    assert (
        database.execute(
            "SELECT COUNT(*) FROM replies WHERE ticket_id = 7 AND status = 'approved'"
        ).fetchone()[0]
        == 0
    )
    rejected = database.execute(
        "SELECT body, status, approved_at FROM replies WHERE ticket_id = 7"
    ).fetchone()
    assert tuple(rejected) == ("Draft reply", "rejected", None)


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        ({"ticket_id": "7"}, "Missing required argument: message"),
        ({"ticket_id": "7", "message": ""}, "Invalid arguments"),
        ({"ticket_id": "7", "message": " "}, "Invalid arguments"),
        (
            {"ticket_id": "7", "message": "x" * (tools.MAX_REPLY_CHARS + 1)},
            "Invalid arguments",
        ),
        ({"ticket_id": 7, "message": "Reply"}, "Invalid type for ticket_id"),
        ({"ticket_id": "abc", "message": "Reply"}, "Invalid arguments"),
    ],
)
def test_registry_rejects_invalid_arguments_without_requesting_approval(
    arguments, expected_error
):
    approval = FakeApprovalInterface(True)

    result = tools.build_tool_registry(approval).dispatch("send_reply", arguments)

    assert result["ok"] is False
    assert expected_error in result["error"]
    assert approval.calls == []


@pytest.mark.parametrize(
    ("ticket_id", "message"),
    [("7", ""), ("7", " "), ("7", "x" * (tools.MAX_REPLY_CHARS + 1)), (7, "Reply")],
)
def test_direct_invalid_arguments_return_readable_errors(database, ticket_id, message):
    result = tools.send_reply(ticket_id, message, conn=database)

    assert result["status"] == "error"
    assert result["sent"] is False
    assert result["message"].startswith("Reply error:")


@pytest.mark.parametrize(
    ("ticket_id", "state"),
    [("999", "No ticket found"), ("9", "already closed"), ("10", "already resolved")],
)
def test_invalid_or_finished_ticket_is_rejected_before_approval(
    database, ticket_id, state
):
    result = tools.send_reply(ticket_id, "Reply", conn=database)

    assert result["status"] == "error"
    assert state in result["message"]
    assert result["sent"] is False


def test_pending_ticket_can_be_approved(database):
    proposal = tools.send_reply("8", "Pending-ticket reply", conn=database)
    assert proposal["status"] == "proposed"

    result = tools._commit_approved_reply(8, "Pending-ticket reply", conn=database)

    assert result["sent"] is True
    assert (
        database.execute("SELECT status FROM tickets WHERE id = 8").fetchone()[0]
        == "resolved"
    )


def test_approval_transaction_rolls_back_both_writes_on_failure(database):
    database.execute(
        "CREATE TRIGGER reject_resolution BEFORE UPDATE OF status ON tickets "
        "WHEN NEW.status = 'resolved' BEGIN "
        "SELECT RAISE(ABORT, 'resolution disabled'); END"
    )
    database.commit()

    result = tools._commit_approved_reply(7, "Approved reply", conn=database)

    assert result["status"] == "error"
    assert result["sent"] is False
    assert "atomically" in result["message"]
    assert (
        database.execute("SELECT COUNT(*) FROM replies WHERE ticket_id = 7").fetchone()[
            0
        ]
        == 0
    )
    assert (
        database.execute("SELECT status FROM tickets WHERE id = 7").fetchone()[0]
        == "open"
    )


def test_rejection_feedback_returns_to_agent_loop(database):
    client = SequenceModelClient(
        [
            NormalizedResponse(
                tool_requests=[
                    ToolRequest(
                        id="reply-1",
                        name="send_reply",
                        arguments={"ticket_id": "7", "message": "First draft"},
                    )
                ],
                usage=TokenUsage(total_tokens=10, reported=True),
            ),
            NormalizedResponse(
                text="I will revise the reply.",
                usage=TokenUsage(total_tokens=10, reported=True),
            ),
        ]
    )
    registry = tools.build_tool_registry(
        FakeApprovalInterface(False, "Use a warmer tone.")
    )

    messages = [
        {"role": "system", "content": "Test prompt"},
        {"role": "user", "content": "Please triage ticket 7:\n\nTicket body"},
    ]
    answer, _, _messages = run_agent_turn(messages, client=client, registry=registry)

    tool_message = client.message_snapshots[1][-1]
    tool_result = json.loads(tool_message["content"])
    assert answer == "I will revise the reply."
    assert tool_message["name"] == "send_reply"
    assert tool_result["result"]["sent"] is False
    assert tool_result["result"]["human_feedback"] == "Use a warmer tone."


def test_send_reply_incremental_upsert_makes_ticket_searchable(
    database, store, embedder
):
    result = tools._commit_approved_reply(
        7,
        "Approved reply",
        conn=database,
        vector_store=store,
        embedder=embedder,
    )

    assert result["sent"] is True
    found = tools.search_tickets(
        "Help", conn=database, vector_store=store, embedder=embedder
    )
    assert 7 in [ticket["id"] for ticket in found["tickets"]]


def test_send_reply_survives_qdrant_outage(database, embedder):
    class BrokenStore:
        def ensure_collection(self):
            raise RuntimeError("qdrant down")

    result = tools._commit_approved_reply(
        7,
        "Approved reply",
        conn=database,
        vector_store=BrokenStore(),
        embedder=embedder,
    )

    assert result["sent"] is True
    assert result["status"] == "approved"
    assert (
        database.execute("SELECT status FROM tickets WHERE id = 7").fetchone()[0]
        == "resolved"
    )
