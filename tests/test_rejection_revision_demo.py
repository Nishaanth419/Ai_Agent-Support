"""Tests for Card 23: Rejection, Revision, and Approval Flow."""

import json

import pytest

from app import tools
from app.agent import run_agent_turn
from app.approval import ApprovalDecision
from scripts.demo_rejection_revision import (
    SequenceApprovalInterface,
    SequenceModelClient,
)
from src.db.schema import init_db
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    now = "2026-09-01T10:00:00Z"
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (1, 'Ada', 'ada@example.com', 'pro', ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, subject, body, status, created_at, updated_at) "
        "VALUES (7, 1, 'Help', 'Ticket body', 'open', ?, ?)",
        (now, now),
    )
    conn.commit()
    yield conn
    conn.close()


def test_rejection_revision_and_approval_flow(database, tmp_path):
    trace_path = tmp_path / "runs.jsonl"

    # Step 1: Rejection (turn 1) -> Approval (turn 2)
    approval = SequenceApprovalInterface(
        [
            ApprovalDecision(
                approved=False,
                feedback="Please make the tone warmer and add troubleshooting steps.",
            ),
            ApprovalDecision(approved=True),
        ]
    )

    # Step 2: Canned model responses (Draft 1 -> Draft 2 -> Final Answer)
    client = SequenceModelClient(
        [
            NormalizedResponse(
                text="Proposing 1st draft.",
                tool_requests=[
                    ToolRequest(
                        id="call-1",
                        name="send_reply",
                        arguments={"ticket_id": "7", "message": "Draft 1"},
                    )
                ],
                usage=TokenUsage(total_tokens=10, reported=True),
            ),
            NormalizedResponse(
                text="Proposing 2nd draft.",
                tool_requests=[
                    ToolRequest(
                        id="call-2",
                        name="send_reply",
                        arguments={
                            "ticket_id": "7",
                            "message": "Draft 2 - warmer tone with troubleshooting.",
                        },
                    )
                ],
                usage=TokenUsage(total_tokens=15, reported=True),
            ),
            NormalizedResponse(
                text="Reply approved and ticket resolved.",
                tool_requests=[],
                usage=TokenUsage(total_tokens=5, reported=True),
            ),
        ]
    )

    registry = tools.build_tool_registry(approval)

    messages = [
        {"role": "system", "content": "Test prompt"},
        {"role": "user", "content": "Please triage ticket 7:\n\nTicket context"},
    ]
    answer, usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        trace_path=trace_path,
        run_id="test-run-23",
    )

    # 1. Assert terminal answer and total usage
    assert answer == "Reply approved and ticket resolved."
    assert usage.total_tokens == 30

    # 2. Assert message history passed to 2nd generate call contains the rejection tool result
    second_call_messages = client.history_records[1]
    rejection_tool_msg = next(
        m for m in second_call_messages if m.get("role") == "tool"
    )
    rejection_data = json.loads(rejection_tool_msg["content"])
    tool_result = rejection_data.get("result", rejection_data)
    assert tool_result["status"] == "rejected"
    assert tool_result["human_feedback"] == (
        "Please make the tone warmer and add troubleshooting steps."
    )

    # 3. Assert approval calls recorded 2 proposals
    assert len(approval.calls) == 2
    assert approval.calls[0] == ("7", "Draft 1")
    assert approval.calls[1] == ("7", "Draft 2 - warmer tone with troubleshooting.")

    # 3. Assert database state: 1 rejected audit row, 1 approved row, ticket status resolved
    replies = database.execute(
        "SELECT body, status, approved_at FROM replies WHERE ticket_id = 7 ORDER BY id ASC"
    ).fetchall()
    ticket_status = database.execute(
        "SELECT status FROM tickets WHERE id = 7"
    ).fetchone()[0]

    assert len(replies) == 2
    # First draft was rejected
    assert replies[0]["body"] == "Draft 1"
    assert replies[0]["status"] == "rejected"
    assert replies[0]["approved_at"] is None

    # Second draft was approved and persisted once
    assert replies[1]["body"] == "Draft 2 - warmer tone with troubleshooting."
    assert replies[1]["status"] == "approved"
    assert replies[1]["approved_at"] is not None
    assert ticket_status == "resolved"

    # 4. Assert structured trace distinguishes proposed, rejected, revised, and approved stages
    record = json.loads(trace_path.read_text().strip())
    assert record["run_id"] == "test-run-23"
    assert record["terminal_reason"] == "completed"

    model_calls = [
        it for it in record["iterations"] if it.get("iteration_type") == "model_call"
    ]
    tool_executions = [
        it
        for it in record["iterations"]
        if it.get("iteration_type") == "tool_execution"
    ]

    # Model Call 1 proposed Draft 1
    assert (
        model_calls[0]["requested_tools"][0]["validated_arguments"]["message"]
        == "Draft 1"
    )
    # Tool Execution 1 recorded rejection with human feedback
    assert (
        tool_executions[0]["executed_tools"][0]["result"]["result"]["status"]
        == "rejected"
    )
    assert tool_executions[0]["executed_tools"][0]["result"]["result"][
        "human_feedback"
    ] == ("Please make the tone warmer and add troubleshooting steps.")

    # Model Call 2 proposed revised Draft 2
    assert model_calls[1]["requested_tools"][0]["validated_arguments"]["message"] == (
        "Draft 2 - warmer tone with troubleshooting."
    )
    # Tool Execution 2 recorded approval
    assert (
        tool_executions[1]["executed_tools"][0]["result"]["result"]["status"]
        == "approved"
    )
    assert tool_executions[1]["executed_tools"][0]["result"]["result"]["sent"] is True
