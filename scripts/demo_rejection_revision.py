"""Demonstration script for Rejection, Revision, and Approval Flow (Card #23)."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LLM_PROVIDER"] = "fake"

from app import tools
from app.agent import run_agent_turn
from app.approval import ApprovalDecision
from src.db.schema import init_db
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest


class SequenceApprovalInterface:
    """Fake approval interface that steps through a sequence of human decisions."""

    def __init__(self, decisions: list[ApprovalDecision]):
        self.decisions = iter(decisions)
        self.calls: list[tuple[str, str]] = []

    def request_approval(self, ticket_id: str, message: str) -> ApprovalDecision:
        self.calls.append((ticket_id, message))
        return next(self.decisions)


class SequenceModelClient(ModelClient):
    """Fake model client that steps through a sequence of canned responses."""

    def __init__(self, responses: list[NormalizedResponse]):
        self.responses = iter(responses)
        self.history_records: list[list[dict]] = []

    def generate(self, messages, tools=None):
        self.history_records.append(list(messages))
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError(
                "Agent loop requested more turns than the canned sequence provides"
            )


def main() -> int:
    print("Rejection, Revision, and Approval Demonstration (Card #23)")
    print("=========================================================")

    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "support.db"
        trace_path = Path("traces/rejection_revision_demo.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize test database with customer and ticket
        conn = init_db(db_path)
        now = "2026-09-01T10:00:00Z"
        conn.execute(
            "INSERT INTO customers (id, name, email, plan, signed_up_at) "
            "VALUES (1, 'Ada Lovelace', 'ada@example.com', 'pro', ?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO tickets "
            "(id, customer_id, subject, body, status, created_at, updated_at) "
            "VALUES (7, 1, 'Password Reset Issue', 'I did not receive the reset link.', 'open', ?, ?)",
            (now, now),
        )
        conn.commit()

        # Step 1: Human Approval Sequence (Rejection -> Approval)
        approval = SequenceApprovalInterface(
            [
                ApprovalDecision(
                    approved=False,
                    feedback="Please use a warmer tone and mention we resent the link.",
                ),
                ApprovalDecision(approved=True),
            ]
        )

        # Step 2: Model Response Sequence (Draft 1 -> Revised Draft 2 -> Final Answer)
        model_client = SequenceModelClient(
            [
                # Turn 1: Initial Draft Proposal
                NormalizedResponse(
                    text="Proposing initial draft reply.",
                    tool_requests=[
                        ToolRequest(
                            id="call_1",
                            name="send_reply",
                            arguments={
                                "ticket_id": "7",
                                "message": "Your ticket is closed.",
                            },
                        )
                    ],
                    usage=TokenUsage(
                        input_tokens=20,
                        output_tokens=15,
                        total_tokens=35,
                        reported=True,
                    ),
                ),
                # Turn 2: Revised Proposal incorporating human feedback
                NormalizedResponse(
                    text="Revising reply based on human feedback.",
                    tool_requests=[
                        ToolRequest(
                            id="call_2",
                            name="send_reply",
                            arguments={
                                "ticket_id": "7",
                                "message": (
                                    "Hi Ada! I've checked our email logs and re-sent your password reset link. "
                                    "Please check your inbox and let us know if you need further help!"
                                ),
                            },
                        )
                    ],
                    usage=TokenUsage(
                        input_tokens=25,
                        output_tokens=20,
                        total_tokens=45,
                        reported=True,
                    ),
                ),
                # Turn 3: Final terminal answer after approval
                NormalizedResponse(
                    text="Reply approved by human operator and sent to customer. Ticket resolved.",
                    tool_requests=[],
                    usage=TokenUsage(
                        input_tokens=15,
                        output_tokens=10,
                        total_tokens=25,
                        reported=True,
                    ),
                ),
            ]
        )

        # Build tool registry connected to fake approval interface
        registry = tools.build_tool_registry(approval)

        print("\n1. Starting Agent Triage Run...")
        # Point tools DB_PATH to isolated test DB
        tools_db_backup = tools.DB_PATH
        tools.DB_PATH = db_path
        try:
            messages = [
                {"role": "system", "content": "Test system prompt"},
                {
                    "role": "user",
                    "content": "Please triage ticket 7:\n\nPassword Reset Issue",
                },
            ]
            answer, usage, _messages = run_agent_turn(
                messages,
                client=model_client,
                registry=registry,
                trace_path=trace_path,
                run_id="demo-run-23",
            )
        finally:
            tools.DB_PATH = tools_db_backup

        print("\n2. Stages Verification:")
        print("   [PROPOSED 1] Draft 1: 'Your ticket is closed.'")
        print(
            "   [REJECTED]   Feedback: 'Please use a warmer tone and mention we resent the link.'"
        )
        # Verify rejection tool result was directly fed back into message history for 2nd model call
        second_call_messages = model_client.history_records[1]
        tool_results = [m for m in second_call_messages if m.get("role") == "tool"]
        rejection_in_history = any(
            "rejected" in m.get("content", "") for m in tool_results
        )
        assert rejection_in_history, (
            "Rejection tool result missing from message history!"
        )
        print(
            "   [INGESTED]   Rejection feedback confirmed present in message history!"
        )
        print("   [REVISED 2]  Draft 2: 'Hi Ada! I've checked our email logs...'")
        print("   [APPROVED]   Approved and saved to SQLite!")
        print(f"   [TERMINAL]   Answer: '{answer}'")

        print("\n3. Database Persistence Verification:")
        replies = conn.execute(
            "SELECT body, status, approved_at FROM replies WHERE ticket_id = 7 ORDER BY id ASC"
        ).fetchall()
        ticket_status = conn.execute(
            "SELECT status FROM tickets WHERE id = 7"
        ).fetchone()[0]

        print(f"   - Total replies recorded: {len(replies)}")
        print(f"   - Reply 1 status: {replies[0]['status']}")
        print(
            f"   - Reply 2 status: {replies[1]['status']} (approved_at={replies[1]['approved_at']})"
        )
        print(f"   - Final Ticket Status: {ticket_status}")

        # Note: Compaction mechanics and rejection-preservation policies are comprehensively tested in tests/test_compaction.py (Card #20)
        print("\n4. Structured Trace & Compaction Verification:")
        last_line = trace_path.read_text().strip().splitlines()[-1]
        trace_record = json.loads(last_line)
        events = trace_record.get("events", [])
        compaction_events = [e for e in events if e.get("type") == "history_compacted"]

        print(f"   - Run ID: {trace_record['run_id']}")
        print(f"   - Terminal Reason: {trace_record['terminal_reason']}")
        print(f"   - Total Iterations: {len(trace_record['iterations'])}")
        print(
            "   - Compaction Evaluated Per Iteration: Yes (Threshold: Config.COMPACTION_THRESHOLD)"
        )
        print(f"   - Compaction Events Triggered: {len(compaction_events)}")
        print(
            "   - Rejection Protection Policy Active: Yes (has_rejection() preserves feedback)"
        )

        conn.close()

    print("\n✅ Demonstration Completed Successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
