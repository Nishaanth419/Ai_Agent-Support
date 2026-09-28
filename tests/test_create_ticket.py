"""Tests for create_ticket: opening a correctly-attributed ticket for a
request that no existing ticket already describes."""

import pytest

from app import tools
from src.db.schema import init_db

NOW = "2026-09-01T10:00:00Z"


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    conn.execute(
        "INSERT INTO customers VALUES (1, 'Ada', 'ada@example.com', 'pro', ?)", (NOW,)
    )
    conn.commit()
    yield conn
    conn.close()


def test_create_ticket_is_registered_as_state_changing():
    tool = tools.build_tool_registry().get("create_ticket")
    assert tool.state_changing is True
    assert tool.parameters["required"] == [
        "customer_id",
        "category",
        "subject",
        "body",
    ]


def test_create_ticket_inserts_an_open_ticket(database):
    result = tools.create_ticket(
        "1",
        "account",
        "Need to update my email",
        "Customer asked support to change the email on file.",
        conn=database,
    )

    assert result["created"] is True
    assert result["status"] == "open"

    row = database.execute(
        "SELECT customer_id, subject, body, status, category_id "
        "FROM tickets WHERE id = ?",
        (result["ticket_id"],),
    ).fetchone()
    assert row["customer_id"] == 1
    assert row["subject"] == "Need to update my email"
    assert row["status"] == "open"

    category_name = database.execute(
        "SELECT name FROM ticket_categories WHERE id = ?", (row["category_id"],)
    ).fetchone()["name"]
    assert category_name == "account"


def test_create_ticket_accepts_integer_customer_id(database):
    result = tools.create_ticket(
        1,
        "account",
        "Need to update my email",
        "Customer asked support to change the email on file.",
        conn=database,
    )

    assert result["created"] is True
    assert result["customer_id"] == 1


def test_newly_created_ticket_can_then_be_escalated(database):
    created = tools.create_ticket(
        "1",
        "account",
        "Need to update my email",
        "Customer asked support to change the email on file.",
        conn=database,
    )

    escalated = tools.escalate(
        str(created["ticket_id"]),
        "Email change requires human account access.",
        conn=database,
    )

    assert escalated["escalated"] is True

    ticket = database.execute(
        "SELECT subject, status FROM tickets WHERE id = ?", (created["ticket_id"],)
    ).fetchone()
    assert ticket["subject"] == "Need to update my email"
    assert ticket["status"] == "escalated"


@pytest.mark.parametrize(
    "customer_id, category, subject, body, fragment",
    [
        ("999", "account", "Subject", "Body", "Unknown customer_id"),
        ("1", "bogus", "Subject", "Body", "Invalid category"),
        ("1", "account", "", "Body", "subject must be a non-empty string"),
        ("1", "account", "Subject", " ", "body must be a non-empty string"),
        ("abc", "account", "Subject", "Body", "Invalid customer_id"),
    ],
)
def test_invalid_arguments_return_readable_errors(
    database, customer_id, category, subject, body, fragment
):
    result = tools.create_ticket(customer_id, category, subject, body, conn=database)

    assert result["status"] == "error"
    assert result["created"] is False
    assert fragment in result["message"]


def test_registry_rejects_missing_or_invalid_fields():
    registry = tools.build_tool_registry()

    missing = registry.dispatch("create_ticket", {"customer_id": "1"})
    assert missing["error"] == "Missing required argument: category"

    bad_category = registry.dispatch(
        "create_ticket",
        {
            "customer_id": "1",
            "category": "bogus",
            "subject": "Subject",
            "body": "Body",
        },
    )
    assert bad_category["ok"] is False


def test_registry_accepts_integer_customer_id(database):
    registry = tools.build_tool_registry()

    result = registry.dispatch(
        "create_ticket",
        {
            "customer_id": 1,
            "category": "account",
            "subject": "Need to update my email",
            "body": "Customer asked support to change the email on file.",
        },
    )

    assert result["ok"] is True
    assert result["result"]["created"] is True
