"""Exercise ticket lookup with isolated, deterministic SQLite data."""

import pytest

from app import tools
from src.db.schema import init_db


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "support.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    conn = init_db(path)
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (1, 'Ada Lovelace', 'ada@example.com', 'pro', '2025-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (7, 1, (SELECT id FROM ticket_categories WHERE name = 'billing'), "
        "'Duplicate charge', 'I was charged twice.', 'open', "
        "'2026-09-01T10:00:00Z', '2026-09-02T11:00:00Z')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_get_ticket_returns_ticket_and_customer(database):
    result = tools.get_ticket("7")

    assert result == {
        "ticket": {
            "id": 7,
            "subject": "Duplicate charge",
            "body": "I was charged twice.",
            "category": "billing",
            "status": "open",
            "created_at": "2026-09-01T10:00:00Z",
            "updated_at": "2026-09-02T11:00:00Z",
            "book_category": None,
            "format": None,
        },
        "customer": {
            "id": 1,
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "plan": "pro",
            "signed_up_at": "2025-01-01T00:00:00Z",
        },
        "message": "Ticket retrieved.",
    }


def test_get_ticket_unknown_id_is_recoverable(database):
    result = tools.get_ticket("999")

    assert result["ticket"] is None
    assert result["customer"] is None
    assert "No ticket found" in result["message"]


@pytest.mark.parametrize("ticket_id", [None, 7, "abc", "0", "-1"])
def test_get_ticket_invalid_id_is_recoverable(ticket_id):
    result = tools.get_ticket(ticket_id)

    assert result["ticket"] is None
    assert result["customer"] is None
    assert result["message"].startswith("Ticket error:")


def test_get_ticket_missing_argument_is_recoverable(database):
    result = tools.build_tool_registry().dispatch("get_ticket", {})

    assert result == {"ok": False, "error": "Missing required argument: ticket_id"}
