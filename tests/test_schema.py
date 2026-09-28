"""Schema tests – tables, constraints, FK enforcement, reply lifecycle."""

import sqlite3

import pytest

from src.db.schema import init_db

NOW = "2026-09-07T14:00:00Z"


@pytest.fixture()
def conn():
    db = init_db()  # in-memory
    yield db
    db.close()


# ── helpers ────────────────────────────────────────────────────────────────


def _add_customer(conn, *, email="a@example.com", plan="free"):
    cur = conn.execute(
        "INSERT INTO customers (name, email, plan, signed_up_at) VALUES (?,?,?,?)",
        ("Alice", email, plan, NOW),
    )
    conn.commit()
    return cur.lastrowid


def _add_ticket(conn, customer_id, *, status="open"):
    cur = conn.execute(
        "INSERT INTO tickets (customer_id, subject, body, status, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?)",
        (customer_id, "Help", "Body text", status, NOW, NOW),
    )
    conn.commit()
    return cur.lastrowid


# ── table existence ────────────────────────────────────────────────────────


REQUIRED_TABLES = {
    "customers",
    "tickets",
    "ticket_history",
    "replies",
    "ticket_categories",
    "escalations",
}


def test_all_tables_exist(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r["name"] for r in rows}
    assert REQUIRED_TABLES <= names


# ── customers ──────────────────────────────────────────────────────────────


def test_customers_required_columns(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(customers)").fetchall()}
    assert {"id", "name", "email", "plan", "signed_up_at"} <= cols


def test_customer_insert(conn):
    cid = _add_customer(conn)
    row = conn.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()
    assert row["email"] == "a@example.com"
    assert row["plan"] == "free"


def test_customer_email_unique(conn):
    _add_customer(conn, email="dup@example.com")
    with pytest.raises(sqlite3.IntegrityError):
        _add_customer(conn, email="dup@example.com")


def test_customer_plan_check(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO customers (name, email, plan, signed_up_at) VALUES (?,?,?,?)",
            ("Bob", "b@x.com", "gold", NOW),
        )


# ── tickets ────────────────────────────────────────────────────────────────


def test_tickets_required_columns(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()}
    assert {"id", "customer_id", "subject", "body", "status", "created_at"} <= cols


def test_ticket_status_check(conn):
    cid = _add_customer(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO tickets (customer_id, subject, body, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (cid, "S", "B", "invalid_status", NOW, NOW),
        )


def test_ticket_fk_customer(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO tickets (customer_id, subject, body, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (9999, "S", "B", "open", NOW, NOW),
        )


# ── ticket_history ─────────────────────────────────────────────────────────


def test_ticket_history_required_columns(conn):
    cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(ticket_history)").fetchall()
    }
    assert {"id", "customer_id", "ticket_id", "summary", "resolved_at"} <= cols


def test_ticket_history_fk_ticket(conn):
    cid = _add_customer(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ticket_history (customer_id, ticket_id, summary)"
            " VALUES (?,?,?)",
            (cid, 9999, "summary"),
        )


def test_ticket_history_cascade_delete(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid)
    conn.execute(
        "INSERT INTO ticket_history (customer_id, ticket_id, summary) VALUES (?,?,?)",
        (cid, tid, "First summary"),
    )
    conn.commit()
    conn.execute("DELETE FROM tickets WHERE id=?", (tid,))
    conn.commit()
    rows = conn.execute(
        "SELECT * FROM ticket_history WHERE ticket_id=?", (tid,)
    ).fetchall()
    assert rows == []


# ── replies ────────────────────────────────────────────────────────────────


def test_replies_required_columns(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(replies)").fetchall()}
    assert {"id", "ticket_id", "body", "status", "created_at", "approved_at"} <= cols


def test_reply_proposed_default(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid)
    conn.execute(
        "INSERT INTO replies (ticket_id, body, created_at) VALUES (?,?,?)",
        (tid, "Draft reply", NOW),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM replies WHERE ticket_id=?", (tid,)).fetchone()
    assert row["status"] == "proposed"
    assert row["approved_at"] is None


def test_reply_approved_sets_timestamp(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid)
    conn.execute(
        "INSERT INTO replies (ticket_id, body, status, created_at, approved_at)"
        " VALUES (?,?,?,?,?)",
        (tid, "Final reply", "approved", NOW, NOW),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM replies WHERE ticket_id=?", (tid,)).fetchone()
    assert row["status"] == "approved"
    assert row["approved_at"] == NOW


def test_reply_invalid_status(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO replies (ticket_id, body, status, created_at) VALUES (?,?,?,?)",
            (tid, "x", "sent", NOW),
        )


def test_reply_fk_ticket(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO replies (ticket_id, body, created_at) VALUES (?,?,?)",
            (9999, "orphan", NOW),
        )


# ── ticket_categories ──────────────────────────────────────────────────────


def test_seed_categories_present(conn):
    rows = conn.execute("SELECT name FROM ticket_categories").fetchall()
    names = {r["name"] for r in rows}
    assert {"billing", "technical", "general"} <= names


# ── escalations ────────────────────────────────────────────────────────────


def test_escalation_insert(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid, status="escalated")
    conn.execute(
        "INSERT INTO escalations (ticket_id, reason, escalated_at) VALUES (?,?,?)",
        (tid, "angry customer", NOW),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM escalations WHERE ticket_id=?", (tid,)).fetchone()
    assert row["resolved_at"] is None
    assert row["assigned_to"] is None


def test_escalation_one_open_per_ticket(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid, status="escalated")
    conn.execute(
        "INSERT INTO escalations (ticket_id, reason, escalated_at) VALUES (?,?,?)",
        (tid, "reason 1", NOW),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO escalations (ticket_id, reason, escalated_at) VALUES (?,?,?)",
            (tid, "reason 2", NOW),
        )


def test_escalation_second_open_allowed_after_resolve(conn):
    cid = _add_customer(conn)
    tid = _add_ticket(conn, cid, status="escalated")
    conn.execute(
        "INSERT INTO escalations (ticket_id, reason, escalated_at, resolved_at)"
        " VALUES (?,?,?,?)",
        (tid, "first", NOW, NOW),
    )
    conn.commit()
    # resolved escalation gone; new open one is fine
    conn.execute(
        "INSERT INTO escalations (ticket_id, reason, escalated_at) VALUES (?,?,?)",
        (tid, "second", NOW),
    )
    conn.commit()


def test_escalation_fk_ticket(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO escalations (ticket_id, reason, escalated_at) VALUES (?,?,?)",
            (9999, "reason", NOW),
        )


# ── FK enforcement sanity check ────────────────────────────────────────────


def test_foreign_keys_pragma_is_on(conn):
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1
