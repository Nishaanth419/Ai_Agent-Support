"""
SQLite schema for the customer-support agent.

All timestamps are stored as TEXT in ISO-8601 UTC format:
  YYYY-MM-DDTHH:MM:SSZ  (e.g. 2026-09-07T14:30:00Z)

Use `datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')` to produce values;
use `datetime.strptime(val, '%Y-%m-%dT%H:%M:%SZ')` to parse them back.

Foreign-key enforcement is NOT on by default in SQLite.
Call `configure_connection(conn)` (or use `connect()`) on every connection
before issuing any DML so that FK constraints are actually checked.
"""

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
-- ── Customers ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE,
    plan        TEXT    NOT NULL DEFAULT 'free'
                        CHECK (plan IN ('free', 'pro', 'enterprise')),
    signed_up_at TEXT   NOT NULL   -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_customers_email ON customers (email);

-- ── Ticket categories ──────────────────────────────────────────────────────
-- Separate lookup table so new categories can be added without a schema
-- migration; the FK on tickets.category_id enforces referential integrity.
CREATE TABLE IF NOT EXISTS ticket_categories (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE   -- e.g. 'billing', 'technical', 'general'
);

-- ── Tickets ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers (id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES ticket_categories (id) ON DELETE SET NULL,
    subject     TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'open'
                        CHECK (status IN (
                            'open', 'pending', 'resolved', 'closed', 'escalated'
                        )),
    created_at  TEXT    NOT NULL,  -- ISO-8601 UTC
    updated_at  TEXT    NOT NULL,  -- ISO-8601 UTC; kept in sync by application
    book_category TEXT  CHECK (book_category IS NULL OR book_category IN (
                            'technology', 'fiction', 'self-help'
                        )),
    format      TEXT    CHECK (format IS NULL OR format IN ('physical', 'ebook'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_customer  ON tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status    ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_category  ON tickets (category_id);

-- ── Ticket history ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticket_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers (id) ON DELETE CASCADE,
    ticket_id   INTEGER NOT NULL REFERENCES tickets   (id) ON DELETE CASCADE,
    summary     TEXT    NOT NULL,
    resolved_at TEXT             -- ISO-8601 UTC; NULL while ticket is open
);

CREATE INDEX IF NOT EXISTS idx_history_ticket   ON ticket_history (ticket_id);
CREATE INDEX IF NOT EXISTS idx_history_customer ON ticket_history (customer_id);

-- ── Replies ────────────────────────────────────────────────────────────────
-- A reply has two lifecycle stages:
--   proposed  – AI-drafted; awaiting human review
--   approved  – human has approved and the message has been sent
--   rejected  – human discarded the draft; a new draft may follow
--
-- The body column always holds the latest text.
-- approved_at is NULL until the reply reaches 'approved' status.
CREATE TABLE IF NOT EXISTS replies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   INTEGER NOT NULL REFERENCES tickets (id) ON DELETE CASCADE,
    body        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'proposed'
                        CHECK (status IN ('proposed', 'approved', 'rejected')),
    created_at  TEXT    NOT NULL,  -- ISO-8601 UTC; when the draft was created
    approved_at TEXT               -- ISO-8601 UTC; NULL until approved/sent
);

CREATE INDEX IF NOT EXISTS idx_replies_ticket ON replies (ticket_id);
CREATE INDEX IF NOT EXISTS idx_replies_status ON replies (status);

-- ── Escalations ────────────────────────────────────────────────────────────
-- Tracks when a ticket leaves the AI queue and enters human handling.
-- resolved_at is NULL while the escalation is still open.
CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       INTEGER NOT NULL REFERENCES tickets (id) ON DELETE CASCADE,
    reason          TEXT    NOT NULL,
    escalated_at    TEXT    NOT NULL,  -- ISO-8601 UTC
    assigned_to     TEXT,              -- agent / team name or ID
    resolved_at     TEXT               -- ISO-8601 UTC; NULL while open
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_escalations_ticket
    ON escalations (ticket_id)
    WHERE resolved_at IS NULL;  -- only one open escalation per ticket

CREATE INDEX IF NOT EXISTS idx_escalations_resolved ON escalations (resolved_at);
"""

# ---------------------------------------------------------------------------
# Seed data for ticket categories
# ---------------------------------------------------------------------------

_SEED_CATEGORIES = [
    "billing",
    "technical",
    "account",
    "general",
    "feature_request",
    "other",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_connection(conn: sqlite3.Connection) -> None:
    """Enable FK enforcement and WAL mode on an existing connection."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row


def connect(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open (or create) the database, apply pragmas, return the connection."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    configure_connection(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and seed lookup data idempotently."""
    conn.executescript(_DDL)
    for name in _SEED_CATEGORIES:
        conn.execute(
            "INSERT OR IGNORE INTO ticket_categories (name) VALUES (?)", (name,)
        )
    conn.commit()


def init_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Convenience: open, apply schema, return ready-to-use connection."""
    conn = connect(db_path)
    apply_schema(conn)
    return conn
