"""
Tests for get_customer_history and get_refund_policy.

Assumption flagged: these live at app/tests/ (next to app/tools.py)
rather than under src/tests/, since src/tests/ already exists for
src/db and friends and this card's code lives entirely in app/. If the
project actually wants one single tests/ root, moving this file is a
one-line change — nothing here depends on its location beyond the
`from app...` imports, which already work from anywhere on the path
seed.py and tools.py assume (repo root on PYTHONPATH / editable install).

These tests build a fresh in-memory database per test (via
src.db.schema) instead of touching the real support.db, and pass that
connection into get_customer_history directly — see the docstring in
app/db.py for why the function accepts `conn` as an optional argument.
"""

import pytest

from app.refund_policy import REFUND_POLICY, GENERAL_POLICY
from app.tool_registry import ToolDefinition, validate_arguments
from app.tools import (
    GET_CUSTOMER_HISTORY_SCHEMA,
    SUPPORTED_BOOK_CATEGORIES,
    build_tool_registry,
    get_customer_history,
    get_refund_policy,
)
from src.db.schema import apply_schema, connect


@pytest.fixture()
def conn():
    connection = connect(":memory:")
    apply_schema(connection)
    yield connection
    connection.close()


@pytest.fixture()
def seeded_conn(conn):
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (1, 'Ada Customer', 'ada@example.com', 'pro', "
        "'2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO customers (id, name, email, plan, signed_up_at) "
        "VALUES (2, 'No History Nolan', 'nolan@example.com', 'free', "
        "'2026-01-02T00:00:00Z')"
    )

    category_id = conn.execute(
        "SELECT id FROM ticket_categories WHERE name = 'billing'"
    ).fetchone()["id"]

    ticket_rows = [
        (
            1,
            category_id,
            "Older resolved ticket",
            "body one",
            "resolved",
            "2026-02-01T00:00:00Z",
            "2026-02-02T00:00:00Z",
        ),
        (
            1,
            category_id,
            "Newer resolved ticket",
            "body two",
            "resolved",
            "2026-03-01T00:00:00Z",
            "2026-03-02T00:00:00Z",
        ),
        (
            1,
            category_id,
            "Still-open ticket",
            "body three",
            "open",
            "2026-04-01T00:00:00Z",
            "2026-04-01T00:00:00Z",
        ),
    ]
    conn.executemany(
        "INSERT INTO tickets "
        "(customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ticket_rows,
    )

    history_rows = [
        (1, 1, "Resolved the older issue.", "2026-02-02T00:00:00Z"),
        (1, 2, "Resolved the newer issue." + ("x" * 300), "2026-03-02T00:00:00Z"),
    ]
    conn.executemany(
        "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) "
        "VALUES (?, ?, ?, ?)",
        history_rows,
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------
# get_customer_history
# ---------------------------------------------------------------------


def test_populated_history_is_scoped_ordered_and_trimmed(seeded_conn):
    result = get_customer_history("1", conn=seeded_conn)

    assert result["customer_id"] == 1
    assert result["resolved_ticket_count"] == 2
    # newest resolved_at first: deterministic order
    assert [row["ticket_id"] for row in result["history"]] == [2, 1]
    # only resolved tickets appear, the open one (id 3) is excluded
    assert all(row["ticket_id"] != 3 for row in result["history"])
    # long summary was truncated, not returned in full
    assert len(result["history"][0]["summary"]) <= 240


def test_populated_history_accepts_integer_customer_id(seeded_conn):
    """Models frequently emit a bare JSON number for numeric-looking ids;
    the schema accepts both so a provider's own strict tool-call
    validation doesn't reject the call before it reaches this function."""
    result = get_customer_history(1, conn=seeded_conn)

    assert result["customer_id"] == 1
    assert result["resolved_ticket_count"] == 2


def test_customer_history_schema_accepts_integer_customer_id():
    tool = ToolDefinition(
        name="get_customer_history",
        description="",
        parameters=GET_CUSTOMER_HISTORY_SCHEMA,
        function=lambda customer_id: None,
    )

    result = validate_arguments(tool, {"customer_id": 1})

    assert result["ok"] is True


def test_empty_history_for_valid_customer_is_not_an_error(seeded_conn):
    result = get_customer_history("2", conn=seeded_conn)

    assert result["customer_id"] == 2
    assert result["resolved_ticket_count"] == 0
    assert result["history"] == []


def test_history_is_bounded_for_a_very_active_customer(seeded_conn):
    category_id = seeded_conn.execute(
        "SELECT id FROM ticket_categories WHERE name = 'billing'"
    ).fetchone()["id"]
    cur = seeded_conn.cursor()
    for i in range(10):
        cur.execute(
            "INSERT INTO tickets "
            "(customer_id, category_id, subject, body, status, created_at, updated_at) "
            "VALUES (2, ?, 'x', 'x', 'resolved', '2026-05-01T00:00:00Z', "
            "'2026-05-01T00:00:00Z')",
            (category_id,),
        )
        ticket_id = cur.lastrowid
        cur.execute(
            "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) "
            "VALUES (2, ?, 'x', '2026-05-01T00:00:00Z')",
            (ticket_id,),
        )
    seeded_conn.commit()

    result = get_customer_history("2", conn=seeded_conn)

    assert result["resolved_ticket_count"] <= 5


def test_unknown_customer_id_raises_readable_error(seeded_conn):
    with pytest.raises(ValueError, match="Unknown customer_id"):
        get_customer_history("999", conn=seeded_conn)


@pytest.mark.parametrize("bad_id", ["abc", "", "1.5", "-1", "0"])
def test_invalid_customer_id_format_raises_readable_error(seeded_conn, bad_id):
    with pytest.raises(ValueError, match="Invalid customer_id"):
        get_customer_history(bad_id, conn=seeded_conn)


def test_invalid_customer_id_surfaces_through_tool_registry(seeded_conn):
    registry = build_tool_registry()
    result = registry.dispatch("get_customer_history", {"customer_id": "abc"})

    assert result["ok"] is False
    assert "Invalid customer_id" in result["error"]


# ---------------------------------------------------------------------
# get_refund_policy
# ---------------------------------------------------------------------


@pytest.mark.parametrize("book_category", SUPPORTED_BOOK_CATEGORIES)
def test_every_supported_book_category_returns_its_policy(book_category):
    result = get_refund_policy(book_category)
    assert result["book_category"] == book_category
    assert result["policy"] == REFUND_POLICY[book_category]
    assert result["general_policy"] == GENERAL_POLICY
    assert result["policy"]  # non-empty


def test_book_category_lookup_is_case_and_whitespace_insensitive():
    result = get_refund_policy(" Fiction ")
    assert result["book_category"] == "fiction"


@pytest.mark.parametrize("bad_category", ["mystery", "SCI_FI", "romance"])
def test_invalid_book_category_raises_readable_error(bad_category):
    with pytest.raises(ValueError, match="Invalid book category"):
        get_refund_policy(bad_category)


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_missing_book_category_value_raises_readable_error(missing):
    with pytest.raises(ValueError, match="Missing book_category"):
        get_refund_policy(missing)


def test_missing_book_category_is_caught_by_schema_before_reaching_the_function():
    registry = build_tool_registry()
    result = registry.dispatch("get_refund_policy", {})

    assert result["ok"] is False
    assert "Missing required argument: book_category" in result["error"]


def test_invalid_book_category_enum_is_caught_by_schema_before_reaching_the_function():
    registry = build_tool_registry()
    result = registry.dispatch("get_refund_policy", {"book_category": "mystery"})

    assert result["ok"] is False
    assert "Invalid value for book_category" in result["error"]


def test_refund_policy_keys_match_supported_book_categories():
    assert set(REFUND_POLICY.keys()) == set(SUPPORTED_BOOK_CATEGORIES)
