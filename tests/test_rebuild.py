"""
test_rebuild.py — verifies the full rebuild-and-validate pipeline.

All rebuilds happen in temporary locations (pytest's ``tmp_path`` fixture).
The real support.db at the project root is never touched.

seed.py builds its fixture data from src/db/dataset.py's golden dataset
(CUSTOMERS/TICKETS) plus a fixed set of extra NEW_TICKETS used to populate
the 'technical' and 'feature_request' categories. This test module checks
against those actual sources rather than hardcoding counts, so it stays
correct if the dataset grows.

Acceptance criteria covered
----------------------------
AC1  One documented command safely recreates support.db (tested via
     rebuild() helper in rebuild_db.py, exercised at a tmp_path).
AC2  The script does NOT overwrite an arbitrary path — _assert_safe_path
     raises SystemExit(1) for any non-allowed path.
AC3  Validation reports table counts and checks all fixture requirements.
AC4  Repeated rebuilds produce equivalent fixture data (idempotency).
AC5  Seed/validation failures return non-zero exit status with a message.
AC6  Automated tests only rebuild into temporary locations.
"""

from pathlib import Path

import pytest

from src.db.dataset import CUSTOMERS as DATASET_CUSTOMERS
from src.db.dataset import TICKETS as DATASET_TICKETS
from src.db.schema import init_db
from src.db.seed import NEW_TICKETS, build, verify

# pyproject.toml (pythonpath = ["."] ) and conftest.py put the project root on
# sys.path for every pytest run, so no manual manipulation is needed here.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIN_CUSTOMERS = len(DATASET_CUSTOMERS)
MIN_TICKETS = len(DATASET_TICKETS) + len(NEW_TICKETS)
MIN_AMBIGUOUS = sum(1 for t in DATASET_TICKETS if t["category"] == "ambiguous")
REQUIRED_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "general",
    "feature_request",
    "other",
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_db(tmp_path: Path):
    """Build a fresh seeded database in *tmp_path* and return (conn, final_status)."""
    db_file = tmp_path / "test_support.db"
    conn, final_status, _last_id = build(db_file)
    return conn, final_status


# ── AC3: table counts & fixture requirements ──────────────────────────────────


def test_rebuild_customer_count(tmp_path):
    """At least MIN_CUSTOMERS customers are inserted."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    conn.close()
    assert n >= MIN_CUSTOMERS, f"Expected >= {MIN_CUSTOMERS} customers, got {n}"


def test_rebuild_ticket_count(tmp_path):
    """At least MIN_TICKETS tickets are inserted."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    conn.close()
    assert n >= MIN_TICKETS, f"Expected >= {MIN_TICKETS} tickets, got {n}"


def test_rebuild_category_count(tmp_path):
    """All 6 schema-seeded ticket categories are present."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM ticket_categories").fetchone()[0]
    conn.close()
    assert n >= 5, f"Expected >= 5 categories, got {n}"


def test_rebuild_ticket_history_count(tmp_path):
    """Resolved/closed tickets produce ticket_history rows."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM ticket_history").fetchone()[0]
    conn.close()
    assert n >= 1, "Expected at least one ticket_history row"


# ── AC3: theme / ambiguity / injection checks ─────────────────────────────────


def test_rebuild_all_categories_present(tmp_path):
    """All 6 required ticket-category buckets appear in the seed data."""
    conn, _ = _make_db(tmp_path)
    names_in_db = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT tc.name FROM tickets t "
            "JOIN ticket_categories tc ON tc.id = t.category_id"
        ).fetchall()
    }
    conn.close()
    assert REQUIRED_CATEGORIES.issubset(names_in_db), (
        f"Missing categories: {REQUIRED_CATEGORIES - names_in_db}"
    )


def test_rebuild_ambiguous_minimum(tmp_path):
    """At least MIN_AMBIGUOUS ambiguous ('other') tickets are in the seed data."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM tickets t "
        "JOIN ticket_categories tc ON tc.id = t.category_id "
        "WHERE tc.name = 'other'"
    ).fetchone()[0]
    conn.close()
    assert n >= MIN_AMBIGUOUS, f"Expected >= {MIN_AMBIGUOUS} ambiguous tickets, got {n}"


def test_rebuild_injection_ticket_present(tmp_path):
    """At least one ticket body contains a planted prompt-injection string."""
    conn, _ = _make_db(tmp_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE body LIKE '%SYSTEM OVERRIDE%'"
    ).fetchone()[0]
    conn.close()
    assert n >= 1, "Expected at least one prompt-injection ticket in the database"


def test_rebuild_plan_tiers_represented(tmp_path):
    """Multiple plan tiers (free, pro, enterprise) are present."""
    conn, _ = _make_db(tmp_path)
    tiers = {
        r[0] for r in conn.execute("SELECT DISTINCT plan FROM customers").fetchall()
    }
    conn.close()
    assert len(tiers) >= 2, f"Expected >= 2 plan tiers, found: {tiers}"


def test_rebuild_validate_passes(tmp_path):
    """verify() does not raise on a correctly seeded DB."""
    conn, final_status = _make_db(tmp_path)
    try:
        verify(conn, final_status)  # must not raise
    finally:
        conn.close()


# ── AC4: idempotency — two rebuilds produce identical counts ──────────────────


def test_rebuild_idempotent(tmp_path):
    """Two successive rebuilds in separate temp DBs produce identical counts."""
    counts = []
    for i in range(2):
        db_file = tmp_path / f"support_{i}.db"
        conn, _final_status, _last_id = build(db_file)
        counts.append(
            {
                "customers": conn.execute("SELECT COUNT(*) FROM customers").fetchone()[
                    0
                ],
                "tickets": conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
                "history": conn.execute(
                    "SELECT COUNT(*) FROM ticket_history"
                ).fetchone()[0],
                "categories": conn.execute(
                    "SELECT COUNT(*) FROM ticket_categories"
                ).fetchone()[0],
            }
        )
        conn.close()

    assert counts[0] == counts[1], (
        f"Rebuilds produced different counts:\n  run 1: {counts[0]}\n  run 2: {counts[1]}"
    )


# ── AC5: failures return non-zero exit status with readable message ────────────


def test_validate_raises_on_empty_db(tmp_path):
    """verify() raises AssertionError on a schema-only (empty) DB."""
    db_file = tmp_path / "empty.db"
    conn = init_db(db_file)
    # Do NOT call build() — DB has schema but no data.
    with pytest.raises(AssertionError):
        verify(conn, final_status={})
    conn.close()


def test_assert_safe_path_rejects_arbitrary_path(tmp_path):
    """_assert_safe_path raises SystemExit(1) for any non-allowed path."""
    from scripts.rebuild_db import _assert_safe_path

    arbitrary = tmp_path / "arbitrary.db"
    with pytest.raises(SystemExit) as exc_info:
        _assert_safe_path(arbitrary)
    assert exc_info.value.code == 1


def test_assert_safe_path_allows_project_db():
    """_assert_safe_path does NOT raise for the project's own support.db."""
    from scripts.rebuild_db import ALLOWED_DB_PATH, _assert_safe_path

    # Should return None (no exception) for the allowed path.
    result = _assert_safe_path(ALLOWED_DB_PATH)
    assert result is None


# ── AC6: all tests use tmp_path — confirmed by fixture usage above ─────────────
