"""Exercise hybrid (vector + keyword) search against isolated SQLite
databases and an in-memory Qdrant store.

FakeEmbedder (src/search/embedder.py) and QdrantClient(location=":memory:")
keep every test here fully offline and deterministic -- no Docker, no
network, no model download.
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app import tools
from src.db.schema import init_db
from src.search.embedder import FakeEmbedder
from src.search.indexing import index_one_ticket, reindex_all
from src.search.vector_store import TicketVectorStore


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
        "INSERT INTO customers (id, name, email, signed_up_at) VALUES (1, ?, ?, ?)",
        ("Customer", "customer@example.com", "2026-01-01T00:00:00Z"),
    )
    conn.commit()
    yield conn
    conn.close()


def add_ticket(
    conn,
    store,
    embedder,
    ticket_id,
    subject="Subject",
    body="Body",
    summary=None,
    category="billing",
    status="resolved",
    created_at="2026-08-01T00:00:00Z",
):
    conn.execute(
        "INSERT INTO tickets "
        "(id, customer_id, category_id, subject, body, status, created_at, updated_at) "
        "VALUES (?, 1, (SELECT id FROM ticket_categories WHERE name = ?), ?, ?, ?, ?, ?)",
        (ticket_id, category, subject, body, status, created_at, created_at),
    )
    if summary is not None:
        conn.execute(
            "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) "
            "VALUES (1, ?, ?, ?)",
            (ticket_id, summary, created_at),
        )
    conn.commit()
    # Only resolved/closed tickets are indexable; index_one_ticket no-ops
    # (via its own SQL filter) for any other status, matching search_tickets'
    # own status restriction.
    index_one_ticket(conn, store, embedder, ticket_id)


def search(conn, store, embedder, query, category=None):
    return tools.search_tickets(
        query, category, conn=conn, vector_store=store, embedder=embedder
    )


def ids(result):
    return [ticket["id"] for ticket in result["tickets"]]


def test_search_tickets_empty_query():
    result = tools.search_tickets("")

    assert result["tickets"] == []
    assert result["message"].startswith("Search error")


@pytest.mark.parametrize("query", ["  ", None, 123, [], {}])
def test_bad_queries_are_recoverable(query):
    direct_result = tools.search_tickets(query)
    assert direct_result["tickets"] == []
    assert "query must be a non-empty string" in direct_result["message"]
    result = tools.build_tool_registry().dispatch("search_tickets", {"query": query})
    assert result["ok"] is False
    assert result["error"]


@pytest.mark.parametrize("category", ["unknown", "", 1, [], {}])
def test_bad_categories_are_recoverable(category):
    direct_result = tools.search_tickets("refund", category)
    assert direct_result["tickets"] == []
    assert "category must be one of" in direct_result["message"]
    result = tools.build_tool_registry().dispatch(
        "search_tickets", {"query": "refund", "category": category}
    )
    assert result["ok"] is False
    assert result["error"]


def test_missing_database_returns_error_without_creating_file(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(tools, "DB_PATH", path)
    result = tools.search_tickets("refund")
    assert result["tickets"] == []
    assert "database is unavailable" in result["message"]
    assert not path.exists()


def test_search_tickets_sql_injection(database, store, embedder):
    add_ticket(database, store, embedder, 1, subject="ordinary")

    result = search(database, store, embedder, "'; DROP TABLE tickets; --")

    assert isinstance(result, dict)
    assert result["tickets"] == []
    assert database.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


@pytest.mark.parametrize("query", ["' OR 1=1 --", "%", "_"])
def test_injection_shaped_queries_are_literal(database, store, embedder, query):
    """No SQL/LIKE-wildcard semantics: a ticket whose subject IS the exact
    literal query string matches; a ticket without it does not. The query
    is only ever used as embedding input or a bound SQL parameter -- never
    interpolated into SQL or a LIKE pattern."""
    add_ticket(database, store, embedder, 1, subject="ordinary")
    assert ids(search(database, store, embedder, query)) == []

    add_ticket(database, store, embedder, 2, subject=query)
    assert ids(search(database, store, embedder, query)) == [2]
    assert database.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 2


def test_qdrant_unavailable_returns_recoverable_error(database, embedder):
    class BrokenStore:
        def ensure_collection(self):
            raise RuntimeError("qdrant down")

        def search(self, *args, **kwargs):
            raise RuntimeError("qdrant down")

    result = tools.search_tickets(
        "refund", conn=database, vector_store=BrokenStore(), embedder=embedder
    )

    assert result == {
        "tickets": [],
        "message": (
            "Search error: hybrid search backend is unavailable. Try again shortly."
        ),
    }


def test_subject_beats_summary_beats_body(database, store, embedder):
    add_ticket(
        database, store, embedder, 1, subject="Subject", body="refund", summary=None
    )
    add_ticket(
        database, store, embedder, 2, subject="Subject", body="Body", summary="refund"
    )
    add_ticket(
        database, store, embedder, 3, subject="refund", body="Body", summary=None
    )

    assert ids(search(database, store, embedder, "refund")) == [3, 2, 1]


def test_non_resolved_or_closed_tickets_never_appear(database, store, embedder):
    add_ticket(database, store, embedder, 1, subject="refund", status="open")
    add_ticket(database, store, embedder, 2, subject="refund", status="pending")
    add_ticket(database, store, embedder, 3, subject="refund", status="closed")

    assert ids(search(database, store, embedder, "refund")) == [3]


def test_equal_relevance_ties_to_more_recent_created_at(database, store, embedder):
    add_ticket(
        database,
        store,
        embedder,
        1,
        subject="refund",
        created_at="2026-08-01T00:00:00Z",
    )
    add_ticket(
        database,
        store,
        embedder,
        2,
        subject="refund",
        created_at="2026-09-01T00:00:00Z",
    )

    assert ids(search(database, store, embedder, "refund")) == [2, 1]


def test_body_only_matches_must_meet_relevance_threshold(database, store, embedder):
    add_ticket(database, store, embedder, 1, body="refund")
    add_ticket(database, store, embedder, 2, body="refund duplicate charge")

    result = search(database, store, embedder, "refund duplicate charge")

    assert ids(result) == [2]


def test_category_filter_excludes_strong_match_in_wrong_category(
    database, store, embedder
):
    """Proves the Qdrant-side payload filter is wired up, not just a
    post-hoc Python filter: a ticket that would otherwise rank highly is
    excluded purely because its category doesn't match."""
    add_ticket(database, store, embedder, 1, subject="refund", category="billing")

    assert ids(search(database, store, embedder, "refund", category="technical")) == []
    assert ids(search(database, store, embedder, "refund", category="billing")) == [1]


def test_category_filter_via_registry(database, store, embedder, monkeypatch):
    monkeypatch.setattr(tools, "_vector_store_singleton", store)
    monkeypatch.setattr(tools, "_embedder_singleton", embedder)
    add_ticket(database, store, embedder, 1, subject="refund", category="technical")
    add_ticket(database, store, embedder, 2, subject="refund", category="billing")

    result = tools.build_tool_registry().dispatch(
        "search_tickets", {"query": "refund", "category": "billing"}
    )

    assert result["ok"] is True
    assert ids(result["result"]) == [2]


def test_result_limit_and_stable_ties(database, store, embedder):
    for ticket_id in range(8, 0, -1):
        add_ticket(database, store, embedder, ticket_id, subject="refund")

    first = search(database, store, embedder, "refund")
    assert ids(first) == [1, 2, 3, 4, 5]
    assert search(database, store, embedder, "refund") == first


def test_truncation_retains_metadata_and_searches_full_text(database, store, embedder):
    add_ticket(
        database,
        store,
        embedder,
        1,
        subject="S" * 250,
        body="B" * 600 + " needle target phrase",
        summary="R" * 400,
    )
    ticket = search(database, store, embedder, "needle target phrase")["tickets"][0]
    for field, limit in (("subject", 200), ("body", 500), ("summary", 300)):
        assert len(ticket[field]) == limit
        assert ticket[field].endswith("…")
    assert ticket["id"] == ticket["customer_id"] == 1
    assert ticket["category"] == "billing"
    assert ticket["status"] == "resolved"
    assert ticket["created_at"] == "2026-08-01T00:00:00Z"
    assert len(database.execute("SELECT body FROM tickets").fetchone()[0]) > 500


def test_latest_summary_without_duplicate_tickets(database, store, embedder):
    add_ticket(database, store, embedder, 1, summary="old phrase")
    for summary, resolved_at in [("latest phrase", "2026-09-01"), ("pending", None)]:
        database.execute(
            "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) "
            "VALUES (1, 1, ?, ?)",
            (summary, resolved_at),
        )
    database.commit()
    # Re-index after mutating history directly, so the vector index reflects
    # the same latest-summary text search_tickets will read back from SQL.
    index_one_ticket(database, store, embedder, 1)

    assert ids(search(database, store, embedder, "phrase")) == [1]
    assert (
        search(database, store, embedder, "phrase")["tickets"][0]["summary"]
        == "latest phrase"
    )
    assert search(database, store, embedder, "old")["tickets"] == []


def test_no_matches_and_null_summary(database, store, embedder):
    add_ticket(database, store, embedder, 1)

    result = search(database, store, embedder, "absent")
    assert result["tickets"] == []
    assert result["message"] == (
        "No matching past tickets. Try using different or broader keywords."
    )
    assert search(database, store, embedder, "Subject")["tickets"][0]["summary"] is None


def test_reindex_all_from_sqlite(database, store, embedder):
    add_ticket(
        database, store, embedder, 1, subject="Xylophone concern", status="resolved"
    )
    add_ticket(database, store, embedder, 2, subject="Yodeling issue", status="closed")
    add_ticket(database, store, embedder, 3, subject="Zeppelin request", status="open")

    count = reindex_all(database, store, embedder)

    assert count == 2
    assert ids(search(database, store, embedder, "xylophone concern")) == [1]
    assert ids(search(database, store, embedder, "zeppelin request")) == []


def test_representative_seed_search(tmp_path, monkeypatch):
    # src/db/seed.py does `import dataset as src`, i.e. it expects src/db
    # itself on sys.path (run as `cd src/db && python seed.py`), not to be
    # imported as the src.db.seed package member. Importing it correctly
    # here -- without editing seed.py -- keeps this out of scope of the
    # separate seed.py/scripts.rebuild_db.py API mismatch.
    src_db_dir = str(Path(__file__).resolve().parents[1] / "src" / "db")
    monkeypatch.syspath_prepend(src_db_dir)
    import seed

    db_path = tmp_path / "seeded.db"
    conn, _final_status, _last_id = seed.build(db_path=str(db_path))
    monkeypatch.setattr(tools, "DB_PATH", db_path)

    embedder = FakeEmbedder(dimension=64)
    store = TicketVectorStore(
        QdrantClient(location=":memory:"), "seeded", embedder.dimension
    )
    reindex_all(conn, store, embedder)

    result = tools.search_tickets(
        "promo code", "billing", conn=conn, vector_store=store, embedder=embedder
    )
    assert ids(result) == [93]
    assert "discount" in result["tickets"][0]["summary"]
