"""
Builds the Qdrant ticket index from SQLite.

Reads directly from the tickets/ticket_categories/ticket_history tables --
never from src/db/dataset.py -- so this stays decoupled from the seed.py/
dataset.py rewrite. Shared by the batch reindex script (scripts/reindex_
qdrant.py) and the incremental upsert hook in app/tools.py's send_reply,
so both paths agree on exactly what text and payload a ticket gets.
"""

import sqlite3

from src.search.embedder import Embedder
from src.search.vector_store import TicketVectorStore

# Indexed statuses match search_tickets's own filter: only resolved/closed
# tickets are ever surfaced as past precedent.
INDEXABLE_STATUSES = ("resolved", "closed")

# Defensive cap on the text handed to the embedder, applied before any
# model-specific token truncation. Subject and summary come first so that,
# if anything gets cut, it's the tail of the body -- the least essential
# part -- rather than the subject.
TICKET_DOC_MAX_CHARS = 2000

_INDEXABLE_TICKETS_SQL = """
    SELECT t.id, t.subject, t.body, t.status, c.name AS category,
           (SELECT h.summary FROM ticket_history h
            WHERE h.ticket_id = t.id AND h.resolved_at IS NOT NULL
            ORDER BY h.resolved_at DESC, h.id DESC LIMIT 1) AS summary
    FROM tickets t
    LEFT JOIN ticket_categories c ON c.id = t.category_id
    WHERE t.status IN ('resolved', 'closed')
"""

_ONE_TICKET_SQL = _INDEXABLE_TICKETS_SQL + " AND t.id = ?"


def build_document_text(subject: str, body: str, summary: str | None) -> str:
    text = f"{subject}\n\n{summary or ''}\n\n{body}"
    return text[:TICKET_DOC_MAX_CHARS]


def build_payload(status: str, category: str | None) -> dict:
    return {"status": status, "category": category}


def fetch_indexable_tickets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(_INDEXABLE_TICKETS_SQL).fetchall()


def reindex_all(
    conn: sqlite3.Connection,
    store: TicketVectorStore,
    embedder: Embedder,
    batch_size: int = 64,
) -> int:
    """Drop, recreate, and fully repopulate the collection from SQLite.

    A full drop/recreate (rather than upsert-only) ensures tickets that
    regressed out of resolved/closed, or were deleted, don't leave stale
    points behind.
    """
    store.recreate_collection()

    rows = fetch_indexable_tickets(conn)
    indexed = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            build_document_text(row["subject"], row["body"], row["summary"])
            for row in batch
        ]
        vectors = embedder.embed(texts)
        points = [
            (row["id"], vector, build_payload(row["status"], row["category"]))
            for row, vector in zip(batch, vectors)
        ]
        store.upsert_many(points)
        indexed += len(points)

    return indexed


def index_one_ticket(
    conn: sqlite3.Connection,
    store: TicketVectorStore,
    embedder: Embedder,
    ticket_id: int,
) -> None:
    """Re-read and upsert a single ticket. Used by send_reply's incremental hook."""
    row = conn.execute(_ONE_TICKET_SQL, (ticket_id,)).fetchone()
    if row is None:
        return

    store.ensure_collection()
    text = build_document_text(row["subject"], row["body"], row["summary"])
    vector = embedder.embed([text])[0]
    store.upsert_one(row["id"], vector, build_payload(row["status"], row["category"]))
