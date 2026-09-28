"""
(Re)build the Qdrant ticket index from the current support.db.

Usage
-----
    python -m scripts.reindex_qdrant

What it does
------------
1. Connects to support.db (read-only; never writes to SQLite).
2. Drops and recreates the Qdrant collection named by Config.QDRANT_COLLECTION.
3. Embeds and upserts every resolved/closed ticket, using the current
   subject/body/latest-ticket_history-summary for each.

This is independent of src/db/seed.py's own API -- it always reads whatever
is actually in support.db's tickets/ticket_categories/ticket_history tables,
regardless of how that data was produced.

Exit codes
----------
0  -- reindex completed successfully.
1  -- support.db is missing, or the reindex failed.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from qdrant_client import QdrantClient

    from src.core.config import Config
    from src.db.schema import connect
    from src.search.embedder import SentenceTransformerEmbedder
    from src.search.indexing import reindex_all
    from src.search.vector_store import TicketVectorStore
except ImportError as exc:
    sys.stderr.write(
        f"[reindex_qdrant] ERROR: could not import project modules: {exc}\n"
        "  Make sure you are running from inside the virtual environment and\n"
        "  that the project root is on your PYTHONPATH.\n"
    )
    sys.exit(1)

DB_PATH = _PROJECT_ROOT / "support.db"


def reindex(
    db_path: Path = DB_PATH,
    qdrant_url: str = Config.QDRANT_URL,
    collection: str = Config.QDRANT_COLLECTION,
    model_name: str = Config.EMBEDDING_MODEL_NAME,
) -> int:
    if not db_path.is_file():
        sys.stderr.write(
            f"[reindex_qdrant] ERROR: {db_path} does not exist. "
            "Seed the database first.\n"
        )
        sys.exit(1)

    print(f"[reindex_qdrant] Connecting to {db_path} ...")
    conn = connect(db_path)

    print(f"[reindex_qdrant] Loading embedding model '{model_name}' ...")
    embedder = SentenceTransformerEmbedder(model_name)

    print(f"[reindex_qdrant] Connecting to Qdrant at {qdrant_url} ...")
    client = QdrantClient(url=qdrant_url)
    store = TicketVectorStore(client, collection, embedder.dimension)

    print(f"[reindex_qdrant] Rebuilding collection '{collection}' ...")
    try:
        count = reindex_all(conn, store, embedder)
    except Exception as exc:
        sys.stderr.write(f"[reindex_qdrant] ERROR: reindex failed: {exc}\n")
        sys.exit(1)
    finally:
        conn.close()

    return count


if __name__ == "__main__":
    n = reindex()
    print(f"[reindex_qdrant] Indexed {n} tickets into '{Config.QDRANT_COLLECTION}'.")
