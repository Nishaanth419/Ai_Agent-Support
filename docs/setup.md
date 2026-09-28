# Fresh‑clone Setup Guide

This guide walks through a complete reproducible setup from a fresh repository clone:

0. **Clone the repository**
   ```bash
   git clone https://github.com/tarkalabs/bootcamp2-ai-agent-capstone.git
   cd bootcamp2-ai-agent-capstone
   ```

1. **Create a virtual environment**
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install exact dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Create the environment file**
   ```bash
   cp .env.example .env
   ```
4. **Seed the development database**
   ```bash
   python -m scripts.rebuild_db
   ```
5. **Run the test suite**
   ```bash
   python -m pytest
   ```

All commands are deterministic and work offline when `LLM_PROVIDER=fake` is set in `.env`.

## Hybrid ticket search (Qdrant)

`search_tickets` ranks past tickets with vector similarity (a local
sentence-transformers model) blended with keyword matching, backed by a
Qdrant vector index. To enable it locally:

6. **Start Qdrant**
   ```bash
   docker compose up -d
   ```
7. **Build the vector index from the seeded database**
   ```bash
   python -m scripts.reindex_qdrant
   ```
   The first run downloads the embedding model (`EMBEDDING_MODEL_NAME` in
   `.env`, default `all-MiniLM-L6-v2`) from Hugging Face Hub; it's cached
   locally after that. Re-run this any time `support.db` changes out from
   under the index (e.g. after reseeding) -- tickets resolved live via
   `send_reply` are indexed incrementally and don't need a rerun.

If Qdrant is unreachable, `search_tickets` fails soft with a clear error
message rather than raising or silently falling back to keyword-only
matching.
