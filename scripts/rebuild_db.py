"""
Rebuild and validate the development/test database in one command.

Usage
-----
    python -m scripts.rebuild_db

What it does
------------
1. Resolves the **fixed** target path:  <project-root>/support.db
   No user-supplied path is accepted — this prevents silent overwrites of
   arbitrary databases.
2. Drops the existing file (if any) and recreates it from the schema.
3. Seeds it with the deterministic fixture data from src/db/seed.py.
4. Validates table counts and all fixture requirements, printing a summary.
5. Exits 0 on success; exits 1 on any error (schema, seed, or validation)
   and prints a readable explanation to stderr.

Exit codes
----------
0  — database rebuilt and all validation checks passed.
1  — unexpected target path, seed failure, or validation failure.
"""

import sqlite3
import sys
from pathlib import Path

# ── path safety ───────────────────────────────────────────────────────────────
# This script lives at  <project-root>/scripts/rebuild_db.py.
# parents[1] is the project root regardless of cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
ALLOWED_DB_PATH = _PROJECT_ROOT / "support.db"


def _assert_safe_path(target: Path) -> None:
    """Abort with exit code 1 if *target* is not the allowed development DB."""
    if target.resolve() != ALLOWED_DB_PATH.resolve():
        sys.stderr.write(
            f"[rebuild_db] ERROR: refusing to rebuild '{target}'.\n"
            f"  Only the named development database is allowed:\n"
            f"    {ALLOWED_DB_PATH}\n"
            "  Remove any --db flag and run from inside the project.\n"
        )
        sys.exit(1)


# ── import the db package ─────────────────────────────────────────────────────
# Support both `python -m scripts.rebuild_db` (project root on sys.path via
# pyproject.toml) and `python scripts/rebuild_db.py` (add root manually).
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.db.seed import build, verify
except ImportError as exc:
    sys.stderr.write(
        f"[rebuild_db] ERROR: could not import project modules: {exc}\n"
        "  Make sure you are running from inside the virtual environment and\n"
        "  that the project root is on your PYTHONPATH.\n"
    )
    sys.exit(1)


# ── main ──────────────────────────────────────────────────────────────────────


def rebuild(db_path: Path = ALLOWED_DB_PATH) -> None:
    """Drop, recreate, seed, and validate *db_path*.

    Raises ``SystemExit(1)`` with a descriptive message on any failure.
    Only ``ALLOWED_DB_PATH`` (``support.db`` at the project root) is
    permitted; passing any other resolved path exits immediately.

    Parameters
    ----------
    db_path:
        Target SQLite file.  Must resolve to ``ALLOWED_DB_PATH``.
    """
    _assert_safe_path(db_path)

    print(f"[rebuild_db] Target: {db_path}")

    # ── schema + seed ─────────────────────────────────────────────────────────
    # build() drops any existing file at db_path itself, applies the schema,
    # and seeds it in one step.
    print("[rebuild_db] Applying schema and seeding fixture data ...")
    try:
        conn, final_status, _last_id = build(db_path)
    except sqlite3.Error as exc:
        sys.stderr.write(f"[rebuild_db] Seed error: {exc}\n")
        sys.exit(1)

    # ── validate ──────────────────────────────────────────────────────────────
    print("[rebuild_db] Validating ...")
    try:
        verify(conn, final_status)
    except AssertionError as exc:
        conn.close()
        sys.stderr.write(f"[rebuild_db] Validation failed: {exc}\n")
        sys.exit(1)
    finally:
        conn.close()

    print(f"\n[rebuild_db] Database ready at: {db_path}")


if __name__ == "__main__":
    rebuild()
