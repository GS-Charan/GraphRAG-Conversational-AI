"""One-off: copy facts from the legacy SQLite graph into Neo4j.

Run from the project root after `docker compose up -d` (or with the native
server running — same bolt URI either way):

    python scripts/migrate_sqlite_to_neo4j.py

Questions are dropped via delete_questions, matching app hygiene.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from companion.graph_neo4j import Neo4jGraphMemory
from companion.settings import CompanionSettings


def main() -> None:
    settings = CompanionSettings()
    if not settings.graph_db.exists():
        print(f"No SQLite graph at {settings.graph_db}; nothing to migrate.")
        return
    neo = Neo4jGraphMemory(settings)
    try:
        with sqlite3.connect(settings.graph_db) as conn:
            rows = conn.execute("SELECT DISTINCT user_id, text FROM facts").fetchall()
        print(f"SQLite facts: {len(rows)}")
        for user_id, text in rows:
            neo.remember(user_id, text)
        for user_id in sorted({user_id for user_id, _ in rows}):
            removed = neo.delete_questions(user_id)
            print(f"user={user_id}: migrated, questions removed: {len(removed)}")
        print("Migration complete. Verify at http://localhost:7474")
    finally:
        neo.close()


if __name__ == "__main__":
    main()
