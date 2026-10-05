from __future__ import annotations

import re
import sqlite3
from pathlib import Path


COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "i", "in", "is",
    "it", "my", "of", "on", "or", "the", "this", "that", "to", "we", "with", "you", "your",
}
FACT_PATTERN = re.compile(
    r"\bI\s+(am|like|love|prefer|work|live|have|need|want|use|play)\s+([^.!?]+)",
    flags=re.IGNORECASE,
)
PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_-]{3,}")
QUESTION_LIKE = re.compile(
    r"^(what'?s?|whast|what|when|where|who|whom|whose|which|why|how|do|does|did|is|are|was|were|can|could|will|would|tell me|remind me|do u|do you)\b",
    flags=re.IGNORECASE,
)


class GraphMemory:
    """Small, persistent relationship graph used to expand memory retrieval."""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                UNIQUE(user_id, name)
            );
            CREATE TABLE IF NOT EXISTS edges (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                weight INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(source_id, target_id, relation),
                FOREIGN KEY(source_id) REFERENCES nodes(id),
                FOREIGN KEY(target_id) REFERENCES nodes(id)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS fact_nodes (
                fact_id INTEGER NOT NULL,
                node_id INTEGER NOT NULL,
                PRIMARY KEY(fact_id, node_id),
                FOREIGN KEY(fact_id) REFERENCES facts(id),
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            );
            """
        )
        self.connection.commit()

    def remember(self, user_id: str, text: str) -> None:
        entities, relations = self._extract_graph_data(text)
        if not entities:
            return

        cursor = self.connection.execute("INSERT INTO facts(user_id, text) VALUES (?, ?)", (user_id, text))
        fact_id = cursor.lastrowid
        node_ids = {entity: self._get_or_create_node(user_id, entity) for entity in entities}
        self.connection.executemany(
            "INSERT OR IGNORE INTO fact_nodes(fact_id, node_id) VALUES (?, ?)",
            [(fact_id, node_id) for node_id in node_ids.values()],
        )

        for subject, relation, target in relations:
            if subject not in node_ids:
                node_ids[subject] = self._get_or_create_node(user_id, subject)
            if target not in node_ids:
                node_ids[target] = self._get_or_create_node(user_id, target)
            self.connection.execute(
                """
                INSERT INTO edges(source_id, target_id, relation, weight) VALUES (?, ?, ?, 1)
                ON CONFLICT(source_id, target_id, relation) DO UPDATE SET weight = weight + 1
                """,
                (node_ids[subject], node_ids[target], relation),
            )
        self.connection.commit()

    def search(self, user_id: str, query: str, limit: int = 4) -> list[str]:
        query_terms = set(self._extract_entities(query))
        query_terms.update(term.lower() for term in WORD_PATTERN.findall(query) if term.lower() not in COMMON_WORDS)
        if not query_terms:
            return []

        node_conditions = " OR ".join("name LIKE ?" for _ in query_terms)
        rows = self.connection.execute(
            f"""
            WITH matching_nodes AS (
                SELECT id FROM nodes
                WHERE user_id = ? AND ({node_conditions})
            ), related_nodes AS (
                SELECT target_id AS id FROM edges WHERE source_id IN matching_nodes
                UNION
                SELECT source_id AS id FROM edges WHERE target_id IN matching_nodes
                UNION
                SELECT id FROM matching_nodes
            )
            SELECT facts.text, COUNT(DISTINCT fact_nodes.node_id) AS matches
            FROM facts
            JOIN fact_nodes ON fact_nodes.fact_id = facts.id
            WHERE facts.user_id = ? AND fact_nodes.node_id IN related_nodes
            GROUP BY facts.id
            ORDER BY matches DESC, facts.id DESC
            LIMIT ?
            """,
            [user_id, *[f"%{term}%" for term in sorted(query_terms)], user_id, limit],
        ).fetchall()
        return [str(row["text"]) for row in rows]

    def delete_questions(self, user_id: str) -> list[str]:
        """Delete stored questions for a user; questions are conversation, not facts."""
        removed: list[str] = []
        rows = self.connection.execute("SELECT id, text FROM facts WHERE user_id = ?", (user_id,)).fetchall()
        for row in rows:
            text = str(row["text"])
            stripped = text.strip()
            if stripped.endswith("?") or QUESTION_LIKE.match(stripped):
                removed.append(text[:80])
                self.connection.execute("DELETE FROM fact_nodes WHERE fact_id = ?", (row["id"],))
                self.connection.execute("DELETE FROM facts WHERE id = ?", (row["id"],))
        self.connection.commit()
        return removed

    def _get_or_create_node(self, user_id: str, name: str) -> int:
        self.connection.execute("INSERT OR IGNORE INTO nodes(user_id, name) VALUES (?, ?)", (user_id, name))
        row = self.connection.execute("SELECT id FROM nodes WHERE user_id = ? AND name = ?", (user_id, name)).fetchone()
        return int(row["id"])

    def _extract_graph_data(self, text: str) -> tuple[set[str], list[tuple[str, str, str]]]:
        entities = set(self._extract_entities(text))
        relations: list[tuple[str, str, str]] = []
        for match in FACT_PATTERN.finditer(text):
            relation = match.group(1).lower()
            target = " ".join(match.group(2).lower().split()[:8]).strip(" ,;:")
            if target:
                entities.update({"user", target})
                relations.append(("user", relation, target))
        return entities, relations

    def _extract_entities(self, text: str) -> list[str]:
        entities = {item.lower() for item in PROPER_NOUN_PATTERN.findall(text)}
        return sorted(entity for entity in entities if entity not in COMMON_WORDS)
