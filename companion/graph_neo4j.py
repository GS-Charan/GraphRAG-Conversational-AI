"""Neo4j-backed relationship graph for GraphRAG memory expansion.

Same interface as the SQLite GraphMemory (remember/search/delete_questions),
but stored as a real property graph:

    (:Entity {user_id, name})-[:MENTIONS]->(:Fact {fact_id, user_id, text})
    (:Entity)-[:RELATES {relation, weight}]->(:Entity)

Requires a running Neo4j server (bolt). See README for setup.
"""
from __future__ import annotations

import uuid
from typing import Any

from neo4j import Driver, GraphDatabase

from companion.graph_memory import (
    COMMON_WORDS,
    FACT_PATTERN,
    PROPER_NOUN_PATTERN,
    QUESTION_LIKE,
    WORD_PATTERN,
)
from companion.settings import CompanionSettings


def _extract_entities(text: str) -> list[str]:
    entities = {item.lower() for item in PROPER_NOUN_PATTERN.findall(text)}
    return sorted(entity for entity in entities if entity not in COMMON_WORDS)


def _extract_graph_data(text: str) -> tuple[set[str], list[dict[str, str]]]:
    entities = set(_extract_entities(text))
    relations: list[dict[str, str]] = []
    for match in FACT_PATTERN.finditer(text):
        relation = match.group(1).lower()
        target = " ".join(match.group(2).lower().split()[:8]).strip(" ,;:")
        if target:
            entities.update({"user", target})
            relations.append({"subject": "user", "relation": relation, "target": target})
    return entities, relations


class Neo4jGraphMemory:
    """Persistent entity/relationship graph in Neo4j."""

    def __init__(self, settings: CompanionSettings):
        self.settings = settings
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        try:
            self.driver.verify_connectivity()
        except Exception as error:
            raise RuntimeError(
                f"Cannot reach Neo4j at {settings.neo4j_uri}: {error}. "
                "Start it with: C:\\Games\\neo4j\\bin\\neo4j.bat console"
            ) from error
        with self.driver.session(database=settings.neo4j_database) as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.user_id, e.name) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE"
            )

    def close(self) -> None:
        self.driver.close()

    def remember(self, user_id: str, text: str) -> None:
        entities, relations = _extract_graph_data(text)
        if not entities:
            return
        fact_id = uuid.uuid4().hex
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.execute_write(
                self._tx_remember, user_id, fact_id, text, sorted(entities), relations
            )

    @staticmethod
    def _tx_remember(
        tx: Any,
        user_id: str,
        fact_id: str,
        text: str,
        entities: list[str],
        relations: list[dict[str, str]],
    ) -> None:
        tx.run(
            """
            CREATE (f:Fact {fact_id: $fact_id, user_id: $user_id, text: $text})
            WITH f
            UNWIND $entities AS name
            MERGE (e:Entity {user_id: $user_id, name: name})
            MERGE (e)-[:MENTIONS]->(f)
            """,
            user_id=user_id,
            fact_id=fact_id,
            text=text,
            entities=entities,
        )
        if relations:
            tx.run(
                """
                UNWIND $relations AS rel
                MERGE (a:Entity {user_id: $user_id, name: rel.subject})
                MERGE (b:Entity {user_id: $user_id, name: rel.target})
                MERGE (a)-[r:RELATES {relation: rel.relation}]->(b)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """,
                user_id=user_id,
                relations=relations,
            )

    def search(self, user_id: str, query: str, limit: int = 4) -> list[str]:
        query_terms = set(_extract_entities(query))
        query_terms.update(
            term.lower() for term in WORD_PATTERN.findall(query) if term.lower() not in COMMON_WORDS
        )
        if not query_terms:
            return []
        with self.driver.session(database=self.settings.neo4j_database) as session:
            records = session.run(
                """
                MATCH (m:Entity)
                WHERE m.user_id = $user_id
                  AND any(term IN $terms WHERE toLower(m.name) CONTAINS term)
                WITH collect(DISTINCT m) AS matched
                UNWIND matched AS m
                OPTIONAL MATCH (m)-[:RELATES]-(nb:Entity)
                WITH matched, collect(DISTINCT nb) AS neighbors
                WITH [n IN matched + neighbors WHERE n IS NOT NULL] AS scope
                UNWIND scope AS node
                MATCH (node)-[:MENTIONS]->(f:Fact)
                WHERE f.user_id = $user_id
                RETURN f.text AS text, count(DISTINCT node) AS matches
                ORDER BY matches DESC
                LIMIT $limit
                """,
                user_id=user_id,
                terms=sorted(query_terms),
                limit=limit,
            )
            return [str(record["text"]) for record in records]

    def delete_questions(self, user_id: str) -> list[str]:
        """Delete stored questions for a user; questions are conversation, not facts."""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            rows = session.run(
                "MATCH (f:Fact {user_id: $user_id}) RETURN f.fact_id AS id, f.text AS text",
                user_id=user_id,
            ).data()
            ids = [
                row["id"]
                for row in rows
                if str(row["text"]).strip().endswith("?")
                or QUESTION_LIKE.match(str(row["text"]).strip())
            ]
            removed = [str(row["text"])[:80] for row in rows if row["id"] in set(ids)]
            if ids:
                session.run(
                    "MATCH (f:Fact) WHERE f.fact_id IN $ids DETACH DELETE f",
                    ids=ids,
                )
                session.run(
                    "MATCH (e:Entity {user_id: $user_id}) WHERE NOT (e)--() DELETE e",
                    user_id=user_id,
                )
        return removed
