from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "companion_data"


@dataclass(frozen=True)
class CompanionSettings:
    model_name: str = "companion-gemma"
    ollama_url: str = "http://localhost:11434"
    user_id: str = "default_user"
    memory_collection: str = "companion_memories"
    voice_url: str = "http://127.0.0.1:8080"
    voice_reference_id: str | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # Overridable without code changes; default keeps local setups working.
    neo4j_password: str = field(
        default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "CompanionGraph01")
    )
    neo4j_database: str = "neo4j"

    @property
    def embedding_model(self) -> Path:
        return MODELS_DIR / "bge-small-en-v1.5"

    @property
    def reranker_model(self) -> Path:
        return MODELS_DIR / "bge-reranker-base"

    @property
    def qdrant_dir(self) -> Path:
        return DATA_DIR / "qdrant"

    @property
    def history_db(self) -> Path:
        return DATA_DIR / "mem0_history.db"

    @property
    def graph_db(self) -> Path:
        return DATA_DIR / "memory_graph.db"
