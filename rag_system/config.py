from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    source_dir: Path
    persist_dir: Path
    embedding_model_path: Path
    reranker_model_path: Path | None = None
    collection_name: str = "rag_chunks"
    chunk_size: int = 800
    chunk_overlap: int = 120
    tfidf_max_features: int = 50000
