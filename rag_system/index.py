from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
import joblib
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_system.chunking import chunk_text
from rag_system.config import RagConfig
from rag_system.ingest import SourceDocument, load_documents


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_path: str
    chunk_index: int
    text: str


class BasicRagIndex:
    def __init__(self, config: RagConfig):
        self.config = config
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(str(config.embedding_model_path))
        self.reranker = (
            CrossEncoder(str(config.reranker_model_path))
            if config.reranker_model_path is not None
            else None
        )
        self.client = chromadb.PersistentClient(path=str(config.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.tfidf_path = config.persist_dir / "tfidf.joblib"
        self.vectorizer: TfidfVectorizer | None = None
        if self.tfidf_path.exists():
            self.vectorizer = joblib.load(self.tfidf_path)

    def build(self) -> int:
        documents = load_documents(self.config.source_dir)
        chunks = self._make_chunks(documents)
        if not chunks:
            return 0

        self._fit_tfidf([chunk.text for chunk in chunks])
        self._upsert_chunks(chunks)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5, candidate_k: int = 20) -> list[dict[str, Any]]:
        query_embedding = self.embedder.encode([query_text], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=max(top_k, candidate_k),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        lexical_scores = self._lexical_scores(query_text, documents)
        for i, text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = float(distances[i]) if i < len(distances) else 1.0
            candidate = {
                "text": text,
                "metadata": meta,
                "vector_similarity": 1.0 - distance,
                "tfidf_score": lexical_scores[i] if i < len(lexical_scores) else 0.0,
            }
            candidates.append(candidate)

        if self.reranker is not None and candidates:
            pairs = [(query_text, item["text"]) for item in candidates]
            rerank_scores = self.reranker.predict(pairs).tolist()
            for item, score in zip(candidates, rerank_scores):
                item["rerank_score"] = float(score)
                item["score"] = self._blend_scores(item)
            candidates.sort(key=lambda item: item["score"], reverse=True)
        else:
            for item in candidates:
                item["score"] = self._blend_scores(item)
            candidates.sort(key=lambda item: item["score"], reverse=True)

        return candidates[:top_k]

    def _blend_scores(self, item: dict[str, Any]) -> float:
        vector_score = float(item.get("vector_similarity", 0.0))
        tfidf_score = float(item.get("tfidf_score", 0.0))
        rerank_score = float(item.get("rerank_score", 0.0))
        if "rerank_score" in item:
            return (0.35 * vector_score) + (0.15 * tfidf_score) + (0.5 * rerank_score)
        return (0.7 * vector_score) + (0.3 * tfidf_score)

    def _make_chunks(self, documents: list[SourceDocument]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for document in documents:
            parts = chunk_text(
                document.text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            for index, part in enumerate(parts):
                chunk_id = f"{document.path.as_posix()}::{index}"
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_path=document.path.as_posix(),
                        chunk_index=index,
                        text=part,
                    )
                )
        return records

    def _fit_tfidf(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features)
        self.vectorizer.fit(texts)
        joblib.dump(self.vectorizer, self.tfidf_path)

    def _lexical_scores(self, query_text: str, documents: list[str]) -> list[float]:
        if self.vectorizer is None or not documents:
            return [0.0 for _ in documents]

        matrix = self.vectorizer.transform(documents)
        query_vector = self.vectorizer.transform([query_text])
        scores = (matrix @ query_vector.T).toarray().ravel()
        return [float(score) for score in scores]

    def _upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.encode(
                [item.text for item in batch],
                normalize_embeddings=True,
            ).tolist()
            self.collection.upsert(
                ids=[item.chunk_id for item in batch],
                documents=[item.text for item in batch],
                metadatas=[
                    {
                        "chunk_id": item.chunk_id,
                        "source_path": item.source_path,
                        "chunk_index": item.chunk_index,
                    }
                    for item in batch
                ],
                embeddings=embeddings,
            )
