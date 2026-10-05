from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.config import RagConfig
from rag_system.index import BasicRagIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basic local RAG system")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--source-dir", type=Path, default=Path("data"))
    shared.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    shared.add_argument("--embedding-model", type=Path, required=True)
    shared.add_argument("--reranker-model", type=Path, default=None)
    shared.add_argument("--collection-name", type=str, default="rag_chunks")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", parents=[shared], help="Ingest files and build the index")

    query_parser = subparsers.add_parser("query", parents=[shared], help="Search the index")
    query_parser.add_argument("text", type=str)
    query_parser.add_argument("--top-k", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = RagConfig(
        source_dir=args.source_dir,
        persist_dir=args.persist_dir,
        embedding_model_path=args.embedding_model,
        reranker_model_path=args.reranker_model,
        collection_name=args.collection_name,
    )

    index = BasicRagIndex(config)

    if args.command == "build":
        count = index.build()
        print(f"indexed {count} chunks")
        return

    if args.command == "query":
        results = index.query(args.text, top_k=args.top_k)
        for rank, item in enumerate(results, start=1):
            meta = item["metadata"]
            print(f"[{rank}] score={item['score']:.4f} source={meta.get('source_path')} chunk={meta.get('chunk_index')}")
            print(item["text"][:500])
            print()


if __name__ == "__main__":
    main()
