#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_no} in {path}: {exc}"
                ) from exc

    return rows


def to_documents(rows: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []

    for row in rows:
        content = (row.get("content") or "").strip()
        if not content:
            continue

        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "section_title": row.get("section_title"),
                "order": row.get("order"),
            }
        )

        if row.get("token_count") is not None:
            metadata["token_count"] = row["token_count"]

        if row.get("embedding_id") is not None:
            metadata["embedding_id"] = row["embedding_id"]

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def save_metadata_backup(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from preprocessed chunks.jsonl."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="processed/chunks.jsonl",
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="vector_store",
        help="Directory to persist ChromaDB",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="confluence_docs",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing persist_directory before rebuilding",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    persist_directory = Path(args.persist_directory)

    rows = load_chunks_jsonl(input_file)
    if not rows:
        raise RuntimeError(f"No chunk rows found in {input_file}")

    documents = to_documents(rows)
    if not documents:
        raise RuntimeError("No valid documents to index")

    print(f"Loaded chunk rows: {len(rows)}")
    print(f"Built LangChain documents: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=args.model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    if args.reset and persist_directory.exists():
        print(f"Removing existing directory: {persist_directory}")
        shutil.rmtree(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    print(f"Embedding model: {args.model_name}")
    print(f"Collection name: {args.collection_name}")
    print("Building ChromaDB...")

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=args.collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # force initialization complete
    _ = db._collection.count()

    metadata_backup_path = persist_directory / "chunks.metadata.json"
    save_metadata_backup(metadata_backup_path, rows)

    print(f"Saved ChromaDB to: {persist_directory}")
    print(f"Saved metadata backup to: {metadata_backup_path}")
    print(f"Indexed documents: {len(documents)}")
    print(f"Collection count: {db._collection.count()}")


if __name__ == "__main__":
    main()