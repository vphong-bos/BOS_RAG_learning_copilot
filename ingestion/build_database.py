#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_documents(input_dir: Path, patterns: List[str]) -> List[Dict]:
    docs: List[Dict] = []

    for pattern in patterns:
        for path in sorted(input_dir.rglob(pattern)):
            if not path.is_file():
                continue

            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue

            if not text:
                continue

            docs.append(
                {
                    "id": str(len(docs)),
                    "source": str(path),
                    "text": text,
                }
            )

    return docs


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(0, end - chunk_overlap)

    return chunks


def build_chunks(docs: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    chunked_docs: List[Dict] = []

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, chunk in enumerate(chunks):
            chunked_docs.append(
                {
                    "id": f'{doc["id"]}_{idx}',
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    return chunked_docs


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors


def save_metadata(metadata_path: Path, records: List[Dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector database from local documents.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing documents")
    parser.add_argument("--output_dir", type=str, default="vector_store", help="Folder to save FAISS index")
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap in words")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    docs = read_documents(input_dir, patterns=["*.txt", "*.md"])
    if not docs:
        raise RuntimeError("No documents found in input_dir")

    chunked_docs = build_chunks(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunked_docs:
        raise RuntimeError("No chunks created from documents")

    print(f"Loaded documents: {len(docs)}")
    print(f"Created chunks: {len(chunked_docs)}")

    model = SentenceTransformer(args.model_name)
    texts = [item["text"] for item in chunked_docs]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # Dùng cosine similarity bằng cách normalize vector rồi search với IndexFlatIP
    embeddings = normalize_embeddings(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss.index"
    metadata_path = output_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    save_metadata(metadata_path, chunked_docs)

    print(f"Saved index to: {index_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Embedding dimension: {dimension}")
    print(f"Total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()