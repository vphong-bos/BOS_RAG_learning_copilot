#!/usr/bin/env python3
"""
Build a LangChain + Chroma vector DB from JSONL chunk data and retrieve top-k chunk_ids.

Supports:
- Proper JSONL (one JSON object per line)
- "Broken" JSONL / pasted blobs where multiple JSON objects appear on one line

Example:
    python build_rag_jsonl.py index \
        --input data.jsonl \
        --persist-dir ./chroma_db \
        --collection aisw_docs

    python build_rag_jsonl.py query \
        --persist-dir ./chroma_db \
        --collection aisw_docs \
        --text "quantization roadmap and task allocation" \
        --k 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def extract_json_objects(raw_text: str) -> List[Dict[str, Any]]:
    """
    Robustly extract JSON objects from text, even if multiple objects are pasted
    together on one line.

    This parser tracks:
    - brace depth
    - string state
    - escape characters

    It returns a list of decoded JSON dicts.
    """
    objects: List[Dict[str, Any]] = []
    buf: List[str] = []
    depth = 0
    in_string = False
    escape = False

    for ch in raw_text:
        if depth == 0:
            if ch.isspace():
                continue
            if ch != "{":
                # Ignore garbage before a JSON object starts
                continue

        buf.append(ch)

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = "".join(buf).strip()
                buf = []
                if candidate:
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed to decode JSON object near: {candidate[:200]}..."
                        ) from exc

    if depth != 0:
        raise ValueError("Input appears to contain an incomplete JSON object.")

    return objects


def load_chunk_records(input_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL-like file.
    Works for normal JSONL and also pasted JSON objects separated by whitespace.
    """
    path = Path(input_path)
    raw_text = path.read_text(encoding="utf-8")
    records = extract_json_objects(raw_text)

    if not records:
        raise ValueError(f"No JSON objects found in file: {path}")

    return records


def normalize_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten/clean metadata for storage in vector DB.
    Chroma metadata should stay JSON-serializable and relatively flat.
    """
    metadata = {}

    # top-level fields
    for key in ["chunk_id", "doc_id", "section_title", "order", "token_count", "embedding_id"]:
        if key in record:
            metadata[key] = record.get(key)

    # nested metadata
    nested = record.get("metadata", {})
    if isinstance(nested, dict):
        for key, value in nested.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[f"meta_{key}"] = value
            else:
                metadata[f"meta_{key}"] = json.dumps(value, ensure_ascii=False)

    # ref_links as serialized JSON for portability
    ref_links = record.get("ref_links", [])
    metadata["ref_links"] = json.dumps(ref_links, ensure_ascii=False)

    return metadata


def records_to_documents(records: Iterable[Dict[str, Any]]) -> List[Document]:
    """
    Convert parsed chunk records to LangChain Documents.
    Uses `content` as page_content and attaches useful metadata.
    """
    docs: List[Document] = []

    for rec in records:
        content = rec.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        metadata = normalize_metadata(rec)
        doc_id = str(rec.get("chunk_id") or rec.get("doc_id") or "")

        doc = Document(
            id=doc_id if doc_id else None,
            page_content=content,
            metadata=metadata,
        )
        docs.append(doc)

    return docs


def get_embeddings(model_name: str) -> OpenAIEmbeddings:
    """
    Initialize embedding model.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    return OpenAIEmbeddings(model=model_name)


def build_vector_db(
    input_path: str | Path,
    persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
) -> Tuple[int, str]:
    """
    Parse records, convert to Documents, embed, and persist to Chroma.
    """
    records = load_chunk_records(input_path)
    docs = records_to_documents(records)

    if not docs:
        raise ValueError("No valid documents were created from the input file.")

    embeddings = get_embeddings(embedding_model)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Add documents. Chroma persists automatically when configured with persist_directory.
    vectorstore.add_documents(docs)

    return len(docs), str(persist_dir)


def load_vector_db(
    persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
) -> Chroma:
    """
    Load an existing persisted Chroma collection.
    """
    embeddings = get_embeddings(embedding_model)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def retrieve_top_k(
    persist_dir: str | Path,
    collection_name: str,
    embedding_model: str,
    query_text: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks.
    Returns rank, chunk_id, score, section_title, and content preview.
    """
    vectorstore = load_vector_db(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Chroma / LangChain returns (Document, score)
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    output: List[Dict[str, Any]] = []
    for rank, (doc, score) in enumerate(results, start=1):
        output.append(
            {
                "rank": rank,
                "chunk_id": doc.metadata.get("chunk_id") or doc.id,
                "doc_id": doc.metadata.get("doc_id"),
                "section_title": doc.metadata.get("section_title"),
                "score": float(score),
                "content_preview": doc.page_content[:300].replace("\n", " "),
                "metadata": doc.metadata,
            }
        )

    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a LangChain RAG vector DB from JSONL chunks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    p_index = subparsers.add_parser("index", help="Build / update vector DB from JSONL")
    p_index.add_argument("--input", required=True, help="Path to input .jsonl file")
    p_index.add_argument("--persist-dir", required=True, help="Directory for persisted Chroma DB")
    p_index.add_argument("--collection", default="rag_chunks", help="Chroma collection name")
    p_index.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name",
    )

    # Query command
    p_query = subparsers.add_parser("query", help="Query existing vector DB")
    p_query.add_argument("--persist-dir", required=True, help="Directory for persisted Chroma DB")
    p_query.add_argument("--collection", default="rag_chunks", help="Chroma collection name")
    p_query.add_argument("--text", required=True, help="Input query text / prompt")
    p_query.add_argument("--k", type=int, default=5, help="Top-k results")
    p_query.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name",
    )

    args = parser.parse_args()

    if args.command == "index":
        count, db_path = build_vector_db(
            input_path=args.input,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "indexed_documents": count,
                    "persist_dir": db_path,
                    "collection": args.collection,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.command == "query":
        results = retrieve_top_k(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            query_text=args.text,
            k=args.k,
        )

        print(json.dumps({"query": args.text, "top_k": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()