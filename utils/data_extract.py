from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

from tqdm import tqdm


def extract_json_objects(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON objects even if the file is not clean JSONL and contains
    multiple JSON objects pasted together.
    """
    objects: List[Dict[str, Any]] = []
    buf: List[str] = []
    depth = 0
    in_string = False
    escape = False
    got_first_obj = 0

    for ch in raw_text:
        if depth == 0:
            if ch.isspace():
                continue
            if ch != "{":
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
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    objects.append(obj)
                # got_first_obj += 1

    if depth != 0:
        raise ValueError("Incomplete JSON object found in input file.")

    return objects


def load_records(input_file: str) -> List[Dict[str, Any]]:
    raw_text = Path(input_file).read_text(encoding="utf-8")
    records = extract_json_objects(raw_text)
    if not records:
        raise ValueError(f"No JSON objects found in {input_file}")
    return records


def flatten_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for key in ["chunk_id", "doc_id", "section_title", "order", "token_count", "embedding_id"]:
        metadata[key] = record.get(key)

    nested = record.get("metadata", {})
    if isinstance(nested, dict):
        for key, value in nested.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[f"meta_{key}"] = value
            else:
                metadata[f"meta_{key}"] = json.dumps(value, ensure_ascii=False)

    ref_links = record.get("ref_links", [])
    metadata["ref_links"] = json.dumps(ref_links, ensure_ascii=False)

    return metadata


def to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []

    for record in records:
        content = record.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        chunk_id = str(record.get("chunk_id", ""))
        doc = Document(
            id=chunk_id if chunk_id else None,
            page_content=content,
            metadata=flatten_metadata(record),
        )
        docs.append(doc)

    return docs