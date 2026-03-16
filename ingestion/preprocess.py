#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_title: Optional[str]
    content: str
    order: int
    token_count: Optional[int] = None
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_doc_id_from_filename(path: Path) -> str:
    match = re.search(r"_(\d+)\.md$", path.name)
    if match:
        return match.group(1)
    return path.stem


def extract_title_from_filename(path: Path) -> str:
    match = re.search(r"^(.*)_(\d+)\.md$", path.name)
    if match:
        return match.group(1).strip()
    return path.stem


def read_md_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to read Markdown file {path}: {exc}") from exc


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_md_noise(text: str) -> str:
    """
    Remove common Markdown/conversion noise from Confluence-exported content.
    """
    lines = text.splitlines()
    cleaned: List[str] = []

    noise_patterns = [
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        r"^\s*Created with Confluence\s*$",
        r"^\s*Atlassian\s+Confluence\s*$",
        r"^\s*Powered by Atlassian\s*$",
        r"^\s*Confidential\s*$",
    ]

    for line in lines:
        stripped = line.rstrip()

        if not stripped.strip():
            cleaned.append("")
            continue

        if any(re.match(pat, stripped.strip(), flags=re.IGNORECASE) for pat in noise_patterns):
            continue

        # normalize bullets
        stripped = re.sub(r"^(\s*)[*•]\s+", r"\1- ", stripped)

        # normalize excessive heading spacing: ##    Title -> ## Title
        stripped = re.sub(r"^(#{1,6})\s*", r"\1 ", stripped)

        cleaned.append(stripped)

    text = "\n".join(cleaned)
    text = normalize_whitespace(text)
    return text


def merge_paragraph_lines(text: str) -> str:
    """
    Merge wrapped lines inside paragraphs, but preserve:
    - markdown headings
    - bullets
    - numbered lists
    - code fences
    """
    lines = text.splitlines()
    result: List[str] = []
    paragraph_buffer: List[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return

        merged = paragraph_buffer[0]
        for line in paragraph_buffer[1:]:
            merged += " " + line.strip()
        result.append(merged)
        paragraph_buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            flush_paragraph()
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            result.append("")
            continue

        is_heading = bool(re.match(r"^#{1,6}\s+", stripped))
        is_bullet = bool(re.match(r"^[-*]\s+", stripped))
        is_numbered = bool(re.match(r"^\d+[.)]\s+", stripped))
        is_table = "|" in stripped
        is_blockquote = stripped.startswith(">")
        is_horizontal_rule = bool(re.match(r"^[-*_]{3,}$", stripped))

        if is_heading or is_bullet or is_numbered or is_table or is_blockquote or is_horizontal_rule:
            flush_paragraph()
            result.append(line)
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()

    merged_text = "\n".join(result)
    merged_text = re.sub(r"\n{3,}", "\n\n", merged_text)
    return merged_text.strip()


def split_into_sections(text: str, fallback_title: str) -> List[Tuple[Optional[str], str]]:
    """
    Split markdown text by headings (#, ##, ###, ...).
    Returns list of (section_title, section_text).
    """
    lines = text.splitlines()
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_title: Optional[str] = fallback_title
    current_body: List[str] = []

    for line in lines:
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)

        if heading_match:
            if current_body:
                body = "\n".join(current_body).strip()
                if body:
                    sections.append((current_title, current_body))
            current_title = heading_match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_title, current_body))

    final_sections: List[Tuple[Optional[str], str]] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            final_sections.append((title, body))

    if not final_sections and text.strip():
        final_sections.append((fallback_title, text.strip()))

    return final_sections


def chunk_text(text: str, chunk_size: int = 350, chunk_overlap: int = 60) -> List[str]:
    """
    Word-based chunking.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - chunk_overlap)

    return chunks


def preprocess_document(md_path: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    raw_text = read_md_text(md_path)
    cleaned = remove_md_noise(raw_text)
    cleaned = merge_paragraph_lines(cleaned)

    doc_id = extract_doc_id_from_filename(md_path)
    title = extract_title_from_filename(md_path)

    sections = split_into_sections(cleaned, fallback_title=title)

    chunks: List[Chunk] = []
    order = 0

    for section_title, section_text in sections:
        subchunks = chunk_text(
            section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for content in subchunks:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{order}",
                    doc_id=doc_id,
                    section_title=section_title,
                    content=content,
                    order=order,
                    token_count=None,
                    embedding_id=None,
                    metadata={
                        "title": title,
                        "source_file": str(md_path),
                    },
                )
            )
            order += 1

    return chunks


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Markdown files into Chunk JSONL for embeddings / FAISS."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="retrieved_docs",
        help="Directory containing exported Markdown files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Directory to save processed JSONL files",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=350,
        help="Chunk size in words",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=60,
        help="Chunk overlap in words",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No Markdown files found in: {input_dir}")

    all_chunks: List[Chunk] = []
    failures: List[Tuple[str, str]] = []

    print(f"Found Markdown files: {len(md_files)}")

    for idx, md_path in enumerate(md_files, start=1):
        print(f"[{idx}/{len(md_files)}] Processing: {md_path.name}")
        try:
            chunks = preprocess_document(
                md_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            all_chunks.extend(chunks)

            doc_id = extract_doc_id_from_filename(md_path)
            print(f"  doc_id={doc_id} chunks={len(chunks)}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures.append((md_path.name, str(exc)))

    chunks_path = output_dir / "chunks.jsonl"
    write_jsonl(chunks_path, (asdict(x) for x in all_chunks))

    print()
    print(f"Chunks saved: {chunks_path}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed files:")
        for name, err in failures:
            print(f"- {name}: {err}")

if __name__ == "__main__":
    main()