#!/usr/bin/env python3
from __future__ import annotations

from ftfy import fix_text
import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        return fix_text(match.group(1).strip())
    return fix_text(path.stem)

def read_md_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    last_error = None
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            return fix_text(text).strip()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to read Markdown file {path}: {last_error}")

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_md_noise(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []

    noise_patterns = [
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

        stripped = re.sub(r"^(\s*)[*•]\s+", r"\1- ", stripped)
        stripped = re.sub(r"^(#{1,6})\s*", r"\1 ", stripped)

        cleaned.append(stripped)

    return normalize_whitespace("\n".join(cleaned))

def merge_paragraph_lines(text: str) -> str:
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()

def split_text_and_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Return list of (block_type, content)
    block_type in {"text", "code"}
    """
    parts: List[Tuple[str, str]] = []
    pattern = re.compile(r"```.*?```", re.DOTALL)

    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            normal_text = text[last_end:match.start()].strip()
            if normal_text:
                parts.append(("text", normal_text))

        code_block = match.group(0).strip()
        if code_block:
            parts.append(("code", code_block))

        last_end = match.end()

    if last_end < len(text):
        tail = text[last_end:].strip()
        if tail:
            parts.append(("text", tail))

    return parts

def split_into_sections(text: str, fallback_title: str) -> List[Tuple[Optional[str], str]]:
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

def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

def split_section_content(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = split_text_and_code_blocks(text)

    chunks: List[str] = []

    for block_type, content in pieces:
        if block_type == "code":
            if len(content) <= chunk_size:
                chunks.append(content)
            else:
                code_chunks = splitter.split_text(content)
                chunks.extend([x.strip() for x in code_chunks if x.strip()])
        else:
            text_chunks = splitter.split_text(content)
            chunks.extend([x.strip() for x in text_chunks if x.strip()])

    return chunks

def merge_small_chunks(
    chunks: List[str],
    min_chunk_chars: int = 180,
    max_chunk_chars: int = 1600,
) -> List[str]:
    if not chunks:
        return []

    merged: List[str] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if not merged:
            merged.append(chunk)
            continue

        is_small = len(chunk) < min_chunk_chars
        can_merge_to_prev = len(merged[-1]) + 2 + len(chunk) <= max_chunk_chars

        if is_small and can_merge_to_prev:
            if merged[-1].endswith("```") or chunk.startswith("```"):
                merged[-1] += "\n" + chunk
            else:
                merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)

    # handle first chunk if still too small by merging forward
    if len(merged) >= 2 and len(merged[0]) < min_chunk_chars:
        if len(merged[0]) + 2 + len(merged[1]) <= max_chunk_chars:
            merged[1] = merged[0] + "\n\n" + merged[1]
            merged = merged[1:]

    return merged

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
        subchunks = split_section_content(
            section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        subchunks = merge_small_chunks(
            subchunks,
            min_chunk_chars=180,
            max_chunk_chars=max(chunk_size + chunk_overlap, chunk_size),
        )

        for content in subchunks:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{order}",
                    doc_id=doc_id,
                    section_title=section_title,
                    content=fix_text(content).strip(),
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
        default=1200,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters",
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