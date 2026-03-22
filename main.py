from __future__ import annotations

import argparse
import json

import os

from dotenv import load_dotenv

from langchain_chroma import Chroma

from utils.emb_model import build_embeddings
from utils.data_extract import to_documents, load_records

load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist local Chroma vector DB from JSONL chunks.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--db-dir", required=True, help="Directory to store local vector DB")
    parser.add_argument("--collection", default="rag_chunks", help="Chroma collection name")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Public Hugging Face embedding model name",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    docs = to_documents(records)

    if not docs:
        raise ValueError("No valid documents created from input data.")

    try:
        token = os.getenv("HF_TOKEN")
    except:
        raise ValueError("Invalid HF_TOKEN!")

    embeddings = build_embeddings(args.embedding_model, HF_TOKEN=token)

    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.db_dir,
    )

    from tqdm import tqdm
    for doc in tqdm(docs):
        vectorstore.add_documents([doc])

    print(json.dumps({
        "status": "ok",
        "indexed_documents": len(docs),
        "db_dir": args.db_dir,
        "collection": args.collection,
        "embedding_model": args.embedding_model,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    