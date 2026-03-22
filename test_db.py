from __future__ import annotations

import argparse
import json
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from utils.emb_model import build_embeddings

load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser(description="Test local Chroma vector DB with a query.")
    parser.add_argument("--db-dir", required=True, help="Directory of persisted Chroma DB")
    parser.add_argument("--collection", default="rag_chunks", help="Chroma collection name")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name used when building the DB",
    )
    parser.add_argument("--query", required=True, help="Query text for similarity search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

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

    collection = vectorstore._collection
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    print("num_docs:", len(data["documents"]))
    print("num_embeddings:", len(data["embeddings"]))
    print("embedding_dim:", len(data["embeddings"][0]))

    results = vectorstore.similarity_search_with_score(args.query, k=args.top_k)

    output = []
    for i, (doc, score) in enumerate(results, start=1):
        output.append(
            {
                "rank": i,
                "score": float(score),
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "query": args.query,
                "top_k": args.top_k,
                "results": output,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()