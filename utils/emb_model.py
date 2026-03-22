import os
from langchain_huggingface import HuggingFaceEmbeddings

def build_embeddings(model_name: str, HF_TOKEN) -> HuggingFaceEmbeddings:
    """
    Build local Hugging Face embeddings.

    Good public model choices:
    - sentence-transformers/all-mpnet-base-v2
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-base-en-v1.5
    """
    encode_kwargs = {"normalize_embeddings": False}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "token": HF_TOKEN
        },
        encode_kwargs=encode_kwargs,
    )