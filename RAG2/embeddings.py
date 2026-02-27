"""
RAG2/embeddings.py

Purpose:
- Provide a single function to generate embeddings for a list of texts.
- Keeps embedding logic isolated from Chroma retrieval and LLM generation.

Inputs:
- List[str] texts

Outputs:
- List[List[float]] embeddings (one vector per input text)
"""

from __future__ import annotations

import os
from typing import List

from openai import OpenAI

from .config import OPENAI_EMBED_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using the configured embedding model.

    Returns:
        A list of embedding vectors. Each vector is a list of floats.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your .env at the project root."
        )

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)

    # OpenAI returns embeddings aligned to the input order
    return [item.embedding for item in resp.data]
