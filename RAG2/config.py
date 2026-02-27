"""
RAG2/config.py

Purpose:
- Single source of truth for configuration used across the RAG pipeline.
- Loads .env once.
- Defines model names, Chroma persistence directory, collection name, and TOP_K.

This prevents config being duplicated across multiple modules and keeps the codebase readable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables once for the whole package
load_dotenv()

# -------------------------
# OpenAI configuration
# -------------------------
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Optional: used by some OpenAI client setups, harmless if unused
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# -------------------------
# Chroma configuration
# -------------------------
# IMPORTANT: point this to your repo's actual chroma folder:
# Example in your repo: EuroNCAP_Tool/RAG2/chroma_ncap_1536
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_ncap_1536"
CHROMA_DIR: str = os.path.abspath(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))

CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "ncap_code")

# Retrieval size
TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))

# -------------------------
# Optional toggles
# -------------------------
# You can keep these for debugging and thesis experiments
STRICT_JSON_OUTPUT: bool = os.getenv("STRICT_JSON_OUTPUT", "1").strip() != "0"
DEBUG_RAG: bool = os.getenv("DEBUG_RAG", "0").strip() == "1"
