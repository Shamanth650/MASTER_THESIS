"""
RAG2/chroma_store.py

Purpose:
- Own all ChromaDB logic: persistent client, collection creation, and retrieval.
- Reuse your existing on-disk Chroma DB folder (chroma.sqlite3 + UUID folder).
- Provide retrieval helpers used by the generators.

Key functions:
- retrieve_context(query_text, k=TOP_K): semantic retrieval
- retrieve_context_filtered(query_text, where=..., k=...): metadata-filtered retrieval (for templates/rules later)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .config import CHROMA_COLLECTION, CHROMA_DIR, TOP_K
from .embeddings import embed_texts


# NOTE:
# This is only used if the collection is empty.
# In your current pipeline, you already had a bootstrap snippet list.
# Keep it here so the RAG system still works even when Chroma is new/empty.
BOOTSTRAP_SNIPPETS = [
    {
        "id": "bootstrap_0",
        "text": (
            "ScenarioRunner Python scenarios should use CarlaDataProvider and py_trees. "
            "Avoid world.load_world() inside scenarios. Use existing actors and blueprints."
        ),
        "metadata": {"doc_type": "bootstrap", "family": "ALL", "version": "v1"},
    },
    {
        "id": "bootstrap_1",
        "text": (
            "OpenSCENARIO (XOSC) should define Entities, Init actions, Storyboard, "
            "and proper SpeedAction structure. Keep XML well-formed."
        ),
        "metadata": {"doc_type": "bootstrap", "family": "ALL", "version": "v1"},
    },
]


_client: Optional[chromadb.Client] = None

from pathlib import Path
import uuid


def ingest_knowledge_base():
    """
    Ingest templates and rules from RAG2/knowledge_base into Chroma.
    This is a one-time (or on-change) operation.
    """
    coll = _ensure_chroma()

    kb_root = Path(__file__).parent / "knowledge_base"

    for doc_type, subdir in [
        ("py_template", kb_root / "templates" / "python"),
        ("xosc_template", kb_root / "templates" / "xosc"),
        ("rule", kb_root / "rules"),
    ]:
        if not subdir.exists():
            continue

        for path in subdir.glob("*.txt"):
            text = path.read_text(encoding="utf-8")

            name = path.stem.lower()
            if name.startswith("aeb"):
                family = "AEB"
            elif name.startswith("vru"):
                family = "VRU"
            elif name.startswith("lss"):
                family = "LSS"
            else:
                family = "UNKNOWN"

            meta = {
                "doc_type": doc_type,
                "family": family,
                "version": "v1",
                "source": str(path),
            }

            emb = embed_texts([text])[0]

            coll.add(
                ids=[f"{doc_type}_{family}_{uuid.uuid4().hex}"],
                documents=[text],
                embeddings=[emb],
                metadatas=[meta],
            )

def _persistent_client() -> chromadb.Client:
    """
    Create (or reuse) a persistent Chroma client that stores data on disk.

    Uses CHROMA_DIR from config.py, which should point to your existing:
    .../RAG2/chroma_ncap_1536/
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _ensure_chroma():
    """
    Ensure the configured collection exists.
    If it doesn't exist, create it and bootstrap minimal docs.
    """
    client = _persistent_client()

    try:
        coll = client.get_collection(name=CHROMA_COLLECTION)
    except Exception:
        coll = client.create_collection(name=CHROMA_COLLECTION)

    # Bootstrap if empty (safe)
    try:
        count = coll.count()
    except Exception:
        count = 0

    if count == 0:
        texts = [d["text"] for d in BOOTSTRAP_SNIPPETS]
        ids = [d["id"] for d in BOOTSTRAP_SNIPPETS]
        metas = [d["metadata"] for d in BOOTSTRAP_SNIPPETS]
        embs = embed_texts(texts)
        coll.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)

    return coll


def retrieve_context(query_text: str, k: int | None = None) -> List[Dict[str, Any]]:
    """
    Semantic retrieval from Chroma (your current rag.py behavior).

    Returns a list of hits with:
    - document: retrieved text chunk
    - metadata: stored metadata for that chunk
    """
    coll = _ensure_chroma()
    k = k or TOP_K

    q_emb = embed_texts([query_text])[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    hits: List[Dict[str, Any]] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        hits.append(
            {
                "id": f"hit_{i}",
                "document": doc,
                "metadata": meta or {},
            }
        )
    return hits


def retrieve_context_filtered(
    query_text: str,
    *,
    where: Dict[str, Any],
    k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic retrieval using Chroma metadata filters.

    This is the function you will use later when you store:
    - templates (doc_type=py_template / xosc_template)
    - rules (doc_type=rule)
    - atoms (doc_type=atom)

    Example:
        retrieve_context_filtered(
            query_text="need VRU python template",
            where={"doc_type": "py_template", "family": "VRU", "version": "v1"},
            k=1,
        )
    """
    coll = _ensure_chroma()
    k = k or TOP_K

    q_emb = embed_texts([query_text])[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k, where=where)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    hits: List[Dict[str, Any]] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        hits.append(
            {
                "id": f"hit_{i}",
                "document": doc,
                "metadata": meta or {},
            }
        )
    return hits
if __name__ == "__main__":
    ingest_knowledge_base()
    print("✅ Knowledge base ingested into Chroma")
