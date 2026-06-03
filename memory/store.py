"""ChromaDB-backed persistent memory for cross-run fact/source reuse (V2).

Each completed run is stored keyed by ``slug``: the fact sheet as the document, the topic's
embedding as the vector, and the sources (+ scores/date) as metadata. Embeddings come from
our own OpenAI model (deterministic in mock mode), so Chroma's default embedder is disabled
(``embedding_function=None``) — no model download, and dimensions always match.

The collection uses cosine space, so ``similarity = 1 - distance``. ``query_similar`` returns
past runs whose topic similarity meets a threshold (default 0.82).

All functions are synchronous (ChromaDB is sync); async callers wrap them in
``asyncio.to_thread``. Each call opens a ``PersistentClient`` against ``persist_dir`` (defaults
to ``config.CHROMA_PERSIST_DIR``), so tests can point at an isolated temp directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings

from pipeline import config, llm
from pipeline.state import MemoryHit, Source

logger = logging.getLogger("memory.store")

_SETTINGS = Settings(anonymized_telemetry=False)


def _collection(persist_dir: str | None = None):
    """Open (or create) the persistent memory collection in cosine space."""

    path = persist_dir or config.CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=path, settings=_SETTINGS)
    return client.get_or_create_collection(
        name=config.MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # we supply embeddings ourselves
    )


def save_run(
    slug: str,
    topic: str,
    fact_sheet: str,
    sources: list[Source],
    final_score: float,
    *,
    persist_dir: str | None = None,
    run_date: str | None = None,
) -> str:
    """Persist (upsert) one completed run so future similar topics can reuse it.

    Returns the slug. Re-saving the same slug overwrites the previous entry.
    """

    collection = _collection(persist_dir)
    embedding = llm.embed_texts([topic])[0]
    metadata: dict[str, Any] = {
        "topic": topic,
        "slug": slug,
        "sources_json": json.dumps([s.model_dump() for s in sources], ensure_ascii=False),
        "run_date": run_date or datetime.now(timezone.utc).isoformat(),
        "final_score": float(final_score),
    }
    collection.upsert(
        ids=[slug], embeddings=[embedding], documents=[fact_sheet or ""], metadatas=[metadata]
    )
    logger.info("memory.save_run: stored '%s' (topic=%r)", slug, topic)
    return slug


def query_similar(
    topic_text: str,
    threshold: float | None = None,
    *,
    persist_dir: str | None = None,
    n_results: int = 5,
) -> list[MemoryHit]:
    """Return past runs whose topic cosine-similarity meets ``threshold`` (desc by similarity)."""

    threshold = config.MEMORY_SIMILARITY_THRESHOLD if threshold is None else threshold
    collection = _collection(persist_dir)
    total = collection.count()
    if total == 0:
        return []

    query_embedding = llm.embed_texts([topic_text])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=min(n_results, total))

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    hits: list[MemoryHit] = []
    for idx, distance in enumerate(distances):
        similarity = 1.0 - float(distance)
        if similarity < threshold:
            continue
        meta = metadatas[idx] or {}
        sources = [Source(**s) for s in json.loads(meta.get("sources_json", "[]"))]
        hits.append(
            MemoryHit(
                slug=ids[idx],
                topic=meta.get("topic", ""),
                similarity=similarity,
                sources=sources,
                fact_sheet=documents[idx] or "",
                run_date=meta.get("run_date", ""),
            )
        )
    return hits


def delete_run(slug: str, *, persist_dir: str | None = None) -> None:
    """Remove a single run from memory (used by tests and the /api/memory DELETE route)."""

    _collection(persist_dir).delete(ids=[slug])
    logger.info("memory.delete_run: removed '%s'", slug)
