"""memory_check — async node: reuse a similar past run's sources before researching.

Input  (state): ``topic``
Output (state update): ``memory_hit`` (bool), ``reused_sources`` (list[Source]), ``warnings``

Embeds the topic and queries the ChromaDB memory layer. On a hit (cosine similarity >=
``MEMORY_SIMILARITY_THRESHOLD``) the downstream research agent can reuse the stored sources
instead of doing a fresh web search. Degrades gracefully if the store is unavailable.

This is the first node of the V2 graph and is async (the V2 graph runs via ``astream``).
"""

from __future__ import annotations

import asyncio
import logging

from memory import store
from pipeline import config
from pipeline.state import V2PipelineState

logger = logging.getLogger("pipeline.memory_check")


async def memory_check(state: V2PipelineState) -> dict:
    """Look up the topic in persistent memory and flag any reusable past run."""

    topic = state["topic"]
    warnings = list(state.get("warnings", []))

    try:
        hits = await asyncio.to_thread(
            store.query_similar, topic, config.MEMORY_SIMILARITY_THRESHOLD
        )
    except Exception as exc:  # ChromaDB unavailable / corrupt store / etc.
        logger.warning("memory_check: memory unavailable (%s); proceeding without reuse", exc)
        warnings.append(f"Memory layer unavailable ({exc}); proceeding with fresh research.")
        return {"memory_hit": False, "reused_sources": [], "warnings": warnings}

    if hits:
        top = hits[0]
        logger.info(
            "memory_check: HIT slug=%s similarity=%.3f -> reuse %d source(s)",
            top.slug, top.similarity, len(top.sources),
        )
        warnings.append(
            f"Memory hit: reusing {len(top.sources)} source(s) from '{top.slug}' "
            f"(similarity {top.similarity:.2f})."
        )
        return {"memory_hit": True, "reused_sources": top.sources, "warnings": warnings}

    logger.info("memory_check: MISS for topic %r -> full research", topic)
    return {"memory_hit": False, "reused_sources": []}
