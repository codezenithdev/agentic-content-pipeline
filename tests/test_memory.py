"""Tests for the ChromaDB memory layer + memory_check node (mock mode, isolated temp dirs)."""

from __future__ import annotations

import asyncio

from memory import store
from pipeline.agents.memory_check import memory_check
from pipeline.state import Source, initial_v2_state


def _src(url: str = "https://example.com/1") -> Source:
    return Source(url=url, title="T", retrieved_at="now", snippet="s", credibility_note="ok")


def test_save_then_query_hits_above_threshold(tmp_path):
    d = str(tmp_path)
    store.save_run("slug-a", "Mediterranean diet benefits", "FACT SHEET", [_src()], 0.9, persist_dir=d)
    hits = store.query_similar("Mediterranean diet benefits", persist_dir=d)  # identical -> sim 1.0
    assert hits, "expected a memory hit for the identical topic"
    assert hits[0].slug == "slug-a"
    assert hits[0].similarity >= 0.82
    assert hits[0].sources[0].url == "https://example.com/1"
    assert hits[0].fact_sheet == "FACT SHEET"


def test_query_misses_below_threshold(tmp_path):
    d = str(tmp_path)
    store.save_run("slug-a", "Mediterranean diet benefits", "FS", [_src()], 0.9, persist_dir=d)
    hits = store.query_similar("Quantum computing hardware roadmap", persist_dir=d)
    assert hits == [], "unrelated topic should fall below the similarity threshold"


def test_persistence_across_client_instances(tmp_path):
    d = str(tmp_path)
    store.save_run("slug-b", "Topic X about widgets", "FS", [_src()], 0.5, persist_dir=d)
    # A fresh query opens a new PersistentClient — proves data is on disk, not just in memory.
    hits = store.query_similar("Topic X about widgets", persist_dir=d)
    assert hits and hits[0].slug == "slug-b"


def test_delete_run(tmp_path):
    d = str(tmp_path)
    store.save_run("slug-c", "Topic Y to delete", "FS", [_src()], 0.5, persist_dir=d)
    store.delete_run("slug-c", persist_dir=d)
    assert store.query_similar("Topic Y to delete", persist_dir=d) == []


def test_memory_check_node_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.config.CHROMA_PERSIST_DIR", str(tmp_path))
    store.save_run("slug-d", "AI agents", "FS", [_src()], 0.9, persist_dir=str(tmp_path))
    out = asyncio.run(memory_check(initial_v2_state("AI agents", "langgraph")))
    assert out["memory_hit"] is True
    assert out["reused_sources"] and out["reused_sources"][0].url == "https://example.com/1"


def test_memory_check_node_miss(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.config.CHROMA_PERSIST_DIR", str(tmp_path))
    store.save_run("slug-e", "Gardening in winter", "FS", [_src()], 0.9, persist_dir=str(tmp_path))
    out = asyncio.run(memory_check(initial_v2_state("Distributed systems consensus", "raft")))
    assert out["memory_hit"] is False
    assert out["reused_sources"] == []
