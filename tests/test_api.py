"""FastAPI backend tests (in-process via httpx ASGITransport, mock mode)."""

from __future__ import annotations

import asyncio
import json

import httpx
from httpx import ASGITransport

from memory import store
from pipeline.state import Source
from v2.backend.main import app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _src() -> Source:
    return Source(url="https://example.com/1", title="T", retrieved_at="now",
                  snippet="s", credibility_note="ok")


def test_health():
    async def run():
        async with _client() as c:
            return (await c.get("/api/health")).json()
    assert asyncio.run(run())["status"] == "ok"


def test_voice_transcribe():
    async def run():
        async with _client() as c:
            files = {"file": ("topic.mp3", b"fake-audio", "audio/mpeg")}
            return (await c.post("/api/voice/transcribe", files=files)).json()
    assert asyncio.run(run())["transcript"]


def test_memory_search_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.config.CHROMA_PERSIST_DIR", str(tmp_path))
    store.save_run("api-slug", "API topic about widgets", "FS", [_src()], 0.9, persist_dir=str(tmp_path))

    async def run():
        async with _client() as c:
            search = (await c.get("/api/memory/search", params={"topic": "API topic about widgets"})).json()
            deleted = (await c.delete("/api/memory/api-slug")).json()
            return search, deleted

    search, deleted = asyncio.run(run())
    assert any(h["slug"] == "api-slug" for h in search["results"])
    assert deleted["status"] == "deleted"


def test_run_endpoint_returns_id():
    """The HTTP POST /api/run route creates a run and returns an id."""
    async def run():
        async with _client() as c:
            return (await c.post("/api/run", json={
                "topic": "AI agents", "target_keyword": "kw", "style_persona": "op-ed"})).json()
    assert asyncio.run(run())["run_id"]


def test_run_stream_approve_flow(tmp_path, monkeypatch):
    """Drive the SSE generator directly (no httpx) — fast + reliable; same code the route uses."""
    monkeypatch.setattr("pipeline.config.OUTPUT_DIR", str(tmp_path))
    from v2.backend import stream as S

    async def run():
        rs = S.start_run("apitest-run", "AI agents", "kw", "op-ed", None)
        events: list[str] = []
        approved = False
        async for sse in S.event_stream(rs):
            ev = json.loads(sse["data"])
            events.append(ev["event"])
            if ev["event"] == "hitl_required" and not approved:
                await S.approve(rs)
                approved = True
            elif ev["event"] in ("complete", "error"):
                break
        return events

    events = asyncio.run(run())
    assert "node_complete" in events
    assert "hitl_required" in events
    assert "complete" in events
    assert "error" not in events


def test_batch_stream_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.config.OUTPUT_DIR", str(tmp_path))
    from v2.backend import stream as S

    async def run():
        bs = S.start_batch("apitest-batch", ["AI agents", "Climate tech"], "kw", "technical deep-dive")
        events: list[str] = []
        async for sse in S.batch_event_stream(bs):
            ev = json.loads(sse["data"])
            events.append(ev["event"])
            if ev["event"] in ("complete", "error"):
                break
        return events

    events = asyncio.run(run())
    assert events.count("topic_complete") == 2
    assert "complete" in events and "error" not in events
