# SPDX-License-Identifier: AGPL-3.0-or-later
"""api-mode (``localm serve`` without the GUI) regression tests for the RAG
plugin (checkup audit, CONSOLIDATED-FINDINGS-2026-07-09 items 8 and 9).

Item 8 (HIGH): rag_add / rag_upload / rag_query read
``request.app.state.self_url`` / ``.active_model`` / ``.jobs`` unguarded. Those
are only published by ``attach_gui`` - a bare ``localm serve`` never calls it -
so any client hitting the documented REST API directly got an unhandled
AttributeError -> opaque HTTP 500. Fixed by mirroring the coder plugin's
``getattr(..., None)`` + clean 503 guard, and degrading self-embedding to
lexical-only (an already-supported ``embed_fn=None`` path) rather than crashing.

Item 9 (HIGH): rag_extract ran extract_bytes() synchronously inside an async
route with no executor offload, freezing the whole single-worker event loop for
every route/user for the duration of an archive extraction. Fixed by offloading
to loop.run_in_executor, mirroring rag_upload's background-job offload.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def api_mode_app(tmp_path, monkeypatch):
    """The rag plugin mounted with NO ``attach_gui`` call - exactly what a bare
    ``localm serve`` (api-mode) looks like. ``app.state.jobs`` / ``.self_url`` /
    ``.active_model`` are all absent, unlike every existing rag test fixture
    (which calls attach_gui and so never exercised this path).

    Pins ``Path.home`` under tmp_path, mirroring test_rag_confinement.py's
    ``home_env`` fixture: rag_add's whitelist confinement only allows paths
    under home/cwd/an allowed root, and pytest's tmp_path is NOT reliably
    nested under the real home (it happens to be on Windows, but is a sibling
    of $HOME under Linux CI's /tmp) - without pinning it, a test target file
    placed directly under tmp_path would spuriously 409 on confinement instead
    of ever reaching the 503 guard under test."""
    from localm.plugins.engine import PluginManager
    home = tmp_path / "userhome"
    home.mkdir()
    localm = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    return app


# --------------------------------------------------------------------------- #
#  Item 8: api-mode must not crash with an AttributeError                     #
# --------------------------------------------------------------------------- #

class TestApiModeDoesNotCrash:
    def test_add_returns_clean_503_not_500(self, api_mode_app):
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            # Under Path.home() (pinned by the fixture), so the confinement
            # whitelist check (which runs BEFORE the jobs guard) passes and the
            # request actually reaches the 503 under test, on every platform.
            target = Path.home() / "doc.txt"
            target.write_text("hello world", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(target)], "embed": False})
            assert r.status_code == 503, r.text
            assert "background indexing" in r.text.lower()
            assert "localm gui" in r.text.lower()

    def test_upload_returns_clean_503_not_500(self, api_mode_app):
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md", "content_b64": _b64(b"hi there")}],
                "embed": False})
            assert r.status_code == 503, r.text
            assert "background indexing" in r.text.lower()
            assert "localm gui" in r.text.lower()

    def test_embedding_set_returns_clean_503_not_500(self, api_mode_app):
        with TestClient(api_mode_app) as c:
            r = c.post("/api/rag/embedding", json={"model": "bge-small-en-v1.5"})
            assert r.status_code == 503, r.text
            # THE POINT (review finding): the message must name what actually
            # needs the GUI here (embedding setup), not the indexing wording
            # used by /add and /upload - _require_jobs's default message would
            # be misleading on this route if not overridden.
            assert "embedding model setup" in r.text.lower()
            assert "background indexing" not in r.text.lower()
            assert "localm gui" in r.text.lower()

    def test_query_succeeds_and_degrades_to_lexical_only(self, api_mode_app):
        # THE POINT (finding 1): query needs no background job, only optional
        # self-embedding, so it must NOT 503 - it degrades to the store's
        # existing lexical-only fallback (embed_fn=None) and keeps working.
        from localm.rag.store import Collection
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            # /add and /upload need a job manager (unavailable here); index
            # directly through the store, exactly as a CLI-driven `localm rag
            # add` would, so this test proves the QUERY path, not indexing.
            coll = Collection("kb")
            coll.add_uploads([{"filename": "notes.md",
                                "data": b"rocm gfx1030 runtime notes"}])
            r = c.post("/api/rag/collections/kb/query",
                       json={"query": "gfx1030", "k": 4})
            assert r.status_code == 200, r.text
            hits = r.json()["hits"]
            assert hits and "gfx1030" in hits[0]["text"].lower()

    def test_query_on_empty_collection_does_not_crash(self, api_mode_app):
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/query",
                       json={"query": "anything", "k": 4})
            assert r.status_code == 200, r.text
            assert r.json()["hits"] == []


# --------------------------------------------------------------------------- #
#  Item 9: rag_extract must offload to a thread, not block the event loop     #
# --------------------------------------------------------------------------- #

def test_rag_extract_offloads_extraction_off_the_event_loop(monkeypatch):
    """A single-worker server means a synchronous extract_bytes() call inside
    the async route would freeze EVERY route for EVERY user for its duration.
    Proven by racing a lightweight ticker coroutine against a deliberately slow
    (but thread-safe, since it is meant to run off-loop) extract_bytes: with the
    executor offload the ticker keeps landing WHILE extraction is in flight;
    without it (the pre-fix inline call) the ticker would get no chance to run
    until the slow call returns."""
    import localm.rag as ragpkg
    from localm.plugins.builtin.rag.plug import RagExtractRequest, rag_extract

    def _slow_extract(data, filename):
        time.sleep(0.3)
        return "extracted text"

    monkeypatch.setattr(ragpkg, "extract_bytes", _slow_extract)

    ticks: list[float] = []

    async def ticker(t0: float, stop_at: float):
        # Timestamps relative to a start captured BEFORE either task exists, not
        # to whenever the ticker coroutine happens to get its first turn - if it
        # measured its own start, a fully-blocked loop would just delay the
        # ticker's first tick rather than skip it, making the test pass either
        # way (this bug was caught by running it against the pre-fix code).
        while time.monotonic() - t0 < stop_at:
            ticks.append(time.monotonic() - t0)
            await asyncio.sleep(0.02)

    async def scenario():
        req = RagExtractRequest(filename="x.txt", content_b64=_b64(b"hello"))
        t0 = time.monotonic()
        t_extract = asyncio.create_task(rag_extract(req))
        t_tick = asyncio.create_task(ticker(t0, 0.4))
        result = await t_extract
        await t_tick
        return result

    result = asyncio.run(scenario())
    assert result["text"] == "extracted text"
    early_ticks = [t for t in ticks if t < 0.25]
    assert len(early_ticks) >= 3, (
        f"event loop appears blocked during extraction - only "
        f"{len(early_ticks)} ticker iterations landed before the 250ms mark "
        f"(the 300ms slow extract_bytes should not have prevented them): "
        f"{ticks}")
