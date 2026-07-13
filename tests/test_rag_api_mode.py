# SPDX-License-Identifier: AGPL-3.0-or-later
"""api-mode (``localm serve`` without the GUI) regression tests for the RAG
plugin (checkup audit, CONSOLIDATED-FINDINGS-2026-07-09 items 8 and 9; then
memory-audit 2026-07-02 cluster 24 / batch F14).

Item 8 (HIGH): rag_add / rag_upload / rag_query read
``request.app.state.self_url`` / ``.active_model`` / ``.jobs`` unguarded. Those
are only published by ``attach_gui`` - a bare ``localm serve`` never calls it -
so any client hitting the documented REST API directly got an unhandled
AttributeError -> opaque HTTP 500. First fixed (2026-07-09) by mirroring the
coder plugin's ``getattr(..., None)`` guard so the crash became a clean 503.

Cluster 24 / F14 (memory campaign): a clean 503 stopped the crash but headless
API users still could not index at all ("run localm gui"). Now /add and /upload
run the index SYNCHRONOUSLY on the plugin pool (off the event loop, like
/extract) when no background job manager is attached, and ``_self_services``
derives self_url/active_model from the kernel's own bind coordinates so
self-embedding works headless too - so a bare ``localm serve`` can actually
index, not just fail cleanly.

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
#  Item 8 / cluster 24: api-mode must not crash AND must actually index        #
# --------------------------------------------------------------------------- #

class TestApiModeIndexesHeadless:
    def test_add_indexes_synchronously_when_no_job_manager(self, api_mode_app):
        """Headless serve has no background job manager (attach_gui is never
        called), yet a documented REST client must still be able to index. Pre-fix
        this 503'd ("run localm gui"); now /add runs the index synchronously on the
        plugin pool and returns the result directly, so a bare ``localm serve`` can
        index (memory-audit cluster 24). The doc is really indexed - a query
        returns it."""
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            # Under Path.home() (pinned by the fixture), so the confinement
            # whitelist check (which runs BEFORE the jobs branch) passes on every
            # platform and the request reaches the synchronous index under test.
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(target)], "embed": False})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body.get("status") == "done", body
            assert body["added"] == 1, body
            assert "job_id" not in body           # synchronous, not a streamed job
            q = c.post("/api/rag/collections/kb/query",
                       json={"query": "gfx1030", "k": 4})
            assert q.status_code == 200, q.text
            hits = q.json()["hits"]
            assert hits and "gfx1030" in hits[0]["text"].lower()

    def test_upload_indexes_synchronously_when_no_job_manager(self, api_mode_app):
        """Same for device-file /upload: synchronous index headless, no 503."""
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md",
                           "content_b64": _b64(b"hello gfx1030 world")}],
                "embed": False})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body.get("status") == "done", body
            assert body["added"] == 1, body
            assert "job_id" not in body

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
        # THE POINT (finding 1): with no bind coordinates on app.state (this bare
        # fixture never runs advertise()), _self_services derives no self_url, so
        # query has no embedder and degrades to the store's lexical-only fallback
        # (embed_fn=None) instead of 503-ing or crashing.
        from localm.rag.store import Collection
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            # Index directly through the store to isolate the QUERY path under
            # test (the synchronous /add path is covered above).
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
#  Cluster 24: self-services derived from the kernel's bind coordinates        #
# --------------------------------------------------------------------------- #

def test_self_services_derived_from_kernel_state_when_headless():
    """attach_gui (the GUI shell) is the only setter of app.state.self_url /
    active_model, but a bare ``localm serve`` still advertises its bind
    coordinates (instance_scheme / instance_port). The rag plugin derives
    self_url + a live-engine active_model from those, so self-embedding (and the
    format / image self-classify helpers) work headless instead of every index
    silently degrading to lexical-only (memory-audit cluster 24)."""
    from types import SimpleNamespace

    from localm.plugins.builtin.rag import plug

    # No self_url / active_model published, but advertise()-style coordinates are.
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        instance_scheme="http", instance_port=8699)))
    self_embed, self_classify, self_describe = plug._self_services(req)
    assert self_embed is not None
    assert self_classify is not None
    assert self_describe is not None


def test_self_services_none_when_no_coordinates():
    """With neither the GUI services nor bind coordinates on app.state (a bare
    create_app, or before advertise()), self-embedding stays off (None trio) and
    indexing degrades cleanly to lexical-only rather than dialling a bogus URL."""
    from types import SimpleNamespace

    from localm.plugins.builtin.rag import plug

    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert plug._self_services(req) == (None, None, None)


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
