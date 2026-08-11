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

LM-DA-015 (design audit, Low): the headless sync call sites (this file's
subject) called add_paths/add_uploads with no ``on_progress``, unlike the
job-manager and CLI paths. The embed-failure degrade warning
("embeddings unavailable ... indexing lexical-only", store.py) is only ever
surfaced through ``on_progress``, so it was silently discarded headless - a
doc that fell back to lexical-only looked like an ordinary success. Fixed by
passing a logging-backed on_progress (``plug._log_progress``).

LM-DA-018 (design audit, Low): /add had no path/file-count cap, unlike
/upload's explicit 50-file cap, despite running on the same shared, bounded
plugin ThreadPoolExecutor (also used by /extract, /query, web fetch, voice
transcription, coder sessions) since #593. Fixed with a 50-path cap mirroring
/upload's.
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
    ``localm serve`` (api-mode) looks like. ``.self_url`` / ``.active_model``
    are absent, unlike every existing rag test fixture (which calls attach_gui
    and so never exercised this path).

    ``app.state.jobs`` IS present, because since ADR-0008 the background-job
    registry is created by ``attach_engine`` rather than ``attach_gui``, so a
    headless server has one. This fixture publishes it directly instead of
    running the whole of attach_engine, which is what makes it api-mode rather
    than a bare app - see ``api_mode_app_no_jobs`` for the app that genuinely
    has none.

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

    from localm.plugins.gui.jobs import JobManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    app.state.jobs = JobManager()
    return app


@pytest.fixture
def api_mode_app_no_jobs(tmp_path, monkeypatch):
    """An app whose routes were mounted WITHOUT attach_engine, so it has no job
    registry at all. Not a real serving mode - it is a construction error - but
    it is the shape audit item 8 was about, and the guard must still turn it
    into a clean 503 rather than an unguarded AttributeError -> opaque 500."""
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


def _await_job(app, job_id, timeout=30.0):
    """Block until a background job leaves 'running', then return it."""
    jobs = app.state.jobs
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        assert job is not None, f"job {job_id} vanished from the registry"
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


# --------------------------------------------------------------------------- #
#  Item 8 / cluster 24: api-mode must not crash AND must actually index        #
# --------------------------------------------------------------------------- #

class TestApiModeIndexesHeadless:
    def test_add_runs_as_a_background_job_headless(self, api_mode_app):
        """A headless ``localm serve`` indexes through the SAME streamed
        background job the GUI uses (ADR-0008).

        History, because this assertion has now been inverted twice: originally
        /add 503'd headless ("run localm gui"); then cluster 24 made it index
        SYNCHRONOUSLY and return the result inline, because no job manager
        existed outside the GUI; now the registry is kernel-level, so headless
        gets a job_id like everyone else and can follow progress instead of
        blocking on one long request. The doc is really indexed - a query
        returns it."""
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            # Under Path.home() (pinned by the fixture), so the confinement
            # whitelist check (which runs BEFORE this) passes on every platform.
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(target)], "embed": False})
            assert r.status_code == 200, r.text
            body = r.json()
            assert "job_id" in body, f"headless /add must start a job now: {body}"
            job = _await_job(api_mode_app, body["job_id"])
            assert job.status == "done", f"job ended {job.status}"
            q = c.post("/api/rag/collections/kb/query",
                       json={"query": "gfx1030", "k": 4})
            assert q.status_code == 200, q.text
            hits = q.json()["hits"]
            assert hits and "gfx1030" in hits[0]["text"].lower()

    def test_upload_runs_as_a_background_job_headless(self, api_mode_app):
        """Same for device-file /upload: a streamed job headless, not a 503 and
        no longer a synchronous inline result."""
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.md",
                           "content_b64": _b64(b"hello gfx1030 world")}],
                "embed": False})
            assert r.status_code == 200, r.text
            body = r.json()
            assert "job_id" in body, f"headless /upload must start a job now: {body}"
            job = _await_job(api_mode_app, body["job_id"])
            assert job.status == "done", f"job ended {job.status}"

    def test_add_without_a_job_registry_is_a_clean_503(self, api_mode_app_no_jobs):
        """Audit item 8 stays fixed. An app whose routes were mounted without
        attach_engine has no registry, and that must still be a clean 503 rather
        than an unguarded AttributeError -> opaque 500. The message must NOT
        blame the GUI, which stopped being the reason when the registry moved to
        kernel level."""
        with TestClient(api_mode_app_no_jobs) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(target)], "embed": False})
            assert r.status_code == 503, r.text
            assert "localm gui" not in r.text.lower(), (
                "the GUI is no longer why a job registry might be missing")

    def test_add_logs_embed_degrade_when_headless(self, api_mode_app, caplog):
        """LM-DA-015: a headless /add whose embedder is broken must not silently
        report ordinary success. Pre-fix, add_paths' on_progress-or-noop
        (store.py) discarded the "embeddings unavailable ... indexing
        lexical-only" line entirely when on_progress was None, which it always
        was for the headless sync call sites - the response looked identical
        to a fully-vectored success. Now the headless call sites pass
        plug._log_progress, so the degrade reaches the debug logger.

        Publishes a real (but unreachable) self_url/active_model on
        app.state, exactly what ``_self_services`` derives ``self_embed``
        from (see ``_make_self_embed``), so ``embed_fn`` genuinely raises
        (a connection error) the same way a real down/misconfigured embedder
        would - not a mocked-away shortcut. The plugin module under test is
        loaded fresh by ``PluginManager`` via ``importlib`` under a private
        ``sys.modules`` key, so monkeypatching the normally-imported
        ``localm.plugins.builtin.rag.plug`` would silently miss the live
        route entirely."""
        api_mode_app.state.self_url = "http://127.0.0.1:1"   # nothing listens
        api_mode_app.state.active_model = lambda: "test-model"
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            with caplog.at_level("WARNING", logger="localm"):
                r = c.post("/api/rag/collections/kb/add",
                           json={"paths": [str(target)], "embed": True})
                # The degrade must not fail the request - it still indexes,
                # lexically. Since ADR-0008 this runs as a job even headless, so
                # wait for it INSIDE the caplog block or the warning lands after
                # capture stops and this passes for the wrong reason.
                assert r.status_code == 200, r.text
                job = _await_job(api_mode_app, r.json()["job_id"])
                assert job.status == "done", f"job ended {job.status}"
        # THE POINT survives the move to the job path: the degrade must still
        # reach the LOG, not only the job's ephemeral event stream, because the
        # log is what a bug report carries (LM-DA-015). See plug._job_progress.
        assert "embeddings unavailable" in caplog.text
        assert "indexing lexical-only" in caplog.text

    def test_add_rejects_too_many_paths(self, api_mode_app):
        """LM-DA-018: /add had no cap on path count, unlike /upload's 50-file
        cap, despite both running on the same shared, bounded executor headless.
        A 51-path request must be rejected the same way /upload already rejects
        a 51-file one - before the missing-file check, so this does not depend
        on any of the paths actually existing."""
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            paths = [str(Path.home() / f"doc{i}.txt") for i in range(51)]
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": paths, "embed": False})
            assert r.status_code == 400, r.text
            assert "too many paths" in r.text.lower()

    def test_embedding_set_starts_a_job_headless(self, api_mode_app):
        """Embedding-model setup needs a job for its download-progress stream,
        and headless now has one, so it starts the job instead of 503-ing with
        "run localm gui". confirm=True: the unconfirmed dry-run (see
        TestEmbeddingSetConfirmGate below) never reaches the job registry at
        all, so this must actually confirm to exercise that path."""
        with TestClient(api_mode_app) as c:
            r = c.post("/api/rag/embedding",
                       json={"model": "bge-small-en-v1.5", "confirm": True})
            assert r.status_code == 200, r.text
            assert "job_id" in r.json(), r.text

    def test_embedding_set_without_a_job_registry_is_a_clean_503(
            self, api_mode_app_no_jobs):
        """The other half of audit item 8: still a 503, never a 500.
        confirm=True: only the confirmed path touches the job registry."""
        with TestClient(api_mode_app_no_jobs) as c:
            r = c.post("/api/rag/embedding",
                       json={"model": "bge-small-en-v1.5", "confirm": True})
            assert r.status_code == 503, r.text
            assert "localm gui" not in r.text.lower()

    def test_embedding_set_unconfirmed_needs_no_job_registry(
            self, api_mode_app_no_jobs):
        """The dry-run report is answered synchronously and reads meta.json
        only - it must work even where _require_jobs would 503, since it
        never reaches that gate."""
        with TestClient(api_mode_app_no_jobs) as c:
            r = c.post("/api/rag/embedding", json={"model": "bge-small-en-v1.5"})
            assert r.status_code == 200, r.text
            assert r.json()["needs_confirm"] is True

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
