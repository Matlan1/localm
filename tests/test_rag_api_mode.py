# SPDX-License-Identifier: AGPL-3.0-or-later
"""api-mode (``localm serve`` without the GUI) regression tests for the RAG
plugin.

rag_add / rag_upload / rag_query read ``request.app.state.self_url`` /
``.active_model`` / ``.jobs``. Those are only published by ``attach_gui`` - a
bare ``localm serve`` never calls it - so reading them unguarded gives any
client hitting the documented REST API directly an unhandled AttributeError,
surfacing as an opaque HTTP 500. The coder plugin's ``getattr(..., None)``
guard turns that into a clean 503.

A clean 503 stops the crash but leaves headless API users unable to index at
all ("run localm gui"). /add and /upload run the index SYNCHRONOUSLY on the
plugin pool (off the event loop, like /extract) when no background job manager
is attached, and ``_self_services`` derives self_url/active_model from the
kernel's own bind coordinates so self-embedding works headless too - so a bare
``localm serve`` can actually index, not just fail cleanly.

rag_extract running extract_bytes() synchronously inside an async route with no
executor offload freezes the whole single-worker event loop for every
route/user for the duration of an archive extraction; it is offloaded to
loop.run_in_executor, mirroring rag_upload's background-job offload.

The headless sync call sites (this file's subject) must pass ``on_progress``,
like the job-manager and CLI paths. The embed-failure degrade warning
("embeddings unavailable ... indexing lexical-only", store.py) is only ever
surfaced through ``on_progress``, so with None it is silently discarded and a
doc that fell back to lexical-only looks like an ordinary success. They pass a
logging-backed on_progress (``plug._log_progress``).

/add caps path/file count at 50, like /upload: both run on the same shared,
bounded plugin ThreadPoolExecutor (also used by /extract, /query, web fetch,
voice transcription, coder sessions).
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
    are absent, unlike every other rag test fixture (which calls attach_gui and
    so never exercises this path).

    ``app.state.jobs`` IS present, because the background-job registry is
    created by ``attach_engine`` rather than ``attach_gui``, so a headless
    server has one. This fixture publishes it directly instead of running the
    whole of attach_engine, which is what makes it api-mode rather than a bare
    app - see ``api_mode_app_no_jobs`` for the app that genuinely has none.

    Pins ``Path.home`` under tmp_path: rag_add's whitelist confinement only
    allows paths under home/cwd/an allowed root, and pytest's tmp_path is NOT
    reliably nested under the real home (it happens to be on Windows, but is a
    sibling of $HOME under Linux CI's /tmp) - without pinning it, a test target
    file placed directly under tmp_path would spuriously 409 on confinement
    instead of ever reaching the 503 guard under test."""
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
    the guard must still turn it into a clean 503 rather than an unguarded
    AttributeError -> opaque 500."""
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
#  api-mode must not crash AND must actually index                             #
# --------------------------------------------------------------------------- #

class TestApiModeIndexesHeadless:
    def test_add_runs_as_a_background_job_headless(self, api_mode_app):
        """A headless ``localm serve`` indexes through the SAME streamed
        background job the GUI uses: the job registry is kernel-level, so
        headless gets a job_id like everyone else and can follow progress
        instead of blocking on one long request. The doc is really indexed - a
        query returns it."""
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
        """An app whose routes were mounted without attach_engine has no
        registry, and that must be a clean 503 rather than an unguarded
        AttributeError -> opaque 500. The message must NOT blame the GUI, which
        stopped being the reason when the registry moved to kernel level."""
        with TestClient(api_mode_app_no_jobs) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(target)], "embed": False})
            assert r.status_code == 503, r.text
            assert "localm gui" not in r.text.lower(), (
                "the GUI is no longer why a job registry might be missing")

    def test_add_logs_embed_degrade_when_headless(self, api_mode_app, caplog,
                                                    monkeypatch):
        """A headless /add whose embedder is broken must not silently report
        ordinary success. add_paths' on_progress-or-noop (store.py) discards the
        "embeddings unavailable ... indexing lexical-only" line entirely when
        on_progress is None, which is what the headless sync call sites would
        pass, making the response identical to a fully-vectored success. The
        headless call sites pass plug._log_progress, so the degrade reaches the
        debug logger.

        Publishes a real (but unreachable) self_url/active_model on
        app.state, exactly what ``_self_services`` derives ``self_embed``
        from (see ``_make_self_embed``), so ``embed_fn`` genuinely raises
        (a connection error) the same way a real down/misconfigured embedder
        would - not a mocked-away shortcut. The plugin module under test is
        loaded fresh by ``PluginManager`` via ``importlib`` under a private
        ``sys.modules`` key, so monkeypatching the normally-imported
        ``localm.plugins.builtin.rag.plug`` would silently miss the live
        route entirely.

        ``_self_services`` additionally withholds ``self_embed`` when no
        embedding model resolves on disk (``resolve_embedding_model_path``),
        which this hermetic ``LOCALM_HOME`` never has - so that check is
        patched open here to isolate the scenario under test (a resolvable
        but unreachable embedder), the same way ``test_rag.py`` patches it for
        ``GET /api/rag/embedding``."""
        from localm.inference import embedder as emb
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda *, allow_download=None: "/models/embeddings/fake.gguf")
        api_mode_app.state.self_url = "http://127.0.0.1:1"   # nothing listens
        api_mode_app.state.active_model = lambda: "test-model"
        with TestClient(api_mode_app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = Path.home() / "doc.txt"
            target.write_text("gfx1030 rocm runtime notes", encoding="utf-8")
            with caplog.at_level("WARNING", logger="localm"):
                r = c.post("/api/rag/collections/kb/add",
                           json={"paths": [str(target)], "embed": True})
                # The degrade must not fail the request: it still indexes,
                # lexically. The indexing runs as a job even headless, so wait
                # for it INSIDE the caplog block or the warning lands after
                # capture stops.
                assert r.status_code == 200, r.text
                job = _await_job(api_mode_app, r.json()["job_id"])
                assert job.status == "done", f"job ended {job.status}"
        # The degrade must reach the LOG, not only the job's ephemeral event
        # stream. See plug._job_progress.
        assert "embeddings unavailable" in caplog.text
        assert "indexing lexical-only" in caplog.text

    def test_add_rejects_too_many_paths(self, api_mode_app):
        """/add caps path count at 50, like /upload's 50-file cap, since both
        run on the same shared, bounded executor headless. A 51-path request
        must be rejected the same way /upload already rejects a 51-file one -
        before the missing-file check, so this does not depend on any of the
        paths actually existing."""
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
        # With no bind coordinates on app.state (this bare fixture never runs
        # advertise()), _self_services derives no self_url, so query has no
        # embedder and degrades to the store's lexical-only fallback
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
#  Self-services derived from the kernel's bind coordinates                    #
# --------------------------------------------------------------------------- #

def test_self_services_derived_from_kernel_state_when_headless(monkeypatch):
    """attach_gui (the GUI shell) is the only setter of app.state.self_url /
    active_model, but a bare ``localm serve`` still advertises its bind
    coordinates (instance_scheme / instance_port). The rag plugin derives
    self_url + a live-engine active_model from those, so self-embedding (and the
    format / image self-classify helpers) work headless instead of every index
    silently degrading to lexical-only, provided an embedding model actually
    resolves - patched open here since this is a test of the DERIVATION, not of
    embedder installedness (see the sibling test below for that)."""
    from types import SimpleNamespace

    from localm.inference import embedder as emb
    from localm.plugins.builtin.rag import plug

    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: "/models/embeddings/fake.gguf")
    # No self_url / active_model published, but advertise()-style coordinates are.
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        instance_scheme="http", instance_port=8699)))
    self_embed, self_classify, self_describe = plug._self_services(req)
    assert self_embed is not None
    assert self_classify is not None
    assert self_describe is not None


def test_self_services_withholds_self_embed_when_no_embedder_installed():
    """The GUI-shell-attached (or kernel-derived) case is not on its own enough
    to promise self-embedding: without an embedding model actually resolving on
    disk, self_embed must be None even though self_classify/self_describe (which
    key off active_model, not the embedder) are still returned - otherwise a
    caller relying on self_embed's mere presence (e.g. rag_repair's
    would_lose_embeddings guard) never sees the degrade."""
    from types import SimpleNamespace

    from localm.plugins.builtin.rag import plug

    # No embedding model configured in this hermetic LOCALM_HOME, so
    # resolve_embedding_model_path is genuinely None here - nothing patched.
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        instance_scheme="http", instance_port=8699)))
    self_embed, self_classify, self_describe = plug._self_services(req)
    assert self_embed is None
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
#  rag_extract must offload to a thread, not block the event loop             #
# --------------------------------------------------------------------------- #

def test_rag_extract_offloads_extraction_off_the_event_loop(monkeypatch):
    """A single-worker server means a synchronous extract_bytes() call inside
    the async route would freeze EVERY route for EVERY user for its duration.
    Proven by racing a lightweight ticker coroutine against a deliberately slow
    (but thread-safe, since it is meant to run off-loop) extract_bytes: with the
    executor offload the ticker keeps landing WHILE extraction is in flight;
    with an inline call the ticker gets no chance to run until the slow call
    returns."""
    import localm.rag as ragpkg
    from localm.plugins.builtin.rag.plug import RagExtractRequest, rag_extract

    def _slow_extract(data, filename):
        time.sleep(0.3)
        return "extracted text"

    monkeypatch.setattr(ragpkg, "extract_bytes", _slow_extract)

    ticks: list[float] = []

    async def ticker(t0: float, stop_at: float):
        # Timestamps relative to a start captured BEFORE either task exists, not
        # to whenever the ticker coroutine gets its first turn: measuring its own
        # start would let a fully-blocked loop merely delay the first tick rather
        # than skip it.
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


# --- an attachment must be the WHOLE file, not a preview --------------------
# Neither chat.js nor coder.js sends RagExtractRequest.max_chars, so an
# attachment must not be trimmed by a default cap. Images take a different path
# (data URI -> image_url) and are unaffected.

def _extract(client, name, blob, **body):
    import base64
    payload = {"filename": name, "content_b64": base64.b64encode(blob).decode()}
    payload.update(body)
    return client.post("/api/rag/extract", json=payload)


def test_extract_returns_the_whole_file_by_default(api_mode_app):
    """The default must be the entire document, with no cap applied."""
    big = ("QA-ATTACH-WHOLE-FILE line %d\n" % 0).encode()
    body = b"".join(b"line %d padding padding padding\n" % i for i in range(4000))
    blob = big + body + b"QA-ATTACH-TAIL-CANARY\n"
    with TestClient(api_mode_app) as c:
        r = _extract(c, "big.txt", blob)
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(blob.decode()) > 24_000, "fixture must exceed the OLD 24k default to be meaningful"
    # Assert on the CONTENT first: a length check alone would pass on a
    # different 24k-plus string, while the tail canary is present only if
    # nothing was cut off the end.
    assert "QA-ATTACH-TAIL-CANARY" in d["text"], "the END of the file did not survive"
    assert d["truncated"] is False
    assert d["chars"] == len(d["text"])


def test_extract_still_honours_an_explicit_max_chars(api_mode_app):
    """A caller that WANTS an excerpt can still ask for one."""
    blob = b"".join(b"line %d padding padding padding\n" % i for i in range(4000))
    with TestClient(api_mode_app) as c:
        r = _extract(c, "big.txt", blob, max_chars=1000)
        # The FULL length comes from an unbounded call, not from the raw bytes:
        # extraction normalises text, so len(blob) is not the extracted length.
        full = _extract(c, "big.txt", blob).json()["text"]
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["text"]) == 1000
    assert d["truncated"] is True
    assert d["chars"] == len(full), "chars must report the FULL length, not the excerpt"
    assert len(full) > 1000, "fixture must exceed the requested excerpt to be meaningful"


def test_extract_handles_n_files_independently(api_mode_app):
    """N attachments: each one comes back whole, and they do not bleed together."""
    blobs = {
        "a.txt": b"AAA-CANARY\n" + b"a" * 30_000 + b"\nAAA-TAIL",
        "b.txt": b"BBB-CANARY\n" + b"b" * 30_000 + b"\nBBB-TAIL",
        "c.txt": b"CCC-CANARY\n" + b"c" * 30_000 + b"\nCCC-TAIL",
    }
    with TestClient(api_mode_app) as c:
      for name, blob in blobs.items():
        d = _extract(c, name, blob).json()
        tag = name[0].upper() * 3
        assert d["text"].startswith(tag + "-CANARY"), f"{name} lost its head"
        assert d["text"].endswith(tag + "-TAIL"), f"{name} lost its tail"
        assert d["truncated"] is False
        for other in "ABC".replace(name[0].upper(), ""):
            assert other * 3 + "-CANARY" not in d["text"], f"{name} bled into another file"
