# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two concurrent ``POST /api/rag/embedding`` confirms must not start TWO
``embed-setup`` jobs. When they do, the second waits - silently, with no timeout
and no error - on the embedder's bare ``_LOAD_LOCK``/``_LOCK`` acquires, and both
jobs sit at "Loading and testing the model..." indefinitely while unrelated reads
time out.

The second one is refused with 409 at the door, the same shape every sibling long
job on this server uses (runtime-update, comfy-setup, comfy-update, doctor).
These tests pin the refusal AND the normal path, so a guard that refuses
everything fails just as loudly as one that refuses nothing.

Built on a job that is GENUINELY IN FLIGHT (blocked inside ``get_embedder``), not
on a hand-set status field: the fixture's value space has to contain the real
concurrent state, or the test cannot fail on the defect.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def rag_home(tmp_path, monkeypatch) -> Path:
    """Same isolation as test_rag_dim_switch_warning.py's fixture of this name:
    a throwaway LOCALM_HOME so config writes and collection reads never touch
    the developer's real data dir."""
    home = tmp_path / "userhome"
    home.mkdir()
    data = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", data)
    monkeypatch.setattr(cfg, "MODELS_DIR", data / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", data / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", data / "registry.json")
    from localm.rag.store import rag_dir
    return rag_dir()


@pytest.fixture
def embedding_route_app(rag_home):
    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui

    app = FastAPI()
    PluginManager(app, external_root=rag_home.parent / "noplugins").install("rag")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _embed_setup_jobs(app) -> list:
    return [j for j in app.state.jobs._jobs.values() if j.kind == "embed-setup"]


class _StubEmbedder:
    @staticmethod
    def embed(texts):
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def blocking_embedder(monkeypatch):
    """``get_embedder`` blocks until the test releases it - the real shape of
    the first job holding the load locks while a second request arrives.

    Yields ``(entered, release)``: wait on *entered* to know the job is truly
    inside the load, then ``release.set()`` to let it finish."""
    import localm.inference.embedder as emb

    entered = threading.Event()
    release = threading.Event()

    def _blocking_get_embedder(**kw):
        entered.set()
        release.wait(30)
        return _StubEmbedder()

    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda **kw: "/fake/new-model.gguf")
    monkeypatch.setattr(emb, "get_embedder", _blocking_get_embedder)
    yield entered, release
    release.set()


class TestConcurrentEmbedSetupIsRefused:
    def test_a_second_confirm_while_one_runs_is_refused_and_starts_no_job(
            self, embedding_route_app, blocking_embedder):
        entered, release = blocking_embedder
        from fastapi.testclient import TestClient

        app = embedding_route_app
        with TestClient(app) as c:
            first = c.post("/api/rag/embedding",
                           json={"model": "model-one", "confirm": True})
            assert first.status_code == 200, first.text
            assert entered.wait(10), "the first job never reached the load"

            second = c.post("/api/rag/embedding",
                            json={"model": "model-two", "confirm": True})

            # Assert on the WORLD first, the status code second: the harm is a
            # second job existing at all and then wedging.
            assert len(_embed_setup_jobs(app)) == 1, (
                "a SECOND embed-setup job was started while one was already "
                "running; it will block on the embedder's unbounded load lock "
                "with no timeout and no error")
            assert second.status_code == 409, second.text
            assert "already running" in second.json()["detail"]

            release.set()

    def test_the_refusal_does_not_write_config_or_reset_the_embedder(
            self, embedding_route_app, blocking_embedder):
        """The refused request must be a no-op. The job body's very first acts
        are ``update_config`` + ``reset_embedder``, so a guard placed after
        start_fn (or not at all) would already have switched the model the
        user was just told was refused."""
        entered, release = blocking_embedder
        import localm.config as cfg
        from fastapi.testclient import TestClient

        app = embedding_route_app
        with TestClient(app) as c:
            first = c.post("/api/rag/embedding",
                           json={"model": "model-one", "confirm": True})
            assert first.status_code == 200, first.text
            assert entered.wait(10)
            # The first job has already written its own choice; capture that,
            # then prove the REFUSED one changes nothing.
            before = cfg.load_config().get("embedding_model")

            second = c.post("/api/rag/embedding",
                            json={"model": "model-two", "confirm": True})
            assert second.status_code == 409, second.text
            assert cfg.load_config().get("embedding_model") == before
            assert before != "model-two"

            release.set()

    def test_a_setup_still_starts_when_none_is_running(
            self, embedding_route_app, monkeypatch):
        """The guard must not refuse the ordinary case. Without this, a guard
        that returned 409 unconditionally would pass the test above."""
        import localm.inference.embedder as emb
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder", lambda **kw: _StubEmbedder())

        from fastapi.testclient import TestClient
        app = embedding_route_app
        with TestClient(app) as c:
            r = c.post("/api/rag/embedding",
                       json={"model": "model-one", "confirm": True})
            assert r.status_code == 200, r.text
            job = app.state.jobs.get(r.json()["job_id"])
            deadline = time.time() + 10
            while job.status == "running" and time.time() < deadline:
                time.sleep(0.02)
            assert job.status == "done", job.status

            # And once it has FINISHED, a further switch is allowed again -
            # the guard keys on "running", not on "one has ever run".
            again = c.post("/api/rag/embedding",
                           json={"model": "model-two", "confirm": True})
            assert again.status_code == 200, again.text

    def test_the_unconfirmed_dry_run_is_never_refused(
            self, embedding_route_app, blocking_embedder):
        """The dry run writes nothing, starts no job, and answers instantly -
        so it must stay available while a setup runs, which is exactly when a
        user is most likely to be looking at that page."""
        entered, release = blocking_embedder
        from fastapi.testclient import TestClient

        with TestClient(embedding_route_app) as c:
            first = c.post("/api/rag/embedding",
                           json={"model": "model-one", "confirm": True})
            assert first.status_code == 200, first.text
            assert entered.wait(10)

            dry = c.post("/api/rag/embedding", json={"model": "model-two"})
            assert dry.status_code == 200, dry.text
            assert dry.json()["needs_confirm"] is True

            release.set()


class TestTheSetupJobIsNotSilentWhileItLoads:
    """The stream must not go silent while the job loads.

    ``get_embedder`` announces coarse stages - the VRAM/eviction wait and the
    native load each carry their own 300 s window, so this is the one call in the
    job that can legitimately run for minutes. A route that passes no sink leaves
    the stream stopped dead at "Loading and testing the model..." for that entire
    time, which is indistinguishable from a wedge to whoever is watching.
    /api/embedding/warmup already consumes the same stages.
    """

    def test_the_load_stages_reach_the_job_stream(
            self, embedding_route_app, monkeypatch):
        import localm.inference.embedder as emb

        seen = {}

        def _staged_get_embedder(*, on_progress=None, **kw):
            seen["sink"] = on_progress
            if on_progress is not None:
                on_progress("QA7-STAGE-EVICTING-THEN-LOADING")
            return _StubEmbedder()

        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder", _staged_get_embedder)

        from fastapi.testclient import TestClient
        app = embedding_route_app
        with TestClient(app) as c:
            r = c.post("/api/rag/embedding",
                       json={"model": "model-one", "confirm": True})
            assert r.status_code == 200, r.text
            job = app.state.jobs.get(r.json()["job_id"])
            deadline = time.time() + 10
            while job.status == "running" and time.time() < deadline:
                time.sleep(0.02)
            assert job.status == "done", job.status
            lines = [e.get("text", "") for e in job._history
                     if e.get("type") == "line"]

            # The sink itself first: a sink that was never handed over cannot
            # arrive from anywhere else.
        assert seen.get("sink") is not None, (
            "the setup job ran get_embedder with no progress sink, so the "
            "minutes-long load stage emits nothing at all")
        assert "QA7-STAGE-EVICTING-THEN-LOADING" in lines
