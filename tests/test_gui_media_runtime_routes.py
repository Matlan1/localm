# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/media/runtimes and POST /api/media/runtimes/unload: the native
media runtimes a finished job leaves loaded are listed with their state, an
idle one is stopped on request, a busy one is left running and reported, and
the routes need models:read and models:write.

The stand-in workers are real child processes installed where the shared
stable-diffusion.cpp worker and the KoboldCpp music runtime live (the fixtures
of test_media_keep_unload), so a stop is the real ``shutdown`` killing them."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.media.koboldcpp import server as kobold_server
from localm.media.sdcpp import shared as sd_shared
from tests.test_media_keep_unload import (  # noqa: F401
    IMAGE_WORKER, MUSIC_RUNTIME, music_runtime, sd_worker)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=lambda name: None,
               active_model=lambda: "")
    with TestClient(app) as c:
        yield c


def test_nothing_loaded_lists_nothing(client):
    assert client.get("/api/media/runtimes").json() == {"runtimes": []}
    assert client.post("/api/media/runtimes/unload").json() == {"outcomes": []}


def test_a_loaded_worker_is_listed_idle_and_stopped_on_request(client, sd_worker):  # noqa: F811
    body = client.get("/api/media/runtimes").json()
    assert body == {"runtimes": [{"name": IMAGE_WORKER, "state": "idle",
                                  "idle_unload_seconds": sd_shared.IDLE_UNLOAD_SECONDS}]}
    out = client.post("/api/media/runtimes/unload").json()
    assert out == {"outcomes": [{"name": IMAGE_WORKER, "outcome": "stopped"}]}
    assert not sd_worker.is_alive()
    assert client.get("/api/media/runtimes").json() == {"runtimes": []}


def test_a_worker_a_job_uses_is_busy_and_left_running(client, sd_worker):  # noqa: F811
    with sd_shared.lock:
        assert client.get("/api/media/runtimes").json()["runtimes"][0]["state"] == "busy"
        out = client.post("/api/media/runtimes/unload").json()
        assert out == {"outcomes": [{"name": IMAGE_WORKER, "outcome": "busy"}]}
        assert sd_worker.is_alive()


def test_both_runtimes_are_listed_with_their_own_idle_time(client, sd_worker,  # noqa: F811
                                                           music_runtime):  # noqa: F811
    rows = {r["name"]: r for r in client.get("/api/media/runtimes").json()["runtimes"]}
    assert set(rows) == {IMAGE_WORKER, MUSIC_RUNTIME}
    assert rows[MUSIC_RUNTIME]["idle_unload_seconds"] == kobold_server.IDLE_SECONDS
    out = client.post("/api/media/runtimes/unload").json()["outcomes"]
    assert sorted((o["name"], o["outcome"]) for o in out) == [
        (IMAGE_WORKER, "stopped"), (MUSIC_RUNTIME, "stopped")]
    assert not sd_worker.is_alive() and not music_runtime.is_alive()


def test_the_routes_need_models_read_and_models_write(client, sd_worker):  # noqa: F811
    from localm import auth
    reader = auth.create_key("reader", [S.MODELS_READ])["key"]
    other = auth.create_key("chat-only", [S.CHAT])["key"]
    hdr = lambda k: {"Authorization": f"Bearer {k}"}  # noqa: E731
    assert client.get("/api/media/runtimes", headers=hdr(reader)).status_code == 200
    assert client.post("/api/media/runtimes/unload", headers=hdr(reader)).status_code == 403
    assert sd_worker.is_alive()
    assert client.get("/api/media/runtimes", headers=hdr(other)).status_code == 403
    writer = auth.create_key("writer", [S.MODELS_WRITE, S.MODELS_READ])["key"]
    out = client.post("/api/media/runtimes/unload", headers=hdr(writer)).json()
    assert out == {"outcomes": [{"name": IMAGE_WORKER, "outcome": "stopped"}]}
