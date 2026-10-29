# SPDX-License-Identifier: AGPL-3.0-or-later
"""With reload-after-generation off, localm's own native media worker stays
loaded after a job. Unload all must stop it when idle (and report it busy, not
kill it, while a job runs), a chat model load that is short of VRAM must stop
it first, and the low-VRAM explanation must name it instead of an outside app.

The stand-in worker is a real child process installed where the shared
stable-diffusion.cpp worker lives (``sdcpp.shared.runner``), or where the music
runtime lives (``koboldcpp.server._server``). The fake GPU reports its VRAM as
held for as long as that child is alive, so every assertion about freed VRAM
follows the real stop path: ``runner.shutdown()`` killing the child.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import discover, gpu_registry
from localm.inference import http_server as hs
from localm.inference import switch_admission
from localm.media.koboldcpp import server as kobold_server
from localm.media.sdcpp import shared as sd_shared

GB = 1024 ** 3
TOTAL = 16 * GB
WORKER_HOLDS = 10 * GB
IMAGE_WORKER = "native image/video worker"
MUSIC_RUNTIME = "native music runtime"


class _ChildProc:
    """A real child process with the ``multiprocessing.Process`` surface
    ``SdRunner.shutdown`` uses."""

    def __init__(self):
        self.popen = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"])

    @property
    def pid(self):
        return self.popen.pid

    def is_alive(self):
        return self.popen.poll() is None

    def join(self, timeout=None):
        try:
            self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def kill(self):
        self.popen.kill()


class _MusicServer:
    """The parts of ``koboldcpp.server._Server`` that ``running_pid`` and
    ``stop`` use, over a real child process."""

    def __init__(self):
        self.mp = _ChildProc()

    def alive(self):
        return self.mp.is_alive()

    def stop(self):
        self.mp.kill()
        self.mp.join(5)


@pytest.fixture
def sd_worker():
    child = _ChildProc()
    sd_shared.runner._proc = child
    sd_shared.runner._req_q = None
    try:
        yield child
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
        sd_shared.cancel_idle_timer()
        sd_shared.runner._reset()


@pytest.fixture
def music_runtime():
    srv = _MusicServer()
    kobold_server._server = srv
    try:
        yield srv.mp
    finally:
        if srv.mp.is_alive():
            srv.stop()
        kobold_server._cancel_idle_timer()
        kobold_server._server = None


def _held_bytes() -> int:
    held = 0
    if sd_shared.worker_pid() is not None:
        held += WORKER_HOLDS
    if kobold_server.running_pid() is not None:
        held += WORKER_HOLDS
    return held


@pytest.fixture
def fake_gpu(monkeypatch):
    """A 16 GB card whose free VRAM drops by 10 GB per live stand-in worker."""
    def vram_capacity(return_status=False, **_kw):
        info = {"free": TOTAL - _held_bytes(), "total": TOTAL,
                "free_scope": discover.FREE_SCOPE_DEVICE}
        return (info, discover.GPU_PROBE_OK) if return_status else info

    monkeypatch.setattr(discover, "vram_capacity", vram_capacity)
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    yield
    for d in (hs._engines, hs._engines_lru, hs._inference_sems):
        d.clear()


@pytest.fixture
def gui_client(fake_gpu):
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: hs._active_model_name)
    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------- #
#  Unload all                                                                  #
# --------------------------------------------------------------------------- #

def test_gui_unload_all_stops_an_idle_image_worker_and_reports_it(gui_client, sd_worker):
    assert TOTAL - _held_bytes() == 6 * GB

    r = gui_client.post("/api/models/unload", json={})

    assert r.status_code == 200, r.text
    data = r.json()
    assert not sd_worker.is_alive(), "the idle image worker is still running"
    assert data["status"] == "unloaded", data
    assert data["unloaded_other"] == [IMAGE_WORKER], data
    assert "skipped_in_use" not in data and "media_left_running" not in data
    assert data["vram_freed"] is True, data
    assert data["vram_after_bytes"] - data["vram_before_bytes"] == WORKER_HOLDS


def test_gui_unload_all_stops_an_idle_music_runtime(gui_client, music_runtime):
    data = gui_client.post("/api/models/unload", json={}).json()
    assert not music_runtime.is_alive()
    assert data["unloaded_other"] == [MUSIC_RUNTIME], data


@pytest.mark.parametrize("force", [False, True])
def test_gui_unload_all_reports_a_busy_image_worker_and_leaves_it_running(
        gui_client, sd_worker, force):
    assert sd_shared.lock.acquire(timeout=5)
    try:
        data = gui_client.post("/api/models/unload", json={"force": force}).json()
        assert sd_worker.is_alive(), "Unload all killed the worker of a running job"
    finally:
        sd_shared.lock.release()
    assert data["status"] == "in_use", data
    assert data["skipped_in_use"] == [IMAGE_WORKER], data
    assert data["media_left_running"] == {IMAGE_WORKER: "busy"}, data
    assert "confirm_required" not in data, data
    assert data["unloaded_other"] == [], data


def test_the_api_unload_takes_media_false_for_the_media_handover(fake_gpu, sd_worker,
                                                                 monkeypatch):
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    app = hs.create_app(None)
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {app.state.shell_token}"

    kept = client.post("/v1/models/unload", params={"media": "false"}).json()
    assert sd_worker.is_alive(), "the media handover's unload stopped the worker it reuses"
    assert kept["status"] == "already_unloaded", kept

    stopped = client.post("/v1/models/unload").json()
    assert not sd_worker.is_alive()
    assert stopped["unloaded_other"] == [IMAGE_WORKER], stopped


def test_the_media_handover_unload_asks_to_keep_media(monkeypatch):
    from localm import selfclient, vram

    paths = []

    class _Resp:
        ok = True
        status_code = 200

        def json(self):
            return {"status": "already_unloaded"}

    monkeypatch.setattr(selfclient, "self_request",
                        lambda method, path, **kw: (paths.append(path), _Resp())[1])
    job = type("J", (), {"push": lambda self, ev: None})()
    assert vram.unload_chat_for_media(job, "http://127.0.0.1:9/v1", "image") is True
    assert paths == ["/models/unload?media=false"]


def test_unload_all_without_media_leaves_the_worker_for_other_callers(fake_gpu, sd_worker):
    data = asyncio.run(hs.unload_all_models())
    assert sd_worker.is_alive()
    assert data["unloaded_other"] == [], data


# --------------------------------------------------------------------------- #
#  A chat model load that is short of VRAM                                    #
# --------------------------------------------------------------------------- #

def _budget(need=8 * GB):
    return switch_admission.LoadBudget(
        name="chat-x", vram_required=need, headroom=0, resident_cap=None,
        pinned=frozenset(), check_split_fit=False)


def _make_room(budget):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(hs._switch_make_room(
            loop, budget, preempt=False, force=False, activate=True))
    finally:
        loop.close()


@pytest.fixture
def no_peers(monkeypatch):
    monkeypatch.setattr(hs, "_attempt_cooperative_unload", lambda **kw: False)
    monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda **kw: [])


def test_a_chat_load_short_of_vram_stops_the_idle_image_worker_first(
        fake_gpu, no_peers, sd_worker, caplog):
    caplog.set_level(logging.INFO, logger="localm")
    assert _make_room(_budget()) is None
    assert not sd_worker.is_alive(), "the chat model loads beside the idle image worker"
    assert TOTAL - _held_bytes() == TOTAL
    assert any(f"stopped the idle {IMAGE_WORKER} to make room for the chat model "
               "'chat-x'" in r.getMessage() for r in caplog.records)


def test_a_chat_load_that_fits_leaves_the_image_worker_loaded(fake_gpu, no_peers, sd_worker):
    assert _make_room(_budget(need=4 * GB)) is None
    assert sd_worker.is_alive()


def test_a_chat_load_leaves_a_busy_image_worker_running(fake_gpu, no_peers, sd_worker):
    assert sd_shared.lock.acquire(timeout=5)
    try:
        assert _make_room(_budget()) is None
        assert sd_worker.is_alive()
    finally:
        sd_shared.lock.release()


# --------------------------------------------------------------------------- #
#  The low-VRAM explanation                                                    #
# --------------------------------------------------------------------------- #

def _hint(tmp_path, monkeypatch):
    from localm.inference.backends.gguf import GgufBackend
    monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda **kw: [])
    monkeypatch.setattr(gpu_registry, "own_status", lambda: None)
    monkeypatch.setattr("localm.discover.resolve_main_gpu_index",
                        lambda configured, **k: 0)
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False,
                       n_ctx=4096)._vram_holder_hint()


def test_the_low_vram_hint_names_localms_own_image_worker(tmp_path, monkeypatch, sd_worker):
    hint = _hint(tmp_path, monkeypatch)
    assert "another GPU app" not in hint
    assert f"this localm server's own {IMAGE_WORKER} is holding it" in hint
    assert "Unload all" in hint


def test_the_low_vram_hint_without_an_own_worker_stays_generic(tmp_path, monkeypatch):
    assert sd_shared.worker_pid() is None
    assert "another GPU app" in _hint(tmp_path, monkeypatch)


def test_the_stand_in_worker_is_a_live_child_until_stopped(sd_worker):
    assert sd_shared.worker_pid() == sd_worker.pid
    assert sd_shared.free_if_idle() is True
    deadline = time.monotonic() + 5
    while sd_worker.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not sd_worker.is_alive()
