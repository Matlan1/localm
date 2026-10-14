# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cancelling a native media job that is waiting to stop the other native runtime.

A native job that does not fit in VRAM stops this process's other native media
runtime (the KoboldCpp music server or the shared stable-diffusion.cpp worker).
Stopping waits for the job that runtime is running. Cancelling the waiting job
ends that wait: nothing is stopped, the chat model is left loaded, and the job
body returns.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from localm import vram
from localm.discover import FREE_SCOPE_DEVICE
from localm.media.koboldcpp import server as kobold_server
from localm.media.sdcpp import shared as sd_shared

from tests.test_media_reload_on_every_exit import (_BODIES, _Job, _fake_vram, _raise)

GB = 1024 ** 3


def _hold(lock):
    """Hold *lock* on a thread until the returned event is set."""
    taken, release = threading.Event(), threading.Event()

    def run():
        with lock:
            taken.set()
            release.wait(10)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert taken.wait(5)
    return release, t


def _cancel_after(job, seconds):
    def run():
        time.sleep(seconds)
        job.cancel_requested = True

    threading.Thread(target=run, daemon=True).start()


def test_sdcpp_free_returns_false_without_stopping_when_cancelled_while_waiting(monkeypatch):
    shutdown = MagicMock()
    monkeypatch.setattr(sd_shared.runner, "shutdown", shutdown)
    release, t = _hold(sd_shared.lock)
    try:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        t0 = time.monotonic()
        assert sd_shared.free(flag.is_set) is False
        assert time.monotonic() - t0 < 3
        shutdown.assert_not_called()
    finally:
        release.set()
        t.join(5)
    assert not sd_shared.lock.locked()


def test_koboldcpp_stop_returns_false_without_stopping_when_cancelled_while_waiting(monkeypatch):
    srv = MagicMock()
    monkeypatch.setattr(kobold_server, "_server", srv)
    release, t = _hold(kobold_server._lock)
    try:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        t0 = time.monotonic()
        assert kobold_server.stop(flag.is_set) is False
        assert time.monotonic() - t0 < 3
        srv.stop.assert_not_called()
        assert kobold_server._server is srv
    finally:
        release.set()
        t.join(5)
    assert not kobold_server._lock.locked()


def test_both_stops_still_wait_for_the_running_job_without_a_cancel(monkeypatch):
    shutdown = MagicMock()
    monkeypatch.setattr(sd_shared.runner, "shutdown", shutdown)
    release, t = _hold(sd_shared.lock)
    threading.Timer(0.5, release.set).start()
    t0 = time.monotonic()
    sd_shared.free()
    assert time.monotonic() - t0 >= 0.4
    shutdown.assert_called_once()
    t.join(5)


def _settings():
    return {"backend": "native", "vram_estimate_bytes": 12 * GB}


def _short_reading():
    return (int(4 * GB), True, FREE_SCOPE_DEVICE)


def test_a_job_cancelled_while_music_is_generating_does_not_stop_the_music_runtime(monkeypatch):
    srv = MagicMock()
    monkeypatch.setattr(kobold_server, "_server", srv)
    monkeypatch.setattr(kobold_server, "running_pid", lambda: 4242)
    monkeypatch.setattr(sd_shared, "worker_pid", lambda: None)
    release, t = _hold(kobold_server._lock)
    job = _Job()
    _cancel_after(job, 0.3)
    try:
        t0 = time.monotonic()
        out = vram.release_other_native_media(
            job, _settings(), "image", read_reading=_short_reading,
            read_free=lambda: 64 * GB)
        assert time.monotonic() - t0 < 3
    finally:
        release.set()
        t.join(5)
    assert out == []
    srv.stop.assert_not_called()
    assert job.lines[-1] == "Cancelled."
    assert not any("kept its models" in x for x in job.lines)


def test_a_job_cancelled_while_an_image_is_generating_does_not_stop_the_sd_worker(monkeypatch):
    shutdown = MagicMock()
    monkeypatch.setattr(sd_shared.runner, "shutdown", shutdown)
    monkeypatch.setattr(sd_shared, "worker_pid", lambda: 4343)
    monkeypatch.setattr(kobold_server, "running_pid", lambda: None)
    release, t = _hold(sd_shared.lock)
    job = _Job()
    _cancel_after(job, 0.3)
    try:
        out = vram.release_other_native_media(
            job, _settings(), "music", read_reading=_short_reading,
            read_free=lambda: 64 * GB)
    finally:
        release.set()
        t.join(5)
    assert out == []
    shutdown.assert_not_called()
    assert job.lines[-1] == "Cancelled."


def test_a_job_cancelled_before_the_stop_starts_stops_nothing(monkeypatch):
    srv = MagicMock()
    monkeypatch.setattr(kobold_server, "_server", srv)
    monkeypatch.setattr(kobold_server, "running_pid", lambda: 4242)
    monkeypatch.setattr(sd_shared, "worker_pid", lambda: None)
    job = _Job()
    job.cancel_requested = True
    out = vram.release_other_native_media(
        job, _settings(), "image", read_reading=_short_reading, read_free=lambda: 64 * GB)
    assert out == []
    srv.stop.assert_not_called()


@pytest.mark.parametrize("body", sorted(_BODIES))
def test_a_cancelled_job_does_not_go_on_to_unload_the_chat_model(monkeypatch, body):
    _fake_vram(monkeypatch)
    unloads = []
    generated = []

    def release_then_cancel(job, s, label):
        job.cancel_requested = True
        return []

    monkeypatch.setattr("localm.vram.release_other_native_media", release_then_cancel)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda *a, **k: unloads.append(a) or True)
    fn = _BODIES[body](monkeypatch, backend_call=lambda *a, **k: generated.append(a) or _raise())
    assert fn(_Job()) is False
    assert unloads == []
    assert generated == []
