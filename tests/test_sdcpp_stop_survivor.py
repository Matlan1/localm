# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stable-diffusion.cpp worker that keeps running after its kill is never
reported stopped: the stop answers False, the media unload reports it failed,
its process id stays readable for the VRAM accounting and the runtime list, and
the next stop kills it again.

The worker is a real child process; only its ``kill`` is made to do nothing,
which is what a kill that does not take looks like to the runner."""

from __future__ import annotations

import logging

import pytest

from localm import vram
from localm.media.sdcpp import shared as sd_shared
from tests.test_media_keep_unload import IMAGE_WORKER, _ChildProc


class _KillDoesNotTake(_ChildProc):
    def __init__(self):
        super().__init__()
        self.kill_takes = False

    def kill(self):
        if self.kill_takes:
            super().kill()


@pytest.fixture
def stubborn_worker():
    child = _KillDoesNotTake()
    sd_shared.runner._proc = child
    sd_shared.runner._req_q = None
    try:
        yield child
    finally:
        child.kill_takes = True
        child.kill()
        child.join(5)
        sd_shared.cancel_idle_timer()
        sd_shared.runner._reset()
        sd_shared.runner._survivors = []


def _quick_join(child, monkeypatch):
    join = child.join
    monkeypatch.setattr(child, "join", lambda timeout=None: join(0.1))


def test_a_worker_that_outlives_its_kill_is_not_reported_stopped(stubborn_worker, monkeypatch,
                                                                caplog):
    caplog.set_level(logging.WARNING, logger="localm")
    _quick_join(stubborn_worker, monkeypatch)
    assert sd_shared.free_if_idle() is False
    assert stubborn_worker.is_alive()
    assert sd_shared.worker_pid() == stubborn_worker.pid
    assert any(f"pid {stubborn_worker.pid}) kept running after it was killed" in r.getMessage()
               for r in caplog.records)


def test_the_media_unload_reports_it_failed_and_still_lists_it(stubborn_worker, monkeypatch):
    _quick_join(stubborn_worker, monkeypatch)
    assert vram.stop_idle_native_media() == {IMAGE_WORKER: vram.MEDIA_STOP_FAILED}
    assert vram.native_media_runtime_states() == {IMAGE_WORKER: "idle"}


def test_the_next_stop_kills_the_survivor_again(stubborn_worker, monkeypatch):
    _quick_join(stubborn_worker, monkeypatch)
    assert sd_shared.free_if_idle() is False
    monkeypatch.undo()
    stubborn_worker.kill_takes = True
    assert sd_shared.free() is True
    assert not stubborn_worker.is_alive()
    assert sd_shared.worker_pid() is None
    assert vram.native_media_runtime_states() == {}
