# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hardware check behind the plugin catalog runs off the event loop.

``catalog.offered`` asks ``hwdetect.detect()`` on Linux, which runs ``lspci`` with
an 8 s timeout. The two routes that build the plugin list on the event loop run
that check on a worker thread first, once per process."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import hwdetect
from localm.plugins import catalog


class _DetectSpy:
    """Stands in for ``hwdetect.detect``; records each call and whether it ran
    in a thread with a running event loop."""

    def __init__(self, vendors=("nvidia",)):
        self.vendors = list(vendors)
        self.on_loop: list[bool] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self):
        try:
            asyncio.get_running_loop()
            self.on_loop.append(True)
        except RuntimeError:
            self.on_loop.append(False)
        self.entered.set()
        self.release.wait(10)
        return hwdetect.Detection(vendors=self.vendors, recommended="vulkan", source="spy")


@pytest.fixture
def spy(monkeypatch):
    """A Linux machine whose hardware check has not run in this process."""
    s = _DetectSpy()
    monkeypatch.setattr(hwdetect, "platform_key", lambda: "linux")
    monkeypatch.setattr(hwdetect, "detect", s)
    monkeypatch.setattr(catalog, "_hardware_checked", False)
    catalog._nvidia_linux.cache_clear()
    yield s
    catalog._nvidia_linux.cache_clear()


def _engine_app() -> FastAPI:
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    return app


def _gui_app() -> FastAPI:
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "model-a")
    return app


def test_the_plugins_route_runs_the_check_off_the_loop(spy):
    with TestClient(_engine_app()) as client:
        r = client.get("/api/plugins")
    assert r.status_code == 200
    assert spy.on_loop == [False]
    assert "exllama" in {p["name"] for p in r.json()["plugins"]}


def test_the_capabilities_route_runs_the_check_off_the_loop(spy):
    with TestClient(_gui_app()) as client:
        r = client.get("/api/capabilities")
    assert r.status_code == 200
    assert spy.on_loop == [False]


def test_a_machine_without_the_hardware_does_not_list_the_plugin(spy):
    spy.vendors = []
    with TestClient(_engine_app()) as client:
        r = client.get("/api/plugins")
    assert spy.on_loop == [False]
    assert "exllama" not in {p["name"] for p in r.json()["plugins"]}


def test_the_check_runs_once_across_requests(spy):
    with TestClient(_engine_app()) as client:
        client.get("/api/plugins")
        client.get("/api/plugins")
    assert len(spy.on_loop) == 1


def test_concurrent_callers_wait_for_one_check(spy):
    spy.release.clear()
    threads = [threading.Thread(target=catalog.check_hardware) for _ in range(3)]
    for t in threads:
        t.start()
    assert spy.entered.wait(10)
    time.sleep(0.2)
    calls_while_running = len(spy.on_loop)
    spy.release.set()
    for t in threads:
        t.join(10)
    assert calls_while_running == 1
    assert len(spy.on_loop) == 1


def test_a_stalled_check_does_not_hold_the_event_loop(spy):
    spy.release.clear()

    async def scenario():
        task = asyncio.ensure_future(catalog.check_hardware_off_loop())
        ticks = 0
        while not spy.entered.is_set():
            await asyncio.sleep(0.01)
            ticks += 1
            assert ticks < 1000
        await asyncio.sleep(0)
        loop_ran_during_the_check = not task.done()
        spy.release.set()
        await task
        return loop_ran_during_the_check

    assert asyncio.run(scenario()) is True
    assert spy.on_loop == [False]


def test_after_the_check_nothing_is_offloaded(spy, monkeypatch):
    catalog.check_hardware()

    def no_thread(*a, **k):
        raise AssertionError("to_thread was called")

    monkeypatch.setattr(asyncio, "to_thread", no_thread)
    asyncio.run(catalog.check_hardware_off_loop())
    assert catalog.offered("exllama") is True
    assert len(spy.on_loop) == 1
