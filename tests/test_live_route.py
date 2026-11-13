# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/live: the status bar's push channel, over a real HTTP connection.

An in-process test client buffers a streaming response until the app finishes,
which an endless stream never does, so these run a real uvicorn server on a
loopback port and read it with httpx.

The fake GPU reports a stale VRAM figure to a routine reading and the true one
only to a forced reading (as the real refresh throttle does), so a frame that
carries the truth proves a kick forced it. The routine tick is set to a minute
so it cannot be what delivered anything.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import types
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import livefeed, portmux
from localm import scopes as S

GIB = 1024 ** 3


class FakeGpu:
    def __init__(self):
        self.truth = 10.0
        self.shown = 10.0
        self.system_calls = 0
        self.fresh_calls = 0

    def _payload(self):
        return {"cpu": {"percent": 10.0},
                "vram": {"used": int(self.shown * GIB), "total": 16 * GIB}}

    def system_stats(self, *, wait_first_vram=False):
        self.system_calls += 1
        return self._payload()

    def fresh_stats(self, *, timeout=3.0):
        self.fresh_calls += 1
        self.shown = self.truth
        return self._payload()

    @property
    def calls(self):
        return self.system_calls + self.fresh_calls


@pytest.fixture
def gpu(monkeypatch):
    g = FakeGpu()
    monkeypatch.setattr("localm.sysstats.system_stats", g.system_stats)
    monkeypatch.setattr("localm.sysstats.fresh_stats", g.fresh_stats)
    monkeypatch.setattr("localm.sysstats.cached_stats", g._payload)
    return g


@pytest.fixture
def app(tmp_path, monkeypatch, gpu):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(livefeed, "INTERVAL_S", 60.0)
    import localm.plugins.gui.web as web
    monkeypatch.setattr(web, "_KEEPALIVE_S", 0.2)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    a = FastAPI()
    attach_engine(a)
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda n: None, active_model=lambda: "model-a")
    yield a
    livefeed.install(None)


class Server:
    def __init__(self, app):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.server = portmux._register_server(uvicorn.Server(uvicorn.Config(
            app, log_level="warning", lifespan="off", timeout_graceful_shutdown=3)))
        self.thread = threading.Thread(
            target=lambda: asyncio.run(self.server.serve(sockets=[sock])), daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.server.started, "uvicorn did not start"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        portmux._servers.discard(self.server)


@pytest.fixture(autouse=True)
def _no_stopped_server_left_registered():
    """A server a test stopped leaves portmux's registry, so portmux.stopping()
    cannot end the next test's stream at once."""
    yield
    for s in list(portmux._servers):
        if getattr(s, "should_exit", False):
            portmux._servers.discard(s)


@pytest.fixture
def server(app):
    s = Server(app)
    yield s
    s.stop()


class Reader(threading.Thread):
    """Reads /api/live and keeps every line with the time it arrived."""

    def __init__(self, url, headers=None):
        super().__init__(daemon=True)
        self.url = url
        self.headers = headers or {}
        self.lines: list[tuple[float, str]] = []
        self.status = None
        self.content_type = None
        self.stop = threading.Event()
        self.started = threading.Event()
        self.dropped: Exception | None = None

    def run(self):
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as c:
                with c.stream("GET", self.url + "/api/live", headers=self.headers) as r:
                    self.status = r.status_code
                    self.content_type = r.headers.get("content-type")
                    self.started.set()
                    for line in r.iter_lines():
                        self.lines.append((time.monotonic(), line))
                        if self.stop.is_set():
                            break
        except httpx.HTTPError as e:
            if not self.stop.is_set():
                self.dropped = e

    def frames(self, kind):
        out = []
        for at, line in list(self.lines):
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev.get("type") == kind:
                    out.append((at, ev))
        return out

    def wait_for(self, cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return bool(cond())

    def close(self, expect_drop=False):
        self.stop.set()
        self.join(timeout=5)
        if not expect_drop:
            assert self.dropped is None, f"the server dropped the stream: {self.dropped!r}"


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _vram_gib(ev):
    return ev["data"]["vram"]["used"] / GIB


# ------------------------------------------------------------------ auth

def test_a_caller_without_a_key_is_refused_on_a_keyed_server(app):
    from localm import auth
    auth.create_key("someone", [S.MCP])
    with TestClient(app) as c:
        assert c.get("/api/live").status_code == 401
        assert c.get("/api/live", headers=_hdr("not-a-key")).status_code == 401


def test_a_key_with_no_extra_scope_streams_like_it_reads_api_stats(app, server):
    """/api/stats and /api/activity are baseline: any valid key. The stream is
    both of them, so it is baseline too."""
    from localm import auth
    narrow = auth.create_key("narrow", [S.MCP])["key"]
    r = Reader(server.url, _hdr(narrow))
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("stats") and r.frames("activity"))
        assert r.status == 200
        assert r.content_type.startswith("text/event-stream")
    finally:
        r.close()


# ------------------------------------------------------------ the push

def test_a_model_unload_reaches_the_meter_within_a_second(app, server, gpu):
    from localm.inference.engine import Engine

    r = Reader(server.url)
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("stats"))
        assert _vram_gib(r.frames("stats")[-1][1]) == 10.0
        gpu.truth = 2.0
        t0 = time.monotonic()
        Engine.unload(types.SimpleNamespace(
            _backend=types.SimpleNamespace(unload=lambda: None)))
        assert r.wait_for(lambda: _vram_gib(r.frames("stats")[-1][1]) == 2.0, timeout=3.0), \
            "the freed VRAM never reached the connected tab"
        landed = r.frames("stats")[-1][0]
        assert landed - t0 < 1.0, f"took {landed - t0:.2f}s"
    finally:
        r.close()


def test_a_value_that_does_not_change_is_not_sent_again(app, server, gpu):
    r = Reader(server.url)
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("stats"))
        livefeed.kick("test")
        livefeed.kick("test")
        assert r.wait_for(lambda: gpu.fresh_calls >= 3)
        time.sleep(0.3)
        assert len(r.frames("stats")) == 1, "identical readings were pushed again"
    finally:
        r.close()


def test_two_tabs_cost_one_reading(app, server, gpu):
    a, b = Reader(server.url), Reader(server.url)
    a.start()
    assert a.wait_for(lambda: a.frames("stats"))
    b.start()
    try:
        assert b.wait_for(lambda: b.frames("stats")), \
            "the second tab got no reading to start from"
        time.sleep(0.2)
        assert gpu.calls == 1, f"{gpu.calls} readings for two tabs"
        assert app.state.live_feed.subscriber_count == 2
    finally:
        a.close()
        b.close()


def test_a_quiet_stream_still_proves_it_is_alive(app, server):
    r = Reader(server.url)
    r.start()
    try:
        assert r.wait_for(lambda: any(line == ": keepalive" for _, line in r.lines)), \
            "nothing arrived while nothing changed; a client could not tell dead from quiet"
    finally:
        r.close()


def test_a_disconnected_tab_lets_the_sampler_stop(app, server, gpu):
    r = Reader(server.url)
    r.start()
    assert r.wait_for(lambda: r.frames("stats"))
    feed = app.state.live_feed
    assert feed.subscriber_count == 1
    r.close()
    deadline = time.monotonic() + 5
    while feed.subscriber_count and time.monotonic() < deadline:
        time.sleep(0.02)
    assert feed.subscriber_count == 0, "the closed connection stayed subscribed"
    assert feed.kick("x") is False


# ------------------------------------------------------------- activity

def _inject(app, *, owner=None, label="Model pull owner/repo"):
    from localm.jobmanager import Job
    j = Job(id=uuid.uuid4().hex[:12], kind="pull", argv=["secret"], owner=owner,
            label=label)
    j.status = "running"
    app.state.jobs._jobs[j.id] = j
    return j


def test_a_job_starting_and_ending_is_pushed_without_waiting_for_a_tick(
        app, server, monkeypatch):
    """The stream re-reads the job list when woken or when its keepalive
    interval passes. The interval is made longer than the test, so only a kick
    can wake it."""
    import localm.plugins.gui.web as web
    monkeypatch.setattr(web, "_KEEPALIVE_S", 30.0)
    r = Reader(server.url)
    r.start()
    release = threading.Event()
    try:
        assert r.wait_for(lambda: r.frames("activity"))
        assert r.frames("activity")[-1][1]["operations"] == []
        job = app.state.jobs.start_fn("pull", lambda j: release.wait(10) or True,
                                      label="Model pull a/b")
        assert r.wait_for(lambda: any(
            op["id"] == job.id and op["status"] == "running"
            for _, ev in r.frames("activity") for op in ev["operations"]), timeout=3.0), \
            "a started job was not pushed"
        release.set()
        assert r.wait_for(lambda: any(
            op["id"] == job.id and op["status"] == "done"
            for _, ev in r.frames("activity") for op in ev["operations"]), timeout=3.0), \
            "a finished job was not pushed"
    finally:
        release.set()
        r.close()


def test_activity_frames_carry_the_server_clock_for_ages(app, server):
    r = Reader(server.url)
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("activity"))
        ev = r.frames("activity")[-1][1]
        assert abs(ev["now"] - time.time()) < 5
        assert "stalls" in ev
    finally:
        r.close()


def test_another_principals_job_never_appears_in_your_stream(app, server):
    from localm import auth
    a = auth.create_key("A", [S.MODELS_READ])["key"]
    b = auth.create_key("B", [S.MODELS_READ])["key"]
    job = _inject(app, owner=auth._hash_key(a), label="Model pull private/repo")
    ra, rb = Reader(server.url, _hdr(a)), Reader(server.url, _hdr(b))
    ra.start()
    rb.start()
    try:
        assert ra.wait_for(lambda: any(
            op["id"] == job.id for _, ev in ra.frames("activity")
            for op in ev["operations"])), "the owner does not see its own job"
        assert rb.wait_for(lambda: rb.frames("activity") and rb.frames("stats"))
        time.sleep(0.3)
        seen = "\n".join(line for _, line in rb.lines)
        assert job.id not in seen and "private/repo" not in seen, \
            "B's stream leaked A's operation"
    finally:
        ra.close()
        rb.close()


def test_the_same_job_is_in_every_open_mode_stream(app, server):
    from localm import auth
    assert not auth.any_key_configured()
    job = _inject(app)
    r = Reader(server.url)
    r.start()
    try:
        assert r.wait_for(lambda: any(
            op["id"] == job.id for _, ev in r.frames("activity")
            for op in ev["operations"]))
    finally:
        r.close()


def test_the_sampler_bounds_nest(app):
    from localm.plugins.gui.routes import live
    from localm import sysstats
    assert live.FRESH_WAIT_S < live.FRESH_BUDGET_S, \
        "the forced reading's own wait must fit inside the budget that cuts it off"
    assert live.FRESH_BUDGET_S > sysstats.STATS_REPLY_BUDGET_S


def test_the_server_stops_at_once_with_a_tab_still_connected(app):
    """uvicorn leaves an in-progress response running for its whole graceful
    shutdown period (3 s here and in production) and then logs an error, so an
    endless stream has to end by itself when the server is asked to stop."""
    srv = Server(app)
    r = Reader(srv.url)
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("stats"))
        t0 = time.monotonic()
        srv.server.should_exit = True
        srv.thread.join(timeout=15)
        assert not srv.thread.is_alive(), "the server never stopped with a stream open"
        assert time.monotonic() - t0 < 2.5, "the stream held the stop for the grace period"
    finally:
        r.close(expect_drop=True)
        portmux._servers.discard(srv.server)


def test_a_stream_ends_when_its_key_is_revoked(app, server, monkeypatch):
    """Authentication at connect time alone would let a revoked key keep
    reading stats and every unowned job."""
    from localm import auth
    from localm.plugins.gui.routes import live
    monkeypatch.setattr(live, "AUTH_RECHECK_S", 0.2)
    made = auth.create_key("A", [S.MCP])
    auth.create_key("B", [S.MCP])   # a second key keeps the server out of open mode
    r = Reader(server.url, _hdr(made["key"]))
    r.start()
    try:
        assert r.wait_for(lambda: r.frames("stats"))
        assert auth.revoke_key(made["id"]) is True
        r.join(timeout=5)
        assert not r.is_alive(), "the stream stayed open after its key was revoked"
    finally:
        r.close(expect_drop=True)
