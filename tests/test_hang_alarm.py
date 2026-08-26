# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hang alarm (localm/inference/_hang_alarm.py): the staged
detect -> surface -> recover pipeline.

The loop-freeze tests use a REAL asyncio event loop with a real heartbeat task
and freeze it with a genuine synchronous time.sleep() injected onto the loop
thread, not a mocked gap. Restart and surface actions are spies: the property
under test is that the alarm DETECTS and ACTS within its thresholds, not that
os.execv works.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from localm.inference import _hang_alarm as ha


class _Spy:
    """Thread-safe event recorder for the alarm's callbacks."""

    def __init__(self):
        self.events: list[tuple[float, str, str]] = []
        self._lock = threading.Lock()

    def record(self, kind: str, text: str = "") -> None:
        with self._lock:
            self.events.append((time.monotonic(), kind, text))

    def kinds(self) -> list[str]:
        with self._lock:
            return [k for _, k, _ in self.events]

    def texts(self, kind: str) -> list[str]:
        with self._lock:
            return [t for _, k, t in self.events if k == kind]

    def wait_for(self, kind: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if kind in self.kinds():
                return True
            time.sleep(0.02)
        return False


def _make_alarm(spy: _Spy, **overrides) -> ha.HangAlarm:
    kwargs = dict(
        heartbeat_gap=lambda: None,
        inflight=None,
        probe_target=None,
        surface=lambda text: spy.record("surface", text),
        recovered=lambda: spy.record("recovered"),
        restart=lambda reason: spy.record("restart", reason),
        dump=lambda reason: spy.record("dump", reason),
        surface_after=0.3,
        restart_after=0.9,
        starvation_after=0.0,
        probe_interval=0.05,
        probe_timeout=0.25,
        probe_surface_fails=2,
        probe_restart_fails=4,
        allow_restart=True,
        poll=0.03,
    )
    kwargs.update(overrides)
    return ha.HangAlarm(**kwargs)


class _RealLoop:
    """A real asyncio loop on its own thread with a real heartbeat task,
    mirroring http_server's _hang_heartbeat_loop + localm-server thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.hb: float | None = None
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        asyncio.set_event_loop(self.loop)

        async def _beat():
            while True:
                self.hb = time.monotonic()
                await asyncio.sleep(self.interval)

        self.loop.create_task(_beat())
        self.loop.run_forever()

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 5
        while self.hb is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.hb is not None, "heartbeat never started ticking"
        return self

    def gap(self):
        return None if self.hb is None else time.monotonic() - self.hb

    def freeze(self, seconds: float):
        """A genuinely blocking synchronous sleep executed ON the loop thread,
        exactly what a rogue handler does."""
        self.loop.call_soon_threadsafe(time.sleep, seconds)

    def close(self):
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=5)
        finally:
            if not self.loop.is_closed():
                self.loop.close()


# --------------------------------------------------------------------------- #
#  Detector A: loop freeze (real loop, real blocking sleep)                    #
# --------------------------------------------------------------------------- #

def test_loop_freeze_surfaces_then_restarts_in_stage_order():
    rl = _RealLoop().start()
    spy = _Spy()
    alarm = _make_alarm(spy, heartbeat_gap=rl.gap).start()
    try:
        rl.freeze(1.6)
        assert spy.wait_for("surface", 5), spy.events
        assert spy.wait_for("restart", 5), spy.events
        kinds = spy.kinds()
        # Staged: surfaced strictly before the restart action fired.
        assert kinds.index("surface") < kinds.index("restart"), kinds
        assert "frozen" in spy.texts("surface")[0]
        # The forensic dump hook fired for the incident too.
        assert "dump" in kinds
    finally:
        alarm.stop()
        rl.close()


def test_brief_freeze_surfaces_then_recovers_and_never_restarts():
    rl = _RealLoop().start()
    spy = _Spy()
    # restart_after far beyond the injected stall: the staged pipeline must
    # surface, then clear once the loop resumes, and never fire the restart.
    alarm = _make_alarm(spy, heartbeat_gap=rl.gap, restart_after=30.0).start()
    try:
        rl.freeze(0.7)
        assert spy.wait_for("surface", 5), spy.events
        assert spy.wait_for("recovered", 5), spy.events
        assert "restart" not in spy.kinds(), spy.events
    finally:
        alarm.stop()
        rl.close()


def test_single_stale_sample_does_not_fire():
    """The sleep/resume race guard: one isolated over-threshold gap reading
    (the alarm thread waking before the loop's first post-resume tick) must
    not surface anything - action requires two consecutive bad samples."""
    spy = _Spy()
    readings = iter([9.9, 0.0, 0.0, 0.0, 0.0])
    alarm = _make_alarm(
        spy, heartbeat_gap=lambda: next(readings, 0.0)).start()
    try:
        time.sleep(0.4)
        assert "surface" not in spy.kinds(), spy.events
        assert "restart" not in spy.kinds(), spy.events
    finally:
        alarm.stop()


# --------------------------------------------------------------------------- #
#  Detector C: transport self-probe                                            #
# --------------------------------------------------------------------------- #

def test_probe_blackhole_surfaces_then_restarts():
    # A listener whose backlog accepts the TCP handshake but which never
    # reads or answers: connects succeed, responses never come - the shape
    # of a wedged accept/relay layer.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    spy = _Spy()
    alarm = _make_alarm(
        spy,
        probe_target=lambda: ("127.0.0.1", port),
        # Loop detector idle (no heartbeat readings) so only the probe acts.
        heartbeat_gap=lambda: None,
    ).start()
    try:
        assert spy.wait_for("surface", 10), spy.events
        assert "not answering" in spy.texts("surface")[0]
        assert spy.wait_for("restart", 10), spy.events
    finally:
        alarm.stop()
        srv.close()


def test_probe_healthy_server_never_fires():
    # A minimal real responder: any HTTP bytes back count as alive (portmux
    # answers plaintext-on-TLS with a 308, which must also count).
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def _respond():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            with conn:
                try:
                    conn.recv(1024)
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                except OSError:
                    pass

    t = threading.Thread(target=_respond, daemon=True)
    t.start()
    spy = _Spy()
    alarm = _make_alarm(
        spy, probe_target=lambda: ("127.0.0.1", port)).start()
    try:
        time.sleep(0.8)   # many probe cycles at 0.05s interval
        assert "surface" not in spy.kinds(), spy.events
        assert "restart" not in spy.kinds(), spy.events
    finally:
        alarm.stop()
        stop.set()
        t.join(timeout=2)
        srv.close()


# --------------------------------------------------------------------------- #
#  Detector S: request starvation                                              #
# --------------------------------------------------------------------------- #

def test_starvation_is_log_only_forensics_never_user_facing():
    """Anything user-facing must have ZERO false positives, and no threshold
    separates "requests wedged by a defect" from "legitimately slow silent work"
    with certainty - so the starvation watch never surfaces, never restarts and
    never touches the window. What it does instead is produce a forensic record:
    name the stuck requests (method + path + age), take an immediate stack
    snapshot, and take a follow-up snapshot one window later so identical frames
    prove "genuinely wedged" over "merely idle"."""
    state = {"snap": ((), 0.0)}
    spy = _Spy()
    alarm = _make_alarm(
        spy,
        inflight=lambda: state["snap"],
        starvation_after=0.3,
    ).start()
    try:
        # Two requests wedged beyond the window plus one young one, nothing
        # progressing at all.
        state["snap"] = (
            ((1.0, "GET /api/stats"), (0.9, "GET /api/models"),
             (0.05, "GET /api/activity")), 1.0)
        assert spy.wait_for("dump", 5), spy.events
        first = spy.texts("dump")[0]
        # WHICH requests: the stuck ones are named, the young one is not.
        assert "GET /api/stats" in first and "GET /api/models" in first, first
        assert "GET /api/activity" not in first, first
        # The follow-up snapshot fires one window later while still stuck.
        deadline = time.monotonic() + 5
        while (time.monotonic() < deadline
               and not any("follow-up" in t for t in spy.texts("dump"))):
            time.sleep(0.02)
        assert any("follow-up" in t for t in spy.texts("dump")), spy.events
        # Log-only: none of the user-facing channels ever moved.
        assert "surface" not in spy.kinds(), spy.events
        assert "restart" not in spy.kinds(), spy.events
        assert "recovered" not in spy.kinds(), spy.events
        # Draining closes the incident; a later, distinct wedge opens a
        # fresh record (proves the clear actually reset the state).
        state["snap"] = ((), 0.01)
        time.sleep(0.2)
        state["snap"] = (((1.0, "POST /api/imagine/comfy-launch"),), 1.0)
        deadline = time.monotonic() + 5
        while (time.monotonic() < deadline
               and not any("comfy-launch" in t for t in spy.texts("dump"))):
            time.sleep(0.02)
        assert any("comfy-launch" in t for t in spy.texts("dump")), spy.events
        assert "surface" not in spy.kinds(), spy.events
    finally:
        alarm.stop()


# --------------------------------------------------------------------------- #
#  Restart storm guard                                                         #
# --------------------------------------------------------------------------- #

def test_storm_guard_suppresses_restart_but_keeps_surfacing(monkeypatch):
    now = time.time()
    monkeypatch.setenv(
        "LOCALM_HANG_RESTART_HISTORY",
        ",".join(f"{now - off:.0f}" for off in (60, 120, 180)))
    rl = _RealLoop().start()
    spy = _Spy()
    alarm = _make_alarm(spy, heartbeat_gap=rl.gap).start()
    try:
        rl.freeze(1.6)
        assert spy.wait_for("surface", 5), spy.events
        # Give it ample time to have fired the restart if it were going to.
        time.sleep(1.0)
        assert "restart" not in spy.kinds(), spy.events
        assert any("suppressed" in t for t in spy.texts("surface")), spy.events
    finally:
        alarm.stop()
        rl.close()


def test_record_restart_appends_and_trims(monkeypatch):
    now = time.time()
    monkeypatch.setenv(
        "LOCALM_HANG_RESTART_HISTORY",
        f"{now - 10000:.0f},{now - 60:.0f}")
    ha._record_restart(now)
    hist = ha._restart_history()
    # The entry outside the storm window was trimmed; the recent one and the
    # new one remain.
    assert len(hist) == 2, hist
    assert all(now - t < ha._STORM_WINDOW_S for t in hist), hist


def test_surface_mode_never_restarts():
    rl = _RealLoop().start()
    spy = _Spy()
    alarm = _make_alarm(spy, heartbeat_gap=rl.gap, allow_restart=False).start()
    try:
        rl.freeze(1.6)
        assert spy.wait_for("surface", 5), spy.events
        time.sleep(1.0)
        assert "restart" not in spy.kinds(), spy.events
    finally:
        alarm.stop()
        rl.close()


# --------------------------------------------------------------------------- #
#  Request-progress middleware                                                 #
# --------------------------------------------------------------------------- #

def test_middleware_tracks_inflight_and_progress():
    tracker = ha.RequestProgress()
    gate = asyncio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await gate.wait()
        await send({"type": "http.response.body", "body": b"x"})

    mw = ha.RequestProgressMiddleware(app, tracker=tracker)

    async def _drive():
        async def receive():
            return {"type": "http.request"}

        sent = []

        async def send(message):
            sent.append(message["type"])

        task = asyncio.ensure_future(
            mw({"type": "http", "path": "/api/stats", "method": "GET"},
               receive, send))
        await asyncio.sleep(0.05)
        count, oldest_age, _ = tracker.snapshot()
        assert count == 1
        assert oldest_age >= 0.04
        entries, _ = tracker.observe()
        assert entries[0][1] == "GET /api/stats", entries
        gate.set()
        await task
        count, oldest_age, progress_age = tracker.snapshot()
        assert count == 0
        assert progress_age < 0.5
        return sent

    sent = asyncio.run(_drive())
    assert "http.response.body" in sent


@pytest.mark.parametrize("path", ["/health", "/whoami"])
def test_middleware_excludes_instrument_endpoints_from_tracking(path):
    """/health and /whoami are liveness/identity instruments (the alarm's own
    probe, external monitors, cross-instance discovery polls); counting them as
    progress would mask the exact starvation they exist to reveal, since /health
    keeps answering in milliseconds while the server is unusable."""
    tracker = ha.RequestProgress()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = ha.RequestProgressMiddleware(app, tracker=tracker)

    async def _drive():
        before = tracker.snapshot()
        seen_inflight = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            seen_inflight.append(tracker.snapshot()[0])

        await mw({"type": "http", "path": path}, receive, send)
        return before, seen_inflight

    before, seen_inflight = asyncio.run(_drive())
    # Never entered the books: in-flight stayed zero at every send, and the
    # progress clock was not touched by the completion.
    assert all(c == 0 for c in seen_inflight), seen_inflight
    after = tracker.snapshot()
    assert after[0] == 0
    # progress_age keeps growing across the /health request instead of being
    # reset by it (compare against the pre-request reading).
    assert after[2] >= before[2]


def test_probe_host_mapping():
    assert ha._probe_host(None) == "127.0.0.1"
    assert ha._probe_host("") == "127.0.0.1"
    assert ha._probe_host("0.0.0.0") == "127.0.0.1"
    assert ha._probe_host("::") == "::1"
    assert ha._probe_host("192.168.1.50") == "192.168.1.50"


def test_recovery_mode_parsing(monkeypatch):
    monkeypatch.delenv("LOCALM_HANG_RECOVERY", raising=False)
    assert ha.recovery_mode() == "restart"
    monkeypatch.setenv("LOCALM_HANG_RECOVERY", "surface")
    assert ha.recovery_mode() == "surface"
    monkeypatch.setenv("LOCALM_HANG_RECOVERY", "OFF")
    assert ha.recovery_mode() == "off"
    monkeypatch.setenv("LOCALM_HANG_RECOVERY", "bogus")
    assert ha.recovery_mode() == "restart"
