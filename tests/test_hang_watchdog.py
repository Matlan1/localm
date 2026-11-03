# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hang-capture instrumentation: the event-loop stall watchdog and the
loopback-only GET /debug/stacks endpoint.

The watchdog runs OFF the event loop and dumps every thread's stack to a file
when the loop stops ticking. /debug/stacks is the on-demand complement, usable
while the loop is still alive.
"""

import asyncio
import re
import sys
import time

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def test_hang_watchdog_env_semantics(monkeypatch):
    """On by default; 0/false/off opts out; 1/true/on also turns on the verbose
    extras."""
    from localm import debuglog

    monkeypatch.delenv("LOCALM_HANG_WATCHDOG", raising=False)
    assert debuglog.hang_watchdog_active() is True      # default ON
    assert debuglog.hang_watchdog_verbose() is False

    for off in ("0", "false", "off", "no", "OFF"):
        monkeypatch.setenv("LOCALM_HANG_WATCHDOG", off)
        assert debuglog.hang_watchdog_active() is False, off
        assert debuglog.hang_watchdog_verbose() is False, off

    for on in ("1", "true", "on", "YES"):
        monkeypatch.setenv("LOCALM_HANG_WATCHDOG", on)
        assert debuglog.hang_watchdog_active() is True, on
        assert debuglog.hang_watchdog_verbose() is True, on

    monkeypatch.delenv("LOCALM_HANG_WATCHDOG_SECS", raising=False)
    assert debuglog.hang_watchdog_threshold() == 10.0   # conservative default
    monkeypatch.setenv("LOCALM_HANG_WATCHDOG_SECS", "3")
    assert debuglog.hang_watchdog_threshold() == 3.0
    monkeypatch.setenv("LOCALM_HANG_WATCHDOG_SECS", "0.5")
    assert debuglog.hang_watchdog_threshold() == 2.0    # floored


def test_diagnostics_allowed_respects_privacy_and_toggle(monkeypatch):
    """Privacy mode writes no automatic trace unless keep_diagnostics is on; the
    log/full modes always allow it. This gates both the hang watchdog and the
    crash-restart breadcrumbs."""
    from localm.inference import http_server as hs
    monkeypatch.delenv("LOCALM_MODE", raising=False)   # config decides the mode

    def _cfg(mode, keep):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"mode": mode, "keep_diagnostics": keep})

    _cfg("privacy", False)
    assert hs._diagnostics_allowed() is False    # privacy + off -> no trace
    _cfg("privacy", True)
    assert hs._diagnostics_allowed() is True     # privacy + toggle -> allowed
    _cfg("log", False)
    assert hs._diagnostics_allowed() is True      # log mode -> allowed
    _cfg("full", False)
    assert hs._diagnostics_allowed() is True      # full mode -> allowed


def test_diagnostics_allowed_fails_safe_to_privacy(monkeypatch):
    """If the mode/config cannot be resolved, default to NO trace (privacy)."""
    from localm.inference import http_server as hs

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", _boom)
    assert hs._diagnostics_allowed() is False


def test_watchdog_dumps_stacks_on_stall(tmp_path, monkeypatch):
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    # Freeze the heartbeat far in the past so the loop looks permanently stalled;
    # the watchdog thread reads this module global every second.
    monkeypatch.setattr(hs, "_hb_monotonic", 0.0)
    stop, thread = hs._start_hang_watchdog(
        threshold=0.2, trace_path=trace, poll=0.05)
    try:
        time.sleep(0.4)     # several 0.05s polls: the watchdog observes + dumps
    finally:
        stop.set()
        thread.join(timeout=2)      # the thread flushes + closes its file on exit
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "LOCALM HANG WATCHDOG" in text, text
    assert "stalled" in text
    # faulthandler.dump_traceback (or the pure-Python fallback) writes per-thread
    # stack frames; either way a "File ..." / "Thread" / "--- thread" marker shows.
    assert ("File " in text or "Thread" in text or "--- thread" in text), text


def test_watchdog_quiet_while_loop_ticks(tmp_path, monkeypatch):
    """No stall -> no dump, and no trace file created at all: a healthy run
    leaves nothing behind."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())
    stop, thread = hs._start_hang_watchdog(
        threshold=0.3, trace_path=trace, poll=0.05)
    try:
        # Keep the heartbeat fresh across many watchdog polls (lag stays < 0.3s).
        for _ in range(8):
            monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())
            time.sleep(0.05)
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not trace.exists(), "a healthy run must not create a hang trace file"


@pytest.mark.anyio
async def test_watchdog_catches_a_real_event_loop_block(tmp_path, monkeypatch):
    """End-to-end: the REAL 1s heartbeat coroutine plus the off-loop watchdog
    catch a genuine event-loop block (a synchronous call that stalls the loop),
    not just a manually-staled heartbeat."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    hb = asyncio.create_task(hs._hang_heartbeat_loop())
    stop, thread = hs._start_hang_watchdog(
        threshold=1.0, trace_path=trace, poll=0.1)
    try:
        await asyncio.sleep(0.3)     # heartbeat ticks; lag stays low
        time.sleep(1.6)              # BLOCK the real event loop past the threshold
        await asyncio.sleep(0.3)     # resume; heartbeat catches up
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        stop.set()
        thread.join(timeout=2)
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "LOCALM HANG WATCHDOG" in text, text


# --------------------------------------------------------------------------- #
# Cold start: _hb_monotonic is None until the heartbeat task's own first tick
# (it is not seeded at module-import time). _loop_lag_seconds() returns None for
# that state, which is distinct from the 0.0 a healthy loop reports, and every
# caller must render it explicitly rather than reusing the 0.0 float path.
# --------------------------------------------------------------------------- #

def test_loop_lag_seconds_is_none_before_the_first_heartbeat_tick(monkeypatch):
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    # With a lot of wall-clock time elapsed and no seed to measure against, the
    # cold-start reading stays None - never a number that grows as
    # time.monotonic() advances, and never the 0.0 a healthy loop reports.
    for now in (0.0, 1_000_000.0, 1_000_071.11):
        monkeypatch.setattr(hs.time, "monotonic", lambda now=now: now)
        assert hs._loop_lag_seconds() is None


def test_watchdog_skips_the_check_before_the_first_heartbeat_tick(tmp_path, monkeypatch):
    """The watchdog thread must not crash (TypeError subtracting None) or
    fabricate a stall dump against a baseline that was never real."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    stop, thread = hs._start_hang_watchdog(
        threshold=0.05, trace_path=trace, poll=0.05)
    try:
        time.sleep(0.3)   # several polls' worth, all with _hb_monotonic still None
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive(), "watchdog thread died instead of skipping cleanly"
    assert not trace.exists(), "must not dump against a fabricated cold-start baseline"


def test_watchdog_resumes_normal_checking_once_ticked_after_a_cold_start(tmp_path, monkeypatch):
    """Once _hb_monotonic transitions from None to a real value (the
    heartbeat's first tick), the watchdog detects a genuine stall again: the
    cold-start skip is not permanent."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    stop, thread = hs._start_hang_watchdog(
        threshold=0.1, trace_path=trace, poll=0.05)
    try:
        time.sleep(0.2)   # cold, skipped
        # The heartbeat's first tick lands, then immediately goes stale.
        monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic() - 1.0)
        time.sleep(0.3)
    finally:
        stop.set()
        thread.join(timeout=2)
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "LOCALM HANG WATCHDOG" in text, text


@pytest.mark.anyio
async def test_real_heartbeat_transitions_from_cold_start_to_a_real_reading(monkeypatch, clean_stalls):
    """End-to-end with the REAL heartbeat coroutine, not a manually-set value:
    force a genuinely cold _hb_monotonic (the module-level global otherwise
    persists real values across earlier tests in this process), confirm
    loop_lag reads None while cold, then confirm it reports real scheduling
    delay again once the coroutine's first tick lands."""
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    assert hs._loop_lag_seconds() is None, "must read None while genuinely cold"

    hb = asyncio.create_task(hs._hang_heartbeat_loop())
    try:
        deadline = time.monotonic() + 3.0
        while hs._hb_monotonic is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert hs._hb_monotonic is not None, "heartbeat task never ticked"
        # Now a real reading, same as the steady-state tests below.
        assert hs._loop_lag_seconds() == pytest.approx(0.0, abs=0.5)
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


# --------------------------------------------------------------------------- #
# The heartbeat TASK that feeds _loop_lag_seconds() starts independently of the
# watchdog THREAD's privacy/env gate; only pytest not being in sys.modules gates
# it. The debug request log and GET /debug/stacks are reachable under a
# different gate, and both must still get a real reading.
# --------------------------------------------------------------------------- #

def _heartbeat_starts_with_watchdog_thread_disabled(tmp_path, monkeypatch, *, config_json):
    """Shared driver for both configs below: builds a real app in a throwaway
    LOCALM_HOME, temporarily lifts the ``"pytest" not in sys.modules`` guard
    for exactly this one lifespan run (restored by monkeypatch on teardown;
    TestClient's own shutdown path cancels hb_task and stops the watchdog
    thread on context exit), and returns (hb_monotonic_after_startup,
    watchdog_thread_start_call_count)."""
    from localm.inference import http_server as hs

    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    if config_json is not None:
        (home / "config.json").write_text(config_json, encoding="utf-8")
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    watchdog_calls = []
    real_start = hs._start_hang_watchdog

    def _spy(*a, **kw):
        watchdog_calls.append((a, kw))
        return real_start(*a, **kw)

    monkeypatch.setattr(hs, "_start_hang_watchdog", _spy)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)

    app = create_app(None)
    with TestClient(app):
        deadline = time.monotonic() + 3.0
        while hs._hb_monotonic is None and time.monotonic() < deadline:
            time.sleep(0.02)
        hb_after = hs._hb_monotonic
    return hb_after, len(watchdog_calls)


def test_heartbeat_starts_when_watchdog_env_disabled(tmp_path, monkeypatch):
    """Config A: LOCALM_HANG_WATCHDOG=0. The watchdog THREAD must stay off (an
    explicit opt-out is honored), but the heartbeat TASK - which the debug
    request log's loop_lag depends on, unconditionally on debug_enabled() -
    must still start."""
    monkeypatch.setenv("LOCALM_HANG_WATCHDOG", "0")
    hb_after, watchdog_calls = _heartbeat_starts_with_watchdog_thread_disabled(
        tmp_path, monkeypatch, config_json=None)
    assert hb_after is not None, (
        "the heartbeat task never started even though the ONLY thing "
        "disabled is the watchdog thread's own env gate - loop_lag would "
        "silently read a permanent, fabricated 0.00")
    assert watchdog_calls == 0, "the watchdog thread must stay off (explicit opt-out)"


def test_heartbeat_starts_in_privacy_mode_on_default_config(tmp_path, monkeypatch):
    """Config B: privacy mode, keep_diagnostics left at its default (False) -
    the "privacy mode, debug log on" combination, where operational lines stay
    on debug_enabled(). The watchdog THREAD stays off (no automatic disk trace
    in privacy mode), while the heartbeat TASK still starts so the debug log's
    loop_lag is real, not fabricated."""
    monkeypatch.delenv("LOCALM_HANG_WATCHDOG", raising=False)
    hb_after, watchdog_calls = _heartbeat_starts_with_watchdog_thread_disabled(
        tmp_path, monkeypatch, config_json='{"mode": "privacy"}')
    assert hb_after is not None, (
        "the heartbeat task never started in privacy mode on default config "
        "- loop_lag would silently read a permanent, fabricated 0.00 in "
        "exactly the config a privacy-conscious bug reporter is most likely "
        "to be running (GitHub #958)")
    assert watchdog_calls == 0, (
        "the watchdog thread must stay off in privacy mode without an "
        "explicit keep_diagnostics opt-in - it writes an automatic disk trace")


def test_lifespan_shutdown_clears_the_hang_alarm_instance(tmp_path, monkeypatch):
    """The lifespan that installs a real HangAlarm must retire it on shutdown:
    http_server._hang_alarm_instance (and _hang_dump_loop) must be None again
    once the TestClient context exits. A stopped alarm left in that global
    still honours trigger_restart(), which re-execs the whole process the next
    time switch_engine exhausts its inconclusive-probe retries."""
    from localm.inference import http_server as hs
    from localm.inference._hang_alarm import HangAlarm

    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_HANG_RECOVERY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(hs, "_hang_alarm_instance", None)
    monkeypatch.setattr(hs, "_hang_dump_loop", None)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)

    with TestClient(create_app(None)):
        installed = hs._hang_alarm_instance
    assert isinstance(installed, HangAlarm), (
        "the lifespan did not install a real HangAlarm, so this test could not "
        f"exercise the shutdown path at all: {installed!r}")
    assert hs._hang_alarm_instance is None, (
        "lifespan shutdown left the stopped HangAlarm published as the process "
        f"restart authority: {hs._hang_alarm_instance!r}")
    assert hs._hang_dump_loop is None, (
        f"lifespan shutdown left the closed loop published: {hs._hang_dump_loop!r}")


# --------------------------------------------------------------------------- #
# _loop_stall_between(start, end) reports the longest event-loop stall that
# overlapped [start, end], measured from the heartbeat ticks that ran late.
# --------------------------------------------------------------------------- #

@pytest.fixture
def clean_stalls(monkeypatch):
    from localm.inference import http_server as hs
    monkeypatch.setattr(hs, "_hb_stalls", type(hs._hb_stalls)(maxlen=512))


def test_loop_stall_between_formula(monkeypatch, clean_stalls):
    """Deterministic: a tick still overdue at `end` counts, a recorded late
    tick counts only for the part inside the interval, a healthy cycle is 0."""
    from localm.inference import http_server as hs

    iv = hs._HEARTBEAT_INTERVAL_S
    monkeypatch.setattr(hs, "_hb_monotonic", 100.0)
    assert hs._loop_stall_between(99.0, 100.0 + iv * 0.5) == 0.0
    assert hs._loop_stall_between(99.0, 100.0 + iv) == 0.0
    assert hs._loop_stall_between(99.0, 100.0 + iv + 4.2) == pytest.approx(4.2)
    assert hs._loop_stall_between(100.0 + iv + 1.0, 100.0 + iv + 4.2) == pytest.approx(3.2)

    hs._hb_stalls.append((50.0, 52.0))
    assert hs._loop_stall_between(49.0, 53.0) == pytest.approx(2.0)
    assert hs._loop_stall_between(51.0, 53.0) == pytest.approx(1.0)
    assert hs._loop_stall_between(52.5, 53.0) == 0.0
    assert hs._loop_stall_between(40.0, 45.0) == 0.0


def test_loop_stall_between_is_none_before_the_first_tick(monkeypatch, clean_stalls):
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    hs._hb_stalls.append((50.0, 52.0))
    assert hs._loop_stall_between(0.0, 1_000_000.0) is None


@pytest.mark.anyio
async def test_heartbeat_records_a_real_block_and_ignores_a_healthy_loop(clean_stalls):
    """The REAL heartbeat coroutine: an awaiting loop records nothing, a
    time.sleep() block is recorded at about its length. time.sleep blocks the OS
    thread whatever the asyncio load, so the block can only run longer."""
    from localm.inference import http_server as hs

    hb = asyncio.create_task(hs._hang_heartbeat_loop())
    try:
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        await asyncio.sleep(0.5)
        assert (hs._loop_stall_between(t0, time.monotonic()) or 0.0) < 0.3
        t1 = time.monotonic()
        time.sleep(0.6)
        await asyncio.sleep(0.2)
        stall = hs._loop_stall_between(t1, time.monotonic())
        assert 0.6 - hs._HEARTBEAT_INTERVAL_S - 0.02 <= stall <= 1.5, stall
        assert hs._loop_lag_seconds() >= 0.45
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


def _request_loop_lags(tmp_path, monkeypatch, caplog, paths):
    """Serve `paths` from a real app with the real heartbeat running (pytest
    hidden from the lifespan gate) and return {path: logged loop_lag seconds}."""
    import localm.config as cfg
    from localm.inference import http_server as hs

    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    monkeypatch.setattr(hs, "_hb_stalls", type(hs._hb_stalls)(maxlen=512))
    monkeypatch.delitem(sys.modules, "pytest", raising=False)

    app = create_app(None, api_landing=True)

    async def _block(secs: float):
        time.sleep(secs)
        return {"ok": True}

    async def _wait(secs: float):
        await asyncio.sleep(secs)
        return {"ok": True}

    app.add_api_route("/_t/block/{secs}", _block)
    app.add_api_route("/_t/wait/{secs}", _wait)
    out = {}
    with caplog.at_level("DEBUG", logger="localm"):
        with TestClient(app) as c:
            deadline = time.monotonic() + 3.0
            while hs._hb_monotonic is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert hs._hb_monotonic is not None, "heartbeat never ticked"
            time.sleep(0.3)
            for path in paths:
                caplog.clear()
                r = c.get(path, headers={"Authorization": f"Bearer {app.state.shell_token}"})
                assert r.status_code == 200, (path, r.status_code, r.text)
                lines = [rec.getMessage() for rec in caplog.records
                         if f"GET {path} " in rec.getMessage()]
                assert lines, (path, [rec.getMessage() for rec in caplog.records])
                m = re.search(r"loop_lag=(\d+\.\d{2})s", lines[-1])
                assert m, lines[-1]
                out[path] = float(m.group(1))
    return out


def test_request_log_reports_the_stall_that_overlapped_the_request(tmp_path, monkeypatch, caplog):
    lags = _request_loop_lags(
        tmp_path, monkeypatch, caplog,
        ["/_t/block/0.5", "/_t/block/2", "/_t/wait/2"])
    interval = 0.1
    assert 0.5 - interval - 0.02 <= lags["/_t/block/0.5"] <= 1.5, lags
    assert 2.0 - interval - 0.02 <= lags["/_t/block/2"] <= 3.0, lags
    assert lags["/_t/wait/2"] <= 0.3, lags


# --------------------------------------------------------------------------- #
# _loop_lag_seconds() returns None (not 0.0) for no-reading-yet; these prove the
# two real consumers render that explicitly rather than crashing on a format
# spec expecting a float or reusing the healthy 0.00s string. The hang watchdog
# thread reads _hb_monotonic directly and is not a consumer.
# --------------------------------------------------------------------------- #

def test_debug_request_log_renders_cold_start_as_na_not_zero(caplog, monkeypatch):
    """A request served before the heartbeat's first tick renders "n/a", not
    "loop_lag=0.00s", which is what a genuinely healthy loop reports. The
    numeric format for this case is asserted ABSENT, not just "n/a" present."""
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    from localm.inference import http_server as hs
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    app = create_app(None, api_landing=True)
    with caplog.at_level("DEBUG", logger="localm"):
        with TestClient(app) as c:
            r = c.get("/health")
    # create_app(None, ...) has no engine, so /health itself 503s. The logging
    # middleware wraps every response regardless of status, so the log line is
    # captured either way.
    assert r.status_code == 503
    lines = [rec.getMessage() for rec in caplog.records if "loop_lag=" in rec.getMessage()]
    assert lines, "no request log line carrying loop_lag was captured"
    assert any("loop_lag=n/a" in ln for ln in lines), lines
    assert not any("loop_lag=0.00s" in ln for ln in lines), lines


def test_debug_request_log_renders_a_real_reading_as_before(caplog, monkeypatch):
    """The other half: once there IS a real reading, the log format is the
    numeric one (%.2fs), never "n/a". _hb_monotonic is stamped right before
    the request, inside the TestClient context: pytest stays in
    sys.modules for this test, so the real heartbeat task (gated on
    "pytest" not in sys.modules) never starts and nothing else can advance
    _hb_monotonic between the stamp and the request - the measured gap is
    therefore just the request's own handling time, not app-creation plus
    lifespan-startup time."""
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    from localm.inference import http_server as hs
    app = create_app(None, api_landing=True)
    with caplog.at_level("DEBUG", logger="localm"):
        with TestClient(app) as c:
            monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())   # "just ticked"
            r = c.get("/health")
    assert r.status_code == 503   # no engine loaded - see the sibling test above
    lines = [rec.getMessage() for rec in caplog.records if "loop_lag=" in rec.getMessage()]
    assert lines, "no request log line carrying loop_lag was captured"
    assert any(re.search(r"loop_lag=\d+\.\d{2}s", ln) for ln in lines), lines
    assert not any("loop_lag=n/a" in ln for ln in lines), lines


def test_debug_stacks_renders_cold_start_as_json_null_not_zero(app, monkeypatch):
    """/debug/stacks' loop_lag_s field is JSON null for a cold-start reading,
    never 0.0 - the same "no reading yet" vs "healthy" distinction the
    debug-log site makes, checked here for the OTHER consumer.

    Uses the `app` fixture defined below; pytest resolves fixtures by name
    across the whole module, so the physical order does not matter."""
    from localm.inference import http_server as hs
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    app.state.bind_host = "127.0.0.1"
    with TestClient(app) as c:
        r = c.get("/debug/stacks",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loop_lag_s"] is None, body


def test_debug_stacks_renders_a_real_reading_as_before(app, monkeypatch, clean_stalls):
    from localm.inference import http_server as hs
    app.state.bind_host = "127.0.0.1"
    with TestClient(app) as c:
        monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())
        r = c.get("/debug/stacks",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["loop_lag_s"], float), body
    assert body["loop_lag_s"] == pytest.approx(0.0, abs=0.5)


@pytest.fixture
def app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return create_app(None, api_landing=True)


def test_debug_stacks_open_mode_loopback(app):
    # Open mode (no key configured) + loopback bind: require_fs_host passes, and
    # the open-mode SHELL TOKEN is required as well.
    app.state.bind_host = "127.0.0.1"
    c = TestClient(app)
    r = c.get("/debug/stacks",
              headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(("pid", "loop_lag_s", "threads", "tasks")) <= set(body)
    assert isinstance(body["threads"], dict) and body["threads"]   # at least one thread


def test_debug_stacks_open_mode_needs_shell_token(app):
    # No token: refused.
    app.state.bind_host = "127.0.0.1"
    assert TestClient(app).get("/debug/stacks").status_code in (401, 403)


def test_debug_stacks_hidden_on_network_bind(app):
    # Still 404 (not 403) with NO credential: the shell-token gate is applied
    # only on a loopback bind, so the route stays hidden rather than
    # exists-but-needs-auth.
    app.state.insecure_bind = True
    for host in ("0.0.0.0", "192.168.1.50", "10.0.0.7"):
        app.state.bind_host = host
        r = TestClient(app).get("/debug/stacks")
        assert r.status_code == 404, f"{host} -> {r.status_code}"
        with_token = TestClient(app).get(
            "/debug/stacks",
            headers={"Authorization": f"Bearer {app.state.shell_token}"})
        assert with_token.status_code == 404, f"{host} -> {with_token.status_code}"


def test_debug_stacks_requires_fs_host_key(tmp_path, monkeypatch):
    # With a key configured, an unauthenticated caller is refused; the owner key
    # (fs_access=host) is allowed.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    app = create_app(None, api_landing=True)
    app.state.bind_host = "127.0.0.1"
    c = TestClient(app)
    assert c.get("/debug/stacks").status_code in (401, 403)     # no credential
    r = c.get("/debug/stacks", headers={"Authorization": "Bearer ownersecret"})
    assert r.status_code == 200, r.text
    assert "threads" in r.json()
