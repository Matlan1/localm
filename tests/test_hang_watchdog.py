# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hang-capture instrumentation: the event-loop stall watchdog and the
loopback-only GET /debug/stacks endpoint.

The watchdog is the load-bearing tool for the diagnosed intermittent server
hang: it runs OFF the event loop and dumps every thread's stack to a file when
the loop stops ticking, so a freeze that would otherwise be lost is captured.
/debug/stacks is the on-demand complement (usable while the loop is still alive).
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def test_hang_watchdog_env_semantics(monkeypatch):
    """On by default (so a tester needs no setup); 0/false/off opts out; 1/true/on
    also turns on the verbose extras."""
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
    """No stall -> no dump, and (lazy file) NO trace file created at all. A healthy
    run - the common case, since the watchdog is on by default - must leave nothing
    behind."""
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
    """End-to-end: the REAL 1s heartbeat coroutine + the off-loop watchdog must
    catch a genuine event-loop block (a synchronous call that stalls the loop -
    exactly the diagnosed hang), not just a manually-staled heartbeat."""
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
# (deliberately NOT seeded at module-import time). A request answered before
# that first tick used to report loop_lag as elapsed-since-import - a number
# that GROWS WITH WALL-CLOCK TIME regardless of what the loop is doing,
# surviving in the one window the #955/#950 fix below never exercised (a
# fresh, never-yet-ticked heartbeat). Measured live during a real model load:
# a /health check read loop_lag=13.50s, and a later request in that same
# still-cold-started run read 71.11s - see dev-notes/restart-loop-lag-
# investigation-2026-08-04.md for the full trace.
# --------------------------------------------------------------------------- #

def test_loop_lag_seconds_is_zero_before_the_first_heartbeat_tick(monkeypatch):
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    # Even with a lot of wall-clock time notionally "elapsed" (no seed to
    # measure against), the cold-start reading must stay 0.0 - never a
    # number that grows just because time.monotonic() advances.
    for now in (0.0, 1_000_000.0, 1_000_071.11):
        monkeypatch.setattr(hs.time, "monotonic", lambda now=now: now)
        assert hs._loop_lag_seconds() == 0.0


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
    heartbeat's first tick), the watchdog must go back to detecting a
    genuine stall normally - the cold-start skip must not become permanent."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", None)
    stop, thread = hs._start_hang_watchdog(
        threshold=0.1, trace_path=trace, poll=0.05)
    try:
        time.sleep(0.2)   # cold, skipped
        # The heartbeat's first tick lands, then immediately goes stale -
        # the exact shape of a real stall right after startup.
        monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic() - 1.0)
        time.sleep(0.3)
    finally:
        stop.set()
        thread.join(timeout=2)
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "LOCALM HANG WATCHDOG" in text, text


@pytest.mark.anyio
async def test_real_heartbeat_transitions_from_cold_start_to_a_real_reading(monkeypatch):
    """End-to-end with the REAL heartbeat coroutine (not a manually-set
    value): force a genuinely cold _hb_monotonic (simulating a fresh module
    import, since the module-level global otherwise persists real values
    across earlier tests in this same process), confirm loop_lag reads 0.0
    while cold, then confirm it goes back to reporting real scheduling delay
    once the coroutine's first tick lands - the cold-start handling must not
    leak into or replace the steady-state behavior #955/#950 already fixed."""
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", None)
    assert hs._loop_lag_seconds() == 0.0, "must read 0.0 while genuinely cold"

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
# GitHub #955/#950: loop_lag telemetry was not lag - it was time-since-last-
# heartbeat-tick, which saws 0..1s on a perfectly healthy loop for no reason
# other than where "now" falls in the heartbeat's own cycle. _loop_lag_seconds()
# replaces the raw gap with a real scheduling-delay figure.
# --------------------------------------------------------------------------- #

def test_loop_lag_seconds_is_real_scheduling_delay_not_time_since_tick(monkeypatch):
    """Deterministic formula check: ~0 at every point in a healthy cycle -
    including right at the next tick boundary, the worst case for the raw
    (unfixed) formula - and the real overshoot only when a tick was genuinely
    late (the loop was actually blocked)."""
    from localm.inference import http_server as hs

    monkeypatch.setattr(hs, "_hb_monotonic", 100.0)

    for now, why in (
        (100.0, "just ticked"),
        (100.0 + hs._HEARTBEAT_INTERVAL_S * 0.5, "mid-cycle"),
        (100.0 + hs._HEARTBEAT_INTERVAL_S, "exactly at the next tick boundary"),
    ):
        monkeypatch.setattr(hs.time, "monotonic", lambda now=now: now)
        assert hs._loop_lag_seconds() == 0.0, why

    # A tick that really was late (the loop was blocked) reports the actual
    # overshoot, not the inflated raw gap.
    late_by = 4.2
    monkeypatch.setattr(
        hs.time, "monotonic",
        lambda: 100.0 + hs._HEARTBEAT_INTERVAL_S + late_by)
    assert hs._loop_lag_seconds() == pytest.approx(late_by)


@pytest.mark.anyio
async def test_loop_lag_seconds_stays_at_floor_within_a_measured_safe_window():
    """End-to-end proof, driving the REAL heartbeat coroutine (not a manually
    staled _hb_monotonic): every sample taken while LESS than one full
    interval has MEASURABLY elapsed since a confirmed real tick must read
    exactly 0.0 - the pre-fix formula would show up to ~1s of fake "lag" here
    purely from sampling mid-cycle (#955/#950).

    Deliberately NOT a wall-clock threshold like `max(samples) < 0.5`: this
    box runs many concurrent sessions, and a fixed bound would eventually
    flake on real contention that delays this test's own sampling loop
    without the fix being wrong (this repo's own load-flake register is full
    of exactly that shape). Instead each sample is gated on MEASURED elapsed
    time since the last confirmed tick, so the assertion is mathematically
    guaranteed by _loop_lag_seconds' own floor(0, gap - interval) formula
    regardless of how much real scheduling jitter slows this loop down - a
    slow box just yields fewer safe samples, never a false failure."""
    from localm.inference import http_server as hs

    hb = asyncio.create_task(hs._hang_heartbeat_loop())
    try:
        # Wait for a confirmed real tick (not a stale global from an earlier
        # test / module import).
        before = hs._hb_monotonic
        wait_deadline = time.monotonic() + 3.0
        while hs._hb_monotonic == before and time.monotonic() < wait_deadline:
            await asyncio.sleep(0.01)
        assert hs._hb_monotonic != before, "heartbeat task never ticked"
        tick_at = hs._hb_monotonic

        # Only samples PROVABLY within one interval of that confirmed tick are
        # asserted on - safe_until is a hard, measured bound, not a guess.
        safe_until = tick_at + hs._HEARTBEAT_INTERVAL_S * 0.9
        checked = 0
        while time.monotonic() < safe_until:
            lag = hs._loop_lag_seconds()
            now = time.monotonic()
            if now < safe_until and hs._hb_monotonic == tick_at:
                assert lag == 0.0, (
                    f"lag={lag:.3f}s only {now - tick_at:.3f}s after a "
                    f"confirmed tick, well under the "
                    f"{hs._HEARTBEAT_INTERVAL_S}s interval")
                checked += 1
            await asyncio.sleep(0.03)
        assert checked >= 3, (
            f"only {checked} sample(s) fell within the measured-safe window - "
            "the box may be too loaded for this test to say anything; not a "
            "false pass, but worth knowing")
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


@pytest.mark.anyio
async def test_loop_lag_seconds_is_clearly_positive_after_a_real_stall():
    """The other half of the property: _loop_lag_seconds() must not just
    unconditionally read 0.0 - a genuine event-loop block (not mid-cycle
    sampling) must produce a clearly positive value. time.sleep() blocks the
    real OS thread regardless of asyncio contention, so under real box load
    this block can only run LONGER than requested, never shorter - the
    generous margin (interval + 0.8s block, asserting only > 0.3s of
    reported lag) stays true even then."""
    from localm.inference import http_server as hs

    hb = asyncio.create_task(hs._hang_heartbeat_loop())
    try:
        await asyncio.sleep(0.1)                         # let the first tick land
        block_for = hs._HEARTBEAT_INTERVAL_S + 0.8
        time.sleep(block_for)                            # BLOCK the real event loop
        lag = hs._loop_lag_seconds()
        assert lag > 0.3, (
            f"a real ~{block_for:.1f}s block reported lag={lag:.3f}s - the "
            "fix must surface a genuine stall, not just read zero "
            "unconditionally")
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


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
    # Open mode (no key configured) + loopback bind -> require_fs_host passes,
    # but the open-mode SHELL TOKEN is now also required (CodeQL 97): keyless
    # effective_fs_access returns "host" for everyone, so require_fs_host alone
    # left this fully unauthenticated. See tests/test_disclosure.py for the
    # refusal side; this asserts the diagnostic still works for the GUI shell.
    app.state.bind_host = "127.0.0.1"
    c = TestClient(app)
    r = c.get("/debug/stacks",
              headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(("pid", "loop_lag_s", "threads", "tasks")) <= set(body)
    assert isinstance(body["threads"], dict) and body["threads"]   # at least one thread


def test_debug_stacks_open_mode_needs_shell_token(app):
    # The other half of the change above: no token -> refused, where it used to
    # return every thread's stack to any local caller.
    app.state.bind_host = "127.0.0.1"
    assert TestClient(app).get("/debug/stacks").status_code in (401, 403)


def test_debug_stacks_hidden_on_network_bind(app):
    # Still 404 (not 403) with NO credential: the shell-token gate is applied
    # only on a loopback bind precisely so this stays "hidden" rather than
    # becoming "exists but needs auth".
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
