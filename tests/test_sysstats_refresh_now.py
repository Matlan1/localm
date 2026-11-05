# SPDX-License-Identifier: AGPL-3.0-or-later
"""sysstats.refresh_vram_now / fresh_stats: a VRAM reading taken NOW.

The routine reading is refreshed at most every _VRAM_REFRESH_INTERVAL_S, so a
meter driven by it shows a freed model's VRAM up to ten seconds late. The forced
reading is what the live feed takes when something changed.
"""

from __future__ import annotations

import threading
import time

import pytest

from localm import sysstats

GIB = 1024 ** 3


def reading(used_gib):
    return {"vram": {"total": 16 * GIB, "used": int(used_gib * GIB),
                     "percent": round(used_gib / 16 * 100, 1)}}


def shown():
    return sysstats._vram_last["vram"]["used"] / GIB


@pytest.fixture
def gpu(monkeypatch):
    """A probe that reports ``truth`` and counts its runs."""
    state = types_ns(truth=10.0, runs=0)

    def fake_refresh():
        state.runs += 1
        return reading(state.truth), None

    monkeypatch.setattr(sysstats, "_refresh_vram", fake_refresh)
    monkeypatch.setattr(sysstats, "_cpu_ram", lambda: {})
    monkeypatch.setattr(sysstats, "_gpu_util", lambda: {})
    return state


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def first_reading():
    sysstats._vram()
    assert sysstats._vram_ready.wait(5), "the first probe never landed"


def test_the_routine_reading_keeps_the_old_value_inside_the_throttle(gpu):
    first_reading()
    assert shown() == 10.0
    gpu.truth = 2.0
    sysstats._vram()
    time.sleep(0.1)
    assert shown() == 10.0 and gpu.runs == 1, \
        "the throttle this feature works around is not in force"


def test_refresh_now_takes_a_new_reading_inside_the_throttle(gpu):
    first_reading()
    gpu.truth = 2.0
    assert sysstats.refresh_vram_now(5.0) is True
    assert shown() == 2.0
    assert gpu.runs == 2


def test_a_probe_already_in_flight_does_not_count_as_fresh(monkeypatch):
    """It began before the change that prompted the refresh, so it may carry the
    old value; a second probe must be started once it lands."""
    release = threading.Event()
    started = threading.Event()
    runs = []

    def probe():
        runs.append(1)
        n = len(runs)
        if n == 1:
            started.set()
            release.wait(5)
            return reading(10.0), None
        return reading(2.0), None

    monkeypatch.setattr(sysstats, "_refresh_vram", probe)
    sysstats._vram()
    assert started.wait(5)
    out = {}
    t = threading.Thread(target=lambda: out.update(ok=sysstats.refresh_vram_now(5.0)))
    t.start()
    time.sleep(0.1)
    assert t.is_alive(), "returned on the strength of a probe that began before the call"
    release.set()
    t.join(5)
    assert out == {"ok": True}
    assert len(runs) == 2
    assert shown() == 2.0


def test_a_probe_that_does_not_land_in_time_returns_false_and_keeps_the_old_reading(
        monkeypatch):
    gate = threading.Event()
    n = {"i": 0}

    def probe():
        n["i"] += 1
        if n["i"] == 1:
            return reading(10.0), None
        gate.wait(5)
        return reading(2.0), None

    monkeypatch.setattr(sysstats, "_refresh_vram", probe)
    first_reading()
    t0 = time.monotonic()
    assert sysstats.refresh_vram_now(0.2) is False
    assert time.monotonic() - t0 < 2.0
    assert shown() == 10.0, "a reading that has not landed replaced the last one"
    gate.set()


def test_fresh_stats_reports_the_forced_reading(gpu):
    first_reading()
    gpu.truth = 3.0
    assert sysstats.fresh_stats(timeout=5.0)["vram"]["used"] == 3 * GIB


def test_a_probe_that_began_in_the_same_clock_tick_as_the_call_is_not_counted(gpu, monkeypatch):
    """The Windows monotonic clock ticks about every 15 ms, so a probe started
    just before the call can carry the same timestamp as the call. Which probe
    began first is decided by counting probes, never by comparing times."""
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    first_reading()
    gpu.truth = 2.0
    assert sysstats.refresh_vram_now(5.0) is True
    assert shown() == 2.0
    assert gpu.runs == 2


def test_a_reset_forgets_which_probe_made_the_reading(gpu):
    first_reading()
    assert sysstats._vram_last_claim > 0
    sysstats._reset_vram_probe_cache()
    assert sysstats._vram_last_claim == 0
    assert sysstats._vram_probe_claim == 0
