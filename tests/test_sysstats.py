# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.sysstats._vram() - the GUI hardware-monitor's VRAM line.

Every VRAM-reading function here is multi/split-GPU aware, including this
status-bar widget."""

import sys
import threading

import pytest
import time
import types
from collections import namedtuple
from unittest.mock import patch

from localm import sysstats
from localm.discover import FREE_SCOPE_DEVICE, GPU_PROBE_OK
from localm.sysstats import _clamped_field_deltas, _CpuMeter, _cpu_ram, _vram


GB = 1024 ** 3

# Fabricated psutil.cpu_times() ntuples. The meter only reads sum(), .idle, and
# (where present) .iowait/.guest/.guest_nice, so these two shapes exercise a
# Windows box (idle only) and a Linux box (iowait + guest fields).
WinTimes = namedtuple("WinTimes", ["user", "system", "idle", "interrupt", "dpc"])
LinuxTimes = namedtuple(
    "LinuxTimes",
    ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal",
     "guest", "guest_nice"])


def _linux(user=0, nice=0, system=0, idle=0, iowait=0, irq=0, softirq=0,
           steal=0, guest=0, guest_nice=0):
    return LinuxTimes(user, nice, system, idle, iowait, irq, softirq, steal,
                      guest, guest_nice)


def _status_aware(value, status=GPU_PROBE_OK):
    """A faithful double of vram_info()/vram_capacity()'s real two-shape
    contract: the bare value by default, ``(value, status)`` when
    return_status=True - mirrors tests/test_vram_reading_honesty.py's
    _list_gpus_double. A plain return_value cannot express this."""
    def _inner(*args, return_status=False, **kwargs):
        return (value, status) if return_status else value
    return _inner


# --- VRAM capacity/used reading ------------------------------------------- #
# _vram() always returns immediately with the last completed reading (or {}
# before the first one lands); the actual discover.vram_capacity() call runs on
# its own single-flighted background thread, throttled to
# _VRAM_REFRESH_INTERVAL_S. Every test below that exercises a freshly-patched
# probe must WAIT for that background thread to land before reading the result,
# the same idiom the _gpu_util tests further down use.

def _reset_vram_cache(monkeypatch):
    monkeypatch.setattr(sysstats, "_vram_last", None)
    monkeypatch.setattr(sysstats, "_vram_last_at", None)
    monkeypatch.setattr(sysstats, "_vram_inflight", False)
    monkeypatch.setattr(sysstats, "_vram_ready", threading.Event())


def _wait_for_vram_cache(timeout=2.0):
    """Poll until the background probe has landed at least once (_vram_last
    moves off its None "never asked yet" sentinel)."""
    deadline = time.monotonic() + timeout
    while sysstats._vram_last is None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_reports_single_gpu_when_no_split_configured(monkeypatch):
    _reset_vram_cache(monkeypatch)
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}
    with patch("localm.discover.vram_info", side_effect=_status_aware(info)):
        assert _vram() == {}, "probe just started in the background, nothing cached yet"
        _wait_for_vram_cache()
        out = _vram()
    assert out == {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}


def test_reports_combined_capacity_with_a_configured_split(monkeypatch):
    """With a configured 2-GPU split, the status bar must
    show the COMBINED total/used, not just the single main GPU's - it now
    goes through discover.vram_capacity(), not vram_info() directly."""
    _reset_vram_cache(monkeypatch)
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    gpus = [
        {"index": 0, "name": "A", "total": 16 * GB, "free": 4 * GB,
         "free_scope": FREE_SCOPE_DEVICE},
        {"index": 1, "name": "B", "total": 8 * GB, "free": 8 * GB,
         "free_scope": FREE_SCOPE_DEVICE},
    ]
    with patch("localm.discover.list_gpus", side_effect=_status_aware(gpus)), \
         patch("localm.config.load_config",
               return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
        assert _vram() == {}
        _wait_for_vram_cache()
        out = _vram()
    assert out["vram"]["total"] == 24 * GB       # 16+8 combined, not 16 alone
    assert out["vram"]["used"] == 12 * GB        # (16-4)+(8-8) combined
    assert out["vram"]["percent"] == 50.0


def test_empty_when_unmeasurable(monkeypatch):
    _reset_vram_cache(monkeypatch)
    with patch("localm.discover.vram_info", side_effect=_status_aware({})):
        assert _vram() == {}
        _wait_for_vram_cache()
        assert _vram() == {}
    # A CONFIRMED empty reading ({}) must still be distinguishable, in the
    # cache itself, from "never asked yet" (None) - see _vram_probe.
    assert sysstats._vram_last == {}


def test_percent_omitted_when_free_unknown(monkeypatch):
    """The registry-fallback tier reports total only (no per-process free
    reading available), so 'used'/'percent' must be omitted, not fabricated
    as 0% used."""
    _reset_vram_cache(monkeypatch)
    with patch("localm.discover.vram_info",
               side_effect=_status_aware({"total": 16 * GB})):
        assert _vram() == {}
        _wait_for_vram_cache()
        out = _vram()
    assert out == {"vram": {"total": 16 * GB}}


def test_exception_from_discover_is_swallowed_not_raised(monkeypatch):
    """_vram() must never raise - a probe failure just omits the section
    (matches the module's own documented 'NEVER raises' contract). And the
    failure must NOT be cached as a CONFIRMED empty reading: _vram_last must
    stay None ("could not look"), never collapse to {} ("looked, found
    nothing") - see _vram_probe's docstring."""
    _reset_vram_cache(monkeypatch)
    with patch("localm.discover.vram_capacity", side_effect=RuntimeError("boom")):
        assert _vram() == {}
        deadline = time.monotonic() + 2
        while sysstats._vram_inflight and time.monotonic() < deadline:
            time.sleep(0.01)
    assert sysstats._vram_last is None, (
        "a failed probe attempt was cached as a confirmed-empty reading")


def test_vram_probe_never_blocks_the_polling_thread(monkeypatch):
    """A slow/wedged torch probe must never park the calling (poll) thread,
    and concurrent polls while one probe is in flight must not start a
    second vram_capacity() call (single-flight)."""
    _reset_vram_cache(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}

    def _hanging_vram_capacity(*args, return_status=False, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(5)   # simulate a slow/cold-init/wedged driver
        return (info, GPU_PROBE_OK) if return_status else info

    with patch("localm.discover.vram_capacity", side_effect=_hanging_vram_capacity):
        t0 = time.monotonic()
        first = _vram()
        elapsed = time.monotonic() - t0

        assert first == {}, "no reading has ever landed yet -> omitted, not fabricated"
        assert elapsed < 0.5, (
            f"_vram() blocked the calling thread for {elapsed:.2f}s on a "
            "hanging vram_capacity() call - the exact regression this fixes")
        assert entered.wait(2), "background probe never started"

        try:
            for _ in range(5):
                t0 = time.monotonic()
                out = _vram()
                assert time.monotonic() - t0 < 0.5
                assert out == {}
            assert len(calls) == 1, (
                f"expected exactly one in-flight probe, vram_capacity was "
                f"invoked {len(calls)} times - polls are stacking again")
        finally:
            release.set()

        _wait_for_vram_cache()
        assert _vram() == {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}


def test_vram_probe_invoked_once_across_n_stats_calls_within_cache_window(monkeypatch):
    """Across many stats calls inside one cache window, the expensive
    vram_capacity() probe must fire exactly ONCE. Asserted from OUTSIDE via a
    plain call-count list, never by raising inside the code under test:
    defensive code in this path catches broadly and would swallow an in-body
    assertion."""
    _reset_vram_cache(monkeypatch)
    calls = []
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}

    def _counting_vram_capacity(*args, return_status=False, **kwargs):
        calls.append(1)
        return (info, GPU_PROBE_OK) if return_status else info

    with patch("localm.discover.vram_capacity", side_effect=_counting_vram_capacity):
        _vram()
        _wait_for_vram_cache()
        expected = {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}
        for _ in range(10):
            assert _vram() == expected
    assert len(calls) == 1, (
        f"expected exactly one probe across 10 stats calls within the cache "
        f"window, vram_capacity was invoked {len(calls)} times - every stats "
        f"poll is spawning its own probe again")


def test_vram_probe_failure_does_not_overwrite_a_cached_good_reading(monkeypatch):
    """A probe failure must not be cached as a successful empty result. Once a
    good reading is cached, a LATER probe attempt that raises must leave the
    cache exactly as it was: "could not look" and "nothing there" need
    different handling."""
    _reset_vram_cache(monkeypatch)
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}
    with patch("localm.discover.vram_info", side_effect=_status_aware(info)):
        _vram()
        _wait_for_vram_cache()
    good = {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}
    assert sysstats._vram_last == good

    # Force the cache stale so the NEXT _vram() call starts a fresh probe,
    # and make THAT probe fail outright.
    monkeypatch.setattr(sysstats, "_vram_last_at", 0.0)
    with patch("localm.discover.vram_capacity", side_effect=RuntimeError("boom")):
        assert _vram() == good, "must keep serving the last-known-good reading"
        deadline = time.monotonic() + 2
        while sysstats._vram_inflight and time.monotonic() < deadline:
            time.sleep(0.01)

    assert sysstats._vram_last == good, (
        "a failed probe attempt overwrote a previously-good cached reading "
        "with an empty one")


def test_wait_first_blocks_until_the_probe_lands(monkeypatch):
    """A ONE-SHOT caller (wait_first=True, what the MCP system_stats tool
    passes) never gets a second poll to pick up a reading that lands later,
    so it must actually BLOCK for a real reading on a cold first call rather
    than getting the {} a repeating-poll caller (the GUI) correctly accepts."""
    _reset_vram_cache(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}

    def _slow_vram_capacity(*args, return_status=False, **kwargs):
        entered.set()
        release.wait(5)
        return (info, GPU_PROBE_OK) if return_status else info

    with patch("localm.discover.vram_capacity", side_effect=_slow_vram_capacity):
        result = {}
        caller_returned = threading.Event()

        def _call():
            result["out"] = _vram(wait_first=True)
            caller_returned.set()

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        try:
            assert entered.wait(2), "background probe never started"
            assert not caller_returned.wait(0.3), (
                "_vram(wait_first=True) returned before the probe landed - "
                "it must block for a real reading, not report {} on a cold "
                "first call the way the non-waiting default does")
        finally:
            release.set()
        assert caller_returned.wait(5), "caller never returned after release"

    assert result["out"] == {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}


def test_wait_first_gives_up_at_the_probe_deadline_not_forever(monkeypatch):
    """A probe that never lands (a genuinely wedged driver) must not hang
    wait_first indefinitely - it gives up at the bounded probe deadline and
    reports the honest {}, same as every other omission in this module."""
    _reset_vram_cache(monkeypatch)
    monkeypatch.setattr("localm.discover._GPU_PROBE_DEADLINE", 0.1)
    release = threading.Event()

    def _hanging(*args, return_status=False, **kwargs):
        release.wait(5)
        return ({}, GPU_PROBE_OK) if return_status else {}

    try:
        with patch("localm.discover.vram_capacity", side_effect=_hanging):
            t0 = time.monotonic()
            out = _vram(wait_first=True)
            elapsed = time.monotonic() - t0
    finally:
        release.set()

    assert out == {}
    # BOTH bounds matter: a no-op that ignored wait_first and returned {}
    # instantly would also satisfy "elapsed < 2.0". The lower bound proves it
    # genuinely blocked close to the patched deadline (0.1s plus the 1.0s margin
    # _vram adds) before giving up.
    assert elapsed > 0.9, (
        f"wait_first returned after only {elapsed:.2f}s - it must actually "
        f"wait close to the probe deadline before giving up, not bail out "
        f"immediately (that would be indistinguishable from wait_first "
        f"being ignored entirely)")
    assert elapsed < 2.0, (
        f"wait_first blocked for {elapsed:.2f}s - it must give up at the "
        f"probe deadline (0.1s here, plus margin), not hang indefinitely "
        f"on a wedged driver")


def test_vram_never_raises_and_unlatches_when_thread_creation_fails(monkeypatch):
    """If the OS cannot spawn the probe thread at all (e.g. thread exhaustion),
    _vram() must still never raise AND must reset the in-flight guard so a
    later call can retry - otherwise a single failed spawn would wedge every
    future poll into believing a probe is permanently running. Mirrors
    _gpu_util's handling of the same failure below."""
    _reset_vram_cache(monkeypatch)

    def _broken_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _broken_start)

    result = _vram()   # must not raise

    assert result == {}
    assert sysstats._vram_inflight is False, (
        "a failed thread spawn left _vram_inflight stuck True - no later "
        "call could ever retry")


# --- CPU-utilisation meter ------------------------------------------------ #
# psutil.cpu_percent(interval=None) reports a fabricated first reading (0 on an
# idle box, up to 100 on a busy-starting one) because it has no prior sample.
# _CpuMeter keeps its OWN previous snapshot and derives the percent over a real
# window, reporting the first (baseline-less) reading as None so the caller omits
# the CPU field.

def test_first_reading_is_none_no_baseline():
    """The very first poll has no prior snapshot, so it must return None (caller
    omits CPU this once) - never a fabricated 0 or 100."""
    m = _CpuMeter()
    assert m.percent(WinTimes(0, 0, 0, 0, 0), now=1000.0) is None


def test_second_reading_is_the_real_percent_over_the_window():
    """With a baseline, the percent is busy_delta/total_delta over the interval:
    30 busy ticks out of 100 total -> 30%."""
    m = _CpuMeter()
    assert m.percent(WinTimes(0, 0, 0, 0, 0), now=1000.0) is None
    # busy = total - idle = 100 - 70 = 30; total_delta 100, busy_delta 30 -> 30%.
    assert m.percent(WinTimes(30, 0, 70, 0, 0), now=1001.0) == 30.0


def test_iowait_counts_as_idle_not_busy():
    """iowait is a wait, not work: it must be folded into idle (matches psutil),
    so user=40 with iowait=20 and idle=40 over a 100-tick window reads 40%."""
    m = _CpuMeter()
    assert m.percent(_linux(), now=0.0) is None
    assert m.percent(_linux(user=40, idle=40, iowait=20), now=1.0) == 40.0


def test_guest_time_is_not_double_counted():
    """On Linux guest time is already inside user/nice, so the total must subtract
    it (matches psutil._cpu_tot_time). user=50 (all guest) + idle=50: total folds
    to 100, busy 50 -> 50%, not the 33% a naive sum(150) would give."""
    m = _CpuMeter()
    assert m.percent(_linux(), now=0.0) is None
    assert m.percent(_linux(user=50, idle=50, guest=50), now=1.0) == 50.0


def test_window_shorter_than_min_interval_reuses_last_and_keeps_baseline():
    """A poll closer than _MIN_INTERVAL is too short a window to trust: it returns
    the last value AND does not advance the baseline, so the next real poll still
    measures from the older sample."""
    m = _CpuMeter()
    assert m.percent(WinTimes(0, 0, 0, 0, 0), now=101.0) is None      # baseline
    assert m.percent(WinTimes(30, 0, 70, 0, 0), now=102.0) == 30.0    # 1s window
    # 0.05s later: too soon -> reuse 30.0, do NOT record this snapshot as baseline.
    assert m.percent(WinTimes(90, 0, 110, 0, 0), now=102.05) == 30.0
    # Next real poll measures from the 102.0 baseline (busy 30 / total 100), not
    # the skipped 102.05 one: busy_delta 80 / total_delta 200 -> 40%.
    assert m.percent(WinTimes(110, 0, 190, 0, 0), now=103.0) == 40.0


def test_percent_clamped_to_100():
    """A busy_delta exceeding total_delta (should not happen physically, but guard
    it) clamps to 100 rather than exceeding it."""
    m = _CpuMeter()
    assert m.percent(WinTimes(0, 0, 100, 0, 0), now=0.0) is None
    # busy_delta 20 / total_delta 10 -> 200% -> clamped to 100.
    assert m.percent(WinTimes(20, 0, 90, 0, 0), now=1.0) == 100.0


def test_percent_never_negative():
    """A negative busy_delta (unphysical, but guard it) floors at 0, not below."""
    m = _CpuMeter()
    assert m.percent(WinTimes(50, 0, 50, 0, 0), now=0.0) is None
    assert m.percent(WinTimes(40, 0, 160, 0, 0), now=1.0) == 0.0


def test_regressing_field_is_clamped_not_left_to_corrupt_the_percent():
    """A field that goes BACKWARD between two samples (a documented Windows/Linux
    kernel-counter quirk, e.g. Windows `interrupt`)
    must be trimmed to a zero delta for that field alone, exactly like psutil's own
    _cpu_times_deltas - not left to pull down the aggregate total/busy computed by
    subtracting whole-snapshot sums. WinTimes = (user, system, idle, interrupt, dpc).
    interrupt regresses 50 -> 30 (delta -20) while every other field advances."""
    m = _CpuMeter()
    assert m.percent(WinTimes(100, 100, 1000, 50, 50), now=0.0) is None
    # Clamped per-field deltas: user +0, system +40, idle +60, interrupt clamped
    # to 0 (was -20), dpc +0 -> total_delta=100, busy_delta (total-idle)=40 ->
    # 40%.
    assert m.percent(WinTimes(100, 140, 1060, 30, 50), now=1.0) == 40.0


def test_clamped_field_deltas_matches_psutil_behavior_on_a_monotonic_sample():
    """Sanity check on the ordinary (non-regressing) case: every field simply
    advances, so the clamp is a no-op and the delta is the plain subtraction."""
    d = _clamped_field_deltas(WinTimes(10, 20, 30, 5, 5), WinTimes(15, 25, 45, 10, 5))
    assert d == WinTimes(5, 5, 15, 5, 0)


def test_zero_total_delta_reuses_last_not_divide_by_zero():
    """Identical cpu_times a full second later (no ticks advanced) must keep the
    last value, never divide by zero or fabricate a 0."""
    m = _CpuMeter()
    assert m.percent(WinTimes(0, 0, 0, 0, 0), now=0.0) is None
    assert m.percent(WinTimes(30, 0, 70, 0, 0), now=1.0) == 30.0
    assert m.percent(WinTimes(30, 0, 70, 0, 0), now=2.0) == 30.0   # no advance


class _StubMeter:
    """Feeds _cpu_ram a scripted sequence of meter readings (None then a number)
    so the wiring can be checked without a real psutil or a wall-clock delay."""
    def __init__(self, seq):
        self._seq, self._i = list(seq), 0

    def percent(self, times, now):
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


def _fake_psutil():
    return types.SimpleNamespace(
        cpu_times=lambda: WinTimes(0, 0, 0, 0, 0),
        virtual_memory=lambda: types.SimpleNamespace(
            total=8 * GB, available=4 * GB, percent=50.0),
    )


def test_cpu_ram_omits_cpu_until_baseline_then_includes_it(monkeypatch):
    """_cpu_ram must drop the CPU field while the meter has no baseline (None) and
    include it once a real percent is available - RAM is unaffected throughout."""
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    monkeypatch.setattr(sysstats, "_cpu_meter", _StubMeter([None, 42.0]))

    first = _cpu_ram()
    assert "cpu" not in first                       # no baseline -> omitted, not faked
    assert first["ram"]["percent"] == 50.0          # RAM still reported

    second = _cpu_ram()
    assert second["cpu"] == {"percent": 42.0}       # real reading now shown
    assert second["ram"]["percent"] == 50.0


def test_cpu_ram_returns_empty_without_psutil(monkeypatch):
    """No psutil (the [monitor] extra absent) -> the whole CPU/RAM section is
    simply absent, never an exception."""
    monkeypatch.setitem(sys.modules, "psutil", None)   # import psutil -> raises
    assert _cpu_ram() == {}


# --- GPU-utilisation probe ------------------------------------------------ #
# _gpu_util() always returns immediately with the last completed reading (or {}
# before the first one lands); the nvidia-smi subprocess runs on its own
# single-flighted background thread, matching the CPU%/VRAM "omit rather than
# fabricate" convention used elsewhere in this module.

def _reset_gpu_util_cache(monkeypatch):
    monkeypatch.setattr(sysstats, "_gpu_util_last", None)
    monkeypatch.setattr(sysstats, "_gpu_util_last_at", None)
    monkeypatch.setattr(sysstats, "_gpu_util_inflight", False)


def test_gpu_util_probe_never_blocks_the_polling_thread(monkeypatch):
    """A hanging/slow nvidia-smi must never park the calling (poll) thread,
    and concurrent polls while one probe is in flight must not spawn a second
    nvidia-smi (single-flight)."""
    _reset_gpu_util_cache(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    class _Proc:
        returncode = 0
        stdout = "42\n"

    def _hanging_run(*args, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(5)   # simulate a slow/wedged nvidia-smi
        return _Proc()

    monkeypatch.setattr("subprocess.run", _hanging_run)

    t0 = time.monotonic()
    first = sysstats._gpu_util()
    elapsed = time.monotonic() - t0

    assert first == {}, "no reading has ever landed yet -> omitted, not fabricated"
    assert elapsed < 0.5, (
        f"_gpu_util() blocked the calling thread for {elapsed:.2f}s on a "
        "hanging nvidia-smi - the old per-poll inline subprocess.run regression")
    assert entered.wait(2), "background probe never started"

    try:
        # Further polls while the probe is still hanging must ALSO return
        # instantly, and must NOT start a second concurrent nvidia-smi.
        for _ in range(5):
            t0 = time.monotonic()
            out = sysstats._gpu_util()
            assert time.monotonic() - t0 < 0.5
            assert out == {}
        assert len(calls) == 1, (
            f"expected exactly one in-flight probe, subprocess.run was "
            f"invoked {len(calls)} times - polls are stacking again")
    finally:
        release.set()

    deadline = time.monotonic() + 2
    while sysstats._gpu_util_last is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert sysstats._gpu_util() == {"gpu": {"percent": 42.0}}


def test_gpu_util_reports_percent_once_probe_lands(monkeypatch):
    _reset_gpu_util_cache(monkeypatch)

    class _Proc:
        returncode = 0
        stdout = "17.5\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())

    assert sysstats._gpu_util() == {}   # first call: probe just started, no reading yet

    deadline = time.monotonic() + 2
    while sysstats._gpu_util_last is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert sysstats._gpu_util() == {"gpu": {"percent": 17.5}}


def test_gpu_util_omits_field_when_nvidia_smi_absent(monkeypatch):
    """AMD/Intel box (no nvidia-smi): the field stays omitted, never a
    fabricated 0%, the missing binary never raises, and a FAILED attempt is
    throttled by the same refresh window as a successful one - otherwise a
    GPU-less box would spawn a new probe thread on every single poll
    forever, since a reading that never succeeds would never look 'fresh'."""
    _reset_gpu_util_cache(monkeypatch)
    calls = []

    def _missing(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", _missing)

    assert sysstats._gpu_util() == {}
    deadline = time.monotonic() + 1
    while sysstats._gpu_util_inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == 1

    for _ in range(10):
        assert sysstats._gpu_util() == {}
    assert len(calls) == 1, (
        f"expected the failed attempt to be throttled by the refresh window, "
        f"subprocess.run was invoked {len(calls)} times")


def test_gpu_util_does_not_reprobe_within_the_refresh_window(monkeypatch):
    """A fresh-enough cached reading must be served without spawning another
    nvidia-smi call - otherwise every single poll would still shell out."""
    _reset_gpu_util_cache(monkeypatch)
    calls = []

    class _Proc:
        returncode = 0
        stdout = "5\n"

    def _run(*args, **kwargs):
        calls.append(1)
        return _Proc()

    monkeypatch.setattr("subprocess.run", _run)

    sysstats._gpu_util()
    deadline = time.monotonic() + 2
    while sysstats._gpu_util_last is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == 1

    # Polling again immediately (well inside the refresh window) must reuse
    # the cached reading rather than shelling out again.
    for _ in range(10):
        assert sysstats._gpu_util() == {"gpu": {"percent": 5.0}}
    assert len(calls) == 1


def test_gpu_util_never_raises_and_unlatches_when_thread_creation_fails(monkeypatch):
    """If the OS cannot spawn the probe thread at all (e.g. thread exhaustion),
    _gpu_util() must still never raise (this module's own documented contract)
    AND must reset the in-flight guard so a later call can retry - otherwise a
    single failed spawn would wedge every future poll into believing a probe is
    permanently running. Mirrors discover.py's list_gpus() handling of the same
    failure."""
    _reset_gpu_util_cache(monkeypatch)

    def _broken_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _broken_start)

    result = sysstats._gpu_util()   # must not raise

    assert result == {}
    assert sysstats._gpu_util_inflight is False, (
        "a failed thread spawn left _gpu_util_inflight stuck True - no later "
        "call could ever retry")


class TestPerDeviceVram:
    """The per-card VRAM breakdown.

    On a multi-GPU board ``vram_capacity`` either sums a configured split (a
    full card and an empty one average into one number) or, with no split
    configured, falls back to the single main GPU and does not represent the
    other cards at all. Neither answers "how full is card 1".
    """

    _GIB = 1024 ** 3

    def _fake(self, free0, free1):
        return [
            {"index": 0, "name": "RTX 4090", "total": 24 * self._GIB, "free": free0,
             "free_scope": "device"},
            {"index": 1, "name": "RTX 3090", "total": 24 * self._GIB, "free": free1,
             "free_scope": "device"},
        ]

    def _compute(self, monkeypatch, devices, *, trusted=True):
        from localm import discover, sysstats
        monkeypatch.setattr(discover, "last_known_gpus", lambda *a, **k: devices)
        monkeypatch.setattr(
            discover, "vram_capacity",
            lambda *a, **k: ({"total": 48 * self._GIB, "free": 26 * self._GIB}, None))
        monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: trusted)
        return sysstats._compute_vram()["vram"]

    def test_each_card_reports_its_own_used_total(self, monkeypatch):
        v = self._compute(monkeypatch, self._fake(4 * self._GIB, 22 * self._GIB))
        assert [d["index"] for d in v["devices"]] == [0, 1]
        # The aggregate reads 22/48 (46%, comfortable) while card 0 is at 20/24
        # (83%, nearly full). Both are visible, not just the mean.
        assert v["used"] == 22 * self._GIB
        assert v["devices"][0]["used"] == 20 * self._GIB
        assert v["devices"][1]["used"] == 2 * self._GIB
        assert v["devices"][0]["percent"] > 80 > v["devices"][1]["percent"]

    def test_an_untrusted_reading_reports_total_only_per_card(self, monkeypatch):
        # Same contract the aggregate honours: a stale or process-scoped `free`
        # overstates what is available, so a per-card figure that cannot be
        # trusted is omitted rather than reported.
        v = self._compute(monkeypatch, self._fake(4 * self._GIB, 22 * self._GIB),
                          trusted=False)
        for d in v["devices"]:
            assert d["total"] == 24 * self._GIB
            assert "used" not in d and "percent" not in d

    def test_a_single_card_sends_no_breakdown(self, monkeypatch):
        # `devices` is only sent when it says something the aggregate does not,
        # so a one-card board keeps the payload it always had.
        one = [{"index": 0, "name": "RTX 4090", "total": 24 * self._GIB,
                "free": 4 * self._GIB, "free_scope": "device"}]
        assert "devices" not in self._compute(monkeypatch, one)

    def test_a_failing_probe_never_costs_the_aggregate(self, monkeypatch):
        # The enrichment failing is visible as a MISSING BREAKDOWN, never as a
        # missing readout and never as a silenced aggregate.
        def boom(*a, **k):
            raise RuntimeError("driver wedged")
        from localm import discover, sysstats
        monkeypatch.setattr(discover, "last_known_gpus", boom)
        monkeypatch.setattr(
            discover, "vram_capacity",
            lambda *a, **k: ({"total": 48 * self._GIB, "free": 26 * self._GIB}, None))
        monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: True)
        v = sysstats._compute_vram()["vram"]
        assert v["total"] == 48 * self._GIB and v["used"] == 22 * self._GIB
        assert "devices" not in v


class TestPerDeviceVramAnyBackend:
    """The per-card breakdown must not be NVIDIA/ROCm-only.

    ``list_gpus`` enumerates via torch.cuda or nvidia-smi and never the Vulkan
    loader, so on a vulkan build (how Intel Arc and many AMD boards run) it is
    structurally blind to the very devices the readout is about. The ggml
    runtime's own registry is the fallback.
    """

    _GIB = 1024 ** 3

    def test_falls_back_to_the_native_registry_when_torch_and_smi_see_nothing(
            self, monkeypatch):
        from localm import discover, sysstats
        from localm.inference.backends.llamacpp import _loader
        # The vulkan-build shape: the torch/nvidia-smi source is EMPTY, not wrong.
        monkeypatch.setattr(discover, "last_known_gpus", lambda *a, **k: [])
        monkeypatch.setattr(
            discover, "vram_capacity",
            lambda *a, **k: ({"total": 32 * self._GIB, "free": 8 * self._GIB}, None))
        monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: True)
        monkeypatch.setattr(_loader, "native_device_inventory", lambda: [
            {"index": 0, "name": "Vulkan0", "total": 16 * self._GIB, "free": 2 * self._GIB},
            {"index": 1, "name": "Vulkan1", "total": 16 * self._GIB, "free": 6 * self._GIB},
        ])
        # The registry tags no scope, so stand in for the correction pass rather
        # than touching real hardware.
        def _tag(gpus):
            for g in gpus:
                g["free_scope"] = "device"
        monkeypatch.setattr(discover, "_apply_device_global_free", _tag)
        v = sysstats._compute_vram()["vram"]
        assert [d["index"] for d in v["devices"]] == [0, 1]
        assert v["devices"][0]["used"] == 14 * self._GIB
        assert v["devices"][1]["used"] == 10 * self._GIB

    def test_the_torch_source_still_wins_when_it_has_devices(self, monkeypatch):
        # last_known_gpus costs nothing (vram_capacity just probed) while the
        # native registry needs the lib resident, so the registry stays the
        # fallback and does not displace a working reading.
        from localm import discover, sysstats
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(discover, "last_known_gpus", lambda *a, **k: [
            {"index": 0, "name": "RTX 4090", "total": 24 * self._GIB,
             "free": 4 * self._GIB, "free_scope": "device"},
            {"index": 1, "name": "RTX 3090", "total": 24 * self._GIB,
             "free": 22 * self._GIB, "free_scope": "device"},
        ])
        monkeypatch.setattr(
            discover, "vram_capacity",
            lambda *a, **k: ({"total": 48 * self._GIB, "free": 26 * self._GIB}, None))
        monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: True)
        def boom():
            raise AssertionError("native inventory must not be consulted here")
        monkeypatch.setattr(_loader, "native_device_inventory", boom)
        v = sysstats._compute_vram()["vram"]
        assert [d["name"] for d in v["devices"]] == ["RTX 4090", "RTX 3090"]


def test_a_process_scoped_card_reports_total_only(monkeypatch):
    """On Windows + AMD the raw driver `free` counts only the CALLING process,
    so a probe holding no VRAM reports almost the whole card as available. A
    process-scoped card therefore reports total only, never a used figure.
    """
    GB = 1024 ** 3
    from localm import discover, sysstats
    monkeypatch.setattr(discover, "last_known_gpus", lambda *a, **k: [
        {"index": 0, "name": "RX 6900 XT", "total": 16 * GB, "free": 15 * GB,
         "free_scope": "process"},
        {"index": 1, "name": "RX 6900 XT", "total": 16 * GB, "free": 15 * GB,
         "free_scope": "device"},
    ])
    monkeypatch.setattr(discover, "vram_capacity",
                        lambda *a, **k: ({"total": 32 * GB, "free": 30 * GB}, None))
    monkeypatch.setattr(sysstats, "_vram_reading_trusted", lambda *a, **k: True)
    devs = sysstats._compute_vram()["vram"]["devices"]
    # Card 0's reading is known blind: total only, no fabricated used/percent.
    assert "used" not in devs[0] and "percent" not in devs[0]
    assert devs[0]["total"] == 16 * GB
    # Card 1's is device-global on the same board, so it still reports in full -
    # the gate is PER CARD, not all-or-nothing.
    assert devs[1]["used"] == 1 * GB


@pytest.fixture(autouse=True)
def _no_wddm_fallback(monkeypatch):
    """Keep the WDDM utilisation fallback OUT of every test that is about the
    nvidia-smi probe path.

    Those tests assert `_gpu_util() == {}` to mean "the nvidia reading has not
    landed, so nothing is fabricated". With a second source live, that empty
    dict would be a statement about the whole machine instead - on a Windows box
    with any GPU the fallback returns a real number.

    The tests that ARE about the fallback stub it back on explicitly.
    """
    from localm import gpu_usage
    monkeypatch.setattr(gpu_usage, "adapter_utilisation", lambda: {})
    # The AMD source is stubbed too: it answers from the card's own sensor with
    # no rate-counter warmup, so it returns a real percentage on the first call
    # and every nvidia-probe test would otherwise assert against live hardware.
    # Tests that ARE about the AMD path stub it back on.
    monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", lambda: None)


class TestVendorNeutralGpuUtilisation:
    """GPU load must not be an NVIDIA-only metric.

    Windows publishes GPU load for every vendor through the counters Task
    Manager itself reads, so an AMD or Intel board reports a real figure rather
    than an omitted field.
    """

    def test_falls_back_to_the_wddm_counter_when_nvidia_smi_has_nothing(
            self, monkeypatch):
        from localm import gpu_usage, sysstats
        monkeypatch.setattr(sysstats, "_gpu_util_last", None, raising=False)
        monkeypatch.setattr(sysstats, "_gpu_util_last_t", 0.0, raising=False)
        # Two adapters, as a real board reports: the busiest one is the compute card.
        monkeypatch.setattr(gpu_usage, "adapter_utilisation",
                            lambda: {"0x0_0xA": 7.1, "0x0_0xB": 0.0})
        assert sysstats._gpu_util() == {"gpu": {"percent": 7.1}}

    def test_omits_the_field_rather_than_fabricating_zero(self, monkeypatch):
        # adapter_utilisation returns {} on its first ever call: a rate counter
        # has nothing to rate against yet.
        from localm import gpu_usage, sysstats
        monkeypatch.setattr(sysstats, "_gpu_util_last", None, raising=False)
        monkeypatch.setattr(sysstats, "_gpu_util_last_t", 0.0, raising=False)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation", lambda: {})
        assert sysstats._gpu_util() == {}

    def test_a_broken_counter_never_breaks_the_readout(self, monkeypatch):
        from localm import gpu_usage, sysstats
        monkeypatch.setattr(sysstats, "_gpu_util_last", None, raising=False)
        monkeypatch.setattr(sysstats, "_gpu_util_last_t", 0.0, raising=False)
        def boom():
            raise OSError("PDH exploded")
        monkeypatch.setattr(gpu_usage, "adapter_utilisation", boom)
        assert sysstats._gpu_util() == {}


class TestAdapterUtilisationAggregation:
    def test_busiest_engine_not_the_sum_across_engines(self):
        """Summing engine types double-counts concurrent work and exceeds 100%.

        Task Manager reports the busiest engine, and so does this readout.
        """
        from localm.gpu_usage import _luid_of
        # The LUID pair identifies the ADAPTER; pid and engine index do not, which
        # is what lets a multi-card board be reported per card.
        a = _luid_of("pid_1234_luid_0x00000000_0x0000C3F1_phys_0_eng_0_engtype_3D")
        b = _luid_of("pid_9999_luid_0x00000000_0x0000C3F1_phys_0_eng_3_engtype_Copy")
        c = _luid_of("pid_1234_luid_0x00000000_0x0000D107_phys_0_eng_0_engtype_3D")
        assert a == b, "same adapter, different process and engine"
        assert a != c, "different adapter"
        assert _luid_of("no_luid_here") is None


from localm import gpu_usage as _gpu_usage_for_capture

# Captured at IMPORT time, before the autouse _no_wddm_fallback fixture can
# replace it, so the tests that exercise the real function are not shadowed by
# the stub.
_REAL_AMD_ACTIVITY = _gpu_usage_for_capture.amd_whole_gpu_activity


class TestAmdWholeGpuActivity:
    """The AMD readout must be WHOLE-GPU load, not whichever engine is busiest.

    The WDDM ``GPU Engine`` max-over-engine-types fold does not track the card:
    it can report the same few percent whether the card is parked or boosted,
    because that number can come from an unrelated process's video encoder. The
    card's own sensor tracks core clock closely.

    The fold is UNRELIABLE, not dead - under a synthetic pure-compute load it
    does read 93-100%.
    """

    @staticmethod
    def _fake_adl(monkeypatch, *, adapters, sensors, activity_rc=0):
        """Fake ADL whose calls write through ctypes byref, as the real DLL does.

        *adapters*: list of (bus, device, function, present, vendor_id).
        *sensors*:  {adapter_index: {sensor_index: (supported, value)}}.
        """
        from localm import gpu_usage

        class _FakeDll:
            def ADL2_Adapter_NumberOfAdapters_Get(self, _ctx, ref):
                ref._obj.value = len(adapters)
                return 0

            def ADL2_Adapter_AdapterInfo_Get(self, _ctx, ref, _size):
                for i, (bus, dev, fn, present, vendor) in enumerate(adapters):
                    a = ref._obj[i]
                    a.iAdapterIndex = i
                    a.iBusNumber, a.iDeviceNumber, a.iFunctionNumber = bus, dev, fn
                    a.iPresent, a.iVendorID = present, vendor
                return 0

            def ADL2_New_QueryPMLogData_Get(self, _ctx, idx, ref):
                if activity_rc != 0:
                    return activity_rc
                for si, (sup, val) in sensors.get(idx, {}).items():
                    ref._obj.sensors[si].supported = sup
                    ref._obj.sensors[si].value = val
                return 0

        monkeypatch.setattr(gpu_usage, "_adl_open",
                            lambda: {"dll": _FakeDll(), "ctx": None})
        # Undo the autouse stub: these tests ARE the AMD path.
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity",
                            _REAL_AMD_ACTIVITY)

    _ONE_CARD = [(45, 0, 0, 1, 1002)]

    def test_reports_the_cards_own_activity_sensor(self, monkeypatch):
        from localm import gpu_usage
        self._fake_adl(monkeypatch, adapters=self._ONE_CARD,
                       sensors={0: {19: (1, 73)}})
        assert gpu_usage._adl_activity_by_bus() == {45: 73.0}
        assert gpu_usage.amd_whole_gpu_activity() == 73.0

    def test_reads_sensor_19_and_not_a_neighbour(self, monkeypatch):
        """A wrong sensor index returns a plausible 0-100 number rather than an
        error, so it would be a SILENT wrong answer. Neighbouring slots are
        populated with values that would be obviously wrong if picked up."""
        from localm import gpu_usage
        self._fake_adl(monkeypatch, adapters=self._ONE_CARD,
                       sensors={0: {18: (1, 11), 19: (1, 73), 20: (1, 22)}})
        assert gpu_usage.amd_whole_gpu_activity() == 73.0

    def test_an_unsupported_sensor_yields_no_reading_not_zero_percent(
            self, monkeypatch):
        """An unsupported sensor yields NO reading, never 0%: a card reported
        IDLE on a measurement that never happened is a failed reading presented
        as a successful one."""
        from localm import gpu_usage
        self._fake_adl(monkeypatch, adapters=self._ONE_CARD,
                       sensors={0: {19: (0, 0)}})
        assert gpu_usage._adl_activity_by_bus() == {}
        assert gpu_usage.amd_whole_gpu_activity() is None

    def test_an_out_of_range_value_is_refused(self, monkeypatch):
        """Outside 0-100 means we are not reading the sensor we think we are -
        a wrong index or a struct-layout drift. Publishing it would turn an ABI
        bug into a confident wrong number on the status bar."""
        from localm import gpu_usage
        self._fake_adl(monkeypatch, adapters=self._ONE_CARD,
                       sensors={0: {19: (1, 4294967295)}})
        assert gpu_usage._adl_activity_by_bus() == {}
        assert gpu_usage.amd_whole_gpu_activity() is None

    def test_logical_adapters_of_one_card_are_deduped(self, monkeypatch):
        """ADL reports several logical adapters per physical card (7 for the one
        measured here). Without the PCI-triple dedupe one card is counted many
        times."""
        from localm import gpu_usage
        self._fake_adl(monkeypatch,
                       adapters=[(45, 0, 0, 1, 1002)] * 4,
                       sensors={i: {19: (1, 55)} for i in range(4)})
        assert gpu_usage._adl_activity_by_bus() == {45: 55.0}

    def test_non_amd_and_absent_adapters_are_skipped(self, monkeypatch):
        from localm import gpu_usage
        self._fake_adl(
            monkeypatch,
            adapters=[(1, 0, 0, 1, 4318),     # NVIDIA vendor id, present
                      (2, 0, 0, 0, 1002),     # AMD but NOT present
                      (45, 0, 0, 1, 1002)],   # the real one
            sensors={i: {19: (1, 41)} for i in range(3)})
        assert gpu_usage._adl_activity_by_bus() == {45: 41.0}

    def test_busiest_card_wins_on_a_multi_gpu_box(self, monkeypatch):
        from localm import gpu_usage
        self._fake_adl(monkeypatch,
                       adapters=[(45, 0, 0, 1, 1002), (67, 0, 0, 1, 1002)],
                       sensors={0: {19: (1, 12)}, 1: {19: (1, 88)}})
        assert gpu_usage.amd_whole_gpu_activity() == 88.0

    def test_adl_unavailable_is_not_an_idle_card(self, monkeypatch):
        from localm import gpu_usage
        monkeypatch.setattr(gpu_usage, "_adl_open", lambda: {})
        assert gpu_usage._adl_activity_by_bus() == {}
        assert gpu_usage.amd_whole_gpu_activity() is None

    def test_a_failing_adl_call_never_breaks_the_readout(self, monkeypatch):
        from localm import gpu_usage
        self._fake_adl(monkeypatch, adapters=self._ONE_CARD, sensors={},
                       activity_rc=-8)   # ADL_ERR_NOT_SUPPORTED
        assert gpu_usage._adl_activity_by_bus() == {}


class TestGpuUtilSourceOrder:
    """nvidia-smi, then the card's own AMD sensor, then the vendor-neutral fold."""

    @staticmethod
    def _cold(monkeypatch):
        from localm import sysstats
        monkeypatch.setattr(sysstats, "_gpu_util_last", None, raising=False)
        monkeypatch.setattr(sysstats, "_gpu_util_last_t", 0.0, raising=False)

    def test_amd_sensor_beats_the_wddm_fold(self, monkeypatch):
        """THE REGRESSION GUARD. With both sources answering, the fold is the one
        that was reporting another process's video encoder as GPU load, so it
        must not win. Fires on the pre-fix code, which had no AMD branch at all
        and returned the fold's 7.1%."""
        from localm import gpu_usage, sysstats
        self._cold(monkeypatch)
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", lambda: 99.0)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation",
                            lambda: {"0x0_0xA": 7.1})
        assert sysstats._gpu_util() == {"gpu": {"percent": 99.0}}

    def test_falls_back_to_the_fold_when_there_is_no_amd_sensor(self, monkeypatch):
        """Intel and any other Windows board: ADL does not exist there, and the
        vendor-neutral counter is the only device-global source localm has."""
        from localm import gpu_usage, sysstats
        self._cold(monkeypatch)
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", lambda: None)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation",
                            lambda: {"0x0_0xA": 7.1, "0x0_0xB": 0.0})
        assert sysstats._gpu_util() == {"gpu": {"percent": 7.1}}

    def test_a_zero_reading_from_the_card_is_reported_not_discarded(
            self, monkeypatch):
        """0.0 is a real measurement of an idle card (measured: 20W, 5MHz core).
        A falsy check here would silently drop it into the fallback path and
        report the fold's number instead of the truth."""
        from localm import gpu_usage, sysstats
        self._cold(monkeypatch)
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", lambda: 0.0)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation",
                            lambda: {"0x0_0xA": 7.1})
        assert sysstats._gpu_util() == {"gpu": {"percent": 0.0}}

    def test_a_broken_amd_source_falls_through_rather_than_breaking(
            self, monkeypatch):
        from localm import gpu_usage, sysstats
        self._cold(monkeypatch)

        def boom():
            raise OSError("ADL exploded")
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", boom)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation",
                            lambda: {"0x0_0xA": 7.1})
        assert sysstats._gpu_util() == {"gpu": {"percent": 7.1}}

    def test_neither_source_fabricates_a_zero(self, monkeypatch):
        from localm import gpu_usage, sysstats
        self._cold(monkeypatch)
        monkeypatch.setattr(gpu_usage, "amd_whole_gpu_activity", lambda: None)
        monkeypatch.setattr(gpu_usage, "adapter_utilisation", lambda: {})
        assert sysstats._gpu_util() == {}
