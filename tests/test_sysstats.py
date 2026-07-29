# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.sysstats._vram() - the GUI hardware-monitor's VRAM line.

No dedicated test file existed for this module before AUDIT-GPU-SPLIT-1: the
maintainer explicitly expected every relevant VRAM-reading function to be
multi/split-GPU aware, not just the ones that gate a load/refuse decision -
including this status-bar widget, which previously stayed on vram_info()'s
single main-GPU number as a deliberate (and, on reflection, unnecessary)
design choice."""

import sys
import threading
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


def test_reports_single_gpu_when_no_split_configured():
    info = {"total": 16 * GB, "free": 4 * GB, "free_scope": FREE_SCOPE_DEVICE}
    with patch("localm.discover.vram_info", side_effect=_status_aware(info)):
        out = _vram()
    assert out == {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}


def test_reports_combined_capacity_with_a_configured_split():
    """AUDIT-GPU-SPLIT-1: with a configured 2-GPU split, the status bar must
    show the COMBINED total/used, not just the single main GPU's - it now
    goes through discover.vram_capacity(), not vram_info() directly."""
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
        out = _vram()
    assert out["vram"]["total"] == 24 * GB       # 16+8 combined, not 16 alone
    assert out["vram"]["used"] == 12 * GB        # (16-4)+(8-8) combined
    assert out["vram"]["percent"] == 50.0


def test_empty_when_unmeasurable():
    with patch("localm.discover.vram_info", side_effect=_status_aware({})):
        assert _vram() == {}


def test_percent_omitted_when_free_unknown():
    """The registry-fallback tier reports total only (no per-process free
    reading available), so 'used'/'percent' must be omitted, not fabricated
    as 0% used."""
    with patch("localm.discover.vram_info",
               side_effect=_status_aware({"total": 16 * GB})):
        out = _vram()
    assert out == {"vram": {"total": 16 * GB}}


def test_exception_from_discover_is_swallowed_not_raised():
    """_vram() must never raise - a probe failure just omits the section
    (matches the module's own documented 'NEVER raises' contract)."""
    with patch("localm.discover.vram_capacity", side_effect=RuntimeError("boom")):
        assert _vram() == {}


# --- CPU-utilisation meter (the status-bar CPU% fix) ---------------------- #
# psutil.cpu_percent(interval=None) reports a fabricated first reading (0 on an
# idle box, up to 100 on a busy-starting one) because it has no prior sample.
# _CpuMeter keeps its OWN previous snapshot and derives the percent over a real
# window, reporting the first (baseline-less) reading as None so the caller omits
# the CPU field rather than showing a made-up number.

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
    # the skipped 102.05 one: busy_delta 80 / total_delta 200 -> 40%. A 20% here
    # would prove the skipped sample had wrongly become the baseline.
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
    kernel-counter quirk: psutil issues #392/#645/#1210, e.g. Windows `interrupt`)
    must be trimmed to a zero delta for that field alone, exactly like psutil's own
    _cpu_times_deltas - not left to pull down the aggregate total/busy computed by
    subtracting whole-snapshot sums. WinTimes = (user, system, idle, interrupt, dpc).
    interrupt regresses 50 -> 30 (delta -20) while every other field advances."""
    m = _CpuMeter()
    assert m.percent(WinTimes(100, 100, 1000, 50, 50), now=0.0) is None
    # Clamped per-field deltas: user +0, system +40, idle +60, interrupt clamped to
    # 0 (was -20), dpc +0 -> total_delta=100, busy_delta (total-idle)=40 -> 40%.
    # An aggregate-then-subtract approach would compute total_delta=80 (100+140+
    # 1060+30+50=1380 minus prior sum of 1300=80) and busy_delta=20 -> a wrong 25%.
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


# --- GPU-utilisation probe (efficiency/robustness, not the #833 freeze) --- #
# The old _gpu_util() ran `subprocess.run(["nvidia-smi", ...], timeout=4)`
# inline on every ~2.5s poll, with no single-flight and no cache: a slow
# nvidia-smi parked the polling (executor) thread for up to 4s, and spawned a
# fresh subprocess on every single poll regardless of whether the last one had
# even finished. The fix moves the actual subprocess call onto its own
# single-flighted background thread; _gpu_util() itself always returns
# immediately with the last completed reading (or {} before the first one
# lands), matching the CPU%/VRAM "omit rather than fabricate" convention used
# everywhere else in this module. NOTE: this is unrelated to issue #833's
# reported freeze - that dump shows this same subprocess.run's reader-thread
# start blocked behind a cold `import torch` holding the Windows loader lock
# elsewhere in the process (see dev-notes/finding-2026-07-29-loader-lock-
# freezes-the-event-loop.md), which is fixed out-of-process in discover.py /
# _torch_gpu_probe.py, not here.

def _reset_gpu_util_cache(monkeypatch):
    monkeypatch.setattr(sysstats, "_gpu_util_last", None)
    monkeypatch.setattr(sysstats, "_gpu_util_last_at", None)
    monkeypatch.setattr(sysstats, "_gpu_util_inflight", False)


def test_gpu_util_probe_never_blocks_the_polling_thread(monkeypatch):
    """A hanging/slow nvidia-smi must never park the calling (poll) thread,
    and concurrent polls while one probe is in flight must not spawn a second
    nvidia-smi (single-flight). An efficiency/robustness guard, independent of
    issue #833's loader-lock freeze (see the module-level note above)."""
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
