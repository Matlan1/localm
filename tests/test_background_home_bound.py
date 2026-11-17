# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background work writes into the home of the code that started it.

``config.home_bound`` pins ``home_dir()`` for the work it wraps, so a background
pass that resolves the home when it writes cannot land in a home that became
current after it started (in the suite: the next test's ``LOCALM_HOME``, where
it created the directory that test's own fixture then failed to create). The
autouse fixture ``_no_home_bound_work_left_running`` fails a test that leaves
tracked work running, naming it.
"""

from __future__ import annotations

import threading
import time

import pytest

from localm import config
from localm.memory.record import MemoryRecord
from localm.memory.store import MemoryStore
from tests._home_bound_guard import unfinished_home_bound_runs


def _run_on_a_thread(fn) -> None:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(10.0)
    assert not t.is_alive(), "the background call never finished"


def _record_thread(sink):
    """A Thread stub that keeps the target instead of starting it."""
    class _Rec:
        def __init__(self, target=None, daemon=None):
            sink.append(target)

        def start(self):
            pass
    return _Rec


# --------------------------------------------------------------- the pin ---- #

def test_a_home_bound_call_resolves_the_home_it_was_made_in(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("LOCALM_HOME", str(first))
    seen = []
    run = config.home_bound(lambda: seen.append(config.home_dir()), name="probe",
                            track=False)
    monkeypatch.setenv("LOCALM_HOME", str(second))

    _run_on_a_thread(run)

    assert seen == [first]
    assert config.home_dir() == second


def test_the_pin_ends_when_the_call_returns(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("LOCALM_HOME", str(first))
    run = config.home_bound(lambda: None, name="probe", track=False)
    monkeypatch.setenv("LOCALM_HOME", str(second))

    run()                                   # on this thread

    assert config.home_dir() == second


# ------------------------------------------------------------- the guard ---- #

def test_the_guard_names_tracked_work_still_running():
    before = set(config.home_bound_runs())
    release = threading.Event()
    run = config.home_bound(lambda: release.wait(10.0), name="stuck pass")
    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 10.0
        while (not any(r.name == "stuck pass" for r in config.home_bound_runs())
               and time.monotonic() < deadline):
            time.sleep(0.01)
        left = unfinished_home_bound_runs(before, timeout=0.2)
        assert [r.name for r in left] == ["stuck pass"]
        assert left[0].thread is t
    finally:
        release.set()
        t.join(10.0)
    assert unfinished_home_bound_runs(before, timeout=5.0) == []


def test_every_test_runs_under_the_guard(request):
    assert "_no_home_bound_work_left_running" in request.fixturenames


# --------------------------------------------------- the memory passes ---- #

@pytest.fixture
def memory_plug(monkeypatch):
    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(plug, "_stop", plug._threading.Event())
    monkeypatch.setattr(plug, "_sweep_running", False)
    monkeypatch.setattr(plug, "_auto_running", False)
    return plug


def test_the_backfill_sweep_writes_into_the_home_that_started_it(
        memory_plug, tmp_path, monkeypatch):
    plug = memory_plug
    started_in = config.home_dir()
    st = MemoryStore("owner", "chat", "", root=started_in / "memory")
    st.add(MemoryRecord(text="a fact with no vector", source="user"), embed_fn=None)
    tracked = []
    monkeypatch.setattr(plug, "_embed_fn", lambda: tracked.extend(
        r.name for r in config.home_bound_runs()))
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._maybe_sweep_backfill()
    assert len(spawned) == 1
    later = tmp_path / "later"
    monkeypatch.setenv("LOCALM_HOME", str(later))
    _run_on_a_thread(spawned[0])

    assert (started_in / "memory" / ".backfill_sweep").is_file()
    assert not later.exists(), "the sweep wrote into a home made current after it started"
    assert "memory backfill sweep" in tracked


def test_auto_consolidation_writes_into_the_home_that_started_it(
        memory_plug, tmp_path, monkeypatch):
    plug = memory_plug
    started_in = config.home_dir()

    class _Engine:
        loaded = True

    engine = _Engine()
    monkeypatch.setattr(plug, "_live_engine", lambda: engine)
    seen = []

    def fake_synthesize(complete, *, principal=None, **kw):
        seen.append((config.home_dir(), [r.name for r in config.home_bound_runs()]))
        return {"status": "ok"}

    monkeypatch.setattr(plug, "synthesize_memory", fake_synthesize)
    spawned = []
    monkeypatch.setattr(plug._threading, "Thread", _record_thread(spawned))

    plug._maybe_auto_consolidate(None)
    assert len(spawned) == 1
    later = tmp_path / "later"
    monkeypatch.setenv("LOCALM_HOME", str(later))
    _run_on_a_thread(spawned[0])

    assert seen and seen[0][0] == started_in
    assert "memory auto-consolidate" in seen[0][1]
    assert (started_in / "memory" / ".auto_consolidate").is_file()
    assert not later.exists(), "the pass wrote into a home made current after it started"


# ------------------------------------------------------------------ jobs ---- #

def test_a_job_resolves_the_home_it_was_started_in(tmp_path, monkeypatch):
    from localm.jobmanager import JobManager
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("LOCALM_HOME", str(first))
    switched = threading.Event()
    seen = []

    def fn(job):
        switched.wait(10.0)
        seen.append(config.home_dir())
        return True

    mgr = JobManager()
    job = mgr.start_fn("probe", fn)
    monkeypatch.setenv("LOCALM_HOME", str(second))
    switched.set()
    deadline = time.monotonic() + 10.0
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    assert seen == [first]
