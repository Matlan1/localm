# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounds on a stored cron expression: its length is capped at every write and
read, the catch-up scan parses it once, and a tick that stalls while deciding
which jobs are due is reported by tick_error()."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from localm.plugins.builtin.jobs import scheduler as sched_mod
from localm.plugins.builtin.jobs.scheduler import (
    MAX_CRON_EXPR_CHARS,
    JobScheduler,
    cron_match,
    parse_cron,
)
from localm.plugins.builtin.jobs.store import Job, JobStore

# Every value of every field listed one by one: the longest expression a
# person plausibly writes.
_FULL_ENUMERATION = " ".join([
    ",".join(str(v) for v in range(0, 60)),
    ",".join(str(v) for v in range(0, 24)),
    ",".join(str(v) for v in range(1, 32)),
    ",".join(str(v) for v in range(1, 13)),
    ",".join(str(v) for v in range(0, 8)),
])


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    return JobStore(root=tmp_path / "jobs")


def _job(**kw):
    base = dict(name="j", task_kind="chat", schedule_kind="cron",
                schedule="0 0 1 1 0", prompt="p")
    base.update(kw)
    return Job(**base)


def test_full_enumeration_fits_under_the_cap():
    assert len(_FULL_ENUMERATION) == 358
    assert len(_FULL_ENUMERATION) <= MAX_CRON_EXPR_CHARS
    parse_cron(_FULL_ENUMERATION)
    assert cron_match(_FULL_ENUMERATION, 0.0) is True


@pytest.mark.parametrize("expr", [
    " " * 100_000 + "0 0 1 1 0",                 # whitespace pad, 5 valid fields
    "0," * 50_000 + "0 * * * *",                 # comma flood, minute set is {0}
    " " * (MAX_CRON_EXPR_CHARS - 8) + "0 0 1 1 0",   # one past the cap
])
def test_overlong_cron_is_refused_at_the_write(expr):
    assert len(expr) > MAX_CRON_EXPR_CHARS
    with pytest.raises(ValueError, match="character limit") as ei:
        _job(schedule=expr)
    assert len(str(ei.value)) < 200


def test_cron_at_the_cap_is_accepted():
    expr = " " * (MAX_CRON_EXPR_CHARS - 9) + "0 0 1 1 0"
    assert len(expr) == MAX_CRON_EXPR_CHARS
    assert _job(schedule=expr).schedule == expr


def test_overlong_cron_already_in_jobs_json_is_skipped_on_load(store):
    good = _job(name="good")
    store.add(good)
    data = json.loads(store._defs_file.read_text(encoding="utf-8"))
    bad = dict(data["jobs"][0], id="bad000000000", name="bad",
               schedule=" " * 100_000 + "0 0 1 1 0")
    data["jobs"].append(bad)
    store._defs_file.write_text(json.dumps(data), encoding="utf-8")

    assert [j.id for j in store.list()] == [good.id]


def test_catch_up_scan_parses_the_expression_once(store, monkeypatch):
    calls = {"n": 0}
    real = sched_mod.parse_cron

    def counting(expr):
        calls["n"] += 1
        return real(expr)

    monkeypatch.setattr(sched_mod, "parse_cron", counting)
    job = _job(last_run=1_000_000.0)
    now = 1_000_000.0 + 86_400.0
    calls["n"] = 0
    assert JobScheduler(store=store)._missed_cron_slot(job, now) is False
    assert calls["n"] == 1


class _BlockingStore:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def list(self):
        self.entered.set()
        assert self.release.wait(10)
        return []


def test_tick_error_reports_a_tick_stalled_choosing_due_jobs(monkeypatch):
    monkeypatch.setattr(sched_mod, "_TICK_STALL_SECONDS", 0.05)
    store = _BlockingStore()
    sched = JobScheduler(store=store)
    t = threading.Thread(target=sched.tick, kwargs={"now": 0.0})
    t.start()
    try:
        assert store.entered.wait(5)
        threading.Event().wait(0.2)
        warning = sched.tick_error()
        assert warning is not None and "deciding which jobs are due" in warning
    finally:
        store.release.set()
        t.join(5)
    assert sched.tick_error() is None


def test_tick_in_progress_under_the_threshold_is_not_reported():
    store = _BlockingStore()
    sched = JobScheduler(store=store)
    t = threading.Thread(target=sched.tick, kwargs={"now": 0.0})
    t.start()
    try:
        assert store.entered.wait(5)
        assert sched.tick_error() is None
    finally:
        store.release.set()
        t.join(5)


def test_loop_logs_a_stalled_tick_and_clears_it_on_recovery(monkeypatch):
    monkeypatch.setattr(sched_mod, "_TICK_STALL_SECONDS", 0.05)
    store = _BlockingStore()
    sched = JobScheduler(store=store, poll_seconds=1)
    seen: dict = {}

    async def drive():
        assert sched.start()
        try:
            loop = asyncio.get_running_loop()
            assert await loop.run_in_executor(None, store.entered.wait, 5)
            for _ in range(40):
                if sched._last_tick_error is not None:
                    break
                await asyncio.sleep(0.1)
            seen["stalled"] = sched._last_tick_error
            store.release.set()
            for _ in range(40):
                if sched._last_tick_error is None:
                    break
                await asyncio.sleep(0.1)
            seen["after"] = sched._last_tick_error
        finally:
            store.release.set()
            sched.stop()

    asyncio.run(drive())
    assert seen["stalled"] is not None
    assert seen["stalled"].startswith("TickStalledError:"), seen["stalled"]
    assert seen["after"] is None
