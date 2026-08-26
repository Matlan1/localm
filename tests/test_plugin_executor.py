# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm/executor.py must give plugin/tool blocking work (rag, web,
voice, coder session management, GUI model routes) a pool that is completely
isolated from the asyncio loop's own default executor - the one
localm/inference/ uses for model load/unload and chat/completion generation.

Without the split, every `loop.run_in_executor(None, ...)` call anywhere in the
server (plugin or inference) draws from the same process-wide pool, so a caller
holding only a narrow plugin scope can pipeline enough slow tool calls to occupy
every worker thread and stall chat completions for every user. These tests
assert the isolation property directly - saturating the plugin pool must not
delay the default pool - not just that the two pools are different objects.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

from localm.executor import get_plugin_executor

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every call site that must use the plugin pool rather than the shared default
# one. None of these may go back to a bare `run_in_executor(None, ...)`.
_PLUGIN_TIER_FILES = [
    "localm/plugins/builtin/web/plug.py",
    "localm/plugins/builtin/voice/plug.py",
    "localm/plugins/builtin/rag/plug.py",
    "localm/plugins/builtin/coder/plug.py",
    "localm/plugins/gui/routes/models.py",
    "localm/plugins/gui/routes/system.py",
]


def test_get_plugin_executor_is_a_process_wide_singleton():
    a = get_plugin_executor()
    b = get_plugin_executor()
    assert a is b


def test_plugin_executor_is_bounded_and_identifiably_named():
    ex = get_plugin_executor()
    # Mirrors asyncio's own default-executor formula.
    expected = min(32, (os.cpu_count() or 1) + 4)
    assert ex._max_workers == expected
    worker = ex.submit(threading.current_thread).result(timeout=5)
    assert worker.name.startswith("localm-plugin")


def test_plugin_executor_is_not_the_loop_default_executor():
    async def _check():
        loop = asyncio.get_running_loop()
        default_worker = await loop.run_in_executor(None, threading.current_thread)
        plugin_worker = await loop.run_in_executor(
            get_plugin_executor(), threading.current_thread)
        return default_worker, plugin_worker

    default_worker, plugin_worker = asyncio.run(_check())
    assert not default_worker.name.startswith("localm-plugin")
    assert plugin_worker.name.startswith("localm-plugin")


def test_saturating_plugin_executor_does_not_stall_default_executor():
    """A burst of slow plugin calls (rag extraction/query, coder session ops,
    ...) would fill a shared default pool and starve chat generation's own
    run_in_executor(None, ...) call. With separate pools, fully saturating the
    plugin pool must leave the default pool's response time unaffected."""
    ex = get_plugin_executor()
    n_workers = ex._max_workers

    release = threading.Event()
    started_lock = threading.Lock()
    started_count = 0
    all_started = threading.Event()

    def _blocking_task():
        nonlocal started_count
        with started_lock:
            started_count += 1
            if started_count == n_workers:
                all_started.set()
        release.wait(10)

    async def _run():
        loop = asyncio.get_running_loop()
        # Occupy every worker in the plugin pool with a task that will not
        # return until release() is called below.
        futures = [loop.run_in_executor(ex, _blocking_task) for _ in range(n_workers)]
        await loop.run_in_executor(None, all_started.wait, 10)
        assert all_started.is_set(), "plugin pool never fully saturated"

        # A task on the DEFAULT executor - what inference generation and
        # model load/unload actually use - must still complete promptly.
        t0 = time.monotonic()
        await loop.run_in_executor(None, lambda: None)
        elapsed = time.monotonic() - t0

        release.set()
        await asyncio.gather(*futures)
        return elapsed

    elapsed = asyncio.run(_run())
    assert elapsed < 1.0, (
        f"a default-executor task took {elapsed:.2f}s while the plugin pool "
        "was fully saturated - plugin and inference work are sharing a pool "
        "again")


def test_plugin_tier_files_never_offload_onto_the_default_executor():
    """Every route that does blocking plugin/tool work must route through
    get_plugin_executor(), never bare `run_in_executor(None, ...)` - that is
    exactly what shares a worker pool with inference again."""
    for rel in _PLUGIN_TIER_FILES:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "run_in_executor(None" not in text, (
            f"{rel} offloads onto the default executor - use "
            "get_plugin_executor() instead")
        assert "get_plugin_executor" in text, (
            f"{rel} no longer imports/uses get_plugin_executor()")


# --------------------------------------------------------------------------- #
#  A shut-down pool is detected, and never handed back as if it were usable    #
#                                                                              #
#  NONE of these tests touch the real process-wide singleton. Each installs    #
#  its OWN pool as the module global and restores the original afterwards:     #
#  shutting the real one down would leave a dead pool behind for every later   #
#  test in this worker, which is the same defect under test, self-inflicted.   #
# --------------------------------------------------------------------------- #

import concurrent.futures.thread as _cf_thread
from concurrent.futures import ThreadPoolExecutor

import pytest

import localm.executor as _ex


@pytest.fixture
def own_pool():
    """Install a throwaway pool as the module global; restore the real one."""
    saved = _ex._executor
    created: list[ThreadPoolExecutor] = []

    def _install() -> ThreadPoolExecutor:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-pool")
        created.append(pool)
        _ex._executor = pool
        return pool

    try:
        yield _install
    finally:
        _ex._executor = saved
        for p in created:
            p.shutdown(wait=False)


def test_a_live_pool_is_still_returned_unchanged(own_pool):
    # A healthy pool is handed straight back: same object, no replacement.
    pool = own_pool()
    assert _ex.get_plugin_executor() is pool
    assert _ex.get_plugin_executor() is pool


def test_a_shut_down_pool_is_replaced_with_one_that_actually_works(own_pool):
    pool = own_pool()
    pool.shutdown(wait=True)

    # Establish the premise rather than assuming it: a shut-down pool refuses
    # every new submit().
    with pytest.raises(RuntimeError):
        pool.submit(lambda: 1)

    replacement = _ex.get_plugin_executor()

    # The load-bearing assertion is that plugin work can be scheduled again, so
    # the test schedules some; an identity check alone would also pass for a
    # replacement that was itself dead.
    assert replacement.submit(lambda: 21 * 2).result(timeout=10) == 42
    assert replacement is not pool


def test_the_replacement_says_so_out_loud(own_pool, caplog):
    # Restoring service must not also hide that the state happened. Nothing in
    # localm shuts this pool down outside process exit, so a replacement is
    # evidence of something unexplained and has to be reported.
    pool = own_pool()
    pool.shutdown(wait=True)
    with caplog.at_level(logging.WARNING, logger="localm"):
        _ex.get_plugin_executor()
    assert any("shut down" in r.getMessage() for r in caplog.records), (
        "replacing a dead pool logged nothing - the state would be invisible")


def test_a_dead_pool_during_interpreter_exit_refuses_instead_of_replacing(
        own_pool, monkeypatch):
    """At teardown, the pool is not replaced at all.

    Once ``concurrent.futures.thread._shutdown`` is set, EVERY pool refuses new
    work, a brand new one included, because ``submit()`` consults that same
    global. A replacement there could only spawn threads nothing will join and
    register an atexit handler mid-atexit, and still fail the call.

    Patches the REAL module global rather than stubbing
    ``_interpreter_is_exiting``, so the guard's own reader is what gets
    exercised.
    """
    pool = own_pool()
    pool.shutdown(wait=True)
    monkeypatch.setattr(_cf_thread, "_shutdown", True)
    assert _ex._interpreter_is_exiting() is True

    with pytest.raises(RuntimeError) as excinfo:
        _ex.get_plugin_executor()

    # The message must name WHICH of the two states it was.
    assert "exiting" in str(excinfo.value)
    # And it must not have left a replacement behind on the way out.
    assert _ex._executor is pool
