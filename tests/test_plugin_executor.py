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
import os
import threading
import time
from pathlib import Path

from localm.executor import get_plugin_executor

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every call site the executor split moved off the shared default pool. None of
# these may go back to a bare `run_in_executor(None, ...)`.
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
