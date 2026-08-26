# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared bounded thread pool for plugin/tool blocking work.

This module is a stdlib-only leaf: it makes no localm imports at module scope,
so both the plugin layer and the inference layer can consume it.

Every plugin route that does blocking I/O or CPU work off the event loop -
rag query/extract, web search/fetch, voice transcription, coder session
management, GUI model-listing routes - offloads it with
``loop.run_in_executor(get_plugin_executor(), fn, ...)``. Model load/unload and
chat/completion generation (``localm/inference/``) are NOT routed through this
pool; they stay on the asyncio loop's own default executor, so no volume of
plugin calls can compete with inference for a worker thread.

ONE CARVE-OUT to that workload rule: the REGISTRY METADATA reads in
``localm/inference/routes/models.py`` (the per-model size probe behind
``GET /v1/models/{id}``) DO use this pool. They are blocking filesystem I/O on
a path taken from registry.json, not inference: a registered UNC path can block
in the Windows SMB redirector for minutes. Sorting is by WORKLOAD (blocking
tool/IO work here, model work on the default pool), so where workload and
directory disagree, workload wins.

This is ONE shared pool across every plugin, not one per plugin or scope, so
plugins are isolated from inference but not from each other.

Job runs (``localm/plugins/builtin/jobs``: the scheduler tick and the "run now"
route) are NOT routed through this pool either, even though they live under
``localm/plugins/``: a job run loads and drives a model in-process. They are
also self-serialising (``jobs.runguard.run_slot``, a process-global
non-reentrant lock), so at most one job run ever occupies a worker thread.
"""

from __future__ import annotations

import atexit
import concurrent.futures.thread as _cf_thread
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def pool_is_shut_down(executor: ThreadPoolExecutor | None) -> bool:
    """True when *executor* has been shut down and can no longer accept work.

    Reads the private ``_shutdown`` attribute, the same
    ``ThreadPoolExecutor`` introspection ``_executor_health.pool_health()``
    relies on. Returns False when the attribute is absent, and for a None
    *executor*.
    """
    if executor is None:
        return False
    return bool(getattr(executor, "_shutdown", False))


def _interpreter_is_exiting() -> bool:
    """True once Python has begun tearing the process down.

    ``concurrent.futures.thread`` sets this module-global from
    ``_python_exit``, which the interpreter runs via ``threading._shutdown()``.
    That runs BEFORE ordinary ``atexit`` handlers, so by the time anything
    registered with ``atexit`` (including this module's own handler below)
    executes, this already reads True.

    Once it is True, EVERY pool refuses new work, a freshly built one included,
    since ``submit()`` checks this same global.
    """
    return bool(getattr(_cf_thread, "_shutdown", False))


def get_plugin_executor() -> ThreadPoolExecutor:
    """The shared bounded pool for plugin/tool blocking work.

    Lazily created on first use and reused for the process lifetime. Sized
    with the same formula asyncio uses for its own default executor,
    ``min(32, cpu_count+4)``.

    A SHUT-DOWN POOL IS DETECTED AND REPLACED, with a WARNING naming that
    state, so the plugin routes that depend on this pool keep working.

    DURING TEARDOWN IT RAISES ``RuntimeError`` INSTEAD of replacing - see
    ``_interpreter_is_exiting``. The error text says which of the two happened.
    """
    global _executor
    current = _executor
    if current is not None and not pool_is_shut_down(current):
        return current

    with _lock:
        current = _executor
        if current is not None and not pool_is_shut_down(current):
            return current

        replacing_dead_pool = current is not None
        if replacing_dead_pool and _interpreter_is_exiting():
            raise RuntimeError(
                "the shared plugin thread pool is shut down because this "
                "process is exiting; no new plugin work can be scheduled")

        if replacing_dead_pool:
            # Imported here, not at module scope: this module is a stdlib-only
            # leaf and makes no localm imports at module scope.
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "the shared plugin thread pool was found shut down while the "
                "server is still running, and has been replaced so plugin work "
                "can continue. Nothing in localm shuts this pool down outside "
                "process exit, so please report this.")

        workers = min(32, (os.cpu_count() or 1) + 4)
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="localm-plugin")
        atexit.register(executor.shutdown, wait=False, cancel_futures=True)
        _executor = executor
        return executor
