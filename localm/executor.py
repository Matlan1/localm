# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared bounded thread pool for plugin/tool blocking work.

WHY THIS LIVES AT THE ROOT and not under ``localm/plugins/``, where it began:
it is a dependency-free leaf (stdlib only, zero localm imports) that BOTH the
plugin layer and the inference layer consume. Under ``plugins/`` it was the sole
cause of the only module-level import cycle in the package graph,
``inference <-> plugins``, because of the deliberate carve-out described below.
Moving it removed that cycle without changing a line of behaviour. The name still
says "plugin executor" because it names the WORKLOAD it serves (plugin/tool
blocking work), which is the split that actually matters here - not the directory
it happens to sit in.

Every plugin route that does blocking I/O or CPU work off the event loop -
rag query/extract, web search/fetch, voice transcription, coder session
management, GUI model-listing routes - offloads it with
``loop.run_in_executor(get_plugin_executor(), fn, ...)``. Model load/unload and
chat/completion generation (``localm/inference/``) are NOT routed through this
pool; they stay on the asyncio loop's own default executor.

ONE CARVE-OUT to that workload rule, and it is the reason the old location
created a cycle: the REGISTRY METADATA reads in
``localm/inference/routes/models.py`` (the per-model size probe behind
``GET /v1/models/{id}``) DO use this pool. They are blocking filesystem I/O on a
path taken from registry.json, not inference: a registered UNC path can block in
the Windows SMB redirector for minutes. Putting that on the default executor is
exactly the cross-boundary starvation described below - a caller repeating a cheap
metadata GET could occupy the same workers chat generation waits on. Sorting by
WORKLOAD (blocking tool/IO work here, model work on the default pool) is what the
directory rule is approximating, so where the two disagree, workload wins.

Why the split: before it existed, every ``loop.run_in_executor(None, ...)``
call anywhere in the server drew from the SAME process-wide default pool
(``min(32, cpu_count+4)`` workers) - including chat generation, which holds
the per-model inference semaphore while it waits for a worker thread. A caller
holding only a narrow plugin scope (or any loopback caller under open/no-key
mode) could pipeline enough concurrent tool calls - archive extraction alone
can legitimately run 8-30s+ per file - to occupy every worker thread in that
pool. That starved the SAME pool's inference slot and stalled chat completions
for every user of the server, including the admin: a cross-privilege-boundary
denial of service. Giving plugin/tool work its own pool means no volume of
plugin calls can compete with inference for a worker thread.

This is ONE shared pool across every plugin, not one per plugin/scope: it
removes the cross-boundary risk against inference (the severe case - narrow
scope vs. every user), which is what actually motivated the split. Isolating
plugins from each other too (e.g. a "web"-scoped caller unable to starve a
"rag"-scoped caller) would need a pool per plugin and is a smaller, same-tier
risk; revisit if it proves to matter in practice.

Job runs (``localm/plugins/builtin/jobs``: the scheduler tick and the
"run now" route) are deliberately NOT routed through this pool either, even
though they live under ``localm/plugins/``: a job run loads/drives a model
in-process, so it belongs with inference, not with tool calls. It is also
already self-serialising (``jobs.runguard.run_slot``, a process-global
non-reentrant lock) - at most one job run ever occupies a worker thread for
any length of time, so it cannot be pipelined into pool exhaustion the way an
unbounded burst of tool calls can.
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

    Reads ``_shutdown`` - private, but the same class of long-stable
    ``ThreadPoolExecutor`` introspection ``_executor_health.pool_health()``
    already documents and relies on, and there is no public equivalent
    (``submit()`` raising is the only public signal, which is too late to be a
    check). Defaults to False if the attribute ever disappears, so a future
    Python renaming it degrades to exactly today's behaviour rather than
    making every pool look dead.
    """
    if executor is None:
        return False
    return bool(getattr(executor, "_shutdown", False))


def _interpreter_is_exiting() -> bool:
    """True once Python has begun tearing the process down.

    ``concurrent.futures.thread`` sets this module-global from ``_python_exit``,
    which the interpreter runs via ``threading._shutdown()``. MEASURED, not
    assumed: that runs BEFORE ordinary ``atexit`` handlers, so by the time
    anything registered with ``atexit`` (including this module's own handler
    below) executes, this already reads True.

    That ordering is what makes this a usable guard rather than a race: once it
    is True, EVERY pool refuses new work - a freshly built one included, since
    ``submit()`` checks this same global - so replacing a dead pool during
    teardown could only ever spawn threads nothing will join and register an
    atexit handler mid-atexit, while still failing the call.
    """
    return bool(getattr(_cf_thread, "_shutdown", False))


def get_plugin_executor() -> ThreadPoolExecutor:
    """The shared bounded pool for plugin/tool blocking work.

    Lazily created on first use and reused for the process lifetime. Sized
    with the same formula asyncio uses for its own default executor
    (``min(32, cpu_count+4)``), so splitting this pool out of the shared one
    does not cut plugin capacity - the fix is isolation from inference, not a
    smaller pool.

    A SHUT-DOWN POOL IS DETECTED AND REPLACED, and the reasoning for that
    choice belongs here rather than in a commit message, because the two
    alternatives are both defensible and the decision is not recoverable from
    the code alone:

    This used to guard on ``_executor is None`` ALONE. Once the pool was shut
    down that check kept returning it, every caller's ``submit()`` raised
    ``RuntimeError: cannot schedule new futures after shutdown``, and nothing
    ever recovered - the plugin routes that depend on this pool (rag, web,
    voice, coder session management, the GUI model-listing routes) then failed
    for the life of the process with an error naming a thread pool the user has
    never heard of.

    REPLACING rather than refusing, and WHY that is not "silently masking the
    cause": a survey of all 12 modules that import this function found NO caller
    anywhere that shuts this pool down. The only shutdown that exists is the
    ``atexit`` registration below, plus ``concurrent.futures``'s own
    ``_python_exit`` - both of which mean the PROCESS IS ENDING, and both of
    which are caught by the guard above rather than replaced. So outside
    teardown a dead pool is a state with no known producer in this tree, and
    for a stateless resource like a thread pool the honest response to that is
    to restore service AND SAY SO LOUDLY (rule 5 is satisfied by surfacing the
    state, not by refusing to work). Hence the WARNING, which names the state
    explicitly so the cause is discoverable instead of absorbed.

    DURING TEARDOWN IT REFUSES INSTEAD, because there replacing is not a
    recovery: see ``_interpreter_is_exiting`` for the measured ordering. The
    error says which of the two it was, so "the server is shutting down" is
    never reported as a mysterious pool failure.
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
            # Imported here, not at module scope: this module is a
            # dependency-free stdlib-only leaf on purpose (see the module
            # docstring - that is what removed the inference/plugins import
            # cycle), and a top-level localm import would put it straight back.
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
