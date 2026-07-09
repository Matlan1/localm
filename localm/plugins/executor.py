# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared bounded thread pool for plugin/tool blocking work.

Every plugin route that does blocking I/O or CPU work off the event loop -
rag query/extract, web search/fetch, voice transcription, coder session
management, GUI model-listing routes - offloads it with
``loop.run_in_executor(get_plugin_executor(), fn, ...)``. Model load/unload and
chat/completion generation (``localm/inference/``) are NOT routed through this
pool; they stay on the asyncio loop's own default executor.

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
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def get_plugin_executor() -> ThreadPoolExecutor:
    """The shared bounded pool for plugin/tool blocking work.

    Lazily created on first use and reused for the process lifetime. Sized
    with the same formula asyncio uses for its own default executor
    (``min(32, cpu_count+4)``), so splitting this pool out of the shared one
    does not cut plugin capacity - the fix is isolation from inference, not a
    smaller pool.
    """
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                workers = min(32, (os.cpu_count() or 1) + 4)
                executor = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="localm-plugin")
                atexit.register(executor.shutdown, wait=False, cancel_futures=True)
                _executor = executor
    return _executor
