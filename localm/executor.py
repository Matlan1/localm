# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared bounded thread pool for plugin/tool blocking work.

It is a dependency-free leaf (stdlib only, zero localm imports) consumed by both
the plugin layer and the inference layer, so it lives at the package root rather
than under ``localm/plugins/``. The name says "plugin executor" because it names
the WORKLOAD it serves (plugin/tool blocking work).

Every plugin route that does blocking I/O or CPU work off the event loop -
rag query/extract, web search/fetch, voice transcription, coder session
management, GUI model-listing routes - offloads it with
``loop.run_in_executor(get_plugin_executor(), fn, ...)``. Model load/unload and
chat/completion generation (``localm/inference/``) are NOT routed through this
pool; they stay on the asyncio loop's own default executor.

ONE CARVE-OUT to that workload rule: the REGISTRY METADATA reads in
``localm/inference/routes/models.py`` (the per-model size probe behind
``GET /v1/models/{id}``) DO use this pool. They are blocking filesystem I/O on a
path taken from registry.json, not inference, and a registered UNC path can block
in the Windows SMB redirector for minutes, so a caller repeating a cheap metadata
GET could otherwise occupy the same default-pool workers chat generation waits
on. Where the directory rule and the workload rule disagree, workload wins.

What the split buys: a single process-wide default pool
(``min(32, cpu_count+4)`` workers) is shared by chat generation, which holds the
per-model inference semaphore while it waits for a worker thread. A caller
holding only a narrow plugin scope (or any loopback caller under open/no-key
mode) can pipeline enough concurrent tool calls - archive extraction alone can
legitimately run 8-30s+ per file - to occupy every worker in that pool and stall
chat completions for every user of the server. A separate pool means no volume of
plugin calls competes with inference for a worker thread.

ONE shared pool across every plugin, not one per plugin/scope: plugins are not
isolated from each other here.

Job runs (``localm/plugins/builtin/jobs``: the scheduler tick and the "run now"
route) are NOT routed through this pool either, even though they live under
``localm/plugins/``: a job run loads and drives a model in-process, so it belongs
with inference. It is also self-serialising (``jobs.runguard.run_slot``, a
process-global non-reentrant lock), so at most one job run ever occupies a worker
thread and it cannot be pipelined into pool exhaustion.
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

    Lazily created on first use and reused for the process lifetime. Sized with
    the same formula asyncio uses for its own default executor
    (``min(32, cpu_count+4)``), so splitting this pool out of the shared one does
    not cut plugin capacity.
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
