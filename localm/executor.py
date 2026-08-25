# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared bounded thread pool for plugin/tool blocking work."""

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def get_plugin_executor() -> ThreadPoolExecutor:
    """The shared bounded pool for plugin/tool blocking work."""
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
