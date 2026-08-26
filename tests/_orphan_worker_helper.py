# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper process for test_spawn_worker_dies_with_parent.py - NOT a test module
(no ``test_`` prefix, so pytest does not collect it).

Run as ``python _orphan_worker_helper.py <gguf|embedder|voice>``: it spawns ONE
real isolated worker via that runner's real spawn path, prints the worker PID on
its first stdout line, then blocks forever. The test then HARD-kills this process
(the worker's parent) and asserts the worker dies on its own.

The ``if __name__ == "__main__"`` guard is required: multiprocessing's spawn
re-imports this module in the worker child, and without the guard the child
re-runs the spawn at import and crashes on ``_check_not_importing_main``."""

from __future__ import annotations

import sys
import time

# Keep every spawned runner alive for the life of this process. The runner owns the
# ctx.Queue()s it handed to the child; dropping the last reference collects them and
# unlinks the POSIX named semaphores the child is still opening.
_keepalive: list = []


def _spawn(kind: str) -> int:
    if kind == "gguf":
        from localm.inference.backends.llamacpp._runner import ModelRunner
        r = ModelRunner()
        r._spawn()
        _keepalive.append(r)
        return r._proc.pid
    if kind == "embedder":
        from localm.inference._embedder_runner import EmbedderRunner
        r = EmbedderRunner()
        r._spawn()
        _keepalive.append(r)
        return r._proc.pid
    if kind == "voice":
        import localm.voice as v
        v._spawn_worker()          # already module-level state; nothing to keep alive
        return v._proc.pid
    raise SystemExit(f"unknown worker kind: {kind!r}")


def main() -> None:
    kind = sys.argv[1]
    pid = _spawn(kind)
    # First stdout line = the worker PID the test will watch; flushed so the test
    # reads it immediately.
    print(pid, flush=True)
    time.sleep(600)   # block; the test hard-kills us long before it returns


if __name__ == "__main__":
    main()
