# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper process for test_spawn_worker_dies_with_parent.py - NOT a test module
(no ``test_`` prefix, so pytest does not collect it).

Run as ``python _orphan_worker_helper.py <gguf|embedder|voice>``: it spawns ONE
real isolated worker via that runner's real spawn path, prints the worker PID on
its first stdout line, then blocks forever. The test then HARD-kills this process
(the worker's parent) and asserts the worker dies on its own.

The ``if __name__ == "__main__"`` guard is load-bearing: multiprocessing's spawn
re-imports this module in the worker child, and without the guard the child would
re-run the spawn at import and crash on ``_check_not_importing_main`` (which looks
exactly like a reaped worker and would fake a passing test)."""

from __future__ import annotations

import sys
import time


def _spawn(kind: str) -> int:
    if kind == "gguf":
        from localm.inference.backends.llamacpp._runner import ModelRunner
        r = ModelRunner()
        r._spawn()
        return r._proc.pid
    if kind == "embedder":
        from localm.inference._embedder_runner import EmbedderRunner
        r = EmbedderRunner()
        r._spawn()
        return r._proc.pid
    if kind == "voice":
        import localm.voice as v
        v._spawn_worker()
        return v._proc.pid
    raise SystemExit(f"unknown worker kind: {kind!r}")


def main() -> None:
    kind = sys.argv[1]
    pid = _spawn(kind)
    # First stdout line = the worker PID the test will watch. Flush so the test
    # reads it immediately, not when the pipe buffer happens to fill.
    print(pid, flush=True)
    time.sleep(600)   # block; the test hard-kills us well before this elapses


if __name__ == "__main__":
    main()
