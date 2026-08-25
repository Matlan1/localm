# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper process for test_spawn_worker_dies_with_parent.py - NOT a test module (no ``test_`` prefix, so pytest does not collect it)."""

from __future__ import annotations

import sys
import time

# Keep every spawned runner alive for the life of this process. LOAD-BEARING ON POSIX,
# and a no-op on Windows - which is exactly why it was missed.
#
# The runner owns the ctx.Queue()s it handed to the child. Drop the last reference (as
# `r = ModelRunner(); ...; return r._proc.pid` did - r died at the return) and the queues
# are garbage-collected, and their finalizers unlink the POSIX named semaphores. The child
# is still unpickling those queues at that moment, so its sem_open() loses the race and
# dies with `FileNotFoundError: [Errno 2]` out of SemLock._rebuild - before the test can
# kill the parent, leaving a zombie and "worker was not alive before the kill".
#
# Windows never showed it: there the semaphore HANDLES are duplicated into the child at
# spawn time (popen_spawn_win32's duplicate_for_child), so the parent dropping its
# reference cannot starve the child. `voice` never showed it either, on either platform:
# it stores its handle at module level (localm.voice._proc), so nothing collected it.
# gguf + embedder were the only two that dropped the reference, and they were the exact
# two that failed on ubuntu CI. Reproduced and fixed under WSL Ubuntu 24.04 / py3.12.
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
    # First stdout line = the worker PID the test will watch. Flush so the test
    # reads it immediately, not when the pipe buffer happens to fill.
    print(pid, flush=True)
    time.sleep(600)   # block; the test hard-kills us well before this elapses


if __name__ == "__main__":
    main()
