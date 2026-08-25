# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a long-lived DAEMON that answers the native ggml
backend's VRAM-view query on request, so a native abort inside the query can
never take down the caller.

Why a daemon, not a fresh subprocess per query (the first cut of this fix):
measured live, a fresh-process-per-call design costs 1.1-2.0s EVERY call
(dominated by re-importing localm's llama.cpp binding and re-loading the ggml/
HIP runtime DLL from scratch each time - see dev-notes/memory-fix-campaign/
worklog-harness-elated.md for the measured breakdown), which is unacceptable
for a fallback path that is the COMMON case (NVIDIA/Linux/macOS/Vulkan/Metal -
see gpu_memory()'s docstring in _loader.py for why). A long-lived daemon pays
that cold-start cost ONCE per process lifetime (or once per crash, since a
crash here only kills THIS daemon and the caller respawns a fresh one on the
next query); every query after the first is a plain newline-delimited pipe
round-trip.

Why this exists at all (see gpu_memory()'s docstring in _loader.py for the full
account): llama.cpp's own CUDA/HIP error-check macro treats ANY driver-level
failure from the underlying hipMemGetInfo/cudaMemGetInfo call as unrecoverable
and calls abort() in C - a hard process crash no Python try/except can catch.
Confirmed live, repeatedly, on this exact call. Ollama - the most comparable
llama.cpp-wrapping project - solves the same class of problem (a native crash
inside GPU-touching code must never take down the parent) by running ALL
model-runner GPU work in a separate subprocess; this applies the identical
principle narrowly, to just this one call, as a long-lived probe rather than
the whole model load.

Protocol (newline-delimited, line-buffered, over stdin/stdout): the caller
writes one line per request and reads one reply line. A "devices" line asks
for the native device inventory (see _loader.native_device_inventory) and is
answered with its one-line JSON, or an ERR reply when it cannot be taken; ANY
other line is the legacy memory request, answered with "<free_bytes>
<total_bytes>" on success or an ERR reply on failure (load_lib() never ran/
failed, or the query itself failed). An ERR reply is the literal "ERR",
optionally followed by " <cause>" when the daemon knows WHY (its own
load_lib() failed at startup): stderr is discarded by the caller, so the
protocol line is the only channel that can carry the reason into the caller's
debug log instead of losing it (rule 5 - the bare "ERR" hid a wrong-
interpreter spawn for months). Callers match on the "ERR" prefix, never
equality. After each reply the daemon waits for the next request. Exits
cleanly when stdin hits EOF (the caller closed the pipe, e.g. on normal exit OR
because the OS closed all its handles on a forceful kill - either way this
process does not outlive its caller). If a query aborts the process outright,
the OS itself closes the pipes, which the caller's next read observes as
EOF/broken-pipe and treats as "unmeasurable" - it then may respawn a fresh
daemon for the next query.

Invoked as `python -m localm.inference.backends.llamacpp._vram_probe` by
loader.gpu_memory_isolated().
"""

from __future__ import annotations

import sys


def main() -> int:
    from localm._mp_spawn import suppress_native_error_dialogs
    suppress_native_error_dialogs()   # a native DLL failure loading llama.dll
                                       # here must degrade to a catchable
                                       # exception (-> "ERR"), never a blocking
                                       # modal dialog on this disposable daemon.

    import json

    from localm.inference.backends.llamacpp import _loader

    # When load_lib() itself failed, every query below answers ERR - and the
    # startup failure is the ONLY cause worth naming, so it is remembered and
    # carried in the ERR reply (single-line: the protocol is one line per
    # reply, and a newline inside the cause would desync it).
    load_err = ""
    try:
        _loader.load_lib()
    except Exception as e:
        load_err = " ".join(f"load_lib failed: {e!r}"[:300].split())

    def _err_reply() -> str:
        return f"ERR {load_err}" if load_err else "ERR"

    for line in sys.stdin:
        if line.strip() == "devices":
            # Native device inventory (GPU-SPLIT-VKINDEX: the vulkan build's
            # selectors need the ggml registry's own index space, which no
            # torch/nvidia-smi source can provide). Same per-query posture as
            # the memory reply: a failure answers ERR, the daemon stays alive.
            try:
                devs = _loader.native_device_inventory()
            except Exception:
                devs = None
            print(_err_reply() if devs is None else json.dumps(devs), flush=True)
            continue
        try:
            mem = _loader.gpu_memory()
        except Exception:
            mem = None
        if mem is None:
            print(_err_reply(), flush=True)
        else:
            print(f"{int(mem[0])} {int(mem[1])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
