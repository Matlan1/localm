# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a long-lived DAEMON that answers the native ggml backend's VRAM-view query on request, so a native abort inside the query can never take down the caller."""

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
