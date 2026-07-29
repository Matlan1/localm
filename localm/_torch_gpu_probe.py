"""Enumerate CUDA/HIP devices via torch, in a CHILD process, and print JSON.

Run as ``python -m localm._torch_gpu_probe``. Prints ONE line of JSON to stdout:
a list of ``{"index", "name", "total", "free"}`` entries, ``[]`` when torch
reports no usable device. Nothing else ever goes to stdout, so the parent can
read exactly one line.

WHY THIS IS NOT DONE IN-PROCESS (issue #833, measured):
``import torch`` on Windows runs ``_load_dll_libraries``, a loop of
``LoadLibraryExW`` calls. ``LoadLibrary`` holds the OS loader lock, and creating
a thread needs that same lock so ``DLL_THREAD_ATTACH`` can run for every loaded
module. So while any thread cold-imports torch, NO thread anywhere in that
process can be created, including the asyncio event loop's worker pool. A
report's hang-watchdog dump caught exactly that: one thread inside
``_load_dll_libraries`` while the request thread and a ``subprocess.run`` both
sat at the last line of ``Thread.start()``, and the event loop stalled 10.9s.

Neither of the timeouts that look like they should have helped could: a probe
deadline bounds how long a CALLER waits, and ``subprocess``'s timeout bounds a
wait on a child that had not been spawned yet. Nothing a caller does can bound a
lock the OS holds on behalf of another thread. Moving the cold import into a
child process is what actually removes it, because the loader lock is per
process. Measured on the dev box: thread creation went from 0.22ms to 341ms
(342x) during a cold torch import, with a warm page cache and an AMD GPU.

The parent keeps the in-process path when torch is ALREADY resident: that import
is a plain ``sys.modules`` cache hit, takes no loader lock, and enumerating there
avoids a process spawn. Only the cold case comes here.

Index space is preserved because the child inherits the parent's environment, so
``CUDA_VISIBLE_DEVICES`` and friends select and order devices identically. That
matters: ``discover.list_gpus`` returns a TORCH index space by contract (see
``resolve_media_default_device``), which is why the probe cannot simply be
reordered to ask ``nvidia-smi`` first.
"""
from __future__ import annotations

import json
import sys


def _enumerate() -> list:
    """torch's CUDA/HIP device list, or [] when torch cannot answer.

    Mirrors the in-process branch of ``discover._list_gpus_probe`` field for
    field, so a caller cannot tell which path produced a reading."""
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            continue   # one device failing to report never hides the rest
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = f"GPU {i}"
        out.append({"index": i, "name": name,
                    "total": int(total), "free": int(free)})
    return out


def main() -> int:
    try:
        devices = _enumerate()
    except BaseException as e:
        # BaseException, not Exception: a torch import can fault hard enough to
        # raise something outside the Exception tree, and a child that dies
        # without printing is indistinguishable to the parent from a hang. The
        # cause rides out on stderr for the parent's debug log rather than dying
        # with a discarded traceback (AGENTS.md rule 5).
        print(f"torch GPU probe failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("[]", flush=True)
        return 1
    print(json.dumps(devices), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
