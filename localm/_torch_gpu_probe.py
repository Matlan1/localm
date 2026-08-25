"""Enumerate CUDA/HIP devices via torch, in a CHILD process, and print JSON."""
from __future__ import annotations

import json
import sys


def _enumerate() -> list:
    """torch's CUDA/HIP device list, or [] when torch cannot answer."""
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
