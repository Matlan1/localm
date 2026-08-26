# SPDX-License-Identifier: AGPL-3.0-or-later
"""A REAL ctypes-backed LlamaBatch for unit tests: the native batch-fill loop in
LlamaCpp._create_batch runs against real memory. Lets a test assert the actual
layout the native decoder sees (token ids, per-token positions, seq ids, logits
flags, n_tokens).

Use it as the side_effect for a mocked ``api.llama_batch_init``:

    mock_api.llama_batch_init.side_effect = fake_batch_init

and inspect the filled arrays afterwards via ``batch_backing(batch)``.
"""

import ctypes

from localm.inference.backends.llamacpp._structs import LlamaBatch

# Keeps the backing ctypes arrays for each created batch alive (a LlamaBatch only
# holds raw void* pointers) and lets a test read back what _create_batch wrote.
_BACKING: dict = {}


def fake_batch_init(n_tokens: int, embd: int = 0, n_seq_max: int = 1) -> LlamaBatch:
    """Allocate a LlamaBatch backed by real ctypes arrays, mirroring the native
    llama_batch_init(n_tokens, 0, 1) allocation _create_batch fills."""
    cap = max(1, int(n_tokens))
    tok = (ctypes.c_int32 * cap)()
    pos = (ctypes.c_int32 * cap)()
    nsi = (ctypes.c_int32 * cap)()
    log = (ctypes.c_int8 * cap)()
    seq_inner = [(ctypes.c_int32 * 1)() for _ in range(cap)]
    seq_ptrs = (ctypes.POINTER(ctypes.c_int32) * cap)()
    for i in range(cap):
        seq_ptrs[i] = ctypes.cast(seq_inner[i], ctypes.POINTER(ctypes.c_int32))

    b = LlamaBatch()
    b.token = ctypes.cast(tok, ctypes.c_void_p)
    b.pos = ctypes.cast(pos, ctypes.c_void_p)
    b.n_seq_id = ctypes.cast(nsi, ctypes.c_void_p)
    b.seq_id = ctypes.cast(seq_ptrs, ctypes.c_void_p)
    b.logits = ctypes.cast(log, ctypes.c_void_p)

    _BACKING[ctypes.addressof(b)] = {
        "token": tok, "pos": pos, "n_seq_id": nsi, "logits": log,
        "seq_inner": seq_inner, "seq_ptrs": seq_ptrs, "cap": cap,
    }
    return b


def batch_backing(batch: LlamaBatch) -> dict:
    """Read back the arrays a filled batch points at, as plain Python lists,
    trimmed to the batch's n_tokens."""
    raw = _BACKING[ctypes.addressof(batch)]
    n = int(batch.n_tokens)
    return {
        "n_tokens": n,
        "token": [raw["token"][i] for i in range(n)],
        "pos": [raw["pos"][i] for i in range(n)],
        "n_seq_id": [raw["n_seq_id"][i] for i in range(n)],
        "logits": [raw["logits"][i] for i in range(n)],
        "seq_id0": [raw["seq_inner"][i][0] for i in range(n)],
    }
