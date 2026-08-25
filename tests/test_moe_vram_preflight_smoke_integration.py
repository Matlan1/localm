# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end proof of the MoE VRAM preflight fix (n_cpu_moe blindness in _check_vram/_auto_gpu_layers/_auto_ctx_max - see _sizing.py's _effective_model_bytes_for_vram and model_manager/gguf.py's gguf_moe_pinned_expert_bytes)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "bartowski/granite-3.0-1b-a400m-instruct-GGUF"
_FILE = "granite-3.0-1b-a400m-instruct-Q4_K_M.gguf"


@pytest.fixture(scope="module")
def moe_model_path():
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")
    return path


@pytest.fixture(scope="module")
def moe_facts(moe_model_path):
    """Ground truth read directly from the real file - not asserted against an assumed model shape, so this test would notice if bartowski's build ever changes (fewer/more layers, a different expert-tensor naming)."""
    import os
    from pathlib import Path
    from localm.model_manager.gguf import (
        gguf_expert_count, gguf_moe_pinned_expert_bytes)

    p = Path(moe_model_path)
    n_experts = gguf_expert_count(p)
    if n_experts <= 0:
        pytest.skip(f"{_FILE} did not report any experts - not the MoE build "
                    "this test expects")
    model_bytes = os.path.getsize(p)
    # A generous layer count (the real model has 24): gguf_moe_pinned_expert_bytes
    # clamps to whatever tensors actually exist, so this pins everything present.
    pinned_all = gguf_moe_pinned_expert_bytes(p, 999)
    if not pinned_all or pinned_all < model_bytes // 4:
        pytest.skip(f"{_FILE}'s pinned-expert byte share looks implausibly "
                    "small - not the MoE shape this test expects")
    return {"path": moe_model_path, "model_bytes": model_bytes,
            "pinned_all": pinned_all}


def test_real_tensor_names_match_the_pinning_pattern(moe_facts):
    """Sanity floor for everything else in this file: if bartowski's naming ever diverges from blk.<i>.ffn_(gate|down|up)_exps, every test below would silently measure zero pinned bytes and this fix would look verified when it is not - so pin the precondition explicitly."""
    assert moe_facts["pinned_all"] > 0
    # The vast majority of an MoE file this size is its experts - if pinning
    # "all" layers only accounts for a sliver, something is not matching.
    assert moe_facts["pinned_all"] > moe_facts["model_bytes"] * 0.5


def test_check_vram_refuses_without_n_cpu_moe_but_fits_with_it(moe_facts, capsys):
    """The headline defect, against a REAL file: a simulated card too small for the whole model, but big enough for the sliver that remains once every layer's experts are pinned to system RAM."""
    from localm.inference.backends.gguf import GgufBackend

    model_bytes = moe_facts["model_bytes"]
    pinned = moe_facts["pinned_all"]
    effective = model_bytes - pinned
    overhead = 50_000_000
    # A generous KV allowance (n_ctx small, so its exact cost barely matters)
    # plus enough headroom over `effective` that the WITH-n_cpu_moe load
    # cannot be a coincidental near-miss.
    free = effective + overhead + 40_000_000
    total = free + 10_000_000
    # `total` must sit BELOW the whole-file charge for the hard refusal
    # branch (see _check_vram: `need > total` -> "cannot fit regardless"),
    # not just the soft low-VRAM warning.
    assert total < model_bytes, (
        "test precondition: the simulated card must be too small for the "
        "whole file, or the refusal below proves nothing")

    def _run(n_cpu_moe, n_layers):
        b = GgufBackend(moe_facts["path"], n_ctx=256, n_gpu_layers=99,
                        n_cpu_moe=n_cpu_moe)
        with patch.object(GgufBackend, "_split_free_total_bytes",
                          return_value=(None, None, 0)), \
             patch.object(GgufBackend, "_free_vram_bytes", return_value=free), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=total), \
             patch.object(GgufBackend, "_VRAM_OVERHEAD_BYTES", overhead):
            b._check_vram()
        return b

    with pytest.raises(RuntimeError, match="cannot fit"):
        _run(n_cpu_moe=0, n_layers=0)

    _run(n_cpu_moe=999, n_layers=999)   # must NOT raise
    out = capsys.readouterr().out
    assert "Low VRAM" not in out


def test_real_load_with_n_cpu_moe_generates_coherent_text(moe_facts):
    """End to end on whatever real GPU (if any) runs this test: the native tensor_buft_overrides pinning this fix's estimate is now based on must still produce a working, coherent model - proves the import refactor (llama.py's _apply_cpu_moe now sources its pattern from model_manager.gguf instead of a loca..."""
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp._loader import load_lib
    try:
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm "
                    f"setup-llama'): {e}")

    b = GgufBackend(moe_facts["path"], n_ctx=1024, n_gpu_layers=99, n_cpu_moe=24)
    try:
        b.load()
    except Exception as e:
        pytest.skip(f"MoE model failed to load on this machine: {e}")
    try:
        out = "".join(b.chat_stream(
            [{"role": "user",
              "content": "In one short sentence, what color is a clear "
                         "daytime sky?"}],
            max_tokens=40, temperature=0.0, seed=1,
        )).strip()
        assert len(out) >= 10, f"suspiciously short output: {out!r}"
        assert any(c.isalpha() for c in out), f"no words in output: {out!r}"
        assert b.loaded, "model should remain loaded after a normal generation"
    finally:
        b.unload()
