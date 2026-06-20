# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the llama.cpp runtime ABI self-check (``_abi``).

No native library required: a fake ``lib`` returns synthetic default-params
structs, so these run on any CI host. Covers the PASS path (with the real field
values probed from the shipped cpu / vulkan / amd-rocm builds), the REFUSE path
for layout drift (negative test), the fail-open path, the escape hatch, and an
offset-invariant guard so the anchor map can never silently drift from the struct
it protects.
"""

from __future__ import annotations

import pytest

from localm.inference.backends.llamacpp import _abi
from localm.inference.backends.llamacpp._abi import (
    AbiMismatch, evaluate, verify_abi,
)
from localm.inference.backends.llamacpp._structs import (
    LlamaContextParams, LlamaModelParams,
)


# --------------------------------------------------------------------------- #
#  Builders: real default-params values probed from every shipped build
#  (amd-rocm b1288 == cpu b9740 == vulkan b9740, byte-for-byte).
# --------------------------------------------------------------------------- #

def good_model() -> LlamaModelParams:
    mp = LlamaModelParams()
    mp.n_gpu_layers = -1
    mp.split_mode = 1          # LLAMA_SPLIT_MODE_LAYER
    mp.main_gpu = 0
    mp.vocab_only = False
    mp.use_mmap = True
    mp.use_extra_bufts = True
    return mp


def good_ctx() -> LlamaContextParams:
    cp = LlamaContextParams()
    cp.n_ctx = 512
    cp.n_batch = 2048
    cp.n_ubatch = 512
    cp.n_seq_max = 1
    cp.n_threads = 4
    cp.n_threads_batch = 4
    cp.rope_scaling_type = -1  # LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    cp.pooling_type = -1       # LLAMA_POOLING_TYPE_UNSPECIFIED
    cp.attention_type = -1     # LLAMA_ATTENTION_TYPE_UNSPECIFIED
    cp.flash_attn_type = -1
    cp.type_k = 1
    cp.type_v = 1
    cp.offload_kqv = True
    return cp


class _FakeFn:
    """A stand-in for a bound ctypes function: accepts restype/argtypes and
    returns a fixed value when called."""

    def __init__(self, value):
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._value


class _FakeLib:
    """A fake CDLL exposing only the two default-params functions verify_abi calls."""

    def __init__(self, mp: LlamaModelParams, cp: LlamaContextParams):
        self.llama_model_default_params = _FakeFn(mp)
        self.llama_context_default_params = _FakeFn(cp)


# --------------------------------------------------------------------------- #
#  PASS path
# --------------------------------------------------------------------------- #

def test_real_build_values_pass():
    v = evaluate(good_model(), good_ctx())
    assert v.status == "ok", v.failures
    assert v.ok
    assert not v.diagnostics   # real values match exactly -> no drift notes


def test_verify_abi_allows_good_lib():
    v = verify_abi(_FakeLib(good_model(), good_ctx()))
    assert v.status == "ok"


# --------------------------------------------------------------------------- #
#  REFUSE path (negative tests - the whole point of the check)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,badval", [
    ("pooling_type", 512),       # a uint/pointer low-word landing here
    ("rope_scaling_type", 0),    # a different enum shifted in
    ("attention_type", 4),
])
def test_keystone_enum_drift_refuses(field, badval):
    cp = good_ctx()
    setattr(cp, field, badval)
    assert evaluate(good_model(), cp).status == "mismatch"
    with pytest.raises(AbiMismatch):
        verify_abi(_FakeLib(good_model(), cp))


def test_split_mode_garbage_refuses():
    mp = good_model()
    mp.split_mode = 99           # not a valid LLAMA_SPLIT_MODE
    assert evaluate(mp, good_ctx()).status == "mismatch"


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_all_valid_split_modes_allowed(mode):
    # NONE/LAYER/ROW/TENSOR = 0/1/2/3 are all real upstream enumerators; none may
    # be mistaken for layout corruption (TENSOR=3 was the regression caught in
    # review - it must NOT false-refuse).
    mp = good_model()
    mp.split_mode = mode
    assert evaluate(mp, good_ctx()).status == "ok"


def test_large_window_is_diagnostic_not_fatal():
    # Magnitudes above the misaligned-read tripwire bounds must NOT refuse a
    # legitimate (ordered) build - only note it. Guards against the upper-bound
    # checks regressing back into a false-positive refusal.
    cp = good_ctx()
    cp.n_ubatch = 2_000_000
    cp.n_batch = 2_000_000       # ordered, but above _MAX_BATCH
    v = evaluate(good_model(), cp)
    assert v.status == "ok"
    assert any("large window" in d for d in v.diagnostics)


def test_batch_ordering_violation_refuses():
    cp = good_ctx()
    cp.n_ubatch = 4096
    cp.n_batch = 512             # n_ubatch > n_batch -> implausible
    assert evaluate(good_model(), cp).status == "mismatch"


def test_nctx_zero_refuses():
    cp = good_ctx()
    cp.n_ctx = 0
    assert evaluate(good_model(), cp).status == "mismatch"


def test_mismatch_error_is_actionable():
    cp = good_ctx()
    cp.pooling_type = 7
    with pytest.raises(AbiMismatch) as exc:
        verify_abi(_FakeLib(good_model(), cp))
    err = exc.value
    # The reason names the offending field and points at the fix + bypass.
    assert "pooling_type" in err.reason
    assert "setup-llama" in err.reason
    assert _abi.SKIP_ENV in err.reason
    assert err.context.get("abi_failures")


# --------------------------------------------------------------------------- #
#  Value drift that is NOT a layout error must still load (no false positive)
# --------------------------------------------------------------------------- #

def test_benign_default_drift_still_loads():
    # A legitimate newer build that changed a non-anchor default (here the
    # window sizes) keeps the structural fingerprint -> loads, just noted.
    cp = good_ctx()
    cp.n_ctx = 4096
    cp.n_batch = 4096
    cp.n_ubatch = 4096
    v = evaluate(good_model(), cp)
    assert v.status == "ok"
    assert any("n_ctx" in d for d in v.diagnostics)


# --------------------------------------------------------------------------- #
#  Safety valves: escape hatch + fail-open
# --------------------------------------------------------------------------- #

def test_escape_hatch_skips(monkeypatch):
    monkeypatch.setenv(_abi.SKIP_ENV, "1")
    cp = good_ctx()
    cp.pooling_type = 123        # would normally refuse
    v = verify_abi(_FakeLib(good_model(), cp))
    assert v.status == "skipped"  # no raise


def test_mechanism_error_fails_open(monkeypatch):
    monkeypatch.delenv(_abi.SKIP_ENV, raising=False)
    # A lib with no default-params symbols -> AttributeError inside the probe.
    v = verify_abi(object())
    assert v.status == "unchecked"
    assert v.ok                   # fail OPEN: the check must not block on itself


# --------------------------------------------------------------------------- #
#  Offset invariant: the anchors must sit where _structs puts them, so a future
#  struct edit cannot silently move a checked field out from under the guard.
# --------------------------------------------------------------------------- #

def test_anchor_offsets_match_struct():
    assert LlamaContextParams.n_ctx.offset == 0
    assert LlamaContextParams.n_batch.offset == 4
    assert LlamaContextParams.n_ubatch.offset == 8
    assert LlamaContextParams.n_seq_max.offset == 12
    assert LlamaContextParams.rope_scaling_type.offset == 36
    assert LlamaContextParams.pooling_type.offset == 40
    assert LlamaContextParams.attention_type.offset == 44
    assert LlamaContextParams.ctx_other.offset == 152
    assert LlamaModelParams.split_mode.offset == 20
    assert LlamaModelParams.use_mmap.offset == 65
