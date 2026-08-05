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

import ctypes

import pytest

from localm.inference.backends.llamacpp import _abi
from localm.inference.backends.llamacpp._abi import (
    MODEL_PARAMS_V1, MODEL_PARAMS_V2, AbiMismatch, evaluate, verify_abi,
)
from localm.inference.backends.llamacpp._structs import (
    LlamaContextParams, LlamaModelParamsV1, LlamaModelParamsV2,
)


# --------------------------------------------------------------------------- #
#  Builders: real default-params values, from llama_model_default_params() in
#  src/llama-model.cpp at each side of the reorder.
#    V1 = 7c158fbb4aec (lemonade b1288, ggml 0.13.1), probed live off the DLL
#    V2 = 07132750825a (lemonade b1307, ggml 0.18.1)
# --------------------------------------------------------------------------- #

def good_model_v1() -> LlamaModelParamsV1:
    mp = LlamaModelParamsV1()
    mp.n_gpu_layers = -1
    mp.split_mode = 1          # LLAMA_SPLIT_MODE_LAYER
    mp.main_gpu = 0
    mp.vocab_only = False
    mp.use_mmap = True
    mp.use_extra_bufts = True
    return mp


def good_model_v2() -> LlamaModelParamsV2:
    mp = LlamaModelParamsV2()
    mp.n_gpu_layers = -1
    mp.split_mode = 1          # LLAMA_SPLIT_MODE_LAYER
    mp.load_mode = 1           # LLAMA_LOAD_MODE_MMAP
    mp.main_gpu = 0
    mp.vocab_only = False
    mp.use_extra_bufts = True
    return mp


# The pre-existing tests were written against the only layout that existed then;
# keep them exercising it by name rather than silently repointing them.
good_model = good_model_v1


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


class _ParamsFn:
    """``llama_model_default_params`` stand-in that honours ``restype``.

    The real one is read TWICE with different restypes: once as raw bytes (the
    layout fingerprint, before any layout is assumed) and once as the chosen
    struct class. A fake that ignored restype would return the struct to the raw
    read and the fingerprint would never exercise its actual code path."""

    def __init__(self, value):
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        if self.restype is _abi._RawParams:
            raw = _abi._RawParams()
            ctypes.memmove(ctypes.byref(raw), ctypes.byref(self._value),
                           ctypes.sizeof(self._value))
            return raw
        return self._value


class _FakeLib:
    """A fake CDLL exposing what verify_abi and the layout probe actually call.

    The ``llama_load_mode_*`` marker symbols exist only when *mp* is a V2 struct,
    mirroring a real build - unless *markers* overrides that, which is how the
    probe-contradiction case is constructed."""

    def __init__(self, mp, cp: LlamaContextParams, markers: bool = None,
                 ggml_version: str = None):
        self.llama_model_default_params = _ParamsFn(mp)
        self.llama_context_default_params = _FakeFn(cp)
        if markers is None:
            markers = isinstance(mp, LlamaModelParamsV2)
        if markers:
            self.llama_load_mode_from_str = _FakeFn(0)
            self.llama_load_mode_name = _FakeFn(0)
        if ggml_version is not None:
            self.ggml_version = _FakeFn(ggml_version.encode())


@pytest.fixture(autouse=True)
def _reset_layout_cache():
    """The detected layout is cached per process; these tests hand verify_abi a
    different fake library each time, so a leaked cache would make later tests
    assert against an earlier test's build."""
    _abi._detected_layout = None
    yield
    _abi._detected_layout = None


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
    assert LlamaModelParamsV1.split_mode.offset == 20
    assert LlamaModelParamsV1.use_mmap.offset == 65
    assert LlamaModelParamsV1.main_gpu.offset == 24
    # The V2 offsets are the whole reason two classes exist: same 72-byte size,
    # main_gpu moved, and the byte V1 calls use_mmap is V2's check_tensors.
    assert LlamaModelParamsV2.split_mode.offset == 20
    assert LlamaModelParamsV2.load_mode.offset == 24
    assert LlamaModelParamsV2.main_gpu.offset == 28
    assert LlamaModelParamsV2.check_tensors.offset == 65
    assert ctypes.sizeof(LlamaModelParamsV1) == ctypes.sizeof(LlamaModelParamsV2)


# --------------------------------------------------------------------------- #
#  llama_model_params layout detection (the b1288 -> b1307 reorder)
# --------------------------------------------------------------------------- #

def test_detects_v1_and_v2_layouts():
    assert verify_abi(_FakeLib(good_model_v1(), good_ctx())).layout == MODEL_PARAMS_V1
    assert verify_abi(_FakeLib(good_model_v2(), good_ctx())).layout == MODEL_PARAMS_V2


def test_v2_bytes_read_as_v1_would_have_been_missed_before_and_are_caught_now():
    """The exact silent-corruption case this whole split exists for.

    A b1307 build's default-params bytes, forced through the OLD V1 class. Both
    structs are 72 bytes so nothing about the size trips, and split_mode at
    offset 20 - previously the ONLY model_params check - is 1 either way. What
    now catches it is that V1's `main_gpu` reads V2's `load_mode`... which is
    also in range, so the DECIDING signal is the probe contradiction: the
    library exports llama_load_mode_* while its bytes are being read as V1.
    """
    v2_bytes = good_model_v2()
    as_v1 = LlamaModelParamsV1()
    ctypes.memmove(ctypes.byref(as_v1), ctypes.byref(v2_bytes),
                   ctypes.sizeof(v2_bytes))
    # Pre-existing checks alone do NOT notice - documents why more was needed.
    assert as_v1.split_mode == 1

    # markers=True + V1-shaped bytes is the contradiction: symbols say v2, the
    # default-value fingerprint says v1.
    lib = _FakeLib(good_model_v1(), good_ctx(), markers=True)
    with pytest.raises(AbiMismatch) as ei:
        verify_abi(lib)
    assert "llama_load_mode" in ei.value.reason


def test_v2_load_mode_out_of_range_refuses():
    mp = good_model_v2()
    mp.load_mode = 77          # not a valid LLAMA_LOAD_MODE
    v = evaluate(mp, good_ctx())
    assert v.status == "mismatch"
    assert any("load_mode" in f for f in v.failures)


@pytest.mark.parametrize("builder", [good_model_v1, good_model_v2])
def test_main_gpu_garbage_refuses(builder):
    """A pointer low-word landing in main_gpu is what a shifted layout looks
    like; before this check model_params had no bound on that field at all."""
    mp = builder()
    mp.main_gpu = 0x7FFF0000
    v = evaluate(mp, good_ctx())
    assert v.status == "mismatch"
    assert any("main_gpu" in f for f in v.failures)


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
def test_all_valid_load_modes_allowed(mode):
    """LLAMA_LOAD_MODE_{NONE,MMAP,MLOCK,MMAP_MLOCK,DIRECT_IO} are all real
    enumerators - narrowing this set would refuse a legitimate build."""
    mp = good_model_v2()
    mp.load_mode = mode
    assert evaluate(mp, good_ctx()).status == "ok"


def test_v2_detection_survives_a_drifted_default():
    """The value fingerprint may go INCONCLUSIVE (a future build changes a
    default) without that becoming a refusal - only a conclusive CONTRADICTION
    refuses. Never-false-positive is the module's stated priority."""
    mp = good_model_v2()
    mp.use_extra_bufts = False        # breaks the v2 fingerprint, not the layout
    v = verify_abi(_FakeLib(mp, good_ctx()))
    assert v.status == "ok"
    assert v.layout == MODEL_PARAMS_V2
    assert any("use_extra_bufts" in d for d in v.diagnostics)


# --------------------------------------------------------------------------- #
#  llama_sampler_init_penalties arity (upstream #26520)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mp_builder,ggml,expected", [
    # V1 implies 4-arg: the layout reorder landed strictly BEFORE the penalties
    # change upstream, so no build is V1 and 5-arg.
    (good_model_v1, "0.13.1", 4),
    (good_model_v1, None,     4),
    # V2 + ggml >= 0.18.1 proves 5-arg (that bump came after the signature change).
    (good_model_v2, "0.18.1", 5),
    (good_model_v2, "0.19.0", 5),
    # V2 + ggml 0.18.0 is the genuinely ambiguous window (upstream ~b10180..b10269):
    # report UNKNOWN rather than guess, because either wrong call mis-marshals.
    (good_model_v2, "0.18.0", 0),
    (good_model_v2, None,     0),
])
def test_penalties_arity(mp_builder, ggml, expected):
    lib = _FakeLib(mp_builder(), good_ctx(), ggml_version=ggml)
    assert _abi.penalties_arity(lib) == expected


# --------------------------------------------------------------------------- #
#  Layout-neutral field writes land in the RIGHT native bytes
#
#  These read the raw struct bytes back rather than reading the field they just
#  wrote. Reading the field back would pass under BOTH layouts by construction -
#  it is the same accessor - and would therefore prove nothing about the thing
#  that actually broke, which is where the byte went.
# --------------------------------------------------------------------------- #

def _raw(mp) -> bytes:
    return bytes(bytearray(
        (ctypes.c_uint8 * ctypes.sizeof(mp)).from_buffer_copy(mp)))


def _u8(mp, off: int) -> int:
    return _raw(mp)[off]


def _i32(mp, off: int) -> int:
    return int.from_bytes(_raw(mp)[off:off + 4], "little", signed=True)


def test_main_gpu_write_lands_at_the_layout_correct_offset():
    """The concrete corruption: on a V2 build, a main_gpu written at V1's
    offset 24 silently becomes load_mode - changing how weights are mapped and
    dropping the user's GPU selection, with no error."""
    v1 = good_model_v1()
    v1.main_gpu = 3
    assert _i32(v1, 24) == 3, "V1 main_gpu must occupy offset 24"

    v2 = good_model_v2()
    v2.main_gpu = 3
    assert _i32(v2, 28) == 3, "V2 main_gpu must occupy offset 28"
    assert _i32(v2, 24) == 1, (
        "writing main_gpu on a V2 struct must leave load_mode at its native "
        "default; if this is 3 the write went to the V1 offset")


def test_set_use_mmap_false_lands_in_the_right_field_per_layout():
    from localm.inference.backends.llamacpp._structs import (
        LLAMA_LOAD_MODE_NONE, get_use_mmap, set_use_mmap)

    v1 = good_model_v1()
    set_use_mmap(v1, False)
    assert _u8(v1, 65) == 0, "V1 use_mmap is the byte at offset 65"
    assert _u8(v1, 68) == 0, "check_tensors (V1 offset 68) must stay false"
    assert get_use_mmap(v1) is False

    v2 = good_model_v2()
    set_use_mmap(v2, False)
    assert _i32(v2, 24) == LLAMA_LOAD_MODE_NONE, (
        "V2 expresses 'no mmap' through load_mode at offset 24")
    assert _u8(v2, 65) == 0, (
        "offset 65 on V2 is check_tensors - a stale `mp.use_mmap = False` would "
        "have written here, and setting check_tensors revalidates every tensor")
    assert get_use_mmap(v2) is False


def test_set_use_mmap_preserves_an_mlock_request_on_v2():
    """V2 folded mmap and mlock into one enum, so a naive mapping of
    'no mmap' -> LLAMA_LOAD_MODE_NONE would silently drop a caller's mlock."""
    from localm.inference.backends.llamacpp._structs import (
        LLAMA_LOAD_MODE_MLOCK, LLAMA_LOAD_MODE_MMAP_MLOCK, set_use_mmap)

    mp = good_model_v2()
    mp.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK
    set_use_mmap(mp, False)
    assert mp.load_mode == LLAMA_LOAD_MODE_MLOCK
    set_use_mmap(mp, True)
    assert mp.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK


def test_set_use_mmap_on_v2_never_touches_the_v1_boolean_block():
    """Belt and braces: every byte from vocab_only onward must be unchanged
    except through load_mode, so no V2 boolean is collaterally flipped."""
    from localm.inference.backends.llamacpp._structs import set_use_mmap

    mp = good_model_v2()
    before = _raw(mp)[64:72]
    set_use_mmap(mp, False)
    assert _raw(mp)[64:72] == before
