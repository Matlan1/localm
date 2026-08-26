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
    CONTEXT_PARAMS_V1, CONTEXT_PARAMS_V2, MODEL_PARAMS_V1, MODEL_PARAMS_V2,
    AbiMismatch, evaluate, verify_abi,
)
from localm.inference.backends.llamacpp._structs import (
    LlamaContextParamsV1, LlamaContextParamsV2, LlamaModelParamsV1,
    LlamaModelParamsV2,
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


# The pre-existing tests exercise the V1 model_params layout, named explicitly.
good_model = good_model_v1


def good_ctx_v1() -> LlamaContextParamsV1:
    cp = LlamaContextParamsV1()
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


def good_ctx_v2() -> LlamaContextParamsV2:
    # Every field this fixture sets is named identically on both layouts (the
    # V1/V2 split is a single INSERTED field, n_outputs_max_per_seq, which nothing
    # here checks), so the values match good_ctx_v1 and only the class differs.
    cp = LlamaContextParamsV2()
    cp.n_ctx = 512
    cp.n_batch = 2048
    cp.n_ubatch = 512
    cp.n_seq_max = 1
    cp.n_threads = 4
    cp.n_threads_batch = 4
    cp.rope_scaling_type = -1
    cp.pooling_type = -1
    cp.attention_type = -1
    cp.flash_attn_type = -1
    cp.type_k = 1
    cp.type_v = 1
    cp.offload_kqv = True
    return cp


# The pre-existing tests exercise the V1 context_params layout, named explicitly.
good_ctx = good_ctx_v1


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
    """``*_default_params`` stand-in that FAITHFULLY honours ``restype``.

    The real function is read TWICE with different restypes: once as raw bytes
    (the layout fingerprint, before any layout is assumed) and once as the
    chosen struct class. Every call constructs a FRESH instance of whatever
    class ``restype`` actually is at call time and copies the underlying
    fixture's bytes into it - exactly what a real by-value FFI return does,
    reinterpreting the same bytes as whatever type the binding declared,
    right or wrong.

    A fake that always handed back the original fixture object would give a
    caller a correctly-typed object no matter what class it requested, so a
    test proving verify_abi() picks the WRONG layout for mismatched bytes
    would pass by construction."""

    def __init__(self, value):
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        out = self.restype()
        n = min(ctypes.sizeof(out), ctypes.sizeof(self._value))
        ctypes.memmove(ctypes.byref(out), ctypes.byref(self._value), n)
        return out


class _FakeLib:
    """A fake CDLL exposing what verify_abi and the layout probe actually call.

    The ``llama_load_mode_*`` marker symbols exist only when *mp* is a V2 struct,
    mirroring a real build - unless *markers* overrides that, which is how the
    probe-contradiction case is constructed."""

    def __init__(self, mp, cp, markers: bool = None,
                 ggml_version: str = None):
        self.llama_model_default_params = _ParamsFn(mp)
        # _ParamsFn, not _FakeFn: detect_context_params_layout reads this TWICE,
        # exactly like llama_model_default_params above - once as raw bytes (the
        # fingerprint, before any layout is assumed) and once as the chosen struct
        # class. A plain _FakeFn ignores restype and hands the raw-bytes reader a
        # typed struct with no .b field.
        self.llama_context_default_params = _ParamsFn(cp)
        # Every real build exports this; only its ARITY differs. Omitting it makes
        # has_penalties_sampler() return False via the symbol-missing branch.
        self.llama_sampler_init_penalties = _FakeFn(0)
        if markers is None:
            markers = isinstance(mp, LlamaModelParamsV2)
        if markers:
            self.llama_load_mode_from_str = _FakeFn(0)
            self.llama_load_mode_name = _FakeFn(0)
        if ggml_version is not None:
            self.ggml_version = _FakeFn(ggml_version.encode())


@pytest.fixture(autouse=True)
def _reset_layout_cache(monkeypatch):
    """Reset the per-process caches, and CUT THE TEST OFF FROM THE REAL RUNTIME.

    Both halves matter.

    The layout/arity are cached per process; these tests hand verify_abi a
    different fake library each time, so a leaked cache would make later tests
    assert against an earlier test's build.

    ``_ggml_version`` falls back to the loader's ggml handles when the passed lib
    does not export ``ggml_version`` - required in production, because the symbol
    lives in ggml-base.dll rather than llama.dll - so a fake lib constructed
    WITHOUT a version would read the machine's actually-installed runtime. The
    fallback is stubbed out here, and "no version" means no version."""
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "_ggml_dev_handles", lambda: [], raising=False)
    _abi._detected_layout = None
    _abi._detected_arity = None
    _abi._layout_assumed = False
    _abi._detected_context_layout = None
    _abi._context_layout_assumed = False
    _abi._last_verdict = None
    yield
    _abi._detected_layout = None
    _abi._detected_arity = None
    _abi._layout_assumed = False
    _abi._detected_context_layout = None
    _abi._context_layout_assumed = False
    _abi._last_verdict = None


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
    # be mistaken for layout corruption.
    mp = good_model()
    mp.split_mode = mode
    assert evaluate(mp, good_ctx()).status == "ok"


def test_large_window_is_diagnostic_not_fatal():
    # Magnitudes above the misaligned-read tripwire bounds must NOT refuse a
    # legitimate (ordered) build - only note it.
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
    assert LlamaContextParamsV1.n_ctx.offset == 0
    assert LlamaContextParamsV1.n_batch.offset == 4
    assert LlamaContextParamsV1.n_ubatch.offset == 8
    assert LlamaContextParamsV1.n_seq_max.offset == 12
    assert LlamaContextParamsV1.rope_scaling_type.offset == 36
    assert LlamaContextParamsV1.pooling_type.offset == 40
    assert LlamaContextParamsV1.attention_type.offset == 44
    assert LlamaContextParamsV1.ctx_other.offset == 152
    assert LlamaContextParamsV2.n_ctx.offset == 0
    assert LlamaContextParamsV2.n_batch.offset == 4
    assert LlamaContextParamsV2.n_ubatch.offset == 8
    assert LlamaContextParamsV2.n_seq_max.offset == 12
    # V2 offsets: n_outputs_max_per_seq was inserted before n_threads, shifting
    # the keystone enums +4 vs V1.
    assert LlamaContextParamsV2.n_outputs_max_per_seq.offset == 24
    assert LlamaContextParamsV2.rope_scaling_type.offset == 40
    assert LlamaContextParamsV2.pooling_type.offset == 44
    assert LlamaContextParamsV2.attention_type.offset == 48
    assert LlamaContextParamsV2.ctx_other.offset == 152
    assert ctypes.sizeof(LlamaContextParamsV1) == ctypes.sizeof(LlamaContextParamsV2)
    assert LlamaModelParamsV1.split_mode.offset == 20
    assert LlamaModelParamsV1.use_mmap.offset == 65
    assert LlamaModelParamsV1.main_gpu.offset == 24
    # V2 offsets: same 72-byte size, main_gpu moved, and the byte V1 calls
    # use_mmap is V2's check_tensors.
    assert LlamaModelParamsV2.split_mode.offset == 20
    assert LlamaModelParamsV2.load_mode.offset == 24
    assert LlamaModelParamsV2.main_gpu.offset == 28
    assert LlamaModelParamsV2.check_tensors.offset == 65
    assert ctypes.sizeof(LlamaModelParamsV1) == ctypes.sizeof(LlamaModelParamsV2)


# --------------------------------------------------------------------------- #
#  llama_model_params layout detection (the lemonade b1288 -> b1307 reorder)
# --------------------------------------------------------------------------- #

def test_detects_v1_and_v2_layouts():
    assert verify_abi(_FakeLib(good_model_v1(), good_ctx())).layout == MODEL_PARAMS_V1
    assert verify_abi(_FakeLib(good_model_v2(), good_ctx())).layout == MODEL_PARAMS_V2


def test_v2_bytes_read_as_v1_would_have_been_missed_before_and_are_caught_now():
    """The silent-corruption case the model_params split covers.

    V2 default-params bytes forced through the V1 class. Both structs are 72
    bytes so the size check does not trip, and split_mode at offset 20 is 1
    either way; V1's `main_gpu` reads V2's `load_mode`, which is also in range.
    The deciding signal is the probe contradiction: the library exports
    llama_load_mode_* while its bytes are being read as V1.
    """
    v2_bytes = good_model_v2()
    as_v1 = LlamaModelParamsV1()
    ctypes.memmove(ctypes.byref(as_v1), ctypes.byref(v2_bytes),
                   ctypes.sizeof(v2_bytes))
    # The size and split_mode checks alone do not notice the misread.
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
    """A main_gpu holding a pointer low-word, which is what a shifted layout
    looks like, is refused as a mismatch."""
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
#  llama_context_params layout detection (the n_outputs_max_per_seq insertion:
#  present on ggml-org b10360, absent on the shipped amd-rocm lemonade b1307,
#  so both layouts are live and one hardcoded struct is wrong for one of them.)
# --------------------------------------------------------------------------- #

def test_detects_ctx_v1_and_v2_layouts():
    assert verify_abi(
        _FakeLib(good_model(), good_ctx_v1())).context_layout == CONTEXT_PARAMS_V1
    assert verify_abi(
        _FakeLib(good_model(), good_ctx_v2())).context_layout == CONTEXT_PARAMS_V2


def test_ctx_v2_bytes_read_as_v1_would_have_been_missed_before_and_are_caught_now():
    """The silent-corruption case the context_params split covers, one axis over
    from the model_params one above. A real (post-insertion) build's
    default-params bytes, forced through the OLD V1 class: rope_scaling_type
    would be read from offset 40, which on V1 is actually pooling_type - so a
    real n_ctx/n_batch/etc. would marshal fine but the keystone read would drift
    from the true field.

    With ONE hardcoded context_params class there is no detection at all: the
    class is correct for whichever build happens to match it and silently reads
    the wrong offsets on the other."""
    v2_bytes = good_ctx_v2()
    as_v1 = LlamaContextParamsV1()
    ctypes.memmove(ctypes.byref(as_v1), ctypes.byref(v2_bytes),
                   ctypes.sizeof(v2_bytes))
    # V2's real ctx_type (0, a plausible small enum) lands at V1's
    # rope_scaling_type offset (36) - readable, plausible, WRONG.
    assert as_v1.rope_scaling_type == 0

    # The fingerprint (the only signal available - no marker symbol exists for
    # this insertion) must still pick V2 for genuinely V2-shaped bytes, so this
    # miscasting is never actually reached in practice.
    layout, notes, assumed = _abi.detect_context_params_layout(
        _FakeLib(good_model(), good_ctx_v2()))
    assert (layout, assumed) == (CONTEXT_PARAMS_V2, False)


def test_ctx_layout_detection_survives_a_drift_the_fingerprint_never_READS():
    """A drift in a field OUTSIDE the fingerprint's window must not disturb it.

    WHAT THIS DOES AND DOES NOT COVER: it drifts n_ctx, which
    _CONTEXT_FINGERPRINT does not read at ANY offset, so it can never fail on a
    scoring defect. The five FINGERPRINTED fields are covered by
    test_ctx_v2_detection_survives_a_drift_in_each_fingerprinted_field and its
    ctx_v1 sibling below."""
    cp = good_ctx_v2()
    cp.n_ctx = 4096                      # unrelated drift, not part of the fingerprint
    v = verify_abi(_FakeLib(good_model(), cp))
    assert v.status == "ok"
    assert v.context_layout == CONTEXT_PARAMS_V2


# The five fields _CONTEXT_FINGERPRINT actually reads: exactly the offsets the
# scoring consumes, not a restatement of the struct.
_CTX_FINGERPRINTED = ["ctx_type", "rope_scaling_type", "pooling_type",
                      "attention_type", "flash_attn_type"]


@pytest.mark.parametrize("field", _CTX_FINGERPRINTED)
@pytest.mark.parametrize("badval", [0, 1, 2, 512])
def test_ctx_v2_detection_survives_a_drift_in_each_fingerprinted_field(
        field, badval):
    """A legitimate ctx_v2 build that drifts ONE fingerprinted default must
    still be IDENTIFIED as ctx_v2 - a determination, not the V1 fallback.

    This is the scoring itself, asserted below verify_abi so a refusal that
    comes from evaluate() (correct, for a genuine keystone drift) cannot be
    confused with a detection failure."""
    cp = good_ctx_v2()
    setattr(cp, field, badval)
    layout, notes, assumed = _abi.detect_context_params_layout(
        _FakeLib(good_model(), cp))
    assert (layout, assumed) == (CONTEXT_PARAMS_V2, False), notes


@pytest.mark.parametrize("field", [f for f in _CTX_FINGERPRINTED
                                   if f != "rope_scaling_type"])
@pytest.mark.parametrize("badval", [0, 1, 2, 512])
def test_ctx_v1_detection_survives_a_drift_in_each_fingerprinted_field(
        field, badval):
    """The ctx_v1 side of the same property, so ctx_v1 builds keep their
    determination too.

    rope_scaling_type is EXCLUDED and covered separately below: it is the one
    offset where the two hypotheses make opposite predictions, so drifting it
    on a ctx_v1 build is genuinely undecidable rather than merely harder."""
    cp = good_ctx_v1()
    setattr(cp, field, badval)
    layout, notes, assumed = _abi.detect_context_params_layout(
        _FakeLib(good_model(), cp))
    assert (layout, assumed) == (CONTEXT_PARAMS_V1, False), notes


def test_ctx_v1_with_a_drifted_rope_scaling_type_stays_inconclusive():
    """The one genuinely ambiguous single drift, and it must NOT be resolved.

    Offset 36 is ctx_v1's rope_scaling_type and ctx_v2's ctx_type, so these
    bytes are explained equally well by "ctx_v1 whose rope_scaling_type
    drifted to 0" and by "ctx_v2 whose flash_attn_type drifted to 0". Each
    reading needs exactly one drift; the fingerprint must decline rather than
    invent a winner. The V1 fallback then binds the class that happens to be
    right, and evaluate() refuses TRUTHFULLY - the build really does have
    rope_scaling_type = 0, which is the keystone violation the refusal names."""
    cp = good_ctx_v1()
    cp.rope_scaling_type = 0
    lib = _FakeLib(good_model(), cp)
    assert _abi.detect_context_params_layout(lib)[2] is True   # assumed
    with pytest.raises(AbiMismatch) as ei:
        verify_abi(lib)
    assert "rope_scaling_type" in ei.value.reason


def test_ctx_v2_with_a_drifted_flash_attn_type_is_not_falsely_refused():
    """A legitimate ctx_v2 build with one drifted default must not be refused.

    flash_attn_type is the field evaluate() lists as NON-FATAL ("a legitimate
    build changed a default - we still load, but note it"). A binary score ties
    4-4 on it, goes inconclusive, falls back to CONTEXT_PARAMS_V1, and
    evaluate() then reads V1's offsets over V2's bytes and reports

        context_params.rope_scaling_type = 0 (expected -1, ...)

    about a runtime whose rope_scaling_type IS -1."""
    cp = good_ctx_v2()
    cp.flash_attn_type = 0
    v = verify_abi(_FakeLib(good_model(), cp))          # must not raise
    assert v.status == "ok", v.failures
    assert v.context_layout == CONTEXT_PARAMS_V2
    # Proof we read the RIGHT field rather than got lucky: the real drift is
    # reported as the non-fatal diagnostic it is...
    assert any("flash_attn_type" in d for d in v.diagnostics), v.diagnostics
    # ...and the phantom one is nowhere, in the verdict or the log.
    assert not any("rope_scaling_type" in d for d in v.diagnostics), v.diagnostics


@pytest.mark.parametrize("field", ["rope_scaling_type", "pooling_type",
                                   "attention_type"])
def test_ctx_v2_keystone_drift_still_refuses_and_names_the_real_field(field):
    """The complement: the three keystones evaluate() hard-refuses on still
    refuse when they genuinely drift, and the message names the field that
    actually moved."""
    cp = good_ctx_v2()
    setattr(cp, field, 0)
    with pytest.raises(AbiMismatch) as ei:
        verify_abi(_FakeLib(good_model(), cp))
    assert f"context_params.{field}" in ei.value.reason
    # No field that is genuinely -1 may be named as a failure.
    others = {"rope_scaling_type", "pooling_type", "attention_type"} - {field}
    for other in others:
        assert not any(other in f for f in ei.value.context["abi_failures"])


@pytest.mark.parametrize("builder,want", [
    (good_ctx_v1, CONTEXT_PARAMS_V1), (good_ctx_v2, CONTEXT_PARAMS_V2)])
def test_context_fingerprint_is_unambiguous_on_the_real_builds(builder, want):
    """The measured baseline the scoring rests on, mirroring
    test_fingerprint_is_unambiguous_on_the_real_builds for model_params."""
    cp = builder()
    raw = bytes(bytearray(
        (ctypes.c_uint8 * ctypes.sizeof(cp)).from_buffer_copy(cp)))
    assert _abi._fingerprint_context_layout(raw) == want


def test_context_fingerprint_still_says_inconclusive_when_it_genuinely_is():
    """All-zero bytes support neither layout's ctx_type-not-(-1) check (0 != -1
    passes for BOTH candidate ctx_type offsets, and the -1 runs both fail since
    everything is 0) - must stay inconclusive rather than picking a winner,
    exactly the same never-false-positive posture as the model_params
    fingerprint."""
    assert _abi._fingerprint_context_layout(bytes(256)) is None


def test_context_params_layout_falls_back_to_v1_when_inconclusive():
    """detect_context_params_layout has no second (symbol) signal to fall back
    on, unlike model_params - an inconclusive fingerprint returns the V1
    layout with `assumed=True` so callers know not to treat it as a
    determination."""

    class _BlankLib:
        def llama_context_default_params(self):
            return _abi._RawParams()

    layout, notes, assumed = _abi.detect_context_params_layout(_BlankLib())
    assert layout == CONTEXT_PARAMS_V1
    assert assumed is True
    assert any("inconclusive" in n for n in notes)


def test_context_params_class_and_layout_helpers():
    assert _abi.context_params_class(CONTEXT_PARAMS_V1) is LlamaContextParamsV1
    assert _abi.context_params_class(CONTEXT_PARAMS_V2) is LlamaContextParamsV2
    assert _abi.context_params_layout(
        _FakeLib(good_model(), good_ctx_v2())) == CONTEXT_PARAMS_V2


def test_unknown_third_layout_still_fails_safe():
    """The guarantee documented on verify_abi(): a layout NEITHER V1 nor V2
    (e.g. a hypothetical future upstream insertion on top of today's V2) must
    never silently misbind. Simulates one by inserting ANOTHER 4-byte field
    at the same position n_outputs_max_per_seq already occupies, shifting
    everything from n_threads onward one more +4 than V2 already does.

    The fingerprint CONFIDENTLY (not inconclusively) misdetects this as V2, so
    layout DETECTION alone is not the safety net here; evaluate()'s independent
    re-check of the actual keystone values at V2's assumed offsets catches the
    resulting garbage and refuses. This is the defense-in-depth verify_abi()'s
    docstring describes."""
    cp = good_ctx_v2()
    raw_v2 = bytes(bytearray(
        (ctypes.c_uint8 * ctypes.sizeof(cp)).from_buffer_copy(cp)))
    # Insert 4 more zero bytes right where n_outputs_max_per_seq already is
    # (offset 24), shifting everything after it +4 again; truncate to keep
    # the original total length (mirrors how the real oversized receptacle
    # always hands back a fixed-size buffer regardless of what wrote to it).
    raw_v3 = raw_v2[:24] + b"\x00\x00\x00\x00" + raw_v2[24:-4]
    assert len(raw_v3) == len(raw_v2)

    # The fingerprint alone picks V2 here rather than falling back to V1 as
    # inconclusive, which is the case evaluate() must still catch. The
    # inconclusive path is covered by
    # test_context_params_layout_falls_back_to_v1_when_inconclusive.
    layout = _abi._fingerprint_context_layout(raw_v3)
    assert layout == CONTEXT_PARAMS_V2, (
        "fixture must exercise the CONFIDENT-misdetection path")

    fake_v3 = LlamaContextParamsV2()
    ctypes.memmove(ctypes.byref(fake_v3), raw_v3, len(raw_v3))
    with pytest.raises(AbiMismatch):
        verify_abi(_FakeLib(good_model(), fake_v3))


# --------------------------------------------------------------------------- #
#  llama_sampler_init_penalties arity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mp_builder,ggml,expected", [
    # V1 implies 4-arg: the layout reorder landed strictly BEFORE the penalties
    # change upstream, so no build is V1 and 5-arg.
    (good_model_v1, "0.13.1", 4),
    (good_model_v1, None,     4),
    # V2 + ggml >= 0.18.1 proves 5-arg (that bump came after the signature change).
    (good_model_v2, "0.18.1", 5),
    (good_model_v2, "0.19.0", 5),
    # V2 + ggml < 0.18.0 implies 4-arg: the 0.17.0 -> 0.18.0 bump landed at
    # upstream b10192, five days BEFORE the signature change, so these predate it.
    # That is the b10105..b10191 window - post-reorder but still 4-arg - which
    # would otherwise be called unknown and lose its repetition penalty.
    (good_model_v2, "0.17.0", 4),
    (good_model_v2, "0.16.3", 4),
    # V2 + ggml == 0.18.0 is the ONLY genuinely ambiguous window (upstream
    # b10192..~b10264): report UNKNOWN rather than guess, because either wrong
    # call mis-marshals every argument.
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
#  wrote: the same accessor passes under BOTH layouts by construction and says
#  nothing about where the byte went.
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


# --------------------------------------------------------------------------- #
#  Per-request cost and log volume
#
#  has_penalties_sampler() / penalties_arity() run inside _build_sampler, once
#  per GENERATION REQUEST, so anything unbounded there scales with request rate.
# --------------------------------------------------------------------------- #

def test_penalties_arity_is_cached_not_re_probed_per_call():
    """ggml_version() is a C call plus a restype/argtypes rebind on a SHARED
    function object; re-doing it per request is pure overhead for an answer that
    cannot change while one library handle is held."""
    lib = _FakeLib(good_model_v2(), good_ctx(), ggml_version="0.18.1")
    calls = []
    real = lib.ggml_version
    lib.ggml_version = lambda *a: (calls.append(1), real(*a))[1]
    lib.ggml_version.restype = None
    lib.ggml_version.argtypes = None

    assert _abi.penalties_arity(lib) == 5
    first = len(calls)
    for _ in range(25):
        assert _abi.penalties_arity(lib) == 5
    assert len(calls) == first, (
        f"ggml_version was re-probed {len(calls) - first} extra times across 25 "
        "calls; penalties_arity must cache")


def test_unknown_arity_warns_once_not_per_request(monkeypatch, caplog):
    """A per-request warning is how a real warning gets ignored. The ambiguous
    build (V2 + ggml 0.18.0) must say so ONCE, not on every generation."""
    import logging

    from localm.inference.backends.llamacpp import _api

    lib = _FakeLib(good_model_v2(), good_ctx(), ggml_version="0.18.0")
    monkeypatch.setattr(_api, "load_lib", lambda: lib)
    monkeypatch.setattr(_api, "_warned_penalties_arity", False, raising=False)
    # Seed the per-process caches off the FAKE lib. Without this the no-arg
    # penalties_arity() inside has_penalties_sampler would reach for the real
    # _loader.load_lib and touch the machine's actual runtime.
    assert _abi.penalties_arity(lib) == 0

    with caplog.at_level(logging.WARNING, logger="localm"):
        results = [_api.has_penalties_sampler() for _ in range(20)]

    assert results == [False] * 20, "an unprovable arity must never be called"
    hits = [r for r in caplog.records
            if "llama_sampler_init_penalties" in r.getMessage()]
    assert len(hits) == 1, (
        f"expected exactly one warning across 20 requests, got {len(hits)}")


def _reinterpret(src, cls):
    """The same 72 native bytes, read through the other layout's class."""
    dst = cls()
    ctypes.memmove(ctypes.byref(dst), ctypes.byref(src), 72)
    return dst


def test_evaluate_cannot_discriminate_the_two_layouts():
    """Pins a LIMIT, not a capability - so the overstated claim cannot creep back.

    evaluate()'s model_params checks catch a MISALIGNED read. They cannot tell
    V1 from V2 on plausible defaults, because every value involved is legal in
    both: V2's load_mode=1 read as V1's main_gpu is a valid device index, and
    V1's main_gpu=0 read as V2's load_mode is a valid LLAMA_LOAD_MODE_NONE.

    The layout DECISION therefore lives in detect_model_params_layout, and the
    probe contradiction, not evaluate(), is what catches a wrong choice."""
    as_v1 = _reinterpret(good_model_v2(), LlamaModelParamsV1)
    assert evaluate(as_v1, good_ctx()).status == "ok"
    as_v2 = _reinterpret(good_model_v1(), LlamaModelParamsV2)
    assert evaluate(as_v2, good_ctx()).status == "ok"


def test_unknown_third_model_params_layout_is_not_caught():
    """Pins the LIMIT of verify_abi's fail-safe on the model_params axis.

    test_unknown_third_layout_still_fails_safe covers the context_params axis,
    where evaluate() re-reads the -1 keystones at wherever the bound class puts
    them. That is a property of that ONE axis; this is its counter-example.

    A hypothetical third model_params layout (one more 4-byte field inserted
    directly before split_mode) is detected as v2 CONFIDENTLY - both signals
    agree, so there is no contradiction to refuse on - and then sails through
    evaluate(), because every model_params check is a RANGE check and every
    shifted value here is still in range. The struct is bound and crossed over
    the FFI by value: a silent misbind."""
    v2 = _raw(good_model_v2())[:72]
    raw3 = v2[:20] + b"\x00\x00\x00\x00" + v2[20:68]
    assert len(raw3) == 72, "the receptacle hands back a fixed-size buffer"

    fake_v3 = LlamaModelParamsV2()
    ctypes.memmove(ctypes.byref(fake_v3), raw3, 72)
    lib = _FakeLib(fake_v3, good_ctx())

    layout, notes, contradiction, assumed = _abi.detect_model_params_layout(lib)
    assert (layout, contradiction, assumed) == (MODEL_PARAMS_V2, None, False), (
        f"fixture must exercise the CONFIDENT-misdetection path; got {notes}")

    v = verify_abi(lib)                       # the point: it does NOT raise
    assert v.status == "ok", v.failures
    assert v.layout == MODEL_PARAMS_V2

    # And it really is a misbind rather than a harmless relabel: the bound
    # class reads split_mode at offset 20, where the inserted field now sits,
    # while this build's actual split_mode has moved to 24.
    assert fake_v3.split_mode == 0
    assert int.from_bytes(raw3[24:28], "little", signed=True) == 1, (
        "the real split_mode must have moved, or this pins nothing")


def test_ctx_v2_with_a_minus_one_ctx_type_is_a_known_undetectable_misbind():
    """The context_params axis's OWN residual - so its fail-safe is not read
    as absolute either.

    The fingerprint locates the layout by finding a non-(-1) ctx_type
    immediately before a four-long run of -1 enums. A ctx_v2 build whose
    ctx_type were ITSELF -1 makes that run five long, and both candidate
    windows then fit inside it: ctx_v1 scores 5, ctx_v2 scores 4, so ctx_v1
    wins CONFIDENTLY and wrongly, and evaluate() cannot object because every
    keystone it reads at V1's offsets is a -1 belonging to V2's run.

    Undecidable from these bytes rather than a scoring flaw: no reading of the
    six offsets involved can locate a boundary inside a longer run. It needs
    ctx_type to adopt the -1 UNSPECIFIED convention its four neighbours use,
    which _CONTEXT_FINGERPRINT documents as false."""
    cp = good_ctx_v2()
    cp.ctx_type = -1
    v = verify_abi(_FakeLib(good_model(), cp))
    assert v.status == "ok", v.failures
    assert v.context_layout == CONTEXT_PARAMS_V1, (
        "if this now resolves to ctx_v2, a new signal was added - update the "
        "fail-safe scope in verify_abi's docstring to match")

    # Prove the wrong bind actually moves a field, or this pins nothing.
    as_v1 = LlamaContextParamsV1()
    ctypes.memmove(ctypes.byref(as_v1), ctypes.byref(cp), ctypes.sizeof(cp))
    assert (cp.n_threads, as_v1.n_threads) == (4, 0)


def test_evaluate_does_catch_a_misaligned_read():
    """The capability the checks above DO have: a pointer low-word, or a -1 in a
    field whose valid range excludes it, is refused - in either layout.

    The -1-IN-load_mode case is NOT here: LLAMA_LOAD_MODE_AUTO = -1 is a
    CORRECT value, so -1 in that field is accepted. See
    test_load_mode_auto_is_accepted_and_costs_the_tripwire_one_value.
    """
    mp = good_model_v2()
    mp.load_mode = 0x6F2A1000             # a pointer low-word, still nonsense
    assert evaluate(mp, good_ctx()).status == "mismatch"
    mp = good_model_v1()
    mp.main_gpu = 0x6F2A1000              # a pointer low-word
    assert evaluate(mp, good_ctx()).status == "mismatch"
    # -1 IS still caught where its range genuinely excludes it.
    mp = good_model_v2()
    mp.main_gpu = -1
    assert evaluate(mp, good_ctx()).status == "mismatch"


def test_load_mode_auto_is_accepted_and_costs_the_tripwire_one_value():
    """LLAMA_LOAD_MODE_AUTO (-1) must pass, and the price must stay visible.

    `llama_model_default_params()` on a current upstream build reports
    load_mode = -1, and `_latest_tag()` resolves at runtime, so a valid set that
    excluded -1 would refuse a shipped install.

    The set mirrors llama.h's enumerators; -1 belongs in it because upstream
    defines it. Widening it to a RANGE, or dropping the check, is the forbidden
    move.

    THE COST: load_mode can no longer flag -1 as a misaligned read, and there is
    no structural discriminator that would narrow this to new builds only. The
    two tags export an IDENTICAL 240-function LLAMA_API surface, so there is no
    marker symbol in the `llama_load_mode_*` spirit to key on, and both obvious
    runtime probes are unsafe on a build that lacks the mode:
    llama_load_mode_name(-1) falls off its switch into GGML_ABORT and kills the
    process, while llama_load_mode_from_str throws std::invalid_argument across
    the C ABI.
    """
    from localm.inference.backends.llamacpp._structs import (
        LLAMA_LOAD_MODE_AUTO, _VALID_LOAD_MODES)

    assert LLAMA_LOAD_MODE_AUTO == -1
    assert LLAMA_LOAD_MODE_AUTO in _VALID_LOAD_MODES

    mp = good_model_v2()
    mp.load_mode = LLAMA_LOAD_MODE_AUTO
    v = evaluate(mp, good_ctx())
    assert v.status == "ok", v.failures

    # AUTO must not read as drift either: it is a genuine upstream default.
    assert not any("load_mode" in d for d in v.diagnostics), v.diagnostics

    # The older default stays equally quiet - two typical values, not a swap.
    mp.load_mode = 1
    v1 = evaluate(mp, good_ctx())
    assert v1.status == "ok", v1.failures
    assert not any("load_mode" in d for d in v1.diagnostics), v1.diagnostics

    # But a value upstream does NOT define still reports as drift.
    mp.load_mode = 3
    assert any("load_mode" in d for d in evaluate(mp, good_ctx()).diagnostics)


def test_skip_env_still_surfaces_a_probe_contradiction():
    """The escape hatch suppresses the REFUSAL, never the FINDING.

    _mismatch_error's text tells users to set this variable when they think the
    refusal is a false alarm, so the skip path is exactly where a real
    contradiction most needs to be readable - and it is the one case where the
    layout localm picked may be the wrong one."""
    import logging

    lib = _FakeLib(good_model_v2(), good_ctx(), markers=False)   # v2 bytes, v1 symbols
    _, _, contradiction, _ = _abi.detect_model_params_layout(lib)
    assert contradiction, "fixture no longer produces a contradiction"

    recs = []

    class _H(logging.Handler):
        def emit(self, r):
            recs.append(r.getMessage())

    lg = logging.getLogger("localm")
    h = _H()
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.WARNING)
    try:
        import os
        os.environ[_abi.SKIP_ENV] = "1"
        try:
            v = verify_abi(lib)
        finally:
            os.environ.pop(_abi.SKIP_ENV, None)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)

    assert v.status == "skipped"
    assert any("bytes say" in d for d in v.diagnostics), (
        f"the contradiction must reach the verdict; got {v.diagnostics}")
    assert any("DISAGREE" in m for m in recs), (
        f"the contradiction must be logged even when skipping; got {recs}")


def test_assumed_layout_does_not_prove_the_penalties_arity():
    """An ASSUMED v1 is not a determined v1.

    detect_model_params_layout falls back to v1 when neither probe is
    conclusive. The "v1 implies 4-arg" inference rests on the upstream ORDERING
    of two changes, which says nothing about a build whose layout was never
    determined, so the arity stays unknown there."""
    # Partially-present markers + a fingerprint matching NEITHER layout is the
    # state that forces the fallback.
    mp = good_model_v2()
    mp.load_mode = 3            # breaks the v2 fingerprint (which expects 1)
    mp.use_extra_bufts = False  # and the v1 one too
    lib = _FakeLib(mp, good_ctx(), markers=False, ggml_version="0.13.1")
    lib.llama_load_mode_from_str = _FakeFn(0)   # ONE marker only -> inconclusive

    layout, notes, contradiction, assumed = _abi.detect_model_params_layout(lib)
    assert contradiction is None
    assert assumed is True, "this fixture must exercise the fallback"
    assert layout == MODEL_PARAMS_V1
    assert any("neither layout probe" in n for n in notes)

    _abi._detected_layout = None
    _abi._layout_assumed = False
    _abi._detected_arity = None
    assert _abi.model_params_layout(lib) == MODEL_PARAMS_V1
    assert _abi.penalties_arity(lib) == 0, (
        "an assumed layout must yield UNKNOWN, never 4")


def test_determined_v1_still_proves_4arg():
    """The complement, so a real lemonade b1288 build keeps its repetition
    penalty: a genuine pre-reorder build exports no llama_load_mode_* symbols
    and fingerprints cleanly as v1."""
    lib = _FakeLib(good_model_v1(), good_ctx(), ggml_version="0.13.1")
    layout, _notes, contradiction, assumed = _abi.detect_model_params_layout(lib)
    assert (layout, contradiction, assumed) == (MODEL_PARAMS_V1, None, False)
    assert _abi.penalties_arity(lib) == 4


# --------------------------------------------------------------------------- #
#  The value fingerprint is SCORED, so one drifted default does not blind it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("builder,want", [
    (good_model_v1, MODEL_PARAMS_V1), (good_model_v2, MODEL_PARAMS_V2)])
def test_fingerprint_survives_one_drifted_default(builder, want):
    """Inconclusive is the state where localm binds on the symbol probe alone,
    with nothing able to corroborate OR contradict it, so it must be rare. With
    one changed default the wrong layout still scores 0 and the right one 2,
    leaving a clear winner."""
    mp = builder()
    mp.use_extra_bufts = False          # drift one long-stable default
    raw = bytes(bytearray(
        (ctypes.c_uint8 * ctypes.sizeof(mp)).from_buffer_copy(mp)))[:72]
    assert _abi._fingerprint_layout(raw) == want


@pytest.mark.parametrize("builder,want", [
    (good_model_v1, MODEL_PARAMS_V1), (good_model_v2, MODEL_PARAMS_V2)])
def test_fingerprint_is_unambiguous_on_the_real_builds(builder, want):
    """The measured baseline the scoring rests on: 3-0, both directions."""
    mp = builder()
    raw = bytes(bytearray(
        (ctypes.c_uint8 * ctypes.sizeof(mp)).from_buffer_copy(mp)))[:72]
    assert _abi._fingerprint_layout(raw) == want


def test_fingerprint_still_says_inconclusive_when_it_genuinely_is():
    """Scoring must not manufacture confidence: bytes that support neither
    layout stay inconclusive rather than picking a weak winner, because a wrong
    'conclusive' answer would trigger a false contradiction and refuse a build
    that is actually fine."""
    assert _abi._fingerprint_layout(b"\x00" * 72) is None


def test_fingerprint_drift_no_longer_forces_the_symbol_probe_to_stand_alone():
    """End to end: a V2 build with a drifted default is now CORROBORATED rather
    than resting on symbols alone, so it carries no 'symbol probe alone' note."""
    mp = good_model_v2()
    mp.use_extra_bufts = False
    layout, notes, contradiction, assumed = _abi.detect_model_params_layout(
        _FakeLib(mp, good_ctx()))
    assert (layout, contradiction, assumed) == (MODEL_PARAMS_V2, None, False)
    assert not any("symbol probe alone" in n for n in notes), notes


def test_penalties_arity_boundary_is_exact():
    """The two ggml bounds are adjacent, so the exact edges are pinned. 0.18.0
    is the one version that must stay undecidable; the versions either side of
    it must not."""
    def arity(v):
        _abi._detected_layout = None
        _abi._layout_assumed = False
        _abi._detected_arity = None
        return _abi.penalties_arity(
            _FakeLib(good_model_v2(), good_ctx(), ggml_version=v))

    assert arity("0.17.99") == 4, "anything below 0.18.0 predates the change"
    assert arity("0.18.0") == 0, "0.18.0 straddles the change - must stay unknown"
    assert arity("0.18.1") == 5, "0.18.1 came after the change"


def test_unparseable_ggml_version_is_unknown_not_guessed():
    """A version string localm cannot parse must not fall through to a bound.
    Guessing here produces a mis-marshalled call, which is the whole thing this
    function exists to avoid."""
    for bogus in ("", "not-a-version", "0", "v0.18"):
        _abi._detected_layout = None
        _abi._layout_assumed = False
        _abi._detected_arity = None
        got = _abi.penalties_arity(
            _FakeLib(good_model_v2(), good_ctx(), ggml_version=bogus))
        assert got in (0, 4, 5), got
        # Specifically: nothing unparseable may be read as >= 0.18.1.
        assert got != 5, f"ggml_version={bogus!r} must not be taken as 5-arg"


# --------------------------------------------------------------------------- #
#  abi_report must report what HAPPENED, not re-derive a fresh verdict
# --------------------------------------------------------------------------- #

def test_abi_report_preserves_a_skipped_verdict(monkeypatch):
    """`localm doctor` must not claim a bypassed check succeeded.

    abi_report reports verify_abi's own verdict; re-running evaluate() would only
    ever return ok/mismatch, so with LOCALM_SKIP_ABI_CHECK set doctor would print
    "native ABI: struct layout matches this build" for a check that never ran."""
    from localm.inference.backends.llamacpp import _loader

    lib = _FakeLib(good_model_v2(), good_ctx())
    monkeypatch.setattr(_loader, "load_lib", lambda: lib)
    monkeypatch.setenv(_abi.SKIP_ENV, "1")

    verify_abi(lib)                      # populates the stored verdict
    v = _abi.abi_report()
    assert v.status == "skipped", (
        f"a bypassed check must not be reported as {v.status!r}")
    assert _abi.SKIP_ENV in v.detail
    assert v.layout == MODEL_PARAMS_V2, "the layout is still known and worth showing"


def test_abi_report_still_reports_ok_when_the_check_actually_ran(monkeypatch):
    """The complement: preserving the verdict must not turn every report into
    'skipped'. A real check still reports ok, with its layout attached."""
    from localm.inference.backends.llamacpp import _loader

    lib = _FakeLib(good_model_v1(), good_ctx())
    monkeypatch.setattr(_loader, "load_lib", lambda: lib)
    monkeypatch.delenv(_abi.SKIP_ENV, raising=False)

    verify_abi(lib)
    v = _abi.abi_report()
    assert v.status == "ok"
    assert v.layout == MODEL_PARAMS_V1


def test_abi_report_carries_the_probe_notes_a_re_derivation_would_drop():
    """The stored verdict is strictly more informative than a re-derived one:
    it keeps the layout-probe notes, which evaluate() never sees."""
    # TWO of the three v2 fingerprint checks must break to reach "inconclusive".
    # The fingerprint is SCORED, not all-or-nothing, so a single drifted default
    # still leaves a 2-0 winner and resolves conclusively.
    mp = good_model_v2()
    mp.load_mode = 3                # breaks the load_mode@24 check
    mp.use_extra_bufts = False      # and the use_extra_bufts@66 check
    lib = _FakeLib(mp, good_ctx())
    v = verify_abi(lib)
    assert v.status == "ok"
    assert any("symbol probe alone" in d for d in v.diagnostics), v.diagnostics
    assert _abi._last_verdict is v
