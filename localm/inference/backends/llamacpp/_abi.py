# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime ABI self-check for the native llama.cpp struct layout."""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from localm.bugreport import LocalmError

from ._structs import (
    _VALID_LOAD_MODES,
    LLAMA_LOAD_MODE_AUTO,
    LLAMA_LOAD_MODE_MMAP,
    LlamaContextParamsV1,
    LlamaContextParamsV2,
    LlamaModelParamsV1,
    LlamaModelParamsV2,
)

# Env var that disables the check (escape hatch for a false alarm on a build the
# maintainer could not test). Any non-empty value enables the bypass.
SKIP_ENV = "LOCALM_SKIP_ABI_CHECK"

# The two llama_model_params layouts localm binds. See _structs' module
# docstring for the byte-level difference; both are 72 bytes, which is why the
# reorder cannot be detected from sizeof.
MODEL_PARAMS_V1 = "v1"   # <= 7c158fbb4aec: use_mmap/use_direct_io/use_mlock, main_gpu@24
MODEL_PARAMS_V2 = "v2"   # >= the load_mode reorder: load_mode@24, main_gpu@28, load_mtp

# The two llama_context_params layouts localm binds. See _structs' module
# docstring for the byte-level difference; both are 224 bytes (the V2 field
# insertion and V1's now-unneeded alignment pad exactly cancel out), which is
# why this reorder ALSO cannot be detected from sizeof - same shape as the
# model_params split above, distinct namespace deliberately (a caller passing
# a MODEL_PARAMS_* constant into a context_params_class() call, or vice versa,
# must get a clear "not a valid layout" rather than silently doing something
# plausible-looking with the wrong axis).
CONTEXT_PARAMS_V1 = "ctx_v1"  # <= 07132750825a (lemonade b1307): no n_outputs_max_per_seq
CONTEXT_PARAMS_V2 = "ctx_v2"  # >= somewhere before b10360: n_outputs_max_per_seq@24 inserted

# Symbols that appear in llama.h in the same change as the V2 reorder. Probed at
# 8 upstream ggml-org tags spanning the flip (b9870, b10000, b10050, b10080,
# b10090, b10180, b10200, b10276): the struct field and BOTH of these are always
# all present or all absent. The flip itself is PINNED to upstream b10105
# (commit e6dd0e29a675, ggml-org/llama.cpp#20834); b10103 is the last release
# without it, and b10104 was never tagged.
#
# This is the STRUCTURAL half of the layout decision; the value fingerprint below
# is an independent second opinion, so the decision does not rest on this
# correlation alone.
_V2_MARKER_SYMBOLS = ("llama_load_mode_from_str", "llama_load_mode_name")

# Byte offsets/values that llama_model_default_params() must produce for each
# layout. Used as a corroborating fingerprint, NOT as the primary signal, and
# allowed to be INCONCLUSIVE (a future build may legitimately drift a default) -
# only a CONCLUSIVE CONTRADICTION of the symbol probe is treated as a mismatch.
#   V1: main_gpu@24 == 0, use_mmap@65 == 1, use_extra_bufts@69 == 1
#   V2: load_mode@24 == 1 (MMAP), check_tensors@65 == 0, use_extra_bufts@66 == 1
_FINGERPRINT = {
    MODEL_PARAMS_V1: ((24, "i", 0), (65, "B", 1), (69, "B", 1)),
    MODEL_PARAMS_V2: ((24, "i", 1), (65, "B", 0), (66, "B", 1)),
}

# Byte offsets that llama_context_default_params() must produce for each
# context_params layout. Unlike the model_params reorder, this insertion
# shipped no accompanying marker SYMBOL (a plain struct field, not a new API),
# so layout detection here rests on the value fingerprint ALONE - there is no
# structural probe to corroborate or contradict it against.
#
# ctx_type sits immediately before the four-field run of long-stable
# UNSPECIFIED(-1) enums (rope_scaling_type/pooling_type/attention_type/
# flash_attn_type - see the keystone check below), 4 bytes earlier in V1 than
# in V2 (n_outputs_max_per_seq was inserted directly before n_threads,
# shifting everything from n_threads onward by +4 - see _structs' module
# docstring). ctx_type's own default is a small enum ("set the context type
# e.g. MTP") that does NOT use the -1 UNSPECIFIED convention the other four
# do (measured 0 on a real b10360 build), so "the byte immediately before a
# run of -1s is itself NOT -1" reliably locates which offset ctx_type is
# actually at, hence which layout is loaded - each entry below is
# (ctx_type_offset, first_offset_of_the_-1_run).
#
# ALL FOUR of the run's fields are checked, not three: an earlier version
# checked only rope_scaling_type/pooling_type/attention_type and MISDETECTED
# a real-shaped V1 struct as V2 whenever exactly rope_scaling_type (V1's
# FIRST run field, which is also V2's ctx_type position) was the one
# corrupted - virtually any non-(-1) value there simultaneously weakens V1's
# own run AND satisfies V2's "ctx_type looks plausible" check, while leaving
# V2's own three fields (which V1's corruption never touches) fully intact.
# Caught by a test proving verify_abi() still refuses a genuinely corrupted
# keystone (test_keystone_enum_drift_refuses[rope_scaling_type-0]) once the
# test double was fixed to faithfully reinterpret bytes through whatever
# class detection actually selects, instead of always handing back the
# original (correctly-typed) fixture object regardless of restype.
_CONTEXT_FINGERPRINT = {
    CONTEXT_PARAMS_V1: (32, 36),
    CONTEXT_PARAMS_V2: (36, 40),
}

# ctx_type's own default, and it is LOAD-BEARING rather than cosmetic - see
# _fingerprint_context_layout's "the one asymmetry" note for why grading on
# this exact value is the only signal that separates the two hypotheses at
# the ONE offset that distinguishes them.
#
# MEASURED, both layouts, not inferred:
#   * ctx_v1: probed live 2026-08-12 off this box's provisioned lemonade
#     b1307 build. llama_context_default_params() int32s read
#     [24] n_threads = 4, [28] n_threads_batch = 4, [32] ctx_type = 0,
#     [36..48] rope/pooling/attention/flash = -1, [52] rope_freq_base = 0.0f.
#   * ctx_v2: reported 0 on a real ggml-org b10360 build (the same
#     measurement the "does NOT use the -1 UNSPECIFIED convention" note
#     above rests on).
_CTX_TYPE_DEFAULT = 0


def _fingerprint_context_layout(raw: bytes) -> Optional[str]:
    """Which llama_context_params layout *raw* is consistent with, or None."""
    scores, run_hits = {}, {}
    for layout, (ctx_off, run_off) in _CONTEXT_FINGERPRINT.items():
        try:
            ctx_val = struct.unpack_from("<i", raw, ctx_off)[0]
            run = struct.unpack_from("<iiii", raw, run_off)
        except struct.error:
            return None
        run_hits[layout] = sum(v == -1 for v in run)
        if ctx_val == _CTX_TYPE_DEFAULT:
            ctx_pts = 2      # a real ctx_type; a thread count is never 0
        elif ctx_val != -1:
            ctx_pts = 1      # still consistent with "not part of the -1 run"
        else:
            ctx_pts = 0      # contradicted: this offset IS part of the run
        scores[layout] = ctx_pts + run_hits[layout]
    best, runner = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    # The run gate is what stops the ctx_type grade from manufacturing
    # confidence on its own: a winner must still see at least three of its
    # four keystone enums (one drifted default allowed), and be strictly
    # ahead overall.
    if run_hits[best[0]] < 3 or best[1] <= runner[1]:
        return None
    return best[0]

# ggml versions that BRACKET the llama_sampler_init_penalties signature change
# (upstream 935cad6497e8, 2026-08-04 06:02Z, which prepended an int32 n_vocab).
# Two bounds, because one alone leaves a large window needlessly undecidable.
#
# UPPER, sufficient: ggml was bumped to 0.18.1 in 15831f579a70 at 08:54Z the
# same day, i.e. AFTER the signature change. So ggml >= 0.18.1 PROVES 5-arg.
#
# LOWER, necessary: ggml went 0.17.0 -> 0.18.0 at upstream release b10192
# (2026-07-30), five days BEFORE the signature change. So ggml < 0.18.0 proves
# the build predates it, i.e. PROVES 4-arg.
#
# MEASURED against real upstream release headers on 2026-08-05, not inferred:
#     b10103  ggml 0.17.0  layout v1  4-arg   <- last pre-reorder release
#     b10105  ggml 0.17.0  layout v2  4-arg   <- reorder lands, ggml unchanged
#     b10178  ggml 0.17.0  layout v2  4-arg
#     b10191  ggml 0.17.0                     <- last 0.17.0
#     b10192  ggml 0.18.0  layout v2  4-arg   <- ggml bump
#     b10252  ggml 0.18.0  layout v2  4-arg
#     b10276  ggml 0.18.1  layout v2  5-arg
# Sampled builds with ggml < 0.18.0 that were 5-arg: ZERO.
#
# Only ggml == 0.18.0 straddles the change, and that window alone is reported
# UNKNOWN. Its exact extent, measured release by release: b10192 (first 0.18.0)
# through b10262 (last 0.18.0 that was actually tagged - b10263 and b10264 never
# were; b10265 is the first 0.18.1).
#
# Within that window the signature itself flips at b10258, the merge of
# ggml-org/llama.cpp#26520 - so b10192..b10257 really are 4-arg and
# b10258..b10262 really are 5-arg. localm reports UNKNOWN for BOTH halves,
# because ggml 0.18.0 does not distinguish them and nothing else in the binary
# does either. That is conservative by roughly five releases at the top end: it
# costs them the repetition-penalty sampler rather than risk a mis-marshalled
# call, which is the intended trade.
#
# Without the LOWER bound, the ~87 V2 releases b10105..b10191 would ALSO be
# called unknown, needlessly - their 4-arg form is provable.
_PENALTIES_5ARG_GGML = (0, 18, 1)
_PENALTIES_4ARG_GGML_BELOW = (0, 18, 0)


class _RawParams(ctypes.Structure):
    """A size-agnostic receptacle for a by-value *_default_params() return."""

    _fields_ = [("b", ctypes.c_uint8 * 256)]

# Misaligned-read tripwire magnitudes (NON-FATAL): a pointer or garbage landing in
# a numeric field reads as a huge value. These are heuristics far above any real
# default, NOT structural invariants, so EXCEEDING them is a diagnostic note, never
# a refusal - a future build may legitimately default higher and must still load.
_MAX_BATCH = 1 << 20      # 1,048,576 tokens
_MAX_CTX = 1 << 24        # 16,777,216 tokens


class AbiMismatch(LocalmError):
    """The loaded llama runtime's struct layout is not the one this build binds."""


@dataclass
class AbiVerdict:
    """Result of an ABI self-check. ``ok`` is True unless drift was proven."""

    status: str                                            # ok|mismatch|skipped|unchecked
    failures: List[str] = field(default_factory=list)      # structural checks that failed
    diagnostics: List[str] = field(default_factory=list)   # value drift notes (not fatal)
    detail: str = ""                                       # human one-liner
    layout: str = ""                                       # MODEL_PARAMS_V1 / _V2
    context_layout: str = ""                                # CONTEXT_PARAMS_V1 / _V2

    @property
    def ok(self) -> bool:
        # Only a PROVEN mismatch is not-ok. skipped (env) and unchecked
        # (mechanism error) both fail open - the check could not condemn the lib.
        return self.status != "mismatch"


# ---------------------------------------------------------------------------
#  llama_model_params layout detection
# ---------------------------------------------------------------------------

def _has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


def _read_raw(lib: ctypes.CDLL, fn_name: str) -> bytes:
    """Return the raw bytes a by-value ``*_default_params()`` writes."""
    fn = getattr(lib, fn_name)
    fn.restype = _RawParams
    fn.argtypes = []
    return bytes(bytearray(fn().b))


def _fingerprint_layout(raw: bytes) -> Optional[str]:
    """Which layout the default-params BYTES are consistent with, or None."""
    scores = {}
    for layout, checks in _FINGERPRINT.items():
        try:
            scores[layout] = sum(
                struct.unpack_from("<" + fmt, raw, off)[0] == want
                for off, fmt, want in checks)
        except struct.error:
            return None
    best, runner = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    # A majority of this layout's checks AND strictly ahead of the other. On the
    # measured real builds this is 3-0 in both directions.
    return best[0] if best[1] >= 2 and best[1] > runner[1] else None


def detect_model_params_layout(
    lib: ctypes.CDLL,
) -> Tuple[str, List[str], Optional[str], bool]:
    """Decide which ``llama_model_params`` layout *lib* uses."""
    notes: List[str] = []

    present = [s for s in _V2_MARKER_SYMBOLS if _has_symbol(lib, s)]
    if len(present) == len(_V2_MARKER_SYMBOLS):
        by_symbol: Optional[str] = MODEL_PARAMS_V2
    elif not present:
        by_symbol = MODEL_PARAMS_V1
    else:
        by_symbol = None
        notes.append(
            f"llama_load_mode symbols partially present ({', '.join(present)}); "
            "the symbol-based layout probe is inconclusive")

    by_value: Optional[str] = None
    try:
        by_value = _fingerprint_layout(_read_raw(lib, "llama_model_default_params"))
    except Exception as e:  # noqa: BLE001 - probe failure must not condemn the lib
        notes.append(f"model_params fingerprint could not be read ({e})")

    if by_symbol and by_value and by_symbol != by_value:
        return by_symbol, notes, (
            f"the llama_load_mode symbols say {by_symbol} but "
            f"llama_model_default_params()'s bytes say {by_value}"), False

    layout = by_symbol or by_value
    assumed = layout is None
    if assumed:
        layout = MODEL_PARAMS_V1
        notes.append(
            "neither layout probe was conclusive; assuming the historical "
            f"{MODEL_PARAMS_V1} llama_model_params layout")
    elif by_value is None:
        notes.append(
            f"model_params layout {layout} rests on the symbol probe alone "
            "(the default-value fingerprint was inconclusive)")
    return layout, notes, None, assumed


def model_params_class(layout: str):
    """The ctypes class for *layout*."""
    return LlamaModelParamsV2 if layout == MODEL_PARAMS_V2 else LlamaModelParamsV1


def detect_context_params_layout(
    lib: ctypes.CDLL,
) -> Tuple[str, List[str], bool]:
    """Decide which ``llama_context_params`` layout *lib* uses."""
    notes: List[str] = []
    layout: Optional[str] = None
    try:
        layout = _fingerprint_context_layout(
            _read_raw(lib, "llama_context_default_params"))
    except Exception as e:  # noqa: BLE001 - probe failure must not condemn the lib
        notes.append(f"context_params fingerprint could not be read ({e})")

    assumed = layout is None
    if assumed:
        layout = CONTEXT_PARAMS_V1
        notes.append(
            "context_params layout probe was inconclusive; assuming the "
            f"historical {CONTEXT_PARAMS_V1} llama_context_params layout")
    return layout, notes, assumed


def context_params_class(layout: str):
    """The ctypes class for *layout*."""
    return (LlamaContextParamsV2 if layout == CONTEXT_PARAMS_V2
            else LlamaContextParamsV1)


def _read_default_params(
    lib: ctypes.CDLL, layout: str, context_layout: str,
) -> Tuple[object, object]:
    """Call the default-params functions directly off *lib*."""
    mfn = lib.llama_model_default_params
    mfn.restype = model_params_class(layout)
    mfn.argtypes = []
    cfn = lib.llama_context_default_params
    cfn.restype = context_params_class(context_layout)
    cfn.argtypes = []
    return mfn(), cfn()


def evaluate(mp, cp) -> AbiVerdict:
    """Score real default-params structs against the expected layout."""
    failures: List[str] = []
    diags: List[str] = []
    is_v2 = isinstance(mp, LlamaModelParamsV2)

    # --- keystone: the long-stable UNSPECIFIED = -1 enums (context_params) ---
    # LLAMA_{ROPE_SCALING,POOLING,ATTENTION}_TYPE_UNSPECIFIED have been -1 for
    # years; default_params sets them so the model's own config decides. Changing
    # them would override every model's trained settings, so upstream will not.
    # Three consecutive int32 reading exactly -1 under a shifted layout is
    # effectively impossible - this is the layout fingerprint. (On the V1
    # layout that is offsets 36/40/44; on V2, where upstream inserted a new
    # n_outputs_max_per_seq field before n_threads sometime between lemonade
    # b1307 and ggml-org b10360, offsets 40/44/48 - see _structs' docstring
    # for the full history. Named-field access below makes this check itself
    # oblivious to which offsets are actually in play.)
    for name in ("rope_scaling_type", "pooling_type", "attention_type"):
        val = getattr(cp, name)
        if val != -1:
            failures.append(
                f"context_params.{name} = {val} (expected -1, the long-stable "
                "UNSPECIFIED default)"
            )

    # --- structural invariants: true for ANY aligned build, value-drift safe ---
    # split_mode is a small enum; accept its FULL valid set (NONE/LAYER/ROW/TENSOR
    # = 0/1/2/3). A misaligned read landing here is overwhelmingly outside 0..3, so
    # this still catches drift without refusing a legitimate value (TENSOR=3 is a
    # real upstream enumerator - do not narrow this set without checking llama.h).
    if mp.split_mode not in (0, 1, 2, 3):
        failures.append(
            f"model_params.split_mode = {mp.split_mode} "
            "(expected a valid LLAMA_SPLIT_MODE: 0, 1, 2 or 3)"
        )
    # model_params used to be checked ONLY by split_mode, at offset 20 - which is
    # exactly the offset the V1 -> V2 reorder did NOT move. These two widen the
    # check to the fields around it.
    #
    # BE PRECISE ABOUT WHAT THEY DO AND DO NOT CATCH, because an overstated guard
    # is worse than none: they catch a MISALIGNED read (a pointer, a -1, or
    # garbage landing in these fields), which is what an unknown future reorder
    # or a shifted struct looks like. They CANNOT detect this build's V1-vs-V2
    # confusion on plausible defaults, and it is not an oversight that they do
    # not - MEASURED: V2's real defaults read through the V1 class give
    # main_gpu = 1 (actually load_mode), which is a legal device index; V1's read
    # through V2 give load_mode = 0, a legal LLAMA_LOAD_MODE_NONE. Both directions
    # return status ok with two soft diagnostics. There is no discriminator
    # available here, because every value involved is valid in both layouts.
    #
    # Choosing the RIGHT class is therefore detect_model_params_layout's job, not
    # this function's, and the thing that actually catches a wrong choice is the
    # probe CONTRADICTION (symbols vs value fingerprint). See
    # test_evaluate_cannot_discriminate_the_two_layouts, which pins this so the
    # stronger claim cannot creep back in.
    # The valid set MIRRORS llama.h and is imported, never spelled out here - a
    # second copy of an upstream enumerator list is a second thing to forget to
    # update, and forgetting it once already cost a live outage: upstream added
    # LLAMA_LOAD_MODE_AUTO = -1 between b10361 and b10373 and made it the
    # default, so every build from b10373 on was refused by this line for
    # reporting a value that was correct. The message renders the set rather
    # than restating it for the same reason.
    if is_v2 and mp.load_mode not in _VALID_LOAD_MODES:
        failures.append(
            f"model_params.load_mode = {mp.load_mode} "
            "(expected a valid LLAMA_LOAD_MODE: "
            f"{', '.join(str(v) for v in _VALID_LOAD_MODES)})"
        )
    # main_gpu is a device INDEX. llama_max_devices() is 16 on every build
    # localm has seen, so this bound is enormously generous; its job is to catch
    # a pointer or garbage landing here under a shifted layout, not to police
    # the value.
    if not (0 <= mp.main_gpu < 4096):
        failures.append(
            f"model_params.main_gpu = {mp.main_gpu} "
            "(expected a plausible device index, 0 <= i < 4096)"
        )
    # Ordering + lower bounds hold for ANY aligned build regardless of defaults:
    if not (1 <= cp.n_ubatch <= cp.n_batch):
        failures.append(
            f"context_params batch sizes implausible: n_ubatch={cp.n_ubatch}, "
            f"n_batch={cp.n_batch} (expected 1 <= n_ubatch <= n_batch)"
        )
    if cp.n_ctx < 1:
        failures.append(f"context_params.n_ctx = {cp.n_ctx} (expected >= 1)")
    if cp.n_seq_max < 1:
        failures.append(
            f"context_params.n_seq_max = {cp.n_seq_max} (expected >= 1)"
        )
    # Absolute magnitude bounds are a misaligned-read tripwire, NOT a structural
    # invariant or config policy, so a value above them is a NON-FATAL diagnostic
    # (a future build may default higher) - never a false refusal.
    if cp.n_batch > _MAX_BATCH or cp.n_ctx > _MAX_CTX:
        diags.append(
            f"unusually large window (n_batch={cp.n_batch}, n_ctx={cp.n_ctx}); "
            "if generation misbehaves, suspect ABI drift"
        )

    # --- diagnostics (NOT fatal): exact stable values observed on every shipped
    # build. A drift here without a structural failure means a legitimate build
    # changed a default - we still load, but note it for `doctor` / the log. ---
    checks = [
        ("context_params.n_ctx", cp.n_ctx, 512),
        ("context_params.n_batch", cp.n_batch, 2048),
        ("context_params.n_ubatch", cp.n_ubatch, 512),
        ("context_params.flash_attn_type", cp.flash_attn_type, -1),
        ("model_params.split_mode", mp.split_mode, 1),
        ("model_params.use_extra_bufts", bool(mp.use_extra_bufts), True),
    ]
    # The mmap default is expressed differently per layout, so name the field
    # that layout actually has rather than a field that may not exist.
    #
    # TWO typical values, not one, and this is what keeps the diagnostic worth
    # reading. Upstream's default moved from MMAP (1) to AUTO (-1) at b10373, so
    # a single expected value makes EVERY load on a current build emit drift
    # noise - and a diagnostic that always fires is how a real one gets ignored.
    # Both are genuine upstream defaults; anything else still reports.
    if is_v2:
        checks.append(("model_params.load_mode", mp.load_mode,
                       (LLAMA_LOAD_MODE_AUTO, LLAMA_LOAD_MODE_MMAP)))
    else:
        checks.append(("model_params.use_mmap", bool(mp.use_mmap), True))
    for label, got, exp in checks:
        expected = exp if isinstance(exp, tuple) else (exp,)
        if got not in expected:
            shown = " or ".join(str(e) for e in expected)
            diags.append(f"{label} = {got} (typical default {shown})")

    layout = MODEL_PARAMS_V2 if is_v2 else MODEL_PARAMS_V1
    context_layout = (CONTEXT_PARAMS_V2 if isinstance(cp, LlamaContextParamsV2)
                       else CONTEXT_PARAMS_V1)
    if failures:
        return AbiVerdict(
            status="mismatch", failures=failures, diagnostics=diags, layout=layout,
            context_layout=context_layout,
            detail=f"{len(failures)} structural ABI check(s) failed",
        )
    return AbiVerdict(
        status="ok", diagnostics=diags, layout=layout, context_layout=context_layout,
        detail="ok (drift noted)" if diags else "ok",
    )


def _log(msg: str, warn: bool = False) -> None:
    try:
        from localm.debuglog import logger
        (logger.warning if warn else logger.info)(msg)
    except Exception:
        pass


def _mismatch_error(verdict: AbiVerdict, lib_path: str = "") -> AbiMismatch:
    where = f" ({lib_path})" if lib_path else ""
    bullets = "\n".join(f"  - {f}" for f in verdict.failures)
    reason = (
        f"Loading the native llama runtime{where} was refused to avoid memory "
        "corruption: its struct layout is not the one this localm build binds. "
        "The runtime's default-params returned values this build does not "
        "recognize:\n"
        f"{bullets}\n"
        "This usually means the provisioned llama library is a different "
        "llama.cpp version than the ctypes layouts in this localm build.\n"
        "Fix it by re-provisioning a matching prebuilt:\n"
        "  localm setup-llama --force\n"
        "or point LLAMA_CPP_LIB at a compatible build. If you are sure this is a "
        f"false alarm, set {SKIP_ENV}=1 to bypass the check - and please report "
        "it so the binding can be updated."
    )
    return AbiMismatch(
        summary="the native llama runtime has an incompatible struct layout (ABI mismatch)",
        reason=reason,
        context={"operation": "load_lib", "abi_failures": verdict.failures,
                 "abi_diagnostics": verdict.diagnostics},
    )


# The verdict verify_abi actually reached, kept so `localm doctor` can report
# what HAPPENED rather than re-deriving a fresh one - see abi_report().
_last_verdict: "Optional[AbiVerdict]" = None

_detected_layout: Optional[str] = None
# True when _detected_layout is a FALLBACK rather than a determination. Kept
# separate from the layout itself because "v1" alone cannot express the
# difference, and reasoning onward from an assumption as if it were proof is
# exactly what penalties_arity must not do.
_layout_assumed: bool = False

# Independent axis, same caching contract - see model_params_layout /
# context_params_layout below and _structs' module docstring for why
# model_params and context_params are two separate V1/V2 decisions.
_detected_context_layout: Optional[str] = None
_context_layout_assumed: bool = False


def model_params_layout(lib: Optional[ctypes.CDLL] = None) -> str:
    """The ``llama_model_params`` layout of the loaded runtime."""
    global _detected_layout, _layout_assumed
    if _detected_layout is None:
        if lib is None:
            from ._loader import load_lib
            lib = load_lib()
        layout, _notes, _contra, assumed = detect_model_params_layout(lib)
        _detected_layout, _layout_assumed = layout, assumed
    return _detected_layout


def context_params_layout(lib: Optional[ctypes.CDLL] = None) -> str:
    """The ``llama_context_params`` layout of the loaded runtime."""
    global _detected_context_layout, _context_layout_assumed
    if _detected_context_layout is None:
        if lib is None:
            from ._loader import load_lib
            lib = load_lib()
        layout, _notes, assumed = detect_context_params_layout(lib)
        _detected_context_layout, _context_layout_assumed = layout, assumed
    return _detected_context_layout


def _ggml_version(lib: ctypes.CDLL) -> Optional[Tuple[int, ...]]:
    """``ggml_version()`` as a comparable tuple, or None if unavailable."""
    fn = None
    if _has_symbol(lib, "ggml_version"):
        fn = lib.ggml_version
    else:
        try:
            from ._loader import _ggml_dev_handles, _ggml_sym
            fn = _ggml_sym(_ggml_dev_handles(), "ggml_version")
        except Exception:  # noqa: BLE001 - a version probe must never break a load
            fn = None
    if fn is None:
        return None
    try:
        fn.restype = ctypes.c_char_p
        fn.argtypes = []
        raw = (fn() or b"").decode(errors="replace").strip()
        parts = raw.split("-")[0].split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:  # noqa: BLE001
        return None


_detected_arity: Optional[int] = None


def penalties_arity(lib: Optional[ctypes.CDLL] = None) -> int:
    """Argument count of ``llama_sampler_init_penalties``: 4, 5, or 0 = unknown."""
    global _detected_arity
    if _detected_arity is not None:
        return _detected_arity
    if lib is None:
        from ._loader import load_lib
        lib = load_lib()
    layout = model_params_layout(lib)
    if _layout_assumed:
        # The layout is a fallback, not a determination, so the "V1 implies
        # 4-arg" inference has nothing under it. Report unknown.
        _detected_arity = 0
    elif layout == MODEL_PARAMS_V1:
        _detected_arity = 4
    else:
        ver = _ggml_version(lib)
        if ver is None:
            _detected_arity = 0
        elif ver >= _PENALTIES_5ARG_GGML:
            _detected_arity = 5      # sufficient: the bump came after the change
        elif ver < _PENALTIES_4ARG_GGML_BELOW:
            _detected_arity = 4      # necessary: the build predates the change
        else:
            _detected_arity = 0      # ggml 0.18.0 alone straddles it
    return _detected_arity


def _remember(verdict: AbiVerdict) -> AbiVerdict:
    """Store *verdict* as this process's authoritative ABI result and return it."""
    global _last_verdict
    _last_verdict = verdict
    return verdict


def verify_abi(lib: ctypes.CDLL, lib_path: str = "") -> AbiVerdict:
    """Verify *lib*'s struct layout matches this build."""
    global _detected_layout, _layout_assumed
    global _detected_context_layout, _context_layout_assumed

    # Detection runs BEFORE the skip check and never raises. Which layout to
    # bind is not part of the safety CHECK: the escape hatch exists to let a
    # user past a false alarm, and it must not silently downgrade them to the
    # wrong struct class, which is the very corruption it is meant to work
    # around. `contradiction` is the one detection outcome that IS a refusal.
    #
    # Two INDEPENDENT layout decisions - model_params (dual-signal: symbol +
    # value) and context_params (value fingerprint only, no marker symbol
    # exists for its insertion) - see _structs' module docstring. Only
    # model_params has a `contradiction` outcome, because only it has two
    # signals that can disagree; context_params either resolves to a layout
    # or falls back, never disagrees with itself.
    layout, notes, contradiction, assumed = detect_model_params_layout(lib)
    _detected_layout = layout
    _layout_assumed = assumed
    for note in notes:
        _log(f"llama model_params layout probe: {note}", warn=True)

    context_layout, ctx_notes, ctx_assumed = detect_context_params_layout(lib)
    _detected_context_layout = context_layout
    _context_layout_assumed = ctx_assumed
    for note in ctx_notes:
        _log(f"llama context_params layout probe: {note}", warn=True)
    notes = notes + ctx_notes

    # Log the contradiction BEFORE the skip check, and carry it into the skipped
    # verdict. The escape hatch suppresses the REFUSAL, and it must not also
    # suppress the finding: _mismatch_error's own text invites the user to set
    # this env var when they believe the refusal is a false alarm, so the skip
    # path is exactly where a genuine contradiction is most likely to be read -
    # and it is the one case where the layout localm picked may be the wrong one.
    if contradiction:
        _log("llama model_params layout probes DISAGREE: " + contradiction
             + f"; proceeding with {layout} - if the model loads on the wrong "
               "GPU or memory behaviour looks wrong, this is why.", warn=True)

    if os.environ.get(SKIP_ENV):
        _log(f"{SKIP_ENV} is set - skipping the llama ABI self-check. A mismatched "
             "layout can corrupt memory; unset it once the runtime is known good.",
             warn=True)
        return _remember(AbiVerdict(
            status="skipped", layout=layout, context_layout=context_layout,
            diagnostics=notes + ([contradiction] if contradiction else []),
            detail=f"skipped via {SKIP_ENV}"))

    if contradiction:
        verdict = _remember(AbiVerdict(
            status="mismatch", failures=[contradiction], diagnostics=notes,
            layout=layout, context_layout=context_layout,
            detail="model_params layout probes disagree"))
        raise _mismatch_error(verdict, lib_path)

    try:
        mp, cp = _read_default_params(lib, layout, context_layout)
    except Exception as e:  # noqa: BLE001 - any mechanism failure must fail open
        _log(f"llama ABI self-check could not run ({e}); continuing unverified.",
             warn=True)
        return _remember(AbiVerdict(status="unchecked", layout=layout,
                                    context_layout=context_layout,
                                    detail=f"mechanism error: {e}"))

    verdict = _remember(evaluate(mp, cp))
    verdict.diagnostics = notes + verdict.diagnostics
    if verdict.status == "mismatch":
        _log("llama ABI mismatch: " + "; ".join(verdict.failures), warn=True)
        raise _mismatch_error(verdict, lib_path)
    if verdict.diagnostics:
        _log("llama ABI ok, with default drift: " + "; ".join(verdict.diagnostics))
    return verdict


def abi_report() -> AbiVerdict:
    """Best-effort ABI verdict for diagnostics (``localm doctor``)."""
    try:
        from ._loader import load_lib
    except Exception as e:  # noqa: BLE001
        return AbiVerdict(status="unchecked", detail=f"loader import failed: {e}")
    try:
        lib = load_lib()
    except AbiMismatch as e:
        return AbiVerdict(status="mismatch",
                          failures=list(e.context.get("abi_failures", [])),
                          detail="struct layout mismatch")
    except Exception as e:  # noqa: BLE001 - not provisioned / not loadable
        return AbiVerdict(status="unchecked", detail=f"runtime not loadable: {e}")
    # Prefer the verdict verify_abi actually reached over re-deriving one.
    #
    # Re-deriving loses the two statuses that are not conclusions about the
    # LIBRARY but about the CHECK: "skipped" (the user set LOCALM_SKIP_ABI_CHECK)
    # and "unchecked" (the probe itself errored). evaluate() can only return
    # ok/mismatch, so a bypassed check was being reported to `localm doctor` as
    # "native ABI: struct layout matches this build" - an affirmative claim that
    # the layout was VERIFIED, for a check that never ran. That is reporting
    # success for a step which did not happen (AGENTS.md rule 5), and it made
    # doctor's own "check skipped" branch unreachable.
    #
    # The stored verdict is also strictly more informative: verify_abi's carries
    # the layout, the probe notes and any contradiction, all of which the
    # re-derivation drops.
    if _last_verdict is not None:
        return _last_verdict
    try:
        layout = model_params_layout(lib)
        mp, cp = _read_default_params(lib, layout)
        return evaluate(mp, cp)
    except Exception as e:  # noqa: BLE001
        return AbiVerdict(status="unchecked", detail=str(e))
