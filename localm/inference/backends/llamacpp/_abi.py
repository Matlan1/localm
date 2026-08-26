# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime ABI self-check for the native llama.cpp struct layout.

The ctypes structs in :mod:`._structs` encode specific llama.cpp struct layouts
(field order + byte offsets). Two of them - ``LlamaModelParams*`` and
``LlamaContextParams`` - cross the FFI boundary BY VALUE
(``llama_load_model_from_file`` / ``llama_init_from_model``).

``llama_model_default_params()`` and ``llama_context_default_params()`` return
the structs BY VALUE with known defaults (no model, no GPU needed); they are
called once at load time and a structural fingerprint is confirmed to land where
this build expects. On drift the load is REFUSED with a typed, reportable
:class:`AbiMismatch`.

The refusal is driven by STRUCTURAL invariants (enum ranges, field ordering,
bounds) plus one value keystone - the long-stable ``*_UNSPECIFIED == -1`` enums -
NOT by exact-default values, so a legitimate build whose defaults drift still
loads. Two further valves:

  * the check fails OPEN: if its own mechanism errors (a symbol missing on a very
    old build, a call raising), it logs and ALLOWS the load;
  * ``LOCALM_SKIP_ABI_CHECK=1`` bypasses it entirely (logged loudly).

Offsets for these POD fields are commit-determined, not OS-determined (natural
alignment is identical on MS-x64 / SysV-x64 / arm64), so a given build matches on
every OS. (Tag namespaces: b1xxx are lemonade-sdk/llamacpp-rocm, b10xxx are
ggml-org/llama.cpp, and they collide.)

Upstream reordered ``llama_model_params`` IN PLACE at an unchanged 72-byte size,
so this module also DECIDES WHICH LAYOUT to bind
(:func:`detect_model_params_layout`) rather than assuming one. That decision is
not part of the safety check and is made even when the check is skipped.
"""

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

# Env var that disables the check. Any non-empty value enables the bypass.
SKIP_ENV = "LOCALM_SKIP_ABI_CHECK"

# The two llama_model_params layouts localm binds. Both are 72 bytes, so the
# reorder is not visible in sizeof.
MODEL_PARAMS_V1 = "v1"   # use_mmap/use_direct_io/use_mlock, main_gpu@24
MODEL_PARAMS_V2 = "v2"   # load_mode@24, main_gpu@28, load_mtp

# The two llama_context_params layouts localm binds. Both are 224 bytes, so the
# reorder is not visible in sizeof. Distinct namespace from MODEL_PARAMS_*; the
# two axes are not interchangeable.
CONTEXT_PARAMS_V1 = "ctx_v1"  # no n_outputs_max_per_seq
CONTEXT_PARAMS_V2 = "ctx_v2"  # n_outputs_max_per_seq@24 inserted

# Symbols that appear in llama.h in the same change as the V2 reorder: present
# together on a V2 build, absent together on a V1 build. This is the structural
# half of the layout decision; the value fingerprint below is the other half.
_V2_MARKER_SYMBOLS = ("llama_load_mode_from_str", "llama_load_mode_name")

# Byte offsets/values that llama_model_default_params() produces for each
# layout, used as a corroborating fingerprint rather than the primary signal.
# An inconclusive result is allowed; only a conclusive contradiction of the
# symbol probe counts as a mismatch.
#   V1: main_gpu@24 == 0, use_mmap@65 == 1, use_extra_bufts@69 == 1
#   V2: load_mode@24 == 1 (MMAP), check_tensors@65 == 0, use_extra_bufts@66 == 1
_FINGERPRINT = {
    MODEL_PARAMS_V1: ((24, "i", 0), (65, "B", 1), (69, "B", 1)),
    MODEL_PARAMS_V2: ((24, "i", 1), (65, "B", 0), (66, "B", 1)),
}

# Byte offsets that llama_context_default_params() produces for each
# context_params layout. This axis has no marker symbol, so detection rests on
# the value fingerprint alone.
#
# ctx_type sits immediately before the four-field run of UNSPECIFIED(-1) enums
# (rope_scaling_type/pooling_type/attention_type/flash_attn_type), 4 bytes
# earlier in V1 than in V2. ctx_type's own default is not -1, so "the int32
# immediately before a run of -1s is itself not -1" locates the layout. Each
# entry is (ctx_type_offset, first_offset_of_the_run_of_-1s).
#
# All four fields of the run are graded.
_CONTEXT_FINGERPRINT = {
    CONTEXT_PARAMS_V1: (32, 36),
    CONTEXT_PARAMS_V2: (36, 40),
}

# ctx_type's own default, graded by _fingerprint_context_layout.
_CTX_TYPE_DEFAULT = 0


def _fingerprint_context_layout(raw: bytes) -> Optional[str]:
    """Which llama_context_params layout *raw* is consistent with, or None.

    Scored per layout over ctx_type plus all FOUR -1 reads from the run
    immediately after it (rope_scaling_type/pooling_type/attention_type/
    flash_attn_type). A tie or weak signal is INCONCLUSIVE and must never be
    treated as a determination on its own; detect_context_params_layout falls
    back to CONTEXT_PARAMS_V1 when this returns None.

    ctx_type is scored GRADED (0/1/2) rather than as a plain "is not -1"
    boolean. The two hypotheses are the same pattern 4 bytes apart, so their -1
    runs OVERLAP: V1 expects -1 at 36/40/44/48, V2 at 40/44/48/52. Offsets
    40/44/48 are predicted "== -1" by BOTH and discriminate nothing; only 32, 36
    and 52 decide anything. A thread count is never 0 on a working build, so
    only a real ctx_type scores the extra mark, which keeps the margins
    symmetric.

    TWO KNOWN LIMITS:

    * ctx_v2 bytes whose ctx_type is itself -1 put a five-long -1 run under both
      windows, so no reading of these six offsets can locate the boundary; that
      case is an undetectable misbind.
    * With TWO simultaneous drifts the result is unreliable.
    """
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
    # A winner must still see at least three of its four keystone enums (one
    # drifted default allowed) and be strictly ahead overall.
    if run_hits[best[0]] < 3 or best[1] <= runner[1]:
        return None
    return best[0]

# ggml version bounds bracketing the llama_sampler_init_penalties signature
# change, which prepended an int32 n_vocab:
#   ggml >= 0.18.1  proves 5-arg
#   ggml <  0.18.0  proves 4-arg
#   ggml == 0.18.0  straddles the change and is reported unknown
_PENALTIES_5ARG_GGML = (0, 18, 1)
_PENALTIES_4ARG_GGML_BELOW = (0, 18, 0)


class _RawParams(ctypes.Structure):
    """A size-agnostic receptacle for a by-value *_default_params() return.

    Both params structs are returned via a hidden pointer on every 64-bit ABI
    localm targets (they are far larger than a register), so handing the callee
    a larger buffer than it writes is safe and lets the raw bytes be read
    without committing to a layout first."""

    _fields_ = [("b", ctypes.c_uint8 * 256)]

# Misaligned-read tripwire magnitudes (NON-FATAL): exceeding them yields a
# diagnostic note, never a refusal.
_MAX_BATCH = 1 << 20      # 1,048,576 tokens
_MAX_CTX = 1 << 24        # 16,777,216 tokens


class AbiMismatch(LocalmError):
    """The loaded llama runtime's struct layout is not the one this build binds.

    A :class:`~localm.bugreport.LocalmError` so the CLI's single graceful handler
    turns it into a "sorry, X because Y" message and offers a bug report."""


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
        # Only a proven mismatch is not-ok; skipped (env) and unchecked
        # (mechanism error) both fail open.
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
    """Return the raw bytes a by-value ``*_default_params()`` writes.

    Read into an over-sized buffer so no layout has to be assumed first. Raises
    on a mechanism failure so callers can fail open."""
    fn = getattr(lib, fn_name)
    fn.restype = _RawParams
    fn.argtypes = []
    return bytes(bytearray(fn().b))


def _fingerprint_layout(raw: bytes) -> Optional[str]:
    """Which layout the default-params BYTES are consistent with, or None.

    SCORED, not all-or-nothing: on real bytes the wrong layout scores 0/3 while
    the right one scores 3/3, so one drifted default still leaves a 2-0 winner.

    Inconclusive (None) is a legitimate answer for a genuine tie or a weak
    winner, and must never be upgraded into a refusal on its own. It is only
    ever combined with the symbol probe.
    """
    scores = {}
    for layout, checks in _FINGERPRINT.items():
        try:
            scores[layout] = sum(
                struct.unpack_from("<" + fmt, raw, off)[0] == want
                for off, fmt, want in checks)
        except struct.error:
            return None
    best, runner = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    # A majority of this layout's checks, and strictly ahead of the other.
    return best[0] if best[1] >= 2 and best[1] > runner[1] else None


def detect_model_params_layout(
    lib: ctypes.CDLL,
) -> Tuple[str, List[str], Optional[str], bool]:
    """Decide which ``llama_model_params`` layout *lib* uses.

    Returns ``(layout, notes, contradiction, assumed)``. ``assumed`` is True
    when NEITHER probe was conclusive and the historical V1 layout was taken as
    a fallback - callers must not treat that as a determination (see
    :func:`penalties_arity`, which downgrades to "unknown" rather than reasoning
    from an assumption). Never raises: a mechanism failure yields the fallback
    plus a note.

    Two INDEPENDENT signals:

    * structural - the presence of the ``llama_load_mode_*`` helper symbols,
      which llama.h introduced together with the reorder;
    * value - the default-params byte fingerprint.

    The structural signal decides. The value signal can agree, be inconclusive
    (defaults are allowed to drift), or CONTRADICT. A contradiction is returned
    to the caller, which turns it into a refusal.
    """
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
    """Decide which ``llama_context_params`` layout *lib* uses.

    Returns ``(layout, notes, assumed)``. Unlike
    :func:`detect_model_params_layout`, there is no accompanying marker SYMBOL
    for the ``n_outputs_max_per_seq`` insertion (a plain struct field, not a
    new API), so this rests on the value fingerprint alone - a single signal,
    not two independent ones to cross-check. ``assumed`` is True when the
    fingerprint was inconclusive and CONTEXT_PARAMS_V1 was taken as a fallback -
    callers must not treat that as a determination, same caveat as
    :func:`detect_model_params_layout`'s ``assumed``. Never raises: a
    mechanism failure yields the fallback plus a note."""
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
    """Call the default-params functions directly off *lib*.

    Bound on the handle (not via :mod:`._api`) so this never re-enters
    ``load_lib``. Raises on a mechanism failure (symbol missing / call error) so
    the caller can fail open."""
    mfn = lib.llama_model_default_params
    mfn.restype = model_params_class(layout)
    mfn.argtypes = []
    cfn = lib.llama_context_default_params
    cfn.restype = context_params_class(context_layout)
    cfn.argtypes = []
    return mfn(), cfn()


def evaluate(mp, cp) -> AbiVerdict:
    """Score real default-params structs against the expected layout.

    Refusal (``status == "mismatch"``) is driven only by structural invariants +
    the -1 enum keystone, all of which hold for ANY correctly aligned build
    regardless of default-value drift. Exact stable values are recorded as
    non-fatal diagnostics.

    The layout of *mp* AND *cp* is taken from their classes (model_params
    V1/V2, context_params V1/V2 - independent axes), so every check below reads
    each field at the offset its actual bound layout uses. The checks name
    fields, never raw offsets:
    ``getattr(cp, name)`` resolves correctly regardless of which of the two
    context_params layouts *cp* actually is."""
    failures: List[str] = []
    diags: List[str] = []
    is_v2 = isinstance(mp, LlamaModelParamsV2)

    # --- keystone: the long-stable UNSPECIFIED = -1 enums (context_params) ---
    # rope_scaling_type/pooling_type/attention_type read -1 on any aligned
    # build. Named-field access keeps this check oblivious to which layout's
    # offsets are in play.
    for name in ("rope_scaling_type", "pooling_type", "attention_type"):
        val = getattr(cp, name)
        if val != -1:
            failures.append(
                f"context_params.{name} = {val} (expected -1, the long-stable "
                "UNSPECIFIED default)"
            )

    # --- structural invariants: true for ANY aligned build, value-drift safe ---
    # split_mode is a small enum; the full valid set is NONE/LAYER/ROW/TENSOR
    # = 0/1/2/3.
    if mp.split_mode not in (0, 1, 2, 3):
        failures.append(
            f"model_params.split_mode = {mp.split_mode} "
            "(expected a valid LLAMA_SPLIT_MODE: 0, 1, 2 or 3)"
        )
    # These two catch a MISALIGNED read (a pointer, a -1, or garbage landing in
    # these fields). They cannot discriminate V1 from V2 on plausible defaults;
    # choosing the right class is detect_model_params_layout's job.
    # The valid load-mode set mirrors llama.h and is imported, never spelled out
    # here; the message renders the set rather than restating it.
    if is_v2 and mp.load_mode not in _VALID_LOAD_MODES:
        failures.append(
            f"model_params.load_mode = {mp.load_mode} "
            "(expected a valid LLAMA_LOAD_MODE: "
            f"{', '.join(str(v) for v in _VALID_LOAD_MODES)})"
        )
    # main_gpu is a device INDEX. The bound catches a pointer or garbage landing
    # here under a shifted layout rather than policing the value.
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
    # Absolute magnitude bounds are a misaligned-read tripwire, so a value above
    # them is a NON-FATAL diagnostic.
    if cp.n_batch > _MAX_BATCH or cp.n_ctx > _MAX_CTX:
        diags.append(
            f"unusually large window (n_batch={cp.n_batch}, n_ctx={cp.n_ctx}); "
            "if generation misbehaves, suspect ABI drift"
        )

    # --- diagnostics (NOT fatal): value drift is noted for `doctor` / the log
    # and the runtime still loads. ---
    checks = [
        ("context_params.n_ctx", cp.n_ctx, 512),
        ("context_params.n_batch", cp.n_batch, 2048),
        ("context_params.n_ubatch", cp.n_ubatch, 512),
        ("context_params.flash_attn_type", cp.flash_attn_type, -1),
        ("model_params.split_mode", mp.split_mode, 1),
        ("model_params.use_extra_bufts", bool(mp.use_extra_bufts), True),
        # kv_unified is opt-in (embedder.configure_embed_context is the one
        # caller that turns it on), so llama_context_default_params()'s own
        # default is False on known builds. A build that reports anything else
        # here is corroborating evidence that this offset, which V1 and V2 share
        # unchanged, is not landing on the real kv_unified field on THIS
        # runtime.
        ("context_params.kv_unified", bool(cp.kv_unified), False),
    ]
    # The mmap default is expressed differently per layout, so name the field
    # that layout actually has. Two values count as typical, AUTO and MMAP;
    # anything else reports.
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


# The verdict verify_abi reached, returned by abi_report().
_last_verdict: "Optional[AbiVerdict]" = None

_detected_layout: Optional[str] = None
# True when _detected_layout is a FALLBACK rather than a determination.
_layout_assumed: bool = False

# Independent axis, same caching contract as _detected_layout.
_detected_context_layout: Optional[str] = None
_context_layout_assumed: bool = False


def model_params_layout(lib: Optional[ctypes.CDLL] = None) -> str:
    """The ``llama_model_params`` layout of the loaded runtime.

    Resolved once per process. ``load_lib`` populates it through
    :func:`verify_abi`, including on the SKIP path."""
    global _detected_layout, _layout_assumed
    if _detected_layout is None:
        if lib is None:
            from ._loader import load_lib
            lib = load_lib()
        layout, _notes, _contra, assumed = detect_model_params_layout(lib)
        _detected_layout, _layout_assumed = layout, assumed
    return _detected_layout


def context_params_layout(lib: Optional[ctypes.CDLL] = None) -> str:
    """The ``llama_context_params`` layout of the loaded runtime.

    Resolved once per process, same caching contract as
    :func:`model_params_layout` (including on the SKIP path)."""
    global _detected_context_layout, _context_layout_assumed
    if _detected_context_layout is None:
        if lib is None:
            from ._loader import load_lib
            lib = load_lib()
        layout, _notes, assumed = detect_context_params_layout(lib)
        _detected_context_layout, _context_layout_assumed = layout, assumed
    return _detected_context_layout


def _ggml_version(lib: ctypes.CDLL) -> Optional[Tuple[int, ...]]:
    """``ggml_version()`` as a comparable tuple, or None if unavailable.

    ``ggml_version`` lives in the ggml base library; on the shipped builds it is
    reachable through the llama handle, but fall back to the ggml handles the
    loader already holds rather than assuming that."""
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
    """Argument count of ``llama_sampler_init_penalties``: 4, 5, or 0 = unknown.

    A newer upstream prepended an ``int32_t n_vocab``. It added no new symbol
    and changed no struct, so there is nothing structural to probe.

    Calling either arity against the other is unsafe: on MS-x64 the 4-arg call
    leaves the callee's ``penalty_last_n`` as whatever is in EDX, and that value
    SIZES A RING-BUFFER ALLOCATION; the 5-arg call against a 4-arg build leaves
    ``penalty_repeat`` as whatever is in XMM1. This function therefore never
    guesses - it returns 0 and the caller drops the penalties stage with a
    warning.

    What it can prove:

    * A DETERMINED V1 model_params implies 4 args: the layout reorder landed
      strictly before the penalties change, so no build can be V1 and 5-arg.
      "Determined" is load-bearing - when neither layout probe was conclusive,
      ``detect_model_params_layout`` FALLS BACK to V1, and an assumed layout
      yields 0, not 4.
    * ggml >= 0.18.1 implies 5 args. Sufficient, not necessary.
    * ggml < 0.18.0 implies 4 args. Necessary, not sufficient.

    The only undecidable window is ggml == 0.18.0, where the signature flips but
    no property of the BINARY reveals where, so both halves report 0.

    Cached per process like the layout: this is called once per GENERATION
    REQUEST, and the answer cannot change while a process holds one library
    handle.
    """
    global _detected_arity
    if _detected_arity is not None:
        return _detected_arity
    if lib is None:
        from ._loader import load_lib
        lib = load_lib()
    layout = model_params_layout(lib)
    if _layout_assumed:
        # A fallback layout does not support the "V1 implies 4-arg" inference,
        # so report unknown.
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
    """Store *verdict* as this process's authoritative ABI result and return it.

    Called on EVERY verify_abi outcome, including the one that then raises."""
    global _last_verdict
    _last_verdict = verdict
    return verdict


def verify_abi(lib: ctypes.CDLL, lib_path: str = "") -> AbiVerdict:
    """Verify *lib*'s struct layout matches this build. Raise on proven drift.

    Returns the verdict on success (status ok / skipped / unchecked). Raises
    :class:`AbiMismatch` only when the structural fingerprint is broken. Called
    once per process from ``load_lib`` (cached with the lib handle), so it adds
    no per-call overhead.

    BEHAVIOUR FOR A FUTURE THIRD (or Nth) LAYOUT this module does not yet
    know about. Detection can be wrong two ways: INCONCLUSIVE (the probes
    report ``assumed=True`` and fall back to V1) or CONFIDENTLY WRONG (an
    unknown layout resembles a known one inside the checked window).
    Whatever layout gets bound - detected OR assumed - is handed to
    :func:`evaluate`, which re-checks real field values at wherever that
    layout says they live. How much that second layer buys differs per axis:

    * ``context_params`` - evaluate() reads the -1 keystones by NAME, so a
      wrong bind reads them at the wrong offsets and a non-(-1) there
      refuses. Not always: bytes whose -1 run is longer than the four
      checked fields (a build whose ctx_type is itself -1) put both
      candidate windows inside one run, and every reading then satisfies
      evaluate().

    * ``model_params`` - the second layer buys NOTHING. evaluate()'s
      model_params checks are RANGE checks over fields whose plausible
      values are legal in BOTH layouts (V2's load_mode=1 read as V1's
      main_gpu is a valid device index; V1's main_gpu=0 read as V2's
      load_mode is a valid LLAMA_LOAD_MODE_NONE), so it returns ok with two
      soft diagnostics in either direction. An unknown third model_params
      layout can be detected confidently, pass evaluate(), and be bound and
      crossed over the FFI by value. What protects model_params is DETECTION
      being dual-signal (the llama_load_mode_* symbols AND the value
      fingerprint) with disagreement itself a refusal.

    So a new model_params layout needs its OWN detection signal; nothing
    downstream will catch a wrong choice on that axis."""
    global _detected_layout, _layout_assumed
    global _detected_context_layout, _context_layout_assumed

    # Detection runs BEFORE the skip check and never raises: the escape hatch
    # suppresses the refusal, not the choice of struct class. `contradiction` is
    # the one detection outcome that is itself a refusal.
    #
    # Two independent layout decisions: model_params (symbol probe plus value
    # fingerprint) and context_params (value fingerprint only). Only
    # model_params can report a contradiction.
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

    # Log the contradiction BEFORE the skip check and carry it into the skipped
    # verdict, so the escape hatch suppresses the refusal but not the finding.
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
    """Best-effort ABI verdict for diagnostics (``localm doctor``). Never raises.

    Distinguishes: not provisioned / not loadable (unchecked) vs proven mismatch
    vs ok. Loads the runtime in-process, the same load ``localm run``
    performs."""
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
    # Prefer the verdict verify_abi reached. It carries the "skipped" and
    # "unchecked" statuses, the layout, the probe notes and any contradiction,
    # none of which a fresh evaluate() can produce.
    if _last_verdict is not None:
        return _last_verdict
    try:
        layout = model_params_layout(lib)
        mp, cp = _read_default_params(lib, layout)
        return evaluate(mp, cp)
    except Exception as e:  # noqa: BLE001
        return AbiVerdict(status="unchecked", detail=str(e))
