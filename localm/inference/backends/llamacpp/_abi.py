# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime ABI self-check for the native llama.cpp struct layout.

The ctypes structs in :mod:`._structs` encode specific llama.cpp struct layouts
(field order + byte offsets). Two of them - ``LlamaModelParams*`` and
``LlamaContextParams`` - cross the FFI boundary BY VALUE
(``llama_load_model_from_file`` / ``llama_init_from_model``). If the loaded
native library's real layout differs, ctypes marshals values into the wrong
offsets and the native side reads/writes the wrong memory: silent corruption or
a hard crash inside the GPU driver.

upstream llama.cpp does not promise ABI stability, so this module turns the
one-time manual "probing" that originally derived those layouts into an enforced
invariant. ``llama_model_default_params()`` and ``llama_context_default_params()``
return the structs BY VALUE with known defaults (no model, no GPU needed); we
call them once at load time and confirm a structural fingerprint lands where this
build expects. On drift we REFUSE to load - a clean, typed, reportable
:class:`AbiMismatch` - instead of letting a wrong layout corrupt memory.

Design priority (see issues/abi-verification-worklog.md): NEVER false-positive.
A false refusal would brick startup on hardware the maintainer cannot test, which
is worse than the status quo. So the refusal is driven by STRUCTURAL invariants
(enum ranges, field ordering, bounds) plus one value keystone - the long-stable
``*_UNSPECIFIED == -1`` enums - NOT by brittle exact-default values. A legitimate
build whose defaults drift still loads. Two more safety valves:

  * the check fails OPEN: if its own mechanism errors (a symbol missing on a very
    old build, a call raising), it logs and ALLOWS the load - the safety check
    must never itself become a new failure source;
  * ``LOCALM_SKIP_ABI_CHECK=1`` bypasses it entirely (logged loudly), so a false
    alarm on an untested build can never permanently block a user.

The fingerprint was validated byte-for-byte against the cpu, vulkan, and
amd-rocm prebuilts localm provisions; offsets for these POD fields are
commit-determined, not OS-determined (natural alignment is identical on MS-x64 /
SysV-x64 / arm64), so a given build matches on every OS. Live probes of the
shipped amd-rocm builds, 2026-08-05: b1288 reports ``ggml_commit() == "7c158fb"``
/ ggml 0.13.1 and the V1 layout; b1307 reports ``0713275`` / ggml 0.18.1 and V2.

Because upstream reordered ``llama_model_params`` IN PLACE at an unchanged
72-byte size, this module also DECIDES WHICH LAYOUT to bind
(:func:`detect_model_params_layout`) rather than assuming one. That decision is
not part of the safety check and is made even when the check is skipped: picking
the wrong struct class is the very corruption the escape hatch exists to work
around, so it must not be a side effect of using it.
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
    LlamaContextParams,
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

# Symbols that appear in llama.h in the same change as the V2 reorder. Probed at
# 8 upstream tags spanning the flip (b9870, b10000, b10050, b10080, b10090,
# b10180, b10200, b10276): the struct field and BOTH of these are always all
# present or all absent. This is the STRUCTURAL half of the layout decision; the
# value fingerprint below is an independent second opinion, so the decision does
# not rest on this correlation alone.
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

# ggml version at/after which llama_sampler_init_penalties takes the leading
# int32 n_vocab. upstream moved n_vocab into the sampler in 935cad6497e8
# (2026-08-04 06:02Z) and bumped ggml to 0.18.1 in 15831f579a70 (08:54Z the same
# day), i.e. AFTER. So this is a SUFFICIENT condition for the 5-argument form,
# never a necessary one - a build made in the ~3h between the two commits has
# ggml 0.18.0 and the 5-arg form, and is reported UNKNOWN rather than guessed at.
_PENALTIES_5ARG_GGML = (0, 18, 1)


class _RawParams(ctypes.Structure):
    """A size-agnostic receptacle for a by-value *_default_params() return.

    Both params structs are returned via a hidden pointer on every 64-bit ABI
    localm targets (they are far larger than a register), so handing the callee
    a larger buffer than it writes is safe and lets us read the raw bytes without
    committing to a layout first - which is the point, since the layout is what
    we are trying to determine."""

    _fields_ = [("b", ctypes.c_uint8 * 256)]

# Misaligned-read tripwire magnitudes (NON-FATAL): a pointer or garbage landing in
# a numeric field reads as a huge value. These are heuristics far above any real
# default, NOT structural invariants, so EXCEEDING them is a diagnostic note, never
# a refusal - a future build may legitimately default higher and must still load.
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
    """Return the raw bytes a by-value ``*_default_params()`` writes.

    Read into an over-sized buffer so no layout has to be assumed first. Raises
    on a mechanism failure so callers can fail open."""
    fn = getattr(lib, fn_name)
    fn.restype = _RawParams
    fn.argtypes = []
    return bytes(bytearray(fn().b))


def _fingerprint_layout(raw: bytes) -> Optional[str]:
    """Which layout the default-params BYTES are consistent with, or None.

    None means inconclusive - zero matches (a default drifted) or, in principle,
    both. Inconclusive is a legitimate answer and must not be upgraded into a
    refusal: see the module docstring's never-false-positive priority."""
    hits = []
    for layout, checks in _FINGERPRINT.items():
        try:
            if all(struct.unpack_from("<" + fmt, raw, off)[0] == want
                   for off, fmt, want in checks):
                hits.append(layout)
        except struct.error:
            return None
    return hits[0] if len(hits) == 1 else None


def detect_model_params_layout(
    lib: ctypes.CDLL,
) -> Tuple[str, List[str], Optional[str]]:
    """Decide which ``llama_model_params`` layout *lib* uses.

    Returns ``(layout, notes, contradiction)``. Never raises: a mechanism
    failure yields the historical V1 layout plus a note, because refusing to
    load over a failed probe would be a worse outcome than the status quo.

    Two INDEPENDENT signals:

    * structural - the presence of the ``llama_load_mode_*`` helper symbols,
      which llama.h introduced together with the reorder;
    * value - the default-params byte fingerprint.

    The structural signal decides. The value signal can agree, be inconclusive
    (fine - defaults are allowed to drift), or CONTRADICT. A contradiction is
    returned to the caller, which turns it into a refusal: when two independent
    probes disagree we genuinely do not know where ``main_gpu`` lives, and both
    possible guesses corrupt something silently.
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
            f"llama_model_default_params()'s bytes say {by_value}")

    layout = by_symbol or by_value
    if layout is None:
        layout = MODEL_PARAMS_V1
        notes.append(
            "neither layout probe was conclusive; assuming the historical "
            f"{MODEL_PARAMS_V1} llama_model_params layout")
    elif by_value is None:
        notes.append(
            f"model_params layout {layout} rests on the symbol probe alone "
            "(the default-value fingerprint was inconclusive)")
    return layout, notes, None


def model_params_class(layout: str):
    """The ctypes class for *layout*."""
    return LlamaModelParamsV2 if layout == MODEL_PARAMS_V2 else LlamaModelParamsV1


def _read_default_params(
    lib: ctypes.CDLL, layout: str,
) -> Tuple[object, LlamaContextParams]:
    """Call the default-params functions directly off *lib*.

    Bound on the handle (not via :mod:`._api`) so this never re-enters
    ``load_lib``. Raises on a mechanism failure (symbol missing / call error) so
    the caller can fail open."""
    mfn = lib.llama_model_default_params
    mfn.restype = model_params_class(layout)
    mfn.argtypes = []
    cfn = lib.llama_context_default_params
    cfn.restype = LlamaContextParams
    cfn.argtypes = []
    return mfn(), cfn()


def evaluate(mp, cp: LlamaContextParams) -> AbiVerdict:
    """Score real default-params structs against the expected layout.

    Refusal (``status == "mismatch"``) is driven only by structural invariants +
    the -1 enum keystone, all of which hold for ANY correctly aligned build
    regardless of default-value drift. Exact stable values are recorded as
    non-fatal diagnostics.

    The layout of *mp* is taken from its class, so the model_params checks below
    read the fields at the offsets that layout actually uses."""
    failures: List[str] = []
    diags: List[str] = []
    is_v2 = isinstance(mp, LlamaModelParamsV2)

    # --- keystone: the long-stable UNSPECIFIED = -1 enums (context_params) ---
    # LLAMA_{ROPE_SCALING,POOLING,ATTENTION}_TYPE_UNSPECIFIED have been -1 for
    # years; default_params sets them so the model's own config decides. Changing
    # them would override every model's trained settings, so upstream will not.
    # Three consecutive int32 reading exactly -1 at 36/40/44 under a shifted
    # layout is effectively impossible - this is the layout fingerprint.
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
    # exactly the offset the V1 -> V2 reorder did NOT move, so the reorder passed
    # this function with a single non-fatal diagnostic while `main_gpu` writes
    # went into `load_mode`. These two close that hole: they read fields whose
    # offsets DO differ between the layouts (load_mode/main_gpu at 24/28 vs
    # main_gpu/pad at 24), so binding the wrong class now fails structurally
    # instead of corrupting a load. Both are enum-range / bounds invariants that
    # hold for any correctly aligned build, not default values.
    if is_v2 and mp.load_mode not in _VALID_LOAD_MODES:
        failures.append(
            f"model_params.load_mode = {mp.load_mode} "
            "(expected a valid LLAMA_LOAD_MODE: 0, 1, 2, 3 or 4)"
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
    if is_v2:
        checks.append(("model_params.load_mode", mp.load_mode, 1))
    else:
        checks.append(("model_params.use_mmap", bool(mp.use_mmap), True))
    for label, got, exp in checks:
        if got != exp:
            diags.append(f"{label} = {got} (typical default {exp})")

    layout = MODEL_PARAMS_V2 if is_v2 else MODEL_PARAMS_V1
    if failures:
        return AbiVerdict(
            status="mismatch", failures=failures, diagnostics=diags, layout=layout,
            detail=f"{len(failures)} structural ABI check(s) failed",
        )
    return AbiVerdict(
        status="ok", diagnostics=diags, layout=layout,
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


_detected_layout: Optional[str] = None


def model_params_layout(lib: Optional[ctypes.CDLL] = None) -> str:
    """The ``llama_model_params`` layout of the loaded runtime.

    Resolved once per process. ``load_lib`` populates it through
    :func:`verify_abi` (including on the SKIP path, since which layout to bind
    is not a safety CHECK - binding the wrong one is exactly what the escape
    hatch must not cause)."""
    global _detected_layout
    if _detected_layout is None:
        if lib is None:
            from ._loader import load_lib
            lib = load_lib()
        _detected_layout = detect_model_params_layout(lib)[0]
    return _detected_layout


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

    upstream #26520 (``935cad6497e8``, 2026-08-04 06:02Z) prepended an
    ``int32_t n_vocab``. It added NO new symbol and changed no struct, so unlike
    the model_params reorder there is nothing structural to probe - diffing every
    ``LLAMA_API`` declaration across b10240 -> b10270 yields only the signature
    line itself.

    Calling either arity against the other is genuinely unsafe rather than merely
    wrong: on MS-x64 the 4-arg call leaves the callee's ``penalty_last_n`` as
    whatever is in EDX, and that value SIZES A RING-BUFFER ALLOCATION; the 5-arg
    call against a 4-arg build leaves ``penalty_repeat`` as whatever is in XMM1.
    So this function never guesses - it returns 0 and the caller drops the
    penalties stage with a warning.

    What it can prove:

    * V1 model_params implies 4 args. The layout reorder landed STRICTLY BEFORE
      the penalties change (measured by bisecting upstream tags: b10180 and
      b10240 are already V1->V2 flipped while still 4-arg), so no build can be
      V1 and 5-arg.
    * ggml >= 0.18.1 implies 5 args. That bump (``15831f579a70``, 08:54Z) came
      AFTER the penalties change the same day. Sufficient, not necessary.

    The gap is a build made in the ~3 hours between them, or any V2 build with
    ggml 0.18.0 (upstream ~b10180..b10269). Those report 0.

    Cached per process like the layout: this is called from ``_build_sampler``,
    i.e. once per GENERATION REQUEST, and re-reading ``ggml_version()`` (which
    also re-binds restype/argtypes on a shared function object) on every request
    would be per-token-stream overhead for an answer that cannot change while a
    process holds one library handle.
    """
    global _detected_arity
    if _detected_arity is not None:
        return _detected_arity
    if lib is None:
        from ._loader import load_lib
        lib = load_lib()
    if model_params_layout(lib) == MODEL_PARAMS_V1:
        _detected_arity = 4
    else:
        ver = _ggml_version(lib)
        _detected_arity = 5 if (ver is not None
                                and ver >= _PENALTIES_5ARG_GGML) else 0
    return _detected_arity


def verify_abi(lib: ctypes.CDLL, lib_path: str = "") -> AbiVerdict:
    """Verify *lib*'s struct layout matches this build. Raise on proven drift.

    Returns the verdict on success (status ok / skipped / unchecked). Raises
    :class:`AbiMismatch` only when the structural fingerprint is broken. Called
    once per process from ``load_lib`` (cached with the lib handle), so it adds
    no per-call overhead."""
    global _detected_layout

    # Detection runs BEFORE the skip check and never raises. Which layout to
    # bind is not part of the safety CHECK: the escape hatch exists to let a
    # user past a false alarm, and it must not silently downgrade them to the
    # wrong struct class, which is the very corruption it is meant to work
    # around. `contradiction` is the one detection outcome that IS a refusal.
    layout, notes, contradiction = detect_model_params_layout(lib)
    _detected_layout = layout
    for note in notes:
        _log(f"llama model_params layout probe: {note}", warn=True)

    if os.environ.get(SKIP_ENV):
        _log(f"{SKIP_ENV} is set - skipping the llama ABI self-check. A mismatched "
             "layout can corrupt memory; unset it once the runtime is known good.",
             warn=True)
        return AbiVerdict(status="skipped", layout=layout,
                          detail=f"skipped via {SKIP_ENV}")

    if contradiction:
        verdict = AbiVerdict(
            status="mismatch", failures=[contradiction], diagnostics=notes,
            layout=layout, detail="model_params layout probes disagree")
        _log("llama ABI mismatch: " + contradiction, warn=True)
        raise _mismatch_error(verdict, lib_path)

    try:
        mp, cp = _read_default_params(lib, layout)
    except Exception as e:  # noqa: BLE001 - any mechanism failure must fail open
        _log(f"llama ABI self-check could not run ({e}); continuing unverified.",
             warn=True)
        return AbiVerdict(status="unchecked", layout=layout,
                          detail=f"mechanism error: {e}")

    verdict = evaluate(mp, cp)
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
    vs ok. Loads the runtime in-process - the same load ``localm run`` performs -
    so a loadable library is safe here."""
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
    # load_lib already verified; re-evaluate for a detailed verdict (lib cached).
    try:
        layout = model_params_layout(lib)
        mp, cp = _read_default_params(lib, layout)
        return evaluate(mp, cp)
    except Exception as e:  # noqa: BLE001
        return AbiVerdict(status="unchecked", detail=str(e))
