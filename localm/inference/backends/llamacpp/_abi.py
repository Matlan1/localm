# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime ABI self-check for the native llama.cpp struct layout.

The ctypes structs in :mod:`._structs` encode ONE specific llama.cpp struct
layout (field order + byte offsets). Two of them - ``LlamaModelParams`` and
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
amd-rocm prebuilts localm provisions (commits b1288..b9740); offsets for these
POD fields are commit-determined, not OS-determined (natural alignment is
identical on MS-x64 / SysV-x64 / arm64), so a given build matches on every OS.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from localm.bugreport import LocalmError

from ._structs import LlamaContextParams, LlamaModelParams

# Env var that disables the check (escape hatch for a false alarm on a build the
# maintainer could not test). Any non-empty value enables the bypass.
SKIP_ENV = "LOCALM_SKIP_ABI_CHECK"

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

    @property
    def ok(self) -> bool:
        # Only a PROVEN mismatch is not-ok. skipped (env) and unchecked
        # (mechanism error) both fail open - the check could not condemn the lib.
        return self.status != "mismatch"


def _read_default_params(lib: ctypes.CDLL) -> Tuple[LlamaModelParams, LlamaContextParams]:
    """Call the default-params functions directly off *lib*.

    Bound on the handle (not via :mod:`._api`) so this never re-enters
    ``load_lib``. Raises on a mechanism failure (symbol missing / call error) so
    the caller can fail open."""
    mfn = lib.llama_model_default_params
    mfn.restype = LlamaModelParams
    mfn.argtypes = []
    cfn = lib.llama_context_default_params
    cfn.restype = LlamaContextParams
    cfn.argtypes = []
    return mfn(), cfn()


def evaluate(mp: LlamaModelParams, cp: LlamaContextParams) -> AbiVerdict:
    """Score real default-params structs against the expected layout.

    Refusal (``status == "mismatch"``) is driven only by structural invariants +
    the -1 enum keystone, all of which hold for ANY correctly aligned build
    regardless of default-value drift. Exact stable values are recorded as
    non-fatal diagnostics."""
    failures: List[str] = []
    diags: List[str] = []

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
    for label, got, exp in (
        ("context_params.n_ctx", cp.n_ctx, 512),
        ("context_params.n_batch", cp.n_batch, 2048),
        ("context_params.n_ubatch", cp.n_ubatch, 512),
        ("context_params.flash_attn_type", cp.flash_attn_type, -1),
        ("model_params.split_mode", mp.split_mode, 1),
        ("model_params.use_mmap", bool(mp.use_mmap), True),
    ):
        if got != exp:
            diags.append(f"{label} = {got} (typical default {exp})")

    if failures:
        return AbiVerdict(
            status="mismatch", failures=failures, diagnostics=diags,
            detail=f"{len(failures)} structural ABI check(s) failed",
        )
    return AbiVerdict(
        status="ok", diagnostics=diags,
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


def verify_abi(lib: ctypes.CDLL, lib_path: str = "") -> AbiVerdict:
    """Verify *lib*'s struct layout matches this build. Raise on proven drift.

    Returns the verdict on success (status ok / skipped / unchecked). Raises
    :class:`AbiMismatch` only when the structural fingerprint is broken. Called
    once per process from ``load_lib`` (cached with the lib handle), so it adds
    no per-call overhead."""
    if os.environ.get(SKIP_ENV):
        _log(f"{SKIP_ENV} is set - skipping the llama ABI self-check. A mismatched "
             "layout can corrupt memory; unset it once the runtime is known good.",
             warn=True)
        return AbiVerdict(status="skipped", detail=f"skipped via {SKIP_ENV}")

    try:
        mp, cp = _read_default_params(lib)
    except Exception as e:  # noqa: BLE001 - any mechanism failure must fail open
        _log(f"llama ABI self-check could not run ({e}); continuing unverified.",
             warn=True)
        return AbiVerdict(status="unchecked", detail=f"mechanism error: {e}")

    verdict = evaluate(mp, cp)
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
        mp, cp = _read_default_params(lib)
        return evaluate(mp, cp)
    except Exception as e:  # noqa: BLE001
        return AbiVerdict(status="unchecked", detail=str(e))
