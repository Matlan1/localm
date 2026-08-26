# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: the compatibility patch set applied to localm's own,
fully-owned ComfyUI checkout.

localm pins a known-good ComfyUI commit and carries a small, versioned set of
localm-owned patches ON TOP of it. Because this is localm's OWN, fully-owned
ComfyUI (never the user's install), a patch is a DIRECT core file edit, not the
in-memory shim the reactive T1 path uses on a user-owned ComfyUI. The patches are
applied right after provisioning (fresh and copy paths) and re-applied by
``localm comfy update`` whenever the pin advances.

The first (and today only) patch is the ``__func__`` tolerance: ComfyUI core's
``comfy_api/internal/__init__.py::make_locked_method_func`` does the bare
``getattr(type_obj, func).__func__``, assuming a node's FUNCTION is a bound method.
A node whose FUNCTION resolves to a plain function (core VAEDecodeAudio, used by
native ACE-Step music) has no ``.__func__`` and raises AttributeError. The patch
rewrites that one access to the tolerant ``attr = getattr(type_obj, func); method
= attr.__func__ if hasattr(attr, "__func__") else attr`` - the SAME change the T1
shim makes in memory (localm/media/comfy_shim/sitecustomize.py), here a persistent
file edit.

Every patch is:
  - GUARDED: it only rewrites when the exact known-fragile shape is present; an
    already-tolerant file (upstream fixed it, or localm patched it earlier) or an
    unexpected shape is left ALONE.
  - IDEMPOTENT: re-applying is a byte no-op (the fragile shape is gone after the
    first apply, so the second finds nothing to do).
  - FAIL-SAFE: a missing/renamed target is skipped WITHOUT error; a rewrite whose
    result would not parse is NEVER written (an ast.parse gate) and is reported
    failed; the write is atomic (temp file + os.replace) so a crash mid-write can
    never leave a half-written core file.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from localm.debuglog import logger

# Outcome statuses. Only FAILED is a non-ok result (a patch that could not be applied
# SAFELY); every other status is a safe outcome - APPLIED (rewrote the file), SKIPPED
# (present but no change needed), or ABSENT (target file not there).
APPLIED = "applied"
SKIPPED = "skipped"
ABSENT = "absent"
FAILED = "failed"

# A transform takes the current file TEXT and returns (new_text, note): new_text is
# None (or unchanged) for a deliberate no-op; the note explains the decision either
# way, so a SKIPPED outcome is never a silent mystery.
Transform = Callable[[str], Tuple[Optional[str], str]]


@dataclass(frozen=True)
class ComfyPatch:
    """One localm-owned edit to the managed ComfyUI checkout.

    ``target`` is the path RELATIVE to the managed ComfyUI root. ``transform`` decides,
    from the file's current text, whether and how to rewrite it (returning None for a
    guarded no-op)."""
    name: str
    description: str
    target: str
    transform: Transform


@dataclass(frozen=True)
class PatchOutcome:
    """What happened to one patch. ``ok`` is False only for FAILED - a patch that
    could not be applied safely; a deliberate no-op (SKIPPED/ABSENT) is ok."""
    name: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAILED


# --------------------------------------------------------------------------- #
#  Patch 1: make_locked_method_func plain-function tolerance.                 #
# --------------------------------------------------------------------------- #

# The exact fragile access: ``method = getattr(type_obj, func).__func__`` on its own
# line. Indentation is captured and re-emitted so the rewrite matches the
# surrounding style. Whitespace inside the call is tolerated; a trailing comment
# makes it NOT match, so an unexpected variant is left alone.
_FRAGILE_RE = re.compile(
    r"^(?P<indent>[ \t]*)method[ \t]*=[ \t]*"
    r"getattr\([ \t]*type_obj[ \t]*,[ \t]*func[ \t]*\)\.__func__[ \t]*$",
    re.MULTILINE,
)


def _func_tolerant_transform(text: str) -> Tuple[Optional[str], str]:
    """Rewrite the fragile ``getattr(type_obj, func).__func__`` into the plain-function
    tolerant form. Returns (new_text, note); new_text is None for a guarded no-op."""
    def _repl(m: "re.Match") -> str:
        indent = m.group("indent")
        return (f"{indent}attr = getattr(type_obj, func)\n"
                f'{indent}method = attr.__func__ if hasattr(attr, "__func__") else attr')

    new_text, n = _FRAGILE_RE.subn(_repl, text)
    if n:
        return new_text, (f"rewrote {n} fragile make_locked_method_func access(es) to "
                          "tolerate a plain-function node")

    # No fragile access present. The note names what was found, so a SKIPPED
    # outcome is never silent.
    if "make_locked_method_func" not in text:
        return None, "make_locked_method_func not found (renamed or restructured upstream)"
    if '"__func__"' in text or "'__func__'" in text:
        return None, "already tolerant (guarded __func__ access present)"
    return None, "make_locked_method_func present but not the known-fragile shape"


FUNC_PATCH = ComfyPatch(
    name="comfy_api-internal-func-tolerance",
    description=("make_locked_method_func: tolerate a node FUNCTION that is a plain "
                 "function (no __func__), fixing the ACE-Step/VAEDecodeAudio crash "
                 "(Comfy-Org/ComfyUI #12116)."),
    target="comfy_api/internal/__init__.py",
    transform=_func_tolerant_transform,
)

# The shipped, ordered patch set. Adding a localm patch = add a ComfyPatch here.
PATCHES: Tuple[ComfyPatch, ...] = (FUNC_PATCH,)


# --------------------------------------------------------------------------- #
#  Applying patches (guarded, idempotent, fail-safe).                          #
# --------------------------------------------------------------------------- #

def _atomic_write(target: Path, text: str) -> None:
    """Write *text* to *target* atomically: a sibling temp file, then os.replace. So a
    crash mid-write never leaves a half-written core file (the old file stays intact
    until the rename)."""
    tmp = target.with_name(target.name + ".localm-patch.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(target))


def apply_patch(patch: ComfyPatch, managed_comfy_dir) -> PatchOutcome:
    """Apply one *patch* to the managed ComfyUI at *managed_comfy_dir*. Never raises;
    always returns an honest PatchOutcome. A missing target is ABSENT (fail-safe skip);
    an unreadable file, an unparseable rewrite, or a failed write is FAILED and leaves
    the file untouched; a guarded no-op is SKIPPED; a real rewrite is APPLIED."""
    target = Path(managed_comfy_dir) / patch.target
    try:
        if not target.is_file():
            return PatchOutcome(patch.name, ABSENT, f"target not present: {patch.target}")
    except OSError as e:
        return PatchOutcome(patch.name, FAILED, f"could not stat {patch.target}: {e}")

    try:
        original = target.read_text(encoding="utf-8")
    except OSError as e:
        return PatchOutcome(patch.name, FAILED, f"could not read {patch.target}: {e}")

    try:
        new_text, note = patch.transform(original)
    except Exception as e:  # a broken transform must never corrupt the checkout
        logger.debug("comfy patch %s transform raised", patch.name, exc_info=True)
        return PatchOutcome(patch.name, FAILED, f"transform error: {e}")

    if new_text is None or new_text == original:
        return PatchOutcome(patch.name, SKIPPED, note)

    # Never write source that does not parse.
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return PatchOutcome(patch.name, FAILED,
                            f"rewrite did not parse, not written: {e}")

    try:
        _atomic_write(target, new_text)
    except OSError as e:
        return PatchOutcome(patch.name, FAILED, f"could not write {patch.target}: {e}")
    return PatchOutcome(patch.name, APPLIED, note)


def apply_patches(managed_comfy_dir, patches: Optional[Tuple[ComfyPatch, ...]] = None,
                  ) -> List[PatchOutcome]:
    """Apply the localm patch set to the managed ComfyUI at *managed_comfy_dir*.
    Returns one PatchOutcome per patch, in order. Never raises. Callers surface any
    FAILED outcome, but a failed COMPAT patch is non-fatal to the base install (it
    only breaks the workflow that needs it), so provisioning does not abort on it."""
    patches = PATCHES if patches is None else patches
    return [apply_patch(p, managed_comfy_dir) for p in patches]
