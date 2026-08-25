# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: the compatibility patch set applied to localm's own, fully-owned ComfyUI checkout."""

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
# way, so a SKIPPED outcome is never a silent mystery (rule 5).
Transform = Callable[[str], Tuple[Optional[str], str]]


@dataclass(frozen=True)
class ComfyPatch:
    """One localm-owned edit to the managed ComfyUI checkout."""
    name: str
    description: str
    target: str
    transform: Transform


@dataclass(frozen=True)
class PatchOutcome:
    """What happened to one patch. ``ok`` is False only for FAILED - a patch that could not be applied safely; a deliberate no-op (SKIPPED/ABSENT) is ok."""
    name: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAILED


# --------------------------------------------------------------------------- #
#  Patch 1: make_locked_method_func plain-function tolerance (#12116).        #
# --------------------------------------------------------------------------- #

# The exact fragile access: ``method = getattr(type_obj, func).__func__`` on its own
# line. Indentation is captured and reproduced so the rewrite matches the surrounding
# style. Whitespace inside the call is tolerated; a trailing comment makes it NOT match
# (we only touch the shape we know exactly), so an unexpected variant is left alone.
_FRAGILE_RE = re.compile(
    r"^(?P<indent>[ \t]*)method[ \t]*=[ \t]*"
    r"getattr\([ \t]*type_obj[ \t]*,[ \t]*func[ \t]*\)\.__func__[ \t]*$",
    re.MULTILINE,
)


def _func_tolerant_transform(text: str) -> Tuple[Optional[str], str]:
    """Rewrite the fragile ``getattr(type_obj, func).__func__`` into the plain-function tolerant form."""
    def _repl(m: "re.Match") -> str:
        indent = m.group("indent")
        return (f"{indent}attr = getattr(type_obj, func)\n"
                f'{indent}method = attr.__func__ if hasattr(attr, "__func__") else attr')

    new_text, n = _FRAGILE_RE.subn(_repl, text)
    if n:
        return new_text, (f"rewrote {n} fragile make_locked_method_func access(es) to "
                          "tolerate a plain-function node")

    # No fragile access present. Say WHY we are not touching it (rule 5: no silent skip).
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
    """Write *text* to *target* atomically: a sibling temp file, then os.replace."""
    tmp = target.with_name(target.name + ".localm-patch.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(target))


def apply_patch(patch: ComfyPatch, managed_comfy_dir) -> PatchOutcome:
    """Apply one *patch* to the managed ComfyUI at *managed_comfy_dir*."""
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

    # Never write source that does not parse (rule 5: no corrupt-but-shipped facade).
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
    """Apply the localm patch set to the managed ComfyUI at *managed_comfy_dir*."""
    patches = PATCHES if patches is None else patches
    return [apply_patch(p, managed_comfy_dir) for p in patches]
