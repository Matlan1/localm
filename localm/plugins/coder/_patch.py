# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Pure-Python unified diff applier.

Handles standard ``patch -u`` output:

    --- a/path/to/file
    +++ b/path/to/file
    @@ -OLD_START[,OLD_COUNT] +NEW_START[,NEW_COUNT] @@
     context line
    -removed line
    +added line

File-header lines (``---``/``+++``) and ``\\ No newline at end of file``
markers are tolerated but ignored.  Line numbers in ``@@`` headers are
used only as *hints* - the applier does a fuzzy search (±20 lines) so
that minor off-by-one errors from the LLM are silently corrected.

Public API
----------
apply_diff(original_text, diff_text) -> str
    Raises ``PatchError`` on any unrecoverable failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
#  Exceptions
# ---------------------------------------------------------------------------

class PatchError(ValueError):
    """Raised when a hunk cannot be applied."""


# ---------------------------------------------------------------------------
#  Internal hunk representation
# ---------------------------------------------------------------------------

@dataclass
class _Hunk:
    old_start: int         # 1-indexed hint from @@ header
    old_count: int         # expected number of old-side lines (context + removals)
    lines: List[str]       # diff lines, each with leading ' ', '+', or '-'


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def _parse_hunks(diff_text: str) -> List[_Hunk]:
    """Parse a unified diff string into a list of _Hunk objects."""
    hunks: List[_Hunk] = []
    current: Optional[_Hunk] = None
    in_hunk = False

    for line in diff_text.splitlines():
        if not in_hunk:
            # Outside a hunk: skip file headers and noise
            if line.startswith(("--- ", "+++ ")):
                continue
            m = _HUNK_RE.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                current = _Hunk(old_start=old_start, old_count=old_count, lines=[])
                in_hunk = True
        else:
            # Inside a hunk
            m = _HUNK_RE.match(line)
            if m:
                # New hunk starts - save current and open next
                assert current is not None
                hunks.append(current)
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                current = _Hunk(old_start=old_start, old_count=old_count, lines=[])
            elif line.startswith((" ", "+", "-")):
                assert current is not None
                current.lines.append(line)
            elif line.startswith("\\"):
                # "\ No newline at end of file" - silently ignore
                pass
            # Blank / unrecognised lines are silently skipped

    if current is not None:
        hunks.append(current)

    return hunks


# ---------------------------------------------------------------------------
#  Fuzzy-match hunk position
# ---------------------------------------------------------------------------

def _old_side(hunk: _Hunk) -> List[str]:
    """Lines that must be present in the original (context + removed)."""
    return [l[1:] for l in hunk.lines if l.startswith((" ", "-"))]


def _find_hunk_position(
    orig: List[str],
    hunk: _Hunk,
    *,
    fuzz: int = 20,
) -> Optional[int]:
    """
    Find the 0-indexed position in *orig* where *hunk*'s old-side matches.

    Tries the hint first, then searches ±fuzz lines around it.
    Returns None if no match is found.
    """
    old_lines = _old_side(hunk)
    n = len(old_lines)

    if n == 0:
        # Pure insertion - no context to match; trust the hint
        return max(0, hunk.old_start - 1)

    def matches(pos: int) -> bool:
        return (
            0 <= pos
            and pos + n <= len(orig)
            and orig[pos : pos + n] == old_lines
        )

    hint = hunk.old_start - 1  # convert to 0-indexed

    # Exact hint
    if matches(hint):
        return hint

    # Expanding search around the hint
    for delta in range(1, fuzz + 1):
        if matches(hint + delta):
            return hint + delta
        if matches(hint - delta):
            return hint - delta

    return None


# ---------------------------------------------------------------------------
#  Applier
# ---------------------------------------------------------------------------

def apply_diff(original_text: str, diff_text: str) -> str:
    """
    Apply *diff_text* (unified diff format) to *original_text*.

    Returns the patched text.
    Raises ``PatchError`` if a hunk cannot be applied.
    """
    # Detect and preserve line ending style
    eol = "\r\n" if "\r\n" in original_text else "\n"

    hunks = _parse_hunks(diff_text)
    if not hunks:
        raise PatchError("No hunks found in the diff string.")

    orig = original_text.splitlines()
    result: List[str] = list(orig)
    offset = 0  # cumulative line-count delta from hunks already applied

    for idx, hunk in enumerate(hunks):
        hint_with_offset = hunk.old_start - 1 + offset
        adjusted_hunk = _Hunk(
            old_start=hint_with_offset + 1,  # _find uses 1-indexed old_start
            old_count=hunk.old_count,
            lines=hunk.lines,
        )
        pos = _find_hunk_position(result, adjusted_hunk)
        if pos is None:
            old_ctx = "\n".join(_old_side(hunk)[:6])
            raise PatchError(
                f"Hunk {idx + 1} (near line {hunk.old_start}) could not be applied - "
                f"context does not match the file.\n"
                f"Expected:\n{old_ctx}\n"
                "Hint: re-read the file and regenerate the diff."
            )

        old_count = len([l for l in hunk.lines if l.startswith((" ", "-"))])
        new_lines = [l[1:] for l in hunk.lines if l.startswith((" ", "+"))]

        result[pos : pos + old_count] = new_lines
        offset += len(new_lines) - old_count

    # Preserve trailing newline behaviour of the original
    patched = eol.join(result)
    if original_text.endswith(("\n", "\r\n")) and not patched.endswith(("\n", "\r\n")):
        patched += eol

    return patched
