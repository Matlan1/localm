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
from typing import Optional


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
    lines: list[str]       # diff lines, each with leading ' ', '+', or '-'


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)

_EOL_RE = re.compile(r"(\r\n|\n|\r)")


def _split_lines(text: str) -> tuple[list[str], list[str]]:
    """Split *text* on CRLF, LF and CR only.

    Returns ``(texts, terms)``: each line without its terminator and the
    terminator it had (``""`` for an unterminated final line). Every other
    character, including the Unicode line separators ``str.splitlines`` honours,
    stays inside the line text.
    """
    parts = _EOL_RE.split(text)
    texts = parts[0:-1:2]
    terms = parts[1::2]
    if parts[-1] != "":
        texts.append(parts[-1])
        terms.append("")
    return texts, terms


def _parse_hunks(diff_text: str) -> list[_Hunk]:
    """Parse a unified diff string into a list of _Hunk objects.

    An empty line inside a hunk is a blank context line whose leading space was
    lost (``git apply`` accepts it the same way). It counts as context when more
    diff lines follow it in the hunk, or when the header counts still expect it.
    """
    hunks: list[_Hunk] = []
    current: Optional[_Hunk] = None
    new_count = 0
    in_hunk = False
    pending_blank = 0

    def _close() -> None:
        nonlocal pending_blank
        assert current is not None
        if pending_blank:
            seen_old = sum(1 for l in current.lines if l.startswith((" ", "-")))
            seen_new = sum(1 for l in current.lines if l.startswith((" ", "+")))
            room = min(current.old_count - seen_old, new_count - seen_new)
            current.lines.extend([" "] * max(0, min(pending_blank, room)))
        pending_blank = 0
        hunks.append(current)

    for line in _split_lines(diff_text)[0]:
        m = _HUNK_RE.match(line)
        if not in_hunk:
            # Outside a hunk: skip file headers and noise
            if line.startswith(("--- ", "+++ ")):
                continue
            if m:
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                current = _Hunk(old_start=int(m.group(1)), old_count=old_count, lines=[])
                in_hunk = True
        else:
            assert current is not None
            if m:
                # New hunk starts - save current and open next
                _close()
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                current = _Hunk(old_start=int(m.group(1)), old_count=old_count, lines=[])
            elif line == "":
                pending_blank += 1
            elif line.startswith((" ", "+", "-")):
                current.lines.extend([" "] * pending_blank)
                pending_blank = 0
                current.lines.append(line)
            elif line.startswith("\\"):
                # "\ No newline at end of file" - silently ignore
                pass
            # Unrecognised lines are silently skipped

    if current is not None:
        _close()

    return hunks


# ---------------------------------------------------------------------------
#  Fuzzy-match hunk position
# ---------------------------------------------------------------------------

def _old_side(hunk: _Hunk) -> list[str]:
    """Lines that must be present in the original (context + removed)."""
    return [l[1:] for l in hunk.lines if l.startswith((" ", "-"))]


def _find_hunk_position(
    orig: list[str],
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
    hunks = _parse_hunks(diff_text)
    if not hunks:
        raise PatchError("No hunks found in the diff string.")

    texts, terms = _split_lines(original_text)
    default_eol = terms[0] if terms and terms[0] else "\n"
    result: list[str] = list(texts)
    result_terms: list[Optional[str]] = list(terms)
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

        # A line that replaces an old one keeps that line's terminator; lines
        # beyond the old span take their terminator from the line above.
        replaced_terms = result_terms[pos : pos + old_count]
        new_terms = [
            replaced_terms[i] if i < len(replaced_terms) else None
            for i in range(len(new_lines))
        ]
        result[pos : pos + old_count] = new_lines
        result_terms[pos : pos + old_count] = new_terms
        offset += len(new_lines) - old_count

    ended_with_newline = bool(terms) and terms[-1] != ""
    if not result:
        return default_eol if ended_with_newline else ""

    last = len(result) - 1
    pieces: list[str] = []
    prev_term = default_eol
    for i, line in enumerate(result):
        term = result_terms[i]
        if i == last and not ended_with_newline:
            term = ""
        elif not term:
            term = prev_term
        prev_term = term or prev_term
        pieces.append(line + term)
    return "".join(pieces)
