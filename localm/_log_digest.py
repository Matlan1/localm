# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a bug-report-ready digest of a localm debug log.

The previous ``_recent_log_tail`` (bugreport.py) took a blind last-N-lines cut
of the log file. That misses the actual failure whenever enough routine
activity follows it before the report is filed - issue #617's own report was
one delay away from this: the error was followed by minutes of routine
``GET /api/stats`` polling (every ~2.5s), and a longer wait would have pushed
it out of the tail entirely.

This module instead:
  * Groups raw lines into RECORDS - a line matching the standard
    ``TIMESTAMP LEVEL NAME: message`` format starts a new record; anything
    that does not match (a traceback frame, a wrapped message) is a
    CONTINUATION of the previous record, so a Python exception's full frames
    travel with the line that logged it.
  * Keeps EVERY WARNING/ERROR/CRITICAL record (or one containing a raw
    traceback) from the whole file, not just the most recent one - the point
    of a "digest" is that no error from the session goes missing.
  * Collapses a run of 3+ consecutive BENIGN records that are near-duplicates
    (same level/logger and message with the numbers masked out - so
    ``GET /api/stats -> 200 (7 ms, loop_lag=0.26s)`` repeated every few
    seconds collapses to one line + a repeat count) instead of listing each
    one - the routine noise a report almost never needs, especially when it
    is just "all is well" polling.
  * If the digest is still over the size budget, trims from the OLDEST
    benign content first and never drops an error record silently - if even
    that is not enough, it says plainly how many earlier errors were omitted
    rather than silently truncating them (AGENTS.md rule 5).
"""

from __future__ import annotations

import re
from typing import List, TypedDict


class LogRecord(TypedDict):
    level: str      # "" when the record did not start with a recognized header
    logger: str
    lines: List[str]


# The logger group starts with `[^:\s]`, not `[^:]`. Whitespace is a SUBSET of
# `[^:]`, so `\s+([^:]+)` let both quantifiers claim the same run and a log line
# with a long space run and no colon after it cost O(n^2): measured 0.011 / 0.185
# / 0.838s at 1,000 / 4,000 / 8,000 spaces. Requiring the logger's FIRST
# character to be neither a colon nor whitespace removes the overlap without
# narrowing what matches - a logger name never begins with a space.
_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (\w+)\s+([^:\s][^:]*): (.*)$"
)
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_ERROR_LEVELS = frozenset({"WARNING", "ERROR", "CRITICAL"})
_TRACEBACK_MARKER = "Traceback (most recent call last)"

# A run shorter than this is left expanded - collapsing "2 identical lines"
# saves nothing and just makes a short, already-readable log harder to follow.
_MIN_RUN_TO_COLLAPSE = 3


def parse_records(text: str) -> List[LogRecord]:
    """Group raw log lines into records (see module docstring)."""
    records: List[LogRecord] = []
    for line in text.splitlines():
        m = _LOG_LINE_RE.match(line)
        if m:
            records.append({"level": m.group(1).upper(), "logger": m.group(2),
                            "lines": [line]})
        elif records:
            records[-1]["lines"].append(line)
        else:
            # The file (or a truncated tail) starts mid-record - keep the raw
            # line as its own record rather than silently dropping it.
            records.append({"level": "", "logger": "", "lines": [line]})
    return records


def is_error_record(rec: LogRecord) -> bool:
    """True for a WARNING+ record, or one carrying a raw traceback (a native
    crash or a print()'d exception can land in the log with no leveled
    prefix line of its own)."""
    if rec["level"] in _ERROR_LEVELS:
        return True
    return any(_TRACEBACK_MARKER in ln for ln in rec["lines"])


def record_template(rec: LogRecord) -> str:
    """A near-duplicate key: level + logger + message with every number
    masked out, so timestamps/latencies/counts that differ run to run do not
    stop two otherwise-identical lines from collapsing together."""
    first = rec["lines"][0]
    m = _LOG_LINE_RE.match(first)
    body = m.group(3) if m else first
    return f"{rec['level']}|{rec['logger']}|{_NUMERIC_RE.sub('#', body)}"


def _record_timestamp(rec: LogRecord) -> str:
    return rec["lines"][0][:19]   # "YYYY-MM-DD HH:MM:SS" prefix, or "" if absent


def collapse_records(records: List[LogRecord]) -> List[str]:
    """Render *records* to lines: every error record kept verbatim; a run of
    _MIN_RUN_TO_COLLAPSE+ consecutive near-duplicate BENIGN records collapses
    into its last occurrence + a repeat count and timespan."""
    out: List[str] = []
    i, n = 0, len(records)
    while i < n:
        rec = records[i]
        if is_error_record(rec):
            out.extend(rec["lines"])
            i += 1
            continue
        tmpl = record_template(rec)
        j = i + 1
        while (j < n and not is_error_record(records[j])
               and record_template(records[j]) == tmpl):
            j += 1
        run_len = j - i
        if run_len >= _MIN_RUN_TO_COLLAPSE:
            first_ts = _record_timestamp(records[i])
            last_ts = _record_timestamp(records[j - 1])
            span = f"{first_ts} .. {last_ts}" if first_ts and last_ts else f"x{run_len}"
            out.append(f"{records[j - 1]['lines'][0]}  (repeated {run_len}x, {span})")
        else:
            for k in range(i, j):
                out.extend(records[k]["lines"])
        i = j
    return out


def build_digest(text: str, *, max_chars: int = 6000) -> str:
    """The full pipeline: parse -> collapse near-duplicate benign runs ->
    keep every error in full -> fit *max_chars*, trimming benign content
    first and never silently dropping an error. Never raises (returns ""
    on any unexpected failure, matching the caller's existing best-effort
    contract)."""
    try:
        records = parse_records(text)
        if not records:
            return ""
        lines = collapse_records(records)
        digest = "\n".join(lines).strip()
        if len(digest) <= max_chars:
            return digest
        return _fit_budget(records, max_chars)
    except Exception:
        return ""


# Marker for an error kept only in part (see _fit_budget). The reader must never
# mistake a cut-down trace for a complete one and diagnose from it as if it were.
_TRUNCATED_MARK = ("... (this error was truncated for space - its start is in "
                   "the full log file) ...\n")
# Below this much room for actual error text, a tail is too small to be worth
# anything; fall back to declaring the error omitted instead.
_MIN_ERROR_TAIL = 40


def _fit_budget(records: List[LogRecord], max_chars: int) -> str:
    """Re-render, this time dropping benign (collapsed or not) lines from the
    FRONT first - oldest activity goes first, errors are never touched -
    until it fits. If the errors alone still exceed the budget, keep the
    MOST RECENT ones (most actionable) and say plainly how many earlier
    errors were omitted.

    The most recent error is kept even when it ALONE exceeds the budget, cut down
    to its tail and marked as truncated. Dropping it whole (keeping only blocks
    that fit entire) returned a digest of just the "N omitted" notice - a bug
    report with no error in it at all, for exactly the giant native-crash
    traceback this digest exists to carry (REG-619)."""
    error_idxs = [i for i, r in enumerate(records) if is_error_record(r)]
    error_blocks = ["\n".join(records[i]["lines"]) for i in error_idxs]
    errors_text = "\n".join(error_blocks)

    if len(errors_text) <= max_chars:
        # Errors fit; add back as much collapsed benign context as the
        # remaining budget allows, most-recent-first (drop the oldest first).
        benign_lines = collapse_records(
            [r for i, r in enumerate(records) if i not in set(error_idxs)])
        budget_left = max_chars - len(errors_text) - 2
        kept: List[str] = []
        for ln in reversed(benign_lines):
            if budget_left - len(ln) - 1 < 0:
                break
            kept.append(ln)
            budget_left -= len(ln) + 1
        kept.reverse()
        parts = []
        if kept:
            parts.append("\n".join(kept))
        parts.append(errors_text)
        return "\n".join(parts).strip()

    # Even the errors alone do not fit: keep the most recent ones (most
    # actionable) and say how many earlier ones were cut - never silent.
    kept_blocks: List[str] = []
    used = 0
    omitted = 0
    for block in reversed(error_blocks):
        room = max_chars - 80 - used - 1             # leave space for the notice
        if len(block) <= room:
            kept_blocks.append(block)
            used += len(block) + 1
            continue
        if not kept_blocks and room > len(_TRUNCATED_MARK) + _MIN_ERROR_TAIL:
            # This is the MOST RECENT error and it does not fit whole. Dropping it
            # (what the loop used to do with anything oversized) leaves the digest
            # with nothing but the "N omitted" notice and ZERO bytes of the actual
            # crash - strictly worse than the _recent_log_tail this replaced, which
            # always returned the last max_chars. Keep its TAIL: a traceback's
            # innermost exception type+message, the one line that names the
            # failure, is at the END (REG-619).
            tail = block[-(room - len(_TRUNCATED_MARK)):]
            kept_blocks.append(_TRUNCATED_MARK + tail)
            used += len(_TRUNCATED_MARK) + len(tail) + 1
            continue
        omitted += 1
    kept_blocks.reverse()
    notice = (f"... ({omitted} earlier error(s) omitted for space - see the "
              "full log file) ..." if omitted else "")
    parts = [notice] if notice else []
    parts.append("\n".join(kept_blocks))
    return "\n".join(p for p in parts if p).strip()
