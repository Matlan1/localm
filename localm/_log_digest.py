# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a bug-report-ready digest of a localm debug log.

This module:
  * Groups raw lines into RECORDS - a line matching the standard
    ``TIMESTAMP LEVEL NAME: message`` format starts a new record; anything
    that does not match (a traceback frame, a wrapped message) is a
    CONTINUATION of the previous record, so a Python exception's full frames
    travel with the line that logged it.
  * Keeps EVERY WARNING/ERROR/CRITICAL record (or one containing a raw
    traceback, or an unleveled continuation line that reads as a crash -
    see _CONTINUATION_ERROR_SIGNAL_RE) from the whole file, not just the
    most recent one. This also covers raw native (ggml/CUDA/HIP) stderr,
    which debuglog.py appends into the log file with no
    "TIMESTAMP LEVEL NAME:" prefix of its own and so always lands as a
    continuation of whatever benign record precedes it.
  * Collapses a run of 3+ consecutive BENIGN records that are near-duplicates
    END TO END - same level/logger and EVERY line's text with the numbers
    masked out, continuation lines included (see record_template) - into one
    line plus a repeat count. The survivor keeps ALL of its own lines.
  * If the digest is still over the size budget, trims from the OLDEST
    benign content first and never drops an error record silently - when
    even that is not enough it states how many earlier errors were omitted.
  * Drops every record that is a known debug_content_enabled()-gated write -
    e.g. the GGUF backend's raw (pre-scrub) model reply - BEFORE anything
    else touches it, so chat content can never survive collapsing, budget
    fitting, or being promoted to an ERROR record by its own text. See
    _CONTENT_MARKER_RES: a fixed marker on each KNOWN write site, not a
    scrubber over arbitrary prose.
  * Collapses a long RUN of near-duplicate lines WITHIN one benign record's
    own continuation lines, not just across records. Raw native (ggml/CUDA/
    HIP) stderr has no "TIMESTAMP LEVEL NAME:" prefix of its own, so a long
    stretch of it always glues onto ONE record as its continuation lines
    (see parse_records), and one giant record is never a "run" of 3+ near-
    duplicate RECORDS for the record-level collapse above to fold. See
    _collapse_line_runs.
"""

from __future__ import annotations

import re
from typing import List, TypedDict


class LogRecord(TypedDict):
    level: str      # "" when the record did not start with a recognized header
    logger: str
    lines: List[str]


# The logger group excludes whitespace from its first character as well as the
# colon, so the two quantifiers cannot both claim the same run of spaces. That
# keeps matching linear on a long space run with no following colon. A logger
# name never begins with a space.
_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (\w+)\s+([^:\s][^:]*): (.*)$"
)
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_ERROR_LEVELS = frozenset({"WARNING", "ERROR", "CRITICAL"})
_TRACEBACK_MARKER = "Traceback (most recent call last)"

# Raw native (ggml/CUDA/HIP) stderr is appended to the debug log with no
# "TIMESTAMP LEVEL NAME:" prefix, so parse_records() attaches it as a
# continuation of the preceding record, usually a routine DEBUG poll. This is a
# broader net alongside the exact traceback marker, so a crash line carrying no
# traceback marker is not collapsed away as a benign near-duplicate.
_CONTINUATION_ERROR_SIGNAL_RE = re.compile(
    r"\b(?:error|exception|fatal|crash(?:ed)?|segfault|segmentation fault|"
    r"core dumped|assert(?:ion)?\s+fail\w*|panic|abort(?:ed)?)\b",
    re.IGNORECASE,
)

# A run shorter than this is left expanded.
_MIN_RUN_TO_COLLAPSE = 3

# Debug-level writes gated on debuglog.debug_content_enabled(), i.e. carrying raw
# chat content rather than operational data. A record matching any of these is
# dropped whole in build_digest, before collapsing or error-promotion. Each
# pattern anchors to a known write site's stable prefix, not to prose content.
_CONTENT_MARKER_RES = (
    # llama.py's _decode_stream() logs "raw model output:" and the reply. The
    # reply rides in as unleveled continuation lines, so anchoring to
    # end-of-line is exact.
    re.compile(r"raw model output:\s*$"),
    # memory/store.py's _embed_one(): the content-bearing branch logs
    # "memory embed_one failed for %r: %s" with the snippet inline. Its
    # privacy-mode sibling logs a different, content-free message.
    re.compile(r"\bmemory embed_one failed for "),
    # jobs/webtool.py's scheduled web-tool loop: the content-bearing branch logs
    # "jobs web tool: %s %s" (tool name plus the model-derived args dict); the
    # privacy-mode sibling logs the tool name alone. Matched structurally.
    re.compile(r"\bjobs web tool: \S+ \{"),
)


def is_content_record(rec: LogRecord) -> bool:
    """True when *rec*'s header line is one of the known debug_content_enabled()
    writes (see _CONTENT_MARKER_RES) - i.e. it carries chat content and must
    never appear in a bug report, regardless of its level or how it would
    otherwise be collapsed. Only the header line is checked; every marked
    write site puts its marker on lines[0]."""
    if not rec["lines"]:
        return False
    header = rec["lines"][0]
    return any(p.search(header) for p in _CONTENT_MARKER_RES)


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


def _unleveled_lines(rec: LogRecord) -> List[str]:
    """The lines of *rec* that never went through the "TIMESTAMP LEVEL NAME:"
    header check - i.e. the ones is_error_record cannot trust rec["level"] to
    have already judged. When the record itself has no recognized header
    (rec["level"] == ""), that covers every line, including lines[0] (see
    parse_records' "file starts mid-record" branch). Otherwise it is every
    CONTINUATION line glued on after the trusted header, lines[1:]."""
    return rec["lines"] if not rec["level"] else rec["lines"][1:]


def is_error_record(rec: LogRecord) -> bool:
    """True for a WARNING+ record, or one carrying a raw traceback or other
    native-crash signal text in an unleveled line (a native ggml/CUDA crash or
    a print()'d exception can land in the log with no leveled prefix line of
    its own - see _CONTINUATION_ERROR_SIGNAL_RE)."""
    if rec["level"] in _ERROR_LEVELS:
        return True
    return any(_TRACEBACK_MARKER in ln or _CONTINUATION_ERROR_SIGNAL_RE.search(ln)
               for ln in _unleveled_lines(rec))


def record_template(rec: LogRecord) -> str:
    """A near-duplicate key: level + logger + EVERY line's text (numbers
    masked out), so timestamps/latencies/counts that differ run to run do not
    stop two otherwise-identical records from collapsing together.

    Keyed on the whole record, not just lines[0], so records only collapse when
    they are near-duplicates END TO END. The "\\x1e" join separator stops two
    different line splits from concatenating to the same key.

    TRADE-OFF: numbers are masked in continuation lines too, so two records
    whose only difference is a number collapse together - e.g. "native worker
    exit code 137" and "...139" (on POSIX, 128 + SIGKILL and 128 + SIGSEGV)
    mask to the same template. The survivor keeps its own real line, so one
    instance's exact number always survives; the DISTINCTION does not. To keep
    a native diagnostic value always distinguishable, widen
    _CONTINUATION_ERROR_SIGNAL_RE so the record is promoted to an error and
    kept verbatim, rather than weakening masking here."""
    lines = rec["lines"]
    first = lines[0]
    m = _LOG_LINE_RE.match(first)
    body = m.group(3) if m else first
    masked = [_NUMERIC_RE.sub("#", body)] + [_NUMERIC_RE.sub("#", ln) for ln in lines[1:]]
    return f"{rec['level']}|{rec['logger']}|" + "\x1e".join(masked)


def _record_timestamp(rec: LogRecord) -> str:
    return rec["lines"][0][:19]   # "YYYY-MM-DD HH:MM:SS" prefix, or "" if absent


def _collapse_line_runs(lines: List[str]) -> List[str]:
    """The line-level twin of collapse_records' record-level collapsing,
    applied WITHIN a single (already-grouped) benign record's own lines: a run
    of _MIN_RUN_TO_COLLAPSE+ consecutive near-duplicate lines (numbers masked,
    same as record_template) collapses to the last of them plus a repeat
    count. A long run of unleveled native (ggml/CUDA/HIP) stderr - no
    "TIMESTAMP LEVEL NAME:" prefix of its own - always lands as CONTINUATION
    LINES of whatever record precedes it (see parse_records), so it is never a
    run of multiple RECORDS for collapse_records itself to fold."""
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        tmpl = _NUMERIC_RE.sub("#", lines[i])
        j = i + 1
        while j < n and _NUMERIC_RE.sub("#", lines[j]) == tmpl:
            j += 1
        run_len = j - i
        if run_len >= _MIN_RUN_TO_COLLAPSE:
            out.append(f"{lines[j - 1]}  (repeated {run_len}x)")
        else:
            out.extend(lines[i:j])
        i = j
    return out


def collapse_records(records: List[LogRecord]) -> List[str]:
    """Render *records* to lines: every error record kept verbatim; a run of
    _MIN_RUN_TO_COLLAPSE+ consecutive near-duplicate BENIGN records (now
    matched on their FULL content via record_template, continuation lines
    included - see its docstring) collapses into the survivor's own lines in
    full, plus a repeat count and timespan on the last of them.

    Multi-line (continuation-carrying) records DO take part in collapsing:
    record_template hashes every line, so records only collapse when they are
    near-duplicates END TO END, and the survivor below keeps ALL of its own
    lines rather than only lines[0]."""
    out: List[str] = []
    i, n = 0, len(records)
    while i < n:
        rec = records[i]
        if is_error_record(rec):
            # Collapse REPEATED lines within this record's own body - not the
            # record itself, and never a different record's lines. An error
            # record with genuinely distinct lines (one header plus a few
            # different traceback frames) has no run to collapse and passes
            # through unchanged; only a run of _MIN_RUN_TO_COLLAPSE+
            # near-duplicate CONSECUTIVE lines folds, and the header plus one
            # real instance of every repeated line still survive.
            out.extend(_collapse_line_runs(rec["lines"]))
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
            survivor = _collapse_line_runs(records[j - 1]["lines"])
            out.extend(survivor[:-1])
            out.append(f"{survivor[-1]}  (repeated {run_len}x, {span})")
        else:
            for k in range(i, j):
                out.extend(_collapse_line_runs(records[k]["lines"]))
        i = j
    return out


def _drop_content_records(
        records: List[LogRecord], *, start_tainted: bool = False
) -> "tuple[List[LogRecord], int]":
    """Remove every content record (is_content_record) AND everything that
    follows one until we are confident we have resynchronized to genuine
    operational logging - BEFORE anything else touches *records*.

    A content-marker match on a record's OWN header is not sufficient by
    itself. debuglog.py's writer is a stock logging.FileHandler with no
    boundary marker between records, so parse_records has no way to know a
    multi-line debug_content_enabled() write's true extent: if the model's
    OWN reply text contains a line shaped like localm's log header
    ("TIMESTAMP LEVEL name: message"), parse_records starts a NEW record right
    there, and that fragment's header matches none of _CONTENT_MARKER_RES.

    So once a content marker fires, every record after it is ALSO dropped -
    regardless of its own claimed level, which is as forgeable as any other
    line - until a run of _MIN_RUN_TO_COLLAPSE (3) CONSECUTIVE, mutually
    near-duplicate (matching record_template()) records appears. Content
    producing 3 back-to-back near-duplicates immediately after the marker,
    with nothing else in between, is the residual, accepted risk; the fix for
    a narrower window is widening _CONTENT_MARKER_RES or the resync run
    length.

    *start_tainted* applies the SAME distrust to the very first record, for
    text that may itself start mid-way through a content write with no header
    at all (parse_records' "starts mid-record" branch gives it level=="").

    Returns (kept, count_dropped)."""
    kept: List[LogRecord] = []
    dropped = 0
    tainted = start_tainted
    i, n = 0, len(records)
    while i < n:
        rec = records[i]
        if is_content_record(rec):
            tainted = True
            dropped += 1
            i += 1
            continue
        if tainted:
            tmpl = record_template(rec)
            j = i + 1
            while (j < n and not is_content_record(records[j])
                   and record_template(records[j]) == tmpl):
                j += 1
            run_len = j - i
            if run_len >= _MIN_RUN_TO_COLLAPSE:
                tainted = False
                kept.extend(records[i:j])
            else:
                dropped += run_len
            i = j
            continue
        kept.append(rec)
        i += 1
    return kept, dropped


def _content_withheld_notice(n: int) -> str:
    """The "N debug record(s) withheld" disclosure. Placed FIRST when present, so
    it survives the front-anchored truncation build_report applies on top of this
    digest."""
    return (f"... ({n} debug record(s) withheld - chat content is never "
            "included in a bug report) ...")


def build_digest(text: str, *, max_chars: int = 6000, start_tainted: bool = False) -> str:
    """The full pipeline: parse -> drop chat-content records (and everything
    until resync, see _drop_content_records) -> collapse near-duplicate
    benign runs -> keep every error in full -> fit *max_chars*, trimming
    benign content first and never silently dropping an error. Never raises
    (returns "" on any unexpected failure, matching the caller's existing
    best-effort contract). *start_tainted*: see _drop_content_records - pass
    True when *text* may itself start mid-record (a truncated tail)."""
    try:
        records = parse_records(text)
        if not records:
            return ""
        records, withheld = _drop_content_records(records, start_tainted=start_tainted)
        notice = _content_withheld_notice(withheld) if withheld else ""
        if not records:
            return notice
        lines = collapse_records(records)
        digest = "\n".join(lines).strip()
        full = "\n".join(p for p in (notice, digest) if p).strip()
        if len(full) <= max_chars:
            return full
        return _fit_budget(records, max_chars, content_notice=notice)
    except Exception:
        return ""


# Marker for an error kept only in part (see _fit_budget).
_TRUNCATED_MARK = ("... (this error was truncated for space - its start is in "
                   "the full log file) ...\n")
# Below this much room for actual error text, a tail is too small to be worth
# anything; fall back to declaring the error omitted instead.
_MIN_ERROR_TAIL = 40


def _fit_budget(records: List[LogRecord], max_chars: int, *,
                content_notice: str = "") -> str:
    """Re-render, this time dropping benign (collapsed or not) lines from the
    FRONT first - oldest activity goes first, errors are never touched -
    until it fits. If the errors alone still exceed the budget, keep the
    MOST RECENT ones (most actionable) and say plainly how many earlier
    errors were omitted.

    The most recent error is kept even when it ALONE exceeds the budget, cut down
    to its tail and marked as truncated.

    *content_notice*, when non-empty, is the "N debug record(s) withheld"
    disclosure from build_digest - already-dropped content records never
    reach *records* here, but the budget must still leave room for the
    notice itself so it is never the thing trimmed away."""
    reserved = len(content_notice) + 1 if content_notice else 0
    budget = max_chars - reserved
    error_idxs = [i for i, r in enumerate(records) if is_error_record(r)]
    error_blocks = ["\n".join(records[i]["lines"]) for i in error_idxs]
    errors_text = "\n".join(error_blocks)

    if len(errors_text) <= budget:
        # Errors fit; add back as much collapsed benign context as the
        # remaining budget allows, most-recent-first (drop the oldest first).
        benign_lines = collapse_records(
            [r for i, r in enumerate(records) if i not in set(error_idxs)])
        budget_left = budget - len(errors_text) - 2
        kept: List[str] = []
        for ln in reversed(benign_lines):
            if budget_left - len(ln) - 1 < 0:
                break
            kept.append(ln)
            budget_left -= len(ln) + 1
        kept.reverse()
        parts = [content_notice] if content_notice else []
        if kept:
            parts.append("\n".join(kept))
        parts.append(errors_text)
        return "\n".join(p for p in parts if p).strip()

    # Even the errors alone do not fit: keep the most recent ones (most
    # actionable) and say how many earlier ones were cut - never silent.
    kept_blocks: List[str] = []
    used = 0
    omitted = 0
    for block in reversed(error_blocks):
        room = budget - 80 - used - 1                 # leave space for the notice
        if len(block) <= room:
            kept_blocks.append(block)
            used += len(block) + 1
            continue
        if not kept_blocks and room > len(_TRUNCATED_MARK) + _MIN_ERROR_TAIL:
            # This is the MOST RECENT error and it does not fit whole. Keep its
            # TAIL: a traceback's innermost exception type+message, the one line
            # that names the failure, is at the END.
            tail = block[-(room - len(_TRUNCATED_MARK)):]
            kept_blocks.append(_TRUNCATED_MARK + tail)
            used += len(_TRUNCATED_MARK) + len(tail) + 1
            continue
        omitted += 1
    kept_blocks.reverse()
    omitted_notice = (f"... ({omitted} earlier error(s) omitted for space - see the "
                      "full log file) ..." if omitted else "")
    parts = [p for p in (content_notice, omitted_notice) if p]
    parts.append("\n".join(kept_blocks))
    return "\n".join(p for p in parts if p).strip()
