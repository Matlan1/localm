# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a bug-report-ready digest of a localm debug log."""

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

# Raw native (ggml/CUDA/HIP) stderr is appended straight into the debug log fd by
# debuglog.py's dedup_native_stderr()/_write_debug() with NO "TIMESTAMP LEVEL NAME:"
# prefix of its own, so parse_records() has no choice but to attach it as a
# CONTINUATION of whichever record precedes it - almost always a routine
# DEBUG-level poll, given how dense e.g. GET /api/stats logging is. The literal
# traceback marker alone misses this entirely: a crash line like "CUDA error:
# operation not permitted when stream is capturing" contains none of it, so the
# combined record inherited the benign DEBUG level and was eligible for
# collapse_records' near-duplicate collapsing - which silently discarded it
# (see _CONTINUATION_ERROR_SIGNAL_RE's use in is_error_record). This is a second,
# broader net alongside the exact traceback marker, not a replacement for it.
_CONTINUATION_ERROR_SIGNAL_RE = re.compile(
    r"\b(?:error|exception|fatal|crash(?:ed)?|segfault|segmentation fault|"
    r"core dumped|assert(?:ion)?\s+fail\w*|panic|abort(?:ed)?)\b",
    re.IGNORECASE,
)

# A run shorter than this is left expanded - collapsing "2 identical lines"
# saves nothing and just makes a short, already-readable log harder to follow.
_MIN_RUN_TO_COLLAPSE = 3

# Debug-level writes gated on localm.debuglog.debug_content_enabled() - i.e.
# they carry raw CHAT CONTENT (a model reply, an embed-failure snippet of a
# memory record, a web-tool query derived from the user's prompt) rather than
# operational data. A bug report must NEVER include chat content (the privacy
# promise the report form itself makes), so a record matching any of these is
# dropped whole in build_digest, before collapsing or error-promotion ever see
# it - see is_content_record. Each pattern anchors to a KNOWN write site's own
# stable prefix, not to prose content, so it does not rot into a scrubber
# arms race against arbitrary generated text:
_CONTENT_MARKER_RES = (
    # llama.py's _decode_stream(): logger.debug("raw model output:\n%s", ...).
    # The message's own newline puts nothing else after the marker on the
    # header line - the reply itself rides in as unleveled CONTINUATION
    # lines - so anchoring to end-of-line is exact, not a substring guess.
    re.compile(r"raw model output:\s*$"),
    # memory/store.py's _embed_one(): the content-bearing branch of that log
    # statement is "memory embed_one failed for %r: %s" (the snippet is
    # inline on the header line). Its privacy-mode sibling logs an entirely
    # different, content-free message ("...failed (content withheld: privacy
    # mode..."), so this prefix can only match the content-bearing branch.
    re.compile(r"\bmemory embed_one failed for "),
    # jobs/webtool.py's scheduled web-tool loop: the content-bearing branch is
    # "jobs web tool: %s %s" (tool name + the model-derived args dict, e.g. the
    # search query); the privacy-mode sibling logs the tool name ALONE with
    # nothing after it ("jobs web tool: %s"). Matched structurally - a known
    # tool name immediately followed by the start of the args dict's repr -
    # rather than by content, so only the args-carrying branch has a "{" here.
    re.compile(r"\bjobs web tool: \S+ \{"),
)


def is_content_record(rec: LogRecord) -> bool:
    """True when *rec*'s header line is one of the known debug_content_enabled() writes (see _CONTENT_MARKER_RES) - i.e. it carries chat content and must never appear in a bug report, regardless of its level or how it would otherwise be collapsed."""
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
    """The lines of *rec* that never went through the 'TIMESTAMP LEVEL NAME:' header check - i.e. the ones is_error_record cannot trust rec['level'] to have already judged."""
    return rec["lines"] if not rec["level"] else rec["lines"][1:]


def is_error_record(rec: LogRecord) -> bool:
    """True for a WARNING+ record, or one carrying a raw traceback or other native-crash signal text in an unleveled line (a native ggml/CUDA crash or a print()'d exception can land in the log with no leveled prefix line of its own - see _CONTINUATION_ERROR_SIGNAL_RE)."""
    if rec["level"] in _ERROR_LEVELS:
        return True
    return any(_TRACEBACK_MARKER in ln or _CONTINUATION_ERROR_SIGNAL_RE.search(ln)
               for ln in _unleveled_lines(rec))


def record_template(rec: LogRecord) -> str:
    """A near-duplicate key: level + logger + EVERY line's text (numbers masked out), so timestamps/latencies/counts that differ run to run do not stop two otherwise-identical records from collapsing together."""
    lines = rec["lines"]
    first = lines[0]
    m = _LOG_LINE_RE.match(first)
    body = m.group(3) if m else first
    masked = [_NUMERIC_RE.sub("#", body)] + [_NUMERIC_RE.sub("#", ln) for ln in lines[1:]]
    return f"{rec['level']}|{rec['logger']}|" + "\x1e".join(masked)


def _record_timestamp(rec: LogRecord) -> str:
    return rec["lines"][0][:19]   # "YYYY-MM-DD HH:MM:SS" prefix, or "" if absent


def _collapse_line_runs(lines: List[str]) -> List[str]:
    """The line-level twin of collapse_records' record-level collapsing, applied WITHIN a single (already-grouped) benign record's own lines: a run of _MIN_RUN_TO_COLLAPSE+ consecutive near-duplicate lines (numbers masked, same as record_template) collapses to the last of them plus a repeat count."""
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
    """Render *records* to lines: every error record kept verbatim; a run of _MIN_RUN_TO_COLLAPSE+ consecutive near-duplicate BENIGN records (now matched on their FULL content via record_template, continuation lines included - see its docstring) collapses into the survivor's own lines in full, plus a repea..."""
    out: List[str] = []
    i, n = 0, len(records)
    while i < n:
        rec = records[i]
        if is_error_record(rec):
            # Collapse REPEATED lines within this record's own body (the same
            # line-run collapse the benign branches below get) - not the
            # record itself, and never a different record's lines. An error
            # record with genuinely distinct lines (the ordinary case: one
            # header + a few different traceback frames) has no run to
            # collapse and passes through _collapse_line_runs unchanged; only
            # a run of _MIN_RUN_TO_COLLAPSE+ near-duplicate CONSECUTIVE lines
            # folds. Raw native (ggml/CUDA/HIP) stderr has no header of its
            # own, so a long repeated run of it always lands as continuation
            # lines of whatever record precedes it - including a genuine
            # WARNING/ERROR, not just a benign one (#958/#952: 122 identical
            # continuation lines glued to one WARNING record survived
            # uncollapsed here while the record-level and other line-level
            # collapsing both worked, because this was the one branch that
            # never called _collapse_line_runs at all). "Errors are kept
            # verbatim" still holds: the header and one real instance of
            # every repeated line survive, with a repeat count in place of
            # the other copies - nothing distinct is discarded.
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
    """Remove every content record (is_content_record) AND everything that follows one until we are confident we have resynchronized to genuine operational logging - BEFORE anything else touches *records*."""
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
    """Placed FIRST when present (same defensive placement as the 'N errors omitted' notice below) so it survives the blunt front-anchored truncation build_report applies on top of this digest (bugreport.py's [:4000] slice)."""
    return (f"... ({n} debug record(s) withheld - chat content is never "
            "included in a bug report) ...")


def build_digest(text: str, *, max_chars: int = 6000, start_tainted: bool = False) -> str:
    """The full pipeline: parse -> drop chat-content records (and everything until resync, see _drop_content_records) -> collapse near-duplicate benign runs -> keep every error in full -> fit *max_chars*, trimming benign content first and never silently dropping an error."""
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


# Marker for an error kept only in part (see _fit_budget). The reader must never
# mistake a cut-down trace for a complete one and diagnose from it as if it were.
_TRUNCATED_MARK = ("... (this error was truncated for space - its start is in "
                   "the full log file) ...\n")
# Below this much room for actual error text, a tail is too small to be worth
# anything; fall back to declaring the error omitted instead.
_MIN_ERROR_TAIL = 40


def _fit_budget(records: List[LogRecord], max_chars: int, *,
                content_notice: str = "") -> str:
    """Re-render, this time dropping benign (collapsed or not) lines from the FRONT first - oldest activity goes first, errors are never touched - until it fits."""
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
    omitted_notice = (f"... ({omitted} earlier error(s) omitted for space - see the "
                      "full log file) ..." if omitted else "")
    parts = [p for p in (content_notice, omitted_notice) if p]
    parts.append("\n".join(kept_blocks))
    return "\n".join(p for p in parts if p).strip()
