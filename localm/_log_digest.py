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
    traceback, or an unleveled continuation line that reads as a crash -
    see _CONTINUATION_ERROR_SIGNAL_RE) from the whole file, not just the
    most recent one - the point of a "digest" is that no error from the
    session goes missing. This also covers raw native (ggml/CUDA/HIP)
    stderr, which debuglog.py appends into the log file with no
    "TIMESTAMP LEVEL NAME:" prefix of its own and so always lands as a
    continuation of whatever benign record precedes it.
  * Collapses a run of 3+ consecutive BENIGN records that are near-duplicates
    END TO END - same level/logger and EVERY line's text with the numbers
    masked out, continuation lines included (see record_template) - so
    ``GET /api/stats -> 200 (7 ms, loop_lag=0.26s)`` repeated every few
    seconds collapses to one line + a repeat count, instead of listing each
    one - the routine noise a report almost never needs, especially when it
    is just "all is well" polling. Matching on the WHOLE record rather than
    just its header line means two records whose continuation content
    actually differs are never wrongly called near-duplicates of each other,
    and the survivor of a genuine collapse keeps ALL of its own lines, not
    just the first - so nothing on the one kept instance is silently
    discarded either.
  * If the digest is still over the size budget, trims from the OLDEST
    benign content first and never drops an error record silently - if even
    that is not enough, it says plainly how many earlier errors were omitted
    rather than silently truncating them (AGENTS.md rule 5).
  * Drops every record that is a known debug_content_enabled()-gated write -
    e.g. the GGUF backend's raw (pre-scrub) model reply - BEFORE anything
    else touches it, so chat content can never survive collapsing, budget
    fitting, or being promoted to an ERROR record by its own text (#961: a
    generated reply routinely contains the words "error"/"exception", which
    is exactly what promotes an otherwise-benign record to be kept verbatim).
    See _CONTENT_MARKER_RES. This is a fixed marker on each KNOWN write site,
    not a scrubber over arbitrary prose - a scrubber is a regex arms race
    against free-form generated text and will not hold.
  * Collapses a long RUN of near-duplicate lines WITHIN one benign record's
    own continuation lines, not just across records. Raw native (ggml/CUDA/
    HIP) stderr has no "TIMESTAMP LEVEL NAME:" prefix of its own, so a long
    stretch of it always glues onto ONE record as its continuation lines
    (see parse_records) - and one giant record is never a "run" of 3+ near-
    duplicate RECORDS for the record-level collapse above to fold, so a
    report's whole budget could be spent on ~70 copies of one native line
    (#958/#952). See _collapse_line_runs.
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
    # http_server.py's _log_assembled_prompt(): "assembled chat prompt (%d
    # message(s)):\n%s" - the count is on the header line, the messages ride
    # in as continuation lines below it.
    re.compile(r"assembled chat prompt \(\d+ message\(s\)\):\s*$"),
)


def is_content_record(rec: LogRecord) -> bool:
    """True when *rec*'s header line is one of the known debug_content_enabled()
    writes (see _CONTENT_MARKER_RES) - i.e. it carries chat content and must
    never appear in a bug report, regardless of its level or how it would
    otherwise be collapsed. Only the header line is checked: every marked
    write site puts its own message (never the caller's content) there, with
    any actual content confined to lines[1:] (raw model output) or inline
    after the marker on lines[0] itself (the other two) - either way the
    marker is on lines[0]."""
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

    Keyed on the whole record, not just lines[0] - a record's continuation
    lines can carry real, DIFFERING content (raw native stderr attaches as a
    continuation of whatever benign record precedes it, so e.g. ggml-cuda's
    "CUDA Graph id N reused" spam rides along with routine polling records).
    Keying on the header alone would call two such records "near-duplicates"
    whenever their headers merely mask to the same shape, even though their
    continuations differ - which used to matter a lot, because the collapsed
    survivor kept only lines[0] (see collapse_records) and every other line,
    across every record in the run, was discarded. Hashing every line means
    records only collapse when they are ACTUALLY near-duplicates end to end,
    and the "\\x1e" join separator (vanishingly unlikely to appear in real log
    text) stops two different line splits from concatenating to the same key.

    KNOWN, ACCEPTED TRADE-OFF: numbers are masked in continuation lines too,
    same as the header. This is NOT merely harmless in theory - measured, and
    the concrete case is worse than "two numbers collapse": "native worker
    exit code 137" vs "...139", neither containing a word
    is_error_record's signal scan recognizes ("error"/"exception"/etc.), mask
    to the SAME template and collapse together. On POSIX, 137 and 139 are
    128 + SIGKILL and 128 + SIGSEGV - so what actually collapses into one
    line with a repeat count is "the OOM killer took the worker" and "the
    worker segfaulted": two DIFFERENT faults needing two different
    investigations, read as one fault twice. (Windows carries no such
    convention for those particular numbers, but the general risk is not
    POSIX-specific, and localm ships on both.) The value itself is not lost -
    the survivor keeps its own real line, so at least one instance's exact
    number always survives - it is the DISTINCTION between two masked-equal-
    but-different values that is. Turning masking off for continuation lines
    is not a free fix either - it was measured too, and it breaks the exact
    case this exists to serve: real
    "CUDA Graph id 5 reused" / "id 6 reused" spam alternates, so without
    masking no two consecutive records would ever match and nothing would
    collapse at all. The escape valve is the signal scan, not the masking: if
    a native diagnostic value needs to always be told apart, that is a reason
    to widen _CONTINUATION_ERROR_SIGNAL_RE (which promotes the record to an
    error, kept verbatim, never collapsed), not to weaken masking here."""
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
    count. Needed because a long run of unleveled native (ggml/CUDA/HIP)
    stderr - no "TIMESTAMP LEVEL NAME:" prefix of its own - always lands as
    CONTINUATION LINES of whatever record precedes it (see parse_records), so
    it is never a run of multiple RECORDS for collapse_records itself to fold:
    one giant multi-line record is not a "run" (#958/#952 - a report whose
    entire tail was ~70 copies of one native line, because nothing above the
    single-record level ever collapsed it)."""
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

    Earlier drafts of this fix excluded every multi-line (continuation-
    carrying) record from collapsing at all, to stop a record's continuation
    content from being silently discarded by a collapse. Measured against the
    exact case this digest exists to help with - a CUDA-crashing box, where
    raw native stderr rides as a continuation of routine polling records, so
    ~all of them are multi-line - that blanket exclusion defeated collapsing
    almost entirely: a 200-record run of otherwise near-duplicate "GET
    /api/stream" polls, each carrying one alternating "CUDA Graph id N/N+1
    reused" continuation line, produced a 5957-char digest with 55 uncollapsed
    repeats instead of one collapsed line - reintroducing exactly the noise
    this module exists to remove, on the reports that most need to be
    readable. record_template hashing every line (not just the header) is the
    correct fix: records only collapse when they are near-duplicates END TO
    END, and the survivor below keeps ALL of its own lines rather than only
    lines[0], so collapsing a genuine near-duplicate run no longer discards
    the one kept instance's continuation content either."""
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
    """Remove every content record (is_content_record) AND everything that
    follows one until we are confident we have resynchronized to genuine
    operational logging - BEFORE anything else touches *records*.

    A content-marker match on a record's OWN header is not sufficient by
    itself. debuglog.py's writer is a stock logging.FileHandler with no
    boundary marker between records, so parse_records has no way to know a
    multi-line debug_content_enabled() write's true extent. If the model's
    OWN reply text contains a line shaped like localm's own log header
    ("TIMESTAMP LEVEL name: message" - trivially produced by a coding
    assistant quoting or discussing a real log line, or by adversarial text
    injected through an untrusted web page a job tool fetched, LM-DA-014),
    parse_records incorrectly starts a NEW record right there. That
    fragment's own header is then attacker/model-controlled text matching
    none of _CONTENT_MARKER_RES, so checking only each record's own header
    lets it straight through - reproduced live: a "raw model output:" record
    whose reply quotes one realistic-looking log line lets everything after
    that quoted line survive untouched, chat content included.

    So once a content marker fires, every record after it is ALSO dropped -
    regardless of its own claimed level, since a level marker in that window
    is exactly as forgeable as any other line - until a run of
    _MIN_RUN_TO_COLLAPSE (3) CONSECUTIVE, mutually near-duplicate (matching
    record_template()) records appears. Real operational traffic (routine
    polling) reliably produces such a run within moments of resuming, so the
    digest stays useful past the immediate vicinity of a content write;
    free-form content producing 3 back-to-back near-duplicates immediately
    after the marker, with nothing else in between, is the residual, accepted
    risk (same shape as the 137-vs-139 masking trade-off documented in
    record_template - the fix for a narrower window is widening
    _CONTENT_MARKER_RES or the resync run length, not weakening this).

    *start_tainted* applies the SAME distrust to the very first record:
    bugreport.py's _recent_log_tail truncates a huge log file to its last
    _LOG_TAIL_READ_BYTES, so the truncated text can start mid-way through a
    content write with no header at all (parse_records' "starts mid-record"
    branch gives it level==""). Without this, that severed fragment would be
    trusted immediately, the same gap via a different door.

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
    """Placed FIRST when present (same defensive placement as the "N errors
    omitted" notice below) so it survives the blunt front-anchored truncation
    build_report applies on top of this digest (bugreport.py's [:4000] slice)."""
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


# Marker for an error kept only in part (see _fit_budget). The reader must never
# mistake a cut-down trace for a complete one and diagnose from it as if it were.
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
    to its tail and marked as truncated. Dropping it whole (keeping only blocks
    that fit entire) returned a digest of just the "N omitted" notice - a bug
    report with no error in it at all, for exactly the giant native-crash
    traceback this digest exists to carry (REG-619).

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
