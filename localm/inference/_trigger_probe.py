# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a long-lived DAEMON that tests a caller-supplied
lazy-grammar trigger pattern against fixed adversarial probe strings, so a
pattern that causes catastrophic regex backtracking can never block or take
down the process asking whether it is safe to use.

Same shape as ``inference/backends/llamacpp/_vram_probe.py``: a long-lived
daemon rather than a fresh subprocess per query, so Python startup cost is not
paid on every validated request. The caller side lives in ``inference/gbnf.py``
(``validate_trigger_patterns``), which owns spawn, timeout and respawn.

A SEPARATE PROCESS, not a thread or an in-process call with a Python-level
timeout: this app ships on Windows, where ``signal.alarm`` does not exist, and
a thread-based "stop waiting" does not reclaim the CPU a runaway ``re.search``
keeps consuming. The daemon process itself is the timeout boundary: a genuinely
catastrophic pattern hangs THIS process inside ``re.search`` so it never gets
to reply; the caller's read times out and the caller kills this process
outright, and a fresh daemon spawns for the next check. This process never
tries to bound its own regex calls - nothing inside a single thread can
interrupt a C-level backtracking match.

Protocol (newline-delimited, line-buffered, over stdin/stdout, JSON payload so
an arbitrary caller-supplied pattern - which may contain literal newlines,
quotes, anything - travels safely as one line): the caller writes one JSON
object per line, ``{"pattern": "<regex source>"}``, and reads one reply line.
``OK`` means the pattern compiled and completed against every probe string
without incident. ``BAD <reason>`` means the pattern is rejected for a reason
the daemon could determine WITHOUT hanging (invalid regex syntax, or a probe
raised some other exception) - still safe, no process kill needed, and the
daemon stays alive for the next request. The DANGEROUS case - the pattern hangs
a probe - never produces a reply line at all; the caller's timeout is the only
thing that ever detects it, from outside.

Invoked as ``python -m localm.inference._trigger_probe`` by
``gbnf._spawn_trigger_probe_daemon()``.
"""

from __future__ import annotations

import json
import re
import sys

# Fixed adversarial probe strings, matched against every candidate pattern. A
# repeating single character is the classic worst case for a backtracking
# engine, and a repeating multi-character row mirrors the real field content (a
# coder-generated markdown table) that triggered the original crash. 60,000
# characters is comfortably past where a known-dangerous pattern costs seconds,
# while costing a safe pattern a near-instant fraction of a millisecond.
_PROBE_REPEAT_CHAR = " " * 60_000
_PROBE_REPEAT_ROW = ("| localm/inference/backends/llamacpp/_runner.py | 1234 |\n" * 1_100)
_ADVERSARIAL_PROBES = (_PROBE_REPEAT_CHAR, _PROBE_REPEAT_ROW)

# A pattern passes this probe if and only if it completes against THESE TWO
# specific strings. Backtracking blowup is highly input-shape-specific, so a
# pattern that is fast against a single repeated CHARACTER and fast against a
# repeated multi-character ROW, yet catastrophic against some THIRD input shape
# neither probe resembles, still passes this check.
#
# _pattern_derived_probes below narrows one specific case: an ambiguous
# alternation such as (a|a)*b is catastrophic yet completes in under 2ms
# against both fixed probes, because its danger is keyed to a character the
# pattern itself names and neither fixed probe repeats that character. No FIXED
# corpus can guess which character an arbitrary caller-supplied pattern is
# keyed to, so the probe is derived from the pattern's OWN content.


_MAX_DERIVED_PROBE_CHARS = 8  # bounds probe COUNT - a first-line, cheap bound.
                               # The internal wall-clock budget in _check_one
                               # below bounds the SUM directly.


def _pattern_derived_probes(pattern: str) -> "tuple[str, ...]":
    """One probe PER distinct character in *pattern* (the
    _MAX_DERIVED_PROBE_CHARS most frequent ones, if there are more distinct
    characters than that), each that character ALONE repeated 60,000 times.
    Returns () for an empty pattern.

    Most-frequent-first, not first-encountered: a character the pattern repeats
    more often is more likely to be involved in whatever ambiguity the pattern
    has, so when the count bound discards candidates it discards the
    least-repeated ones first.

    MUST be single-character-per-probe, not all extracted characters
    concatenated together: a probe interleaving the pattern's OTHER characters
    (for ``(a|a)*b``, its own literal 'b') among the repeated 'a's acts as a
    periodic terminator that regularly resets the backtracking search space,
    defeating the exact ambiguity being probed for.

    Every DISTINCT character in *pattern* is a candidate, not filtered to
    isalnum(): an ambiguous-alternation pattern's danger is keyed to whichever
    specific character it names, and that character is as often punctuation (a
    literal dot or comma inside an escape) as it is a letter or digit.
    Over-inclusive rather than a precise literal-only parse: a character that
    is really part of regex syntax (the 'd' of a digit-class escape, or a
    delimiter) also gets a probe, which costs microseconds. Bounded to
    _MAX_DERIVED_PROBE_CHARS distinct characters, which bounds probe COUNT;
    _check_one's own internal wall-clock budget is what bounds total COST
    regardless of count, so a caller-controlled property cannot scale
    validation cost past a fixed ceiling."""
    counts: "dict[str, int]" = {}
    for ch in pattern:
        counts[ch] = counts.get(ch, 0) + 1
    most_frequent = sorted(counts, key=lambda ch: counts[ch], reverse=True)
    return tuple(ch * 60_000 for ch in most_frequent[:_MAX_DERIVED_PROBE_CHARS])


# Internal wall-clock budget for the WHOLE probe loop of a SINGLE pattern
# (fixed probes plus all derived ones), checked BETWEEN probes. Distinct from
# the caller's own round-trip timeout in gbnf.py: that one catches a SINGLE
# probe that hangs forever, since nothing can check a deadline while blocked
# inside one C-level regex call. This one catches many probes that are each
# individually FINITE but not free - a pattern naming many distinct characters
# costs one probe PER character - which sums to real time spent holding the
# single validation lock for a pattern that may never trip the outer timeout.
_PROBE_LOOP_BUDGET_SECONDS = 0.5


def _check_one(pattern: str) -> str:
    """"OK" or "BAD <reason>" for *pattern* - never raises. A pattern whose
    FIRST slow probe hangs forever never returns from this function at all: the
    caller's own round-trip timeout is what detects that case, since nothing
    inside one thread can check a deadline while blocked inside a single
    C-level regex call. A pattern that instead costs a little on EACH of many
    probes is different: this function checks its own budget between probes and
    bails out rather than let many small costs sum unbounded - see
    _PROBE_LOOP_BUDGET_SECONDS."""
    import time

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"BAD invalid regex: {e}"
    probes = _ADVERSARIAL_PROBES + _pattern_derived_probes(pattern)
    deadline = time.monotonic() + _PROBE_LOOP_BUDGET_SECONDS
    try:
        for probe in probes:
            if time.monotonic() > deadline:
                return (
                    "BAD exceeded the internal probe-loop budget "
                    f"({_PROBE_LOOP_BUDGET_SECONDS}s) - too many distinct "
                    "characters, or each individually too costly, to keep "
                    "checking; cannot prove this pattern safe in bounded time")
            compiled.search(probe)   # result unused - only completion is being tested
    except Exception as e:
        # A compiled pattern's .search() should not raise, but caller input is
        # not trusted: an unexpected exception here must not crash a long-lived
        # daemon over one bad request.
        return f"BAD probe raised {type(e).__name__}: {e}"
    return "OK"


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            pattern = json.loads(line)["pattern"]
        except Exception as e:
            print(f"BAD malformed request: {e}", flush=True)
            continue
        print(_check_one(pattern), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
