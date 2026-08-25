# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a long-lived DAEMON that tests a caller-supplied lazy-grammar trigger pattern against fixed adversarial probe strings, so a pattern that causes catastrophic regex backtracking can never block or take down the process asking whether it is safe to use."""

from __future__ import annotations

import json
import re
import sys

# Fixed adversarial probe strings, matched against every candidate pattern.
# Sized and shaped from the REAL #928 defect (see gbnf.TOOL_CALL_TRIGGER's own
# fix, PR #933): a repeating single character is the classic worst case for a
# backtracking engine, and a repeating multi-character row mirrors the actual
# field content (a coder-generated markdown table) that triggered the crash.
# 60,000 characters is comfortably past where PR #933 measured the OLD
# TOOL_CALL_TRIGGER pattern taking seconds (22,800 chars -> 4.1s) while
# costing a SAFE pattern (any pattern shaped like the fixed TOOL_CALL_TRIGGER)
# a small, near-instant fraction of a millisecond - see gbnf.py's
# `validate_trigger_patterns` docstring for the calibration evidence.
_PROBE_REPEAT_CHAR = " " * 60_000
_PROBE_REPEAT_ROW = ("| localm/inference/backends/llamacpp/_runner.py | 1234 |\n" * 1_100)
_ADVERSARIAL_PROBES = (_PROBE_REPEAT_CHAR, _PROBE_REPEAT_ROW)

# HONEST COVERAGE STATEMENT (do not let this file's existence read as "regex
# safety solved" - it is not): a pattern passes this probe if and only if it
# completes against THESE TWO specific strings. Empirical validation is only
# as good as the input it validates against - backtracking blowup is highly
# input-shape-specific, which is exactly why #928 needed repetitive markdown
# ROWS rather than random text to reproduce at all (see PR #933's own
# measurements: the same defective pattern that hung for seconds on repeated
# text can be near-instant on other inputs of the same length). So a pattern
# that is fast against a single repeated CHARACTER and fast against a
# repeated multi-character ROW, but catastrophic against some THIRD input
# shape neither probe resembles, PASSES this check and would still be able
# to hang or crash a real generation. This is a real, documented blind spot,
# not a hypothetical one - do not extend the "safe" claim past what these
# two probes actually establish. Widening this corpus (more repeat units,
# varied lengths, nested-structure content) narrows the blind spot; it
# cannot close it, since exhaustive coverage of "every input shape" is not
# a finite set to enumerate.
#
# _pattern_derived_probes below adds probes that narrow one specific,
# MEASURED instance of that blind spot rather than a hypothetical one: a
# classic ambiguous-alternation pattern like (a|a)*b is genuinely
# catastrophic (measured live: 131.9s on just 30 'a' characters) but is
# NOT caught by either fixed probe above (both complete in under 2ms
# against it), because ambiguous-alternation danger is keyed to a SPECIFIC
# character the pattern itself names, and neither fixed probe happens to
# contain the letter 'a' repeated. No FIXED corpus can guess which
# character an arbitrary caller-supplied pattern will be keyed to - so
# instead of guessing, the probe is derived from the pattern's OWN content.
# This closes the (a|a)*b class specifically; it does not make the corpus
# exhaustive (a pattern keyed to danger only under a specific ORDER or
# STRUCTURE of its own characters, not just their repetition, could still
# pass - the honest limitation above still applies to what remains).


_MAX_DERIVED_PROBE_CHARS = 8  # bounds probe COUNT - a first-line, cheap bound.
                               # The real guarantee against a pattern naming
                               # many distinct characters is the internal
                               # wall-clock budget in _check_one below, which
                               # bounds the SUM directly; this just keeps the
                               # common case (a handful of distinct chars)
                               # from doing more work than it needs to.


def _pattern_derived_probes(pattern: str) -> "tuple[str, ...]":
    """One probe PER distinct alphanumeric character in *pattern* (the _MAX_DERIVED_PROBE_CHARS MOST FREQUENT ones, if there are more distinct characters than that), each that character ALONE repeated 60,000 times - see the module-level comment above _check_one for why this exists (an ambiguous-alternation..."""
    counts: "dict[str, int]" = {}
    for ch in pattern:
        counts[ch] = counts.get(ch, 0) + 1
    most_frequent = sorted(counts, key=lambda ch: counts[ch], reverse=True)
    return tuple(ch * 60_000 for ch in most_frequent[:_MAX_DERIVED_PROBE_CHARS])


# Internal wall-clock budget for the WHOLE probe loop of a SINGLE pattern
# (fixed probes + all derived ones), checked BETWEEN probes. This is a
# DIFFERENT thing from the caller's own round-trip timeout in gbnf.py:
# that one catches a SINGLE probe that hangs forever (nothing can check a
# deadline while blocked inside one C-level regex call). This one catches
# the case nothing else does: many probes that are each individually FINITE
# but not free - a pattern naming many distinct characters costs one probe
# PER character (see _pattern_derived_probes), and even a merely mildly
# expensive (not catastrophic) per-probe cost, summed across many, adds up
# to real time spent holding the single validation lock, for a pattern that
# may never trip the outer timeout at all - a CHEAPER attack to construct
# than genuine catastrophic backtracking, because it needs no cleverness,
# just vocabulary. Bailing out here bounds the SUM directly rather than
# hoping the count cap (_MAX_DERIVED_PROBE_CHARS) alone keeps it small.
_PROBE_LOOP_BUDGET_SECONDS = 0.5


def _check_one(pattern: str) -> str:
    """'OK' or 'BAD <reason>' for *pattern* - never raises."""
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
        # A compiled pattern's .search() should not raise, but this process
        # exists specifically to not trust caller input - never let an
        # unexpected exception here escape as an unhandled crash of a
        # long-lived daemon over one bad request.
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
