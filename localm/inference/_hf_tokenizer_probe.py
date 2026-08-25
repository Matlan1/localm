# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a subprocess that tests caller-supplied tokenizer pre_tokenizer/normalizer/decoder Regex patterns against fixed adversarial probe strings using the REAL engine those patterns run on - Oniguruma, via ``tokenizers.Regex`` - so a pattern that causes catastrophic native backtrack..."""

from __future__ import annotations

import json
import sys

# Fixed adversarial probes, applied to EVERY candidate pattern regardless of
# its own content. Sized at 60,000 chars to match the calibrated precedent in
# _trigger_probe.py (that module's own docstring: comfortably past where a
# known-dangerous pattern measured in the seconds, while a safe pattern costs
# a near-instant fraction of a millisecond) - independently re-confirmed for
# THIS engine and vulnerability class: the lookahead-in-a-loop shape found
# while building this validator (see hf_tokenizer_safety.py's module
# docstring for the measurements) was already hanging Oniguruma at 2,000
# characters, so 60,000 gives generous margin to trip on anything in that
# class while a genuinely linear/safe pattern (measured: the real GPT-2
# byte-level pre-tokenizer pattern, and every textbook-safe shape tried while
# building this) stays in the low milliseconds even at this length.
#
# Broader than _trigger_probe.py's two fixed probes (a single repeated
# character plus one historical row shape) on purpose: that corpus was
# calibrated against ONE specific historical defect shape (a markdown table).
# Tokenizer pre_tokenizer/normalizer patterns are written against natural-
# language and code text, and lean heavily on escape CLASSES (\s, \d, \w,
# \p{L}, \p{N} - see the real GPT-2 pattern this validator was checked
# against) rather than literal characters, so a probe corpus of only
# literal-character repeats would systematically miss a pattern whose danger
# is keyed to a CLASS rather than a character it happens to spell out.
_PROBE_WHITESPACE = " " * 60_000
_PROBE_DIGITS = "0" * 60_000
_PROBE_LETTERS = "a" * 60_000
_PROBE_PUNCTUATION = "!.,;:-_'\"()[]{}" * 3_750   # 60,000 chars
_PROBE_PROSE = ("The quick brown fox jumps over the lazy dog, 1234567890! "
                * 1_034)   # ~60,000 chars of realistic mixed word/digit/punct text
_FIXED_PROBES = (
    _PROBE_WHITESPACE, _PROBE_DIGITS, _PROBE_LETTERS, _PROBE_PUNCTUATION,
    _PROBE_PROSE,
)

# HONEST COVERAGE STATEMENT, same discipline as _trigger_probe.py's own: a
# pattern passes this probe if and only if it completes against THIS fixed
# corpus plus the patterns derived from its own content below. Empirical
# validation only proves what it actually tried - a pattern fast against
# every probe here but catastrophic against some input shape neither this
# corpus nor the derived probes resemble would still pass. Widening this
# corpus narrows that blind spot; it cannot close it.

_MAX_DERIVED_PROBE_CHARS = 8   # same rationale as _trigger_probe.py: a cheap,
                               # first-line bound on probe COUNT, not the real
                               # guarantee against a pattern naming many
                               # distinct characters (that is the wall-clock
                               # budget below, which bounds the SUM directly).


def _pattern_derived_probes(pattern: str) -> "tuple[str, ...]":
    """One probe per distinct character IN the pattern (the _MAX_DERIVED_PROBE_CHARS most frequent, most-frequent-first), each that character alone repeated 60,000 times."""
    counts: "dict[str, int]" = {}
    for ch in pattern:
        counts[ch] = counts.get(ch, 0) + 1
    most_frequent = sorted(counts, key=lambda ch: counts[ch], reverse=True)
    return tuple(ch * 60_000 for ch in most_frequent[:_MAX_DERIVED_PROBE_CHARS])


# Internal wall-clock budget for the WHOLE probe loop of a SINGLE pattern -
# same purpose as _trigger_probe.py's _PROBE_LOOP_BUDGET_SECONDS: bounds the
# case where many probes are each individually finite but not free, summing
# to real cost for a pattern that may never trip the caller's own per-pattern
# timeout. Not interruptible mid-probe (nothing can check a deadline while
# blocked inside one native match) - checked only BETWEEN probes.
_PROBE_LOOP_BUDGET_SECONDS = 2.0


def _check_one(pattern: str) -> str:
    """'OK' or 'BAD <reason>' for *pattern*."""
    import time

    import tokenizers

    try:
        regex = tokenizers.Regex(pattern)
        splitter = tokenizers.pre_tokenizers.Split(regex, behavior="isolated")
    except BaseException as e:   # noqa: BLE001 - see module docstring
        return f"BAD invalid regex: {type(e).__name__}: {e}"

    probes = _FIXED_PROBES + _pattern_derived_probes(pattern)
    deadline = time.monotonic() + _PROBE_LOOP_BUDGET_SECONDS
    try:
        for probe in probes:
            if time.monotonic() > deadline:
                return (
                    "BAD exceeded the internal probe-loop budget "
                    f"({_PROBE_LOOP_BUDGET_SECONDS}s) - too many distinct "
                    "characters, or each individually too costly, to keep "
                    "checking; cannot prove this pattern safe in bounded time")
            splitter.pre_tokenize_str(probe)   # result unused - only completion matters
    except BaseException as e:   # noqa: BLE001 - see module docstring
        return f"BAD probe raised {type(e).__name__}: {e}"
    return "OK"


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return 0
    try:
        patterns = json.loads(line)["patterns"]
    except Exception as e:
        print(f"BAD malformed request: {e}", flush=True)
        return 0
    for pattern in patterns:
        print(_check_one(pattern), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
