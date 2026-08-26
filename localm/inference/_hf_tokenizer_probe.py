# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a subprocess that tests caller-supplied tokenizer
pre_tokenizer/normalizer/decoder Regex patterns against fixed adversarial probe
strings using the REAL engine those patterns run on - Oniguruma, via
``tokenizers.Regex`` - so a pattern that causes catastrophic native backtracking
can never block or take down the process asking whether it is safe to load.

Nothing inside a single thread can interrupt a stuck C-level regex match on
Windows, so the process boundary IS the timeout mechanism: the caller's own
round-trip read timeout, not anything in here, is what detects a hang.

This is NOT a long-lived, request-per-line daemon. Validating a MODEL's
tokenizer.json happens once per ``HFBackend.load()`` call, so a fresh subprocess
per model load needs none of ``_trigger_probe.py``'s lock/cache/prewarm
machinery, which exists to keep a hot path cheap. The caller side lives in
``inference/hf_tokenizer_safety.py``.

The per-pattern check here catches ``BaseException``, not ``Exception``: a
pattern that trips Oniguruma's own internal backtrack ceiling raises
``pyo3_runtime.PanicException``, whose MRO is ``(PanicException, BaseException,
object)``, so a bare ``except Exception:`` would let it propagate uncaught.
``HFBackend`` runs fully in-process with no worker isolation, so the same panic
reaching a real ``tokenizer.encode()`` call would be a Rust panic crossing the
FFI boundary in the server's own process; catching it here turns that into a
clean "BAD" verdict.

Protocol: the caller writes ONE JSON object as one line,
``{"patterns": ["<regex1>", ...]}``, then reads one reply line PER PATTERN, in
order, as they complete - ``OK`` means that pattern compiled and completed
against every probe string without incident; ``BAD <reason>`` means it was
rejected for a reason determined WITHOUT hanging (invalid regex syntax, or a
probe raised or panicked). The DANGEROUS case - a probe hangs - never produces a
reply line for that pattern; the caller's per-line timeout is what detects it,
from outside, and the process is killed. This process exits after the input line
is fully processed (or is killed mid-pattern by the caller).

Invoked as ``python -m localm.inference._hf_tokenizer_probe`` by
``hf_tokenizer_safety.validate_tokenizer_json()``.
"""

from __future__ import annotations

import json
import sys

# Fixed adversarial probes, applied to EVERY candidate pattern regardless of
# its own content, 60,000 characters each. A genuinely linear pattern stays in
# the low milliseconds at that length, while a pattern in the
# lookahead-in-a-loop class already hangs Oniguruma far below it.
#
# Broader than _trigger_probe.py's two fixed probes: tokenizer
# pre_tokenizer/normalizer patterns are written against natural-language and
# code text and lean heavily on escape CLASSES (whitespace, digit, word,
# Unicode letter/number) rather than literal characters, so a probe corpus of
# only literal-character repeats would miss a pattern whose danger is keyed to
# a CLASS rather than to a character it happens to spell out.
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

# A pattern passes this probe if and only if it completes against THIS fixed
# corpus plus the patterns derived from its own content below. A pattern fast
# against every probe here but catastrophic against some input shape neither
# resembles would still pass.

_MAX_DERIVED_PROBE_CHARS = 8   # first-line bound on probe COUNT; the wall-clock
                               # budget below bounds the SUM directly.


def _pattern_derived_probes(pattern: str) -> "tuple[str, ...]":
    """One probe per distinct character IN the pattern (the
    _MAX_DERIVED_PROBE_CHARS most frequent, most-frequent-first), each that
    character alone repeated 60,000 times. Returns () for an empty pattern.

    One character per probe, never all extracted characters concatenated: a mix
    acts as a periodic terminator that defeats the exact ambiguity being probed
    for.

    Every DISTINCT character in *pattern* is a candidate, not filtered to
    isalnum(): an ambiguous-alternation or nested-quantifier pattern's danger is
    keyed to whichever specific character it names, and that character is as
    often punctuation as a letter or digit. _PROBE_PUNCTUATION in _FIXED_PROBES
    does NOT substitute for this - it interleaves 16 distinct punctuation
    characters, so no single one of them repeats consecutively long enough to
    trip an ambiguity keyed to that ONE character. Over-inclusive rather than a
    precise literal-only parse: a character that is really part of regex syntax
    also gets a probe, which costs microseconds. Bounded to
    _MAX_DERIVED_PROBE_CHARS distinct characters, which bounds probe COUNT;
    _check_one's own wall-clock budget (_PROBE_LOOP_BUDGET_SECONDS) is what
    bounds total COST."""
    counts: "dict[str, int]" = {}
    for ch in pattern:
        counts[ch] = counts.get(ch, 0) + 1
    most_frequent = sorted(counts, key=lambda ch: counts[ch], reverse=True)
    return tuple(ch * 60_000 for ch in most_frequent[:_MAX_DERIVED_PROBE_CHARS])


# Internal wall-clock budget for the WHOLE probe loop of a SINGLE pattern:
# bounds the case where many probes are each individually finite but not free,
# summing to real cost for a pattern that may never trip the caller's own
# per-pattern timeout. Not interruptible mid-probe (nothing can check a deadline
# while blocked inside one native match) - checked only BETWEEN probes.
_PROBE_LOOP_BUDGET_SECONDS = 2.0


def _check_one(pattern: str) -> str:
    """"OK" or "BAD <reason>" for *pattern*. Catches BaseException, not
    Exception: Oniguruma's own retry-limit surfaces as
    pyo3_runtime.PanicException, a BaseException. A pattern whose FIRST slow
    probe genuinely hangs never returns from this function at all; the caller's
    own round-trip timeout is what detects that, from outside."""
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
