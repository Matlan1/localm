# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone entry point: a subprocess that tests caller-supplied tokenizer
pre_tokenizer/normalizer/decoder Regex patterns against fixed adversarial probe
strings using the REAL engine those patterns run on - Oniguruma, via
``tokenizers.Regex`` - so a pattern that causes catastrophic native backtracking
can never block or take down the process asking whether it is safe to load.

Same isolation rationale as ``_trigger_probe.py`` (read that file's docstring
for the fuller argument: this app ships on Windows, where nothing inside a
single thread can interrupt a stuck C-level regex match, so the process
boundary IS the timeout mechanism - the caller's own round-trip read timeout,
not anything in here, is what detects a hang). Two things are DIFFERENT from
that daemon, both deliberate:

1. This is NOT a long-lived, request-per-line daemon. ``_trigger_probe.py``
   validates a caller-supplied grammar trigger on every ``/v1/chat/completions``
   request with ``grammar_triggers`` set - a hot path, where paying Python
   startup cost per check was rejected in favor of one persistent process
   reused across many requests. Validating a MODEL's tokenizer.json happens
   once per ``HFBackend.load()`` call - a cold, infrequent operation that
   already takes seconds to minutes (multi-GB weight files), so a fresh
   subprocess per model load is a negligible addition and needs none of the
   other daemon's lock/cache/prewarm machinery, which exists solely to keep a
   HOT path cheap. Caller side lives in
   ``inference/hf_tokenizer_safety.py``.
2. The per-pattern check here catches ``BaseException``, not ``Exception``.
   MEASURED live: a pattern that trips Oniguruma's own internal backtrack
   ceiling does not return an error value or raise a normal exception - it
   raises ``pyo3_runtime.PanicException``, whose MRO is
   ``(PanicException, BaseException, object)`` - it is NOT an ``Exception``
   subclass. A bare ``except Exception:`` (the style ``_trigger_probe.py``
   uses for its own engine, where this does not arise) would let this
   propagate uncaught. Uncaught here just means an ugly-but-safe subprocess
   crash (the caller's "process died" handling still rejects the pattern
   correctly - see ``_probe_pattern_is_safe``'s three-way "unsafe" reasoning in
   gbnf.py, which this module's caller mirrors) - but the whole reason this
   validation exists is that ``HFBackend`` itself runs fully in-process with no
   worker isolation (Q1/Q2 of the 2026-07-30 release-gate decision), so if this
   exact panic reached a REAL ``tokenizer.encode()`` call inside the server
   instead of this isolated probe, an uncaught ``BaseException`` there would be
   a Rust panic crossing the FFI boundary in the server's own process - a
   stronger failure mode than "just a hang". Catching ``BaseException`` here
   is what turns that into a clean "BAD" verdict instead of a crash, in the
   one place a crash is contained by design.

Protocol (matches _trigger_probe.py's shape, minus the persistence): the
caller writes ONE JSON object as one line, ``{"patterns": ["<regex1>", ...]}``,
then reads one reply line PER PATTERN, in order, as they complete - ``OK``
means that pattern compiled and completed against every probe string without
incident; ``BAD <reason>`` means it was rejected for a reason determined
WITHOUT hanging (invalid regex syntax, or a probe raised/panicked). The
DANGEROUS case - a probe hangs - never produces a reply line for that pattern;
the caller's per-line timeout is what detects it, from outside, and the
process is killed. This process then exits after the input line is fully
processed (or is killed mid-pattern by the caller).

Invoked as ``python -m localm.inference._hf_tokenizer_probe`` by
``hf_tokenizer_safety.validate_tokenizer_json()``.
"""

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
    """One probe per distinct character IN the pattern (the
    _MAX_DERIVED_PROBE_CHARS most frequent, most-frequent-first), each that
    character alone repeated 60,000 times. See _trigger_probe.py's function of
    the same name for the fuller rationale (an ambiguous/nested-quantifier
    pattern's danger is typically keyed to whichever specific character IT
    names, which no fixed corpus can anticipate) and for why this MUST be
    single-character-per-probe rather than all extracted characters
    concatenated (a mix acts as a periodic terminator that defeats the exact
    ambiguity being probed for - caught live in that module's own history).

    Every DISTINCT character in *pattern* is a candidate, not filtered to
    isalnum(): an ambiguous-alternation or nested-quantifier pattern's
    danger is keyed to whichever specific character it names, and that
    character is exactly as often punctuation (a literal ',' or '.' inside
    an escape like '\\.') as it is a letter or digit - isalnum() silently
    dropped every such character from consideration, so a pattern keyed to
    punctuation got no derived probe at all and this whole layer could never
    catch it. _PROBE_PUNCTUATION in _FIXED_PROBES does NOT substitute for
    this: it interleaves 16 distinct punctuation characters, so no single
    one of them repeats consecutively long enough to trip an ambiguity keyed
    to that ONE character - the same "a mix defeats the probe" failure this
    function's single-character-per-probe design exists to avoid, just
    re-introduced one level up, at the class-corpus level. Deliberately
    over-inclusive rather than a precise literal-only parse: this also picks
    up a character that is really part of regex syntax (a metacharacter
    escape like the 's' in '\\s+', or a delimiter like '('), which is
    harmless - an extra probe costs microseconds - whereas under-extracting
    would silently narrow exactly the coverage this exists to add. Bounded
    to _MAX_DERIVED_PROBE_CHARS distinct characters - a cheap, first-line
    bound on probe COUNT, unaffected by widening which characters are
    eligible; _check_one's own internal wall-clock budget
    (_PROBE_LOOP_BUDGET_SECONDS) is what actually bounds total COST
    regardless of count, since a caller-controlled property (how many
    distinct characters a pattern names) must never be able to scale
    validation cost past a fixed ceiling. Returns () for an empty
    pattern."""
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
    """"OK" or "BAD <reason>" for *pattern*. Catches BaseException, not
    Exception - see the module docstring for why (Oniguruma's own retry-limit
    surfaces as pyo3_runtime.PanicException, a BaseException). A pattern whose
    FIRST slow probe genuinely hangs never returns from this function at all;
    the caller's own round-trip timeout is what detects that, from outside."""
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
