# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse tokenizer input that would crash or stall llama.cpp's pre-tokenizer.

A GGUF's ``tokenizer.ggml.pre`` metadata selects one of llama.cpp's hardcoded
pre-tokenizer regex lists, which ``unicode_regex_split`` then runs over the
text. Some of those regexes contain a nested quantifier over a character class;
on a long run of characters drawn from that class, MSVC's ``std::regex`` throws
``std::regex_error``, ``unicode.cpp`` rethrows it as ``std::runtime_error``, and
nothing between there and Python catches it - the worker process dies with an
uncaught native fault rather than returning an error.

``check_text`` scans text in Python BEFORE any of it reaches
``llama_tokenize`` and raises :class:`PretokenizerUnsafeInputError` when the
text would reach one of those regexes with a run long enough to matter. It is a
no-op for every ``tokenizer.ggml.pre`` value not in :data:`UNSAFE_PRE_TYPES`,
which is all but a handful of them.

Two of the bounds prevent a CRASH and two prevent unbounded COST; they are not
the same claim and the refusal messages say which:

* ``max_run`` over ``char_class``, and ``newline_run`` over ``[\r\n]``, sit below
  a length measured to THROW. A run past either kills the worker.
* ``cost_budget``, and ``jais-2``'s whitespace ``max_run``, bound how long
  ``unicode_regex_split`` RUNS. Neither pre-tokenizer has been observed to throw
  on those inputs; they go quadratic, not fatal.

INVARIANT the scan depends on: a run must be measured over a class at least as
wide as the pattern's own, and the limit must sit below the length at which that
pattern throws or becomes slow. Widening a class or lowering a limit stays safe;
narrowing a class or raising a limit can let a crashing or pathological input
through.

INVARIANT ``cost_budget`` depends on: cost grows as
``len(text) * longest run of characters with no ASCII whitespace in it``, so the
product of those two is what must stay under the budget. Bounding either alone
does not bound the cost, and bounding a run of characters OUTSIDE the pattern's
own class does not bound it at all.

Both are pinned by ``tests/test_pretokenizer_guard.py::TestCalibration``.

The scan's own patterns are a single bounded character-class quantifier, which
backtracks linearly and cannot itself blow up on hostile input.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import regex

from localm.inference.backends.base import PretokenizerUnsafeInputError

__all__ = [
    "PRE_TYPE_KEY",
    "Policy",
    "PretokenizerUnsafeInputError",
    "UNSAFE_PRE_TYPES",
    "check_text",
    "hazard_note",
    "policy_for",
    "read_pre_type",
]

# Widest-first: each value is a `regex` character class covering at least the
# characters its pre-tokenizer pattern can carry in one unbroken run.
_CLASS_LETTER = r"[\p{L}\p{M}]"
_CLASS_LETTER_SPACE = r"[\p{L}\p{M} ]"
_CLASS_DIGIT = r"[\p{N}]"
_CLASS_NEWLINE = r"[\r\n]"
_CLASS_WHITESPACE = r"[\s]"
# The run cost_budget measures. Deliberately the complement of ASCII whitespace
# rather than \S, even though non-ASCII whitespace also ends a block in
# llama.cpp: counting fewer characters as whitespace yields a run at least as
# long as the real one, and over-measuring the run is the safe direction here.
# Matching Python's idea of \s to llama.cpp's would have to be exact, and any
# character Python called whitespace that llama.cpp did not would under-measure.
_CLASS_UNBROKEN = r"[^ \t\n\r\f\v]"

# Runs over these classes bound COST, not a crash, so their refusal says so.
_COST_ONLY_CLASSES = frozenset({_CLASS_WHITESPACE})

_RUN_KIND = {
    _CLASS_LETTER: "letters",
    _CLASS_LETTER_SPACE: "letters and spaces",
    _CLASS_DIGIT: "digits",
    _CLASS_WHITESPACE: "whitespace characters",
    _CLASS_NEWLINE: "line breaks",
}

# GGUF metadata key naming the pre-tokenizer.
PRE_TYPE_KEY = "tokenizer.ggml.pre"


class Policy(NamedTuple):
    """Limits for one pre-tokenizer.

    ``char_class``   a ``regex`` character class for the run that must be
                     bounded, or ``None`` when this pre-tokenizer has no run
                     bound.
    ``max_run``      longest permitted unbroken run of that class, in characters.
    ``max_chars``    longest permitted total text, or ``None`` for no total limit.
    ``label``        the pre-tokenizer name used in the refusal message.
    ``newline_run``  longest permitted unbroken run of ``\r``/``\n``, or ``None``
                     when a newline run does not throw for this pre-tokenizer.
    ``cost_budget``  cap on ``len(text)`` multiplied by the longest run carrying
                     no ASCII whitespace, or ``None`` when unbounded.
    """

    char_class: Optional[str]
    max_run: Optional[int]
    max_chars: Optional[int]
    label: str
    newline_run: Optional[int] = None
    cost_budget: Optional[int] = None


_LETTER_RUN_LIMIT = 64
_DIGIT_RUN_LIMIT = 96
# Bounding runs alone is not enough: for several of these pre-tokenizers the
# cost also grows with total length however short the individual runs are kept.
_TOTAL_LIMIT = 65536
# Longest CR/LF run allowed. Every pre-tokenizer whose pattern list carries a
# `\s*[\r\n]+` branch throws on a run far longer than this; the limit sits well
# under that length rather than just below it, because the cost of runs SHORTER
# than the one that throws still accumulates over the whole text. The length
# that throws is pinned as TestCalibration.MEASURED_NEWLINE_CRASH.
_NEWLINE_RUN_LIMIT = 64
_WHITESPACE_RUN_LIMIT = 1024
_COST_BUDGET = 512 * _TOTAL_LIMIT


def _letters(label: str) -> Policy:
    return Policy(_CLASS_LETTER, _LETTER_RUN_LIMIT, _TOTAL_LIMIT, label,
                  newline_run=_NEWLINE_RUN_LIMIT)


# Keyed by the literal `tokenizer.ggml.pre` string a GGUF declares, NOT by
# llama.cpp's LLAMA_VOCAB_PRE_TYPE_* enum name: several distinct strings share
# one enum value and therefore one pattern.
UNSAFE_PRE_TYPES: Dict[str, Policy] = {
    # LLAMA_VOCAB_PRE_TYPE_EXAONE_MOE. Its run alternates letters with single
    # spaces, so a space does not end a run here as it does for the others.
    "exaone-moe": Policy(
        _CLASS_LETTER_SPACE, _LETTER_RUN_LIMIT, _TOTAL_LIMIT, "exaone-moe"),
    # LLAMA_VOCAB_PRE_TYPE_GPT4O.
    "gpt-4o": _letters("gpt-4o"),
    "llama4": _letters("llama4"),
    "kanana2": _letters("kanana2"),
    "talkie": _letters("talkie"),
    # LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2.
    "minimax-m2": _letters("minimax-m2"),
    # LLAMA_VOCAB_PRE_TYPE_TEKKEN.
    "tekken": _letters("tekken"),
    # LLAMA_VOCAB_PRE_TYPE_GRANITE_EMB_MULTI.
    "granite-embed-multi-97m": _letters("granite-embed-multi-97m"),
    # LLAMA_VOCAB_PRE_TYPE_SUPERBPE. Digits, not letters.
    "superbpe": Policy(_CLASS_DIGIT, _DIGIT_RUN_LIMIT, _TOTAL_LIMIT, "superbpe"),
    # LLAMA_VOCAB_PRE_TYPE_TINY_AYA and _YOUTU.
    "tiny_aya": _letters("tiny_aya"),
    "cohere2moe": _letters("cohere2moe"),
    "youtu": _letters("youtu"),
    # LLAMA_VOCAB_PRE_TYPE_JAIS2. Its whitespace run bounds COST, not a crash;
    # its newline run bounds a crash, and is the stricter of the two.
    "jais-2": Policy(_CLASS_WHITESPACE, _WHITESPACE_RUN_LIMIT, _TOTAL_LIMIT,
                     "jais-2", newline_run=_NEWLINE_RUN_LIMIT),
    # LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM. No run bound and no newline bound: a
    # newline run neither throws nor is slow here, and bounding a run of
    # characters outside its patterns' classes was measured not to bound the
    # cost at all. The budget is the whole guard for this one.
    "deepseek-llm": Policy(None, None, _TOTAL_LIMIT, "deepseek-llm",
                           cost_budget=_COST_BUDGET),
}


def _validate_table() -> None:
    """Reject a policy whose limits cannot be compiled into a scanner.

    A missing scanner would surface as a KeyError from ``check_text`` instead of
    :class:`PretokenizerUnsafeInputError`, and the worker reads an untyped error
    as a native fault: it would evict the loaded model and report a permanently
    refused input as a temporary failure. Failing here makes that
    unrepresentable.
    """
    for name, p in UNSAFE_PRE_TYPES.items():
        if p.char_class is not None and p.max_run is None:
            raise ValueError(f"{name}: char_class without max_run")
        if p.cost_budget is not None and not p.max_chars:
            raise ValueError(f"{name}: cost_budget without max_chars")
        # _RUN_KIND names the class in the refusal and in hazard_note, and
        # hazard_note runs at model LOAD, so an unmapped class would fail the
        # load rather than one request.
        if p.char_class is not None and p.char_class not in _RUN_KIND:
            raise ValueError(f"{name}: char_class missing from _RUN_KIND")


_validate_table()


def _run_scanner_keys():
    for p in UNSAFE_PRE_TYPES.values():
        if p.char_class is not None:
            yield (p.char_class, p.max_run)
        if p.newline_run is not None:
            yield (_CLASS_NEWLINE, p.newline_run)


# One compiled scanner per distinct (class, limit) pair, built once at import.
# Keyed by the pair itself rather than by the two values concatenated, so no two
# policies can ever share a key.
_SCANNERS: Dict[tuple, "regex.Pattern[str]"] = {
    (cls, limit): regex.compile(cls + "{" + str(limit + 1) + ",}")
    for cls, limit in _run_scanner_keys()
}

# For cost_budget, only runs longer than budget // max_chars can ever breach it,
# because len(text) is itself capped at max_chars. Scanning just those keeps the
# ordinary case one linear pass that matches nothing.
_COST_SCANNERS: Dict[tuple, "regex.Pattern[str]"] = {
    (p.cost_budget, p.max_chars):
        regex.compile(_CLASS_UNBROKEN + "{" + str(p.cost_budget // p.max_chars + 1) + ",}")
    for p in UNSAFE_PRE_TYPES.values()
    if p.cost_budget is not None and p.max_chars
}


def policy_for(pre_type: Optional[str]) -> Optional[Policy]:
    """The :class:`Policy` for a ``tokenizer.ggml.pre`` value, or ``None`` when
    that pre-tokenizer is not one of the affected ones (including when
    *pre_type* is ``None``, i.e. the model declares no pre-tokenizer)."""
    if not pre_type:
        return None
    return UNSAFE_PRE_TYPES.get(pre_type)


def hazard_note(pre_type: Optional[str]) -> Optional[str]:
    """One clause naming why this pre-tokenizer's input is bounded, for the
    load-time warning, or ``None`` when it is unaffected.

    Says the process ABORTS only for the bounds measured to throw. The
    cost-bounded pre-tokenizers have never been observed to throw, and telling
    an operator their model crashes would overstate the measurement.
    """
    policy = policy_for(pre_type)
    if policy is None:
        return None
    crashes, slows = [], []
    if policy.char_class is not None:
        where = slows if policy.char_class in _COST_ONLY_CLASSES else crashes
        where.append("a long unbroken run of " + _RUN_KIND[policy.char_class])
    if policy.newline_run is not None:
        crashes.append("a long unbroken run of line breaks")
    if policy.cost_budget is not None:
        slows.append("a long block of text carrying no spaces or line breaks")
    parts = []
    if crashes:
        parts.append("aborts the process on " + " or ".join(crashes))
    if slows:
        parts.append("becomes extremely slow on " + " or ".join(slows))
    return ", and ".join(parts)


def check_text(pre_type: Optional[str], text: str) -> None:
    """Raise :class:`PretokenizerUnsafeInputError` if *text* must not be handed
    to the pre-tokenizer named by *pre_type*; return ``None`` otherwise.

    Returns without scanning for any *pre_type* outside :data:`UNSAFE_PRE_TYPES`,
    so the ordinary case costs one dict lookup.

    INVARIANT: the message quotes NO part of *text*, and carries no offset into
    it. Callers surface it as an HTTP detail, which ``_log_http_exception``
    writes to the debug log gated on ``debug_enabled()`` rather than
    ``debug_content_enabled()`` because a detail is server-authored operational
    text. Echoing any of the caller's text here would put chat content in the
    debug log, including under privacy mode. Pinned by
    ``TestTheRefusalQuotesNoCallerText``.
    """
    policy = policy_for(pre_type)
    if policy is None:
        return
    if policy.max_chars is not None and len(text) > policy.max_chars:
        raise PretokenizerUnsafeInputError(
            f"This model's pre-tokenizer ({policy.label}) cannot safely handle "
            f"more than {policy.max_chars} characters at once; this input is "
            f"{len(text)}. Send less text per request."
        )
    if policy.newline_run is not None:
        _refuse_long_run(policy, text, _CLASS_NEWLINE, policy.newline_run)
    # Runs at or below `cost_budget // max_chars` can never breach the budget,
    # and _COST_SCANNERS only matches longer ones. That holds because the
    # max_chars refusal above has already bounded len(text); this check must
    # stay after it.
    if policy.cost_budget is not None and text:
        allowed = policy.cost_budget // len(text)
        for hit in _COST_SCANNERS[(policy.cost_budget, policy.max_chars)].finditer(text):
            block = hit.end() - hit.start()
            if block > allowed:
                raise PretokenizerUnsafeInputError(
                    f"This model's pre-tokenizer ({policy.label}) becomes "
                    f"extremely slow on a long block of text carrying no spaces "
                    f"or line breaks; at {len(text)} characters the longest such "
                    f"block may be {allowed} and this input has a block of {block}. "
                    f"Break it with a space or a line break, or send less text "
                    f"per request."
                )
    if policy.char_class is not None:
        _refuse_long_run(policy, text, policy.char_class, policy.max_run)


def _refuse_long_run(policy: Policy, text: str, char_class: str, limit: int) -> None:
    """Raise if *text* carries a run of *char_class* longer than *limit*."""
    hit = _SCANNERS[(char_class, limit)].search(text)
    if hit is None:
        return
    run = hit.end() - hit.start()
    kind = _RUN_KIND[char_class]
    verb = ("becomes extremely slow on" if char_class in _COST_ONLY_CLASSES
            else "crashes on")
    raise PretokenizerUnsafeInputError(
        f"This model's pre-tokenizer ({policy.label}) {verb} an unbroken "
        f"run of more than {limit} {kind}; this input has a run of "
        f"{run}. Break the run with punctuation or a line break, or use a model "
        f"with a different tokenizer."
    )


def count_tokens_or_estimate(count_tokens, text: str, what: str) -> int:
    """``count_tokens(text)``, falling back to the chars/4 estimate when the
    tokenizer REFUSES *text*, and logging that it did.

    For counts that REPORT ON WORK ALREADY DONE - a usage block for a reply that
    was streamed, or for embeddings that were computed. Raising there fails an
    operation that already succeeded, and on a streamed response it ends the
    body with no terminal chunk and no ``[DONE]``. A model can emit a run the
    pre-tokenizer refuses just as a caller can send one.

    NOT for a count that GATES the work: a pre-generation count must refuse, so
    those callers let the error propagate and map it to a 400 instead.

    Every other exception propagates: only this refusal is answerable with an
    estimate.
    """
    try:
        return count_tokens(text)
    except PretokenizerUnsafeInputError:
        from localm.debuglog import logger
        logger.warning(
            "usage: %s carries a run this model's pre-tokenizer refuses, so the "
            "reported token count is an estimate", what)
        return max(1, len(text) // 4)


def read_pre_type(model_ptr, api) -> Optional[str]:
    """The model's declared ``tokenizer.ggml.pre``, or ``None`` when it declares
    none or the runtime has no metadata API.

    Never raises: a model whose pre-tokenizer cannot be read is treated as
    unaffected, which is what every model was before this guard existed. A read
    that FAILS leaves the guard off for that model, so it is logged at WARNING
    rather than passed over silently; a runtime with no metadata API at all is
    the ordinary older-build case and is logged at DEBUG.
    """
    from localm.debuglog import logger
    try:
        if not api.has_model_meta_api():
            logger.debug("pretokenizer guard: runtime exposes no model metadata "
                         "API, so tokenizer.ggml.pre cannot be read")
            return None
        return api.llama_model_meta_val_str(model_ptr, PRE_TYPE_KEY)
    except Exception as exc:
        logger.warning(
            "pretokenizer guard: could not read tokenizer.ggml.pre (%s: %s); "
            "this model's input will NOT be checked against the pre-tokenizers "
            "known to abort the process", type(exc).__name__, exc)
        return None
