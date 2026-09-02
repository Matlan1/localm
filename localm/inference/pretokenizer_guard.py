# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse tokenizer input that would crash llama.cpp's pre-tokenizer.

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

INVARIANT the scan depends on: a run must be measured over a class at least as
wide as the pattern's own, and the limit must sit below the length at which that
pattern throws. Widening a class or lowering a limit stays safe; narrowing a
class or raising a limit can let a crashing input through. The class/limit pairs
in :data:`UNSAFE_PRE_TYPES` are pinned by
``tests/test_pretokenizer_guard.py::TestCalibration``.

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
    "policy_for",
    "read_pre_type",
]

# Widest-first: each value is a `regex` character class covering at least the
# characters its pre-tokenizer pattern can carry in one unbroken run.
_CLASS_LETTER = r"[\p{L}\p{M}]"
_CLASS_LETTER_SPACE = r"[\p{L}\p{M} ]"
_CLASS_DIGIT = r"[\p{N}]"

# GGUF metadata key naming the pre-tokenizer.
PRE_TYPE_KEY = "tokenizer.ggml.pre"


class Policy(NamedTuple):
    """Limits for one pre-tokenizer.

    ``char_class``  a ``regex`` character class for the run that must be bounded.
    ``max_run``     longest permitted unbroken run of that class, in characters.
    ``max_chars``   longest permitted total text, or ``None`` for no total limit.
    ``label``       the pre-tokenizer name used in the refusal message.
    """

    char_class: str
    max_run: int
    max_chars: Optional[int]
    label: str


_LETTER_RUN_LIMIT = 64
_DIGIT_RUN_LIMIT = 96
# Bounding runs alone is not enough: for several of these pre-tokenizers the
# cost also grows with total length however short the individual runs are kept.
_TOTAL_LIMIT = 65536


def _letters(label: str) -> Policy:
    return Policy(_CLASS_LETTER, _LETTER_RUN_LIMIT, _TOTAL_LIMIT, label)


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
}

# One compiled scanner per distinct (class, limit) pair, built once at import.
# Keyed by the pair itself rather than by the two values concatenated, so no two
# policies can ever share a key.
_SCANNERS: Dict[tuple, "regex.Pattern[str]"] = {
    (p.char_class, p.max_run):
        regex.compile(p.char_class + "{" + str(p.max_run + 1) + ",}")
    for p in UNSAFE_PRE_TYPES.values()
}


def policy_for(pre_type: Optional[str]) -> Optional[Policy]:
    """The :class:`Policy` for a ``tokenizer.ggml.pre`` value, or ``None`` when
    that pre-tokenizer is not one of the affected ones (including when
    *pre_type* is ``None``, i.e. the model declares no pre-tokenizer)."""
    if not pre_type:
        return None
    return UNSAFE_PRE_TYPES.get(pre_type)


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
    hit = _SCANNERS[(policy.char_class, policy.max_run)].search(text)
    if hit is None:
        return
    run = hit.end() - hit.start()
    kind = "digits" if policy.char_class == _CLASS_DIGIT else "letters"
    if policy.char_class == _CLASS_LETTER_SPACE:
        kind = "letters and spaces"
    raise PretokenizerUnsafeInputError(
        f"This model's pre-tokenizer ({policy.label}) crashes on an unbroken "
        f"run of more than {policy.max_run} {kind}; this input has a run of "
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
