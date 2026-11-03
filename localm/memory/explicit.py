# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit "remember this" requests in a chat message.

A message that asks to remember something ("remember that I love RPGs", "merk dir,
dass ich Kaffee mag") is detected here without a model, so the fact is stored
whether or not the model would have called a tool. Detection is a closed set of
English and German phrasings: a recollection ("remember when we ...", "remember to
buy milk", "do you remember ...") is not a request to store a fact.
"""

from __future__ import annotations

import re
from typing import Optional

from .store import MAX_TEXT_LEN

_LEAD = (
    r"^\s*(?:(?:hey|hi|hello|hallo|ok|okay|so|also|and|und)[,!.]?\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?|(?:kannst|k[oö]nntest|w[uü]rdest)\s+du\s+(?:bitte\s+)?"
    r"|please\s+|bitte\s+)?"
)
_EN_VERB = (
    r"(?P<verb>remember|memori[sz]e|make\s+a\s+note|note\s+down|keep\s+in\s+mind|"
    r"don'?t\s+forget|do\s+not\s+forget)"
)
_DE_VERB = (
    r"(?P<verb>merk(?:e)?\s+dir|vergiss\s+nicht|denk(?:e)?\s+daran|"
    r"behalte\s+im\s+kopf|notiere\s+dir)"
)
_EN_RE = re.compile(_LEAD + _EN_VERB + r"(?P<rest>(?:\s*[:,]|\s)\s*\S.*)$",
                    re.IGNORECASE | re.DOTALL)
_DE_RE = re.compile(_LEAD + _DE_VERB + r"(?P<rest>(?:\s*[:,]|\s)\s*\S.*)$",
                    re.IGNORECASE | re.DOTALL)

_THAT_RE = re.compile(r"^(?:\s*[:,]\s*|\s+)that\b[\s:,]*", re.IGNORECASE)
_DASS_RE = re.compile(r"^(?:\s*[:,]\s*|\s+)(?:bitte\s+)?(?:,\s*)?dass\b[\s:,]*", re.IGNORECASE)
_BITTE_RE = re.compile(r"^\s*bitte\b\s*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"^\s*[:,]\s*")
_FIRST_PERSON_RE = re.compile(
    r"^(?:i|i'm|i've|i'll|i'd|my|me|we|we're|our|mine|ich|mein\w*|meine\w*|wir|unser\w*)\b",
    re.IGNORECASE)
_RECOLLECTION_RE = re.compile(
    r"^(?:time|day|night|moment|one\s+time|summer|winter|trip|conversation|chat)\b",
    re.IGNORECASE)
_ASKED_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|hallo|ok|okay|so|also|and|und)[,!.]?\s+)?"
    r"(?:can|could|would|kannst|k[oÃ¶]nntest|w[uÃ¼]rdest)\b", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_FACT_CHARS = 6


def _first_sentence(text: str) -> str:
    """The first sentence of the first line of *text*, terminator kept."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return _SENTENCE_END_RE.split(line, maxsplit=1)[0].strip()


def explicit_fact(message: str) -> Optional[str]:
    """The fact a chat *message* asks to remember, or None when it is not such a
    request.

    The message must open with the request (after an optional greeting or
    "please"). English "remember"/"keep in mind"/"don't forget" need a "that" or a
    colon before the fact, or a fact that starts with a first-person word ("remember
    my dog is called Rex"); "remember to ...", "remember when ..." are not
    requests. German "merk dir" takes the fact directly; "vergiss nicht" and
    "denk daran" need "dass". A message ending in "?" is a request only when it
    asks ("can you remember that ...?") and carries "that" or "dass". The fact is
    the first sentence after the request, trailing punctuation removed, cut to
    MAX_TEXT_LEN."""
    text = message.strip() if isinstance(message, str) else ""
    if not text:
        return None
    for rx, english in ((_EN_RE, True), (_DE_RE, False)):
        m = rx.match(text)
        if m is None:
            continue
        verb = " ".join(m.group("verb").lower().split())
        rest = m.group("rest")
        keyword_re = _THAT_RE if english else _DASS_RE
        stripped = keyword_re.sub("", rest, count=1)
        keyword = stripped != rest
        rest = stripped
        punct = False
        if not keyword:
            stripped = _PUNCT_RE.sub("", rest, count=1)
            punct = stripped != rest
            rest = stripped
        if english:
            if not (keyword or punct or _FIRST_PERSON_RE.match(rest.strip())):
                return None
            if _RECOLLECTION_RE.match(rest.strip()):
                return None
        else:
            rest = _BITTE_RE.sub("", rest, count=1)
        if not english and not keyword and verb in ("vergiss nicht", "denk daran", "denke daran"):
            return None
        sentence = _first_sentence(rest)
        if sentence.endswith("?"):
            if not (_ASKED_RE.match(text) is not None and keyword):
                return None
        fact = sentence.strip("\"'").rstrip(".!?,;: ").strip()
        if len(fact) < _MIN_FACT_CHARS or not any(c.isalpha() for c in fact):
            return None
        return fact[:MAX_TEXT_LEN]
    return None
