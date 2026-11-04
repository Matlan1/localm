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

_GREETING = r"(?:(?:hey|hi|hello|hallo|ok|okay|so|also|and|und)[,!.]?\s+)?"
_ASKING = (
    r"(?:can|could|would|kannst|könntest|würdest)\s+(?:you|du)\s+(?:bitte\s+|please\s+)?"
)
_LEAD = (
    r"^\s*" + _GREETING + r"(?:" + _ASKING
    + r"|(?:i\s+want|i'd\s+like|i\s+would\s+like)\s+you\s+to\s+|please\s+|bitte\s+)?"
)
_EN_VERB = (
    r"(?P<verb>remember|memori[sz]e|make\s+a\s+note|note\s+down|keep\s+in\s+mind|"
    r"don'?t\s+forget|do\s+not\s+forget)"
)
_DE_VERB = (
    r"(?P<verb>merk'?\s+dir|merke\s+dir|dir\s+merken|vergiss\s+nicht|"
    r"denk(?:e)?\s+(?:bitte\s+)?daran|erinnere\s+dich\s+daran|behalte\s+im\s+kopf|"
    r"notiere\s+dir)"
)
_EN_RE = re.compile(_LEAD + _EN_VERB + r"(?P<rest>(?=[\s:,]).*)$", re.IGNORECASE | re.DOTALL)
_DE_RE = re.compile(_LEAD + _DE_VERB + r"(?P<rest>(?=[\s:,]).*)$", re.IGNORECASE | re.DOTALL)
_ASKED_RE = re.compile(r"^\s*" + _GREETING + _ASKING, re.IGNORECASE)

_CONNECT_RE = re.compile(
    r"^(?:\s*(?:bitte|please)\b)?\s*(?P<punct>[:,])?\s*"
    r"(?:(?P<kw>that(?![\w'’])|dass|daß|this(?=\s*:))[\s:,]*)?",
    re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(
    r"^(?:i|i'm|i’m|i've|i'll|i'd|my|me|we|we're|our|mine|ich|mein\w*|meine\w*|wir|unser\w*)\b",
    re.IGNORECASE)
_RECOLLECTION_RE = re.compile(
    r"^(?:time|day|night|moment|one\s+time|summer|winter|trip|conversation|chat)\b",
    re.IGNORECASE)
_ADDRESSES_ASSISTANT_RE = re.compile(r"^(?:you|du)\b", re.IGNORECASE)
_TASK_VERBS = (
    r"(?:fix|explain|add|summari[sz]e|rewrite|refactor|debug|review|translate|change|"
    r"update|improve|edit|delete|remove|run|test|check|continue|repeat|show|send|use|"
    r"finish|complete|help|look)"
)
_TASK_ON_IT = _TASK_VERBS + r"\s+(?:it|this|that|them|those)\b"
_REQUEST_RE = re.compile(
    r"\b(?:can|could|would|will)\s+you\b|\b(?:and|then)\s+(?:please\s+)?" + _TASK_ON_IT,
    re.IGNORECASE)
_BACK_REFERENCE_RE = re.compile(
    r"\bfrom\s+(?:earlier|above)\b"
    r"|\byou\s+(?:just\s+)?(?:wrote|sent|gave|made|said|showed|suggested|mentioned|"
    r"posted|generated|created)\b"
    r"|\bwe\s+(?:just\s+)?(?:discussed|talked\s+about)\b",
    re.IGNORECASE)
_IMPERATIVE_NEXT_RE = re.compile(
    r"^\s*(?:(?:and|then|now|please|pls|ok|okay)[,!]?\s+)*" + _TASK_ON_IT,
    re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_TERMINATOR_RE = re.compile(r"[.!?]+(?=\s|$)")
_ABBREVIATIONS = frozenset((
    "e.g i.e z.b d.h u.a z.t v.a u.s.w etc vs bzw usw ca inc ltd co corp dr mr mrs ms "
    "prof st jr sr nr tel str evtl ggf bspw approx no").split())
_MIN_FACT_CHARS = 6
_MAX_SCAN_CHARS = 4000


def _first_sentence(text: str) -> str:
    """The first sentence of the first paragraph of *text*, whitespace collapsed,
    terminator kept. A full stop after an abbreviation ("Inc.", "z.B.") or before
    a lower-case word does not end it."""
    para = " ".join(_PARAGRAPH_RE.split(text.strip())[0].split())
    for m in _TERMINATOR_RE.finditer(para):
        end = m.start()
        if end and para[end - 1] == " ":
            end -= 1
        word = para[para.rfind(" ", 0, end) + 1:end].lower().strip("\"'([")
        nxt = m.end() + 1 if para[m.end():m.end() + 1] == " " else m.end()
        if m.group().startswith(".") and (word in _ABBREVIATIONS
                                          or para[nxt:nxt + 1].islower()):
            continue
        return para[:m.end()]
    return para


def explicit_fact(message: str) -> Optional[str]:
    """The fact a chat *message* asks to remember, or None when it is not such a
    request.

    The message must open with the request (after an optional greeting, "please"
    or "can you"). English "remember"/"keep in mind"/"don't forget" need a "that",
    "this:" or a colon before the fact, or a fact that starts with a first-person
    word ("remember my dog is called Rex"); "remember to ...", "remember when ..."
    are not requests. German "merk dir" takes the fact directly; "vergiss nicht",
    "denk daran" and "erinnere dich daran" need "dass". A fact that addresses the
    assistant ("remember that you are a pirate") is an instruction for this chat,
    not a fact about the user, and is not stored. A first sentence ending in "?" is
    a request only when it asks ("can you remember that ...?") and carries "that"
    or "dass". After a bare "that" (no colon), a fact that does not start with a
    first-person word is not stored when it carries a request ("can you", "and
    explain it"), refers back to earlier chat ("from earlier", "you wrote"), or is
    followed by a task on it ("... . Fix it."). Only the first _MAX_SCAN_CHARS
    characters of *message* are read. The fact is the first sentence after the
    request, trailing punctuation removed, cut to MAX_TEXT_LEN."""
    text = message[:_MAX_SCAN_CHARS].strip() if isinstance(message, str) else ""
    if not text:
        return None
    for rx, english in ((_EN_RE, True), (_DE_RE, False)):
        m = rx.match(text)
        if m is None:
            continue
        verb = " ".join(m.group("verb").lower().split())
        connect = _CONNECT_RE.match(m.group("rest"))
        if connect is None:
            return None
        rest = m.group("rest")[connect.end():]
        keyword = connect.group("kw") is not None
        punct = connect.group("punct") is not None
        if english:
            if not (keyword or punct or _FIRST_PERSON_RE.match(rest)):
                return None
            if _RECOLLECTION_RE.match(rest):
                return None
        elif not keyword and verb.split()[0] in ("vergiss", "denk", "denke", "erinnere"):
            return None
        sentence = _first_sentence(rest)
        if sentence.endswith("?") and not (_ASKED_RE.match(text) and keyword):
            return None
        fact = sentence.strip("\"'").rstrip(".!?,;: ").strip()
        if len(fact) < _MIN_FACT_CHARS or not any(c.isalpha() for c in fact):
            return None
        if _ADDRESSES_ASSISTANT_RE.match(fact):
            return None
        if (english and keyword and connect.group("kw").lower() == "that"
                and ":" not in connect.group(0) and not _FIRST_PERSON_RE.match(fact)):
            following = " ".join(rest.split())[len(sentence):]
            if (_REQUEST_RE.search(fact) or _BACK_REFERENCE_RE.search(fact)
                    or _IMPERATIVE_NEXT_RE.match(following)):
                return None
        return fact[:MAX_TEXT_LEN]
    return None
