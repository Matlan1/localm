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
    r"(?:(?P<kw>that(?![\w'\u2019])|dass|daß|this(?=\s*:))[\s:,]*)?",
    re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(
    r"^(?:i|i'm|i\u2019m|i've|i'll|i'd|my|me|we|we're|our|mine|ich|mein\w*|meine\w*|wir|unser\w*)\b",
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
_MAX_FACTS = 8

# A fact runs on into the next line only when its own line is visibly unfinished.
_OPEN_END_RE = re.compile(
    r"(?:[,-]|\b(?:a|an|the|my|your|our|his|her|their|is|are|am|was|were|and|or|but|"
    r"of|in|on|at|to|for|from|with|by|as|than|that|because|"
    r"der|die|das|den|dem|des|ein|eine|einen|einem|einer|mein|meine|meinen|meinem|"
    r"meiner|ist|sind|bin|war|und|oder|aber|von|vom|im|am|an|auf|zu|zum|zur|"
    r"für|mit|bei|als|dass|weil))\s*$", re.IGNORECASE)
_JOINING_START_RE = re.compile(
    r"(?:and|or|but|because|which|with|is|are|was|of|in|on|at|for|to|from|"
    r"und|oder|aber|weil|mit|ist|sind|war|von|im|am|auf|für|zu)\b")
_FOREIGN_LINE_RE = re.compile(r"\s*(?:[>|\"'`~#\u201c\u201e\u00ab]|\*{2,}|-{2,})")
# Words that announce somebody else's text after them.
_SOURCE = (
    r"(?:wrote|writes|written|says|said|saying|texted|texts|mailed|emailed|e-mailed|"
    r"messaged|posted|replied|answered|asked|told|sent|quote[sd]?|reads|mail|email|"
    r"e-mail|message|letter|text|note|article|page|site|website|comment|post|review|"
    r"schrieb|schreibt|geschrieben|sagt|sagte|gesagt|meint|meinte|fragt|fragte|"
    r"antwortet|antwortete|geantwortet|schickt|schickte|geschickt|erzählt|erzählte|"
    r"nachricht|brief|artikel|seite|webseite|kommentar|zitat|notiz)")
_INTRODUCES_RE = re.compile(
    r"\b" + _SOURCE + r"\b[^:]*:(?:\s|$)"
    r"|\b" + _SOURCE + r"\b.*\b(?:this|the\s+following|that|das|dies|dieses|folgendes)\s*[.!]*$"
    r"|\b(?:below|the\s+following|as\s+follows|forwarded|pasted|copied|"
    r"unten|folgende[nmrs]?|weitergeleitet\w*|kopiert\w*)\b",
    re.IGNORECASE)
_COURTESY_RE = re.compile(
    r"^\s*(?:thanks(?:\s+(?:a\s+lot|so\s+much|very\s+much))?|thank\s+you(?:\s+(?:so|very)\s+much)?|"
    r"thx|cheers|great|perfect|ok(?:ay)?|danke(?:\s+(?:dir|schön|sehr))?|vielen\s+dank|"
    r"super|prima|gut)[ \t]*(?:[.!,]+|\n)\s*", re.IGNORECASE)
_LETTER_OPENING_RE = re.compile(r"^\s*(?:hey|hi|hello|hallo)\b", re.IGNORECASE)
_QUOTE_PAIRS = (("\u201c", "\u201d"), ("\u201e", "\u201c"), ("\u00ab", "\u00bb"))


def _opens_request(text: str) -> bool:
    return bool(_EN_RE.match(text) or _DE_RE.match(text))


def _runs_on(line: str, following: str) -> bool:
    """True when *following* continues the unfinished text line *line*."""
    if _FOREIGN_LINE_RE.match(following) or following.endswith(":"):
        return False
    if line.endswith((".", "!", "?", ":", ";")) or _opens_request(following):
        return False
    return bool(_OPEN_END_RE.search(line) or _JOINING_START_RE.match(following))


def _logical_lines(para: str) -> list[str]:
    """The lines of the paragraph *para*, whitespace collapsed, with a line joined
    to the one before it when that one is visibly unfinished. A first line that
    announces a list of its own ("my hobbies are:") takes the lines after it, unless
    it announces somebody else's words or the next line is itself a request."""
    out: list[str] = []
    for raw in para.split("\n"):
        line = " ".join(raw.split())
        if not line:
            continue
        if out and _runs_on(out[-1], line):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    if len(out) > 1 and out[0].endswith(":") and not _INTRODUCES_RE.search(out[0]):
        own = 1
        while own < len(out) and not (_FOREIGN_LINE_RE.match(out[own])
                                      or _opens_request(out[own])):
            own += 1
        out[:own] = [" ".join(out[:own])]
    return out


def _quotes_open(text: str) -> bool:
    """True when *text* opens a quotation or code span it does not close."""
    if text.count('"') % 2 or text.count("`") % 2:
        return True
    if any(text.count(opening) > text.count(closing) for opening, closing in _QUOTE_PAIRS):
        return True
    opened = len(re.findall(r"(?:^|\s)['\u2018](?=\S)", text))
    closed = len(re.findall(r"(?<=\S)['\u2019](?=[\s.,!?;:]|$)", text))
    return opened > closed


def _ends_chain(sentence: str) -> bool:
    """True when what follows *sentence* in the message may be somebody else's
    words, so no further fact is taken from it."""
    return (sentence.endswith(":") or _quotes_open(sentence)
            or bool(_INTRODUCES_RE.search(sentence)))


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
    """The first fact a chat *message* asks to remember (see :func:`explicit_facts`),
    or None when it is not such a request."""
    facts = explicit_facts(message)
    return facts[0] if facts else None


def explicit_facts(message: str) -> list[str]:
    """Every fact a chat *message* asks to remember, in order, or [] when it is not
    such a request. A sentence or line of the same paragraph that directly follows
    a fact and is itself a request ("Remember that X. Remember that Y.", with at
    most a "Thanks!" between) adds its own fact, up to _MAX_FACTS. The chain ends at
    a paragraph break, at anything that is not such a request, and after a fact that
    announces or opens somebody else's words ("Bob wrote me this:", an unclosed
    quotation).

    The rules below apply to each request in turn.

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
    request, trailing punctuation removed, cut to MAX_TEXT_LEN; a line break ends it
    unless the line is visibly unfinished."""
    text = message[:_MAX_SCAN_CHARS].strip() if isinstance(message, str) else ""
    facts: list[str] = []
    while text and len(facts) < _MAX_FACTS:
        got = _one_request(text)
        if got is None:
            break
        fact, text = got
        if fact not in facts:
            facts.append(fact)
        while (polite := _COURTESY_RE.match(text)) is not None:
            text = text[polite.end():]
        if _LETTER_OPENING_RE.match(text):
            break
    return facts


def _one_request(text: str) -> Optional[tuple]:
    """``(fact, the rest of the request's paragraph)`` for the request that opens
    *text*, or None when *text* does not open with a request for a fact. The rest is
    "" when the chain of requests ends with this fact."""
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
        lines = _logical_lines(_PARAGRAPH_RE.split(rest.strip())[0])
        first = lines[0] if lines else ""
        sentence = _first_sentence(first)
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
        if (_ends_chain(sentence)
                or _PARAGRAPH_RE.search(m.group("rest")[:connect.end()])):
            return fact[:MAX_TEXT_LEN], ""
        after = "\n".join([first[len(sentence):].strip(), *lines[1:]]).strip()
        return fact[:MAX_TEXT_LEN], after
    return None
