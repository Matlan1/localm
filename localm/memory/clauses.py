# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a chat turn says about the person who wrote it, read clause by clause.

``self_statement_stems`` collects the content words of the clauses in which the
user states something about themselves ("I live in Oslo", "my dog is called Rex",
"ich bin Vegetarier"). A clause whose only first-person word is the object of a
request to the assistant ("tell me ...", "I want to know ...", "sag mir ..."), or
the subject of a question ("what should I cook"), states nothing. A request that
names something of the user's ("a poem about my cat Tom") states that thing only.
English and German; a rule of words, so an unusual phrasing can land on either side.
"""

from __future__ import annotations

import re

from .store import _OTHER_PERSON, _content_tokens

STEM_CHARS = 4

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_I = frozenset({"i", "i'm", "i've", "i'd", "i'll", "ich"})
_ME = frozenset({"me", "myself", "mir", "mich"})
_MY = frozenset({"my", "mine", "mein", "meine", "meinen", "meinem", "meiner", "meines"})
_FIRST = r"(?:i|i['\u2019]\w+|ich|my|mein\w*|mir|mich)\b"

_PUNCTUATION = re.compile(r"[;()]|[,:](?=\s|$)|\s(?:-+|\u2013|\u2014)\s|\u2014")
_CONJUNCTION = re.compile(
    r"\s+(?=(?P<word>and|but|so|because|since|as|while|although|though|cause|cuz|where|who|"
    r"und|aber|weil|da|denn|obwohl|doch|sondern|damit|wo)\s+(?:also\s+|auch\s+|then\s+)?"
    + _FIRST + r")")
# Split here, the conjunction starts a clause that belongs to the request before it.
_DEPENDENT = frozenset({"so", "where", "who", "damit", "wo"})
_REPORTED = re.compile(
    r"\b(?:note|remember|know|mind|aware|forget|consider|account|noted|"
    r"beachte|beachten|bedenke|bedenken|wisse|merk(?:e)?\s+dir|vergiss\s+nicht)\s+"
    r"(?:that\s+|dass\s+|daß\s+)?(?=" + _FIRST + r")")

_FILLERS = frozenset(
    "ok okay now hey hi hello please so well also then and but thanks great alright right "
    "yes yeah no nope first next again sorry um uh just maybe btw anyway cool hmm oh "
    "bitte also und aber dann jetzt nun hallo danke ja nein gut mal doch nochmal erstmal "
    "claude chatgpt gpt assistant ai bot localm buddy dude man bro".split())
_WH = frozenset(
    "what which who whom whose where when why how what's who's where's how's "
    "was welche welcher welches welchen wer wem wen wie wo wann warum wieso weshalb "
    "woher wohin wozu womit".split())
_WHEN = frozenset({"when", "wann"})
_WHY = frozenset({"why", "warum", "wieso", "weshalb"})
_AUX = frozenset(
    "can could would will should shall do does did am is are was were have has had may might "
    "must don't doesn't didn't can't couldn't won't isn't aren't haven't "
    "kann kannst können könnte könntest soll sollte will willst würde würdest darf muss "
    "bin bist ist sind war hast habe hat".split())
_MODALS = frozenset(
    "should can could would must might may shall ought need'd cannot "
    "soll sollte kann könnte muss müsste darf dürfte".split())
_SUBJECTS = _I | _MY | frozenset(
    "you it this that there he she we they one anyone someone somebody the a "
    "du er sie es wir ihr man das der die jemand".split())
_SUBORDINATE_SUBJECTS = _I | _MY | frozenset("we our he she they wir unser".split())
_INDIRECT = frozenset(
    "what which how whether if why whatever was welche welcher welches welchen wie ob wenn "
    "warum".split())
_IMPERATIVES = frozenset(
    "tell give show explain write help send describe teach translate summarize summarise generate "
    "create draw recommend suggest make find get remind let bring read sing name list inform "
    "advise educate enlighten update offer provide play quiz test convince walk guide lead "
    "compare define outline discuss talk elaborate continue plan compose draft cook bake do "
    "note consider remember look search check share add say "
    "erzähl erzähle gib zeig zeige erkläre erklär schreib schreibe hilf übersetze beschreibe "
    "nenne nenn sag sage empfiehl empfehle schlag mach mache finde such suche erinnere lass "
    "lies plane erstelle liste vergleiche fass hol hole".split())
_NAMING = frozenset({"call", "nenn", "nenne"})
_LETS_IN = frozenset("you please to and then or just du sie bitte".split())
_ASKED_FOR = (
    r"(?:facts?|info|information|answers?|explanation|summary|list|recipes?|story|stories|jokes?|"
    r"poems?|examples?|advice|help|tips?|suggestions?|recommendations?|ideas?|details|overview|"
    r"translation|essay|text|letter|song|questions?|opinion|feedback)")
_ASKS = re.compile(
    r"(?:i|i'm|i'd|i've|i was|i am)\s+(?:\w+\s+){0,2}?"
    r"(?:(?:want|wanna|need|like|love|hoping|hope|trying|wish|looking|asking|curious|got|have)\s+"
    r"(?:to\s+(?:know|hear|find\s+out|ask|learn\s+(?:one|a|an|some|more|something|about))"
    r"|know\b|(?:for\s+)?you(?:r)?\b(?:\s+to\b)?|if\s+you\b"
    r"|(?:for\s+)?(?:(?:a|an|one|some|the|more|another|any|your|\d+)\s+)?(?:\w+\s+)?"
    + _ASKED_FOR + r"\b)"
    r"|wonder(?:ing)?\b|asking\b|curious\s+(?:what|how|which|whether|if|why|about)\b)"
    r"|ich\s+(?:(?:möchte|will|wollte|würde|hätte|brauche|suche|habe)\s+)(?:\w+\s+){0,4}?"
    r"(?:wissen|hören|erfahren|fragen|fakt|fakten|tatsache|information\w*|antwort|erklärung|"
    r"rezepte?|geschichte|witz|gedicht|beispiele?|rat|hilfe|tipps?|vorschl\w+|empfehlung\w*|"
    r"ideen?|frage|überblick|zusammenfassung)\b"
    r"|ich\s+frage?\s+mich\b|ich\s+will\s+dass\s+du\b|ich\s+möchte\s*$|ich\s+will\s*$")
# "I like good jokes", "I have a song on the radio": a like or a possession, not a request.
_STATES_LIKING = re.compile(
    r"(?:i|ich)\s+(?:(?:really|also|always|do)\s+)?(?:like|love|have)\s+(?!to\s+(?:know|hear|ask)\b)"
    r"(?!(?:a\s+|an\s+|one\s+|another\s+)?(?:\w+\s+)?questions?\b)")
_GERMAN_FUNCTION_WORDS = frozenset(
    "ich bin habe hab hast hat bist ist sind war waren wurde werde werden und oder aber auch in im "
    "an am auf aus bei mit von vom zu zum zur für über unter nach vor der die das den dem des ein "
    "eine einen einem einer nicht kein keine keinen mein meine meinen meinem meiner mir mich sehr "
    "gern gerne dass weil wenn als wie so noch schon nur möchte will kann muss soll heiße nutzer "
    "nutzerin benutzer benutzerin".split())


def _tokens(text: str) -> list:
    return [m.group().lower() for m in _WORD.finditer(text.replace("\u2019", "'"))]


def stems(text: str) -> set:
    """The first letters of every content word of *text* (stopwords of English and
    German left out)."""
    return {w[:STEM_CHARS] for w in _content_tokens(text)
            if len(w) > 1 and w not in _GERMAN_FUNCTION_WORDS}


def _clauses(sentence: str) -> list:
    """``(text, dependent)`` for the clauses of *sentence*, in order. *dependent*
    marks a clause a conjunction ties to the one before it ("so I can ...")."""
    out: list = []
    for part in _PUNCTUATION.split(sentence):
        begin, dependent = 0, False
        cuts = [(m.start(), m.end(), m.group("word").lower() in _DEPENDENT)
                for m in _CONJUNCTION.finditer(part.lower())]
        cuts += [(m.end(), m.end(), False) for m in _REPORTED.finditer(part.lower())]
        for end, start, dep in sorted(cuts):
            if end < begin:
                continue
            out.append((part[begin:end], dependent))
            begin, dependent = start, dep
        out.append((part[begin:], dependent))
    return [(t, d) for t, d in out if t.strip()]


def _is_wh_question(words: list) -> bool:
    """True when *words* (fillers removed) open a question with a question word."""
    if words[0] not in _WH:
        return False
    second = words[1] if len(words) > 1 else ""
    if words[0] in _WHEN:
        return second in _AUX
    if words[0] in _WHY and second in _AUX - _MODALS and len(words) > 2 and words[2] in _I:
        return False
    return second not in _SUBORDINATE_SUBJECTS


def _is_request(words: list) -> bool:
    """True when the clause *words* (fillers removed) asks the assistant for
    something: a question, an instruction, or "I want to know ..."."""
    if not words:
        return False
    second = words[1] if len(words) > 1 else ""
    if _is_wh_question(words):
        return True
    if words[0] in _AUX and second in _SUBJECTS:
        return True
    if second in _ME or second == "us":
        return (not (words[0] in _NAMING and second in ("me", "mich"))
                and words[0] not in _SUBJECTS
                and words[0] not in ("there", "with", "for", "to", "about", "like"))
    if words[0] in _IMPERATIVES:
        return True
    joined = " ".join(words)
    return bool(_ASKS.match(joined)) and not _STATES_LIKING.match(joined)


def _named_things(text: str, words: list) -> set:
    """The stems of what a request calls the user's own ("my cat Tom": cat, Tom).
    "What is my name" names nothing: it asks for the value."""
    spelled = [m.group() for m in _WORD.finditer(text.replace("\u2019", "'"))]
    out: set = set()
    for i, word in enumerate(words):
        if word not in _MY or i + 1 >= len(words) or words[i + 1] in _OTHER_PERSON:
            continue
        before = words[i - 1] if i else ""
        if before in ("is", "are", "was", "were", "ist", "sind", "war", "lautet", "heißt",
                      "what's", "who's", "where's", "how's"):
            continue
        end = i + 2
        while end < len(spelled) and spelled[end][:1].isupper():
            end += 1
        out |= stems(" ".join(spelled[i + 1:end]))
    return out


def _request_stems(text: str, words: list) -> set:
    """What a request or question *text* still states about the user: the things it
    calls theirs and an embedded clause of their own ("a restaurant I liked")."""
    out = _named_things(text, words)
    for i, word in enumerate(words):
        if word not in _I or i < 2 or i + 1 >= len(words):
            continue
        if (words[i - 1] in _INDIRECT or words[i - 1] in _AUX
                or words[i + 1] in _MODALS or word in ("i'd", "i'll")
                or _ASKS.match(" ".join(words[i:]))):
            continue
        spelled = [m for m in _WORD.finditer(text.replace("\u2019", "'"))]
        out |= stems(text[spelled[i].start():])
    return out


def _opening(text: str) -> tuple:
    """``(body, words)`` of the clause *text* without the fillers it opens with
    ("Ok, now please ...")."""
    spelled = list(_WORD.finditer(text.replace("\u2019", "'")))
    n = 0
    while n + 1 < len(spelled) and spelled[n].group().lower() in _FILLERS:
        n += 1
    body = text[spelled[n].start():] if spelled else ""
    return body, [m.group().lower() for m in spelled[n:]]


def _read_clause(text: str, dependent: bool, asked: bool, last_question: bool) -> tuple:
    """``(stems, states, request)``: what the clause *text* says about the user,
    whether it is a statement about them, and whether it asks the assistant for
    something. The stems are None when the clause carries no first-person word and
    is no request, so it belongs to its neighbours.
    *asked* is whether the clause before it was a request, *last_question* whether
    this clause ends in a question mark."""
    body, words = _opening(text)
    if not words:
        return None, False, False
    if _is_request(words):
        return _request_stems(body, words), False, True
    has_i = any(w in _I for w in words)
    has_me = any(w in _ME for w in words)
    mine = [i for i, w in enumerate(words) if w in _MY]
    if not (has_i or has_me or mine):
        return None, False, False
    if has_i:
        at = next(i for i, w in enumerate(words) if w in _I)
        modal = at + 1 < len(words) and words[at + 1] in _MODALS
        if modal and ((dependent and asked) or last_question or words[0] in _WH):
            return _named_things(body, words), False, True
        return stems(body), True, False
    if any(i + 1 >= len(words) or words[i + 1] not in _OTHER_PERSON for i in mine):
        return stems(body), True, False
    if has_me and not last_question and not (
            {"please", "bitte", "you", "du"} & set(_tokens(text))):
        for i, w in enumerate(words):
            if w in _ME and i >= 2 and words[i - 1] in _IMPERATIVES and words[i - 2] in _LETS_IN:
                return set(), False, True
        return stems(body), True, False
    return set(), False, has_me


def self_statement_stems(user_text: str) -> set:
    """The first letters of every content word of the clauses of *user_text* in
    which the user states something about themselves. A clause without a
    first-person word ("..., hiking and chess") goes with the clause before it, or
    with the one after it when it opens the sentence."""
    out: set = set()
    for sentence in _SENTENCE_SPLIT.split(user_text):
        clauses = _clauses(sentence)
        question = sentence.rstrip().endswith("?")
        opened_with: set = set()
        about_user = asked = False
        first = True
        for n, (text, dependent) in enumerate(clauses):
            got, about, request = _read_clause(text, dependent, asked,
                                               question and n == len(clauses) - 1)
            if got is None:
                if first:
                    opened_with |= stems(text)
                elif about_user:
                    out |= stems(text)
                continue
            first, asked, about_user = False, request, about
            out |= got
            if about_user:
                out |= opened_with
            opened_with = set()
    return out
