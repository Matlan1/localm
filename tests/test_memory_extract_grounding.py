# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fact the extractor proposes must be about the user and supported by a
sentence in which the user speaks of themselves: a sentence taken from the
assistant's own reply, or from a request about the world, is never stored as a
fact about the user."""

from __future__ import annotations

import json

from localm.memory.consolidate import extract

ESSAY = (
    "Assistant: Rivers carve valleys across continents over millennia. "
    "Their deltas feed vast wetlands and shape human settlement."
)
ESSAY_FACTS = (
    "Rivers carve valleys across continents over millennia.",
    "Their deltas feed vast wetlands and shape human settlement.",
)


def _model(*facts):
    return lambda prompt: json.dumps(
        {"facts": [{"fact": f, "confidence": 0.9} for f in facts]})


def _kept(facts, session):
    return [c["text"] for c in extract(_model(*facts), session)]


def test_a_request_with_no_first_person_sentence_stores_nothing():
    session = "User: Write 300 words about rivers.\n" + ESSAY
    assert _kept(ESSAY_FACTS + ("User is interested in rivers",), session) == []


def test_a_paraphrase_of_a_first_person_statement_is_kept():
    assert _kept(["The user follows a vegan diet."],
                 "User: I'm vegan.\nAssistant: Noted!") == [
        "The user follows a vegan diet."]


def test_facts_from_a_self_introduction_are_kept():
    facts = ["User is named Ana", "User lives in Porto"]
    assert _kept(facts, "User: My name is Ana and I live in Porto.\n"
                 "Assistant: Hello Ana!") == facts


def test_a_kayaker_asking_about_rivers_keeps_the_kayaking_fact_only():
    session = ("User: Write 300 words about rivers. I kayak on rivers every "
               "weekend.\n" + ESSAY)
    kept = _kept(ESSAY_FACTS + ("User kayaks on rivers every weekend",), session)
    assert kept == ["User kayaks on rivers every weekend"]


def test_a_transcript_without_user_turns_is_not_filtered():
    assert _kept(["The project uses sensor polling"],
                 "plain notes with no turn markers") == [
        "The project uses sensor polling"]
