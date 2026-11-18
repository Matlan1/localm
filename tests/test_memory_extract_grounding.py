# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fact the extractor proposes must be supported by what the user wrote: a
sentence taken from the assistant's own reply is never stored as a fact about
the user."""

from __future__ import annotations

import json

from localm.memory.consolidate import extract

SESSION = (
    "User: Write 300 words about rivers.\n"
    "Assistant: Rivers carve valleys across continents over millennia. "
    "Their deltas feed vast wetlands and shape human settlement."
)


def _model(*facts):
    return lambda prompt: json.dumps(
        {"facts": [{"fact": f, "confidence": 0.9} for f in facts]})


def test_a_fact_taken_from_the_assistants_essay_is_dropped():
    out = extract(_model(
        "Rivers carve valleys across continents over millennia.",
        "Their deltas feed vast wetlands and shape human settlement."), SESSION)
    assert out == []


def test_a_fact_the_user_stated_is_kept_even_when_reworded():
    session = ("User: I love playing RPG games and I prefer metric units.\n"
               "Assistant: Noted!")
    out = extract(_model("User loves playing RPG games",
                         "User prefers metric units"), session)
    assert [c["text"] for c in out] == [
        "User loves playing RPG games", "User prefers metric units"]


def test_a_transcript_without_user_turns_is_not_filtered():
    out = extract(_model("The project uses sensor polling"),
                  "plain notes with no turn markers")
    assert [c["text"] for c in out] == ["The project uses sensor polling"]
