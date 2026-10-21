# SPDX-License-Identifier: AGPL-3.0-or-later
"""A word that appears in many of a store's facts does not, on its own, make a fact
eligible for recall; a query whose only content word is that word still recalls."""

from localm.memory.store import MemoryRecord, MemoryStore

FACTS = [
    "I like to use the standard library whenever possible",
    "I use vim as my editor",
    "I use Linux on my laptop at work",
    "I use dark mode in every app",
    "I use pnpm for JavaScript projects",
    "Project Atlas stores billing data in PostgreSQL",
    "I keep pelicans as a hobby on weekends",
    "My name is Sam",
    "I live in Lisbon",
    "I prefer tabs over spaces",
    "My cat is called Miso",
    "I work as a backend engineer",
]


def _store(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for text in FACTS:
        s.add(MemoryRecord(text=text, source="user", importance=0.5))
    return s


def _texts(store, query):
    return [r.text for r in store.recall(query, k=6)]


def test_a_frequent_word_does_not_drag_unrelated_facts_behind_a_real_match(tmp_path):
    got = _texts(_store(tmp_path), "what database does Project Atlas use for billing")
    assert got == ["Project Atlas stores billing data in PostgreSQL"]


def test_a_distinctive_word_still_recalls_its_fact(tmp_path):
    s = _store(tmp_path)
    assert _texts(s, "pelicans hobby weekends") == ["I keep pelicans as a hobby on weekends"]
    assert _texts(s, "what editor do I use") == ["I use vim as my editor"]


def test_a_query_made_only_of_the_frequent_word_still_recalls(tmp_path):
    assert len(_texts(_store(tmp_path), "use")) == 5


def test_a_small_store_keeps_the_single_word_hit(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for text in FACTS[:5]:
        s.add(MemoryRecord(text=text, source="user", importance=0.5))
    assert len(_texts(s, "what database does Project Atlas use for billing")) == 5
