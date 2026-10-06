# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generic-word recall, stored junk and the session floor (follow-up to the
trivial-episode filter).

- A word present in many records of a corpus is generic and no longer matches.
- The coder episode store applies the same lexical gate as the chat memory store.
- Already-stored episodes are judged once; a DROP is archived and removed.
- Greeting and acknowledgement words do not count toward the session floor.
"""

from __future__ import annotations

import pytest

from localm.memory import MemoryRecord, MemoryStore, judge_episode
from localm.memory.relevance import (GENERIC_MIN_CORPUS, generic_tokens,
                                     lexical_match)
from localm.plugins.builtin.memory import plug

NOW = 1_800_000_000.0


# ------------------------------------------------------------- relevance unit #

def test_small_corpus_has_no_generic_tokens():
    corpus = [{"local", "model"}] * (GENERIC_MIN_CORPUS - 1)
    assert generic_tokens(corpus) == frozenset()


def test_token_in_many_records_is_generic():
    corpus = [{"local", "model", f"topic{i}"} for i in range(10)]
    assert generic_tokens(corpus) == frozenset({"local", "model"})


def test_rare_token_is_not_generic():
    corpus = [{"local", f"topic{i}"} for i in range(10)]
    assert "topic3" not in generic_tokens(corpus)


@pytest.mark.parametrize("query,record,generic,expected", [
    ({"local"}, {"local", "model"}, frozenset(), True),            # one-word query
    ({"local", "server"}, {"local", "model"}, frozenset(), False),  # one of two
    ({"local", "model"}, {"local", "model"}, frozenset(), True),
    ({"local", "model"}, {"local", "model"}, frozenset({"local", "model"}), False),
    ({"local", "darts"}, {"local", "darts"}, frozenset({"local"}), True),
    (set(), {"local"}, frozenset(), False),
])
def test_lexical_match(query, record, generic, expected):
    assert lexical_match(query, record, generic) is expected


# ------------------------------------------------------ chat store: generic df #

def _episode_store(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(9):
        s.add(MemoryRecord(text=f"Worked on local model loading for topic{i}",
                           kind="episodic", source="synth", importance=0.4))
    s.add(MemoryRecord(text="Planned a darts tournament schedule for local clubs",
                       kind="episodic", source="synth", importance=0.4))
    return s


def test_generic_words_do_not_recall_episodes_in_a_large_store(tmp_path):
    s = _episode_store(tmp_path)
    assert s.recall("which local model fits my graphics card", k=6, now=NOW) == []


def test_distinctive_words_still_recall_in_a_large_store(tmp_path):
    s = _episode_store(tmp_path)
    hits = s.recall("when is the darts tournament", k=6, now=NOW)
    assert [h.text for h in hits] == [
        "Planned a darts tournament schedule for local clubs"]


def test_user_fact_is_not_subject_to_the_generic_filter(tmp_path):
    s = _episode_store(tmp_path)
    s.add(MemoryRecord(text="User prefers a local model", source="user"))
    hits = s.recall("which local model fits my graphics card", k=6, now=NOW)
    assert [h.text for h in hits] == ["User prefers a local model"]


# ---------------------------------------------------------- coder episode store #

def _coder_store(tmp_path, monkeypatch):
    from localm.plugins.coder import episodes as E
    monkeypatch.setattr(E, "_embed_fn", lambda: None)
    cwd = tmp_path / "proj"
    cwd.mkdir()
    st = E.EpisodeStore(cwd, root=tmp_path / "store")
    for t, s_, lesson in [
        ("Fix the flaky upload test", "Chased the flaky upload test to a cleanup race",
         "Wait for the cleanup fixture"),
        ("Load a local model with more GPU layers", "Raised n_gpu_layers so the local model fits",
         "Check free VRAM first"),
        ("Add a CSV export to the reports page", "Built a CSV export endpoint",
         "Stream rows"),
    ]:
        st.add(E.Episode(task=t, summary=s_, lesson=lesson))
    return st


@pytest.mark.parametrize("task", [
    "is the local server running",
    "which model should I use",
    "write a test for the parser",
])
def test_coder_one_generic_word_injects_nothing(tmp_path, monkeypatch, task):
    assert _coder_store(tmp_path, monkeypatch).search(task) == []


def test_coder_two_shared_words_still_recall(tmp_path, monkeypatch):
    hits = _coder_store(tmp_path, monkeypatch).search("raise the gpu layers for the local model")
    assert [h.task for h in hits] == ["Load a local model with more GPU layers"]


def test_coder_single_word_task_can_still_hit(tmp_path, monkeypatch):
    hits = _coder_store(tmp_path, monkeypatch).search("csv")
    assert [h.task for h in hits] == ["Add a CSV export to the reports page"]


# ------------------------------------------------------------------ judge unit #

@pytest.mark.parametrize("reply,keep", [
    ("DROP", False), ("drop.", False), ("**DROP**", False),
    ("<think>hmm KEEP?</think>\nDROP", False),
    ("KEEP", True), ("keep", True), ("", True), ("I am not sure", True),
])
def test_judge_episode(reply, keep):
    assert judge_episode(lambda p: reply, "Mentioned plans to play darts.") is keep


def test_judge_episode_model_failure_keeps():
    def boom(_p):
        raise RuntimeError("no model")
    assert judge_episode(boom, "Mentioned plans to play darts.") is True


# ------------------------------------------------------------ stored-junk prune #

@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    (tmp_path / "sessions").mkdir()
    return tmp_path


def _add(store, text, **kw):
    return store.add(MemoryRecord(text=text, kind="episodic", source="synth",
                                  importance=0.4, **kw))


def _judge_stub(calls):
    def complete(prompt):
        calls.append(prompt)
        return "DROP" if "darts" in prompt else "KEEP"
    return complete


def test_prune_removes_a_dropped_episode_and_archives_it(memhome):
    store = plug._chat_store()
    darts = _add(store, "Mentioned plans to play darts and test their local LLM runner.")
    real = _add(store, "Planned the sensor polling loop for the greenhouse controller.")
    calls: list = []
    assert plug._prune_trivial_episodes(store, _judge_stub(calls)) == 1
    live = {r.id: r for r in store.all()}
    assert darts.id not in live and real.id in live
    assert live[real.id].meta.get("judged") is True
    assert any(d.get("id") == darts.id for d in store.forgotten())


def test_prune_judges_each_episode_only_once(memhome):
    store = plug._chat_store()
    _add(store, "Planned the sensor polling loop for the greenhouse controller.")
    calls: list = []
    plug._prune_trivial_episodes(store, _judge_stub(calls))
    assert len(calls) == 1
    plug._prune_trivial_episodes(store, _judge_stub(calls))
    assert len(calls) == 1


def test_prune_leaves_user_facts_and_user_episodes_alone(memhome):
    store = plug._chat_store()
    store.add(MemoryRecord(text="User plays darts on Fridays", source="user"))
    store.add(MemoryRecord(text="Mentioned darts", kind="episodic", source="user"))
    calls: list = []
    assert plug._prune_trivial_episodes(store, _judge_stub(calls)) == 0
    assert calls == [] and len(store.all()) == 2


def test_prune_is_bounded_per_run(memhome):
    store = plug._chat_store()
    for i in range(plug.EPISODIC_JUDGE_MAX_PER_RUN + 3):
        _add(store, f"Planned darts event number {i} for the club")
    assert plug._prune_trivial_episodes(store, lambda p: "DROP") == \
        plug.EPISODIC_JUDGE_MAX_PER_RUN
    assert len(store.all()) == 3


def test_prune_keeps_the_record_when_the_archive_fails(memhome, monkeypatch):
    store = plug._chat_store()
    darts = _add(store, "Mentioned plans to play darts.")
    monkeypatch.setattr(type(store), "_archive_forgotten", lambda self, recs: False)
    assert plug._prune_trivial_episodes(store, lambda p: "DROP") == 0
    assert [r.id for r in store.all()] == [darts.id]


def test_prune_model_failure_keeps_everything(memhome):
    store = plug._chat_store()
    _add(store, "Mentioned plans to play darts.")

    def boom(_p):
        raise RuntimeError("no model")
    assert plug._prune_trivial_episodes(store, boom) == 0
    assert len(store.all()) == 1


# ------------------------------------------------------------- session floor #

@pytest.mark.parametrize("user_text,expected", [
    ("hi, thanks a lot please", False),
    ("ok great thanks", False),
    ("hello there, good morning", False),
    ("ok thanks, fix the upload failure", True),
])
def test_filler_words_do_not_count_toward_the_floor(user_text, expected):
    assert plug._is_substantive_session(f"User: {user_text}\nAssistant: Sure.") is expected
