# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reranker-score relevance gate against RECORDED real-model signals.

``tests/fixtures/rag_rerank_eval/replay.json.gz`` holds the embeddings and the
reranker scores that bge-small-en-v1.5, Qwen3-Reranker-0.6B and
bge-reranker-v2-m3 (the measured Q8_0 files) produced for the fixture corpus, the
labelled on-topic and off-topic questions and the follow-up pairs. Replaying them
runs the real ``Collection.query`` gates (floor, reranker score, previous-message
context) with no model, so a change to the gates or to the calibrated minimum
scores moves recall and precision here exactly as it would with the models.
Re-record with ``python -m tests.rag_rerank_eval --embed --reranker A --reranker B
--record PATH`` after changing the corpus, the chunker or a model.
"""

from __future__ import annotations

import gzip
import json

import pytest

from localm.rag.rerank import CALIBRATED_MIN_SCORES
from tests import rag_rerank_eval as ev


@pytest.fixture(scope="module")
def replay():
    return ev.load_replay()


@pytest.fixture(scope="module")
def replayed(replay, tmp_path_factory):
    suite, coll = ev.replay_suite(replay, tmp_path_factory.mktemp("replay"))
    return suite, coll


def _gate_names(replay):
    return {name: (f"{name}: floor, then rerank",
                   f"{name}: score >= {info['min_score']:g}",
                   f"{name}: score >= {info['min_score']:g}, no context",
                   f"{name}: score >= {info['min_score']:g}, context")
            for name, info in replay.data["rerank"].items()}


class TestRecordingIsCurrent:
    def test_the_recorded_chunks_are_the_corpus_chunks(self, replay, replayed):
        _, coll = replayed
        assert ev.chunk_digest(coll) == replay.data["chunk_digest"], \
            "the corpus or chunker changed: re-record the replay"
        assert coll._vectors and all(coll._vectors), "a chunk has no recorded vector"
        assert coll.vector_degrade_reason is None

    def test_the_minimum_scores_are_the_shipped_calibration(self, replay):
        assert set(replay.data["rerank"]) >= {"qwen3-reranker-0.6b-q8_0",
                                              "bge-reranker-v2-m3-Q8_0"}
        for name, info in replay.data["rerank"].items():
            assert CALIBRATED_MIN_SCORES[info["sha256"]] == info["min_score"], name

    def test_the_gates_reproduce_the_recorded_results(self, replay, replayed):
        suite, _ = replayed
        assert json.loads(json.dumps(suite)) == replay.data["expected"]

    def test_every_call_the_gates_make_was_recorded(self, replay, replayed):
        assert replay.misses == [], replay.misses[:3]

    def test_a_call_the_recording_lacks_is_loud(self, replay):
        fn = replay.rerank_fn("qwen3-reranker-0.6b-q8_0")
        with pytest.raises(ev.ReplayMiss):
            fn("a question nobody recorded", ["some text"])
        with pytest.raises(ev.ReplayMiss):
            replay.embed_fn(["a text nobody recorded"])
        assert len(replay.misses) == 2
        replay.misses.clear()


class TestScoreGateWithRecordedModels:
    def test_the_score_gate_answers_more_without_leaking_more(self, replay, replayed):
        suite, _ = replayed
        for name, (before, gate, _, _) in _gate_names(replay).items():
            g0, g1 = suite["gates"][before], suite["gates"][gate]
            assert g1["recall"] >= g0["recall"] + 0.15, (name, g0, g1)
            assert g1["precision"] >= g0["precision"], (name, g0, g1)
            assert g1["answered"] > g0["answered"], name

    def test_unrelated_chitchat_and_conversation_questions_return_nothing(
            self, replay, replayed):
        suite, _ = replayed
        for name, (_, gate, _, _) in _gate_names(replay).items():
            categories = suite["gates"][gate]["categories"]
            for category in ("unrelated", "chitchat", "conversation"):
                assert categories[category] == 1.0, (name, category)

    def test_the_floor_alone_is_the_weakest_gate(self, replay, replayed):
        suite, _ = replayed
        floor = suite["gates"]["floor only"]
        for name, (_, gate, _, _) in _gate_names(replay).items():
            assert suite["gates"][gate]["recall"] > floor["recall"], name

    def test_the_minimum_score_decides_what_is_answered(self, replay, replayed):
        _, coll = replayed
        name = "qwen3-reranker-0.6b-q8_0"
        fn = replay.rerank_fn(name)
        queries, offtopic = ev.load_queries(), ev.load_offtopic()

        def answered(minimum):
            return ev.evaluate_gate(coll, queries, offtopic, embed_fn=replay.embed_fn,
                                    rerank_fn=fn, min_score=minimum)["answered"]

        only_the_floor = 1 / len(queries)
        assert answered(-1.0) >= answered(0.5) > answered(0.9) > answered(2.0)
        assert answered(2.0) <= only_the_floor, \
            "above every score only a conversation reference, judged by the floor, answers"


class TestFollowUpsWithRecordedModels:
    def test_the_previous_message_lets_a_follow_up_find_its_topic(
            self, replay, replayed):
        suite, _ = replayed
        for name, (_, _, without, with_context) in _gate_names(replay).items():
            plain, ctx = suite["followups"][without], suite["followups"][with_context]
            assert ctx["recall"] >= plain["recall"] + 0.3, (name, plain, ctx)
            assert ctx["precision"] >= 0.85, (name, ctx)

    def test_follow_ups_without_context_keep_the_conversation_rule(
            self, replay, replayed):
        suite, _ = replayed
        for name, (_, _, without, _) in _gate_names(replay).items():
            assert suite["followups"][without]["precision"] == 1.0, name


class TestRecorderRoundTrip:
    """``record_replay`` with stand-in models: what it writes reloads into a replay
    that reproduces the gate results it reported."""

    def test_recorded_file_replays_to_the_reported_results(self, tmp_path):
        import hashlib

        def embed(texts):
            out = []
            for t in texts:
                digest = hashlib.sha256(t.encode()).digest()
                out.append([(b - 127.5) / 127.5 for b in digest[:16]])
            return out

        def reranker(query, texts):
            return [(sum(map(ord, query + t)) % 997) / 997.0 for t in texts]

        path = tmp_path / "replay.json.gz"
        reported = ev.record_replay(
            path, embed, {"stub": (reranker, "0" * 64, 0.5)},
            model_name="bge-small-en-v1.5")
        replay = ev.load_replay(path)
        suite, _ = ev.replay_suite(replay, tmp_path / "again")
        assert json.loads(json.dumps(suite)) == json.loads(json.dumps(reported))
        assert json.loads(json.dumps(reported)) == replay.data["expected"]
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            assert json.load(fh)["rerank"]["stub"]["sha256"] == "0" * 64
        assert len(replay.data["rerank"]["stub"]["calls"]) > 100
