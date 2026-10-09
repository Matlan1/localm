# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reranker support in the embedder worker: RANK pooling is honoured, a rank-pooled
model scores query / document pairs and refuses to embed, and the parent-side handle
chunks, pins and recovers rerank calls the way it does embed calls.

The native layer is faked at the _api boundary (one token per byte); the real
model path is covered by tests/test_rerank_real_gguf.py.
"""

import ctypes
import threading

import pytest

from localm.inference import embedder as emb
from localm.inference.backends.base import RerankInputError
from localm.inference.rerank_pairs import VocabSpecials

BOS, EOS, SEP = 1, 2, 3


class FakeNative:
    """Enough of the llama binding for GGUFEmbedder.rerank / embed: byte-per-token
    tokenizer, a ctypes-backed batch, and a decode that records what it was given
    and answers get_embeddings_seq from a scoring function of each sequence."""

    def __init__(self, n_cls_out=1, score=None):
        self.n_cls_out = n_cls_out
        self.score = score or (lambda toks: sum(toks) % 997 / 10.0)
        self.decodes = []            # per decode: {seq: [tokens]}
        self.positions = []          # per decode: {seq: [positions]}
        self.cleared = 0
        self.decode_code = 0
        self.pointer = {}            # seq -> override returned by get_embeddings_seq
        self._last = {}

    def llama_tokenize(self, vocab, raw, ln, buf, cap, add_special, parse_special):
        toks = [(b % 250) + 10 for b in raw[:ln]]
        if len(toks) > cap:
            return -len(toks)
        for i, t in enumerate(toks):
            buf[i] = t
        return len(toks)

    def llama_batch_init(self, n_tokens, embd, n_seq_max):
        class Batch:
            pass
        b = Batch()
        b.token = (ctypes.c_int32 * n_tokens)()
        b.pos = (ctypes.c_int32 * n_tokens)()
        b.n_seq_id = (ctypes.c_int32 * n_tokens)()
        b.logits = (ctypes.c_int8 * n_tokens)()
        self._seq_arrays = [(ctypes.c_int32 * 1)() for _ in range(n_tokens)]
        b.seq_id = (ctypes.POINTER(ctypes.c_int32) * n_tokens)(
            *[ctypes.cast(a, ctypes.POINTER(ctypes.c_int32)) for a in self._seq_arrays])
        b.n_tokens = 0
        return b

    def llama_batch_free(self, batch):
        pass

    def llama_memory_clear(self, mem, data):
        self.cleared += 1

    def llama_decode(self, ctx, batch):
        if self.decode_code:
            return self.decode_code
        seqs, poss = {}, {}
        for i in range(batch.n_tokens):
            seq = self._seq_arrays[i][0]
            seqs.setdefault(seq, []).append(batch.token[i])
            poss.setdefault(seq, []).append(batch.pos[i])
        self.decodes.append(seqs)
        self.positions.append(poss)
        self._last = seqs
        return 0

    def llama_get_embeddings_seq(self, ctx, seq):
        if seq in self.pointer:
            return self.pointer[seq]
        toks = self._last[seq]
        base = self.score(toks)
        return [base, 100.0 - base][: self.n_cls_out] + [0.0] * max(0, self.n_cls_out - 2)


def make_ranker(native=None, *, n_ctx=64, n_seq_max=32, template=None, pooling=emb._POOLING_RANK,
                specials=None, effective_seq_ctx=None, n_cls_out=1):
    native = native or FakeNative(n_cls_out=n_cls_out)
    e = emb.GGUFEmbedder.__new__(emb.GGUFEmbedder)
    e._api = native
    e._llama_token = ctypes.c_int32
    e._llama_pos = ctypes.c_int32
    e._llama_seq_id = ctypes.c_int32
    e._lock = threading.RLock()
    e.n_ctx = n_ctx
    e._effective_seq_ctx = n_ctx if effective_seq_ctx is None else effective_seq_ctx
    e._mem = object()
    e._vocab = None
    e._ctx = object()
    e.dim = 8
    e.model_path = "reranker.gguf"
    e._n_seq_max = n_seq_max
    e._pre_type = None
    e.pooling_type = pooling
    e.n_cls_out = n_cls_out
    e.cls_labels = []
    e._rerank_template = template
    e._specials = specials or VocabSpecials(
        bos=BOS, eos=EOS, sep=SEP, add_bos=True, add_eos=True, add_sep=True)
    return e, native


def pair_tokens(e, query, doc):
    from localm.inference.rerank_pairs import build_pair
    return build_pair(e._tokenize_plain, e._specials, e._rerank_template,
                      e._effective_seq_ctx, query, doc).tokens


class TestPoolingResolution:
    def test_a_declared_rank_model_is_never_pooled_any_other_way(self):
        for requested in (emb._POOLING_UNSET, emb.POOLING_AUTO, emb._POOLING_MEAN,
                          emb._POOLING_CLS, emb._POOLING_LAST, emb._POOLING_NONE):
            assert emb._effective_pooling(requested, emb._POOLING_RANK) == emb._POOLING_RANK

    def test_other_declarations_still_follow_the_requested_pooling(self):
        assert emb._effective_pooling(emb._POOLING_CLS, emb._POOLING_MEAN) == emb._POOLING_CLS
        assert emb._effective_pooling(emb._POOLING_UNSET, emb._POOLING_LAST) == emb._POOLING_LAST
        assert emb._effective_pooling(emb.POOLING_AUTO, emb._POOLING_CLS) == emb._POOLING_CLS
        assert emb._effective_pooling(emb._POOLING_UNSET, None) == emb._POOLING_DEFAULT

    def test_an_explicit_rank_request_for_a_model_declaring_nothing_is_honoured(self):
        assert emb._effective_pooling(emb._POOLING_RANK, None) == emb._POOLING_RANK

    def test_rank_is_not_a_choice_for_embedding_pooling(self):
        assert "rank" not in emb.POOLING_CHOICES
        assert emb.pooling_name(emb._POOLING_RANK) == "rank"

    def test_the_user_setting_cannot_name_rank(self):
        assert emb.resolve_pooling_setting("rank") == emb._POOLING_UNSET


class TestRerankScoring:
    def test_scores_come_back_one_per_pair_in_order(self):
        e, native = make_ranker()
        pairs = [("alpha", "one"), ("alpha", "two words"), ("beta", "x")]
        out = e.rerank(pairs)
        assert len(out) == 3
        for (q, d), res in zip(pairs, out, strict=True):
            toks = pair_tokens(e, q, d)
            assert res["scores"] == [native.score(toks)]
            assert res["tokens"] == len(toks)
            assert res["truncated"] is False

    def test_every_pair_is_decoded_as_its_own_sequence_from_position_zero(self):
        e, native = make_ranker()
        e.rerank([("q", "aa"), ("q", "bbb")])
        assert len(native.decodes) == 1
        seqs, poss = native.decodes[0], native.positions[0]
        assert sorted(seqs) == [0, 1]
        assert seqs[0] == pair_tokens(e, "q", "aa")
        assert seqs[1] == pair_tokens(e, "q", "bbb")
        assert poss[0] == list(range(len(seqs[0])))
        assert poss[1] == list(range(len(seqs[1])))

    def test_pairs_are_packed_into_groups_of_at_most_n_seq_max(self):
        e, native = make_ranker(n_seq_max=2)
        out = e.rerank([("q", "a"), ("q", "b"), ("q", "c")])
        assert [len(d) for d in native.decodes] == [2, 1]
        assert len(out) == 3

    def test_a_group_never_exceeds_the_summed_token_budget(self):
        e, native = make_ranker(n_ctx=24, n_seq_max=32)
        e.rerank([("q", "aaaa"), ("q", "bbbb"), ("q", "cccc")])
        for decode in native.decodes:
            assert sum(len(t) for t in decode.values()) <= 24
        assert len(native.decodes) > 1

    def test_results_stay_aligned_when_grouping_reorders_nothing(self):
        e, native = make_ranker(n_seq_max=2)
        pairs = [("q", w) for w in ("a", "bb", "ccc", "dddd", "eeeee")]
        out = e.rerank(pairs)
        assert [r["tokens"] for r in out] == [len(pair_tokens(e, *p)) for p in pairs]

    def test_every_label_output_is_returned_in_label_order(self):
        e, native = make_ranker(n_cls_out=2)
        (res,) = e.rerank([("q", "doc")])
        assert len(res["scores"]) == 2
        assert res["scores"][0] == native.score(pair_tokens(e, "q", "doc"))
        assert res["scores"][0] + res["scores"][1] == pytest.approx(100.0)

    def test_a_document_cut_to_fit_is_flagged(self):
        e, _native = make_ranker(n_ctx=12)
        (res,) = e.rerank([("q", "x" * 200)])
        assert res["truncated"] is True
        assert res["tokens"] == 12

    def test_the_rerank_template_is_used_when_the_model_has_one(self):
        e, native = make_ranker(template="[{query}]{document}")
        e.rerank([("qq", "dd")])
        text_tokens = [(b % 250) + 10 for b in b"[qq]dd"]
        assert native.decodes[0][0] == text_tokens

    def test_a_query_with_no_room_for_a_document_is_refused_before_any_decode(self):
        e, native = make_ranker(n_ctx=8)
        with pytest.raises(RerankInputError):
            e.rerank([("q" * 30, "d")])
        assert native.decodes == []

    def test_no_pairs_make_no_native_call(self):
        e, native = make_ranker()
        assert e.rerank([]) == []
        assert native.decodes == []

    def test_a_nonzero_decode_return_raises(self):
        e, native = make_ranker()
        native.decode_code = -3
        with pytest.raises(RuntimeError, match=r"batched rerank decode failed \(code -3\)"):
            e.rerank([("q", "d")])

    def test_a_missing_score_pointer_raises_rather_than_scoring_zero(self):
        e, native = make_ranker()
        native.pointer = {0: None}
        with pytest.raises(RuntimeError, match="no score for sequence 0"):
            e.rerank([("q", "d")])

    def test_a_non_finite_score_raises(self):
        e, native = make_ranker()
        native.pointer = {0: [float("nan")]}
        with pytest.raises(RuntimeError, match="non-finite score"):
            e.rerank([("q", "d")])

    def test_a_closed_embedder_refuses(self):
        e, _ = make_ranker()
        e._ctx = None
        with pytest.raises(RuntimeError, match="closed"):
            e.rerank([("q", "d")])


class TestKindRefusals:
    def test_a_rank_pooled_model_refuses_to_embed(self):
        e, native = make_ranker()
        with pytest.raises(RuntimeError, match="is a reranker"):
            e.embed(["hello"])
        assert native.decodes == []

    def test_an_embedding_model_refuses_to_rerank(self):
        e, native = make_ranker(pooling=emb._POOLING_MEAN)
        with pytest.raises(RuntimeError, match="not rank-pooled"):
            e.rerank([("q", "d")])
        assert native.decodes == []

    def test_a_non_rank_model_embeds_as_before(self):
        e, _ = make_ranker(pooling=emb._POOLING_MEAN)
        e._api.llama_get_embeddings_seq = lambda ctx, seq: [3.0, 4.0] + [0.0] * 6
        vecs = e.embed(["hello", "world"])
        assert [v[:2] for v in vecs] == [pytest.approx([0.6, 0.8])] * 2


class FakeRunner:
    """Parent-side runner double: records rerank calls, can be told to die."""

    def __init__(self, owner=None):
        self.calls = []
        self.alive = True
        self.fail_with = None
        self.pinned = []
        self.owner = owner

    def is_alive(self):
        return self.alive

    def rerank(self, pairs):
        if self.owner is not None:
            self.pinned.append(self.owner.active_requests)
        if self.fail_with is not None:
            err, self.fail_with = self.fail_with, None
            self.alive = False
            raise err
        self.calls.append(list(pairs))
        return [{"scores": [float(len(q) + len(d))], "tokens": 1, "truncated": False}
                for q, d in pairs]

    def embed(self, texts):
        raise AssertionError("embed must not reach the worker")

    def shutdown(self, grace=5.0):
        self.alive = False


def make_handle(pooling=emb._POOLING_RANK):
    h = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    h.model_path = "reranker.gguf"
    h.n_gpu_layers = 0
    h.gpu_fallback_reason = "forced cpu"
    h._rpc_lock = threading.RLock()
    h.active_requests = 0
    h.effective_pooling = pooling
    h.n_cls_out = 1
    h.cls_labels = []
    h._runner = FakeRunner(h)
    return h


class TestIsolatedRerank:
    def test_scores_are_returned_in_order(self):
        h = make_handle()
        out = h.rerank([("ab", "c"), ("abc", "dd")])
        assert [r["scores"] for r in out] == [[3.0], [5.0]]

    def test_a_long_pair_list_goes_to_the_worker_in_chunks_and_keeps_its_order(self, monkeypatch):
        monkeypatch.setattr(emb, "RERANK_PAIRS_PER_CALL", 2)
        h = make_handle()
        pairs = [("q", "d" * i) for i in range(5)]
        out = h.rerank(pairs)
        assert [len(c) for c in h._runner.calls] == [2, 2, 1]
        assert [r["scores"][0] for r in out] == [1.0 + i for i in range(5)]

    def test_the_handle_stays_pinned_for_the_whole_call(self, monkeypatch):
        monkeypatch.setattr(emb, "RERANK_PAIRS_PER_CALL", 1)
        h = make_handle()
        h.rerank([("q", "a"), ("q", "b")])
        assert min(h._runner.pinned) >= 1
        assert h.active_requests == 0

    def test_the_pin_is_released_when_the_worker_fails(self):
        h = make_handle()
        h._runner.fail_with = RuntimeError("boom")
        h.n_gpu_layers = 0
        with pytest.raises(RuntimeError, match="boom"):
            h.rerank([("q", "a")])
        assert h.active_requests == 0

    def test_a_dead_worker_is_reloaded_before_the_call(self):
        h = make_handle()
        h._runner.alive = False
        fresh = FakeRunner(h)

        def reload():
            h._runner = fresh
        h._reload = reload
        h.rerank([("q", "a")])
        assert fresh.calls == [[("q", "a")]]

    def test_a_gpu_crash_falls_back_to_cpu_once_and_retries_the_call(self):
        h = make_handle()
        h.n_gpu_layers = 99
        h.gpu_fallback_reason = None
        h._runner.fail_with = RuntimeError("worker exited")
        fresh = FakeRunner(h)

        def reload():
            h._runner = fresh
        h._reload = reload
        out = h.rerank([("q", "a")])
        assert out[0]["scores"] == [2.0]
        assert h.n_gpu_layers == 0 and h.gpu_fallback_reason

    def test_a_model_that_is_not_rank_pooled_refuses_without_calling_the_worker(self):
        h = make_handle(pooling=emb._POOLING_MEAN)
        with pytest.raises(RuntimeError, match="not loaded for reranking"):
            h.rerank([("q", "a")])
        assert h._runner.calls == []

    def test_a_reranker_handle_refuses_embed_without_calling_the_worker(self):
        h = make_handle()
        with pytest.raises(RuntimeError, match="is a reranker"):
            h.embed(["x"])

    def test_no_pairs_make_no_worker_call(self):
        h = make_handle()
        assert h.rerank([]) == []
        assert h._runner.calls == []


class TestGetEmbedderRefusesAReranker:
    def test_a_reranker_configured_as_the_embedding_model_is_reported_not_used(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"embedding_model": "rr", "n_gpu_layers": 99, "net_mode": "off"})
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda *, allow_download=None, on_progress=None: "/models/rr.gguf")
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
        closed = []

        class Ranker:
            dim = 8
            effective_pooling = emb._POOLING_RANK

            def __init__(self, path, **kw):
                self.model_path = path

            def close(self):
                closed.append(True)

        monkeypatch.setattr(emb, "IsolatedEmbedder", Ranker)
        emb.reset_embedder()
        try:
            assert emb.get_embedder() is None
            assert "reranker" in emb.last_error()
            assert closed == [True]
            assert emb.loaded_dim() is None
        finally:
            emb.reset_embedder()


class TestWorkerRerankCommand:
    """The isolated worker's dispatch loop (driven in-process, no subprocess)."""

    def _drive(self, monkeypatch, commands, *, pooling=emb._POOLING_RANK, rerank=None):
        import contextlib
        import queue as _q

        from localm.inference import _embedder_runner as runner_mod

        events = []

        @contextlib.contextmanager
        def spy_dedup():
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr("localm.debuglog.dedup_native_stderr", spy_dedup)

        class Stub:
            def __init__(self, **kwargs):
                self.dim = 4
                self.declared_pooling = None
                self.pooling_type = pooling
                self.n_ctx = 512

            def rank_head_meta(self):
                return {"n_cls_out": 2, "cls_labels": ["yes", "no"], "has_rerank_template": True}

            def embed(self, texts):
                events.append(f"embed:{len(texts)}")
                return [[0.0] * 4 for _ in texts]

            def rerank(self, pairs):
                events.append(f"rerank:{len(pairs)}")
                if rerank is not None:
                    return rerank(pairs)
                return [{"scores": [1.0], "tokens": 2, "truncated": False} for _ in pairs]

            def close(self):
                events.append("close")

        monkeypatch.setattr("localm.inference.embedder.GGUFEmbedder", Stub)
        req_q, resp_q = _q.Queue(), _q.Queue()
        for cmd in commands:
            req_q.put(cmd)
        runner_mod._runner_main(req_q, resp_q)
        out = []
        while not resp_q.empty():
            out.append(resp_q.get_nowait())
        return out, events

    LOAD = ("load", dict(model_path="x.gguf", n_gpu_layers=0, n_ctx=512, pooling_type=4))

    def test_a_rank_pooled_load_reports_the_classifier_head(self, monkeypatch):
        out, _ = self._drive(monkeypatch, [self.LOAD, ("shutdown", None)])
        meta = out[0][1]
        assert out[0][0] == "ok"
        assert meta["effective_pooling"] == emb._POOLING_RANK
        assert meta["n_cls_out"] == 2 and meta["cls_labels"] == ["yes", "no"]
        assert meta["has_rerank_template"] is True

    def test_any_other_load_reports_no_classifier_head(self, monkeypatch):
        out, _ = self._drive(monkeypatch, [self.LOAD, ("shutdown", None)], pooling=emb._POOLING_MEAN)
        assert out[0][0] == "ok" and "n_cls_out" not in out[0][1]

    def test_rerank_returns_the_workers_results(self, monkeypatch):
        out, events = self._drive(monkeypatch, [self.LOAD, ("rerank", [("q", "a"), ("q", "b")]),
                                                ("shutdown", None)])
        assert out[1] == ("ok", [{"scores": [1.0], "tokens": 2, "truncated": False}] * 2)
        assert "rerank:2" in events

    def test_rerank_and_embed_share_one_stderr_scope(self, monkeypatch):
        _, events = self._drive(monkeypatch, [self.LOAD, ("rerank", [("q", "a")]), ("embed", ["x"]),
                                              ("rerank", [("q", "b")]), ("shutdown", None)])
        assert events == ["enter", "rerank:1", "embed:1", "rerank:1", "exit", "close"]

    def test_a_query_with_no_room_comes_back_as_a_typed_refusal_and_the_worker_keeps_serving(self, monkeypatch):
        def refuse(pairs):
            raise RerankInputError("the query takes 99 tokens")

        out, _ = self._drive(monkeypatch, [self.LOAD, ("rerank", [("q", "a")]), ("embed", ["x"]),
                                           ("shutdown", None)], rerank=refuse)
        assert out[1] == ("error", "the query takes 99 tokens", "RerankInputError")
        assert out[2][0] == "ok"

    def test_a_generic_failure_is_an_untagged_error(self, monkeypatch):
        def boom(pairs):
            raise RuntimeError("decode failed")

        out, _ = self._drive(monkeypatch, [self.LOAD, ("rerank", [("q", "a")]), ("shutdown", None)],
                             rerank=boom)
        assert out[1] == ("error", "decode failed")


class TestParentRunnerRerank:
    def _runner(self, responses):
        import queue as _q

        from localm.inference._embedder_runner import EmbedderRunner

        r = EmbedderRunner()
        r._req_q, r._resp_q = _q.Queue(), _q.Queue()
        r._proc = type("P", (), {"is_alive": lambda self: True})()
        for resp in responses:
            r._resp_q.put(resp)
        return r

    def test_the_pairs_are_sent_as_a_rerank_command_and_the_results_returned(self):
        r = self._runner([("ok", [{"scores": [0.5]}])])
        assert r.rerank([("q", "d")]) == [{"scores": [0.5]}]
        assert r._req_q.get_nowait() == ("rerank", [("q", "d")])

    def test_a_typed_refusal_is_raised_typed_without_stopping_the_worker(self):
        r = self._runner([("error", "the query takes 99 tokens", "RerankInputError")])
        with pytest.raises(RerankInputError, match="99 tokens"):
            r.rerank([("q", "d")])
        assert r._proc is not None

    def test_an_untagged_error_is_a_runtime_error(self):
        r = self._runner([("error", "decode failed")])
        with pytest.raises(RuntimeError, match="decode failed"):
            r.rerank([("q", "d")])
