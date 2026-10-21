# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face cross-encoders (``*ForSequenceClassification`` with one output) as rerankers.

``localm add`` registers such a folder as an embedding model flagged as a reranker;
``/v1/rerank`` and ``localm rerank`` then score pairs through an isolated HF worker
and return the head's raw logit. The real-model tests save a tiny random
BertForSequenceClassification and compare the scores with a forward pass computed
here.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from localm.config import load_registry
from localm.inference import reranker
from localm.inference.backends.base import RerankerHeadMissingError, RerankInputError
from localm.inference.hf_head_check import classifier_head_present
from localm.model_manager import add_local
from localm.model_manager.registry import (
    _detect_local_model_type,
    entry_is_reranker,
    hf_reranker_state,
)

needs_stack = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("tokenizers") is None,
    reason="needs torch, transformers and tokenizers")


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as mm
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "ensure_dirs", lambda: (home / "models").mkdir(parents=True, exist_ok=True))
    return home


@pytest.fixture(autouse=True)
def _release_reranker():
    """Every test scores on the CPU (``embedding_gpu_layers`` 0), in a fresh worker."""
    from localm.config import update_config
    update_config(lambda c: c.__setitem__("embedding_gpu_layers", 0))
    reranker.reset_reranker()
    yield
    reranker.reset_reranker()


def _config(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    return folder


class TestConfigState:
    def test_single_score_head_is_a_reranker(self, tmp_path):
        folder = _config(tmp_path / "m", architectures=["XLMRobertaForSequenceClassification"],
                         id2label={"0": "LABEL_0"})
        assert hf_reranker_state(folder) is True

    def test_num_labels_one_without_id2label_is_a_reranker(self, tmp_path):
        folder = _config(tmp_path / "m", architectures=["BertForSequenceClassification"],
                         num_labels=1)
        assert hf_reranker_state(folder) is True

    def test_a_multi_label_classifier_is_not(self, tmp_path):
        folder = _config(tmp_path / "m", architectures=["BertForSequenceClassification"],
                         id2label={"0": "neg", "1": "neu", "2": "pos"})
        assert hf_reranker_state(folder) is False

    def test_a_plain_encoder_is_not(self, tmp_path):
        assert hf_reranker_state(_config(tmp_path / "m", architectures=["BertModel"])) is False

    def test_an_unreadable_config_is_unknown(self, tmp_path):
        assert hf_reranker_state(tmp_path) is None
        (tmp_path / "config.json").write_text("[1]", encoding="utf-8")
        assert hf_reranker_state(tmp_path) is None

    def test_the_registry_types_a_cross_encoder_as_embedding(self, tmp_path):
        folder = _config(tmp_path / "m", architectures=["BertForSequenceClassification"],
                         id2label={"0": "LABEL_0"})
        assert _detect_local_model_type(folder, is_gguf=False, is_hf=True)[0] == "embedding"

    def test_the_registry_leaves_a_multi_label_classifier_unknown(self, tmp_path):
        folder = _config(tmp_path / "m", architectures=["BertForSequenceClassification"],
                         id2label={"0": "a", "1": "b"})
        assert _detect_local_model_type(folder, is_gguf=False, is_hf=True)[0] == "unknown"

    @pytest.mark.parametrize("where", ["tokenizer_config", "jinja", "json"])
    def test_a_single_score_head_with_a_chat_template_is_a_reward_model(self, tmp_path, where):
        folder = _config(tmp_path / "m", architectures=["LlamaForSequenceClassification"],
                         id2label={"0": "LABEL_0"})
        if where == "tokenizer_config":
            (folder / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
        elif where == "jinja":
            (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        else:
            (folder / "chat_template.json").write_text(
                json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
        assert hf_reranker_state(folder) is False
        assert _detect_local_model_type(folder, is_gguf=False, is_hf=True)[0] == "unknown"

    def test_a_reward_model_stored_as_a_reranker_is_classified_again(
            self, tmp_path, isolated_home):
        import localm.model_manager as mm
        folder = _config(tmp_path / "rm", architectures=["LlamaForSequenceClassification"],
                         id2label={"0": "LABEL_0"})
        (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        entry = {"model_type": "embedding", "reranker": True, "path": str(folder)}
        mm.update_registry(lambda reg: reg.__setitem__("rm", dict(entry)))
        assert entry_is_reranker("rm", entry) is False
        stored = load_registry()["rm"]
        assert stored["model_type"] == "unknown"
        assert "reranker" not in stored

    @staticmethod
    def _qwen3_seq_cls_reranker(folder):
        _config(folder, architectures=["Qwen3ForSequenceClassification"],
                id2label={"0": "LABEL_0"}, label2id={"LABEL_0": 0},
                classifier_from_token=["no", "yes"], is_original_qwen3_reranker=False)
        (folder / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
        return folder

    @staticmethod
    def _reward_model(folder):
        _config(folder, architectures=["Qwen3ForSequenceClassification"],
                id2label={"0": "LABEL_0"})
        (folder / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
        return folder

    @staticmethod
    def _cross_encoder(folder):
        _config(folder, architectures=["XLMRobertaForSequenceClassification"],
                id2label={"0": "LABEL_0"})
        (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return folder

    def test_a_qwen3_seq_cls_reranker_with_a_chat_template_is_a_reranker(self, tmp_path):
        folder = self._qwen3_seq_cls_reranker(tmp_path / "q")
        assert hf_reranker_state(folder) is True
        assert _detect_local_model_type(folder, is_gguf=False, is_hf=True)[0] == "embedding"

    def test_a_reward_model_without_the_reranker_token_mapping_stays_one(self, tmp_path):
        folder = self._reward_model(tmp_path / "r")
        assert hf_reranker_state(folder) is False

    def test_a_stored_qwen3_seq_cls_reranker_stays_a_reranker_and_unchanged(
            self, tmp_path, isolated_home):
        import localm.model_manager as mm
        folder = self._qwen3_seq_cls_reranker(tmp_path / "q")
        entry = {"model_type": "embedding", "reranker": True, "path": str(folder)}
        mm.update_registry(lambda reg: reg.__setitem__("q", dict(entry)))
        assert entry_is_reranker("q", entry) is True
        assert load_registry()["q"] == entry

    @pytest.mark.parametrize("build", ["_qwen3_seq_cls_reranker", "_reward_model", "_cross_encoder"])
    def test_reclassifying_a_stored_reranker_equals_a_fresh_registration(
            self, build, tmp_path, isolated_home):
        import localm.model_manager as mm
        fresh_dir = getattr(self, build)(tmp_path / "fresh")
        stale_dir = getattr(self, build)(tmp_path / "stale")
        assert add_local(str(fresh_dir), name="fresh", on_duplicate="skip", no_hash=True)
        fresh = load_registry()["fresh"]
        stale = {"model_type": "embedding", "reranker": True, "path": str(stale_dir)}
        mm.update_registry(lambda reg: reg.__setitem__("stale", dict(stale)))
        assert entry_is_reranker("stale", stale) is fresh.get("reranker", False)
        reclassified = load_registry()["stale"]
        for key in ("model_type", "reranker"):
            assert reclassified.get(key) == fresh.get(key), key

    def test_a_reclassified_reward_model_is_no_longer_an_embedding_model(
            self, tmp_path, isolated_home):
        import localm.model_manager as mm
        folder = self._reward_model(tmp_path / "rm")
        entry = {"model_type": "embedding", "reranker": True, "path": str(folder)}
        mm.update_registry(lambda reg: reg.__setitem__("rm", dict(entry)))
        assert entry_is_reranker("rm", entry) is False
        assert load_registry()["rm"]["model_type"] != "embedding"
        assert entry_is_reranker("rm", load_registry()["rm"]) is False

    def test_a_stored_gguf_reranker_flag_is_read_without_touching_the_file(
            self, tmp_path, monkeypatch):
        import localm.model_manager.registry as registry
        monkeypatch.setattr(registry, "hf_reranker_state",
                            lambda folder: pytest.fail("the GGUF was classified again"))
        entry = {"model_type": "embedding", "reranker": True,
                 "path": str(tmp_path / "missing.gguf")}
        assert entry_is_reranker("r", entry) is True

    @pytest.mark.parametrize("tok_cfg", [
        {}, {"chat_template": None}, {"chat_template": ""}, "not json"])
    def test_a_tokenizer_config_without_a_template_keeps_the_reranker(self, tmp_path, tok_cfg):
        folder = _config(tmp_path / "m", architectures=["XLMRobertaForSequenceClassification"],
                         id2label={"0": "LABEL_0"})
        (folder / "tokenizer_config.json").write_text(
            tok_cfg if isinstance(tok_cfg, str) else json.dumps(tok_cfg), encoding="utf-8")
        assert hf_reranker_state(folder) is True


class TestHeadCheck:
    @staticmethod
    def _safetensors(path, names):
        import struct
        header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4 * max(1, len(names)))
        return path

    def test_classifier_and_score_tensors_count_as_a_head(self, tmp_path):
        a = self._safetensors(tmp_path / "a.safetensors", ["bert.x", "classifier.weight"])
        b = self._safetensors(tmp_path / "b.safetensors", ["model.x", "score.weight"])
        assert classifier_head_present([a]) is True
        assert classifier_head_present([b]) is True

    def test_a_listing_without_one_is_reported_missing(self, tmp_path):
        a = self._safetensors(tmp_path / "a.safetensors", ["bert.x"])
        assert classifier_head_present([a]) is False

    def test_the_head_may_sit_in_any_shard(self, tmp_path):
        a = self._safetensors(tmp_path / "a.safetensors", ["bert.x"])
        b = self._safetensors(tmp_path / "b.safetensors", ["classifier.bias"])
        assert classifier_head_present([a, b]) is True

    @pytest.mark.parametrize("name", ["classification_head.out_proj.weight", "logits_proj.weight"])
    def test_other_families_name_their_head_differently(self, tmp_path, name):
        a = self._safetensors(tmp_path / "a.safetensors", ["model.x", name])
        assert classifier_head_present([a]) is True

    def test_no_files_or_an_unreadable_one_is_unknown(self, tmp_path):
        assert classifier_head_present([]) is None
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"xx")
        assert classifier_head_present([bad]) is None


def _build_cross_encoder(folder, *, num_labels=1):
    """A tiny random BertForSequenceClassification with a word-level tokenizer."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=32, num_labels=num_labels)
    BertForSequenceClassification(cfg).save_pretrained(folder)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({f"w{i}": 4 + i for i in range(60)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", model_max_length=32).save_pretrained(folder)


def _reference_scores(folder, query, documents):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).eval()
    out = []
    with torch.no_grad():
        for doc in documents:
            enc = tok(query, doc, return_tensors="pt")
            out.append(model(**enc).logits[0].tolist())
    return out


@pytest.fixture(scope="module")
def cross_encoder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rr") / "ce"
    _build_cross_encoder(folder)
    return folder


_QUERY = "w1 w2"
_DOCS = ["w1 w2 w3", "w9 w8 w7 w6", "w1"]


class TestResidentModelKey:
    def test_a_folder_model_has_a_new_key_when_a_file_in_it_changes(self, tmp_path):
        folder = tmp_path / "m"
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"a" * 10)
        (folder / "config.json").write_text("{}")
        before = reranker._file_key(str(folder))
        (folder / "model.safetensors").write_bytes(b"b" * 25)
        assert reranker._file_key(str(folder)) != before


@needs_stack
class TestRealCrossEncoder:
    def test_registered_as_a_reranker_and_resolved_by_name(self, cross_encoder, isolated_home):
        assert add_local(str(cross_encoder), name="ce") is True
        entry = load_registry()["ce"]
        assert entry["model_type"] == "embedding" and entry["reranker"] is True
        assert entry_is_reranker("ce", entry) is True
        assert reranker.registered_rerankers() == ["ce"]
        name, path = reranker.resolve_reranker(None)
        assert name == "ce" and path == str(cross_encoder.resolve())

    def test_scores_match_a_forward_pass_computed_here(self, cross_encoder):
        outcome = reranker.rerank(str(cross_encoder), _QUERY, _DOCS)
        expected = _reference_scores(cross_encoder, _QUERY, _DOCS)
        assert len(outcome.scored) == len(_DOCS)
        for got, want in zip(outcome.scored, expected, strict=True):
            assert got["scores"] == pytest.approx(want, abs=1e-5)
            assert got["truncated"] is False
        assert outcome.labels == []

    def test_results_rank_best_first_by_the_raw_score(self, cross_encoder):
        outcome = reranker.rerank(str(cross_encoder), _QUERY, _DOCS)
        rows = reranker.rank_results(outcome.scored)
        raw = [item["scores"][0] for item in outcome.scored]
        assert [r["index"] for r in rows] == sorted(range(3), key=lambda i: (-raw[i], i))

    def test_a_document_beyond_the_window_is_cut_and_flagged(self, cross_encoder):
        long = " ".join(f"w{i}" for i in range(1, 50))
        outcome = reranker.rerank(str(cross_encoder), _QUERY, [long, "w3"])
        assert outcome.scored[0]["truncated"] is True
        assert outcome.scored[0]["tokens"] <= 32
        assert outcome.scored[1]["truncated"] is False

    def test_a_query_that_fills_the_window_is_refused_and_the_worker_survives(
            self, cross_encoder):
        query = " ".join(f"w{i}" for i in range(1, 40))
        with pytest.raises(RerankInputError):
            reranker.rerank(str(cross_encoder), query, ["w3"])
        again = reranker.rerank(str(cross_encoder), _QUERY, ["w3"])
        assert len(again.scored) == 1

    def test_a_second_call_reuses_the_resident_worker(self, cross_encoder):
        reranker.rerank(str(cross_encoder), _QUERY, ["w3"])
        resident, pid = reranker._RERANKER, reranker._RERANKER._runner._proc.pid
        reranker.rerank(str(cross_encoder), _QUERY, ["w4"])
        assert reranker._RERANKER is resident
        assert resident._runner._proc.pid == pid
        assert reranker.reranker_info()["path"] == str(cross_encoder)

    def test_a_dead_worker_is_started_again_on_the_next_call(self, cross_encoder):
        before = reranker.rerank(str(cross_encoder), _QUERY, _DOCS)
        resident = reranker._RERANKER
        old = resident._runner._proc
        old.kill()
        old.join(10)
        after = reranker.rerank(str(cross_encoder), _QUERY, _DOCS)
        assert reranker._RERANKER is resident
        assert resident._runner._proc.pid != old.pid
        assert after.scored == before.scored

    def test_a_dead_worker_waits_for_a_model_load_in_progress_before_restarting(
            self, cross_encoder):
        import threading

        from localm.inference.engine import _LOAD_LOCK
        reranker.rerank(str(cross_encoder), _QUERY, ["w3"])
        resident = reranker._RERANKER
        old = resident._runner._proc
        old.kill()
        old.join(10)
        held, release = threading.Event(), threading.Event()

        def _model_load():
            with _LOAD_LOCK:
                held.set()
                release.wait(60)

        loader = threading.Thread(target=_model_load, daemon=True)
        loader.start()
        assert held.wait(10)
        result: list = []
        call = threading.Thread(
            target=lambda: result.append(reranker.rerank(str(cross_encoder), _QUERY, ["w3"])),
            daemon=True)
        try:
            call.start()
            call.join(3)
            assert call.is_alive(), "the reranker restarted while a model load held the lock"
            assert resident._runner is None or resident._runner._proc is old
        finally:
            release.set()
            loader.join(10)
        call.join(120)
        assert not call.is_alive() and len(result) == 1
        assert resident._runner._proc.pid != old.pid

    def test_more_than_one_chunk_of_documents_keeps_the_request_order(self, cross_encoder):
        docs = [f"w{i} w{i + 1}" for i in range(1, 41)]
        outcome = reranker.rerank(str(cross_encoder), _QUERY, docs)
        expected = _reference_scores(cross_encoder, _QUERY, docs)
        got = [item["scores"][0] for item in outcome.scored]
        assert got == pytest.approx([e[0] for e in expected], abs=1e-5)

    def test_a_checkpoint_without_its_head_is_refused_by_name(self, cross_encoder, tmp_path):
        import shutil
        from safetensors.torch import load_file, save_file
        folder = tmp_path / "headless"
        shutil.copytree(cross_encoder, folder)
        tensors = {k: v.contiguous() for k, v in load_file(str(folder / "model.safetensors")).items()
                   if not k.startswith("classifier.")}
        save_file(tensors, str(folder / "model.safetensors"))
        with pytest.raises(RerankerHeadMissingError, match="classification head"):
            reranker.rerank(str(folder), _QUERY, ["w3"])


class _FailingBackend:
    loaded = False
    _runner = None

    def __init__(self):
        self.loads = 0
        self.lock_held_at_load: list = []

    def unload(self):
        pass

    def load(self):
        from localm.inference.engine import _LOAD_LOCK
        self.loads += 1
        self.lock_held_at_load.append(_LOAD_LOCK._is_owned())
        raise RuntimeError("worker exited during load")

    def rerank(self, pairs):
        raise AssertionError("scored on a backend that is not loaded")


class TestFailedRestart:
    def _resident(self, tmp_path):
        folder = tmp_path / "ce"
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        hf = object.__new__(reranker.HFReranker)
        hf.model_path = str(folder)
        hf.active_requests = 0
        hf._call_lock = __import__("threading").Lock()
        hf._backend = _FailingBackend()
        hf.cls_labels = []
        hf._runner = None
        reranker._RERANKER = hf
        reranker._RERANKER_KEY = reranker._file_key(str(folder))
        return hf

    def test_the_restart_holds_the_engine_load_lock(self, tmp_path):
        hf = self._resident(tmp_path)
        with pytest.raises(reranker.RerankerUnavailableError):
            hf.rerank([("q", "d")])
        assert hf._backend.lock_held_at_load == [True]

    def test_a_released_reranker_is_not_loaded_again_by_a_waiting_request(self, tmp_path):
        hf = self._resident(tmp_path)
        reranker._RERANKER = None
        reranker._RERANKER_KEY = None
        with pytest.raises(reranker.RerankerUnavailableError) as ei:
            hf.rerank([("q", "d")])
        assert "was released" in str(ei.value)
        assert hf._backend.loads == 0

    def test_a_failed_restart_releases_the_reranker_and_latches_its_folder(
            self, tmp_path, monkeypatch):
        hf = self._resident(tmp_path)
        with pytest.raises(reranker.RerankerUnavailableError) as ei:
            hf.rerank([("q", "d")])
        assert "worker exited during load" in str(ei.value)
        assert reranker._RERANKER is None and reranker._RERANKER_KEY is None
        monkeypatch.setattr(reranker, "_check_hf_head", lambda path: None)
        monkeypatch.setattr(reranker, "HFReranker", lambda *a, **k: pytest.fail(
            "a latched folder was loaded again"))
        with pytest.raises(reranker.RerankerUnavailableError) as again:
            reranker.get_reranker(hf.model_path)
        assert "worker exited during load" in str(again.value)
        assert hf._backend.loads == 1


def _build_reward_model(folder):
    """A tiny random LlamaForSequenceClassification (one output) whose tokenizer
    carries a chat template, as a reward model ships."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64, num_labels=1, pad_token_id=0)
    LlamaForSequenceClassification(cfg).save_pretrained(folder)
    tok = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    fast.save_pretrained(folder)


@needs_stack
class TestRewardModelBesideAReranker:
    def test_adding_a_reward_model_keeps_automatic_reranking(
            self, cross_encoder, isolated_home, tmp_path):
        from localm.config import update_config
        from localm.rag.rerank import rerank_plan
        reward = tmp_path / "rm"
        _build_reward_model(reward)
        assert add_local(str(cross_encoder), name="ce") is True
        assert add_local(str(reward), name="rm") is True
        update_config(lambda c: c.update({"rag_rerank": True, "rag_rerank_model": ""}))
        plan = rerank_plan()
        assert plan.note is None, plan.note
        assert plan.model == "ce" and plan.fn is not None
        assert reranker.registered_rerankers() == ["ce"]
        assert load_registry()["rm"]["model_type"] != "embedding"


@needs_stack
class TestMultiLabelHead:
    def test_the_label_names_come_back_with_every_output(self, tmp_path):
        folder = tmp_path / "ce3"
        _build_cross_encoder(folder, num_labels=3)
        outcome = reranker.rerank(str(folder), _QUERY, ["w3"])
        assert len(outcome.scored[0]["scores"]) == 3
        assert outcome.labels == ["LABEL_0", "LABEL_1", "LABEL_2"]


@needs_stack
def test_a_backend_not_built_for_reranking_refuses():
    from localm.inference.backends.hf import HFBackend
    with pytest.raises(RuntimeError, match="not loaded for reranking"):
        HFBackend("nowhere").rerank([("q", "d")])


_QWEN3_CARD_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n")
_QWEN3_CARD_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _qwen3_card_text(query, document):
    return (f"{_QWEN3_CARD_PREFIX}<Instruct>: Given a web search query, retrieve relevant "
            f"passages that answer the query\n<Query>: {query}\n<Document>: {document}"
            f"{_QWEN3_CARD_SUFFIX}")


def _build_qwen3_seq_cls(folder):
    """A tiny random Qwen3ForSequenceClassification with a byte-level tokenizer that
    knows the chat markers as single tokens."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForSequenceClassification

    torch.manual_seed(0)
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab, [], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tok.add_special_tokens(specials)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<|endoftext|>", model_max_length=1024)
    cfg = Qwen3Config(
        vocab_size=len(vocab) + len(specials), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=1024, num_labels=1,
        pad_token_id=fast.pad_token_id)
    Qwen3ForSequenceClassification(cfg).save_pretrained(folder)
    fast.save_pretrained(folder)


def _qwen3_reference(folder, query, document):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).eval()
    ids = tok(_qwen3_card_text(query, document), add_special_tokens=False,
              return_tensors="pt")
    with torch.no_grad():
        return model(**ids).logits[0].tolist(), tok


@pytest.fixture(scope="module")
def qwen3_reranker(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rr") / "qwen3"
    _build_qwen3_seq_cls(folder)
    return folder


@needs_stack
class TestQwen3SequenceClassificationPrompt:
    def test_the_pair_is_scored_as_the_instruct_prompt_the_model_was_trained_on(
            self, qwen3_reranker):
        query, document = "what is a cat", "a cat is a small pet animal"
        outcome = reranker.rerank(str(qwen3_reranker), query, [document])
        want, tok = _qwen3_reference(qwen3_reranker, query, document)
        assert outcome.scored[0]["scores"] == pytest.approx(want, abs=1e-5)
        assert outcome.scored[0]["tokens"] == len(
            tok(_qwen3_card_text(query, document), add_special_tokens=False)["input_ids"])
        assert outcome.scored[0]["truncated"] is False

    def test_that_score_differs_from_a_plain_sentence_pair_encoding(self, qwen3_reranker):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        query, document = "what is a cat", "a cat is a small pet animal"
        tok = AutoTokenizer.from_pretrained(qwen3_reranker)
        model = AutoModelForSequenceClassification.from_pretrained(qwen3_reranker).eval()
        with torch.no_grad():
            plain = model(**tok(query, document, return_tensors="pt")).logits[0].tolist()
        got = reranker.rerank(str(qwen3_reranker), query, [document]).scored[0]["scores"]
        assert got != pytest.approx(plain, abs=1e-4)

    def test_a_long_document_is_cut_inside_the_window_and_flagged(self, qwen3_reranker):
        long = " ".join(f"word{i}" for i in range(1, 600))
        outcome = reranker.rerank(str(qwen3_reranker), "what is a cat", [long, "a cat"])
        assert outcome.scored[0]["truncated"] is True
        assert outcome.scored[0]["tokens"] <= 1024
        assert outcome.scored[1]["truncated"] is False

    def test_a_query_that_fills_the_window_is_refused_and_the_worker_survives(
            self, qwen3_reranker):
        query = "x" * 1100
        with pytest.raises(RerankInputError):
            reranker.rerank(str(qwen3_reranker), query, ["a cat"])
        assert len(reranker.rerank(str(qwen3_reranker), "cat", ["a cat"]).scored) == 1
