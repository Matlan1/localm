# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recognising a reranker from the GGUF's own header, and recording it on the
registry entry.

A reranker is an ``embedding``-type model with a classification head. Real
conversions mark it three different ways: ``<arch>.pooling_type`` = rank
(Qwen3-Reranker), ``<arch>.classifier.output_labels``, or - bge-reranker-v2-m3 and
jina-reranker-v1-tiny-en as published - neither key, only the ``cls.*`` tensors.
"""

import struct

import pytest

from localm import model_manager as mm
from localm.model_manager import gguf as G
from localm.model_manager import registry as R
from tests._gguf_builder import (ARRAY, BOOL, INT32, STRING, UINT32, build_gguf,
                                 embedding_bytes, reranker_bytes)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")
    return home


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestMetadataProbe:
    def test_rank_pooling_value_is_read(self, tmp_path):
        f = write(tmp_path, "q.gguf", build_gguf("qwen3", [("qwen3.pooling_type", UINT32, 4)]))
        meta = G._gguf_metadata_probe(f)
        assert meta["pooling_type"] == 4 and meta["has_pooling_type"] is True

    def test_a_signed_pooling_value_is_read_too(self, tmp_path):
        f = write(tmp_path, "q.gguf", build_gguf("qwen3", [("qwen3.pooling_type", INT32, 1)]))
        assert G._gguf_metadata_probe(f)["pooling_type"] == 1

    def test_a_pooling_key_that_is_not_a_number_still_counts_as_declared(self, tmp_path):
        f = write(tmp_path, "q.gguf", build_gguf("qwen3", [("qwen3.pooling_type", STRING, "1")]))
        meta = G._gguf_metadata_probe(f)
        assert meta["has_pooling_type"] is True and meta["pooling_type"] is None

    def test_only_the_architectures_own_pooling_key_gives_the_value(self, tmp_path):
        f = write(tmp_path, "q.gguf",
                  build_gguf("qwen3", [("qwen3.classifier.pooling_type", UINT32, 4)]))
        meta = G._gguf_metadata_probe(f)
        assert meta["has_pooling_type"] is True and meta["pooling_type"] is None

    def test_classifier_labels_are_noticed(self, tmp_path):
        f = write(tmp_path, "c.gguf",
                  build_gguf("bert", [("bert.classifier.output_labels", ARRAY, ["a", "b"])]))
        assert G._gguf_metadata_probe(f)["has_classifier_labels"] is True

    def test_a_file_without_either_key_reports_neither(self, tmp_path):
        f = write(tmp_path, "e.gguf", embedding_bytes())
        meta = G._gguf_metadata_probe(f)
        assert meta["pooling_type"] is None and meta["has_classifier_labels"] is False

    def test_a_pooling_key_after_the_tokenizer_keys_is_found_for_a_decoder(self, tmp_path):
        kvs = [("tokenizer.ggml.model", STRING, "gpt2"), ("qwen3.pooling_type", UINT32, 4)]
        f = write(tmp_path, "q.gguf", build_gguf("qwen3", kvs))
        assert G._gguf_metadata_probe(f)["pooling_type"] == 4

    def test_the_walk_still_stops_at_the_tokenizer_for_an_embedding_architecture(self, tmp_path):
        kvs = [("tokenizer.ggml.model", STRING, "bert"), ("bert.attention.causal", BOOL, False)]
        f = write(tmp_path, "b.gguf", build_gguf("bert", kvs))
        assert G._gguf_metadata_probe(f)["non_causal"] is False


class TestRerankerSignal:
    def test_declared_rank_pooling(self, tmp_path):
        f = write(tmp_path, "q.gguf", reranker_bytes("qwen3", rank_key=True, tensors=()))
        assert G.gguf_reranker_signal(f) is True

    def test_classifier_labels_alone(self, tmp_path):
        f = write(tmp_path, "c.gguf", reranker_bytes("bert", labels=["x"], tensors=()))
        assert G.gguf_reranker_signal(f) is True

    def test_head_tensors_alone_as_published_by_community_conversions(self, tmp_path):
        f = write(tmp_path, "bge.gguf", reranker_bytes("bert"))
        assert G.gguf_reranker_signal(f) is True

    def test_a_single_projection_head_is_enough(self, tmp_path):
        f = write(tmp_path, "jina.gguf", reranker_bytes("jina-bert-v2", tensors=("cls.weight",)))
        assert G.gguf_reranker_signal(f) is True

    def test_only_the_output_projection_is_enough(self, tmp_path):
        f = write(tmp_path, "q.gguf", reranker_bytes("bert", tensors=("cls.output.weight",)))
        assert G.gguf_reranker_signal(f) is True

    def test_a_plain_embedding_model_is_not_a_reranker_yet_still_an_embedding_model(self, tmp_path):
        f = write(tmp_path, "e.gguf", embedding_bytes("bert"))
        assert G.gguf_reranker_signal(f) is False
        assert G.gguf_embedding_signal(f) is True

    def test_a_decoder_embedding_model_is_not_a_reranker(self, tmp_path):
        f = write(tmp_path, "e.gguf", embedding_bytes("qwen3", pooling=3))
        assert G.gguf_reranker_signal(f) is False

    def test_a_chat_model_is_never_a_reranker(self, tmp_path):
        f = write(tmp_path, "chat.gguf", build_gguf("llama", tensors=("cls.weight",)))
        assert G.gguf_reranker_signal(f) is False

    @pytest.mark.parametrize("arch", sorted(G.GGUF_DIFFUSION_ARCHITECTURES))
    def test_a_diffusion_language_model_is_never_a_reranker(self, tmp_path, arch):
        f = write(tmp_path, "d.gguf", build_gguf(arch, tensors=("cls.weight",)))
        assert G.gguf_reranker_signal(f) is False

    def test_a_file_that_is_not_a_gguf_is_not_a_reranker(self, tmp_path):
        f = write(tmp_path, "x.bin", b"not a gguf at all" * 100)
        assert G.gguf_reranker_signal(f) is False

    def test_a_missing_file_is_not_a_reranker(self, tmp_path):
        assert G.gguf_reranker_signal(tmp_path / "nope.gguf") is False

    def test_a_classifier_labels_key_makes_the_file_an_embedding_type(self, tmp_path):
        f = write(tmp_path, "c.gguf", build_gguf("llama", [("llama.classifier.output_labels", ARRAY, ["a"])]))
        assert G.gguf_embedding_signal(f) is True


class TestClassifierHeadTensors:
    def test_both_head_tensors_are_listed(self, tmp_path):
        f = write(tmp_path, "r.gguf", reranker_bytes("bert"))
        assert G.gguf_classifier_head_tensors(f) == frozenset({"cls.weight", "cls.output.weight"})

    def test_bias_and_norm_tensors_alone_are_not_a_head(self, tmp_path):
        f = write(tmp_path, "r.gguf", reranker_bytes("bert", tensors=("cls.bias", "cls.norm.weight")))
        assert G.gguf_classifier_head_tensors(f) == frozenset()

    def test_a_complete_tensor_list_without_a_head_is_an_empty_set_not_unknown(self, tmp_path):
        f = write(tmp_path, "e.gguf", embedding_bytes("bert"))
        assert G.gguf_classifier_head_tensors(f) == frozenset()

    def test_an_unreadable_header_is_unknown(self, tmp_path):
        f = write(tmp_path, "x.gguf", b"GGUF" + struct.pack("<I", 3) + b"\x00" * 8)
        assert G.gguf_classifier_head_tensors(f) is None

    def test_one_part_of_a_split_file_is_unknown_because_the_head_may_be_elsewhere(self, tmp_path):
        f = write(tmp_path, "model-00001-of-00002.gguf", embedding_bytes("bert"))
        assert G.gguf_classifier_head_tensors(f) is None


class TestRegistryFlag:
    def test_a_registered_reranker_carries_the_flag(self, isolated_home, tmp_path):
        f = write(tmp_path, "bge.gguf", reranker_bytes("bert"))
        R._register("bge", f, model_type="embedding")
        entry = mm.load_registry()["bge"]
        assert entry["model_type"] == "embedding" and entry["reranker"] is True

    def test_a_registered_embedding_model_is_recorded_as_not_a_reranker(self, isolated_home, tmp_path):
        f = write(tmp_path, "e.gguf", embedding_bytes("bert"))
        R._register("emb", f, model_type="embedding")
        assert mm.load_registry()["emb"]["reranker"] is False

    def test_a_chat_model_entry_has_no_flag(self, isolated_home, tmp_path):
        f = write(tmp_path, "chat.gguf", build_gguf("llama"))
        R._register("chat", f, model_type="llm")
        assert "reranker" not in mm.load_registry()["chat"]

    def test_a_reranker_with_no_pooling_key_is_typed_embedding_and_recognised(self, tmp_path):
        f = write(tmp_path, "bge-reranker.gguf", reranker_bytes("bert"))
        mtype, _meta = R._detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype == "embedding"
        assert G.gguf_reranker_signal(f) is True

    def test_the_stored_flag_is_trusted_without_reading_the_file(self, isolated_home, tmp_path):
        entry = {"model_type": "embedding", "path": str(tmp_path / "gone.gguf"), "reranker": True}
        assert R.entry_is_reranker("x", entry) is True
        entry["reranker"] = False
        assert R.entry_is_reranker("x", entry) is False

    def test_an_entry_without_the_flag_is_classified_once_and_the_answer_is_stored(self, isolated_home, tmp_path):
        f = write(tmp_path, "old.gguf", reranker_bytes("bert"))
        reg = mm.load_registry()
        reg["old"] = {"path": str(f.resolve()), "source": "local", "model_type": "embedding"}
        mm.update_registry(lambda r: r.update(reg))
        assert R.entry_is_reranker("old", mm.load_registry()["old"]) is True
        assert mm.load_registry()["old"]["reranker"] is True

    def test_a_missing_file_answers_false_and_stores_nothing(self, isolated_home, tmp_path):
        entry = {"path": str(tmp_path / "gone.gguf"), "source": "local", "model_type": "embedding"}
        mm.update_registry(lambda r: r.update({"gone": entry}))
        assert R.entry_is_reranker("gone", mm.load_registry()["gone"]) is False
        assert "reranker" not in mm.load_registry()["gone"]

    def test_only_embedding_type_entries_can_be_rerankers(self, isolated_home, tmp_path):
        f = write(tmp_path, "r.gguf", reranker_bytes("bert"))
        assert R.entry_is_reranker("r", {"path": str(f), "model_type": "llm"}) is False


class TestUndecidableFiles:
    def test_a_file_whose_tensor_list_cannot_be_read_is_unknown_not_not_a_reranker(self, tmp_path):
        f = write(tmp_path, "model-00001-of-00002.gguf", reranker_bytes("bert"))
        assert G.gguf_reranker_state(f) is None
        assert G.gguf_reranker_signal(f) is False

    def test_the_state_is_decided_when_a_key_alone_settles_it(self, tmp_path):
        f = write(tmp_path, "model-00001-of-00002.gguf", reranker_bytes("qwen3", rank_key=True, tensors=()))
        assert G.gguf_reranker_state(f) is True

    def test_the_state_is_false_for_a_chat_model_without_reading_tensors(self, tmp_path):
        f = write(tmp_path, "model-00001-of-00002.gguf", build_gguf("llama"))
        assert G.gguf_reranker_state(f) is False

    def test_an_undecidable_file_is_registered_without_a_flag_and_asked_again_later(self, isolated_home, tmp_path):
        f = write(tmp_path, "model-00001-of-00002.gguf", reranker_bytes("bert"))
        R._register("split", f, model_type="embedding")
        assert "reranker" not in mm.load_registry()["split"]
        assert R.entry_is_reranker("split", mm.load_registry()["split"]) is False
        assert "reranker" not in mm.load_registry()["split"]
