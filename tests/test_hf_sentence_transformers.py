# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sentence-transformers folders embed through the stages their ``modules.json`` lists.

A folder is a transformer (the folder's own config.json and weights) followed by a
Pooling module and optionally Dense and Normalize modules. ``HFWorker.embed`` runs
those stages; ``add`` registers such a folder as an embedding model. The real-model
tests build a tiny BERT and compute every expected vector by hand from its hidden
states.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest

from localm.inference.backends import _st_plan
from localm.inference.backends._st_plan import StUnsupportedError, load_plan

needs_stack = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("tokenizers") is None,
    reason="needs torch, transformers and tokenizers")

_TEXTS = ["w1 w2 w3", "w4 w5 w6 w7 w8 w9"]


def _write(folder, rel, payload):
    path = folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _modules(*stages):
    kinds = {"transformer": "sentence_transformers.base.modules.transformer.Transformer",
             "pooling": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
             "dense": "sentence_transformers.base.modules.dense.Dense",
             "normalize": "sentence_transformers.base.modules.normalize.Normalize"}
    folders = {"transformer": "", "pooling": "1_Pooling", "dense": "2_Dense",
               "normalize": "3_Normalize"}
    return [{"idx": i, "name": str(i), "path": folders[s], "type": kinds[s]}
            for i, s in enumerate(stages)]


class TestPlanParsing:
    def test_no_modules_json_is_not_a_sentence_transformers_folder(self, tmp_path):
        assert _st_plan.is_st_folder(str(tmp_path)) is False
        assert load_plan(str(tmp_path)) is None

    def test_pooling_and_normalize_stages_in_order(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling", "normalize"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "cls"})
        plan = load_plan(str(tmp_path))
        assert [s.kind for s in plan.stages] == ["pooling", "normalize"]
        assert plan.stages[0].pooling_modes == ("cls",)
        assert plan.max_seq_length == _st_plan.DEFAULT_MAX_SEQ_LENGTH

    def test_legacy_boolean_pooling_flags_concatenate_in_the_library_order(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", {
            "word_embedding_dimension": 8, "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False})
        assert load_plan(str(tmp_path)).stages[0].pooling_modes == ("cls", "mean")

    def test_a_list_of_pooling_modes_keeps_its_order(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": ["mean", "cls"]})
        assert load_plan(str(tmp_path)).stages[0].pooling_modes == ("mean", "cls")

    def test_max_seq_length_is_read(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        _write(tmp_path, "sentence_bert_config.json", {"max_seq_length": 128})
        assert load_plan(str(tmp_path)).max_seq_length == 128

    @pytest.mark.parametrize("bad", [0, -1, "x", True, None])
    def test_a_bad_max_seq_length_falls_back_to_the_default(self, tmp_path, bad):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        _write(tmp_path, "sentence_bert_config.json", {"max_seq_length": bad})
        assert load_plan(str(tmp_path)).max_seq_length == _st_plan.DEFAULT_MAX_SEQ_LENGTH

    def test_an_unknown_module_is_refused_by_name(self, tmp_path):
        mods = _modules("transformer", "pooling")
        mods.append({"idx": 2, "name": "2", "path": "2_LSTM",
                     "type": "sentence_transformers.models.LSTM"})
        _write(tmp_path, "modules.json", mods)
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        with pytest.raises(StUnsupportedError, match="LSTM"):
            load_plan(str(tmp_path))

    def test_an_unknown_pooling_mode_is_refused_by_name(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "attention"})
        with pytest.raises(StUnsupportedError, match="attention"):
            load_plan(str(tmp_path))

    @pytest.mark.parametrize("config", [
        {"pooling_mode_cls_token": False}, {"word_embedding_dimension": 8}])
    def test_no_pooling_mode_set_means_mean_pooling_like_the_library(self, tmp_path, config):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        _write(tmp_path, "1_Pooling/config.json", config)
        assert load_plan(str(tmp_path)).stages[0].pooling_modes == ("mean",)

    @pytest.mark.parametrize("dense_config, why", [
        ({"activation_function": "torch.nn.modules.activation.Tanh", "use_residual": True},
         "residual"),
        ({"activation_function": "torch.nn.modules.activation.Tanh",
          "module_input_name": "token_embeddings"}, "token_embeddings"),
        ({"in_features": 8}, "activation_function"),
    ])
    def test_a_dense_stage_localm_cannot_reproduce_is_refused(self, tmp_path, dense_config, why):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling", "dense"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        _write(tmp_path, "2_Dense/config.json", dense_config)
        (tmp_path / "2_Dense" / "model.safetensors").write_bytes(b"x")
        with pytest.raises(StUnsupportedError, match=why):
            load_plan(str(tmp_path))

    def test_a_dense_stage_with_only_a_pickle_is_refused_by_name(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling", "dense"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        _write(tmp_path, "2_Dense/config.json", {"activation_function": "torch.nn.modules.linear.Identity"})
        (tmp_path / "2_Dense" / "pytorch_model.bin").write_bytes(b"x")
        with pytest.raises(StUnsupportedError, match="pytorch_model.bin"):
            load_plan(str(tmp_path))

    def test_an_unknown_dense_activation_is_refused_by_name(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "pooling", "dense"))
        _write(tmp_path, "1_Pooling/config.json", {"pooling_mode": "mean"})
        _write(tmp_path, "2_Dense/config.json", {"activation_function": "torch.nn.modules.activation.Mish"})
        (tmp_path / "2_Dense" / "model.safetensors").write_bytes(b"x")
        with pytest.raises(StUnsupportedError, match="Mish"):
            load_plan(str(tmp_path))

    def test_a_transformer_in_a_subfolder_is_refused(self, tmp_path):
        mods = _modules("transformer", "pooling")
        mods[0]["path"] = "0_Transformer"
        _write(tmp_path, "modules.json", mods)
        with pytest.raises(StUnsupportedError, match="0_Transformer"):
            load_plan(str(tmp_path))

    def test_a_folder_without_pooling_is_refused(self, tmp_path):
        _write(tmp_path, "modules.json", _modules("transformer", "normalize"))
        with pytest.raises(StUnsupportedError, match="Pooling"):
            load_plan(str(tmp_path))


class TestRegistryType:
    def test_a_sentence_transformers_folder_registers_as_embedding(self, tmp_path):
        from localm.model_manager.registry import _detect_local_model_type
        _write(tmp_path, "config.json", {"architectures": ["BertModel"], "model_type": "bert"})
        _write(tmp_path, "modules.json", _modules("transformer", "pooling"))
        assert _detect_local_model_type(tmp_path, is_gguf=False, is_hf=True)[0] == "embedding"

    def test_the_same_folder_without_modules_json_is_not(self, tmp_path):
        from localm.model_manager.registry import _detect_local_model_type
        _write(tmp_path, "config.json", {"architectures": ["BertModel"], "model_type": "bert"})
        assert _detect_local_model_type(tmp_path, is_gguf=False, is_hf=True)[0] == "unknown"

    def test_a_modules_json_without_pooling_is_not(self, tmp_path):
        from localm.model_manager.registry import _detect_local_model_type
        _write(tmp_path, "config.json", {"architectures": ["BertModel"], "model_type": "bert"})
        _write(tmp_path, "modules.json", _modules("transformer", "normalize"))
        assert _detect_local_model_type(tmp_path, is_gguf=False, is_hf=True)[0] == "unknown"


def _build_bert(folder):
    """A tiny random BertModel with a word-level tokenizer (``w1``..``w59``)."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=64,
                         max_position_embeddings=64)).save_pretrained(folder)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({f"w{i}": 4 + i for i in range(60)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]").save_pretrained(folder)


def _hidden_states(folder):
    """Per text: the transformer's last hidden states (seq, dim), computed outside
    the code under test."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModel.from_pretrained(folder).eval()
    with torch.no_grad():
        return [model(**tok(t, return_tensors="pt")).last_hidden_state[0] for t in _TEXTS]


@pytest.fixture(scope="module")
def bert_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("st") / "bert"
    _build_bert(folder)
    return folder


def _variant(bert_folder, tmp_path, stages, pooling_config, dense=None, bert_config=None):
    import shutil
    dest = tmp_path / "model"
    shutil.copytree(bert_folder, dest)
    _write(dest, "modules.json", _modules(*stages))
    _write(dest, "1_Pooling/config.json", pooling_config)
    if bert_config is not None:
        _write(dest, "sentence_bert_config.json", bert_config)
    if "normalize" in stages:
        _write(dest, "3_Normalize/config.json", {})
    if dense is not None:
        from safetensors.torch import save_file
        _write(dest, "2_Dense/config.json", {
            "in_features": 32, "out_features": 16, "bias": True,
            "activation_function": "torch.nn.modules.activation.Tanh"})
        save_file({"linear.weight": dense[0].contiguous(), "linear.bias": dense[1].contiguous()},
                  str(dest / "2_Dense" / "model.safetensors"))
    return dest


def _embed(folder, texts=_TEXTS):
    from localm.inference.backends._hf_worker import HFWorker
    worker = HFWorker(str(folder), device="cpu")
    worker.load()
    try:
        return worker.embed(texts)
    finally:
        worker.unload()


def _close(actual, expected, tol=1e-5):
    import torch
    assert torch.allclose(torch.tensor(actual), torch.stack(expected), atol=tol), (
        torch.tensor(actual), torch.stack(expected))


@needs_stack
class TestEmbeddingThroughTheStages:
    def test_mean_pooling_then_normalize(self, bert_folder, tmp_path):
        import torch
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling", "normalize"),
                          {"pooling_mode": "mean"})
        expected = [torch.nn.functional.normalize(h.mean(dim=0), dim=0)
                    for h in _hidden_states(folder)]
        _close(_embed(folder), expected)

    def test_cls_pooling_is_the_first_token_not_the_mean(self, bert_folder, tmp_path):
        import torch
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "cls"})
        hidden = _hidden_states(folder)
        got = _embed(folder)
        _close(got, [h[0] for h in hidden])
        assert not torch.allclose(torch.tensor(got), torch.stack([h.mean(dim=0) for h in hidden]),
                                  atol=1e-3)

    def test_last_token_pooling(self, bert_folder, tmp_path):
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "lasttoken"})
        _close(_embed(folder), [h[-1] for h in _hidden_states(folder)])

    def test_max_pooling(self, bert_folder, tmp_path):
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "max"})
        _close(_embed(folder), [h.max(dim=0).values for h in _hidden_states(folder)])

    def test_mean_sqrt_len_pooling(self, bert_folder, tmp_path):
        import math
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "mean_sqrt_len_tokens"})
        _close(_embed(folder), [h.sum(dim=0) / math.sqrt(h.shape[0])
                                for h in _hidden_states(folder)])

    def test_weighted_mean_pooling(self, bert_folder, tmp_path):
        import torch
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "weightedmean"})
        expected = []
        for h in _hidden_states(folder):
            w = torch.arange(1, h.shape[0] + 1, dtype=h.dtype).unsqueeze(-1)
            expected.append((h * w).sum(dim=0) / w.sum())
        _close(_embed(folder), expected)

    def test_legacy_flags_with_two_modes_concatenate(self, bert_folder, tmp_path):
        import torch
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"word_embedding_dimension": 32, "pooling_mode_cls_token": True,
                           "pooling_mode_mean_tokens": True})
        _close(_embed(folder), [torch.cat([h[0], h.mean(dim=0)]) for h in _hidden_states(folder)])

    def test_dense_projection_with_tanh_then_normalize(self, bert_folder, tmp_path):
        import torch
        gen = torch.Generator().manual_seed(1)
        weight, bias = torch.randn(16, 32, generator=gen), torch.randn(16, generator=gen)
        folder = _variant(bert_folder, tmp_path,
                          ("transformer", "pooling", "dense", "normalize"),
                          {"pooling_mode": "cls"}, dense=(weight, bias))
        expected = [torch.nn.functional.normalize(torch.tanh(h[0] @ weight.T + bias), dim=0)
                    for h in _hidden_states(folder)]
        _close(_embed(folder), expected)

    def test_texts_are_cut_at_max_seq_length(self, bert_folder, tmp_path):
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "mean"}, bert_config={"max_seq_length": 5})
        cut, three_words, two_words = _embed(
            folder, ["w1 w2 w3 w4 w5 w6", "w1 w2 w3", "w1 w2"])
        assert cut == pytest.approx(three_words, abs=1e-6)
        assert cut != pytest.approx(two_words, abs=1e-6)

    def test_a_folder_declaring_a_generative_architecture_still_counts_as_an_embedder(
            self, bert_folder, tmp_path):
        from localm.inference.backends._hf_worker import HFWorker
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "lasttoken"})
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        config["architectures"] = ["BertLMHeadModel"]
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        worker = HFWorker(str(folder), device="cpu")
        worker.load()
        try:
            assert worker.can_embed is True
        finally:
            worker.unload()

    def test_a_roberta_style_model_loses_two_positions_from_the_window(self):
        from localm.inference.backends._hf_worker import HFWorker
        worker = HFWorker("nowhere", device="cpu")
        worker.context_capacity = 514
        worker._model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="roberta"))
        assert worker._usable_positions() == 512
        worker._model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="bert"))
        assert worker._usable_positions() == 514

    def test_the_encoder_is_loaded_without_a_language_model_head(self, bert_folder, tmp_path):
        from localm.inference.backends._hf_worker import HFWorker
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "mean"})
        worker = HFWorker(str(folder), device="cpu")
        worker.load()
        try:
            assert type(worker._model).__name__ == "BertModel"
        finally:
            worker.unload()

    def test_a_folder_the_loader_cannot_run_fails_the_load_with_the_reason(
            self, bert_folder, tmp_path):
        from localm.inference.backends._hf_worker import HFWorker
        folder = _variant(bert_folder, tmp_path, ("transformer", "pooling"),
                          {"pooling_mode": "attention"})
        with pytest.raises(StUnsupportedError, match="attention"):
            HFWorker(str(folder), device="cpu").load()


@needs_stack
class TestPlainEncoder:
    def test_an_encoder_without_modules_json_still_embeds_by_mean_pooling(
            self, bert_folder):
        """Loading a bare BertModel used to die in transformers' loading report
        (the stdout filter had no isatty); it must load and embed."""
        expected = [h.mean(dim=0) for h in _hidden_states(bert_folder)]
        _close(_embed(bert_folder), expected)


def test_the_stdout_filter_passes_through_what_a_terminal_probe_asks(monkeypatch):
    import io

    from localm.inference.backends._hf_worker import _filter_docstring_leak

    class Tty(io.StringIO):
        encoding = "latin-1"

        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    with _filter_docstring_leak():
        assert sys.stdout.isatty() is True
        assert sys.stdout.encoding == "latin-1"


def test_a_stdout_filter_without_a_target_raises_attribute_error_not_recursion():
    from localm.inference.backends._hf_worker import _DocstringLeakFilter

    bare = object.__new__(_DocstringLeakFilter)
    with pytest.raises(AttributeError):
        _ = bare.isatty
