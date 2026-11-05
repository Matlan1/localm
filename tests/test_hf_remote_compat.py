# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility shim for Ouro's bundled code on the pinned transformers.

The bundled modeling code reads ``config.pad_token_id``, looks up
``ROPE_INIT_FUNCTIONS["default"]`` and calls ``create_causal_mask(input_embeds=...,
cache_position=...)``; transformers 5.x provides none of the three. The shim runs
only for ``model_type`` ``ouro`` while ``hf_trust_remote_code`` is on, and adds
only what is missing.
"""

import inspect
import json
import logging
import sys
import types
from types import SimpleNamespace

import pytest

from localm.inference.backends import _hf_remote_compat as compat
from localm.inference.backends._hf_worker import HFWorker


def _model_dir(tmp_path, model_type="ouro"):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}),
                                          encoding="utf-8")
    return str(tmp_path)


class TestForModel:
    def test_an_ouro_directory_with_trust_on_gets_the_compat(self, tmp_path):
        assert isinstance(compat.for_model(_model_dir(tmp_path), True),
                          compat.RemoteCodeCompat)

    def test_trust_off_gets_nothing(self, tmp_path):
        assert compat.for_model(_model_dir(tmp_path), False) is None

    @pytest.mark.parametrize("model_type", ["llama", "qwen3", ""])
    def test_another_model_type_gets_nothing(self, tmp_path, model_type):
        assert compat.for_model(_model_dir(tmp_path, model_type), True) is None

    def test_a_missing_or_broken_config_gets_nothing(self, tmp_path):
        assert compat.for_model(str(tmp_path), True) is None
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
        assert compat.for_model(str(tmp_path), True) is None
        (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
        assert compat.for_model(str(tmp_path), True) is None


class TestConfig:
    def _tr(self, config=None, error=None):
        def from_pretrained(path, trust_remote_code):
            assert trust_remote_code is True
            if error:
                raise error
            return config
        return SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=from_pretrained))

    def test_a_missing_pad_token_id_becomes_none(self):
        config = SimpleNamespace(bos_token_id=0)
        out = compat.RemoteCodeCompat("m").config(self._tr(config))
        assert out is config and out.pad_token_id is None

    def test_a_stated_pad_token_id_is_kept(self):
        config = SimpleNamespace(pad_token_id=7)
        assert compat.RemoteCodeCompat("m").config(self._tr(config)).pad_token_id == 7

    def test_an_unreadable_config_is_reported_and_left_to_the_load(self, caplog):
        tr = self._tr(error=OSError("no config"))
        with caplog.at_level(logging.WARNING, logger=compat.logger.name):
            assert compat.RemoteCodeCompat("m").config(tr) is None
        assert "could not read the config" in caplog.text and "no config" in caplog.text


class TestRopeDefault:
    @pytest.fixture
    def table(self, monkeypatch):
        from transformers import modeling_rope_utils
        isolated = {"linear": object()}
        monkeypatch.setattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", isolated)
        return isolated

    def test_the_default_function_exists_only_while_the_model_is_built(self, table):
        with compat.RemoteCodeCompat("m").loading():
            assert table["default"] is compat.default_rope_parameters
        assert "default" not in table and "linear" in table

    def test_a_failed_load_still_removes_it(self, table):
        with pytest.raises(RuntimeError):
            with compat.RemoteCodeCompat("m").loading():
                assert "default" in table
                raise RuntimeError("load failed")
        assert "default" not in table

    def test_an_existing_default_is_neither_replaced_nor_removed(self, table):
        sentinel = object()
        table["default"] = sentinel
        with compat.RemoteCodeCompat("m").loading():
            assert table["default"] is sentinel
        assert table["default"] is sentinel

    def test_the_real_table_is_unchanged_afterwards(self):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        before = dict(ROPE_INIT_FUNCTIONS)
        with compat.RemoteCodeCompat("m").loading():
            pass
        assert dict(ROPE_INIT_FUNCTIONS) == before

    def test_the_default_function_is_unscaled_rotary_embedding(self):
        torch = pytest.importorskip("torch")
        config = SimpleNamespace(rope_theta=10000.0, head_dim=8)
        inv_freq, factor = compat.default_rope_parameters(config)
        assert factor == 1.0
        expected = [1.0 / (10000.0 ** (i / 8)) for i in range(0, 8, 2)]
        assert torch.allclose(inv_freq, torch.tensor(expected, dtype=torch.float))

    def test_the_head_width_falls_back_to_hidden_size_over_heads(self):
        pytest.importorskip("torch")
        config = SimpleNamespace(rope_theta=10000.0, hidden_size=64, num_attention_heads=8,
                                 partial_rotary_factor=0.5)
        inv_freq, _ = compat.default_rope_parameters(config)
        assert inv_freq.shape[0] == 2


def _new_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
    return ("mask", config, inputs_embeds, attention_mask, past_key_values, position_ids)


class TestMaskFunction:
    def test_the_old_keywords_are_renamed_and_dropped(self):
        wrapped = compat.adapt_mask_function(_new_mask)
        out = wrapped(config="c", input_embeds="e", attention_mask="a",
                      cache_position="dropped", past_key_values="p", position_ids="i")
        assert out == ("mask", "c", "e", "a", "p", "i")

    def test_the_new_keywords_pass_through(self):
        wrapped = compat.adapt_mask_function(_new_mask)
        assert wrapped("c", inputs_embeds="e", attention_mask="a",
                       past_key_values="p") == ("mask", "c", "e", "a", "p", None)

    def test_a_function_that_takes_the_old_keyword_is_left_alone(self):
        def old(config, input_embeds, attention_mask, cache_position):
            return input_embeds
        assert compat.adapt_mask_function(old) is old

    def test_a_function_with_var_keywords_is_left_alone(self):
        def open_ended(config, **kwargs):
            return kwargs
        assert compat.adapt_mask_function(open_ended) is open_ended

    def test_the_wrapper_keeps_the_original_reachable(self):
        assert compat.adapt_mask_function(_new_mask).__wrapped__ is _new_mask

    def test_any_other_unknown_keyword_still_raises(self):
        wrapped = compat.adapt_mask_function(_new_mask)
        with pytest.raises(TypeError, match="bogus"):
            wrapped(config="c", input_embeds="e", attention_mask="a",
                    past_key_values="p", bogus=1)

    def test_a_var_keywords_function_that_names_inputs_embeds_still_gets_the_rename(self):
        def open_ended(config, inputs_embeds, **kwargs):
            return inputs_embeds, kwargs
        out = compat.adapt_mask_function(open_ended)(
            config="c", input_embeds="e", cache_position=1)
        assert out == ("e", {"cache_position": 1})

    @pytest.mark.parametrize("name", ["create_causal_mask", "create_sliding_window_causal_mask"])
    def test_the_installed_transformers_functions_need_the_adapter(self, name):
        masking_utils = pytest.importorskip("transformers.masking_utils")
        fn = getattr(masking_utils, name)
        wrapped = compat.adapt_mask_function(fn)
        assert wrapped is not fn and wrapped.__wrapped__ is fn


class TestAfterLoad:
    def _model(self, monkeypatch, **functions):
        module = types.ModuleType("fake_remote_modeling_ouro")
        for name, fn in functions.items():
            setattr(module, name, fn)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module, type("OuroForCausalLM", (), {"__module__": module.__name__})()

    def test_both_mask_functions_of_the_defining_module_are_adapted(self, monkeypatch):
        module, model = self._model(monkeypatch, create_causal_mask=_new_mask,
                                    create_sliding_window_causal_mask=_new_mask)
        compat.RemoteCodeCompat("m").after_load(model)
        for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            assert getattr(module, name) is not _new_mask
            assert getattr(module, name)(config="c", input_embeds="e", attention_mask="a",
                                         cache_position=1, past_key_values="p")[2] == "e"

    def test_a_module_with_only_one_of_them_adapts_that_one(self, monkeypatch):
        module, model = self._model(monkeypatch, create_causal_mask=_new_mask)
        compat.RemoteCodeCompat("m").after_load(model)
        assert module.create_causal_mask is not _new_mask
        assert not hasattr(module, "create_sliding_window_causal_mask")

    def test_a_module_with_none_is_reported(self, monkeypatch, caplog):
        _module, model = self._model(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=compat.logger.name):
            compat.RemoteCodeCompat("m").after_load(model)
        assert "no mask function to adapt" in caplog.text

    def test_an_unknown_module_is_reported(self, caplog):
        model = type("M", (), {"__module__": "no_such_remote_module_anywhere"})()
        with caplog.at_level(logging.WARNING, logger=compat.logger.name):
            compat.RemoteCodeCompat("m").after_load(model)
        assert "no_such_remote_module_anywhere" in caplog.text


class TestWorkerWiring:
    def test_load_applies_the_compat_around_the_model_load(self):
        src = inspect.getsource(HFWorker.load)
        make = src.index("_hf_remote_compat.for_model(self.model_path, trust_remote_code)")
        config = src.index('load_kwargs["config"] = compat_config')
        loop = src.index("for cls_name in classes:")
        scope = src.index("compat.loading() if compat is not None")
        after = src.index("compat.after_load(self._model)")
        assert make < config < loop < scope < after
        assert src.index("from_pretrained(self.model_path, **load_kwargs)") > scope
