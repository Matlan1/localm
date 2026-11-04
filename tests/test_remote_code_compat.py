# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model's bundled code written for an older transformers keeps working: the mask
functions accept the older keyword names, a model with its own cache keeps it, and a
model that insists on flash attention falls back to sdpa when flash_attn is absent."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
masking = pytest.importorskip("transformers.masking_utils")

from localm.inference.backends import _remote_code_compat as compat  # noqa: E402

LEGACY_CODE = (
    "causal_mask = create_causal_mask(config=self.config, input_embeds=inputs_embeds,\n"
    "    attention_mask=attention_mask, cache_position=cache_position,\n"
    "    past_key_values=past_key_values, position_ids=position_ids)\n")


@pytest.fixture
def mask_functions(monkeypatch):
    for name in compat._MASK_FUNCTIONS:
        monkeypatch.setattr(masking, name, getattr(masking, name))


def _folder(tmp_path, code: str):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "modeling_x.py").write_text(code, encoding="utf-8")
    return folder


def test_the_real_mask_function_accepts_the_older_keywords_for_a_model_that_uses_them(
        tmp_path, mask_functions):
    from transformers import LlamaConfig
    config = LlamaConfig(hidden_size=8, num_attention_heads=2, num_hidden_layers=1,
                         intermediate_size=8, vocab_size=16)
    embeds = torch.zeros(1, 4, 8)
    legacy = dict(config=config, input_embeds=embeds, attention_mask=None,
                  cache_position=torch.arange(4), past_key_values=None)
    with pytest.raises(TypeError):
        masking.create_causal_mask(**legacy)
    assert compat.apply_for(str(_folder(tmp_path, LEGACY_CODE))) is True
    current = masking.create_causal_mask(**legacy)
    expected = masking.create_causal_mask(config=config, inputs_embeds=embeds,
                                          attention_mask=None, past_key_values=None)
    assert (current is None and expected is None) or torch.equal(current, expected)


def test_a_model_that_never_calls_the_mask_functions_leaves_them_alone(tmp_path, mask_functions):
    before = masking.create_causal_mask
    assert compat.apply_for(str(_folder(tmp_path, "import torch\n"))) is False
    assert masking.create_causal_mask is before


def test_wrapping_twice_wraps_once(tmp_path, mask_functions):
    folder = str(_folder(tmp_path, LEGACY_CODE))
    assert compat.apply_for(folder) is True
    wrapped = masking.create_causal_mask
    assert compat.apply_for(folder) is False
    assert masking.create_causal_mask is wrapped


def _stateful_model(monkeypatch, name="Fake"):
    module = types.ModuleType(f"fake_remote_{name}")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Cache:
        def __init__(self):
            self.seen = 3

        def get_seq_length(self, layer_idx=0):
            return self.seen

        def get_mask_sizes(self, cache_position, layer_idx):
            return cache_position.shape[0] + self.seen, 0

    class Model:
        _is_stateful = True
        config = types.SimpleNamespace(_attn_implementation="sdpa")

        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    for cls in (Cache, Model):
        cls.__module__ = module.__name__
        setattr(module, cls.__name__, cls)
    return Model(), Cache


def test_a_model_with_its_own_cache_keeps_it_and_the_cache_gets_the_current_api(monkeypatch):
    model, cache_cls = _stateful_model(monkeypatch)
    assert type(model)._supports_default_dynamic_cache() is True
    assert compat.apply_to_model(model) is True
    assert type(model)._supports_default_dynamic_cache() is False
    cache = cache_cls()
    assert cache.get_mask_sizes(5, 0) == (8, 0)
    assert cache.get_mask_sizes(torch.arange(5), 0) == (8, 0)
    assert cache.get_query_offset(0) == 3
    assert compat.apply_to_model(model) is False


def test_a_model_without_its_own_cache_is_left_alone(monkeypatch):
    model, _ = _stateful_model(monkeypatch, "Plain")
    type(model)._is_stateful = False
    assert compat.apply_to_model(model) is False
    assert type(model)._supports_default_dynamic_cache() is True


@pytest.mark.parametrize("installed,expected", [(False, "sdpa"), (True, "flash_attention_2")])
def test_flash_attention_is_replaced_by_sdpa_only_when_flash_attn_is_missing(
        monkeypatch, installed, expected):
    model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="flash_attention_2"))
    monkeypatch.setattr(compat.importlib.util, "find_spec",
                        lambda name: object() if installed else None)
    assert compat.apply_to_model(model) is (not installed)
    assert model.config._attn_implementation == expected
