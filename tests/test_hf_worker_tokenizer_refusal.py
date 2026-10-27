# SPDX-License-Identifier: AGPL-3.0-or-later
"""The text-only tokenizer fallback of the HF worker refuses a model it cannot
load. Only a configuration that fails validation is called an invalid
configuration; a missing, unresolvable or unreadable tokenizer is reported as a
tokenizer failure with its real reason."""

import json

import pytest

transformers = pytest.importorskip("transformers")

from localm.inference.backends import _hf_worker  # noqa: E402
from localm.inference.backends.base import ModelRefusedError  # noqa: E402

_LLAMA = {"model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
          "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
          "vocab_size": 100}


def _model_dir(tmp_path, config=None, files=None):
    d = tmp_path / "tiny-model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config or _LLAMA))
    for name, data in (files or {}).items():
        (d / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    return d


def _refusal(model_dir):
    with pytest.raises(ModelRefusedError) as raised:
        _hf_worker._load_text_tokenizer(transformers, str(model_dir), False)
    return str(raised.value)


def test_a_model_without_tokenizer_files_is_a_tokenizer_failure(tmp_path):
    msg = _refusal(_model_dir(tmp_path))
    assert "its tokenizer could not be loaded" in msg
    assert "configuration is not valid" not in msg
    assert msg.startswith("tiny-model cannot be loaded")


def test_an_unresolvable_tokenizer_class_is_a_tokenizer_failure(tmp_path):
    msg = _refusal(_model_dir(tmp_path, files={
        "tokenizer_config.json": json.dumps({"tokenizer_class": "NoSuchTokenizerXyz"})}))
    assert "its tokenizer could not be loaded" in msg
    assert "configuration is not valid" not in msg


def test_a_lone_sentencepiece_file_is_a_tokenizer_failure(tmp_path):
    model_dir = _model_dir(tmp_path, files={"tokenizer.model": b"not a real model"})
    try:
        transformers.AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        pass
    else:
        pytest.skip("a tokenizer package is installed that reads this file")
    msg = _refusal(model_dir)
    assert "its tokenizer could not be loaded" in msg
    assert "configuration is not valid" not in msg


def test_an_invalid_configuration_is_still_called_invalid(tmp_path):
    bad = dict(_LLAMA, hidden_size=65)
    msg = _refusal(_model_dir(tmp_path, config=bad))
    assert "its configuration is not valid" in msg
    assert "tokenizer could not be loaded" not in msg


def test_a_value_error_from_an_invalid_config_is_called_invalid(tmp_path, monkeypatch):
    class _Tr:
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*a, **k):
                raise ValueError("tokenizer broke")

        class AutoConfig:
            @staticmethod
            def from_pretrained(*a, **k):
                raise ValueError("hidden_size must be divisible by heads")

    with pytest.raises(ModelRefusedError) as raised:
        _hf_worker._load_text_tokenizer(_Tr, str(tmp_path / "m"), False)
    assert "its configuration is not valid" in str(raised.value)
    assert "hidden_size must be divisible" in str(raised.value)
