# SPDX-License-Identifier: AGPL-3.0-or-later
"""CTranslate2 exports and Diffusers pipelines are named when localm is pointed at them.

localm runs neither format as a model, so ``localm add`` and the HF backend answer
with one sentence naming the format and the nearest runnable alternative instead
of a bare "Not a model" or a transformers load error.
"""

from __future__ import annotations

import json

import pytest

from localm.inference.backends.base import UnsupportedModelRoleError
from localm.inference.backends.hf import _check_format_supported
from localm.model_manager.unsupported import (
    ctranslate2_refusal,
    diffusers_refusal,
    explain_unsupported_model,
    hf_folder_refusal,
)


def _ct2(folder, *, config=None, extra=()):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.bin").write_bytes(b"\0" * 64)
    (folder / "shared_vocabulary.json").write_text("[]")
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    for name in extra:
        (folder / name).write_bytes(b"x")
    return folder


class TestCTranslate2:
    def test_a_translation_export_without_a_config_is_named(self, tmp_path):
        sentence = ctranslate2_refusal(_ct2(tmp_path / "nllb"))
        assert sentence and "CTranslate2" in sentence and "translation model" in sentence

    def test_a_ct2_config_without_architectures_is_named(self, tmp_path):
        folder = _ct2(tmp_path / "marian", config={"bos_token": "<s>", "eos_token": "</s>"})
        assert ctranslate2_refusal(folder) is not None

    def test_a_faster_whisper_export_is_named_too(self, tmp_path):
        folder = _ct2(tmp_path / "whisper", config={"alignment_heads": [[1, 2]]},
                      extra=("tokenizer.json", "preprocessor_config.json"))
        assert ctranslate2_refusal(folder) is not None

    @pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin", "x.gguf"])
    def test_a_folder_with_other_weights_beside_model_bin_is_not(self, tmp_path, weights):
        folder = _ct2(tmp_path / "mixed", extra=(weights,))
        assert ctranslate2_refusal(folder) is None

    def test_a_hugging_face_config_beside_model_bin_is_not(self, tmp_path):
        folder = _ct2(tmp_path / "hf", config={"architectures": ["BertModel"]})
        assert ctranslate2_refusal(folder) is None
        folder = _ct2(tmp_path / "hf2", config={"model_type": "bert"})
        assert ctranslate2_refusal(folder) is None

    def test_a_folder_without_model_bin_is_not(self, tmp_path):
        (tmp_path / "f").mkdir()
        (tmp_path / "f" / "shared_vocabulary.json").write_text("[]")
        assert ctranslate2_refusal(tmp_path / "f") is None

    def test_the_folder_checks_and_the_explainer_use_it(self, tmp_path):
        folder = _ct2(tmp_path / "nllb")
        assert hf_folder_refusal(folder) == ctranslate2_refusal(folder)
        assert explain_unsupported_model(folder) == ctranslate2_refusal(folder)

    def test_adding_one_registers_it_with_the_sentence_and_the_backend_refuses_it(
            self, tmp_path, monkeypatch, capsys):
        import localm.config as cfg
        import localm.model_manager as mm
        from localm.model_manager import add_local
        home = tmp_path / ".localm"
        (home / "models").mkdir(parents=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
        folder = _ct2(tmp_path / "nllb", config={"bos_token": "<s>"}, extra=("tokenizer.json",))
        assert add_local(str(folder)) is True
        assert "cannot load it" in " ".join(capsys.readouterr().out.split())

    def test_the_hf_backend_refuses_to_load_it_before_a_worker_is_spawned(self, tmp_path):
        folder = _ct2(tmp_path / "nllb", config={"bos_token": "<s>"})
        with pytest.raises(UnsupportedModelRoleError, match="CTranslate2"):
            _check_format_supported(str(folder))


class TestDiffusers:
    def _pipeline(self, folder, index):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_index.json").write_text(json.dumps(index))
        return folder

    def test_a_pipeline_with_a_class_name_is_named(self, tmp_path):
        folder = self._pipeline(tmp_path / "p", {"_class_name": "StableDiffusionPipeline"})
        assert "Diffusers pipeline" in (diffusers_refusal(folder) or "")

    def test_a_pipeline_with_only_a_version_is_named(self, tmp_path):
        folder = self._pipeline(tmp_path / "p", {"_diffusers_version": "0.30.0"})
        assert diffusers_refusal(folder) is not None

    @pytest.mark.parametrize("payload", [{}, {"note": "x"}, [1, 2]])
    def test_another_model_index_is_not(self, tmp_path, payload):
        assert diffusers_refusal(self._pipeline(tmp_path / "p", payload)) is None

    def test_a_folder_without_model_index_is_not(self, tmp_path):
        assert diffusers_refusal(tmp_path) is None

    def test_the_explainer_uses_it(self, tmp_path):
        folder = self._pipeline(tmp_path / "p", {"_class_name": "FluxPipeline"})
        assert explain_unsupported_model(folder) == diffusers_refusal(folder)
