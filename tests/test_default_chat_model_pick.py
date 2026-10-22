# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model a plain `localm serve` / `localm gui` starts with is a chat model.

A registered speech-to-text model (Qwen3-ASR loads through the chat path and is
registered as an ``llm``) used to win on its name alone: "asr" sorts before
"q3". It now comes after every other candidate, and is still picked when it is
the only one.
"""

from rich.console import Console

from localm.config import save_registry
from localm.model_manager import auto_chat_candidates, is_transcription_model
from localm.plugins.gui.cli import _select_startup_model


def _write_model(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"GGUF")
    return str(path)


def _pick(tmp_path, entries):
    registry = {name: {"model_type": "llm", "path": _write_model(tmp_path, filename)}
                for name, filename in entries.items()}
    save_registry(registry)
    _registry, model, model_less = _select_startup_model(
        Console(quiet=True), None, no_model=False, pull_spec=None, api_mode=True)
    return model, model_less


def test_an_asr_model_is_not_picked_over_a_chat_model(tmp_path):
    model, model_less = _pick(tmp_path, {
        "asr": "Qwen3-ASR-0.6B-Q8_0.gguf", "q3": "Qwen3-0.6B-Q8_0.gguf"})
    assert (model, model_less) == ("q3", False)


def test_an_asr_model_is_still_the_default_when_it_is_the_only_one(tmp_path):
    model, model_less = _pick(tmp_path, {"asr": "Qwen3-ASR-0.6B-Q8_0.gguf"})
    assert (model, model_less) == ("asr", False)


def test_the_file_name_marks_a_transcription_model_when_the_alias_does_not(tmp_path):
    model, _ = _pick(tmp_path, {
        "a-voice": "whisper-small-Q8_0.gguf", "b-chat": "gemma-4-E4B-it-Q4_K_M.gguf"})
    assert model == "b-chat"


def test_chat_models_keep_their_name_order(tmp_path):
    model, _ = _pick(tmp_path, {
        "zeta": "zeta-Q8_0.gguf", "alpha": "alpha-Q8_0.gguf", "asr": "asr-Q8_0.gguf"})
    assert model == "alpha"


def test_words_that_merely_contain_asr_are_not_transcription_models():
    assert not is_transcription_model("jurassic-chat", {"path": "/m/jurassic-7b.gguf"})
    assert not is_transcription_model("pasrat", {"path": "/m/x.gguf"})
    assert is_transcription_model("Qwen3-ASR", {"path": "/m/x.gguf"})
    assert auto_chat_candidates({"a": {"path": "/m/asr.gguf"}, "b": {"path": "/m/b.gguf"}}) == ["b", "a"]


def test_the_music_components_setup_music_registers_are_never_the_startup_chat_model(tmp_path):
    entries = {
        "Qwen3-Embedding-0.6B-Q8_0": ("Qwen3-Embedding-0.6B-Q8_0.gguf", "text-encoder"),
        "acestep-v15-turbo-Q4_K_M": ("acestep-v15-turbo-Q4_K_M.gguf", "diffusion-unet"),
        "vae-BF16": ("vae-BF16.gguf", "vae"),
        "acestep-5Hz-lm-0.6B-Q8_0": ("acestep-5Hz-lm-0.6B-Q8_0.gguf", "unknown"),
        "embedding-bge-small-en-v1.5": ("bge-small-en-v1.5-q8_0.gguf", "embedding"),
        "qcoder": ("Qwen2.5-Coder-7B-Instruct-Q6_K.gguf", "llm"),
    }
    registry = {name: {"model_type": kind, "path": _write_model(tmp_path, filename)}
                for name, (filename, kind) in entries.items()}
    save_registry(registry)
    _registry, model, model_less = _select_startup_model(
        Console(quiet=True), None, no_model=False, pull_spec=None, api_mode=True)
    assert (model, model_less) == ("qcoder", False)


def test_a_registry_of_only_components_starts_model_less(tmp_path):
    registry = {"Qwen3-Embedding-0.6B-Q8_0": {
        "model_type": "text-encoder", "path": _write_model(tmp_path, "te.gguf")}}
    save_registry(registry)
    _registry, model, model_less = _select_startup_model(
        Console(quiet=True), None, no_model=False, pull_spec=None, api_mode=True)
    assert (model, model_less) == (None, True)


def test_no_component_type_is_auto_chat_eligible():
    from localm.model_manager import is_auto_chat_eligible
    from localm.model_manager.registry import NON_CHAT_TYPES
    for kind in sorted(NON_CHAT_TYPES):
        assert is_auto_chat_eligible({"model_type": kind, "path": "/m/x.safetensors"}) is False, kind
    assert is_auto_chat_eligible({"model_type": "llm", "path": "/m/x.gguf"}) is True
