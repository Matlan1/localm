# SPDX-License-Identifier: AGPL-3.0-or-later
"""A scheduled job with no model set picks a chat model, never a speech-to-text one."""

from unittest.mock import MagicMock

import pytest

import localm.plugins.builtin.jobs.runner as runner_mod

REGISTRY = {
    "Qwen3-ASR-0.6B": {"model_type": "llm", "path": "asr/Qwen3-ASR-0.6B.gguf"},
    "whisper-small": {"model_type": "llm", "path": "w/whisper-small.gguf"},
    "zeta-chat": {"model_type": "llm", "path": "z/zeta-chat.gguf"},
}


@pytest.fixture
def no_model_configured(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.config.load_registry", lambda: dict(REGISTRY))
    monkeypatch.setattr("localm.inference.http_server._engine", None)


def test_default_chat_model_skips_speech_to_text(no_model_configured):
    assert sorted(REGISTRY)[0] == "Qwen3-ASR-0.6B"
    assert runner_mod._default_chat_model() == "zeta-chat"


def test_default_chat_model_falls_back_to_speech_to_text_when_alone(monkeypatch):
    only_asr = {"Qwen3-ASR-0.6B": REGISTRY["Qwen3-ASR-0.6B"]}
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.config.load_registry", lambda: dict(only_asr))
    assert runner_mod._default_chat_model() == "Qwen3-ASR-0.6B"


def test_load_engine_with_no_model_loads_the_chat_model(monkeypatch, no_model_configured):
    resolved = []

    def fake_info(name):
        resolved.append(name)
        return ("/models/" + name, name)

    monkeypatch.setattr("localm.model_manager.get_model_info", fake_info)
    engine_cls = MagicMock()
    monkeypatch.setattr("localm.inference.engine.Engine", engine_cls)

    eng, reused = runner_mod._load_engine(None)

    assert resolved == ["zeta-chat"]
    assert reused is False
    assert eng is engine_cls.return_value
