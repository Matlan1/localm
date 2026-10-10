# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a GGUF IS comes from its own header: chat model (diffusion language
models included), embedding, vision projector, or one of the llama.cpp
architectures that load but cannot chat (speculative-decoding draft heads, T5,
audio codec).

The architecture names are the ones in llama.cpp's own architecture table.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import pytest

from localm.inference.backends.base import UnsupportedModelRoleError
from localm.inference.backends.gguf import GgufBackend, _load_failure_message
from localm.model_manager import (
    gguf_architecture, gguf_chat_refusal, gguf_embedding_signal,
)
from localm.model_manager.gguf import gguf_tool_use_signal
from localm.model_manager.registry import _detect_local_model_type

_T_BOOL, _T_UINT32, _T_STRING = 7, 4, 8


def _kv_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", _T_STRING) + struct.pack("<Q", len(raw)) + raw


def _kv_bool(value: bool) -> bytes:
    return struct.pack("<I", _T_BOOL) + struct.pack("<?", value)


def _kv_uint32(value: int) -> bytes:
    return struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _gguf(path: Path, arch: str, extra=()) -> Path:
    kv = [("general.architecture", _kv_string(arch))] + list(extra)
    body = b""
    for key, encoded in kv:
        kb = key.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb + encoded
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    path.write_bytes(head + body + b"\0" * 2048)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


def _type_of(path: Path) -> str:
    return _detect_local_model_type(path, is_gguf=True, is_hf=False)[0]


CHAT_ARCHITECTURES = [
    "llama", "qwen3", "qwen3next", "qwen35", "gpt-oss", "gemma3", "gemma3n", "gemma4",
    "glm4moe", "lfm2", "smollm3", "mistral3", "minimax-m2", "deepseek2", "mamba2",
    "rwkv7", "granitehybrid", "qwen3vl", "pangu-embedded", "hunyuan-moe", "seed_oss",
    "t5",
]
EMBEDDING_ARCHITECTURES = [
    "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert", "jina-bert-v2",
    "jina-bert-v3", "eurobert", "gemma-embedding", "gemma-embedding2", "t5encoder",
    "llama-embed",
]
NON_CHAT_ARCHITECTURES = [
    "eagle3", "dflash", "gemma4-assistant",
    "wavtokenizer-dec", "pockettts",
]
TTS_ARCHITECTURES = ["qwen3tts"]
DIFFUSION_LM_ARCHITECTURES = ["dream", "llada", "llada-moe", "rnd1"]


IMAGE_ARCHITECTURES = [
    "flux", "sd3", "aura", "hidream", "cosmos", "hyvid", "wan", "ltxv", "sdxl", "sd1",
    "lumina2",
]

MUSIC_ARCHITECTURES = [
    ("acestep-dit", "diffusion-unet"), ("acestep-vae", "vae"),
    ("acestep-text-enc", "text-encoder"), ("acestep-lm", "unknown"),
]


class TestArchitectureRoles:
    @pytest.mark.parametrize("arch", CHAT_ARCHITECTURES)
    def test_chat_architectures_stay_llm(self, tmp_path, arch):
        assert _type_of(_gguf(tmp_path / "m.gguf", arch)) == "llm"

    @pytest.mark.parametrize("arch", EMBEDDING_ARCHITECTURES)
    def test_encoder_architectures_are_embeddings(self, tmp_path, arch):
        assert _type_of(_gguf(tmp_path / "m.gguf", arch)) == "embedding"

    @pytest.mark.parametrize("arch", NON_CHAT_ARCHITECTURES)
    def test_non_chat_architectures_are_not_llm(self, tmp_path, arch):
        f = _gguf(tmp_path / "m.gguf", arch)
        mtype, meta = _detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype == "unknown"
        assert meta["architecture"] == arch

    @pytest.mark.parametrize("arch", TTS_ARCHITECTURES)
    def test_text_to_speech_architectures_are_tts_and_never_chat(self, tmp_path, arch):
        f = _gguf(tmp_path / "m.gguf", arch)
        mtype, meta = _detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype == "tts" and meta["architecture"] == arch
        assert gguf_chat_refusal(arch) is not None

    @pytest.mark.parametrize("arch", IMAGE_ARCHITECTURES)
    def test_image_and_video_checkpoints_are_diffusion_models(self, tmp_path, arch):
        f = _gguf(tmp_path / "m.gguf", arch)
        assert _type_of(f) == "diffusion-unet"
        assert gguf_chat_refusal(arch) is not None

    @pytest.mark.parametrize("arch,mtype", MUSIC_ARCHITECTURES)
    def test_acestep_components_take_their_component_type(self, tmp_path, arch, mtype):
        f = _gguf(tmp_path / "m.gguf", arch)
        mtype_seen, meta = _detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype_seen == mtype
        assert meta["architecture"] == arch

    def test_pooling_type_key_marks_a_decoder_architecture_as_embedding(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", "qwen3", [("qwen3.pooling_type", _kv_uint32(3))])
        assert _type_of(f) == "embedding"

    def test_non_causal_key_marks_an_unlisted_encoder_as_embedding(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", "future-encoder",
                  [("future-encoder.attention.causal", _kv_bool(False))])
        assert gguf_embedding_signal(f) is True
        assert _type_of(f) == "embedding"

    def test_causal_true_stays_llm(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", "llama", [("llama.attention.causal", _kv_bool(True))])
        assert _type_of(f) == "llm"

    @pytest.mark.parametrize("arch", DIFFUSION_LM_ARCHITECTURES)
    def test_diffusion_lms_are_chat_models(self, tmp_path, arch):
        assert _type_of(_gguf(tmp_path / "m.gguf", arch)) == "llm"
        assert gguf_chat_refusal(arch) is None

    @pytest.mark.parametrize("arch", DIFFUSION_LM_ARCHITECTURES)
    def test_diffusion_lm_declaring_non_causal_is_never_an_embedding(self, tmp_path, arch):
        f = _gguf(tmp_path / "m.gguf", arch, [
            (f"{arch}.attention.causal", _kv_bool(False)),
            (f"{arch}.pooling_type", _kv_uint32(1)),
        ])
        assert gguf_embedding_signal(f) is False
        assert _type_of(f) == "llm"

    @pytest.mark.parametrize("arch", DIFFUSION_LM_ARCHITECTURES)
    def test_diffusion_lms_are_not_tool_callers(self, tmp_path, arch):
        f = _gguf(tmp_path / "m.gguf", arch, [
            ("tokenizer.chat_template",
             _kv_string("{% if tools %}<tool_call>{% endif %}{{ messages }}")),
        ])
        assert gguf_tool_use_signal(f) is False

    def test_encoder_decoder_declaring_non_causal_stays_llm(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", "t5", [("t5.attention.causal", _kv_bool(False))])
        assert gguf_embedding_signal(f) is False
        assert _type_of(f) == "llm"

    def test_mmproj_is_still_mmproj(self, tmp_path):
        assert _type_of(_gguf(tmp_path / "m.gguf", "clip")) == "mmproj"


class TestChatRefusal:
    @pytest.mark.parametrize("arch,what", [
        ("eagle3", "draft head"), ("dflash", "draft head"), ("gemma4-assistant", "draft head"),
        ("wavtokenizer-dec", "audio codec"),
        ("qwen3tts", "text-to-speech"), ("flux", "image or video generation"),
        ("acestep-dit", "music generation"), ("acestep-lm", "music generation"),
        ("acestep-vae", "music generation"), ("acestep-text-enc", "music generation"),
    ])
    def test_message_names_the_architecture_and_what_it_is(self, arch, what):
        msg = gguf_chat_refusal(arch)
        assert f"'{arch}'" in msg and what in msg

    @pytest.mark.parametrize("arch", CHAT_ARCHITECTURES + EMBEDDING_ARCHITECTURES
                             + DIFFUSION_LM_ARCHITECTURES + ["clip", "", None])
    def test_no_refusal_for_anything_else(self, arch):
        assert gguf_chat_refusal(arch) is None

    def test_architecture_reader(self, tmp_path):
        assert gguf_architecture(_gguf(tmp_path / "m.gguf", "qwen3next")) == "qwen3next"
        assert gguf_architecture(tmp_path / "missing.gguf") is None


class TestBackendRefusesBeforeLoading:
    @pytest.mark.parametrize("arch", ["eagle3", "wavtokenizer-dec", "flux", "acestep-lm",
                                      "qwen3tts"])
    def test_load_raises_before_any_vram_probe_or_worker(self, tmp_path, monkeypatch, arch):
        f = _gguf(tmp_path / "m.gguf", arch)
        backend = GgufBackend(str(f))
        for hook in ("_check_vram", "_load_native", "_effective_gpu_layers"):
            monkeypatch.setattr(backend, hook, lambda *a, _hook=hook, **k: pytest.fail(
                f"{_hook} ran for a model that cannot chat"))
        with pytest.raises(UnsupportedModelRoleError) as caught:
            backend.load()
        assert f"'{arch}'" in str(caught.value)
        assert "setup-llama" not in str(caught.value)


class TestUnknownArchitectureLoadMessage:
    NATIVE_TAIL = (
        "Failed to load model: m.gguf\n"
        "llama_model_load: error loading model: error loading model architecture: "
        "unknown model architecture: 'qwen9next'\n")

    def test_names_the_architecture_and_blames_the_runtime_age_not_the_install(self):
        msg = _load_failure_message(RuntimeError(self.NATIVE_TAIL))
        assert "'qwen9next'" in msg
        assert "newer than the runtime" in msg
        assert "Provision or repair" not in msg
        assert "setup-llama --tag latest" in msg

    def test_other_runtime_errors_keep_the_repair_advice(self):
        msg = _load_failure_message(RuntimeError("Failed to load model: m.gguf"))
        assert "Provision or repair it" in msg
