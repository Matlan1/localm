# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audio input as a model capability: the projector header probe, the registry
tri-state, capability routing, and the model-info fields."""
import struct
from pathlib import Path

import pytest

from localm.inference import capability_routing as cr
from localm.model_manager import capabilities as caps
from localm.model_manager.gguf import gguf_mmproj_modalities
from localm.model_manager.registry import (
    model_audio_capability,
    model_vision_capability,
)

_STR, _BOOL, _U32 = 8, 7, 4


def _kv(key: str, vtype: int, value) -> bytes:
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
    if vtype == _STR:
        v = value.encode()
        return out + struct.pack("<Q", len(v)) + v
    if vtype == _BOOL:
        return out + struct.pack("<?", value)
    return out + struct.pack("<I", value)


def _gguf(path: Path, kvs) -> Path:
    body = b"".join(_kv(*kv) for kv in kvs)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)
    return path


def _projector(path: Path, *, vision=None, audio=None, arch="clip") -> Path:
    kvs = [("general.architecture", _STR, arch), ("general.type", _STR, "mmproj")]
    if vision is not None:
        kvs.append(("clip.has_vision_encoder", _BOOL, vision))
    if audio is not None:
        kvs.append(("clip.has_audio_encoder", _BOOL, audio))
    kvs.append(("clip.audio.num_mel_bins", _U32, 128))
    return _gguf(path, kvs)


class TestProjectorModalities:
    @pytest.mark.parametrize("vision,audio,expected", [
        (None, True, {"vision": False, "audio": True}),
        (True, None, {"vision": True, "audio": False}),
        (True, True, {"vision": True, "audio": True}),
        (False, False, {"vision": False, "audio": False}),
    ])
    def test_reads_the_encoder_flags(self, tmp_path, vision, audio, expected):
        p = _projector(tmp_path / "mmproj.gguf", vision=vision, audio=audio)
        assert gguf_mmproj_modalities(p) == expected

    def test_a_model_that_is_not_a_projector_is_none(self, tmp_path):
        p = _projector(tmp_path / "m.gguf", audio=True, arch="llama")
        assert gguf_mmproj_modalities(p) is None

    def test_unreadable_or_garbage_is_none(self, tmp_path):
        assert gguf_mmproj_modalities(tmp_path / "missing.gguf") is None
        (tmp_path / "x.gguf").write_bytes(b"x")
        assert gguf_mmproj_modalities(tmp_path / "x.gguf") is None

    def test_truncated_before_the_flags_is_none_not_false(self, tmp_path):
        p = _projector(tmp_path / "mmproj.gguf", audio=True)
        raw = p.read_bytes()
        cut = raw.index(b"clip.has_audio_encoder") + 10
        p.write_bytes(raw[:cut])
        assert gguf_mmproj_modalities(p) is None

    def test_truncated_after_both_flags_still_answers(self, tmp_path):
        p = _projector(tmp_path / "mmproj.gguf", vision=False, audio=True)
        raw = p.read_bytes()
        p.write_bytes(raw[:raw.index(b"clip.audio.num_mel_bins") + 5])
        assert gguf_mmproj_modalities(p) == {"vision": False, "audio": True}

    def test_the_real_qwen3_asr_projector(self):
        import os
        path = os.environ.get("LOCALM_TEST_AUDIO_MMPROJ")
        if not path or not Path(path).is_file():
            pytest.skip("LOCALM_TEST_AUDIO_MMPROJ names no projector file")
        assert gguf_mmproj_modalities(Path(path)) == {"vision": False, "audio": True}


@pytest.fixture
def library(tmp_path):
    """One folder per model, each with its own projector."""
    def model(name, **proj):
        d = tmp_path / name
        d.mkdir()
        (d / f"{name}.gguf").write_bytes(b"x")
        if proj.get("garbage"):
            (d / f"mmproj-{name}.gguf").write_bytes(b"x")
        elif proj:
            _projector(d / f"mmproj-{name}.gguf", **proj)
        return {"path": str(d / f"{name}.gguf"), "source": "local", "model_type": "llm"}

    return {
        "asr": model("asr", audio=True),
        "vlm": model("vlm", vision=True),
        "omni": model("omni", vision=True, audio=True),
        "text": model("text"),
        "legacy": model("legacy", garbage=True),
        "gone": {"path": "Z:/nonexistent/gone.gguf", "source": "local",
                 "model_type": "llm"},
    }


class TestRegistryTriState:
    @pytest.mark.parametrize("name,vision,audio", [
        ("asr", False, True),
        ("vlm", True, False),
        ("omni", True, True),
        ("text", False, False),
        ("legacy", True, None),
        ("gone", None, None),
    ])
    def test_capabilities_come_from_the_projector_header(self, library, name,
                                                         vision, audio):
        assert model_vision_capability(name, reg=library) is vision
        assert model_audio_capability(name, reg=library) is audio
        assert caps.model_capability(name, caps.AUDIO, reg=library) is audio

    def test_model_capabilities_lists_audio_input(self, library):
        got = caps.model_capabilities("asr", reg=library)
        assert got[caps.AUDIO] is True and got[caps.VISION] is False

    def test_hf_directory_audio_signals(self, tmp_path):
        import json
        whisper = tmp_path / "whisper"
        whisper.mkdir()
        (whisper / "preprocessor_config.json").write_text(
            json.dumps({"feature_extractor_type": "WhisperFeatureExtractor"}))
        clip = tmp_path / "clipvlm"
        clip.mkdir()
        (clip / "preprocessor_config.json").write_text(
            json.dumps({"feature_extractor_type": "CLIPFeatureExtractor"}))
        omni = tmp_path / "omni"
        omni.mkdir()
        (omni / "config.json").write_text(json.dumps({"audio_config": {}}))
        reg = {n: {"path": str(tmp_path / n), "source": "local", "model_type": "llm"}
               for n in ("whisper", "clipvlm", "omni")}
        assert model_audio_capability("whisper", reg=reg) is True
        assert model_audio_capability("clipvlm", reg=reg) is False
        assert model_audio_capability("omni", reg=reg) is True


def _audio_message():
    return [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "text", "text": "what is said"}]}]


class TestRouting:
    def test_a_message_with_audio_needs_audio_input(self):
        needs = cr.request_needs(_audio_message())
        assert caps.AUDIO in needs.capabilities
        assert caps.VISION not in needs.capabilities

    def test_audio_is_a_capability_a_request_names(self):
        from localm.inference.protocol import ChatRequest
        req = ChatRequest(model="m", messages=[{"role": "user", "content": "x"}],
                          required_capabilities=[caps.AUDIO])
        assert req.required_capabilities == [caps.AUDIO]

    @pytest.mark.parametrize("mode", ["image", "auto"])
    def test_audio_moves_a_request_off_a_model_that_cannot_hear(self, library, mode):
        decision = cr.plan_route("text", cr.request_needs(_audio_message()),
                                 pinned=False, resident=["text"], reg=library,
                                 mode=mode)
        assert decision.routed
        assert decision.resolved in ("asr", "omni")

    def test_a_model_that_hears_keeps_the_request(self, library):
        decision = cr.plan_route("asr", cr.request_needs(_audio_message()),
                                 pinned=False, resident=["asr"], reg=library,
                                 mode="auto")
        assert not decision.routed and decision.resolved == "asr"

    def test_a_loaded_engine_confirming_audio_is_not_a_gap(self, library):
        decision = cr.plan_route("legacy", cr.request_needs(_audio_message()),
                                 pinned=False, resident=["legacy"], reg=library,
                                 current_known={caps.AUDIO: True}, mode="auto")
        assert not decision.routed

    def test_an_audio_only_model_is_not_routed_an_image(self, library):
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
        decision = cr.plan_route("asr", cr.request_needs(msgs), pinned=False,
                                 resident=["asr"], reg=library, mode="auto")
        assert decision.routed and decision.resolved in ("vlm", "omni", "legacy")
        assert "asr" not in decision.candidates
