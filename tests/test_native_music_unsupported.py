# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Music picker lists registered music checkpoints the native backend cannot
run (ACE-Step v1 all-in-one safetensors) with a reason, instead of hiding them."""

import json
import struct

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import native_models


def _safetensors(path, names):
    header = {"__metadata__": {}}
    offset = 0
    for n in names:
        header[n] = {"dtype": "F16", "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)
    return path


V1_TENSORS = [
    "text_encoders.umt5base.logit_scale",
    "model.diffusion_model.genre_embedder.weight",
    "model.diffusion_model.lyric_embs.weight",
    "model.diffusion_model.final_layer.linear.bias",
]


def _reg(tmp_path):
    v1 = _safetensors(tmp_path / "ace_step_v1_3.5b.safetensors", V1_TENSORS)
    sd = _safetensors(tmp_path / "sd15.safetensors", ["model.diffusion_model.input_blocks.0.0.weight"])
    partial = _safetensors(tmp_path / "partial.safetensors", V1_TENSORS[:2])
    gguf = tmp_path / "acestep-v15-turbo.gguf"
    gguf.write_bytes(b"GGUF")
    lora = _safetensors(tmp_path / "v1-lora.safetensors", V1_TENSORS)
    return {
        "ace_step_v1_3.5b.safetensors": {"path": str(v1), "model_type": "diffusion-unet"},
        "sd15": {"path": str(sd), "model_type": "diffusion-unet"},
        "partial": {"path": str(partial), "model_type": "diffusion-unet"},
        "acestep-v15-turbo": {"path": str(gguf), "model_type": "diffusion-unet",
                              "architecture": "acestep-dit"},
        "v1-lora": {"path": str(lora), "model_type": "lora"},
        "gone": {"path": str(tmp_path / "missing.safetensors"), "model_type": "diffusion-unet"},
        "llm": {"path": str(tmp_path / "x.gguf"), "model_type": "llm"},
        "bad-entry": "not-a-dict",
    }


def test_only_the_acestep_v1_checkpoint_is_listed_with_a_reason(tmp_path):
    out = native_models.unsupported_music(_reg(tmp_path))
    assert out == [{"name": "ace_step_v1_3.5b.safetensors",
                    "file": "ace_step_v1_3.5b.safetensors",
                    "reason": native_models.ACESTEP_V1_REASON}]
    assert "ComfyUI" in out[0]["reason"] and "ACE-Step v1" in out[0]["reason"]


def test_a_header_with_only_some_v1_markers_is_not_flagged(tmp_path):
    p = _safetensors(tmp_path / "x.safetensors", V1_TENSORS[:2])
    assert native_models.is_acestep_v1_checkpoint(p) is False


def test_unreadable_and_oversize_headers_are_not_flagged(tmp_path, monkeypatch):
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")
    assert native_models.is_acestep_v1_checkpoint(short) is False
    assert native_models.is_acestep_v1_checkpoint(tmp_path / "nope.safetensors") is False
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    assert native_models.is_acestep_v1_checkpoint(huge) is False


def test_the_music_route_reports_the_unsupported_list(tmp_path, monkeypatch):
    import localm.config as cfgmod
    reg = _reg(tmp_path)
    monkeypatch.setattr(cfgmod, "load_registry", lambda: reg)

    class _Backend:
        @staticmethod
        def settings(_cfg):
            return {"native": {}}

    app = FastAPI()
    app.include_router(native_models.make_music_router(_Backend()))
    with TestClient(app) as c:
        body = c.get("/api/music/native-models").json()
    assert [u["name"] for u in body["unsupported"]] == ["ace_step_v1_3.5b.safetensors"]
    assert body["components"]["dit"][0]["name"] == "acestep-v15-turbo"
