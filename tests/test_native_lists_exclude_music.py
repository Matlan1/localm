# SPDX-License-Identifier: AGPL-3.0-or-later
"""ACE-Step music files never show up as image or video models.

The Images and Video model pickers (``pairing.native_options``, behind
``GET /api/imagine/native-models`` and the video route) and the Music list
(``native_models.unsupported_music``) read one classifier,
``families.music_entry``. A registry holding ACE-Step files in every format
localm reads next to a real image and a real video model must list only the
image model as an image model and only the video model as a video model.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import native_models
from localm.media.sdcpp import families, pairing


def _safetensors(path: Path, shapes: dict) -> Path:
    header = {name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 16)
    return path


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path: Path, shapes: dict, arch: str) -> Path:
    kvs = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string(arch)
    infos = b""
    for i, (name, shape) in enumerate(shapes.items()):
        ne = tuple(reversed(shape))
        infos += _gguf_string(name) + struct.pack("<I", len(ne))
        infos += struct.pack(f"<{len(ne)}Q", *ne) + struct.pack("<I", 0)
        infos += struct.pack("<Q", i * 32)
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(shapes), 1) + kvs + infos
    body += b"\0" * ((32 - len(body) % 32) % 32) + b"\0" * 64
    path.write_bytes(body)
    return path


# The ComfyUI all-in-one ACE-Step v1 checkpoint (ace_step_v1_3.5b.safetensors).
ACESTEP_V1_CHECKPOINT = {
    "model.diffusion_model.genre_embedder.weight": (2560, 1024),
    "model.diffusion_model.lyric_embs.weight": (6693, 1024),
    "model.diffusion_model.lyric_encoder.after_norm.weight": (1024,),
    "model.diffusion_model.transformer_blocks.0.attn.to_q.weight": (2560, 2560),
    "model.diffusion_model.final_layer.linear.weight": (32, 2560),
    "text_encoders.umt5base.transformer.shared.weight": (256384, 768),
    "vae.decoder.conv_in.weight": (512, 8, 3, 3),
}
# The same transformer as a diffusers file (ace_step_transformer/).
ACESTEP_V1_DIT = {k[len("model.diffusion_model."):]: v
                  for k, v in ACESTEP_V1_CHECKPOINT.items()
                  if k.startswith("model.diffusion_model.")}
# The ACE-Step 1.5 DiT under the tensor names its GGUF carries.
ACESTEP_15_DIT = {"decoder.condition_embedder.weight": (2048, 2048),
                  "decoder.layers.0.self_attn.q_proj.weight": (2048, 2048),
                  "encoder.lyric_encoder.embed_tokens.weight": (2048, 1024),
                  "detokenizer.embed_tokens.weight": (2048, 64),
                  "null_condition_emb": (1, 1, 2048)}
SDXL_UNET = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
             "model.diffusion_model.label_emb.0.0.weight": (1280, 2816)}
WAN21_T2V = {"model.diffusion_model.patch_embedding.weight": (1536, 16, 1, 2, 2),
             "model.diffusion_model.blocks.0.cross_attn.k.weight": (1536, 1536),
             "model.diffusion_model.blocks.0.cross_attn.norm_k.weight": (1536,)}

MUSIC = {"ace_step_v1_3.5b.safetensors", "ace_step_transformer",
         "acestep-v15-dit-safetensors", "acestep-v15-turbo-Q4_K_M"}


@pytest.fixture
def reg(tmp_path):
    def entry(path, **extra):
        return {"path": str(path), "model_type": "diffusion-unet", **extra}
    return {
        "ace_step_v1_3.5b.safetensors": entry(_safetensors(
            tmp_path / "ace_step_v1_3.5b.safetensors", ACESTEP_V1_CHECKPOINT)),
        "ace_step_transformer": entry(_safetensors(
            tmp_path / "diffusion_pytorch_model.safetensors", ACESTEP_V1_DIT)),
        "acestep-v15-dit-safetensors": entry(_safetensors(
            tmp_path / "acestep-v15-turbo.safetensors", ACESTEP_15_DIT)),
        "acestep-v15-turbo-Q4_K_M": entry(_gguf(
            tmp_path / "acestep-v15-turbo-Q4_K_M.gguf",
            {"decoder.condition_embedder.weight": (2048, 2048)}, "acestep-dit"),
            architecture="acestep-dit"),
        "sdxl-unet": entry(_safetensors(tmp_path / "sdxl_unet.safetensors", SDXL_UNET)),
        "wan2.1_t2v": entry(_safetensors(tmp_path / "wan2.1_t2v.safetensors", WAN21_T2V)),
    }


def _names(options: dict) -> list:
    return sorted(m["name"] for m in options["models"])


def test_the_image_list_offers_the_image_model_and_no_music_file(reg):
    assert _names(pairing.native_options("image", reg=reg)) == ["sdxl-unet"]


def test_the_video_list_offers_the_video_model_and_no_music_file(reg):
    assert _names(pairing.native_options("video", reg=reg)) == ["wan2.1_t2v"]


def test_no_music_file_is_a_component_candidate(reg):
    assert not MUSIC & {c.name for c in pairing.registry_candidates(reg)}


def test_every_music_entry_is_classified_as_music_by_the_one_classifier(reg):
    assert {n for n, e in reg.items() if families.music_entry(e) is not None} == MUSIC


def test_the_music_list_names_the_v1_files_it_cannot_run(reg):
    out = native_models.unsupported_music(reg)
    assert [u["name"] for u in out] == ["ace_step_transformer", "ace_step_v1_3.5b.safetensors"]
    assert all(u["reason"] == native_models.ACESTEP_V1_REASON for u in out)


def test_the_image_route_excludes_the_music_checkpoint(reg, monkeypatch):
    import localm.config as cfgmod
    import localm.model_manager as mm
    monkeypatch.setattr(cfgmod, "load_registry", lambda: reg)
    monkeypatch.setattr(mm, "load_registry", lambda: reg)

    class _Backend:
        @staticmethod
        def settings(_cfg):
            return {"native": {}}

    app = FastAPI()
    app.include_router(native_models.make_router("image", "imagine", _Backend()))
    with TestClient(app) as c:
        r = c.get("/api/imagine/native-models")
    assert r.status_code == 200, r.text
    assert sorted(m["name"] for m in r.json()["models"]) == ["sdxl-unet"]
