# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-folder scan typing: each scanned file is typed from its header
(``localm.media.sdcpp.families``) with the folder name as the fallback, ComfyUI's
current folder names and StabilityMatrix's ``Data/Models`` tree are recognised,
and a StabilityMatrix install, data or models folder is a valid scan source."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from localm import model_manager as mm
from localm.model_manager import scan
from localm.model_manager.scan import (
    SUBFOLDER_MAPPING, _discover_comfy_files, preview_comfy_models, scan_comfy_models,
)
from tests.test_sdcpp_families import (
    CLIP_VISION, FLUX_AE, FLUX_DEV, FLUX_LORA, LLM_LORA, T5XXL, write_gguf, write_safetensors,
)
from tests.test_unified_model_browser import temp_registry  # noqa: F401


@pytest.fixture(autouse=True)
def _comfy_offline():
    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        yield


def _discover(models: Path) -> dict[str, str]:
    found, method = _discover_comfy_files(models)
    assert found is not None and method == "folder-walk"
    return {p.relative_to(models).as_posix(): t for p, t in found.items()}


def test_header_overrides_the_folder_a_file_was_dropped_in(tmp_path):
    models = tmp_path / "models"
    (models / "checkpoints").mkdir(parents=True)
    write_safetensors(models / "checkpoints" / "a.safetensors", FLUX_AE)
    (models / "vae").mkdir()
    write_safetensors(models / "vae" / "flux_dev.safetensors", FLUX_DEV)
    (models / "unet").mkdir()
    write_safetensors(models / "unet" / "mine.safetensors", FLUX_LORA)
    assert _discover(models) == {
        "checkpoints/a.safetensors": "vae",
        "vae/flux_dev.safetensors": "diffusion-unet",
        "unet/mine.safetensors": "lora",
    }


def test_header_types_files_in_folders_the_table_does_not_know(tmp_path):
    models = tmp_path / "models"
    (models / "my_stuff").mkdir(parents=True)
    write_safetensors(models / "my_stuff" / "t5.safetensors", T5XXL)
    (models / "my_stuff" / "readme.bin").write_bytes(b"x")
    assert _discover(models) == {
        "my_stuff/t5.safetensors": "text-encoder",
        "my_stuff/readme.bin": "unknown",
    }


def test_folder_types_a_file_the_header_cannot_classify(tmp_path):
    models = tmp_path / "models"
    for sub in ("DiffusionModels", "ClipVision", "TextEncoders", "Lora", "ControlNet"):
        (models / sub).mkdir(parents=True)
        (models / sub / "opaque.safetensors").write_bytes(b"DATA")
    assert _discover(models) == {
        "DiffusionModels/opaque.safetensors": "diffusion-unet",
        "ClipVision/opaque.safetensors": "text-encoder",
        "TextEncoders/opaque.safetensors": "text-encoder",
        "Lora/opaque.safetensors": "lora",
        "ControlNet/opaque.safetensors": "controlnet",
    }


def test_clip_vision_file_is_a_text_encoder_by_header_and_by_folder(tmp_path):
    models = tmp_path / "models"
    (models / "clip_vision").mkdir(parents=True)
    write_safetensors(models / "clip_vision" / "h.safetensors", CLIP_VISION)
    assert _discover(models) == {"clip_vision/h.safetensors": "text-encoder"}


@pytest.mark.parametrize("folder,expected", [
    ("diffusion_models", "diffusion-unet"), ("DIFFUSIONMODELS", "diffusion-unet"),
    ("StableDiffusion", "diffusion-unet"), ("text_encoders", "text-encoder"),
    ("TextEncoders", "text-encoder"), ("clip_vision", "text-encoder"),
    ("ClipVision", "text-encoder"), ("VAE", "vae"), ("ApproxVAE", "vae"),
    ("Lora", "lora"), ("LyCORIS", "lora"), ("upscale_models", "upscaler"),
    ("ESRGAN", "upscaler"), ("embeddings", "textual-inversion"), ("controlnet", "controlnet"),
    ("Embeddings", "textual-inversion"), ("ControlNet", "controlnet"), ("ipadapter", "unknown"),
])
def test_folder_names_are_matched_case_insensitively(tmp_path, folder, expected):
    models = tmp_path / "models"
    (models / folder).mkdir(parents=True)
    (models / folder / "opaque.safetensors").write_bytes(b"DATA")
    assert _discover(models) == {f"{folder}/opaque.safetensors": expected}


def test_every_mapping_key_is_lower_case_and_a_registry_type():
    assert all(k == k.lower() for k in SUBFOLDER_MAPPING)
    assert set(SUBFOLDER_MAPPING.values()) <= mm.MODEL_TYPES


def test_llm_lora_and_llm_gguf_keep_their_folder_type(tmp_path):
    models = tmp_path / "models"
    (models / "loras").mkdir(parents=True)
    write_safetensors(models / "loras" / "chat.safetensors", LLM_LORA)
    (models / "text_encoders").mkdir()
    write_gguf(models / "text_encoders" / "qwen.gguf", {"token_embd.weight": (151936, 2560)}, "qwen3")
    assert _discover(models) == {
        "loras/chat.safetensors": "lora",
        "text_encoders/qwen.gguf": "text-encoder",
    }


def test_files_inside_a_component_directory_take_the_folder_type(tmp_path):
    models = tmp_path / "models"
    part = models / "Diffusers" / "sd-model" / "vae"
    part.mkdir(parents=True)
    (part / "config.json").write_text("{}", encoding="utf-8")
    write_safetensors(part / "diffusion_pytorch_model.safetensors", FLUX_AE)
    assert _discover(models) == {
        "Diffusers/sd-model/vae/diffusion_pytorch_model.safetensors": "unknown",
    }


def test_object_info_still_refines_the_header_type(tmp_path):
    models = tmp_path / "models"
    (models / "unet").mkdir(parents=True)
    write_safetensors(models / "unet" / "m.safetensors", FLUX_DEV)
    info = {"VAELoader": {"input": {"required": {"vae_name": [["unet/m.safetensors"]]}}}}
    with patch("localm.model_manager.scan.comfy_object_info", return_value=info):
        found, method = _discover_comfy_files(models)
    assert method.startswith("hybrid")
    assert list(found.values()) == ["vae"]


def _stability_matrix(root: Path) -> Path:
    models = root / "Data" / "Models"
    layout = {
        "StableDiffusion": FLUX_DEV, "VAE": FLUX_AE, "TextEncoders": T5XXL,
        "Lora": FLUX_LORA, "ClipVision": CLIP_VISION,
    }
    for sub, shapes in layout.items():
        (models / sub).mkdir(parents=True)
        write_safetensors(models / sub / f"{sub.lower()}.safetensors", shapes)
    (models / "ESRGAN").mkdir()
    (models / "ESRGAN" / "x4.pth").write_bytes(b"x")
    (models / "Karlo").mkdir()
    return models


EXPECTED_SM = {
    "StableDiffusion/stablediffusion.safetensors": "diffusion-unet",
    "VAE/vae.safetensors": "vae",
    "TextEncoders/textencoders.safetensors": "text-encoder",
    "Lora/lora.safetensors": "lora",
    "ClipVision/clipvision.safetensors": "text-encoder",
    "ESRGAN/x4.pth": "upscaler",
}


@pytest.mark.parametrize("pick", ["install", "data", "models"])
def test_stability_matrix_install_data_and_models_folders_are_scan_sources(tmp_path, pick):
    models = _stability_matrix(tmp_path / "StabilityMatrix")
    root = {"install": models.parent.parent, "data": models.parent, "models": models}[pick]
    with patch("localm.model_manager.scan.load_registry", return_value={}):
        prev = preview_comfy_models(workdir=str(root))
    assert prev.method == "folder-walk"
    assert prev.already_registered == 0
    expect = {}
    for t in EXPECTED_SM.values():
        expect[t] = expect.get(t, 0) + 1
    assert prev.counts == expect


def test_comfyui_models_folder_itself_is_a_scan_source(tmp_path):
    models = tmp_path / "ComfyUI" / "models"
    (models / "diffusion_models").mkdir(parents=True)
    write_safetensors(models / "diffusion_models" / "m.safetensors", FLUX_DEV)
    with patch("localm.model_manager.scan.load_registry", return_value={}):
        by_checkout = preview_comfy_models(workdir=str(models.parent))
        by_models = preview_comfy_models(workdir=str(models))
    assert by_checkout.counts == by_models.counts == {"diffusion-unet": 1}


def test_a_folder_with_no_models_is_still_reported_as_not_found(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    res = preview_comfy_models(workdir=str(empty))
    assert res.method.startswith("none (models folder not found under ")
    assert res.counts == {}


def test_scan_registers_stability_matrix_files_with_their_header_types(tmp_path, temp_registry):  # noqa: F811
    models = _stability_matrix(tmp_path / "StabilityMatrix")
    res = scan_comfy_models(workdir=str(models.parent.parent))
    assert (res.added, res.skipped) == (len(EXPECTED_SM), 0)
    reg = mm.load_registry()
    assert {Path(e["path"]).relative_to(models).as_posix(): e["model_type"]
            for e in reg.values()} == EXPECTED_SM
    again = scan_comfy_models(workdir=str(models.parent.parent))
    assert (again.added, again.skipped) == (0, len(EXPECTED_SM))


def test_scan_module_uses_the_registry_media_kind_table():
    from localm.model_manager.registry import _MEDIA_KIND_TYPES
    assert scan._MEDIA_KIND_TYPES is _MEDIA_KIND_TYPES
