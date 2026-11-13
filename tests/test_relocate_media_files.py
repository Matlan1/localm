# SPDX-License-Identifier: AGPL-3.0-or-later
"""Relocate re-points every kind of path registration accepts, checked the way
registration checks it: an image or video model's .safetensors / .sft component
(diffusion model, VAE, LoRA, text encoder) and a .pth upscaler, an Ollama blob, a
HuggingFace model folder, a GGUF, and any other single file by its extension. A
path of another kind is refused for that entry."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from localm.config import load_registry
from localm.model_manager import add_local, save_registry
from localm.model_manager.registry import (
    _registered_file_suffix, relocate_model, relocate_target,
)
from tests.test_folder_add_loose_media import (  # noqa: F401
    NOT_MEDIA, _media_folder, home, write_safetensors,
)

_TYPES = {"wan_unet": "diffusion-unet", "wan_vae": "vae", "flux_lora": "lora",
          "clip_l": "text-encoder", "esrgan": "upscaler"}


@pytest.mark.parametrize("key", sorted(_TYPES))
def test_each_registered_media_kind_can_be_relocated(home, tmp_path, monkeypatch, key):  # noqa: F811
    files = _media_folder(tmp_path / "in", monkeypatch)
    src = files[key]
    assert add_local(str(src), key, on_duplicate="skip", fast=True)
    assert load_registry()[key]["model_type"] == _TYPES[key]
    moved = tmp_path / "moved" / src.name
    moved.parent.mkdir(parents=True)
    shutil.move(str(src), moved)
    p, reason = relocate_target(str(moved), key)
    assert reason is None and p == moved
    assert relocate_model(key, str(moved))
    assert Path(load_registry()[key]["path"]) == moved.resolve()


def test_a_file_of_another_kind_is_refused_for_the_entry(home, tmp_path, monkeypatch):  # noqa: F811
    files = _media_folder(tmp_path / "in", monkeypatch)
    assert add_local(str(files["wan_vae"]), "vae1", on_duplicate="skip", fast=True)
    junk = write_safetensors(tmp_path / "other" / "notes.safetensors", NOT_MEDIA)
    p, reason = relocate_target(str(junk), "vae1")
    assert p is None and reason == f"Not a vae model file: {junk}"
    assert not relocate_model("vae1", str(junk))
    assert Path(load_registry()["vae1"]["path"]) == files["wan_vae"].resolve()


def test_without_a_media_entry_a_safetensors_file_is_still_refused(home, tmp_path,  # noqa: F811
                                                                    monkeypatch):
    files = _media_folder(tmp_path / "in", monkeypatch)
    p, reason = relocate_target(str(files["wan_vae"]))
    assert p is None and reason.startswith("Not a GGUF model file")


def _gguf_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * 2048)
    return path


def test_an_ollama_blob_relocates(home, tmp_path):  # noqa: F811
    old = tmp_path / "ollama" / "blobs" / "sha256-abc"
    save_registry({"blobby": {"path": str(old), "source": "local", "model_type": "llm",
                              "missing": True}})
    new = _gguf_file(tmp_path / "moved" / "blobs" / "sha256-abc")
    p, reason = relocate_target(str(new), "blobby")
    assert reason is None and p == new
    assert relocate_model("blobby", str(new))


def test_a_folder_must_be_a_huggingface_model_folder_as_registration_requires(
        home, tmp_path):  # noqa: F811
    save_registry({"hf": {"path": str(tmp_path / "gone" / "hfmodel"), "source": "local",
                          "model_type": "llm", "missing": True}})
    gguf_repo = tmp_path / "repo"
    gguf_repo.mkdir()
    (gguf_repo / "config.json").write_text("{}", encoding="utf-8")
    (gguf_repo / "tokenizer.json").write_text("{}", encoding="utf-8")
    _gguf_file(gguf_repo / "model.gguf")
    p, reason = relocate_target(str(gguf_repo), "hf")
    assert p is None and reason == f"Not a HuggingFace model directory: {gguf_repo}"

    hf_dir = tmp_path / "hfmodel"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    write_safetensors(hf_dir / "model.safetensors", NOT_MEDIA)
    p, reason = relocate_target(str(hf_dir), "hf")
    assert reason is None and p == hf_dir


def test_another_single_file_kind_relocates_to_a_file_with_its_extension(
        home, tmp_path):  # noqa: F811
    save_registry({"voice": {"path": str(tmp_path / "gone" / "voice.onnx"), "source": "hf:o/r",
                             "model_type": "tts", "missing": True}})
    new = tmp_path / "moved" / "voice.onnx"
    new.parent.mkdir()
    new.write_bytes(b"onnx bytes")
    p, reason = relocate_target(str(new), "voice")
    assert reason is None and p == new
    assert relocate_model("voice", str(new))
    assert Path(load_registry()["voice"]["path"]) == new.resolve()

    other = tmp_path / "moved" / "voice.bin"
    other.write_bytes(b"bin bytes")
    p, reason = relocate_target(str(other), "voice")
    assert p is None
    assert reason == f"Not a .onnx file like the one this model was registered with: {other}"


def test_a_dotted_folder_name_is_not_read_as_a_file_extension(home, tmp_path):  # noqa: F811
    entry = {"path": str(tmp_path / "gone" / "Qwen2.5-7B"), "source": "local", "model_type": "llm"}
    save_registry({"qwen": entry})
    assert _registered_file_suffix(entry) is None
    odd = tmp_path / "x.5-7b"
    odd.write_bytes(b"not a model")
    p, reason = relocate_target(str(odd), "qwen")
    assert p is None and reason == f"Not a GGUF model file: {odd}"


def test_a_media_model_registered_as_gguf_relocates_to_a_gguf(home, tmp_path):  # noqa: F811
    save_registry({"unet": {"path": str(tmp_path / "gone" / "flux.gguf"), "source": "local",
                            "model_type": "diffusion-unet", "missing": True}})
    new = _gguf_file(tmp_path / "moved" / "flux.gguf")
    p, reason = relocate_target(str(new), "unet")
    assert reason is None and p == new
