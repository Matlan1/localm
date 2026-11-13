# SPDX-License-Identifier: AGPL-3.0-or-later
"""Relocate re-points every kind of file registration accepts: an image or video
model's .safetensors / .sft component (diffusion model, VAE, LoRA, text encoder)
and a .pth upscaler, checked the way registration checks them. A file of another
kind is refused for that entry."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from localm.config import load_registry
from localm.model_manager import add_local
from localm.model_manager.registry import relocate_model, relocate_target
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
