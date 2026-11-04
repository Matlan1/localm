# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native image and video model pickers never offer an ACE-Step music
component or a registered folder as the main model, while a diffusion-unet file
whose header names no known family stays listed."""

import json
import os
import struct
import time
from pathlib import Path

import pytest

from localm.media.sdcpp import pairing
from localm.model_manager import add_local

_T_STRING = 8


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as mm
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "ensure_dirs", lambda: (home / "models").mkdir(parents=True, exist_ok=True))
    return home


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path: Path, arch: str) -> Path:
    kv = [("general.architecture", arch), ("general.name", arch)]
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, val in kv:
        out += [_s(key), struct.pack("<I", _T_STRING), _s(val)]
    body = b"".join(out)
    path.write_bytes(body + bytes(4096 - len(body)))
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


@pytest.mark.parametrize("media", ["image", "video"])
def test_the_music_diffusion_component_is_not_a_media_model_choice(
        tmp_path, isolated_home, media):
    assert add_local(str(_gguf(tmp_path / "acestep-dit.gguf", "acestep-dit"))) is True
    from localm.config import load_registry
    reg = load_registry()
    assert reg["acestep-dit"]["model_type"] == "diffusion-unet"
    assert reg["acestep-dit"]["architecture"] == "acestep-dit"

    names = {m["name"] for m in pairing.native_options(media)["models"]}
    assert "acestep-dit" not in names


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_diffusion_unet_with_no_known_family_is_still_offered(
        tmp_path, isolated_home, media):
    from localm.config import load_registry, save_registry
    assert add_local(str(_gguf(tmp_path / "mystery-unet.gguf", "mystery"))) is True
    reg = load_registry()
    reg["mystery-unet"]["model_type"] = "diffusion-unet"
    save_registry(reg)

    options = pairing.native_options(media)["models"]
    assert [m["name"] for m in options] == ["mystery-unet"]
    assert options[0]["kind"] == "unknown"


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_pipeline_part_folder_is_not_a_media_model_choice(isolated_home, media):
    from localm.config import load_registry, save_registry
    from localm.model_manager import sync_models_dir
    folder = isolated_home / "models" / "ace_step_transformer"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"_class_name": "ACEStepTransformer2DModel"}), encoding="utf-8")
    reg = load_registry()
    reg["ace_step_transformer"] = {"path": str(folder), "source": "hf", "model_type": "unknown"}
    save_registry(reg)
    sync_models_dir(backfill_mmproj=False)
    assert load_registry()["ace_step_transformer"]["model_type"] == "diffusion-unet"

    assert pairing.native_options(media)["models"] == []


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_folder_entry_is_left_out_beside_a_listed_file(tmp_path, isolated_home, media):
    from localm.config import load_registry, save_registry
    assert add_local(str(_gguf(tmp_path / "mystery-unet.gguf", "mystery"))) is True
    folder = tmp_path / "a-folder"
    folder.mkdir()
    reg = load_registry()
    reg["mystery-unet"]["model_type"] = "diffusion-unet"
    reg["a-folder"] = {"path": str(folder), "source": "local", "model_type": "diffusion-unet"}
    save_registry(reg)

    assert [m["name"] for m in pairing.native_options(media)["models"]] == ["mystery-unet"]


def test_the_folder_check_never_touches_a_network_path(isolated_home, monkeypatch):
    from localm.config import load_registry, save_registry
    reg = load_registry()
    reg["on-a-share"] = {"path": r"\\server\share\m.gguf", "source": "local",
                         "model_type": "diffusion-unet"}
    save_registry(reg)
    touched = []
    real = Path.is_dir

    def spy(self, *a, **k):
        touched.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "is_dir", spy)
    pairing.native_options("image")
    assert not [p for p in touched if "server" in p]
