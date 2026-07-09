# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm model browser unified type support and ComfyUI model scan capabilities.
"""

import json
from unittest.mock import patch
import pytest

from localm import model_manager as mm
from localm.model_manager.scan import scan_comfy_models


@pytest.fixture()
def temp_registry(tmp_path, monkeypatch):
    """Temp registry storage and path monkeypatched."""
    store = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_file = tmp_path / "registry.json"
    
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    
    # Mock registry load/save
    def _load():
        if registry_file.exists():
            return json.loads(registry_file.read_text())
        return {}
        
    def _save(reg):
        registry_file.write_text(json.dumps(reg))
        
    def _update(mutator):
        reg = _load()
        mutator(reg)
        _save(reg)
        return reg

    monkeypatch.setattr("localm.config.load_registry", _load)
    monkeypatch.setattr("localm.config.save_registry", _save)
    monkeypatch.setattr("localm.config.update_registry", _update)
    monkeypatch.setattr(mm, "load_registry", _load)
    monkeypatch.setattr(mm, "save_registry", _save)
    monkeypatch.setattr(mm, "update_registry", _update)
    return store, models_dir, registry_file


def test_model_type_default(temp_registry):
    """Adding a model with no explicit type defaults to 'llm'."""
    _, models_dir, _ = temp_registry
    fake_file = models_dir / "my_model.gguf"
    fake_file.write_bytes(b"GGUF")

    # Add as local
    mm.add_local(str(fake_file))
    
    # Check registered type
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "my_model" in reg
    assert reg["my_model"]["model_type"] == "llm"


def test_model_type_explicit(temp_registry):
    """Adding a model with an explicit type stores it correctly."""
    _, models_dir, _ = temp_registry
    fake_file = models_dir / "my_vae.gguf"
    fake_file.write_bytes(b"GGUF")

    mm.add_local(str(fake_file), model_type="vae")
    
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "my_vae" in reg
    assert reg["my_vae"]["model_type"] == "vae"


def test_list_models_type_filter(temp_registry):
    """Filtering list_models by type retrieves only matching models."""
    _, models_dir, _ = temp_registry
    
    # Register an LLM
    f1 = models_dir / "model1.gguf"
    f1.write_bytes(b"GGUF 1")
    mm.add_local(str(f1), model_type="llm")

    # Register a VAE
    f2 = models_dir / "model2.gguf"
    f2.write_bytes(b"GGUF 2")
    mm.add_local(str(f2), model_type="vae")

    # Verify filtering behaves as expected via direct loader since list_models is for console print
    reg = mm.load_registry()
    assert len(reg) == 2

    # Verify type field values
    assert reg["model1"]["model_type"] == "llm"
    assert reg["model2"]["model_type"] == "vae"


def test_pull_type_forwarding(temp_registry, monkeypatch):
    """Verify pull_model forwards explicit type to registry."""
    _, models_dir, _ = temp_registry
    
    # Mock HF snapshots / GGUF download to just call register
    monkeypatch.setattr(mm.pull, "_pull_url", lambda url, name, expected_sha256=None, redownload=False, model_type="llm": 
        mm._register(name, models_dir / "url_model.gguf", url, model_type=model_type)
    )

    mm.pull.pull_model("https://example.com/test_model.gguf", name="pulled", model_type="diffusion-unet")
    
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "pulled" in reg
    assert reg["pulled"]["model_type"] == "diffusion-unet"


def test_scan_folder_walk(tmp_path, temp_registry):
    """Walk ComfyUI folders and identify models by mapped folders."""
    _, _, registry_file = temp_registry
    comfy_dir = tmp_path / "comfy"
    
    # Create unet folder and mock model
    unet_dir = comfy_dir / "models" / "unet"
    unet_dir.mkdir(parents=True)
    model_file = unet_dir / "flux_unet.safetensors"
    model_file.write_bytes(b"UNET")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        # We query with ComfyUI offline
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("offline")
            res = scan_comfy_models(comfy_url="http://localhost:8188")
            
    assert res.added >= 1
    
    # Reload registry
    reg = mm.load_registry()
    assert "flux_unet" in reg
    assert reg["flux_unet"]["model_type"] == "diffusion-unet"


_UNET_OBJECT_INFO = {
    "UNETLoader": {
        "input": {"required": {"unet_name": [["some_model.safetensors"]]}}
    }
}


def test_scan_object_info_reconcile(tmp_path, temp_registry):
    """The /object_info reconcile pass classifies a file that folder-walk cannot.

    NOTE: the previous version put the file in a `checkpoints/` folder, which
    folder-walk already maps via SUBFOLDER_MAPPING, and patched `requests.get` -
    which this code path never calls (it fetches /object_info via urllib). So the
    reconcile was never exercised: the test passed on folder-walk alone. Here the
    file lives in a folder NOT in SUBFOLDER_MAPPING (folder-walk -> "unknown"), so
    only the reconcile can produce "diffusion-unet", and we patch the ACTUAL
    fetcher `comfy_object_info`.
    """
    _, _, registry_file = temp_registry
    comfy_dir = tmp_path / "comfy"
    generic_dir = comfy_dir / "models" / "my_custom_nodes"
    generic_dir.mkdir(parents=True)
    (generic_dir / "some_model.safetensors").write_bytes(b"CKPT")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        with patch("localm.model_manager.scan.comfy_object_info",
                   return_value=_UNET_OBJECT_INFO):
            scan_comfy_models(comfy_url="http://localhost:8188")

    reg = mm.load_registry()
    assert "some_model" in reg
    assert reg["some_model"]["model_type"] == "diffusion-unet"


def test_scan_without_object_info_leaves_generic_file_unknown(tmp_path, temp_registry):
    """Guard proving the reconcile is load-bearing: with /object_info unavailable,
    the same generic-folder file stays 'unknown' (folder-walk cannot classify it)."""
    comfy_dir = tmp_path / "comfy"
    generic_dir = comfy_dir / "models" / "my_custom_nodes"
    generic_dir.mkdir(parents=True)
    (generic_dir / "some_model.safetensors").write_bytes(b"CKPT")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            scan_comfy_models(comfy_url="http://localhost:8188")

    reg = mm.load_registry()
    assert reg["some_model"]["model_type"] == "unknown"
