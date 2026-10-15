# SPDX-License-Identifier: AGPL-3.0-or-later
"""A single .safetensors file pulled from Hugging Face is typed by its own
tensor layout, the same way ``localm add`` types it, not by the repository's
tags. Only the network (the HF metadata query, the download and its HEAD) is
faked; the typing runs for real."""

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import localm.model_manager as mm
from localm.config import load_registry

SDXL_LIGHTNING_REPO_META = {
    "pipeline_tag": "text-to-image", "library_name": "diffusers",
    "tags": ["diffusers", "text-to-image", "stable-diffusion", "license:openrail++"],
}
SDXL_LORA = {
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
        (64, 2048),
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight":
        (640, 64),
    "lora_unet_down_blocks_2_attentions_0_transformer_blocks_2_attn1_to_k.lora_down.weight":
        (64, 1280),
}
SDXL_UNET = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
             "model.diffusion_model.label_emb.0.0.weight": (1280, 2816)}
SPEC = "ByteDance/SDXL-Lightning:sdxl_lightning_4step_lora.safetensors"


def _entry(filename: str) -> dict:
    """The one registry entry whose file is *filename*."""
    hits = [e for e in load_registry().values()
            if Path(str(e.get("path") or "")).name == filename]
    assert len(hits) == 1, load_registry()
    return hits[0]


def _safetensors_bytes(shapes: dict) -> bytes:
    header = {name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 16


@pytest.fixture
def hf_network(tmp_path, monkeypatch):
    """Fakes Hugging Face: the repo metadata query answers with
    *SDXL_LIGHTNING_REPO_META* and a download writes ``state["body"]``."""
    import huggingface_hub
    state = {"body": b"", "downloads": 0, "queries": []}

    def _fake_get(url, params=None, *a, **kw):
        state["queries"].append(url)
        return dict(SDXL_LIGHTNING_REPO_META)

    def _fake_download(repo_id, filename, local_dir, **kw):
        state["downloads"] += 1
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(state["body"])
        return str(p)

    monkeypatch.setattr("localm.discover._get", _fake_get)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", MagicMock())
    monkeypatch.setattr("localm.model_manager.pull._policy_head",
                        lambda *a, **k: MagicMock(
                            headers={"content-length": str(len(state["body"]))}))
    monkeypatch.setattr("localm.model_manager.pull._hf_endpoint_refusal", lambda: None)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, fn: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    state["models"] = models
    return state


def test_the_repo_tags_alone_would_say_diffusion_unet(hf_network):
    from localm.model_manager.pull import _hf_pipeline_tag_to_type
    assert _hf_pipeline_tag_to_type("ByteDance/SDXL-Lightning") == "diffusion-unet"
    assert hf_network["queries"]


def test_a_pulled_lora_safetensors_registers_as_a_lora(hf_network):
    hf_network["body"] = _safetensors_bytes(SDXL_LORA)
    assert mm.pull_model(SPEC) is True
    assert hf_network["downloads"] == 1
    entry = _entry("sdxl_lightning_4step_lora.safetensors")
    assert entry["model_type"] == "lora"


def test_an_already_downloaded_lora_safetensors_registers_as_a_lora(hf_network):
    hf_network["body"] = _safetensors_bytes(SDXL_LORA)
    (hf_network["models"] / "sdxl_lightning_4step_lora.safetensors").write_bytes(
        hf_network["body"])
    assert mm.pull_model(SPEC) is True
    assert hf_network["downloads"] == 0
    assert _entry("sdxl_lightning_4step_lora.safetensors")["model_type"] == "lora"


def test_pull_and_add_give_the_same_type_for_the_same_file(hf_network, tmp_path):
    hf_network["body"] = _safetensors_bytes(SDXL_LORA)
    assert mm.pull_model(SPEC) is True
    local = tmp_path / "elsewhere" / "sdxl_lightning_4step_lora.safetensors"
    local.parent.mkdir()
    local.write_bytes(_safetensors_bytes(SDXL_LORA))
    from localm.model_manager.registry import _media_component_type
    assert _media_component_type(local, None) == "lora"
    assert _entry("sdxl_lightning_4step_lora.safetensors")["model_type"] == "lora"


def test_a_pulled_unet_safetensors_of_the_same_repo_stays_a_diffusion_model(hf_network):
    hf_network["body"] = _safetensors_bytes(SDXL_UNET)
    assert mm.pull_model("ByteDance/SDXL-Lightning:sdxl_lightning_4step_unet.safetensors") is True
    assert _entry("sdxl_lightning_4step_unet.safetensors")["model_type"] == "diffusion-unet"


def test_an_explicit_type_is_kept(hf_network):
    hf_network["body"] = _safetensors_bytes(SDXL_LORA)
    assert mm.pull_model(SPEC, model_type="diffusion-unet") is True
    assert _entry("sdxl_lightning_4step_lora.safetensors")["model_type"] == "diffusion-unet"
