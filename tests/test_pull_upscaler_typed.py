# SPDX-License-Identifier: AGPL-3.0-or-later
"""An ESRGAN upscaler pulled from Hugging Face is typed 'upscaler' from its
tensor layout, the way 'localm add --type upscaler' types it."""

import json
import struct
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.media.sdcpp.families import classify_tensors
from localm.model_manager.scan import _type_for_file

RRDB = {
    "conv_first.weight": (64, 3, 3, 3), "conv_first.bias": (64,),
    "body.0.rdb1.conv1.weight": (32, 64, 3, 3), "body.0.rdb1.conv1.bias": (32,),
    "body.22.rdb3.conv5.weight": (64, 192, 3, 3),
    "conv_body.weight": (64, 64, 3, 3), "conv_up1.weight": (64, 64, 3, 3),
    "conv_up2.weight": (64, 64, 3, 3), "conv_hr.weight": (64, 64, 3, 3),
    "conv_last.weight": (3, 64, 3, 3), "conv_last.bias": (3,),
}


def _safetensors(path: Path, shapes: dict) -> Path:
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 4
        for d in shape:
            size *= d
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return path


def test_the_rrdbnet_layout_is_an_upscaler():
    info = classify_tensors(RRDB)
    assert info.kind == "upscaler"


@pytest.mark.parametrize("change", [
    lambda t: t.pop("conv_last.weight"),
    lambda t: t.update({"conv_last.weight": (4, 64, 3, 3)}),
    lambda t: t.pop("conv_first.weight"),
    lambda t: [t.pop(k) for k in [k for k in t if k.startswith("body.")]],
])
def test_a_layout_missing_a_part_of_rrdbnet_is_not_an_upscaler(change):
    tensors = dict(RRDB)
    change(tensors)
    assert classify_tensors(tensors).kind != "upscaler"


def test_a_scanned_upscaler_file_takes_the_upscaler_type(tmp_path):
    path = _safetensors(tmp_path / "RealESRGAN_x4plus.safetensors", RRDB)
    assert _type_for_file(path, "unknown", {}) == "upscaler"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")
    return home


def _pull(monkeypatch, shapes, spec):
    import huggingface_hub

    def fake_download(repo_id, filename, local_dir, **kw):
        return str(_safetensors(Path(local_dir) / filename, shapes))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
    monkeypatch.setattr(
        "localm.model_manager.pull._policy_head",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))
    monkeypatch.setattr("localm.model_manager.pull._hf_endpoint_refusal", lambda: None)
    monkeypatch.setattr("localm.model_manager.pull._hf_pipeline_tag_to_type",
                        lambda repo_id: "unknown")
    return mm.pull_model(spec)


def test_pulling_an_esrgan_file_registers_an_upscaler(isolated_home, monkeypatch):
    assert _pull(monkeypatch, RRDB,
                 "owner/Real-ESRGAN_repackaged:RealESRGAN_x4plus.safetensors") is True
    entry = mm.load_registry()["RealESRGAN_x4plus.safetensors"]
    assert entry["model_type"] == "upscaler"


def test_pulling_a_file_with_no_known_layout_still_registers_unknown(isolated_home, monkeypatch):
    assert _pull(monkeypatch, {"something.weight": (2, 2)},
                 "owner/repo:odd.safetensors") is True
    assert mm.load_registry()["odd.safetensors"]["model_type"] == "unknown"


def test_pulling_again_retypes_an_entry_registered_unknown(isolated_home, monkeypatch, capsys):
    spec = "owner/Real-ESRGAN_repackaged:RealESRGAN_x4plus.safetensors"
    assert _pull(monkeypatch, {"something.weight": (2, 2)}, spec) is True
    registry = mm.load_registry()
    entry = registry["RealESRGAN_x4plus.safetensors"]
    assert entry["model_type"] == "unknown"
    _safetensors(Path(entry["path"]), RRDB)
    capsys.readouterr()

    assert mm.pull_model(spec) is True
    assert mm.load_registry()["RealESRGAN_x4plus.safetensors"]["model_type"] == "upscaler"
    assert "Typed RealESRGAN_x4plus.safetensors as upscaler" in capsys.readouterr().out


def test_pulling_again_never_changes_a_type_that_is_already_set(isolated_home, monkeypatch):
    spec = "owner/Real-ESRGAN_repackaged:RealESRGAN_x4plus.safetensors"
    assert _pull(monkeypatch, RRDB, spec) is True

    def set_vae(reg):
        reg["RealESRGAN_x4plus.safetensors"]["model_type"] = "vae"

    mm.update_registry(set_vae)
    assert mm.pull_model(spec) is True
    assert mm.load_registry()["RealESRGAN_x4plus.safetensors"]["model_type"] == "vae"


ORIGINAL_ESRGAN = {
    "model.0.weight": (64, 3, 3, 3), "model.0.bias": (64,),
    "model.1.sub.0.RDB1.conv1.0.weight": (32, 64, 3, 3),
    "model.1.sub.0.RDB1.conv1.0.bias": (32,),
    "model.1.sub.22.RDB3.conv5.0.weight": (64, 160, 3, 3),
    "model.1.sub.23.weight": (64, 64, 3, 3),
    "model.3.weight": (64, 64, 3, 3), "model.6.weight": (64, 64, 3, 3),
    "model.8.weight": (64, 64, 3, 3),
    "model.10.weight": (3, 64, 3, 3), "model.10.bias": (3,),
}

SRVGG_COMPACT = {
    "body.0.weight": (64, 3, 3, 3), "body.0.bias": (64,),
    "body.1.weight": (64,),
    "body.2.weight": (64, 64, 3, 3), "body.2.bias": (64,),
    "body.3.weight": (64,),
    "body.4.weight": (48, 64, 3, 3), "body.4.bias": (48,),
}


def test_the_original_esrgan_layout_is_an_upscaler():
    assert classify_tensors(ORIGINAL_ESRGAN).kind == "upscaler"


def test_an_srvgg_compact_layout_is_an_upscaler():
    assert classify_tensors(SRVGG_COMPACT).kind == "upscaler"


def test_an_srvgg_compact_layout_without_activation_weights_is_an_upscaler():
    tensors = {k: v for k, v in SRVGG_COMPACT.items() if k not in ("body.1.weight", "body.3.weight")}
    assert classify_tensors(tensors).kind == "upscaler"


@pytest.mark.parametrize("change", [
    lambda t: t.pop("model.0.weight"),
    lambda t: t.pop("model.1.sub.0.RDB1.conv1.0.weight"),
    lambda t: t.update({"model.10.weight": (4, 64, 3, 3)}),
    lambda t: t.update({"model.10.weight": (3, 64)}),
])
def test_an_original_esrgan_layout_missing_a_part_is_not_an_upscaler(change):
    tensors = dict(ORIGINAL_ESRGAN)
    change(tensors)
    assert classify_tensors(tensors).kind != "upscaler"


@pytest.mark.parametrize("change", [
    lambda t: t.update({"head.weight": (3, 3, 3, 3)}),
    lambda t: t.update({"body.5.weight": (64, 64, 3, 3), "body.6.weight": (48, 64, 3, 3)}),
    lambda t: t.update({"body.4.weight": (50, 64, 3, 3)}),
    lambda t: t.update({"body.0.weight": (64, 4, 3, 3)}),
    lambda t: [t.pop(k) for k in ("body.2.weight", "body.2.bias")],
])
def test_a_layout_that_is_not_srvgg_compact_is_not_an_upscaler(change):
    tensors = dict(SRVGG_COMPACT)
    change(tensors)
    assert classify_tensors(tensors).kind != "upscaler"


@pytest.mark.parametrize("shapes", [ORIGINAL_ESRGAN, SRVGG_COMPACT])
def test_a_scanned_file_in_another_upscaler_layout_takes_the_upscaler_type(tmp_path, shapes):
    path = _safetensors(tmp_path / "4x-UltraSharp.safetensors", shapes)
    assert _type_for_file(path, "unknown", {}) == "upscaler"
