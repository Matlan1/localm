# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm add <folder>`` registers the loose image and video model files in it
(safetensors diffusion models, VAEs, LoRAs, text encoders and zip-format ESRGAN
``.pth`` upscalers) with the same type ``localm add <file>`` gives each, and a
checkpoint is read without ever being unpickled."""

from __future__ import annotations

import io
import json
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.config import load_registry
from localm.model_manager import add_local
from localm.model_manager.ckpt_tensors import read_checkpoint_tensors
from localm.model_manager.gguf import _find_model_units, _scan_model_units

UNET = {"blocks.0.cross_attn.k.weight": (8, 8), "patch_embedding.weight": (8, 16, 1, 2, 2)}
VAE = {"decoder.conv_in.weight": (8, 4, 3, 3), "encoder.conv_in.weight": (8, 3, 3, 3)}
LORA = {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": (4, 8),
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": (8, 4)}
CLIP_L = {"text_model.embeddings.token_embedding.weight": (10, 768)}
ESRGAN = {"conv_first.weight": (4, 3, 3, 3), "conv_first.bias": (4,),
          "body.0.rdb1.conv1.weight": (4, 4, 3, 3), "conv_last.weight": (3, 4, 3, 3)}
NOT_MEDIA = {"foo": (2, 2)}
LLM_LAYERS = {"model.embed_tokens.weight": (4, 4), "lm_head.weight": (4, 4)}


def write_safetensors(path: Path, tensors: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 2
        for d in shape:
            size *= d
        header[name] = {"dtype": "F16", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


class _Storage:
    pass


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks):
    return ("tensor", tuple(size))


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"


class _TensorStandIn:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __reduce__(self):
        return (_rebuild_tensor_v2, (_Storage(), 0, self.shape, (1,) * len(self.shape), False, {}))


class _StorageLoader(pickle.Unpickler):
    def persistent_load(self, pid):
        return None


class _TorchPickler(pickle.Pickler):
    def persistent_id(self, obj):
        return ("storage", "FloatStorage", "0", "cpu", 1) if isinstance(obj, _Storage) else None


def write_pth(path: Path, tensors: dict, *, monkeypatch, wrapper: str | None = "params_ema",
              extra: dict | None = None) -> Path:
    """A zip-format checkpoint shaped like ``torch.save`` writes one: a
    ``data.pkl`` whose tensors are ``torch._utils._rebuild_tensor_v2`` calls on a
    persistent storage id."""
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch_mod._utils = utils_mod
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch._utils", utils_mod)
    state = {name: _TensorStandIn(shape) for name, shape in tensors.items()}
    obj = {wrapper: state} if wrapper else state
    obj.update(extra or {})
    buf = io.BytesIO()
    _TorchPickler(buf, protocol=2).dump(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/data/0", b"\0" * 8)
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    (h / "models").mkdir()
    return h


def _media_folder(root: Path, monkeypatch) -> dict:
    """The five recognised fixtures plus two files that are not image or video
    models, spread over the first three folder levels."""
    files = {
        "wan_unet": write_safetensors(root / "wan_unet.safetensors", UNET),
        "wan_vae": write_safetensors(root / "vae" / "wan_vae.safetensors", VAE),
        "flux_lora": write_safetensors(root / "loras" / "deep" / "flux_lora.safetensors", LORA),
        "clip_l": write_safetensors(root / "clip_l.sft", CLIP_L),
        "esrgan": write_pth(root / "upscale" / "RealESRGAN_x4plus.pth", ESRGAN,
                            monkeypatch=monkeypatch),
    }
    write_safetensors(root / "junk.safetensors", NOT_MEDIA)
    write_pth(root / "llm.pth", LLM_LAYERS, wrapper=None, monkeypatch=monkeypatch)
    return files


def _types(reg: dict) -> dict:
    return {Path(e["path"]).name: e["model_type"] for e in reg.values()}


class TestCheckpointReader:
    def test_reads_names_and_shapes_from_a_wrapped_state_dict(self, tmp_path, monkeypatch):
        pth = write_pth(tmp_path / "x.pth", ESRGAN, monkeypatch=monkeypatch)
        assert read_checkpoint_tensors(pth) == ESRGAN

    def test_reads_an_unwrapped_state_dict(self, tmp_path, monkeypatch):
        pth = write_pth(tmp_path / "x.pt", ESRGAN, wrapper=None, monkeypatch=monkeypatch)
        assert read_checkpoint_tensors(pth) == ESRGAN

    def test_a_pickle_that_would_run_code_is_never_run(self, tmp_path, monkeypatch):
        marker = tmp_path / "executed.txt"
        evil = (Path.write_text, (marker, "ran"))

        class Payload:
            def __reduce__(self):
                return evil

        pth = write_pth(tmp_path / "evil.pth", ESRGAN, extra={"payload": Payload()},
                        monkeypatch=monkeypatch)
        with zipfile.ZipFile(pth) as zf:
            blob = zf.read("archive/data.pkl")
        assert not marker.exists()
        assert read_checkpoint_tensors(pth) == ESRGAN
        assert not marker.exists()
        _StorageLoader(io.BytesIO(blob)).load()
        assert marker.read_text() == "ran"

    def test_a_pickle_global_is_never_imported(self, tmp_path, monkeypatch):
        pth = write_pth(tmp_path / "x.pth", ESRGAN, monkeypatch=monkeypatch)
        monkeypatch.delitem(sys.modules, "torch")
        monkeypatch.delitem(sys.modules, "torch._utils")
        assert read_checkpoint_tensors(pth) == ESRGAN
        assert "torch" not in sys.modules

    @pytest.mark.parametrize("content", [b"", b"not a zip", b"PK\x03\x04truncated"])
    def test_a_file_that_is_not_a_checkpoint_reads_as_none(self, tmp_path, content):
        bad = tmp_path / "bad.pth"
        bad.write_bytes(content)
        assert read_checkpoint_tensors(bad) is None

    def test_a_zip_without_a_pickle_reads_as_none(self, tmp_path):
        bad = tmp_path / "bad.pth"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("archive/other.txt", "x")
        assert read_checkpoint_tensors(bad) is None

    def test_a_pickle_with_an_unmodelled_opcode_reads_as_none(self, tmp_path):
        bad = tmp_path / "bad.pth"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("archive/data.pkl", b"\x80\x02\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
        assert read_checkpoint_tensors(bad) is None

    def test_other_suffixes_are_not_read(self, tmp_path, monkeypatch):
        pth = write_pth(tmp_path / "x.pth", ESRGAN, monkeypatch=monkeypatch)
        renamed = pth.rename(tmp_path / "x.bin")
        assert read_checkpoint_tensors(renamed) is None


class TestScan:
    def test_loose_media_files_are_listed_by_the_same_walk(self, tmp_path, monkeypatch):
        _media_folder(tmp_path, monkeypatch)
        ggufs, hf_dirs, media = _scan_model_units(tmp_path, max_depth=3)
        assert (ggufs, hf_dirs) == ([], [])
        assert sorted(p.relative_to(tmp_path).as_posix() for p in media) == [
            "clip_l.sft", "junk.safetensors", "llm.pth", "loras/deep/flux_lora.safetensors",
            "upscale/RealESRGAN_x4plus.pth", "vae/wan_vae.safetensors", "wan_unet.safetensors"]
        assert _find_model_units(tmp_path, max_depth=3) == ([], [])

    def test_the_walk_stops_at_the_depth_limit(self, tmp_path):
        write_safetensors(tmp_path / "a" / "b" / "ok.safetensors", VAE)
        write_safetensors(tmp_path / "a" / "b" / "c" / "too_deep.safetensors", VAE)
        _g, _h, media = _scan_model_units(tmp_path, max_depth=3)
        assert [p.name for p in media] == ["ok.safetensors"]

    def test_files_of_a_model_directory_are_not_loose(self, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        (hf / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        (hf / "tokenizer.json").write_text("{}")
        write_safetensors(hf / "model.safetensors", NOT_MEDIA)
        pipe = tmp_path / "pipe"
        write_safetensors(pipe / "unet" / "diffusion_pytorch_model.safetensors", UNET)
        (pipe / "model_index.json").write_text("{}")
        _g, hf_dirs, media = _scan_model_units(tmp_path, max_depth=3)
        assert hf_dirs == [hf]
        assert media == []


class TestFolderAdd:
    def test_a_folder_add_registers_every_media_file_with_its_type(
            self, tmp_path, home, monkeypatch):
        _media_folder(tmp_path / "models_in", monkeypatch)
        assert add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        assert _types(load_registry()) == {
            "wan_unet.safetensors": "diffusion-unet", "wan_vae.safetensors": "vae",
            "flux_lora.safetensors": "lora", "clip_l.sft": "text-encoder",
            "RealESRGAN_x4plus.pth": "upscaler"}

    def test_a_folder_add_and_a_file_add_give_the_same_type(self, tmp_path, home, monkeypatch):
        files = _media_folder(tmp_path / "models_in", monkeypatch)
        for name, path in files.items():
            assert add_local(str(path), None, on_duplicate="skip", fast=True), name
        by_file = _types(load_registry())
        assert len(by_file) == len(files)
        mm.save_registry({})
        assert add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        assert _types(load_registry()) == by_file

    def test_the_registered_path_is_the_original_file(self, tmp_path, home, monkeypatch):
        files = _media_folder(tmp_path / "models_in", monkeypatch)
        add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        registered = {Path(e["path"]).resolve() for e in load_registry().values()}
        assert registered == {p.resolve() for p in files.values()}
        assert all(p.exists() for p in files.values())

    def test_adding_the_folder_again_adds_nothing(self, tmp_path, home, monkeypatch):
        _media_folder(tmp_path / "models_in", monkeypatch)
        add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        first = dict(load_registry())
        add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        assert load_registry() == first

    def test_files_that_are_not_media_models_are_reported_not_registered(
            self, tmp_path, home, monkeypatch, capsys):
        _media_folder(tmp_path / "models_in", monkeypatch)
        texts = []
        monkeypatch.setattr(mm, "_import_status", lambda text, **kw: texts.append(text))
        add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True)
        out = capsys.readouterr().out
        assert "junk.safetensors" in out and "llm.pth" in out
        assert "5 registered" in texts[-1] and "2 not usable" in texts[-1]
        assert any("5 image or video model file(s)" in t for t in texts)

    def test_a_folder_of_only_unrecognised_files_is_refused(self, tmp_path, home, monkeypatch):
        write_safetensors(tmp_path / "in" / "junk.safetensors", NOT_MEDIA)
        write_pth(tmp_path / "in" / "llm.pth", LLM_LAYERS, wrapper=None, monkeypatch=monkeypatch)
        assert add_local(str(tmp_path / "in"), None, on_duplicate="skip", fast=True) is False
        assert load_registry() == {}

    def test_a_single_media_file_folder_takes_the_given_name(self, tmp_path, home):
        write_safetensors(tmp_path / "in" / "wan_vae.safetensors", VAE)
        assert add_local(str(tmp_path / "in"), "my-vae", on_duplicate="skip", fast=True)
        assert list(load_registry()) == ["my-vae"]

    def test_a_non_media_type_leaves_media_files_alone(self, tmp_path, home, monkeypatch):
        _media_folder(tmp_path / "models_in", monkeypatch)
        add_local(str(tmp_path / "models_in"), None, on_duplicate="skip", fast=True,
                  model_type="llm")
        assert load_registry() == {}

    def test_media_files_and_ggufs_in_one_folder_both_register(self, tmp_path, home):
        from tests.test_model_folder_discovery import _write_gguf
        _write_gguf(tmp_path / "in" / "chat.gguf", "llama")
        write_safetensors(tmp_path / "in" / "wan_vae.safetensors", VAE)
        assert add_local(str(tmp_path / "in"), None, on_duplicate="skip", fast=True)
        assert _types(load_registry()) == {"chat.gguf": "llm", "wan_vae.safetensors": "vae"}


class TestCheckpointFileAdd:
    def test_an_esrgan_checkpoint_registers_as_an_upscaler(self, tmp_path, home, monkeypatch):
        pth = write_pth(tmp_path / "RealESRGAN_x4plus.pth", ESRGAN, monkeypatch=monkeypatch)
        assert add_local(str(pth), None, on_duplicate="skip", fast=True)
        assert _types(load_registry()) == {"RealESRGAN_x4plus.pth": "upscaler"}

    def test_any_other_checkpoint_is_still_refused(self, tmp_path, home, monkeypatch, capsys):
        pth = write_pth(tmp_path / "llm.pth", LLM_LAYERS, wrapper=None, monkeypatch=monkeypatch)
        assert add_local(str(pth), None, on_duplicate="skip", fast=True) is False
        assert "Not a model" in capsys.readouterr().out
        assert load_registry() == {}

    def test_an_unreadable_checkpoint_is_refused(self, tmp_path, home, capsys):
        bad = tmp_path / "bad.pth"
        bad.write_bytes(b"not a zip")
        assert add_local(str(bad), None, on_duplicate="skip", fast=True) is False
        assert "Not a model" in capsys.readouterr().out

    def test_a_checkpoint_never_takes_another_media_type(self, tmp_path, home, monkeypatch):
        pth = write_pth(tmp_path / "RealESRGAN_x4plus.pth", ESRGAN, monkeypatch=monkeypatch)
        assert add_local(str(pth), None, on_duplicate="skip", fast=True, model_type="vae") is False
        assert load_registry() == {}
