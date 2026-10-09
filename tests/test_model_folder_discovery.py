# SPDX-License-Identifier: AGPL-3.0-or-later
"""Folder discovery finds every model layout a user already has on disk.

LM Studio keeps ``<publisher>/<repo>/<file>.gguf`` (and HF safetensors repos in
the same shape), llama.cpp users keep loose or nested GGUFs. Both ``localm add
<folder>`` and the models-folder auto-sync must find GGUFs AND HuggingFace
(safetensors) model directories at any depth up to ``import_max_depth``, and a
folder import must classify each GGUF from its own header (mmproj / embedding /
llm) instead of calling everything a chat model.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.config import load_registry
from localm.model_manager import add_local
from localm.model_manager.gguf import _find_model_units

_T_STRING = 8


def _kv_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", _T_STRING) + struct.pack("<Q", len(raw)) + raw


def _write_gguf(path: Path, arch: str = "llama", extra=()) -> Path:
    """A real GGUF header (magic, version, KV block) padded past the size floor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kv = [("general.architecture", _kv_string(arch))] + list(extra)
    body = b""
    for key, encoded in kv:
        kb = key.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb + encoded
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    path.write_bytes(head + body + b"\0" * 4096)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


def _write_hf(d: Path, *, archs=("LlamaForCausalLM",), weights=("model.safetensors",),
              tokenizer=True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": list(archs)}))
    for w in weights:
        (d / w).write_bytes(b"\0" * 64)
    if tokenizer:
        (d / "tokenizer.json").write_text("{}")
    return d


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


def _lmstudio_tree(root: Path) -> None:
    _write_gguf(root / "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", "qwen3")
    _write_gguf(root / "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf", "gemma3")
    _write_gguf(root / "lmstudio-community/gemma-3-4b-it-GGUF/mmproj-model-f16.gguf", "clip")
    _write_gguf(root / "bartowski/Big-GGUF/Big-Q8_0-00001-of-00002.gguf", "llama")
    _write_gguf(root / "bartowski/Big-GGUF/Big-Q8_0-00002-of-00002.gguf", "llama")
    _write_gguf(root / "nomic-ai/nomic-embed-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-bert")
    _write_hf(root / "mlx-community/Qwen3-4B-4bit", archs=("Qwen3ForCausalLM",))
    _write_hf(root / "pub/sharded", weights=("model-00001-of-00002.safetensors",
                                              "model-00002-of-00002.safetensors"))
    _write_hf(root / "pub/no-tokenizer", tokenizer=False)


def _by_suffix(reg: dict, suffix: str) -> dict:
    hits = {n: e for n, e in reg.items()
            if str(e["path"]).replace("\\", "/").rstrip("/").endswith(suffix)}
    assert len(hits) == 1, f"{suffix!r}: expected exactly one entry, got {sorted(hits)}"
    return next(iter(hits.values()))


class TestFindModelUnits:
    def test_finds_gguf_and_hf_dirs_at_lmstudio_depth(self, tmp_path):
        _lmstudio_tree(tmp_path)
        ggufs, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert sorted(p.name for p in ggufs) == [
            "Big-Q8_0-00001-of-00002.gguf", "Qwen3-8B-Q4_K_M.gguf",
            "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-model-f16.gguf",
            "nomic-embed-text-v1.5.Q4_K_M.gguf"]
        assert sorted(p.name for p in hf_dirs) == ["Qwen3-4B-4bit", "no-tokenizer", "sharded"]

    def test_hf_dir_is_one_unit_and_its_contents_are_not_walked(self, tmp_path):
        d = _write_hf(tmp_path / "repo")
        _write_gguf(d / "extra.gguf")
        _write_hf(d / "nested-inside")
        ggufs, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert hf_dirs == [d] and ggufs == []

    def test_config_json_alone_is_not_a_model_dir(self, tmp_path):
        d = tmp_path / "half"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert _find_model_units(tmp_path, max_depth=3) == ([], [])

    def test_depth_cap_applies_to_hf_dirs(self, tmp_path):
        _write_hf(tmp_path / "a" / "found")          # folder level 3
        _write_hf(tmp_path / "a" / "b" / "toodeep")  # folder level 4
        _, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert [p.name for p in hf_dirs] == ["found"]

    def test_diffusers_pipeline_is_not_walked_into(self, tmp_path):
        pipe = tmp_path / "sdxl-base"
        pipe.mkdir()
        (pipe / "model_index.json").write_text("{}")
        for comp in ("unet", "vae", "text_encoder"):
            _write_hf(pipe / comp, weights=("diffusion_pytorch_model.safetensors",),
                      tokenizer=False)
        _write_gguf(tmp_path / "beside.gguf")
        ggufs, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert hf_dirs == []
        assert [p.name for p in ggufs] == ["beside.gguf"]

    def test_diffusers_pipeline_root_given_directly_yields_nothing(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        _write_hf(tmp_path / "unet", weights=("diffusion_pytorch_model.safetensors",))
        assert _find_model_units(tmp_path, max_depth=3) == ([], [])

    def test_skip_dirs_are_neither_walked_nor_reported(self, tmp_path):
        multi = tmp_path / "multi"
        _write_hf(multi / "part-a")
        _write_gguf(multi / "inner.gguf")
        _write_gguf(tmp_path / "outer.gguf")
        ggufs, hf_dirs = _find_model_units(
            tmp_path, max_depth=3, skip_dirs=frozenset({str(multi.resolve())}))
        assert hf_dirs == []
        assert [p.name for p in ggufs] == ["outer.gguf"]

    def test_gguf_repo_with_config_and_tokenizer_but_no_weights_yields_its_ggufs(self, tmp_path):
        d = tmp_path / "gemma-qat-gguf"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["Gemma3ForConditionalGeneration"]}))
        (d / "tokenizer.json").write_text("{}")
        _write_gguf(d / "gemma-q4_0.gguf", "gemma3")
        _write_gguf(d / "mmproj-model-f16.gguf", "clip")
        ggufs, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert hf_dirs == []
        assert [p.name for p in ggufs] == ["gemma-q4_0.gguf", "mmproj-model-f16.gguf"]

    def test_gguf_beside_real_weights_stays_one_hf_unit(self, tmp_path):
        d = _write_hf(tmp_path / "repo")
        _write_gguf(d / "extra.gguf")
        ggufs, hf_dirs = _find_model_units(tmp_path, max_depth=3)
        assert hf_dirs == [d] and ggufs == []

    def test_hidden_dirs_skipped_only_on_request(self, tmp_path):
        _write_gguf(tmp_path / ".staging" / "x.gguf")
        assert [p.name for p in _find_model_units(tmp_path, 3)[0]] == ["x.gguf"]
        assert _find_model_units(tmp_path, 3, skip_hidden=True) == ([], [])

    def test_split_set_contributes_only_its_first_part_in_any_case(self, tmp_path):
        for name in ("Big-00001-of-00002.GGUF", "Big-00002-of-00002.GGUF",
                     "small-00001-of-00002.gguf", "small-00002-of-00002.gguf"):
            _write_gguf(tmp_path / name)
        assert [p.name for p in _find_model_units(tmp_path, 1)[0]] == [
            "Big-00001-of-00002.GGUF", "small-00001-of-00002.gguf"]

    def test_uppercase_extension_is_a_gguf(self, tmp_path):
        _write_gguf(tmp_path / "Model-Q4.GGUF")
        assert [p.name for p in _find_model_units(tmp_path, 1)[0]] == ["Model-Q4.GGUF"]


class TestSplitNames:
    def test_sibling_names_keep_the_extension_case_of_the_given_part(self):
        from localm.model_manager.gguf import first_split_part, split_gguf_parts
        assert split_gguf_parts("Model-00001-of-00002.GGUF") == [
            "Model-00001-of-00002.GGUF", "Model-00002-of-00002.GGUF"]
        assert split_gguf_parts("model-00002-of-00002.gguf") == [
            "model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
        assert first_split_part("Model-00002-of-00002.GGUF") == "Model-00001-of-00002.GGUF"


class TestFolderImport:
    def test_registers_gguf_and_safetensors_dirs_under_a_parent_folder(self, tmp_path, home):
        src = tmp_path / "lmstudio-models"
        _lmstudio_tree(src)
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        reg = load_registry()
        for suffix in ("Qwen3-8B-Q4_K_M.gguf", "mlx-community/Qwen3-4B-4bit",
                       "pub/sharded", "pub/no-tokenizer"):
            _by_suffix(reg, suffix)
        assert _by_suffix(reg, "mlx-community/Qwen3-4B-4bit")["source"] == "hf"
        assert _by_suffix(reg, "mlx-community/Qwen3-4B-4bit")["model_type"] == "llm"

    def test_gguf_types_come_from_each_files_own_header(self, tmp_path, home):
        src = tmp_path / "lmstudio-models"
        _lmstudio_tree(src)
        add_local(str(src), on_duplicate="skip", no_hash=True)
        reg = load_registry()
        assert _by_suffix(reg, "mmproj-model-f16.gguf")["model_type"] == "mmproj"
        assert _by_suffix(reg, "nomic-embed-text-v1.5.Q4_K_M.gguf")["model_type"] == "embedding"
        qwen = _by_suffix(reg, "Qwen3-8B-Q4_K_M.gguf")
        assert qwen["model_type"] == "llm"
        assert qwen["architecture"] == "qwen3"

    def test_explicit_type_still_wins_over_detection(self, tmp_path, home):
        src = tmp_path / "drop"
        _write_gguf(src / "nomic-embed.gguf", "nomic-bert")
        add_local(str(src), on_duplicate="skip", no_hash=True, model_type="llm")
        assert _by_suffix(load_registry(), "nomic-embed.gguf")["model_type"] == "llm"

    def test_hf_only_folder_registers(self, tmp_path, home):
        src = tmp_path / "hf-only"
        _write_hf(src / "one")
        _write_hf(src / "two")
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        assert set(load_registry()) == {"one", "two"}

    def test_same_named_hf_dirs_in_different_parents_both_register(self, tmp_path, home):
        src = tmp_path / "pubs"
        _write_hf(src / "alice" / "model")
        _write_hf(src / "bob" / "model")
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        reg = load_registry()
        assert len(reg) == 2
        assert {Path(e["path"]).parent.name for e in reg.values()} == {"alice", "bob"}

    def test_hf_dir_depth_follows_import_max_depth(self, tmp_path, home):
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["import_max_depth"] = 2
        save_config(cfg)
        src = tmp_path / "drop"
        _write_hf(src / "shallow")                 # folder level 2
        _write_hf(src / "a" / "deep")              # folder level 3
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        assert set(load_registry()) == {"shallow"}

    def test_diffusers_pipeline_components_are_not_registered_as_models(self, tmp_path, home):
        src = tmp_path / "pubs"
        pipe = src / "sdxl-base"
        pipe.mkdir(parents=True)
        (pipe / "model_index.json").write_text("{}")
        for comp in ("unet", "vae"):
            _write_hf(pipe / comp, weights=("diffusion_pytorch_model.safetensors",),
                      tokenizer=False)
        _write_gguf(src / "chat.gguf")
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        assert set(load_registry()) == {"chat"}

    def test_gguf_repo_with_config_and_tokenizer_registers_its_ggufs_not_an_hf_entry(
            self, tmp_path, home):
        src = tmp_path / "drop"
        d = src / "gemma-qat-gguf"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "tokenizer.json").write_text("{}")
        _write_gguf(d / "gemma-q4_0.gguf", "gemma3")
        assert add_local(str(src), on_duplicate="skip", no_hash=True) is True
        entry = _by_suffix(load_registry(), "gemma-q4_0.gguf")
        assert entry["source"] == "local" and entry["model_type"] == "llm"
        assert len(load_registry()) == 1

    def test_pointing_directly_at_that_gguf_repo_registers_the_ggufs(self, tmp_path, home):
        d = tmp_path / "gemma-qat-gguf"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "tokenizer.json").write_text("{}")
        _write_gguf(d / "gemma-q4_0.gguf", "gemma3")
        assert add_local(str(d), on_duplicate="skip", no_hash=True) is True
        assert _by_suffix(load_registry(), "gemma-q4_0.gguf")["source"] == "local"

    def test_folder_with_no_models_still_refused(self, tmp_path, home):
        d = tmp_path / "junk"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "readme.txt").write_text("x")
        assert add_local(str(d)) is False
        assert load_registry() == {}

    def test_single_uppercase_gguf_file_is_recognised(self, tmp_path, home):
        f = _write_gguf(tmp_path / "Model-Q4.GGUF")
        assert add_local(str(f), on_duplicate="skip", no_hash=True) is True
        assert "Model-Q4" in load_registry()


class TestModelsFolderSync:
    def test_nested_gguf_and_hf_dirs_register(self, home):
        _lmstudio_tree(home / "models")
        result = mm.sync_models_dir(prune=False, backfill_mmproj=False)
        reg = load_registry()
        for suffix in ("Qwen3-8B-Q4_K_M.gguf", "mlx-community/Qwen3-4B-4bit",
                       "pub/sharded", "pub/no-tokenizer"):
            _by_suffix(reg, suffix)
        assert result.added == len(reg)

    def test_nested_types_from_header(self, home):
        _lmstudio_tree(home / "models")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        reg = load_registry()
        assert _by_suffix(reg, "mmproj-model-f16.gguf")["model_type"] == "mmproj"
        assert _by_suffix(reg, "nomic-embed-text-v1.5.Q4_K_M.gguf")["model_type"] == "embedding"

    def test_non_first_split_part_is_not_its_own_entry(self, home):
        _lmstudio_tree(home / "models")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        paths = [str(e["path"]) for e in load_registry().values()]
        assert not any(p.endswith("00002-of-00002.gguf") for p in paths)
        assert any(p.endswith("00001-of-00002.gguf") for p in paths)

    def test_split_set_registers_under_its_stem_like_a_folder_import(self, home):
        _lmstudio_tree(home / "models")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert "Big-Q8_0" in load_registry()

    def test_second_sync_adds_nothing(self, home):
        _lmstudio_tree(home / "models")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        before = load_registry()
        again = mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert again.added == 0
        assert load_registry() == before

    def test_diffusers_pipeline_components_are_not_registered_as_models(self, home):
        pipe = home / "models" / "sdxl-base"
        pipe.mkdir(parents=True)
        (pipe / "model_index.json").write_text("{}")
        for comp in ("unet", "vae", "text_encoder"):
            _write_hf(pipe / comp, weights=("diffusion_pytorch_model.safetensors",),
                      tokenizer=False)
        assert mm.sync_models_dir(prune=False, backfill_mmproj=False).added == 0
        assert load_registry() == {}

    def test_an_already_registered_folder_is_not_walked_into(self, home):
        multi = home / "models" / "multi"
        _write_hf(multi / "part-a")
        _write_hf(multi / "part-b")
        mm._register("multi", multi, model_type="diffusion-unet")
        assert mm.sync_models_dir(prune=False, backfill_mmproj=False).added == 0
        assert set(load_registry()) == {"multi"}

    def test_gguf_repo_with_config_and_tokenizer_registers_its_gguf(self, home):
        d = home / "models" / "gemma-qat-gguf"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "tokenizer.json").write_text("{}")
        _write_gguf(d / "gemma-q4_0.gguf", "gemma3")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert _by_suffix(load_registry(), "gemma-q4_0.gguf")["source"] == "local"

    def test_hidden_staging_dir_is_not_scanned(self, home):
        _write_gguf(home / "models" / ".pull-staging" / "half.gguf")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert load_registry() == {}

    def test_depth_one_setting_still_finds_direct_hf_dirs(self, home):
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["import_max_depth"] = 1
        save_config(cfg)
        _write_hf(home / "models" / "direct")
        _write_hf(home / "models" / "pub" / "nested")
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert set(load_registry()) == {"direct"}

    def test_mid_copy_nested_gguf_waits(self, home):
        f = _write_gguf(home / "models" / "pub" / "repo" / "fresh.gguf")
        os.utime(f, None)   # just written
        mm.sync_models_dir(prune=False, backfill_mmproj=False)
        assert load_registry() == {}
