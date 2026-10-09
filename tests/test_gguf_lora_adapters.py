# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF LoRA adapters: header classification, registry association, the
architecture compatibility check, the native binding, and the load path that
applies an attached adapter to its base model.

The native llama layer is faked where a real model is impractical (unit tests
below); tests/test_gguf_lora_adapters_integration.py drives real files.
"""

from __future__ import annotations

import ctypes
import os
import queue
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import localm.model_manager as mm
from localm.inference.backends.base import AdapterLoadError, UnsupportedModelRoleError
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _api as api
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp
from localm.model_manager import gguf as _gguf_mod
from tests._bare_llama import make_bare_llama

# --------------------------------------------------------------------------- #
#  Fixtures and GGUF byte builders                                             #
# --------------------------------------------------------------------------- #


def _kv(key: str, value) -> bytes:
    kb = key.encode("utf-8")
    head = struct.pack("<Q", len(kb)) + kb
    if isinstance(value, float):
        return head + struct.pack("<I", 6) + struct.pack("<f", value)
    vb = str(value).encode("utf-8")
    return head + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _gguf_bytes(arch="llama", *, general_type=None, adapter_type=None, alpha=None,
                extra=()) -> bytes:
    """A structurally valid GGUF header padded to the size floor, declaring the
    keys a llama.cpp LoRA converter writes when asked to."""
    kvs = [("general.architecture", arch)]
    if general_type is not None:
        kvs.append(("general.type", general_type))
    if adapter_type is not None:
        kvs.append(("adapter.type", adapter_type))
    if alpha is not None:
        kvs.append(("adapter.lora.alpha", float(alpha)))
    kvs.extend(extra)
    buf = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
           + struct.pack("<Q", len(kvs)))
    for key, value in kvs:
        buf += _kv(key, value)
    return buf.ljust(max(len(buf), _gguf_mod._GGUF_MIN_BYTES), b"\0")


def _lora_bytes(arch="llama", alpha=16.0) -> bytes:
    return _gguf_bytes(arch, general_type="adapter", adapter_type="lora", alpha=alpha)


def _model_bytes(arch="llama") -> bytes:
    return _gguf_bytes(arch, general_type="model")


def _write(path: Path, data: bytes, age: float = 3600.0) -> Path:
    """Write *data* followed by the file's own name, so two files built from the
    same header never have identical bytes (the registry dedups on content)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + b"\0" + path.name.encode("utf-8"))
    old = time.time() - age
    os.utime(path, (old, old))
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def pair(home, tmp_path):
    """A registered chat model "base" and a registered LoRA adapter "adp", both
    llama architecture, files outside the models dir."""
    base = _write(tmp_path / "files" / "base.gguf", _model_bytes("llama"))
    adapter = _write(tmp_path / "files" / "adp.gguf", _lora_bytes("llama"))
    assert mm.add_local(str(base), name="base") is True
    assert mm.add_local(str(adapter), name="adp") is True
    return SimpleNamespace(base=base, adapter=adapter, home=home, tmp=tmp_path)


# --------------------------------------------------------------------------- #
#  Header probe and classification                                             #
# --------------------------------------------------------------------------- #


class TestHeaderClassification:
    def test_probe_reports_general_type_and_adapter_type(self, tmp_path):
        f = _write(tmp_path / "a.gguf", _lora_bytes("qwen3"))
        meta = _gguf_mod._gguf_metadata_probe(f)
        assert meta["general_type"] == "adapter"
        assert meta["adapter_type"] == "lora"
        assert meta["architecture"] == "qwen3"

    def test_probe_reports_none_for_keys_the_file_does_not_declare(self, tmp_path):
        f = _write(tmp_path / "m.gguf", _gguf_bytes("llama"))
        meta = _gguf_mod._gguf_metadata_probe(f)
        assert meta["general_type"] is None
        assert meta["adapter_type"] is None

    @pytest.mark.parametrize("data,expected", [
        (_gguf_bytes("llama", general_type="adapter", adapter_type="lora"), "lora"),
        (_gguf_bytes("llama", general_type="adapter"), "unknown"),
        (_gguf_bytes("llama", general_type="adapter", adapter_type="control_vector"),
         "control_vector"),
        (_gguf_bytes("llama", general_type="model"), None),
        (_gguf_bytes("llama"), None),
        (_gguf_bytes("llama", adapter_type="lora"), None),
    ])
    def test_adapter_kind_comes_from_general_type(self, tmp_path, data, expected):
        f = _write(tmp_path / "x.gguf", data)
        assert _gguf_mod.gguf_adapter_kind(f) == expected

    def test_adapter_kind_of_a_non_gguf_file_is_none(self, tmp_path):
        f = tmp_path / "notes.gguf"
        f.write_bytes(b"not a gguf at all" * 100)
        assert _gguf_mod.gguf_adapter_kind(f) is None

    @pytest.mark.parametrize("arch", ["llama", "bert", "clip", "t5"])
    def test_adapter_signal_wins_over_the_architecture_signals(self, tmp_path, arch):
        from localm.model_manager.registry import _detect_local_model_type
        f = _write(tmp_path / "a.gguf", _lora_bytes(arch))
        mtype, meta = _detect_local_model_type(f, is_gguf=True, is_hf=False)
        assert mtype == "lora"
        assert meta["architecture"] == arch

    def test_an_embedding_model_is_still_an_embedding(self, tmp_path):
        from localm.model_manager.registry import _detect_local_model_type
        f = _write(tmp_path / "e.gguf", _model_bytes("bert"))
        assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "embedding"

    def test_a_plain_model_stays_llm(self, tmp_path):
        from localm.model_manager.registry import _detect_local_model_type
        f = _write(tmp_path / "m.gguf", _model_bytes("llama"))
        assert _detect_local_model_type(f, is_gguf=True, is_hf=False)[0] == "llm"

    def test_add_local_registers_an_adapter_as_lora(self, home, tmp_path):
        f = _write(tmp_path / "my-lora.gguf", _lora_bytes())
        assert mm.add_local(str(f)) is True
        entry = mm.load_registry()["my-lora"]
        assert entry["model_type"] == "lora"
        assert mm.is_gguf_adapter_entry(entry) is True
        assert mm.is_llm(entry) is False
        assert mm.is_auto_chat_eligible(entry) is False

    def test_folder_import_types_each_gguf_from_its_own_header(self, home, tmp_path):
        folder = tmp_path / "pack"
        _write(folder / "chat.gguf", _model_bytes("llama"))
        _write(folder / "tuned.gguf", _lora_bytes("llama"))
        assert mm.add_local(str(folder)) is True
        reg = mm.load_registry()
        assert reg["chat"]["model_type"] == "llm"
        assert reg["tuned"]["model_type"] == "lora"

    def test_models_dir_sync_types_an_adapter_as_lora(self, home):
        _write(home / "models" / "tuned.gguf", _lora_bytes())
        result = mm.sync_models_dir()
        assert result.added == 1
        assert mm.load_registry()["tuned"]["model_type"] == "lora"

    def test_pull_registers_an_adapter_as_lora(self, home, monkeypatch):
        import huggingface_hub
        import requests
        data = _lora_bytes()

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(data)
            return str(p)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
        monkeypatch.setattr(
            requests, "head",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))
        assert mm.pull_model("owner/repo:tuned-f16.gguf") is True
        assert mm.load_registry()["tuned-f16"]["model_type"] == "lora"

    def test_pull_with_an_explicit_type_is_not_overridden(self, home, monkeypatch):
        import huggingface_hub
        import requests
        data = _lora_bytes()

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(data)
            return str(p)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
        monkeypatch.setattr(
            requests, "head",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))
        assert mm.pull_model("owner/repo:forced.gguf", model_type="llm") is True
        assert mm.load_registry()["forced"]["model_type"] == "llm"

    def test_entry_predicate(self):
        assert mm.is_gguf_adapter_entry({"model_type": "lora", "path": "/x/a.GGUF"}) is True
        assert mm.is_gguf_adapter_entry({"model_type": "lora", "path": "/x/a.safetensors"}) is False
        assert mm.is_gguf_adapter_entry({"model_type": "llm", "path": "/x/a.gguf"}) is False
        assert mm.is_gguf_adapter_entry({"model_type": "lora"}) is False
        assert mm.is_gguf_adapter_entry("not a dict") is False

    def test_a_comfyui_lora_stays_auto_chat_eligible_as_before(self):
        assert mm.is_auto_chat_eligible({"model_type": "lora"}) is True
        assert mm.is_auto_chat_eligible(
            {"model_type": "lora", "path": "/x/a.safetensors"}) is True

    def test_a_gguf_adapter_is_not_offered_as_an_image_lora(self):
        from localm.plugins.media_roles import registry_models_of_type
        reg = {
            "text-lora": {"model_type": "lora", "path": "/m/text.gguf"},
            "img-lora": {"model_type": "lora", "path": "/m/style.safetensors"},
        }
        assert [m["name"] for m in registry_models_of_type("lora", reg)] == ["img-lora"]


# --------------------------------------------------------------------------- #
#  Compatibility                                                               #
# --------------------------------------------------------------------------- #


class TestCompatibility:
    def test_matching_architectures_are_compatible(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("qwen3"))
        b = _write(tmp_path / "b.gguf", _model_bytes("qwen3"))
        assert _gguf_mod.gguf_adapter_incompatibility(a, b) is None

    def test_a_mismatch_names_both_architectures_and_both_files(self, tmp_path):
        a = _write(tmp_path / "tuned.gguf", _lora_bytes("llama"))
        b = _write(tmp_path / "chat.gguf", _model_bytes("qwen3"))
        reason = _gguf_mod.gguf_adapter_incompatibility(a, b)
        assert "'llama'" in reason and "'qwen3'" in reason
        assert "tuned.gguf" in reason and "chat.gguf" in reason

    def test_an_unreadable_base_architecture_is_not_a_refusal(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("llama"))
        no_arch = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                   + struct.pack("<Q", 0)).ljust(4096, b"\0")
        b = _write(tmp_path / "b.gguf", no_arch)
        assert _gguf_mod.gguf_adapter_incompatibility(a, b) is None

    def test_a_base_without_the_gguf_magic_is_refused(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("llama"))
        b = tmp_path / "b.gguf"
        b.write_bytes(b"\0" * 4096)
        assert "not a GGUF file" in _gguf_mod.gguf_adapter_incompatibility(a, b)

    def test_a_file_that_is_not_an_adapter_is_refused(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _model_bytes("llama"))
        b = _write(tmp_path / "b.gguf", _model_bytes("llama"))
        assert "not a GGUF adapter" in _gguf_mod.gguf_adapter_incompatibility(a, b)

    def test_a_non_lora_adapter_is_refused(self, tmp_path):
        a = _write(tmp_path / "a.gguf",
                   _gguf_bytes("llama", general_type="adapter", adapter_type="control_vector"))
        b = _write(tmp_path / "b.gguf", _model_bytes("llama"))
        assert "control_vector" in _gguf_mod.gguf_adapter_incompatibility(a, b)

    def test_a_base_that_is_not_a_gguf_file_is_refused(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("llama"))
        assert "GGUF base models only" in _gguf_mod.gguf_adapter_incompatibility(
            a, tmp_path / "hf-model-dir")
        folder = tmp_path / "a-folder"
        folder.mkdir()
        assert "GGUF base models only" in _gguf_mod.gguf_adapter_incompatibility(a, folder)

    def test_a_gguf_base_without_an_extension_is_accepted(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("llama"))
        blob = _write(tmp_path / "sha256-0123456789abcdef", _model_bytes("llama"))
        assert _gguf_mod.gguf_adapter_incompatibility(a, blob) is None

    def test_loading_an_adapter_file_alone_is_refused_before_any_vram_probe(self, tmp_path):
        a = _write(tmp_path / "a.gguf", _lora_bytes("llama"))
        backend = GgufBackend(str(a), n_ctx=512)
        with patch.object(GgufBackend, "_check_vram") as vram, \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load") as spawn:
            with pytest.raises(UnsupportedModelRoleError) as ei:
                backend.load()
        message = str(ei.value)
        assert "adapter" in message and "localm adapter attach" in message
        vram.assert_not_called()
        spawn.assert_not_called()


# --------------------------------------------------------------------------- #
#  Registry association                                                        #
# --------------------------------------------------------------------------- #


class TestAssociation:
    def test_attach_records_base_and_scale(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        entry = mm.load_registry()["adp"]
        assert entry["base"] == "base"
        assert entry["scale"] == 0.5
        assert mm.get_model_adapters("base") == [(str(pair.adapter.resolve()), 0.5)]
        assert mm.get_model_adapters("adp") == []

    def test_detach_stops_applying_it_and_keeps_it_registered(self, pair):
        mm.attach_adapter("adp", "base")
        assert mm.detach_adapter("adp") is True
        assert mm.get_model_adapters("base") == []
        entry = mm.load_registry()["adp"]
        assert entry["model_type"] == "lora" and "base" not in entry and "scale" not in entry
        assert mm.detach_adapter("adp") is False
        assert mm.detach_adapter("no-such-model") is False

    def test_attaching_again_moves_the_adapter_to_the_new_base(self, pair):
        other = _write(pair.tmp / "files" / "other.gguf", _model_bytes("llama"))
        assert mm.add_local(str(other), name="other") is True
        mm.attach_adapter("adp", "base", 1.0)
        mm.attach_adapter("adp", "other", 2.0)
        assert mm.get_model_adapters("base") == []
        assert mm.get_model_adapters("other") == [(str(pair.adapter.resolve()), 2.0)]

    def test_adapters_of_one_base_come_back_in_name_order(self, pair):
        second = _write(pair.tmp / "files" / "aaa.gguf", _lora_bytes("llama"))
        assert mm.add_local(str(second), name="aaa") is True
        mm.attach_adapter("adp", "base", 0.25)
        mm.attach_adapter("aaa", "base", 0.75)
        assert [s for _p, s in mm.get_model_adapters("base")] == [0.75, 0.25]

    def test_an_architecture_mismatch_is_refused_naming_both_and_changes_nothing(
            self, pair):
        qwen = _write(pair.tmp / "files" / "qwen.gguf", _model_bytes("qwen3"))
        assert mm.add_local(str(qwen), name="qwen") is True
        before = mm.load_registry()
        with pytest.raises(mm.AdapterError) as ei:
            mm.attach_adapter("adp", "qwen")
        message = str(ei.value)
        assert "'llama'" in message and "'qwen3'" in message
        assert "adp" in message and "qwen" in message
        assert mm.load_registry() == before

    def test_a_file_that_is_not_an_adapter_cannot_be_attached(self, pair):
        plain = _write(pair.tmp / "files" / "plain.gguf", _model_bytes("llama"))
        assert mm.add_local(str(plain), name="plain") is True
        with pytest.raises(mm.AdapterError, match="not a GGUF adapter"):
            mm.attach_adapter("plain", "base")

    def test_unknown_names_are_refused(self, pair):
        with pytest.raises(mm.AdapterError, match="'nope' is not a registered model"):
            mm.attach_adapter("nope", "base")
        with pytest.raises(mm.AdapterError, match="'nope' is not a registered model"):
            mm.attach_adapter("adp", "nope")

    def test_an_adapter_cannot_be_attached_to_itself_or_to_an_adapter(self, pair):
        with pytest.raises(mm.AdapterError, match="itself"):
            mm.attach_adapter("adp", "adp")
        second = _write(pair.tmp / "files" / "second.gguf", _lora_bytes("llama"))
        assert mm.add_local(str(second), name="second") is True
        with pytest.raises(mm.AdapterError, match="not a chat model"):
            mm.attach_adapter("adp", "second")

    def test_a_base_that_is_a_huggingface_folder_is_refused(self, pair):
        import json
        d = pair.tmp / "files" / "hf"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        (d / "tokenizer.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\0" * 64)
        assert mm.add_local(str(d), name="hf") is True
        with pytest.raises(mm.AdapterError, match="GGUF base models only"):
            mm.attach_adapter("adp", "hf")

    @pytest.mark.parametrize("scale", [0, 0.0, float("nan"), float("inf"), -float("inf"), "x", None])
    def test_an_unusable_scale_is_refused(self, pair, scale):
        with pytest.raises(mm.AdapterError, match="invalid scale"):
            mm.attach_adapter("adp", "base", scale)
        assert "base" not in mm.load_registry()["adp"]

    def test_a_negative_scale_is_allowed(self, pair):
        mm.attach_adapter("adp", "base", -0.5)
        assert mm.get_model_adapters("base")[0][1] == -0.5

    def test_an_adapter_registered_before_adapters_were_recognised_is_retyped(self, pair):
        legacy = _write(pair.tmp / "files" / "legacy.gguf", _lora_bytes("llama"))
        assert mm.add_local(str(legacy), name="legacy", model_type="llm") is True
        assert mm.load_registry()["legacy"]["model_type"] == "llm"
        mm.attach_adapter("legacy", "base")
        assert mm.load_registry()["legacy"]["model_type"] == "lora"
        assert len(mm.get_model_adapters("base")) == 1

    def test_a_missing_adapter_file_is_refused(self, pair):
        pair.adapter.unlink()
        with pytest.raises(mm.AdapterError, match="missing"):
            mm.attach_adapter("adp", "base")

    def test_an_alias_of_an_attached_adapter_is_not_a_second_attachment(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        assert mm.alias_model("adp", "adp-copy") is True
        copy = mm.load_registry()["adp-copy"]
        assert "base" not in copy and "scale" not in copy
        assert len(mm.get_model_adapters("base")) == 1
        assert mm.detach_adapter("adp") is True
        assert mm.get_model_adapters("base") == []

    def test_a_hand_copied_entry_for_the_same_adapter_file_counts_once(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        mm.update_registry(lambda r: r.__setitem__("adp-dup", dict(r["adp"])))
        assert len(mm.get_model_adapters("base")) == 1

    def test_an_alias_of_the_base_gets_the_same_adapters(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        assert mm.alias_model("base", "daily") is True
        assert mm.get_model_adapters("daily") == mm.get_model_adapters("base")
        assert len(mm.get_model_adapters("daily")) == 1

    def test_an_adapter_for_another_file_is_not_applied_through_a_name_collision(self, pair):
        other = _write(pair.tmp / "files" / "other.gguf", _model_bytes("llama"))
        assert mm.add_local(str(other), name="other") is True
        mm.attach_adapter("adp", "base", 0.5)
        assert mm.get_model_adapters("other") == []

    def test_pruning_a_missing_base_detaches_its_adapters(self, home, tmp_path):
        base = _write(home / "models" / "owned.gguf", _model_bytes("llama"))
        adapter = _write(tmp_path / "files" / "adp.gguf", _lora_bytes("llama"))
        assert mm.add_local(str(base), name="owned") is True
        assert mm.add_local(str(adapter), name="adp") is True
        mm.attach_adapter("adp", "owned", 0.5)
        base.unlink()
        result = mm.sync_models_dir(prune=True)
        assert result.pruned == 1
        assert "owned" not in mm.load_registry()
        assert "base" not in mm.load_registry()["adp"]

    def test_renaming_the_base_keeps_its_adapters_attached(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        ok, _notes = mm.rename_model_with_notes("base", "base-renamed")
        assert ok is True
        assert mm.get_model_adapters("base") == []
        assert len(mm.get_model_adapters("base-renamed")) == 1
        assert mm.load_registry()["adp"]["base"] == "base-renamed"

    def test_removing_the_base_detaches_its_adapters(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        mm.remove_model("base")
        entry = mm.load_registry()["adp"]
        assert "base" not in entry and "scale" not in entry
        assert entry["model_type"] == "lora"

    def test_a_hand_edited_invalid_scale_is_an_error_not_a_default(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        mm.update_registry(lambda r: r["adp"].__setitem__("scale", "oops"))
        with pytest.raises(mm.AdapterError, match="invalid scale"):
            mm.get_model_adapters("base")

    def test_list_adapters(self, pair):
        assert mm.list_adapters() == [
            {"name": "adp", "path": str(pair.adapter.resolve()), "base": None, "scale": None}]
        mm.attach_adapter("adp", "base", 0.5)
        assert mm.list_adapters()[0]["base"] == "base"
        assert mm.list_adapters()[0]["scale"] == 0.5


class TestCli:
    def _invoke(self, *args):
        import localm.cli as cli_pkg
        return CliRunner().invoke(cli_pkg.main, ["adapter", *args])

    def test_attach_detach_list(self, pair):
        res = self._invoke("attach", "adp", "base", "--scale", "0.5")
        assert res.exit_code == 0, res.output
        assert mm.get_model_adapters("base")[0][1] == 0.5
        listed = self._invoke("list")
        assert listed.exit_code == 0 and "adp" in listed.output and "base" in listed.output
        res = self._invoke("detach", "adp")
        assert res.exit_code == 0, res.output
        assert mm.get_model_adapters("base") == []
        assert self._invoke("detach", "adp").exit_code == 1

    def test_attach_refusal_exits_non_zero_and_names_both_architectures(self, pair):
        qwen = _write(pair.tmp / "files" / "qwen.gguf", _model_bytes("qwen3"))
        assert mm.add_local(str(qwen), name="qwen") is True
        res = self._invoke("attach", "adp", "qwen")
        assert res.exit_code == 1
        assert "llama" in res.output and "qwen3" in res.output
        assert mm.get_model_adapters("qwen") == []

    def test_list_with_no_adapters(self, home):
        res = self._invoke("list")
        assert res.exit_code == 0 and "No adapters registered" in res.output


# --------------------------------------------------------------------------- #
#  The load path: engine, backend, worker, runner                              #
# --------------------------------------------------------------------------- #


class TestEngineResolvesAttachedAdapters:
    def _engine(self, pair, display_name="base", path=None):
        from localm.inference.engine import Engine
        return Engine(str(path or pair.base), display_name=display_name)

    def test_load_hands_the_current_attachments_to_the_backend(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        eng = self._engine(pair)
        seen = {}
        with patch.object(GgufBackend, "load",
                          lambda self: seen.update(adapters=list(self.adapters))):
            eng.load()
        assert seen["adapters"] == [(str(pair.adapter.resolve()), 0.5)]

    def test_a_reload_sees_an_attachment_made_after_construction(self, pair):
        eng = self._engine(pair)
        seen = []
        with patch.object(GgufBackend, "load",
                          lambda self: seen.append(list(self.adapters))):
            eng.load()
            mm.attach_adapter("adp", "base", 0.5)
            eng.load()
        assert seen == [[], [(str(pair.adapter.resolve()), 0.5)]]

    def test_a_detached_adapter_is_not_applied_on_the_next_load(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        eng = self._engine(pair)
        seen = []
        with patch.object(GgufBackend, "load",
                          lambda self: seen.append(list(self.adapters))):
            eng.load()
            mm.detach_adapter("adp")
            eng.unload()
            eng.load()
        assert seen[0] != [] and seen[1] == []

    def test_the_automatic_reload_in_chat_stream_also_refreshes_the_attachments(self, pair):
        eng = self._engine(pair)
        mm.attach_adapter("adp", "base", 0.5)
        seen = []
        with patch.object(GgufBackend, "load",
                          lambda self: seen.append(list(self.adapters))), \
                patch.object(GgufBackend, "chat_stream",
                             lambda self, *a, **k: iter(["x"])):
            list(eng.chat_stream([{"role": "user", "content": "hi"}]))
        assert seen == [[(str(pair.adapter.resolve()), 0.5)]]

    def test_a_name_that_points_at_another_file_gets_no_adapters(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        elsewhere = _write(pair.tmp / "files" / "elsewhere.gguf", _model_bytes("llama"))
        eng = self._engine(pair, display_name="base", path=elsewhere)
        seen = []
        with patch.object(GgufBackend, "load",
                          lambda self: seen.append(list(self.adapters))):
            eng.load()
        assert seen == [[]]

    def test_an_unregistered_model_gets_no_adapters(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        eng = self._engine(pair, display_name="not-registered")
        seen = []
        with patch.object(GgufBackend, "load",
                          lambda self: seen.append(list(self.adapters))):
            eng.load()
        assert seen == [[]]

    def test_an_unusable_attachment_fails_the_load_with_the_reason(self, pair):
        mm.attach_adapter("adp", "base", 0.5)
        mm.update_registry(lambda r: r["adp"].__setitem__("scale", "oops"))
        eng = self._engine(pair)
        with patch.object(GgufBackend, "load") as load:
            with pytest.raises(AdapterLoadError, match="invalid scale"):
                eng.load()
        load.assert_not_called()

    def test_applied_adapters_report_names_never_paths(self, pair):
        eng = self._engine(pair)
        eng._backend.applied_adapters = [{"path": str(pair.adapter), "scale": 0.5}]
        assert eng.applied_adapters == [{"name": "adp.gguf", "scale": 0.5}]
        eng._backend.applied_adapters = []
        assert eng.applied_adapters == []

    def test_the_load_response_fields_carry_the_applied_adapters(self):
        from localm.inference import http_server as hs
        engine = SimpleNamespace(
            gpu_placement=None, mmap_state=None,
            applied_adapters=[{"name": "adp.gguf", "scale": 0.5}])
        assert hs._gpu_placement_fields(engine)["adapters"] == [
            {"name": "adp.gguf", "scale": 0.5}]
        bare = SimpleNamespace(gpu_placement=None, mmap_state=None, applied_adapters=[])
        assert "adapters" not in hs._gpu_placement_fields(bare)


class TestBackendLoadPath:
    def _backend(self, tmp_path, adapters, arch="llama"):
        base = _write(tmp_path / "base.gguf", _model_bytes(arch))
        return GgufBackend(str(base), n_ctx=512, adapters=adapters), base

    def test_a_missing_adapter_file_is_refused_before_vram_and_spawn(self, tmp_path):
        backend, _ = self._backend(tmp_path, [(str(tmp_path / "gone.gguf"), 1.0)])
        with patch.object(GgufBackend, "_check_vram") as vram, \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load") as spawn:
            with pytest.raises(AdapterLoadError) as ei:
                backend.load()
        assert "gone.gguf" in str(ei.value) and "localm adapter detach" in str(ei.value)
        vram.assert_not_called()
        spawn.assert_not_called()

    def test_an_architecture_mismatch_is_refused_before_vram_and_spawn(self, tmp_path):
        adapter = _write(tmp_path / "adp.gguf", _lora_bytes("llama"))
        backend, _ = self._backend(tmp_path, [(str(adapter), 1.0)], arch="qwen3")
        with patch.object(GgufBackend, "_check_vram") as vram, \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load") as spawn:
            with pytest.raises(AdapterLoadError) as ei:
                backend.load()
        assert "'llama'" in str(ei.value) and "'qwen3'" in str(ei.value)
        vram.assert_not_called()
        spawn.assert_not_called()

    def test_the_adapters_reach_the_worker_and_the_applied_ones_come_back(self, tmp_path):
        adapter = _write(tmp_path / "adp.gguf", _lora_bytes("llama"))
        backend, _ = self._backend(tmp_path, [(str(adapter), 0.5)])
        backend.effective_gpu_layers = 0
        captured = {}

        def _spawn(self, params, **kw):
            captured.update(params)
            return {"n_layers": 4, "kv_bytes_per_token": 0, "supports_images": False,
                    "adapters": [{"path": str(adapter), "scale": 0.5}]}

        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", _spawn):
            backend._load_native()
        assert captured["adapters"] == [[str(adapter.resolve()), 0.5]]
        assert backend.applied_adapters == [{"path": str(adapter), "scale": 0.5}]
        backend.unload()
        assert backend.applied_adapters == []

    def test_a_model_without_adapters_sends_none(self, tmp_path):
        backend, _ = self._backend(tmp_path, None)
        backend.effective_gpu_layers = 0
        captured = {}

        def _spawn(self, params, **kw):
            captured.update(params)
            return {"n_layers": 4, "kv_bytes_per_token": 0, "supports_images": False}

        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", _spawn):
            backend._load_native()
        assert "adapters" not in captured
        assert backend.applied_adapters == []


class TestWorkerAndRunner:
    def test_the_worker_forwards_adapters_and_reports_the_applied_ones(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        seen = {}

        class _FakeLlama:
            supports_images = False
            applied_adapters = [{"path": "/a/adp.gguf", "scale": 0.5}]

            def __init__(self, **kw):
                seen.update(kw)

        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
                patch("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama):
            meta = GgufWorker("m.gguf", None, 512, 99, None, 512,
                              adapters=[["/a/adp.gguf", 0.5]]).load()
        assert seen["adapters"] == [("/a/adp.gguf", 0.5)]
        assert meta["adapters"] == [{"path": "/a/adp.gguf", "scale": 0.5}]

    def test_the_worker_passes_no_adapter_argument_when_there_are_none(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        seen = {}

        class _FakeLlama:
            supports_images = False

            def __init__(self, **kw):
                seen.update(kw)

        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
                patch("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama):
            meta = GgufWorker("m.gguf", None, 512, 99, None, 512).load()
        assert "adapters" not in seen
        assert meta["adapters"] == []

    def _runner_with_reply(self, reply):
        from localm.inference.backends.llamacpp._runner import ModelRunner
        r = ModelRunner()

        def fake_spawn():
            r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
            r._resp_q.put(reply)
            r._proc = MagicMock()
            r._proc.is_alive.return_value = True
        r._spawn = fake_spawn
        return r

    def test_the_parent_re_raises_the_tag_as_the_typed_error(self):
        r = self._runner_with_reply(("error", "adapter x refused", "AdapterLoadError"))
        with pytest.raises(AdapterLoadError) as ei:
            r.spawn_and_load({}, timeout=5.0)
        assert str(ei.value) == "adapter x refused"

    def test_the_dispatch_loop_tags_an_adapter_failure(self, monkeypatch):
        import localm._mp_spawn as mp_spawn
        from localm.inference.backends.llamacpp import _runner, _worker

        monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
        monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)

        class _RefusingWorker:
            def __init__(self, **kw):
                pass

            def load(self):
                raise AdapterLoadError("the llama runtime refused LoRA adapter x.gguf")

            def close(self):
                pass

        monkeypatch.setattr(_worker, "GgufWorker", _RefusingWorker)
        req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        died = []

        def _run():
            try:
                _runner._runner_main(req_q, resp_q, ctrl_q)
            except BaseException as e:      # noqa: BLE001 - escaping = the worker dies
                died.append(e)

        t = threading.Thread(target=_run, name="dispatch-under-test", daemon=True)
        t.start()
        try:
            req_q.put(("load", {}))
            envelope = resp_q.get(timeout=5)
            assert envelope[0] == "error" and "x.gguf" in envelope[1]
            assert len(envelope) > 2 and envelope[2] == "AdapterLoadError", envelope
            assert not died, died
        finally:
            req_q.put(None)
            t.join(timeout=5)

    def test_the_backend_does_not_wrap_an_adapter_failure_in_runtime_advice(self, tmp_path):
        adapter = _write(tmp_path / "adp.gguf", _lora_bytes("llama"))
        base = _write(tmp_path / "base.gguf", _model_bytes("llama"))
        backend = GgufBackend(str(base), n_ctx=512, adapters=[(str(adapter), 1.0)])
        with patch.object(GgufBackend, "_check_vram"), \
                patch("localm.discover.list_gpus", return_value=([], "ok")), \
                patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                      "spawn_and_load", side_effect=AdapterLoadError("refused: shape")):
            with pytest.raises(AdapterLoadError) as ei:
                backend.load()
        assert str(ei.value) == "refused: shape"


# --------------------------------------------------------------------------- #
#  The native binding                                                          #
# --------------------------------------------------------------------------- #


class TestBinding:
    def test_set_adapters_marshals_handles_and_scales(self):
        seen = {}

        def fn(ctx, handles, n, factors):
            seen.update(ctx=ctx, handles=[handles[i] for i in range(n)], n=n,
                        factors=[factors[i] for i in range(n)])
            return 0

        with patch.object(api, "_bind", return_value=fn):
            rc = api.llama_set_adapters_lora(ctypes.c_void_p(7), [11, 12], [0.5, -0.25])
        assert rc == 0
        assert seen["n"] == 2 and seen["handles"] == [11, 12]
        assert seen["factors"] == [0.5, -0.25]

    def test_clearing_passes_null_arrays_and_a_zero_count(self):
        seen = []
        with patch.object(api, "_bind", return_value=lambda *a: seen.append(a) or 0):
            api.llama_set_adapters_lora(ctypes.c_void_p(7), [], [])
        assert seen[0][1:] == (None, 0, None)

    def test_mismatched_lengths_never_reach_the_native_call(self):
        fn = MagicMock(return_value=0)
        with patch.object(api, "_bind", return_value=fn):
            with pytest.raises(ValueError, match="2 adapters but 1 scales"):
                api.llama_set_adapters_lora(ctypes.c_void_p(7), [1, 2], [1.0])
        fn.assert_not_called()

    def test_a_native_refusal_comes_back_as_none(self):
        with patch.object(api, "_bind", return_value=lambda model, path: None):
            assert api.llama_adapter_lora_init(ctypes.c_void_p(1), "x.gguf") is None

    def test_a_runtime_without_the_exports_reports_no_lora_api(self):
        class _Lib:
            llama_adapter_lora_init = object()
            llama_adapter_lora_free = object()

        with patch.object(api, "load_lib", return_value=_Lib()):
            assert api.has_lora_api() is False

        class _Full(_Lib):
            llama_set_adapters_lora = object()

        with patch.object(api, "load_lib", return_value=_Full()):
            assert api.has_lora_api() is True

    @pytest.mark.integration
    def test_the_shipped_runtime_exports_the_functions_this_binding_loads(self):
        from tests._real_gguf import native_runtime_lib_path, require_native_runtime
        if native_runtime_lib_path() is None:
            pytest.skip("native llama runtime not provisioned (run 'localm setup-llama')")
        require_native_runtime()
        assert api.has_lora_api() is True


# --------------------------------------------------------------------------- #
#  LlamaCpp: load, apply, rebuild, free                                        #
# --------------------------------------------------------------------------- #


def _mock_api(order=None, *, init_result="auto", set_rc=0, lora_api=True):
    """A MagicMock standing in for the ctypes llama API, recording the order of
    the native calls this feature adds into *order*."""
    order = order if order is not None else []
    m = MagicMock()
    m.llama_model_default_params.return_value = SimpleNamespace(
        main_gpu=0, n_gpu_layers=0, use_mmap=True, load_mtp=False)
    m.has_lora_api.return_value = lora_api
    handles = iter(range(1000, 1100))

    def lora_init(model, path):
        order.append(("lora_init", os.path.basename(path)))
        if init_result is None:
            return None
        return ctypes.c_void_p(next(handles))

    def set_lora(ctx, adapters, scales):
        order.append(("set_lora", getattr(ctx, "value", ctx), list(scales)))
        return set_rc

    m.llama_adapter_lora_init.side_effect = lora_init
    m.llama_set_adapters_lora.side_effect = set_lora
    m.llama_adapter_lora_free.side_effect = lambda h: order.append(("lora_free", h.value))
    m.llama_init_from_model.side_effect = (
        lambda model, cp: order.append(("ctx_init",)) or ctypes.c_void_p(2))
    m.llama_free.side_effect = lambda ctx: order.append(("ctx_free",))
    m.llama_free_model.side_effect = lambda model: order.append(("model_free",))
    m.llama_load_model_from_file.return_value = ctypes.c_void_p(1)
    return m


class TestLlamaCppLifecycle:
    def _build(self, mock_api, **kw):
        with patch.object(llama_mod, "api", mock_api):
            return LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True, **kw)

    def test_adapters_load_after_the_model_and_apply_after_the_context(self):
        order = []
        mock_api = _mock_api(order)
        with patch.object(llama_mod, "api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                           adapters=[("/a/one.gguf", 0.5), ("/a/two.gguf", 2.0)])
            assert order[:4] == [("lora_init", "one.gguf"), ("lora_init", "two.gguf"),
                                 ("ctx_init",), ("set_lora", 2, [0.5, 2.0])]
            assert llm.applied_adapters == [{"path": "/a/one.gguf", "scale": 0.5},
                                            {"path": "/a/two.gguf", "scale": 2.0}]
            llm.close()

    def test_close_frees_the_context_then_the_adapters_then_the_model(self):
        order = []
        mock_api = _mock_api(order)
        with patch.object(llama_mod, "api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                           adapters=[("/a/one.gguf", 1.0), ("/a/two.gguf", 1.0)])
            order.clear()
            llm.close()
        assert order == [("ctx_free",), ("lora_free", 1000), ("lora_free", 1001),
                         ("model_free",)]
        assert llm.applied_adapters == []

    def test_a_model_without_adapters_makes_no_adapter_call(self):
        order = []
        mock_api = _mock_api(order)
        with patch.object(llama_mod, "api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            llm.close()
        mock_api.llama_adapter_lora_init.assert_not_called()
        mock_api.llama_set_adapters_lora.assert_not_called()
        mock_api.llama_adapter_lora_free.assert_not_called()

    def test_a_refused_adapter_frees_the_model_and_raises_with_the_reason(self):
        order = []
        mock_api = _mock_api(order, init_result=None)

        def refuse(model, path):
            os.write(2, b"llama_adapter_lora_init: tensor 'blk.0.attn_q.weight' has "
                        b"incorrect shape (hint: maybe wrong base model?)\n")
            return None

        mock_api.llama_adapter_lora_init.side_effect = refuse
        with patch.object(llama_mod, "api", mock_api):
            with pytest.raises(AdapterLoadError) as ei:
                LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=False,
                         adapters=[("/a/bad.gguf", 1.0)])
        message = str(ei.value)
        assert "bad.gguf" in message and "incorrect shape" in message
        mock_api.llama_init_from_model.assert_not_called()
        mock_api.llama_free_model.assert_called_once()

    def test_a_runtime_without_the_lora_exports_refuses_with_the_remedy(self):
        mock_api = _mock_api(lora_api=False)
        with patch.object(llama_mod, "api", mock_api):
            with pytest.raises(AdapterLoadError, match="localm setup-llama"):
                LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                         adapters=[("/a/one.gguf", 1.0)])
        mock_api.llama_free_model.assert_called_once()

    def test_a_failed_apply_frees_everything_and_raises(self):
        order = []
        mock_api = _mock_api(order, set_rc=-1)
        with patch.object(llama_mod, "api", mock_api):
            with pytest.raises(AdapterLoadError, match="could not apply"):
                LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                         adapters=[("/a/one.gguf", 1.0)])
        assert ("ctx_free",) in order and ("lora_free", 1000) in order
        assert order[-1] == ("model_free",)

    def test_an_adapter_does_not_switch_speculation_off(self):
        mock_api = _mock_api()
        with patch.object(llama_mod, "api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                           spec_source="mtp", adapters=[("/a/one.gguf", 1.0)])
            assert mock_api.llama_model_default_params.return_value.load_mtp is True
            mock_api.llama_model_mtp_support.assert_called()
            assert llm.mtp_status != "disabled"
            llm.close()

    def test_a_rebuilt_context_gets_the_adapters_before_its_first_decode(self):
        order = []
        mock_api = _mock_api(order)
        mock_api.llama_context_default_params.return_value = SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, offload_kqv=True)
        mock_api.llama_init_from_model.side_effect = (
            lambda model, cp: order.append(("ctx_init", cp.n_ctx)) or ctypes.c_void_p(77))
        mock_api.llama_decode.side_effect = (
            lambda ctx, batch: order.append(("decode", ctx.value)) or 0)
        llm = make_bare_llama(
            _model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2),
            _mtp_enabled=False,
            _adapter_specs=[("/a/one.gguf", 0.5)],
            _adapter_handles=[ctypes.c_void_p(1000)])
        llm._target_ctx = lambda needed: 8192
        llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace()
        with patch.object(llama_mod, "api", mock_api):
            llm._prefill_fresh_context([1, 2, 3], 10)
        assert ("set_lora", 77, [0.5]) in order
        assert order.index(("set_lora", 77, [0.5])) < order.index(("decode", 77)), order
        assert llm.applied_adapters == [{"path": "/a/one.gguf", "scale": 0.5}]

    def test_a_rebuilt_context_that_cannot_take_the_adapters_is_not_left_running(self):
        order = []
        mock_api = _mock_api(order, set_rc=-1)
        mock_api.llama_context_default_params.return_value = SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, offload_kqv=True)
        llm = make_bare_llama(
            _model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2),
            _mtp_enabled=False,
            _adapter_specs=[("/a/one.gguf", 0.5)],
            _adapter_handles=[ctypes.c_void_p(1000)])
        llm._target_ctx = lambda needed: 8192
        with patch.object(llama_mod, "api", mock_api):
            with pytest.raises(AdapterLoadError):
                llm._prefill_fresh_context([1, 2, 3], 10)
        assert llm._ctx_ptr is None
        mock_api.llama_decode.assert_not_called()

    def test_a_rebuilt_context_without_adapters_makes_no_adapter_call(self):
        mock_api = _mock_api()
        mock_api.llama_context_default_params.return_value = SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, offload_kqv=True)
        mock_api.llama_decode.return_value = 0
        llm = make_bare_llama(
            _model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2), _mtp_enabled=False)
        llm._target_ctx = lambda needed: 8192
        llm._create_batch = lambda tokens, pos, **kw: SimpleNamespace()
        with patch.object(llama_mod, "api", mock_api):
            llm._prefill_fresh_context([1, 2, 3], 10)
        mock_api.llama_set_adapters_lora.assert_not_called()
