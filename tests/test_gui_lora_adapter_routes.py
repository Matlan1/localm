# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI routes that attach and detach GGUF LoRA adapters, and the adapter
fields ``GET /api/models`` reports, driven against a real registry and real
GGUF headers."""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.model_manager as mm
from localm.inference import http_server as _hs
from localm.model_manager import gguf as _gguf_mod
from localm.plugins.gui.web import attach_gui


def _kv(key: str, value: str) -> bytes:
    kb = key.encode("utf-8")
    vb = value.encode("utf-8")
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8)
            + struct.pack("<Q", len(vb)) + vb)


def _gguf(arch: str, general_type: str) -> bytes:
    kvs = [("general.architecture", arch), ("general.type", general_type)]
    if general_type == "adapter":
        kvs.append(("adapter.type", "lora"))
    buf = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
           + struct.pack("<Q", len(kvs)))
    for key, value in kvs:
        buf += _kv(key, value)
    return buf.ljust(max(len(buf), _gguf_mod._GGUF_MIN_BYTES), b"\0")


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + b"\0" + path.name.encode("utf-8"))
    old = time.time() - 3600.0
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
def registered(home, tmp_path):
    """A llama base "base", a llama adapter "adp", a qwen adapter "other-adp"."""
    files = tmp_path / "files"
    base = _write(files / "base.gguf", _gguf("llama", "model"))
    adp = _write(files / "adp.gguf", _gguf("llama", "adapter"))
    other = _write(files / "other-adp.gguf", _gguf("qwen3", "adapter"))
    for path, name in ((base, "base"), (adp, "adp"), (other, "other-adp")):
        assert mm.add_local(str(path), name=name) is True
    return SimpleNamespace(base=base, adp=adp, other=other)


@pytest.fixture
def client():
    app = FastAPI()

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "")
    with TestClient(app) as c:
        yield c


def _row(client, name):
    models = client.get("/api/models").json()["models"]
    return next(m for m in models if m["name"] == name)


def _attach(client, adapter="adp", base="base", scale=1.0):
    return client.post("/api/models/adapters/attach",
                       json={"adapter": adapter, "base": base, "scale": scale})


class TestAttachRoute:
    def test_attach_records_base_and_scale_in_the_registry(self, registered, client):
        r = _attach(client, scale=0.8)
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "attached", "adapter": "adp", "base": "base",
                            "scale": 0.8, "needs_reload": False}
        entry = mm.load_registry()["adp"]
        assert (entry["base"], entry["scale"]) == ("base", 0.8)

    def test_a_mismatched_architecture_is_a_400_naming_both_and_changes_nothing(
            self, registered, client):
        r = _attach(client, adapter="other-adp")
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "qwen3" in detail and "llama" in detail
        assert "base" not in mm.load_registry()["other-adp"]

    @pytest.mark.parametrize("scale", [0, 0.0])
    def test_a_zero_scale_is_refused(self, registered, client, scale):
        r = _attach(client, scale=scale)
        assert r.status_code == 400
        assert "scale" in r.json()["detail"]
        assert "base" not in mm.load_registry()["adp"]

    def test_unknown_names_are_404(self, registered, client):
        assert _attach(client, adapter="nope").status_code == 404
        assert _attach(client, base="nope").status_code == 404

    def test_a_chat_model_is_not_an_adapter(self, registered, client):
        r = _attach(client, adapter="base", base="adp")
        assert r.status_code == 400
        assert "base" not in mm.load_registry()["base"] or \
            mm.load_registry()["base"].get("base") != "adp"

    def test_attaching_to_a_resident_base_says_a_reload_is_needed(
            self, registered, client, monkeypatch):
        engine = SimpleNamespace(loaded=True, model_path=str(registered.base),
                                 applied_adapters=[])
        monkeypatch.setitem(_hs._engines, "base", engine)
        assert _attach(client).json()["needs_reload"] is True

    def test_a_resident_alias_of_the_base_counts(self, registered, client, monkeypatch):
        assert mm.alias_model("base", "base-alias")
        engine = SimpleNamespace(loaded=True, model_path=str(registered.base),
                                 applied_adapters=[])
        monkeypatch.setitem(_hs._engines, "base-alias", engine)
        assert _attach(client).json()["needs_reload"] is True

    def test_a_resident_other_model_does_not_count(self, registered, client, monkeypatch):
        other = _write(registered.base.parent / "tiny.gguf", _gguf("llama", "model"))
        assert mm.add_local(str(other), name="tiny") is True
        engine = SimpleNamespace(loaded=True, model_path=str(other), applied_adapters=[])
        monkeypatch.setitem(_hs._engines, "tiny", engine)
        assert _attach(client).json()["needs_reload"] is False


class TestDetachRoute:
    def test_detach_clears_the_attachment_and_keeps_the_adapter_registered(
            self, registered, client):
        _attach(client, scale=0.5)
        r = client.post("/api/models/adapters/detach", json={"adapter": "adp"})
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "detached", "adapter": "adp", "needs_reload": False}
        entry = mm.load_registry()["adp"]
        assert "base" not in entry and "scale" not in entry
        assert entry["model_type"] == "lora"

    def test_detaching_an_unattached_adapter_is_404(self, registered, client):
        r = client.post("/api/models/adapters/detach", json={"adapter": "adp"})
        assert r.status_code == 404

    def test_detaching_an_unknown_name_is_404(self, registered, client):
        assert client.post("/api/models/adapters/detach",
                           json={"adapter": "nope"}).status_code == 404

    def test_detaching_from_a_resident_base_says_a_reload_is_needed(
            self, registered, client, monkeypatch):
        _attach(client)
        engine = SimpleNamespace(loaded=True, model_path=str(registered.base),
                                 applied_adapters=[{"name": "adp.gguf", "scale": 1.0}])
        monkeypatch.setitem(_hs._engines, "base", engine)
        r = client.post("/api/models/adapters/detach", json={"adapter": "adp"})
        assert r.json()["needs_reload"] is True


class TestModelListAdapterFields:
    def test_an_adapter_row_reports_its_attachment(self, registered, client):
        _attach(client, scale=0.8)
        row = _row(client, "adp")
        assert row["adapter"] is True
        assert row["base"] == "base" and row["base_registered"] is True
        assert row["scale"] == 0.8
        assert row["adapter_file"] == "adp.gguf"

    def test_an_unattached_adapter_row_has_no_base(self, registered, client):
        row = _row(client, "adp")
        assert row["adapter"] is True
        assert "base" not in row and "scale" not in row

    def test_a_base_row_lists_what_is_attached_to_it(self, registered, client):
        _attach(client, scale=0.8)
        assert _row(client, "base")["adapters"] == [
            {"name": "adp", "file": "adp.gguf", "scale": 0.8}]

    def test_an_alias_of_the_base_lists_the_same_adapters(self, registered, client):
        assert mm.alias_model("base", "base-alias")
        _attach(client, scale=0.8)
        assert [a["name"] for a in _row(client, "base-alias")["adapters"]] == ["adp"]

    def test_a_loaded_base_reports_the_adapters_it_runs_with(
            self, registered, client, monkeypatch):
        _attach(client, scale=0.8)
        applied = [{"name": "adp.gguf", "scale": 0.8}]
        engine = SimpleNamespace(loaded=True, model_path=str(registered.base),
                                 applied_adapters=applied)
        monkeypatch.setitem(_hs._engines, "base", engine)
        row = _row(client, "base")
        assert row["loaded"] is True
        assert row["applied_adapters"] == applied

    def test_a_loaded_base_with_no_adapters_reports_none(
            self, registered, client, monkeypatch):
        engine = SimpleNamespace(loaded=True, model_path=str(registered.base),
                                 applied_adapters=[])
        monkeypatch.setitem(_hs._engines, "base", engine)
        row = _row(client, "base")
        assert "adapters" not in row and "applied_adapters" not in row

    def test_a_plain_model_row_carries_no_adapter_fields(self, registered, client):
        row = _row(client, "base")
        assert not {"adapter", "adapters", "applied_adapters", "base", "scale"} & set(row)

    def test_an_adapter_whose_base_was_removed_is_flagged(self, registered, client):
        _attach(client)

        def _drop(reg):
            reg.pop("base")

        mm.update_registry(_drop)
        row = _row(client, "adp")
        assert row["base"] == "base" and row["base_registered"] is False
