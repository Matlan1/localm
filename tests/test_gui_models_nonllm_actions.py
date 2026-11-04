# SPDX-License-Identifier: AGPL-3.0-or-later
"""Alias, rename and remove on registered models that are not chat models: the
GUI routes carry no type gate, and the removal itself keeps a file registered in
place while deleting one the data folder owns."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui

NON_LLM_TYPES = ["embedding", "diffusion-unet", "text-encoder", "vae", "lora",
                 "upscaler", "controlnet", "unknown", "mmproj", "tts"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as _mm
    monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
    return home


@pytest.fixture(autouse=True)
def _no_engine_left_by_another_test(monkeypatch):
    """The remove route asks the loaded engines whether they hold the file."""
    from localm.inference import http_server
    monkeypatch.setattr(http_server, "_engines", {})
    monkeypatch.setattr(http_server, "_engine", None)


@pytest.fixture
def app():
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: {"status": "loaded", "model": name},
               active_model=lambda: "some-chat-model")
    return app


def _seed(entries):
    from localm import model_manager as mm
    mm.update_registry(lambda reg: reg.update(entries))
    return mm


@pytest.mark.parametrize("model_type", NON_LLM_TYPES)
def test_alias_and_rename_work_for_every_non_chat_type(app, home, model_type):
    f = home / "models" / "part.bin"
    f.write_bytes(b"x")
    mm = _seed({"the-part": {"path": str(f), "source": "local", "model_type": model_type}})
    with TestClient(app) as client:
        r = client.post("/api/models/alias", json={"model": "the-part", "alias": "second name"})
        assert r.status_code == 200, r.text
        assert r.json()["alias"] == "second-name"
        r = client.post("/api/models/rename", json={"model": "the-part", "new_name": "renamed-part"})
        assert r.status_code == 200, r.text
    reg = mm.load_registry()
    assert "the-part" not in reg
    assert reg["renamed-part"]["model_type"] == model_type
    assert reg["second-name"]["model_type"] == model_type
    assert reg["renamed-part"]["path"] == str(f) and f.exists(), "rename never touches the file"


@pytest.mark.parametrize("model_type", NON_LLM_TYPES)
def test_remove_route_accepts_every_non_chat_type(app, home, monkeypatch, model_type):
    f = home / "models" / "part.bin"
    f.write_bytes(b"x")
    _seed({"the-part": {"path": str(f), "source": "local", "model_type": model_type}})
    captured = {}

    class _Job:
        id = "job-1"

    def fake_start_cli(self, kind, cli_args, **kw):
        captured["args"] = list(cli_args)
        return _Job()

    monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)
    with TestClient(app) as client:
        r = client.post("/api/models/remove", json={"model": "the-part"})
    assert r.status_code == 200, r.text
    assert captured["args"] == ["rm", "the-part", "--yes"]


def test_remove_deletes_an_owned_non_chat_file_and_keeps_one_registered_in_place(home, tmp_path):
    from localm.model_manager.registry import remove_model
    owned = home / "models" / "ae.safetensors"
    owned.write_bytes(b"x")
    external_dir = tmp_path / "central"
    external_dir.mkdir()
    external = external_dir / "clip_l.safetensors"
    external.write_bytes(b"x")
    mm = _seed({
        "owned-vae": {"path": str(owned), "source": "local", "model_type": "vae"},
        "central-clip": {"path": str(external), "source": "local", "model_type": "text-encoder"},
    })
    remove_model("owned-vae")
    remove_model("central-clip")
    reg = mm.load_registry()
    assert "owned-vae" not in reg and "central-clip" not in reg
    assert not owned.exists(), "a file in the data folder is deleted, as for a chat model"
    assert external.exists(), "a file registered in place is only unregistered"
