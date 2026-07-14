# SPDX-License-Identifier: AGPL-3.0-or-later
"""The two new GUI routes for the ComfyUI missing-model auto-download offer:
POST /api/media/{kind}/preflight (read-only pre-check) and
POST /api/models/pull-comfy-source (curated-only pull trigger, never a
client-supplied repo/path). Fixture mirrors test_key_scope_gui.py's scoped_app."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S


@pytest.fixture
def scoped_app(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    switched = []

    async def switch_model(name):
        switched.append(name)

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model,
               active_model=lambda: switched[-1] if switched else "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


class TestPreflightRoute:
    def test_unknown_kind_is_404(self, scoped_app):
        with TestClient(scoped_app) as c:
            r = c.post("/api/media/bogus/preflight", json={})
        assert r.status_code == 404

    def test_no_comfy_running_reports_nothing_missing(self, scoped_app):
        # Best-effort, matching preflight_models: an unreachable ComfyUI never
        # surfaces a false "missing" list.
        with TestClient(scoped_app) as c:
            r = c.post("/api/media/image/preflight", json={})
        assert r.status_code == 200
        assert r.json() == {"missing": []}

    def test_reports_missing_curated_file_with_source_and_dest(self, scoped_app, tmp_path):
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["comfy_workdir"] = str(tmp_path / "external-comfy")
        _cfg.save_config(cfg)

        fake_info = {
            "UnetLoaderGGUF": {
                "input": {"required": {"unet_name": [["other.gguf"], {}]}}
            }
        }
        fake_wf = tmp_path / "wf.json"
        fake_wf.write_text(json.dumps({
            "1": {"class_type": "UnetLoaderGGUF",
                  "inputs": {"unet_name": "flux1-dev-Q8_0.gguf"}}
        }))

        from localm.media import comfy_client as cc
        with patch.object(cc, "comfy_object_info", return_value=fake_info), \
             patch("localm.image_gen.comfy._workflow_path", return_value=fake_wf):
            with TestClient(scoped_app) as c:
                r = c.post("/api/media/image/preflight", json={})

        assert r.status_code == 200
        missing = r.json()["missing"]
        assert len(missing) == 1
        entry = missing[0]
        assert entry["filename"] == "flux1-dev-Q8_0.gguf"
        assert entry["source"]["repo"] == "city96/FLUX.1-dev-gguf"
        assert entry["source"]["file"] == "flux1-dev-Q8_0.gguf"
        assert entry["source"]["size_bytes"] == 12_708_281_504
        assert entry["dest_dir"] == str(Path(str(tmp_path / "external-comfy")) / "models" / "unet")

    def test_reports_uncurated_missing_file_with_null_source(self, scoped_app, tmp_path):
        fake_info = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["other.safetensors"], {}]}}
            }
        }
        fake_wf = tmp_path / "wf.json"
        fake_wf.write_text(json.dumps({
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "totally-custom-model.safetensors"}}
        }))
        from localm.media import comfy_client as cc
        with patch.object(cc, "comfy_object_info", return_value=fake_info), \
             patch("localm.image_gen.comfy._workflow_path", return_value=fake_wf):
            with TestClient(scoped_app) as c:
                r = c.post("/api/media/image/preflight", json={})
        assert r.status_code == 200
        missing = r.json()["missing"]
        assert len(missing) == 1
        assert missing[0]["source"] is None
        assert missing[0]["dest_dir"] is None


class TestPullComfySourceRoute:
    def test_uncurated_filename_is_400(self, scoped_app):
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-a-curated-model.gguf"})
        assert r.status_code == 400

    def test_curated_filename_with_no_routable_comfy_is_400(self, scoped_app):
        # No comfy_workdir configured and no managed instance installed -> no
        # known destination -> a clear 400, never a silent MODELS_DIR fallback.
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "flux1-dev-Q8_0.gguf"})
        assert r.status_code == 400

    def test_curated_filename_with_workdir_starts_a_job(self, scoped_app, tmp_path):
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["comfy_workdir"] = str(tmp_path / "external-comfy")
        _cfg.save_config(cfg)
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "ae.safetensors"})
        assert r.status_code == 200
        assert "job_id" in r.json()


class TestComfyRoutesAreScoped:
    def test_underscoped_key_is_403(self, scoped_app):
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        with TestClient(scoped_app) as c:
            r1 = c.post("/api/media/image/preflight", json={}, headers=_hdr(narrow))
            r2 = c.post("/api/models/pull-comfy-source",
                       json={"filename": "ae.safetensors"}, headers=_hdr(narrow))
        assert r1.status_code == 403
        assert r2.status_code == 403

    def test_models_write_key_reaches_both_routes(self, scoped_app):
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped_app) as c:
            r1 = c.post("/api/media/image/preflight", json={}, headers=_hdr(writer))
            r2 = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-curated.gguf"}, headers=_hdr(writer))
        assert r1.status_code != 403
        assert r2.status_code != 403   # reaches the route (400 for uncurated, not 403)
