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
        # surfaces a false "missing" list. Mocks comfy_object_info directly rather
        # than relying on nothing answering the real ComfyUI default port.
        from localm.media import comfy_client as cc
        with patch.object(cc, "comfy_object_info", return_value=None):
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
             patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf):
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

    def test_dest_dir_resolves_via_plugin_only_workdir(self, scoped_app, tmp_path):
        """NEW-COMFY-DOWNLOAD-DEST-IGNORES-PLUGIN-WORKDIR: media_preflight
        already has `kind` in scope and now passes it through to
        comfy_models_dest_dir(), so a workdir set ONLY per-plugin (the shape
        the modern Settings UI actually produces - no global comfy_workdir at
        all) resolves correctly here too, not just at the download route."""
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["plugins"] = {"image": {"comfy": {"workdir": str(tmp_path / "per-plugin-comfy")}}}
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
             patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf):
            with TestClient(scoped_app) as c:
                r = c.post("/api/media/image/preflight", json={})

        assert r.status_code == 200
        missing = r.json()["missing"]
        assert len(missing) == 1
        assert missing[0]["dest_dir"] == str(
            Path(str(tmp_path / "per-plugin-comfy")) / "models" / "unet")

    @pytest.mark.parametrize("bad_name", [
        "../secrets.safetensors", "..\\secrets.safetensors",
        "sub/dir.safetensors", "sub\\dir.safetensors",
        "C:evil.safetensors", "..",
    ])
    def test_lora_name_traversal_rejected(self, scoped_app, bad_name):
        """Mirrors plug.py's identical test for the real /api/imagine route -
        the preflight route reuses the same is_safe_lora_name() predicate (see
        image_gen/comfy.py) rather than re-implementing the check, so a
        traversal/absolute/UNC-shaped value must be rejected here too, before
        it ever reaches _build_check_workflow's LoraLoader injection."""
        with TestClient(scoped_app) as c:
            r = c.post("/api/media/image/preflight", json={"lora_name": bad_name})
        assert r.status_code == 400

    def test_lora_name_whitespace_is_trimmed_before_use(self, scoped_app, tmp_path):
        """A safe lora_name with incidental leading/trailing whitespace must not
        be rejected, and the TRIMMED value (not the raw one) is what reaches
        _build_image_workflow. Asserts the exact kwarg _build_image_workflow is
        called with directly (rather than checking describe_missing_models's
        output) because _pick_variant's precision/quant-insensitive matching
        tokenizes on any non-alphanumeric character - including whitespace - so
        a padded and unpadded filename normalize to the SAME base and an
        untrimmed value would still resolve as "not missing" even without the
        strip, so the value itself is asserted."""
        fake_wf = tmp_path / "wf.json"
        fake_wf.write_text(json.dumps({}))
        captured = {}

        def fake_build(workflow, **kwargs):
            captured.update(kwargs)
            return True, "", None

        from localm.media import comfy_client as cc
        with patch.object(cc, "comfy_object_info", return_value=None), \
             patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf), \
             patch("localm.image_gen.comfy._build_image_workflow", side_effect=fake_build):
            with TestClient(scoped_app) as c:
                r = c.post("/api/media/image/preflight",
                           json={"lora_name": "  my_style.safetensors  "})
        assert r.status_code == 200, r.text
        assert captured.get("lora_name") == "my_style.safetensors"

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
             patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf):
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

    def test_curated_filename_with_plugin_only_workdir_and_plugin_hint_starts_a_job(
            self, scoped_app, tmp_path):
        """End to end through the real route and request model: a workdir set
        ONLY via the per-plugin comfy.workdir field (no global comfy_workdir at
        all, the shape the Settings UI produces) resolves correctly when the
        request names which plugin is asking, which is what the browser flow
        sends (helpers.js's checkModelsBeforeGenerate ->
        _offerModelDownload)."""
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["plugins"] = {"image": {"comfy": {"workdir": str(tmp_path / "per-plugin-comfy")}}}
        _cfg.save_config(cfg)
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "ae.safetensors", "plugin": "image"})
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_curated_filename_with_plugin_only_workdir_and_no_plugin_hint_is_400(
            self, scoped_app, tmp_path):
        """Without a plugin hint (an older/direct API caller), resolution
        still falls back to the legacy global key only - documented,
        unchanged behavior; the real browser flow always sends `plugin` now,
        so this is not a regression this fix needs to close."""
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["plugins"] = {"image": {"comfy": {"workdir": str(tmp_path / "per-plugin-comfy")}}}
        _cfg.save_config(cfg)
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "ae.safetensors"})
        assert r.status_code == 400

    def test_unrecognized_plugin_value_falls_back_gracefully(self, scoped_app, tmp_path):
        """req.plugin is a SELECTOR into trusted server config, validated
        against MEDIA_PLUGINS (web.py's own contract comment) - an
        unrecognized value must not error, just fall back to no plugin
        context, same as omitting it entirely."""
        import localm.config as _cfg
        cfg = _cfg.load_config()
        cfg["comfy_workdir"] = str(tmp_path / "external-comfy")
        _cfg.save_config(cfg)
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "ae.safetensors", "plugin": "not-a-real-plugin"})
        assert r.status_code == 200, r.text


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

    def test_models_write_key_reaches_preflight(self, scoped_app):
        """preflight is READ-ONLY (it reports which models ComfyUI is missing), so
        models:write alone still reaches it."""
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped_app) as c:
            r1 = c.post("/api/media/image/preflight", json={}, headers=_hdr(writer))
        assert r1.status_code != 403

    def test_pull_comfy_source_now_needs_host_fs_access(self, scoped_app):
        """A plain models:write key does NOT reach this route.

        The route downloads into comfy_models_dest_dir(), which resolves through
        `comfy_workdir` whenever the managed ComfyUI instance is not active, the
        default on a fresh install. `comfy_workdir` is admin_only, but an admin
        may have configured it already and this route's own caller only needs
        MODELS_WRITE, so without its own gate any MODELS_WRITE key could stream
        a multi-gigabyte download into that host directory. Picking the
        directory the server writes gigabytes into is host filesystem reach, and
        a UNC value there draws outbound SMB authentication from the server.

        models:write governs WHICH MODELS may be added, not WHERE ON THE DISK
        the server may write."""
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-curated.gguf"}, headers=_hdr(writer))
        assert r.status_code == 403

    def test_pull_comfy_source_reaches_the_route_with_host_fs_access(self, scoped_app):
        """The control for the test above: with host filesystem access the
        request gets PAST the gate and is answered on its merits (400 for an
        uncurated name). The 403 above would otherwise also pass against a route
        that had become unreachable for everyone."""
        from localm import auth
        writer = auth.create_key("hostwriter", [S.MODELS_WRITE], fs_access="host")["key"]
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-curated.gguf"}, headers=_hdr(writer))
        assert r.status_code == 400, r.text
