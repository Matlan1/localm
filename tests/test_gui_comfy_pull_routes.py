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
        # surfaces a false "missing" list. Mocks comfy_object_info directly
        # (matching the sibling tests below) instead of relying on nothing
        # answering the real ComfyUI default port - a real ComfyUI on 8188
        # (a common setup for this project's contributors) would otherwise
        # make this an unmocked network call whose result, and this test's
        # outcome, this test does not control (the exact shape fixed in
        # test_gui.py's comfy-model-picker test).
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
        be rejected, and the TRIMMED value (not the raw one) is what reaches the
        check-workflow - otherwise a literal space in the injected LoraLoader
        node would never match ComfyUI's actual (untrimmed) installed filename,
        making every such LoRA look permanently missing.

        The base template starts with NO LoraLoader node: _build_image_workflow
        injects a FRESH one via next_node_id() rather than reusing/overwriting an
        existing one (comfy.py's "Inject LoRA" step), so seeding a stale one here
        would itself show up as a spurious missing entry unrelated to trimming."""
        fake_info = {"LoraLoader": {"input": {"required": {
            "lora_name": [["my_style.safetensors"], {}],
        }}}}
        fake_wf = tmp_path / "wf.json"
        fake_wf.write_text(json.dumps({}))
        from localm.media import comfy_client as cc
        with patch.object(cc, "comfy_object_info", return_value=fake_info), \
             patch("localm.image_gen.comfy.workflow_path", return_value=fake_wf):
            with TestClient(scoped_app) as c:
                r = c.post("/api/media/image/preflight",
                           json={"lora_name": "  my_style.safetensors  "})
        assert r.status_code == 200, r.text
        # The trimmed name is installed (per fake_info), so nothing is missing.
        assert r.json()["missing"] == []

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
        """REVISED CONTRACT. This test previously asserted that a plain
        models:write key REACHES this route (`!= 403`, "400 for uncurated, not
        403"). That is no longer true, deliberately.

        The route downloads into comfy_models_dest_dir(), which resolves through
        `comfy_workdir` whenever the managed ComfyUI instance is not active - the
        default on a fresh install. `comfy_workdir` carries no admin_only flag, so
        a config:write key chooses that folder; the route then mkdir -p's it and
        streams a multi-gigabyte download into it from the server process. Picking
        the directory the server writes gigabytes into is host filesystem reach,
        and a UNC value there draws outbound SMB authentication from the server -
        the same capability /api/fs/dirs is gated on, and the same reason the
        sibling /api/models/scan route is gated.

        models:write governs WHICH MODELS may be added; it does not govern WHERE
        ON THE DISK the server may write. Those came apart here."""
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-curated.gguf"}, headers=_hdr(writer))
        assert r.status_code == 403

    def test_pull_comfy_source_reaches_the_route_with_host_fs_access(self, scoped_app):
        """Fires-control for the test above: with host filesystem access the
        request gets PAST the gate and is answered on its merits (400 for an
        uncurated name). Without this, the 403 above would also pass against a
        route that had simply become unreachable for everyone."""
        from localm import auth
        writer = auth.create_key("hostwriter", [S.MODELS_WRITE], fs_access="host")["key"]
        with TestClient(scoped_app) as c:
            r = c.post("/api/models/pull-comfy-source",
                       json={"filename": "not-curated.gguf"}, headers=_hdr(writer))
        assert r.status_code == 400, r.text
