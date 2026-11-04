# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI settings every media plugin reads: where the API URL comes from, which
Model weight dtype values reach a workflow, and the launcher hint for the Settings
page."""

from __future__ import annotations

import pytest

from localm import settings_schema as ss
from localm.media import comfy_client
from localm.music_gen import comfy as music_comfy
from localm.plugins.builtin.image import backend as image_backend
from localm.plugins.builtin.music import backend as music_backend
from localm.plugins.builtin.video import backend as video_backend
from localm.video_gen import comfy as video_comfy

BACKENDS = [("image", image_backend), ("music", music_backend), ("video", video_backend)]


@pytest.fixture(autouse=True)
def _user_comfyui(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)


@pytest.mark.parametrize("name,backend", BACKENDS)
class TestApiUrlPrecedence:
    def test_a_set_shared_url_wins_over_the_environment(self, name, backend, monkeypatch):
        monkeypatch.setenv("FLUX_API_URL", "http://127.0.0.1:9001")
        s = backend.settings({"comfy_target": "user", "comfy_api_url": "http://127.0.0.1:9002"})
        assert s["api_url"] == "http://127.0.0.1:9002"

    def test_a_set_plugin_url_wins_over_both(self, name, backend, monkeypatch):
        monkeypatch.setenv("FLUX_API_URL", "http://127.0.0.1:9001")
        s = backend.settings({
            "comfy_target": "user", "comfy_api_url": "http://127.0.0.1:9002",
            "plugins": {name: {"comfy": {"api_url": "http://127.0.0.1:9003"}}}})
        assert s["api_url"] == "http://127.0.0.1:9003"

    def test_blank_uses_the_environment(self, name, backend, monkeypatch):
        monkeypatch.setenv("FLUX_API_URL", "http://127.0.0.1:9001")
        monkeypatch.setattr("localm.config.load_config", lambda: {"comfy_target": "user"})
        s = backend.settings({"comfy_target": "user"})
        assert s["api_url"] == "http://127.0.0.1:9001"


def _workflow():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m", "weight_dtype": "default"}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "m"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m"}},
    }


class TestWeightDtype:
    def test_options_are_what_comfyui_accepts(self):
        per_plugin = next(f for f in ss.MEDIA_PLUGIN_FIELDS if f.key == "float_type")
        assert tuple(per_plugin.options) == comfy_client.UNET_WEIGHT_DTYPES

    @pytest.mark.parametrize("old", ["fp16", "bf16", "fp32"])
    def test_a_value_comfyui_refuses_is_no_longer_accepted(self, old):
        with pytest.raises(ValueError):
            ss.validate_update({"comfy_float_type": old})

    def test_only_unetloader_receives_it(self):
        wf = _workflow()
        comfy_client.apply_weight_dtype(wf, "fp8_e5m2")
        assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e5m2"
        assert "weight_dtype" not in wf["2"]["inputs"]
        assert "weight_dtype" not in wf["3"]["inputs"]

    @pytest.mark.parametrize("value", [None, "", "default", "fp16", "bf16", "fp32"])
    def test_default_and_old_values_leave_the_workflow_alone(self, value):
        wf = _workflow()
        comfy_client.apply_weight_dtype(wf, value)
        assert wf == _workflow()

    @pytest.mark.parametrize("name,backend", BACKENDS[1:])
    def test_an_old_stored_value_warns(self, name, backend):
        s = backend.settings({"comfy_target": "user",
                              "plugins": {name: {"comfy": {"float_type": "fp16"}}}})
        assert "fp16" in (s["warning"] or "")

    @pytest.mark.parametrize("name,backend", BACKENDS[1:])
    def test_an_accepted_value_does_not_warn(self, name, backend):
        s = backend.settings({"comfy_target": "user",
                              "plugins": {name: {"comfy": {"float_type": "fp8_e4m3fn"}}}})
        assert s["float_type"] == "fp8_e4m3fn"
        assert "dtype" not in (s["warning"] or "")

    @pytest.mark.parametrize("block,config", [
        ({"comfy": {"float_type": "fp16"}}, {}),
        ({}, {"comfy_float_type": "fp16"}),
    ], ids=["own", "shared"])
    def test_the_settings_page_shows_an_old_stored_value(self, block, config):
        offered = next(f for f in ss.MEDIA_PLUGIN_FIELDS if f.key == "float_type").options
        before = list(offered)
        rows = ss.media_schema_json("video", block, config)
        field = next(f for f in rows if f["key"] == "float_type")
        assert field["value"] == "fp16"
        assert field["options"] == [*before, "fp16"]
        assert offered == before, "the shared option list was changed"

    def test_a_stored_value_that_is_not_text_is_shown_as_text(self):
        rows = ss.media_schema_json("video", {"comfy": {"float_type": 16}}, {})
        field = next(f for f in rows if f["key"] == "float_type")
        assert field["value"] == "16"
        assert "16" in field["options"]

    def test_an_accepted_value_adds_no_option(self):
        offered = next(f for f in ss.MEDIA_PLUGIN_FIELDS if f.key == "float_type").options
        rows = ss.media_schema_json("video", {"comfy": {"float_type": "fp8_e5m2"}}, {})
        field = next(f for f in rows if f["key"] == "float_type")
        assert field["value"] == "fp8_e5m2"
        assert field["options"] == offered

    def test_generators_import_the_writer(self):
        assert music_comfy.apply_weight_dtype is comfy_client.apply_weight_dtype
        assert video_comfy.apply_weight_dtype is comfy_client.apply_weight_dtype


_OWNER_KEY = "owner-key-for-dtype-route-tests"
_OWNER = {"Authorization": f"Bearer {_OWNER_KEY}"}


@pytest.mark.parametrize("name,backend", BACKENDS[1:])
@pytest.mark.parametrize("stored", [
    {"plugins": {"video": {"comfy": {"float_type": "fp16"}},
                 "music": {"comfy": {"float_type": "fp16"}}}},
    {"comfy_float_type": "bf16"},
], ids=["own", "shared"])
class TestSavingDefaultClearsAnOldDtype:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setenv("LOCALM_API_KEY", _OWNER_KEY)
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        from fastapi.testclient import TestClient

        from localm.inference.http_server import create_app
        with TestClient(create_app(None)) as c:
            yield c, cfg

    def test_the_real_route_clears_the_note(self, client, name, backend, stored):
        c, cfg = client
        cfg.update_config(lambda conf: conf.update(stored))
        assert "dtype" in (backend.settings(cfg.load_config())["warning"] or "")

        shown = c.get("/v1/media/config", headers=_OWNER).json()
        field = next(f for p in shown["plugins"] if p["plugin"] == name
                     for f in p["fields"] if f["key"] == "float_type")
        assert field["value"] in ("fp16", "bf16") and field["value"] in field["options"]

        r = c.post(f"/v1/media/config/{name}", json={"float_type": "default"}, headers=_OWNER)
        assert r.status_code == 200, r.text
        saved = next(f for f in r.json()["fields"] if f["key"] == "float_type")
        assert saved["value"] == "default"
        assert saved["options"] == ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
        assert "dtype" not in (backend.settings(cfg.load_config())["warning"] or "")


class TestLauncherHint:
    def _field(self, block, cfg):
        rows = ss.media_schema_json("image", block, cfg)
        return next(f for f in rows if f["key"] == "launch_cmd")

    def test_a_found_launcher_is_reported(self, tmp_path):
        script = tmp_path / ("run.bat" if comfy_client.os.name == "nt" else "run.sh")
        script.write_text("echo", encoding="utf-8")
        found = comfy_client.discover_launch_cmd(tmp_path)
        field = self._field({"comfy": {"workdir": str(tmp_path)}}, {"comfy_target": "user"})
        assert found and field["auto"] == found
        assert "action" not in field

    def test_a_folder_with_no_launcher_offers_to_create_one(self, tmp_path):
        field = self._field({"comfy": {"workdir": str(tmp_path)}}, {"comfy_target": "user"})
        assert field["auto"] == ""
        assert field["action"]["endpoint"].startswith("/api/comfyui/create-launcher?workdir=")

    def test_a_network_path_is_not_probed(self, monkeypatch):
        probed = []
        monkeypatch.setattr("localm.image_gen.comfy.discover_launch_cmd",
                            lambda folder: probed.append(folder))
        field = self._field({"comfy": {"workdir": r"\\192.0.2.1\share"}}, {"comfy_target": "user"})
        assert "auto" not in field
        assert probed == []

    def test_localm_own_comfyui_gets_no_hint(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.media.managed_comfy.is_managed_comfy_installed", lambda: True)
        field = self._field({"comfy": {"workdir": str(tmp_path)}}, {"comfy_target": "own"})
        assert "auto" not in field
        assert "action" not in field

    def test_a_lookup_failure_is_logged_not_raised(self, tmp_path, monkeypatch):
        logged = []
        monkeypatch.setattr("localm.debuglog.logger.debug",
                            lambda fmt, *a: logged.append(fmt % a))

        def boom(_folder):
            raise OSError("no access")

        monkeypatch.setattr("localm.image_gen.comfy.discover_launch_cmd", boom)
        field = self._field({"comfy": {"workdir": str(tmp_path)}}, {"comfy_target": "user"})
        assert "auto" not in field
        assert any("no access" in m for m in logged)
