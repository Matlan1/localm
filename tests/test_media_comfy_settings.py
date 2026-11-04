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

    def test_the_settings_page_shows_default_for_an_old_stored_value(self):
        rows = ss.media_schema_json("video", {"comfy": {"float_type": "fp16"}}, {})
        field = next(f for f in rows if f["key"] == "float_type")
        assert field["value"] == "default"
        assert field["value"] in field["options"]

    def test_generators_import_the_writer(self):
        assert music_comfy.apply_weight_dtype is comfy_client.apply_weight_dtype
        assert video_comfy.apply_weight_dtype is comfy_client.apply_weight_dtype


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
        def probe(_folder):
            raise AssertionError("probed a UNC path")

        monkeypatch.setattr("localm.image_gen.comfy.discover_launch_cmd", probe)
        field = self._field({"comfy": {"workdir": r"\\192.0.2.1\share"}}, {"comfy_target": "user"})
        assert "auto" not in field

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
