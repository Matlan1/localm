# SPDX-License-Identifier: AGPL-3.0-or-later
"""comfy_models_dest_dir(): destination routing for a downloaded ComfyUI model
file. Must land wherever resolve_comfy_target() says the ACTIVE ComfyUI instance
actually is - managed, external-with-workdir, or nowhere known-safe when external
mode has no comfy_workdir configured."""

from pathlib import Path

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _install_managed(home_dir: Path) -> mc.ManagedComfyPaths:
    # is_managed_comfy_installed() also requires the provisioning completion
    # marker, not just main.py plus venv.
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
    assert mc.is_managed_comfy_installed()
    return paths


class TestComfyModelsDestDir:
    def test_external_with_workdir(self, home):
        cfg_dict = {"managed_comfy_enabled": False, "comfy_workdir": r"D:\my\comfy"}
        dest = mc.comfy_models_dest_dir("unet", cfg_dict)
        assert dest == Path(r"D:\my\comfy") / "models" / "unet"

    def test_external_without_workdir_is_unroutable(self, home):
        cfg_dict = {"managed_comfy_enabled": False, "comfy_workdir": None}
        assert mc.comfy_models_dest_dir("unet", cfg_dict) is None

    def test_managed_config_flags_alone_are_not_enough(self, home):
        # managed_comfy_active() requires an actually installed instance, not just
        # the enabled flag plus target; without one this falls through to external.
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own"}
        assert mc.comfy_models_dest_dir("unet", cfg_dict) is None

    def test_managed_active_routes_to_comfyui_models(self, home):
        _install_managed(home)
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own"}
        dest = mc.comfy_models_dest_dir("clip", cfg_dict)
        assert dest == mc.managed_comfy_paths().models_dir / "clip"
        assert dest == home / "comfyui-models" / "clip"

    def test_managed_active_ignores_external_workdir(self, home):
        # When managed is active, route to the managed dir even if a comfy_workdir
        # is also configured; resolve_comfy_target() encodes that precedence.
        _install_managed(home)
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own",
                    "comfy_workdir": r"D:\some\other\comfy"}
        dest = mc.comfy_models_dest_dir("vae", cfg_dict)
        assert dest == home / "comfyui-models" / "vae"


class TestComfyModelsDestDirPerPlugin:
    """resolve_comfy_target()'s non-managed branch reads the per-plugin
    comfy.workdir field, threaded in through the `plugin` param of
    resolve_comfy_target()/comfy_models_dest_dir(), as well as the bare global
    comfy_workdir."""

    def test_plugin_only_workdir_resolves_with_the_plugin_arg(self, home):
        cfg_dict = {"managed_comfy_enabled": False,
                    "plugins": {"image": {"comfy": {"workdir": r"D:\my\comfy"}}}}
        dest = mc.comfy_models_dest_dir("unet", cfg_dict, plugin="image")
        assert dest == Path(r"D:\my\comfy") / "models" / "unet"

    def test_plugin_only_workdir_is_invisible_without_the_plugin_arg(self, home):
        # Without a plugin argument, a per-plugin-only value has no context to
        # resolve against. model_pull_comfy_source covers the no-context case by
        # trying all three plugins itself; this asserts what the bare function does.
        cfg_dict = {"managed_comfy_enabled": False,
                    "plugins": {"image": {"comfy": {"workdir": r"D:\my\comfy"}}}}
        assert mc.comfy_models_dest_dir("unet", cfg_dict) is None

    def test_plugin_workdir_wins_over_legacy_global(self, home):
        # Same precedence backend.py's settings() applies: a per-plugin override
        # beats the legacy global fallback.
        cfg_dict = {
            "comfy_workdir": r"D:\stale\global",
            "plugins": {"music": {"comfy": {"workdir": r"D:\deliberate\music-comfy"}}},
        }
        target = mc.resolve_comfy_target(cfg_dict, plugin="music")
        assert target.workdir == r"D:\deliberate\music-comfy"

    def test_plugin_with_no_override_falls_back_to_legacy_global(self, home):
        # A plugin with no per-plugin override still gets the legacy global fallback.
        cfg_dict = {"comfy_workdir": r"D:\shared\comfy", "plugins": {}}
        target = mc.resolve_comfy_target(cfg_dict, plugin="video")
        assert target.workdir == r"D:\shared\comfy"

    def test_managed_active_still_ignores_plugin_workdir(self, home):
        # The plugin param never leaks into the MANAGED branch; it stays absolute.
        _install_managed(home)
        cfg_dict = {"comfy_target": "own",
                    "plugins": {"image": {"comfy": {"workdir": r"D:\some\other\comfy"}}}}
        dest = mc.comfy_models_dest_dir("checkpoints", cfg_dict, plugin="image")
        assert dest == home / "comfyui-models" / "checkpoints"
