# SPDX-License-Identifier: AGPL-3.0-or-later
"""comfy_models_dest_dir(): the destination-routing fix (gap B) for a downloaded
ComfyUI model file. Must land wherever resolve_comfy_target() says the ACTIVE
ComfyUI instance actually is - managed, external-with-workdir, or (correctly)
nowhere known-safe when external mode has no comfy_workdir configured."""

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
    # marker (not just main.py + venv) - see its docstring / test_managed_comfy_s1.py.
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
        # managed_comfy_active() requires an ACTUALLY installed instance, not
        # just the enabled flag + target - without one, this must still refuse
        # (fall through to external, which also has no workdir here) rather
        # than pretend the managed dir is active.
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own"}
        assert mc.comfy_models_dest_dir("unet", cfg_dict) is None

    def test_managed_active_routes_to_comfyui_models(self, home):
        _install_managed(home)
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own"}
        dest = mc.comfy_models_dest_dir("clip", cfg_dict)
        assert dest == mc.managed_comfy_paths().models_dir / "clip"
        assert dest == home / "comfyui-models" / "clip"

    def test_managed_active_ignores_external_workdir(self, home):
        # When managed IS active, route to the managed dir even if a
        # comfy_workdir also happens to be configured - resolve_comfy_target()
        # already encodes that precedence; this just checks we don't invent a
        # different one here.
        _install_managed(home)
        cfg_dict = {"managed_comfy_enabled": True, "comfy_target": "own",
                    "comfy_workdir": r"D:\some\other\comfy"}
        dest = mc.comfy_models_dest_dir("vae", cfg_dict)
        assert dest == home / "comfyui-models" / "vae"
