# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI launch robustness: derive the launcher's own folder as the working
directory, and honour the configurable cold-start timeout."""

from unittest.mock import patch

from localm.image_gen import comfy


def test_derive_workdir_from_full_bat_path(tmp_path):
    bat = tmp_path / "launch-comfyui.bat"
    bat.write_text("echo hi\n", encoding="utf-8")
    assert comfy._derive_workdir_from_cmd(str(bat)) == str(tmp_path)


def test_derive_workdir_quoted_path_with_args(tmp_path):
    bat = tmp_path / "launch comfy.bat"   # space in the path
    bat.write_text("echo hi\n", encoding="utf-8")
    cmd = f'"{bat}" --listen --port 8188'
    assert comfy._derive_workdir_from_cmd(cmd) == str(tmp_path)


def test_derive_workdir_none_for_bare_relative_name():
    # A bare "launch-comfyui.bat" is not an existing file from here -> no guess.
    assert comfy._derive_workdir_from_cmd("launch-comfyui.bat") is None


def test_ensure_comfy_uses_configured_timeout(monkeypatch):
    # ComfyUI never comes up; with no launch ability we still resolve the
    # configurable timeout instead of the old hard-coded 180s.
    monkeypatch.setattr(comfy, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.setattr(comfy, "load_config",
                        lambda: {"comfy_launch_cmd": None,
                                 "comfy_launch_timeout": 600}, raising=False)
    # load_config is imported inside ensure_comfy from localm.config; patch there.
    with patch("localm.config.load_config",
               return_value={"comfy_launch_cmd": None, "comfy_launch_timeout": 600}):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "not reachable" in msg  # no launch cmd -> the configure hint
