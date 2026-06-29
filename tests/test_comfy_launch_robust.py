# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI launch robustness: derive the launcher's own folder as the working
directory, and honour the configurable cold-start timeout."""

from unittest.mock import MagicMock, patch

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


# --------------------------------------------------------------------------- #
#  Launcher auto-discovery from the ComfyUI folder (work with the user's setup) #
# --------------------------------------------------------------------------- #

def _ext():
    import os
    return "bat" if os.name == "nt" else "sh"


def test_discover_prefers_user_launcher(tmp_path):
    # When both a custom launch-comfyui.* and the stock comfyui.* exist, the
    # user's own launcher wins - localm uses the setup they already have.
    (tmp_path / f"comfyui.{_ext()}").write_text("stock\n", encoding="utf-8")
    (tmp_path / f"launch-comfyui.{_ext()}").write_text("mine\n", encoding="utf-8")
    cmd = comfy.discover_launch_cmd(tmp_path)
    assert cmd is not None
    assert "launch-comfyui" in cmd
    assert str(tmp_path) in cmd          # absolute, quoted


def test_discover_stock_launcher(tmp_path):
    (tmp_path / f"comfyui.{_ext()}").write_text("stock\n", encoding="utf-8")
    cmd = comfy.discover_launch_cmd(tmp_path)
    assert cmd is not None and "comfyui" in cmd


def test_discover_none_when_no_launcher(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing here\n", encoding="utf-8")
    assert comfy.discover_launch_cmd(tmp_path) is None


def test_discover_main_py_with_venv(tmp_path):
    import os
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")
    if os.name == "nt":
        venv = tmp_path / "venv" / "Scripts"
        py = venv / "python.exe"
    else:
        venv = tmp_path / "venv" / "bin"
        py = venv / "python"
    venv.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    cmd = comfy.discover_launch_cmd(tmp_path)
    assert cmd is not None
    assert "main.py" in cmd and str(py) in cmd


def test_ensure_comfy_discovers_launcher_in_workdir(tmp_path):
    # comfy_workdir set + no launch_cmd -> localm finds the launcher itself and
    # spawns it with the folder as cwd. The old code gave up ("not reachable").
    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("echo hi\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}
    alive = iter([False, True])     # dead, then up after spawn
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"], spawned["cwd"] = argv, cwd
        proc = MagicMock()
        proc.poll.return_value = None        # still running (not an immediate exit)
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is True, msg
    assert spawned["cwd"] == str(tmp_path)
    assert "comfyui" in str(spawned["argv"])


def _spawn_with_cfg(tmp_path, cfg):
    """Run ensure_comfy with a discoverable launcher in tmp_path and capture the
    spawned argv. Returns the argv (str on Windows, list on POSIX)."""
    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("echo hi\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30, **cfg}
    alive = iter([False, True])
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"] = argv
        proc = MagicMock()
        proc.poll.return_value = None        # still running (not an immediate exit)
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is True, msg
    return spawned["argv"]


def test_disable_auto_launch_appended_when_enabled(tmp_path):
    # MEDIA-2: comfy_disable_auto_launch=True -> launch command gets the flag so
    # ComfyUI starts headless instead of opening its own web page.
    argv = _spawn_with_cfg(tmp_path, {"comfy_disable_auto_launch": True})
    assert "--disable-auto-launch" in str(argv)


def test_disable_auto_launch_absent_by_default(tmp_path):
    # NEGATIVE case: unset (and explicit False) keep the current behavior, so the
    # flag must NOT be appended. This is what guards against changing the default.
    argv_unset = _spawn_with_cfg(tmp_path, {})
    assert "--disable-auto-launch" not in str(argv_unset)
    argv_false = _spawn_with_cfg(tmp_path, {"comfy_disable_auto_launch": False})
    assert "--disable-auto-launch" not in str(argv_false)


def test_ensure_comfy_error_points_at_the_folder():
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}
    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy, "_comfy_alive", return_value=False):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "comfy_workdir" in msg          # guides the user to set the folder


# --------------------------------------------------------------------------- #
#  Fast GGUF dequant (the 36 s/it -> ~6-7 s/it fix)                            #
# --------------------------------------------------------------------------- #

def test_apply_fast_dequant_rewrites_float32():
    wf = {"30": {"class_type": "UnetLoaderGGUFAdvanced",
                 "inputs": {"dequant_dtype": "float32"}}}
    assert comfy.apply_fast_dequant(wf) == 1
    assert wf["30"]["inputs"]["dequant_dtype"] == "default"


def test_apply_fast_dequant_leaves_explicit_choices():
    wf = {
        "a": {"class_type": "UnetLoaderGGUF", "inputs": {"dequant_dtype": "float16"}},
        "b": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {"dequant_dtype": "bfloat16"}},
        "c": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {"dequant_dtype": "default"}},
        "d": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    }
    assert comfy.apply_fast_dequant(wf) == 0
    assert wf["a"]["inputs"]["dequant_dtype"] == "float16"
    assert wf["b"]["inputs"]["dequant_dtype"] == "bfloat16"


def test_shipped_example_workflow_uses_fast_dequant():
    # The committed template must not regress to the slow float32 dequant.
    import json
    wf = json.loads(comfy._WORKFLOW_EXAMPLE_PATH.read_text(encoding="utf-8"))
    loaders = [n for n in wf.values()
               if n.get("class_type") in comfy._GGUF_UNET_LOADERS]
    assert loaders, "example should load the UNet via a GGUF loader"
    for n in loaders:
        assert n["inputs"].get("dequant_dtype") != "float32"
