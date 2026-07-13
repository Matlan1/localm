# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI launch robustness: derive the launcher's own folder as the working
directory, and honour the configurable cold-start timeout."""

import os
import sys
from unittest.mock import MagicMock, patch

from localm.image_gen import comfy
# ensure_comfy and the launcher/discovery helpers moved into the shared client.
# ensure_comfy calls _comfy_alive as a bare global in this module, so a test that
# stubs the reachability probe must patch it on comfy_client (its home), not on
# the image_gen.comfy re-export. (Helpers tested directly - discover_launch_cmd,
# _amd_rocm_launch_env, _derive_workdir_from_cmd, apply_fast_dequant - stay on the
# comfy re-export and need no change.)
from localm.media import comfy_client


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


# --- ROCm/bin on PATH for a ZLUDA ComfyUI launch (cublas64_11.dll dependency) ---

def test_amd_rocm_launch_env_none_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert comfy._amd_rocm_launch_env() is None


def test_amd_rocm_launch_env_none_without_hip_path(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HIP_PATH", raising=False)
    assert comfy._amd_rocm_launch_env() is None


def test_amd_rocm_launch_env_prepends_rocm_bin(monkeypatch, tmp_path):
    rocm_bin = tmp_path / "bin"
    rocm_bin.mkdir()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    env = comfy._amd_rocm_launch_env()
    assert env is not None
    assert env["PATH"].split(os.pathsep)[0] == str(rocm_bin)
    assert r"C:\Windows\System32" in env["PATH"]


def test_amd_rocm_launch_env_noop_when_already_on_path(monkeypatch, tmp_path):
    rocm_bin = tmp_path / "bin"
    rocm_bin.mkdir()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", str(rocm_bin) + os.pathsep + r"C:\Windows\System32")
    assert comfy._amd_rocm_launch_env() is None


def test_amd_rocm_launch_env_none_when_bin_missing(monkeypatch, tmp_path):
    # HIP_PATH set but no bin subdir under it -> nothing to add.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    assert comfy._amd_rocm_launch_env() is None


def test_ensure_comfy_uses_configured_timeout(monkeypatch):
    # ComfyUI never comes up; with no launch ability we still resolve the
    # configurable timeout instead of the old hard-coded 180s.
    monkeypatch.setattr(comfy_client, "_comfy_alive", lambda *a, **k: False)
    monkeypatch.setattr(comfy_client, "load_config",
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
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is True, msg
    assert spawned["cwd"] == str(tmp_path)
    assert "comfyui" in str(spawned["argv"])


def test_ensure_comfy_reports_launcher_immediate_exit(tmp_path):
    # The spawn exit-guard (comfy.py): if the launcher dies right after spawn
    # with a non-zero code (broken .bat, missing venv, bad path), ensure_comfy
    # must surface that failure NOW instead of waiting out the whole cold-start
    # timeout. This is the FIRING side of the guard the success tests step over
    # (where poll() is None == still running).
    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("exit 1\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}

    def fake_popen(argv, cwd=None, **kw):
        proc = MagicMock()
        proc.poll.return_value = 1        # already exited (not None)
        proc.returncode = 1              # with a non-zero code
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: False), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "exited immediately with code 1" in msg


def _spawn_with_cfg(tmp_path, cfg):
    """Run ensure_comfy with a discoverable launcher in tmp_path and capture the
    spawned argv. Returns the argv (str on Windows, list on POSIX).

    Clears the readiness cache first: callers (e.g. the two back-to-back
    scenarios in test_disable_auto_launch_absent_by_default) reuse the same
    hardcoded 127.0.0.1:8188 URL to test independent launch attempts, which
    would otherwise short-circuit on the SECOND call via the cache the first
    call just populated (see conftest.py's _reset_comfy_readiness_cache for
    the cross-TEST half of this same isolation concern)."""
    comfy_client._confirmed_alive.clear()
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
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
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


def test_ensure_comfy_launches_the_managed_instance_when_active(tmp_path):
    """#621 follow-up: when localm's own managed ComfyUI is installed and
    selected, ensure_comfy must launch IT (its own venv + main.py) - the
    managed install is a raw checkout with no bundled launcher script for
    discovery to find, so before this fix it fell through to "not reachable,
    configure your own ComfyUI install" even with a working managed instance."""
    comfy_client._confirmed_alive.clear()
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}
    managed_root = tmp_path / "comfyui"
    managed_root.mkdir()
    managed_cmd = f'"{tmp_path / "venv" / "python.exe"}" "{managed_root / "main.py"}" --listen 127.0.0.1 --port 8189'
    alive = iter([False, True])
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"], spawned["cwd"] = argv, cwd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("localm.media.managed_comfy.managed_comfy_active", return_value=True), \
         patch("localm.media.managed_comfy.managed_comfy_workdir", return_value=str(managed_root)), \
         patch("localm.media.managed_comfy.managed_comfy_launch_cmd", return_value=managed_cmd), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8189")

    assert ok is True, msg
    assert spawned["cwd"] == str(managed_root)
    assert "main.py" in str(spawned["argv"])
    assert "8189" in str(spawned["argv"])


def test_ensure_comfy_caller_override_beats_managed_routing(tmp_path):
    """A caller that passes its OWN explicit workdir/launch_cmd (e.g. a
    per-plugin override) must win over managed routing - same "caller override
    wins" precedent default_api_url() already follows for the URL."""
    comfy_client._confirmed_alive.clear()
    own_launcher = tmp_path / f"comfyui.{_ext()}"
    own_launcher.write_text("echo hi\n", encoding="utf-8")
    alive = iter([False, True])
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"], spawned["cwd"] = argv, cwd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch("localm.config.load_config", return_value={"comfy_launch_timeout": 30}), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("localm.media.managed_comfy.managed_comfy_active", return_value=True), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188", workdir=str(tmp_path))

    assert ok is True, msg
    assert spawned["cwd"] == str(tmp_path)
    assert "comfyui" in str(spawned["argv"])


def test_ensure_comfy_error_points_at_the_folder():
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}
    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", return_value=False):
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


def test_comfy_launch_argv_safety(tmp_path, monkeypatch):
    import sys
    import subprocess
    from localm.media import comfy_client
    
    if not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    
    # We want to check how argv is calculated inside comfy_client.py
    # Since we can mock subprocess.Popen, let's call _spawn_launcher and check spawned argv
    cfg = {"comfy_launch_cmd": 'C:\\path\\python.exe main.py --port 8188',
           "comfy_workdir": str(tmp_path), "comfy_launch_timeout": 30,
           "comfy_disable_auto_launch": True}
    
    spawned = []
    def fake_popen(argv, **kw):
        spawned.append(argv)
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    # On Windows, python.exe should run directly as list (no cmd wrapper)
    monkeypatch.setattr(sys, "platform", "win32")
    alive_1 = iter([False, True])
    with patch("localm.config.load_config", return_value=cfg), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive_1)):
        comfy.ensure_comfy("http://127.0.0.1:8188")
        
    assert len(spawned) == 1
    assert spawned[0] == ['C:\\path\\python.exe', 'main.py', '--port', '8188', '--disable-auto-launch']

    # On Windows, comfy.bat (batch file) should prepend cmd /d /c. A second,
    # independent ensure_comfy() attempt at the same URL - clear the
    # readiness cache the first call just set, else this short-circuits
    # before ever reaching subprocess.Popen (see the note on _spawn_with_cfg
    # above for the same isolation concern).
    comfy_client._confirmed_alive.clear()
    cfg_bat = {"comfy_launch_cmd": 'C:\\path\\comfy.bat --port 8188',
               "comfy_workdir": str(tmp_path), "comfy_launch_timeout": 30,
               "comfy_disable_auto_launch": True}
    spawned.clear()
    alive_2 = iter([False, True])
    with patch("localm.config.load_config", return_value=cfg_bat), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive_2)):
        comfy.ensure_comfy("http://127.0.0.1:8188")
        
    assert len(spawned) == 1
    assert spawned[0] == ['cmd', '/d', '/c', 'C:\\path\\comfy.bat', '--port', '8188', '--disable-auto-launch']

