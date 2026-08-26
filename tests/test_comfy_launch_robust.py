# SPDX-License-Identifier: AGPL-3.0-or-later
"""ComfyUI launch robustness: derive the launcher's own folder as the working
directory, and honour the configurable cold-start timeout."""

import os
import sys
from unittest.mock import MagicMock, patch

from localm.image_gen import comfy
# ensure_comfy calls _comfy_alive as a bare global in comfy_client, so a test
# that stubs the reachability probe must patch it on comfy_client, not on the
# image_gen.comfy re-export. The helpers tested directly (discover_launch_cmd,
# _amd_rocm_launch_env, _derive_workdir_from_cmd, apply_fast_dequant) stay on
# the comfy re-export.
from localm.media import comfy_client
from localm.media import managed_comfy as mc


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
    monkeypatch.setenv("PATH", r"Z:\Windows\System32")
    env = comfy._amd_rocm_launch_env()
    assert env is not None
    assert env["PATH"].split(os.pathsep)[0] == str(rocm_bin)
    assert r"Z:\Windows\System32" in env["PATH"]


def test_amd_rocm_launch_env_noop_when_already_on_path(monkeypatch, tmp_path):
    rocm_bin = tmp_path / "bin"
    rocm_bin.mkdir()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", str(rocm_bin) + os.pathsep + r"Z:\Windows\System32")
    assert comfy._amd_rocm_launch_env() is None


def test_amd_rocm_launch_env_none_when_bin_missing(monkeypatch, tmp_path):
    # HIP_PATH set but no bin subdir under it -> nothing to add.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", r"Z:\Windows\System32")
    assert comfy._amd_rocm_launch_env() is None


def test_ensure_comfy_uses_configured_timeout(monkeypatch):
    # ComfyUI never comes up; with no launch ability the configurable timeout
    # is still resolved.
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
#  comfy_launch_wait_seconds - so a route-level timeout budget can read the   #
#  SAME number ensure_comfy will honour.                                      #
# --------------------------------------------------------------------------- #

def test_launch_wait_seconds_reads_configured_timeout():
    assert comfy_client.comfy_launch_wait_seconds({"comfy_launch_timeout": 120}) == 120


def test_launch_wait_seconds_defaults_to_300_when_unset():
    assert comfy_client.comfy_launch_wait_seconds({}) == 300


def test_launch_wait_seconds_floors_at_30():
    assert comfy_client.comfy_launch_wait_seconds({"comfy_launch_timeout": 5}) == 30


def test_launch_wait_seconds_falls_back_to_300_on_a_malformed_value():
    assert comfy_client.comfy_launch_wait_seconds({"comfy_launch_timeout": "not-a-number"}) == 300


def test_launch_wait_seconds_loads_config_when_none_given(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"comfy_launch_timeout": 90})
    assert comfy_client.comfy_launch_wait_seconds() == 90


def test_ensure_comfy_delegates_to_the_shared_helper(monkeypatch, tmp_path):
    """ensure_comfy's own wait_seconds resolution must actually CALL
    comfy_launch_wait_seconds() rather than re-implement the same logic
    separately - a route computing its timeout budget from the helper is
    only safe if ensure_comfy is PROVABLY calling the same code, not just
    producing the same number today by coincidence that could silently
    drift apart on a future edit to either side."""
    import time as _time_mod

    # A URL unused by any other test in this file/process - _comfy_alive
    # results are cached module-globally (mark_comfy_alive/_confirmed_alive),
    # so a shared URL could let an earlier test's confirmation short-circuit
    # this one before it ever reaches the code path under test.
    url = "http://127.0.0.1:8199"

    calls = []
    real_helper = comfy_client.comfy_launch_wait_seconds

    def _spy(cfg=None):
        calls.append(cfg)
        return real_helper(cfg)

    monkeypatch.setattr(comfy_client, "comfy_launch_wait_seconds", _spy)

    # False on the pre-launch check, True once inside the post-launch poll
    # loop - so it succeeds on the loop's first iteration with no real wait.
    alive_calls = {"n": 0}

    def _alive(api_url, timeout=3.0):
        # > 2, not > 1: the double-checked lock adds a RE-CHECK under
        # _launch_lock_for() between the pre-lock probe and the post-spawn poll
        # loop - three _comfy_alive() calls minimum before a launch is confirmed
        # up (dead, dead-under-lock, up).
        alive_calls["n"] += 1
        return alive_calls["n"] > 2

    monkeypatch.setattr(comfy_client, "_comfy_alive", _alive)
    monkeypatch.setattr(_time_mod, "sleep", lambda s: None)

    class _Proc:
        returncode = 0

        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: _Proc())

    cfg = {"comfy_launch_timeout": 45}
    with patch("localm.config.load_config", return_value=cfg):
        ok, msg = comfy.ensure_comfy(url, launch_cmd="echo hi", workdir=str(tmp_path))
    assert ok is True, msg
    assert calls, "ensure_comfy did not call comfy_launch_wait_seconds at all"
    assert calls[0] == cfg


# --------------------------------------------------------------------------- #
#  Launcher auto-discovery from the ComfyUI folder (work with the user's setup) #
# --------------------------------------------------------------------------- #

def _ext():
    import os
    return "bat" if os.name == "nt" else "sh"


def test_discover_prefers_user_launcher(tmp_path):
    # When both a custom launch-comfyui.* and the stock comfyui.* exist, the
    # user's own launcher wins.
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
    # spawns it with the folder as cwd.
    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("echo hi\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}
    # dead, dead-under-lock (the double-checked re-check), then up after spawn.
    alive = iter([False, False, True])
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
    # with a non-zero code, ensure_comfy surfaces that failure immediately
    # instead of waiting out the whole cold-start timeout.
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


def test_ensure_comfy_immediate_exit_includes_the_launch_log_tail(tmp_path):
    """An immediate-exit failure must say WHY, not just the exit code - the real
    reason (a traceback) is what the launcher wrote to its own captured output
    before dying, and the caller should not have to go find comfy-launch.log by
    hand. Writes through the SAME stdout handle ensure_comfy itself opened and
    passed to Popen, so this proves the file is read back rather than that some
    string was appended."""
    from localm.config import home_dir
    home_dir().mkdir(parents=True, exist_ok=True)  # ensure_comfy expects this to exist

    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("exit 1\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}

    def fake_popen(argv, cwd=None, stdout=None, **kw):
        if hasattr(stdout, "write"):
            stdout.write("Traceback (most recent call last):\n"
                         "ModuleNotFoundError: No module named 'sqlalchemy'\n")
            stdout.flush()
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: False), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "exited immediately with code 1" in msg
    assert "ModuleNotFoundError" in msg
    assert "sqlalchemy" in msg


def test_ensure_comfy_timeout_message_includes_the_launch_log_tail(monkeypatch, tmp_path):
    """The "did not come up within N minutes" message already names the log FILE;
    it must also fold in the file's own tail so the reason is visible without a
    second trip to disk. Fast-forwards the deadline poll loop to zero iterations
    via a fake monotonic clock rather than waiting out a real 30s timeout."""
    import time as time_mod
    from localm.config import home_dir
    home_dir().mkdir(parents=True, exist_ok=True)  # ensure_comfy expects this to exist

    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("echo hi\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}

    def fake_popen(argv, cwd=None, stdout=None, **kw):
        if hasattr(stdout, "write"):
            stdout.write("aiohttp.client_exceptions.ClientConnectorError\n")
            stdout.flush()
        proc = MagicMock()
        proc.poll.return_value = None   # never exits - times out instead
        return proc

    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        # 1st call sets the deadline (0 + wait_seconds); every call after must
        # already be past it, so the poll loop body runs zero times.
        return 0.0 if calls["n"] == 1 else 10_000.0

    monkeypatch.setattr(time_mod, "monotonic", fake_monotonic)
    monkeypatch.setattr(time_mod, "sleep", lambda s: None)

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: False), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "did not come up within" in msg
    assert "aiohttp.client_exceptions.ClientConnectorError" in msg


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
    # dead, dead-under-lock (the double-checked re-check), then up after spawn.
    alive = iter([False, False, True])
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
    # comfy_disable_auto_launch=True -> the launch command gets the flag so
    # ComfyUI starts headless instead of opening its own web page.
    argv = _spawn_with_cfg(tmp_path, {"comfy_disable_auto_launch": True})
    assert "--disable-auto-launch" in str(argv)


def test_disable_auto_launch_absent_by_default(tmp_path):
    # NEGATIVE case: unset (and explicit False) must not append the flag.
    argv_unset = _spawn_with_cfg(tmp_path, {})
    assert "--disable-auto-launch" not in str(argv_unset)
    argv_false = _spawn_with_cfg(tmp_path, {"comfy_disable_auto_launch": False})
    assert "--disable-auto-launch" not in str(argv_false)


def test_ensure_comfy_launches_the_managed_instance_when_active(tmp_path):
    """When localm's own managed ComfyUI is installed and selected, ensure_comfy
    must launch IT (its own venv + main.py): the managed install is a raw
    checkout with no bundled launcher script for discovery to find, so without
    managed routing it falls through to "not reachable, configure your own
    ComfyUI install" even with a working managed instance.

    Only managed_comfy_paths() is faked (not managed_comfy_launch_cmd() or
    managed_comfy_workdir() themselves), so the REAL command-building and
    quoting logic in managed_comfy.py actually runs and is asserted on."""
    comfy_client._confirmed_alive.clear()
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}
    managed_root = tmp_path / "comfyui"
    managed_root.mkdir()
    fake_paths = mc.ManagedComfyPaths(
        root=managed_root,
        models_dir=tmp_path / "comfyui-models",
        main_py=managed_root / "main.py",
        venv_python=managed_root / "venv" / "Scripts" / "python.exe",
        extra_model_paths=managed_root / "extra_model_paths.yaml",
    )
    # dead, dead-under-lock (the double-checked re-check), then up after spawn.
    alive = iter([False, False, True])
    spawned = {}

    def fake_popen(argv, cwd=None, **kw):
        spawned["argv"], spawned["cwd"] = argv, cwd
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive)), \
         patch("localm.media.managed_comfy.managed_comfy_active", return_value=True), \
         patch("localm.media.managed_comfy.managed_comfy_paths", return_value=fake_paths), \
         patch("subprocess.Popen", side_effect=fake_popen):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8189")

    assert ok is True, msg
    assert spawned["cwd"] == str(managed_root)
    # List MEMBERSHIP, not a stringified-list substring check: str(a_list) reprs
    # each element, escaping backslashes.
    argv = spawned["argv"]
    assert str(fake_paths.venv_python) in argv
    assert str(fake_paths.main_py) in argv
    assert "--listen" in argv and "127.0.0.1" in argv
    assert "--port" in argv and "8189" in argv


def test_ensure_comfy_caller_override_beats_managed_routing(tmp_path):
    """A caller that passes its OWN explicit workdir/launch_cmd (e.g. a
    per-plugin override) must win over managed routing - same "caller override
    wins" precedent default_api_url() already follows for the URL."""
    comfy_client._confirmed_alive.clear()
    own_launcher = tmp_path / f"comfyui.{_ext()}"
    own_launcher.write_text("echo hi\n", encoding="utf-8")
    # dead, dead-under-lock (the double-checked re-check), then up after spawn.
    alive = iter([False, False, True])
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


def test_managed_workdir_resolution_is_atomic_if_launch_cmd_raises(tmp_path):
    """If managed_comfy_workdir() succeeds but managed_comfy_launch_cmd() then
    raises, workdir must NOT stay pointed at the managed folder - it falls back
    to the ordinary (non-managed) resolution cleanly rather than leaving a
    workdir-with-no-matching-launch_cmd inconsistent state.

    managed_root is populated with a real main.py + venv/Scripts/python.exe
    (discover_launch_cmd's own fallback-detection targets) so this test
    DISCRIMINATES the two behaviours: with a leaked workdir,
    discover_launch_cmd(managed_root) wrongly succeeds and launches a ComfyUI
    missing --listen/--port (defaulting to the WRONG port, 8188, while
    ensure_comfy keeps polling 8189). An empty managed_root would make both
    outcomes read as "not reachable"."""
    comfy_client._confirmed_alive.clear()
    managed_root = tmp_path / "comfyui"
    (managed_root / "venv" / "Scripts").mkdir(parents=True)
    (managed_root / "main.py").write_text("# fake ComfyUI entry point\n", encoding="utf-8")
    (managed_root / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    # No comfy_workdir/comfy_launch_cmd configured either -> the fallback is
    # "no launch_cmd found", never a launch attempt against managed_root.
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}

    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", return_value=False), \
         patch("localm.media.managed_comfy.managed_comfy_active", return_value=True), \
         patch("localm.media.managed_comfy.managed_comfy_workdir",
               return_value=str(managed_root)), \
         patch("localm.media.managed_comfy.managed_comfy_launch_cmd",
               side_effect=RuntimeError("simulated failure")), \
         patch("subprocess.Popen") as popen:
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8189")

    assert ok is False
    assert "not reachable" in msg
    popen.assert_not_called()


def test_concurrent_ensure_comfy_calls_spawn_only_one_process(tmp_path):
    """Two independent triggers for the SAME api_url (a generate submission and
    the separate "Launch ComfyUI" button, at minimum) must not each independently
    decide ComfyUI is down and spawn a competing process. Fires two genuinely
    concurrent ensure_comfy() calls at a dead api_url, with a mocked
    slow-but-eventually-successful launch, and asserts only ONE subprocess is
    actually spawned - the double-checked _launch_lock_for() must serialize the
    decision, not just the bookkeeping."""
    import threading

    comfy_client._confirmed_alive.clear()
    launcher = tmp_path / f"comfyui.{_ext()}"
    launcher.write_text("echo hi\n", encoding="utf-8")
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": str(tmp_path),
           "comfy_launch_timeout": 30}
    # A URL unused by any other test in this module - the readiness cache is
    # process-global, so a shared URL could let a sibling test's leftover
    # confirmation short-circuit this one before it ever reaches the lock.
    url = "http://127.0.0.1:8196"

    spawn_count = {"n": 0}
    spawn_lock = threading.Lock()
    # Barrier so BOTH threads reach ensure_comfy()'s pre-lock aliveness check
    # at the same instant.
    start_barrier = threading.Barrier(2)

    def fake_popen(argv, cwd=None, **kw):
        with spawn_lock:
            spawn_count["n"] += 1
        proc = MagicMock()
        proc.poll.return_value = None        # still running, not an immediate exit
        return proc

    def fake_alive(api_url, timeout=3.0):
        # "Up" only once something has actually been spawned - both callers'
        # PRE-LOCK check must see False (spawn_count starts at 0), so both
        # proceed to race for the lock; whichever wins spawns once, and the
        # loser's RE-CHECK under the lock then sees True without ever calling
        # Popen itself.
        with spawn_lock:
            return spawn_count["n"] > 0

    results = [None, None]

    def _call(i):
        start_barrier.wait(timeout=10)
        results[i] = comfy.ensure_comfy(url)

    # Patch ONCE from this (the main) thread, wrapping both workers' entire
    # run - never let each worker thread enter/exit its OWN `with patch(...)`
    # on the SAME targets. unittest.mock.patch's __enter__/__exit__ save and
    # restore a plain module attribute with no locking of their own, so two
    # threads independently patching the identical target race and can leave
    # the attribute permanently pointed at a mock. Patching once here means
    # both workers merely READ the same already-substituted functions.
    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", side_effect=fake_alive), \
         patch("subprocess.Popen", side_effect=fake_popen):
        threads = [threading.Thread(target=_call, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert all(not t.is_alive() for t in threads), (
        "a worker thread did not finish within the join timeout - "
        "the lock may be deadlocked")

    assert all(r is not None for r in results), f"a thread did not finish: {results}"
    assert results[0][0] is True, results[0]
    assert results[1][0] is True, results[1]
    # Exactly one of the two is the WINNER that reached _launch_and_wait()'s
    # poll loop ("ComfyUI is up."); the other found it already up on either its
    # pre-lock check or its re-check under the lock ("ComfyUI is running.") and
    # never spawned. Which one wins is non-deterministic; the 1-and-1 split is
    # not.
    messages = sorted(r[1] for r in results)
    assert messages == ["ComfyUI is running.", "ComfyUI is up."], messages
    assert spawn_count["n"] == 1, (
        f"expected exactly ONE subprocess spawn from two concurrent callers, "
        f"got {spawn_count['n']}")


def test_ensure_comfy_error_points_at_the_folder():
    cfg = {"comfy_launch_cmd": None, "comfy_workdir": None,
           "comfy_launch_timeout": 30}
    with patch("localm.config.load_config", return_value=cfg), \
         patch.object(comfy_client, "_comfy_alive", return_value=False):
        ok, msg = comfy.ensure_comfy("http://127.0.0.1:8188")
    assert ok is False
    assert "comfy_workdir" in msg          # guides the user to set the folder


# --------------------------------------------------------------------------- #
#  Fast GGUF dequant                                                           #
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
    # The committed template does not use the slow float32 dequant.
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
    
    # Call _spawn_launcher with subprocess.Popen mocked and check the argv it
    # spawns.
    cfg = {"comfy_launch_cmd": 'Z:\\path\\python.exe main.py --port 8188',
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
    # dead, dead-under-lock (the double-checked re-check), then up after spawn.
    alive_1 = iter([False, False, True])
    with patch("localm.config.load_config", return_value=cfg), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive_1)):
        comfy.ensure_comfy("http://127.0.0.1:8188")
        
    assert len(spawned) == 1
    assert spawned[0] == ['Z:\\path\\python.exe', 'main.py', '--port', '8188', '--disable-auto-launch']

    # On Windows, comfy.bat (batch file) should prepend cmd /d /c. Clear the
    # readiness cache the first call just set, else this short-circuits before
    # ever reaching subprocess.Popen.
    comfy_client._confirmed_alive.clear()
    cfg_bat = {"comfy_launch_cmd": 'Z:\\path\\comfy.bat --port 8188',
               "comfy_workdir": str(tmp_path), "comfy_launch_timeout": 30,
               "comfy_disable_auto_launch": True}
    spawned.clear()
    # dead, dead-under-lock, then up after spawn (see the note above).
    alive_2 = iter([False, False, True])
    with patch("localm.config.load_config", return_value=cfg_bat), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(comfy_client, "_comfy_alive", side_effect=lambda *a, **k: next(alive_2)):
        comfy.ensure_comfy("http://127.0.0.1:8188")
        
    assert len(spawned) == 1
    assert spawned[0] == ['cmd', '/d', '/c', 'Z:\\path\\comfy.bat', '--port', '8188', '--disable-auto-launch']

