# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applying an update brings the native llama.cpp runtime to the release's pin:
the decision (staged pins vs the installed marker, the user's pin or "latest"
choice, amd-rocm's own build), the re-provision after the swap, a runtime held
by another process (nothing changed, finished on the next start), a failed
re-provision (previous runtime put back, never reported as success), and a
rollback that restores the runtime it replaced."""

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _apply_update as au
from localm import _version, updater, updater_runtime
from tests._net_config import set_config
from tests._update_signing import envelope, keyset

_PRIV = Ed25519PrivateKey.generate()
_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]

_PINS_031 = ('_ROCM_TAG = "b1349"\n_ROCM_CPU_TAG = "b11719"\n'
             '_ROCM_BUILD = f"{_ROCM_TAG}-cpu-{_ROCM_CPU_TAG}"\n_PINNED_TAG = "b11757"\n')
_PINS_032 = ('_ROCM_TAG = "b1355"\n_ROCM_CPU_TAG = "b11908"\n'
             '_ROCM_BUILD = f"{_ROCM_TAG}-cpu-{_ROCM_CPU_TAG}"\n_PINNED_TAG = "b11928"\n')
_BIG = 70 * 1024


def _lib(root: Path) -> Path:
    return root / au.RUNTIME_LIB


def _provision(root: Path, backend: str, build, dll: bytes = b"old-dll") -> Path:
    """A provisioned runtime dir with a real marker, a library and the sentinel."""
    lib = _lib(root)
    lib.mkdir(parents=True, exist_ok=True)
    (lib / ".gitkeep").write_text("", encoding="utf-8")
    (lib / "llama.dll").write_bytes(dll * (_BIG // len(dll) + 1))
    (lib / ".localm-backend").write_text(f"{backend} {build}" if build else backend,
                                         encoding="utf-8")
    return lib


def _tree(root: Path, pins: str = _PINS_032, version: str = "0.3.2") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "localm" / "setup_llama").mkdir(parents=True, exist_ok=True)
    (root / "localm" / "setup_llama" / "pins.py").write_text(pins, encoding="utf-8")
    (root / "localm" / "setup_llama" / "cli.py").write_text('opt = "--keep-choice"\n',
                                                             encoding="utf-8")
    (root / "VERSION").write_text(version, encoding="utf-8")
    return root


def _fake_setup_llama(root: Path, build: str, *, backend=None, rc=0, calls=None):
    """A runner standing in for the `setup-llama` child: clears the runtime dir
    the way setup-llama does, then writes a new library and marker."""
    def run(cmd):
        if calls is not None:
            calls.append(list(cmd))
        if "setup-llama" not in cmd:
            return 0
        lib = _lib(root)
        for f in lib.iterdir():
            if f.is_file() and f.name not in (".gitkeep",):
                f.unlink()
        (lib / "llama.dll").write_bytes(b"new-dll" * (_BIG // 7 + 1))
        if rc == 0:
            used = backend or cmd[cmd.index("--backend") + 1]
            (lib / ".localm-backend").write_text(f"{used} {build}", encoding="utf-8")
        return rc
    return run


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A data dir and a mutable config (the user's runtime pin goes here)."""
    conf = {"bugreport_upload_url": "https://w"}
    set_config(monkeypatch, lambda: conf)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    return conf


# ------------------------------- the decision --------------------------------

def test_staged_pins_reads_the_pins_file_as_text(tmp_path):
    pins = updater_runtime.staged_pins(_tree(tmp_path / "s"))
    assert pins == {"pinned_tag": "b11928", "rocm_tag": "b1355", "rocm_cpu_tag": "b11908",
                    "rocm_build": "b1355-cpu-b11908"}


def test_staged_pins_of_an_older_single_file_layout(tmp_path):
    root = tmp_path / "s"
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "setup_llama.py").write_text(
        'X = 1\n_ROCM_TAG = "b1288"\n_PINNED_TAG = "b9000"\n', encoding="utf-8")
    pins = updater_runtime.staged_pins(root)
    assert pins["pinned_tag"] == "b9000"
    assert pins["rocm_build"] == "b1288", "before _ROCM_BUILD the marker recorded _ROCM_TAG"


def test_a_newer_upstream_pin_reprovisions_the_installed_backend(tmp_path, cfg):
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "vulkan", "b11757")
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert (p["reprovision"], p["backend"], p["current"], p["target"]) == (
        True, "vulkan", "b11757", "b11928")
    assert "b11757 to b11928" in p["note"]


def test_the_same_pin_needs_nothing(tmp_path, cfg):
    inst = _tree(tmp_path / "inst")
    _provision(inst, "vulkan", "b11928")
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is False and p["note"] is None


def test_amd_rocm_compares_its_own_build_not_the_upstream_pin(tmp_path, cfg):
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "amd-rocm", "b1349-cpu-b11719")
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is True
    assert p["target"] == "b1355-cpu-b11908"


def test_amd_rocm_whose_overlay_failed_for_the_staged_tag_is_current(tmp_path, cfg):
    inst = _tree(tmp_path / "inst")
    lib = _provision(inst, "amd-rocm", "b1355")
    (lib / ".localm-cpu-overlay-failed").write_text('{"tag": "b11908"}', encoding="utf-8")
    assert updater_runtime.plan(_tree(tmp_path / "s"), inst)["reprovision"] is False
    (lib / ".localm-cpu-overlay-failed").write_text('{"tag": "b11719"}', encoding="utf-8")
    assert updater_runtime.plan(_tree(tmp_path / "s"), inst)["reprovision"] is True


def test_a_user_pin_is_kept_and_named(tmp_path, cfg):
    cfg["llama_runtime_pin"] = "b11600"
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "vulkan", "b11600")
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is False
    assert p["note"] == "kept your pinned llama.cpp build b11600; this release ships b11928"


def test_a_pin_does_not_apply_to_amd_rocm(tmp_path, cfg):
    cfg["llama_runtime_pin"] = "b11600"
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "amd-rocm", "b1349-cpu-b11719")
    assert updater_runtime.plan(_tree(tmp_path / "s"), inst)["reprovision"] is True


def test_tracking_latest_is_kept_without_a_network_call(tmp_path, cfg, monkeypatch):
    cfg["llama_runtime_pin"] = "latest"
    import localm.setup_llama as sl
    monkeypatch.setattr(sl, "_latest_tag", lambda: pytest.fail("no network for the update"))
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "vulkan", "b11999")
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is False
    assert "follow upstream's newest" in p["note"] and "b11928" in p["note"]


def test_an_unrecorded_build_is_left_alone_and_said(tmp_path, cfg):
    inst = _tree(tmp_path / "inst", _PINS_031)
    _provision(inst, "vulkan", None)
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is False
    assert "does not record its build" in p["note"]
    assert "localm setup-llama --backend vulkan --force" in p["note"]


def test_nothing_provisioned_needs_nothing(tmp_path, cfg):
    inst = _tree(tmp_path / "inst", _PINS_031)
    p = updater_runtime.plan(_tree(tmp_path / "s"), inst)
    assert p["reprovision"] is False and p["note"] is None


def test_check_runtime_update_and_the_updater_share_one_target(tmp_path, cfg, monkeypatch):
    import localm.setup_llama as sl
    from localm.setup_llama import versions
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: _lib(tmp_path / "inst"))
    _provision(tmp_path / "inst", "vulkan", "b11757")
    assert versions.check_runtime_update()["target"] == versions.runtime_target("vulkan")[0]
    assert versions.runtime_target("amd-rocm", pin="b1")[1] == "release"
    assert versions.runtime_target("vulkan", pin="b1") == ("b1", "pin")
    assert versions.runtime_target("vulkan", tracking=True) == (None, "latest")


# ------------------------------ apply end to end -----------------------------

def _zip(version, files, deps='"click"'):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", f'[project]\nname = "localm"\ndependencies = [{deps}]\n')
        z.writestr("localm/__init__.py", f"# {version}")
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def _download(data):
    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener


@pytest.fixture
def install(tmp_path, cfg, monkeypatch):
    """An install at 0.3.1 with its pins and the amd-rocm b1349 runtime provisioned."""
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [_PRIV], _RECOVERY))
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.1")
    inst = _tree(tmp_path / "inst", _PINS_031, "0.3.1")
    (inst / "localm" / "__init__.py").write_text("# 0.3.1", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
    _provision(inst, "amd-rocm", "b1349-cpu-b11719")
    return inst


def _release(version="0.3.2", pins=_PINS_032, extra=None, deps='"click"'):
    files = {"localm/setup_llama/pins.py": pins,
             "localm/setup_llama/cli.py": 'opt = "--keep-choice"\n'}
    files.update(extra or {})
    data = _zip(version, files, deps)
    return data, envelope(data, [_PRIV])


def _marker(inst):
    return (_lib(inst) / ".localm-backend").read_text(encoding="utf-8")


def test_update_reprovisions_the_runtime_to_the_new_pin(install, tmp_path):
    data, sig = _release()
    calls = []
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=_fake_setup_llama(install, "b1355-cpu-b11908", calls=calls))
    assert res["klass"] == "runtime"
    assert res["runtime"]["state"] == "updated", res["runtime"]
    assert _marker(install) == "amd-rocm b1355-cpu-b11908"
    assert calls[0][1:] == ["-m", "localm", "setup-llama", "--backend", "amd-rocm",
                            "--force", "--yes", "--keep-choice"]
    assert calls[0][0] == sys.executable
    assert updater_runtime.read_pending() is None
    assert "b1349-cpu-b11719 -> b1355-cpu-b11908" in res["runtime_note"]


def test_deps_and_runtime_both_run_in_that_order(install, monkeypatch):
    monkeypatch.setattr(updater, "_uv_for_deps", lambda t: "/x/uv")
    data, sig = _release(deps='"click", "httpx"')
    calls = []
    res = updater.apply(1, signature=sig, installed=install,
                        download_opener=_download(data),
                        runner=_fake_setup_llama(install, "b1355-cpu-b11908", calls=calls))
    assert [c[0] for c in calls] == ["/x/uv", sys.executable]
    assert res["runtime"]["state"] == "updated"


def test_a_failed_reprovision_puts_the_old_runtime_back_and_says_so(install):
    before = au._tree_signature(_lib(install))
    data, sig = _release()
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=_fake_setup_llama(install, "x", rc=1))
    assert au._tree_signature(_lib(install)) == before
    assert res["runtime"]["state"] == "failed"
    assert "NOT updated" in res["runtime_note"] and "put back" in res["runtime_note"]
    assert (install / "VERSION").read_text(encoding="utf-8") == "0.3.2", "code update kept"
    assert updater_runtime.last_status()["state"] == "failed"


def test_a_fallback_to_another_backend_counts_as_failed(install):
    before = au._tree_signature(_lib(install))
    data, sig = _release()
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=_fake_setup_llama(install, "b11928", backend="vulkan"))
    assert res["runtime"]["state"] == "failed"
    assert "vulkan" in res["runtime"]["error"]
    assert au._tree_signature(_lib(install)) == before
    assert _marker(install) == "amd-rocm b1349-cpu-b11719"


def test_a_user_pin_is_never_reprovisioned_and_the_summary_names_it(install, cfg):
    _provision(install, "vulkan", "b11600")
    cfg["llama_runtime_pin"] = "b11600"
    data, sig = _release()
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=lambda c: pytest.fail(f"unexpected step {c}"))
    assert res["klass"] == "reboot"
    assert res["runtime_note"] == ("kept your pinned llama.cpp build b11600; this release "
                                   "ships b11928")


def test_the_server_defers_the_reprovision_to_its_restart(install):
    data, sig = _release()
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=lambda c: pytest.fail(f"unexpected step {c}"),
                        defer_runtime=True)
    assert res["klass"] == "runtime" and res["runtime"] is None
    assert res["runtime_note"].endswith("while localm restarts")
    pending = updater_runtime.read_pending()
    assert (pending["backend"], pending["target"]) == ("amd-rocm", "b1355-cpu-b11908")
    status = updater_runtime.finish_pending(
        root=install, runner=_fake_setup_llama(install, "b1355-cpu-b11908"))
    assert status["state"] == "updated"
    assert updater_runtime.read_pending() is None


def test_a_stale_pending_reprovision_is_dropped_by_an_update_that_needs_none(install):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  defer_runtime=True)
    assert updater_runtime.read_pending() is not None
    data, sig = _release("0.3.3", pins=_PINS_031)
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data))
    assert updater_runtime.read_pending() is None


def test_rollback_restores_the_runtime_the_update_replaced(install):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  runner=_fake_setup_llama(install, "b1355-cpu-b11908"))
    assert _marker(install) == "amd-rocm b1355-cpu-b11908"
    updater.rollback_last(installed=install)
    assert (install / "VERSION").read_text(encoding="utf-8") == "0.3.1"
    assert _marker(install) == "amd-rocm b1349-cpu-b11719"
    assert (_lib(install) / "llama.dll").read_bytes().startswith(b"old-dll")


def test_rollback_after_an_update_without_a_runtime_step_leaves_the_runtime(install):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  runner=_fake_setup_llama(install, "b1355-cpu-b11908"))
    data, sig = _release("0.3.3")
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  runner=lambda c: pytest.fail(f"unexpected step {c}"))
    assert not au.runtime_snapshot_dir(updater._backup_dir()).exists(), \
        "the older update's runtime copy goes with its backup"
    updater.rollback_last(installed=install)
    assert (install / "VERSION").read_text(encoding="utf-8") == "0.3.2"
    assert _marker(install) == "amd-rocm b1355-cpu-b11908"


def test_a_downgrade_returns_the_runtime_to_the_old_pins_and_names_what_it_keeps(install,
                                                                                 monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    _provision(install, "amd-rocm", "b1355-cpu-b11908")
    (install / "VERSION").write_text("0.3.2", encoding="utf-8")
    old_updater = '_SIG_FORMAT = "localm-update-sig/2"\n\n\ndef verify_signature(d, s):\n    return None\n'
    data, sig = _release("0.3.1", pins=_PINS_031, extra={"localm/updater.py": old_updater})
    res = updater.apply(1, signature=sig, target_version="0.3.1", installed=install,
                        download_opener=_download(data),
                        runner=_fake_setup_llama(install, "b1349-cpu-b11719"))
    assert res["downgrade"] is True
    assert res["runtime"]["state"] == "updated"
    assert _marker(install) == "amd-rocm b1349-cpu-b11719"
    assert res["kept_for_updates"] == ["this version's update signature check and trusted keys"]


# ------------------------- a runtime held by a process -----------------------

def _hold(path: Path):
    """A real second process holding *path* the way a loaded runtime is held:
    loaded as a DLL on Windows, mapped on Linux."""
    if sys.platform == "win32":
        code = ("import ctypes, os, sys; ctypes.CDLL(sys.argv[1]); "
                "print('ready', os.getpid(), flush=True); sys.stdin.read()")
    else:
        code = ("import mmap, sys; f = open(sys.argv[1], 'rb'); "
                "m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); import os; "
                "print('ready', os.getpid(), flush=True); sys.stdin.read()")
    p = subprocess.Popen([sys.executable, "-I", "-c", code, str(path)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    word, pid = p.stdout.readline().split()
    assert word == "ready"
    p.holder_pid = int(pid)
    return p


def _release_hold(p):
    p.stdin.close()
    p.wait(timeout=30)


def _real_library(lib: Path) -> Path:
    """Put a real loadable library at lib/llama.dll."""
    if sys.platform == "win32":
        import _socket
        src = Path(_socket.__file__)
    else:
        src = Path(sys.executable).resolve()
    dst = lib / "llama.dll"
    dst.write_bytes(src.read_bytes())
    return dst


@pytest.mark.skipif(not (sys.platform == "win32" or sys.platform.startswith("linux")),
                    reason="holder detection covers Windows and Linux")
def test_a_held_runtime_is_left_intact_and_finished_on_the_next_start(install, monkeypatch):
    held = _real_library(_lib(install))
    before = au._tree_signature(_lib(install))
    holder = _hold(held)
    try:
        data, sig = _release()
        res = updater.apply(1, signature=sig, installed=install,
                            download_opener=_download(data),
                            runner=lambda c: pytest.fail(f"must not provision while held: {c}"))
        assert res["runtime"]["state"] == "waiting"
        assert au._tree_signature(_lib(install)) == before
        assert any(h.get("pid") == holder.holder_pid for h in res["runtime"]["holders"]), \
            res["runtime"]["holders"]
        assert f"pid {holder.holder_pid}" in res["runtime_note"]
        assert "next start" in res["runtime_note"]
        assert updater_runtime.read_pending() is not None
    finally:
        _release_hold(holder)

    from localm.cli import _core
    calls = []
    fake = _fake_setup_llama(install, "b1355-cpu-b11908", calls=calls)
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    _core._finish_pending_runtime_update("ps")
    assert calls == [], "a command that loads no model does not finish it"
    _core._finish_pending_runtime_update("gui")
    assert calls and "setup-llama" in calls[0]
    assert _marker(install) == "amd-rocm b1355-cpu-b11908"
    assert updater_runtime.read_pending() is None


def test_the_restart_boot_does_not_start_a_download(install, monkeypatch):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  defer_runtime=True)
    from localm.cli import _core
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no provision"))
    _core._finish_pending_runtime_update("gui")
    assert updater_runtime.read_pending() is not None


@pytest.mark.skipif(not (sys.platform == "win32" or sys.platform.startswith("linux")),
                    reason="holder detection covers Windows and Linux")
def test_a_rollback_never_replaces_a_held_runtime(install):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  runner=_fake_setup_llama(install, "b1355-cpu-b11908"))
    held = _real_library(_lib(install))
    before = au._tree_signature(_lib(install))
    holder = _hold(held)
    try:
        with pytest.raises(RuntimeError) as ei:
            updater.rollback_last(installed=install)
    finally:
        _release_hold(holder)
    assert "native runtime not restored" in str(ei.value)
    assert au._tree_signature(_lib(install)) == before
    assert (install / "VERSION").read_text(encoding="utf-8") == "0.3.1", \
        "the code was still rolled back"


def test_the_restart_path_carries_out_a_pending_reprovision(install, monkeypatch):
    data, sig = _release()
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  defer_runtime=True)
    import localm.inference.backends.llamacpp._loader as loader
    import localm.inference.http_server as hs
    stopped = []
    monkeypatch.setattr(loader, "stop_probe_daemon", lambda *a, **k: stopped.append(1))
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    fake = _fake_setup_llama(install, "b1355-cpu-b11908")
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    hs._finish_pending_runtime_update()
    assert stopped == [1]
    assert _marker(install) == "amd-rocm b1355-cpu-b11908"
    assert updater_runtime.last_status()["state"] == "updated"


# ------------------------------------ doctor ---------------------------------

def test_doctor_flags_a_runtime_behind_the_release_pin(tmp_path, cfg, monkeypatch):
    import localm.setup_llama as sl
    import importlib
    doctor = importlib.import_module("localm.cli.doctor")
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: _lib(tmp_path / "inst"))
    _provision(tmp_path / "inst", "vulkan", "b11000")
    out = io.StringIO()
    from rich.console import Console
    monkeypatch.setattr(doctor, "console", Console(file=out, width=300))
    doctor._check_runtime_matches_release("b11000")
    text = out.getvalue()
    from localm.setup_llama.pins import _PINNED_TAG
    assert f"ships llama.cpp build {_PINNED_TAG} for vulkan, but b11000" in text
    assert "localm setup-llama --backend vulkan --force" in text
    out.truncate(0)
    out.seek(0)
    doctor._check_runtime_matches_release(_PINNED_TAG)
    assert "ships" not in out.getvalue()


def test_pending_marker_round_trip(tmp_path, cfg):
    updater_runtime.write_pending({"backend": "vulkan", "current": "a", "target": "b",
                                   "pins": {}}, root=tmp_path, version="9")
    data = updater_runtime.read_pending()
    assert (data["backend"], data["from"], data["target"]) == ("vulkan", "a", "b")
    assert json.loads((updater._updates_dir() / updater_runtime.PENDING_FILE)
                      .read_text(encoding="utf-8"))["version"] == "9"
    updater_runtime.clear_pending()
    assert updater_runtime.read_pending() is None
