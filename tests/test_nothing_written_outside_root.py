# SPDX-License-Identifier: AGPL-3.0-or-later
"""A self-contained install writes nothing outside its own root folder.

The real server (`python -m localm gui --no-model --no-browser`) runs with its data
folder inside a source checkout (the portable layout) and with every place a
program could write to by default redirected to an empty folder: the user profile,
AppData, LocalAppData, the temp folders and the XDG folders. Afterwards:

  * every redirected folder is still empty (this also catches native code, which a
    Python-level hook cannot see), and
  * a Python audit hook that ran inside the server and every process it started
    recorded no file write, create, rename or directory creation outside the data
    folder, but for the record of running processes: it names the processes of the
    install, whatever its data folder is, and lies in the install's own folder.

A control run, with a child process that does write into a redirected folder, proves
both instruments can see a violation.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: See localm/procs.py. The one thing a server writes that is not in its data folder.
PROCESS_RECORD = ROOT / ".localm-run"

_HOOK = r'''
import os, sys
_log = os.environ.get("LOCALM_AUDIT_LOG")
if _log:
    _fd = os.open(_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    _WRITE = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def _emit(kind, path):
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            return
        try:
            os.write(_fd, (kind + "\t" + os.path.abspath(path) + "\n").encode("utf-8", "replace"))
        except OSError:
            pass

    def _hook(event, args):
        if event == "open":
            path, _mode, flags = args
            if isinstance(flags, int) and flags & _WRITE:
                _emit("write", path)
        elif event == "os.mkdir":
            _emit("mkdir", args[0])
        elif event in ("os.rename", "os.replace"):
            _emit("rename", args[1])

    sys.addaudithook(_hook)
'''

_IGNORED_PREFIXES = ("\\\\.\\", "//./", "/dev/", "/proc/", "/sys/")
_IGNORED_NAMES = {"nul", "con", "conout$", "conin$"}


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _violations(log: Path, allowed: list) -> list:
    """Recorded writes whose path is not under any *allowed* folder."""
    allowed = [_norm(a) for a in allowed]
    bad = []
    if not log.exists():
        return bad
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, _, path = line.partition("\t")
        if not path or path.startswith(_IGNORED_PREFIXES):
            continue
        if os.path.basename(path).lower() in _IGNORED_NAMES:
            continue
        n = _norm(path)
        if not any(n == a or n.startswith(a + os.sep) for a in allowed):
            bad.append((kind, path))
    return bad


def _listing(folder: Path) -> list:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


_POWERSHELL_CACHE = Path("AppData", "Local", "Microsoft", "Windows", "PowerShell",
                         "StartupProfileData-NonInteractive")
_ALLOWED_IN_PROFILE = (
    {str(_POWERSHELL_CACHE), *(str(p) for p in _POWERSHELL_CACHE.parents
                                if len(p.parts) > 2)}
    if os.name == "nt" else set())


def _new_entries(sandbox, name: str) -> list:
    """Entries added to a redirected folder since the fixture ran, without Windows
    PowerShell's own startup cache in the user profile (written by
    `powershell -NoProfile -Command`, which localm.hwdetect runs to read GPU names).
    See test_powershell_alone_writes_only_its_startup_cache."""
    new = set(_listing(sandbox["redirected"][name])) - set(sandbox["baseline"][name])
    if name == "userprofile":
        new -= _ALLOWED_IN_PROFILE
    return sorted(new)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop the server and every process it started, whatever they detached from."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=60)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def sandbox(tmp_path):
    """Redirected default-write locations, a data folder inside the checkout, and
    the audit hook."""
    redirected = {}
    for name in ("userprofile", "appdata", "localappdata", "temp", "xdg"):
        d = tmp_path / name
        d.mkdir()
        redirected[name] = d
    if os.name == "nt":
        (redirected["userprofile"] / "AppData" / "Roaming").mkdir(parents=True)
        (redirected["userprofile"] / "AppData" / "Local").mkdir()
    baseline = {name: _listing(d) for name, d in redirected.items()}
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(_HOOK, encoding="utf-8")
    log = tmp_path / "audit.log"
    scratch = ROOT / "scratch" / f"outside-root-{uuid.uuid4().hex[:8]}"
    home = scratch / "home"
    scratch.mkdir(parents=True)
    env = dict(os.environ)
    for key in ("HF_HOME", "LOCALM_TMPDIR", "LOCALM_PEER_DETECTION", "PYTHONHOME"):
        env.pop(key, None)
    env.update({
        "LOCALM_HOME": str(home),
        "USERPROFILE": str(redirected["userprofile"]), "HOME": str(redirected["userprofile"]),
        "APPDATA": str(redirected["appdata"]), "LOCALAPPDATA": str(redirected["localappdata"]),
        "TMP": str(redirected["temp"]), "TEMP": str(redirected["temp"]),
        "TMPDIR": str(redirected["temp"]),
        "XDG_CACHE_HOME": str(redirected["xdg"]), "XDG_CONFIG_HOME": str(redirected["xdg"]),
        "XDG_DATA_HOME": str(redirected["xdg"]), "XDG_STATE_HOME": str(redirected["xdg"]),
        "PYTHONPATH": os.pathsep.join([str(hook_dir), str(ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LOCALM_AUDIT_LOG": str(log),
    })
    try:
        yield {"env": env, "home": home, "redirected": redirected, "log": log,
               "hook_dir": hook_dir, "baseline": baseline}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            (ROOT / "scratch").rmdir()
        except OSError:
            pass


def _wait_for_whoami(port: int, proc: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"the server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/whoami", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise AssertionError("the server did not answer /whoami in time")


def test_the_instrument_sees_a_write_outside_the_root(sandbox):
    """The control: a child that writes into a redirected folder is caught by BOTH
    the audit hook and the empty-folder check."""
    target = sandbox["redirected"]["appdata"] / "stray.txt"
    r = subprocess.run(
        [sys.executable, "-c", f"open({str(target)!r}, 'w').write('x')"],
        env=sandbox["env"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    bad = _violations(sandbox["log"], [sandbox["home"]])
    assert [p for _k, p in bad if _norm(p) == _norm(target)], bad
    assert _listing(sandbox["redirected"]["appdata"]) == ["stray.txt"]


def test_the_record_of_processes_is_the_only_thing_allowed_beside_the_data_folder(sandbox):
    from localm import procs
    assert PROCESS_RECORD.name == procs.FOLDER
    log = sandbox["log"]
    log.write_text("".join(f"write\t{p}\n" for p in (
        PROCESS_RECORD / "x.proc", ROOT / "stray.txt", ROOT / ".localm-running" / "x")),
        encoding="utf-8")
    bad = _violations(log, [sandbox["home"], PROCESS_RECORD])
    assert [Path(p).name for _k, p in bad] == ["stray.txt", "x"]


def test_a_write_inside_the_profile_skeleton_is_still_caught(sandbox):
    """The profile folders a real Windows profile already has are not a write;
    a file put inside them is."""
    up = sandbox["redirected"]["userprofile"]
    target = up / "AppData" / "Roaming" / "stray.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    new = sorted(set(_listing(up)) - set(sandbox["baseline"]["userprofile"]))
    assert str(Path("AppData") / "Roaming" / "stray.txt") in new


@pytest.mark.skipif(os.name != "nt" or not shutil.which("powershell"),
                    reason="Windows PowerShell is a Windows program")
def test_powershell_alone_writes_only_its_startup_cache(sandbox):
    """Windows PowerShell started with the sandbox's profile writes nothing in the
    user profile beyond the entries the server check leaves out."""
    up = sandbox["redirected"]["userprofile"]
    r = subprocess.run(
        [shutil.which("powershell"), "-NoProfile", "-Command", "exit 0"],
        env=sandbox["env"], capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW)
    assert r.returncode == 0, r.stderr
    new = set(_listing(up)) - set(sandbox["baseline"]["userprofile"])
    assert new <= _ALLOWED_IN_PROFILE, sorted(new - _ALLOWED_IN_PROFILE)
    for name in ("appdata", "localappdata", "temp", "xdg"):
        assert _new_entries(sandbox, name) == [], name


@pytest.mark.skipif(os.name != "nt", reason="the allowance exists on Windows only")
def test_a_file_beside_the_powershell_cache_is_still_caught(sandbox):
    folder = sandbox["redirected"]["userprofile"] / _POWERSHELL_CACHE.parent
    folder.mkdir(parents=True)
    (folder / _POWERSHELL_CACHE.name).write_text("x", encoding="utf-8")
    (folder / "stray.txt").write_text("x", encoding="utf-8")
    assert _new_entries(sandbox, "userprofile") == [str(_POWERSHELL_CACHE.parent / "stray.txt")]


def test_a_running_server_writes_only_inside_its_own_root(sandbox):
    env, home = sandbox["env"], sandbox["home"]
    port = _free_port()
    out = sandbox["log"].parent / "server.out"
    proc = subprocess.Popen(
        [sys.executable, "-m", "localm", "gui", "--no-model", "--no-browser",
         "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=open(out, "w"), stderr=subprocess.STDOUT,
        **({"start_new_session": True} if os.name != "nt" else {}))
    try:
        _wait_for_whoami(port, proc)
        for path in ("/", "/v1/models", "/v1/instances/status"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20).read()
            except Exception:
                pass
    finally:
        _kill_tree(proc)

    assert home.is_dir(), "the server never created its data folder"
    assert (home / "tmp").is_dir(), "a self-contained install keeps its temp files in its own root"
    for name in sandbox["redirected"]:
        new = _new_entries(sandbox, name)
        assert new == [], f"{name} was written to: {new}"
    bad = _violations(sandbox["log"], [home, sandbox["log"].parent, PROCESS_RECORD])
    assert bad == [], f"writes outside the root: {bad[:20]}"
    assert sandbox["log"].exists() and sandbox["log"].stat().st_size > 0, (
        "the audit hook recorded nothing, so the check above proved nothing")
