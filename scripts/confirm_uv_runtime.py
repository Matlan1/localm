#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a uv release WORKS with localm before it is pinned.

localm's setup scripts install one uv release (Astral's package manager) from its
release installer, after checking the installer's sha256, and the Docker image
installs the Linux tarball of the same release. This script is what earns the
word "confirmed" for a candidate release, or re-proves the release the repo pins
today (``--current``).

CHECKS (``required`` ones decide the verdict; a required check that could not run
is SKIP and makes the verdict INCONCLUSIVE, never PASS):

  release_listing      required  the GitHub release lists uv-installer.sh,
                                 uv-installer.ps1 and the Linux tarball, each with
                                 a sha256 digest
  pins_match_release   required  ``--current`` only: all five pinned places name
                                 one version and each carries the digest of its
                                 asset; SKIP (not required) for a candidate
  installer_digests    required  the installers downloaded from the release URL
                                 setup uses hash to the API digests (and, with
                                 ``--current``, to the pinned values)
  isolation            required  every directory uv and the installer may write
                                 resolves under --workdir
  installer_run        required  the host's installer (uv-installer.ps1 under
                                 Windows PowerShell, uv-installer.sh elsewhere)
                                 runs from the workdir and installs uv
  release_binary       required  Windows only: the installed uv.exe is
                                 byte-identical to uv.exe in the digest-verified
                                 release zip; SKIP (not required) elsewhere
  version              required  ``uv --version`` equals the release
  venv_python          required  ``uv venv --python 3.12 --python-preference
                                 only-managed`` makes a venv whose interpreter is
                                 3.12 and lives under the workdir
  pip_install          required  ``uv pip install`` of a tiny pinned package
                                 into that venv imports
  lock_check           required  ``uv lock --check`` run by the candidate uv in
                                 this repo reports the lock is current (CI runs
                                 the same command) and leaves uv.lock untouched
  containment          required  the user's real PATH registry value, uv
                                 receipt directory, user bin directory and Python
                                 registry keys are unchanged
  cleanup              optional  everything this run created is removed

ISOLATION. The installer runs with UV_UNMANAGED_INSTALL pointing into the
workdir, which the installer script itself maps to "do not modify PATH" and "do
not write an update receipt". LOCALAPPDATA, APPDATA, XDG_*, TEMP, TMP, TMPDIR and
every UV_* variable are replaced for the child, and uv's cache, managed Python,
tool and bin directories are set under the workdir. The real machine state is
read before and after and compared by the containment check. Only processes this
script starts are tracked by PID and are killed as a tree in a ``finally``.

OUTCOMES. Exit 0 = PASS, 1 = FAIL (the release is bad for localm), 2 =
INCONCLUSIVE (could not measure: network, download, missing PowerShell). An
uncaught exception is INCONCLUSIVE. The receipt JSON is written atomically on
every exit path; scripts/bump_uv_pin.py reads it.

Usage:
    python scripts/confirm_uv_runtime.py --tag 0.14.0 --workdir <scratch> --receipt r.json
    python scripts/confirm_uv_runtime.py --current --workdir <scratch> --receipt r.json

Needs localm importable (the verified HTTPS opener). Nothing under localm/
imports this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from typing import NamedTuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
VERDICT_PASS, VERDICT_FAIL, VERDICT_INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

IS_WINDOWS = os.name == "nt"
UV_NAME = "uv.exe" if IS_WINDOWS else "uv"
DOWNLOAD_URL = "https://github.com/astral-sh/uv/releases/download/%s/%s"
ZIP_ASSET = "uv-x86_64-pc-windows-msvc.zip"
PYTHON_SERIES = "3.12"
EXPECTED_REQUIRED = ("release_listing", "installer_digests", "isolation", "installer_run",
                     "containment", "version", "venv_python", "pip_install", "lock_check")
PIP_PACKAGE = ("six", "1.17.0")

_TAG_RE = re.compile(r"^\d+\.\d+\.\d+$")
_VERSION_RE = re.compile(r"^uv (\d+\.\d+\.\d+)\b")
_NETWORK_RE = re.compile(
    r"error sending request|dns error|failed to lookup|connection (?:reset|refused|closed|"
    r"timed out)|timed out|timeout|network|temporary failure|failed to download|"
    r"unable to connect|tls|handshake|certificate", re.I)
_LOCK_STALE_RE = re.compile(r"lockfile at .{0,20}uv\.lock.{0,20} needs to be updated", re.I)

# Variables removed from every child environment before the redirects are applied.
_DROPPED = {"GITHUB_PATH", "PSMODULEPATH", "PYTHONPATH", "VIRTUAL_ENV", "XDG_BIN_HOME",
            "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"}
_DROPPED_PREFIXES = ("UV_", "CARGO_DIST", "INSTALLER_")


def _bump():
    """The bump script's module: the one place the five pinned sites are defined."""
    path = Path(__file__).resolve().parent / "bump_uv_pin.py"
    spec = importlib.util.spec_from_file_location("bump_uv_pin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bump_uv_pin", mod)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
#  Receipt                                                                     #
# --------------------------------------------------------------------------- #

def verdict_of(checks: dict, expected: tuple = ()) -> str:
    """FAIL when any required check FAILed, else INCONCLUSIVE when any required
    check (or any name in *expected*) is not PASS or absent, else PASS."""
    required = [c for c in checks.values() if c.get("required")]
    if any(c.get("status") == FAIL for c in required):
        return VERDICT_FAIL
    missing = [n for n in expected if checks.get(n, {}).get("status") != PASS]
    if not required or missing or any(c.get("status") != PASS for c in required):
        return VERDICT_INCONCLUSIVE
    return VERDICT_PASS


def write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* through a temporary sibling and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Receipt:
    """The evidence one run collects: per-check status, the release asset
    digests the API published, and a note for an unexpected failure."""

    def __init__(self, tag: str, current: bool):
        self.tag = tag
        self.current = current
        self.checks: dict = {}
        self.assets: dict = {}
        self.listing: dict = {}
        self.note = ""

    def expected(self) -> tuple:
        return EXPECTED_REQUIRED + (("pins_match_release",) if self.current else ())

    def add(self, name: str, status: str, detail: str, *, required: bool = True) -> str:
        self.checks[name] = {"status": status, "required": required, "detail": detail}
        print(f"  [{status:4s}] {name}: {detail}", flush=True)
        return status

    def passed(self, name: str) -> bool:
        return self.checks.get(name, {}).get("status") == PASS

    def verdict(self) -> str:
        verdict = verdict_of(self.checks, self.expected())
        if self.note and verdict == VERDICT_PASS:
            return VERDICT_INCONCLUSIVE
        return verdict

    def why(self) -> str:
        verdict = self.verdict()
        if verdict == VERDICT_PASS:
            return "every required check passed"
        key = FAIL if verdict == VERDICT_FAIL else SKIP
        names = [n for n, c in self.checks.items() if c["required"] and c["status"] == key]
        names += [f"{n} (never ran)" for n in self.expected() if n not in self.checks]
        if self.note:
            names.append(self.note)
        return ("required check(s) FAILED: " if key == FAIL else "required check(s) not measured: "
                ) + "; ".join(names or ["no required check ran"])

    def to_json(self) -> dict:
        return {
            "schema": 1, "component": "uv", "tag": self.tag, "current": self.current,
            "verdict": self.verdict(), "why": self.why(),
            "written_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hardware": {"os": platform.platform(), "machine": platform.machine(),
                         "python": platform.python_version(), "cpus": os.cpu_count()},
            "checks": self.checks, "assets": self.assets,
        }

    def write(self, path: Path) -> None:
        write_atomic(path, json.dumps(self.to_json(), indent=2, default=str))

    def exit_code(self) -> int:
        return {VERDICT_PASS: 0, VERDICT_FAIL: 1, VERDICT_INCONCLUSIVE: 2}[self.verdict()]


# --------------------------------------------------------------------------- #
#  Processes and files                                                         #
# --------------------------------------------------------------------------- #

class Result(NamedTuple):
    """The outcome of one child process: exit code (None on timeout), combined
    output, and whether it was killed for running too long."""

    rc: int | None
    out: str
    timed_out: bool = False


def run_process(cmd: list, env: dict, cwd: Path | None, timeout: int) -> Result:
    """Run *cmd* to completion capturing stdout+stderr; kill its whole process
    tree by PID when it outlives *timeout* or this function is interrupted."""
    kwargs = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, env=env, cwd=str(cwd) if cwd else None,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kwargs)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return Result(proc.returncode, out.decode("utf-8", "replace"))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, _ = proc.communicate()
        return Result(None, out.decode("utf-8", "replace"), timed_out=True)
    finally:
        if proc.poll() is None:
            _kill_tree(proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=60)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def default_opener():
    """localm's verified HTTPS opener when localm is importable, else urllib's."""
    try:
        from localm.http_ssl import verified_urlopen
        return verified_urlopen
    except Exception:
        return urllib.request.urlopen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, opener, attempts: int = 3) -> str:
    """Download *url* to *dest*; returns the sha256 of the bytes written.
    Retried; raises the last error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        part = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "localm-confirm-uv"})
            h = hashlib.sha256()
            with opener(req, timeout=120) as resp, open(part, "wb") as f:
                for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                    h.update(chunk)
                    f.write(chunk)
            os.replace(part, dest)
            return h.hexdigest()
        except Exception as e:
            last = e
            part.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise last


def rmtree(path: Path) -> list:
    """Remove *path* (read-only files included); returns the failures."""
    failures = []

    def onerror(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError as e:
            failures.append(f"{p}: {e}")

    if path.exists():
        shutil.rmtree(path, onerror=onerror)
    return failures


# --------------------------------------------------------------------------- #
#  Workdir and environment                                                     #
# --------------------------------------------------------------------------- #

SUBDIRS = ("uv-bin", "cache", "python", "pybin", "tool", "toolbin", "venv", "tmp", "home",
           "dl", "appdata", "localappdata", "xdg-config", "xdg-data", "xdg-cache")


class Dirs:
    """The workdir and its named subdirectories (``d.uv_bin`` is ``uv-bin``)."""

    def __init__(self, root: Path):
        self.root = root
        self.created_root = False

    def __getattr__(self, name):
        sub = name.replace("_", "-")
        if sub in SUBDIRS:
            return self.root / sub
        raise AttributeError(name)

    def make(self) -> list:
        """Create every subdirectory; returns the ones that did not exist."""
        self.created_root = not self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        made = []
        for sub in SUBDIRS:
            p = self.root / sub
            if not p.exists():
                p.mkdir(parents=True)
                made.append(p)
        return made


def contained_env(base: dict, d: Dirs) -> dict:
    """*base* with every uv-related variable removed and every location uv, the
    installer and the temp directory may write redirected under the workdir."""
    env = {k: v for k, v in base.items()
           if k.upper() not in _DROPPED and not k.upper().startswith(_DROPPED_PREFIXES)}
    env.update({
        "UV_CACHE_DIR": str(d.cache), "UV_PYTHON_INSTALL_DIR": str(d.python),
        "UV_PYTHON_BIN_DIR": str(d.pybin), "UV_TOOL_DIR": str(d.tool),
        "UV_TOOL_BIN_DIR": str(d.toolbin), "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_REGISTRY": "0", "UV_SYSTEM_CERTS": "1",
        "UV_NO_MODIFY_PATH": "1", "UV_DISABLE_UPDATE": "1",
        "LOCALM_HOME": str(d.home), "TEMP": str(d.tmp), "TMP": str(d.tmp),
        "TMPDIR": str(d.tmp), "APPDATA": str(d.appdata), "LOCALAPPDATA": str(d.localappdata),
        "XDG_CONFIG_HOME": str(d.xdg_config), "XDG_DATA_HOME": str(d.xdg_data),
        "XDG_CACHE_HOME": str(d.xdg_cache),
    })
    if not IS_WINDOWS:
        env["HOME"] = str(d.home)
    return env


def installer_env(base: dict, d: Dirs) -> dict:
    """The contained environment plus UV_UNMANAGED_INSTALL for the installer."""
    env = contained_env(base, d)
    env["UV_UNMANAGED_INSTALL"] = str(d.uv_bin)
    return env


def _under(path: str, root: Path) -> bool:
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    root = root.resolve()
    return resolved == root or root in resolved.parents


def check_env_contained(env: dict, d: Dirs) -> list:
    """The names of environment variables that should point under the workdir and do not."""
    wanted = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "UV_TOOL_DIR",
              "UV_TOOL_BIN_DIR", "LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "APPDATA",
              "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")
    bad = [k for k in wanted if k not in env or not _under(env[k], d.root)]
    if "GITHUB_PATH" in env:
        bad.append("GITHUB_PATH")
    bad += [k for k in env if k.upper().startswith("UV_")
            and k not in {"UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR",
                          "UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "UV_PYTHON_INSTALL_BIN",
                          "UV_PYTHON_INSTALL_REGISTRY", "UV_SYSTEM_CERTS", "UV_NO_MODIFY_PATH",
                          "UV_DISABLE_UPDATE", "UV_UNMANAGED_INSTALL"}]
    return bad


# --------------------------------------------------------------------------- #
#  The real machine state the run must not change                              #
# --------------------------------------------------------------------------- #

def _entries(directory: Path, names: tuple) -> list:
    """(name, size, mtime) for each of *names* that exists in *directory*."""
    found = []
    for name in names:
        try:
            st = (directory / name).stat()
        except OSError:
            continue
        found.append((name, st.st_size, int(st.st_mtime)))
    return found


_UV_BINARIES = ("uv", "uvx", "uvw", "uv.exe", "uvx.exe", "uvw.exe")
_UV_RECEIPT = ("uv-receipt.json",)


def real_state(env: dict | None = None) -> dict:
    """What the installer or uv could change outside the workdir: the user PATH
    registry value, the uv binaries in the user bin directory, the installer's
    update receipt and the per-user Python registry keys (Windows); the uv
    binaries and receipt (elsewhere)."""
    env = os.environ if env is None else env
    home = Path(env.get("USERPROFILE") or env.get("HOME") or ".")
    state: dict = {"user_bin": _entries(home / ".local" / "bin", _UV_BINARIES)}
    if IS_WINDOWS:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                state["hkcu_path"] = winreg.QueryValueEx(k, "Path")[0]
        except OSError:
            state["hkcu_path"] = None
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Python") as k:
                names = []
                i = 0
                while True:
                    try:
                        names.append(winreg.EnumKey(k, i))
                    except OSError:
                        break
                    i += 1
                state["hkcu_python"] = sorted(names)
        except OSError:
            state["hkcu_python"] = None
        local = env.get("LOCALAPPDATA")
        state["localappdata_receipt"] = (
            _entries(Path(local) / "uv", _UV_RECEIPT) if local else None)
    else:
        cfg = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
        state["config_receipt"] = _entries(cfg / "uv", _UV_RECEIPT)
    return state


def state_diff(before: dict, after: dict) -> list:
    return [k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]


# --------------------------------------------------------------------------- #
#  Checks                                                                      #
# --------------------------------------------------------------------------- #

def classify_failure(output: str) -> str:
    """SKIP when *output* reads as a network problem, else FAIL."""
    return SKIP if _NETWORK_RE.search(output or "") else FAIL


def check_release_listing(r: Receipt, bump, tag: str, opener):
    """Fill r.assets; returns the release body, or None when it could not be read."""
    try:
        body = bump.fetch_release(tag, opener)
    except bump.ListingUnreadable as e:
        r.add("release_listing", SKIP, str(e))
        return None
    try:
        r.assets = {n: bump.asset_digest(body, n) for n in bump.ASSETS}
    except bump.Refused as e:
        r.add("release_listing", FAIL, str(e))
        return body
    r.add("release_listing", PASS,
          "the release lists " + ", ".join(f"{n} ({d[:12]}...)" for n, d in r.assets.items()))
    return body


def check_pins_match(r: Receipt, bump, root: Path, tag: str) -> None:
    """With --current: every pinned place names one version and carries the
    digest the API publishes for its asset at that version."""
    if not r.current:
        r.add("pins_match_release", SKIP, "candidate run: the pins are what a bump rewrites",
              required=False)
        return
    pins = {}
    try:
        for s in bump.SITES:
            text = (root / s.path).read_bytes().decode("utf-8").replace("\r\n", "\n")
            pins[s.path] = bump.read_site(s, text)
    except (OSError, bump.Refused) as e:
        r.add("pins_match_release", FAIL, f"the pins could not be read: {e}")
        return
    versions = {p: v for p, (v, _) in pins.items()}
    if len(set(versions.values())) != 1:
        r.add("pins_match_release", FAIL,
              "the five places pin different uv releases: "
              + ", ".join(f"{p}={v}" for p, v in versions.items()))
        return
    problems = []
    for s in bump.SITES:
        want = r.assets.get(s.asset)
        if want is None:
            r.add("pins_match_release", SKIP, f"no API digest for {s.asset} to compare with")
            return
        if pins[s.path][1] != want:
            problems.append(f"{s.path} pins {pins[s.path][1][:12]}... for {s.asset}, "
                            f"the API publishes {want[:12]}...")
    if problems:
        r.add("pins_match_release", FAIL, "; ".join(problems))
        return
    r.add("pins_match_release", PASS,
          f"all five places pin {tag} with the API digest of their asset")


def check_installer_digests(r: Receipt, bump, root: Path, tag: str, d: Dirs, opener) -> dict:
    """Download both installers from the URL setup uses; returns {name: path}."""
    paths, problems = {}, []
    for name in (bump.ASSET_SH, bump.ASSET_PS1):
        dest = d.dl / name
        try:
            got = download(DOWNLOAD_URL % (tag, name), dest, opener)
        except Exception as e:
            r.add("installer_digests", SKIP, f"could not download {name}: {type(e).__name__}: {e}")
            return {}
        paths[name] = dest
        if got != r.assets[name]:
            problems.append(f"{name} hashes to {got[:12]}..., the API publishes "
                            f"{r.assets[name][:12]}...")
    if r.current:
        for s in bump.SITES:
            if s.asset in paths:
                text = (root / s.path).read_bytes().decode("utf-8").replace("\r\n", "\n")
                pinned = bump.read_site(s, text)[1]
                if pinned != sha256_file(paths[s.asset]):
                    problems.append(f"{s.path} pins {pinned[:12]}..., the downloaded "
                                    f"{s.asset} hashes differently")
    if problems:
        r.add("installer_digests", FAIL, "; ".join(problems))
        return {}
    r.add("installer_digests", PASS,
          "uv-installer.sh and uv-installer.ps1 downloaded from the release hash to the "
          "API digests" + (" and to the pinned values" if r.current else ""))
    return paths


def check_isolation(r: Receipt, d: Dirs, env: dict) -> bool:
    bad = check_env_contained(installer_env(env, d), d)
    local = env.get("LOCALAPPDATA")
    inside_real = bool(local) and _under(str(d.root), Path(local) / "uv")
    if bad or inside_real:
        r.add("isolation", FAIL,
              "not contained: " + ", ".join(bad + (["workdir inside the real uv directory"]
                                                   if inside_real else [])))
        return False
    r.add("isolation", PASS,
          f"every cache, Python, tool, bin, temp, appdata and XDG location resolves under "
          f"{d.root}; GITHUB_PATH and inherited UV_* variables are removed")
    return True


def find_powershell() -> str | None:
    """Windows PowerShell (the shell setup.bat runs the installer with), or None."""
    return shutil.which("powershell")


def check_installer_run(r: Receipt, d: Dirs, env: dict, paths: dict, runner, bump):
    """Run the host's installer contained; returns the installed uv path or None."""
    cenv = installer_env(env, d)
    if IS_WINDOWS:
        shell = find_powershell()
        if not shell:
            r.add("installer_run", SKIP, "Windows PowerShell is not on PATH")
            return None
        cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(paths[bump.ASSET_PS1])]
    else:
        cmd = ["sh", str(paths[bump.ASSET_SH])]
    res = runner(cmd, cenv, d.root, 900)
    uv = d.uv_bin / UV_NAME
    if res.timed_out:
        r.add("installer_run", SKIP, "the installer did not finish within 900 seconds")
        return None
    tail = " | ".join((res.out or "").strip().splitlines()[-4:])
    if res.rc != 0:
        r.add("installer_run", classify_failure(res.out),
              f"the installer exited {res.rc}: {tail}")
        return None
    if not uv.is_file():
        r.add("installer_run", FAIL, f"the installer exited 0 but {uv} does not exist: {tail}")
        return None
    r.add("installer_run", PASS, f"{Path(cmd[0]).name} installer exited 0 and installed {uv.name} "
          f"into the workdir ({uv.stat().st_size} bytes)")
    return uv


def check_release_binary(r: Receipt, bump, tag: str, uv: Path, cache: Path, d: Dirs, opener):
    if not IS_WINDOWS:
        r.add("release_binary", SKIP, "compared against the Windows zip only", required=False)
        return
    sha = r.listing.get(ZIP_ASSET)
    if not sha:
        r.add("release_binary", SKIP, f"the listing has no sha256 digest for {ZIP_ASSET}")
        return
    zpath = cache / tag / ZIP_ASSET
    try:
        if not (zpath.is_file() and sha256_file(zpath) == sha):
            got = download(DOWNLOAD_URL % (tag, ZIP_ASSET), zpath, opener)
            if got != sha:
                zpath.unlink(missing_ok=True)
                r.add("release_binary", FAIL,
                      f"{ZIP_ASSET} hashes to {got[:12]}..., the API publishes {sha[:12]}...")
                return
    except Exception as e:
        r.add("release_binary", SKIP, f"could not fetch {ZIP_ASSET}: {type(e).__name__}: {e}")
        return
    with zipfile.ZipFile(zpath) as z:
        member = next((n for n in z.namelist() if Path(n).name == UV_NAME), None)
        if member is None:
            r.add("release_binary", FAIL, f"{ZIP_ASSET} contains no {UV_NAME}")
            return
        want = hashlib.sha256(z.read(member)).hexdigest()
    have = sha256_file(uv)
    if want == have:
        r.add("release_binary", PASS,
              f"installed {UV_NAME} (sha256 {have[:12]}...) is byte-identical to {UV_NAME} in the "
              f"digest-verified {ZIP_ASSET}")
    else:
        r.add("release_binary", FAIL,
              f"installed {UV_NAME} hashes to {have[:12]}..., the release zip's to {want[:12]}...")


def check_version(r: Receipt, uv: Path, tag: str, env: dict, d: Dirs, runner) -> None:
    res = runner([str(uv), "--version"], contained_env(env, d), d.root, 120)
    m = _VERSION_RE.match((res.out or "").strip())
    if res.rc == 0 and m and m.group(1) == tag:
        r.add("version", PASS, (res.out or "").strip())
    else:
        r.add("version", FAIL, f"expected 'uv {tag}', got rc={res.rc} {(res.out or '').strip()!r}")


def check_venv_python(r: Receipt, uv: Path, env: dict, d: Dirs, runner) -> Path | None:
    cenv = contained_env(env, d)
    res = runner([str(uv), "venv", "--python", PYTHON_SERIES, "--python-preference",
                  "only-managed", "--clear", str(d.venv)], cenv, d.root, 1200)
    if res.timed_out:
        r.add("venv_python", SKIP, "uv venv did not finish within 1200 seconds")
        return None
    py = d.venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if res.rc != 0 or not py.is_file():
        r.add("venv_python", classify_failure(res.out),
              f"uv venv exited {res.rc}: {' | '.join((res.out or '').strip().splitlines()[-4:])}")
        return None
    probe = runner([str(py), "-c", "import sys; print(sys.version.split()[0]); "
                    "print(sys.base_prefix)"], cenv, d.root, 120)
    lines = (probe.out or "").strip().splitlines()
    if probe.rc != 0 or len(lines) < 2:
        r.add("venv_python", FAIL, f"the venv interpreter did not run: {probe.out!r}")
        return None
    version, base = lines[0], lines[-1]
    if not version.startswith(PYTHON_SERIES + "."):
        r.add("venv_python", FAIL, f"the venv interpreter is {version}, expected {PYTHON_SERIES}")
        return None
    if not _under(base, d.python):
        r.add("venv_python", FAIL, f"the interpreter's base {base} is not under the managed "
              f"Python directory {d.python}")
        return None
    r.add("venv_python", PASS,
          f"uv venv --python {PYTHON_SERIES} --python-preference only-managed made a venv on "
          f"Python {version} managed under the workdir")
    return py


def check_pip_install(r: Receipt, uv: Path, py: Path, env: dict, d: Dirs, runner) -> None:
    name, version = PIP_PACKAGE
    cenv = contained_env(env, d)
    res = runner([str(uv), "pip", "install", "-p", str(py), f"{name}=={version}"],
                 cenv, d.root, 900)
    if res.timed_out:
        r.add("pip_install", SKIP, "uv pip install did not finish within 900 seconds")
        return
    if res.rc != 0:
        r.add("pip_install", classify_failure(res.out),
              f"uv pip install exited {res.rc}: {' | '.join((res.out or '').strip().splitlines()[-4:])}")
        return
    probe = runner([str(py), "-c", f"import {name}; print({name}.__version__)"],
                   cenv, d.root, 120)
    if probe.rc == 0 and (probe.out or "").strip() == version:
        r.add("pip_install", PASS, f"uv pip install {name}=={version} installed and imports")
    else:
        r.add("pip_install", FAIL, f"{name} did not import at {version} after install: "
              f"{(probe.out or '').strip()!r}")


def check_lock(r: Receipt, uv: Path, root: Path, env: dict, d: Dirs, runner) -> None:
    lock, project = root / "uv.lock", root / "pyproject.toml"
    try:
        before = (lock.read_bytes(), project.read_bytes())
    except OSError as e:
        r.add("lock_check", FAIL, f"cannot read uv.lock or pyproject.toml: {e}")
        return
    res = runner([str(uv), "lock", "--check"], contained_env(env, d), root, 1800)
    try:
        after = (lock.read_bytes(), project.read_bytes())
    except OSError:
        after = None
    if after != before:
        lock.write_bytes(before[0])
        project.write_bytes(before[1])
        r.add("lock_check", FAIL, "uv lock --check modified uv.lock or pyproject.toml "
              "(restored to the original bytes)")
        return
    tail = " | ".join((res.out or "").strip().splitlines()[-4:])
    if res.timed_out:
        r.add("lock_check", SKIP, "uv lock --check did not finish within 1800 seconds")
    elif res.rc == 0:
        r.add("lock_check", PASS, f"uv lock --check reports the lock is current: {tail}")
    elif _LOCK_STALE_RE.search(res.out or ""):
        r.add("lock_check", FAIL, f"this uv says uv.lock is out of date: {tail}")
    else:
        r.add("lock_check", classify_failure(res.out), f"uv lock --check exited {res.rc}: {tail}")


def check_containment(r: Receipt, before: dict, after: dict) -> None:
    changed = state_diff(before, after)
    if changed:
        r.add("containment", FAIL, "real machine state changed outside the workdir: "
              + ", ".join(changed))
    else:
        r.add("containment", PASS,
              "unchanged outside the workdir: " + ", ".join(sorted(before)))


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #

def confirm(tag: str | None, current: bool, workdir: Path, cache_dir: Path | None, *,
            root: Path = REPO, opener=None, runner=run_process, keep: bool = False,
            base_env: dict | None = None, state_reader=None) -> Receipt:
    """Run every check; returns the receipt. Never raises."""
    bump = _bump()
    base_env = dict(os.environ) if base_env is None else base_env
    opener = opener or default_opener()
    state_reader = state_reader or real_state
    d = Dirs(workdir.resolve())
    receipt = Receipt(tag or "unknown", current)
    created: list = []
    try:
        if current:
            try:
                text = (root / "setup.sh").read_bytes().decode("utf-8").replace("\r\n", "\n")
                receipt.tag = bump.read_site(bump.SITES[0], text)[0]
            except (OSError, bump.Refused) as e:
                receipt.add("pins_match_release", FAIL, f"the pinned version cannot be read: {e}")
                return receipt
        tag = receipt.tag
        print(f"Confirming uv {tag} ({'the pinned build' if current else 'candidate'}); "
              f"workdir {d.root}", flush=True)

        body = check_release_listing(receipt, bump, tag, opener)
        if body is None or not receipt.assets:
            receipt.add("pins_match_release", SKIP, "the release listing was not usable",
                        required=current)
            _skip_rest(receipt, "release_listing was not PASS")
            return receipt
        receipt.listing = {a["name"]: a["digest"].removeprefix("sha256:")
                           for a in body["assets"]
                           if isinstance(a, dict) and isinstance(a.get("name"), str)
                           and isinstance(a.get("digest"), str)}
        check_pins_match(receipt, bump, root, tag)

        created = d.make()
        before = state_reader(base_env)
        paths = check_installer_digests(receipt, bump, root, tag, d, opener)
        if not check_isolation(receipt, d, base_env):
            _skip_rest(receipt, "isolation was not PASS")
            return receipt
        if not paths:
            _skip_rest(receipt, "installer_digests was not PASS")
            return receipt
        uv = check_installer_run(receipt, d, base_env, paths, runner, bump)
        if uv is not None:
            cache = cache_dir if cache_dir else d.dl
            check_release_binary(receipt, bump, tag, uv, cache, d, opener)
            check_version(receipt, uv, tag, base_env, d, runner)
            py = check_venv_python(receipt, uv, base_env, d, runner)
            if py is not None:
                check_pip_install(receipt, uv, py, base_env, d, runner)
            else:
                receipt.add("pip_install", SKIP, "venv_python was not PASS")
            check_lock(receipt, uv, root, base_env, d, runner)
        else:
            for n in ("version", "venv_python", "pip_install", "lock_check"):
                receipt.add(n, SKIP, "installer_run was not PASS")
            receipt.add("release_binary", SKIP, "installer_run was not PASS", required=False)
        check_containment(receipt, before, state_reader(base_env))
        return receipt
    except Exception as e:
        traceback.print_exc()
        receipt.note = f"unexpected {type(e).__name__}: {e}"
        return receipt
    finally:
        if keep:
            print(f"Kept: {d.root}", flush=True)
        else:
            failures = []
            for p in created:
                failures += rmtree(p)
            if d.created_root and not any(d.root.iterdir() if d.root.exists() else []):
                d.root.rmdir()
            if created or failures:
                receipt.add("cleanup", FAIL if failures else PASS,
                            f"could not remove: {failures}" if failures
                            else f"removed {len(created)} directories this run created",
                            required=False)


def _skip_rest(receipt: Receipt, reason: str) -> None:
    for n in ("installer_digests", "isolation", "installer_run", "version", "venv_python",
              "pip_install", "lock_check", "containment"):
        if n not in receipt.checks:
            receipt.add(n, SKIP, reason)
    if "release_binary" not in receipt.checks:
        receipt.add("release_binary", SKIP, reason, required=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tag", help="the uv release to confirm, e.g. 0.14.0")
    which.add_argument("--current", action="store_true",
                       help="confirm the release the repo pins today")
    ap.add_argument("--workdir", required=True,
                    help="scratch directory; the run creates and removes its own subdirectories")
    ap.add_argument("--receipt", required=True, help="where the receipt JSON is written")
    ap.add_argument("--cache-dir", default=None,
                    help="persistent directory for the digest-verified release zip "
                         "(default: inside the workdir, removed with it)")
    ap.add_argument("--keep", action="store_true", help="do not delete what the run created")
    args = ap.parse_args(argv)

    receipt_path = Path(args.receipt)
    tag = args.tag.strip() if args.tag else None
    if tag is not None and not _TAG_RE.match(tag):
        receipt = Receipt(tag, False)
        receipt.note = f"{tag!r} is not a uv release version (MAJOR.MINOR.PATCH)"
        receipt.write(receipt_path)
        print(f"REFUSED: {receipt.note}")
        return 2
    receipt = Receipt(tag or "unknown", bool(args.current))
    try:
        receipt = confirm(tag, bool(args.current), Path(args.workdir),
                          Path(args.cache_dir) if args.cache_dir else None, keep=args.keep)
    except BaseException as e:
        receipt.note = f"unexpected {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        receipt.write(receipt_path)
    print(f"\nRESULT: {receipt.verdict()} - {receipt.why()}")
    print(f"receipt written: {receipt_path}")
    return receipt.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
