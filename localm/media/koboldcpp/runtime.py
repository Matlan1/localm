# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing and locating the KoboldCpp runtime used for native music generation.

Runtimes live under ``<LOCALM_HOME>/runtimes/koboldcpp/<tag>-<build>/``: the
release binary is downloaded, its size and sha256 are checked against
``pins.ASSETS`` before it is ever executed, it is unpacked once with its own
``--unpack`` (so later starts do not re-extract it into a temp directory), and
the unpacked launcher must report ``pins.VERSION``. A directory counts as
installed once it carries ``MARKER``.
"""

from __future__ import annotations

import hashlib
import contextlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import pins

MARKER = "localm-koboldcpp.json"

BACKENDS = ("cuda", "vulkan", "cpu", "metal")

# Backends each build can run.
BUILD_BACKENDS = {
    "cuda": ("cuda", "vulkan", "cpu"),
    "nocuda": ("vulkan", "cpu"),
    "metal": ("metal", "cpu"),
}

_DOWNLOAD_STALL_TIMEOUT = 60
# How long a recorded backend crash keeps ``auto`` from trying that backend.
FAILED_TTL = 24 * 3600
UNPACK_TIMEOUT = 600
VERSION_TIMEOUT = 60

Progress = Callable[[str], None]

_records_lock = threading.Lock()


class ProvisionError(RuntimeError):
    """The runtime could not be installed or did not pass its check."""


class DownloadError(ProvisionError):
    """A download was refused, failed, or did not verify."""


class InstallCancelled(ProvisionError):
    """The install was cancelled; nothing was installed."""


@dataclass
class Runtime:
    """An installed runtime: its build, directory and launcher executable."""
    build: str
    path: Path
    launcher: Path


def platform_key() -> Optional[str]:
    """``"windows"``, ``"linux"``, ``"macos-arm64"``, or None where KoboldCpp
    publishes no build."""
    from localm.media.sdcpp.runtime import platform_key as _platform_key
    return _platform_key()


def available_builds(plat: Optional[str] = None) -> list[str]:
    """Builds published for *plat* (default: this machine)."""
    plat = plat if plat is not None else platform_key()
    return [b for b in BUILD_BACKENDS if (plat, b) in pins.ASSETS]


def available_backends(plat: Optional[str] = None) -> list[str]:
    """Backends some published build can run on *plat* (default: this machine)."""
    found: list[str] = []
    for b in available_builds(plat):
        for backend in BUILD_BACKENDS[b]:
            if backend not in found:
                found.append(backend)
    return [b for b in BACKENDS if b in found]


def recommended_backend(det=None) -> str:
    """The backend ``auto`` uses on this machine: metal on Apple Silicon, cuda
    for NVIDIA where a CUDA build is published, cpu when no GPU was found,
    vulkan otherwise (AMD and Intel included)."""
    plat = platform_key()
    if plat == "macos-arm64":
        return "metal"
    from localm import hwdetect
    d = det if det is not None else hwdetect.detect()
    if d.gpu_state == "none":
        return "cpu"
    if "nvidia" in d.vendors and (plat, "cuda") in pins.ASSETS:
        return "cuda"
    return "vulkan"


def fallback_order(choice: str = "auto") -> list[str]:
    """The backends to try for *choice*, best first. An explicit backend is the
    only entry; ``auto`` is the recommended backend, then vulkan, then cpu,
    limited to backends this platform has a build for."""
    choice = (choice or "auto").strip().lower()
    if choice != "auto":
        return [choice]
    usable = available_backends()
    return [b for b in dict.fromkeys([recommended_backend(), "vulkan", "cpu"]) if b in usable]


def build_for(backend: str, plat: Optional[str] = None) -> str:
    """The build that runs *backend* on *plat*: an installed build that can run
    it, else the smallest published one. Raises :class:`ProvisionError` when no
    published build runs it."""
    plat = plat if plat is not None else platform_key()
    candidates = [b for b in ("nocuda", "metal", "cuda")
                  if (plat, b) in pins.ASSETS and backend in BUILD_BACKENDS[b]]
    if not candidates:
        raise ProvisionError(
            f"KoboldCpp publishes no build that runs '{backend}' on this platform "
            f"(available: {', '.join(available_backends(plat)) or 'none'}).")
    for b in candidates:
        if installed(b) is not None:
            return b
    return candidates[0]


def runtimes_root() -> Path:
    from localm.config import home_dir
    return home_dir() / "runtimes" / "koboldcpp"


def runtime_dir(build: str) -> Path:
    return runtimes_root() / f"{pins.TAG}-{build}"


def launcher_name() -> str:
    return "koboldcpp-launcher.exe" if sys.platform == "win32" else "koboldcpp-launcher"


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def installed(build: str) -> Optional[Runtime]:
    """The installed runtime for *build* at the pinned tag and asset, or None."""
    plat = platform_key()
    pinned = pins.ASSETS.get((plat, build)) if plat else None
    if pinned is None:
        return None
    d = runtime_dir(build)
    meta = _read_json(d / MARKER)
    if (not meta or meta.get("tag") != pins.TAG or meta.get("version") != pins.VERSION
            or meta.get("build") != build or meta.get("asset") != pinned[0]
            or meta.get("sha256") != pinned[2]):
        return None
    launcher = d / launcher_name()
    if not launcher.is_file():
        return None
    return Runtime(build=build, path=d, launcher=launcher)


def _private_temp_env(tmp: Path) -> dict:
    env = dict(os.environ)
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = str(tmp)
    return env


def _download(url: str, dest: Path, on_progress: Progress, label: str,
              cancel_check: Optional[Callable[[], bool]] = None) -> None:
    from localm.http_ssl import verified_urlopen
    from localm.model_manager.pull import _ssrf_resolve_final_url
    final = _ssrf_resolve_final_url(url)
    req = urllib.request.Request(final, headers={"User-Agent": "localm-setup-music"})
    last_pct = -1
    nread = 0
    try:
        with verified_urlopen(req, timeout=_DOWNLOAD_STALL_TIMEOUT) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                if cancel_check is not None and cancel_check():
                    raise InstallCancelled("the download was cancelled")
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                nread += len(chunk)
                if total > 0:
                    pct = nread * 100 // total
                    if pct >= last_pct + 10 or pct == 100 and last_pct != 100:
                        last_pct = pct
                        on_progress(f"Downloading {label}: {pct}% of "
                                    f"{total / 1024 ** 2:.0f} MB")
    except TimeoutError as e:
        raise DownloadError(
            f"the download of {label} stalled after {nread} bytes; retry on a stable "
            "network") from e
    except OSError as e:
        raise DownloadError(f"the download of {label} failed after {nread} bytes: {e}") from e


def verify_file(path: Path, size: int, sha256: str, label: str) -> None:
    """Raise :class:`DownloadError` unless *path* has exactly *size* bytes and
    the given sha256."""
    actual_size = path.stat().st_size
    if actual_size != size:
        raise DownloadError(
            f"{label} is {actual_size} bytes, expected {size}; the download is "
            "incomplete or not the pinned release asset")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise DownloadError(
            f"{label} does not match its pinned sha256 (got {h.hexdigest()[:16]}..., "
            f"expected {sha256[:16]}...); it was not run")


def _make_executable(path: Path) -> None:
    if sys.platform != "win32":
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _unpack(binary: Path, staging: Path, tmp: Path,
            cancel_check: Optional[Callable[[], bool]] = None) -> None:
    """Run the verified release binary's own ``--unpack`` into *staging*."""
    _make_executable(binary)
    log = tmp / "unpack.log"
    kwargs = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as out:
            proc = subprocess.Popen(
                [str(binary), "--unpack", str(staging)], cwd=str(tmp),
                env=_private_temp_env(tmp), stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT, **kwargs)
            deadline = time.monotonic() + UNPACK_TIMEOUT
            while proc.poll() is None:
                if cancel_check is not None and cancel_check():
                    _kill_tree(proc)
                    raise InstallCancelled("unpacking was cancelled")
                if time.monotonic() > deadline:
                    _kill_tree(proc)
                    raise ProvisionError(
                        f"unpacking KoboldCpp did not finish in {UNPACK_TIMEOUT} s")
                time.sleep(0.25)
    except OSError as e:
        raise ProvisionError(f"could not run the KoboldCpp binary to unpack it: {e}") from e
    launcher = staging / launcher_name()
    if not launcher.is_file():
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        tail = "\n".join(text.strip().splitlines()[-5:])
        raise ProvisionError(
            f"unpacking KoboldCpp produced no {launcher_name()} (exit {proc.returncode}): "
            f"{tail or 'no output'}")
    _make_executable(launcher)


def launcher_version(launcher: Path, tmp: Optional[Path] = None) -> str:
    """What ``<launcher> --version`` prints (its last non-empty line)."""
    work = tmp if tmp is not None else launcher.parent
    try:
        proc = subprocess.run(
            [str(launcher), "--version"], cwd=str(work), env=_private_temp_env(work),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=VERSION_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise ProvisionError(
            f"the KoboldCpp launcher did not answer --version in {VERSION_TIMEOUT} s") from e
    except OSError as e:
        raise ProvisionError(f"the KoboldCpp launcher could not be started: {e}") from e
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


_install_locks: dict = {}
_install_locks_guard = threading.Lock()

# A lock directory younger than this with no holder PID written yet belongs to an
# install that is still starting.
_LOCK_STARTUP_GRACE = 30.0


def _lock_holder(lockdir: Path) -> Optional[int]:
    try:
        return int((lockdir / "pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _lock_is_stale(lockdir: Path) -> bool:
    from localm.instances import pid_alive
    holder = _lock_holder(lockdir)
    if holder is None:
        try:
            age = time.time() - lockdir.stat().st_mtime
        except OSError:
            return False
        return age > _LOCK_STARTUP_GRACE
    return holder != os.getpid() and not pid_alive(holder)


@contextlib.contextmanager
def _install_lock(build: str, say: Progress, cancel_check: Optional[Callable[[], bool]]):
    """Hold the install of *build* against other threads and other localm
    processes. The cross-process half is a directory created atomically beside
    the runtimes, released on exit and taken over when its holder PID is gone."""
    with _install_locks_guard:
        lk = _install_locks.setdefault(build, threading.Lock())
    told = False

    def wait_once() -> None:
        nonlocal told
        if not told:
            say("Waiting for another install of the native music runtime to finish...")
            told = True
        if cancel_check is not None and cancel_check():
            raise InstallCancelled("cancelled while waiting for another install")

    while not lk.acquire(timeout=0.5):
        wait_once()
    try:
        root = runtimes_root()
        root.mkdir(parents=True, exist_ok=True)
        lockdir = root / f".install-{build}.lock"
        while True:
            try:
                lockdir.mkdir()
            except FileExistsError:
                if _lock_is_stale(lockdir):
                    shutil.rmtree(lockdir, ignore_errors=True)
                    continue
                wait_once()
                time.sleep(1.0)
                continue
            (lockdir / "pid").write_text(str(os.getpid()), encoding="utf-8")
            break
        try:
            yield
        finally:
            shutil.rmtree(lockdir, ignore_errors=True)
    finally:
        lk.release()


def install(build: str, *, force: bool = False,
            on_progress: Optional[Progress] = None,
            cancel_check: Optional[Callable[[], bool]] = None,
            unpack: Optional[Callable[[Path, Path, Path], None]] = None,
            version_of: Optional[Callable[[Path], str]] = None) -> Runtime:
    """Download, verify, unpack and check the runtime for *build*.

    Network access goes through ``netpolicy`` as an explicit download, every
    redirect hop checked. The binary is executed only after its size and sha256
    match the pin. Installs of one build are serialised across threads and
    processes; one that finds the runtime installed by the holder before it
    returns that. *cancel_check* is polled during the download and the unpack
    (raising :class:`InstallCancelled`). *unpack* and *version_of* replace the
    real subprocess steps (tests only)."""
    say = on_progress or (lambda _m: None)
    plat = platform_key()
    if plat is None or (plat, build) not in pins.ASSETS:
        raise ProvisionError(
            f"KoboldCpp publishes no '{build}' build for this platform "
            f"(available: {', '.join(available_builds(plat)) or 'none'}).")
    if not force:
        rt = installed(build)
        if rt is not None:
            return rt
    from localm import netpolicy
    name, size, sha = pins.ASSETS[(plat, build)]
    root = runtimes_root()
    dest = runtime_dir(build)
    with _install_lock(build, say, cancel_check):
        if not force:
            rt = installed(build)
            if rt is not None:
                return rt
        for old in root.glob(".koboldcpp-*"):
            shutil.rmtree(old, ignore_errors=True)
        with tempfile.TemporaryDirectory(prefix=".koboldcpp-", dir=root) as tmp_s:
            tmp = Path(tmp_s)
            binary = tmp / name
            staging = tmp / "staging"
            say(f"Installing the native music runtime (KoboldCpp {pins.TAG}, {build} build, "
                f"{size / 1024 ** 2:.0f} MB)")
            try:
                _download(pins.asset_url(name), binary, say, name, cancel_check)
            except netpolicy.NetworkPolicyError as e:
                raise DownloadError(
                    f"the download was refused by the network policy: {e}") from e
            verify_file(binary, size, sha, name)
            say("Unpacking the native music runtime...")
            if unpack is not None:
                unpack(binary, staging, tmp)
            else:
                _unpack(binary, staging, tmp, cancel_check)
            version = (version_of or (lambda p: launcher_version(p, tmp)))(
                staging / launcher_name())
            if version != pins.VERSION:
                raise ProvisionError(
                    f"the unpacked KoboldCpp reports version {version or 'nothing'}, "
                    f"expected {pins.VERSION}")
            (staging / MARKER).write_text(json.dumps(
                {"tag": pins.TAG, "version": pins.VERSION, "build": build, "asset": name,
                 "sha256": sha}, indent=2), encoding="utf-8")
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                os.replace(staging, dest)
            except OSError as e:
                raise ProvisionError(
                    f"could not install into {dest} ({e}); stop any music generation that "
                    "is using it and retry") from e
    say(f"Native music runtime ready (KoboldCpp {pins.VERSION}, {build} build)")
    return Runtime(build=build, path=dest, launcher=dest / launcher_name())


def _backends_file() -> Path:
    return runtimes_root() / f"{pins.TAG}-backends.json"


def backend_failed(backend: str) -> Optional[str]:
    """The recorded reason *backend* did not work at the pinned tag, or None when
    it has no failure recorded in the last ``FAILED_TTL`` seconds."""
    data = _read_json(_backends_file()) or {}
    entry = data.get(backend)
    if not isinstance(entry, dict) or entry.get("ok") is not False:
        return None
    try:
        age = time.time() - float(entry.get("at", 0))
    except (TypeError, ValueError):
        return None
    if age < 0 or age > FAILED_TTL:
        return None
    return str(entry.get("reason") or "it did not work here")


def backend_worked(backend: str) -> bool:
    """True when *backend* generated music at the pinned tag before."""
    entry = (_read_json(_backends_file()) or {}).get(backend)
    return isinstance(entry, dict) and entry.get("ok") is True


def record_backend(backend: str, ok: bool, reason: str = "") -> None:
    """Remember whether *backend* worked, so ``auto`` skips one that crashed
    (for ``FAILED_TTL``) and a later success clears the failure."""
    with _records_lock:
        path = _backends_file()
        data = _read_json(path) or {}
        data[backend] = {"ok": bool(ok), "reason": reason[:500], "at": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def clear_backend_records() -> None:
    """Forget every recorded backend result at the pinned tag."""
    with _records_lock:
        try:
            _backends_file().unlink()
        except FileNotFoundError:
            pass


def ensure_for_backend(backend: str, *, force: bool = False,
                       on_progress: Optional[Progress] = None,
                       cancel_check: Optional[Callable[[], bool]] = None) -> Runtime:
    """The installed runtime that runs *backend*, installing its build first
    when needed."""
    return install(build_for(backend), force=force, on_progress=on_progress,
                   cancel_check=cancel_check)
