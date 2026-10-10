# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing and locating the stable-diffusion.cpp runtime.

Runtimes live under ``<LOCALM_HOME>/runtimes/sdcpp/<tag>-<backend>/``. A
directory counts as installed once it carries ``MARKER`` (written after the
runtime loaded and passed the ABI check in a worker); a backend whose load-test
failed carries ``FAILED_MARKER`` instead and is skipped by ``auto``.
"""

from __future__ import annotations

import json
import os
import shutil
import site
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import pins

MARKER = "localm-sdcpp.json"
FAILED_MARKER = "localm-sdcpp-failed.json"

BACKENDS = ("cpu", "vulkan", "cuda", "rocm", "metal")

_DOWNLOAD_STALL_TIMEOUT = 60

Progress = Callable[[str], None]


class ProvisionError(RuntimeError):
    """The runtime could not be installed or did not pass its load-test."""


class DownloadError(ProvisionError):
    """A download was refused, failed, or did not verify."""


@dataclass
class Runtime:
    """An installed runtime: its directory, backend, and the extra library
    directories its loader needs."""
    backend: str
    path: Path
    extra_dirs: list = field(default_factory=list)
    devices: list = field(default_factory=list)


def platform_key() -> Optional[str]:
    """``"windows"``, ``"linux"``, ``"macos-arm64"``, or None where upstream
    publishes no build."""
    import platform as _platform
    machine = _platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows"
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux"
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64"
    return None


def available_backends(plat: Optional[str] = None) -> list[str]:
    """Backends upstream ships for *plat* (default: this machine)."""
    plat = plat if plat is not None else platform_key()
    return [b for b in BACKENDS if (plat, b) in pins.ASSETS]


def runtimes_root() -> Path:
    from localm.config import home_dir
    return home_dir() / "runtimes" / "sdcpp"


def runtime_dir(backend: str) -> Path:
    return runtimes_root() / f"{pins.TAG}-{backend}"


def _hipblas_name() -> str:
    return "hipblas.dll" if sys.platform == "win32" else "libhipblas.so"


def rocm_library_dirs() -> list[Path]:
    """Directories holding hipBLAS (and rocBLAS beside it) for the ROCm build:
    the rocm-sdk wheels in this venv, then ``/opt/rocm`` on Linux. Empty when
    none is installed."""
    roots: set[Path] = set()
    try:
        for p in site.getsitepackages():
            roots.add(Path(p))
    except Exception:
        pass
    roots.add(Path(sys.prefix) / ("Lib/site-packages" if sys.platform == "win32" else "lib"))
    pattern = "_rocm_sdk_*/bin" if sys.platform == "win32" else "_rocm_sdk_*/lib"
    found: list[Path] = []
    name = _hipblas_name()
    for root in sorted(roots):
        try:
            for d in sorted(root.glob(pattern)):
                if any(d.glob(name + "*")) and d not in found:
                    found.append(d)
        except OSError:
            continue
    if sys.platform.startswith("linux"):
        for d in (Path("/opt/rocm/lib"), Path("/opt/rocm/lib64")):  # hygiene-ok: generic ROCm system path
            try:
                if any(d.glob(name + "*")) and d not in found:
                    found.append(d)
            except OSError:
                continue
    return found


_detection_cache: list = []


def _detected():
    """``hwdetect.detect()``, run once per process."""
    if not _detection_cache:
        from localm import hwdetect
        _detection_cache.append(hwdetect.detect())
    return _detection_cache[0]


def recommended_backend(det=None) -> str:
    """The backend ``auto`` installs on this machine: metal on Apple Silicon,
    cuda for NVIDIA where upstream ships a CUDA build, rocm for AMD when hipBLAS
    is resolvable, cpu when no GPU was found, vulkan otherwise."""
    plat = platform_key()
    if plat == "macos-arm64":
        return "metal"
    d = det if det is not None else _detected()
    if d.gpu_state == "none":
        return "cpu"
    if "nvidia" in d.vendors and (plat, "cuda") in pins.ASSETS:
        return "cuda"
    if "amd" in d.vendors and (plat, "rocm") in pins.ASSETS and rocm_library_dirs():
        return "rocm"
    return "vulkan" if (plat, "vulkan") in pins.ASSETS else "cpu"


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extra_dirs_for(backend: str) -> list[Path]:
    return rocm_library_dirs() if backend == "rocm" else []


def installed(backend: str) -> Optional[Runtime]:
    """The installed, load-tested runtime for *backend* at the pinned tag, or None."""
    d = runtime_dir(backend)
    meta = _read_json(d / MARKER)
    if not meta or meta.get("tag") != pins.TAG or meta.get("backend") != backend:
        return None
    try:
        from ._binding import lib_filename
        if not (d / lib_filename()).is_file():
            return None
    except OSError:
        return None
    return Runtime(backend=backend, path=d, extra_dirs=extra_dirs_for(backend),
                   devices=list(meta.get("devices") or []))


def load_test_failed(backend: str) -> Optional[str]:
    """The recorded load-test failure for *backend* at the pinned tag, or None."""
    meta = _read_json(runtime_dir(backend) / FAILED_MARKER)
    if meta and meta.get("tag") == pins.TAG:
        return str(meta.get("reason") or "load-test failed")
    return None


def resolve(choice: str = "auto") -> Optional[Runtime]:
    """The runtime to use for *choice* (``auto`` or a backend name), if installed.

    ``auto`` prefers the recommended backend, then any other installed one in
    the order vulkan, cpu."""
    choice = (choice or "auto").strip().lower()
    if choice != "auto":
        return installed(choice)
    order = [recommended_backend(), "vulkan", "cpu"]
    for b in dict.fromkeys(order):
        rt = installed(b)
        if rt is not None:
            return rt
    for b in BACKENDS:
        rt = installed(b)
        if rt is not None:
            return rt
    return None


def _download(url: str, dest: Path, on_progress: Progress, label: str) -> None:
    from localm.http_ssl import verified_urlopen
    from localm.model_manager.pull import _ssrf_resolve_final_url
    import socket
    final = _ssrf_resolve_final_url(url)
    req = urllib.request.Request(final, headers={"User-Agent": "localm-setup-sdcpp"})
    prev = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_DOWNLOAD_STALL_TIMEOUT)
    last_pct = -1
    nread = 0
    try:
        with verified_urlopen(req, timeout=_DOWNLOAD_STALL_TIMEOUT) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
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
    finally:
        socket.setdefaulttimeout(prev)


def _fetch_into(name: str, sha256: str, staging: Path, tmp: Path,
                on_progress: Progress) -> None:
    from localm.setup_llama.download import ArtifactError, _extract_archive, _validate_archive
    arc = tmp / name
    _download(pins.asset_url(name), arc, on_progress, name)
    try:
        _validate_archive(arc, expected_sha256=sha256)
        _extract_archive(arc, staging)
    except ArtifactError as e:
        raise DownloadError(f"{name}: {e}") from e
    finally:
        try:
            arc.unlink()
        except OSError:
            pass


def _flatten(staging: Path) -> None:
    """Move the files of a single top-level directory up into *staging*, for an
    archive that wraps its contents in one folder."""
    from ._binding import lib_filename
    if (staging / lib_filename()).exists():
        return
    hits = list(staging.rglob(lib_filename()))
    if len(hits) != 1:
        return
    src = hits[0].parent
    for item in list(src.iterdir()):
        shutil.move(str(item), str(staging / item.name))


def install(backend: str, *, force: bool = False,
            on_progress: Optional[Progress] = None,
            probe: Optional[Callable[[Runtime], dict]] = None) -> Runtime:
    """Download, verify, extract and load-test the runtime for *backend*.

    Network access goes through ``netpolicy`` as an explicit download. On a
    failed load-test the directory is kept with ``FAILED_MARKER`` and
    :class:`ProvisionError` is raised. *probe* replaces the real worker
    load-test (tests only)."""
    say = on_progress or (lambda _m: None)
    plat = platform_key()
    if plat is None or (plat, backend) not in pins.ASSETS:
        raise ProvisionError(
            f"stable-diffusion.cpp publishes no '{backend}' build for this platform "
            f"(available: {', '.join(available_backends(plat)) or 'none'}).")
    if not force:
        rt = installed(backend)
        if rt is not None:
            return rt
    from localm import netpolicy
    root = runtimes_root()
    root.mkdir(parents=True, exist_ok=True)
    dest = runtime_dir(backend)
    assets = [pins.ASSETS[(plat, backend)]] + list(pins.EXTRA_ASSETS.get((plat, backend), []))
    with tempfile.TemporaryDirectory(prefix=".sdcpp-", dir=root) as tmp_s:
        tmp = Path(tmp_s)
        staging = tmp / "staging"
        staging.mkdir()
        try:
            for name, sha in assets:
                say(f"Installing the native image runtime (stable-diffusion.cpp "
                    f"{pins.TAG}, {backend}): {name}")
                _fetch_into(name, sha, staging, tmp, say)
        except netpolicy.NetworkPolicyError as e:
            raise DownloadError(f"the download was refused by the network policy: {e}") from e
        _flatten(staging)
        if dest.exists():
            try:
                shutil.rmtree(dest)
            except OSError as e:
                raise ProvisionError(
                    f"could not replace {dest} ({e}); stop any image generation that is "
                    "using it and retry") from e
        os.replace(staging, dest)
    rt = Runtime(backend=backend, path=dest, extra_dirs=extra_dirs_for(backend))
    say(f"Testing the native image runtime ({backend})...")
    try:
        info = (probe or _probe)(rt)
    except Exception as e:  # noqa: BLE001
        (dest / FAILED_MARKER).write_text(json.dumps(
            {"tag": pins.TAG, "backend": backend, "reason": str(e)}), encoding="utf-8")
        raise ProvisionError(f"the {backend} runtime did not load: {e}") from e
    devices = [list(d) for d in info.get("devices") or []]
    if not devices:
        (dest / FAILED_MARKER).write_text(json.dumps(
            {"tag": pins.TAG, "backend": backend, "reason": "no compute device"}),
            encoding="utf-8")
        raise ProvisionError(f"the {backend} runtime loaded but found no compute device")
    try:
        (dest / FAILED_MARKER).unlink()
    except OSError:
        pass
    (dest / MARKER).write_text(json.dumps(
        {"tag": pins.TAG, "commit": pins.COMMIT, "backend": backend,
         "devices": devices}, indent=2), encoding="utf-8")
    rt.devices = devices
    names = ", ".join(f"{n} ({desc})" for n, desc in devices)
    say(f"Native image runtime ready ({backend}): {names}")
    return rt


def _probe(rt: Runtime) -> dict:
    from .runner import SdRunner
    return SdRunner().probe(rt.path, rt.extra_dirs)


def provision(choice: str = "auto", *, force: bool = False,
              on_progress: Optional[Progress] = None,
              probe: Optional[Callable[[Runtime], dict]] = None) -> Runtime:
    """Install (if needed) and return the runtime for *choice*.

    An explicit backend is installed or the call fails with the reason.
    ``auto`` tries the recommended backend, then vulkan, then cpu, skipping a
    backend whose load-test already failed at this tag, and reports each
    fallback through *on_progress*. A network failure stops the attempt."""
    say = on_progress or (lambda _m: None)
    choice = (choice or "auto").strip().lower()
    if choice != "auto":
        if choice not in BACKENDS:
            raise ProvisionError(f"unknown runtime backend {choice!r} "
                                 f"(choose from: auto, {', '.join(BACKENDS)})")
        return install(choice, force=force, on_progress=say, probe=probe)
    if not force:
        rt = resolve("auto")
        if rt is not None and rt.backend == recommended_backend():
            return rt
    plat = platform_key()
    order = [b for b in dict.fromkeys([recommended_backend(), "vulkan", "cpu"])
             if (plat, b) in pins.ASSETS]
    if not order:
        raise ProvisionError("stable-diffusion.cpp publishes no build for this platform.")
    errors = []
    for b in order:
        if not force and (rt := installed(b)) is not None:
            return rt
        failed = load_test_failed(b)
        if failed and not force:
            errors.append(f"{b}: {failed}")
            continue
        try:
            return install(b, force=force, on_progress=say, probe=probe)
        except DownloadError:
            raise
        except ProvisionError as e:
            errors.append(f"{b}: {e}")
            say(f"The {b} runtime did not work here ({e}); trying the next one.")
    raise ProvisionError("no stable-diffusion.cpp runtime works on this machine: "
                         + "; ".join(errors))
