# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing and locating the stable-diffusion.cpp runtime.

Runtimes live under ``<LOCALM_HOME>/runtimes/sdcpp/<tag>-<backend>/``. A
directory counts as installed once it carries ``MARKER`` (written after the
runtime loaded and passed the ABI check in a worker); a backend whose load-test
failed carries ``FAILED_MARKER`` instead and is skipped by ``auto``.
"""

from __future__ import annotations

import json
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
Ask = Callable[[str], bool]


class ProvisionError(RuntimeError):
    """The runtime could not be installed or did not pass its load-test."""


class DownloadError(ProvisionError):
    """A download was refused, failed, or did not verify."""


#: Written in :func:`runtimes_root` when an install found Visual C++ runtime
#: DLLs absent: ``{"absent": [dll names]}``.
VC_MISSING_MARKER = ".vc-runtime-missing"


class LoadTestError(ProvisionError):
    """The runtime was fetched and does not work on this machine."""


class VcRuntimeMissingError(ProvisionError):
    """Windows lacks the Microsoft Visual C++ Redistributable the runtime needs,
    so no backend of it loads; nothing was installed."""


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


def other_gpu_runtimes(det=None) -> list[str]:
    """The names of the GPU runtimes besides Vulkan this machine's GPUs can
    use where upstream ships them: ``"CUDA"`` for an NVIDIA GPU, ``"ROCm"`` for
    an AMD GPU."""
    plat = platform_key()
    d = det if det is not None else _detected()
    out = []
    if "nvidia" in d.vendors and (plat, "cuda") in pins.ASSETS:
        out.append("CUDA")
    if "amd" in d.vendors and (plat, "rocm") in pins.ASSETS:
        out.append("ROCm")
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def extra_dirs_for(backend: str) -> list[Path]:
    return rocm_library_dirs() if backend == "rocm" else []


def _has_backend_device(backend: str, devices: list) -> bool:
    """Whether *devices* (``[name, description]`` pairs) include one the
    *backend* runs on: any device for ``cpu``, a device other than the CPU for
    a GPU backend."""
    if backend == "cpu":
        return bool(devices)
    return any(isinstance(d, (list, tuple)) and d and str(d[0]).strip().upper() != "CPU"
               for d in devices)


def installed(backend: str) -> Optional[Runtime]:
    """The installed, load-tested runtime for *backend* at the pinned tag, or
    None. A GPU runtime whose load test found only the CPU is not installed."""
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
    devices = list(meta.get("devices") or [])
    if not _has_backend_device(backend, devices):
        return None
    return Runtime(backend=backend, path=d, extra_dirs=extra_dirs_for(backend),
                   devices=devices)


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


# The notice written beside the NVIDIA CUDA runtime libraries of a cudart
# bundle, which carries no license file of its own. Same text as
# localm.setup_llama.cuda.NVIDIA_CUDA_LICENSE_NOTICE (see
# test_sdcpp_nvidia_notice_matches_setup_llama).
NVIDIA_CUDA_LICENSE_NOTICE = """\
The NVIDIA CUDA runtime libraries in this folder (cudart, cuBLAS, cuBLASLt)
are proprietary NVIDIA software, not open source. They are covered by the
NVIDIA CUDA Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/index.html
"""

NVIDIA_CUDA_LICENSE = "LICENSE.nvidia-cuda"


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


def _require_vc_runtime(staging: Path, say: Progress, ask: Optional[Ask]) -> None:
    """Windows: make sure the Microsoft Visual C++ runtime the build in
    *staging* imports is present, offering to install it through *ask* (None:
    not offered) with :func:`localm.vc_runtime.ensure`. Raises
    :class:`VcRuntimeMissingError` when a DLL of it is still absent from the
    system folder; *say* has then had the message that names it and the remedy.
    Returns when only an older version is present, and the load test decides."""
    from localm import netpolicy, vc_runtime
    marker = runtimes_root() / VC_MISSING_MARKER
    if vc_runtime.ensure(
            staging, ask=ask, say=say, refusal=netpolicy.download_refusal,
            download=lambda url, dest: _download(url, dest, say,
                                                 "the Visual C++ Redistributable")):
        _forget_vc_missing(marker)
        return
    absent = vc_runtime.absent_from_system(staging)
    if not absent:
        return
    try:
        marker.write_text(json.dumps({"absent": absent}), encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("sdcpp runtime: could not record the missing Visual C++ runtime: %s", e)
    raise _vc_missing_error()


def _vc_missing_error() -> VcRuntimeMissingError:
    from localm import vc_runtime
    return VcRuntimeMissingError(
        f"the {RUNTIME_LABEL} needs the Microsoft Visual C++ Redistributable, which this "
        f"Windows lacks. {vc_runtime.install_hint()}")


def _vc_known_missing() -> bool:
    """Windows: whether an earlier install recorded (:data:`VC_MISSING_MARKER`)
    Visual C++ runtime DLLs that the system folder still lacks. Removes the
    record once they are present. False when there is no readable record."""
    from localm import vc_runtime
    marker = runtimes_root() / VC_MISSING_MARKER
    try:
        names = json.loads(marker.read_text(encoding="utf-8"))["absent"]
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return False
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return False
    if vc_runtime.absent_names(names):
        return True
    _forget_vc_missing(marker)
    return False


def _forget_vc_missing(marker: Path) -> None:
    try:
        marker.unlink(missing_ok=True)
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("sdcpp runtime: could not remove %s: %s", marker, e)


RUNTIME_LABEL = "native image and video runtime"


def install(backend: str, *, force: bool = False,
            on_progress: Optional[Progress] = None,
            probe: Optional[Callable[[Runtime], dict]] = None,
            ask: Optional[Ask] = None, into: Optional[Path] = None) -> Runtime:
    """Download, verify, extract and load-test the runtime for *backend*.

    Network access goes through ``netpolicy`` as an explicit download. The runtime
    is tested in a folder of its own and then moved in under a record
    (``update_mechanism.steps``); an installed one of the same build stays, also
    when the new one fails or this process ends. On a failed load-test
    ``FAILED_MARKER`` is written, unless the same build is installed and works, and
    :class:`LoadTestError` is raised. With *into* (a folder on the volume of
    :func:`runtimes_root`) the tested runtime is left there as ``<tag>-<backend>``
    and nothing of the installed ones changes. On
    Windows without the Microsoft Visual C++ Redistributable it is offered
    through *ask* (None: not offered), and :class:`VcRuntimeMissingError` is
    raised, with nothing installed, when it stays missing; with *ask* None it
    is raised before any download while the DLLs an earlier install recorded as
    absent (:data:`VC_MISSING_MARKER`) are still absent. Every progress
    message, and the failure, is also written to the debug log.
    *probe* replaces the real worker load-test (tests only)."""
    from localm.debuglog import logger
    from localm.media.runtime_files import logged
    say = logged(on_progress, "sdcpp runtime")
    try:
        return _install(backend, force=force, say=say, probe=probe, ask=ask, into=into)
    except ProvisionError as e:
        logger.warning("sdcpp runtime: installing the %s runtime failed: %s", backend, e)
        raise


def _install(backend: str, *, force: bool, say: Progress,
             probe: Optional[Callable[[Runtime], dict]],
             ask: Optional[Ask] = None, into: Optional[Path] = None) -> Runtime:
    plat = platform_key()
    if plat is None or (plat, backend) not in pins.ASSETS:
        raise ProvisionError(
            f"stable-diffusion.cpp publishes no '{backend}' build for this platform "
            f"(available: {', '.join(available_backends(plat)) or 'none'}).")
    if into is None:
        force = _settle() or force
    if not force and into is None:
        rt = installed(backend)
        if rt is not None:
            return rt
    if ask is None and sys.platform == "win32" and _vc_known_missing():
        raise _vc_missing_error()
    from localm import netpolicy
    from localm.media.runtime_files import put_runtime, refused_download_message
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
                say(f"Installing the {RUNTIME_LABEL} (stable-diffusion.cpp "
                    f"{pins.TAG}, {backend}): {name}")
                _fetch_into(name, sha, staging, tmp, say)
        except netpolicy.NetworkPolicyError as e:
            raise DownloadError(refused_download_message(e)) from e
        _flatten(staging)
        _require_vc_runtime(staging, say, ask)
        if any(name.startswith("cudart-")
               for name, _sha in pins.EXTRA_ASSETS.get((plat, backend), [])):
            (staging / NVIDIA_CUDA_LICENSE).write_text(NVIDIA_CUDA_LICENSE_NOTICE,
                                                       encoding="utf-8")
        rt = Runtime(backend=backend, path=staging, extra_dirs=extra_dirs_for(backend))
        say(f"Testing the {RUNTIME_LABEL} ({backend})...")
        try:
            devices = _tested_devices(rt, probe or _probe)
        except LoadTestError as e:
            if into is None and installed(backend) is None:
                dest.mkdir(exist_ok=True)
                (dest / FAILED_MARKER).write_text(json.dumps(
                    {"tag": pins.TAG, "backend": backend, "reason": str(e.__cause__ or e)}),
                    encoding="utf-8")
            raise
        (staging / MARKER).write_text(json.dumps(
            {"tag": pins.TAG, "commit": pins.COMMIT, "backend": backend,
             "devices": devices}, indent=2), encoding="utf-8")
        put_runtime(into if into is not None else root, dest.name, staging,
                    record=into is None, error=ProvisionError,
                    busy="stop any image or video generation that is using it and retry")
    if into is not None:
        dest = Path(into) / dest.name
    rt = Runtime(backend=backend, path=dest, extra_dirs=extra_dirs_for(backend),
                 devices=devices)
    names = ", ".join(f"{n} ({desc})" for n, desc in devices)
    say(f"The {RUNTIME_LABEL} is ready ({backend}): {names}")
    return rt


def _tested_devices(rt: Runtime, probe: Callable[[Runtime], dict]) -> list:
    """The compute devices the runtime *rt* registers in a worker. Raises
    :class:`LoadTestError` when it does not load or has none of its backend's."""
    try:
        info = probe(rt)
    except Exception as e:  # noqa: BLE001
        raise LoadTestError(f"the {rt.backend} runtime did not load: {e}") from e
    devices = [list(d) for d in info.get("devices") or []]
    if not devices:
        raise LoadTestError(f"the {rt.backend} runtime loaded but found no compute device")
    if not _has_backend_device(rt.backend, devices):
        raise LoadTestError(f"the {rt.backend} runtime loaded but found no {rt.backend} device, "
                            "only the CPU")
    return devices


def _settle() -> bool:
    """``runtime_files.settle_runtimes`` of :func:`runtimes_root`."""
    from localm.media.runtime_files import settle_runtimes
    return settle_runtimes(runtimes_root(), ProvisionError)


def _probe(rt: Runtime) -> dict:
    from .runner import SdRunner
    return SdRunner().probe(rt.path, rt.extra_dirs)


def provision(choice: str = "auto", *, force: bool = False,
              on_progress: Optional[Progress] = None,
              probe: Optional[Callable[[Runtime], dict]] = None,
              ask: Optional[Ask] = None) -> Runtime:
    """Install (if needed) and return the runtime for *choice*.

    An explicit backend is installed or the call fails with the reason.
    ``auto`` tries the recommended backend, then vulkan, then cpu, skipping a
    backend whose load-test already failed at this tag, and reports each
    fallback through *on_progress*. A network failure, and a missing Microsoft
    Visual C++ Redistributable (offered through *ask*, see :func:`install`),
    stop the attempt."""
    say = on_progress or (lambda _m: None)
    choice = (choice or "auto").strip().lower()
    if choice != "auto":
        if choice not in BACKENDS:
            raise ProvisionError(f"unknown runtime backend {choice!r} "
                                 f"(choose from: auto, {', '.join(BACKENDS)})")
        return install(choice, force=force, on_progress=say, probe=probe, ask=ask)
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
            return install(b, force=force, on_progress=say, probe=probe, ask=ask)
        except (DownloadError, VcRuntimeMissingError):
            raise
        except ProvisionError as e:
            errors.append(f"{b}: {e}")
            say(f"The {b} runtime did not work here ({e}); trying the next one.")
    raise ProvisionError("no stable-diffusion.cpp runtime works on this machine: "
                         + "; ".join(errors))


def ensure(choice: str = "auto", *, on_progress: Optional[Progress] = None,
           probe: Optional[Callable[[Runtime], dict]] = None) -> Runtime:
    """The runtime a generation uses for *choice*, installing one when needed.

    An explicit backend returns its installed runtime, else :func:`provision`
    installs it. ``auto`` with nothing installed calls :func:`provision`.
    ``auto`` with an installed GPU runtime returns it. ``auto`` with only the
    cpu runtime installed, on a machine whose recommended backend is a GPU one,
    installs the recommended backend (then vulkan), skipping a backend whose
    load test failed at this tag; when the download fails or is refused, or no
    GPU runtime loads, it returns the cpu runtime and reports why through
    *on_progress*. Raises :class:`ProvisionError` only when no runtime can be
    used."""
    say = on_progress or (lambda _m: None)
    choice = (choice or "auto").strip().lower()
    rt = resolve(choice)
    if choice != "auto":
        return rt if rt is not None else provision(choice, on_progress=say, probe=probe)
    if rt is None:
        return provision("auto", on_progress=say, probe=probe)
    best = recommended_backend()
    if rt.backend != "cpu" or best == "cpu":
        return rt
    plat = platform_key()
    for b in dict.fromkeys([best, "vulkan"]):
        if (plat, b) not in pins.ASSETS or load_test_failed(b):
            continue
        try:
            return install(b, on_progress=say, probe=probe)
        except DownloadError as e:
            say(f"The {b} runtime could not be downloaded ({e}); using the installed "
                "cpu runtime.")
            return rt
        except ProvisionError as e:
            say(f"The {b} runtime did not work here ({e}); trying the next one.")
    return rt
