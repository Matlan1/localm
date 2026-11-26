# SPDX-License-Identifier: AGPL-3.0-or-later
"""AMD's ROCm libraries for the upstream Windows hip build.

Upstream's Windows hip archive ships ggml-hip.dll and the HIP runtime
(amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll), but ggml-hip.dll also links
hipblas.dll, which links rocBLAS, rocSOLVER, hipBLASLt and origami, and rocBLAS
needs GPU kernels per architecture. The archive carries none of them: its build
expects AMD's ROCm installed beside it. :func:`supply_hip_libraries` adds them
from AMD's own wheels of the ROCm release the bundled HIP runtime belongs to
(``_HIP_SDK_WHEELS`` in pins.py, verified by sha256): the libraries ggml-hip.dll
needs from rocm-sdk-libraries, and the kernels of each GPU in this machine from
its rocm-sdk-device-<gfx> wheel. They go into ``HIP_LIBRARIES_DIR`` of the
runtime folder in the wheels' own layout, which the loader puts on the DLL
search path (``_loader.hip_library_dirs``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from rich.markup import escape

from localm import pe_deps
from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.download import ArtifactError, _validate_archive
from localm.setup_llama.library_files import HIP_LIBRARIES_DIR, HIP_LIBRARIES_MARKER
from localm.setup_llama.pins import _HIP_SDK_INDEX
import localm.setup_llama as _sl

# The folder the set is assembled in before it is renamed to HIP_LIBRARIES_DIR.
_ASSEMBLY_DIR = HIP_LIBRARIES_DIR + ".tmp"

# The folder all files of both wheels sit under.
_WHEEL_ROOT = "_rocm_sdk_libraries/"

# Parts of a device wheel that are Python packaging, not runtime files.
_PACKAGING_PARTS = ("__init__.py", ".devel_links/")

# The DLL the upstream hip build links and does not ship.
_HIPBLAS = "hipblas.dll"

_PROBE = Path(__file__).with_name("_hip_probe.py")
_PROBE_LIMIT_S = 120
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class HipLibrariesError(Exception):
    """The ROCm libraries could not be added; the message says why."""


def _hip_images(target: Path) -> list[Path]:
    return sorted(p for p in target.glob("ggml-hip*.dll") if p.is_file())


def recorded_archs(target: Path) -> Optional[list[str]]:
    """The GPU architectures the ``HIP_LIBRARIES_MARKER`` of *target*'s
    ``HIP_LIBRARIES_DIR`` records, or None when that folder holds no hipblas.dll
    or no readable marker."""
    folder = target / HIP_LIBRARIES_DIR
    try:
        if not (folder / "bin" / _HIPBLAS).is_file():
            return None
        record = json.loads((folder / HIP_LIBRARIES_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    archs = record.get("archs") if isinstance(record, dict) else None
    if not isinstance(archs, list):
        return None
    return [str(a) for a in archs]


def needs_hip_libraries(target: Path) -> bool:
    """Whether a ggml-hip library in *target* imports hipblas.dll, *target*
    does not hold it, and *target*'s ``HIP_LIBRARIES_DIR`` is not a complete set
    for this machine: missing, without its marker (:func:`recorded_archs`), or
    lacking the kernels of a GPU now reported (:func:`hip_gpu_archs`) that the
    pins cover. A GPU question that fails keeps a complete set (logged)."""
    images = _hip_images(target)
    if not images or (target / _HIPBLAS).is_file():
        return False
    if not any(_HIPBLAS in (n.lower() for n in pe_deps.imported_dlls(p)) for p in images):
        return False
    recorded = recorded_archs(target)
    if recorded is None:
        return True
    try:
        detected = _sl.hip_gpu_archs(target)
    except HipLibrariesError as e:
        logger.warning("hip build: kept the ROCm libraries in place, the GPU check failed: %s", e)
        return False
    found = rocm_release(target)
    pinned = _sl._HIP_SDK_WHEELS.get(found[0], {}).get("devices", {}) if found else {}
    return any(a in pinned and a not in recorded for a in detected)


def rocm_release(target: Path) -> Optional[tuple[str, tuple]]:
    """``("major.minor", file version)`` of the HIP runtime (``amdhip64_*.dll``)
    in *target*, or None when it has none or its version cannot be read."""
    found = sorted(target.glob("amdhip64_*.dll"))
    if not found:
        return None
    version = pe_deps.file_version(found[-1])
    if version is None:
        return None
    return f"{version[0]}.{version[1]}", version


def hip_gpu_archs(target: Path) -> list[str]:
    """The architecture names (``gfx1030``) of the GPUs the HIP runtime in
    *target* reports, read in a child interpreter (``_hip_probe.py``). Raises
    :class:`HipLibrariesError` when the runtime could not be asked."""
    try:
        r = subprocess.run([sys.executable, "-I", str(_PROBE), str(target)],
                           capture_output=True, text=True, timeout=_PROBE_LIMIT_S,
                           creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise HipLibrariesError(f"the GPU check could not run: {e}") from e
    if r.stderr.strip():
        logger.warning("hip GPU check: %s", r.stderr.strip())
    if r.returncode != 0:
        raise HipLibrariesError(r.stderr.strip() or f"the GPU check ended with exit code "
                                f"{r.returncode}")
    archs: list[str] = []
    for line in r.stdout.splitlines():
        arch = line.strip()
        if arch and arch not in archs:
            archs.append(arch)
    return archs


def _wheel_url(project: str, name: str) -> str:
    return f"{_HIP_SDK_INDEX}{project}/{name}"


def _fetch_wheel(url: str, dest: Path, size: int, sha256: str) -> None:
    """Download *url* to *dest* and check that it is a zip of at least *size*
    bytes with the digest *sha256*. Raises :class:`ArtifactError`."""
    dl = _sl._download(url, dest)
    _validate_archive(dest, expected_sha256=sha256, min_size=size, dl=dl)


def _safe_member(name: str) -> bool:
    parts = Path(name).parts
    return not (name.startswith(("/", "\\")) or ".." in parts or ":" in name)


def _library_closure(wheel: zipfile.ZipFile, target: Path) -> dict[str, str]:
    """``{dll name: wheel member}`` of the wheel's ``bin`` DLLs the ggml-hip
    libraries in *target* need: those they import that *target* does not hold,
    and in turn the DLLs those import from the wheel."""
    in_bin = {}
    for name in wheel.namelist():
        if not _safe_member(name):
            raise ArtifactError(f"unsafe path in the ROCm libraries wheel: {name!r}")
        rel = name[len(_WHEEL_ROOT):] if name.startswith(_WHEEL_ROOT) else None
        if rel and rel.startswith("bin/") and rel.count("/") == 1 and rel.lower().endswith(".dll"):
            in_bin[rel[4:].lower()] = name
    have = {p.name.lower() for p in target.iterdir()}
    chosen: dict[str, str] = {}
    queue = [n.lower() for image in _hip_images(target) for n in pe_deps.imported_dlls(image)]
    while queue:
        dll = queue.pop(0)
        if dll in chosen or dll in have or dll not in in_bin:
            continue
        chosen[dll] = in_bin[dll]
        queue.extend(n.lower() for n in pe_deps.imported_dlls(wheel.read(in_bin[dll])))
    return chosen


def _extract(wheel: zipfile.ZipFile, members: dict[str, str], dest: Path) -> int:
    """Write the wheel *members* (``{relative path: member}``) under *dest*."""
    for rel, member in members.items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with wheel.open(member) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return len(members)


def _device_members(wheel: zipfile.ZipFile) -> dict[str, str]:
    """``{relative path: member}`` of the runtime files in a device wheel."""
    out = {}
    for info in wheel.infolist():
        name = info.filename
        if not _safe_member(name):
            raise ArtifactError(f"unsafe path in the ROCm kernel wheel: {name!r}")
        if info.is_dir() or not name.startswith(_WHEEL_ROOT):
            continue
        rel = name[len(_WHEEL_ROOT):]
        if rel and not rel.startswith(_PACKAGING_PARTS):
            out[rel] = name
    return out


def _add(target: Path, release: str, version: tuple, archs: list[str]) -> list[str]:
    """Download the libraries and the kernels of *archs* into ``_ASSEMBLY_DIR``
    of *target*, write the marker, and rename the folder to
    ``HIP_LIBRARIES_DIR``. Returns the DLL names added. Raises."""
    spec = _sl._HIP_SDK_WHEELS[release]
    lib_name, lib_size, lib_sha = spec["libraries"]
    dest = target / _ASSEMBLY_DIR
    if dest.exists():
        shutil.rmtree(dest)
    record = {"rocm": spec["version"], "hip_runtime": ".".join(map(str, version)),
              "archs": archs, "wheels": {lib_name: lib_sha}}
    with tempfile.TemporaryDirectory() as tmp:
        wheel_path = Path(tmp) / lib_name
        _fetch_wheel(_wheel_url("rocm-sdk-libraries", lib_name), wheel_path, lib_size, lib_sha)
        with zipfile.ZipFile(wheel_path) as wheel:
            closure = _library_closure(wheel, target)
            if _HIPBLAS not in closure:
                raise HipLibrariesError(f"{lib_name} does not contain {_HIPBLAS}")
            _extract(wheel, {f"bin/{m.rsplit('/', 1)[-1]}": m for m in closure.values()}, dest)
        wheel_path.unlink()
        for arch in archs:
            size, sha = spec["devices"][arch]
            name = f"rocm_sdk_device_{arch}-{spec['version']}-py3-none-win_amd64.whl"
            wheel_path = Path(tmp) / name
            _fetch_wheel(_wheel_url(f"rocm-sdk-device-{arch}", name), wheel_path, size, sha)
            with zipfile.ZipFile(wheel_path) as wheel:
                if not _extract(wheel, _device_members(wheel), dest):
                    raise HipLibrariesError(f"{name} contains no kernel files")
            wheel_path.unlink()
            record["wheels"][name] = sha
    (dest / HIP_LIBRARIES_MARKER).write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(dest, target / HIP_LIBRARIES_DIR)
    return sorted(closure)


def _warn(why: str) -> None:
    logger.warning("hip build: AMD ROCm libraries not added: %s", why)
    console.print(f"[yellow]Warning: the AMD ROCm libraries the hip build needs were not "
                  f"added ({escape(why)}). It does not load without them. Retry with "
                  "localm setup-llama --backend hip --force, or use --backend vulkan."
                  "[/yellow]")


def supply_hip_libraries(target: Path) -> Optional[list[str]]:
    """When the ggml-hip library in *target* needs hipblas.dll and *target*
    lacks it (:func:`needs_hip_libraries`), add AMD's ROCm libraries and the
    kernels of this machine's GPUs for the ROCm release of the HIP runtime in
    *target*. Returns the DLL names added, or None when nothing was added: not
    needed, or not possible (a release localm pins no libraries for, no HIP GPU
    found, a GPU no kernels are pinned for, a failed download or checksum),
    which is printed and logged, and *target* then holds no partial set.

    Never raises."""
    try:
        if not needs_hip_libraries(target):
            return None
    except OSError as e:
        _warn(f"the runtime folder could not be read: {e}")
        return None
    try:
        found = rocm_release(target)
        if found is None:
            _warn("the build ships no HIP runtime whose version can be read, so the "
                  "matching ROCm release is unknown")
            return None
        release, version = found
        wheels = _sl._HIP_SDK_WHEELS
        if release not in wheels:
            _warn(f"this build ships the HIP runtime of ROCm {release} (amdhip64 "
                  f"{'.'.join(map(str, version))}), and localm pins AMD's libraries for "
                  f"ROCm {', '.join(sorted(wheels))} only; the build localm pins is "
                  "installed by localm setup-llama --backend hip --tag default")
            return None
        archs = _sl.hip_gpu_archs(target)
        if not archs:
            _warn("the build's HIP runtime found no AMD GPU")
            return None
        covered = [a for a in archs if a in wheels[release]["devices"]]
        other = [a for a in archs if a not in covered]
        if not covered:
            _warn(f"localm pins no ROCm {release} kernels for {', '.join(other)}, a GPU "
                  "this hip build does not support")
            return None
        if other:
            console.print(f"[yellow]No ROCm {release} kernels are pinned for "
                          f"{', '.join(other)}; the hip build runs on "
                          f"{', '.join(covered)} only.[/yellow]")
        stale = target / HIP_LIBRARIES_DIR
        if stale.exists():
            shutil.rmtree(stale)
        console.print(f"[dim]Adding AMD's ROCm {wheels[release]['version']} libraries and "
                      f"the {', '.join(covered)} kernels the hip build needs (no HIP SDK "
                      "install needed)[/dim]")
        added = _add(target, release, version, covered)
    except Exception as e:  # noqa: BLE001 - reported by _warn, provisioning continues
        partial = target / _ASSEMBLY_DIR
        shutil.rmtree(partial, ignore_errors=True)
        if partial.exists():
            logger.warning("hip build: could not remove the partial %s", partial)
        _warn(f"{type(e).__name__}: {e}")
        return None
    console.print(f"[dim]ROCm libraries:[/dim] {', '.join(added)}")
    return added
