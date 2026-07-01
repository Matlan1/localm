# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Makes localm self-contained: the native inference runtime (the llama shared
library + its ggml deps, plus a matched GPU runtime when the prebuilt ships one)
is placed inside the project's own ``localm-llama-runtime`` wheel rather than
depending on a folder elsewhere on disk.

Backends (``--backend``), so any machine has a working out-of-the-box path:
  * ``auto`` (default) - detect the GPU and pick the broadest WORKING backend:
    AMD on Windows -> the self-contained ROCm build (AMD on Linux -> ``vulkan``);
    any other GPU -> ``vulkan`` (runs on NVIDIA/Intel/AMD through the normal
    display driver, no vendor toolkit); Apple Silicon -> ``metal``; no GPU ->
    ``cpu``.
  * ``vulkan`` - universal GPU build from upstream llama.cpp (recommended for
    NVIDIA/Intel, and a no-toolkit option for AMD).
  * ``cuda`` / ``sycl`` / ``cpu`` - upstream llama.cpp prebuilts. ``cuda`` and
    ``sycl`` deliver peak vendor performance but need that vendor's runtime
    (CUDA toolkit / oneAPI) present; ``vulkan`` and ``cpu`` are self-contained.
  * ``amd-rocm`` - the self-contained gfx103X (RDNA2) ROCm build (bundles its
    own ROCm runtime; the current default for AMD RX 6000).

Sources, in order of preference:
  * ``--from <dir>``  - copy from a local llama.cpp build output (any backend).
  * ``--url <url>``   - an explicit prebuilt archive URL.
  * ``--backend ...`` - resolve the matching upstream llama.cpp release asset
    (``ggml-org/llama.cpp``); the latest release is used, with a pinned
    fallback if the release lookup is unavailable.

After placing the files it installs the runtime wheel editable so the loader can
import it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from localm.debuglog import logger

console = Console(highlight=False)

# Self-contained AMD build: lemonade-sdk llama.cpp ROCm build for gfx103X
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so AMD RX 6000 users need
# no separate HIP SDK. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1288/llama-b1288-windows-rocm-gfx103X-x64.zip"
)

# Upstream llama.cpp prebuilts (ggml-org/llama.cpp). We resolve the latest
# release tag at runtime; this pin is the fallback if that lookup fails.
_UPSTREAM_REPO = "ggml-org/llama.cpp"
_FALLBACK_TAG = "b9682"

# Per-backend asset matcher: substrings that must appear in the release asset
# name for (platform, backend). Substring matching (not exact names) keeps this
# robust to upstream version suffixes drifting (e.g. cuda-12.4, rocm-7.2).
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-x64"],
        "vulkan": ["bin-win-vulkan-x64"],
        "cuda":   ["bin-win-cuda-12"],          # prefer the 12.x runtime line
        "sycl":   ["bin-win-sycl-x64"],
        "hip":    ["bin-win-hip-radeon-x64"],   # needs AMD HIP SDK present
    },
    "linux": {
        "cpu":    ["bin-ubuntu-x64"],
        "vulkan": ["bin-ubuntu-vulkan-x64"],
        "cuda":   ["bin-ubuntu-cuda"],
        "sycl":   ["bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        "hip":    ["bin-ubuntu-rocm"],
    },
    "darwin": {
        "cpu":    ["bin-macos-arm64", "bin-macos-x64"],
        "metal":  ["bin-macos-arm64"],
    },
}

# Backends a user may request directly (in addition to the special "auto" and
# the self-contained "amd-rocm").
_UPSTREAM_BACKENDS = ("vulkan", "cuda", "sycl", "hip", "cpu", "metal")

# SEC-8: a prebuilt llama runtime archive is many megabytes. Anything below this
# floor is almost certainly an error page, a redirect stub, or a truncated
# transfer, never the real artifact. This is the always-on lower bound; the
# valid-archive structural check below is the second always-on guard. A pinned
# sha256 is the opt-in third guard (we do not hardcode a brittle hash for the
# live URLs, which move with every upstream release).
_MIN_ARTIFACT_BYTES = 256 * 1024   # 256 KiB


class ArtifactError(Exception):
    """A downloaded artifact failed integrity validation (size, archive shape,
    or a provided sha256 pin) and must NOT be extracted or installed."""


def _platform_key() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _lib_name() -> str:
    """The loadable llama library filename for this platform."""
    if sys.platform == "win32":
        return "llama.dll"
    if sys.platform == "darwin":
        return "libllama.dylib"
    return "libllama.so"


# A tiny marker file recording WHICH backend currently occupies the runtime lib
# dir. It exists so the "already provisioned" guard can be backend-aware: a later
# `setup-llama --backend cuda` on a box that already has a vulkan/cpu build must
# still fetch CUDA (R23), instead of short-circuiting on the mere presence of a
# library. A dotfile, like the venv's .localm-venv marker; never loaded as code.
_BACKEND_MARKER = ".localm-backend"


def _record_provisioned_backend(target: Path, backend: str) -> None:
    """Record *backend* as the one now provisioned in *target*. Best-effort: the
    marker only optimises the guard, so a write failure is non-fatal (the guard
    then conservatively re-provisions an explicit pick rather than skipping it).
    Written AFTER provisioning because _clear_target wipes the dir's files."""
    try:
        (target / _BACKEND_MARKER).write_text((backend or "").strip() + "\n",
                                              encoding="utf-8")
    except OSError:
        pass


def _provisioned_backend(target: Path) -> "Optional[str]":
    """The backend last provisioned into *target*, or None if unknown (no marker
    - e.g. an install predating the marker, or a hand-placed build). 'Unknown'
    is treated conservatively by the guard: an explicit pick is re-provisioned."""
    try:
        val = (target / _BACKEND_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return val or None


def _is_wanted(f: Path) -> bool:
    """Whether to copy *f*: the loadable library, its ggml deps, and the runtime
    libraries - matched by platform-appropriate naming (incl. versioned .so.N)."""
    n = f.name.lower()
    if sys.platform == "win32":
        return n.endswith(".dll") or n.endswith(".exe")
    if sys.platform == "darwin":
        return n.endswith(".dylib")
    return ".so" in n          # libfoo.so and libfoo.so.1


def _repo_runtime_lib() -> Path:
    """The localm-llama-runtime wheel's lib/ dir."""
    try:
        import localm_llama_runtime
        return Path(localm_llama_runtime.LIB_DIR)
    except Exception as e:
        # The wheel is legitimately ABSENT before `setup-llama` installs it, so
        # the repo-relative fallback is correct then - do NOT hard-fail. But a
        # BROKEN install (import error other than not-found) would otherwise be
        # invisible and lead to loading stale binaries, so surface it at debug
        # level (rule 5: do not silence) without breaking the not-yet-installed
        # path. Mirrors the visible-fallback pattern in _auto_backend below.
        logger.debug("localm_llama_runtime import failed (%s); "
                     "using the repo-relative runtime lib dir", e)
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / "runtime" / "localm_llama_runtime" / "lib"


def _runtime_pkg_dir() -> Path:
    """The runtime wheel project dir (for `pip install -e`)."""
    return _repo_runtime_lib().parent.parent


# --------------------------------------------------------------------------- #
#  Backend resolution                                                          #
# --------------------------------------------------------------------------- #

def _auto_backend() -> str:
    """Pick the broadest WORKING backend for this machine (see module docstring).

    AMD keeps the self-contained ROCm build (no toolkit, current behaviour);
    every other GPU uses vulkan (no vendor toolkit needed); Apple Silicon uses
    metal; a machine with no GPU uses cpu."""
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception as e:
        # Surface the skipped GPU setup so a detection failure is visible and the
        # user knows how to force a GPU backend, rather than a silent CPU default.
        console.print(f"[yellow]GPU detection failed ({e}); defaulting to CPU - "
                      "override with --backend.[/yellow]")
        return "cpu"
    if not det.has_gpu:
        return "cpu" if sys.platform != "darwin" else det.recommended
    if det.vendors == ["amd"] and sys.platform == "win32":
        return "amd-rocm"                 # self-contained gfx103X build
    if "apple" in det.vendors:
        return "metal"
    return "vulkan"                       # NVIDIA / Intel / mixed: universal


def _latest_tag() -> str:
    """The latest ggml-org/llama.cpp release tag, or the pinned fallback if the
    lookup is unavailable (offline, rate-limited, etc.)."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases/latest"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        tag = data.get("tag_name")
        if isinstance(tag, str) and tag:
            return tag
    except Exception:
        pass
    # Surface the fallback so the user knows the build may not be current (the
    # latest-release lookup was unreachable); offer to rerun later for the latest.
    console.print(f"[yellow]Could not reach GitHub; using pinned llama.cpp "
                  f"{_FALLBACK_TAG} - rerun later for the latest.[/yellow]")
    return _FALLBACK_TAG


def _resolve_backend_url(backend: str) -> str:
    """Resolve a backend name to a downloadable archive URL.

    ``amd-rocm`` is the self-contained lemonade build (special-cased). Every
    other backend maps to an upstream llama.cpp release asset for this platform.
    Raises ``click.ClickException`` if the backend is not available here."""
    if backend == "amd-rocm":
        if sys.platform != "win32":
            raise click.ClickException(
                "the self-contained 'amd-rocm' build is Windows-only; on Linux "
                "use --backend hip (needs ROCm) or build with --from.")
        return DEFAULT_URL

    plat = _platform_key()
    matchers = _ASSET_MATCH.get(plat, {}).get(backend)
    if not matchers:
        avail = ", ".join(sorted(_ASSET_MATCH.get(plat, {})))
        raise click.ClickException(
            f"backend {backend!r} is not available on this platform "
            f"({plat}). Available: {avail or 'none'}.")

    tag = _latest_tag()
    # Ask the release for its assets so we match the real (version-suffixed)
    # name; fall back to a templated guess if the asset list is unavailable.
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assets = json.loads(r.read().decode("utf-8")).get("assets", [])
        for a in assets:
            name = str(a.get("name", "")).lower()
            # Exclude the cudart RUNTIME bundle: cudart-llama-bin-win-cuda-12.x
            # contains the same "bin-win-cuda-12" fragment as the build and is
            # often listed first, so without this the cuda build resolves to the
            # runtime-only zip (no llama.dll) and provisioning aborts with "the
            # archive did not contain llama.dll" (NEW-CUDADLL). The build is
            # always llama-..., never cudart-..., so this is safe for every backend.
            if (any(m in name for m in matchers) and "cudart" not in name
                    and a.get("browser_download_url")):
                return a["browser_download_url"]
    except Exception:
        pass
    # Fallback: construct the canonical URL from the first matcher token. The
    # asset list could not be fetched, so this URL is a guess that was NOT
    # confirmed against the release - warn so a later 404 is attributable to it
    # (offline/rate-limited) rather than looking like a silent failure.
    stem = matchers[0]
    ext = "zip" if plat == "win32" else "tar.gz"
    fname = f"llama-{tag}-{stem}.{ext}"
    guess = f"https://github.com/{_UPSTREAM_REPO}/releases/download/{tag}/{fname}"
    console.print(f"[yellow]Could not verify release asset list; using unverified URL: {guess}[/yellow]\n"
                  "[yellow]If download fails, pass --from <build dir> or --url <archive>.[/yellow]")
    return guess


# --------------------------------------------------------------------------- #
#  Download / validate / extract                                              #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> None:
    console.print(f"[dim]Downloading {url}[/dim]")
    last = [-1]

    def _hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, block * block_size * 100 // total)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            mb = total / 1024 ** 2
            console.print(f"[dim]  {pct:3d}%  ({mb:.0f} MB)[/dim]", end="\r")

    urllib.request.urlretrieve(url, dest, _hook)
    console.print()


def _sha256_file(path: Path) -> str:
    """Stream the file through sha256 so a multi-hundred-MB artifact is not
    read into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_supported_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _validate_archive(
    path: Path,
    expected_sha256: Optional[str] = None,
    min_size: int = _MIN_ARTIFACT_BYTES,
) -> None:
    """SEC-8: validate a downloaded artifact BEFORE it is extracted or installed.
    Raises :class:`ArtifactError` on any failure.

    Three checks, in cheapest-first order:
      1. size: a real prebuilt runtime archive is many MB; a tiny/empty body is
         an error page, a redirect stub, or a truncated transfer (always on).
      2. shape: it must be a structurally valid zip OR tar archive, so a
         200-with-HTML or a half-transferred file is rejected before we hand it
         to extraction (always on).
      3. provenance: when *expected_sha256* is given, the file's digest must
         match it (opt-in; refuses on mismatch). Comparison is whitespace- and
         case-insensitive so a pasted hash from any source works.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ArtifactError(f"could not stat downloaded file: {e}") from e
    if size < min_size:
        raise ArtifactError(
            f"download is too small ({size} bytes < {min_size} minimum) - "
            "this is almost certainly an error page or a truncated transfer, "
            "not the prebuilt runtime."
        )
    if not _is_supported_archive(path):
        raise ArtifactError(
            "download is not a valid zip or tar archive - it may be a truncated "
            "transfer, an HTML error page served with a 200, or a tampered "
            "payload."
        )
    if expected_sha256:
        want = expected_sha256.strip().lower()
        got = _sha256_file(path)
        if got != want:
            raise ArtifactError(
                "download sha256 does not match the expected pin "
                f"(expected {want}, got {got}). Refusing to install a "
                "possibly tampered or wrong artifact."
            )


def _safe_extractall_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe tar extraction for Python < 3.12, which has no
    extraction ``filter`` keyword. Backports the 'data' filter's core guarantee:
    every member, and every symlink/hardlink TARGET, must resolve INSIDE *dest*.
    Absolute, drive-letter, ``..`` and escaping-link members are refused. On
    Python 3.12+ ``filter="data"`` is used directly (see _extract_archive), so
    this is only the older-interpreter path - but it must be just as safe."""
    dest_resolved = dest.resolve()

    def _contained(p: Path) -> bool:
        rp = p.resolve()
        return rp == dest_resolved or dest_resolved in rp.parents

    for m in tf.getmembers():
        name = m.name
        if name.startswith(("/", "\\")) or ".." in Path(name).parts \
                or (len(name) > 1 and name[1] == ":"):
            raise ArtifactError(f"unsafe path in archive: {name!r}")
        if not _contained(dest / name):
            raise ArtifactError(f"unsafe path in archive: {name!r}")
        if m.issym() or m.islnk():
            link = m.linkname
            if link.startswith(("/", "\\")) or (len(link) > 1 and link[1] == ":") \
                    or not _contained(dest / Path(name).parent / link):
                raise ArtifactError(
                    f"unsafe link target in archive: {name!r} -> {link!r}")
    tf.extractall(dest)


def _extract_archive(path: Path, dest: Path) -> None:
    """Extract a validated zip or tar.gz into *dest*, refusing any member that
    would escape *dest* (an absolute path, a drive letter, or a ``..`` segment).
    Tar uses the 'data' filter (Python 3.12+), or a hand-rolled equivalent
    (_safe_extractall_tar) on older interpreters, for the same guarantee."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.startswith(("/", "\\")) or ".." in Path(n).parts \
                        or (len(n) > 1 and n[1] == ":"):
                    raise ArtifactError(f"unsafe path in archive: {n!r}")
            zf.extractall(dest)
        return
    with tarfile.open(path) as tf:
        try:
            tf.extractall(dest, filter="data")     # py3.12+: path-traversal safe
        except TypeError:
            _safe_extractall_tar(tf, dest)         # py<3.12: same guarantee, by hand


# Fallback notice written next to the bundled binaries when the upstream archive
# ships no LICENSE file, so a release never redistributes the MIT-licensed
# llama.cpp/ggml binaries without their license text (MIT requires it).
_LLAMA_CPP_MIT_NOTICE = """MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_LICENSE_NAME_PREFIXES = ("license", "licence", "copying", "notice")


def _copy_license_files(src_dir: Path, target: Path) -> int:
    """Copy upstream license/notice files from *src_dir* into *target* so the MIT
    text travels with the binaries. Falls back to a bundled llama.cpp/ggml MIT
    notice when the archive ships none. Returns the number of files written."""
    found = [f for f in sorted(src_dir.rglob("*"))
             if _safe_is_file(f)
             and any(f.name.lower().startswith(k) for k in _LICENSE_NAME_PREFIXES)]
    if found:
        written = 0
        for i, f in enumerate(found):
            dest = target / ("LICENSE.llama-cpp" if i == 0
                             else f"LICENSE.llama-cpp.{i}")
            try:
                shutil.copy2(f, dest)
                written += 1
            except OSError:
                pass
        if written:
            return written
    (target / "LICENSE.llama-cpp").write_text(_LLAMA_CPP_MIT_NOTICE, encoding="utf-8")
    return 1


def _safe_is_file(f: Path) -> bool:
    try:
        return f.is_file()
    except OSError:
        return False


def _copy_binaries(src_dir: Path, target: Path) -> int:
    """Copy the llama/ggml/runtime libraries from *src_dir* (recursively) into
    *target*. Returns the number of files copied."""
    n = 0
    for f in src_dir.rglob("*"):
        if f.is_file() and _is_wanted(f):
            shutil.copy2(f, target / f.name)
            n += 1
    # MIT requires the license to accompany the binaries; capture it (or a
    # bundled fallback) alongside them whenever we actually placed binaries.
    if n:
        _copy_license_files(src_dir, target)
    return n


def _install_runtime_wheel(pkg_dir: Path) -> bool:
    """Install the runtime wheel editable into the active venv. Tries uv, then
    pip. Returns True on success."""
    last_err = ""
    for cmd in (["uv", "pip", "install", "-e", str(pkg_dir)],
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return True
            # Keep the real pip/uv failure instead of discarding it, so the user
            # can see the actual cause (missing build tools, conflicting deps).
            last_err = (r.stderr or r.stdout or "").strip()
        except FileNotFoundError:
            continue
    # Surface the last failed attempt's output: full to the debug log, a trimmed
    # tail to stderr so the caller's "did not load" path has the real reason.
    if last_err:
        logger.debug("runtime wheel install failed: %s", last_err)
        tail = "\n".join(last_err.splitlines()[-8:])
        console.print(f"[yellow]Runtime wheel install failed:[/yellow]\n{tail}")
    return False


# --------------------------------------------------------------------------- #
#  NVIDIA / CUDA preflight + self-assembly                                     #
#                                                                              #
#  CUDA is the visible "peak NVIDIA performance" option, so picking it has to  #
#  LAND. The CUDA llama build needs the CUDA *runtime* libraries (cudart /     #
#  cublas) at load time. Upstream ships a self-contained                       #
#  ``cudart-llama-bin-win-cuda-<ver>`` bundle in the SAME release, so we make  #
#  CUDA work WITHOUT the user installing the full CUDA Toolkit: fetch the      #
#  build + the matching cudart bundle into the same lib dir. The one thing we  #
#  cannot self-assemble is the GPU DRIVER (a system component needing admin +  #
#  a reboot); a too-old driver is the single "you must do this part" branch.   #
# --------------------------------------------------------------------------- #

# Target the CUDA 12.x line: it is the broad-compatibility choice (runs on any
# driver new enough for CUDA 12.4). Upstream also ships a 13.x line that needs a
# newer driver; we resolve the build and its matching cudart bundle from 12.x.
_CUDA_LINE = "cuda-12"
_MIN_DRIVER_CUDA = (12, 4)


def _ver_tuple(v: str) -> Optional[tuple]:
    # Return None (not (0,0)) on an unparseable version so an unmeasurable
    # capability reads as "unknown", never as a too-old driver we falsely block.
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return None


@dataclass
class NvidiaInfo:
    """What nvidia-smi told us. Advisory only; every field may be empty."""
    present: bool = False           # an NVIDIA GPU + usable driver was found
    gpu_name: str = ""
    driver_version: str = ""
    cuda_capability: str = ""       # max CUDA the driver supports, e.g. "12.4"

    @property
    def driver_ok(self) -> bool:
        """True when the driver is new enough for the CUDA build we install.
        Unknown capability is treated as OK (do not block on a parse miss)."""
        if not self.cuda_capability:
            return True
        parsed = _ver_tuple(self.cuda_capability)
        # An unparseable capability is unknown, not old: cannot judge, do not block.
        if parsed is None:
            return True
        return parsed >= _MIN_DRIVER_CUDA


def _nvidia_smi(*args: str) -> str:
    """Combined nvidia-smi output, or "" if it is not present/usable."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=8)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def nvidia_preflight() -> NvidiaInfo:
    """Detect the NVIDIA GPU + driver and the max CUDA version it supports.

    Never raises. Parses the nvidia-smi banner ("Driver Version: X  CUDA
    Version: Y") and asks explicitly for the (untruncated) GPU name."""
    out = _nvidia_smi()
    if not out.strip():
        return NvidiaInfo(present=False)
    info = NvidiaInfo(present=True)
    m = re.search(r"Driver Version:\s*([0-9.]+)", out)
    if m:
        info.driver_version = m.group(1)
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    if m:
        info.cuda_capability = m.group(1)
    name = _nvidia_smi("--query-gpu=name", "--format=csv,noheader").strip().splitlines()
    if name:
        info.gpu_name = name[0].strip()
    return info


def _release_assets(tag: str) -> list:
    """The asset list for a release tag, or [] if the API is unavailable."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8")).get("assets", [])
    except Exception:
        return []


def _pick_asset(assets: list, *needles: str, exclude: tuple = ()) -> Optional[dict]:
    """First asset whose (lowercased) name contains ALL *needles* and NONE of
    *exclude*."""
    for a in assets:
        name = str(a.get("name", "")).lower()
        if (all(n in name for n in needles)
                and not any(x in name for x in exclude)
                and a.get("browser_download_url")):
            return a
    return None


def _resolve_cuda_pair(tag: str) -> tuple:
    """(build_asset, cudart_asset) for the Windows CUDA 12.x line. Either may be
    None when the release listing is unavailable or lacks it.

    The build and the cudart runtime share the "...bin-win-cuda-12.x..." name
    fragment (the runtime is e.g. cudart-llama-bin-win-cuda-12.4-x64.zip), and
    the runtime is often listed FIRST, so the build matcher MUST exclude
    "cudart" - otherwise build resolves to the runtime-only zip (CUDA DLLs, no
    llama.dll) and provisioning aborts with "the archive did not contain
    llama.dll" (NEW-CUDADLL)."""
    assets = _release_assets(tag)
    build = _pick_asset(assets, "bin-win-" + _CUDA_LINE, exclude=("cudart",))
    cudart = _pick_asset(assets, "cudart", "win-" + _CUDA_LINE)
    return build, cudart


def _human_mb(nbytes) -> str:
    try:
        return f"{int(nbytes) / 1024 ** 2:.0f} MB"
    except Exception:
        return "?"


def _fetch_and_place(url: str, target: Path, sha256: Optional[str] = None) -> int:
    """Download -> validate -> extract -> copy one prebuilt archive into
    *target*. Returns the number of binary files copied. Raises on a download or
    validation failure (the caller decides fatal-vs-fallback)."""
    with tempfile.TemporaryDirectory() as tmp:
        suffix = ".zip" if url.lower().endswith(".zip") else ".tar.gz"
        arc = Path(tmp) / f"llama-prebuilt{suffix}"
        _download(url, arc)
        _validate_archive(arc, expected_sha256=sha256)   # SEC-8 gate, pre-extract
        ex = Path(tmp) / "x"
        _extract_archive(arc, ex)
        return _copy_binaries(ex, target)


def _clear_target(target: Path) -> None:
    """Remove previously provisioned library files so a re-provision (or a
    fallback to a different backend) never mixes DLLs from two builds. Only
    touches files in the dir, never subdirectories."""
    try:
        for f in target.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _provision_backend(chosen: str, target: Path, sha256: Optional[str],
                       with_cudart: bool) -> None:
    """Resolve + fetch the prebuilt(s) for *chosen* into *target*. For CUDA with
    *with_cudart* it also fetches the matching cudart runtime bundle so the
    build is self-contained (no CUDA Toolkit needed). Raises on a fatal error."""
    if chosen == "cuda" and with_cudart and sys.platform == "win32":
        tag = _latest_tag()
        build, cudart = _resolve_cuda_pair(tag)
        if build is None:
            # Asset listing unavailable: fall back to the templated build URL and
            # warn that the runtime bundle could not be resolved automatically.
            console.print("[yellow]Could not resolve CUDA assets; fetching build only.[/yellow]\n"
                          "[yellow]If it fails to load, use --backend vulkan or install CUDA Toolkit.[/yellow]")
            _fetch_and_place(_resolve_backend_url("cuda"), target, sha256)
            return
        console.print(f"[dim]CUDA build:[/dim] {build['name']} ({_human_mb(build.get('size'))})")
        _fetch_and_place(build["browser_download_url"], target, sha256)
        if cudart is not None:
            if sha256:
                # The pin is a single hash; it can only cover the build. Be honest
                # that the cudart bundle is validated by size + archive shape, not
                # by the pinned digest (upstream publishes no per-asset hash here).
                console.print("[yellow]Note:[/yellow] --sha256 pins the CUDA build only.")
            console.print(f"[dim]CUDA runtime:[/dim] {cudart['name']} "
                          f"({_human_mb(cudart.get('size'))}) - no Toolkit install needed")
            _fetch_and_place(cudart["browser_download_url"], target)
        else:
            console.print("[yellow]No cudart bundle found; CUDA Toolkit may be required.[/yellow]")
        return
    # Every other backend is a single archive resolved from the chosen name.
    _fetch_and_place(_resolve_backend_url(chosen), target, sha256)


def _native_loads_ok() -> tuple:
    """Load-test the provisioned native library in a FRESH interpreter, exactly
    as ``localm run`` will, AND confirm it registered a compute backend. A build
    can load cleanly yet register ZERO backends ("no backends are loaded"), which
    must count as a FAILED provision, not a silent success (AGENTS.md rule 5) -
    otherwise _provision_with_fallback's "prove it loads" guarantee holds only for
    self-registering builds and a broken runtime slips through, failing only at
    the first model load with the real cause already lost. A subprocess keeps the
    setup process clean (the loader mutates the DLL/lib search path) and matches
    the real run environment. Returns (ok, last_error_line)."""
    # Exit 88 distinguishes "loaded but no compute backend" from a load crash
    # (non-zero with a native traceback) and a clean, computing load (0).
    code = ("from localm.inference.backends.llamacpp import _loader; "
            "_loader.load_lib(); "
            "import sys; sys.exit(0 if _loader.compute_backends_available() else 88)")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, str(e)
    if r.returncode == 0:
        return True, ""
    if r.returncode == 88:
        return False, ('runtime loaded but registered no compute backends '
                       '("no backends are loaded") - this build does not fit this machine')
    detail = (r.stderr or r.stdout or "").strip()
    return False, (detail.splitlines()[-1] if detail else "library failed to load")


def _warn_off_profile(chosen: str) -> None:
    """One-line heads-up when a vendor-specific backend was chosen for a vendor
    we did NOT detect. We respect the user's choice - no block, no nag, no
    re-prompt - just flag it once so a misclick is visible."""
    vendor_specific = {"cuda": "nvidia", "amd-rocm": "amd", "hip": "amd",
                       "sycl": "intel", "metal": "apple"}
    owner = vendor_specific.get(chosen)
    if not owner:
        return
    try:
        from localm import hwdetect
        vendors = hwdetect.detect().vendors or []
    except Exception:
        return
    if vendors and owner not in vendors:
        seen = ", ".join(vendors)
        console.print(f"[yellow]Heads up:[/yellow] Picked [bold]{chosen}[/bold] but detected [bold]{seen}[/bold].\n"
                      "[yellow]Proceeding. Hardware must be present.[/yellow]")


def _cuda_setup_dialogue(info: NvidiaInfo, assume_yes: bool) -> tuple:
    """Given the preflight, walk the user through making CUDA land. Returns
    ``(backend_to_provision, fetch_cudart_bundle)``.

    Branches:
      * driver new enough  -> offer the self-contained build+runtime fetch
        (default yes); declining falls back to vulkan.
      * driver too old     -> a driver cannot be self-assembled; recommend
        vulkan now and tell them how to enable CUDA later.
      * no NVIDIA detected  -> the user forced cuda; confirm-continue, else
        vulkan. (warn-once, do not block.)
    """
    console.print("[bold]CUDA selected[/bold] (peak NVIDIA performance). "
                  "Checking your system...")
    if info.present:
        console.print(f"  [green]OK[/green] NVIDIA GPU: {info.gpu_name or 'detected'}")
        if info.cuda_capability:
            colour = "green" if info.driver_ok else "red"
            mark = "OK " if info.driver_ok else "no "
            console.print(f"  [{colour}]{mark}[/{colour}] Driver {info.driver_version} "
                          f"supports CUDA {info.cuda_capability} "
                          f"(need >= {_MIN_DRIVER_CUDA[0]}.{_MIN_DRIVER_CUDA[1]})")
    else:
        console.print("  [yellow]?[/yellow] Could not run nvidia-smi - no NVIDIA driver "
                      "detected here (or it is not on PATH).")

    # Driver too old: the one thing we cannot fetch for the user.
    if info.present and info.cuda_capability and not info.driver_ok:
        console.print("  GPU driver update required for CUDA.")
        console.print("  [dim]To enable later: update driver, reboot, run setup-llama --backend cuda[/dim]")
        console.print("  [green]Using Vulkan now[/green].")
        return "vulkan", False

    # No NVIDIA detected, but the user explicitly asked for cuda: their call.
    if not info.present:
        if assume_yes:
            console.print("  [dim]--yes: using Vulkan (no NVIDIA GPU detected).[/dim]")
            return "vulkan", False
        if click.confirm("  Continue with CUDA anyway? (No = use Vulkan)", default=False):
            return "cuda", True
        return "vulkan", False

    # Driver OK (or capability unknown but a GPU is present): offer the fetch.
    console.print("  [yellow]i[/yellow] Fetching self-contained CUDA runtime bundle. [bold]No Toolkit needed[/bold].")
    if assume_yes or click.confirm("  Download the CUDA build + runtime now?", default=True):
        return "cuda", True
    console.print("  [dim]Falling back to Vulkan (works on your driver).[/dim]")
    return "vulkan", False


def _provision_with_fallback(chosen: str, target: Path, sha256: Optional[str],
                             with_cudart: bool, assume_yes: bool = False) -> str:
    """Provision *chosen* and prove it loads. If it does not load, NEVER swap the
    user's pick silently (the never-override rule): inform WHY, then OFFER the
    universal Vulkan build when interactive (or fall back with a LOUD warning when
    *assume_yes* / no tty), and always say how to retry the chosen backend with
    --force. Returns the backend that ended up working. Exits non-zero if the user
    declines the fallback, or if NOTHING loads (a genuine environment fault).

    vulkan and cpu are self-contained and treated as terminal: if the user
    explicitly chose one and it does not load, that is an environment problem we
    report rather than paper over with a different backend."""
    lib_name = _lib_name()

    def _try(backend: str, cudart: bool) -> None:
        _clear_target(target)
        _provision_backend(backend, target, sha256 if backend == chosen else None, cudart)
        if not (target / lib_name).exists():
            raise ArtifactError(f"the archive did not contain {lib_name}")
        _install_runtime_wheel(_runtime_pkg_dir())

    notes = {
        "vulkan": "universal GPU build (AMD/NVIDIA/Intel via the display driver)",
        "amd-rocm": "self-contained AMD ROCm build (gfx103X / RX 6000)",
        "cuda": "NVIDIA CUDA build + self-contained runtime",
        "sycl": "Intel oneAPI build (needs the oneAPI runtime present)",
        "hip": "AMD ROCm build (needs the ROCm/HIP runtime present)",
        "cpu": "CPU-only build (no GPU)",
        "metal": "Apple Silicon (Metal) build",
    }
    console.print(f"[dim]Backend:[/dim] [bold]{chosen}[/bold]  ({notes.get(chosen, chosen)})")

    provisioned = True
    try:
        _try(chosen, with_cudart)
    except click.ClickException as e:
        console.print(f"[red]{e.message}[/red]")
        provisioned = False
    except (ArtifactError, OSError) as e:
        console.print(f"[red]Provisioning {chosen} failed:[/red] {e}")
        provisioned = False
    except Exception as e:
        console.print(f"[red]Provisioning {chosen} failed:[/red] {e}")
        provisioned = False

    loaded, detail = (_native_loads_ok() if provisioned else (False, "not provisioned"))
    if loaded:
        console.print(f"[green]OK - {chosen} runtime loads on this machine.[/green]")
        return chosen

    # An explicit --sha256 pin means "exactly this artifact" - never silently
    # swap to a different (unpinned) build, even to recover. Report and stop.
    if sha256:
        why = "failed validation" if not provisioned else "provisioned but did not load"
        console.print(f"[red]The pinned artifact {why}.[/red] Not falling back "
                      "(an explicit --sha256 was set).")
        sys.exit(1)

    # A self-contained backend the user pinned: do not silently swap to another.
    if chosen in ("vulkan", "cpu"):
        if not provisioned:
            sys.exit(1)
        # Provisioned but would not load: vulkan/cpu are the universal fallbacks,
        # so this is an unexpected environment fault worth a report - not an exit-0
        # "success" on a broken runtime.
        from localm.bugreport import LocalmError
        raise LocalmError(
            f"{chosen} was provisioned but the native library did not load",
            reason=(f"{chosen} is the self-contained fallback and still failed to load "
                    f"({detail}) - likely a broken/incompatible binary or a missing OS "
                    "dependency. See docs/gpu-setup.md."),
            context={"operation": "setup-llama", "backend": chosen})

    # chosen needs a runtime and did not load HERE. Honour the user's pick: never
    # swap it silently. INFORM why, then OFFER the universal build (interactive)
    # or fall back with a LOUD warning (non-interactive), and always say how to
    # retry the real pick once the cause is fixed (R20 / never-override-user-selection).
    why = detail
    console.print(f"[yellow]'{chosen}' backend provisioned but failed to load: {why}[/yellow]")
    console.print(f"[dim]To retry later: localm setup-llama --backend {chosen} --force[/dim]")
    interactive = (not assume_yes) and sys.stdin.isatty()
    if interactive:
        if not click.confirm(
                f"  Install the universal Vulkan build now so you have a working "
                f"setup? (your '{chosen}' pick is kept, not changed; decline to stop "
                f"and fix it yourself)", default=True):
            console.print(f"[yellow]Keeping your '{chosen}' choice and stopping.[/yellow] "
                          "It does not load here yet. Fix the cause, then re-run: "
                          f"localm setup-llama --backend {chosen} --force")
            sys.exit(1)
    else:
        console.print("[yellow][!] Non-interactive: falling back to universal build.[/yellow]")
    for fb in ("vulkan", "cpu"):
        console.print(f"[yellow]Trying {fb}...[/yellow]")
        try:
            _try(fb, False)
        except Exception as e:
            console.print(f"[red]{fb} provisioning failed:[/red] {e}")
            continue
        ok, _ = _native_loads_ok()
        if ok:
            console.print(f"[green]OK - {fb} runtime loads.[/green]")
            return fb
    # Nothing loaded - the one genuinely stuck case. Raise a typed, reportable
    # error and let the CLI's single graceful handler say sorry + offer a bug
    # report. setup-llama describes the failure; it does not own reporting.
    from localm.bugreport import LocalmError
    raise LocalmError(
        "no llama.cpp backend could be provisioned and loaded",
        reason=(f"tried {chosen}, then vulkan and cpu - none loaded on this machine "
                f"(last error: {detail or 'unknown'}). You can provide a local build "
                "with: localm setup-llama --from <build dir>, or see docs/gpu-setup.md."),
        context={"operation": "setup-llama", "requested_backend": chosen})


@click.command("setup-llama", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Copy binaries from a local llama.cpp build directory instead of downloading.")
@click.option("--backend", default="auto",
              type=click.Choice(["auto", "vulkan", "cuda", "sycl", "hip", "cpu",
                                 "metal", "amd-rocm"], case_sensitive=False),
              help="Which prebuilt to fetch. 'auto' detects your GPU and picks "
                   "the broadest working backend: vulkan for NVIDIA/Intel; the "
                   "self-contained ROCm build for AMD on Windows (vulkan for AMD "
                   "on Linux); cpu if no GPU.")
@click.option("--url", default=None, help="Override with an explicit prebuilt archive URL.")
@click.option("--sha256", "sha256", default=None,
              help="Expected sha256 of the downloaded archive. When given, the "
                   "download is refused unless its digest matches (opt-in "
                   "integrity pin).")
@click.option("--force", is_flag=True, help="Re-provision even if binaries are already present.")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Non-interactive: accept the recommended action at every prompt "
                   "(e.g. fetch the self-contained CUDA runtime). Used by the "
                   "one-click installer and for scripted setups.")
def main(from_dir: Optional[str], backend: str, url: Optional[str],
         sha256: Optional[str], force: bool, assume_yes: bool) -> None:
    """Download or copy the native llama.cpp binaries into localm's own venv.

    The chosen backend is load-tested after provisioning. If it cannot load on
    this machine (e.g. CUDA without a new-enough driver) your pick is NOT changed
    silently: setup explains why and (interactively) offers the universal Vulkan
    build instead, or - in a non-interactive install - falls back with a loud
    warning and tells you how to retry your backend once the cause is fixed.

    \b
      localm setup-llama                        # auto-detect GPU, fetch the right prebuilt
      localm setup-llama --backend vulkan       # universal GPU build (any vendor)
      localm setup-llama --backend cuda         # NVIDIA: checks the driver, fetches a
                                                #   self-contained CUDA runtime (no Toolkit)
      localm setup-llama --backend cpu          # no GPU
      localm setup-llama --from /path/to/llama.cpp/build/bin
      localm setup-llama --url https://.../llama-...zip
      localm setup-llama --sha256 <hex>         # pin the expected archive digest
    """
    lib_name = _lib_name()
    target = _repo_runtime_lib()
    target.mkdir(parents=True, exist_ok=True)

    already = (target / lib_name).exists()
    if already and not force:
        # Backend-aware guard (R23): 'auto' means "give me something that works",
        # and something already does, so do not re-download. An EXPLICIT backend
        # is honoured - short-circuit only when we can confirm THAT backend is the
        # one on disk; otherwise (a different recorded backend, or none recorded)
        # fall through and provision what the user asked for. This is what lets
        # `setup-llama --backend cuda` on a box that already has a vulkan/cpu build
        # actually fetch CUDA, instead of keeping the old runtime silently.
        want = backend.lower()
        have = _provisioned_backend(target)
        if want == "auto" or (have is not None and have == want):
            label = f" ({have})" if have else ""
            console.print(f"[green]Already provisioned[/green]{label} at {target}")
            console.print("[dim]Use --force to re-download/replace.[/dim]")
            _ensure_importable()
            return
        if have:
            console.print(f"[yellow]Replacing {have} build with {want}.[/yellow]")
        else:
            console.print(f"[yellow]Replacing unrecorded build with {want}.[/yellow]")

    if from_dir:
        src = Path(from_dir)
        console.print(f"Copying binaries from [bold]{src}[/bold] ...")
        _clear_target(target)
        n = _copy_binaries(src, target)
        if not (target / lib_name).exists():
            console.print(f"[red]No {lib_name} found in the source directory.[/red] "
                          f"Point --from at the build output containing {lib_name}.")
            sys.exit(1)
        console.print(f"[green]Copied {n} file(s)[/green] into {target}")
        _install_runtime_wheel(_runtime_pkg_dir())
        loaded, detail = _native_loads_ok()
        if loaded:
            console.print("[green]OK - the provided build loads on this machine.[/green]")
            _record_provisioned_backend(target, "custom")
        else:
            # The user pinned this build, so we do NOT fall back - but we must not
            # report success on a library that will not load. Exit non-zero with a
            # clear reason rather than leaving a broken runtime behind.
            console.print(f"[red]Copied, but the library did not load[/red] "
                          f"({detail}) - is it built for this OS/GPU? "
                          "See docs/gpu-setup.md.")
            sys.exit(1)
    elif url:
        console.print(f"[dim]Fetching:[/dim] {url}")
        try:
            _clear_target(target)
            _fetch_and_place(url, target, sha256)
        except ArtifactError as e:
            console.print(f"[red]Refusing to install:[/red] {e}")
            console.print("Provide a local build with --from instead, or a different "
                          "--url (and --sha256 if you pin one).")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {e}")
            console.print("Provide a local build with --from instead, or a different --url.")
            sys.exit(1)
        if not (target / lib_name).exists():
            console.print(f"[red]The archive did not contain {lib_name}.[/red] "
                          "Try a different --url or use --from.")
            sys.exit(1)
        _install_runtime_wheel(_runtime_pkg_dir())
        loaded, detail = _native_loads_ok()
        if loaded:
            console.print("[green]OK - the fetched build loads on this machine.[/green]")
            _record_provisioned_backend(target, "custom")
        else:
            # A user-pinned --url: do not fall back, but do not claim success on a
            # library that will not load. Exit non-zero with a clear reason.
            console.print(f"[red]Placed, but the library did not load[/red] "
                          f"({detail}). Is this build right for your OS/GPU? "
                          "See docs/gpu-setup.md.")
            sys.exit(1)
    else:
        chosen = _auto_backend() if backend == "auto" else backend
        # warn-once-then-comply: an explicit off-profile choice is the user's to
        # make, but flag a vendor mismatch a single time so a misclick is visible.
        if backend != "auto":
            _warn_off_profile(chosen)
        # CUDA is the visible peak-NVIDIA option: detect the driver, then offer to
        # fetch a self-contained runtime (no Toolkit) or fall back cleanly.
        with_cudart = False
        if chosen == "cuda" and sys.platform == "win32":
            chosen, with_cudart = _cuda_setup_dialogue(nvidia_preflight(), assume_yes)
        result = _provision_with_fallback(chosen, target, sha256, with_cudart,
                                          assume_yes)
        _record_provisioned_backend(target, result)

    _verify()


def _ensure_importable() -> None:
    try:
        import localm_llama_runtime  # noqa: F401
    except Exception:
        if _install_runtime_wheel(_runtime_pkg_dir()):
            console.print("[green]OK[/green] localm-llama-runtime installed.")


def _verify() -> None:
    try:
        from localm.inference.backends.llamacpp._loader import runtime_binary_dir
        d = runtime_binary_dir()
        if d:
            console.print(f"[bold green]Native runtime ready[/bold green] -> {d}")
            console.print("Try it:  [bold]localm run <model>[/bold]")
        else:
            console.print("[yellow]Binaries placed but not yet resolvable - "
                          "restart your shell so the new package is importable.[/yellow]")
    except Exception:
        pass
