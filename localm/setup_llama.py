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
import socket
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
# release tag with uploaded assets at runtime; this pin is the fallback if that
# lookup is unavailable.
_UPSTREAM_REPO = "ggml-org/llama.cpp"
_FALLBACK_TAG = "b9870"

# Pinned fallback checksums for tag b9870 and b1288 (lemonade AMD build) release assets
_PINNED_FALLBACK_SHA256 = {
    # tag b1288 ROCm assets
    "llama-b1288-windows-rocm-gfx103X-x64.zip": "18a85d4be9052f8377ca7e7ade4bae6c0a2818b3367989a6eb3297bcb4282b5e",
    "llama-b1288-windows-rocm-gfx110X-x64.zip": "1271b7088d934e6f443ac7e32b206a2aaa9297b232f52492f0039d9a8a8820aa",
    "llama-b1288-windows-rocm-gfx1150-x64.zip": "33f7f8917ce0dd3f09d9b84269d60db286f2684b238f6b558c0788a4ef54df3a",
    "llama-b1288-windows-rocm-gfx1151-x64.zip": "c94198d329256ac6f33c2e9701885a960574cd79d9df61422091e60976bd572f",
    "llama-b1288-windows-rocm-gfx120X-x64.zip": "d7416e3a5bf7c4c058b5e729cf98d540eaf095614e66d4b207fc38e27af8ae24",
    "llama-b1288-windows-rocm-gfx908-x64.zip": "fd2fb30d31671fd16ba4e4e9288c2706c913d02725a700409bbcc80143318e8f",
    "llama-b1288-windows-rocm-gfx90a-x64.zip": "6f6d49bda990ecb2795ece9c1bf04c6f5827c770889dabe842f98b8d82e5c927",
    "llama-b1288-ubuntu-rocm-gfx103X-x64.zip": "bc793bf354444a1fe3a2c57aa4a190da6c00bf6953c6864de7ab83208cb1a1a5",
    "llama-b1288-ubuntu-rocm-gfx110X-x64.zip": "3bd617bbb21731d727cf5948e0536caa6d7be4d7605542306d219704d6a01da3",
    "llama-b1288-ubuntu-rocm-gfx1150-x64.zip": "3db631e7fded551af4be1e63ffa74c78b1cfa493f83a28fb54ef7f5cdd2a7d2a",
    "llama-b1288-ubuntu-rocm-gfx1151-x64.zip": "4fec75e80673511a43c4f98e33ee019143bf2193778c17b1b26ef3e20afdad2e",
    "llama-b1288-ubuntu-rocm-gfx120X-x64.zip": "f0f5534e902cacbd1a159bca3708abb3dd76a1de597227896ee7f3e561b24381",
    "llama-b1288-ubuntu-rocm-gfx908-x64.zip": "bb54bdc6e4eafa0a20888613f8b62c6caf79b1bd941fb3e2d11f4741b9977984",
    "llama-b1288-ubuntu-rocm-gfx90a-x64.zip": "8c1d54115b820f30c6fff64b4a37bed6700dd4b882c39ee09d638558798cd19b",
    # tag b9870 upstream assets
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.3-x64.zip": "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    "llama-b9870-bin-android-arm64.tar.gz": "f9f8e6207ba97b6d34c98e6877c8beba23d462f08a73ace59ffb2ff5d134d26c",
    "llama-b9870-bin-macos-arm64.tar.gz": "9384fc29bfad58a665a617f3c5e490d5ab9f1f5506383b011d912f1bcc92804a",
    "llama-b9870-bin-macos-x64.tar.gz": "8f12b275bec2083caa13643471bd86083549659f48b2d1fad72c61e84bd5ee59",
    "llama-b9870-bin-ubuntu-arm64.tar.gz": "227564dead2145adf388d8fe3edbee8aeeea61e53cb151d03375661885ad8b1b",
    "llama-b9870-bin-ubuntu-openvino-2026.2.1-x64.tar.gz": "e6892a3531d70d079803075c8cfef9429a9f55510f58e39e8eb10ed84da3e18b",
    "llama-b9870-bin-ubuntu-rocm-7.2-x64.tar.gz": "b15e673147678ec5d89002cd77316f2eca1aa006e7a23b8fef499b6d5c7e9723",
    "llama-b9870-bin-ubuntu-s390x.tar.gz": "dbe9d21482f356fc8e058bcc35594fb59d69082444fb338f5bcf4bdb35f631e2",
    "llama-b9870-bin-ubuntu-sycl-fp16-x64.tar.gz": "77194249f0c800c26230c1ce919e282ab59647b75f8c9fc3e3f5ed59ab711d3a",
    "llama-b9870-bin-ubuntu-sycl-fp32-x64.tar.gz": "0abb480beb83f230678b397b93b9316b829485008819616abd6509d883ccd06a",
    "llama-b9870-bin-ubuntu-vulkan-arm64.tar.gz": "ba444f0d50b1e3807e8fab44adeb05455fa2da04f0c97f2e40fdbb3c410b0e46",
    "llama-b9870-bin-ubuntu-vulkan-x64.tar.gz": "e8a54099fb3e7afc48d85992ce45b5529298819da814d539a0593da61efe65c2",
    "llama-b9870-bin-ubuntu-x64.tar.gz": "16897263ccd016dd76c72a4d9b6ee27f975dae19bf652b4855b37dffbe7d4df1",
    "llama-b9870-bin-win-cpu-arm64.zip": "97b77bfbfd1889da5485552d0103f1e73a13b9ec4dfe924bf6d98543d225dab1",
    "llama-b9870-bin-win-cpu-x64.zip": "71be86e7af277e9503847c6050948ecd943d5e34b941e178a8af0c161b2d9a9e",
    "llama-b9870-bin-win-cuda-12.4-x64.zip": "10ced0b05eb1fdf47981dfe39e820a9465804b9250811f1173d935a22d336d6f",
    "llama-b9870-bin-win-cuda-13.3-x64.zip": "864a0a80b802124b34f19d3ce4cf327a2bd5fe9d41fe2dc21f7c63a0ed561979",
    "llama-b9870-bin-win-hip-radeon-x64.zip": "834196230bffe3847a553b680398c1438bfc85bdbab2d5061be28db5fd8648bb",
    "llama-b9870-bin-win-opencl-adreno-arm64.zip": "5b86328a27841d1aae3c477a414782d92f8135e46f5113dbca102474ae08115e",
    "llama-b9870-bin-win-openvino-2026.2.1-x64.zip": "802ac0c28c4096d126062644b0bcf94d4cd13c4dc7eed08f835b29164f3e6643",
    "llama-b9870-bin-win-sycl-x64.zip": "3b96c98aacac996ece92cd532c0a1a36215f14cd01c840c097fe72812c5d0c4b",
    "llama-b9870-bin-win-vulkan-x64.zip": "8687a8405447853ccbd6b15bd7ccda23bb79cf85dd83243401e514bd9e45ed8a",
    "llama-b9870-ui.tar.gz": "c0f6299ff94678fe9799cd09ef6c7f2d6fae9f55b86365b8825b5fc9c93772fa",
    "llama-b9870-xcframework.zip": "792cb6560abc2e04262b105eb9ca3d5890814f358f998adea4e28497788e59f7",
}

# Per-backend asset matcher: substrings that must appear in the release asset
# name for (platform, backend). Substring matching (not exact names) keeps this
# robust to upstream version suffixes drifting (e.g. cuda-12.4, rocm-7.2).
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-x64"],
        "vulkan": ["bin-win-vulkan-x64"],
        "cuda":   ["bin-win-cuda-12.4-x64", "bin-win-cuda-12"],          # prefer the 12.x runtime line
        "sycl":   ["bin-win-sycl-x64"],
        "hip":    ["bin-win-hip-radeon-x64"],   # needs AMD HIP SDK present
    },
    "linux": {
        "cpu":    ["bin-ubuntu-x64"],
        "vulkan": ["bin-ubuntu-vulkan-x64"],
        "cuda":   ["bin-ubuntu-cuda"],
        "sycl":   ["bin-ubuntu-sycl-fp16-x64", "bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        "hip":    ["bin-ubuntu-rocm-7.2-x64", "bin-ubuntu-rocm"],
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

# Per-read socket timeout for the archive download. urlretrieve honours the
# default socket timeout as an idle (between-reads) deadline, NOT a total-
# transfer cap, so a large-but-progressing download is never killed; only a
# genuinely stalled connection (no bytes for this many seconds) trips it. This
# turns an indefinite hang on a dropped/throttled transfer into a clear, loud
# error the caller reports, instead of a frozen progress line with no diagnostic.
_DOWNLOAD_STALL_TIMEOUT = 60   # seconds


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
    """The newest ggml-org/llama.cpp release tag that actually has its build
    assets uploaded, or the pinned fallback if no such release can be found
    (offline, rate-limited, etc.).

    Upstream publishes a release (tag + notes) as soon as it is cut, then its CI
    matrix uploads the ~25 platform archives afterwards - which can take a while.
    Right after publish, ``/releases/latest`` can point at a tag whose ``assets``
    array is still genuinely empty even though the release body already lists
    the (soon-to-exist) download URLs. Resolving to that tag anyway used to
    produce a confident-looking match that 404s, because the linked file simply
    is not there yet. So we scan recent releases newest-first and use the first
    one that already has assets, skipping any still-uploading release."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases?per_page=10"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            releases = json.loads(r.read().decode("utf-8"))
        for rel in releases:
            if rel.get("draft") or rel.get("prerelease"):
                continue
            tag = rel.get("tag_name")
            if isinstance(tag, str) and tag and rel.get("assets"):
                return tag
    except Exception:
        pass
    # Surface the fallback so the user knows the build may not be current (the
    # release lookup was unreachable, or none of the recent releases has its
    # assets uploaded yet); offer to rerun later for the latest.
    console.print(f"[yellow]Could not find a ggml-org/llama.cpp release with "
                  f"uploaded assets; using pinned llama.cpp {_FALLBACK_TAG} - "
                  "rerun later for the latest.[/yellow]")
    return _FALLBACK_TAG


def _resolve_backend_asset(backend: str) -> tuple[str, Optional[str]]:
    """Resolve a backend name to a (url, sha256_digest) pair.

    If the release listing is available, resolves it dynamically and gets the
    sha256 from the digest field. If offline, falls back to the templated guess
    and queries the local pinned checksum dictionary.
    """
    if backend == "amd-rocm":
        if sys.platform != "win32":
            raise click.ClickException(
                "the self-contained 'amd-rocm' build is Windows-only; on Linux "
                "use --backend hip (needs ROCm) or build with --from.")
        # Try to resolve dynamically first
        tag = "b1288"
        assets = _release_assets(tag, repo="lemonade-sdk/llamacpp-rocm")
        for a in assets:
            if "windows-rocm-gfx103X" in a.get("name", ""):
                url = a.get("browser_download_url") or DEFAULT_URL
                digest = a.get("digest")
                sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
                if not sha:
                    sha = "18a85d4be9052f8377ca7e7ade4bae6c0a2818b3367989a6eb3297bcb4282b5e"
                return url, sha
        return DEFAULT_URL, "18a85d4be9052f8377ca7e7ade4bae6c0a2818b3367989a6eb3297bcb4282b5e"

    plat = _platform_key()
    matchers = _ASSET_MATCH.get(plat, {}).get(backend)
    if not matchers:
        avail = ", ".join(sorted(_ASSET_MATCH.get(plat, {})))
        raise click.ClickException(
            f"backend {backend!r} is not available on this platform "
            f"({plat}). Available: {avail or 'none'}.")

    tag = _latest_tag()
    assets = _release_assets(tag)
    for a in assets:
        name = str(a.get("name", "")).lower()
        if (any(m in name for m in matchers) and "cudart" not in name
                and a.get("browser_download_url")):
            url = a["browser_download_url"]
            digest = a.get("digest")
            sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
            if not sha:
                sha = _PINNED_FALLBACK_SHA256.get(a.get("name", ""))
            return url, sha

    # Fallback: construct the canonical URL from the first matcher token.
    stem = matchers[0]
    ext = "zip" if plat == "win32" else "tar.gz"
    fname = f"llama-{tag}-{stem}.{ext}"
    guess = f"https://github.com/{_UPSTREAM_REPO}/releases/download/{tag}/{fname}"
    sha = _PINNED_FALLBACK_SHA256.get(fname)
    console.print(f"[yellow]Could not verify release asset list; using unverified URL: {guess}[/yellow]\n"
                  "[yellow]If download fails, pass --from <build dir> or --url <archive>.[/yellow]")
    return guess, sha


def _resolve_backend_url(backend: str) -> str:
    """Resolve a backend name to a downloadable archive URL.

    ``amd-rocm`` is the self-contained lemonade build (special-cased). Every
    other backend maps to an upstream llama.cpp release asset for this platform.
    Raises ``click.ClickException`` if the backend is not available here."""
    url, _sha = _resolve_backend_asset(backend)
    return url


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

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_DOWNLOAD_STALL_TIMEOUT)
    try:
        urllib.request.urlretrieve(url, dest, _hook)
    except (socket.timeout, TimeoutError) as e:
        raise ArtifactError(
            f"download stalled (no data for {_DOWNLOAD_STALL_TIMEOUT}s) - the "
            "connection was interrupted or throttled. Retry on a stable network, "
            "or provision from a local build with 'localm setup-llama --from "
            "<build-dir>' / '--url <archive-url>'."
        ) from e
    finally:
        socket.setdefaulttimeout(prev_timeout)
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


def _release_assets(tag: str, repo: str = _UPSTREAM_REPO) -> list:
    """The REAL uploaded asset list for a release tag, or [] if the API is
    unavailable or the release has none (yet).

    Deliberately does NOT fall back to scraping download links out of the
    release body: those links describe files upstream's CI intends to upload,
    not files that necessarily exist yet (see ``_latest_tag``), so trusting them
    produces a plausible-looking match that 404s instead of a caught "no
    assets" case. ``_latest_tag`` already skips a release in that state; a tag
    passed in explicitly by the caller (--url, --force, etc.) should get an
    honest empty list rather than a guess."""
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("assets", [])
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
            url, fallback_sha = _resolve_backend_asset("cuda")
            _fetch_and_place(url, target, sha256 or fallback_sha)
            return
        
        # Resolve build sha256
        build_digest = build.get("digest")
        build_sha = build_digest.split("sha256:")[-1].strip() if build_digest and "sha256:" in build_digest else None
        if not build_sha:
            build_sha = _PINNED_FALLBACK_SHA256.get(build["name"])
        
        console.print(f"[dim]CUDA build:[/dim] {build['name']} ({_human_mb(build.get('size'))})")
        _fetch_and_place(build["browser_download_url"], target, sha256 or build_sha)
        if cudart is not None:
            if sha256:
                # The pin is a single hash; it can only cover the build. Be honest
                # that the cudart bundle is validated by size + archive shape, not
                # by the pinned digest (upstream publishes no per-asset hash here).
                console.print("[yellow]Note:[/yellow] --sha256 pins the CUDA build only.")
            
            # Resolve cudart sha256
            cudart_digest = cudart.get("digest")
            cudart_sha = cudart_digest.split("sha256:")[-1].strip() if cudart_digest and "sha256:" in cudart_digest else None
            if not cudart_sha:
                cudart_sha = _PINNED_FALLBACK_SHA256.get(cudart["name"])
            
            console.print(f"[dim]CUDA runtime:[/dim] {cudart['name']} "
                          f"({_human_mb(cudart.get('size'))}) - no Toolkit install needed")
            _fetch_and_place(cudart["browser_download_url"], target, cudart_sha)
        else:
            console.print("[yellow]No cudart bundle found; CUDA Toolkit may be required.[/yellow]")
        return
    # Every other backend is a single archive resolved from the chosen name.
    url, fallback_sha = _resolve_backend_asset(chosen)
    _fetch_and_place(url, target, sha256 or fallback_sha)


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


def _flush_stdin() -> None:
    """Discard any input the OS/terminal buffered while we were NOT actually
    waiting on it (e.g. a stray Enter pressed while a driver probe or a
    multi-hundred-MB download was running). Without this, that buffered
    keystroke is silently consumed the instant the NEXT ``click.confirm()``
    prompt appears - answering a question the user never actually read, rather
    than the one they meant to answer (or none at all). Call this immediately
    before every interactive prompt in the setup flow.

    Best-effort and silent on failure: a piped/non-tty stdin (tests, CI, a
    non-interactive install) has nothing to flush and isatty() already guards
    that; any other failure just leaves stray input in place - the pre-fix
    behaviour - which is not a regression, so there is nothing worth surfacing."""
    if not sys.stdin.isatty():
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


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
        _flush_stdin()
        if click.confirm("  Continue with CUDA anyway? (No = use Vulkan)", default=False):
            return "cuda", True
        return "vulkan", False

    # Driver OK (or capability unknown but a GPU is present): offer the fetch.
    console.print("  [yellow]i[/yellow] Fetching self-contained CUDA runtime bundle. [bold]No Toolkit needed[/bold].")
    _flush_stdin()
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
        _flush_stdin()
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
            if not assume_yes and sys.stdin and sys.stdin.isatty() and click.confirm("Do you want to re-download/replace them?", default=False):
                force = True
                console.print("[yellow]Replacing existing build...[/yellow]")
            else:
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
        if not sha256:
            console.print("[yellow]Warning: Custom URL download is unverified (no --sha256 provided).[/yellow]")
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
        else:
            # Surface, do not swallow: the runtime is neither importable nor
            # installable, so a later `localm run` will fail. Say so now.
            console.print("[yellow]Warning:[/yellow] localm-llama-runtime is not "
                          "importable and could not be installed. Re-run "
                          "[bold]localm setup-llama[/bold] or check the network/log; "
                          "[bold]localm doctor[/bold] will show what is missing.")


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
    except Exception as e:
        # Surface a verify failure instead of exiting silently after "setup done":
        # a swallowed error here is exactly the "looks fine, actually broken" trap.
        console.print(f"[yellow]Warning:[/yellow] could not verify the native runtime "
                      f"({e}); it may not load. Run [bold]localm doctor[/bold] to check.")
