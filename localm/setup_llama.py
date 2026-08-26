# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Places the native inference runtime (the llama shared library + its ggml deps,
plus a matched GPU runtime when the prebuilt ships one) inside the project's own
``localm-llama-runtime`` wheel.

Backends (``--backend``):
  * ``auto`` (default) - detect the GPU and pick the fastest backend that works
    with no user-installed toolkit: NVIDIA, any OS -> ``cuda`` (self-contained
    build + runtime fetch on both Windows and Linux, see below); AMD on Windows
    (RX 6000 / unknown) -> the self-contained ROCm build; AMD elsewhere with a
    system ROCm/HIP toolkit detected present -> ``hip``; Intel and AMD with no
    toolkit detected -> ``vulkan`` (runs on NVIDIA/Intel/AMD through the normal
    display driver, no vendor toolkit); Apple Silicon -> ``metal``; no GPU ->
    ``cpu``. See ``hwdetect.recommended_install_backend`` for the full policy.
  * ``vulkan`` - universal GPU build from upstream llama.cpp (a no-toolkit
    fallback for any vendor; the default for Intel, and for AMD with no ROCm/HIP
    toolkit detected).
  * ``cuda`` - NVIDIA peak performance, self-contained on BOTH Windows and Linux:
    the matching ``cudart`` runtime bundle (Windows) or CUDA runtime libraries
    (Linux, fetched from PyPI) are fetched alongside the build, so NO CUDA
    Toolkit is needed on either OS; a driver preflight + load-test fall back to
    ``vulkan`` if the driver is too old. The CUDA asset LINE is also chosen from
    the detected GPU architecture on both platforms (Blackwell - sm_100/sm_120 -
    gets the newer 13.x line; every older architecture stays on the
    broad-compatibility 12.x line).
  * ``hip`` - AMD peak performance via an already-installed system ROCm/HIP
    toolkit (a real downloadable prebuilt binary on both Windows and Linux;
    needs that toolkit present to load - see ``_rocm_toolkit_present`` in
    hwdetect.py).
  * ``sycl`` / ``cpu`` - upstream llama.cpp prebuilts. ``sycl`` delivers peak
    Intel performance; the Windows build bundles the whole oneAPI DPC++
    runtime and is self-contained, while the Linux build does not and needs
    oneAPI installed separately (there is no Intel-GPU-presence probe for
    either, so it stays opt-in); ``cpu`` is self-contained.
  * ``amd-rocm`` - the self-contained gfx103X (RDNA2) ROCm build (bundles its
    own ROCm runtime; the current default for AMD RX 6000 on Windows, which
    needs no system toolkit).

Sources, in order of preference:
  * ``--from <dir>``  - copy from a local llama.cpp build output (any backend).
  * ``--url <url>``   - an explicit prebuilt archive URL.
  * ``--backend ...`` - resolve the matching asset of the PINNED llama.cpp
    release (``ggml-org/llama.cpp``); see below.

Which BUILD, as distinct from which backend:
  * localm installs ``_PINNED_TAG``: one upstream release confirmed to load AND
    generate, decided in this file. No version is computed while setup is
    running.
  * The installed release tag is recorded in the runtime dir's marker alongside
    the backend, so ``localm doctor`` and a bug report can name the build.
  * ``--tag <tag>`` installs one exact release and PINS it, so later runs and
    ``localm update``'s re-provision keep it. ``--tag latest`` opts IN to
    upstream's newest release, which localm has not confirmed; ``--tag default``
    returns to the shipped pin. All three live in one config key
    (``llama_runtime_pin``) read by ``_tag_for``.
  * ``--rollback`` returns to the previous build recorded for this backend.

After placing the files it installs the runtime wheel editable so the loader can
import it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from localm import config
from localm.debuglog import logger
from localm.http_ssl import RedirectDowngradeRefused, verified_urlopen

console = Console(highlight=False)

# Self-contained AMD build: lemonade-sdk llama.cpp ROCm build for gfx103X
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so no separate HIP SDK
# is needed.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1307/llama-b1307-windows-rocm-gfx103X-x64.zip"
)

# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
DEFAULT_URL_SHA256 = (
    "495323bfb522f2f5297a0786d8a2bec23f57421abdb01a1a07ff3b04d9ee7f0b"
)

# The lemonade-sdk release tag DEFAULT_URL points at. These b1xxx tags are
# lemonade-sdk's own numbering, not ggml-org's.
_ROCM_TAG = "b1307"

# Upstream llama.cpp prebuilts (ggml-org/llama.cpp).
_UPSTREAM_REPO = "ggml-org/llama.cpp"

# The upstream llama.cpp release tag localm installs, for every backend. One
# constant, decided here, never computed while a user is running setup.
#
# ONE CONSTANT, NOT ONE PER BACKEND: upstream ships ONE llama.dll for every
# backend of a given tag (the backend lives in the separate ggml-* plugin
# libraries), so the struct layout this gate cares about cannot differ between
# them.
#
# `--tag latest` opts out of this constant and tracks upstream's newest.
_PINNED_TAG = "b10375"

# What each backend's pin rests on: either a load-plus-generate check on real
# hardware, or ABI compatibility only.
#
# Every entry rests on the byte-identity above: the ABI/struct compatibility is
# carried by one shared llama library. An 'ABI only' entry carries no evidence
# that THAT backend's ggml plugin produces tokens on that hardware.
_PIN_CONFIRMATION = {
    "cpu": "load + generate, measured (Windows x64; devices: CPU only, which is "
           "also the control proving the GPU column below is not vacuous)",
    "vulkan": "load + generate, measured (Windows x64, AMD RX 6900 XT / gfx1030; "
              "the runtime registered a Vulkan0 GPU device)",
    "cuda": "ABI only (shared llama library); generation NOT measured - no NVIDIA hardware",
    "sycl": "ABI only (shared llama library); generation NOT measured - no Intel GPU",
    "hip": "ABI only (shared llama library); generation NOT measured - needs a system ROCm toolkit",
    "metal": "ABI only (shared llama library); generation NOT measured - no Apple Silicon",
    # The lemonade-sdk build, pinned separately as _ROCM_TAG. Not an upstream
    # tag, so this table's subject (_PINNED_TAG) does not describe it.
    "amd-rocm": "out of scope for _PINNED_TAG - pinned separately as _ROCM_TAG, "
                "whose generation was NOT measured by this pin's confirmation",
}

# Stored in the `llama_runtime_pin` config key to mean "track upstream's newest
# release". A sentinel rather than an empty value, which means the shipped pin.
#
# Never returned by pinned_tag(): every caller of that function interpolates
# what it returns into a release URL path segment. tracks_latest() is the second
# accessor, and _tag_for() is the only place that consults both.
_TRACK_LATEST = "latest"

# The word meaning "return to the build localm ships and confirmed", a
# different destination from `--tag latest`.
_TRACK_DEFAULT = "default"

# Third-party Linux CUDA prebuilt. Upstream publishes no bare Linux CUDA binary
# itself, so this fetches from hybridgroup/llama-cpp-builder, which tracks
# upstream's bNNNNN tag numbering 1:1 and publishes upstream's own asset-name
# convention. Not a localm-built or localm-hosted binary.
_CUDA_LINUX_REPO = "hybridgroup/llama-cpp-builder"

# Offline checksums for the assets of the pinned builds. Only consulted when the
# release API is unreachable or publishes no `digest` - the online path reads the
# digest straight off the asset listing.
#
# The table holds exactly the tags this file pins, and a test enforces that: the
# pin and its digests move together, so the pinned tag is never installed
# unverified when the API is unreachable but the download works.
#
# The values are the API's own `digest` fields.
_PINNED_FALLBACK_SHA256 = {
    # tag b10375 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names; upstream re-uploads the same file each release.
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.3-x64.zip": "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "5a40dc7c5fa3d0a80ceeba4f16f9e8d25d87bcf1399c9233588953c43436c33c",
    "llama-b10375-bin-android-arm64.tar.gz": "9c3816ee68ccddde5972395e15a61d9f0e744b92494b2f9eeae0a4f11cbd9ddc",
    "llama-b10375-bin-macos-arm64.tar.gz": "ebbeed128cde32077c5b430feafe57ce20b1bca545f430ff142472014f03bcec",
    "llama-b10375-bin-macos-x64.tar.gz": "12b4ff47c112329048e826da3fb49c674a381c2fe913311d1050f54d1f5024ab",
    "llama-b10375-bin-ubuntu-arm64.tar.gz": "36fb8a1d1836f575db78e56a875d040ddcd19694a60b67f4cce8bb6531d872ac",
    "llama-b10375-bin-ubuntu-openvino-2026.2.1-x64.tar.gz": "44e22331a613cab97ec4692749ba442b943c31ec9f7caafd7742504d8a39a7cd",
    "llama-b10375-bin-ubuntu-rocm-7.14-x64.tar.gz": "712cee42f49d4ae627f621eaa352ccfcacf51547d0afc58ef4d1873c0d9d1e25",
    "llama-b10375-bin-ubuntu-s390x.tar.gz": "600349fc3d5176421e8e2a8481e7460d10acdd56f6856cd40c2e86269857f839",
    "llama-b10375-bin-ubuntu-sycl-fp16-x64.tar.gz": "6ca1e348d2c7c2fd4810d5ceac221792bb027aa937a8dd65bb06490853a84abb",
    "llama-b10375-bin-ubuntu-sycl-fp32-x64.tar.gz": "b48221206882da9061e84b0c3e7365eb3ea23ff1bd40b5a82f9c138d06ac5796",
    "llama-b10375-bin-ubuntu-vulkan-arm64.tar.gz": "d17bcd861df0b302696eca81214a0a26db368103b5f4d6f4910396a4cf5b74d4",
    "llama-b10375-bin-ubuntu-vulkan-x64.tar.gz": "cbf7354e70f9bcda5a389e1f02e2293414d47fe525b271c3a8063327754e3ef9",
    "llama-b10375-bin-ubuntu-x64.tar.gz": "b6a7ed005240eccd61e1af42debd75b876c639c1416bfa90985fd02618919a88",
    "llama-b10375-bin-win-cpu-arm64.zip": "e57bfde78450effc75810898067934d1d482a76d9ce6e0ed181682bb9eb612e6",
    "llama-b10375-bin-win-cpu-x64.zip": "c18ad6aa9cef9d119e957472d71e34eb5183848eb9c57f51647fd18692a456c7",
    "llama-b10375-bin-win-cuda-12.4-x64.zip": "dd840b604c508b2f57f2ed467f70c711d1840c07b0d09a3bba8f6dfbd8b3da84",
    "llama-b10375-bin-win-cuda-13.3-x64.zip": "5e352df7d32abe99427160d26069e8eedab79ae08fbfe737616c6cd62837975a",
    "llama-b10375-bin-win-cuda-13.4-arm64.zip": "98dbcd67ae451cafce668285f233c8d664e5550b52e5017a167ecdd54fbe2759",
    "llama-b10375-bin-win-opencl-adreno-arm64.zip": "9c388ba3adcaae8bd1d80761deb7ac3ff7de4a55714bef0322b406cf1d421bd3",
    "llama-b10375-bin-win-openvino-2026.2.1-x64.zip": "16f8149a3792d1b56507dda3c6399c5f49dab0bfef1a69593d10e751a9d565d9",
    "llama-b10375-bin-win-rocm-7.14-x64.zip": "46464da654280440970ea8e742d3b9294291a1bcf6adb88c5160734082250334",
    "llama-b10375-bin-win-sycl-x64.zip": "a76acaedb824b32dd573c4376ed293df9f313ce6dea92010bec9fc7371168b78",
    "llama-b10375-bin-win-vulkan-x64.zip": "1fef77a8b7742485c3f9f0acd16b68330ca9d5f447b73eb80d32862e4b2c7cfa",
    "llama-b10375-ui.tar.gz": "4150e8b4b3cd24623c954d94a791f0afa80efc976555e4d6a666bce61288bcb9",
    "llama-b10375-xcframework.zip": "904bbf9fd613ff4567bd22597d5d1391c3a88c69c327fb2ff2dd722f74231c77",
    # tag b1307 ROCm assets (llama.cpp 07132750825a, ROCm 10.1.0a20260804)
    "llama-b1307-windows-rocm-gfx103X-x64.zip": "495323bfb522f2f5297a0786d8a2bec23f57421abdb01a1a07ff3b04d9ee7f0b",
    "llama-b1307-windows-rocm-gfx110X-x64.zip": "90dfa8a2ad803cf2f6a9bc069a599a6e89aa2c0a86ea46f4469b8ecf4e340978",
    "llama-b1307-windows-rocm-gfx1150-x64.zip": "fbc4ad15db7019f513760dd4ee73e39a030b5773b5efd1aacbff9973ece9865c",
    "llama-b1307-windows-rocm-gfx1151-x64.zip": "075c2cbb9c1d075295b6fa9ec6643b37629c7ef32532811f1cad6dec4aa91610",
    "llama-b1307-windows-rocm-gfx120X-x64.zip": "432c56fb566511e81a81a4558809c1773c82ada7dcc8b1223be5c1a5251be167",
    "llama-b1307-windows-rocm-gfx908-x64.zip": "43f81dbc884d1d08d929103a49a2f6ebf54916f208c40ecf4f476fc1148e2d65",
    "llama-b1307-windows-rocm-gfx90a-x64.zip": "6ff31e1124d267706d113b7de0a55a5509f971fa671f44993b0dc56c41970944",
    "llama-b1307-ubuntu-rocm-gfx103X-x64.zip": "d316029a29bab71fbb751d70034989ebabff343ed2a32bcac07c54e87fba5ea7",
    "llama-b1307-ubuntu-rocm-gfx110X-x64.zip": "9e6ca73dedcd58857918df2852d2cb6b7d6c1c9c12754e9afb5d666b6b7420c2",
    "llama-b1307-ubuntu-rocm-gfx1150-x64.zip": "b65630ae9062f0d1f2fc02012acfa743969074866b821afe9e9eb6877a2e1835",
    "llama-b1307-ubuntu-rocm-gfx1151-x64.zip": "846af2c097475e0640f2011bfb9a39852e1dfecf74fee7093d63dbe16d334b9d",
    "llama-b1307-ubuntu-rocm-gfx120X-x64.zip": "74a38230048a1081a2ebf86825c27f394de6fc5447a155ab4c8ebe81ffc3de30",
    "llama-b1307-ubuntu-rocm-gfx908-x64.zip": "9d10467e59ee05e26d21131a251077d45f41f76ceefee8de88a17222a0c8500b",
    "llama-b1307-ubuntu-rocm-gfx90a-x64.zip": "149e3d871830bf9e429c5edfa2d4d427830529a3813ad4d789925095cdd57ade",
}

# Per-backend asset matcher: substrings that must appear in the release asset
# name for (platform, backend). Substrings, not exact names, so an upstream
# version suffix (cuda-12.4, rocm-7.2) can drift without breaking the match.
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-x64"],
        "vulkan": ["bin-win-vulkan-x64"],
        # Keyed by CUDA LINE (NvidiaInfo.cuda_line), not a flat preference list:
        # a Blackwell-class GPU must never fall through to a 12.x asset, whose
        # fatbin has no kernels for it.
        "cuda": {
            "cuda-12": ["bin-win-cuda-12.4-x64", "bin-win-cuda-12"],
            "cuda-13": ["bin-win-cuda-13.3-x64", "bin-win-cuda-13"],
        },
        "sycl":   ["bin-win-sycl-x64"],
        # Matched in declared order: the newest upstream naming
        # ("bin-win-rocm-<version>-x64") first, the pre-rename
        # "bin-win-hip-radeon-x64" last so an explicit --tag on an older release
        # still resolves.
        "hip":    ["bin-win-rocm-7.14-x64", "bin-win-rocm", "bin-win-hip-radeon-x64"],
    },
    "linux": {
        "cpu":    ["bin-ubuntu-x64"],
        "vulkan": ["bin-ubuntu-vulkan-x64"],
        "cuda":   ["bin-ubuntu-cuda"],
        "sycl":   ["bin-ubuntu-sycl-fp16-x64", "bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        # Newest versioned name first, generic last, like the Windows entry.
        "hip":    ["bin-ubuntu-rocm-7.14-x64", "bin-ubuntu-rocm-7.2-x64", "bin-ubuntu-rocm"],
    },
    "darwin": {
        "cpu":    ["bin-macos-arm64", "bin-macos-x64"],
        "metal":  ["bin-macos-arm64"],
    },
}

# Backends a user may request directly (in addition to the special "auto" and
# the self-contained "amd-rocm").
_UPSTREAM_BACKENDS = ("vulkan", "cuda", "sycl", "hip", "cpu", "metal")

# Lower bound on a prebuilt llama runtime archive, which is many megabytes.
# Anything smaller is an error page, a redirect stub, or a truncated transfer.
# Always on, alongside the valid-archive structural check below; a pinned sha256
# is the opt-in third guard.
_MIN_ARTIFACT_BYTES = 256 * 1024   # 256 KiB

# Per-read socket timeout for the archive download: an idle (between-reads)
# deadline, NOT a total-transfer cap, so a large-but-progressing download is
# never killed. Only a stalled connection trips it.
_DOWNLOAD_STALL_TIMEOUT = 60   # seconds


@dataclass
class _DownloadResult:
    """What actually happened on the wire. ``content_length`` is 0 when the
    server sent none (completeness could not be checked structurally);
    ``final_url`` is the URL after following redirects."""
    bytes_received: int
    content_length: int
    content_type: str
    final_url: str


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


# Marker file recording WHICH backend currently occupies the runtime lib dir, so
# the "already provisioned" guard is backend-aware: `setup-llama --backend cuda`
# on a box holding a vulkan/cpu build still fetches CUDA. A dotfile; never
# loaded as code.
_BACKEND_MARKER = ".localm-backend"


def _record_provisioned_backend(target: Path, backend: str,
                                build: "Optional[str]" = None) -> None:
    """Record *backend* as the one now provisioned in *target*, optionally with
    the *build* tag it came from. Best-effort: a write failure is non-fatal, and
    the guard then re-provisions an explicit pick rather than skipping it. Must
    run AFTER provisioning, since _clear_target wipes the dir's files.

    Format is ``<backend>`` or ``<backend> <build>``, whitespace-separated. The
    second token is optional and is omitted when the tag is not known for free.
    A marker with no build reads back identically for the guard's purposes (see
    _provisioned_backend)."""
    line = (backend or "").strip()
    if build:
        line = f"{line} {str(build).strip()}"
    try:
        (target / _BACKEND_MARKER).write_text(line + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_marker(target: Path) -> "Optional[list]":
    """The marker's whitespace-separated tokens, or None when there is no
    readable marker. One reader for both accessors below, so the two can never
    disagree about how the file is split."""
    try:
        raw = (target / _BACKEND_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw.split() or None


def _provisioned_backend(target: Path) -> "Optional[str]":
    """The backend last provisioned into *target*, or None if unknown (no marker
    - e.g. an install predating the marker, or a hand-placed build). 'Unknown'
    is treated conservatively by the guard: an explicit pick is re-provisioned.

    Reads THE FIRST WHITESPACE TOKEN, never the whole file, so "amd-rocm" and
    "amd-rocm b1307" both answer "amd-rocm" and the provision guard's
    ``have == want`` comparison is unaffected by the optional build tag."""
    parts = _read_marker(target)
    return parts[0] if parts else None


def _provisioned_build(target: Path) -> "Optional[str]":
    """The build tag recorded alongside the backend, or None when the marker
    predates the two-token format or the tag was not knowable at provision time.

    ABSENCE IS NORMAL, never corruption: _record_provisioned_backend omits the
    tag whenever it is not free to obtain, so every reader must treat None as
    "not recorded" rather than guess a version."""
    parts = _read_marker(target)
    return parts[1] if parts and len(parts) > 1 else None


def installed_backend() -> "Optional[str]":
    """The backend actually provisioned on this box right now, or None when
    nothing is provisioned yet (a fresh install, or one that predates the
    marker).

    Public and read-only, for callers outside this module that need "what is
    installed" rather than "what would be recommended fresh". Resolves the
    target directory the same way the provisioning code does
    (_repo_runtime_lib), so it reads the marker the real install wrote."""
    return _provisioned_backend(_repo_runtime_lib())


def installed_build() -> "Optional[str]":
    """The llama.cpp release tag actually provisioned on this box right now, or
    None when nothing is provisioned or the marker predates tag recording.

    Public and read-only, the same shape as installed_backend() above.

    None is NORMAL and every caller must render it as "not recorded" rather than
    guessing a version. See _provisioned_build."""
    return _provisioned_build(_repo_runtime_lib())


# How many past provisions to remember. Rollback only needs the previous
# DISTINCT tag; the cap bounds the config key's growth.
_RUNTIME_HISTORY_MAX = 20


# The complete set of values --backend accepts, and the ONE place that decides
# it. The GUI's runtime route validates a caller-supplied backend against this
# too, so the CLI and the route accept the same names. "auto" is a real member,
# resolved through _auto_backend, and is the default for a first provision.
BACKENDS: "tuple[str, ...]" = ("auto", "vulkan", "cuda", "sycl", "hip", "cpu",
                               "metal", "amd-rocm")


# A release tag is interpolated straight into a GitHub API path and a download
# URL, so it is validated as a PATH SEGMENT: a value carrying '/', '..', '?' or
# '#' is refused. Broader than upstream's own bNNNNN shape.
_TAG_SAFE_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


# What a usable tag looks like, in one sentence, for whoever has to REFUSE one.
# Shared by the CLI's ClickException and the GUI route's 400, which must state
# the same rule.
TAG_HELP = ("Use a tag as upstream publishes it, for example 'b10355' (letters, "
            "digits, dot, dash and underscore only), or "
            f"{_TRACK_DEFAULT!r} for the build localm ships and confirmed, or "
            f"{_TRACK_LATEST!r} for upstream's newest.")


def is_safe_tag(tag: "Optional[str]") -> bool:
    """Whether *tag* is safe to interpolate into a release URL path segment.

    Public: the CLI and the GUI's runtime route both refuse a tag through this
    one predicate, so they cannot disagree about what a usable tag is.
    _validated_tag delegates here rather than carrying its own regex."""
    tag = (tag or "").strip()
    return bool(_TAG_SAFE_RE.match(tag)) and ".." not in tag


def tracks_latest() -> bool:
    """Whether this install has opted IN to upstream's newest release rather than
    the confirmed build localm ships (``setup-llama --tag latest``).

    A SEPARATE accessor from pinned_tag(), never a third return value from it:
    every caller of pinned_tag() interpolates what it gets into a release URL
    path segment, and the sentinel must never reach one.

    Never raises, same contract as pinned_tag(): an unreadable config reads as
    "not tracking", which means the shipped, confirmed pin."""
    try:
        raw = config.load_config().get("llama_runtime_pin") or ""
    except Exception:
        return False
    return str(raw).strip().lower() == _TRACK_LATEST


def pinned_tag() -> "Optional[str]":
    """The exact llama.cpp release tag the user has pinned, or None when they
    have not pinned one (the default, which installs _PINNED_TAG, and the
    ``--tag latest`` tracking mode, which is tracks_latest()'s business).

    VALIDATED ON READ, not only where --tag writes it: the key is HIDDEN with no
    coercion branch, so PATCH /v1/config stores whatever it is handed, and
    config.json is a plain file a user can edit by hand. Checking here covers
    every entry point at the one place the value is used, so a tag that would
    escape its URL path segment cannot reach _release_assets. An unsafe stored
    value is treated as NO PIN and reported, never silently obeyed or dropped.

    Never raises: an unreadable config degrades to "no pin" rather than breaking
    setup."""
    try:
        raw = config.load_config().get("llama_runtime_pin") or ""
    except Exception:
        return None
    raw = str(raw).strip()
    if not raw or raw.lower() == _TRACK_LATEST:
        return None
    if not is_safe_tag(raw):
        console.print(f"[yellow]Warning:[/yellow] ignoring the stored llama.cpp "
                      f"pin {raw!r} - it is not a usable release tag. Set one "
                      "with [bold]localm setup-llama --tag <tag>[/bold].")
        logger.warning("ignoring an unsafe llama_runtime_pin from config: %r", raw)
        return None
    return raw


def set_pinned_tag(tag: "Optional[str]") -> None:
    """Store the user's build choice: an exact tag, the _TRACK_LATEST sentinel,
    or falsy to clear it back to the shipped _PINNED_TAG. Raises on a config
    write failure, so an explicitly requested choice never silently fails to
    stick."""
    value = (tag or "").strip()
    config.update_config(lambda cfg: cfg.__setitem__("llama_runtime_pin", value))


def _record_runtime_history(backend: str, tag: "Optional[str]") -> None:
    """Append a successful provision to the rollback history. Best-effort - the
    provision itself already succeeded, so failing to journal it must not turn a
    working install into an error - but a failure is LOGGED rather than
    swallowed.

    A repeat of the newest entry is collapsed rather than appended, so re-running
    setup-llama on the same build does not push the previous distinct tag out of
    the bounded list."""
    if not tag:
        # Nothing to roll back TO: a tagless provision (--from, --url, an
        # unrecorded backend) cannot name a build, and an entry with no tag would
        # let --rollback offer a target it cannot install.
        return
    entry = {"backend": backend, "tag": tag, "at": int(time.time())}

    def _mutate(cfg: dict) -> None:
        hist = cfg.get("llama_runtime_history")
        hist = list(hist) if isinstance(hist, list) else []
        if hist and isinstance(hist[-1], dict) and \
                hist[-1].get("backend") == backend and hist[-1].get("tag") == tag:
            hist[-1] = entry
        else:
            hist.append(entry)
        cfg["llama_runtime_history"] = hist[-_RUNTIME_HISTORY_MAX:]

    try:
        config.update_config(_mutate)
    except Exception as e:
        logger.debug("could not record the runtime history entry %r: %s", entry, e)
        # Said here rather than only in the debug log, so the user can act on it
        # at the moment it happens. Not fatal: the install itself succeeded.
        console.print(f"[yellow]Warning:[/yellow] installed {backend} {tag}, but "
                      f"could not record it for rollback ({e}). "
                      "[bold]localm setup-llama --rollback[/bold] will not offer "
                      "this build later.")


def runtime_history() -> list:
    """The recorded provisions, oldest first. Filtered to well-formed entries
    whose tag is SAFE, so neither a hand-edited config nor a verbatim PATCH can
    make --rollback offer a nonsense - or hostile - target. --rollback takes a
    tag from this list and pins it without passing through _validated_tag."""
    try:
        raw = config.load_config().get("llama_runtime_history")
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [e for e in raw
            if isinstance(e, dict) and is_safe_tag(str(e.get("tag") or ""))]


def previous_tag(backend: str) -> "Optional[str]":
    """The most recent recorded tag for *backend* that is NOT the one currently
    installed - i.e. what --rollback goes back to. None when there is no such
    build to return to.

    Compared against the MARKER (what is actually on disk), not against the
    newest history entry, so a rollback still works after a history write failed
    or after the runtime dir was re-provisioned by something that did not
    journal. The marker is the ground truth for "what is installed"; history is
    only the list of candidates."""
    current = installed_build()
    for entry in reversed(runtime_history()):
        if entry.get("backend") != backend:
            continue
        tag = str(entry.get("tag")).strip()
        if tag and tag != current:
            return tag
    return None


def check_runtime_update() -> dict:
    """Compare the installed llama.cpp runtime against what ``setup-llama``
    would install right now, without provisioning anything: the read-only
    counterpart to a real re-provision, for a "check for updates" surface (the
    GUI's runtime-update card; see localm/plugins/gui/routes/runtime.py).

    The comparison target is whatever ``setup-llama`` would install right now: an
    exact PIN if one is set, else upstream's newest when the user opted into
    tracking, else the shipped ``_PINNED_TAG``. ``amd-rocm`` compares against its
    fixed ``_ROCM_TAG``, since that build is never resolved from an upstream tag.

    ONLY THE TRACKING CASE MAKES A NETWORK CALL. The default path answers from a
    constant.

    The target is a CANDIDATE, not a proof that it loads on THIS machine; that is
    only established by attempting the provision, which ``_provision_with_fallback``
    does on every install path. This function only says whether the installed
    build differs from the candidate, and never re-provisions anything.

    Returns ``{installed, backend, current, target, newer, pinned, previous}``.
    ``installed`` is False when nothing has been provisioned yet. ``previous`` is
    ``--rollback``'s own target (``previous_tag(backend)``), so one read-only
    check answers both "update available" and "is there anything to roll back
    to". Never raises: an unreadable pin/marker degrades to the "nothing to
    report" shape."""
    backend = installed_backend()
    if not backend:
        return {"installed": False, "backend": None, "current": None,
                "target": None, "newer": False, "pinned": None, "previous": None}
    current = installed_build()
    pin = pinned_tag()
    if backend == "amd-rocm":
        target = _ROCM_TAG
    elif pin:
        target = pin
    elif tracks_latest():
        target = _latest_tag()
    else:
        target = _PINNED_TAG
    newer = bool(target) and target != current
    return {"installed": True, "backend": backend, "current": current,
            "target": target, "newer": newer, "pinned": pin,
            "previous": previous_tag(backend)}


def _tag_for(backend: str) -> str:
    """The upstream llama.cpp release tag to provision for *backend*: the user's
    exact pin when one is set, else upstream's newest if they opted into tracking
    it, else the confirmed build localm ships (_PINNED_TAG).

    THE ONLY PLACE THAT DECIDES A TAG for the upstream-resolved backends. It is
    NOT consulted for amd-rocm, whose build comes from lemonade-sdk's own release
    numbering (_ROCM_TAG), a different tag space in which an upstream bNNNNN
    means nothing; _pin_note_for_backend says so out loud.

    The default branch makes NO network call."""
    pin = pinned_tag()
    if pin:
        return pin
    if tracks_latest():
        return _latest_tag()
    return _PINNED_TAG


def _pin_note_for_backend(backend: str) -> None:
    """Say plainly when a pin the user set does not apply to the backend being
    provisioned, instead of dropping it silently. Only amd-rocm is in that
    position: its tag is lemonade-sdk's, not upstream's."""
    if backend != "amd-rocm":
        return
    pin = pinned_tag()
    if pin:
        console.print(
            f"[yellow]Note:[/yellow] the pinned llama.cpp build {pin} does not "
            f"apply to the amd-rocm backend - it ships from lemonade-sdk's own "
            f"release numbering ({_ROCM_TAG}), a different tag series. The pin "
            "stays set and applies to every other backend.")
    elif tracks_latest():
        # '--tag latest' is equally inapplicable to this backend, and silence
        # would read as this install tracking upstream.
        console.print(
            "[yellow]Note:[/yellow] '--tag latest' does not apply to the "
            f"amd-rocm backend - it ships from lemonade-sdk's own release "
            f"numbering ({_ROCM_TAG}), a different tag series, fixed by the "
            "localm release you are running. The setting stays and applies to "
            "every other backend.")


def _is_wanted(f: Path) -> bool:
    """Whether to copy *f*: the loadable library, its ggml deps, and the runtime
    libraries - matched by platform-appropriate naming (incl. versioned .so.N).

    LIBRARIES ONLY, NEVER EXECUTABLES. localm loads the native runtime in-process
    through ctypes and never shells out to a bundled binary, so the upstream
    archives' command-line tools (llama-cli, llama-server, llama-bench,
    ggml-rpc-server, ...) are not copied. The darwin and Linux branches below
    match libraries only (.dylib / .so) and so have never copied an executable.

    Libraries are kept WHOLESALE: a .dll may be an OS-resolved link dependency of
    ggml-hip/llama rather than something localm opens by name (amd_comgr,
    rocblas, hipblaslt, rocsolver, origami, rocm_kpack all are).
    """
    n = f.name.lower()
    if sys.platform == "win32":
        return n.endswith(".dll")
    if sys.platform == "darwin":
        return n.endswith(".dylib")
    return ".so" in n          # libfoo.so and libfoo.so.1


# rocBLAS and hipBLASLt (ROCm's vendor BLAS libraries) resolve their GPU-arch-
# specific GEMM kernels ("Tensile" library) at RUNTIME from a "<name>/library/"
# data directory sitting next to their DLL; the kernels are NOT linked into the
# DLL itself. That data is .dat/.hsaco/.co files, which _is_wanted() does not
# match and _copy_binaries' flat `target / f.name` copy would strip of its
# required subdirectory layout. These names are listed so the directories are
# copied whole.
#
# Without that data rocBLAS fails to init its Tensile host and hard-crashes the
# native process outright, uncatchable from Python, on the first workload that
# dispatches a GEMM through Tensile (the embedder's non-causal batch encode).
#
# Both names stay listed for the gfx110X/gfx120X archives; a missing directory
# is a no-op here.
_BLAS_LIBRARY_DIRS = ("rocblas", "hipblaslt")

# Of those, the ones whose kernel data is genuinely REQUIRED by an install that
# ships the matching vendor library. Only rocblas: without its Tensile data it
# hard-crashes the native process on the first GEMM dispatched through it.
# hipblaslt ships as a library with no kernel directory at all on the gfx103X
# archive, and that install is healthy.
_BLAS_DIRS_REQUIRING_KERNELS = ("rocblas",)


def _has_vendor_library(target: Path, name: str) -> bool:
    """True when *target* holds the shared library for BLAS vendor *name*.

    Matches both naming conventions, which the archives use on the same platform
    (`rocblas.dll` and `libhipblaslt.dll`), and covers `.so` version suffixes
    (librocblas.so.4) the same way _is_wanted does."""
    for f in target.iterdir():
        if not f.is_file():
            continue
        stem = f.name.lower()
        if stem.startswith("lib"):
            stem = stem[3:]
        if stem.startswith(name + "."):
            return True
    return False


def blas_kernel_problems(target: Path) -> "list[str]":
    """Human-readable problems with the BLAS kernel data in a provisioned runtime.

    Empty list means nothing to report, INCLUDING for every non-ROCm backend: the
    check is keyed on whether the install actually ships the vendor library, so a
    vulkan / cuda / cpu / metal install has nothing to match and is silently fine.
    No platform test and no backend marker is consulted, since the marker is
    written last during a provision and a half-finished install can be missing it.

    Catches "the library is installed but its runtime kernel data is not", the
    silent failure: provisioning succeeds, chat works, and the crash arrives on
    the first Tensile GEMM. Does not catch a missing library, which already fails
    loudly at load."""
    problems: "list[str]" = []
    try:
        if not target.is_dir():
            return problems
        for name in _BLAS_DIRS_REQUIRING_KERNELS:
            if not _has_vendor_library(target, name):
                continue
            d = target / name
            if not d.is_dir():
                problems.append(f"{name} is installed but its {name}/ kernel "
                                f"directory is missing entirely")
                continue
            n = sum(1 for p in d.rglob("*") if p.is_file())
            if n == 0:
                problems.append(f"{name} is installed but its {name}/ kernel "
                                f"directory is empty")
    except OSError as e:
        # Cannot read the install: say so rather than returning "no problems".
        problems.append(f"could not inspect BLAS kernel data: {e}")
    return problems


def _copy_blas_library_dirs(src_dir: Path, target: Path) -> int:
    """Copy any of ``_BLAS_LIBRARY_DIRS`` found under *src_dir* into *target*,
    preserving their internal directory structure (unlike _copy_binaries' flat
    DLL copy - rocBLAS/hipBLASLt resolve this data by RELATIVE PATH, not by file
    name). Searches one level of nesting too, in case an archive wraps its
    contents in a single top-level folder. Returns the number of files copied."""
    n = 0
    for name in _BLAS_LIBRARY_DIRS:
        src = src_dir / name
        if not src.is_dir():
            nested = list(src_dir.glob(f"*/{name}"))
            src = nested[0] if nested else None
        if not src or not src.is_dir():
            continue
        dest_root = target / name
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            out = dest_root / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    return n


def _repo_runtime_lib() -> Path:
    """The localm-llama-runtime wheel's lib/ dir."""
    try:
        import localm_llama_runtime
        return Path(localm_llama_runtime.LIB_DIR)
    except Exception as e:
        # The wheel is legitimately ABSENT before `setup-llama` installs it, so
        # the repo-relative fallback is correct then and must not hard-fail. A
        # BROKEN install (an import error other than not-found) is surfaced at
        # debug level.
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
    """Pick the broadest WORKING backend for this machine, via the SAME policy the
    installers use (``hwdetect.recommended_install_backend``):

      NVIDIA, any OS -> cuda (self-contained build + runtime fetch on both
      Windows and Linux, peak performance); AMD on Windows (RX 6000 / unknown)
      -> the self-contained ROCm build; AMD elsewhere with a system ROCm/HIP
      toolkit detected -> hip; Apple Silicon -> metal; every other GPU (Intel,
      AMD with no toolkit detected) -> vulkan; no GPU -> cpu."""
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception as e:
        # Surface the skipped GPU setup, and how to force a GPU backend, rather
        # than defaulting silently to CPU.
        console.print(f"[yellow]GPU detection failed ({e}); defaulting to CPU - "
                      "override with --backend.[/yellow]")
        return "cpu"
    return hwdetect.recommended_install_backend(det)


def _latest_tag() -> str:
    """The newest ggml-org/llama.cpp release tag that actually has its build
    assets uploaded, or _PINNED_TAG if no such release can be found (offline,
    rate-limited, etc.).

    ONLY REACHED WHEN THE USER OPTED IN with ``--tag latest``; see _tag_for. What
    it returns is a build nobody here has run.

    Upstream publishes a release (tag + notes) as soon as it is cut, and its CI
    matrix uploads the platform archives afterwards, so right after publish
    ``/releases/latest`` can point at a tag whose ``assets`` array is empty even
    though the release body already lists the download URLs, and those links
    404. Recent releases are therefore scanned newest-first and the first one
    that already has assets is used."""
    tags = _recent_tags()
    if tags:
        return tags[0]
    # Name what was installed instead of upstream's newest, and why.
    console.print(f"[yellow]Could not find a ggml-org/llama.cpp release with "
                  f"uploaded assets (the release lookup was unreachable, or the "
                  f"newest releases have not finished uploading). Installing "
                  f"localm's confirmed build {_PINNED_TAG} instead - rerun later "
                  "for upstream's newest.[/yellow]")
    return _PINNED_TAG


def _recent_tags(limit: int = 10) -> list:
    """Upstream release tags that already have their build assets uploaded,
    NEWEST FIRST. Empty when the lookup is unavailable.

    One list, one call, one skip rule, shared with _latest_tag and the tag
    walk-back, so "which releases are candidates" has a single answer."""
    api = f"https://api.github.com/repos/{_UPSTREAM_REPO}/releases?per_page={int(limit)}"
    out: list = []
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with verified_urlopen(req, timeout=10) as r:
            releases = json.loads(r.read().decode("utf-8"))
        for rel in releases:
            if rel.get("draft") or rel.get("prerelease"):
                continue
            tag = rel.get("tag_name")
            # A release is published before its CI uploads the archives, so a tag
            # with an empty assets array 404s on download.
            if isinstance(tag, str) and tag and rel.get("assets"):
                out.append(tag)
    except Exception as e:
        # Best-effort: every caller has a pinned fallback, so this must not
        # raise, but an unavailable lookup stays discoverable in the debug log.
        logger.debug("release tag listing failed for %s (%s)", api, e)
        return []
    return out


def _resolve_backend_asset(backend: str, cuda_line: Optional[str] = None,
                           tag: Optional[str] = None
                           ) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve a backend name to a (url, sha256_digest, tag) triple.

    If the release listing is available, resolves it dynamically and gets the
    sha256 from the digest field. If offline, falls back to the templated guess
    and queries the local pinned checksum dictionary.

    *cuda_line* selects which asset-name substrings to match for the 'cuda'
    backend on Windows AND Linux (see NvidiaInfo.cuda_line) - ignored for
    every other backend/platform, which have a single, non-line-specific
    matcher list. Defaults to _CUDA_LINE, resolved below rather than bound as a
    literal default, since _CUDA_LINE is defined later in this module.

    *tag* lets a caller that has ALREADY resolved one (the Windows CUDA branch,
    which needs it to pair the build with its cudart bundle) pass it in rather
    than resolve a second time. When omitted this resolves its own through
    _tag_for - pin, else upstream's newest.

    The third element is the release tag this resolution used, which the caller
    records in the marker. It is None for amd-rocm, whose build comes from
    lemonade-sdk's own release numbering (_ROCM_TAG) rather than an upstream tag;
    the caller supplies that constant itself."""
    cuda_line = cuda_line or _CUDA_LINE
    if backend == "amd-rocm":
        if sys.platform != "win32":
            raise click.ClickException(
                "the self-contained 'amd-rocm' build is Windows-only; on Linux "
                "use --backend hip (needs ROCm) or build with --from.")
        # Try to resolve dynamically first
        tag = _ROCM_TAG
        assets = _release_assets(tag, repo="lemonade-sdk/llamacpp-rocm")
        for a in assets:
            if "windows-rocm-gfx103X" in a.get("name", ""):
                url = a.get("browser_download_url") or DEFAULT_URL
                digest = a.get("digest")
                sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
                if not sha:
                    sha = DEFAULT_URL_SHA256
                return url, sha, None
        # Surface the fallback: the lemonade-sdk release lookup was unreachable,
        # or this release is missing the expected gfx103X asset, so the build may
        # not be current.
        console.print("[yellow]Could not find a lemonade-sdk/llamacpp-rocm release asset "
                      f"for {tag}; using pinned amd-rocm build - rerun later for the "
                      "latest.[/yellow]")
        return DEFAULT_URL, DEFAULT_URL_SHA256, None

    if backend == "cuda" and _platform_key() == "linux":
        # Upstream (ggml-org/llama.cpp) publishes no bare Linux CUDA binary at
        # all, so the generic _ASSET_MATCH path below would only construct a
        # guessed URL that 404s. Resolves against hybridgroup/llama-cpp-builder,
        # which tracks upstream's tag numbering 1:1 and publishes upstream's own
        # asset-name convention, so the same tag applies here.
        #
        # cuda_line-aware, like the win32 cuda branch below: hybridgroup
        # publishes both a cuda-12 asset ("...-cuda-x64.tar.gz") and a cuda-13
        # one ("...-cuda-13-x64.tar.gz").
        suffix = "-cuda-13-x64.tar.gz" if cuda_line == "cuda-13" else "-cuda-x64.tar.gz"
        tag = tag or _tag_for(backend)
        assets = _release_assets(tag, repo=_CUDA_LINUX_REPO)
        for a in assets:
            name = str(a.get("name", "")).lower()
            if name.endswith(suffix) and a.get("browser_download_url"):
                url = a["browser_download_url"]
                digest = a.get("digest")
                sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
                return url, sha, tag
        # Genuinely unresolvable (hybridgroup has not built that exact upstream
        # tag yet): raise click.ClickException, which the caller catches and
        # turns into the same offer/force-vulkan-fallback path every other
        # provisioning failure uses. No guessed URL is constructed here.
        raise click.ClickException(
            f"no Linux CUDA build found for llama.cpp tag {tag!r} on "
            f"{_CUDA_LINUX_REPO} (dev-notes/ADR-0010) - falling back to vulkan.")

    plat = _platform_key()
    entry = _ASSET_MATCH.get(plat, {}).get(backend)
    # The 'cuda' entry on win32 is keyed by cuda_line (a dict), not a flat list,
    # since the right asset depends on the GPU's architecture.
    matchers = entry.get(cuda_line) if isinstance(entry, dict) else entry
    if not matchers:
        avail = ", ".join(sorted(_ASSET_MATCH.get(plat, {})))
        raise click.ClickException(
            f"backend {backend!r} is not available on this platform "
            f"({plat}). Available: {avail or 'none'}.")

    tag = tag or _tag_for(backend)
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
            return url, sha, tag

    # Fallback: the release listing was unavailable, so the asset name comes
    # from _PINNED_FALLBACK_SHA256, which for a tag this file pins IS that
    # release's asset list, so the exact filename needs no guessing. Matchers
    # are tried in their declared order, which encodes a preference the names
    # alone do not: linux sycl lists the fp16 build first and a bare
    # "bin-ubuntu-sycl" would also match fp32.
    #
    # The template below builds `llama-<tag>-<matcher>`, which stops matching
    # whenever upstream renames an asset and cannot express an extension other
    # than tar.gz off win32.
    fname = ""
    for m in matchers:
        hits = sorted(n for n in _PINNED_FALLBACK_SHA256
                      if n.startswith(f"llama-{tag}-") and m in n.lower()
                      and "cudart" not in n)
        if hits:
            fname = hits[0]
            break
    if not fname:
        # An unpinned tag (--tag <something>): nothing to read, so a constructed
        # name is the only option, and it is right while upstream's naming holds.
        ext = "zip" if plat == "win32" else "tar.gz"
        fname = f"llama-{tag}-{matchers[0]}.{ext}"
    guess = f"https://github.com/{_UPSTREAM_REPO}/releases/download/{tag}/{fname}"
    sha = _PINNED_FALLBACK_SHA256.get(fname)
    console.print(f"[yellow]Could not verify release asset list; using unverified URL: {guess}[/yellow]\n"
                  "[yellow]If download fails, pass --from <build dir> or --url <archive>.[/yellow]")
    return guess, sha, tag


def _resolve_backend_url(backend: str, cuda_line: Optional[str] = None) -> str:
    """Resolve a backend name to a downloadable archive URL.

    ``amd-rocm`` is the self-contained lemonade build (special-cased). Every
    other backend maps to an upstream llama.cpp release asset for this platform.
    *cuda_line* is passed straight through to _resolve_backend_asset (see its
    docstring). No production code calls this function; main() resolves via
    _provision_backend -> _resolve_backend_asset directly.
    Raises ``click.ClickException`` if the backend is not available here."""
    url, _sha, _tag = _resolve_backend_asset(backend, cuda_line)
    return url


# --------------------------------------------------------------------------- #
#  Download / validate / extract                                              #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> _DownloadResult:
    """Stream *url* to *dest*, capturing what actually happened on the wire, not
    just whether it succeeded. Distinguishes three failure shapes, each reported
    with its own specific cause:

    * a STALL (no bytes for ``_DOWNLOAD_STALL_TIMEOUT``s) - the connection is
      alive but frozen;
    * a transport-level drop mid-transfer (connection reset, broken pipe, ...) -
      the connection died outright, with however many bytes had arrived so far;
    * a CLEAN completion that is nonetheless short of what was promised - the
      server (or something between it and us) considers the response finished,
      it is just not the archive. This third case is NOT an error here - it is
      returned normally and diagnosed by the caller once the file is on disk,
      because "too short" alone does not yet say WHY (see
      :func:`_diagnose_bad_artifact`)."""
    console.print(f"[dim]Downloading {url}[/dim]")
    last = [-1]

    def _report(nread: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, nread * 100 // total)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            mb = total / 1024 ** 2
            console.print(f"[dim]  {pct:3d}%  ({mb:.0f} MB)[/dim]", end="\r")

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_DOWNLOAD_STALL_TIMEOUT)
    total = 0
    nread = 0
    try:
        # verified_urlopen (see localm/http_ssl.py) follows the GitHub ->
        # release-CDN 302 and verifies both hops. Its default HttpsOnlyRedirect
        # refuses a redirect off https and raises RedirectDowngradeRefused,
        # handled below. _validate_archive's digest check is opt-in (its
        # expected_sha256, i.e. --sha256), so an unpinned archive has no
        # cryptographic check on its content. Streams in chunks so a
        # multi-hundred-MB archive is never held in memory; the default socket
        # timeout is the between-reads stall deadline, not a total cap.
        req = urllib.request.Request(url, headers={"User-Agent": "localm-setup-llama"})
        with verified_urlopen(req, timeout=_DOWNLOAD_STALL_TIMEOUT) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            content_type = r.headers.get("Content-Type") or ""
            # geturl() is standard on every real urllib response; guarded for a
            # test double that does not implement it. The final URL is a
            # diagnostic, not load-bearing.
            try:
                final_url = r.geturl() or url
            except Exception:
                final_url = url
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                nread += len(chunk)
                _report(nread, total)
    except RedirectDowngradeRefused as e:
        # BEFORE the OSError clause below, which RedirectDowngradeRefused would
        # otherwise hit (it is a URLError, and URLError is an OSError). That
        # clause's "dropped or flaky connection, retry" advice is wrong for a
        # refused downgrade off https.
        raise ArtifactError(
            f"refused to follow this download off HTTPS ({e}) - the archive "
            "would have arrived in cleartext, where anything on the network "
            "path can replace it, and its bytes are loaded as a native "
            "library. This is not a transient network fault: check the URL, "
            "or provision from a local build with 'localm setup-llama --from "
            "<build-dir>'."
        ) from e
    except (socket.timeout, TimeoutError) as e:
        raise ArtifactError(
            f"download stalled (no data for {_DOWNLOAD_STALL_TIMEOUT}s, after "
            f"{nread} of {total or 'an unknown number of'} bytes) - the "
            "connection was interrupted or throttled. Retry on a stable network, "
            "or provision from a local build with 'localm setup-llama --from "
            "<build-dir>' / '--url <archive-url>'."
        ) from e
    except OSError as e:
        # A live transport failure mid-stream (connection reset, broken pipe, a
        # proxy dropping the connection outright). A download that completes
        # normally but turns out short is not an exception at all; see the
        # docstring. Report the partial state instead of a generic failure.
        raise ArtifactError(
            f"the connection was interrupted after {nread} of "
            f"{total or 'an unknown number of'} bytes ({e}) - this looks like a "
            "dropped or flaky connection, not a blocked download. Retry, or "
            "provision from a local build with 'localm setup-llama --from "
            "<build-dir>' / '--url <archive-url>'."
        ) from e
    finally:
        socket.setdefaulttimeout(prev_timeout)
    console.print()
    return _DownloadResult(bytes_received=nread, content_length=total,
                           content_type=content_type, final_url=final_url)


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


def _sniff_content_kind(path: Path, peek: int = 4096) -> str:
    """Classify what the file's own bytes actually look like, independent of what
    it was supposed to be, which is what tells a substituted HTML/JSON response
    apart from a genuinely truncated archive. A real llama.cpp archive's opening
    bytes never decode as text.

    Returns one of: 'empty', 'zip_truncated', 'gzip_truncated', 'html', 'xml',
    'json', 'text', 'binary'. The two '..._truncated' results specifically mean
    the file STARTS with a real archive's magic bytes but is not (yet, or ever
    going to be) a complete one - a different cause than a substituted page."""
    try:
        with open(path, "rb") as f:
            head = f.read(peek)
    except OSError:
        return "binary"
    if not head:
        return "empty"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip_truncated"
    if head[:2] == b"\x1f\x8b":
        return "gzip_truncated"
    # Structural markers are pure ASCII and sit at or near the very start of a
    # real error/block page regardless of the page's OVERALL encoding, so they
    # are matched with a lossy decode first (never raises, so it still finds
    # HTML/JSON/XML served as e.g. windows-1252). A strict decode is only needed
    # for the weaker 'text vs binary' distinction below.
    lossy = head.decode("ascii", errors="replace").lstrip()
    lower = lossy[:200].lower()
    if lower.startswith("<!doctype html") or lower.startswith("<html") or "<html" in lower:
        return "html"
    if lower.startswith("<?xml") or "<error>" in lower:
        return "xml"
    if lossy[:1] in ("{", "["):
        return "json"
    try:
        head.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def _diagnose_bad_artifact(path: Path, dl: Optional["_DownloadResult"]) -> str:
    """Turn what the bytes on disk actually look like - plus, when available,
    what the response claimed (:func:`_download`'s result for this same file) -
    into ONE specific, evidence-backed explanation. Never states a cause the
    evidence does not support: the fallback case says 'not clear' plainly
    instead of picking the most likely-sounding story."""
    kind = _sniff_content_kind(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if kind == "empty":
        cause = "nothing was received at all"
    elif kind == "html":
        cause = ("the response is an HTML page, not the archive - almost always "
                 "a network that blocks or filters this download (a corporate "
                 "proxy or security product), not a problem with the release itself")
    elif kind in ("json", "xml"):
        cause = (f"the response is {kind.upper()}, not the archive - most likely "
                 "an error response from a proxy or the CDN standing in for the "
                 "real file (again typically a corporate network filter)")
    elif kind in ("zip_truncated", "gzip_truncated"):
        cause = ("the response starts like the real archive but cuts off "
                 "partway through - this looks like a genuinely interrupted "
                 "transfer (a dropped or throttled connection), not a deliberate "
                 "block")
    else:
        cause = ("the content does not clearly indicate the cause - it is "
                 "neither a recognisable webpage nor a valid archive")

    detail_bits = [f"{size} bytes received"]
    if dl is not None:
        detail_bits.append(f"{dl.content_length} expected (Content-Length)"
                           if dl.content_length else "no Content-Length given")
        if dl.content_type:
            detail_bits.append(f"Content-Type: {dl.content_type}")
        if dl.final_url:
            detail_bits.append(f"final URL: {dl.final_url}")
    return f"{cause} ({'; '.join(detail_bits)})."


def _validate_archive(
    path: Path,
    expected_sha256: Optional[str] = None,
    min_size: int = _MIN_ARTIFACT_BYTES,
    dl: Optional[_DownloadResult] = None,
) -> None:
    """Validate a downloaded artifact BEFORE it is extracted or installed.
    Raises :class:`ArtifactError` on any failure.

    Three checks, in cheapest-first order:
      1. size: a real prebuilt runtime archive is many MB; a tiny/empty body is
         an error page, a redirect stub, or a truncated transfer (always on).
      2. shape: it must be a structurally valid zip OR tar archive, so a
         200-with-HTML or a half-transferred file is rejected before it reaches
         extraction (always on).
      3. provenance: when *expected_sha256* is given, the file's digest must
         match it (opt-in; refuses on mismatch). Comparison is whitespace- and
         case-insensitive so a pasted hash from any source works.

    *dl*, when given (the :func:`_download` result for this same file), lets
    checks 1 and 2 explain WHY from real evidence - what the bytes actually
    look like, plus what the response claimed - instead of a generic hedge (see
    :func:`_diagnose_bad_artifact`).
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ArtifactError(f"could not stat downloaded file: {e}") from e
    if size < min_size:
        raise ArtifactError(
            f"download is too small ({size} bytes < {min_size} minimum): "
            f"{_diagnose_bad_artifact(path, dl)}"
        )
    if not _is_supported_archive(path):
        raise ArtifactError(
            f"download is not a valid zip or tar archive: "
            f"{_diagnose_bad_artifact(path, dl)}"
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
    Python 3.12+ ``filter="data"`` is used directly (see _extract_archive)."""
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
# ships no LICENSE file. MIT requires the license text to accompany the
# redistributed llama.cpp/ggml binaries.
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
    # rocBLAS/hipBLASLt Tensile kernel data (see _BLAS_LIBRARY_DIRS) - a no-op
    # on every non-ROCm backend, whose src_dir has no rocblas/hipblaslt dir.
    n += _copy_blas_library_dirs(src_dir, target)
    # MIT requires the license to accompany the binaries: capture it (or a
    # bundled fallback) alongside them whenever binaries were actually placed.
    if n:
        _copy_license_files(src_dir, target)
    return n


def _install_runtime_wheel(pkg_dir: Path) -> bool:
    """Install the runtime wheel editable into the active venv. Tries uv, then
    pip. Returns True on success.

    ``env`` pins uv's AND pip's caches inside the data dir, same as the
    plugin-extra installer (plugins/deps.py): build isolation pulls the build
    backend (setuptools/wheel) into the tool's cache, which either tool would
    otherwise put in a per-user location OUTSIDE the data dir. See
    ``config.contained_pip_env``."""
    env = config.contained_pip_env()
    last_err = ""
    for cmd in (["uv", "pip", "install", "-e", str(pkg_dir)],
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode == 0:
                return True
            # Keep the real pip/uv failure so the user can see the actual cause
            # (missing build tools, conflicting deps).
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
#  The CUDA llama build needs the CUDA *runtime* libraries (cudart /           #
#  cublas) at load time. Upstream ships a self-contained                       #
#  ``cudart-llama-bin-win-cuda-<ver>`` bundle in the SAME release, so CUDA     #
#  works WITHOUT the user installing the full CUDA Toolkit. The GPU DRIVER     #
#  cannot be self-assembled (a system component needing admin + a reboot);     #
#  a too-old driver is the single you-must-do-this-part branch.                #
# --------------------------------------------------------------------------- #

# Which upstream CUDA asset line to fetch, as a function of the GPU's
# ARCHITECTURE rather than the platform: upstream ships both a 12.x line (broad
# compatibility, runs on any driver new enough for CUDA 12.4) and a 13.x line
# (needed for Blackwell-class GPUs, itself needing a newer driver and dropping
# some pre-Turing arch support). The pinned 12.4 build's fatbin has no Blackwell
# kernels, so a Blackwell card gets the 13.x line and every older architecture
# stays on 12.x. _CUDA_LINE is the fallback when no architecture information is
# available at all (see NvidiaInfo.cuda_line).
_CUDA_LINE = "cuda-12"

# Compute-capability floor for "needs the 13.x line" (nvidia-smi's
# ``compute_cap`` query, e.g. "8.9", "12.0" - the GPU's sm/arch level, NOT the
# driver's max CUDA version). Blackwell datacenter parts report 10.0 (sm_100)
# and Blackwell consumer/workstation parts report 12.0 (sm_120), so >= 10.0
# catches both variants and any later architecture.
_BLACKWELL_MIN_CAP = (10, 0)

# Minimum driver-reported CUDA version ("cuda_capability") to trust each line's
# build, keyed by the line. Both match the PINNED asset's own X.Y, not just its
# major version, so a borderline driver is routed to the Vulkan fallback rather
# than handed a build that fails to load.
_MIN_DRIVER_CUDA = {
    "cuda-12": (12, 4),
    "cuda-13": (13, 3),
}


def _ver_tuple(v: str) -> Optional[tuple]:
    # None (not (0,0)) on an unparseable version, so an unreadable capability
    # reads as "unknown" rather than as a too-old driver that is falsely blocked.
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return None


def _ver_at_least(parsed: tuple, minimum: tuple) -> bool:
    """*parsed* >= *minimum*, treating a bare-major version (no minor component,
    e.g. "10" -> (10,)) as ".0". Plain tuple comparison gets this wrong: Python
    considers a tuple that is a strict PREFIX of another to be the smaller one
    regardless of the missing component's value, so (10,) >= (10, 0) is False
    even though 10 == 10. _ver_tuple's own contract - a bare major parses to a
    1-element tuple, not padded - is unchanged; the padding belongs here, at the
    comparison."""
    padded = parsed + (0,) * (len(minimum) - len(parsed))
    return padded >= minimum


@dataclass
class NvidiaInfo:
    """What nvidia-smi told us. Advisory only; every field may be empty."""
    present: bool = False           # an NVIDIA GPU + usable driver was found
    gpu_name: str = ""
    driver_version: str = ""
    cuda_capability: str = ""       # max CUDA the driver supports, e.g. "12.4"
    compute_capability: str = ""    # the GPU's own sm/arch level, e.g. "12.0" (Blackwell/sm_120)

    @property
    def cuda_line(self) -> str:
        """Which upstream CUDA asset line this GPU's ARCHITECTURE needs:
        'cuda-12' (broad-compatibility default) or 'cuda-13' (required for
        Blackwell and newer - see _BLACKWELL_MIN_CAP). Unknown or unparseable
        capability stays on cuda-12, since an unread architecture is not
        evidence that the newer, narrower-compatibility line is needed."""
        cap = _ver_tuple(self.compute_capability)
        if cap is not None and _ver_at_least(cap, _BLACKWELL_MIN_CAP):
            return "cuda-13"
        return "cuda-12"

    @property
    def driver_ok(self) -> bool:
        """True when the driver is new enough for the CUDA line THIS GPU's
        architecture needs (see cuda_line) - the minimum is not a single fixed
        threshold, since Blackwell and older cards need different lines.
        Unknown driver capability is treated as OK."""
        if not self.cuda_capability:
            return True
        parsed = _ver_tuple(self.cuda_capability)
        # An unparseable capability is unknown, not old: cannot judge, do not block.
        if parsed is None:
            return True
        return _ver_at_least(parsed, _MIN_DRIVER_CUDA[self.cuda_line])


def _nvidia_smi(*args: str) -> str:
    """Combined nvidia-smi output, or "" if it is not present/usable."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=8)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def nvidia_preflight() -> NvidiaInfo:
    """Detect the NVIDIA GPU + driver, the max CUDA version the DRIVER supports,
    and the GPU's own compute capability (its architecture - e.g. "12.0" for
    Blackwell/sm_120). These are two different questions: the driver's version
    says what it CAN run; the compute capability says what the CARD IS, which
    decides whether the 12.x build's fatbin has kernels for it (see
    NvidiaInfo.cuda_line).

    Never raises. Parses the nvidia-smi banner ("Driver Version: X  CUDA
    Version: Y") and asks explicitly for the (untruncated) GPU name and its
    compute capability."""
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
    cap = _nvidia_smi("--query-gpu=compute_cap", "--format=csv,noheader").strip().splitlines()
    if cap:
        info.compute_capability = cap[0].strip()
    return info


def _release_assets(tag: str, repo: str = _UPSTREAM_REPO) -> list:
    """The REAL uploaded asset list for a release tag, or [] if the API is
    unavailable or the release has none (yet).

    Does NOT fall back to scraping download links out of the release body: those
    links describe files upstream's CI intends to upload, not files that
    necessarily exist yet (see ``_latest_tag``), so trusting them produces a
    plausible-looking match that 404s instead of a caught "no assets" case. A
    tag passed in explicitly by the caller (--url, --force, etc.) gets an honest
    empty list rather than a guess."""
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with verified_urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("assets", [])
    except Exception as e:
        # Best-effort probe: every caller has a pinned fallback for this case
        # (offline, rate-limited, API down), so this must not raise.
        logger.debug("release asset lookup failed for %s (%s)", api, e)
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


def _resolve_cuda_pair(tag: str, line: str = _CUDA_LINE) -> tuple:
    """(build_asset, cudart_asset) for the Windows CUDA *line* ('cuda-12' or
    'cuda-13' - see NvidiaInfo.cuda_line). Either may be None when the release
    listing is unavailable or lacks it.

    The build and the cudart runtime share the "...bin-win-cuda-X.Y..." name
    fragment (the runtime is e.g. cudart-llama-bin-win-cuda-12.4-x64.zip), and
    the runtime is often listed FIRST, so the build matcher MUST exclude
    "cudart" - otherwise build resolves to the runtime-only zip (CUDA DLLs, no
    llama.dll) and provisioning aborts with "the archive did not contain
    llama.dll"."""
    assets = _release_assets(tag)
    build = _pick_asset(assets, "bin-win-" + line, exclude=("cudart",))
    cudart = _pick_asset(assets, "cudart", "win-" + line)
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
        dl = _download(url, arc)
        _validate_archive(arc, expected_sha256=sha256, dl=dl)   # archive gate, pre-extract
        ex = Path(tmp) / "x"
        _extract_archive(arc, ex)
        return _copy_binaries(ex, target)


# Tracked git sentinel files living in the runtime lib dir (see
# runtime/localm_llama_runtime/lib/.gitignore, which keeps the downloaded native
# binaries out of version control). Unlike _BACKEND_MARKER, these are never
# regenerated by setup-llama, and deleting them empties the .gitignore, so a
# later `git add -A` would stage the downloaded DLLs into git.
_PRESERVED_TARGET_FILES = (".gitignore", ".gitkeep")


class RuntimeInUseError(Exception):
    """Something has the installed runtime open, so it cannot be replaced.

    NOT an ArtifactError and not a load failure. Those two mean a build is bad;
    this one means both builds are fine and a process is merely holding the
    files. A build that will not load earns the Vulkan fallback; this earns
    "close it and retry" with the existing install left completely intact."""

    def __init__(self, locked: "list[Path]", partial: bool = False):
        self.locked = list(locked)
        # True only when files were already deleted before the lock was hit (the
        # probe-to-unlink race). The install is then half cleared, so the two
        # cases are tracked apart and reported apart.
        self.partial = partial
        shown = ", ".join(sorted(p.name for p in self.locked[:6]))
        more = f" (+{len(self.locked) - 6} more)" if len(self.locked) > 6 else ""
        super().__init__(f"{len(self.locked)} runtime file(s) in use: {shown}{more}")


def _clearable_files(target: Path) -> "list[Path]":
    """The files _clear_target WOULD delete, in deletion order."""
    out: "list[Path]" = []
    try:
        for f in target.iterdir():
            if f.is_file():
                if f.name in _PRESERVED_TARGET_FILES:
                    continue
                out.append(f)
            elif f.is_dir() and f.name in _BLAS_LIBRARY_DIRS:
                out.extend(p for p in f.rglob("*") if p.is_file())
    except OSError:
        pass
    return out


def _files_in_use(target: Path) -> "list[Path]":
    """Of the files a provision would delete, those that cannot be replaced now.

    Probed with ``open(..., "r+b")``: it writes no bytes and creates nothing, and
    it asks the OS the exact question deletion asks - a DLL reports WRITABLE
    before ``ctypes.CDLL`` and PermissionError errno 13 after, the identical
    error ``unlink()`` raises on the same handle.

    Naturally platform-correct with no platform test. Windows maps a loaded DLL
    without FILE_SHARE_WRITE/DELETE, so the probe refuses exactly when deletion
    would. POSIX has no mandatory locking and unlinking an open file SUCCEEDS
    (the directory entry goes, the inode lives until the last close), so there is
    no half-state to prevent there and the probe correctly finds nothing.

    An unprobeable file counts as NOT in use, so an inconclusive answer never
    blocks a legitimate install."""
    locked: "list[Path]" = []
    for f in _clearable_files(target):
        try:
            with open(f, "r+b"):
                pass
        except PermissionError:
            locked.append(f)
        except OSError:
            # Not a "someone holds it" answer (gone, unreadable, a device). Let
            # the delete itself deal with it; _clear_target reports what remains.
            pass
    return locked


def _clear_target(target: Path) -> "list[Path]":
    """Remove previously provisioned library files so a re-provision (or a
    fallback to a different backend) never mixes DLLs from two builds. Only
    touches files in the dir, plus the _BLAS_LIBRARY_DIRS subdirectories
    _copy_blas_library_dirs may have created (never any OTHER subdirectory) -
    and never _PRESERVED_TARGET_FILES, the tracked git sentinels.

    RETURNS THE FILES IT COULD NOT REMOVE, and the return value is load-bearing:
    every caller must treat a non-empty result as a failed provision. A locked
    file left behind and reported as success would let the caller copy a new
    build over the survivors, producing the mixed-build state this prevents."""
    left: "list[Path]" = []
    try:
        for f in target.iterdir():
            if f.is_file():
                if f.name in _PRESERVED_TARGET_FILES:
                    continue
                try:
                    f.unlink()
                except OSError:
                    left.append(f)
            elif f.is_dir() and f.name in _BLAS_LIBRARY_DIRS:
                # ignore_errors keeps rmtree from raising part-way and stranding
                # the rest of the sweep; whatever survives is reported instead.
                shutil.rmtree(f, ignore_errors=True)
                if f.exists():
                    left.extend(p for p in f.rglob("*") if p.is_file())
    except OSError:
        pass
    return left


def _clear_target_or_refuse(target: Path) -> None:
    """Clear *target*, but REFUSE BEFORE DELETING ANYTHING if the runtime is in
    use. Raises RuntimeInUseError with the existing install untouched.

    Checking first makes the half-state unreachable rather than merely reported:
    a clear that reports what it could not remove has, by then, already deleted
    everything it could.

    The post-clear check is not redundant: a process can open a file in the
    window between the probe and the unlink. That race leaves a half-state and
    raises the same error rather than continuing."""
    in_use = _files_in_use(target)
    if in_use:
        raise RuntimeInUseError(in_use, partial=False)
    left = _clear_target(target)
    if left:
        raise RuntimeInUseError(left, partial=True)


def _exit_runtime_in_use(e: RuntimeInUseError) -> None:
    """Report a refused provision and exit non-zero. Never falls back to another
    backend: the user's chosen backend is not the problem, so a different one is
    never installed here."""
    console.print(f"[red]Cannot replace the installed runtime: it is in use.[/red] {e}")
    if e.partial:
        console.print("[yellow]Some files were already removed before the lock "
                      "appeared, so this install is now incomplete - re-run the "
                      "same command once nothing is using it.[/yellow]")
    else:
        console.print("[green]Your existing install was left untouched.[/green]")
    console.print("[dim]Close anything using the runtime (a running `localm serve` "
                  "or `localm gui`, a Python session that imported localm, another "
                  "setup-llama) and run the same command again.[/dim]")
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Single-flight: only ONE process may PROVISION the runtime lib dir at once   #
# --------------------------------------------------------------------------- #
# Provisioning clears then refills `target`, so two provisions racing on the
# same directory corrupt the install. The lock is CROSS-PROCESS (the GUI route
# spawns setup-llama as a child process, which a threading.Lock cannot guard)
# and is claimed with mkdir, which is atomic; stat-then-create is not.
_PROVISION_LOCK_OWNER = "owner.json"


class ProvisioningBusyError(Exception):
    """Another process already holds the provisioning lock."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _provision_lock_path(target: Path) -> Path:
    """Where the provisioning lock lives: a SIBLING of the runtime lib dir,
    never inside it, so this command's own directory clear cannot disturb the
    lock protecting it."""
    return target.parent / (target.name + ".setup.lock")


def _provision_lock_holder_pid(lock: Path) -> "Optional[int]":
    try:
        data = json.loads((lock / _PROVISION_LOCK_OWNER).read_text(encoding="utf-8"))
        pid = data.get("pid") if isinstance(data, dict) else None
        return pid if isinstance(pid, int) else None
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def _provisioning_lock(target: Path):
    """Cross-process, fail-fast single-flight guard around a run that mutates
    *target*. FAILS FAST rather than blocking: a provision can legitimately run
    for minutes (a download over a slow link), and a caller blocked that long is
    indistinguishable from a hang.

    Staleness is judged by PID LIVENESS, never elapsed time: the operation is
    unbounded, so any fixed timeout would eventually reclaim a LIVE holder's lock.
    ``pid_alive`` is conservative - when it cannot tell, it returns True - so an
    uncertain answer keeps the lock rather than stealing it.

    Raises :class:`ProvisioningBusyError` with an honest reason when the lock
    cannot be acquired. Always released in ``finally``."""
    from localm.instances import pid_alive
    lock = _provision_lock_path(target)
    acquired = False
    for attempt in (1, 2):
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(str(lock))              # ATOMIC: creates or raises
            acquired = True
            break
        except FileExistsError:
            pid = _provision_lock_holder_pid(lock)
            if pid is not None and not pid_alive(pid):
                # The holder is provably gone. Reclaim once, then retry the
                # atomic create - the retry may still lose to another caller.
                with contextlib.suppress(OSError):
                    shutil.rmtree(str(lock))
                if attempt == 1:
                    continue
                raise ProvisioningBusyError(
                    "Another setup-llama run is already provisioning the "
                    "runtime. Wait for it to finish, then try again.")
            if pid is None:
                # Cannot tell who holds it: do NOT steal it. Say how to clear it
                # by hand.
                raise ProvisioningBusyError(
                    f"A provisioning lock exists at {lock} but its owner could "
                    "not be read. If no setup-llama run is in progress, remove "
                    "that folder and try again.")
            raise ProvisioningBusyError(
                f"Another setup-llama run is already provisioning the runtime "
                f"(process {pid}). Wait for it to finish, then try again.")
        except OSError as e:
            raise ProvisioningBusyError(
                f"Could not take the provisioning lock at {lock}: {e}")
    if not acquired:
        raise ProvisioningBusyError(
            "Another setup-llama run is already provisioning the runtime. "
            "Wait for it to finish, then try again.")
    # Record who holds it, so a future caller's staleness check can ask
    # instances.pid_alive() instead of guessing from elapsed time.
    try:
        (lock / _PROVISION_LOCK_OWNER).write_text(
            json.dumps({"pid": os.getpid()}), encoding="utf-8")
    except OSError:
        pass
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(str(lock))


def _exit_provisioning_busy(e: ProvisioningBusyError) -> None:
    """Report a refused provision and exit non-zero, mirroring
    _exit_runtime_in_use. The existing install is left completely untouched,
    since the lock is taken before anything is cleared."""
    console.print(f"[red]Cannot provision the runtime right now:[/red] {e.reason}")
    sys.exit(1)


def _fetch_verified(url: str, target: Path, sha: Optional[str], what: str = "release asset") -> None:
    """Fetch + place an archive, WARNING honestly when no checksum is available
    to verify it. A GitHub asset can publish no `digest`, and the offline hash
    table only covers the tags this file pins, so the provenance check can be
    skipped - and it must never be skipped silently. Size + archive-shape checks
    still apply either way.

    _PINNED_TAG's own assets ARE in that table, so the DEFAULT install stays
    verified even when the release listing is unavailable. The `--tag latest` and
    arbitrary `--tag <x>` paths are the ones that can land here with nothing to
    check against."""
    if not sha:
        console.print(
            f"[yellow]Warning: this {what} publishes no checksum, so the download's "
            "integrity is not cryptographically verified (its size and archive "
            "shape are still checked). Pass --sha256 <hex> to pin one.[/yellow]")
    _fetch_and_place(url, target, sha)


# NVIDIA publishes its own CUDA runtime libraries (cudart, cuBLAS) as plain PyPI
# wheels. The CUDA-13-line packages are UNSUFFIXED (nvidia-cublas-cu13 etc. are
# deprecated stubs pointing at bare nvidia-cublas), so both lines are listed
# explicitly.
#
# NCCL is NOT included: the binary this fetches alongside
# (hybridgroup/llama-cpp-builder's Linux CUDA build, see _CUDA_LINUX_REPO) does
# not link against it.
_CUDA_RUNTIME_PYPI_PACKAGES = {
    "cuda-12": ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
    "cuda-13": ("nvidia-cuda-runtime", "nvidia-cublas"),
}


def _pypi_wheel_url_and_sha(package: str) -> tuple:
    """The (url, sha256) of *package*'s latest Linux x86_64 wheel from PyPI's
    JSON API, or (None, None) if unavailable. Never raises, the same contract as
    _release_assets: a best-effort lookup whose caller always has a fallback."""
    api = f"https://pypi.org/pypi/{package}/json"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/json",
                                                    "User-Agent": "localm-setup-llama"})
        with verified_urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        version = data["info"]["version"]
        for f in data["releases"].get(version, []):
            name = str(f.get("filename", ""))
            if name.endswith(".whl") and "x86_64" in name and "linux" in name.lower():
                sha = (f.get("digests") or {}).get("sha256")
                url = f.get("url")
                if url:
                    return url, sha
    except Exception as e:
        logger.debug("PyPI wheel lookup failed for %s (%s)", package, e)
    return None, None


def _fetch_pypi_runtime_lib(package: str, target: Path) -> int:
    """Download *package*'s Linux wheel from PyPI, verify it, and copy every
    ``.so*`` file it contains into *target* (flat - matches how the llama.cpp
    runtime dir is already laid out). Returns the number of files copied.
    Raises :class:`ArtifactError` on a download or validation failure, same
    contract as :func:`_fetch_and_place`, so callers decide fatal-vs-fallback
    identically for either artifact source.

    NEVER reads from any OTHER environment already on the user's machine: this
    always fetches and places a PRIVATE copy into *target*, exactly like the
    Windows cudart bundle, which is never detected on the user's system and only
    ever fetched fresh."""
    url, sha = _pypi_wheel_url_and_sha(package)
    if url is None:
        raise ArtifactError(f"could not resolve a PyPI Linux wheel for {package!r}")
    with tempfile.TemporaryDirectory() as tmp:
        wheel = Path(tmp) / f"{package}.whl"
        dl = _download(url, wheel)
        _validate_archive(wheel, expected_sha256=sha, dl=dl)   # a wheel is a zip
        ex = Path(tmp) / "x"
        _extract_archive(wheel, ex)
        n = 0
        for f in sorted(ex.rglob("*.so*")):
            if f.is_file() and not f.is_symlink():
                shutil.copy2(f, target / f.name)
                n += 1
        return n


def _fetch_cuda_runtime_libs(cuda_line: str, target: Path) -> int:
    """Fetch every PyPI-hosted CUDA runtime library for *cuda_line* ('cuda-12'
    or 'cuda-13') into *target*. Returns the total files copied. Raises
    ArtifactError from the first failing package: a single package's failure is
    not swallowed, so no partially-assembled CUDA runtime is left behind."""
    packages = _CUDA_RUNTIME_PYPI_PACKAGES.get(cuda_line, ())
    total = 0
    for pkg in packages:
        console.print(f"[dim]Fetching CUDA runtime library:[/dim] {pkg}")
        total += _fetch_pypi_runtime_lib(pkg, target)
    return total


def _provision_backend(chosen: str, target: Path, sha256: Optional[str],
                       with_cudart: bool, cuda_line: str = _CUDA_LINE,
                       tag: Optional[str] = None) -> Optional[str]:
    """Resolve + fetch the prebuilt(s) for *chosen* into *target*. For CUDA with
    *with_cudart* it also fetches the matching cudart runtime bundle so the
    build is self-contained (no CUDA Toolkit needed). *cuda_line* picks which
    upstream CUDA asset line to fetch ('cuda-12' or 'cuda-13' - see
    NvidiaInfo.cuda_line); it is ignored for every other backend. Raises on a
    fatal error.

    RETURNS the release tag this provision used, so the caller can record which
    build is now on disk, or None when there is no upstream tag to report
    (amd-rocm ships from lemonade-sdk's own numbering, which the caller already
    has as _ROCM_TAG). Each branch keeps the tag it already resolved, so
    recording the build costs no extra network call.

    *tag* pins this ONE provision to a specific upstream release, overriding
    the pin/newest resolution. It exists for the ABI walk-back, which retries
    the SAME backend against an older release, and does NOT touch the stored
    pin. Ignored for amd-rocm, which has no upstream tag."""
    if chosen == "cuda" and with_cudart and sys.platform == "win32":
        # Resolved here because this branch needs the tag to PAIR the build with
        # its matching cudart bundle, and hands the same tag to
        # _resolve_backend_asset in the no-assets fallback below.
        tag = tag or _tag_for(chosen)
        build, cudart = _resolve_cuda_pair(tag, cuda_line)
        if build is None:
            # Asset listing unavailable: fall back to the templated build URL and
            # warn that the runtime bundle could not be resolved automatically.
            console.print("[yellow]Could not resolve CUDA assets; fetching build only.[/yellow]\n"
                          "[yellow]If it fails to load, use --backend vulkan or install CUDA Toolkit.[/yellow]")
            url, fallback_sha, _t = _resolve_backend_asset("cuda", cuda_line, tag=tag)
            _fetch_verified(url, target, sha256 or fallback_sha, "CUDA build asset")
            return tag

        # Resolve build sha256
        build_digest = build.get("digest")
        build_sha = build_digest.split("sha256:")[-1].strip() if build_digest and "sha256:" in build_digest else None
        if not build_sha:
            build_sha = _PINNED_FALLBACK_SHA256.get(build["name"])
        
        console.print(f"[dim]CUDA build:[/dim] {build['name']} ({_human_mb(build.get('size'))})")
        _fetch_verified(build["browser_download_url"], target, sha256 or build_sha, "CUDA build asset")
        if cudart is not None:
            if sha256:
                # The pin is a single hash; it can only cover the build. The
                # cudart bundle is validated by size + archive shape, not by the
                # pinned digest (upstream publishes no per-asset hash here).
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
        return tag
    if chosen == "cuda" and with_cudart and sys.platform not in ("win32", "darwin"):
        # Self-contained Linux CUDA: the binary comes from a third-party
        # prebuilt, hybridgroup/llama-cpp-builder (_resolve_backend_asset's
        # linux-cuda special case, above), and the runtime libraries
        # (cudart/cublas) from PyPI wheels (_fetch_cuda_runtime_libs) - never
        # from scanning anything already on the user's machine. A raise from
        # _resolve_backend_asset (no matching build exists yet for this exact
        # upstream tag) propagates to _provision_with_fallback's caller exactly
        # like every other provisioning failure, which offers or forces the
        # vulkan fallback.
        url, fallback_sha, tag = _resolve_backend_asset("cuda", cuda_line, tag=tag)
        _fetch_verified(url, target, sha256 or fallback_sha, "CUDA build asset")
        if sha256:
            console.print("[yellow]Note:[/yellow] --sha256 pins the CUDA build only; "
                          "the PyPI runtime libraries are verified by their own "
                          "published checksums instead.")
        n = _fetch_cuda_runtime_libs(cuda_line, target)
        console.print(f"[dim]CUDA runtime:[/dim] {n} librar{'y' if n == 1 else 'ies'} "
                      "fetched from PyPI - no CUDA Toolkit install needed")
        return tag
    # Every other backend is a single archive resolved from the chosen name.
    # Also reached for chosen == "cuda" with with_cudart False; forwarding
    # cuda_line here keeps that combination from reverting to the cuda-12
    # default.
    url, fallback_sha, tag = _resolve_backend_asset(chosen, cuda_line, tag=tag)
    _fetch_verified(url, target, sha256 or fallback_sha, "release asset")
    return tag


_EXC_HEADER_RE = re.compile(
    r"^(?:[\w.]+\.)?\w*(?:Error|Exception|Warning|Interrupt|Exit)(?::|\s|\Z)")


def _informative_error_line(text: str) -> str:
    """Pull the line that actually explains a failed load from a subprocess's
    captured output.

    A Python traceback ends with the exception, but when that exception carries a
    MULTI-LINE message the literal last line is not the cause. ``load_lib()``
    raises a ``RuntimeError`` whose first line is the real dlopen error (e.g.
    ``libgomp.so.1: cannot open shared object file``) followed by four
    re-provision hint lines, so a blind ``splitlines()[-1]`` returns the last
    hint and throws the actual cause away.

    Prefers the exception HEADER line (``SomeError: <cause>``), which carries the
    real error even when the message spans several lines; falls back to the last
    non-empty line when the output is not a recognisable traceback."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "library failed to load"
    for ln in reversed(lines):
        if _EXC_HEADER_RE.match(ln.lstrip()):
            return ln.strip()
    return lines[-1].strip()


# Exit codes the load probe uses to tell its outcomes apart STRUCTURALLY rather
# than by matching text in a traceback. 89 marks an ABI rejection specifically,
# which is what the tag walk-back fires on; a CUDA build refusing to load
# because the driver is too old needs the opposite response and is not 89.
_PROBE_NO_BACKENDS = 88
_PROBE_ABI_MISMATCH = 89

# The prefix _native_loads_ok puts on an ABI rejection, matched by
# _is_abi_rejection. A string owned on both ends, not a substring search over a
# traceback.
_ABI_REJECT_PREFIX = "the runtime does not match this build's struct layout"

# load_lib() runs verify_abi and RE-RAISES (see _loader.py: `except Exception:
# _loaded_lib = None; raise`), so AbiMismatch propagates out uncaught and can be
# caught here.
_LOAD_PROBE_CODE = f"""\
import sys
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._abi import AbiMismatch
try:
    _loader.load_lib()
except AbiMismatch as e:
    sys.stderr.write(str(e))
    sys.exit({_PROBE_ABI_MISMATCH})
sys.exit(0 if _loader.compute_backends_available() else {_PROBE_NO_BACKENDS})
"""


def _is_abi_rejection(detail: "Optional[str]") -> bool:
    """Whether *detail* is _native_loads_ok reporting OUR OWN ABI gate refusing
    the runtime, as opposed to any other load failure.

    The discriminator for the tag walk-back: an ABI rejection means the BUILD is
    wrong for this code, which a different release can fix; every other load
    failure is about this machine, which a different release cannot."""
    return str(detail or "").startswith(_ABI_REJECT_PREFIX)


def _native_loads_ok() -> tuple:
    """Load-test the provisioned native library in a FRESH interpreter, exactly
    as ``localm run`` will, AND confirm it registered a compute backend. A build
    can load cleanly yet register ZERO backends ("no backends are loaded"), which
    counts as a FAILED provision rather than a silent success - otherwise a
    broken runtime slips through and fails only at the first model load, with the
    real cause already lost. A subprocess keeps the setup process clean (the
    loader mutates the DLL/lib search path) and matches the real run environment.
    Returns (ok, last_error_line)."""
    try:
        r = subprocess.run([sys.executable, "-c", _LOAD_PROBE_CODE],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, str(e)
    if r.returncode == 0:
        return True, ""
    if r.returncode == _PROBE_NO_BACKENDS:
        return False, ('runtime loaded but registered no compute backends '
                       '("no backends are loaded") - this build does not fit this machine')
    if r.returncode == _PROBE_ABI_MISMATCH:
        # Behind its own prefix so callers can recognise this specific outcome
        # without re-parsing upstream's wording - see _is_abi_rejection.
        why = _informative_error_line((r.stderr or "").strip()) or "layout drift"
        return False, f"{_ABI_REJECT_PREFIX}: {why}"
    detail = (r.stderr or r.stdout or "").strip()
    return False, _informative_error_line(detail)


def _warn_off_profile(chosen: str):
    """One-line heads-up when a vendor-specific backend was chosen for a vendor
    that was NOT detected. The user's choice stands - no block, no nag, no
    re-prompt - and the mismatch is flagged once.

    Returns the ``hwdetect.Detection`` used for the check (or ``None`` if it was
    never computed, or detection failed), so a caller needing the SAME vendor
    info downstream - the CUDA dialogue, to name what IS present instead of a
    generic "not found" - does not call ``hwdetect.detect()`` a second time and
    cannot see different hardware."""
    vendor_specific = {"cuda": "nvidia", "amd-rocm": "amd", "hip": "amd",
                       "sycl": "intel", "metal": "apple"}
    owner = vendor_specific.get(chosen)
    if not owner:
        return None
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception:
        return None
    vendors = det.vendors or []
    if vendors and owner not in vendors:
        seen = ", ".join(vendors)
        console.print(f"[yellow]Heads up:[/yellow] Picked [bold]{chosen}[/bold] but detected [bold]{seen}[/bold].\n"
                      "[yellow]Proceeding. Hardware must be present.[/yellow]")
    return det


def _flush_stdin() -> None:
    """Discard any input the OS/terminal buffered while nothing was waiting on it
    (e.g. a stray Enter pressed while a driver probe or a multi-hundred-MB
    download was running). Without this, that buffered keystroke is consumed the
    instant the NEXT ``click.confirm()`` prompt appears, answering a question the
    user never read. Call this immediately before every interactive prompt in the
    setup flow.

    Best-effort and silent on failure: a piped/non-tty stdin (tests, CI, a
    non-interactive install) has nothing to flush and isatty() already guards
    that; any other failure leaves stray input in place."""
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


def _cuda_setup_dialogue(info: NvidiaInfo, assume_yes: bool, det=None) -> tuple:
    """Given the preflight, walk the user through making CUDA land. Returns
    ``(backend_to_provision, fetch_cudart_bundle)``.

    Branches:
      * driver new enough  -> offer the self-contained build+runtime fetch
        (default yes); declining falls back to vulkan.
      * driver too old     -> a driver cannot be self-assembled; recommend
        vulkan now and tell them how to enable CUDA later.
      * no NVIDIA detected, hardware unknown -> the user forced cuda;
        confirm-continue, else vulkan (warn-once, do not block).
      * no NVIDIA detected, but *det* (the SAME hwdetect.Detection
        _warn_off_profile already computed) shows a DIFFERENT vendor is
        actually present -> name it, recommend the policy-backed match for it
        (hwdetect.recommended_install_backend), and offer a three-way choice
        (continue / switch to the recommendation / quit).

    *det* is optional; when it is None or shows no vendor other than nvidia the
    dialogue takes the generic branches above.
    """
    console.print("[bold]CUDA selected[/bold] (peak NVIDIA performance). "
                  "Checking your system...")
    other_vendors = [v for v in (det.vendors if det else []) if v != "nvidia"]
    if info.present:
        console.print(f"  [green]OK[/green] NVIDIA GPU: {info.gpu_name or 'detected'}")
        if info.compute_capability:
            line_note = " (Blackwell)" if info.cuda_line == "cuda-13" else ""
            console.print(f"  [dim]Compute capability {info.compute_capability}{line_note} "
                          f"-> {info.cuda_line} line[/dim]")
        if info.cuda_capability:
            colour = "green" if info.driver_ok else "red"
            mark = "OK " if info.driver_ok else "no "
            need = _MIN_DRIVER_CUDA[info.cuda_line]
            console.print(f"  [{colour}]{mark}[/{colour}] Driver {info.driver_version} "
                          f"supports CUDA {info.cuda_capability} "
                          f"(need >= {need[0]}.{need[1]} for the {info.cuda_line} line)")
    elif other_vendors:
        seen = "/".join(v.upper() for v in other_vendors)
        console.print(f"  [yellow]?[/yellow] Could not run nvidia-smi - this machine "
                      f"looks like [bold]{seen}[/bold], not NVIDIA.")
    else:
        console.print("  [yellow]?[/yellow] Could not run nvidia-smi - no NVIDIA driver "
                      "detected here (or it is not on PATH).")

    # Driver too old: the one thing we cannot fetch for the user.
    if info.present and info.cuda_capability and not info.driver_ok:
        console.print(f"  GPU driver update required for CUDA "
                      f"(the {info.cuda_line} line this GPU needs).")
        console.print("  [dim]To enable later: update driver, reboot, run setup-llama --backend cuda[/dim]")
        console.print("  [green]Using Vulkan now[/green].")
        return "vulkan", False

    # No NVIDIA detected, but the user explicitly asked for cuda: their call.
    if not info.present:
        if not other_vendors:
            # No specific alternative to recommend - the generic
            # continue-or-vulkan choice, which is also where a fully headless
            # machine, or one where hwdetect itself failed, lands.
            if assume_yes:
                console.print("  [dim]--yes: using Vulkan (no NVIDIA GPU detected).[/dim]")
                return "vulkan", False
            _flush_stdin()
            if click.confirm("  Continue with CUDA anyway? (No = use Vulkan)", default=False):
                return "cuda", True
            return "vulkan", False

        # Recommend the real match for what IS here, via the SAME policy
        # setup.bat/sh use.
        from localm import hwdetect as _hwdetect
        recommended = _hwdetect.recommended_install_backend(det)
        if assume_yes:
            seen = "/".join(v.upper() for v in other_vendors)
            console.print(f"  [dim]--yes: using {recommended} "
                          f"({seen} detected, not NVIDIA).[/dim]")
            return recommended, False
        _flush_stdin()
        console.print("    [1] Continue with CUDA anyway")
        console.print(f"    [2] Switch to {recommended} (recommended for your hardware)")
        console.print("    [3] Quit")
        pick = click.prompt("  Pick 1-3", type=click.Choice(["1", "2", "3"]), default="2")
        if pick == "1":
            return "cuda", True
        if pick == "3":
            sys.exit(1)
        return recommended, False

    # Driver OK (or capability unknown but a GPU is present): offer the fetch.
    console.print("  [yellow]i[/yellow] Fetching self-contained CUDA runtime bundle. [bold]No Toolkit needed[/bold].")
    _flush_stdin()
    if assume_yes or click.confirm("  Download the CUDA build + runtime now?", default=True):
        return "cuda", True
    console.print("  [dim]Falling back to Vulkan (works on your driver).[/dim]")
    return "vulkan", False


# A FLOOR at _PINNED_TAG, not a walk over recent upstream releases: it has
# exactly ONE destination, the confirmed build, so it can only move the user
# towards that build.
_FLOOR_TAG_DESCRIPTION = "the confirmed build localm ships"


def _floor_at_pinned_tag(chosen: str, with_cudart: bool, rejected_tag: str,
                         try_fn, detail: str) -> tuple:
    """After our own ABI gate refused *rejected_tag*, fall back to _PINNED_TAG -
    the one confirmed build - and only from an install that had opted OUT of it.
    Returns ``(ok, tag)``.

    LOUD IN EVERY BRANCH, including the three that install nothing. Each branch
    names the build involved, the ABI reason, and the --tag command that changes
    it.

    THE FOUR CASES:

      * an exact USER PIN - not moved, ever, and the user is told which commands
        change it. The check lives here rather than at the call site so a second
        caller cannot bypass it.
      * TRACKING upstream (--tag latest) - the case the floor exists for. Exactly
        one attempt: the destination is a constant, so there is nothing to
        iterate over.
      * ALREADY ON THE PIN - nothing to fall back TO; the floor IS what was just
        refused, which means localm shipped a pin its own binding rejects. Said
        out loud; no older release is installed instead.
      * the pin itself then fails to load - reported with both causes."""
    pin = pinned_tag()
    if pin:
        console.print(
            f"[red]The pinned llama.cpp build {pin} does not load on this "
            f"machine:[/red] {detail}")
        console.print("[dim]Your pin is kept, not changed. Move it with: "
                      "localm setup-llama --rollback  (previous build), "
                      "localm setup-llama --tag default  (the build localm "
                      "ships and confirmed), or localm setup-llama --tag latest "
                      "(track upstream).[/dim]")
        return False, None

    console.print(f"[yellow]llama.cpp {rejected_tag} was rejected by localm's own "
                  f"ABI check:[/yellow] {detail}")
    console.print("[dim]That is a mismatch between the release and this build of "
                  "localm, not a fault of your machine.[/dim]")

    if rejected_tag == _PINNED_TAG or not tracks_latest():
        # No floor below the floor: no older release is hunted for, since such a
        # build is one nobody confirmed.
        console.print(
            f"[red]{_PINNED_TAG} is {_FLOOR_TAG_DESCRIPTION}, so there is no "
            "more-tested build to fall back to.[/red]")
        console.print("[dim]This means this localm and its own pinned llama.cpp "
                      "build disagree, which is a bug in localm rather than in "
                      "the release. Please report it. To try another build "
                      "meanwhile: localm setup-llama --tag <release>  (for "
                      "example --tag b10361).[/dim]")
        return False, None

    console.print(f"[yellow]Falling back to {_PINNED_TAG}, {_FLOOR_TAG_DESCRIPTION}"
                  f".[/yellow]")
    try:
        try_fn(chosen, with_cudart, _PINNED_TAG)
    except Exception as e:
        console.print(f"[red]{_PINNED_TAG} could not be provisioned:[/red] {e}")
        return False, None
    ok, why = _native_loads_ok()
    if not ok:
        console.print(f"[red]{_PINNED_TAG} did not load either:[/red] "
                      f"{why or 'unknown'}")
        return False, None
    # State the outcome and how it differs from what was asked for.
    console.print(f"[green]OK - llama.cpp {_PINNED_TAG} loads on this machine."
                  "[/green]")
    console.print(f"[dim]Installed {_PINNED_TAG} rather than {rejected_tag}: you "
                  "asked to track upstream's newest (--tag latest) and that "
                  "release does not match this build of localm. Update localm "
                  "and re-run 'localm setup-llama --force' to move forward "
                  "again.[/dim]")
    return True, _PINNED_TAG


def _sycl_backend_note() -> str:
    """Describe the SYCL build's runtime dependency for the current OS.

    Windows and Linux ship different SYCL archives: the Windows zip bundles the
    whole oneAPI DPC++ runtime alongside ggml-sycl.dll (sycl8.dll, mkl_*.dll,
    ur_adapter_level_zero*.dll, ur_adapter_opencl.dll, tbb12.dll,
    libiomp5md.dll, dnnl.dll, sycl-ls.exe, ...), while the Linux tarball ships
    only libggml-sycl.so and still requires a separate oneAPI install."""
    if sys.platform == "win32":
        return "Intel oneAPI build + self-contained oneAPI runtime"
    return "Intel oneAPI build (needs the oneAPI runtime present)"


def _provision_with_fallback(chosen: str, target: Path, sha256: Optional[str],
                             with_cudart: bool, assume_yes: bool = False,
                             cuda_line: str = _CUDA_LINE) -> tuple[str, Optional[str]]:
    """Provision *chosen* and prove it loads. If it does not load, NEVER swap the
    user's pick silently: inform WHY, then OFFER the universal Vulkan build when
    interactive (or fall back with a LOUD warning when *assume_yes* / no tty),
    and always say how to retry the chosen backend with --force. Exits non-zero
    if the user declines the fallback, or if NOTHING loads.

    Returns ``(backend, tag)``: the backend that ended up working AND the release
    tag it was provisioned from (None when that backend has no upstream tag).
    The tag belongs to the attempt that SUCCEEDED, so on a cuda-to-vulkan
    fallback it is vulkan's build, not the one that failed.

    *cuda_line* is the CUDA asset line to fetch when *chosen* is 'cuda' (see
    NvidiaInfo.cuda_line); irrelevant otherwise.

    vulkan and cpu are self-contained and treated as terminal: if the user
    explicitly chose one and it does not load, that is reported as an environment
    problem rather than papered over with a different backend."""
    lib_name = _lib_name()

    # The tag of the attempt currently in flight. _try writes it; the success
    # paths below read it. A list rather than a rebound local because _try is a
    # closure and would otherwise need a `nonlocal` declaration.
    used_tag: list = [None]

    def _try(backend: str, cudart: bool, tag: Optional[str] = None) -> None:
        _clear_target_or_refuse(target)
        # Cleared FIRST, so a failed attempt can never leave the previous
        # attempt's tag standing to be recorded against this backend.
        used_tag[0] = None
        used_tag[0] = _provision_backend(
            backend, target, sha256 if backend == chosen else None,
            cudart, cuda_line, tag=tag)
        if not (target / lib_name).exists():
            raise ArtifactError(f"the archive did not contain {lib_name}")
        _install_runtime_wheel(_runtime_pkg_dir())

    notes = {
        "vulkan": "universal GPU build (AMD/NVIDIA/Intel via the display driver)",
        "amd-rocm": "self-contained AMD ROCm build (gfx103X / RX 6000)",
        "cuda": "NVIDIA CUDA build + self-contained runtime",
        "sycl": _sycl_backend_note(),
        "hip": "AMD ROCm build (needs the ROCm/HIP runtime present)",
        "cpu": "CPU-only build (no GPU)",
        "metal": "Apple Silicon (Metal) build",
    }
    console.print(f"[dim]Backend:[/dim] [bold]{chosen}[/bold]  ({notes.get(chosen, chosen)})")

    provisioned = True
    try:
        _try(chosen, with_cudart)
    except RuntimeInUseError as e:
        # MUST precede the handlers below, and must NOT fall through to the
        # Vulkan fallback: falling back is right when the CHOSEN BUILD cannot run
        # here, and wrong when the chosen build is fine and a process is merely
        # holding a file.
        _exit_runtime_in_use(e)
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
        return chosen, used_tag[0]

    # ---- The installer must never hand the user a runtime our OWN gate rejects.
    #
    # This runs BEFORE the backend fallback below. An ABI rejection means the
    # BUILD is wrong for this code, so EVERY backend from that release fails
    # identically - one shared llama library carries the struct (see
    # _PINNED_TAG). Falling back by backend first cannot help, burns the whole
    # chain, and ends in "no backend could be provisioned".
    #
    # Gated on an ABI rejection SPECIFICALLY, never on any load failure: a cuda
    # build refusing to load because the driver is too old is about this
    # MACHINE, and that case must still reach the vulkan fallback.
    #
    # No tag is hard-coded as bad; the property enforced is "never ship a runtime
    # our gate rejects", whichever tag and whatever the cause.
    if _is_abi_rejection(detail) and used_tag[0]:
        floored, floor_tag = _floor_at_pinned_tag(chosen, with_cudart, used_tag[0],
                                                  _try, detail)
        if floored:
            return chosen, floor_tag
        # Fall through: the confirmed build did not load either (or there was
        # none to fall back to), so this is not release drift and the backend
        # fallback is next.

    # An explicit --sha256 pin means "exactly this artifact" - never swap to a
    # different (unpinned) build, even to recover. Report and stop.
    if sha256:
        why = "failed validation" if not provisioned else "provisioned but did not load"
        console.print(f"[red]The pinned artifact {why}.[/red] Not falling back "
                      "(an explicit --sha256 was set).")
        sys.exit(1)

    # A self-contained backend the user pinned: do not silently swap to another.
    if chosen in ("vulkan", "cpu"):
        if not provisioned:
            # The exception handler above already printed the SPECIFIC cause
            # (from _diagnose_bad_artifact, on a download/validation failure);
            # this adds the escape hatches, which that message does not mention.
            console.print(
                f"[dim]If your network blocks or filters this download (common on "
                f"a corporate network), download the archive yourself through a "
                f"browser and use --from <extracted-dir>, or point --url at a "
                f"mirror your network allows. Retry the same command once the "
                f"cause is fixed: localm setup-llama --backend {chosen}[/dim]")
            sys.exit(1)
        # Provisioned but would not load: vulkan/cpu are the universal
        # fallbacks, so this is an unexpected environment fault worth a report,
        # not an exit-0 "success" on a broken runtime.
        from localm.bugreport import LocalmError
        raise LocalmError(
            f"{chosen} was provisioned but the native library did not load",
            reason=(f"{chosen} is the self-contained fallback and still failed to load "
                    f"({detail}) - likely a broken/incompatible binary or a missing OS "
                    "dependency. See docs/gpu-setup.md."),
            context={"operation": "setup-llama", "backend": chosen})

    # chosen needs a runtime and did not load HERE. The user's pick is never
    # swapped silently: INFORM why, then OFFER the universal build (interactive)
    # or fall back with a LOUD warning (non-interactive), and always say how to
    # retry the real pick once the cause is fixed.
    why = detail
    # Every attempt's own cause, chosen backend first - NOT just the last one
    # tried. The final LocalmError's *reason* is the only thing that reaches the
    # saved bug-report file and the "Sorry - X because Y" console line.
    attempts = [(chosen, why)]
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
            attempts.append((fb, str(e)))
            continue
        ok, fb_detail = _native_loads_ok()
        if ok:
            console.print(f"[green]OK - {fb} runtime loads.[/green]")
            return fb, used_tag[0]
        console.print(f"[red]{fb} provisioned but failed to load:[/red] {fb_detail or 'unknown'}")
        attempts.append((fb, fb_detail or "unknown"))
    # Nothing loaded. Raise a typed, reportable error and let the CLI's single
    # graceful handler say sorry and offer a bug report. setup-llama describes
    # the failure; it does not own reporting.
    from localm.bugreport import LocalmError
    tried = "; ".join(f"{b}: {d}" for b, d in attempts)
    raise LocalmError(
        "no llama.cpp backend could be provisioned and loaded",
        reason=(f"none of {len(attempts)} backends loaded on this machine - {tried}. "
                "You can provide a local build "
                "with: localm setup-llama --from <build dir>, or see docs/gpu-setup.md."),
        context={"operation": "setup-llama", "requested_backend": chosen})


def _validated_tag(raw: str) -> str:
    """*raw* as a usable release tag, or a ClickException naming the problem.

    The CLI-facing half of is_safe_tag: the same predicate, so the flag and the
    stored-value check cannot disagree about what a usable tag is."""
    tag = (raw or "").strip()
    if not is_safe_tag(tag):
        raise click.ClickException(
            f"{raw!r} is not a usable release tag. {TAG_HELP}")
    return tag


def _apply_version_request(tag: Optional[str], rollback: bool, backend: str,
                           from_dir: Optional[str], url: Optional[str]) -> None:
    """Act on --tag / --rollback BEFORE any provisioning: validate them, resolve
    what --rollback means, and move the pin.

    The pin is written FIRST, so the rest of main() provisions through the normal
    _tag_for() path with no special-casing, and the pin also applies to `localm
    update`, which re-invokes this command with nothing but --backend (see
    _apply_update.post_swap_command).

    Anything this cannot honour is REFUSED with a reason rather than ignored."""
    # `tag is None` (the flag was not passed) is distinguished from `tag == ""`
    # (passed empty, e.g. a shell variable that expanded to nothing). The empty
    # string falls through to _validated_tag and is refused with a reason.
    if tag is not None and rollback:
        raise click.ClickException(
            "--tag and --rollback both choose a build; pass only one. "
            "--rollback goes to the previous recorded build, --tag names one.")
    if tag is None and not rollback:
        return
    if from_dir or url:
        # --from/--url install an artifact this command did not resolve from a
        # release, so there is no tag to record or pin. The flag is refused
        # rather than accepted with no effect.
        which = "--from" if from_dir else "--url"
        raise click.ClickException(
            f"{'--tag' if tag is not None else '--rollback'} selects an upstream "
            f"llama.cpp release, so it cannot be combined with {which}, which "
            "installs a build you supply. Run them separately.")

    if tag is not None:
        # TWO WORDS, naming two destinations: the unpinned default is the build
        # localm confirmed, and upstream's newest is a separate request. Both are
        # words rather than an empty --tag, so the intent is visible in shell
        # history and a shell variable that expanded to nothing cannot silently
        # change what an install tracks.
        if tag.strip().lower() == _TRACK_DEFAULT:
            set_pinned_tag(None)
            console.print(f"[green]Back to localm's confirmed build[/green] "
                          f"({_PINNED_TAG}) - the one this release was tested "
                          "with. Re-run setup-llama --force to install it now.")
            return
        if tag.strip().lower() == _TRACK_LATEST:
            set_pinned_tag(_TRACK_LATEST)
            console.print("[yellow]Now tracking upstream's newest llama.cpp "
                          "release.[/yellow] That build is whatever ggml-org "
                          "published most recently and localm has NOT tested it; "
                          "upstream has shipped releases this code cannot load. "
                          f"Go back with: [bold]localm setup-llama --tag "
                          f"{_TRACK_DEFAULT}[/bold] ({_PINNED_TAG}).")
            return
        wanted = _validated_tag(tag)
        set_pinned_tag(wanted)
        console.print(f"[green]Pinned[/green] llama.cpp {wanted} - setup-llama "
                      "and localm update will keep this build until you run "
                      f"[bold]localm setup-llama --tag {_TRACK_DEFAULT}[/bold] "
                      f"(back to localm's confirmed {_PINNED_TAG}).")
        return

    # --rollback. The backend is the one the user named, else whatever is
    # installed: history is per-backend, because a cuda tag and a vulkan tag are
    # different builds even when the tag string matches.
    which = backend.lower() if backend and backend.lower() != "auto" else installed_backend()
    if not which:
        raise click.ClickException(
            "--rollback needs to know which backend to roll back, and nothing is "
            "recorded as installed on this machine. Name it explicitly, for "
            "example: localm setup-llama --rollback --backend vulkan")
    if which == "amd-rocm":
        # amd-rocm's build is fixed by the _ROCM_TAG CONSTANT in this file, not
        # resolved from a release listing, so there is exactly one amd-rocm build
        # per localm release and a pin cannot move it. Rollback is refused with
        # that reason.
        raise click.ClickException(
            f"the amd-rocm backend cannot be rolled back: its build is fixed by "
            f"the localm release you are running ({_ROCM_TAG}, from lemonade-sdk), "
            "not chosen from upstream llama.cpp releases. To try a different "
            "llama.cpp build on this machine, switch backend, for example: "
            "localm setup-llama --backend vulkan --tag <tag>")
    prev = previous_tag(which)
    if not prev:
        current = installed_build()
        have = f" The build installed now is {current}." if current else ""
        raise click.ClickException(
            f"no earlier llama.cpp build is recorded for the {which} backend, so "
            f"there is nothing to roll back to.{have} Install a specific build "
            "instead, for example: localm setup-llama --tag b10355")
    set_pinned_tag(prev)
    console.print(f"[green]Rolling back[/green] the {which} runtime to llama.cpp "
                  f"{prev}, and pinning it.")


@click.command("setup-llama", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Copy binaries from a local llama.cpp build directory instead of downloading.")
@click.option("--backend", default="auto",
              type=click.Choice(list(BACKENDS), case_sensitive=False),
              help="Which prebuilt to fetch. 'auto' detects your GPU and picks "
                   "the best-performing backend it can run out of the box: cuda "
                   "for NVIDIA on both Windows and Linux (self-contained, falls "
                   "back to vulkan if the driver is too old); the self-contained "
                   "ROCm build for AMD RX 6000 on Windows; hip for AMD elsewhere "
                   "when a system ROCm/HIP toolkit is detected; vulkan for Intel "
                   "and for AMD with no toolkit detected; cpu if no GPU.")
@click.option("--url", default=None, help="Override with an explicit prebuilt archive URL.")
@click.option("--sha256", "sha256", default=None,
              help="Expected sha256 of the downloaded archive. When given, the "
                   "download is refused unless its digest matches (opt-in "
                   "integrity pin).")
@click.option("--force", is_flag=True, help="Re-provision even if binaries are already present.")
@click.option("--tag", "tag", default=None, metavar="TAG",
              help="Install a specific llama.cpp release (e.g. 'b10355') and PIN "
                   "it, so later setup-llama runs and 'localm update' keep that "
                   "exact build. Two words are special: 'latest' opts in to "
                   "upstream's newest release, which localm has not tested, and "
                   "'default' returns to the build localm ships and confirmed.")
@click.option("--rollback", is_flag=True,
              help="Go back to the previous llama.cpp build recorded for this "
                   "backend and pin it. For when an upstream release turns out to "
                   "be broken on your hardware. See 'localm doctor' for what is "
                   "installed now.")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Non-interactive: accept the recommended action at every prompt "
                   "(e.g. fetch the self-contained CUDA runtime). Used by the "
                   "one-click installer and for scripted setups.")
def main(from_dir: Optional[str], backend: str, url: Optional[str],
         sha256: Optional[str], force: bool, tag: Optional[str],
         rollback: bool, assume_yes: bool) -> None:
    """Download or copy the native llama.cpp binaries into localm's own venv.

    The chosen backend is load-tested after provisioning. If it cannot load on
    this machine (e.g. CUDA without a new-enough driver) your pick is NOT changed
    silently: setup explains why and (interactively) offers the universal Vulkan
    build instead, or - in a non-interactive install - falls back with a loud
    warning and tells you how to retry your backend once the cause is fixed.

    --tag pins one exact build and --rollback returns to the previous one; both
    stick, including across 'localm update'.

    \b
      localm setup-llama                        # auto-detect GPU, fetch the right prebuilt
      localm setup-llama --backend vulkan       # universal GPU build (any vendor)
      localm setup-llama --backend cuda         # NVIDIA: checks the driver, fetches a
                                                #   self-contained CUDA runtime (no Toolkit)
      localm setup-llama --backend cpu          # no GPU
      localm setup-llama --from /path/to/llama.cpp/build/bin
      localm setup-llama --url https://.../llama-...zip
      localm setup-llama --sha256 <hex>         # pin the expected archive digest
      localm setup-llama --tag b10355           # install exactly b10355 and keep it
      localm setup-llama --tag latest           # track upstream's newest (untested here)
      localm setup-llama --tag default          # back to the build localm confirmed
      localm setup-llama --rollback             # back to the previous build
    """
    lib_name = _lib_name()
    target = _repo_runtime_lib()
    _apply_version_request(tag, rollback, backend, from_dir, url)
    # A version request is inherently a re-provision: the guard below compares
    # BACKENDS, while a --tag/--rollback changes the BUILD with the backend
    # unchanged. Without this an explicit --tag/--rollback on an
    # already-provisioned box would print "Already provisioned" and change
    # nothing, having just moved the pin, so config and disk would disagree.
    if tag is not None or rollback:
        force = True
    target.mkdir(parents=True, exist_ok=True)

    already = (target / lib_name).exists()
    if already and not force:
        # Backend-aware guard: 'auto' means "give me something that works", and
        # something already does, so nothing is re-downloaded. An EXPLICIT
        # backend short-circuits only when THAT backend is confirmed to be the
        # one on disk; otherwise it falls through and provisions what was asked
        # for.
        want = backend.lower()
        have = _provisioned_backend(target)
        if want == "auto" or (have is not None and have == want):
            # Name the BUILD as well as the backend when it is recorded.
            build = _provisioned_build(target) if have else None
            label = f" ({have} {build})" if build else (f" ({have})" if have else "")
            console.print(f"[green]Already provisioned[/green]{label} at {target}")
            if not assume_yes and sys.stdin and sys.stdin.isatty() and click.confirm("Do you want to re-download/replace them?", default=False):
                force = True
                console.print("[yellow]Replacing existing build...[/yellow]")
            else:
                _ensure_importable()
                return
        # Four distinct situations. Two of them - a re-download of the same
        # backend, and 'auto' as the requested backend - get their own wording;
        # the other two keep their existing wording.
        if not have:
            console.print(f"[yellow]Replacing unrecorded build with {want}.[/yellow]")
        elif want == "auto":
            console.print(f"[yellow]Replacing {have} build with the "
                          f"auto-detected backend.[/yellow]")
        elif have == want:
            # The marker carries the installed build tag when it is known (see
            # _provisioned_build), so an upgrade can name the build it replaces.
            # The build about to be installed can only be named for free by
            # amd-rocm (_ROCM_TAG is a constant), so only amd-rocm gets the
            # "X -> Y" arrow; the others name the build being replaced and stop
            # there.
            #
            # A marker with no build tag reads back None and keeps the original
            # wording.
            have_build = _provisioned_build(target)
            if want == "amd-rocm" and have_build and have_build != _ROCM_TAG:
                console.print(f"[yellow]Upgrading the {have} build: "
                              f"{have_build} -> {_ROCM_TAG}.[/yellow]")
            elif have_build:
                console.print(f"[yellow]Re-downloading the {have} build "
                              f"({have_build}).[/yellow]")
            else:
                # NOT named `tag`: that is this command's --tag parameter, and
                # rebinding it here would shadow the user's request.
                tag_label = f" ({_ROCM_TAG})" if want == "amd-rocm" else ""
                console.print(
                    f"[yellow]Re-downloading the {have} build{tag_label}.[/yellow]")
        else:
            console.print(f"[yellow]Replacing {have} build with {want}.[/yellow]")

    # Everything below MUTATES target (clear + refill), so it is guarded by the
    # cross-process provisioning lock - see _provisioning_lock's docstring.
    # Nothing above this point (the "already provisioned" read and its
    # short-circuit) touches disk, so it runs unlocked.
    try:
        with _provisioning_lock(target):
            if from_dir:
                src = Path(from_dir)
                console.print(f"Copying binaries from [bold]{src}[/bold] ...")
                try:
                    _clear_target_or_refuse(target)
                except RuntimeInUseError as e:
                    _exit_runtime_in_use(e)
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
                    # The user pinned this build, so there is no fallback - but a
                    # library that will not load must not be reported as success.
                    # Exit non-zero with a clear reason.
                    console.print(f"[red]Copied, but the library did not load[/red] "
                                  f"({detail}) - is it built for this OS/GPU? "
                                  "See docs/gpu-setup.md.")
                    sys.exit(1)
            elif url:
                if not sha256:
                    console.print("[yellow]Warning: Custom URL download is unverified (no --sha256 provided).[/yellow]")
                console.print(f"[dim]Fetching:[/dim] {url}")
                try:
                    _clear_target_or_refuse(target)
                    _fetch_and_place(url, target, sha256)
                except RuntimeInUseError as e:
                    # Ahead of the broad handlers below, which would otherwise
                    # report a locked file as "Download failed".
                    _exit_runtime_in_use(e)
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
                # warn-once-then-comply: an explicit off-profile choice stands,
                # but a vendor mismatch is flagged a single time. Also captures
                # the SAME detection for the CUDA dialogue below, so a "no NVIDIA
                # found" fallback can name the vendor that IS present without
                # computing it a second time.
                det = _warn_off_profile(chosen) if backend != "auto" else None
                # CUDA is the visible peak-NVIDIA option: detect the driver, then offer to
                # fetch a self-contained runtime (no Toolkit) or fall back cleanly.
                with_cudart = False
                cuda_line = _CUDA_LINE
                # NOT platform-gated to win32: nvidia_preflight() and
                # _cuda_setup_dialogue() are both platform-neutral (nvidia-smi
                # runs on Linux too, and the dialogue's text and branches
                # reference no OS). Only darwin is excluded - CUDA is not a real
                # path on Apple Silicon.
                if chosen == "cuda" and sys.platform != "darwin":
                    # Preflight ONCE and reuse it for both the dialogue and the
                    # asset line: a second nvidia-smi call could see different
                    # hardware and pick a line the dialogue never displayed.
                    info = nvidia_preflight()
                    cuda_line = info.cuda_line
                    chosen, with_cudart = _cuda_setup_dialogue(info, assume_yes, det)
                _pin_note_for_backend(chosen)
                result, used_tag = _provision_with_fallback(chosen, target, sha256,
                                                            with_cudart, assume_yes,
                                                            cuda_line)
                # Record the build tag for EVERY backend, not only amd-rocm.
                # _provision_with_fallback returns the tag of the attempt that
                # SUCCEEDED, so this costs no additional lookup and, on a
                # fallback, records the build actually installed rather than the
                # one that failed.
                #
                # amd-rocm supplies _ROCM_TAG from the constant, because its
                # build is not resolved from an upstream tag at all (used_tag is
                # None for it).
                build = _ROCM_TAG if result == "amd-rocm" else used_tag
                _record_provisioned_backend(target, result, build=build)
                _record_runtime_history(result, build)

            _verify()
    except ProvisioningBusyError as e:
        _exit_provisioning_busy(e)


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
        # Surface a verify failure instead of exiting silently after "setup done".
        console.print(f"[yellow]Warning:[/yellow] could not verify the native runtime "
                      f"({e}); it may not load. Run [bold]localm doctor[/bold] to check.")
