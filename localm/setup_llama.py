# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Makes localm self-contained: the native inference runtime (the llama shared
library + its ggml deps, plus a matched GPU runtime when the prebuilt ships one)
is placed inside the project's own ``localm-llama-runtime`` wheel rather than
depending on a folder elsewhere on disk.

Backends (``--backend``), so any machine has a working out-of-the-box path:
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
    automatically gets the newer 13.x line; every older architecture stays on
    the broad-compatibility 12.x line).
  * ``hip`` - AMD peak performance via an already-installed system ROCm/HIP
    toolkit (a real downloadable prebuilt binary on both Windows and Linux;
    needs that toolkit present to load - see ``_rocm_toolkit_present`` in
    hwdetect.py).
  * ``sycl`` / ``cpu`` - upstream llama.cpp prebuilts. ``sycl`` delivers peak
    Intel performance; the Windows build bundles the whole oneAPI DPC++
    runtime and is self-contained, while the Linux build does not and needs
    oneAPI installed separately (no Intel-GPU-presence probe for either yet,
    so it stays opt-in); ``cpu`` is self-contained.
  * ``amd-rocm`` - the self-contained gfx103X (RDNA2) ROCm build (bundles its
    own ROCm runtime; the current default for AMD RX 6000 on Windows, since it
    needs no system toolkit at all).

Sources, in order of preference:
  * ``--from <dir>``  - copy from a local llama.cpp build output (any backend).
  * ``--url <url>``   - an explicit prebuilt archive URL.
  * ``--backend ...`` - resolve the matching asset of the PINNED llama.cpp
    release (``ggml-org/llama.cpp``); see below.

Which BUILD, as distinct from which backend:
  * localm installs ``_PINNED_TAG``: one upstream release we confirmed loads AND
    generates, decided in this file. No version is ever computed while setup is
    running, so a release published by a third party cannot change what an
    install gets. See _PINNED_TAG's own comment for what "confirmed" covers per
    backend, and scripts/confirm_llama_runtime.py for the check that earns it.
  * The installed release tag is recorded in the runtime dir's marker alongside
    the backend, so ``localm doctor`` and a bug report can name the build
    instead of inferring it from library filenames.
  * ``--tag <tag>`` installs one exact release and PINS it, so later runs and
    ``localm update``'s re-provision keep it. ``--tag latest`` opts IN to
    upstream's newest release, which localm has not confirmed; ``--tag default``
    returns to the shipped pin. All three live in one config key
    (``llama_runtime_pin``) read by ``_tag_for``, which is why the updater
    inherits the choice without knowing it exists.
  * ``--rollback`` returns to the previous build recorded for this backend.
    This exists because upstream can ship a release that is broken on a given
    machine - the ``llama_context_params`` ABI shift, an asset rename, a
    backend-specific fault - and until now there was no way to step off it.

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
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so AMD RX 6000 users need
# no separate HIP SDK. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1307/llama-b1307-windows-rocm-gfx103X-x64.zip"
)

# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "495323bfb522f2f5297a0786d8a2bec23f57421abdb01a1a07ff3b04d9ee7f0b"
)

# The lemonade-sdk release tag DEFAULT_URL points at. b1307 is built from
# llama.cpp 07132750825a (ggml 0.18.1), which carries the reordered
# llama_model_params and the 5-argument llama_sampler_init_penalties - see
# inference/backends/llamacpp/_structs.py and _abi.py, which bind BOTH that
# layout and the older lemonade b1288 one so an already-provisioned runtime
# keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones - the two
# schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1307"

# Maps hwdetect.amd_gfx_family()'s return value to the lemonade-sdk asset name
# fragment for the self-contained amd-rocm build of that family. Only the three
# families amd_gfx_family() can currently distinguish are listed; an
# unrecognised card (empty string) falls back to gfx103X, the only build
# verified on real hardware. _PINNED_FALLBACK_SHA256 also carries entries for
# gfx1150/gfx1151/gfx908/gfx90a, but there is no adapter-name heuristic for
# those families yet, so they stay unreachable until one exists.
_AMD_ROCM_ASSET_TAG = {"gfx103x": "gfx103X", "gfx110x": "gfx110X", "gfx120x": "gfx120X"}

# Upstream llama.cpp prebuilts (ggml-org/llama.cpp).
_UPSTREAM_REPO = "ggml-org/llama.cpp"

# THE BUILD localm INSTALLS. One constant, decided here, never computed while a
# user is running setup.
#
# This used to resolve upstream's newest release with uploaded assets at RUNTIME,
# on every install, released builds included. So a third party publishing a
# release could break every fresh and updating install with no localm change at
# all - which is not hypothetical: b10373 shipped a new default
# LLAMA_LOAD_MODE_AUTO, localm's own ABI gate refused it, and a fresh
# `setup-llama` was dead on arrival until localm was changed. Nothing about that
# release was broken. It was simply never tested against this code, because
# nothing tested anything before installing it.
#
# WHAT "CONFIRMED" MEANS HERE, and it is deliberately not "it loads": the build
# was downloaded, loaded through localm's real loader, and made to GENERATE
# TOKENS with a real model. "It loads so it will be fine" is what produced the
# outage above. scripts/confirm_llama_runtime.py is that check, and the per-
# backend record of what each pin actually rests on is _PIN_CONFIRMATION below.
#
# ONE CONSTANT, NOT ONE PER BACKEND, and the reason is structural rather than a
# simplification: upstream ships ONE llama.dll for every backend of a given tag
# (the backend lives in the separate ggml-* plugin libraries), so the struct
# layout this gate cares about cannot differ between them. Measured, not
# inferred - byte-identical across cpu/vulkan/cuda at b10361 (4a587d89) and at
# b10373 (f2021c86), and re-measured at each new pin by the confirm script
# rather than inherited. GENERATION is the part that IS backend-specific, which
# is exactly what _PIN_CONFIRMATION records.
#
# STAYING CLOSE TO UPSTREAM IS PART OF THE REQUIREMENT, not an afterthought: a
# pin nobody advances fails a user as surely as tracking latest does, just more
# slowly. scripts/check_llama_pin.py reports how far behind this constant has
# fallen (the same shape as the ComfyUI pin's own currency check), and
# `--tag latest` is the escape hatch for a user who needs an upstream fix today.
_PINNED_TAG = "b10375"

# WHAT THE PIN RESTS ON, PER BACKEND. Deliberately not a boolean and deliberately
# not a single "confirmed" flag: a confirmation job that is green because it
# SILENTLY SKIPPED the backends it could not test would be worse than no job,
# because it would carry the word "confirmed".
#
# The asymmetry is hardware, and it is not going away: a GitHub runner has no
# GPU, so CI can honestly generate on cpu only. vulkan is confirmed on the
# maintainer's box. cuda, sycl, hip and metal need hardware nobody here has.
#
# What every entry DOES rest on, including the untested ones, is the byte-
# identity above: the ABI/struct compatibility that broke in the incident this
# pin exists to prevent is carried by one shared llama library, so confirming it
# once confirms it for all of them. What an untested entry does NOT rest on is
# any evidence that THAT backend's ggml plugin produces tokens on that hardware.
# Say which of the two you have; never round the second up to the first.
_PIN_CONFIRMATION = {
    "cpu": "load + generate, measured (Windows x64; devices: CPU only, which is "
           "also the control proving the GPU column below is not vacuous)",
    "vulkan": "load + generate, measured (Windows x64, AMD RX 6900 XT / gfx1030; "
              "the runtime registered a Vulkan0 GPU device)",
    "cuda": "ABI only (shared llama library); generation NOT measured - no NVIDIA hardware",
    "sycl": "ABI only (shared llama library); generation NOT measured - no Intel GPU",
    "hip": "ABI only (shared llama library); generation NOT measured - needs a system ROCm toolkit",
    "metal": "ABI only (shared llama library); generation NOT measured - no Apple Silicon",
    # Not an upstream tag at all: the lemonade-sdk build, pinned separately by
    # _ROCM_TAG, so this table's subject (_PINNED_TAG) does not describe it and
    # nothing here was measured against it. Stated rather than omitted, because
    # an absent row reads as "covered like the others".
    "amd-rocm": "out of scope for _PINNED_TAG - pinned separately as _ROCM_TAG, "
                "whose generation was NOT measured by this pin's confirmation",
}

# Stored in the `llama_runtime_pin` config key to mean "track upstream's newest
# release", the opt-in to bleeding edge. A SENTINEL rather than an empty value
# because empty now means the shipped pin, which is the safe default; a user who
# wants upstream's newest has to say so, and it stays visible in their config.
#
# Kept OUT of pinned_tag()'s return value on purpose. Every existing caller of
# that function treats what it returns as an exact release tag and interpolates
# it into a URL path segment, so letting the sentinel through would produce a
# confident request for a release literally named "latest". tracks_latest() is
# the second accessor instead, and _tag_for() is the only place that consults
# both.
_TRACK_LATEST = "latest"

# The documented word for "go back to the build localm ships and confirmed".
# Needed because `--tag latest` no longer means that: before the pin existed,
# clearing the pin WAS how you got the default, and the default was upstream's
# newest. Those are now two different destinations and each needs its own word.
_TRACK_DEFAULT = "default"

# Third-party Linux CUDA prebuilt (see dev-notes/ADR-0010): upstream publishes
# no bare Linux CUDA binary itself (verified live against their releases), so
# this fetches from an actively-maintained third party instead - the same
# shape as the amd-rocm backend's own dependency on lemonade-sdk/
# llamacpp-rocm, just below. hybridgroup/llama-cpp-builder tracks upstream's
# bNNNNN tag numbering 1:1 and publishes the same asset-name convention
# upstream itself uses for every other Linux backend - verified live before
# wiring this in: a real asset downloaded, its ELF NEEDED list and glibc
# floor parsed (dev-notes/ADR-0010). Deliberately NOT a localm-built/-hosted
# binary - see the ADR for why self-building was considered and rejected. A
# public repo slug in a URL is not personal disclosure (AGENTS.md rule 2's
# own documented carve-out), same as every other GitHub URL in this file.
_CUDA_LINUX_REPO = "hybridgroup/llama-cpp-builder"

# Offline checksums for the assets of the pinned builds. Only consulted when the
# release API is unreachable or publishes no `digest` - the online path reads the
# digest straight off the asset listing.
#
# THE _PINNED_TAG ENTRIES ARE WHAT KEEP THE PIN SELF-CONTAINED. The API and the
# download CDN are different hosts with different failure modes, so "the release
# listing is unavailable but the download works" is a real state (an API rate
# limit is the common way in). Without a checksum for the pinned tag, that state
# would install the pin UNVERIFIED - a quiet downgrade of the integrity guarantee
# in exactly the situation the pin exists to be reliable in. So the pin and its
# digests move together: bump one, bump the other.
#
# THE TABLE HOLDS EXACTLY THE TAGS THIS FILE PINS, and a test enforces that. It
# used to also carry the whole asset list of b9870, the old dynamic-resolution
# fallback. Those entries went with _FALLBACK_TAG: nothing resolved to that tag
# any more, nobody had ever confirmed it loads or generates under the current
# binding, and an explicit `--tag b9870` is not special enough to deserve offline
# checksums that `--tag <any other release>` does not get. Keeping them would
# have been a stale pin reading as coverage, which is the exact thing
# test_every_pinned_asset_belongs_to_a_pinned_tag exists to catch.
#
# The values are the API's own `digest` fields. That is not taken on trust: the
# b1307 gfx103X asset was downloaded and hashed locally on 2026-08-05 and came
# out byte-identical to the digest GitHub reports, which is what makes the rest
# usable without pulling several GB.
_PINNED_FALLBACK_SHA256 = {
    # tag b10375 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release -
    # verified: byte-identical digests for 12.4 and 13.3 at b9870 and at b10375.
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
# name for (platform, backend). Substring matching (not exact names) keeps this
# robust to upstream version suffixes drifting (e.g. cuda-12.4, rocm-7.2).
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-x64"],
        "vulkan": ["bin-win-vulkan-x64"],
        # Keyed by CUDA LINE (NvidiaInfo.cuda_line), not a flat preference
        # list: a Blackwell-class GPU must never fall through to a 12.x asset
        # even as a "closest match" fallback, since that build's fatbin has no
        # kernels for it (see the _CUDA_LINE block for the full rationale).
        "cuda": {
            "cuda-12": ["bin-win-cuda-12.4-x64", "bin-win-cuda-12"],
            "cuda-13": ["bin-win-cuda-13.3-x64", "bin-win-cuda-13"],
        },
        "sycl":   ["bin-win-sycl-x64"],
        # Upstream renamed the Windows ROCm/HIP asset at b10356 (2026-08-11,
        # PR 25775 "Add CI targets for ROCm 7.14"): "bin-win-hip-radeon-x64"
        # -> "bin-win-rocm-<version>-x64" (e.g. bin-win-rocm-7.14-x64), the
        # same shape as the Linux asset's own versioned name. Without a fragment
        # matching the new name this backend silently guessed a URL for a
        # filename that no longer exists and 404'd - caught 2026-08-11 while
        # building the AMD ROCm-detection escalation
        # (dev-notes/BLACKWELL-FIELD-FIXES-fix_plan.md, U5).
        #
        # ORDER: newest naming first, mirroring the Linux entry's "specific, then
        # generic" shape below. The OLD name used to be first because the frozen
        # b9870 fallback tag's pinned assets were genuinely still named that way;
        # that tag and its entries are gone with _FALLBACK_TAG, so leading with a
        # name upstream no longer publishes would only ever mis-resolve. It stays
        # LAST so an explicit --tag on a pre-rename release still works.
        "hip":    ["bin-win-rocm-7.14-x64", "bin-win-rocm", "bin-win-hip-radeon-x64"],
    },
    "linux": {
        "cpu":    ["bin-ubuntu-x64"],
        "vulkan": ["bin-ubuntu-vulkan-x64"],
        "cuda":   ["bin-ubuntu-cuda"],
        "sycl":   ["bin-ubuntu-sycl-fp16-x64", "bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        # Same versioned-name drift as the Windows entry above: 7.2 at the old
        # fallback tag, 7.14 at the pinned one. Newest first, generic last.
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

# SEC-8: a prebuilt llama runtime archive is many megabytes. Anything below this
# floor is almost certainly an error page, a redirect stub, or a truncated
# transfer, never the real artifact. This is the always-on lower bound; the
# valid-archive structural check below is the second always-on guard. A pinned
# sha256 is the opt-in third guard (we do not hardcode a brittle hash for the
# live URLs, which move with every upstream release).
_MIN_ARTIFACT_BYTES = 256 * 1024   # 256 KiB

# Per-read socket timeout for the archive download. The chunked urlopen read loop
# honours the default socket timeout as an idle (between-reads) deadline, NOT a
# total-transfer cap, so a large-but-progressing download is never killed; only a
# genuinely stalled connection (no bytes for this many seconds) trips it. This
# turns an indefinite hang on a dropped/throttled transfer into a clear, loud
# error the caller reports, instead of a frozen progress line with no diagnostic.
_DOWNLOAD_STALL_TIMEOUT = 60   # seconds


@dataclass
class _DownloadResult:
    """What actually happened on the wire, captured for diagnosis - never
    guessed after the fact. ``content_length`` is 0 when the server sent none
    (completeness could not be checked structurally); ``final_url`` is the URL
    after following redirects (a proxy that redirects to its own block page
    shows up here even when the request otherwise looked normal)."""
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


# A tiny marker file recording WHICH backend currently occupies the runtime lib
# dir. It exists so the "already provisioned" guard can be backend-aware: a later
# `setup-llama --backend cuda` on a box that already has a vulkan/cpu build must
# still fetch CUDA (R23), instead of short-circuiting on the mere presence of a
# library. A dotfile, like the venv's .localm-venv marker; never loaded as code.
_BACKEND_MARKER = ".localm-backend"


def _record_provisioned_backend(target: Path, backend: str,
                                build: "Optional[str]" = None) -> None:
    """Record *backend* as the one now provisioned in *target*, optionally with
    the *build* tag it came from. Best-effort: the marker only optimises the
    guard, so a write failure is non-fatal (the guard then conservatively
    re-provisions an explicit pick rather than skipping it). Written AFTER
    provisioning because _clear_target wipes the dir's files.

    Format is ``<backend>`` or ``<backend> <build>``, whitespace-separated. The
    second token is OPTIONAL by design and is omitted whenever the tag is not
    known for free - see the call sites. A marker with no build reads back
    identically for the guard's purposes (see _provisioned_backend), so adding
    the tag needs no migration and no version detection."""
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

    THE FIRST WHITESPACE TOKEN, never the whole file. That is what makes the
    optional build tag safe to add: "amd-rocm" and "amd-rocm b1307" both answer
    "amd-rocm", so the provision guard's ``have == want`` comparison is byte for
    byte the decision it always made. A bare .strip() of the whole file would
    have returned "amd-rocm b1307", matched no backend name, and re-provisioned
    on EVERY invocation - which on a shared box is precisely the destructive
    path the runtime-in-use refusal exists to stop. Backward and forward
    compatible by construction rather than by a version check, which is what a
    file written by releases you cannot revise needs."""
    parts = _read_marker(target)
    return parts[0] if parts else None


def _provisioned_build(target: Path) -> "Optional[str]":
    """The build tag recorded alongside the backend, or None when the marker
    predates the two-token format or the tag was not knowable at provision time.

    ABSENCE IS NORMAL, never corruption: _record_provisioned_backend is
    best-effort and omits the tag whenever it is not free to obtain, so every
    reader must treat None as "not recorded" and say something honest rather
    than guess a version."""
    parts = _read_marker(target)
    return parts[1] if parts and len(parts) > 1 else None


def installed_backend() -> "Optional[str]":
    """The backend actually provisioned on this box right now, or None when
    nothing is provisioned yet (a fresh install, or one that predates the
    marker).

    Public, read-only convenience for callers OUTSIDE this module that need
    "what is installed" rather than "what would be recommended fresh" -
    updater.py's own backend-preservation (an update must never silently swap
    a user to a different backend just because the hardware-recommendation
    policy changed) and the Settings page's backend display. Resolves the
    target directory the same way the provisioning code does
    (_repo_runtime_lib), so it always reads the marker the real install
    actually wrote."""
    return _provisioned_backend(_repo_runtime_lib())


def installed_build() -> "Optional[str]":
    """The llama.cpp release tag actually provisioned on this box right now, or
    None when nothing is provisioned or the marker predates tag recording.

    Public, read-only, and deliberately the exact shape of installed_backend()
    above (PR #1236's pattern) rather than a second mechanism: doctor and the
    bug reporter need "which build is on disk", and inferring it from library
    filenames - which is what triage had to do for the b10360/b10361 question -
    is a guess, not a lookup.

    None is NORMAL and every caller must render it as "not recorded" rather than
    guessing a version. See _provisioned_build."""
    return _provisioned_build(_repo_runtime_lib())


# How many past provisions to remember. Rollback only ever needs the previous
# DISTINCT tag, but keeping a short run of them means a user who rolled back and
# then re-pinned can still see where they have been, and it bounds a config key
# that would otherwise grow without limit on a box that re-provisions often.
_RUNTIME_HISTORY_MAX = 20


# The complete set of values --backend accepts, and the ONE place that decides
# it. Public and module-level rather than inline in the click.Choice below,
# because a second surface now offers the same choice: the GUI's runtime route
# validates a caller-supplied backend against this, so a name the CLI accepts
# and the route rejects (or the reverse) is not representable. "auto" is a real
# member, not a sentinel - it is what a bare `setup-llama` resolves through
# _auto_backend, and it is the right default for a first provision.
BACKENDS: "tuple[str, ...]" = ("auto", "vulkan", "cuda", "sycl", "hip", "cpu",
                               "metal", "amd-rocm")


# A release tag is interpolated straight into a GitHub API path and a download
# URL, so it is validated as a PATH SEGMENT, not merely as "looks like a tag": a
# value carrying '/', '..', '?' or '#' would silently retarget the request at a
# different endpoint. Deliberately broader than upstream's own bNNNNN shape so a
# future tag scheme is not refused by a cosmetic rule - the check is about what
# is safe in a URL, which is the part that must never be relaxed.
_TAG_SAFE_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


# What a usable tag looks like, in one sentence, for whoever has to REFUSE one.
# Shared for the same reason is_safe_tag is: the CLI raises a ClickException and
# the GUI route raises a 400, and a user who reads one and then the other must
# not be told two different rules (diff-review-discipline.md item 28 - a fact
# stated in more than one place diverges exactly where it costs most).
TAG_HELP = ("Use a tag as upstream publishes it, for example 'b10355' (letters, "
            "digits, dot, dash and underscore only), or "
            f"{_TRACK_DEFAULT!r} for the build localm ships and confirmed, or "
            f"{_TRACK_LATEST!r} for upstream's newest.")


def is_safe_tag(tag: "Optional[str]") -> bool:
    """Whether *tag* is safe to interpolate into a release URL path segment.

    PUBLIC because it is no longer only this module's business: the GUI's
    runtime route accepts a caller-supplied tag and must refuse a bad one with
    a 400 BEFORE dispatching a job, and it has to refuse EXACTLY what the CLI
    refuses. One predicate, one answer, whichever surface asks - the same
    reason _validated_tag delegates here rather than carrying its own regex."""
    tag = (tag or "").strip()
    return bool(_TAG_SAFE_RE.match(tag)) and ".." not in tag


def tracks_latest() -> bool:
    """Whether this install has opted IN to upstream's newest release rather than
    the confirmed build localm ships (``setup-llama --tag latest``).

    Deliberately a SEPARATE accessor from pinned_tag() rather than a third
    possible return value from it. Every caller of pinned_tag() interpolates what
    it gets into a release URL path segment, so a sentinel leaking through there
    would produce a confident request for a release named "latest"; keeping the
    two apart makes that unrepresentable rather than merely unlikely.

    Never raises, same contract as pinned_tag(): an unreadable config degrades to
    "not tracking", which is the conservative answer - it means the shipped,
    confirmed pin."""
    try:
        raw = config.load_config().get("llama_runtime_pin") or ""
    except Exception:
        return False
    return str(raw).strip().lower() == _TRACK_LATEST


def pinned_tag() -> "Optional[str]":
    """The exact llama.cpp release tag the user has pinned, or None when they
    have not pinned one (the default, which installs _PINNED_TAG, and the
    ``--tag latest`` tracking mode, which is tracks_latest()'s business).

    VALIDATED ON READ, not only where --tag writes it. The CLI flag is not the
    only way this value can arrive: the key is HIDDEN with no coercion branch, so
    PATCH /v1/config stores whatever it is handed (owner-gated, but stored
    verbatim - see test_config_plugin_state_gate.py), and config.json is a plain
    file a user can edit by hand, which no route can police. Checking here covers
    every entry point at the one place the value is actually used, so a tag that
    would escape its URL path segment can never reach _release_assets. An unsafe
    stored value is treated as NO PIN and said out loud rather than silently
    obeyed or silently dropped.

    Never raises: a pin is read on the provisioning path, and an unreadable
    config must degrade to "no pin" rather than break setup entirely."""
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
    write failure: unlike recording history, a choice the user explicitly asked
    for must never silently fail to stick (a silently-unpinned install is exactly
    the surprise this whole area exists to remove)."""
    value = (tag or "").strip()
    config.update_config(lambda cfg: cfg.__setitem__("llama_runtime_pin", value))


def _record_runtime_history(backend: str, tag: "Optional[str]") -> None:
    """Append a successful provision to the rollback history. Best-effort by
    design - the provision itself already succeeded, so failing to journal it
    must not turn a working install into an error - but a failure is LOGGED
    rather than swallowed, because the visible symptom otherwise is a --rollback
    that cannot find a build the user knows they had.

    A repeat of the newest entry is collapsed rather than appended, so
    re-running setup-llama on the same build does not push the previous distinct
    tag out of a bounded list and quietly destroy the rollback target."""
    if not tag:
        # Nothing to roll back TO. A tagless provision (--from, --url, an
        # unrecorded backend) is a real event, but it cannot name a build, and
        # journalling it as an entry with no tag would let --rollback offer a
        # target it cannot install.
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
        # Visible, not only in the debug log: without this the failure shows up
        # much later as "no earlier build is recorded ... nothing to roll back
        # to", which collapses two different situations into one message - "you
        # have only ever had this build" and "we could not write down that you
        # had another one". Said here, at the moment it happens, the user can
        # act on it; said later by --rollback, it is indistinguishable from
        # normal. Still not fatal: the install itself succeeded.
        console.print(f"[yellow]Warning:[/yellow] installed {backend} {tag}, but "
                      f"could not record it for rollback ({e}). "
                      "[bold]localm setup-llama --rollback[/bold] will not offer "
                      "this build later.")


def runtime_history() -> list:
    """The recorded provisions, oldest first. Filtered to well-formed entries
    whose tag is SAFE, so neither a hand-edited config nor a verbatim PATCH can
    make --rollback offer a nonsense - or hostile - target.

    The safety filter matters here and not only in pinned_tag(): --rollback takes
    a tag from this list and pins it, so an unchecked entry would become a pin by
    a route that never passed through _validated_tag."""
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

    The comparison target is whatever ``setup-llama`` would install right now,
    which is the point - the two must never disagree, or the GUI offers an
    update to a build the command would not install. So: an exact PIN if one is
    set (an install that pinned away from a broken release must not be told a
    newer build is "available" - that newer build is exactly what it pinned away
    from), else upstream's newest when the user opted into tracking, else the
    shipped ``_PINNED_TAG``. ``amd-rocm`` compares against its fixed
    ``_ROCM_TAG``, since that build is never resolved from an upstream tag at all.

    ONLY THE TRACKING CASE MAKES A NETWORK CALL NOW. The default path answers
    from a constant, so the GUI's runtime-update card no longer reaches GitHub on
    every check. The offload in localm/plugins/gui/routes/runtime.py stays right
    regardless: a tracking install still needs it, and a route cannot know which
    kind of install it is serving.

    This target is a CANDIDATE, not a proof that it loads on THIS machine: that
    is only established by attempting the provision, which is what
    ``_provision_with_fallback`` does on every install path. This function only
    says whether the installed build differs from the candidate, and never
    re-provisions anything.

    Returns ``{installed, backend, current, target, newer, pinned, previous}``.
    ``installed`` is False when nothing has been provisioned yet (there is no
    "update" for a runtime that was never set up - that is initial setup, a
    different action). ``previous`` is ``--rollback``'s own target
    (``previous_tag(backend)``), included here so the same read-only check
    that powers the "update available" card can also decide whether a
    rollback affordance has anything to offer, without a second round trip.
    Never raises: an unreadable pin/marker degrades to the safe "nothing to
    report" shape rather than breaking the check (mirrors ``pinned_tag()``'s
    own never-raises contract)."""
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

    THE ONLY PLACE THAT DECIDES A TAG for the upstream-resolved backends, so a
    choice cannot be honoured on one code path and ignored on another. Note it is
    NOT consulted for amd-rocm, whose build comes from lemonade-sdk's own
    release numbering (_ROCM_TAG) - a different tag space entirely, in which an
    upstream bNNNNN means nothing. _pin_note_for_backend says so out loud rather
    than letting the pin look applied.

    THE DEFAULT BRANCH MAKES NO NETWORK CALL, and that is the substance of the
    change rather than a side benefit: while this function ended in a live
    release-listing lookup, what an install got was decided by whoever had
    published most recently, on every machine, for every released localm - so a
    build nobody here had ever run could arrive without a single localm change.
    A constant cannot do that."""
    pin = pinned_tag()
    if pin:
        return pin
    if tracks_latest():
        return _latest_tag()
    return _PINNED_TAG


def _pin_note_for_backend(backend: str) -> None:
    """Say plainly when a pin the user set does not apply to the backend being
    provisioned, instead of dropping it silently (AGENTS.md rule 5). Only
    amd-rocm is in that position today: its tag is lemonade-sdk's, not
    upstream's."""
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
        # Same reason, other choice: --tag latest is equally inapplicable here,
        # and saying nothing would let a user believe this install is tracking
        # upstream when this backend cannot.
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
    archives' ~49 command-line tools (llama-cli, llama-server, llama-bench,
    ggml-rpc-server, ...) were dead weight in every Windows install.

    Verified before removing them, rather than inferred from their names:
      * no subprocess call anywhere in localm reaches the runtime binary dir -
        the only executables it ever runs are nvidia-smi, sys.executable and
        uv/pip;
      * no documented workflow in docs/ or README tells a user to run one;
      * nothing in setup-llama or doctor invokes one (both isolate via a plain
        python subprocess).

    THE DECIDING EVIDENCE IS THE PLATFORM ASYMMETRY: the darwin and Linux
    branches below have ALWAYS matched libraries only (.dylib / .so), so those
    archives' extensionless `llama-cli` and friends were never copied and those
    installs have never carried a single bundled executable. Windows was the
    lone outlier. Dropping .exe makes it agree with the platforms that already
    demonstrate the product does not need them.

    Libraries are kept WHOLESALE and deliberately - a .dll may be an OS-resolved
    link dependency of ggml-hip/llama rather than something localm opens by name
    (amd_comgr, rocblas, hipblaslt, rocsolver, origami, rocm_kpack all are), so
    proving one unused would need a link-graph walk. Unproven means keep: a
    retained stray file costs disk, a removed dependency costs a broken install
    on hardware nobody here can test.

    Incidentally removes ggml-rpc-server.exe, which carries a critical
    unauthenticated-RCE advisory in its own component (CVE-2026-34159, fixed in
    the build we ship). localm never ran it, so this is not a vulnerability fix -
    but an unnecessary network daemon has no business in the install directory of
    an offline-first app. See dev-notes/SECURITY-llamacpp-parser-memory-safety-
    2026-08-05.md.
    """
    n = f.name.lower()
    if sys.platform == "win32":
        return n.endswith(".dll")
    if sys.platform == "darwin":
        return n.endswith(".dylib")
    return ".so" in n          # libfoo.so and libfoo.so.1


# rocBLAS and hipBLASLt (ROCm's vendor BLAS libraries) resolve their GPU-arch-
# specific GEMM kernels ("Tensile" library) at RUNTIME from a "<name>/library/"
# data directory sitting next to their DLL - the kernels are NOT linked into the
# DLL itself. That data is pure .dat/.hsaco/.co files, so _is_wanted() (by
# design: it must not copy the source tree's .py/.md/etc) never matches them,
# and _copy_binaries' flat `target / f.name` copy would lose their required
# subdirectory layout even if it did. The result: every ROCm/HIP provision
# (amd-rocm auto-detect, or --from/--url pointed at the identical archive)
# silently shipped rocblas.dll/hipblaslt.dll with NO kernel data at all. Nothing
# failed at provision time - ggml's own hand-written HIP kernels cover ordinary
# chat decode - so this went undetected until a workload that dispatches a GEMM
# through Tensile (the embedder's non-causal batch encode) hit it: rocBLAS
# fails to init its Tensile host and hard-crashes the native process outright
# (uncatchable from Python - the whole reason the embedder load/embed calls run
# in an isolated child, see inference/embedder.py). Confirmed 2026-07-17: the
# lemonade-sdk b1288 gfx103X archive DOES ship a complete rocblas/library/ (410
# files, including gfx1030 kernels) and hipblaslt/library/ - the data was always
# there, just always dropped on the way in.
#
# Re-measured 2026-08-05 on the lemonade b1307 gfx103X archive: rocblas/library/ is still
# there and now carries 142 gfx1030 kernel files (up from 88), while
# hipblaslt/library/ is GONE. That is not a regression to chase - lemonade b1288's
# hipblaslt data contained gfx1100 kernels ONLY, zero gfx1030, so it was never
# usable by the gfx103X target this archive is built for. Both names stay listed
# because the same code provisions the gfx110X/gfx120X archives too, and a
# missing directory is already a no-op here.
_BLAS_LIBRARY_DIRS = ("rocblas", "hipblaslt")

# Of those, the ones whose kernel data is genuinely REQUIRED by an install that
# ships the matching vendor library. Only rocblas, and the asymmetry is measured,
# not assumed:
#   * rocblas WITHOUT its Tensile data hard-crashes the native process on the
#     first GEMM dispatched through it (the embedder's batch encode) - the
#     uncatchable crash documented above.
#   * hipblaslt is present as a library and has NO kernel directory at all on the
#     b1307 gfx103X archive we ship, and that install is healthy (verified
#     2026-08-05: a real model generating and the embedder producing correct
#     cosines). Its b1288 data held gfx1100 kernels only, zero gfx1030, so this
#     target never had usable data to lose.
# So requiring a hipblaslt directory would fire on every healthy gfx103X install:
# a check that cries wolf on the normal case is worse than no check, because it
# trains people to ignore it. If a gfx110X/gfx120X user ever reports a hipblaslt
# Tensile crash, add it here with that evidence.
_BLAS_DIRS_REQUIRING_KERNELS = ("rocblas",)


def _has_vendor_library(target: Path, name: str) -> bool:
    """True when *target* holds the shared library for BLAS vendor *name*.

    Matches both naming conventions because the archives use both, on the same
    platform: the b1307 Windows build ships `rocblas.dll` AND `libhipblaslt.dll`.
    Covers `.so` version suffixes (librocblas.so.4) the same way _is_wanted does."""
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
    No platform test and no backend marker is consulted - the marker is written
    last during a provision, so a half-finished install can be missing it exactly
    when this check matters most.

    Scope, deliberately: this catches "the library is installed but its runtime
    kernel data is not", which is the SILENT failure - provisioning succeeds, chat
    works, and the crash arrives later on the first Tensile GEMM. It does not try
    to catch a missing library, because that one already fails loudly at load."""
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
        # Cannot read the install: say so rather than returning "no problems",
        # which is what an unreadable directory would otherwise look like.
        problems.append(f"could not inspect BLAS kernel data: {e}")
    return problems


def _copy_blas_library_dirs(src_dir: Path, target: Path) -> int:
    """Copy any of ``_BLAS_LIBRARY_DIRS`` found under *src_dir* into *target*,
    preserving their internal directory structure (unlike _copy_binaries' flat
    DLL copy - rocBLAS/hipBLASLt resolve this data by RELATIVE PATH, not by
    file name, so flattening it would be as useless as dropping it). Searches
    one level of nesting too, in case an archive wraps its contents in a single
    top-level folder. Returns the number of files copied."""
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
    """Pick the broadest WORKING backend for this machine - via the SAME policy the
    installers use (``hwdetect.recommended_install_backend``), so bare
    ``setup-llama`` and setup.bat / setup.sh can never drift:

      NVIDIA, any OS -> cuda (self-contained build + runtime fetch on both
      Windows and Linux, peak performance); AMD on Windows (RX 6000 / unknown)
      -> the self-contained ROCm build; AMD elsewhere with a system ROCm/HIP
      toolkit detected -> hip; Apple Silicon -> metal; every other GPU (Intel,
      AMD with no toolkit detected) -> vulkan; no GPU -> cpu."""
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception as e:
        # Surface the skipped GPU setup so a detection failure is visible and the
        # user knows how to force a GPU backend, rather than a silent CPU default.
        console.print(f"[yellow]GPU detection failed ({e}); defaulting to CPU - "
                      "override with --backend.[/yellow]")
        return "cpu"
    return hwdetect.recommended_install_backend(det)


def _latest_tag() -> str:
    """The newest ggml-org/llama.cpp release tag that actually has its build
    assets uploaded, or _PINNED_TAG if no such release can be found (offline,
    rate-limited, etc.).

    ONLY REACHED WHEN THE USER OPTED IN with ``--tag latest``. It is no longer
    the default: see _tag_for. What it returns is by construction a build nobody
    here has run, which is the trade that flag exists to offer.

    The unavailable case falls back to _PINNED_TAG and not to some older release,
    because the alternative is handing a user a build that is BOTH unconfirmed
    and not what they asked for. The confirmed pin is the one thing we can offer
    honestly when the lookup fails, and the message below says which they got.

    Upstream publishes a release (tag + notes) as soon as it is cut, then its CI
    matrix uploads the ~25 platform archives afterwards - which can take a while.
    Right after publish, ``/releases/latest`` can point at a tag whose ``assets``
    array is still genuinely empty even though the release body already lists
    the (soon-to-exist) download URLs. Resolving to that tag anyway used to
    produce a confident-looking match that 404s, because the linked file simply
    is not there yet. So we scan recent releases newest-first and use the first
    one that already has assets, skipping any still-uploading release."""
    tags = _recent_tags()
    if tags:
        return tags[0]
    # Surface it: the user asked to track upstream and is not getting upstream's
    # newest, which they would otherwise discover much later as "localm installed
    # an old build". Name what they got and why.
    console.print(f"[yellow]Could not find a ggml-org/llama.cpp release with "
                  f"uploaded assets (the release lookup was unreachable, or the "
                  f"newest releases have not finished uploading). Installing "
                  f"localm's confirmed build {_PINNED_TAG} instead - rerun later "
                  "for upstream's newest.[/yellow]")
    return _PINNED_TAG


def _recent_tags(limit: int = 10) -> list:
    """Upstream release tags that already have their build assets uploaded,
    NEWEST FIRST. Empty when the lookup is unavailable.

    Split out of _latest_tag rather than added beside it: that function ALREADY
    fetched ten releases and returned only the first, discarding the rest. The
    tag walk-back needs those discarded entries, so exposing them costs ZERO
    extra requests - the same shape as the tag that was already resolved and
    thrown away in the record/pin unit. One list, one call, one skip rule, so
    "which releases are candidates" cannot be answered two different ways."""
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
            # The asset check is the whole point of scanning rather than taking
            # /releases/latest: a release is published before its CI uploads the
            # ~25 archives, so a tag with an empty assets array 404s on download.
            if isinstance(tag, str) and tag and rel.get("assets"):
                out.append(tag)
    except Exception as e:
        # Best-effort like its two siblings (_release_assets, _pypi_wheel_url_
        # and_sha), and logged like them: every caller has a pinned fallback, so
        # this must not raise, but "the lookup was unavailable" must stay
        # discoverable (rule 5). It was the ONE verified_urlopen call site that
        # swallowed with no trace at all, which meant a refused downgrade
        # redirect here would have been the only one nothing recorded.
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
    matcher list. Defaults to _CUDA_LINE (None resolved below, not bound as a
    literal default - _CUDA_LINE is defined later in this module, after this
    function).

    *tag* lets a caller that has ALREADY resolved one (the Windows CUDA branch,
    which needs it to pair the build with its cudart bundle) pass it in rather
    than resolve a second time. When omitted this resolves its own through
    _tag_for - pin, else upstream's newest.

    THE THIRD ELEMENT IS WHY THIS RETURNS A TRIPLE: the tag is what the marker
    has to record, and this is the function that decides it. Returning it is
    what lets the caller record the installed build with NO extra lookup. The
    two alternatives are both worse - re-resolving in the caller doubles the
    network call this whole design exists to avoid, and parsing the tag back
    out of the returned URL is a second derivation of a value already in hand
    that would diverge on the templated-guess path below, where the URL is
    CONSTRUCTED rather than read from the release listing.

    It is None for amd-rocm, whose build comes from lemonade-sdk's own release
    numbering (_ROCM_TAG) rather than an upstream tag; the caller supplies that
    constant itself."""
    cuda_line = cuda_line or _CUDA_LINE
    if backend == "amd-rocm":
        if sys.platform != "win32":
            raise click.ClickException(
                "the self-contained 'amd-rocm' build is Windows-only; on Linux "
                "use --backend hip (needs ROCm) or build with --from.")
        # Try to resolve dynamically first
        tag = _ROCM_TAG
        try:
            from localm import hwdetect
            fam = hwdetect.amd_gfx_family(hwdetect.detect().gpu_names)
        except Exception as e:
            # Best-effort like _auto_backend's own detection call: a hiccup
            # here must not block an explicit --backend amd-rocm request. ""
            # is the same conservative default amd_gfx_family() itself returns
            # for an unrecognised card, and it resolves to gfx103X below.
            logger.debug("AMD gfx-family detection failed (%s); defaulting to gfx103X", e)
            fam = ""
        asset_tag = _AMD_ROCM_ASSET_TAG.get(fam, "gfx103X")
        if asset_tag == "gfx103X":
            fallback_url, fallback_sha = DEFAULT_URL, DEFAULT_URL_SHA256
        else:
            asset_name = f"llama-{tag}-windows-rocm-{asset_tag}-x64.zip"
            fallback_url = ("https://github.com/lemonade-sdk/llamacpp-rocm/releases/"
                            f"download/{tag}/{asset_name}")
            fallback_sha = _PINNED_FALLBACK_SHA256.get(asset_name)
        assets = _release_assets(tag, repo="lemonade-sdk/llamacpp-rocm")
        for a in assets:
            if f"windows-rocm-{asset_tag}" in a.get("name", ""):
                url = a.get("browser_download_url") or fallback_url
                digest = a.get("digest")
                sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
                if not sha:
                    sha = fallback_sha
                return url, sha, None
        # Surface the fallback so the user knows the build may not be current
        # (the lemonade-sdk release lookup was unreachable, or this release is
        # missing the expected asset for this GPU family); mirrors the
        # visible-fallback warning in the general (non-ROCm) path below
        # instead of silently handing back a possibly-stale pinned URL/checksum.
        console.print("[yellow]Could not find a lemonade-sdk/llamacpp-rocm release asset "
                      f"for {tag} ({asset_tag}); using pinned amd-rocm build - rerun later "
                      "for the latest.[/yellow]")
        return fallback_url, fallback_sha, None

    if backend == "cuda" and _platform_key() == "linux":
        # Upstream (ggml-org/llama.cpp) publishes no bare Linux CUDA binary at
        # all - verified live against their releases, see dev-notes/ADR-0010 -
        # so the generic _ASSET_MATCH path below would only ever construct a
        # guessed URL that 404s. Resolve against hybridgroup/llama-cpp-builder
        # instead, the same shape as the amd-rocm -> lemonade-sdk branch just
        # above: they track upstream's own tag numbering 1:1 and publish
        # upstream's own asset-name convention, so the same tag every other
        # Linux backend uses applies here too.
        #
        # cuda_line-aware, like the win32 cuda branch below: hybridgroup
        # publishes both a cuda-12 asset ("...-cuda-x64.tar.gz") and a
        # cuda-13 one ("...-cuda-13-x64.tar.gz") - verified live, real bytes
        # downloaded for the cuda-12 one (dev-notes/ADR-0010).
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
        # Genuinely unresolvable (hybridgroup has not built that exact
        # upstream tag yet - a real, occasionally-expected lag for a third
        # party, not an error to paper over): raise click.ClickException,
        # which _provision_with_fallback's caller already catches and turns
        # into the same offer/force-vulkan-fallback path every other
        # provisioning failure in this file uses - reusing that existing,
        # already-tested mechanism rather than inventing a new one. Never
        # construct a guessed URL here the way the generic path below does;
        # a guessed URL against a third party's repo is even less trustworthy
        # than one against upstream itself.
        raise click.ClickException(
            f"no Linux CUDA build found for llama.cpp tag {tag!r} on "
            f"{_CUDA_LINUX_REPO} - falling back to vulkan.")

    plat = _platform_key()
    entry = _ASSET_MATCH.get(plat, {}).get(backend)
    # The 'cuda' entry on win32 is keyed by cuda_line (a dict), not a flat
    # list, since the right asset depends on the GPU's architecture, not just
    # the platform - see _ASSET_MATCH's comment.
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

    # Fallback: the release listing was unavailable, so the asset name has to
    # come from somewhere else.
    #
    # PREFER A REAL NAME WE ALREADY KNOW over a constructed one. For a tag this
    # file pins, _PINNED_FALLBACK_SHA256 IS that release's asset list, so the
    # exact filename is in hand and needs no guessing. Matchers are tried in
    # their declared order, because that order encodes a preference the names
    # alone do not: linux sycl lists the fp16 build first and a bare
    # "bin-ubuntu-sycl" would also match fp32.
    #
    # This is not a tidy-up. The template below builds `llama-<tag>-<matcher>`,
    # which silently stops matching whenever upstream renames an asset - and it
    # had: b10356 renamed the Windows ROCm asset from `bin-win-hip-radeon-x64` to
    # `bin-win-rocm-<version>-x64`, and the Linux ROCm asset at the pinned tag is
    # `bin-ubuntu-rocm-7.14-x64.ZIP`, an extension the template cannot even
    # express (it assumes tar.gz off win32). Both produced a confident 404 offline.
    # Reading the name from the table fixes every such rename at once, and keeps
    # fixing them: bump the pin and its digests, and this follows.
    fname = ""
    for m in matchers:
        hits = sorted(n for n in _PINNED_FALLBACK_SHA256
                      if n.startswith(f"llama-{tag}-") and m in n.lower()
                      and "cudart" not in n)
        if hits:
            fname = hits[0]
            break
    if not fname:
        # An unpinned tag (--tag <something>), so there is nothing to read and a
        # constructed name is the only option left. Still worth attempting: it is
        # right whenever upstream's naming has not drifted.
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
    docstring); no production code calls this function (main() resolves via
    _provision_backend -> _resolve_backend_asset directly), but it is kept
    line-aware so it cannot silently drift back to a hardcoded cuda-12 default
    if something starts calling it again.
    Raises ``click.ClickException`` if the backend is not available here."""
    url, _sha, _tag = _resolve_backend_asset(backend, cuda_line)
    return url


# --------------------------------------------------------------------------- #
#  Download / validate / extract                                              #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> _DownloadResult:
    """Stream *url* to *dest*, capturing what actually happened on the wire (not
    just whether it succeeded) so a caller can diagnose a bad result from real
    evidence instead of a guess. Distinguishes three distinct failure shapes,
    each reported with its own specific cause:

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
        # verified_urlopen (see localm/http_ssl.py) follows the GitHub -> release-CDN
        # 302 and verifies both hops. "Over HTTPS" is ENFORCED, not assumed: its
        # default HttpsOnlyRedirect refuses a redirect off https and raises
        # RedirectDowngradeRefused, handled below. Until that guard existed this
        # comment asserted a property nothing checked, on the one download whose
        # bytes become a loaded native DLL - and _validate_archive's digest check
        # is opt-in (its expected_sha256, i.e. --sha256), so an unpinned archive
        # has no cryptographic check on its content either. Stream in chunks so a
        # multi-hundred-MB archive is never held in memory; the default socket
        # timeout is the between-reads stall deadline (not a total cap).
        req = urllib.request.Request(url, headers={"User-Agent": "localm-setup-llama"})
        with verified_urlopen(req, timeout=_DOWNLOAD_STALL_TIMEOUT) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            content_type = r.headers.get("Content-Type") or ""
            # geturl() is standard on every real urllib response, but stay
            # defensive for the rare test double that does not implement it -
            # the final URL is a nice-to-have diagnostic, not load-bearing.
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
        # BEFORE the OSError clause below, which it would otherwise hit
        # (RedirectDowngradeRefused is a URLError, and URLError is an OSError):
        # that clause tells the user this "looks like a dropped or flaky
        # connection" and to retry. Retrying an attempt to hand us a native DLL
        # over cleartext is the opposite of the right advice, and collapsing the
        # two into one message is exactly the failure rule 5 forbids.
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
        # proxy dropping the connection outright) - distinct from a download
        # that completes normally but turns out short (that is not an
        # exception at all; see the docstring). Report the partial state
        # honestly instead of a generic "download failed".
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
    """Classify what the file's own bytes actually look like, independent of
    what it was supposed to be - the one honest way to tell a substituted
    HTML/JSON response apart from a genuinely truncated archive. The content
    itself is authoritative here: a header or URL can be wrong, spoofed, or
    just uninformative, but a real llama.cpp archive's opening bytes never
    decode as text.

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
    # Structural markers are pure ASCII and sit at/near the very start of a
    # real error/block page regardless of the page's OVERALL encoding, so look
    # for them with a lossy decode first (never raises, so it still finds
    # HTML/JSON/XML served as e.g. windows-1252, not just UTF-8). A strict
    # decode is only needed for the weaker 'text vs binary' distinction below.
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
    evidence does not actually support: AGENTS.md rule 5 - a confident wrong
    guess is worse than an honest 'not clear', so the fallback case says so
    plainly instead of picking the most likely-sounding story."""
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

    *dl*, when given (the :func:`_download` result for this same file), lets
    checks 1 and 2 explain WHY from real evidence - what the bytes actually
    look like, plus what the response claimed - instead of a generic hedge
    that names every possible cause without saying which one actually
    happened (see :func:`_diagnose_bad_artifact`).
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
    # rocBLAS/hipBLASLt Tensile kernel data (see _BLAS_LIBRARY_DIRS) - a no-op
    # on every non-ROCm backend, since src_dir then has no rocblas/hipblaslt dir.
    n += _copy_blas_library_dirs(src_dir, target)
    # MIT requires the license to accompany the binaries; capture it (or a
    # bundled fallback) alongside them whenever we actually placed binaries.
    if n:
        _copy_license_files(src_dir, target)
    return n


def _install_runtime_wheel(pkg_dir: Path) -> bool:
    """Install the runtime wheel editable into the active venv. Tries uv, then
    pip. Returns True on success.

    ``env`` pins uv's AND pip's caches inside the data dir (rule 4: self-contained),
    same as the plugin-extra installer (plugins/deps.py). An editable install of a
    local dir is a smaller leak than a full torch download, but build isolation still
    pulls the build backend (setuptools/wheel) into the tool's cache, and either tool
    would otherwise put it in a per-user location OUTSIDE the data dir. See
    ``config.contained_pip_env``."""
    env = config.contained_pip_env()
    last_err = ""
    for cmd in (["uv", "pip", "install", "-e", str(pkg_dir)],
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
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

# Which upstream CUDA asset line to fetch is a function of the GPU's
# ARCHITECTURE, not just the platform: upstream ships both a 12.x line (broad
# compatibility - runs on any driver new enough for CUDA 12.4) and a 13.x line
# (needed for Blackwell-class GPUs, but itself needing a newer driver and
# dropping some pre-Turing arch support). NVIDIA Blackwell (datacenter sm_100,
# consumer/workstation sm_120 - e.g. RTX 50-series) is not supported by our
# pinned 12.4 build's fatbin: upstream only added Blackwell kernels starting
# CUDA 12.8, and our two pinned lines are 12.4 and 13.3. So a Blackwell card
# needs the 13.x line; every older architecture stays on 12.x, the
# broad-compatibility default. _CUDA_LINE is the fallback used when no
# architecture information is available at all (see NvidiaInfo.cuda_line).
_CUDA_LINE = "cuda-12"

# Compute-capability floor for "needs the 13.x line" (nvidia-smi's
# ``compute_cap`` query, e.g. "8.9", "12.0" - the GPU's sm/arch level, NOT the
# driver's max CUDA version). Per NVIDIA's published architecture numbers,
# Blackwell datacenter parts (B100/B200/GB100) report compute capability 10.0
# (sm_100) and Blackwell consumer/workstation parts (RTX 5090/5080/5070
# Ti/5070/5060) report 12.0 (sm_120); CUDA 12.8 was the first toolkit release
# to add Blackwell kernels. This is NOT verified against real Blackwell
# hardware (none is available here) - only the offline selection logic below
# is. >= 10.0 catches both variants and any later architecture
# without a new special case each time.
_BLACKWELL_MIN_CAP = (10, 0)

# Minimum driver-reported CUDA version ("cuda_capability") to trust each
# line's build, keyed by the line itself since a newer line needs a newer
# driver. Both match the PINNED asset's own X.Y (not just its major version):
# the 12.4 entry is the original, long-verified threshold; the 13.3 entry
# mirrors that same convention rather than relying on CUDA's minor-version-
# compatibility guarantee across the (very recent) 13.x series, which there is
# no Blackwell hardware here to confirm against directly. Being exact-match
# here is the conservative side of that unknown: it can only route a
# borderline driver to the safe Vulkan fallback, never hand it a build that
# fails to load.
_MIN_DRIVER_CUDA = {
    "cuda-12": (12, 4),
    "cuda-13": (13, 3),
}


def _ver_tuple(v: str) -> Optional[tuple]:
    # Return None (not (0,0)) on an unparseable version so an unmeasurable
    # capability reads as "unknown", never as a too-old driver we falsely block.
    try:
        return tuple(int(x) for x in str(v).split(".")[:2])
    except Exception:
        return None


def _ver_at_least(parsed: tuple, minimum: tuple) -> bool:
    """*parsed* >= *minimum*, treating a bare-major version (no minor
    component, e.g. "10" -> (10,)) as ".0". Plain tuple comparison would
    otherwise get this wrong: Python considers a tuple that is a strict
    PREFIX of another to be the smaller one regardless of the missing
    component's value, so (10,) >= (10, 0) is False even though 10 == 10.
    _ver_tuple's own contract (a bare major parses to a 1-element tuple, not
    padded) is intentional and unchanged - this is where the padding belongs,
    at the comparison, not the parse."""
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
        capability stays on cuda-12: an unmeasured architecture is not
        evidence it needs the newer, narrower-compatibility line (same
        "unknown != too old" reasoning as driver_ok below)."""
        cap = _ver_tuple(self.compute_capability)
        if cap is not None and _ver_at_least(cap, _BLACKWELL_MIN_CAP):
            return "cuda-13"
        return "cuda-12"

    @property
    def driver_ok(self) -> bool:
        """True when the driver is new enough for the CUDA line THIS GPU's
        architecture needs (see cuda_line) - the minimum is not a single fixed
        threshold, since Blackwell and older cards need different lines.
        Unknown driver capability is treated as OK (do not block on a parse
        miss)."""
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
    """Detect the NVIDIA GPU + driver, the max CUDA version the DRIVER
    supports, and the GPU's own compute capability (its architecture - e.g.
    "12.0" for Blackwell/sm_120). These are two different questions: the
    driver's version says what it CAN run; the compute capability says what
    the CARD IS, which is what decides whether the 12.x build's fatbin even
    has kernels for it (see NvidiaInfo.cuda_line).

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
        with verified_urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("assets", [])
    except Exception as e:
        # Best-effort probe: every caller has a pinned fallback for exactly
        # this case (offline, rate-limited, API down), so this must not raise -
        # but the cause should stay discoverable (rule 5) instead of vanishing.
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
    llama.dll" (NEW-CUDADLL)."""
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
        _validate_archive(arc, expected_sha256=sha256, dl=dl)   # SEC-8 gate, pre-extract
        ex = Path(tmp) / "x"
        _extract_archive(arc, ex)
        return _copy_binaries(ex, target)


# Tracked git sentinel files living in the runtime lib dir (see
# runtime/localm_llama_runtime/lib/.gitignore, which keeps the downloaded
# native binaries out of version control). Unlike _BACKEND_MARKER, which
# _clear_target deliberately wipes and _record_provisioned_backend rewrites
# after every provision, these two are never regenerated by setup-llama:
# deleting them empties the .gitignore, so a later `git add -A`/`git add .`
# touching this directory would stage the freshly-downloaded DLLs straight
# into git - the exact thing the file exists to prevent (AGENTS.md rule 1).
_PRESERVED_TARGET_FILES = (".gitignore", ".gitkeep")


class RuntimeInUseError(Exception):
    """Something has the installed runtime open, so it cannot be replaced.

    Deliberately NOT an ArtifactError and not a load failure. Those two mean a
    build is bad; this one means both builds are fine and a process is merely
    holding the files. That difference decides the response: a build that will
    not load earns the Vulkan fallback, this earns "close it and retry" with the
    existing install left completely intact."""

    def __init__(self, locked: "list[Path]", partial: bool = False):
        self.locked = list(locked)
        # True only when files were already deleted before the lock was hit (the
        # probe-to-unlink race). The install is then half cleared, and saying
        # "nothing was changed" would be a lie - so the two cases are tracked
        # apart and reported apart.
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
    it asks the OS the exact question deletion asks. MEASURED on this codebase's
    own runtime: a DLL reports WRITABLE before ``ctypes.CDLL`` and PermissionError
    errno 13 after, which is the IDENTICAL error ``unlink()`` raises on the same
    handle - so the probe predicts the deletion rather than merely correlating
    with it.

    Naturally platform-correct with no platform test. Windows maps a loaded DLL
    without FILE_SHARE_WRITE/DELETE, so the probe refuses exactly when deletion
    would. POSIX has no mandatory locking and unlinking an open file SUCCEEDS
    (the directory entry goes, the inode lives until the last close), so there is
    no half-state to prevent there and the probe correctly finds nothing.

    An unprobeable file counts as NOT in use. This gate exists to prevent a
    destructive half-state, so an inconclusive answer must not become a new way
    to block a legitimate install."""
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
    every caller must treat a non-empty result as a failed provision. This
    function used to swallow every OSError and return None, so a locked file left
    it silently reporting success on a directory it had only half cleared - the
    caller then copied a new build over the survivors and produced exactly the
    mixed-build state this docstring promises to prevent (AGENTS.md rule 5: a
    step that fails must never report success)."""
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

    Checking first is what makes the half-state unreachable rather than merely
    reported. Reporting alone (returning what could not be removed) is a real
    improvement over silence, but by the time it can report, it has already
    deleted everything it could - an honest error plus a runtime missing half its
    files. Probing first turns that into "could not update, something is using
    the runtime" with nothing lost.

    The post-clear check is not redundant: a process can open a file in the
    window between the probe and the unlink. That race leaves a half-state, which
    is why it raises the same error rather than continuing - it cannot be
    prevented here, but it must never be silent."""
    in_use = _files_in_use(target)
    if in_use:
        raise RuntimeInUseError(in_use, partial=False)
    left = _clear_target(target)
    if left:
        raise RuntimeInUseError(left, partial=True)


def _exit_runtime_in_use(e: RuntimeInUseError) -> None:
    """Report a refused provision and exit non-zero. Never falls back to another
    backend: the user's chosen backend is not the problem, so silently installing
    a different one would be exactly the never-override-the-user's-choice
    mistake, dressed as a recovery."""
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
# Provisioning CLEARS then REFILLS `target` (_clear_target_or_refuse + a
# download/copy). Until the GUI grew its own standalone runtime-update button,
# this ran from effectively one caller at a time: a user's own CLI invocation,
# or `localm update`'s runtime-class post-swap step (itself serialized against
# OTHER `localm update` calls by updater.py's own _apply_lock, but that lock
# knows nothing about a bare `setup-llama` run in a terminal or a second,
# independent caller like the GUI route below). Two provisions racing on the
# SAME directory is not merely slow, it is a corrupted install: both clear and
# refill it, so one process's half-written file can be read - or deleted - by
# the other's _clear_target/_copy_binaries. Same hazard, same fix shape, as
# managed_comfy_update.py's update lock (diff-review-discipline.md item 26):
# do NOT copy managed_comfy._remove_lock (a threading.Lock) here either - the
# GUI route spawns `setup-llama` as a CHILD PROCESS, so the contenders are
# separate interpreters and a threading.Lock would guard nothing. mkdir is
# atomic; stat-then-create is not.
_PROVISION_LOCK_OWNER = "owner.json"


class ProvisioningBusyError(Exception):
    """Another process already holds the provisioning lock."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _provision_lock_path(target: Path) -> Path:
    """Where the provisioning lock lives: a SIBLING of the runtime lib dir,
    never inside it, so this command's own directory clear can never disturb
    the lock protecting it (same reasoning as managed_comfy_update.py's
    _update_lock_path)."""
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
    for minutes (a download over a slow link), and a caller blocked that long
    is indistinguishable from a hang - a GUI button would spin forever with no
    way to tell the user why.

    Staleness is judged by PID LIVENESS, never elapsed time, for the same
    reason managed_comfy_update.py's lock is: the operation is unbounded, so
    any fixed timeout would eventually reclaim a LIVE holder's lock and
    recreate the exact race this exists to prevent. ``pid_alive`` is
    conservative - when it genuinely cannot tell, it returns True - so an
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
                # atomic create - never assume the retry wins, another caller
                # may have taken it in the meantime.
                with contextlib.suppress(OSError):
                    shutil.rmtree(str(lock))
                if attempt == 1:
                    continue
                raise ProvisioningBusyError(
                    "Another setup-llama run is already provisioning the "
                    "runtime. Wait for it to finish, then try again.")
            if pid is None:
                # Cannot tell who holds it: do NOT steal (that is how two
                # provisions end up interleaved). Say how to clear it by hand.
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
    _exit_runtime_in_use: the existing install is left completely untouched
    (the lock is taken before anything is cleared), so this is honest, not
    alarming."""
    console.print(f"[red]Cannot provision the runtime right now:[/red] {e.reason}")
    sys.exit(1)


def _fetch_verified(url: str, target: Path, sha: Optional[str], what: str = "release asset") -> None:
    """Fetch + place an archive, WARNING honestly when no checksum is available
    to verify it. A security step that silently does not run is a hidden problem
    (AGENTS.md rule 5): a GitHub asset can publish no `digest`, and the offline
    hash table only covers the tags this file pins, so the provenance check would
    otherwise be skipped in complete silence - while the CHANGELOG tells the user
    downloads are checksum-verified by default (AUDIT-MED-19). Size +
    archive-shape checks still apply either way.

    Which path is exposed changed with the pin: _PINNED_TAG's own assets ARE in
    that table, so the DEFAULT install stays verified even when the release
    listing is unavailable. It is the `--tag latest` and arbitrary `--tag <x>`
    paths that can land here with nothing to check against."""
    if not sha:
        console.print(
            f"[yellow]Warning: this {what} publishes no checksum, so the download's "
            "integrity is not cryptographically verified (its size and archive "
            "shape are still checked). Pass --sha256 <hex> to pin one.[/yellow]")
    _fetch_and_place(url, target, sha)


# NVIDIA publishes its own CUDA runtime libraries (cudart, cuBLAS) as plain
# PyPI wheels - the SAME channel localm's own HF/torch backend already
# depends on for the identical libraries (recommended_torch_variant's cu126
# index, hwdetect.py). NVIDIA renamed the CUDA-13-line packages to be
# UNSUFFIXED (nvidia-cublas-cu13 etc. are now deprecated stubs pointing at
# bare nvidia-cublas) - both lines are listed explicitly so neither naming
# scheme is guessed at.
#
# NCCL deliberately NOT included: the actual binary this fetches alongside
# (hybridgroup/llama-cpp-builder's Linux CUDA build, see _CUDA_LINUX_REPO)
# was checked directly - every one of its 26 shared libraries' raw bytes
# grepped for "libnccl", none found - so it does not link against it at all.
# An earlier prototype (extracting from ggml-org's own CUDA Docker image,
# see dev-notes/ADR-0010) DID need it; that image is not what ships here, so
# do not re-add nccl from that earlier finding without re-checking the
# binary actually in use at the time.
_CUDA_RUNTIME_PYPI_PACKAGES = {
    "cuda-12": ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
    "cuda-13": ("nvidia-cuda-runtime", "nvidia-cublas"),
}


def _pypi_wheel_url_and_sha(package: str) -> tuple:
    """The (url, sha256) of *package*'s latest Linux x86_64 wheel from PyPI's
    JSON API, or (None, None) if unavailable. Never raises (mirrors
    _release_assets' contract: a best-effort lookup whose caller always has a
    fallback path, so a network hiccup here must not crash setup)."""
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

    NEVER reads from any OTHER environment already on the user's machine
    (AGENTS.md rule 4: self-contained, no sibling folder on disk) - this
    always fetches and places a PRIVATE copy into *target*, exactly like the
    Windows cudart bundle is never "detected" on the user's system, only ever
    fetched fresh. An earlier draft of this design considered scanning an
    existing venv for an already-usable runtime; that was rejected for
    exactly this reason before any code was written."""
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
    ArtifactError (from the first failing package) on any failure - a
    partially-assembled CUDA runtime is worse than none, so this does not
    swallow a single package's failure and continue with the rest."""
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
    has as _ROCM_TAG).

    Every branch already resolved a tag and threw it away, which is the whole
    reason the marker could only ever record a version for amd-rocm. Each now
    keeps it, so recording the build costs NO extra network call - and the
    lookup stays exactly where it always was rather than being hoisted to the
    top of this function, which would make it run for backends that never
    needed it and would move it out from behind _resolve_backend_asset, the
    seam every caller and test already isolates.

    *tag* pins this ONE provision to a specific upstream release, overriding
    the pin/newest resolution. It exists for the ABI walk-back, which retries
    the SAME backend against an older release; it deliberately does NOT touch
    the stored pin, so a walk-back is a recovery rather than a silent change
    to what the user asked for. Ignored for amd-rocm, which has no upstream
    tag."""
    if chosen == "cuda" and with_cudart and sys.platform == "win32":
        # Resolved here, not in _resolve_backend_asset, because this branch
        # needs the tag to PAIR the build with its matching cudart bundle - a
        # cudart from a different release is exactly the mismatch this pairing
        # exists to prevent - and only reaches _resolve_backend_asset in the
        # no-assets fallback below, to which it then hands the same tag.
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
        return tag
    if chosen == "cuda" and with_cudart and sys.platform not in ("win32", "darwin"):
        # sys.platform, not _platform_key(): matches this function's OWN
        # existing style two lines up (the win32 cudart branch), rather than
        # mixing the two equivalent-in-production-but-differently-mockable
        # spellings within one function - caught by a test that mocked
        # _platform_key alone and still hit the win32 branch on a real
        # Windows test box, since sys.platform itself never moved.
        #
        # Self-contained Linux CUDA (dev-notes/ADR-0010): the binary comes
        # from a third-party prebuilt, hybridgroup/llama-cpp-builder
        # (_resolve_backend_asset's linux-cuda special case, above -
        # deliberately NOT a localm-built binary, see the ADR), the runtime
        # libraries (cudart/cublas) from PyPI wheels (_fetch_cuda_runtime_libs,
        # Unit 1) - never from scanning anything already on the user's
        # machine (AGENTS.md rule 4). If _resolve_backend_asset raises (no
        # matching build exists yet for this exact upstream tag on
        # hybridgroup's repo), that propagates to _provision_with_fallback's
        # caller exactly like every other provisioning failure, which
        # offers/forces the vulkan fallback - nothing new to handle here.
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
    # Also reached for chosen == "cuda" with with_cudart False (no current
    # caller produces that combination - see _cuda_setup_dialogue - but
    # forwarding cuda_line here means it never silently reverts to the
    # cuda-12 default if one ever does).
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
    re-provision hint lines; a blind ``splitlines()[-1]`` returns the last hint
    (``localm setup-llama --backend amd-rocm --force  (AMD RX 6000)``) and throws
    the actual cause away, so setup reports a nonsensical "still failed to load
    (<a command>)" reason (issue #451, AGENTS.md rule 5: do not hide the problem).

    Prefer the exception HEADER line (``SomeError: <cause>``), which carries the
    real error even when the message spans several lines; fall back to the last
    non-empty line when the output is not a recognisable traceback."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "library failed to load"
    for ln in reversed(lines):
        if _EXC_HEADER_RE.match(ln.lstrip()):
            return ln.strip()
    return lines[-1].strip()


# Exit codes the load probe uses to tell its outcomes apart STRUCTURALLY rather
# than by matching text in a traceback. 88 predates this. 89 exists so the tag
# walk-back can fire on an ABI rejection SPECIFICALLY and not on, say, a CUDA
# build refusing to load because the driver is too old - those need opposite
# responses (walk back a release vs fall back to another backend), and telling
# them apart by grepping an exception message would depend on wording that is
# upstream's to change, not ours.
_PROBE_NO_BACKENDS = 88
_PROBE_ABI_MISMATCH = 89

# The prefix _native_loads_ok puts on an ABI rejection. A string WE own on both
# ends - written here, matched by _is_abi_rejection - so it cannot drift with
# anyone else's message. Not a substring search over a traceback.
_ABI_REJECT_PREFIX = "the runtime does not match this build's struct layout"

# load_lib() runs verify_abi and RE-RAISES (see _loader.py: `except Exception:
# _loaded_lib = None; raise`), so AbiMismatch propagates out uncaught and can be
# caught here. Verified by reading that call site, not assumed.
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
    must count as a FAILED provision, not a silent success (AGENTS.md rule 5) -
    otherwise _provision_with_fallback's "prove it loads" guarantee holds only for
    self-registering builds and a broken runtime slips through, failing only at
    the first model load with the real cause already lost. A subprocess keeps the
    setup process clean (the loader mutates the DLL/lib search path) and matches
    the real run environment. Returns (ok, last_error_line)."""
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
        # Kept behind its own prefix so callers can recognise this specific
        # outcome without re-parsing upstream's wording - see _is_abi_rejection.
        why = _informative_error_line((r.stderr or "").strip()) or "layout drift"
        return False, f"{_ABI_REJECT_PREFIX}: {why}"
    detail = (r.stderr or r.stdout or "").strip()
    return False, _informative_error_line(detail)


def _warn_off_profile(chosen: str):
    """One-line heads-up when a vendor-specific backend was chosen for a vendor
    we did NOT detect. We respect the user's choice - no block, no nag, no
    re-prompt - just flag it once so a misclick is visible.

    Returns the ``hwdetect.Detection`` used for the check (or ``None`` if it
    was never computed, or detection failed), so a caller that needs the SAME
    vendor info downstream - the CUDA dialogue, to name what IS actually
    present instead of a generic "not found" hedge - does not need a second,
    redundant ``hwdetect.detect()`` call, and the two can never see different
    hardware if detection is non-deterministic (e.g. a flaky WMI query)."""
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
        actually present -> name it, recommend the real policy-backed match
        for it (hwdetect.recommended_install_backend - not a hardcoded
        vulkan regardless of hardware), and offer a genuine three-way choice
        (continue / switch to the recommendation / quit) instead of a binary
        confirm whose "no" silently imposes vulkan either way.

    *det* is optional and changes nothing when it is None or shows no vendor
    other than nvidia (the vast majority of existing callers/tests) - the
    dialogue falls back to the original generic behaviour unchanged.
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
            # No specific alternative to recommend - the original generic
            # continue-or-vulkan choice (also what a fully headless machine,
            # or one where hwdetect itself failed, falls back to).
            if assume_yes:
                console.print("  [dim]--yes: using Vulkan (no NVIDIA GPU detected).[/dim]")
                return "vulkan", False
            _flush_stdin()
            if click.confirm("  Continue with CUDA anyway? (No = use Vulkan)", default=False):
                return "cuda", True
            return "vulkan", False

        # We KNOW what IS actually here - recommend the real match for it
        # rather than a hardcoded vulkan, via the SAME policy setup.bat/sh use.
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


# WAS a bounded WALK over the last few upstream releases, picking whichever one
# happened to load. It is now a FLOOR at _PINNED_TAG, and the difference is the
# whole point rather than a tidy-up.
#
# A walk SELECTS A VERSION WHILE SETUP IS RUNNING, which is exactly what the pin
# exists to stop; and it selects it on the ABI gate alone, so its destination is
# "an older build that LOADS" - a build nobody has ever generated a token with.
# Under a pin it also inverts: landing the user on an older, less-tested release
# is a worse outcome than the one it was rescuing them from, and it looks like a
# success.
#
# A floor has exactly ONE destination and it is a constant: the build we
# confirmed. It cannot go anywhere a human did not decide, and it can only ever
# move the user TOWARDS the tested build, never away from it.
_FLOOR_TAG_DESCRIPTION = "the confirmed build localm ships"


def _floor_at_pinned_tag(chosen: str, with_cudart: bool, rejected_tag: str,
                         try_fn, detail: str) -> tuple:
    """After our own ABI gate refused *rejected_tag*, fall back to _PINNED_TAG -
    the one build we confirmed - and only from an install that had opted OUT of
    it. Returns ``(ok, tag)``.

    LOUD IN EVERY BRANCH, including the ones that do nothing. That is the point:
    three of the four outcomes here install nothing at all, and a recovery (or a
    refusal) the user cannot see is the failure mode this area keeps producing.
    Each branch names the build involved, the ABI reason, and the --tag command
    that changes it.

    THE FOUR CASES, which need genuinely different answers:

      * an exact USER PIN - not moved, ever. Moving off it is the override this
        project forbids, and the user is told which commands change it. The check
        lives here rather than at the call site so a future second caller cannot
        bypass it.
      * TRACKING upstream (--tag latest) - the case the floor exists for. The
        user asked for a build nobody had confirmed, got one our gate refuses,
        and the confirmed pin is the honest destination. Exactly one attempt: the
        destination is a constant, so there is nothing to iterate over.
      * ALREADY ON THE PIN - nothing to fall back TO; the floor IS what was just
        refused. This means localm shipped a pin its own binding rejects, which
        is a localm bug rather than an upstream one, and saying so is the whole
        value of this branch. Silently installing some older release here would
        replace a loud localm bug with a quiet unconfirmed runtime.
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
        # No floor below the floor. Do not go hunting for some older release that
        # happens to load: that build is one nobody confirmed, and installing it
        # would turn a localm bug we can fix into a runtime we cannot vouch for.
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
    # State the outcome AND why it is not what was asked for: a user who is not
    # told will report "localm installed an old runtime" as a bug.
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

    Windows and Linux ship different SYCL archives. Confirmed by inspecting
    both pinned b10375 assets: the Windows zip bundles the whole oneAPI
    DPC++ runtime alongside ggml-sycl.dll (sycl8.dll, mkl_*.dll,
    ur_adapter_level_zero*.dll, ur_adapter_opencl.dll, tbb12.dll,
    libiomp5md.dll, dnnl.dll, sycl-ls.exe, ...), while the Linux tarball
    ships only libggml-sycl.so with none of that - a separate oneAPI
    install is still required there."""
    if sys.platform == "win32":
        return "Intel oneAPI build + self-contained oneAPI runtime"
    return "Intel oneAPI build (needs the oneAPI runtime present)"


def _provision_with_fallback(chosen: str, target: Path, sha256: Optional[str],
                             with_cudart: bool, assume_yes: bool = False,
                             cuda_line: str = _CUDA_LINE) -> tuple[str, Optional[str]]:
    """Provision *chosen* and prove it loads. If it does not load, NEVER swap the
    user's pick silently (the never-override rule): inform WHY, then OFFER the
    universal Vulkan build when interactive (or fall back with a LOUD warning when
    *assume_yes* / no tty), and always say how to retry the chosen backend with
    --force. Exits non-zero if the user declines the fallback, or if NOTHING
    loads (a genuine environment fault).

    Returns ``(backend, tag)``: the backend that ended up working AND the release
    tag it was provisioned from (None when that backend has no upstream tag).
    The tag belongs to the attempt that SUCCEEDED, which is why it is returned
    from here and not recomputed by the caller - on a cuda-to-vulkan fallback
    the installed build is vulkan's, and a caller re-deriving it would record
    the tag of the backend that failed.

    *cuda_line* is the CUDA asset line to fetch when *chosen* is 'cuda' (see
    NvidiaInfo.cuda_line); irrelevant otherwise.

    vulkan and cpu are self-contained and treated as terminal: if the user
    explicitly chose one and it does not load, that is an environment problem we
    report rather than paper over with a different backend."""
    lib_name = _lib_name()

    # The tag of the attempt currently in flight. _try writes it; the success
    # paths below read it. A list rather than a rebound local because _try is a
    # closure and Python would otherwise need a `nonlocal` declaration that is
    # easy to forget when a new branch is added.
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
        # Vulkan fallback. Falling back is right when the CHOSEN BUILD cannot
        # run here; it is wrong when the chosen build is fine and a process is
        # merely holding a file. Swapping the user's backend for that reason
        # would answer a question nobody asked, and the honest fix (close it and
        # retry) is one the user can actually act on.
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
    # This runs BEFORE the backend fallback below, and the order is the whole
    # point. An ABI rejection means the BUILD is wrong for this code, so EVERY
    # backend from that release fails identically - field issue 1208 reports
    # cuda, vulkan AND cpu all AbiMismatch together, and the structural reason is
    # that one shared llama library carries the struct (see _PINNED_TAG).
    # Falling back by backend first therefore cannot help, burns the whole chain,
    # and ends in "no backend could be provisioned" having also moved the user
    # off the backend they asked for. Addressing the RELEASE addresses the cause.
    #
    # Gated on an ABI rejection SPECIFICALLY, never on any load failure: "cuda
    # will not load, the driver is too old" is about this MACHINE and a different
    # release cannot fix it, so that case must still reach the vulkan fallback.
    #
    # NOT written as "avoid a known-bad tag", and deliberately so. The property
    # is "never ship a runtime our gate rejects", whichever tag and whatever the
    # cause; a hard-coded bad tag would be wrong the moment the binding is fixed.
    # With the pin in place this should now be unreachable on a default install:
    # the pinned build is one we loaded and generated with. If it fires there
    # anyway, that is a finding about the pin, and _floor_at_pinned_tag says so
    # rather than quietly installing something else.
    if _is_abi_rejection(detail) and used_tag[0]:
        floored, floor_tag = _floor_at_pinned_tag(chosen, with_cudart, used_tag[0],
                                                  _try, detail)
        if floored:
            return chosen, floor_tag
        # Fall through: the confirmed build did not load either (or there was
        # none to fall back to), so this is not release drift after all and the
        # backend fallback is next.

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
            # The exception handler above already printed the SPECIFIC cause
            # (from _diagnose_bad_artifact, when it was a download/validation
            # failure) - this adds the escape hatches, which that message does
            # not otherwise mention, so a genuinely blocked network is not a
            # dead end.
            console.print(
                f"[dim]If your network blocks or filters this download (common on "
                f"a corporate network), download the archive yourself through a "
                f"browser and use --from <extracted-dir>, or point --url at a "
                f"mirror your network allows. Retry the same command once the "
                f"cause is fixed: localm setup-llama --backend {chosen}[/dim]")
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
    # Every attempt's own cause, chosen backend first - NOT just the last one
    # tried. The final LocalmError's *reason* is the only thing that survives
    # into the saved bug-report file and the "Sorry - X because Y" console
    # line (report_failure/build_report render summary+reason only; the
    # console.print calls below are not threaded into that context). A user
    # who explicitly picked cuda and only ever sees the final message needs to
    # know THAT failed too, not only whatever the last fallback's problem was.
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
    # Nothing loaded - the one genuinely stuck case. Raise a typed, reportable
    # error and let the CLI's single graceful handler say sorry + offer a bug
    # report. setup-llama describes the failure; it does not own reporting.
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

    The CLI-facing half of is_safe_tag: same predicate, so the flag and the
    stored-value check can never disagree about what a usable tag is."""
    tag = (raw or "").strip()
    if not is_safe_tag(tag):
        raise click.ClickException(
            f"{raw!r} is not a usable release tag. {TAG_HELP}")
    return tag


def _apply_version_request(tag: Optional[str], rollback: bool, backend: str,
                           from_dir: Optional[str], url: Optional[str]) -> None:
    """Act on --tag / --rollback BEFORE any provisioning: validate them, resolve
    what --rollback means, and move the pin.

    The pin is written FIRST, so the rest of main() provisions through the
    normal _tag_for() path with no special-casing - one code path decides a tag
    whether it came from a flag or from a pin set weeks ago. That is also what
    makes the pin apply to `localm update`, which re-invokes this command with
    nothing but --backend (see _apply_update.post_swap_command).

    Anything this cannot honour is REFUSED with a reason rather than ignored: a
    silently-dropped --tag would leave the user believing a build is pinned when
    it is not, which is worse than the drift the pin exists to stop."""
    # `tag is None` (the flag was not passed) is deliberately distinguished from
    # `tag == ""` (it was passed empty, e.g. a shell variable that expanded to
    # nothing). Treating the empty string as "no request" would DROP a request
    # the user made, which is the exact failure this function exists to prevent;
    # it falls through to _validated_tag and is refused with a reason.
    if tag is not None and rollback:
        raise click.ClickException(
            "--tag and --rollback both choose a build; pass only one. "
            "--rollback goes to the previous recorded build, --tag names one.")
    if tag is None and not rollback:
        return
    if from_dir or url:
        # --from/--url install an artifact this command did not resolve from a
        # release, so there is no tag to record or pin. Refusing beats accepting
        # a flag that could not take effect.
        which = "--from" if from_dir else "--url"
        raise click.ClickException(
            f"{'--tag' if tag is not None else '--rollback'} selects an upstream "
            f"llama.cpp release, so it cannot be combined with {which}, which "
            "installs a build you supply. Run them separately.")

    if tag is not None:
        # TWO WORDS, because they name two different destinations. Before the pin
        # existed there was only one: clearing the pin meant tracking upstream's
        # newest, so "latest" could mean both "unpin" and "track upstream" at
        # once. Now the unpinned default is the build localm confirmed, so a user
        # who wants upstream's newest is asking for something the default is NOT,
        # and reusing one word for both would silently give one of them the other.
        #
        # Both are spelled as words rather than an empty --tag so the intent is
        # visible in shell history and in a script, and so a shell variable that
        # expanded to nothing cannot silently change what an install tracks.
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
        # per localm release and a pin cannot move it. Without this refusal the
        # command printed "Rolling back the amd-rocm runtime to llama.cpp b1288"
        # and then, moments later, the pin note saying that build does not apply
        # to amd-rocm - two contradictory sentences for one action that changed
        # nothing. Refusing with the real reason beats promising and retracting.
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

    By default the newest upstream llama.cpp release is used, which means an
    upstream build that is broken on your hardware arrives on your next install.
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
    # BACKENDS, and the whole point here is to change the BUILD while the
    # backend stays the same. Without this an explicit --tag/--rollback on an
    # already-provisioned box would print "Already provisioned" and change
    # nothing, having just moved the pin - the worst outcome available, because
    # the config and the disk would then disagree with no sign of it.
    if tag is not None or rollback:
        force = True
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
            # Name the BUILD as well as the backend when it is recorded: "which
            # llama.cpp is on this box" was previously unanswerable without
            # inspecting library filenames, which is exactly how a field report
            # ended up guessing between two candidate builds.
            build = _provisioned_build(target) if have else None
            label = f" ({have} {build})" if build else (f" ({have})" if have else "")
            console.print(f"[green]Already provisioned[/green]{label} at {target}")
            if not assume_yes and sys.stdin and sys.stdin.isatty() and click.confirm("Do you want to re-download/replace them?", default=False):
                force = True
                console.print("[yellow]Replacing existing build...[/yellow]")
            else:
                _ensure_importable()
                return
        # Four distinct situations, and the two that reach here via the confirm
        # above used to read as nonsense: "Replacing amd-rocm build with
        # amd-rocm" (a re-download, not a replacement) and "Replacing amd-rocm
        # build with auto" (auto is not a backend, it is how one gets chosen).
        # The two genuinely-wrong cases are fixed; the other two keep their exact
        # existing wording, which is already correct and is asserted elsewhere.
        # Restyling a correct message would only churn the tests that pin it.
        if not have:
            console.print(f"[yellow]Replacing unrecorded build with {want}.[/yellow]")
        elif want == "auto":
            console.print(f"[yellow]Replacing {have} build with the "
                          f"auto-detected backend.[/yellow]")
        elif have == want:
            # Three genuinely different events that all used to print as
            # "Re-downloading", which reads as a no-op even when it is an
            # upgrade. The marker now carries the installed build tag when it is
            # known (see _provisioned_build), so a real b1288 -> b1307 upgrade
            # can finally say so.
            #
            # The INSTALLED build is now recorded for every tag-based backend,
            # not only amd-rocm, so naming it here no longer needs a network
            # call - it is read from the marker. What still cannot be named for
            # free is the build we are about to install: only amd-rocm knows
            # that without a lookup (_ROCM_TAG is a constant), so only amd-rocm
            # gets the "X -> Y" arrow. The others say which build is being
            # replaced and stop there, which is honest rather than guessing.
            #
            # A marker written before tag recording existed still reads back
            # None, and that case keeps its original wording.
            have_build = _provisioned_build(target)
            if want == "amd-rocm" and have_build and have_build != _ROCM_TAG:
                console.print(f"[yellow]Upgrading the {have} build: "
                              f"{have_build} -> {_ROCM_TAG}.[/yellow]")
            elif have_build:
                console.print(f"[yellow]Re-downloading the {have} build "
                              f"({have_build}).[/yellow]")
            else:
                # NOT named `tag`: that is this command's --tag parameter, and
                # rebinding it here would silently shadow the user's request.
                tag_label = f" ({_ROCM_TAG})" if want == "amd-rocm" else ""
                console.print(
                    f"[yellow]Re-downloading the {have} build{tag_label}.[/yellow]")
        else:
            console.print(f"[yellow]Replacing {have} build with {want}.[/yellow]")

    # Everything below actually MUTATES target (clear + refill), so it is guarded
    # by the cross-process provisioning lock - see _provisioning_lock's docstring
    # for why this needs to be atomic across separate processes rather than a
    # threading.Lock. Nothing above this point (the "already provisioned" read
    # and its short-circuit) touches disk, so it deliberately runs unlocked.
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
                    _clear_target_or_refuse(target)
                    _fetch_and_place(url, target, sha256)
                except RuntimeInUseError as e:
                    # Ahead of the broad handlers below, which would otherwise report a
                    # locked file as "Download failed" - a cause the user would go and
                    # investigate instead of the one that is true.
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
                # warn-once-then-comply: an explicit off-profile choice is the user's to
                # make, but flag a vendor mismatch a single time so a misclick is visible.
                # Also captures the SAME detection for the CUDA dialogue below, so a
                # "no NVIDIA found" fallback can name the vendor that IS actually
                # present and recommend the real match for it, rather than compute it
                # a second time (or not have it at all).
                det = _warn_off_profile(chosen) if backend != "auto" else None
                # CUDA is the visible peak-NVIDIA option: detect the driver, then offer to
                # fetch a self-contained runtime (no Toolkit) or fall back cleanly.
                with_cudart = False
                cuda_line = _CUDA_LINE
                # NOT platform-gated to win32: nvidia_preflight() and _cuda_setup_dialogue()
                # are both fully platform-neutral (nvidia-smi runs on Linux too, and the
                # dialogue's text/branches reference no OS). Restricting this to win32 was
                # an accident of when Linux CUDA support was added (_provision_backend's
                # Linux cudart branch, _resolve_backend_asset's Linux cuda_line-aware
                # matcher, and _fetch_cuda_runtime_libs already handle cuda_line correctly
                # for Linux and are unit-tested for it - see test_linux_cuda_runtime_
                # provisioning.py) without this call site being revisited, so a real
                # Blackwell (or any cuda-13-line) GPU on Linux silently got the cuda-12
                # line - a build with no kernels for it - and no PyPI cudart runtime
                # fetch, producing a runtime that LOADS (passes the ABI check) but
                # registers zero usable GPU devices (found live, 2026-08-11, on a
                # 3x-Blackwell Linux box: 'GPU: none in the loaded runtime (cuda)').
                # Only darwin is excluded - CUDA is not a real path on Apple Silicon.
                if chosen == "cuda" and sys.platform != "darwin":
                    # Preflight ONCE and reuse it for both the dialogue and the asset
                    # line - a second nvidia-smi call could (rarely) see different
                    # hardware and pick a line the dialogue never actually displayed.
                    info = nvidia_preflight()
                    cuda_line = info.cuda_line
                    chosen, with_cudart = _cuda_setup_dialogue(info, assume_yes, det)
                _pin_note_for_backend(chosen)
                result, used_tag = _provision_with_fallback(chosen, target, sha256,
                                                            with_cudart, assume_yes,
                                                            cuda_line)
                # Record the build tag for EVERY backend now, not only amd-rocm. The old
                # restriction rested on "the upstream backends resolve theirs through
                # _latest_tag(), a NETWORK CALL, and recording a version is not worth
                # making one". The premise was that the tag was not in hand; it was -
                # the fetch had already resolved it and simply discarded it.
                # _provision_with_fallback now returns the tag of the attempt that
                # SUCCEEDED, so this costs no additional lookup and, on a fallback,
                # records the build actually installed rather than the one that failed.
                #
                # amd-rocm still supplies _ROCM_TAG from the constant, because its build
                # is not resolved from an upstream tag at all (used_tag is None for it).
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
        # Surface a verify failure instead of exiting silently after "setup done":
        # a swallowed error here is exactly the "looks fine, actually broken" trap.
        console.print(f"[yellow]Warning:[/yellow] could not verify the native runtime "
                      f"({e}); it may not load. Run [bold]localm doctor[/bold] to check.")
