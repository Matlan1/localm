# SPDX-License-Identifier: AGPL-3.0-or-later
"""The llama.cpp builds ``localm setup-llama`` installs, and the asset tables used to
find and verify them.

scripts/bump_llama_pin.py, scripts/bump_rocm_pin.py, scripts/check_llama_pin.py,
scripts/check_llama_rocm_pin.py and scripts/check_mtp_arch_allowlist.py read and
rewrite these constants as text, so each keeps its one-assignment shape.
"""

from __future__ import annotations

import re

# Self-contained AMD build: lemonade-sdk llama.cpp ROCm build for gfx103X
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so AMD RX 6000 users need
# no separate HIP SDK. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1384/llama-b1384-windows-rocm-gfx103X-x64.zip"
)


# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "4c76e7ad6d7ba09467195c2a1b1577eb8390bef22ae5d881f83af4370b1ddc09"
)


# The lemonade-sdk release tag DEFAULT_URL points at. The struct layouts it
# carries are bound by inference/backends/llamacpp/_structs.py and _abi.py,
# which bind every layout from lemonade b1288 on, so an already-provisioned
# runtime keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones -
# the two schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1384"


# The upstream llama.cpp release built from the SAME commit as _ROCM_TAG. Its
# Windows CPU archive supplies the SIMD (AVX2/AVX-512) ggml-cpu variants the
# amd-rocm build lacks: that build compiles its CPU backend with every x86
# instruction-set option off. The variant binds to that build's ggml-base, so
# this must name the upstream release of _ROCM_TAG's commit. See
# localm/setup_llama/rocm_cpu.py.
_ROCM_CPU_TAG = "b12617"
_ROCM_CPU_ASSET = f"llama-{_ROCM_CPU_TAG}-bin-win-cpu-x64.zip"

# The amd-rocm build identity recorded in the runtime marker once the SIMD CPU
# backend is installed over it. Path-segment safe (see versions._TAG_SAFE_RE).
_ROCM_BUILD = f"{_ROCM_TAG}-cpu-{_ROCM_CPU_TAG}"


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
# Resolving upstream's newest release with uploaded assets at RUNTIME would let
# a third party's publish break every fresh and updating install with no localm
# change at all, so the tag is a constant instead.
#
# "CONFIRMED" here means the build was downloaded, loaded through localm's real
# loader, and made to GENERATE TOKENS with a real model - not merely that it
# loads. scripts/confirm_llama_runtime.py is that check, and the per-backend
# record of what each pin rests on is _PIN_CONFIRMATION below.
#
# ONE CONSTANT, NOT ONE PER BACKEND: upstream ships ONE llama.dll for every
# backend of a given tag (the backend lives in the separate ggml-* plugin
# libraries), so the struct layout this gate cares about cannot differ between
# them, and the confirm script re-checks that byte-identity at each new pin.
# GENERATION is the part that IS backend-specific, which is what
# _PIN_CONFIRMATION records.
#
# STAYING CLOSE TO UPSTREAM IS PART OF THE REQUIREMENT: a pin nobody advances
# fails a user as surely as tracking latest does. scripts/check_llama_pin.py
# reports how far behind this constant has fallen, and `--tag latest` is the
# escape hatch for a user who needs an upstream fix today.
_PINNED_TAG = "b12343"


# WHAT THE PIN RESTS ON, PER BACKEND. Not a boolean and not a single "confirmed"
# flag: a confirmation job that is green because it SILENTLY SKIPPED the backends
# it could not test would carry the word "confirmed" without the evidence.
#
# The asymmetry is hardware: a GitHub runner has no
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
    # TO CONFIRM cuda: on NVIDIA hardware, run
    #     python scripts/confirm_llama_runtime.py --tag <_PINNED_TAG> --backend cuda --receipt <file>.json
    # and on a PASS, change this entry to the same "load + generate, measured
    # (<platform>, <card>)" shape as "cpu"/"vulkan" above, citing the receipt.
    # Maintainer-run, occasional cadence.
    "cuda": "ABI only (shared llama library); generation NOT measured - no NVIDIA hardware",
    "sycl": "ABI only (shared llama library); generation NOT measured - no Intel GPU",
    "hip": "ABI only (shared llama library); generation NOT measured - needs a system ROCm toolkit",
    "metal": "ABI only (shared llama library); generation NOT measured - no Apple Silicon",
    # Not an upstream tag at all: the lemonade-sdk build, pinned separately by
    # _ROCM_TAG, so this table's subject (_PINNED_TAG) does not describe it.
    "amd-rocm": "out of scope for _PINNED_TAG - pinned separately as _ROCM_TAG, "
                "whose generation was NOT measured by this pin's confirmation",
}


# Stored in the `llama_runtime_pin` config key to mean "track upstream's newest
# release", the opt-in to bleeding edge. A SENTINEL rather than an empty value
# because empty now means the shipped pin, which is the safe default; a user who
# wants upstream's newest has to say so, and it stays visible in their config.
#
# Kept OUT of pinned_tag()'s return value. Every existing caller of that
# function treats what it returns as an exact release tag and interpolates
# it into a URL path segment, so letting the sentinel through would produce a
# confident request for a release literally named "latest". tracks_latest() is
# the second accessor instead, and _tag_for() is the only place that consults
# both.
_TRACK_LATEST = "latest"


# The documented word for "go back to the build localm ships and confirmed".
# Distinct from `--tag latest`: those are two different destinations, so each
# has its own word.
_TRACK_DEFAULT = "default"


# Linux x64 CUDA builds come from this third-party repo, which uses upstream's
# bNNNNN tags and asset-name convention. Linux ARM64 CUDA builds come from
# upstream's own release.
_CUDA_LINUX_REPO = "hybridgroup/llama-cpp-builder"

# Offline checksums for the Linux CUDA assets of _PINNED_TAG on _CUDA_LINUX_REPO.
# Only consulted when that repo's release listing is unavailable.
_CUDA_LINUX_PINNED_SHA256 = {
    "llama-b12343-bin-ubuntu-cuda-13-x64.tar.gz": "b59b5a25160c2044e3bb3366eccebb458a798c79137ebc48b87da5ffc33c1518",
    "llama-b12343-bin-ubuntu-cuda-x64.tar.gz": "31875eae261f73cb95a23037d8e4b94fded28729ac4342de08ef9519527c6059",
}


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
# THE TABLE HOLDS EXACTLY THE TAGS THIS FILE PINS, and a test enforces that: an
# entry for a tag nothing resolves to is a stale pin that reads as coverage.
#
# The values are the API's own `digest` fields.
_PINNED_FALLBACK_SHA256 = {
    # tag b12343 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b12343-bin-ubuntu-cuda-12.8-x64.tar.gz": "1f1695f1c86c04e90a50a00a08daa0c0ddf388d53dbd60d6a970ebc97daed17b",
    "cudart-llama-b12343-bin-ubuntu-cuda-13.5-arm64.tar.gz": "ff2507250c8929ac14d3061afd1ded3426d125c892f5d32abc3e62a10712a5be",
    "cudart-llama-b12343-bin-ubuntu-cuda-13.5-x64.tar.gz": "0992a9431950e95f061d76dc6d4c48719968bfdfbc7d3556a7f36fc2bafec33e",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.5-arm64.zip": "f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca",
    "cudart-llama-bin-win-cuda-13.5-x64.zip": "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
    "llama-b12343-bin-android-arm64-snapdragon.tar.gz": "a0dde32ac28bf64b4e4b401aef1fa6f2c705e4b958615757fcc3d66e93e335a2",
    "llama-b12343-bin-android-arm64.tar.gz": "96767b5e5b1a57ae884e336ddedb336ef6cf4d3cf1389e4c0362e7f3ec67ecad",
    "llama-b12343-bin-linux-arm64-snapdragon.tar.gz": "982ede88508bbdf9aa1983a7f61367349349b7dfad449e0902d627f6c089682a",
    "llama-b12343-bin-macos-arm64.tar.gz": "58f740df2a0c4002f116e9c4fbc6f49fd2d3645063e3faa5fcfb5f37d67b328e",
    "llama-b12343-bin-macos-x64.tar.gz": "f71709a91fc06f057d2552ba3e9a91f285ea258edc9f4bb3ca0be05018b97e35",
    "llama-b12343-bin-ubuntu-26.04-s390x.tar.gz": "c506332ab69575bb056328965d990e5df15775cb77ca753aef558d8c14174622",
    "llama-b12343-bin-ubuntu-arm64.tar.gz": "a6ff539bf3f0c30efe849c78ef33ab2d9158f8581d89f4606859aa9fd2c3b362",
    "llama-b12343-bin-ubuntu-cuda-12.8-x64.tar.gz": "ad866c99256c6f985aa3bb83283d5adfcb7b9cd1a43b6b516a46e29189283703",
    "llama-b12343-bin-ubuntu-cuda-13.5-arm64.tar.gz": "42d2bf63c1a8964b02a21985f4b8ff638638d9442e4aa87a24fbfa3686d67e0b",
    "llama-b12343-bin-ubuntu-cuda-13.5-x64.tar.gz": "801628c1a48ed395694de31fabe3382cae60326f0186eb077a88422d0ae6097f",
    "llama-b12343-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "ffa29e69fb7e5d869ee8c8a3f93c591d645faea4c675c188af6d8d67ef2739b2",
    "llama-b12343-bin-ubuntu-rocm-10.0-x64.tar.gz": "8df74f66b3b9a632d1d00f5990f516cdd6a8e45080089d02d3b70604ef5df502",
    "llama-b12343-bin-ubuntu-s390x.tar.gz": "9a1075910693a641a0097936768488d4079abb9406202e06da6a45130b140d60",
    "llama-b12343-bin-ubuntu-sycl-fp16-x64.tar.gz": "e8d2db0167cc8f93a2bff03268eafdd8944e3331229e0cc56393653360b2bb30",
    "llama-b12343-bin-ubuntu-sycl-fp32-x64.tar.gz": "daf88f00140fdcb3376461520a8b04d10b0dbf9f0dbf86c3af8087c59d97cabb",
    "llama-b12343-bin-ubuntu-vulkan-arm64.tar.gz": "835b6b361e606710ff94fcd5b94c4e8404093795f126bb307b04d4ef98516e7d",
    "llama-b12343-bin-ubuntu-vulkan-x64.tar.gz": "74e64ec74da3345963e35142120f2671c0e6a21b8144e0c62d4f208c03952efb",
    "llama-b12343-bin-ubuntu-x64.tar.gz": "fe4f16988ceb35a5b01ca6306b29602ef3ea4ef7a877ff596ed7d309c6714335",
    "llama-b12343-bin-ubuntu-zdnn-s390x.tar.gz": "376df7f69f9d387d332260eb0dc74738b2f0ed1ebdb4c06b75a6515d0afb3b0f",
    "llama-b12343-bin-win-cpu-arm64.zip": "5b735fa6f2e81727da2e9f2f2d8990506c6f1f629e38494be7be9ec8a44900c9",
    "llama-b12343-bin-win-cpu-x64.zip": "3f343af50e5068e40af6628c8079a221bf0d4a7a1f4402643bdd8412dc68a226",
    "llama-b12343-bin-win-cuda-12.4-x64.zip": "8e81b8da78666e05c83f033c603e596a5c24db031ad400b4dc454aa8dad1a0c5",
    "llama-b12343-bin-win-cuda-13.5-arm64.zip": "5a8acb454d66d39acd952b93b8ac7def9f71e26dcddb7f2ad45c667bb93915c9",
    "llama-b12343-bin-win-cuda-13.5-x64.zip": "e705e5ae4f7025c7fa95dd28955d5734d4eeac4acbccd2966261458c9e2edb91",
    "llama-b12343-bin-win-opencl-adreno-arm64.zip": "aadb4cd4a48dcbc2c7c6aa833a36419159a86ebb15d7c57cca9c232739765f8b",
    "llama-b12343-bin-win-openvino-2026.4.1-x64.zip": "29ca2b10fe44b498e5ff0b9b0d009e2c151cdde46072045e47024789897c4bb5",
    "llama-b12343-bin-win-rocm-10.0-x64.zip": "2f243f7fe4e25be26382ef0ca6ef4dc3b1a5ed620dfe4d348144a3bf682610a8",
    "llama-b12343-bin-win-sycl-x64.zip": "5804a1b65bda0b05fb911181a121e674f78f651e69e839be58396313e5e2bb40",
    "llama-b12343-bin-win-vulkan-arm64.zip": "176b7325d8dd589e1f5916c2934940aea9f20cc4fd28af9096f63d674c60c9d1",
    "llama-b12343-bin-win-vulkan-x64.zip": "0170da0d41eb20250beb16085bc2ac397163b8e82bd725aab5632f732b8512b1",
    "llama-b12343-ui.tar.gz": "15f84359e4181a1080858549b92968458147aa71c239b7b401e35a459c60af8e",
    "llama-b12343-xcframework.zip": "8de59ad8f2dc13e816cfac6457e5ebae00bf27c2985af2cc046cff7cf75e1244",
    # tag b1384 ROCm assets (llama.cpp 44c20b1c0a1c, ROCm 10.3.0a20261121)
    "llama-b1384-windows-rocm-gfx103X-x64.zip": "4c76e7ad6d7ba09467195c2a1b1577eb8390bef22ae5d881f83af4370b1ddc09",
    "llama-b1384-windows-rocm-gfx110X-x64.zip": "c9af3436ed233dd3a4900530acccfa4817ceb787e9aa90e3e07c2c9883b32696",
    "llama-b1384-windows-rocm-gfx1150-x64.zip": "a9be0e1902aaf3ead0642c7e3c64df603e53bf53b754c14ca7766889b4afa79e",
    "llama-b1384-windows-rocm-gfx1151-x64.zip": "0dbda3cdff0a64049dd29df50abd5143f57a3bffec9c01a33f76d213a62c4237",
    "llama-b1384-windows-rocm-gfx1152-x64.zip": "e6c03b1a7c99b58d9551d37c7495338bc1e62dc180260b163cf0441a149f9849",
    "llama-b1384-windows-rocm-gfx1153-x64.zip": "bce30114dc39a506d00f81e8d539908b5700d99b29a850306d6b93d13cced24c",
    "llama-b1384-windows-rocm-gfx117X-x64.zip": "b5275d3c79299c7528f62452e61a93d66349ee72f95266d1c2e9aa6e452c2341",
    "llama-b1384-windows-rocm-gfx120X-x64.zip": "f0786e944b7d6e2462cb868a33cadd6f7042af51094abf5673f8aaa96fb3098c",
    "llama-b1384-windows-rocm-gfx908-x64.zip": "e816d358d4791f2154317f2e0c6a5d5c09da905ccaf508334af385570acefc39",
    "llama-b1384-windows-rocm-gfx90a-x64.zip": "b89948f17a4ae33bd5c8725dfe8e446c94eeabc3c5641a88c13785201a04055a",
    "llama-b1384-ubuntu-rocm-gfx103X-x64.zip": "a970f78e1535992629520bff0e159ef91fe49e3c0c6356c86382a49f684e9d14",
    "llama-b1384-ubuntu-rocm-gfx110X-x64.zip": "4a72489636383ded38e85186e26d516d4bb2b26b661a34dad0c4a81128435dd9",
    "llama-b1384-ubuntu-rocm-gfx1150-x64.zip": "30ddc82ab26301747bc70d58662642373b5b18173d6214696fb210076e0e2d17",
    "llama-b1384-ubuntu-rocm-gfx1151-x64.zip": "a838b224d2e12039158c4b44931d8311b47bc6c3e8557674adcc22f9433a0943",
    "llama-b1384-ubuntu-rocm-gfx1152-x64.zip": "57dc1115b54a2e6f0d66080f951d2c0cf5911b779d7db340185749aef88b2d26",
    "llama-b1384-ubuntu-rocm-gfx1153-x64.zip": "5fd6682a81b927c2100b0e3ebc1121a86696fe9cad50c204c5404e31f3053b0f",
    "llama-b1384-ubuntu-rocm-gfx117X-x64.zip": "0397350eaabd8626e3be516e4ed731f5e73b8ca94c13f3281d16c3f1ad24091f",
    "llama-b1384-ubuntu-rocm-gfx120X-x64.zip": "171c1b1e45471f15f479732fcefc2ca8474bff633218236a9fdd094c17f75d3f",
    "llama-b1384-ubuntu-rocm-gfx908-x64.zip": "8c9549fe9391fffe2eba2bd0390a43a1fc327d53b086dc0b78e006391bb0414b",
    "llama-b1384-ubuntu-rocm-gfx90a-x64.zip": "a2346faf088a7aea376e3c8d17a1e4d5cd448e96914cb25e557a4dca6de65815",
    # tag b12617 upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm
    # build's SIMD CPU backend
    "llama-b12617-bin-win-cpu-x64.zip": "d682c5ad5b689626c1f2d33f48a68008a0f3ed0498ffdfa27a9358b55a11561e",
}


def _pinned_cuda_toolkit(os_fragment: str, major: str) -> str:
    """The ``X.Y`` CUDA toolkit of the _PINNED_TAG build for *os_fragment*
    (``"win"`` or ``"ubuntu"``) and CUDA *major* (``"12"`` or ``"13"``), read
    from the asset names in _PINNED_FALLBACK_SHA256. Raises RuntimeError
    unless exactly one such version is pinned."""
    rx = re.compile(rf"^llama-{re.escape(_PINNED_TAG)}-bin-{os_fragment}-cuda-"
                    rf"({major}\.\d+)-(?:x64|arm64)\.(?:zip|tar\.gz)$")
    found = {m.group(1) for name in _PINNED_FALLBACK_SHA256 if (m := rx.match(name))}
    if len(found) != 1:
        raise RuntimeError(f"_PINNED_FALLBACK_SHA256 pins {sorted(found) or 'no'} CUDA "
                           f"{major}.x build(s) of {_PINNED_TAG} for {os_fragment}; "
                           "expected exactly one")
    return found.pop()


# The CUDA toolkit of each pinned build line, per platform:
# {"win": {"cuda-12": "12.4", "cuda-13": "13.5"}, "ubuntu": {...}}.
_CUDA_TOOLKIT = {os_fragment: {f"cuda-{major}": _pinned_cuda_toolkit(os_fragment, major)
                               for major in ("12", "13")}
                 for os_fragment in ("win", "ubuntu")}


# Per-backend asset matcher: substrings that must appear in the release asset
# name for (platform, backend). Substring matching (not exact names) keeps this
# robust to upstream version suffixes drifting (e.g. cuda-12.4, rocm-7.2).
# "{arch}" is replaced by this process's architecture (hwdetect.machine_arch),
# and an asset whose name carries a different architecture never matches. Which
# architectures each backend has a build for is hwdetect.PREBUILT_ARCHES.
_ASSET_MATCH = {
    "win32": {
        "cpu":    ["bin-win-cpu-{arch}"],
        "vulkan": ["bin-win-vulkan-{arch}"],
        # Keyed by CUDA LINE (NvidiaInfo.cuda_line), not a flat preference
        # list: a Blackwell-class GPU must never fall through to a 12.x asset
        # even as a "closest match" fallback, since that build's fatbin has no
        # kernels for it (see the _CUDA_LINE block for the full rationale).
        "cuda": {
            "cuda-12": [f"bin-win-cuda-{_CUDA_TOOLKIT['win']['cuda-12']}-{{arch}}",
                        "bin-win-cuda-12"],
            "cuda-13": [f"bin-win-cuda-{_CUDA_TOOLKIT['win']['cuda-13']}-{{arch}}",
                        "bin-win-cuda-13"],
        },
        "sycl":   ["bin-win-sycl-x64"],
        # Upstream renamed the Windows ROCm/HIP asset at b10356:
        # "bin-win-hip-radeon-x64" -> "bin-win-rocm-<version>-x64" (e.g.
        # bin-win-rocm-7.14-x64), the same shape as the Linux asset's own
        # versioned name. Without a fragment matching the new name this backend
        # guesses a URL for a filename that does not exist and 404s.
        #
        # ORDER: newest naming first, mirroring the Linux entry's "specific, then
        # generic" shape below. The OLD name is LAST so an explicit --tag on a
        # pre-rename release still resolves.
        "hip":    ["bin-win-rocm-7.14-x64", "bin-win-rocm", "bin-win-hip-radeon-x64"],
    },
    "linux": {
        "cpu":    ["bin-ubuntu-{arch}"],
        "vulkan": ["bin-ubuntu-vulkan-{arch}"],
        "cuda": {
            "cuda-12": [f"bin-ubuntu-cuda-{_CUDA_TOOLKIT['ubuntu']['cuda-12']}-{{arch}}",
                        "bin-ubuntu-cuda-12"],
            "cuda-13": [f"bin-ubuntu-cuda-{_CUDA_TOOLKIT['ubuntu']['cuda-13']}-{{arch}}",
                        "bin-ubuntu-cuda-13"],
        },
        "sycl":   ["bin-ubuntu-sycl-fp16-x64", "bin-ubuntu-sycl-fp16", "bin-ubuntu-sycl"],
        # Same versioned-name drift as the Windows entry above: 7.2 at the old
        # fallback tag, 7.14 at the pinned one. Newest first, generic last.
        "hip":    ["bin-ubuntu-rocm-7.14-x64", "bin-ubuntu-rocm-7.2-x64", "bin-ubuntu-rocm"],
    },
    "darwin": {
        "cpu":    ["bin-macos-{arch}"],
        "metal":  ["bin-macos-arm64"],
    },
}


# Backends a user may request directly (in addition to the special "auto" and
# the self-contained "amd-rocm").
_UPSTREAM_BACKENDS = ("vulkan", "cuda", "sycl", "hip", "cpu", "metal")
