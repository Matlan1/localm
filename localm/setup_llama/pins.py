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
_PINNED_TAG = "b12676"


# WHAT THE PIN RESTS ON, PER BACKEND. Not a boolean and not a single "confirmed"
# flag: a confirmation job that is green because it SILENTLY SKIPPED the backends
# it could not test would carry the word "confirmed" without the evidence.
#
# The asymmetry is hardware: a GitHub runner has no
# GPU, so CI can honestly generate on cpu only. vulkan and hip are confirmed on
# the maintainer's box. cuda, sycl and metal need hardware nobody here has.
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
    "hip": "load + generate, measured (Windows x64, AMD RX 6900 XT / gfx1030; the "
           "runtime registered a ROCm0 GPU device, with AMD's ROCm libraries added by "
           "setup)",
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
    "llama-b12676-bin-ubuntu-cuda-13-x64.tar.gz": "914d65bff738351974903910b4e12fde381019b3f783f2fabb5f1808b5be5448",
    "llama-b12676-bin-ubuntu-cuda-x64.tar.gz": "9cd0c4f3f558bf84b2260d44cd5da3fcd1a24875a022fedf7b59667611e5867d",
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
    # tag b12676 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b12676-bin-ubuntu-cuda-12.8-x64.tar.gz": "7950380b5b88ce0f7ef082f3a86c23a93a11f39dad9c3a7a0d838ad017d1e4b3",
    "cudart-llama-b12676-bin-ubuntu-cuda-13.5-arm64.tar.gz": "f2686443bff3f03abd5b3d785b784100c13a1f1a7da630023343e4088ef3eadd",
    "cudart-llama-b12676-bin-ubuntu-cuda-13.5-x64.tar.gz": "76177793e1d12df47aaebbc3b97a03d33d295d88ea3c7b264ac834822412f3a3",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.5-arm64.zip": "f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca",
    "cudart-llama-bin-win-cuda-13.5-x64.zip": "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
    "llama-b12676-bin-android-arm64-snapdragon.tar.gz": "3640b7cf2afea3a7bebda95ce07b6300a86c2a078b991ab66c9843be86f847e2",
    "llama-b12676-bin-android-arm64.tar.gz": "b357eab59d818f4f25f6908d4ea1d3142710c5a2c9fd5755d1e66f297a140dcc",
    "llama-b12676-bin-linux-arm64-snapdragon.tar.gz": "3ab830a7a31fa7f676c0b9b92f3c8247f174406fcbce9d3ff5b0607ca8eba7c6",
    "llama-b12676-bin-macos-arm64.tar.gz": "a222d1ba06a3f0215fa7d22f372af6ba74c5b4cfca5f6b2aa6974c7b63017b59",
    "llama-b12676-bin-macos-x64.tar.gz": "10f36c786606e874f7440a87b48716dbd19f4bc6670e7dcd005e43c61977e6ef",
    "llama-b12676-bin-ubuntu-26.04-s390x.tar.gz": "356060e4e82fc909e74233e4ca7e08494ce27b15d5fc0c94d4a942f443da5944",
    "llama-b12676-bin-ubuntu-arm64.tar.gz": "05707ec03e09e83a4e5bde447adf7e59280035eeeced28dbe167e716bd432f04",
    "llama-b12676-bin-ubuntu-cuda-12.8-x64.tar.gz": "f92c568b56fa39afb92ead65a899a618d2ef29cff4ca2b739d51eb777c55b12e",
    "llama-b12676-bin-ubuntu-cuda-13.5-arm64.tar.gz": "49706f0b4106987421d3fb5bfc903fd5f4cba16ef0e5253bb6ae04c7330d4ca4",
    "llama-b12676-bin-ubuntu-cuda-13.5-x64.tar.gz": "bcb9629f51e99c4ad771822e9799e3c4119178faf784ed324111ce0ffeb88e88",
    "llama-b12676-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "97608439ec45f4c5894d46eb742c6721eae302aa4390608c1ad06a8f5b93d273",
    "llama-b12676-bin-ubuntu-rocm-10.0-x64.tar.gz": "ab5678cd5f77638994ce883ff98f5c33558133588b8bf68da78dd01f8150f208",
    "llama-b12676-bin-ubuntu-s390x.tar.gz": "a18fc36a3e373dac825edcfc50ea3bc397b0551c024a946c4b081c7c9b4688bd",
    "llama-b12676-bin-ubuntu-sycl-fp16-x64.tar.gz": "e2b304de3d5cb5ce28d0020c157085863e2619594e020898752fee01ccc77c82",
    "llama-b12676-bin-ubuntu-sycl-fp32-x64.tar.gz": "730b7969bfb64073a9214b08e7c9d414ca5b620c3c94c9fafb0a4cc530b33b95",
    "llama-b12676-bin-ubuntu-vulkan-arm64.tar.gz": "8deadf4b99e84fc302d4cf16b20a9080d7b779deb4e543b7bc0e1d7f933b97d1",
    "llama-b12676-bin-ubuntu-vulkan-x64.tar.gz": "cd355471644b3e905f7fc678a230ec60f93cb0c39b3947dc696c9a449bbc007e",
    "llama-b12676-bin-ubuntu-x64.tar.gz": "c65369a8574d096d90c593fef0e77a1ba2abfdfa7e7e9426bbf8939781ef8a07",
    "llama-b12676-bin-ubuntu-zdnn-s390x.tar.gz": "764df1c5a8f63a07df48620aaddedc3fd8d9eb9be79dc902ec9aca89f8a249ff",
    "llama-b12676-bin-win-cpu-arm64.zip": "272cce4ec53aff079cc32d0362932684b444b08d9f9e34924b9f7ebcae680b9d",
    "llama-b12676-bin-win-cpu-x64.zip": "fdf78a3924fe92ccfb24af58544a6dccbc1feeb0587e77fca9ec3bef6737086c",
    "llama-b12676-bin-win-cuda-12.4-x64.zip": "1d5692d90213a39aa01ef7da8d2c662ee815e0c6e4ad63b6f56473cc5b8ddb84",
    "llama-b12676-bin-win-cuda-13.5-arm64.zip": "9efa07ac4ec90040e3a725139ba9d5e8100344eba6a6fcdab66b391b39ed93fc",
    "llama-b12676-bin-win-cuda-13.5-x64.zip": "39f2cdae1833fc58b4ab6c7ae29b0100f6ee656008ee7ae62285bc6d812d0258",
    "llama-b12676-bin-win-opencl-adreno-arm64.zip": "b58527931712689852238e989dae483a5aad062b197a25560233cd29779cf61f",
    "llama-b12676-bin-win-openvino-2026.4.1-x64.zip": "8148b92a59d11d952c07989860d1e464bec398543db9351f8ae38fcd8414d06d",
    "llama-b12676-bin-win-rocm-10.0-x64.zip": "0e9ae56154dce138fa0c86143c5770b74fa440325d7de0af3c5eb710bb4a7227",
    "llama-b12676-bin-win-sycl-x64.zip": "a58002fb8068e74673ba0f221d71f39c083201d579beea1012e5156207a9b512",
    "llama-b12676-bin-win-vulkan-arm64.zip": "84efb8cfa7fa84e77ccb26ec37500b7d2434d2d9312a1194ed87e2073f1b851f",
    "llama-b12676-bin-win-vulkan-x64.zip": "4a0732dd3b68ea1d7346fde885253a315c0c5eb7279da897d936892e0f709108",
    "llama-b12676-ui.tar.gz": "d6b992e354c30d5df4fd27d9b015889f67fd667937fae187d672b968a24b41ad",
    "llama-b12676-xcframework.zip": "0a8985e744bde1306c18271a34a9d7e9d093e3e93d48e1d7eff6d3e6a2122c94",
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


# AMD's ROCm library wheels for the upstream Windows hip build. That build ships
# ggml-hip.dll with the HIP runtime (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll)
# but not the BLAS libraries ggml-hip.dll links (hipblas.dll and its rocBLAS,
# rocSOLVER, hipBLASLt and origami dependencies) nor their GPU kernels. This is
# the index of the AMD wheels upstream builds that archive against
# ("rocm[libraries,devel]" at the ROCm release its asset name carries).
# localm/setup_llama/hip_libraries.py fetches rocm-sdk-libraries for the
# libraries and one rocm-sdk-device-<gfx> wheel per GPU architecture for the
# kernels.
_HIP_SDK_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"

# The ROCm release ("major.minor" of the bundled amdhip64_7.dll's file version,
# which is also the version in the upstream asset name, e.g. win-rocm-10.0) ->
# the wheels of that release: (file name, size in bytes, sha256). The device
# entries are the GPU architectures the build's ggml-hip.dll carries code for.
_HIP_SDK_WHEELS = {
    "10.0": {
        "version": "10.0.0",
        "libraries": ("rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl", 116831755,
                      "bfc7a1bd3b6050fb4d36d8ac2eea80d3afd70c5604a29142463d6cb890fc73bf"),
        "devices": {
            "gfx1010": (47525108, "2b6da925376de9689ab9f22ce7cc55b6d476722d9458fbd1ec52d4aa066d3e6c"),
            "gfx1011": (47190349, "1a6de3efe7bc92b1e55367a767eecf0468231b03d9319494d1e4b077571b6757"),
            "gfx1012": (47187798, "51f2998d2f33164e4a14594107e6c33db54da1d9867fa6947ea993997c4ae06e"),
            "gfx1030": (52153384, "19ada0359b499c19c1ecb1cba21b067fce89b4f0b0de6f0f7b76cbef85dcb85b"),
            "gfx1031": (52179193, "ab7cbfb41e1ada00fb7254c974f7386303cdd1139a170fb022e84e6f10d9f5f2"),
            "gfx1032": (52182885, "5d0581e077cc1a482e70430539c63e60c4b4d7b9b82be0ac0bab6d62c3feb4de"),
            "gfx1033": (52186039, "46ba5b395bf75756f9c5f1154b2083dd576221dfbd15620b5fa597f498aad798"),
            "gfx1034": (52183796, "8d527c2f127d42ecb098e0abc45c15ca28689b6317f01b99d455e6d3562a3b52"),
            "gfx1035": (52181539, "c06479bad4963f4aecde4e0fe102eb39ae652e80da62546c3bd3fb0b9079a0a4"),
            "gfx1036": (52183283, "655f343cd805f21bfd45244902f99afba9b7ada323288a1b9ba3336bd41c1ccc"),
            "gfx1100": (189830160, "db39cbc5aae7a67c73d76692899a6a89bbc10eafc92f856f48e9c03247a91f8e"),
            "gfx1101": (185030273, "7b8edd82943a11c29c80f84c95822e36ddd794b7024878e9660170d050a29243"),
            "gfx1102": (183114327, "b9e0f3fa492f0a05a5c4f479b80915477911f8402956c0e2c593b593d6214531"),
            "gfx1103": (55710049, "3de6ba204d78e0cb2a32cf064371c181d7a17d1ca8b25537b1a7f11eaa66fa17"),
            "gfx1150": (134574199, "22c19fcee580102cdc2f9379edf056643843c86ac6312dcf34d8ed21964c2cc4"),
            "gfx1151": (139346623, "1ab27ebfe61ea3c3fd2c5c1a0e780ae22ed9b662106ce0aefda0a30e3b11577f"),
            "gfx1152": (134480596, "88db4e79f31bef68bf8cb321fb83704dda6d11ad6da8dcfbdc8660f077c7195c"),
            "gfx1153": (134629087, "1fde64d87edd815403014743700000a016435f040c43114e764b5cc7b919413c"),
            "gfx1200": (377648520, "f33226078932ee2200f5595c521895edf5d7abf3170975d8c6f72a2a3949f516"),
            "gfx1201": (312303802, "c6e4c5a0da941230bed975765e644597570081c9a41654d5a052c277167f4576"),
        },
    },
}


# Backends a user may request directly (in addition to the special "auto" and
# the self-contained "amd-rocm").
_UPSTREAM_BACKENDS = ("vulkan", "cuda", "sycl", "hip", "cpu", "metal")
