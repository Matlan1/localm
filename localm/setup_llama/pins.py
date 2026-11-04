# SPDX-License-Identifier: AGPL-3.0-or-later
"""The llama.cpp builds ``localm setup-llama`` installs, and the asset tables used to
find and verify them.

scripts/bump_llama_pin.py, scripts/bump_rocm_pin.py, scripts/check_llama_pin.py,
scripts/check_llama_rocm_pin.py and scripts/check_mtp_arch_allowlist.py read and
rewrite these constants as text, so each keeps its one-assignment shape.
"""

from __future__ import annotations

# Self-contained AMD build: lemonade-sdk llama.cpp ROCm build for gfx103X
# (RDNA2), Windows-only. Bundles its own ROCm runtime, so AMD RX 6000 users need
# no separate HIP SDK. See rocm-canary-forge/windows-native for the provenance.
DEFAULT_URL = (
    "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
    "b1363/llama-b1363-windows-rocm-gfx103X-x64.zip"
)


# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "00f56bc36ab7ec1f4cb1a4260c8835189814a0d33c0fa1756fa19a568224b8a8"
)


# The lemonade-sdk release tag DEFAULT_URL points at. The struct layouts it
# carries are bound by inference/backends/llamacpp/_structs.py and _abi.py,
# which bind every layout from lemonade b1288 on, so an already-provisioned
# runtime keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones -
# the two schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1363"


# The upstream llama.cpp release built from the SAME commit as _ROCM_TAG. Its
# Windows CPU archive supplies the SIMD (AVX2/AVX-512) ggml-cpu variants the
# amd-rocm build lacks: that build compiles its CPU backend with every x86
# instruction-set option off. The variant binds to that build's ggml-base, so
# this must name the upstream release of _ROCM_TAG's commit. See
# localm/setup_llama/rocm_cpu.py.
_ROCM_CPU_TAG = "b12086"
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
_PINNED_TAG = "b12087"


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
    "llama-b12087-bin-ubuntu-cuda-13-x64.tar.gz": "b13a2a46f5665bd292443ad6abcf775679145c89ae90dd1fa14157f8bb790c37",
    "llama-b12087-bin-ubuntu-cuda-x64.tar.gz": "c6a8010a650542aa74251865a5c7669fb5c9f814c2832cb0377e07a13c8c2434",
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
    # tag b12087 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b12087-bin-ubuntu-cuda-12.8-x64.tar.gz": "ec4eee424f8bbcc0df7c43e863cf2592d8b2d64f80182b78cffcd949be163ac1",
    "cudart-llama-b12087-bin-ubuntu-cuda-13.4-arm64.tar.gz": "fab3de7db880dc63e55223b5dc3cc713228e2e2c2435c2755cc09d83c89b500c",
    "cudart-llama-b12087-bin-ubuntu-cuda-13.4-x64.tar.gz": "6d4db5f29916217a684ee9082a112f22308fff155154561d2e2ad127ccfcd8ea",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b12087-bin-android-arm64-snapdragon.tar.gz": "042b7b9ac3c19cc65a81e1e1f13f9520b2048e4f89646abea595bb6fef1dd9b4",
    "llama-b12087-bin-android-arm64.tar.gz": "7c9cbe939b89445037ab3c7737b15fe9a794f393aff169e8fd8f8c881f3a97e7",
    "llama-b12087-bin-linux-arm64-snapdragon.tar.gz": "41038f65658f357ba5b282ea7f19c5488ac8616200a5cc76c8bb9557deebb070",
    "llama-b12087-bin-macos-arm64.tar.gz": "dd3ae5ba49cafc166b2f90ade0b7e2dad4c66ddf10837822dff2adfcd9e807ae",
    "llama-b12087-bin-macos-x64.tar.gz": "411a38d166267dce5f75916bb411b075029100f0d2d54586945ca0b727955a5a",
    "llama-b12087-bin-ubuntu-26.04-s390x.tar.gz": "1c37871a578d248ef51d21c85ac8b89be96e87a8b2de2890ea345aaae91c6958",
    "llama-b12087-bin-ubuntu-arm64.tar.gz": "b542b977193dae18ad6ad25671729b1a3e116bdc073347ac1b6d28b927f8e728",
    "llama-b12087-bin-ubuntu-cuda-12.8-x64.tar.gz": "266f3444fbf97fa6d6978c72a8b40a7f9d0916dc32e38afca9bed8793dfee041",
    "llama-b12087-bin-ubuntu-cuda-13.4-arm64.tar.gz": "b9bbbe67397864bd6fb2fff4e8db8bfc9633539b1af8122e4835c161dc093fbd",
    "llama-b12087-bin-ubuntu-cuda-13.4-x64.tar.gz": "bf76c16cfa5f19c5808e96ced95965fb61bd826dca9da291e4ef66799996970d",
    "llama-b12087-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "61a75e07f57a08dcf604503e3d6aa59c7f6a24936461a2416f192c6207c58deb",
    "llama-b12087-bin-ubuntu-rocm-10.0-x64.tar.gz": "c372171b08e5b78780d691ac5dfe416da02f1da69afc1c6c3fc44fed73504c95",
    "llama-b12087-bin-ubuntu-s390x.tar.gz": "3ce4c47679622164ae1e778e278dbfdb806a52837ab51220b1c4ee32da25ee0b",
    "llama-b12087-bin-ubuntu-sycl-fp16-x64.tar.gz": "ea9efe9ef1097393114b3114359868925e91b4c9871bb4cb04efa7f777db9c3e",
    "llama-b12087-bin-ubuntu-sycl-fp32-x64.tar.gz": "6318407f2830012934fa03c1972ac8006b70ccd9de1af462093b65c4bd46cd81",
    "llama-b12087-bin-ubuntu-vulkan-arm64.tar.gz": "ff6a8944bd0aa67d789be22df509cd9e48d492f332ad2fd614b8fae47c40068c",
    "llama-b12087-bin-ubuntu-vulkan-x64.tar.gz": "910a9ad139edca029852d251844994753dbbe04e3a719108d0f0a58dc1d9d5b4",
    "llama-b12087-bin-ubuntu-x64.tar.gz": "b64f5cb3efe10f3d74fc074933dc9d542a0e2ed32ddcad34e53f6f5ca01569c3",
    "llama-b12087-bin-ubuntu-zdnn-s390x.tar.gz": "3c89a05cbdcbe58a184d932d30ad91b0af07c3f736c52838d6abe035089b01cd",
    "llama-b12087-bin-win-cpu-arm64.zip": "e0d2a295ca51e810e29a48f26616d1c8a761058e0b816769b6c1fc443794f24b",
    "llama-b12087-bin-win-cpu-x64.zip": "cdd0f79ea411571cbb2d74d4be1604bcced9dc71c19ea95fa5ab8f31a8d42e6c",
    "llama-b12087-bin-win-cuda-12.4-x64.zip": "b7050847a935d1e67adc444e32f3c9439ec0093eea4e2eb6a6ade4c0f2527514",
    "llama-b12087-bin-win-cuda-13.4-arm64.zip": "c445536f6973fa6ed622b5f5717433263a254f9115c8f0920ce5a9f518552104",
    "llama-b12087-bin-win-cuda-13.4-x64.zip": "853c7fbc8b8b1efee964590297f127441b6abede2955719a6f084551e9a72e5f",
    "llama-b12087-bin-win-opencl-adreno-arm64.zip": "4e137a02e3e95aa1fbebdd174aa59ea479dd942de8eee9d2e41790d68ee3eb78",
    "llama-b12087-bin-win-openvino-2026.4.1-x64.zip": "5f60121b3f14d9120c618989d28b366d4bdbdaed12801a3ec5517b99ea40c390",
    "llama-b12087-bin-win-rocm-10.0-x64.zip": "a8261cf3f4dd63c76d00740089493443fd64a16e26b921978d22cd117a362aaf",
    "llama-b12087-bin-win-sycl-x64.zip": "b5af4d3f1590deb9d530a7a2d89a2b08d4d5e5d4f5af1c3244bf7f7733b8a74c",
    "llama-b12087-bin-win-vulkan-arm64.zip": "b4f13332f4b5ba8a50dccdf1e218f22a84434a4ba671f24687cf704e128aca3e",
    "llama-b12087-bin-win-vulkan-x64.zip": "8c6ac7df6fabc6435c5f044eb8a57dee1c5f9539ae4db4626aee82d7fb64e4d5",
    "llama-b12087-ui.tar.gz": "d389d6e10da15dd632b9ef7262803e56a9a5861079b5ab96e68e6318ef41e89f",
    "llama-b12087-xcframework.zip": "9f8821571c9e7807fa2c749f84ccf1edd1aeb72333e43b144031824d74acd3b8",
    # tag b1363 ROCm assets (llama.cpp 749db7f034d4, ROCm 10.3.0a20261103)
    "llama-b1363-windows-rocm-gfx103X-x64.zip": "00f56bc36ab7ec1f4cb1a4260c8835189814a0d33c0fa1756fa19a568224b8a8",
    "llama-b1363-windows-rocm-gfx110X-x64.zip": "056bbaffed3d0cca59e08afd235507266ef10310b1b7c0f607177ffa21d73122",
    "llama-b1363-windows-rocm-gfx1150-x64.zip": "2dfb9e5c37f1503232450f6f79f80e7d7cd8bad3822d2bd4b19bd28e1ff9ba27",
    "llama-b1363-windows-rocm-gfx1151-x64.zip": "28e79658e47daa64096bbde7775c73194ec968fdd5721c383d81c4f5354fa184",
    "llama-b1363-windows-rocm-gfx120X-x64.zip": "6ef9e8a4df94c25ac030bf4c4ea0e32d445a21e0ccbfa82ae9f1f035fe7916ff",
    "llama-b1363-windows-rocm-gfx908-x64.zip": "9347e29e0c70b17fd5b9a7e0254a4a738fc02f683eae223cc48427868c0d0c74",
    "llama-b1363-windows-rocm-gfx90a-x64.zip": "ace2ed12ed62c6d2864d141740a948c3298b7c624f7bf8cee41f679cb3e30346",
    "llama-b1363-ubuntu-rocm-gfx103X-x64.zip": "6fbf4970ed6ea40ec8b4be4d8b9fe0f013853d6175f8c16d3e948b5b02e46018",
    "llama-b1363-ubuntu-rocm-gfx110X-x64.zip": "a0cda946267e7ece4d28b94fcfa0ba50f07f218347ab0c4ffbfcb8a6bc66927f",
    "llama-b1363-ubuntu-rocm-gfx1150-x64.zip": "65ad1b88b2f080a9b11829512731030bd56c10b2c4dd4cb49be47ad9647a5cc5",
    "llama-b1363-ubuntu-rocm-gfx1151-x64.zip": "61bd4cd1e07ebb9c4cd931d413d88f1aba4278053e94fd244820b11921720659",
    "llama-b1363-ubuntu-rocm-gfx120X-x64.zip": "b894138ea373fe9a9033d6c42687229390e810b675153031d94244016098ceb3",
    "llama-b1363-ubuntu-rocm-gfx908-x64.zip": "4edfb8c76fb949a5ef332129560df27bde2e1028b211bb6401a1254bfcbe25ab",
    "llama-b1363-ubuntu-rocm-gfx90a-x64.zip": "d767785dc1f171de88bbd03872b2c59ff123d5c990e5ee8a7937b8377d83fa27",
    # tag b12086 upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm
    # build's SIMD CPU backend
    "llama-b12086-bin-win-cpu-x64.zip": "bfb33f52394c35a6c1ba3866a0283b094b538c3fe566df4a7259632d757da074",
}


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
            "cuda-12": ["bin-win-cuda-12.4-{arch}", "bin-win-cuda-12"],
            "cuda-13": ["bin-win-cuda-13.4-{arch}", "bin-win-cuda-13"],
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
            "cuda-12": ["bin-ubuntu-cuda-12.8-{arch}", "bin-ubuntu-cuda-12"],
            "cuda-13": ["bin-ubuntu-cuda-13.4-{arch}", "bin-ubuntu-cuda-13"],
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
