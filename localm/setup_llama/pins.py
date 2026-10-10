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
    "b1342/llama-b1342-windows-rocm-gfx103X-x64.zip"
)


# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "2236d8d3074e570e871dfbf03b77f031c18df6a96ce078cba7b1ae37047006d5"
)


# The lemonade-sdk release tag DEFAULT_URL points at. The struct layouts it
# carries are bound by inference/backends/llamacpp/_structs.py and _abi.py,
# which bind every layout from lemonade b1288 on, so an already-provisioned
# runtime keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones -
# the two schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1342"


# The upstream llama.cpp release built from the SAME commit as _ROCM_TAG. Its
# Windows CPU archive supplies the SIMD (AVX2/AVX-512) ggml-cpu variants the
# amd-rocm build lacks: that build compiles its CPU backend with every x86
# instruction-set option off. The variant binds to that build's ggml-base, so
# this must name the upstream release of _ROCM_TAG's commit. See
# localm/setup_llama/rocm_cpu.py.
_ROCM_CPU_TAG = "b11513"
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
_PINNED_TAG = "b11541"


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
    # Maintainer-run, occasional cadence; see RELEASE.md's pin-currency section.
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


# Third-party Linux CUDA prebuilt: upstream publishes no bare Linux CUDA binary
# itself, so this fetches from an actively-maintained third party instead - the
# same shape as the amd-rocm backend's own dependency on
# lemonade-sdk/llamacpp-rocm, just below. hybridgroup/llama-cpp-builder tracks
# upstream's bNNNNN tag numbering 1:1 and publishes the same asset-name
# convention upstream itself uses for every other Linux backend. NOT a
# localm-built or localm-hosted binary.
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
# THE TABLE HOLDS EXACTLY THE TAGS THIS FILE PINS, and a test enforces that: an
# entry for a tag nothing resolves to is a stale pin that reads as coverage.
#
# The values are the API's own `digest` fields.
_PINNED_FALLBACK_SHA256 = {
    # tag b11541 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz": "6c8acf749cb80a5932b8c5ba16476c3626ce72b09bcfe9078c27b1087ae39c4f",
    "cudart-llama-b11541-bin-ubuntu-cuda-13.4-arm64.tar.gz": "735c30989a66950ec90b6a339a4c38cf0b08c88692a87bfe0ee290fa0d5e2a97",
    "cudart-llama-b11541-bin-ubuntu-cuda-13.4-x64.tar.gz": "4d6dba50d1d16b3987b6a8cff73a58338ea1eba7ded1d3295ea174b4f515a746",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11541-bin-android-arm64-snapdragon.tar.gz": "0ce0a3742ba5b4fbed2f18ad18bc9dd1d4292bc290c5aa9a30c5456716c8bc18",
    "llama-b11541-bin-android-arm64.tar.gz": "af3e6d1fec872911c5d33036ea64dc1d78926ca074991979cf0d10b18c0b405d",
    "llama-b11541-bin-linux-arm64-snapdragon.tar.gz": "673a323f7df4e4f20bb364bf7a9d3d79fea37e2229c90ec765ead6cdd6b04cc3",
    "llama-b11541-bin-macos-arm64.tar.gz": "226929037cfcb956f50ae1d54bac13d3d9446d8d819c439580e1e2c00197da5e",
    "llama-b11541-bin-macos-x64.tar.gz": "f74c3e79f68e9d0099bd2adc45c72208ec9748cd39e4098eeef7e49011445565",
    "llama-b11541-bin-ubuntu-arm64.tar.gz": "fcd44afd8ecdd2068d0c8505c0f96fa32014b4814d0dc74947a471deddcb9998",
    "llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz": "551ee13d02dd42599861c8ffbd71c7a110f379a51cb9c5947cb799cdbbc610b2",
    "llama-b11541-bin-ubuntu-cuda-13.4-arm64.tar.gz": "cc9795af6195f360c6595ea17797035a26f878c60dfb49539e4b087689468ab9",
    "llama-b11541-bin-ubuntu-cuda-13.4-x64.tar.gz": "2d6f7589eb46f80905cd8fadcc41c67a0a5f5d6fe96ddd4fb37fd7a73a0872d5",
    "llama-b11541-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "6d6688e456c736602c6d025c13a988acbd85976d86c4d28de20df518aa2f1f13",
    "llama-b11541-bin-ubuntu-rocm-10.0-x64.tar.gz": "22d3c8165192f88bbf9c77e2a72d44dede698faf5679a738b610291e3d0755ee",
    "llama-b11541-bin-ubuntu-s390x.tar.gz": "41f007793c747261cd30c7dd56479fdcbcbcd7a1d5caee78392d7f81df47a48b",
    "llama-b11541-bin-ubuntu-sycl-fp16-x64.tar.gz": "33d2b28fff500e8129e17a92142ff7a9ed0bad6f59b1bd6a0d59883ba6320944",
    "llama-b11541-bin-ubuntu-sycl-fp32-x64.tar.gz": "21a1ad0145967181570dbac553568b74930eac86cdac678b497d3dc6dec564f6",
    "llama-b11541-bin-ubuntu-vulkan-arm64.tar.gz": "cfc991be4e09135ab00870582cbb0f403b48683b9f5d33d547ee327a2a604c78",
    "llama-b11541-bin-ubuntu-vulkan-x64.tar.gz": "bf91ce14ddfda55c01c8b8846aa7d6093a07b0213913b6216ed4a2d6ea78b1f8",
    "llama-b11541-bin-ubuntu-x64.tar.gz": "36ca310be4405acb7ba59b32dbdce32bb9cd7b1ef95358997b7eaed2c6f0188b",
    "llama-b11541-bin-win-cpu-arm64.zip": "8f112d39c0bde44451ae5878709fdf2196e8d680cb8cb9354ca5332337f0942b",
    "llama-b11541-bin-win-cpu-x64.zip": "cdc0535d11038bb337dfba3c15682050eaa9c07aedd0f915b4c09e8a8e8c0a5c",
    "llama-b11541-bin-win-cuda-12.4-x64.zip": "c5ccdd22e6b11c013efd6a931f93c3707e6fa5ab2f90df5c7d1838785540ece3",
    "llama-b11541-bin-win-cuda-13.4-arm64.zip": "192f64a3516dfac9f872c535f5f4335f399a0009e5a51bc1299c5a5b57fcf40d",
    "llama-b11541-bin-win-cuda-13.4-x64.zip": "927672eae5bb9cf4dde89a00a764c33e69e98db4b3d8be42f8990bf383cfdf61",
    "llama-b11541-bin-win-opencl-adreno-arm64.zip": "0ed2d450453b52505a9340f17ab2b88fcbe15f173281dc5b123c2256ad9a7b24",
    "llama-b11541-bin-win-openvino-2026.4.1-x64.zip": "f06d01c12b011b2b15ab251549115d18b6e95ae2f21698f50633dda0788f9aaf",
    "llama-b11541-bin-win-rocm-10.0-x64.zip": "cdcb0b4482fe80ede8c561e9e52406a8ea7d3b6a6586b77f5a9564882ec32ce5",
    "llama-b11541-bin-win-sycl-x64.zip": "abc9b144abd8e869edaa06d9ef79ff8f2c1bb3471405fc8189220851fb14e21e",
    "llama-b11541-bin-win-vulkan-arm64.zip": "c31124767a0c61dc78e021f574dfb2d0b5a8a0837525a72bf0a0dfacc1ffcade",
    "llama-b11541-bin-win-vulkan-x64.zip": "37bba511d120222f13c3d520dd875f739a530cc124e78f7cbce0026de0173d8a",
    "llama-b11541-ui.tar.gz": "e2da5841a74bda189fe22165ca0a243141b99df85a82484173adbcf152e0c073",
    "llama-b11541-xcframework.zip": "c6635419d2527e2a2fd052b6756f14c3e0e84c346e3638a43b305434e6753dc4",
    # tag b1342 ROCm assets (llama.cpp 71ad0590f480, ROCm 10.2.0a20261008)
    "llama-b1342-windows-rocm-gfx103X-x64.zip": "2236d8d3074e570e871dfbf03b77f031c18df6a96ce078cba7b1ae37047006d5",
    "llama-b1342-windows-rocm-gfx110X-x64.zip": "2fa2904d323fa785c19bda1844dc366db126339b7f775666b7fdcf6de1a9cbe8",
    "llama-b1342-windows-rocm-gfx1150-x64.zip": "890ba3afe24337a069d59e927d9e5d1904170dee74e4c9150c19cdd106640448",
    "llama-b1342-windows-rocm-gfx1151-x64.zip": "553a722e20bdd126224ad1c66ef023f9f0508d0d9121978719fac10c3af0acbc",
    "llama-b1342-windows-rocm-gfx120X-x64.zip": "fab097799c01f5b1dfaa04d562d9ececd1a7655329624d6a3afaf88d6cd6935c",
    "llama-b1342-windows-rocm-gfx908-x64.zip": "bfb125f6f13bc9aa6a507107465529a3e64fb539638002cfad0eeee516efffbd",
    "llama-b1342-windows-rocm-gfx90a-x64.zip": "b90fb996ac54c837616f7668083185575a01531a009332425a252c15549dbf0b",
    "llama-b1342-ubuntu-rocm-gfx103X-x64.zip": "8bf462b1a328b4d203bc4d76b90109dca57e21b32ce9dc4bfcce361671a9ca2d",
    "llama-b1342-ubuntu-rocm-gfx110X-x64.zip": "e7111fbde1c9a39f0a67dc37be11f3f809da835769312cd1d9d4e65ea122719e",
    "llama-b1342-ubuntu-rocm-gfx1150-x64.zip": "616eca00da9a8f0d42d3cb654dadedbccf148ae9bf993a06732c03b3d76a9f7c",
    "llama-b1342-ubuntu-rocm-gfx1151-x64.zip": "726250806fb8ee008c1e054c7ce17f2c0004771461af67f2c1783d3f43c46b59",
    "llama-b1342-ubuntu-rocm-gfx120X-x64.zip": "4910ccea06736eaedf900581caeeef3c997c8f5e1770b67cd4f8176b6bd5511b",
    "llama-b1342-ubuntu-rocm-gfx908-x64.zip": "f5527fc010420927349aa2aae538a43647e3499dc6097bc6aca383de6daca1d1",
    "llama-b1342-ubuntu-rocm-gfx90a-x64.zip": "454809a6ec52cc96d4bdc4b64db86a73a3baf4fd5964ac061dad223e803d987d",
    # tag b11513 upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm
    # build's SIMD CPU backend
    "llama-b11513-bin-win-cpu-x64.zip": "34166ebb2593b31b7f644a6e3fddaade42716fb4a12bb5bc3a6650ddc5ae1e39",
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
            "cuda-13": ["bin-win-cuda-13.4-x64", "bin-win-cuda-13"],
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
