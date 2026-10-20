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
_PINNED_TAG = "b11756"


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
    "llama-b11756-bin-ubuntu-cuda-13-x64.tar.gz": "7736a018d7db18f15829ef6798947a1895a31531dadfffdb38fd86d5c769704f",
    "llama-b11756-bin-ubuntu-cuda-x64.tar.gz": "a6354d5784d0dae270068884703cb3ef8bb71e2d8c8a30259d4d2a0ec077c196",
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
    # tag b11756 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b11756-bin-ubuntu-cuda-12.8-x64.tar.gz": "24955c1ba560b2e418880b611c3e4b093c7be738a8ca9ced501ade42c41cf167",
    "cudart-llama-b11756-bin-ubuntu-cuda-13.4-arm64.tar.gz": "75066f0c85127d0136700ba107adfe6ce58052c30e4f9a7941ee90230ff68e86",
    "cudart-llama-b11756-bin-ubuntu-cuda-13.4-x64.tar.gz": "96c9d683c7127fede1abaf53f61dffc55b08778d1d9a90d6687585070583060e",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11756-bin-android-arm64-snapdragon.tar.gz": "b6f08d2595c1395c1b33ccf68ad5a2836ae4fd89aa7aa7ad4c494051cd6cb70d",
    "llama-b11756-bin-android-arm64.tar.gz": "ee8ea11d367b4118e5468dfdc243b95a6a793d1d6e4b1158034ce0e3362c119f",
    "llama-b11756-bin-linux-arm64-snapdragon.tar.gz": "2036a34fffccb4dcaa15087b598fbd0738cd46c87e7143293897bb79b07cb838",
    "llama-b11756-bin-macos-arm64.tar.gz": "f86b5eb838b985a17587e794dee9c6b34193aca6dceec018031a4ba7951fb7e6",
    "llama-b11756-bin-macos-x64.tar.gz": "6d71968268a01ae975acde18441a70b2c7ba128b7a127f46b5ced4e8c5dd4eca",
    "llama-b11756-bin-ubuntu-arm64.tar.gz": "42bb5ef275743e0183fa53e3c93b2b54796997d49a9d0d540d0a8cc945f1ed43",
    "llama-b11756-bin-ubuntu-cuda-12.8-x64.tar.gz": "a9047502a1a2896a4d087c1ae3c3cffe59c7157448648a5a2901ed42fe9c7add",
    "llama-b11756-bin-ubuntu-cuda-13.4-arm64.tar.gz": "0adc2e7bc2a5928ae5e38b5dbf8568c094d354d1bba0d592ad72196b2dff17c0",
    "llama-b11756-bin-ubuntu-cuda-13.4-x64.tar.gz": "3412f04f8da2e6c7b2aaab4ff1e02dcc0eb227fb19fce43406f9949457d489b6",
    "llama-b11756-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "0f6b5ee7709e7f9ca4e0bca5ce52143fb3cbffe165d606021c3a2d8e3043e348",
    "llama-b11756-bin-ubuntu-rocm-10.0-x64.tar.gz": "75fd3f82b30686570bd7efa5b2954eeffad0fb3d766545c496fc7b1bd9d46fc6",
    "llama-b11756-bin-ubuntu-s390x.tar.gz": "2561ff0e551e400a99b1b31c4ad7a091125c2138447a0fb350a3709fcf07ae75",
    "llama-b11756-bin-ubuntu-sycl-fp16-x64.tar.gz": "da3dec7da4ca48964eb06f3cc83fb085218d94e8c5b9b0fe9932ab339e4616a7",
    "llama-b11756-bin-ubuntu-sycl-fp32-x64.tar.gz": "cc8edf85f6d87d3bf793c431502ba436b33396c61b59e15b143b341e42f3b40e",
    "llama-b11756-bin-ubuntu-vulkan-arm64.tar.gz": "bbaacf9fbe9e565bb3a2c437a91de49bc02305cebc58bb8898a3cd1d80b86910",
    "llama-b11756-bin-ubuntu-vulkan-x64.tar.gz": "7b23a9192e013ee8cf2011c192fd6a6197a41b562ff1b1a8e41f523a933d75cf",
    "llama-b11756-bin-ubuntu-x64.tar.gz": "9def57d9e9b0149979a9ceba40896a4e2bf88c94734069a5567d2f487ceab551",
    "llama-b11756-bin-win-cpu-arm64.zip": "ba47a12a87d45a3a717d717e81334e65b725df051c8adf8f68accd824d05a126",
    "llama-b11756-bin-win-cpu-x64.zip": "d164195b8f3e92f8fd92bc3a5c0eeaef8707b2f1a979e36360ce8192ebf4fbb3",
    "llama-b11756-bin-win-cuda-12.4-x64.zip": "bfb275464b8ef633a12d96f454e5fcf2efc642d926043208ce95d33449dad0e0",
    "llama-b11756-bin-win-cuda-13.4-arm64.zip": "3cad67a2d2405469125a2a66febed5e290a96168ae836cfb2c11e0b3b8b0269d",
    "llama-b11756-bin-win-cuda-13.4-x64.zip": "96cba974f5abfd5b84fd46aafff401a5c0a65dc4afadcc989ed81fd7eefb5741",
    "llama-b11756-bin-win-opencl-adreno-arm64.zip": "46444637cad921a21d3b28a9caa64c11ee46e26e942f994e225294b5c5cf0c35",
    "llama-b11756-bin-win-openvino-2026.4.1-x64.zip": "c126233f411d7c912cfcf6b1dce5d21592faad5fa53648e3d4bf2adbb3ac217c",
    "llama-b11756-bin-win-rocm-10.0-x64.zip": "5f6afd706dba657cc195561548a5893ff6793b487d6f232bdfc00b0d92764ae5",
    "llama-b11756-bin-win-sycl-x64.zip": "bc8b34551104097fa35f398768f034fd8a8b030e7cf26e0434b6b74cce2ace7f",
    "llama-b11756-bin-win-vulkan-arm64.zip": "3ea056e7f3f0eee0a58c9ea1b4e3ecba259e8433afb902bf9862f45e0eb4fda7",
    "llama-b11756-bin-win-vulkan-x64.zip": "75ce69fdda503c3d6d3b3401f61369bd072194746ec65fbbc5fd7e749edd2914",
    "llama-b11756-ui.tar.gz": "5e0a7895409e8695157384a5de182372ef892c5f40cfb3d43e75bed1ee77c56b",
    "llama-b11756-xcframework.zip": "0198d534fe4d0661c500ef9f828c92e181735db32f835316a08af7db4c2b10f3",
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
