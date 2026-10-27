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
    "b1355/llama-b1355-windows-rocm-gfx103X-x64.zip"
)


# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "a13eef0c9b967dbe2b71132de96c1f1d9f0102411b82ed8365a5d64267e454db"
)


# The lemonade-sdk release tag DEFAULT_URL points at. The struct layouts it
# carries are bound by inference/backends/llamacpp/_structs.py and _abi.py,
# which bind every layout from lemonade b1288 on, so an already-provisioned
# runtime keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones -
# the two schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1355"


# The upstream llama.cpp release built from the SAME commit as _ROCM_TAG. Its
# Windows CPU archive supplies the SIMD (AVX2/AVX-512) ggml-cpu variants the
# amd-rocm build lacks: that build compiles its CPU backend with every x86
# instruction-set option off. The variant binds to that build's ggml-base, so
# this must name the upstream release of _ROCM_TAG's commit. See
# localm/setup_llama/rocm_cpu.py.
_ROCM_CPU_TAG = "b11908"
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
_PINNED_TAG = "b11757"


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
    "llama-b11757-bin-ubuntu-cuda-13-x64.tar.gz": "8df9af7001a33aeddaef1dee5f50badc472d7617a747c3ca72898fcd5a6f5a7a",
    "llama-b11757-bin-ubuntu-cuda-x64.tar.gz": "361e8171d51572cf55503a1e80155d36cc1d0eabe459c7d3b204b64d1a19cb4a",
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
    # tag b11757 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b11757-bin-ubuntu-cuda-12.8-x64.tar.gz": "d156bdf88cd88def090fa535023bf4e5b05bcef4ed0b8b7ba22e0df537e89264",
    "cudart-llama-b11757-bin-ubuntu-cuda-13.4-arm64.tar.gz": "76ae2532804df20479efd367299ca4fe242c847c61e7dda782d008f189442595",
    "cudart-llama-b11757-bin-ubuntu-cuda-13.4-x64.tar.gz": "d6397b518a7ca6546879fe345e5597cb26bb5636679c54a97d8d7bed585a419b",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11757-bin-android-arm64-snapdragon.tar.gz": "bc9f18f97c5988ec4ceed9aaa1703d97e489d528e9f797cd7882e9aecf2787e9",
    "llama-b11757-bin-android-arm64.tar.gz": "d84b498bfbe9b8d339c6cfce1affe0c2598da168a5a3d67b49aa37374b385651",
    "llama-b11757-bin-linux-arm64-snapdragon.tar.gz": "bc977b5d1193a08a81623c03790ee1cea2d1d3d41e8ddc61e5bd1df9a38c7ce3",
    "llama-b11757-bin-macos-arm64.tar.gz": "7213e3ad873881455e4535d8167ad5aa49262c9c6d3acd2e262974785e94156f",
    "llama-b11757-bin-macos-x64.tar.gz": "e14f1f76373cc064706fb509c83795ed2b2c0792e93cc64a204da4aa3bc75d0c",
    "llama-b11757-bin-ubuntu-arm64.tar.gz": "3948f9469b51270b958e359e46107b6ca38a08c53f556caef3021ca85d0f9b89",
    "llama-b11757-bin-ubuntu-cuda-12.8-x64.tar.gz": "68e71a1d02b8d3b3b9ae9fb0bd1a65f820781005f60705e04d8c0690ec07129b",
    "llama-b11757-bin-ubuntu-cuda-13.4-arm64.tar.gz": "d3f4a6c3c715aeb6263e0fc7c80a65246cce3efb82c503c8b53c166cd7a9660c",
    "llama-b11757-bin-ubuntu-cuda-13.4-x64.tar.gz": "0f21f8f57a6ea565aaf215e180c4ec4d992d0a849b4e3996ce92269efd34c72d",
    "llama-b11757-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "adb164aeeed5797e979671228397de7641369584a1665e967e32a71da3b0eed0",
    "llama-b11757-bin-ubuntu-rocm-10.0-x64.tar.gz": "d9e68392517725289f940d65f2f65c7684e1b61cc5beddc62988244bf4dba9b1",
    "llama-b11757-bin-ubuntu-s390x.tar.gz": "a151c81ffd62986647dc231fdbb3f01511f9bcc22bd3fb53f2b142a77319a70b",
    "llama-b11757-bin-ubuntu-sycl-fp16-x64.tar.gz": "c43190b922b106269a030953e599b7798e7bcdec432e1b9692e319208171b62c",
    "llama-b11757-bin-ubuntu-sycl-fp32-x64.tar.gz": "32de7d83db56e65b316bbe70658beee524257dd66c0ee343eaa8d55181960507",
    "llama-b11757-bin-ubuntu-vulkan-arm64.tar.gz": "0f56227cfd23e40c580fbbaa1f28b072e0bb87d367d9dce23495457686f5d883",
    "llama-b11757-bin-ubuntu-vulkan-x64.tar.gz": "1fa24f064850d256720db56b9ef16d6313bb3275125473ff84edbbeb6a4bbb87",
    "llama-b11757-bin-ubuntu-x64.tar.gz": "87aae0ada6ace2aa82e098994ef92ed76677b221941ce940c83757a89f2b1a6e",
    "llama-b11757-bin-win-cpu-arm64.zip": "25b9e352d4b296c50de8b08cd59bbac5da4bbc40c611787c6443e7445d6447aa",
    "llama-b11757-bin-win-cpu-x64.zip": "d4e4b86bf125716e26514ffb20605aff2e7628f5dc063211fdf57ac7c150f797",
    "llama-b11757-bin-win-cuda-12.4-x64.zip": "43ba04a99c4bea739bdd9830127250280b3c5065aeb8aade047c1917795aea3e",
    "llama-b11757-bin-win-cuda-13.4-arm64.zip": "2ef88d2210a160a0eb09b637be77113847f9b8989d515ae3103d93f63b383f78",
    "llama-b11757-bin-win-cuda-13.4-x64.zip": "38b117014c385bbaa330581929c8f7b59d47f6d66ab44d41affc20088b8fce0c",
    "llama-b11757-bin-win-opencl-adreno-arm64.zip": "059c3cca74262067dd8d3ba344ae89e685ad02d2d750405c240528dc96934818",
    "llama-b11757-bin-win-openvino-2026.4.1-x64.zip": "295c42c8692051baf21ad5586cd7d64c2f2e0dc2515adf2edc9e40e43805077e",
    "llama-b11757-bin-win-rocm-10.0-x64.zip": "a98d874fff3a5c817e3af32902d42447671556c722234c7fa813dae6c48234e7",
    "llama-b11757-bin-win-sycl-x64.zip": "df0fd0f35b7c3e558c08cb8e2f6b27f839cdd67aef7cf72c25a243774e7c2d40",
    "llama-b11757-bin-win-vulkan-arm64.zip": "f8a0b7fc7a50665da909ab92d90cb946fbc4429c86c08bed50b54f4470b81fc2",
    "llama-b11757-bin-win-vulkan-x64.zip": "5561afe6522d9b9f3cd29c4712e56f9e262716abfb17f7e11c3a832d134c4467",
    "llama-b11757-ui.tar.gz": "d87bef7fe5771feee04ebb051dd7206dacb410fe2ab64e42c4f103343c4ac5ce",
    "llama-b11757-xcframework.zip": "a6ab02056b7a83b5794fb6d0a6c70eca5cda05fe3465da00f0deeace4888d812",
    # tag b1355 ROCm assets (llama.cpp a4699400e3cc, ROCm 10.3.0a20261026)
    "llama-b1355-windows-rocm-gfx103X-x64.zip": "a13eef0c9b967dbe2b71132de96c1f1d9f0102411b82ed8365a5d64267e454db",
    "llama-b1355-windows-rocm-gfx110X-x64.zip": "ed9bda05eefa9d0f7232ca8896ff740a95e7ebad6cb3ba0610807fdc4f3174e1",
    "llama-b1355-windows-rocm-gfx1150-x64.zip": "bb3e413d627dcf112e88702113ec7a307dd6d57dd495e49c9baa3faeeaa62838",
    "llama-b1355-windows-rocm-gfx1151-x64.zip": "112815553a187951c9f21f0e50eb8374133f87bf5bebfa9ddcff07cd2daa04a3",
    "llama-b1355-windows-rocm-gfx120X-x64.zip": "fb045a6e08fff89d4183b1768f7d8c3bfe6e86a61bdd4cfbd6f2de66d9f55e5a",
    "llama-b1355-windows-rocm-gfx908-x64.zip": "1434c4b41f7c763bbf4965f875bd06bd170cd5b3ddbfb255183b6880ef9aa3d2",
    "llama-b1355-windows-rocm-gfx90a-x64.zip": "cf6d14f68547980fc4940c312ba11eeaf5b8e3757004ef4b4a855616b7117972",
    "llama-b1355-ubuntu-rocm-gfx103X-x64.zip": "0115169fc6ce747793b9a197fa003b6188f9519c6e805b525078983e7da92ecb",
    "llama-b1355-ubuntu-rocm-gfx110X-x64.zip": "5fde0155a6bff051908681555d30db51f3da1ccdcd9f053b81cacc2b603856b4",
    "llama-b1355-ubuntu-rocm-gfx1150-x64.zip": "a8f36382dd336f3d07f67f1dfb213107c9dfb1040b85d4dbac14357883cf1b40",
    "llama-b1355-ubuntu-rocm-gfx1151-x64.zip": "430124c28ff6f580f42d6d14ee54dd36b37dd98cdb8c90a7ab4f1e0a58be77de",
    "llama-b1355-ubuntu-rocm-gfx120X-x64.zip": "f7c10f00299c82372f46effb606ac20ae265bf9dc6ae7af3bb4be301d04ded3c",
    "llama-b1355-ubuntu-rocm-gfx908-x64.zip": "9801cef915fb38c95717f3ca378b7170b11c7dd311ccb2540770bc66120d2db9",
    "llama-b1355-ubuntu-rocm-gfx90a-x64.zip": "79cf4d5768e39215048a72a205f6937c05a92e31455ff3501b4fc9c97001993d",
    # tag b11908 upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm
    # build's SIMD CPU backend
    "llama-b11908-bin-win-cpu-x64.zip": "4f16d1da0b6d58d258002f10f25b79d484b776078f172a569453b73a068f0ed5",
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
