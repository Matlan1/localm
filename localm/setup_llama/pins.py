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
_PINNED_TAG = "b11118"


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
    # tag b11118 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b11118-bin-ubuntu-cuda-12.8-x64.tar.gz": "5b8c10e8cbefd2983ab30275d19116416247de8d7f333ce6359f01989062eaa1",
    "cudart-llama-b11118-bin-ubuntu-cuda-13.4-arm64.tar.gz": "1afa94bd4fd2d654e4d4550821d83eaf50b5d5ad9bf79535a07b063b2b0b0c99",
    "cudart-llama-b11118-bin-ubuntu-cuda-13.4-x64.tar.gz": "075fdc0fba066151259906d5d9c93922ba8754be1c3cd3087cde2f01821746d9",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11118-bin-android-arm64-snapdragon.tar.gz": "0cc858c0540aa689dff71d4a36c1c7419de9c121647d28ec28b5c4dd6480d36d",
    "llama-b11118-bin-android-arm64.tar.gz": "3bd8261654a9dc40d4bf29bfbf0e20d5de968b576032339610248235bbebbd45",
    "llama-b11118-bin-linux-arm64-snapdragon.tar.gz": "9b321cc24c6e1804b45762121e46d48ea30d947a155e2d5648059c164db385b8",
    "llama-b11118-bin-macos-arm64.tar.gz": "ca0ea3156257b21eeb11d0628f2baecd3928013a3d060e2e192042276e5b1f35",
    "llama-b11118-bin-macos-x64.tar.gz": "e80340da2dadb736405261d5203b747ee7101bed8bfba5ba99946e77dab91844",
    "llama-b11118-bin-ubuntu-arm64.tar.gz": "8ff18aee896c11c041c1f9c04954f530fffd991128f6c73e75b85517a29620cd",
    "llama-b11118-bin-ubuntu-cuda-12.8-x64.tar.gz": "31a47d8785e62ca00b36437ab108dafedeecf85db8d328e9bcc99de531cba285",
    "llama-b11118-bin-ubuntu-cuda-13.4-arm64.tar.gz": "7487c7de15330666da813ca6f51fb68b39891e93fe8f606fde5cbcf973f892b6",
    "llama-b11118-bin-ubuntu-cuda-13.4-x64.tar.gz": "a8fc7d01500c8de5dc7b4e1dad70408446ea7f3ee854df091892d640023ea7ac",
    "llama-b11118-bin-ubuntu-openvino-2026.4-x64.tar.gz": "f2b7bf700efc26197ea0ea2d27ef8e5ae14c086ff8adef01e2228516188b1075",
    "llama-b11118-bin-ubuntu-rocm-10.0-x64.tar.gz": "38096210f4df755bb9fc885c2f1f46440489c0993b2c6e8932ba40d41c13f58f",
    "llama-b11118-bin-ubuntu-s390x.tar.gz": "ea1e4bced40dbbe990b5114c8a1672d91ff1abec35f6d7c487cb46842541ed51",
    "llama-b11118-bin-ubuntu-sycl-fp16-x64.tar.gz": "196af0b89383243a015dfb09d44a12a3cbec951dec4dbdacbedff650b4045fb4",
    "llama-b11118-bin-ubuntu-sycl-fp32-x64.tar.gz": "b87a72dc7fc2c136ec89246040d7708f39c52a7e55d0e2dd66db4375c0531aaa",
    "llama-b11118-bin-ubuntu-vulkan-arm64.tar.gz": "8487e456407e6e32760aa6bdfc99f3c8d81bd2d1b9702b180ee71ebe81ad774b",
    "llama-b11118-bin-ubuntu-vulkan-x64.tar.gz": "145fdde715c1e9f4808f1fe0749944149c91713a9d681fd00818f87b6cbdaccf",
    "llama-b11118-bin-ubuntu-x64.tar.gz": "20f3067d5bc7e48c2be49ada32497a08dc75c730d90fce852d1d7cf12fe36995",
    "llama-b11118-bin-win-cpu-arm64.zip": "c6efe37936466e568745843b9860f221798da001026d67be9ec2296193583e70",
    "llama-b11118-bin-win-cpu-x64.zip": "7f8431c69471cf8991f43da4af6e80f3a66778a63055004a9211ebba00d68084",
    "llama-b11118-bin-win-cuda-12.4-x64.zip": "09d262cb22c26d276a8c2cd38e0ddb405404c4c60e355c2ed96cda8e77538c59",
    "llama-b11118-bin-win-cuda-13.4-arm64.zip": "ccdb4bd815e12717ccd3d31b5805732a540ef2beafb44a3c2efb818b445450b8",
    "llama-b11118-bin-win-cuda-13.4-x64.zip": "5825a03f9360e5aaf50817f32c408cfbb33b55f451c78c513b637678a541eb31",
    "llama-b11118-bin-win-opencl-adreno-arm64.zip": "004372a04d2fefd7fcc48534a58af20ded52d265f8a0892061d1ce946ad7c830",
    "llama-b11118-bin-win-openvino-2026.4-x64.zip": "f83333fa5a9a9013b619391794b6bb1b1074be11516238f70c4ab94dab2306b9",
    "llama-b11118-bin-win-rocm-10.0-x64.zip": "9bb51f9fa4ae28064ecacfe1d081481133ca273f11eda3b7142d73d5ffa2ec67",
    "llama-b11118-bin-win-sycl-x64.zip": "8da355712f8065e25912fba8088025645792dd3cd83075739d9808da03abfe66",
    "llama-b11118-bin-win-vulkan-x64.zip": "5cb80f42a602965f7491cac4977329ff47c6aa4f850272406708854b5d290547",
    "llama-b11118-ui.tar.gz": "d5266c1f2d8e896251021655d10580eed48dcc2162a427bd990545a01581c390",
    "llama-b11118-xcframework.zip": "0ae8f4f39b50c64f224e8cfbbd3e5216398a7423a52161bbb52ac9757afb8855",
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
