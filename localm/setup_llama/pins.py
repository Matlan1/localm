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
_PINNED_TAG = "b11928"


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
    "llama-b11928-bin-ubuntu-cuda-13-x64.tar.gz": "eb28b1825c6dc31f4999c3f014349c719098542707b9c02469974f3222d30160",
    "llama-b11928-bin-ubuntu-cuda-x64.tar.gz": "30f123e8cf6857e1d899652ddd68cfd8f1c9e44e93f4b2484244a9262fe99b3c",
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
    # tag b11928 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b11928-bin-ubuntu-cuda-12.8-x64.tar.gz": "2855beb2ab7553eb5473c6d464ba42ae7ae93f224d96869b898565b3f9c5af47",
    "cudart-llama-b11928-bin-ubuntu-cuda-13.4-arm64.tar.gz": "c5df9b8aceb413b25999dfc6075dc88aab7fc21bd37a5152865362104f6ecd43",
    "cudart-llama-b11928-bin-ubuntu-cuda-13.4-x64.tar.gz": "6d8e7662d59c91823383c1df62955b56c82429ad522eac447524a5bdd8e8800a",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip": "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11928-bin-android-arm64-snapdragon.tar.gz": "5c1fc76eca6c879d56c8579fa48e763108ef514ea710e3fa1d1a72c6a1e4fe4b",
    "llama-b11928-bin-android-arm64.tar.gz": "40d210628f0821f015576c4c1013a9a843c5379bf309f03d3f8b3d2355c8236d",
    "llama-b11928-bin-linux-arm64-snapdragon.tar.gz": "944a415484c0dd2c85f1da9bac45d3ab5c1175f089a54f1ac77d65cf156e1d20",
    "llama-b11928-bin-macos-arm64.tar.gz": "eaccccd39f670049506377834a341c59c88c3b48ac3a9e6a5b6166152a9e32dd",
    "llama-b11928-bin-macos-x64.tar.gz": "095085a85dd52bffd11d2fbc3fa2fa61315905bd18ecc0444107ce1fe4fd3e98",
    "llama-b11928-bin-ubuntu-arm64.tar.gz": "2019f0d926458d4a433ed4223ce8d6e035283ea8b51acc72819225c2c65f5168",
    "llama-b11928-bin-ubuntu-cuda-12.8-x64.tar.gz": "5259e1da1da6957808b0172d879a514f4929e8e9ea7663b7deb8283ffb163370",
    "llama-b11928-bin-ubuntu-cuda-13.4-arm64.tar.gz": "3e50da032b4542d423b82ff54c62a85c11d67bd9561cbcd2fb275b7f13d95a6c",
    "llama-b11928-bin-ubuntu-cuda-13.4-x64.tar.gz": "6eef5f6f9f690049882018c80c3b784c6c1a1d663e7d0514bf8bf3f15ba9535a",
    "llama-b11928-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "9cfd5b0c9d7831f1c4058d2e7eb840af2b2eb7c5b4f740b1fa52c15208ededc7",
    "llama-b11928-bin-ubuntu-rocm-10.0-x64.tar.gz": "c6f73527c7c728824196a3ce721cc408d2e846fdbbf78d4084f6a3d2a65b5b0d",
    "llama-b11928-bin-ubuntu-s390x.tar.gz": "d7d30d7fdb210dca899c0ebc2595d51305e0b4f31bf2d6203cfe3371126f162d",
    "llama-b11928-bin-ubuntu-sycl-fp16-x64.tar.gz": "25747ed69d67eb1fb48d982051ef00d8e791c1537ad8c167b875d0c40002ee8e",
    "llama-b11928-bin-ubuntu-sycl-fp32-x64.tar.gz": "8e26b91589432d5293f7773588b0784322765d5ee152c372170d0fdd08b0f87f",
    "llama-b11928-bin-ubuntu-vulkan-arm64.tar.gz": "26fe079a5fbdcc0b853b41e0217427ccb3def841679e06b232aadf0249b9a1bf",
    "llama-b11928-bin-ubuntu-vulkan-x64.tar.gz": "80b04a7ee996607629db79998036111442ac08f3b1101ac736f8dd05334078d8",
    "llama-b11928-bin-ubuntu-x64.tar.gz": "c4fdc4b9913c3fc7b0ede9ed1970a43271aff7366889444797e05472f7c365da",
    "llama-b11928-bin-win-cpu-arm64.zip": "7f6080ce914b9944acd83d7bf41b693e037c507ddacf2e2c0dab8c08a33a2f47",
    "llama-b11928-bin-win-cpu-x64.zip": "5e1395524a2a75161fdebbf951d26cc04ee5f0b2c6d8d680cfb88812762caac0",
    "llama-b11928-bin-win-cuda-12.4-x64.zip": "50262bccb5e5072b02c274037555892fda6ef98819728652420c83f9cabd6a90",
    "llama-b11928-bin-win-cuda-13.4-arm64.zip": "629e4785781796f63d7fbbf0ef00a1ee2ac2425eaa892ae94ad9928a260ed86d",
    "llama-b11928-bin-win-cuda-13.4-x64.zip": "7409f3aa17c5344bda524126e985b31983bfba2bb0d08cbb2aa89e8ecd844daa",
    "llama-b11928-bin-win-opencl-adreno-arm64.zip": "8434e02d0a84df743f130c44ee0a9708ccd532a7e6ca616397d0979e1ad038bd",
    "llama-b11928-bin-win-openvino-2026.4.1-x64.zip": "9cde4dfccdda549efc173af8e5d6f79736a0546d74720306b8b7e848d1c746ee",
    "llama-b11928-bin-win-rocm-10.0-x64.zip": "ac9b9a76062d12ce2bba3ab977430fb974fb6f8182a15d1a13eaa1ca7277d5fb",
    "llama-b11928-bin-win-sycl-x64.zip": "e6bf50cf07dfd445c01e7acd724dee5cf79410ef1adcccf41adfa76afaae9537",
    "llama-b11928-bin-win-vulkan-arm64.zip": "e14c280d181a965f33776875698113e93d7eef83a5abec81d813e362ddb01985",
    "llama-b11928-bin-win-vulkan-x64.zip": "e090b3f6a91a2facf80b443883484b24b0e74bea89175a18404feaedaeb5f15a",
    "llama-b11928-ui.tar.gz": "e3e2ed635a45fec03b5557386863a1b161f176e97200629d9e3b4a64d3858b7c",
    "llama-b11928-xcframework.zip": "5895df91a6272739338d3ae1f1ca866443da7c03b2981f87a60c9db2da4e49c8",
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
