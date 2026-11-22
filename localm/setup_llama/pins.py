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
    "b1372/llama-b1372-windows-rocm-gfx103X-x64.zip"
)


# sha256 of the DEFAULT_URL asset, used when the release lookup is unavailable.
# Named rather than repeated inline so the URL and its pin cannot drift apart.
DEFAULT_URL_SHA256 = (
    "c9163fc8e7d1b5c70db707ef1ff821020fc3c74c38cc43fb600507690d28ec50"
)


# The lemonade-sdk release tag DEFAULT_URL points at. The struct layouts it
# carries are bound by inference/backends/llamacpp/_structs.py and _abi.py,
# which bind every layout from lemonade b1288 on, so an already-provisioned
# runtime keeps working. (b1xxx here are lemonade-sdk tags, NOT ggml-org ones -
# the two schemes collide; see inference/backends/llamacpp/_structs.py.)
_ROCM_TAG = "b1372"


# The upstream llama.cpp release built from the SAME commit as _ROCM_TAG. Its
# Windows CPU archive supplies the SIMD (AVX2/AVX-512) ggml-cpu variants the
# amd-rocm build lacks: that build compiles its CPU backend with every x86
# instruction-set option off. The variant binds to that build's ggml-base, so
# this must name the upstream release of _ROCM_TAG's commit. See
# localm/setup_llama/rocm_cpu.py.
_ROCM_CPU_TAG = "b12228"
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
_PINNED_TAG = "b12620"


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
    "llama-b12620-bin-ubuntu-cuda-13-x64.tar.gz": "584b840435efa9f41957158ef20de146a5cf28a0ebb74f75b4e75da946d3165a",
    "llama-b12620-bin-ubuntu-cuda-x64.tar.gz": "f2e5faa2789bda298f07b5fd5f6a5c08ec7a34ea58f244023af03779a9175d1e",
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
    # tag b12620 upstream assets (_PINNED_TAG). The three cudart bundles carry no
    # tag in their names and upstream re-uploads the same file each release.
    "cudart-llama-b12620-bin-ubuntu-cuda-12.8-x64.tar.gz": "b67527ca01ecaf97a57c582392b85e4dce44efcb42fb7ecfb664fe7382df0c5c",
    "cudart-llama-b12620-bin-ubuntu-cuda-13.5-arm64.tar.gz": "97ac5a470fd869fa688fa6be9401b1a28eb745db57999b230133608cfe9a63a1",
    "cudart-llama-b12620-bin-ubuntu-cuda-13.5-x64.tar.gz": "7c1db91dc7d5cdca60357e3201ad40d9d2eee7b6d2c2a5e4be414dd9ff06e0a9",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.5-arm64.zip": "f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca",
    "cudart-llama-bin-win-cuda-13.5-x64.zip": "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
    "llama-b12620-bin-android-arm64-snapdragon.tar.gz": "9703e4e0918ae2fde38d41247aadd83c0dbfe9466be09bd5c62feccbfc7b1572",
    "llama-b12620-bin-android-arm64.tar.gz": "386d567389de4d6d0f81eb4216780d475da08ee3bfde52925262f20442463d0b",
    "llama-b12620-bin-linux-arm64-snapdragon.tar.gz": "b5134784abd92945f8dbe9fe65f8956971f8e15e079d5e807c2705500304d688",
    "llama-b12620-bin-macos-arm64.tar.gz": "7e0c724681cbec4a62b95c23bfde3a341fcb3c15fff1b2f52ab8026f56812c1d",
    "llama-b12620-bin-macos-x64.tar.gz": "c9854c1504ec308d524d3a735145f3fffd4269d69b05bfb1e4607c6f932a4c9a",
    "llama-b12620-bin-ubuntu-26.04-s390x.tar.gz": "16c28621ec1725bad64b675c1a7e940f58ca46d976ae88ece919b43aa959611c",
    "llama-b12620-bin-ubuntu-arm64.tar.gz": "3027dfc592c6b4d88c237280b0551c3db8d45af947c836cbdc94a4ea3f275e8d",
    "llama-b12620-bin-ubuntu-cuda-12.8-x64.tar.gz": "6dcdbd5aacf63e91b4a11a10c2c3d63fa1a00b61e3468998cb016ecf1a2a1b46",
    "llama-b12620-bin-ubuntu-cuda-13.5-arm64.tar.gz": "147712866f0986a1e11041305b84ce6b319410d1435ee542ab2f92e5de77c6f4",
    "llama-b12620-bin-ubuntu-cuda-13.5-x64.tar.gz": "d00665f29ad7df994aee9221a6f733cd1788e0019d5c59e80731b881b607d104",
    "llama-b12620-bin-ubuntu-openvino-2026.4.1-x64.tar.gz": "62d10940e4b10fab4866859f97b0d30dd6b0a233415d5b422845eb41253dd6a9",
    "llama-b12620-bin-ubuntu-rocm-10.0-x64.tar.gz": "94a89fb20d0c2be439960db0c58be7eaeed396779934153c65d5a722425ef4a0",
    "llama-b12620-bin-ubuntu-s390x.tar.gz": "162b9349d448c45863019c8f38169c91160ca750a5d6afbc31799cf547cb8c09",
    "llama-b12620-bin-ubuntu-sycl-fp16-x64.tar.gz": "ab5099c1aba6cb3aaccf0c411b4d53eb8c783459a74d5f330505e78443c32f3f",
    "llama-b12620-bin-ubuntu-sycl-fp32-x64.tar.gz": "95008ffaf05e45ad70cb495eb2d4a11d0a236c8825e4f2fd5822575aeb29a71c",
    "llama-b12620-bin-ubuntu-vulkan-arm64.tar.gz": "eded4ded67b7c35a8f852a579689e63a6b38c08a447900a570fb4270484ce3e5",
    "llama-b12620-bin-ubuntu-vulkan-x64.tar.gz": "445c1aa4b8b90b2120692ed742f7999c0bbdb15da1d3da8de357ac574521b289",
    "llama-b12620-bin-ubuntu-x64.tar.gz": "74bb9abb909c85cc831a5ca7689e5478eaffd911705257f9e329ee2d37ae81e3",
    "llama-b12620-bin-ubuntu-zdnn-s390x.tar.gz": "a0bd1e5c781cc79022dd96813ef114d91c14f33c89d3f4d827861c1a08c2c8fe",
    "llama-b12620-bin-win-cpu-arm64.zip": "b237a81829acbf6fa0f7a1b74878a12d6c4b5fe98d41665c8271c2264649d722",
    "llama-b12620-bin-win-cpu-x64.zip": "7a7b5359b7af7704c1b23d442eb537cdad0214c1ad55bc861038d6057c7dbf9b",
    "llama-b12620-bin-win-cuda-12.4-x64.zip": "5c972d389313227bcefa90b281363f8c0ad3947d3fbcfe5b8f50f826d7ae61df",
    "llama-b12620-bin-win-cuda-13.5-arm64.zip": "92a1cd19eaeba4b55838f71962211020b0a0404c0ddc93b7275a9fd1fd5b0eeb",
    "llama-b12620-bin-win-cuda-13.5-x64.zip": "7a9bf4fbbfbedf63b17aeaf9989e9a62ba3abeb06aee8e7fb65dc43f1f56cf56",
    "llama-b12620-bin-win-opencl-adreno-arm64.zip": "44123c7bb882d5b5521e7941d82565368b683ce50c98a7691492a33b8a145d4b",
    "llama-b12620-bin-win-openvino-2026.4.1-x64.zip": "b3cc0ed5bab049f079da0481c644db72e1f54b92f32792043054751257ed2232",
    "llama-b12620-bin-win-rocm-10.0-x64.zip": "1de0e1c8a3465cdc8ff7c5a66180f1da638898432b8d6319578cb644d22a2f60",
    "llama-b12620-bin-win-sycl-x64.zip": "84cdbb6fee221c6107019cc41b042987adcbaf5309b93a3717fd3c56dffcf9f3",
    "llama-b12620-bin-win-vulkan-arm64.zip": "7a681a10971cbf5887f3211d9ef733c4c637c7b11a70e811ff31bb5b4da21520",
    "llama-b12620-bin-win-vulkan-x64.zip": "eb3fb8dd7767da9b9826e378987fa36bc514adc850c4af78b86b5ff553dbfaf9",
    "llama-b12620-ui.tar.gz": "6cbfa8a568305715e851288b3c6dee48f98a70e52c00b52772321cd78b40b8cd",
    "llama-b12620-xcframework.zip": "13831f866aaa33ac768b9c6e1adea34b04eac96964a14c4097e1d243732efd24",
    # tag b1372 ROCm assets (llama.cpp 0eebb6eddcd5, ROCm 10.3.0a20261109)
    "llama-b1372-windows-rocm-gfx103X-x64.zip": "c9163fc8e7d1b5c70db707ef1ff821020fc3c74c38cc43fb600507690d28ec50",
    "llama-b1372-windows-rocm-gfx110X-x64.zip": "48ba544e013ff484c689a0affdede60e559e320e9a7ff31ab96e30d7e6dc12f9",
    "llama-b1372-windows-rocm-gfx1150-x64.zip": "d5046d0877ff19f1b77332bd7049c5aff46fafac9cf5b9403a7869b5e95d52a9",
    "llama-b1372-windows-rocm-gfx1151-x64.zip": "85a7e0a81697a3ef9cb092afa82ab88f304cfb69ee31906475985ca5fdebcc83",
    "llama-b1372-windows-rocm-gfx1152-x64.zip": "cea57137645927d62d84427eb3548184ff41e193dd5ac6f8ff354b16ce47d7cf",
    "llama-b1372-windows-rocm-gfx1153-x64.zip": "dce39ba3885da44468e06339b88b05109c18365425fc8a0a7ee8ac113f2542c1",
    "llama-b1372-windows-rocm-gfx117X-x64.zip": "123ae16896dd661fc07be719d1b4b8759cdef4246a1cddfe3c15838ae863c97c",
    "llama-b1372-windows-rocm-gfx120X-x64.zip": "a98bbeee2e130ccffe73bea94492e097ac1dc753dcd7b7c1a42479dbe29ba897",
    "llama-b1372-windows-rocm-gfx908-x64.zip": "4d952106ba3d8a976890c5b471c70ea9b342a60cf2a69f6a3bf8415c51d52c85",
    "llama-b1372-windows-rocm-gfx90a-x64.zip": "b510030c3ca5703eb53efd57f423906c4c9c1bc6a2d6a862a0235a8fe6129210",
    "llama-b1372-ubuntu-rocm-gfx103X-x64.zip": "6d8e9d835e871a6beeab8b942b4652ba2ff1cc872f6ed2af065ce6311f1b8d61",
    "llama-b1372-ubuntu-rocm-gfx110X-x64.zip": "a7d541b6c6361aef559b5c8408e49f81e7c6a30260a2f67780eb513a10d8ab31",
    "llama-b1372-ubuntu-rocm-gfx1150-x64.zip": "eeb19dd0242e576ac36207a788161355c08e2df8cb836d9edb4c786e06b18022",
    "llama-b1372-ubuntu-rocm-gfx1151-x64.zip": "2cde9cf51d5bd6801b3533766422507f2f770babfdbd97fd20cf7e96cffb3ddd",
    "llama-b1372-ubuntu-rocm-gfx1152-x64.zip": "dd2b7fe37fd35dd854dc19925fa88a8fc239af06afdb92472950046dc8f858af",
    "llama-b1372-ubuntu-rocm-gfx1153-x64.zip": "77fa49568716bec8aa2c50d5e4aa1d80df28d777825caed6f8322006a488ccf5",
    "llama-b1372-ubuntu-rocm-gfx117X-x64.zip": "4d84f5878d1893ee1d48b630b1667a9cf7bbe067ef60c32b98c71b53965e87f2",
    "llama-b1372-ubuntu-rocm-gfx120X-x64.zip": "ae3e826db53a20397059f89c729ee25b4e52e55331a0fabbbd50a73b273c7efa",
    "llama-b1372-ubuntu-rocm-gfx908-x64.zip": "2ab0b5c8789449051e27b15ce7ac2c6e29a9a922981d09d2b5967d2a75b8564c",
    "llama-b1372-ubuntu-rocm-gfx90a-x64.zip": "0ef74ec3542c3ab19310d062e67751cdaa6aa63f85245f84f21f4e6ffa7efebd",
    # tag b12228 upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm
    # build's SIMD CPU backend
    "llama-b12228-bin-win-cpu-x64.zip": "58e0dfefb963c0031ee10a390db6203f9a225528000b2fd6a5d16c20aa044384",
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
