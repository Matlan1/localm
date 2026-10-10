# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp build localm installs, and the asset table used to
find and verify it.

One upstream release tag, one commit, and the sha256 of every archive localm
may install from it. ``_binding.py`` binds the C structs of exactly this commit;
``_binding.verify_abi`` refuses a library whose ``sd_commit()`` differs.
"""

from __future__ import annotations

REPO = "leejet/stable-diffusion.cpp"

TAG = "master-951-f89d9b1"

COMMIT = "f89d9b13d730eabeede7314ce49dacd18d3c90c2"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, backend) -> (asset name, sha256). platform is "windows", "linux"
# or "macos-arm64".
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "cpu"): (
        "sd-master-f89d9b1-bin-win-cpu-x64.zip",
        "7c46bc1f4e89b22af9dd70e8d510dea2167c619676d23d313b83deffcd9b1ecc"),
    ("windows", "vulkan"): (
        "sd-master-f89d9b1-bin-win-vulkan-x64.zip",
        "2c5bed22585ff678ff2c9077f5088d55c2d9c0faa948db7f2e689e1ccea701a9"),
    ("windows", "cuda"): (
        "sd-master-f89d9b1-bin-win-cuda12-x64.zip",
        "f129da40b1a85565c79a00a73bdf696164773cfa63dd06450d29d018508208a3"),
    ("windows", "rocm"): (
        "sd-master-f89d9b1-bin-win-rocm-7.14.0-x64.zip",
        "e283cb551cac313bd1c654ed39a29740cc21071edae1775c6b72183b4a6440b2"),
    ("linux", "cpu"): (
        "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "2390e744f0c97f452df851581262c145218aa54157db795010c3244a46e07319"),
    ("linux", "vulkan"): (
        "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "441a12efaf319bebd577f09a35fff862e154ebae2a19bdfdd61185e0d6071718"),
    ("linux", "rocm"): (
        "sd-master-f89d9b1-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip",
        "5792ceb913af57faa2a124a5588a62ee3bc10df3d01666007ccf959e0daa50f6"),
    ("macos-arm64", "metal"): (
        "sd-master-f89d9b1-bin-Darwin-macOS-26.6.2-arm64.zip",
        "8057dbfb1529b4e0bc07d35eaf68fa821e44bca6852428c539bf70602189bf92"),
}

# Extra archives installed into the same directory as the main one.
EXTRA_ASSETS: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("windows", "cuda"): [(
        "cudart-sd-bin-win-cu12-x64.zip",
        "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d")],
}


def asset_url(name: str) -> str:
    """The download URL of release asset *name* at ``TAG``."""
    return _BASE_URL + name
