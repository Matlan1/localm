# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp build localm installs, and the asset table used to
find and verify it.

One upstream release tag, one commit, and the sha256 of every archive localm
may install from it. ``_binding.py`` binds the C structs of exactly this commit;
``_binding.verify_abi`` refuses a library whose ``sd_commit()`` differs.
"""

from __future__ import annotations

REPO = "leejet/stable-diffusion.cpp"

TAG = "master-982-0518b34"

COMMIT = "0518b34374761cb82aeda5ad3c2f0686fdad2cec"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, backend) -> (asset name, sha256). platform is "windows", "linux"
# or "macos-arm64".
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "cpu"): (
        "sd-master-0518b34-bin-win-cpu-x64.zip",
        "4d8ec59037f70d0b61954bcca1a0008dfe238fabd9e21f8d354b16be9e7c3d4b"),
    ("windows", "vulkan"): (
        "sd-master-0518b34-bin-win-vulkan-x64.zip",
        "c2ab91b33206beba434f739a9131916539a3d6229da153e6e0311013e6308567"),
    ("windows", "cuda"): (
        "sd-master-0518b34-bin-win-cuda12-x64.zip",
        "f4dd12ee7fa974f1858ced85191b90e5d3c288dd4b10f296be91175138b5b421"),
    ("windows", "rocm"): (
        "sd-master-0518b34-bin-win-rocm-10.1.0-x64.zip",
        "5390bbaa95e3e9ec30913912f9c92f4a48d9a3c08457c100657d50f34b63e0c7"),
    ("linux", "cpu"): (
        "sd-master-0518b34-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "929527ad778f14e52fcd2f2340c427b935eb113bca1709b3de4791a2859e6288"),
    ("linux", "vulkan"): (
        "sd-master-0518b34-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "8bc413b3e6307fbefdf985cc562f7db1cc4c68e2dc3dca9d69b66ce98b9cbb5a"),
    ("linux", "rocm"): (
        "sd-master-0518b34-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "a1dfd769e36c1d5f4c6c37daff401c7da35ca78f97d86d774c016dce36e1f34d"),
    ("macos-arm64", "metal"): (
        "sd-master-0518b34-bin-Darwin-macOS-26.6.2-arm64.zip",
        "b5b995203c1eed71e252d5363237325d6b45b8a64ab14911f3104b57f8c7644a"),
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
