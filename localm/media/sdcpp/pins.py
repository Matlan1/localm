# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp build localm installs, and the asset table used to
find and verify it.

One upstream release tag, one commit, and the sha256 of every archive localm
may install from it. ``_binding.py`` binds the C structs of exactly this commit;
``_binding.verify_abi`` refuses a library whose ``sd_commit()`` differs.
"""

from __future__ import annotations

REPO = "leejet/stable-diffusion.cpp"

TAG = "master-1043-3be2378"

COMMIT = "3be23788fc0c9f35a2c24b1211caae11d4a54dbc"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, backend) -> (asset name, sha256). platform is "windows", "linux"
# or "macos-arm64".
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "cpu"): (
        "sd-master-3be2378-bin-win-cpu-x64.zip",
        "10ef877a334c97d014e3e23b4bf6c9077f6b78ff9da3bb157664d2103cb1ae0f"),
    ("windows", "vulkan"): (
        "sd-master-3be2378-bin-win-vulkan-x64.zip",
        "a5e1fa679561a3bb99b114a8f4a41e67239da29173ff725391ecfa5772e869ee"),
    ("windows", "cuda"): (
        "sd-master-3be2378-bin-win-cuda12-x64.zip",
        "76da083337834ff9c383c5f80a49c8cf7485f8fa28d1fac1ef5ce057bbcb3e51"),
    ("windows", "rocm"): (
        "sd-master-3be2378-bin-win-rocm-10.1.0-x64.zip",
        "df5ce14e778784ab9f5302aab72bda49c3e4b512f15fea0ad5f32c616e5c5015"),
    ("linux", "cpu"): (
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "01e1c773c1bc2a6276952f6dbd13294e58b7255c659b4a770086934c4e8fad3a"),
    ("linux", "vulkan"): (
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "7608901016fb4a348bc5b75c46fc971085dadadf44e6f30daa97301844b1ce2e"),
    ("linux", "rocm"): (
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "0058c3547163d455c478ab593653908bfaadaf2d48e4224ad9669e9509b1c67d"),
    ("macos-arm64", "metal"): (
        "sd-master-3be2378-bin-Darwin-macOS-26.6.2-arm64.zip",
        "ec40c7b84d86333ea89a9a52745fce25c3081122f6207a4370f35b504df99252"),
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
