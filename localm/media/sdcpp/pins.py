# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp build localm installs, and the asset table used to
find and verify it.

One upstream release tag, one commit, and the sha256 of every archive localm
may install from it. ``_binding.py`` binds the C structs of exactly this commit;
``_binding.verify_abi`` refuses a library whose ``sd_commit()`` differs.
"""

from __future__ import annotations

REPO = "leejet/stable-diffusion.cpp"

TAG = "master-964-1d0a58a"

COMMIT = "1d0a58a35050d285b1d6fb1bc68a92924d35825e"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, backend) -> (asset name, sha256). platform is "windows", "linux"
# or "macos-arm64".
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "cpu"): (
        "sd-master-1d0a58a-bin-win-cpu-x64.zip",
        "8e4e7bde781ff246817019a182040a087314caf4d21bc15fe4ff045167bd336d"),
    ("windows", "vulkan"): (
        "sd-master-1d0a58a-bin-win-vulkan-x64.zip",
        "74211cb5c382aec2aeb086b64f2fdf525efec5faec4200a2d3257bc2328954ab"),
    ("windows", "cuda"): (
        "sd-master-1d0a58a-bin-win-cuda12-x64.zip",
        "2f08572e143f0b15a764197ea937d219dd70f0cc7eefb30eb84c9dae6a7cba6d"),
    ("windows", "rocm"): (
        "sd-master-1d0a58a-bin-win-rocm-10.1.0-x64.zip",
        "b83ff4e04f11b9027516f487e90833589ac19555702122a4e227385bc8ca52d3"),
    ("linux", "cpu"): (
        "sd-master-1d0a58a-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "2c2e6d4c0ad6a183283bda586e6e42c946eebfd9019278405d31763920ac36fa"),
    ("linux", "vulkan"): (
        "sd-master-1d0a58a-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "bade142929edfa7aad56aba46376e55142d14390262b904f4a2deb3a03ceb21d"),
    ("linux", "rocm"): (
        "sd-master-1d0a58a-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "a00fcd3233e8acfc8f5aadb0a1ab4e137a37d05dc61e2a37ddf94d7232b121ca"),
    ("macos-arm64", "metal"): (
        "sd-master-1d0a58a-bin-Darwin-macOS-26.6.2-arm64.zip",
        "45001839e925f98ecd18ef5251725f74cff874eeac4599f1136acba65b26911e"),
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
