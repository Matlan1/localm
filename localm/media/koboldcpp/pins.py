# SPDX-License-Identifier: AGPL-3.0-or-later
"""The KoboldCpp release localm installs for native music generation, and the
asset table used to find and verify it.

One release tag; every asset localm may install from it with its exact size and
sha256. The music backend talks to KoboldCpp over its HTTP API only, so a pin
bump needs the live generation check re-run, not a struct layout check.
"""

from __future__ import annotations

REPO = "LostRuins/koboldcpp"

TAG = "v1.122.1"

# What ``koboldcpp-launcher --version`` prints for this tag.
VERSION = "1.122.1"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, build) -> (asset name, size in bytes, sha256). platform is
# "windows", "linux" or "macos-arm64". The "cuda" builds also carry the Vulkan
# and CPU backends; "nocuda" carries Vulkan and CPU; "metal" is the Apple
# Silicon build.
ASSETS: dict[tuple[str, str], tuple[str, int, str]] = {
    ("windows", "cuda"): (
        "koboldcpp.exe", 636192312,
        "c0955f60aac10d139cfdb3ffbd73ad5c7045bfff32656b4a971363d08ff31289"),
    ("windows", "nocuda"): (
        "koboldcpp-nocuda.exe", 117328777,
        "c314724e02b310c4db066a8dade8890a1628bc4b65aa9c2b658309219ca7a779"),
    ("linux", "cuda"): (
        "koboldcpp-linux-x64", 641836648,
        "724b81ad4d0557e6cf6b1e634c1b973e9a56c0a99b5164a9cf1ecbb8c9ad53d8"),
    ("linux", "nocuda"): (
        "koboldcpp-linux-x64-nocuda", 136689280,
        "5532ead66f460a59c744fc74a45715bf2b0ef2fe2fb05a6c146a6d2df2145d29"),
    ("macos-arm64", "metal"): (
        "koboldcpp-mac-arm64", 67057424,
        "4dc85e7f0414812ec5b1fd810b2009a73844ff17347edac41ff85ffe7fc70412"),
}


def asset_url(name: str) -> str:
    """The download URL of release asset *name* at ``TAG``."""
    return _BASE_URL + name
