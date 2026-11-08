# SPDX-License-Identifier: AGPL-3.0-or-later
"""The KoboldCpp release localm installs for native music generation, and the
asset table used to find and verify it.

One release tag; every asset localm may install from it with its exact size and
sha256. The music backend talks to KoboldCpp over its HTTP API only, so a pin
bump needs the live generation check re-run, not a struct layout check.
"""

from __future__ import annotations

REPO = "LostRuins/koboldcpp"

TAG = "v1.124.2"

# What ``koboldcpp-launcher --version`` prints for this tag.
VERSION = "1.124.2"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, build) -> (asset name, size in bytes, sha256). platform is
# "windows", "linux" or "macos-arm64". The "cuda" builds also carry the Vulkan
# and CPU backends; "nocuda" carries Vulkan and CPU; "metal" is the Apple
# Silicon build.
ASSETS: dict[tuple[str, str], tuple[str, int, str]] = {
    ("windows", "cuda"): (
        "koboldcpp.exe", 622841516,
        "e306b58f9770fbca5265c7a0a6cc1c096d1bc3535a6d7e1f8e78c6810564faf7"),
    ("windows", "nocuda"): (
        "koboldcpp-nocuda.exe", 89911337,
        "c66550f4852a03ba992132a55bc0b1bb401851967005f10a3b7b7434484aed14"),
    ("linux", "cuda"): (
        "koboldcpp-linux-x64", 628981944,
        "c66ae0106673c7d9c41bafcc92c95bc5a6256709e9d853e5e2255bf536c1b069"),
    ("linux", "nocuda"): (
        "koboldcpp-linux-x64-nocuda", 107362352,
        "e7bc9457030fbbc454248d801d9f6971fc4b2d3fd16e0aec71f064eed26a05f4"),
    ("macos-arm64", "metal"): (
        "koboldcpp-mac-arm64", 68408816,
        "f6424d36081a22ba8d3e6c89413dde39bc3b0c46c3d435cbd6118d49263e5973"),
}


def asset_url(name: str) -> str:
    """The download URL of release asset *name* at ``TAG``."""
    return _BASE_URL + name
