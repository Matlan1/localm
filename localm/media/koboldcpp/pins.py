# SPDX-License-Identifier: AGPL-3.0-or-later
"""The KoboldCpp release localm installs for native music generation, and the
asset table used to find and verify it.

One release tag; every asset localm may install from it with its exact size and
sha256. The music backend talks to KoboldCpp over its HTTP API only, so a pin
bump needs the live generation check re-run, not a struct layout check.
"""

from __future__ import annotations

REPO = "LostRuins/koboldcpp"

TAG = "v1.123"

# What ``koboldcpp-launcher --version`` prints for this tag.
VERSION = "1.123"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, build) -> (asset name, size in bytes, sha256). platform is
# "windows", "linux" or "macos-arm64". The "cuda" builds also carry the Vulkan
# and CPU backends; "nocuda" carries Vulkan and CPU; "metal" is the Apple
# Silicon build.
ASSETS: dict[tuple[str, str], tuple[str, int, str]] = {
    ("windows", "cuda"): (
        "koboldcpp.exe", 642223685,
        "ec20acbad449aaef75e44eac8b585b11cffb681eea7b98b74230f980f30fab38"),
    ("windows", "nocuda"): (
        "koboldcpp-nocuda.exe", 117318959,
        "226410e2e412764b494bd12c52eba1cf3de43f5657f447242201df430ad7ae58"),
    ("linux", "cuda"): (
        "koboldcpp-linux-x64", 648386296,
        "a86d140d5895ad88aeef0fb9e3c500202287072d8ceba0c45108953da09d5ff3"),
    ("linux", "nocuda"): (
        "koboldcpp-linux-x64-nocuda", 137030248,
        "89bf9b17ea6a43533a170d2db516acf25e0763363ed4f9825a61efb201ac8be7"),
    ("macos-arm64", "metal"): (
        "koboldcpp-mac-arm64", 64490784,
        "c82fd647c5a56188575c7e978ce13d3441fe52ce9159f93c4b8422dcf0df1377"),
}


def asset_url(name: str) -> str:
    """The download URL of release asset *name* at ``TAG``."""
    return _BASE_URL + name
