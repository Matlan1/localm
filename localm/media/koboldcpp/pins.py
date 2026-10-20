# SPDX-License-Identifier: AGPL-3.0-or-later
"""The KoboldCpp release localm installs for native music generation, and the
asset table used to find and verify it.

One release tag; every asset localm may install from it with its exact size and
sha256. The music backend talks to KoboldCpp over its HTTP API only, so a pin
bump needs the live generation check re-run, not a struct layout check.
"""

from __future__ import annotations

REPO = "LostRuins/koboldcpp"

TAG = "v1.123.1"

# What ``koboldcpp-launcher --version`` prints for this tag.
VERSION = "1.123.1"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, build) -> (asset name, size in bytes, sha256). platform is
# "windows", "linux" or "macos-arm64". The "cuda" builds also carry the Vulkan
# and CPU backends; "nocuda" carries Vulkan and CPU; "metal" is the Apple
# Silicon build.
ASSETS: dict[tuple[str, str], tuple[str, int, str]] = {
    ("windows", "cuda"): (
        "koboldcpp.exe", 642170109,
        "5619a9de2b694a3f8ea7c28b0f9f61f73e6bded17c00c214cdde80f8ac9f0512"),
    ("windows", "nocuda"): (
        "koboldcpp-nocuda.exe", 117254353,
        "ab53b529c6c9925ed5f67b2884a5b35cc6d13c645f310a71c7a9e5c996950376"),
    ("linux", "cuda"): (
        "koboldcpp-linux-x64", 648375216,
        "f23cd1faacd96b5627ae009012e384f9b9b57f1a3e8e49b3c2fb10cd7fd6fc79"),
    ("linux", "nocuda"): (
        "koboldcpp-linux-x64-nocuda", 137022184,
        "6b1aa62b69f825199c3296b7dd36ff6b332b894098f50db8aba0061861c14d2b"),
    ("macos-arm64", "metal"): (
        "koboldcpp-mac-arm64", 64505520,
        "7bff84f74da1c8662c82828d80c8a2313aeccbde430a09b422e74964db281809"),
}


def asset_url(name: str) -> str:
    """The download URL of release asset *name* at ``TAG``."""
    return _BASE_URL + name
