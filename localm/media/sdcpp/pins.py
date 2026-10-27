# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stable-diffusion.cpp build localm installs, and the asset table used to
find and verify it.

One upstream release tag, one commit, and the sha256 of every archive localm
may install from it. ``_binding.py`` binds the C structs of exactly this commit;
``_binding.verify_abi`` refuses a library whose ``sd_commit()`` differs.
"""

from __future__ import annotations

REPO = "leejet/stable-diffusion.cpp"

TAG = "master-1017-be48649"

COMMIT = "be486498eadc29dc3cd234d944c2180ed42b1415"

_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"

# (platform, backend) -> (asset name, sha256). platform is "windows", "linux"
# or "macos-arm64".
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "cpu"): (
        "sd-master-be48649-bin-win-cpu-x64.zip",
        "f0f76c8176a90cafc431ca7d5049cbe8c5cea61ca7cb92b78e07361d41d8fb21"),
    ("windows", "vulkan"): (
        "sd-master-be48649-bin-win-vulkan-x64.zip",
        "833414dfe255d9709c6184e699017028ee8d5522058ad53b201d9602805c6b41"),
    ("windows", "cuda"): (
        "sd-master-be48649-bin-win-cuda12-x64.zip",
        "a63c85596fb76cfd0668cd82dd3faff7cc511325891f0411117c99ee1f360c84"),
    ("windows", "rocm"): (
        "sd-master-be48649-bin-win-rocm-10.1.0-x64.zip",
        "13e643b777c3155f81943ee710148b94aa90207022c1a6967cbf24afc2c0a23a"),
    ("linux", "cpu"): (
        "sd-master-be48649-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "1016ea927e00b0176ca23aa9f5dc5de68290ea7e115eb3fe6c2399acca41701f"),
    ("linux", "vulkan"): (
        "sd-master-be48649-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "794c913d077aaf72025e772b2acfb6a30f49156e92d072461ae15ce958d95c14"),
    ("linux", "rocm"): (
        "sd-master-be48649-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "5d8d282f0be04a2f289a8719b13e4086020921005be138d83f108df4b87c930a"),
    ("macos-arm64", "metal"): (
        "sd-master-be48649-bin-Darwin-macOS-26.6.2-arm64.zip",
        "6478d12a084b5d7f4158d45606e1fbbd56698e44bea60363e90709c9d6d9aaf1"),
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
