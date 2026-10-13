# SPDX-License-Identifier: AGPL-3.0-or-later
"""uv.lock must resolve each platform to its own torch.

The win32 ROCm torch is a Windows-only URL wheel. If the lock's darwin or linux
resolution reaches that wheel (or any `rocm*` package), `uv sync` / `uv export`
from the lock cannot install there. These tests walk the lock's dependency graph
the way an installer on each platform would, following only edges whose marker
holds on that platform, and assert on what is reachable.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]

_ENVIRONMENTS = {
    "win32": {
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "os_name": "nt",
    },
    "linux": {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
    },
    "darwin": {
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "os_name": "posix",
    },
}
_COMMON = {
    "python_version": "3.12",
    "python_full_version": "3.12.0",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}


def _load_lock(text: str | None = None) -> dict:
    if text is None:
        text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    return tomllib.loads(text)


def _holds(marker: str | None, platform: str) -> bool:
    if not marker:
        return True
    env = {**_COMMON, **_ENVIRONMENTS[platform], "extra": ""}
    return Marker(marker).evaluate(env)


def _index(lock: dict) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for pkg in lock["package"]:
        by_name.setdefault(pkg["name"], []).append(pkg)
    return by_name


def _target(by_name: dict[str, list[dict]], edge: dict) -> dict:
    candidates = by_name[edge["name"]]
    if "version" in edge:
        candidates = [c for c in candidates if c.get("version") == edge["version"]]
    if "source" in edge:
        candidates = [c for c in candidates if c.get("source") == edge["source"]]
    assert len(candidates) == 1, f"ambiguous or missing lock edge target: {edge}"
    return candidates[0]


def reachable_packages(lock: dict, platform: str) -> list[dict]:
    """Every package an install of the project with ALL extras and groups pulls in on *platform*."""
    by_name = _index(lock)
    root = next(p for p in lock["package"] if p["name"] == "localm")
    seen: dict[int, dict] = {}
    stack: list[tuple[dict, tuple[str, ...]]] = [(root, tuple(root.get("optional-dependencies", {})))]
    visited: set[tuple[int, tuple[str, ...]]] = set()
    while stack:
        pkg, extras = stack.pop()
        key = (id(pkg), extras)
        if key in visited:
            continue
        visited.add(key)
        seen[id(pkg)] = pkg
        edges = list(pkg.get("dependencies", []))
        for extra in extras:
            edges += pkg.get("optional-dependencies", {}).get(extra, [])
        if pkg is root:
            for group_edges in pkg.get("dev-dependencies", {}).values():
                edges += group_edges
        for edge in edges:
            if not _holds(edge.get("marker"), platform):
                continue
            stack.append((_target(by_name, edge), tuple(edge.get("extra", ()))))
    return list(seen.values())


def _is_rocm_artefact(pkg: dict) -> bool:
    source = pkg.get("source", {})
    return (
        pkg["name"].startswith("rocm")
        or "+rocm" in pkg.get("version", "")
        or "win_amd64" in source.get("url", "")
    )


@pytest.fixture(scope="module")
def lock() -> dict:
    return _load_lock()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_resolutions_reach_no_rocm_artefact(lock, platform):
    offenders = [
        f"{p['name']} {p.get('version', '')}"
        for p in reachable_packages(lock, platform)
        if _is_rocm_artefact(p)
    ]
    assert offenders == [], f"uv.lock hands {platform} a Windows-only ROCm artefact: {offenders}"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_resolutions_get_pypi_torch(lock, platform):
    torches = [p for p in reachable_packages(lock, platform) if p["name"] == "torch"]
    assert len(torches) == 1
    assert torches[0]["source"] == {"registry": "https://pypi.org/simple"}
    assert Version(torches[0]["version"]) >= Version("2.13.0")


def test_windows_resolution_still_gets_the_pinned_rocm_torch(lock):
    torches = [p for p in reachable_packages(lock, "win32") if p["name"] == "torch"]
    assert len(torches) == 1
    assert torches[0]["version"].endswith("+rocm7.13.0")
    assert "win_amd64" in torches[0]["source"]["url"]
    names = {p["name"] for p in reachable_packages(lock, "win32")}
    assert {"rocm", "rocm-sdk-core", "rocm-sdk-libraries-gfx103x-all"} <= names


_FORKED_LOCK_WRONG = """
version = 1
requires-python = "==3.12.*"

[[package]]
name = "localm"
version = "0.0.0"
source = { editable = "." }
dependencies = [
    { name = "accelerate" },
]

[[package]]
name = "accelerate"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "torch", version = "2.11.0+rocm7.13.0", source = { url = "https://example.invalid/torch-2.11.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" }, marker = "sys_platform == 'darwin' or sys_platform == 'win32'" },
    { name = "torch", version = "2.13.0", source = { registry = "https://pypi.org/simple" }, marker = "sys_platform != 'darwin' and sys_platform != 'win32'" },
]

[[package]]
name = "torch"
version = "2.11.0+rocm7.13.0"
source = { url = "https://example.invalid/torch-2.11.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" }
dependencies = [
    { name = "rocm" },
]

[[package]]
name = "torch"
version = "2.13.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "rocm"
version = "7.13.0"
source = { registry = "https://repo.amd.com/rocm/whl/gfx103X-all/" }
"""


def test_walker_flags_a_lock_that_hands_darwin_the_rocm_wheel():
    wrong = _load_lock(_FORKED_LOCK_WRONG)
    darwin = {p["name"] for p in reachable_packages(wrong, "darwin") if _is_rocm_artefact(p)}
    linux = {p["name"] for p in reachable_packages(wrong, "linux") if _is_rocm_artefact(p)}
    assert darwin == {"torch", "rocm"}
    assert linux == set()
