# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-llama fetches the llama.cpp build for the CPU architecture of the
running process, and refuses a backend that has no build for it.

Driven against the pinned release's real asset names, in the order the release
listing returns them (name order), so an architecture-blind match picks the
first look-alike exactly as it would against the live listing."""

from __future__ import annotations

import sys

import click
import pytest

from localm import hwdetect, setup_llama as sl
from localm.setup_llama import assets as sl_assets

PIN = sl._PINNED_TAG
UPSTREAM = "ggml-org/llama.cpp"
REAL_NAMES = sorted(n for n in sl._PINNED_FALLBACK_SHA256
                    if n.startswith(f"llama-{PIN}-") or n.startswith("cudart-llama-"))


def _listing(names=REAL_NAMES) -> list:
    return [{"name": n,
             "browser_download_url": f"https://github.com/{UPSTREAM}/releases/download/{PIN}/{n}",
             "digest": "sha256:" + sl._PINNED_FALLBACK_SHA256[n]}
            for n in names]


@pytest.fixture
def machine(monkeypatch):
    """``machine(plat, arch, listing=...)``: run as *plat* / *arch* with the
    release listing *listing* (default: the pinned release's real assets; []
    for an unavailable listing)."""
    monkeypatch.setattr(sl_assets.console, "print", lambda *a, **k: None)

    def _set(plat: str, arch: str, listing=None):
        rows = _listing() if listing is None else listing
        monkeypatch.setattr(sl, "_platform_key", lambda: plat)
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)
        monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: list(rows))
    return _set


def _name(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _refusal(backend, line=None, tag=None) -> str:
    with pytest.raises(click.ClickException) as ei:
        sl._resolve_backend_asset(backend, line, tag=tag)
    return ei.value.message


# ------------------------------- macOS ------------------------------------ #

def test_the_listing_names_the_arm64_mac_build_before_the_x64_one():
    assert (REAL_NAMES.index(f"llama-{PIN}-bin-macos-arm64.tar.gz")
            < REAL_NAMES.index(f"llama-{PIN}-bin-macos-x64.tar.gz"))


@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_intel_mac_cpu_gets_the_x64_build(machine, listing):
    machine("darwin", "x64", listing)
    url, sha, tag = sl._resolve_backend_asset("cpu")
    assert _name(url) == f"llama-{PIN}-bin-macos-x64.tar.gz"
    assert sha == sl._PINNED_FALLBACK_SHA256[_name(url)] and tag == PIN


@pytest.mark.parametrize("backend", ["cpu", "metal"])
@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_apple_silicon_gets_the_arm64_build(machine, listing, backend):
    machine("darwin", "arm64", listing)
    url, _sha, _tag = sl._resolve_backend_asset(backend)
    assert _name(url) == f"llama-{PIN}-bin-macos-arm64.tar.gz"


def test_metal_on_an_intel_mac_is_refused_naming_what_is_available(machine):
    machine("darwin", "x64")
    assert _refusal("metal") == (
        "backend 'metal' has no prebuilt build for darwin on x64. Available here: "
        "cpu. Or build llama.cpp yourself and pass --from <build dir>.")


@pytest.mark.parametrize("backend", ["vulkan", "cuda", "hip", "sycl"])
def test_backends_without_a_mac_build_are_refused_on_macos(machine, backend):
    machine("darwin", "arm64")
    assert _refusal(backend) == (
        f"backend {backend!r} is not available on this platform (darwin). "
        "Available: cpu, metal.")


# ------------------------------- Linux ------------------------------------ #

@pytest.mark.parametrize("arch, backend, expected", [
    ("arm64", "cpu", "bin-ubuntu-arm64.tar.gz"),
    ("arm64", "vulkan", "bin-ubuntu-vulkan-arm64.tar.gz"),
    ("s390x", "cpu", "bin-ubuntu-s390x.tar.gz"),
    ("x64", "cpu", "bin-ubuntu-x64.tar.gz"),
    ("x64", "vulkan", "bin-ubuntu-vulkan-x64.tar.gz"),
])
@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_linux_gets_the_build_for_its_architecture(machine, listing, arch, backend, expected):
    machine("linux", arch, listing)
    url, sha, _tag = sl._resolve_backend_asset(backend)
    assert _name(url) == f"llama-{PIN}-{expected}"
    assert sha == sl._PINNED_FALLBACK_SHA256[_name(url)]


@pytest.mark.parametrize("backend", ["cuda", "hip", "sycl"])
def test_linux_arm64_refuses_x64_only_backends(machine, backend):
    machine("linux", "arm64")
    assert _refusal(backend) == (
        f"backend {backend!r} has no prebuilt build for linux on arm64. Available "
        "here: cpu, vulkan. Or build llama.cpp yourself and pass --from <build dir>.")


@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_an_architecture_with_no_build_at_all_is_refused(machine, backend):
    machine("linux", "riscv64")
    assert _refusal(backend) == (
        f"backend {backend!r} has no prebuilt build for linux on riscv64. Available "
        "here: none. Or build llama.cpp yourself and pass --from <build dir>.")


def test_a_listing_without_this_architectures_build_is_refused_not_guessed(machine):
    x64_only = [n for n in REAL_NAMES if "arm64" not in n]
    machine("linux", "arm64", _listing(x64_only))
    assert _refusal("cpu") == (
        f"llama.cpp release {PIN} has no cpu build for linux on arm64. Pick another "
        "backend, or build llama.cpp yourself and pass --from <build dir>.")


def test_an_unpinned_tag_offline_guesses_this_architectures_name(machine):
    machine("linux", "arm64", [])
    url, sha, tag = sl._resolve_backend_asset("cpu", tag="b1")
    assert url == f"https://github.com/{UPSTREAM}/releases/download/b1/llama-b1-bin-ubuntu-arm64.tar.gz"
    assert sha is None and tag == "b1"


# ------------------------------- Windows ---------------------------------- #

def test_the_listing_names_the_arm64_cuda_13_assets_before_the_x64_ones():
    assert (REAL_NAMES.index(f"llama-{PIN}-bin-win-cuda-13.4-arm64.zip")
            < REAL_NAMES.index(f"llama-{PIN}-bin-win-cuda-13.4-x64.zip"))
    assert (REAL_NAMES.index("cudart-llama-bin-win-cuda-13.4-arm64.zip")
            < REAL_NAMES.index("cudart-llama-bin-win-cuda-13.4-x64.zip"))


@pytest.mark.parametrize("arch", ["x64", "arm64"])
def test_windows_cuda_13_pairs_the_build_and_runtime_of_this_architecture(machine, arch):
    machine("win32", arch)
    build, cudart = sl._resolve_cuda_pair(PIN, "cuda-13")
    assert build["name"] == f"llama-{PIN}-bin-win-cuda-13.4-{arch}.zip"
    assert cudart["name"] == f"cudart-llama-bin-win-cuda-13.4-{arch}.zip"


def test_windows_x64_cuda_12_pair_is_x64(machine):
    machine("win32", "x64")
    build, cudart = sl._resolve_cuda_pair(PIN, "cuda-12")
    assert build["name"] == f"llama-{PIN}-bin-win-cuda-12.4-x64.zip"
    assert cudart["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"


@pytest.mark.parametrize("arch", ["x64", "arm64"])
@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_windows_cuda_13_build_asset_is_this_architectures(machine, listing, arch):
    machine("win32", arch, listing)
    url, _sha, _tag = sl._resolve_backend_asset("cuda", "cuda-13")
    assert _name(url) == f"llama-{PIN}-bin-win-cuda-13.4-{arch}.zip"


@pytest.mark.parametrize("backend, expected", [
    ("cpu", "bin-win-cpu-arm64.zip"),
    ("vulkan", "bin-win-vulkan-arm64.zip"),
])
def test_windows_arm64_gets_arm64_builds(machine, backend, expected):
    machine("win32", "arm64")
    url, _sha, _tag = sl._resolve_backend_asset(backend)
    assert _name(url) == f"llama-{PIN}-{expected}"


def test_windows_arm64_cuda_12_has_no_build(machine):
    machine("win32", "arm64")
    assert _refusal("cuda", "cuda-12") == (
        f"llama.cpp release {PIN} has no cuda build for win32 on arm64. Pick another "
        "backend, or build llama.cpp yourself and pass --from <build dir>.")


def test_amd_rocm_is_refused_on_windows_arm64(machine, monkeypatch):
    machine("win32", "arm64")
    monkeypatch.setattr(sys, "platform", "win32")
    assert _refusal("amd-rocm") == (
        "backend 'amd-rocm' has no prebuilt build for win32 on arm64. Available here: "
        "cpu, cuda, vulkan. Or build llama.cpp yourself and pass --from <build dir>.")


# ------------------------- table vs resolution ---------------------------- #

_ARCHES = sorted({a for backends in hwdetect.PREBUILT_ARCHES.values()
                  for arches in backends.values() for a in arches})


@pytest.mark.parametrize("plat, backend, arch", [
    (plat, backend, arch)
    for plat in hwdetect.PREBUILT_ARCHES
    for backend in sl._ASSET_MATCH[plat]
    if (plat, backend) not in (("linux", "cuda"), ("win32", "cuda"))
    for arch in _ARCHES
])
def test_resolution_succeeds_exactly_where_the_table_says_a_build_exists(
        machine, plat, backend, arch):
    machine(plat, arch)
    if hwdetect.has_prebuilt(backend, plat, arch):
        url, _sha, _tag = sl._resolve_backend_asset(backend)
        assert sl_assets._asset_arch_ok(_name(url).lower(), arch)
        assert _name(url) in sl._PINNED_FALLBACK_SHA256
    else:
        with pytest.raises(click.ClickException):
            sl._resolve_backend_asset(backend)


# ------------------------------- helpers ---------------------------------- #

@pytest.mark.parametrize("name, arch, ok", [
    ("llama-b1-bin-macos-arm64.tar.gz", "arm64", True),
    ("llama-b1-bin-macos-arm64.tar.gz", "x64", False),
    ("llama-b1-bin-ubuntu-x64.tar.gz", "x64", True),
    ("llama-b1-bin-ubuntu-s390x.tar.gz", "x64", False),
    ("llama-b1-bin-ubuntu-cuda-13-x64.tar.gz", "arm64", False),
    ("llama-1342-windows-rocm-gfx103x-x64.zip", "x64", True),
    ("llama-b1-bin-linux-x86_64.tar.gz", "x64", True),
    ("llama-b1-bin-linux-aarch64.tar.gz", "arm64", True),
    ("llama-b1-ui.tar.gz", "arm64", True),
])
def test_asset_arch_ok(name, arch, ok):
    assert sl_assets._asset_arch_ok(name, arch) is ok


@pytest.mark.parametrize("plat, sysconfig_plat, machine_name, maxsize, expected", [
    ("win32", "win-amd64", "ARM64", 2**63 - 1, "x64"),
    ("win32", "win-arm64", "ARM64", 2**63 - 1, "arm64"),
    ("win32", "win32", "AMD64", 2**31 - 1, "x86"),
    ("darwin", "macosx-11.0-arm64", "arm64", 2**63 - 1, "arm64"),
    ("darwin", "macosx-10.9-universal2", "x86_64", 2**63 - 1, "x64"),
    ("linux", "linux-aarch64", "aarch64", 2**63 - 1, "arm64"),
    ("linux", "linux-x86_64", "x86_64", 2**63 - 1, "x64"),
    ("linux", "linux-s390x", "s390x", 2**63 - 1, "s390x"),
    ("linux", "linux-riscv64", "riscv64", 2**63 - 1, "riscv64"),
    ("linux", "linux-armv7l", "aarch64", 2**31 - 1, "arm"),
    ("linux", "linux-i686", "x86_64", 2**31 - 1, "x86"),
    ("win32", "linux-x86_64", "x86_64", 2**63 - 1, "x64"),
    ("linux", "win-amd64", "AMD64", 2**63 - 1, "x64"),
])
def test_machine_arch_names_the_process_architecture(
        monkeypatch, plat, sysconfig_plat, machine_name, maxsize, expected):
    monkeypatch.setattr(hwdetect.sys, "platform", plat)
    monkeypatch.setattr(hwdetect.sysconfig, "get_platform", lambda: sysconfig_plat)
    monkeypatch.setattr(hwdetect.platform, "machine", lambda: machine_name)
    monkeypatch.setattr(hwdetect.sys, "maxsize", maxsize)
    arch = hwdetect.machine_arch()
    monkeypatch.undo()
    assert arch == expected


@pytest.mark.parametrize("plat, arch, vendors, expected", [
    ("linux", "arm64", ["nvidia"], "vulkan"),
    ("linux", "x64", ["nvidia"], "cuda"),
    ("linux", "s390x", ["nvidia"], "cpu"),
    ("linux", "arm64", [], "cpu"),
    ("win32", "arm64", ["intel"], "vulkan"),
    ("win32", "arm64", ["nvidia"], "cuda"),
    ("darwin", "arm64", ["apple"], "metal"),
    ("darwin", "x64", [], "cpu"),
])
def test_recommended_install_backend_only_picks_a_backend_with_a_build(
        monkeypatch, plat, arch, vendors, expected):
    monkeypatch.setattr(hwdetect.sys, "platform", plat)
    monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: False)
    got = hwdetect.recommended_install_backend(hwdetect.Detection(vendors=vendors))
    assert got == expected
