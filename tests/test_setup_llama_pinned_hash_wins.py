# SPDX-License-Identifier: AGPL-3.0-or-later
"""The in-code sha256 pin is what a pinned llama.cpp runtime asset is verified
against. A release listing digest that disagrees with the pin is a refusal,
before anything is downloaded; the listing digest is used only for an asset
name the pins do not cover.

Driven through the real resolvers against the pinned release's real asset
names. Only the network (release listing, downloads) is faked."""

from __future__ import annotations

from pathlib import Path

import pytest

from localm import hwdetect, setup_llama as sl
from localm.setup_llama.download import ArtifactError
from localm.setup_llama.pins import _CUDA_LINUX_PINNED_SHA256

PIN = sl._PINNED_TAG
UPSTREAM = "ggml-org/llama.cpp"
OTHER = "0" * 64


def _row(name: str, digest, repo: str = UPSTREAM, tag: str = PIN) -> dict:
    row = {"name": name, "size": 1,
           "browser_download_url": f"https://github.com/{repo}/releases/download/{tag}/{name}"}
    if digest is not None:
        row["digest"] = digest
    return row


def _pinned(prefix: str) -> str:
    hits = sorted(n for n in sl._PINNED_FALLBACK_SHA256 if n.startswith(prefix))
    assert hits, prefix
    return hits[0]


@pytest.fixture
def machine(monkeypatch):
    """``machine(plat, arch, rows)``: run on *plat*/*arch* with *rows* as every
    release listing. Returns the list of download URLs that were fetched."""
    monkeypatch.setattr(sl.assets.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(sl.provision.console, "print", lambda *a, **k: None)
    fetched: list = []

    def _set(plat: str, arch: str, rows: list):
        monkeypatch.setattr(sl, "_platform_key", lambda: plat)
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)
        monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: list(rows))

        def fake_fetch(url, target, sha, *a, **k):
            fetched.append((url, sha))
            return True
        monkeypatch.setattr(sl, "_fetch_verified", fake_fetch)
        monkeypatch.setattr(sl, "_fetch_and_place", fake_fetch)
        return fetched
    return _set


# ------------------------------ the rule ----------------------------------- #

def test_pinned_name_returns_the_pin_when_the_listing_agrees():
    name = _pinned(f"llama-{PIN}-bin-win-vulkan-")
    pin = sl._PINNED_FALLBACK_SHA256[name]
    assert sl._asset_sha256(_row(name, "sha256:" + pin.upper())) == pin


def test_pinned_name_returns_the_pin_when_the_listing_has_no_digest():
    name = _pinned(f"llama-{PIN}-bin-win-vulkan-")
    assert sl._asset_sha256(_row(name, None)) == sl._PINNED_FALLBACK_SHA256[name]


def test_pinned_name_with_a_disagreeing_listing_digest_is_refused():
    name = _pinned(f"llama-{PIN}-bin-win-vulkan-")
    with pytest.raises(ArtifactError) as ei:
        sl._asset_sha256(_row(name, "sha256:" + OTHER))
    msg = str(ei.value)
    assert name in msg and OTHER in msg and sl._PINNED_FALLBACK_SHA256[name] in msg


def test_unpinned_name_uses_the_listing_digest():
    assert sl._asset_sha256(_row("llama-b1-bin-win-vulkan-x64.zip", "sha256:" + OTHER)) == OTHER


def test_unpinned_name_without_a_digest_has_no_sha():
    assert sl._asset_sha256(_row("llama-b1-bin-win-vulkan-x64.zip", None)) is None


# --------------------------- the call sites -------------------------------- #

def test_generic_resolver_refuses_a_disagreeing_digest(machine):
    name = _pinned(f"llama-{PIN}-bin-win-vulkan-x64")
    machine("win32", "x64", [_row(name, "sha256:" + OTHER)])
    with pytest.raises(ArtifactError):
        sl._resolve_backend_asset("vulkan", tag=PIN)


def test_generic_resolver_returns_the_pin_not_the_listing(machine):
    name = _pinned(f"llama-{PIN}-bin-win-vulkan-x64")
    machine("win32", "x64", [_row(name, None)])
    url, sha, tag = sl._resolve_backend_asset("vulkan", tag=PIN)
    assert url.endswith("/" + name) and sha == sl._PINNED_FALLBACK_SHA256[name] and tag == PIN


def test_linux_cuda_third_party_resolver_refuses_a_disagreeing_digest(machine):
    name = f"llama-{PIN}-bin-ubuntu-cuda-x64.tar.gz"
    assert name in _CUDA_LINUX_PINNED_SHA256
    machine("linux", "x64", [_row(name, "sha256:" + OTHER, repo=sl._CUDA_LINUX_REPO)])
    with pytest.raises(ArtifactError):
        sl._resolve_backend_asset("cuda", "cuda-12", tag=PIN)


def test_linux_cudart_resolver_refuses_a_disagreeing_digest(machine):
    name = _pinned(f"cudart-llama-{PIN}-bin-ubuntu-cuda-13")
    machine("linux", "arm64" if "arm64" in name else "x64", [_row(name, "sha256:" + OTHER)])
    with pytest.raises(ArtifactError):
        sl._resolve_linux_cudart_asset(PIN, "cuda-13")


def test_amd_rocm_resolver_refuses_a_disagreeing_digest(machine, monkeypatch):
    monkeypatch.setattr(sl.assets.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "amd_gfx_family", lambda names: "gfx103X")
    monkeypatch.setattr(hwdetect, "detect", lambda: hwdetect.Detection(vendors=["amd"], gpu_names="rx 6900"))
    name = f"llama-{sl._ROCM_TAG}-windows-rocm-gfx103X-x64.zip"
    assert name in sl._PINNED_FALLBACK_SHA256
    machine("win32", "x64", [_row(name, "sha256:" + OTHER, repo="lemonade-sdk/llamacpp-rocm",
                                  tag=sl._ROCM_TAG)])
    with pytest.raises(ArtifactError):
        sl._resolve_backend_asset("amd-rocm")


@pytest.mark.parametrize("which", ["build", "cudart"])
def test_windows_cuda_pair_refuses_a_disagreeing_digest_before_any_download(
        machine, monkeypatch, tmp_path: Path, which):
    build = _pinned(f"llama-{PIN}-bin-win-cuda-12")
    cudart = _pinned("cudart-llama-bin-win-cuda-12")
    monkeypatch.setattr(sl.provision.sys, "platform", "win32")
    rows = [_row(build, "sha256:" + (OTHER if which == "build" else sl._PINNED_FALLBACK_SHA256[build])),
            _row(cudart, "sha256:" + (OTHER if which == "cudart" else sl._PINNED_FALLBACK_SHA256[cudart]))]
    fetched = machine("win32", "x64", rows)
    with pytest.raises(ArtifactError):
        sl.provision._provision_backend("cuda", tmp_path, None, True, "cuda-12", tag=PIN)
    if which == "build":
        assert fetched == []
    else:
        assert fetched == [(rows[0]["browser_download_url"], sl._PINNED_FALLBACK_SHA256[build])]


def test_windows_cuda_pair_verifies_against_the_pins(machine, monkeypatch, tmp_path: Path):
    build = _pinned(f"llama-{PIN}-bin-win-cuda-12")
    cudart = _pinned("cudart-llama-bin-win-cuda-12")
    monkeypatch.setattr(sl.provision.sys, "platform", "win32")
    rows = [_row(build, None), _row(cudart, None)]
    fetched = machine("win32", "x64", rows)
    assert sl.provision._provision_backend("cuda", tmp_path, None, True, "cuda-12", tag=PIN) == PIN
    assert [sha for _u, sha in fetched] == [sl._PINNED_FALLBACK_SHA256[build],
                                            sl._PINNED_FALLBACK_SHA256[cudart]]
