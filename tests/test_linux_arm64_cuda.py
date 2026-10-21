# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linux ARM64 CUDA: the build and the runtime bundle both come from upstream's
release at one tag, only the cuda-13 line has an ARM64 build, and Linux x64
CUDA keeps its own resolver.

Driven against the pinned release's real asset names. Only the network
(release listing, downloads) is faked."""

from __future__ import annotations

import click
import pytest

from localm import hwdetect, setup_llama as sl

PIN = sl._PINNED_TAG
UPSTREAM = "ggml-org/llama.cpp"
BUILD = f"llama-{PIN}-bin-ubuntu-cuda-13.4-arm64.tar.gz"
CUDART = f"cudart-llama-{PIN}-bin-ubuntu-cuda-13.4-arm64.tar.gz"
REAL_NAMES = sorted(n for n in sl._PINNED_FALLBACK_SHA256
                    if n.startswith(f"llama-{PIN}-") or n.startswith(f"cudart-llama-{PIN}-"))


def _listing(names=REAL_NAMES) -> list:
    return [{"name": n,
             "browser_download_url": f"https://github.com/{UPSTREAM}/releases/download/{PIN}/{n}",
             "digest": "sha256:" + sl._PINNED_FALLBACK_SHA256[n]}
            for n in names]


@pytest.fixture
def machine(monkeypatch):
    """``machine(arch, listing=None)``: run as Linux on *arch* with the pinned
    release's real listing (``[]`` for an unavailable one). Returns the list of
    repos the release listing was asked for."""
    monkeypatch.setattr(sl.assets.console, "print", lambda *a, **k: None)
    repos: list = []

    def _set(arch: str, listing=None):
        rows = _listing() if listing is None else listing
        monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)

        def fake(tag, repo=None):
            repos.append(repo)
            return list(rows)
        monkeypatch.setattr(sl, "_release_assets", fake)
        return repos
    return _set


def _name(url: str) -> str:
    return url.rsplit("/", 1)[-1]


# ------------------------------ the table ---------------------------------- #

def test_linux_arm64_has_a_cuda_build_in_the_table():
    assert hwdetect.has_prebuilt("cuda", "linux", "arm64")
    assert "cuda" in hwdetect.prebuilt_backends("linux", "arm64")
    assert not hwdetect.has_prebuilt("cuda", "linux", "s390x")


def test_the_pinned_release_publishes_the_arm64_cuda_assets_the_resolver_names():
    assert BUILD in sl._PINNED_FALLBACK_SHA256
    assert CUDART in sl._PINNED_FALLBACK_SHA256


@pytest.mark.parametrize("caps,arch,expected", [
    ([(8, 7)], "arm64", "vulkan"),
    ([(9, 0)], "arm64", "vulkan"),
    ([], "arm64", "vulkan"),
    ([(12, 0)], "arm64", "cuda"),
    ([(8, 6), (10, 0)], "arm64", "cuda"),
    ([(8, 6)], "x64", "cuda"),
])
def test_arm64_recommendation_follows_the_gpus_cuda_line(monkeypatch, caps, arch, expected):
    monkeypatch.setattr(hwdetect, "platform_key", lambda: "linux")
    monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: caps)
    det = hwdetect.Detection(vendors=["nvidia"], gpu_names="nvidia gpu")
    assert hwdetect.recommended_install_backend(det) == expected


# ------------------------------ the build ---------------------------------- #

@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_arm64_cuda_13_build_comes_from_upstream_not_the_third_party_repo(machine, listing):
    repos = machine("arm64", listing)
    url, sha, tag = sl._resolve_backend_asset("cuda", "cuda-13")
    assert _name(url) == BUILD
    assert url.startswith(f"https://github.com/{UPSTREAM}/releases/download/{PIN}/")
    assert sha == sl._PINNED_FALLBACK_SHA256[BUILD] and tag == PIN
    assert sl._CUDA_LINUX_REPO not in repos


@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
@pytest.mark.parametrize("tag", [None, "b1"], ids=["pinned", "unpinned"])
def test_arm64_cuda_12_line_is_refused_whatever_the_listing(machine, listing, tag):
    machine("arm64", listing)
    with pytest.raises(click.ClickException) as ei:
        sl._resolve_backend_asset("cuda", "cuda-12", tag=tag)
    assert ei.value.message == (
        f"llama.cpp release {tag or PIN} has no cuda build for linux on arm64. Pick "
        "another backend, or build llama.cpp yourself and pass --from <build dir>.")


def test_x64_cuda_still_resolves_through_the_third_party_repo(machine):
    repos = machine("x64", [])
    url, _sha, _tag = sl._resolve_backend_asset("cuda", "cuda-12")
    assert _name(url) == f"llama-{PIN}-bin-ubuntu-cuda-x64.tar.gz"
    assert repos == [sl._CUDA_LINUX_REPO]


# --------------------------- the runtime bundle ---------------------------- #

@pytest.mark.parametrize("listing", [None, []], ids=["listing", "offline"])
def test_arm64_cudart_is_this_architectures_bundle_at_the_same_tag(machine, listing):
    machine("arm64", listing)
    url, sha = sl._resolve_linux_cudart_asset(PIN, "cuda-13")
    assert _name(url) == CUDART
    assert url == f"https://github.com/{UPSTREAM}/releases/download/{PIN}/{CUDART}"
    assert sha == sl._PINNED_FALLBACK_SHA256[CUDART]


def test_cudart_never_picks_the_other_architectures_bundle(machine):
    machine("arm64", _listing([n for n in REAL_NAMES if "arm64" not in n]))
    with pytest.raises(click.ClickException) as ei:
        sl._resolve_linux_cudart_asset(PIN, "cuda-13")
    assert ei.value.message == (
        f"llama.cpp release {PIN} has no Linux cuda-13 CUDA runtime bundle for arm64 "
        "- falling back to vulkan.")


def test_cudart_for_an_unpinned_tag_with_no_listing_is_refused(machine):
    machine("arm64", [])
    with pytest.raises(click.ClickException, match="no Linux cuda-13 CUDA runtime bundle"):
        sl._resolve_linux_cudart_asset("b1", "cuda-13")


def test_cudart_digest_comes_from_the_listing_when_it_has_one(machine):
    rows = _listing()
    for row in rows:
        if row["name"] == CUDART:
            row["digest"] = "sha256:" + "ab" * 32
    machine("arm64", rows)
    assert sl._resolve_linux_cudart_asset(PIN, "cuda-13")[1] == "ab" * 32


# ------------------------------ provisioning -------------------------------- #

@pytest.fixture
def provision(monkeypatch, machine, tmp_path):
    """Run _provision_backend as Linux ARM64, recording every download and
    every PyPI runtime fetch."""
    machine("arm64")
    monkeypatch.setattr(sl.sys, "platform", "linux")
    log = {"build": [], "cudart": [], "pypi": []}
    monkeypatch.setattr(sl, "_fetch_verified",
                        lambda url, target, sha, what: log["build"].append((_name(url), sha)))
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha=None: log["cudart"].append((_name(url), sha)) or 3)
    monkeypatch.setattr(sl, "_fetch_cuda_runtime_libs",
                        lambda line, target: log["pypi"].append(line) or 0)
    monkeypatch.setattr(sl, "_resolve_cuda_pair",
                        lambda *a, **k: pytest.fail("the Windows pair resolver ran"))

    def run(cuda_line="cuda-13", sha256=None):
        return sl._provision_backend("cuda", tmp_path, sha256, True, cuda_line)
    run.log = log
    return run


def test_arm64_provision_fetches_the_build_then_its_runtime_bundle(provision):
    assert provision() == PIN
    assert provision.log == {
        "build": [(BUILD, sl._PINNED_FALLBACK_SHA256[BUILD])],
        "cudart": [(CUDART, sl._PINNED_FALLBACK_SHA256[CUDART])],
        "pypi": []}


def test_arm64_provision_of_the_12_line_downloads_nothing(provision):
    with pytest.raises(click.ClickException, match="has no cuda build for linux on arm64"):
        provision("cuda-12")
    assert provision.log == {"build": [], "cudart": [], "pypi": []}


def test_arm64_provision_without_a_runtime_bundle_downloads_nothing(provision, machine):
    machine("arm64", _listing([n for n in REAL_NAMES if n != CUDART]))
    with pytest.raises(click.ClickException, match="no Linux cuda-13 CUDA runtime bundle"):
        provision()
    assert provision.log == {"build": [], "cudart": [], "pypi": []}


def test_arm64_provision_with_an_empty_runtime_bundle_is_an_error(provision, monkeypatch):
    monkeypatch.setattr(sl, "_fetch_and_place", lambda url, target, sha=None: 0)
    with pytest.raises(sl.ArtifactError, match="contained no libraries"):
        provision()


def test_arm64_sha256_pin_covers_the_build_only(provision):
    provision(sha256="f" * 64)
    assert provision.log["build"] == [(BUILD, "f" * 64)]
    assert provision.log["cudart"] == [(CUDART, sl._PINNED_FALLBACK_SHA256[CUDART])]


def test_x64_linux_provision_keeps_the_pypi_runtime(monkeypatch, machine, tmp_path):
    machine("x64")
    monkeypatch.setattr(sl.sys, "platform", "linux")
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("https://x/b.tar.gz", None, PIN))
    monkeypatch.setattr(sl, "_fetch_verified", lambda *a, **k: None)
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda *a, **k: pytest.fail("the upstream runtime bundle ran on x64"))
    pypi = []
    monkeypatch.setattr(sl, "_fetch_cuda_runtime_libs",
                        lambda line, target: pypi.append(line) or 0)
    sl._provision_backend("cuda", tmp_path, None, True, "cuda-12")
    assert pypi == ["cuda-12"]
