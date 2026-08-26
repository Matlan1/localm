# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-contained CUDA on Linux, end to end:
  - fetching NVIDIA's CUDA runtime libraries (cudart/cublas) as plain
    PyPI wheels;
  - resolving the compiled binary from a third-party prebuilt
    (hybridgroup/llama-cpp-builder), since upstream publishes none for
    Linux and localm does not build its own binaries;
  - _provision_backend wiring the two together on Linux.

These tests exercise the REAL zip validation and extraction path (a wheel is
a zip); only the network transfer itself (``_download``) is faked, so the
sha256 check, the archive-shape check, and the .so-only filter all run for
real. The resolver tests mock only ``_release_assets``/``_latest_tag``,
matching the sibling amd-rocm resolver tests.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import click
import pytest

from localm import setup_llama as sl


def _make_wheel_zip(path: Path, so_names: tuple, extra_names: tuple = ()) -> None:
    """A real, valid zip shaped like an nvidia-* wheel: some .so* files under a
    package dir, plus non-.so metadata files that must NOT be copied out.

    Padded past _MIN_ARTIFACT_BYTES (256 KiB), which _validate_archive's
    anti-truncation floor requires and a real multi-MB cudart/cublas .so
    exceeds. Uncompressible (random) bytes, so ZIP_STORED-vs-DEFLATED cannot
    shrink the archive back under the floor."""
    import os
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in so_names:
            zf.writestr(f"nvidia/pkg/lib/{name}", b"\x7fELF" + os.urandom(300 * 1024))
        for name in extra_names:
            zf.writestr(f"nvidia_pkg-1.0.dist-info/{name}", b"not a library")


_REAL_SHAPED_PYPI_RESPONSE = {
    "info": {"version": "12.9.79"},
    "releases": {
        "12.9.79": [
            {
                "filename": "nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
                "url": "https://files.pythonhosted.org/packages/aa/aarch64.whl",
                "digests": {"sha256": "aaaa"},
            },
            {
                "filename": "nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "url": "https://files.pythonhosted.org/packages/bb/x86_64.whl",
                "digests": {"sha256": "bbbb"},
            },
            {
                "filename": "nvidia_cuda_runtime_cu12-12.9.79.tar.gz",
                "url": "https://files.pythonhosted.org/packages/cc/sdist.tar.gz",
                "digests": {"sha256": "cccc"},
            },
        ]
    },
}


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self.headers = {}

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------- _pypi_wheel_url_and_sha ------------------------- #

def test_pypi_wheel_url_and_sha_picks_linux_x86_64_wheel(monkeypatch):
    """Against the real PyPI JSON shape: picks the x86_64 linux .whl, never the
    aarch64 wheel or the sdist tarball that also satisfy a looser match."""
    import json as _json
    monkeypatch.setattr(sl, "verified_urlopen",
                        lambda req, timeout=10: _FakeResponse(
                            _json.dumps(_REAL_SHAPED_PYPI_RESPONSE).encode()))
    url, sha = sl._pypi_wheel_url_and_sha("nvidia-cuda-runtime-cu12")
    assert url == "https://files.pythonhosted.org/packages/bb/x86_64.whl"
    assert sha == "bbbb"


def test_pypi_wheel_url_and_sha_never_raises_on_network_error(monkeypatch):
    def boom(req, timeout=10):
        raise OSError("network down")
    monkeypatch.setattr(sl, "verified_urlopen", boom)
    assert sl._pypi_wheel_url_and_sha("nvidia-cublas-cu12") == (None, None)


def test_pypi_wheel_url_and_sha_none_when_no_linux_wheel(monkeypatch):
    import json as _json
    payload = {"info": {"version": "1.0"}, "releases": {"1.0": [
        {"filename": "pkg-1.0-py3-none-win_amd64.whl",
         "url": "https://x/win.whl", "digests": {"sha256": "x"}},
    ]}}
    monkeypatch.setattr(sl, "verified_urlopen",
                        lambda req, timeout=10: _FakeResponse(_json.dumps(payload).encode()))
    assert sl._pypi_wheel_url_and_sha("nvidia-cublas-cu12") == (None, None)


# ------------------------- _fetch_pypi_runtime_lib -------------------------- #

def test_fetch_pypi_runtime_lib_copies_only_so_files(monkeypatch, tmp_path):
    wheel_src = tmp_path / "src" / "fake.whl"
    wheel_src.parent.mkdir()
    _make_wheel_zip(wheel_src,
                    so_names=("libcudart.so.12", "libcudart.so.12.9.79"),
                    extra_names=("METADATA", "RECORD"))

    def fake_download(url, dest):
        dest.write_bytes(wheel_src.read_bytes())
        return sl._DownloadResult(bytes_received=dest.stat().st_size,
                                  content_length=dest.stat().st_size,
                                  content_type="application/zip", final_url=url)

    monkeypatch.setattr(sl, "_download", fake_download)
    monkeypatch.setattr(sl, "_pypi_wheel_url_and_sha",
                        lambda pkg: ("https://fake/nvidia-cuda-runtime-cu12.whl", None))

    target = tmp_path / "runtime"
    target.mkdir()
    n = sl._fetch_pypi_runtime_lib("nvidia-cuda-runtime-cu12", target)

    assert n == 2
    copied = sorted(p.name for p in target.iterdir())
    assert copied == ["libcudart.so.12", "libcudart.so.12.9.79"]
    assert not (target / "METADATA").exists()
    assert not (target / "RECORD").exists()


def test_fetch_pypi_runtime_lib_refuses_wrong_checksum(monkeypatch, tmp_path):
    """THE NEGATIVE CASE: a checksum mismatch refuses the file entirely."""
    wheel_src = tmp_path / "src" / "fake.whl"
    wheel_src.parent.mkdir()
    _make_wheel_zip(wheel_src, so_names=("libcublas.so.12",))

    def fake_download(url, dest):
        dest.write_bytes(wheel_src.read_bytes())
        return sl._DownloadResult(bytes_received=dest.stat().st_size,
                                  content_length=dest.stat().st_size,
                                  content_type="application/zip", final_url=url)

    monkeypatch.setattr(sl, "_download", fake_download)
    # A real sha256 shape, but WRONG - the actual digest of the fixture differs.
    monkeypatch.setattr(sl, "_pypi_wheel_url_and_sha",
                        lambda pkg: ("https://fake/nvidia-cublas-cu12.whl",
                                     "0" * 64))

    target = tmp_path / "runtime"
    target.mkdir()
    with pytest.raises(sl.ArtifactError, match="sha256"):
        sl._fetch_pypi_runtime_lib("nvidia-cublas-cu12", target)

    assert list(target.iterdir()) == []   # nothing copied on a refused download


def test_fetch_pypi_runtime_lib_raises_when_no_wheel_resolved(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "_pypi_wheel_url_and_sha", lambda pkg: (None, None))
    with pytest.raises(sl.ArtifactError, match="nvidia-cublas-cu12"):
        sl._fetch_pypi_runtime_lib("nvidia-cublas-cu12", tmp_path)


# ------------------------- _fetch_cuda_runtime_libs -------------------------- #

def test_fetch_cuda_runtime_libs_fetches_every_package_for_the_line(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sl, "_fetch_pypi_runtime_lib",
                        lambda pkg, target: calls.append(pkg) or 1)
    total = sl._fetch_cuda_runtime_libs("cuda-12", tmp_path)
    assert calls == list(sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-12"])
    assert total == len(calls)


def test_fetch_cuda_runtime_libs_stops_on_first_failure(monkeypatch, tmp_path):
    """One package's failure stops the fetch rather than leaving a partially
    assembled CUDA runtime."""
    calls = []

    def fake_fetch(pkg, target):
        calls.append(pkg)
        if pkg == sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-12"][1]:
            raise sl.ArtifactError("boom")
        return 1

    monkeypatch.setattr(sl, "_fetch_pypi_runtime_lib", fake_fetch)
    with pytest.raises(sl.ArtifactError, match="boom"):
        sl._fetch_cuda_runtime_libs("cuda-12", tmp_path)
    assert calls == list(sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-12"][:2])   # never reached the third


def test_cuda_13_line_uses_unsuffixed_package_names():
    """The cuda-13 line uses the UNSUFFIXED package names; the -cu13 spellings
    (nvidia-cublas-cu13 and friends) are deprecated stubs on PyPI."""
    pkgs = sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-13"]
    assert "nvidia-cublas" in pkgs
    assert "nvidia-cuda-runtime" in pkgs
    assert "nvidia-cublas-cu13" not in pkgs
    assert "nvidia-cuda-runtime-cu13" not in pkgs


def test_no_line_fetches_nccl():
    """No CUDA line fetches nccl: none of the shared libraries in
    hybridgroup/llama-cpp-builder's binary reference libnccl."""
    for pkgs in sl._CUDA_RUNTIME_PYPI_PACKAGES.values():
        assert not any("nccl" in p for p in pkgs)


# ------------------------- linux cuda binary resolver ------------------------ #
# _resolve_backend_asset's linux-cuda special case: resolves against
# hybridgroup/llama-cpp-builder (a third-party prebuilt) instead of upstream,
# which publishes none. Only _release_assets/_latest_tag are mocked, never the
# network itself.

_DUMMY_LINUX_CUDA_ASSETS = [
    {
        "name": "llama-b9870-bin-ubuntu-cuda-x64.tar.gz",
        "browser_download_url": "https://dummy.github/hybridgroup/llama-cpp-builder/releases/download/b9870/llama-b9870-bin-ubuntu-cuda-x64.tar.gz",
        "digest": "sha256:dummylinuxcudasha",
    },
    {
        "name": "llama-b9870-bin-ubuntu-cuda-x64.tar.gz.sha256",
        "browser_download_url": "https://dummy.github/hybridgroup/llama-cpp-builder/releases/download/b9870/llama-b9870-bin-ubuntu-cuda-x64.tar.gz.sha256",
        "digest": "sha256:dummysidecarsha",
    },
    {
        "name": "llama-b9870-bin-ubuntu-cuda-13-x64.tar.gz",
        "browser_download_url": "https://dummy.github/hybridgroup/llama-cpp-builder/releases/download/b9870/llama-b9870-bin-ubuntu-cuda-13-x64.tar.gz",
        "digest": "sha256:dummylinuxcuda13sha",
    },
]


def test_resolve_linux_cuda_asset_finds_tarball(monkeypatch):
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the Linux CUDA branch must resolve the pin, "
                            "not upstream's newest"))
    seen = {}

    def fake_release_assets(tag, repo=None):
        seen["tag"], seen["repo"] = tag, repo
        return _DUMMY_LINUX_CUDA_ASSETS

    monkeypatch.setattr(sl, "_release_assets", fake_release_assets)

    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line="cuda-12")

    assert url == _DUMMY_LINUX_CUDA_ASSETS[0]["browser_download_url"]
    assert sha == "dummylinuxcudasha"
    # Resolved against hybridgroup's repo, by the SAME bare tag every other
    # Linux backend uses, not upstream (ggml-org/llama.cpp) and not a prefixed
    # tag scheme.
    assert seen["repo"] == sl._CUDA_LINUX_REPO
    assert seen["tag"] == sl._PINNED_TAG


def test_resolve_linux_cuda_asset_picks_cuda13_line(monkeypatch):
    """hybridgroup publishes both a cuda-12 asset (bare "-cuda-x64.tar.gz")
    and a cuda-13 one ("-cuda-13-x64.tar.gz"); a Blackwell GPU
    (NvidiaInfo.cuda_line == 'cuda-13') gets the cuda-13 asset."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the Linux CUDA branch must resolve the pin, "
                            "not upstream's newest"))
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: _DUMMY_LINUX_CUDA_ASSETS)

    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line="cuda-13")

    assert url == _DUMMY_LINUX_CUDA_ASSETS[2]["browser_download_url"]
    assert sha == "dummylinuxcuda13sha"


def test_resolve_linux_cuda_asset_never_picks_the_sha256_sidecar(monkeypatch):
    """THE NEGATIVE CASE: the release also lists a .sha256 sidecar file, which
    is never mistaken for the tarball itself."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the Linux CUDA branch must resolve the pin, "
                            "not upstream's newest"))
    # Sidecar listed FIRST, so a naive "first match" would pick it.
    reordered = [_DUMMY_LINUX_CUDA_ASSETS[1], _DUMMY_LINUX_CUDA_ASSETS[0]]
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: reordered)

    url, _, _tag = sl._resolve_backend_asset("cuda", cuda_line="cuda-12")

    assert url.endswith(".tar.gz")
    assert not url.endswith(".sha256")


def test_resolve_linux_cuda_asset_raises_when_not_yet_built(monkeypatch):
    """When hybridgroup has not built this upstream tag yet, the resolver
    raises click.ClickException - which _provision_with_fallback's caller turns
    into the vulkan fallback - rather than constructing a guessed URL."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])

    # Matched on the PIN rather than a stubbed tag: the message names the build
    # that was actually looked for, which under the pin is _PINNED_TAG.
    with pytest.raises(click.ClickException, match=sl._PINNED_TAG):
        sl._resolve_backend_asset("cuda")


def test_resolve_backend_asset_windows_cuda_unaffected_by_linux_branch(monkeypatch):
    """The linux-cuda special case does not intercept the Windows cuda path,
    which has its own resolver (_resolve_cuda_pair)."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(
        sl, "_latest_tag",
        lambda: pytest.fail("the Linux CUDA branch must resolve the pin, "
                            "not upstream's newest"))
    calls = []
    monkeypatch.setattr(sl, "_release_assets",
                        lambda tag, repo=None: calls.append(repo) or [
                            {"name": "llama-b9870-bin-win-cuda-12.4-x64.zip",
                             "browser_download_url": "https://dummy/win-cuda.zip",
                             "digest": "sha256:wincudasha"}])

    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line="cuda-12")

    assert url == "https://dummy/win-cuda.zip"
    # Called against upstream (None/default), never the Linux-CUDA third party.
    assert sl._CUDA_LINUX_REPO not in calls


# ------------------------- linux cuda provisioning wiring --------------------- #
# _provision_backend's linux-cuda branch: fetches the binary via the resolver
# above, then the runtime libs via _fetch_cuda_runtime_libs. Both are the real
# mechanisms; only the network itself is mocked here.

def test_provision_backend_linux_cuda_fetches_binary_and_runtime_libs(monkeypatch, tmp_path):
    # sys.platform, not _platform_key(): _provision_backend's branch checks
    # sys.platform directly, matching its neighbouring win32 branch, so mocking
    # _platform_key alone would leave sys.platform untouched and the win32
    # branch would still win on a real Windows test box.
    monkeypatch.setattr(sl.sys, "platform", "linux")
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("https://dummy/linux-cuda.tar.gz", "binsha", "bTEST"))
    fetched = []
    monkeypatch.setattr(sl, "_fetch_verified",
                        lambda url, target, sha, what: fetched.append((url, sha)))
    runtime_calls = []
    monkeypatch.setattr(sl, "_fetch_cuda_runtime_libs",
                        lambda cuda_line, target: runtime_calls.append(cuda_line) or 3)

    sl._provision_backend("cuda", tmp_path, sha256=None, with_cudart=True, cuda_line="cuda-12")

    assert fetched == [("https://dummy/linux-cuda.tar.gz", "binsha")]
    assert runtime_calls == ["cuda-12"]


def test_provision_backend_linux_cuda_never_calls_windows_cuda_pair(monkeypatch, tmp_path):
    """The Linux branch takes its own path and does not fall through into the
    Windows-only _resolve_cuda_pair/cudart-bundle logic above it in the same
    function."""
    monkeypatch.setattr(sl.sys, "platform", "linux")
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("https://dummy/linux-cuda.tar.gz", None, "bTEST"))
    monkeypatch.setattr(sl, "_fetch_verified", lambda *a, **k: None)
    monkeypatch.setattr(sl, "_fetch_cuda_runtime_libs", lambda cuda_line, target: 0)

    def boom(*a, **k):
        raise AssertionError("_resolve_cuda_pair is Windows-only and must not be called on Linux")
    monkeypatch.setattr(sl, "_resolve_cuda_pair", boom)

    sl._provision_backend("cuda", tmp_path, sha256=None, with_cudart=True)   # must not raise


# ------------------------- real network smoke test -------------------------- #

@pytest.mark.integration
def test_resolve_linux_cuda_asset_real_network(monkeypatch):
    """Not mocked beyond forcing the linux branch: a real GitHub Releases
    lookup against hybridgroup/llama-cpp-builder, so the asset-name convention
    and repo are checked against the live release rather than a fixture."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "linux")
    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line="cuda-12")
    assert url.startswith(f"https://github.com/{sl._CUDA_LINUX_REPO}/releases/download/")
    assert url.endswith("-cuda-x64.tar.gz")
    assert sha and len(sha) == 64   # a real sha256 hex digest


@pytest.mark.integration
def test_fetch_pypi_runtime_lib_real_network(tmp_path):
    """Not mocked: a real PyPI JSON lookup and a real wheel download. Picks the
    smallest real package (cuda-runtime, not cublas, which is ~100MB+).
    Excluded from the default run (-m "not integration") like every other
    network-touching test here."""
    target = tmp_path / "runtime"
    target.mkdir()
    n = sl._fetch_pypi_runtime_lib("nvidia-cuda-runtime-cu12", target)
    assert n > 0
    so_files = list(target.iterdir())
    assert any("cudart" in f.name for f in so_files)
