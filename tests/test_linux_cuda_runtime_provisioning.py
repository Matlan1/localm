# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetching NVIDIA's CUDA runtime libraries (cudart/cublas/nccl) as plain PyPI
wheels for the Linux CUDA backend - see dev-notes/ADR-0010.

These tests exercise the REAL zip validation and extraction path (a wheel is
a zip); only the network transfer itself (``_download``) is faked, so a
regression in the sha256 check, the archive-shape check, or the .so-only
filter is caught here, not just "the function was called."
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from localm import setup_llama as sl


def _make_wheel_zip(path: Path, so_names: tuple, extra_names: tuple = ()) -> None:
    """A real, valid zip shaped like an nvidia-* wheel: some .so* files under a
    package dir, plus non-.so metadata files that must NOT be copied out.

    Padded well past _MIN_ARTIFACT_BYTES (256 KiB): a real cudart/cublas .so is
    genuinely multi-MB, and a too-small fixture would trip _validate_archive's
    own anti-truncation floor before ever reaching the logic under test here -
    that is not a fixture detail, it is what a real download would do too, so
    the fixture must not be smaller than a real one could ever legitimately be.
    Uncompressible (random) bytes so ZIP_STORED-vs-DEFLATED cannot shrink the
    archive back under the floor."""
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
    """Matches the REAL PyPI JSON shape (confirmed live against pypi.org this
    session) - must pick the x86_64 linux .whl, never the aarch64 wheel or the
    sdist tarball that also satisfy a looser match."""
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
    """THE NEGATIVE CASE: a checksum mismatch must refuse the file entirely -
    proves the guard actually guards, not just that the happy path works."""
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
    with pytest.raises(sl.ArtifactError, match="nvidia-nccl-cu12"):
        sl._fetch_pypi_runtime_lib("nvidia-nccl-cu12", tmp_path)


# ------------------------- _fetch_cuda_runtime_libs -------------------------- #

def test_fetch_cuda_runtime_libs_fetches_every_package_for_the_line(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sl, "_fetch_pypi_runtime_lib",
                        lambda pkg, target: calls.append(pkg) or 1)
    total = sl._fetch_cuda_runtime_libs("cuda-12", tmp_path)
    assert calls == list(sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-12"])
    assert total == len(calls)


def test_fetch_cuda_runtime_libs_stops_on_first_failure(monkeypatch, tmp_path):
    """A partially-assembled CUDA runtime is worse than none - must not
    swallow one package's failure and silently continue with the rest."""
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
    """Pins the naming-migration finding from this session (verified live
    against PyPI: nvidia-cublas-cu13 etc. are deprecated stubs) so a future
    edit cannot silently regress to the deprecated -cu13 names."""
    pkgs = sl._CUDA_RUNTIME_PYPI_PACKAGES["cuda-13"]
    assert "nvidia-cublas" in pkgs
    assert "nvidia-cuda-runtime" in pkgs
    assert "nvidia-cublas-cu13" not in pkgs
    assert "nvidia-cuda-runtime-cu13" not in pkgs
    # nccl has NOT migrated yet (confirmed live this session) - must still be cu12.
    assert "nvidia-nccl-cu12" in pkgs


# ------------------------- real network smoke test -------------------------- #

@pytest.mark.integration
def test_fetch_pypi_runtime_lib_real_network(tmp_path):
    """Not mocked at all: a real PyPI JSON lookup and a real wheel download,
    proving the actual mechanism this feature depends on, not a fixture of
    it. Picks the smallest real package (cuda-runtime, not cublas, which is
    ~100MB+) to keep this fast. Excluded from the default run
    (-m "not integration") like every other network-touching test in this
    repo; run explicitly to verify the real thing."""
    target = tmp_path / "runtime"
    target.mkdir()
    n = sl._fetch_pypi_runtime_lib("nvidia-cuda-runtime-cu12", target)
    assert n > 0
    so_files = list(target.iterdir())
    assert any("cudart" in f.name for f in so_files)
