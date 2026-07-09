# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backend selection + asset resolution for `localm setup-llama`, and the
`hwdetect` helper. Pure/offline: network calls are monkeypatched to fail so the
URL-resolution FALLBACK path is exercised deterministically.
"""

from __future__ import annotations

import sys

import pytest

from localm import hwdetect, setup_llama as sl


# --------------------------- hwdetect ------------------------------------- #

def test_detect_returns_valid_shape():
    det = hwdetect.detect()                       # must never raise
    assert det.recommended in ("vulkan", "cpu", "metal")
    assert set(det.vendors) <= set(hwdetect.VENDORS) | {"apple"}
    assert det.has_gpu == bool(det.vendors)


def test_recommended_install_backend_policy(monkeypatch):
    """The ONE installer-backend policy both setup.bat and setup.sh share."""
    def rec(vendors, names, platform="win32"):
        monkeypatch.setattr(hwdetect.sys, "platform", platform)
        return hwdetect.recommended_install_backend(
            hwdetect.Detection(vendors=vendors, gpu_names=names))
    # AMD on Windows: RX 6000 / unknown keep the self-contained gfx103X ROCm build...
    assert rec(["amd"], "amd radeon rx 6900 xt") == "amd-rocm"
    assert rec(["amd"], "amd radeon graphics") == "amd-rocm"
    # ...but a CLEARLY non-gfx103X AMD downgrades to the universal Vulkan build.
    assert rec(["amd"], "amd radeon rx 7800 xt") == "vulkan"
    assert rec(["amd"], "amd radeon rx 9070") == "vulkan"
    assert rec(["amd"], "amd radeon rx 5700") == "vulkan"
    # AMD on Linux is always vulkan (the self-contained bundle is Windows-only).
    assert rec(["amd"], "amd radeon rx 6900 xt", platform="linux") == "vulkan"
    # Any other GPU -> vulkan; no GPU -> cpu; Apple Silicon -> metal.
    assert rec(["nvidia"], "nvidia geforce rtx 4090") == "vulkan"
    assert rec(["intel"], "intel arc a770") == "vulkan"
    assert rec([], "") == "cpu"
    assert rec(["apple"], "", platform="darwin") == "metal"


def test_hwdetect_cli_prints_vendor_and_backend(capsys):
    """`python -m localm.hwdetect` emits '<vendor> <install-backend>' for the shells."""
    assert hwdetect.main() == 0
    out = capsys.readouterr().out.strip().split()
    assert len(out) == 2
    assert out[1] in ("vulkan", "cpu", "metal", "amd-rocm")


# --------------------------- auto backend policy -------------------------- #

def _fake_detect(vendors, recommended):
    return lambda: hwdetect.Detection(vendors=list(vendors), recommended=recommended)


def test_auto_backend_no_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect([], "cpu"))
    if sys.platform == "darwin":
        pytest.skip("darwin no-gpu path returns the detector's recommendation")
    assert sl._auto_backend() == "cpu"


@pytest.mark.parametrize("vendor", ["nvidia", "intel"])
def test_auto_backend_single_vendor_is_vulkan(monkeypatch, vendor):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect([vendor], "vulkan"))
    assert sl._auto_backend() == "vulkan"


def test_auto_backend_mixed_amd_nvidia_is_vulkan(monkeypatch):
    # A box with both must use the universal backend, not the AMD-only build.
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia", "amd"], "vulkan"))
    assert sl._auto_backend() == "vulkan"


@pytest.mark.skipif(sys.platform != "win32", reason="amd-rocm bundled build is Windows-only")
def test_auto_backend_amd_only_is_rocm_on_windows(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["amd"], "vulkan"))
    assert sl._auto_backend() == "amd-rocm"


# --------------------------- URL resolution ------------------------------- #

@pytest.mark.skipif(sys.platform != "win32", reason="checks the Windows AMD bundle URL")
def test_resolve_amd_rocm_is_self_contained_url():
    assert sl._resolve_backend_url("amd-rocm") == sl.DEFAULT_URL


def test_resolve_unsupported_backend_raises():
    import click
    bad = "metal" if sys.platform == "win32" else "amd-rocm"
    with pytest.raises(click.ClickException):
        sl._resolve_backend_url(bad)


def test_extract_archive_rejects_path_traversal(tmp_path):
    """A zip member that escapes the destination (absolute, drive, or ..) is
    refused before extraction, not written outside the target."""
    import zipfile
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ok/llama.dll", b"x")
        zf.writestr("../escape.dll", b"x")          # path traversal
    with pytest.raises(sl.ArtifactError):
        sl._extract_archive(z, tmp_path / "out")
    assert not (tmp_path / "escape.dll").exists()   # nothing escaped


def test_extract_archive_accepts_clean_zip(tmp_path):
    import zipfile
    z = tmp_path / "good.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("bin/llama.dll", b"x")
        zf.writestr("bin/ggml.dll", b"y")
    out = tmp_path / "out"
    sl._extract_archive(z, out)
    assert (out / "bin" / "llama.dll").is_file()


def test_resolve_offline_falls_back_to_pinned_tag(monkeypatch):
    """With the release API unreachable, resolution must still produce a sane
    upstream URL built from the pinned fallback tag and the right backend."""
    def _boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr("urllib.request.urlopen", _boom)

    plat = sl._platform_key()
    # 'vulkan' exists on win/linux; on darwin use 'cpu'.
    backend = "vulkan" if plat in ("win32", "linux") else "cpu"
    url = sl._resolve_backend_url(backend)
    assert sl._FALLBACK_TAG in url
    assert backend.replace("amd-rocm", "rocm") in url or "macos" in url
    assert url.startswith(f"https://github.com/{sl._UPSTREAM_REPO}/releases/download/")


def test_download_stall_raises_and_restores_timeout(monkeypatch, tmp_path):
    """A stalled transfer must become a loud ArtifactError, not an infinite
    hang, and the global socket default timeout must be restored afterward."""
    import socket

    sentinel = object()
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: sentinel)
    restored = []
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda v: restored.append(v))

    def _stall(url, dest, hook):
        # the timeout must be armed to the stall value while the fetch runs
        assert restored and restored[-1] == sl._DOWNLOAD_STALL_TIMEOUT
        raise socket.timeout("timed out")

    monkeypatch.setattr(sl.urllib.request, "urlretrieve", _stall)

    with pytest.raises(sl.ArtifactError, match="stalled"):
        sl._download("https://example.invalid/big.zip", tmp_path / "a.zip")

    # last action was to restore the previous default timeout (the sentinel)
    assert restored[-1] is sentinel


def test_download_restores_timeout_on_success(monkeypatch, tmp_path):
    """On a normal download the previous socket timeout is still restored."""
    import socket

    sentinel = object()
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: sentinel)
    restored = []
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda v: restored.append(v))
    monkeypatch.setattr(sl.urllib.request, "urlretrieve",
                        lambda url, dest, hook: None)

    sl._download("https://example.invalid/big.zip", tmp_path / "a.zip")

    assert restored[0] == sl._DOWNLOAD_STALL_TIMEOUT
    assert restored[-1] is sentinel


def test_resolve_backend_asset_resolves_sha256(monkeypatch):
    dummy_assets = [
        {
            "name": "llama-b9870-bin-win-vulkan-x64.zip",
            "browser_download_url": "https://dummy.github/releases/download/b9870/llama-vulkan.zip",
            "digest": "sha256:dummysha256value"
        }
    ]
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: dummy_assets)
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9870")
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")

    url, sha = sl._resolve_backend_asset("vulkan")
    assert url == "https://dummy.github/releases/download/b9870/llama-vulkan.zip"
    assert sha == "dummysha256value"


def test_resolve_backend_asset_fallback_uses_pinned_sha256(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9870")
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")

    url, sha = sl._resolve_backend_asset("vulkan")
    assert "llama-b9870-bin-win-vulkan-x64.zip" in url
    assert sha == "8687a8405447853ccbd6b15bd7ccda23bb79cf85dd83243401e514bd9e45ed8a"


def test_provision_backend_verifies_default_sha256(monkeypatch):
    from pathlib import Path
    passed_sha256 = []
    def fake_fetch_and_place(url, target, sha256=None):
        passed_sha256.append(sha256)
        return 1

    monkeypatch.setattr(sl, "_fetch_and_place", fake_fetch_and_place)
    monkeypatch.setattr(sl, "_resolve_backend_asset", lambda backend: ("https://dummy.url", "dummysha"))

    sl._provision_backend("vulkan", Path("dummy_target"), sha256=None, with_cudart=False)
    assert passed_sha256 == ["dummysha"]


def test_custom_url_warning_printed(monkeypatch):
    monkeypatch.setattr(sl, "_fetch_and_place", lambda url, target, sha256=None: 1)
    monkeypatch.setattr(sl, "_clear_target", lambda target: None)
    monkeypatch.setattr(sl, "_verify", lambda: None)

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(sl.main, ["--url", "https://dummy.invalid/llama.zip", "--force"])
    assert "Warning: Custom URL download is unverified" in result.output
