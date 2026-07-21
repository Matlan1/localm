# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backend selection + asset resolution for `localm setup-llama`, and the
`hwdetect` helper. Pure/offline: network calls are monkeypatched to fail so the
URL-resolution FALLBACK path is exercised deterministically.
"""

from __future__ import annotations

import ssl
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
    # NVIDIA: cuda on Windows (the release ships a self-contained cudart bundle),
    # but vulkan on Linux (the Linux cuda build needs a system CUDA toolkit).
    assert rec(["nvidia"], "nvidia geforce rtx 4090") == "cuda"
    assert rec(["nvidia"], "nvidia geforce rtx 4090", platform="linux") == "vulkan"
    # Intel -> vulkan; no GPU -> cpu; Apple Silicon -> metal.
    assert rec(["intel"], "intel arc a770") == "vulkan"
    assert rec([], "") == "cpu"
    assert rec(["apple"], "", platform="darwin") == "metal"


def test_hwdetect_cli_prints_vendor_and_backend(capsys):
    """`python -m localm.hwdetect` emits '<vendor> <install-backend>' for the shells."""
    assert hwdetect.main() == 0
    out = capsys.readouterr().out.strip().split()
    assert len(out) == 2
    assert out[1] in ("vulkan", "cuda", "cpu", "metal", "amd-rocm")


# --------------------------- auto backend policy -------------------------- #

def _fake_detect(vendors, recommended):
    return lambda: hwdetect.Detection(vendors=list(vendors), recommended=recommended)


def test_auto_backend_no_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect([], "cpu"))
    if sys.platform == "darwin":
        pytest.skip("darwin no-gpu path returns the detector's recommendation")
    assert sl._auto_backend() == "cpu"


def test_auto_backend_intel_is_vulkan(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["intel"], "vulkan"))
    assert sl._auto_backend() == "vulkan"


def test_auto_backend_nvidia_is_cuda_on_windows(monkeypatch):
    # bare `setup-llama` (auto) must match the installer: NVIDIA on Windows -> cuda.
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia"], "vulkan"))
    assert sl._auto_backend() == "cuda"


def test_auto_backend_nvidia_is_vulkan_on_linux(monkeypatch):
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia"], "vulkan"))
    assert sl._auto_backend() == "vulkan"


def test_auto_backend_mixed_amd_nvidia(monkeypatch):
    # A box with both: cuda on Windows (NVIDIA is the priority vendor and its
    # cudart bundle is self-contained), vulkan on Linux (no self-contained cuda).
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia", "amd"], "vulkan"))
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert sl._auto_backend() == "cuda"
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    assert sl._auto_backend() == "vulkan"


@pytest.mark.skipif(sys.platform != "win32", reason="amd-rocm bundled build is Windows-only")
def test_auto_backend_amd_only_is_rocm_on_windows(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["amd"], "vulkan"))
    assert sl._auto_backend() == "amd-rocm"


# --------------------------- TLS verification (SSL) ----------------------- #
# The SSL context itself is tested in tests/test_http_ssl.py (the shared helper);
# these two lock that setup-llama's GitHub calls actually PASS a verifying context.

def test_release_assets_passes_verifying_context(monkeypatch):
    # Regression: the GitHub API lookups MUST present a verifying SSL context.
    # Without one, a box whose OS cert store lacks the release-CDN CA fails every
    # call with CERTIFICATE_VERIFY_FAILED (the reported setup-llama failure).
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"assets": []}'

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp()

    monkeypatch.setattr(sl.urllib.request, "urlopen", fake_urlopen)
    sl._release_assets("btag")
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_download_passes_verifying_context(monkeypatch, tmp_path):
    # Regression: the archive download itself must verify too (it was the raw
    # urlretrieve call with no context that surfaced in the failure report).
    seen = {"done": False}

    class _Resp:
        headers = {"Content-Length": "4"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            if not seen["done"]:
                seen["done"] = True
                return b"data"
            return b""

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp()

    monkeypatch.setattr(sl.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "a.bin"
    sl._download("https://example/x.zip", dest)
    assert dest.read_bytes() == b"data"
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


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


def test_release_assets_logs_the_swallowed_cause(monkeypatch, caplog):
    """_release_assets() must not silently swallow the failure (AGENTS.md rule
    5): the exception must be discoverable at debug level, not just an empty
    list with no trace of why."""
    import logging

    def _boom(*a, **k):
        raise OSError("simulated DNS failure")
    monkeypatch.setattr("urllib.request.urlopen", _boom)

    with caplog.at_level(logging.DEBUG, logger="localm"):
        assets = sl._release_assets("b9870")

    assert assets == []
    assert "simulated dns failure" in caplog.text.lower()


def test_download_stall_raises_and_restores_timeout(monkeypatch, tmp_path):
    """A stalled transfer must become a loud ArtifactError, not an infinite
    hang, and the global socket default timeout must be restored afterward."""
    import socket

    sentinel = object()
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: sentinel)
    restored = []
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda v: restored.append(v))

    class _StallResp:
        headers = {"Content-Length": "1000000"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            # the timeout must be armed to the stall value while the fetch runs
            assert restored and restored[-1] == sl._DOWNLOAD_STALL_TIMEOUT
            raise socket.timeout("timed out")

    monkeypatch.setattr(sl.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: _StallResp())

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

    class _OkResp:
        headers = {"Content-Length": "4"}
        def __init__(self): self._sent = False
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            if self._sent:
                return b""
            self._sent = True
            return b"data"

    monkeypatch.setattr(sl.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: _OkResp())

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


def test_resolve_amd_rocm_asset_warns_when_release_lookup_fails(monkeypatch, capsys):
    """A failed lemonade-sdk release lookup must be surfaced the same way the
    general (non-ROCm) fallback already is, not silently skipped (AGENTS.md
    rule 5)."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])

    url, sha = sl._resolve_backend_asset("amd-rocm")

    assert url == sl.DEFAULT_URL
    out = capsys.readouterr().out.lower()
    assert "rocm" in out and "could not find" in out, (
        f"a failed lemonade-sdk release lookup must be surfaced; got: {out!r}")


def test_resolve_amd_rocm_asset_warns_when_gfx103x_asset_missing(monkeypatch, capsys):
    """The release lookup can succeed yet still not contain the expected
    gfx103X asset (e.g. only other GPU variants are listed) - that must warn
    too, not just the empty-list case."""
    dummy_assets = [{
        "name": "llama-b1288-windows-rocm-gfx110X-x64.zip",
        "browser_download_url": "https://dummy.github/releases/download/b1288/llama-rocm-gfx110X.zip",
        "digest": "sha256:othergpuvariant",
    }]
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: dummy_assets)

    url, sha = sl._resolve_backend_asset("amd-rocm")

    assert url == sl.DEFAULT_URL
    out = capsys.readouterr().out.lower()
    assert "rocm" in out and "could not find" in out, (
        f"a release with no gfx103X asset must be surfaced; got: {out!r}")


def test_resolve_amd_rocm_asset_no_warning_when_release_found(monkeypatch, capsys):
    dummy_assets = [{
        "name": "llama-b1288-windows-rocm-gfx103X-x64.zip",
        "browser_download_url": "https://dummy.github/releases/download/b1288/llama-rocm.zip",
        "digest": "sha256:dummyrocmsha",
    }]
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: dummy_assets)

    url, sha = sl._resolve_backend_asset("amd-rocm")

    assert url == "https://dummy.github/releases/download/b1288/llama-rocm.zip"
    assert sha == "dummyrocmsha"
    out = capsys.readouterr().out.lower()
    assert "could not find" not in out


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


def test_custom_url_warning_printed(monkeypatch, tmp_path):
    """A custom --url with no --sha256 must warn. The runtime-lib target is
    isolated to tmp_path, not the real repo runtime dir: on a box that already
    has a build provisioned there (the shared venv's editable install resolves
    _repo_runtime_lib() to it), the exists() check would pass without this
    test's own fetch ever writing anything, falling through to a REAL
    `_install_runtime_wheel` (a real `uv pip install -e`) that dumps the
    runtime wheel's native DLLs into this test's own tmp_path-scoped uv cache."""
    target = tmp_path / "lib"
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)

    def fake_fetch_and_place(url, target, sha256=None):
        (target / sl._lib_name()).write_bytes(b"stub")
        return 1

    monkeypatch.setattr(sl, "_fetch_and_place", fake_fetch_and_place)
    monkeypatch.setattr(sl, "_clear_target", lambda target: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg_dir: True)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    monkeypatch.setattr(sl, "_verify", lambda: None)

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(sl.main, ["--url", "https://dummy.invalid/llama.zip", "--force"])
    assert "Warning: Custom URL download is unverified" in result.output
