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
from tests._fake_https import patch_https_transport


# --------------------------- hwdetect ------------------------------------- #

def test_detect_returns_valid_shape():
    det = hwdetect.detect()                       # must never raise
    assert det.recommended in ("vulkan", "cpu", "metal")
    assert set(det.vendors) <= set(hwdetect.VENDORS) | {"apple"}
    assert det.has_gpu == bool(det.vendors)


def test_recommended_install_backend_policy(monkeypatch):
    """The ONE installer-backend policy both setup.bat and setup.sh share."""
    # The NO-SYSTEM-TOOLKIT baseline, deterministic regardless of the
    # test-running machine's own ROCm install. The with-a-detected-toolkit
    # escalation to hip has its own tests.
    monkeypatch.setattr(hwdetect, "_rocm_toolkit_present", lambda: False)
    def rec(vendors, names, platform="win32"):
        monkeypatch.setattr(hwdetect.sys, "platform", platform)
        return hwdetect.recommended_install_backend(
            hwdetect.Detection(vendors=vendors, gpu_names=names))
    # AMD on Windows: RX 6000 / unknown keep the self-contained gfx103X ROCm build...
    assert rec(["amd"], "amd radeon rx 6900 xt") == "amd-rocm"
    assert rec(["amd"], "amd radeon graphics") == "amd-rocm"
    # ...but a CLEARLY non-gfx103X AMD with no ROCm/HIP toolkit detected
    # downgrades to the universal Vulkan build.
    assert rec(["amd"], "amd radeon rx 7800 xt") == "vulkan"
    assert rec(["amd"], "amd radeon rx 9070") == "vulkan"
    assert rec(["amd"], "amd radeon rx 5700") == "vulkan"
    # AMD on Linux with no toolkit detected is vulkan too: no self-contained
    # bundle exists there and hip needs the toolkit this asserts is absent.
    assert rec(["amd"], "amd radeon rx 6900 xt", platform="linux") == "vulkan"
    # NVIDIA: cuda on BOTH Windows and Linux - llama.cpp ships a self-contained
    # cudart bundle on both.
    assert rec(["nvidia"], "nvidia geforce rtx 4090") == "cuda"
    assert rec(["nvidia"], "nvidia geforce rtx 4090", platform="linux") == "cuda"
    # Intel -> vulkan; no GPU -> cpu; Apple Silicon -> metal.
    assert rec(["intel"], "intel arc a770") == "vulkan"
    assert rec([], "") == "cpu"
    assert rec(["apple"], "", platform="darwin") == "metal"


def test_hwdetect_cli_prints_vendor_and_backend(capsys):
    """`python -m localm.hwdetect` emits '<vendor> <install-backend>' for the shells."""
    assert hwdetect.main() == 0
    out = capsys.readouterr().out.strip().split()
    assert len(out) == 2
    assert out[1] in ("vulkan", "cuda", "cpu", "metal", "amd-rocm", "hip")


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


def test_auto_backend_nvidia_is_cuda_on_linux(monkeypatch):
    # bare `setup-llama` (auto) must match the installer: NVIDIA on Linux -> cuda.
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia"], "vulkan"))
    assert sl._auto_backend() == "cuda"


def test_auto_backend_mixed_amd_nvidia(monkeypatch):
    # A box with both: cuda on either OS, since NVIDIA is the priority vendor
    # and its cudart bundle is self-contained on both.
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["nvidia", "amd"], "vulkan"))
    monkeypatch.setattr(hwdetect.sys, "platform", "win32")
    assert sl._auto_backend() == "cuda"
    monkeypatch.setattr(hwdetect.sys, "platform", "linux")
    assert sl._auto_backend() == "cuda"


@pytest.mark.skipif(sys.platform != "win32", reason="amd-rocm bundled build is Windows-only")
def test_auto_backend_amd_only_is_rocm_on_windows(monkeypatch):
    monkeypatch.setattr(hwdetect, "detect", _fake_detect(["amd"], "vulkan"))
    assert sl._auto_backend() == "amd-rocm"


# ------------------- _ASSET_MATCH vs the real upstream release ------------- #

@pytest.mark.integration
def test_every_asset_fragment_resolves_against_the_real_latest_release():
    """Every substring in _ASSET_MATCH must match at least one real asset name
    in the CURRENT live ggml-org/llama.cpp release - not a cached snapshot, not
    a fixture. An upstream rename makes the old fragment stop matching, and
    _resolve_backend_asset then falls through to a GUESSED url (the "could not
    verify release asset list" branch) that 404s against the live tag.

    Scoped to _ASSET_MATCH's own generic upstream-repo resolution. TWO
    combinations are excluded, both documented at their own call sites in
    setup_llama.py:
      * linux/cuda - ggml-org publishes no bare Linux CUDA binary at all;
        _resolve_backend_asset special-cases this combination and resolves
        against hybridgroup/llama-cpp-builder BEFORE reaching _ASSET_MATCH, so
        that entry is unreachable data rather than a real matcher.
      * amd-rocm is not in _ASSET_MATCH at all - it resolves against
        lemonade-sdk/llamacpp-rocm via its own special case, checked
        separately."""
    tag = sl._latest_tag()
    assets = sl._release_assets(tag)
    names = [str(a.get("name", "")).lower() for a in assets]
    assert names, (
        f"could not fetch a real asset listing for tag {tag!r} - a network "
        "problem here means this test proves nothing, not that everything "
        "matched; investigate rather than treat this as a pass")

    excluded = {("linux", "cuda")}
    unmatched = []
    for plat, backends in sl._ASSET_MATCH.items():
        for backend, matchers in backends.items():
            if (plat, backend) in excluded:
                continue
            # cuda on win32 is a dict keyed by cuda_line, not a flat list.
            groups = matchers.values() if isinstance(matchers, dict) else [matchers]
            for group in groups:
                if not any(m in n for m in group for n in names):
                    unmatched.append((plat, backend, group))

    assert not unmatched, (
        f"these _ASSET_MATCH fragments match NOTHING in the real {tag!r} "
        f"release (upstream likely renamed an asset): {unmatched}")


# --------------------------- TLS verification (SSL) ----------------------- #
# These lock that setup-llama's GitHub calls pass a verifying SSL context.

def test_release_assets_passes_verifying_context(monkeypatch):
    # The GitHub API lookups must present a verifying SSL context. Without one,
    # a box whose OS cert store lacks the release-CDN CA fails every call with
    # CERTIFICATE_VERIFY_FAILED.
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"assets": []}'

    def fake_urlopen(req, timeout=None, context=None):
        seen["context"] = context
        return _Resp()

    patch_https_transport(monkeypatch, fake_urlopen)
    sl._release_assets("btag")
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_download_passes_verifying_context(monkeypatch, tmp_path):
    # The archive download itself must verify too.
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

    patch_https_transport(monkeypatch, fake_urlopen)
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
    patch_https_transport(monkeypatch, _boom)

    plat = sl._platform_key()
    # 'vulkan' exists on win/linux; on darwin use 'cpu'.
    backend = "vulkan" if plat in ("win32", "linux") else "cpu"
    url = sl._resolve_backend_url(backend)
    assert sl._PINNED_TAG in url
    assert backend.replace("amd-rocm", "rocm") in url or "macos" in url
    assert url.startswith(f"https://github.com/{sl._UPSTREAM_REPO}/releases/download/")


def test_release_assets_logs_the_swallowed_cause(monkeypatch, caplog):
    """_release_assets() must not silently swallow the failure: the exception
    must be discoverable at debug level, not just an empty list."""
    import logging

    def _boom(*a, **k):
        raise OSError("simulated DNS failure")
    patch_https_transport(monkeypatch, _boom)

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

    patch_https_transport(monkeypatch,
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

    patch_https_transport(monkeypatch,
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

    url, sha, _tag = sl._resolve_backend_asset("vulkan")
    assert url == "https://dummy.github/releases/download/b9870/llama-vulkan.zip"
    assert sha == "dummysha256value"


def test_resolve_backend_asset_fallback_uses_pinned_sha256(monkeypatch):
    """A DEFAULT install stays checksum-verified when the release listing is
    unavailable. The API and the download CDN are different hosts, so "the
    listing is unreachable but the download works" is a real state (a rate limit
    is the usual way in) - and it is precisely the state in which shipping an
    unverified binary would be least acceptable."""
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")

    url, sha, _tag = sl._resolve_backend_asset("vulkan")
    fname = f"llama-{sl._PINNED_TAG}-bin-win-vulkan-x64.zip"
    assert fname in url
    assert sha == sl._PINNED_FALLBACK_SHA256[fname]
    assert sha, "the pin must ship a checksum, or the offline path is unverified"


@pytest.mark.parametrize("plat, backend", [
    ("win32", "cpu"), ("win32", "vulkan"), ("win32", "sycl"), ("win32", "hip"),
    ("linux", "cpu"), ("linux", "vulkan"), ("linux", "sycl"), ("linux", "hip"),
    ("darwin", "cpu"), ("darwin", "metal"),
])
def test_offline_resolution_names_a_real_pinned_asset_for_every_backend(
        monkeypatch, plat, backend):
    """WITH THE LISTING UNAVAILABLE, every backend must still resolve to a
    filename the pinned table actually has - i.e. to a real asset of the pinned
    release, with a checksum.

    Catches an upstream RENAME, which a templated guess produces a 404 URL and
    no checksum for. Driven per (platform, backend) rather than for one
    representative, since a rename hits exactly one cell of that grid.

    linux/cuda is excluded: upstream publishes no bare Linux CUDA asset, so it
    resolves against a third party entirely."""
    monkeypatch.setattr(sl, "_platform_key", lambda: plat)
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])

    url, sha, tag = sl._resolve_backend_asset(backend)

    fname = url.rsplit("/", 1)[-1]
    assert tag == sl._PINNED_TAG
    assert fname in sl._PINNED_FALLBACK_SHA256, (
        f"{plat}/{backend} offline resolved to {fname!r}, which is not an asset "
        f"of the pinned release {sl._PINNED_TAG}")
    assert sha == sl._PINNED_FALLBACK_SHA256[fname]


def test_every_pinned_asset_belongs_to_a_pinned_tag():
    """The offline table looks assets up by FILENAME, so an entry for a tag
    nothing installs can never be hit while its presence reads as coverage.

    Enforced across every backend, not for ROCm alone, so bumping the pin
    without refreshing the digests fails here."""
    known = {sl._PINNED_TAG, sl._ROCM_TAG}
    stale = [k for k in sl._PINNED_FALLBACK_SHA256
             # cudart bundles carry no tag in their names: upstream re-uploads
             # the same file every release, so they are legitimately tag-free.
             if not k.startswith("cudart-")
             and not any(f"-{t}-" in k or k.startswith(f"llama-{t}-") for t in known)]
    assert not stale, (
        f"pinned assets for a tag localm does not pin ({', '.join(sorted(known))}): "
        f"{stale}")


def test_resolve_amd_rocm_asset_warns_when_release_lookup_fails(monkeypatch, capsys):
    """A failed lemonade-sdk release lookup must be surfaced the same way the
    general (non-ROCm) fallback is, not silently skipped."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: [])

    url, sha, _tag = sl._resolve_backend_asset("amd-rocm")

    assert url == sl.DEFAULT_URL
    out = capsys.readouterr().out.lower()
    assert "rocm" in out and "could not find" in out, (
        f"a failed lemonade-sdk release lookup must be surfaced; got: {out!r}")


def test_resolve_amd_rocm_asset_warns_when_gfx103x_asset_missing(monkeypatch, capsys):
    """The release lookup can succeed yet still not contain the expected
    gfx103X asset (e.g. only other GPU variants are listed) - that must warn
    too, not just the empty-list case."""
    dummy_assets = [{
        "name": "llama-b1307-windows-rocm-gfx110X-x64.zip",
        "browser_download_url": "https://dummy.github/releases/download/b1307/llama-rocm-gfx110X.zip",
        "digest": "sha256:othergpuvariant",
    }]
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: dummy_assets)

    url, sha, _tag = sl._resolve_backend_asset("amd-rocm")

    assert url == sl.DEFAULT_URL
    out = capsys.readouterr().out.lower()
    assert "rocm" in out and "could not find" in out, (
        f"a release with no gfx103X asset must be surfaced; got: {out!r}")


def test_resolve_amd_rocm_asset_no_warning_when_release_found(monkeypatch, capsys):
    dummy_assets = [{
        "name": "llama-b1307-windows-rocm-gfx103X-x64.zip",
        "browser_download_url": "https://dummy.github/releases/download/b1307/llama-rocm.zip",
        "digest": "sha256:dummyrocmsha",
    }]
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: dummy_assets)

    url, sha, _tag = sl._resolve_backend_asset("amd-rocm")

    assert url == "https://dummy.github/releases/download/b1307/llama-rocm.zip"
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
    monkeypatch.setattr(sl, "_resolve_backend_asset", lambda backend, cuda_line=None, tag=None: ("https://dummy.url", "dummysha", "bTEST"))

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


def test_sycl_backend_note_is_platform_conditional(monkeypatch):
    """The pinned b10375 Windows SYCL zip (llama-b10375-bin-win-sycl-x64.zip)
    bundles the whole oneAPI DPC++ runtime alongside ggml-sycl.dll (sycl8.dll,
    mkl_core.2.dll, mkl_sycl_blas.5.dll, ur_adapter_level_zero.dll,
    ur_adapter_opencl.dll, tbb12.dll, libiomp5md.dll, dnnl.dll, sycl-ls.exe,
    ...) - confirmed by downloading and inspecting the archive against its
    pinned sha256. The matching Linux tarball ships only libggml-sycl.so with
    none of that, so a system oneAPI install is still required there. A prior
    version of this note claimed the Linux-only fact ('needs the oneAPI
    runtime present') unconditionally on every platform, which was never true
    on Windows."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    assert sl._sycl_backend_note() == "Intel oneAPI build + self-contained oneAPI runtime"
    monkeypatch.setattr(sl.sys, "platform", "linux")
    assert sl._sycl_backend_note() == "Intel oneAPI build (needs the oneAPI runtime present)"


def test_help_text_does_not_claim_nvidia_amd_default_to_vulkan_on_linux():
    """Pins the live `--backend` --help string against the b8878c2b / 22cabce0
    policy flip: NVIDIA now gets cuda on Linux too (not vulkan), and AMD gets
    hip on any OS when a system ROCm/HIP toolkit is detected (not just vulkan).
    A prior version of this text claimed 'vulkan for Intel and for NVIDIA/AMD
    on Linux', which stopped being true the moment recommended_install_backend()
    changed - this test exists so the next such flip cannot silently leave the
    printed --help text behind again."""
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(sl.main, ["--help"])
    assert result.exit_code == 0
    help_text = " ".join(result.output.split())     # collapse click's line-wrapping
    assert "vulkan for Intel and for NVIDIA" not in help_text
    assert "cuda for NVIDIA on both Windows and Linux" in help_text
    assert "hip for AMD" in help_text


# --------------------------- bad-download diagnosis ----------------------- #
#
# _sniff_content_kind and _diagnose_bad_artifact tell the causes of a too-small
# or invalid download apart from real evidence (actual bytes, actual response
# metadata), and name no confident cause when the evidence is ambiguous.

import gzip
import io
import zipfile as _zipfile_mod


def _real_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with _zipfile_mod.ZipFile(buf, "w") as zf:
        zf.writestr("bin/llama.dll", b"x" * 5000)
        zf.writestr("bin/ggml.dll", b"y" * 5000)
    return buf.getvalue()


def _real_gzip_bytes() -> bytes:
    return gzip.compress(b"not actually a tar but real gzip framing" * 200)


class TestSniffContentKind:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"")
        assert sl._sniff_content_kind(p) == "empty"

    def test_truncated_real_zip_is_zip_truncated_not_binary(self, tmp_path):
        # A GENUINE zip, cut short mid-stream - starts with the real local-file
        # header magic, so this must be told apart from a substituted page.
        full = _real_zip_bytes()
        assert not _zipfile_mod.is_zipfile(io.BytesIO(full[:50])), "sanity: our cut is actually incomplete"
        p = tmp_path / "a.zip"
        p.write_bytes(full[:50])
        assert sl._sniff_content_kind(p) == "zip_truncated"

    def test_truncated_real_gzip_is_gzip_truncated(self, tmp_path):
        full = _real_gzip_bytes()
        p = tmp_path / "a.tar.gz"
        p.write_bytes(full[:10])
        assert sl._sniff_content_kind(p) == "gzip_truncated"

    def test_html_with_doctype(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"<!DOCTYPE html>\n<html><body>Access to this content is blocked "
                       b"by your network administrator.</body></html>")
        assert sl._sniff_content_kind(p) == "html"

    def test_html_without_doctype(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"<html><head><title>Blocked</title></head><body>Denied</body></html>")
        assert sl._sniff_content_kind(p) == "html"

    def test_html_non_utf8_encoding_still_detected(self, tmp_path):
        # A block page served as windows-1252 (a smart quote / accented char
        # later in the body) must NOT be misclassified as 'binary' just
        # because a STRICT utf-8 decode of the whole peek window would fail -
        # the structural marker at the start is pure ASCII and must win first.
        body = "<!DOCTYPE html><html><body>Café access blocked</body></html>"
        p = tmp_path / "a"
        p.write_bytes(body.encode("windows-1252"))
        assert sl._sniff_content_kind(p) == "html"

    def test_json_object(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b'{"error": "forbidden", "code": 403}')
        assert sl._sniff_content_kind(p) == "json"

    def test_json_array(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b'[{"message": "blocked"}]')
        assert sl._sniff_content_kind(p) == "json"

    def test_xml_s3_style_error(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'
                       b'<Error><Code>AccessDenied</Code></Error>')
        assert sl._sniff_content_kind(p) == "xml"

    def test_plain_text_no_markup(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"This content is not available in your region.\n")
        assert sl._sniff_content_kind(p) == "text"

    def test_genuine_binary_garbage_is_honestly_unclear(self, tmp_path):
        # Bytes that are neither a known archive signature nor decodable text -
        # a genuinely ambiguous case. Must NOT be misclassified as any specific
        # kind (that would be a confident wrong guess).
        p = tmp_path / "a"
        p.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81, 0xC0, 0xC1]) * 100)
        assert sl._sniff_content_kind(p) == "binary"


class TestDiagnoseBadArtifact:
    def test_html_cause_mentions_network_filtering_not_github(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"<!DOCTYPE html><html>blocked</html>")
        msg = sl._diagnose_bad_artifact(p, None)
        low = msg.lower()
        assert "html" in low or "webpage" in low
        assert "network" in low or "proxy" in low or "filter" in low
        assert "bytes received" in low

    def test_truncated_zip_cause_says_interrupted_not_blocked(self, tmp_path):
        full = _real_zip_bytes()
        p = tmp_path / "a.zip"
        p.write_bytes(full[:50])
        msg = sl._diagnose_bad_artifact(p, None).lower()
        assert "interrupted" in msg or "dropped" in msg or "throttled" in msg
        # Must NOT claim this is a deliberate block - the evidence (valid
        # archive framing, just cut short) does not support that conclusion.
        assert "block" not in msg or "not a deliberate block" in msg

    def test_ambiguous_binary_is_honest_not_a_confident_guess(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(bytes([0xFF, 0xFE, 0x80, 0x81]) * 200)
        msg = sl._diagnose_bad_artifact(p, None).lower()
        assert "not clear" in msg or "does not clearly indicate" in msg
        # Must not fabricate a specific cause it has no evidence for.
        assert "corporate" not in msg
        assert "html" not in msg

    def test_includes_response_metadata_when_available(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"<html>blocked</html>")
        dl = sl._DownloadResult(bytes_received=21, content_length=33554432,
                                content_type="text/html; charset=utf-8",
                                final_url="https://proxy.corp.example/blocked")
        msg = sl._diagnose_bad_artifact(p, dl)
        assert "33554432" in msg
        assert "text/html" in msg
        assert "proxy.corp.example" in msg

    def test_no_content_length_is_stated_honestly(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"\x00" * 10)
        dl = sl._DownloadResult(bytes_received=10, content_length=0,
                                content_type="", final_url="")
        msg = sl._diagnose_bad_artifact(p, dl)
        assert "no content-length given" in msg.lower()


class TestValidateArchiveRichDiagnosis:
    def test_too_small_html_error_names_the_real_cause(self, tmp_path):
        p = tmp_path / "a.zip"
        p.write_bytes(b"<!DOCTYPE html><html>Access blocked by policy</html>")
        with pytest.raises(sl.ArtifactError) as exc_info:
            sl._validate_archive(p)
        msg = str(exc_info.value).lower()
        assert "too small" in msg
        assert "html" in msg or "webpage" in msg

    def test_shape_check_also_gets_rich_diagnosis(self, tmp_path):
        # Big enough to pass the SIZE gate, but still not a real archive: the
        # shape check's message must also be evidence-based rather than a hedge
        # naming every possible cause at once.
        p = tmp_path / "a.zip"
        p.write_bytes(b"<!DOCTYPE html>" + b" " * sl._MIN_ARTIFACT_BYTES)
        with pytest.raises(sl.ArtifactError) as exc_info:
            sl._validate_archive(p)
        msg = str(exc_info.value).lower()
        assert "not a valid zip or tar" in msg
        assert "html" in msg or "webpage" in msg

    def test_valid_archive_passes_regardless_of_diagnosis_machinery(self, tmp_path):
        # STORED (uncompressed) so the on-disk size is genuinely >= the floor -
        # padding bytes AFTER a real zip's end (rather than growing an entry
        # inside it) would move the End-Of-Central-Directory record and make
        # zipfile correctly reject it as a different, real corruption.
        buf = io.BytesIO()
        with _zipfile_mod.ZipFile(buf, "w") as zf:
            zf.writestr("bin/llama.dll", b"x" * (sl._MIN_ARTIFACT_BYTES + 1000),
                       compress_type=_zipfile_mod.ZIP_STORED)
        p = tmp_path / "a.zip"
        p.write_bytes(buf.getvalue())
        assert p.stat().st_size >= sl._MIN_ARTIFACT_BYTES, "sanity: actually exceeds the floor"
        sl._validate_archive(p)   # must not raise


class TestDownloadTransportDiagnosis:
    def test_connection_reset_mid_transfer_is_distinct_from_stall(self, monkeypatch, tmp_path):
        """A live connection drop (ConnectionResetError, an OSError - NOT a
        timeout) must be reported as an interrupted connection, not confused
        with a stalled/frozen one, and must include how many bytes DID arrive."""
        class _DropResp:
            headers = {"Content-Length": "1000000", "Content-Type": "application/zip"}
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def geturl(self): return "https://example/big.zip"
            _sent = False
            def read(self, n=-1):
                if not type(self)._sent:
                    type(self)._sent = True
                    return b"x" * 1000
                raise ConnectionResetError("connection reset by peer")

        patch_https_transport(monkeypatch,
                              lambda req, timeout=None, context=None: _DropResp())
        with pytest.raises(sl.ArtifactError) as exc_info:
            sl._download("https://example/big.zip", tmp_path / "a.zip")
        msg = str(exc_info.value).lower()
        assert "interrupted" in msg
        assert "1000" in msg                 # bytes received so far, not hidden
        assert "stalled" not in msg          # must not be conflated with the timeout case

    def test_clean_short_completion_returns_full_metadata(self, monkeypatch, tmp_path):
        """A response that ends normally (no exception) but is short of its
        own Content-Length is NOT an error at the _download level - the
        caller (_validate_archive) decides that, using the metadata returned
        here. Confirms content_type and final_url are actually captured."""
        class _ShortResp:
            headers = {"Content-Length": "33554432", "Content-Type": "text/html; charset=utf-8"}
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def geturl(self): return "https://proxy.corp.example/blocked-notice"
            _sent = False
            def read(self, n=-1):
                if type(self)._sent:
                    return b""
                type(self)._sent = True
                return b"<html>blocked</html>"

        patch_https_transport(monkeypatch,
                              lambda req, timeout=None, context=None: _ShortResp())
        dl = sl._download("https://github.com/x/releases/download/y/z.zip", tmp_path / "a.zip")
        assert dl.bytes_received == len(b"<html>blocked</html>")
        assert dl.content_length == 33554432
        assert dl.content_type == "text/html; charset=utf-8"
        assert dl.final_url == "https://proxy.corp.example/blocked-notice"

    def test_geturl_missing_falls_back_gracefully(self, monkeypatch, tmp_path):
        """A response object without geturl() (an unusual test double, or a
        future urllib change) must not crash the download - the final URL is
        a diagnostic nice-to-have, never load-bearing."""
        class _NoGeturlResp:
            headers = {"Content-Length": "4"}
            def __enter__(self): return self
            def __exit__(self, *a): return False
            _sent = False
            def read(self, n=-1):
                if type(self)._sent:
                    return b""
                type(self)._sent = True
                return b"data"

        patch_https_transport(monkeypatch,
                              lambda req, timeout=None, context=None: _NoGeturlResp())
        dl = sl._download("https://example/x.zip", tmp_path / "a.zip")
        assert dl.final_url == "https://example/x.zip"   # fell back to the request URL


class TestVulkanCpuTerminalGuidance:
    """The explicit-vulkan/cpu-pick-failed exit must name the --from/--url
    escape hatches before exiting, like every other failure path in this
    function."""

    def test_vulkan_not_provisioned_shows_escape_hatches_and_exits(self, monkeypatch, tmp_path, capsys):
        def fake_provision_backend(chosen, target, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
            raise sl.ArtifactError("download is too small (196608 bytes < 262144 minimum): "
                                   "the response is an HTML page, not the archive.")

        monkeypatch.setattr(sl, "_provision_backend", fake_provision_backend)
        monkeypatch.setattr(sl, "_clear_target", lambda target: None)

        with pytest.raises(SystemExit) as exc_info:
            sl._provision_with_fallback("vulkan", tmp_path, None, False)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "--from" in out
        assert "--url" in out
        assert "vulkan" in out
        # The underlying specific cause (from _diagnose_bad_artifact via the
        # exception message) must still be visible, not swallowed by the new
        # guidance replacing it.
        assert "HTML page" in out or "html page" in out.lower()


class TestIssue827Reproduction:
    """End-to-end reproduction of the exact reported failure: picking vulkan
    on a network that substitutes a small response for the real archive."""

    def test_reproduces_and_correctly_diagnoses_the_reported_failure(self, monkeypatch, tmp_path):
        # The reported byte counts: 196608 received where 262144 is the floor and
        # the real archive is ~32MB, modelled here as an HTML block page. The
        # filler has to be long enough BEFORE slicing, or the slice is a no-op and
        # the body ends up shorter than the exact count asserted below.
        prefix = b"<!DOCTYPE html><html><body>"
        filler = b"Access to this file is restricted by your organization's security policy. " * 5000
        assert len(prefix + filler) >= 196608, "sanity: filler exceeds the slice target"
        body = (prefix + filler)[:196608]
        assert len(body) == 196608, "sanity: reproduces the EXACT reported byte count"
        assert len(body) < sl._MIN_ARTIFACT_BYTES, "sanity: reproduces the undersized report"

        class _CorpProxyResp:
            headers = {"Content-Length": "33554432", "Content-Type": "text/html; charset=UTF-8"}
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def geturl(self): return "https://github.com/ggml-org/llama.cpp/releases/download/b10092/llama-b10092-bin-win-vulkan-x64.zip"
            _sent = False
            def read(self, n=-1):
                if type(self)._sent:
                    return b""
                type(self)._sent = True
                return body

        patch_https_transport(monkeypatch,
                              lambda req, timeout=None, context=None: _CorpProxyResp())

        url = "https://github.com/ggml-org/llama.cpp/releases/download/b10092/llama-b10092-bin-win-vulkan-x64.zip"
        with pytest.raises(sl.ArtifactError) as exc_info:
            sl._fetch_and_place(url, tmp_path / "target")
        msg = str(exc_info.value).lower()
        # Must be SPECIFIC, naming the actual cause, rather than a generic hedge.
        assert "webpage" in msg or "html" in msg
        assert "network" in msg or "proxy" in msg or "filter" in msg
        assert "196608" in msg or "196,608" in str(exc_info.value)
        assert "33554432" in msg


# --------------------------------------------------------------------------- #
#  The pinned AMD build and its checksum must not drift apart
# --------------------------------------------------------------------------- #

def test_default_url_tag_and_pin_are_coherent():
    """DEFAULT_URL, _ROCM_TAG, DEFAULT_URL_SHA256 and the pinned table are four
    places that name the same artifact. Leaving one behind on a bump yields an
    offline fallback that downloads one file and verifies it against another's
    hash - a hard failure with no release-API access, and invisible online,
    where the digest comes off the API instead. Asserts they agree."""
    fname = sl.DEFAULT_URL.rsplit("/", 1)[-1]

    assert f"/{sl._ROCM_TAG}/" in sl.DEFAULT_URL, (
        f"DEFAULT_URL {sl.DEFAULT_URL!r} does not point at tag {sl._ROCM_TAG!r}")
    assert sl._ROCM_TAG in fname, (
        f"asset name {fname!r} does not carry tag {sl._ROCM_TAG!r}")
    assert fname in sl._PINNED_FALLBACK_SHA256, (
        f"{fname!r} has no entry in _PINNED_FALLBACK_SHA256")
    assert sl._PINNED_FALLBACK_SHA256[fname] == sl.DEFAULT_URL_SHA256, (
        "DEFAULT_URL_SHA256 disagrees with the pinned table entry for the same file")
    assert len(sl.DEFAULT_URL_SHA256) == 64, "sha256 pins are 64 hex chars"


def test_every_rocm_pin_matches_the_current_tag():
    """A stale pin from a previous build is dead weight that reads as coverage:
    the offline path looks up by FILENAME, so an entry for an older tag can
    never be hit, while its presence suggests that build is still supported."""
    # Match the lemonade-sdk naming (`-rocm-gfx<target>-`) specifically. A bare
    # "rocm" substring also catches upstream's own ggml-org asset
    # `llama-<tag>-bin-ubuntu-rocm-7.2-x64.tar.gz`, which is pinned against
    # _PINNED_TAG and has nothing to do with _ROCM_TAG.
    rocm = [k for k in sl._PINNED_FALLBACK_SHA256 if "-rocm-gfx" in k]
    assert rocm, "the pinned table should still carry the lemonade ROCm assets"
    stale = [k for k in rocm if sl._ROCM_TAG not in k]
    assert not stale, f"pinned ROCm assets for a tag other than {sl._ROCM_TAG}: {stale}"
    # And upstream's OWN rocm assets must track _PINNED_TAG, so this test cannot
    # be satisfied by simply deleting entries.
    upstream_rocm = [k for k in sl._PINNED_FALLBACK_SHA256
                     if "rocm" in k and "-rocm-gfx" not in k]
    assert upstream_rocm, "upstream's own rocm assets should be pinned too"
    assert all(sl._PINNED_TAG in k for k in upstream_rocm), upstream_rocm
