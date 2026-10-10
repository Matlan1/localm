# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing and choosing the stable-diffusion.cpp runtime: verification,
load-test markers, the auto fallback order, explicit choices, and the network
policy gate. Archives are real zip files built here; the download itself and the
worker load-test are replaced where a test says so."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from localm import hwdetect
from localm.media.sdcpp import _binding, pins, runtime


def _make_zip(path: Path, *, nested: str = "") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{nested}/" if nested else ""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(prefix + _binding.lib_filename(), os.urandom(300 * 1024))
        zf.writestr(prefix + "ggml.txt", "license")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """Point every backend's asset at a local zip and serve "downloads" by copy."""
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    served = {}
    for backend in ("cpu", "vulkan", "rocm", "cuda"):
        name = f"fake-{backend}.zip"
        sha = _make_zip(tmp_path / "release" / name)
        served[name] = tmp_path / "release" / name
        monkeypatch.setitem(pins.ASSETS, ("windows", backend), (name, sha))
    monkeypatch.setitem(pins.EXTRA_ASSETS, ("windows", "cuda"), [])
    calls = []

    def fake_download(url, dest, on_progress, label):
        calls.append(label)
        shutil.copyfile(served[label], dest)

    monkeypatch.setattr(runtime, "_download", fake_download)
    return calls


def _ok_probe(rt):
    return {"devices": [["Dev0", f"{rt.backend} device"]]}


def test_install_verifies_extracts_and_marks(fake_release):
    rt = runtime.install("vulkan", probe=_ok_probe)
    assert rt.path == runtime.runtime_dir("vulkan")
    assert (rt.path / _binding.lib_filename()).is_file()
    meta = json.loads((rt.path / runtime.MARKER).read_text(encoding="utf-8"))
    assert meta["tag"] == pins.TAG and meta["backend"] == "vulkan"
    assert runtime.installed("vulkan").devices == [["Dev0", "vulkan device"]]
    assert runtime.resolve("vulkan").backend == "vulkan"


def test_an_installed_runtime_is_not_downloaded_again(fake_release):
    runtime.install("cpu", probe=_ok_probe)
    runtime.install("cpu", probe=_ok_probe)
    assert fake_release == ["fake-cpu.zip"]
    runtime.install("cpu", probe=_ok_probe, force=True)
    assert fake_release == ["fake-cpu.zip", "fake-cpu.zip"]


def test_a_wrong_sha256_refuses_and_installs_nothing(fake_release, monkeypatch):
    name, _sha = pins.ASSETS[("windows", "vulkan")]
    monkeypatch.setitem(pins.ASSETS, ("windows", "vulkan"), (name, "0" * 64))
    with pytest.raises(runtime.DownloadError, match="sha256 does not match"):
        runtime.install("vulkan", probe=_ok_probe)
    assert not runtime.runtime_dir("vulkan").exists()
    assert runtime.installed("vulkan") is None


def test_an_archive_with_one_top_folder_is_flattened(fake_release, tmp_path, monkeypatch):
    name = "nested.zip"
    sha = _make_zip(tmp_path / "release2" / name, nested="sd-bin")
    monkeypatch.setitem(pins.ASSETS, ("windows", "cpu"), (name, sha))
    monkeypatch.setattr(runtime, "_download",
                        lambda url, dest, p, label: shutil.copyfile(tmp_path / "release2" / name, dest))
    rt = runtime.install("cpu", probe=_ok_probe)
    assert (rt.path / _binding.lib_filename()).is_file()


def test_a_failed_load_test_is_recorded_and_not_installed(fake_release):
    def bad_probe(rt):
        raise RuntimeError("hipblas.dll not found")

    with pytest.raises(runtime.ProvisionError, match="hipblas.dll not found"):
        runtime.install("rocm", probe=bad_probe)
    assert runtime.installed("rocm") is None
    assert "hipblas.dll not found" in runtime.load_test_failed("rocm")


def test_a_runtime_without_devices_is_refused(fake_release):
    with pytest.raises(runtime.ProvisionError, match="no compute device"):
        runtime.install("vulkan", probe=lambda rt: {"devices": []})
    assert runtime.installed("vulkan") is None


def test_auto_falls_back_to_vulkan_when_the_recommended_backend_fails(fake_release, monkeypatch):
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "rocm")
    said = []

    def probe(rt):
        if rt.backend == "rocm":
            raise RuntimeError("no hipblas")
        return _ok_probe(rt)

    rt = runtime.provision("auto", on_progress=said.append, probe=probe)
    assert rt.backend == "vulkan"
    assert any("rocm runtime did not work" in m for m in said)
    rt2 = runtime.provision("auto", probe=probe)
    assert rt2.backend == "vulkan"
    assert fake_release.count("fake-rocm.zip") == 1


def test_an_explicit_backend_is_never_swapped_for_another(fake_release):
    def probe(rt):
        raise RuntimeError("no hipblas")

    with pytest.raises(runtime.ProvisionError, match="no hipblas"):
        runtime.provision("rocm", probe=probe)
    assert runtime.installed("vulkan") is None
    assert "fake-vulkan.zip" not in fake_release


def test_a_download_failure_stops_auto_without_trying_other_backends(fake_release, monkeypatch):
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "rocm")

    def broken(url, dest, on_progress, label):
        fake_release.append(label)
        raise runtime.DownloadError("connection reset")

    monkeypatch.setattr(runtime, "_download", broken)
    with pytest.raises(runtime.DownloadError, match="connection reset"):
        runtime.provision("auto", probe=_ok_probe)
    assert fake_release == ["fake-rocm.zip"]


def test_net_mode_off_refuses_before_any_connection(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    import localm.http_ssl as http_ssl
    import localm.netpolicy as netpolicy
    calls = []

    def no_network(*a, **k):
        calls.append(a)
        raise AssertionError("a connection was attempted")

    monkeypatch.setattr(netpolicy, "pinned_request", no_network)
    monkeypatch.setattr(http_ssl, "verified_urlopen", no_network)
    with pytest.raises(runtime.DownloadError, match="network policy"):
        runtime.install("cpu", probe=_ok_probe)
    assert calls == []
    assert not runtime.runtime_dir("cpu").exists()


def test_unknown_or_unavailable_backends_are_refused(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "linux")
    with pytest.raises(runtime.ProvisionError, match="no 'cuda' build"):
        runtime.install("cuda")
    with pytest.raises(runtime.ProvisionError, match="unknown runtime backend"):
        runtime.provision("opencl")


def _det(vendors, state="found"):
    return hwdetect.Detection(vendors=list(vendors), probe_ok=state != "unknown",
                              gpu_names=" ".join(vendors))


@pytest.mark.parametrize("plat,vendors,rocm,expected", [
    ("macos-arm64", [], False, "metal"),
    ("windows", [], False, "cpu"),
    ("windows", ["nvidia"], False, "cuda"),
    ("linux", ["nvidia"], False, "vulkan"),
    ("windows", ["amd"], True, "rocm"),
    ("windows", ["amd"], False, "vulkan"),
    ("linux", ["amd"], True, "rocm"),
    ("windows", ["intel"], False, "vulkan"),
])
def test_recommended_backend(monkeypatch, plat, vendors, rocm, expected):
    monkeypatch.setattr(runtime, "platform_key", lambda: plat)
    monkeypatch.setattr(runtime, "rocm_library_dirs", lambda: [Path("x")] if rocm else [])
    assert runtime.recommended_backend(_det(vendors)) == expected


def test_recommended_backend_with_an_unanswered_gpu_probe_is_vulkan(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    det = hwdetect.Detection(vendors=[], probe_ok=False, probe_error="no wmic")
    assert det.gpu_state == "unknown"
    assert runtime.recommended_backend(det) == "vulkan"


def test_rocm_library_dirs_needs_hipblas(tmp_path, monkeypatch):
    site_dir = tmp_path / "site"
    sub = "bin" if os.name == "nt" else "lib"
    with_lib = site_dir / "_rocm_sdk_libraries_gfx103X_all" / sub
    without = site_dir / "_rocm_sdk_core" / sub
    with_lib.mkdir(parents=True)
    without.mkdir(parents=True)
    (with_lib / ("hipblas.dll" if os.name == "nt" else "libhipblas.so.3")).write_bytes(b"x")
    (without / "amdhip64_7.dll").write_bytes(b"x")
    import site
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_dir)])
    found = runtime.rocm_library_dirs()
    assert with_lib in found
    assert without not in found


def test_setup_sdcpp_status_lists_every_backend(cli_runner, monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "vulkan")
    from localm.media.sdcpp.cli import main
    result = cli_runner.invoke(main, ["--status"])
    assert result.exit_code == 0, result.output
    for b in ("cpu", "vulkan", "cuda", "rocm"):
        assert f"{b}: not installed" in result.output
    assert pins.TAG in result.output


def test_setup_sdcpp_reports_a_failure_and_exits_1(cli_runner, monkeypatch):
    def boom(*a, **k):
        raise runtime.ProvisionError("no build for this platform")

    monkeypatch.setattr(runtime, "provision", boom)
    from localm.media.sdcpp.cli import main
    result = cli_runner.invoke(main, [])
    assert result.exit_code == 1
    assert "no build for this platform" in result.output


def test_setup_sdcpp_is_a_localm_command():
    from localm.cli import main
    assert "setup-sdcpp" in main.commands
