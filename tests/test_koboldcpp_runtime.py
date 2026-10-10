# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing and choosing the KoboldCpp music runtime: the binary is verified
before it is ever run, a failed check installs nothing, the network policy gates
the download, and backends are chosen per machine. The download is served from a
local file and the unpack/version steps are replaced where a test says so."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from types import SimpleNamespace

import pytest

from localm.media.koboldcpp import pins, runtime

BODY = b"MZ-not-really-koboldcpp" + os.urandom(4096)
SHA = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """Point the windows assets at local files and serve "downloads" by copy."""
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    release = tmp_path / "release"
    release.mkdir()
    for build in ("nocuda", "cuda"):
        name = f"fake-{build}.exe"
        (release / name).write_bytes(BODY)
        monkeypatch.setitem(pins.ASSETS, ("windows", build), (name, len(BODY), SHA))
    calls = []

    def fake_download(url, dest, on_progress, label):
        calls.append(label)
        shutil.copyfile(release / label, dest)

    monkeypatch.setattr(runtime, "_download", fake_download)
    return SimpleNamespace(calls=calls, release=release)


def _fake_unpack(record=None):
    def unpack(binary, staging, tmp):
        if record is not None:
            record.append(binary.read_bytes())
        staging.mkdir(parents=True)
        (staging / runtime.launcher_name()).write_bytes(b"launcher")
        (staging / "koboldcpp_vulkan.dll").write_bytes(b"dll")
    return unpack


def _version(_launcher):
    return pins.VERSION


def test_install_verifies_unpacks_and_marks(fake_release):
    ran = []
    rt = runtime.install("nocuda", unpack=_fake_unpack(ran), version_of=_version)
    assert rt.path == runtime.runtime_dir("nocuda")
    assert rt.launcher.is_file()
    assert ran == [BODY]
    meta = json.loads((rt.path / runtime.MARKER).read_text(encoding="utf-8"))
    assert meta == {"tag": pins.TAG, "version": pins.VERSION, "build": "nocuda",
                    "asset": "fake-nocuda.exe", "sha256": SHA}
    assert runtime.installed("nocuda") is not None
    leftovers = [p.name for p in runtime.runtimes_root().iterdir()]
    assert leftovers == [rt.path.name]


def test_an_installed_runtime_is_not_downloaded_again(fake_release):
    runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    assert fake_release.calls == ["fake-nocuda.exe"]


def test_force_reinstalls(fake_release):
    runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    runtime.install("nocuda", force=True, unpack=_fake_unpack(), version_of=_version)
    assert fake_release.calls == ["fake-nocuda.exe", "fake-nocuda.exe"]


@pytest.mark.parametrize("tamper,match", [
    (lambda p: p.write_bytes(BODY[:-1] + b"X"), "pinned sha256"),
    (lambda p: p.write_bytes(BODY + b"extra"), "expected"),
])
def test_a_binary_that_does_not_verify_is_never_run(fake_release, tamper, match):
    tamper(fake_release.release / "fake-nocuda.exe")
    ran = []
    with pytest.raises(runtime.DownloadError, match=match):
        runtime.install("nocuda", unpack=_fake_unpack(ran), version_of=_version)
    assert ran == []
    assert not runtime.runtime_dir("nocuda").exists()
    assert runtime.installed("nocuda") is None


def test_a_wrong_version_installs_nothing(fake_release):
    with pytest.raises(runtime.ProvisionError, match="reports version 1.0.0"):
        runtime.install("nocuda", unpack=_fake_unpack(), version_of=lambda p: "1.0.0")
    assert not runtime.runtime_dir("nocuda").exists()


def test_a_marker_for_another_tag_is_not_installed(fake_release):
    rt = runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    meta = json.loads((rt.path / runtime.MARKER).read_text(encoding="utf-8"))
    meta["tag"] = "v0.0.1"
    (rt.path / runtime.MARKER).write_text(json.dumps(meta), encoding="utf-8")
    assert runtime.installed("nocuda") is None


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
    ran = []
    with pytest.raises(runtime.DownloadError, match="network policy"):
        runtime.install("nocuda", unpack=_fake_unpack(ran), version_of=_version)
    assert calls == [] and ran == []
    assert not runtime.runtime_dir("nocuda").exists()


def test_the_real_unpack_refuses_a_binary_that_writes_no_launcher(tmp_path):
    if sys.platform != "win32":
        fake = tmp_path / "fake-binary"
        fake.write_text("#!/bin/sh\necho unpack failed\n", encoding="utf-8")
    else:
        fake = tmp_path / "fake-binary.cmd"
        fake.write_text("@echo unpack failed\r\n", encoding="utf-8")
    staging = tmp_path / "staging"
    with pytest.raises(runtime.ProvisionError, match="produced no"):
        runtime._unpack(fake, staging, tmp_path)


def test_unknown_builds_are_refused(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "macos-arm64")
    with pytest.raises(runtime.ProvisionError, match="no 'cuda' build"):
        runtime.install("cuda")
    monkeypatch.setattr(runtime, "platform_key", lambda: None)
    with pytest.raises(runtime.ProvisionError, match="available: none"):
        runtime.install("nocuda")


def _det(state="present", vendors=()):
    return SimpleNamespace(gpu_state=state, vendors=list(vendors))


@pytest.mark.parametrize("plat,det,expected", [
    ("windows", _det(vendors=["nvidia"]), "cuda"),
    ("linux", _det(vendors=["nvidia"]), "cuda"),
    ("windows", _det(vendors=["amd"]), "vulkan"),
    ("linux", _det(vendors=["intel"]), "vulkan"),
    ("windows", _det(state="none"), "cpu"),
    ("macos-arm64", _det(vendors=["apple"]), "metal"),
])
def test_recommended_backend(monkeypatch, plat, det, expected):
    monkeypatch.setattr(runtime, "platform_key", lambda: plat)
    assert runtime.recommended_backend(det) == expected


def test_fallback_order(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "cuda")
    assert runtime.fallback_order("auto") == ["cuda", "vulkan", "cpu"]
    assert runtime.fallback_order("cpu") == ["cpu"]
    monkeypatch.setattr(runtime, "platform_key", lambda: "macos-arm64")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "metal")
    assert runtime.fallback_order("auto") == ["metal", "cpu"]


def test_build_for_prefers_an_installed_build_that_can_run_the_backend(fake_release):
    assert runtime.build_for("vulkan") == "nocuda"
    assert runtime.build_for("cuda") == "cuda"
    runtime.install("cuda", unpack=_fake_unpack(), version_of=_version)
    assert runtime.build_for("vulkan") == "cuda"
    with pytest.raises(runtime.ProvisionError, match="no build that runs 'metal'"):
        runtime.build_for("metal")


def test_backend_results_are_remembered(monkeypatch):
    assert runtime.backend_failed("cuda") is None
    runtime.record_backend("cuda", False, "exit code 3")
    runtime.record_backend("vulkan", True)
    assert runtime.backend_failed("cuda") == "exit code 3"
    assert runtime.backend_worked("vulkan") is True
    assert runtime.backend_worked("cuda") is False
    runtime.record_backend("cuda", True)
    assert runtime.backend_failed("cuda") is None


def test_the_pins_cover_every_platform_with_a_cpu_capable_build():
    for plat in ("windows", "linux", "macos-arm64"):
        assert "cpu" in runtime.available_backends(plat), plat
    for (_plat, build), (name, size, sha) in pins.ASSETS.items():
        assert build in runtime.BUILD_BACKENDS
        assert size > 10 * 1024 ** 2 and len(sha) == 64 and int(sha, 16) >= 0
        assert pins.asset_url(name).startswith(
            f"https://github.com/{pins.REPO}/releases/download/{pins.TAG}/")


def test_setup_music_status(cli_runner, monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "recommended_backend", lambda det=None: "vulkan")
    from localm.media.koboldcpp.cli import main
    result = cli_runner.invoke(main, ["--status"])
    assert result.exit_code == 0, result.output
    assert pins.TAG in result.output
    assert "nocuda build (vulkan, cpu): not installed" in result.output
    assert "default dit model: not downloaded" in result.output


def test_setup_music_reports_a_failure_and_exits_1(cli_runner, monkeypatch):
    def boom(*a, **k):
        raise runtime.ProvisionError("no build for this platform")

    monkeypatch.setattr(runtime, "ensure_for_backend", boom)
    from localm.media.koboldcpp.cli import main
    result = cli_runner.invoke(main, ["--backend", "vulkan"])
    assert result.exit_code == 1
    assert "no build for this platform" in result.output


def test_setup_music_is_a_localm_command():
    from localm.cli import main
    assert "setup-music" in main.commands
