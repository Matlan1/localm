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
import subprocess
import sys
import threading
import time
from pathlib import Path
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
    state = SimpleNamespace(calls=calls, release=release, delay=0.0)

    def fake_download(url, dest, on_progress, label, cancel_check=None):
        calls.append(label)
        time.sleep(state.delay)
        shutil.copyfile(release / label, dest)

    monkeypatch.setattr(runtime, "_download", fake_download)
    return state


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

    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "ensure_for_backend", boom)
    from localm.media.koboldcpp.cli import main
    result = cli_runner.invoke(main, ["--backend", "vulkan"])
    assert result.exit_code == 1
    assert "no build for this platform" in result.output


def test_setup_music_is_a_localm_command():
    from localm.cli import main
    assert "setup-music" in main.commands


def test_a_marker_for_another_asset_or_checksum_is_not_installed(fake_release):
    rt = runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    good = json.loads((rt.path / runtime.MARKER).read_text(encoding="utf-8"))
    for key, value in (("asset", "other.exe"), ("sha256", "0" * 64)):
        meta = dict(good, **{key: value})
        (rt.path / runtime.MARKER).write_text(json.dumps(meta), encoding="utf-8")
        assert runtime.installed("nocuda") is None
    (rt.path / runtime.MARKER).write_text(json.dumps(good), encoding="utf-8")
    assert runtime.installed("nocuda") is not None


def test_two_installs_at_once_download_once(fake_release):
    fake_release.delay = 1.0
    results, errors = [], []

    def go():
        try:
            results.append(runtime.install("nocuda", unpack=_fake_unpack(),
                                           version_of=_version))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert len(results) == 2 and all(r.path == runtime.runtime_dir("nocuda") for r in results)
    assert fake_release.calls == ["fake-nocuda.exe"]
    assert not list(runtime.runtimes_root().glob(".install-*"))


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_a_lock_left_by_a_dead_install_is_taken_over(fake_release):
    root = runtime.runtimes_root()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".install-nocuda.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(_dead_pid()), encoding="utf-8")
    (root / ".koboldcpp-leftover").mkdir()
    (root / ".koboldcpp-leftover" / "part.exe").write_bytes(b"x" * 100)
    rt = runtime.install("nocuda", unpack=_fake_unpack(), version_of=_version)
    assert rt.launcher.is_file()
    assert not lock.exists()
    assert not (root / ".koboldcpp-leftover").exists()


def test_a_live_install_elsewhere_is_waited_for_and_cancel_ends_the_wait(fake_release):
    root = runtime.runtimes_root()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".install-nocuda.lock"
    lock.mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (lock / "pid").write_text(str(holder.pid), encoding="utf-8")
        lines = []
        t0 = time.monotonic()
        with pytest.raises(runtime.InstallCancelled):
            runtime.install("nocuda", on_progress=lines.append,
                            cancel_check=lambda: time.monotonic() - t0 > 2,
                            unpack=_fake_unpack(), version_of=_version)
        assert any("Waiting for another install" in ln for ln in lines)
        assert fake_release.calls == []
        assert lock.exists()
    finally:
        holder.kill()
        holder.wait()


class _SlowResponse:
    headers = {"Content-Length": str(10 * 1024 * 1024)}

    def __init__(self):
        self.reads = 0

    def read(self, n):
        self.reads += 1
        time.sleep(0.05)
        return b"\0" * n if self.reads <= 40 else b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cancel_during_the_download_stops_it_and_installs_nothing(monkeypatch):
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    import localm.http_ssl as http_ssl
    import localm.model_manager.pull as pull
    resp = _SlowResponse()
    monkeypatch.setattr(pull, "_ssrf_resolve_final_url", lambda url: url)
    monkeypatch.setattr(http_ssl, "verified_urlopen", lambda req, timeout=None: resp)
    ran = []
    with pytest.raises(runtime.InstallCancelled):
        runtime.install("nocuda", cancel_check=lambda: resp.reads >= 3,
                        unpack=_fake_unpack(ran), version_of=_version)
    assert resp.reads == 3 and ran == []
    assert runtime.installed("nocuda") is None
    assert not list(runtime.runtimes_root().glob(".koboldcpp-*"))


def test_cancel_during_the_unpack_kills_it(tmp_path):
    if sys.platform == "win32":
        fake = tmp_path / "slow-unpack.cmd"
        fake.write_text("@ping -n 60 127.0.0.1 > nul\r\n", encoding="utf-8")
    else:
        fake = tmp_path / "slow-unpack"
        fake.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
    t0 = time.monotonic()
    with pytest.raises(runtime.InstallCancelled):
        runtime._unpack(fake, tmp_path / "staging", tmp_path,
                        cancel_check=lambda: time.monotonic() - t0 > 1.5)
    assert time.monotonic() - t0 < 20


def test_a_remembered_failure_expires(monkeypatch):
    runtime.record_backend("cuda", False, "exit code 3")
    assert runtime.backend_failed("cuda") == "exit code 3"
    real_time = time.time
    monkeypatch.setattr(runtime.time, "time", lambda: real_time() + runtime.FAILED_TTL + 60)
    assert runtime.backend_failed("cuda") is None


def test_setup_music_force_forgets_backend_results(cli_runner, monkeypatch):
    runtime.record_backend("cuda", False, "exit code 3")
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")

    def boom(*a, **k):
        raise runtime.ProvisionError("stop here")

    monkeypatch.setattr(runtime, "ensure_for_backend", boom)
    from localm.media.koboldcpp.cli import main
    result = cli_runner.invoke(main, ["--backend", "vulkan", "--force"])
    assert result.exit_code == 1
    assert runtime.backend_failed("cuda") is None


# --------------------------------------------------------------------------- #
#  Model files: each component must carry its own architecture                 #
# --------------------------------------------------------------------------- #

def _gguf_file(path, arch):
    from tests.test_gguf_architecture_roles import _gguf
    path.parent.mkdir(parents=True, exist_ok=True)
    return _gguf(path, arch)


def test_a_configured_model_of_the_wrong_kind_is_refused(tmp_path):
    from localm.media.koboldcpp import models
    good = {c: str(_gguf_file(tmp_path / f"{c}.gguf", a))
            for c, a in models.ARCHITECTURES.items()}
    ms = models.resolve_models(good, use_lm=True, pull_missing=False)
    assert ms.dit == good["dit"] and ms.lm == good["lm"]
    wrong = dict(good, text_encoder=str(_gguf_file(tmp_path / "qwen.gguf", "qwen3")))
    with pytest.raises(models.ModelError,
                       match="not an ACE-Step text encoder .*qwen3.*acestep-text-enc"):
        models.resolve_models(wrong, pull_missing=False)


def test_another_model_under_a_default_name_is_not_used(tmp_path, monkeypatch):
    from localm import model_manager as mm
    from localm.media.koboldcpp import models
    plain, distinct = models.default_names("text_encoder")
    clash = _gguf_file(tmp_path / f"{plain}.gguf", "qwen3")
    assert mm.add_local(str(clash)) is True
    assert models.resolve_path(plain) is not None
    assert models.find_default("text_encoder") is None
    assert models.default_pull("text_encoder") == (
        f"{models.DEFAULT_REPO}:{models.DEFAULT_FILES['text_encoder']}", distinct)
    pulled = []

    def fake_pull(spec, name, say, cancel):
        pulled.append((spec, name))
        target = _gguf_file(tmp_path / "pulled" / f"{name}.gguf", "acestep-text-enc")
        assert mm.add_local(str(target)) is True

    monkeypatch.setattr(models, "_pull", fake_pull)
    others = {c: str(_gguf_file(tmp_path / f"{c}.gguf", a))
              for c, a in models.ARCHITECTURES.items() if c != "text_encoder"}
    ms = models.resolve_models(others, use_lm=True, pull_missing=True)
    assert pulled == [(f"{models.DEFAULT_REPO}:{models.DEFAULT_FILES['text_encoder']}",
                       distinct)]
    assert Path(ms.text_encoder).name == f"{distinct}.gguf"


def test_a_free_default_name_is_pulled_under_its_usual_name():
    from localm.media.koboldcpp import models
    assert models.default_pull("dit") == (
        f"{models.DEFAULT_REPO}:{models.DEFAULT_FILES['dit']}", None)
