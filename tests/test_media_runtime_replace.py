# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reinstalling a native media runtime (stable-diffusion.cpp, KoboldCpp) never
loses the installed one: a folder that cannot be moved aside is left whole, and
a new folder that cannot be moved in puts the old one back. Every install step
and its failure reach the debug log, and a download refused because network
access is off names the Settings place and the terminal command.

The downloads are served from local files and the load test or unpack steps are
replaced; the folder moves are real. On Windows a file held open inside the
installed folder makes the real rename fail."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from localm.media import runtime_files
from localm.media.koboldcpp import pins as kpins
from localm.media.koboldcpp import runtime as kruntime
from localm.media.sdcpp import _binding, pins as spins
from localm.media.sdcpp import runtime as sruntime

BODY = b"MZ-not-really-koboldcpp" + os.urandom(4096)
SHA = hashlib.sha256(BODY).hexdigest()


def _zip(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(_binding.lib_filename(), os.urandom(300 * 1024))
        zf.writestr("a-first.txt", "license")
        zf.writestr("z-last.txt", "notes")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sdcpp(tmp_path, monkeypatch):
    monkeypatch.setattr(sruntime, "platform_key", lambda: "windows")
    name = "fake-cpu.zip"
    sha = _zip(tmp_path / "release" / name)
    monkeypatch.setitem(spins.ASSETS, ("windows", "cpu"), (name, sha))
    monkeypatch.setattr(sruntime, "_download",
                        lambda url, dest, on_progress, label:
                        shutil.copyfile(tmp_path / "release" / label, dest))

    def install(force=False, on_progress=None):
        return sruntime.install("cpu", force=force, on_progress=on_progress,
                                probe=lambda rt: {"devices": [["CPU", "cpu device"]]})
    return install, sruntime.runtime_dir("cpu"), sruntime.ProvisionError


@pytest.fixture
def koboldcpp(tmp_path, monkeypatch):
    monkeypatch.setattr(kruntime, "platform_key", lambda: "windows")
    release = tmp_path / "krelease"
    release.mkdir()
    (release / "fake-nocuda.exe").write_bytes(BODY)
    monkeypatch.setitem(kpins.ASSETS, ("windows", "nocuda"), ("fake-nocuda.exe", len(BODY), SHA))
    monkeypatch.setattr(kruntime, "_download",
                        lambda url, dest, on_progress, label, cancel_check=None:
                        shutil.copyfile(release / label, dest))

    def unpack(binary, staging, tmp):
        staging.mkdir(parents=True)
        (staging / "a-first.dll").write_bytes(b"dll")
        (staging / kruntime.launcher_name()).write_bytes(b"launcher")
        (staging / "z-last.txt").write_bytes(b"notes")

    def install(force=False, on_progress=None):
        return kruntime.install("nocuda", force=force, on_progress=on_progress,
                                unpack=unpack, version_of=lambda _p: kpins.VERSION)
    return install, kruntime.runtime_dir("nocuda"), kruntime.ProvisionError


@pytest.fixture(params=["sdcpp", "koboldcpp"])
def media_runtime(request):
    return request.getfixturevalue(request.param)


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_folder_that_cannot_be_moved_aside_is_left_whole(media_runtime, monkeypatch):
    install, dest, provision_error = media_runtime
    install()
    before = _snapshot(dest)
    real_replace = os.replace

    def replace(src, dst):
        if Path(src) == dest:
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_replace(src, dst)
    monkeypatch.setattr(runtime_files.os, "replace", replace)
    with pytest.raises(provision_error) as info:
        install(force=True)
    assert _snapshot(dest) == before
    assert "is in use and was left as it is" in str(info.value)


def test_a_new_folder_that_cannot_be_moved_in_puts_the_old_one_back(media_runtime,
                                                                   monkeypatch):
    install, dest, provision_error = media_runtime
    install()
    before = _snapshot(dest)
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst) == dest and Path(src).name == "staging":
            raise PermissionError(13, "Access is denied", str(dst))
        return real_replace(src, dst)
    monkeypatch.setattr(runtime_files.os, "replace", replace)
    with pytest.raises(provision_error) as info:
        install(force=True)
    assert _snapshot(dest) == before
    assert "the installed ones were kept" in str(info.value)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to move a folder "
                    "with an open file; POSIX allows it")
def test_a_file_held_open_keeps_every_installed_file(media_runtime):
    install, dest, provision_error = media_runtime
    install()
    before = _snapshot(dest)
    held = sorted(p for p in dest.iterdir() if p.is_file())[-1]
    with open(held, "rb"):
        with pytest.raises(provision_error):
            install(force=True)
        assert _snapshot(dest) == before


def test_a_forced_reinstall_replaces_the_folder_and_leaves_no_copy(media_runtime):
    install, dest, _err = media_runtime
    install()
    (dest / "stale.txt").write_text("from the old install", encoding="utf-8")
    install(force=True)
    assert not (dest / "stale.txt").exists()
    assert [p.name for p in dest.parent.iterdir() if p.name.startswith(".")] == []


def test_install_steps_and_the_failure_reach_the_debug_log(media_runtime, monkeypatch,
                                                         caplog):
    install, dest, provision_error = media_runtime
    shown = []
    with caplog.at_level(logging.INFO, logger="localm"):
        install(on_progress=shown.append)
    logged = [r.getMessage() for r in caplog.records if r.name == "localm"]
    assert shown and all(any(m in line for line in logged) for m in shown)
    caplog.clear()
    monkeypatch.setattr(runtime_files.os, "replace", lambda s, d: (_ for _ in ()).throw(
        PermissionError(13, "denied", str(s))))
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(provision_error):
        install(force=True)
    failures = [r for r in caplog.records
                if r.name == "localm" and r.levelno == logging.WARNING]
    assert len(failures) == 1 and "failed" in failures[0].getMessage()


def test_the_sdcpp_runtime_is_named_for_images_and_video(sdcpp):
    install, _dest, _err = sdcpp
    shown = []
    install(on_progress=shown.append)
    assert not any("native image runtime" in line for line in shown)
    assert any("native image and video runtime" in line for line in shown)


@pytest.mark.parametrize("which", ["sdcpp", "koboldcpp"])
def test_net_mode_off_names_the_settings_place_and_the_command(which, monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    mod = sruntime if which == "sdcpp" else kruntime
    monkeypatch.setattr(mod, "platform_key", lambda: "windows")
    import localm.http_ssl as http_ssl
    import localm.netpolicy as netpolicy

    def no_network(*a, **k):
        raise AssertionError("a connection was attempted")
    monkeypatch.setattr(netpolicy, "pinned_request", no_network)
    monkeypatch.setattr(http_ssl, "verified_urlopen", no_network)
    with pytest.raises(mod.DownloadError) as info:
        if which == "sdcpp":
            mod.install("cpu", probe=lambda rt: {"devices": [["CPU", "cpu"]]})
        else:
            mod.install("nocuda", unpack=lambda *a: None, version_of=lambda _p: kpins.VERSION)
    text = str(info.value)
    assert "Settings > Server & network" in text
    assert "localm config net_mode ask" in text
    assert text.count("localm config") == 1


def test_a_policy_refusal_that_is_not_net_mode_off_keeps_its_reason():
    from localm.netpolicy import NetworkPolicyError
    text = runtime_files.refused_download_message(
        NetworkPolicyError("example.org is on the deny list"))
    assert text == "the download was refused by the network policy: example.org is on the deny list"


def test_when_the_old_folder_cannot_be_put_back_its_files_stay_where_the_error_says(
        media_runtime, monkeypatch):
    install, dest, provision_error = media_runtime
    install()
    before = _snapshot(dest)
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst) == dest:
            raise PermissionError(13, "Access is denied", str(dst))
        return real_replace(src, dst)
    monkeypatch.setattr(runtime_files.os, "replace", replace)
    with pytest.raises(provision_error) as info:
        install(force=True)
    kept = [p for p in dest.parent.iterdir() if p.name.startswith(f".{dest.name}.previous-")]
    assert len(kept) == 1 and _snapshot(kept[0]) == before
    assert str(kept[0]) in str(info.value)
