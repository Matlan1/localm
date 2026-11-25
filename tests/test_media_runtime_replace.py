# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reinstalling a native media runtime (stable-diffusion.cpp, KoboldCpp) never
loses the installed one: a file that cannot be moved aside leaves the folder whole,
and a new file that cannot be moved in puts the old ones back. Every install step
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
from localm.update_mechanism import disk

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


def _installed(install, dest: Path) -> dict:
    """Install the runtime and make each of its files differ from what is installed
    next; returns the :func:`_snapshot`."""
    install()
    for path in dest.iterdir():
        if path.suffix in (".txt", ".dll") or path.name == kruntime.launcher_name():
            path.write_bytes(path.read_bytes() + b"installed")
    return _snapshot(dest)


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture(autouse=True)
def short_wait(monkeypatch):
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.2)
    monkeypatch.setattr(runtime_files, "RETRY_S", 0.01)


def _refuse_rename(monkeypatch, refused, error=None):
    """Make ``os.rename(src, dst)`` fail where *refused* says so."""
    real = os.rename

    def rename(src, dst):
        if refused(disk.short_path(Path(src)), disk.short_path(Path(dst))):
            raise error or PermissionError(13, "The process cannot access the file", str(src))
        return real(src, dst)
    monkeypatch.setattr(os, "rename", rename)
    return lambda: monkeypatch.setattr(os, "rename", real)


def _inside(path: str, folder: Path) -> bool:
    return Path(path).is_relative_to(os.path.realpath(folder))


def test_a_file_that_cannot_be_moved_aside_leaves_the_folder_whole(media_runtime, monkeypatch):
    install, dest, provision_error = media_runtime
    before = _installed(install, dest)
    _refuse_rename(monkeypatch, lambda src, dst: src.endswith("z-last.txt") and _inside(src, dest))
    with pytest.raises(provision_error) as info:
        install(force=True)
    assert _snapshot(dest) == before
    assert "is in use and was left as it is" in str(info.value)


def test_a_new_file_that_cannot_be_moved_in_puts_the_old_ones_back(media_runtime, monkeypatch):
    install, dest, provision_error = media_runtime
    before = _installed(install, dest)
    _refuse_rename(monkeypatch,
                   lambda src, dst: _inside(dst, dest) and f"{os.sep}new{os.sep}" in src,
                   OSError(28, "No space left on device"))
    with pytest.raises(provision_error) as info:
        install(force=True)
    assert _snapshot(dest) == before
    assert "the installed ones were kept" in str(info.value)


def test_a_failed_test_of_a_new_build_keeps_the_installed_one(sdcpp):
    install, dest, provision_error = sdcpp
    install()
    before = _snapshot(dest)

    def broken(_rt):
        raise RuntimeError("no driver")
    with pytest.raises(sruntime.LoadTestError, match="no driver"):
        sruntime.install("cpu", force=True, probe=broken)
    assert _snapshot(dest) == before
    assert sruntime.load_test_failed("cpu") is None


def test_a_failed_test_of_a_first_build_is_remembered(sdcpp):
    _install, dest, _error = sdcpp

    def broken(_rt):
        raise RuntimeError("no driver")
    with pytest.raises(sruntime.LoadTestError):
        sruntime.install("cpu", probe=broken)
    assert sruntime.load_test_failed("cpu") == "no driver"
    assert [f.name for f in dest.iterdir()] == [sruntime.FAILED_MARKER]


def test_a_build_for_an_update_stays_aside(media_runtime, tmp_path):
    install, dest, _error = media_runtime
    install()
    before = _snapshot(dest)
    into = dest.parent / "aside"
    into.mkdir()
    if dest.parent.name == "sdcpp":
        sruntime.install("cpu", into=into, probe=lambda rt: {"devices": [["CPU", "cpu"]]})
        marker = sruntime.MARKER
    else:
        def unpack(binary, staging, tmp):
            staging.mkdir(parents=True)
            (staging / kruntime.launcher_name()).write_bytes(b"another")
        kruntime.install("nocuda", into=into, unpack=unpack, version_of=lambda _p: kpins.VERSION)
        marker = kruntime.MARKER
    assert _snapshot(dest) == before
    assert (into / dest.name / marker).is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to move a folder "
                    "with an open file; POSIX allows it")
def test_a_file_held_open_keeps_every_installed_file(media_runtime):
    install, dest, provision_error = media_runtime
    before = _installed(install, dest)
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
    assert [p.name for p in dest.parent.iterdir() if p.name.startswith(".")] == [".localm-update"]
    assert [p.name for p in (dest.parent / ".localm-update").iterdir()] == ["lock"]


def test_install_steps_and_the_failure_reach_the_debug_log(media_runtime, monkeypatch,
                                                         caplog):
    install, dest, provision_error = media_runtime
    shown = []
    with caplog.at_level(logging.INFO, logger="localm"):
        install(on_progress=shown.append)
    logged = [r.getMessage() for r in caplog.records if r.name == "localm"]
    assert shown and all(any(m in line for line in logged) for m in shown)
    caplog.clear()
    _refuse_rename(monkeypatch, lambda src, dst: True)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(provision_error):
        install(force=True)
    warnings = [r for r in caplog.records
                if r.name == "localm" and r.levelno == logging.WARNING]
    failures = [r for r in warnings if "was refused" not in r.getMessage()]
    assert len(failures) == 1 and "failed" in failures[0].getMessage()


def test_a_runtime_that_cannot_be_put_back_is_installed_again(media_runtime, monkeypatch):
    from localm.update_mechanism import disk
    install, dest, provision_error = media_runtime
    _installed(install, dest)
    allow = _refuse_rename(monkeypatch, lambda src, dst: _inside(dst, dest))
    with pytest.raises(provision_error):
        install(force=True)
    allow()
    kept = [p for p in (dest.parent / ".localm-update").rglob("old") if p.is_dir()]
    assert kept and all(disk.remove_tree(p) for p in kept)
    install()
    assert sorted(p.name for p in (dest.parent / ".localm-update").iterdir()) == ["lock"]
    reinstalled = _snapshot(dest)
    install(force=True)
    assert _snapshot(dest) == reinstalled


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


def test_files_that_cannot_be_put_back_are_put_back_by_the_next_install(
        media_runtime, monkeypatch):
    install, dest, provision_error = media_runtime
    before = _installed(install, dest)
    allow = _refuse_rename(monkeypatch, lambda src, dst: _inside(dst, dest))
    with pytest.raises(provision_error, match="The process cannot access the file"):
        install(force=True)
    with pytest.raises(provision_error, match="an interrupted install in .* could not be undone"):
        install()
    allow()
    install()
    assert _snapshot(dest) == before


@pytest.fixture
def refused_tree_moves(monkeypatch):
    """``os.replace`` of a fetched tree into place refused with the Windows
    sharing error a handle on a file inside causes, ``times`` times (None:
    every time). Off Windows, where that error is not retried, the storekit
    rule is set to the Windows one for this error."""
    from localm import storekit
    state = {"times": 0, "refused": 0}
    real_replace = os.replace

    def replace(src, dst):
        if Path(src).name == "tree" and (state["times"] is None
                                         or state["refused"] < state["times"]):
            state["refused"] += 1
            err = PermissionError(13, "Access is denied", str(src))
            err.winerror = 5
            raise err
        return real_replace(src, dst)
    monkeypatch.setattr(os, "replace", replace)
    if os.name != "nt":
        monkeypatch.setattr(storekit, "_is_transient_permission_error",
                            lambda e: getattr(e, "winerror", None) in (5, 32))
    return state


def test_a_tree_move_refused_twice_then_allowed_completes_the_fetch(
        monkeypatch, tmp_path, refused_tree_moves, caplog):
    from localm.media import source_archive as sa
    from tests._source_archives import FakeGitHub
    caplog.set_level(logging.WARNING, logger="localm")
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "R", {"a.py": "1\n"})
    refused_tree_moves["times"] = 2
    dest = tmp_path / "home" / "comfyui"
    assert sa.fetch_tree(repo, commit, dest) == ["a.py"]
    assert (dest / "a.py").read_text() == "1\n"
    assert refused_tree_moves["refused"] == 2
    retries = [r for r in caplog.records if "was refused" in r.getMessage()]
    assert [r.getMessage().split(": attempt ")[1].split(" ")[0] for r in retries] == ["1", "2"]
    assert all(r.levelno == logging.WARNING for r in retries)
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".localm-src-")]


def test_a_tree_move_refused_every_time_names_the_folder(monkeypatch, tmp_path,
                                                         refused_tree_moves):
    from localm import storekit
    from localm.media import source_archive as sa
    from tests._source_archives import FakeGitHub
    monkeypatch.setattr(storekit, "_transient_backoff", lambda attempt: None)
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "R", {"a.py": "1\n"})
    refused_tree_moves["times"] = None
    dest = tmp_path / "home" / "comfyui"
    with pytest.raises(sa.ArchiveError) as info:
        sa.fetch_tree(repo, commit, dest)
    assert f"could not be moved into {dest}" in str(info.value)
    assert "Access is denied" in str(info.value)
    assert refused_tree_moves["refused"] == storekit._REPLACE_RETRIES
    assert not dest.exists()
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".localm-src-")]


def test_a_runtime_whose_install_was_killed_is_whole_before_it_is_used(media_runtime):
    from localm.update_mechanism import job, mover, steps
    install, dest, _error = media_runtime
    module = sruntime if dest.parent.name == "sdcpp" else kruntime
    build = "cpu" if module is sruntime else "nocuda"
    before = _installed(install, dest)
    place = steps.Place(dest.parent, "ab" * 16)
    staged = place.new(mover.WHOLE) / dest.name
    shutil.copytree(dest, staged)
    (staged / "z-last.txt").write_text("new")
    with job.Lock(dest.parent) as lock:
        job.start(lock, place.id, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        steps.move_in(dest.parent, place.id, mover.WHOLE)
        assert module.installed(build) is None
    assert _snapshot(dest) != before
    assert module.installed(build) is not None
    assert _snapshot(dest) == before


def test_a_test_that_fails_once_is_made_again(sdcpp, koboldcpp, monkeypatch):
    answers = [RuntimeError("held by a scanner"), {"devices": [["CPU", "cpu"]]}]

    def probe(_rt):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
    assert sruntime.install("cpu", probe=probe).devices == [["CPU", "cpu"]]
    versions = ["", kpins.VERSION]

    def unpack(binary, staging, tmp):
        staging.mkdir(parents=True)
        (staging / kruntime.launcher_name()).write_bytes(b"launcher")
    assert kruntime.install("nocuda", unpack=unpack, version_of=lambda _p: versions.pop(0))
    with pytest.raises(kruntime.LoadTestError):
        kruntime.install("nocuda", force=True, unpack=unpack, version_of=lambda _p: "")
