# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download lock and partial owner records can carry the pid-space id from
before the Linux machine-id was part of it. One written on this host is judged
as this pid space, so its interrupted pull is reclaimed once its process is
gone; one that cannot be tied to this host still never is."""

import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

import localm.config as config
import localm.model_manager as model_manager
from localm import instances
from localm.model_manager import pull
from localm.model_manager.pull import PullInFlight, _part_lock, _part_lock_dir


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(model_manager, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    return h


def _exited_pid() -> int:
    for _ in range(20):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait(timeout=60)
        if not instances.pid_alive(p.pid):
            return p.pid
    pytest.fail("every exited child's pid was immediately reused")


def _this_host_start() -> dict | None:
    """A start identity that ties a record to this host and boot."""
    return instances.process_start_identity(os.getpid())


def _legacy_or_skip() -> str:
    legacy = instances.legacy_pid_space_id()
    if legacy is None or not instances.legacy_record_is_this_host(
            {"start": _this_host_start()}):
        pytest.skip("this host cannot tie a 0.2.x record to itself")
    return legacy


class TestTheLegacyId:
    def test_it_is_the_formula_the_earlier_version_used(self, monkeypatch):
        monkeypatch.setattr(instances, "sys", types.SimpleNamespace(platform="win32"))
        monkeypatch.setattr(instances, "machine_guid", lambda: "GUID")
        monkeypatch.setattr("platform.node", lambda: "host")
        monkeypatch.setattr(instances.os, "stat", _no_proc_stat(os.stat))
        expected = hashlib.sha256(
            "\x1f".join(["win32", "GUID", "host"]).encode()).hexdigest()[:16]
        assert instances.legacy_pid_space_id() == expected

    def test_it_differs_from_the_current_id(self):
        legacy = instances.legacy_pid_space_id()
        if legacy is None:
            pytest.skip("no machine identifier readable here")
        assert legacy != instances.pid_space_id()

    def test_it_is_none_when_nothing_identifies_the_host(self, monkeypatch):
        monkeypatch.setattr(instances, "sys", types.SimpleNamespace(platform="linux"))
        monkeypatch.setattr(instances, "machine_guid", lambda: "")
        monkeypatch.setattr("platform.node", lambda: "")
        monkeypatch.setattr(instances.os, "stat", _no_proc_stat(os.stat))
        assert instances.legacy_pid_space_id() is None


def _no_proc_stat(real):
    def stat(path, *a, **kw):
        if str(path).startswith("/proc/"):
            raise OSError("no /proc here")
        return real(path, *a, **kw)
    return stat


class TestTyingARecordToThisHost:
    @staticmethod
    def _linux(monkeypatch, tmp_path, boot="boot-a"):
        monkeypatch.setattr(instances, "sys", types.SimpleNamespace(platform="linux"))
        boot_file = tmp_path / "boot_id"
        boot_file.write_text(boot + "\n", encoding="ascii")
        monkeypatch.setattr(instances, "_LINUX_BOOT_ID", boot_file)

    def test_linux_this_boot_is_this_host(self, monkeypatch, tmp_path):
        self._linux(monkeypatch, tmp_path)
        assert instances.legacy_record_is_this_host(
            {"start": {"boot": "boot-a", "ticks": 5}}) is True

    @pytest.mark.parametrize("start", [
        {"boot": "boot-b", "ticks": 5}, {"boot": None, "ticks": 5}, None, "x"])
    def test_linux_another_or_unknown_boot_is_not(self, monkeypatch, tmp_path, start):
        self._linux(monkeypatch, tmp_path)
        assert instances.legacy_record_is_this_host({"start": start}) is False

    def test_linux_unreadable_boot_id_is_not(self, monkeypatch, tmp_path):
        self._linux(monkeypatch, tmp_path)
        monkeypatch.setattr(instances, "_LINUX_BOOT_ID", tmp_path / "missing")
        assert instances.legacy_record_is_this_host(
            {"start": {"boot": "boot-a", "ticks": 5}}) is False

    def test_windows_with_a_machine_guid_is(self, monkeypatch):
        monkeypatch.setattr(instances, "sys", types.SimpleNamespace(platform="win32"))
        monkeypatch.setattr(instances, "machine_guid", lambda: "GUID")
        assert instances.legacy_record_is_this_host({"start": None}) is True
        monkeypatch.setattr(instances, "machine_guid", lambda: "")
        assert instances.legacy_record_is_this_host({"start": None}) is False

    def test_another_platform_never_is(self, monkeypatch):
        monkeypatch.setattr(instances, "sys", types.SimpleNamespace(platform="darwin"))
        assert instances.legacy_record_is_this_host({"start": None}) is False
        assert instances.legacy_record_is_this_host("not a record") is False


def _write_lock(d, pid, *, space, start) -> str:
    d.mkdir(parents=True)
    text = json.dumps({"pid": pid, "filename": "m.gguf", "space": space,
                       "start": start})
    (d / "owner.json").write_text(text, encoding="utf-8")
    return text


class TestDownloadLock:
    def test_a_dead_holder_from_before_the_upgrade_is_reclaimed(self, home):
        legacy = _legacy_or_skip()
        d = _part_lock_dir("m.gguf")
        dead = _exited_pid()
        _write_lock(d, dead, space=legacy, start=_this_host_start())
        with _part_lock("m.gguf"):
            rec = json.loads((d / "owner.json").read_text(encoding="utf-8"))
        assert rec["pid"] == os.getpid(), "the 0.2.x lock was not reclaimed"

    def test_a_live_holder_from_before_the_upgrade_keeps_it(self, home):
        legacy = _legacy_or_skip()
        other = subprocess.Popen([sys.executable, "-c",
                                  "import sys; sys.stdin.read()"],
                                 stdin=subprocess.PIPE)
        try:
            start = instances.process_start_identity(other.pid)
            d = _part_lock_dir("m.gguf")
            before = _write_lock(d, other.pid, space=legacy, start=start)
            with pytest.raises(PullInFlight):
                with _part_lock("m.gguf"):
                    pass
            assert (d / "owner.json").read_text(encoding="utf-8") == before
        finally:
            other.stdin.close()
            other.wait(timeout=60)

    def test_a_legacy_record_not_tied_to_this_host_is_never_reclaimed(
            self, home, monkeypatch):
        legacy = _legacy_or_skip()
        monkeypatch.setattr(instances, "legacy_record_is_this_host",
                            lambda rec: False)
        d = _part_lock_dir("m.gguf")
        before = _write_lock(d, _exited_pid(), space=legacy,
                             start=_this_host_start())
        with pytest.raises(PullInFlight):
            with _part_lock("m.gguf"):
                pass
        assert (d / "owner.json").read_text(encoding="utf-8") == before


class TestPartialOwner:
    @staticmethod
    def _partial(tmp_path, pid, *, space, start):
        partial = tmp_path / "blob.123.incomplete"
        partial.write_bytes(b"partial")
        pull._partial_owner_path(partial).write_text(
            json.dumps({"pid": pid, "space": space, "start": start}),
            encoding="utf-8")
        return partial

    def test_a_dead_owner_from_before_the_upgrade_is_gone(self, tmp_path):
        legacy = _legacy_or_skip()
        partial = self._partial(tmp_path, _exited_pid(), space=legacy,
                                start=_this_host_start())
        assert pull._partial_owner_is_gone(partial) is True

    def test_one_not_tied_to_this_host_is_not(self, tmp_path, monkeypatch):
        legacy = _legacy_or_skip()
        monkeypatch.setattr(instances, "legacy_record_is_this_host",
                            lambda rec: False)
        partial = self._partial(tmp_path, _exited_pid(), space=legacy,
                                start=_this_host_start())
        assert pull._partial_owner_is_gone(partial) is False

    def test_another_pid_space_is_still_not(self, tmp_path):
        partial = self._partial(tmp_path, _exited_pid(),
                                space="0123456789abcdef", start=_this_host_start())
        assert pull._partial_owner_is_gone(partial) is False
