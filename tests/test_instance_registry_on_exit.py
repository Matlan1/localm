# SPDX-License-Identifier: AGPL-3.0-or-later
"""An instance's registry entry goes away when it stops or restarts, even though
both paths leave the process without running advertise()'s own exit (os._exit,
os.execv), and `localm stop` never acts on an entry a restart left behind."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localm import instances
from localm.inference import http_server


def _raise_exit(*_a, **_kw):
    raise SystemExit(0)


@pytest.fixture
def advertised(tmp_path):
    """An advertised instance in a throwaway home: (instance id, its file)."""
    app = SimpleNamespace(state=SimpleNamespace())
    ctx = instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                              mode="full", project=str(tmp_path))
    info = ctx.__enter__()
    yield info["instance_id"], Path(info["path"])
    ctx.__exit__(None, None, None)
    instances._OWN_PATHS.pop(info["instance_id"], None)


class TestOwnEntry:
    def test_unregister_own_removes_the_file_advertise_wrote(self, advertised):
        iid, path = advertised
        assert path.is_file()
        entry = instances.unregister_own(iid)
        assert not path.exists()
        assert entry["instance_id"] == iid and entry["token"]
        assert instances.unregister_own(iid) is None

    def test_restore_own_writes_it_back(self, advertised, tmp_path):
        iid, path = advertised
        entry = instances.unregister_own(iid)
        instances.restore_own(tmp_path, entry)
        back = instances.read_entry(path)
        assert back is not None
        for key in ("instance_id", "port", "host", "root_dir", "mode", "token",
                    "scheme", "started"):
            assert back[key] == entry[key], key
        assert instances.unregister_own(iid) is not None

    def test_another_instances_file_is_never_removed(self, tmp_path):
        path = instances.register_instance(
            tmp_path, instance_id="someone-else", port=2, host="127.0.0.1",
            root_dir=str(tmp_path), mode="api", token="t")
        assert instances.unregister_own("someone-else") is None
        assert instances.unregister_own(None) is None
        assert path.is_file()

    def test_advertise_exit_still_removes_it_and_forgets_it(self, tmp_path):
        app = SimpleNamespace(state=SimpleNamespace())
        with instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                                 mode="api", project=str(tmp_path)) as info:
            assert info["instance_id"] in instances._OWN_PATHS
        assert not Path(info["path"]).exists()
        assert info["instance_id"] not in instances._OWN_PATHS

    def test_an_isolated_instance_records_nothing(self, tmp_path):
        app = SimpleNamespace(state=SimpleNamespace())
        with instances.advertise(app, tmp_path, host="127.0.0.1", port=1,
                                 mode="api", isolated=True) as info:
            assert info["instance_id"] not in instances._OWN_PATHS
            assert instances.unregister_own(info["instance_id"]) is None


class TestStopAndRestartRemoveTheEntry:
    def test_shutdown_removes_it(self, advertised, monkeypatch):
        iid, path = advertised
        monkeypatch.setattr(http_server, "_engine", None)
        monkeypatch.setattr(os, "_exit", _raise_exit)
        with pytest.raises(SystemExit):
            http_server._do_shutdown(instance_id=iid)
        assert not path.exists(), "a stopped instance left its registry entry"

    def test_restart_removes_it_before_the_re_exec(self, advertised, monkeypatch):
        iid, path = advertised
        seen = {}

        def _fake_execv(exe, argv):
            seen["listed"] = path.exists()
            raise SystemExit(0)

        monkeypatch.setattr(http_server, "_engine", None)
        monkeypatch.setattr(os, "execv", _fake_execv)
        with pytest.raises(SystemExit):
            http_server._do_restart(instance_id=iid)
        assert seen == {"listed": False}, (
            "the re-exec'd process starts while the old entry is still listed")

    def test_a_failed_re_exec_writes_the_entry_back(self, advertised, monkeypatch):
        iid, path = advertised
        monkeypatch.setattr(http_server, "_engine", None)
        monkeypatch.setattr(os, "execv", lambda exe, argv: (_ for _ in ()).throw(
            OSError("exec format error")))
        with pytest.raises(OSError):
            http_server._do_restart(instance_id=iid)
        entry = instances.read_entry(path)
        assert entry is not None and entry["instance_id"] == iid, (
            "a server still running after a failed restart is no longer listed")

    def test_the_forced_hang_restart_removes_it(self, advertised, monkeypatch):
        iid, path = advertised
        seen = {}

        def _graceful_fails(**kw):
            raise RuntimeError("graceful restart failed")

        def _fake_execv(exe, argv):
            seen["listed"] = path.exists()
            raise SystemExit(0)

        monkeypatch.setattr(http_server, "_do_restart", _graceful_fails)
        monkeypatch.setattr(http_server, "_mark_fds_noninheritable", lambda: None)
        monkeypatch.setattr(http_server, "_restart_argv",
                            lambda port=None: ["python", "-m", "localm"])
        monkeypatch.setattr(os, "execv", _fake_execv)
        app = SimpleNamespace(state=SimpleNamespace(instance_port=None,
                                                    instance_id=iid))
        with pytest.raises(SystemExit):
            http_server._hang_restart_action(app)
        assert seen == {"listed": False}


class TestStopIgnoresAnEntryARestartLeftBehind:
    @staticmethod
    def _entries(tmp_path, *, new_pid=4242):
        old = {"instance_id": "old-instance-id", "pid": 4242, "port": 8642,
               "host": "127.0.0.1", "scheme": "http"}
        new = {"instance_id": "new-instance-id", "pid": new_pid, "port": 8642,
               "host": "127.0.0.1", "scheme": "http"}
        return old, new

    @staticmethod
    def _answering(monkeypatch, instance_id):
        monkeypatch.setattr(
            instances, "fetch_any_whoami",
            lambda *a, **kw: {"app": "localm", "instance_id": instance_id})

    def test_same_pid_answering_as_another_instance_is_a_leftover(
            self, tmp_path, monkeypatch):
        from localm.cli.models import _restarted_as
        old, new = self._entries(tmp_path)
        self._answering(monkeypatch, new["instance_id"])
        assert _restarted_as(instances, old, [old, new]) == "new-instance-id"

    def test_the_live_entry_itself_is_not_a_leftover(self, tmp_path, monkeypatch):
        from localm.cli.models import _restarted_as
        old, new = self._entries(tmp_path)
        self._answering(monkeypatch, new["instance_id"])
        assert _restarted_as(instances, new, [old, new]) is None

    def test_another_pid_on_the_port_is_not_a_leftover(self, tmp_path, monkeypatch):
        from localm.cli.models import _restarted_as
        old, new = self._entries(tmp_path, new_pid=9999)
        self._answering(monkeypatch, new["instance_id"])
        assert _restarted_as(instances, old, [old, new]) is None

    def test_nothing_answering_is_not_a_leftover(self, tmp_path, monkeypatch):
        from localm.cli.models import _restarted_as
        old, new = self._entries(tmp_path)
        monkeypatch.setattr(instances, "fetch_any_whoami", lambda *a, **kw: None)
        assert _restarted_as(instances, old, [old, new]) is None

    def test_stop_old_id_neither_shuts_down_nor_kills(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from localm.cli import main
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        old_path = instances.register_instance(
            tmp_path, instance_id="old-instance-id", port=8642, host="127.0.0.1",
            root_dir=str(tmp_path), mode="full", token="t1")
        new_path = instances.register_instance(
            tmp_path, instance_id="new-instance-id", port=8642, host="127.0.0.1",
            root_dir=str(tmp_path), mode="full", token="t2")
        assert instances.read_entry(old_path)["pid"] == os.getpid()
        self._answering(monkeypatch, "new-instance-id")
        acted = []
        monkeypatch.setattr(instances, "kill_pid",
                            lambda pid, **kw: acted.append(("kill", pid)) or True)
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda url, **kw: acted.append(("post", url)))

        result = CliRunner().invoke(main, ["stop", "old-inst"])

        assert result.exit_code == 0, result.output
        assert acted == [], f"stop acted on the live server: {acted}"
        assert not old_path.exists()
        assert new_path.is_file()
        assert "left over from a restart" in " ".join(result.output.split())


class TestWatchdogRetiresWhenItsMarkerIsCleared:
    @staticmethod
    def _module():
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "crash_recovery_watchdog_under_test",
            root / "scripts" / "crash_recovery_watchdog.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_a_cleared_marker_ends_the_watch_while_the_pid_runs(
            self, tmp_path, monkeypatch):
        wd = self._module()
        relaunched = []
        monkeypatch.setattr(wd, "relaunch",
                            lambda *a, **kw: relaunched.append(a) or None)
        monkeypatch.setattr(wd, "pid_alive", lambda pid: True)
        code = wd.run(pid=os.getpid(), host="127.0.0.1", port=1, scheme="http",
                      instance_id="restarted", crash_dir=tmp_path,
                      relaunch_argv=["x"], restart_history=[],
                      poll_interval=0.01, grace_s=0.1, request_timeout=0.1,
                      log_path=None)
        assert code == wd.EXIT_OK
        assert relaunched == []

    def test_a_present_marker_keeps_the_watch(self, tmp_path, monkeypatch):
        wd = self._module()
        wd.marker_path(tmp_path, "running").write_text("{}", encoding="utf-8")
        checks = []

        def alive(pid):
            checks.append(pid)
            return len(checks) < 5

        monkeypatch.setattr(wd, "pid_alive", alive)
        monkeypatch.setattr(wd, "poll_for_new_instance",
                            lambda *a, **kw: ("next", 1))
        code = wd.run(pid=12345, host="127.0.0.1", port=1, scheme="http",
                      instance_id="running", crash_dir=tmp_path,
                      relaunch_argv=["x"], restart_history=[],
                      poll_interval=0.01, grace_s=0.1, request_timeout=0.1,
                      log_path=None)
        assert len(checks) == 5, "the watch ended while the marker was present"
        assert code == wd.EXIT_OK
