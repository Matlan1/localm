"""The pid-space id shared by the pull lock and the collection lock."""

from __future__ import annotations

import sys

from localm import instances
from localm.model_manager import pull
from localm.rag import collection_lock


def _fresh(monkeypatch, **patches) -> str:
    monkeypatch.setattr(instances, "_PID_SPACE", None)
    for name, value in patches.items():
        monkeypatch.setattr(instances, name, value)
    return instances.pid_space_id()


def test_both_locks_use_the_one_pid_space_id():
    assert pull._pid_space_id() == instances.pid_space_id()
    assert collection_lock._machine_id() == instances.pid_space_id()


def test_the_id_is_cached_and_stable():
    assert instances.pid_space_id() == instances.pid_space_id()


def test_a_windows_machine_guid_separates_two_machines(monkeypatch):
    first = _fresh(monkeypatch, machine_guid=lambda: "guid-a",
                   linux_machine_id=lambda: "")
    second = _fresh(monkeypatch, machine_guid=lambda: "guid-b",
                    linux_machine_id=lambda: "")
    assert first != second


def test_a_linux_machine_id_separates_two_hosts_with_one_name(monkeypatch):
    first = _fresh(monkeypatch, machine_guid=lambda: "",
                   linux_machine_id=lambda: "machine-id-a")
    second = _fresh(monkeypatch, machine_guid=lambda: "",
                    linux_machine_id=lambda: "machine-id-b")
    assert first != second


def test_the_linux_machine_id_is_read_from_etc_then_dbus(monkeypatch, tmp_path):
    etc, dbus = tmp_path / "etc-machine-id", tmp_path / "dbus-machine-id"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(instances, "_LINUX_MACHINE_ID_FILES", (etc, dbus))
    assert instances.linux_machine_id() == ""
    dbus.write_text("from-dbus\n", encoding="ascii")
    assert instances.linux_machine_id() == "from-dbus"
    etc.write_text("from-etc\n", encoding="ascii")
    assert instances.linux_machine_id() == "from-etc"
    etc.write_text("\n", encoding="ascii")
    assert instances.linux_machine_id() == "from-dbus"


def test_the_linux_machine_id_is_empty_off_linux(monkeypatch, tmp_path):
    f = tmp_path / "machine-id"
    f.write_text("present", encoding="ascii")
    monkeypatch.setattr(instances, "_LINUX_MACHINE_ID_FILES", (f,))
    monkeypatch.setattr(sys, "platform", "win32")
    assert instances.linux_machine_id() == ""


def test_the_machine_guid_is_empty_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert instances.machine_guid() == ""
