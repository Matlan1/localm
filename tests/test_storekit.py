# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.storekit: the atomic-write and per-namespace-lock mechanics shared by
rag/store.py and memory/store.py. Both stores must delegate to this ONE
implementation, never a copy, and the shared implementation carries the
PermissionError retry.
"""

import threading

import pytest

from localm.storekit import NamespaceLockRegistry, atomic_write


# --------------------------------------------------------------------------- #
#  atomic_write                                                                #
# --------------------------------------------------------------------------- #

def test_atomic_write_creates_file_with_content(tmp_path):
    target = tmp_path / "data.json"
    atomic_write(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'


def test_atomic_write_overwrites_existing_file(tmp_path):
    target = tmp_path / "data.json"
    atomic_write(target, "first")
    atomic_write(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    target = tmp_path / "data.json"
    atomic_write(target, "hello")
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_atomic_write_retries_on_permission_error(tmp_path, monkeypatch):
    """The Windows AV-lock workaround: a transient PermissionError on
    replace() must be retried, not raised immediately."""
    target = tmp_path / "data.json"
    calls = {"n": 0}
    real_replace = type(target).replace

    def flaky_replace(self, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated AV lock")
        return real_replace(self, dst)

    monkeypatch.setattr("pathlib.Path.replace", flaky_replace)
    monkeypatch.setattr("time.sleep", lambda s: None)  # do not actually sleep
    atomic_write(target, "content")
    assert target.read_text(encoding="utf-8") == "content"
    assert calls["n"] == 3


def test_atomic_write_gives_up_after_five_attempts(tmp_path, monkeypatch):
    def always_fails(self, dst):
        raise PermissionError("stuck AV lock")

    monkeypatch.setattr("pathlib.Path.replace", always_fails)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(PermissionError):
        atomic_write(tmp_path / "data.json", "content")


def test_atomic_write_temp_names_unique_per_thread(tmp_path):
    """Two concurrent writers to DIFFERENT files must not collide on the temp
    name even when called from different threads at the same instant."""
    results = {}

    def writer(n):
        p = tmp_path / f"f{n}.json"
        atomic_write(p, f"content-{n}")
        results[n] = p.read_text(encoding="utf-8")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(8):
        assert results[i] == f"content-{i}"


# --------------------------------------------------------------------------- #
#  NamespaceLockRegistry                                                       #
# --------------------------------------------------------------------------- #

def test_same_key_returns_same_lock_instance():
    reg = NamespaceLockRegistry()
    a = reg.get("collection-a")
    b = reg.get("collection-a")
    assert a is b


def test_different_keys_return_different_locks():
    reg = NamespaceLockRegistry()
    a = reg.get("collection-a")
    b = reg.get("collection-b")
    assert a is not b


def test_lock_is_reentrant():
    reg = NamespaceLockRegistry()
    lock = reg.get("ns")
    acquired = []
    with lock:
        with lock:  # RLock: must not deadlock on a second acquire same thread
            acquired.append(True)
    assert acquired == [True]


def test_registries_are_independent_between_instances():
    """rag/store.py and memory/store.py each own their OWN registry instance
    (keyed by collection name vs. namespace hash) - they must not share locks
    just because a key string happens to collide."""
    reg1 = NamespaceLockRegistry()
    reg2 = NamespaceLockRegistry()
    assert reg1.get("same-key") is not reg2.get("same-key")
