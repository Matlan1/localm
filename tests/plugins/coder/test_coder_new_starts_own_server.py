# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --new` starts a dedicated server while another one serves the
project, and connects to the server it started, not to the one that was
already running.

The real backend builder runs; the spawn, the clock and attach discovery are
stubbed. One test drives the real registry lookup with a stubbed probe."""

import click

import localm.instances as instances
import localm.plugins.coder.cli as ccli
from localm.plugins.coder.cli import _main

OLD = {"base_url": "http://127.0.0.1:1111/v1", "token": None, "port": 1111,
       "mode": "api"}
NEW = {"base_url": "http://127.0.0.1:2222/v1", "token": None, "port": 2222,
       "mode": "api"}


class _LiveProc:
    returncode = None

    def poll(self):
        return None


def _stub_world(monkeypatch, tmp_path):
    monkeypatch.setattr(instances, "resolve_root_dir", lambda **kw: "/proj")
    monkeypatch.setattr(instances, "list_entries",
                        lambda home: [{"instance_id": "old", "root_dir": "/proj"}])
    monkeypatch.setattr("time.sleep", lambda s: None)
    lookups = []

    def attach_target(home, root, probe=None, *, exclude_ids=(), launch_tag=None):
        lookups.append(frozenset(exclude_ids))
        if "old" not in exclude_ids:
            return OLD
        return NEW if len(lookups) >= 3 else None

    monkeypatch.setattr(instances, "attach_target", attach_target)
    spawned = []
    monkeypatch.setattr("subprocess.Popen",
                        lambda cmd, **kw: spawned.append(cmd) or _LiveProc())
    built = []
    monkeypatch.setattr(_main, "HTTPBackend",
                        lambda base_url, *a, **kw: built.append(base_url) or object())
    return lookups, spawned, built


def test_coder_new_passes_new_and_connects_to_the_server_it_started(
        monkeypatch, tmp_path):
    lookups, spawned, built = _stub_world(monkeypatch, tmp_path)
    with click.Context(ccli.main):
        _main._build_backend(None, None, "m", "k", False, None, False, True,
                             str(tmp_path), False)
    assert len(spawned) == 1
    assert "--new" in spawned[0]
    assert all("old" in ids for ids in lookups), lookups
    assert built == [NEW["base_url"]]


def test_coder_without_new_attaches_to_the_running_server_and_spawns_nothing(
        monkeypatch, tmp_path):
    lookups, spawned, built = _stub_world(monkeypatch, tmp_path)
    with click.Context(ccli.main):
        _main._build_backend(None, None, "m", "k", False, None, False, False,
                             str(tmp_path), False)
    assert spawned == []
    assert built == [OLD["base_url"]]


def test_find_attachable_skips_excluded_instances_without_reaping_them(
        monkeypatch, tmp_path):
    entries = [
        {"instance_id": "old", "root_dir": "/proj", "pid": 1, "_path": "old.json"},
        {"instance_id": "new", "root_dir": "/proj", "pid": 2, "_path": "new.json"},
    ]
    monkeypatch.setattr(instances, "list_entries", lambda home: entries)
    monkeypatch.setattr(instances, "_canon", lambda p: p)
    probed = []
    reaped = []
    monkeypatch.setattr(instances, "unregister_instance", reaped.append)

    def probe(entry):
        probed.append(entry["instance_id"])
        return True

    assert instances.find_attachable(tmp_path, "/proj", probe=probe)["instance_id"] == "old"
    probed.clear()
    got = instances.find_attachable(tmp_path, "/proj", probe=probe,
                                    exclude_ids={"old"})
    assert got["instance_id"] == "new"
    assert probed == ["new"]
    assert reaped == []
