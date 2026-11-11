# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --new` attaches only to the server it started.

The coder passes a launch tag to the server it spawns (LOCALM_LAUNCH_TAG);
the server's real ``instances.advertise`` records it in its registry entry, and
the startup wait matches that tag exactly. These tests drive the real registry,
the real ``advertise`` and the real ``find_attachable``; the spawn, the clock
and the /whoami probe are stubbed."""

import contextlib
import json
import os
import subprocess
import types

import click
import pytest

import localm.instances as instances
import localm.plugins.coder.cli as ccli
from localm.config import home_dir
from localm.plugins.coder.cli import _main


class _LiveProc:
    returncode = None

    def poll(self):
        return None


def _app():
    return types.SimpleNamespace(state=types.SimpleNamespace())


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A project dir, a probe that accepts every entry, a recording spawn and
    a clock whose ticks run scripted registrations. Yields a namespace whose
    ``script`` maps a tick number to a callable taking the spawn env."""
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.setattr(instances, "default_probe", lambda entry, **kw: True)
    monkeypatch.delenv(instances.LAUNCH_TAG_ENV, raising=False)
    stack = contextlib.ExitStack()
    ns = types.SimpleNamespace(proj=proj, script={}, spawned=[], built=[],
                               ticks=0, stack=stack)

    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        if list(cmd[1:4]) != ["-m", "localm", "gui"]:
            return real_popen(cmd, **kw)
        ns.spawned.append((cmd, kw.get("env") or {}))
        return _LiveProc()

    def sleep(_s):
        ns.ticks += 1
        step = ns.script.get(ns.ticks)
        if step is not None:
            step(ns.spawned[-1][1])

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr(_main, "HTTPBackend",
                        lambda base_url, *a, **kw: ns.built.append(base_url) or object())
    with stack:
        yield ns


def _serve(ns, port, env=None):
    """Register a server for the project through the real advertise(), with
    *env*'s launch tag (if any) in the environment, as that server would."""
    tag = (env or {}).get(instances.LAUNCH_TAG_ENV)
    if tag is not None:
        os.environ[instances.LAUNCH_TAG_ENV] = tag
    try:
        ns.stack.enter_context(instances.advertise(
            _app(), home_dir(), host="127.0.0.1", port=port, mode="api",
            project=str(ns.proj)))
    finally:
        os.environ.pop(instances.LAUNCH_TAG_ENV, None)


def _run_coder_new(ns):
    with click.Context(ccli.main):
        _main._build_backend(None, None, "m", "k", False, None, False, True,
                             str(ns.proj), False)


def test_coder_new_picks_its_child_when_a_concurrent_serve_registers_first(world):
    world.script[1] = lambda env: _serve(world, 3333)
    world.script[2] = lambda env: _serve(world, 4444, env)
    _run_coder_new(world)
    assert len(world.spawned) == 1
    cmd, env = world.spawned[0]
    assert "--new" in cmd
    assert env.get(instances.LAUNCH_TAG_ENV)
    entries = {e["port"]: e for e in instances.list_entries(home_dir())}
    assert sorted(entries) == [3333, 4444]
    assert "launch_tag" not in entries[3333]
    assert entries[4444]["launch_tag"] == env[instances.LAUNCH_TAG_ENV]
    assert world.built == ["http://127.0.0.1:4444/v1"]


def test_coder_new_does_not_take_a_concurrent_serve_and_times_out(world, capsys):
    world.script[1] = lambda env: _serve(world, 3333)
    with pytest.raises(SystemExit) as exc:
        _run_coder_new(world)
    assert exc.value.code == 1
    assert world.built == []
    assert world.ticks == 40
    assert [e["port"] for e in instances.list_entries(home_dir())] == [3333]
    assert "Failed to attach to the auto-started server" in capsys.readouterr().out


def test_coder_without_new_sets_no_launch_tag(world, monkeypatch):
    monkeypatch.setenv(instances.LAUNCH_TAG_ENV, "ab" * 16)
    world.script[1] = lambda env: _serve(world, 3333)
    with click.Context(ccli.main):
        _main._build_backend(None, None, "m", "k", False, None, False, False,
                             str(world.proj), False)
    _cmd, env = world.spawned[0]
    assert instances.LAUNCH_TAG_ENV not in env
    assert world.built == ["http://127.0.0.1:3333/v1"]


def test_advertise_records_the_launch_tag_and_removes_it_from_the_environment(
        monkeypatch, tmp_path):
    tag = instances.new_launch_tag()
    monkeypatch.setenv(instances.LAUNCH_TAG_ENV, tag)
    with instances.advertise(_app(), tmp_path, host="127.0.0.1", port=5555,
                             mode="api", project=str(tmp_path)) as info:
        assert instances.LAUNCH_TAG_ENV not in os.environ
        entry = json.loads(open(info["path"], encoding="utf-8").read())
        assert entry["launch_tag"] == tag


@pytest.mark.parametrize("raw", ["", "short", "ZZ" * 16, "ab" * 40, "ab" * 16 + "\n"])
def test_advertise_drops_a_malformed_launch_tag(monkeypatch, tmp_path, raw):
    monkeypatch.setenv(instances.LAUNCH_TAG_ENV, raw)
    with instances.advertise(_app(), tmp_path, host="127.0.0.1", port=5555,
                             mode="api", project=str(tmp_path)) as info:
        assert instances.LAUNCH_TAG_ENV not in os.environ
        entry = json.loads(open(info["path"], encoding="utf-8").read())
        assert "launch_tag" not in entry


def test_entries_without_a_launch_tag_stay_attachable_without_one(tmp_path):
    root = str(tmp_path)
    instances.register_instance(tmp_path, instance_id="old", port=1, host="127.0.0.1",
                                root_dir=root, mode="api", token="t")
    instances.register_instance(tmp_path, instance_id="tagged", port=2,
                                host="127.0.0.1", root_dir=root, mode="api",
                                token="t", launch_tag="cd" * 16)
    old = json.loads((instances.run_dir(tmp_path) / "old.json").read_text("utf-8"))
    assert "launch_tag" not in old

    def probe(entry):
        return True

    seen = []
    for e in instances.list_entries(tmp_path):
        if instances.find_attachable(tmp_path, root, probe=probe,
                                     exclude_ids={x for x in ("old", "tagged")
                                                  if x != e["instance_id"]}):
            seen.append(e["instance_id"])
    assert seen == ["old", "tagged"]
    got = instances.find_attachable(tmp_path, root, probe=probe,
                                    launch_tag="cd" * 16)
    assert got["instance_id"] == "tagged"
    assert instances.find_attachable(tmp_path, root, probe=probe,
                                     launch_tag="ef" * 16) is None
