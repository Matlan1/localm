# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record that an install's components are at their pins is a file in the
data dir in log and full mode, and is kept in memory only in privacy mode: a
start in privacy mode writes nothing to ``updates/`` for it and creates no
``updates/`` folder, and still checks only once per process."""

import json

import pytest

from localm import _version, config, updater_components as uc

_VERSION = "0.3.4"


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "install"
    root.mkdir()
    (root / "VERSION").write_text(_VERSION, encoding="utf-8")
    monkeypatch.setattr(_version, "read_version", lambda: _VERSION)
    monkeypatch.setattr(uc, "_checked_in_process", set())
    return root


@pytest.fixture
def plan_calls(monkeypatch):
    calls = []

    def fake_plan(staged_root, installed_root, **kw):
        calls.append((staged_root, installed_root))
        return [{"name": n, "label": uc.LABELS[n], "update": False, "note": None}
                for n in uc.ORDER]

    monkeypatch.setattr(uc, "plan", fake_plan)
    return calls


def _updates():
    return config.home_dir() / "updates"


def test_privacy_mode_catch_up_writes_no_check_record(install, plan_calls, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    assert not _updates().exists()
    assert uc.catch_up(root=install) is None
    assert len(plan_calls) == 1
    assert not (_updates() / uc.CHECKED_FILE).exists()
    assert not _updates().exists(), "the check created the updates folder"


def test_privacy_mode_catch_up_checks_once_per_process(install, plan_calls, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    assert uc.catch_up(root=install) is None
    assert uc.catch_up(root=install) is None
    assert len(plan_calls) == 1, "the second catch_up compared the components again"
    monkeypatch.setattr(uc, "_checked_in_process", set())
    assert uc.catch_up(root=install) is None
    assert len(plan_calls) == 2, "a new process found a record of the check on disk"


def test_privacy_mode_current_finish_writes_no_check_record(install, plan_calls,
                                                             monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    uc.write_pending([{"name": "uv", "update": True}], root=install, version=_VERSION)
    status = uc.finish_pending(root=install)
    assert status is not None and status["state"] == "current"
    assert uc.read_pending() is None
    assert not (_updates() / uc.CHECKED_FILE).exists()
    assert uc.catch_up(root=install) is None
    assert len(plan_calls) == 1, "catch_up after finish_pending compared the components again"


@pytest.mark.parametrize("mode", ["log", "full"])
def test_log_and_full_mode_catch_up_record_the_check_on_disk(install, plan_calls,
                                                             monkeypatch, mode):
    monkeypatch.setenv("LOCALM_MODE", mode)
    assert uc.catch_up(root=install) is None
    data = json.loads((_updates() / uc.CHECKED_FILE).read_text(encoding="utf-8"))
    assert data == {"root": str(install.resolve()), "version": _VERSION}
    assert uc._checked_in_process == set()
    assert uc.catch_up(root=install) is None
    assert len(plan_calls) == 1


def test_log_mode_current_finish_records_the_check_on_disk(install, plan_calls,
                                                           monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    uc.write_pending([{"name": "uv", "update": True}], root=install, version=_VERSION)
    assert uc.finish_pending(root=install)["state"] == "current"
    data = json.loads((_updates() / uc.CHECKED_FILE).read_text(encoding="utf-8"))
    assert data == {"root": str(install.resolve()), "version": _VERSION}


@pytest.mark.parametrize("argv, written", [
    (["serve", "--mode", "privacy", "--port", "not-a-port"], False),
    (["serve", "--mode=PRIVACY", "--port", "not-a-port"], False),
    (["serve", "--port", "not-a-port"], True),
])
def test_a_mode_flag_reaches_the_start_up_check(install, plan_calls, monkeypatch,
                                                argv, written):
    from click.testing import CliRunner

    from localm import updater, updater_runtime
    from localm.cli._core import main
    for name in ("LOCALM_MODE", "LOCALM_UPDATE_STEPS_DONE", "LOCALM_RESTART_IN_PROGRESS"):
        monkeypatch.delenv(name, raising=False)
    config.update_config(lambda c: c.update(mode="log"))
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    monkeypatch.setattr(updater_runtime, "read_pending", lambda: None)
    monkeypatch.setattr(updater_runtime, "catch_up", lambda *a, **k: None)
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == 2, result.output
    assert "not-a-port" in result.output
    assert len(plan_calls) == 1
    assert (_updates() / uc.CHECKED_FILE).exists() is written


def _group_with(option):
    import click

    from localm.cli._core import SUBCOMMAND_ARGS, _GracefulGroup, _export_subcommand_mode
    seen = {}

    @click.group(cls=_GracefulGroup)
    @click.pass_context
    def group(ctx):
        _export_subcommand_mode(ctx)
        import os
        seen["mode"] = os.environ.get("LOCALM_MODE")
        seen["args"] = ctx.meta.get(SUBCOMMAND_ARGS)

    @group.command()
    @option
    def sub(mode):
        pass
    return group, seen


def test_an_unrelated_mode_option_sets_no_session_mode(monkeypatch):
    import click
    from click.testing import CliRunner
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    group, seen = _group_with(click.option(
        "--mode", type=click.Choice(["log", "full", "tile"]), default=None))
    result = CliRunner().invoke(group, ["sub", "--mode", "log"])
    assert result.exit_code == 0, result.output
    assert seen["args"] == ["sub", "--mode", "log"]
    assert seen["mode"] is None


def test_a_failing_mode_read_never_stops_the_command(monkeypatch):
    import click
    from click.testing import CliRunner

    from localm.cli import _core
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    group, seen = _group_with(click.option(
        "--mode", type=click.Choice(["privacy", "log", "full"]), default=None))

    def broken(cmd):
        raise RuntimeError("boom")
    monkeypatch.setattr(_core, "_session_mode_param", broken)
    result = CliRunner().invoke(group, ["sub", "--mode", "privacy"])
    assert result.exit_code == 0, result.output
    assert seen["mode"] is None
