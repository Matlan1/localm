# SPDX-License-Identifier: AGPL-3.0-or-later
"""LOCALM_HOME pointing at a regular FILE (not a directory) is user
misconfiguration, not a localm bug.

``ensure_dirs()`` raises a ``click.ClickException`` - the pass-through the CLI's
graceful handler never routes to the bug reporter - rather than letting
``HOME_DIR.mkdir(exist_ok=True)`` surface a raw ``FileExistsError`` (WinError
183 on Windows) as "Sorry - localm hit an unexpected error" plus a bug-report
prompt.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

import localm.config as cfg
from localm import bugreport


def _point_home_at_file(tmp_path, monkeypatch):
    """Make config's HOME_DIR a regular file and return that path. Mirrors a user
    who set LOCALM_HOME to a file: the env var and the frozen module globals both
    point at it (config.py freezes HOME_DIR at import, so the env alone is not
    enough - see conftest.cli_runner)."""
    home_file = tmp_path / "localm_home_is_a_file"
    home_file.write_text("i am a regular file, not a directory", encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(home_file))
    monkeypatch.setattr(cfg, "HOME_DIR", home_file)
    monkeypatch.setattr(cfg, "MODELS_DIR", home_file / "models")
    return home_file


def test_ensure_dirs_rejects_home_that_is_a_file(tmp_path, monkeypatch):
    home_file = _point_home_at_file(tmp_path, monkeypatch)

    with pytest.raises(click.ClickException) as excinfo:
        cfg.ensure_dirs()

    msg = str(excinfo.value)
    assert "LOCALM_HOME" in msg
    assert "not a directory" in msg
    assert str(home_file) in msg
    # It must NOT be the reportable bug type (that routes to the bug reporter),
    # and not a raw OSError/FileExistsError (that hits the unexpected-error path).
    assert not isinstance(excinfo.value, bugreport.LocalmError)
    assert not isinstance(excinfo.value, OSError)


def test_ensure_dirs_still_creates_dirs_when_home_is_a_directory(tmp_path, monkeypatch):
    # The normal case: a home directory that does not yet exist is created,
    # along with its models subdir.
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")

    cfg.ensure_dirs()

    assert home.is_dir()
    assert (home / "models").is_dir()


def test_ensure_dirs_rejects_models_path_that_is_a_file(tmp_path, monkeypatch):
    # The rarer sibling: HOME_DIR is a valid directory but a regular file named
    # "models" sits inside it. Still a clean error, not an unexpected crash.
    home = tmp_path / ".localm"
    home.mkdir()
    models = home / "models"
    models.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", models)

    with pytest.raises(click.ClickException) as excinfo:
        cfg.ensure_dirs()

    msg = str(excinfo.value)
    assert "not a directory" in msg
    assert str(models) in msg


def test_cli_info_gives_clean_message_not_bug_report(tmp_path, monkeypatch):
    # End-to-end through the REAL `localm info` command and the graceful
    # handler: the misconfiguration surfaces as a clean error (exit 1, no report
    # written), never the "Sorry - unexpected error" + bug-report path.
    from localm.cli import main

    home_file = _point_home_at_file(tmp_path, monkeypatch)

    res = CliRunner().invoke(main, ["info"])

    assert res.exit_code == 1
    assert "LOCALM_HOME" in res.output
    assert "not a directory" in res.output
    assert str(home_file) in res.output
    # None of the unexpected-error / bug-report machinery fired.
    assert "Sorry -" not in res.output
    assert "unexpected error" not in res.output
    assert "bug report" not in res.output.lower()
