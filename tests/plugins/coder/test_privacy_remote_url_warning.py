# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --url <server>` in privacy mode: a model server that is not on
this machine gets the same privacy-mode warning as --online / --anthropic,
naming the destination host (never the URL's credentials) and saying when the
URL came from LOCALCODER_URL. A loopback server, a non-privacy session and a
provider flag (which warns on its own) print no URL warning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from localm.plugins.coder.cli._main import main

_CONSOLE = "localm.plugins.coder.privacy.console"


def _activate_coder_plugin(monkeypatch, home: Path):
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALCODER_URL", raising=False)
    _activate_coder_plugin(monkeypatch, tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()

    def _run(*args, env=None):
        printed: list = []
        with patch(_CONSOLE) as console:
            console.print.side_effect = lambda *a, **k: printed.append(str(a[0]) if a else "")
            result = CliRunner().invoke(
                main, ["--no-server", "-m", "stub-model", "--cwd", str(proj), *args],
                catch_exceptions=True, input="\n", env=env)
        return result, "\n".join(printed)

    return _run


def test_an_off_machine_url_in_privacy_mode_warns_with_the_host(run_cli):
    result, printed = run_cli(
        "--mode", "privacy", "--url", "http://user:s3cret@192.0.2.10:9/v1?token=abc")

    assert result.exception is None, result.output
    assert "off-machine model server" in printed
    assert "192.0.2.10:9" in printed
    assert "s3cret" not in printed and "token=abc" not in printed
    assert "LOCALCODER_URL" not in printed


def test_an_off_machine_url_from_the_environment_says_where_it_came_from(run_cli):
    result, printed = run_cli(
        "--mode", "privacy", env={"LOCALCODER_URL": "http://192.0.2.10:9/v1"})

    assert result.exception is None, result.output
    assert "off-machine model server" in printed
    assert "from LOCALCODER_URL" in printed


def test_a_loopback_url_in_privacy_mode_does_not_warn(run_cli):
    result, printed = run_cli("--mode", "privacy", "--url", "http://127.0.0.1:1/v1")

    assert result.exception is None, result.output
    assert "off-machine" not in printed


def test_an_off_machine_url_outside_privacy_mode_does_not_warn(run_cli):
    result, printed = run_cli("--mode", "log", "--url", "http://192.0.2.10:9/v1")

    assert result.exception is None, result.output
    assert "off-machine" not in printed


def test_the_url_warning_names_only_host_and_port():
    from localm.plugins.coder.privacy import warn_external_url
    with patch(_CONSOLE) as console:
        warn_external_url("https://u:p@models.example.org/v1/x?k=v", from_env=False)
    text = console.print.call_args[0][0]
    assert "models.example.org" in text
    assert "u:p" not in text and "/v1/x" not in text and "k=v" not in text
