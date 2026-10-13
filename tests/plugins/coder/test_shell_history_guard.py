# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``_no_real_shell_history`` fixture (tests/plugins/coder/conftest.py): a
privacy-mode coder CLI exit inside a test finds no shell history files, even
when APPDATA and HISTFILE point at real-looking ones."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner


def _history_files(tmp_path: Path, monkeypatch) -> list:
    line = "localm coder --mode privacy fix it\nunrelated command\n"
    appdata = tmp_path / "appdata"
    ps = appdata / "Microsoft" / "Windows" / "PowerShell" / "PSReadLine"
    ps.mkdir(parents=True)
    ps_hist = ps / "ConsoleHost_history.txt"
    ps_hist.write_text(line, encoding="utf-8")
    bash_hist = tmp_path / "bash_history"
    bash_hist.write_text(line, encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("HISTFILE", str(bash_hist))
    return [ps_hist, bash_hist]


def test_no_test_reaches_the_real_shell_history(tmp_path, monkeypatch):
    from localm.plugins.coder import privacy
    files = _history_files(tmp_path, monkeypatch)

    assert privacy.clear_shell_history_traces() == 0

    for f in files:
        assert "localm coder" in f.read_text(encoding="utf-8")


def test_a_privacy_mode_cli_exit_leaves_history_files_alone(tmp_path, monkeypatch):
    files = _history_files(tmp_path, monkeypatch)
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.delenv("LOCALCODER_URL", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    proj = tmp_path / "proj"
    proj.mkdir()
    from localm.plugins.coder.cli import _main
    calls = []
    real = _main.clear_shell_history_traces
    monkeypatch.setattr(_main, "clear_shell_history_traces",
                        lambda *a, **k: calls.append(1) or real(*a, **k))

    result = CliRunner().invoke(
        _main.main, ["--no-server", "--mode", "privacy", "--url", "http://127.0.0.1:1/v1",
                     "-m", "stub-model", "--cwd", str(proj)],
        catch_exceptions=True, input="\n")

    assert result.exception is None, result.output
    assert calls == [1], "the privacy-mode exit did not run the history scrub"
    for f in files:
        assert "localm coder" in f.read_text(encoding="utf-8")
