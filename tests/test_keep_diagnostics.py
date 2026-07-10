# SPDX-License-Identifier: AGPL-3.0-or-later
"""The keep_diagnostics privacy toggle: when on, the server also enables the debug
log at startup (even in privacy mode, without --debug) so a bug report has
request/operation context. When off (the default), privacy mode writes nothing."""

from pathlib import Path

from localm.config import load_config, save_config


def _run_gui(cli_runner, monkeypatch):
    from localm.plugins.gui import cli as guicli
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.portmux.run_server", lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr("localm.debuglog.enable_debug",
                        lambda: (calls.append(1), Path("dbg.log"))[1])
    result = cli_runner.invoke(guicli.main, ["--no-model", "--no-browser"])
    return result, calls


def test_keep_diagnostics_on_enables_debug_log(cli_runner, monkeypatch):
    cfg = load_config()
    cfg["keep_diagnostics"] = True          # privacy mode (default) + toggle on
    save_config(cfg)
    result, calls = _run_gui(cli_runner, monkeypatch)
    assert result.exit_code == 0, result.output
    assert calls, "keep_diagnostics on should enable the debug log at startup"


def test_keep_diagnostics_off_does_not_enable_debug_log(cli_runner, monkeypatch):
    # Default config: keep_diagnostics False, no --debug -> no debug log written.
    result, calls = _run_gui(cli_runner, monkeypatch)
    assert result.exit_code == 0, result.output
    assert not calls, "with the toggle off, privacy mode must not open a debug log"
