# SPDX-License-Identifier: AGPL-3.0-or-later
"""The keep_diagnostics privacy toggle: when on, the server also enables the debug
log at startup (even in privacy mode, without --debug) so a bug report has
request/operation context. When off (the default), privacy mode writes nothing."""

from pathlib import Path

from localm.config import load_config, save_config


def _run_gui(cli_runner, monkeypatch, extra_args=()):
    from localm.plugins.gui import cli as guicli
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.portmux.run_server", lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr("localm.debuglog.enable_debug",
                        lambda: (calls.append(1), Path("dbg.log"))[1])
    result = cli_runner.invoke(
        guicli.main, ["--no-model", "--no-browser", *extra_args])
    return result, calls


def test_keep_diagnostics_config_on_enables_debug_log(cli_runner, monkeypatch):
    cfg = load_config()
    cfg["keep_diagnostics"] = True          # WebUI toggle: privacy (default) + on
    save_config(cfg)
    result, calls = _run_gui(cli_runner, monkeypatch)
    assert result.exit_code == 0, result.output
    assert calls, "keep_diagnostics config on should enable the debug log at startup"


def test_keep_diagnostics_flag_enables_debug_log(cli_runner, monkeypatch):
    # The launcher passes --keep-diagnostics (a per-run override of the config).
    result, calls = _run_gui(cli_runner, monkeypatch, extra_args=["--keep-diagnostics"])
    assert result.exit_code == 0, result.output
    assert calls, "--keep-diagnostics flag should enable the debug log at startup"


def test_keep_diagnostics_off_does_not_enable_debug_log(cli_runner, monkeypatch):
    # Default config: keep_diagnostics False, no flag, no --debug -> no debug log.
    result, calls = _run_gui(cli_runner, monkeypatch)
    assert result.exit_code == 0, result.output
    assert not calls, "with the toggle off, privacy mode must not open a debug log"


def test_keep_diagnostics_enable_failure_warns_not_silent(cli_runner, monkeypatch):
    # The user opted into keep_diagnostics, but enable_debug() cannot open the log
    # (e.g. an unwritable or full LOCALM_HOME). Startup continues, and a visible
    # warning says the debug log will be missing from bug reports.
    from localm.plugins.gui import cli as guicli
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.portmux.run_server", lambda *a, **kw: None)

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr("localm.debuglog.enable_debug", _boom)
    result = cli_runner.invoke(
        guicli.main, ["--no-model", "--no-browser", "--keep-diagnostics"])
    assert result.exit_code == 0, result.output   # startup continues past the failure
    assert "could not enable" in result.output and "keep_diagnostics" in result.output, (
        "the failed keep_diagnostics debug log must be surfaced, not silently swallowed:\n"
        + result.output)
