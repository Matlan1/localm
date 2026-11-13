# SPDX-License-Identifier: AGPL-3.0-or-later
"""A data file the CLI cannot read, or a store lock another process holds, ends
a command with a plain error line and exit 1, not the unexpected-error bug
report path."""

import localm.config as cfg
from localm.cli import main
from localm.storekit import ConfigLockTimeout


def _no_report(monkeypatch):
    calls = []
    monkeypatch.setattr("localm.bugreport.report_failure",
                        lambda *a, **k: calls.append(k) or None)
    return calls


def test_an_unreadable_registry_is_a_plain_error(cli_runner, monkeypatch):
    reports = _no_report(monkeypatch)
    cfg.REGISTRY_FILE.write_text("{ not json", encoding="utf-8")

    res = cli_runner.invoke(main, ["list"])

    assert res.exit_code == 1, res.output
    assert "registry.json exists but could not be read" in " ".join(res.output.split())
    assert "unexpected error" not in res.output
    assert reports == []


def test_a_held_store_lock_is_a_plain_error(cli_runner, monkeypatch):
    reports = _no_report(monkeypatch)

    def _held(*a, **k):
        raise ConfigLockTimeout("timed out after 5s waiting for config.json.lock, "
                                "held by another localm process")
    monkeypatch.setattr("localm.cli.models.update_config", _held)

    res = cli_runner.invoke(main, ["config", "temperature", "0.5"])

    assert res.exit_code == 1, res.output
    flat = " ".join(res.output.split())
    assert "held by another localm process - nothing was written; try again." in flat
    assert reports == []
