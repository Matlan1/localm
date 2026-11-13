# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm update --to ""` says a version is missing instead of running the
plain update check, and the coder stand-in names why the coder plugin could not
be loaded instead of pointing at a pip extra that adds nothing."""

from click.testing import CliRunner

from localm.cli import main
from localm.cli import maintenance


def test_update_to_an_empty_version_is_refused(monkeypatch):
    checked = []
    monkeypatch.setattr("localm.updater.check", lambda *a, **k: checked.append(1),
                        raising=False)

    res = CliRunner().invoke(main, ["update", "--to", ""])

    assert res.exit_code == 2, res.output
    assert "--to needs a version" in " ".join(res.output.split())
    assert checked == []


def test_the_coder_stand_in_names_the_import_failure(monkeypatch):
    monkeypatch.setattr("localm.plugins.engine.PluginManager.cli_entries",
                        lambda self: [("coder", "localm_no_such_module_for_test:main")])
    saved = main.commands.pop("coder", None)
    try:
        maintenance._wire_plugin_cli_entries()
        res = CliRunner().invoke(main, ["coder"])
    finally:
        main.commands.pop("coder", None)
        if saved is not None:
            main.commands["coder"] = saved

    flat = " ".join(res.output.split())
    assert res.exit_code == 1, res.output
    assert "localm_no_such_module_for_test" in flat
    assert "localm update --repair" in flat
    assert "localm[coder]" not in flat
