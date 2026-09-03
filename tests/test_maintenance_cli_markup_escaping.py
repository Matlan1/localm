# SPDX-License-Identifier: AGPL-3.0-or-later
"""``rich.console.Console.print()`` parses any ``[...]`` in a printed string
as markup, so an unescaped non-literal value interpolated into a markup
f-string is silently mangled on screen - never a crash, just wrong text shown
to the user. Rich renders these as:

    Console().print('report[draft]')      -> prints "report"        (dropped)
    Console().print('notes[bold red]')    -> prints "notes"          (consumed
                                              as a style directive)

``localm/cli/maintenance.py`` is the highest-value target in the wider sweep
across ``localm/cli/*.py``:
`localm issues` / `localm issues <n>` print an issue's ``title``/``html_url``/
``state`` fetched LIVE from the GitHub API through the bug-report proxy
(``issue_tracker.get_issue``/``list_issues``, whose own docstring says
"trimmed", never sanitized) - genuinely externally-controlled content, not
merely a local filesystem path. ``localm update`` and ``localm bug-report``
also interpolate proxy-sourced text (release notes, HTTP error bodies via
``LocalmError.reason`` - confirmed at ``localm/_proxy.py``'s ``request()``,
which puts the raw response body straight into ``reason``) and user-typed CLI
input.

Every test here drives the REAL `Click` command end-to-end via `CliRunner`
(the `cli_runner` fixture from conftest.py - a throwaway `LOCALM_HOME`), with
only the underlying network/filesystem-touching call monkeypatched to return
a realistic payload - never the `console.print` call itself. Covers both
failure shapes: BRACKET_DROP (Rich silently drops the bracketed span) and
BRACKET_STYLE (Rich consumes it as a bogus style directive).
"""

from __future__ import annotations

from pathlib import Path

import pytest

BRACKET_DROP = "report[draft] crashed"
BRACKET_STYLE = "notes[bold red] mangled"


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    """rich.console.Console() reads COLUMNS at construction time; without it,
    a non-tty width default (80) hard-wraps mid-word inside the long pytest
    basetemp paths some tests here assert on verbatim - same fix
    test_rag_cli_markup_escaping.py's own `env` fixture applies."""
    import rich.console
    monkeypatch.setattr(rich.console.Console, "is_dumb_terminal", False)
    monkeypatch.setenv("COLUMNS", "300")


# --------------------------------------------------------------------------- #
#  `localm issues <number>` - the single highest-value site: title/html_url/  #
#  state come straight from the GitHub API via the bug-report proxy.          #
# --------------------------------------------------------------------------- #

class TestIssuesDetailMarkupEscaping:
    def test_bracket_drop_title_and_number_survive_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(
            issue_tracker, "get_issue",
            lambda number, **_: {"number": 42, "state": "open",
                                 "title": BRACKET_DROP,
                                 "html_url": "https://github.com/o/r/issues/42"})

        r = cli_runner.invoke(issues_cmd, ["42"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, (
            f"an issue title fetched from GitHub must survive verbatim, not be "
            f"silently mangled by Rich markup parsing: {r.output!r}")

    def test_bracket_style_title_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(
            issue_tracker, "get_issue",
            lambda number, **_: {"number": 7, "state": "closed",
                                 "title": BRACKET_STYLE,
                                 "html_url": "https://github.com/o/r/issues/7"})

        r = cli_runner.invoke(issues_cmd, ["7"])
        assert r.exit_code == 0, r.output
        assert BRACKET_STYLE in r.output, (
            f"a title segment that happens to look like a style tag must be "
            f"shown as literal text, not consumed as Rich styling: {r.output!r}")

    def test_html_url_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        bracketed_url = "https://github.com/o/r/issues/9?ref=[legacy]"
        monkeypatch.setattr(
            issue_tracker, "get_issue",
            lambda number, **_: {"number": 9, "state": "open", "title": "x",
                                 "html_url": bracketed_url})

        r = cli_runner.invoke(issues_cmd, ["9"])
        assert r.exit_code == 0, r.output
        assert bracketed_url in r.output, (
            f"the issue URL must survive verbatim: {r.output!r}")

    def test_state_field_survives_verbatim(self, cli_runner, monkeypatch):
        """``state`` is documented as "open"/"closed" by GitHub, but the proxy
        JSON is not locally validated against that enum - defense in depth."""
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(
            issue_tracker, "get_issue",
            lambda number, **_: {"number": 1, "state": BRACKET_DROP,
                                 "title": "x", "html_url": ""})

        r = cli_runner.invoke(issues_cmd, ["1"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, r.output


class TestIssuesListMarkupEscaping:
    def test_bracket_drop_title_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(
            issue_tracker, "list_issues",
            lambda state, **_: [{"number": 3, "state": "open",
                                 "title": BRACKET_DROP, "html_url": ""}])

        r = cli_runner.invoke(issues_cmd, [])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, r.output

    def test_bracket_style_title_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(
            issue_tracker, "list_issues",
            lambda state, **_: [{"number": 4, "state": "closed",
                                 "title": BRACKET_STYLE, "html_url": ""}])

        r = cli_runner.invoke(issues_cmd, [])
        assert r.exit_code == 0, r.output
        assert BRACKET_STYLE in r.output, r.output


class TestIssuesLoadErrorMarkupEscaping:
    def test_summary_and_reason_survive_verbatim(self, cli_runner, monkeypatch):
        """Simulates the LocalmError _proxy.request() raises on a bad HTTP
        response, whose `reason` is built from the raw response BODY
        (`f"HTTP {status}: {text[:300]}"`) - genuinely attacker/server-
        controlled text, confirmed by reading localm/_proxy.py directly."""
        from localm.cli.maintenance import issues_cmd
        from localm import issue_tracker
        from localm.bugreport import LocalmError

        def _raise(state, **_):
            raise LocalmError(f"the proxy failed: {BRACKET_DROP}",
                              reason=f"HTTP 500: {BRACKET_STYLE}")
        monkeypatch.setattr(issue_tracker, "available", lambda: True)
        monkeypatch.setattr(issue_tracker, "list_issues", _raise)

        r = cli_runner.invoke(issues_cmd, [])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, r.output
        assert BRACKET_STYLE in r.output, r.output


# --------------------------------------------------------------------------- #
#  `localm update` - LocalmError.reason can carry a raw HTTP response body    #
#  (see localm/_proxy.py), release notes/version tags come from a GitHub      #
#  release, both externally sourced.                                         #
# --------------------------------------------------------------------------- #

class TestUpdateCheckMarkupEscaping:
    def test_check_error_summary_and_reason_survive_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import update_cmd
        from localm import updater
        from localm.bugreport import LocalmError

        def _raise(**_):
            raise LocalmError(BRACKET_DROP, reason=BRACKET_STYLE)
        monkeypatch.setattr(updater, "available", lambda: True)
        monkeypatch.setattr(updater, "check", _raise)

        r = cli_runner.invoke(update_cmd, ["--check"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, r.output
        assert BRACKET_STYLE in r.output, r.output

    def test_latest_cur_and_notes_survive_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import update_cmd
        from localm import updater
        monkeypatch.setattr(updater, "available", lambda: True)
        monkeypatch.setattr(updater, "check", lambda **_: {
            "current": "0.1.0[legacy]",
            "latest": BRACKET_STYLE,
            "newer": True,
            "comparable": True,
            "notes": f"see the {BRACKET_DROP} section",
            "asset": {"id": "abc"},
            "signature": None,
        })

        r = cli_runner.invoke(update_cmd, ["--check"])
        assert r.exit_code == 0, r.output
        assert "0.1.0[legacy]" in r.output, r.output
        assert BRACKET_STYLE in r.output, r.output
        assert BRACKET_DROP in r.output, r.output


class TestUpdateApplyMarkupEscaping:
    def test_applied_version_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import update_cmd
        from localm import updater
        monkeypatch.setattr(updater, "available", lambda: True)
        monkeypatch.setattr(updater, "check", lambda **_: {
            "current": "0.1.0", "latest": "0.2.0", "newer": True,
            "comparable": True, "notes": None, "asset": {"id": "abc"},
            "signature": None,
        })
        monkeypatch.setattr(
            updater, "apply",
            lambda asset_id, *, signature=None: {
                "version": BRACKET_DROP, "klass": "reboot"})

        r = cli_runner.invoke(update_cmd, ["--yes"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, (
            f"the applied version string must survive verbatim: {r.output!r}")


class TestUpdateRollbackMarkupEscaping:
    def test_rollback_error_survives_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import update_cmd
        from localm import updater
        from localm.bugreport import LocalmError

        def _raise(**_):
            raise LocalmError(BRACKET_STYLE, reason=BRACKET_DROP)
        monkeypatch.setattr(updater, "rollback_last", _raise)

        r = cli_runner.invoke(update_cmd, ["--rollback"])
        assert r.exit_code == 0, r.output
        assert BRACKET_STYLE in r.output, r.output
        assert BRACKET_DROP in r.output, r.output


# --------------------------------------------------------------------------- #
#  `localm bug-report` - the report title is derived from user-typed text     #
#  (report_title), and the saved path is echoed back for the user to open.    #
# --------------------------------------------------------------------------- #

class TestBugReportMarkupEscaping:
    def test_summary_title_survives_verbatim(self, cli_runner):
        """Real end-to-end: no mocking at all, the report is genuinely saved
        under the isolated LOCALM_HOME the cli_runner fixture provides."""
        from localm.cli.maintenance import bug_report_cmd

        r = cli_runner.invoke(bug_report_cmd, ["-w", BRACKET_STYLE])
        assert r.exit_code == 0, r.output
        assert BRACKET_STYLE in r.output, (
            f"the bug report title (derived from -w) must survive verbatim in "
            f"'Filing a bug report:': {r.output!r}")

    def test_saved_path_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        """save_user_report's return path cannot naturally contain brackets
        under an ordinary LOCALM_HOME, so the lower-level call is monkeypatched
        to return one - the CLI's own print/escaping logic (and the real,
        unmocked offer_to_send tail, same as the other bug-report CLI tests)
        still runs for real. A REAL file (not a nonexistent path) so the
        command's own `path.read_text(...)` call succeeds."""
        from localm.cli.maintenance import bug_report_cmd
        from localm import bugreport
        bracketed_dir = tmp_path / "bug-reports[legacy]"
        bracketed_dir.mkdir()
        real_path = bracketed_dir / "bug-report.md"
        real_path.write_text("# localm bug report\n", encoding="utf-8")
        monkeypatch.setattr(bugreport, "save_user_report", lambda *a, **k: real_path)

        r = cli_runner.invoke(bug_report_cmd, ["-w", "it crashed"])
        assert r.exit_code == 0, r.output
        assert str(real_path) in r.output, (
            f"the saved report path must survive verbatim: {r.output!r}")


# --------------------------------------------------------------------------- #
#  `localm setup-embeddings` - the model name is a free-text CLI argument     #
#  ("a known key, a registered model name, or a path to a GGUF" per its own   #
#  --help), and the embedding-switch impact report echoes it and per-         #
#  collection provenance back.                                                #
# --------------------------------------------------------------------------- #

def _stub_install(monkeypatch, tmp_path, name="new-model.gguf"):
    fake = tmp_path / name
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "localm.inference.embedder.resolve_embedding_model_path",
        lambda allow_download=True: str(fake))
    return fake


class TestSetupEmbeddingsMarkupEscaping:
    def test_installing_model_name_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        _stub_install(monkeypatch, tmp_path)

        r = cli_runner.invoke(setup_embeddings, ["--model", BRACKET_DROP])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, (
            f"the --model name must survive verbatim in 'Installing embedding "
            f"model:': {r.output!r}")

    def test_ready_path_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        from localm.cli.maintenance import setup_embeddings
        bracketed_dir = tmp_path / "models[legacy]"
        bracketed_dir.mkdir()
        fake = bracketed_dir / "bge.gguf"
        fake.write_bytes(b"x")
        monkeypatch.setattr(
            "localm.inference.embedder.resolve_embedding_model_path",
            lambda allow_download=True: str(fake))

        r = cli_runner.invoke(setup_embeddings, [])
        assert r.exit_code == 0, r.output
        assert "models[legacy]" in r.output, (
            f"the installed model path must survive verbatim in 'Embedding "
            f"model ready:': {r.output!r}")

    def test_provenance_note_model_name_survives_verbatim(
            self, cli_runner, monkeypatch, tmp_path):
        """collection_provenance_note() embeds *model* (the --model argument)
        directly into its own returned string - proving the escape() wrapped
        around the WHOLE call catches a bracket regardless of which nested
        function produced it."""
        from localm.cli.maintenance import setup_embeddings
        from localm.rag.store import Collection, rag_dir
        _stub_install(monkeypatch, tmp_path, name="target-model.gguf")

        c = Collection("docs", base=rag_dir()).create()
        c._chunks = [{"source": "doc0.txt", "pos": 0, "text": "alpha"}]
        c._vectors = [[0.1] * 8]
        c._meta["embedding_model"] = "old-model"
        c._save()

        r = cli_runner.invoke(setup_embeddings, ["--model", BRACKET_STYLE],
                              input="n\n")
        assert BRACKET_STYLE in r.output, (
            f"the --model name embedded in the impact note must survive "
            f"verbatim: {r.output!r}")

    def test_provenance_built_with_survives_verbatim(self, cli_runner, monkeypatch, tmp_path):
        """c['built_with'] is the PRIOR embedding_model recorded on the
        collection - a free-text value (not restricted to a safe charset the
        way a collection NAME is by check_collection_name), so it can
        legitimately contain brackets."""
        from localm.cli.maintenance import setup_embeddings
        from localm.rag.store import Collection, rag_dir
        _stub_install(monkeypatch, tmp_path, name="target-model.gguf")

        c = Collection("docs", base=rag_dir()).create()
        c._chunks = [{"source": "doc0.txt", "pos": 0, "text": "alpha"}]
        c._vectors = [[0.1] * 8]
        c._meta["embedding_model"] = BRACKET_DROP
        c._save()

        r = cli_runner.invoke(setup_embeddings, ["--model", "new-model"], input="n\n")
        assert BRACKET_DROP in r.output, (
            f"the prior 'built with' model name must survive verbatim: "
            f"{r.output!r}")


# --------------------------------------------------------------------------- #
#  `localm make-launcher` - notes carry raw exception text (see              #
#  applaunch.py: `notes=[f"could not build {dst.name}: {e}"]`), and the      #
#  launcher path is under whatever directory localm is installed into.       #
# --------------------------------------------------------------------------- #

class TestMakeLauncherMarkupEscaping:
    def test_notes_and_path_survive_verbatim(self, cli_runner, monkeypatch):
        from localm.cli.maintenance import make_launcher_cmd
        from localm import applaunch
        from localm.applaunch import LauncherResult

        fake_path = Path("C:/fake/App[legacy]/LocaLM.exe")
        fake_desktop = Path("/fake/App[bold red]/LocaLM.desktop")
        monkeypatch.setattr(
            applaunch, "make_launcher",
            lambda **_: LauncherResult(
                ok=True, path=fake_path, desktop_file=fake_desktop,
                notes=[f"copied dll {BRACKET_DROP}", f"stamped icon {BRACKET_STYLE}"]))

        r = cli_runner.invoke(make_launcher_cmd, [])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP in r.output, r.output
        assert BRACKET_STYLE in r.output, r.output
        assert str(fake_path) in r.output, r.output
        assert str(fake_desktop) in r.output, r.output
