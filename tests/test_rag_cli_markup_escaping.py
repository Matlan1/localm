# SPDX-License-Identifier: AGPL-3.0-or-later
"""A document path, filename, or chunk excerpt shown by the `rag` CLI must
survive verbatim - Rich's ``Console.print()`` parses ``[...]`` in ANY
interpolated string as markup, not just inside a command's own literal
``[style]`` tags. Reproduced directly against this venv's rich:

    Console().print('report[draft].txt')       -> prints "report.txt"
    Console().print('notes[bold red].md')       -> prints "notes.md"

The bracketed span is either dropped outright or consumed as a (bogus)
style directive, in both cases silently. `rag docs`, `rag query`,
`rag resync`, and `rag add`'s failure report can all show a user a path
that differs from the real one on disk - which matters most exactly when
that path is meant to be copied back in as an argument (`rag rm-doc`).

Every case here indexes/removes/queries a REAL document under a name that
exercises the bug, the same convention test_rag_cli_docs_rmdoc.py uses,
rather than mocking the display layer. The one exception
(TestLockMessageEscaping) forces a real CollectionLockedError the same way
test_rag_cli_docs_rmdoc.py's own lock test does, since a lock message's
content (who holds it) is not something a test can produce by writing an
ordinary file to disk.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

# One name Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes described in the module docstring above.
BRACKET_DROP_NAME = "report[draft].txt"
BRACKET_STYLE_NAME = "notes[bold red].md"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Own copy rather than importing the one in
    test_rag_reg589_repair_noninteractive.py, which is module-private (same
    reasoning as test_rag_cli_docs_rmdoc.py's identical fixture)."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    # rich.console.Console() reads COLUMNS at construction time; without it, a
    # non-tty width default (80) hard-wraps mid-word inside the long pytest
    # basetemp paths these tests assert on - see test_rag_cli_docs_rmdoc.py.
    monkeypatch.setenv("COLUMNS", "300")
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def ragcli():
    from localm.cli import rag as ragcli
    return ragcli


@pytest.fixture
def runner():
    return CliRunner()


class TestRagDocsMarkupEscaping:
    def test_bracket_drop_filename_survives_verbatim(self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / BRACKET_DROP_NAME).write_text(
            "content about turbines and gearboxes", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        r = runner.invoke(ragcli.rag_group, ["docs", "kb"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP_NAME in r.output, (
            f"the bracketed filename must survive verbatim in 'rag docs', not "
            f"be silently mangled by Rich markup parsing: {r.output!r}")

    def test_bracket_style_filename_survives_verbatim(self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / BRACKET_STYLE_NAME).write_text(
            "content about compressors and bearings", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        r = runner.invoke(ragcli.rag_group, ["docs", "kb"])
        assert r.exit_code == 0, r.output
        assert BRACKET_STYLE_NAME in r.output, (
            f"a filename segment that happens to look like a style tag must be "
            f"shown as literal text, not consumed as Rich styling: {r.output!r}")


class TestRagRmDocMarkupEscaping:
    def test_removed_message_shows_bracketed_path_verbatim(
            self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / BRACKET_DROP_NAME).write_text("alpha content", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        from localm.rag import Collection
        target = next(iter(Collection("kb").documents()))
        assert BRACKET_DROP_NAME in target, "setup did not index the bracketed name"

        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", target])
        assert r.exit_code == 0, r.output
        assert target in r.output, (
            f"the 'Removed' message must echo the real indexed path verbatim, "
            f"so it can be copied back into another rm-doc call: {r.output!r}")

    def test_not_in_collection_message_shows_bracketed_path_verbatim(
            self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / "plain.txt").write_text("plain content", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        never_indexed = str(tmp_path / BRACKET_STYLE_NAME)
        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", never_indexed])
        assert r.exit_code == 1, r.output
        assert never_indexed in r.output, (
            f"the 'Not in this collection' error must echo the exact path the "
            f"caller passed, verbatim: {r.output!r}")


class TestRagQueryMarkupEscaping:
    def test_hit_source_survives_verbatim(self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / BRACKET_DROP_NAME).write_text(
            "turbine efficiency depends on blade pitch and gearbox ratio",
            encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        r = runner.invoke(ragcli.rag_group, ["query", "kb", "turbine gearbox"])
        assert r.exit_code == 0, r.output
        assert BRACKET_DROP_NAME in r.output, (
            f"a query hit's source path must survive verbatim: {r.output!r}")

    def test_excerpt_content_survives_verbatim(self, runner, ragcli, tmp_path):
        """The indexed CHUNK TEXT is shown too (not just the source path), and
        real document content routinely contains '[...]' on its own - a
        markdown link, a citation, a code snippet - so this is arguably the
        most common real-world trigger, not just the filename."""
        d = tmp_path / "docs"
        d.mkdir()
        (d / "notes.txt").write_text(
            "see the [draft] design doc for turbine specifications",
            encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        r = runner.invoke(ragcli.rag_group, ["query", "kb", "turbine specifications"])
        assert r.exit_code == 0, r.output
        assert "[draft] design doc" in r.output, (
            f"indexed chunk text containing '[...]' must be shown verbatim, "
            f"not parsed as markup: {r.output!r}")


class TestRagResyncMarkupEscaping:
    def test_missing_document_line_shows_bracketed_path_verbatim(
            self, runner, ragcli, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / BRACKET_STYLE_NAME).write_text("beta content", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        from localm.rag import Collection
        target = next(iter(Collection("kb").documents()))
        assert BRACKET_STYLE_NAME in target, "setup did not index the bracketed name"
        (d / BRACKET_STYLE_NAME).unlink()   # the file really vanishes

        r = runner.invoke(ragcli.rag_group, ["resync", "kb"])
        assert r.exit_code == 0, r.output
        assert target in r.output, (
            f"the 'missing:' line must show the real indexed path verbatim: "
            f"{r.output!r}")


class TestRagAddFailureMarkupEscaping:
    def test_failed_path_report_shows_bracketed_path_verbatim(
            self, runner, ragcli, tmp_path):
        """A file with an unindexable suffix (.bin: binary/media/model
        weights) is a REAL, naturally-occurring add_paths() failure -
        _report_add_paths_result (localm/cli/errors.py) prints its path and
        error message the same unescaped way rag.py's own sites did.

        Passed as an EXPLICIT file path, not a folder to recurse: a folder
        walk drops .bin via BLACKLISTED_SUFFIXES in Collection._expand()
        before add_paths ever sees it (0 added/updated/skipped/failed -
        confirmed empirically), while an explicitly-named file bypasses that
        filter (_expand's `if p.is_file(): out.append(...)` branch, "the
        local CLI ... still honours an explicit pick") and reaches the
        UNINDEXABLE_SUFFIXES check inside _add_paths_locked instead, which is
        what actually populates result["failed"]."""
        d = tmp_path / "docs"
        d.mkdir()
        bad_name = "firmware[legacy].bin"
        bad_file = d / bad_name
        bad_file.write_bytes(b"\x00\x01\x02binary")

        r = runner.invoke(ragcli.rag_group, ["add", "kb", str(bad_file)])
        assert r.exit_code == 1, r.output
        assert bad_name in r.output, (
            f"a failed-file report must show the real path verbatim: {r.output!r}")


class TestLockMessageEscaping:
    def test_collection_locked_message_survives_verbatim(
            self, runner, ragcli, tmp_path, monkeypatch):
        """_refuse_if_locked's `except CollectionLockedError as e` wraps *e*
        into the SAME unescaped f-string shape as every other site in this
        file. Forcing the real exception (as
        test_rag_cli_docs_rmdoc.py::test_a_locked_collection_is_reported_not_left_to_escape
        does for its own purpose) proves the fix at this exact call site
        without needing to fabricate a real cross-process lock holder whose
        recorded identity happens to contain brackets."""
        d = tmp_path / "docs"
        d.mkdir()
        (d / "plain.txt").write_text("plain content", encoding="utf-8")
        add = runner.invoke(ragcli.rag_group, ["add", "kb", str(d)])
        assert add.exit_code == 0, add.output

        from localm.rag import Collection, CollectionLockedError

        # A single bracket pair with a plain lowercase word inside, same shape
        # as this file's own BRACKET_DROP_NAME case - the one empirically
        # confirmed (see the module docstring's repro) to be silently DROPPED
        # by Rich rather than raising a MarkupError, so this exercises the
        # exact bug rather than an unrelated malformed-tag failure mode.
        locked_name = "kb[heldelsewhere]"

        def _locked(self, source):
            raise CollectionLockedError(locked_name, None, 1.0)
        monkeypatch.setattr(Collection, "remove_doc", _locked)

        target = next(iter(Collection("kb").documents()))
        r = runner.invoke(ragcli.rag_group, ["rm-doc", "kb", target])
        assert r.exit_code == 1, r.output
        assert locked_name in r.output, (
            f"a lock-refusal message containing '[...]' must survive verbatim, "
            f"not be parsed as markup: {r.output!r}")
