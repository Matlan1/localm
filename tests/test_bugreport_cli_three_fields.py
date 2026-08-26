# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm bug-report`` shares the SAME three-field template and the same
report_title/offer_to_send machinery as the GUI/API producer
(save_user_report): "what I was doing", "what I expected" and "what happened",
plus the log-attach option."""

import glob
import os

import localm.config as cfg


def _saved_report_text(home):
    files = sorted(glob.glob(os.path.join(str(home), "bug-reports", "bug-*.md")))
    assert files, "no report file was written"
    return open(files[-1], encoding="utf-8").read()


def test_three_flags_produce_the_same_three_distinct_sections_as_the_gui(cli_runner):
    from localm.cli import main
    result = cli_runner.invoke(main, [
        "bug-report", "-m", "I clicked generate on the image tab",
        "-e", "a picture of a cat",
        "-w", "a blank grey square appeared instead",
    ])
    assert result.exit_code == 0, result.output
    text = _saved_report_text(cfg.HOME_DIR)
    assert "I clicked generate on the image tab" in text
    assert "a picture of a cat" in text
    assert "a blank grey square appeared instead" in text
    assert text.startswith("# localm bug report: a blank grey square appeared instead")


def test_message_only_renders_not_stated_not_a_duplicate(cli_runner):
    """-m alone must not echo into both the title and the "What happened"
    body."""
    from localm.cli import main
    result = cli_runner.invoke(main, [
        "bug-report", "-m", "the image generator crashed when I clicked twice",
    ])
    assert result.exit_code == 0, result.output
    text = _saved_report_text(cfg.HOME_DIR)
    assert "## What happened\n(not stated)" in text
    happened_section = text.split("## What happened", 1)[1]
    assert "the image generator crashed" not in happened_section


def test_no_flags_no_tty_refuses_without_writing_a_file(cli_runner):
    from localm.cli import main
    result = cli_runner.invoke(main, ["bug-report"])
    assert result.exit_code == 0, result.output
    assert "Describe the problem first" in result.output
    reports_dir = cfg.HOME_DIR / "bug-reports"
    assert not reports_dir.is_dir() or not list(reports_dir.glob("bug-*.md"))


def test_happened_alone_is_enough_like_the_gui(cli_runner):
    from localm.cli import main
    result = cli_runner.invoke(main, ["bug-report", "-w", "it crashed on startup"])
    assert result.exit_code == 0, result.output
    text = _saved_report_text(cfg.HOME_DIR)
    assert "it crashed on startup" in text


def test_no_log_flag_is_threaded_through_to_save_user_report(cli_runner, monkeypatch):
    from localm.cli import main
    from localm import bugreport
    captured = {}
    real = bugreport.save_user_report

    def _spy(*a, **kw):
        captured["include_log"] = kw.get("include_log")
        return real(*a, **kw)

    monkeypatch.setattr(bugreport, "save_user_report", _spy)
    result = cli_runner.invoke(main, ["bug-report", "-m", "x", "--no-log"])
    assert result.exit_code == 0, result.output
    assert captured["include_log"] is False


def test_log_attach_defaults_true_without_the_flag(cli_runner, monkeypatch):
    from localm.cli import main
    from localm import bugreport
    captured = {}
    real = bugreport.save_user_report

    def _spy(*a, **kw):
        captured["include_log"] = kw.get("include_log")
        return real(*a, **kw)

    monkeypatch.setattr(bugreport, "save_user_report", _spy)
    result = cli_runner.invoke(main, ["bug-report", "-m", "x"])
    assert result.exit_code == 0, result.output
    assert captured["include_log"] is True


def test_interactive_no_flags_prompts_for_all_three_fields(cli_runner, monkeypatch):
    """A real terminal (isatty True) with no flags gets the same three
    questions the GUI form asks, not a bare one-line prompt.

    CliRunner.invoke() REPLACES sys.stdin with a fresh
    click.testing._NamedTextIOWrapper for the duration of the call, so the
    CLASS is patched rather than the object captured before invoke()."""
    from click.testing import _NamedTextIOWrapper
    from localm.cli import main
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda self: True)
    result = cli_runner.invoke(
        main, ["bug-report"],
        # A 4th blank line answers the interactive send-menu prompt
        # ("[Enter] not now") reached after the three field questions.
        input="doing X\nexpected Y\nactually Z\n\n")
    assert result.exit_code == 0, result.output
    assert "What were you doing?" in result.output
    assert "What did you expect to happen?" in result.output
    assert "What actually happened?" in result.output
    text = _saved_report_text(cfg.HOME_DIR)
    assert "doing X" in text
    assert "expected Y" in text
    assert "actually Z" in text


def test_interactive_flags_skip_only_their_own_prompt(cli_runner, monkeypatch):
    """A flag that already answers a question suppresses that question; only
    the remaining, un-answered ones are asked."""
    from click.testing import _NamedTextIOWrapper
    from localm.cli import main
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda self: True)
    result = cli_runner.invoke(
        main, ["bug-report", "-m", "already given"],
        # skip "expected" (optional), answer "happened", blank line for the
        # send-menu prompt reached afterward.
        input="\nactually Z\n\n")
    assert result.exit_code == 0, result.output
    assert "What were you doing?" not in result.output
    assert "What actually happened?" in result.output
    text = _saved_report_text(cfg.HOME_DIR)
    assert "already given" in text
    assert "actually Z" in text
