# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end coverage for the cross-cutting graceful-failure handler
(_GracefulGroup in cli.py).

An unexpected crash in ANY subcommand must become a "sorry X for Y" + bug report
(exit 1), not a raw traceback; a typed LocalmError carries its own summary/reason;
ordinary Click usage errors and Ctrl+C pass through untouched; and a failure
INSIDE reporting must not recurse into the hard crash we are trying to avoid.
"""

from __future__ import annotations

import click
from click.testing import CliRunner

from localm import bugreport
from localm.cli import _GracefulGroup


def _group():
    @click.group(cls=_GracefulGroup)
    def g():
        pass

    @g.command()
    def boom():
        raise RuntimeError("kaboom")

    @g.command()
    def known():
        raise bugreport.LocalmError("known bad thing", reason="because reasons")

    @g.command("usage")
    def usage_cmd():
        raise click.ClickException("you used it wrong")

    return g


def test_unexpected_exception_is_caught_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    res = CliRunner().invoke(_group(), ["boom"])
    assert res.exit_code == 1
    assert "Sorry -" in res.output
    assert "unexpected error" in res.output
    assert list((tmp_path / "bug-reports").glob("*.md"))   # a report was written


def test_localm_error_uses_summary_and_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    res = CliRunner().invoke(_group(), ["known"])
    assert res.exit_code == 1
    assert "known bad thing" in res.output
    assert "because reasons" in res.output


def test_click_usage_error_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    res = CliRunner().invoke(_group(), ["usage"])
    # Rendered by Click as a usage error, NOT routed to the bug reporter.
    assert "Sorry -" not in res.output
    assert "you used it wrong" in res.output


def test_reporting_failure_does_not_recurse(tmp_path, monkeypatch):
    # If report_failure itself blows up, the handler must still exit cleanly (1),
    # not propagate a second exception.
    def boom_reporter(**k):
        raise RuntimeError("reporter itself broke")
    monkeypatch.setattr(bugreport, "report_failure", boom_reporter)
    res = CliRunner().invoke(_group(), ["boom"])
    assert res.exit_code == 1
    assert "localm failed" in res.output


def test_broken_pipe_passes_through_without_reporting(tmp_path, monkeypatch):
    # A closed pipe (localm ... | head) surfaces as BrokenPipeError: no report,
    # no traceback cascade, a clean SIGPIPE-style exit.
    import errno as _errno
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    reported = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: reported.append(k))

    @click.group(cls=_GracefulGroup)
    def g():
        pass

    @g.command()
    def bpipe():
        raise BrokenPipeError(_errno.EPIPE, "broken pipe")

    res = CliRunner().invoke(g, ["bpipe"])
    assert res.exit_code == 0               # clean exit, not a crash/exit 1
    assert reported == []                   # NOT reported as a bug
    assert "Sorry -" not in res.output
    assert "localm failed" not in res.output
