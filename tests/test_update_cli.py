# SPDX-License-Identifier: AGPL-3.0-or-later
"""The `localm update` and `localm issues` CLI commands (monkeypatched updater /
issue_tracker so no network). Apply is user-initiated; failures are surfaced, not
faked."""

from click.testing import CliRunner

from localm import issue_tracker, updater
from localm.cli import main


# ------------------------------ update ----------------------------------

def test_update_not_configured(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: False)
    r = CliRunner().invoke(main, ["update"])
    assert r.exit_code == 0 and "not configured" in r.output


def test_update_check_up_to_date(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.2.0", "latest": "0.2.0", "newer": False, "notes": "", "asset": None})
    r = CliRunner().invoke(main, ["update", "--check"])
    assert "up to date" in r.output


def test_update_check_unrecognized_tag_reports_uncertainty_not_up_to_date(monkeypatch):
    """When the comparator could not order the tags (info's "comparable" is
    False), the CLI says so and never reports "up to date"."""
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.4", "latest": "nightly", "newer": False, "comparable": False,
        "notes": "", "asset": None})
    r = CliRunner().invoke(main, ["update", "--check"])
    assert "Could not tell" in r.output
    assert "up to date" not in r.output


def test_update_check_blocked_by_net_policy(monkeypatch):
    """A net-policy refusal flows through the 'Could not check for updates'
    path: it shows the short reason and never claims the update is up to
    date."""
    from localm.bugreport import LocalmError
    monkeypatch.setattr(updater, "available", lambda: True)

    def boom():
        raise LocalmError(
            "update checks are off because network access is set to off",
            reason='turn on "Check for updates even when network access is off" '
                   "in Settings, or set network access to ask or allow")

    monkeypatch.setattr(updater, "check", boom)
    r = CliRunner().invoke(main, ["update", "--check"])
    output = " ".join(r.output.split())   # rich wraps long lines; normalize whitespace first
    assert "Could not check for updates" in output
    assert "network access is set to off" in output
    assert "up to date" not in output


def test_update_check_reports_available(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "cool stuff",
        "asset": {"id": 3}})
    r = CliRunner().invoke(main, ["update", "--check"])
    assert "Update available: v0.2.0" in r.output
    assert "Run `localm update` to apply" in r.output


def test_update_applies_with_yes(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "", "asset": {"id": 3}})
    captured = {}

    def fake_apply(aid, **kw):
        captured["aid"] = aid
        return {"applied": True, "version": "0.2.0", "klass": "reboot", "backup": "b"}

    monkeypatch.setattr(updater, "apply", fake_apply)
    r = CliRunner().invoke(main, ["update", "--yes"])
    assert captured["aid"] == 3
    assert "Updated to 0.2.0" in r.output
    assert "Restart localm" in r.output


def test_update_apply_failure_is_surfaced(monkeypatch):
    from localm.bugreport import LocalmError
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "", "asset": {"id": 3}})

    def boom(aid, **kw):
        raise LocalmError("the post-update step failed; rolled back", reason="uv exited 1")

    monkeypatch.setattr(updater, "apply", boom)
    r = CliRunner().invoke(main, ["update", "--yes"])
    assert "Update failed" in r.output and "rolled back" in r.output


def test_update_setup_class_warns_to_run_setup(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "", "asset": {"id": 3}})
    monkeypatch.setattr(updater, "apply", lambda aid, **kw: {
        "applied": True, "version": "0.2.0", "klass": "setup", "backup": "b"})
    r = CliRunner().invoke(main, ["update", "--yes"])
    assert "setup.bat" in r.output


def test_update_rollback(monkeypatch):
    monkeypatch.setattr(updater, "rollback_last", lambda: {"rolled_back": True, "backup": "b"})
    r = CliRunner().invoke(main, ["update", "--rollback"])
    assert "Rolled back" in r.output


# ------------------------------ issues ----------------------------------

def test_issues_list(monkeypatch):
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "list_issues", lambda state="all": [
        {"number": 5, "state": "open", "title": "bug a"},
        {"number": 6, "state": "closed", "title": "bug b"}])
    r = CliRunner().invoke(main, ["issues"])
    assert "#5" in r.output and "#6" in r.output and "2 issue" in r.output


def test_issues_one(monkeypatch):
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "get_issue",
                        lambda n: {"number": 12, "state": "closed", "title": "fixed", "html_url": "u"})
    r = CliRunner().invoke(main, ["issues", "12"])
    assert "#12" in r.output and "closed" in r.output


def test_issues_not_configured(monkeypatch):
    monkeypatch.setattr(issue_tracker, "available", lambda: False)
    r = CliRunner().invoke(main, ["issues"])
    assert "not configured" in r.output
