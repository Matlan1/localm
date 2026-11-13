# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bug reports: only the reported run's own log is attached, two reports saved
in one second keep separate files, the review text is the uploaded text, a
failed `bug-report --send` exits 1, and `issues NUMBER` says when this install
did not file the issue."""

import importlib
import os

import pytest

import localm.bugreport as br
from localm.bugreport import LocalmError
from localm.cli import main

maintenance = importlib.import_module("localm.cli.maintenance")


def _log(home, name):
    d = home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("2026-11-13 08:00:00,000 WARNING something from another run\n",
                 encoding="utf-8")
    return p


def test_a_known_run_never_gets_another_runs_log(tmp_path):
    _log(tmp_path, "localm_20261113_080000_111.log")

    assert br._find_run_log(tmp_path, 222) is None
    digest, reason = br._recent_log_tail_result(tmp_path, pid=222)
    assert digest == ""
    assert reason == br._LOG_UNAVAILABLE_NO_FILE


def test_a_known_run_gets_its_own_log(tmp_path):
    _log(tmp_path, "localm_20261113_080000_111.log")
    own = _log(tmp_path, "localm_20261113_080001_222.log")

    assert br._find_run_log(tmp_path, 222) == own


def test_two_reports_saved_in_one_second_keep_both(monkeypatch, tmp_path):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)

    first = br.save_report("first", when="20261113-080000")
    second = br.save_report("second", when="20261113-080000")

    assert first is not None and second is not None and first != second
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"
    from localm.bugreport.persistence import _is_saved_report_name
    assert _is_saved_report_name(second.name)


def test_the_review_text_is_what_an_upload_sends():
    text = br.build_report("Something broke", context={"what_happened": "it broke"})
    footer = br._report_footer()
    assert footer in text

    review = br.report_for_review(text)

    assert footer not in review
    assert review == br._strip_report_footer(text)
    assert review.rstrip().endswith("---") is False


@pytest.fixture
def bug_report_cli(cli_runner, monkeypatch, tmp_path):
    saved = tmp_path / "bug-x.md"
    saved.write_text("# Something broke\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(br, "save_user_report", lambda *a, **k: saved)
    monkeypatch.setattr(br, "live_server_hang_trace", lambda *a, **k: "")
    monkeypatch.setattr(br, "live_server_log_pid", lambda *a, **k: None)
    monkeypatch.setattr(maintenance, "_live_server_stall_stacks", lambda: "")
    monkeypatch.setattr(br, "stdin_is_interactive", lambda: False)
    monkeypatch.setattr(br, "upload_config", lambda: ("https://example.invalid/report", None))
    return cli_runner


def test_a_failed_send_exits_1(bug_report_cli, monkeypatch):
    def fail(*a, **k):
        raise LocalmError("upload failed", "the server said no", stage="server_rejected")
    monkeypatch.setattr(br, "upload_report", fail)

    res = bug_report_cli.invoke(main, ["bug-report", "-w", "it broke", "--send"])

    assert "Could not send it" in res.output
    assert res.exit_code == 1, res.output


def test_a_send_with_no_channel_exits_1(bug_report_cli, monkeypatch):
    monkeypatch.setattr(br, "upload_config", lambda: (None, None))

    res = bug_report_cli.invoke(main, ["bug-report", "-w", "it broke", "--send"])

    assert res.exit_code == 1, res.output


def test_a_send_refused_by_network_access_exits_1_and_connects_nowhere(bug_report_cli,
                                                                      monkeypatch):
    from localm import http_ssl
    connects = []
    monkeypatch.setattr(http_ssl, "verified_urlopen",
                        lambda *a, **k: connects.append(a) or pytest.fail("connected"))
    monkeypatch.setenv("LOCALM_NET_MODE", "off")

    res = bug_report_cli.invoke(main, ["bug-report", "-w", "it broke", "--send"])

    assert connects == []
    assert "Network access is off" in " ".join(res.output.split())
    assert res.exit_code == 1, res.output


def test_an_unreadable_config_names_the_file_when_it_refuses(monkeypatch, tmp_path):
    from localm import config
    from localm.bugreport.transport import refuse_when_network_off
    bad = tmp_path / "config.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", bad)
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)

    with pytest.raises(LocalmError) as caught:
        refuse_when_network_off("the report was not sent", exempt=False)

    err = caught.value
    assert err.stage == "network_off"
    assert str(bad) in err.hint
    assert str(bad) in err.summary
    assert "net_mode=off" not in err.hint


def test_a_readable_config_with_network_off_keeps_the_switch_wording(monkeypatch, tmp_path):
    from localm import config
    from localm.bugreport.transport import refuse_when_network_off
    good = tmp_path / "config.json"
    good.write_text('{"net_mode": "off"}', encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", good)
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)

    with pytest.raises(LocalmError) as caught:
        refuse_when_network_off("the report was not sent", exempt=False)

    assert caught.value.hint == config.net_off_refusal("the report was not sent")


def test_the_cli_attaches_the_live_servers_own_log(monkeypatch, tmp_path):
    from localm import instances
    _log(tmp_path, "localm_20261113_080000_111.log")
    server = _log(tmp_path, "localm_20261113_080001_4242.log")
    monkeypatch.setattr(instances, "list_entries", lambda home: [
        {"pid": 9999, "started": "2026-11-13T09:00:00"},
        {"pid": 4242, "started": "2026-11-13T08:00:01"},
        {"pid": 111, "started": "2026-11-13T07:00:00"},
    ])
    monkeypatch.setattr(instances, "entry_process_state",
                        lambda e: ("gone", "") if e["pid"] == 111 else ("same", ""))

    assert br.live_server_log_pid(tmp_path) == 4242
    assert br._find_run_log(tmp_path, 4242) == server


@pytest.mark.parametrize("state", ["gone", "stale", "unknown"])
def test_an_ended_run_whose_pid_is_reused_gives_no_pid(monkeypatch, tmp_path, state):
    from localm import instances
    _log(tmp_path, "localm_20261113_080000_111.log")
    monkeypatch.setattr(instances, "list_entries", lambda home: [{"pid": 111}])
    monkeypatch.setattr(instances, "pid_alive", lambda pid: True)
    monkeypatch.setattr(instances, "entry_process_state", lambda e: (state, ""))
    assert br.live_server_log_pid(tmp_path) is None


def test_a_cli_report_reads_the_log_of_the_pid_it_is_given(monkeypatch, tmp_path):
    from localm.bugreport import persistence
    asked = []
    monkeypatch.setattr(persistence, "_recent_log_tail_result",
                        lambda **k: asked.append(k["pid"]) or ("", "none"))
    monkeypatch.setattr(br, "save_report", lambda text: tmp_path / "bug.md")

    br.save_user_report("x", include_log=True, log_pid=4242)
    br.save_user_report("x", include_log=True)

    assert asked == [4242, os.getpid()]


def test_the_cli_passes_the_live_servers_pid(cli_runner, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(br, "save_user_report",
                        lambda *a, **k: seen.update(k) or None)
    monkeypatch.setattr(br, "live_server_hang_trace", lambda *a, **k: "")
    monkeypatch.setattr(br, "live_server_log_pid", lambda *a, **k: 4242)
    monkeypatch.setattr(maintenance, "_live_server_stall_stacks", lambda: "")
    monkeypatch.setattr(br, "stdin_is_interactive", lambda: False)
    monkeypatch.setattr(br, "upload_config", lambda: (None, None))

    cli_runner.invoke(main, ["bug-report", "-w", "it broke"])
    assert seen["log_pid"] == 4242
    cli_runner.invoke(main, ["bug-report", "-w", "it broke", "--no-log"])
    assert seen["log_pid"] is None and seen["include_log"] is False


def test_a_sent_report_exits_0(bug_report_cli, monkeypatch):
    monkeypatch.setattr(br, "upload_report", lambda *a, **k: {"url": "https://example.invalid/1"})

    res = bug_report_cli.invoke(main, ["bug-report", "-w", "it broke", "--send"])

    assert "Sent to the maintainer" in res.output
    assert res.exit_code == 0, res.output


@pytest.mark.parametrize("filed,note", [([], True), ([{"number": 7, "title": "x"}], False)])
def test_issues_number_says_when_this_install_did_not_file_it(cli_runner, monkeypatch,
                                                              filed, note):
    from localm import issue_tracker
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "get_issue",
                        lambda n, **k: {"number": n, "state": "open", "title": "t"})
    monkeypatch.setattr(issue_tracker, "recorded", lambda: filed)

    res = cli_runner.invoke(main, ["issues", "7"])

    assert res.exit_code == 0, res.output
    assert ("This install did not file this issue" in res.output) is note


def test_the_current_process_is_not_given_its_own_log_as_another_runs(tmp_path):
    _log(tmp_path, f"localm_20261113_080000_{os.getpid()}.log")
    assert br._find_run_log(tmp_path, None) is None
