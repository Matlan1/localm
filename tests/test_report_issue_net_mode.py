# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone reporters (scripts/report_issue.py, scripts/report_issue.ps1 and
the curl fallback in report-issue.sh) follow localm's Network access setting: off
sends nothing and says how to turn it on; a setting that cannot be read asks before
anything is sent; ask and allow send as before."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from localm import config

_REPO = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO / "scripts" / "report_issue.py"
_spec = importlib.util.spec_from_file_location("report_issue_net_mode", _MOD_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

REFUSAL = config.net_off_refusal("the report was not sent")


def test_the_reporters_copy_the_net_off_remedy():
    assert ri.NET_OFF_REMEDY == config.NET_OFF_REMEDY
    assert ri.NET_OFF_REFUSAL == REFUSAL
    ps1 = (_REPO / "scripts" / "report_issue.ps1").read_text(encoding="utf-8")
    assert f"$netOffRemedy = '{config.NET_OFF_REMEDY}'" in ps1
    assert ("Network access is off (net_mode=off), so the report was not sent. "
            "$netOffRemedy.") in ps1
    sh = (_REPO / "report-issue.sh").read_text(encoding="utf-8")
    assert f'net_off_remedy="{config.NET_OFF_REMEDY}"' in sh
    assert ("Network access is off (net_mode=off), so the report was not sent. "
            "${net_off_remedy}.") in sh


# ----------------------------- the Python reporter ------------------------ #

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "data"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    return h


@pytest.mark.parametrize("text, mode", [
    ('{"net_mode": "off"}', "off"),
    ('{"net_mode": " OFF "}', "off"),
    ('{"net_mode": "allow"}', "allow"),
    ('{"net_mode": "ask"}', "ask"),
    ('{}', "ask"),
    ('{"net_mode": "sometimes"}', "ask"),
    ('﻿{"net_mode": "off"}', "off"),
])
def test_the_setting_is_read_from_config_json(home, text, mode):
    (home / "config.json").write_text(text, encoding="utf-8")
    assert ri.network_mode() == (mode, "")


@pytest.mark.parametrize("text, why", [
    (None, "does not exist"),
    ("{not json", "is not valid JSON"),
    ("[]", "does not hold a settings object"),
])
def test_a_missing_or_unreadable_setting_is_unknown(home, text, why):
    if text is not None:
        (home / "config.json").write_text(text, encoding="utf-8")
    mode, reason = ri.network_mode()
    assert mode is None
    assert why in reason


def test_the_environment_wins_over_config_json(home, monkeypatch):
    (home / "config.json").write_text('{"net_mode": "allow"}', encoding="utf-8")
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    assert ri.network_mode() == ("off", "")


def test_the_data_dir_comes_from_localm_home_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    data = tmp_path / "elsewhere"
    data.mkdir()
    (data / "config.json").write_text('{"net_mode": "off"}', encoding="utf-8")
    (tmp_path / "localm-home.cfg").write_text(f"\n{data}\n", encoding="utf-8")
    assert ri.network_mode() == ("off", "")
    (tmp_path / "localm-home.cfg").unlink()
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "config.json").write_text('{"net_mode": "allow"}',
                                                   encoding="utf-8")
    assert ri.network_mode() == ("allow", "")


def _tty(monkeypatch, interactive, answers=()):
    monkeypatch.setattr(ri, "stdin_is_interactive", lambda: interactive)
    asked = []
    pending = list(answers)

    def _input(prompt=""):
        asked.append(prompt)
        return pending.pop(0) if pending else ""
    monkeypatch.setattr("builtins.input", _input)
    return asked


def _sends(monkeypatch):
    sent = []
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy.invalid/r", None))
    monkeypatch.setattr(ri, "post_report",
                        lambda *a, **k: sent.append(a) or {"url": "https://x/1"})
    return sent


def test_off_sends_nothing_and_says_how_to_turn_it_on(home, monkeypatch, capsys):
    (home / "config.json").write_text('{"net_mode": "off"}', encoding="utf-8")
    asked = _tty(monkeypatch, True)
    sent = _sends(monkeypatch)
    rc = ri.main(["--yes", "--summary", "x", "--detail", "d"])
    out = capsys.readouterr().out
    assert sent == []
    assert rc == 1
    assert REFUSAL in out
    assert asked == []
    assert list((ri.REPO_ROOT / "home" / "bug-reports").glob("bug-*.md"))


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_ask_and_allow_send_as_before(home, monkeypatch, capsys, mode):
    (home / "config.json").write_text(json.dumps({"net_mode": mode}), encoding="utf-8")
    _tty(monkeypatch, False)
    sent = _sends(monkeypatch)
    assert ri.main(["--yes", "--summary", "x"]) == 0
    assert len(sent) == 1
    assert "Sent to the maintainer" in capsys.readouterr().out


def test_an_unreadable_setting_with_no_one_to_ask_sends_nothing(home, monkeypatch, capsys):
    (home / "config.json").write_text("{not json", encoding="utf-8")
    _tty(monkeypatch, False)
    sent = _sends(monkeypatch)
    rc = ri.main(["--yes", "--summary", "x"])
    out = capsys.readouterr().out
    assert sent == []
    assert rc == 1
    assert "could not be read" in out and "is not valid JSON" in out
    assert "nothing was sent" in out


@pytest.mark.parametrize("answer, sends", [("y", True), ("yes", True), ("", False),
                                           ("n", False)])
def test_an_unreadable_setting_asks_once_and_defaults_to_no(home, monkeypatch, capsys,
                                                             answer, sends):
    asked = _tty(monkeypatch, True, ["x", "", answer])
    sent = _sends(monkeypatch)
    rc = ri.main(["--yes"])
    out = capsys.readouterr().out
    assert len(sent) == (1 if sends else 0)
    assert rc == 0
    assert "does not exist" in out
    assert [a for a in asked if "anyway" in a] == ["  Send it anyway? [y/N]: "]
    assert not [a for a in asked if "Send this to the maintainer" in a]


# ------------------------ the PowerShell and bash reporters ---------------- #

class _Recorder(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        self.rfile.read(n)
        type(self).hits.append(self.path)
        body = b'{"url": "https://example.invalid/1"}'
        self.send_response(201)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def recorder():
    _Recorder.hits = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Recorder)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/report", _Recorder.hits
    finally:
        srv.shutdown()
        srv.server_close()


def _clean_env(tmp_path, url, data):
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCALM_NET_MODE", "LOCALM_BUGREPORT_TOKEN")}
    env.update(LOCALM_BUGREPORT_URL=url, LOCALM_HOME=str(data))
    return env


def _ps1_case(tmp_path, recorder, config_text, *args):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    url, hits = recorder
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(_REPO / "scripts" / "report_issue.ps1", tmp_path / "scripts")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if config_text is not None:
        (data / "config.json").write_text(config_text, encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
         str(tmp_path / "scripts" / "report_issue.ps1"), "--summary", "x", *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180,
        env=_clean_env(tmp_path, url, data))
    return proc.returncode, " ".join((proc.stdout + proc.stderr).split()), hits


def test_powershell_off_sends_nothing(tmp_path, recorder):
    rc, out, hits = _ps1_case(tmp_path, recorder, '{"net_mode": "off"}', "--yes")
    assert hits == []
    assert rc == 1, out
    assert REFUSAL in out


@pytest.mark.parametrize("config_text", [None, "{not json", "[]"])
def test_powershell_unreadable_setting_with_no_one_to_ask_sends_nothing(
        tmp_path, recorder, config_text):
    rc, out, hits = _ps1_case(tmp_path, recorder, config_text, "--yes")
    assert hits == []
    assert rc == 1, out
    assert "Network access setting could not be read" in out
    assert "nothing was sent" in out


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_powershell_ask_and_allow_send(tmp_path, recorder, mode):
    rc, out, hits = _ps1_case(tmp_path, recorder, json.dumps({"net_mode": mode}), "--yes")
    assert hits == ["/report"], out
    assert rc == 0, out
    assert "Sent to the maintainer" in out


def _sh_case(tmp_path, config_text, answers):
    """Run report-issue.sh's curl fallback (its terminal check dropped) with a fake
    curl that records a call, typing *answers*."""
    bash = shutil.which("bash")
    if not bash or "system32" in bash.lower() or not shutil.which("perl"):
        pytest.skip("no bash with perl on PATH")
    text = (_REPO / "report-issue.sh").read_text(encoding="utf-8")
    fallback = text[text.index("# --- No Python at all"):]
    assert fallback.count("if [ ! -t 0 ]; then") == 1
    fallback = fallback.replace("if [ ! -t 0 ]; then", "if false; then")
    work = tmp_path / "clone"
    fake = work / "bin"
    fake.mkdir(parents=True)
    (work / "fallback.sh").write_text(fallback, encoding="utf-8", newline="\n")
    sink = work / "curl_called"
    (fake / "curl").write_text('#!/bin/sh\ntouch "' + sink.as_posix() + '"\necho 201\n',
                               encoding="utf-8", newline="\n")
    (fake / "curl").chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    if config_text is not None:
        (data / "config.json").write_text(config_text, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "LOCALM_NET_MODE"}
    env["LOCALM_BUGREPORT_URL"] = "http://127.0.0.1:9/report"
    proc = subprocess.run(
        [bash, "-c",
         'p="$(cygpath -u "$1" 2>/dev/null || printf %s "$1")"; '
         'export LOCALM_HOME="$(cygpath -u "$3" 2>/dev/null || printf %s "$3")"; '
         'PATH="$p:$PATH"; cd "$2" && bash fallback.sh',
         "_", fake.as_posix(), work.as_posix(), data.as_posix()],
        input="".join(a + "\n" for a in answers), capture_output=True, text=True,
        timeout=120, env=env)
    return proc.returncode, " ".join((proc.stdout + proc.stderr).split()), sink.exists()


def test_bash_fallback_off_sends_nothing(tmp_path):
    rc, out, sent = _sh_case(tmp_path, '{"net_mode": "off"}', ["x", "y"])
    assert not sent, out
    assert rc == 1, out
    assert REFUSAL in out


@pytest.mark.parametrize("config_text", [None, "{not json", "[]"])
@pytest.mark.parametrize("answer, sends", [("y", True), ("", False), ("n", False)])
def test_bash_fallback_unreadable_setting_asks_and_defaults_to_no(tmp_path, config_text,
                                                                   answer, sends):
    rc, out, sent = _sh_case(tmp_path, config_text, ["x", answer])
    assert sent is sends, out
    assert "Network access setting could not be read" in out
    assert "Send it anyway? [y/N]" in out
    assert "Send this to the maintainer now?" not in out


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_bash_fallback_ask_and_allow_send_after_the_usual_confirm(tmp_path, mode):
    rc, out, sent = _sh_case(tmp_path, json.dumps({"net_mode": mode}), ["x", ""])
    assert sent, out
    assert rc == 0, out
    assert "Send this to the maintainer now? [Y/n]" in out
    assert "Sent to the maintainer" in out
