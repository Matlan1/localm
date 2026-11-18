# SPDX-License-Identifier: AGPL-3.0-or-later
"""A blank ``bugreport_upload_url`` in the data folder's config.json means no
bug-report destination, in every standalone reporter.

localm itself reads that key from the data folder over the shipped default, and
an empty value turns the upload channel off. The three reporters that run when
localm will not start (scripts/report_issue.py, scripts/report_issue.ps1 and the
curl fallback in report-issue.sh) used to read only the shipped default, so a
person who blanked the URL still saw the shipped host in the preview and a yes
at the send question sent there.
"""

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

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "report_issue_blank_url_under_test", _REPO / "scripts" / "report_issue.py")
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

SHIPPED = "https://shipped.example.invalid"


def _shipped_config(path: Path, url: str = SHIPPED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('D = {\n    "bugreport_upload_url": "' + url + '",\n'
                    '    "bugreport_upload_token": "shipped-token",\n}\n', encoding="utf-8")
    return path


def _home(tmp_path: Path, monkeypatch, config) -> None:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if config is not None:
        (data / "config.json").write_text(
            config if isinstance(config, str) else json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.delenv("LOCALM_BUGREPORT_URL", raising=False)
    monkeypatch.delenv("LOCALM_BUGREPORT_TOKEN", raising=False)


# ------------------------------- the Python reporter ----------------------- #

@pytest.mark.parametrize("blank", ["", "   "])
def test_python_reporter_a_blank_url_in_the_data_folder_is_no_destination(
        tmp_path, monkeypatch, blank):
    _home(tmp_path, monkeypatch, {"bugreport_upload_url": blank})
    url, token = ri.read_proxy(_shipped_config(tmp_path / "config.py"))
    assert url is None
    assert token == "shipped-token"


def test_python_reporter_a_blank_token_in_the_data_folder_drops_the_shipped_token(
        tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, {"bugreport_upload_token": ""})
    url, token = ri.read_proxy(_shipped_config(tmp_path / "config.py"))
    assert url == SHIPPED
    assert token is None


@pytest.mark.parametrize("config", [None, {}, {"net_mode": "ask"}, "{not json",
                                    {"bugreport_upload_url": None},
                                    {"bugreport_upload_url": 5}])
def test_python_reporter_keeps_the_shipped_url_when_the_data_folder_does_not_say(
        tmp_path, monkeypatch, config):
    _home(tmp_path, monkeypatch, config)
    url, _ = ri.read_proxy(_shipped_config(tmp_path / "config.py"))
    assert url == SHIPPED


def test_python_reporter_a_url_in_the_data_folder_replaces_the_shipped_one(
        tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, {"bugreport_upload_url": " https://mine.example.invalid/x "})
    url, _ = ri.read_proxy(_shipped_config(tmp_path / "config.py"))
    assert url == "https://mine.example.invalid/x"


def test_python_reporter_the_environment_override_still_wins(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, {"bugreport_upload_url": ""})
    monkeypatch.setenv("LOCALM_BUGREPORT_URL", "https://env.example.invalid/x")
    url, _ = ri.read_proxy(_shipped_config(tmp_path / "config.py"))
    assert url == "https://env.example.invalid/x"


def test_python_reporter_preview_says_none_configured_and_sends_nothing(
        tmp_path, monkeypatch, capsys):
    _home(tmp_path, monkeypatch, {"bugreport_upload_url": "", "net_mode": "ask"})
    _shipped_config(tmp_path / "localm" / "config.py")
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri.sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())

    def boom(*a, **k):
        raise AssertionError("a report was sent although the destination is blank")

    monkeypatch.setattr(ri, "post_report", boom)
    rc = ri.main(["--yes", "--summary", "x"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Destination: none configured" in out
    assert "shipped.example.invalid" not in out


# ------------------------------- the PowerShell reporter ------------------- #

class _Recorder(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
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
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/report", _Recorder.hits
    finally:
        srv.shutdown()
        srv.server_close()


def _env(data: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCALM_NET_MODE", "LOCALM_BUGREPORT_URL", "LOCALM_BUGREPORT_TOKEN")}
    env["LOCALM_HOME"] = str(data)
    return env


def _ps1(tmp_path: Path, shipped: str, config) -> tuple:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    (tmp_path / "scripts").mkdir()
    shutil.copy(_REPO / "scripts" / "report_issue.ps1", tmp_path / "scripts")
    _shipped_config(tmp_path / "localm" / "config.py", shipped)
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
         str(tmp_path / "scripts" / "report_issue.ps1"), "--summary", "x", "--yes"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180,
        env=_env(data))
    return proc.returncode, " ".join((proc.stdout + proc.stderr).split())


def test_powershell_reporter_a_blank_url_sends_nothing(tmp_path, recorder):
    url, hits = recorder
    rc, out = _ps1(tmp_path, url, {"net_mode": "allow", "bugreport_upload_url": ""})
    assert hits == [], out
    assert rc == 1, out
    assert "No bug-report endpoint is configured" in out


def test_powershell_reporter_without_the_key_still_sends_to_the_shipped_url(
        tmp_path, recorder):
    url, hits = recorder
    rc, out = _ps1(tmp_path, url, {"net_mode": "allow"})
    assert hits == ["/report"], out
    assert rc == 0, out


# ------------------------------- the curl fallback ------------------------- #

def _sh(tmp_path: Path, shipped: str, config) -> tuple:
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
    _shipped_config(work / "localm" / "config.py", shipped)
    sink = work / "curl_called"
    (fake / "curl").write_text('#!/bin/sh\ntouch "' + sink.as_posix() + '"\necho 201\n',
                               encoding="utf-8", newline="\n")
    (fake / "curl").chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.run(
        [bash, "-c",
         'p="$(cygpath -u "$1" 2>/dev/null || printf %s "$1")"; '
         'export LOCALM_HOME="$(cygpath -u "$3" 2>/dev/null || printf %s "$3")"; '
         'PATH="$p:$PATH"; w="$2"; cd "$w" && bash fallback.sh',
         "_", fake.as_posix(), work.as_posix(), data.as_posix()],
        input="x\n\n", capture_output=True, text=True, timeout=120, env=_env(data))
    return proc.returncode, " ".join((proc.stdout + proc.stderr).split()), sink.exists()


def test_curl_fallback_a_blank_url_sends_nothing(tmp_path):
    rc, out, sent = _sh(tmp_path, SHIPPED, {"net_mode": "allow", "bugreport_upload_url": ""})
    assert not sent, out
    assert rc == 1, out
    assert "Destination: none configured" in out


def test_curl_fallback_without_the_key_still_sends_to_the_shipped_url(tmp_path):
    rc, out, sent = _sh(tmp_path, SHIPPED, {"net_mode": "allow"})
    assert sent, out
    assert rc == 0, out
