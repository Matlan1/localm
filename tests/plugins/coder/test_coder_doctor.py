# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm coder doctor: each check against a real server, repo, MCP process."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm.plugins.coder import doctor
from localm.plugins.coder.backends.http import (
    HTTPBackend,
    make_localm_backend,
    make_openai_backend,
)
from localm.plugins.coder.doctor import FAIL, INFO, OK, SKIP, WARN, Check


def _serve(status: int, body: dict | None):
    """A real loopback HTTP server answering every GET with *status*/*body*."""
    payload = json.dumps(body or {}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                     # noqa: N802
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def serve():
    servers = []

    def make(status=200, body=None):
        srv = _serve(status, body)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1"

    yield make
    for s in servers:
        s.shutdown()
        s.server_close()


# ---------------------------------------------------------------- model

def test_model_ok_when_server_lists_it(serve):
    url = serve(200, {"data": [{"id": "m1"}, {"id": "m2"}]})
    c = doctor.check_model(HTTPBackend(url, "m1"))
    assert (c.status, c.name) == (OK, "model")


def test_model_warns_when_not_listed(serve):
    url = serve(200, {"data": [{"id": "other"}]})
    c = doctor.check_model(HTTPBackend(url, "m1"))
    assert c.status == WARN and "does not list" in c.detail


def test_model_placeholder_name_is_not_flagged(serve):
    url = serve(200, {"data": [{"id": "other"}]})
    assert doctor.check_model(HTTPBackend(url, "localm")).status == OK


@pytest.mark.parametrize("code", [401, 403])
def test_model_auth_rejection_fails(serve, code):
    c = doctor.check_model(HTTPBackend(serve(code), "m1"))
    assert c.status == FAIL and "API key" in c.detail and str(code) in c.detail


def test_model_server_error_fails(serve):
    assert doctor.check_model(HTTPBackend(serve(503), "m1")).status == FAIL


def test_model_missing_models_endpoint_warns_not_fails(serve):
    c = doctor.check_model(HTTPBackend(serve(404), "m1"))
    assert c.status == WARN and "404" in c.detail


def test_model_unreachable_fails():
    srv = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.server_close()
    c = doctor.check_model(HTTPBackend(f"http://127.0.0.1:{port}/v1", "m1"))
    assert c.status == FAIL and "not reachable" in c.detail


def test_model_backend_without_http_is_skipped():
    class InProcess:
        model_id = "local"

    assert doctor.check_model(InProcess()).status == SKIP


# ---------------------------------------------------------------- tool calls

def test_tools_native_for_openai_backend():
    c = doctor.check_tools(make_openai_backend())
    assert c.status == OK and "native" in c.detail


def test_tools_grammar_for_localm_server():
    c = doctor.check_tools(make_localm_backend("m"))
    assert c.status == OK and "grammar" in c.detail


def test_tools_unconstrained_for_plain_url_without_native_flag():
    c = doctor.check_tools(HTTPBackend("http://127.0.0.1:1/v1", "m"))
    assert c.status == WARN and "not grammar" in c.detail


def test_tools_native_flag_ignored_on_localm_server():
    b = make_localm_backend("m", native_tools=True)
    assert "grammar" in doctor.check_tools(b).detail


# ---------------------------------------------------------------- context

class _Map:
    def __init__(self, chars: int, files: int = 7):
        self._chars, self._files = chars, files

    def to_context_string(self):
        return "x" * self._chars

    def file_count(self):
        return self._files


class _Cap:
    def __init__(self, cap):
        self._cap = cap

    def context_capacity(self):
        return self._cap


@pytest.mark.parametrize("map_tokens,status", [
    (100, OK), (250, WARN), (499, WARN), (500, FAIL), (4000, FAIL)])
def test_context_thresholds(tmp_path, map_tokens, status):
    c = doctor.check_context(_Cap(1000), tmp_path, _Map(map_tokens * 4))
    assert c.status == status


def test_context_without_reported_window_is_info(tmp_path):
    c = doctor.check_context(_Cap(None), tmp_path, _Map(400))
    assert c.status == INFO and "~100 tokens" in c.detail


def test_context_builds_a_real_map_when_none_given(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    c = doctor.check_context(_Cap(100000), tmp_path)
    assert c.status == OK and "1 files" in c.detail


# ---------------------------------------------------------------- git

def _git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True,
                   capture_output=True)


def test_git_clean_then_dirty(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "trunk")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@e.x",
         "commit", "--allow-empty", "-q", "-m", "init")
    clean = doctor.check_git(tmp_path)
    assert clean.status == OK and "branch trunk, clean" in clean.detail
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    dirty = doctor.check_git(tmp_path)
    assert dirty.status == INFO and "1 changed file" in dirty.detail


def test_git_that_times_out_warns_instead_of_failing(tmp_path, monkeypatch):
    def slow(work_dir, *args):
        raise subprocess.TimeoutExpired("git", 15)

    monkeypatch.setattr(doctor, "_git", slow)
    c = doctor.check_git(tmp_path)
    assert c.status == WARN and "did not answer" in c.detail


def test_git_missing_from_path_warns(tmp_path, monkeypatch):
    def missing(work_dir, *args):
        raise FileNotFoundError("git")

    monkeypatch.setattr(doctor, "_git", missing)
    assert doctor.check_git(tmp_path).status == WARN


def test_git_outside_a_work_tree_warns(tmp_path):
    c = doctor.check_git(tmp_path)
    assert c.status == WARN and "not a git work tree" in c.detail


# ---------------------------------------------------------------- verify

def test_verify_resolvable_command_ok(tmp_path):
    c = doctor.check_verify(tmp_path, f'"{sys.executable}" -c pass')
    assert c.status == OK and "resolves" in c.detail


def test_verify_unresolvable_program_warns(tmp_path):
    c = doctor.check_verify(tmp_path, "no-such-program-xyz --flag")
    assert c.status == WARN and "no-such-program-xyz" in c.detail


def test_verify_list_command_uses_first_element(tmp_path):
    assert doctor.check_verify(tmp_path, [sys.executable, "-c", "pass"]).status == OK


def test_verify_absent_is_info(tmp_path):
    c = doctor.check_verify(tmp_path)
    assert c.status == INFO and "no verify command" in c.detail


def test_verify_picks_up_the_project_config_command(tmp_path):
    cfg = tmp_path / ".localcoder"
    cfg.mkdir()
    (cfg / "config.toml").write_text('verify = "no-such-program-xyz"\n',
                                     encoding="utf-8")
    assert "no-such-program-xyz" in doctor.check_verify(tmp_path).detail


# ---------------------------------------------------------------- config

def test_config_unreadable_fails(tmp_path):
    cfg = tmp_path / ".localcoder"
    cfg.mkdir()
    (cfg / "config.toml").write_text("this is = = not toml", encoding="utf-8")
    assert doctor.check_project_config(tmp_path).status == FAIL


def test_config_readable_ok(tmp_path):
    cfg = tmp_path / ".localcoder"
    cfg.mkdir()
    (cfg / "config.toml").write_text("max_turns = 3\n", encoding="utf-8")
    assert doctor.check_project_config(tmp_path).status == OK


def test_config_absent_is_info(tmp_path):
    assert doctor.check_project_config(tmp_path).status == INFO


# ---------------------------------------------------------------- mcp

_MCP_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        out = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
    elif msg.get("method") == "tools/list":
        out = {"jsonrpc": "2.0", "id": msg["id"],
               "result": {"tools": [{"name": "a"}, {"name": "b"}]}}
    else:
        continue
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
'''


def _write_mcp_config(tmp_path, servers: dict):
    cfg = tmp_path / ".localcoder"
    cfg.mkdir()
    lines = []
    for name, spec in servers.items():
        lines.append(f"[mcp.servers.{name}]")
        lines.append(f"command = {json.dumps(spec['command'])}")
        lines.append(f"args = {json.dumps(spec.get('args', []))}")
    (cfg / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_mcp_none_configured_is_skip(tmp_path):
    assert [c.status for c in doctor.check_mcp(tmp_path)] == [SKIP]


def test_mcp_reachable_and_broken_servers_reported_separately(tmp_path):
    script = tmp_path / "srv.py"
    script.write_text(_MCP_SERVER, encoding="utf-8")
    _write_mcp_config(tmp_path, {
        "good": {"command": sys.executable, "args": [str(script)]},
        "gone": {"command": "no-such-mcp-binary-xyz"},
    })
    got = {c.name: c for c in doctor.check_mcp(tmp_path)}
    assert got["mcp:good"].status == OK and "2 tool(s)" in got["mcp:good"].detail
    assert got["mcp:gone"].status == FAIL and "not found" in got["mcp:gone"].detail


def test_mcp_check_leaves_no_server_process_running(tmp_path):
    script = tmp_path / "srv.py"
    script.write_text(_MCP_SERVER, encoding="utf-8")
    _write_mcp_config(tmp_path, {
        "good": {"command": sys.executable, "args": [str(script)]}})
    from localm.plugins.coder import mcp
    before = dict(mcp._POOL)
    doctor.check_mcp(tmp_path)
    assert dict(mcp._POOL) == before


# ---------------------------------------------------------------- runner

def test_a_raising_check_is_reported_as_a_failure_not_dropped():
    got = doctor._guard("boom", lambda: (_ for _ in ()).throw(ValueError("bad")))
    assert got == [Check("boom", FAIL, "check raised ValueError: bad")]


def test_run_doctor_runs_every_check_in_order(serve, tmp_path):
    url = serve(200, {"data": [{"id": "m1"}]})
    checks = doctor.run_doctor(tmp_path, HTTPBackend(url, "m1"))
    assert [c.name for c in checks] == [
        "model", "tool-calls", "context", "git", "verify", "config", "mcp"]
    assert doctor.report_ok(checks)


def test_run_doctor_without_mcp_omits_it(serve, tmp_path):
    url = serve(200, {"data": []})
    checks = doctor.run_doctor(tmp_path, HTTPBackend(url, "m1"),
                               check_mcp_servers=False)
    assert "mcp" not in [c.name for c in checks]


def test_report_not_ok_when_any_check_fails():
    assert not doctor.report_ok([Check("a", OK, ""), Check("b", FAIL, "")])
    assert doctor.report_ok([Check("a", WARN, ""), Check("b", INFO, "")])


# ---------------------------------------------------------------- formats

_SAMPLE = [Check("model", OK, "m: server reachable"),
           Check("git", WARN, "x"), Check("mcp:s", FAIL, "boom")]


def test_text_format_has_one_line_per_check_and_a_tally():
    lines = doctor.render(_SAMPLE, "text").splitlines()
    assert lines[0].startswith("[ok  ] model")
    assert lines[2].startswith("[FAIL] mcp:s")
    assert lines[-1] == "3 checks: 1 failed, 1 warning(s)"


def test_text_format_strips_terminal_control_characters():
    out = doctor.render([Check("model", FAIL, "bad\x1b[31mred")], "text")
    assert "\x1b" not in out


def test_json_format_round_trips():
    data = json.loads(doctor.render(_SAMPLE, "json"))
    assert data["ok"] is False
    assert [c["name"] for c in data["checks"]] == ["model", "git", "mcp:s"]


def test_jsonl_format_is_one_valid_object_per_line_ending_in_summary():
    rows = [json.loads(ln) for ln in doctor.render(_SAMPLE, "jsonl").splitlines()]
    assert [r["type"] for r in rows] == ["check", "check", "check", "summary"]
    assert rows[-1] == {"type": "summary", "ok": False, "failed": 1, "warnings": 1}


_HUNG_MCP_SERVER = "import sys\nfor line in sys.stdin:\n    pass\n"


def test_mcp_server_that_never_answers_fails_within_the_init_timeout(
        tmp_path, monkeypatch):
    import time
    from localm.plugins.coder import mcp
    monkeypatch.setattr(mcp, "_INIT_TIMEOUT", 1)
    script = tmp_path / "hung.py"
    script.write_text(_HUNG_MCP_SERVER, encoding="utf-8")
    _write_mcp_config(tmp_path, {
        "hung": {"command": sys.executable, "args": [str(script)]}})
    started = time.monotonic()
    got = doctor.check_mcp(tmp_path)
    assert time.monotonic() - started < 10
    assert [c.status for c in got] == [FAIL]
    assert got[0].name == "mcp:hung"
