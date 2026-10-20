# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder doctor` and `--output-format jsonl`, driven through the real
click command: doctor against a loopback server, jsonl against a real Agent with
a scripted backend."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

from localm.plugins.coder import display
from localm.plugins.coder.cli import _main


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """An active coder plugin under a throwaway HOME, and an empty project."""
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    p = tmp_path / "proj"
    p.mkdir()
    return p


def _invoke(*args):
    return CliRunner().invoke(_main.main, list(args))


# ---------------------------------------------------------------- doctor

@pytest.fixture
def models_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                     # noqa: N802
            body = json.dumps({"data": [{"id": "m1"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_doctor_json_reports_every_check_and_exits_zero(proj, models_server):
    res = _invoke("doctor", "-m", "m1", "-u", models_server, "--cwd", str(proj),
                  "--output-format", "json")
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["ok"] is True
    assert [c["name"] for c in data["checks"]][:3] == [
        "model", "tool-calls", "context"]


def test_doctor_text_is_the_default_format(proj, models_server):
    res = _invoke("doctor", "-m", "m1", "-u", models_server, "--cwd", str(proj))
    assert res.exit_code == 0, res.output
    assert res.stdout.splitlines()[0].startswith("[ok  ] model")


def test_doctor_exits_one_when_the_server_is_unreachable(proj):
    dead = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = dead.server_address[1]
    dead.server_close()
    res = _invoke("doctor", "-m", "m1", "-u", f"http://127.0.0.1:{port}/v1",
                  "--cwd", str(proj))
    assert res.exit_code == 1
    assert "[FAIL] model" in res.stdout


def test_doctor_never_starts_a_model_server(proj, monkeypatch):
    spawned = []
    real_popen = subprocess.Popen

    class Spy(real_popen):
        def __init__(self, args, *a, **k):
            spawned.append(args)
            super().__init__(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", Spy)
    res = _invoke("doctor", "-m", "m1", "--cwd", str(proj),
                  "--output-format", "json")
    assert res.exit_code == 1, res.output
    assert json.loads(res.stdout)["checks"][0]["status"] == "fail"
    assert spawned, "the spawn spy saw nothing, so it proves nothing"
    assert not [a for a in spawned if "gui" in map(str, a)]


def test_doctor_runs_no_agent(proj, models_server, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("doctor built an agent")

    monkeypatch.setattr(_main, "build_agent", boom)
    res = _invoke("doctor", "-m", "m1", "-u", models_server, "--cwd", str(proj))
    assert res.exit_code == 0, res.output


def test_doctor_reports_an_unreadable_project_config_instead_of_exiting(
        proj, models_server):
    cfg = proj / ".localcoder"
    cfg.mkdir()
    (cfg / "config.toml").write_text("not = = toml", encoding="utf-8")
    res = _invoke("doctor", "-m", "m1", "-u", models_server, "--cwd", str(proj),
                  "--output-format", "json")
    assert res.exit_code == 1
    by_name = {c["name"]: c for c in json.loads(res.stdout)["checks"]}
    assert by_name["config"]["status"] == "fail"
    assert by_name["model"]["status"] == "ok"


def test_doctor_jsonl_ends_with_a_summary_line(proj, models_server):
    res = _invoke("doctor", "-m", "m1", "-u", models_server, "--cwd", str(proj),
                  "--output-format", "jsonl")
    rows = [json.loads(ln) for ln in res.stdout.splitlines()]
    assert rows[-1]["type"] == "summary" and rows[-1]["ok"] is True


# ---------------------------------------------------------------- jsonl

class _ScriptedBackend:
    model_id = "stub-model"
    native_tools = False
    supports_grammar = False

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply

    def chat_stream(self, messages, **kw):
        return iter([self.chat(messages, **kw)])


def _write_call(**args):
    return "<tool_call>\n" + json.dumps({"name": "write_file", "args": args}) \
        + "\n</tool_call>"


@pytest.fixture
def scripted(monkeypatch):
    def install(replies):
        backend = _ScriptedBackend(replies)
        monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
        return backend
    return install


def test_jsonl_run_streams_events_then_a_final_record(proj, scripted):
    scripted([_write_call(path="NOTE.txt", content="hi\n"), "All done."])
    res = _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
                  "--output-format", "jsonl", "write NOTE.txt")
    assert res.exit_code == 0, res.output + res.stderr
    rows = [json.loads(ln) for ln in res.stdout.splitlines()]
    assert (proj / "NOTE.txt").read_text(encoding="utf-8") == "hi\n"
    types = [r["type"] for r in rows]
    assert "tool_call" in types and "tool_result" in types
    call = next(r for r in rows if r["type"] == "tool_call")
    assert call["tool"] == "write_file" and call["args"]["path"] == "NOTE.txt"
    final = rows[-1]
    assert final["type"] == "final" and final["success"] is True
    assert final["response"] and final["turns"] >= 1 and final["denied"] == []
    assert types.count("final") == 1


def test_jsonl_stdout_holds_nothing_but_json_lines(proj, scripted):
    scripted(["Nothing to do."])
    res = _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
                  "--output-format", "jsonl", "say hi")
    assert res.exit_code == 0, res.output + res.stderr
    assert res.stdout.strip()
    for line in res.stdout.splitlines():
        assert isinstance(json.loads(line), dict), line


def test_jsonl_routes_the_clis_own_messages_to_stderr(proj, scripted):
    scripted(["Nothing to do."])
    res = _invoke("-m", "stub-model", "--cwd", str(proj),
                  "--output-format", "jsonl", "say hi")
    assert "Unattended one-shot" in res.stderr
    assert "Unattended one-shot" not in res.stdout


def test_jsonl_reports_a_denied_call_in_the_final_record(proj, scripted):
    shell = ("<tool_call>\n"
             + json.dumps({"name": "run_shell", "args": {"command": "echo hi"}})
             + "\n</tool_call>")
    scripted([shell, "gave up"])
    res = _invoke("-m", "stub-model", "--cwd", str(proj),
                  "--output-format", "jsonl", "run echo")
    rows = [json.loads(ln) for ln in res.stdout.splitlines()]
    final = rows[-1]
    assert final["type"] == "final"
    assert final["success"] is False
    assert final["denied"] and final["denied"][0]["tool"] == "run_shell"


def test_jsonl_needs_a_task(proj, scripted):
    res = _invoke("-m", "stub-model", "--cwd", str(proj),
                  "--output-format", "jsonl")
    assert res.exit_code == 1
    assert res.stdout == ""
    assert "needs a TASK" in res.stdout + res.stderr + res.output


def test_jsonl_run_closes_the_session_in_the_foreground(proj, scripted,
                                                        monkeypatch):
    built = []
    real = _main.build_agent

    def spy(*a, **k):
        agent = real(*a, **k)
        built.append(agent)
        return agent

    monkeypatch.setattr(_main, "build_agent", spy)
    scripted(["ok"])
    _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
            "--output-format", "jsonl", "say hi")
    assert built and built[0].reflect_in_background is False


def test_jsonl_does_not_leave_the_console_pointed_at_stderr(proj, scripted):
    scripted(["ok"])
    _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
            "--output-format", "jsonl", "say hi")
    assert display.console.file is sys.stdout


def test_jsonl_estimate_is_one_estimate_record(proj, scripted, monkeypatch):
    scripted(["unused"])
    monkeypatch.setattr(
        "localm.plugins.coder.cli.estimate.estimate_task",
        lambda agent, task: {"estimate": "plan", "total_tokens": 3})
    res = _invoke("-m", "stub-model", "--cwd", str(proj), "--estimate",
                  "--output-format", "jsonl", "do x")
    assert res.exit_code == 0, res.output + res.stderr
    rows = [json.loads(ln) for ln in res.stdout.splitlines()]
    assert rows == [{"type": "estimate", "estimate": "plan", "total_tokens": 3}]


def test_json_stdout_is_exactly_one_parseable_document(proj, scripted):
    scripted(["Nothing to do."])
    res = _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
                  "--output-format", "json", "say hi")
    assert res.exit_code == 0, res.output + res.stderr
    doc = json.loads(res.stdout)
    assert set(doc) == {"success", "response", "turns", "total_tokens", "denied"}
    assert doc["response"].startswith("Nothing to do.")


def test_json_run_reply_text_goes_to_stderr_not_stdout(proj, scripted):
    scripted(["A distinctive final reply."])
    res = _invoke("-m", "stub-model", "--cwd", str(proj), "--yes",
                  "--output-format", "json", "say hi")
    assert "A distinctive final reply." in res.stderr
    assert res.stdout.lstrip().startswith("{")


def test_json_without_a_task_leaves_the_repl_on_stdout(proj, scripted, monkeypatch):
    scripted(["unused"])
    seen = {}
    monkeypatch.setattr(
        _main, "_repl",
        lambda agent: seen.update(file=display.console.file, out=sys.stdout))
    res = _invoke("-m", "stub-model", "--cwd", str(proj),
                  "--output-format", "json")
    assert res.exit_code == 0, res.output + res.stderr
    assert seen["file"] is seen["out"]


# ---------------------------------------------------------------- REPL /doctor

class _SessionAgent:
    """What /doctor reads from a live session."""

    def __init__(self, cwd, backend, verify_cmd):
        self.cwd = cwd
        self.backend = backend
        self.verify_cmd = verify_cmd

        class _Map:
            def to_context_string(self):
                return "x" * 400

            def file_count(self):
                return 3

        self._project_map = _Map()


def test_repl_doctor_prints_the_report_for_the_live_session(
        tmp_path, models_server, capsys):
    from localm.plugins.coder.backends.http import HTTPBackend
    from localm.plugins.coder.cli import repl
    agent = _SessionAgent(tmp_path, HTTPBackend(models_server, "m1"),
                          f'"{sys.executable}" -c pass')
    assert repl._handle_command("/doctor", agent) is False
    out = capsys.readouterr().out
    assert "[ok  ] model" in out
    assert "repo map is ~100 tokens (3 files)" in out
    assert "resolves" in out


def test_repl_doctor_says_so_when_the_session_has_verification_off(
        tmp_path, models_server, capsys):
    from localm.plugins.coder.backends.http import HTTPBackend
    from localm.plugins.coder.cli import repl
    agent = _SessionAgent(tmp_path, HTTPBackend(models_server, "m1"), None)
    repl._handle_command("/doctor", agent)
    assert "verification is off for this session" in capsys.readouterr().out


def test_doctor_is_a_listed_slash_command_and_in_the_help():
    from localm.plugins.coder.cli import repl
    assert "/doctor" in repl._SLASH_COMMANDS
    assert "/doctor" in display.HELP_TEXT
