# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the model is doing reaches the user on the CLI (`localm run`), through
the attach client, and to an MCP client that asked for progress."""

import io
import json
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.base import (
    LOADING_MODEL_STATUS, UnsupportedInputError, VISION_CPU_FALLBACK_STATUS,
)
from localm.inference.http_engine import HttpEngine


def _sse(chunks):
    lines = ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]
    r = MagicMock()
    r.status_code = 200
    r.iter_lines = lambda decode_unicode=False: iter(lines)
    return r


def _refusal(code, detail):
    """The chunks the server ends an early-opened stream with."""
    from localm.inference.http_server import _prep_error_lines
    return [json.loads(line[6:]) for line in _prep_error_lines(detail, code, "m", "id", 0)
            if line.startswith("data: {")]


def _drain(gen, yielded):
    for piece in gen:
        yielded.append(piece)


class TestHttpEngineInStreamRefusal:
    def test_an_image_refusal_raises_unsupported_input(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _sse(
            _refusal(400, "This model cannot accept image input (text-only).")))
        yielded = []
        with pytest.raises(UnsupportedInputError):
            _drain(HttpEngine("http://x/v1").chat_stream(
                [{"role": "user", "content": "x"}]), yielded)
        assert yielded == [], "the refusal must not be shown as reply text"

    def test_another_refusal_raises_with_its_status_and_detail(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: _sse(
            _refusal(503, "Model load was cancelled: superseded")))
        yielded = []
        with pytest.raises(RuntimeError) as caught:
            _drain(HttpEngine("http://x/v1").chat_stream(
                [{"role": "user", "content": "x"}]), yielded)
        assert yielded == []
        assert str(caught.value) == ("server error (HTTP 503): "
                                     "Model load was cancelled: superseded")


class TestCliStatusLines:
    @pytest.fixture
    def captured(self, monkeypatch):
        from rich.console import Console
        import localm.cli.chat as cli_chat
        buf = io.StringIO()
        monkeypatch.setattr(cli_chat, "err_console",
                            Console(file=buf, force_terminal=False, width=200))
        return buf

    def test_each_new_status_prints_once_until_the_first_token(self, captured):
        from localm.cli.chat import _StatusLines
        order = []
        lines = _StatusLines(before_first=lambda: order.append("label"))
        lines.status(LOADING_MODEL_STATUS)
        lines.status("Processing prompt...")
        lines.status("Processing prompt...")
        lines.first_token()
        lines.first_token()
        lines.status("Generating response...")
        assert captured.getvalue().splitlines() == [LOADING_MODEL_STATUS,
                                                     "Processing prompt..."]
        assert order == ["label"]

    def test_the_vision_fallback_warning_always_prints(self, captured):
        from localm.cli.chat import _StatusLines
        lines = _StatusLines()
        lines.first_token()
        lines.status(VISION_CPU_FALLBACK_STATUS)
        assert VISION_CPU_FALLBACK_STATUS in captured.getvalue()

    def test_one_shot_run_prints_statuses_before_the_reply(self, captured, monkeypatch, capsys):
        from localm.cli.chat import _stream_once
        engine = MagicMock()

        def _chat_stream(messages, on_status=None, **kw):
            on_status(LOADING_MODEL_STATUS)
            yield "hi"

        engine.chat_stream.side_effect = _chat_stream
        engine.count_tokens.return_value = 1
        assert _stream_once(engine, [{"role": "user", "content": "x"}]) == "hi"
        assert LOADING_MODEL_STATUS in captured.getvalue()
        assert "hi" in capsys.readouterr().out


class TestMcpProgress:
    def _server(self, handler):
        from localm.plugins.mcpserver.server import MCPStdioServer
        return MCPStdioServer({"slow": {"description": "", "inputSchema": {},
                                        "handler": handler}})

    def _call(self, server, meta, arguments=None):
        params = {"name": "slow", "arguments": arguments or {}}
        if meta is not None:
            params["_meta"] = meta
        req = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                          "params": params}) + "\n"
        out = io.StringIO()
        server.run_stdio(stdin=io.StringIO(req), stdout=out)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def _handler(self, args):
        from localm.plugins.mcpserver.server import _text_result, report_progress
        report_progress(LOADING_MODEL_STATUS)
        report_progress(LOADING_MODEL_STATUS)
        report_progress("Processing prompt...")
        return _text_result("done")

    def test_a_call_with_a_progress_token_gets_one_notification_per_status(self):
        msgs = self._call(self._server(self._handler), {"progressToken": "tok-1"})
        notes = [m for m in msgs if m.get("method") == "notifications/progress"]
        assert [n["params"] for n in notes] == [
            {"progressToken": "tok-1", "progress": 1, "message": LOADING_MODEL_STATUS},
            {"progressToken": "tok-1", "progress": 2, "message": "Processing prompt..."},
        ]
        assert msgs[-1]["id"] == 7 and "result" in msgs[-1]

    @pytest.mark.parametrize("meta", [None, {}, {"progressToken": True},
                                      {"progressToken": {"x": 1}}])
    def test_a_call_without_a_usable_token_gets_no_notification(self, meta):
        msgs = self._call(self._server(self._handler), meta)
        assert [m.get("id") for m in msgs] == [7]

    def test_progress_outside_a_call_is_a_no_op(self):
        from localm.plugins.mcpserver.server import report_progress
        report_progress(LOADING_MODEL_STATUS)

    def test_the_chat_tool_passes_engine_statuses_as_progress(self, monkeypatch):
        import sys
        import threading
        from localm.plugins.mcpserver.tools import chat as chat_tool
        live = sys.modules["localm.plugins.mcpserver.server"]
        assert chat_tool._srv is live, (
            "the chat tool reads its progress sink from a different copy of "
            "localm.plugins.mcpserver.server than the one serving the call "
            f"(tool: id {id(chat_tool._srv.__dict__)}, live: id {id(live.__dict__)}); "
            "an earlier test in this process removed or re-imported the module")
        engine = MagicMock()

        def _chat_stream(messages, on_status=None, **kw):
            on_status("Processing prompt...")
            yield "answer"

        engine.chat_stream.side_effect = _chat_stream
        engines = MagicMock()
        engines.route.return_value = MagicMock(routed=False, current="m", resolved="m",
                                               pinned=True, skipped=(), load_errors=())
        engines.get_loaded_chat.return_value = engine
        engines.is_peer.return_value = False
        tools = chat_tool.build(engines)
        msgs = self._call(self._server(tools["chat"]["handler"]),
                          {"progressToken": 3}, {"prompt": "hi"})
        notes = [m["params"]["message"] for m in msgs
                 if m.get("method") == "notifications/progress"]
        others = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
        assert notes == ["Processing prompt..."], (
            f"sink after the call: {live._progress_sink!r}; other threads alive: {others}")
        assert msgs[-1]["result"]["content"] == [{"type": "text", "text": "answer"}]


_FIRST_IMPORT_UNDER_PATCH = """
import io, json, sys
from unittest.mock import MagicMock
import localm.plugins.mcpserver.server as srv

real = srv.report_progress
patched = []
srv.report_progress = patched.append
from localm.plugins.mcpserver.tools import chat as chat_tool
srv.report_progress = real

engine = MagicMock()
def _stream(messages, on_status=None, **kw):
    on_status("Processing prompt...")
    yield "answer"
engine.chat_stream.side_effect = _stream
engines = MagicMock()
engines.route.return_value = MagicMock(routed=False, current="m", resolved="m",
                                       pinned=True, skipped=(), load_errors=())
engines.get_loaded_chat.return_value = engine
engines.is_peer.return_value = False
server = srv.MCPStdioServer({"chat": chat_tool.build(engines)["chat"]})
req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "chat", "arguments": {"prompt": "hi"},
                             "_meta": {"progressToken": 1}}}) + "\\n"
out = io.StringIO()
server.run_stdio(stdin=io.StringIO(req), stdout=out)
notes = [json.loads(l)["params"]["message"] for l in out.getvalue().splitlines()
         if json.loads(l).get("method") == "notifications/progress"]
print(json.dumps({"notes": notes, "patched": patched}))
"""


class TestMcpChatToolProgressSurvivesAPatchedFirstImport:
    def test_a_progress_function_patched_while_the_chat_tool_is_first_imported_is_not_kept(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        repo = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": repo + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run([sys.executable, "-c", _FIRST_IMPORT_UNDER_PATCH],
                              capture_output=True, text=True, timeout=120, env=env)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result == {"notes": ["Processing prompt..."], "patched": []}, result
