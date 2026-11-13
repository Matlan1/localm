# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run MODEL` attached to a running server, on a terminal: when the
server refuses a turn and its routing names a model that could answer it
(``suggested`` with ``ask_reason`` in ``X-Localm-Model-Routing``, or in the
``localm_error`` of a stream it opened early), the user is asked the same
question the in-process path asks, with the server's reason; a yes sends the
turn again naming that model, a no prints the refusal and keeps MODEL.
Without a terminal, or with --pin-model, nothing is asked.

A loopback stand-in answers the real HTTP requests the real HttpEngine sends;
only attach discovery and the terminal check are stubbed."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli import chat as chat_mod
from localm.inference.backends.base import UnsupportedInputError
from localm.inference.http_engine import HttpEngine

REASON = "it would run on the CPU only, because no layer fits on the GPU"
REFUSAL = ("This model cannot accept image input, so the attached image would be "
           "ignored. A vision-capable model is already in your library: seer.")


class _Server:
    """Serves ``plain`` as the active model. A chat request naming ``plain``
    is refused per *mode*: ``"image"`` (400 with the routing header, image
    turns only), ``"stream"`` (the same refusal inside an early-opened stream)
    or ``"context"`` (413 for every turn). A request naming ``seer`` streams
    "a cat" as seer. Unpinned refusals suggest ``seer``; pinned ones do not."""

    def __init__(self, mode: str):
        self.mode = mode
        self.bodies: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, headers=None, ctype="application/json"):
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [
                        {"id": "plain", "loaded": True, "active": True}]})
                else:
                    self._send(404, {"detail": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path != "/v1/chat/completions":
                    self._send(404, {"detail": "not found"})
                    return
                outer.bodies.append(body)
                self._chat(body)

            def _chat(self, body):
                model = body.get("model")
                if model == "seer":
                    self._sse([{"model": "seer", "choices": [{"delta": {"content": "a cat"}}]}])
                    return
                has_image = "image_url" in json.dumps(body.get("messages"))
                if outer.mode in ("image", "stream") and not has_image:
                    self._sse([{"model": "plain", "choices": [{"delta": {"content": "plain"}}]}])
                    return
                gap = "context_length" if outer.mode == "context" else "vision"
                routing = {"resolved": "plain", "routed": False,
                           "pinned": body.get("pin_model") is not False,
                           "gaps": {gap: "absent"}, "unmet": [gap]}
                if not routing["pinned"]:
                    routing.update(suggested="seer", ask_reason=REASON, fit="cpu_only")
                if outer.mode == "context":
                    self._send(413, {"detail": "Prompt (9000 tokens) exceeds the 4096-token "
                                               "context window this model was trained for."},
                               {"X-Localm-Model-Routing": json.dumps(routing)})
                elif outer.mode == "image":
                    self._send(400, {"detail": REFUSAL},
                               {"X-Localm-Refusal": "image",
                                "X-Localm-Model-Routing": json.dumps(routing)})
                else:
                    err = {"status": 400, "detail": REFUSAL, "refusal": "image",
                           "routing": routing}
                    self._sse([{"model": "plain", "localm_error": err,
                                "choices": [{"delta": {"content": REFUSAL}}]},
                               {"model": "plain", "localm_error": err,
                                "choices": [{"delta": {}, "finish_reason": "error"}]}])

            def _sse(self, chunks):
                raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                self._send(200, raw.encode(), ctype="text/event-stream")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def reg(tmp_path, monkeypatch):
    for n in ("plain", "seer"):
        (tmp_path / f"{n}.gguf").write_bytes(b"GGUF")
    proj = tmp_path / "seer-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    registry = {
        "plain": {"path": str(tmp_path / "plain.gguf"), "source": "local",
                  "model_type": "llm", "context_length": 4096},
        "seer": {"path": str(tmp_path / "seer.gguf"), "source": "local",
                 "model_type": "llm", "context_length": 8192, "mmproj": str(proj)},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return img


@pytest.fixture
def attached(monkeypatch, reg):
    """Start a stand-in in the given mode, attach `localm run` to it, and
    answer the terminal question with *answer* (None: no terminal). Returns
    the server and the list of questions asked."""
    holder = {}
    started = []
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.audit.effective_mode", lambda *a, **k: SessionMode.LOG)
    monkeypatch.setattr("subprocess.Popen",
                        lambda *a, **k: started.append(a) or (_ for _ in ()).throw(
                            OSError("a background server was started")))

    def make(mode, answer):
        srv = _Server(mode)
        holder["srv"] = srv
        monkeypatch.setattr("localm.instances.attach_target",
                            lambda *a, **k: {"base_url": srv.base_url, "token": None})
        asked = []
        monkeypatch.setattr(chat_mod, "_terminal", lambda: answer is not None)

        def confirm(prompt, **_):
            asked.append(" ".join(str(prompt).split()))
            return bool(answer)
        monkeypatch.setattr("rich.prompt.Confirm.ask", confirm)
        return srv, asked

    yield make
    if "srv" in holder:
        holder["srv"].close()
    assert started == [], "a background server was started"


def _flat(output: str) -> str:
    return " ".join(output.split())


QUESTION = f"plain lacks vision. Use seer for this turn ({REASON})?"


def test_a_yes_answers_the_turn_with_the_suggested_model(attached, reg):
    srv, asked = attached("image", True)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "what is this?", "--image", str(reg)])
    out = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert asked == [QUESTION]
    assert [b["model"] for b in srv.bodies] == ["plain", "seer"]
    assert srv.bodies[0].get("pin_model") is False
    assert "pin_model" not in srv.bodies[1], "the turn sent again names seer, pinned"
    assert "image_url" in json.dumps(srv.bodies[1]["messages"]), "the same turn is sent again"
    assert "a cat" in out
    assert "answered by seer: plain lacks reading images" in out
    assert REFUSAL not in out


def test_a_no_keeps_the_model_and_prints_the_refusal(attached, reg):
    srv, asked = attached("image", False)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "what is this?", "--image", str(reg)])
    assert result.exit_code == 0, result.output
    assert asked == [QUESTION]
    assert [b["model"] for b in srv.bodies] == ["plain"]
    assert "a cat" not in result.output
    assert "seer" in _flat(result.output), "the refusal guidance is printed"


def test_without_a_terminal_nothing_is_asked(attached, reg):
    srv, asked = attached("image", None)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "what is this?", "--image", str(reg)])
    assert result.exit_code == 0, result.output
    assert asked == []
    assert [b["model"] for b in srv.bodies] == ["plain"]


def test_pin_model_asks_nothing(attached, reg):
    srv, asked = attached("image", True)
    result = CliRunner().invoke(
        chat_mod.run, ["plain", "--pin-model", "-p", "what is this?", "--image", str(reg)])
    assert result.exit_code == 0, result.output
    assert asked == []
    assert [b["model"] for b in srv.bodies] == ["plain"]


def test_a_refusal_inside_an_early_opened_stream_is_asked_too(attached, reg):
    srv, asked = attached("stream", True)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "what is this?", "--image", str(reg)])
    assert result.exit_code == 0, result.output
    assert asked == [QUESTION]
    assert [b["model"] for b in srv.bodies] == ["plain", "seer"]
    assert "a cat" in result.output


def test_a_context_refusal_is_asked_with_its_gap(attached, reg):
    srv, asked = attached("context", True)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "a long question"])
    assert result.exit_code == 0, result.output
    assert asked == [f"plain lacks context_length. Use seer for this turn ({REASON})?"]
    assert [b["model"] for b in srv.bodies] == ["plain", "seer"]
    assert "a cat" in result.output


def test_a_context_refusal_answered_no_exits_with_the_server_error(attached, reg):
    srv, asked = attached("context", False)
    result = CliRunner().invoke(chat_mod.run, ["plain", "-p", "a long question"])
    assert result.exit_code == 1
    assert len(asked) == 1
    assert "context window this model was trained for" in _flat(result.output)
    assert [b["model"] for b in srv.bodies] == ["plain"]


def test_the_interactive_chat_asks_and_keeps_the_turn(attached, reg, monkeypatch):
    """In the REPL a yes answers the image turn with seer, and the turn stays in
    the conversation; the next turn is sent to plain again, whose refusal of
    the image still in the conversation is asked about again."""
    srv, asked = attached("image", True)
    engine = HttpEngine(srv.base_url, model="plain", display_name="plain", pin_model=False)
    router = chat_mod._TurnRouter(engine, "plain", pinned=False)
    router._confirm = router.ask
    inputs = iter([f"/image {reg}", "what is this?", "thanks", KeyboardInterrupt()])

    def fake_input(_prompt):
        v = next(inputs)
        if isinstance(v, BaseException):
            raise v
        return v
    monkeypatch.setattr(chat_mod.console, "input", fake_input)
    chat_mod._interactive(engine, None, {}, router=router)
    assert asked == [QUESTION, QUESTION]
    assert [b["model"] for b in srv.bodies] == ["plain", "seer", "plain", "seer"]
    assert srv.bodies[3]["messages"][-2] == {"role": "assistant", "content": "a cat"}
    assert srv.bodies[3]["messages"][-1] == {"role": "user", "content": "thanks"}


def test_the_refusal_routing_is_kept_only_for_the_request_that_carried_it(attached, reg):
    srv, _ = attached("image", None)
    engine = HttpEngine(srv.base_url, model="plain", pin_model=False)
    image_turn = [chat_mod._build_user_message("what is this?", [str(reg)])]
    with pytest.raises(UnsupportedInputError):
        list(engine.chat_stream(image_turn))
    assert engine.refusal_routing["suggested"] == "seer"
    assert list(engine.chat_stream([{"role": "user", "content": "hi"}])) == ["plain"]
    assert engine.refusal_routing is None
