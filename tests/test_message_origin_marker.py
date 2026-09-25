# SPDX-License-Identifier: AGPL-3.0-or-later
"""A user-role row the client marks ``origin: "tool"`` is not the user's words.

The GUI sends each web tool event (a web_search or fetch_url result, or a
control note such as "[pending action] ...") as a role "user" message so that
strict chat templates keep user/assistant alternation, and marks that row
``origin: "tool"``. The server carries the marker into the plain-dict messages
and leaves a marked row out of the two places that treat user-role text as
something the user said:

- the long-term-memory recall query (``_recall_query``), where a note or a
  fetched page full of common words ("answer", "tool", "results", "source")
  otherwise lets the word-overlap gate admit loosely related facts;
- the audit/session log's user line (``_last_user_text``), which memory
  consolidation later learns facts from.

A client that omits the field gets exactly the previous behaviour, so every
test below also runs the same conversation with the marker stripped and pins
the old result.
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from localm.inference.http_server import (
    _audit_exchange, _last_user_text, _protocol_messages_to_dicts,
)
from localm.inference.protocol import ChatRequest, Message
from localm.plugins.builtin.memory.plug import _recall_query

QUESTION = "what is the tallest building in Oslo"
RESULTS = ('[Results of web_search "tallest building Oslo"] (page-backed)\n'
           "<untrusted_content>[S1] Oslo towers - source answer results"
           "</untrusted_content>")
PENDING = ("[pending action] Your reply announced a web lookup but made no "
           "tool call. Answer now, or emit the tool call in the required format.")


def _unmarked(messages):
    """The same conversation as a client that does not know the field sends it."""
    return [{k: v for k, v in m.items() if k != "origin"} for m in messages]


def _web_turn():
    """A question, the model's tool call, the GUI's rendered result, the model's
    announcement, and the GUI's repair note: the shape of one web round."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": QUESTION},
        {"role": "assistant",
         "content": '<tool_call>{"name": "web_search", "args": '
                    '{"query": "tallest building Oslo"}}</tool_call>'},
        {"role": "user", "content": RESULTS, "origin": "tool"},
        {"role": "assistant", "content": "I will look that up and report back."},
        {"role": "user", "content": PENDING, "origin": "tool"},
    ]


# --------------------------------------------------------------------------- #
#  Wire: the marker survives the request model into the plain-dict messages    #
# --------------------------------------------------------------------------- #

def test_the_marker_reaches_the_dicts_and_an_unmarked_row_is_untouched():
    wire = json.loads(json.dumps({"model": "m", "messages": [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": RESULTS, "origin": "tool"},
        {"role": "user", "content": [{"type": "text", "text": PENDING}],
         "origin": "tool"},
    ]}))
    back = _protocol_messages_to_dicts(ChatRequest(**wire).messages)
    assert back[2]["origin"] == "tool"
    assert back[3]["origin"] == "tool", "a multimodal-shaped row keeps it too"
    # An unmarked row is byte-for-byte the dict it always was: no key at all.
    assert back[0] == {"role": "user", "content": QUESTION}
    assert back[1] == {"role": "assistant", "content": "a"}
    # The field is request-only: a response message never grows an "origin" key.
    assert "origin" not in Message(role="assistant", content="x").model_dump()


def test_only_the_known_origin_is_accepted():
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[
            {"role": "user", "content": "x", "origin": "assistant"}])
    # Omitted and explicit null both mean "typed by the user", as before.
    back = _protocol_messages_to_dicts(ChatRequest(model="m", messages=[
        {"role": "user", "content": "x"},
        {"role": "user", "content": "y", "origin": None}]).messages)
    assert back == [{"role": "user", "content": "x"},
                    {"role": "user", "content": "y"}]


# --------------------------------------------------------------------------- #
#  Memory recall query                                                          #
# --------------------------------------------------------------------------- #

def test_recall_query_skips_marked_rows():
    messages = _web_turn()
    assert _recall_query(messages) == QUESTION
    # Unmarked, the same conversation queries exactly as before: newest first,
    # the notes included.
    assert _recall_query(_unmarked(messages)) == "\n".join(
        [PENDING, RESULTS, QUESTION])[:400].strip()


def test_marked_rows_do_not_use_up_the_recall_window():
    """Three tool rounds after two real turns: the window of three user turns
    still reaches both of the user's own messages."""
    messages = [
        {"role": "user", "content": "tell me about the Vim editor"},
        {"role": "assistant", "content": "Vim is a modal text editor."},
        {"role": "user", "content": "yes, look up its latest release"},
    ]
    for n in range(3):
        messages += [{"role": "assistant", "content": f"<tool_call>{n}</tool_call>"},
                     {"role": "user", "content": f"[tool-call format] fix {n}",
                      "origin": "tool"}]
    q = _recall_query(messages)
    assert q == "yes, look up its latest release\ntell me about the Vim editor"
    assert "[tool-call format]" not in _recall_query(messages)
    old = _recall_query(_unmarked(messages))
    assert old == "[tool-call format] fix 2\n[tool-call format] fix 1\n" \
                  "[tool-call format] fix 0"


def test_recall_query_is_empty_when_only_marked_rows_remain():
    messages = [{"role": "user", "content": PENDING, "origin": "tool"}]
    assert _recall_query(messages) == ""
    assert _recall_query(_unmarked(messages)) == PENDING[:400].strip()


# --------------------------------------------------------------------------- #
#  Audit / session log user line                                                #
# --------------------------------------------------------------------------- #

def test_audit_user_line_skips_marked_rows():
    messages = _web_turn()
    assert _last_user_text(messages) == QUESTION
    audit, transcript = MagicMock(), MagicMock()
    _audit_exchange(audit, transcript, messages, "It is the Oslo tower.")
    audit.user.assert_called_once_with(QUESTION)
    transcript.exchange.assert_called_once_with(QUESTION, "It is the Oslo tower.")

    # Unmarked, the newest user-role row is logged, exactly as before.
    audit, transcript = MagicMock(), MagicMock()
    _audit_exchange(audit, transcript, _unmarked(messages), "It is the Oslo tower.")
    audit.user.assert_called_once_with(PENDING)
    transcript.exchange.assert_called_once_with(PENDING, "It is the Oslo tower.")


def test_audit_user_line_skips_a_marked_multimodal_row():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": QUESTION}]},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": [{"type": "text", "text": RESULTS}],
         "origin": "tool"},
    ]
    assert _last_user_text(messages) == QUESTION
    assert _last_user_text(_unmarked(messages)) == RESULTS


# --------------------------------------------------------------------------- #
#  End to end through /v1/chat/completions                                      #
# --------------------------------------------------------------------------- #

def _engine(captured: dict):
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        captured["messages"] = [dict(m) for m in messages]
        yield "ok"

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 1
    engine.count_messages_tokens.return_value = 10
    engine.display_name = "m"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: True)
    return engine


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def test_route_audits_the_user_s_words_not_the_gui_note(isolated_home, monkeypatch):
    import localm.inference.http_server as hs
    recorded = []
    real = hs._audit_exchange

    def _spy(audit, transcript, messages, reply, outcome="success"):
        spy = types.SimpleNamespace(user=recorded.append, llm=lambda *_: None,
                                    notice=lambda *_: None)
        real(spy, None, messages, reply, outcome=outcome)

    monkeypatch.setattr(hs, "_audit_exchange", _spy)
    app = hs.create_app(_engine({}))
    body = {"model": "m", "stream": False, "messages": _web_turn()}
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=body).status_code == 200
        body["messages"] = _unmarked(body["messages"])
        assert c.post("/v1/chat/completions", json=body).status_code == 200
    assert recorded == [QUESTION, PENDING]


def test_route_recalls_for_the_user_s_question_not_the_gui_note(isolated_home):
    """A fact that shares words only with the fetched page is not recalled for
    a question about something else; unmarked, the page text still pulls it in."""
    from localm.inference.http_server import create_app
    captured: dict = {}
    app = create_app(_engine(captured))
    mgr = app.state.plugin_manager
    mgr.install("memory")
    mgr.enable("memory")
    shell = getattr(app.state, "shell_token", None)
    hdr = {"Authorization": f"Bearer {shell}"} if shell else {}

    def _system_text():
        return " ".join(m.get("content") or "" for m in captured["messages"]
                        if m.get("role") == "system")

    page = ('[Content of https://example.org/build]\n<untrusted_content>'
            "fix a rust cargo build</untrusted_content>")
    messages = [
        {"role": "user", "content": "what is the capital of France"},
        {"role": "assistant", "content": '<tool_call>{"name": "fetch_url"}</tool_call>'},
        {"role": "user", "content": page, "origin": "tool"},
    ]
    with TestClient(app) as c:
        assert c.post("/api/memory/append", headers=hdr,
                      json={"text": "User prefers Rust and cargo"}).status_code == 200
        assert c.post("/v1/chat/completions",
                      json={"model": "m", "messages": messages}).status_code == 200
        assert "Rust and cargo" not in _system_text()
        assert c.post("/v1/chat/completions",
                      json={"model": "m",
                            "messages": _unmarked(messages)}).status_code == 200
        assert "Rust and cargo" in _system_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
