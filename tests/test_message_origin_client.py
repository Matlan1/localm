# SPDX-License-Identifier: AGPL-3.0-or-later
"""A user-role row the client marks ``origin: "client"`` is not the user's words.

The GUI compacts a long conversation by asking /v1/chat/completions for a
summary: one role "user" message holding "Summarise the following conversation
..." plus an excerpt of the older turns (USER:/ASSISTANT:/WEB: rows). Nobody
typed that text, so the GUI marks it ``origin: "client"``, and the server leaves
a marked row out of the two places that treat user-role text as something the
user said:

- the long-term-memory recall query (``_recall_query``). Unmarked, the prompt's
  opening words and the start of the excerpt recall (and reinforce) facts for a
  question nobody asked;
- the audit/session log's user line (``_last_user_text``). Unmarked, the whole
  prompt is logged as if typed, and memory consolidation later re-learns the
  earlier turns from it.

A client that omits the field gets exactly the previous behaviour, so every test
below also runs the same request with the marker stripped and pins the old
result.
"""

from __future__ import annotations

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

# The GUI's compaction request (compactConversation in static/app/chat.js).
EXCERPT = ("USER: help me fix my rust cargo build\n\n"
           "ASSISTANT: Run cargo clean, then build again.\n\n"
           "WEB: [Results of web_search \"cargo build cache\"] (page-backed)")
SUMMARISE = ("Summarise the following conversation in under 200 words. "
             "Keep facts, names, decisions, and anything the user asked to "
             "remember. Reply with the summary only.\n\n" + EXCERPT)


def _summarise_request():
    return [{"role": "user", "content": SUMMARISE, "origin": "client"}]


def _unmarked(messages):
    """The same request as a client that does not know the field sends it."""
    return [{k: v for k, v in m.items() if k != "origin"} for m in messages]


# --------------------------------------------------------------------------- #
#  Wire: the marker survives the request model into the plain-dict messages    #
# --------------------------------------------------------------------------- #

def test_the_marker_reaches_the_dicts_and_an_unmarked_row_is_untouched():
    back = _protocol_messages_to_dicts(ChatRequest(model="m", messages=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": SUMMARISE, "origin": "client"},
        {"role": "user", "content": [{"type": "text", "text": SUMMARISE}],
         "origin": "client"},
    ]).messages)
    assert back[2] == {"role": "user", "content": SUMMARISE, "origin": "client"}
    assert back[3]["origin"] == "client", "a multimodal-shaped row keeps it too"
    # An unmarked row is exactly the dict it always was: no key at all.
    assert back[0] == {"role": "user", "content": "hello"}
    assert back[1] == {"role": "assistant", "content": "a"}
    # The field is request-only: a response message never grows an "origin" key.
    assert "origin" not in Message(role="assistant", content="x").model_dump()


def test_only_a_known_origin_is_accepted():
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[
            {"role": "user", "content": "x", "origin": "user"}])
    # Omitted and explicit null both mean "typed by the user", as before.
    back = _protocol_messages_to_dicts(ChatRequest(model="m", messages=[
        {"role": "user", "content": "x"},
        {"role": "user", "content": "y", "origin": None}]).messages)
    assert back == [{"role": "user", "content": "x"},
                    {"role": "user", "content": "y"}]


# --------------------------------------------------------------------------- #
#  Memory recall query                                                          #
# --------------------------------------------------------------------------- #

def test_a_marked_summarise_prompt_gives_an_empty_recall_query():
    # Empty query: _memory_inlet returns before recall, so nothing is injected
    # and nothing is reinforced.
    assert _recall_query(_summarise_request()) == ""
    # Unmarked, the prompt's head plus the start of the excerpt is the query,
    # exactly as before.
    assert _recall_query(_unmarked(_summarise_request())) == SUMMARISE[:400].strip()


def test_a_marked_row_is_skipped_but_the_user_s_own_turns_still_query():
    """The mark removes that one row; it does not blank the whole query, and it
    does not use up a slot in the window of recent user turns."""
    messages = [
        {"role": "user", "content": "tell me about the Vim editor"},
        {"role": "assistant", "content": "Vim is a modal text editor."},
        {"role": "user", "content": "what is its latest release"},
        {"role": "user", "content": SUMMARISE, "origin": "client"},
    ]
    assert _recall_query(messages, max_user_turns=2) == \
        "what is its latest release\ntell me about the Vim editor"
    assert _recall_query(_unmarked(messages), max_user_turns=2) == \
        (SUMMARISE + "\nwhat is its latest release")[:400].strip()


# --------------------------------------------------------------------------- #
#  Audit / session log user line                                                #
# --------------------------------------------------------------------------- #

def test_a_marked_summarise_prompt_is_not_the_audit_user_line():
    assert _last_user_text(_summarise_request()) == ""
    audit, transcript = MagicMock(), MagicMock()
    _audit_exchange(audit, transcript, _summarise_request(), "A summary.")
    audit.user.assert_called_once_with("")
    transcript.exchange.assert_called_once_with("", "A summary.")

    # Unmarked, the whole prompt is logged as the user's line, exactly as before.
    audit, transcript = MagicMock(), MagicMock()
    _audit_exchange(audit, transcript, _unmarked(_summarise_request()), "A summary.")
    audit.user.assert_called_once_with(SUMMARISE)
    transcript.exchange.assert_called_once_with(SUMMARISE, "A summary.")


def test_the_audit_user_line_is_the_newest_unmarked_row():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": [{"type": "text", "text": SUMMARISE}],
         "origin": "client"},
    ]
    assert _last_user_text(messages) == "hello"
    assert _last_user_text(_unmarked(messages)) == SUMMARISE


# --------------------------------------------------------------------------- #
#  End to end through /v1/chat/completions                                      #
# --------------------------------------------------------------------------- #

def _engine(captured: dict):
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        captured["messages"] = [dict(m) for m in messages]
        yield "A summary."

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


def test_route_injects_no_memory_into_a_marked_summarise_request(isolated_home):
    """A fact sharing words with the excerpt ("rust", "cargo") is neither injected
    into nor reinforced by the marked request; unmarked, it is both."""
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

    def _uses(c):
        items = c.get("/api/memory", headers=hdr).json()["items"]
        return [m["uses"] for m in items]

    with TestClient(app) as c:
        assert c.post("/api/memory/append", headers=hdr,
                      json={"text": "User prefers Rust and cargo"}).status_code == 200
        before = _uses(c)
        assert c.post("/v1/chat/completions", json={
            "model": "m", "messages": _summarise_request()}).status_code == 200
        assert "Rust and cargo" not in _system_text()
        assert _uses(c) == before, "a marked request reinforces nothing"

        assert c.post("/v1/chat/completions", json={
            "model": "m",
            "messages": _unmarked(_summarise_request())}).status_code == 200
        assert "Rust and cargo" in _system_text()
        assert _uses(c) != before


def test_route_does_not_audit_the_summarise_prompt_as_the_user_s_line(
        isolated_home, monkeypatch):
    import localm.inference.http_server as hs
    recorded = []
    real = hs._audit_exchange

    def _spy(audit, transcript, messages, reply, outcome="success"):
        spy = types.SimpleNamespace(user=recorded.append, llm=lambda *_: None,
                                    notice=lambda *_: None)
        real(spy, None, messages, reply, outcome=outcome)

    monkeypatch.setattr(hs, "_audit_exchange", _spy)
    app = hs.create_app(_engine({}))
    body = {"model": "m", "stream": False, "messages": _summarise_request()}
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=body).status_code == 200
        body["messages"] = _unmarked(body["messages"])
        assert c.post("/v1/chat/completions", json=body).status_code == 200
    assert recorded == ["", SUMMARISE]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
