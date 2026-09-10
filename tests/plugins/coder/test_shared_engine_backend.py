# SPDX-License-Identifier: AGPL-3.0-or-later
"""SharedEngineBackend: the coder over an engine somebody else owns.

Pins what the live MCP run needs from it: every generation kwarg the coder
sends reaches the engine, including the lazy tool-call grammar pair (dropping
the lazy flag turns a trigger-gated grammar into a hard one, and the model can
then never give a plain final answer); a grammar the engine refuses surfaces as
the same error text the server would send; generations on one engine are
serialised; the engine is never loaded or unloaded by the backend.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.base import (
    GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE,
    GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
    GrammarUnsupportedError,
    InvalidGrammarError,
)
from localm.plugins.coder.backends.http import CoderServerError
from localm.plugins.coder.backends.shared_engine import (
    SharedEngineBackend,
    engine_lock,
)


def _engine(reply="ok"):
    engine = MagicMock()
    engine.display_name = "m"
    engine.supports_grammar = True
    engine.chat_stream.side_effect = lambda messages, **kw: iter([reply])
    engine.count_messages_tokens.return_value = 7
    engine.count_tokens.return_value = 3
    engine.context_capacity.return_value = 2048
    return engine


MSG = [{"role": "user", "content": "hi"}]


def test_forwards_the_lazy_grammar_pair_to_the_engine():
    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    backend.chat(MSG, max_tokens=5, grammar="root ::= x", grammar_lazy=True,
                 grammar_triggers=["<tool_call>"], bogus="dropped", seed=None)
    _, kwargs = engine.chat_stream.call_args
    assert kwargs == {"max_tokens": 5, "grammar": "root ::= x",
                      "grammar_lazy": True, "grammar_triggers": ["<tool_call>"]}
    engine.validate_grammar.assert_called_once_with("root ::= x", lazy=True)


def test_chat_stream_forwards_the_same_kwargs():
    engine = _engine("piece")
    backend = SharedEngineBackend(engine, "m")
    assert list(backend.chat_stream(MSG, grammar="g", grammar_lazy=True,
                                    grammar_triggers=["t"])) == ["piece"]
    _, kwargs = engine.chat_stream.call_args
    assert kwargs["grammar_lazy"] is True and kwargs["grammar_triggers"] == ["t"]


def test_lazy_fields_are_dropped_without_a_grammar():
    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    backend.chat(MSG, grammar_lazy=True, grammar_triggers=["t"], temperature=0.1)
    _, kwargs = engine.chat_stream.call_args
    assert kwargs == {"temperature": 0.1}
    engine.validate_grammar.assert_not_called()


def test_unsupported_lazy_grammar_surfaces_as_the_servers_message():
    engine = _engine()
    engine.validate_grammar.side_effect = GrammarUnsupportedError(
        GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
    backend = SharedEngineBackend(engine, "m")
    with pytest.raises(CoderServerError) as info:
        backend.chat(MSG, grammar="g", grammar_lazy=True, grammar_triggers=["t"])
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(info.value)
    engine.chat_stream.assert_not_called()


def test_invalid_grammar_is_refused_before_generation():
    engine = _engine()
    engine.validate_grammar.side_effect = InvalidGrammarError("unbalanced")
    backend = SharedEngineBackend(engine, "m")
    with pytest.raises(CoderServerError, match="Invalid grammar: unbalanced"):
        backend.chat(MSG, grammar="g")
    engine.chat_stream.assert_not_called()


def test_lazy_grammar_without_triggers_is_refused():
    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    with pytest.raises(CoderServerError) as info:
        backend.chat(MSG, grammar="g", grammar_lazy=True)
    assert GRAMMAR_LAZY_NO_TRIGGERS_MESSAGE in str(info.value)
    engine.validate_grammar.assert_not_called()


def test_usage_is_counted_with_the_engines_tokenizer():
    engine = _engine("reply")
    backend = SharedEngineBackend(engine, "m")
    assert backend.last_usage == {}
    backend.chat(MSG)
    assert backend.last_usage == {"prompt_tokens": 7, "completion_tokens": 3,
                                  "total_tokens": 10}
    engine.count_tokens.side_effect = RuntimeError("no tokenizer")
    backend.chat(MSG)
    assert backend.last_usage == {}


def test_context_capacity_comes_from_the_engine():
    engine = _engine()
    assert SharedEngineBackend(engine, "m").context_capacity() == 2048
    engine.context_capacity.side_effect = RuntimeError("not loaded")
    assert SharedEngineBackend(engine, "m").context_capacity() is None


def test_backend_never_loads_or_unloads_the_engine():
    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    backend.chat(MSG)
    list(backend.chat_stream(MSG))
    engine.load.assert_not_called()
    engine.unload.assert_not_called()
    assert not hasattr(backend, "unload")


def test_supports_grammar_and_native_tools_flags():
    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    assert backend.supports_grammar is True
    assert backend.native_tools is False
    assert backend.supports_native_tools is False
    assert backend.model_id == "m"
    engine.supports_grammar = False
    assert SharedEngineBackend(engine, "m").supports_grammar is False


def test_each_generation_pins_the_engine_and_releases_it():
    engine = _engine()
    engine.active_requests = 0
    seen = []

    def chat_stream(messages, **kw):
        seen.append(engine.active_requests)
        return iter(["x"])

    engine.chat_stream.side_effect = chat_stream
    backend = SharedEngineBackend(engine, "m")
    backend.chat(MSG)
    list(backend.chat_stream(MSG))
    assert seen == [1, 1]
    assert engine.active_requests == 0


def test_pin_is_released_when_the_engine_raises():
    engine = _engine()
    engine.active_requests = 0
    engine.chat_stream.side_effect = RuntimeError("worker died")
    backend = SharedEngineBackend(engine, "m")
    with pytest.raises(RuntimeError):
        backend.chat(MSG)
    assert engine.active_requests == 0


def test_refuses_to_generate_once_the_owner_dropped_the_engine():
    from localm.plugins.coder.backends.shared_engine import ENGINE_GONE_MESSAGE
    engine = _engine()
    resident = [True]
    backend = SharedEngineBackend(engine, "m", still_resident=lambda: resident[0])
    assert backend.chat(MSG) == "ok"
    resident[0] = False
    with pytest.raises(CoderServerError) as info:
        backend.chat(MSG)
    assert ENGINE_GONE_MESSAGE in str(info.value)
    with pytest.raises(CoderServerError):
        list(backend.chat_stream(MSG))
    assert engine.chat_stream.call_count == 1


def test_a_supplied_lock_is_the_one_held():
    engine = _engine()
    lock = threading.Lock()
    backend = SharedEngineBackend(engine, "m", lock=lock)
    assert backend._lock is lock
    with lock:
        t = threading.Thread(target=backend.chat, args=(MSG,))
        t.start()
        t.join(0.2)
        assert t.is_alive(), "chat did not wait for the supplied lock"
    t.join(5)
    assert not t.is_alive()


def test_sub_agent_model_override_is_refused_on_this_backend(tmp_path, monkeypatch):
    """A child asking for a different model used to get an HTTP backend at a
    guessed port; on a shared engine there is no server, so the override is
    refused with a message instead of silently changing the model."""
    from unittest.mock import patch
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.audit import SessionMode
    from localm.plugins.coder.tools.agents import tool_spawn_agent

    engine = _engine()
    backend = SharedEngineBackend(engine, "m")
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        parent = Agent(backend=backend, cwd=tmp_path, mode=SessionMode.LOG,
                       auto_approve=True)
    made = []
    monkeypatch.setattr("localm.plugins.coder.backends.http.make_localm_backend",
                        lambda *a, **k: made.append((a, k)))
    try:
        result = tool_spawn_agent(tmp_path, "do it", model="other",
                                  _parent_agent=parent)
    finally:
        parent.close()
    assert result.ok is False
    assert "not available" in result.output
    assert made == []


def test_one_lock_per_engine():
    a, b = _engine(), _engine()
    assert engine_lock(a) is engine_lock(a)
    assert engine_lock(a) is not engine_lock(b)
    assert SharedEngineBackend(a, "m")._lock is SharedEngineBackend(a, "m")._lock


def test_generations_on_one_engine_never_overlap():
    engine = _engine()
    active = []
    overlaps = []

    def chat_stream(messages, **kw):
        active.append(1)
        if len(active) > 1:
            overlaps.append(1)
        time.sleep(0.05)
        active.pop()
        return iter(["x"])

    engine.chat_stream.side_effect = chat_stream
    backends = [SharedEngineBackend(engine, "m") for _ in range(4)]
    threads = [threading.Thread(target=b.chat, args=(MSG,)) for b in backends]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine.chat_stream.call_count == 4
    assert overlaps == []
