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


# --------------------------------------------------------------------------- #
#  A thinking model's scratchpad never reaches the answer                     #
# --------------------------------------------------------------------------- #

_THINK_REPLY = (
    "<think>\nI should delete everything first.\n"
    '<tool_call>{"name": "delete_file", "args": {"path": "a.py"}}</tool_call>\n'
    "No, that is wrong.\n</think>\n"
    'Done: <tool_call>{"name": "read_file", "args": {"path": "a.py"}}</tool_call>'
)


def test_chat_returns_the_answer_and_latches_the_reasoning():
    from localm.plugins.coder.parser import parse_tool_calls
    engine = _engine(_THINK_REPLY)
    backend = SharedEngineBackend(engine, "m")
    answer = backend.chat(MSG)
    assert "<think>" not in answer and "</think>" not in answer
    assert "delete everything" not in answer
    assert answer.lstrip().startswith("Done:")
    assert "delete everything first" in backend.last_reasoning
    # A tool call the model wrote while thinking is not a call at all.
    calls = parse_tool_calls(answer, tool_names={"delete_file", "read_file"})
    assert [c.name for c in calls] == ["read_file"]


def test_chat_stream_routes_reasoning_to_the_callback_and_never_yields_it():
    from localm.plugins.coder.parser import parse_tool_calls
    # Pieces split the tags themselves, the way a token stream does.
    pieces = ["<thi", "nk>\nplan: ", '<tool_call>{"name": "delete_file", ',
              '"args": {"path": "a.py"}}</tool_call>', "\n</thi", "nk>\nDone: ",
              '<tool_call>{"name": "read_file", "args": {"path": "a.py"}}</tool_call>']
    engine = _engine()
    engine.chat_stream.side_effect = lambda messages, **kw: iter(pieces)
    backend = SharedEngineBackend(engine, "m")
    reasoning = []
    visible = "".join(backend.chat_stream(MSG, on_reasoning=reasoning.append))
    assert "think" not in visible
    assert "delete_file" not in visible
    assert visible.lstrip().startswith("Done:")
    assert "delete_file" in "".join(reasoning)
    assert backend.last_reasoning == "".join(reasoning)
    calls = parse_tool_calls(visible, tool_names={"delete_file", "read_file"})
    assert [c.name for c in calls] == ["read_file"]


def test_an_unclosed_think_block_is_all_reasoning():
    engine = _engine("<think>still thinking when the budget ran out")
    backend = SharedEngineBackend(engine, "m")
    assert backend.chat(MSG) == ""
    assert "still thinking" in backend.last_reasoning
    engine.chat_stream.side_effect = lambda messages, **kw: iter(
        ["<think>still ", "thinking"])
    assert "".join(backend.chat_stream(MSG)) == ""
    assert backend.last_reasoning == "still thinking"


def test_usage_counts_the_reasoning_tokens_too():
    engine = _engine(_THINK_REPLY)
    backend = SharedEngineBackend(engine, "m")
    backend.chat(MSG)
    engine.count_tokens.assert_called_once_with(_THINK_REPLY)


# --------------------------------------------------------------------------- #
#  Usage is counted under the lock, per thread                                #
# --------------------------------------------------------------------------- #

def test_usage_is_counted_while_the_generation_lock_is_still_held():
    engine = _engine("reply")
    lock = threading.Lock()
    held_at_count = []
    engine.count_tokens.side_effect = lambda text: held_at_count.append(lock.locked()) or 3
    backend = SharedEngineBackend(engine, "m", lock=lock)
    backend.chat(MSG)
    list(backend.chat_stream(MSG))
    assert held_at_count == [True, True]


def test_usage_and_reasoning_are_per_thread_on_a_shared_backend():
    engine = _engine()
    replies = {"a": "<think>ra</think>A", "b": "<think>rb</think>B"}
    gate = threading.Barrier(2)
    engine.count_tokens.side_effect = lambda text: len(text)
    engine.count_messages_tokens.side_effect = lambda messages: 0

    def chat_stream(messages, **kw):
        return iter([replies[messages[0]["content"]]])

    engine.chat_stream.side_effect = chat_stream
    backend = SharedEngineBackend(engine, "m")
    seen = {}

    def run(tag):
        gate.wait()
        text = backend.chat([{"role": "user", "content": tag}])
        # The other thread's generation lands between this thread's call and
        # its read of last_usage/last_reasoning.
        time.sleep(0.05)
        seen[tag] = (text, backend.last_usage["total_tokens"], backend.last_reasoning)

    threads = [threading.Thread(target=run, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["a"] == ("A", len(replies["a"]), "ra")
    assert seen["b"] == ("B", len(replies["b"]), "rb")


def test_a_stream_broken_off_early_still_records_its_usage():
    engine = _engine()
    engine.active_requests = 0
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["one", "two", "three"])
    engine.count_tokens.side_effect = lambda text: len(text)
    backend = SharedEngineBackend(engine, "m")
    stream = backend.chat_stream(MSG)
    assert next(stream) == "one"
    stream.close()
    assert backend.last_usage["completion_tokens"] == len("one")
    assert engine.active_requests == 0


# --------------------------------------------------------------------------- #
#  Cancellation                                                               #
# --------------------------------------------------------------------------- #

def test_cancel_aborts_the_generation_in_flight_and_refuses_the_next():
    engine = _engine()
    produced = []

    def chat_stream(messages, **kw):
        def gen():
            for i in range(1000):
                produced.append(i)
                yield f"p{i} "
                if i == 3:
                    backend.cancel("timed out")
        return gen()

    engine.chat_stream.side_effect = chat_stream
    backend = SharedEngineBackend(engine, "m")
    text = backend.chat(MSG)
    assert text.startswith("p0 p1 p2 p3")
    assert len(produced) <= 5, "the engine stream was not closed on cancel"
    assert backend.cancelled
    with pytest.raises(CoderServerError, match="cancelled"):
        backend.chat(MSG)
    with pytest.raises(CoderServerError, match="cancelled"):
        list(backend.chat_stream(MSG))
    assert engine.chat_stream.call_count == 1


def test_an_unkeyable_engine_still_serialises_on_one_lock():
    class Unhashable:
        __hash__ = None

    a, b = Unhashable(), Unhashable()
    assert engine_lock(a) is engine_lock(b)
    assert engine_lock(a) is engine_lock(a)
