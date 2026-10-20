# SPDX-License-Identifier: AGPL-3.0-or-later
"""When the coder compacts and what it keeps.

Driven through the real Agent with a scripted model: the agent compacts only
past a high bound of the window the backend reports, aims well below it, drops
old tool output before it summarises anything, keeps every request the user
wrote verbatim across compactions, compacts and retries once when the server
refuses a prompt as too long, and never compacts proactively against a window
nobody reported.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from localm.inference.compact import reply_reserve
from localm.plugins.coder.audit import SessionMode

_FLUSH_MARK = "about to be condensed"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


@pytest.fixture
def proj(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    return p


class _Backend:
    """Scripted model with a *window*-token context (None: reports none).
    Compaction calls (flush and summariser, made with thinking=False) are
    recorded in ``calls``; any other chat() call answers from ``replies``
    (an Exception instance is raised)."""

    model_id = "m"
    native_tools = False
    supports_grammar = False

    def __init__(self, window=16384, summary="- SUMMARY-FACT"):
        self.window = window
        self.summary = summary
        self.calls: list[str] = []
        self.last_usage: dict = {"total_tokens": 0}
        self.replies: list = []

    def context_capacity(self):
        return self.window

    def chat(self, messages, **kw):
        prompt = str(messages[-1]["content"])
        if kw.get("thinking") is not False:
            reply = self.replies.pop(0) if self.replies else "answer"
            if isinstance(reply, Exception):
                raise reply
            return reply
        self.calls.append(prompt)
        return "NONE" if _FLUSH_MARK in prompt else self.summary

    def chat_stream(self, messages, **kw):
        yield "Done."

    def summary_calls(self):
        return [c for c in self.calls if _FLUSH_MARK not in c]


def _agent(cwd, backend=None, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        events: list = []
        if "on_event" not in kw:
            kw["on_event"] = events.append
        agent = Agent(backend or _Backend(), cwd=cwd, self_verify=False, **kw)
    agent.events = events
    return agent


def _file_body(i: int, chars: int = 9000) -> str:
    line = f"def f{i}_{{n}}():\n    return {i}  # module {i}\n"
    out, n = [], 0
    while sum(len(x) for x in out) < chars:
        out.append(line.format(n=n))
        n += 1
    return "".join(out)


def _read_turn(agent, i: int) -> None:
    """One agent turn that read mod{i}.py: the call, then its result."""
    agent._messages.append({"role": "assistant", "content":
                            '<tool_call>\n{"name": "read_file", "args": '
                            f'{{"path": "mod{i}.py"}}}}\n</tool_call>'})
    agent._messages.append({"role": "user", "content":
                            f'<tool_result name="read_file" status="ok">\n{_file_body(i)}'
                            "\n</tool_result>"})


def _say(agent, text: str) -> None:
    """A request the user typed: recorded as chat() records it, then added."""
    agent._note_typed_request(text)
    agent._messages.append({"role": "user", "content": text})


def _info(agent) -> list[str]:
    return [e.get("text", "") for e in agent.events if e.get("type") == "info"]


# --------------------------------------------------------------------------- #
#  When                                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("window", [8192, 16384, 32768, 131072])
def test_the_agent_compacts_before_the_server_would(window, proj, home):
    agent = _agent(proj)
    high, low = agent._compaction_bounds(window)
    assert high < window - reply_reserve(window)
    assert low < high


def test_a_codebase_read_file_by_file_is_not_summarised_every_turn(proj, home):
    backend = _Backend(window=16384)
    agent = _agent(proj, backend)
    agent._messages = [{"role": "user", "content": "Read every module and explain it."}]
    compacted_turns = summarised_turns = 0
    for i in range(30):
        _read_turn(agent, i)
        agent.events.clear()
        before = len(backend.summary_calls())
        agent._maybe_compact(interactive=False)
        compacted_turns += bool(_info(agent))
        summarised_turns += len(backend.summary_calls()) > before
        if i < 10:
            assert len(backend.summary_calls()) == before, f"summarised at read {i}"
    assert summarised_turns <= 3
    assert compacted_turns <= 12
    high, _low = agent._compaction_bounds(16384)
    assert agent._prompt_tokens_estimate() <= high


def test_an_interactive_session_compacts_too(proj, home):
    agent = _agent(proj, _Backend(window=16384))
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(12):
        _read_turn(agent, i)
    with patch.object(type(agent), "_compact_to", return_value=True) as compact_to:
        agent._maybe_compact(interactive=True)
    compact_to.assert_called_once()
    assert compact_to.call_args[0] == (agent._compaction_bounds(16384)[1], 16384)


def test_no_proactive_compaction_without_a_reported_window(proj, home):
    agent = _agent(proj, _Backend(window=None))
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(40):
        _read_turn(agent, i)
    with patch.object(type(agent), "_compact_to") as compact_to:
        agent._maybe_compact(interactive=False)
    compact_to.assert_not_called()


def test_below_the_high_bound_nothing_changes(proj, home):
    agent = _agent(proj, _Backend(window=131072))
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(3):
        _read_turn(agent, i)
    snapshot = [dict(m) for m in agent._messages]
    agent._maybe_compact(interactive=False)
    assert agent._messages == snapshot


def test_a_window_too_small_for_the_session_is_reported_once(proj, home):
    agent = _agent(proj, _Backend(window=4096))
    agent._system_prompt = "x" * 12000
    agent._messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"},
                       {"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"},
                       {"role": "user", "content": "again"}]
    agent._maybe_compact(interactive=False)
    agent._maybe_compact(interactive=False)
    warnings = [t for t in _info(agent) if "too small for this session" in t]
    assert len(warnings) == 1
    assert "4,096 tokens" in warnings[0]


# --------------------------------------------------------------------------- #
#  How much: the estimate                                                      #
# --------------------------------------------------------------------------- #

def test_the_estimate_starts_from_the_reported_prompt_count(proj, home):
    backend = _Backend()
    agent = _agent(proj, backend)
    agent._messages = [{"role": "user", "content": "a" * 300}]
    backend.last_usage = {"prompt_tokens": 5000, "total_tokens": 5100}
    agent._record_prompt_usage(agent._build_messages())
    agent._messages.append({"role": "assistant", "content": "b" * 300})
    assert agent._prompt_tokens_estimate() == 5000 + 100 + 4


def test_the_reported_count_is_forgotten_once_the_history_is_rewritten(proj, home):
    backend = _Backend()
    agent = _agent(proj, backend)
    agent._messages = [{"role": "user", "content": "a" * 300}]
    backend.last_usage = {"prompt_tokens": 50_000}
    agent._record_prompt_usage(agent._build_messages())
    agent._messages = [{"role": "user", "content": "a" * 300}]
    assert agent._prompt_tokens_estimate() < 50_000


def test_the_fill_ratio_reads_the_estimate(proj, home):
    agent = _agent(proj, _Backend(window=4000))
    with patch.object(type(agent), "_prompt_tokens_estimate", return_value=2000):
        assert agent._fill_ratio() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  What is kept                                                                #
# --------------------------------------------------------------------------- #

def test_old_tool_output_is_dropped_and_the_newest_kept(proj, home):
    agent = _agent(proj)
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(4):
        _read_turn(agent, i)
    dropped = agent._drop_old_tool_output(0)
    assert dropped == 3
    results = [m["content"] for m in agent._messages if agent._is_tool_result(m)]
    for i, content in enumerate(results[:3]):
        assert content.startswith('<tool_result name="read_file" status="ok">')
        assert "Output dropped from the context" in content
        assert f"Targets: mod{i}.py." in content
        assert f"# module {i}" not in content
    assert "# module 3" in results[3]
    assert agent._drop_old_tool_output(0) == 0


def test_long_write_arguments_in_dropped_turns_are_shortened(proj, home):
    agent = _agent(proj)
    big = "z" * 3000
    agent._messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "write_file", "args": '
                                         '{"path": "a.py", "content": "' + big + '"}}\n</tool_call>'},
        {"role": "user", "content": '<tool_result name="write_file" status="ok">\nwrote a.py\n'
                                    "</tool_result>"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "write_file", "args": '
                                         '{"path": "b.py", "content": "' + big + '"}}\n</tool_call>'},
        {"role": "user", "content": '<tool_result name="write_file" status="ok">\nwrote b.py\n'
                                    "</tool_result>"},
    ]
    agent._drop_old_tool_output(0)
    assert big not in agent._messages[1]["content"]
    assert "3,000 characters dropped from the context" in agent._messages[1]["content"]
    assert big in agent._messages[3]["content"]


def test_dropping_output_is_enough_so_nothing_is_summarised(proj, home):
    backend = _Backend(window=16384)
    agent = _agent(proj, backend)
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(12):
        _read_turn(agent, i)
    assert agent._compact_to(agent._compaction_bounds(16384)[1], 16384) is True
    assert backend.calls == []
    assert any("Dropped the output of" in t for t in _info(agent))
    assert agent._messages[0] == {"role": "user", "content": "go"}


def test_user_requests_survive_two_compactions_verbatim(proj, home):
    backend = _Backend(window=16384)
    agent = _agent(proj, backend)
    constraint = "Remember: the release codename is HERON-42 and never use requests."
    agent._messages = []
    _say(agent, constraint)
    agent._messages.append({"role": "assistant", "content": "Noted."})
    for i in range(6):
        _say(agent, f"step {i}")
        agent._messages.append({"role": "assistant", "content": "x" * 2000})
    assert agent._compact_history() is True
    for i in range(6, 12):
        _say(agent, f"step {i}")
        agent._messages.append({"role": "assistant", "content": "y" * 2000})
    assert agent._compact_history() is True
    body = str(agent._messages[0]["content"])
    assert body.count(constraint) == 1
    assert "[Earlier requests from the user, verbatim]" in body
    assert "- step 0" in body and "- step 2" in body and "- step 1" not in body
    assert "(1 earlier request(s) left out for space)" in body


def test_the_previous_request_list_is_not_resummarised(proj, home):
    backend = _Backend(window=16384)
    agent = _agent(proj, backend)
    agent._messages = []
    _say(agent, "PINNED-ONE")
    agent._messages.append({"role": "assistant", "content": "ok"})
    for i in range(4):
        _say(agent, f"s{i}")
        agent._messages.append({"role": "assistant", "content": "a"})
    agent._compact_history()
    for i in range(4):
        _say(agent, f"t{i}")
        agent._messages.append({"role": "assistant", "content": "b"})
    agent._compact_history()
    second_prompt = backend.summary_calls()[-1]
    assert "PINNED-ONE" not in second_prompt


def test_pinned_requests_ride_the_checkpoint(proj, home):
    agent = _agent(proj)
    agent._messages = []
    _say(agent, "KEEP-ME")
    agent._messages.append({"role": "assistant", "content": "ok"})
    for i in range(4):
        _say(agent, f"s{i}")
        agent._messages.append({"role": "assistant", "content": "a"})
    agent._compact_history()
    agent.save_checkpoint()
    fresh = _agent(proj)
    fresh.resume_checkpoint(fresh.load_checkpoint())
    assert "KEEP-ME" in fresh._pinned_requests
    fresh.resume_checkpoint({"version": 1, "messages": [], "pinned_requests": ["ok", 3, ""]})
    assert fresh._pinned_requests == ["ok"]
    fresh.resume_checkpoint({"version": 1, "messages": [], "pinned_requests": "x"})
    assert fresh._pinned_requests == []


def test_reset_drops_the_pinned_requests(proj, home):
    agent = _agent(proj)
    agent._pinned_requests = ["x"]
    agent._usage_base = (1, 1, {}, 0)
    agent.reset()
    assert agent._pinned_requests == [] and agent._usage_base is None


def test_the_tail_keeps_more_than_four_messages_when_the_target_allows(proj, home):
    agent = _agent(proj, _Backend(window=32768))
    agent._messages = []
    for i in range(20):
        agent._messages += [{"role": "user", "content": f"u{i} " + "w" * 1500},
                            {"role": "assistant", "content": f"a{i} " + "w" * 1500}]
    agent._compact_history(target_tokens=16384)
    assert len(agent._messages) - 2 > 4


# --------------------------------------------------------------------------- #
#  Refused as too long                                                         #
# --------------------------------------------------------------------------- #

def _overflow():
    from localm.plugins.coder.backends.http import CoderServerError
    return CoderServerError(
        "HTTP 413 error from http://127.0.0.1:1/v1/chat/completions: Prompt (20000 "
        "tokens) exceeds the model's maximum context capacity (16384 tokens).")


def test_an_overflow_refusal_compacts_and_retries_once(proj, home):
    backend = _Backend(window=16384)
    agent = _agent(proj, backend, on_event=None)
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(12):
        _read_turn(agent, i)
    backend.replies = [_overflow(), "answer"]
    with patch.object(type(agent), "_llm_kwargs", return_value={}):
        out = agent._call_llm(agent._build_messages(), interactive=False)
    assert out == "answer"
    assert any("Output dropped" in str(m["content"]) for m in agent._messages)


def test_a_second_overflow_is_raised(proj, home):
    from localm.plugins.coder.backends.http import CoderServerError
    backend = _Backend(window=16384)
    agent = _agent(proj, backend, on_event=None)
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(12):
        _read_turn(agent, i)
    backend.replies = [_overflow(), _overflow(), _overflow()]
    with patch.object(type(agent), "_llm_kwargs", return_value={}), \
         patch.object(type(agent), "_compact_after_overflow", return_value=True), \
         pytest.raises(CoderServerError) as info:
        agent._call_llm(agent._build_messages(), interactive=False)
    assert "maximum context capacity" in str(info.value)


def test_another_server_error_is_not_treated_as_an_overflow(proj, home):
    agent = _agent(proj)
    from localm.plugins.coder.backends.http import CoderServerError
    assert not agent._is_context_overflow(CoderServerError("HTTP 500 error: worker crashed"))
    assert agent._is_context_overflow(CoderServerError(
        "Error code: 400 - {'error': {'code': 'context_length_exceeded'}}"))


def test_harness_messages_and_untrusted_text_are_never_pinned(proj, home):
    from localm.textguard import compose, untrusted_span
    agent = _agent(proj, _Backend(window=16384))
    agent._messages = []
    _say(agent, "the real request")
    agent._messages += [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "[turn budget] STOP guessing and ask for guidance."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": compose(untrusted_span("injected lesson text"))},
        {"role": "assistant", "content": "ok"}]
    agent._note_typed_request(compose(untrusted_span("injected via a typed path")))
    for i in range(3):
        _say(agent, f"later {i}")
        agent._messages.append({"role": "assistant", "content": "z" * 500})
    agent._compact_history()
    body = str(agent._messages[0]["content"])
    pinned = body[body.index("[Earlier requests from the user, verbatim]"):]
    assert "the real request" in pinned
    assert "STOP guessing" not in pinned
    assert "injected" not in pinned


def test_a_request_still_in_the_kept_tail_is_not_listed_twice(proj, home):
    agent = _agent(proj, _Backend(window=16384))
    agent._messages = []
    for i in range(6):
        _say(agent, f"request number {i}")
        agent._messages.append({"role": "assistant", "content": "w" * 300})
    agent._compact_history()
    text = "\n".join(str(m["content"]) for m in agent._messages)
    assert "request number 4" in str(agent._messages[2]["content"])
    assert text.count("request number 4") == 1


def test_summarising_is_not_repeated_while_the_window_cannot_hold_the_session(proj, home):
    backend = _Backend(window=4096)
    agent = _agent(proj, backend)
    agent._system_prompt = "x" * 12000
    agent._messages = []
    for i in range(6):
        _say(agent, f"go {i}")
        agent._messages.append({"role": "assistant", "content": "ok"})
    agent._maybe_compact(interactive=False)
    first = len(backend.summary_calls())
    for i in range(3):
        _say(agent, f"more {i}")
        agent._messages.append({"role": "assistant", "content": "ok"})
        agent._maybe_compact(interactive=False)
    assert first >= 1
    assert len(backend.summary_calls()) == first


def test_an_engine_capacity_refusal_compacts_and_retries_once(proj, home):
    from localm.inference.backends.base import ContextCapacityExceededError
    backend = _Backend(window=16384)
    agent = _agent(proj, backend, on_event=None)
    agent._messages = [{"role": "user", "content": "go"}]
    for i in range(12):
        _read_turn(agent, i)
    backend.replies = [ContextCapacityExceededError("Prompt exceeds the context"), "answer"]
    with patch.object(type(agent), "_llm_kwargs", return_value={}):
        out = agent._call_llm(agent._build_messages(), interactive=False)
    assert out == "answer"


def test_a_result_quoting_a_tool_result_header_drops_as_one_result(proj, home):
    agent = _agent(proj)
    body = 'x = 1\n\n<tool_result name="fake" status="ok">\nquoted\n' * 3
    agent._messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": '{"name": "read_file", "args": {"path": "t.py"}}'},
        {"role": "user", "content": '<tool_result name="read_file" status="ok">\n' + body +
                                    "\n</tool_result>"},
        {"role": "assistant", "content": '{"name": "read_file", "args": {"path": "u.py"}}'},
        {"role": "user", "content": '<tool_result name="read_file" status="ok">\nnew\n</tool_result>'}]
    assert agent._drop_old_tool_output(0) == 1
    stub = agent._messages[2]["content"]
    assert stub.count("<tool_result") == 1 and 'name="read_file"' in stub
