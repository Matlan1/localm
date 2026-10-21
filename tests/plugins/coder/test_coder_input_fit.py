# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder builds most of what it sends a model. When the model's
pre-tokenizer refuses a request, the coder cuts what it built (tool output,
file contents) and retries; only text the user typed is reported back to the
user, with what to do about it."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.plugins.coder.agent.inputfit import InputTooLongForModel, shrink
from localm.plugins.coder.backends.http import CoderServerError
from localm.textguard import GuardedText, untrusted_spans_of

_PART_REFUSAL = ("400 Bad Request: This model's pre-tokenizer (tekken) cannot safely "
                 "handle more than 65536 characters in one message; one message in "
                 "this input is 81584. Send that text in smaller parts.")
_TOTAL_REFUSAL = ("400 Bad Request: This model's pre-tokenizer (tekken) cannot safely "
                  "handle more than 65536 characters at once; this input is 81584. "
                  "Send less text per request.")
_RUN_REFUSAL = ("This model's pre-tokenizer (tekken) crashes on an unbroken run of "
                "more than 64 letters; this input has a run of 200. Break the run "
                "with punctuation or a line break, or use a model with a different "
                "tokenizer.")


def _make_agent(tmp_path: Path):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    backend.supports_grammar = False
    backend.last_usage = {}
    backend.last_reasoning = ""
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, self_verify=False)
    agent.report_progress = False
    agent.on_event = None
    return agent


def _tool_result(n: int) -> GuardedText:
    body = ("line of tool output\n" * (n // 20 + 1))[:n]
    from localm.textguard import compose, untrusted_span
    return compose('<tool_result name="read_file" status="ok">\n',
                   untrusted_span(body), "\n</tool_result>")


def _calls(agent):
    return agent.backend.chat.call_args_list


def test_an_oversized_tool_result_is_cut_and_the_call_retried(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("read the big file")
    agent._add_user("read the big file")
    agent._add_assistant("<tool_call>...</tool_call>")
    agent._add_user(_tool_result(80000))
    agent.backend.chat.side_effect = [CoderServerError(_PART_REFUSAL), "done"]
    out = agent._call_llm(agent._build_messages(), interactive=False)
    assert out == "done"
    retried = _calls(agent)[1].args[0]
    sizes = [len(str(m["content"])) for m in retried]
    assert max(sizes) < 65536
    cut = str(retried[-1]["content"])
    assert "characters omitted to fit this model's input limit" in cut
    assert untrusted_spans_of(retried[-1]["content"])
    assert any("read the big file" == str(m["content"]) for m in retried)


def test_the_users_own_oversized_message_is_reported_not_cut(tmp_path):
    agent = _make_agent(tmp_path)
    typed = "please review: " + "word " * 20000
    agent._record_typed_request(typed)
    agent._add_user(typed)
    agent.backend.chat.side_effect = [CoderServerError(_PART_REFUSAL)]
    with pytest.raises(InputTooLongForModel) as exc:
        agent._call_llm(agent._build_messages(), interactive=False)
    assert "Your message is too long for this model" in str(exc.value)
    assert "65536" in str(exc.value)
    assert "removed from this conversation" in str(exc.value)
    assert "message from the user was removed" in str(agent._messages[-1]["content"])
    # The next, smaller message goes through.
    agent._record_typed_request("now a short one")
    agent._add_user("now a short one")
    agent.backend.chat.side_effect = ["fine"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "fine"


def test_typed_text_behind_a_lessons_preamble_is_still_the_users(tmp_path):
    agent = _make_agent(tmp_path)
    typed = "please review: " + "word " * 20000
    agent._record_typed_request(typed)
    agent._add_user("## Lessons\n- something learned\n\n## Task\n" + typed + "\n[plan note]")
    agent.backend.chat.side_effect = [CoderServerError(_PART_REFUSAL)]
    with pytest.raises(InputTooLongForModel):
        agent._call_llm(agent._build_messages(), interactive=False)
    assert len(_calls(agent)) == 1


def test_text_the_agent_sent_itself_is_not_treated_as_typed(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("fix the build")
    agent._add_user("fix the build")
    feedback = "[verify] the check failed:\n" + ("traceback line\n" * 5000)
    agent._last_user_request = feedback
    agent._add_user(feedback)
    agent.backend.chat.side_effect = [CoderServerError(_PART_REFUSAL), "ok"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "ok"
    assert "characters omitted" in str(_calls(agent)[1].args[0][-1]["content"])


def test_a_local_engine_refusal_is_fitted_the_same_way(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("go")
    agent._add_user("go")
    agent._add_user(_tool_result(70000))
    agent.backend.chat.side_effect = [
        PretokenizerUnsafeInputError(_PART_REFUSAL.split(": ", 1)[1]), "ok"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "ok"


def test_a_whole_input_refusal_shrinks_the_largest_built_messages(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("go")
    agent._add_user("go")
    for _ in range(4):
        agent._add_user(_tool_result(20000))
    agent.backend.chat.side_effect = [CoderServerError(_TOTAL_REFUSAL), "ok"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "ok"
    retried = _calls(agent)[1].args[0]
    assert sum(len(str(m["content"])) for m in retried) < 65536


def test_a_long_run_in_tool_output_is_elided(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("go")
    agent._add_user("go")
    agent._add_user(_tool_result(100) + GuardedText("a" * 200))
    agent.backend.chat.side_effect = [CoderServerError(_RUN_REFUSAL), "ok"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "ok"
    last = str(_calls(agent)[1].args[0][-1]["content"])
    assert "a" * 65 not in last and "characters omitted" in last


def test_an_unrelated_server_error_still_propagates(tmp_path):
    agent = _make_agent(tmp_path)
    agent._add_user("go")
    agent.backend.chat.side_effect = [CoderServerError("500: something else")]
    with pytest.raises(CoderServerError):
        agent._call_llm(agent._build_messages(), interactive=False)


def test_fitting_is_bounded(tmp_path):
    agent = _make_agent(tmp_path)
    agent._record_typed_request("go")
    agent._add_user("go")
    agent._add_user(_tool_result(200000))
    agent.backend.chat.side_effect = CoderServerError(_RUN_REFUSAL)
    agent._add_user(GuardedText("b" * 300))
    with pytest.raises((CoderServerError, InputTooLongForModel)):
        agent._call_llm(agent._build_messages(), interactive=False)
    assert len(_calls(agent)) <= 5


def test_shrink_keeps_head_and_tail_within_budget():
    text = "H" + "x" * 9998 + "T"
    out = shrink(text, 1000)
    assert len(str(out)) <= 1000
    assert str(out).startswith("H") and str(out).endswith("T")
    assert shrink("short", 1000) == "short"
