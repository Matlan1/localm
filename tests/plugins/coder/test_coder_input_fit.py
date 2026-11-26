# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder builds most of what it sends a model. When the model's
pre-tokenizer refuses a request, the coder cuts what it built (tool output,
file contents) and retries; only text the user typed is reported back to the
user, with what to do about it."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference import pretokenizer_guard as guard
from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.plugins.coder.agent.inputfit import (
    InputTooLongForModel, fit_messages, shrink, user_message_refusal)
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


# --------------------------------------------------------------------------- #
#  Every text the guard refuses is one the coder cuts or reports               #
# --------------------------------------------------------------------------- #

_LIMIT = guard.UNSAFE_PRE_TYPES["deepseek-v3"].newline_run
_LINE_BREAK_TEXTS = {
    "contiguous": "a" + "\n" * (_LIMIT + 40) + "b",
    "indented blank lines": "def f():\n" + "        \n" * (_LIMIT + 40) + "    return 1\n",
    "CRLF tab rows": "name\tvalue\r\n" + "\t\t\t\r\n" * 200 + "end\t1\r\n",
    "padded terminal rows": "top\n" + (" " * 80 + "\n") * (_LIMIT + 10) + "$ ",
}


def _refused_texts(pre_type):
    """(kind, text) for every bound of *pre_type*'s policy."""
    p = guard.UNSAFE_PRE_TYPES[pre_type]
    if p.char_class is not None:
        kind = guard._RUN_KIND[p.char_class]
        unit = {"letters": "ab", "letters and spaces": "a b ", "digits": "57",
                "whitespace characters": " \t"}[kind]
        yield kind, "x. " + (unit * p.max_run)[:p.max_run + 30] + ". y"
    if p.digit_run is not None:
        yield "digits", "x " + "7" * (p.digit_run + 30) + " y"
    if p.newline_run is not None:
        for text in _LINE_BREAK_TEXTS.values():
            yield "line breaks", text
    if p.cost_budget is not None:
        n = int(p.cost_budget ** 0.5) + 1
        yield "block", "x " + "\u4e2d" * n + " y"
    if p.max_chars is not None:
        yield "chars", "ok. " * (p.max_chars // 4 + 8)


_REFUSED_CASES = [(pre, kind, i) for pre in sorted(guard.UNSAFE_PRE_TYPES)
                  for i, (kind, _text) in enumerate(_refused_texts(pre))]


def _refusal_of(pre_type, text):
    with pytest.raises(PretokenizerUnsafeInputError) as exc:
        guard.check_text(pre_type, text)
    return str(exc.value)


@pytest.fixture
def _table_limits(monkeypatch):
    monkeypatch.setattr(guard, "_LINUX", False)


@pytest.mark.parametrize("pre_type,kind,i", _REFUSED_CASES)
def test_a_refused_text_the_coder_built_is_cut_until_it_passes(pre_type, kind, i, _table_limits):
    text = [t for _k, t in _refused_texts(pre_type)][i]
    messages = [{"role": "user", "content": "go"}, {"role": "user", "content": text}]
    for _ in range(4):
        try:
            guard.check_text(pre_type, str(messages[1]["content"]))
            break
        except PretokenizerUnsafeInputError as e:
            refusal = guard.parse_refusal("400 Bad Request: " + str(e))
            assert refusal is not None, str(e)
            changed, offenders = fit_messages(messages, refusal, protected={0}, reserve=0)
            assert changed == [1] and not offenders, (kind, str(e))
    guard.check_text(pre_type, str(messages[1]["content"]))
    assert messages[0]["content"] == "go"


@pytest.mark.parametrize("pre_type,kind,i", _REFUSED_CASES)
def test_a_refused_text_the_user_typed_is_named_with_advice(pre_type, kind, i, _table_limits):
    text = [t for _k, t in _refused_texts(pre_type)][i]
    refusal = guard.parse_refusal(_refusal_of(pre_type, text))
    messages = [{"role": "user", "content": text}, {"role": "user", "content": "other"}]
    changed, offenders = fit_messages(messages, refusal, protected={0, 1}, reserve=0)
    assert changed == []
    assert 0 in offenders
    assert messages[0]["content"] == text, "typed text is never changed"
    advice = user_message_refusal(refusal)
    assert advice and (refusal.kind != "run" or guard.RUN_ADVICE[refusal.run_kind] in advice)


def test_every_kind_of_run_a_refusal_names_has_its_advice():
    assert set(guard.RUN_ADVICE) == set(guard._RUN_KIND.values()) | {guard.BLOCK}
    blank = guard.RUN_ADVICE["line breaks"]
    assert "blank lines" in blank and "punctuation" not in blank


@pytest.mark.parametrize("shape", sorted(_LINE_BREAK_TEXTS))
@pytest.mark.parametrize("pre_type", ["gpt-4o", "deepseek-v3"])
def test_a_tool_result_of_blank_lines_is_cut_and_the_call_retried(
        tmp_path, shape, pre_type, _table_limits):
    text = _LINE_BREAK_TEXTS[shape]
    agent = _make_agent(tmp_path)
    agent._record_typed_request("go")
    agent._add_user("go")
    agent._add_user(_tool_result(100) + GuardedText(text))
    agent.backend.chat.side_effect = [CoderServerError(_refusal_of(pre_type, text)), "ok"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "ok"
    sent = _calls(agent)[1].args[0]
    for m in sent:
        guard.check_text(pre_type, str(m["content"]))


@pytest.mark.parametrize("shape", sorted(_LINE_BREAK_TEXTS))
def test_a_typed_message_of_blank_lines_does_not_stay_in_the_way(tmp_path, shape, _table_limits):
    typed = _LINE_BREAK_TEXTS[shape]
    agent = _make_agent(tmp_path)
    agent._record_typed_request(typed)
    agent._add_user(typed)
    agent.backend.chat.side_effect = [CoderServerError(_refusal_of("deepseek-v3", typed))]
    with pytest.raises(InputTooLongForModel) as exc:
        agent._call_llm(agent._build_messages(), interactive=False)
    assert "Remove blank lines" in str(exc.value)
    assert "removed from this conversation" in str(exc.value)
    agent._record_typed_request("now a short one")
    agent._add_user("now a short one")
    agent.backend.chat.side_effect = ["fine"]
    assert agent._call_llm(agent._build_messages(), interactive=False) == "fine"
    for m in _calls(agent)[-1].args[0]:
        guard.check_text("deepseek-v3", str(m["content"]))


def test_a_run_that_spans_two_messages_is_cut_where_it_touches_the_edge(_table_limits):
    half = "\n" * (_LIMIT - 20)
    refusal = guard.parse_refusal(_refusal_of("deepseek-v3", "a" + half + half + "b"))
    messages = [{"content": "a" + half}, {"content": half + "b"},
                {"content": "x" + half + "y"}]
    changed, offenders = fit_messages(messages, refusal, protected=set(), reserve=0)
    assert changed == [0, 1] and not offenders, "a run inside a message cannot go on past it"
    guard.check_text("deepseek-v3", messages[0]["content"] + messages[1]["content"])


def test_a_refusal_no_message_explains_changes_none():
    refusal = guard.Refusal("run", 64, 90, "letters")
    messages = [{"content": "a" * 40 + " b"}, {"content": "see " + "a" * 40 + " b"}]
    before = [dict(m) for m in messages]
    changed, offenders = fit_messages(messages[1:], refusal, protected=set(), reserve=0)
    assert (changed, offenders) == ([], [])
    assert messages[1:] == before[1:]
