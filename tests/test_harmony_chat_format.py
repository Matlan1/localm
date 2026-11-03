# SPDX-License-Identifier: AGPL-3.0-or-later
"""A gpt-oss GGUF chat prompt is rendered in the harmony format.

llama.cpp's built-in ``llama_chat_apply_template`` writes a gpt-oss
conversation without the channel on earlier assistant replies
(``<|start|>assistant<|message|>...<|return|>``), so from the second turn the
model starts its reply outside the channel format and its reasoning reaches the
answer. These templates are rendered by ``_chat_formats.format_harmony``; the
expected strings are the official template's output for the same conversations
(tests/fixtures/harmony_chat_templates).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import no_think_prompt
from localm.inference.backends.llamacpp._chat_formats import (
    format_harmony,
    is_gemma4_template,
    is_harmony_template,
)
from localm.inference.backends.llamacpp.llama import (
    _content_spans_in_prompt,
    _render_template,
)

_LLAMA_API = "localm.inference.backends.llamacpp.llama.api"
_FIXTURES = Path(__file__).parent / "fixtures" / "harmony_chat_templates"
_GOLDEN = json.loads((_FIXTURES / "expected.json").read_text(encoding="utf-8"))
_TEMPLATE = (_FIXTURES / "gpt-oss-20b.jinja").read_text(encoding="utf-8")
_EXPECTED = _GOLDEN["expected"]["gpt-oss-20b.jinja"]
_TODAY = _GOLDEN["today"]

GEMMA3 = "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
CHATML = "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"


def _pairs(case):
    return [tuple(pair) for pair in _GOLDEN["cases"][case]]


def test_official_template_is_detected():
    assert is_harmony_template(_TEMPLATE)
    assert not is_gemma4_template(_TEMPLATE)


@pytest.mark.parametrize("tmpl", [None, "", GEMMA3, CHATML,
                                  "{{ '<|start|>developer<|message|>' }}",
                                  "{{ '<|channel|>final<|message|>' }}"])
def test_other_templates_are_not_detected(tmpl):
    assert not is_harmony_template(tmpl)


@pytest.mark.parametrize("case", sorted(_GOLDEN["cases"]))
def test_render_matches_the_official_template_output(case):
    assert format_harmony(_pairs(case), today=_TODAY) == _EXPECTED[case]


def test_earlier_assistant_reply_is_a_final_channel_message():
    prompt = format_harmony(_pairs("multi_turn"), today=_TODAY)
    assert "<|start|>assistant<|channel|>final<|message|>Hello! How can I help?<|end|>" in prompt
    assert "<|start|>assistant<|message|>" not in prompt
    assert "<|return|>" not in prompt
    assert prompt.endswith("<|start|>assistant")


def test_assistant_text_is_kept_verbatim():
    text = "<think>\nreasoning\n</think>\nthe answer"
    prompt = format_harmony([("user", "q"), ("assistant", text), ("user", "q2")], today=_TODAY)
    assert "<|channel|>final<|message|>" + text + "<|end|>" in prompt


def test_empty_system_message_keeps_its_developer_message():
    prompt = format_harmony([("system", ""), ("user", "hi")], today=_TODAY)
    assert "<|start|>developer<|message|># Instructions\n\n\n\n<|end|><|start|>user" in prompt


def test_user_reasoning_markers_are_kept():
    prompt = format_harmony([("user", "what is <think>?")], today=_TODAY)
    assert "<|start|>user<|message|>what is <think>?<|end|>" in prompt


def test_later_system_message_keeps_its_role():
    prompt = format_harmony([("system", "s"), ("user", "u"), ("system", "again")], today=_TODAY)
    assert "<|start|>system<|message|>again<|end|><|start|>assistant" in prompt


def test_date_defaults_to_the_local_date():
    assert format_harmony([("user", "hi")]).count(
        "Current date: " + datetime.date.today().isoformat() + "\n") == 1


def test_no_messages_is_the_system_message_and_the_generation_prompt():
    prompt = format_harmony([], today=_TODAY)
    assert prompt.startswith("<|start|>system<|message|>")
    assert prompt.endswith("<|end|><|start|>assistant")


def test_render_template_routes_through_harmony_without_llama_cpp():
    apply = MagicMock(return_value=-1)
    with patch(_LLAMA_API + ".llama_model_chat_template", return_value=_TEMPLATE), \
         patch(_LLAMA_API + ".llama_chat_apply_template", apply):
        prompt, reason = _render_template(object(), [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": [{"type": "text", "text": "again"}]}])
    apply.assert_not_called()
    assert reason is None
    assert "<|start|>assistant<|channel|>final<|message|>hello<|end|>" in prompt
    assert prompt.endswith("<|start|>user<|message|>again<|end|><|start|>assistant")


def test_no_think_prompt_prefills_the_final_channel_on_a_second_turn():
    prompt = format_harmony(_pairs("multi_turn"), today=_TODAY)
    assert no_think_prompt(prompt, _TEMPLATE) == prompt + "<|channel|>final<|message|>"


def test_none_content_renders_as_empty_text():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None},
                {"role": "user", "content": "again"}]
    with patch(_LLAMA_API + ".llama_model_chat_template", return_value=_TEMPLATE):
        prompt, reason = _render_template(object(), messages)
        spans = _content_spans_in_prompt(object(), messages, prompt, reason)
    assert "None" not in prompt
    assert spans is not None
    assert [prompt[a:b] for a, b, _lead in spans] == ["hi", "", "again"]


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": ""}, {"role": "user", "content": "<tool_response>x</tool_response>"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>r</think>\nans"},
     {"role": "user", "content": "<tool_response>x</tool_response>"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
     {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}],
])
def test_every_message_content_is_locatable_in_the_prompt(messages):
    with patch(_LLAMA_API + ".llama_model_chat_template", return_value=_TEMPLATE):
        prompt, reason = _render_template(object(), messages)
        spans = _content_spans_in_prompt(object(), messages, prompt, reason)
    assert spans is not None
    assert [prompt[a:b] for a, b, _lead in spans] == [m["content"] for m in messages]
