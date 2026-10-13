# SPDX-License-Identifier: AGPL-3.0-or-later
"""The prompt text a transformers processor renders for a conversation.

SmolVLM's ``apply_chat_template`` seeds default frame-sampling arguments into
``**kwargs`` unless ``processor_kwargs`` is non-empty, and transformers then logs a
warning on every render. ``_render_processor_prompt`` keeps that warning out of the
log without changing the rendered text.
"""

from __future__ import annotations

import logging

import pytest

from localm.inference.backends._hf_worker import _render_processor_prompt

_MESSAGES = [{"role": "user",
              "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}:"
    "{% for c in m['content'] %}{% if c['type']=='image' %}<image>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}")
_WARNING = "have to be in `processor_kwargs` dict"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def smolvlm_processor():
    pytest.importorskip("transformers")
    pytest.importorskip("num2words")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
    from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
    from transformers.models.smolvlm.video_processing_smolvlm import SmolVLMVideoProcessor

    tok = Tokenizer(models.WordLevel({"<unk>": 0, "hi": 1}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    fast.add_special_tokens({"additional_special_tokens": [
        "<image>", "<end_of_utterance>", "<fake_token_around_image>", "<global-img>"]})
    return SmolVLMProcessor(SmolVLMImageProcessor(), fast, SmolVLMVideoProcessor(),
                            chat_template=_TEMPLATE)


@pytest.fixture
def transformers_log():
    handler = _Capture()
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


class TestRealSmolVLMProcessor:
    def test_render_logs_no_kwargs_warning(self, smolvlm_processor, transformers_log):
        text = _render_processor_prompt(smolvlm_processor, _MESSAGES)
        assert text == "user:<image>hiassistant:"
        assert not [m for m in transformers_log if _WARNING in m], transformers_log

    def test_direct_call_does_log_the_warning(self, smolvlm_processor, transformers_log):
        smolvlm_processor.apply_chat_template(
            _MESSAGES, tokenize=False, add_generation_prompt=True)
        assert [m for m in transformers_log if _WARNING in m], (
            "the warning this helper avoids no longer fires on a direct call")

    def test_generation_prompt_flag_is_honoured(self, smolvlm_processor):
        text = _render_processor_prompt(
            smolvlm_processor, _MESSAGES, add_generation_prompt=False)
        assert text == "user:<image>hi"


class TestArgumentsPassedToAnyProcessor:
    def test_only_processor_kwargs_and_template_flags_are_passed(self):
        seen = {}

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                seen.update(kwargs)
                return "rendered"

        assert _render_processor_prompt(Processor(), _MESSAGES) == "rendered"
        assert seen == {"processor_kwargs": {"add_special_tokens": False},
                        "tokenize": False, "add_generation_prompt": True}
