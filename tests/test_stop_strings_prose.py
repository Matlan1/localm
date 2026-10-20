# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reply that writes an end-of-turn marker as text ("<|im_end|>" in an
explanation of chat templates) keeps going; the reply ends only on the
marker's own token, or on a text marker the model's chat template uses."""

from __future__ import annotations

import pytest

from localm.inference.backends.llamacpp.llama import (
    _STOP_STRINGS,
    LlamaCpp,
    model_stop_strings,
    split_stop_strings,
)

# A ChatML-style template, as Qwen ships it: the only markers it uses are
# <|im_start|> and <|im_end|>.
CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n")
# A Mistral v1 style template that writes [/INST] as plain text.
MISTRAL_V1_TEMPLATE = "{{ bos_token }}[INST] {{ message.content }} [/INST]"

IM_END = 151645


class _Tok:
    """A vocabulary where <|im_end|> and <|endoftext|> are single control
    tokens and every other character is its own token (its code point)."""

    SPECIAL = {"<|im_end|>": IM_END, "<|endoftext|>": 151643}

    def __init__(self, fail: bool = False):
        self.fail = fail

    def encode(self, text, add_bos=False):
        if self.fail:
            raise RuntimeError("tokenizer unavailable")
        if text in self.SPECIAL:
            return [self.SPECIAL[text]]
        return [ord(c) for c in text]

    def is_eog(self, token):
        return False

    def token_to_piece_bytes(self, token):
        for s, i in self.SPECIAL.items():
            if i == token:
                return s.encode()
        return chr(token).encode("utf-8")


def _decode(text_tokens, stop_ids, stops):
    model = LlamaCpp.__new__(LlamaCpp)
    model._tokenizer = _Tok()
    model._stop_token_ids = stop_ids
    model._stop_strings = stops
    return "".join(LlamaCpp._decode_stream(model, iter(text_tokens)))


def _chars(s):
    return [ord(c) for c in s]


def test_single_token_markers_match_by_id_and_unused_text_markers_are_dropped():
    ids, text = split_stop_strings(_Tok(), _STOP_STRINGS, CHATML_TEMPLATE)
    assert IM_END in ids
    assert text == frozenset(), "no multi-token marker appears in a ChatML template"


def test_a_text_marker_the_template_uses_stays_a_text_stop():
    ids, text = split_stop_strings(_Tok(), _STOP_STRINGS, MISTRAL_V1_TEMPLATE)
    assert "[/INST]" in text
    assert "</s>" not in text


def test_without_a_template_every_multi_token_marker_stays_a_text_stop():
    ids, text = split_stop_strings(_Tok(), _STOP_STRINGS, None)
    assert text == _STOP_STRINGS - {"<|im_end|>", "<|endoftext|>"}


def test_a_tokenizer_failure_keeps_every_marker_as_text():
    ids, text = split_stop_strings(_Tok(fail=True), _STOP_STRINGS, CHATML_TEMPLATE)
    assert ids == frozenset()
    assert text == _STOP_STRINGS


def test_a_marker_written_as_prose_does_not_end_the_reply():
    ids, text = split_stop_strings(_Tok(), model_stop_strings(_Tok()), CHATML_TEMPLATE)
    prose = "In ChatML each message ends with <|im_end|>, and Llama 2 used </s>. THE END."
    assert _decode(_chars(prose), ids, text) == prose


def test_the_marker_token_still_ends_the_reply():
    ids, text = split_stop_strings(_Tok(), model_stop_strings(_Tok()), CHATML_TEMPLATE)
    tokens = _chars("Done.") + [IM_END] + _chars("user: next turn")
    assert _decode(tokens, ids, text) == "Done."


@pytest.mark.parametrize("template", [MISTRAL_V1_TEMPLATE, None])
def test_a_text_stop_still_cuts_where_the_template_uses_it(template):
    ids, text = split_stop_strings(_Tok(), model_stop_strings(_Tok()), template)
    assert _decode(_chars("Answer. [/INST] fake turn"), ids, text) == "Answer. "
