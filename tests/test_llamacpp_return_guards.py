# SPDX-License-Identifier: AGPL-3.0-or-later
"""Defensive return-code guards on the llama.cpp text path.

Two ctypes call sites trusted the native return code after their resize-and-
retry: token_to_piece_bytes sliced buf.raw[:n] with a possibly still-negative
n (garbage bytes instead of an error), and _apply_model_template treated a
zero-length render as success (an empty prompt instead of the ChatML
fallback). A correct runtime never hits either, but a genuinely bad decode or
template must surface or fall back, not silently produce garbage (rule 5)."""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import (
    _Tokenizer,
    _apply_model_template,
    _format_chatml,
)

_LLAMA_API = "localm.inference.backends.llamacpp.llama.api"


def _tokenizer():
    tok = _Tokenizer.__new__(_Tokenizer)
    tok._vocab = MagicMock()
    tok._ctx = None
    return tok


class TestTokenToPieceGuard:
    def test_negative_after_retry_raises(self):
        mock_api = MagicMock()
        mock_api.llama_token_to_piece.return_value = -7   # first call AND retry fail
        with patch(_LLAMA_API, mock_api):
            with pytest.raises(RuntimeError, match=r"token 123.*-7"):
                _tokenizer().token_to_piece_bytes(123)
        assert mock_api.llama_token_to_piece.call_count == 2

    def test_first_call_success_unchanged(self):
        mock_api = MagicMock()

        def fake_piece(vocab, token, buf, size, lstrip, special):
            buf[:5] = b"hello"
            return 5

        mock_api.llama_token_to_piece.side_effect = fake_piece
        with patch(_LLAMA_API, mock_api):
            assert _tokenizer().token_to_piece_bytes(1) == b"hello"

    def test_retry_success_unchanged(self):
        mock_api = MagicMock()
        calls = {"n": 0}

        def fake_piece(vocab, token, buf, size, lstrip, special):
            calls["n"] += 1
            if calls["n"] == 1:
                return -300                       # buffer too small, need 300
            buf[:3] = b"abc"
            return 3

        mock_api.llama_token_to_piece.side_effect = fake_piece
        with patch(_LLAMA_API, mock_api):
            assert _tokenizer().token_to_piece_bytes(1) == b"abc"


class TestApplyTemplateEmptyRenderGuard:
    _MSGS = [{"role": "user", "content": "hi"}]

    def _mock_api(self, apply_results):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = "{{ template }}"
        mock_api.llama_chat_apply_template.side_effect = list(apply_results)
        return mock_api

    def test_zero_render_first_call_falls_back_to_chatml(self):
        mock_api = self._mock_api([0])
        with patch(_LLAMA_API, mock_api):
            out = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)

    def test_negative_first_call_still_falls_back(self):
        mock_api = self._mock_api([-1])
        with patch(_LLAMA_API, mock_api):
            out = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)

    def test_zero_render_after_realloc_falls_back_to_chatml(self):
        # First call: needed (10000) exceeds the initial buffer, forcing the
        # realloc + retry; the retry then renders nothing.
        mock_api = self._mock_api([10000, 0])
        with patch(_LLAMA_API, mock_api):
            out = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        assert mock_api.llama_chat_apply_template.call_count == 2

    def test_realloc_success_unchanged(self):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = "{{ template }}"
        calls = {"n": 0}

        def fake_apply(tmpl, arr, n, add_assistant, buf, size):
            calls["n"] += 1
            if calls["n"] == 1:
                return 10000                      # too big for the initial buffer
            rendered = b"RENDERED PROMPT"
            buf[:len(rendered)] = rendered
            return len(rendered)

        mock_api.llama_chat_apply_template.side_effect = fake_apply
        with patch(_LLAMA_API, mock_api):
            out = _apply_model_template(1, self._MSGS)
        assert out == "RENDERED PROMPT"

    def test_no_embedded_template_falls_back(self):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        with patch(_LLAMA_API, mock_api):
            out = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        mock_api.llama_chat_apply_template.assert_not_called()


class TestApplyTemplateFallbackIsSurfaced:
    """RAG-VISION-1: llama_chat_apply_template is not a real Jinja engine - it
    pattern-matches against ~54 hardcoded template signatures in llama.cpp and
    returns -1 for anything it does not recognize (e.g. moondream2 and other
    non-mainstream VLMs). The ChatML fallback used to be completely silent -
    no log, no error - so a model fed an out-of-distribution prompt it was
    never fine-tuned on gave no signal why its output degenerated. Confirmed
    root cause of moondream2 producing garbage/hallucinated RAG image
    descriptions during a RELEASE.md verification pass."""

    _MSGS = [{"role": "user", "content": "hi"}]

    def test_no_embedded_template_logs_a_warning(self, caplog):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        with caplog.at_level("WARNING", logger="localm"):
            with patch(_LLAMA_API, mock_api):
                _apply_model_template(1, self._MSGS)
        assert any("chat template not recognized" in r.message for r in caplog.records)

    def test_unrecognized_template_logs_a_warning(self, caplog):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = "{{ some unrecognized template }}"
        mock_api.llama_chat_apply_template.return_value = -1
        with caplog.at_level("WARNING", logger="localm"):
            with patch(_LLAMA_API, mock_api):
                _apply_model_template(1, self._MSGS)
        assert any("chat template not recognized" in r.message for r in caplog.records)

    def test_a_recognized_template_logs_nothing(self, caplog):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = "{{ chatml }}"

        def fake_apply(tmpl, arr, n, add_assistant, buf, size):
            rendered = b"RENDERED PROMPT"
            buf[:len(rendered)] = rendered
            return len(rendered)

        mock_api.llama_chat_apply_template.side_effect = fake_apply
        with caplog.at_level("WARNING", logger="localm"):
            with patch(_LLAMA_API, mock_api):
                out = _apply_model_template(1, self._MSGS)
        assert out == "RENDERED PROMPT"
        assert not any("chat template not recognized" in r.message for r in caplog.records)
