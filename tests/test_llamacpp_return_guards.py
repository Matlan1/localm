# SPDX-License-Identifier: AGPL-3.0-or-later
"""Defensive return-code guards on the llama.cpp text path.

Two ctypes call sites must not trust the native return code after their
resize-and-retry: token_to_piece_bytes must not slice buf.raw[:n] with a
still-negative n (garbage bytes instead of an error), and
_apply_model_template must not treat a zero-length render as success (an empty
prompt instead of the ChatML fallback). A correct runtime never hits either,
but a genuinely bad decode or template must surface or fall back, not silently
produce garbage."""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import (
    LlamaCpp,
    _Tokenizer,
    _apply_model_template,
    _format_chatml,
)
from tests._bare_llama import make_bare_llama

import localm.inference.backends.llamacpp._api as _api

_LLAMA_API = "localm.inference.backends.llamacpp.llama.api"
_API = "localm.inference.backends.llamacpp._api"
_APPLY_TEMPLATE = "localm.inference.backends.llamacpp.llama._apply_model_template"


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
            out, reason = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        assert reason

    def test_negative_first_call_still_falls_back(self):
        mock_api = self._mock_api([-1])
        with patch(_LLAMA_API, mock_api):
            out, reason = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        assert reason

    def test_zero_render_after_realloc_falls_back_to_chatml(self):
        # First call: needed (10000) exceeds the initial buffer, forcing the
        # realloc + retry; the retry then renders nothing.
        mock_api = self._mock_api([10000, 0])
        with patch(_LLAMA_API, mock_api):
            out, reason = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        assert reason
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
            out, reason = _apply_model_template(1, self._MSGS)
        assert out == "RENDERED PROMPT"
        assert reason is None

    def test_no_embedded_template_falls_back(self):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        with patch(_LLAMA_API, mock_api):
            out, reason = _apply_model_template(1, self._MSGS)
        assert out == _format_chatml(self._MSGS)
        assert reason
        mock_api.llama_chat_apply_template.assert_not_called()


class TestApplyTemplateFallbackIsSurfaced:
    """llama_chat_apply_template is not a real Jinja engine - it
    pattern-matches against ~54 hardcoded template signatures in llama.cpp and
    returns -1 for anything it does not recognize (e.g. moondream2 and other
    non-mainstream VLMs). The ChatML fallback must not be silent: a model fed
    an out-of-distribution prompt it was never fine-tuned on otherwise gives no
    signal why its output degenerated.

    A debug-log warning alone does not reach the user, because debuglog.logger's
    file handler only exists under --debug. _apply_model_template also RETURNS
    the reason (asserted below) so a caller can propagate it to a channel
    visible without --debug; tests/test_chatml_fallback_visibility.py covers the
    full worker -> runner -> backend -> console path this return value feeds."""

    _MSGS = [{"role": "user", "content": "hi"}]

    def test_no_embedded_template_logs_a_warning(self, caplog):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        with caplog.at_level("WARNING", logger="localm"):
            with patch(_LLAMA_API, mock_api):
                _prompt, reason = _apply_model_template(1, self._MSGS)
        assert any("chat template not recognized" in r.message for r in caplog.records)
        assert reason, "the reason must also be RETURNED, not only logged"

    def test_unrecognized_template_logs_a_warning(self, caplog):
        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = "{{ some unrecognized template }}"
        mock_api.llama_chat_apply_template.return_value = -1
        with caplog.at_level("WARNING", logger="localm"):
            with patch(_LLAMA_API, mock_api):
                _prompt, reason = _apply_model_template(1, self._MSGS)
        assert any("chat template not recognized" in r.message for r in caplog.records)
        assert reason, "the reason must also be RETURNED, not only logged"

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
                out, reason = _apply_model_template(1, self._MSGS)
        assert out == "RENDERED PROMPT"
        assert reason is None
        assert not any("chat template not recognized" in r.message for r in caplog.records)


class TestFallbackReasonReachesTheInstance:
    """create_chat_completion (and _generate_image, identical two-line shape)
    must RECORD the reason _apply_model_template returns onto
    self.chat_template_fallback_reason: that attribute is the only thing
    GgufWorker's chatml_fallback_reason property reads. No other test covers
    this link - the return-guard tests call _apply_model_template directly, and
    the visibility tests fake the runner below the LlamaCpp layer entirely."""

    def _bare_llama(self) -> LlamaCpp:
        return make_bare_llama(_model_ptr=111, _ctx_ptr=222)

    # Fake-pointer teardown is handled globally by tests/conftest.py's
    # autouse _neutralise_bare_llama_pointers fixture.

    def test_create_chat_completion_records_the_fallback_reason(self):
        llm = self._bare_llama()
        with patch(_APPLY_TEMPLATE,
                   return_value=("hi", "model has no embedded chat template")), \
             patch.object(llm, "_generate", return_value=iter([])):
            llm.create_chat_completion(
                [{"role": "user", "content": "x"}], stream=False)
        assert llm.chat_template_fallback_reason == "model has no embedded chat template"

    def test_create_chat_completion_leaves_it_none_when_template_is_fine(self):
        llm = self._bare_llama()
        with patch(_APPLY_TEMPLATE, return_value=("hi", None)), \
             patch.object(llm, "_generate", return_value=iter([])):
            llm.create_chat_completion(
                [{"role": "user", "content": "x"}], stream=False)
        assert llm.chat_template_fallback_reason is None


class TestMropeProbeDoesNotSwallowANativeFault:
    """llama_model_has_mrope probes an optional export and falls back to GGUF
    metadata when that export is unusable. An access violation is not
    "unusable": it means the model pointer is dead, and the fallback goes on to
    dereference that SAME pointer, so swallowing the first fault only buys a
    second one reported from the generic metadata reader instead (rule 5)."""

    def _lib_whose_rope_type(self, side_effect):
        lib = MagicMock()
        lib.llama_model_rope_type = MagicMock(side_effect=side_effect)
        return lib

    def test_access_violation_reaches_the_caller(self):
        lib = self._lib_whose_rope_type(
            OSError("exception: access violation reading 0x000000000000744F"))
        with patch(_API + ".load_lib", return_value=lib),              patch(_API + ".has_model_meta_api", return_value=True),              patch(_API + ".llama_model_meta_val_str") as meta:
            with pytest.raises(OSError, match="access violation"):
                _api.llama_model_has_mrope(111)
        # The load-bearing half: it must not walk on and dereference the same
        # dead pointer a second time.
        meta.assert_not_called()

    def test_unusable_export_still_falls_back_to_metadata(self):
        """A binding-shape problem genuinely is recoverable, and that fallback
        has to keep working - narrowing the guard must not become removing it."""
        lib = self._lib_whose_rope_type(TypeError("wrong argument type"))
        with patch(_API + ".load_lib", return_value=lib),              patch(_API + ".has_model_meta_api", return_value=True),              patch(_API + ".llama_model_meta_val_str",
                   return_value="qwen2vl") as meta:
            assert _api.llama_model_has_mrope(111) is True
        meta.assert_called()
