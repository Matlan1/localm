# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for internal-marker scrubbing and the debug log module."""

import logging
import os
from unittest.mock import patch

import pytest

from localm import debuglog
from localm.inference.backends.llamacpp.llama import _scrub_stream


def _scrub(pieces):
    return "".join(_scrub_stream(iter(pieces)))


class TestMarkerScrub:
    def test_harmony_channel_tags_become_think(self):
        text = ("<|channel|>analysis<|message|>Reasoning here."
                "<|channel|>final<|message|>The answer.")
        assert _scrub([text]) == "<think>\nReasoning here.\n</think>\nThe answer."

    def test_mangled_channel_tags_become_think(self):
        # The exact garbage observed in the bug report
        text = "<|channel>thought\nhmm<channel|>Good morning! How can I help?"
        assert _scrub([text]) == \
            "<think>\nhmm\n</think>\nGood morning! How can I help?"

    def test_gemma4_dialect_scrubbed(self):
        text = ('<|turn>model\nHe said <|"|>hi<|"|>.<turn|>')
        assert _scrub([text]) == 'He said "hi".'

    def test_gemma4_think_token_removed(self):
        assert _scrub(["<|think|>\nrest"]) == "\nrest"

    def test_gemma4_turn_marker_straddling_pieces(self):
        # '<|turn>' arrives as its own token; the role must not leak
        pieces = ["<|turn>", "model", "\nAnswer text long enough to flush "
                  "the holding buffer entirely, with room to spare."]
        out = _scrub(pieces)
        assert out.lstrip().startswith("Answer text")

    def test_unused_tokens_removed(self):
        assert _scrub(["before <unused2> after"]) == "before  after"

    def test_truncated_unused_token_at_stream_end(self):
        # The crash trace ended mid-token: "<unused2" with no closing ">"
        assert _scrub(["text then <unused2"]) == "text then "

    def test_marker_straddling_chunks(self):
        pieces = ["safe text <|chan", "nel|>thought", " more text"]
        assert _scrub(pieces) == "safe text <think>\n more text"

    def test_repeated_markers_all_normalised(self):
        pieces = ["<|channel>thought\n<channel|>"] * 5 + ["hello"]
        out = _scrub(pieces)
        assert "<|channel" not in out and "<channel|" not in out
        assert out.endswith("hello")

    def test_plain_text_untouched(self):
        text = "The square root: $\\sqrt{2}$ is about 1.414 < 2 and a | pipe."
        assert _scrub([text]) == text

    def test_html_like_text_untouched(self):
        text = "use <div> and <span> tags, x < y, a <= b"
        assert _scrub([text]) == text


class TestDebugLog:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
        monkeypatch.setattr(debuglog, "logs_dir", lambda: tmp_path / "logs")
        saved = list(debuglog.logger.handlers)
        yield
        for h in debuglog.logger.handlers[:]:
            if h not in saved:
                h.close()
                debuglog.logger.removeHandler(h)

    def test_disabled_by_default(self):
        assert debuglog.debug_enabled() is False
        assert debuglog.log_file_path() is None
        assert debuglog.native_stderr_target() is None

    def test_enable_creates_log_and_env(self):
        path = debuglog.enable_debug()
        assert path.is_file()
        assert debuglog.debug_enabled() is True
        assert debuglog.log_file_path() == path
        assert os.environ["LOCALM_DEBUG"] == str(path)
        debuglog.logger.debug("hello from test")
        for h in debuglog.logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
        assert "hello from test" in path.read_text(encoding="utf-8")

    def test_enable_is_idempotent(self):
        first = debuglog.enable_debug()
        second = debuglog.enable_debug()
        assert first == second

    def test_native_stderr_target_appends_to_log(self):
        path = debuglog.enable_debug()
        fd = debuglog.native_stderr_target()
        assert fd is not None
        os.write(fd, b"GGML_ABORT: simulated native crash\n")
        os.close(fd)
        assert "simulated native crash" in path.read_text(encoding="utf-8")

    def test_debug_mode_scrubs_output_and_logs_raw(self, tmp_path):
        """_decode_stream always scrubs; debug mode logs the raw text."""
        import logging as _logging
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        from unittest.mock import MagicMock

        llm = LlamaCpp.__new__(LlamaCpp)
        llm._tokenizer = MagicMock()
        llm._tokenizer.token_to_piece.side_effect = \
            lambda t: {1: "<|channel|>", 2: "thought", 3: " hi"}[t]
        try:
            normal = "".join(llm._decode_stream(iter([1, 2, 3])))
            assert normal == "<think>\n hi"        # normalised in normal mode

            path = debuglog.enable_debug()
            debug = "".join(llm._decode_stream(iter([1, 2, 3])))
            assert debug == normal                 # scrubbed in debug mode too
            for h in debuglog.logger.handlers:
                if isinstance(h, _logging.FileHandler):
                    h.flush()
            # raw, unscrubbed output captured in the debug log
            assert "<|channel|>thought hi" in path.read_text(encoding="utf-8")
        finally:
            llm._tokenizer = None
            llm._model_ptr = None
            llm._ctx_ptr = None
