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
    def test_harmony_channel_tags_removed(self):
        text = "<|channel|>analysis<|message|>Reasoning here.<|return|>"
        assert _scrub([text]) == "Reasoning here."

    def test_mangled_channel_tags_removed(self):
        # The exact garbage observed in the bug report
        text = "<|channel>thought\n<channel|>Good morning! How can I help?"
        assert _scrub([text]) == "\nGood morning! How can I help?"

    def test_unused_tokens_removed(self):
        assert _scrub(["before <unused2> after"]) == "before  after"

    def test_truncated_unused_token_at_stream_end(self):
        # The crash trace ended mid-token: "<unused2" with no closing ">"
        assert _scrub(["text then <unused2"]) == "text then "

    def test_marker_straddling_chunks(self):
        pieces = ["safe text <|chan", "nel|>thought", " more text"]
        assert _scrub(pieces) == "safe text  more text"

    def test_repeated_markers_all_removed(self):
        pieces = ["<|channel>thought\n<channel|>"] * 5 + ["hello"]
        assert _scrub(pieces) == "\n\n\n\n\nhello"

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

    def test_scrub_bypassed_in_debug_mode(self, tmp_path):
        """_decode_stream shows raw markers when debug is on."""
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        from unittest.mock import MagicMock

        llm = LlamaCpp.__new__(LlamaCpp)
        llm._tokenizer = MagicMock()
        llm._tokenizer.token_to_piece.side_effect = \
            lambda t: {1: "<|channel|>", 2: "thought", 3: " hi"}[t]
        try:
            raw_off = "".join(llm._decode_stream(iter([1, 2, 3])))
            assert raw_off == " hi"          # scrubbed in normal mode

            debuglog.enable_debug()
            raw_on = "".join(llm._decode_stream(iter([1, 2, 3])))
            assert raw_on == "<|channel|>thought hi"   # raw in debug mode
        finally:
            llm._tokenizer = None
            llm._model_ptr = None
            llm._ctx_ptr = None
