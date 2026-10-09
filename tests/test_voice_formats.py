# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.voice_formats: SRT, WebVTT and verbose_json rendering."""

import pytest

from localm import voice_formats as vf


def _seg(i, start, end, text, words=None):
    return {"id": i, "seek": 0, "start": start, "end": end, "text": text,
            "tokens": [1, 2], "temperature": 0.0, "avg_logprob": -0.25,
            "compression_ratio": 1.5, "no_speech_prob": 0.01,
            "words": words or []}


SEGS = [_seg(0, 0.0, 1.5, "Hello world."), _seg(1, 2.0, 3.25, "Second line.")]


class TestSrt:
    def test_exact_output(self):
        assert vf.to_srt(SEGS) == (
            "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n"
            "\n"
            "2\n00:00:02,000 --> 00:00:03,250\nSecond line.\n")

    def test_no_segments_is_empty(self):
        assert vf.to_srt([]) == ""

    def test_hour_and_millisecond_rounding(self):
        out = vf.to_srt([_seg(0, 3661.0004, 3661.9996, "x")])
        assert "01:01:01,000 --> 01:01:02,000" in out

    def test_blank_line_inside_text_cannot_end_the_cue(self):
        out = vf.to_srt([_seg(0, 0, 1, "one\n\ntwo\r\nthree")])
        assert out == "1\n00:00:00,000 --> 00:00:01,000\none two three\n"

    def test_negative_time_clamps_to_zero(self):
        assert "00:00:00,000" in vf.to_srt([_seg(0, -1.0, 1.0, "x")])


class TestVtt:
    def test_exact_output(self):
        assert vf.to_vtt(SEGS) == (
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:01.500\nHello world.\n"
            "\n"
            "00:00:02.000 --> 00:00:03.250\nSecond line.\n")

    def test_no_segments_is_just_the_header(self):
        assert vf.to_vtt([]) == "WEBVTT\n"

    def test_markup_characters_are_escaped(self):
        out = vf.to_vtt([_seg(0, 0, 1, "a <b> & c --> d")])
        assert "a &lt;b&gt; &amp; c --&gt; d" in out

    def test_uses_dot_not_comma_for_milliseconds(self):
        assert "," not in vf.to_vtt(SEGS)


class TestVerboseJson:
    def _detail(self):
        words = [{"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                 {"word": " world.", "start": 0.6, "end": 1.4, "probability": 0.8}]
        return {"text": "Hello world. Second line.", "language": "en",
                "duration": 3.5,
                "segments": [_seg(0, 0.0, 1.5, "Hello world.", words),
                             _seg(1, 2.0, 3.25, "Second line.")]}

    def test_segments_only(self):
        body = vf.to_verbose_json(self._detail(), include_segments=True,
                                  include_words=False)
        assert body["task"] == "transcribe"
        assert body["language"] == "english"
        assert body["duration"] == 3.5
        assert body["text"] == "Hello world. Second line."
        assert "words" not in body
        assert list(body["segments"][0]) == [
            "id", "seek", "start", "end", "text", "tokens", "temperature",
            "avg_logprob", "compression_ratio", "no_speech_prob"]

    def test_words_only_omits_segments(self):
        body = vf.to_verbose_json(self._detail(), include_segments=False,
                                  include_words=True)
        assert "segments" not in body
        assert body["words"] == [
            {"word": " Hello", "start": 0.0, "end": 0.5},
            {"word": " world.", "start": 0.6, "end": 1.4}]

    def test_both(self):
        body = vf.to_verbose_json(self._detail(), include_segments=True,
                                  include_words=True)
        assert len(body["words"]) == 2 and len(body["segments"]) == 2


class TestLanguageNames:
    def test_known_and_unknown(self):
        assert vf.language_name("fr") == "french"
        assert vf.language_name("zz") == "zz"
        assert vf.language_name("") == ""
        assert vf.language_name(None) == ""

    def test_covers_every_language_faster_whisper_can_report(self):
        tokenizer = pytest.importorskip("faster_whisper.tokenizer")
        missing = set(tokenizer._LANGUAGE_CODES) - set(vf.LANGUAGE_NAMES)
        assert not missing, sorted(missing)
