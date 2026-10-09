# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.voice detailed transcription: the segment/word result, the worker
request shape, and the unchanged text-only path."""

import importlib.util
import queue
from types import SimpleNamespace

import pytest

from localm import voice


class _Seg:
    def __init__(self, i, start, end, text, words=None, temperature=0.0):
        self.id, self.seek, self.start, self.end = i, 0, start, end
        self.text, self.tokens = text, [11, 22]
        self.avg_logprob, self.compression_ratio = -0.3, 1.4
        self.no_speech_prob, self.words, self.temperature = 0.02, words, temperature


class _Word:
    def __init__(self, word, start, end, probability=0.9):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class _Model:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        segs = [_Seg(1, 0.0, 1.23456, " Hello there. ",
                     [_Word(" Hello", 0.0, 0.5), _Word(" there.", 0.5, 1.2)]),
                _Seg(2, 1.5, 2.0, "Bye.", None, None)]
        return iter(segs), SimpleNamespace(language="en")


AUDIO = [0.0] * 32000        # two seconds at 16 kHz


class TestTranscribeDetailed:
    def test_rows_are_plain_data_with_stripped_text(self):
        text, detail = voice._transcribe_detailed(
            _Model(), AUDIO, None, {"word_timestamps": True})
        assert text == "Hello there. Bye."
        assert detail["language"] == "en" and detail["duration"] == 2.0
        first, second = detail["segments"]
        assert (first["id"], second["id"]) == (0, 1)
        assert first["text"] == "Hello there." and first["end"] == 1.235
        assert first["tokens"] == [11, 22] and first["temperature"] == 0.0
        assert first["words"] == [
            {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " there.", "start": 0.5, "end": 1.2, "probability": 0.9}]
        assert second["words"] == [] and second["temperature"] == 0.0

    def test_unset_options_are_not_passed_to_whisper(self):
        m = _Model()
        voice._transcribe_detailed(
            m, AUDIO, None, {"prompt": None, "temperature": None,
                             "word_timestamps": False})
        assert m.kwargs == {"language": None, "word_timestamps": False}

    def test_set_options_are_passed_to_whisper(self):
        m = _Model()
        voice._transcribe_detailed(
            m, AUDIO, "fr", {"prompt": "glossary", "temperature": 0.4,
                             "word_timestamps": True})
        assert m.kwargs == {"language": "fr", "word_timestamps": True,
                            "initial_prompt": "glossary", "temperature": 0.4}

    def test_zero_temperature_keeps_whispers_fallback_schedule(self):
        m = _Model()
        voice._transcribe_detailed(m, AUDIO, None, {"temperature": 0.0})
        assert "temperature" not in m.kwargs

    def test_positive_temperature_fixes_the_temperature(self):
        m = _Model()
        voice._transcribe_detailed(m, AUDIO, None, {"temperature": 0.01})
        assert m.kwargs["temperature"] == 0.01

    def test_detail_is_json_plain_data_so_it_crosses_the_process_boundary(self):
        import json
        _text, detail = voice._transcribe_detailed(_Model(), AUDIO, None, {})
        assert json.loads(json.dumps(detail)) == detail


class TestUnknownLanguage:
    @pytest.mark.parametrize("code", [None, "", "en", "fr", "yue"])
    def test_known_or_empty_is_accepted(self, code):
        assert voice._unknown_language(code) == ""

    def test_unknown_code_is_named_in_the_refusal(self):
        if importlib.util.find_spec("faster_whisper") is None:
            pytest.skip("faster-whisper is not installed")
        msg = voice._unknown_language("english")
        assert "'english'" in msg and "ISO 639-1" in msg


class _AliveProc:
    exitcode = None

    def is_alive(self):
        return True


@pytest.fixture
def fake_worker(monkeypatch):
    sent, replies = queue.Queue(), queue.Queue()
    monkeypatch.setattr(voice, "_ensure_worker", lambda: None)
    monkeypatch.setattr(voice, "_proc", _AliveProc())
    monkeypatch.setattr(voice, "_req_q", sent)
    monkeypatch.setattr(voice, "_resp_q", replies)
    return SimpleNamespace(sent=sent, replies=replies)


class TestWorkerProtocol:
    def test_text_only_request_keeps_the_five_element_shape(self, fake_worker):
        fake_worker.replies.put(("ok", "hello"))
        assert voice._run_in_worker(b"a", "base", None, 5.0) == "hello"
        assert len(fake_worker.sent.get_nowait()) == 5

    def test_detailed_request_appends_the_options(self, fake_worker):
        fake_worker.replies.put(("ok", "hello", {"language": "en"}))
        opts = {"prompt": None, "temperature": None, "word_timestamps": False}
        text, detail = voice._run_in_worker_detailed(
            b"a", "base", None, 5.0, opts=opts)
        assert (text, detail) == ("hello", {"language": "en"})
        request = fake_worker.sent.get_nowait()
        assert len(request) == 6 and request[5] == opts

    def test_text_only_reply_has_no_detail(self, fake_worker):
        fake_worker.replies.put(("ok", "hello"))
        assert voice._run_in_worker_detailed(b"a", "base", None, 5.0) == ("hello", None)

    def test_silence_is_empty_text_for_the_detailed_path(self, fake_worker):
        fake_worker.replies.put(("ok", "", {"segments": []}))
        text, detail = voice._run_in_worker_detailed(
            b"a", "base", None, 5.0, opts={})
        assert text == "" and detail == {"segments": []}

    def test_silence_still_raises_no_speech_for_the_text_path(self, fake_worker):
        fake_worker.replies.put(("ok", ""))
        with pytest.raises(voice.VoiceError) as e:
            voice._run_in_worker(b"a", "base", None, 5.0)
        assert e.value.code == "no-speech"

    def test_bad_request_reply_becomes_a_coded_error_with_the_worker_message(
            self, fake_worker):
        fake_worker.replies.put(("error", "bad-request", "'zz' is not supported"))
        with pytest.raises(voice.VoiceError) as e:
            voice._run_in_worker_detailed(b"a", "base", "zz", 5.0, opts={})
        assert e.value.code == "bad-request"
        assert str(e.value) == "'zz' is not supported"


class TestTranscribeDetailedEntryPoint:
    def test_empty_upload_is_refused_before_any_worker(self, monkeypatch):
        monkeypatch.setattr(voice, "_run_in_worker_detailed",
                            lambda *a, **k: pytest.fail("worker must not run"))
        with pytest.raises(voice.VoiceError) as e:
            voice.transcribe_detailed(b"")
        assert e.value.code == "empty"

    def test_missing_package_is_reported_before_any_worker(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
        monkeypatch.setattr(voice, "_run_in_worker_detailed",
                            lambda *a, **k: pytest.fail("worker must not run"))
        with pytest.raises(voice.VoiceError) as e:
            voice.transcribe_detailed(b"audio")
        assert e.value.code == "needs-faster-whisper"

    def test_result_merges_text_and_detail_and_forwards_the_options(
            self, monkeypatch):
        seen = {}

        def _fake(data, name, language, timeout, *, opts, local_files_only,
                  blocked_reason):
            seen.update(data=data, name=name, language=language, opts=opts,
                        lfo=local_files_only)
            return "hi", {"language": "en", "duration": 1.0, "segments": []}

        monkeypatch.setattr(voice, "_stt_plan", lambda lang: {
            "name": "tiny", "language": lang, "timeout": 5.0,
            "local_files_only": True, "blocked_reason": None})
        monkeypatch.setattr(voice, "_run_in_worker_detailed", _fake)
        out = voice.transcribe_detailed(b"audio", "de", prompt="p",
                                        temperature=0.3, word_timestamps=True)
        assert out == {"text": "hi", "language": "en", "duration": 1.0,
                       "segments": []}
        assert seen["language"] == "de" and seen["name"] == "tiny"
        assert seen["opts"] == {"prompt": "p", "temperature": 0.3,
                                "word_timestamps": True}
