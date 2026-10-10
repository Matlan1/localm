# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model that cannot take audio must REJECT an input_audio part, never drop it:
at the GGUF and HF backends, at /v1/chat/completions, and in `localm run`."""
import base64
import struct
import unittest
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    AUDIO_UNSUPPORTED_MESSAGE,
    AudioInputError,
    UnsupportedInputError,
    VisionInputError,
    messages_contain_audio,
)
from localm.inference.http_server import create_app


def _wav_b64(n=3200, rate=16000):
    frames = struct.pack("<%dh" % n, *([1000] * n))
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(frames)) + frames)
    return base64.b64encode(b"RIFF" + struct.pack("<I", len(body)) + body).decode()


_AUDIO_MSG = [{"role": "user", "content": [
    {"type": "text", "text": "what is said?"},
    {"type": "input_audio", "input_audio": {"data": _wav_b64(), "format": "wav"}},
]}]
_TEXT_MSG = [{"role": "user", "content": "hello"}]


class TestMessagesContainAudio:
    def test_detection(self):
        assert messages_contain_audio(_AUDIO_MSG) is True
        assert messages_contain_audio(_TEXT_MSG) is False
        assert messages_contain_audio(
            [{"role": "user", "content": ["loose", {"type": "text", "text": "x"}]}]) is False


class TestBackendsRefuse:
    def test_gguf_without_audio_raises_before_the_worker(self):
        from localm.inference.backends.gguf import GgufBackend
        backend = GgufBackend("does-not-need-to-exist.gguf")
        assert backend.supports_audio is False
        with pytest.raises(UnsupportedInputError) as info:
            next(backend.chat_stream(_AUDIO_MSG))
        assert str(info.value) == AUDIO_UNSUPPORTED_MESSAGE

    def test_gguf_supports_audio_comes_from_the_load_response(self):
        from localm.inference.backends.gguf import GgufBackend
        backend = GgufBackend("x.gguf")
        backend._loaded = True
        backend._runner = MagicMock(is_alive=lambda: True)
        backend._supports_audio = True
        assert backend.supports_audio is True
        backend._runner = MagicMock(is_alive=lambda: False)
        assert backend.supports_audio is False

    def test_gguf_with_a_projector_is_worth_loading_to_find_out(self):
        from localm.inference.backends.gguf import GgufBackend
        assert GgufBackend("x.gguf", mmproj_path="p.gguf").can_be_multimodal is True

    def test_hf_backend_refuses_before_load_and_reads_the_load_response(self):
        from localm.inference.backends.hf import HFBackend
        backend = HFBackend("does-not-need-to-exist")
        assert backend.supports_audio is False
        backend._loaded = True
        backend._supports_audio = True
        assert backend.supports_audio is True

    @pytest.mark.parametrize("audio", [True, False])
    def test_worker_load_response_carries_supports_audio(self, monkeypatch, audio):
        from localm.inference.backends.llamacpp import _worker

        class _FakeLlama:
            def __init__(self, **kw):
                self.supports_images = False
                self.supports_audio = audio
        monkeypatch.setattr("localm.inference.backends.llamacpp._loader.load_lib", lambda: None)
        monkeypatch.setattr("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama)
        meta = _worker.GgufWorker("m.gguf", "p.gguf", 512, 0, None, 512).load()
        assert meta["supports_audio"] is audio


def _mock_engine(*, images=False, audio=False, can_be_multimodal=False, loaded=True,
                 mmproj_path=None):
    engine = MagicMock()
    state = {"loaded": loaded}
    engine.load.side_effect = lambda: state.__setitem__("loaded", True)

    def _chat_stream(messages, **kwargs):
        yield "ok"

    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "test-model"
    engine.supports_images = images
    engine.supports_audio = audio
    engine.can_be_multimodal = can_be_multimodal
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine._backend = MagicMock(mmproj_path=mmproj_path, model_path=None)
    type(engine).loaded = property(lambda self: state["loaded"])
    return engine


class TestWorkerKeepsTheErrorType:
    @pytest.mark.parametrize("name", ["AudioInputError", "AudioDecodeUnavailable",
                                      "VisionInputError", "ImageDecodeUnavailable"])
    def test_typed_input_errors_cross_the_worker_boundary(self, name):
        from localm.inference.backends import base
        from localm.inference.backends.llamacpp import _runner
        cls = getattr(base, name)
        assert _runner._INPUT_ERROR_TYPES[name] is cls
        assert issubclass(cls, UnsupportedInputError)

    def test_the_child_tags_the_subclass_and_keeps_serving(self, monkeypatch):
        import queue

        from localm.inference.backends.base import AudioDecodeUnavailable
        from localm.inference.backends.llamacpp import _runner
        for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                     "suppress_native_error_dialogs"):
            monkeypatch.setattr(f"localm._mp_spawn.{name}", lambda: None)
        monkeypatch.setattr("localm.debuglog.attach_child_logging", lambda: None)

        class _Worker:
            def __init__(self, cancel_event=None, **payload):
                self.stream_cancel = None

            def load(self):
                return {}

            def chat_stream(self, on_status=None, **payload):
                raise AudioDecodeUnavailable("needs the voice extra")

            def count_tokens(self, text):
                return 7

            def close(self):
                pass

        monkeypatch.setattr("localm.inference.backends.llamacpp._worker.GgufWorker", _Worker)
        req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        req_q.put(("load", {}))
        req_q.put(("chat_stream", {"messages": []}))
        req_q.put(("count_tokens", "x"))
        req_q.put(None)
        _runner._runner_main(req_q, resp_q, ctrl_q)
        ctrl_q.put(None)
        assert resp_q.get_nowait()[0] == "ok"
        assert resp_q.get_nowait() == ("error", "needs the voice extra",
                                       "AudioDecodeUnavailable")
        assert resp_q.get_nowait() == ("ok", 7)


class TestRoute(unittest.TestCase):
    def _post(self, engine, messages):
        with TestClient(create_app(engine)) as client:
            return client.post("/v1/chat/completions", json={
                "model": "test-model", "messages": messages, "stream": False})

    def test_a_model_without_audio_rejects_audio_with_400(self):
        engine = _mock_engine()
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 400)
        self.assertIn("cannot accept audio", r.json()["detail"])
        engine.load.assert_not_called()
        engine.chat_stream.assert_not_called()

    def test_a_vision_only_projector_rejects_audio_without_blaming_the_projector(self):
        engine = _mock_engine(images=True, can_be_multimodal=True, mmproj_path="p.gguf")
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("failed to load", r.json()["detail"])

    def test_a_projector_that_gave_neither_is_named(self):
        engine = _mock_engine(can_be_multimodal=True, mmproj_path="p.gguf")
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 400)
        self.assertIn("failed to load", r.json()["detail"])

    def test_an_unloaded_model_with_a_projector_is_loaded_before_deciding(self):
        engine = _mock_engine(can_be_multimodal=True, loaded=False, mmproj_path="p.gguf")
        self._post(engine, _AUDIO_MSG)
        engine.load.assert_called_once()

    def test_a_model_that_hears_gets_the_audio(self):
        engine = _mock_engine(audio=True, can_be_multimodal=True)
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 200)
        sent = engine.chat_stream.call_args[0][0]
        part = [p for p in sent[-1]["content"] if p["type"] == "input_audio"][0]
        self.assertEqual(part["input_audio"]["format"], "wav")
        self.assertEqual(part["input_audio"]["data"], _wav_b64())

    def test_a_missing_audio_decoder_is_501(self):
        from localm.inference.backends.base import AudioDecodeUnavailable
        engine = _mock_engine(audio=True, can_be_multimodal=True)

        def _chat_stream(messages, **kwargs):
            raise AudioDecodeUnavailable("needs the voice extra")
            yield  # pragma: no cover

        engine.chat_stream.side_effect = _chat_stream
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 501)
        self.assertIn("voice extra", r.json()["detail"])

    def test_an_image_to_an_audio_only_model_says_so(self):
        engine = _mock_engine(audio=True, can_be_multimodal=True, mmproj_path="p.gguf")
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
        r = self._post(engine, msgs)
        self.assertEqual(r.status_code, 400)
        self.assertIn("reads audio only", r.json()["detail"])

    def test_an_audio_decode_failure_mid_request_is_a_400(self):
        engine = _mock_engine(audio=True, can_be_multimodal=True)

        def _chat_stream(messages, **kwargs):
            raise AudioInputError("The attached audio is not valid base64 data.")
            yield  # pragma: no cover

        engine.chat_stream.side_effect = _chat_stream
        r = self._post(engine, _AUDIO_MSG)
        self.assertEqual(r.status_code, 400)
        self.assertIn("not valid base64", r.json()["detail"])


class TestCli:
    def test_build_user_message_embeds_audio(self, tmp_path):
        from localm.cli.chat import _build_user_message
        f = tmp_path / "clip.WAV"
        f.write_bytes(b"RIFFdata")
        msg = _build_user_message("transcribe", [], [str(f)])
        assert msg["content"][0] == {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(b"RIFFdata").decode(), "format": "wav"}}
        assert msg["content"][-1] == {"type": "text", "text": "transcribe"}
        assert _build_user_message("hi", [], [])["content"] == "hi"

    def test_run_passes_audio_to_the_model(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from localm.audit import SessionMode
        from localm.cli.chat import run
        clip = tmp_path / "c.wav"
        clip.write_bytes(base64.b64decode(_wav_b64()))
        seen = {}
        engine = MagicMock()

        def _chat_stream(messages, on_status=None, **kwargs):
            seen["messages"] = messages
            yield "the quick brown fox"

        engine.chat_stream.side_effect = _chat_stream
        engine.count_tokens.return_value = 4
        monkeypatch.setattr("localm.instances.attach_target",
                            lambda *a, **k: {"base_url": "http://127.0.0.1:1/v1",
                                             "token": "t"})
        monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
        monkeypatch.setattr("localm.inference.http_engine.HttpEngine",
                            MagicMock(return_value=engine))
        monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                            lambda *a, **k: ("loaded", "asr"))
        monkeypatch.setattr("localm.audit.effective_mode",
                            lambda *a, **k: SessionMode.PRIVACY)
        result = CliRunner().invoke(run, ["asr", "-p", "transcribe", "--audio", str(clip)])
        assert result.exit_code == 0, result.output
        parts = seen["messages"][-1]["content"]
        assert parts[0]["type"] == "input_audio"
        assert parts[0]["input_audio"]["data"] == _wav_b64()

    _IMAGE = {"type": "image_url", "image_url": {"url": "x"}}

    @pytest.mark.parametrize("engine_flags,messages,exc,expect", [
        ({}, _AUDIO_MSG, UnsupportedInputError("x"), "cannot accept audio"),
        ({"supports_audio": True}, _AUDIO_MSG, VisionInputError("the projector could "
                                                                "not process this audio"),
         "could not process this audio"),
        ({"supports_audio": True}, _AUDIO_MSG, AudioInputError("too short"), "too short"),
        ({"supports_audio": True, "mmproj": "p.gguf"},
         [{"role": "user", "content": [_IMAGE]}],
         UnsupportedInputError("x"), "reads audio only"),
        ({"supports_audio": True},
         [{"role": "user", "content": [_IMAGE]}],
         UnsupportedInputError("x"), "cannot accept image"),
        ({}, [{"role": "user", "content": [_IMAGE]}],
         UnsupportedInputError("x"), "cannot accept image"),
        ({"supports_audio": True, "supports_images": True, "mmproj": "p.gguf"},
         [{"role": "user", "content": [_IMAGE] + _AUDIO_MSG[0]["content"]}],
         UnsupportedInputError("The attached WAV file has no format or data section."),
         "no format or data section"),
        ({"supports_images": True, "mmproj": "p.gguf"},
         [{"role": "user", "content": [_IMAGE] + _AUDIO_MSG[0]["content"]}],
         UnsupportedInputError("x"), "cannot accept audio"),
    ])
    def test_refusal_text(self, engine_flags, messages, exc, expect):
        from localm.cli.chat import _input_refusal_text
        engine = MagicMock()
        engine.supports_images = engine_flags.get("supports_images", False)
        engine.supports_audio = engine_flags.get("supports_audio", False)
        engine._backend = MagicMock(mmproj_path=engine_flags.get("mmproj"), model_path=None)
        assert expect in _input_refusal_text(engine, messages, exc)

    def test_interactive_attaches_audio_given_without_a_prompt(self, tmp_path, monkeypatch):
        from localm.cli import chat as chat_mod
        clip = tmp_path / "c.wav"
        clip.write_bytes(base64.b64decode(_wav_b64()))
        seen = []
        engine = MagicMock()

        def _chat_stream(messages, **kwargs):
            seen.append([dict(m) for m in messages])
            yield "ok"

        engine.chat_stream.side_effect = _chat_stream
        engine.display_name = "asr"
        engine.count_tokens.return_value = 1
        engine.count_messages_tokens.return_value = 3
        engine.context_capacity.return_value = 4096
        inputs = iter(["first", "second"])

        def _fake_input(*a, **kw):
            try:
                return next(inputs)
            except StopIteration as e:
                raise EOFError() from e

        monkeypatch.setattr(chat_mod.console, "input", _fake_input)
        chat_mod._interactive(engine, None, {}, audios=[str(clip)])
        first, second = seen[0][-1]["content"], seen[1][-1]["content"]
        assert first[0]["type"] == "input_audio" and first[-1]["text"] == "first"
        assert second == "second"

    def test_projector_known_reads_the_header(self, tmp_path):
        from localm.cli.chat import _projector_known
        from localm.model_manager import capabilities as caps
        from tests.test_audio_input_capability import _projector
        assert _projector_known(None) is None
        asr = _projector(tmp_path / "a.gguf", audio=True)
        assert _projector_known(str(asr)) == {caps.AUDIO: True}
        (tmp_path / "junk.gguf").write_bytes(b"x")
        assert _projector_known(str(tmp_path / "junk.gguf")) == {caps.VISION: True}
