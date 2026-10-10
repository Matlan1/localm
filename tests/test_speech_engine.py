# SPDX-License-Identifier: AGPL-3.0-or-later
"""The resident speech model: one at a time, loaded on first use, a failed load
not retried for a minute, a crashed worker dropped, released by the server's
unload-all, VRAM eviction, shutdown and restart paths, and the voices folder."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference import reranker as rr
from localm.inference import speech


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class _FakeEngine:
    loads = []

    def __init__(self, model, *, n_gpu_layers, n_ctx=None, n_threads=None):
        if getattr(model, "name", "") == "broken":
            raise RuntimeError(f"failed to load {model.path}")
        _FakeEngine.loads.append(model.name)
        self.model = model
        self.sample_rate = 24000
        self.projector_on_gpu = False
        self.active_requests = 0
        self.alive_flag = True
        self.closed = False
        self.outcome = None
        self.broken = False

    @property
    def alive(self):
        return self.alive_flag and not self.broken

    def claim(self):
        self.active_requests += 1

    def release(self):
        self.active_requests -= 1

    def speak(self, text, **kw):
        if self.outcome is not None:
            raise self.outcome
        return speech.SpeechOutput(wav=b"RIFF", sample_rate=24000, n_samples=24000,
                                   frames=12, seed=1)

    def close(self, grace=5.0):
        self.closed = True


@pytest.fixture
def engines(home, monkeypatch):
    monkeypatch.setattr(speech, "SpeechEngine", _FakeEngine)
    monkeypatch.setattr(speech, "_choose_gpu_layers", lambda model, n_ctx: (0, None))
    _FakeEngine.loads.clear()
    speech.reset_speech()
    yield
    speech.reset_speech()


def _model(home, name):
    p = home / f"{name}.gguf"
    p.write_bytes(b"GGUF" + name.encode())
    mm = home / f"mmproj-{name}.gguf"
    mm.write_bytes(b"GGUF")
    return speech.SpeechModel(name, str(p), str(mm))


class TestResidency:
    def test_loaded_once_and_reused(self, engines, home):
        m = _model(home, "a")
        progress = []
        speech.synthesize(m, "hi", on_progress=progress.append)
        speech.synthesize(m, "hi", on_progress=progress.append)
        assert _FakeEngine.loads == ["a"]
        assert [e["stage"] for e in progress].count("loading") == 1

    def test_another_model_replaces_the_resident_one(self, engines, home):
        a, b = _model(home, "a"), _model(home, "b")
        speech.synthesize(a, "hi")
        first = speech._ENGINE
        speech.synthesize(b, "hi")
        assert first.closed is True and _FakeEngine.loads == ["a", "b"]

    def test_a_busy_model_is_not_replaced(self, engines, home):
        a, b = _model(home, "a"), _model(home, "b")
        speech.synthesize(a, "hi")
        speech._ENGINE.active_requests = 1
        with pytest.raises(speech.SpeechUnavailableError, match="still speaking"):
            speech.get_engine(b)

    def test_a_failed_load_is_latched_then_cleared_by_reset(self, engines, home):
        m = _model(home, "broken")
        with pytest.raises(speech.SpeechUnavailableError):
            speech.get_engine(m)
        with pytest.raises(speech.SpeechUnavailableError):
            speech.get_engine(m)
        assert speech._LOAD_FAILED
        speech.reset_speech()
        assert not speech._LOAD_FAILED

    def test_the_load_failure_message_carries_no_path(self, engines, home):
        m = _model(home, "broken")
        with pytest.raises(speech.SpeechUnavailableError) as info:
            speech.get_engine(m)
        assert str(home) not in str(info.value)

    def test_a_crashed_worker_is_dropped_so_the_next_request_respawns(self, engines, home):
        m = _model(home, "a")
        speech.synthesize(m, "hi")
        eng = speech._ENGINE
        eng.outcome = RuntimeError("The speech worker crashed")
        eng.alive_flag = False
        with pytest.raises(RuntimeError):
            speech.synthesize(m, "hi")
        assert speech._ENGINE is None
        speech.synthesize(m, "hi")
        assert _FakeEngine.loads == ["a", "a"]

    def test_a_clean_error_keeps_the_worker(self, engines, home):
        from localm.inference.backends.llamacpp.mtmd_gen import SpeechInputError
        m = _model(home, "a")
        speech.synthesize(m, "hi")
        eng = speech._ENGINE
        eng.outcome = SpeechInputError("bad")
        with pytest.raises(SpeechInputError):
            speech.synthesize(m, "hi")
        assert speech._ENGINE is eng

    def test_reset_with_a_request_in_flight_is_refused_when_not_forced(self, engines, home):
        speech.synthesize(_model(home, "a"), "hi")
        speech._ENGINE.active_requests = 1
        assert speech.reset_speech(force=False) is False and speech.is_loaded()
        speech._ENGINE.active_requests = 0
        assert speech.reset_speech(force=False) is True and not speech.is_loaded()


class TestReviewFixes:
    def test_a_load_that_cannot_synthesize_stays_unavailable_not_a_load_error(
            self, engines, home, monkeypatch):
        from localm.inference.backends.llamacpp.mtmd_gen import SpeechUnavailable

        class NoSpeech(_FakeEngine):
            def __init__(self, model, **kw):
                raise SpeechUnavailable("runtime predates the speech interface")
        monkeypatch.setattr(speech, "SpeechEngine", NoSpeech)
        m = _model(home, "a")
        for _ in range(2):
            with pytest.raises(SpeechUnavailable) as info:
                speech.get_engine(m)
            assert not isinstance(info.value, speech.SpeechUnavailableError)

    def test_a_claimed_engine_cannot_be_released_before_it_speaks(self, engines, home):
        m = _model(home, "a")
        eng = speech.get_engine(m, claim=True)
        assert eng.active_requests == 1
        assert speech.reset_speech(force=False) is False and speech.is_loaded()
        eng.release()
        assert speech.reset_speech(force=False) is True

    def test_synthesize_releases_its_claim(self, engines, home):
        speech.synthesize(_model(home, "a"), "hi")
        assert speech._ENGINE.active_requests == 0

    def test_an_engine_that_became_unusable_is_dropped_and_reloaded(self, engines, home):
        from localm.inference.backends.llamacpp.mtmd_gen import SpeechUnavailable
        m = _model(home, "a")
        speech.synthesize(m, "hi")
        eng = speech._ENGINE
        eng.outcome = SpeechUnavailable("projector could not be reopened")
        with pytest.raises(SpeechUnavailable):
            speech.synthesize(m, "hi")
        assert speech._ENGINE is None and eng.closed is True
        speech.synthesize(m, "hi")
        assert _FakeEngine.loads == ["a", "a"]


class _FakeRunner:
    def __init__(self):
        self.payloads = []

    def is_alive(self):
        return True

    def speak(self, payload, *, on_progress=None, should_cancel=None):
        self.payloads.append(payload)
        for n in (1, 2, 25):
            on_progress(n)
        return {"wav": b"RIFF", "sample_rate": 24000, "n_samples": 48000,
                "frames": 25, "seed": 3}


def _real_engine():
    import threading
    eng = object.__new__(speech.SpeechEngine)
    eng.model = speech.SpeechModel("a", "a.gguf", "p.gguf")
    eng._runner = _FakeRunner()
    eng._rpc_lock = threading.Lock()
    eng._count_lock = threading.Lock()
    eng.active_requests = 0
    eng.broken = False
    eng.sample_rate = 24000
    eng.encoder_sample_rate = 24000
    eng.projector_on_gpu = False
    return eng


class TestRealEngineSpeak:
    def test_progress_is_reported_in_seconds_of_audio(self):
        eng = _real_engine()
        events = []
        out = eng.speak("hi", seed=3, on_progress=events.append)
        assert out.frames == 25 and out.seconds == 2.0
        assert events[0] == {"stage": "speaking", "frames": 0, "seconds": 0.0}
        assert events[-1] == {"stage": "speaking", "frames": 25, "seconds": 2.0}

    def test_a_reference_wav_is_sent_as_float32_at_the_encoder_rate(self):
        import struct
        eng = _real_engine()
        pcm = struct.pack("<2h", 16384, -16384)
        wav = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
               + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
               + b"data" + struct.pack("<I", len(pcm)) + pcm)
        eng.speak("hi", reference_wav=wav)
        assert eng._runner.payloads[0]["reference"] == struct.pack("<2f", 0.5, -0.5)

    def test_a_reference_that_is_not_a_wav_is_an_input_error(self):
        from localm.inference.backends.llamacpp.mtmd_gen import SpeechInputError
        eng = _real_engine()
        with pytest.raises(SpeechInputError, match="could not be read"):
            eng.speak("hi", reference_wav=b"ID3 not a wav")
        assert eng._runner.payloads == []

    def test_a_second_request_waits_and_can_be_cancelled_while_waiting(self):
        from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled
        eng = _real_engine()
        eng._rpc_lock.acquire()
        events = []
        with pytest.raises(SpeechCancelled, match="while waiting"):
            eng.speak("hi", on_progress=events.append, should_cancel=lambda: True)
        assert events == [{"stage": "waiting"}] and eng._runner.payloads == []


class TestVoices:
    def test_default_is_always_listed(self, home):
        assert speech.list_voices() == ["default"]

    def test_wav_files_become_voices(self, home):
        (home / "voices").mkdir()
        (home / "voices" / "narrator.wav").write_bytes(b"RIFF")
        (home / "voices" / "notes.txt").write_text("x")
        assert speech.list_voices() == ["default", "narrator"]
        assert speech.voice_reference("narrator") == b"RIFF"
        assert speech.voice_reference("default") is None and speech.voice_reference(None) is None

    @pytest.mark.parametrize("name", ["..", "../x", "a\\b", "", " alloy"])
    def test_names_cannot_leave_the_folder(self, home, name):
        if not name.strip():
            assert speech.voice_reference(name) is None
            return
        with pytest.raises(speech.SpeechModelError) as info:
            speech.voice_reference(name)
        assert info.value.status == 400


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    monkeypatch.setattr(rr, "is_loaded", lambda: False)
    yield


class TestUnloadOne:
    PATH = "Z:/models/voice.gguf"

    def _resident(self, monkeypatch, *, active=0, clears=True):
        monkeypatch.setattr("localm.config.load_registry", lambda: {
            "voice": {"path": self.PATH, "model_type": "tts"},
            "other": {"path": "Z:/models/other.gguf"}})
        monkeypatch.setattr(emb, "loaded_path", lambda: None)
        monkeypatch.setattr(rr, "reranker_info", lambda: None)
        monkeypatch.setattr(speech, "speech_info",
                            lambda: {"name": "voice", "path": self.PATH, "sample_rate": 24000})
        monkeypatch.setattr(speech, "active_requests", lambda: active)
        resets = []
        monkeypatch.setattr(speech, "reset_speech",
                            lambda force=True: (resets.append(force), clears)[1])
        return resets

    def test_an_idle_speech_model_is_released_by_its_name(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch)
        res = asyncio.run(hs.unload_one_model("voice"))
        assert resets == [False] and res["status"] == "unloaded" and res["model"] == "voice"

    def test_a_busy_speech_model_is_reported_in_use_without_a_release(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch, active=1)
        res = asyncio.run(hs.unload_one_model("voice"))
        assert res["status"] == "in_use" and resets == []

    def test_a_request_arriving_after_the_precheck_is_reported_in_use(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch, clears=False)
        res = asyncio.run(hs.unload_one_model("voice"))
        assert resets == [False] and res["status"] == "in_use"

    def test_another_model_leaves_the_speech_model_alone(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch)
        assert asyncio.run(hs.unload_one_model("other"))["status"] == "already_unloaded"
        assert resets == []

    def test_speech_info_reports_the_resident_path(self, engines, home):
        m = _model(home, "a")
        speech.synthesize(m, "hi")
        assert speech.speech_info() == {"name": "a", "path": m.path, "sample_rate": 24000}


class TestServerLifecycle:
    def test_unload_all_releases_an_idle_speech_model(self, isolated, monkeypatch):
        calls = []
        monkeypatch.setattr(speech, "is_loaded", lambda: True)
        monkeypatch.setattr(speech, "reset_speech",
                            lambda force=True: (calls.append(force), True)[1])
        res = asyncio.run(hs.unload_all_models())
        assert calls == [False] and res["status"] == "unloaded"

    def test_unload_all_skips_and_reports_a_busy_speech_model(self, isolated, monkeypatch):
        monkeypatch.setattr(speech, "is_loaded", lambda: True)
        monkeypatch.setattr(speech, "reset_speech", lambda force=True: False)
        res = asyncio.run(hs.unload_all_models())
        assert "speech model" in res["skipped_in_use"] and res["status"] == "in_use"

    def test_the_vram_eviction_for_a_chat_load_frees_an_idle_speech_model(self, monkeypatch):
        resets = []
        monkeypatch.setattr(rr, "is_loaded", lambda: False)
        monkeypatch.setattr(speech, "is_loaded", lambda: True)
        monkeypatch.setattr(speech, "reset_speech",
                            lambda force=True: (resets.append(force), True)[1])
        embedder_mod = SimpleNamespace(loaded_dim=lambda: None,
                                       reset_embedder=lambda force=True: False)
        attempt = SimpleNamespace(embedder_attempted=False)
        loop = asyncio.new_event_loop()
        try:
            freed = loop.run_until_complete(hs._switch_evict_embedder(
                loop, SimpleNamespace(measurable=False, free=0), attempt, embedder_mod))
        finally:
            loop.close()
        assert freed is True and resets == [False] and attempt.embedder_attempted is True

    def _spy(self, monkeypatch):
        released = []
        monkeypatch.setattr(speech, "release_for_exit", lambda: (released.append(1), True)[1])
        monkeypatch.setattr(speech, "reset_speech", lambda *a, **k: pytest.fail("took the lock"))
        monkeypatch.setattr(rr, "release_for_exit", lambda: False)
        monkeypatch.setattr(emb, "release_for_exit", lambda: False)
        monkeypatch.setattr(hs, "_engine", None)
        return released

    def test_shutdown_releases_the_speech_worker_without_the_lock(self, monkeypatch):
        released = self._spy(monkeypatch)
        monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            hs._do_shutdown()
        assert released == [1]

    def test_restart_releases_the_speech_worker_without_the_lock(self, monkeypatch):
        released = self._spy(monkeypatch)
        monkeypatch.setattr(os, "execv", lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            hs._do_restart()
        assert released == [1]
