# SPDX-License-Identifier: AGPL-3.0-or-later
"""The VRAM the shared embedding, reranker and speech models and the native
media runtimes hold, as an explicit model switch counts it before evicting
anything (http_server._shared_vram_holds)."""

from types import SimpleNamespace

import pytest

from localm import vram
from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference import reranker, speech

MB = 1024 ** 2


def _bare_embedder(path, *, n_gpu_layers=99, fallback=None):
    e = object.__new__(emb.IsolatedEmbedder)
    e.model_path = str(path)
    e.n_gpu_layers = n_gpu_layers
    e.gpu_fallback_reason = fallback
    return e


@pytest.fixture
def model_file(tmp_path):
    f = tmp_path / "embed.gguf"
    f.write_bytes(b"\0" * (10 * MB))
    return f


class TestEmbedderHold:
    @pytest.mark.parametrize("layers", [99, 120, -1])
    def test_a_full_offload_holds_the_file_size_plus_a_fifth(self, model_file, layers):
        assert _bare_embedder(model_file, n_gpu_layers=layers).vram_hold_bytes == 12 * MB

    def test_a_cpu_embedder_holds_nothing(self, model_file):
        assert _bare_embedder(model_file, n_gpu_layers=0).vram_hold_bytes == 0

    def test_an_embedder_that_fell_back_to_the_cpu_holds_nothing(self, model_file):
        assert _bare_embedder(model_file, fallback="crashed").vram_hold_bytes == 0

    def test_a_partial_offload_is_unknown(self, model_file):
        assert _bare_embedder(model_file, n_gpu_layers=12).vram_hold_bytes is None

    def test_a_missing_file_is_unknown(self, tmp_path):
        assert _bare_embedder(tmp_path / "gone.gguf").vram_hold_bytes is None

    def test_the_resident_hold_is_read_without_loading(self, monkeypatch, model_file):
        monkeypatch.setattr(emb, "_EMBEDDER", None)
        assert emb.resident_vram_hold() == (False, None)
        monkeypatch.setattr(emb, "_EMBEDDER", _bare_embedder(model_file))
        assert emb.resident_vram_hold() == (True, 12 * MB)
        monkeypatch.setattr(emb, "_EMBEDDER", SimpleNamespace(vram_hold_bytes=None))
        assert emb.resident_vram_hold() == (True, None)
        monkeypatch.setattr(emb, "_EMBEDDER", SimpleNamespace(dim=5))
        assert emb.resident_vram_hold() == (True, None), "a hold it cannot read is unknown"


class TestRerankerHold:
    def test_an_hf_reranker_on_the_cpu_holds_nothing_and_on_a_gpu_is_unknown(self):
        r = object.__new__(reranker.HFReranker)
        r._backend = SimpleNamespace(loaded_device="cpu")
        assert r.vram_hold_bytes == 0
        r._backend = SimpleNamespace(loaded_device="cuda")
        assert r.vram_hold_bytes is None

    def test_the_resident_hold_covers_a_gguf_reranker(self, monkeypatch, model_file):
        monkeypatch.setattr(reranker, "_RERANKER", None)
        assert reranker.resident_vram_hold() == (False, None)
        monkeypatch.setattr(reranker, "_RERANKER", _bare_embedder(model_file))
        assert reranker.resident_vram_hold() == (True, 12 * MB)

    def test_hf_backend_records_the_device_its_load_ran_on(self, monkeypatch, tmp_path):
        from localm.inference.backends import hf

        class _Runner:
            def spawn_and_load(self, params, timeout=None):
                return {"device": "cpu", "rerank_labels": []}

        for target in ("localm.inference.backends.hf._check_custom_code_allowed",
                       "localm.inference.backends.hf._check_format_supported",
                       "localm.inference.hf_tokenizer_safety.validate_tokenizer_json",
                       "localm.inference.hf_shard_index_safety.validate_shard_index"):
            monkeypatch.setattr(target, lambda *a, **k: None)
        monkeypatch.setattr(hf, "HFRunner", _Runner)
        backend = hf.HFBackend(str(tmp_path), device="cpu", task="rerank")
        assert backend.loaded_device is None
        backend.load()
        assert backend.loaded_device == "cpu"


class TestSpeechHold:
    def _engine(self, *, layers=99, projector=True):
        e = object.__new__(speech.SpeechEngine)
        e.model = speech.SpeechModel("tts", "tts.gguf", "mmproj.gguf")
        e.n_ctx = 4096
        e.n_gpu_layers = layers
        e.projector_on_gpu = projector
        return e

    def test_a_full_offload_holds_its_estimate(self, monkeypatch):
        seen = []

        def _estimate(model, n_ctx):
            seen.append((model.name, n_ctx))
            return 900 * MB

        monkeypatch.setattr(speech, "_estimate_bytes", _estimate)
        assert self._engine().vram_hold_bytes == 900 * MB
        assert seen == [("tts", 4096)]

    def test_a_cpu_load_holds_nothing(self):
        assert self._engine(layers=0).vram_hold_bytes == 0

    @pytest.mark.parametrize("layers, projector", [(20, True), (99, False)])
    def test_a_partial_placement_is_unknown(self, monkeypatch, layers, projector):
        monkeypatch.setattr(speech, "_estimate_bytes", lambda m, n: 900 * MB)
        assert self._engine(layers=layers, projector=projector).vram_hold_bytes is None

    def test_the_resident_hold(self, monkeypatch):
        monkeypatch.setattr(speech, "_ENGINE", None)
        assert speech.resident_vram_hold() == (False, None)
        monkeypatch.setattr(speech, "_ENGINE", self._engine(layers=0))
        assert speech.resident_vram_hold() == (True, 0)


class TestBusySharedModelsFreeNothing:
    def test_an_embedder_serving_a_request_frees_nothing(self, monkeypatch):
        monkeypatch.setattr(emb, "_EMBEDDER",
                            SimpleNamespace(vram_hold_bytes=5 * MB, active_requests=1))
        assert emb.resident_vram_hold() == (True, 0)
        monkeypatch.setattr(emb, "_EMBEDDER",
                            SimpleNamespace(vram_hold_bytes=5 * MB, active_requests=0))
        assert emb.resident_vram_hold() == (True, 5 * MB)

    def test_a_reranker_pinned_by_a_request_frees_nothing(self, monkeypatch):
        monkeypatch.setattr(reranker, "_RERANKER",
                            SimpleNamespace(vram_hold_bytes=5 * MB, active_requests=0))
        monkeypatch.setattr(reranker, "_RESIDENT_REQUESTS", 1)
        assert reranker.resident_vram_hold() == (True, 0)
        monkeypatch.setattr(reranker, "_RESIDENT_REQUESTS", 0)
        assert reranker.resident_vram_hold() == (True, 5 * MB)

    def test_a_speech_model_speaking_frees_nothing(self, monkeypatch):
        monkeypatch.setattr(speech, "_ENGINE",
                            SimpleNamespace(vram_hold_bytes=5 * MB, active_requests=2))
        assert speech.resident_vram_hold() == (True, 0)


class TestMediaRuntimeStates:
    def test_loaded_runtimes_report_busy_or_idle_and_nothing_is_stopped(
            self, monkeypatch):
        monkeypatch.setattr(vram, "_native_runtime_loaded", lambda key: True)
        monkeypatch.setattr(vram, "_native_runtime_busy", lambda key: key == "sdcpp")
        monkeypatch.setattr(vram, "_native_runtime_stop_if_idle",
                            lambda key: pytest.fail("a state read stopped a runtime"))
        assert vram.native_media_runtime_states() == {
            "native image/video worker": "busy", "native music runtime": "idle"}

    def test_an_unreadable_runtime_is_unknown(self, monkeypatch):
        def _loaded(key):
            if key == "koboldcpp":
                raise OSError("gone")
            return False

        monkeypatch.setattr(vram, "_native_runtime_loaded", _loaded)
        assert vram.native_media_runtime_states() == {"native music runtime": "unknown"}

    def test_the_job_locks_read_busy_while_held(self):
        from localm.media.koboldcpp import server
        from localm.media.sdcpp import shared
        for module, lock in ((shared, shared.lock), (server, server._lock)):
            assert module.is_busy() is False
            with lock:
                assert module.is_busy() is True
            assert module.is_busy() is False


class TestSharedVramHolds:
    def _modules(self, monkeypatch, embedder=(False, None), rerank=(False, None),
                 speak=(False, None), media=None):
        monkeypatch.setattr(emb, "resident_vram_hold", lambda: embedder)
        monkeypatch.setattr(reranker, "resident_vram_hold", lambda: rerank)
        monkeypatch.setattr(speech, "resident_vram_hold", lambda: speak)
        monkeypatch.setattr(vram, "native_media_runtime_states", lambda: dict(media or {}))

    def test_nothing_loaded(self, monkeypatch):
        self._modules(monkeypatch)
        assert hs._shared_vram_holds() == hs.SharedVramHolds()

    def test_known_holds_unknown_kinds_and_busy_media(self, monkeypatch):
        self._modules(monkeypatch, embedder=(True, 400 * MB), rerank=(True, None),
                      speak=(True, 0),
                      media={"native image/video worker": "busy",
                             "native music runtime": "idle"})
        assert hs._shared_vram_holds() == hs.SharedVramHolds(
            holds=(400 * MB, 0), unknown=("reranker model", "native music runtime"),
            busy_media=("native image/video worker",))

    def test_an_unreadable_media_state_is_unknown(self, monkeypatch):
        self._modules(monkeypatch, media={"native music runtime": "unknown"})
        assert hs._shared_vram_holds().unknown == ("native music runtime",)
