# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.inference.http_server - unload/load endpoints and usage in responses."""

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def _make_mock_engine(loaded: bool = True, gpu_placement=None):
    """Mock Engine with controllable loaded state and fake chat_stream.

    gpu_placement mirrors the real Engine.gpu_placement property (None by
    default, matching a never-loaded/HF-backend engine that cannot report
    per-layer GPU placement - see test_load_endpoint_gpu_placement below)."""
    engine = MagicMock()
    _state = {"loaded": loaded}

    def _unload():
        _state["loaded"] = False

    def _load():
        _state["loaded"] = True

    def _chat_stream(messages, **kwargs):
        yield "Hello"
        yield " world"

    engine.unload.side_effect = _unload
    engine.load.side_effect = _load
    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.gpu_placement = gpu_placement
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        self.client = TestClient(create_app(self.engine))

    def test_health_when_loaded(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["loaded"])

    def test_health_when_unloaded(self):
        self.engine.unload()
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)      # still 200, not 503
        self.assertFalse(r.json()["loaded"])


class TestModelsEndpoint(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        self.client = TestClient(create_app(self.engine))

    def test_models_list_includes_loaded(self):
        r = self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["object"], "list")
        self.assertTrue(data["data"][0]["loaded"])

    def test_models_list_shows_unloaded(self):
        self.engine.unload()
        r = self.client.get("/v1/models")
        self.assertFalse(r.json()["data"][0]["loaded"])


GB = 1024 ** 3


class TestUnloadEndpoint(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        # Model unload is a MODELS_WRITE management route; open mode needs the
        # loopback shell token (the GUI carries it).
        _app = create_app(self.engine)
        self.client = TestClient(
            _app, headers={"Authorization": f"Bearer {_app.state.shell_token}"})
        # VRAM unmeasurable by default so the release guard returns at once,
        # rather than polling its full timeout for VRAM a mock engine never frees.
        p = patch("localm.discover.vram_info", return_value={})
        p.start()
        self.addCleanup(p.stop)

    def test_unload_when_loaded(self):
        r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "unloaded")
        self.engine.unload.assert_called_once()

    def test_unload_when_already_unloaded(self):
        self.engine.unload()   # unload first
        r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "already_unloaded")

    def test_unload_reports_vram_freed_once_released(self):
        # The release guard: free VRAM is low before the unload, then rises after
        # the native free completes. The endpoint waits for the rise and reports it.
        reads = iter([{"free": 10 * GB}, {"free": 20 * GB}])
        with patch("localm.discover.vram_info",
                   side_effect=lambda: next(reads, {"free": 20 * GB})):
            r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")
        self.assertTrue(body["vram_freed"])
        self.assertEqual(body["vram_before_bytes"], 10 * GB)
        self.assertEqual(body["vram_after_bytes"], 20 * GB)

    def test_unload_vram_reading_marked_uncertain_on_a_stale_probe(self):
        """/v1/models/unload must not report a specific
        vram_before/after_bytes + vram_freed as fact when the underlying GPU
        probe has timed out and fallen back to a stale last-known-good reading.
        The response must say the reading is uncertain."""
        # A real vram_capacity() only returns a tuple when EXPLICITLY asked
        # (return_status=True); the bare-call polling _free() closure inside
        # wait_for_vram_release expects a plain dict, so the mock mirrors the
        # real contract rather than fixing one shape regardless of the call.
        from localm.discover import GPU_PROBE_TIMEOUT

        def _capacity(*, return_status=False):
            info = {"free": 10 * GB}
            return (info, GPU_PROBE_TIMEOUT) if return_status else info

        with patch("localm.discover.vram_capacity", side_effect=_capacity):
            r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")
        # The byte numbers are still reported (best-effort, still informative)...
        self.assertIn("vram_before_bytes", body)
        # ...but now honestly flagged as unreliable, not presented as fact.
        self.assertTrue(body["vram_reading_uncertain"])
        self.assertIn("stale", body["vram_note"])

    def test_unload_vram_reading_not_flagged_on_a_fresh_probe(self):
        """The common case: a fresh probe must NOT carry the uncertainty
        flag - only a demonstrably stale/timed-out one should."""
        from localm.discover import GPU_PROBE_OK

        def _capacity(*, return_status=False):
            info = {"free": 10 * GB}
            return (info, GPU_PROBE_OK) if return_status else info

        with patch("localm.discover.vram_capacity", side_effect=_capacity):
            r = self.client.post("/v1/models/unload")
        body = r.json()
        self.assertNotIn("vram_reading_uncertain", body)
        self.assertNotIn("vram_note", body)

    def test_unload_release_detected_via_combined_split_capacity(self):
        """The release-guard's before/after free-VRAM delta must be measured
        against discover.vram_capacity() (combined split capacity), not just the
        single main GPU - a model that frees VRAM mostly on a NON-main split
        device must still be detected as released. Measuring only the main GPU's
        free shows no rise at all here and reports vram_freed=False despite VRAM
        genuinely being freed."""
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()
        states = iter([
            # before: GPU0 (main) 10GB free, GPU1 5GB free -> combined 15GB
            [{"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
             {"index": 1, "name": "B", "total": 16 * GB, "free": 5 * GB}],
            # after: the model was mostly on GPU1 - GPU0 unchanged, GPU1 freed
            # 10GB -> combined 25GB. Repeats for subsequent polls.
            [{"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
             {"index": 1, "name": "B", "total": 16 * GB, "free": 15 * GB}],
        ])
        last = []

        def _list_gpus():
            nonlocal last
            last = next(states, last)
            return last

        with patch("localm.discover.list_gpus", side_effect=_list_gpus), \
             patch("localm.config.load_config",
                   return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
            r = self.client.post("/v1/models/unload")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "unloaded")
        self.assertTrue(
            body["vram_freed"],
            "combined split free must show a rise even though the single "
            "main GPU (10GB -> 10GB) alone never would")
        self.assertEqual(body["vram_before_bytes"], 15 * GB)
        self.assertEqual(body["vram_after_bytes"], 25 * GB)


class TestLoadEndpoint(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=False)
        # Model load is a MODELS_WRITE management route; open mode needs the
        # loopback shell token (the GUI carries it).
        _app = create_app(self.engine)
        self.client = TestClient(
            _app, headers={"Authorization": f"Bearer {_app.state.shell_token}"})

    def test_load_when_unloaded(self):
        r = self.client.post("/v1/models/load")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "loaded")
        self.engine.load.assert_called_once()

    def test_load_when_already_loaded(self):
        self.engine.load()     # load first
        r = self.client.post("/v1/models/load")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "already_loaded")

    def test_load_omits_gpu_placement_when_backend_cannot_report_it(self):
        # Default mock (gpu_placement=None, e.g. an HF backend or a model that
        # has never been loaded): no gpu_layers_* keys are fabricated without
        # evidence.
        r = self.client.post("/v1/models/load")
        body = r.json()
        self.assertNotIn("gpu_layers_offloaded", body)
        self.assertNotIn("gpu_layers_total", body)
        self.assertNotIn("degraded", body)

    def test_load_reports_full_gpu_offload_as_not_degraded(self):
        self.engine.gpu_placement = {
            "gpu_layers_offloaded": 32, "gpu_layers_total": 32,
            "degraded": False,
        }
        r = self.client.post("/v1/models/load")
        body = r.json()
        self.assertEqual(body["status"], "loaded")
        self.assertEqual(body["gpu_layers_offloaded"], 32)
        self.assertEqual(body["gpu_layers_total"], 32)
        self.assertFalse(body["degraded"])

    def test_load_reports_partial_gpu_offload_as_degraded(self):
        # A model too big to fully fit VRAM still loads (the backend's own sizing
        # defers to a partial/zero GPU offload rather than refusing), so the
        # response must not read as an unqualified success: it says how much of
        # the model actually landed on the GPU.
        self.engine.gpu_placement = {
            "gpu_layers_offloaded": 12, "gpu_layers_total": 32,
            "degraded": True,
        }
        r = self.client.post("/v1/models/load")
        body = r.json()
        self.assertEqual(body["status"], "loaded")
        self.assertEqual(body["gpu_layers_offloaded"], 12)
        self.assertEqual(body["gpu_layers_total"], 32)
        self.assertTrue(body["degraded"],
                         "a partial GPU offload must not report unqualified success")

    def test_load_reports_zero_gpu_offload_as_degraded(self):
        # The extreme end of the same silent fallback: the whole model ran on
        # CPU, and "loaded" must still not read as a normal full GPU load.
        self.engine.gpu_placement = {
            "gpu_layers_offloaded": 0, "gpu_layers_total": 32,
            "degraded": True,
        }
        r = self.client.post("/v1/models/load")
        body = r.json()
        self.assertEqual(body["gpu_layers_offloaded"], 0)
        self.assertTrue(body["degraded"])

    def test_load_already_loaded_also_reports_gpu_placement(self):
        # A caller polling load-status of an already-resident model (a second
        # pre-warm call, or a race) is equally entitled to the truth about
        # placement, not just a fresh "loaded".
        self.engine.load()
        self.engine.gpu_placement = {
            "gpu_layers_offloaded": 12, "gpu_layers_total": 32,
            "degraded": True,
        }
        r = self.client.post("/v1/models/load")
        body = r.json()
        self.assertEqual(body["status"], "already_loaded")
        self.assertTrue(body["degraded"])


class TestChatCompletionsNonStreaming(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        self.client = TestClient(create_app(self.engine))

    def _post(self, messages=None, stream=False, **kwargs):
        payload = {
            "model": "test-model",
            "messages": messages or [{"role": "user", "content": "hi"}],
            "stream": stream,
            **kwargs,
        }
        return self.client.post("/v1/chat/completions", json=payload)

    def test_non_streaming_has_content(self):
        r = self._post()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        self.assertIn("Hello", content)
        self.assertIn("world", content)

    def test_non_streaming_has_usage(self):
        r = self._post()
        data = r.json()
        self.assertIn("usage", data)
        self.assertIsNotNone(data["usage"])
        self.assertGreater(data["usage"]["prompt_tokens"], 0)
        self.assertGreater(data["usage"]["total_tokens"], 0)

    def test_non_streaming_finish_reason_stop(self):
        r = self._post()
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "stop")


class TestChatCompletionsStreaming(unittest.TestCase):
    def setUp(self):
        self.engine = _make_mock_engine(loaded=True)
        self.client = TestClient(create_app(self.engine))

    def _stream_chunks(self, messages=None):
        payload = {
            "model": "test-model",
            "messages": messages or [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        chunks = []
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as r:
            self.assertEqual(r.status_code, 200)
            for line in r.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    chunks.append(json.loads(line[6:]))
        return chunks

    def test_streaming_yields_content_tokens(self):
        chunks = self._stream_chunks()
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        )
        self.assertIn("Hello", content)

    def test_streaming_final_chunk_has_usage(self):
        chunks = self._stream_chunks()
        done_chunks = [c for c in chunks if c["choices"][0].get("finish_reason") == "stop"]
        self.assertEqual(len(done_chunks), 1)
        usage = done_chunks[0].get("usage")
        self.assertIsNotNone(usage)
        self.assertGreater(usage["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
