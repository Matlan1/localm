# SPDX-License-Identifier: AGPL-3.0-or-later
"""The applied GPU-split status display (maintainer follow-up to the auto-split
feature #772): the split distribution actually applied to the loaded model -
auto free-VRAM-proportional, pinned, or the equal fallback - was visible only
in the INFO log / bug-report ring buffer. It is now recorded parent-side at
load time (GgufBackend.applied_gpu_split), exposed on GET /api/models as
``active_gpu_split`` for the ACTIVE model, and rendered in the sidebar's
loaded-model status (tests-js/model-split-line.test.mjs).

The recording mirrors the exact resolution the worker's apply_gpu_split will
perform (auto override when computed, else the config ratios, else equal -
through the same resolve_gpu_split validation), normalized to shares summing
1.0. GGUF-backend only: the HF backend's device_map="auto" placement is
torch-internal and not recorded (the route simply omits the key)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from localm import discover
from localm.inference.backends.gguf import GgufBackend
from localm.plugins.gui.web import attach_gui

GB = 1024 ** 3


@pytest.fixture
def gui_app():
    """Minimal attach_gui harness, mirroring tests/test_gui.py's gui_app
    (module-local there, so replicated rather than imported)."""
    app = FastAPI()

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(
        app,
        self_url="http://127.0.0.1:9/v1",   # never dialled in these tests
        switch_model=switch_model,
        active_model=lambda: "model-a",
    )
    return app, None


def _backend(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False,
                       n_ctx=512)


def _load(backend):
    # discover.list_gpus is stubbed by _cfg() (always called before _load() in
    # this file) - it now backs both resolve_gpu_split's bare-list validation
    # call AND GgufBackend._load_native's own before/after VRAM-display reads
    # (see gguf.py, #960), so no separate _vram_levels patch is needed here.
    with patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load",
               return_value={"n_layers": 1, "kv_bytes_per_token": 0,
                             "supports_images": False}):
        backend._load_native()


class TestBackendRecordsAppliedSplit:
    def _cfg(self, monkeypatch, *, indices, ratios=None):
        from localm.config import load_config as real_load_config
        base = real_load_config()
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {**base, "gpu_split_indices": indices,
                     "gpu_split_ratios": ratios})
        # Must tolerate BOTH callers: resolve_gpu_split's own bare list_gpus()
        # call (validating configured indices) and GgufBackend._load_native's
        # list_gpus(return_status=True, wait_for_inflight=True) VRAM-display
        # reads (#960) - a zero-arg-only double would TypeError on the latter.
        # These entries carry no "total", so _load_native's print loop skips
        # them (display content is out of scope for this file's tests).
        monkeypatch.setattr(
            discover, "list_gpus",
            lambda **kw: ([{"index": 0}, {"index": 1}], "ok")
            if kw.get("return_status") else [{"index": 0}, {"index": 1}])
        monkeypatch.setattr(discover, "_native_backend_has_vulkan",
                            lambda: False)

    def test_auto_split_recorded_with_normalized_shares(
            self, tmp_path, monkeypatch):
        self._cfg(monkeypatch, indices=[0, 1])
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: [0.6, 0.4])
        b = _backend(tmp_path)
        _load(b)
        assert b.applied_gpu_split == {
            "source": "auto",
            "devices": [{"index": 0, "share": pytest.approx(0.6)},
                        {"index": 1, "share": pytest.approx(0.4)}],
        }

    def test_pinned_ratios_recorded_normalized(self, tmp_path, monkeypatch):
        self._cfg(monkeypatch, indices=[0, 1], ratios=[3.0, 1.0])
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: None)
        b = _backend(tmp_path)
        _load(b)
        assert b.applied_gpu_split["source"] == "pinned"
        shares = [d["share"] for d in b.applied_gpu_split["devices"]]
        assert shares == [pytest.approx(0.75), pytest.approx(0.25)]

    def test_equal_fallback_recorded(self, tmp_path, monkeypatch):
        """Auto declined (unmeasurable) with no pinned ratios: the loader
        applies the equal split, and the display must say so rather than
        showing nothing - the user configured a split and deserves to see
        what it actually did."""
        self._cfg(monkeypatch, indices=[0, 1])
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: None)
        b = _backend(tmp_path)
        _load(b)
        assert b.applied_gpu_split["source"] == "equal"
        shares = [d["share"] for d in b.applied_gpu_split["devices"]]
        assert shares == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_no_split_records_none(self, tmp_path, monkeypatch):
        self._cfg(monkeypatch, indices=None)
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: None)
        b = _backend(tmp_path)
        _load(b)
        assert b.applied_gpu_split is None

    def test_degraded_split_records_none(self, tmp_path, monkeypatch):
        """A split that resolve_gpu_split collapses (stale index, single
        survivor) applies NO split at load - recording one would display a
        distribution that does not exist."""
        self._cfg(monkeypatch, indices=[0, 5])
        monkeypatch.setattr(discover, "resolve_auto_split_ratios",
                            lambda *a, **k: None)
        b = _backend(tmp_path)
        _load(b)
        assert b.applied_gpu_split is None


class TestModelsRouteExposesActiveSplit:
    """GET /api/models carries active_gpu_split for the ACTIVE model's engine
    (absent when there is no split, no backend recording, or no active
    model) - the sidebar's 30s model refresh is the natural carrier, no new
    endpoint or poll."""

    def _client(self, gui_app, monkeypatch, split):
        import localm.inference.http_server as hs
        from fastapi.testclient import TestClient
        app, _ = gui_app
        monkeypatch.setattr(
            "localm.config.load_registry",
            lambda: {"model-a": {"path": "Z:/nonexistent/a.gguf",
                                 "source": "local"}})
        # loaded=True: the route reads engine.loaded per registry row.
        eng = SimpleNamespace(loaded=True,
                              _backend=SimpleNamespace(applied_gpu_split=split))
        monkeypatch.setitem(hs._engines, "model-a", eng)
        return TestClient(app)

    _SPLIT = {"source": "auto",
              "devices": [{"index": 0, "share": 1 / 3},
                          {"index": 1, "share": 2 / 3}]}

    def test_active_split_included(self, gui_app, monkeypatch):
        with self._client(gui_app, monkeypatch, self._SPLIT) as client:
            data = client.get("/api/models?type=llm").json()
        assert data["active"] == "model-a"
        assert data["active_gpu_split"] == {
            "source": "auto",
            "devices": [{"index": 0, "share": pytest.approx(1 / 3)},
                        {"index": 1, "share": pytest.approx(2 / 3)}]}

    def test_absent_when_backend_recorded_none(self, gui_app, monkeypatch):
        with self._client(gui_app, monkeypatch, None) as client:
            data = client.get("/api/models?type=llm").json()
        assert "active_gpu_split" not in data

    def test_absent_when_no_engine_for_active(self, gui_app, monkeypatch):
        import localm.inference.http_server as hs
        from fastapi.testclient import TestClient
        app, _ = gui_app
        monkeypatch.setattr(
            "localm.config.load_registry",
            lambda: {"model-a": {"path": "Z:/nonexistent/a.gguf",
                                 "source": "local"}})
        monkeypatch.setattr(hs, "_engines", {})
        with TestClient(app) as client:
            data = client.get("/api/models?type=llm").json()
        assert "active_gpu_split" not in data
