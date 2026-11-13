# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model results say what happened: a scan counts a file it could not register as
failed (not as already there), POST /api/models/load keeps a 404 a 404 and puts
back the type "Load anyway" changed when the load does not happen, and the VRAM
estimate charges the saved overhead and names the model of a first load in flight."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from localm import model_manager as mm
from localm.model_manager.scan import scan_comfy_models
from localm.plugins.gui.web import attach_gui
from tests.test_sdcpp_families import FLUX_AE, FLUX_DEV, write_safetensors
from tests.test_unified_model_browser import temp_registry  # noqa: F401


@pytest.fixture(autouse=True)
def _comfy_offline():
    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        yield


def _comfy(tmp_path: Path) -> Path:
    models = tmp_path / "ComfyUI" / "models"
    (models / "diffusion_models").mkdir(parents=True)
    (models / "vae").mkdir()
    write_safetensors(models / "diffusion_models" / "good.safetensors", FLUX_DEV)
    write_safetensors(models / "vae" / "bad.safetensors", FLUX_AE)
    return models


def test_a_file_that_fails_to_register_is_counted_as_failed(tmp_path, temp_registry):  # noqa: F811
    models = _comfy(tmp_path)
    from localm.model_manager import scan as scan_mod
    real = scan_mod._register_with_dedup

    def flaky(name, path, **kw):
        if path.name == "bad.safetensors":
            raise OSError("disk is read-only")
        return real(name, path, **kw)

    with patch.object(scan_mod, "_register_with_dedup", flaky):
        res = scan_comfy_models(workdir=str(models.parent))
    assert (res.added, res.skipped, res.failed) == (1, 0, 1)
    assert res.failures == ("bad.safetensors: disk is read-only",)
    assert [Path(e["path"]).name for e in mm.load_registry().values()] == ["good.safetensors"]


def test_a_registration_that_writes_nothing_is_counted_as_failed(tmp_path, temp_registry):  # noqa: F811
    models = _comfy(tmp_path)
    from localm.model_manager import scan as scan_mod
    with patch.object(scan_mod, "_register_with_dedup", lambda *a, **k: False):
        res = scan_comfy_models(workdir=str(models.parent))
    assert (res.added, res.skipped, res.failed) == (0, 0, 2)


def _load_client(monkeypatch, switch_model):
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: None)
    return TestClient(app)


def test_a_404_from_the_switch_stays_a_404(monkeypatch, temp_registry, tmp_path):  # noqa: F811
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 4096)
    mm.save_registry({"m": {"path": str(f), "source": "local", "model_type": "llm"}})

    async def switch_model(model, **kw):
        raise HTTPException(404, f"Model files not found: {model}")

    with _load_client(monkeypatch, switch_model) as client:
        r = client.post("/api/models/load", json={"model": "m"})
    assert r.status_code == 404
    assert r.json()["detail"] == "Model files not found: m"


@pytest.mark.parametrize("outcome", ["raises", "http", "cancelled", "confirm"])
def test_load_anyway_puts_the_type_back_when_the_model_does_not_load(
        monkeypatch, temp_registry, outcome):  # noqa: F811
    mm.save_registry({"emb": {"path": "/x/emb.gguf", "source": "local",
                              "model_type": "embedding"}})

    async def switch_model(model, **kw):
        assert mm.load_registry()["emb"]["model_type"] == "llm"
        if outcome == "raises":
            raise RuntimeError("out of memory")
        if outcome == "http":
            raise HTTPException(503, "busy")
        if outcome == "cancelled":
            return {"status": "cancelled", "model": model}
        return {"status": "confirm_required", "model": model, "detail": "frees x"}

    with _load_client(monkeypatch, switch_model) as client:
        client.post("/api/models/load", json={"model": "emb", "as_chat": True})
    entry = mm.load_registry()["emb"]
    assert entry["model_type"] == "embedding"
    assert "type_user_set" not in entry


def test_load_anyway_keeps_the_new_type_when_the_model_loads(monkeypatch, temp_registry):  # noqa: F811
    mm.save_registry({"emb": {"path": "/x/emb.gguf", "source": "local",
                              "model_type": "embedding"}})

    async def switch_model(model, **kw):
        return {"status": "loaded", "model": model}

    with _load_client(monkeypatch, switch_model) as client:
        r = client.post("/api/models/load", json={"model": "emb", "as_chat": True})
    assert r.json()["status"] == "loaded"
    assert mm.load_registry()["emb"]["model_type"] == "llm"


def test_the_estimate_charges_the_overhead_it_is_given():
    from localm.sysstats import estimate_vram
    from localm.vram import VRAM_OVERHEAD_BYTES
    assert estimate_vram(10**9, 4096)["overhead"] == VRAM_OVERHEAD_BYTES
    est = estimate_vram(10**9, 4096, overhead_bytes=500 * 1024 ** 2)
    assert est["overhead"] == 500 * 1024 ** 2
    assert est["needed"] == est["weights"] + est["kv_cache"] + 500 * 1024 ** 2


def _estimate(monkeypatch, cfg: dict, *, loading: str | None = None, model: str = ""):
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_switch_loading", loading)
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda **k: ({"total": 16 * 1024 ** 3, "free": 8 * 1024 ** 3}, "ok"))
    monkeypatch.setattr("localm.sysstats._vram_reading_trusted", lambda info, status: True)
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=None, active_model=lambda: None)
    with TestClient(app) as client:
        return client.get("/api/vram-estimate", params={"model": model} if model else {}).json()


def test_the_estimate_route_charges_the_saved_overhead(monkeypatch, temp_registry, tmp_path):  # noqa: F811
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 4096)
    mm.save_registry({"m": {"path": str(f), "source": "local", "model_type": "llm"}})
    body = _estimate(monkeypatch, {"vram_overhead_mb": 300}, model="m")
    assert body["overhead"] == 300 * 1024 ** 2


def test_the_estimate_names_a_first_load_in_flight(monkeypatch, temp_registry, tmp_path):  # noqa: F811
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 4096)
    mm.save_registry({"m": {"path": str(f), "source": "local", "model_type": "llm"}})
    body = _estimate(monkeypatch, {}, loading="m")
    assert body["model"] == "m"
    assert body["loading"] is True
    assert body["fits"] is None
