# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI /api/discover/search route: the `formats` toggle CSV must reach
hf_search, and the response must carry `hf_backend_available` so the search page
can warn (without blocking) that a transformers model needs the .[gpu] extra to
run. hf_search itself (the format merge) is unit-tested in test_discover.py; this
covers the thin route plumbing end to end through the mounted GUI app."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import attach_gui


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    """GUI mounted on a throwaway home with an owner key, so MODELS_READ passes
    and no real HuggingFace call or GPU probe happens (both are monkeypatched per
    test). Mirrors tests/test_key_scope_gui.py's scoped_app."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = admin (all scopes)
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")          # not "off": discovery allowed
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    # No real network / GPU: vram_info is deterministic; hf_search/backend are
    # replaced per test. The route imports these from localm.discover at call
    # time, so patching the module attribute is what the closure resolves.
    import localm.discover as disc
    monkeypatch.setattr(disc, "vram_info", lambda: {})
    return app, disc


def _hdr():
    return {"Authorization": "Bearer ownersecret"}


def test_formats_param_reaches_hf_search_and_response_has_backend_flag(gui_app, monkeypatch):
    app, disc = gui_app
    seen = {}

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        seen["query"] = query
        seen["formats"] = list(formats)
        return [{"id": "org/hf", "downloads": 1, "likes": 0,
                 "updated": "", "formats": ["hf"]}]

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: False)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=gemma&formats=gguf,hf", headers=_hdr())
    assert r.status_code == 200, r.text
    body = r.json()
    assert seen["query"] == "gemma"
    assert seen["formats"] == ["gguf", "hf"]        # CSV parsed into the wanted list
    assert body["hf_backend_available"] is False    # surfaced for the non-blocking hint
    assert body["results"][0]["formats"] == ["hf"]


def test_default_formats_is_gguf(gui_app, monkeypatch):
    app, disc = gui_app
    seen = {}

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        seen["formats"] = list(formats)
        return []

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())   # no formats param
    assert r.status_code == 200, r.text
    assert seen["formats"] == ["gguf"]              # back-compat default
    assert r.json()["hf_backend_available"] is True


def test_bad_format_token_is_422(gui_app, monkeypatch):
    app, disc = gui_app
    # Real hf_search: an all-invalid formats CSV raises DiscoverError -> 422.
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=bogus", headers=_hdr())
    assert r.status_code == 422, r.text


def test_types_param_reaches_hf_search(gui_app, monkeypatch):
    app, disc = gui_app
    seen = {}

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        seen["model_types"] = model_types
        return []

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&types=vae,lora", headers=_hdr())
    assert r.status_code == 200, r.text
    assert seen["model_types"] == ["vae", "lora"]


def test_empty_types_maps_to_none(gui_app, monkeypatch):
    """An empty/absent `types` maps to model_types=None - the legacy untyped
    request shape, protecting any caller that predates the `types` param."""
    app, disc = gui_app
    seen = {}

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        seen["model_types"] = model_types
        return []

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r1 = c.get("/api/discover/search?q=x&types=", headers=_hdr())
        assert seen["model_types"] is None
        r2 = c.get("/api/discover/search?q=x", headers=_hdr())     # no types= at all
        assert seen["model_types"] is None
    assert r1.status_code == 200 and r2.status_code == 200


def test_invalid_types_param_is_422(gui_app, monkeypatch):
    app, disc = gui_app
    # Real hf_search: an all-invalid types CSV raises DiscoverError -> 422.
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&types=bogus", headers=_hdr())
    assert r.status_code == 422, r.text


def test_detected_type_passed_through_response(gui_app, monkeypatch):
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [{"id": "stabilityai/sd-vae-ft-mse", "downloads": 1, "likes": 0,
                 "updated": "", "formats": ["hf"], "detected_type": "unknown"}]

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=vae&types=vae", headers=_hdr())
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["detected_type"] == "unknown"


def test_hf_result_gets_fit_from_size(gui_app, monkeypatch):
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [
            {"id": "org/small", "downloads": 1, "likes": 0, "updated": "",
             "formats": ["hf"], "size_bytes": 2_000_000_000},   # ~2 GB weights
            {"id": "org/unknown", "downloads": 1, "likes": 0, "updated": "",
             "formats": ["hf"], "size_bytes": None},            # no estimate
        ]

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    monkeypatch.setattr(disc, "vram_info", lambda: {"total": 16_000_000_000})
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r.status_code == 200, r.text
    by_id = {x["id"]: x for x in r.json()["results"]}
    assert by_id["org/small"]["fit"] == "fits"    # 2 GB on 16 GB VRAM -> fits
    assert "fit" not in by_id["org/unknown"]       # no size -> no fit (GUI: unknown)


def _configure_split(monkeypatch, gpu_split_indices):
    """Overlay gpu_split_indices onto the REAL (test-isolated) config rather
    than replacing load_config() outright - other GUI routes read other config
    keys too, and a stripped-down fake dict would break those unrelated paths."""
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()

    def _cfg():
        return {**base_cfg, "gpu_split_indices": gpu_split_indices}

    monkeypatch.setattr("localm.config.load_config", _cfg)


def test_hf_result_fit_reflects_combined_split_capacity(gui_app, monkeypatch):
    """AUDIT-GPU-SPLIT-1: with a configured 2-GPU split, the fit badge must
    weigh a result against the COMBINED capacity, not just vram_info()'s
    single main-GPU number - a model too big for one GPU alone but that fits
    split across both must badge "fits", not "too-big"."""
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [{"id": "org/split-fit", "downloads": 1, "likes": 0, "updated": "",
                 "formats": ["hf"], "size_bytes": 15_000_000_000}]   # ~15 GB weights

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    # need ~= 15e9*1.1 + 1.5e9 = 18e9: exceeds the 16 GB main GPU alone
    # (too-big), but fits under 0.85 * the 24 GB combined split (fits).
    monkeypatch.setattr(disc, "list_gpus", lambda: [
        {"index": 0, "name": "A", "total": 16_000_000_000, "free": 16_000_000_000},
        {"index": 1, "name": "B", "total": 8_000_000_000, "free": 8_000_000_000},
    ])
    _configure_split(monkeypatch, [0, 1])
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["vram"]["total"] == 24_000_000_000   # combined, not just the 16 GB main GPU
    assert body["results"][0]["fit"] == "fits"

    # Guard: the SAME result against only the single main GPU (no split
    # configured) must NOT fit - proves the badge genuinely depends on the split,
    # not a coincidence of the fit_label threshold math. vram_capacity() falls
    # back to vram_info() here, so pin that directly to the single-GPU number
    # (same convention test_hf_result_gets_fit_from_size already uses above).
    monkeypatch.setattr(disc, "vram_info",
                        lambda: {"total": 16_000_000_000, "free": 16_000_000_000})
    _configure_split(monkeypatch, None)
    with TestClient(app) as c:
        r2 = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r2.json()["vram"]["total"] == 16_000_000_000
    assert r2.json()["results"][0]["fit"] == "too-big"
