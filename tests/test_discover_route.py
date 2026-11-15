# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI /api/discover/search route: the `formats` toggle CSV must reach
hf_search, and the response must carry `hf_backend_available` so the search page
can warn (without blocking) that a transformers model needs the .[gpu] extra to
run. This covers the thin route plumbing end to end through the mounted GUI
app."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.discover import GPU_PROBE_OK
from localm.plugins.gui.web import attach_gui


def _vram_dict_stub(payload):
    """A vram_info()/list_gpus()-shaped test double that honors return_status
    the same way the real functions do: a bare value by default, (value,
    GPU_PROBE_OK) when the caller opts in via return_status=True (and any
    other kwarg, e.g. deadline/wait_for_inflight, is accepted and ignored).
    _vram_total() (used by every /api/discover/* route) always calls
    vram_capacity(return_status=True) so it can gate `free` on
    sysstats._vram_reading_trusted() - a bare no-arg/no-status stand-in would
    either TypeError or return the wrong shape for that unpack."""
    def _inner(*a, **kw):
        return (payload, GPU_PROBE_OK) if kw.get("return_status") else payload
    return _inner


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    """GUI mounted on a throwaway home with an owner key, so MODELS_READ passes
    and no real HuggingFace call or GPU probe happens (both are monkeypatched per
    test)."""
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
    # No real network or GPU: vram_info is deterministic, hf_search/backend are
    # replaced per test. The route imports these from localm.discover at call
    # time, so patching the module attribute is what the closure resolves.
    import localm.discover as disc
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub({}))
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


def test_net_mode_off_refuses_with_the_settings_place_and_the_command(
        gui_app, monkeypatch):
    """The GUI shows the one net-off download refusal every surface shows."""
    from localm import config
    app, disc = gui_app
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())
    assert r.status_code == 403, r.text
    assert r.json()["detail"] == config.net_off_download_refusal("nothing was searched")


def test_net_mode_off_but_downloads_allowed_reaches_the_real_search(
        gui_app, monkeypatch):
    """net_allow_model_downloads must reach all the way through the route,
    not just discover.py's own front-door check - proves the deeper
    netpolicy.check_url layer (hf_search's actual HTTP fetch) also honours it."""
    app, disc = gui_app
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    import localm.config as _cfg
    monkeypatch.setattr(
        _cfg, "load_config",
        lambda: {"net_mode": "off", "net_allow_model_downloads": True})
    monkeypatch.setattr(disc, "hf_backend_available", lambda: False)

    def _fake_fetch(url, **kw):
        assert kw.get("allow_when_off") is True, \
            "the route's real hf_search must forward the override to safe_fetch_bytes"
        import json
        return url, "application/json", json.dumps([]).encode("utf-8")

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _fake_fetch)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())
    assert r.status_code == 200, r.text


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


@pytest.mark.parametrize("source", ["hf", "civitai"])
@pytest.mark.parametrize("raised,status", [
    (ConnectionError("the remote host closed the connection"), 502),
    (TimeoutError("timed out"), 502),
    (ValueError("not JSON"), 502),
    ("deny", 403),
    ("unresolvable", 502),
])
def test_search_status_follows_the_failure_not_its_wording(gui_app, monkeypatch,
                                                         source, raised, status):
    from localm.netpolicy import NetworkPolicyError, UnresolvableHost
    app, disc = gui_app
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    if raised == "deny":
        raised = NetworkPolicyError("'example.org' is on the deny list (net_deny). ")
    elif raised == "unresolvable":
        raised = UnresolvableHost("'huggingface.co' could not be resolved")

    def _fail(url, **kw):
        raise raised

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _fail)
    with TestClient(app) as c:
        r = c.get(f"/api/discover/search?q=x&source={source}", headers=_hdr())
    assert r.status_code == status, r.text


@pytest.mark.parametrize("http_status", [401, 403, 404])
def test_a_repo_the_hub_does_not_show_is_422(gui_app, monkeypatch, http_status):
    app, disc = gui_app

    class _Answer:
        status_code = http_status

    class _HttpError(Exception):
        response = _Answer()

    def _fail(url, **kw):
        raise _HttpError(f"{http_status} from the Hub")

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _fail)
    with TestClient(app) as c:
        r = c.get("/api/discover/files?repo=org/missing", headers=_hdr())
    assert r.status_code == 422, r.text
    assert "was not found" in r.json()["detail"]


@pytest.mark.parametrize("http_status", [401, 403, 404])
def test_a_civitai_version_civitai_does_not_show_is_422(gui_app, monkeypatch, http_status):
    app, _disc = gui_app

    class _Answer:
        status_code = http_status

    class _HttpError(Exception):
        response = _Answer()

    def _fail(url, **kw):
        raise _HttpError(f"{http_status} from CivitAI")

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _fail)
    with TestClient(app) as c:
        r = c.get("/api/discover/files?repo=999999999&source=civitai", headers=_hdr())
    assert r.status_code == 422, r.text
    assert f"HTTP {http_status}" in r.json()["detail"]


def test_a_civitai_server_error_stays_502(gui_app, monkeypatch):
    app, _disc = gui_app

    class _Answer:
        status_code = 503

    class _HttpError(Exception):
        response = _Answer()

    def _fail(url, **kw):
        raise _HttpError("503 from CivitAI")

    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", _fail)
    with TestClient(app) as c:
        r = c.get("/api/discover/files?repo=1&source=civitai", headers=_hdr())
    assert r.status_code == 502, r.text


def test_a_discover_error_without_a_cause_is_422_whatever_it_says(gui_app, monkeypatch):
    from localm.discover import DiscoverError
    app, disc = gui_app

    def _refuse(*a, **kw):
        raise DiscoverError("request failed: net_mode words in a validation error")

    monkeypatch.setattr(disc, "hf_search", _refuse)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())
    assert r.status_code == 422, r.text


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


def test_a_result_larger_than_video_memory_and_ram_together_says_so(gui_app, monkeypatch):
    from localm.plugins.gui.routes.models import discovery
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [{"id": "org/huge", "downloads": 1, "likes": 0, "updated": "", "formats": ["hf"],
                 "size_bytes": 91_000_000_000},
                {"id": "org/big", "downloads": 1, "likes": 0, "updated": "", "formats": ["hf"],
                 "size_bytes": 40_000_000_000}]

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub({"total": 16_000_000_000}))
    monkeypatch.setattr(discovery, "_total_ram", lambda: 32_000_000_000)
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    by_id = {x["id"]: x for x in r.json()["results"]}
    assert by_id["org/huge"]["exceeds_memory"] is True and by_id["org/huge"]["fit"] == "too-big"
    assert "exceeds_memory" not in by_id["org/big"] and by_id["org/big"]["fit"] == "too-big"


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
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub({"total": 16_000_000_000}))
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r.status_code == 200, r.text
    by_id = {x["id"]: x for x in r.json()["results"]}
    assert by_id["org/small"]["fit"] == "fits"    # 2 GB on 16 GB VRAM -> fits
    assert "fit" not in by_id["org/unknown"]       # no size -> no fit (GUI: unknown)


def test_vram_free_withheld_when_reading_is_untrusted(gui_app, monkeypatch):
    """_vram_total() must gate `vram.free` on sysstats._vram_reading_trusted()
    (fresh AND device-global), exactly like /api/vram-estimate and /api/stats do,
    so an untrusted free is never forwarded verbatim. `total` is a static
    hardware fact and stands regardless."""
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return []

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    # A PROCESS-scoped reading: total stands, free must be withheld.
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub(
        {"total": 16_000_000_000, "free": 15_000_000_000,
         "free_scope": disc.FREE_SCOPE_PROCESS}))
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())
    assert r.status_code == 200, r.text
    vram = r.json()["vram"]
    assert vram["total"] == 16_000_000_000
    assert "free" not in vram


def test_vram_free_shown_when_reading_is_trusted(gui_app, monkeypatch):
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return []

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub(
        {"total": 16_000_000_000, "free": 15_000_000_000,
         "free_scope": disc.FREE_SCOPE_DEVICE}))
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x", headers=_hdr())
    assert r.status_code == 200, r.text
    vram = r.json()["vram"]
    assert vram["total"] == 16_000_000_000
    assert vram["free"] == 15_000_000_000


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
    """With a configured 2-GPU split, the fit badge must weigh a result against
    the COMBINED capacity, not just vram_info()'s single main-GPU number: a model
    too big for one GPU alone but that fits split across both must badge "fits",
    not "too-big"."""
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [{"id": "org/split-fit", "downloads": 1, "likes": 0, "updated": "",
                 "formats": ["hf"], "size_bytes": 15_000_000_000}]   # ~15 GB weights

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    # need ~= 15e9*1.1 + 1.5e9 = 18e9: exceeds the 16 GB main GPU alone
    # (too-big), but fits under 0.85 * the 24 GB combined split (fits).
    monkeypatch.setattr(disc, "list_gpus", _vram_dict_stub([
        {"index": 0, "name": "A", "total": 16_000_000_000, "free": 16_000_000_000},
        {"index": 1, "name": "B", "total": 8_000_000_000, "free": 8_000_000_000},
    ]))
    _configure_split(monkeypatch, [0, 1])
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["vram"]["total"] == 24_000_000_000   # combined, not just the 16 GB main GPU
    assert body["results"][0]["fit"] == "fits"

    # The same result against only the single main GPU (no split configured)
    # must NOT fit. vram_capacity() falls back to vram_info() here, so pin that
    # directly to the single-GPU number.
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub(
        {"total": 16_000_000_000, "free": 16_000_000_000}))
    _configure_split(monkeypatch, None)
    with TestClient(app) as c:
        r2 = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r2.json()["vram"]["total"] == 16_000_000_000
    assert r2.json()["results"][0]["fit"] == "too-big"


def _psutil_import_raises(monkeypatch, exc):
    """Make `import psutil` raise *exc*, as a half-initialised psutil does."""
    import builtins
    import sys
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "psutil" or name.startswith("psutil."):
            raise exc
        return real_import(name, *a, **kw)

    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_total_ram_reports_zero_and_warns_when_psutil_import_breaks(monkeypatch, caplog):
    from localm.plugins.gui.routes.models import discovery
    _psutil_import_raises(monkeypatch, NotImplementedError("platform linux is not supported"))
    with caplog.at_level("WARNING", logger="localm"):
        assert discovery._total_ram() == 0
    assert any("platform linux is not supported" in r.getMessage()
               and r.levelname == "WARNING" for r in caplog.records)


def test_search_still_answers_when_psutil_import_breaks(gui_app, monkeypatch):
    app, disc = gui_app

    def spy(query, limit=20, formats=("gguf",), model_types=None):
        return [{"id": "org/big", "downloads": 1, "likes": 0, "updated": "", "formats": ["hf"],
                 "size_bytes": 40_000_000_000}]

    monkeypatch.setattr(disc, "hf_search", spy)
    monkeypatch.setattr(disc, "hf_backend_available", lambda: True)
    monkeypatch.setattr(disc, "vram_info", _vram_dict_stub({"total": 16_000_000_000}))
    _psutil_import_raises(monkeypatch, NotImplementedError("platform linux is not supported"))
    with TestClient(app) as c:
        r = c.get("/api/discover/search?q=x&formats=hf", headers=_hdr())
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["fit"] == "too-big"
    assert "exceeds_memory" not in result
