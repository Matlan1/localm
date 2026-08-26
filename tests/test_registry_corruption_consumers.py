# SPDX-License-Identifier: AGPL-3.0-or-later
"""A malformed registry.json entry must never crash a registry consumer.

The shared `_entry_path()` helper (exported from `localm.model_manager`) is what
registry.py's own consumers and the MCP `list_models` route through, so a single
JSON-valid-but-wrong-shape entry is SKIPPED or shown corrupt rather than
crashing. Five other registry-iterating consumers reach the same entry. Each
test below drives the real consumer with a good model plus one malformed sibling
and asserts it does not 500 or raise.

Sites covered:
  1. GET /api/models            (plugins/gui/routes/models.py list loop)
  2. GET /v1/models/{id}        (inference/routes/models.py model_detail)
  3. _pull_hf_snapshot dedup    (model_manager/pull.py "same repo?" scan)
  4. scan_comfy_models          (model_manager/scan.py existing_paths build)
  5. GET /api/vram-estimate     (plugins/gui/routes/models.py, same file as #1)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app
from localm.plugins.gui.web import attach_gui

# The shapes a hand-edited / half-written / cross-version registry.json can take.
# Kept identical to tests/test_model_dedup.py::BAD_ENTRIES so the two suites cover
# the exact same adversarial inputs.
BAD_ENTRIES = {
    "string_entry": "oops",                            # not a dict at all
    "null_entry": None,                                # null value
    "no_path": {"source": "local"},                    # dict missing 'path'
    "null_path": {"path": None, "source": "local"},    # path is null
    "int_path": {"path": 123},                         # path is not a string
    "empty_path": {"path": "", "source": "local"},     # path is empty
}
_BAD = list(BAD_ENTRIES.items())

_GOOD = {"path": "Z:/nonexistent/good.gguf", "source": "local", "model_type": "llm"}


# --------------------------------------------------------------------------- #
# Site 1 + 5: the GUI model routes (plugins/gui/routes/models.py)
# --------------------------------------------------------------------------- #

def _gui_app():
    app = FastAPI()

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(
        app,
        self_url="http://127.0.0.1:9/v1",   # never dialled in these tests
        switch_model=switch_model,
        active_model=lambda: "good",
    )
    return app


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_gui_models_list_survives_malformed_entry(bad_key, bad_val):
    """Site 1: GET /api/models must 200 and still list the good model; the
    malformed entry is skipped (never a 500 that blanks the whole Models page)."""
    app = _gui_app()
    reg = {"good": _GOOD, bad_key: bad_val}
    with patch("localm.config.load_registry", return_value=reg):
        with TestClient(app) as client:
            r = client.get("/api/models")
    assert r.status_code == 200, r.text
    names = [m["name"] for m in r.json()["models"]]
    assert "good" in names               # the good model is still listed
    assert bad_key not in names          # the malformed entry is skipped, not crashing


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_gui_models_list_good_survives_bad_sibling_with_type_filter(bad_key, bad_val):
    """The `type` filter path must also tolerate a malformed sibling."""
    app = _gui_app()
    reg = {"good": _GOOD, bad_key: bad_val}
    with patch("localm.config.load_registry", return_value=reg):
        with TestClient(app) as client:
            r = client.get("/api/models?type=llm")
    assert r.status_code == 200, r.text
    assert [m["name"] for m in r.json()["models"]] == ["good"]


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_gui_vram_estimate_survives_malformed_named_entry(bad_key, bad_val):
    """Site 5: GET /api/vram-estimate?model=<corrupt> must 200 with model_bytes 0,
    never a 500 from entry.get(...) / Path(None) on the corrupt entry (the route's
    own try/except only catches OSError, so the crash escaped it)."""
    app = _gui_app()
    reg = {"good": _GOOD, bad_key: bad_val}
    # vram_capacity(return_status=True) returns (info, status), not a bare
    # dict - a plain return_value ignores the kwarg and hands back the dict
    # itself, which the route's `info, status = ...` then unpacks by KEY
    # instead of raising. side_effect mirrors the real two-shape contract
    # (see tests/test_vram_reading_honesty.py's _list_gpus_double).
    from localm.discover import GPU_PROBE_OK
    vram = {"free": 8 * 1024 ** 3, "total": 16 * 1024 ** 3}
    with patch("localm.config.load_registry", return_value=reg), \
         patch("localm.discover.vram_capacity",
               side_effect=lambda *a, return_status=False, **kw:
                   (vram, GPU_PROBE_OK) if return_status else vram):
        with TestClient(app) as client:
            r = client.get(f"/api/vram-estimate?model={bad_key}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == bad_key
    assert body["model_bytes"] == 0      # corrupt entry -> no size, still a valid estimate


# --------------------------------------------------------------------------- #
# Site 2: GET /v1/models/{id} (inference/routes/models.py model_detail)
# --------------------------------------------------------------------------- #

def _inf_client():
    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "startup-model"      # the default; NOT a BAD_ENTRIES key
    type(engine).loaded = property(lambda self: True)
    return TestClient(create_app(engine), raise_server_exceptions=True)


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_model_detail_malformed_entry_never_500(bad_key, bad_val):
    """Site 2: model_detail on a corrupt id must never 500. A non-dict / null value
    is 404 (not a usable model record); a dict with a missing / null / non-string /
    empty path renders as a pathless model (path="", size None), the same contract
    test_model_detail_empty_path_does_not_walk_cwd pins down."""
    client = _inf_client()
    reg = {"good": _GOOD, bad_key: bad_val}
    with patch("localm.config.load_registry", return_value=reg):
        r = client.get(f"/v1/models/{bad_key}")
    assert r.status_code != 500, r.text
    if isinstance(bad_val, dict):
        assert r.status_code == 200, r.text
        assert r.json()["path"] == ""            # a corrupt path is scrubbed, not walked
        assert r.json()["size_bytes"] is None
    else:
        assert r.status_code == 404, r.text      # non-dict / null: not registered


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_model_detail_good_model_with_malformed_sibling(bad_key, bad_val):
    """The `aliases` scan iterates EVERY entry, so a malformed SIBLING must not
    crash a detail lookup for a healthy model (a non-dict sibling raises
    AttributeError unguarded)."""
    client = _inf_client()
    reg = {"good": _GOOD, bad_key: bad_val}
    with patch("localm.config.load_registry", return_value=reg):
        r = client.get("/v1/models/good")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "good"


# --------------------------------------------------------------------------- #
# Site 3: _pull_hf_snapshot "same repo already pulled?" dedup scan (pull.py)
# --------------------------------------------------------------------------- #

def _offline_hfapi(monkeypatch):
    """Force the repo-listing fetch down its offline fallback (caught + logged),
    so no network is dialled and the dedup scan is still reached."""
    import huggingface_hub

    class _NoNetApi:
        def __init__(self, *a, **kw):
            pass

        def model_info(self, *a, **k):
            raise RuntimeError("offline in test")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoNetApi)


def _seed_pull(monkeypatch, tmp_path, reg):
    import localm.model_manager as mm
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(mm, "load_registry", lambda: dict(reg))
    # Non-interactive so a same-repo match auto-skips (returns True) after the scan.
    monkeypatch.setattr("localm.model_manager.pull.sys.stdin",
                        MagicMock(isatty=lambda: False))


@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_pull_hf_dedup_survives_malformed_sibling(tmp_path, monkeypatch,
                                                  bad_key, bad_val):
    """Site 3: the pull-dedup scan iterates every entry; a malformed sibling must
    not crash `localm pull owner/repo`. The good same-repo dir is still detected."""
    from localm.model_manager import pull as pull_mod

    repo_id = "owner/repo"
    hf_dir = tmp_path / "hfmodel"
    hf_dir.mkdir()
    reg = {
        "good": {"path": str(hf_dir), "source": f"hf:{repo_id}"},   # a real same-repo dir
        bad_key: bad_val,
    }
    _seed_pull(monkeypatch, tmp_path, reg)
    _offline_hfapi(monkeypatch)

    # Must not raise; a same-repo match with a non-tty session returns True (skip).
    assert pull_mod._pull_hf_snapshot(repo_id, "newname") is True


def test_pull_hf_dedup_survives_bad_path_on_matching_source(tmp_path, monkeypatch):
    """The Path(info.get("path","")) leg of the scan: a sibling whose source DOES
    match the repo but whose path is null must not TypeError (only reachable once
    the source check passes, so BAD_ENTRIES' non-matching-source shapes miss it)."""
    from localm.model_manager import pull as pull_mod

    repo_id = "owner/repo"
    hf_dir = tmp_path / "hfmodel"
    hf_dir.mkdir()
    reg = {
        "good": {"path": str(hf_dir), "source": f"hf:{repo_id}"},
        "poison": {"path": None, "source": f"hf:{repo_id}"},   # matching source, null path
    }
    _seed_pull(monkeypatch, tmp_path, reg)
    _offline_hfapi(monkeypatch)

    assert pull_mod._pull_hf_snapshot(repo_id, "newname") is True


# --------------------------------------------------------------------------- #
# Site 4: scan_comfy_models existing_paths build (model_manager/scan.py)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_key,bad_val", _BAD)
def test_scan_comfy_models_survives_malformed_entry(tmp_path, monkeypatch,
                                                    bad_key, bad_val):
    """Site 4: the launch/GUI-button scan builds existing_paths from every entry;
    a null entry (`"path" in None`) or a null/int path (`Path(...)`) crashes the
    whole scan unguarded. It must complete (nothing to add here) instead."""
    import localm.model_manager.scan as scan_mod

    workdir = tmp_path / "comfy"
    (workdir / "models").mkdir(parents=True)     # empty models dir: nothing to add
    monkeypatch.setattr(scan_mod, "get_comfy_workdir", lambda: str(workdir))
    monkeypatch.setattr(scan_mod, "comfy_object_info", lambda url: None)  # no comfy probe
    reg = {
        "good": {"path": str(tmp_path / "good.gguf"), "source": "local"},
        bad_key: bad_val,
    }
    monkeypatch.setattr(scan_mod, "load_registry", lambda: dict(reg))

    res = scan_mod.scan_comfy_models()
    assert isinstance(res, scan_mod.ScanResult)
    assert res.added == 0        # empty models dir; the point is it did not crash
