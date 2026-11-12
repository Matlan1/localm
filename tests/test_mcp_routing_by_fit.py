# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP server routes by fit like the HTTP server: a vision model that would
read from disk, run on the CPU only, be unsized or unload the model the client
uses is named, never loaded; one that fits is loaded beside it; a load the
fresh measurement refuses is turned into that suggestion; and a routed load
never evicts a model routing did not load."""

from __future__ import annotations

import pytest

from localm.inference import capability_routing as cr
from localm.inference import http_server as hs
from localm.inference import routing_fit
from localm.plugins.mcpserver.server import EngineCache
from tests.test_mcp_routing import _call, _cache, reg  # noqa: F401

_BIG = 1 << 50


@pytest.fixture(autouse=True)
def _auto(monkeypatch):
    monkeypatch.setattr(cr, "configured_mode", lambda: "auto")


def _measure(monkeypatch, cls):
    monkeypatch.setattr(hs, "routing_vram_reading", lambda: routing_fit.VramReading(
        free=1, total=_BIG, usable=True))
    from localm.inference.backends.llamacpp import _sizing
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (_BIG, _BIG))
    monkeypatch.setattr(routing_fit, "classify",
                        lambda *a, **k: (cls, 5 << 30 if cls == cr.FIT_DISK else 0))


@pytest.mark.parametrize("cls", [cr.FIT_GPU, cr.FIT_EXPERTS_RAM, cr.FIT_PARTIAL])
def test_a_fitting_vision_model_answers(reg, monkeypatch, cls):
    _, img, _ = reg
    _measure(monkeypatch, cls)
    engines = _cache(lazy=True)
    res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)]})
    assert res["content"][0]["text"] == "reply-from-seer"
    assert "seer" in engines._routing_loaded


@pytest.mark.parametrize("cls", [cr.FIT_CPU_ONLY, cr.FIT_UNSIZED, cr.FIT_DISK])
def test_a_vision_model_that_needs_asking_is_named_not_loaded(reg, monkeypatch, cls):
    _, img, _ = reg
    _measure(monkeypatch, cls)
    engines = _cache(lazy=True)
    res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)]})
    text = " ".join(c.get("text", "") for c in res["content"])
    assert res["isError"] is True
    assert "seer would answer if you switch" in text
    assert cr.Fit(cls, disk_read_bytes=5 << 30 if cls == cr.FIT_DISK else 0).ask_reason() in text
    assert not getattr(engines.made.get("seer"), "loaded", False)


def test_a_load_the_fresh_measurement_refuses_is_named_not_loaded(reg, monkeypatch):
    _, img, _ = reg
    _measure(monkeypatch, cr.FIT_GPU)
    engines = _cache(lazy=True)
    late = cr.Fit(cr.FIT_DISK, disk_read_bytes=9 << 30)

    def refuse(self, name, engine):
        raise hs.RouteRefused(name, late)
    monkeypatch.setattr(EngineCache, "_refuse_if_routing_must_ask", refuse)
    res = _call(engines, "chat", {"prompt": "what is this?", "images": [str(img)]})
    text = " ".join(c.get("text", "") for c in res["content"])
    assert res["isError"] is True and late.ask_reason() in text
    assert not getattr(engines.made.get("seer"), "loaded", False)
    assert "seer" not in engines._routing_loaded
    assert "seer" not in engines._engines, "the refused engine is not left in the cache"


def test_a_routed_load_never_evicts_the_clients_model(reg, monkeypatch):
    """With the resident cap at one, a routed load would have to evict plain;
    plain is protected, so it is kept and the cap is reported as missed."""
    _, img, _ = reg
    _measure(monkeypatch, cr.FIT_GPU)
    monkeypatch.setattr("localm.inference.residency.resident_cap", lambda cfg: 1)
    engines = _cache(lazy=True)
    plain = engines.get_loaded("plain")
    assert plain.loaded
    engines.get_loaded_chat("seer", routed=True)
    assert plain.loaded and not plain.released
    assert "seer" in engines._routing_loaded
