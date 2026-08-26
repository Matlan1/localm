# SPDX-License-Identifier: AGPL-3.0-or-later
"""Idle-unload TTL: the inference server frees the model from VRAM after
``idle_unload_seconds`` of no request, then reloads it lazily on the next one.
Opt-in: 0 (the default) keeps the model resident. The decision lives in a
sleep-free helper so it is testable without waiting on the background loop's
cadence."""

import asyncio
import time

import pytest

from localm.config import DEFAULT_CONFIG
from localm.inference import http_server as hs
from localm.settings_schema import validate_update


def test_default_is_disabled():
    # Off by default: a fresh install must not start dropping the model.
    assert DEFAULT_CONFIG["idle_unload_seconds"] == 0


def test_config_key_validates():
    assert validate_update({"idle_unload_seconds": 120})["idle_unload_seconds"] == 120
    # The GUI form submits strings; they coerce to int.
    assert validate_update({"idle_unload_seconds": "90"})["idle_unload_seconds"] == 90
    with pytest.raises(ValueError):
        validate_update({"idle_unload_seconds": "soon"})   # not a number
    with pytest.raises(ValueError):
        validate_update({"idle_unload_seconds": -5})        # below min=0


@pytest.fixture(autouse=True)
def _isolate_engine_registry(monkeypatch):
    """Isolate the module-level _engines registry.

    _idle_unload_once scans the _engines REGISTRY, not just _engine, and that
    registry is module state that outlives a test: switch_engine/get_engine write
    it and nothing clears it afterwards, so a LOADED engine left behind by an
    earlier test in the same xdist worker would be unloaded instead, flipping
    every "should not unload" assertion here.

    Autouse and module-wide: the tests that take no engine fixture read the same
    registry and need the same isolation."""
    monkeypatch.setattr(hs, "_engines", {})
    monkeypatch.setattr(hs, "_engines_lru", [])


class _FakeEngine:
    def __init__(self, loaded=True):
        self.loaded = loaded
        self.display_name = "fake-model"
        self.unloads = 0

    def unload(self):
        self.loaded = False
        self.unloads += 1


@pytest.fixture
def fake_engine(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(hs, "_engine", eng)
    # Semaphore binds to the running loop lazily on first use (3.10+), so building
    # it here (outside asyncio.run) and using it inside is safe - mirrors the app.
    monkeypatch.setattr(hs, "_inference_sem", asyncio.Semaphore(1))
    return eng


def _set_idle(monkeypatch, seconds_ago):
    monkeypatch.setattr(hs, "_last_activity", time.monotonic() - seconds_ago)


def test_unloads_when_idle_past_ttl(fake_engine, monkeypatch):
    _set_idle(monkeypatch, 1000)
    did = asyncio.run(hs._idle_unload_once(60))
    assert did is True
    assert fake_engine.loaded is False
    assert fake_engine.unloads == 1


def test_keeps_when_recently_active(fake_engine, monkeypatch):
    _set_idle(monkeypatch, 1)            # a request basically just now
    did = asyncio.run(hs._idle_unload_once(60))
    assert did is False
    assert fake_engine.loaded is True
    assert fake_engine.unloads == 0


def test_ttl_zero_never_unloads(fake_engine, monkeypatch):
    _set_idle(monkeypatch, 1000)
    did = asyncio.run(hs._idle_unload_once(0))   # 0 = disabled
    assert did is False
    assert fake_engine.loaded is True


def test_noop_when_already_unloaded(fake_engine, monkeypatch):
    fake_engine.loaded = False
    _set_idle(monkeypatch, 1000)
    did = asyncio.run(hs._idle_unload_once(60))
    assert did is False
    assert fake_engine.unloads == 0


def test_noop_when_no_engine(monkeypatch):
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(hs, "_inference_sem", asyncio.Semaphore(1))
    _set_idle(monkeypatch, 1000)
    assert asyncio.run(hs._idle_unload_once(60)) is False


def test_touch_activity_resets_timer(monkeypatch):
    monkeypatch.setattr(hs, "_last_activity", time.monotonic() - 1000)
    hs._touch_activity()
    assert (time.monotonic() - hs._last_activity) < 1.0
