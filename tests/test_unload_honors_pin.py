# SPDX-License-Identifier: AGPL-3.0-or-later
"""The explicit-unload paths honor the in-flight-request pin.

A request pins its engine (`active_requests > 0`) for its whole lifetime, and
the VRAM-eviction loop and the idle-unload loop both skip a pinned engine.
`unload_all_models` and `unload_one_model` (the owner `/v1/models/unload` route
and the sibling-instance cooperate-unload) broadcast a cancel first (see
residency.cancel_all) and give it a short grace period; if the pin is STILL
held after that, the owner's action is final via `force=True` (evicted
regardless of what is running), and without force it reports
"confirm_required" with a description, never a bare refusal.
"""

import asyncio

import pytest

from localm.inference import http_server as hs


class FakeEngine:
    def __init__(self, name, active=0):
        self.display_name = name
        self._loaded = True
        self.active_requests = active
        self.unload_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self.unload_calls += 1
        self._loaded = False


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._coder_session_manager = None
    yield


def _install(name, active=0):
    e = FakeEngine(name, active)
    hs._engines[name] = e
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    if hs._active_model_name is None:
        hs._active_model_name = name
        hs._engine = e
    return e


def test_unload_one_model_skips_pinned(isolated, monkeypatch):
    # A pin that never clears is exactly the case that pays the full grace
    # period this path now waits out before giving up - shorten it so this
    # test does not spend 2 real seconds proving a negative. Captured BEFORE
    # patching: referencing hs._wait_for_pin_clear from inside the
    # replacement would call the replacement itself once installed.
    _real = hs._wait_for_pin_clear
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: _real(engine, timeout=0.05, poll_interval=0.01))
    e = _install("m", active=1)          # an in-flight request pinned it
    res = asyncio.run(hs.unload_one_model("m"))
    assert e.loaded, "a pinned engine must NOT be unloaded without force"
    assert e.unload_calls == 0
    assert res.get("status") == "confirm_required", res
    assert res.get("model") == "m"
    assert "detail" in res


def test_unload_one_model_force_evicts_a_pinned_engine_regardless(isolated, monkeypatch):
    """The owner's explicit force wins unconditionally, no matter what the
    engine is doing - it is not asked to clear first."""
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: (_ for _ in ()).throw(
                            AssertionError("force=True must not wait for the pin")))
    e = _install("m", active=1)
    res = asyncio.run(hs.unload_one_model("m", force=True))
    assert not e.loaded and e.unload_calls == 1
    assert res.get("status") == "unloaded", res


def test_unload_one_model_a_cleared_pin_needs_no_confirmation(isolated, monkeypatch):
    """The common case this whole mechanism exists for: the pin was only
    still held because nobody had told it to stop yet. cancel_all's signal
    clears it within the grace period, and the caller never sees a confirm
    box at all - the same request that pressed Stop, then Unload, just
    works."""
    e = _install("m", active=1)

    async def _clears_soon(engine, **kw):
        engine.active_requests = 0
        return True

    monkeypatch.setattr(hs, "_wait_for_pin_clear", _clears_soon)
    res = asyncio.run(hs.unload_one_model("m"))
    assert not e.loaded and e.unload_calls == 1
    assert res.get("status") == "unloaded", res


def test_unload_one_model_unloads_idle(isolated):
    e = _install("m", active=0)
    res = asyncio.run(hs.unload_one_model("m"))
    assert not e.loaded and e.unload_calls == 1
    assert res.get("status") == "unloaded", res


def test_unload_all_models_skips_only_pinned(isolated, monkeypatch):
    _real = hs._wait_for_pin_clear
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: _real(engine, timeout=0.05, poll_interval=0.01))
    busy = _install("busy", active=1)
    idle = _install("idle", active=0)
    res = asyncio.run(hs.unload_all_models())
    assert busy.loaded and busy.unload_calls == 0, "pinned engine must survive unload-all"
    assert not idle.loaded and idle.unload_calls == 1, "idle engine should be unloaded"
    assert "idle" in res.get("unloaded_models", []), res
    assert "busy" not in res.get("unloaded_models", []), res
    assert res.get("confirm_required", {}).get("busy"), \
        "a still-pinned engine must be described in confirm_required, not just skipped"


def test_unload_all_models_force_evicts_the_pinned_one_too(isolated, monkeypatch):
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: (_ for _ in ()).throw(
                            AssertionError("force=True must not wait for the pin")))
    busy = _install("busy", active=1)
    idle = _install("idle", active=0)
    res = asyncio.run(hs.unload_all_models(force=True))
    assert not busy.loaded and busy.unload_calls == 1, "force must evict the pinned engine too"
    assert not idle.loaded and idle.unload_calls == 1
    assert set(res.get("unloaded_models", [])) == {"busy", "idle"}
    assert not res.get("confirm_required"), "nothing left to confirm once forced"


def test_in_use_description_reports_a_bare_pin_as_a_generic_count(isolated):
    e = _install("m", active=2)
    assert hs._in_use_description("m", e) == "2 other active requests"


def test_in_use_description_names_a_busy_coder_session(isolated, monkeypatch):
    e = _install("m", active=1)

    class _FakeManager:
        def list(self, is_owner=True):
            return [{"id": "s1", "cwd": "/proj", "busy": True, "model": "m"},
                    {"id": "s2", "cwd": "/other", "busy": False, "model": "m"},
                    {"id": "s3", "cwd": "/x", "busy": True, "model": "different-model"}]

    monkeypatch.setattr(hs, "_coder_session_manager", _FakeManager())
    detail = hs._in_use_description("m", e)
    assert "/proj" in detail, detail
    # The busy session's own request is one of the 1 active_requests pin
    # counted above - it must not ALSO be reported as an extra "1 other
    # active request".
    assert "other active request" not in detail, detail


def test_coder_sessions_using_is_empty_with_no_coder_gui_mounted(isolated):
    """An --isolated/API-only instance never mounts the GUI, so
    _coder_session_manager stays None - the probe must degrade to empty,
    never raise."""
    assert hs._coder_session_manager is None
    assert hs._coder_sessions_using("m") == []


def test_coder_sessions_using_never_raises_on_a_broken_manager(isolated, monkeypatch):
    class _Broken:
        def list(self, is_owner=True):
            raise RuntimeError("boom")

    monkeypatch.setattr(hs, "_coder_session_manager", _Broken())
    assert hs._coder_sessions_using("m") == []


def test_unload_one_model_actually_broadcasts_cancel_for_the_right_model(isolated, monkeypatch):
    """Not just that the grace-period result is honored (the other tests mock
    _wait_for_pin_clear directly and never touch cancel_all at all) - that
    unload_one_model really calls residency.cancel_all(name), which is the
    whole point: without it, a just-stopped generation would never actually
    be told to stop by the unload attempt itself."""
    from localm.inference import residency
    calls = []
    monkeypatch.setattr(residency, "cancel_all", lambda name: calls.append(name) or 0)
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: _immediate_clear(engine))
    _install("m", active=1)
    asyncio.run(hs.unload_one_model("m"))
    assert calls == ["m"], f"expected cancel_all('m') exactly once, got {calls}"


async def _immediate_clear(engine):
    engine.active_requests = 0
    return True
