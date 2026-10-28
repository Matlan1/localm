# SPDX-License-Identifier: AGPL-3.0-or-later
"""The explicit-unload paths honor the in-flight-request pin.

A request pins its engine (`active_requests > 0`) for its whole lifetime, and
the VRAM-eviction loop and the idle-unload loop both skip a pinned engine.
`unload_all_models` and `unload_one_model` (the owner `/v1/models/unload` route
and the sibling-instance cooperate-unload) cancel nothing without `force=True`:
they wait a short grace period for the requests to finish on their own, and if
the pin is still held they report "confirm_required" with a description and
leave the model and its reply running. With `force=True` they cancel the busy
model's generations (residency.cancel_all) and evict it.
"""

import asyncio

import pytest

from localm.inference import http_server as hs
from localm.inference import residency


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
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
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
    assert res.get("detail") == ("'m' is in use (1 other active request); unloading "
                                 "it anyway stops it and cuts off the reply it is writing")


def _registered_reply(name):
    reply = residency.GenerationCancel()
    residency.register_cancel(name, reply)
    return reply


def _short_wait_recording(monkeypatch, reply):
    """Shorten the grace period and record whether *reply* was already
    cancelled each time the wait starts."""
    real_wait = hs._wait_for_pin_clear
    seen = []

    async def _short_wait(engine, **kwargs):
        seen.append(reply.is_set())
        return await real_wait(engine, timeout=0.05, poll_interval=0.01)

    monkeypatch.setattr(hs, "_wait_for_pin_clear", _short_wait)
    return seen


def test_unload_one_never_cancels_a_busy_model_before_asking(isolated, monkeypatch):
    e = _install("m", active=1)
    reply = _registered_reply("m")
    seen = _short_wait_recording(monkeypatch, reply)
    try:
        asked = asyncio.run(hs.unload_one_model("m"))

        assert asked["status"] == "confirm_required", asked
        assert seen == [False]
        assert not reply.is_set(), "the busy model's reply was cut before asking"
        assert e.loaded and e.unload_calls == 0 and e.active_requests == 1
        assert hs._active_model_name == "m"

        forced = asyncio.run(hs.unload_one_model("m", force=True))

        assert forced["status"] == "unloaded", forced
        assert reply.is_set() and reply.by_server, "force stops the busy model's reply"
        assert not e.loaded and e.unload_calls == 1
    finally:
        residency.unregister_cancel("m", reply)


def test_unload_all_never_cancels_a_busy_model_before_asking(isolated, monkeypatch):
    busy = _install("busy", active=1)
    idle = _install("idle", active=0)
    reply = _registered_reply("busy")
    seen = _short_wait_recording(monkeypatch, reply)
    try:
        asked = asyncio.run(hs.unload_all_models())

        assert asked["skipped_in_use"] == ["busy"], asked
        assert asked["confirm_required"] == {"busy": "1 other active request"}
        assert seen == [False]
        assert not reply.is_set(), "the busy model's reply was cut before asking"
        assert busy.loaded and busy.unload_calls == 0 and busy.active_requests == 1
        assert not idle.loaded and idle.unload_calls == 1

        forced = asyncio.run(hs.unload_all_models(force=True))

        assert forced["unloaded_models"] == ["busy"], forced
        assert reply.is_set() and reply.by_server, "force stops the busy model's reply"
        assert not busy.loaded and busy.unload_calls == 1
    finally:
        residency.unregister_cancel("busy", reply)


def test_unload_all_waits_for_every_busy_model_at_once(isolated, monkeypatch):
    """One grace period for all busy models, not one after another."""
    import time
    real_wait = hs._wait_for_pin_clear
    monkeypatch.setattr(hs, "_wait_for_pin_clear",
                        lambda engine, **kw: real_wait(engine, timeout=0.5,
                                                       poll_interval=0.01))
    engines = [_install(f"busy{i}", active=1) for i in range(4)]
    started = time.monotonic()
    res = asyncio.run(hs.unload_all_models())
    elapsed = time.monotonic() - started
    assert sorted(res["skipped_in_use"]) == ["busy0", "busy1", "busy2", "busy3"]
    assert all(e.loaded for e in engines)
    assert elapsed < 1.5, f"the busy models were waited for one after another ({elapsed:.2f}s)"


def test_unload_all_skips_a_model_pinned_after_the_wait(isolated, monkeypatch):
    """A request that pins an idle model while the busy ones are waited for
    keeps that model loaded and its reply running."""
    busy = _install("busy", active=1)
    late = _install("late", active=0)
    reply = _registered_reply("late")

    async def _late_request_arrives(engine, **kw):
        late.active_requests = 1
        engine.active_requests = 0
        return True

    monkeypatch.setattr(hs, "_wait_for_pin_clear", _late_request_arrives)
    try:
        res = asyncio.run(hs.unload_all_models())
        assert res["unloaded_models"] == ["busy"], res
        assert res["skipped_in_use"] == ["late"]
        assert not reply.is_set()
        assert late.loaded and late.unload_calls == 0 and not busy.loaded
    finally:
        residency.unregister_cancel("late", reply)


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
    """A reply that finishes on its own within the grace period: unloaded
    with no confirm box and nothing cancelled."""
    e = _install("m", active=1)
    cancelled = []
    monkeypatch.setattr(residency, "cancel_all", lambda name: cancelled.append(name) or 0)

    async def _clears_soon(engine, **kw):
        engine.active_requests = 0
        return True

    monkeypatch.setattr(hs, "_wait_for_pin_clear", _clears_soon)
    res = asyncio.run(hs.unload_one_model("m"))
    assert not e.loaded and e.unload_calls == 1
    assert res.get("status") == "unloaded", res
    assert cancelled == []


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
        def reap_idle_in_background(self):
            return False

        def list(self, is_owner=True, reap=True):
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
        def reap_idle_in_background(self):
            return False

        def list(self, is_owner=True, reap=True):
            raise RuntimeError("boom")

    monkeypatch.setattr(hs, "_coder_session_manager", _Broken())
    assert hs._coder_sessions_using("m") == []


def test_unload_one_model_force_broadcasts_cancel_for_the_right_model(isolated, monkeypatch):
    """Only force calls residency.cancel_all, once, for the unloaded model."""
    calls = []
    monkeypatch.setattr(residency, "cancel_all", lambda name: calls.append(name) or 0)
    _install("m", active=1)
    _install("other", active=1)
    asyncio.run(hs.unload_one_model("m", force=True))
    assert calls == ["m"], f"expected cancel_all('m') exactly once, got {calls}"
