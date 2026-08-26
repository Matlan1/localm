# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM eviction must not crash when the victim is removed during its unload.

switch_engine picks an idle victim, then `await`s its unload (an executor call -
an event-loop yield). During that yield a CONCURRENT remover (another API load
that picked the SAME idle victim - get_engine loads with preempt=False so they
coexist - or an idle/explicit unload) may have already dropped the victim from
`_engines`/`_engines_lru`. A bare `del _engines[evict_name]` /
`_engines_lru.remove(evict_name)` then raises KeyError/ValueError and surfaces
as an HTTP 500 with a traceback (plus a leaked `_inference_sems` entry). This
pins the guarded removal.

The concurrent removal is simulated deterministically by the victim's own
unload() dropping itself from the registry (exactly what a racing remover does
during the same await window) and freeing enough VRAM for the incoming load.

The GatedEngine tests further down pin the deeper case: the
pin-arrives-during-the-native-unload-await window itself, on every evict/unload
path. A request pins its engine lock-free (active_requests += 1) AFTER
get_engine's active_requests==0 check has passed (the pin is synchronous), so a
QUEUED request can pin the victim DURING the unload await and get it freed out
from under it, then silently auto-reloaded (VRAM over-subscription / a tight-box
500). The eviction victim is detached from the live registry BEFORE the free,
and the kept-registered unload paths are flagged `unloading`, so
get_engine/switch_engine's fast paths refuse an engine that is being freed -
WITHOUT taking the victim's own semaphore, which would reintroduce a two-switch
lock-ordering deadlock.
"""

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from localm.inference import http_server as hs
from tests.conftest import probe_double


class _IncomingEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, ev):
        pass

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False


class _RacyVictim:
    """An idle victim whose unload() ALSO removes it from the registry (like a
    concurrent remover during the same await window) and frees VRAM."""

    def __init__(self, name, vram_state):
        self.display_name = name
        self._loaded = True
        self.active_requests = 0
        self._vram = vram_state

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self._loaded = False
        # Simulate a concurrent removal landing during this (executor) unload...
        hs._engines.pop(self.display_name, None)
        if self.display_name in hs._engines_lru:
            hs._engines_lru.remove(self.display_name)
        # ...and the freed VRAM so the incoming load now fits and the loop breaks.
        self._vram["free"] = 8 * 1024 ** 3


@pytest.fixture
def evicting(monkeypatch):
    vram = {"free": 3 * 1024 ** 3}   # below the ~5.8 GB the incoming load needs
    monkeypatch.setattr("localm.discover.vram_capacity",
                        probe_double(lambda: {"free": vram["free"],
                                              "total": 16 * 1024 ** 3}))
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    reg = {"victim": {"path": "models/victim.gguf", "source": "local"},
           "incoming": {"path": "models/incoming.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "hint"))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    # Also reset by unload_one_model/unload_all_models/_idle_unload_once
    # whenever they clear _active_model_name, so a name remembered by an earlier
    # test does not leak into this one.
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return vram


def test_eviction_survives_victim_removed_during_unload(evicting):
    vram = evicting
    victim = _RacyVictim("victim", vram)
    hs._engines["victim"] = victim
    hs._engines_lru.append("victim")
    hs._inference_sems["victim"] = asyncio.Semaphore(1)
    hs._active_model_name = "victim"

    incoming = _IncomingEngine("incoming")
    from unittest.mock import patch
    with patch.object(hs, "_engine_factory", lambda n: incoming):
        # Without the guarded removal this raises KeyError (-> HTTP 500); with it
        # the load completes.
        engine = asyncio.run(hs.get_engine("incoming"))

    assert engine is incoming and incoming.loaded, "incoming model should have loaded"
    assert "victim" not in hs._engines, "victim was evicted"
    assert "victim" not in hs._inference_sems, "victim's semaphore must not leak"


# --------------------------------------------------------------------------- #
#  The pin-arrives-during-the-native-unload-await window.                      #
#  Each test drives an evict/unload to a controllable STOP inside the          #
#  native unload() and asserts the victim is not fast-path-pinnable there.     #
# --------------------------------------------------------------------------- #


class GatedEngine:
    """Fake engine whose ``unload()`` blocks until ``release`` is set and signals
    ``entered`` the instant it starts - so a test can inspect / act on server
    state DURING the native free (the TOCTOU window)."""

    def __init__(self, display_name, *, gate=False):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.unloading = False
        self.supports_images = False
        self.can_be_multimodal = False
        self.model_path = f"models/{display_name}.gguf"
        self._gate = gate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.unload_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self.unload_calls += 1
        if self._gate:
            self.entered.set()
            # Bounded so a broken test can never wedge the worker thread forever.
            self.release.wait(5)
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 10

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **gen_kwargs):
        yield "hi from " + self.display_name


def _install(monkeypatch, engines, *, total_gb=10):
    """Register *engines* (a name -> GatedEngine dict) with a measurable,
    usage-aware VRAM budget: each loaded model reports the code's 4 GB
    unregistered-file default (-> vram_required ~= 4.8 GB, + 1 GB headroom =
    ~5.8 GB), and total free is *total_gb* GB. At the default 10 GB exactly ONE
    fits, so loading a second model forces eviction of the first."""
    reg = {name: {"path": f"models/{name}.gguf", "source": "local"} for name in engines}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)

    def _vram():
        loaded = sum(1 for e in engines.values() if e.loaded)
        return {"free": (total_gb * 1024 ** 3) - int(loaded * 4.8 * 1024 ** 3),
                "total": total_gb * 1024 ** 3}

    monkeypatch.setattr("localm.discover.vram_info", probe_double(_vram))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: engines[name])

    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    # Also reset by unload_one_model/unload_all_models/_idle_unload_once
    # whenever they clear _active_model_name, so a model unloaded by an EARLIER
    # test in this file cannot leak its remembered name into a later test that
    # expects a genuinely unresolvable state.
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None


def _fast_path_pinnable(name):
    """Mirror get_engine's fast-path predicate: would a queued request for
    *name* be handed this engine (and then _pin it) right now?"""
    e = hs._engines.get(name)
    return e is not None and e.loaded and getattr(e, "unloading", False) is not True


async def _wait_entered(engine):
    for _ in range(1000):          # up to ~5s
        if engine.entered.is_set():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the native unload() was never reached")


def test_eviction_victim_not_pinnable_during_native_free(monkeypatch):
    a = GatedEngine("model-a", gate=True)
    b = GatedEngine("model-b")
    engines = {"model-a": a, "model-b": b}
    _install(monkeypatch, engines)

    async def scenario():
        assert (await hs.switch_engine(
            "model-a", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert a.loaded

        # Loading model-b forces eviction of idle model-a; a.unload() blocks,
        # parking the eviction exactly in the TOCTOU window.
        tb = asyncio.create_task(
            hs.switch_engine("model-b", hs._engine_factory, preempt=False))
        await _wait_entered(a)

        # While the victim is being freed it is NOT fast-path-pinnable, so a
        # queued get_engine cannot hand it back and pin a doomed engine.
        assert not _fast_path_pinnable("model-a"), (
            "victim is fast-path-pinnable while it is being freed: a queued "
            "request would pin an engine about to be unloaded out from under it")

        # Drive the REAL queued-request resolution + pin during the window: it
        # either 503s (the victim's VRAM is still held and nothing else is
        # evictable) or reloads fresh, and never holds a pin on a dead engine.
        pinned = None
        try:
            resolved = await hs.get_engine("model-a")
            hs._pin(resolved)
            pinned = resolved
        except HTTPException:
            pinned = None

        a.release.set()
        assert (await tb)["status"] == "loaded"

        if pinned is not None:
            assert pinned.loaded, (
                "a queued request pinned an engine that was then unloaded "
                "(the pin-arrives-during-the-unload-await window)")
            hs._unpin(pinned)

    asyncio.run(scenario())
    assert b.loaded
    assert "model-a" not in hs._engines          # victim genuinely evicted
    assert a.active_requests == 0                 # no pin leaked onto the doomed engine


def test_unload_one_victim_not_pinnable_during_native_free(monkeypatch):
    a = GatedEngine("model-a", gate=True)
    engines = {"model-a": a}
    _install(monkeypatch, engines)

    async def scenario():
        assert (await hs.switch_engine(
            "model-a", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert a.loaded

        tu = asyncio.create_task(hs.unload_one_model("model-a"))
        await _wait_entered(a)

        # The unload helper KEEPS the engine registered (lazy-reload contract),
        # but flags it unloading for the duration of the free, so a queued
        # request still cannot fast-path-pin it.
        assert "model-a" in hs._engines
        assert not _fast_path_pinnable("model-a"), (
            "engine is fast-path-pinnable while unload_one_model is freeing it")

        a.release.set()
        assert (await tu)["status"] == "unloaded"

    asyncio.run(scenario())
    assert not a.loaded
    assert a.unloading is False                   # flag cleared after the free


def test_unload_all_victim_not_pinnable_during_native_free(monkeypatch):
    a = GatedEngine("model-a", gate=True)
    engines = {"model-a": a}
    _install(monkeypatch, engines)

    async def scenario():
        assert (await hs.switch_engine(
            "model-a", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert a.loaded

        tu = asyncio.create_task(hs.unload_all_models())
        await _wait_entered(a)

        assert not _fast_path_pinnable("model-a"), (
            "engine is fast-path-pinnable while unload_all_models is freeing it")

        a.release.set()
        res = await tu
        assert "model-a" in res["unloaded_models"]

    asyncio.run(scenario())
    assert not a.loaded
    assert a.unloading is False


def test_idle_unload_victim_not_pinnable_during_native_free(monkeypatch):
    a = GatedEngine("model-a", gate=True)
    engines = {"model-a": a}
    _install(monkeypatch, engines)

    async def scenario():
        assert (await hs.switch_engine(
            "model-a", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert a.loaded

        # Make model-a look long-idle so _idle_unload_once decides to free it.
        past = time.monotonic() - 10_000
        hs._last_activity_per_model["model-a"] = past
        hs._last_activity = past

        ti = asyncio.create_task(hs._idle_unload_once(1))
        await _wait_entered(a)

        assert not _fast_path_pinnable("model-a"), (
            "engine is fast-path-pinnable while _idle_unload_once is freeing it")

        a.release.set()
        assert (await ti) is True

    asyncio.run(scenario())
    assert not a.loaded
    assert a.unloading is False


def test_localm_request_resolving_to_no_model_is_503_not_500(monkeypatch):
    """Clearing _active_model_name on an active-model eviction (and unload_all /
    idle-unload) can leave a 'localm'/empty request resolving to None when no
    default model is configured (e.g. `gui --no-model` with a populated registry).
    That must be an honest 503, NOT a 500 from switch_engine(None) ->
    get_model_info(None) -> Path(None) TypeError."""
    a = GatedEngine("model-a")
    _install(monkeypatch, {"model-a": a})   # leaves _active_model_name / _default_model_name = None

    async def scenario():
        for probe in ("localm", "", "   "):
            try:
                await hs.get_engine(probe)
            except HTTPException as e:
                assert e.status_code == 503, f"model={probe!r} -> {e.status_code}, want 503"
            else:
                raise AssertionError(f"model={probe!r} did not raise (expected 503)")

    asyncio.run(scenario())


def test_eviction_skips_a_candidate_already_being_unloaded(monkeypatch):
    """A switch_engine eviction must NOT pick a victim that another unload path is
    already mid-freeing (unloading=True). Those paths keep the engine in
    _engines_lru until the native free completes, so without the guard the
    eviction would call _backend.unload() a SECOND time concurrently - a double
    free of one native context that the per-model semaphore does not serialise."""
    a = GatedEngine("model-a")   # flagged unloading (pretend another path owns its free)
    b = GatedEngine("model-b")   # the genuinely-idle, evictable model
    c = GatedEngine("model-c")   # the new model whose load forces one eviction
    engines = {"model-a": a, "model-b": b, "model-c": c}
    # total 12 GB: model-a + model-b (~4.8 GB each) coexist; loading model-c
    # (needs ~5.8 GB) forces exactly ONE eviction.
    _install(monkeypatch, engines, total_gb=12)

    async def scenario():
        assert (await hs.switch_engine("model-a", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert (await hs.switch_engine("model-b", hs._engine_factory, preempt=False))["status"] == "loaded"
        assert a.loaded and b.loaded

        # Simulate another path mid-freeing model-a: it stays registered + in the
        # LRU (as unload_one/idle keep it) but is flagged unloading.
        a.unloading = True

        # Loading model-c forces one eviction. The only correct victim is the
        # genuinely-idle model-b; model-a must be skipped despite being idle.
        assert (await hs.switch_engine("model-c", hs._engine_factory, preempt=False))["status"] == "loaded"

    asyncio.run(scenario())
    assert c.loaded
    assert "model-b" not in hs._engines, "the idle non-unloading model should have been evicted"
    assert b.unload_calls == 1
    assert "model-a" in hs._engines, "the already-unloading model must not be evicted"
    assert a.unload_calls == 0, (
        "eviction called unload() on an engine another path was already freeing "
        "(a concurrent double free of one native backend)")
