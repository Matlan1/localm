# SPDX-License-Identifier: AGPL-3.0-or-later
"""switch_engine()'s cooperative-unload eviction fallback: when a load cannot fit
and no LOCAL eviction candidate is free, it asks a live sibling localm instance
(found by ``gpu_registry.list_gpu_peers``) to release its model, and falls back to
the exact 503 it always had when there is no peer or the peer declines.

The peers here are fake lists returned in place of live detection; detection
itself and the request / vouch protocol are covered in
test_gpu_peer_detection.py.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from localm import gpu_registry
from localm.inference import http_server as hs
from tests.conftest import probe_double


@pytest.fixture(autouse=True)
def _no_coordination_state():
    """Guarantee hs._gpu_coord starts and ends each test as None (no cross-test
    leakage of coordination state)."""
    hs._gpu_coord = None
    yield
    hs._gpu_coord = None


# ------------------------------------------------------------------ #
#  switch_engine(): cooperative-unload eviction fallback              #
# ------------------------------------------------------------------ #

class _FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, event):
        pass


def _make_engine(name):
    return _FakeEngine(name)


class _UnfittableEngine(_FakeEngine):
    """Simulates the backend's OWN final sizing decision genuinely refusing
    (GgufBackend._check_vram raising because the model cannot fit even at 0 GPU
    layers - llamacpp/_sizing.py). switch_engine does not hard-refuse on its own
    crude whole-model estimate once local and cooperative eviction are exhausted:
    it falls through to a real load attempt and lets the backend decide,
    converting a genuine backend RuntimeError into a clean 503 (see
    switch_engine's `except RuntimeError` around new_engine.load). These
    cooperative-unload tests are about the COOPERATION SEQUENCING (was the
    registry queried, did a peer get asked, does failure never escalate past
    503), not about whole-model sizing, so the model-b factory here must simulate
    a load that genuinely cannot fit or the fall-through would just succeed
    (200) instead of reaching a 503 to assert on."""

    def load(self):
        raise RuntimeError("VRAM exhausted: cannot fit even at 0 GPU layers")


def _make_unfittable_engine(name):
    return _UnfittableEngine(name)


@pytest.fixture
def multi_model_registry(monkeypatch):
    fake_registry = {
        "model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
        "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    yield fake_registry


def _dynamic_vram(free_gate=None):
    """10 GB total; ~8 GB consumed per loaded model, UNLESS free_gate()
    reports the coordination freed things up (used to simulate a peer's
    unload actually releasing driver-level VRAM)."""
    def _read():
        if free_gate is not None and free_gate():
            return {"free": 10 * 1024 ** 3, "total": 10 * 1024 ** 3}
        loaded = sum(1 for e in hs._engines.values() if e.loaded)
        free = (10 * 1024 ** 3) - int(loaded * 8 * 1024 ** 3)
        return {"free": free, "total": 10 * 1024 ** 3}
    return _read


def _pin(monkeypatch, *names):
    """Protect *names* from every eviction path, busy or idle - the real
    mechanism a deployment uses to guarantee a shared model is never
    unloaded out from under it, and the only thing left in this codebase
    that makes a resident model completely unevictable regardless of
    active_requests (switch_engine's own busy-eviction attempt excludes a
    pinned candidate exactly like the idle path does)."""
    from localm.config import load_config as _real
    base = _real()

    def fake():
        cfg = dict(base)
        cfg["pinned_models"] = list(names)
        return cfg

    monkeypatch.setattr("localm.config.load_config", fake)


class TestSwitchEngineCooperativeUnload:
    def test_falls_back_to_503_without_coordination(self, multi_model_registry, monkeypatch):
        """hs._gpu_coord unset (the default for every existing test and every
        --isolated run) -> the coordination branch is a pure no-op, cooperation
        is never attempted, and the load still ends in a clean 503 when the
        backend's own sizing (simulated here - see _UnfittableEngine) genuinely
        cannot fit it. force=True: without it this now returns
        confirm_required BEFORE ever reaching the backend (the estimate
        alone already says it will not fit) - see
        TestSwitchEngineDeferToBackendConfirm for that gate itself."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        _pin(monkeypatch, "model-a")

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1  # not locally evictable
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine, force=True)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503
        assert "VRAM exhausted" in exc.detail

    def test_cooperation_attempted_but_no_holder_falls_back_to_503(
            self, multi_model_registry, monkeypatch):
        """Coordination IS configured, but no live peer holds a model - the
        attempt is genuinely made (proving the wiring runs), and the load
        still ends in a clean 503 once the backend's own sizing (simulated -
        see _UnfittableEngine) genuinely cannot fit it, never a harder
        failure."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http"}
        calls = {"n": 0}

        def fake_list_peers(exclude_self_id=None):
            calls["n"] += 1
            assert exclude_self_id == "self1"
            return []

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", fake_list_peers)
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        _pin(monkeypatch, "model-a")

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine, force=True)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503
        assert calls["n"] >= 1, "must actually query the registry before giving up"

    def test_cooperation_failure_falls_back_to_503_not_harder(
            self, multi_model_registry, monkeypatch):
        """A peer exists but declines/fails cooperation - the load still ends
        in a clean 503 once the backend's own sizing (simulated -
        see _UnfittableEngine) genuinely cannot fit it, never escalated."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http"}
        peer_entry = {"instance_id": "peer1", "port": 9100, "scheme": "http",
                      "model": "peer-model"}
        monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda exclude_self_id=None: [peer_entry])
        monkeypatch.setattr(gpu_registry, "request_cooperative_unload", lambda peer, **k: False)
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        _pin(monkeypatch, "model-a")

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine, force=True)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503

    def test_successful_cooperation_frees_vram_and_load_succeeds(
            self, multi_model_registry, monkeypatch):
        """A live peer holding a model cooperates - the request IS made (to that
        peer) and, once it reports success, the load proceeds without any LOCAL
        eviction (model-a stays resident)."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http"}
        peer_entry = {"instance_id": "peer1", "port": 9200, "scheme": "http",
                      "model": "peer-model"}
        state = {"cooperated": False}

        def fake_list_peers(exclude_self_id=None):
            return [] if state["cooperated"] else [peer_entry]

        def fake_request(peer, **k):
            assert peer["instance_id"] == "peer1"
            state["cooperated"] = True
            return True

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", fake_list_peers)
        monkeypatch.setattr(gpu_registry, "request_cooperative_unload", fake_request)
        monkeypatch.setattr("localm.discover.vram_info",
                            probe_double(_dynamic_vram(free_gate=lambda: state["cooperated"])))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine)

        result = asyncio.run(scenario())
        assert result["status"] == "loaded"
        assert result["model"] == "model-b"
        assert state["cooperated"] is True
        assert "model-b" in hs._engines
        assert "model-a" in hs._engines, "freed via cooperation, never locally evicted"


# ------------------------------------------------------------------ #
#  switch_engine's busy-eviction attempt: the last local resort,     #
#  after cooperative unload, gated to explicit switches only         #
# ------------------------------------------------------------------ #

class TestSwitchEngineBusyEviction:
    def _shorten_wait(self, monkeypatch):
        """A pin that never clears is exactly the case that pays the full
        grace period before confirm_required - shorten it so a test proving
        that does not spend 2 real seconds on it. Captured BEFORE patching,
        same reason as test_unload_honors_pin.py: referencing
        hs._wait_for_pin_clear from inside the replacement would call the
        replacement itself once installed."""
        real = hs._wait_for_pin_clear
        monkeypatch.setattr(
            hs, "_wait_for_pin_clear",
            lambda engine, **kw: real(engine, timeout=0.05, poll_interval=0.01))

    def test_get_engine_style_load_never_touches_a_busy_peer(
            self, multi_model_registry, monkeypatch):
        """preempt=False (exactly what get_engine's own auto-load-on-demand
        call uses) must never cancel or evict a busy peer, even though one
        exists and nothing else can free room - an unrelated request naming
        a different model must not be able to kill someone else's
        generation. Falls through to the pre-existing cooperative-unload/
        defer-to-backend path unchanged, ending in the same 503 as before
        this feature existed."""
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 0.2)
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        from localm.inference import residency
        cancelled = []
        monkeypatch.setattr(residency, "cancel_all", lambda name: cancelled.append(name) or 0)

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine, preempt=False)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503
        assert cancelled == [], "an API-routed load must never cancel a busy peer"
        assert hs._engines["model-a"].active_requests == 1
        assert hs._engines["model-a"].loaded, "the busy peer must survive untouched"

    def test_explicit_switch_against_an_uncleared_busy_peer_returns_confirm_required(
            self, multi_model_registry, monkeypatch):
        """An explicit switch (preempt=True, the GUI/CLI path) against a
        busy local peer that never clears its pin, with no cooperation
        configured, asks for confirmation rather than silently refusing or
        silently killing the generation."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        self._shorten_wait(monkeypatch)

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine)

        result = asyncio.run(scenario())
        assert result["status"] == "confirm_required"
        assert result["model"] == "model-b"
        assert "model-a" in result["detail"]
        assert "model-b" not in hs._engines, "must not have loaded without confirmation"
        assert hs._engines["model-a"].loaded, "must not have evicted without confirmation"

    def test_explicit_switch_force_evicts_a_busy_peer_regardless_of_the_pin(
            self, multi_model_registry, monkeypatch):
        """The owner's explicit force wins unconditionally - it is not asked
        to clear first, mirroring unload_one_model's own force contract."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))
        monkeypatch.setattr(
            hs, "_wait_for_pin_clear",
            lambda engine, **kw: (_ for _ in ()).throw(
                AssertionError("force=True must not wait for the pin")))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine, force=True)

        result = asyncio.run(scenario())
        assert result["status"] == "loaded"
        assert result["model"] == "model-b"
        assert "model-b" in hs._engines
        assert "model-a" not in hs._engines, "force must evict the busy peer"

    def test_explicit_switch_a_pin_that_clears_during_the_grace_period_needs_no_confirmation(
            self, multi_model_registry, monkeypatch):
        """The common case this mechanism exists for: the caller pressed
        Stop, then switched models. cancel_all's signal clears the pin
        within the grace period, and the switch just succeeds - no confirm
        box at all."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))

        async def _clears_soon(engine, **kw):
            engine.active_requests = 0
            return True

        monkeypatch.setattr(hs, "_wait_for_pin_clear", _clears_soon)

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine)

        result = asyncio.run(scenario())
        assert result["status"] == "loaded"
        assert "model-b" in hs._engines
        assert "model-a" not in hs._engines

    def test_busy_eviction_is_tried_only_after_cooperative_unload_fails(
            self, multi_model_registry, monkeypatch):
        """Ordering: asking a peer instance to free ITS own VRAM is tried
        BEFORE ever touching a busy local peer's generation - a successful
        cooperation must never trigger cancel_all on the local busy model at
        all, not merely never actually evict it."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http"}
        peer_entry = {"instance_id": "peer1", "port": 9300, "scheme": "http",
                      "model": "peer-model"}
        state = {"cooperated": False}

        def fake_list_peers(exclude_self_id=None):
            return [] if state["cooperated"] else [peer_entry]

        def fake_request(peer, **k):
            state["cooperated"] = True
            return True

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", fake_list_peers)
        monkeypatch.setattr(gpu_registry, "request_cooperative_unload", fake_request)
        monkeypatch.setattr("localm.discover.vram_info",
                            probe_double(_dynamic_vram(free_gate=lambda: state["cooperated"])))
        from localm.inference import residency
        cancelled = []
        monkeypatch.setattr(residency, "cancel_all", lambda name: cancelled.append(name) or 0)

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine)

        result = asyncio.run(scenario())
        assert result["status"] == "loaded"
        assert state["cooperated"] is True
        assert cancelled == [], "cooperation succeeded - the busy local peer must never be touched"
        assert "model-a" in hs._engines, "freed via cooperation, never locally evicted"


# ------------------------------------------------------------------ #
#  switch_engine's final defer-to-the-backend's-own-sizing branch:    #
#  same confirm/force contract, nothing left to evict at all         #
# ------------------------------------------------------------------ #

def _tight_vram():
    """2 GB free, 6 GB total, unaffected by what is loaded - small enough
    that even a single registered model's UNKNOWN_FOOTPRINT_BYTES estimate
    (4 GB, since these tests' fake registry paths do not exist on disk)
    never fits, with nothing resident to evict in the first place. Isolates
    the defer-to-backend gate from every eviction path above it."""
    return lambda: {"free": 2 * 1024 ** 3, "total": 6 * 1024 ** 3}


class TestSwitchEngineDeferToBackendConfirm:
    def test_explicit_switch_confirms_before_a_degraded_load_with_nothing_to_evict(
            self, multi_model_registry, monkeypatch):
        """Nothing is resident, nothing is pinned or busy, no coordination
        is configured - every eviction path above legitimately finds
        nothing to do, and the crude whole-model estimate alone already
        says this will not fit. An explicit switch must ask before letting
        the backend fall back to a degraded (partial-CPU-offload) load,
        never attempt it silently."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_tight_vram()))

        result = asyncio.run(hs.switch_engine("model-a", _make_unfittable_engine))

        assert result["status"] == "confirm_required"
        assert result["model"] == "model-a"
        assert "model-a" not in hs._engines, \
            "must not have attempted the load without confirmation"

    def test_get_engine_style_load_defers_silently_with_nothing_to_evict(
            self, multi_model_registry, monkeypatch):
        """preempt=False (get_engine's own auto-load-on-demand) must keep
        proceeding straight to the backend's own sizing, exactly as before
        this gate existed - there is no caller in a position to confirm
        anything, and refusing an ordinary chat request outright over a
        crude estimate would be a worse regression than a load that might
        turn out fine, or might degrade to partial CPU offload."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_tight_vram()))

        result = asyncio.run(hs.switch_engine("model-a", _make_engine, preempt=False))

        assert result["status"] == "loaded"
        assert "model-a" in hs._engines

    def test_explicit_switch_force_proceeds_to_a_genuinely_unfittable_backend_failure(
            self, multi_model_registry, monkeypatch):
        """force=True skips the confirmation and lets the backend's own
        final sizing decide - if it genuinely cannot fit even at 0 GPU
        layers, that is still a clean 503, never a crash."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_tight_vram()))

        with pytest.raises(HTTPException) as exc:
            asyncio.run(hs.switch_engine("model-a", _make_unfittable_engine, force=True))
        assert exc.value.status_code == 503

    def test_explicit_switch_force_succeeds_when_the_backend_can_actually_fit_it(
            self, multi_model_registry, monkeypatch):
        """The estimate is deliberately crude (a whole-model guess); once
        confirmed, a backend that CAN make it fit (partial offload, or
        simply a better real sizing than our guess) must still succeed,
        not be held to our own pessimistic estimate a second time."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_tight_vram()))

        result = asyncio.run(hs.switch_engine("model-a", _make_engine, force=True))

        assert result["status"] == "loaded"
        assert "model-a" in hs._engines
