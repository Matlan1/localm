# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM eviction safety.

These exercise the multi-model switch/eviction path in http_server.switch_engine:

  A request must PIN its engine (active_requests>=1) the instant it takes
  ownership - synchronously after get_engine, before any await - so a
  concurrent model load can never evict an engine out from under an
  in-flight request. Observed via active_requests from a chat inlet hook,
  which runs after get_engine.

  When vram_info() reports no measurable "free" (the default GGUF-only
  non-NVIDIA install), model switching must fall back to single-resident
  (evict idle before load) instead of stacking models until the driver
  OOMs. Loading a->b->c with unmeasurable VRAM must leave exactly model-c.

  After an eviction the loop must wait for the native VRAM free to land
  before re-checking, so it does not over-evict on a stale-low reading.
"""

import asyncio
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, display_name, *, fails_to_fit=False):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.can_be_multimodal = False
        self.model_path = f"models/{display_name}.gguf"
        # Simulates a REAL backend's OWN final sizing decision (GgufBackend's
        # _effective_gpu_layers/_check_vram, llamacpp/_sizing.py): False
        # (default) means the backend fits the load somehow, by full or partial
        # GPU-layer offload, and succeeds. True simulates the backend's own
        # "cannot fit even at 0 GPU layers" hard refusal, where _check_vram
        # raises RuntimeError because need > total VRAM.
        self.fails_to_fit = fails_to_fit

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self.fails_to_fit:
            raise RuntimeError(
                "Context too large for available VRAM: this load needs more "
                "than this GPU has in total - freeing other VRAM will not "
                "help, it cannot fit regardless.")
        self._loaded = True

    def unload(self):
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
        yield "Hello from " + self.display_name

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _install_fakes(monkeypatch, *, free, status=None, fails_to_fit=False):
    fake_registry = {
        "model-a": {"path": "models/model-a.gguf", "source": "local"},
        "model-b": {"path": "models/model-b.gguf", "source": "local"},
        "model-c": {"path": "models/model-c.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    # `free`=None models an install where VRAM cannot be measured. `status` (default
    # GPU_PROBE_OK via probe_double) lets a caller simulate a probe that did NOT
    # complete (GPU_PROBE_TIMEOUT/BUSY), in which case any `free` here is the frozen
    # last-known-good the guard would have served.
    info = {"total": 16 * 1024 ** 3}
    if free is not None:
        info["free"] = free
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double(lambda: dict(info), status=status))
    monkeypatch.setattr(
        hs, "_engine_factory",
        lambda name: FakeEngine(name, fails_to_fit=fails_to_fit))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None


def _chat(client, model):
    return client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })


def test_handler_pins_engine_before_inlet(monkeypatch):
    """CRIT-1: the engine is pinned (active_requests>=1) before the inlet runs,
    which is the window a concurrent load could otherwise evict it in."""
    _install_fakes(monkeypatch, free=10 * 1024 ** 3)
    app = hs.create_app(None)

    observed = {}

    def inlet_hook(messages, ctx):
        eng = hs._engines.get(ctx.model_id)
        # run_inlet swallows exceptions, so RECORD (do not assert) here.
        observed["active_requests"] = getattr(eng, "active_requests", None) if eng else "no-engine"
        return messages

    app.state.chat_pipeline.add_hook("inlet", inlet_hook)
    client = TestClient(app)

    r = _chat(client, "model-a")
    assert r.status_code == 200, r.text
    assert observed["active_requests"] == 1, (
        "engine must be pinned before the inlet runs; observed "
        f"{observed['active_requests']!r} (0 means the eviction-race window is open)")
    # And the pin is released once the request completes.
    assert hs._engines["model-a"].active_requests == 0


def test_unmeasurable_vram_is_single_resident(monkeypatch):
    """CRIT-2: with no measurable free VRAM, switching evicts idle models instead
    of stacking them until the driver OOMs."""
    _install_fakes(monkeypatch, free=None)
    app = hs.create_app(None)
    client = TestClient(app)

    for m in ("model-a", "model-b", "model-c"):
        assert _chat(client, m).status_code == 200

    loaded = sorted(n for n, e in hs._engines.items() if e.loaded)
    assert loaded == ["model-c"], (
        f"unmeasurable VRAM must stay single-resident; loaded={loaded} "
        "(stacking all three is the OOM bug)")


def test_measurable_vram_allows_coexistence(monkeypatch):
    """Guard: the CRIT-2 fix must NOT regress the intended multi-model behavior -
    with plenty of measurable free VRAM, several small models coexist."""
    _install_fakes(monkeypatch, free=10 * 1024 ** 3)
    app = hs.create_app(None)
    client = TestClient(app)

    for m in ("model-a", "model-b", "model-c"):
        assert _chat(client, m).status_code == 200

    loaded = sorted(n for n, e in hs._engines.items() if e.loaded)
    assert loaded == ["model-a", "model-b", "model-c"], (
        f"ample VRAM must keep multi-model coexistence; loaded={loaded}")


def _knobs(monkeypatch, **over):
    """Override just the residency knobs, keeping the rest of the real config."""
    from localm.config import load_config as _real
    base = _real()

    def fake():
        cfg = dict(base)
        cfg.update(over)
        return cfg

    monkeypatch.setattr("localm.config.load_config", fake)


def test_resident_cap_bounds_coexistence(monkeypatch):
    """max_resident_models caps how many models stay loaded even when there is
    ample VRAM for all of them (test_measurable_vram_allows_coexistence is the
    same scenario with no cap, and keeps all three)."""
    _install_fakes(monkeypatch, free=10 * 1024 ** 3)
    _knobs(monkeypatch, max_resident_models=2)
    app = hs.create_app(None)
    client = TestClient(app)

    for m in ("model-a", "model-b", "model-c"):
        assert _chat(client, m).status_code == 200

    loaded = sorted(n for n, e in hs._engines.items() if e.loaded)
    assert loaded == ["model-b", "model-c"], (
        f"cap of 2 must keep only the 2 most recent; loaded={loaded}")


def test_pinned_model_survives_an_over_cap_load(monkeypatch):
    _install_fakes(monkeypatch, free=10 * 1024 ** 3)
    _knobs(monkeypatch, max_resident_models=1, pinned_models=["model-a"])
    app = hs.create_app(None)
    client = TestClient(app)

    assert _chat(client, "model-a").status_code == 200
    assert _chat(client, "model-b").status_code == 200

    loaded = sorted(n for n, e in hs._engines.items() if e.loaded)
    assert loaded == ["model-a", "model-b"], (
        f"a pinned model must never be the victim; loaded={loaded}")


def test_unmet_cap_never_yanks_a_sibling_instance(monkeypatch):
    """A cap is a user PREFERENCE; free VRAM is the safety constraint.

    With the cap exceeded but every peer pinned, there is nothing local to
    evict. That must NOT fall through to the VRAM-exhaustion handling, which
    asks a SIBLING localm instance to dump ITS models - destroying another
    instance's work to satisfy a local preference, over VRAM that was never
    short - and then logs the miss as a whole-model VRAM shortfall, a reason
    the readings do not support.
    """
    _install_fakes(monkeypatch, free=10 * 1024 ** 3)
    _knobs(monkeypatch, max_resident_models=1, pinned_models=["model-a"])
    asked = []
    monkeypatch.setattr(hs, "_attempt_cooperative_unload",
                        lambda **kw: asked.append(kw) or False)
    app = hs.create_app(None)
    client = TestClient(app)

    assert _chat(client, "model-a").status_code == 200
    assert _chat(client, "model-b").status_code == 200
    assert asked == [], (
        "an unmeetable resident cap asked a sibling instance to unload, even "
        f"though free VRAM was sufficient: {asked}")


def test_a_pin_never_costs_a_sibling_instance_its_models(monkeypatch):
    """A pin is a local preference, like the cap - and must not be paid for out
    of ANOTHER localm instance's VRAM.

    Here free VRAM is genuinely short (unlike the cap case), so local eviction
    really is needed; the only idle peer is simply pinned. Escalating to
    _attempt_cooperative_unload would ask a sibling instance to dump its models
    to satisfy this instance's pin. Deferring to the backend's own sizing
    (partial offload) honors the pin at OUR expense instead of theirs.
    """
    _install_fakes(monkeypatch, free=2 * 1024 ** 3)
    _knobs(monkeypatch, pinned_models=["model-a"])
    asked = []
    monkeypatch.setattr(hs, "_attempt_cooperative_unload",
                        lambda **kw: asked.append(kw) or False)
    app = hs.create_app(None)
    client = TestClient(app)

    assert _chat(client, "model-a").status_code == 200
    # The FIRST load legitimately asks a peer: VRAM is short and nothing is
    # resident yet, so local eviction is empty for reasons that have nothing to
    # do with pinning. Measure only the asks made during the SECOND load, where
    # the pinned model-a is the sole reason nothing local can be freed.
    asked_before = len(asked)
    assert _chat(client, "model-b").status_code == 200

    assert len(asked) == asked_before, (
        "a pin made this instance yank a sibling's models: "
        f"{asked[asked_before:]}")
    assert hs._engines["model-a"].loaded, "the pinned model must survive"


def test_an_unpinned_shortfall_still_asks_a_peer(monkeypatch):
    """Guard on the test above: with nothing pinned and the only peer BUSY,
    local eviction is exhausted for a real reason, and the cooperative path
    must still run - the pin guard must not disable it wholesale."""
    _install_fakes(monkeypatch, free=2 * 1024 ** 3)
    _knobs(monkeypatch)
    asked = []
    monkeypatch.setattr(hs, "_attempt_cooperative_unload",
                        lambda **kw: asked.append(kw) or False)
    app = hs.create_app(None)
    client = TestClient(app)

    assert _chat(client, "model-a").status_code == 200
    hs._engines["model-a"].active_requests = 1        # busy, not evictable
    asked_before = len(asked)                         # ignore the first load's ask
    assert _chat(client, "model-b").status_code == 200

    assert len(asked) > asked_before, (
        "an unpinned, genuinely exhausted shortfall must still ask a peer - "
        "the pin guard must not disable the cooperative path wholesale")


def test_busy_chat_peer_not_evicted_but_new_load_still_succeeds(monkeypatch):
    """Mirrors test_busy_embedder_not_evicted_for_chat_load's proof, for a
    busy CHAT peer instead of the shared embedder: with a resident engine
    pinned (active_requests>0) so local eviction cannot free it, and no
    cooperative peer configured, local+cooperative eviction is fully
    exhausted. The incoming load must still succeed via the backend's own
    partial offload rather than refuse, and the busy peer must survive
    untouched."""
    _install_fakes(monkeypatch, free=3 * 1024 ** 3)
    app = hs.create_app(None)
    client = TestClient(app)

    busy = FakeEngine("model-a")
    busy._loaded = True
    busy.active_requests = 1
    hs._engines["model-a"] = busy
    hs._engines_lru.append("model-a")

    r = _chat(client, "model-b")
    assert r.status_code == 200, (
        f"a busy chat peer correctly stays resident, but the incoming load "
        f"should still succeed via the backend's own partial offload rather "
        f"than the removed 'all other loaded models are busy' refusal: {r.text}")
    assert hs._engines["model-a"] is busy and busy.loaded, (
        "the busy chat engine must not be evicted")
    assert hs._engines["model-b"].loaded


def _install_embedder_fakes(monkeypatch, *, free_with_embedder, free_after_evict,
                            embedder_active=0):
    """A fake shared embedder resident in VRAM, plus the same single-model chat
    registry `_install_fakes` uses. `free_with_embedder` is what the probe
    reports while the embedder is still resident; a successful
    `reset_embedder()` mutates the reading to `free_after_evict`, simulating
    the VRAM it actually holds landing back as free - mirroring how a real
    native unload changes what the next probe reads.

    The fake `reset_embedder(force=True)` mirrors the real function's
    force-gated, single-call atomic check-and-clear contract (embedder.py) so
    these tests exercise the same call shape switch_engine actually uses
    (`functools.partial(reset_embedder, force=False)`), and counts calls in
    `state["reset_calls"]` so a test can prove the eviction branch was
    genuinely entered and consulted the pin, not merely that eviction did
    not happen (which is also true of code that never looks at the embedder
    at all)."""
    fake_registry = {"model-a": {"path": "models/model-a.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)

    state = {"embedder_loaded": True, "free": free_with_embedder,
             "active_requests": embedder_active, "reset_calls": 0}

    monkeypatch.setattr(
        "localm.discover.vram_info",
        probe_double(lambda: {"total": 16 * 1024 ** 3, "free": state["free"]}))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    monkeypatch.setattr(
        "localm.inference.embedder.loaded_dim",
        lambda: (768 if state["embedder_loaded"] else None))
    monkeypatch.setattr(
        "localm.inference.embedder.active_requests",
        lambda: state["active_requests"])

    def _fake_reset_embedder(force=True):
        state["reset_calls"] += 1
        if not state["embedder_loaded"]:
            return False
        if not force and state["active_requests"] > 0:
            return False
        state["embedder_loaded"] = False
        state["free"] = free_after_evict
        return True

    monkeypatch.setattr("localm.inference.embedder.reset_embedder", _fake_reset_embedder)
    return state


def test_idle_embedder_evicted_to_make_room_for_chat_load(monkeypatch):
    """Reported bug: an embedding run leaves the shared embedder resident in
    VRAM; loading a chat model that would fit once the embedder is freed must
    NOT 503 - switch_engine's auto-eviction must free the idle embedder the
    same way it frees an idle chat engine, not just refuse."""
    state = _install_embedder_fakes(
        monkeypatch, free_with_embedder=3 * 1024 ** 3,
        free_after_evict=9 * 1024 ** 3)
    app = hs.create_app(None)
    client = TestClient(app)

    r = _chat(client, "model-a")
    assert r.status_code == 200, r.text
    assert hs._engines["model-a"].loaded is True
    assert state["embedder_loaded"] is False
    assert state["reset_calls"] == 1, (
        "the embedder-eviction branch must be attempted exactly once per load")


def test_busy_embedder_not_evicted_for_chat_load(monkeypatch):
    """The embedder pin: a request mid-embed() (active_requests>0) must not
    have its embedder freed out from under it just because a chat load is
    short on VRAM.

    The chat load itself still SUCCEEDS despite the pin: switch_engine does
    not hard-refuse on its own crude whole-model estimate once local +
    cooperative eviction is exhausted - it defers to the backend, which fits
    the model via partial GPU-layer offload using whatever is left (3 GB
    here), without ever needing the embedder's VRAM. So a resource the pin
    protects stays protected, AND the request the pin would have starved
    still gets served.

    Asserts reset_calls == 1, not just embedder_loaded staying True: a
    reset_embedder(force=False) call that correctly declines because the
    embedder is busy is indistinguishable, via embedder_loaded alone, from
    the eviction branch never having been reached at all (e.g. code with no
    embedder-eviction feature) - both leave embedder_loaded True. reset_calls
    proves the branch actually executed and consulted the pin."""
    state = _install_embedder_fakes(
        monkeypatch, free_with_embedder=3 * 1024 ** 3,
        free_after_evict=9 * 1024 ** 3, embedder_active=1)
    app = hs.create_app(None)
    client = TestClient(app)

    r = _chat(client, "model-a")
    assert r.status_code == 200, (
        f"the busy embedder correctly stays resident, but the chat load "
        f"should still succeed via the backend's own partial offload rather "
        f"than needlessly refusing: {r.text}")
    assert state["embedder_loaded"] is True, (
        "a busy (pinned) embedder must not be evicted")
    assert state["reset_calls"] == 1, (
        "the eviction branch must have been reached and consulted the pin, "
        "not merely have skipped the embedder entirely")


def test_idle_embedder_evicted_for_split_per_device_shortfall(monkeypatch, tmp_path):
    """The shared embedder is resident on a GPU that is also part of a
    configured chat-model split, and it is specifically its PER-DEVICE share
    (discover.gpu_split_shortfall), not just the aggregate, that goes from
    short to sufficient once the embedder is evicted - so the
    embedder-eviction branch composes with the split-aware gate, not just the
    plain single-GPU path the other embedder tests here use. Ratios are PINNED
    equal: with them unset the auto free-VRAM-proportional split shrinks GPU
    0's share below its free and no per-device pressure exists to drive this
    eviction."""
    model_a_file = tmp_path / "model-a.gguf"
    fake_registry = {"model-a": {"path": str(model_a_file), "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (str(model_a_file), "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {**base_cfg, "gpu_split_indices": [0, 1],
                 "gpu_split_ratios": [1.0, 1.0]})

    # GPU 0 holds the embedder: short on its own ~equal-split share of the
    # ~6.15 GiB aggregate threshold (4 GiB default file size * 1.2 + 1 GiB
    # headroom) while the embedder is resident, sufficient once evicted. GPU 1 is
    # ample throughout, so the AGGREGATE (32 GiB) is already past the threshold
    # before any eviction and only the PER-DEVICE gate blocks the load.
    state = {"embedder_loaded": True, "reset_calls": 0}

    def _list_gpus():
        gpu0_free = (2 if state["embedder_loaded"] else 6) * 1024 ** 3
        return [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": gpu0_free},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
        ]

    monkeypatch.setattr("localm.discover.list_gpus", probe_double(_list_gpus))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    monkeypatch.setattr(
        "localm.inference.embedder.loaded_dim",
        lambda: (768 if state["embedder_loaded"] else None))
    monkeypatch.setattr("localm.inference.embedder.active_requests", lambda: 0)

    def _fake_reset_embedder(force=True):
        state["reset_calls"] += 1
        if not state["embedder_loaded"]:
            return False
        state["embedder_loaded"] = False
        return True

    monkeypatch.setattr("localm.inference.embedder.reset_embedder", _fake_reset_embedder)

    app = hs.create_app(None)
    client = TestClient(app)
    r = _chat(client, "model-a")
    assert r.status_code == 200, (
        f"evicting the idle embedder must relieve GPU 0's per-device split "
        f"shortfall, not just the aggregate: {r.text}")
    assert hs._engines["model-a"].loaded
    assert state["embedder_loaded"] is False
    assert state["reset_calls"] == 1


def _mb_figure_in(text):
    """True if *text* quotes a concrete free-VRAM figure ('<N> MB free'). The
    gate must quote one only for a reading it actually measured."""
    import re
    return re.search(r"\d+\s*MB free", text) is not None


class TestInconclusiveProbeDoesNotSkipTheGate:
    """`measurable` must not conflate 'this box CANNOT measure free VRAM'
    (permanent -> best-effort load is correct) with 'this PROBE did not
    complete' (transient -> NOT a licence to skip the VRAM check). A timed-out
    probe serving free=None -> measurable=False -> `if not measurable: break`
    loads with NO VRAM CHECK on a fresh server with nothing to evict - the
    first load after a server start on any box whose cold driver init overruns
    the probe cap. These pin the fixed behaviour by forcing the probe status.
    """

    def test_inconclusive_probe_refuses_instead_of_loading_unchecked(self, monkeypatch):
        """The prize: a timed-out probe on a fresh server must REFUSE, not load
        blind. Negative-tested: revert to `if not measurable: break` and this
        returns 200 (the unguarded-first-load bug) instead of 503."""
        from localm.discover import GPU_PROBE_TIMEOUT
        # Probe did not complete and served no reading (cold init, no last-known-good).
        _install_fakes(monkeypatch, free=None, status=GPU_PROBE_TIMEOUT)
        app = hs.create_app(None)
        client = TestClient(app)

        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            "an inconclusive (timed-out) probe on a fresh server must refuse, not "
            f"load with no VRAM check; got {r.status_code}: {r.text[:200]}")
        assert not hs._engines.get("model-a", FakeEngine("x")).loaded, (
            "the model must NOT have loaded on an unmeasured probe")

    def test_inconclusive_refusal_quotes_no_figure(self, monkeypatch):
        """The inconclusive 503 must not state a free-VRAM figure it never
        measured. Contrast test_measured_refusal_quotes_the_figure below."""
        from localm.discover import GPU_PROBE_TIMEOUT
        _install_fakes(monkeypatch, free=None, status=GPU_PROBE_TIMEOUT)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert not _mb_figure_in(r.text), (
            "the inconclusive-probe 503 quotes an MB figure it did not measure "
            f"(rule 5): {r.text[:200]}")

    def test_stale_high_reading_does_not_permit_a_load(self, monkeypatch):
        """The OOM direction: a timed-out probe that happens to serve a HIGH stale
        free reading must NOT permit a load. Negative-tested: drop the probe_ok
        requirement from the permit check and the 15 GB stale reading passes the
        fit test -> the model loads (200) on top of whatever really holds the GPU."""
        from localm.discover import GPU_PROBE_TIMEOUT
        # 15 GB 'free' is ample for the ~5.8 GB the load needs, but the probe TIMED
        # OUT, so that figure is a frozen last-known-good, not a live measurement.
        _install_fakes(monkeypatch, free=15 * 1024 ** 3, status=GPU_PROBE_TIMEOUT)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            "a stale-HIGH reading from a timed-out probe must not be trusted to "
            f"permit a load; got {r.status_code}")
        assert not hs._engines.get("model-a", FakeEngine("x")).loaded

    def test_measured_low_reading_defers_to_the_backend_instead_of_refusing(
            self, monkeypatch):
        """Contrast with the stale-HIGH-reading test above: a genuinely
        MEASURED (probe_ok) reading, even a low one, is trustworthy enough to
        let the backend attempt the load - switch_engine's own crude
        whole-model estimate (~5.8 GB) is not met by 2 GB free, but that does
        not mean refuse outright: the backend's own sizing can still fit a
        partial-offload load in 2 GB, and this GPU's 16 GB total easily covers
        it. The stale case above must still refuse (an untrustworthy reading);
        this one, being trustworthy, gets to try."""
        from localm.discover import GPU_PROBE_OK
        _install_fakes(monkeypatch, free=2 * 1024 ** 3, status=GPU_PROBE_OK)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a measured (not stale) low reading should defer to the "
            f"backend's own sizing rather than refuse outright: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_backend_refusal_still_produces_a_clean_message(self, monkeypatch):
        """The backstop: when the backend's OWN sizing decides the model
        genuinely cannot fit even at 0 GPU layers (GgufBackend._check_vram
        raising because need > total VRAM - see llamacpp/_sizing.py), the
        failure must still reach the caller as a clean, specific message, not
        a raw/generic error. Hits /v1/chat/completions specifically: this
        OpenAI-compatible route does NOT wrap switch_engine/get_engine in its
        own try/except (unlike the GUI's load-model button), so without
        switch_engine's RuntimeError->HTTPException(503) conversion this exact
        case falls through to Starlette's generic "Internal server error"
        handler, discarding the real reason."""
        from localm.discover import GPU_PROBE_OK
        _install_fakes(monkeypatch, free=2 * 1024 ** 3, status=GPU_PROBE_OK,
                       fails_to_fit=True)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 503, r.text
        assert "cannot fit regardless" in r.text, (
            f"a genuine backend refusal must surface its real message, not a "
            f"generic 500: {r.text}")
        assert not hs._engines.get("model-a", FakeEngine("x")).loaded

    def test_cannot_measure_still_loads_best_effort(self, monkeypatch):
        """Guard the permanent case: a box that genuinely cannot report free VRAM
        (probe COMPLETES but returns no 'free' - CPU-only / GGUF-only / registry
        tier) must still load best-effort, NOT be refused. The inconclusive refusal
        must not brick these boxes."""
        from localm.discover import GPU_PROBE_OK
        _install_fakes(monkeypatch, free=None, status=GPU_PROBE_OK)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            "a box that cannot measure VRAM (probe OK, no free) must load "
            f"best-effort, not be refused; got {r.status_code}: {r.text[:200]}")
        assert hs._engines["model-a"].loaded


def _flaky_probe_double(monkeypatch, *, timeout_calls, free_once_ok):
    """A discover.vram_info double whose STATUS changes across calls: TIMEOUT
    for the first *timeout_calls* calls, then GPU_PROBE_OK reporting
    *free_once_ok* bytes free from then on. probe_double's status is fixed at
    wrap time, so this needs its own counter rather than that helper."""
    from localm.discover import GPU_PROBE_OK, GPU_PROBE_TIMEOUT
    calls = {"n": 0}

    def _double(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= timeout_calls:
            value = {"total": 16 * 1024 ** 3}
            status = GPU_PROBE_TIMEOUT
        else:
            value = {"total": 16 * 1024 ** 3, "free": free_once_ok}
            status = GPU_PROBE_OK
        if kwargs.get("return_status"):
            return value, status
        return value

    monkeypatch.setattr("localm.discover.vram_info", _double)
    return calls


class TestInconclusiveProbeRetriesBeforeRefusing:
    """A single inconclusive probe must not immediately hand the caller a 503
    to retry themselves: the exact scenario this guards is a transient
    concurrent-import race (see discover._torch_gpus_resident_bounded) that
    typically clears within a couple of seconds, well inside
    _INCONCLUSIVE_LOAD_RETRIES retries."""

    def test_a_probe_that_clears_within_the_retry_budget_loads_successfully(
            self, monkeypatch):
        """The property the user actually wants: a transient inconclusive
        reading must resolve into a successful load with NO caller-visible
        error, not a 503 the caller has to retry by hand."""
        monkeypatch.setattr(hs, "_INCONCLUSIVE_LOAD_RETRY_DELAY", 0)
        _install_fakes(monkeypatch, free=None)  # baseline: registry, engine factory
        calls = _flaky_probe_double(
            monkeypatch, timeout_calls=hs._INCONCLUSIVE_LOAD_RETRIES,
            free_once_ok=10 * 1024 ** 3)

        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")

        assert r.status_code == 200, (
            f"a probe that clears within the retry budget must load "
            f"transparently, not surface a 503 for the caller to retry: "
            f"{r.text[:200]}")
        assert hs._engines["model-a"].loaded
        assert calls["n"] == hs._INCONCLUSIVE_LOAD_RETRIES + 1, (
            f"expected exactly {hs._INCONCLUSIVE_LOAD_RETRIES + 1} probe "
            f"attempts (the retries did not run as designed); got {calls['n']}")

    def test_exhausting_every_retry_still_refuses_cleanly(self, monkeypatch):
        """When the condition genuinely never clears, the caller still gets a
        503 (never a silent bad load) - but it must not instruct the caller to
        do anything, and must say retries were already attempted."""
        monkeypatch.setattr(hs, "_INCONCLUSIVE_LOAD_RETRY_DELAY", 0)
        _install_fakes(monkeypatch, free=None)
        calls = _flaky_probe_double(
            monkeypatch, timeout_calls=10 ** 6, free_once_ok=10 * 1024 ** 3)

        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")

        assert r.status_code == 503, r.text
        assert calls["n"] == hs._INCONCLUSIVE_LOAD_RETRIES + 1, (
            f"expected exactly {hs._INCONCLUSIVE_LOAD_RETRIES + 1} probe "
            f"attempts before giving up; got {calls['n']}")
        assert str(hs._INCONCLUSIVE_LOAD_RETRIES + 1) in r.text, (
            f"the refusal must state how many attempts were already made "
            f"automatically: {r.text[:200]}")
        for instruction in ("restart the gpu app", "retry shortly",
                            "unload another model"):
            assert instruction not in r.text.lower(), (
                f"the refusal must state what happened, not instruct the "
                f"user what to do next ({instruction!r} found): {r.text[:200]}")
        assert not hs._engines.get("model-a", FakeEngine("x")).loaded


def _fake_stat_size(monkeypatch, path: Path, size_bytes: int):
    """Make ``path.stat().st_size`` report *size_bytes* without writing that
    many real bytes to disk. *path* must already exist (a real, tiny
    placeholder file) so ``Path.is_file()`` - which itself calls ``.stat()``
    and checks ``S_ISREG(st_mode)`` - keeps working: only ``st_size`` is
    swapped out; every other field (including ``st_mode``) comes from the
    real underlying stat of the real tiny file.

    Drives the exact same code path (``p.is_file()`` True,
    ``file_size = p.stat().st_size``) with zero real disk cost and nothing to
    orphan. Truncating a real file to the target size is not an option here:
    ``truncate()`` is NOT sparse on this platform, so a 15-40 GB target
    writes that many bytes for real and an interrupted run leaves them
    behind."""
    path.touch()
    real_stat = Path.stat

    def fake_stat(self, *, follow_symlinks=True):
        result = real_stat(self, follow_symlinks=follow_symlinks)
        if self == path:
            seq = (result.st_mode, result.st_ino, result.st_dev, result.st_nlink,
                   result.st_uid, result.st_gid, size_bytes,
                   result.st_atime, result.st_mtime, result.st_ctime)
            return os.stat_result(seq)
        return result

    monkeypatch.setattr(Path, "stat", fake_stat)


class TestSplitAwareCapacityGate:
    """vram_info() alone is single-GPU (see discover.py), so the pre-load
    refusal gate (switch_engine) must weigh a load against
    discover.vram_capacity() - the COMBINED total/free across a configured
    multi-GPU split - not just the single main GPU. A model too big for one
    GPU alone but that fits split across 2+ configured devices must load, not
    503; a model that still does not fit even combined must still be refused
    (no over-correction to "always assume it fits").

    Uses a real (but tiny) model file with a FAKED stat().st_size (see
    _fake_stat_size) to actually drive file_size = p.stat().st_size through
    switch_engine's real code path, rather than the "unregistered path ->
    fixed 4 GB" fallback other tests in this file rely on, so the assertions
    run against the same real, size-derived vram_required arithmetic."""

    # free_scope=device: resolve_auto_split_ratios() requires a device-global
    # reading on every configured device before computing a real proportion; an
    # untagged double declines to the equal-split fallback, which is numerically
    # identical here. The distinction still matters: switch_engine defers a
    # combined-short load to the backend's own sizing only when shares_adaptive
    # is True, never on the declined/equal-fallback path.
    _SPLIT_GPUS = [
        {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 14 * 1024 ** 3,
         "free_scope": "device"},
        {"index": 1, "name": "B", "total": 16 * 1024 ** 3, "free": 14 * 1024 ** 3,
         "free_scope": "device"},
    ]

    def _install(self, monkeypatch, tmp_path, *, size_bytes, gpus, gpu_split_indices,
                 fails_to_fit=False):
        model_file = tmp_path / "model-a.gguf"
        _fake_stat_size(monkeypatch, model_file, size_bytes)
        fake_registry = {"model-a": {"path": str(model_file), "source": "local"}}
        monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (str(model_file), "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
        # Overlay just gpu_split_indices onto the REAL (test-isolated) config
        # rather than replacing load_config() outright - create_app()/switch_engine
        # read other config keys too, and a stripped-down fake dict would break
        # those unrelated paths.
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()

        def _cfg():
            return {**base_cfg, "gpu_split_indices": gpu_split_indices}

        monkeypatch.setattr("localm.config.load_config", _cfg)
        monkeypatch.setattr("localm.discover.list_gpus", probe_double(gpus))
        monkeypatch.setattr(
            hs, "_engine_factory",
            lambda name: FakeEngine(name, fails_to_fit=fails_to_fit))
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        hs._engine = None
        hs._inference_sem = None

    def test_fits_combined_split_but_not_single_main_gpu_loads(
            self, monkeypatch, tmp_path):
        # 15 GB file -> vram_required ~= 18 GB (*1.2) + 1 GB headroom = 19 GB.
        # Exceeds either GPU's 14 GB free alone, but fits the 28 GB combined free.
        self._install(monkeypatch, tmp_path, size_bytes=15 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a model needing ~19GB should load via the 28GB COMBINED split "
            f"free, not be refused against one 14GB GPU alone: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_same_model_refused_without_a_configured_split(
            self, monkeypatch, tmp_path):
        """Guard: the fix must not regress to 'always assume combined capacity
        even with no split configured' - with NO split configured (single GPU
        only), this model is judged against that ONE GPU's real free VRAM
        (14 GB), not a fictional combined figure. It is not refused outright,
        though: switch_engine's own crude whole-model estimate (~19 GB)
        exceeds that 14 GB, but the backend's own partial-offload sizing can
        still fit this model on the ONE real GPU by putting some layers on
        CPU. fails_to_fit=False (the fake's default) stands in for that real
        capability."""
        self._install(monkeypatch, tmp_path, size_bytes=15 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS[:1], gpu_split_indices=None)
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a model too big for full offload on the single real GPU should "
            f"still load via the backend's own partial offload: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_exceeds_even_the_combined_split_defers_to_backend(
            self, monkeypatch, tmp_path):
        """Combined capacity is a bigger ceiling, not an unlimited one - but
        exceeding it is not a gate-level 503 with UNSET ratios: the auto
        free-VRAM-proportional split defers to the backend's own split-aware
        sizing, which partial-offloads or refuses with the accurate figure -
        the same posture as the single-GPU too-big case above. FakeEngine's
        default load() stands in for a successful partial offload; the
        pinned-ratio hard refusal and the backend's own clean refusal are
        covered by the auto-ratio suite."""
        # 40 GB file -> needs ~49 GB, exceeds the 28 GB combined free.
        self._install(monkeypatch, tmp_path, size_bytes=40 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"an auto-split combined-short load must defer to the backend's "
            f"split-aware sizing (partial offload), not 503: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_stale_split_index_not_currently_detected_falls_back_to_single_gpu(
            self, monkeypatch, tmp_path):
        """A gpu_split_indices referencing a device that vanished (e.g. it was
        unplugged) must degrade to single-GPU capacity (resolve_gpu_split's own
        contract), not silently keep using a combined number for hardware that
        is no longer there - proven by the SAME real-single-GPU outcome as the
        no-split test above (this model loads via partial offload against the
        ONE real 14 GB GPU), not the 28 GB a still-combined (stale) reading
        would have granted."""
        self._install(monkeypatch, tmp_path, size_bytes=15 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS[:1],   # device 1 no longer detected
                      gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a split referencing a since-removed GPU must degrade to single-"
            f"GPU capacity and still let the backend try (partial offload "
            f"against the ONE real 14 GB GPU): {r.text}")
        assert hs._engines["model-a"].loaded


class TestPerDeviceSplitFitGate:
    """vram_capacity()'s AGGREGATE check alone is not enough for a GGUF-backend
    load - apply_gpu_split() divides a model by a STATIC per-config ratio with
    no live per-device capacity awareness of its own when gpu_split_ratios is
    PINNED (unlike the HF backend's device_map="auto", which self-corrects from
    live per-device free VRAM instead). An asymmetric split - e.g. another
    already-loaded model sits on one configured device more than another - can
    then pass the aggregate check while one device's actual pinned share is
    short, reaching the native loader with too little room on that device.
    discover.gpu_split_shortfall() is the per-device gate that catches this
    before the native load, refusing cleanly instead of risking a native
    crash (llama.cpp can hard-abort rather than raise). With ratios UNSET
    the loader adapts (auto free-VRAM-proportional split), so the
    static-share cases here pin ratios explicitly; the auto behavior is
    covered below."""

    def _install(self, monkeypatch, tmp_path, *, filename, gpus, gpu_split_indices,
                 gpu_split_ratios=None, fails_to_fit=False):
        model_file = tmp_path / filename
        # Unregistered-on-disk path: file_size falls back to the code's own
        # documented 4 GB default, so vram_required is always int(4 GiB * 1.2)
        # ~= 5.15 GiB, plus the fixed 1 GiB headroom ~= 6.15 GiB aggregate
        # threshold - comfortably covered by the GPUs below, so only the
        # PER-DEVICE gate can block these tests.
        fake_registry = {"model-a": {"path": str(model_file), "source": "local"}}
        monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (str(model_file), "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()

        def _cfg():
            return {**base_cfg, "gpu_split_indices": gpu_split_indices,
                    "gpu_split_ratios": gpu_split_ratios}

        monkeypatch.setattr("localm.config.load_config", _cfg)
        monkeypatch.setattr("localm.discover.list_gpus", probe_double(gpus))
        monkeypatch.setattr(
            hs, "_engine_factory",
            lambda name: FakeEngine(name, fails_to_fit=fails_to_fit))
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        hs._engine = None
        hs._inference_sem = None

    def test_aggregate_fits_but_one_device_short_is_refused_pinned(
            self, monkeypatch, tmp_path):
        # GPU0: 2 GiB free (short of its ~2.58 GiB PINNED equal-split share of
        # the ~5.15 GiB required). GPU1: 30 GiB free (comfortably covers its
        # share). Combined 32 GiB free is well past the ~6.15 GiB aggregate
        # threshold, so only the per-device gate refuses this.
        #
        # With PINNED ratios the loader applies the user's static shares
        # regardless of live free VRAM, and the backend's sizing
        # (_auto_gpu_layers/_check_vram, llamacpp/_sizing.py) budgets the split's
        # COMBINED capacity rather than one pinned share, so the refusal stays
        # hard here.
        gpus = [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
        ]
        self._install(monkeypatch, tmp_path, filename="model-a.gguf",
                      gpus=gpus, gpu_split_indices=[0, 1],
                      gpu_split_ratios=[1.0, 1.0])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            f"aggregate free (32GiB) covers the ~6.15GiB threshold, but GPU 0's "
            f"own pinned equal share does not fit its 2GiB free - must still "
            f"refuse rather than reach the native loader: {r.text}")
        assert "configured split" in r.text
        assert "GPU 0" in r.text

    def test_aggregate_fits_one_device_occupied_loads_via_auto_ratio(
            self, monkeypatch, tmp_path):
        """THE feature's headline case, end to end through switch_engine: the
        SAME asymmetric occupancy that the pinned test above refuses now
        LOADS with ratios unset - the auto free-VRAM-proportional split gives
        the occupied GPU 0 only its ~6% share (~0.4 GiB vs 2 GiB free), so no
        device is short and no eviction pressure exists.

        free_scope=device on both entries: resolve_auto_split_ratios() now
        requires a device-global reading on every configured device before
        computing a real proportion (see its TRUSTWORTHINESS docstring
        section) - an untagged double here would decline to the SAFE equal
        split instead, which genuinely 503s in this asymmetric scenario
        (that IS the point of the pinned test above), defeating what this
        test exists to prove."""
        from localm.discover import FREE_SCOPE_DEVICE
        gpus = [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3,
             "free_scope": FREE_SCOPE_DEVICE},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3,
             "free_scope": FREE_SCOPE_DEVICE},
        ]
        self._install(monkeypatch, tmp_path, filename="model-a.gguf",
                      gpus=gpus, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"with ratios unset, the auto split adapts each device's share to "
            f"its free VRAM - this load fits and must not 503: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_per_device_fit_satisfied_with_asymmetric_ratio_loads(self, monkeypatch, tmp_path):
        """Guard: the new gate must not over-correct into refusing every
        asymmetric setup - a deliberately lopsided gpu_split_ratios that DOES
        fit each device's real free VRAM must still load normally."""
        # ~5.15 GiB required, ratio 1:4 (GPU0:GPU1) -> GPU0 needs ~1.03 GiB
        # (has 2 GiB, fine), GPU1 needs ~3.84 GiB (has 30 GiB, fine).
        gpus = [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
        ]
        self._install(monkeypatch, tmp_path, filename="model-a.gguf", gpus=gpus,
                      gpu_split_indices=[0, 1], gpu_split_ratios=[1.0, 4.0])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a ratio that genuinely fits each device's free VRAM must load: {r.text}")
        assert hs._engines["model-a"].loaded

    def test_shortfall_triggers_additional_eviction_beyond_aggregate(
            self, monkeypatch, tmp_path):
        """The two gates COMPOSE: the eviction loop's exit condition is
        aggregate-fits AND per-device-fits, not aggregate alone. A second,
        real model ("model-b") loads first and occupies GPU 0 (simulating its
        real footprint by shrinking GPU 0's reported free once its .load()
        actually runs); model-a's load then hits a per-device shortfall on
        GPU 0 even though AGGREGATE free is already sufficient, and must
        evict the idle model-b to relieve it - proving the per-device gate
        genuinely drives additional eviction, not just a one-shot refusal.
        Ratios PINNED equal: unset, the auto proportional split would shrink
        GPU 0's share below its post-load free and remove the very pressure
        this test exists to exercise."""
        model_a_file = tmp_path / "model-a.gguf"
        fake_registry = {
            "model-a": {"path": str(model_a_file), "source": "local"},
            "model-b": {"path": "models/model-b.gguf", "source": "local"},
        }
        monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
        monkeypatch.setattr(
            "localm.model_manager.get_model_info",
            lambda name: (str(model_a_file), "hint") if name == "model-a"
            else ("models/model-b.gguf", "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {**base_cfg, "gpu_split_indices": [0, 1],
                     "gpu_split_ratios": [1.0, 1.0]})

        # Three-phase GPU state, driven by model-b's REAL load()/unload()
        # events (not a call-count guess): plenty of room on both devices
        # while nothing is loaded -> GPU 0 shrinks once model-b actually
        # loads (simulating its real footprint) -> GPU 0 partially recovers
        # once model-b is evicted for model-a.
        state = {"phase": "empty"}

        def _list_gpus():
            if state["phase"] == "evicted":
                return [
                    {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 6 * 1024 ** 3},
                    {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
                ]
            if state["phase"] == "b_loaded":
                return [
                    {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3},
                    {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
                ]
            return [   # "empty": nothing loaded yet, plenty of room everywhere
                {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 16 * 1024 ** 3},
                {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
            ]

        fake_b = FakeEngine("model-b")
        real_b_load, real_b_unload = fake_b.load, fake_b.unload

        def _b_load_and_shrink_gpu0():
            real_b_load()
            state["phase"] = "b_loaded"

        def _b_unload_and_relieve_gpu0():
            real_b_unload()
            state["phase"] = "evicted"

        fake_b.load = _b_load_and_shrink_gpu0
        fake_b.unload = _b_unload_and_relieve_gpu0

        monkeypatch.setattr("localm.discover.list_gpus", probe_double(_list_gpus))
        monkeypatch.setattr(
            hs, "_engine_factory",
            lambda name: fake_b if name == "model-b" else FakeEngine(name))
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        hs._engine = None
        hs._inference_sem = None

        app = hs.create_app(None)
        client = TestClient(app)

        # model-b loads first, while both devices have plenty of room -
        # its own load flips the GPU state to "b_loaded" (shrinking GPU 0).
        assert _chat(client, "model-b").status_code == 200
        assert hs._engines["model-b"].loaded

        # model-a's load now hits GPU 0's per-device shortfall (aggregate
        # free is 2+30=32GiB, well over the ~6.15GiB threshold) - it must
        # evict the idle model-b to relieve GPU 0, not refuse outright.
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"model-b must be evicted to relieve GPU 0's per-device shortfall "
            f"even though aggregate free (32GiB) was already enough: {r.text}")
        assert hs._engines["model-a"].loaded
        assert "model-b" not in hs._engines, "model-b should have been evicted"

    def test_non_gguf_path_skips_the_per_device_gate(self, monkeypatch, tmp_path):
        """The per-device gate only applies to the GGUF/llama.cpp backend
        (identified by file extension) - the HF backend's device_map="auto"
        already self-corrects from live per-device free VRAM, so applying
        this same simplistic proportional-by-ratio estimate to an HF load
        would risk FALSE refusals (accelerate's real bin-packing can be
        smarter than a uniform ratio assumption). Same asymmetric GPUs as the
        refused GGUF case above, but a non-.gguf path - must load, not 503."""
        gpus = [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
        ]
        self._install(monkeypatch, tmp_path, filename="model-a",  # no .gguf suffix
                      gpus=gpus, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 200, (
            f"a non-GGUF path must skip the per-device gate entirely: {r.text}")
        assert hs._engines["model-a"].loaded


@pytest.mark.anyio
async def test_switch_engine_vram_probe_does_not_block_event_loop(monkeypatch):
    """switch_engine's pre-load eviction loop calls
    discover.vram_capacity()/gpu_split_shortfall(), which route through the
    deadline-bounded discover.list_gpus() probe - but a bounded (up to ~4s)
    block is still a block if it runs directly on the single event loop,
    freezing every OTHER concurrent coroutine for that long. This loop can
    re-probe multiple times per eviction.

    Proven with an independent heartbeat coroutine ticking on a fixed
    interval, NOT by racing against switch_engine's own request-completion
    timing, which depends on how many other awaits happen to run first and is
    not a reliable signal. If the event loop is genuinely blocked, the
    heartbeat CANNOT tick during that window - a deterministic, unambiguous
    signal regardless of switch_engine's internal scheduling."""
    fake_registry = {"model-a": {"path": "models/model-a.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: ("models/model-a.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    # A configured split means BOTH vram_capacity() and gpu_split_shortfall()
    # (a GGUF path) will each independently call list_gpus() - exercising both
    # of this loop's executor-wrapped probe call sites, not just one.
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    monkeypatch.setattr(
        "localm.config.load_config",
        lambda: {**base_cfg, "gpu_split_indices": [0, 1]})
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    probe_entered = threading.Event()
    release = threading.Event()

    def _blocking_probe():
        # Simulate a slow/wedged GPU driver call. If vram_capacity() (or
        # gpu_split_shortfall()) ran this on the event loop instead of a
        # worker thread, the whole server would freeze here until release.
        probe_entered.set()
        release.wait(3)
        return [{"index": 0, "name": "X", "total": 16 * 1024 ** 3, "free": 16 * 1024 ** 3},
                {"index": 1, "name": "Y", "total": 16 * 1024 ** 3, "free": 16 * 1024 ** 3}]

    monkeypatch.setattr("localm.discover._list_gpus_probe", _blocking_probe)

    ticks = {"n": 0}

    async def _heartbeat():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    hb_task = asyncio.ensure_future(_heartbeat())
    try:
        switch_task = asyncio.ensure_future(
            hs.switch_engine("model-a", hs._engine_factory, preempt=False))

        # A precondition wait (has the executor thread actually been scheduled by
        # the OS yet), not the property under test: the later ticks_before and
        # ticks_after assertions are what prove the event loop stayed responsive.
        # The budget is generous because under heavy parallel load the OS can
        # take seconds to give this test's own worker thread its first timeslice.
        probe_started = False
        for _ in range(1500):
            if probe_entered.is_set():
                probe_started = True
                break
            await asyncio.sleep(0.01)
        assert probe_started, "probe never started (executor not reached)"

        ticks_before = ticks["n"]
        await asyncio.sleep(0.5)   # the probe is still blocked (releases at 3s)
        ticks_after = ticks["n"]
        assert not switch_task.done(), (
            "switch_engine already completed while its own probe should "
            "still be blocked for ~3s in a worker thread - this means the "
            "probe call actually ran SYNCHRONOUSLY on the event loop (which "
            "also starved this test's own heartbeat/polling coroutines until "
            "the whole chain finished) rather than being offloaded via "
            "run_in_executor")
        assert ticks_after - ticks_before > 20, (
            f"the heartbeat barely advanced ({ticks_before} -> {ticks_after} "
            f"in 0.5s, expected ~50) while the VRAM probe was blocked in its "
            f"worker thread - the event loop was NOT actually free, meaning "
            f"the probe call was NOT offloaded")
    finally:
        release.set()
        result = await switch_task
        hb_task.cancel()

    assert result.get("status") == "loaded"
    assert hs._engines["model-a"].loaded


def test_eviction_waits_for_vram_release(monkeypatch):
    """An eviction under measurable VRAM pressure waits for the freed VRAM to
    land (wait_for_vram_release) before re-checking / loading."""
    _install_fakes(monkeypatch, free=None)
    # MEASURABLE, dynamic: total 6GB, ~4.8GB used per loaded model. One model
    # fits; a second needs the first evicted first. After the unload lands, free
    # returns to 6GB and the loop can proceed.
    total = 6 * 1024 ** 3
    per_model = int(4.8 * 1024 ** 3)

    def dyn_vram():
        used = sum(per_model for e in hs._engines.values() if e.loaded)
        return {"free": max(0, total - used), "total": total}

    monkeypatch.setattr("localm.discover.vram_info", probe_double(dyn_vram))

    calls = {"n": 0}
    import localm.vram as vram

    def fake_wait(free_fn, before_bytes=None, **kw):
        calls["n"] += 1
        return 0, free_fn()

    monkeypatch.setattr(vram, "wait_for_vram_release", fake_wait)

    app = hs.create_app(None)
    client = TestClient(app)
    assert _chat(client, "model-a").status_code == 200, "first model should fit"
    # model-a idle now; loading model-b must evict a AND wait for the free.
    assert _chat(client, "model-b").status_code == 200
    assert calls["n"] >= 1, "eviction must wait_for_vram_release before re-checking"
    loaded = sorted(n for n, e in hs._engines.items() if e.loaded)
    assert loaded == ["model-b"]
