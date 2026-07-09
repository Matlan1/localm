# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM eviction safety (Antigravity-audit CRIT-1 / CRIT-2 / MED-11).

These exercise the multi-model switch/eviction path in http_server.switch_engine:

  CRIT-1  A request must PIN its engine (active_requests>=1) the instant it takes
          ownership - synchronously after get_engine, before any await - so a
          concurrent model load can never evict an engine out from under an
          in-flight request. Proven by observing active_requests from a chat
          inlet hook (which runs after get_engine): 0 on the broken code, 1 once
          the pin is moved early.

  CRIT-2  When vram_info() reports no measurable "free" (the default GGUF-only
          non-NVIDIA install), model switching must fall back to single-resident
          (evict idle before load) instead of stacking models until the driver
          OOMs. Proven by loading a->b->c with unmeasurable VRAM: broken code
          leaves all three loaded, the fix leaves exactly model-c.

  MED-11  After an eviction the loop must wait for the native VRAM free to land
          before re-checking, so it does not over-evict on a stale-low reading.
"""

from fastapi.testclient import TestClient

from localm.inference import http_server as hs


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.can_be_multimodal = False
        self.model_path = f"models/{display_name}.gguf"

    @property
    def loaded(self):
        return self._loaded

    def load(self):
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


def _install_fakes(monkeypatch, *, free):
    fake_registry = {
        "model-a": {"path": "models/model-a.gguf", "source": "local"},
        "model-b": {"path": "models/model-b.gguf", "source": "local"},
        "model-c": {"path": "models/model-c.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    # `free`=None models an install where VRAM cannot be measured.
    info = {"total": 16 * 1024 ** 3}
    if free is not None:
        info["free"] = free
    monkeypatch.setattr("localm.discover.vram_info", lambda: dict(info))
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
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


def test_eviction_waits_for_vram_release(monkeypatch):
    """MED-11: an eviction under measurable VRAM pressure waits for the freed
    VRAM to land (wait_for_vram_release) before re-checking / loading."""
    _install_fakes(monkeypatch, free=None)
    # MEASURABLE, dynamic: total 6GB, ~4.8GB used per loaded model. One model
    # fits; a second needs the first evicted first. After the unload lands, free
    # returns to 6GB and the loop can proceed.
    total = 6 * 1024 ** 3
    per_model = int(4.8 * 1024 ** 3)

    def dyn_vram():
        used = sum(per_model for e in hs._engines.values() if e.loaded)
        return {"free": max(0, total - used), "total": total}

    monkeypatch.setattr("localm.discover.vram_info", dyn_vram)

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
