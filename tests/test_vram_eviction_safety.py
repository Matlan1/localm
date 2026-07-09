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

import os
from pathlib import Path

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


def _fake_stat_size(monkeypatch, path: Path, size_bytes: int):
    """Make ``path.stat().st_size`` report *size_bytes* without writing that
    many real bytes to disk. *path* must already exist (a real, tiny
    placeholder file) so ``Path.is_file()`` - which itself calls ``.stat()``
    and checks ``S_ISREG(st_mode)`` - keeps working: only ``st_size`` is
    swapped out; every other field (including ``st_mode``) comes from the
    real underlying stat of the real tiny file.

    A prior version of this test truncated a real file to the full target
    size (15-40 GB) to drive switch_engine's real ``p.stat().st_size`` code
    path. A code review caught that ``truncate()`` is NOT sparse on this
    platform (verified directly: allocated blocks matched the apparent size
    exactly) - each run wrote tens of GB for real, took minutes, and an
    interrupted run (Ctrl-C, a CI timeout, an OOM-kill) orphaned multi-GB
    files permanently since the cleanup ``finally`` block never got to run.
    Faking just the stat result proves the exact same code path
    (``p.is_file()`` True, ``file_size = p.stat().st_size``) with zero real
    disk cost and nothing to orphan."""
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
    """AUDIT-GPU-SPLIT-1: vram_info() alone is single-GPU (see discover.py), so
    the pre-load refusal gate (switch_engine) must weigh a load against
    discover.vram_capacity() - the COMBINED total/free across a configured
    multi-GPU split - not just the single main GPU. A model too big for one
    GPU alone but that fits split across 2+ configured devices must load, not
    503; a model that still does not fit even combined must still be refused
    (no over-correction to "always assume it fits").

    Uses a real (but tiny) model file with a FAKED stat().st_size (see
    _fake_stat_size) to actually drive file_size = p.stat().st_size through
    switch_engine's real code path, rather than the "unregistered path ->
    fixed 4 GB" fallback other tests in this file rely on - this proves the
    fix against the same real, size-derived vram_required arithmetic the
    maintainer's original bug report hit, not just a hardcoded default."""

    _SPLIT_GPUS = [
        {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 14 * 1024 ** 3},
        {"index": 1, "name": "B", "total": 16 * 1024 ** 3, "free": 14 * 1024 ** 3},
    ]

    def _install(self, monkeypatch, tmp_path, *, size_bytes, gpus, gpu_split_indices):
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
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
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
        """Guard: the fix must not regress to 'always assume combined capacity' -
        with NO split configured (single GPU only), the same oversized model is
        still correctly refused against that one GPU's real free VRAM."""
        self._install(monkeypatch, tmp_path, size_bytes=15 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS[:1], gpu_split_indices=None)
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert "Not enough VRAM" in r.text

    def test_exceeds_even_the_combined_split_still_refused(
            self, monkeypatch, tmp_path):
        """Guard: combined capacity is a bigger ceiling, not an unlimited one -
        a model too big even for both configured GPUs together is still 503'd."""
        # 40 GB file -> needs ~49 GB, exceeds the 28 GB combined free.
        self._install(monkeypatch, tmp_path, size_bytes=40 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert "Not enough VRAM" in r.text

    def test_stale_split_index_not_currently_detected_falls_back_to_single_gpu(
            self, monkeypatch, tmp_path):
        """A gpu_split_indices referencing a device that vanished (e.g. it was
        unplugged) must degrade to single-GPU capacity (resolve_gpu_split's own
        contract - rule 5, do-not-hide-problems), not silently keep using a
        combined number for hardware that is no longer there."""
        self._install(monkeypatch, tmp_path, size_bytes=15 * 1024 ** 3,
                      gpus=self._SPLIT_GPUS[:1],   # device 1 no longer detected
                      gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            "a split referencing a since-removed GPU must fall back to "
            "single-GPU capacity, not keep refusing/granting off a stale combined number")


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
