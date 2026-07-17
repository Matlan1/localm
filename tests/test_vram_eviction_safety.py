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

import asyncio
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


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


def _install_fakes(monkeypatch, *, free, status=None):
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


def _mb_figure_in(text):
    """True if *text* quotes a concrete free-VRAM figure ('<N> MB free'). The gate
    must quote one only for a reading it actually measured (AGENTS.md rule 5)."""
    import re
    return re.search(r"\d+\s*MB free", text) is not None


class TestInconclusiveProbeDoesNotSkipTheGate:
    """The root-cause split: `measurable` conflated 'this box CANNOT measure free
    VRAM' (permanent -> best-effort load is correct) with 'this PROBE did not
    complete' (transient -> NOT a licence to skip the VRAM check). On master a
    timed-out probe served free=None -> measurable=False -> on a fresh server
    (nothing to evict) `if not measurable: break` loaded with NO VRAM CHECK - the
    first load after every server start on any box whose cold driver init overruns
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
        """rule 5: the inconclusive 503 must not state a free-VRAM figure it never
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

    def test_measured_refusal_still_quotes_the_figure(self, monkeypatch):
        """Guard the honest refusal: when the probe COMPLETES and genuinely shows
        too little VRAM, the 503 still refuses AND still quotes the measured figure.
        Without this, a fix could regress into 'never quote / never refuse' and the
        rule-5 test above would pass vacuously."""
        from localm.discover import GPU_PROBE_OK
        # 2 GB free, probe OK: below the ~5.8 GB the load needs -> honest refusal.
        _install_fakes(monkeypatch, free=2 * 1024 ** 3, status=GPU_PROBE_OK)
        client = TestClient(hs.create_app(None))
        r = _chat(client, "model-a")
        assert r.status_code == 503
        assert _mb_figure_in(r.text), (
            "a refusal from a COMPLETED probe must quote the measured figure; "
            f"got: {r.text[:200]}")

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
        monkeypatch.setattr("localm.discover.list_gpus", probe_double(gpus))
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


class TestPerDeviceSplitFitGate:
    """AUDIT-GPU-SPLIT-2: vram_capacity()'s AGGREGATE check alone is not
    enough for a GGUF-backend load - apply_gpu_split() divides a model by a
    STATIC per-config ratio with no live per-device capacity awareness of its
    own (unlike the HF backend's device_map="auto", which self-corrects from
    live per-device free VRAM instead). An asymmetric split - e.g. another
    already-loaded model sits on one configured device more than another -
    can pass the aggregate check while one device's actual proportional
    share is short, reaching the native loader with too little room on that
    device. discover.gpu_split_shortfall() is the per-device gate that
    catches this before the native load, refusing cleanly instead of risking
    a native crash (llama.cpp can hard-abort rather than raise)."""

    def _install(self, monkeypatch, tmp_path, *, filename, gpus, gpu_split_indices,
                 gpu_split_ratios=None):
        model_file = tmp_path / filename
        # Unregistered-on-disk path (matches _install_fakes' convention at the
        # top of this file): file_size falls back to the code's own documented
        # 4 GB default, so vram_required is always int(4 GiB * 1.2) ~= 5.15
        # GiB, + the fixed 1 GiB headroom ~= 6.15 GiB aggregate threshold -
        # comfortably covered by the GPUs below, so only the PER-DEVICE gate
        # is what can block these tests.
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
        monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
        hs._engines.clear()
        hs._engines_lru.clear()
        hs._inference_sems.clear()
        hs._last_activity_per_model.clear()
        hs._active_model_name = None
        hs._default_model_name = None
        hs._engine = None
        hs._inference_sem = None

    def test_aggregate_fits_but_one_device_short_is_refused(self, monkeypatch, tmp_path):
        # GPU0: 2 GiB free (short of its ~2.58 GiB equal-split share of the
        # ~5.15 GiB required). GPU1: 30 GiB free (comfortably covers its
        # share). Combined 32 GiB free >> the ~6.15 GiB aggregate threshold -
        # the OLD aggregate-only gate would have let this through.
        gpus = [
            {"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 2 * 1024 ** 3},
            {"index": 1, "name": "B", "total": 32 * 1024 ** 3, "free": 30 * 1024 ** 3},
        ]
        self._install(monkeypatch, tmp_path, filename="model-a.gguf",
                      gpus=gpus, gpu_split_indices=[0, 1])
        app = hs.create_app(None)
        client = TestClient(app)
        r = _chat(client, "model-a")
        assert r.status_code == 503, (
            f"aggregate free (32GiB) covers the ~6.15GiB threshold, but GPU 0's "
            f"own equal-split share does not fit its 2GiB free - must still "
            f"refuse rather than reach the native loader: {r.text}")
        assert "configured split" in r.text
        assert "GPU 0" in r.text

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
        genuinely drives additional eviction, not just a one-shot refusal."""
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
            lambda: {**base_cfg, "gpu_split_indices": [0, 1]})

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
    """Regression guard for the diagnosed idle-hang root cause PR #541 fixed
    elsewhere: switch_engine's pre-load eviction loop calls
    discover.vram_capacity()/gpu_split_shortfall(), which route through the
    deadline-bounded discover.list_gpus() probe - but a bounded (up to ~4s)
    block is still a block if it runs directly on the single event loop,
    freezing every OTHER concurrent coroutine for that long. This loop can
    re-probe multiple times per eviction, so it is exposed to exactly the
    same hang class the other call sites were fixed for.

    Proven with an independent heartbeat coroutine ticking on a fixed
    interval, NOT by racing against switch_engine's own request-completion
    timing (which depends on how many other awaits happen to run first and is
    not a reliable signal - confirmed by hand: an httpx/ASGI-based version of
    this test kept passing even with the fix reverted, because enough
    incidental await points elsewhere let it interleave anyway). If the event
    loop is genuinely blocked, the heartbeat CANNOT tick during that window -
    a deterministic, unambiguous signal regardless of switch_engine's
    internal scheduling."""
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

        # This is a precondition wait (has the executor thread actually been
        # scheduled by the OS yet), not itself the property under test - the
        # LATER ticks_before/ticks_after assertions are what actually prove
        # the event loop stayed responsive. Found live (not assumed): under
        # genuine full-suite-scale parallel load (~5000 tests, many
        # concurrent xdist worker PROCESSES all competing for real CPU
        # cores), the OS can take longer than 2s just to give this test's
        # OWN worker thread its first timeslice - confirmed by re-running
        # standalone (passes 1/1) and this whole file alone under -n auto
        # (passes 3/3); it only ever failed at full-suite scale. A generous
        # budget here tolerates that real scheduling jitter without
        # weakening what the test actually proves.
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
