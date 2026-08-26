# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedder's own VRAM-aware chat-model swap (get_embedder() ->
embedder._maybe_swap_for_embedder -> vram.decide_embedder_swap /
vram.evict_chat_for_embedder).

The shared embedder (localm.inference.embedder) lives entirely outside the chat
model's VRAM lifecycle in localm/inference/http_server.py (_engines,
switch_engine). Without get_embedder() calling the swap-out mechanism the
image/music/video plugins already use before their own model load, a resident
chat model and the embedder are both asked to be resident simultaneously -
VRAM/RAM pressure, most visible with a large embedding model (e.g.
Qwen3-Embedding-8B, ~7.5 GB) alongside a large chat model.

evict_chat_for_embedder's cross-thread eviction path mirrors jobs/runner.py's
_evict_shared_engine_for_media: it must route the unload through the guarded
http_server.unload_all_models() ON the server's event loop, not a raw off-loop
engine.unload(), so it honors the in-flight-request pin and serializes with
get_engine.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localm.inference import embedder as emb
import localm.inference.http_server as hs
from localm.vram import evict_chat_for_embedder

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _reset_embedder():
    emb.reset_embedder()
    yield
    emb.reset_embedder()


@pytest.fixture
def hsclean():
    """Clear the http_server engine-registry globals this test mutates, and
    restore them to empty/None afterwards so no state leaks to another test."""
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._server_loop = None
    yield
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._server_loop = None


class _FakeEngine:
    """Minimal stand-in mirroring the Engine surface unload_all_models touches."""

    def __init__(self, name, *, active_requests=0):
        self.display_name = name
        self.active_requests = active_requests
        self._loaded = True
        self.unloaded = 0
        self.unloading = False

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self._loaded = False
        self.unloaded += 1


# --------------------------------------------------------------------------- #
#  vram.evict_chat_for_embedder: the cross-thread guarded eviction            #
# --------------------------------------------------------------------------- #

def _drive_evict(monkeypatch, *, free=None):
    """Run evict_chat_for_embedder the way get_embedder() does: on an EXECUTOR
    thread (it blocks on fut.result) while unload_all_models runs on the server
    loop - the real cross-thread path, not a mock of it."""
    monkeypatch.setattr("localm.discover.vram_capacity", lambda: {"free": free})
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)

    async def _drive():
        hs._server_loop = asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, evict_chat_for_embedder)

    return asyncio.run(_drive())


def test_evict_frees_idle_chat_engine(hsclean, monkeypatch):
    live = _FakeEngine("gemma")
    hs._engines["gemma"] = live
    hs._engines_lru.append("gemma")
    hs._inference_sems["gemma"] = asyncio.Semaphore(1)
    hs._active_model_name = "gemma"
    hs._engine = live

    status = _drive_evict(monkeypatch)

    assert status == "unloaded"
    assert live.unloaded == 1
    assert live.loaded is False
    assert hs._active_model_name is None, "unload_all_models must clear the active pointer"


def test_evict_honors_the_in_flight_pin(hsclean, monkeypatch):
    """A chat is generating on the shared engine (active_requests > 0). The gate
    must not free it - discriminating vs a raw off-loop unload() that would."""
    live = _FakeEngine("gemma", active_requests=1)
    hs._engines["gemma"] = live
    hs._engines_lru.append("gemma")
    hs._inference_sems["gemma"] = asyncio.Semaphore(1)
    hs._active_model_name = "gemma"
    hs._engine = live

    status = _drive_evict(monkeypatch)

    assert status == "in_use"
    assert live.unloaded == 0, "a pinned (in-use) shared engine must never be evicted"
    assert live.loaded is True


def test_evict_frees_multiple_idle_engines(hsclean, monkeypatch):
    a = _FakeEngine("model-a")
    b = _FakeEngine("model-b")
    hs._engines["model-a"] = a
    hs._engines["model-b"] = b
    hs._engines_lru += ["model-a", "model-b"]
    hs._inference_sems["model-a"] = asyncio.Semaphore(1)
    hs._inference_sems["model-b"] = asyncio.Semaphore(1)
    hs._active_model_name = "model-b"

    status = _drive_evict(monkeypatch)

    assert status == "unloaded"
    assert a.unloaded == 1 and b.unloaded == 1


def test_evict_noop_when_nothing_loaded(hsclean, monkeypatch):
    status = _drive_evict(monkeypatch)
    assert status == "already_unloaded"


def test_evict_degrades_safely_when_server_loop_unreachable(hsclean, monkeypatch):
    """No live server (e.g. a bare CLI embed): must NOT raw-unload off-loop
    (that reintroduces the pin race) - just report the degrade."""
    live = _FakeEngine("gemma")
    hs._engine = live
    hs._server_loop = None

    status = evict_chat_for_embedder()

    assert status == "skipped"
    assert live.unloaded == 0
    assert live.loaded is True


def test_evict_reports_error_on_timeout(hsclean, monkeypatch):
    """A wedged/slow server loop must not hang the caller forever or raise -
    the degrade is reported, never a crash."""
    class _StuckLoop:
        def is_running(self):
            return True

    hs._server_loop = _StuckLoop()

    def _boom(coro, *a, **k):
        coro.close()   # avoid a "coroutine was never awaited" warning from the probe
        raise RuntimeError("run_coroutine_threadsafe needs a real loop")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _boom)
    assert evict_chat_for_embedder(timeout_s=1.0) == "error"


# --------------------------------------------------------------------------- #
#  embedder._maybe_swap_for_embedder: the pre-load decision + trigger         #
# --------------------------------------------------------------------------- #

def _touch(path, size=1024):
    path.write_bytes(b"0" * size)
    return path


def test_maybe_swap_skips_cpu_only_load(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("localm.vram.decide_embedder_swap",
                        lambda *a, **k: calls.append("decide"))
    p = _touch(tmp_path / "e.gguf")
    emb._maybe_swap_for_embedder(str(p), 0)
    assert calls == [], "n_gpu_layers<=0 never contends for VRAM - must skip the check entirely"


def test_maybe_swap_skips_when_file_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("localm.vram.decide_embedder_swap",
                        lambda *a, **k: calls.append("decide"))
    emb._maybe_swap_for_embedder(str(tmp_path / "nope.gguf"), 99)
    assert calls == []


def test_maybe_swap_evicts_when_decided(tmp_path, monkeypatch):
    p = _touch(tmp_path / "e.gguf", size=10 * 1024 * 1024)
    monkeypatch.setattr("localm.config.load_config", lambda: {"model_swap_policy": "auto"})
    decide_calls = []
    monkeypatch.setattr(
        "localm.vram.decide_embedder_swap",
        lambda estimate, *, policy: decide_calls.append((estimate, policy)) or True)
    evict_calls = []
    monkeypatch.setattr("localm.vram.evict_chat_for_embedder", lambda: evict_calls.append(1))

    emb._maybe_swap_for_embedder(str(p), 99)

    assert decide_calls and decide_calls[0][1] == "auto"
    assert decide_calls[0][0] == int(p.stat().st_size * 1.2)
    assert evict_calls == [1]


def test_maybe_swap_skips_eviction_when_not_decided(tmp_path, monkeypatch):
    p = _touch(tmp_path / "e.gguf")
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr("localm.vram.decide_embedder_swap", lambda *a, **k: False)
    evict_calls = []
    monkeypatch.setattr("localm.vram.evict_chat_for_embedder", lambda: evict_calls.append(1))

    emb._maybe_swap_for_embedder(str(p), 99)

    assert evict_calls == [], "must not evict the chat model when it already fits"


def test_maybe_swap_honors_never_policy(tmp_path, monkeypatch):
    """model_swap_policy=never must reach decide_embedder_swap with policy='never'
    (resolve_swap_policy is reused, not reimplemented) so a user's explicit choice
    to keep chat hot is honored, not silently overridden."""
    p = _touch(tmp_path / "e.gguf")
    monkeypatch.setattr("localm.config.load_config", lambda: {"model_swap_policy": "never"})
    seen = {}
    monkeypatch.setattr(
        "localm.vram.decide_embedder_swap",
        lambda estimate, *, policy: seen.setdefault("policy", policy) or False)
    emb._maybe_swap_for_embedder(str(p), 99)
    assert seen["policy"] == "never"


# --------------------------------------------------------------------------- #
#  get_embedder() actually invokes the swap check before loading             #
# --------------------------------------------------------------------------- #

def test_get_embedder_invokes_swap_check_before_native_load(monkeypatch, tmp_path):
    p = _touch(tmp_path / "e.gguf")
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": str(p), "n_gpu_layers": 42,
                                 "net_mode": "off"})
    order = []
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder",
                        lambda path, ngl: order.append(("swap", path, ngl)))

    class _FakeEmbedder:
        dim = 3

        def __init__(self, path, **kw):
            order.append(("load", path, kw.get("n_gpu_layers")))
            self.model_path = path

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)

    e = emb.get_embedder()

    assert e is not None
    assert order == [("swap", str(p), 42), ("load", str(p), 42)], (
        "the swap decision must run BEFORE the native load, using the same "
        "resolved path/n_gpu_layers")


def test_get_embedder_swap_check_does_not_deadlock_cross_thread_lock_use(monkeypatch, tmp_path):
    """_maybe_swap_for_embedder must run OUTSIDE embedder._LOCK, not inside the
    `with _LOCK:` block get_embedder() holds for its state checks.

    The real eviction path (evict_chat_for_embedder) submits
    http_server.unload_all_models() onto the SERVER'S EVENT LOOP THREAD via
    run_coroutine_threadsafe and blocks the calling thread waiting for it; that
    coroutine calls embedder.loaded_dim(), which itself needs embedder._LOCK.
    embedder._LOCK is a plain threading.RLock (thread-owned, not reentrant
    ACROSS threads) - so if get_embedder() still held it while blocking on that
    coroutine, this is a genuine cross-thread deadlock: the calling thread
    waits on the coroutine, the coroutine's thread waits on the calling
    thread's lock. It resolves only via evict_chat_for_embedder's eviction
    timeout (300s in production) - a multi-minute hang of /v1/embeddings under
    tight VRAM rather than a permanent one, but a severe DoS.

    This simulates the cross-thread call the real coroutine makes (a second
    thread calling embedder.loaded_dim(), which needs _LOCK) inside a fake
    evict_chat_for_embedder, and asserts it completes promptly - proving
    get_embedder() is not holding _LOCK while the swap/eviction step runs."""
    p = _touch(tmp_path / "e.gguf")
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": str(p), "n_gpu_layers": 99,
                                 "net_mode": "off"})
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None: str(p))
    monkeypatch.setattr("localm.vram.decide_embedder_swap", lambda *a, **k: True)

    cross_thread_completed = threading.Event()

    def _fake_evict_chat_for_embedder(**kw):
        # Mirrors the real function's shape: block THIS thread while ANOTHER
        # thread (standing in for the server's event loop) needs
        # embedder._LOCK, exactly as unload_all_models() -> loaded_dim() does.
        def _cross_thread_call():
            emb.loaded_dim()
            cross_thread_completed.set()

        t = threading.Thread(target=_cross_thread_call)
        t.start()
        t.join(timeout=5.0)
        return "unloaded"

    monkeypatch.setattr("localm.vram.evict_chat_for_embedder", _fake_evict_chat_for_embedder)

    class _FakeEmbedder:
        dim = 3

        def __init__(self, path, **kw):
            self.model_path = path

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)

    e = emb.get_embedder()

    assert e is not None
    assert cross_thread_completed.is_set(), (
        "the cross-thread loaded_dim() call never completed within 5s - "
        "get_embedder() is still holding _LOCK while the swap/eviction path "
        "runs, reproducing the 2026-07-14 cross-thread deadlock")


# --------------------------------------------------------------------------- #
#  GPU-vs-CPU placement (_choose_embedder_gpu_layers + get_embedder wiring)    #
# --------------------------------------------------------------------------- #

class TestChooseEmbedderGpuLayers:
    """When eviction cannot or should not free room (per-turn memory recall
    pins the chat engine, so evict_chat_for_embedder returns "in_use"),
    force-loading all layers onto a card that cannot hold it thrashes the
    resident chat model. The placement chooser must degrade to CPU instead -
    without ever overriding an explicit user choice."""

    @staticmethod
    def _gguf(tmp_path, size=4096):
        f = tmp_path / "emb.gguf"
        f.write_bytes(b"\0" * size)
        return str(f)

    def test_explicit_embedding_gpu_layers_cpu_wins(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"embedding_gpu_layers": 0},
            read_free=lambda: 64 * GB)
        assert (ngl, reason) == (0, None)

    def test_explicit_embedding_gpu_layers_gpu_wins_even_when_tight(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"embedding_gpu_layers": 99},
            read_free=lambda: 1)
        assert (ngl, reason) == (99, None)

    def test_inherits_explicit_global_n_gpu_layers(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 24}, read_free=lambda: 1)
        assert (ngl, reason) == (24, None)

    def test_inherits_global_cpu_only(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 0}, read_free=lambda: 64 * GB)
        assert (ngl, reason) == (0, None)

    def test_auto_fits_stays_on_gpu(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 99},
            read_free=lambda: 64 * GB)
        assert (ngl, reason) == (99, None)

    def test_auto_does_not_fit_goes_cpu_with_reason(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        # file 4096 B -> estimate ~4915 B; only 1000 B free
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 99}, read_free=lambda: 1000)
        assert ngl == 0
        assert reason is not None and "VRAM" in reason

    def test_auto_unmeasurable_free_keeps_gpu(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 99}, read_free=lambda: None)
        assert (ngl, reason) == (99, None)

    def test_unreadable_file_keeps_gpu(self, tmp_path):
        from localm.inference.embedder import _choose_embedder_gpu_layers
        ngl, reason = _choose_embedder_gpu_layers(
            str(tmp_path / "missing.gguf"), {"n_gpu_layers": 99},
            read_free=lambda: 1000)
        assert (ngl, reason) == (99, None)

    def test_default_read_free_ignores_a_stale_probe(self, tmp_path, monkeypatch):
        """The DEFAULT read_free (used whenever a caller does not inject its
        own, i.e. every real load) must not trust a served last-known-good
        reading: a fresh GPU_PROBE_OK reading of 1000 bytes would send this
        load to CPU (see test_auto_does_not_fit_goes_cpu_with_reason above),
        but the SAME reading under GPU_PROBE_TIMEOUT/BUSY is not a current
        measurement, so it must fall back to "unmeasurable" and keep the
        full-GPU-offload attempt - exactly like an unreadable file or a None
        reading above. Calling vram_capacity() bare (no return_status) would
        use a stale reading exactly like a fresh one."""
        import localm.discover as disc
        from localm.inference.embedder import _choose_embedder_gpu_layers
        monkeypatch.setattr(
            disc, "vram_capacity",
            lambda *a, **kw: ({"free": 1000}, disc.GPU_PROBE_TIMEOUT)
            if kw.get("return_status") else {"free": 1000})
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 99})
        assert (ngl, reason) == (99, None)

    def test_default_read_free_uses_a_fresh_probe(self, tmp_path, monkeypatch):
        """The complement: a FRESH (GPU_PROBE_OK) reading through the default
        closure is used exactly as the injected-read_free tests above expect."""
        import localm.discover as disc
        from localm.inference.embedder import _choose_embedder_gpu_layers
        monkeypatch.setattr(
            disc, "vram_capacity",
            lambda *a, **kw: ({"free": 1000}, disc.GPU_PROBE_OK)
            if kw.get("return_status") else {"free": 1000})
        ngl, reason = _choose_embedder_gpu_layers(
            self._gguf(tmp_path), {"n_gpu_layers": 99})
        assert ngl == 0
        assert reason is not None and "VRAM" in reason


class _FakeEmbedder:
    """Stands in for IsolatedEmbedder in wiring tests: records construction
    kwargs, satisfies everything get_embedder touches afterwards."""
    instances: list = []

    def __init__(self, path, *, n_gpu_layers=99, pooling_type=None,
                 gpu_fallback_reason=None):
        self.path = path
        self.n_gpu_layers = n_gpu_layers
        self.pooling_type = pooling_type
        self.gpu_fallback_reason = gpu_fallback_reason
        self.dim = 3
        self.effective_pooling = None
        self.active_requests = 0
        self._runner = None
        _FakeEmbedder.instances.append(self)

    def close(self):
        pass


class TestGetEmbedderPlacementWiring:
    def test_cpu_choice_reaches_the_constructor_with_reason(
            self, monkeypatch, tmp_path):
        f = tmp_path / "emb.gguf"
        f.write_bytes(b"\0" * 4096)
        calls = []
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: str(f))
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder",
                            lambda p, n: calls.append(("swap", n)))
        monkeypatch.setattr(
            emb, "_choose_embedder_gpu_layers",
            lambda p, c, **kw: calls.append(("choose",)) or (0, "no VRAM room"))
        _FakeEmbedder.instances.clear()
        monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)

        got = emb.get_embedder()

        assert got is _FakeEmbedder.instances[-1]
        assert got.n_gpu_layers == 0
        assert got.gpu_fallback_reason == "no VRAM room"
        # The eviction swap ran BEFORE placement, so a successful eviction
        # still yields a GPU choice.
        assert calls == [("swap", 99), ("choose",)]

    def test_gpu_choice_constructs_without_reason(self, monkeypatch, tmp_path):
        f = tmp_path / "emb.gguf"
        f.write_bytes(b"\0" * 4096)
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: str(f))
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda p, n: None)
        monkeypatch.setattr(emb, "_choose_embedder_gpu_layers",
                            lambda p, c, **kw: (99, None))
        _FakeEmbedder.instances.clear()
        monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)

        got = emb.get_embedder()

        assert got.n_gpu_layers == 99
        assert got.gpu_fallback_reason is None
