# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedder's pre-load chat-model eviction in a process with no HTTP server
loop: the MCP server's EngineCache registers itself as an evictor
(vram.register_embedder_evictor), evict_chat_for_embedder asks it, and
embedder.py acts on the returned status."""

from __future__ import annotations

import logging
import threading
import time

import pytest

import localm.inference.http_server as hs
import localm.vram as vram
from localm.inference import embedder as emb
from localm.plugins.mcpserver.server import EngineCache


class _FakeEngine:
    """The Engine surface EngineCache's eviction path reads."""

    def __init__(self, name, *, active_requests=0):
        self.display_name = name
        self.active_requests = active_requests
        self.unloading = False
        self.loaded = True
        self.unloaded = 0

    def unload(self):
        self.loaded = False
        self.unloaded += 1


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    emb.reset_embedder()
    monkeypatch.setattr(hs, "_server_loop", None)
    monkeypatch.setattr(vram, "_EMBEDDER_EVICTORS", [])
    monkeypatch.setattr(vram, "_NO_EVICTOR_WARNED", False)
    # before=None makes EngineCache._evict's VRAM-release wait a no-op.
    monkeypatch.setattr(vram, "_vram_free_reading", lambda: (None, False, None))
    yield
    emb.reset_embedder()


def _cfg(monkeypatch, **over):
    cfg = {"max_resident_models": None, "pinned_models": None,
           "model_swap_policy": "auto", "net_mode": "off"}
    cfg.update(over)
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)
    return cfg


def _cache_with(*engines):
    cache = EngineCache(engine_factory=lambda name: _FakeEngine(name))
    for eng in engines:
        cache._engines[eng.display_name] = eng
        cache._touch(eng.display_name)
    return cache


def _gguf(tmp_path, size=4096):
    p = tmp_path / "embed.gguf"
    p.write_bytes(b"0" * size)
    return p


# --------------------------------------------------------------------------- #
#  EngineCache as a registered owner                                          #
# --------------------------------------------------------------------------- #

def test_registered_engine_cache_unloads_idle_resident(monkeypatch, tmp_path):
    _cfg(monkeypatch)
    monkeypatch.setattr(vram, "decide_embedder_swap", lambda *a, **k: True)
    a, b = _FakeEngine("chat-a"), _FakeEngine("chat-b")
    cache = _cache_with(a, b)
    unregister = vram.register_embedder_evictor(cache.evict_for_embedder)
    try:
        status = emb._maybe_swap_for_embedder(str(_gguf(tmp_path)), 99)
    finally:
        unregister()

    assert status == "unloaded"
    assert a.unloaded == 1 and b.unloaded == 1
    assert cache.resident == []


def test_serving_resident_is_kept_and_status_is_warned(monkeypatch, tmp_path, caplog):
    _cfg(monkeypatch)
    monkeypatch.setattr(vram, "decide_embedder_swap", lambda *a, **k: True)
    busy = _FakeEngine("chat-a", active_requests=1)
    cache = _cache_with(busy)
    unregister = vram.register_embedder_evictor(cache.evict_for_embedder)
    try:
        with caplog.at_level(logging.WARNING, logger="localm"):
            status = emb._maybe_swap_for_embedder(str(_gguf(tmp_path)), 99)
    finally:
        unregister()

    assert status == "in_use"
    assert busy.unloaded == 0
    assert cache.resident == ["chat-a"]
    assert any("serving a request" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)


def test_config_pinned_resident_is_kept(monkeypatch, tmp_path):
    _cfg(monkeypatch, pinned_models=["chat-a"])
    pinned = _FakeEngine("chat-a")
    cache = _cache_with(pinned)

    assert cache.evict_for_embedder(5.0) == "pinned"
    assert pinned.unloaded == 0


def test_partly_freed_reports_unloaded(monkeypatch):
    _cfg(monkeypatch)
    idle, busy = _FakeEngine("chat-a"), _FakeEngine("chat-b", active_requests=1)
    cache = _cache_with(idle, busy)

    assert cache.evict_for_embedder(5.0) == "unloaded"
    assert idle.unloaded == 1 and busy.unloaded == 0
    assert cache.resident == ["chat-b"]


def test_empty_cache_reports_already_unloaded(monkeypatch):
    _cfg(monkeypatch)
    assert _cache_with().evict_for_embedder(5.0) == "already_unloaded"


def test_not_loaded_resident_is_left_in_place(monkeypatch):
    _cfg(monkeypatch)
    lazy = _FakeEngine("chat-a")
    lazy.loaded = False
    cache = _cache_with(lazy)

    assert cache.evict_for_embedder(5.0) == "already_unloaded"
    assert lazy.unloaded == 0
    assert cache.resident == ["chat-a"]


def test_busy_cache_lock_times_out_as_error(monkeypatch):
    _cfg(monkeypatch)
    cache = _cache_with(_FakeEngine("chat-a"))
    held, release = threading.Event(), threading.Event()

    def _hold():
        with cache._lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=_hold)
    t.start()
    try:
        assert held.wait(5)
        started = time.monotonic()
        status = cache.evict_for_embedder(0.3)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        t.join(5)

    assert status == "error"
    assert elapsed < 5.0


# --------------------------------------------------------------------------- #
#  evict_chat_for_embedder with no HTTP server loop                           #
# --------------------------------------------------------------------------- #

def test_no_owner_warns_once_and_skips(caplog):
    with caplog.at_level(logging.WARNING, logger="localm"):
        first = vram.evict_chat_for_embedder()
        second = vram.evict_chat_for_embedder()

    assert first == second == "skipped"
    warned = [r for r in caplog.records if r.levelno == logging.WARNING
              and "no chat model owner" in r.getMessage()]
    assert len(warned) == 1


def test_raising_evictor_is_error_not_crash(caplog):
    def _boom(timeout_s):
        raise RuntimeError("unload failed")

    unregister = vram.register_embedder_evictor(_boom)
    try:
        with caplog.at_level(logging.WARNING, logger="localm"):
            status = vram.evict_chat_for_embedder()
    finally:
        unregister()

    assert status == "error"
    assert any("unload failed" in r.getMessage() for r in caplog.records)


def test_non_string_status_counts_as_error():
    unregister = vram.register_embedder_evictor(lambda timeout_s: None)
    try:
        assert vram.evict_chat_for_embedder() == "error"
    finally:
        unregister()


def test_worst_status_wins_across_owners():
    unregisters = [vram.register_embedder_evictor(lambda t: "unloaded"),
                   vram.register_embedder_evictor(lambda t: "in_use")]
    try:
        assert vram.evict_chat_for_embedder() == "in_use"
    finally:
        for u in unregisters:
            u()


def test_unregister_is_idempotent():
    unregister = vram.register_embedder_evictor(lambda t: "unloaded")
    unregister()
    unregister()
    assert vram._EMBEDDER_EVICTORS == []


# --------------------------------------------------------------------------- #
#  serve_stdio registers its EngineCache for the run, then unregisters        #
# --------------------------------------------------------------------------- #

def test_serve_stdio_registers_its_engine_cache_for_the_run(monkeypatch):
    import localm.plugins.mcpserver.server as server

    seen = {}

    def _run(self, stdin=None, stdout=None):
        seen["evictors"] = list(vram._EMBEDDER_EVICTORS)

    monkeypatch.setattr(server.MCPStdioServer, "run_stdio", _run)
    monkeypatch.setattr(server, "_redirect_consoles_to_stderr", lambda: None)
    monkeypatch.setattr(server, "build_tools", lambda *a, **k: {})
    server.serve_stdio(model="chat-a")

    assert len(seen["evictors"]) == 1
    assert isinstance(getattr(seen["evictors"][0], "__self__", None), EngineCache)
    assert vram._EMBEDDER_EVICTORS == []


# --------------------------------------------------------------------------- #
#  get_embedder end to end: lock state and status consumption                 #
# --------------------------------------------------------------------------- #

class _FakeEmbedder:
    dim = 3

    def __init__(self, path, **kw):
        self.model_path = path

    def close(self):
        pass


def _drive_get_embedder(monkeypatch, tmp_path, evictor):
    p = _gguf(tmp_path)
    _cfg(monkeypatch, embedding_model=str(p), n_gpu_layers=99)
    monkeypatch.setattr(emb, "resolve_embedding_model_path",
                        lambda *, allow_download=None, on_progress=None: str(p))
    monkeypatch.setattr(vram, "decide_embedder_swap", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_choose_embedder_gpu_layers",
                        lambda path, cfg, **k: (99, None))
    monkeypatch.setattr(emb, "IsolatedEmbedder", _FakeEmbedder)
    stages: list = []
    unregister = vram.register_embedder_evictor(evictor)
    try:
        result = emb.get_embedder(on_progress=stages.append)
    finally:
        unregister()
    return result, stages


def test_degraded_status_reaches_get_embedder_progress(monkeypatch, tmp_path):
    busy = _FakeEngine("chat-a", active_requests=1)
    cache = _cache_with(busy)

    result, stages = _drive_get_embedder(monkeypatch, tmp_path,
                                         cache.evict_for_embedder)

    assert result is not None
    assert any(s.startswith("Could not free VRAM") and "serving a request" in s
               for s in stages), stages


def test_freed_status_adds_no_degraded_stage(monkeypatch, tmp_path):
    idle = _FakeEngine("chat-a")
    cache = _cache_with(idle)

    result, stages = _drive_get_embedder(monkeypatch, tmp_path,
                                         cache.evict_for_embedder)

    assert result is not None
    assert idle.unloaded == 1
    assert not any(s.startswith("Could not free VRAM") for s in stages), stages


def test_evictor_runs_holding_neither_embedder_lock_nor_load_lock(monkeypatch, tmp_path):
    from localm.inference.engine import _LOAD_LOCK

    cache = _cache_with(_FakeEngine("chat-a"))
    probe = {}

    def _probing_evictor(timeout_s):
        def _try():
            got_load = _LOAD_LOCK.acquire(timeout=2.0)
            if got_load:
                _LOAD_LOCK.release()
            got_emb = emb._LOCK.acquire(timeout=2.0)
            if got_emb:
                emb._LOCK.release()
            probe["load_lock_free"] = got_load
            probe["embedder_lock_free"] = got_emb

        t = threading.Thread(target=_try)
        t.start()
        t.join(10)
        return cache.evict_for_embedder(timeout_s)

    result, _stages = _drive_get_embedder(monkeypatch, tmp_path, _probing_evictor)

    assert result is not None
    assert probe == {"load_lock_free": True, "embedder_lock_free": True}
    assert cache.resident == []
