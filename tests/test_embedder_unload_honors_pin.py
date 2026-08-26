# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared embedder's release paths must honor the in-flight-request pin,
exactly like the chat-engine unload paths do. unload_all_models,
unload_one_model (via _unload_embedder_if_matches), _do_shutdown and _do_restart
all call embedder.reset_embedder(), and with no pin check any of the four could
free the embedder - and the isolated worker process a request is mid-embed() on
- out from under it, turning a losing race into a plain 500 instead of the skip
a pinned chat Engine gets.

IsolatedEmbedder.active_requests (mirroring Engine.active_requests) plus the
embedder.active_requests() accessor close that gap; these tests pin that the
four call sites skip a pinned embedder instead of releasing it.
"""

import asyncio
import os
import threading

import pytest

from localm.inference import embedder as emb
from localm.inference import http_server as hs


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
    yield


# --------------------------------------------------------------------------- #
#  IsolatedEmbedder.active_requests / embedder.active_requests()              #
# --------------------------------------------------------------------------- #

def test_active_requests_zero_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr(emb, "_EMBEDDER", None)
    assert emb.active_requests() == 0


def test_active_requests_reflects_loaded_embedder(monkeypatch):
    class _Fake:
        active_requests = 3
    monkeypatch.setattr(emb, "_EMBEDDER", _Fake())
    assert emb.active_requests() == 3


def test_embed_pins_and_unpins_around_the_call():
    """IsolatedEmbedder.embed() must increment active_requests before the RPC
    and decrement it afterward, so a concurrent active_requests() check sees
    the pin exactly while the call is in flight, never before or after."""
    e = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    e.active_requests = 0
    e._rpc_lock = threading.RLock()   # normally set in __init__; see embed()
    seen = []

    class _FakeRunner:
        def is_alive(self):
            return True

        def embed(self, texts):
            seen.append(e.active_requests)   # pinned DURING the call
            return [[0.1] * 4 for _ in texts]

    e._runner = _FakeRunner()
    out = e.embed(["hi"])
    assert out == [[0.1] * 4]
    assert seen == [1]
    assert e.active_requests == 0


def test_embed_unpins_even_on_failure():
    """The pin must release on a losing or crashed embed() too, so no stuck pin
    wedges every future unload or shutdown. Records the value DURING the failing
    call, not just after: active_requests is 0 both before and after regardless
    of whether embed() pins at all, so only a mid-call value of 1 proves the pin
    was taken."""
    e = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    e.active_requests = 0
    e._rpc_lock = threading.RLock()   # normally set in __init__; see embed()
    seen = []

    class _FakeRunner:
        def is_alive(self):
            return True

        def embed(self, texts):
            seen.append(e.active_requests)
            raise RuntimeError("worker crashed")

    e._runner = _FakeRunner()
    with pytest.raises(RuntimeError):
        e.embed(["hi"])
    assert seen == [1], "must be pinned WHILE the failing call runs"
    assert e.active_requests == 0


# --------------------------------------------------------------------------- #
#  unload_all_models / unload_one_model                                       #
# --------------------------------------------------------------------------- #

def test_unload_all_models_skips_pinned_embedder(isolated, monkeypatch):
    """reset_embedder(force=False) makes the busy/idle decision atomically under
    the embedder lock; a separate, unlocked active_requests() check before an
    unconditional reset_embedder() call would be a TOCTOU window. The fake here
    simulates a pinned embedder by returning False, exactly like the real
    function does when active_requests() > 0 under the lock."""
    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    calls = []

    def _fake_reset(force=True):
        calls.append(force)
        return False

    monkeypatch.setattr(emb, "reset_embedder", _fake_reset)

    res = asyncio.run(hs.unload_all_models())
    assert calls == [False], (
        "unload_all_models must call reset_embedder(force=False), not force=True")
    assert res.get("embedder_unloaded") is False
    assert "embedding model" in res.get("skipped_in_use", [])
    assert res.get("status") == "in_use"


def test_unload_all_models_releases_idle_embedder(isolated, monkeypatch):
    """Negative control: an idle (unpinned) embedder is still released exactly
    as before - the pin check must not regress the existing happy path."""
    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    calls = []

    def _fake_reset(force=True):
        calls.append(force)
        return True

    monkeypatch.setattr(emb, "reset_embedder", _fake_reset)

    res = asyncio.run(hs.unload_all_models())
    assert calls == [False]
    assert res.get("embedder_unloaded") is True
    assert res.get("status") == "unloaded"


def test_unload_one_model_skips_pinned_matching_embedder(isolated, monkeypatch):
    """The ATOMIC layer catches a pin that arrives AFTER the cheap
    active_requests() precheck (below) already passed - the narrow race
    window that precheck alone cannot close, which is exactly why
    reset_embedder(force=False) still makes the authoritative decision under
    embedder._LOCK rather than the precheck being trusted on its own."""
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb-model": {"path": "Z:/models/emb.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/emb.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 0)  # precheck: idle
    calls = []

    def _fake_reset(force=True):
        calls.append(force)
        return False

    monkeypatch.setattr(emb, "reset_embedder", _fake_reset)

    res = asyncio.run(hs.unload_one_model("emb-model"))
    assert calls == [False], (
        "unload_one_model must call reset_embedder(force=False), not force=True")
    assert res.get("status") == "in_use"


def test_unload_one_model_precheck_skips_pinned_embedder_without_probing_vram(
        isolated, monkeypatch):
    """_unload_embedder_if_matches must decide busy/idle via the cheap
    active_requests() precheck BEFORE paying for _vram_free_reading()'s hardware
    probe, not after. Folding the busy check entirely into
    reset_embedder(force=False), which runs AFTER the probe, would make a busy
    embedder pay for that probe on every targeted unload. The probe is NOT
    executor-offloaded and can block the whole single-threaded event loop for up
    to discover._GPU_PROBE_DEADLINE seconds."""
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb-model": {"path": "Z:/models/emb.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/emb.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 1)  # precheck: busy

    probe_calls = []
    monkeypatch.setattr("localm.vram._vram_free_reading",
                        lambda: (probe_calls.append(1), (None, True, None))[1])
    reset_calls = []
    monkeypatch.setattr(emb, "reset_embedder",
                        lambda force=True: (reset_calls.append(force), False)[1])

    res = asyncio.run(hs.unload_one_model("emb-model"))
    assert res.get("status") == "in_use"
    assert probe_calls == [], (
        "a busy embedder must be rejected by the cheap precheck BEFORE the "
        "VRAM probe runs, not after - probe_calls proves the probe was "
        f"skipped entirely: {probe_calls}")
    assert reset_calls == [], (
        "reset_embedder must not even be attempted once the precheck already "
        f"knows the embedder is busy: {reset_calls}")


def test_unload_one_model_releases_idle_matching_embedder(isolated, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb-model": {"path": "Z:/models/emb.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "Z:/models/emb.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 0)  # precheck: idle
    calls = []

    def _fake_reset(force=True):
        calls.append(force)
        return True

    monkeypatch.setattr(emb, "reset_embedder", _fake_reset)

    res = asyncio.run(hs.unload_one_model("emb-model"))
    assert calls == [False]
    assert res.get("status") == "unloaded"


# --------------------------------------------------------------------------- #
#  _do_shutdown / _do_restart                                                 #
# --------------------------------------------------------------------------- #

def _spy_exit_release(monkeypatch):
    """Record which release path _do_shutdown/_do_restart take."""
    reset_calls, released = [], []
    monkeypatch.setattr(emb, "reset_embedder", lambda: reset_calls.append(1))
    monkeypatch.setattr(emb, "release_for_exit",
                        lambda: (released.append(1), True)[1])
    monkeypatch.setattr(hs, "_engine", None)
    return reset_calls, released


def test_do_shutdown_releases_the_embedder_without_the_load_lock(monkeypatch):
    """The exit paths do NOT go through reset_embedder(): it takes the
    embedder's load lock, which get_embedder() holds for a whole
    embedding-model load, so a stop issued mid-load would block there and never
    reach the worker teardown. They call the lock-free release_for_exit()
    instead, which makes the busy/idle decision itself: busy terminates at once,
    idle closes politely, neither blocks on _LOCK, and the worker really does
    die."""
    reset_calls, released = _spy_exit_release(monkeypatch)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    try:
        hs._do_shutdown()
    except SystemExit:
        pass
    assert released == [1], "shutdown must release the embedder worker"
    assert reset_calls == [], (
        "the exit path must not take the lock-holding reset_embedder() route")


def test_do_restart_releases_the_embedder_without_the_load_lock(monkeypatch):
    """Same contract as shutdown above: os.execv replaces this process image but
    not the worker child, so the release must run - and must not be able to hang
    on the embedder load lock."""
    reset_calls, released = _spy_exit_release(monkeypatch)
    monkeypatch.setattr(os, "execv", lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))

    try:
        hs._do_restart()
    except SystemExit:
        pass
    assert released == [1], "restart must release the embedder worker"
    assert reset_calls == [], (
        "the exit path must not take the lock-holding reset_embedder() route")
