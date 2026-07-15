# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared embedder's release paths must honor the in-flight-request pin
(AUDIT-CRIT-1), exactly like the chat-engine unload paths already do (see
test_unload_honors_pin.py). unload_all_models, unload_one_model (via
_unload_embedder_if_matches), _do_shutdown, and _do_restart all call
embedder.reset_embedder() unconditionally - previously with no pin check, so
any of the four could free the embedder (and the isolated worker process a
request is mid-embed() on) out from under it, turning a losing race into a
plain 500 instead of being skipped like a pinned chat Engine.

IsolatedEmbedder.active_requests (mirroring Engine.active_requests) plus the
new embedder.active_requests() accessor close that gap; these tests pin that
the four call sites now skip a pinned embedder instead of releasing it.
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
    e._rpc_lock = threading.RLock()   # normally __init__'s; see embed() (REG-643)
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
    """The pin must release on a losing/crashed embed() too (rule 5: never
    leave a stuck pin that would wedge every future unload/shutdown). Records
    the value DURING the failing call, not just after - active_requests is 0
    both before and after regardless of whether embed() pins at all, so only
    observing a mid-call value of 1 actually proves the pin was taken."""
    e = emb.IsolatedEmbedder.__new__(emb.IsolatedEmbedder)
    e.active_requests = 0
    e._rpc_lock = threading.RLock()   # normally __init__'s; see embed() (REG-643)
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
    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    monkeypatch.setattr(emb, "active_requests", lambda: 1)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: calls.append(1))

    res = asyncio.run(hs.unload_all_models())
    assert calls == [], "a pinned embedder must NOT be released"
    assert res.get("embedder_unloaded") is False
    assert "embedding model" in res.get("skipped_in_use", [])
    assert res.get("status") == "in_use"


def test_unload_all_models_releases_idle_embedder(isolated, monkeypatch):
    """Negative control: an idle (unpinned) embedder is still released exactly
    as before - the pin check must not regress the existing happy path."""
    monkeypatch.setattr(emb, "loaded_dim", lambda: 384)
    monkeypatch.setattr(emb, "active_requests", lambda: 0)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: calls.append(1))

    res = asyncio.run(hs.unload_all_models())
    assert calls == [1]
    assert res.get("embedder_unloaded") is True
    assert res.get("status") == "unloaded"


def test_unload_one_model_skips_pinned_matching_embedder(isolated, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb-model": {"path": "C:/models/emb.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "C:/models/emb.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 1)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: calls.append(1))

    res = asyncio.run(hs.unload_one_model("emb-model"))
    assert calls == [], "a pinned embedder must NOT be released"
    assert res.get("status") == "in_use"


def test_unload_one_model_releases_idle_matching_embedder(isolated, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb-model": {"path": "C:/models/emb.gguf"}})
    monkeypatch.setattr(emb, "loaded_path", lambda: "C:/models/emb.gguf")
    monkeypatch.setattr(emb, "active_requests", lambda: 0)
    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: calls.append(1))

    res = asyncio.run(hs.unload_one_model("emb-model"))
    assert calls == [1]
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
    """The exit paths deliberately do NOT go through reset_embedder() (nor the
    active_requests() guard that used to gate it): both take the embedder's load
    lock, which get_embedder() holds for a whole embedding-model load, so a stop
    issued mid-load blocked there and never reached the worker teardown. They
    call the lock-free release_for_exit() instead, which makes the busy/idle
    decision itself - see tests/test_embedder_worker_reaped_on_exit.py for that
    full contract (busy -> terminate at once, idle -> polite close, never blocks
    on _LOCK, and the worker really does die)."""
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
