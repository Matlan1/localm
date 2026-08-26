# SPDX-License-Identifier: AGPL-3.0-or-later
"""The jobs runner must never free the live server's SHARED engine unguarded.

Two properties of ``localm/plugins/builtin/jobs/runner.py``:

(b) OWNERSHIP: when a chat/memory job runs with ``engine=None`` and ``_load_engine``
    REUSES the live server's shared engine (``http_server._engine``) - the job model
    matches the loaded one, or is unspecified - ``run_job``'s finally must NOT unload
    it, or the host's live chat model is freed out from under the running server. A
    genuinely fresh, runner-loaded engine IS still freed (negative control).

(a) PIN-DURING-UNLOAD: the VRAM gate must not raw-``live.unload()`` the shared engine
    on the worker thread. It routes the unload through the guarded
    ``http_server.unload_one_model`` ON the server event loop, which HONORS the
    in-flight-request pin (``active_requests`` > 0 -> ``in_use``, no unload) and
    serializes with ``get_engine``, since the loop is the mutex.

The eviction tests drive ``_evict_shared_engine_for_media`` exactly as production does
- from an EXECUTOR thread while ``unload_one_model`` runs on the loop - so they
exercise the real cross-thread path rather than a mock of it.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localm.plugins.builtin.jobs import runner
import localm.inference.http_server as hs


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated LOCALM_HOME so Job construction and any config read stay off the
    user's real data."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def hsclean():
    """Clear the http_server engine-registry globals this test mutates, and restore
    them to empty/None afterwards so no state leaks to another test."""
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


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="t", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


class _FakeEngine:
    """Minimal stand-in mirroring the Engine surface the runner + unload_one_model
    touch: ``display_name``, integer ``active_requests``, a ``loaded`` flag, and an
    ``unload()`` that records (and optionally blocks, to inspect mid-free state)."""

    def __init__(self, name, *, active_requests=0, block=False):
        self.display_name = name
        self.active_requests = active_requests
        self._loaded = True
        self.unloaded = 0
        self._block = block
        self.started = threading.Event()   # set when unload() begins
        self.proceed = threading.Event()   # unload() waits for this when blocking

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self.started.set()
        if self._block:
            # Block mid-free so the caller can inspect the guarded state.
            assert self.proceed.wait(timeout=5), "proceed was never signalled"
        self._loaded = False
        self.unloaded += 1

    # webtool.run_chat_with_web is monkeypatched, so chat_stream is never reached;
    # provide it anyway so an accidental real call fails loudly instead of silently.
    def chat_stream(self, *a, **k):
        raise AssertionError("chat_stream should not be called in these tests")


# --------------------------------------------------------------------------- #
#  (b) Ownership: a reused shared engine is never unloaded by run_job          #
# --------------------------------------------------------------------------- #

def test_reused_live_engine_is_not_unloaded_by_run_job(home, hsclean, monkeypatch):
    """run_job(engine=None) whose _load_engine REUSES the shared live engine must
    NOT unload it in the finally."""
    live = _FakeEngine("gemma")
    hs._engine = live                       # the live server's shared engine, loaded
    monkeypatch.setattr(
        "localm.plugins.builtin.jobs.webtool.run_chat_with_web",
        lambda eng, prompt: "reply")

    # model=None -> _load_engine returns the reused _live (no VRAM gate, no fresh load)
    job = _make_job(model=None, prompt="hello")
    result = runner.run_job(job, engine=None)

    assert result["status"] == "ok"
    assert result["output"] == "reply"
    assert live.unloaded == 0, (
        "run_job must NOT unload the host's shared engine it merely reused")
    assert live.loaded is True


def test_fresh_runner_loaded_engine_is_unloaded(home, hsclean, monkeypatch):
    """Negative control: a genuinely fresh engine the runner loaded itself IS freed
    by the finally, so the ownership guard does not over-correct into a VRAM leak."""
    fresh = _FakeEngine("fresh")
    hs._engine = None                       # fresh is NOT the shared engine
    # _load_engine returns (engine, reused); a fresh runner-loaded engine is reused=False.
    monkeypatch.setattr(runner, "_load_engine", lambda model: (fresh, False))
    monkeypatch.setattr(
        "localm.plugins.builtin.jobs.webtool.run_chat_with_web",
        lambda eng, prompt: "reply")

    result = runner.run_job(_make_job(model="fresh", prompt="hi"), engine=None)

    assert result["status"] == "ok"
    assert fresh.unloaded == 1, "the runner's own fresh engine must be freed after the run"


def test_passed_in_engine_never_unloaded(home, hsclean, monkeypatch):
    """A live-server-passed engine (the engine= argument) is never owned or
    unloaded."""
    passed = _FakeEngine("passed")
    monkeypatch.setattr(
        "localm.plugins.builtin.jobs.webtool.run_chat_with_web",
        lambda eng, prompt: "reply")
    result = runner.run_job(_make_job(prompt="x"), engine=passed)
    assert result["status"] == "ok"
    assert passed.unloaded == 0


# --------------------------------------------------------------------------- #
#  (a) VRAM gate routes the unload through the guarded, pin-honoring path      #
# --------------------------------------------------------------------------- #

def _drive_evict(live, monkeypatch, *, wait_started=False):
    """Run _evict_shared_engine_for_media the way production does: on an EXECUTOR
    thread (it blocks on fut.result) while unload_one_model runs on the server loop.
    Returns (status, drive-time inspection hook)."""
    # unload_one_model reads free VRAM via discover.vram_capacity(); {"free": None}
    # makes before=None so it skips wait_for_vram_release entirely (no real GPU probe).
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": None})
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    # Bound the worker-thread block so a routing regression fails fast.
    monkeypatch.setattr(runner, "_EVICT_TIMEOUT_S", 15.0)

    async def _drive():
        hs._server_loop = asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, runner._evict_shared_engine_for_media, live)

    return asyncio.run(_drive())


def test_vram_gate_honors_pin_does_not_free_busy_engine(hsclean, monkeypatch):
    """A chat is generating on the shared engine (active_requests > 0), so the
    gate does NOT free it."""
    live = _FakeEngine("gemma", active_requests=1)
    hs._engines["gemma"] = live
    hs._engines_lru.append("gemma")
    hs._inference_sems["gemma"] = asyncio.Semaphore(1)
    hs._active_model_name = "gemma"
    hs._engine = live

    status = _drive_evict(live, monkeypatch)

    assert status == "in_use"
    assert live.unloaded == 0, "a pinned (in-use) shared engine must never be evicted"
    assert live.loaded is True
    assert hs._active_model_name == "gemma", "pointers untouched when nothing was freed"


def test_vram_gate_frees_idle_engine_through_guarded_path(hsclean, monkeypatch):
    """An idle shared engine IS freed, and via unload_one_model, which clears the
    active pointer and the LRU - a raw live.unload() would leave both
    untouched."""
    live = _FakeEngine("gemma", active_requests=0)
    hs._engines["gemma"] = live
    hs._engines_lru.append("gemma")
    hs._inference_sems["gemma"] = asyncio.Semaphore(1)
    hs._active_model_name = "gemma"
    hs._engine = live

    status = _drive_evict(live, monkeypatch)

    assert status == "unloaded"
    assert live.unloaded == 1
    assert hs._active_model_name is None, "unload_one_model must clear the active pointer"
    assert "gemma" not in hs._engines_lru, "unload_one_model must drop it from the LRU"


def test_vram_gate_free_is_serialized_under_the_per_model_semaphore(hsclean, monkeypatch):
    """While the guarded free is in flight, unload_one_model HOLDS the per-model
    semaphore and the engine is not yet freed, so the free is serialized on the
    loop rather than a bare off-loop unload()."""
    live = _FakeEngine("gemma", active_requests=0, block=True)
    hs._engines["gemma"] = live
    hs._engines_lru.append("gemma")
    sem = asyncio.Semaphore(1)
    hs._inference_sems["gemma"] = sem
    hs._active_model_name = "gemma"
    hs._engine = live

    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": None})
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    monkeypatch.setattr(runner, "_EVICT_TIMEOUT_S", 15.0)

    async def _drive():
        hs._server_loop = asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, runner._evict_shared_engine_for_media, live)
        # Wait (off-loop) until unload_one_model has entered live.unload() on the loop.
        await loop.run_in_executor(None, live.started.wait, 5.0)
        # Mid-free inspection: the guarded path holds the per-model semaphore and the
        # engine has not been freed yet.
        assert sem.locked(), "unload_one_model must hold the per-model semaphore mid-free"
        assert live.unloaded == 0, "unload has not completed yet (still blocked)"
        assert live.loaded is True
        # Release the block and let the guarded free finish.
        live.proceed.set()
        return await fut

    status = asyncio.run(_drive())

    assert status == "unloaded"
    assert live.unloaded == 1
    assert not sem.locked(), "the semaphore must be released after the guarded free"
    assert hs._active_model_name is None


def test_vram_gate_degrades_safely_when_server_loop_unreachable(hsclean, monkeypatch):
    """When the server loop is unreachable (no live server), the gate does NOT
    raw-unload the shared engine: it leaves the engine resident and reports the
    degrade."""
    live = _FakeEngine("gemma", active_requests=0)
    hs._engine = live
    hs._server_loop = None                  # no reachable loop

    status = runner._evict_shared_engine_for_media(live)

    assert status == "skipped"
    assert live.unloaded == 0, "must not raw-unload the shared engine off-loop"
    assert live.loaded is True
