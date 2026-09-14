# SPDX-License-Identifier: AGPL-3.0-or-later
"""_gpu_registry_sync() must never run on the server event loop.

It does blocking work on every model load/unload: a registry temp-file
write + os.replace, a _model_file_size() stat/rglob walk, and - when a non-zero
main_gpu_index is configured - _current_gpu_index() -> resolve_main_gpu_index()
-> discover.list_gpus(), a real torch/nvidia-smi hardware probe bounded to a 4s
deadline. On the single event loop that stalls EVERY concurrent request and
stream for up to that deadline.

These tests assert the property directly - the sync work runs on a thread OTHER
than the event-loop thread - rather than trying to measure a stall.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading

import pytest

from localm.inference import http_server as hs
from localm.inference.http_server import create_app


class _ThreadProbe:
    """Stands in for _gpu_registry_sync, recording which thread it ran on."""

    def __init__(self):
        self.threads = []

    def __call__(self):
        self.threads.append(threading.get_ident())


@pytest.fixture
def probe(monkeypatch):
    p = _ThreadProbe()
    monkeypatch.setattr(hs, "_gpu_registry_sync", p)
    # No real hardware probe / VRAM wait in a unit test.
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": 32 * 1024 ** 3,
                                             "total": 32 * 1024 ** 3})
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return p


class FakeEngine:
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

    def set_load_cancel(self, cancel):
        pass


def _install_loaded(name):
    eng = FakeEngine(name)
    eng.load()
    hs._engines[name] = eng
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    hs._active_model_name = name
    hs._engine = eng
    hs._inference_sem = hs._inference_sems[name]
    return eng


def _assert_off_loop(probe, loop_thread, where):
    assert probe.threads, f"_gpu_registry_sync never ran in {where}"
    assert all(t != loop_thread for t in probe.threads), (
        f"{where} ran the gpu-registry sync (registry file I/O + a GPU driver "
        f"probe) ON the event loop thread, stalling every concurrent request")


def test_switch_engine_syncs_registry_off_the_loop(probe):
    """Every successful load hits this path."""

    async def scenario():
        engines = {"A": FakeEngine("A")}
        await hs.switch_engine("A", engines.__getitem__)
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "switch_engine")


def test_unload_all_models_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_all_models()
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_all_models")


def test_unload_one_model_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_one_model("A")
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_one_model")


def test_unload_embedder_if_matches_syncs_registry_off_the_loop(probe, monkeypatch, tmp_path):
    """The targeted-unload counterpart: loaded_path(), the active_requests()
    precheck, reset_embedder(force=False) and the VRAM wait are already
    offloaded, and the registry sync must be too."""
    model = tmp_path / "emb.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: str(model))
    monkeypatch.setattr("localm.inference.embedder.active_requests", lambda: 0)
    monkeypatch.setattr("localm.inference.embedder.reset_embedder",
                        lambda force=True: True)
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb": {"path": str(model), "source": "local"}})
    monkeypatch.setattr("localm.model_manager._entry_path", lambda entry: str(model))

    async def scenario():
        loop = asyncio.get_running_loop()
        res = await hs._unload_embedder_if_matches("emb", loop)
        assert res is not None and res["status"] == "unloaded", res
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "_unload_embedder_if_matches")


def test_idle_unload_once_syncs_registry_off_the_loop(probe):
    """The 5th real call site, found only by building the AST sentinel below
    - this test file's own docstring never mentioned _idle_unload_once at
    all until this test was added. It was already correctly offloaded (not
    a live bug), but had zero coverage here, the same shape as the startup
    call site that was NOT already correct."""
    import time
    async def scenario():
        _install_loaded("A")
        hs._last_activity_per_model["A"] = time.monotonic() - 1000
        unloaded = await hs._idle_unload_once(ttl=1)
        assert unloaded is True, "the idle check never actually unloaded anything"
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "_idle_unload_once")


def test_every_gpu_registry_sync_reference_has_a_dedicated_test():
    """Sentinel: enumerate every function in http_server.py that references
    _gpu_registry_sync at all (a direct call, or passed by name to
    run_in_executor - which is how every real call site here actually
    invokes it), and require each one to be named in THIS file.

    This is the mechanical guardrail for the exact class of bug this file
    was written to catch (item 6, 2026-09-14 regression triage): the
    off-loop invariant was correctly enforced for four call sites, then a
    FIFTH (lifespan's one-shot startup call) was added later with no test
    added here, and nothing caught the gap until a live server froze on
    startup. A lesson recorded only in a docstring or a dev-notes file
    goes stale the moment a new caller is added and nobody remembers to
    come back here - this makes staleness a test failure instead of a
    silent gap: add a new caller, this test breaks until you also cover it.

    Matched by NAME in the source of every test function in this module,
    not by an editable allowlist here, so approving a new caller means
    writing a test that actually exercises it - not just typing its name
    into a list.
    """
    import localm.inference.http_server as hs_mod

    source = inspect.getsource(hs_mod)
    tree = ast.parse(source)

    referencing_functions: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def _visit_fn(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _visit_fn
        visit_AsyncFunctionDef = _visit_fn

        def visit_Name(self, node):
            if node.id == "_gpu_registry_sync" and self.stack:
                referencing_functions.add(self.stack[-1])
            self.generic_visit(node)

    _Visitor().visit(tree)

    assert referencing_functions, (
        "the AST walk found no references to _gpu_registry_sync at all - "
        "this sentinel is broken, not proof there is nothing to cover")

    this_file_source = open(__file__, encoding="utf-8").read()

    uncovered = [name for name in sorted(referencing_functions)
                if name not in this_file_source]
    assert not uncovered, (
        f"these http_server.py functions reference _gpu_registry_sync but "
        f"are never named in this test file: {uncovered} - add a test here "
        f"(see the existing off-loop tests for the pattern) before this "
        f"sentinel will pass. This is exactly the gap that let the "
        f"startup-hang regression through: a 5th/6th caller with no "
        f"coverage here.")


def test_lifespan_startup_syncs_gpu_registry_off_the_loop(tmp_path, monkeypatch):
    """The ONE-SHOT startup call to _gpu_registry_sync (lifespan(), distinct
    from the recurring heartbeat covered by the tests above) must also run
    off the event loop.

    Live incident this pins: at server startup, this call sat bare (no
    run_in_executor) while its heartbeat twin was already correctly
    offloaded - the exact "we fixed it in one place and never backported
    it to the other call site" gap this whole file exists to catch, except
    this call site had no test here at all. Symptom on a real box: /api/gpus,
    /api/doctor, /api/stats and /api/instances all blocked for the full
    ~15s GPU-probe deadline on ordinary GUI page load, with the hang alarm
    firing CRITICAL "event loop frozen" repeatedly.
    """
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    threads = []
    real_sync = hs._gpu_registry_sync

    def _spy():
        threads.append(threading.get_ident())
        return real_sync()

    monkeypatch.setattr(hs, "_gpu_registry_sync", _spy)

    app = create_app(None)
    # Only a real, non-isolated advertise()'d instance reaches this branch
    # (see the comment at its call site) - a bare create_app() never sets
    # these, so the test arms them itself to exercise the guarded path.
    app.state.instance_id = "test-startup-offload-instance"
    app.state.instance_port = 0
    app.state.instance_scheme = "http"
    app.state.bind_host = "127.0.0.1"

    async def scenario():
        async with app.router.lifespan_context(app):
            pass
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    assert threads, (
        "_gpu_registry_sync never ran during lifespan startup - this test "
        "did not exercise the guarded instance_id branch at all")
    assert all(t != loop_thread for t in threads), (
        "lifespan's one-shot startup call ran _gpu_registry_sync (registry "
        "file I/O + a GPU driver probe) ON the event loop thread, stalling "
        "every concurrent request during startup")


# --------------------------------------------------------------------------- #
#  The heartbeat's failure warning is throttled                                #
#                                                                              #
#  A heartbeat failure is usually PERSISTENT (an unwritable registry path, a   #
#  wedged driver probe). Warning unconditionally on a 20s tick emitted three   #
#  lines a minute, each with a full traceback, for the life of the server -    #
#  which is how the one line that mattered gets buried.                        #
# --------------------------------------------------------------------------- #

import logging


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _run_heartbeat_until(monkeypatch, sync_impl, target_calls):
    """Drive the real loop with a fast interval until *sync_impl* has been
    called *target_calls* times, then cancel it. Returns the captured records.

    Deterministic by construction rather than by clock: the stand-in counts its
    OWN calls and signals when it has been driven enough, so an assertion about
    "how many lines across N ticks" is a fact about the throttle rather than a
    race. The loop is the REAL one - only the tick period is overridden.
    """
    from localm.debuglog import logger as _dbg

    calls = {"n": 0}
    enough = threading.Event()

    def _counting():
        calls["n"] += 1
        if calls["n"] >= target_calls:
            enough.set()
        return sync_impl(calls["n"])

    monkeypatch.setattr(hs, "_gpu_registry_sync", _counting)

    handler = _Capture()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)

    async def scenario():
        task = asyncio.create_task(hs._gpu_registry_heartbeat_loop(interval=0.01))
        try:
            await asyncio.get_running_loop().run_in_executor(None, enough.wait, 10)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(scenario())
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    assert calls["n"] >= target_calls, (
        f"the heartbeat only ticked {calls['n']} times - this test never "
        "exercised the throttle")
    return handler.records


def _heartbeat_warnings(records):
    return [r for r in records
            if r[0] == "WARNING" and "gpu-registry heartbeat failed" in r[1]]


def test_a_persistently_failing_heartbeat_warns_once_then_throttles(monkeypatch):
    def _always_fails(n):
        raise OSError("registry path is unwritable")

    records = _run_heartbeat_until(monkeypatch, _always_fails, target_calls=5)

    warnings = _heartbeat_warnings(records)
    throttled = [r for r in records
                 if r[0] == "DEBUG" and "heartbeat still failing" in r[1]]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING across 5 failing ticks, got "
        f"{len(warnings)}: {warnings}")
    assert throttled, "the repeats vanished entirely instead of dropping to DEBUG"
    # The throttled line still has to identify the cause, or a CHANGE of cause
    # after the first warning would be invisible.
    assert "OSError" in throttled[0][1]


def test_a_heartbeat_that_recovers_warns_again_on_a_LATER_failure(monkeypatch):
    """A success must re-arm the warning.

    This is the half a plain "only ever warn once" flag gets wrong, and it is
    the difference between a throttle and a permanent silence: a second,
    unrelated outage hours later would otherwise never be reported at all.
    """
    def _fails_recovers_fails(n):
        if n == 3:
            return None          # one good tick in the middle
        raise OSError("registry path is unwritable")

    records = _run_heartbeat_until(monkeypatch, _fails_recovers_fails, target_calls=6)

    warnings = _heartbeat_warnings(records)
    assert len(warnings) == 2, (
        f"expected a second WARNING after the recovery, got {len(warnings)}")
