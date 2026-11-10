# SPDX-License-Identifier: AGPL-3.0-or-later
"""The per-GPU load lease (localm.inference.load_lease), the bounded load lock,
the "busy" answers the HTTP and MCP layers give, and the shared embedder,
reranker and speech model as residents the MCP cache can release."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference import load_lease

REPO = Path(__file__).resolve().parents[1]
GB = 1024 ** 3
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _child_env(home: Path) -> dict:
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _spawn(script: Path, *args, home: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(script), *map(str, args)],
                            env=_child_env(home), cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_NO_WINDOW)


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=_NO_WINDOW)
    p.wait()
    return p.pid


def _write_lease(root: Path, key: str, rec: dict) -> Path:
    d = root / (key + load_lease.LEASE_SUFFIX)
    d.mkdir(parents=True)
    (d / load_lease.OWNER_FILE).write_text(json.dumps(rec), encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def _no_leftover_holds():
    yield
    assert load_lease.held_keys() == {}, "a test left a lease hold behind"


_RACER = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    from localm.inference import load_lease

    device, out, go, need = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    while not go.exists():
        time.sleep(0.002)

    def admit():
        if int(device.read_text()) < need:
            return False
        time.sleep(0.6)
        device.write_text(str(int(device.read_text()) - need))
        return True

    try:
        with load_lease.held(["gpu0"], timeout=30, purpose="race test"):
            out.write_text(json.dumps({"admitted": admit()}))
    except load_lease.LoadBusyError as e:
        out.write_text(json.dumps({"busy": str(e)}))
''')


def test_two_interpreters_racing_admission_admit_exactly_one(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "racer.py"
    script.write_text(_RACER, encoding="utf-8")
    device = tmp_path / "device.txt"
    device.write_text(str(10 * GB))
    go = tmp_path / "go"
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    procs = [_spawn(script, device, out, go, 8 * GB, home=home) for out in outs]
    try:
        time.sleep(1.5)
        go.write_text("go")
        for p in procs:
            _, err = p.communicate(timeout=60)
            assert p.returncode == 0, err.decode(errors="replace")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    results = [json.loads(o.read_text()) for o in outs]
    assert sorted(r.get("admitted") for r in results) == [False, True], results
    assert int(device.read_text()) == 2 * GB
    assert not list((home / "run" / "gpu-lease").glob("*.lease"))


def test_a_dead_holders_lease_is_reclaimed(tmp_path):
    from localm import instances
    root = tmp_path / "lease"
    _write_lease(root, "gpu0", {"pid": _dead_pid(), "space": instances.pid_space_id(),
                                "start": None, "token": "gone", "purpose": "x"})
    taken = load_lease.acquire(["gpu0"], timeout=0.5, root=root)
    try:
        rec = json.loads((root / "gpu0.lease" / "owner.json").read_text())
        assert rec["pid"] == os.getpid()
        assert rec["token"] == load_lease._TOKEN
    finally:
        load_lease.release(taken, root=root)
    assert not (root / "gpu0.lease").exists()


def test_a_reused_pid_with_another_start_identity_is_reclaimed(tmp_path):
    from localm import instances
    root = tmp_path / "lease"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             creationflags=_NO_WINDOW)
    try:
        current = load_lease._start_identity(child.pid)
        if current is None:
            pytest.skip("this platform cannot read a process start identity")
        other = ({"filetime": current["filetime"] + 1} if "filetime" in current
                 else {**current, "ticks": current.get("ticks", 0) + 1,
                       "created": (current.get("created") or 0) + 100.0})
        _write_lease(root, "gpu0", {"pid": child.pid, "space": instances.pid_space_id(),
                                    "start": other, "token": "old"})
        taken = load_lease.acquire(["gpu0"], timeout=0.5, root=root)
        try:
            rec = json.loads((root / "gpu0.lease" / "owner.json").read_text())
        finally:
            load_lease.release(taken, root=root)
        assert taken == ("gpu0",)
        assert rec["token"] == load_lease._TOKEN
    finally:
        child.kill()
        child.wait()


@pytest.mark.parametrize("rec", [
    {"space": "another-pid-space", "start": None, "token": "t"},
    {"space": None, "start": None, "token": "t"},
], ids=["other_pid_space", "no_pid_space"])
def test_a_lease_whose_holder_cannot_be_judged_is_kept(tmp_path, rec):
    root = tmp_path / "lease"
    _write_lease(root, "all", {"pid": _dead_pid(), **rec})
    caught = None
    try:
        load_lease.acquire(["gpu1"], timeout=0.3, root=root)
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    assert (root / "all.lease").is_dir()
    assert type(caught) is load_lease.LoadBusyError


_HOLDER = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    from localm.inference import load_lease
    held, stop = Path(sys.argv[1]), Path(sys.argv[2])
    with load_lease.held([sys.argv[3]], timeout=10, purpose="holder test"):
        held.write_text(str(os.getpid()))
        while not stop.exists():
            time.sleep(0.01)
''')


def test_a_live_holder_keeps_its_lease_until_it_releases(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    held, stop = tmp_path / "held", tmp_path / "stop"
    proc = _spawn(script, held, stop, "all", home=home)
    root = home / "run" / "gpu-lease"
    try:
        deadline = time.monotonic() + 30
        while not held.exists():
            assert proc.poll() is None, proc.communicate()[1].decode(errors="replace")
            assert time.monotonic() < deadline
            time.sleep(0.01)
        caught = None
        started = time.monotonic()
        try:
            load_lease.acquire(["gpu3"], timeout=0.4, root=root)
        except BaseException as e:      # noqa: BLE001 - the type is asserted below
            caught = e
        assert (root / "all.lease").is_dir()
        assert type(caught) is load_lease.LoadBusyError
        assert f"process {held.read_text()}" in str(caught)
        assert time.monotonic() - started >= 0.4
        stop.write_text("stop")
        proc.communicate(timeout=30)
        taken = load_lease.acquire(["gpu3"], timeout=5, root=root)
        load_lease.release(taken, root=root)
        assert taken == ("gpu3",)
    finally:
        stop.write_text("stop")
        if proc.poll() is None:
            proc.kill()


def test_keys_on_different_gpus_do_not_conflict_and_all_conflicts_with_each(tmp_path):
    from localm import instances
    root = tmp_path / "lease"
    _write_lease(root, "gpu1", {"pid": os.getppid() or 1, "space": "x", "token": "t"})
    taken = load_lease.acquire(["gpu0"], timeout=0.3, root=root)
    load_lease.release(taken, root=root)
    assert taken == ("gpu0",)
    caught = None
    try:
        load_lease.acquire([load_lease.ALL_DEVICES], timeout=0.3, root=root)
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    assert type(caught) is load_lease.LoadBusyError
    assert instances.pid_space_id() != "x"


def test_lease_keys_follow_the_configured_split():
    assert load_lease.lease_keys({"gpu_split_indices": [2, 0, 2]}) == ("gpu0", "gpu2")
    assert load_lease.lease_keys({"gpu_split_indices": None}) == (load_lease.ALL_DEVICES,)
    assert load_lease.lease_keys({"gpu_split_indices": [True, -1, "1"]}) == (
        load_lease.ALL_DEVICES,)
    assert load_lease.lease_keys({}, device="cpu") == ()
    assert load_lease.lease_keys({}, n_gpu_layers=0) == ()


def test_wait_bound_sits_below_the_native_load_ceiling():
    from localm.inference.backends import _hf_runner
    from localm.inference.backends.llamacpp import _runner
    assert 0 < load_lease.LOAD_WAIT_TIMEOUT_S < _runner.LOAD_TIMEOUT_DEFAULT
    assert load_lease.LOAD_WAIT_TIMEOUT_S < _hf_runner.LOAD_TIMEOUT_DEFAULT


class _Backend:
    can_embed = True

    def __init__(self, on_load=None):
        self.loaded = False
        self.loads = 0
        self._on_load = on_load

    def load(self):
        if self._on_load is not None:
            self._on_load()
        self.loads += 1
        self.loaded = True

    def set_load_cancel(self, event):
        pass


def _bare_engine(backend, *, device="cuda"):
    from localm.inference.engine import Engine
    e = Engine.__new__(Engine)
    e._backend = backend
    e._backend_fresh = True
    e._backend_args = {"device": device}
    e.display_name = "fake-model"
    e.model_path = "fake.gguf"
    e.loading = False
    return e


def test_a_load_lock_waiter_is_answered_busy_after_the_bound(monkeypatch):
    from localm.inference import engine as eng_mod
    monkeypatch.setattr(load_lease, "LOAD_WAIT_TIMEOUT_S", 0.3)
    holding, done = threading.Event(), threading.Event()

    def hold():
        with eng_mod._LOAD_LOCK:
            holding.set()
            done.wait(10)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert holding.wait(5)
    backend = _Backend()
    engine = _bare_engine(backend, device="cpu")
    caught = None
    started = time.monotonic()
    try:
        engine.load()
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    finally:
        done.set()
        t.join(5)
    assert backend.loads == 0
    assert type(caught) is load_lease.LoadBusyError
    assert 0.25 <= time.monotonic() - started < 5
    assert "Busy: a model is loading" in str(caught)


def test_lock_order_lease_outer_load_lock_inner(monkeypatch):
    from localm.inference import engine as eng_mod
    order = []
    real_acquire = load_lease.acquire

    def acquire(*a, **k):
        order.append("lease")
        return real_acquire(*a, **k)

    class RecordingLock:
        def __init__(self, inner):
            self.inner = inner

        def acquire(self, blocking=True, timeout=-1):
            order.append("load_lock")
            return self.inner.acquire(blocking, timeout)

        def release(self):
            self.inner.release()

    monkeypatch.setattr(load_lease, "acquire", acquire)

    def on_load():
        order.append("load")
        assert load_lease.held_keys() == {load_lease.ALL_DEVICES: 1}
        assert eng_mod._LOAD_LOCK._is_owned()

    engine = _bare_engine(_Backend(on_load))
    engine._load_lock = RecordingLock(eng_mod._LOAD_LOCK)
    with patch("localm.inference.engine.console"), \
            patch.object(type(engine), "_attach_registered_adapters", lambda self: None):
        engine.load()
    assert order == ["lease", "load_lock", "load"]


def test_a_thread_holding_the_load_lock_gets_a_lease_its_process_holds(tmp_path):
    from localm.inference import engine as eng_mod
    root = tmp_path / "lease"
    outer = load_lease.acquire(["all"], timeout=1, root=root)
    try:
        got = []

        def inner():
            with eng_mod._LOAD_LOCK:
                got.append(load_lease.acquire(["gpu0"], timeout=1, root=root))

        t = threading.Thread(target=inner)
        t.start()
        t.join(5)
        assert got == [("gpu0",)]
        load_lease.release(got[0], root=root)
    finally:
        load_lease.release(outer, root=root)
    assert not list(root.glob("*.lease"))


def test_http_renders_busy_as_503_with_retry_after():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from localm.inference.app_assembly.errors import register_exception_handlers
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/x")
    def _x():
        raise load_lease.LoadBusyError(load_lease.busy_message(3.0))

    r = TestClient(app).get("/x")
    assert r.status_code == 503
    assert r.headers["Retry-After"] == str(load_lease.BUSY_RETRY_AFTER_S)
    assert r.json()["detail"].startswith("Busy: a model is loading")


def test_switch_load_passes_busy_through_without_latching_a_failure():
    from localm.inference import http_server as hs
    engine = MagicMock()
    engine.load.side_effect = load_lease.LoadBusyError("Busy: a model is loading.")

    async def go():
        return await hs._switch_load(asyncio.get_running_loop(), "busy-model", engine,
                                     preempt=False)

    caught = None
    try:
        asyncio.run(go())
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    assert hs._routing_latch.failure("busy-model") is None
    assert type(caught) is load_lease.LoadBusyError


def test_mcp_renders_busy_and_does_not_latch_it():
    from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools

    def factory(name):
        e = MagicMock()
        e.display_name = name
        e.loaded = False
        e.active_requests = 0
        e.unloading = False
        e.load.side_effect = load_lease.LoadBusyError("Busy: a model is loading.")
        return e

    cache = EngineCache("m", engine_factory=factory)
    caught = None
    with patch("localm.config.load_config",
               return_value={"max_resident_models": None, "pinned_models": None}):
        try:
            cache.get_loaded("m")
        except BaseException as e:      # noqa: BLE001 - the type is asserted below
            caught = e
    assert type(caught) is load_lease.LoadBusyError
    assert cache.routing_latch().failure("m") is None

    server = MCPStdioServer(build_tools(cache))
    server.tools["chat"]["handler"] = MagicMock(
        side_effect=load_lease.LoadBusyError("Busy: a model is loading."))
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "chat", "arguments": {"prompt": "x"}}})
    assert resp["result"]["isError"] is True
    assert resp["result"]["content"][0]["text"].startswith("Busy: a model is loading")


def test_mcp_cache_holds_one_lease_from_admission_through_the_load():
    from localm.plugins.mcpserver.server import EngineCache
    seen = {}

    def factory(name):
        e = MagicMock()
        e.display_name = name
        e.loaded = False
        e.active_requests = 0
        e.unloading = False

        def load():
            seen["at_load"] = load_lease.held_keys()
            e.loaded = True

        e.load.side_effect = load
        return e

    cache = EngineCache("m", engine_factory=factory)
    real = EngineCache._make_room_for

    def room(self, name):
        seen["at_admission"] = load_lease.held_keys()
        return real(self, name)

    with patch.object(EngineCache, "_make_room_for", room), \
            patch("localm.config.load_config",
                  return_value={"max_resident_models": None, "pinned_models": None}):
        cache.get_loaded("m")
    assert seen["at_admission"] == {load_lease.ALL_DEVICES: 1}
    assert seen["at_load"] == {load_lease.ALL_DEVICES: 1}


# --- the shared embedder, reranker and speech model as residents -----------


def _shared_embedder(monkeypatch, *, hold):
    from localm.inference import embedder, reranker, speech
    state = {"loaded": True}
    released = []
    monkeypatch.setattr(embedder, "resident_vram_hold",
                        lambda: (state["loaded"], hold if state["loaded"] else None))

    def reset(*, force=True):
        released.append(force)
        if hold == 0:
            return False
        state["loaded"] = False
        return True

    monkeypatch.setattr(embedder, "reset_embedder", reset)
    monkeypatch.setattr(reranker, "resident_vram_hold", lambda: (False, None))
    monkeypatch.setattr(speech, "resident_vram_hold", lambda: (False, None))
    return state, released


def _cache_with(fits):
    from localm.plugins.mcpserver.server import EngineCache
    loads = []

    def factory(name):
        loads.append(name)
        e = MagicMock()
        e.display_name = name
        e.loaded = True
        e.active_requests = 0
        e.unloading = False
        return e

    cache = EngineCache("m", engine_factory=factory)
    cache.loads = loads
    return cache, patch.object(EngineCache, "_fits_alongside",
                               lambda self, name, required: fits())


def _cfg():
    return patch("localm.config.load_config",
                 return_value={"max_resident_models": None, "pinned_models": None})


def _sized():
    from localm.plugins.mcpserver.server import EngineCache
    return patch.object(EngineCache, "_model_required_bytes", lambda self, n: 4 * GB)


def test_an_idle_embedder_counts_in_admission_and_is_released_for_a_chat_model(monkeypatch):
    state, released = _shared_embedder(monkeypatch, hold=2 * GB)
    cache, fits = _cache_with(lambda: not state["loaded"])
    with fits, _cfg(), _sized():
        cache.get("a")
    assert released == [False]
    assert cache.loads == ["a"]


def test_an_embedder_in_use_is_never_released(monkeypatch):
    state, released = _shared_embedder(monkeypatch, hold=0)
    cache, fits = _cache_with(lambda: False)
    with fits, _cfg(), _sized():
        cache.get("a")
    assert released == []
    assert state["loaded"] is True
    assert cache.loads == ["a"]


def test_an_idle_chat_model_is_evicted_before_the_embedder(monkeypatch):
    from localm.plugins.mcpserver.server import EngineCache
    state, released = _shared_embedder(monkeypatch, hold=2 * GB)
    cache, _ = _cache_with(lambda: True)
    with patch.object(EngineCache, "_fits_alongside", lambda self, n, r: True), \
            _cfg(), _sized():
        a = cache.get("a")
    answers = iter([False, True])
    with patch.object(EngineCache, "_fits_alongside", lambda self, n, r: next(answers)), \
            patch.object(EngineCache, "_evict",
                         lambda self, victim, loading: (cache._lru.remove(victim),
                                                        cache._engines.pop(victim), True)[2]), \
            _cfg(), _sized():
        cache.get("b")
    assert released == []
    assert state["loaded"] is True
    assert cache.resident == ["b"]
    assert a is not None


def test_shared_residents_lists_an_unreadable_module_as_unknown(monkeypatch):
    from localm.inference import embedder, reranker, residency, speech

    def broken():
        raise OSError("state unreadable")

    monkeypatch.setattr(embedder, "resident_vram_hold", broken)
    monkeypatch.setattr(reranker, "resident_vram_hold", lambda: (True, 0))
    monkeypatch.setattr(speech, "resident_vram_hold", lambda: (False, None))
    got = residency.shared_residents()
    assert got == [residency.SharedResident("embedding model", None),
                   residency.SharedResident("reranker model", 0)]
    assert residency.pick_shared_eviction_victim(got).kind == "embedding model"
    assert residency.pick_shared_eviction_victim(got[1:]) is None


# --- review fixes -----------------------------------------------------------


def test_a_lease_whose_reclaim_failed_is_not_counted_as_taken(tmp_path, monkeypatch):
    from localm import instances
    root = tmp_path / "lease"
    _write_lease(root, "gpu0", {"pid": _dead_pid(), "space": instances.pid_space_id(),
                                "start": None, "token": "gone"})
    monkeypatch.setattr(load_lease, "_remove_dir", lambda d: None)
    caught = None
    try:
        load_lease.acquire(["gpu0"], timeout=0, root=root)
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    assert type(caught) is load_lease.LoadBusyError
    assert "could not be removed" in str(caught)
    assert json.loads((root / "gpu0.lease" / "owner.json").read_text())["token"] == "gone"


def _foreign_holder(home: Path) -> Path:
    root = home / "run" / "gpu-lease"
    from localm import instances
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             creationflags=_NO_WINDOW)
    _write_lease(root, load_lease.ALL_DEVICES,
                 {"pid": child.pid, "space": instances.pid_space_id(),
                  "start": load_lease._start_identity(child.pid), "token": "other"})
    return child


def test_switch_try_take_answers_false_while_another_process_holds_the_gpu():
    from localm import config
    from localm.inference import http_server as hs
    child = _foreign_holder(config.HOME_DIR)
    try:
        lease = hs._SwitchLease("m")

        async def go():
            return await lease.try_take(asyncio.get_running_loop())

        started = time.monotonic()
        assert asyncio.run(go()) is False
        assert time.monotonic() - started < 5
        assert lease.keys is None
    finally:
        child.kill()
        child.wait()


def test_a_cancelled_switch_take_releases_the_lease_it_got_late(tmp_path):
    from localm.inference import http_server as hs
    gate = threading.Event()
    real = load_lease.acquire

    def slow_acquire(*a, **k):
        gate.wait(5)
        return real(*a, **k)

    lease = hs._SwitchLease("m")

    async def go():
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(lease.take(loop))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if load_lease.held_keys() == {}:
                break

    with patch.object(load_lease, "acquire", slow_acquire):
        asyncio.run(go())
    assert load_lease.held_keys() == {}


def test_a_busy_context_grow_is_not_latched_as_a_failed_grow(monkeypatch):
    from localm.inference import http_server as hs
    engine = MagicMock()
    engine.can_reserve_context.return_value = True
    engine.context_capacity.return_value = 4096
    engine.loaded = True
    engine.unloading = False
    engine.context_grow_failed = None
    monkeypatch.setitem(hs._engines, "grow-m", engine)
    monkeypatch.setattr(hs, "engine_context_limit", lambda e: (None, None))

    async def busy(*a, **k):
        raise load_lease.LoadBusyError("Busy: a model is loading.")

    monkeypatch.setattr(hs, "switch_engine", busy)
    caught = None
    try:
        asyncio.run(hs.grow_context_window(engine, 9000))
    except BaseException as e:      # noqa: BLE001 - the type is asserted below
        caught = e
    assert engine.context_grow_failed is None
    assert type(caught) is load_lease.LoadBusyError


def test_cpu_only_shared_loads_take_no_gpu_lease():
    assert load_lease.lease_keys({"n_gpu_layers": 0}, n_gpu_layers=0) == ()
    assert load_lease.lease_keys({}, device="cpu") == ()
    assert load_lease.lease_keys({}, n_gpu_layers=None) == (load_lease.ALL_DEVICES,)
