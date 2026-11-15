# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reads of config.json and registry.json never hold the server's event loop
for longer than ``config.LOOP_READ_WAIT_S``, and no route handler makes the
data folders or writes the store files on the loop."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import pytest

from localm import config

REPO_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_AFTER_S = 30.0


@pytest.fixture(autouse=True)
def _fresh_loop_reads():
    config._loop_reads_invalidate()
    yield
    config._loop_reads_invalidate()


def _write_config(data: dict) -> None:
    config.HOME_DIR.mkdir(parents=True, exist_ok=True)
    config.CONFIG_FILE.write_text(json.dumps(data), encoding="utf-8")


class _Gate:
    """Makes a function wait on an Event before running, and counts entries."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def wrap(self, fn):
        def blocked(*a, **k):
            self.calls += 1
            self.entered.set()
            self.release.wait(_RELEASE_AFTER_S)
            return fn(*a, **k)
        return blocked


def test_health_answers_while_a_config_read_and_mkdir_are_blocked(monkeypatch):
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    _write_config({"temperature": 0.3})
    app = create_app(None)
    gate = _Gate()
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
        assert client.get("/v1/config").status_code == 200
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        monkeypatch.setattr(config, "_read_json", gate.wrap(config._read_json))
        monkeypatch.setattr(config, "_mkdir_or_explain", gate.wrap(config._mkdir_or_explain))
        monkeypatch.setattr(config, "_dirs_made", set())
        slow: dict = {}

        def blocked_request():
            slow["response"] = client.get("/v1/config")

        t = threading.Thread(target=blocked_request, daemon=True)
        try:
            t.start()
            assert gate.entered.wait(10), "the request never reached the config read"
            started = time.monotonic()
            health = client.get("/health")
            took = time.monotonic() - started
        finally:
            gate.release.set()
            t.join(_RELEASE_AFTER_S)
            slow_done = not t.is_alive()
    assert health.status_code in (200, 503)
    assert took < 2 * config.LOOP_READ_WAIT_S + 1.0, took
    assert slow_done, "the request whose read was blocked never answered"
    assert slow["response"].status_code == 200
    assert slow["response"].json()["temperature"] == 0.3


def _wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)
    return cond()


def test_a_blocked_read_on_the_loop_answers_from_the_last_complete_read(monkeypatch, caplog):
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        first = config.load_config()["temperature"]
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        _write_config({"temperature": 0.75})
        t0 = time.monotonic()
        during = [config.load_config()["temperature"] for _ in range(5)]
        waited = time.monotonic() - t0
        gate.release.set()
        settled = _wait_for(lambda: "config.json" not in config._loop_inflight)
        after = config.load_config()["temperature"]
        fast_again = _wait_for(lambda: "config.json" not in config._loop_slow)
        return first, during, waited, settled, after, fast_again

    with caplog.at_level(logging.INFO, logger="localm"):
        first, during, waited, settled, after, fast_again = asyncio.run(scenario())
    assert first == 0.25
    assert during == [0.25] * 5
    assert waited < config.LOOP_READ_WAIT_S + 0.5, waited
    assert settled and fast_again
    assert after == 0.75
    msgs = [r.getMessage() for r in caplog.records if "config.json" in r.getMessage()]
    assert sum("last complete read" in m for m in msgs) == 1, msgs
    assert sum("are fast again" in m for m in msgs) == 1, msgs


def test_a_write_by_this_process_is_what_the_loop_reads_while_the_disk_is_slow(monkeypatch):
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        before = config.load_config()["temperature"]
        config.update_config(lambda c: c.__setitem__("temperature", 0.6))
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        t0 = time.monotonic()
        during = config.load_config()["temperature"]
        waited = time.monotonic() - t0
        gate.release.set()
        return before, during, waited

    before, during, waited = asyncio.run(scenario())
    assert before == 0.25
    assert during == 0.6
    assert waited < config.LOOP_READ_WAIT_S + 0.5, waited


def test_a_write_during_a_slow_read_never_blocks_the_loop(monkeypatch):
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        config.load_config()
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        config.load_config()
        assert "config.json" in config._loop_slow
        writer = threading.Thread(
            target=config.update_config,
            args=(lambda c: c.__setitem__("temperature", 0.6),), daemon=True)
        writer.start()
        assert gate.entered.wait(5)
        t0 = time.monotonic()
        while_writing = [config.load_config()["temperature"] for _ in range(3)]
        took = time.monotonic() - t0
        gate.release.set()
        writer.join(_RELEASE_AFTER_S)
        return while_writing, took, writer.is_alive(), config.load_config()["temperature"]

    while_writing, took, writer_alive, after = asyncio.run(scenario())
    assert while_writing == [0.25] * 3
    assert took < 0.5, took
    assert not writer_alive
    assert after == 0.6


def test_a_slow_registry_read_does_not_hold_the_loop_for_config_reads(monkeypatch):
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        config.load_config()
        config.load_registry()
        monkeypatch.setattr(config, "_read_json", gate.wrap(config._read_json))
        config.load_registry()
        assert "registry.json" in config._loop_slow
        t0 = time.monotonic()
        values = [config.load_config()["temperature"] for _ in range(3)]
        took = time.monotonic() - t0
        gate.release.set()
        return values, took

    values, took = asyncio.run(scenario())
    assert values == [0.25] * 3
    assert took < config.LOOP_READ_WAIT_S + 0.5, took


def test_a_change_by_another_process_is_in_force_within_the_kept_read_bound(monkeypatch):
    monkeypatch.setattr(config, "LOOP_KEPT_MAX_AGE_S", 1.0)
    monkeypatch.setattr(config, "LOOP_FRESH_READ_WAIT_S", 5.0)
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        config.load_config()
        kept_at = time.monotonic()
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        _write_config({"temperature": 0.9})
        threading.Timer(1.8, gate.release.set).start()
        answers = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            value = config.load_config()["temperature"]
            answers.append((time.monotonic() - kept_at, value, time.monotonic() - t0))
            if value == 0.9:
                break
            time.sleep(0.02)
        return answers

    try:
        answers = asyncio.run(scenario())
    finally:
        gate.release.set()
    old = [a for a in answers if a[1] == 0.25]
    assert old, answers
    assert max(a[0] for a in old) <= 1.0 + 0.2, old[-3:]
    assert answers[-1][1] == 0.9, answers[-3:]
    assert max(a[2] for a in answers) >= 0.5, "past the bound the loop did not wait"


def test_past_the_bound_the_loop_waits_for_the_fresh_read(monkeypatch):
    monkeypatch.setattr(config, "LOOP_KEPT_MAX_AGE_S", 0.3)
    monkeypatch.setattr(config, "LOOP_FRESH_READ_WAIT_S", 5.0)
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        config.load_config()
        time.sleep(0.4)
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        _write_config({"temperature": 0.9})
        threading.Timer(0.8, gate.release.set).start()
        t0 = time.monotonic()
        value = config.load_config()["temperature"]
        return value, time.monotonic() - t0

    try:
        value, took = asyncio.run(scenario())
    finally:
        gate.release.set()
    assert value == 0.9
    assert took >= 0.7, took


def test_past_the_bound_a_fresh_read_slower_than_its_wait_is_refused(monkeypatch, caplog):
    monkeypatch.setattr(config, "LOOP_KEPT_MAX_AGE_S", 0.2)
    monkeypatch.setattr(config, "LOOP_FRESH_READ_WAIT_S", 0.6)
    _write_config({"temperature": 0.25})
    gate = _Gate()

    async def scenario():
        config.load_config()
        time.sleep(0.3)
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        results = []
        for _ in range(3):
            t0 = time.monotonic()
            try:
                results.append((config.load_config()["temperature"], time.monotonic() - t0))
            except config.ConfigReadSlow as e:
                results.append((e, time.monotonic() - t0))
        gate.release.set()
        settled = _wait_for(lambda: "config.json" not in config._loop_inflight)
        return results, settled, config.load_config()["temperature"]

    with caplog.at_level(logging.WARNING, logger="localm"):
        try:
            results, settled, after = asyncio.run(scenario())
        finally:
            gate.release.set()
    assert all(isinstance(r[0], config.ConfigReadSlow) for r in results), results
    assert results[0][0].name == "config.json"
    assert results[0][1] < 0.6 + 0.5, results
    assert max(r[1] for r in results[1:]) < 0.2, results
    assert settled and after == 0.25
    refused = [r.getMessage() for r in caplog.records
               if "are refused until the read finishes" in r.getMessage()]
    assert len(refused) == 1, refused


def test_the_idle_unload_check_survives_a_slow_settings_read(monkeypatch):
    from localm.inference import http_server as hs

    def slow():
        raise config.ConfigReadSlow("config.json", 2.5)

    monkeypatch.setattr(config, "load_config", slow)
    assert hs._idle_unload_ttl() == 0


def test_the_fresh_read_wait_is_at_least_the_kept_read_wait():
    assert config.LOOP_READ_WAIT_S <= config.LOOP_FRESH_READ_WAIT_S


def test_a_request_whose_settings_read_is_too_slow_gets_a_503(monkeypatch):
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    _write_config({"temperature": 0.3})
    app = create_app(None)
    gate = _Gate()
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
        assert client.get("/v1/config").status_code == 200
        monkeypatch.setattr(config, "LOOP_KEPT_MAX_AGE_S", 0.0)
        monkeypatch.setattr(config, "LOOP_FRESH_READ_WAIT_S", 0.3)
        monkeypatch.setattr(config, "_read_json_checked", gate.wrap(config._read_json_checked))
        try:
            t0 = time.monotonic()
            refused = client.get("/v1/config")
            took = time.monotonic() - t0
        finally:
            gate.release.set()
    assert gate.calls >= 1
    assert refused.status_code == 503, refused.text
    assert refused.headers["retry-after"] == "5"
    assert "config.json" in refused.json()["detail"]
    assert took < 0.3 + 2.0, took


def test_a_slow_read_raised_by_a_middleware_or_a_route_is_answered_503():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from localm.inference.app_assembly import errors

    app = FastAPI()

    @app.middleware("http")
    async def _reads_settings(request, call_next):
        if request.url.path == "/in-middleware":
            raise config.ConfigReadSlow("config.json", 2.5)
        return await call_next(request)

    @app.get("/in-route")
    async def _route():
        raise config.ConfigReadSlow("registry.json", 2.5)

    errors.add_config_read_slow_answer(app)
    errors.register_exception_handlers(app)
    client = TestClient(app)
    for path, name in (("/in-middleware", "config.json"), ("/in-route", "registry.json")):
        r = client.get(path)
        assert r.status_code == 503, (path, r.text)
        assert r.headers["retry-after"] == "5"
        assert name in r.json()["detail"]


def test_a_change_made_by_another_process_is_read_fresh_on_the_loop():
    _write_config({"temperature": 0.25})

    async def scenario():
        before = config.load_config()["temperature"]
        _write_config({"temperature": 0.9})
        return before, config.load_config()["temperature"]

    assert asyncio.run(scenario()) == (0.25, 0.9)


def test_each_caller_gets_its_own_copy():
    _write_config({"plugins": {"a": {"x": 1}}})

    async def scenario():
        config.load_config()
        one = config.load_config()
        one["plugins"]["a"]["x"] = 2
        return config.load_config()["plugins"]["a"]["x"]

    assert asyncio.run(scenario()) == 1


def test_a_read_inside_a_store_lock_on_the_loop_reads_inline(monkeypatch):
    _write_config({"temperature": 0.25})
    submitted = []
    real_submit = config._submit_loop_read
    monkeypatch.setattr(config, "_submit_loop_read",
                        lambda job, fn: (submitted.append(job.name), real_submit(job, fn)))

    async def scenario():
        config.load_config()
        with config._io_lock:
            t0 = time.monotonic()
            value = config.load_config()["temperature"]
            return value, time.monotonic() - t0

    value, took = asyncio.run(scenario())
    assert value == 0.25
    assert submitted == []
    assert took < config.LOOP_READ_WAIT_S


def test_readers_make_the_data_folders_once():
    calls = []
    real = config._mkdir_or_explain

    def counting(path, *, is_home):
        calls.append(path)
        real(path, is_home=is_home)

    import unittest.mock as um
    with um.patch.object(config, "_mkdir_or_explain", counting):
        config.load_config()
        config.load_config()
        config.load_registry()
    assert calls == [config.HOME_DIR, config.MODELS_DIR]


_STORE_READS = {"_read_json", "_read_json_checked", "_read_json_state"}
_WRITERS = {"save_config", "update_config", "update_registry"}


def _calls(fn) -> set:
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_every_store_read_in_config_goes_through_the_loop_guard():
    tree = ast.parse((REPO_ROOT / "localm" / "config.py").read_text(encoding="utf-8"))
    funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    guarded = set()
    for f in funcs.values():
        for n in ast.walk(f):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "_loop_safe_read" and len(n.args) == 2
                    and isinstance(n.args[1], ast.Name)):
                guarded.add(n.args[1].id)
    readers = {name for name, f in funcs.items() if _calls(f) & _STORE_READS}
    unguarded = sorted(readers - guarded - _WRITERS)
    assert not unguarded, (
        f"config.py functions read config.json or registry.json without "
        f"_loop_safe_read: {unguarded}")
    called_raw = sorted(name for name, f in funcs.items()
                        for g in guarded if g in _calls(f) and name != g)
    assert not called_raw, f"a guarded reader is called directly: {called_raw}"


_LOOP_FORBIDDEN = {"ensure_dirs", "_mkdir_or_explain", "save_config", "update_config",
                   "save_registry", "update_registry", "_load_config_checked_now",
                   "_load_registry_now", "_peek_registry_now", "_model_index_now",
                   "_instance_id_now"}


def _direct_calls_in_async_bodies(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        stack = list(fn.body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                              ast.ClassDef)):
                continue
            if isinstance(n, ast.Call):
                f = n.func
                name = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else None)
                if name in _LOOP_FORBIDDEN:
                    yield fn.name, n.lineno, name
            stack.extend(ast.iter_child_nodes(n))


def test_no_async_function_makes_folders_or_writes_the_store_on_the_loop():
    found = []
    for path in sorted((REPO_ROOT / "localm").rglob("*.py")):
        for fname, line, name in _direct_calls_in_async_bodies(path):
            found.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line} "
                         f"{fname} calls {name}")
    assert not found, ("an async function calls a store write or a folder "
                       "creation directly on the event loop; call it through "
                       "run_in_threadpool: " + "; ".join(found))


def test_the_async_scan_finds_a_direct_call(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "async def handler():\n"
        "    from localm import config\n"
        "    config.ensure_dirs()\n"
        "    def inner():\n"
        "        config.save_config({})\n",
        encoding="utf-8")
    assert list(_direct_calls_in_async_bodies(sample)) == [("handler", 3, "ensure_dirs")]
