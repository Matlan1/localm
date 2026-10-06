# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder session pinned to privacy by its project's config must also keep chat
content out of the debug log from the isolated GGUF worker child.

``localm.audit`` keeps that pin in a parent-process counter, and the worker is a
spawned child whose copy of the counter is always zero. These tests drive a REAL
spawned child through the worker's own entry point, with LOCALM_MODE=log so the
mode env alone would allow content.
"""

from __future__ import annotations

import multiprocessing as mp
import types

import pytest

from localm import audit
from localm.audit import SessionMode
from localm.inference.backends.llamacpp import _runner


def _probe_main(req_q, resp_q, ctrl_q) -> None:
    from localm.debuglog import debug_content_enabled
    while True:
        cmd = req_q.get()
        if cmd is None:
            return
        resp_q.put(debug_content_enabled())


def _entry_with_probe(req_q, resp_q, ctrl_q, crash_trace_path=None,
                      coder_privacy=None) -> None:
    _runner._runner_main = _probe_main
    _runner._runner_entry(req_q, resp_q, ctrl_q, crash_trace_path, coder_privacy)


@pytest.fixture
def debug_log_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setenv("LOCALM_DEBUG", str(tmp_path / "debug.log"))


class _Child:
    def __init__(self, coder_privacy):
        ctx = mp.get_context("spawn")
        self.req_q, self.resp_q, self.ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(
            target=_entry_with_probe,
            args=(self.req_q, self.resp_q, self.ctrl_q, None, coder_privacy),
            daemon=True)
        self.proc.start()

    def content_enabled(self) -> bool:
        self.req_q.put("probe")
        return self.resp_q.get(timeout=60)

    def close(self) -> None:
        self.req_q.put(None)
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()


@pytest.fixture
def child_factory():
    made = []

    def make(coder_privacy):
        c = _Child(coder_privacy)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def test_child_without_a_coder_session_still_allows_content(
        debug_log_env, child_factory):
    assert audit.any_coder_session_is_privacy() is False
    child = child_factory(audit.shared_coder_privacy_value())
    assert child.content_enabled() is True


def test_privacy_coder_session_in_the_parent_suppresses_content_in_the_child(
        debug_log_env, child_factory):
    audit.register_coder_session_mode(SessionMode.PRIVACY)
    try:
        from localm.debuglog import debug_content_enabled
        assert debug_content_enabled() is False
        child = child_factory(audit.shared_coder_privacy_value())
        assert child.content_enabled() is False
    finally:
        audit.unregister_coder_session_mode(SessionMode.PRIVACY)


def test_register_and_unregister_reach_an_already_running_child(
        debug_log_env, child_factory):
    child = child_factory(audit.shared_coder_privacy_value())
    assert child.content_enabled() is True
    audit.register_coder_session_mode(SessionMode.PRIVACY)
    try:
        assert child.content_enabled() is False
    finally:
        audit.unregister_coder_session_mode(SessionMode.PRIVACY)
    assert child.content_enabled() is True


def test_a_non_privacy_session_does_not_flip_the_child(
        debug_log_env, child_factory):
    child = child_factory(audit.shared_coder_privacy_value())
    audit.register_coder_session_mode(SessionMode.LOG)
    try:
        assert child.content_enabled() is True
    finally:
        audit.unregister_coder_session_mode(SessionMode.LOG)


def test_an_unreadable_shared_value_counts_as_privacy():
    class _Broken:
        @property
        def value(self):
            raise OSError("shared memory gone")

    try:
        audit.adopt_shared_coder_privacy(_Broken())
        assert audit.any_coder_session_is_privacy() is True
    finally:
        audit.adopt_shared_coder_privacy(None)
    assert audit.any_coder_session_is_privacy() is False


def test_spawn_hands_the_shared_value_to_the_worker_entry(monkeypatch):
    seen = {}
    real_ctx = mp.get_context("spawn")

    class _Ctx:
        def Queue(self):
            return real_ctx.Queue()

        def Process(self, target, args, name, daemon):
            seen["target"], seen["args"] = target, args
            return types.SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(
        _runner, "mp", types.SimpleNamespace(get_context=lambda method: _Ctx()))
    runner = _runner.ModelRunner()
    runner._spawn()
    assert seen["target"] is _runner._runner_entry
    assert seen["args"][4] is audit.shared_coder_privacy_value()
