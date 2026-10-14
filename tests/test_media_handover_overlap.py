# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two media jobs never interleave a chat-model reload with a media model load.

Every native media job unloads the chat model, generates, then reloads the chat
model. The reload takes seconds. A second job that reached its media model load
while the first job's reload was still running loaded the two models onto the
card together: the second job's generation ran several times slower and the chat
model came back into whatever VRAM was left.

These tests run real threads through the real ``media_handover_job`` wrapper and
the real ``reload_chat_after_media``; only the chat-load HTTP request and the
backend's ``free_vram`` are faked, and every ordering is asserted on the recorded
timeline.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from localm import vram

WAIT = 10.0


class FakeJob:
    def __init__(self, name):
        self.name = name
        self.lines = []
        self.cancel_requested = False

    def push(self, event):
        self.lines.append(event.get("text"))


class FakeBackend:
    def __init__(self, timeline, name="backend"):
        self.timeline = timeline
        self.name = name
        self.free_calls = 0

    def free_vram(self, s):
        self.free_calls += 1
        self.timeline.append(f"free:{self.name}")
        return True


class Timeline(list):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def append(self, item):
        with self._lock:
            super().append(item)


@pytest.fixture
def timeline():
    return Timeline()


@pytest.fixture
def chat_load(monkeypatch, timeline):
    """Replace the chat-model load request. ``gate`` (when cleared) holds a load
    in flight until the test sets it."""
    class Ctl:
        gate = threading.Event()
        started = threading.Event()
        calls = 0

    Ctl.gate.set()

    class Resp:
        ok = True
        status_code = 200

    def self_request(method, path, **kw):
        assert path == "/models/load", path
        Ctl.calls += 1
        timeline.append("chat_load_start")
        Ctl.started.set()
        assert Ctl.gate.wait(WAIT)
        timeline.append("chat_load_end")
        return Resp()

    monkeypatch.setattr("localm.selfclient.self_request", self_request)
    return Ctl


@pytest.fixture(autouse=True)
def _handover_state_is_clean():
    assert vram._handover_jobs == 0 and not vram._handover_reloading
    yield
    assert vram._handover_jobs == 0, "a job left itself counted as in flight"
    assert not vram._handover_reloading and vram._handover_deferred is None


def _reload(job, backend):
    vram.reload_chat_after_media(job, "http://127.0.0.1:1", {"reload_after": True},
                                 backend, "image", None)


def _wait_for(predicate):
    done = threading.Event()
    for _ in range(int(WAIT / 0.01)):
        if predicate():
            return
        done.wait(0.01)
    raise AssertionError("condition not reached")


def _run(fn, job):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("rv", fn(job)), daemon=True)
    t.start()
    return t, out


def test_a_second_job_in_flight_takes_over_the_chat_reload(timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()
    a_reload_returned = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        timeline.append("A:generated")
        assert b_entered.wait(WAIT)
        _reload(job, backend)
        timeline.append("A:reload_returned")
        a_reload_returned.set()
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        assert a_reload_returned.wait(WAIT)
        timeline.append("B:media_model_load")
        _reload(job, backend)
        return True

    ta, oa = _run(job_a, a)
    # B enters only after A is counted, so A's reload sees a job in flight.
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, ob = _run(job_b, b)
    ta.join(WAIT)
    tb.join(WAIT)

    assert oa["rv"] is True and ob["rv"] is True
    assert chat_load.calls == 1
    assert timeline.index("B:media_model_load") < timeline.index("chat_load_start")
    assert backend.free_calls == 1
    assert any("reloads after the last one" in (t or "") for t in a.lines)


def test_a_job_starting_during_a_reload_waits_for_it(timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b = FakeJob("A"), FakeJob("B")
    chat_load.gate.clear()

    @vram.media_handover_job
    def job_a(job):
        _reload(job, backend)
        return True

    @vram.media_handover_job
    def job_b(job):
        timeline.append("B:media_model_load")
        return True

    ta, _ = _run(job_a, a)
    assert chat_load.started.wait(WAIT)
    tb, ob = _run(job_b, b)
    # B is parked behind the reload: it has said so and has not started.
    _wait_for(lambda: any("finish reloading" in (t or "") for t in b.lines))
    assert any("finish reloading" in (t or "") for t in b.lines)
    threading.Event().wait(0.5)
    assert "B:media_model_load" not in timeline
    chat_load.gate.set()
    ta.join(WAIT)
    tb.join(WAIT)

    assert ob["rv"] is True
    assert timeline.index("chat_load_end") < timeline.index("B:media_model_load")


def test_cancel_while_waiting_for_a_reload_skips_the_job(timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b = FakeJob("A"), FakeJob("B")
    chat_load.gate.clear()
    ran = []

    @vram.media_handover_job
    def job_a(job):
        _reload(job, backend)
        return True

    @vram.media_handover_job
    def job_b(job):
        ran.append(True)
        return True

    ta, _ = _run(job_a, a)
    assert chat_load.started.wait(WAIT)
    tb, ob = _run(job_b, b)
    _wait_for(lambda: any("finish reloading" in (t or "") for t in b.lines))
    b.cancel_requested = True
    tb.join(WAIT)
    assert ob["rv"] is False and ran == []
    assert b.lines[-1] == "Cancelled."
    chat_load.gate.set()
    ta.join(WAIT)


def test_the_last_job_reloads_even_when_it_did_not_swap(timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, backend)
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        _wait_for(lambda: any("reloads after the last one" in (t or "") for t in a.lines))
        timeline.append("B:done_without_reload")
        return True

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, ob = _run(job_b, b)
    ta.join(WAIT)
    tb.join(WAIT)

    assert ob["rv"] is True
    assert chat_load.calls == 1
    assert timeline.index("B:done_without_reload") < timeline.index("chat_load_start")
    assert "Chat model ready." in a.lines + b.lines


def test_a_failing_body_and_a_failing_deferred_reload_leave_nothing_counted(timeline):
    class Boom(FakeBackend):
        def free_vram(self, s):
            raise RuntimeError("driver gone")

    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, Boom(timeline))
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        _wait_for(lambda: any("reloads after the last one" in (t or "") for t in a.lines))
        raise ValueError("generation blew up")

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    results = {}

    def run_b():
        try:
            job_b(b)
        except ValueError as e:
            results["err"] = str(e)

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()
    ta.join(WAIT)
    tb.join(WAIT)

    assert results["err"] == "generation blew up"
    assert any("Reload deferred to the next message (driver gone)" in (t or "")
               for t in a.lines + b.lines)


def test_a_deferred_reload_still_frees_the_deferring_jobs_backend(timeline, chat_load):
    music, image = FakeBackend(timeline, "music"), FakeBackend(timeline, "image")
    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()
    a_deferred = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, music)
        a_deferred.set()
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        assert a_deferred.wait(WAIT)
        _reload(job, image)
        return True

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, _ = _run(job_b, b)
    ta.join(WAIT)
    tb.join(WAIT)

    assert music.free_calls == 1 and image.free_calls == 1
    load = timeline.index("chat_load_start")
    assert timeline.index("free:music") < load and timeline.index("free:image") < load


def test_a_single_job_reloads_exactly_as_before(timeline, chat_load):
    backend = FakeBackend(timeline)
    job = FakeJob("A")

    @vram.media_handover_job
    def only(job):
        _reload(job, backend)
        return True

    t, out = _run(only, job)
    t.join(WAIT)

    assert out["rv"] is True
    assert timeline == ["free:backend", "chat_load_start", "chat_load_end"]
    assert job.lines == ["Reloading the chat model...", "Chat model ready."]


_PLUG_FILES = ("image", "video", "music")


def _own_calls(fn):
    """Calls in *fn* itself, not in a function defined inside it."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _job_bodies_that_reload(tree):
    """Every function that itself opens a ``restore_chat_after_media`` block."""
    return [fn for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(n, ast.With)
                    and any(isinstance(i.context_expr, ast.Call)
                            and getattr(i.context_expr.func, "id", None)
                            == "restore_chat_after_media" for i in n.items)
                    for n in _own_nodes(fn))]


def _own_nodes(fn):
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def test_every_media_job_body_runs_inside_the_handover_wrapper():
    root = Path(vram.__file__).resolve().parent / "plugins" / "builtin"
    seen = 0
    for kind in _PLUG_FILES:
        tree = ast.parse((root / kind / "plug.py").read_text(encoding="utf-8"))
        wrapped_by_call = {
            n.args[0].id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "media_handover_job"
            and n.args and isinstance(n.args[0], ast.Name)}
        for fn in _job_bodies_that_reload(tree):
            decorated = any(getattr(d, "id", None) == "media_handover_job"
                            for d in fn.decorator_list)
            assert decorated or fn.name in wrapped_by_call, (kind, fn.name)
            seen += 1
    assert seen >= 4, "expected the image generate, image upscale, video and music job bodies"


def test_a_last_job_with_reload_off_does_not_replay_an_earlier_jobs_reload(
        timeline, chat_load):
    music, image = FakeBackend(timeline, "music"), FakeBackend(timeline, "image")
    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()
    a_deferred = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, music)
        a_deferred.set()
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        assert a_deferred.wait(WAIT)
        vram.reload_chat_after_media(job, "http://127.0.0.1:1", {"reload_after": False},
                                     image, "image", None)
        return True

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, ob = _run(job_b, b)
    ta.join(WAIT)
    tb.join(WAIT)

    assert ob["rv"] is True
    assert chat_load.calls == 0
    assert music.free_calls == 0 and image.free_calls == 0
    assert any("Keeping the image backend loaded" in (t or "") for t in b.lines)


def test_a_reload_off_job_that_is_not_last_leaves_the_deferred_reload_to_the_last(
        timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b, c = FakeJob("A"), FakeJob("B"), FakeJob("C")
    b_entered, c_entered = threading.Event(), threading.Event()
    a_deferred, b_done = threading.Event(), threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, backend)
        a_deferred.set()
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        assert a_deferred.wait(WAIT)
        assert c_entered.wait(WAIT)
        vram.reload_chat_after_media(job, "http://127.0.0.1:1", {"reload_after": False},
                                     FakeBackend(timeline, "other"), "image", None)
        b_done.set()
        return True

    @vram.media_handover_job
    def job_c(job):
        c_entered.set()
        assert b_done.wait(WAIT)
        return True

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, _ = _run(job_b, b)
    _wait_for(b_entered.is_set)
    tc, _ = _run(job_c, c)
    for t in (ta, tb, tc):
        t.join(WAIT)

    assert chat_load.calls == 1


def test_a_last_job_that_did_not_swap_with_reload_off_drops_the_deferred_reload(
        timeline, chat_load):
    backend = FakeBackend(timeline)
    a, b = FakeJob("A"), FakeJob("B")
    b_entered = threading.Event()
    a_deferred = threading.Event()

    @vram.media_handover_job
    def job_a(job):
        assert b_entered.wait(WAIT)
        _reload(job, backend)
        a_deferred.set()
        return True

    @vram.media_handover_job
    def job_b(job):
        b_entered.set()
        assert a_deferred.wait(WAIT)
        with vram.restore_chat_after_media(job, "http://127.0.0.1:1",
                                           {"reload_after": False}, backend, "image",
                                           None, swapped=False):
            pass
        return True

    ta, _ = _run(job_a, a)
    _wait_for(lambda: vram._handover_jobs >= 1)
    tb, ob = _run(job_b, b)
    ta.join(WAIT)
    tb.join(WAIT)

    assert ob["rv"] is True
    assert chat_load.calls == 0 and backend.free_calls == 0
