# SPDX-License-Identifier: AGPL-3.0-or-later
"""A media job that unloaded the chat model reloads it however the job ends.

A swap unloads the chat model before the media model loads. When the backend then
raises (a driver error, out of memory, a bug), the job ends "failed" and the chat
model must still come back: the reload runs once, best effort, and the original
exception is what the job reports.

The real job bodies (image generate, image upscale, video, music) are driven with
a backend whose generate raises. Only the chat-load request and the backend are
faked.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import vram

GB = 1024 ** 3


class _Job:
    def __init__(self):
        self.lines = []
        self.result = None
        self.outcomes = []
        self.cancel_requested = False

    def push(self, ev):
        self.lines.append(ev.get("text"))

    def progress(self, **kw):
        pass

    def mark_outcome(self, status):
        self.outcomes.append(status)


class _Resp:
    ok = True
    status_code = 200


@pytest.fixture
def chat(monkeypatch):
    """Record chat-load requests; no real server."""
    calls = []

    def self_request(method, path, **kw):
        calls.append((method, path))
        return _Resp()

    monkeypatch.setattr("localm.selfclient.self_request", self_request)
    return calls


def _fake_vram(monkeypatch, *, swap=True):
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: swap)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    monkeypatch.setattr("localm.vram.release_other_native_media",
                        lambda job, s, label: vram.ReleaseResult())
    monkeypatch.setattr("localm.vram.note_other_vram_holders", lambda job, s, label: None)
    monkeypatch.setattr("localm.vram.unload_chat_for_media",
                        lambda job, url, label, instance_token=None: True)


def _settings(**extra):
    return {"reload_after": True, "warning": "", "api_url": "http://127.0.0.1:8188",
            "backend": "comfy", "vram_estimate_bytes": 12 * GB, **extra}


def _capture_route(plug, route_name, request_obj):
    captured = {}

    class _FakeJobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            captured["fn"] = fn
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.headers = {}
    request.cookies = {}
    asyncio.run(getattr(plug, route_name)(request_obj, request))
    return captured["fn"]


def _image_generate(monkeypatch, *, backend_call):
    from localm.plugins.builtin.image import plug
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: _settings())
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "up"))
    monkeypatch.setattr(plug._backend, "generate", backend_call)
    monkeypatch.setattr(plug._backend, "free_vram", lambda s: True)
    return _capture_route(plug, "imagine", plug.ImagineRequest(prompt="a cat"))


def _video_generate(monkeypatch, *, backend_call):
    from localm.plugins.builtin.video import plug
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: _settings())
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "up"))
    monkeypatch.setattr(plug._backend, "generate", backend_call)
    monkeypatch.setattr(plug._backend, "free_vram", lambda s: True)
    return _capture_route(plug, "video", plug.VideoRequest(prompt="a cat"))


def _music_generate(monkeypatch, *, backend_call):
    from localm.plugins.builtin.music import plug
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: _settings())
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "up"))
    monkeypatch.setattr(plug._backend, "generate", backend_call)
    monkeypatch.setattr(plug._backend, "free_vram", lambda s: True)
    return _capture_route(plug, "music", plug.MusicRequest(tags="lofi"))


def _image_upscale(monkeypatch, *, backend_call):
    from localm.plugins.builtin.image import plug
    from localm.plugins.builtin.image.backends import native
    monkeypatch.setattr(plug._backend, "settings",
                        lambda cfg: _settings(backend="native"))
    monkeypatch.setattr(native, "ensure_upscaler_available",
                        lambda s, on_progress=None: (True, "up"))
    monkeypatch.setattr(native, "upscaler_vram_estimate_bytes", lambda name: 2 * GB)
    monkeypatch.setattr(native, "upscale", backend_call)
    monkeypatch.setattr(plug._backend, "free_vram", lambda s: True)
    return plug._make_upscale(source=Path("in.png"), upscaler="x", out_path=Path("out.png"),
                              owner=None, self_url="http://127.0.0.1:8642/v1",
                              instance_token=None, extra_meta={})


_BODIES = {
    "image_generate": _image_generate,
    "image_upscale": _image_upscale,
    "video": _video_generate,
    "music": _music_generate,
}


def _raise(*a, **k):
    raise RuntimeError("driver exploded")


@pytest.mark.parametrize("body", sorted(_BODIES))
def test_chat_model_reloads_when_the_backend_raises_after_a_swap(monkeypatch, chat, body):
    _fake_vram(monkeypatch)
    fn = _BODIES[body](monkeypatch, backend_call=_raise)
    job = _Job()

    with pytest.raises(RuntimeError) as err:
        fn(job)

    assert str(err.value) == "driver exploded"
    assert chat == [("POST", "/models/load")]
    assert "Chat model ready." in job.lines


@pytest.mark.parametrize("body", sorted(_BODIES))
def test_a_failing_reload_never_masks_the_original_exception(monkeypatch, chat, body):
    _fake_vram(monkeypatch)
    fn = _BODIES[body](monkeypatch, backend_call=_raise)

    def free_vram_raises(s):
        raise OSError("free_vram blew up")

    monkeypatch.setattr(f"localm.plugins.builtin.{body.split('_')[0]}.plug._backend.free_vram",
                        free_vram_raises)
    job = _Job()

    with pytest.raises(RuntimeError) as err:
        fn(job)

    assert str(err.value) == "driver exploded"
    assert any("Reload deferred to the next message (free_vram blew up)" in (t or "")
               for t in job.lines)
    assert chat == []


@pytest.mark.parametrize("body", sorted(_BODIES))
def test_no_reload_when_the_chat_model_was_never_unloaded(monkeypatch, chat, body):
    _fake_vram(monkeypatch, swap=False)
    fn = _BODIES[body](monkeypatch, backend_call=_raise)

    with pytest.raises(RuntimeError):
        fn(_Job())

    assert chat == []


@pytest.mark.parametrize("body", ["image_generate", "video", "music"])
def test_a_clean_job_reloads_exactly_once(monkeypatch, chat, body):
    _fake_vram(monkeypatch)
    fn = _BODIES[body](monkeypatch, backend_call=lambda *a, **k: (False, "nope"))
    job = _Job()

    assert fn(job) is False

    assert chat == [("POST", "/models/load")]
    assert job.outcomes == ["failed"]


def test_the_outcome_is_recorded_before_the_reload_runs(monkeypatch):
    order = []
    job = _Job()
    job.mark_outcome = lambda status: order.append(("outcome", status))
    monkeypatch.setattr(
        "localm.vram.reload_chat_after_media",
        lambda job, url, s, backend, label, instance_token=None: order.append("reload"))
    _fake_vram(monkeypatch)
    fn = _image_generate(monkeypatch, backend_call=lambda *a, **k: (True, "saved"))

    assert fn(job) is True
    assert order == [("outcome", "done"), "reload"]


def test_the_restore_block_reloads_once_on_a_raise_and_propagates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "localm.vram.reload_chat_after_media",
        lambda job, url, s, backend, label, instance_token=None: calls.append(label))

    with pytest.raises(KeyError):
        with vram.restore_chat_after_media(_Job(), "u", {"reload_after": True}, object(),
                                           "image", None):
            raise KeyError("x")

    assert calls == ["image"]


def test_a_failing_reload_after_a_clean_block_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("reload failed")

    monkeypatch.setattr("localm.vram.reload_chat_after_media", boom)

    with pytest.raises(RuntimeError, match="reload failed"):
        with vram.restore_chat_after_media(_Job(), "u", {"reload_after": True}, object(),
                                           "image", None):
            pass


_PLUG_FILES = ("image", "video", "music")


def _call_name(call):
    return getattr(call.func, "id", getattr(call.func, "attr", None))


def _is_backend_call(node):
    """The chat unload, or ``_backend.generate`` / ``native.upscale``."""
    if not isinstance(node, ast.Call):
        return False
    if _call_name(node) == "unload_chat_for_media":
        return True
    owner = getattr(getattr(node.func, "value", None), "id", None)
    return (owner, _call_name(node)) in (("_backend", "generate"), ("native", "upscale"))


def _is_restore_block(node):
    return isinstance(node, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and _call_name(i.context_expr)
        == "restore_chat_after_media" for i in node.items)


def test_each_job_body_calls_the_backend_inside_the_restore_block():
    root = Path(vram.__file__).resolve().parent / "plugins" / "builtin"
    bodies = 0
    for kind in _PLUG_FILES:
        tree = ast.parse((root / kind / "plug.py").read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not (isinstance(fn, ast.FunctionDef) and fn.name in ("_generate", "_upscale")):
                continue
            guarded = {id(c) for w in ast.walk(fn) if _is_restore_block(w)
                       for c in ast.walk(w) if isinstance(c, ast.Call)}
            calls = [c for c in ast.walk(fn) if _is_backend_call(c)]
            assert calls, (kind, fn.name)
            unguarded = [c.lineno for c in calls if id(c) not in guarded]
            assert not unguarded, (kind, fn.name, unguarded)
            assert not [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                        and _call_name(c) == "reload_chat_after_media"], (kind, fn.name)
            bodies += 1
    assert bodies == 4
