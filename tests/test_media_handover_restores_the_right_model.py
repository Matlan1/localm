# SPDX-License-Identifier: AGPL-3.0-or-later
"""After a media VRAM handover, chat comes back to the model you were using.

THE SEQUENCE, which is the whole defect:

1. An image/music/video job needs VRAM, so ``vram.unload_chat_for_media`` POSTs
   ``/models/unload``. Every unload path parks the outgoing name in
   ``_last_active_model_name`` and sets ``_active_model_name`` to None - the
   Engine stays in ``_engines`` for a lazy reload, and that parked name is the
   ONLY record of what the user actually had loaded.
2. The job generates.
3. ``vram.reload_chat_after_media`` POSTs ``/models/load`` WITH NO NAME.

Step 3 resolves ``_active_model_name or _default_model_name``. Step 1 has just
set the first to None, so without the fix it falls through to
``_default_model_name`` - write-once at startup and never updated by a model
switch. A user who boots on model-a, switches to model-b, generates an image
and goes back to chat is then talking to model-a, with nothing anywhere saying
so.

FIXTURE PREMISE: the failing case needs a server that SWITCHED to a second
model, and it needs ``_last_active_model_name`` actually populated. A fixture
with one model, or one that leaves that field None, cannot express this defect
at all - both readings resolve to the same name and every assertion passes on
the broken code. Each test asserts those premises before asserting the
behaviour.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    """Stands in for the model backend only."""

    def __init__(self, display_name: str):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.load_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


def _auth(app) -> dict:
    """The bearer header the media handover itself sends."""
    return {"Authorization": f"Bearer {app.state.instance_token}"}


def _reset_server_state():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None


@pytest.fixture
def switched_server(monkeypatch):
    """Boots on model-a (startup/default), then switches to model-b - the
    state a user is in when they have picked a model and then ask for an
    image. Plenty of free VRAM, so the switch does not evict model-a and the
    two names stay genuinely distinct."""
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    monkeypatch.setattr(
        "localm.config.load_registry",
        lambda: {"model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
                 "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", factory)

    _reset_server_state()
    startup = factory("model-a")
    startup.load()
    app = hs.create_app(startup)

    # /models/load is management-scoped, and with no API key configured the
    # open-mode gate wants the per-process instance token. That is exactly the
    # credential the real caller presents (selfclient.self_request is handed
    # request.app.state.instance_token), so this authenticates the way
    # production does rather than disabling the gate for the test.
    app.state.instance_token = "test-instance-token-0123456789"

    asyncio.run(hs.switch_engine("model-b", hs._engine_factory, preempt=False))
    assert hs._active_model_name == "model-b", "test premise: the switch must succeed"
    assert hs._default_model_name == "model-a", (
        "test premise: the startup model must DIFFER from the switched-to one, "
        "or falling back to it would be indistinguishable from success")
    try:
        yield app, engines
    finally:
        _reset_server_state()


def test_an_unnamed_load_after_a_handover_restores_the_model_in_use(switched_server):
    app, engines = switched_server

    # Step 1: what unload_chat_for_media does to free VRAM for the media model.
    asyncio.run(hs.unload_all_models())

    # THE PREMISE THIS TEST LIVES OR DIES ON. Without _last_active_model_name
    # populated, the broken chain and the fixed one both return model-a and
    # this test is decorative.
    assert hs._active_model_name is None, (
        "test premise: the handover must have cleared the active pointer")
    assert hs._last_active_model_name == "model-b", (
        "test premise: the unload must have parked the in-use name - if this "
        "is None the test cannot express the defect at all")

    # Step 3: reload_chat_after_media POSTs with NO model name.
    with TestClient(app) as client:
        resp = client.post("/v1/models/load", headers=_auth(app))

    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "model-b", (
        f"chat came back on {resp.json()['model']!r} instead of model-b, the "
        "model that was actually in use before the media handover - the "
        "unnamed load fell through to the startup model")
    assert engines["model-b"].loaded, "the in-use model was never reloaded"


def test_an_explicitly_named_load_is_unaffected(switched_server):
    # The unnamed path is the only one that changed. A caller that names a
    # model must still get exactly that model, or the fix has widened into
    # something nobody asked for.
    app, engines = switched_server
    asyncio.run(hs.unload_all_models())

    with TestClient(app) as client:
        resp = client.post("/v1/models/load", params={"model": "model-a"},
                           headers=_auth(app))

    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "model-a"


def test_the_media_reload_really_does_send_no_model_name():
    """Pins the coupling that makes the route's unnamed resolution load-bearing.

    If ``reload_chat_after_media`` ever started naming the model, the route
    test above would keep passing while covering a path production no longer
    takes. This is the cheap guard against that drift.
    """
    import inspect

    from localm import vram

    src = inspect.getsource(vram.reload_chat_after_media)
    assert '"/models/load"' in src, (
        "reload_chat_after_media no longer posts to /models/load - re-check "
        "which path the handover now takes")
    assert "model=" not in src and '"model"' not in src, (
        "reload_chat_after_media now sends a model name, so the unnamed "
        "resolution this defect was about is no longer the path it uses")
