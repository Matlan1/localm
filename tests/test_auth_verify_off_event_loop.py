# SPDX-License-Identifier: AGPL-3.0-or-later
"""``auth.verify()`` must not be called on the event loop.

``verify()`` reads the keystore and, on a successful match, takes the keystore's
CROSS-PROCESS lock to stamp ``last_used`` (and to mark a legacy row's hash
algorithm). That lock waits up to ``config._CROSS_LOCK_TIMEOUT`` seconds for
another localm process, so a ``verify()`` call made directly from an ``async
def`` route body freezes the whole server for that long: every other request,
every SSE stream, every in-flight generation.

Routes that gate through a SYNC dependency (``require_scope``) are already safe -
FastAPI runs a sync dependency in a worker thread. The three routes here call
``verify()`` from their own async bodies and so must hand it to a thread
themselves.

Each test drives the request and a trivial concurrent coroutine together: if the
trivial one cannot get a turn while ``verify()`` is in flight, the loop is
blocked.
"""

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localm import auth, scopes
from localm.inference.http_server import create_app

SECRET = "test-off-loop-owner-key-xyz"
BLOCK_S = 2.0


def _engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5
    state = {"loaded": True}
    engine.unload.side_effect = lambda: state.update(loaded=False)
    engine.load.side_effect = lambda: state.update(loaded=True)
    type(engine).loaded = property(lambda self: state["loaded"])
    return engine


def _slow_verify(result):
    """A ``verify`` stand-in that takes BLOCK_S and records the thread it ran on."""
    calls = []

    def fake(presented):
        on_loop = True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop = False
        calls.append(on_loop)
        time.sleep(BLOCK_S)
        return result

    return fake, calls


async def _request_with_a_concurrent_coroutine(app, method, path, **kw):
    """Issue one request while a trivial coroutine tries to run. Returns
    ``(seconds the trivial coroutine waited, response)``; raises AssertionError
    when it never got a turn."""
    ticks = []

    async def trivial():
        for _ in range(3):
            await asyncio.sleep(0)
        ticks.append(time.monotonic())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        t0 = time.monotonic()
        spinner = asyncio.ensure_future(trivial())
        main = asyncio.ensure_future(client.request(method, path, **kw))
        try:
            await asyncio.wait_for(asyncio.shield(spinner), timeout=BLOCK_S * 0.5)
        except TimeoutError as e:
            main.cancel()
            try:
                await main
            except BaseException:
                pass
            raise AssertionError(
                f"a trivial coroutine never got a turn within {BLOCK_S * 0.5:.1f}s "
                f"while {method} {path} was verifying a key: auth.verify() ran on "
                "the event loop, so a contended keystore lock freezes the whole "
                "server") from e
        waited = ticks[0] - t0
        resp = await asyncio.wait_for(main, timeout=BLOCK_S + 30)
    return waited, resp


@pytest.fixture()
def keyed_app():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        yield create_app(_engine())


def test_session_login_verifies_off_the_event_loop(keyed_app):
    fake, calls = _slow_verify({scopes.ADMIN})
    with patch.object(auth, "verify", fake):
        waited, resp = asyncio.run(_request_with_a_concurrent_coroutine(
            keyed_app, "POST", "/api/session", json={"key": SECRET}))
    assert calls == [False], (
        "auth.verify was not called, or was called on the event loop, from "
        f"POST /api/session (on_loop flags: {calls})")
    assert waited < BLOCK_S * 0.5
    assert resp.status_code == 200


def test_surface_mount_verifies_off_the_event_loop(keyed_app):
    # verify -> None, so the route answers 403 without mounting anything; the
    # point is only where the call ran.
    fake, calls = _slow_verify(None)
    with patch.object(auth, "verify", fake):
        waited, resp = asyncio.run(_request_with_a_concurrent_coroutine(
            keyed_app, "POST", "/v1/surfaces/gui",
            headers={"Authorization": "Bearer not-the-attach-token"}))
    assert calls == [False], (
        "auth.verify was not called, or was called on the event loop, from "
        f"POST /v1/surfaces/gui (on_loop flags: {calls})")
    assert waited < BLOCK_S * 0.5
    assert resp.status_code == 403


def test_pairing_qr_for_a_minted_key_verifies_off_the_event_loop():
    from fastapi import FastAPI

    from localm.plugins.gui.routes import pairing as pairing_routes

    app = FastAPI()
    pairing_routes.register(app, MagicMock())
    endpoint = next(r.endpoint for r in app.routes
                    if getattr(r, "path", None) == "/api/pairing/qr"
                    and "POST" in r.methods)
    fake, calls = _slow_verify(None)

    async def drive():
        ticks = []

        async def trivial():
            for _ in range(3):
                await asyncio.sleep(0)
            ticks.append(time.monotonic())

        t0 = time.monotonic()
        spinner = asyncio.ensure_future(trivial())
        main = asyncio.ensure_future(endpoint(body={"key": "a-minted-key"}))
        try:
            await asyncio.wait_for(asyncio.shield(spinner), timeout=BLOCK_S * 0.5)
        except TimeoutError as e:
            main.cancel()
            try:
                await main
            except BaseException:
                pass
            raise AssertionError(
                "a trivial coroutine never got a turn while POST /api/pairing/qr "
                "was verifying a key: auth.verify() ran on the event loop") from e
        waited = ticks[0] - t0
        with pytest.raises(Exception) as excinfo:
            await asyncio.wait_for(main, timeout=BLOCK_S + 30)
        return waited, excinfo.value

    with patch.object(auth, "verify", fake):
        waited, err = asyncio.run(drive())
    assert calls == [False], (
        "auth.verify was not called, or was called on the event loop, from "
        f"POST /api/pairing/qr (on_loop flags: {calls})")
    assert waited < BLOCK_S * 0.5
    assert getattr(err, "status_code", None) == 400


def test_a_scope_gated_route_also_verifies_off_the_event_loop(keyed_app):
    """The control: a route gated by the sync ``require_scope`` dependency was
    already safe, so the probe above must report the same answer for it."""
    fake, calls = _slow_verify({scopes.ADMIN})
    with patch.object(auth, "verify", fake):
        waited, resp = asyncio.run(_request_with_a_concurrent_coroutine(
            keyed_app, "GET", "/v1/models",
            headers={"Authorization": f"Bearer {SECRET}"}))
    assert calls and not any(calls), f"on_loop flags: {calls}"
    assert waited < BLOCK_S * 0.5
    assert resp.status_code == 200


def _slow_recorder(result):
    """A stand-in taking BLOCK_S that records whether it ran on the event loop."""
    calls = []

    def fake(*args, **kwargs):
        on_loop = True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop = False
        calls.append(on_loop)
        time.sleep(BLOCK_S)
        return result

    return fake, calls


def test_session_login_derives_the_key_identity_off_the_event_loop(keyed_app):
    fake_hash, calls = _slow_recorder("a" * 64)
    with patch.object(auth, "verify", lambda presented: {scopes.ADMIN}), \
            patch.object(auth, "_hash_key", fake_hash):
        waited, resp = asyncio.run(_request_with_a_concurrent_coroutine(
            keyed_app, "POST", "/api/session", json={"key": SECRET}))
    assert calls == [False], (
        "auth._hash_key was not called, or was called on the event loop, from "
        f"POST /api/session (on_loop flags: {calls})")
    assert waited < BLOCK_S * 0.5
    assert resp.status_code == 200


def test_session_state_resolves_the_caller_off_the_event_loop(keyed_app):
    import localm.inference.http_server as hs
    fake, calls = _slow_recorder(({scopes.ADMIN}, "a" * 64, "host", []))
    with patch.object(hs, "_principal_from_token", fake):
        waited, resp = asyncio.run(_request_with_a_concurrent_coroutine(
            keyed_app, "GET", "/api/session",
            headers={"Authorization": f"Bearer {SECRET}"}))
    assert calls and not any(calls), (
        "_principal_from_token was not called, or was called on the event loop, "
        f"from GET /api/session (on_loop flags: {calls})")
    assert waited < BLOCK_S * 0.5
    assert resp.status_code == 200
    assert resp.json()["authed"] is True
