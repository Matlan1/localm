# SPDX-License-Identifier: AGPL-3.0-or-later
"""Revoking a key ends what runs under it: the coder sessions it created and
the streaming responses it opened, not only its next request."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from localm import auth, scopes
from localm.plugins.coder.sessions import SessionManager


class _Session:
    def __init__(self, sid, key_hash):
        self.id = sid
        self.key_hash = key_hash
        self.principal = key_hash
        self.closed = False

    def close(self):
        self.closed = True


def test_a_revoke_tells_listeners_the_key_hash_and_moves_the_epoch(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret-ownersecret")
    created = auth.create_key("A", [scopes.CODER])
    seen = []
    auth.add_revoke_listener(seen.append)
    try:
        before = auth.revocation_epoch()
        assert auth.revoke_key(created["id"]) is True
        assert seen == [auth._hash_key(created["key"])]
        assert auth.revocation_epoch() > before
    finally:
        auth.remove_revoke_listener(seen.append)


def test_clearing_every_key_tells_listeners_none(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key("owner-key-owner-key-1234")
    auth.create_key("A", [scopes.CODER])
    seen = []
    auth.add_revoke_listener(seen.append)
    try:
        assert auth.clear_api_key() == []
        assert seen == [None]
    finally:
        auth.remove_revoke_listener(seen.append)


def test_setting_the_owner_key_moves_the_epoch_without_ending_sessions(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    seen = []
    auth.add_revoke_listener(seen.append)
    try:
        before = auth.revocation_epoch()
        auth.set_api_key("owner-key-owner-key-5678")
        assert auth.revocation_epoch() > before
        assert seen == []
    finally:
        auth.remove_revoke_listener(seen.append)


def test_revoking_a_key_ends_only_the_coder_sessions_it_created(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret-ownersecret")
    a = auth.create_key("A", [scopes.CODER])
    b = auth.create_key("B", [scopes.CODER])
    mgr = SessionManager()
    sa = mgr.create(_Session("sa", auth._hash_key(a["key"])))
    sb = mgr.create(_Session("sb", auth._hash_key(b["key"])))
    owner = mgr.create(_Session("so", None))
    assert auth.revoke_key(a["id"]) is True
    assert sa.closed and mgr.get("sa") is None
    assert not sb.closed and mgr.get("sb") is sb
    assert not owner.closed and mgr.get("so") is owner
    mgr.close_all()


def test_a_key_removed_elsewhere_or_expired_ends_its_sessions(monkeypatch):
    monkeypatch.setattr("localm.plugins.coder.sessions._KEY_WATCH_SECONDS", 0.05)
    mgr = SessionManager()
    live = {"h1": True}
    monkeypatch.setattr(auth, "key_hash_live_checked", lambda h: live.get(h, False))
    s = mgr.create(_Session("s1", "h1"))
    time.sleep(0.3)
    assert not s.closed
    live["h1"] = False
    deadline = time.monotonic() + 5
    while not s.closed and time.monotonic() < deadline:
        time.sleep(0.05)
    assert s.closed and mgr.get("s1") is None


def test_a_revoked_key_s_open_stream_ends(tmp_path, monkeypatch):
    """Through the real app stack: a stream opened with a keystore key stops
    within a chunk of the revoke, and its generator is closed."""
    from unittest.mock import MagicMock

    from fastapi import Depends
    from fastapi.responses import StreamingResponse

    from localm.inference.http_server import _require_auth, create_app
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret-ownersecret")
    created = auth.create_key("S", [scopes.CHAT])
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)
    sent = []
    closed = threading.Event()

    async def _gen():
        try:
            for i in range(200):
                sent.append(i)
                yield f"data: {i}\n\n"
                if i == 2:
                    auth.revoke_key(created["id"])
        finally:
            closed.set()

    @app.get("/v1/test-stream", dependencies=[Depends(_require_auth)])
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(app) as c:
        r = c.get("/v1/test-stream",
                  headers={"Authorization": f"Bearer {created['key']}"})
        body = r.text
    assert r.status_code == 200
    assert "data: 2" in body
    assert "data: 5" not in body
    assert closed.wait(5)
    assert len(sent) < 10


def test_an_owner_stream_is_not_ended_by_a_revoke(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi import Depends
    from fastapi.responses import StreamingResponse

    from localm.inference.http_server import _require_auth, create_app
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret-ownersecret")
    other = auth.create_key("O", [scopes.CHAT])
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)

    async def _gen():
        for i in range(20):
            yield f"data: {i}\n\n"
            if i == 2:
                auth.revoke_key(other["id"])

    @app.get("/v1/test-stream", dependencies=[Depends(_require_auth)])
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(app) as c:
        r = c.get("/v1/test-stream",
                  headers={"Authorization": "Bearer ownersecret-ownersecret"})
    assert r.status_code == 200
    assert "data: 19" in r.text


def test_an_open_mode_stream_with_a_revoke_runs_to_its_end(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.responses import StreamingResponse

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)

    async def _gen():
        for i in range(50):
            yield f"data: {i}\n\n"
            if i == 2:
                auth._credentials_changed(None)

    @app.get("/v1/test-stream")
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(app) as c:
        r = c.get("/v1/test-stream",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200
    assert "data: 49" in r.text


def test_an_unreadable_keystore_ends_no_session(monkeypatch):
    mgr = SessionManager()
    s = mgr.create(_Session("s1", "h1"))
    monkeypatch.setattr(auth, "_load_keystore_checked", lambda: ([], False))
    assert mgr.end_sessions_of_dead_keys() == []
    assert not s.closed and mgr.get("s1") is s
    monkeypatch.setattr(auth, "_load_keystore_checked", lambda: ([], True))
    assert mgr.end_sessions_of_dead_keys() == ["s1"]


def test_an_open_mode_stream_survives_the_first_key(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.responses import StreamingResponse

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)

    async def _gen():
        for i in range(30):
            yield f"data: {i}\n\n"
            if i == 2:
                auth.set_api_key("owner-key-owner-key-9999")

    @app.get("/v1/test-stream")
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(app) as c:
        r = c.get("/v1/test-stream",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert "data: 29" in r.text


def test_an_open_mode_stream_ends_when_the_first_key_comes_before_its_first_chunk(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.responses import StreamingResponse

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)
    sent = []

    async def _gen():
        auth.set_api_key("owner-key-owner-key-9999")
        for i in range(200):
            sent.append(i)
            yield f"data: {i}\n\n"

    @app.get("/v1/test-stream")
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(app) as c:
        r = c.get("/v1/test-stream",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200
    assert "data: 0" in r.text
    assert "data: 5" not in r.text
    assert len(sent) < 10


def test_the_credential_recheck_reads_key_files_once_per_stream_and_never_on_the_loop(monkeypatch):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    from localm.inference.app_assembly.transport import CredentialRecheckMiddleware
    calls = []
    loop_threads = []
    monkeypatch.setattr(auth, "any_key_configured",
                        lambda: calls.append(threading.get_ident()) or True)
    inner = FastAPI()

    @inner.get("/plain")
    async def _plain():
        loop_threads.append(threading.get_ident())
        return JSONResponse({"ok": True})

    async def _gen():
        loop_threads.append(threading.get_ident())
        for i in range(20):
            yield f"data: {i}\n\n"

    @inner.get("/stream")
    async def _stream():
        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(CredentialRecheckMiddleware(inner)) as c:
        hdr = {"Authorization": "Bearer some-key-some-key"}
        assert c.get("/plain", headers=hdr).json() == {"ok": True}
        assert calls == []
        assert "data: 19" in c.get("/stream", headers=hdr).text
    assert len(calls) == 1
    assert calls[0] not in loop_threads
