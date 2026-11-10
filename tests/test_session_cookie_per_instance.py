# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI session cookie name carries the port, so two instances on one host keep
separate sessions (browsers share cookies across ports)."""

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from localm.inference.http_server import SESSION_COOKIE, create_app, session_cookie_name

SECRET = "test-secret-per-instance-cookie"


def _app():
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    return create_app(engine)


def _cookie_names(resp):
    return [c.split("=", 1)[0] for c in resp.headers.get_list("set-cookie")]


def test_two_ports_get_two_cookie_names():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        with TestClient(_app(), base_url="http://127.0.0.1:8791") as a, \
                TestClient(_app(), base_url="http://127.0.0.1:8792") as b:
            ra = a.post("/api/session", json={"key": SECRET})
            rb = b.post("/api/session", json={"key": SECRET})
    assert f"{SESSION_COOKIE}_8791" in _cookie_names(ra)
    assert f"{SESSION_COOKIE}_8792" in _cookie_names(rb)
    assert SESSION_COOKIE not in _cookie_names(ra)


def test_instance_cookie_authenticates_and_other_port_cookie_does_not():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        with TestClient(_app(), base_url="http://127.0.0.1:8791") as a:
            sid = a.post("/api/session", json={"key": SECRET}).cookies.get(f"{SESSION_COOKIE}_8791")
            assert sid
            a.cookies.clear()
            a.cookies.set(f"{SESSION_COOKIE}_8791", sid)
            assert a.get("/v1/config").status_code == 200
            a.cookies.clear()
            a.cookies.set(f"{SESSION_COOKIE}_8792", sid)
            assert a.get("/v1/config").status_code == 401


def test_legacy_cookie_name_still_read():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        with TestClient(_app(), base_url="http://127.0.0.1:8791") as a:
            sid = a.post("/api/session", json={"key": SECRET}).cookies.get(f"{SESSION_COOKIE}_8791")
            a.cookies.clear()
            a.cookies.set(SESSION_COOKIE, sid)
            assert a.get("/v1/config").status_code == 200


def test_no_explicit_port_uses_legacy_name():
    req = MagicMock()
    req.url.port = None
    assert session_cookie_name(req) == SESSION_COOKIE
