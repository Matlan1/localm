# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI session cookie is named by the install's id: a change of port keeps the
login, two installs on one host keep separate ones (browsers share cookies across
ports), and a cookie an older build set keeps working."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import localm.config as cfg
from localm import sessions
from localm.inference.http_server import (
    CSRF_HEADER, SESSION_COOKIE, create_app, session_cookie_name)

SECRET = "test-secret-per-instance-cookie"


def _app():
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    return create_app(engine)


def _client(base_url="http://127.0.0.1:8791"):
    return TestClient(_app(), base_url=base_url)


def _set_cookie_names(resp):
    return [c.split("=", 1)[0] for c in resp.headers.get_list("set-cookie")]


def _login(c):
    r = c.post("/api/session", json={"key": SECRET})
    assert r.status_code == 200, r.text
    [name] = _set_cookie_names(r)
    return name, r.cookies.get(name)


def _with_cookies(c, jar):
    c.cookies.clear()
    for n, v in jar.items():
        c.cookies.set(n, v)
    return c


@pytest.fixture(autouse=True)
def _keyed():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        yield


@pytest.fixture
def other_install(tmp_path):
    """Switches the process to a second install's data folder and back."""
    home = cfg.HOME_DIR

    def switch(on):
        target = tmp_path / "other" if on else home
        target.mkdir(parents=True, exist_ok=True)
        cfg.HOME_DIR = target
        os.environ["LOCALM_HOME"] = str(target)
        sessions._CACHE["mtime"] = None
        sessions._CACHE["records"] = None

    before = os.environ.get("LOCALM_HOME")
    yield switch
    switch(False)
    if before is None:
        os.environ.pop("LOCALM_HOME", None)
    else:
        os.environ["LOCALM_HOME"] = before


def test_the_name_is_the_same_on_every_port_and_hides_the_id():
    with _client("http://127.0.0.1:8791") as a:
        name_a, sid = _login(a)
    with _client("http://127.0.0.1:8792") as b:
        name_b, _ = _login(b)
        assert name_a == name_b
        _with_cookies(b, {name_a: sid})
        assert b.get("/v1/config").status_code == 200
    with _client("http://127.0.0.1") as c:
        assert _login(c)[0] == name_a
    assert name_a.startswith(SESSION_COOKIE + "_")
    assert cfg.instance_id() not in name_a


def test_another_install_uses_another_name_and_its_cookie_is_no_credential(other_install):
    with _client() as a:
        name_a, sid_a = _login(a)
        other_install(True)
        name_b, sid_b = _login(a)
        assert name_b != name_a
        _with_cookies(a, {name_a: sid_a})
        assert a.get("/v1/config").status_code == 401
        jar = {name_a: sid_a, name_b: sid_b}
        _with_cookies(a, jar)
        assert a.get("/v1/config").status_code == 200
        other_install(False)
        _with_cookies(a, jar)
        assert a.get("/v1/config").status_code == 200


def test_logging_out_of_one_install_leaves_the_other_installs_cookie(other_install):
    with _client() as a:
        name_a, sid_a = _login(a)
        other_install(True)
        name_b, sid_b = _login(a)
        _with_cookies(a, {name_a: sid_a, name_b: sid_b, SESSION_COOKIE: sid_a})
        csrf = a.get("/api/session").json()["csrf"]
        r = a.post("/api/session/logout", headers={CSRF_HEADER: csrf})
        assert r.status_code == 200
        assert _set_cookie_names(r) == [name_b]
        assert sessions.lookup(sid_b) is None
        other_install(False)
        assert sessions.lookup(sid_a) is not None


@pytest.mark.parametrize("old", [SESSION_COOKIE, f"{SESSION_COOKIE}_8791",
                                 f"{SESSION_COOKIE}_7777"])
def test_a_cookie_of_an_older_build_keeps_the_login_and_gets_the_new_name(old):
    with _client() as a:
        name, sid = _login(a)
        _with_cookies(a, {old: sid})
        assert a.get("/v1/config").status_code == 200
        r = a.get("/api/session")
        assert r.json()["authed"] is True and r.json()["csrf"]
        assert _set_cookie_names(r) == [name]
        assert r.cookies.get(name) == sid
        _with_cookies(a, {old: sid, name: sid})
        assert _set_cookie_names(a.get("/api/session")) == []
        r = a.post("/v1/keys", json={"name": "n", "scopes": ["chat"]},
                   headers={CSRF_HEADER: a.get("/api/session").json()["csrf"]})
        assert r.status_code == 200, r.text
        assert a.post("/v1/keys", json={"name": "n", "scopes": ["chat"]}).status_code == 403


def test_a_stale_cookie_does_not_hide_a_live_one():
    with _client() as a:
        name, sid = _login(a)
        _with_cookies(a, {f"{SESSION_COOKIE}_8791": "someone-elses", SESSION_COOKIE: sid})
        assert a.get("/v1/config").status_code == 200
        _with_cookies(a, {name: "ended", SESSION_COOKIE: sid})
        assert a.get("/v1/config").status_code == 200


def test_logout_removes_the_old_name_too_and_ends_both_sessions():
    with _client() as a:
        name, sid = _login(a)
        _, sid_old = _login(a)
        _with_cookies(a, {name: sid, SESSION_COOKIE: sid_old})
        csrf = a.get("/api/session").json()["csrf"]
        r = a.post("/api/session/logout", headers={CSRF_HEADER: csrf})
        assert sorted(_set_cookie_names(r)) == sorted([name, SESSION_COOKIE])
        assert sessions.lookup(sid) is None and sessions.lookup(sid_old) is None


def test_logout_with_a_live_session_still_needs_the_csrf_token():
    with _client() as a:
        name, sid = _login(a)
        _with_cookies(a, {name: sid})
        assert a.post("/api/session/logout").status_code == 403
        assert sessions.lookup(sid) is not None


def test_https_uses_a_secure_cookie_of_its_own_name():
    with _client() as a:
        plain, _ = _login(a)
    with _client("https://127.0.0.1:8791") as b:
        r = b.post("/api/session", json={"key": SECRET})
        [header] = r.headers.get_list("set-cookie")
        assert header.startswith(plain + "_tls=")
        low = header.lower()
        assert "secure" in low and "httponly" in low and "samesite=strict" in low
        assert b.get("/api/session").json()["authed"] is True


def test_only_so_many_session_cookies_are_looked_up(monkeypatch):
    import localm.inference.http_server as hs
    with _client() as a:
        name, sid = _login(a)
        seen = []
        real = sessions.lookup
        monkeypatch.setattr(sessions, "lookup", lambda s: seen.append(s) or real(s))
        req = MagicMock()
        req.url.scheme = "http"
        req.cookies = {f"{SESSION_COOKIE}_{i}": f"junk{i}" for i in range(200)}
        req.cookies[name] = sid
        assert hs.session_cookie_value(req) == sid
        assert seen[0] == sid and len(seen) == hs._MAX_SESSION_COOKIES
        assert session_cookie_name(req) == name
