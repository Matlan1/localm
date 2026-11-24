# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI session cookie is named by the install: a change of port keeps the login,
two installs on one host keep separate ones (browsers share cookies across ports),
and a cookie an older build set keeps working. One browser has one login per
install: no other session it presents outlives a login, a logout, a revoke or an
expiry."""

import os
import re
import time
from http.cookies import SimpleCookie
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import localm.config as cfg
import localm.inference.http_server as hs
from localm import auth, sessions
from localm.inference.http_server import (
    CSRF_HEADER, SESSION_COOKIE, create_app, session_cookie_name)

SECRET = "test-secret-per-instance-cookie"
HTTP = "http://127.0.0.1:8791"
HTTPS = "https://127.0.0.1:8791"


def _app():
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    return create_app(engine)


class Browser:
    """A browser with one cookie jar for every port and scheme of the host: a
    Secure cookie is sent over https only, and an http answer neither replaces nor
    deletes one."""

    def __init__(self, client):
        self.c = client
        self.jar = {}
        self.secure = set()

    def req(self, method, path, base=HTTP, csrf=None, **kw):
        sent = {n: v for n, v in self.jar.items()
                if base.startswith("https") or n not in self.secure}
        headers = dict(kw.pop("headers", {}))
        if sent:
            headers["Cookie"] = "; ".join(f"{n}={v}" for n, v in sent.items())
        if csrf:
            headers[CSRF_HEADER] = csrf
        self.c.cookies.clear()
        r = self.c.request(method, base + path, headers=headers, **kw)
        self.c.cookies.clear()
        for line in r.headers.get_list("set-cookie"):
            parsed = SimpleCookie()
            parsed.load(line)
            for name, m in parsed.items():
                if name in self.secure and not base.startswith("https"):
                    continue
                if m["max-age"] == "0" or not m.value:
                    self.jar.pop(name, None)
                    self.secure.discard(name)
                else:
                    self.jar[name] = m.value
                    if m["secure"]:
                        self.secure.add(name)
        return r

    def login(self, key=SECRET, base=HTTP):
        r = self.req("POST", "/api/session", base, json={"key": key})
        assert r.status_code == 200, r.text
        return r

    def state(self, base=HTTP):
        return self.req("GET", "/api/session", base).json()

    def scopes(self, base=HTTP):
        return self.state(base)["scopes"]

    def privileged(self, base=HTTP):
        """The status of two calls only the owner may make, sent with the CSRF
        token the server hands this browser."""
        csrf = self.state(base)["csrf"]
        mint = self.req("POST", "/v1/keys", base, csrf=csrf,
                        json={"name": "n", "scopes": ["admin"]})
        return [self.req("GET", "/v1/keys", base).status_code, mint.status_code]

    def logout(self, base=HTTP):
        return self.req("POST", "/api/session/logout", base,
                        csrf=self.state(base)["csrf"])


@pytest.fixture(autouse=True)
def _keyed():
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        yield


@pytest.fixture
def client():
    with TestClient(_app()) as c:
        yield c


@pytest.fixture
def browser(client):
    return Browser(client)


@pytest.fixture
def limited():
    return auth.create_key("phone", ["chat"])


@pytest.fixture
def folder(tmp_path):
    """Points the process at another data folder, and back with None."""
    home = cfg.HOME_DIR
    before = os.environ.get("LOCALM_HOME")

    def switch(name, instance_id=None):
        target = tmp_path / name if name else home
        target.mkdir(parents=True, exist_ok=True)
        if instance_id:
            (target / "instance_id.txt").write_text(instance_id, encoding="utf-8")
        cfg.HOME_DIR = target
        if name:
            os.environ["LOCALM_HOME"] = str(target)
        elif before is None:
            os.environ.pop("LOCALM_HOME", None)
        else:
            os.environ["LOCALM_HOME"] = before
        sessions._CACHE["mtime"] = None
        sessions._CACHE["records"] = None

    yield switch
    switch(None)


def _own(base=HTTP):
    return session_cookie_name(MagicMock(url=MagicMock(scheme=base.split(":")[0])))


def _owner_session(**kw):
    return sessions.create(scopes={"admin"}, key_hash=auth._hash_key(SECRET),
                           fs_access="host", owner_key_minted=True, **kw)


def _limited_session(key, **kw):
    return sessions.create(scopes={"chat"}, key_hash=auth._hash_key(key), **kw)


def _signed_out(browser, base=HTTP):
    assert browser.state(base)["authed"] is False
    assert browser.scopes(base) == []
    assert browser.privileged(base) == [401, 401]


def _is_limited(browser, base=HTTP):
    assert browser.scopes(base) == ["chat"]
    assert browser.privileged(base) == [403, 403]


def _is_owner(browser, base=HTTP):
    assert browser.scopes(base) == ["admin"]
    assert browser.privileged(base) == [200, 200]


# --------------------------------------------------------------------------- #
#  The name                                                                    #
# --------------------------------------------------------------------------- #

def test_the_name_is_the_same_on_every_port_and_hides_the_id(browser):
    for base in (HTTP, "http://127.0.0.1:9999", "http://127.0.0.1"):
        browser.login(base=base)
        assert list(browser.jar) == [_own()]
    _is_owner(browser, "http://127.0.0.1:5555")
    assert re.fullmatch(SESSION_COOKIE + r"_[0-9a-f]{12}", _own())
    assert cfg.instance_id() not in _own()


def test_another_install_uses_another_name_and_its_cookie_is_no_credential(
        browser, folder):
    browser.login()
    a_jar = dict(browser.jar)
    folder("other")
    assert _own() not in a_jar
    _signed_out(browser)
    browser.login()
    assert a_jar.items() < browser.jar.items()
    _is_owner(browser)
    folder(None)
    _is_owner(browser)


def test_a_copied_data_folder_with_the_same_install_id_uses_another_name(browser, folder):
    folder("a", instance_id="copied-id")
    browser.login()
    name_a, jar = _own(), dict(browser.jar)
    rows = (cfg.HOME_DIR / "sessions.json").read_text(encoding="utf-8")
    folder("b", instance_id="copied-id")
    (cfg.HOME_DIR / "sessions.json").write_text(rows, encoding="utf-8")
    sessions._CACHE["mtime"] = None
    assert _own() != name_a
    _signed_out(browser)
    browser.login()
    browser.logout()
    assert browser.jar == jar
    folder("a")
    _is_owner(browser)


def test_the_mount_under_a_path_is_part_of_the_folders_identity(tmp_path):
    info = tmp_path / "mountinfo"
    info.write_text(
        "1 0 0:30 / / rw - overlay overlay rw\n"
        "2 1 8:1 /docker/volumes/one/_data /data rw master:1 - ext4 /dev/sda1 rw\n"
        "3 1 8:1 /docker/volumes/two/_data /data/nested rw - ext4 /dev/sda1 rw\n"
        "broken line\n", encoding="utf-8")
    read = lambda p: hs._mount_id(p, str(info))  # noqa: E731
    assert read("/data/home") == "/docker/volumes/one/_data|ext4"
    assert read("/data/nested/x") == "/docker/volumes/two/_data|ext4"
    assert read("/database").startswith("/|overlay|")
    assert hs._mount_id("/x", str(tmp_path / "absent")) == ""


def test_https_uses_a_secure_cookie_of_its_own_name(browser):
    r = browser.login(base=HTTPS)
    [header] = r.headers.get_list("set-cookie")
    assert header.startswith(_own() + "_tls=")
    low = header.lower()
    assert "secure" in low and "httponly" in low and "samesite=strict" in low
    _is_owner(browser, HTTPS)


# --------------------------------------------------------------------------- #
#  One login per browser and install                                           #
# --------------------------------------------------------------------------- #

def test_a_limited_login_beside_an_old_owner_cookie_never_turns_owner(browser, limited):
    browser.jar["localm_session_8791"] = old = _owner_session()
    _is_owner(browser)
    browser.login(limited["key"])
    assert list(browser.jar) == [_own()]
    assert sessions.lookup(old) is None
    _is_limited(browser)
    assert auth.revoke_key(limited["id"])
    _signed_out(browser)


def test_a_limited_login_before_the_old_owner_cookie_was_moved(browser, limited):
    browser.jar["localm_session_8791"] = old = _owner_session()
    browser.login(limited["key"])
    assert list(browser.jar) == [_own()] and sessions.lookup(old) is None
    _is_limited(browser)
    assert auth.revoke_key(limited["id"])
    browser.jar["localm_session_8791"] = old
    _signed_out(browser)


def test_a_limited_https_login_after_an_owner_http_login_never_turns_owner(
        browser, limited):
    browser.login(base=HTTP)
    _is_owner(browser, HTTPS)
    browser.login(base=HTTP)
    browser.login(limited["key"], base=HTTPS)
    _is_limited(browser, HTTPS)
    assert auth.revoke_key(limited["id"])
    _signed_out(browser, HTTPS)
    _signed_out(browser, HTTP)


def test_the_same_key_over_the_other_scheme_stays_signed_in(browser):
    browser.login(base=HTTP)
    browser.login(base=HTTPS)
    _is_owner(browser, HTTP)
    _is_owner(browser, HTTPS)


@pytest.mark.parametrize("legacy", [SESSION_COOKIE, f"{SESSION_COOKIE}_8791",
                                    f"{SESSION_COOKIE}_7777"])
def test_owner_then_limited(browser, limited, legacy):
    browser.login()
    browser.jar[legacy] = extra = _owner_session()
    owner_sid = browser.jar[_own()]
    browser.login(limited["key"])
    _is_limited(browser)
    assert sessions.lookup(owner_sid) is None and sessions.lookup(extra) is None
    assert list(browser.jar) == [_own()]


def test_limited_then_owner(browser, limited):
    browser.login(limited["key"])
    limited_sid = browser.jar[_own()]
    browser.login()
    _is_owner(browser)
    assert sessions.lookup(limited_sid) is None
    auth.revoke_key(limited["id"])
    _is_owner(browser)


@pytest.mark.parametrize("how", ["logout", "revoke", "expiry", "idle", "clear-row"])
def test_a_limited_login_that_ends_leaves_nothing(browser, how, monkeypatch):
    key = auth.create_key("p", ["chat"], expires=time.time() + 3600)
    browser.jar["localm_session_8791"] = _owner_session()
    browser.login()
    browser.login(key["key"])
    _is_limited(browser)
    sid = browser.jar[_own()]
    if how == "logout":
        assert browser.logout().json() == {"authed": False, "warnings": []}
        assert browser.jar == {}
    elif how == "revoke":
        auth.revoke_key(key["id"])
    elif how == "expiry":
        rows = auth._load_keystore()
        for r in rows:
            r["expires"] = time.time() - 1
        auth._save_keystore(rows)
    elif how == "idle":
        rows = sessions._load_for_update()
        for r in rows:
            r["last_used"] = r["issued"] = time.time() - 40 * 86400
        sessions._save(rows)
    else:
        sessions.revoke(sid)
    _signed_out(browser)
    browser.jar["localm_session"] = "junk"
    browser.jar[_own()] = sid
    _signed_out(browser)


def test_a_dead_own_cookie_never_falls_through_to_another_live_one(browser):
    for name in (SESSION_COOKIE, f"{SESSION_COOKIE}_8791", _own() + "_tls"):
        browser.jar = {_own(): "ended", name: _owner_session()}
        _signed_out(browser)


def test_every_login_path_ends_the_other_sessions_of_the_browser(tmp_path, monkeypatch):
    from localm.plugins.gui.web import attach_gui, mint_launch_grant
    app = _app()

    async def switch(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch,
               active_model=lambda: "m")
    with TestClient(app) as c:
        for path in ("login", "grant", "rotate", "first-key"):
            b = Browser(c)
            stale = _limited_session("gone") if path != "login" else _owner_session()
            b.jar[SESSION_COOKIE] = stale
            if path == "login":
                b.login()
            elif path == "grant":
                grant = mint_launch_grant(app)
                assert b.req("GET", f"/?localm_token={grant}",
                             follow_redirects=False).status_code == 303
            else:
                monkeypatch.delenv("LOCALM_API_KEY", raising=False)
                auth.clear_api_key()
                shell = {"Authorization": f"Bearer {app.state.shell_token}"}
                r = (b.req("POST", "/api/auth/key/rotate", headers=shell)
                     if path == "rotate" else
                     b.req("POST", "/v1/keys", headers=shell,
                           json={"name": "n", "scopes": ["chat"]}))
                assert r.status_code == 200, r.text
            assert list(b.jar) == [_own()], path
            assert sessions.lookup(stale) is None, path
            assert b.scopes() == ["admin"], path


# --------------------------------------------------------------------------- #
#  Cookies of older builds                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("old", [SESSION_COOKIE, f"{SESSION_COOKIE}_8791",
                                 f"{SESSION_COOKIE}_7777"])
def test_a_cookie_of_an_older_build_keeps_the_login_and_moves_to_the_new_name(
        browser, old):
    browser.jar[old] = sid = _owner_session()
    assert browser.req("GET", "/v1/config").status_code == 200
    assert browser.jar == {old: sid}
    r = browser.req("GET", "/api/session")
    assert r.json()["authed"] is True and r.json()["csrf"]
    assert browser.jar == {_own(): sid}
    assert browser.req("GET", "/api/session").headers.get_list("set-cookie") == []
    _is_owner(browser)
    assert browser.req("POST", "/v1/keys",
                       json={"name": "n", "scopes": ["chat"]}).status_code == 403


def test_of_several_older_cookies_the_newest_counts_and_the_others_end(browser, limited):
    old, new = _owner_session(), _limited_session(limited["key"])
    browser.jar = {f"{SESSION_COOKIE}_8791": old, f"{SESSION_COOKIE}_7777": new}
    assert browser.req("GET", "/v1/keys").status_code == 403
    assert sessions.lookup(old) is None
    _is_limited(browser)
    auth.revoke_key(limited["id"])
    _signed_out(browser)

    b = Browser(browser.c)
    first, second = _limited_session(auth.create_key("q", ["chat"])["key"]), _owner_session()
    b.jar = {SESSION_COOKIE: first, f"{SESSION_COOKIE}_8791": second}
    assert b.state()["authed"] is True
    assert b.jar == {_own(): second} and sessions.lookup(first) is None


def test_older_cookies_sign_nobody_in_when_the_others_cannot_be_ended(browser, monkeypatch):
    browser.jar = {SESSION_COOKIE: _owner_session(), f"{SESSION_COOKIE}_7777": _owner_session()}
    monkeypatch.setattr(sessions, "revoke_many", lambda sids: None)
    _signed_out(browser)


def test_the_cookie_the_older_build_used_for_this_port_must_be_live(browser, limited):
    revoked = _limited_session(limited["key"])
    auth.revoke_key(limited["id"])
    for jar in ({f"{SESSION_COOKIE}_8791": revoked, SESSION_COOKIE: _owner_session()},
                {f"{SESSION_COOKIE}_8791": "x", f"{SESSION_COOKIE}_7777": _owner_session()},
                {SESSION_COOKIE: "x", f"{SESSION_COOKIE}_7777": _owner_session()}):
        browser.jar = dict(jar)
        _signed_out(browser)


def test_a_session_whose_key_expired_does_not_hide_a_live_one(browser):
    k = auth.create_key("p", ["chat"], expires=time.time() + 3600)
    expired, live = _limited_session(k["key"]), _owner_session()
    rows = auth._load_keystore()
    rows[0]["expires"] = time.time() - 1
    auth._save_keystore(rows)
    browser.jar = {f"{SESSION_COOKIE}_7777": expired, f"{SESSION_COOKIE}_6666": live}
    _is_owner(browser)


def test_the_number_of_older_cookies_looked_up_is_bounded(monkeypatch):
    seen = []
    monkeypatch.setattr(hs, "_valid_session", lambda v: seen.append(v))
    req = MagicMock(cookies={f"{SESSION_COOKIE}_{i}": f"junk{i}" for i in range(200)},
                    url=MagicMock(scheme="http", port=8791))
    hs.session_cookie_value(req)
    assert len(seen) == hs._MAX_OLDER_SESSION_COOKIES


# --------------------------------------------------------------------------- #
#  Signing out                                                                 #
# --------------------------------------------------------------------------- #

def test_logging_out_of_one_install_leaves_the_other_installs_cookie(browser, folder):
    browser.login()
    jar_a = dict(browser.jar)
    folder("other")
    browser.login()
    assert browser.logout().status_code == 200
    assert browser.jar == jar_a
    folder(None)
    _is_owner(browser)


def test_logout_ends_every_session_of_this_install(browser):
    browser.login()
    browser.login(base=HTTPS)
    browser.jar[SESSION_COOKIE] = old = _owner_session()
    sids = list(browser.jar.values())
    assert browser.logout(HTTPS).status_code == 200
    assert browser.jar == {}
    assert old in sids and not any(sessions.lookup(s) for s in sids)


def test_logout_with_a_live_session_needs_the_csrf_token(browser):
    browser.login()
    sid = browser.jar[_own()]
    assert browser.req("POST", "/api/session/logout").status_code == 403
    assert sessions.lookup(sid) is not None


def test_clearing_the_owner_key_removes_this_installs_cookies(browser):
    browser.login()
    browser.jar[SESSION_COOKIE] = _owner_session()
    csrf = browser.state()["csrf"]
    assert browser.req("POST", "/api/auth/key/clear", csrf=csrf).status_code == 200
    assert browser.jar == {}


# --------------------------------------------------------------------------- #
#  Only session ids are ever set                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bearer", [False, True])
def test_an_api_key_is_never_set_as_a_cookie(client, bearer):
    scoped = auth.create_key("k", ["chat"])["key"]
    for key in (SECRET, scoped):
        headers = {"Authorization": f"Bearer {key}"} if bearer else {}
        for jar in ({SESSION_COOKIE: key}, {f"{SESSION_COOKIE}_8791": key}, {_own(): key},
                    {SESSION_COOKIE: _owner_session(), f"{SESSION_COOKIE}_1": key},
                    {f"{SESSION_COOKIE}_1": key, SESSION_COOKIE: _owner_session()}):
            for method, path in (("GET", "/api/session"), ("GET", "/"),
                                 ("POST", "/api/session/logout")):
                b = Browser(client)
                b.jar = dict(jar)
                r = b.req(method, path, headers=headers)
                assert key not in " ".join(r.headers.get_list("set-cookie"))
            if not bearer and list(jar.values()) == [key]:
                b = Browser(client)
                b.jar = dict(jar)
                assert b.state()["authed"] is False
