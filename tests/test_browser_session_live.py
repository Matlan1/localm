# SPDX-License-Identifier: AGPL-3.0-or-later
"""The automated browser driven for real, against a recording origin.

tests/test_browser_netgate.py pins the DECISION. This pins that the decision is
actually applied by a real Chromium: a refused destination is one the origin
server never sees a request for, read off that server's own log rather than
inferred from the gate returning a reason.

The subresource case is the one that matters most. A browser fetches far more
than the caller asked for, so gating only the top-level navigation would leave
every image, script and fetch on a loaded page ungoverned.

Skipped unless the browser extra is installed, so this does not run in CI. Run
it locally with the extra present.
"""

from __future__ import annotations

import http.server
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="the browser extra is not installed")

from localm.browser import session as bsession    # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    _cfg.ensure_dirs()
    return home


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):                                # noqa: N802
        type(self).seen.append(self.path)
        if self.path.endswith(".png"):
            body = b"\x89PNG\r\n\x1a\n"
            ctype = "image/png"
        elif self.path.startswith("/ws-page"):
            body = (b"<html><body>ws<script>try{new WebSocket("
                    b"'ws://127.0.0.1:%d/ws');}catch(e){}</script></body></html>"
                    % self.server.server_address[1])
            ctype = "text/html"
        else:
            body = (b"<html><body><h1>page</h1>"
                    b"<img src='http://127.0.0.1:%d/sub.png'>"
                    b"</body></html>" % self.server.server_address[1])
            ctype = "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                       # quiet
        pass


@pytest.fixture
def origin():
    """A loopback origin that RECORDS every path it is asked for."""
    _Recorder.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def browser(cfg_home):
    made = []

    def _make(**kw):
        s = bsession.BrowserSession("test-live", **kw)
        s.start()
        made.append(s)
        return s
    yield _make
    for s in made:
        s.stop()


def test_an_allowed_origin_is_reached_and_the_server_sees_it(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is True, res
    assert res["status"] == 200
    assert any(p.endswith("index.html") for p in _Recorder.seen), _Recorder.seen
    assert "page" in b.read_text()


def test_a_denied_host_is_never_contacted(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["localhost"])
    b = browser()
    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is False
    assert "deny list" in (res.get("refused") or ""), res
    assert _Recorder.seen == [], "the origin server saw a request it should not have"


def test_net_mode_off_stops_the_browser_entirely(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="off")
    b = browser()
    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is False
    assert "net_mode=off" in (res.get("refused") or ""), res
    assert _Recorder.seen == []


def test_a_subresource_on_a_denied_host_is_blocked_while_the_page_loads(
        browser, origin):
    """The page is served from localhost and pulls an image from 127.0.0.1.
    Both are loopback; only the image's host is denied."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is True, res
    assert any(p.endswith("index.html") for p in _Recorder.seen)
    assert not any(p.endswith("sub.png") for p in _Recorder.seen), (
        "a subresource on a denied host reached the origin: %r" % (_Recorder.seen,))
    blocked = b.blocked_requests()
    assert any("sub.png" in x["url"] for x in blocked), blocked


def test_a_file_url_is_refused(browser, tmp_path):
    _set(net_mode="allow", net_allow_private=True)
    target = tmp_path / "secret.txt"
    target.write_text("TOPSECRET", encoding="utf-8")
    b = browser()
    res = b.navigate(target.as_uri())
    assert res["ok"] is False
    assert "not allowed" in (res.get("refused") or ""), res


def test_a_closed_session_leaves_no_profile_on_disk(browser, origin):
    """No-traces: the browser writes its profile to the OS temp dir while it
    runs, not into localm's data dir, and a clean stop() removes it.

    A HARD KILL of the process is a different case and is not covered: nothing
    runs to do the removal, so the temp profile survives until the OS clears it.
    """
    import glob
    import os
    import tempfile
    import time

    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    root = tempfile.gettempdir()

    def snap():
        found = set()
        for pat in ("playwright*", "*chromium*", "scoped_dir*"):
            found |= set(glob.glob(os.path.join(root, pat)))
        return found

    before = snap()
    b = browser()
    assert b.navigate("http://localhost:%d/index.html" % port)["ok"] is True
    secret = "SECRET-NO-TRACES-7Q4M"
    b._call(lambda: b._page.evaluate(
        "(s) => { document.cookie = 'probe=' + s; localStorage.setItem('k', s); }",
        secret))
    written = b._call(lambda: b._page.evaluate(
        "() => document.cookie + '|' + localStorage.getItem('k')"))
    assert secret in written, "nothing was written, so the check below is vacuous"

    created = snap() - before
    assert created, "no profile directory was created, so nothing is being measured"

    b.stop()
    time.sleep(2.0)
    leftover = created & snap()
    assert not leftover, "a closed session left a profile behind: %r" % (sorted(leftover),)


def _ws_seen():
    return [p for p in _Recorder.seen if p.startswith("/ws")
            and not p.startswith("/ws-page")]


def test_a_websocket_to_a_denied_host_never_connects(browser, origin):
    """WebSockets are NOT covered by the ordinary request route and need their
    own handler. Without one they reach any host the page names, whatever the
    policy says."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    assert b.navigate("http://localhost:%d/ws-page.html" % port)["ok"] is True
    time.sleep(1.5)
    assert _ws_seen() == [], (
        "a websocket reached a denied host: %r" % (_Recorder.seen,))
    assert any("/ws" in x["url"] for x in b.blocked_requests()), b.blocked_requests()


def test_a_websocket_to_an_allowed_host_still_connects(browser, origin):
    """The control for the test above: without it, 'no websocket arrived' is
    equally consistent with having broken every websocket."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/ws-page.html" % port)["ok"] is True
    time.sleep(1.5)
    assert _ws_seen(), (
        "an allowed websocket never reached the origin: %r" % (_Recorder.seen,))
