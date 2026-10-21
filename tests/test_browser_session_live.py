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


#: A link covering the top-left corner, so a click at (50, 50) follows it.
_CORNER_LINK = (b"<a id=go href='%s' target='_blank' style='position:absolute;"
                b"left:0;top:0;width:300px;height:120px;background:#ccc'>go</a>")

#: Pages served by path. PORT is replaced with the origin's own port.
_PAGES = {
    "/opener": b"<html><body>" + _CORNER_LINK % b"/popup-page" + b"</body></html>",
    "/popup-page": (
        b"<html><body><div id=n>0</div><script>"
        b"let i=0;setInterval(()=>{document.getElementById('n').textContent=++i;"
        b"console.log('popup-alive '+i);},100);</script></body></html>"),
    "/scripted": (
        b"<html><body>scripted<script>"
        b"setTimeout(()=>window.open('/popup-page'),300);</script></body></html>"),
    "/closer": b"<html><body>" + _CORNER_LINK % b"/selfclose" + b"</body></html>",
    "/selfclose": (
        b"<html><body>closing<script>setTimeout(()=>window.close(),300);"
        b"</script></body></html>"),
    "/denied-opener": (
        b"<html><body>" + _CORNER_LINK % b"http://127.0.0.1:PORT/TARGET"
        + b"</body></html>"),
    "/typing": (
        b"<html><body><textarea id=t style='position:absolute;left:0;top:0;"
        b"width:300px;height:120px'></textarea><script>"
        b"document.getElementById('t').addEventListener('keydown',()=>{"
        b"const t=performance.now();while(performance.now()-t<25){}});"
        b"</script></body></html>"),
    "/blank-then-go": (
        b"<html><body><button id=b style='position:absolute;left:0;top:0;"
        b"width:300px;height:120px' onclick=\"const w=window.open('');"
        b"setTimeout(()=>{w.location='/popup-page';},500);\">go</button>"
        b"</body></html>"),
    "/late-link": (
        b"<html><body>waiting<script>setTimeout(()=>{document.body.innerHTML="
        b"\"<a id=late href='/popup-page' target='_blank'>late</a>\";},6000);"
        b"</script></body></html>"),
    "/stalled": (
        b"<html><body><img src='/slow.png'><script>"
        b"setTimeout(()=>window.open('/popup-page'),100);</script></body></html>"),
}

#: Worker test files by path, as (content type, text); set by the worker tests.
_WORKER_FILES: dict = {}

#: Seconds the origin takes to answer /slow.png.
_SLOW_SECONDS = 8


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):                                # noqa: N802
        type(self).seen.append(self.path)
        port = self.server.server_address[1]
        # Redirect fixtures. Without a 3xx in the fixture's value space, no test
        # here can express a redirect bypass at all.
        hops = {
            "/redir-denied": "http://127.0.0.1:%d/TARGET" % port,
            "/chain-1":      "http://localhost:%d/chain-2" % port,
            "/chain-2":      "http://127.0.0.1:%d/TARGET" % port,
            "/redir-ok":     "http://localhost:%d/index.html" % port,
        }
        if self.path in hops:
            self.send_response(302)
            self.send_header("Location", hops[self.path])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.split("?")[0] in _WORKER_FILES:
            ctype, text = _WORKER_FILES[self.path.split("?")[0]]
            body = text.encode()
        elif self.path in _PAGES:
            body = _PAGES[self.path].replace(b"PORT", str(port).encode())
            ctype = "text/html"
        elif self.path == "/slow.png":
            time.sleep(_SLOW_SECONDS)
            body = b"\x89PNG\r\n\x1a\n"
            ctype = "image/png"
        elif self.path.endswith(".png"):
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


def test_no_websocket_reaches_its_host(browser, origin):
    """WebSockets are refused, and the refusal is reported rather than left as a
    socket that looks open and dies. The denied-host reason is the policy's own;
    an otherwise-allowed host is refused for the stated availability reason."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    assert b.navigate("http://localhost:%d/ws-page.html" % port)["ok"] is True
    time.sleep(1.5)
    assert _ws_seen() == [], (
        "a websocket reached its host: %r" % (_Recorder.seen,))
    blocked = [x for x in b.blocked_requests() if x["url"].startswith("ws")]
    assert blocked, b.blocked_requests()
    assert "deny list" in blocked[0]["reason"], blocked


def test_an_allowed_websocket_is_refused_for_the_stated_reason(browser, origin):
    """The control for the shape above: with nothing denied, the refusal must
    still happen and must NOT claim a policy reason it does not have."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/ws-page.html" % port)["ok"] is True
    time.sleep(1.5)
    assert _ws_seen() == [], _Recorder.seen
    blocked = [x for x in b.blocked_requests() if x["url"].startswith("ws")]
    assert blocked, "the refusal was not recorded, so nothing tells the caller"
    assert "not available" in blocked[0]["reason"], blocked


# --------------------------------------------------------------------------- #
#  Workers. Their WebSockets, and a SharedWorker's requests, are not routed, so
#  they must fail inside the browser without connecting anywhere.
# --------------------------------------------------------------------------- #

class _ConnectCounter:
    """A raw TCP listener on loopback (IPv4 by default, ::1 when *ipv6*) that
    counts every connection accepted."""

    def __init__(self, ipv6: bool = False):
        import socket
        self.accepted = 0
        if ipv6:
            self._sock = socket.socket(socket.AF_INET6)
            self._sock.bind(("::1", 0))
        else:
            self._sock = socket.socket()
            self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self.accepted += 1
            conn.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()


@pytest.fixture
def counter(request):
    c = _ConnectCounter(ipv6=getattr(request, "param", None) == "[::1]")
    try:
        yield c
    finally:
        c.close()


def _worker_page(kind: str, ws_url: str) -> None:
    """Serve /wk-page, which starts one *kind* of worker that opens a WebSocket
    to *ws_url* and then reports that it ran by a request for /ran?src=KIND."""
    opener = ("try { new WebSocket(%r); } catch (e) {} " % ws_url)
    if kind == "worker":
        page = "new Worker('/wk.js');"
        script = opener + "fetch('/ran?src=worker');"
    elif kind == "shared":
        page = ("const w = new SharedWorker('/wk.js'); "
                "w.port.onmessage = () => fetch('/ran?src=shared'); w.port.start();")
        script = opener + "onconnect = (e) => e.ports[0].postMessage('ran');"
    else:
        page = "navigator.serviceWorker.register('/wk.js');"
        script = ("self.addEventListener('install', (e) => { self.skipWaiting(); "
                  + opener + "fetch('/ran?src=sw'); });")
    _WORKER_FILES.clear()
    _WORKER_FILES["/wk-page"] = (
        "text/html", "<html><body>workers<script>" + page + "</script></body></html>")
    _WORKER_FILES["/wk.js"] = ("text/javascript", script)


def _wait_for(path_prefix: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.startswith(path_prefix) for p in _Recorder.seen):
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("kind", ["worker", "shared", "sw"])
@pytest.mark.parametrize("counter, host", [
    ("127.0.0.1", "127.0.0.1"), ("127.0.0.1", "localhost"), ("[::1]", "[::1]"),
], indirect=["counter"])
def test_a_worker_websocket_never_connects(browser, origin, counter, kind, host):
    """A WebSocket opened from a dedicated worker, a SharedWorker or a service
    worker never makes a TCP connection, even to a host the policy allows."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _worker_page(kind, "ws://%s:%d/ws" % (host, counter.port))
    b = browser()
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/ran?src=" + kind), (
        "the %s never ran, so nothing was measured: %r" % (kind, _Recorder.seen))
    time.sleep(1.5)
    assert counter.accepted == 0, (
        "a %s WebSocket connected to its host %d time(s)" % (kind, counter.accepted))


def test_the_system_engine_carries_the_resolver_rule(browser, origin, counter,
                                                     monkeypatch):
    """The system engine, driven with the downloaded Chromium as the installed
    browser, keeps a worker's WebSocket from connecting too."""
    from localm.browser import discovery, provision
    exe = provision.chromium_executable_path()
    if exe is None or not exe.exists():
        pytest.skip("the downloaded Chromium is not installed")
    monkeypatch.setattr(discovery, "find_system_browsers",
                        lambda **kw: [discovery.SystemBrowser("Chromium", str(exe))])
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _worker_page("worker", "ws://127.0.0.1:%d/ws" % counter.port)
    b = browser(engine="system")
    assert b.browser_name == "Chromium"
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/ran?src=worker"), _Recorder.seen
    time.sleep(1.5)
    assert counter.accepted == 0, (
        "a worker WebSocket from the system engine connected %d time(s)"
        % counter.accepted)


@pytest.fixture(params=["Google Chrome", "Microsoft Edge"])
def branded(request, monkeypatch):
    """Only the installed browser named by the parameter is offered to the
    system engine."""
    from localm.browser import discovery
    found = [b for b in discovery.find_system_browsers() if b.name == request.param]
    if not found:
        pytest.skip("%s is not installed" % request.param)
    monkeypatch.setattr(discovery, "find_system_browsers", lambda **kw: found)
    return request.param


@pytest.mark.parametrize("kind", ["worker", "shared", "sw"])
def test_a_branded_browser_keeps_a_worker_websocket_from_connecting(
        browser, origin, counter, branded, kind):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _worker_page(kind, "ws://127.0.0.1:%d/ws" % counter.port)
    b = browser(engine="system")
    assert b.browser_name == branded
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/ran?src=" + kind), _Recorder.seen
    time.sleep(1.5)
    assert counter.accepted == 0, (
        "a %s WebSocket from %s connected %d time(s)"
        % (kind, branded, counter.accepted))


def test_a_branded_browser_leaves_no_profile_folder_after_stop(
        browser, origin, branded):
    import glob
    import os
    import tempfile
    pattern = os.path.join(tempfile.gettempdir(), "localm-browser-*")
    before = set(glob.glob(pattern))
    _set(net_mode="allow", net_allow_private=True)
    b = browser(engine="system")
    assert b.navigate("http://localhost:%d/index.html" % origin.server_address[1])["ok"]
    during = set(glob.glob(pattern)) - before
    assert len(during) == 1, during
    b.stop()
    assert not (set(glob.glob(pattern)) - before)


def test_a_branded_browser_connects_without_the_resolver_rule(
        browser, origin, counter, branded, monkeypatch):
    """The control for the test above: the same page, with the launch switches
    dropped, does reach the listener, so a zero there is a measurement."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    monkeypatch.setattr(bsession, "_LAUNCH_ARGS", ())
    _worker_page("worker", "ws://127.0.0.1:%d/ws" % counter.port)
    b = browser(engine="system")
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/ran?src=worker"), _Recorder.seen
    deadline = time.monotonic() + 5
    while counter.accepted == 0 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert counter.accepted > 0, "%s never connected without the rule" % branded


def test_a_shared_worker_request_never_reaches_its_host(browser, origin):
    """A SharedWorker's fetch is not routed, so it must not leave the browser."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _WORKER_FILES.clear()
    _WORKER_FILES["/wk-page"] = (
        "text/html",
        "<html><body>shared<script>const w = new SharedWorker('/wk.js'); "
        "w.port.onmessage = () => fetch('/ran?src=shared'); w.port.start();"
        "</script></body></html>")
    _WORKER_FILES["/wk.js"] = (
        "text/javascript",
        "onconnect = (e) => { fetch('/shared-fetch')"
        ".catch(() => {}).finally(() => e.ports[0].postMessage('ran')); };")
    b = browser()
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/ran?src=shared"), _Recorder.seen
    assert not any(p.startswith("/shared-fetch") for p in _Recorder.seen), (
        "a SharedWorker request reached its host: %r" % (_Recorder.seen,))


def test_a_routed_request_still_loads_under_the_resolver_rule(browser, origin):
    """The control for the worker tests: a dedicated worker's own fetch is
    routed, so it is decided by the gate and reaches an allowed host."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _WORKER_FILES.clear()
    _WORKER_FILES["/wk-page"] = (
        "text/html", "<html><body>w<script>new Worker('/wk.js');</script></body></html>")
    _WORKER_FILES["/wk.js"] = (
        "text/javascript", "fetch('http://127.0.0.1:%d/routed-fetch');" % port)
    b = browser()
    assert b.navigate("http://localhost:%d/wk-page" % port)["ok"] is True
    assert _wait_for("/routed-fetch"), _Recorder.seen
    assert any(u.endswith("/routed-fetch") for u in b.allowed_requests())


class _DatagramCounter:
    """A UDP socket on 127.0.0.1 that counts every datagram it receives."""

    def __init__(self):
        import socket
        self.received = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sock.recvfrom(4096)
            except OSError:
                continue
            self.received += 1

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()


@pytest.fixture
def datagrams():
    d = _DatagramCounter()
    try:
        yield d
    finally:
        d.close()


def _rtc_page(udp_port: int, tcp_port: int) -> None:
    """Serve /rtc-page: a peer connection with a STUN server, a TURN server over
    TCP and one over UDP, all given as IP literals, that then starts ICE checks
    toward a remote candidate on *udp_port*; it reports how far it got with
    /ran?step=N (4 = every call returned)."""
    script = (
        "var step = 0;"
        "setTimeout(() => fetch('/ran?step=' + step), 5000);"
        "(async () => {"
        " const NL = String.fromCharCode(13, 10);"
        " const cand = 'candidate:1 1 udp 2122260223 127.0.0.1 %(u)d typ host';"
        " const pc = new RTCPeerConnection({iceServers: ["
        "  {urls: 'stun:127.0.0.1:%(u)d'},"
        "  {urls: 'turn:127.0.0.1:%(t)d?transport=tcp', username: 'u', credential: 'p'},"
        "  {urls: 'turn:127.0.0.1:%(u)d?transport=udp', username: 'u', credential: 'p'}]});"
        " step = 1; pc.createDataChannel('x');"
        " await pc.setLocalDescription(await pc.createOffer()); step = 2;"
        " const remote = new RTCPeerConnection();"
        " await remote.setRemoteDescription(pc.localDescription);"
        " const ans = await remote.createAnswer();"
        " const sdp = ans.sdp.split(NL).filter(l => !l.startsWith('a=candidate'))"
        "  .join(NL).trim() + NL + 'a=' + cand + NL;"
        " await pc.setRemoteDescription({type: 'answer', sdp}); step = 3;"
        " await pc.addIceCandidate({candidate: cand, sdpMid: '0', sdpMLineIndex: 0});"
        " step = 4;"
        "})();") % {"u": udp_port, "t": tcp_port}
    _WORKER_FILES.clear()
    _WORKER_FILES["/rtc-page"] = (
        "text/html", "<html><body>rtc<script>" + script + "</script></body></html>")


def test_webrtc_sends_nothing_to_an_ip_literal(browser, origin, counter, datagrams):
    """A page's RTCPeerConnection sends no UDP datagram (STUN, TURN over UDP,
    ICE connectivity checks) and opens no TCP connection (TURN over TCP) to a
    host given as an IP literal, which the resolver rule does not cover."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _rtc_page(datagrams.port, counter.port)
    b = browser()
    assert b.navigate("http://localhost:%d/rtc-page" % port)["ok"] is True
    assert _wait_for("/ran?step=4"), (
        "the peer connection calls did not all return, so nothing was "
        "measured: %r" % (_Recorder.seen,))
    time.sleep(1.5)
    assert datagrams.received == 0, (
        "WebRTC sent %d UDP datagram(s) to an IP literal" % datagrams.received)
    assert counter.accepted == 0, (
        "WebRTC opened %d TCP connection(s) to an IP literal" % counter.accepted)


def test_a_branded_browser_sends_no_webrtc_traffic_to_an_ip_literal(
        browser, origin, counter, datagrams, branded):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    _rtc_page(datagrams.port, counter.port)
    b = browser(engine="system")
    assert b.navigate("http://localhost:%d/rtc-page" % port)["ok"] is True
    assert _wait_for("/ran?step=4"), _Recorder.seen
    time.sleep(1.5)
    assert datagrams.received == 0, (
        "%s sent %d UDP datagram(s) to an IP literal" % (branded, datagrams.received))
    assert counter.accepted == 0, (
        "%s opened %d TCP connection(s) to an IP literal" % (branded, counter.accepted))


# --------------------------------------------------------------------------- #
#  Redirects. A browser follows these itself, and a followed redirect is not
#  routed, so its target would never be decided. Every hop is gated instead.
# --------------------------------------------------------------------------- #

def test_a_redirect_to_a_denied_host_never_reaches_it(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    res = b.navigate("http://localhost:%d/redir-denied" % port)
    assert res["ok"] is False, res
    assert "deny list" in (res.get("refused") or ""), res
    assert res.get("refused_url", "").endswith("/TARGET"), res
    assert "/TARGET" not in _Recorder.seen, (
        "the denied redirect target was fetched: %r" % (_Recorder.seen,))


def test_a_two_hop_redirect_chain_is_gated_at_every_hop(browser, origin):
    """One hop of checking is not enough: an allowed host can redirect to
    another allowed host that redirects to a denied one."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    res = b.navigate("http://localhost:%d/chain-1" % port)
    assert res["ok"] is False, res
    assert "deny list" in (res.get("refused") or ""), res
    assert "/chain-2" in _Recorder.seen, "the allowed middle hop should be followed"
    assert "/TARGET" not in _Recorder.seen, (
        "the denied end of the chain was fetched: %r" % (_Recorder.seen,))


def test_an_allowed_redirect_still_follows(browser, origin):
    """The control. Without it, 'the target was never fetched' is equally
    consistent with having broken redirects altogether."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    res = b.navigate("http://localhost:%d/redir-ok" % port)
    assert res["ok"] is True, res
    assert "/index.html" in _Recorder.seen, _Recorder.seen
    assert "page" in b.read_text()


def test_a_gate_that_raises_refuses_the_request(browser, origin, monkeypatch):
    """A handler that raises would otherwise leave the request hanging until the
    caller's own timeout, which reads as a slow site rather than a broken gate."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()

    async def boom(*a, **kw):
        raise RuntimeError("gate exploded")
    monkeypatch.setattr(bsession.netgate, "decide_async", boom)

    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is False, res
    assert _Recorder.seen == [], "the request went out while the gate was broken"
    assert any("gate failed" in x["reason"] for x in b.blocked_requests()), \
        b.blocked_requests()


# --------------------------------------------------------------------------- #
#  Windows a page opens. The session keeps exactly one page: a window a click
#  opened replaces the page, any other window is closed and the refusal
#  recorded, and a shown window that closes itself leaves a blank page behind.
# --------------------------------------------------------------------------- #

def _pages(b):
    """The URLs of every open page, and the URL of the page being driven."""
    async def snap():
        return [p.url for p in b._ctx.pages], b._page.url
    return b._call(snap)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(0.1)


def test_a_window_opened_by_a_click_replaces_the_page(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    frames = []
    b = browser(on_frame=lambda data: frames.append(time.monotonic()))
    assert b.navigate("http://localhost:%d/opener" % port)["ok"] is True
    target = "http://localhost:%d/popup-page" % port
    res = b.click_coords(50, 50)
    assert res["ok"] is True, res
    _wait_until(lambda: _pages(b) == ([target], target))
    switched = time.monotonic()
    time.sleep(1.5)
    urls, driven = _pages(b)
    assert urls == [target], "open pages after the click: %r" % (urls,)
    assert driven == target
    assert any("popup-alive" in m["text"] for m in b.console_messages()), (
        "the console of the page now shown is not being recorded")
    assert any(t > switched for t in frames), (
        "no live-view frame arrived from the page now shown")
    assert any(m["type"] == "localm" and target in m["text"]
               for m in b.console_messages()), b.console_messages()


def test_a_window_opened_without_a_click_is_closed(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    start = "http://localhost:%d/scripted" % port
    assert b.navigate(start)["ok"] is True

    def refused():
        return [x for x in b.blocked_requests()
                if x["url"].endswith("/popup-page")]
    _wait_until(lambda: refused() and len(_pages(b)[0]) == 1)
    urls, driven = _pages(b)
    assert urls == [start], "open pages: %r" % (urls,)
    assert driven == start
    assert refused(), "the closed window was not recorded: %r" % (b.blocked_requests(),)
    assert "click" in refused()[0]["reason"], refused()


def test_a_scripted_window_that_does_not_load_is_recorded_by_its_address(
        browser, origin, monkeypatch):
    import socket
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    target = "http://127.0.0.1:%d/popup-target" % dead_port
    monkeypatch.setitem(_PAGES, "/scripted-dead", (
        b"<html><body>scripted<script>setTimeout(()=>window.open('"
        + target.encode() + b"'),300);</script></body></html>"))
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    start = "http://localhost:%d/scripted-dead" % port
    assert b.navigate(start)["ok"] is True

    def refused():
        return [x for x in b.blocked_requests() if x["url"] == target]
    _wait_until(lambda: refused() and len(_pages(b)[0]) == 1)
    urls, _driven = _pages(b)
    assert urls == [start], "open pages: %r" % (urls,)
    assert refused(), (
        "the closed window was not recorded by its address: %r" % (b.blocked_requests(),))
    assert "did not load" in refused()[0]["reason"], refused()


def test_a_shown_window_that_closes_itself_leaves_a_working_page(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/closer" % port)["ok"] is True
    assert b.click_coords(50, 50)["ok"] is True
    _wait_until(lambda: "/selfclose" in _Recorder.seen)
    _wait_until(lambda: _pages(b) == (["about:blank"], "about:blank"))
    urls, driven = _pages(b)
    assert urls == ["about:blank"], "open pages: %r" % (urls,)
    assert driven == "about:blank"
    res = b.navigate("http://localhost:%d/index.html" % port)
    assert res["ok"] is True, res
    assert any(m["type"] == "localm" and "closed itself" in m["text"]
               for m in b.console_messages()), b.console_messages()


def test_a_clicked_window_that_does_not_load_is_closed(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True, net_deny=["127.0.0.1"])
    b = browser()
    start = "http://localhost:%d/denied-opener" % port
    assert b.navigate(start)["ok"] is True
    assert b.click_coords(50, 50)["ok"] is True
    _wait_until(lambda: any(x["url"].endswith("/TARGET")
                            for x in b.blocked_requests())
                and any("did not load" in x["reason"]
                        for x in b.blocked_requests())
                and len(_pages(b)[0]) == 1)
    urls, driven = _pages(b)
    assert urls == [start], "open pages: %r" % (urls,)
    assert driven == start
    assert "/TARGET" not in _Recorder.seen
    assert any(x["url"].endswith("/TARGET") and "deny list" in x["reason"]
               for x in b.blocked_requests()), b.blocked_requests()
    assert any("did not load" in x["reason"] for x in b.blocked_requests()), (
        b.blocked_requests())


def test_typing_stops_when_the_call_gives_up(browser, origin):
    """On a page whose key handler is slow, typing stops within a few
    characters of type_text reporting that it timed out."""
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/typing" % port)["ok"] is True
    assert b.click_coords(50, 50)["ok"] is True

    def typed():
        return b._call(lambda: b._page.eval_on_selector(
            "#t", "e => e.value.length"))
    res = b.type_text("a" * 2000, timeout_ms=1000)
    after = typed()
    time.sleep(2.0)
    later = typed()
    assert later - after <= 32, (
        "%d characters kept landing after type_text reported failure"
        % (later - after))
    assert after > 0, "nothing was typed, so the check above is vacuous"
    assert res["ok"] is False, res
    assert "typing" in res["error"], res


def test_a_clicked_blank_window_is_shown_once_its_opener_sends_it_on(
        browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/blank-then-go" % port)["ok"] is True
    target = "http://localhost:%d/popup-page" % port
    assert b.click_coords(50, 50)["ok"] is True
    _wait_until(lambda: _pages(b) == ([target], target))
    urls, driven = _pages(b)
    assert urls == [target], "open pages: %r" % (urls,)
    assert driven == target


def test_a_window_from_a_selector_click_that_waited_is_shown(browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    assert b.navigate("http://localhost:%d/late-link" % port)["ok"] is True
    target = "http://localhost:%d/popup-page" % port
    res = b.click("#late")
    assert res["ok"] is True, res
    _wait_until(lambda: _pages(b) == ([target], target))
    urls, driven = _pages(b)
    assert urls == [target], "open pages: %r" % (urls,)
    assert driven == target
    assert not any("without a click" in x["reason"] for x in b.blocked_requests())


def test_a_navigation_timeout_is_not_reported_as_a_refused_window(
        browser, origin):
    port = origin.server_address[1]
    _set(net_mode="allow", net_allow_private=True)
    b = browser()
    res = b.navigate("http://localhost:%d/stalled" % port, timeout_ms=3000)
    _wait_until(lambda: any(x["url"].endswith("/popup-page")
                            for x in b.blocked_requests()))
    assert any(x["url"].endswith("/popup-page") for x in b.blocked_requests()), (
        "the window the page opened was not refused, so this checks nothing")
    assert res["ok"] is False, res
    assert res["refused"] is None, (
        "a navigation timeout was reported as a refusal: %r" % (res,))
