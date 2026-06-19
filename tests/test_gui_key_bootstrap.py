"""C1 (GUI half): a fresh loopback `localm gui` launch must not be locked out
when require_auth is on. The SPA shell seeds the configured API key into
localStorage on a loopback bind, but never hands it to a non-loopback LAN client.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.web import (
    attach_gui,
    _index_html_with_key,
    _is_loopback_host,
)

BOOT = "localStorage.setItem('localm.apiKey'"


def _app(bind_host):
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    app.state.bind_host = bind_host
    return app


class TestIsLoopbackHost:
    def test_loopback(self):
        assert _is_loopback_host("127.0.0.1")
        assert _is_loopback_host("127.0.0.5")
        assert _is_loopback_host("::1")
        assert _is_loopback_host("localhost")

    def test_not_loopback(self):
        assert not _is_loopback_host("0.0.0.0")
        assert not _is_loopback_host("192.168.1.10")
        assert not _is_loopback_host("")
        assert not _is_loopback_host("testclient")


class TestIndexInjection:
    def test_injects_key(self):
        html = _index_html_with_key("KEY-abc_123")
        assert BOOT in html
        assert '"KEY-abc_123"' in html

    def test_no_key_no_injection(self):
        assert BOOT not in _index_html_with_key("")

    def test_script_breakout_escaped(self):
        # a "<" in the (operator-set) key is unicode-escaped so it can never
        # close the <script> element.
        html = _index_html_with_key("a</script>b")
        assert "a</script>b" not in html
        assert "a\\u003c/script>b" in html


class TestShellRoute:
    def test_loopback_bind_seeds_key(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("127.0.0.1")).get("/")
        assert r.status_code == 200
        assert BOOT in r.text
        assert "REALKEY123" in r.text

    def test_no_key_no_seed(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        r = TestClient(_app("127.0.0.1")).get("/")
        assert r.status_code == 200
        assert BOOT not in r.text

    def test_lan_bind_never_seeds(self, monkeypatch):
        # A non-loopback bind never seeds the key, regardless of the client - we
        # do not trust request.client.host (a same-host proxy would read as
        # loopback for remote users). The same-machine user enters the key.
        monkeypatch.setenv("LOCALM_API_KEY", "REALKEY123")
        r = TestClient(_app("0.0.0.0")).get("/")
        assert r.status_code == 200
        assert "REALKEY123" not in r.text
