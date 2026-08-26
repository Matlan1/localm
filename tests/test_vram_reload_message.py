# SPDX-License-Identifier: AGPL-3.0-or-later
"""reload_chat_after_media must report the REAL reload outcome, not a blanket
"Chat model ready.".

A non-2xx from /v1/models/load (503 "No model specified", 401, or a 500 while the
media backend still holds VRAM) is reported as a deferral, matching its sibling
unload_chat_for_media's resp.ok check.
"""

from localm import vram


class _FakeJob:
    def __init__(self):
        self.lines = []

    def push(self, ev):
        self.lines.append(ev.get("text", ""))

    def text(self):
        return " | ".join(self.lines)


class _FakeBackend:
    def free_vram(self, s):
        return True          # VRAM handed back -> the reload path runs


class _Resp:
    def __init__(self, ok, status_code=200):
        self.ok = ok
        self.status_code = status_code


def _reload(monkeypatch, resp_or_exc):
    job = _FakeJob()

    def fake_self_request(*a, **k):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    # reload_chat_after_media imports self_request lazily from localm.selfclient.
    monkeypatch.setattr("localm.selfclient.self_request", fake_self_request)
    vram.reload_chat_after_media(
        job, "http://127.0.0.1:8642/v1", {"reload_after": True},
        _FakeBackend(), "image")
    return job


def test_reload_success_reports_ready(monkeypatch):
    job = _reload(monkeypatch, _Resp(ok=True))
    assert "Chat model ready." in job.text()


def test_reload_http_error_is_not_reported_as_success(monkeypatch):
    """A 503 must NOT surface as "Chat model ready."; it must defer with the HTTP
    status."""
    job = _reload(monkeypatch, _Resp(ok=False, status_code=503))
    text = job.text()
    assert "Chat model ready." not in text          # no false success line
    assert "deferred" in text.lower()
    assert "503" in text


def test_reload_transport_error_still_defers(monkeypatch):
    """The transport-exception path defers instead of claiming success."""
    job = _reload(monkeypatch, RuntimeError("connection refused"))
    text = job.text()
    assert "Chat model ready." not in text
    assert "deferred" in text.lower()


def test_reload_skipped_when_off():
    """reload_after off -> no reload attempted, no false "ready"."""
    job = _FakeJob()
    vram.reload_chat_after_media(
        job, "http://127.0.0.1:8642/v1", {"reload_after": False},
        _FakeBackend(), "image")
    assert "Chat model ready." not in job.text()
