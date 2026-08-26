# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm status` must say what a running server is DOING, and must never say
"nothing" on the evidence of not having found out.

"I asked and nothing is running" and "I could not ask" are separate states:
`read_activity` reports which one occurred, and these tests pin each state at
that seam together with what `_print_activity` renders for it.
"""

from __future__ import annotations

import json
import time

import pytest
import requests

from localm.cli.models import _fmt_age, _print_activity
from localm.selfclient import read_activity


class _Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def _patch(monkeypatch, resp=None, exc=None):
    def _get(url, **kw):
        if exc is not None:
            raise exc
        return resp
    monkeypatch.setattr(requests, "get", _get)


# ---------------------------------------------------------- the seam itself

def test_a_connection_failure_is_not_an_empty_activity_list(monkeypatch):
    """THE CENTRAL ONE. A server that cannot be reached must not read as idle."""
    _patch(monkeypatch, exc=requests.exceptions.ConnectionError("refused"))
    state, _ = read_activity("http", 1234)
    assert state == "unreachable"


def test_a_timeout_is_also_unreachable(monkeypatch):
    _patch(monkeypatch, exc=requests.exceptions.Timeout("slow"))
    assert read_activity("http", 1234)[0] == "unreachable"


@pytest.mark.parametrize("code", [401, 403])
def test_an_auth_refusal_is_its_own_state(monkeypatch, code):
    _patch(monkeypatch, resp=_Resp(code, {}))
    assert read_activity("http", 1234)[0] == "unauthorized"


def test_an_older_server_without_the_route_is_its_own_state(monkeypatch):
    """A running 0.1.3 has no /api/activity. That is "cannot tell me", not
    "nothing is running"."""
    _patch(monkeypatch, resp=_Resp(404, {}))
    assert read_activity("http", 1234)[0] == "unsupported"


def test_a_200_that_is_not_json_is_not_an_empty_list(monkeypatch):
    """Something other than localm answered on that port. An empty operation
    list would be a fabricated answer."""
    _patch(monkeypatch, resp=_Resp(200, None, text="<html>hello</html>"))
    assert read_activity("http", 1234)[0] == "http"


def test_an_empty_list_is_a_real_answer(monkeypatch):
    _patch(monkeypatch, resp=_Resp(200, {"now": 1.0, "operations": []}))
    state, body = read_activity("http", 1234)
    assert state == "ok"
    assert body["operations"] == []


# -------------------------------------------------------------- attach token

def test_instance_token_used_when_no_api_key(monkeypatch):
    """A genuinely open server has no API key to send, so the caller's only
    proof of being a local process is the instance's own attach token (the
    0600 registry file's 'token' field). Without it the default, keyless
    install answers a wrong "needs a key" 403."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def _get(url, headers=None, **kw):
        captured["headers"] = headers
        return _Resp(200, {"now": 1.0, "operations": []})
    monkeypatch.setattr(requests, "get", _get)

    state, _ = read_activity("http", 1234, "the-instance-token")
    assert state == "ok"
    assert captured["headers"]["Authorization"] == "Bearer the-instance-token"


def test_api_key_still_wins_over_instance_token(monkeypatch):
    """A protected (keyed) server keeps using the real owner key - the instance
    token is a fallback for open mode only, never a competing credential."""
    monkeypatch.setenv("LOCALM_API_KEY", "owner-secret")
    captured = {}

    def _get(url, headers=None, **kw):
        captured["headers"] = headers
        return _Resp(200, {"now": 1.0, "operations": []})
    monkeypatch.setattr(requests, "get", _get)

    read_activity("http", 1234, "the-instance-token")
    assert captured["headers"]["Authorization"] == "Bearer owner-secret"


def test_no_instance_token_and_no_key_sends_no_auth_header(monkeypatch):
    """No header at all when the caller genuinely has neither (an older client, or
    a direct-path run with no registry entry)."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def _get(url, headers=None, **kw):
        captured["headers"] = headers
        return _Resp(200, {"now": 1.0, "operations": []})
    monkeypatch.setattr(requests, "get", _get)

    read_activity("http", 1234)
    assert "Authorization" not in captured["headers"]


def test_unauthorized_reads_as_blindness_not_an_optional_tip(monkeypatch, capsys):
    """Wording like "This server needs an API key..." reads as a hardening
    suggestion. What a 401/403 here actually means is an inability to answer,
    identical whether the server is busy or idle, so the message must use the
    SAME "could not ask" framing the unreachable branch uses instead of leading
    with the requirement."""
    _patch(monkeypatch, resp=_Resp(401, {}))
    _print_activity("http", 1234)
    out = _out(capsys)
    assert "could not ask this server what it is doing" in out.lower()
    # The requirement-first wording is pinned as absent.
    assert "this server needs an api key" not in out.lower()


# ------------------------------------------------------------- what prints

def _out(capsys):
    return capsys.readouterr().out


@pytest.mark.parametrize("kind,exc,resp", [
    ("unreachable", requests.exceptions.ConnectionError("x"), None),
    ("unauthorized", None, _Resp(401, {})),
    ("unsupported", None, _Resp(404, {})),
    ("http", None, _Resp(500, {})),
])
def test_no_failure_path_ever_claims_nothing_is_running(monkeypatch, capsys,
                                                        kind, exc, resp):
    """Whatever went wrong, the user must not be told the server is idle."""
    _patch(monkeypatch, resp=resp, exc=exc)
    _print_activity("http", 1234)
    out = _out(capsys).lower()
    assert "nothing running" not in out, f"{kind} printed an idle claim: {out!r}"
    assert out.strip(), f"{kind} printed nothing at all"


def test_the_idle_case_says_so_plainly(monkeypatch, capsys):
    _patch(monkeypatch, resp=_Resp(200, {"now": 1.0, "operations": []}))
    _print_activity("http", 1234)
    assert "nothing running" in _out(capsys).lower()


def test_a_running_operation_is_listed_with_its_label(monkeypatch, capsys):
    now = time.time()
    _patch(monkeypatch, resp=_Resp(200, {"now": now, "operations": [
        {"id": "a", "kind": "pull", "label": "Model pull owner/repo",
         "status": "running", "created_at": now - 65, "finished_at": None,
         "cancellable": True, "pct": 41.2}]}))
    _print_activity("http", 1234)
    out = _out(capsys)
    assert "Model pull owner/repo" in out
    assert "running" in out
    assert "41%" in out
    assert "1m" in out, f"expected an age from the server clock: {out!r}"


def test_the_age_uses_the_server_clock_not_this_machines(monkeypatch, capsys):
    """The payload carries the server's `now` precisely so a skewed local clock
    cannot produce a confident, wrong duration. Here the server says the job is
    one minute old while this machine's clock is an hour off; the printed age
    must follow the server."""
    server_now = time.time() - 3600
    _patch(monkeypatch, resp=_Resp(200, {"now": server_now, "operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": server_now - 60, "finished_at": None,
         "cancellable": True}]}))
    _print_activity("http", 1234)
    out = _out(capsys)
    assert "1m" in out, out
    assert "1h" not in out, f"age was computed against the local clock: {out!r}"


def test_no_percentage_is_printed_when_none_was_reported(monkeypatch, capsys):
    """R1: an operation that has not reported progress is at an UNKNOWN
    percentage, never 0%."""
    now = time.time()
    _patch(monkeypatch, resp=_Resp(200, {"now": now, "operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": now - 5, "finished_at": None, "cancellable": True}]}))
    _print_activity("http", 1234)
    out = _out(capsys)
    assert "%" not in out, f"printed a percentage nobody reported: {out!r}"


def test_a_missing_server_clock_suppresses_the_age_rather_than_faking_one(
        monkeypatch, capsys):
    now = time.time()
    _patch(monkeypatch, resp=_Resp(200, {"operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": now - 7200, "finished_at": None, "cancellable": True}]}))
    _print_activity("http", 1234)
    out = _out(capsys)
    assert "P" in out
    # Assert against the OPERATION's own line, not the whole output: any
    # unrelated line containing an "h" would fail a whole-output match. The
    # second assertion names the age this fixture would otherwise produce.
    op_line = next(ln for ln in out.splitlines() if "running" in ln)
    assert "h" not in op_line, (
        f"invented an age with no reference clock: {op_line!r}")
    assert "2h00m" not in out, f"invented an age with no reference clock: {out!r}"


# --------------------------------------------------------------- formatting

@pytest.mark.parametrize("secs,want", [
    (0, "0s"), (5, "5s"), (59, "59s"), (60, "1m"), (599, "9m"),
    (3600, "1h00m"), (3725, "1h02m"),
])
def test_age_formatting(secs, want):
    assert _fmt_age(secs) == want


def test_a_negative_or_absent_age_renders_as_nothing():
    """Clock skew can make now - created_at negative. Print nothing, never "-3s",
    which reads as a real measurement of something impossible."""
    assert _fmt_age(-3) == ""
    assert _fmt_age(None) == ""
