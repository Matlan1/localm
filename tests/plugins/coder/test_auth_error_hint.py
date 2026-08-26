# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder 401/403 must surface an actionable API-key hint, not a bare
'401 Client Error'. The HTTP backend raises CoderAuthError (carrying how to
find/set the key) for auth statuses, and behaves normally otherwise.

A non-auth error status whose response carries server-provided detail
(FastAPI's {"detail": ...} body, e.g. the grammar-worker-fault 503 from
inference/routes/chat.py) must also surface that detail:
requests.raise_for_status() never reads the body at all.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from localm.plugins.coder.backends.http import (
    CoderAuthError, CoderServerError, HTTPBackend, _response_detail)

_PATCH = "localm.plugins.coder.backends.http._post_with_retry"


def _resp(status, *, raise_exc=None, json_body=None, text_body=None):
    r = MagicMock()
    r.status_code = status
    r.url = "http://127.0.0.1:8080/v1/chat/completions"
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    if raise_exc is not None:
        r.raise_for_status.side_effect = raise_exc
    if json_body is not None:
        r.json.return_value = json_body
    else:
        r.json.side_effect = ValueError("no JSON body")
    r.text = text_body if text_body is not None else ""
    return r


def _backend():
    return HTTPBackend("http://127.0.0.1:8080/v1", "test-model")


@pytest.mark.parametrize("status", [401, 403])
def test_chat_auth_status_raises_actionable_auth_error(status):
    with patch(_PATCH, return_value=_resp(status)):
        with pytest.raises(CoderAuthError) as ei:
            _backend().chat([{"role": "user", "content": "hi"}])
    msg = str(ei.value)
    assert "localm key show --reveal" in msg
    assert "LOCALM_API_KEY" in msg


def test_chat_stream_401_raises_auth_error():
    with patch(_PATCH, return_value=_resp(401)):
        with pytest.raises(CoderAuthError):
            list(_backend().chat_stream([{"role": "user", "content": "hi"}]))


def test_non_auth_status_still_raises_plain_httperror():
    """No JSON/text detail available on the response - falls back to the
    original resp.raise_for_status() behaviour unchanged."""
    err = requests.HTTPError("500 Server Error")
    with patch(_PATCH, return_value=_resp(500, raise_exc=err)):
        with pytest.raises(requests.HTTPError):
            _backend().chat([{"role": "user", "content": "hi"}])


def test_grammar_worker_fault_503_surfaces_server_detail():
    """The exact shape inference/routes/chat.py's grammar-worker-fault
    503 sends (a FastAPI {"detail": "..."} body) must reach the raised
    exception's message, not just the bare status line."""
    body = {"detail": "Grammar validation failed: the model worker faulted "
                      "(the model process crashed)."}
    with patch(_PATCH, return_value=_resp(503, json_body=body)):
        with pytest.raises(CoderServerError) as ei:
            _backend().chat([{"role": "user", "content": "hi"}])
    msg = str(ei.value)
    assert "503" in msg
    assert "the model worker faulted" in msg


def test_non_json_text_body_detail_still_surfaces():
    """A server that returns a plain-text error body (not FastAPI's JSON
    shape) still gets its detail surfaced, not silently dropped."""
    with patch(_PATCH, return_value=_resp(502, text_body="upstream refused the connection")):
        with pytest.raises(CoderServerError) as ei:
            _backend().chat([{"role": "user", "content": "hi"}])
    assert "upstream refused the connection" in str(ei.value)


def test_response_detail_never_raises_on_a_malformed_response():
    """A response whose .json()/.text access themselves misbehave must not
    turn an error-reporting helper into a NEW crash - degrade to ""."""
    r = MagicMock()
    r.json.side_effect = RuntimeError("boom")
    type(r).text = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _response_detail(r) == ""


def test_response_detail_truncates_a_huge_body():
    body = {"detail": "x" * 5000}
    r = MagicMock()
    r.json.return_value = body
    assert len(_response_detail(r)) == 500
