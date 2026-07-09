# SPDX-License-Identifier: AGPL-3.0-or-later
"""U9: a coder 401/403 must surface an actionable API-key hint, not a bare
'401 Client Error'. The HTTP backend raises CoderAuthError (carrying how to
find/set the key) for auth statuses, and behaves normally otherwise.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from localm.plugins.coder.backends.http import CoderAuthError, HTTPBackend

_PATCH = "localm.plugins.coder.backends.http._post_with_retry"


def _resp(status, *, raise_exc=None):
    r = MagicMock()
    r.status_code = status
    r.url = "http://127.0.0.1:8080/v1/chat/completions"
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    if raise_exc is not None:
        r.raise_for_status.side_effect = raise_exc
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
    err = requests.HTTPError("500 Server Error")
    with patch(_PATCH, return_value=_resp(500, raise_exc=err)):
        with pytest.raises(requests.HTTPError):
            _backend().chat([{"role": "user", "content": "hi"}])
