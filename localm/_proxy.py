# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared HTTP client for the localm proxy (the Cloudflare Worker)."""

from __future__ import annotations

import json as _json
from typing import Optional

from localm.bugreport import LocalmError


def _default_opener(method: str, url: str, data, headers: dict, timeout: float):
    """Return (status, body_bytes)."""
    import urllib.error
    import urllib.request

    from localm.http_ssl import verified_urlopen
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        # verified_urlopen (see localm/http_ssl.py) - native cert store first,
        # certifi as fallback.
        with verified_urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        detail = b""
        try:
            detail = e.read()
        except Exception:
            pass
        raise LocalmError(
            "the localm proxy returned an error",
            reason=f"HTTP {e.code}: {detail.decode('utf-8', 'replace')[:300]}".strip())
    except (urllib.error.URLError, OSError) as e:
        raise LocalmError("could not reach the localm proxy",
                          reason=str(getattr(e, "reason", e)))


def request(base: str, path: str, *, method: str = "GET", token: Optional[str] = None,
            body=None, timeout: float = 15.0, opener=None):
    """Call ``<base><path>`` on the proxy and return the parsed JSON dict."""
    if not base:
        raise LocalmError("no proxy endpoint is configured",
                          reason="set the upload/update URL in config")
    url = base.rstrip("/") + path
    headers = {"User-Agent": "localm"}
    if token:
        headers["X-Localm-Token"] = token
    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    op = opener or _default_opener
    status, raw = op(method, url, data, headers, timeout)
    if not (200 <= int(status) < 300):
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        raise LocalmError("the localm proxy returned an error",
                          reason=f"HTTP {status}: {text[:300]}".strip())
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    if not text.strip():
        return {}
    try:
        return _json.loads(text)
    except ValueError:
        raise LocalmError("the localm proxy returned a non-JSON response",
                          reason=text[:200])
