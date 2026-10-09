# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mailto link and the upload channel: the configured endpoint, the POST to
the bug-report proxy, and the diagnosis of a failed upload.
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from localm.bugreport._common import MAINTAINER_EMAIL
from localm.bugreport.errors import LocalmError, RateLimitedError
from localm.bugreport.assembly import _strip_report_footer
import localm.bugreport as _br


# Cap on the mailto body, which mail clients and some browsers truncate: the
# prefilled body is a short pointer and the full detail lives in the saved file
# the user attaches or pastes.
_MAX_PREFILL_BODY = 1500


def _truncate_body(body: str) -> str:
    if len(body) <= _MAX_PREFILL_BODY:
        return body
    return body[:_MAX_PREFILL_BODY] + "\n\n... (truncated - see the saved report file)"


def mailto_url(summary: str, body: str) -> str:
    q = urllib.parse.urlencode(
        {"subject": f"localm bug: {summary}", "body": _truncate_body(body)},
        quote_via=urllib.parse.quote)
    return f"mailto:{MAINTAINER_EMAIL}?{q}"


# --------------------------------------------------------------------------- #
#  Upload channel: file the report as a GitHub issue WITHOUT a tester account  #
#                                                                              #
#  The app POSTs the (user-reviewed) report to a configured endpoint - the     #
#  localm bug-report proxy, a small Cloudflare Worker that holds a fine-grained #
#  GitHub token SERVER-SIDE and creates the issue. No token ships in the app,   #
#  the token can only file issues through the rate-limited proxy (never read    #
#  the repo), and it rotates at the proxy without re-shipping. See              #
#  tools/bugreport-proxy/. Uploading is ALWAYS explicit (a user action), never  #
#  automatic - a crash report is only saved to a file.                         #
# --------------------------------------------------------------------------- #

def upload_config() -> tuple:
    """(url, token) for the upload channel from config, or (None, None) when not
    configured. The token is an optional shared secret. Never raises."""
    try:
        from localm.config import load_config
        cfg = load_config()
        url = (cfg.get("bugreport_upload_url") or "").strip() or None
        token = (cfg.get("bugreport_upload_token") or "").strip() or None
        return url, token
    except Exception:
        return None, None


def upload_available() -> bool:
    """True when an upload endpoint is configured (the GUI uses this to decide
    whether to offer a 'Send to maintainer' button)."""
    return _br.upload_config()[0] is not None


def _retry_after_from(headers, body_text) -> int:
    """Best-effort retry delay (seconds) for a 429: the ``Retry-After`` response
    header, else a ``retry_after`` field in the JSON body, else a 30s default."""
    val = None
    try:
        if headers is not None:
            val = headers.get("Retry-After")
    except Exception:
        # Header access is best-effort; a missing/odd header just falls through to
        # the body/default below - never let it break the (already-failed) upload.
        val = None
    if val:
        try:
            return max(1, int(str(val).strip()))
        except (TypeError, ValueError):
            pass
    if body_text:
        try:
            import json as _json
            obj = _json.loads(body_text)
            if isinstance(obj, dict) and obj.get("retry_after"):
                return max(1, int(obj["retry_after"]))
        except Exception:
            # A non-JSON or malformed body simply yields the default delay.
            pass
    return 30


def _classify_url_error(exc) -> tuple:
    """(stage, hint) diagnosing WHERE a failed upload broke, from the ACTUAL error
    only - we never contact a third-party host to test connectivity (offline-first
    + privacy). ``stage`` is offline_or_dns / tls / timeout / unreachable / unknown;
    ``hint`` is a friendly, actionable line (the caller adds the "report is saved /
    email it" part)."""
    import socket
    import ssl

    # A urllib URLError wraps the real cause in .reason (usually an OSError); unwrap
    # so isinstance checks see gaierror/SSLError/timeout, not the URLError shell.
    reason = getattr(exc, "reason", None)
    inner = reason if isinstance(reason, BaseException) else exc

    if isinstance(inner, socket.gaierror):
        return ("offline_or_dns",
                "Could not look up the bug-report server's address. You may be "
                "offline, or a DNS/network setting is blocking the connection.")
    if isinstance(inner, ssl.SSLError):
        return ("tls",
                "A secure (TLS) connection to the bug-report server could not be "
                "established.")
    if isinstance(inner, (TimeoutError, socket.timeout)):
        return ("timeout",
                "The bug-report server did not respond in time. It may be slow or "
                "temporarily unreachable.")
    if isinstance(inner, ConnectionError):
        return ("unreachable",
                "Reached the network but could not connect to the bug-report "
                "server. It may be down, or a firewall/proxy is blocking it.")
    if isinstance(inner, OSError):
        txt = str(getattr(inner, "strerror", "") or inner).lower()
        if "unreachable" in txt or "not known" in txt or "no address" in txt:
            return ("offline_or_dns",
                    "The network appears to be unavailable (could not reach the "
                    "bug-report server). Check your internet connection.")
        return ("unreachable",
                "Could not connect to the bug-report server. It may be down, or a "
                "firewall/proxy is blocking it.")
    return ("unknown",
            "The report could not be sent for an unexpected reason.")


def upload_report(title: str, body: str, *, url: Optional[str] = None,
                  token: Optional[str] = None, timeout: float = 15.0,
                  opener=None) -> dict:
    """POST a user-reviewed report to the upload endpoint and return its JSON
    response (e.g. ``{"url": "<issue url>"}``).

    Only the report text (plus an optional shared secret) is sent; the GitHub
    token lives at the proxy. Raises :class:`LocalmError` on any failure - no
    endpoint, a network error, or a non-2xx response - so a failed send is NEVER
    reported as success (we do not hide problems). *opener* is injectable for
    tests (defaults to urllib)."""
    import json as _json

    # Scrub the title at the upload boundary: it becomes a PUBLIC GitHub issue
    # title, and callers pass the user's raw summary / description first line
    # (report_failure, inference/routes/admin.py). This is the single choke
    # point every uploaded title flows through, so scrubbing here covers every
    # caller. Idempotent; no-ops on empty text.
    title = _br._scrub_secrets(title)

    # The same, for the body's trailing edit-disclaimer: every caller here
    # (report_failure's auto_send/interactive-upload, inference/routes/admin.py)
    # gathers *body* from build_report()'s output or the saved file, both of
    # which carry "you can edit anything above before sending" - true right up
    # until this exact call sends it. Stripped here, the one choke point every
    # uploaded body flows through, so it can never reach a PUBLIC GitHub issue.
    body = _strip_report_footer(body or "")

    # Fill each of url/token from config independently when not explicitly passed,
    # so an explicit token does not suppress loading the url from config (and vice
    # versa).
    if url is None or token is None:
        cfg_url, cfg_token = _br.upload_config()
        if url is None:
            url = cfg_url
        if token is None:
            token = cfg_token
    if not url:
        raise LocalmError("no upload endpoint is configured",
                          reason="set bugreport_upload_url to enable the Send channel",
                          stage="no_endpoint",
                          hint="No bug-report server is configured in this build, so "
                               "the report cannot be filed automatically.")
    payload = _json.dumps(
        {"title": (title or "localm bug report")[:200], "body": body or ""}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "localm-bugreport"}
    if token:
        headers["X-Localm-Token"] = token

    if opener is None:
        import urllib.error
        import urllib.request

        from localm.http_ssl import verified_urlopen

        def opener(u, data, hdrs, to):  # noqa: E306
            req = urllib.request.Request(u, data=data, headers=hdrs, method="POST")
            try:
                # verified_urlopen (see localm/http_ssl.py): this is the very path
                # the setup-llama failure message tells the user to use to report
                # it, so it must work under the same conditions setup-llama does.
                with verified_urlopen(req, timeout=to) as resp:
                    return resp.status, resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    # The error body is best-effort extra context; if it cannot be
                    # read we still raise the failure below (with an empty detail),
                    # so the rejection is never hidden.
                    pass
                if e.code == 429:
                    raise RateLimitedError(
                        _retry_after_from(getattr(e, "headers", None), detail),
                        reason=detail) from None
                raise LocalmError(
                    "the bug-report server rejected the upload",
                    reason=f"HTTP {e.code}: {detail}".strip(),
                    stage="server_rejected",
                    hint=(f"The bug-report server received the report but rejected it "
                          f"(HTTP {e.code}). This is likely a temporary server-side "
                          f"issue, not your connection.")) from None
            except (urllib.error.URLError, OSError) as e:
                stage, hint = _classify_url_error(e)
                raise LocalmError("could not reach the bug-report server",
                                  reason=str(getattr(e, "reason", e)),
                                  stage=stage, hint=hint) from None

    status, raw = opener(url, payload, headers, timeout)
    if int(status) == 429:
        raise RateLimitedError(_retry_after_from(None, raw), reason=str(raw)[:300])
    if not (200 <= int(status) < 300):
        raise LocalmError(
            "the bug-report server rejected the upload",
            reason=f"HTTP {status}: {raw[:300]}".strip(),
            stage="server_rejected",
            hint=(f"The bug-report server rejected the report (HTTP {status}). This is "
                  f"likely a temporary server-side issue, not your connection."))
    try:
        return _json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {"raw": raw[:300]}
