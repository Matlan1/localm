# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only issues tracker for testers.

Lists the repo's open/closed issues through the localm proxy (the same Worker
that files bug reports), so a tester can see whether a bug they filed is
acknowledged or fixed.

The proxy uses the server-side Issues token (read-only here); no GitHub account
is needed. Failures raise :class:`~localm.bugreport.LocalmError`.
"""

from __future__ import annotations

from typing import Optional


def endpoint() -> tuple:
    """(base_url, token) for the issues list: reuses the bug-report proxy config
    (one Worker hosts report + issues + update). (None, None) when not configured.
    Never raises."""
    try:
        from localm.config import load_config
        cfg = load_config()
        url = (cfg.get("bugreport_upload_url") or "").strip() or None
        token = (cfg.get("bugreport_upload_token") or "").strip() or None
        return url, token
    except Exception:
        return None, None


def available() -> bool:
    """True when an issues endpoint is configured (GUI/CLI use this to decide
    whether to offer the issues view)."""
    return endpoint()[0] is not None


def list_issues(state: str = "all", *, per_page: int = 30, opener=None) -> list:
    """Return a list of trimmed issue dicts (``{number, title, state, created_at,
    closed_at, html_url, labels}``), newest-updated first. *state* is open/closed/all.
    Raises LocalmError when not configured or the proxy fails."""
    from localm import _proxy
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the issues tracker is not configured",
                          reason="set bugreport_upload_url to enable it")
    st = state if state in ("open", "closed", "all") else "all"
    per = max(1, min(int(per_page or 30), 100))
    data = _proxy.request(base, f"/issues?state={st}&per_page={per}",
                          token=token, opener=opener)
    issues = data.get("issues") if isinstance(data, dict) else None
    return issues if isinstance(issues, list) else []


def get_issue(number: int, *, opener=None) -> Optional[dict]:
    """Return one issue's trimmed dict, or None if not found. Raises LocalmError on
    a transport failure."""
    from localm import _proxy
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the issues tracker is not configured",
                          reason="set bugreport_upload_url to enable it")
    try:
        n = int(number)
    except (TypeError, ValueError):
        raise LocalmError("bad issue number", reason=str(number))
    data = _proxy.request(base, f"/issues?number={n}", token=token, opener=opener)
    issue = data.get("issue") if isinstance(data, dict) else None
    return issue if isinstance(issue, dict) else None
