# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only tracker for the bug reports this install filed.

Each report filed through the proxy is remembered by issue number and title
(:func:`record_filed`); :func:`filed_status` looks those numbers up one at a time
through the localm proxy (the same Worker that files bug reports), so a tester
can see whether a bug they filed is acknowledged or fixed. The repo's other
issues are never listed, and no request carries anything that identifies the
install. The lookups together take at most ``FILED_STATUS_BUDGET_S``.

The record is ``<data dir>/bug-reports/filed-issues.json``. In privacy mode
nothing is written: the numbers are kept in memory for the running process only.
The proxy uses the server-side Issues token (read-only here); no GitHub account
is needed. Failures raise :class:`~localm.bugreport.LocalmError`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ISSUES_URL = "https://github.com/Matlan1/localm/issues"
MAX_RECORDED = 100
MAX_LISTED = 30
LOOKUP_TIMEOUT_S = 15.0
# Must stay at or above LOOKUP_TIMEOUT_S so the first lookup gets its full timeout.
FILED_STATUS_BUDGET_S = 30.0
NOT_CHECKED = "not checked (out of time)"
_RECORD_NAME = "filed-issues.json"
_LOCK = threading.Lock()
_SESSION_ONLY: list = []


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


def _refuse_when_network_off() -> None:
    """Raise LocalmError when ``net_mode`` is ``off``. An unreadable config
    resolves to ``off`` (``netpolicy.network_mode``), so it refuses too."""
    from localm.bugreport import LocalmError
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise LocalmError("network access is disabled (net_mode = off)",
                          reason="the issues list needs the network; set "
                                 "net_mode to ask or allow to see it")


def get_issue(number: int, *, opener=None, timeout: float = LOOKUP_TIMEOUT_S) -> Optional[dict]:
    """Return one issue's trimmed dict, or None if not found. *timeout* is seconds
    per socket operation, so one call can take several times that. Raises
    LocalmError when not configured, when ``net_mode`` is ``off`` (no request is
    made), or on a transport failure."""
    from localm import _proxy
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the issues tracker is not configured",
                          reason="set bugreport_upload_url to enable it")
    try:
        n = int(number)
    except (TypeError, ValueError):
        raise LocalmError("bad issue number", reason=str(number)) from None
    _refuse_when_network_off()
    data = _proxy.request(base, f"/issues?number={n}", token=token, opener=opener, timeout=timeout)
    issue = data.get("issue") if isinstance(data, dict) else None
    return issue if isinstance(issue, dict) else None


def _record_path() -> Path:
    from localm.config import home_dir
    return home_dir() / "bug-reports" / _RECORD_NAME


def _privacy_mode() -> bool:
    """True when nothing may be written to disk. An unresolvable mode counts as
    privacy mode."""
    try:
        from localm.audit import automatic_write_allowed
        return not automatic_write_allowed("server")
    except Exception:
        return True


def persists() -> bool:
    """True when filed-issue numbers are written to disk; False in privacy mode,
    where they are kept in memory for the running process only."""
    return not _privacy_mode()


def _clean_entry(entry) -> Optional[dict]:
    if not isinstance(entry, dict):
        return None
    number = entry.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    return {"number": number, "title": str(entry.get("title") or "")[:200]}


def _read_disk() -> list:
    """The recorded entries on disk, oldest first. A missing file is an empty
    list; an unreadable or malformed one, or one with no list under ``issues``, is
    logged at WARNING and read as empty."""
    path = _record_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("could not read %s: %s", path, e)
        return []
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as e:
        logger.warning("%s is not valid JSON: %s", path, e)
        return []
    items = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(items, list):
        logger.warning("%s has no list of issues", path)
        return []
    return [c for c in (_clean_entry(i) for i in items) if c]


def recorded() -> list:
    """The issues this install filed, oldest first: the on-disk record plus any
    kept in memory (privacy mode). Never raises."""
    with _LOCK:
        try:
            disk = _read_disk()
        except Exception as e:
            logger.warning("could not read the filed-issues record: %s", e)
            disk = []
        seen = {e["number"] for e in disk}
        return disk + [e for e in _SESSION_ONLY if e["number"] not in seen]


def record_filed(number, title: str = "") -> bool:
    """Remember that this install filed issue *number*. Returns True when it is
    remembered (on disk, or in memory only in privacy mode), False for an invalid
    number or a failed write (logged at WARNING). Never raises."""
    entry = _clean_entry({"number": number, "title": title})
    if entry is None:
        return False
    with _LOCK:
        try:
            if _privacy_mode():
                if all(e["number"] != entry["number"] for e in _SESSION_ONLY):
                    _SESSION_ONLY.append(entry)
                    del _SESSION_ONLY[:-MAX_RECORDED]
                return True
            items = [e for e in _read_disk() if e["number"] != entry["number"]]
            items.append(entry)
            path = _record_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            from localm.config import atomic_write_private
            atomic_write_private(path, json.dumps({"issues": items[-MAX_RECORDED:]}))
            return True
        except Exception as e:
            logger.warning("could not record filed issue #%s: %s", entry["number"], e)
            return False


def _lookup_row(entry: dict, opener, timeout: float) -> dict:
    """The ``filed_status`` row of one recorded *entry*. Never raises."""
    from localm.bugreport import LocalmError
    row = {"number": entry["number"], "title": entry["title"]}
    try:
        issue = get_issue(entry["number"], opener=opener, timeout=timeout)
    except LocalmError as e:
        row["error"] = e.summary
    except Exception as e:
        logger.warning("looking up issue #%s failed: %s: %s", entry["number"],
                       type(e).__name__, e)
        row["error"] = "the lookup failed"
    else:
        if issue is None:
            row["error"] = "not found"
        else:
            row.update({k: issue.get(k) for k in ("title", "state", "closed_at", "html_url")
                        if issue.get(k) is not None})
    return row


def filed_status(*, opener=None) -> list:
    """Look up each recorded issue, newest first, one proxy request per number,
    and return within ``FILED_STATUS_BUDGET_S`` seconds whatever the network does.

    Each row is ``{number, title, state, closed_at, html_url}`` from the proxy;
    a lookup that fails or finds nothing keeps ``number`` and the recorded
    ``title`` and carries ``error`` text instead of a state, and so does a row the
    budget did not reach (``NOT_CHECKED``). A lookup still running when the budget
    ends continues on a daemon thread until its socket times out and its answer is
    discarded. Never raises."""
    entries = list(reversed(recorded()[-MAX_LISTED:]))
    rows = [{"number": e["number"], "title": e["title"], "error": NOT_CHECKED} for e in entries]
    budget = FILED_STATUS_BUDGET_S
    deadline = time.monotonic() + budget

    def look_up() -> None:
        for i, entry in enumerate(entries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            rows[i] = _lookup_row(entry, opener, min(LOOKUP_TIMEOUT_S, remaining))

    worker = threading.Thread(target=look_up, name="issues-lookup", daemon=True)
    worker.start()
    worker.join(budget)
    return list(rows)
