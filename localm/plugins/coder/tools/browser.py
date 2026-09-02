# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser tools: thin wrappers that delegate to localm.browser.session.

Unlike the one-shot tools around them, these act on a STATEFUL, long-lived
browser. One browser session belongs to one coder session, keyed on the agent's
``job_owner`` (process-wide, stable across a checkpoint resume, and inherited by
a spawned sub-agent, so a child drives the parent's browser rather than opening
a second one).

Every tool re-checks ``_session.browser_enabled`` itself. The agent already
removes these tools from the model's schema when the capability is off; this
second check means a future refactor of that narrowing cannot silently open the
browser back up.

Output from a visited page is attacker-influenceable, so every tool that returns
page content is registered with ``untrusted_output=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import ToolResult, _confine

#: Live browser sessions by agent job_owner.
_OWNED: dict = {}


def _disabled() -> ToolResult:
    return ToolResult.error(
        "Browser automation is not enabled for this session. It needs the "
        "'browser' capability on the key and the browser setting switched on.")


def _enabled(session) -> bool:
    return bool(getattr(session, "browser_enabled", False))


def _owner_of(session) -> str:
    return str(getattr(session, "job_owner", "") or "default")


def _existing(session):
    from localm.browser import session as bsession
    sid = _OWNED.get(_owner_of(session))
    return bsession.get(sid) if sid else None


def _open_session(session, *, headless: bool = True):
    """Return this coder session's browser, starting one if needed."""
    from localm.browser import session as bsession
    live = _existing(session)
    if live is not None:
        return live
    owner = _owner_of(session)
    sid = "coder-" + owner
    cfg = _browser_config()
    live = bsession.BrowserSession(
        sid,
        headless=headless and cfg["headless"],
        extra_deny=cfg["deny"],
        extra_allow=cfg["allow"],
        engine=cfg["engine"],
    )
    live.start()
    bsession.register(live)
    _OWNED[owner] = sid
    return live


def _browser_config() -> dict:
    """The browser settings, with the custom domain rules applied only when
    their opt-in is on."""
    from localm.config import load_config
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    custom = bool(cfg.get("browser_custom_domain_rules", False))
    engine = str(cfg.get("browser_engine", "bundled") or "bundled")
    return {
        "headless": bool(cfg.get("browser_headless", True)),
        "engine": engine if engine in ("bundled", "system") else "bundled",
        "deny": list(cfg.get("browser_deny") or []) if custom else [],
        "allow": list(cfg.get("browser_allow") or []) if custom else [],
    }


def tool_browser_navigate(cwd: Path, url: str, _session=None) -> ToolResult:
    """Open *url* in this session's browser, starting it if needed."""
    if not _enabled(_session):
        return _disabled()
    try:
        browser = _open_session(_session)
    except Exception as exc:                        # noqa: BLE001
        return ToolResult.error(str(exc))
    res = browser.navigate(url)
    if res.get("refused"):
        return ToolResult.error(
            "Refused by the network policy: " + res["refused"])
    if not res.get("ok"):
        return ToolResult.error(
            "Could not load " + url + ": " + str(res.get("error", "unknown")))
    return ToolResult.success(
        "Loaded " + str(res.get("url")) + "\nTitle: " + str(res.get("title"))
        + "\nStatus: " + str(res.get("status")),
        summary="opened " + str(res.get("url")))


def tool_browser_read(cwd: Path, max_chars: int = 8000, _session=None) -> ToolResult:
    """Read the visible text of the current page."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    text = browser.read_text(max_chars=int(max_chars))
    return ToolResult.success(text, summary="read " + str(len(text)) + " chars")


def tool_browser_click(cwd: Path, selector: str, _session=None) -> ToolResult:
    """Click the first element matching a CSS selector."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    res = browser.click(selector)
    if not res.get("ok"):
        return ToolResult.error("Could not click " + selector + ": "
                                + str(res.get("error")))
    return ToolResult.success("Clicked " + selector + "\nNow at: "
                              + str(res.get("url")),
                              summary="clicked " + selector)


def tool_browser_fill(cwd: Path, selector: str, value: str,
                      _session=None) -> ToolResult:
    """Type *value* into the form field matching a CSS selector."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    res = browser.fill(selector, value)
    if not res.get("ok"):
        return ToolResult.error("Could not fill " + selector + ": "
                                + str(res.get("error")))
    return ToolResult.success("Filled " + selector, summary="filled " + selector)


def tool_browser_screenshot(cwd: Path, output_path: str = "screenshot.png",
                            full_page: bool = False,
                            _session=None) -> ToolResult:
    """Save a PNG screenshot of the current page into the project."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    try:
        out = _confine(cwd, output_path)
    except PermissionError as exc:
        return ToolResult.error(str(exc))
    try:
        data = browser.screenshot(full_page=bool(full_page))
    except Exception as exc:                        # noqa: BLE001
        return ToolResult.error("Screenshot failed: " + str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    rel = out.relative_to(cwd) if out.is_relative_to(cwd) else out
    return ToolResult.success("Saved screenshot to " + str(rel),
                              summary="screenshot " + str(rel))


def tool_browser_console(cwd: Path, _session=None) -> ToolResult:
    """Read the console messages the page has logged."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    msgs = browser.console_messages()
    if not msgs:
        return ToolResult.success("(no console messages)", summary="0 messages")
    lines = [str(m.get("type", "log")) + ": " + str(m.get("text", ""))
             for m in msgs]
    return ToolResult.success("\n".join(lines),
                              summary=str(len(lines)) + " console messages")


def tool_browser_network(cwd: Path, _session=None) -> ToolResult:
    """List the requests the page made, and the ones the policy refused."""
    if not _enabled(_session):
        return _disabled()
    browser = _existing(_session)
    if browser is None:
        return ToolResult.error("No browser session is open. Navigate first.")
    allowed = browser.allowed_requests()
    blocked = browser.blocked_requests()
    lines = ["Allowed (" + str(len(allowed)) + "):"]
    lines += ["  " + u for u in allowed[:100]]
    lines.append("Refused (" + str(len(blocked)) + "):")
    lines += ["  " + b["url"] + "  -  " + b["reason"] for b in blocked[:100]]
    return ToolResult.success(
        "\n".join(lines),
        summary=str(len(allowed)) + " allowed, " + str(len(blocked)) + " refused")


def tool_browser_close(cwd: Path, _session=None) -> ToolResult:
    """Close this session's browser and forget its state."""
    if not _enabled(_session):
        return _disabled()
    from localm.browser import session as bsession
    owner = _owner_of(_session)
    sid = _OWNED.pop(owner, None)
    if sid is None:
        return ToolResult.success("No browser session was open.",
                                  summary="nothing to close")
    bsession.close(sid)
    return ToolResult.success("Closed the browser session.", summary="closed")


def close_for_owner(owner: str) -> bool:
    """Close the browser belonging to *owner*, if any. Used at session teardown
    so a browser never outlives the coder session that opened it."""
    from localm.browser import session as bsession
    sid = _OWNED.pop(str(owner), None)
    if sid is None:
        return False
    return bsession.close(sid)


def owned_session_id(owner: str) -> Optional[str]:
    """The browser session id belonging to *owner*, or None."""
    return _OWNED.get(str(owner))
