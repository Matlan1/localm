#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone bug reporter for when localm will not start.

This is the fallback the ``report-issue`` entry (and the setup scripts' failure
paths) use when the normal ``localm bug-report`` cannot run because the install is
broken or not there yet. It depends ONLY on the Python standard library and on
sitting inside a localm clone: it reads the bug-report proxy URL + token straight
out of ``localm/config.py`` (no duplicated secret, picks up any rotation), collects
a minimal, username-scrubbed diagnostic snapshot, asks for a one-line description,
PREVIEWS the exact text alongside the destination host, and only files it - as an
account-less GitHub issue via the proxy - after the user confirms. Only http/https
endpoints are accepted, so an overridden proxy is visible rather than silent. A
failed or declined send never reports success: it saves the report to a file and
points at the maintainer email.

What it NEVER collects: environment variables (which can hold a key), config
secrets, or chat/transcript content. Only OS / arch / Python, a few version
markers, whether a venv exists, and the tail of the most recent log - all
username-scrubbed before it is shown or sent.

Run it directly:  python scripts/report_issue.py
Optional flags used by the setup wiring:
  --summary TEXT    one-line title (else you are prompted)
  --detail  TEXT    prefilled "what happened" (e.g. the failing setup step)
  --log     PATH    a specific log file to attach the tail of (else newest found)
  --yes             SEND immediately, skipping the confirm prompt (the preview
                    still prints to the console first, but does not gate the
                    send); for scripted/automated sends
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Maintainer contact for bug reports; kept in sync by hand with localm/bugreport.py.
MAINTAINER_EMAIL = "theilige@gmail.com"


# --------------------------------------------------------------------------- #
#  Proxy config: read the shipped constants from localm/config.py             #
# --------------------------------------------------------------------------- #

#: Schemes accepted for a bug-report endpoint.
ALLOWED_ENDPOINT_SCHEMES = ("http", "https")


def endpoint_is_allowed(url: str | None) -> bool:
    """True if *url* is a POST-able http(s) endpoint with a host.

    The URL's only producers are the process environment and localm's own source
    (read_proxy), so this makes the accepted scheme explicit rather than trusting
    whatever urlopen() would dispatch on.
    """
    if not url:
        return False
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:      # malformed authority, e.g. a bad IPv6 literal
        return False
    return parts.scheme.lower() in ALLOWED_ENDPOINT_SCHEMES and bool(parts.netloc)


def endpoint_host(url: str) -> str:
    """The 'host:port' of *url*, for showing the user where a report is going."""
    try:
        return urllib.parse.urlsplit(url.strip()).netloc or url
    except ValueError:
        return url


def read_proxy(config_path: Path | None = None) -> tuple:
    """(url, token) for the bug-report proxy. LOCALM_BUGREPORT_URL /
    LOCALM_BUGREPORT_TOKEN override per value (a custom proxy, or a test double);
    otherwise the shipped constants are read as TEXT from localm/config.py so the
    account-less GitHub-issue channel works with NO import of a possibly-broken
    localm/venv. Returns (None, None) if nothing is configured or readable."""
    import os
    url_cfg = token_cfg = None
    if config_path is None:
        config_path = REPO_ROOT / "localm" / "config.py"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if text:
        def _grab(key: str) -> str | None:
            m = re.search(
                r'["\']' + re.escape(key) + r'["\']\s*:\s*["\']([^"\']+)["\']', text)
            return m.group(1) if m else None
        url_cfg, token_cfg = _grab("bugreport_upload_url"), _grab("bugreport_upload_token")
    url = os.environ.get("LOCALM_BUGREPORT_URL") or url_cfg
    token = os.environ.get("LOCALM_BUGREPORT_TOKEN") or token_cfg
    if url:
        url = url.strip()
        if not endpoint_is_allowed(url):
            # Drop a disallowed endpoint and say so; nothing is sent.
            print(f"  Ignoring bug-report endpoint {scrub(str(url))!r}: only "
                  f"{' and '.join(s + '://' for s in ALLOWED_ENDPOINT_SCHEMES)} "
                  f"are allowed. Nothing will be sent.")
            url = None
    return url, token


# --------------------------------------------------------------------------- #
#  Privacy scrub (mirrors localm/bugreport.py _scrub_home / _scrub_secrets)    #
# --------------------------------------------------------------------------- #

_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")
_APIKEY_RE = re.compile(r"(?i)\b(?:sk|localm[_-]sk)-[A-Za-z0-9._\-]{12,}")

# Credential-named URL query parameters and pasted header lines. Kept
# byte-identical to the counterparts in localm/bugreport.py.
_QUERY_SECRET_RE = re.compile(
    r"(?i)((?:"
    # 1. Immediately after a query delimiter.
    r"(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth"
    r"|access[_-]?token|sig|signature)"
    # 2. Anywhere else, unprefixed.
    r"|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd"
    r"|access[_-]?token|signature)"
    # 3. Anywhere else, prefixed by a separator that must stay MANDATORY:
    #    [_-] and never [_-]?. The names begin with an alnum, so an optional
    #    separator lets the prefix split more than one way and makes matching
    #    superlinear on hostile pasted text.
    r"|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd"
    r"|access[_-]?token|signature|key|auth|sig)"
    r")=)"
    # Leave a value that cannot be a secret alone (true/false/none/0/1/...);
    # the literal must be the whole value, closing markup aside.
    r"(?![\"']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])"
    r"[`\"'\)\]\}]{0,4}(?:[\s&#]|$))"
    r"(?:\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^&\s#\"'\)\]\}]*)"
)
_HEADER_SECRET_RE = re.compile(
    r"(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token|authorization)\s*:\s*)"
    r"(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+"
)


def scrub(text: str) -> str:
    """Strip the account name from any path and any obvious credential from free
    text before it is shown or sent. A privacy scrub must fail safe: if it cannot
    run it must NOT pass the text through as if scrubbed, so the home-root strip
    below ALWAYS runs (it never depends on Path.home succeeding)."""
    if not text:
        return text
    # Replace the real home dir first (both separator forms), when resolvable.
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    if home:
        variants = {home, home.replace("\\", "/")}
        if sys.platform == "win32":
            for v in variants:
                text = re.sub(re.escape(v), "~", text, flags=re.IGNORECASE)
        else:
            for v in variants:
                text = text.replace(v, "~")
    # Strip the user segment under any common home root, not only Path.home().
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    text = re.sub(r"([A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)[^\\/\r\n]+",
                  r"\1<redacted>", text, flags=flags)
    # Strip user:pass@ credentials from any URL-ish value.
    text = re.sub(r"(://)[^/@\s]+@", r"\1<redacted>@", text)
    # Credential-named query params and header lines.
    text = _QUERY_SECRET_RE.sub(r"\1<redacted>", text)
    text = _HEADER_SECRET_RE.sub(r"\1<redacted>", text)
    # Bearer tokens and API keys anywhere in the text.
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    text = _APIKEY_RE.sub("<redacted>", text)
    return text


# --------------------------------------------------------------------------- #
#  Minimal, safe diagnostics                                                   #
# --------------------------------------------------------------------------- #

def _venv_python() -> Path | None:
    """The clone's venv interpreter, if it exists (both OS layouts)."""
    cands = (REPO_ROOT / ".venv" / "Scripts" / "python.exe",
             REPO_ROOT / ".venv" / "bin" / "python3",
             REPO_ROOT / ".venv" / "bin" / "python")
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def _newest_log(explicit: str | None = None) -> tuple:
    """(path, scrubbed tail) of a relevant log, or (None, ''). Looks at an explicit
    path first, else the newest *.log under the clone's home/logs. Never raises."""
    candidates: list[Path] = []
    if explicit:
        p = Path(explicit)
        if p.is_file():
            candidates = [p]
    if not candidates:
        for base in (REPO_ROOT / "home" / "logs", REPO_ROOT / "home"):
            try:
                if base.is_dir():
                    candidates += sorted(base.glob("*.log"),
                                         key=lambda p: p.stat().st_mtime,
                                         reverse=True)
            except OSError:
                continue
    for p in candidates:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-120:]).strip()
            if tail:
                return p, scrub(tail)[-4000:]
        except OSError:
            continue
    return None, ""


def collect_diagnostics() -> dict:
    """A small, safe environment snapshot. Never raises."""
    diag: dict = {}
    try:
        diag["platform"] = platform.platform()
        diag["machine"] = platform.machine()
        diag["python"] = platform.python_version()
    except Exception:
        pass
    try:
        vp = _venv_python()
        diag["venv_present"] = vp is not None
    except Exception:
        diag["venv_present"] = False
    try:
        vf = REPO_ROOT / "localm" / "_version.py"
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                      vf.read_text(encoding="utf-8")) if vf.is_file() else None
        # _version.py may define read_version() instead of a literal; fall back to
        # the package __init__ constant, else "unknown".
        if not m:
            initf = REPO_ROOT / "localm" / "__init__.py"
            m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                          initf.read_text(encoding="utf-8")) if initf.is_file() else None
        diag["localm_version"] = m.group(1) if m else "unknown"
    except Exception:
        diag["localm_version"] = "unknown"
    return diag


# --------------------------------------------------------------------------- #
#  Report building + sending                                                   #
# --------------------------------------------------------------------------- #

def build_body(summary: str, description: str, diag: dict,
               log_path, log_tail: str) -> str:
    """Render the editable markdown report (already-scrubbed inputs)."""
    lines = [
        f"# localm bug report: {summary}",
        "",
        "## What happened",
        (description.strip() or "(no description given)"),
        "",
        "## How it was filed",
        "- Filed via the standalone report-issue tool (localm could not start, so "
        "the normal in-app reporter was unavailable).",
        "",
        "## Environment",
    ]
    label = {"localm_version": "localm", "python": "Python", "platform": "OS",
             "machine": "Arch", "venv_present": "Venv present"}
    for key, name in label.items():
        if key in diag:
            val = diag[key]
            if isinstance(val, bool):
                val = "yes" if val else "no"
            lines.append(f"- {name}: {val}")
    if log_tail:
        where = scrub(str(log_path)) if log_path else "log"
        lines += ["", f"## Recent log tail ({where})", "```", log_tail, "```"]
    lines += ["", "---",
              f"Sent to the localm maintainer ({MAINTAINER_EMAIL}). "
              "You can edit anything above before sending."]
    return "\n".join(lines)


def post_report(url: str, token: str | None, title: str, body: str,
                *, timeout: float = 15.0, opener=None) -> dict:
    """POST the report to the proxy and return its JSON response (e.g.
    {"url": "<issue url>"}). Raises RuntimeError on any non-2xx / network error so a
    failed send is never mistaken for success. *opener* is injectable for tests."""
    # Reject a non-http(s) endpoint at the sink, before anything is built or opened.
    if not endpoint_is_allowed(url):
        raise RuntimeError(
            f"refusing to send to {scrub(str(url))!r}: only "
            f"{' and '.join(s + '://' for s in ALLOWED_ENDPOINT_SCHEMES)} "
            f"endpoints are allowed")
    payload = json.dumps({"title": (title or "localm bug report")[:200],
                          "body": body or ""}).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "localm-report-issue"}
    if token:
        headers["X-Localm-Token"] = token

    if opener is None:
        def opener(u, data, hdrs, to):  # noqa: E306
            req = urllib.request.Request(u, data=data, headers=hdrs, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=to) as resp:
                    return resp.status, resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {e.code}: {detail}".strip())
            except (urllib.error.URLError, OSError) as e:
                raise RuntimeError(f"could not reach the server: "
                                   f"{getattr(e, 'reason', e)}")

    status, raw = opener(url, payload, headers, timeout)
    if not (200 <= int(status) < 300):
        raise RuntimeError(f"HTTP {status}: {str(raw)[:300]}".strip())
    try:
        return json.loads(raw) if raw and raw.strip() else {}
    except ValueError:
        return {"raw": str(raw)[:300]}


def save_report(body: str, when: str) -> Path | None:
    """Write the report next to the clone so the user can always retrieve it.
    Returns the path, or None on failure. *when* is caller-supplied (test-injectable)."""
    base = REPO_ROOT / "home" / "bug-reports"
    try:
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"bug-{when}.md"
        path.write_text(body, encoding="utf-8")
        try:
            path.chmod(0o600)  # no-op on Windows; harmless if unsupported
        except OSError:
            pass
        return path
    except OSError:
        return None


# --------------------------------------------------------------------------- #
#  CLI orchestration                                                           #
# --------------------------------------------------------------------------- #

def _timestamp() -> str:
    import time
    return time.strftime("%Y%m%d-%H%M%S")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("--summary", default="")
    ap.add_argument("--detail", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())

    print()
    print("  localm - report a problem")
    print("  (localm could not start, so this standalone reporter files the issue.)")
    print()

    summary = args.summary.strip()
    description = args.detail.strip()
    if interactive and not summary:
        try:
            summary = input("  One line - what went wrong? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled - nothing sent.")
            return 1
    if interactive and not description:
        try:
            description = input("  Any more detail (optional, Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            description = ""
    summary = summary or "localm would not start"

    diag = collect_diagnostics()
    log_path, log_tail = _newest_log(args.log or None)
    body = build_body(scrub(summary), scrub(description), diag, log_path, log_tail)

    # Resolved before the preview so the destination is shown in it.
    url, token = read_proxy()

    # Preview exactly what will be sent.
    print()
    print("  ----- this is exactly what will be sent (edit later if you prefer) -----")
    for line in body.splitlines():
        print("  " + line)
    print("  -----------------------------------------------------------------------")
    if url:
        print(f"  Destination: {endpoint_host(url)} "
              f"(over {urllib.parse.urlsplit(url).scheme})")
    else:
        print("  Destination: none configured - the report will only be saved locally.")
    print()

    when = _timestamp()

    # Confirm. Not a tty and not --yes: save only.
    do_send = args.yes
    if not do_send:
        if not interactive:
            path = save_report(body, when)
            where = str(path) if path else "the text above"
            print(f"  Not a terminal, so nothing was sent. The report is saved at "
                  f"{where}.")
            print(f"  Send it by running report-issue again, or email {MAINTAINER_EMAIL}.")
            return 0
        try:
            ans = input("  Send this to the maintainer now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        do_send = not ans.startswith("n")

    if not do_send:
        path = save_report(body, when)
        where = str(path) if path else "the text above"
        print(f"  Not sent. Saved at {where} - email it to {MAINTAINER_EMAIL} if you "
              "change your mind.")
        return 0

    if not url:
        path = save_report(body, when)
        where = str(path) if path else "the text above"
        print(f"  No bug-report endpoint is configured. Saved at {where}; email it to "
              f"{MAINTAINER_EMAIL}.")
        return 1

    try:
        # Scrub the title too; it becomes a public issue title.
        res = post_report(url, token, scrub(summary), body)
    except Exception as e:
        # A failed send is not reported as success.
        path = save_report(body, when)
        where = str(path) if path else "the text above"
        print(f"  Could not send it ({e}). The report is saved at {where} - email it "
              f"to {MAINTAINER_EMAIL} instead.")
        return 1

    link = res.get("url") if isinstance(res, dict) else None
    print("  Sent to the maintainer. Thank you!"
          + (f" Tracking issue: {link}" if link else ""))
    # Keep a local copy.
    save_report(body, when)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
