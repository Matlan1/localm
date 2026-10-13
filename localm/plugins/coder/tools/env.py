# SPDX-License-Identifier: AGPL-3.0-or-later
"""Environment-reading tool: reads .env + the process environment with
secret-looking values, home paths and the account name redacted."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from localm.bugreport.scrub import _scrub_secrets

from .base import ToolResult, _confine

# Env var names (substring match, case-insensitive) whose values are secrets
_SECRET_MARKERS = (
    "key", "token", "secret", "password", "passwd", "passphrase", "credential",
    "auth", "cookie", "private", "cert", "signature", "dsn", "mnemonic",
    "phrase", "wallet", "webhook",
)

# Env var name segments (split on non-alphanumerics, case-insensitive) whose
# values are secrets: GITHUB_PAT, STRIPE_SK.
_SECRET_NAME_SEGMENTS = frozenset({"pat", "sk"})

# Env var names whose values identify the account or the machine.
_IDENTITY_NAMES = frozenset({
    "user", "username", "logname", "userdomain", "userdomain_roamingprofile",
    "computername", "hostname", "logonserver", "userdnsdomain", "homepath",
    "mail", "sudo_user",
})

# Credential value shapes recognised under any variable name: GitHub, GitLab,
# Hugging Face, Slack, Stripe, AWS and Google API key formats, JWTs, and
# Discord/Slack webhook URLs.
_TOKEN_VALUE_RE = re.compile(
    r"(?:\b(?:gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|xox[abeoprs]-[A-Za-z0-9\-]{10,}"
    r"|[rsp]k_(?:live|test)_[A-Za-z0-9]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_\-]{35}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})"
    r"|discord(?:app)?\.com/api/webhooks/"
    r"|hooks\.slack\.com/)")


def _redacted(value: str) -> str:
    return f"<redacted, {len(value)} chars>"


def _account_names() -> frozenset[str]:
    """The account names to remove from values: the home directory's last
    segment and the USERNAME / USER / LOGNAME values, at least 2 characters."""
    names = set()
    try:
        names.add(Path.home().name)
    except Exception:
        pass
    for var in ("USERNAME", "USER", "LOGNAME"):
        names.add(os.environ.get(var, ""))
    return frozenset(n for n in names if len(n) >= 2)


def _strip_account(value: str, accounts: frozenset[str]) -> str:
    """*value* with each of *accounts* replaced by ``<user>`` where it is a whole
    path segment (bounded by a separator or the ends of the value);
    case-insensitive on Windows."""
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    for account in accounts:
        value = re.sub(r"(?:(?<=[\\/])|^)" + re.escape(account) + r"(?=[\\/]|$)",
                       "<user>", value, flags=flags)
    return value


def _redact_env_value(name: str, value: str,
                      accounts: frozenset[str] = frozenset()) -> tuple[str, bool]:
    """Return ``(shown_value, redacted)`` for one variable.

    The whole value is replaced with a length marker when the NAME looks
    secret (a marker substring, a ``_PAT``/``_SK`` segment) or names the
    account or machine, or when the VALUE holds a recognised credential
    format. Otherwise the value is passed through ``_scrub_secrets``, which
    replaces URL ``user:pass@`` credentials, credential-named query
    parameters, bearer tokens, ``sk-`` API keys, email addresses and the home
    directory / account name in paths, then each of *accounts* is replaced by
    ``<user>`` where it is a whole path segment. *redacted* is True when the
    shown value differs from the original."""
    lowered = name.lower()
    if (any(marker in lowered for marker in _SECRET_MARKERS)
            or _SECRET_NAME_SEGMENTS.intersection(re.split(r"[^a-z0-9]+", lowered))
            or lowered in _IDENTITY_NAMES
            or _TOKEN_VALUE_RE.search(value)):
        return _redacted(value), True
    shown = _strip_account(_scrub_secrets(value), accounts)
    return shown, shown != value


def tool_read_env(cwd: Path, path: str = "") -> ToolResult:
    """
    Read environment configuration with secrets stripped.

    Without arguments, reads the project's ``.env`` file (if present) and the
    active process environment. With ``path``, reads that env-format file
    instead. See ``_redact_env_value`` for what is redacted; a secret in an
    unrecognised format under an ordinary variable name is shown as is. The
    summary reports how many values were redacted.
    """
    lines: list[str] = []
    n_redacted = 0
    accounts = _account_names()

    def _add(name: str, value: str) -> None:
        nonlocal n_redacted
        shown, redacted = _redact_env_value(name, value, accounts)
        n_redacted += redacted
        lines.append(f"{name}={shown}")

    try:
        env_file = _confine(cwd, path) if path else (cwd / ".env")
    except PermissionError as e:
        return ToolResult.error(str(e))
    if env_file.is_file():
        lines.append(f"# {env_file.name}")
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            name = name.strip().removeprefix("export ").strip()
            _add(name, value.strip())
    elif path:
        return ToolResult.error(f"Env file not found: {path}")

    if not path:
        lines.append("")
        lines.append("# process environment")
        for name in sorted(os.environ):
            _add(name, os.environ[name])

    output = "\n".join(lines).strip() or "(no environment variables found)"
    n_vars = sum(1 for l in lines if "=" in l)
    return ToolResult.success(
        output, summary=f"read_env ({n_vars} vars, {n_redacted} values redacted)")
