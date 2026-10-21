# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redaction applied to the free-form text a report carries: home paths and user
names, URL credentials, credential-named query parameters, header lines and JSON
keys, bearer tokens and API keys, email addresses, and machine names.
"""

from __future__ import annotations

import os
import re
import socket

from localm import pathscrub
from localm.bugreport._common import MAINTAINER_EMAIL


# The home/username policy lives in localm.pathscrub, shared with the
# API-response scrubber (embedder status, /debug/stacks). It replaces
# Path.home() with "~" in both separator forms (case-insensitively on Windows),
# and ALWAYS applies the known-home-root username strip as a backstop, so a path
# that is not exactly Path.home() - a different account - is caught too.
#
# A report keeps the INSTALL dir, which pathscrub.scrub_paths would drop.
# Responses to a lower-privileged caller use scrub_paths instead.
_scrub_home = pathscrub.scrub_user_paths


def _scrub_url_creds(text: str) -> str:
    """Strip ``user:pass@`` credentials from any URL-ish value before it goes in a
    report (a SearXNG / ComfyUI / reviewer URL can carry an inline secret)."""
    if not text:
        return text
    return re.sub(r"(://)[^/@\s]+@", r"\1<redacted>@", text)


# A credential is at least as often carried as a URL QUERY PARAMETER
# (?api_key=..., ?token=...) as it is via user:pass@ syntax - that is the
# ordinary way ComfyUI/SearXNG/a generic reviewer endpoint carry a key.
# Redaction is by PARAMETER NAME, never by the value's format: an opaque token
# has no reliable shape, while a parameter literally called
# api_key/token/secret/... is a name that can be trusted regardless of what it
# holds. The parameter NAME is kept, so the report still shows THAT the endpoint
# was credentialed and which endpoint it is; only the value goes. Do not widen
# this with value-shape regexes (sk-..., 24-char-opaque, ...): the name is the
# durable signal.
#
# The name may sit ANYWHERE a ``name=value`` pair can appear, not only right
# after a ``?`` or ``&``: a user pasting a .env fragment or a shell line into a
# report writes ``api_key=...`` at the start of a line, and a prefixed name
# (``OPENAI_API_KEY=``, ``pull_token=``) never touches a query delimiter at all.
# Hence three branches, all feeding ONE capture group so the substitution keeps
# the name and drops only the value:
#
#   1. after a literal ?/& - a bare ``?key=`` or ``?sig=`` in a URL;
#   2. anywhere else, UNPREFIXED - only the names that mean a credential and
#      nothing else (api_key, token, secret, password, passwd, pwd, ...);
#   3. anywhere else, PREFIXED (matched from the separator) - the same names
#      PLUS the short generic ones (key, auth, sig), admitted only behind a
#      prefix.
#
# The short generic names belong in branch 3 ONLY. Admitting ``key``/``auth``/
# ``sig`` UNPREFIXED (branch 2) would eat an ordinary ``key=value`` line out of a
# pasted config dump and an ``auth=none`` out of a log; dropping them from branch
# 3 as well would leave ``SECRET_KEY=``, ``PRIVATE_KEY=``, ``APP_KEY=``,
# ``x-auth=`` and ``req_sig=`` in the clear. Branch 3 alone keeps both:
# ``SECRET_KEY=`` redacts, ``key=`` and ``monkey=`` do not.
_QUERY_SECRET_RE = re.compile(
    r"(?i)((?:"
    # 1. Immediately after a query delimiter.
    r"(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth"
    r"|access[_-]?token|sig|signature)"
    # 2. Anywhere else, UNPREFIXED: only the names that mean a credential
    #    and nothing else.
    r"|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd"
    r"|access[_-]?token|signature)"
    # 3. Anywhere else, PREFIXED - matched from the SEPARATOR rather than
    #    from the prefix, so the whole pattern carries no repetition for a
    #    backtracking engine to walk. The prefix is never consumed and simply
    #    survives outside the match, which leaves the result identical:
    #    ``SECRET_KEY=x`` matches ``_KEY=x`` and reads back as
    #    ``SECRET_KEY=<redacted>``. This is also the branch that admits the
    #    short generic names.
    r"|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd"
    r"|access[_-]?token|signature|key|auth|sig)"
    r")=)"
    # A value that CANNOT be a secret is left alone: a short list of literals no
    # credential is ever equal to, so ``LOCALM_REQUIRE_AUTH=1``,
    # ``require_auth=true``, ``digital_signature=True`` and ``has_token=false``
    # survive into the report and still show whether auth was ON.
    #
    # The literal has to be the WHOLE value, so ``api_key=truesecret123`` and
    # ``api_key=10`` still redact. Closing markup may follow it, because a
    # report carries prose: ``(require_auth=1)`` and ``` `require_auth=1` ```
    # are the same flag. Anything else after that markup is NOT a flag, so
    # ``api_key=1)SECRET`` stays redacted.
    r"(?![\"']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])"
    r"[`\"'\)\]\}]{0,4}(?:[\s&#]|$))"
    # The value may be QUOTED, which is how a .env writes it far more often than
    # not, so the match spans the whole quoted value rather than stopping at the
    # opening quote. An unterminated quote redacts to the NEXT quote on the
    # line, or to the end of the line when there is none, and never across a
    # newline - two different spans, both over-redacting.
    r"(?:\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^&\s#\"'\)\]\}]*)"
)


# The same credential can arrive as a pasted HTTP header line instead of a URL
# (a browser console error, a bundled log tail) - "X-Api-Key: <value>". Matched
# by NAME like the query-string case above, over a small explicit set of
# credential header names rather than a bearer-style catch-all, so an unrelated
# header is never touched, including "Authorization" (value redacted whole,
# leading scheme word such as Bearer/Basic included).
_HEADER_SECRET_RE = re.compile(
    r"(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token|authorization)\s*:\s*)"
    r"(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+"
)


# The same credential arrives JSON-SERIALISED when an object is logged instead
# of a string: the GUI's console capture calls ``JSON.stringify`` on any
# non-string, non-Error argument (static/app/client-log.js), and that ring is
# attached to a share-intended report. The quote before the colon defeats
# _HEADER_SECRET_RE, and there is no ``=`` for _QUERY_SECRET_RE, so
# ``{"api_key":"..."}`` reaches the report in the clear while the unquoted
# ``api_key: ...`` beside it is redacted.
#
# Same policy as the two above and NOT a value-shape widening: the quoted JSON
# key IS the name, so this matches the NAME, keeps it, and drops only the
# value. The short generic names (key, auth, sig) stay prefix-only here for the
# reason branch 3 gives above - a bare ``"key": "gpt-4"`` is ordinary data.
# The value stops at the JSON structural characters rather than at whitespace,
# so a redaction never eats the rest of the object.
_JSON_SECRET_RE = re.compile(
    r"(?i)([\"'](?:"
    r"(?:[A-Za-z0-9]+[_-])?(?:api[_-]?key|token|secret|password|passwd|pwd"
    r"|access[_-]?token|signature)"
    r"|[A-Za-z0-9]+[_-](?:key|auth|sig)"
    r")[\"']\s*:\s*)"
    # Same flag carve-out as _QUERY_SECRET_RE: ``"has_token": false`` survives.
    # The leading ``\s*`` is load-bearing: the ``\s*`` inside the name group can
    # give the separator space back on backtracking, so without it the guard is
    # tested one character too late and matches nothing.
    r"(?!\s*[\"']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])"
    r"[\"']?\s*(?:[,}\]]|$))"
    r"(?:\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^,}\]\s]*)"
)


def _scrub_query_and_header_secrets(text: str) -> str:
    """Redact credential-ish URL query parameters, HTTP header lines and JSON
    object keys, by NAME (see the regexes above). Idempotent: the
    ``<redacted>`` replacement contains none of the delimiters the value
    patterns stop on, so a second pass matches nothing new and leaves an
    already-redacted value unchanged."""
    if not text:
        return text
    text = _QUERY_SECRET_RE.sub(r"\1<redacted>", text)
    text = _HEADER_SECRET_RE.sub(r"\1<redacted>", text)
    text = _JSON_SECRET_RE.sub(r"\1<redacted>", text)
    return text


# Bearer tokens ("Authorization: Bearer <token>") and API keys (OpenAI-style
# sk-..., or a localm key) can appear in a pasted console error, a fetch log
# line, or a mistyped config value. A bug report is share-intended, so they are
# stripped defensively.
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")


_APIKEY_RE = re.compile(r"(?i)\b(?:sk|localm[_-]sk)-[A-Za-z0-9._\-]{12,}")


# Characters that end an address token: ASCII whitespace and controls, C1
# controls, no-break space, the double quote, backtick, ``< > ( ) [ ]`` and
# ``, ; : / \ = ? & *``. Every other character, non-ASCII and the single quote
# included, belongs to the token.
_EMAIL_TOKEN_END = r"\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*"

# A whole token that contains ``local@domain.tld`` (or ``local%40domain.tld``),
# matched from the token's first character to its last. The TLD is at least two
# letters, ASCII or non-ASCII. Same pattern as scripts/report_issue.py and
# scripts/report_issue.ps1. See test_email_pattern_does_not_grow_superlinearly.
_EMAIL_RE = re.compile(
    "(?<![^" + _EMAIL_TOKEN_END + "])"
    "[^" + _EMAIL_TOKEN_END + "]*?"
    "[^@" + _EMAIL_TOKEN_END + "](?:@|%40)"
    "[^@.%+" + _EMAIL_TOKEN_END + "]+"
    r"(?:\.[^@.%+" + _EMAIL_TOKEN_END + "]+)*"
    r"\.[^\x00-\x40\x5b-\x60\x7b-\xa0]{2,}"
    "[^" + _EMAIL_TOKEN_END + "]*"
)


def _email_replacement(m: re.Match) -> str:
    token = m.group(0)
    core = token.lstrip("'")
    lead = token[:len(token) - len(core)]
    core = core.rstrip("'.")
    trail = token[len(lead) + len(core):]
    address = core.replace("%40", "@")
    if address.isascii() and address.lower() == MAINTAINER_EMAIL.lower():
        return token
    return lead + "<redacted-email>" + trail


def _scrub_emails(text: str) -> str:
    """Replace every token that contains an email address with
    ``<redacted-email>``, keeping its leading single quotes and its trailing
    single quotes and periods. A token that is exactly ``MAINTAINER_EMAIL``
    once those are set aside (ASCII, compared case-insensitively, ``%40`` read
    as ``@``) is kept as written. Idempotent: the
    replacement contains no ``@``. A ``user@`` URL credential already rewritten
    to ``<redacted>@`` by ``_scrub_url_creds`` is left as it is."""
    if not text:
        return text
    return _EMAIL_RE.sub(_email_replacement, text)


# The machine name of a UNC path: ``\\NAME\share``, its doubled form
# ``\\\\NAME\\share`` (repr() / JSON) and the ``\\?\UNC\NAME\share`` device form.
# The run of backslashes must start the text, follow whitespace, a quote, a
# backtick or one of ``( [ { = , ; | < *``, or follow a colon that ends a word
# of two or more characters (``path:``), so a doubled separator inside a path
# (``C:\\Users\\x``) is not read as one. ``wsl$``, ``wsl.localhost`` and
# ``localhost`` are kept. Same
# pattern as scripts/report_issue.py, scripts/report_issue.ps1 and
# report-issue.sh. See test_host_patterns_do_not_grow_superlinearly.
_UNC_HOST_RE = re.compile(
    r"(?:(?<![^\s\x22\x27(\[{=,;|\x60<*])|(?<=[A-Za-z0-9_]{2}:))"
    r"(\\{1,2}\\{1,2}(?:\?\\{1,2}UNC\\{1,2})?)"
    r"(?!(?:wsl\$|wsl\.localhost|localhost)(?![A-Za-z0-9._$-]))"
    r"[A-Za-z0-9][A-Za-z0-9._$-]*",
    re.IGNORECASE)

# A host name under a local-network suffix (``nas.local``, ``pc.lan``,
# ``printer.home.arpa``, ``db.internal``, ``box.localdomain``). The suffix is
# kept: ``nas.local`` becomes ``<host>.local``. Not one: a name continuing past
# the suffix (``config.local.json``), a call (``threading.local()``), localm's
# own mDNS name ``localm.local`` and Docker's ``host``/``gateway``/``kubernetes``
# ``.docker.internal``. The same rule as the local-network regex in
# scripts/report_issue.ps1 and report-issue.sh; scripts/report_issue.py has the
# same functions. See
# test_host_patterns_do_not_grow_superlinearly.
_LAN_SUFFIXES = (("home", "arpa"), ("local",), ("lan",), ("internal",), ("localdomain",))
_LAN_KEEP = ("localm.local", "host.docker.internal", "gateway.docker.internal",
             "kubernetes.docker.internal")
_LAN_TOKEN_RE = re.compile(r"[A-Za-z0-9.-]+")


def _lan_boundary(rest: str) -> bool:
    """Whether *rest* (the first two characters after a candidate name inside
    its token) ends the name: empty, or a dot not followed by a letter or
    digit."""
    return not rest or (rest[0] == "." and not rest[1:2].isalnum())


def _lan_token(token: str, after: str) -> str:
    """*token*, a maximal run of letters, digits, dots and hyphens, with a host
    name under a local-network suffix at its start replaced by ``<host>`` (the
    suffix kept). *after* is the character following the token. The longest
    such name wins; a name followed inside the token by a dot and a letter or
    digit, or followed by ``(`` at the token's end, is not one, and a token
    starting with one of ``_LAN_KEEP`` is kept whole."""
    lowered = token.lower()
    for keep in _LAN_KEEP:
        if lowered.startswith(keep) and _lan_boundary(token[len(keep):len(keep) + 2]):
            return token
    parts = token.split(".")
    labels = parts.index("") if "" in parts else len(parts)
    ends = []
    pos = -1
    for part in parts[:labels]:
        pos += len(part) + 1
        ends.append(pos)
    for k in range(labels, 1, -1):
        end = ends[k - 1]
        if end < len(token):
            if not _lan_boundary(token[end:end + 2]):
                continue
        elif after == "(":
            continue
        tail = (parts[k - 2].lower(), parts[k - 1].lower())
        for suffix in _LAN_SUFFIXES:
            if k > len(suffix) and tail[-len(suffix):] == suffix:
                return "<host>." + ".".join(parts[k - len(suffix):k]) + token[end:]
    return token


def _scrub_lan_hosts(text: str) -> str:
    """Every host name under a local-network suffix replaced by ``<host>``
    (``_lan_token``). A token right after an underscore is part of a longer
    identifier and is kept."""
    def replace(match):
        start, end = match.start(), match.end()
        if start and text[start - 1] == "_":
            return match.group(0)
        return _lan_token(match.group(0), text[end:end + 1])
    return _LAN_TOKEN_RE.sub(replace, text)


# Host names never treated as this machine's own: placeholders and default
# names that are also ordinary words in a report.
_OWN_NAME_SKIP = frozenset({"admin", "archlinux", "build", "builder", "client", "cloud", "computer",
                            "container", "debian", "default", "desktop", "dev", "devbox", "docker",
                            "fedora", "gpu", "home", "host", "kali", "laptop", "linux", "localhost",
                            "localm", "macbook", "machine", "main", "node", "notebook", "office",
                            "prod", "production", "raspberrypi", "redacted", "root", "runner",
                            "server", "staging", "system", "test", "ubuntu", "user", "windows",
                            "work", "worker", "workstation"})


def _own_host_names() -> list:
    """This machine's names (``socket.gethostname()``, its first label, and
    ``COMPUTERNAME``), longest first. Names shorter than three characters and
    those in ``_OWN_NAME_SKIP`` are left out. Empty when none can be read."""
    raw = []
    try:
        raw.append(socket.gethostname())
    except OSError:
        pass
    raw.append(os.environ.get("COMPUTERNAME", ""))
    names = set()
    for name in raw:
        name = (name or "").strip().rstrip(".")
        if name:
            names.add(name)
            names.add(name.split(".")[0])
    return sorted((n for n in names
                   if len(n) >= 3 and n.lower() not in _OWN_NAME_SKIP),
                  key=lambda n: (-len(n), n))


def _scrub_hosts(text: str) -> str:
    """Replace machine names with ``<host>``: a UNC path's machine name
    (``_UNC_HOST_RE``), a host name under a local-network suffix
    (``_scrub_lan_hosts``, suffix kept), and this machine's own names wherever they
    stand as a whole word (case-insensitive). Idempotent."""
    if not text:
        return text
    text = _UNC_HOST_RE.sub(r"\1<host>", text)
    text = _scrub_lan_hosts(text)
    names = _own_host_names()
    if names:
        own = re.compile(r"(?<![A-Za-z0-9_./\\-])(?:"
                         + "|".join(re.escape(n) for n in names)
                         + r")(?![A-Za-z0-9_-]|[./\\][A-Za-z0-9_])",
                         re.IGNORECASE)
        text = own.sub("<host>", text)
    return text


def _scrub_secrets(text: str) -> str:
    """Run every scrubber over untrusted text: home paths (username), URL
    ``user:pass@`` credentials, credential-named query params / header lines,
    bearer / API-key tokens, email addresses other than the maintainer's, and
    machine names (``_scrub_hosts``).
    Used for client-supplied fields and the bundled log tails / activity ring
    a share-intended report carries - each of which is untrusted, free-form
    text that could contain a secret the plain home-scrub alone would leave
    in."""
    if not text:
        return text
    text = _scrub_home(text)
    text = _scrub_url_creds(text)
    text = _scrub_query_and_header_secrets(text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    text = _APIKEY_RE.sub("<redacted>", text)
    text = _scrub_emails(text)
    text = _scrub_hosts(text)
    return text
