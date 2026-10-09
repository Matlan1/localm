# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redaction applied to the free-form text a report carries: home paths and user
names, URL credentials, credential-named query parameters, header lines and JSON
keys, bearer tokens and API keys.
"""

from __future__ import annotations

import re

from localm import pathscrub

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


def _scrub_secrets(text: str) -> str:
    """Run every scrubber over untrusted text: home paths (username), URL
    ``user:pass@`` credentials, credential-named query params / header lines,
    and bearer / API-key tokens. Used for client-supplied fields and the
    bundled log tails / activity ring a share-intended report carries - each
    of which is untrusted, free-form text that could contain a secret the
    plain home-scrub alone would leave in."""
    if not text:
        return text
    text = _scrub_home(text)
    text = _scrub_url_creds(text)
    text = _scrub_query_and_header_secrets(text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    text = _APIKEY_RE.sub("<redacted>", text)
    return text
