# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Central network-access policy for model-initiated requests.

Every network capability that a *model* can trigger (the coder's fetch_url /
web_search tools, the GUI chat's web access) is routed through this module.
Explicit user actions (``localm pull``, typing ``/web`` in chat) are consent
by definition, but still respect ``net_mode = off`` so one switch really does
kill everything.

Config keys (set via ``localm config``, the GUI Settings page, or /v1/config)
------------------------------------------------------------------------------
net_mode           "off" | "ask" | "allow"   (default "ask")
                   off   - every policy-routed request fails fast
                   ask   - allowed, but surfaces that support confirmation
                           ask first (the coder routes network tools through
                           its destructive-tool approval flow)
                   allow - no confirmation
                   The LOCALM_NET_MODE env var overrides the config value.
net_allow          list of domains (or comma-separated string). Empty = any
                   domain. "example.com" matches example.com and *.example.com.
net_deny           same format; matches are always refused (wins over allow).
net_allow_private  False (default) blocks loopback/private/link-local targets
                   (SSRF guard). True restores access to local dev services.
net_search_url     None = the built-in search: DuckDuckGo, then Brave Search
                   when DuckDuckGo fails (no API key). Or the base URL of a
                   SearXNG instance (its JSON API, or its HTML results page
                   when JSON is off); only that instance is then used.

What this module does NOT govern
--------------------------------
Child processes spawned by run_shell (pip, npm, git, …) talk to the network
on their own; the shell-command approval is the gate for those. Model pulls
and online coder providers (OpenAI/Anthropic opt-ins) are explicit user
choices outside this policy.

SSRF guard: the hostname is resolved and validated once, then the socket is
pinned to the validated IPs (see netpin.py), so the connection cannot
re-resolve to a rebound address and an unresolvable host fails closed.
Redirects are re-validated hop by hop.
"""

from __future__ import annotations

import codecs
import html.parser
import ipaddress
import os
import re
import socket
import urllib.parse
from typing import Optional

from localm import __version__
from localm.debuglog import logger

NET_MODES = ("off", "ask", "allow")
NET_MODE_ENV_VAR = "LOCALM_NET_MODE"

_DEFAULT_TIMEOUT = 15
_DEFAULT_TOTAL_TIMEOUT = 45
_DEFAULT_MAX_BYTES = 1_000_000
_MAX_REDIRECTS = 5
_MAX_PINNED_ADDRESSES = 4
_USER_AGENT = (f"Mozilla/5.0 (compatible; localm/{__version__}; "
               "+https://github.com/Matlan1/localm)")


class NetworkPolicyError(Exception):
    """A request was refused by the network policy. The message says why
    and how to change the policy - safe to show to the model, and to a CLI
    user (it names a CLI command).

    *off* is True specifically for the net_mode=off refusal. A caller that
    surfaces this to a browser (a GUI/API route, never the model) has no CLI
    to run and should build its own remedy instead of showing this message's
    CLI-flavored one, the same way DiscoverError.off is used in
    localm/plugins/gui/routes/models/discovery.py's _run_discover."""

    def __init__(self, message: str, *, off: bool = False):
        super().__init__(message)
        self.off = off


# ---------------------------------------------------------------------------
#  Policy resolution
# ---------------------------------------------------------------------------

def network_mode() -> str:
    """Resolve the active mode: LOCALM_NET_MODE env > config > "ask".

    A config-read failure resolves to "off", not "ask", and logs a warning.
    That covers both a load that raises and a config.json (with its .bak) that
    is present but cannot be read, which load_config_checked reports as
    read_ok False. An unset or unrecognised config value resolves to "ask"."""
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    if env in NET_MODES:
        return env
    try:
        from localm.config import load_config_checked
        cfg, read_ok = load_config_checked()
    except Exception as exc:
        logger.warning("netpolicy: could not load config (%s); resolving "
                       "net_mode to 'off' (fail-safe) so an unreadable config "
                       "cannot silently re-enable network access", exc)
        return "off"
    if not read_ok:
        logger.warning("netpolicy: config.json is present but unreadable; "
                       "resolving net_mode to 'off' (fail-safe) so an "
                       "unreadable config cannot silently re-enable network "
                       "access")
        return "off"
    mode = str(cfg.get("net_mode", "ask")).strip().lower()
    return mode if mode in NET_MODES else "ask"


def downloads_allowed_when_off() -> bool:
    """Whether an explicit user download (a model pull, HuggingFace search, or
    a vision-projector/voice/embedding-model fetch) may proceed despite
    network_mode() == "off", via settings_schema.py's
    net_allow_model_downloads (admin_only, default False).

    Never consulted by check_url's model-initiated choke point - that stays
    governed by net_mode alone. Best-effort: an unreadable config resolves to
    blocked, matching network_mode()'s own failure posture."""
    try:
        from localm.config import load_config
        return bool(load_config().get("net_allow_model_downloads", False))
    except Exception:
        return False


def _canonical_host(host: str) -> str:
    """*host* lowercased, without a trailing dot, and with a non-ASCII name
    converted to its IDNA A-label form (UTS #46 mapping, the same encoding
    ``requests`` applies to the host it connects to). An ASCII host is
    returned unchanged apart from case and the trailing dot. Raises
    NetworkPolicyError when a non-ASCII name has no valid A-label form."""
    host = host.lower().rstrip(".")
    if host.isascii():
        return host
    import idna
    try:
        return idna.encode(host, uts46=True).decode("ascii").lower()
    except (idna.IDNAError, UnicodeError) as exc:
        raise NetworkPolicyError(
            f"'{host}' is not a valid internationalised host name; "
            "refusing it.") from exc


def _canonical_entry(entry: str) -> str:
    """A net_deny / net_allow entry in the form ``_canonical_host`` gives a
    URL's host; an entry with no valid A-label form is kept lowercased."""
    try:
        return _canonical_host(entry)
    except NetworkPolicyError:
        return entry.lower()


def _domain_list(value) -> list[str]:
    """Accept a list or a comma-separated string (CLI sets strings). Entries
    are lowercased and a non-ASCII entry is converted to its IDNA A-label
    form, so it matches the host ``check_url_shape`` returns."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [_canonical_entry(str(d).strip().lower().lstrip("*."))
            for d in value if str(d).strip()]


def _host_matches(host: str, pattern: str) -> bool:
    """Suffix match: "example.com" covers example.com and api.example.com."""
    host = host.lower().rstrip(".")
    return host == pattern or host.endswith("." + pattern)


def _config() -> dict:
    """Best-effort config read for callers OTHER than check_url.

    check_url reads the config itself, once, up front, and refuses outright on
    a read failure, so it never reaches this fallback. The remaining callers
    are _resolve_pinned (net_allow_private), _session_for (net_search_url, for
    the search provider's private-address exemption) and the search provider
    selection (net_search_url);
    none reads net_deny/net_allow, so an unreadable config here only means
    those two settings fall back to their defaults (False / unset) for this
    call, which also withdraws the exemption."""
    try:
        from localm.config import load_config
        return load_config()
    except Exception as exc:
        logger.warning("netpolicy: could not load config (%s); using "
                       "defaults for this call", exc)
        return {}


def check_url_shape(url: str) -> str:
    """Validate a URL's SHAPE and return its host in canonical form
    (``_canonical_host``: lowercased, IDNA A-labels for a non-ASCII name).

    The part of the policy that is about the URL STRING rather than about where
    it points: the parser-differential guard, the http/https scheme restriction,
    the host-is-present check and the host's IDNA validity. Raises
    NetworkPolicyError on any of them.

    ``check_url`` calls this, so there is exactly one implementation. A caller
    that wants THESE checks without the destination ones calls it directly: an
    OWNER-CONFIGURED service endpoint (localm's own ``comfy_api_url``, and a
    loopback ``coder_reviewer`` URL or coder model server) points at a local
    server, which the public-address arm of ``check_url`` refuses.

    RUN THIS BEFORE CLASSIFYING A URL AS LOCAL OR REMOTE. Until the
    backslash/control-character check has passed, ``urlparse``'s host can
    disagree with the host the HTTP client actually dials, so any decision made
    from it is a decision about a different destination.
    """
    # Parser-differential SSRF guard. urllib.parse and the HTTP client
    # (requests/urllib3) disagree on backslashes and raw control characters in
    # the authority: 'http://127.0.0.1\\@public/' parses HERE as host 'public'
    # (so it clears every gate below) but requests terminates the userinfo at
    # the backslash and connects to 127.0.0.1. A conformant http(s) URL
    # percent-encodes a backslash or control char, so any raw one is refused
    # rather than the host urllib.parse extracts being trusted. Broader than
    # the authority: a raw '\\' or control char in the path or query is
    # refused too.
    if "\\" in url or any(ord(c) < 0x20 or ord(c) == 0x7F for c in url):
        raise NetworkPolicyError(
            "URL contains a backslash or control character; refusing it "
            "(possible SSRF parser differential).")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NetworkPolicyError(
            f"Only http/https URLs are allowed (got '{parsed.scheme}:'). "
            "Reading local files via file:// is not allowed.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise NetworkPolicyError(f"URL has no host: {url}")
    return _canonical_host(host)


def _origin(url: str) -> Optional[tuple]:
    """``(scheme, host, port)`` of an http(s) *url*, the host in
    :func:`_canonical_host` form (lowercase, no trailing dot, IDNA A-label) and
    the port filled in from the scheme. None for any other URL, one without a
    host, one whose host has no A-label form, or one whose port does not
    parse."""
    try:
        parsed = urllib.parse.urlparse(str(url).strip())
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    try:
        host = _canonical_host(parsed.hostname or "")
    except NetworkPolicyError:
        return None
    if scheme not in ("http", "https") or not host:
        return None
    return scheme, host, port or (443 if scheme == "https" else 80)


_LOCAL_NAME_SUFFIXES = (".local", ".lan", ".internal", ".home.arpa", ".localdomain",
                        ".localhost")


def _is_local_name_or_literal(host: str) -> bool:
    """True when *host* names this machine or the local network by its form:
    an IP literal (any numeric spelling), ``localhost``, a name without a dot,
    or a name under a local-network suffix (``_LOCAL_NAME_SUFFIXES``)."""
    if _literal_ipv4(host) is not None:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    return host == "localhost" or "." not in host or host.endswith(_LOCAL_NAME_SUFFIXES)


def is_search_backend_origin(url: str, cfg: dict) -> bool:
    """True when *url* has the same scheme, host and port as the SearXNG
    instance configured in *cfg*'s ``net_search_url``, and that configured host
    is local by its form (:func:`_is_local_name_or_literal`). False when none
    is configured, and for a public host name."""
    configured = cfg.get("net_search_url") if isinstance(cfg, dict) else None
    if not configured:
        return False
    mine = _origin(configured)
    return (mine is not None and _is_local_name_or_literal(mine[1])
            and _origin(url) == mine)


_SEARCH_BACKEND_REFUSED_V6 = (ipaddress.ip_network("fd00:ec2::/32"),)


def _search_backend_address_ok(ip: ipaddress._BaseAddress) -> bool:
    """Whether the search backend may connect to *ip*: a public address
    (``_is_blocked_ip`` False), an IPv4 loopback or private address that is
    not link-local or unspecified (an IPv4-mapped IPv6 address is judged by its
    IPv4 address), ``::1``, or an IPv6 unique-local address outside the cloud
    metadata range fd00:ec2::/32. Every other address, including link-local in
    NAT64, IPv4-compatible and 6to4 form, is refused."""
    if not _is_blocked_ip(ip):
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.version == 4:
        return bool((ip.is_loopback or ip.is_private)
                    and not ip.is_link_local and not ip.is_unspecified)
    if ip == ipaddress.IPv6Address("::1"):
        return True
    return (ip in ipaddress.ip_network("fc00::/7")
            and not any(ip in net for net in _SEARCH_BACKEND_REFUSED_V6))


def _check_search_backend_address(host: str) -> None:
    """:func:`_check_public_address` for the search backend: *host* (or every
    address it resolves to) must pass :func:`_search_backend_address_ok`.
    Unresolvable hosts pass, as there."""
    literal = _literal_ipv4(host)
    if literal is not None:
        ips = [literal]
    else:
        try:
            ips = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                infos = socket.getaddrinfo(host, None)
            except (socket.gaierror, ValueError):
                return
            ips = []
            for info in infos:
                try:
                    ips.append(ipaddress.ip_address(info[4][0]))
                except ValueError:
                    continue
    for ip in ips:
        if not _search_backend_address_ok(ip):
            raise NetworkPolicyError(
                f"'{host}' is or resolves to {ip}, which the search backend set in "
                "net_search_url cannot use (link-local or cloud-metadata address).")


def check_url(url: str, *, allow_when_off: bool = False,
              search_backend: bool = False) -> None:
    """
    Validate one URL against the policy. Raises NetworkPolicyError with an
    actionable message when refused; returns silently when allowed.

    Checks, in order: mode, malformed-authority, scheme, deny list, allow list,
    resolved-IP class.

    Reads the config exactly ONCE, up front: net_mode, net_allow_model_downloads
    and the net_deny/net_allow lists come from the same snapshot. A load that
    raises, or a config.json (with its .bak) that is present but unreadable
    (read_ok False), has exactly one outcome - refuse - no matter what
    LOCALM_NET_MODE says.

    *allow_when_off*, default False: pass True only from an explicit-download
    caller (a model pull's SSRF/redirect validation, HuggingFace search) so
    net_allow_model_downloads can exempt it from the off floor. Never pass True
    for a model-initiated request (coder fetch_url/web_search, the GUI chat web
    toggle, the jobs webtool) - those stay governed by net_mode alone.

    *search_backend*, default False: pass True only from the SearXNG search
    provider. When *url* is then on the configured ``net_search_url`` origin and
    that host is local by its form (:func:`is_search_backend_origin`), a
    loopback or private address is permitted without ``net_allow_private``
    (:func:`_search_backend_address_ok`); link-local and cloud-metadata
    addresses, in any embedded form, are still refused. Every other URL, and
    every other caller, gets the full address check.
    """
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    unreadable = NetworkPolicyError(
        "Network policy configuration could not be read; refusing this "
        "request as a precaution. Retry once the config is readable.")
    try:
        from localm.config import load_config_checked
        cfg, read_ok = load_config_checked()
    except Exception as exc:
        logger.warning(
            "netpolicy: could not load config (%s); refusing this request "
            "(fail-safe) rather than resolving net_mode and net_deny/"
            "net_allow from different reads", exc)
        raise unreadable from exc
    if not read_ok:
        logger.warning(
            "netpolicy: config.json is present but unreadable; refusing this "
            "request (fail-safe) rather than enforcing net_mode, net_deny and "
            "net_allow from defaults")
        raise unreadable

    mode = env if env in NET_MODES else str(cfg.get("net_mode", "ask")).strip().lower()
    if mode not in NET_MODES:
        mode = "ask"
    if mode == "off" and not (
            allow_when_off and bool(cfg.get("net_allow_model_downloads", False))):
        raise NetworkPolicyError(
            "Network access is disabled (net_mode=off). Enable it with:  "
            "localm config net_mode ask", off=True)

    host = check_url_shape(url)

    deny = _domain_list(cfg.get("net_deny"))
    for pattern in deny:
        if _host_matches(host, pattern):
            raise NetworkPolicyError(
                f"'{host}' is on the deny list (net_deny). ")
    allow = _domain_list(cfg.get("net_allow"))
    if allow and not any(_host_matches(host, p) for p in allow):
        raise NetworkPolicyError(
            f"'{host}' is not on the allow list (net_allow). Add it with:  "
            f"localm config net_allow "
            f"\"{', '.join(allow + [host])}\"")

    if not cfg.get("net_allow_private", False):
        if search_backend and is_search_backend_origin(url, cfg):
            _check_search_backend_address(host)
        else:
            _check_public_address(host)


# Special-use ranges that the stdlib marks is_global=True (so neither
# ``not is_global`` nor any is_* flag catches them) yet are not ordinary public
# hosts. The deprecated 6to4 relay anycast prefix (RFC 7526) sends packets to
# whatever 6to4 relay the local network advertises; the deprecated IPv6
# site-local prefix (RFC 3879) addresses hosts inside the local site.
_EXTRA_BLOCKED_NETS = (
    ipaddress.ip_network("192.88.99.0/24"),   # 6to4 relay anycast (deprecated)
    ipaddress.ip_network("2002::/16"),         # 6to4
    ipaddress.ip_network("fec0::/10"),         # IPv6 site-local (deprecated)
)


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for addresses the SSRF guard refuses: anything that is not a
    globally-routable public address.

    ``not ip.is_global`` is the primary predicate. It rejects loopback, RFC1918
    private, link-local (incl. 169.254.169.254 cloud metadata), the CGNAT
    shared space 100.64.0.0/10 (RFC 6598), benchmarking, documentation and
    other special-use ranges in one shot, and it stays correct as the stdlib
    adds new reserved ranges.

    The explicit special-use flags catch what it does not. The stdlib marks a
    few deprecated IPv6 forms (IPv4-compatible ``::127.0.0.1``, NAT64-embedded
    ``64:ff9b::7f00:1``) is_global=True even though they still route to
    internal IPv4, and is_reserved catches those. _EXTRA_BLOCKED_NETS covers
    the residual special-use ranges (6to4, IPv6 site-local). A genuine public address
    (is_global True with no special-use flag, outside the extra nets) is the
    only thing that passes."""
    return bool(not ip.is_global
                or ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or any(ip.version == net.version and ip in net
                       for net in _EXTRA_BLOCKED_NETS))


def _literal_ipv4(host: str) -> Optional[ipaddress.IPv4Address]:
    """If ``host`` is a numeric / short-form IPv4 literal (dotless decimal
    '2130706433', hex '0x7f000001', octal '0177.0.0.1', short '127.1', or the
    plain dotted form), return its canonical IPv4Address. Otherwise None.

    ipaddress.ip_address() refuses the dotless / hex / octal / short forms and
    socket.getaddrinfo may raise for them, so without this '2130706433'
    (== 127.0.0.1) reaches the policy unclassified and clears the
    public-address check. socket.inet_aton parses the historical IPv4 forms,
    and the canonical address it yields is what gets classified. Normal
    hostnames contain letters, or dots with letters, and make inet_aton raise,
    so they fall through."""
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    try:
        return ipaddress.IPv4Address(packed)
    except ValueError:
        return None


def _check_public_address(host: str) -> None:
    """SSRF guard: refuse hosts that resolve to loopback / private /
    link-local / reserved addresses (cloud metadata, router admin pages,
    the localm API itself…). Unresolvable hosts pass - the fetch will fail
    with a normal DNS error anyway.

    Numeric / short-form IPv4 literals are normalized and classified directly
    (see _literal_ipv4) so they cannot evade the check by being unresolvable or
    unparseable by the ipaddress module."""
    literal = _literal_ipv4(host)
    if literal is not None:
        if _is_blocked_ip(literal):
            raise NetworkPolicyError(
                f"'{host}' is the non-public address {literal}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, ValueError):
        return
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise NetworkPolicyError(
                f"'{host}' resolves to the non-public address {ip}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")


def is_link_local_host(host: str) -> bool:
    """True if *host* is (or resolves to) a link-local address
    (169.254.0.0/16, fe80::/10) - including every numeric/short-form IPv4
    spelling of one (see _literal_ipv4), which ipaddress.ip_address() and
    socket.getaddrinfo alike can fail to classify directly. Unresolvable or
    unparseable hosts return False, the same contract _check_public_address
    uses for a host it cannot resolve.

    This is narrower than _check_public_address: it flags ONLY link-local,
    never loopback/private/public, for a caller (an owner-configured local
    service endpoint) that must allow loopback and LAN but still refuse
    cloud-metadata addresses."""
    lit = _literal_ipv4(host)
    if lit is not None:
        return lit.is_link_local
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:
        pass
    try:
        for info in socket.getaddrinfo(host, None):
            try:
                if ipaddress.ip_address(info[4][0]).is_link_local:
                    return True
            except ValueError:
                continue
    except (socket.gaierror, OSError):
        return False
    return False


def _interleave_families(ips: list) -> list:
    """*ips* (ipaddress objects, resolver order) reordered to alternate address
    families, starting with the family of the first one; order within a family
    is kept."""
    if not ips:
        return []
    first = [ip for ip in ips if ip.version == ips[0].version]
    other = [ip for ip in ips if ip.version != ips[0].version]
    out = []
    for i in range(max(len(first), len(other))):
        if i < len(first):
            out.append(first[i])
        if i < len(other):
            out.append(other[i])
    return out


def _resolve_pinned_all(host: str, *, search_backend: bool = False) -> list[str]:
    """Resolve *host* ONCE to the addresses a connection may be pinned to, in
    dial order, closing the check-and-connect DNS-rebinding TOCTOU.
    ``check_url`` resolves and validates the host, but ``requests``
    re-resolves at connect time, so a TTL-0 attacker can answer 'public' for
    the check and 'internal' for the connect. Every address returned here is
    one the socket may dial and none is looked up again.

    Returns at most ``_MAX_PINNED_ADDRESSES`` canonical IP strings,
    de-duplicated, address families interleaved (first family first), or an
    empty list when the host is unresolvable (the caller then fails closed).
    Numeric/short-form and IPv6 literals are returned directly, already
    validated by check_url. When net_allow_private is False, an address that
    fails the SSRF class check is never returned, and a host whose every
    address fails it raises ``NetworkPolicyError``, on the exact IPs to be
    dialled, which catches a rebind that slipped past check_url's separate
    lookup.

    *search_backend* True (the caller has established that the request is the
    search provider's, to the configured ``net_search_url`` origin) returns only
    addresses :func:`_search_backend_address_ok` accepts, without
    net_allow_private."""
    allow_private = bool(_config().get("net_allow_private", False))

    def _refused(ip_obj) -> bool:
        if allow_private:
            return False
        if search_backend:
            return not _search_backend_address_ok(ip_obj)
        return _is_blocked_ip(ip_obj)

    def _guard(ip_obj) -> None:
        if _refused(ip_obj):
            raise NetworkPolicyError(
                f"'{host}' resolves to the non-public address {ip_obj}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")

    literal = _literal_ipv4(host)
    if literal is not None:
        _guard(literal)
        return [str(literal)]
    try:
        ip_obj = ipaddress.ip_address(host)
        _guard(ip_obj)
        return [str(ip_obj)]
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, ValueError, OSError):
        return []
    blocked = None
    usable: list = []
    for info in infos:
        try:
            ip_obj = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _refused(ip_obj):
            blocked = ip_obj
            continue
        if ip_obj not in usable:
            usable.append(ip_obj)
    if not usable:
        if blocked is not None:
            raise NetworkPolicyError(
                f"'{host}' resolves to the non-public address {blocked}. "
                "Requests to local/private networks are blocked "
                "(set net_allow_private true to permit them).")
        return []
    ordered = _interleave_families(usable)[:_MAX_PINNED_ADDRESSES]
    return [str(ip) for ip in ordered]


def _resolve_pinned(host: str) -> Optional[str]:
    """The first address ``_resolve_pinned_all`` returns for *host* (the first
    usable one in resolver order), or None when the host is unresolvable.
    Raises ``NetworkPolicyError`` exactly when ``_resolve_pinned_all`` does."""
    ips = _resolve_pinned_all(host)
    return ips[0] if ips else None


def _host_header(parsed) -> str:
    """The Host header value for a pinned request: the original hostname in
    ``_canonical_host`` form (so virtual-host routing survives the IP pin),
    with the port only when it is non-default for the scheme."""
    host = _canonical_host(parsed.hostname or "")
    port = parsed.port
    if port and not ((parsed.scheme == "http" and port == 80)
                     or (parsed.scheme == "https" and port == 443)):
        return f"{host}:{port}"
    return host


class UnresolvableHost(NetworkPolicyError):
    """The host could not be resolved to an address the policy validated, so
    no connection was opened."""


def _session_for(url: str, *, search_backend: bool = False):
    """A ``requests.Session`` whose socket is pinned to *url*'s pre-validated
    IPs (``_resolve_pinned_all``: the next one is dialled only when the TCP
    connect to the previous one fails). ``check_url`` MUST already have
    passed on *url*. This is the single
    network-transport seam: production pins here, tests double it here.
    The caller sends ``_host_header(url)`` as the Host header and closes the
    session (use it as a context manager).

    Fails CLOSED when the host cannot be resolved to a validated address: there
    is no fallback to a re-resolving session. A host that is NXDOMAIN at
    validation time (check_url lets unresolvable hosts through) but flips to a
    private A record at connect time (TTL-0 DNS rebinding) would otherwise
    reach an internal service unvalidated.

    *search_backend*: see :func:`check_url`. It permits a private address only
    when *url* is on the configured ``net_search_url`` origin."""
    from localm import netpin
    parsed = urllib.parse.urlparse(url)
    exempt = search_backend and is_search_backend_origin(url, _config())
    host = _canonical_host(parsed.hostname or "")
    ips = _resolve_pinned_all(host, search_backend=True) if exempt else _resolve_pinned_all(host)
    if not ips:
        raise UnresolvableHost(
            f"Could not resolve '{parsed.hostname}' to an address; refusing the "
            "request rather than connecting through an unvalidated re-resolution.")
    return netpin.pinned_session(ips)


class ResponseTooLarge(Exception):
    """A response body was longer than the caller's byte cap."""


def pinned_request(method: str, url: str, *, max_bytes: Optional[int] = None,
                   search_backend: bool = False, **kwargs):
    """A single policy-pinned HTTP request for callers that manage their own
    response (streamed downloads, HEAD probes) instead of going through
    safe_fetch_bytes. ``check_url`` MUST already have passed on *url*.

    The socket is pinned to the pre-validated IP and the original hostname is sent
    as the ``Host`` header. The pinned session is attached to the returned response
    (``resp._localm_pin_session``) so a streamed body stays usable until the caller
    is done with the response; it is released when the response is GC'd, matching
    how requests' own streamed responses are managed. Raises NetworkPolicyError
    when the host cannot be resolved to a validated address (fail-closed).

    With *max_bytes*, the body is read here, one network read at a time and
    decoded at most *max_bytes* + 1 bytes per decompression step, before the
    response is returned: ``.content``, ``.text`` and ``.json()`` then read the
    loaded body, and the connection and session are closed. The request
    advertises only ``PER_READ_ACCEPT_ENCODING``. Raises ``ResponseTooLarge``
    when the decoded body is longer than *max_bytes*.

    *search_backend* is passed to :func:`_session_for` (see :func:`check_url`)."""
    session = (_session_for(url, search_backend=True) if search_backend
               else _session_for(url))
    headers = {**(kwargs.pop("headers", None) or {}),
               "Host": _host_header(urllib.parse.urlparse(url))}
    if max_bytes is not None:
        return _capped_request(session, method, url, headers, max_bytes, kwargs)
    resp = session.request(method, url, headers=headers, **kwargs)
    resp._localm_pin_session = session   # tie the session lifetime to the response
    return resp


def _capped_request(session, method: str, url: str, headers: dict,
                    max_bytes: int, kwargs: dict):
    """``pinned_request``'s *max_bytes* path: send on *session*, read and
    decode the body one network read at a time, store it as the response's
    content, close the response and *session*. Raises ``ResponseTooLarge``
    when the decoded body is longer than *max_bytes*."""
    headers = {**headers, "Accept-Encoding": PER_READ_ACCEPT_ENCODING}
    try:
        resp = session.request(method, url, headers=headers,
                               **{**kwargs, "stream": True})
        try:
            chunks, size = [], 0
            for chunk in _body_chunks(resp, True, max_bytes + 1):
                size += len(chunk)
                if size > max_bytes:
                    raise ResponseTooLarge(
                        f"The response from {urllib.parse.urlparse(url).hostname} "
                        f"is larger than {max_bytes} bytes; refusing to read it.")
                chunks.append(chunk)
            resp._content = b"".join(chunks)
            resp._content_consumed = True
        finally:
            resp.close()
    finally:
        session.close()
    return resp


# ---------------------------------------------------------------------------
#  Fetching
# ---------------------------------------------------------------------------

_BODY_CHUNK = 65536


PER_READ_ACCEPT_ENCODING = "gzip, deflate"


class _BodyDecoder:
    """Content-Encoding decoder for identity, ``gzip`` / ``x-gzip`` (several
    members included) and ``deflate`` (zlib-wrapped or raw). Raises
    ``requests.exceptions.ContentDecodingError`` for any other encoding and
    for corrupt data."""

    def __init__(self, encoding: str):
        import zlib
        self._zlib = zlib
        parts = [p.strip().lower() for p in (encoding or "").split(",")
                 if p.strip()]
        name = ",".join(parts)
        self._first = True
        if name in ("", "identity"):
            self._mode = None
        elif name in ("gzip", "x-gzip"):
            self._mode = "gzip"
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif name == "deflate":
            self._mode = "deflate"
            self._obj = zlib.decompressobj()
        else:
            self._fail(f"unsupported content encoding {encoding!r}")

    @staticmethod
    def _fail(message: str):
        import requests
        raise requests.exceptions.ContentDecodingError(message)

    def decode(self, data: bytes, limit: int) -> bytes:
        """Decoded bytes for *data*, at most *limit* per decompression step
        (a caller stops once it holds *limit* bytes)."""
        if self._mode is None:
            return data
        zlib = self._zlib
        try:
            if self._mode == "deflate":
                if self._first:
                    self._first = False
                    try:
                        return self._obj.decompress(data, limit)
                    except zlib.error:
                        self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
                return self._obj.decompress(data, limit)
            out = self._obj.decompress(data, limit)
            while self._obj.eof and self._obj.unused_data and len(out) < limit:
                rest = self._obj.unused_data
                self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
                out += self._obj.decompress(rest, limit - len(out))
            return out
        except zlib.error as exc:
            self._fail(f"corrupt {self._mode} body: {exc}")

    def flush(self) -> bytes:
        if self._mode is None:
            return b""
        try:
            return self._obj.flush()
        except self._zlib.error as exc:
            self._fail(f"corrupt {self._mode} body: {exc}")


def _body_chunks(resp, per_read: bool, limit: int):
    """Yield *resp*'s decoded body.

    With *per_read* and a raw response that has ``read1``, every yielded
    chunk (possibly empty) comes from exactly one network read of at most
    ``_BODY_CHUNK`` bytes, decoded here by ``_BodyDecoder``, so a caller can
    stop between any two reads; urllib3 read errors are raised as the
    ``requests`` exceptions ``iter_content`` raises (``ConnectionError`` for
    a read timeout, ``ChunkedEncodingError``, ``SSLError``) and decoding
    errors as ``ContentDecodingError``. The request must have been sent with
    ``Accept-Encoding: PER_READ_ACCEPT_ENCODING``. Otherwise this is
    ``iter_content``."""
    raw = getattr(resp, "raw", None)
    read1 = getattr(raw, "read1", None) if per_read else None
    if read1 is None:
        yield from resp.iter_content(chunk_size=_BODY_CHUNK)
        return
    import requests
    from urllib3.exceptions import ProtocolError, ReadTimeoutError, SSLError
    decoder = _BodyDecoder(resp.headers.get("Content-Encoding", ""))
    while True:
        try:
            data = read1(_BODY_CHUNK, decode_content=False)
        except ReadTimeoutError as exc:
            raise requests.exceptions.ConnectionError(exc) from exc
        except ProtocolError as exc:
            raise requests.exceptions.ChunkedEncodingError(exc) from exc
        except SSLError as exc:
            raise requests.exceptions.SSLError(exc) from exc
        if not data:
            tail = decoder.flush()
            if tail:
                yield tail
            return
        yield decoder.decode(data, limit)


def safe_fetch_bytes(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
    allow_when_off: bool = False,
    extra_headers: Optional[dict] = None,
    total_timeout: Optional[float] = None,
) -> tuple[str, str, bytes]:
    """
    Policy-checked GET returning RAW bytes. Returns (final_url, content_type,
    body_bytes).

    Same protections as safe_fetch - redirects are followed manually so every
    hop is re-validated against the policy (a public page cannot bounce the
    fetch into 127.0.0.1), and the body is capped at max_bytes - but the body is
    NOT decoded, so this is the entry point for binary payloads (images fetched
    for vision input). Text callers go through safe_fetch / fetch_text.

    *allow_when_off*: forwarded to every hop's check_url - see that function's
    docstring for who may pass True.

    *extra_headers*: sent on every hop alongside the fixed User-Agent/Host
    pair below, which always win on a key collision - a caller-supplied dict
    can add a header (e.g. an optional Authorization bearer token) but can
    never override the pinned Host or User-Agent. Stripped entirely on any
    hop whose scheme, host or port differs from the ORIGINAL request's
    (:func:`_origin`), so a redirect can never carry a caller's credential to
    a different host, to another port, or from https to plain http.

    *timeout* bounds each connect attempt and each wait for data; a host
    with several addresses may use up to ``netpin._CONNECT_BUDGET_FACTOR``
    times it to connect. *total_timeout*, when set, bounds the whole call
    across hops: each hop's connect gets at most the time left divided by
    that factor and each wait for data at most the time left; the body is
    read and decoded one network read at a time (``_body_chunks``) and a
    body still arriving when the time runs out raises
    ``netpin.ReadBudgetExceeded`` (a ``requests.exceptions.Timeout``), so
    the call overruns *total_timeout* by at most one wait for data. Such
    a request advertises only gzip and deflate and raises
    ``ContentDecodingError`` for any other Content-Encoding.

    Raises NetworkPolicyError (policy refusal) or requests exceptions.
    """
    import time

    from localm.netpin import _CONNECT_BUDGET_FACTOR, ReadBudgetExceeded

    deadline = (time.monotonic() + total_timeout
                if total_timeout is not None else None)
    current = url
    original_origin = _origin(url)
    for _ in range(_MAX_REDIRECTS + 1):
        check_url(current, allow_when_off=allow_when_off)
        parsed = urllib.parse.urlparse(current)
        hop_headers = (extra_headers if original_origin is not None
                       and _origin(current) == original_origin else None)
        hop_timeout = timeout
        fixed_headers = {"User-Agent": _USER_AGENT, "Host": _host_header(parsed)}
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ReadBudgetExceeded(total_timeout, url)
            hop_timeout = (min(timeout, left / _CONNECT_BUDGET_FACTOR),
                           min(timeout, left))
            fixed_headers["Accept-Encoding"] = PER_READ_ACCEPT_ENCODING
        # Pin the socket to the just-validated IP for this hop; each redirect
        # target is independently re-checked and re-pinned.
        with _session_for(current) as session:
            resp = session.get(
                current,
                timeout=hop_timeout,
                stream=True,
                allow_redirects=False,
                headers={**(hop_headers or {}), **fixed_headers},
            )
            try:
                if resp.is_redirect or resp.is_permanent_redirect:
                    location = resp.headers.get("Location", "")
                    if not location:
                        raise NetworkPolicyError(
                            f"Redirect from {current} without a Location header.")
                    current = urllib.parse.urljoin(current, location)
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                chunks, size = [], 0
                for chunk in _body_chunks(resp, deadline is not None,
                                          max_bytes):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= max_bytes:
                        break
                    if deadline is not None and time.monotonic() > deadline:
                        raise ReadBudgetExceeded(total_timeout, url)
                return current, content_type, b"".join(chunks)[:max_bytes]
            finally:
                resp.close()
    raise NetworkPolicyError(
        f"Too many redirects (>{_MAX_REDIRECTS}) fetching {url}")


_CHARSET_RE_HEADER = re.compile(r'charset\s*=\s*["\']?([\w.:-]+)', re.IGNORECASE)
_CHARSET_RE_META = re.compile(
    rb'<meta[^>]{0,512}?charset\s*=\s*["\']?([\w.:-]+)', re.IGNORECASE)


def _declared_charset(content_type: str, body: bytes) -> str | None:
    """The charset named by the Content-Type header, else by an HTML
    <meta charset> or <meta http-equiv=Content-Type content=...charset=...>
    tag in the first 2048 bytes. None when neither declares one, or the
    declared name is not a codec Python has.
    """
    m = _CHARSET_RE_HEADER.search(content_type or "")
    name = m.group(1) if m else None
    if not name:
        m = _CHARSET_RE_META.search(body[:2048])
        if m:
            name = m.group(1).decode("ascii", errors="ignore")
    if not name:
        return None
    try:
        codecs.lookup(name)
    except LookupError:
        return None
    return name


def safe_fetch(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
    total_timeout: Optional[float] = None,
) -> tuple[str, str, str]:
    """
    Policy-checked GET. Returns (final_url, content_type, body_text).

    Thin text wrapper over safe_fetch_bytes (which does the policy check,
    per-hop redirect re-validation, size cap and the optional
    *total_timeout*). The body is decoded using
    the charset the response declares - the Content-Type header, then an
    HTML <meta charset> tag - falling back to UTF-8 when neither declares
    one or the declared one fails to decode.

    Raises NetworkPolicyError (policy refusal) or requests exceptions.
    """
    final_url, content_type, body = safe_fetch_bytes(
        url, max_bytes=max_bytes, timeout=timeout, total_timeout=total_timeout)
    charset = _declared_charset(content_type, body)
    if charset:
        try:
            return final_url, content_type, body.decode(charset, errors="replace")
        except Exception:
            pass
    return final_url, content_type, body.decode("utf-8", errors="replace")


class _HTMLStripper(html.parser.HTMLParser):
    """Extract readable text from an HTML document."""

    _SKIP = {"script", "style", "head", "meta", "link", "noscript", "svg",
             "template"}
    # Void elements have no end tag and must NOT move the skip counter: a
    # <meta>/<link> in <head> would increment it with no matching decrement,
    # leaving _skip > 0 forever so the whole <body> is dropped.
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self._VOID:
            if t in self._BLOCK:          # <br> -> line break
                self._buf.append("\n")
            return                        # never touch the skip counter
        if t in self._SKIP:
            self._skip += 1
        elif t in self._BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self._VOID:
            return
        if t in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        raw = re.sub(r"[ \t]+", " ", raw)
        return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def html_to_text(markup: str) -> str:
    """Best-effort plain text from HTML."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(markup)
    except Exception:
        # HTMLParser can choke on malformed markup; return whatever was parsed
        # before the error rather than failing.
        pass
    return stripper.get_text()


def fetch_text(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
    total_timeout: Optional[float] = _DEFAULT_TOTAL_TIMEOUT,
    on_endpoint=None,
) -> tuple[str, str]:
    """safe_fetch + HTML stripping. Returns (final_url, plain_text).

    A GitHub repository or file URL, or a Stack Exchange question URL, is
    read from that site's content endpoint when it yields text, with the
    page itself as the fallback (``localm.web_retrieval.sites.read_url``).
    *total_timeout* bounds each underlying fetch (see safe_fetch_bytes).
    *on_endpoint* is called with each content-endpoint URL before it is
    requested (see ``read_url``)."""
    from localm.web_retrieval.sites import read_url

    def _fetch(target: str, *, timeout: float) -> tuple[str, str, str]:
        return safe_fetch(target, max_bytes=max_bytes, timeout=timeout,
                          total_timeout=total_timeout)

    final_url, content_type, body = read_url(url, _fetch, timeout=timeout,
                                             on_endpoint=on_endpoint)
    if "html" in content_type.lower():
        return final_url, html_to_text(body)
    return final_url, body.strip()


# ---------------------------------------------------------------------------
#  Web search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web. Returns [{"title", "url", "snippet"}, ...].

    Backend: the SearXNG instance set in net_search_url (its JSON API, or
    its HTML results page when the instance refuses JSON), otherwise the
    built-in chain of DuckDuckGo's HTML page, DuckDuckGo's lite page and
    Brave Search. The providers live in ``localm.web_retrieval.providers``
    and send every request through this module's policy check and pinned
    transport.
    Raises NetworkPolicyError when the policy refuses; SearchProviderError
    (a RuntimeError) when the search returns no results, every built-in
    service failed, or a SearXNG instance's answer is unusable; and a
    configured SearXNG instance's transport failure as the ``requests``
    exception.
    """
    from localm.web_retrieval.providers import search
    return [r.to_legacy() for r in search(query, max_results)]


def format_results(results: list[dict]) -> str:
    """Stable plain-text rendering shared by the coder tool and the chat."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)
