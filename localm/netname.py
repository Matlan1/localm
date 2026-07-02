# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reach localm by NAME, not just by IP.

On a network bind localm already prints reachable IP addresses and serves HTTPS.
This module adds a stable, app-chosen NAME so a phone or laptop opens
``https://localm.local:PORT`` instead of typing an IP that changes with DHCP:

  * **LAN (mDNS / Bonjour):** advertise ``<mdns_name>.local`` (default
    ``localm.local``) over multicast DNS. Works with no configuration on
    Windows 11, macOS, iOS, Android, and Linux with Avahi. This is the portable
    cross-platform name; a bare ``localm`` with no ``.local`` suffix is not
    reliably resolvable across machines, so ``.local`` is what we guarantee.
  * **Tailscale (MagicDNS):** Tailscale gives each node a name like
    ``<node>.<tailnet>.ts.net`` (and a bare ``<node>`` on tailnet peers). mDNS
    does NOT cross Tailscale, so the name there is the node's MagicDNS name, which
    we detect via the ``tailscale`` CLI and add to the certificate + printed URLs.
    Reaching it as literally ``localm`` needs the node named ``localm`` (a user
    action: ``tailscale up --hostname=localm``) - we print the hint, never run it.

Everything here is best-effort and never raises into the serving path: if
zeroconf is missing, the LAN has no usable address, the name is already taken, or
Tailscale is not installed, we log a note and fall back to IP-based access. The
name is also folded into the TLS certificate SANs (see ``cert_hostnames``) so
name-based HTTPS has no certificate-name-mismatch warning after the one-time CA
trust.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
from typing import Optional

from localm.debuglog import logger

_DEFAULT_NAME = "localm"
# A single DNS label: ASCII letters, digits, hyphen; <=63 chars; no leading or
# trailing hyphen. Anything else in a configured name is folded to a hyphen.
_LABEL_STRIP_RE = re.compile(r"[^a-z0-9-]")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# ``tailscale status --json`` can hang if the daemon is wedged; bound it so a
# broken Tailscale never stalls server startup.
_TS_TIMEOUT = 4


# ------------------------------------------------------------------ #
#  Config                                                            #
# ------------------------------------------------------------------ #

def _config() -> dict:
    try:
        from localm.config import load_config
        return load_config()
    except Exception:
        # Best-effort: if config cannot be read we fall back to the built-in
        # defaults (mDNS on, name "localm"). A note beats crashing the bind.
        logger.debug("netname: config unreadable; using defaults", exc_info=True)
        return {}


def mdns_enabled() -> bool:
    """Whether to advertise the ``.local`` name over mDNS on network binds."""
    return bool(_config().get("mdns_enabled", True))


def normalize_label(raw: str) -> str:
    """Reduce *raw* to a valid single DNS label: lowercased, only ``[a-z0-9-]``,
    collapsed hyphens, no leading/trailing hyphen, capped at 63 chars. Returns ""
    when nothing usable remains (e.g. all-punctuation or non-ASCII input) - there
    is NO default fallback here, so a caller that validates user input can tell an
    unusable name apart from a real one. Use ``mdns_label`` for the fallback."""
    s = _LABEL_STRIP_RE.sub("-", str(raw).strip().lower())
    return re.sub(r"-{2,}", "-", s).strip("-")[:63].strip("-")


def mdns_label(name: Optional[str] = None) -> str:
    """The advertised DNS label from *name* (or the configured ``mdns_name``),
    sanitized. Falls back to ``localm`` when nothing usable remains, so a
    hand-edited config can never produce an invalid mDNS name."""
    raw = name if name is not None else (_config().get("mdns_name") or _DEFAULT_NAME)
    return normalize_label(raw) or _DEFAULT_NAME


def mdns_fqdn(name: Optional[str] = None) -> str:
    """The advertised LAN name, e.g. ``localm.local``."""
    return f"{mdns_label(name)}.local"


def _hostname_label() -> str:
    """This machine's short hostname (first label), lowercased, or ""."""
    try:
        h = socket.gethostname()
    except OSError:
        return ""
    return (h or "").split(".")[0].strip().lower()


def hostname_fqdn() -> str:
    """This machine's own ``<hostname>.local`` (the OS advertises it via mDNS
    independently), or "" when the hostname is undetermined. Included in the cert
    so the machine's real name also works over HTTPS without a warning."""
    h = _hostname_label()
    return f"{h}.local" if h else ""


# ------------------------------------------------------------------ #
#  Tailscale (MagicDNS)                                              #
# ------------------------------------------------------------------ #

_STATUS_UNSET = object()
_status_memo: object = _STATUS_UNSET


def _tailscale_cli() -> Optional[str]:
    """Path to the ``tailscale`` CLI if it is on PATH, else None. We only trust
    PATH (never a hardcoded install location), so a machine where Tailscale is not
    installed simply skips Tailscale integration."""
    return shutil.which("tailscale") or shutil.which("tailscale.exe")


def _run_tailscale_status() -> Optional[dict]:
    """Parsed ``tailscale status --json``, or None when Tailscale is absent, not
    running, or unreadable. Best-effort; never raises."""
    cli = _tailscale_cli()
    if not cli:
        return None
    try:
        proc = subprocess.run(
            [cli, "status", "--json"],
            capture_output=True, text=True, timeout=_TS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("netname: tailscale status failed: %s", e)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except (ValueError, TypeError):
        logger.debug("netname: tailscale status returned non-JSON")
        return None
    return data if isinstance(data, dict) else None


def tailscale_status(*, refresh: bool = False) -> Optional[dict]:
    """Memoized ``tailscale status`` for this process (server startup is a point
    in time; several callers read it). Pass ``refresh=True`` to re-run."""
    global _status_memo
    if refresh or _status_memo is _STATUS_UNSET:
        _status_memo = _run_tailscale_status()
    return _status_memo  # type: ignore[return-value]


def _reset_status_cache() -> None:
    """Test hook: forget the memoized Tailscale status."""
    global _status_memo
    _status_memo = _STATUS_UNSET


def _tailscale_magicdns(status: Optional[dict]) -> bool:
    """True when the tailnet has MagicDNS enabled, so the node's DNS name actually
    resolves for peers. Falls back to the presence of a MagicDNS suffix."""
    if not status:
        return False
    ct = status.get("CurrentTailnet") or {}
    if isinstance(ct, dict) and "MagicDNSEnabled" in ct:
        return bool(ct.get("MagicDNSEnabled"))
    return bool(status.get("MagicDNSSuffix"))


def _tailscale_running(status: Optional[dict]) -> bool:
    return bool(status) and status.get("BackendState") == "Running"


def tailscale_node_label(status: Optional[dict] = None) -> str:
    """The tailnet node's short name (first label of its MagicDNS name), lowercased,
    or "" when Tailscale is not up. E.g. ``mybox`` for ``mybox.tailnet.ts.net``."""
    status = status if status is not None else tailscale_status()
    self_ = (status or {}).get("Self") or {}
    dns = str(self_.get("DNSName") or "").strip().rstrip(".").lower()
    if dns:
        return dns.split(".")[0]
    # Fall back to the reported hostname if DNSName is absent.
    return str(self_.get("HostName") or "").strip().split(".")[0].lower()


def tailscale_names(status: Optional[dict] = None) -> list[str]:
    """DNS names for THIS node on the tailnet, resolvable for peers when MagicDNS
    is on: the full MagicDNS name (``mybox.tailnet.ts.net``) and its short label
    (``mybox``). Empty when Tailscale is not installed or not running. The names
    are returned even if MagicDNS is momentarily off, because they belong in the
    certificate regardless; callers that need resolvability check
    ``tailscale_name_reachable``. Best-effort; never raises."""
    status = status if status is not None else tailscale_status()
    if not _tailscale_running(status):
        return []
    self_ = (status or {}).get("Self") or {}
    dns = str(self_.get("DNSName") or "").strip().rstrip(".").lower()
    if not dns:
        return []
    out = [dns]
    short = dns.split(".")[0]
    if short and short != dns:
        out.append(short)
    return out


def tailscale_name_reachable(status: Optional[dict] = None) -> bool:
    """True when this node's MagicDNS name will actually resolve for tailnet peers
    (Tailscale running AND MagicDNS enabled)."""
    status = status if status is not None else tailscale_status()
    return _tailscale_running(status) and _tailscale_magicdns(status)


def tailscale_rename_hint() -> Optional[str]:
    """A one-line hint on how to make this node reachable as ``<mdns_name>`` on the
    tailnet, or None when Tailscale is down or already named that. Purely
    informational - we never rename the node (that is the user's choice)."""
    status = tailscale_status()
    if not _tailscale_running(status):
        return None
    node = tailscale_node_label(status)
    want = mdns_label()
    if not node or node == want:
        return None
    return (f"to reach it as '{want}' on Tailscale, rename this node: "
            f"tailscale up --hostname={want}")


# ------------------------------------------------------------------ #
#  Certificate SANs + reachable URLs                                 #
# ------------------------------------------------------------------ #

def cert_hostnames() -> list[str]:
    """Extra DNS names the server certificate should cover so NAME-based HTTPS has
    no certificate-name-mismatch warning: the mDNS name (``localm.local``), this
    machine's ``<hostname>.local``, and the Tailscale MagicDNS name(s). De-duped,
    lowercased, order preserved. Best-effort - a failure to detect one just
    certifies fewer names."""
    names: list[str] = []
    if mdns_enabled():
        names.append(mdns_fqdn())
    h = hostname_fqdn()
    if h:
        names.append(h)
    names.extend(tailscale_names())
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        n = (n or "").strip().lower()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def network_targets(*, mdns_name: Optional[str] = None) -> list[tuple[str, str]]:
    """Ordered ``(label, host)`` pairs a client can use to reach this server on a
    network bind - names before IPs (friendlier), IPs kept as fallbacks. Excludes
    loopback (callers print that themselves). Best-effort.

    *mdns_name* is the ``<name>.local`` that the LIVE mDNS advertiser is publishing:
    pass it ONLY when ``start_advertiser`` actually succeeded, else leave it None so
    we never recommend a name this host does not own (a loopback/--isolated bind, a
    name-conflict, or a machine with no advertisable address). The Tailscale name is
    gated separately on its own reachability (MagicDNS), since we do not advertise
    that - the tailnet resolves it."""
    from localm import tls
    addrs = tls.companion_addresses()
    status = tailscale_status()
    out: list[tuple[str, str]] = []

    if mdns_name:
        out.append(("on this network", mdns_name))
    if addrs.get("lan"):
        out.append(("on this network (IP)", addrs["lan"]))

    ts_names = tailscale_names(status)
    if ts_names and tailscale_name_reachable(status):
        out.append(("via Tailscale", ts_names[0]))
    if addrs.get("tailscale"):
        out.append(("via Tailscale (IP)", addrs["tailscale"]))
    return out


def ca_trust_host(mdns_name: Optional[str] = None) -> str:
    """The best host to show in the "trust the certificate" hint / CA-download URL
    on a network bind: the live mDNS name when we are advertising it (*mdns_name*),
    else the primary LAN IP, else "". Purely for display. Pass *mdns_name* only
    when advertising actually succeeded."""
    if mdns_name:
        return mdns_name
    from localm import tls
    return tls.companion_addresses().get("lan") or ""


# ------------------------------------------------------------------ #
#  mDNS advertiser                                                   #
# ------------------------------------------------------------------ #

class _Advertiser:
    """Handle for a live mDNS advertisement; call ``close()`` to withdraw it."""

    def __init__(self, zc, infos: list) -> None:
        self._zc = zc
        self._infos = infos

    def close(self) -> None:
        for info in self._infos:
            try:
                self._zc.unregister_service(info)
            except Exception:
                # Best-effort withdrawal on shutdown; the responder is closed next
                # regardless, and the record ages out of peer caches on its TTL.
                pass
        try:
            self._zc.close()
        except Exception:
            pass


def start_advertiser(port: int, *, tls: bool,
                     addresses: Optional[list[str]] = None) -> Optional["_Advertiser"]:
    """Advertise this server over mDNS so it is reachable as ``<mdns_name>.local``
    on the LAN, and return a handle whose ``close()`` withdraws it.

    Registers an A-record hostname (``localm.local`` -> the LAN address) plus an
    ``_https._tcp`` / ``_http._tcp`` service so generic Bonjour/Avahi browsers
    discover it too. Returns None (and logs a note) when mDNS is disabled in
    config, zeroconf is unavailable, no LAN address can be determined, or the name
    is already taken on this network. NEVER raises - name-based access is a
    convenience layered on the always-working IP-based access.

    The caller is responsible for only invoking this on a network bind that is not
    ``--isolated``; this function additionally honours the ``mdns_enabled`` config.
    """
    if not mdns_enabled():
        return None
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except Exception as e:
        logger.debug("netname: zeroconf unavailable, skipping mDNS: %s", e)
        return None
    try:
        from zeroconf import NonUniqueNameException
    except Exception:
        NonUniqueNameException = Exception  # type: ignore[assignment,misc]

    from localm import tls as _tls
    ips = [ip for ip in (addresses or []) if ip]
    if not ips:
        prim = _tls._primary_lan_ip()
        if prim:
            ips.append(prim)
    packed = []
    for ip in ips:
        try:
            packed.append(socket.inet_aton(ip))
        except OSError:
            continue
    if not packed:
        logger.warning("netname: no LAN IPv4 to advertise; name-based access "
                       "unavailable this run (IP access still works)")
        return None

    label = mdns_label()
    server = f"{label}.local."
    svc_type = "_https._tcp.local." if tls else "_http._tcp.local."
    info = ServiceInfo(
        svc_type,
        f"{label}.{svc_type}",
        addresses=packed,
        port=port,
        properties={"path": "/", "tls": "true" if tls else "false"},
        server=server,
    )
    try:
        zc = Zeroconf()
    except Exception as e:
        logger.warning("netname: could not start mDNS responder: %s", e)
        return None
    try:
        zc.register_service(info)
    except NonUniqueNameException:
        logger.warning(
            "netname: %s is already claimed on this network - not advertising "
            "(a name-based URL could reach a different host). Use a different "
            "'mdns_name', or reach this server by IP.", server)
        try:
            zc.close()
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning("netname: mDNS advertise failed: %s", e)
        try:
            zc.close()
        except Exception:
            pass
        return None

    logger.debug("netname: advertising %s -> %s:%d over mDNS", server, ips, port)
    return _Advertiser(zc, [info])
