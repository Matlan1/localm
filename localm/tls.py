# SPDX-License-Identifier: AGPL-3.0-or-later
"""Built-in TLS for network binds (NET-1).

localm serves HTTPS itself the moment it binds past loopback, so the API key and
all traffic are encrypted out of the box - no reverse proxy, no tailscale, no
manual certificate wrangling. We run a tiny local certificate authority: a
long-lived CA whose certificate the user installs ONCE per device (one tap),
plus a short-lived leaf certificate that the server actually presents.

Installing the CA removes the browser's "not secure" warning and lets the PWA
install. Because the CA is REUSED whenever the leaf is regenerated, a device
trusted once stays trusted even if this machine's LAN IP later changes.

This is the same model as Caddy's ``tls internal`` and mkcert. The CA and leaf
private keys are stored as unencrypted PEM under ``<LOCALM_HOME>/tls/`` (uvicorn
loads them without a passphrase); the directory and key files are locked down
where the OS permits it (POSIX chmod; best-effort and harmless elsewhere).
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

# Validity windows. The CA outlives many leaf regenerations, so trusting it once
# survives an IP change; the leaf stays under Apple's 825-day ceiling (iOS and
# Safari reject longer-lived server certs even from a privately trusted CA).
_CA_DAYS = 3650          # ~10 years
_LEAF_DAYS = 730         # ~2 years (< 825)
# Regenerate a leaf this far ahead of its real expiry so a long-running or
# rarely-restarted instance never ends up serving an expired certificate.
_RENEW_MARGIN_DAYS = 30

# Tailscale hands out addresses in the 100.64.0.0/10 CGNAT range.
_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")

# Adapter-NAME substrings that mark a VPN/tunnel virtual interface rather than
# a real physical LAN NIC. Most VPN clients (WireGuard, OpenVPN, and virtually
# every commercial VPN app) hand their tunnel adapter an ordinary RFC1918
# address - e.g. 10.x.x.x - and install it as the default route, which would
# otherwise be indistinguishable from a genuine LAN address by IP range alone.
# Best-effort name match, not exhaustive; lowercased comparison.
_VPN_ADAPTER_NAME_MARKERS = (
    "tap", "tun", "vpn", "wireguard", "wintun", "ppp", "utun",
    "nordlynx", "openvpn", "globalprotect", "anyconnect", "zscaler",
)


# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

def tls_dir(home: Path) -> Path:
    """Directory holding the CA + leaf material for this install."""
    return Path(home) / "tls"


def ca_cert_path(home: Path) -> Path:
    """The CA certificate users download and trust (``GET /localm-ca.crt``)."""
    return tls_dir(home) / "ca.crt"


def _ca_key_path(home: Path) -> Path:
    return tls_dir(home) / "ca.key"


def _leaf_cert_path(home: Path) -> Path:
    return tls_dir(home) / "server.crt"


def _leaf_key_path(home: Path) -> Path:
    return tls_dir(home) / "server.key"


def _meta_path(home: Path) -> Path:
    return tls_dir(home) / "meta.json"


# ------------------------------------------------------------------ #
#  SAN discovery                                                      #
# ------------------------------------------------------------------ #

def _vpn_adapter_ips() -> set[str]:
    """IPv4 addresses bound to an adapter whose NAME looks VPN/tunnel-like
    (best-effort, via psutil when the ``[monitor]`` extra is installed; empty
    otherwise, same degrade as ``_iface_ips``). A VPN's virtual adapter often
    carries an ordinary RFC1918 address indistinguishable from a real LAN
    address by IP range alone, so callers that need to tell them apart cross-
    reference against the owning adapter's name instead."""
    out: set[str] = set()
    try:
        import psutil
    except Exception:
        return out
    try:
        for name, addrs in psutil.net_if_addrs().items():
            if not any(marker in name.lower() for marker in _VPN_ADAPTER_NAME_MARKERS):
                continue
            for a in addrs:
                if getattr(a, "family", None) == socket.AF_INET:
                    out.add(a.address)
    except Exception:
        # Best-effort, same as _iface_ips: an enumeration failure just means we
        # cannot tell a VPN adapter apart from a real one this call, not a
        # reason to fail whatever the caller was doing.
        pass
    return out


def _primary_lan_ip() -> str:
    """Best-effort primary outbound LAN IPv4, or "" if undetermined. Opens a UDP
    socket toward a TEST-NET address to learn the outbound interface; no packets
    are actually sent.

    A VPN client can legitimately become the default route, in which case this
    trick reports the VPN's virtual tunnel address instead of the machine's
    real LAN address - reachable only through the VPN, not by another device on
    the physical LAN. When the discovered address maps to an adapter whose name
    looks VPN/tunnel-like, it is discarded (treated as undetermined) rather than
    reported as the LAN IP; callers that need a fallback (``companion_addresses``)
    still have other probes to try."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 80))   # TEST-NET-1 (RFC 5737); no traffic
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return ""
    if ip in _vpn_adapter_ips():
        return ""
    return ip


def _host_ips() -> list[str]:
    """All IPv4 addresses bound to this machine's hostname (best-effort). On
    Windows this usually includes the LAN address and any Tailscale address."""
    out: list[str] = []
    try:
        hostname = socket.gethostname()
    except OSError:
        return out
    try:
        _, _, addrs = socket.gethostbyname_ex(hostname)
        out.extend(addrs)
    except OSError:
        # Best-effort SAN enrichment: if reverse/forward resolution fails we just
        # certify fewer addresses. The loopback/explicit SANs added elsewhere
        # still cover the working case; never fail cert minting over this.
        pass
    return out


def _norm_ip(value: str) -> Optional[str]:
    """Canonicalise an IP string (so ``::1`` and its long form compare equal), or
    None when it is not a valid IP address."""
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def san_targets(
    extra_hostnames: Optional[Iterable[str]] = None,
    extra_ips: Optional[Iterable[str]] = None,
) -> tuple[list[str], list[str]]:
    """Return ``(hostnames, ip_strings)`` the server cert should cover: loopback,
    this machine's hostname, its primary LAN IP, any Tailscale IP, plus any
    caller-supplied extras. Best-effort and never raises; results are sorted and
    de-duplicated. IPs are canonicalised so equivalent forms do not duplicate."""
    hostnames: set[str] = {"localhost"}
    ips: set[str] = {"127.0.0.1", "::1"}

    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = ""
    if hostname:
        hostnames.add(hostname)

    for ip in _host_ips():
        norm = _norm_ip(ip)
        if norm:
            ips.add(norm)

    lan = _primary_lan_ip()
    if lan:
        norm = _norm_ip(lan)
        if norm:
            ips.add(norm)

    for h in extra_hostnames or ():
        h = (h or "").strip()
        # A bare IP passed as a hostname (e.g. -H 0.0.0.0 / a specific bind IP)
        # belongs in the IP SAN list, not the DNS one. 0.0.0.0 means "all
        # interfaces" and is never itself a connectable address, so drop it.
        if not h or h == "0.0.0.0":
            continue
        norm = _norm_ip(h)
        if norm:
            ips.add(norm)
        else:
            hostnames.add(h)

    for ip in extra_ips or ():
        norm = _norm_ip(ip)
        if norm:
            ips.add(norm)

    return sorted(hostnames), sorted(ips, key=lambda s: (":" in s, s))


def is_tailscale_ip(value: str) -> bool:
    """True when *value* is a Tailscale (100.64.0.0/10 CGNAT) address."""
    try:
        return ipaddress.ip_address(value) in _TAILSCALE_NET
    except ValueError:
        return False


def _iface_ips() -> list[str]:
    """Every IPv4 bound to a local interface, via psutil when it is installed
    (the ``[monitor]`` extra). Catches a Tailscale address that hostname
    resolution misses - notably on Linux, where ``gethostbyname_ex`` often
    returns only loopback. Empty when psutil is absent; never raises."""
    out: list[str] = []
    try:
        import psutil
    except Exception:
        return out
    try:
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if getattr(a, "family", None) == socket.AF_INET:
                    out.append(a.address)
    except Exception:
        # Best-effort: enumerating interface IPs via psutil is optional. If psutil
        # is absent or errors we return whatever was gathered; the caller (the
        # Companion address card) degrades to fewer addresses, not a failure.
        pass
    return out


def companion_addresses() -> dict:
    """Best-effort ``{"lan": ip, "tailscale": ip}`` for the Companion-app card, so
    a phone is shown a REACHABLE address instead of the meaningless loopback one
    (``127.0.0.1`` on a phone is the phone itself).

    ``lan`` is this machine's primary private (RFC 1918) IPv4 - the interface a
    phone on the same Wi-Fi reaches; ``tailscale`` is a 100.64.0.0/10 (CGNAT)
    address when Tailscale is up, else "". Either may be "" when it cannot be
    determined. A VPN's virtual tunnel adapter is never picked for ``lan`` even
    if it offers an RFC 1918 address, since it is reachable only through the
    VPN. Purely informational - never raises."""
    lan = ""
    tailscale = ""
    primary = _norm_ip(_primary_lan_ip()) or ""
    vpn_ips = _vpn_adapter_ips()
    seen: set[str] = set()
    # Candidate IPv4s this machine owns: the primary outbound interface first
    # (the preferred LAN pick), then hostname + per-interface addresses.
    for ip in (primary, *_host_ips(), *_iface_ips()):
        norm = _norm_ip(ip)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            addr = ipaddress.ip_address(norm)
        except ValueError:
            continue
        if addr.version != 4:
            continue
        # Check Tailscale BEFORE is_private: Python treats 100.64.0.0/10 as
        # private, so this ordering keeps a Tailscale IP out of the LAN slot.
        if is_tailscale_ip(norm):
            if not tailscale:
                tailscale = norm
            continue
        if addr.is_loopback or addr.is_link_local or not addr.is_private:
            continue
        if norm in vpn_ips:
            continue
        # The primary outbound interface always wins the LAN slot; otherwise
        # take the first private address seen.
        if norm == primary:
            lan = norm
        elif not lan:
            lan = norm
    return {"lan": lan, "tailscale": tailscale}


def requests_verify(url: str):
    """The value to pass as ``requests``' ``verify=`` for an in-process call to
    *url*.

    Built-in TLS gives a network bind an HTTPS endpoint whose leaf is signed by
    this install's local CA, so an in-process loopback self-call (the coder
    agent, media-job model reloads, RAG self-embedding) must trust that CA.
    Returns the CA bundle path for a loopback HTTPS URL - or False when the CA
    file is somehow absent, which is safe because the connection never leaves
    127.0.0.1. Returns True (normal public verification) for plain HTTP and for
    any non-loopback HTTPS URL, e.g. the OpenAI / Anthropic APIs - so this never
    weakens verification of a real external endpoint."""
    low = (url or "").lower()
    if not low.startswith("https:"):
        return True
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    if host not in ("127.0.0.1", "::1", "localhost"):
        return True
    from localm.config import home_dir
    ca = ca_cert_path(home_dir())
    return str(ca) if ca.is_file() else False


# ------------------------------------------------------------------ #
#  Filesystem helpers                                                 #
# ------------------------------------------------------------------ #

def _lock_down_dir(path: Path) -> None:
    """Restrict the tls dir to the owner where the OS supports it (best-effort)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:
        # Defense-in-depth, proven low-risk. chmod is a Windows no-op and only
        # raises on POSIX filesystems with no per-file perms (FAT, some network
        # mounts) - exactly the case where NO process could restrict perms anyway.
        # The tls dir lives under the user-owned data home, so its parent perms
        # already scope it; this 0o700 is a belt-and-braces tightening, not the
        # sole protection.
        pass


def _write_private(path: Path, data: bytes) -> None:
    """Write key material with 0600 perms where the OS supports it."""
    path.write_bytes(data)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Private key perms - the most sensitive write here, so the reasoning is
        # explicit. chmod is a Windows no-op and only raises on POSIX filesystems
        # that have no per-file perms at all (FAT, some network mounts). On such a
        # filesystem NO API could restrict this file, and the same limit applied
        # to the enclosing tls dir (already 0o700-attempted by _lock_down_dir) and
        # to the user data home it sits in: the key is exactly as protected as
        # everything else the user stores there. So a failure here cannot single
        # out the key as newly exposed; it is the user's storage choice, not a
        # localm regression. Hence safe to proceed rather than refuse to serve TLS.
        pass


# ------------------------------------------------------------------ #
#  Certificate authority + leaf                                       #
# ------------------------------------------------------------------ #

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_ca(home: Path):
    """Create and persist a fresh local CA (cert + private key). Returns
    ``(ca_cert, ca_key)``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localm local CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "localm"),
    ])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))   # tolerate clock skew
        .not_valid_after(now + timedelta(days=_CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    _lock_down_dir(tls_dir(home))
    ca_cert_path(home).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_private(_ca_key_path(home), key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return cert, key


def _load_or_make_ca(home: Path):
    """Reuse the persisted CA when present and unexpired, else mint a new one."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cp, kp = ca_cert_path(home), _ca_key_path(home)
    if cp.exists() and kp.exists():
        try:
            cert = x509.load_pem_x509_certificate(cp.read_bytes())
            key = serialization.load_pem_private_key(kp.read_bytes(), password=None)
            if cert.not_valid_after_utc > _now():
                return cert, key
        except Exception:
            # The files EXIST (guarded above) but are corrupt/unreadable: fall
            # through and mint a fresh CA. The "missing" case is the other branch
            # (the enclosing if), so this only ever handles the genuinely-bad file,
            # by regenerating - never by serving a broken CA.
            pass
    return _make_ca(home)


def _make_leaf(home: Path, ca_cert, ca_key, hostnames: list[str], ips: list[str]):
    """Create + persist the leaf cert/key signed by *ca_cert*, covering the given
    SANs. Records a meta.json describing what it covers."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san: list[x509.GeneralName] = [x509.DNSName(h) for h in hostnames]
    for ip in ips:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            continue

    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localm")]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=_LEAF_DAYS))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=True, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    _lock_down_dir(tls_dir(home))
    _leaf_cert_path(home).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_private(_leaf_key_path(home), key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    try:
        _meta_path(home).write_text(json.dumps({
            "hostnames": hostnames,
            "ips": ips,
            "not_after": cert.not_valid_after_utc.isoformat(),
        }, indent=2), encoding="utf-8")
    except OSError:
        # Best-effort: the meta sidecar is a reuse optimization (lets us skip
        # re-minting when the SANs are unchanged). If it cannot be written the
        # leaf is simply regenerated next time; the cert itself is already saved.
        pass
    return cert


def _cert_sans(cert) -> tuple[set[str], set[str]]:
    """The DNS and (canonicalised) IP SANs present in *cert*."""
    from cryptography import x509
    hosts: set[str] = set()
    ips: set[str] = set()
    try:
        ext = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        hosts.update(ext.get_values_for_type(x509.DNSName))
        ips.update(str(ip) for ip in ext.get_values_for_type(x509.IPAddress))
    except x509.ExtensionNotFound:
        # A cert with no SubjectAlternativeName simply has no SANs to report;
        # return the empty sets. Narrow catch (only ExtensionNotFound), so a
        # malformed-cert error would still surface, not be swallowed here.
        pass
    return hosts, ips


def _leaf_is_reusable(home: Path, hostnames: list[str], ips: list[str]) -> bool:
    """True when the persisted leaf is safe to reuse: it and the CA exist, it is
    not near expiry, it is issued by the current CA, and it already covers every
    required SAN. Anything else (missing, expired, wrong issuer, missing SAN)
    forces a regenerate."""
    from cryptography import x509

    cp, kp, cap = _leaf_cert_path(home), _leaf_key_path(home), ca_cert_path(home)
    if not (cp.exists() and kp.exists() and cap.exists()):
        return False
    try:
        cert = x509.load_pem_x509_certificate(cp.read_bytes())
        ca = x509.load_pem_x509_certificate(cap.read_bytes())
    except Exception:
        return False

    if cert.not_valid_after_utc - timedelta(days=_RENEW_MARGIN_DAYS) <= _now():
        return False
    # Leaf must chain to the CA we currently serve for download, else a trusted
    # device would still see a warning.
    if cert.issuer != ca.subject:
        return False

    have_hosts, have_ips = _cert_sans(cert)
    if not set(hostnames).issubset(have_hosts):
        return False
    if not set(ips).issubset(have_ips):
        return False
    return True


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #

def ensure_cert(
    home: Path,
    hostnames: Optional[Iterable[str]] = None,
    ips: Optional[Iterable[str]] = None,
) -> tuple[str, str]:
    """Ensure a usable server certificate + key exist under ``<home>/tls`` and
    return their paths as ``(cert_path, key_path)``.

    Generates a local CA once (persisted, reused across regenerations) and a
    leaf certificate covering the loopback, LAN, Tailscale, and hostname SANs
    (plus any *hostnames* / *ips* the caller supplies). Reuses an existing leaf
    when it still covers every required SAN, is issued by the current CA, and is
    not near expiry; otherwise regenerates the leaf while reusing the CA so a
    device that trusted localm once stays trusted.
    """
    home = Path(home)
    _lock_down_dir(tls_dir(home))

    want_hosts, want_ips = san_targets(hostnames, ips)

    if _leaf_is_reusable(home, want_hosts, want_ips):
        return str(_leaf_cert_path(home)), str(_leaf_key_path(home))

    ca_cert, ca_key = _load_or_make_ca(home)
    _make_leaf(home, ca_cert, ca_key, want_hosts, want_ips)
    return str(_leaf_cert_path(home)), str(_leaf_key_path(home))
