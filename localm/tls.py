# SPDX-License-Identifier: AGPL-3.0-or-later
"""Built-in TLS for network binds (NET-1)."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from localm.debuglog import logger

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

# Adapter-NAME markers for a VPN/tunnel virtual interface rather than a real
# physical LAN NIC. Most VPN clients (WireGuard, OpenVPN, and virtually every
# commercial VPN app) hand their tunnel adapter an ordinary RFC1918 address -
# e.g. 10.x.x.x - and install it as the default route, which would otherwise be
# indistinguishable from a genuine LAN address by IP range alone.
#
# Matched on WORD BOUNDARIES, never as a bare substring (REG-533). The short
# generic markers are interface *names* (tun0, tap5, utun3) or a leading word in
# a Windows friendly name ("TAP-Windows Adapter V9"), so an unanchored substring
# match also hit any real NIC whose name merely CONTAINS those three letters
# ("Fortune", "Neptune", "Datapath"). A false positive here is not cosmetic:
# _primary_lan_ip() discards the address and san_targets() then leaves it out of
# the leaf certificate, so a device reaching that machine on its real address
# gets a TLS SAN mismatch that used to validate.
#
# "ppp" is deliberately NOT a marker. It is ambiguous - a PPPoE/DSL link (ppp0,
# often the default route and the machine's REAL address) and a legacy PPTP/L2TP
# VPN both use it - and the two failure directions are not symmetric: wrongly
# calling a real WAN a VPN breaks TLS for an address the user genuinely reaches,
# while wrongly calling a VPN real only adds an extra SAN entry / a less useful
# suggested address. Prefer the harmless error.
#
# Best-effort, not exhaustive; matched case-insensitively.
_VPN_ADAPTER_NAME_RE = re.compile(
    r"\b(?:tap|tun|utun|wintun|vpn|wireguard|nordlynx|openvpn|globalprotect"
    r"|anyconnect|zscaler)\d*\b",
    re.IGNORECASE,
)


def _is_vpn_adapter_name(name: str) -> bool:
    """True when an adapter NAME looks like a VPN/tunnel virtual interface."""
    return bool(_VPN_ADAPTER_NAME_RE.search(name or ""))


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
    """IPv4 addresses bound to an adapter whose NAME looks VPN/tunnel-like (best-effort, via psutil when the ``[monitor]`` extra is installed; empty otherwise, same degrade as ``_iface_ips``)."""
    out: set[str] = set()
    try:
        import psutil
    except Exception:
        return out
    try:
        for name, addrs in psutil.net_if_addrs().items():
            if not _is_vpn_adapter_name(name):
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
    """Best-effort primary outbound LAN IPv4, or '' if undetermined."""
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
    """All IPv4 addresses bound to this machine's hostname (best-effort)."""
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
    """Canonicalise an IP string (so ``::1`` and its long form compare equal), or None when it is not a valid IP address."""
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def san_targets(
    extra_hostnames: Optional[Iterable[str]] = None,
    extra_ips: Optional[Iterable[str]] = None,
) -> tuple[list[str], list[str]]:
    """Return ``(hostnames, ip_strings)`` the server cert should cover: loopback, this machine's hostname, its primary LAN IP, any Tailscale IP, plus any caller-supplied extras."""
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
        # belongs in the IP SAN list, not the DNS one. Both wildcards mean "all
        # interfaces" and neither is ever itself a connectable address, so they
        # are dropped rather than certified: nothing can present a certificate
        # for an address no client can dial. ``::1`` is NOT a wildcard and is
        # already in the default IP SAN set above, so an IPv6 loopback bind is
        # covered without a special case here.
        if not h or h in ("0.0.0.0", "::"):
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
    """Every IPv4 bound to a local interface, via psutil when it is installed (the ``[monitor]`` extra)."""
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
    """Best-effort ``{'lan': ip, 'tailscale': ip}`` for the Companion-app card, so a phone is shown a REACHABLE address instead of the meaningless loopback one (``127.0.0.1`` on a phone is the phone itself)."""
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
    """The value to pass as ``requests``' ``verify=`` for an in-process call to *url*."""
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
    except OSError:
        # Defense-in-depth, proven low-risk. mkdir only fails here on a location
        # nothing could write to anyway, which ensure_cert's own writes will hit
        # and surface next; nothing further to report at this layer.
        return
    from localm.config import restrict_file_perms
    if not restrict_file_perms(path, mode=0o700):
        # LM-DA-044: this directory holds the CA private key. Unlike a digest
        # file (sessions.json, jobs.json), a leaked CA key mints certificates
        # every device that ever trusted this install's CA will accept - the
        # compromise is not local - so this is louder than
        # config._perm_warn's debug-only default for an ordinary credential.
        logger.warning(
            "could not restrict %s to the current user; its contents "
            "(including the CA private key) rely on the data directory's own "
            "permissions instead", path)


def _write_private(path: Path, data: bytes) -> None:
    """Write key material at 0600 from the moment it exists, then run the Windows-aware lockdown every other credential file in this project uses."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    from localm.config import restrict_file_perms
    if not restrict_file_perms(path):
        # LM-DA-044: a leaked CA or leaf private key mints certificates every
        # device that ever trusted this install's CA will accept - the
        # compromise is not local - so this is louder than
        # config._perm_warn's debug-only default for an ordinary credential.
        logger.warning(
            "could not restrict %s to the current user; the data "
            "directory's own permissions are this key's only remaining "
            "protection", path.name)


# ------------------------------------------------------------------ #
#  Certificate authority + leaf                                       #
# ------------------------------------------------------------------ #

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_ca(home: Path):
    """Create and persist a fresh local CA (cert + private key)."""
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
    """Create + persist the leaf cert/key signed by *ca_cert*, covering the given SANs."""
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
    """True when the persisted leaf is safe to reuse: it and the CA exist, neither is near its own expiry, the leaf's SIGNATURE actually verifies against the CA we currently serve for download, and the leaf already covers every required SAN."""
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import padding

    cp, kp, cap = _leaf_cert_path(home), _leaf_key_path(home), ca_cert_path(home)
    if not (cp.exists() and kp.exists() and cap.exists()):
        return False
    try:
        cert = x509.load_pem_x509_certificate(cp.read_bytes())
        ca = x509.load_pem_x509_certificate(cap.read_bytes())
    except Exception:
        return False

    # A CA nearing (or past) its own expiry must force regeneration of BOTH:
    # a leaf signed by an expiring CA is no safer than an expiring leaf, and
    # _load_or_make_ca's own expiry check is unreachable in the steady state
    # (ensure_cert only calls it once the leaf is judged NOT reusable).
    if ca.not_valid_after_utc - timedelta(days=_RENEW_MARGIN_DAYS) <= _now():
        return False
    if cert.not_valid_after_utc - timedelta(days=_RENEW_MARGIN_DAYS) <= _now():
        return False

    # Prove the leaf actually chains to the CA we currently serve for
    # download - not merely that their subject/issuer NAMES match.
    try:
        ca.public_key().verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            padding.PKCS1v15(),
            cert.signature_hash_algorithm,
        )
    except (InvalidSignature, ValueError, TypeError):
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
    """Ensure a usable server certificate + key exist under ``<home>/tls`` and return their paths as ``(cert_path, key_path)``."""
    home = Path(home)
    _lock_down_dir(tls_dir(home))

    want_hosts, want_ips = san_targets(hostnames, ips)

    if _leaf_is_reusable(home, want_hosts, want_ips):
        return str(_leaf_cert_path(home)), str(_leaf_key_path(home))

    ca_cert, ca_key = _load_or_make_ca(home)
    _make_leaf(home, ca_cert, ca_key, want_hosts, want_ips)
    return str(_leaf_cert_path(home)), str(_leaf_key_path(home))
