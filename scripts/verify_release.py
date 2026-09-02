#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify a downloaded localm release asset (CHK-UPDATER-INTEGRITY).

Checks a `localm-<version>.zip` release download against the same trust
localm's auto-updater already relies on: an Ed25519 signature against the
public key pinned in `localm/updater.py`, and/or a plain SHA256 digest.

THIS VERIFIES THE GITHUB RELEASE **ASSET** (`localm-<version>.zip` plus its
`localm-<version>.zip.sig`, both listed under a release's "Assets" section),
NOT GitHub's own auto-generated "Source code (zip)" link. The asset is built
from `release-manifest.toml`'s file list and carries a baked version file;
the auto-zipball is the raw tracked tree with neither, so its bytes differ
and a release's signature will never match it.

This only covers a release asset obtained some other way than the recommended
install (a browser download, a mirror, a forwarded copy). The recommended
`git clone` install path (setup.sh / setup.bat / install.sh) never downloads a
zip at all, so it is unaffected by and unchanged by this script - it relies
on git+HTTPS+GitHub the same way any other cloned project does.

Usage:
  python scripts/verify_release.py <path-to-zip> [--sig PATH] [--sha256 HEX_OR_PATH]

Signature check: runs when --sig is given, or, when neither --sig nor
--sha256 is given at all, when a file named "<path-to-zip>.sig" sits next to
the zip (the default name scripts/sign_release.py itself writes). Verifies
against the same pinned Ed25519 key the auto-updater trusts
(localm/updater.py _UPDATE_PUBKEYS) via localm.updater.verify_signature -
never re-implemented here.

SHA256 check: runs when --sha256 is given, either a bare 64-character hex
digest or a path to a file holding one (a plain hex string, or the standard
two-column "<hex>  <filename>" sha256sum format). Needs only the standard
library.

Exit codes:
  0  verified OK (says which check(s) passed)
  1  verification FAILED (tampered bytes, a wrong or unmatched key, or a
     digest mismatch)
  2  could not verify at all (missing file, a needed dependency is not
     installed, no --sig/--sha256 given and none inferrable, or bad
     arguments) - never reported as success
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def sha256_of(path: Path) -> str:
    """The lowercase hex SHA256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_hex64(s: str) -> bool:
    return len(s) == 64 and all(c in _HEX_CHARS for c in s)


def parse_sha256_arg(value: str) -> str:
    """A bare lowercase 64-hex-character digest from *value*: either the
    digest itself, or a path to a file holding one (a plain hex string, or
    the standard two-column "<hex>  <filename>" sha256sum format - only the
    first whitespace-separated token of the first line is read). Raises
    ValueError if neither applies."""
    stripped = value.strip()
    if _looks_like_hex64(stripped):
        return stripped.lower()
    p = Path(value)
    if not p.is_file():
        raise ValueError(
            f"--sha256 value {value!r} is neither a 64-character hex digest "
            "nor an existing file")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"could not read {value}: {e}") from e
    lines = text.splitlines()
    token = lines[0].split()[0] if lines and lines[0].split() else ""
    if not _looks_like_hex64(token):
        raise ValueError(
            f"{value}: first line does not start with a 64-character hex digest")
    return token.lower()


def verify_ed25519(data: bytes, sig_text: str) -> tuple[bool, str]:
    """Verify *sig_text* (base64 Ed25519 signature) over *data* against the
    key pinned in localm.updater - the same check and the same pinned key
    the auto-updater uses, reused rather than reimplemented. Returns
    (ok, message). Raises ImportError if localm.updater or one of its own
    dependencies (cryptography, rich) is not installed."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from localm import updater
    from localm.bugreport import LocalmError
    try:
        updater.verify_signature(data, sig_text)
    except LocalmError as e:
        detail = f"{e.summary} ({e.reason})" if e.reason else e.summary
        return False, detail
    return True, "verifies against the pinned update key (localm/updater.py _UPDATE_PUBKEYS)"


def _import_error_message(e: BaseException) -> str:
    return (
        f"could not check the signature: {e}. This needs the same base "
        "dependencies (cryptography, rich) a set-up localm install already "
        "has. Run this from an already-set-up localm .venv "
        "(.venv/bin/python or .venv\\Scripts\\python.exe "
        "scripts/verify_release.py ...), install the missing package "
        "directly (pip install cryptography), or, if uv is already on this "
        "machine (setup.bat/setup.sh install it): "
        "uv run --with cryptography python scripts/verify_release.py ...")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        add_help=True, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip_path", type=Path, metavar="PATH",
                   help="the downloaded release zip to verify")
    p.add_argument("--sig", type=Path, default=None,
                   help="path to the .sig file (default: <PATH>.sig if it exists)")
    p.add_argument("--sha256", default=None,
                   help="expected SHA256: a 64-char hex digest, or a file holding one")
    args = p.parse_args(argv)

    zip_path = args.zip_path
    if not zip_path.is_file():
        print(f"error: file not found: {zip_path}", file=sys.stderr)
        return 2

    digest = sha256_of(zip_path)
    print(f"SHA256 ({zip_path}): {digest}")

    checks_run: list = []      # [(name, ok, message), ...]
    could_not: list = []       # [(name, message), ...]

    sig_source = args.sig
    if sig_source is None and args.sha256 is None:
        candidate = Path(str(zip_path) + ".sig")
        if candidate.is_file():
            sig_source = candidate

    if sig_source is not None:
        if not sig_source.is_file():
            could_not.append(("signature", f"--sig file not found: {sig_source}"))
        else:
            sig_text = sig_source.read_text(encoding="utf-8", errors="replace").strip()
            try:
                ok, msg = verify_ed25519(zip_path.read_bytes(), sig_text)
            except ImportError as e:
                could_not.append(("signature", _import_error_message(e)))
            else:
                checks_run.append(("signature", ok, msg))

    if args.sha256 is not None:
        try:
            expected = parse_sha256_arg(args.sha256)
        except ValueError as e:
            could_not.append(("sha256", str(e)))
        else:
            ok = expected == digest
            msg = ("matches the computed digest" if ok else
                   f"does NOT match - expected {expected}, computed {digest}")
            checks_run.append(("sha256", ok, msg))

    for name, ok, msg in checks_run:
        print(f"{name}: {'OK' if ok else 'FAILED'} - {msg}")
    for name, msg in could_not:
        print(f"{name}: COULD NOT CHECK - {msg}", file=sys.stderr)

    if not checks_run and not could_not:
        print("error: nothing to verify against - pass --sig, --sha256, or "
              "place a <PATH>.sig file next to PATH", file=sys.stderr)
        p.print_usage(sys.stderr)
        return 2
    if any(not ok for _, ok, _ in checks_run):
        return 1
    if checks_run:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
