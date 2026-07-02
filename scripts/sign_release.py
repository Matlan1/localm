#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sign a localm release build for the self-updater (CHK-UPDATER-INTEGRITY).

The updater verifies each release ``build.zip`` against an Ed25519 PUBLIC key
pinned in ``localm/updater.py`` (``_UPDATE_PUBKEYS``). Applying an update executes
the downloaded code, so a build that does not verify is never installed. This
script is the maintainer-side companion:

  1. Generate the keypair ONCE (offline):

         python scripts/sign_release.py --generate-key

     Writes the PRIVATE key to ``update_signing_key.pem`` (gitignored) and prints
     the PUBLIC key hex. Paste that hex into ``_UPDATE_PUBKEYS`` in
     ``localm/updater.py`` and ship it. Keep the PRIVATE key OFFLINE - never in the
     repo, the update proxy, or CI. Anyone with it can push code to every install.

  2. Sign each release build:

         python scripts/sign_release.py sign build.zip --key update_signing_key.pem

     Prints the base64 signature and writes ``build.zip.sig``. Serve that signature
     from the update proxy's ``/update`` JSON (the ``signature`` field) so the
     updater can fetch it with the release metadata.

Only the PUBLIC key is embedded in the shipped code (public data, pinned so an
attacker cannot swap it); the PRIVATE key is the sole trust anchor and stays with
the maintainer.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path


def _gen(out: Path) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if out.exists():
        print(f"refusing to overwrite an existing key file: {out}\n"
              "move it aside first (losing a signing key means re-pinning a new "
              "public key in a shipped release).", file=sys.stderr)
        return 2
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    out.write_bytes(pem)
    try:
        out.chmod(0o600)   # best-effort: restrict on POSIX (no-op semantics on Windows)
    except OSError:
        pass
    pub_hex = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    print(f"PRIVATE key written to {out} - keep this OFFLINE and out of git.\n")
    print("Pin this PUBLIC key in localm/updater.py _UPDATE_PUBKEYS:\n")
    print(f'    _UPDATE_PUBKEYS = (\n        "{pub_hex}",\n    )\n')
    return 0


def _sign(build: Path, key: Path, out: Path | None) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not build.is_file():
        print(f"build not found: {build}", file=sys.stderr)
        return 2
    if not key.is_file():
        print(f"signing key not found: {key}", file=sys.stderr)
        return 2
    priv = serialization.load_pem_private_key(key.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        print("the key file is not an Ed25519 private key", file=sys.stderr)
        return 2
    sig = priv.sign(build.read_bytes())
    sig_b64 = base64.b64encode(sig).decode("ascii")
    sig_path = out or Path(str(build) + ".sig")
    sig_path.write_text(sig_b64 + "\n", encoding="utf-8")
    print(f"signature written to {sig_path}\n")
    print("base64 signature (serve this from the update proxy /update JSON "
          "'signature' field):\n")
    print(sig_b64)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sign a localm release build for the self-updater.")
    p.add_argument("--generate-key", action="store_true",
                   help="generate a new Ed25519 signing keypair")
    p.add_argument("--out", type=Path, default=Path("update_signing_key.pem"),
                   help="key output path for --generate-key (default: update_signing_key.pem)")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("sign", help="sign a build.zip with the private key")
    sp.add_argument("build", type=Path, help="the release build.zip to sign")
    sp.add_argument("--key", type=Path, required=True, help="the Ed25519 private key PEM")
    sp.add_argument("--out", type=Path, default=None,
                    help="signature output path (default: <build>.sig)")

    args = p.parse_args(argv)
    if args.generate_key:
        return _gen(args.out)
    if args.cmd == "sign":
        return _sign(args.build, args.key, args.out)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
