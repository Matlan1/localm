#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut a SIGNED localm release build (NEW-RELEASE-FILEMANIFEST + CHK-UPDATER-INTEGRITY).

One command for the release signer: assemble the build.zip from release-manifest.toml,
sign it with the offline Ed25519 private key, and (optionally) publish both to a GitHub
Release. A self-updating client downloads the build via the proxy, reads the signature
from the proxy's /update JSON, and verifies it against the PUBLIC key pinned in
localm/updater.py before applying.

  export LOCALM_SIGNING_KEY=/path/to/update_signing_key.pem   # kept OUT of the repo
  python scripts/make_release.py                              # build + sign -> dist/
  python scripts/make_release.py --publish                    # + gh release create

The signing-key PATH comes from --key or $LOCALM_SIGNING_KEY - deliberately NOT baked
into this tracked file (which must stay machine-path-free). Before publishing, this
SELF-CHECKS that the signature verifies against the key pinned in
updater._UPDATE_PUBKEYS, so a build the shipped clients would reject is never released.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts
import build_release  # noqa: E402
import sign_release    # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _pinned_pubkeys() -> tuple:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from localm import updater
    return tuple(updater._UPDATE_PUBKEYS)


def _verify_against_pinned(zip_path: Path, sig_path: Path) -> None:
    """Confirm the freshly-made signature verifies against a key pinned in the shipped
    updater. This is the gate that stops us publishing a build clients cannot verify -
    e.g. signed with a key whose public half was never pinned, or a stale pin."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    keys = _pinned_pubkeys()
    if not keys:
        raise SystemExit(
            "updater._UPDATE_PUBKEYS is EMPTY: pin the release public key before cutting a "
            "signed release, or a keyed client cannot verify this build.")
    data = zip_path.read_bytes()
    sig = base64.b64decode(sig_path.read_text(encoding="utf-8").strip(), validate=True)
    for hexkey in keys:
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(hexkey).strip())).verify(sig, data)
            return
        except (InvalidSignature, ValueError):
            continue
    raise SystemExit(
        "the signature does NOT verify against any key pinned in updater._UPDATE_PUBKEYS - "
        "the signing key and the pinned public key disagree. Refusing to publish a build the "
        "shipped clients would reject.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Assemble + sign a localm release build.")
    p.add_argument("--key", type=Path, default=None,
                   help="Ed25519 private key PEM (else $LOCALM_SIGNING_KEY)")
    p.add_argument("--out", type=Path, default=None,
                   help="build.zip path (default: dist/localm-<version>.zip)")
    p.add_argument("--publish", action="store_true",
                   help="gh release create vX.Y.Z with the zip + .sig")
    args = p.parse_args(argv)

    keypath = args.key
    if keypath is None and os.environ.get("LOCALM_SIGNING_KEY"):
        keypath = Path(os.environ["LOCALM_SIGNING_KEY"])
    if keypath is None:
        raise SystemExit("no signing key: pass --key or set LOCALM_SIGNING_KEY (kept OUT of the repo).")
    if not keypath.is_file():
        raise SystemExit(f"signing key not found: {keypath}")

    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    out = args.out or (REPO / "dist" / f"localm-{version}.zip")

    # 1. assemble from the manifest (refuses a dirty manifest; self-verifies verify_zip)
    members = build_release.build(out, force=True)
    # 2. sign it (writes <out>.sig)
    sig_path = Path(str(out) + ".sig")
    if sign_release._sign(out, keypath, sig_path) != 0:
        raise SystemExit("signing failed")
    # 3. self-check: the signed build must verify against the SHIPPED pinned key
    _verify_against_pinned(out, sig_path)
    print(f"built + signed {out} ({len(members)} files) and {sig_path.name}")
    print("signature verifies against the pinned key in updater._UPDATE_PUBKEYS")

    tag = f"v{version}"
    if args.publish:
        cmd = ["gh", "release", "create", tag, str(out), str(sig_path),
               "--title", version, "--notes", f"localm {version}"]
        print("publishing:", " ".join(cmd))
        if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            raise SystemExit("gh release create failed")
        print(f"published {tag}")
    else:
        print(f"\nnext: gh release create {tag} {out} {sig_path} --title {version} --notes ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
