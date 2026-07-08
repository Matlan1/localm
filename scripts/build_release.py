#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Assemble a localm release build.zip from RELEASE-INCLUDE (NEW-RELEASE-FILEMANIFEST).

The self-updater (localm/updater.py -> localm/_apply_update.py) downloads a
``build.zip``, verifies its signature, then swaps every top-level entry (except a
hard NEVER_TOUCH set) into the install. So build.zip IS the release artifact. This
script assembles it from the SAME source of truth the manifest gate enforces: every
git-tracked file that matches a ``release.include`` pattern and no ``release.exclude``
pattern in ``release-manifest.toml``. Dev-only files (tests/, .github/, tools/, the
GUI test harness, dev scripts) are left out.

The archive is FLAT (paths relative to the repo root, so ``VERSION`` and
``pyproject.toml`` sit at the zip root exactly as ``_apply_update.verify_zip``
requires). After writing it, this self-verifies with ``verify_zip`` when localm is
importable, so a mispackaged build fails here, not on a user's machine.

It also refuses to build from a DIRTY manifest: if the classification gate is red,
the include set is untrustworthy, so it stops. Fix the manifest first.

Hand the output to the signer:
    python scripts/build_release.py --out build.zip
    python scripts/sign_release.py sign build.zip --key update_signing_key.pem

Usage:
    python scripts/build_release.py [--out build.zip] [--force]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Import the manifest checker as a library (same dir). It owns the pattern-matching
# and classification so the builder and the gate never disagree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_manifest as cm   # noqa: E402


def build(out: Path, *, force: bool = False) -> list[str]:
    """Write the release build.zip to *out* and return the sorted list of members.

    Fails (raises ValueError) if the manifest gate is red, if the tree is not a git
    checkout, or if the required root files are missing."""
    problems = cm.check_manifest()
    if problems:
        raise ValueError(
            "refusing to build from a manifest that does not verify - fix these "
            "first (run scripts/check_manifest.py):\n  " + "\n  ".join(problems))

    man = cm.load_manifest()
    tracked = cm.tracked_files()
    if tracked is None:
        raise ValueError("not a git checkout (git ls-files unavailable) - cannot "
                         "determine the release file set")
    members = cm.release_files(tracked, man["include"], man["exclude"])

    # Sanity: the two files the updater's own gates require MUST be present, or the
    # build would be rejected on the user's side. Fail here instead (fail loud).
    for required in ("VERSION", "pyproject.toml"):
        if required not in members:
            raise ValueError(f"release set is missing {required!r} - build.zip would "
                             "fail updater.verify_zip; check release-manifest.toml")

    if out.exists() and not force:
        raise ValueError(f"{out} already exists (pass --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic, sorted: a plain DEFLATE zip of the tracked bytes at their
    # repo-relative paths.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in members:
            src = cm.REPO / rel
            if not src.is_file():   # a tracked path that is not a regular file (submodule/symlink)
                raise ValueError(f"tracked release member is not a regular file: {rel}")
            z.write(src, arcname=rel)

    _self_verify(out)
    return members


def _self_verify(out: Path) -> None:
    """Prove the archive passes the updater's own acceptance gate, when localm is
    importable. Not a silent skip: if localm cannot be imported (running the script
    outside an install), say so - the build still stands, but it went unverified."""
    # The repo root holds the localm/ package; add it so this works from a plain dev
    # checkout too, not only a pip-installed one (running `python scripts/build_release.py`
    # puts scripts/ on sys.path, not the repo root). _apply_update is stdlib-only.
    if str(cm.REPO) not in sys.path:
        sys.path.insert(0, str(cm.REPO))
    try:
        from localm import _apply_update as au
    except Exception as e:
        print(f"note: localm not importable, skipped verify_zip self-check ({e})",
              file=sys.stderr)
        return
    au.verify_zip(out)   # raises ValueError if the archive does not look like a build


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assemble a localm release build.zip from "
                                            "RELEASE-INCLUDE (release-manifest.toml).")
    p.add_argument("--out", type=Path, default=Path("build.zip"),
                   help="output path (default: build.zip)")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = p.parse_args(argv)
    try:
        members = build(args.out, force=args.force)
    except ValueError as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({len(members)} files, {size/1024:.0f} KiB)")
    print("next: sign it with scripts/sign_release.py sign "
          f"{args.out} --key update_signing_key.pem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
