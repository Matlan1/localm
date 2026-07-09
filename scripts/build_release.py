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

Pass ``--commit <sha>`` to build from a PINNED commit instead of the live working
tree: the release file list, the release-manifest.toml include/exclude patterns that
select members from it, AND every member's bytes all come from the git object
database at that exact commit (``git ls-tree`` + ``git show`` + ``git archive``), not
from whatever is on disk right now. This is what scripts/make_release.py uses for a
signed release, so a tracked-file edit OR a manifest-pattern edit landing on disk
after the commit was gated (e.g. during the pre-publish CI wait) cannot change what
ships.

Hand the output to the signer:
    python scripts/build_release.py --out build.zip
    python scripts/sign_release.py sign build.zip --key update_signing_key.pem

Usage:
    python scripts/build_release.py [--out build.zip] [--force] [--commit <sha>]
"""

from __future__ import annotations

import argparse
import io
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

# Import the manifest checker as a library (same dir). It owns the pattern-matching
# and classification so the builder and the gate never disagree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_manifest as cm   # noqa: E402


def _git_archive_zip(commit: str, repo: Path) -> zipfile.ZipFile:
    """The full git tree at *commit*, as an in-memory zip - the read source for a
    PINNED-commit build. File bytes come from the git object database at the exact
    commit that was gated and CI-validated, never from whatever happens to be sitting
    on disk right now. This closes the release TOCTOU: a tracked-file edit made after
    the commit/HEAD gates passed (e.g. during the multi-minute CI-wait window) cannot
    change what ships. *repo* is an explicit argument (not read from ``cm.REPO``) so
    this is testable against a throwaway scratch repo, not only the real checkout."""
    r = subprocess.run(["git", "archive", "--format=zip", commit], cwd=str(repo),
                       capture_output=True)
    if r.returncode != 0:
        raise ValueError(f"git archive {commit!r} failed: "
                         f"{r.stderr.decode('utf-8', 'replace').strip()}")
    return zipfile.ZipFile(io.BytesIO(r.stdout))


def build(out: Path, *, force: bool = False, commit: str | None = None) -> list[str]:
    """Write the release build.zip to *out* and return the sorted list of members.

    Fails (raises ValueError) if the manifest gate is red, if the tree is not a git
    checkout, or if the required root files are missing.

    *commit*: when given (including ``""`` - checked via ``is not None``, not
    truthiness, so an accidentally-empty value fails loud instead of silently
    degrading to a disk build), EVERYTHING that decides what ships is read from the
    git tree at that commit, never from the live working tree: the release file LIST
    (``git ls-tree``), the release-manifest.toml INCLUDE/EXCLUDE PATTERNS that select
    members from that list (``git show``), and every member's BYTES (``git archive``).
    release-manifest.toml is itself a tracked file, so pinning only the file list and
    bytes (and reading the manifest live) would leave a narrower but still-open gap: a
    manifest-pattern edit landing on disk during the pre-publish CI wait could still
    change which already-tracked files ship, unvalidated by CI. Pinning the manifest
    too closes that. This is the pinned-commit path scripts/make_release.py uses for a
    signed release. When omitted (the default, plain dev-build entry point), the file
    list, manifest, and bytes are all read from disk, as before.

    KNOWN LIMITATION: the commit-mode manifest gate runs the pure
    :func:`check_manifest.classify_problems` (file-classification drift only), not
    the full live-disk :func:`check_manifest.check_manifest`, so it does NOT run
    ``check_manifest._local_ignore_problems`` (the ``git check-ignore`` cross-check
    confirming a ``local_only`` pattern is still genuinely covered by .gitignore) -
    that check is inherently a live-working-tree operation with no clean "as of a
    past commit" analogue. For scripts/make_release.py's --publish flow this is
    covered in practice: CI (which must pass before build() runs) independently runs
    the full live check_manifest() via check_hygiene.py against a checkout of the
    same commit. A standalone ``--commit`` invocation of this script outside that
    flow does not get that cross-check and should not be treated as a full manifest-
    hygiene gate on its own."""
    if commit is not None:
        tracked = cm.tracked_files_at(commit)
        if tracked is None:
            raise ValueError(f"could not list the git tree at commit {commit!r} "
                             "(git ls-tree unavailable or the commit does not "
                             "resolve) - cannot determine the release file set")
        man = cm.load_manifest_at(commit)
        # classify_problems() covers file classification (unclassified/ambiguous/
        # stale-pattern/local-only-leak) but not the check-ignore cross-check - see
        # "KNOWN LIMITATION" above.
        problems = cm.classify_problems(tracked, man["include"], man["exclude"], man["local_only"])
        if problems:
            raise ValueError(
                f"refusing to build from a manifest that does not verify at the "
                f"pinned commit {commit} - fix these first (run "
                "scripts/check_manifest.py):\n  " + "\n  ".join(problems))
    else:
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
    if commit is not None:
        archive = _git_archive_zip(commit, cm.REPO)
        archived = set(archive.namelist())
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in members:
                if rel not in archived:
                    # A tracked git submodule (gitlink, mode 160000) archives as a
                    # DIRECTORY-PLACEHOLDER zip entry named "name/" (git does not
                    # recurse into it without --recurse-submodules), never the bare
                    # "name" git ls-tree / release_files() produced - give a clear,
                    # intentional error for that specific, expected case instead of
                    # letting it fall through to the generic mismatch message below.
                    if (rel + "/") in archived:
                        raise ValueError(
                            f"tracked release member archived as a directory, not a "
                            f"regular file: {rel!r} (at commit {commit}) - likely a "
                            "git submodule (gitlink); a pinned-commit build cannot "
                            "safely ship it; resolve or exclude it in "
                            "release-manifest.toml")
                    raise ValueError(
                        f"tracked release member missing from the git archive at "
                        f"{commit}: {rel!r} (tree/index mismatch?)")
                info = archive.getinfo(rel)
                # Defense in depth for the directory-placeholder case above, in case
                # a future git/zip representation ever emitted the BARE name (no
                # trailing slash) for a non-regular entry - would otherwise silently
                # slip past the rel+"/" check.
                if info.is_dir():
                    raise ValueError(
                        f"tracked release member archived as a directory, not a "
                        f"regular file: {rel!r} (at commit {commit}) - likely a git "
                        "submodule (gitlink); a pinned-commit build cannot safely "
                        "ship it; resolve or exclude it in release-manifest.toml")
                # A tracked symlink (git mode 120000) archives as a real symlink
                # entry whose "content" is the raw target-PATH STRING, not the
                # referenced file's bytes (unlike the disk-read branch below, which
                # follows a working symlink via Path.is_file()/ZipFile.write and
                # ships the real target content). Copying that path string through
                # would silently corrupt the shipped file - the updater's plain
                # zipfile.extractall() does not reconstruct symlinks, so it would
                # land on a user's disk as a small text file containing the target
                # path. Fail loud instead of shipping wrong bytes (AGENTS.md rule 5).
                mode = info.external_attr >> 16
                if mode and stat.S_ISLNK(mode):
                    raise ValueError(
                        f"tracked release member is a symlink, not a regular file: "
                        f"{rel!r} (at commit {commit}) - a pinned-commit build cannot "
                        "safely ship it; resolve or exclude it in release-manifest.toml")
                member = zipfile.ZipInfo(rel, date_time=info.date_time)
                member.external_attr = info.external_attr   # preserve the file mode
                member.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(member, archive.read(rel))
    else:
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
    p.add_argument("--commit", default=None,
                   help="read the release file list, the release-manifest.toml "
                        "include/exclude patterns, AND every file's bytes from this "
                        "git commit instead of the live working tree - the "
                        "pinned-commit path a signed release build uses")
    args = p.parse_args(argv)
    try:
        members = build(args.out, force=args.force, commit=args.commit)
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
