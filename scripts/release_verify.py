#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-publish LIVE functional verification of a release build (cold install).

A release must not be published until the release build has been COLD-INSTALLED and
every feature the changelog claims has been exercised for REAL - a model loads AND
chat/RAG/etc. actually work front to back, not merely "it imports". This is the one
gate CI cannot cover: it needs real model runs, the GUI, media backends, and human/
agent judgment. The procedure and the changelog-to-check mapping live in
RELEASE.md.

This script provides:
  - `cold-install`: extract a build.zip into a THROWAWAY dir, make a fresh venv, install
    the package + a native runtime backend, so you get a clean instance matching what a
    user gets, with NO shared state with the dev checkout. You then drive it by hand.
  - the record helpers behind the make_release --publish gate: publishing refuses unless
    a PASSING verification record exists for the EXACT release commit.

Flow:  make_release (build) -> release_verify cold-install -> exercise per RELEASE.md ->
       record the verdict (dev-notes/release-verify/<sha>.md) -> make_release --publish.

Stdlib only for the record helpers (so make_release can import them anywhere);
cold-install shells out to uv / localm.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_RECORD_DIR = "dev-notes/release-verify"


# --------------------------------------------------------------------------- #
#  verification record (backs the make_release --publish gate)                #
# --------------------------------------------------------------------------- #

def current_sha(repo: Path = REPO) -> str | None:
    """The full HEAD commit sha, or None when git is unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out or None


def record_path(sha: str, repo: Path = REPO) -> Path:
    """Where the functional-verification record for commit *sha* lives (local, in
    dev-notes/, gitignored - it is release evidence for the release machine)."""
    return repo / _RECORD_DIR / f"{sha}.md"


def read_verdict(sha: str, repo: Path = REPO) -> str | None:
    """The recorded verdict for *sha* ("PASS"/"FAIL"/other), or None if no record.

    The record is a markdown file with a line ``VERDICT: PASS`` (or FAIL). The exact
    sha keying it to the commit is what makes a stale record for older code useless."""
    p = record_path(sha, repo)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.upper().startswith("VERDICT:"):
            return s.split(":", 1)[1].strip().upper()
    return None


def has_passing_record(sha: str, repo: Path = REPO) -> bool:
    return read_verdict(sha, repo) == "PASS"


# --------------------------------------------------------------------------- #
#  cold install (reproducible clean instance from a build.zip)                #
# --------------------------------------------------------------------------- #

def _default_backend() -> str:
    """The backend a user on THIS machine would actually run (hwdetect recommendation),
    so the cold install exercises the REAL GPU path, not a cpu lowest-common-denominator
    that could hide a GPU-path bug. Vendor-neutral 'vulkan' if detection fails."""
    try:
        from localm import hwdetect
        return hwdetect.detect().recommended or "vulkan"
    except Exception:
        return "vulkan"


def cold_install(zip_path: Path, dest: Path, *, extras: str = "coder,voice,monitor",
                 backend: str | None = None, runner=None) -> Path:
    """Extract *zip_path* into *dest* (cleared), make a fresh venv, install the package
    editable with *extras*, and provision the native llama runtime. *backend* defaults to
    the hardware-detected one (the AMD GPU here), NOT cpu - verify the path users run.
    Returns the venv's localm launcher dir. This is what a user gets - isolated, no
    dev-checkout state. Raises SystemExit on any step failure (never a partial 'ok').

    *runner* is injectable for tests; defaults to subprocess.run with check."""
    import shutil
    run = runner or (lambda cmd, **kw: subprocess.run(cmd, check=True, **kw))
    backend = backend or _default_backend()
    zip_path, dest = Path(zip_path), Path(dest)
    if not zip_path.is_file():
        raise SystemExit(f"cold-install: build not found: {zip_path}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    root = dest
    if not (root / "pyproject.toml").is_file():   # descend into a single wrapper dir
        subs = [p for p in dest.iterdir() if p.is_dir()]
        if len(subs) == 1 and (subs[0] / "pyproject.toml").is_file():
            root = subs[0]
    venv = root / ".venv"
    try:
        run(["uv", "venv", str(venv), "--python", "3.12"], cwd=str(root))
        run(["uv", "pip", "install", "--python", str(venv),
             "-e", (f".[{extras}]" if extras else ".")], cwd=str(root))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(f"cold-install: venv/install failed ({e}). Is uv installed?")
    localm = venv / ("Scripts" if sys.platform == "win32" else "bin")
    py = localm / ("python.exe" if sys.platform == "win32" else "python")
    try:
        run([str(py), "-m", "localm", "setup-llama", "--backend", backend, "--yes"], cwd=str(root))
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"cold-install: setup-llama --backend {backend} failed ({e}).")
    print(f"cold install ready at {root}  (native backend: {backend})\n"
          f"  launcher dir: {localm}\n"
          f"  drive it (throwaway home!):  LOCALM_HOME=<tmp> \"{py}\" -m localm ...\n"
          f"  then verify per RELEASE.md and record the verdict.")
    return localm


def changed_areas(since: str, repo: Path = REPO) -> list[str]:
    """Top-level dirs/files that changed since ref *since* (the last release tag),
    sorted and de-duplicated. Used to SCOPE the verification: real-run the features whose
    code moved; the rest are covered by their tests + the import/smoke gate, so there is
    no need to re-generate an image or re-run every model when that path did not change."""
    try:
        out = subprocess.run(["git", "diff", "--name-only", f"{since}..HEAD"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    tops = {line.split("/", 1)[0] for line in out.splitlines() if line}
    return sorted(tops)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-publish live release verification helper.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ci = sub.add_parser("cold-install", help="cold-install a build.zip into a throwaway dir")
    ci.add_argument("--zip", type=Path, required=True)
    ci.add_argument("--dest", type=Path, required=True)
    ci.add_argument("--extras", default="coder,voice,monitor")
    ci.add_argument("--backend", default=None,
                    help="native backend (default: hardware-detected, e.g. the AMD GPU - "
                         "not cpu, so the real GPU path is exercised)")
    ck = sub.add_parser("check", help="exit 0 iff a PASSING record exists for a sha (default HEAD)")
    ck.add_argument("sha", nargs="?", default=None)
    ch = sub.add_parser("changed", help="top-level areas changed since a ref (to scope the pass)")
    ch.add_argument("--since", required=True, help="the last release tag/ref, e.g. v0.1.0")
    args = p.parse_args(argv)

    if args.cmd == "cold-install":
        cold_install(args.zip, args.dest, extras=args.extras, backend=args.backend)
        return 0
    if args.cmd == "changed":
        areas = changed_areas(args.since)
        print("changed since", args.since + ":", " ".join(areas) or "(nothing)")
        print("real-run the features whose code moved above; spot-check the rest "
              "(their tests + the smoke gate cover unchanged paths).")
        return 0
    if args.cmd == "check":
        sha = args.sha or current_sha()
        if not sha:
            print("release verify: cannot determine sha", file=sys.stderr)
            return 1
        if has_passing_record(sha):
            print(f"PASS record found for {sha[:12]}")
            return 0
        print(f"no PASS record for {sha[:12]} (see {record_path(sha)})", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
