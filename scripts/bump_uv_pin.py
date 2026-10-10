#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing the pinned uv (Astral's package manager).

uv is pinned in five places that must always name the same release:

  setup.sh, setup-gui.sh     UV_INSTALLER_VERSION and UV_INSTALLER_SHA256, the
                             sha256 of the release's uv-installer.sh
  setup.bat, setup-gui.bat   UV_INSTALLER_VERSION and UV_INSTALLER_SHA256, the
                             sha256 of the release's uv-installer.ps1
  docker/Dockerfile          ARG UV_VERSION and ARG UV_SHA256, the sha256 of the
                             release's uv-x86_64-unknown-linux-gnu.tar.gz

This script rewrites all ten values in one pass. Without ``--write`` it prints a
unified diff and changes nothing.

DIGEST SOURCE. The sha256 values come from the ``digest`` field the GitHub
releases API publishes for each asset (the hash of the exact asset). Nothing is
downloaded to be hashed.

WHAT IT CHECKS BEFORE WRITING:
  * the receipt (from ``scripts/confirm_uv_runtime.py --tag <tag> --receipt``)
    is schema 1, component uv, names exactly this tag, is not a ``--current``
    receipt, has verdict PASS, carries every required check and has each
    required check at PASS;
  * the asset digests recorded in the receipt equal the digests the API
    publishes now;
  * the tag is a plain MAJOR.MINOR.PATCH release, is not older than any of the
    five pinned versions, and is newer than at least one of them;
  * every edited region is found exactly once in its file;
  * after the rewrite, reading the five files back yields the tag and the
    digest for each.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused.

Environment: GITHUB_TOKEN is sent as a bearer token for the release lookup.

Usage:
    python scripts/bump_uv_pin.py --tag 0.14.0                       # dry run, no receipt
    python scripts/bump_uv_pin.py --tag 0.14.0 --receipt confirm.json
    python scripts/bump_uv_pin.py --tag 0.14.0 --receipt confirm.json --write

Needs localm importable (the verified HTTPS opener). Nothing under localm/
imports this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.request
from typing import NamedTuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

UPSTREAM_REPO = "astral-sh/uv"
RELEASE_URL = "https://api.github.com/repos/%s/releases/tags/%s"

ASSET_SH = "uv-installer.sh"
ASSET_PS1 = "uv-installer.ps1"
ASSET_LINUX = "uv-x86_64-unknown-linux-gnu.tar.gz"
ASSETS = (ASSET_SH, ASSET_PS1, ASSET_LINUX)

# The checks scripts/confirm_uv_runtime.py must report, each required and PASS.
REQUIRED_CHECKS = (
    "release_listing", "installer_digests", "isolation", "installer_run",
    "containment", "version", "venv_python", "pip_install", "lock_check",
)

_TAG_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


class Site(NamedTuple):
    """One file that pins uv: its path, the asset whose digest it carries and
    the two regexes that locate its version and sha256 values."""

    path: str
    asset: str
    version_re: re.Pattern
    sha_re: re.Pattern


def _quoted(name: str) -> re.Pattern:
    return re.compile(rf'^(?P<head>{name}=")(?P<value>[^"\n]*)(?P<tail>")[ \t]*$', re.M)


def _bat(name: str) -> re.Pattern:
    return re.compile(rf'^(?P<head>[ \t]*set "{name}=)(?P<value>[^"\n]*)(?P<tail>")[ \t]*$', re.M)


def _arg(name: str) -> re.Pattern:
    return re.compile(rf'^(?P<head>ARG {name}=)(?P<value>[^\s]*)(?P<tail>)[ \t]*$', re.M)


SITES = (
    Site("setup.sh", ASSET_SH, _quoted("UV_INSTALLER_VERSION"), _quoted("UV_INSTALLER_SHA256")),
    Site("setup-gui.sh", ASSET_SH, _quoted("UV_INSTALLER_VERSION"), _quoted("UV_INSTALLER_SHA256")),
    Site("setup.bat", ASSET_PS1, _bat("UV_INSTALLER_VERSION"), _bat("UV_INSTALLER_SHA256")),
    Site("setup-gui.bat", ASSET_PS1, _bat("UV_INSTALLER_VERSION"), _bat("UV_INSTALLER_SHA256")),
    Site("docker/Dockerfile", ASSET_LINUX, _arg("UV_VERSION"), _arg("UV_SHA256")),
)


# --------------------------------------------------------------------------- #
#  Versions and file text                                                      #
# --------------------------------------------------------------------------- #

def parse_semver(text: str) -> tuple:
    """(major, minor, patch) for a plain MAJOR.MINOR.PATCH release string."""
    if not isinstance(text, str) or not _TAG_RE.match(text):
        raise Refused(f"{text!r} is not a uv release version (MAJOR.MINOR.PATCH)")
    return tuple(int(p) for p in text.split("."))


def _read(path: Path) -> tuple[str, str]:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def _once(pattern: re.Pattern, text: str, what: str) -> re.Match:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    return matches[0]


def read_site(site: Site, text: str) -> tuple[str, str]:
    """(version, sha256) currently pinned in *text* for *site*."""
    version = _once(site.version_re, text, f"{site.path} version").group("value")
    sha = _once(site.sha_re, text, f"{site.path} sha256").group("value")
    return version, sha


def _replace(pattern: re.Pattern, text: str, value: str, what: str) -> str:
    m = _once(pattern, text, what)
    return text[:m.start("value")] + value + text[m.end("value"):]


def rewrite_site(site: Site, text: str, tag: str, digests: dict) -> str:
    """*text* with the site's version set to *tag* and its sha256 set to the
    digest of the asset the site pins."""
    text = _replace(site.version_re, text, tag, f"{site.path} version")
    return _replace(site.sha_re, text, digests[site.asset], f"{site.path} sha256")


def check_forward(versions: dict, tag: str) -> None:
    """Raise Refused unless *tag* is not older than any pinned version and is
    newer than at least one."""
    target = parse_semver(tag)
    pinned = {path: parse_semver(v) for path, v in versions.items()}
    newest = max(pinned.values())
    oldest = min(pinned.values())
    if target < newest:
        holder = next(p for p, v in pinned.items() if v == newest)
        raise Refused(f"{tag} is older than the version pinned in {holder} "
                      f"({versions[holder]}); a bump only moves forward")
    if target == oldest:
        raise Refused(f"every pin is already at {tag}; nothing to bump")


def rewrite(texts: dict, tag: str, digests: dict) -> dict:
    """{path: new text} for all five sites, checked by reading them back."""
    versions = {s.path: read_site(s, texts[s.path])[0] for s in SITES}
    check_forward(versions, tag)
    new = {s.path: rewrite_site(s, texts[s.path], tag, digests) for s in SITES}
    verify(new, tag, digests)
    return new


def verify(texts: dict, tag: str, digests: dict) -> None:
    """Raise Refused unless every site in *texts* pins *tag* with its digest."""
    for s in SITES:
        version, sha = read_site(s, texts[s.path])
        if version != tag or sha != digests[s.asset]:
            raise Refused(f"{s.path} reads back as {version} / {sha[:12]}..., "
                          f"expected {tag} / {digests[s.asset][:12]}...")


# --------------------------------------------------------------------------- #
#  Upstream                                                                    #
# --------------------------------------------------------------------------- #

class ListingUnreadable(Refused):
    """The release listing could not be read, or is not the release asked for."""


def fetch_release(tag: str, opener=None) -> dict:
    """The GitHub release listing of *tag*, as the API returns it.

    Raises ListingUnreadable when the request fails, the body is not JSON, or
    the listing is for another tag or carries no asset list."""
    if opener is None:
        from localm.http_ssl import verified_urlopen
        opener = verified_urlopen
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-bump-uv-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(RELEASE_URL % (UPSTREAM_REPO, tag), headers=headers)
    try:
        with opener(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ListingUnreadable(f"could not read the {tag} release from the GitHub API: "
                                f"{type(e).__name__}: {e}") from e
    if not isinstance(body, dict) or body.get("tag_name") != tag:
        got = body.get("tag_name") if isinstance(body, dict) else None
        raise ListingUnreadable(f"the release listing is for {got!r}, not {tag}")
    if not isinstance(body.get("assets"), list):
        raise ListingUnreadable(f"the {tag} release listing carries no asset list")
    return body


def asset_digest(body: dict, name: str) -> str:
    """The sha256 the listing *body* publishes for the asset called *name*.

    Raises Refused when the asset is absent or has no well-formed digest."""
    tag = body.get("tag_name")
    asset = next((a for a in body["assets"]
                  if isinstance(a, dict) and a.get("name") == name), None)
    if asset is None:
        raise Refused(f"the {tag} release has no asset named {name}")
    digest = asset.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise Refused(f"asset {name} of {tag} has no sha256 digest in the API listing")
    sha = digest.split("sha256:", 1)[1].strip()
    if not _SHA_RE.match(sha):
        raise Refused(f"asset {name} of {tag} has a malformed digest {digest!r}")
    return sha


def fetch_digests(tag: str, opener=None) -> dict:
    """{asset name: sha256} for the three assets the five pins carry."""
    body = fetch_release(tag, opener)
    return {name: asset_digest(body, name) for name in ASSETS}


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str) -> dict:
    """The {asset name: sha256} digests recorded in the receipt at *path*.

    Raises Refused unless the receipt confirms exactly *tag* as a candidate:
    schema 1, component uv, current false, verdict PASS, and every name in
    REQUIRED_CHECKS present, required and PASS."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}") from e
    if not isinstance(receipt, dict):
        raise Refused("the receipt is not a JSON object")
    if receipt.get("schema") != 1 or receipt.get("component") != "uv":
        raise Refused("the receipt is not a schema 1 receipt for the uv component")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for {receipt.get('tag')!r}, not {tag}; "
                      "confirm the target tag itself")
    if receipt.get("current") is not False:
        raise Refused("the receipt confirms the pinned build (--current), not a candidate")
    if receipt.get("verdict") != "PASS":
        raise Refused(f"the receipt verdict is {receipt.get('verdict')!r}, not PASS: "
                      f"{receipt.get('why', '')}")
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        raise Refused("the receipt carries no checks")
    problems = []
    for name in REQUIRED_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict):
            problems.append(f"{name}: absent")
        elif check.get("required") is not True:
            problems.append(f"{name}: not marked required")
        elif check.get("status") != "PASS":
            problems.append(f"{name}: {check.get('status')}")
    for name, check in checks.items():
        if isinstance(check, dict) and check.get("required") is True \
                and check.get("status") != "PASS" and name not in REQUIRED_CHECKS:
            problems.append(f"{name}: {check.get('status')}")
    if problems:
        raise Refused("the receipt does not confirm every required check: "
                      + "; ".join(problems))
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or any(
            not isinstance(assets.get(a), str) or not _SHA_RE.match(assets[a])
            for a in ASSETS):
        raise Refused("the receipt does not record a sha256 for each of "
                      + ", ".join(ASSETS))
    return {a: assets[a] for a in ASSETS}


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def checklist(tag: str) -> str:
    return "\n".join([
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_uv_installer_pin.py tests/test_docker_image_files.py "
        "tests/test_bump_uv_pin.py",
        "  2. python scripts/check_hygiene.py",
        f"  3. python scripts/confirm_uv_runtime.py --current --workdir <scratch> "
        f"--receipt <file>   (must PASS and report all five pins at {tag})",
        "  4. CHANGELOG.md, [Unreleased]: one bullet if the uv that setup installs moved",
    ])


def main(argv=None, opener=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the uv release to pin, e.g. 0.14.0")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_uv_runtime.py for --tag; "
                         "required with --write")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    ap.add_argument("--repo-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    root = Path(args.repo_root) if args.repo_root else REPO
    tag = args.tag.strip()
    try:
        parse_semver(tag)
        texts, newlines = {}, {}
        for s in SITES:
            try:
                texts[s.path], newlines[s.path] = _read(root / s.path)
            except (OSError, UnicodeDecodeError) as e:
                raise Refused(f"could not read {s.path}: {e}") from e

        recorded = None
        if args.receipt:
            recorded = load_receipt(Path(args.receipt), tag)
            print(f"receipt: {tag} confirmed (all required checks PASS)")
        elif args.write:
            raise Refused("--write needs --receipt: a bump without the confirm is "
                          "the untested-build problem the pin exists to remove")
        else:
            print("no receipt given: dry run only, nothing is confirmed")

        print(f"reading the {tag} release listing ...")
        digests = fetch_digests(tag, opener)
        for name in ASSETS:
            print(f"  {name}  sha256 {digests[name]}")
        if recorded is not None and recorded != digests:
            moved = [a for a in ASSETS if recorded[a] != digests[a]]
            raise Refused("the asset digests in the receipt differ from the ones the "
                          f"API publishes now for: {', '.join(moved)}")

        new_texts = rewrite(texts, tag, digests)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    pending = [s for s in SITES if new_texts[s.path] != texts[s.path]]
    for s in pending:
        if not args.write:
            sys.stdout.writelines(difflib.unified_diff(
                texts[s.path].splitlines(keepends=True),
                new_texts[s.path].splitlines(keepends=True),
                fromfile=f"a/{s.path}", tofile=f"b/{s.path}"))
    if args.write:
        written = []
        try:
            for s in pending:
                _write(root / s.path, new_texts[s.path], newlines[s.path])
                written.append(s)
        except OSError as e:
            for s in written:
                _write(root / s.path, texts[s.path], newlines[s.path])
            print(f"REFUSED: could not write {s.path}: {e}; every file was restored")
            return 1
        for s in pending:
            print(f"wrote {s.path}")
    else:
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
