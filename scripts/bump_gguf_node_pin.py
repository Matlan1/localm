#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advance the pinned ComfyUI-GGUF custom node commit in localm's managed ComfyUI.

``localm/media/managed_comfy_fresh.py`` clones ComfyUI-GGUF into a fresh managed
ComfyUI at the commit held in ``_GGUF_NODE``. This script moves that commit.
It is review-only: a person reads the diff, the upstream change list and the
printed checklist, then merges.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched):

  localm/media/managed_comfy_fresh.py
    _GGUF_NODE             the commit of the CustomNodePin only; its name and
                           repository URL are never touched

WHAT IT CHECKS BEFORE WRITING:
  * ``--tag`` is a full 40-character commit sha and differs from the pinned one;
  * the pinned repository is a github.com repository and the commit exists in
    it;
  * the commit is a strict descendant of the pinned commit (the compare API
    reports it ahead with nothing behind and the pinned commit as merge base);
  * the commit is reachable from the repository's default branch, so a commit
    that only exists in a fork or an unmerged pull request is refused;
  * the edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the upstream
commit list and changed files, the targeted tests, whether the node still works
with the pinned ComfyUI, the comment that names the old short sha, and a real
GGUF image generation.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused.

Environment: GITHUB_TOKEN is sent as a bearer token to the GitHub API.

Usage:
    python scripts/bump_gguf_node_pin.py --tag <40-hex commit sha>
    python scripts/bump_gguf_node_pin.py --tag <40-hex commit sha> --write
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN_REL = "localm/media/managed_comfy_fresh.py"
API = "https://api.github.com/repos/%s"
MAX_JSON_BYTES = 16 * 1024 * 1024
SCAN_DIRS = ("localm", "tests", "docs", "scripts")
SCAN_SUFFIXES = (".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".mjs", ".js")
MAX_SCAN_BYTES = 2 * 1024 * 1024
LIST_LIMIT = 25

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PIN_RE = re.compile(
    r'(?P<head>^_GGUF_NODE = CustomNodePin\((?P<args>[^)]*?)commit=")'
    r'(?P<sha>[0-9a-f]{40})(?P<tail>")', re.M | re.S)
_REPO_RE = re.compile(r'repo="https://github\.com/(?P<slug>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?"')
_COMFY_VERSION_RE = re.compile(r'^COMFYUI_PINNED_VERSION\s*=\s*"([^"]+)"', re.M)


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


# --------------------------------------------------------------------------- #
#  Upstream (injectable)                                                      #
# --------------------------------------------------------------------------- #

def _default_open(req, timeout):
    from localm.http_ssl import verified_urlopen
    return verified_urlopen(req, timeout=timeout)


def fetch_json(url: str, opener=None) -> object:
    """GET *url* from the GitHub API and decode it as JSON.

    Raises Refused when it cannot be read; a 404 says the object does not
    exist."""
    opener = opener or _default_open
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-bump-gguf-node-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener(req, 30) as resp:
            raw = resp.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("response is larger than the allowed size")
        return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Refused(f"GitHub has no such object at {url} (HTTP 404)") from e
        raise Refused(f"could not read {url}: HTTP {e.code}") from e
    except Exception as e:
        raise Refused(f"could not read {url}: {type(e).__name__}: {e}") from e


def verify_descendant(slug: str, old: str, new: str, fetch_json_fn=fetch_json) -> dict:
    """The compare document for ``old...new``, verified.

    Raises Refused unless *new* exists in *slug*, is ahead of *old* with nothing
    behind and *old* as merge base, and is reachable from the default branch."""
    meta = fetch_json_fn(API % slug)
    branch = meta.get("default_branch") if isinstance(meta, dict) else None
    if not isinstance(branch, str) or not branch:
        raise Refused(f"{slug} reports no default branch")
    cmp = fetch_json_fn(API % f"{slug}/compare/{old}...{new}")
    if not isinstance(cmp, dict):
        raise Refused(f"the compare answer for {old[:7]}...{new[:7]} is not an object")
    status, ahead, behind = cmp.get("status"), cmp.get("ahead_by"), cmp.get("behind_by")
    if status == "identical":
        raise Refused(f"{new} is the commit already pinned")
    if status != "ahead" or behind != 0 or not isinstance(ahead, int) or ahead < 1:
        raise Refused(f"{new} is not a descendant of the pinned {old}: compare says "
                      f"status {status!r}, ahead_by {ahead!r}, behind_by {behind!r}")
    base = (cmp.get("merge_base_commit") or {}).get("sha")
    if base != old:
        raise Refused(f"the merge base of the compare is {base!r}, not the pinned {old}")
    commits = cmp.get("commits")
    if not isinstance(commits, list) or (len(commits) == ahead
                                         and (commits[-1] or {}).get("sha") != new):
        raise Refused(f"the compare lists a different tip than {new}")
    onto = fetch_json_fn(API % f"{slug}/compare/{new}...{branch}")
    if not isinstance(onto, dict) or onto.get("status") not in ("ahead", "identical") \
            or onto.get("behind_by") != 0:
        raise Refused(f"{new} is not reachable from {slug}'s default branch {branch!r} "
                      f"(compare says {onto.get('status') if isinstance(onto, dict) else onto!r}); "
                      "a commit that only exists in a fork or an unmerged pull request is refused")
    cmp["_default_branch"] = branch
    return cmp


# --------------------------------------------------------------------------- #
#  The tree                                                                   #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> tuple:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def read_pin(text: str) -> tuple:
    """(repository slug, pinned commit) of ``_GGUF_NODE``. Raises Refused when
    the pin is not found exactly once or its repository is not on github.com."""
    matches = list(_PIN_RE.finditer(text))
    if len(matches) != 1:
        raise Refused(f"_GGUF_NODE: expected exactly one CustomNodePin commit, found "
                      f"{len(matches)}; the file shape this script edits has changed")
    m = matches[0]
    repo = _REPO_RE.search(m.group("args"))
    if not repo:
        raise Refused("_GGUF_NODE does not name a github.com repository")
    return repo.group("slug"), m.group("sha")


def rewrite(text: str, commit: str) -> str:
    """*text* with the ``_GGUF_NODE`` commit moved to *commit*."""
    read_pin(text)
    m = _PIN_RE.search(text)
    return text[:m.start("sha")] + commit + text[m.end("sha"):]


def find_mentions(root: Path, old: str, skip: Path) -> list:
    """``path:line: text`` for every line under the scanned directories that
    names the old commit, full or as a 7-character prefix, outside *skip*."""
    pattern = re.compile(re.escape(old[:7]) + r"[0-9a-f]*")
    out = []
    for name in SCAN_DIRS:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if (not path.is_file() or path.suffix not in SCAN_SUFFIXES or path == skip
                    or any(part in ("node_modules", "__pycache__", ".git") for part in path.parts)
                    or path.stat().st_size > MAX_SCAN_BYTES):
                continue
            try:
                lines = path.read_text(encoding="utf-8").split("\n")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(lines, 1):
                m = pattern.search(line)
                if m and (len(m.group(0)) == 7 or old.startswith(m.group(0))):
                    out.append(f"{path.relative_to(root).as_posix()}:{number}: {line.strip()[:100]}")
    return out


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #

def summarise(cmp: dict, old: str, new: str, slug: str) -> str:
    commits = cmp.get("commits") or []
    files = cmp.get("files") or []
    lines = [f"upstream change {old[:7]}...{new[:7]} in {slug}: {cmp.get('ahead_by')} commit(s), "
             f"{len(files)} file(s) changed"]
    for c in commits[-LIST_LIMIT:]:
        subject = ((c.get("commit") or {}).get("message") or "").split("\n")[0][:90]
        lines.append(f"  {str(c.get('sha', ''))[:7]} {subject}")
    if len(commits) > LIST_LIMIT:
        lines.append(f"  (showing the newest {LIST_LIMIT} of {len(commits)})")
    for f in files[:60]:
        lines.append(f"  {f.get('status', '?'):9} {f.get('filename', '?')}")
    return "\n".join(lines)


def checklist(slug: str, old: str, new: str, comfy_version: str, mentions: list) -> str:
    lines = [
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_bump_gguf_node_pin.py tests/test_managed_comfy_s1.py",
        "       tests/test_managed_comfy_s3.py tests/test_managed_comfy_s4.py tests/test_managed_comfy_s5_gui.py",
        f"  2. read the upstream change list above; compare view: https://github.com/{slug}/compare/{old}...{new}",
        f"       the node runs inside the pinned ComfyUI ({comfy_version or 'version not found'}): "
        "check the node's requirements still hold",
        "  3. a real GGUF image generation on a managed ComfyUI provisioned at this commit; this",
        "       script does not clone or run the node",
        "  4. CHANGELOG.md, [Unreleased]: one bullet if the loader behaviour changes for users",
    ]
    if mentions:
        lines.append(f"  5. text outside the pin that still names {old[:7]} (edit only if now false):")
        lines += [f"       {m}" for m in mentions]
    return "\n".join(lines)


def main(argv=None, *, fetch_json_fn=fetch_json) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help="the 40-character commit sha of ComfyUI-GGUF to pin")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    pin_path = REPO / PIN_REL
    new = args.tag.strip().lower()
    try:
        if not _COMMIT_RE.match(new):
            raise Refused(f"{args.tag!r} is not a full 40-character hex commit sha")
        text, newline = _read(pin_path)
        slug, old = read_pin(text)
        if new == old:
            raise Refused(f"{new} is the commit already pinned")
        print(f"checking {new[:7]} against {slug} (pinned {old[:7]}) ...")
        cmp = verify_descendant(slug, old, new, fetch_json_fn)
        print(f"  descendant of the pin, reachable from {cmp['_default_branch']}")
        new_text = rewrite(text, new)
        mentions = find_mentions(REPO, old, pin_path)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    print(summarise(cmp, old, new, slug))
    print()
    if args.write:
        _write(pin_path, new_text, newline)
        print(f"wrote {PIN_REL}")
    else:
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{PIN_REL}", tofile=f"b/{PIN_REL}"))
        print("\n(dry run: nothing written; add --write to apply)")
    comfy = _COMFY_VERSION_RE.search(text)
    print()
    print(checklist(slug, old, new, comfy.group(1) if comfy else "", mentions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
