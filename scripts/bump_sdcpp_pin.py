#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing localm's pinned stable-diffusion.cpp build.

The pin is one procedure, not a constant edit: the release tag, the full commit the
ctypes binding is checked against, and the archive table (asset name and sha256 per
platform and backend) must move together. This script rewrites all of them in
localm/media/sdcpp/pins.py and refuses to write unless a receipt from
``scripts/confirm_sdcpp_runtime.py --receipt`` shows the target tag PASSED.

WHAT IT REWRITES (with ``--write``; without it a unified diff is printed and nothing
is touched), all in localm/media/sdcpp/pins.py:

  TAG            the target release tag, master-<N>-<short commit>
  COMMIT         the 40-hex commit the release was built from
  ASSETS         the archive for every (platform, backend) key the file already
                 lists, with the sha256 the GitHub release API publishes for it
  EXTRA_ASSETS   the companion archives, same source

Asset names embed the short commit and sometimes a toolkit version (rocm-7.14.0,
macOS-26.6.2), so each key is located by a name pattern, not by string replacement.
A key with no matching asset, or with two, refuses the bump.

WHAT IT CHECKS BEFORE WRITING:
  * the receipt is schema 1, component sdcpp, for exactly this tag, not a
    ``--current`` run, verdict PASS, and every check in it that is marked required
    is PASS, including the cpu checks listed in MANDATORY_CHECKS (a receipt that
    omits a check does not pass by omitting it);
  * the tag is strictly newer than the pinned one (forward-only);
  * the release listing carries a size and a sha256 digest for every asset, and
    the commit and the asset table the receipt recorded equal what the API says
    now (a release that was re-uploaded after the confirm is refused);
  * each edited region is found exactly once, and nothing else in the tree (outside
    CHANGELOG) still names the old tag, commit or short commit.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the targeted tests,
the changelog bullet, and the behaviours the confirm does not measure.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when refused.

Environment: GITHUB_TOKEN is sent as a bearer token for the API lookups.

Usage:
    python scripts/bump_sdcpp_pin.py --tag master-952-abcdef0                  # dry run
    python scripts/bump_sdcpp_pin.py --tag master-952-abcdef0 --receipt c.json
    python scripts/bump_sdcpp_pin.py --tag master-952-abcdef0 --receipt c.json --write

Needs localm importable (the verified HTTPS opener). Nothing under localm/ imports
this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import difflib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PINS_PATH = REPO / "localm" / "media" / "sdcpp" / "pins.py"

UPSTREAM_REPO = "leejet/stable-diffusion.cpp"
RELEASE_URL = "https://api.github.com/repos/%s/releases/tags/%s"
COMMIT_URL = "https://api.github.com/repos/%s/commits/%s"

SCHEMA = 1
COMPONENT = "sdcpp"

# Checks a PASS receipt must contain, each marked required and PASS. confirm names
# the per-backend ones <check>_<backend>; cpu runs on every machine.
MANDATORY_CHECKS = ("isolation", "release_assets", "header_layout", "download_cpu",
                    "abi_cpu", "device_cpu", "generate_cpu")

_TAG_RE = re.compile(r"^master-(\d+)-([0-9a-f]{7,12})$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TAG_LINE_RE = re.compile(r'^(TAG\s*=\s*")([^"\n]+)(")[ \t]*$', re.M)
_COMMIT_LINE_RE = re.compile(r'^(COMMIT\s*=\s*")([^"\n]+)(")[ \t]*$', re.M)

# (platform, backend) -> pattern of the archive name with {s} standing for the short
# commit. Fullmatch, so "x86_64.zip" and "x86_64-vulkan.zip" stay distinct.
_VER = r"\d+(?:\.\d+)*"
ASSET_PATTERNS = {
    ("windows", "cpu"): r"sd-master-{s}-bin-win-cpu-x64\.zip",
    ("windows", "vulkan"): r"sd-master-{s}-bin-win-vulkan-x64\.zip",
    ("windows", "cuda"): r"sd-master-{s}-bin-win-cuda\d+-x64\.zip",
    ("windows", "rocm"): rf"sd-master-{{s}}-bin-win-rocm-{_VER}-x64\.zip",
    ("linux", "cpu"): rf"sd-master-{{s}}-bin-Linux-Ubuntu-{_VER}-x86_64\.zip",
    ("linux", "vulkan"): rf"sd-master-{{s}}-bin-Linux-Ubuntu-{_VER}-x86_64-vulkan\.zip",
    ("linux", "rocm"): rf"sd-master-{{s}}-bin-Linux-Ubuntu-{_VER}-x86_64-rocm-{_VER}\.zip",
    ("macos-arm64", "metal"): rf"sd-master-{{s}}-bin-Darwin-macOS-{_VER}-arm64\.zip",
}
EXTRA_PATTERNS = {
    ("windows", "cuda"): [r"cudart-sd-bin-win-cu\d+-x64\.zip"],
}

# Files that may legitimately name an old release.
_STALE_SCAN_SKIP = ("CHANGELOG.md", "CHANGELOG-FULL.md")
_STALE_SCAN_DIRS = ("localm", "tests", "tests-js", "docs", "scripts", "docker", "installer")
_STALE_SCAN_TOP = ("README.md", "SECURITY.md", "THIRD-PARTY-NOTICES.md", "pyproject.toml",
                   "setup.sh", "setup.bat", "install.sh")
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".js", ".mjs",
                  ".html", ".css", ".sh", ".bat", ".ps1", ".cfg", ".ini", ".rst", ""}


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


class UpstreamUnreadable(Refused):
    """The GitHub API could not be read or answered with something unusable."""


class IncompleteRelease(Refused):
    """The release lacks an archive localm installs."""


# --------------------------------------------------------------------------- #
#  Tags                                                                        #
# --------------------------------------------------------------------------- #

def parse_tag(tag: str) -> tuple[int, str]:
    """(build number, short commit) of a ``master-<N>-<short>`` tag."""
    m = _TAG_RE.match(tag or "")
    if not m:
        raise Refused(f"{tag!r} is not a stable-diffusion.cpp release tag "
                      "(master-<number>-<short commit>)")
    return int(m.group(1)), m.group(2)


def forward_only(current_tag: str, target_tag: str) -> None:
    """Raises Refused unless *target_tag* has a strictly higher build number than
    *current_tag*."""
    cur_n, _ = parse_tag(current_tag)
    new_n, _ = parse_tag(target_tag)
    if new_n <= cur_n:
        raise Refused(f"{target_tag} is not newer than the pinned {current_tag}; this "
                      "script only ever advances the pin forward")


# --------------------------------------------------------------------------- #
#  Upstream                                                                    #
# --------------------------------------------------------------------------- #

def _api_get(url: str, opener=None):
    if opener is None:
        from localm.http_ssl import verified_urlopen
        opener = verified_urlopen
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-bump-sdcpp-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise UpstreamUnreadable(f"could not read {url} from the GitHub API: "
                                 f"{type(e).__name__}: {e}") from e


def fetch_release(tag: str, opener=None) -> dict:
    """``{"tag", "commit", "assets": {name: {"size", "sha256"}}}`` for the release
    tagged *tag* of stable-diffusion.cpp.

    The commit comes from the commit lookup of the tag and must start with the short
    commit the tag names. Raises Refused when either lookup fails, an asset has no
    size or no sha256 digest, or the release lists no assets."""
    _, short = parse_tag(tag)
    body = _api_get(RELEASE_URL % (UPSTREAM_REPO, tag), opener)
    if not isinstance(body, dict) or body.get("tag_name") != tag:
        raise Refused(f"the release listing is not for {tag}")
    listing = body.get("assets")
    if not isinstance(listing, list) or not listing:
        raise Refused(f"the {tag} release listing carries no asset list")
    assets = {}
    for a in listing:
        name = a.get("name") if isinstance(a, dict) else None
        digest = a.get("digest") if isinstance(a, dict) else None
        size = a.get("size") if isinstance(a, dict) else None
        if not isinstance(name, str) or not isinstance(digest, str) \
                or not digest.startswith("sha256:"):
            raise IncompleteRelease(f"asset {name!r} of {tag} has no sha256 digest in the API "
                                    "listing; the table cannot be filled from it")
        sha = digest.split("sha256:", 1)[1].strip().lower()
        if not _SHA_RE.match(sha):
            raise IncompleteRelease(f"asset {name!r} of {tag} has a malformed digest "
                                    f"{digest!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise IncompleteRelease(f"asset {name!r} of {tag} has no usable size in the API "
                                    "listing")
        if name in assets:
            raise Refused(f"asset {name!r} is listed twice in {tag}")
        assets[name] = {"size": size, "sha256": sha}
    commit_body = _api_get(COMMIT_URL % (UPSTREAM_REPO, tag), opener)
    commit = commit_body.get("sha") if isinstance(commit_body, dict) else None
    if not isinstance(commit, str) or not _COMMIT_RE.match(commit):
        raise Refused(f"the commit lookup for {tag} did not return a 40-hex sha")
    if not commit.startswith(short):
        raise Refused(f"{tag} resolves to commit {commit}, which does not start with "
                      f"the short commit {short} the tag names")
    target = body.get("target_commitish")
    if isinstance(target, str) and _COMMIT_RE.match(target) and target != commit:
        raise Refused(f"the release targets {target} but the tag resolves to {commit}")
    return {"tag": tag, "commit": commit, "assets": assets}


def classify_assets(short: str, names, asset_keys, extra_keys) -> dict:
    """Locate each wanted archive among *names*.

    Returns ``{"assets": {key: name}, "extra": {key: [name, ...]}, "missing":
    [description, ...], "ambiguous": [description, ...]}`` for the keys in
    *asset_keys* and *extra_keys*. Raises Refused for a key with no pattern."""
    names = list(names)
    out = {"assets": {}, "extra": {}, "missing": [], "ambiguous": []}
    for key in asset_keys:
        pat = ASSET_PATTERNS.get(tuple(key))
        if pat is None:
            raise Refused(f"no name pattern for {tuple(key)}; add it to ASSET_PATTERNS "
                          "before bumping")
        rx = re.compile(pat.format(s=re.escape(short)))
        hits = [n for n in names if rx.fullmatch(n)]
        if not hits:
            out["missing"].append(f"{key[0]}/{key[1]}")
        elif len(hits) > 1:
            out["ambiguous"].append(f"{key[0]}/{key[1]}: {', '.join(sorted(hits))}")
        else:
            out["assets"][tuple(key)] = hits[0]
    for key in extra_keys:
        pats = EXTRA_PATTERNS.get(tuple(key))
        if pats is None:
            raise Refused(f"no name pattern for the extra archives of {tuple(key)}; add "
                          "it to EXTRA_PATTERNS before bumping")
        found = []
        for pat in pats:
            rx = re.compile(pat)
            hits = [n for n in names if rx.fullmatch(n)]
            if not hits:
                out["missing"].append(f"{key[0]}/{key[1]} extra {pat}")
            elif len(hits) > 1:
                out["ambiguous"].append(f"{key[0]}/{key[1]} extra: {', '.join(sorted(hits))}")
            else:
                found.append(hits[0])
        if len(found) == len(pats):
            out["extra"][tuple(key)] = found
    return out


def build_tables(release: dict, current_assets: dict, current_extra: dict) -> tuple[dict, dict]:
    """(ASSETS, EXTRA_ASSETS) for *release*, with exactly the keys, in the order, of
    the current tables. Raises Refused when an archive is missing or ambiguous."""
    _, short = parse_tag(release["tag"])
    found = classify_assets(short, release["assets"], list(current_assets),
                            list(current_extra))
    if found["ambiguous"]:
        raise Refused("more than one archive matches: " + "; ".join(found["ambiguous"])
                      + ". The release naming changed; tighten ASSET_PATTERNS.")
    if found["missing"]:
        raise IncompleteRelease(
            f"{release['tag']} has no archive for: " + ", ".join(found["missing"])
            + ". Not bumping to a release that drops something localm installs.")
    assets = {k: (found["assets"][k], release["assets"][found["assets"][k]]["sha256"])
              for k in current_assets}
    extra = {k: [(n, release["assets"][n]["sha256"]) for n in found["extra"][k]]
             for k in current_extra}
    return assets, extra


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str, extra_required: tuple = ()) -> dict:
    """The receipt dict, validated for a bump to *tag*.

    Raises Refused when it is unreadable, is not schema 1 / component sdcpp, names
    another tag, is a ``--current`` run, has a verdict other than PASS, lacks a
    mandatory check, or has any required check that is not PASS."""
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}") from e
    if not isinstance(receipt, dict):
        raise Refused("the receipt is not a JSON object")
    if receipt.get("schema") != SCHEMA or receipt.get("component") != COMPONENT:
        raise Refused(f"the receipt is not a schema {SCHEMA} {COMPONENT} receipt")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for {receipt.get('tag')!r}, not {tag}; confirm the "
                      "target tag itself")
    if receipt.get("current") is not False:
        raise Refused("the receipt is a --current run; it confirms the pin that is "
                      "already installed, not a candidate")
    try:
        _dt.datetime.strptime(str(receipt.get("written_at")), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as e:
        raise Refused("the receipt has no valid written_at timestamp") from e
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        raise Refused("the receipt has no checks")
    names = list(MANDATORY_CHECKS) + [c for c in extra_required if c not in MANDATORY_CHECKS]
    problems = []
    for name in names:
        c = checks.get(name)
        if not isinstance(c, dict):
            problems.append(f"{name}: not in the receipt")
        elif c.get("status") != "PASS" or c.get("required") is not True:
            problems.append(f"{name}: {c.get('status')} required={c.get('required')} "
                            f"({c.get('detail', '')})")
    for name, c in checks.items():
        if isinstance(c, dict) and c.get("required") is True and c.get("status") != "PASS" \
                and not any(p.startswith(name + ":") for p in problems):
            problems.append(f"{name}: {c.get('status')} ({c.get('detail', '')})")
    if problems:
        raise Refused(f"the receipt does not confirm {tag}: " + "; ".join(problems))
    if receipt.get("verdict") != "PASS":
        raise Refused(f"the receipt verdict is {receipt.get('verdict')!r}, not PASS, "
                      "although its checks pass; the receipt is inconsistent")
    hw = receipt.get("hardware") or {}
    ran = set(hw.get("backends") or [])
    if "cpu" not in ran:
        raise Refused("the receipt does not list cpu among the backends it ran")
    if hw.get("gpu") and ran <= {"cpu"}:
        raise Refused("this machine has a GPU but the receipt ran only cpu; the GPU "
                      "backend was not measured")
    cand = receipt.get("candidate")
    if not isinstance(cand, dict) or not _COMMIT_RE.match(str(cand.get("commit"))) \
            or not isinstance(cand.get("assets"), dict):
        raise Refused("the receipt carries no candidate commit and asset table")
    return receipt


def compare_with_receipt(receipt: dict, release: dict) -> None:
    """Raises Refused unless the commit and archives the receipt recorded equal what
    the release API reports now."""
    cand = receipt["candidate"]
    if cand["commit"] != release["commit"]:
        raise Refused(f"the receipt confirmed commit {cand['commit']} but {release['tag']} "
                      f"now resolves to {release['commit']}")
    diffs = []
    for name, rec in cand["assets"].items():
        now = release["assets"].get(name)
        if now is None:
            diffs.append(f"{name}: gone from the release")
        elif now != {"size": rec.get("size"), "sha256": rec.get("sha256")}:
            diffs.append(f"{name}: size or sha256 differs from what was confirmed")
    if diffs:
        raise Refused("the release changed after it was confirmed: " + "; ".join(diffs))


# --------------------------------------------------------------------------- #
#  The pins file                                                               #
# --------------------------------------------------------------------------- #

def _assign(tree: ast.Module, name: str):
    nodes = [n for n in tree.body
             if (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == name)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name
                                                   for t in n.targets))]
    if len(nodes) != 1:
        raise Refused(f"{name}: expected exactly one assignment, found {len(nodes)}; the "
                      "file shape this script edits has changed")
    return nodes[0]


def read_pins(text: str) -> dict:
    """``{"tag", "commit", "assets", "extra"}`` as pins.py *text* defines them."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise Refused(f"pins.py does not parse: {e}") from e
    out = {}
    for field, name in (("tag", "TAG"), ("commit", "COMMIT"), ("assets", "ASSETS"),
                        ("extra", "EXTRA_ASSETS")):
        try:
            out[field] = ast.literal_eval(_assign(tree, name).value)
        except (ValueError, SyntaxError) as e:
            raise Refused(f"{name} is not a literal: {e}") from e
    if not isinstance(out["assets"], dict) or not isinstance(out["extra"], dict):
        raise Refused("ASSETS or EXTRA_ASSETS is not a dict")
    return out


def render_assets(assets: dict) -> str:
    lines = ["ASSETS: dict[tuple[str, str], tuple[str, str]] = {"]
    for (plat, backend), (name, sha) in assets.items():
        lines += [f'    ("{plat}", "{backend}"): (', f'        "{name}",', f'        "{sha}"),']
    lines.append("}")
    return "\n".join(lines)


def render_extra(extra: dict) -> str:
    lines = ["EXTRA_ASSETS: dict[tuple[str, str], list[tuple[str, str]]] = {"]
    for (plat, backend), entries in extra.items():
        for i, (name, sha) in enumerate(entries):
            head = f'    ("{plat}", "{backend}"): [(' if i == 0 else "    ("
            tail = ")]," if i == len(entries) - 1 else "),"
            lines += [f"{head}", f'        "{name}",', f'        "{sha}"{tail}']
    lines.append("}")
    return "\n".join(lines)


def _replace_lines(text: str, node, new_block: str) -> str:
    lines = text.split("\n")
    lines[node.lineno - 1:node.end_lineno] = new_block.split("\n")
    return "\n".join(lines)


def _replace_once(pattern: re.Pattern, text: str, value: str, what: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; the "
                      "file shape this script edits has changed")
    m = matches[0]
    return text[:m.start()] + m.group(1) + value + m.group(3) + text[m.end():]


def rewrite(text: str, tag: str, commit: str, assets: dict, extra: dict) -> str:
    """pins.py *text* with TAG, COMMIT, ASSETS and EXTRA_ASSETS replaced."""
    parse_tag(tag)
    if not _COMMIT_RE.match(commit):
        raise Refused(f"{commit!r} is not a 40-character hex commit")
    text = _replace_once(_TAG_LINE_RE, text, tag, "TAG")
    text = _replace_once(_COMMIT_LINE_RE, text, commit, "COMMIT")
    # EXTRA_ASSETS sits after ASSETS: replace the later block first so the earlier
    # block's line numbers stay valid.
    for name, block in (("EXTRA_ASSETS", render_extra(extra)), ("ASSETS", render_assets(assets))):
        node = _assign(ast.parse(text), name)
        text = _replace_lines(text, node, block)
    return text


# --------------------------------------------------------------------------- #
#  Nothing else may still name the old release                                 #
# --------------------------------------------------------------------------- #

def _candidate_files(root: Path):
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
                           timeout=60)
        if r.returncode == 0 and r.stdout:
            for rel in r.stdout.decode("utf-8", "replace").split("\0"):
                if rel:
                    yield root / rel
            return
    except (OSError, subprocess.SubprocessError):
        pass
    for d in _STALE_SCAN_DIRS:
        base = root / d
        if base.is_dir():
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [x for x in dirnames if x not in {"node_modules", "__pycache__",
                                                                  ".git", ".venv"}]
                for f in filenames:
                    yield Path(dirpath) / f
    for f in _STALE_SCAN_TOP:
        if (root / f).is_file():
            yield root / f


def find_stale_references(root: Path, old_tag: str, old_commit: str,
                          skip: tuple = ()) -> list[str]:
    """``path:line`` of every text file under *root* that names the old tag, the old
    commit or the old short commit, other than pins.py, *skip* and the changelogs."""
    short = old_commit[:7]
    rx = re.compile("|".join([re.escape(old_tag), re.escape(old_commit),
                              rf"(?<![0-9a-fA-F]){re.escape(short)}(?![0-9a-fA-F])"]),
                    re.I)
    pins_file = (root / "localm" / "media" / "sdcpp" / "pins.py").resolve()
    skip_paths = {(root / s).resolve() for s in skip}
    hits = []
    for path in _candidate_files(root):
        try:
            if path.suffix.lower() not in _TEXT_SUFFIXES or path.name in _STALE_SCAN_SKIP:
                continue
            resolved = path.resolve()
            if resolved == pins_file or resolved in skip_paths or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{path.relative_to(root).as_posix()}:{n}")
    return sorted(hits)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> tuple[str, str]:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    """Write *text* (LF newlines) using the file's own newline sequence."""
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def checklist(tag: str) -> str:
    return "\n".join([
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_sdcpp_binding.py tests/test_sdcpp_runtime.py "
        "tests/test_sdcpp_runner.py",
        "       tests/test_image_native_backend.py tests/test_video_native_backend.py",
        "       tests/test_bump_sdcpp_pin.py tests/test_confirm_sdcpp_runtime.py",
        f"  2. CHANGELOG.md, [Unreleased]: one bullet, the native image runtime moved to {tag}",
        "  3. Not measured by the confirm: video generation, and the cuda, rocm and metal",
        "       builds on hardware this machine lacks; the receipt's checks list what ran.",
    ])


def main(argv=None, opener=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the release to pin, e.g. master-952-abcdef0")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_sdcpp_runtime.py --receipt for "
                         "--tag; required with --write")
    ap.add_argument("--require", action="append", default=None,
                    help="a further check the receipt must report required and PASS "
                         "(repeatable)")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    tag = args.tag.strip()
    try:
        parse_tag(tag)
        text, newline = _read(PINS_PATH)
        cur = read_pins(text)
        forward_only(cur["tag"], tag)

        receipt = None
        if args.receipt:
            receipt = load_receipt(Path(args.receipt), tag, tuple(args.require or ()))
            print(f"receipt: {tag} PASS; backends run: "
                  f"{', '.join(receipt['hardware']['backends'])}")
        elif args.write:
            raise Refused("--write needs --receipt: a bump without the confirm is the "
                          "untested-build problem the pin exists to remove")
        else:
            print("no receipt given: dry run only, nothing is confirmed")

        print(f"reading the {tag} release listing ...")
        release = fetch_release(tag, opener)
        print(f"  {len(release['assets'])} assets with sha256 digests; commit {release['commit']}")
        if receipt is not None:
            compare_with_receipt(receipt, release)
        assets, extra = build_tables(release, cur["assets"], cur["extra"])
        new_text = rewrite(text, tag, release["commit"], assets, extra)
        stale = find_stale_references(REPO, cur["tag"], cur["commit"])
        if stale:
            msg = ("these files still name the old release and would be left behind: "
                   + ", ".join(stale))
            if args.write:
                raise Refused(msg)
            print(f"WARNING: {msg}")
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    rel = PINS_PATH.relative_to(REPO).as_posix()
    if new_text == text:
        print(f"nothing to change: the tree already pins {tag}")
    elif args.write:
        _write(PINS_PATH, new_text, newline)
        print(f"wrote {rel}")
    else:
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
