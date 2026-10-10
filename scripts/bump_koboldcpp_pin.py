#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing localm's pinned KoboldCpp release.

Rewrites localm/media/koboldcpp/pins.py:

  TAG       the target release tag
  VERSION   the version the launcher printed during the confirm run
  ASSETS    size and sha256 of every pinned asset, taken from the digest field of
            the GitHub release listing (asset names are kept)

``_BASE_URL`` is derived from TAG in that file and needs no edit; the script
checks that it still is. Nothing else in the tree names the pinned tag or an
asset hash; the script refuses to write if it finds such a mention in another
tracked file.

Without --write it prints a unified diff and changes nothing. It refuses
(exit 1) unless:

  * the receipt, written by scripts/confirm_koboldcpp_runtime.py for exactly
    this tag with --tag (not --current), has verdict PASS, every required
    check present and PASS, and no failed check;
  * the receipt's version equals the tag without its leading v;
  * the target tag is strictly newer than the pinned TAG;
  * the release listing carries every pinned asset with a sha256 digest and
    its sizes and digests equal the table the confirm run installed from;
  * each edited region is found exactly once.

Exit codes: 0 when the edit was applied or the dry run completed; 1 refused.

Environment: GITHUB_TOKEN, else the token `gh auth token` prints, is sent as a
bearer token to the API.

Usage:
    python scripts/bump_koboldcpp_pin.py --tag v1.123 --receipt r.json
    python scripts/bump_koboldcpp_pin.py --tag v1.123 --receipt r.json --write

Needs localm importable (the verified HTTPS opener). Nothing under localm/
imports this script.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PINS_PATH = REPO / "localm" / "media" / "koboldcpp" / "pins.py"
CONFIRM_PATH = REPO / "scripts" / "confirm_koboldcpp_runtime.py"

# Files allowed to name the pinned tag or an asset hash besides pins.py.
MENTION_EXEMPT = ("localm/media/koboldcpp/pins.py", "CHANGELOG.md")

_TAG_LINE_RE = re.compile(r'^(TAG\s*=\s*")([^"\n]+)(")', re.M)
_VERSION_LINE_RE = re.compile(r'^(VERSION\s*=\s*")([^"\n]+)(")', re.M)
_BASE_URL_RE = re.compile(
    r'^_BASE_URL = f"https://github\.com/\{REPO\}/releases/download/\{TAG\}/"$', re.M)
_ENTRY_RE = re.compile(
    r'\("(?P<plat>[^"\n]+)", "(?P<build>[^"\n]+)"\): \(\n'
    r'(?P<i1>[ \t]+)"(?P<name>[^"\n]+)", (?P<size>\d+),\n'
    r'(?P<i2>[ \t]+)"(?P<sha>[0-9a-f]{64})"\),')
_ENTRY_START_RE = re.compile(r'^[ \t]+\("[^"\n]+", "[^"\n]+"\): \(', re.M)


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


def _load_confirm():
    spec = importlib.util.spec_from_file_location("confirm_koboldcpp_runtime",
                                                  CONFIRM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("confirm_koboldcpp_runtime", mod)
    spec.loader.exec_module(mod)
    return mod


confirm = _load_confirm()


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str) -> dict:
    """The receipt at *path*, validated as a PASS for exactly *tag*.

    Raises Refused when it is unreadable, for another component or tag, a
    --current run, not PASS, missing a required check, or carries any failed or
    unmet required check."""
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}") from e
    if not isinstance(receipt, dict):
        raise Refused("the receipt is not a JSON object")
    if receipt.get("schema") != 1 or receipt.get("component") != confirm.COMPONENT:
        raise Refused("the receipt is not a schema 1 koboldcpp receipt")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for {receipt.get('tag')!r}, not {tag}; "
                      "confirm the target tag itself")
    if receipt.get("current") is not False:
        raise Refused("the receipt is from a --current run (it confirms the build "
                      "pinned today), not a candidate; confirm the target with --tag")
    if receipt.get("verdict") != "PASS":
        raise Refused(f"the receipt verdict is {receipt.get('verdict')!r}: "
                      f"{receipt.get('why', '')}")
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        raise Refused("the receipt has no checks")
    problems = []
    for name in confirm.ALWAYS_REQUIRED:
        c = checks.get(name)
        if not isinstance(c, dict):
            problems.append(f"{name}: missing")
        elif c.get("required") is not True:
            problems.append(f"{name}: not marked required")
        elif c.get("status") != "PASS":
            problems.append(f"{name}: {c.get('status')} ({c.get('detail', '')})")
    for name, c in checks.items():
        if not isinstance(c, dict):
            problems.append(f"{name}: malformed")
        elif c.get("status") == "FAIL" and name not in confirm.ADVISORY_CHECKS:
            problems.append(f"{name}: FAIL ({c.get('detail', '')})")
        elif c.get("required") is True and c.get("status") != "PASS":
            problems.append(f"{name}: {c.get('status')} ({c.get('detail', '')})")
    if problems:
        raise Refused("the receipt does not confirm " + tag + ": "
                      + "; ".join(sorted(set(problems))))
    if receipt.get("version") != confirm.version_of_tag(tag):
        raise Refused(f"the receipt records version {receipt.get('version')!r}, "
                      f"expected {confirm.version_of_tag(tag)!r}")
    receipt["_advisory"] = [
        f"{n}: {c.get('detail', '')}" for n, c in checks.items()
        if n in confirm.ADVISORY_CHECKS and c.get("status") == "FAIL"]
    try:
        receipt["_table"] = confirm.table_from_json(receipt.get("assets"))
    except ValueError as e:
        raise Refused(f"the receipt's asset table is unusable: {e}") from e
    return receipt


def fetch_published(tag: str, opener=None) -> dict:
    """{asset name: (size, sha256)} for the release *tag*, via the GitHub API."""
    try:
        body = confirm.fetch_release_body(tag, opener=opener)
    except Exception as e:
        raise Refused(f"could not read the {tag} release from the GitHub API: "
                      f"{type(e).__name__}: {e}") from e
    try:
        return confirm.parse_release_assets(body)
    except ValueError as e:
        raise Refused(f"the {tag} release listing is unusable: {e}") from e


# --------------------------------------------------------------------------- #
#  pins.py parsing and rewriting (pure text -> text)                           #
# --------------------------------------------------------------------------- #

def _one(pattern: re.Pattern, text: str, what: str) -> re.Match:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    return matches[0]


def pinned_tag(text: str) -> str:
    return _one(_TAG_LINE_RE, text, "TAG").group(2)


def pinned_table(text: str) -> dict:
    """{(platform, build): (name, size, sha256)} as written in pins.py."""
    entries = list(_ENTRY_RE.finditer(text))
    starts = list(_ENTRY_START_RE.finditer(text))
    if not entries or len(entries) != len(starts):
        raise Refused(f"ASSETS: {len(starts)} entries start but {len(entries)} match "
                      "the shape this script edits")
    table: dict = {}
    for m in entries:
        key = (m.group("plat"), m.group("build"))
        if key in table:
            raise Refused(f"ASSETS: {key} appears more than once")
        table[key] = (m.group("name"), int(m.group("size")), m.group("sha"))
    return table


def rewrite(text: str, tag: str, version: str, table: dict) -> str:
    """pins.py text with TAG, VERSION and every ASSETS entry set for the release.

    Raises Refused unless each region is found once, _BASE_URL is still derived
    from TAG, and *table* covers exactly the keys written in the file with the
    same asset names."""
    _one(_BASE_URL_RE, text, "_BASE_URL (derived from TAG)")
    current = pinned_table(text)
    if set(current) != set(table):
        raise Refused("the asset table keys differ from pins.py: "
                      f"{sorted(set(current) ^ set(table))}")
    for key, (name, _size, _sha) in current.items():
        if table[key][0] != name:
            raise Refused(f"asset name for {key} changed ({name!r} -> {table[key][0]!r}); "
                          "that is not a mechanical bump")
    _one(_TAG_LINE_RE, text, "TAG")
    text = _TAG_LINE_RE.sub(lambda m: m.group(1) + tag + m.group(3), text, count=1)
    _one(_VERSION_LINE_RE, text, "VERSION")
    text = _VERSION_LINE_RE.sub(lambda m: m.group(1) + version + m.group(3), text,
                                count=1)

    def entry(m: re.Match) -> str:
        _name, size, sha = table[(m.group("plat"), m.group("build"))]
        return (f'("{m.group("plat")}", "{m.group("build")}"): (\n'
                f'{m.group("i1")}"{m.group("name")}", {size},\n'
                f'{m.group("i2")}"{sha}"),')
    return _ENTRY_RE.sub(entry, text)


def check_rewritten(new_text: str, tag: str, version: str, table: dict) -> None:
    """Parse the rewritten pins.py and confirm its TAG, VERSION and ASSETS
    literals state the release exactly."""
    try:
        tree = ast.parse(new_text)
    except SyntaxError as e:
        raise Refused(f"the rewritten pins.py does not parse: {e}") from e
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1                 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)                 and node.value is not None:
            name, value = node.target.id, node.value
        else:
            continue
        if name in ("TAG", "VERSION", "ASSETS"):
            try:
                found[name] = ast.literal_eval(value)
            except ValueError as e:
                raise Refused(f"the rewritten {name} is not a literal: {e}") from e
    if found != {"TAG": tag, "VERSION": version, "ASSETS": table}:
        raise Refused("the rewritten pins.py does not state the target release "
                      "(TAG, VERSION and ASSETS must all agree)")


# --------------------------------------------------------------------------- #
#  Other mentions                                                              #
# --------------------------------------------------------------------------- #

def git_grep_mentions(repo: Path, needles: list) -> list:
    """Tracked files under *repo* containing any of *needles*. Raises Refused
    when git cannot answer."""
    args = ["git", "-C", str(repo), "grep", "-I", "-l", "-F"]
    for n in needles:
        args += ["-e", n]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise Refused(f"could not search the tree for other mentions: {e}") from e
    if r.returncode == 1:
        return []
    if r.returncode != 0:
        raise Refused("could not search the tree for other mentions: "
                      f"git grep exited {r.returncode}: {r.stderr.strip()}")
    return sorted(p for p in r.stdout.splitlines() if p)


def other_mentions(repo: Path, old_text: str, scanner=git_grep_mentions) -> list:
    """Tracked files besides MENTION_EXEMPT that assign the pinned tag, link a
    release download at it, or contain any pinned asset sha256."""
    tag = pinned_tag(old_text)
    needles = ([f'TAG = "{tag}"', f"download/{tag}/"]
               + [v[2] for v in pinned_table(old_text).values()])
    return [p for p in scanner(repo, needles) if p not in MENTION_EXEMPT]


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> tuple[str, str]:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def plan(text: str, tag: str, receipt_path, opener=None, scanner=git_grep_mentions,
         repo: Path = REPO, require_receipt: bool = False) -> tuple[str, dict | None]:
    """(new pins.py text, the validated receipt or None). Raises Refused."""
    if confirm.parse_tag(tag) is None:
        raise Refused(f"{tag!r} is not a vX.Y[.Z] release tag")
    current_tag = pinned_tag(text)
    cur, new = confirm.parse_tag(current_tag), confirm.parse_tag(tag)
    if cur is None:
        raise Refused(f"the pinned TAG {current_tag!r} is not a vX.Y[.Z] release tag")
    if new <= cur:
        raise Refused(f"{tag} is not newer than the pinned {current_tag}; this script "
                      "only ever advances the pin")
    receipt = None
    if receipt_path:
        receipt = load_receipt(receipt_path, tag)
    elif require_receipt:
        raise Refused("--write needs --receipt: a bump without the confirm is the "
                      "untested-build problem the pin exists to remove")
    version = confirm.version_of_tag(tag)
    published = fetch_published(tag, opener)
    try:
        table = confirm.build_table(pinned_table(text), published)
    except ValueError as e:
        raise Refused(str(e)) from e
    if receipt is not None and receipt["_table"] != table:
        raise Refused("the release's published sizes or digests differ from the table "
                      "the confirm run installed from; confirm the release again")
    mentions = other_mentions(repo, text, scanner)
    if mentions:
        raise Refused("other tracked files name the pinned tag or an asset hash and "
                      f"would be left behind: {', '.join(mentions)}")
    new_text = rewrite(text, tag, version, table)
    check_rewritten(new_text, tag, version, table)
    return new_text, receipt


def checklist(tag: str, receipt: dict | None) -> str:
    lines = [
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_koboldcpp_runtime.py tests/test_koboldcpp_server.py",
        "       tests/test_bump_koboldcpp_pin.py tests/test_confirm_koboldcpp_runtime.py",
        f"  2. CHANGELOG.md, [Unreleased]: one bullet, native music generation now runs "
        f"KoboldCpp {confirm.version_of_tag(tag)}",
    ]
    if receipt is not None and receipt.get("_advisory"):
        lines.append("  - ADVISORY failures in the receipt (they did not block the bump):")
        lines += [f"       - {a}" for a in receipt["_advisory"]]
    if receipt is not None:
        lines.append("  3. NOT measured by the confirm run (say so rather than claiming "
                     "them):")
        lines += [f"       - {item}" for item in receipt.get("not_measured", [])]
    return "\n".join(lines)


def main(argv=None, *, opener=None, scanner=git_grep_mentions) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the release to pin, e.g. v1.123")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_koboldcpp_runtime.py --tag; "
                         "required with --write")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)
    tag = args.tag.strip()
    try:
        text, newline = _read(PINS_PATH)
        new_text, receipt = plan(text, tag, args.receipt, opener=opener,
                                 scanner=scanner, require_receipt=args.write)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1
    if receipt is None:
        print("no receipt given: dry run only, nothing is confirmed")
    else:
        print(f"receipt: {tag} confirmed (verdict PASS, "
              f"{len(receipt['checks'])} checks)")
    rel = (PINS_PATH.relative_to(REPO).as_posix() if PINS_PATH.is_relative_to(REPO)
           else PINS_PATH.name)
    if args.write:
        _write(PINS_PATH, new_text, newline)
        print(f"wrote {rel}")
    else:
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(tag, receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
