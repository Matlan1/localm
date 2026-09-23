#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing localm's pinned ComfyUI commit.

Direct structural analog of scripts/bump_llama_pin.py, adapted for a git-clone
pin instead of a downloaded-release pin: there is no sha256 asset table to
maintain (ComfyUI is fetched by ``git clone`` + ``git checkout``, not release
archives) and no MTP-style derived constant. It refuses to write unless a
receipt from ``scripts/confirm_comfyui_runtime.py --receipt`` shows the
target tag AND commit confirmed, with every check named by ``--require`` PASS.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched), both in localm/media/managed_comfy_fresh.py:

  COMFYUI_PINNED_COMMIT   the target 40-hex commit sha
  COMFYUI_PINNED_VERSION  the target tag

COMFYUI_REPO and COMFYUI_PLACEMENT_MIN_VERSION are never touched.

WHAT IT CHECKS BEFORE WRITING:
  * the receipt names the target tag AND commit, and every ``--require``d
    check is PASS;
  * the target tag is strictly newer than the currently pinned version
    (forward-only - a same-or-older tag is refused, never silently applied);
  * the edited region is found exactly once.

NO NETWORK CALLS: the receipt already carries the tag->commit resolution,
cross-checked against a real clone by confirm_comfyui_runtime.py itself. That
keeps this script's own tests fully offline, unlike bump_llama_pin.py (which
still needs the GitHub API for the sha256 asset table).

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the targeted
tests, ``check_comfyui_pin.py --gate``, a CHANGELOG bullet (with
``localm comfy update --reinstall-requirements`` advice when the receipt shows
``requirements.txt`` changed - plain ``update`` does not reinstall
requirements by default), and a manual model-based run per
qa/test-plans/media-gen.md.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused (missing or inconsistent evidence, a same-or-older tag, the edited
region not found exactly once).

Usage:
    python scripts/bump_comfyui_pin.py --tag v0.32.0 --commit <40-hex>              # dry run
    python scripts/bump_comfyui_pin.py --tag v0.32.0 --commit <40-hex> --receipt confirm.json
    python scripts/bump_comfyui_pin.py --tag v0.32.0 --commit <40-hex> --receipt c.json --write
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSTANTS_PATH = REPO / "localm" / "media" / "managed_comfy_fresh.py"
CHECK_PIN_SCRIPT = REPO / "scripts" / "check_comfyui_pin.py"

_TAG_RE = re.compile(r"^v\d+(?:\.\d+)+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PIN_COMMIT_RE = re.compile(r'^(COMFYUI_PINNED_COMMIT\s*=\s*")([0-9a-f]{40})(")', re.M)
_PIN_VERSION_RE = re.compile(r'^(COMFYUI_PINNED_VERSION\s*=\s*")([^"]+)(")', re.M)

# Every check scripts/confirm_comfyui_runtime.py's receipt can report, in the
# SAME names that script uses - this is the shared contract between the two
# scripts. Kept as one canonical tuple so neither can drift from the other.
CHECK_NAMES = ("isolation", "provision", "checkout", "custom_nodes", "localm_patches",
              "torch_device", "identity", "nodes_registered", "shipped_workflows",
              "gpu_roundtrip")
DEFAULT_REQUIRE = CHECK_NAMES


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str, commit: str, require: "tuple[str, ...]") -> dict:
    """The receipt dict, validated: it must be for exactly *tag* AND *commit*,
    and every ``require``d check must report PASS.

    Raises Refused when the receipt is unreadable, names another tag or
    commit, or any required check is not PASS. Returns the whole receipt
    (not just the PASS set) so the caller can read requirements_changed etc.
    off ``receipt["baseline"]`` for the checklist."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for tag {receipt.get('tag')!r}, not {tag}; "
                      "confirm the target tag itself")
    if receipt.get("commit") != commit:
        raise Refused(f"the receipt is for commit {receipt.get('commit')!r}, not {commit}; "
                      "confirm the target commit itself")
    checks = receipt.get("checks") or {}
    passed = {c for c, r in checks.items()
              if isinstance(r, dict) and r.get("verdict") == "PASS"}
    missing = [c for c in require if c not in passed]
    if missing:
        reasons = "; ".join(
            f"{c}: {checks.get(c, {}).get('verdict', 'not run')} "
            f"({checks.get(c, {}).get('why', '')})" for c in missing)
        raise Refused(f"the receipt does not confirm {tag}@{commit} on {', '.join(missing)}: "
                      f"{reasons}")
    return receipt


def _current_pin_version(text: str) -> str:
    m = _PIN_VERSION_RE.search(text)
    if not m:
        raise Refused("COMFYUI_PINNED_VERSION not found in "
                      f"{CONSTANTS_PATH} - has the constant been renamed?")
    return m.group(2)


def _forward_only(current_tag: str, target_tag: str) -> None:
    """Raises Refused unless target_tag is a STRICTLY newer version than
    current_tag - a same-or-older tag must never silently "advance" the pin
    (or silently no-op past a real mistake in the target)."""
    spec = importlib.util.spec_from_file_location("check_comfyui_pin", CHECK_PIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    current_v = mod._parse_version(current_tag)
    target_v = mod._parse_version(target_tag)
    if current_v is None or target_v is None:
        raise Refused(f"could not compare versions ({current_tag!r} vs {target_tag!r}) - "
                      "one of them is not a plain vX.Y[.Z] tag")
    if target_v <= current_v:
        raise Refused(f"{target_tag} is not newer than the currently pinned {current_tag}; "
                      "this script only ever advances the pin forward")


# --------------------------------------------------------------------------- #
#  Rewrites (pure text -> text)                                                #
# --------------------------------------------------------------------------- #

def _replace_once(pattern: "re.Pattern", text: str, repl, what: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    m = matches[0]
    return text[:m.start()] + repl(m) + text[m.end():]


def set_commit(text: str, commit: str) -> str:
    return _replace_once(_PIN_COMMIT_RE, text,
                         lambda m: m.group(1) + commit + m.group(3), "COMFYUI_PINNED_COMMIT")


def set_version(text: str, tag: str) -> str:
    return _replace_once(_PIN_VERSION_RE, text,
                         lambda m: m.group(1) + tag + m.group(3), "COMFYUI_PINNED_VERSION")


def rewrite(text: str, tag: str, commit: str) -> str:
    return set_version(set_commit(text, commit), tag)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> "tuple[str, str]":
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    """Write *text* (LF newlines) using the file's own newline sequence."""
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def checklist(tag: str, requirements_changed: bool) -> str:
    lines = [
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_check_comfyui_pin.py tests/test_bump_comfyui_pin.py",
        "       tests/test_confirm_comfyui_runtime.py tests/test_media_placement.py",
        "       tests/test_managed_comfy_s1.py tests/test_managed_comfy_s3.py",
        "       tests/test_managed_comfy_s4.py tests/test_managed_comfy_s5_gui.py",
        "       tests/test_comfy_cli_outcome_honesty.py",
        "  2. CHANGELOG.md, [Unreleased]: one bullet, the managed ComfyUI moved to "
        f"{tag}",
    ]
    if requirements_changed:
        lines.append(
            "       - mention 'localm comfy update --reinstall-requirements': plain "
            "'update' does not reinstall requirements.txt by default, and the "
            "receipt shows it changed for this bump")
    lines += [
        "  3. python scripts/check_comfyui_pin.py --gate",
        "       must report the pin as current",
        "  4. A real model-based run per qa/test-plans/media-gen.md - this bump's own",
        "       confirm is deliberately model-free (see scripts/confirm_comfyui_runtime.py's",
        "       own not_covered list in the receipt)",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the upstream release to pin, e.g. v0.32.0")
    ap.add_argument("--commit", required=True,
                    help="the 40-hex commit sha --tag resolves to")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_comfyui_runtime.py --receipt "
                         "for --tag/--commit; required with --write")
    ap.add_argument("--require", action="append", default=None,
                    help="a confirm check the receipt must report PASS for (repeatable; "
                         f"default: every check - {', '.join(DEFAULT_REQUIRE)})")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    tag = args.tag.strip()
    commit = args.commit.strip().lower()
    require = tuple(args.require) if args.require else DEFAULT_REQUIRE
    requirements_changed = False
    try:
        if not _TAG_RE.match(tag):
            raise Refused(f"{tag!r} is not an upstream release tag (vX.Y[.Z...])")
        if not _COMMIT_RE.match(commit):
            raise Refused(f"{commit!r} is not a 40-character hex commit sha")
        text, newline = _read(CONSTANTS_PATH)
        _forward_only(_current_pin_version(text), tag)

        if args.receipt:
            receipt = load_receipt(Path(args.receipt), tag, commit, require)
            print(f"receipt: {tag}@{commit} confirmed on "
                 f"{', '.join(sorted(c for c in receipt.get('checks', {}) if receipt['checks'][c].get('verdict') == 'PASS'))}")
            requirements_changed = bool(
                (receipt.get("baseline") or {}).get("requirements_changed"))
        elif args.write:
            raise Refused("--write needs --receipt: a bump without the confirm is "
                          "the untested-build problem the pin exists to remove")
        else:
            print("no receipt given: dry run only, nothing is confirmed")

        new_text = rewrite(text, tag, commit)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    if new_text == text:
        print(f"nothing to change: the tree already pins {tag}@{commit}")
    elif args.write:
        _write(CONSTANTS_PATH, new_text, newline)
        print(f"wrote {CONSTANTS_PATH.relative_to(REPO).as_posix()}")
    else:
        rel = CONSTANTS_PATH.relative_to(REPO).as_posix()
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(tag, requirements_changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
