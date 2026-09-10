#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing localm's pinned llama.cpp build.

A bump is one procedure, not a constant edit. This script rewrites the values that
must move together and prints the steps it cannot do. It refuses to write unless a
receipt from ``scripts/confirm_llama_runtime.py --receipt`` shows the target tag
loaded AND generated on every backend named by ``--require``.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched):

  localm/setup_llama.py
    _PINNED_TAG                   the target tag
    _PINNED_FALLBACK_SHA256       the upstream block only: every asset of the
                                  target release with the sha256 digest the
                                  GitHub API publishes for it
  localm/inference/backends/llamacpp/_api.py
    MTP_ARCH_SOURCE_TAG           the target tag
    MTP_GRAPH_ARCHITECTURES       re-derived from upstream source at the target
                                  tag by scripts/check_mtp_arch_allowlist.refresh

WHAT IT CHECKS BEFORE WRITING:
  * the receipt names the target tag and every ``--require`` backend is PASS;
  * the "measured" set in setup_llama._PIN_CONFIRMATION equals the receipt's
    PASS set (a dry run only warns about a mismatch; ``--write`` refuses);
  * the release has a sha256 digest for every asset and at least
    ``MIN_ASSETS`` of them;
  * each edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the ctypes
binding (scripts/check_llama_abi.py), the pre-tokenizer regex check
(scripts/check_pretokenizer_redos.py), _PIN_CONFIRMATION's wording, the tests,
and the changelog bullet.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when refused
(missing or inconsistent evidence, upstream unreadable, a file region not found
exactly once).

Environment: GITHUB_TOKEN is sent as a bearer token for the release lookup.

Usage:
    python scripts/bump_llama_pin.py --tag b10649                       # dry run, no receipt
    python scripts/bump_llama_pin.py --tag b10649 --receipt confirm.json
    python scripts/bump_llama_pin.py --tag b10649 --receipt confirm.json --write
    python scripts/bump_llama_pin.py --tag b10649 --receipt c.json --require cpu

Needs localm importable (the verified HTTPS opener). Nothing under localm/
imports this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SETUP_PATH = REPO / "localm" / "setup_llama.py"
API_PATH = REPO / "localm" / "inference" / "backends" / "llamacpp" / "_api.py"
MTP_SCRIPT = REPO / "scripts" / "check_mtp_arch_allowlist.py"

UPSTREAM_REPO = "ggml-org/llama.cpp"
RELEASE_URL = "https://api.github.com/repos/%s/releases/tags/%s"
# Fewer usable assets than this means the release listing is not the one localm
# installs from (b10375 publishes 27).
MIN_ASSETS = 16
DEFAULT_REQUIRE = ("cpu", "vulkan")

_TAG_RE = re.compile(r"^b\d+$")
_PIN_RE = re.compile(r'^(_PINNED_TAG\s*=\s*")([^"]+)(")', re.M)
_MTP_TAG_RE = re.compile(r'^(MTP_ARCH_SOURCE_TAG\s*=\s*")([^"]+)(")', re.M)
_SHA_BLOCK_RE = re.compile(
    r'(?P<indent>[ \t]+)# tag (?P<tag>\S+) upstream assets \(_PINNED_TAG\)\.(?P<rest>[^\n]*)\n'
    r'(?P<comment>(?:[ \t]+#[^\n]*\n)*)'
    r'(?P<entries>(?:[ \t]+"[^"\n]+": "[0-9a-f]{64}",\n)+)')
_MTP_SET_RE = re.compile(
    r'(?P<head>MTP_GRAPH_ARCHITECTURES = frozenset\(\{\n)'
    r'(?P<entries>(?:[ \t]+"[^"\n]+",\n)*)'
    r'(?P<tail>\}\))')
_MEASURED = "load + generate, measured"


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


# --------------------------------------------------------------------------- #
#  Upstream                                                                    #
# --------------------------------------------------------------------------- #

def fetch_release_assets(tag: str, opener=None) -> dict:
    """{asset name: sha256} for every asset of the release tagged *tag*.

    Raises Refused when the release cannot be read, an asset carries no sha256
    digest, or fewer than MIN_ASSETS assets are listed."""
    if opener is None:
        from localm.http_ssl import verified_urlopen
        opener = verified_urlopen
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-bump-llama-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(RELEASE_URL % (UPSTREAM_REPO, tag), headers=headers)
    try:
        with opener(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise Refused(f"could not read the {tag} release from the GitHub API: "
                      f"{type(e).__name__}: {e}")
    assets = body.get("assets") if isinstance(body, dict) else None
    if not isinstance(assets, list):
        raise Refused(f"the {tag} release listing carries no asset list")
    digests = {}
    for a in assets:
        name = a.get("name") if isinstance(a, dict) else None
        digest = a.get("digest") if isinstance(a, dict) else None
        if not isinstance(name, str) or not isinstance(digest, str) \
                or not digest.startswith("sha256:"):
            raise Refused(f"asset {name!r} of {tag} has no sha256 digest in the API "
                          "listing; the offline table cannot be filled from it")
        sha = digest.split("sha256:", 1)[1].strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise Refused(f"asset {name!r} of {tag} has a malformed digest {digest!r}")
        digests[name] = sha
    if len(digests) < MIN_ASSETS:
        raise Refused(f"{tag} lists only {len(digests)} asset(s) with digests; "
                      f"expected at least {MIN_ASSETS}, so this is not a complete "
                      "release listing")
    return digests


def derive_mtp_architectures(tag: str) -> set:
    """The MTP-capable architecture set at *tag*, via
    scripts/check_mtp_arch_allowlist.refresh."""
    spec = importlib.util.spec_from_file_location("check_mtp_arch_allowlist", MTP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        derived = mod.refresh(tag)
    except Exception as e:
        raise Refused(f"could not re-derive MTP_GRAPH_ARCHITECTURES at {tag}: "
                      f"{type(e).__name__}: {e}")
    if not derived:
        raise Refused(f"the MTP re-derivation at {tag} returned an empty set")
    return set(derived)


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str, require: "tuple[str, ...]") -> set:
    """The set of backends the receipt reports PASS for *tag*.

    Raises Refused when the receipt is unreadable, names another tag, or any
    ``require`` backend is not PASS."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for {receipt.get('tag')!r}, not {tag}; "
                      "confirm the target tag itself")
    backends = receipt.get("backends") or {}
    passed = {b for b, r in backends.items()
              if isinstance(r, dict) and r.get("verdict") == "PASS"}
    missing = [b for b in require if b not in passed]
    if missing:
        reasons = "; ".join(
            f"{b}: {backends.get(b, {}).get('verdict', 'not run')} "
            f"({backends.get(b, {}).get('why', '')})" for b in missing)
        raise Refused(f"the receipt does not confirm {tag} on {', '.join(missing)}: "
                      f"{reasons}")
    return passed


def measured_backends(setup_text: str) -> set:
    """Backends whose _PIN_CONFIRMATION entry claims a measurement."""
    m = re.search(r"_PIN_CONFIRMATION = \{\n(?P<body>.*?)\n\}", setup_text, re.S)
    if not m:
        raise Refused("_PIN_CONFIRMATION not found in setup_llama.py")
    measured = set()
    for entry in re.finditer(r'^\s+"([^"]+)":\s*((?:"[^"]*"\s*)+),', m.group("body"), re.M):
        note = "".join(re.findall(r'"([^"]*)"', entry.group(2)))
        if _MEASURED in note:
            measured.add(entry.group(1))
    return measured


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


def set_pin(setup_text: str, tag: str) -> str:
    return _replace_once(_PIN_RE, setup_text,
                         lambda m: m.group(1) + tag + m.group(3), "_PINNED_TAG")


def set_sha_block(setup_text: str, tag: str, digests: dict) -> str:
    def repl(m):
        indent = m.group("indent")
        head = f"{indent}# tag {tag} upstream assets (_PINNED_TAG).{m.group('rest')}\n"
        entries = "".join(f'{indent}"{name}": "{digests[name]}",\n'
                          for name in sorted(digests))
        return head + m.group("comment") + entries
    return _replace_once(_SHA_BLOCK_RE, setup_text, repl, "_PINNED_FALLBACK_SHA256 upstream block")


def set_mtp_tag(api_text: str, tag: str) -> str:
    return _replace_once(_MTP_TAG_RE, api_text,
                         lambda m: m.group(1) + tag + m.group(3), "MTP_ARCH_SOURCE_TAG")


def set_mtp_set(api_text: str, archs: set) -> str:
    def repl(m):
        entries = "".join(f'    "{a}",\n' for a in sorted(archs))
        return m.group("head") + entries + m.group("tail")
    return _replace_once(_MTP_SET_RE, api_text, repl, "MTP_GRAPH_ARCHITECTURES")


def rewrite(setup_text: str, api_text: str, tag: str, digests: dict,
            archs: set) -> "tuple[str, str]":
    """(new setup_llama.py text, new _api.py text)."""
    setup_text = set_sha_block(set_pin(setup_text, tag), tag, digests)
    api_text = set_mtp_set(set_mtp_tag(api_text, tag), archs)
    return setup_text, api_text


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


def checklist(tag: str) -> str:
    return "\n".join([
        "REMAINING STEPS, not automated - each has its own check:",
        f"  1. python scripts/check_llama_abi.py --ref {tag}",
        "       must pass; if it fails, update _structs.py/_abi.py first, then rerun this script",
        "  2. python scripts/check_mtp_arch_allowlist.py --gate",
        "       must pass against the rewritten allowlist",
        f"  3. python scripts/check_pretokenizer_redos.py --ref {tag}",
        "       needs MSVC: run it on a Windows box with Build Tools, or dispatch",
        "       .github/workflows/llama-pin-currency.yml with candidate_tag",
        "  4. _PIN_CONFIRMATION in setup_llama.py: re-read every entry against the receipt",
        "       (which backends generated, on what hardware); a backend not in the",
        "       receipt must still say 'NOT measured'",
        "  5. pytest tests/test_llama_pin_constant_and_currency.py tests/test_mtp_arch_allowlist.py",
        "       tests/test_llama_runtime_version_pin.py tests/test_setup_llama_abi_walkback.py",
        "       tests/test_setup_llama_backends.py tests/test_cuda_arch_line_selection.py",
        "       tests/test_llamacpp_abi.py tests/test_check_llama_abi.py",
        "  6. CHANGELOG.md, [Unreleased]: one bullet, the shipped runtime build moved",
        "  7. python scripts/check_llama_pin.py --gate",
        "       must report the pin as current",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the upstream release to pin, e.g. b10649")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_llama_runtime.py --receipt "
                         "for --tag; required with --write")
    ap.add_argument("--require", action="append", default=None,
                    help="a backend the receipt must report PASS for (repeatable; "
                         f"default {', '.join(DEFAULT_REQUIRE)})")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    tag = args.tag.strip()
    require = tuple(args.require) if args.require else DEFAULT_REQUIRE
    try:
        if not _TAG_RE.match(tag):
            raise Refused(f"{tag!r} is not an upstream build tag (bNNNNN)")
        setup_text, setup_nl = _read(SETUP_PATH)
        api_text, api_nl = _read(API_PATH)

        passed = None
        if args.receipt:
            passed = load_receipt(Path(args.receipt), tag, require)
            print(f"receipt: {tag} confirmed on {', '.join(sorted(passed))}")
        elif args.write:
            raise Refused("--write needs --receipt: a bump without the confirm is "
                          "the untested-build problem the pin exists to remove")
        else:
            print("no receipt given: dry run only, nothing is confirmed")

        measured = measured_backends(setup_text)
        if passed is not None and passed != measured:
            msg = (f"_PIN_CONFIRMATION claims a measurement for "
                   f"{', '.join(sorted(measured)) or 'nothing'}; the receipt confirms "
                   f"{', '.join(sorted(passed)) or 'nothing'}. Edit the table to say "
                   "what was actually measured for this tag.")
            if args.write:
                raise Refused(msg)
            print(f"WARNING: {msg}")

        print(f"reading the {tag} release listing ...")
        digests = fetch_release_assets(tag)
        print(f"  {len(digests)} assets with sha256 digests")
        print(f"re-deriving MTP_GRAPH_ARCHITECTURES at {tag} ...")
        archs = derive_mtp_architectures(tag)
        print(f"  {len(archs)} architecture(s): {', '.join(sorted(archs))}")

        new_setup, new_api = rewrite(setup_text, api_text, tag, digests, archs)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    changed = False
    for path, old, new, newline in ((SETUP_PATH, setup_text, new_setup, setup_nl),
                                    (API_PATH, api_text, new_api, api_nl)):
        if old == new:
            continue
        changed = True
        rel = path.relative_to(REPO).as_posix()
        if args.write:
            _write(path, new, newline)
            print(f"wrote {rel}")
        else:
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    if not changed:
        print(f"nothing to change: the tree already pins {tag} with matching digests "
              "and allowlist")
    elif not args.write:
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
