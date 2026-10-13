#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advance localm's pinned Linux CUDA runtime wheels (nvidia-cuda-runtime, nvidia-cublas).

``localm/setup_llama/cuda.py`` pins each CUDA runtime package to an exact
version and the sha256 of its Linux x86_64 wheel (``_CUDA_RUNTIME_PIN``). This
script rewrites those entries for the packages named in ``--tag``. It is
review-only: a person reads the diff and the printed checklist, then merges.

``--tag`` is a comma-separated list of ``package==version`` pairs. A package
named at its pinned version, or left out, keeps its pin; at least one named
package must move.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched):

  localm/setup_llama/cuda.py
    _CUDA_RUNTIME_PIN      the version and wheel sha256 of each named package
  tests/test_linux_cuda_runtime_provisioning.py
    only a literal copy of an old wheel sha256, when the file carries one

WHAT IT CHECKS BEFORE WRITING:
  * every named package is already pinned, appears once in ``--tag``, and the
    version is a plain dotted release number;
  * a moving package's new version is strictly newer than the pinned one (an
    older one is refused) and stays on the same major line (the cuda-12 packages stay 12.x, the unsuffixed ones stay on
    their pinned major);
  * PyPI lists that exact version of that exact package, not yanked, with
    exactly one Linux x86_64 wheel, and the wheel carries a sha256 digest;
  * each edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the targeted
tests, confirming the cudart and cuBLAS of one CUDA line come from the same
CUDA release, the driver floor for the line, and the CHANGELOG bullet. The
wheels are not loaded on a GPU here.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused.

Usage:
    python scripts/bump_cuda_runtime_pin.py --tag "nvidia-cublas-cu12==12.9.3.1"
    python scripts/bump_cuda_runtime_pin.py --tag "nvidia-cuda-runtime==13.4.99,nvidia-cublas==13.9.0.1" --write
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CUDA_REL = "localm/setup_llama/cuda.py"
TEST_REL = "tests/test_linux_cuda_runtime_provisioning.py"
PYPI_URL = "https://pypi.org/pypi/%s/%s/json"
MAX_JSON_BYTES = 8 * 1024 * 1024

_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PIN_BLOCK_RE = re.compile(r"^_CUDA_RUNTIME_PIN = \{\n(?P<body>.*?)^\}\n", re.M | re.S)
_ENTRY_RE = re.compile(
    r'(?P<pre>^[ \t]*"(?P<pkg>[^"\n]+)":[ \t]*\([ \t]*")(?P<ver>[^"\n]+)'
    r'(?P<mid>"[ \t]*,[ \t]*(?:\n[ \t]*)?")(?P<sha>[0-9a-f]{64})(?P<post>"[ \t]*\),?)',
    re.M)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


# --------------------------------------------------------------------------- #
#  The request                                                                #
# --------------------------------------------------------------------------- #

def parse_versions(text: str) -> dict:
    """{package: version} from ``pkg==ver,pkg==ver``. Raises Refused on an
    empty list, a malformed pair, a duplicate package or a malformed version."""
    out: dict = {}
    pairs = [p.strip() for p in (text or "").split(",")]
    if not text or not text.strip() or any(not p for p in pairs):
        raise Refused("--tag needs one or more package==version pairs, comma separated")
    for pair in pairs:
        pkg, sep, ver = pair.partition("==")
        if not sep or not _PACKAGE_RE.match(pkg) or "=" in ver:
            raise Refused(f"{pair!r} is not package==version")
        if not _VERSION_RE.match(ver):
            raise Refused(f"{pkg}: {ver!r} is not a plain dotted release version")
        if pkg in out:
            raise Refused(f"{pkg} appears more than once in --tag")
        out[pkg] = ver
    return out


def version_tuple(version: str) -> tuple:
    return tuple(int(p) for p in version.split("."))


def is_newer(new: str, old: str) -> bool:
    a, b = version_tuple(new), version_tuple(old)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


# --------------------------------------------------------------------------- #
#  Upstream (injectable)                                                      #
# --------------------------------------------------------------------------- #

def _default_open(req, timeout):
    from localm.http_ssl import verified_urlopen
    return verified_urlopen(req, timeout=timeout)


def fetch_json(url: str, opener=None) -> object:
    """GET *url* and decode it as JSON. Raises Refused when it cannot be read."""
    opener = opener or _default_open
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "localm-bump-cuda-runtime-pin"})
    try:
        with opener(req, 30) as resp:
            raw = resp.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("response is larger than the allowed size")
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise Refused(f"could not read {url}: {type(e).__name__}: {e}") from e


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch_wheel_sha(package: str, version: str, fetch_json_fn=fetch_json) -> str:
    """The sha256 of *package* *version*'s Linux x86_64 wheel, from PyPI's JSON.

    Raises Refused unless PyPI answers for exactly this package and version and
    lists exactly one non-yanked Linux x86_64 wheel carrying a sha256 digest."""
    url = PYPI_URL % (urllib.parse.quote(package, safe=""), version)
    body = fetch_json_fn(url)
    if not isinstance(body, dict) or not isinstance(body.get("info"), dict):
        raise Refused(f"PyPI's answer for {package}=={version} is not a release document")
    info = body["info"]
    if _canonical(str(info.get("name", ""))) != _canonical(package) \
            or info.get("version") != version:
        raise Refused(f"PyPI answered with {info.get('name')!r} {info.get('version')!r}, "
                      f"not {package}=={version}")
    if info.get("yanked"):
        raise Refused(f"{package}=={version} is yanked on PyPI")
    urls = body.get("urls")
    if not isinstance(urls, list):
        raise Refused(f"PyPI lists no files for {package}=={version}")
    wheels = []
    for entry in urls:
        name = str(entry.get("filename", "")) if isinstance(entry, dict) else ""
        if name.endswith(".whl") and "x86_64" in name and "linux" in name.lower():
            wheels.append(entry)
    if len(wheels) != 1:
        raise Refused(f"{package}=={version}: expected exactly one Linux x86_64 wheel, "
                      f"found {len(wheels)}")
    wheel = wheels[0]
    if wheel.get("yanked"):
        raise Refused(f"the {package}=={version} Linux x86_64 wheel is yanked on PyPI")
    sha = (wheel.get("digests") or {}).get("sha256")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise Refused(f"the {package}=={version} Linux x86_64 wheel has no sha256 digest "
                      f"in the PyPI listing ({sha!r})")
    return sha


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


def read_pin(text: str) -> dict:
    """{package: (version, sha256)} from the ``_CUDA_RUNTIME_PIN`` block.

    Raises Refused when the block is not found exactly once or an entry is
    repeated."""
    blocks = list(_PIN_BLOCK_RE.finditer(text))
    if len(blocks) != 1:
        raise Refused(f"_CUDA_RUNTIME_PIN: expected exactly one block, found {len(blocks)}; "
                      "the file shape this script edits has changed")
    pin: dict = {}
    for m in _ENTRY_RE.finditer(blocks[0].group("body")):
        if m.group("pkg") in pin:
            raise Refused(f"_CUDA_RUNTIME_PIN lists {m.group('pkg')} more than once")
        pin[m.group("pkg")] = (m.group("ver"), m.group("sha"))
    if not pin:
        raise Refused("_CUDA_RUNTIME_PIN holds no entries this script can read")
    return pin


def rewrite_pin(text: str, package: str, version: str, sha: str) -> str:
    """*text* with *package*'s entry moved to *version* and *sha*."""
    matches = [m for m in _ENTRY_RE.finditer(text) if m.group("pkg") == package]
    block = list(_PIN_BLOCK_RE.finditer(text))
    if len(block) != 1 or len(matches) != 1:
        raise Refused(f"{package}: expected exactly one pin entry, found {len(matches)}; "
                      "the file shape this script edits has changed")
    m = matches[0]
    if not (block[0].start("body") <= m.start() < block[0].end("body")):
        raise Refused(f"{package}: the entry found is outside _CUDA_RUNTIME_PIN")
    return (text[:m.start("ver")] + version + m.group("mid") + sha + m.group("post")
            + text[m.end():])


def rewrite_test_sha(text: str, old_sha: str, new_sha: str) -> str:
    """*text* with a literal *old_sha* replaced by *new_sha*. Raises Refused when
    the digest appears more than once."""
    count = text.count(old_sha)
    if count > 1:
        raise Refused(f"{TEST_REL} carries the old digest {count} times; "
                      "the file shape this script edits has changed")
    return text.replace(old_sha, new_sha) if count else text


def check_move(package: str, old: str, new: str) -> bool:
    """True when *new* is strictly newer than *old* on the same major; False when
    it equals *old* (the package stays as pinned). Raises Refused when *new* is
    older or leaves the major line."""
    if not _VERSION_RE.match(old):
        raise Refused(f"{package}: the pinned version {old!r} is not a plain dotted version")
    if version_tuple(new)[0] != version_tuple(old)[0]:
        raise Refused(f"{package}: {new} leaves the {version_tuple(old)[0]}.x line of the "
                      f"pinned {old}; a different CUDA major is a different runtime line")
    if package.endswith("-cu12") and version_tuple(new)[0] != 12:
        raise Refused(f"{package}: a -cu12 package must stay on 12.x, got {new}")
    if not is_newer(new, old):
        if not is_newer(old, new):
            return False
        raise Refused(f"{package}: {new} is not newer than the pinned {old}; "
                      "this script only moves a pin forward")
    return True


def moving(pin: dict, wanted: dict) -> dict:
    """The subset of *wanted* that moves a pin forward. Raises Refused for an
    unpinned package, an older or cross-major version, or when nothing moves."""
    out = {}
    for package, version in wanted.items():
        if package not in pin:
            raise Refused(f"{package} is not pinned in _CUDA_RUNTIME_PIN "
                          f"(pinned: {', '.join(sorted(pin))})")
        if check_move(package, pin[package][0], version):
            out[package] = version
    if not out:
        raise Refused("every named package is already at its pinned version; nothing to bump")
    return out


def plan(cuda_text: str, test_text: str, moves: dict, shas: dict) -> tuple:
    """(new cuda.py text, new test text, review lines) for the packages in
    *moves*. Raises Refused."""
    pin = read_pin(cuda_text)
    moves = moving(pin, moves)
    new_cuda, new_test, review = cuda_text, test_text, []
    for package, version in moves.items():
        old_version, old_sha = pin[package]
        new_cuda = rewrite_pin(new_cuda, package, version, shas[package])
        new_test = rewrite_test_sha(new_test, old_sha, shas[package])
        for number, line in enumerate(test_text.split("\n"), 1):
            if re.search(rf"(?<![\d.]){re.escape(old_version)}(?![\d])", line):
                review.append(f"{TEST_REL}:{number}: {line.strip()[:100]}")
    return new_cuda, new_test, review


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #

def checklist(moves: dict, pin: dict, review: list) -> str:
    kept = sorted(set(pin) - set(moves))
    lines = ["REMAINING STEPS, not automated - each has its own check:"]
    lines += [
        "  1. pytest tests/test_linux_cuda_runtime_provisioning.py tests/test_bump_cuda_runtime_pin.py",
        "       tests/test_cuda_staged_runtime.py tests/test_cuda_setup.py tests/test_setup_cuda_asset.py",
        "  2. cudart and cuBLAS of one CUDA line must come from the same CUDA release: compare the",
        "       versions of the pairs below (a half-moved pair loads a mismatched runtime)",
    ]
    for package in sorted(pin):
        mark = "moved" if package in moves else "kept "
        version = moves.get(package, pin[package][0])
        lines.append(f"       {mark} {package}=={version}")
    if kept:
        lines.append(f"       (not moved, still pinned: {', '.join(kept)})")
    lines += [
        "  3. _MIN_DRIVER_CUDA in localm/setup_llama/cuda.py: the driver floor per line still",
        "       covers the new runtime",
        "  4. These wheels are not loaded on a GPU by this script; a Linux NVIDIA box (or the CUDA",
        "       container check) is the only place the new runtime is exercised",
        "  5. CHANGELOG.md, [Unreleased]: one bullet, the CUDA runtime fetched for Linux moved",
    ]
    if review:
        lines.append("  6. lines in the test file that name an old version (fixtures or history; "
                     "edit only if now false):")
        lines += [f"       {r}" for r in review]
    return "\n".join(lines)


def main(argv=None, *, fetch_json_fn=fetch_json) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help='package==version pairs, comma separated, e.g. '
                         '"nvidia-cublas-cu12==12.9.3.1,nvidia-cuda-runtime-cu12==12.9.99"')
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    cuda_path, test_path = REPO / CUDA_REL, REPO / TEST_REL
    try:
        wanted = parse_versions(args.tag)
        cuda_text, cuda_nl = _read(cuda_path)
        test_text, test_nl = _read(test_path) if test_path.is_file() else ("", "\n")
        pin = read_pin(cuda_text)
        moves = moving(pin, wanted)
        shas = {}
        for package, version in moves.items():
            print(f"reading {package}=={version} from PyPI ...")
            shas[package] = fetch_wheel_sha(package, version, fetch_json_fn)
            print(f"  Linux x86_64 wheel sha256 {shas[package]}")
        new_cuda, new_test, review = plan(cuda_text, test_text, moves, shas)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    changed = False
    for path, rel, old, new, newline in ((cuda_path, CUDA_REL, cuda_text, new_cuda, cuda_nl),
                                         (test_path, TEST_REL, test_text, new_test, test_nl)):
        if old == new:
            continue
        changed = True
        if args.write:
            _write(path, new, newline)
            print(f"wrote {rel}")
        else:
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    if not changed:
        print("nothing to change: the tree already pins these versions")
    elif not args.write:
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(moves, pin, review))
    return 0


if __name__ == "__main__":
    sys.exit(main())
