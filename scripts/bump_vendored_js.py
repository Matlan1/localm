#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replace one vendored GUI library with the byte-exact upstream npm release.

The GUI ships marked, DOMPurify, highlight.js and KaTeX as plain files under
localm/plugins/gui/static/vendor/. This script swaps one of them for a newer
release and moves every pin that names the old bytes. It is review-only: a
person reads the diff and the printed checklist, then merges.

WHAT IT REWRITES (with ``--write``; without it, a summary and a unified diff
are printed and nothing is touched):

  marked        vendor/marked.min.js
                tests-js/vendor-marked.test.mjs        VENDORED_VERSION,
                  PINNED_HASH, the dist.shasum and ``npm pack`` provenance
                  comments, the banner test title
  DOMPurify     vendor/purify.min.js
                (tests-js/vendor-dompurify.test.mjs pins a version floor, not
                a version, and is not edited)
  highlight.js  vendor/highlight.min.js, and vendor/github-dark.min.css when
                the release carries a different one
                tests-js/vendor-highlightjs.test.mjs   VENDORED_VERSION,
                  PINNED_HASH, the ``npm pack`` comment, the banner test title
  KaTeX         vendor/katex.min.js, katex.min.css, auto-render.min.js and the
                whole fonts/*.woff2 set (replaced, added and removed to match
                the release)
                tests-js/vendor-katex.test.mjs         VENDORED_VERSION, the
                  three PINNED hashes, the ``npm pack`` comment

Library files are written exactly as the npm tarball carries them. A pinned hash
is sha256, base64, over CRLF-normalised bytes.

WHAT IT CHECKS BEFORE WRITING:
  * the tag is a plain MAJOR.MINOR.PATCH release and strictly newer than the
    vendored version;
  * the vendored version read from the test pin agrees with the file's own
    banner (or, for KaTeX, its embedded version string), and the vendored bytes
    still match the recorded pin;
  * the registry lists exactly this package and version, not deprecated, with a
    sha512 ``dist.integrity``; the downloaded tarball matches that integrity
    and the ``dist.shasum``;
  * each wanted tarball member exists exactly once and is a regular file;
  * every KaTeX woff2 font the new CSS references is in the new font set;
  * each edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the jsdom suite
(``npm ci && npm test``), reading upstream's changelog when the major version
changes (a 0.x minor change counts), re-checking advisory status for the new
version, the remaining prose that names the old version, the CHANGELOG bullet
and a look at rendered output.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused.

Environment: none required. Needs localm importable for the verified HTTPS
opener unless the registry seams are injected.

Usage:
    python scripts/bump_vendored_js.py --lib DOMPurify --tag 3.4.16
    python scripts/bump_vendored_js.py --lib KaTeX --tag 0.19.0 --write
"""

import argparse
import base64
import difflib
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDOR_REL = "localm/plugins/gui/static/vendor"
TESTS_REL = "tests-js"
NOTICE_RELS = (f"{VENDOR_REL}/README.md", "THIRD-PARTY-NOTICES.md")

NPM_REGISTRY = "https://registry.npmjs.org"
MAX_TARBALL_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_VERSION_CONST_RE = re.compile(r'^(const VENDORED_VERSION = ")([^"]+)(";)', re.M)
_HASH_CONST_RE = re.compile(r'^(const PINNED_HASH = ")([^"]+)(";)', re.M)
_SHASUM_RE = re.compile(r"(dist\.shasum \()([0-9a-f]{40})(\))")
_FONT_NAME_RE = re.compile(r"^KaTeX_[A-Za-z0-9_-]+\.woff2$")
_CSS_FONT_RE = re.compile(r"url\(fonts/(KaTeX_[A-Za-z0-9_-]+\.woff2)\)")
_TEXT_SUFFIXES = (".js", ".css")


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


@dataclass(frozen=True)
class Lib:
    """How one vendored library maps onto npm and onto its tests-js pin file."""
    key: str
    npm: str
    test: str
    files: tuple            # ((tar member, vendor-relative destination), ...)
    banner_re: str | None   # version in the vendored file's banner, if it has one
    banner_file: str | None
    pinned: bool            # the test file records the version and content hashes
    extra_files: tuple = ()  # replaced when the release differs, not pinned by a test
    fonts: bool = False
    shasum_comment: bool = False
    title_re: str | None = None


LIBS = {
    "marked": Lib(
        key="marked", npm="marked", test="vendor-marked.test.mjs",
        files=(("package/marked.min.js", "marked.min.js"),),
        banner_re=r"marked v(\d+\.\d+\.\d+)", banner_file="marked.min.js",
        pinned=True, shasum_comment=True,
        title_re=r'(the banner comment says )(\d+\.\d+\.\d+)(")'),
    "DOMPurify": Lib(
        key="DOMPurify", npm="dompurify", test="vendor-dompurify.test.mjs",
        files=(("package/dist/purify.min.js", "purify.min.js"),),
        banner_re=r"@license DOMPurify (\d+\.\d+\.\d+)", banner_file="purify.min.js",
        pinned=False),
    "highlight.js": Lib(
        key="highlight.js", npm="@highlightjs/cdn-assets", test="vendor-highlightjs.test.mjs",
        files=(("package/highlight.min.js", "highlight.min.js"),),
        extra_files=(("package/styles/github-dark.min.css", "github-dark.min.css"),),
        banner_re=r"Highlight\.js v(\d+\.\d+\.\d+)", banner_file="highlight.min.js",
        pinned=True, title_re=r"(and both say )(\d+\.\d+\.\d+)(\")"),
    "KaTeX": Lib(
        key="KaTeX", npm="katex", test="vendor-katex.test.mjs",
        files=(("package/dist/katex.min.js", "katex.min.js"),
               ("package/dist/katex.min.css", "katex.min.css"),
               ("package/dist/contrib/auto-render.min.js", "auto-render.min.js")),
        banner_re=r'version:"(\d+\.\d+\.\d+)"', banner_file="katex.min.js",
        pinned=True, fonts=True),
}


@dataclass
class Change:
    """One file edit: *new* is None to delete, *old* is None to create."""
    path: Path
    rel: str
    old: bytes | None
    new: bytes | None
    text: bool = False       # an edited source file: shown as a diff, keeps its newline
    newline: str = "\n"


# --------------------------------------------------------------------------- #
#  Versions and hashes                                                        #
# --------------------------------------------------------------------------- #

def parse_version(text: str) -> tuple:
    m = _VERSION_RE.match(text or "")
    if not m:
        raise Refused(f"{text!r} is not a plain MAJOR.MINOR.PATCH release version")
    return tuple(int(g) for g in m.groups())


def crosses_major(old: tuple, new: tuple) -> bool:
    """True when *new* is a different major than *old*; for 0.x versions a
    different minor counts too (a 0.x minor change may break)."""
    if old[0] != new[0]:
        return True
    return old[0] == 0 and old[1] != new[1]


def normalised_sha256_b64(data: bytes) -> str:
    """sha256, base64, over the bytes with CRLF folded to LF."""
    lf = data.replace(b"\r\n", b"\n")
    return base64.b64encode(hashlib.sha256(lf).digest()).decode("ascii")


def _same_content(old: bytes, new: bytes, text: bool) -> bool:
    if old == new:
        return True
    return text and old.replace(b"\r\n", b"\n") == new.replace(b"\r\n", b"\n")


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
        "Accept": "application/json", "User-Agent": "localm-bump-vendored-js"})
    try:
        with opener(req, 30) as resp:
            raw = resp.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("response is larger than the allowed size")
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise Refused(f"could not read {url}: {type(e).__name__}: {e}") from e


def fetch_bytes(url: str, opener=None) -> bytes:
    """GET *url* and return its body. Raises Refused when it cannot be read or
    exceeds MAX_TARBALL_BYTES."""
    opener = opener or _default_open
    req = urllib.request.Request(url, headers={"User-Agent": "localm-bump-vendored-js"})
    try:
        with opener(req, 120) as resp:
            raw = resp.read(MAX_TARBALL_BYTES + 1)
    except Exception as e:
        raise Refused(f"could not download {url}: {type(e).__name__}: {e}") from e
    if len(raw) > MAX_TARBALL_BYTES:
        raise Refused(f"{url} is larger than {MAX_TARBALL_BYTES} bytes")
    return raw


def release_url(npm: str, version: str) -> str:
    return f"{NPM_REGISTRY}/{urllib.parse.quote(npm, safe='@')}/{version}"


def fetch_release(npm: str, version: str, fetch_json_fn=fetch_json,
                  fetch_bytes_fn=fetch_bytes) -> tuple:
    """(tarball bytes, dist.shasum) for npm package *npm* at *version*, verified.

    Raises Refused unless the registry lists exactly this package and version,
    it is not deprecated, it carries a sha512 integrity, and the downloaded
    tarball matches that integrity and the dist.shasum."""
    body = fetch_json_fn(release_url(npm, version))
    if not isinstance(body, dict):
        raise Refused(f"the registry answer for {npm}@{version} is not an object")
    if body.get("name") != npm or body.get("version") != version:
        raise Refused(f"the registry answered with {body.get('name')!r} "
                      f"{body.get('version')!r}, not {npm}@{version}")
    if body.get("deprecated"):
        raise Refused(f"{npm}@{version} is deprecated upstream: {body.get('deprecated')}")
    dist = body.get("dist")
    if not isinstance(dist, dict):
        raise Refused(f"{npm}@{version} carries no dist block")
    integrity = dist.get("integrity")
    sha512 = None
    for token in str(integrity or "").split():
        if token.startswith("sha512-"):
            sha512 = token[len("sha512-"):]
    if not sha512:
        raise Refused(f"{npm}@{version} carries no sha512 dist.integrity ({integrity!r})")
    shasum = dist.get("shasum")
    if not isinstance(shasum, str) or not re.fullmatch(r"[0-9a-f]{40}", shasum):
        raise Refused(f"{npm}@{version} carries no usable dist.shasum ({shasum!r})")
    tarball = dist.get("tarball")
    parsed = urllib.parse.urlsplit(str(tarball or ""))
    if parsed.scheme != "https" or parsed.hostname != urllib.parse.urlsplit(NPM_REGISTRY).hostname:
        raise Refused(f"{npm}@{version} tarball URL {tarball!r} is not on {NPM_REGISTRY}")
    data = fetch_bytes_fn(tarball)
    got = base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")
    if got != sha512:
        raise Refused(f"the {npm}@{version} tarball does not match its registry "
                      "integrity (sha512); nothing was edited")
    if hashlib.sha1(data).hexdigest() != shasum:
        raise Refused(f"the {npm}@{version} tarball does not match its registry shasum")
    return data, shasum


def read_tarball(data: bytes, wanted: tuple, font_prefix: str | None,
                 required: tuple = None) -> tuple:
    """({member name: bytes} for each wanted member present, {font file: bytes}).

    Raises Refused when a *required* member (default: every wanted one) is
    missing, a wanted member is duplicated or not a regular file, or a font
    member is not a plain woff2 file name. A missing member lists the
    minified or UMD files the tarball does carry."""
    required = wanted if required is None else required
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    except (tarfile.TarError, OSError) as e:
        raise Refused(f"the tarball cannot be opened: {e}") from e
    found: dict = {}
    fonts: dict = {}
    with tf:
        members = tf.getmembers()
        for member in members:
            name = member.name
            is_font = bool(font_prefix and name.startswith(font_prefix)
                           and name.endswith(".woff2"))
            if name not in wanted and not is_font:
                continue
            if not member.isfile():
                raise Refused(f"tarball member {name} is not a regular file")
            if member.size > MAX_MEMBER_BYTES:
                raise Refused(f"tarball member {name} is larger than {MAX_MEMBER_BYTES} bytes")
            payload = tf.extractfile(member).read()
            if is_font:
                base = name[len(font_prefix):]
                if not _FONT_NAME_RE.match(base):
                    raise Refused(f"unexpected font member {name}")
                if base in fonts:
                    raise Refused(f"font member {name} appears more than once")
                fonts[base] = payload
            else:
                if name in found:
                    raise Refused(f"tarball member {name} appears more than once")
                found[name] = payload
        missing = [m for m in required if m not in found]
        if missing:
            offered = sorted(m.name for m in members
                             if m.isfile() and re.search(r"\.(min|umd)\.[cm]?js$", m.name))
            raise Refused(f"the tarball has no {', '.join(missing)}; the minified or UMD "
                          f"files it does carry: {', '.join(offered) or 'none'}. This release "
                          "needs a code update to choose which file index.html loads")
    return found, fonts


# --------------------------------------------------------------------------- #
#  Reading the tree                                                           #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> tuple:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    return data.replace("\r\n", "\n"), newline


def _replace_once(pattern: re.Pattern, text: str, repl, what: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    m = matches[0]
    return text[:m.start()] + repl(m) + text[m.end():]


def read_state(lib: Lib, repo: Path) -> tuple:
    """(vendored files {dest: bytes}, test text, test newline, vendored version).

    Raises Refused when a pinned vendored file is missing or the version
    sources disagree."""
    vendor = repo / VENDOR_REL
    vendor_files = {}
    for member, dest in lib.files + lib.extra_files:
        path = vendor / dest
        if path.is_file():
            vendor_files[dest] = path.read_bytes()
        elif (member, dest) in lib.files:
            raise Refused(f"vendored file {VENDOR_REL}/{dest} is missing")
    test_text, test_nl = _read(repo / TESTS_REL / lib.test)
    return vendor_files, test_text, test_nl, current_version(lib, vendor_files, test_text)


def current_version(lib: Lib, vendor_files: dict, test_text: str) -> str:
    """The vendored version: the test pin when the library has one, else the
    file banner; the two must agree wherever both exist."""
    banner = None
    if lib.banner_re:
        head = vendor_files[lib.banner_file].decode("latin1")
        head = head if lib.key == "KaTeX" else head[:400]
        m = re.search(lib.banner_re, head)
        banner = m.group(1) if m else None
    pin = None
    if lib.pinned:
        matches = _VERSION_CONST_RE.findall(test_text)
        if len(matches) != 1:
            raise Refused(f"{lib.test}: VENDORED_VERSION expected exactly once, "
                          f"found {len(matches)}")
        pin = matches[0][1]
    if pin is None and banner is None:
        raise Refused(f"cannot read the vendored {lib.key} version")
    if pin is not None and banner is not None and pin != banner:
        raise Refused(f"{lib.test} pins {lib.key} {pin} but the vendored file says "
                      f"{banner}; the tree is inconsistent, fix it before bumping")
    version = pin or banner
    parse_version(version)
    return version


def _pin_hashes(lib: Lib, test_text: str) -> dict:
    """{vendor file: recorded normalised hash} read from the test file."""
    if lib.key == "KaTeX":
        out = {}
        for _, dest in lib.files:
            ms = re.findall(rf'^\s+"{re.escape(dest)}": "([^"]+)",', test_text, re.M)
            if len(ms) != 1:
                raise Refused(f"{lib.test}: the {dest} hash expected exactly once, "
                              f"found {len(ms)}")
            out[dest] = ms[0]
        return out
    ms = _HASH_CONST_RE.findall(test_text)
    if len(ms) != 1:
        raise Refused(f"{lib.test}: PINNED_HASH expected exactly once, found {len(ms)}")
    return {lib.files[0][1]: ms[0][1]}


# --------------------------------------------------------------------------- #
#  Rewrites (pure data -> data)                                               #
# --------------------------------------------------------------------------- #

def rewrite_test(lib: Lib, text: str, old: str, new: str, new_hashes: dict,
                 shasum: str) -> str:
    """The new text of the library's tests-js pin file (LF newlines)."""
    if not lib.pinned:
        return text
    text = _replace_once(_VERSION_CONST_RE, text,
                         lambda m: m.group(1) + new + m.group(3), "VENDORED_VERSION")
    if lib.key == "KaTeX":
        for dest, digest in new_hashes.items():
            pat = re.compile(rf'^(\s+"{re.escape(dest)}": ")([^"]+)(",)', re.M)
            text = _replace_once(pat, text, lambda m, d=digest: m.group(1) + d + m.group(3),
                                 f"{dest} hash")
    else:
        digest = new_hashes[lib.files[0][1]]
        text = _replace_once(_HASH_CONST_RE, text,
                             lambda m: m.group(1) + digest + m.group(3), "PINNED_HASH")
    pack = re.compile(r"(npm pack " + re.escape(lib.npm) + r"@)(\d+\.\d+\.\d+)")
    text = _replace_once(pack, text, lambda m: m.group(1) + new, "npm pack comment")
    if lib.shasum_comment:
        text = _replace_once(_SHASUM_RE, text,
                             lambda m: m.group(1) + shasum + m.group(3), "dist.shasum comment")
    if lib.title_re:
        title = re.compile(lib.title_re)
        text = _replace_once(title, text,
                             lambda m: m.group(1) + new + m.group(3), "banner test title")
    return text


def check_font_set(css: bytes, fonts: dict) -> None:
    """Raises Refused when the CSS references a woff2 font the release lacks."""
    referenced = set(_CSS_FONT_RE.findall(css.decode("utf-8", "replace")))
    missing = sorted(referenced - set(fonts))
    if missing:
        raise Refused("the new katex.min.css references woff2 font(s) the release does "
                      f"not ship: {', '.join(missing)}")
    if not fonts:
        raise Refused("the release ships no woff2 fonts")


def build_plan(lib: Lib, tag: str, tar_files: dict, tar_fonts: dict, shasum: str,
               repo: Path = None) -> dict:
    """The full edit plan: {"current", "new", "changes", "major", "mentions"}.

    Raises Refused on any inconsistency; reads the tree, writes nothing."""
    repo = repo or REPO
    vendor = repo / VENDOR_REL
    test_path = repo / TESTS_REL / lib.test
    new = tag
    new_v = parse_version(new)

    vendor_files, test_text, test_nl, old = read_state(lib, repo)
    old_v = parse_version(old)
    if new_v <= old_v:
        raise Refused(f"{lib.key} {new} is not newer than the vendored {old}; "
                      "this script only moves a library forward")

    if lib.pinned:
        recorded = _pin_hashes(lib, test_text)
        for dest, digest in recorded.items():
            if normalised_sha256_b64(vendor_files[dest]) != digest:
                raise Refused(f"vendored {dest} does not match the hash recorded in "
                              f"{lib.test}; the tree is inconsistent, fix it before bumping")

    new_bytes = {}
    for member, dest in lib.files:
        if member not in tar_files:
            raise Refused(f"the {lib.npm}@{new} tarball has no {member}; this release "
                          "needs a code update to choose which file index.html loads")
        new_bytes[dest] = tar_files[member]
    for member, dest in lib.extra_files:
        if member in tar_files:
            new_bytes[dest] = tar_files[member]

    changes = []
    for dest, data in new_bytes.items():
        old_data = vendor_files.get(dest)
        text = dest.endswith(_TEXT_SUFFIXES)
        if old_data is not None and _same_content(old_data, data, text):
            continue
        changes.append(Change(vendor / dest, f"{VENDOR_REL}/{dest}", old_data, data))

    font_notes = None
    if lib.fonts:
        check_font_set(new_bytes["katex.min.css"], tar_fonts)
        font_dir = vendor / "fonts"
        existing = {p.name: p.read_bytes() for p in sorted(font_dir.glob("KaTeX_*.woff2"))} \
            if font_dir.is_dir() else {}
        counts = {"unchanged": 0, "replaced": 0, "added": 0, "removed": 0}
        for name in sorted(set(existing) | set(tar_fonts)):
            rel = f"{VENDOR_REL}/fonts/{name}"
            if name not in tar_fonts:
                counts["removed"] += 1
                changes.append(Change(font_dir / name, rel, existing[name], None))
            elif name not in existing:
                counts["added"] += 1
                changes.append(Change(font_dir / name, rel, None, tar_fonts[name]))
            elif existing[name] != tar_fonts[name]:
                counts["replaced"] += 1
                changes.append(Change(font_dir / name, rel, existing[name], tar_fonts[name]))
            else:
                counts["unchanged"] += 1
        font_notes = counts

    new_hashes = {dest: normalised_sha256_b64(new_bytes[dest])
                  for _, dest in lib.files}
    new_test = rewrite_test(lib, test_text, old, new, new_hashes, shasum)
    if new_test != test_text:
        changes.append(Change(test_path, f"{TESTS_REL}/{lib.test}",
                              test_text.encode("utf-8"), new_test.encode("utf-8"),
                              text=True, newline=test_nl))

    mentions = find_mentions(repo, old, new_test, test_path, lib)
    return {"current": old, "new": new, "changes": changes,
            "major": crosses_major(old_v, new_v), "fonts": font_notes,
            "mentions": mentions}


def find_mentions(repo: Path, old: str, new_test: str, test_path: Path, lib: Lib) -> list:
    """Lines that still name *old* after the rewrite, in the test file and the
    prose files, as ``path:line: text``."""
    out = []
    sources = [(f"{TESTS_REL}/{lib.test}", new_test)]
    for rel in NOTICE_RELS:
        path = repo / rel
        if path.is_file():
            sources.append((rel, path.read_bytes().decode("utf-8").replace("\r\n", "\n")))
    for rel, text in sources:
        for number, line in enumerate(text.split("\n"), 1):
            if re.search(rf"(?<![\d.]){re.escape(old)}(?![\d])", line):
                out.append(f"{rel}:{number}: {line.strip()[:110]}")
    return out


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #

def describe(change: Change) -> str:
    if change.new is None:
        return f"  remove  {change.rel}"
    verb = "create " if change.old is None else "replace"
    size = f"{len(change.new)} bytes"
    if change.old is not None:
        size = f"{len(change.old)} -> {len(change.new)} bytes"
    digest = hashlib.sha256(change.new).hexdigest()[:16]
    return f"  {verb} {change.rel}  ({size}, sha256 {digest})"


def checklist(lib: Lib, plan: dict) -> str:
    lines = ["REMAINING STEPS, not automated - each has its own check:"]
    n = 1
    if plan["major"]:
        lines += [
            f"  !! MAJOR BOUNDARY: {lib.key} {plan['current']} -> {plan['new']} changes the "
            "major version",
            "       (for 0.x a minor change counts). A person must read upstream's changelog",
            "       and release notes for every release in between before this merges.",
        ]
    lines += [
        f"  {n}. npm ci && npm test",
        f"       the jsdom suite; tests-js/{lib.test} and the render pipeline tests must pass",
    ]
    n += 1
    if lib.key == "KaTeX":
        lines += [f"  {n}. katex.min.js, katex.min.css, auto-render.min.js and fonts/ moved together; "
                  "the matched-set test",
                  "       in the KaTeX guard must stay green; look at the font counts above"]
        n += 1
    if lib.key == "highlight.js":
        lines += [f"  {n}. github-dark.min.css was compared with the release and "
                  "is listed above only if it differed"]
        n += 1
    if lib.key == "marked":
        lines += [f"  {n}. the marked guard pins bytes only; a new release can add a runtime "
                  "version, and the",
                  "       guard's 'no runtime version' test then fails on purpose: upgrade it "
                  "to a floor"]
        n += 1
    lines += [
        f"  {n}. re-check advisory status for {lib.key} {plan['new']} (OSV, GitHub advisory "
        "database); the pins",
        "       cannot say a newer release is safe",
    ]
    n += 1
    if plan["mentions"]:
        lines += [f"  {n}. lines still naming {plan['current']} (history or advisory prose; "
                  "edit only if now false):"]
        lines += [f"       {m}" for m in plan["mentions"]]
        n += 1
    lines += [
        f"  {n}. CHANGELOG.md, [Unreleased]: one user-facing bullet, the bundled {lib.key} "
        f"moved to {plan['new']}",
        f"  {n + 1}. python scripts/check_hygiene.py",
        f"  {n + 2}. load the GUI and look at rendered markdown, code, math and a sanitised "
        "message",
    ]
    return "\n".join(lines)


def _write_change(change: Change) -> None:
    if change.new is None:
        change.path.unlink()
        return
    change.path.parent.mkdir(parents=True, exist_ok=True)
    if change.text:
        text = change.new.decode("utf-8").replace("\n", change.newline)
        change.path.write_bytes(text.encode("utf-8"))
    else:
        change.path.write_bytes(change.new)


def main(argv=None, *, fetch_json_fn=fetch_json, fetch_bytes_fn=fetch_bytes) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", required=True, choices=sorted(LIBS),
                    help="the vendored library to bump")
    ap.add_argument("--tag", required=True, help="the npm release to vendor, e.g. 3.4.16")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a summary and diff, change nothing)")
    args = ap.parse_args(argv)

    lib = LIBS[args.lib]
    tag = args.tag.strip()
    try:
        parse_version(tag)
        font_prefix = "package/dist/fonts/" if lib.fonts else None
        wanted = tuple(m for m, _ in lib.files + lib.extra_files)
        required = tuple(m for m, _ in lib.files)
        old = _precheck(lib, tag)
        print(f"{lib.key}: vendored {old} -> {tag}; reading {lib.npm}@{tag} from the registry ...")
        data, shasum = fetch_release(lib.npm, tag, fetch_json_fn, fetch_bytes_fn)
        print(f"  tarball {len(data)} bytes, sha512 integrity and sha1 shasum verified")
        tar_files, tar_fonts = read_tarball(data, wanted, font_prefix, required)
        plan = build_plan(lib, tag, tar_files, tar_fonts, shasum)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    changes = plan["changes"]
    if plan["major"]:
        print(f"MAJOR BOUNDARY: {lib.key} {plan['current']} -> {plan['new']}")
    if not changes:
        print(f"nothing to change: the tree already carries {lib.key} {tag}")
    else:
        print(f"FILES EDITED ({len(changes)}):")
        for change in changes:
            print(describe(change))
        if plan["fonts"] is not None:
            print("  fonts: " + ", ".join(f"{v} {k}" for k, v in plan["fonts"].items()))
        for change in changes:
            if change.text:
                sys.stdout.writelines(difflib.unified_diff(
                    change.old.decode("utf-8").splitlines(keepends=True),
                    change.new.decode("utf-8").splitlines(keepends=True),
                    fromfile=f"a/{change.rel}", tofile=f"b/{change.rel}"))
        if args.write:
            for change in changes:
                _write_change(change)
            print(f"wrote {len(changes)} file(s)")
        else:
            print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(lib, plan))
    return 0


def _precheck(lib: Lib, tag: str) -> str:
    """The vendored version, after refusing a tag that is not strictly newer
    (before anything is downloaded)."""
    old = read_state(lib, REPO)[3]
    if parse_version(tag) <= parse_version(old):
        raise Refused(f"{lib.key} {tag} is not newer than the vendored {old}; "
                      "this script only moves a library forward")
    return old


if __name__ == "__main__":
    sys.exit(main())
