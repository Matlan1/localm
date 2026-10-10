#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform the mechanical half of advancing localm's pinned AMD ROCm llama.cpp build.

The amd-rocm runtime is a lemonade-sdk/llamacpp-rocm release (``_ROCM_TAG``,
bNNNN) plus the Windows CPU archive of the upstream ggml-org/llama.cpp release
built from the SAME commit (``_ROCM_CPU_TAG``, bNNNNN), whose SIMD CPU backends
replace the ROCm build's scalar one. A bump moves both tags and their checksum
tables together. This script rewrites them and refuses to write unless a receipt
from ``scripts/confirm_rocm_runtime.py --receipt`` shows the target tag passed.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched), all in localm/setup_llama/pins.py:

  DEFAULT_URL, DEFAULT_URL_SHA256   the gfx103X Windows asset of the target tag
  _ROCM_TAG                         the target tag
  _ROCM_CPU_TAG                     the upstream release built from the same commit
  _PINNED_FALLBACK_SHA256           the ROCm asset block (every asset of the
                                    target release) and the CPU archive entry,
                                    with the sha256 digests the GitHub API
                                    publishes for those exact assets

HOW THE CPU TAG IS FOUND: the lemonade release body names the llama.cpp commit
(``Llama.cpp Commit Hash``, an abbreviated hash). The upstream release whose
``target_commitish`` starts with that hash and whose tag ref points at the same
commit is the CPU tag. No match, or more than one, is a refusal.

WHAT IT CHECKS BEFORE WRITING:
  * the receipt is for exactly this tag, is not a ``--current`` receipt, has verdict
    PASS, and every required check in it is PASS (the verdict field alone is not
    trusted);
  * the tag is strictly newer than the pin (forward only);
  * the live release still carries the digests, CPU tag and commit the receipt
    recorded;
  * the target release carries every asset the current table has;
  * each edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the struct binding
check against the upstream source (scripts/check_llama_abi.py), the targeted
tests, docs/llamacpp-binding.md when the struct layout generation changed, and
the changelog bullet.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when refused.

Environment: GITHUB_TOKEN is sent as a bearer token for the release lookups.

Usage:
    python scripts/bump_rocm_pin.py --tag b1350 --receipt confirm.json            # dry run
    python scripts/bump_rocm_pin.py --tag b1350 --receipt confirm.json --write

Needs localm importable for the verified HTTPS opener. Nothing under localm/
imports this; it never runs from a user's install.
"""

import argparse
import datetime as _dt
import difflib
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PINS_PATH = REPO / "localm" / "setup_llama" / "pins.py"

COMPONENT = "rocm"
RECEIPT_SCHEMA = 1
LEMONADE_REPO = "lemonade-sdk/llamacpp-rocm"
UPSTREAM_REPO = "ggml-org/llama.cpp"
API_ROOT = "https://api.github.com/"

# The checks scripts/confirm_rocm_runtime.py records. The bump requires every one
# to be present and PASS; the confirm script names its checks from this tuple.
REQUIRED_CHECKS = ("isolation", "hardware", "candidate", "resolve", "install", "abi",
                   "ggml_identity", "gpu_device", "model", "gpu_generate",
                   "cpu_backend", "cpu_generate")

# Upstream releases are scanned newest first until one is published this long
# before the lemonade release, and never more than this many pages.
UPSTREAM_WINDOW = _dt.timedelta(days=45)
UPSTREAM_MAX_PAGES = 30
_PER_PAGE = 100

_TAG_RE = re.compile(r"^b(\d+)$")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHORT_SHA_RE = re.compile(r"^[0-9a-f]{5,40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_LINE_RE = re.compile(r"\*\*Llama\.cpp Commit Hash\*\*:[ \t]*([0-9a-fA-F]+)")
_ROCM_VERSION_RE = re.compile(r"\*\*ROCm Version\*\*:[ \t]*([^\s*]+)")
_GFX103X = "windows-rocm-gfx103X-x64.zip"


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


class UpstreamUnreadable(Refused):
    """An upstream lookup failed or returned a body that is not the expected
    shape: the evidence could not be read, which says nothing about the build."""


Fetcher = Callable[[str], Any]


def build_number(tag: str) -> Optional[int]:
    """The integer of a ``bNNNN`` tag, or None when *tag* is not that shape."""
    m = _TAG_RE.match((tag or "").strip())
    return int(m.group(1)) if m else None


def parse_date(value: Any) -> Optional[_dt.datetime]:
    """``2026-08-12T12:18:24Z`` as an aware UTC datetime; anything else None."""
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.UTC)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  Upstream                                                                    #
# --------------------------------------------------------------------------- #

def github_get(path: str) -> Any:
    """Parsed JSON of a GitHub API path (or absolute api.github.com URL).

    Raises UpstreamUnreadable on any transport, status or decoding failure."""
    url = path if path.startswith("https://") else API_ROOT + path.lstrip("/")
    if not url.startswith(API_ROOT):
        raise UpstreamUnreadable(f"refusing to fetch outside the GitHub API: {url}")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "localm-bump-rocm-pin"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        from localm.http_ssl import verified_urlopen
        with verified_urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", None)
        label = f"HTTP {code}" if code else f"{type(e).__name__}: {e}"
        raise UpstreamUnreadable(f"could not read {url}: {label}") from e


@dataclass(frozen=True)
class Candidate:
    """Everything the pins need to know about one lemonade release."""

    tag: str
    lemonade_commit: str
    rocm_version: str
    published_at: _dt.datetime
    rocm_assets: dict[str, str] = field(default_factory=dict)
    cpu_tag: str = ""
    cpu_commit: str = ""
    cpu_asset: str = ""
    cpu_sha256: str = ""

    @property
    def gfx103x_asset(self) -> str:
        return f"llama-{self.tag}-{_GFX103X}"

    @property
    def gfx103x_sha256(self) -> str:
        return self.rocm_assets[self.gfx103x_asset]

    def to_receipt(self) -> dict:
        """The JSON-able record a confirm receipt carries for this candidate."""
        return {"tag": self.tag, "lemonade_commit": self.lemonade_commit,
                "rocm_version": self.rocm_version,
                "published_at": self.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gfx103X_asset": self.gfx103x_asset, "gfx103X_sha256": self.gfx103x_sha256,
                "rocm_assets": dict(self.rocm_assets),
                "cpu_tag": self.cpu_tag, "cpu_commit": self.cpu_commit,
                "cpu_asset": self.cpu_asset, "cpu_sha256": self.cpu_sha256}


def _digest_of(asset: dict, what: str) -> str:
    digest = asset.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise Refused(f"{what} has no sha256 digest in the API listing")
    sha = digest.split("sha256:", 1)[1].strip().lower()
    if not _SHA256_RE.match(sha):
        raise Refused(f"{what} has a malformed digest {digest!r}")
    return sha


def parse_lemonade_release(tag: str, body: Any) -> Candidate:
    """The lemonade-side half of a Candidate (CPU fields empty) from the
    ``releases/tags/<tag>`` API body. Raises UpstreamUnreadable for a body that
    is not a release, Refused for a release that cannot be pinned."""
    if not isinstance(body, dict):
        raise UpstreamUnreadable(f"the {tag} release lookup returned something that is not an object")
    if body.get("tag_name") != tag:
        raise UpstreamUnreadable(f"the release lookup for {tag} returned tag {body.get('tag_name')!r}")
    if body.get("draft") or body.get("prerelease"):
        raise Refused(f"{LEMONADE_REPO} {tag} is a draft or prerelease")
    published = parse_date(body.get("published_at"))
    if published is None:
        raise Refused(f"{tag} has no readable published_at")
    text = body.get("body")
    text = text if isinstance(text, str) else ""
    commits = {c.lower() for c in _COMMIT_LINE_RE.findall(text)}
    if len(commits) != 1:
        raise Refused(f"the {tag} release notes name {len(commits)} llama.cpp commit hashes; "
                      "expected exactly one 'Llama.cpp Commit Hash' line")
    commit = next(iter(commits))
    if not _SHORT_SHA_RE.match(commit):
        raise Refused(f"the {tag} release notes carry an unusable commit hash {commit!r}")
    versions = _ROCM_VERSION_RE.findall(text)
    assets = body.get("assets")
    if not isinstance(assets, list):
        raise UpstreamUnreadable(f"the {tag} release carries no asset list")
    name_re = re.compile(rf"^llama-{re.escape(tag)}-(?:windows|ubuntu)-rocm-[A-Za-z0-9]+-x64\.zip$")
    digests: dict[str, str] = {}
    for a in assets:
        name = a.get("name") if isinstance(a, dict) else None
        if isinstance(name, str) and name_re.match(name):
            digests[name] = _digest_of(a, f"asset {name}")
    wanted = f"llama-{tag}-{_GFX103X}"
    if wanted not in digests:
        raise Refused(f"the {tag} release has no {wanted} asset")
    return Candidate(tag=tag, lemonade_commit=commit,
                     rocm_version=versions[0] if len(set(versions)) == 1 else "",
                     published_at=published, rocm_assets=digests)


def find_upstream_release(short: str, since: _dt.datetime, fetch: Fetcher) -> dict:
    """The one ggml-org release built from the commit starting with *short*.

    Scans the release list newest first until a release older than
    ``UPSTREAM_WINDOW`` before *since*. Raises Refused when none matches, more
    than one does, or the matching tag ref does not point at that commit."""
    matches: dict[str, dict] = {}
    pages = 0
    for page in range(1, UPSTREAM_MAX_PAGES + 1):
        releases = fetch(f"repos/{UPSTREAM_REPO}/releases?per_page={_PER_PAGE}&page={page}")
        if not isinstance(releases, list):
            raise UpstreamUnreadable("the upstream release list is not a list")
        if not releases:
            break
        pages += 1
        for rel in releases:
            if not isinstance(rel, dict):
                continue
            commit, tag = rel.get("target_commitish"), rel.get("tag_name")
            if (isinstance(commit, str) and _FULL_SHA_RE.match(commit)
                    and commit.startswith(short) and isinstance(tag, str)
                    and build_number(tag) is not None):
                matches[tag] = rel
        oldest = parse_date(releases[-1].get("published_at")) if isinstance(releases[-1], dict) else None
        if oldest is not None and oldest < since - UPSTREAM_WINDOW:
            break
    if not matches:
        raise Refused(f"no upstream {UPSTREAM_REPO} release was built from commit {short} "
                      f"(scanned {pages} page(s) of releases); the CPU archive cannot be paired")
    commits = {rel["target_commitish"] for rel in matches.values()}
    if len(matches) > 1 or len(commits) > 1:
        raise Refused(f"commit {short} matches more than one upstream release "
                      f"({', '.join(sorted(matches))}); refusing to guess")
    tag, rel = next(iter(matches.items()))
    commit = rel["target_commitish"]
    ref = fetch(f"repos/{UPSTREAM_REPO}/git/ref/tags/{tag}")
    obj = ref.get("object") if isinstance(ref, dict) else None
    if not isinstance(obj, dict):
        raise UpstreamUnreadable(f"the tag ref lookup for {tag} returned no object")
    if obj.get("type") == "tag" and isinstance(obj.get("url"), str):
        inner = fetch(obj["url"])
        obj = inner.get("object") if isinstance(inner, dict) else None
        if not isinstance(obj, dict):
            raise UpstreamUnreadable(f"the annotated tag {tag} could not be dereferenced")
    if obj.get("sha") != commit:
        raise Refused(f"upstream tag {tag} points at {obj.get('sha')!r}, not at the release's "
                      f"commit {commit}")
    return rel


def fetch_candidate(tag: str, fetch: Optional[Fetcher] = None) -> Candidate:
    """The full Candidate for lemonade release *tag*, with the CPU tag found from
    the llama.cpp commit its release notes name."""
    if build_number(tag) is None:
        raise Refused(f"{tag!r} is not a lemonade-sdk build tag (bNNNN)")
    fetch = fetch or github_get
    base = parse_lemonade_release(tag, fetch(f"repos/{LEMONADE_REPO}/releases/tags/{tag}"))
    rel = find_upstream_release(base.lemonade_commit, base.published_at, fetch)
    cpu_tag = rel["tag_name"]
    cpu_asset = f"llama-{cpu_tag}-bin-win-cpu-x64.zip"
    assets = rel.get("assets")
    entry = next((a for a in assets or [] if isinstance(a, dict) and a.get("name") == cpu_asset), None)
    if entry is None:
        raise Refused(f"upstream {cpu_tag} has no {cpu_asset} asset")
    return Candidate(tag=base.tag, lemonade_commit=base.lemonade_commit,
                     rocm_version=base.rocm_version, published_at=base.published_at,
                     rocm_assets=base.rocm_assets, cpu_tag=cpu_tag,
                     cpu_commit=rel["target_commitish"], cpu_asset=cpu_asset,
                     cpu_sha256=_digest_of(entry, f"asset {cpu_asset}"))


# --------------------------------------------------------------------------- #
#  pins.py: read and rewrite                                                   #
# --------------------------------------------------------------------------- #

_ROCM_TAG_RE = re.compile(r'^(_ROCM_TAG\s*=\s*")([^"]+)(")', re.M)
_CPU_TAG_RE = re.compile(r'^(_ROCM_CPU_TAG\s*=\s*")([^"]+)(")', re.M)
_URL_RE = re.compile(
    r'(?P<head>^DEFAULT_URL = \(\n[ \t]+"https://github\.com/lemonade-sdk/llamacpp-rocm/releases/download/"\n[ \t]+")'
    r'(?P<dir>b\d+)/llama-(?P<file>b\d+)(?P<tail>-windows-rocm-gfx103X-x64\.zip"\n\))', re.M)
_URL_SHA_RE = re.compile(r'(?P<head>^DEFAULT_URL_SHA256 = \(\n[ \t]+")(?P<sha>[0-9a-f]{64})(?P<tail>"\n\))', re.M)
_ROCM_BLOCK_RE = re.compile(
    r'(?P<indent>[ \t]+)# tag (?P<tag>b\d+) ROCm assets \((?P<info>[^\n]*)\)\n'
    r'(?P<entries>(?:[ \t]+"llama-b\d+-(?:windows|ubuntu)-rocm-[A-Za-z0-9]+-x64\.zip": "[0-9a-f]{64}",\n)+)')
_CPU_BLOCK_RE = re.compile(
    r'(?P<indent>[ \t]+)# tag (?P<tag>b\d+) upstream Windows CPU archive \(_ROCM_CPU_ASSET\), the amd-rocm\n'
    r'(?P<comment>(?:[ \t]+#[^\n]*\n)*)'
    r'(?P<entries>[ \t]+"llama-b\d+-bin-win-cpu-x64\.zip": "[0-9a-f]{64}",\n)')
_ENTRY_RE = re.compile(r'"([^"\n]+)": "([0-9a-f]{64})"')


@dataclass(frozen=True)
class PinsView:
    """The ROCm pin values as they stand in a pins.py text."""

    tag: str
    cpu_tag: str
    url_tag: str
    url_file_tag: str
    url_sha256: str
    rocm_block_tag: str
    rocm_table: dict[str, str]
    cpu_block_tag: str
    cpu_table: dict[str, str]


def _only(pattern: re.Pattern, text: str, what: str) -> re.Match:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    return matches[0]


def read_pins(text: str) -> PinsView:
    """The ROCm pin values in *text*. Raises Refused when a region is missing or
    appears more than once."""
    url = _only(_URL_RE, text, "DEFAULT_URL")
    rocm = _only(_ROCM_BLOCK_RE, text, "ROCm asset block of _PINNED_FALLBACK_SHA256")
    cpu = _only(_CPU_BLOCK_RE, text, "CPU archive entry of _PINNED_FALLBACK_SHA256")
    return PinsView(
        tag=_only(_ROCM_TAG_RE, text, "_ROCM_TAG").group(2),
        cpu_tag=_only(_CPU_TAG_RE, text, "_ROCM_CPU_TAG").group(2),
        url_tag=url.group("dir"), url_file_tag=url.group("file"),
        url_sha256=_only(_URL_SHA_RE, text, "DEFAULT_URL_SHA256").group("sha"),
        rocm_block_tag=rocm.group("tag"), rocm_table=dict(_ENTRY_RE.findall(rocm.group("entries"))),
        cpu_block_tag=cpu.group("tag"), cpu_table=dict(_ENTRY_RE.findall(cpu.group("entries"))))


def pins_problems(view: PinsView, cand: Candidate) -> list[str]:
    """How the pins text disagrees with *cand*; empty when it matches exactly."""
    problems = []
    if view.tag != cand.tag:
        problems.append(f"_ROCM_TAG is {view.tag}, not {cand.tag}")
    if view.url_tag != cand.tag or view.url_file_tag != cand.tag:
        problems.append(f"DEFAULT_URL names {view.url_tag}/{view.url_file_tag}, not {cand.tag}")
    if view.cpu_tag != cand.cpu_tag:
        problems.append(f"_ROCM_CPU_TAG is {view.cpu_tag}, the release's commit is upstream {cand.cpu_tag}")
    if view.url_sha256 != cand.gfx103x_sha256:
        problems.append("DEFAULT_URL_SHA256 differs from the release's published digest")
    if view.rocm_block_tag != cand.tag:
        problems.append(f"the ROCm asset block is labelled {view.rocm_block_tag}, not {cand.tag}")
    if view.rocm_table != cand.rocm_assets:
        changed = sorted(set(view.rocm_table) ^ set(cand.rocm_assets)
                         | {n for n in set(view.rocm_table) & set(cand.rocm_assets)
                            if view.rocm_table[n] != cand.rocm_assets[n]})
        problems.append(f"the ROCm asset table differs from the release in: {', '.join(changed)}")
    if view.cpu_block_tag != cand.cpu_tag:
        problems.append(f"the CPU archive entry is labelled {view.cpu_block_tag}, not {cand.cpu_tag}")
    if view.cpu_table != {cand.cpu_asset: cand.cpu_sha256}:
        problems.append("the CPU archive entry differs from the upstream release")
    return problems


def check_asset_set(view: PinsView, cand: Candidate) -> None:
    """Raises Refused when *cand* lacks an asset the current table carries
    (matched by name with the tag removed)."""
    old = {n.replace(f"llama-{view.rocm_block_tag}-", "", 1) for n in view.rocm_table}
    new = {n.replace(f"llama-{cand.tag}-", "", 1) for n in cand.rocm_assets}
    missing = sorted(old - new)
    if missing:
        raise Refused(f"{cand.tag} no longer publishes: {', '.join(missing)}; a person has to "
                      "decide what happens to those hardware families")


def _rocm_order(name: str) -> tuple:
    return (0 if "-windows-" in name else 1, name)


def rewrite_pins(text: str, cand: Candidate) -> str:
    """*text* (LF newlines) with every ROCm pin region moved to *cand*."""
    def swap(pattern, what, fn):
        nonlocal text
        m = _only(pattern, text, what)
        text = text[:m.start()] + fn(m) + text[m.end():]

    swap(_ROCM_TAG_RE, "_ROCM_TAG", lambda m: m.group(1) + cand.tag + m.group(3))
    swap(_CPU_TAG_RE, "_ROCM_CPU_TAG", lambda m: m.group(1) + cand.cpu_tag + m.group(3))
    swap(_URL_RE, "DEFAULT_URL", lambda m: m.group("head") + cand.tag + "/llama-" + cand.tag + m.group("tail"))
    swap(_URL_SHA_RE, "DEFAULT_URL_SHA256", lambda m: m.group("head") + cand.gfx103x_sha256 + m.group("tail"))

    def rocm_block(m):
        indent = m.group("indent")
        info = f"llama.cpp {cand.cpu_commit[:12]}"
        if cand.rocm_version:
            info += f", ROCm {cand.rocm_version}"
        entries = "".join(f'{indent}"{n}": "{cand.rocm_assets[n]}",\n'
                          for n in sorted(cand.rocm_assets, key=_rocm_order))
        return f"{indent}# tag {cand.tag} ROCm assets ({info})\n{entries}"

    def cpu_block(m):
        indent = m.group("indent")
        return (f"{indent}# tag {cand.cpu_tag} upstream Windows CPU archive (_ROCM_CPU_ASSET), the amd-rocm\n"
                f"{m.group('comment')}{indent}\"{cand.cpu_asset}\": \"{cand.cpu_sha256}\",\n")

    swap(_ROCM_BLOCK_RE, "ROCm asset block of _PINNED_FALLBACK_SHA256", rocm_block)
    swap(_CPU_BLOCK_RE, "CPU archive entry of _PINNED_FALLBACK_SHA256", cpu_block)
    leftovers = pins_problems(read_pins(text), cand)
    if leftovers:
        raise Refused("the rewritten pins do not match the release: " + "; ".join(leftovers))
    return text


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def load_receipt(path: Path, tag: str) -> dict:
    """The receipt at *path*, validated as a PASS for exactly *tag*.

    Raises Refused when it is unreadable, for another component, tag or schema,
    a ``--current`` receipt, not PASS, or any required check is missing, not
    required, or not PASS. The verdict field is never trusted on its own."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"could not read the receipt {path}: {e}") from e
    if not isinstance(receipt, dict):
        raise Refused("the receipt is not a JSON object")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("component") != COMPONENT:
        raise Refused(f"the receipt is not a schema {RECEIPT_SCHEMA} {COMPONENT!r} receipt")
    if receipt.get("tag") != tag:
        raise Refused(f"the receipt is for {receipt.get('tag')!r}, not {tag}; confirm the target tag itself")
    if receipt.get("current") is not False:
        raise Refused("the receipt confirms the build the repo pins today (--current), not a candidate")
    if receipt.get("verdict") != "PASS":
        raise Refused(f"the receipt verdict is {receipt.get('verdict')!r}: {receipt.get('why', '')}")
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        raise Refused("the receipt has no checks")
    for name in REQUIRED_CHECKS:
        entry = checks.get(name)
        if not isinstance(entry, dict):
            raise Refused(f"the receipt has no {name!r} check")
        if entry.get("required") is not True:
            raise Refused(f"the receipt does not mark the {name!r} check as required")
    bad = []
    for name, entry in checks.items():
        if not isinstance(entry, dict):
            bad.append(f"{name}: not an object")
        elif entry.get("status") != "PASS" and (entry.get("required") is True
                                               or entry.get("status") == "FAIL"):
            bad.append(f"{name}: {entry.get('status')}")
    if bad:
        raise Refused("the receipt does not confirm every check: " + "; ".join(sorted(bad)))
    if not isinstance(receipt.get("candidate"), dict):
        raise Refused("the receipt carries no candidate record")
    return receipt


def receipt_mismatches(recorded: dict, cand: Candidate) -> list[str]:
    """How the live release differs from what the confirm run recorded."""
    live = cand.to_receipt()
    return [f"{k}: confirmed {recorded.get(k)!r}, release now says {live[k]!r}"
            for k in ("tag", "lemonade_commit", "gfx103X_sha256", "rocm_assets",
                      "cpu_tag", "cpu_commit", "cpu_sha256")
            if recorded.get(k) != live[k]]


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def read_lf(path: Path) -> str:
    """The text of *path* with LF newlines."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _read(path: Path) -> tuple[str, str]:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    return data.replace("\r\n", "\n"), "\r\n" if "\r\n" in data else "\n"


def _write(path: Path, text: str, newline: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.replace("\n", newline).encode("utf-8"))
    os.replace(tmp, path)


def checklist(cand: Candidate) -> str:
    return "\n".join([
        "REMAINING STEPS, not automated - each has its own check:",
        f"  1. python scripts/check_llama_abi.py --ref {cand.cpu_tag}",
        "       must pass; if it fails, update _structs.py/_abi.py first, then rerun this script",
        "  2. pytest tests/test_llama_pin_constant_and_currency.py tests/test_llama_rocm_pin_currency.py",
        "       tests/test_setup_llama_backends.py tests/test_setup_llama_contract.py",
        "       tests/test_llamacpp_abi.py tests/test_check_llama_abi.py",
        "       tests/test_bump_rocm_pin.py tests/test_confirm_rocm_runtime.py",
        "  3. docs/llamacpp-binding.md names the bundled build and its struct layouts:",
        "       update it when the layout generation changed",
        "  4. CHANGELOG.md, [Unreleased]: one bullet, the bundled AMD (ROCm) runtime moved",
        "  5. python scripts/check_llama_rocm_pin.py --gate",
        "       must report the pin as current",
    ])


def main(argv=None, *, fetch: Optional[Fetcher] = None, pins_path: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the lemonade-sdk release to pin, e.g. b1350")
    ap.add_argument("--receipt", default=None,
                    help="JSON written by scripts/confirm_rocm_runtime.py --tag <tag> --receipt")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    tag = args.tag.strip()
    path = pins_path or PINS_PATH
    try:
        if build_number(tag) is None:
            raise Refused(f"{tag!r} is not a lemonade-sdk build tag (bNNNN)")
        if not args.receipt:
            raise Refused("--receipt is required: a bump without the confirm is the untested-build "
                          "problem the pin exists to remove")
        text, newline = _read(path)
        view = read_pins(text)
        if build_number(view.tag) is None or build_number(tag) <= build_number(view.tag):
            raise Refused(f"{tag} is not newer than the pinned {view.tag}; the pin only moves forward")
        receipt = load_receipt(Path(args.receipt), tag)
        print(f"receipt: {tag} confirmed, every required check PASS")

        print(f"reading the {tag} release and its llama.cpp commit ...")
        cand = fetch_candidate(tag, fetch)
        print(f"  llama.cpp {cand.lemonade_commit} is upstream {cand.cpu_tag}; "
              f"{len(cand.rocm_assets)} ROCm assets with digests")
        drift = receipt_mismatches(receipt["candidate"], cand)
        if drift:
            raise Refused("the release changed after it was confirmed: " + "; ".join(drift))
        check_asset_set(view, cand)
        new_text = rewrite_pins(text, cand)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    rel = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name
    if new_text == text:
        print(f"nothing to change: the tree already pins {tag} with matching digests")
    elif args.write:
        _write(path, new_text, newline)
        print(f"wrote {rel}")
    else:
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(cand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
