#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One currency report for every external runtime, binary and library localm pins.

Rows: the four existing gates (llama.cpp, the lemonade ROCm build, ComfyUI, the
AMD ROCm wheels) run as subprocesses, plus a gate for each pin that had none:
koboldcpp, stable-diffusion.cpp, the ComfyUI-GGUF node, uv (installers and the
Docker image must agree), the Linux CUDA build source, the ROCm CPU archive, the
Linux CUDA runtime wheels, and the vendored browser libraries.

A pin is STALE once a newer upstream version has been available for longer than
its tolerance. It reads upstream only (GitHub API, PyPI, npm): no clone, no
download of a build.

Exit codes with --gate: 0 nothing stale, 1 at least one row STALE or INCONSISTENT,
2 nothing stale but at least one row could not be checked. 3 (with or without --gate)
means the command line named no known pin. Without --gate the report is otherwise
printed and the exit code is 0.

Usage:
    python scripts/check_pins.py
    python scripts/check_pins.py --gate
    python scripts/check_pins.py --gate --new-only --json out.json
    python scripts/check_pins.py --gate --exclude "llama.cpp,ComfyUI"
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CURRENT = "CURRENT"
BEHIND = "BEHIND"
STALE = "STALE"
INCONSISTENT = "INCONSISTENT"
UNKNOWN = "UNKNOWN"
UNVERSIONED = "UNVERSIONED"
BLOCKED = "BLOCKED"

EXIT_OK = 0
EXIT_STALE = 1
EXIT_UNKNOWN = 2
EXIT_USAGE = 3

DEFAULT_TOLERANCE_DAYS = 30
SECURITY_TOLERANCE_DAYS = 60
SLOW_TOLERANCE_DAYS = 180

_TIMEOUT = 20


class FetchError(Exception):
    """An upstream lookup failed; the row reads UNKNOWN, never CURRENT. *status* is the
    HTTP status of a failed request, when there was one."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LocalReadError(FetchError):
    """A pin constant in this repository could not be read; the row reads INCONSISTENT."""


@dataclass
class Row:
    name: str
    status: str
    pinned: str = ""
    latest: str = ""
    days: int | None = None
    tolerance: int | None = None
    detail: str = ""
    advancer: str = ""
    group: str = ""


@dataclass
class PinSpec:
    name: str
    group: str
    advancer: str
    check: Callable[[_dt.datetime], Row]
    legacy: bool = False
    compat: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Network                                                                    #
# --------------------------------------------------------------------------- #

def _headers(url: str = "") -> dict:
    """Request headers for *url*. The GitHub token is sent to api.github.com only."""
    headers = {"User-Agent": "localm-check-pins", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and urllib.parse.urlsplit(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    return headers


_RETRY_DELAYS = (2.0, 6.0)


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500 or error.code == 429
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def _urlopen_retry(req, *, read: bool = False):
    """urlopen *req* and return the response body bytes when *read*, else the response
    headers. A server error or a dropped connection is retried after a short delay."""
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310 - fixed https:// URLs
                return r.read() if read else r.headers
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt >= len(_RETRY_DELAYS) or not _is_transient(e):
                raise
            time.sleep(_RETRY_DELAYS[attempt])
    raise AssertionError("unreachable")


def _get_json(url: str):
    """GET *url* as JSON. Raises FetchError on any failure, including a body that
    is not JSON. Tests replace this function; nothing else opens a socket."""
    req = urllib.request.Request(url, headers=_headers(url))
    try:
        return json.loads(_urlopen_retry(req, read=True).decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise FetchError(f"{url}: {e}", status=getattr(e, "code", None)) from e


def _parse_date(value) -> _dt.datetime | None:
    """'2026-08-12T12:18:24Z' (fractional seconds and a zone suffix are ignored)
    -> aware UTC datetime; anything else -> None."""
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_dt.UTC)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  Version handling                                                           #
# --------------------------------------------------------------------------- #

_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _semver_key(tag) -> tuple[int, ...] | None:
    """'v1.122.1' / '3.4.13' / '12.9.2.10' / '1.22.0-dev.2025' -> a 4-tuple padded with
    zeros, e.g. (1, 122, 1, 0); anything else None."""
    if not isinstance(tag, str):
        return None
    m = _SEMVER_RE.search(tag.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.groups(default="0"))


_SDCPP_RE = re.compile(r"^master-(\d+)-[0-9a-f]+$")


def _sdcpp_key(tag) -> tuple[int, ...] | None:
    if not isinstance(tag, str):
        return None
    m = _SDCPP_RE.match(tag.strip())
    return (int(m.group(1)),) if m else None


def _read_text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _read_const(rel: str, pattern: str) -> str:
    """The first capture group of *pattern* in *rel*; raises FetchError when the
    file or the constant is missing (a renamed constant must not read as current)."""
    try:
        text = _read_text(rel)
    except OSError as e:
        raise LocalReadError(f"cannot read {rel}: {e}") from e
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise LocalReadError(f"{rel}: pattern {pattern!r} not found (renamed or reformatted?)")
    return m.group(1)


# --------------------------------------------------------------------------- #
#  Upstream lookups                                                           #
# --------------------------------------------------------------------------- #

def _github_releases(repo: str) -> list[tuple[str, _dt.datetime]]:
    """Published, non-draft, non-prerelease releases of *repo* as (tag, date)."""
    data = _get_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    if not isinstance(data, list):
        raise FetchError(f"{repo}: unexpected releases response shape")
    out = []
    for rel in data:
        if not isinstance(rel, dict) or rel.get("draft") or rel.get("prerelease"):
            continue
        date = _parse_date(rel.get("published_at"))
        tag = rel.get("tag_name")
        if date is not None and isinstance(tag, str):
            out.append((tag, date))
    return out


def _npm_releases(package: str) -> list[tuple[str, _dt.datetime]]:
    data = _get_json(f"https://registry.npmjs.org/{package}")
    if not isinstance(data, dict) or not isinstance(data.get("time"), dict):
        raise FetchError(f"{package}: unexpected npm response shape")
    out = []
    for version, stamp in data["time"].items():
        if version in ("created", "modified") or "-" in version:
            continue
        date = _parse_date((stamp or "")[:19] + "Z") if isinstance(stamp, str) else None
        if date is not None:
            out.append((version, date))
    return out


def _pypi_latest(package: str) -> tuple[str, _dt.datetime | None]:
    data = _get_json(f"https://pypi.org/pypi/{package}/json")
    try:
        version = data["info"]["version"]
    except (KeyError, TypeError) as e:
        raise FetchError(f"{package}: unexpected PyPI response shape") from e
    stamp = None
    for f in (data.get("releases") or {}).get(version, []) if isinstance(data, dict) else []:
        stamp = _parse_date(str(f.get("upload_time_iso_8601", ""))[:19] + "Z")
        if stamp:
            break
    return version, stamp


def _assess_releases(name: str, group: str, advancer: str, pinned: str,
                     releases: list[tuple[str, _dt.datetime]],
                     key: Callable[[str], tuple[int, ...] | None],
                     tolerance: int, now: _dt.datetime, *, detail: str = "") -> Row:
    """Compare *pinned* with *releases*. days = how long the OLDEST release newer
    than the pin has been available (a lower bound when the listing ends before the
    pin). STALE past *tolerance*, BEHIND within it, CURRENT when nothing is newer."""
    pin_key = key(pinned)
    row = Row(name=name, status=UNKNOWN, pinned=pinned, tolerance=tolerance,
              advancer=advancer, group=group)
    if pin_key is None:
        row.detail = f"cannot parse pinned version {pinned!r}"
        return row
    keyed = [(key(t), t, d) for t, d in releases]
    keyed = [(k, t, d) for k, t, d in keyed if k is not None]
    if not keyed:
        row.detail = "no usable upstream releases"
        return row
    newest = max(keyed, key=lambda x: x[0])
    row.latest = newest[1]
    newer = [(k, t, d) for k, t, d in keyed if k > pin_key]
    if not newer:
        row.status = CURRENT
        row.days = 0
        row.detail = detail
        return row
    first_available = min(d for _, _, d in newer)
    row.days = max((now - first_available).days, 0)
    row.status = STALE if row.days > tolerance else BEHIND
    row.detail = f"{len(newer)} newer release(s); newest {newest[1]}" + (f"; {detail}" if detail else "")
    return row


def _unknown(name: str, group: str, advancer: str, err: Exception, pinned: str = "") -> Row:
    """The row for a failed lookup. A pin this repository cannot read is INCONSISTENT (a
    renamed constant is a defect to fix); an upstream lookup that failed is UNKNOWN."""
    if isinstance(err, LocalReadError):
        return Row(name=name, status=INCONSISTENT, pinned=pinned, advancer=advancer, group=group,
                   detail=f"cannot read the pin: {err}")
    return Row(name=name, status=UNKNOWN, pinned=pinned, detail=str(err), advancer=advancer,
               group=group)


# --------------------------------------------------------------------------- #
#  New gates                                                                  #
# --------------------------------------------------------------------------- #

def _check_github_tag_pin(name, group, advancer, repo, pin_file, pin_pattern, key,
                          tolerance=DEFAULT_TOLERANCE_DAYS):
    def check(now):
        pinned = ""
        try:
            pinned = _read_const(pin_file, pin_pattern)
            releases = _github_releases(repo)
        except FetchError as e:
            return _unknown(name, group, advancer, e, pinned)
        return _assess_releases(name, group, advancer, pinned, releases, key, tolerance, now)
    return check


def _check_gguf_node(now):
    name, group, advancer = "ComfyUI-GGUF node", "comfyui", "pin_weekly.py (review PR)"
    pinned = ""
    try:
        pinned = _read_const("localm/media/managed_comfy_fresh.py",
                             r'name="ComfyUI-GGUF",\s*repo="[^"]+",\s*commit="([0-9a-f]{40})"')
        head = _get_json("https://api.github.com/repos/city96/ComfyUI-GGUF/commits/main")
        pin_commit = _get_json(f"https://api.github.com/repos/city96/ComfyUI-GGUF/commits/{pinned}")
        head_sha = head["sha"]
        head_date = _parse_date(head["commit"]["committer"]["date"])
        pin_date = _parse_date(pin_commit["commit"]["committer"]["date"])
    except (FetchError, KeyError, TypeError) as e:
        return _unknown(name, group, advancer, e, pinned[:12])
    if head_date is None or pin_date is None:
        return _unknown(name, group, advancer, FetchError("commit date unreadable"), pinned[:12])
    row = Row(name=name, status=CURRENT, pinned=pinned[:12], latest=head_sha[:12],
              tolerance=SLOW_TOLERANCE_DAYS, advancer=advancer, group=group, days=0)
    if head_sha != pinned:
        row.days = max((head_date - pin_date).days, 0)
        row.status = STALE if row.days > SLOW_TOLERANCE_DAYS else BEHIND
        row.detail = f"main is {row.days} day(s) of history past the pinned commit"
    return row


_UV_SITES = (
    ("setup.sh", r'^UV_INSTALLER_VERSION="([^"]+)"'),
    ("setup-gui.sh", r'^UV_INSTALLER_VERSION="([^"]+)"'),
    ("setup.bat", r'^set "UV_INSTALLER_VERSION=([^"]+)"'),
    ("setup-gui.bat", r'^set "UV_INSTALLER_VERSION=([^"]+)"'),
    ("docker/Dockerfile", r'^ARG UV_VERSION=(\S+)'),
)


def _check_uv(now):
    name, group, advancer = "uv (installers + Docker)", "tooling", "pin_weekly.py"
    try:
        found = {rel: _read_const(rel, pat) for rel, pat in _UV_SITES}
        releases = _github_releases("astral-sh/uv")
    except FetchError as e:
        return _unknown(name, group, advancer, e)
    versions = sorted(set(found.values()), key=lambda v: _semver_key(v) or (0,))
    oldest = versions[0]
    row = _assess_releases(name, group, advancer, oldest, releases, _semver_key,
                           DEFAULT_TOLERANCE_DAYS, now)
    if len(versions) > 1:
        row.status = INCONSISTENT
        row.detail = "pins disagree: " + ", ".join(f"{k}={v}" for k, v in found.items())
        row.pinned = "/".join(versions)
    return row


def _check_cuda_linux_source(now):
    name, group = "Linux CUDA build source", "cuda"
    advancer = "rides the llama.cpp pin"
    pinned = ""
    try:
        pinned = _read_const("localm/setup_llama/pins.py", r'^_PINNED_TAG = "([^"]+)"')
        repo = _read_const("localm/setup_llama/pins.py", r'^_CUDA_LINUX_REPO = "([^"]+)"')
        rel = _get_json(f"https://api.github.com/repos/{repo}/releases/tags/{pinned}")
    except FetchError as e:
        if e.status == 404 and not isinstance(e, LocalReadError):
            return Row(name=name, status=STALE, pinned=pinned, advancer=advancer, group=group,
                       detail="the CUDA build source has no release for the pinned llama.cpp tag; "
                              "Linux CUDA users silently fall back to vulkan")
        return _unknown(name, group, advancer, e, pinned)
    assets = [a.get("name", "") for a in rel.get("assets", []) if isinstance(a, dict)]
    cuda = [a for a in assets if "cuda" in a]
    if not cuda:
        return Row(name=name, status=STALE, pinned=pinned, advancer=advancer, group=group,
                   detail="the release for the pinned tag carries no CUDA asset")
    return Row(name=name, status=CURRENT, pinned=pinned, latest=pinned, days=0, advancer=advancer,
               group=group, detail=f"{len(cuda)} CUDA asset(s) for the pinned tag")


def _check_rocm_cpu_asset(now):
    name, group = "ROCm CPU archive", "rocm"
    advancer = "rides the ROCm pin"
    pinned = ""
    try:
        pinned = _read_const("localm/setup_llama/pins.py", r'^_ROCM_CPU_TAG = "([^"]+)"')
        repo = _read_const("localm/setup_llama/pins.py", r'^_UPSTREAM_REPO = "([^"]+)"')
        rel = _get_json(f"https://api.github.com/repos/{repo}/releases/tags/{pinned}")
    except FetchError as e:
        if e.status == 404 and not isinstance(e, LocalReadError):
            return Row(name=name, status=STALE, pinned=pinned, advancer=advancer, group=group,
                       detail="upstream no longer has a release for the pinned CPU tag")
        return _unknown(name, group, advancer, e, pinned)
    want = f"llama-{pinned}-bin-win-cpu-x64.zip"
    assets = [a.get("name") for a in rel.get("assets", []) if isinstance(a, dict)]
    if want not in assets:
        return Row(name=name, status=STALE, pinned=pinned, advancer=advancer, group=group,
                   detail=f"{want} is not among the release's assets")
    return Row(name=name, status=CURRENT, pinned=pinned, latest=pinned, days=0, advancer=advancer,
               group=group, detail="asset present; its tag must track the ROCm build's llama.cpp commit")


_MANIFEST_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def _registry_digest(repo: str, tag: str) -> str:
    """The digest Docker Hub's registry currently serves for library/<repo>:<tag>
    (the Docker-Content-Digest of its manifest list). Raises FetchError."""
    token_url = ("https://auth.docker.io/token?service=registry.docker.io"
                 f"&scope=repository:library/{repo}:pull")
    try:
        token = _get_json(token_url)["token"]
        req = urllib.request.Request(
            f"https://registry-1.docker.io/v2/library/{repo}/manifests/{tag}", method="HEAD",
            headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT,
                     "User-Agent": "localm-check-pins"})
        digest = _urlopen_retry(req).get("Docker-Content-Digest")
    except (KeyError, TypeError) as e:
        raise FetchError(f"{token_url}: unexpected token response ({e})") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise FetchError(f"registry-1.docker.io {repo}:{tag}: {e}") from e
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise FetchError(f"registry-1.docker.io {repo}:{tag}: no Docker-Content-Digest header")
    return digest


def _docker_library_updated(repo: str) -> _dt.datetime | None:
    """When docker-library/official-images last changed library/<repo>, i.e. when
    its tags were last republished."""
    data = _get_json("https://api.github.com/repos/docker-library/official-images/commits"
                     f"?path=library/{repo}&per_page=1")
    try:
        return _parse_date(data[0]["commit"]["committer"]["date"])
    except (KeyError, IndexError, TypeError) as e:
        raise FetchError("unexpected official-images commits response shape") from e


def _check_docker_base(now):
    name, group, advancer = "Docker base image", "tooling", "pin_weekly.py (review PR)"
    pinned = ""
    try:
        ref = _read_const("docker/Dockerfile",
                          r"^ARG UBUNTU_IMAGE=(ubuntu:[\d.]+@sha256:[0-9a-f]{64})$")
        image, _, pinned = ref.partition("@")
        repo, _, tag = image.partition(":")
        current = _registry_digest(repo, tag)
        updated = _docker_library_updated(repo) if current != pinned else None
    except FetchError as e:
        return _unknown(name, group, advancer, e, pinned[:19])
    row = Row(name=name, status=CURRENT, pinned=pinned[:19], latest=current[:19], days=0,
              tolerance=DEFAULT_TOLERANCE_DAYS, advancer=advancer, group=group)
    if current != pinned:
        if updated is None:
            return _unknown(name, group, advancer, FetchError("republish date unreadable"),
                            pinned[:19])
        row.days = max((now - updated).days, 0)
        row.status = STALE if row.days > DEFAULT_TOLERANCE_DAYS else BEHIND
        row.detail = f"{image} has been republished with a new digest"
    return row


_ASSET_TABLE_RE = re.compile(r"^_PINNED_FALLBACK_SHA256 = \{\n(.*?)^\}", re.MULTILINE | re.DOTALL)
_ASSET_ENTRY_RE = re.compile(r'"([^"]+\.(?:zip|tar\.gz))":\s*"([0-9a-f]{64})"')
_LEMONADE_REPO = "lemonade-sdk/llamacpp-rocm"


def _check_llama_asset_checksums(now):
    """Every sha256 pinned for a llama.cpp or ROCm build against the digest GitHub now
    publishes for that asset, so a re-uploaded asset or a mistyped checksum is seen."""
    name, group, advancer = "llama.cpp asset checksums", "llama", "pin_weekly.py (llama, rocm)"
    pins = "localm/setup_llama/pins.py"
    try:
        tag = _read_const(pins, r'^_PINNED_TAG = "([^"]+)"')
        rocm_tag = _read_const(pins, r'^_ROCM_TAG = "([^"]+)"')
        cpu_tag = _read_const(pins, r'^_ROCM_CPU_TAG = "([^"]+)"')
        upstream = _read_const(pins, r'^_UPSTREAM_REPO = "([^"]+)"')
        block = _ASSET_TABLE_RE.search(_read_text(pins))
        if not block:
            raise LocalReadError(f"{pins}: _PINNED_FALLBACK_SHA256 table not found")
        table = dict(_ASSET_ENTRY_RE.findall(block.group(1)))
        if not table:
            raise LocalReadError(f"{pins}: _PINNED_FALLBACK_SHA256 table is empty")
        sources = {(upstream, tag): {}, (upstream, cpu_tag): {}, (_LEMONADE_REPO, rocm_tag): {}}
        for asset, sha in table.items():
            if asset.startswith(f"llama-{rocm_tag}-"):
                sources[(_LEMONADE_REPO, rocm_tag)][asset] = sha
            elif asset.startswith(f"llama-{cpu_tag}-"):
                sources[(upstream, cpu_tag)][asset] = sha
            else:
                sources[(upstream, tag)][asset] = sha
        wrong, missing, checked = [], [], 0
        for (repo, release_tag), pinned in sources.items():
            if not pinned:
                continue
            release = _get_json(f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}")
            published = {a.get("name"): str(a.get("digest") or "").removeprefix("sha256:").lower()
                         for a in release.get("assets", []) if isinstance(a, dict)}
            for asset, sha in pinned.items():
                if asset not in published:
                    if release_tag in asset:
                        missing.append(asset)
                    continue
                checked += 1
                if published[asset] != sha:
                    wrong.append(asset)
    except FetchError as e:
        return _unknown(name, group, advancer, e)
    pinned_label = f"{tag}, {cpu_tag}, {rocm_tag}"
    if wrong or missing:
        return Row(name=name, status=STALE if wrong else INCONSISTENT, pinned=pinned_label,
                   advancer=advancer, group=group,
                   detail=(f"{len(wrong)} checksum(s) differ from the published digest: "
                           f"{', '.join(wrong[:3])}" if wrong else "")
                   + (f" {len(missing)} pinned asset(s) are not in the release: {', '.join(missing[:3])}"
                      if missing else ""))
    return Row(name=name, status=CURRENT, pinned=pinned_label, latest=pinned_label, days=0,
               advancer=advancer, group=group, detail=f"{checked} pinned checksum(s) match the published digests")


_CUDA_RUNTIME_RE = re.compile(r'^_CUDA_RUNTIME_PIN = \{(.*?)^\}', re.MULTILINE | re.DOTALL)


def _check_cuda_runtime_wheels(now):
    name, group = "Linux CUDA runtime wheels", "cuda"
    advancer = "pin_weekly.py (review PR)"
    try:
        text = _read_text("localm/setup_llama/cuda.py")
    except OSError as e:
        return _unknown(name, group, advancer, e)
    m = _CUDA_RUNTIME_RE.search(text)
    if not m:
        return Row(name=name, status=INCONSISTENT, advancer=advancer, group=group,
                   detail="the CUDA runtime wheels are not pinned: cuda.py takes PyPI's latest at "
                          "setup time, so what users get is not what was tested")
    pins = re.findall(r'"([a-z0-9._-]+)":\s*\(\s*"([^"]+)"', m.group(1))
    rows = []
    try:
        for package, version in pins:
            latest, stamp = _pypi_latest(package)
            rows.append((package, version, latest, stamp))
    except FetchError as e:
        return _unknown(name, group, advancer, e)
    behind = [(p, v, latest) for p, v, latest, _ in rows
              if (_semver_key(latest) or (0,)) > (_semver_key(v) or (0,))]
    row = Row(name=name, status=CURRENT, pinned=", ".join(f"{p}=={v}" for p, v, *_ in rows),
              latest=", ".join(f"{p}=={latest}" for p, _, latest, _ in rows), days=0,
              tolerance=DEFAULT_TOLERANCE_DAYS, advancer=advancer, group=group)
    if behind:
        behind_names = {p for p, _, _ in behind}
        stamps = [s for p, _, _, s in rows if p in behind_names and s]
        row.days = max((now - min(stamps)).days, 0) if stamps else None
        row.status = BEHIND if (row.days or 0) <= DEFAULT_TOLERANCE_DAYS else STALE
        row.detail = "newer: " + ", ".join(f"{p} {latest}" for p, _, latest in behind)
    return row


# (name, vendored file, version regex, release source, tolerance)
_VENDORED = (
    ("marked", "localm/plugins/gui/static/vendor/marked.min.js", r"marked v(\d+\.\d+\.\d+)",
     ("gh", "markedjs/marked"), SECURITY_TOLERANCE_DAYS),
    ("DOMPurify", "localm/plugins/gui/static/vendor/purify.min.js", r"DOMPurify (\d+\.\d+\.\d+)",
     ("gh", "cure53/DOMPurify"), SECURITY_TOLERANCE_DAYS),
    ("highlight.js", "localm/plugins/gui/static/vendor/highlight.min.js",
     r"Highlight\.js v(\d+\.\d+\.\d+)", ("gh", "highlightjs/highlight.js"), SECURITY_TOLERANCE_DAYS),
    ("KaTeX", "localm/plugins/gui/static/vendor/katex.min.js", r'version:"(\d+\.\d+\.\d+)"',
     ("gh", "KaTeX/KaTeX"), SECURITY_TOLERANCE_DAYS),
    ("kokoro-js", "localm/plugins/builtin/tts/static/vendor/NOTICE.md",
     r"\| `kokoro-js` \| ([0-9.]+) \|", ("npm", "kokoro-js"), SLOW_TOLERANCE_DAYS),
    ("transformers.js", "localm/plugins/builtin/tts/static/vendor/NOTICE.md",
     r"\| `@huggingface/transformers` \(transformers\.js\) \| ([0-9.]+) \|",
     ("gh", "huggingface/transformers.js"), SLOW_TOLERANCE_DAYS),
    ("onnxruntime-web", "localm/plugins/builtin/tts/static/vendor/NOTICE.md",
     r"\| `onnxruntime-web` \| ([0-9.]+)", ("gh", "microsoft/onnxruntime"), SLOW_TOLERANCE_DAYS),
)

_UNVERSIONED_VENDORED = (
    ("jsQR", "localm/plugins/gui/static/vendor/jsQR.js",
     "carries no version string; pinned by content hash in tests-js/vendor-jsqr.test.mjs"),
    ("Inter font", "localm/plugins/gui/static/vendor/inter/OFL.txt",
     "woff2 files carry no version string"),
)


_AUTO_VENDORED = ("marked", "DOMPurify", "highlight.js", "KaTeX")
_TTS_BUNDLE = ("kokoro-js", "transformers.js")


def _vendored_advancer(name: str) -> str:
    if name in _AUTO_VENDORED:
        return "pin_weekly.py (GUI tests; review for a major)"
    if name in _TTS_BUNDLE:
        return "pin_weekly.py (review PR; rebuilds the TTS bundle)"
    if name == "onnxruntime-web":
        return "rides the transformers.js pin"
    return "none"


def _check_vendored(spec):
    name, rel, pattern, (source, ref), tolerance = spec
    label = f"vendored {name}"

    def check(now):
        pinned = ""
        try:
            pinned = _read_const(rel, pattern)
            releases = _github_releases(ref) if source == "gh" else _npm_releases(ref)
        except FetchError as e:
            return _unknown(label, "vendored", _vendored_advancer(name), e, pinned)
        return _assess_releases(label, "vendored", _vendored_advancer(name), pinned, releases,
                                _semver_key, tolerance, now)
    return check


def _not_checkable(name, rel, why):
    def check(now):
        exists = (REPO / rel).is_file()
        return Row(name=name, status=UNVERSIONED if exists else INCONSISTENT, group="unchecked",
                   advancer="none", detail=why if exists else f"{rel} is missing; update this list")
    return check


def _unversioned(name, rel, why):
    def check(now):
        exists = (REPO / rel).is_file()
        return Row(name=f"vendored {name}", status=UNVERSIONED if exists else UNKNOWN,
                   advancer="none (hand-maintained)", group="vendored",
                   detail=why if exists else f"{rel} is missing")
    return check


# --------------------------------------------------------------------------- #
#  Existing gates, run as subprocesses                                        #
# --------------------------------------------------------------------------- #

_LEGACY = (
    ("llama.cpp", "llama", "scripts/check_llama_pin.py", "pin_weekly.py (llama)",
     r"localm pins llama\.cpp (\S+)", r"upstream newest with assets: (\S+)"),
    ("ROCm llama (lemonade)", "rocm", "scripts/check_llama_rocm_pin.py", "pin_weekly.py (rocm)",
     r"localm pins the lemonade-sdk ROCm build at (\S+)", r"upstream newest with assets: (\S+)"),
    ("ComfyUI", "comfyui", "scripts/check_comfyui_pin.py", "pin_weekly.py (comfyui)",
     r"ComfyUI bundled pin: (\S+)", r"upstream latest: (\S+)"),
    ("AMD ROCm wheels", "rocm", "scripts/check_amd_rocm_wheels_pin.py", "pin_weekly.py (review PR)",
     None, None),
)

_VERDICT_LINE_RE = re.compile(r"^(OK:|STALE|BEHIND|COULD NOT|within the|.*: STALE|.*: current)")


def _first_group(pattern, text) -> str:
    if not pattern:
        return ""
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else ""


def _legacy_check(name, group, script, advancer, pinned_re=None, latest_re=None):
    def check(now):
        cmd = [sys.executable, str(REPO / script), "--gate"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=REPO,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except (OSError, subprocess.TimeoutExpired) as e:
            return _unknown(name, group, advancer, e)
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        verdict_lines = [line for line in lines if _VERDICT_LINE_RE.match(line)]
        tail = " | ".join(verdict_lines or lines[-3:])[:300]
        status = {0: CURRENT, 1: STALE, 2: UNKNOWN}.get(proc.returncode, UNKNOWN)
        if proc.returncode == 0 and re.search(r"\bBEHIND\b", proc.stdout):
            status = BEHIND
        return Row(name=name, status=status, advancer=advancer, group=group,
                   pinned=_first_group(pinned_re, proc.stdout),
                   latest=_first_group(latest_re, proc.stdout), detail=tail)
    return check


# --------------------------------------------------------------------------- #
#  Pinned Hugging Face file digests                                           #
# --------------------------------------------------------------------------- #

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_PINS_NAME = "Hugging Face file digests"


def _hf_linked_sha256(repo: str, path: str, revision: str = "main") -> str | None:
    """The sha256 Hugging Face reports for *path* of *repo* at *revision* (the
    ``X-Linked-ETag`` of the unredirected resolve response), or None for a file
    that is not stored as LFS. Raises FetchError on any failure, with the HTTP
    status for 4xx/5xx. Tests replace this function; nothing else opens a socket."""
    url = (f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/"
           f"{urllib.parse.quote(path)}")
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "localm-check-pins"})
    opener = urllib.request.build_opener(_NoRedirect)
    headers = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            with opener.open(req, timeout=_TIMEOUT) as r:
                headers = r.headers
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                headers = e.headers
                break
            if attempt < len(_RETRY_DELAYS) and _is_transient(e):
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            raise FetchError(f"{url}: HTTP {e.code}", status=e.code) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < len(_RETRY_DELAYS) and _is_transient(e):
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            raise FetchError(f"{url}: {e}") from e
    etag = str((headers or {}).get("X-Linked-ETag") or "").strip().strip('"').lower()
    return etag if _HEX64_RE.match(etag) else None


def _py_value(rel: str, name: str) -> ast.expr:
    """The value expression of the module-level assignment to *name* in *rel*,
    read without importing the module (this script runs without localm's
    dependencies installed). Raises LocalReadError when it is missing."""
    try:
        tree = ast.parse(_read_text(rel))
    except (OSError, SyntaxError) as e:
        raise LocalReadError(f"cannot read {rel}: {e}") from e
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)                 and node.target.id == name and node.value is not None:
            return node.value
    raise LocalReadError(f"{rel}: {name} not found (renamed?)")


def _py_literal(rel: str, name: str):
    try:
        return ast.literal_eval(_py_value(rel, name))
    except ValueError as e:
        raise LocalReadError(f"{rel}: {name} is not a literal: {e}") from e


def _hf_music_pins() -> list[tuple[str, str, str, str, str]]:
    rel = "localm/media/koboldcpp/models.py"
    repo, revision = _py_literal(rel, "DEFAULT_REPO"), _py_literal(rel, "DEFAULT_REVISION")
    files, shas = _py_literal(rel, "DEFAULT_FILES"), _py_literal(rel, "DEFAULT_SHA256")
    return [(f"music {comp}", repo, files[comp], revision, shas[comp]) for comp in files]


def _hf_comfy_pins() -> list[tuple[str, str, str, str, str]]:
    rel = "localm/model_manager/registry.py"
    table = _py_value(rel, "COMFY_MODEL_SOURCES")
    if not isinstance(table, ast.Dict):
        raise LocalReadError(f"{rel}: COMFY_MODEL_SOURCES is not a dict literal")
    out = []
    for key, call in zip(table.keys, table.values, strict=True):
        try:
            filename = ast.literal_eval(key)
            args = [ast.literal_eval(a) for a in call.args]
            repo, path = args[0].split(":", 1)
            sha = args[4]
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            raise LocalReadError(f"{rel}: a COMFY_MODEL_SOURCES entry is not "
                                 f"ComfySource(spec, type, folder, size, sha256): {e}") from e
        out.append((f"ComfyUI {filename}", repo, path, "main", sha))
    return out


def _hf_starter_pins() -> list[tuple[str, str, str, str, str]]:
    rel = "localm/model_manager/registry.py"
    table = _py_literal(rel, "_STARTER_SHA256")
    out = []
    for spec, sha in table.items():
        repo, path = spec.split(":", 1)
        out.append((f"starter {path}", repo, path, "main", sha))
    return out


def _hf_whisper_pins() -> list[tuple[str, str, str, str, str]]:
    table = _py_literal("localm/voice.py", "_STT_PINS")
    return [(f"whisper {repo.rsplit('-', 1)[-1]}", repo, "model.bin", commit, sha)
            for repo, (commit, sha) in table.items()]


_HF_PIN_PROVIDERS = (_hf_music_pins, _hf_comfy_pins, _hf_whisper_pins, _hf_starter_pins)


def hf_file_pins() -> list[tuple[str, str, str, str, str]]:
    """``(label, repo, path, revision, sha256)`` for every Hugging Face file whose
    digest the product pins."""
    pins = []
    for provider in _HF_PIN_PROVIDERS:
        pins += provider()
    return pins


def _check_hf_file_digests(now):
    """Every pinned Hugging Face file digest against what the hub serves now, so a
    re-uploaded file or a removed or gated source is seen before a user's pull
    fails closed on it."""
    name, group, advancer = _HF_PINS_NAME, "models", "by hand: re-read the digest on the hub and update the table"
    try:
        pins = hf_file_pins()
        if not pins:
            raise LocalReadError("no pinned Hugging Face file found")
        differ, gone = [], []
        for label, repo, path, revision, sha in pins:
            if not _HEX64_RE.match(str(sha)):
                raise LocalReadError(f"{label}: the pinned digest is not 64 lowercase hex characters")
            try:
                served = _hf_linked_sha256(repo, path, revision)
            except FetchError as e:
                if e.status in (401, 403, 404):
                    gone.append(f"{label} (HTTP {e.status})")
                    continue
                raise
            if served != sha:
                differ.append(label)
    except FetchError as e:
        return _unknown(name, group, advancer, e)
    label = f"{len(pins)} files"
    if differ or gone:
        parts = []
        if differ:
            parts.append(f"{len(differ)} file(s) now have another digest: {', '.join(differ[:4])}")
        if gone:
            parts.append(f"{len(gone)} file(s) cannot be fetched anonymously: {', '.join(gone[:4])}")
        return Row(name=name, status=STALE if differ else INCONSISTENT, pinned=label, advancer=advancer,
                   group=group, detail="; ".join(parts))
    return Row(name=name, status=CURRENT, pinned=label, latest=label, days=0, advancer=advancer,
               group=group, detail=f"{len(pins)} pinned digest(s) match what the hub serves")


_HF_KERNEL_NAME = "finegrained-fp8 Hub kernel"


def _check_hf_kernel(now):
    """The commit pinned for the finegrained-fp8 Hub kernel against the commit its
    kernel version points at now, and whether the pinned commit still exists."""
    name, group = _HF_KERNEL_NAME, "models"
    advancer = "by hand: read the version's commit from the kernel hub and update the pin"
    worker = "localm/inference/backends/_hf_worker.py"
    try:
        repo = _read_const(worker, r'^FP8_KERNEL_REPO = "([^"]+)"')
        version = _read_const(worker, r"^FP8_KERNEL_VERSION = (\d+)")
        pinned = _read_const(worker, r'^FP8_KERNEL_REVISION = "([0-9a-f]{40})"')
        base = f"https://huggingface.co/api/kernels/{repo}"
        refs = _get_json(f"{base}/refs")
        branches = {b.get("name"): b.get("targetCommit") for b in refs.get("branches", [])
                    if isinstance(b, dict)} if isinstance(refs, dict) else {}
        try:
            _get_json(f"{base}/tree/{pinned}/build?recursive=false&expand=false")
        except FetchError as e:
            if e.status == 404:
                return Row(name=name, status=INCONSISTENT, pinned=pinned[:12], advancer=advancer,
                           group=group, detail="the pinned commit no longer exists on the kernel hub")
            raise
    except FetchError as e:
        return _unknown(name, group, advancer, e)
    current = branches.get(f"v{version}")
    if current is None:
        return Row(name=name, status=INCONSISTENT, pinned=pinned[:12], advancer=advancer, group=group,
                   detail=f"the kernel hub has no v{version} branch of {repo}")
    if current != pinned:
        return Row(name=name, status=BEHIND, pinned=pinned[:12], latest=current[:12], advancer=advancer,
                   group=group, detail=f"v{version} of {repo} now points at {current[:12]}")
    return Row(name=name, status=CURRENT, pinned=pinned[:12], latest=current[:12], days=0,
               advancer=advancer, group=group, detail=f"v{version} of {repo} is the pinned commit")


# A pin that is stale for a known reason no automation can fix is reported as BLOCKED, with
# the reason, until the expiry date; after it the row reads STALE again and has to be
# decided afresh. name -> (expiry ISO date, reason).
ACCEPTED_BLOCKERS: dict[str, tuple[str, str]] = {
    "vendored marked": ("2026-12-31", "marked 16 and later no longer ships the file the page "
                        "loads, so moving past 15.x needs a code change in the GUI"),
    "vendored transformers.js": ("2026-12-31", "3.8.1 is the last 3.x release and 4.x ships a "
                                 "different onnxruntime runtime than the vendored one, so moving "
                                 "needs a code change in the TTS plugin; bump_tts_bundle.py refuses it"),
    "vendored onnxruntime-web": ("2026-12-31", "the version is whatever the vendored transformers.js "
                                 "pins, so it moves only with that release"),
}


def apply_blockers(rows: list[Row], now: _dt.datetime,
                   blockers: dict[str, tuple[str, str]] | None = None) -> list[Row]:
    """Turn a STALE or BEHIND row with an unexpired accepted blocker into BLOCKED."""
    blockers = ACCEPTED_BLOCKERS if blockers is None else blockers
    for row in rows:
        entry = blockers.get(row.name)
        if entry is None or row.status not in (STALE, BEHIND):
            continue
        expiry, reason = entry
        try:
            expires = _dt.datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=_dt.UTC)
        except ValueError:
            continue
        if now <= expires:
            row.status = BLOCKED
            row.detail = f"accepted until {expiry}: {reason}. {row.detail}".strip()
    return rows


# Sources the registry cannot compare with upstream, each reported on every run with the
# reason: (name, file that must still exist, why it is not checked).
_NOT_CHECKABLE = (
    ("torch wheel indexes (cpu, cu126, cu130, xpu, rocm6.2)", "localm/hwdetect.py",
     "floating package indexes; the torch version itself comes from uv.lock"),
    ("libgomp package for the Linux bundle", "localm/setup_llama/native_deps.py",
     "pinned by sha256 against an immutable snapshot URL"),
    ("native image and video model files", "localm/plugins/builtin/image/backends/native.py",
     "pinned by sha256 and size on Hugging Face; the files are model weights, not runtimes"),
    ("default embedding models", "localm/inference/embedder.py",
     "Hugging Face model repositories chosen by name; no version to compare"),
    ("managed Python 3.12", "pyproject.toml",
     "uv installs the newest 3.12 patch; the floor is requires-python"),
)


# --------------------------------------------------------------------------- #
#  Registry                                                                   #
# --------------------------------------------------------------------------- #

def build_registry() -> list[PinSpec]:
    specs: list[PinSpec] = []
    for name, group, script, advancer, pinned_re, latest_re in _LEGACY:
        specs.append(PinSpec(name, group, advancer,
                             _legacy_check(name, group, script, advancer, pinned_re, latest_re),
                             legacy=True))
    specs += [
        PinSpec("koboldcpp", "media", "pin_weekly.py", _check_github_tag_pin(
            "koboldcpp", "media", "pin_weekly.py", "LostRuins/koboldcpp",
            "localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"', _semver_key)),
        PinSpec("stable-diffusion.cpp", "media", "pin_weekly.py", _check_github_tag_pin(
            "stable-diffusion.cpp", "media", "pin_weekly.py", "leejet/stable-diffusion.cpp",
            "localm/media/sdcpp/pins.py", r'^TAG = "([^"]+)"', _sdcpp_key)),
        PinSpec("ComfyUI-GGUF node", "comfyui", "pin_weekly.py (review PR)", _check_gguf_node),
        PinSpec("uv (installers + Docker)", "tooling", "pin_weekly.py", _check_uv),
        PinSpec("Docker base image", "tooling", "pin_weekly.py (review PR)", _check_docker_base),
        PinSpec("llama.cpp asset checksums", "llama", "pin_weekly.py (llama, rocm)", _check_llama_asset_checksums),
        PinSpec("Linux CUDA build source", "cuda", "rides the llama.cpp pin", _check_cuda_linux_source),
        PinSpec("ROCm CPU archive", "rocm", "rides the ROCm pin", _check_rocm_cpu_asset),
        PinSpec("Linux CUDA runtime wheels", "cuda", "pin_weekly.py (review PR)", _check_cuda_runtime_wheels),
        PinSpec(_HF_PINS_NAME, "models", "by hand: re-read the digest on the hub and update the table",
                _check_hf_file_digests),
        PinSpec(_HF_KERNEL_NAME, "models", "by hand: read the version's commit from the kernel hub and update the pin",
                _check_hf_kernel),
    ]
    for spec in _VENDORED:
        specs.append(PinSpec(f"vendored {spec[0]}", "vendored", _vendored_advancer(spec[0]),
                             _check_vendored(spec)))
    for name, rel, why in _UNVERSIONED_VENDORED:
        specs.append(PinSpec(f"vendored {name}", "vendored", "pin_weekly.py (GUI tests; review for a major)",
                             _unversioned(name, rel, why)))
    for name, rel, why in _NOT_CHECKABLE:
        specs.append(PinSpec(name, "unchecked", "none", _not_checkable(name, rel, why)))
    return specs


# --------------------------------------------------------------------------- #
#  Running and reporting                                                      #
# --------------------------------------------------------------------------- #

def run_checks(specs: list[PinSpec], now: _dt.datetime | None = None) -> list[Row]:
    now = now or _dt.datetime.now(_dt.UTC)
    rows = []
    for spec in specs:
        try:
            row = spec.check(now)
        except Exception as e:  # noqa: BLE001 - one broken check must not hide the others
            row = Row(name=spec.name, status=UNKNOWN, advancer=spec.advancer, group=spec.group,
                      detail=f"check crashed: {type(e).__name__}: {e}")
        row.group = row.group or spec.group
        row.advancer = row.advancer or spec.advancer
        rows.append(row)
    return apply_blockers(rows, now)


def exit_code(rows: list[Row]) -> int:
    if any(r.status in (STALE, INCONSISTENT) for r in rows):
        return EXIT_STALE
    if any(r.status == UNKNOWN for r in rows):
        return EXIT_UNKNOWN
    return EXIT_OK


def format_table(rows: list[Row]) -> str:
    head = ("| pin | status | pinned | newest | newer for (days) | tolerance | advanced by | detail |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r.name} | {r.status} | {r.pinned or '-'} | {r.latest or '-'} | "
        f"{'-' if r.days is None else r.days} | {'-' if r.tolerance is None else r.tolerance} | "
        f"{r.advancer or '-'} | {r.detail.replace('|', '/')[:200] or '-'} |\n"
        for r in rows)
    return head + body


def _annotate(level: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def _write_summary(text: str) -> None:
    try:
        ci_runner_files.append(ci_runner_files.STEP_SUMMARY, text)
    except OSError as e:
        print(f"(could not write the step summary: {e})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 when any pin is STALE or INCONSISTENT, 2 when none is but "
                         "one could not be checked (default: always exit 0)")
    ap.add_argument("--new-only", action="store_true",
                    help="skip the four pins that already have their own gate script")
    ap.add_argument("--only", default="", help="comma-separated pin names to check")
    ap.add_argument("--exclude", default="", help="comma-separated pin names to skip")
    ap.add_argument("--json", default="", help="also write the rows as JSON to this path")
    args = ap.parse_args(argv)

    specs = build_registry()
    if args.new_only:
        specs = [s for s in specs if not s.legacy]
    if args.exclude:
        skipped = {w.strip().lower() for w in args.exclude.split(",") if w.strip()}
        known = {s.name.lower() for s in specs}
        if skipped - known:
            print(f"--exclude names no known pin: {sorted(skipped - known)}", file=sys.stderr)
            return EXIT_USAGE
        specs = [s for s in specs if s.name.lower() not in skipped]
    if args.only:
        wanted = {w.strip().lower() for w in args.only.split(",") if w.strip()}
        specs = [s for s in specs if s.name.lower() in wanted]
        if not specs:
            print(f"no pin matches --only {args.only!r}", file=sys.stderr)
            return EXIT_USAGE

    rows = run_checks(specs)
    table = format_table(rows)
    print(table)
    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for r in rows:
        if r.status in (STALE, INCONSISTENT):
            _annotate("error", f"{r.name}: {r.status} ({r.pinned} -> {r.latest}); {r.detail}")
        elif r.status in (BEHIND, UNKNOWN, UNVERSIONED, BLOCKED):
            _annotate("warning", f"{r.name}: {r.status}; {r.detail}")
    _write_summary("## Pin currency (all runtimes)\n\n" + table)

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    return exit_code(rows) if args.gate else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
