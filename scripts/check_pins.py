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
2 nothing stale but at least one row could not be checked. Without --gate the
report is printed and the exit code is always 0.

Usage:
    python scripts/check_pins.py
    python scripts/check_pins.py --gate
    python scripts/check_pins.py --gate --new-only --json out.json
    python scripts/check_pins.py --gate --exclude "llama.cpp,ComfyUI"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CURRENT = "CURRENT"
BEHIND = "BEHIND"
STALE = "STALE"
INCONSISTENT = "INCONSISTENT"
UNKNOWN = "UNKNOWN"
UNVERSIONED = "UNVERSIONED"

EXIT_OK = 0
EXIT_STALE = 1
EXIT_UNKNOWN = 2

DEFAULT_TOLERANCE_DAYS = 30
SECURITY_TOLERANCE_DAYS = 60
SLOW_TOLERANCE_DAYS = 180

_TIMEOUT = 20


class FetchError(Exception):
    """An upstream lookup failed; the row reads UNKNOWN, never CURRENT."""


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

def _headers() -> dict:
    headers = {"User-Agent": "localm-check-pins", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str):
    """GET *url* as JSON. Raises FetchError on any failure, including a body that
    is not JSON. Tests replace this function; nothing else opens a socket."""
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310 - fixed https:// URLs
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise FetchError(f"{url}: {e}") from e


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

_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def _semver_key(tag) -> tuple[int, ...] | None:
    """'v1.122.1' / '3.4.13' / '1.22.0-dev.2025' -> (1, 122, 1); anything else None."""
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
        raise FetchError(f"cannot read {rel}: {e}") from e
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise FetchError(f"{rel}: pattern {pattern!r} not found (renamed or reformatted?)")
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
    name, group, advancer = "ComfyUI-GGUF node", "comfyui", "none (no pipeline)"
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
    name, group, advancer = "uv (installers + Docker)", "tooling", "none (hand-edited)"
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
        if "404" in str(e):
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
        if "404" in str(e):
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
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310 - fixed https:// URL
            digest = r.headers.get("Docker-Content-Digest")
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
    name, group, advancer = "Docker base image", "tooling", "none (hand-edited)"
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


_CUDA_RUNTIME_RE = re.compile(r'^_CUDA_RUNTIME_PIN = \{(.*?)^\}', re.MULTILINE | re.DOTALL)


def _check_cuda_runtime_wheels(now):
    name, group = "Linux CUDA runtime wheels", "cuda"
    advancer = "none (hand-edited)"
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
        stamps = [s for *_, s in rows if s]
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


def _check_vendored(spec):
    name, rel, pattern, (source, ref), tolerance = spec
    label = f"vendored {name}"

    def check(now):
        pinned = ""
        try:
            pinned = _read_const(rel, pattern)
            releases = _github_releases(ref) if source == "gh" else _npm_releases(ref)
        except FetchError as e:
            return _unknown(label, "vendored", "none (hand-maintained)", e, pinned)
        return _assess_releases(label, "vendored", "none (hand-maintained)", pinned, releases,
                                _semver_key, tolerance, now)
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
#  Registry                                                                   #
# --------------------------------------------------------------------------- #

def build_registry() -> list[PinSpec]:
    specs: list[PinSpec] = []
    for name, group, script, advancer, pinned_re, latest_re in _LEGACY:
        specs.append(PinSpec(name, group, advancer,
                             _legacy_check(name, group, script, advancer, pinned_re, latest_re),
                             legacy=True))
    specs += [
        PinSpec("koboldcpp", "media", "none (no pipeline)", _check_github_tag_pin(
            "koboldcpp", "media", "none (no pipeline)", "LostRuins/koboldcpp",
            "localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"', _semver_key)),
        PinSpec("stable-diffusion.cpp", "media", "none (no pipeline)", _check_github_tag_pin(
            "stable-diffusion.cpp", "media", "none (no pipeline)", "leejet/stable-diffusion.cpp",
            "localm/media/sdcpp/pins.py", r'^TAG = "([^"]+)"', _sdcpp_key)),
        PinSpec("ComfyUI-GGUF node", "comfyui", "none (no pipeline)", _check_gguf_node),
        PinSpec("uv (installers + Docker)", "tooling", "none (hand-edited)", _check_uv),
        PinSpec("Docker base image", "tooling", "none (hand-edited)", _check_docker_base),
        PinSpec("Linux CUDA build source", "cuda", "rides the llama.cpp pin", _check_cuda_linux_source),
        PinSpec("ROCm CPU archive", "rocm", "rides the ROCm pin", _check_rocm_cpu_asset),
        PinSpec("Linux CUDA runtime wheels", "cuda", "none (hand-edited)", _check_cuda_runtime_wheels),
    ]
    for spec in _VENDORED:
        specs.append(PinSpec(f"vendored {spec[0]}", "vendored", "none (hand-maintained)",
                             _check_vendored(spec)))
    for name, rel, why in _UNVERSIONED_VENDORED:
        specs.append(PinSpec(f"vendored {name}", "vendored", "none (hand-maintained)",
                             _unversioned(name, rel, why)))
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
    return rows


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
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
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
            return EXIT_UNKNOWN if args.gate else EXIT_OK
        specs = [s for s in specs if s.name.lower() not in skipped]
    if args.only:
        wanted = {w.strip().lower() for w in args.only.split(",") if w.strip()}
        specs = [s for s in specs if s.name.lower() in wanted]
        if not specs:
            print(f"no pin matches --only {args.only!r}", file=sys.stderr)
            return EXIT_UNKNOWN if args.gate else EXIT_OK

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
        elif r.status in (BEHIND, UNKNOWN, UNVERSIONED):
            _annotate("warning", f"{r.name}: {r.status}; {r.detail}")
    _write_summary("## Pin currency (all runtimes)\n\n" + table)

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    return exit_code(rows) if args.gate else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
