#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advance the digest of localm's pinned Docker base image.

``docker/Dockerfile`` builds on ``ARG UBUNTU_IMAGE=ubuntu:<tag>@sha256:<digest>``.
This script moves the digest and keeps the tag part. It is review-only: a person
reads the diff and the printed checklist, then merges.

WHAT IT REWRITES (with ``--write``; without it, a unified diff is printed and
nothing is touched):

  docker/Dockerfile
    ARG UBUNTU_IMAGE       the sha256 digest only; the image name and tag stay

WHAT IT CHECKS BEFORE WRITING:
  * ``--tag`` is ``sha256:`` plus 64 lowercase hex characters and differs from
    the pinned digest;
  * the image is a Docker Hub image, and the registry serves a manifest for
    that digest whose bytes hash to the digest;
  * the manifest is an image index that lists a linux/amd64 image;
  * the pinned tag resolves, at this moment, to that same digest, so the pin
    moves to what the tag publishes and not to an arbitrary or stale digest;
  * the edited region is found exactly once.

WHAT IT LEAVES TO A PERSON, printed as the remaining checklist: the targeted
tests, building and starting the image for the backends, and the CHANGELOG
decision. The image is not built or run here.

Exit codes: 0 when the edit was applied or the dry run completed; 1 when
refused.

Usage:
    python scripts/bump_docker_base_pin.py --tag sha256:<64 hex>
    python scripts/bump_docker_base_pin.py --tag sha256:<64 hex> --write
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE_REL = "docker/Dockerfile"
AUTH_URL = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:%s:pull"
MANIFEST_URL = "https://registry-1.docker.io/v2/%s/manifests/%s"
MAX_BODY_BYTES = 4 * 1024 * 1024
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json")
ACCEPT = ", ".join(INDEX_TYPES + (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json"))

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARG_RE = re.compile(
    r"^(?P<head>ARG UBUNTU_IMAGE=(?P<name>[a-z0-9][a-z0-9._/-]*):(?P<tag>[A-Za-z0-9._-]+)@)"
    r"(?P<digest>sha256:[0-9a-f]{64})(?P<tail>[ \t]*)$", re.M)


class Refused(Exception):
    """The bump cannot proceed; the message says why."""


# --------------------------------------------------------------------------- #
#  Upstream (injectable)                                                      #
# --------------------------------------------------------------------------- #

def _default_open(req, timeout):
    from localm.http_ssl import verified_urlopen
    return verified_urlopen(req, timeout=timeout)


def fetch(url: str, headers: dict, opener=None) -> tuple:
    """GET *url*; return ``(status, body bytes)``. An HTTP error status is
    returned, not raised; any other failure raises Refused."""
    opener = opener or _default_open
    req = urllib.request.Request(url, headers={"User-Agent": "localm-bump-docker-base-pin",
                                               **headers})
    try:
        with opener(req, 30) as resp:
            body = resp.read(MAX_BODY_BYTES + 1)
            status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        raise Refused(f"could not read {url}: {type(e).__name__}: {e}") from e
    if len(body) > MAX_BODY_BYTES:
        raise Refused(f"{url} answered with more than {MAX_BODY_BYTES} bytes")
    return status, body


def hub_repository(name: str) -> str:
    """The Docker Hub repository path of image *name* (``ubuntu`` is
    ``library/ubuntu``). Raises Refused for an image on another registry."""
    parts = name.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        raise Refused(f"{name} is not a Docker Hub image; only Docker Hub is supported")
    return name if len(parts) > 1 else f"library/{name}"


def verify_digest(name: str, tag: str, digest: str, fetch_fn=fetch) -> list:
    """The platforms ("os/arch") of the image index behind *digest*, verified.

    Raises Refused unless the registry serves *digest* with matching bytes, as
    an index listing linux/amd64, and *tag* currently resolves to *digest*."""
    repo = hub_repository(name)
    status, body = fetch_fn(AUTH_URL % repo, {})
    try:
        token = json.loads(body.decode("utf-8")).get("token") if status == 200 else None
    except (ValueError, AttributeError):
        token = None
    if not isinstance(token, str) or not token:
        raise Refused(f"could not obtain a pull token for {repo} (HTTP {status})")
    auth = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}

    status, body = fetch_fn(MANIFEST_URL % (repo, digest), auth)
    if status == 404:
        raise Refused(f"the registry has no manifest {digest} for {repo}")
    if status != 200:
        raise Refused(f"the registry answered HTTP {status} for {repo}@{digest}")
    got = "sha256:" + hashlib.sha256(body).hexdigest()
    if got != digest:
        raise Refused(f"the manifest served for {digest} hashes to {got}")
    try:
        doc = json.loads(body.decode("utf-8"))
    except ValueError as e:
        raise Refused(f"the manifest for {digest} is not JSON: {e}") from e
    if not isinstance(doc, dict) or doc.get("mediaType") not in INDEX_TYPES \
            or not isinstance(doc.get("manifests"), list):
        raise Refused(f"{digest} is not an image index (mediaType "
                      f"{doc.get('mediaType') if isinstance(doc, dict) else None!r})")
    platforms = []
    for entry in doc["manifests"]:
        plat = entry.get("platform") if isinstance(entry, dict) else None
        if isinstance(plat, dict) and plat.get("os") not in (None, "unknown"):
            platforms.append(f"{plat.get('os')}/{plat.get('architecture')}"
                             + (f"/{plat['variant']}" if plat.get("variant") else ""))
    if "linux/amd64" not in platforms:
        raise Refused(f"{digest} lists no linux/amd64 image (platforms: "
                      f"{', '.join(platforms) or 'none'})")

    status, body = fetch_fn(MANIFEST_URL % (repo, tag), auth)
    if status != 200:
        raise Refused(f"the registry answered HTTP {status} for the tag {repo}:{tag}")
    current = "sha256:" + hashlib.sha256(body).hexdigest()
    if current != digest:
        raise Refused(f"{name}:{tag} resolves to {current} now, not {digest}; this script "
                      "moves the pin to what the tag publishes")
    return sorted(set(platforms))


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
    """(image name, tag, digest) of the ``ARG UBUNTU_IMAGE`` line. Raises
    Refused when it is not found exactly once."""
    matches = list(_ARG_RE.finditer(text))
    if len(matches) != 1:
        raise Refused(f"ARG UBUNTU_IMAGE: expected exactly one name:tag@sha256 line, found "
                      f"{len(matches)}; the file shape this script edits has changed")
    m = matches[0]
    return m.group("name"), m.group("tag"), m.group("digest")


def rewrite(text: str, digest: str) -> str:
    """*text* with the ``ARG UBUNTU_IMAGE`` digest replaced by *digest*."""
    read_pin(text)
    m = _ARG_RE.search(text)
    return text[:m.start("digest")] + digest + text[m.end("digest"):]


def checklist(name: str, tag: str, platforms: list) -> str:
    return "\n".join([
        "REMAINING STEPS, not automated - each has its own check:",
        "  1. pytest tests/test_bump_docker_base_pin.py tests/test_docker_image_files.py",
        f"  2. {name}:{tag} now publishes: {', '.join(platforms)}",
        "       build docker/Dockerfile for the backends you can (cpu always) and start the image;",
        "       this script does not build or run it",
        "  3. CHANGELOG.md, [Unreleased]: only if the published image's contents change for users",
    ])


def main(argv=None, *, fetch_fn=fetch) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="the new image digest, sha256:<64 hex>")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a diff and change nothing)")
    args = ap.parse_args(argv)

    path = REPO / DOCKERFILE_REL
    digest = args.tag.strip()
    try:
        if not _DIGEST_RE.match(digest):
            raise Refused(f"{args.tag!r} is not sha256: followed by 64 lowercase hex characters")
        text, newline = _read(path)
        name, tag, old = read_pin(text)
        if digest == old:
            raise Refused(f"{digest} is the digest already pinned")
        print(f"checking {name}:{tag}@{digest[:19]}... (pinned {old[:19]}...) ...")
        platforms = verify_digest(name, tag, digest, fetch_fn)
        print(f"  served by the registry, hashes to itself, publishes {', '.join(platforms)}, "
              f"and {tag} resolves to it")
        new_text = rewrite(text, digest)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    if args.write:
        _write(path, new_text, newline)
        print(f"wrote {DOCKERFILE_REL}")
    else:
        sys.stdout.writelines(difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{DOCKERFILE_REL}", tofile=f"b/{DOCKERFILE_REL}"))
        print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(name, tag, platforms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
