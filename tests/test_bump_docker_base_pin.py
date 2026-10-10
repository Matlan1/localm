# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_docker_base_pin.py: advancing the pinned Docker base image digest.

Offline: Docker Hub is replaced by an in-memory registry that serves real-shaped
image indexes and hashes them the way a registry does. The tests cover:

  * a dry run changes nothing; --write moves the digest only and keeps the image
    name and tag, in the shape tests/test_docker_image_files.py requires;
  * a refusal edits nothing: a malformed or unchanged digest, one the registry
    does not serve, bytes that do not hash to the digest, a manifest that is not
    an index or lacks linux/amd64, a tag that resolves to another digest, an
    image outside Docker Hub, a failed token request;
  * the ARG line is found exactly once;
  * the line on the real tree parses and the rewrite keeps the tested shape.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_docker_base_pin.py"

INDEX_TYPE = "application/vnd.oci.image.index.v1+json"


def index_bytes(platforms=("linux/amd64", "linux/arm64"), salt="", media=INDEX_TYPE):
    manifests = []
    for p in platforms:
        os_, arch = p.split("/")
        manifests.append({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                          "digest": "sha256:" + hashlib.sha256((p + salt).encode()).hexdigest(),
                          "size": 424, "platform": {"architecture": arch, "os": os_}})
    manifests.append({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                      "digest": "sha256:" + "e" * 64, "size": 566,
                      "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
                      "platform": {"architecture": "unknown", "os": "unknown"}})
    return json.dumps({"schemaVersion": 2, "mediaType": media, "manifests": manifests}).encode()


def digest_of(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


OLD_BODY = index_bytes(salt="old")
NEW_BODY = index_bytes(salt="new")
OLD = digest_of(OLD_BODY)
NEW = digest_of(NEW_BODY)

DOCKERFILE_FIXTURE = f'''# localm server image.

ARG UBUNTU_IMAGE=ubuntu:24.04@{OLD}
FROM ${{UBUNTU_IMAGE}}

ARG BACKEND=cpu
ARG UV_SHA256=b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6
'''


def _load():
    spec = importlib.util.spec_from_file_location("bump_docker_base_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    path = tmp_path / bump.DOCKERFILE_REL
    path.parent.mkdir(parents=True)
    path.write_text(DOCKERFILE_FIXTURE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(bump, "REPO", tmp_path)

    def no_network(req, timeout):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(bump, "_default_open", no_network)
    return path


class FakeRegistry:
    """Docker Hub's token endpoint and manifest endpoint for library/ubuntu."""

    def __init__(self, tag_body=NEW_BODY, blobs=None, token_status=200, token_body=None):
        self.blobs = {NEW: NEW_BODY, OLD: OLD_BODY}
        self.blobs.update(blobs or {})
        self.tags = {"24.04": tag_body}
        self.token_status = token_status
        self.token_body = token_body if token_body is not None else json.dumps({"token": "t"}).encode()
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        if url.startswith("https://auth.docker.io/token"):
            assert "repository:library/ubuntu:pull" in url
            return self.token_status, self.token_body
        m = re.fullmatch(r"https://registry-1\.docker\.io/v2/library/ubuntu/manifests/(.+)", url)
        assert m, url
        assert headers.get("Authorization") == "Bearer t"
        ref = m.group(1)
        body = self.blobs.get(ref) if ref.startswith("sha256:") else self.tags.get(ref)
        return (200, body) if body is not None else (404, b"")


def run(bump, argv, fake):
    return bump.main(argv, fetch_fn=fake)


# --------------------------------------------------------------------------- #
#  The edit                                                                   #
# --------------------------------------------------------------------------- #

def test_dry_run_changes_nothing_and_prints_the_diff(bump, tree, capsys):
    before = tree.read_bytes()
    fake = FakeRegistry()
    assert run(bump, ["--tag", NEW], fake) == 0
    out = capsys.readouterr().out
    assert f"-ARG UBUNTU_IMAGE=ubuntu:24.04@{OLD}" in out
    assert f"+ARG UBUNTU_IMAGE=ubuntu:24.04@{NEW}" in out
    assert "linux/amd64, linux/arm64" in out and "unknown" not in out.split("publishes")[1].split("\n")[0]
    assert "dry run: nothing written" in out and "REMAINING STEPS" in out
    assert tree.read_bytes() == before


def test_write_moves_the_digest_and_keeps_the_name_and_tag(bump, tree, capsys):
    assert run(bump, ["--tag", NEW, "--write"], FakeRegistry()) == 0
    text = tree.read_text(encoding="utf-8")
    assert text == DOCKERFILE_FIXTURE.replace(OLD, NEW)
    assert re.search(r"^ARG UBUNTU_IMAGE=ubuntu:[\d.]+@sha256:[0-9a-f]{64}$", text, re.M)
    capsys.readouterr()
    assert run(bump, ["--tag", NEW, "--write"], FakeRegistry()) == 1
    assert "is the digest already pinned" in capsys.readouterr().out


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_the_file_keeps_its_own_line_endings(bump, tree, newline):
    tree.write_bytes(DOCKERFILE_FIXTURE.replace("\n", newline).encode("utf-8"))
    assert run(bump, ["--tag", NEW, "--write"], FakeRegistry()) == 0
    data = tree.read_bytes()
    assert data.count(newline.encode()) == data.count(b"\n")
    assert (b"\r" in data) == (newline == "\r\n")


# --------------------------------------------------------------------------- #
#  Refusals                                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag", ["", NEW.split(":")[1], NEW.upper(), "sha256:" + "a" * 63,
                                 "sha256:" + "a" * 65, "sha512:" + "a" * 64, "sha256:" + "g" * 64,
                                 "ubuntu:24.04", f"{NEW}\n{NEW}", f" {NEW}x"])
def test_a_malformed_digest_is_refused_before_any_request(bump, tree, capsys, tag):
    before = tree.read_bytes()
    fake = FakeRegistry()
    assert run(bump, ["--tag", tag, "--write"], fake) == 1
    assert "sha256: followed by 64 lowercase hex" in capsys.readouterr().out
    assert fake.calls == [] and tree.read_bytes() == before


def test_the_pinned_digest_is_refused_before_any_request(bump, tree, capsys):
    fake = FakeRegistry()
    assert run(bump, ["--tag", OLD, "--write"], fake) == 1
    assert "is the digest already pinned" in capsys.readouterr().out
    assert fake.calls == []


def test_a_digest_the_registry_does_not_serve_is_refused(bump, tree, capsys):
    before = tree.read_bytes()
    unknown = "sha256:" + "9" * 64
    assert run(bump, ["--tag", unknown, "--write"], FakeRegistry()) == 1
    assert "has no manifest" in capsys.readouterr().out
    assert tree.read_bytes() == before


def test_bytes_that_do_not_hash_to_the_digest_are_refused(bump, tree, capsys):
    fake = FakeRegistry(blobs={NEW: index_bytes(salt="forged")})
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    assert "hashes to" in capsys.readouterr().out


def test_a_tag_that_resolves_elsewhere_is_refused(bump, tree, capsys):
    before = tree.read_bytes()
    other = index_bytes(salt="newer")
    fake = FakeRegistry(tag_body=other)
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    out = capsys.readouterr().out
    assert f"ubuntu:24.04 resolves to {digest_of(other)} now" in out
    assert tree.read_bytes() == before


def test_a_tag_the_registry_does_not_have_is_refused(bump, tree, capsys):
    fake = FakeRegistry()
    fake.tags.clear()
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    assert "HTTP 404 for the tag library/ubuntu:24.04" in capsys.readouterr().out


@pytest.mark.parametrize("body, fragment", [
    (index_bytes(platforms=("linux/arm64",), salt="n"), "lists no linux/amd64 image"),
    (index_bytes(platforms=("windows/amd64",), salt="n"), "lists no linux/amd64 image"),
    (index_bytes(media="application/vnd.oci.image.manifest.v1+json", salt="n"), "is not an image index"),
    (b'{"mediaType": 7}', "is not an image index"),
    (b"[]", "is not an image index"),
    (b"<html>", "is not JSON"),
])
def test_a_manifest_that_is_not_a_usable_index_is_refused(bump, tree, capsys, body, fragment):
    fake = FakeRegistry(tag_body=body, blobs={digest_of(body): body})
    assert run(bump, ["--tag", digest_of(body), "--write"], fake) == 1
    assert fragment in capsys.readouterr().out


@pytest.mark.parametrize("status, body", [(401, b""), (200, b"not json"), (200, b'{"token": ""}'),
                                          (200, b'{"nope": 1}'), (200, b"[]")])
def test_a_failed_token_request_is_refused(bump, tree, capsys, status, body):
    fake = FakeRegistry(token_status=status, token_body=body)
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    assert "could not obtain a pull token" in capsys.readouterr().out


def test_an_unreachable_registry_is_refused(bump, tree, capsys):
    def down(url, headers):
        raise bump.Refused("could not read the url: URLError: down")
    assert run(bump, ["--tag", NEW, "--write"], down) == 1
    assert "could not read" in capsys.readouterr().out


def test_an_image_outside_docker_hub_is_refused(bump, tree, capsys):
    tree.write_text(DOCKERFILE_FIXTURE.replace("ubuntu:24.04", "ghcr.io/o/ubuntu:24.04"),
                    encoding="utf-8")
    fake = FakeRegistry()
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    assert "not a Docker Hub image" in capsys.readouterr().out
    assert fake.calls == []


def test_a_namespaced_hub_image_uses_its_own_repository(bump):
    assert bump.hub_repository("ubuntu") == "library/ubuntu"
    assert bump.hub_repository("someorg/base") == "someorg/base"
    with pytest.raises(bump.Refused):
        bump.hub_repository("localhost/base")
    with pytest.raises(bump.Refused):
        bump.hub_repository("registry.example:5000/base")


# --------------------------------------------------------------------------- #
#  The ARG region                                                             #
# --------------------------------------------------------------------------- #

def test_the_arg_line_must_exist_exactly_once(bump):
    with pytest.raises(bump.Refused, match="found 0"):
        bump.read_pin("FROM ubuntu\n")
    with pytest.raises(bump.Refused, match="found 2"):
        bump.read_pin(DOCKERFILE_FIXTURE + DOCKERFILE_FIXTURE)
    with pytest.raises(bump.Refused, match="found 0"):
        bump.read_pin("ARG UBUNTU_IMAGE=ubuntu:24.04\n")


def test_the_line_on_the_real_tree_parses_and_keeps_the_tested_shape(bump):
    text, _ = bump._read(_ROOT / bump.DOCKERFILE_REL)
    name, tag, digest = bump.read_pin(text)
    assert (name, re.fullmatch(r"[\d.]+", tag) is not None) == ("ubuntu", True)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    new = bump.rewrite(text, NEW)
    assert re.search(r"^ARG UBUNTU_IMAGE=ubuntu:[\d.]+@sha256:[0-9a-f]{64}$", new, re.M)
    assert new == text.replace(digest, NEW) and text.count(digest) == 1
