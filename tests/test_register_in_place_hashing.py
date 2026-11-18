# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registering a file in place reads it only when a registered file of the same
size could be a copy of it, and a hashing speed is never reported from the first
few microseconds of a hash."""

import pytest

from localm import model_manager as mm
from localm.model_manager import gguf as gguf_mod
from tests.test_model_dedup import _file, fake_registry  # noqa: F401


@pytest.fixture()
def reads(monkeypatch):
    """The paths ``_sha256_file`` was asked to read."""
    seen: list = []
    real = mm._sha256_file

    def spy(path, *args, **kwargs):
        seen.append(path.name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(mm, "_sha256_file", spy)
    return seen


def test_a_file_no_registered_file_matches_in_size_is_registered_unread(
        fake_registry, tmp_path, reads):  # noqa: F811
    store, _ = fake_registry
    assert mm.add_local(str(_file(tmp_path, "a.gguf", b"a" * 100)), "a")
    assert mm.add_local(str(_file(tmp_path, "b.gguf", b"b" * 101)), "b")
    assert set(store) == {"a", "b"}
    assert reads == []


def test_adding_a_registered_path_again_reads_nothing(fake_registry, tmp_path, reads):  # noqa: F811
    store, _ = fake_registry
    f = _file(tmp_path, "a.gguf", b"a" * 100)
    assert mm.add_local(str(f), "a")
    assert mm.add_local(str(f), "a")
    assert reads == []


def test_a_same_size_copy_is_found_by_content_and_the_first_gets_its_digest(
        fake_registry, tmp_path, reads):  # noqa: F811
    store, _ = fake_registry
    first = _file(tmp_path, "orig.gguf", b"same-bytes")
    copy = _file(tmp_path, "copy.gguf", b"same-bytes")
    assert mm.add_local(str(first), "orig")
    assert "sha256" not in store["orig"]
    mm.add_local(str(copy), "copy", on_duplicate="skip")
    assert "copy" not in store, "a byte-identical copy is still a duplicate"
    assert store["orig"]["sha256"] == mm._sha256_file(first)
    assert sorted(reads[:2]) == ["copy.gguf", "orig.gguf"]


def test_a_same_size_file_with_other_content_is_registered(fake_registry, tmp_path):  # noqa: F811
    store, _ = fake_registry
    assert mm.add_local(str(_file(tmp_path, "x.gguf", b"x" * 64)), "x")
    assert mm.add_local(str(_file(tmp_path, "y.gguf", b"y" * 64)), "y")
    assert set(store) == {"x", "y"}
    assert store["x"]["sha256"] != store["y"]["sha256"]


def test_a_folder_of_files_with_different_sizes_is_registered_unread(
        fake_registry, tmp_path, reads):  # noqa: F811
    store, _ = fake_registry
    folder = tmp_path / "lib"
    folder.mkdir()
    for k in range(3):
        _file(folder, f"m{k}.gguf", b"z" * (200 + k))
    assert mm.add_local(str(folder))
    assert {"m0", "m1", "m2"} <= set(store)
    assert reads == []


def test_the_first_hash_sample_carries_no_speed(monkeypatch, tmp_path):
    frames: list = []
    monkeypatch.setattr(mm, "_emit_progress", lambda *a, **kw: frames.append(kw))
    monkeypatch.setattr(mm, "_import_status", lambda *a, **kw: None)

    def fake_hash(path, progress=None):
        progress(4 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
        return "0" * 64

    monkeypatch.setattr(mm, "_sha256_file", fake_hash)
    f = _file(tmp_path, "big.gguf", b"q")
    assert gguf_mod._hash_reporting(f, 1, "Hashing big.gguf") == "0" * 64
    assert frames[0]["rate"] is None, "a speed from the first instant is not a speed"
