# SPDX-License-Identifier: AGPL-3.0-or-later
"""`model add` SHA-256 hashing, and the option to skip it:

  (a) the default add path hashes off the main thread with a live progress bar
      for large files - a progress callback on `_sha256_file` and a
      `_hash_with_progress` helper that yields the same digest, including via
      its threaded path;
  (c) a `--fast` flag that skips content hashing and dedups on file SIZE, for
      known-unique bulk imports.
"""

import pytest

from localm import model_manager as mm


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager."""
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)
    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)
    monkeypatch.setattr(mm, "update_registry", _update)
    return store, models_dir


def _file(d, name, content=b"model-bytes"):
    p = d / name
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------- #
#  (a) progress-aware hashing                                                  #
# --------------------------------------------------------------------------- #

def test_sha256_progress_callback_reports_and_matches(tmp_path):
    # 65536-byte block size, so > 1 block exercises multiple callbacks.
    data = b"x" * (65536 * 3 + 17)
    f = _file(tmp_path, "big.bin", data)
    seen = []
    digest = mm._sha256_file(f, progress=lambda done, total: seen.append((done, total)))

    assert digest == mm._sha256_file(f)        # identical to the no-callback digest
    assert seen, "progress callback never fired"
    assert all(t == len(data) for _, t in seen)  # total stays the file size
    dones = [d for d, _ in seen]
    assert dones == sorted(dones)              # monotonic
    assert dones[-1] == len(data)              # ends exactly at the total


def test_hash_with_progress_matches_plain(tmp_path):
    f = _file(tmp_path, "m.gguf", b"abc" * 1000)
    assert mm._hash_with_progress(f) == mm._sha256_file(f)


def test_hash_with_progress_uses_threaded_path(tmp_path, monkeypatch):
    # Force the large-file (threaded + progress bar) branch with a tiny file by
    # dropping the size threshold to 0; the digest must still be correct.
    monkeypatch.setattr(mm, "_HASH_PROGRESS_MIN_BYTES", 0)
    f = _file(tmp_path, "m.gguf", b"thread-me" * 50)
    assert mm._hash_with_progress(f) == mm._sha256_file(f)


def test_hash_with_progress_dir_returns_none(tmp_path):
    d = tmp_path / "hf-model"
    d.mkdir()
    assert mm._hash_with_progress(d) is None


# --------------------------------------------------------------------------- #
#  (c) --fast: skip hashing, dedup on size                                     #
# --------------------------------------------------------------------------- #

def test_find_by_size(fake_registry, tmp_path):
    store, _ = fake_registry
    a = _file(tmp_path, "a.gguf", b"1234567890")   # 10 bytes
    b = _file(tmp_path, "b.gguf", b"abcdefghij")   # 10 bytes (distinct content)
    c = _file(tmp_path, "c.gguf", b"short")        # 5 bytes
    store["a"] = {"path": str(a), "source": "local"}
    store["b"] = {"path": str(b), "source": "local"}
    store["c"] = {"path": str(c), "source": "local"}
    assert mm.find_by_size(10) == ["a", "b"]
    assert mm.find_by_size(5) == ["c"]
    assert mm.find_by_size(0) == []


def test_fast_skips_hash(fake_registry, tmp_path, monkeypatch):
    store, _ = fake_registry
    f = _file(tmp_path, "m.gguf")

    def _boom(*a, **k):
        raise AssertionError("--fast must not hash file content")
    monkeypatch.setattr(mm, "_sha256_file", _boom)
    monkeypatch.setattr(mm, "_hash_with_progress", _boom)

    mm.add_local(str(f), "m", on_duplicate="register", fast=True)
    assert "m" in store
    assert "sha256" not in store["m"]          # no digest stored in fast mode


def test_fast_dedups_on_size(fake_registry, tmp_path):
    store, _ = fake_registry
    # Two DISTINCT files of the SAME byte length.
    f1 = _file(tmp_path, "a.gguf", b"AAAAAAAAAA")
    f2 = _file(tmp_path, "b.gguf", b"BBBBBBBBBB")
    mm.add_local(str(f1), "first", on_duplicate="register", fast=True)
    mm.add_local(str(f2), "second", on_duplicate="skip", fast=True)
    assert "first" in store
    assert "second" not in store               # flagged as a size duplicate, skipped


def test_fast_registers_distinct_size(fake_registry, tmp_path):
    store, _ = fake_registry
    f1 = _file(tmp_path, "a.gguf", b"AAAAAAAAAA")   # 10 bytes
    f2 = _file(tmp_path, "b.gguf", b"BBB")          # 3 bytes
    mm.add_local(str(f1), "first", on_duplicate="register", fast=True)
    mm.add_local(str(f2), "second", on_duplicate="skip", fast=True)
    assert "first" in store and "second" in store   # different sizes -> both kept


def test_default_path_still_hashes(fake_registry, tmp_path):
    # Without --fast, distinct same-size files are NOT deduped (the content hash
    # distinguishes them) and a digest is stored.
    store, _ = fake_registry
    f1 = _file(tmp_path, "a.gguf", b"AAAAAAAAAA")
    f2 = _file(tmp_path, "b.gguf", b"BBBBBBBBBB")
    mm.add_local(str(f1), "first", on_duplicate="register")
    mm.add_local(str(f2), "second", on_duplicate="skip")
    assert store["first"].get("sha256")
    assert "second" in store                    # distinct content -> registered


# --------------------------------------------------------------------------- #
#  CLI wiring                                                                  #
# --------------------------------------------------------------------------- #

def test_cli_add_fast_passes_flag(cli_runner, tmp_path, monkeypatch):
    import localm.cli as cli
    captured = {}

    def _fake_add_local(path, name=None, on_duplicate="ask", no_hash=False, fast=False,
                        model_type=None, store=None):
        captured["fast"] = fast
        captured["no_hash"] = no_hash
        captured["model_type"] = model_type
        captured["store"] = store
        return True
    monkeypatch.setattr(cli, "add_local", _fake_add_local)

    f = _file(tmp_path, "m.gguf")
    res = cli_runner.invoke(cli.main, ["add", str(f), "--fast"])
    assert res.exit_code == 0, res.output
    assert captured["fast"] is True
