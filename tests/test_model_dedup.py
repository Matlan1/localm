"""
Tests for duplicate model detection, aliasing, and alias-aware removal.

Two-tier identity: resolved path first, stored sha256 second.
"""

import json
from pathlib import Path
from unittest.mock import patch

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
    # _register now goes through update_registry (atomic read-modify-write);
    # route it at the in-memory store so the fake stays faithful.
    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)
    monkeypatch.setattr(mm, "update_registry", _update)
    return store, models_dir


def _file(tmp_path: Path, name: str, content: bytes = b"model-bytes") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
#  Identity helpers
# ---------------------------------------------------------------------------

class TestFindHelpers:
    def test_find_aliases_by_path(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["a"] = {"path": str(f), "source": "local"}
        store["b"] = {"path": str(f), "source": "local"}
        store["c"] = {"path": str(tmp_path / "other.gguf"), "source": "local"}
        assert mm.find_aliases_by_path(f) == ["a", "b"]

    def test_find_by_sha256_case_insensitive(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "x", "source": "local", "sha256": "ABCDEF"}
        assert mm.find_by_sha256("abcdef") == ["a"]

    def test_find_by_sha256_ignores_entries_without_digest(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "x", "source": "local"}
        assert mm.find_by_sha256("abcdef") == []

    def test_find_by_sha256_empty_digest(self, fake_registry):
        assert mm.find_by_sha256("") == []


# ---------------------------------------------------------------------------
#  alias_model
# ---------------------------------------------------------------------------

class TestAliasModel:
    def test_creates_alias_with_same_entry(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local", "sha256": "abc"}
        assert mm.alias_model("orig", "second") is True
        assert store["second"] == store["orig"]

    def test_unknown_source_fails(self, fake_registry):
        assert mm.alias_model("ghost", "x") is False

    def test_taken_name_fails(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        store["b"] = {"path": "q", "source": "local"}
        assert mm.alias_model("a", "b") is False


# ---------------------------------------------------------------------------
#  _register stores sha256
# ---------------------------------------------------------------------------

class TestRegisterDigest:
    def test_sha256_stored_lowercase(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm._register("m", f, "local", sha256="ABC123")
        assert store["m"]["sha256"] == "abc123"

    def test_no_sha256_key_when_absent(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm._register("m", f, "local")
        assert "sha256" not in store["m"]


# ---------------------------------------------------------------------------
#  add_local duplicate flows
# ---------------------------------------------------------------------------

class TestAddLocalDedup:
    def test_same_name_same_path_is_noop(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "m", on_duplicate="register")
        before = dict(store)
        mm.add_local(str(f), "m", on_duplicate="register")
        assert store == before

    def test_noop_backfills_digest(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["m"] = {"path": str(f.resolve()), "source": "local"}  # no digest
        mm.add_local(str(f), "m", on_duplicate="register")
        assert store["m"].get("sha256") == mm._sha256_file(f)

    def test_same_path_new_name_alias(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")
        mm.add_local(str(f), "second", on_duplicate="alias")
        assert store["second"]["path"] == store["first"]["path"]

    def test_same_path_new_name_skip(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")
        mm.add_local(str(f), "second", on_duplicate="skip")
        assert "second" not in store

    def test_hash_tier_catches_copy_at_other_path(self, fake_registry, tmp_path):
        """Same bytes at a different location → detected via stored sha256."""
        store, _ = fake_registry
        f1 = _file(tmp_path, "a.gguf", b"identical-bytes")
        f2 = _file(tmp_path / "sub", "b.gguf", b"identical-bytes") \
            if (tmp_path / "sub").mkdir() is None else None
        f2 = tmp_path / "sub" / "b.gguf"
        f2.write_bytes(b"identical-bytes")
        mm.add_local(str(f1), "orig", on_duplicate="register")
        assert store["orig"]["sha256"]
        mm.add_local(str(f2), "dupe", on_duplicate="alias")
        # Alias points at the ORIGINAL file, no new path entry
        assert store["dupe"]["path"] == store["orig"]["path"]

    def test_copy_imports_into_models_dir(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")
        mm.add_local(str(f), "copy-name", on_duplicate="copy")
        assert (models_dir / "m.gguf").is_file()
        assert store["copy-name"]["path"] == str((models_dir / "m.gguf").resolve())
        assert f.exists()  # original untouched

    def test_move_imports_and_updates_aliases(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")
        mm.add_local(str(f), "moved", on_duplicate="move")
        assert not f.exists()
        assert (models_dir / "m.gguf").is_file()
        new_path = str((models_dir / "m.gguf").resolve())
        # Both the new name AND the pre-existing alias follow the file
        assert store["moved"]["path"] == new_path
        assert store["first"]["path"] == new_path

    def test_name_conflict_different_file_skipped_non_tty(
            self, fake_registry, tmp_path):
        store, _ = fake_registry
        f1 = _file(tmp_path, "a.gguf", b"one")
        f2 = _file(tmp_path, "b.gguf", b"two")
        mm.add_local(str(f1), "m", on_duplicate="register")
        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            mm.add_local(str(f2), "m", on_duplicate="register")
        # Original entry untouched
        assert store["m"]["path"] == str(f1.resolve())

    def test_no_hash_flag_skips_digest(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "m", on_duplicate="register", no_hash=True)
        assert "sha256" not in store["m"]

    def test_prompt_skips_without_tty(self, fake_registry, tmp_path):
        """on_duplicate='ask' must never block or register in scripts."""
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")
        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            mm.add_local(str(f), "second")   # default: ask
        assert "second" not in store


# ---------------------------------------------------------------------------
#  remove_model alias awareness
# ---------------------------------------------------------------------------

class TestRemoveAliasAware:
    def test_file_kept_while_other_alias_exists(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        f = models_dir / "m.gguf"
        f.write_bytes(b"x")
        store["a"] = {"path": str(f.resolve()), "source": "local"}
        store["b"] = {"path": str(f.resolve()), "source": "local"}
        mm.remove_model("a")
        assert f.exists()
        assert "a" not in store
        assert "b" in store

    def test_file_deleted_with_last_alias(self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        f = models_dir / "m.gguf"
        f.write_bytes(b"x")
        store["only"] = {"path": str(f.resolve()), "source": "local"}
        mm.remove_model("only")
        assert not f.exists()
        assert store == {}


# ---------------------------------------------------------------------------
#  Pre-download dedup in pull paths
# ---------------------------------------------------------------------------

class TestPullDedup:
    def test_gguf_pull_aliases_instead_of_downloading(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        f = _file(tmp_path, "have.gguf")
        store["have"] = {"path": str(f.resolve()), "source": "hf:o/r",
                         "sha256": "deadbeef"}
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: "deadbeef")
        monkeypatch.setattr(
            mm, "_prompt_predownload_dup", lambda d, n: "alias")

        download_called = []
        import huggingface_hub
        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download",
            lambda **kw: download_called.append(1))

        mm._pull_gguf_file("o/r:new.gguf", None)

        assert download_called == []
        assert store["new"]["path"] == store["have"]["path"]

    def test_gguf_pull_skip_downloads_nothing(
            self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        store["have"] = {"path": "x", "source": "hf:o/r", "sha256": "deadbeef"}
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: "deadbeef")
        monkeypatch.setattr(
            mm, "_prompt_predownload_dup", lambda d, n: "skip")
        mm._pull_gguf_file("o/r:new.gguf", None)
        assert "new" not in store

    def test_redownload_bypasses_check(
            self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        store["have"] = {"path": "x", "source": "hf:o/r", "sha256": "deadbeef"}
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: "deadbeef")
        prompted = []
        monkeypatch.setattr(
            mm, "_prompt_predownload_dup",
            lambda d, n: prompted.append(1) or "skip")

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(b"fresh")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)

        mm._pull_gguf_file("o/r:new.gguf", None, redownload=True)

        assert prompted == []           # never asked
        assert "new" in store           # actually downloaded + registered

    def test_url_pull_stores_digest(self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        content = b"url-model-bytes"

        class FakeResp:
            status_code = 200
            headers = {"content-length": str(len(content))}
            def raise_for_status(self): pass
            def iter_content(self, n): yield content

        import requests
        monkeypatch.setattr(requests, "head",
                            lambda *a, **k: FakeResp())
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: FakeResp())
        monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)

        mm._pull_url("https://x.test/url-model.gguf", "urlm")

        assert store["urlm"]["sha256"] == mm._sha256_file(
            models_dir / "url-model.gguf")
