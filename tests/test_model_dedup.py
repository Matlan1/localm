# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for duplicate model detection, aliasing, and alias-aware removal.

Two-tier identity: resolved path first, stored sha256 second.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from localm import model_manager as mm


def _backdate(path, seconds=60):
    """Set path's mtime `seconds` in the past, so sync_models_dir's settle check
    reads it as settled rather than possibly still mid-copy."""
    old = time.time() - seconds
    os.utime(path, (old, old))


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
    # _register goes through update_registry (atomic read-modify-write), routed
    # here at the in-memory store.
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

    # ---- alias sanitizes its new name -----------------------------------------
    @pytest.mark.parametrize("raw, expected", [
        ("../../evil", "evil"),        # traversal collapses to a bare component
        ("a/b/c", "a-b-c"),            # path separators -> hyphen
        ("", "model"),                 # empty -> the sanitizer's fallback, never ""
        (r"\\host\share\x", "host-share-x"),   # UNC path -> safe key
    ])
    def test_alias_sanitizes_new_name(self, fake_registry, tmp_path, raw, expected):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local", "sha256": "abc"}
        assert mm.alias_model("orig", raw) is True
        # The raw, unsafe string is NEVER a registry key ...
        assert raw not in store
        # ... a sanitized, path-safe key is created instead, pointing at the
        # same entry as the original.
        assert expected in store
        assert store[expected] == store["orig"]
        # No key may contain a path separator, '..', or be empty.
        for k in store:
            assert k != ""
            assert "/" not in k and "\\" not in k
            assert ".." not in k

    def test_alias_collision_checked_on_sanitized_name(self, fake_registry, tmp_path):
        # Two raw names that sanitize to the SAME key collide on the sanitized
        # value, not the raw string.
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local"}
        assert mm.alias_model("orig", "a/b") is True      # -> "a-b"
        assert "a-b" in store
        assert mm.alias_model("orig", "a\\b") is False     # also -> "a-b": taken


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

    def test_move_registry_update_is_atomic_on_failure(
            self, fake_registry, tmp_path, monkeypatch):
        """A move updates the registry in ONE atomic write. When that write
        fails after the file is physically moved, the registry is left FULLY
        pre-move, never half-updated with the pre-existing alias repointed and
        the new name missing. The file still lands under MODELS_DIR, where a
        launch-time sync_models_dir recovers it."""
        store, models_dir = fake_registry
        f = _file(tmp_path, "m.gguf")
        mm.add_local(str(f), "first", on_duplicate="register")   # 'first' -> external f
        before = {k: dict(v) for k, v in store.items()}

        def _boom(_mutator):
            raise RuntimeError("registry write failed mid-move")
        monkeypatch.setattr(mm, "update_registry", _boom)

        with pytest.raises(RuntimeError):
            mm.add_local(str(f), "moved", on_duplicate="move")

        # The file DID move (it must land under MODELS_DIR so sync can recover it).
        assert (models_dir / "m.gguf").is_file()
        # But the registry is fully pre-move: no partial 'first'-repointed state,
        # and 'moved' was never added.
        assert store == before
        assert "moved" not in store

    def test_move_crash_state_is_recovered_by_sync(self, fake_registry, tmp_path):
        """The state a crash mid-move leaves behind - the file already under
        MODELS_DIR, the registry still pointing the alias at the vanished old
        path - is reconciled by sync_models_dir, which runs on launch: the moved
        file is re-registered and the stale entry is flagged missing."""
        store, models_dir = fake_registry
        gone = tmp_path / "downloads" / "m.gguf"        # external path, file moved away
        store["first"] = {"path": str(gone.resolve()), "source": "local"}
        moved = models_dir / "m.gguf"
        moved.write_bytes(b"GGUF" + b"\x00" * 2048)     # the moved file, now in MODELS_DIR
        _backdate(moved)   # the crash happened a while ago, not mid-copy right now

        result = mm.sync_models_dir(prune=False)

        # The moved file is recovered (registered), not lost.
        assert any(e.get("path") == str(moved.resolve()) for e in store.values())
        assert result.added >= 1
        # The stale external entry is flagged missing, never silently deleted.
        assert store["first"].get("missing") is True

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

    # ---- --store move/copy checks the name collision BEFORE moving or copying
    #      the file into MODELS_DIR, and reports the collision rather than
    #      success. test_name_conflict_different_file_skipped_non_tty above
    #      covers the same collision WITHOUT --store.
    #      ---------------------------------------------------------------------
    def test_store_move_name_conflict_does_not_touch_file_non_tty(
            self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        existing = _file(tmp_path, "existing.gguf", b"one")
        mm.add_local(str(existing), "collision", on_duplicate="register")
        external = _file(tmp_path, "mymodel.gguf", b"two")

        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            result = mm.add_local(str(external), "collision", store="move")

        assert result is False, "a refused registration must not report success"
        assert external.exists(), "the source file must never be moved when " \
            "registration is known in advance to be refused"
        assert not (models_dir / "mymodel.gguf").exists(), \
            "nothing may land in MODELS_DIR for a refused non-interactive move"
        assert store["collision"]["path"] == str(existing.resolve()), \
            "the pre-existing entry must be completely untouched"
        assert not any(
            mm._entry_path(e) == str((models_dir / "mymodel.gguf").resolve())
            for e in store.values()
        ), "the moved-away file must not become an orphaned, unregistered entry"

    def test_store_copy_name_conflict_does_not_touch_file_non_tty(
            self, fake_registry, tmp_path):
        store, models_dir = fake_registry
        existing = _file(tmp_path, "existing.gguf", b"one")
        mm.add_local(str(existing), "collision", on_duplicate="register")
        external = _file(tmp_path, "mymodel.gguf", b"two")

        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            result = mm.add_local(str(external), "collision", store="copy")

        assert result is False
        assert external.exists(), "the source is untouched either way for copy"
        assert not (models_dir / "mymodel.gguf").exists(), \
            "no wasted copy should be made for a refusal known in advance"
        assert store["collision"]["path"] == str(existing.resolve())

    def test_store_move_still_succeeds_without_a_collision(
            self, fake_registry, tmp_path):
        """The pre-move gate refuses only a GENUINE name conflict, so an
        ordinary non-colliding --store move still succeeds."""
        store, models_dir = fake_registry
        external = _file(tmp_path, "mymodel.gguf")

        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            result = mm.add_local(str(external), "fresh-name", store="move")

        assert result is True
        assert not external.exists()
        assert (models_dir / "mymodel.gguf").is_file()
        assert store["fresh-name"]["path"] == str((models_dir / "mymodel.gguf").resolve())

    def test_store_move_name_conflict_interactive_decline_reports_failure(
            self, fake_registry, tmp_path):
        """Interactively declining an overwrite still leaves the file moved
        (there IS a real prompt here, unlike the non-tty case), so add_local
        reports False rather than success."""
        store, models_dir = fake_registry
        existing = _file(tmp_path, "existing.gguf", b"one")
        mm.add_local(str(existing), "collision", on_duplicate="register")
        external = _file(tmp_path, "mymodel.gguf", b"two")

        with patch("localm.model_manager.sys.stdin") as fake_stdin, \
             patch("click.confirm", return_value=False):
            fake_stdin.isatty.return_value = True
            result = mm.add_local(str(external), "collision", store="move")

        assert result is False, "a declined overwrite must not report success"
        assert store["collision"]["path"] == str(existing.resolve())


class TestNameCollision:
    """_name_collision: the shared predicate add_local's pre-move gate and
    _register_with_dedup's conflict check both use."""

    def test_no_entry_at_all_is_no_conflict(self, fake_registry, tmp_path):
        from localm.model_manager.registry import _name_collision
        f = _file(tmp_path, "m.gguf")
        assert _name_collision("free-name", f, {}) is None

    def test_same_name_same_file_is_no_conflict(self, fake_registry, tmp_path):
        from localm.model_manager.registry import _name_collision
        f = _file(tmp_path, "m.gguf")
        reg = {"m": {"path": str(f.resolve()), "source": "local"}}
        assert _name_collision("m", f, reg) is None

    def test_same_name_different_file_is_a_conflict(self, fake_registry, tmp_path):
        from localm.model_manager.registry import _name_collision
        other = _file(tmp_path, "other.gguf")
        new = _file(tmp_path, "new.gguf")
        reg = {"m": {"path": str(other.resolve()), "source": "local"}}
        assert _name_collision("m", new, reg) == str(other.resolve())


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

        # check_url resolves the host; pin it public so the SSRF guard passes, and
        # double the pinned-transport seam the pull path uses.
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            lambda method, url, **k: FakeResp())
        monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)

        mm._pull_url("https://x.test/url-model.gguf", "urlm")

        assert store["urlm"]["sha256"] == mm._sha256_file(
            models_dir / "url-model.gguf")


# ---------------------------------------------------------------------------
#  Registry read-modify-write must be atomic (no lost update on a concurrent
#  writer): load_registry() (lock, read, release) -> mutate a stale snapshot ->
#  save_registry() (lock, write) clobbers a write that landed in between.
#  update_registry() re-reads inside the lock, so nothing is lost.
# ---------------------------------------------------------------------------

class TestRegistryRmwAtomicity:
    def _racy_load(self, mm_mod, store, monkeypatch, inject):
        """Patch load_registry so a concurrent writer lands right after the FIRST
        read (the caller's initial snapshot), simulating another thread writing
        between the caller's load and its save."""
        real_load = mm_mod.load_registry
        seen = {"n": 0}

        def racy():
            snap = real_load()
            seen["n"] += 1
            if seen["n"] == 1:
                store.update(inject)      # concurrent write, after the snapshot
            return snap
        monkeypatch.setattr(mm_mod, "load_registry", racy)

    def test_relocate_no_lost_update(self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        newf = models_dir / "m.gguf"
        newf.write_bytes(b"GGUF" + b"\x00" * 2048)
        store["target"] = {"path": str(tmp_path / "gone.gguf"),
                           "source": "local", "missing": True}
        self._racy_load(mm, store, monkeypatch,
                        {"concurrent": {"path": "x", "source": "local"}})

        assert mm.relocate_model("target", str(newf)) is True
        assert "concurrent" in store, "lost update: relocate clobbered a concurrent write"
        assert store["target"]["path"] == str(newf.resolve())
        assert "missing" not in store["target"]

    def test_alias_no_lost_update(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        store["orig"] = {"path": str(tmp_path / "m.gguf"), "source": "local"}
        self._racy_load(mm, store, monkeypatch,
                        {"concurrent": {"path": "x", "source": "local"}})

        assert mm.alias_model("orig", "orig2") is True
        assert "concurrent" in store, "lost update: alias clobbered a concurrent write"
        assert store["orig2"]["path"] == store["orig"]["path"]

    def test_remove_no_lost_update(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        ext = tmp_path / "m.gguf"
        ext.write_bytes(b"GGUF")
        store["m"] = {"path": str(ext), "source": "local"}   # external -> kept, only unregistered
        self._racy_load(mm, store, monkeypatch,
                        {"concurrent": {"path": "x", "source": "local"}})

        mm.remove_model("m")
        assert "m" not in store
        assert "concurrent" in store, "lost update: remove clobbered a concurrent write"


# ---------------------------------------------------------------------------
#  Malformed registry entries must never crash a read/list/remove/dedup/sync.
#  load_registry already promises a damaged FILE never takes the app down; this
#  extends it to a damaged ENTRY.
# ---------------------------------------------------------------------------

# The shapes a hand-edited / half-written / cross-version registry.json can take.
BAD_ENTRIES = {
    "string_entry": "oops",                       # not a dict at all
    "null_entry": None,                            # null value
    "no_path": {"source": "local"},               # dict missing 'path'
    "null_path": {"path": None, "source": "local"},   # path is null
    "int_path": {"path": 123},                     # path is not a string
    "empty_path": {"path": "", "source": "local"},    # path is empty
}


class TestMalformedRegistryResilience:
    def _seed(self, store, tmp_path, bad_key, bad_val):
        good = _file(tmp_path, "good.gguf")
        store["good"] = {"path": str(good), "source": "local", "model_type": "llm"}
        store[bad_key] = bad_val
        return good

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_list_models_survives_one_bad_entry(self, fake_registry, tmp_path,
                                                bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # Must not raise; the good model is still listed and the bad entry is
        # shown as a corrupt row (never a traceback that hides the whole list).
        mm.list_models()

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_remove_drops_a_corrupt_entry(self, fake_registry, tmp_path,
                                          bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # Removing the malformed entry is the CLI recovery path: it just drops
        # the name, no crash, and leaves the good model intact.
        mm.remove_model(bad_key)
        assert bad_key not in store
        assert "good" in store

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_remove_good_model_with_a_bad_sibling(self, fake_registry, tmp_path,
                                                  bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # A bad SIBLING entry must not block removing a healthy model (the dedup
        # helper find_aliases_by_path iterates every entry).
        mm.remove_model("good")
        assert "good" not in store

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_dedup_helpers_skip_bad_entry(self, fake_registry, tmp_path,
                                          bad_key, bad_val):
        store, _ = fake_registry
        good = self._seed(store, tmp_path, bad_key, bad_val)
        # None of the identity/dedup scans (run by add/pull) may crash.
        assert mm.find_aliases_by_path(good) == ["good"]
        assert mm.find_by_sha256("deadbeef") == []
        assert mm.find_by_size(good.stat().st_size) == ["good"]

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_get_model_info_on_bad_named_entry(self, fake_registry, tmp_path,
                                               bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # Resolving the malformed name falls through to path resolution and
        # returns None instead of crashing run/benchmark/serve.
        assert mm.get_model_info(bad_key) is None

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_sync_models_dir_survives_bad_entry(self, fake_registry, tmp_path,
                                                bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # The launch-time reconcile scan must tolerate a malformed entry: it
        # leaves it in place (shown corrupt by list) rather than crashing.
        result = mm.sync_models_dir()
        assert isinstance(result, mm.ModelSyncResult)
        assert bad_key in store   # not silently dropped by sync

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_vision_and_external_helpers_survive(self, fake_registry, tmp_path,
                                                 bad_key, bad_val):
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        # The vision-routing scan and the external-path check must not crash on a
        # malformed entry (a str entry's .get would raise AttributeError).
        assert isinstance(mm.vision_capable_models(), list)
        assert mm.model_is_external(bad_key) is False
        assert isinstance(mm.model_is_external("good"), bool)

    @pytest.mark.parametrize("bad_key,bad_val", list(BAD_ENTRIES.items()))
    def test_register_over_a_corrupt_named_entry(self, fake_registry, tmp_path,
                                                 bad_key, bad_val):
        # Registering a NEW file under the exact name of a corrupt entry must not
        # crash: _register_with_dedup's "same name, different file" conflict branch
        # guards against a non-dict entry. Non-tty -> it reports the conflict and
        # skips, leaving the corrupt entry in place (cleared with `localm rm
        # <name>`) and the good model untouched.
        store, _ = fake_registry
        self._seed(store, tmp_path, bad_key, bad_val)
        new = _file(tmp_path, "brand-new.gguf")
        with patch("localm.model_manager.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            mm._register_with_dedup(bad_key, new, "local", on_duplicate="register")
        assert bad_key in store          # not crashed, not silently overwritten
        assert "good" in store
