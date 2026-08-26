# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for rename_model: registry MOVE (not copy, unlike alias_model), sibling
aliases left untouched, and best-effort migration of the config/jobs/RAG
references that name the old model. Mirrors the sibling
alias_model/registry-RMW test patterns.
"""

import json
from pathlib import Path

import pytest

from localm import model_manager as mm


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager (registry
    only - config/jobs/RAG are untouched by this fixture, see real_home for
    tests that need the migration side effects too)."""
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


@pytest.fixture()
def real_home(tmp_path, monkeypatch):
    """A full isolated <data dir>: registry.json AND config.json are real
    files under a throwaway home, and the lazy home_dir() jobs.json/RAG use
    resolve there too (LOCALM_HOME env). Unlike fake_registry, this lets
    rename_model's migration step (update_config, JobStore, rag Collection -
    all real file I/O) run fully hermetically. Mirrors conftest.py's
    cli_runner fixture without the CliRunner overhead."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _file(tmp_path: Path, name: str, content: bytes = b"model-bytes") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
#  rename_model core mechanics
# ---------------------------------------------------------------------------

class TestRenameModel:
    def test_moves_not_copies(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local", "sha256": "abc"}
        assert mm.rename_model("orig", "renamed") is True
        assert "orig" not in store, "rename must MOVE the key, not copy it (that is alias)"
        assert store["renamed"]["path"] == str(f)
        assert store["renamed"]["sha256"] == "abc"

    def test_unknown_old_name_fails(self, fake_registry):
        assert mm.rename_model("ghost", "x") is False

    def test_taken_name_fails_and_leaves_both_entries_intact(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        store["b"] = {"path": "q", "source": "local"}
        assert mm.rename_model("a", "b") is False
        assert store["a"]["path"] == "p", "a failed rename must not touch the old entry"
        assert store["b"]["path"] == "q"

    def test_self_rename_is_a_noop_success(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        assert mm.rename_model("a", "a") is True
        assert store["a"]["path"] == "p"
        assert list(store.keys()) == ["a"]

    def test_self_rename_after_sanitizing_is_a_noop_success(self, fake_registry):
        # "a " sanitizes to "a" - the SAME key the entry is already under, not
        # a distinct name that happens to collide.
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        assert mm.rename_model("a", "a ") is True
        assert store["a"]["path"] == "p"
        assert list(store.keys()) == ["a"]

    def test_sanitizes_new_name(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local"}
        assert mm.rename_model("orig", "daily driver") is True
        assert "daily driver" not in store
        assert "daily-driver" in store
        assert "orig" not in store

    def test_sibling_alias_left_untouched(self, fake_registry, tmp_path):
        # alias_model creates an INDEPENDENT entry pointing at the same file;
        # renaming ONE of the two names must not delete or repoint the other.
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local"}
        assert mm.alias_model("orig", "sibling") is True
        assert mm.rename_model("orig", "renamed") is True
        assert "orig" not in store
        assert "renamed" in store
        assert "sibling" in store, "renaming one name must not remove a sibling alias"
        assert store["sibling"]["path"] == str(f)
        assert store["renamed"]["path"] == str(f)


class TestRenameModelWithNotes:
    """rename_model() is a thin bool-only wrapper over this; a caller that
    needs to SHOW the migration notes to a user (the GUI route) must call
    this instead, or the honest "here is what could not be migrated" report
    never leaves the server log."""

    def test_successful_rename_always_reports_the_unreachable_localcoder_note(
            self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local"}
        ok, notes = mm.rename_model_with_notes("orig", "renamed")
        assert ok is True
        assert any(".localcoder" in n for n in notes), (
            "the caller-facing notes must always carry this caveat, not just "
            "the server console rename_model_with_notes also prints to")

    def test_failure_returns_no_notes(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        store["b"] = {"path": "q", "source": "local"}
        ok, notes = mm.rename_model_with_notes("a", "b")
        assert ok is False
        assert notes == []

    def test_self_rename_noop_returns_no_notes(self, fake_registry):
        store, _ = fake_registry
        store["a"] = {"path": "p", "source": "local"}
        ok, notes = mm.rename_model_with_notes("a", "a")
        assert ok is True
        assert notes == []

    def test_rename_model_bool_wrapper_matches_the_notes_version(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = _file(tmp_path, "m.gguf")
        store["orig"] = {"path": str(f), "source": "local"}
        assert mm.rename_model("orig", "renamed") is True
        assert "renamed" in store


class TestRenameModelRace:
    """Same atomic-RMW discipline as the registry dedup tests: a concurrent
    writer landing between the precheck read and the atomic move must never be
    lost."""

    def _racy_load(self, mm_mod, store, monkeypatch, inject):
        real_load = mm_mod.load_registry
        seen = {"n": 0}

        def racy():
            snap = real_load()
            seen["n"] += 1
            if seen["n"] == 1:
                store.update(inject)      # concurrent write, after the snapshot
            return snap
        monkeypatch.setattr(mm_mod, "load_registry", racy)

    def test_rename_no_lost_update(self, fake_registry, tmp_path, monkeypatch):
        store, _ = fake_registry
        store["orig"] = {"path": str(tmp_path / "m.gguf"), "source": "local"}
        self._racy_load(mm, store, monkeypatch,
                        {"concurrent": {"path": "x", "source": "local"}})

        assert mm.rename_model("orig", "orig2") is True
        assert "concurrent" in store, "lost update: rename clobbered a concurrent write"
        assert "orig" not in store
        assert store["orig2"]["path"] == str(tmp_path / "m.gguf")

    def test_rename_honestly_reports_a_lost_race(self, fake_registry, monkeypatch):
        # The precheck sees taken is free, but a concurrent writer claims it (for
        # an unrelated entry) before the atomic move runs. rename_model must
        # notice the move never happened and return False rather than claim
        # success on a step that silently did nothing.
        store, _ = fake_registry
        store["orig"] = {"path": "p", "source": "local"}
        self._racy_load(mm, store, monkeypatch,
                        {"taken": {"path": "q", "source": "local"}})
        assert mm.rename_model("orig", "taken") is False
        assert store["orig"]["path"] == "p", "a lost race must leave the old entry alone"
        assert store["taken"]["path"] == "q", "the concurrent writer's entry must survive"


# ---------------------------------------------------------------------------
#  Reference migration (config.json, jobs.json, RAG collection meta)
# ---------------------------------------------------------------------------

class TestRenameModelMigratesReferences:
    def test_migrates_config_jobs_and_rag(self, real_home, tmp_path):
        from localm.config import load_config, save_config
        f = tmp_path / "m.gguf"
        f.write_bytes(b"model-bytes")
        mm.add_local(str(f), "orig", on_duplicate="register")

        cfg = load_config()
        cfg["pinned_models"] = ["orig", "other"]
        cfg["embedding_model"] = "orig"
        cfg["coder_reviewer_model"] = "orig"
        save_config(cfg)

        from localm.plugins.builtin.jobs.store import Job, JobStore
        store = JobStore()
        job = store.add(Job(name="j1", task_kind="chat", prompt="hi", model="orig"))
        # A job naming a DIFFERENT model must be left alone.
        other_job = store.add(Job(name="j2", task_kind="chat", prompt="hi", model="other"))

        coll_dir = real_home / "rag" / "mycoll"
        coll_dir.mkdir(parents=True)
        (coll_dir / "meta.json").write_text(
            json.dumps({"name": "mycoll", "docs": {}, "embedding_model": "orig"}),
            encoding="utf-8")

        assert mm.rename_model("orig", "renamed") is True

        new_cfg = load_config()
        assert new_cfg["pinned_models"] == ["renamed", "other"]
        assert new_cfg["embedding_model"] == "renamed"
        assert new_cfg["coder_reviewer_model"] == "renamed"

        assert store.get(job.id).model == "renamed"
        assert store.get(other_job.id).model == "other"

        from localm.rag.store import Collection
        coll = Collection("mycoll")
        assert coll.embedding_model() == "renamed"

    def test_migration_never_touches_an_unrelated_pin(self, real_home, tmp_path):
        from localm.config import load_config, save_config
        f = tmp_path / "m.gguf"
        f.write_bytes(b"model-bytes")
        mm.add_local(str(f), "orig", on_duplicate="register")

        cfg = load_config()
        cfg["pinned_models"] = ["someone-else"]
        save_config(cfg)

        assert mm.rename_model("orig", "renamed") is True
        assert load_config()["pinned_models"] == ["someone-else"]

    def test_migrate_references_reports_the_unreachable_localcoder_case(self, real_home):
        from localm.model_manager import registry as reg_mod
        notes = reg_mod._migrate_model_references("orig", "renamed")
        assert any(".localcoder" in n for n in notes), (
            "a rename must say plainly that a per-project .localcoder/config.toml "
            "cannot be reached, never silently drop it")


# ---------------------------------------------------------------------------
#  `localm rename` CLI command, driven end to end through conftest.py's
#  cli_runner
# ---------------------------------------------------------------------------

class TestRenameCliCommand:
    def test_rename_command_moves_the_entry(self, cli_runner, tmp_path):
        from localm.cli import main
        from localm import model_manager as mm

        f = tmp_path / "m.gguf"
        f.write_bytes(b"model-bytes")
        result = cli_runner.invoke(main, ["add", str(f), "-n", "orig"])
        assert result.exit_code == 0, result.output

        result = cli_runner.invoke(main, ["rename", "orig", "daily driver"])
        assert result.exit_code == 0, result.output

        reg = mm.load_registry()
        assert "orig" not in reg
        assert "daily-driver" in reg

    def test_rename_command_exits_nonzero_on_unknown_model(self, cli_runner):
        from localm.cli import main
        result = cli_runner.invoke(main, ["rename", "ghost", "whatever"])
        assert result.exit_code != 0
