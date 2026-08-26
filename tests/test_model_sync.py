# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for sync_models_dir - registry/folder reconciliation and autoprune.

This runs on every launch and can delete registry entries. Pins: loose-file
registration, missing->flagged (default), flag clearing on reappear, prune
deletion with a registry backup, the all-missing guardrail, and that external
(out-of-folder) models are never pruned.
"""

import os
import time
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


def _backdate(path, seconds=60):
    """Set *path*'s mtime `seconds` in the past, so it reads as settled (not
    mid-copy) regardless of how fast the test itself runs."""
    old = time.time() - seconds
    os.utime(path, (old, old))


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager, with a
    spy standing in for the on-disk registry backup."""
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

    backup_spy = MagicMock(return_value=None)
    monkeypatch.setattr(mm, "_backup_registry", backup_spy)
    return store, models_dir, backup_spy


def _managed_entry(models_dir, name, exists=True):
    """A registry entry whose file lives under the models folder."""
    path = models_dir / f"{name}.gguf"
    if exists:
        path.write_bytes(b"gguf-bytes-" + name.encode())
    return {"path": str(path), "source": "local"}


class TestRegisterLooseFiles:
    def test_loose_gguf_is_registered(self, fake_registry):
        store, models_dir, _ = fake_registry
        f = models_dir / "fresh.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00" + b"w" * 2048)   # magic + past the size floor
        _backdate(f)   # settled, not mid-copy
        result = mm.sync_models_dir(prune=False)
        assert result.added == 1
        assert any(e["path"].endswith("fresh.gguf") for e in store.values())

    def test_foreign_or_partial_gguf_is_not_registered(self, fake_registry):
        # A non-GGUF file renamed .gguf (or a 0-byte / partial copy with no header)
        # is not auto-registered. Only real GGUFs (magic b"GGUF" plus a plausible
        # size) register; a header-only stub with the magic is skipped.
        store, models_dir, _ = fake_registry
        foreign = models_dir / "foreign.gguf"
        empty = models_dir / "empty.gguf"
        stub = models_dir / "stub.gguf"
        real = models_dir / "real.gguf"
        foreign.write_bytes(b"this is not a model")
        empty.write_bytes(b"")
        stub.write_bytes(b"GGUF\x03\x00\x00\x00")
        real.write_bytes(b"GGUF\x03\x00\x00\x00" + b"d" * 2048)   # magic + past the floor
        for f in (foreign, empty, stub, real):
            _backdate(f)   # isolate this test to the magic/floor check, not settling
        result = mm.sync_models_dir(prune=False)
        assert result.added == 1                              # only real.gguf
        names = [e["path"] for e in store.values()]
        assert any(p.endswith("real.gguf") for p in names)
        assert not any(p.endswith("foreign.gguf") or p.endswith("empty.gguf")
                       or p.endswith("stub.gguf") for p in names)


class TestSettlePeriod:
    """A mid-copy of a *valid* GGUF clears the magic+size floor long before the
    copy finishes (the floor only needs ~1KiB to have landed), so sync_models_dir
    must not auto-register it until its mtime has gone quiet."""

    def test_freshly_written_gguf_is_not_registered_yet(self, fake_registry):
        store, models_dir, _ = fake_registry
        f = models_dir / "copying.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00" + b"c" * 2048)   # magic + past the floor
        # mtime is "now" - written this instant, exactly like the tail end of
        # an in-progress copy. Must NOT register on this call.
        result = mm.sync_models_dir(prune=False)
        assert result.added == 0
        assert not store

    def test_same_file_registers_once_it_has_settled(self, fake_registry):
        store, models_dir, _ = fake_registry
        f = models_dir / "copying.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00" + b"c" * 2048)
        first = mm.sync_models_dir(prune=False)
        assert first.added == 0                # still looks mid-copy
        assert not store

        _backdate(f)                            # simulate the copy going quiet
        second = mm.sync_models_dir(prune=False)
        assert second.added == 1
        assert any(e["path"].endswith("copying.gguf") for e in store.values())


class TestMissingFlagging:
    def test_missing_managed_is_flagged_not_deleted(self, fake_registry):
        store, models_dir, backup = fake_registry
        store["m"] = _managed_entry(models_dir, "gone", exists=False)
        result = mm.sync_models_dir(prune=False)
        assert result.flagged == 1 and result.pruned == 0
        assert store["m"]["missing"] is True       # kept, just flagged
        backup.assert_not_called()

    def test_reappeared_file_clears_flag(self, fake_registry):
        store, models_dir, _ = fake_registry
        entry = _managed_entry(models_dir, "back", exists=True)
        entry["missing"] = True
        store["m"] = entry
        result = mm.sync_models_dir(prune=False)
        assert result.restored == 1
        assert "missing" not in store["m"]


class TestAutoprune:
    def test_prune_deletes_missing_and_backs_up(self, fake_registry):
        store, models_dir, backup = fake_registry
        store["keep"] = _managed_entry(models_dir, "keep", exists=True)
        store["gone"] = _managed_entry(models_dir, "gone", exists=False)
        result = mm.sync_models_dir(prune=True)
        assert result.pruned == 1
        assert "gone" not in store and "keep" in store
        backup.assert_called_once()               # snapshot taken before deleting

    def test_single_missing_is_pruned(self, fake_registry):
        # The guardrail only trips for >= 2 managed models all missing at once.
        store, models_dir, _ = fake_registry
        store["only"] = _managed_entry(models_dir, "only", exists=False)
        result = mm.sync_models_dir(prune=True)
        assert result.pruned == 1 and "only" not in store

    def test_all_missing_guardrail_refuses_to_prune(self, fake_registry):
        store, models_dir, backup = fake_registry
        store["a"] = _managed_entry(models_dir, "a", exists=False)
        store["b"] = _managed_entry(models_dir, "b", exists=False)
        result = mm.sync_models_dir(prune=True)
        assert result.pruned == 0
        assert result.flagged == 2 and result.note
        assert "a" in store and "b" in store       # nothing deleted
        backup.assert_not_called()

    def test_external_model_never_pruned(self, fake_registry, tmp_path):
        store, models_dir, _ = fake_registry
        # A registered model whose file lives OUTSIDE the models folder, missing.
        store["ext"] = {"path": str(tmp_path / "elsewhere" / "ext.gguf"),
                        "source": "local"}
        result = mm.sync_models_dir(prune=True)
        assert result.pruned == 0 and result.flagged == 1
        assert store["ext"]["missing"] is True
