# SPDX-License-Identifier: AGPL-3.0-or-later
"""REG-631 (regression audit 2026-07-14): the gallery ownership index's atomic write had no retry and no temp cleanup, so on Windows an external open of the index file failed the whole request and orphaned a temp file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _flaky_replace(monkeypatch, fail_times: int):
    """Make the atomic swap raise PermissionError the first *fail_times* calls, standing in for an AV / Search-Indexer handle on the index file."""
    calls = {"n": 0}
    real_replace = os.replace

    def _replace(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise PermissionError(5, "The process cannot access the file because "
                                     "it is being used by another process")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", _replace)
    return calls


def _temps(index_path: Path):
    return [p for p in index_path.parent.iterdir() if p != index_path]


# --------------------------------------------------------------------------- #
#  THE REGRESSION: a transient external lock must not fail the write           #
# --------------------------------------------------------------------------- #

def test_write_index_rides_out_a_transient_external_lock(home, monkeypatch):
    from localm.media import gallery
    calls = _flaky_replace(monkeypatch, fail_times=2)

    gallery._write_index("image", {"a.png": "owner1"})

    assert calls["n"] >= 3, "the replace was not retried past the transient lock"
    assert gallery._read_index("image") == {"a.png": "owner1"}
    assert _temps(gallery._index_path("image")) == []


def test_stamp_owner_survives_a_transient_lock(home, monkeypatch):
    """End-to-end through the real caller: a generation's owner stamp must land even though an AV scanner briefly held the index open."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=2)

    gallery.stamp_owner("image", "new.png", "ownerX")

    assert gallery.owner_of("image", "new.png") == "ownerX"
    assert _temps(gallery._index_path("image")) == []


def test_forget_owner_survives_a_transient_lock(home, monkeypatch):
    from localm.media import gallery
    gallery.stamp_owner("image", "a.png", "ownerX")
    _flaky_replace(monkeypatch, fail_times=2)

    gallery.forget_owner("image", "a.png")

    assert gallery.owner_of("image", "a.png") is None
    assert _temps(gallery._index_path("image")) == []


def test_rename_owner_survives_a_transient_lock(home, monkeypatch):
    from localm.media import gallery
    gallery.stamp_owner("image", "a.png", "ownerX")
    _flaky_replace(monkeypatch, fail_times=2)

    gallery.rename_owner("image", "a.png", "b.png")

    assert gallery.owner_of("image", "b.png") == "ownerX"
    assert _temps(gallery._index_path("image")) == []


# --------------------------------------------------------------------------- #
#  Temp cleanup: a give-up must not leave litter behind                        #
# --------------------------------------------------------------------------- #

def test_a_permanent_lock_leaves_no_orphan_temp(home, monkeypatch):
    """When the lock never clears, the write legitimately fails - but it must not accumulate one orphaned temp file per failure in gallery_index/."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)

    with pytest.raises(PermissionError):
        gallery._write_index("image", {"a.png": "owner1"})

    idx = gallery._index_path("image")
    assert _temps(idx) == [], "a give-up orphaned its temp file"


def test_repeated_permanent_failures_do_not_accumulate_temps(home, monkeypatch):
    """The audit's 'one leftover per failure': prove it does not pile up."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)

    for _ in range(5):
        with pytest.raises(PermissionError):
            gallery._write_index("image", {"a.png": "owner1"})

    assert _temps(gallery._index_path("image")) == []


def test_a_permanent_lock_still_raises_not_silently_swallowed(home, monkeypatch):
    """Rule 5: riding out a TRANSIENT lock is right; silently pretending a permanently-failed write succeeded is not."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)
    with pytest.raises(PermissionError):
        gallery._write_index("image", {"a.png": "owner1"})


# --------------------------------------------------------------------------- #
#  The atomicity / correctness the write already had must survive              #
# --------------------------------------------------------------------------- #

def test_success_path_still_leaves_no_temp(home):
    from localm.media import gallery
    gallery._write_index("image", {"a.png": "owner1"})
    assert gallery._read_index("image") == {"a.png": "owner1"}
    assert _temps(gallery._index_path("image")) == []


def test_write_index_overwrites_cleanly(home):
    from localm.media import gallery
    gallery._write_index("image", {"a.png": "owner1"})
    gallery._write_index("image", {"b.png": "owner2"})
    assert gallery._read_index("image") == {"b.png": "owner2"}
    assert _temps(gallery._index_path("image")) == []


def test_a_failed_write_does_not_corrupt_the_existing_index(home, monkeypatch):
    """The whole point of the atomic swap: a failed write must leave the previous index intact and readable, never half-written."""
    from localm.media import gallery
    gallery._write_index("image", {"a.png": "owner1"})
    _flaky_replace(monkeypatch, fail_times=10_000)

    with pytest.raises(PermissionError):
        gallery._write_index("image", {"b.png": "owner2"})

    # _read_index only reads, so the patched os.replace is not in its path - no
    # monkeypatch.undo() (which would also revert the home fixture's config
    # patches and send this read at the real data dir).
    assert gallery._read_index("image") == {"a.png": "owner1"}
