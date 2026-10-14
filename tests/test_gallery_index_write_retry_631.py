# SPDX-License-Identifier: AGPL-3.0-or-later
"""The gallery ownership index's atomic write needs a bounded retry and temp
cleanup, or on Windows an external open of the index file fails the whole request
and orphans a temp file.

``gallery._write_index`` (called from ``stamp_owner`` on EVERY image/music/video
generation, and from ``forget_owner`` / ``rename_owner`` on delete/move/rename)
writes a unique temp then calls ``os.replace(tmp, p)``. If any external process
holds ``gallery_index/<kind>.json`` open WITHOUT FILE_SHARE_DELETE at that
instant - an antivirus mid-scan, the Windows Search Indexer, a backup agent, or
the user's own file browser - os.replace raises PermissionError (WinError 5/32).
Without a bounded retry and a try/finally:

 (a) the exception propagates and 500s the request even though the media file is
     already written to disk, so the caller believes generation/delete failed
     while the new file sits there un-stamped (hence untracked, and by
     owner_of()'s untracked default, open); and
 (b) the unique temp file is orphaned in gallery_index/, one per failure.

_write_index therefore delegates to ``storekit.atomic_write``, the shared kernel
helper with the bounded PermissionError retry that rag/store.py and
memory/store.py also use, and the temp cleanup lives in storekit so every caller
gets it.
"""

from __future__ import annotations

import json
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
    """Make the atomic swap raise PermissionError the first *fail_times* calls,
    standing in for an AV / Search-Indexer handle on the index file. Counts every
    attempt so a test can prove the retry actually happened.

    Patches ``os.replace``, deliberately: that is the one chokepoint BOTH spellings
    go through (a bare ``os.replace(tmp, p)`` and ``Path.replace``, which calls
    os.replace internally). Patching Path.replace alone would silently miss a
    caller using the bare form and the test would prove nothing.
    """
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
#  A transient external lock must not fail the write                           #
# --------------------------------------------------------------------------- #

def test_write_index_rides_out_a_transient_external_lock(home, monkeypatch):
    from localm.media import gallery
    calls = _flaky_replace(monkeypatch, fail_times=2)

    gallery._write_index("image", {"a.png": "owner1"})

    assert calls["n"] >= 3, "the replace was not retried past the transient lock"
    assert gallery._read_index("image") == {"a.png": "owner1"}
    assert _temps(gallery._index_path("image")) == []


def test_stamp_owner_survives_a_transient_lock(home, monkeypatch):
    """End-to-end through the real caller: a generation's owner stamp must land
    even though an AV scanner briefly held the index open. Otherwise the media
    file is on disk but untracked - and owner_of() defaults untracked to open."""
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
    """When the lock never clears, the write legitimately fails - but it must not
    accumulate one orphaned temp file per failure in gallery_index/."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)

    with pytest.raises(PermissionError):
        gallery._write_index("image", {"a.png": "owner1"})

    idx = gallery._index_path("image")
    assert _temps(idx) == [], "a give-up orphaned its temp file"


def test_repeated_permanent_failures_do_not_accumulate_temps(home, monkeypatch):
    """The audit's "one leftover per failure": prove it does not pile up."""
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)

    for _ in range(5):
        with pytest.raises(PermissionError):
            gallery._write_index("image", {"a.png": "owner1"})

    assert _temps(gallery._index_path("image")) == []


def test_a_permanent_lock_still_raises_not_silently_swallowed(home, monkeypatch):
    """Rule 5: riding out a TRANSIENT lock is right; silently pretending a
    permanently-failed write succeeded is not. It must still raise."""
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
    """The whole point of the atomic swap: a failed write must leave the previous
    index intact and readable, never half-written."""
    from localm.media import gallery
    gallery._write_index("image", {"a.png": "owner1"})
    _flaky_replace(monkeypatch, fail_times=10_000)

    with pytest.raises(PermissionError):
        gallery._write_index("image", {"b.png": "owner2"})

    # _read_index only reads, so the patched os.replace is not in its path - no
    # monkeypatch.undo() (which would also revert the home fixture's config
    # patches and send this read at the real data dir).
    assert gallery._read_index("image") == {"a.png": "owner1"}



# --------------------------------------------------------------------------- #
#  The READ half of the same calls: a lock on the index read                   #
# --------------------------------------------------------------------------- #

_REAL_READ_TEXT = Path.read_text


def _flaky_index_read(monkeypatch, index_path: Path, fail_times: int):
    """Make ``Path.read_text`` raise PermissionError for *index_path* only, the
    first *fail_times* calls. Counts the index reads."""
    calls = {"n": 0}
    real_read_text = Path.read_text
    target = Path(index_path)

    def _read_text(self, *a, **kw):
        if Path(self) == target:
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise PermissionError(32, "The process cannot access the file because "
                                          "it is being used by another process")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _read_text)
    return calls


def test_stamp_owner_survives_a_transient_read_lock(home, monkeypatch):
    from localm.media import gallery
    gallery.stamp_owner("image", "old.png", "ownerA")
    calls = _flaky_index_read(monkeypatch, gallery._index_path("image"), fail_times=2)

    assert gallery.stamp_owner("image", "new.png", "ownerX") is None

    assert calls["n"] >= 3, "the index read was not retried past the transient lock"
    assert json.loads(gallery._index_path("image").read_text(encoding="utf-8")) == {
        "old.png": "ownerA", "new.png": "ownerX"}


def test_owner_of_survives_a_transient_read_lock(home, monkeypatch):
    from localm.media import gallery
    gallery.stamp_owner("image", "a.png", "ownerA")
    _flaky_index_read(monkeypatch, gallery._index_path("image"), fail_times=2)

    assert gallery.owner_of("image", "a.png") == "ownerA"


def test_a_stamp_that_cannot_read_the_index_is_held_reported_and_written_later(
        home, monkeypatch):
    from localm.media import gallery
    idx_path = gallery._index_path("image")
    gallery.stamp_owner("image", "old.png", "ownerA")
    _flaky_index_read(monkeypatch, idx_path, fail_times=10_000)

    warning = gallery.stamp_owner("image", "new.png", "ownerX")

    assert warning and "new.png" in warning and "owner" in warning
    monkeypatch.setattr(Path, "read_text", _REAL_READ_TEXT)
    assert gallery.owner_of("image", "new.png") == "ownerX", (
        "the failed stamp left the artifact unowned, which owner_of treats as open")
    assert "new.png" not in json.loads(idx_path.read_text(encoding="utf-8"))

    assert gallery.stamp_owner("image", "next.png", "ownerY") is None
    assert json.loads(idx_path.read_text(encoding="utf-8")) == {
        "old.png": "ownerA", "new.png": "ownerX", "next.png": "ownerY"}


def test_a_stamp_against_a_corrupt_index_is_held_and_survives_the_repair(
        home, monkeypatch):
    from localm.media import gallery
    idx_path = gallery._index_path("image")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text("{ not valid json", encoding="utf-8")

    warning = gallery.stamp_owner("image", "new.png", "ownerX")

    assert warning and "new.png" in warning
    assert idx_path.read_text(encoding="utf-8") == "{ not valid json"
    with pytest.raises(gallery.GalleryIndexUnreadable):
        gallery.owner_of("image", "new.png")
    idx_path.write_text("{}", encoding="utf-8")
    assert gallery.owner_of("image", "new.png") == "ownerX"


def test_a_stamp_whose_write_never_lands_is_held_and_reported(home, monkeypatch):
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)

    warning = gallery.stamp_owner("image", "new.png", "ownerX")

    assert warning and "new.png" in warning
    assert gallery.owner_of("image", "new.png") == "ownerX"
    assert _temps(gallery._index_path("image")) == []


def test_a_held_stamp_follows_rename_and_forget(home, monkeypatch):
    from localm.media import gallery
    idx_path = gallery._index_path("image")
    gallery.stamp_owner("image", "seed.png", "ownerA")
    _flaky_index_read(monkeypatch, idx_path, fail_times=10_000)
    assert gallery.stamp_owner("image", "a.png", "ownerX")
    assert gallery.stamp_owner("image", "gone.png", "ownerX")
    monkeypatch.setattr(Path, "read_text", _REAL_READ_TEXT)

    gallery.rename_owner("image", "a.png", "b.png")
    gallery.forget_owner("image", "gone.png")

    assert json.loads(idx_path.read_text(encoding="utf-8")) == {
        "seed.png": "ownerA", "b.png": "ownerX"}
    assert gallery.owner_of("image", "a.png") is None
    assert gallery.owner_of("image", "gone.png") is None


def test_held_stamps_do_not_leak_into_another_home(home, monkeypatch, tmp_path):
    from localm.media import gallery
    _flaky_replace(monkeypatch, fail_times=10_000)
    assert gallery.stamp_owner("image", "new.png", "ownerX")
    other = tmp_path / "other-home"
    other.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(other))

    assert gallery.owner_of("image", "new.png") is None

