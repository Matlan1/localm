# SPDX-License-Identifier: AGPL-3.0-or-later
"""A degraded vectors.json survives every write, above all an UNATTENDED one.

``_load`` rejects a corrupt, stale, or non-finite vectors.json, degrades
retrieval to BM25, records WHY, and LEAVES the file on disk: it is both the only
remaining copy of those vectors and the only evidence of the fault.

The scheduled folder re-sync reaches ``_add_paths_locked`` on every ordinary
tick, so these tests pin the properties that keep it from erasing the evidence:

  * ``_add_paths_locked`` does not run a full ``_save()`` when it indexed nothing;
  * ``_save`` never DELETES a vectors.json that ``_load`` rejected - while chunks
    remain it sets the file aside instead, so it can be neither lost nor silently
    re-adopted against the wrong chunks;
  * ``resync`` REPORTS the degrade in its result, because a scheduled job's output
    is the only place an unattended run is ever seen;
  * a genuine rebuild still replaces the file and clears the degrade (the guard is
    not sticky), and an emptied collection still drops the orphan.

Plus the availability check a re-sync makes before it prunes anything: on POSIX an
unmounted mount point is an ordinary, existing, EMPTY directory, which ``is_dir()``
happily calls available.

No mocks of the thing under test: every case corrupts the real sidecar on disk and
drives the real Collection against a real tmp_path collection dir.
"""

from __future__ import annotations

import json
import os

import pytest

from localm.rag import Collection
from localm.rag.store import _REJECTED_VECTORS


def _embed(texts):
    """A deterministic 3-dim fake embedder."""
    return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def base(tmp_path):
    """A collections root under tmp_path (never the user's real rag dir)."""
    d = tmp_path / "collections"
    d.mkdir()
    return d


@pytest.fixture
def docs(tmp_path):
    """An indexable folder with two documents."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alpha.txt").write_text("alpha content about turbines", encoding="utf-8")
    (d / "beta.txt").write_text("beta content about gearboxes", encoding="utf-8")
    return d


def _indexed(base, docs, name="kb", embed_fn=None):
    coll = Collection(name, base=base).create()
    coll.add_paths([docs], embed_fn=embed_fn)
    return coll


def _degraded(base, docs, name="kb"):
    """Index *docs* WITH embeddings, then drop the FIRST stored vector.

    That is the real shape of a partial/interrupted embed: a structurally valid
    vectors.json whose length no longer matches the chunk count. Dropping the
    FIRST one also leaves the survivors mis-aligned with the chunks they would be
    paired with.

    Returns a freshly loaded Collection that has already diagnosed the degrade.
    """
    _indexed(base, docs, name=name, embed_fn=_embed)
    p = base / name / "vectors.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["vectors"]) >= 2, "fixture must index at least two chunks"
    data["vectors"].pop(0)
    p.write_text(json.dumps(data), encoding="utf-8")
    fresh = Collection(name, base=base)
    assert fresh.vector_degrade_reason, "precondition: the store must reject this"
    return fresh


# --------------------------------------------------------------------------- #
#  An unattended re-sync must not erase the evidence                          #
# --------------------------------------------------------------------------- #

def test_resync_with_nothing_changed_keeps_the_degraded_vectors_file(base, docs):
    """The scheduled-job scenario: a tick where every file is unchanged. Drives
    several ticks, not one.
    """
    coll = _degraded(base, docs)
    vec = base / "kb" / "vectors.json"
    before = vec.read_bytes()

    for tick in range(3):
        result = coll.resync(embed_fn=None, policy=None)
        assert result["skipped"], f"tick {tick}: nothing was re-checked"
        assert not result["added"] and not result["updated"]
        assert vec.is_file(), f"tick {tick} deleted the degraded vectors.json"
        assert vec.read_bytes() == before, f"tick {tick} rewrote it"
        # The job's own output has to say so: _log.warning is not a report.
        assert "a partial embed" in (result["vector_degrade_reason"] or "")

    # ... and it is still diagnosable on a COLD load.
    assert "a partial embed" in (Collection("kb", base=base).vector_degrade_reason
                                 or "")


def test_manual_add_of_unchanged_paths_keeps_the_degraded_vectors_file(base, docs):
    """`rag add` over an unchanged folder also ends in an unconditional _save(),
    so the same erasure is reachable without the scheduler."""
    coll = _degraded(base, docs)
    vec = base / "kb" / "vectors.json"

    result = coll.add_paths([docs], embed_fn=None)

    assert not result["added"] and not result["updated"] and result["skipped"]
    assert vec.is_file(), "a no-op `rag add` deleted the degraded vectors.json"
    assert "a partial embed" in (Collection("kb", base=base).vector_degrade_reason
                                 or "")


def test_resync_that_indexes_a_changed_file_sets_the_file_aside(base, docs):
    """The case the 'skip the save' half does NOT cover, and the reason _save
    itself has to refuse: a run that really does index something, with no embedder
    available, still ends up with no vectors to write."""
    coll = _degraded(base, docs)
    (docs / "alpha.txt").write_text("alpha content about turbines and blades",
                                    encoding="utf-8")

    result = coll.resync(embed_fn=None, policy=None)

    assert result["updated"] == 1
    kept = base / "kb" / _REJECTED_VECTORS
    assert kept.is_file(), "the vectors were deleted instead of being set aside"
    assert json.loads(kept.read_text(encoding="utf-8"))["vectors"], "kept empty"
    # Set aside is not swept under the rug: it still reports as degraded.
    assert "set aside" in (Collection("kb", base=base).vector_degrade_reason or "")


def test_a_set_aside_index_is_never_silently_re_adopted(base, docs):
    """The file is RENAMED, not left in place.

    ``_load`` decides a sidecar is usable partly by comparing its length to the
    chunk count, so a rejected file left in place is picked up again the moment
    an unrelated change makes those counts agree, and every semantic score after
    that is computed against the wrong chunk.
    """
    coll = _degraded(base, docs)
    kept_vectors = len(json.loads(
        (base / "kb" / "vectors.json").read_text(encoding="utf-8"))["vectors"])

    coll.remove_doc(str((docs / "alpha.txt").resolve()))
    reloaded = Collection("kb", base=base)

    # The counts line up, so a file left in place would load clean and mis-score
    # every query from here on.
    assert reloaded.stats()["n_chunks"] == kept_vectors
    assert not (base / "kb" / "vectors.json").exists()
    assert (base / "kb" / _REJECTED_VECTORS).is_file()
    assert reloaded._vectors is None, "mis-paired vectors were adopted"
    assert reloaded.vector_degrade_reason


def test_rebuilding_with_an_embedder_clears_the_degrade(base, docs):
    """The guard must not be sticky: a real rebuild replaces the file and the
    collection reports healthy again."""
    coll = _degraded(base, docs)

    result = coll.add_paths([docs], embed_fn=_embed, force=True)

    assert result["updated"] == 2
    data = json.loads((base / "kb" / "vectors.json").read_text(encoding="utf-8"))
    assert len(data["vectors"]) == coll.stats()["n_chunks"]
    assert coll.vector_degrade_reason is None
    assert coll.stats()["vector_degrade_reason"] is None
    assert Collection("kb", base=base).vector_degrade_reason is None


def test_emptying_the_collection_drops_the_orphaned_vectors_file(base, tmp_path):
    """The documented exception. With no chunks left there is nothing for stored
    vectors to be realigned against, so keeping them would only pin a permanent,
    unrepairable degrade on an empty collection."""
    solo = tmp_path / "solo"
    solo.mkdir()
    (solo / "only.txt").write_text("only content about turbines", encoding="utf-8")
    coll = Collection("kb", base=base).create()
    coll.add_paths([solo], embed_fn=_embed)
    p = coll.dir / "vectors.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["vectors"].append([9.0, 9.0, 9.0])          # orphaned entry
    p.write_text(json.dumps(data), encoding="utf-8")
    coll = Collection("kb", base=base)
    assert coll.vector_degrade_reason

    coll.remove_doc(str((solo / "only.txt").resolve()))

    assert coll.stats()["n_chunks"] == 0
    assert not p.exists()
    assert not (coll.dir / _REJECTED_VECTORS).exists()


def test_absent_vectors_still_means_no_degrade(base, docs):
    """The benign case must stay benign: a collection that never had embeddings
    has no vectors.json to protect and reports no degrade. The guard has to
    distinguish 'no vectors' from 'vectors we refused'."""
    coll = _indexed(base, docs)                       # no embed_fn

    coll.resync(embed_fn=None, policy=None)

    assert not (base / "kb" / "vectors.json").exists()
    assert coll.vector_degrade_reason is None
    assert Collection("kb", base=base).vector_degrade_reason is None


def test_the_scheduled_job_output_states_the_degrade(base, docs):
    """The runner formats the result an unattended run is judged by. A degraded
    index has to appear there, not only in a log line nobody reads."""
    from localm.plugins.builtin.jobs.runner import _format_rag_result

    coll = _degraded(base, docs)
    result = coll.resync(embed_fn=None, policy=None)

    out = _format_rag_result("kb", result, [], embedded=False, had_vectors=True)

    assert "degraded" in out
    assert "a partial embed" in out
    assert "left in place, not deleted" in out
    assert "localm rag repair kb --embed" in out


# --------------------------------------------------------------------------- #
#  An unmounted mount point is not an available root                          #
# --------------------------------------------------------------------------- #

def test_an_empty_mount_point_with_indexed_docs_is_skipped_whole(base, docs,
                                                                monkeypatch):
    """A POSIX unmount leaves the mount point as a real, existing, EMPTY dir, so
    is_dir() calls it available, every document under it fails exists(), and an
    explicit --prune-missing run wipes the folder's whole index - the opposite of
    the documented "an unplugged drive cannot destroy the index".

    Windows has no unmounted-but-present mount point to create for real, so
    os.path.ismount is the ONE thing faked here; the empty folder, the vanished
    files, the prune request and the index are all real.
    """
    coll = _indexed(base, docs)
    root = str(docs.resolve())
    for f in docs.iterdir():
        f.unlink()
    monkeypatch.setattr(os.path, "ismount", lambda p: str(p) == root)

    result = coll.resync(embed_fn=None, policy=None, prune_missing=True)

    assert not result["pruned"], "an unmounted drive destroyed the index"
    assert not result["missing"], "documents under an unjudgeable root were flagged"
    assert [r["root"] for r in result["unavailable_roots"]] == [root]
    assert "unmounted" in result["unavailable_roots"][0]["reason"]
    assert coll.stats()["n_docs"] == 2, "the index was modified anyway"


def test_an_ordinary_empty_folder_still_prunes(base, docs):
    """The control that keeps the guard narrow: same empty folder, same vanished
    files, but NOT a mount point, so an explicit prune still works."""
    coll = _indexed(base, docs)
    for f in docs.iterdir():
        f.unlink()

    result = coll.resync(embed_fn=None, policy=None, prune_missing=True)

    assert len(result["pruned"]) == 2
    assert not result["unavailable_roots"]
    assert coll.stats()["n_docs"] == 0


def test_a_mounted_volume_that_still_has_files_is_indexed_normally(base, docs,
                                                                  monkeypatch):
    """The other control: being a mount point is not itself a problem. A NAS
    share or second drive that is mounted and legitimately indexed keeps
    indexing."""
    coll = _indexed(base, docs)
    monkeypatch.setattr(os.path, "ismount", lambda p: True)
    (docs / "gamma.txt").write_text("gamma content about bearings", encoding="utf-8")

    result = coll.resync(embed_fn=None, policy=None)

    assert result["added"] == 1
    assert not result["unavailable_roots"]
    assert coll.stats()["n_docs"] == 3
