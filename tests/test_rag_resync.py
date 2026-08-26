# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scheduled RAG folder re-sync: persisted roots + Collection.resync().

The point of the feature is drift a plain re-index CANNOT see, so these tests
mutate a real folder on disk between runs and assert against the real index:

  * a file ADDED to an indexed folder after the initial index is picked up;
  * a file DELETED from it is FLAGGED, not dropped (the documented semantics),
    and its chunks stay searchable;
  * the flag clears by itself when the file comes back;
  * deletion happens only with the explicit prune opt-in;
  * an unreachable root (unplugged drive / unmounted share) is skipped WHOLE -
    nothing under it is indexed, flagged, or pruned;
  * the confinement policy is applied to a scheduled re-sync exactly as it is to
    an interactive add;
  * roots survive a store reload.

No mocks of the thing under test: every case drives Collection.add_paths /
Collection.resync against a tmp_path collection dir.
"""

from __future__ import annotations

import json

import pytest

from localm.rag import Collection


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


def _indexed(base, docs, name="kb"):
    coll = Collection(name, base=base)
    coll.create()
    coll.add_paths([docs])
    return coll


def _sources(coll):
    return {p for p in coll.documents()}


# --------------------------------------------------------------------------- #
#  Persisted roots                                                             #
# --------------------------------------------------------------------------- #

def test_folder_add_records_the_root(base, docs):
    coll = _indexed(base, docs)
    assert coll.roots() == [str(docs.resolve())]


def test_roots_survive_a_store_reload(base, docs):
    _indexed(base, docs)
    # A brand new Collection instance reads meta.json from disk.
    reloaded = Collection("kb", base=base)
    assert reloaded.roots() == [str(docs.resolve())]
    # And it is really persisted, not reconstructed in memory.
    meta = json.loads((base / "kb" / "meta.json").read_text(encoding="utf-8"))
    assert list(meta["roots"]) == [str(docs.resolve())]


def test_individually_added_file_is_not_recorded_as_a_root(base, tmp_path):
    """Adding ONE file must not silently enlist its whole parent folder - that
    would index every sibling on the next re-sync."""
    f = tmp_path / "loose" / "note.txt"
    f.parent.mkdir()
    f.write_text("a single note", encoding="utf-8")
    coll = Collection("solo", base=base)
    coll.create()
    coll.add_paths([f])
    assert coll.roots() == []
    assert len(coll.documents()) == 1


def test_empty_folder_is_still_recorded_as_a_root(base, tmp_path):
    """An add that indexes nothing must still record the folder, or the first
    document dropped into it tomorrow is invisible forever."""
    empty = tmp_path / "empty"
    empty.mkdir()
    coll = Collection("kb", base=base)
    coll.create()
    result = coll.add_paths([empty])
    assert result["added"] == 0
    assert Collection("kb", base=base).roots() == [str(empty.resolve())]


# --------------------------------------------------------------------------- #
#  Added / changed files                                                       #
# --------------------------------------------------------------------------- #

def test_resync_picks_up_a_file_added_after_the_initial_index(base, docs):
    coll = _indexed(base, docs)
    assert len(coll.documents()) == 2

    (docs / "gamma.txt").write_text("gamma content about flywheels",
                                    encoding="utf-8")
    result = coll.resync()

    assert result["added"] == 1
    assert result["skipped"] == 2           # the incremental hash skip, reused
    assert str((docs / "gamma.txt").resolve()) in _sources(coll)
    # The new document is really searchable, not merely listed.
    hits = Collection("kb", base=base).query("flywheels", k=3)
    assert any("flywheel" in h["text"] for h in hits)


def test_resync_reindexes_a_changed_file(base, docs):
    coll = _indexed(base, docs)
    (docs / "alpha.txt").write_text("alpha now discusses hydraulics",
                                    encoding="utf-8")
    result = coll.resync()
    assert result["updated"] == 1
    hits = Collection("kb", base=base).query("hydraulics", k=3)
    assert hits and "hydraulics" in hits[0]["text"]


def test_resync_of_an_unchanged_folder_changes_nothing(base, docs):
    coll = _indexed(base, docs)
    result = coll.resync()
    assert (result["added"], result["updated"]) == (0, 0)
    assert result["skipped"] == 2
    assert result["missing"] == [] and result["pruned"] == []


def test_resync_refreshes_an_individually_added_file_too(base, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("first version", encoding="utf-8")
    coll = Collection("solo", base=base)
    coll.create()
    coll.add_paths([f])
    f.write_text("second version, mentioning kingfishers", encoding="utf-8")

    result = coll.resync()
    assert result["updated"] == 1
    hits = Collection("solo", base=base).query("kingfishers", k=3)
    assert hits and "kingfishers" in hits[0]["text"]


# --------------------------------------------------------------------------- #
#  Deleted files: FLAGGED, never silently dropped                              #
# --------------------------------------------------------------------------- #

def test_deleted_file_is_flagged_not_removed(base, docs):
    coll = _indexed(base, docs)
    gone = str((docs / "beta.txt").resolve())
    (docs / "beta.txt").unlink()

    result = coll.resync()

    assert result["missing"] == [gone]
    assert result["pruned"] == []
    # Still indexed, still counted, still SEARCHABLE - the flag records that the
    # index is ahead of the disk, it does not delete anything.
    reloaded = Collection("kb", base=base)
    assert gone in _sources(reloaded)
    assert reloaded.stats()["n_missing"] == 1
    assert reloaded.query("gearboxes", k=3)
    entry = next(d for d in reloaded.docs() if d["path"] == gone)
    assert entry["missing"] is True and entry["missing_since"] > 0


def test_flagging_is_idempotent_across_runs(base, docs):
    coll = _indexed(base, docs)
    (docs / "beta.txt").unlink()
    first = coll.resync()
    second = coll.resync()
    assert len(first["missing"]) == 1
    assert second["missing"] == []          # already flagged, not re-reported
    assert second["missing_total"] == 1     # but still honestly counted


def test_a_returning_file_clears_its_missing_flag(base, docs):
    coll = _indexed(base, docs)
    path = docs / "beta.txt"
    body = path.read_text(encoding="utf-8")
    path.unlink()
    coll.resync()
    assert Collection("kb", base=base).stats()["n_missing"] == 1

    path.write_text(body, encoding="utf-8")     # the drive came back
    result = coll.resync()

    assert result["restored"] == [str(path.resolve())]
    reloaded = Collection("kb", base=base)
    assert reloaded.stats()["n_missing"] == 0
    assert "missing" not in next(d for d in reloaded.docs()
                                 if d["path"] == str(path.resolve()))


def test_a_file_that_comes_back_changed_is_still_reported_restored(base, docs):
    """The re-index rewrites the doc entry and drops the flag as a side effect,
    so the run must not lose track of the fact that the file had been missing."""
    coll = _indexed(base, docs)
    path = docs / "beta.txt"
    path.unlink()
    coll.resync()

    path.write_text("beta came back, now about camshafts", encoding="utf-8")
    result = coll.resync()

    assert result["restored"] == [str(path.resolve())]
    assert result["updated"] == 1
    reloaded = Collection("kb", base=base)
    assert reloaded.stats()["n_missing"] == 0
    hits = reloaded.query("camshafts", k=3)
    assert hits and "camshafts" in hits[0]["text"]


def test_prune_missing_is_opt_in(base, docs):
    coll = _indexed(base, docs)
    gone = str((docs / "beta.txt").resolve())
    (docs / "beta.txt").unlink()

    result = coll.resync(prune_missing=True)

    assert result["pruned"] == [gone]
    assert result["missing"] == []
    reloaded = Collection("kb", base=base)
    assert gone not in _sources(reloaded)
    # Its chunks went with it, and only its chunks.
    assert not reloaded.query("gearboxes", k=3)
    assert reloaded.query("turbines", k=3)


def test_an_upload_doc_is_never_reported_missing(base, docs):
    """upload: docs have no source file by design - they must not be flagged."""
    coll = _indexed(base, docs)
    coll.add_uploads([{"filename": "pasted.txt", "data": b"uploaded content"}])
    result = coll.resync()
    assert result["missing"] == [] and result["pruned"] == []
    assert "upload:pasted.txt" in _sources(Collection("kb", base=base))


# --------------------------------------------------------------------------- #
#  Unreachable root: the transient-condition guard                             #
# --------------------------------------------------------------------------- #

def test_an_unavailable_root_touches_nothing(base, docs):
    """A folder that is gone at re-sync time (unplugged drive, unmounted share)
    must NOT be read as 'every file under it was deleted'."""
    import shutil
    coll = _indexed(base, docs)
    before = _sources(coll)
    shutil.rmtree(docs)

    result = coll.resync()

    assert [r["root"] for r in result["unavailable_roots"]] == [str(docs.resolve())]
    assert result["missing"] == [] and result["pruned"] == []
    reloaded = Collection("kb", base=base)
    assert _sources(reloaded) == before
    assert reloaded.stats()["n_missing"] == 0
    assert reloaded.query("turbines", k=3)


def test_prune_missing_cannot_delete_under_an_unavailable_root(base, docs):
    """The guard holds even when the caller explicitly asked to prune: an
    unreachable folder yields no verdict at all, so there is nothing to prune."""
    import shutil
    coll = _indexed(base, docs)
    before = _sources(coll)
    shutil.rmtree(docs)

    result = coll.resync(prune_missing=True)

    assert result["pruned"] == []
    assert _sources(Collection("kb", base=base)) == before


def test_a_root_replaced_by_a_file_is_reported_as_such(base, docs):
    import shutil
    coll = _indexed(base, docs)
    shutil.rmtree(docs)
    docs.write_text("not a folder any more", encoding="utf-8")

    result = coll.resync()

    assert len(result["unavailable_roots"]) == 1
    assert "not a directory" in result["unavailable_roots"][0]["reason"]
    assert result["missing"] == []


# --------------------------------------------------------------------------- #
#  Confinement: a scheduled re-sync indexes nothing an add would refuse        #
# --------------------------------------------------------------------------- #

def test_a_root_outside_the_policy_is_skipped_and_reported(base, docs):
    """The owner put the folder on their deny list after the collection was
    built. The scheduled re-sync must refuse the root, say so, and leave its
    documents alone - not index it because it was legal once.

    An explicit deny (rather than a whitelist miss) so the test is deterministic
    wherever tmp_path lives: whitelist mode always allows the home folder and the
    working directory, and a pytest tmp dir can be under either."""
    coll = _indexed(base, docs)
    policy = {"mode": "blacklist", "allowed": [], "denied": [docs.resolve()]}
    # Prove the same policy really does refuse this path through the shared gate.
    from localm.rag.store import ConfinementError, confine_index_path
    with pytest.raises(ConfinementError):
        confine_index_path(docs, policy)

    (docs / "gamma.txt").write_text("gamma content", encoding="utf-8")
    result = coll.resync(policy=policy)

    assert [r["root"] for r in result["blocked_roots"]] == [str(docs.resolve())]
    assert result["added"] == 0
    assert result["missing"] == []          # blocked is not a deletion verdict
    assert str((docs / "gamma.txt").resolve()) not in _sources(
        Collection("kb", base=base))


def test_a_permissive_policy_still_resyncs(base, docs):
    """Negative control for the test above: with the folder allowed, the same
    call indexes it - so the skip really was the policy, not a broken path."""
    coll = _indexed(base, docs)
    policy = {"mode": "blacklist", "allowed": [], "denied": []}
    (docs / "gamma.txt").write_text("gamma content", encoding="utf-8")

    result = coll.resync(policy=policy)

    assert result["blocked_roots"] == []
    assert result["added"] == 1


def test_a_secret_file_dropped_into_an_indexed_folder_is_not_swept_in(base, docs):
    """The folder-walk filters that protect an interactive add protect the
    unattended re-sync too."""
    coll = _indexed(base, docs)
    # The filters key on the NAME and suffix (is_secret_index_name /
    # SECRET_SUFFIXES), never on content, so placeholder bodies are enough.
    (docs / "id_rsa").write_text("placeholder key material", encoding="utf-8")
    (docs / "deploy.pem").write_text("placeholder key material", encoding="utf-8")

    result = coll.resync(policy={"mode": "blacklist", "allowed": [], "denied": []})

    assert result["added"] == 0
    sources = _sources(Collection("kb", base=base))
    assert not any(s.endswith("id_rsa") or s.endswith("deploy.pem")
                   for s in sources)


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #

def test_resync_reports_roots_and_progress_lines(base, docs):
    coll = _indexed(base, docs)
    (docs / "beta.txt").unlink()
    lines: list = []
    result = coll.resync(on_progress=lines.append)
    assert result["roots"] == [str(docs.resolve())]
    assert any(line.startswith("missing:") for line in lines)


def test_resync_on_a_collection_with_no_roots_is_a_clean_no_op(base):
    coll = Collection("bare", base=base)
    coll.create()
    result = coll.resync()
    assert result["roots"] == []
    assert (result["added"], result["updated"], result["skipped"]) == (0, 0, 0)
    assert result["missing"] == [] and result["chunks"] == 0
