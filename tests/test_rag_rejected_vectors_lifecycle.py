# SPDX-License-Identifier: AGPL-3.0-or-later
"""The full life of a REJECTED vectors.json: preserved, reported, cleared.

#795 (62013ffa) promised that a vector index localm refuses to use is never
deleted to make the problem go away, and that the degrade keeps being reported
until the index is rebuilt. An adversarial review of the drain merges found the
promise fails on the commonest path, plus two ways the preserved copy itself can
be lost or become a permanent nuisance. These are those three, and each one is
paired with the fires-control that proves the test can fail: the assertions here
all go RED against the pre-fix ``_save`` (verified by running this file's
scenarios against ``git show origin/master:localm/rag/store.py``).

  X5  - any successful embed took the WRITE branch, which overwrote the rejected
        file with a new vectors.json AND cleared vector_degrade_reason. The
        quarantine lived inside the other branch, so on the commonest live path
        (an embedder is available and one document changed) the evidence was
        destroyed and an unattended re-sync reported a clean run over a
        knowingly-degraded collection. Note the obvious `all()` fix would be
        WRONG: partial vector coverage is a first-class supported state.
  X12 - emptying a collection left the sidecar behind, pinning a degrade whose
        prescribed remedy ('rag repair') returns early on a collection with no
        documents, so it could never be cleared.
  X13 - a second incident's os.replace overwrote the first preserved copy while
        the log said "Nothing was deleted".
"""

from __future__ import annotations

import json

import pytest

from localm.rag import Collection

DIM = 4


def _embed(texts):
    """EmbedFn: list[str] -> list[list[float]]. Deterministic, no model."""
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
def base(tmp_path):
    d = tmp_path / "collections"
    d.mkdir()
    return d


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("alpha turbines maintenance", encoding="utf-8")
    (d / "b.txt").write_text("beta gearbox inspection", encoding="utf-8")
    return d


def _sidecars(base, name="kb"):
    return sorted(p.name for p in (base / name).glob("vectors.json.rejected*"))


def _degrade_it(base, docs, name="kb"):
    """An indexed collection whose vectors.json _load() REFUSES (wrong count)."""
    coll = Collection(name, base=base)
    coll.create()
    coll.add_paths([docs], embed_fn=_embed)
    (base / name / "vectors.json").write_text(
        json.dumps({"dim": DIM, "vectors": [[0.1] * DIM]}), encoding="utf-8")
    coll = Collection(name, base=base)
    assert coll.vector_degrade_reason is not None, "setup: it should be rejected"
    assert coll._vectors_file_rejected
    return coll


def test_a_successful_embed_does_not_destroy_the_rejected_index(base, docs):
    """X5, the headline. An embedder is available and ONE document changed - the
    normal state of a live scheduled re-sync. The rejected bytes must survive and
    the degrade must still be reported."""
    coll = _degrade_it(base, docs)
    (docs / "a.txt").write_text("alpha turbines maintenance, revised",
                                encoding="utf-8")

    result = coll.resync(embed_fn=_embed, policy=None)

    assert _sidecars(base), (
        "the rejected vectors.json was overwritten by the new one: the only copy "
        "of that data AND the only evidence of the fault are gone")
    assert not (base / "kb" / "vectors.json.rejected").is_dir()
    assert result.get("vector_degrade_reason"), (
        "the re-sync reported a clean run over a collection it knows is degraded")


def test_the_degrade_survives_a_cold_reload_after_a_partial_re_embed(base, docs):
    """The same case seen by the NEXT process. The mixed file (real vectors for
    the changed doc, nulls for the rest) is structurally valid and loads clean,
    so without an independent check the fault becomes invisible for good."""
    coll = _degrade_it(base, docs)
    (docs / "a.txt").write_text("alpha turbines maintenance, revised",
                                encoding="utf-8")
    coll.resync(embed_fn=_embed, policy=None)

    reloaded = Collection("kb", base=base)
    assert reloaded.vector_degrade_reason is not None, (
        "a cold reload reported a healthy collection; the mixed index is NOT "
        "self-correcting, so this is the fault becoming folklore")
    assert "set aside" in reloaded.vector_degrade_reason


def test_partial_coverage_alone_is_not_treated_as_a_fault(base, docs):
    """The guard the routing note asked for: partial vector coverage is a
    legitimate state (a collection mid-embed), NOT a degrade. Only a set-aside
    sidecar makes it one. Without this, tightening the write branch to `all()`
    would refuse to persist vectors for every partially embedded collection."""
    coll = Collection("kb", base=base)
    coll.create()
    coll.add_paths([docs / "a.txt"], embed_fn=_embed)      # embedded
    coll.add_paths([docs / "b.txt"], embed_fn=None)        # not embedded

    assert not _sidecars(base), "nothing was ever rejected here"
    assert coll.vector_degrade_reason is None, (
        f"partial coverage was reported as a fault: {coll.vector_degrade_reason}")
    assert (base / "kb" / "vectors.json").is_file(), (
        "the partially covered index must still be persisted")
    assert Collection("kb", base=base).vector_degrade_reason is None


def test_a_full_rebuild_clears_the_degrade_but_keeps_the_evidence(base, docs):
    """The prescribed remedy has to actually work, or the message that tells the
    user to run it is a dead end (that is X12's real complaint). A COMPLETE index
    is the all-clear; the preserved bytes stay on disk regardless."""
    coll = _degrade_it(base, docs)
    # A re-sync that indexes NOTHING does not rewrite anything, so it never
    # reaches _save and never quarantines. Change a file, so this run writes.
    (docs / "a.txt").write_text("alpha turbines maintenance, revised",
                                encoding="utf-8")
    coll.resync(embed_fn=None, policy=None)               # quarantines
    assert _sidecars(base)

    repaired = Collection("kb", base=base)
    repaired.add_paths(repaired.documents(), force=True, embed_fn=_embed)

    assert repaired.vector_degrade_reason is None, (
        f"'rag repair --embed' did not clear it: {repaired.vector_degrade_reason}")
    assert Collection("kb", base=base).vector_degrade_reason is None
    assert _sidecars(base), (
        "clearing the degrade must not delete the preserved copy - the fault is "
        "fixed, but the record of it costs nothing to keep")


def test_emptying_a_collection_does_not_pin_an_unclearable_degrade(base, docs):
    """X12. Two documents, so the collection goes degraded -> quarantined ->
    empty (the PR's own test used ONE document, so it never quarantined and its
    assertion passed vacuously). With no chunks there is nothing to realign the
    vectors to, and 'rag repair' returns early on a collection with no documents,
    so a sidecar kept here could never be cleared by any means the product
    offers."""
    coll = _degrade_it(base, docs)
    coll.remove_doc(coll.documents()[0])                   # chunks remain
    assert _sidecars(base), "setup: the first removal should have quarantined"

    for d in list(coll.documents()):
        coll.remove_doc(d)                                 # now empty

    assert not _sidecars(base), (
        f"an emptied collection kept {_sidecars(base)}, which pins a degrade "
        f"that no rebuild can clear")
    assert Collection("kb", base=base).vector_degrade_reason is None


def test_a_second_quarantine_does_not_destroy_the_first_copy(base, docs):
    """X13. os.replace overwrites its destination silently, so a fixed sidecar
    name meant incident two destroyed incident one's preserved bytes while the
    log asserted "Nothing was deleted" - data loss plus a false safety claim."""
    coll = _degrade_it(base, docs)
    (docs / "a.txt").write_text("alpha turbines maintenance, revised",
                                encoding="utf-8")          # so the run really writes
    coll.resync(embed_fn=None, policy=None)
    first = sorted((base / "kb").glob("vectors.json.rejected*"))[0]
    first.write_text("FIRST-COPY", encoding="utf-8")

    # A second, independent incident on the same collection.
    (base / "kb" / "vectors.json").write_text(
        json.dumps({"dim": DIM, "vectors": [[0.1] * DIM]}), encoding="utf-8")
    again = Collection("kb", base=base)
    assert again._vectors_file_rejected, "setup: the new file should be rejected"
    again.add_paths([docs / "b.txt"], force=True, embed_fn=None)

    assert first.read_text(encoding="utf-8") == "FIRST-COPY", (
        "the second quarantine overwrote the first preserved copy")
    assert len(_sidecars(base)) >= 2, (
        f"the second copy needs a name of its own: {_sidecars(base)}")


def test_the_scheduled_job_reports_the_degrade_it_would_have_hidden(base, docs,
                                                                    tmp_path,
                                                                    monkeypatch):
    """End to end on the surface that matters: an unattended re-sync with an
    embedder must SAY the collection is degraded. A job history showing a clean
    run here is the "reported success after failing" AGENTS.md rule 5 forbids,
    and it is the only place this run is ever seen."""
    import types

    from localm.plugins.builtin.jobs import runner
    rag = tmp_path / "rag"
    monkeypatch.setattr("localm.rag.store.rag_dir", lambda: rag)
    monkeypatch.setattr(runner, "_rag_embed_fn", lambda: _embed)
    # The scheduled path confines indexing to the owner's allowed folders and a
    # pytest tmp dir is outside them, so the policy is disabled here and the run
    # really re-indexes. This test is about what the run REPORTS.
    monkeypatch.setattr("localm.rag.store.indexing_policy", lambda: None)
    _degrade_it(rag, docs)
    (docs / "a.txt").write_text("alpha turbines maintenance, revised",
                                encoding="utf-8")

    job = types.SimpleNamespace(id="j1", task_kind="rag", collection="kb",
                                model=None, prompt="")
    result = runner.run_job(job)

    assert result["status"] == "ok", result
    assert "degraded" in result["output"], (
        f"the unattended run hid the degrade:\n{result['output']}")
    assert _sidecars(rag), "and it destroyed the evidence while doing so"
