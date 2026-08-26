# SPDX-License-Identifier: AGPL-3.0-or-later
"""`rag repair` must still repair when run non-interactively.

`click.confirm(..., abort=True)` collapses two different answers into one Abort:

    user typed "n"   -> Abort
    EOF / no stdin   -> Abort

So run from cron, CI, or any script with no stdin, repair exits NON-ZERO having
done NOTHING, on exactly the stale or corrupt hybrid indexes it exists to
rebuild.

Aborting loses the repair; proceeding would silently delete semantic search. The
branch taken loses NOTHING - the embeddings the collection already has are
preserved - and it says so, so the repair happens and no capability is lost.

Without `abort=True`, an explicit "n" RETURNS False while EOF still raises Abort,
so the two are separable.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Own copy rather than importing a module-private fixture."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _fake_repair_collection(monkeypatch, ragcli, *, has_vectors, captured):
    """Mirrors the rag lifecycle tests' helper."""
    class _FakeColl:
        def __init__(self, name):
            self.name = name

        @property
        def corrupt(self):
            return False

        def documents(self):
            return ["/x/a.txt"]

        def stats(self):
            return {"has_vectors": has_vectors}

        def add_paths(self, paths, *, force=False, embed_fn=None, on_progress=None,
                     model_name=None):
            captured["embed_fn"] = embed_fn
            captured["ran"] = True
            return {"added": 0, "updated": len(paths), "skipped": 0,
                    "chunks": len(paths), "failed": []}

    import localm.rag as ragpkg
    monkeypatch.setattr(ragcli, "Collection", _FakeColl, raising=False)
    monkeypatch.setattr(ragpkg, "Collection", _FakeColl, raising=False)


@pytest.fixture
def _cli(monkeypatch):
    from click.testing import CliRunner
    from localm.cli import rag as ragcli
    captured: dict = {}
    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn",
                        lambda url: (lambda texts: [[0.1, 0.2]] * len(texts)))
    return CliRunner(), ragcli, captured, monkeypatch


class TestNonInteractiveRepairStillRepairs:
    def test_no_stdin_on_hybrid_collection_repairs_instead_of_aborting(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        # input="" == EOF == what cron / CI / `repair < /dev/null` produces.
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="")
        assert r.exit_code == 0, r.output
        assert captured.get("ran"), f"repair did NOT run: {r.output}"

    def test_no_stdin_preserves_the_embeddings_rather_than_dropping_them(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="")
        assert r.exit_code == 0, r.output
        assert callable(captured["embed_fn"]), (
            "a non-interactive repair must KEEP semantic search, not silently "
            f"drop it: {r.output}")

    def test_the_non_interactive_choice_is_surfaced_not_silent(self, _cli):
        # It may pick a default, but it must not pick one silently.
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="")
        low = r.output.lower()
        assert "embeddings" in low and ("--yes" in low or "not an interactive" in low), \
            r.output

    def test_no_stdin_on_a_bm25_only_collection_still_repairs(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=False,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="")
        assert r.exit_code == 0, r.output
        assert captured.get("ran"), r.output


class TestInteractiveBehaviourIsUnchanged:
    """NEGATIVE CASES: the embedding guard survives - repair must not silently
    strip a hybrid collection's embeddings."""

    def test_explicit_no_still_aborts_without_repairing(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="n\n")
        assert r.exit_code != 0, r.output
        assert not captured.get("ran"), "declining must not run add_paths"

    def test_explicit_yes_still_repairs_and_drops_embeddings(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll"], input="y\n")
        assert r.exit_code == 0, r.output
        assert captured.get("ran"), r.output
        assert captured["embed_fn"] is None, \
            "an explicit yes means the user accepted the drop"

    def test_yes_flag_still_drops_embeddings_without_asking(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll", "--yes"])
        assert r.exit_code == 0, r.output
        assert captured["embed_fn"] is None, r.output

    def test_embed_flag_still_keeps_embeddings_without_asking(self, _cli):
        runner, ragcli, captured, monkeypatch = _cli
        _fake_repair_collection(monkeypatch, ragcli, has_vectors=True,
                                captured=captured)
        r = runner.invoke(ragcli.rag_repair, ["coll", "--embed"])
        assert r.exit_code == 0, r.output
        assert callable(captured["embed_fn"]), r.output
