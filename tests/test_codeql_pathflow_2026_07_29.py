# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three check-then-use gaps on the path-injection surface, each closed by
making the sink consume the value the guard returned instead of the caller's
original.

All three share one shape: a guard is called for its exception and its RETURN
VALUE thrown away, so the code decides on one value and acts on another.
``registry.py`` documents that shape as an escape at its own remove_model gate.

1. ``Collection._add_paths_locked`` must walk the path
   ``confine_index_path(p, policy)`` returned, not the caller's unresolved one.
2. ``model_manager/pull.py`` must read a registry path through the
   ``_entry_path`` choke point, never ``load_registry()[name]["path"]`` raw,
   because that value decides an ``unlink()``.
3. ``registry.py`` must read ``entry["mmproj"]`` through the same choke point,
   so the recorded projector path gets the type check and the ``..`` rejection
   every other stored path gets.

Where a test asserts an ABSENCE (no crash, no traversal), it is paired with a
control that proves the same test can still observe the positive case.
"""

from pathlib import Path

import pytest

from localm.model_manager.registry import _entry_path
from localm.rag.store import Collection


def _wl(*allowed):
    """A whitelist policy allowing (home + cwd, always) plus *allowed*."""
    return {"mode": "whitelist", "allowed": [Path(a) for a in allowed], "denied": []}


# --------------------------------------------------------------------------- #
#  1. RAG indexing walks the CONFINED path, not the caller's original          #
# --------------------------------------------------------------------------- #

class TestIndexActsOnTheConfinedPath:
    """``confine_index_path`` returns the RESOLVED path it validated, and the
    walk has to use THAT value.

    What is asserted is the WIRING - that the top-level walk root is the value
    confinement returned - not the repair of a live escape: ``_expand``
    re-confines every file it emits on that file's own resolved path.

    Asserting it needs a collaborator whose return differs observably from its
    input, because every real difference (resolve(), symlink following) is
    either normalised away downstream or needs a symlink privilege the test may
    not hold on Windows. So ``confine_index_path`` is substituted with one that
    returns a DIFFERENT directory. The unit under test is
    ``_add_paths_locked``'s wiring; the substituted function is its
    collaborator.
    """

    def test_the_walk_uses_the_path_confinement_returned(self, tmp_path, monkeypatch):
        asked = tmp_path / "asked"
        asked.mkdir()
        (asked / "asked.txt").write_text("the caller named this one", encoding="utf-8")
        returned = tmp_path / "returned"
        returned.mkdir()
        (returned / "returned.txt").write_text(
            "confinement returned this one", encoding="utf-8")

        import localm.rag.store as store
        monkeypatch.setattr(store, "confine_index_path",
                            lambda p, policy=None: returned)

        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([asked], policy=_wl())

        names = sorted(Path(d["path"]).name for d in c.docs())
        assert res["added"] == 1
        assert names == ["returned.txt"], (
            f"indexed {names}; the walk used the caller's original path instead "
            "of the one confine_index_path returned")

    def test_control_identity_confinement_still_indexes(self, tmp_path, monkeypatch):
        # With a pass-through collaborator the same wiring still indexes normally.
        import localm.rag.store as store
        real = store.confine_index_path
        monkeypatch.setattr(store, "confine_index_path",
                            lambda p, policy=None: real(p, policy))

        work = tmp_path / "work"
        (work / "docs").mkdir(parents=True)
        (work / "docs" / "a.txt").write_text("gfx1030 runtime notes", encoding="utf-8")
        monkeypatch.chdir(work)

        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([work / "docs"], policy=_wl())

        assert res["added"] == 1
        assert Path(c.docs()[0]["path"]).name == "a.txt"

    def test_cli_path_is_unchanged_by_the_fix(self, tmp_path, monkeypatch):
        # policy=None is the CLI contract: unconfined and not rewritten, so a
        # relative CLI pick keeps indexing as before.
        work = tmp_path / "work"
        (work / "docs").mkdir(parents=True)
        (work / "docs" / "a.txt").write_text("gfx1030 runtime notes", encoding="utf-8")
        monkeypatch.chdir(work)

        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([Path("docs")], policy=None)

        assert res["added"] == 1


# --------------------------------------------------------------------------- #
#  2 + 3. Every stored registry path goes through the _entry_path choke point  #
# --------------------------------------------------------------------------- #

# The same matrix, applied to the projector field rather than to ``path``.
BAD_MMPROJ = {
    "null_mmproj": {"path": "Z:/m.gguf", "mmproj": None},
    "int_mmproj": {"path": "Z:/m.gguf", "mmproj": 123},
    "empty_mmproj": {"path": "Z:/m.gguf", "mmproj": ""},
    "traversal_mmproj": {"path": "Z:/m.gguf", "mmproj": "../../etc/shadow"},
    "win_traversal_mmproj": {"path": "Z:/m.gguf", "mmproj": r"..\..\Windows\x"},
    "not_a_dict": "oops",
}


class TestEntryPathCoversTheProjectorField:
    @pytest.mark.parametrize("label,entry", list(BAD_MMPROJ.items()))
    def test_malformed_projector_is_refused(self, label, entry):
        assert _entry_path(entry, "mmproj") is None

    def test_control_good_projector_is_returned(self):
        entry = {"path": "Z:/m.gguf", "mmproj": "Z:/models/proj.gguf"}
        assert _entry_path(entry, "mmproj") == "Z:/models/proj.gguf"

    def test_default_field_is_still_path(self):
        # The added parameter must not change any existing call site.
        assert _entry_path({"path": "Z:/m.gguf"}) == "Z:/m.gguf"
        assert _entry_path({"path": "../../etc/shadow"}) is None

    @pytest.mark.parametrize("label,entry", list(BAD_MMPROJ.items()))
    def test_resolve_mmproj_survives_a_malformed_projector(
            self, label, entry, tmp_path, monkeypatch):
        # The real consumer, which hands the result to the native mtmd loader.
        import localm.model_manager as _mm
        import localm.model_manager.registry as reg

        monkeypatch.setattr(_mm, "load_registry", lambda: {"m": entry})
        monkeypatch.setattr(reg, "get_model_info", lambda *a, **k: None)

        got = reg.get_model_mmproj("m")

        assert got is None or ".." not in Path(got).parts


class TestPullDedupRoutesThroughTheChokePoint:
    """The post-download dedup branch offers "alias and delete the duplicate".
    It must read the sibling's path through the choke point: a raw read crashes
    the pull with TypeError on a malformed sibling entry, and the value read
    decides an unlink()."""

    @staticmethod
    def _drive_dedup(monkeypatch, tmp_path, sibling_entry, answer="a"):
        """Run _pull_url to completion against a registry holding one sibling
        with a matching sha256, answering the interactive dedup prompt."""
        import click

        import localm.model_manager as mm

        content = b"url-model-bytes"
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        digest = __import__("hashlib").sha256(content).hexdigest()

        entry = dict(sibling_entry) if isinstance(sibling_entry, dict) else sibling_entry
        if isinstance(entry, dict):
            entry["sha256"] = digest
        store = {"twin": entry}

        monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

        def _update(mutator):
            reg = dict(store)
            mutator(reg)
            store.clear()
            store.update(reg)
            return dict(store)

        monkeypatch.setattr(mm, "update_registry", _update)
        monkeypatch.setattr(mm, "save_registry",
                            lambda reg: (store.clear(), store.update(reg)))

        class FakeResp:
            status_code = 200
            headers = {"content-length": str(len(content))}

            def raise_for_status(self):
                pass

            def iter_content(self, n):
                yield content

        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            lambda method, url, **k: FakeResp())
        monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)
        # Take the interactive branch: that is the one holding the raw read.
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *a, **k: answer)

        ok = mm._pull_url("https://x.test/url-model.gguf", "fresh")
        return ok, store, models_dir / "url-model.gguf"

    @pytest.mark.parametrize("label,entry", [
        ("null_path", {"path": None}),
        ("int_path", {"path": 123}),
        ("no_path", {"source": "local"}),
        ("empty_path", {"path": ""}),
        ("traversal_path", {"path": "../../etc/shadow"}),
        ("not_a_dict", "oops"),
    ])
    def test_malformed_sibling_never_deletes_the_download(
            self, label, entry, tmp_path, monkeypatch):
        # The file just downloaded must survive a null, an int, and a '..' sibling
        # path.
        ok, store, dest = self._drive_dedup(monkeypatch, tmp_path, entry)

        assert dest.is_file(), (
            "the freshly-downloaded file was deleted on the strength of a "
            "malformed sibling registry entry")
        assert ok is True
        assert "fresh" in store, "the download was neither aliased nor registered"

    def test_control_good_sibling_still_aliases_and_deletes(
            self, tmp_path, monkeypatch):
        # With a WELL-FORMED sibling the branch still aliases and drops the
        # duplicate file.
        twin = tmp_path / "twin.gguf"
        twin.write_bytes(b"url-model-bytes")
        ok, store, dest = self._drive_dedup(
            monkeypatch, tmp_path, {"path": str(twin), "source": "local"})

        assert ok is True
        assert not dest.exists(), "the duplicate download should have been removed"
        assert "fresh" in store, "the alias was not created"
