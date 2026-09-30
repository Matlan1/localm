# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder project registry, and above all what it must NOT record: a
privacy-mode project is never written here.
"""

import json

import pytest

from localm.plugins.coder import projects


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point home_dir at a throwaway directory.

    Patched where `projects` RESOLVES it (localm.config.home_dir), not on the
    projects module: the functions import it inside the call, so patching a name
    on `projects` would leave the real home untouched.
    """
    import localm.config as cfg
    monkeypatch.setattr(cfg, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(cfg, "load_config", lambda: {})
    return tmp_path


class TestPrivacyModeIsNeverRecorded:
    """Privacy mode is never recorded, and that is not configurable."""

    def test_a_privacy_session_writes_no_entry(self, tmp_path):
        assert projects.record_project(tmp_path, "privacy") is False
        assert projects.list_projects() == []
        # And nothing at all on disk: not even an empty list file.
        assert not (tmp_path / "coder-projects.json").exists()

    @pytest.mark.parametrize("spelling", ["privacy", "PRIVACY", "Privacy", " privacy "])
    def test_the_refusal_does_not_depend_on_spelling(self, spelling, tmp_path):
        # The CLI accepts the mode case-insensitively, so a differently-cased
        # value reaching here is still not recordable.
        assert projects.record_project(tmp_path, spelling) is False
        assert projects.list_projects() == []

    def test_a_privacy_session_does_not_disturb_an_existing_list(self, tmp_path):
        other = tmp_path / "kept"
        other.mkdir()
        assert projects.record_project(other, "log") is True
        before = json.loads((tmp_path / "coder-projects.json").read_text("utf-8"))

        secret = tmp_path / "secret"
        secret.mkdir()
        assert projects.record_project(secret, "privacy") is False

        after = json.loads((tmp_path / "coder-projects.json").read_text("utf-8"))
        assert after == before, "a privacy session must not even rewrite the file"
        assert all("secret" not in e["path"] for e in after)


class TestRecording:
    def test_log_and_full_are_both_recorded(self, tmp_path):
        # Both persist SOMEWHERE, so both are part of "what have I worked on";
        # only where the transcript lands differs.
        for mode in ("log", "full"):
            d = tmp_path / mode
            d.mkdir()
            assert projects.record_project(d, mode) is True
        assert {e["name"] for e in projects.list_projects()} == {"log", "full"}

    def test_the_same_project_moves_to_the_front_and_counts_up(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        projects.record_project(a, "log")
        projects.record_project(b, "log")
        projects.record_project(a, "log")
        got = projects.list_projects()
        assert [e["name"] for e in got] == ["a", "b"], "most recent first"
        assert got[0]["sessions"] == 2, "a second visit counts, it does not duplicate"
        assert len(got) == 2, "and it must not append a second row for the same path"

    def test_the_limit_is_honoured(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {"coder_projects_remembered": 2})
        for n in "abc":
            d = tmp_path / n
            d.mkdir()
            projects.record_project(d, "log")
        assert [e["name"] for e in projects.list_projects()] == ["c", "b"]

    def test_remembering_can_be_turned_off(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {"coder_remember_projects": False})
        assert projects.record_project(tmp_path, "log") is False
        assert projects.list_projects() == []


class TestListing:
    def test_a_deleted_project_is_reported_unavailable_not_dropped(self, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        projects.record_project(gone, "log")
        gone.rmdir()

        got = projects.list_projects()
        # RETURNED, not filtered: the caller can say "moved or deleted".
        assert len(got) == 1
        assert got[0]["available"] is False

    def test_forget_all_removes_the_file(self, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        projects.record_project(d, "log")
        assert (tmp_path / "coder-projects.json").exists()
        projects.forget_all()
        assert not (tmp_path / "coder-projects.json").exists()
        assert projects.list_projects() == []

    def test_a_corrupt_store_does_not_break_a_session(self, tmp_path):
        (tmp_path / "coder-projects.json").write_text("{not json", encoding="utf-8")
        # Reported (a warning is logged) but never raised.
        assert projects.list_projects() == []
        d = tmp_path / "p"
        d.mkdir()
        assert projects.record_project(d, "log") is True


class TestConcurrentWriters:
    """record_project and forget_project run on different executor threads;
    each read-modify-write of the store must see the other's result."""

    def test_a_removal_racing_a_new_session_keeps_both_results(
            self, tmp_path, monkeypatch):
        import threading
        import time

        kept = [tmp_path / f"kept{i}" for i in range(3)]
        gone = tmp_path / "gone"
        newcomer = tmp_path / "newcomer"
        for d in kept + [gone, newcomer]:
            d.mkdir()
        for d in kept + [gone]:
            assert projects.record_project(d, "log") is True

        real_load = projects._load

        def slow_load():
            entries = real_load()
            time.sleep(0.05)
            return entries

        monkeypatch.setattr(projects, "_load", slow_load)
        start = threading.Barrier(2)
        errors = []

        def run(fn):
            try:
                start.wait(5)
                fn()
            except Exception as e:                             # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=run, args=(lambda: projects.forget_project(gone),)),
            threading.Thread(target=run,
                             args=(lambda: projects.record_project(newcomer, "log"),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert errors == []

        listed = {e["path"] for e in json.loads(
            (tmp_path / "coder-projects.json").read_text("utf-8"))}
        assert str(gone.resolve()) not in listed, "the removed project came back"
        missing = {str(d.resolve()) for d in kept + [newcomer]} - listed
        assert missing == set(), f"a concurrent write dropped {missing}"

    def test_a_failed_rewrite_leaves_the_old_list_whole(self, tmp_path, monkeypatch):
        import localm.config as cfg

        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            assert projects.record_project(d, "log") is True
        store = tmp_path / "coder-projects.json"
        before = store.read_bytes()

        def refuse(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(cfg, "_replace_atomic", refuse)
        with pytest.raises(OSError, match="disk full"):
            projects.forget_project(a)

        assert store.read_bytes() == before, "a failed rewrite changed the list"
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
        assert projects.record_project(b, "log") is False
        assert store.read_bytes() == before
