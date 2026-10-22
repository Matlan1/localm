# SPDX-License-Identifier: AGPL-3.0-or-later
"""A full-repo pull holds a cross-process lock on its destination folder from
before the claim check until it has registered (or failed): a second pull into
the same folder is refused up front and downloads nothing, a repull afterwards
works, the lock file lives beside the models dir rather than in the folder, and
a holder that dies releases it. The holder is a real second interpreter; the
hub is replaced by the fakes of ``test_pull_revision``."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod
from tests.test_pull_revision import COMMIT, hub  # noqa: F401  (pytest fixture)

REPO_ROOT = Path(__file__).resolve().parent.parent

_HOLDER = """
import pathlib, sys
import localm
from localm import model_manager as mm
from localm.model_manager import pull
mm.MODELS_DIR = pathlib.Path(sys.argv[1])
with pull._snapshot_dest_lock(sys.argv[2]):
    print("HELD", localm.__file__, flush=True)
    sys.stdin.read()
"""


class _Holder:
    """A second interpreter holding the lock on one models folder."""

    def __init__(self, models_dir: Path, name: str, home: Path):
        self.lock_path = models_dir.parent / "locks" / ("snapshot-" + name.lower() + ".lock")
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), LOCALM_HOME=str(home))
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(models_dir), name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True)
        line: list[str] = []
        reader = threading.Thread(target=lambda: line.append(self.proc.stdout.readline()))
        reader.start()
        reader.join(60)
        if not line or not line[0].startswith("HELD"):
            self.kill()
            raise AssertionError(f"holder did not take the lock: {line!r} "
                                 f"{self.proc.stderr.read()!r}")
        self.imported_from = Path(line[0].split(" ", 1)[1].strip())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(60)
        self._close()
        self._until_free()

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait(60)
        self._close()
        self._until_free()

    def _until_free(self) -> None:
        """Wait for the OS to drop the dead holder's lock: Windows does so a
        moment after the process is reaped."""
        from filelock import FileLock
        probe = FileLock(str(self.lock_path), timeout=10)
        probe.acquire()
        probe.release()

    def _close(self) -> None:
        for f in (self.proc.stdout, self.proc.stderr):
            if f:
                f.close()


@pytest.fixture()
def holder(hub, tmp_path):  # noqa: F811
    held: list[_Holder] = []

    def start(name: str) -> _Holder:
        h = _Holder(hub.models_dir, name, tmp_path / "holder-home")
        assert REPO_ROOT in h.imported_from.parents
        held.append(h)
        return h

    yield start
    for h in held:
        if h.proc.poll() is None:
            h.kill()


def _said(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


class TestAPullIntoAHeldFolderIsRefused:
    def test_a_second_revision_into_the_folder_downloads_nothing(self, hub, holder, capsys):  # noqa: F811
        holder("q")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        assert "already running" in _said(capsys)
        assert hub.snapshots == []
        assert not (hub.models_dir / "q").exists()
        assert "q" not in mm.load_registry()

    def test_the_same_revision_is_refused_too(self, hub, holder, capsys):  # noqa: F811
        holder("q")
        assert mm.pull_model("o/r@2.0bpw", name="q", model_type="llm") is False
        assert "already running" in _said(capsys)
        assert hub.snapshots == []

    def test_the_default_folder_name_is_held_by_its_own_name(self, hub, holder, capsys):  # noqa: F811
        holder("r-4.0bpw")
        assert mm.pull_model("o/r@4.0bpw", model_type="llm") is False
        assert "already running" in _said(capsys)
        assert hub.snapshots == []

    def test_a_folder_spelled_in_another_case_is_held_too(self, hub, holder, capsys):  # noqa: F811
        holder("Q")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        assert "already running" in _said(capsys)
        assert hub.snapshots == []

    def test_a_pull_into_another_folder_is_not_held_up(self, hub, holder):  # noqa: F811
        holder("q")
        assert mm.pull_model("o/r@4.0bpw", name="other", model_type="llm") is True
        assert hub.snapshots == [("o/r", COMMIT)]

    def test_the_hub_is_not_asked_anything_while_the_folder_is_held(self, hub, holder):  # noqa: F811
        holder("q")
        mm.pull_model("o/r@4.0bpw", name="q", model_type="llm")
        assert [c for c in hub.info_calls if c[2]] == []


class TestTheHolderLeaving:
    def test_a_released_folder_can_be_pulled(self, hub, holder):  # noqa: F811
        h = holder("q")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        h.release()
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert hub.snapshots == [("o/r", COMMIT)]

    def test_a_killed_holder_does_not_block_the_folder(self, hub, holder):  # noqa: F811
        h = holder("q")
        h.kill()
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert hub.snapshots == [("o/r", COMMIT)]


class TestTheLockOfAPull:
    def test_a_sequential_repull_works(self, hub):  # noqa: F811
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm", redownload=True) is True
        assert hub.snapshots == [("o/r", COMMIT)]

    def test_a_failed_pull_releases_the_folder(self, hub):  # noqa: F811
        hub.snapshot_error = OSError("disk full")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        hub.snapshot_error = None
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True

    def test_a_failed_download_is_not_reported_as_a_pull_in_flight(self, hub, capsys):  # noqa: F811
        hub.snapshot_error = OSError("disk full")
        mm.pull_model("o/r@4.0bpw", name="q", model_type="llm")
        said = _said(capsys)
        assert "Download failed" in said and "already running" not in said

    def test_an_oserror_outside_the_download_is_not_reported_as_a_pull_in_flight(
            self, hub, monkeypatch):  # noqa: F811
        def broken():
            raise OSError("cannot create the models folder")

        monkeypatch.setattr(mm, "ensure_dirs", broken)
        with pytest.raises(OSError, match="cannot create the models folder"):
            mm.pull_model("o/r@4.0bpw", name="q", model_type="llm")

    def test_the_lock_is_held_while_the_download_runs(self, hub, monkeypatch, capsys):  # noqa: F811
        import huggingface_hub
        inner = huggingface_hub.snapshot_download
        seen: list[bool] = []

        def snapshot_that_probes_the_lock(**kw):
            try:
                with pull_mod._snapshot_dest_lock("q"):
                    seen.append(False)
            except pull_mod.PullInFlight:
                seen.append(True)
            return inner(**kw)

        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            snapshot_that_probes_the_lock)
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert seen == [True]

    def test_the_lock_file_is_beside_the_models_dir_not_in_the_folder(self, hub):  # noqa: F811
        lock = pull_mod._snapshot_lock_path("q")
        assert lock.parent == hub.models_dir.parent / "locks"
        assert hub.models_dir not in lock.parents
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        folder = hub.models_dir / "q"
        assert not [p for p in folder.rglob("*") if p.suffix == ".lock"]
        assert sorted(p.name for p in hub.models_dir.iterdir()) == ["q"]

    def test_deleting_the_folder_does_not_touch_the_lock(self, hub):  # noqa: F811
        import shutil
        with pull_mod._snapshot_dest_lock("q"):
            (hub.models_dir / "q").mkdir()
            shutil.rmtree(hub.models_dir / "q")
            assert pull_mod._snapshot_lock_path("q").exists()


class TestDiscardingPartialDataOfAHeldFolder:
    def test_discard_refuses_while_a_pull_holds_the_folder(self, hub, holder):  # noqa: F811
        holder("q")
        with pytest.raises(pull_mod.PullInFlight) as e:
            pull_mod.discard_partial_download("o/r@4.0bpw", name="q")
        assert "already running" in str(e.value)

    def test_discard_works_once_the_holder_is_gone(self, hub, holder):  # noqa: F811
        holder("q").kill()
        assert pull_mod.discard_partial_download("o/r@4.0bpw", name="q") == 0
