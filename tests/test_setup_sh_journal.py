# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh keeps a journal of what it has done, so an interrupted run can be picked up.

The real script runs in a bare folder with ``LOCALM_SETUP_ABORT_AFTER`` set, which
stops it right after the named step (exit 99) the way an interruption would, without
installing anything. The journal it leaves is the same file ``localm.install_manifest``
reads.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from localm import install_manifest as im

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "setup.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "clone"
    d.mkdir()
    shutil.copy(SETUP_SH, d / "setup.sh")
    return d


def _setup(folder, *, abort_after="portable-choice", extra_env=None):
    env = dict(os.environ)
    env.pop("LOCALM_SETUP_ABORT_AFTER", None)
    if abort_after:
        env["LOCALM_SETUP_ABORT_AFTER"] = abort_after
    env.update(extra_env or {})
    r = subprocess.run([shutil.which("bash"), "setup.sh", "--yes"], cwd=str(folder),
                       env=env, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def _journal(folder) -> list:
    return (folder / ".localm-setup-journal").read_text(encoding="utf-8").splitlines()


class TestAnInterruptedRun:
    def test_it_stops_after_the_named_step_and_leaves_a_journal_without_complete(self, folder):
        rc, out = _setup(folder)
        assert rc == 99, out
        assert _journal(folder) == ["begin\tportable-choice", "done\tportable-choice"]
        assert not im.journal_state(folder)["complete"]

    def test_the_python_reader_understands_what_the_shell_wrote(self, folder):
        _setup(folder)
        st = im.journal_state(folder)
        assert st["done"] == ["portable-choice"] and st["started"] == []
        assert "stopped after 'portable-choice'" in im.describe_journal(st)


class TestPickingItUp:
    def test_the_next_run_says_where_the_last_one_stopped(self, folder):
        _setup(folder)
        rc, out = _setup(folder)
        assert rc == 99
        assert "A previous setup in this folder stopped after 'portable-choice'" in out
        assert "Picking it up" in out
        assert "resume" in _journal(folder)

    def test_a_step_left_open_is_named(self, folder):
        (folder / ".localm-setup-journal").write_text(
            "begin\tportable-choice\ndone\tportable-choice\nbegin\tvenv\n", encoding="utf-8")
        rc, out = _setup(folder)
        assert "stopped after 'portable-choice', while running 'venv'" in out

    def test_a_run_that_never_finished_a_step_says_so(self, folder):
        (folder / ".localm-setup-journal").write_text("begin\tuv-portable\n", encoding="utf-8")
        rc, out = _setup(folder)
        assert "before finishing its first step ('uv-portable')" in out

    def test_a_final_line_cut_off_mid_write_is_ignored(self, folder):
        (folder / ".localm-setup-journal").write_bytes(
            b"begin\tvenv\ndone\tvenv\nbegin\tnative-runt")
        rc, out = _setup(folder)
        assert "stopped after 'venv'" in out
        assert "native-runt" not in out

    def test_a_finished_journal_is_a_fresh_start(self, folder):
        (folder / ".localm-setup-journal").write_text(
            "begin\told-step\ndone\told-step\ncomplete\n", encoding="utf-8")
        rc, out = _setup(folder)
        assert "previous setup" not in out
        assert _journal(folder) == ["begin\tportable-choice", "done\tportable-choice"]


class TestAJournalThatCannotBeWritten:
    def test_setup_warns_once_and_carries_on(self, folder):
        (folder / ".localm-setup-journal").mkdir()      # a folder where the file belongs
        rc, out = _setup(folder)
        assert rc == 99, out                              # it still reached the abort hook
        assert out.count("Could not write .localm-setup-journal") == 1


class TestTheScriptItself:
    """Pins the shape of the instrumentation, so a step cannot be added without its
    journal lines."""

    text = SETUP_SH.read_text(encoding="utf-8")

    def test_every_step_that_begins_also_finishes(self):
        begun = re.findall(r'step_begin\s+"?([\w$-]+)"?', self.text)
        done = re.findall(r'step_done\s+"?([\w$-]+)"?', self.text)
        assert sorted(b for b in begun if not b.startswith("$")) == \
            sorted(d for d in done if not d.startswith("$"))
        assert begun, "no steps are journaled"

    def test_the_script_ends_by_marking_the_install_complete(self):
        final = re.search(r"^jr complete$", self.text, re.M)
        assert final, "setup never marks the install complete"
        assert final.start() > self.text.index("step_done record")

    def test_an_artifact_outside_the_folder_is_announced_before_it_is_created(self):
        menu = self.text.index('jr intend shortcut "$apps/localm.desktop"')
        assert menu < self.text.index('cat > "$apps/localm.desktop"')
        cmd = self.text.index('jr intend command "$HOME/.local/bin/localm"')
        assert cmd < self.text.index("-m localm.globalcmd install")

    def test_a_cut_short_runtime_download_is_forced(self):
        assert 'SL_FORCE="--force"' in self.text
        assert 'setup-llama --backend "$BACKEND" $SL_FORCE' in self.text


_UV_STUB = """#!/bin/sh
echo "uv $*" >> "$STUB_LOG"
case "$1" in
  venv) for last; do :; done; mkdir -p "$last/bin" ;;
esac
exit 0
"""


class TestACutShortVenv:
    """The venv step is the one an interruption can leave looking fine and be broken:
    ``uv venv`` writes the marker only when it succeeds, so a half-made ``.venv`` has
    none, and setup would treat it as somebody else's and keep it. With a stub uv,
    setup runs up to the venv step and shows whether it rebuilds."""

    def _run(self, folder, tmp_path, journal):
        uv_dir = folder / ".uv"
        uv_dir.mkdir()
        stub = uv_dir / "uv"
        stub.write_bytes(_UV_STUB.replace("\r\n", "\n").encode("utf-8"))
        stub.chmod(0o755)
        (folder / ".venv").mkdir()                       # half made: no marker file
        (folder / ".venv" / "half-written").write_text("x", encoding="utf-8")
        if journal is not None:
            (folder / ".localm-setup-journal").write_text(journal, encoding="utf-8")
        log = tmp_path / "uv.log"
        log.write_text("", encoding="utf-8")
        rc, out = _setup(folder, abort_after="venv", extra_env={"STUB_LOG": str(log)})
        return rc, out, log.read_text(encoding="utf-8")

    def test_a_venv_the_journal_says_was_cut_short_is_rebuilt(self, folder, tmp_path):
        rc, out, uv_log = self._run(
            folder, tmp_path,
            "begin\tportable-choice\ndone\tportable-choice\nbegin\tvenv\n")
        assert rc == 99, out
        assert "stopped while creating .venv; recreating it" in out
        assert "uv venv" in uv_log and "--clear" in uv_log, uv_log
        assert (folder / ".venv" / ".localm-venv").is_file()

    def test_without_that_journal_a_marker_less_venv_is_left_alone(self, folder, tmp_path):
        rc, out, uv_log = self._run(folder, tmp_path, None)
        assert rc == 99, out
        assert "uv venv" not in uv_log, uv_log
        assert (folder / ".venv" / "half-written").is_file()
