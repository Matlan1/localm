# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat keeps a journal of what it has done, so an interrupted run can be picked up,
and keeps its scratch files inside the clone.

The real script runs in a bare folder with ``LOCALM_SETUP_ABORT_AFTER`` set, which stops
it right after the named step (exit 99) the way an interruption would, without
installing anything. The journal it leaves is the file ``localm.install_manifest``
reads. A stub ``uv.exe`` stands in for uv where a test needs setup to reach the
environment step.
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
SETUP_BAT = ROOT / "setup.bat"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup.bat runs under cmd.exe")


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "clone"
    d.mkdir()
    shutil.copy(SETUP_BAT, d / "setup.bat")
    return d


def _setup(folder, *, abort_after="portable-choice", answers="\n\n\n\n", extra_env=None,
           path_prefix=None):
    env = dict(os.environ)
    env.pop("LOCALM_SETUP_ABORT_AFTER", None)
    if abort_after:
        env["LOCALM_SETUP_ABORT_AFTER"] = abort_after
    if path_prefix:
        env["PATH"] = str(path_prefix) + os.pathsep + env["PATH"]
    env.update(extra_env or {})
    r = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", r".\setup.bat"],
                       cwd=str(folder), env=env, input=answers, capture_output=True,
                       text=True, timeout=180)
    return r.returncode, r.stdout + r.stderr


def _journal_bytes(folder) -> bytes:
    return (folder / ".localm-setup-journal").read_bytes()


class TestAnInterruptedRun:
    def test_it_stops_after_the_named_step_and_leaves_a_journal_without_complete(self, folder):
        rc, out = _setup(folder)
        assert rc == 99, out
        assert _journal_bytes(folder) == b"begin\tportable-choice\r\ndone\tportable-choice\r\n"
        assert not im.journal_state(folder)["complete"]

    def test_the_python_reader_understands_what_the_batch_script_wrote(self, folder):
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
        assert b"resume\r\n" in _journal_bytes(folder)
        assert "Could not write" not in out

    def test_a_step_left_open_is_named(self, folder):
        (folder / ".localm-setup-journal").write_bytes(
            b"begin\tportable-choice\r\ndone\tportable-choice\r\nbegin\tvenv\r\n")
        rc, out = _setup(folder)
        assert "stopped after 'portable-choice', while running 'venv'" in out

    def test_a_run_that_never_finished_a_step_says_so(self, folder):
        (folder / ".localm-setup-journal").write_bytes(b"begin\tuv-portable\r\n")
        rc, out = _setup(folder)
        assert "before finishing its first step, while running 'uv-portable'" in out

    def test_a_journal_written_by_the_python_side_is_read_too(self, folder):
        im.journal_event(folder, "begin", "venv")
        im.journal_event(folder, "done", "venv")
        im.journal_event(folder, "begin", "native-runtime")
        rc, out = _setup(folder)
        assert "stopped after 'venv', while running 'native-runtime'" in out

    def test_a_finished_journal_is_a_fresh_start(self, folder):
        (folder / ".localm-setup-journal").write_bytes(
            b"begin\told-step\r\ndone\told-step\r\ncomplete\r\n")
        rc, out = _setup(folder)
        assert "previous setup" not in out
        assert _journal_bytes(folder) == b"begin\tportable-choice\r\ndone\tportable-choice\r\n"


class TestAJournalThatCannotBeWritten:
    def test_setup_warns_once_and_carries_on(self, folder):
        (folder / ".localm-setup-journal").mkdir()      # a folder where the file belongs
        rc, out = _setup(folder)
        assert rc == 99, out                              # it still reached the abort hook
        assert out.count("Could not write .localm-setup-journal") == 1


_UV_STUB_SOURCE = r"""
using System;
using System.IO;
public class Stub {
    public static int Main(string[] a) {
        string log = Environment.GetEnvironmentVariable("STUB_LOG");
        if (log != null) File.AppendAllText(log, "uv " + string.Join(" ", a) + "\r\n");
        if (a.Length > 0 && a[0] == "venv") Directory.CreateDirectory(Path.Combine(".venv", "Scripts"));
        return 0;
    }
}
"""


@pytest.fixture(scope="module")
def uv_stub_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("uvstub")
    src = d / "stub.cs"
    src.write_text(_UV_STUB_SOURCE, encoding="utf-8")
    build = ("Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath $env:STUB_SRC) "
             "-OutputAssembly $env:STUB_OUT -OutputType ConsoleApplication")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", build],
                       env={**os.environ, "STUB_SRC": str(src), "STUB_OUT": str(d / "uv.exe")},
                       capture_output=True, text=True, timeout=120)
    assert (d / "uv.exe").is_file(), r.stdout + r.stderr
    return d


class TestACutShortVenv:
    """The venv step is the one an interruption can leave looking fine and be broken:
    ``uv venv`` writes the marker only when it succeeds, so a half-made ``.venv`` has
    none, and setup would treat it as somebody else's and keep it. With a stub uv, setup
    runs up to the venv step and shows whether it rebuilds."""

    def _run(self, folder, tmp_path, journal, stubs, *, marker=False):
        (folder / ".venv").mkdir()                        # half made: no marker file
        if marker:
            (folder / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
        (folder / ".venv" / "half-written").write_text("x", encoding="utf-8")
        if journal is not None:
            (folder / ".localm-setup-journal").write_bytes(journal)
        log = tmp_path / "uv.log"
        log.write_text("", encoding="utf-8")
        temp = tmp_path / "systemtemp"
        temp.mkdir()
        answers = ("1\n" if marker else "") + "2\n\n\n\n\n"
        rc, out = _setup(folder, abort_after="venv", answers=answers,
                         extra_env={"STUB_LOG": str(log), "TEMP": str(temp),
                                    "TMP": str(temp)}, path_prefix=stubs)
        return rc, out, log.read_text(encoding="utf-8"), temp

    def test_a_venv_the_journal_says_was_cut_short_is_rebuilt(self, folder, tmp_path, uv_stub_dir):
        rc, out, uv_log, temp = self._run(
            folder, tmp_path,
            b"begin\tportable-choice\r\ndone\tportable-choice\r\nbegin\tvenv\r\n",
            uv_stub_dir)
        assert rc == 99, out
        assert "stopped while creating .venv; recreating it" in out
        assert "uv venv" in uv_log and "--clear" in uv_log, uv_log
        assert (folder / ".venv" / ".localm-venv").is_file()

    def test_a_cut_short_venv_is_still_rebuilt_after_a_run_that_stopped_earlier(
            self, folder, tmp_path, uv_stub_dir):
        rc, out, uv_log, temp = self._run(
            folder, tmp_path,
            b"begin\tportable-choice\r\ndone\tportable-choice\r\nbegin\tvenv\r\n"
            b"resume\r\nbegin\tportable-choice\r\n", uv_stub_dir)
        assert rc == 99, out
        assert "recreating it" in out
        assert "--clear" in uv_log, uv_log

    def test_a_venv_that_setup_finished_making_is_not_wiped_on_resume(
            self, folder, tmp_path, uv_stub_dir):
        rc, out, uv_log, temp = self._run(
            folder, tmp_path,
            b"begin\tportable-choice\r\ndone\tportable-choice\r\nbegin\tvenv\r\n",
            uv_stub_dir, marker=True)
        assert rc == 99, out
        assert "recreating it" not in out
        assert "uv venv" not in uv_log, uv_log
        assert (folder / ".venv" / "half-written").is_file()

    def test_without_that_journal_a_marker_less_venv_is_left_alone(self, folder, tmp_path, uv_stub_dir):
        rc, out, uv_log, temp = self._run(folder, tmp_path, None, uv_stub_dir)
        assert rc == 99, out
        assert "uv venv" not in uv_log, uv_log
        assert (folder / ".venv" / "half-written").is_file()

    def test_the_scratch_files_stay_inside_the_folder(self, folder, tmp_path, uv_stub_dir):
        rc, out, uv_log, temp = self._run(
            folder, tmp_path,
            b"begin\tportable-choice\r\ndone\tportable-choice\r\nbegin\tvenv\r\n",
            uv_stub_dir)
        assert rc == 99, out
        assert [p.name for p in temp.iterdir() if p.name.lower().startswith("localm")] == []
        assert (folder / ".localm-setup-tmp" / "localm_uv_err.txt").is_file()


class TestAFolderWithABangInItsName:
    """``!`` in the install path must not change where scratch files and the journal go."""

    def test_the_scratch_files_and_the_journal_work_in_a_folder_with_a_bang(
            self, tmp_path, uv_stub_dir):
        clone = tmp_path / "bang!clone"
        clone.mkdir()
        shutil.copy(SETUP_BAT, clone / "setup.bat")
        log = tmp_path / "uv.log"
        log.write_text("", encoding="utf-8")
        rc, out = _setup(clone, abort_after="venv", answers="2\n\n\n\n\n",
                         extra_env={"STUB_LOG": str(log)}, path_prefix=uv_stub_dir)
        assert rc == 99, out
        assert "cannot find the path" not in out.lower(), out
        assert "uv venv" in log.read_text(encoding="utf-8")
        assert (clone / ".venv" / ".localm-venv").is_file()
        assert (clone / ".localm-setup-tmp" / "localm_uv_err.txt").is_file()
        assert im.journal_state(clone)["done"] == ["portable-choice", "venv"]


class TestTheScriptItself:
    """Pins the shape of the instrumentation, so a step cannot be added without its
    journal lines, and scratch files cannot drift back to the system temp folder."""

    text = SETUP_BAT.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().lower().startswith("rem"))

    def test_no_scratch_file_goes_to_the_system_temp_folder(self):
        assert "%TEMP%" not in self.text and "%TMP%" not in self.text

    def test_every_step_that_begins_also_finishes(self):
        begun = re.findall(r"call :jr begin (\S+)", self.code)
        done = re.findall(r"call :step_done (\S+)", self.code)
        assert sorted(b for b in begun if not b.startswith("%")) == \
            sorted(d for d in done if not d.startswith("%"))
        assert begun, "no steps are journaled"

    def test_the_script_ends_by_marking_the_install_complete(self):
        final = re.search(r"^call :jr complete\s*$", self.text, re.M)
        assert final, "setup never marks the install complete"
        assert final.start() > self.text.index("call :step_done record")

    def test_every_step_done_can_stop_the_script(self):
        calls = re.findall(r"^.*call :step_done .*$", self.code, re.M)
        assert calls and all("|| exit /b 99" in c for c in calls), calls

    def test_an_artifact_outside_the_folder_is_announced_before_it_is_created(self):
        shortcut = self.text.index("call :intend_shortcut")
        assert shortcut < self.text.index("CreateShortcut")
        command = self.text.index("call :intend_command")
        assert command < self.text.index("-m localm.globalcmd install")

    def test_the_command_intent_survives_a_bang_in_the_path(self):
        block = self.code[self.code.index("\n:intend_command\n"):]
        assert '"!CD!\\bin\\localm.cmd"' in block.split("goto :eof")[0]

    def test_a_cut_short_runtime_download_is_forced(self):
        assert 'set "SLFORCE=--force"' in self.text
        assert "setup-llama --backend %BACKEND% %SLFORCE%" in self.text

    def test_uninstall_removes_the_journal_with_the_record(self):
        assert '".localm-install.json" ".localm-setup-journal"' in self.text
