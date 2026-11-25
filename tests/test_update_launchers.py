# SPDX-License-Identifier: AGPL-3.0-or-later
"""An update or rollback leaves the launch scripts working.

Windows: cmd.exe reads a batch file one line at a time and, after a command
returns, resumes at the byte offset where that command's line ended, in whatever
file is at that path by then. An update or rollback replaces localm.bat (and
rollback.bat) while the localm it started is running. The tests run the REAL
scripts through a real cmd.exe while a fake localm swaps the file underneath, with
the shipped files of older releases (tests/fixtures/old_launchers, byte for byte
what those releases installed) on the other side of the swap.

Linux and macOS: the launch scripts keep their execute permission through an
update, both when this build's updater extracts the archive and, through the
build's finishing step (localm/_finish_update.py), when an older updater did.
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from localm import _apply_update as au
from tests._fake_clone import CREATE_NO_WINDOW, make_windows_clone

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "tests" / "fixtures" / "old_launchers"
OLD_LOCALM_BATS = ("localm-0.3.3.bat", "localm-0.3.4b3.bat")
ACCESS_VIOLATION = "-1073741819"
nt_only = pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
posix_only = pytest.mark.skipif(os.name == "nt", reason="file modes are POSIX")


def _crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def _resume_offsets(bat: bytes) -> list:
    """The byte offsets cmd.exe resumes at after each line of *bat* that starts
    localm (the end of that line)."""
    offsets, pos = [], 0
    for line in bat.split(b"\n"):
        pos += len(line) + 1
        text = line.strip()
        if b'"%PY%" -m localm' in text and not text.lower().startswith(b"rem"):
            offsets.append(pos)
    return offsets


# --------------------------------------------------------------------------- #
#  localm.bat: where an older localm.bat resumes                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("old", OLD_LOCALM_BATS)
def test_every_older_resume_offset_starts_a_resume_line(old):
    """Runs on every platform: it reads bytes only."""
    shipped = _crlf((ROOT / "localm.bat").read_bytes())
    offsets = _resume_offsets(_crlf((OLD / old).read_bytes()))
    assert len(offsets) == 2, offsets
    for at in offsets:
        assert shipped[at - 2:at] == b"\r\n", (old, at)
        assert shipped[at:].startswith(b'set "RC=%errorlevel%" & goto :resumed\r\n'), (old, at)


def test_each_command_that_starts_localm_ends_in_exit():
    """After localm returns, nothing more is read from localm.bat: the command
    that starts it is one block whose every branch ends in exit."""
    text = _crlf((ROOT / "localm.bat").read_bytes()).decode()
    lines = text.split("\r\n")
    starts = [i for i, ln in enumerate(lines) if ln.startswith('"%PY%" -m localm')]
    assert len(starts) == 2
    for i in starts:
        block, depth = [], 0
        for ln in lines[i:]:
            block.append(ln)
            depth += ln.count("(") - ln.count(")")
            if depth == 0:
                break
        assert "exit /b 0" in block[0] or any("exit /b 0" in ln for ln in block)
        assert block[-2].strip() == "call exit /b %%RC%%" and block[-1].strip() == ")", block


# --------------------------------------------------------------------------- #
#  localm.bat and rollback.bat replaced while they run, through a real cmd.exe #
# --------------------------------------------------------------------------- #

_SWAP_MAIN = (
    "import os, shutil, sys\n"
    "src = os.environ['SWAP_FROM']\n"
    "dst = os.environ['SWAP_TO']\n"
    "shutil.copyfile(src, dst)\n"
    "print('FAKE', ' '.join(sys.argv[1:]), flush=True)\n"
    "sys.exit(int(os.environ.get('SWAP_RC', '0')))\n"
)


def _swap_run(tmp_path, running: bytes, replacement: bytes, rc: str, args: list,
              script: str = "localm.bat", extra_env: dict | None = None,
              call: bool = True):
    """Run *running* as *script* in a throwaway install whose fake program
    replaces it with *replacement* and exits with *rc*: through ``cmd /c call``
    (a prompt or another batch file) or, without *call*, ``cmd /c`` alone (a
    shortcut, Explorer, a scheduled task). Returns the completed process."""
    clone = make_windows_clone(tmp_path, script, text=_crlf(running).decode())
    (clone / script).write_bytes(_crlf(running))
    swap = tmp_path / "replacement.bat"
    swap.write_bytes(_crlf(replacement))
    (tmp_path / "fake" / "localm" / "__main__.py").write_text(_SWAP_MAIN, encoding="utf-8")
    if script == "rollback.bat":
        (clone / "scripts").mkdir()
        (clone / "scripts" / "rollback_update.py").write_text(_SWAP_MAIN, encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCALM_MODEL", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update({"PYTHONPATH": str(tmp_path / "fake"), "PYTHONNOUSERSITE": "1",
                "SWAP_FROM": str(swap), "SWAP_TO": str(clone / script), "SWAP_RC": rc})
    env.update(extra_env or {})
    return subprocess.run(["cmd", "/d", "/c", *(["call"] if call else []), str(clone / script),
                           *args],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          cwd=str(tmp_path), env=env, timeout=60,
                          creationflags=CREATE_NO_WINDOW)


def _check_clean(done, rc: str, *, held: bool):
    out = [ln for ln in done.stdout.splitlines() if ln.strip()]
    stray = [ln for ln in out if not ln.startswith(("FAKE", "Press any key",
                                                     "Starting LocaLM"))]
    assert not stray and not done.stderr.strip(), (done.stdout, done.stderr)
    assert any(ln.startswith("FAKE") for ln in out), done.stdout
    assert ("press any key" in done.stdout.lower()) == held, done.stdout
    assert done.returncode & 0xFFFFFFFF == int(rc) & 0xFFFFFFFF, done.returncode


@nt_only
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("old", OLD_LOCALM_BATS)
@pytest.mark.parametrize("rc", ["0", "3", ACCESS_VIOLATION])
def test_an_update_from_an_older_localm_bat_ends_clean(tmp_path, old, rc, call):
    """`localm.bat update -y` on an older release, whose updater installs this
    localm.bat: the old file's resume offset lands on a resume line."""
    done = _swap_run(tmp_path, (OLD / old).read_bytes(),
                     (ROOT / "localm.bat").read_bytes(), rc, ["update", "-y"], call=call)
    _check_clean(done, rc, held=rc != "0")


@nt_only
@pytest.mark.parametrize("old", OLD_LOCALM_BATS)
def test_a_chat_ended_after_an_update_from_an_older_localm_bat_ends_clean(tmp_path, old):
    done = _swap_run(tmp_path, (OLD / old).read_bytes(), (ROOT / "localm.bat").read_bytes(),
                     "0", [], extra_env={"LOCALM_MODEL": "m"})
    _check_clean(done, "0", held=False)


# Any line read from this file prints STRAY.
_ALL_STRAY = b"@echo STRAY\r\n" * 600


@nt_only
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("replacement", ["localm-0.3.3.bat", "stray"])
@pytest.mark.parametrize("rc,args,held", [
    ("0", ["update", "-y"], False),
    ("0", ["update", "--rollback", "-y"], False),
    ("3", ["update", "--rollback", "-y"], True),
    (ACCESS_VIOLATION, ["update", "-y"], True),
    ("0", ["update", "-y", "--debug"], True),
])
def test_this_localm_bat_replaced_while_localm_runs_ends_clean(tmp_path, replacement, rc,
                                                               args, held, call):
    """A rollback to an older release, or a later update, replaces this localm.bat
    while it runs; whatever replaced it, nothing more is read from it, and the
    exit code of localm reaches the caller."""
    other = _ALL_STRAY if replacement == "stray" else (OLD / replacement).read_bytes()
    done = _swap_run(tmp_path, (ROOT / "localm.bat").read_bytes(), other, rc, args,
                     call=call)
    _check_clean(done, rc, held=held)


@nt_only
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("args,held", [(["--yes"], False), (["-y"], False), ([], True)])
@pytest.mark.parametrize("rc", ["0", "2"])
def test_rollback_bat_replaced_while_it_runs_ends_clean(tmp_path, rc, args, held, call):
    done = _swap_run(tmp_path, (ROOT / "rollback.bat").read_bytes(), _ALL_STRAY, rc,
                     args, script="rollback.bat", call=call)
    _check_clean(done, rc, held=held)


# --------------------------------------------------------------------------- #
#  Execute permissions                                                         #
# --------------------------------------------------------------------------- #

def test_executables_match_the_tracked_file_modes():
    """EXECUTABLES lists exactly the files git tracks with mode 100755."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s"], capture_output=True,
                          text=True, check=True,
                          creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    tracked = {line.split("\t", 1)[1] for line in done.stdout.splitlines()
               if line.startswith("100755 ")}
    assert tracked and set(au.EXECUTABLES) == tracked


def _build_zip(path: Path, members: dict, *, create_system: int) -> Path:
    """A zip of *members* (name -> (mode, bytes)) whose entries record *mode*."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, (mode, data) in members.items():
            info = zipfile.ZipInfo(name)
            info.create_system = create_system
            info.external_attr = mode << 16
            z.writestr(info, data)
    path.write_bytes(buf.getvalue())
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
@pytest.mark.parametrize("create_system", [0, 3])
def test_extract_keeps_the_execute_permission_the_archive_records(tmp_path, create_system):
    """create_system 0 is what a build made on Windows records; the mode bits are
    there either way."""
    z = _build_zip(tmp_path / "build.zip", {
        "localm.sh": (0o100755, b"#!/bin/sh\n"),
        "VERSION": (0o100600, b"9.9.9\n"),
        "pyproject.toml": (0o100644, b""),
    }, create_system=create_system)
    plain = tmp_path / "plain"
    plain.write_bytes(b"")
    root = au.extract(z, tmp_path / "staging")
    assert _mode(root / "localm.sh") & 0o100
    assert _mode(root / "localm.sh") == _mode(plain) | ((_mode(plain) & 0o444) >> 2)
    assert _mode(root / "VERSION") == _mode(plain)
    assert not _mode(root / "pyproject.toml") & 0o111


@posix_only
def test_an_extracted_build_keeps_its_modes_through_the_swap(tmp_path):
    z = _build_zip(tmp_path / "build.zip", {
        "localm.sh": (0o100755, b"#!/bin/sh\n"),
        "VERSION": (0o100644, b"9.9.9\n"),
        "pyproject.toml": (0o100644, b""),
    }, create_system=0)
    staged = au.extract(z, tmp_path / "staging")
    install = tmp_path / "install"
    install.mkdir()
    (install / "localm.sh").write_bytes(b"old\n")
    os.chmod(install / "localm.sh", 0o644)
    au.apply_files(staged, install, au.swap_entries(staged))
    assert _mode(install / "localm.sh") & 0o100


@posix_only
def test_restore_executables_sets_only_the_listed_files(tmp_path):
    for rel in ("localm.sh", "setup.sh", "README.md"):
        (tmp_path / rel).write_text("x", encoding="utf-8")
        os.chmod(tmp_path / rel, 0o644)
    assert au.restore_executables(tmp_path) == ["localm.sh", "setup.sh"]
    assert _mode(tmp_path / "localm.sh") == 0o755
    assert _mode(tmp_path / "README.md") == 0o644
    assert au.restore_executables(tmp_path) == []


def _finish_install(tmp_path: Path) -> Path:
    """An install holding this build's finishing step and its helper."""
    install = tmp_path / "install"
    (install / "localm").mkdir(parents=True)
    for name in ("_finish_update.py", "_apply_update.py"):
        shutil.copy2(ROOT / "localm" / name, install / "localm" / name)
    return install


def _run_finish(install: Path, result: Path):
    """The command line localm.updater_finish.run uses."""
    from localm import updater_finish
    cmd = [sys.executable, "-I", "-X", "utf8", "-u", str(install / updater_finish.MODULE),
           "0.3.3", "0.3.4", "reboot", str(result)]
    return subprocess.run(cmd, cwd=str(install), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60,
                          creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)


@posix_only
def test_the_finishing_step_gives_the_launch_scripts_their_execute_permission(tmp_path):
    install = _finish_install(tmp_path)
    for rel in ("localm.sh", "rollback.sh"):
        (install / rel).write_text("#!/bin/sh\n", encoding="utf-8")
        os.chmod(install / rel, 0o644)
    result = tmp_path / "finish-result.json"
    done = _run_finish(install, result)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _mode(install / "localm.sh") == 0o755
    assert _mode(install / "rollback.sh") == 0o755
    assert "localm.sh" in done.stdout and not done.stderr
    assert not result.exists()


@posix_only
def test_the_finishing_step_reports_a_permission_it_cannot_set_and_exits_0(tmp_path):
    install = _finish_install(tmp_path)
    (install / "localm.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(install / "localm.sh", 0o644)
    (install / "localm" / "_apply_update.py").write_text("raise OSError('denied')\n",
                                                          encoding="utf-8")
    done = _run_finish(install, tmp_path / "finish-result.json")
    assert done.returncode == 0
    assert "Could not give the launch scripts their execute permission" in done.stdout
    assert "chmod +x" in done.stdout


@nt_only
def test_the_finishing_step_does_nothing_on_windows(tmp_path):
    install = _finish_install(tmp_path)
    result = tmp_path / "finish-result.json"
    done = _run_finish(install, result)
    assert done.returncode == 0 and not done.stdout.strip() and not done.stderr.strip()
    assert not result.exists()


def test_the_finishing_step_takes_over_no_built_in_step(tmp_path):
    from localm import updater_finish
    assert updater_finish.present(ROOT)
    install = _finish_install(tmp_path)
    result = tmp_path / "finish-result.json"
    done = _run_finish(install, result)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not result.exists()
    assert updater_finish._read_result(result, None) == frozenset()
