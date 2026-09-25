# SPDX-License-Identifier: AGPL-3.0-or-later
"""The uninstall option of the console setups, end to end.

Each test builds a synthetic install in a throwaway folder (the real setup
script copied in, a real pip-less venv, the in-clone runtime folders, runtime
files, a data folder, and a real install record written by
localm.install_manifest), then runs the real setup script and checks what is
left on disk.

Nothing outside tmp_path is touched: the user PATH, shortcuts and the home
folder are not part of these installs, and the only process stopped is one a
test started from its own throwaway folder.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import install_manifest as im

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"

pytestmark = pytest.mark.skipif(not WINDOWS, reason="setup.bat runs under cmd.exe")


@pytest.fixture(scope="module")
def venv_template(tmp_path_factory):
    d = tmp_path_factory.mktemp("uninstall-venv") / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d)],
                   check=True, capture_output=True, timeout=300)
    return d


def _install(clone: Path, template: Path, *, data="portable") -> dict:
    """A recorded Portable install in *clone*; returns key paths."""
    clone.mkdir(parents=True)
    shutil.copy2(ROOT / "setup.bat", clone / "setup.bat")
    shutil.copytree(template, clone / ".venv")
    (clone / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
    for name in (".python", ".cache", ".uv"):
        (clone / name / "sub").mkdir(parents=True)
        (clone / name / "sub" / "f.bin").write_bytes(b"x")
    lib = clone / "runtime" / "localm_llama_runtime" / "lib"
    (lib / "rocblas" / "library").mkdir(parents=True)
    (lib / "llama.dll").write_bytes(b"x")
    (lib / "LICENSE.llama-cpp").write_text("MIT", encoding="utf-8")
    (lib / ".gitkeep").write_text("", encoding="utf-8")
    im.record(clone, venv=str(clone / ".venv"), lib_dir=str(lib),
              runtime_contained=True, python_dir=str(clone / ".python"),
              cache_dir=str(clone / ".cache"), uv_dir=str(clone / ".uv"))
    if data == "portable":
        home = im.prepare_data(clone, portable=True)
    else:
        home = im.prepare_data(clone, data_dir=str(data))
    (home / "chats").mkdir()
    (home / "chats" / "c1.json").write_text("{}", encoding="utf-8")
    return {"lib": lib, "home": home}


def _run(clone: Path, *args, stdin="", env_path=None, timeout=240):
    """Run the clone's setup.bat. The answers are fed from a FILE: `set /p`
    reading a pipe takes the whole buffer and keeps one line, while from a
    file each line reaches its own prompt."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    if env_path is not None:
        env["PATH"] = env_path
    answers = clone.parent / (clone.name + "-answers.txt")
    answers.write_text(stdin, encoding="utf-8", newline="")
    with open(answers, "rb") as fh:
        return subprocess.run(["cmd", "/c", str(clone / "setup.bat"), *args],
                              cwd=str(clone), stdin=fh, capture_output=True,
                              text=True, timeout=timeout, env=env)


def _runtime_gone(clone: Path) -> bool:
    return not any((clone / n).exists() for n in (".venv", ".python", ".cache", ".uv"))


def test_uninstall_yes_removes_the_install_and_keeps_the_data(tmp_path, venv_template):
    clone = tmp_path / "bang!clone"
    p = _install(clone, venv_template)
    out = _run(clone, "uninstall", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert _runtime_gone(clone), out.stdout
    assert not (p["lib"] / "llama.dll").exists()
    assert not (p["lib"] / "rocblas").exists()
    assert not (p["lib"] / "LICENSE.llama-cpp").exists()
    assert (p["lib"] / ".gitkeep").exists()
    assert (p["home"] / "chats" / "c1.json").exists()
    assert not (clone / im.MANIFEST_NAME).exists()
    assert not (clone / im.PENDING_NAME).exists()
    assert "KEPT:" in out.stdout and "LocaLM was removed from this folder." in out.stdout


def test_uninstall_yes_purge_deletes_the_saved_data_too(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, "uninstall", "--purge-data", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert _runtime_gone(clone)
    assert not p["home"].exists()
    assert "DELETED:" in out.stdout


def test_uninstall_purge_keeps_a_shared_folders_other_files(tmp_path, venv_template):
    shared = tmp_path / "Shared AI"
    (shared / "models").mkdir(parents=True)
    (shared / "models" / "theirs.gguf").write_bytes(b"x")
    clone = tmp_path / "clone"
    p = _install(clone, venv_template, data=shared)
    out = _run(clone, "uninstall", "--purge-data", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert (shared / "models" / "theirs.gguf").exists()
    assert not (shared / "chats").exists()
    assert not (clone / "localm-home.cfg").exists()
    assert p["home"] == shared


def test_menu_offers_uninstall_for_an_existing_install(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, stdin="2\r\ny\r\ny\r\n")
    assert "LocaLM is already set up in this folder." in out.stdout, out.stdout
    assert "Also delete your saved data? [y/N]" in out.stdout, out.stdout
    assert _runtime_gone(clone), out.stdout
    assert not p["home"].exists()


def test_menu_keep_data_path(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, stdin="2\r\nn\r\ny\r\n")
    assert _runtime_gone(clone), out.stdout
    assert (p["home"] / "chats" / "c1.json").exists(), out.stdout


def test_menu_cancel_changes_nothing(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, stdin="3\r\n")
    assert "Nothing changed." in out.stdout, out.stdout
    assert (clone / ".venv").is_dir() and (p["lib"] / "llama.dll").exists()
    assert (clone / im.MANIFEST_NAME).exists()


def test_declining_the_final_question_changes_nothing(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, "uninstall", stdin="n\r\nn\r\nn\r\nn\r\n")
    assert "Nothing changed." in out.stdout, out.stdout
    assert (clone / ".venv").is_dir() and (p["lib"] / "llama.dll").exists()


def test_a_running_localm_is_stopped_first(tmp_path, venv_template):
    clone = tmp_path / "clone"
    _install(clone, venv_template)
    proc = subprocess.Popen([str(clone / ".venv" / "Scripts" / "python.exe"), "-c",
                             "import time; time.sleep(120)"], cwd=str(clone))
    try:
        time.sleep(1.0)
        out = _run(clone, "uninstall", "--yes")
        assert out.returncode == 0, (out.stdout, out.stderr)
        assert "Stopped:" in out.stdout, out.stdout
        proc.wait(timeout=30)
        assert _runtime_gone(clone), out.stdout
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


def test_without_any_python_only_the_fixed_folders_go(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    shutil.rmtree(clone / ".venv" / "Scripts")          # the environment is broken
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    out = _run(clone, "uninstall", "--yes", env_path=str(system32))
    assert "No Python was found" in out.stdout, (out.stdout, out.stderr)
    assert _runtime_gone(clone), out.stdout
    assert (clone / im.MANIFEST_NAME).exists()           # kept for a later full run
    assert (p["lib"] / "llama.dll").exists()


def test_finish_uninstall_removes_only_allowlisted_pending_names(tmp_path):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup.bat", clone / "setup.bat")
    for name in (".venv", ".python", "src"):
        (clone / name).mkdir()
        (clone / name / "f").write_bytes(b"x")
    (clone / im.PENDING_NAME).write_text(".venv\r\n.python\r\nsrc\r\n..\r\n", encoding="ascii")
    (clone / im.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    out = _run(clone, "finish-uninstall")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert not (clone / ".venv").exists() and not (clone / ".python").exists()
    assert (clone / "src" / "f").exists()
    assert not (clone / im.PENDING_NAME).exists()
    assert not (clone / im.MANIFEST_NAME).exists()


def _gui_finish(clone: Path):
    """Run setup-gui.bat's exit-42 path (the `if "!RC!"=="42"` jump and the
    :finish_uninstall section, verbatim) in *clone*."""
    gui_bat = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    jump = 'if "!RC!"=="42" goto finish_uninstall'
    assert jump in gui_bat, "the exit-42 jump moved"
    section = gui_bat[gui_bat.index("\n:finish_uninstall\n"):]
    probe = clone / "probe.bat"
    probe.write_text(("@echo off\nsetlocal EnableExtensions EnableDelayedExpansion\n"
                      'set "RC=42"\n' + jump + "\necho NOT_REACHED\nexit /b 9\n"
                      + section).replace("\n", "\r\n"), encoding="utf-8")
    return subprocess.run(["cmd", "/c", str(probe)], cwd=str(clone), capture_output=True,
                          text=True, timeout=120, stdin=subprocess.DEVNULL)


def _pending_layout(clone: Path) -> None:
    clone.mkdir()
    shutil.copy2(ROOT / "setup.bat", clone / "setup.bat")
    for name in (".venv", ".python", ".uv"):
        (clone / name).mkdir()
        (clone / name / "f").write_bytes(b"x")
    (clone / im.PENDING_NAME).write_text(".venv\n.python\n.uv\n", encoding="ascii")
    (clone / im.MANIFEST_NAME).write_text("{}", encoding="utf-8")


def test_setup_gui_finishes_an_uninstall_the_window_left_pending(tmp_path):
    """setup-gui.bat hands exit code 42 from the window to setup.bat
    finish-uninstall, which removes the runtime folders the window ran on."""
    clone = tmp_path / "clone"
    _pending_layout(clone)
    out = _gui_finish(clone)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "LocaLM was uninstalled." in out.stdout, out.stdout
    assert "NOT_REACHED" not in out.stdout
    assert _runtime_gone(clone)
    assert not (clone / im.MANIFEST_NAME).exists()


def test_setup_gui_reports_folders_it_could_not_remove(tmp_path):
    clone = tmp_path / "clone"
    _pending_layout(clone)
    held = open(clone / ".python" / "f", "rb")
    try:
        out = _gui_finish(clone)
    finally:
        held.close()
    assert out.returncode == 1, (out.stdout, out.stderr)
    assert "LocaLM was uninstalled." not in out.stdout
    assert "Some LocaLM folders could not be removed" in out.stdout
    assert (clone / im.PENDING_NAME).exists()


def test_a_folder_held_open_is_reported_and_the_record_kept(tmp_path, venv_template):
    clone = tmp_path / "clone"
    _install(clone, venv_template)
    held = open(clone / ".python" / "sub" / "f.bin", "rb")
    try:
        out = _run(clone, "uninstall", "--yes", timeout=300)
    finally:
        held.close()
    assert out.returncode == 1, (out.stdout, out.stderr)
    assert "Could not remove .\\.python" in out.stdout, out.stdout
    assert (clone / im.PENDING_NAME).exists()
    data = json.loads((clone / im.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data.get("python_dir")
    out2 = _run(clone, "finish-uninstall")
    assert out2.returncode == 0, (out2.stdout, out2.stderr)
    assert _runtime_gone(clone)
    assert not (clone / im.MANIFEST_NAME).exists()
