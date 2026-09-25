# SPDX-License-Identifier: AGPL-3.0-or-later
"""The uninstall option of setup.sh, end to end, on Linux/macOS.

Each test builds a synthetic install in a throwaway folder (setup.sh copied in,
a real pip-less venv, the in-clone runtime folders, runtime files, a data
folder, and a real install record written by localm.install_manifest), runs the
real setup.sh, and checks what is left on disk. HOME points into tmp_path, so
shell startup files and the user profile are never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import install_manifest as im

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                                reason="setup.sh runs under bash on Linux/macOS")


@pytest.fixture(scope="module")
def venv_template(tmp_path_factory):
    d = tmp_path_factory.mktemp("sh-uninstall-venv") / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d)],
                   check=True, capture_output=True, timeout=300)
    return d


def _install(clone: Path, template: Path, *, data="portable") -> dict:
    clone.mkdir(parents=True)
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    shutil.copytree(template, clone / ".venv", symlinks=True)
    (clone / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
    for name in (".python", ".cache", ".uv"):
        (clone / name / "sub").mkdir(parents=True)
        (clone / name / "sub" / "f.bin").write_bytes(b"x")
    lib = clone / "runtime" / "localm_llama_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "libllama.so").write_bytes(b"x")
    (lib / ".gitkeep").write_text("", encoding="utf-8")
    (clone / "LocaLM.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    im.record(clone, venv=str(clone / ".venv"), lib_dir=str(lib),
              runtime_contained=True, python_dir=str(clone / ".python"),
              cache_dir=str(clone / ".cache"), uv_dir=str(clone / ".uv"),
              files=[str(clone / "LocaLM.desktop")])
    home = (im.prepare_data(clone, portable=True) if data == "portable"
            else im.prepare_data(clone, data_dir=str(data)))
    (home / "chats").mkdir()
    (home / "chats" / "c1.json").write_text("{}", encoding="utf-8")
    return {"lib": lib, "home": home}


def _run(clone: Path, tmp_path: Path, *args, stdin="", path=None, timeout=240):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    home = tmp_path / "userhome"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env.pop("XDG_CONFIG_HOME", None)
    bash = shutil.which("bash")
    if path is not None:
        env["PATH"] = path
    return subprocess.run([bash, str(clone / "setup.sh"), *args], cwd=str(clone),
                          input=stdin, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _runtime_gone(clone: Path) -> bool:
    return not any((clone / n).exists() for n in (".venv", ".python", ".cache", ".uv"))


def test_uninstall_yes_removes_the_install_and_keeps_the_data(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, tmp_path, "--uninstall", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert _runtime_gone(clone), out.stdout
    assert not (p["lib"] / "libllama.so").exists()
    assert (p["lib"] / ".gitkeep").exists()
    assert not (clone / "LocaLM.desktop").exists()
    assert (p["home"] / "chats" / "c1.json").exists()
    assert not (clone / im.MANIFEST_NAME).exists()
    assert not (clone / im.PENDING_NAME).exists()
    assert "LocaLM was removed from this folder." in out.stdout


def test_uninstall_purge_in_a_shared_folder_keeps_the_rest(tmp_path, venv_template):
    shared = tmp_path / "Shared AI"
    (shared / "models").mkdir(parents=True)
    (shared / "models" / "theirs.gguf").write_bytes(b"x")
    clone = tmp_path / "clone"
    _install(clone, venv_template, data=shared)
    out = _run(clone, tmp_path, "--uninstall", "--purge-data", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert (shared / "models" / "theirs.gguf").exists()
    assert not (shared / "chats").exists()
    assert not (clone / "localm-home.cfg").exists()


def test_menu_uninstall_with_data_deletion(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, tmp_path, stdin="2\ny\ny\n")
    assert "LocaLM is already set up in this folder." in out.stdout, out.stdout
    assert _runtime_gone(clone), (out.stdout, out.stderr)
    assert not p["home"].exists()


def test_menu_cancel_changes_nothing(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    out = _run(clone, tmp_path, stdin="3\n")
    assert "Nothing changed." in out.stdout, out.stdout
    assert (clone / ".venv").is_dir() and (p["lib"] / "libllama.so").exists()


def test_a_running_localm_is_stopped_first(tmp_path, venv_template):
    clone = tmp_path / "clone"
    _install(clone, venv_template)
    proc = subprocess.Popen([str(clone / ".venv" / "bin" / "python"), "-c",
                             "import time; time.sleep(120)"], cwd=str(clone))
    try:
        time.sleep(1.0)
        out = _run(clone, tmp_path, "--uninstall", "--yes")
        assert out.returncode == 0, (out.stdout, out.stderr)
        assert "Stopped:" in out.stdout, out.stdout
        proc.wait(timeout=30)
        assert _runtime_gone(clone)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


def test_stale_uv_lines_for_this_folder_are_removed(tmp_path, venv_template):
    clone = tmp_path / "clone"
    _install(clone, venv_template)
    home = tmp_path / "userhome"
    home.mkdir()
    profile = home / ".profile"
    profile.write_text(f'export A=1\n. "{clone}/.uv/env"\n. "/elsewhere/.uv/env"\n',
                       encoding="utf-8")
    out = _run(clone, tmp_path, "--uninstall", "--yes")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert profile.read_text(encoding="utf-8") == 'export A=1\n. "/elsewhere/.uv/env"\n'


def test_without_any_python_only_the_fixed_folders_go(tmp_path, venv_template):
    clone = tmp_path / "clone"
    p = _install(clone, venv_template)
    shutil.rmtree(clone / ".venv" / "bin")
    empty = tmp_path / "emptybin"
    empty.mkdir()
    for tool in ("rm", "chmod", "cat", "dirname", "uname"):
        found = shutil.which(tool)
        if found:
            (empty / tool).symlink_to(found)
    out = _run(clone, tmp_path, "--uninstall", "--yes", path=str(empty))
    assert "No Python was found" in out.stdout, (out.stdout, out.stderr)
    assert _runtime_gone(clone), out.stdout
    assert (clone / im.MANIFEST_NAME).exists()
    assert (p["lib"] / "libllama.so").exists()


def test_setup_gui_finishes_an_uninstall_the_window_left_pending(tmp_path):
    """setup-gui.sh hands exit code 42 from the window to setup.sh
    --finish-uninstall. The window is stood in for by a uv that exits 42."""
    clone = tmp_path / "clone"
    clone.mkdir()
    for name in ("setup.sh", "setup-gui.sh"):
        shutil.copy2(ROOT / name, clone / name)
    (clone / ".uv").mkdir()
    fake_uv = clone / ".uv" / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    (clone / ".venv").mkdir()
    (clone / ".python").mkdir()
    (clone / im.PENDING_NAME).write_text(".venv\n.python\n.uv\n", encoding="ascii")
    (clone / im.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    out = subprocess.run([shutil.which("bash"), str(clone / "setup-gui.sh")], cwd=str(clone),
                         capture_output=True, text=True, timeout=120, env=env,
                         stdin=subprocess.DEVNULL)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert _runtime_gone(clone)
    assert not (clone / im.MANIFEST_NAME).exists()


def test_finish_uninstall_removes_only_allowlisted_names(tmp_path):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    for name in (".venv", ".uv", "src"):
        (clone / name).mkdir()
        (clone / name / "f").write_bytes(b"x")
    (clone / im.PENDING_NAME).write_text(".venv\n.uv\nsrc\n..\n", encoding="ascii")
    (clone / im.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    out = _run(clone, tmp_path, "--finish-uninstall")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert not (clone / ".venv").exists() and not (clone / ".uv").exists()
    assert (clone / "src" / "f").exists()
    assert not (clone / im.PENDING_NAME).exists()
    assert not (clone / im.MANIFEST_NAME).exists()
