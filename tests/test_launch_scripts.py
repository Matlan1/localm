# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shipped launch scripts: localm.bat, localm.sh, localm-launcher.bat,
localm-launcher.sh, localm.py and ``python -m localm``.

The batch tests run the REAL shipped files from a throwaway install folder
(tests/_fake_clone.py) whose fake ``localm`` package records the working
directory and arguments it was started with.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._fake_clone import CREATE_NO_WINDOW, make_windows_clone, run_bat

ROOT = Path(__file__).resolve().parents[1]
nt_only = pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
posix_only = pytest.mark.skipif(os.name == "nt", reason="bash scripts run on POSIX")


# --------------------------------------------------------------------------- #
#  python -m localm outside localm's environment                               #
# --------------------------------------------------------------------------- #

def _base_python() -> str | None:
    exe = getattr(sys, "_base_executable", None)
    if not exe or os.path.normcase(exe) == os.path.normcase(sys.executable):
        return None
    return exe


def test_python_m_localm_without_click_prints_the_venv_sentence(tmp_path):
    base = _base_python()
    if base is None:
        pytest.skip("no base interpreter outside the venv")
    flags = CREATE_NO_WINDOW if os.name == "nt" else 0
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    has_click = subprocess.run([base, "-c", "import click"], capture_output=True,
                               env=env, creationflags=flags, timeout=60)
    if has_click.returncode == 0:
        pytest.skip("the base interpreter has click installed")
    done = subprocess.run([base, "-m", "localm", "--help"], capture_output=True, text=True,
                          cwd=str(ROOT), env=env, creationflags=flags, timeout=60)
    assert done.returncode != 0
    assert "must be run from its virtual environment" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr, done.stderr


def test_python_m_localm_still_imports_the_cli_at_module_level():
    tree = (ROOT / "localm" / "__main__.py").read_text(encoding="utf-8")
    guard = tree.index("require_venv()")
    assert tree.index("from localm.cli import main") > guard


def test_localm_py_names_this_platforms_entry_point():
    done = subprocess.run([sys.executable, str(ROOT / "localm.py")], capture_output=True,
                          text=True, timeout=60,
                          creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert done.returncode == 1
    want = ".venv\\Scripts\\localm" if os.name == "nt" else ".venv/bin/localm"
    other = ".venv/bin/localm" if os.name == "nt" else ".venv\\Scripts\\localm"
    assert want in done.stderr and other not in done.stderr
    assert "system Python" not in done.stderr


# --------------------------------------------------------------------------- #
#  localm.bat                                                                  #
# --------------------------------------------------------------------------- #

@nt_only
class TestLocalmBat:
    def test_a_relative_argument_keeps_the_folder_it_was_typed_in(self, tmp_path):
        clone = make_windows_clone(tmp_path)
        here = tmp_path / "work"
        here.mkdir()
        done, probe = run_bat(clone, ["add", ".\\model.gguf"], cwd=here)
        assert probe is not None, done.stdout + done.stderr
        assert os.path.normcase(probe["cwd"]) == os.path.normcase(str(here))
        assert probe["argv"] == ["add", ".\\model.gguf"]

    def test_no_argument_opens_the_first_chat_model_not_an_embedder(self, tmp_path):
        clone = make_windows_clone(tmp_path)
        reg = {"aaa-embed": {"model_type": "embedding"}, "bbb-vae": {"model_type": "vae"},
               "zzz-chat": {"model_type": "llm"}, "mmm-legacy": {}}
        done, probe = run_bat(clone, [], cwd=tmp_path, registry=reg)
        assert probe is not None, done.stdout + done.stderr
        assert probe["argv"] == ["run", "mmm-legacy"]

    def test_localm_model_with_a_space_is_one_argument(self, tmp_path):
        clone = make_windows_clone(tmp_path)
        done, probe = run_bat(clone, [], cwd=tmp_path,
                              extra_env={"LOCALM_MODEL": "my model.gguf"})
        assert probe is not None, done.stdout + done.stderr
        assert probe["argv"] == ["run", "my model.gguf"]

    def test_no_chat_model_names_this_script_and_holds_the_window(self, tmp_path):
        clone = make_windows_clone(tmp_path)
        done, probe = run_bat(clone, [], cwd=tmp_path,
                              registry={"only-embed": {"model_type": "embedding"}})
        assert probe is None
        assert "No chat model registered yet" in done.stdout
        assert f'"{clone}\\localm.bat" pull' in done.stdout
        assert "press any key" in done.stdout.lower()
        assert done.returncode == 1

    def test_no_venv_says_so_and_runs_no_other_python(self, tmp_path):
        clone = make_windows_clone(tmp_path, venv=False)
        done, probe = run_bat(clone, ["--version"], cwd=tmp_path)
        assert probe is None
        assert "No .venv found" in done.stdout
        assert "setup.bat" in done.stdout
        assert done.returncode == 1


@nt_only
class TestLocalmLauncherBat:
    def test_no_venv_never_falls_back_to_a_python_on_path(self, tmp_path):
        text = (ROOT / "localm-launcher.bat").read_text(encoding="utf-8")
        assert "where pythonw" not in text
        clone = make_windows_clone(tmp_path, "localm-launcher.bat", venv=False)
        done, _ = run_bat(clone, [], cwd=tmp_path, script="localm-launcher.bat")
        assert "No .venv found" in done.stdout
        assert "setup.bat" in done.stdout


# --------------------------------------------------------------------------- #
#  localm.sh and localm-launcher.sh                                            #
# --------------------------------------------------------------------------- #

def _posix_clone(tmp_path: Path, *, entry: bool = True, tk: bool = True) -> Path:
    clone = tmp_path / "install dir"
    (clone / ".venv" / "bin").mkdir(parents=True)
    for name in ("localm.sh", "localm-launcher.sh"):
        shutil.copy2(ROOT / name, clone / name)
        (clone / name).chmod(0o755)
    py = clone / ".venv" / "bin" / "python"
    py.write_text("#!/bin/sh\n"
                  + ('if [ "$1" = "-c" ]; then exit 1; fi\n' if not tk else
                     'if [ "$1" = "-c" ]; then exit 0; fi\n')
                  + 'echo "PY $(pwd) $*"\n', encoding="utf-8")
    py.chmod(0o755)
    if entry:
        cli = clone / ".venv" / "bin" / "localm"
        cli.write_text('#!/bin/sh\necho "CLI $(pwd) $*"\n', encoding="utf-8")
        cli.chmod(0o755)
    return clone


@posix_only
class TestPosixScripts:
    def test_localm_sh_keeps_the_callers_folder(self, tmp_path):
        clone = _posix_clone(tmp_path)
        here = tmp_path / "work"
        here.mkdir()
        done = subprocess.run(["bash", str(clone / "localm.sh"), "add", "./m.gguf"],
                              capture_output=True, text=True, cwd=str(here), timeout=60)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"CLI {here} add ./m.gguf"

    def test_localm_sh_missing_entry_point_points_at_setup_not_a_bare_uv(self, tmp_path):
        clone = _posix_clone(tmp_path, entry=False)
        done = subprocess.run(["bash", str(clone / "localm.sh"), "gui"],
                              capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
        assert done.returncode == 1
        assert "setup.sh" in done.stderr
        assert "uv pip" not in done.stderr

    def test_launcher_sh_without_tk_says_what_to_install(self, tmp_path):
        clone = _posix_clone(tmp_path, tk=False)
        done = subprocess.run(["bash", str(clone / "localm-launcher.sh")],
                              capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
        assert done.returncode == 1
        assert "needs Tk" in done.stderr
        assert "PY " not in done.stdout

    def test_launcher_sh_with_tk_starts_the_launcher(self, tmp_path):
        clone = _posix_clone(tmp_path, tk=True)
        done = subprocess.run(["bash", str(clone / "localm-launcher.sh")],
                              capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"PY {clone} {clone}/launcher.pyw"
