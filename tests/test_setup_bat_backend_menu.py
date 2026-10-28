# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat's llama.cpp backend menu comes from localm.setup_spec.

The menu rows, the prompt range and the "provision later" hint come from
``python -m localm.setup_spec backend-menu <rec>`` and ``backend-hint``, the
same calls setup.sh makes. These tests cut the real blocks out of setup.bat and
run them under cmd.exe, with ``.venv\\Scripts\\python`` replaced by a stub that
prints canned rows (or by the real interpreter), so what they pin is the batch
behaviour itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = (ROOT / "setup.bat").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="runs setup.bat blocks under cmd.exe")

MENU_START = "rem ---- choose the llama.cpp backend"
MENU_END = "rem ---- PyTorch + transformers for HuggingFace models"
HINT_START = "rem  SLB: the backends setup-llama has a build of"
HINT_END = 'echo.\nif /i "%BACKEND%"=="own" ('

ROWS = [
    ("1", "cuda", "cuda - NVIDIA, peak performance   (recommended for your hardware)"),
    ("2", "vulkan", "vulkan - any GPU (AMD/NVIDIA/Intel), no vendor toolkit"),
    ("3", "cuda", "cuda - NVIDIA, peak performance   (same as [1])"),
    ("4", "cpu", "cpu - no GPU"),
    ("5", "own", "I will build / provide my own (skip the download)"),
]


def _between(start: str, end: str) -> str:
    text = BAT.replace("\r\n", "\n")
    a = text.index(start)
    return text[a:text.index(end, a)]


_FAKE_SPEC = '''import sys
from pathlib import Path
here = Path(__file__).parent
with open(here / "calls.txt", "a", encoding="utf-8") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
answer = {"backend-menu": "menu.txt", "backend-hint": "hint.txt"}.get(sys.argv[1])
if answer and (here / answer).is_file():
    sys.stdout.write((here / answer).read_text(encoding="utf-8"))
'''


def _venv_python(tmp_path: Path) -> None:
    """A copy of this interpreter at .venv\\Scripts\\python.exe."""
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.executable, scripts / "python.exe")
    if (Path(sys.prefix) / "pyvenv.cfg").is_file():
        shutil.copy2(Path(sys.prefix) / "pyvenv.cfg", tmp_path / ".venv" / "pyvenv.cfg")


def _stub_python(tmp_path: Path, *, menu: list[tuple[str, str, str]] | None,
                 hint: str | None = None) -> Path:
    """.venv\\Scripts\\python.exe with a stand-in localm.setup_spec on
    PYTHONPATH (see _env) that prints menu.txt / hint.txt and records each
    call's arguments in calls.txt."""
    _venv_python(tmp_path)
    pkg = tmp_path / "fake" / "localm"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "setup_spec.py").write_text(_FAKE_SPEC, encoding="utf-8")
    if menu is not None:
        (pkg / "menu.txt").write_text("".join("\t".join(r) + "\n" for r in menu),
                                      encoding="utf-8")
    if hint is not None:
        (pkg / "hint.txt").write_text(hint + "\n", encoding="utf-8")
    return pkg / "calls.txt"


def _env(tmp_path: Path) -> dict[str, str]:
    return {"PYTHONPATH": str(tmp_path / "fake")}


def _run(tmp_path: Path, body: str, stdin: str = "\n", env: dict[str, str] | None = None) -> str:
    script = (
        "@echo off\n"
        "setlocal EnableDelayedExpansion\n"
        'set "STMP=stmp"\n'
        + body
        + "\nexit /b 0\n"
        ":flush\nexit /b 0\n"
    )
    (tmp_path / "stmp").mkdir(exist_ok=True)
    driver = tmp_path / "driver.bat"
    driver.write_bytes(script.replace("\n", "\r\n").encode("utf-8"))
    run_env = {**os.environ, **(env or {})}
    done = subprocess.run(
        ["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path, input=stdin,
        capture_output=True, text=True, timeout=60, env=run_env, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _menu(tmp_path: Path, rec: str, pick: str = "", rows=ROWS) -> str:
    body = f'set "REC={rec}"\n' + _between(MENU_START, MENU_END) + "\necho BACKEND=!BACKEND!\n"
    _stub_python(tmp_path, menu=rows)
    return _run(tmp_path, body, stdin=pick + "\n", env=_env(tmp_path))


def _listed(out: str) -> list[str]:
    return [line.strip().split("]")[0].lstrip("[") for line in out.splitlines()
            if line.strip().startswith("[") and "]" in line and "is not offered" not in line]


def test_every_row_the_spec_prints_is_listed_with_its_label(tmp_path):
    out = _menu(tmp_path, "cuda")
    assert _listed(out) == ["1", "2", "3", "4", "5"]
    for n, _key, label in ROWS:
        assert f"   [{n}] {label}" in out
    assert "Pick 1-5 [1]:" in out
    assert "BACKEND=cuda" in out


def test_the_recommendation_is_passed_to_the_spec(tmp_path):
    body = 'set "REC=vulkan"\n' + _between(MENU_START, MENU_END)
    calls = _stub_python(tmp_path, menu=ROWS)
    _run(tmp_path, body, env=_env(tmp_path))
    assert calls.read_text(encoding="utf-8").split() == ["backend-menu", "vulkan"]


@pytest.mark.parametrize("pick, backend", [("2", "vulkan"), ("3", "cuda"), ("4", "cpu"),
                                           ("5", "own"), ("1", "cuda")])
def test_a_listed_number_picks_its_backend(tmp_path, pick, backend):
    out = _menu(tmp_path, "cuda", pick=pick)
    assert f"BACKEND={backend}" in out
    assert "is not offered" not in out


@pytest.mark.parametrize("pick", ["6", "9", "12"])
def test_an_unlisted_number_returns_to_the_recommendation_with_a_message(tmp_path, pick):
    out = _menu(tmp_path, "cuda", pick=pick)
    assert f"[{pick}] is not offered on this machine; using cuda." in out
    assert "BACKEND=cuda" in out


def test_blank_and_junk_picks_take_the_recommendation_silently(tmp_path):
    assert "BACKEND=cuda" in _menu(tmp_path, "cuda")
    out = _menu(tmp_path, "cuda", pick="zz")
    assert "BACKEND=cuda" in out and "is not offered" not in out


def test_the_prompt_range_follows_the_number_of_rows(tmp_path):
    rows = [("1", "cpu", "cpu - no GPU   (recommended for your hardware)"),
            ("2", "cpu", "cpu - no GPU   (same as [1])"),
            ("3", "own", "I will build / provide my own (skip the download)")]
    out = _menu(tmp_path, "cpu", pick="3", rows=rows)
    assert "Pick 1-3 [1]:" in out
    assert "BACKEND=own" in out


def test_no_menu_from_the_spec_says_so_and_uses_the_recommendation(tmp_path):
    out = _menu(tmp_path, "vulkan", rows=None)
    assert "[!] Could not list the llama.cpp backends for this machine; using vulkan." in out
    assert "Pick " not in out
    assert "BACKEND=vulkan" in out


def test_an_empty_menu_file_counts_as_no_menu(tmp_path):
    out = _menu(tmp_path, "cpu", rows=[])
    assert "Could not list the llama.cpp backends" in out
    assert "BACKEND=cpu" in out


def test_the_real_spec_menu_parses(tmp_path):
    """The real interpreter and the real localm.setup_spec: every row it prints
    is listed under its own number, and picking a number gives its key."""
    from localm import setup_spec

    _venv_python(tmp_path)
    rows = setup_spec.backend_menu("vulkan")
    last = rows[-1]
    body = 'set "REC=vulkan"\n' + _between(MENU_START, MENU_END) + "\necho BACKEND=!BACKEND!\n"
    out = _run(tmp_path, body, stdin=last[0] + "\n", env={"PYTHONPATH": str(ROOT)})
    for n, _key, label in rows:
        assert f"   [{n}] {label}" in out, out
    assert f"Pick 1-{last[0]} [1]:" in out
    assert f"BACKEND={last[1]}" in out


def _hint(tmp_path: Path, hint: str | None) -> str:
    _stub_python(tmp_path, menu=None, hint=hint)
    return _run(tmp_path, _between(HINT_START, HINT_END) + "\necho SLB=!SLB!\n",
                env=_env(tmp_path))


def test_provision_later_hint_comes_from_the_spec(tmp_path):
    assert "SLB=vulkan|cuda|cpu" in _hint(tmp_path, "vulkan|cuda|cpu")


def test_provision_later_hint_without_an_answer_is_a_placeholder(tmp_path):
    assert "SLB=backend" in _hint(tmp_path, None)


def test_no_backend_list_is_hardcoded():
    code = [ln for ln in BAT.splitlines() if not ln.strip().lower().startswith("rem")]
    text = "\n".join(code)
    assert "localm.setup_spec backend-menu" in text
    assert "localm.setup_spec backend-hint" in text
    for gone in ("AVAILFALL", "hwdetect backends", ":backend_pickrange", ":backend_unoffered",
                 ":backend_hint", "amd-rocm sycl", "vulkan cuda"):
        assert gone not in text, gone


def test_the_menu_reads_the_same_spec_setup_sh_does():
    sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
    for call in ("localm.setup_spec backend-menu", "localm.setup_spec backend-hint"):
        assert call in BAT and call in sh
