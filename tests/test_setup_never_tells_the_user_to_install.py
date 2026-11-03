# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup, the installers and the launchers never tell the user to install
anything.

localm brings what it needs. Where a step fails, setup retries it with its own
uv and says that running setup again retries it; it never prints a package
manager command, a pip or uv command line, or a pointer to a manual recipe.
The static check reads every line a user can see in these files; the setup.sh
PyTorch block is also run against a stub uv.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FILES = ("setup.bat", "setup.sh", "setup-gui.bat", "setup-gui.sh", "installer/gui.py",
         "install.sh", "localm.bat", "localm.sh", "localm-launcher.bat",
         "localm-launcher.sh", "rollback.bat", "rollback.sh", "report-issue.bat",
         "report-issue.sh", "launcher.pyw")

# Anywhere in a non-comment line.
ANYWHERE = [
    re.compile(r"\bwinget\b", re.I),
    re.compile(r"\byourself\b", re.I),
    re.compile(r"\bmanually\b", re.I),
    re.compile(r"python3-tk|python3-tkinter|\bapt(-get)? install|\bdnf install|"
               r"\bbrew install|\bpacman -S|\bzypper install", re.I),
    re.compile(r"docs/gpu-setup\.md", re.I),
    re.compile(r"\binstall (it|uv|git|python|torch|pytorch)\b[^\n]*\b(first|then re-?run)", re.I),
    re.compile(r"\b(is|are) required - install", re.I),
]
# Only in a line that prints something.
PRINTED = re.compile(r"^\s*(@?echo\b|say\b|printf\b|print\(|.*\bemit\()", re.I)
IN_PRINTED = [re.compile(r"\bpip install\b", re.I), re.compile(r"--reinstall\b")]


def _is_comment(name: str, line: str) -> bool:
    s = line.strip()
    if name.endswith(".bat"):
        return s.lower().startswith(("rem ", "rem\t", "::")) or s.lower() == "rem"
    return s.startswith("#")


def _findings(name: str) -> list:
    text = (ROOT / name).read_text(encoding="utf-8")
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        if _is_comment(name, line):
            continue
        hits = [p.pattern for p in ANYWHERE if p.search(line)]
        if PRINTED.search(line):
            hits += [p.pattern for p in IN_PRINTED if p.search(line)]
        if hits:
            found.append((i, line.strip(), hits))
    return found


@pytest.mark.parametrize("name", FILES)
def test_no_message_tells_the_user_to_install_something(name):
    assert not _findings(name), f"{name}: {_findings(name)}"


def test_the_check_catches_the_messages_it_exists_for(tmp_path, monkeypatch):
    bad = {
        "a.bat": 'echo  Install uv yourself (winget install astral-sh.uv), then run this again.\r\n'
                 'echo  [^^!] torch install failed. ^(see docs/gpu-setup.md^)\r\n',
        "b.sh": 'say "  Fix it with:  \\"$UVEXE\\" pip install -p .venv -e . --reinstall"\n'
                'say "  (e.g. a system Python without it), install your distro\'s python3-tk."\n'
                'command -v git || { echo "git is required - install it first."; exit 1; }\n',
    }
    for n, t in bad.items():
        (tmp_path / n).write_text(t, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert len(_findings("a.bat")) == 2
    assert len(_findings("b.sh")) == 3


# --------------------------------------------------------------------------- #
#  setup.sh's PyTorch step retries with its own uv                             #
# --------------------------------------------------------------------------- #

def _bash():
    found = shutil.which("bash")
    return None if not found or "system32" in found.lower() else found


def _torch_block() -> str:
    src = (ROOT / "setup.sh").read_text(encoding="utf-8")
    start = src.index('if [ -n "$TORCHSPEC" ]; then')
    end = src.index("\nstep_done torch\n", start)
    return src[start:end] + "\n"


def _run_torch_block(tmp_path: Path, torchspec: str, rcs: list) -> tuple:
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'N=$(cat .uv-calls 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > .uv-calls\n'
        'echo "$*" >> .uv-args\n'
        'eval "RC=\\${STUB_RC_$N:-0}"\n'
        'exit "$RC"\n', encoding="utf-8", newline="\n")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    from tests._posix_sysdirs import bash_path
    script = ("set -euo pipefail\n"
              'say() { printf "%s\\n" "$*"; }\n'
              f'UVEXE="{bash_path(uv)}"\nTORCHSPEC="{torchspec}"\nBACKEND=cuda\n'
              + _torch_block() + 'echo "BLOCK-DONE"\n')
    env = dict(os.environ)
    for n, rc in enumerate(rcs, 1):
        env[f"STUB_RC_{n}"] = str(rc)
    r = subprocess.run([_bash(), "-c", script], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=60)
    args = (tmp_path / ".uv-args")
    return r, args.read_text().splitlines() if args.exists() else []


needs_bash = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")


@needs_bash
def test_a_failed_torch_install_is_retried_once_and_then_succeeds_quietly(tmp_path):
    r, calls = _run_torch_block(tmp_path, "torch torchvision", [1, 0, 0])
    assert "BLOCK-DONE" in r.stdout, r.stdout + r.stderr
    assert [c.startswith("pip install -p .venv torch") for c in calls] == [True, True, False]
    assert "[!]" not in r.stdout, r.stdout


@needs_bash
def test_torch_failing_twice_is_reported_without_a_manual_route(tmp_path):
    r, calls = _run_torch_block(tmp_path, "torch torchvision", [1, 1, 1, 1])
    assert "BLOCK-DONE" in r.stdout, r.stdout + r.stderr
    assert len(calls) == 4, calls
    assert "PyTorch could not be installed" in r.stdout
    assert "transformers could not be installed" in r.stdout
    assert "running setup again retries it" in r.stdout
    assert "pip install" not in r.stdout and "gpu-setup" not in r.stdout, r.stdout


@needs_bash
def test_a_runtime_without_torch_prints_no_install_commands(tmp_path):
    r, calls = _run_torch_block(tmp_path, "", [])
    assert "BLOCK-DONE" in r.stdout, r.stdout + r.stderr
    assert calls == []
    assert "Skipping the PyTorch/transformers stack" in r.stdout
    assert "pip install" not in r.stdout and "gpu-setup" not in r.stdout, r.stdout
