# SPDX-License-Identifier: AGPL-3.0-or-later
"""The POSIX setup scripts and launchers use only what localm brought.

A static sentinel over setup.sh, setup-gui.sh, localm.sh, localm-launcher.sh,
rollback.sh and report-issue.sh:

- the isolation block comes first: every inherited ``UV_*`` / ``PIP_*`` variable,
  PYTHONPATH, PYTHONHOME, PYTHONSTARTUP, PYTHONUSERBASE, VIRTUAL_ENV, CONDA_PREFIX
  and LOCALM_HOME are cleared and UV_NO_CONFIG=1, PIP_CONFIG_FILE=/dev/null,
  PYTHONNOUSERSITE=1 are set before any program runs;
- the scripts that run system programs set PATH to the fixed system folders
  first, and probe those folders by full path (``sys_prog``), never ``command -v``
  or ``which``;
- no bare ``uv``, ``python``, ``python3``, ``py``, ``curl``, ``wget`` or ``git`` is
  run; uv is always ``"$UVEXE"`` (localm's own store) and a Python is a venv or
  localm's own managed one;
- no PATH entry for ``~/.local/bin`` or ``~/.cargo/bin``;
- every ``uv venv`` / ``uv run`` passes ``--python-preference only-managed``;
- no ``/tmp``: setup's temporary files live in ./.localm-setup-tmp.

A behavioural check runs each script's isolation block in bash with a poisoned
environment and reads what is left.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._posix_sysdirs import SYSDIRS_LINE

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ("setup.sh", "setup-gui.sh", "localm.sh", "localm-launcher.sh", "rollback.sh",
           "report-issue.sh")
FIXED_PATH = ("setup.sh", "setup-gui.sh", "rollback.sh", "report-issue.sh")
CLEARED = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV",
           "CONDA_PREFIX", "LOCALM_HOME")
SET_LINE = "export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1"
BANNED = ("uv", "python", "python3", "py", "curl", "wget", "git")

# What a command may be preceded by on a line: start, a separator, a keyword.
_CMD_START = (r"(?:^|[;&|(!{]|\$\(|\b(?:then|do|else|elif|if|while|until|exec|time|nice|env)\s)"
              r"\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s\"']*)\s+)*")
_BARE = re.compile(_CMD_START + r"(" + "|".join(BANNED) + r")(?=\s|$|;|\))")


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _code_lines(name: str) -> list[tuple[int, str]]:
    """(line number, code) for every line that is not a comment, a heredoc body
    or text a message prints; single-quoted strings are blanked."""
    out = []
    heredoc = None
    for i, raw in enumerate(_text(name).splitlines(), 1):
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        m = re.search(r"<<-?\s*'?([A-Z_]+)'?", raw)
        if m:
            heredoc = m.group(1)
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"(say|echo|printf)\b", line):
            continue
        line = re.sub(r"'[^']*'", "''", line)
        line = re.sub(r"\s#\s.*$", "", line)
        out.append((i, line))
    return out


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_bare_tool_is_run(name):
    hits = [(i, ln) for i, ln in _code_lines(name) if _BARE.search(ln)]
    assert hits == [], f"{name}: a program from PATH is run: {hits}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_path_lookup(name):
    hits = [(i, ln) for i, ln in _code_lines(name)
            if re.search(r"\bcommand\s+-v\b|\bwhich\b|\btype\s+-p\b|\bhash\s", ln)]
    assert hits == [], f"{name}: looks a program up on PATH: {hits}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_user_tool_folder_goes_on_path(name):
    hits = [(i, ln) for i, ln in enumerate(_text(name).splitlines(), 1)
            if re.search(r"PATH=.*(\.local/bin|\.cargo/bin)", ln)]
    assert hits == [], f"{name}: puts a user tool folder on PATH: {hits}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_system_temp_folder(name):
    hits = [(i, ln) for i, ln in enumerate(_text(name).splitlines(), 1)
            if not ln.lstrip().startswith("#")
            and re.search(r"(?<![\w$}])/tmp\b|\bmktemp\b", ln)]
    assert hits == [], f"{name}: uses the system temp folder: {hits}"


def _isolation_end(name: str) -> int:
    lines = _text(name).splitlines()
    return next(i for i, ln in enumerate(lines) if ln.strip() == SET_LINE)


_ALLOWED_BEFORE = re.compile(
    r"^(#.*|set -euo pipefail|LOCALM_SYSDIRS=\"[^\"]*\"|USER_PATH=\"\$\{PATH:-\}\""
    r"|export PATH=\"\$LOCALM_SYSDIRS\"|for _var in \$\(compgen -e\); do"
    r"|case \"\$_var\" in UV_\*\|PIP_\*\) unset \"\$_var\" ;; esac|done|unset [A-Z_ ]+)$")


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_isolation_block_comes_before_anything_runs(name):
    lines = _text(name).splitlines()
    end = _isolation_end(name)
    before = [ln.strip() for ln in lines[1:end] if ln.strip()]
    stray = [ln for ln in before if not _ALLOWED_BEFORE.match(ln)]
    assert stray == [], f"{name}: runs something before its isolation block: {stray}"
    unset = " ".join(ln for ln in before if ln.startswith("unset "))
    assert all(v in unset.split() for v in CLEARED), unset
    assert 'case "$_var" in UV_*|PIP_*) unset "$_var" ;; esac' in before
    if name in FIXED_PATH:
        assert SYSDIRS_LINE in before and 'export PATH="$LOCALM_SYSDIRS"' in before
        assert before.index('export PATH="$LOCALM_SYSDIRS"') > before.index(SYSDIRS_LINE)


@pytest.mark.parametrize("name", ("setup.sh", "setup-gui.sh"))
def test_every_uv_call_is_localm_own_uv_with_only_managed_python(name):
    text = _text(name)
    calls = [(i, ln) for i, ln in _code_lines(name)
             if re.search(r'(\buv|"\$UVEXE"|\$UVEXE)\s+(venv|run|pip)\b', ln)]
    assert len(calls) >= (5 if name == "setup.sh" else 1), f"{name}: the sentinel found no uv call"
    for i, ln in calls:
        assert re.search(r'"\$UVEXE" (venv|run|pip)\b', ln), (name, i, ln)
        assert not re.search(r'(?<!")\buv\s+(venv|run|pip)\b', ln), (name, i, ln)
        if re.search(r'"\$UVEXE"\s+(venv|run)\b', ln):
            assert "--python-preference only-managed" in ln or "$PYPREF" in ln, (name, i, ln)
    if name == "setup.sh":
        assert 'PYPREF="--python-preference only-managed"' in text
    assert "export UV_PYTHON_PREFERENCE=only-managed" in text
    assigned = re.findall(r'^\s*UVEXE="([^"]*)"', text, re.M)
    assert assigned and all(a == "$STORE_UV/uv" for a in assigned), assigned
    stores = re.findall(r'STORE_UV="([^"]*)"', text)
    assert sorted(set(stores)) == ["$ROOT/.uv", "$SHARED_STORE/uv"], stores


@pytest.mark.parametrize("name", ("setup.sh", "setup-gui.sh"))
def test_setup_temp_and_shared_store_are_fixed(name):
    text = _text(name)
    assert 'STMP="$ROOT/.localm-setup-tmp"' in text
    assert 'export TMPDIR="$STMP" TMP="$STMP" TEMP="$STMP"' in text
    assert text.index('export TMPDIR="$STMP"') < text.index("fetch_uv_installer \"$UVTMP")
    assert 'SHARED_STORE="$XDG_DATA_HOME/localm/shared"' in text
    assert 'SHARED_STORE="${HOME:+$HOME/.local/share/localm/shared}"' in text
    cred = 'export UV_CREDENTIALS_DIR="$STORE_UV/credentials"'
    assert text.count(cred) == 1
    first_use = min(text.index(s) for s in ('"$UVEXE" venv', '"$UVEXE" run') if s in text)
    assert text.index(cred) < first_use


@pytest.mark.parametrize("name", ("localm.sh", "localm-launcher.sh", "rollback.sh",
                                  "report-issue.sh"))
def test_launchers_run_only_localm_own_python(name):
    runs = [ln for _, ln in _code_lines(name) if re.search(r"\bexec\b", ln)]
    assert runs, name
    for ln in runs:
        assert re.search(r'exec (\.venv/bin/|"\$(RBPY|RIPY)")', ln), (name, ln)
    if name in ("rollback.sh", "report-issue.sh"):
        text = _text(name)
        assert "for c in .venv/bin/python3 .venv/bin/python .python/*/bin/python3 \\\n" in text
        assert '    "${SHARED_STORE:-/nonexistent}"/python/*/bin/python3; do\n' in text
        assert text.count("for c in ") == 1


def _bash() -> str | None:
    found = shutil.which("bash")
    if found is None:
        return None
    real = Path(found).parent.parent / "usr" / "bin" / Path(found).name
    return str(real) if real.is_file() else found


@pytest.mark.skipif(_bash() is None, reason="no bash on PATH")
@pytest.mark.parametrize("name", SCRIPTS)
def test_the_isolation_block_clears_a_poisoned_environment(tmp_path, name):
    lines = _text(name).splitlines()
    block = "\n".join(lines[1:_isolation_end(name) + 1]) + "\n"
    script = tmp_path / "probe.sh"
    script.write_text(block + 'env | sort\n', encoding="utf-8", newline="\n")
    poison = {"UV_INDEX_URL": "x", "UV_PYTHON_INSTALL_MIRROR": "x", "UV_PYTHON": "x",
              "UV_CACHE_DIR": "x", "UV_PYTHON_INSTALL_DIR": "x", "UV_NO_CONFIG": "0",
              "PIP_INDEX_URL": "x", "PIP_CONFIG_FILE": "x", "PIP_USER": "1",
              "PYTHONNOUSERSITE": "", **{v: "x" for v in CLEARED}}
    env = dict(os.environ, **poison)
    r = subprocess.run([_bash(), script.as_posix()], env=env, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0, r.stderr
    seen = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln)
    left = sorted(k for k in seen if k.startswith(("UV_", "PIP_")) or k in CLEARED)
    assert left == ["PIP_CONFIG_FILE", "UV_NO_CONFIG"], left
    assert seen["UV_NO_CONFIG"] == "1" and seen["PIP_CONFIG_FILE"] == "/dev/null"
    assert seen["PYTHONNOUSERSITE"] == "1"
    if name in FIXED_PATH:
        assert seen["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


@pytest.mark.parametrize("name", ("setup.sh", "setup-gui.sh", "install.sh"))
def test_curl_and_wget_come_only_from_the_fixed_system_folders(name):
    """A fresh macOS ships /usr/bin/curl and every distro ships curl or wget in
    /usr/bin, so the download helper looks nowhere else (no Homebrew or
    /usr/local folder)."""
    text = (ROOT / name).read_text(encoding="utf-8")
    start = text.index("download() {")
    body = text[start:text.index("\n}\n", start)]
    assert "/usr/local" not in body and "homebrew" not in body.lower(), body
    assert "uname" not in body, body
    assert re.search(r"\$LOCALM_SYSDIRS|sys_prog (curl|wget)", body), body
    assert text.count(SYSDIRS_LINE) == 1
