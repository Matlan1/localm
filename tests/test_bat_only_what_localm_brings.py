# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Windows setup scripts and launchers use only what localm brought.

A static read of setup.bat, setup-gui.bat, localm.bat, localm-launcher.bat,
rollback.bat and report-issue.bat:

- no program is started by a bare name that Windows would look up on PATH (uv,
  python, py, powershell, where, attrib, taskkill, icacls, findstr, git, curl,
  timeout, ping); system programs run by their full path under %SystemRoot%;
- no per-user uv folder (``.local\\bin``, ``.cargo\\bin``) appears;
- every ``uv venv`` and ``uv run`` asks for uv's managed Python only;
- inherited uv, pip and Python settings are cleared before the first program
  starts, and the setups point TEMP and TMP into the install folder before it;
- the uv every setup uses is localm's own pinned copy in its store, installed
  without touching the user PATH.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUPS = ("setup.bat", "setup-gui.bat")
HELPERS = ("localm.bat", "localm-launcher.bat", "rollback.bat", "report-issue.bat")
ALL = SETUPS + HELPERS

TOOLS = ("uv", "uvx", "python", "pythonw", "python3", "py", "powershell", "pwsh", "where",
         "attrib", "taskkill", "icacls", "findstr", "git", "curl", "timeout", "ping", "tar")
_BARE = re.compile(
    r"(?<![\w\\/.\-:%!$\[])(?<!==\")(?<!== \")(" + "|".join(TOOLS) + r")(\.exe)?(?![\w\-.\\/:\]=])",
    re.I)


def _lines(name: str) -> list[str]:
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _code(line: str) -> str:
    """The part of a batch line that can start a program: comments, echo text,
    ``set /p`` prompts, ``title`` text and the quoted arguments of ``call :label``
    removed."""
    s = line.strip()
    low = s.lower()
    if not s or low.startswith(("rem ", "rem\t", "::", ":")) or low == "rem":
        return ""
    if low.startswith("set /p"):
        return ""
    if re.match(r"(.*\s)?call\s+:\w+", s, re.I):
        s = re.sub(r'"[^"]*"', '""', s)
    s = re.sub(r"\btitle\b[^&|]*", "", s, flags=re.I)

    def no_echo(m: re.Match) -> str:
        rest = m.group(1)
        cut = re.search(r"(?<!\^)[&|]", rest)
        return "echo" + (rest[cut.start():] if cut else "")
    return re.sub(r"\becho\b(.*)$", no_echo, s, flags=re.I)


def bare_invocations(name: str) -> list[tuple[int, str]]:
    """(line number, line) of every line in *name* that names a program by a
    bare name."""
    hits = []
    for i, line in enumerate(_lines(name), 1):
        if _BARE.search(_code(line)):
            hits.append((i, line.strip()))
    return hits


@pytest.mark.parametrize("name", ALL)
def test_no_program_is_started_by_a_bare_name(name):
    assert bare_invocations(name) == []


def test_the_bare_name_check_sees_what_it_must():
    for line in ('uv venv .venv', 'where uv >nul 2>nul', 'powershell -NoProfile -Command "x"',
                 'for /f "delims=" %%T in (\'powershell -Command "[char]9"\') do set "T=%%T"',
                 '"%PSEXE%" -Command "... & powershell -NoProfile -File $f"',
                 'if not errorlevel 1 call :try_python py -3', 'call python -m x',
                 'echo %* | findstr /I /C:"--debug" >nul', 'attrib -R "x" /S',
                 '    uv pip install -p .venv x || echo  [!] failed', 'start "" pythonw "launcher.pyw"',
                 'for %%P in (py.exe python.exe) do where %%P'):
        assert _BARE.search(_code(line)), line
    for line in ('"!UVEXE!" venv --python 3.12', '.venv\\Scripts\\python -m localm x',
                 '"%SYS32%\\findstr.exe" /i "x" y', 'echo  Installing uv 0.13.6 ...',
                 'set "UVSTEP=uv-bootstrap"', 'if /i "%%a"=="uv" set "V=%%b"',
                 'rem uv is localm\'s own', 'call :offer_report "could not install uv" "x"',
                 'set /p "X=  Install uv? "', "'https://x/'+$v+'/uv-installer.ps1'",
                 'set "PSEXE=%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"'):
        assert not _BARE.search(_code(line)), line


@pytest.mark.parametrize("name", ALL)
def test_no_per_user_uv_folder(name):
    text = (ROOT / name).read_text(encoding="utf-8").lower().replace("/", "\\")
    assert ".local\\bin" not in text and ".cargo\\bin" not in text


def _uv_calls(name: str) -> list[str]:
    return [ln for ln in _lines(name) if re.search(r'"[%!]UVEXE[%!]"\s+(venv|run)\b', ln)]


@pytest.mark.parametrize("name", SETUPS)
def test_every_venv_and_run_uses_only_managed_python(name):
    calls = _uv_calls(name)
    assert calls, f"{name} runs no uv venv / uv run"
    text = "\n".join(_lines(name))
    for call in calls:
        if "%PYPREF%" in call:
            assert re.search(r'^set "PYPREF=--python-preference only-managed"$', text, re.M)
        else:
            assert "--python-preference only-managed" in call, call
    assert re.search(r'^set "UV_PYTHON_PREFERENCE=only-managed"$', text, re.M)


def _first_program_line(lines: list[str]) -> int:
    pat = re.compile(r'%PSEXE%|[%!]UVEXE[%!]"|\.venv\\Scripts\\|[%!](RBPY|RIPY|PYBIN|PY)[%!]|'
                     r'powershell\.exe"|\bstart\s+""|\bcall\s+"', re.I)
    for i, line in enumerate(lines):
        code = _code(line)
        if not code or re.match(r'\s*set\s+"\w+=', code, re.I) or code.startswith("if exist"):
            continue
        if pat.search(code):
            return i
    raise AssertionError("no program is started")


def _index(lines: list[str], pattern: str) -> int:
    hits = [i for i, ln in enumerate(lines) if re.search(pattern, ln)]
    assert hits, pattern
    return hits[0]


_CLEARS = (r"""^for /f "delims==" %%v in \('set UV_ 2\^>nul'\) do set "%%v="$""",
           r"""^for /f "delims==" %%v in \('set PIP_ 2\^>nul'\) do set "%%v="$""",
           r"^for %%v in \(PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV "
           r"CONDA_PREFIX[ )]",
           r'^set "UV_NO_CONFIG=1"$', r'^set "PIP_CONFIG_FILE=NUL"$', r'^set "PYTHONNOUSERSITE=1"$',
           r'^set "ComSpec=[%!]SystemRoot[%!]\\System32\\cmd\.exe"$')


@pytest.mark.parametrize("name", ALL)
def test_inherited_settings_are_cleared_before_the_first_program(name):
    lines = _lines(name)
    first = _first_program_line(lines)
    for pattern in _CLEARS:
        assert _index(lines, pattern) < first, (pattern, lines[first])


@pytest.mark.parametrize("name", HELPERS)
def test_a_helper_clears_an_inherited_localm_home(name):
    lines = _lines(name)
    clear = _index(lines, r"^for %%v in \(PYTHONPATH .*\bLOCALM_HOME\)")
    assert clear < _first_program_line(lines)
    assert _index(lines, r"^setlocal\b") < clear


@pytest.mark.parametrize("name", SETUPS)
def test_setup_temp_is_the_install_folder_before_uv_runs(name):
    lines = _lines(name)
    first_uv = _index(lines, r'[%!]UVEXE[%!]"|%PSEXE%" -NoProfile -ExecutionPolicy Bypass -Command "\$Progress')
    for var in ("TEMP", "TMP"):
        assert _index(lines, rf'^set "{var}=[%!]CD[%!]\\\.localm-setup-tmp"$') < first_uv, var
    text = "\n".join(lines)
    assert "GetTempPath" not in text


@pytest.mark.parametrize("name", ALL)
def test_no_system_temp_fallback(name):
    assert "GetTempPath" not in (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", SETUPS)
def test_uv_is_localm_own_pinned_copy_in_its_store(name):
    text = "\n".join(_lines(name))
    assert re.search(r'^set "UVEXE=[%!]UVDIR[%!]\\uv\.exe"$', text, re.M)
    assert re.search(r'^set "UV_INSTALL_DIR=[%!]UVDIR[%!]"$', text, re.M)
    assert re.search(r'^set "UV_UNMANAGED_INSTALL=[%!]UVDIR[%!]"$', text, re.M)
    assert re.search(r'^\s*set "UVDIR=[%!]LOCALAPPDATA[%!]\\localm\\shared\\uv"$', text, re.M)
    assert re.search(r'^\s*set "UVDIR=[%!]CD[%!]\\\.uv"$', text, re.M)
    assert re.search(r'^if "[%!]UVVER[%!]"=="%UV_INSTALLER_VERSION%" set "UVOK=1"$', text, re.M)
    assert "uv-install" not in re.findall(r"SETUP-Q:\s*([a-z0-9-]+)", text)


@pytest.mark.parametrize("name", SETUPS)
def test_uv_credentials_store_is_in_the_store_before_uv_runs(name):
    lines = _lines(name)
    pin = _index(lines, r'^set "UV_CREDENTIALS_DIR=[%!]UVDIR[%!]\\credentials"$')
    assert pin < _index(lines, r'call :uv_version|"[%!]UVEXE[%!]" ')


_POISON = {"UV_PYTHON_INSTALL_MIRROR": "http://127.0.0.1:9/mirror", "UV_INDEX_URL": "http://127.0.0.1:9/i",
           "UV_EXTRA_INDEX_URL": "http://127.0.0.1:9/e", "UV_PYTHON_PREFERENCE": "only-system",
           "UV_CACHE_DIR": "poisoned", "UV_CONFIG_FILE": "poisoned.toml",
           "PIP_INDEX_URL": "http://127.0.0.1:9/p", "PIP_CONFIG_FILE": "poisoned.ini",
           "PYTHONPATH": "poisoned", "PYTHONHOME": "poisoned", "PYTHONSTARTUP": "poisoned.py",
           "PYTHONUSERBASE": "poisoned", "VIRTUAL_ENV": "poisoned", "CONDA_PREFIX": "poisoned",
           "LOCALM_HOME": "poisoned", "ComSpec": "poisoned.exe"}


def _isolation_block(name: str) -> str:
    """The script from its first line through its isolation settings."""
    lines = _lines(name)
    end = _index(lines, r'^set "PYTHONNOUSERSITE=1"$')
    if any(re.match(r'^set "UV_PYTHON_PREFERENCE=only-managed"$', ln) for ln in lines):
        end = max(end, _index(lines, r'^set "UV_PYTHON_PREFERENCE=only-managed"$'))
    return "\r\n".join(lines[:end + 1]) + "\r\n"


@pytest.mark.skipif(__import__("os").name != "nt", reason="runs cmd.exe")
@pytest.mark.parametrize("name", ALL)
def test_the_isolation_block_really_clears_what_it_inherits(name, tmp_path):
    import os
    import subprocess
    probe = tmp_path / "probe.bat"
    probe.write_text(_isolation_block(name) + "set\r\nexit /b 0\r\n", encoding="utf-8")
    env = {**os.environ, **_POISON}
    cmd = os.path.join(os.environ["SystemRoot"], "System32", "cmd.exe")
    r = subprocess.run([cmd, "/d", "/c", str(probe)], cwd=str(tmp_path), env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    seen = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln)
    seen = {k.upper(): v for k, v in seen.items()}
    assert "poisoned" not in r.stdout and "127.0.0.1:9" not in r.stdout, r.stdout
    assert seen.get("UV_NO_CONFIG") == "1" and seen.get("PIP_CONFIG_FILE") == "NUL"
    assert seen.get("PYTHONNOUSERSITE") == "1"
    assert seen.get("COMSPEC", "").lower() == os.path.join(
        os.environ["SystemRoot"], "System32", "cmd.exe").lower()
    if name in SETUPS:
        assert seen.get("UV_PYTHON_PREFERENCE") == "only-managed"


@pytest.fixture(scope="module")
def decoy_python(tmp_path_factory):
    """A folder holding python.exe, pythonw.exe and py.exe stand-ins that log
    every start to STUB_LOG."""
    from tests import _fake_uv as fake_uv
    d = tmp_path_factory.mktemp("decoy")
    built = fake_uv.build(tmp_path_factory.mktemp("decoybuild"))
    for name in ("python.exe", "pythonw.exe", "py.exe", "python3.exe"):
        (d / name).write_bytes(built.read_bytes())
    return d


@pytest.mark.skipif(__import__("os").name != "nt", reason="runs cmd.exe")
@pytest.mark.parametrize("name,said", [
    ("localm.bat", "LocaLM is not set up in this folder yet"),
    ("localm-launcher.bat", "LocaLM is not set up in this folder yet"),
    ("rollback.bat", "LocaLM's own Python is not in this folder"),
])
def test_a_helper_without_localm_python_says_so_and_never_runs_a_path_python(
        name, said, tmp_path, decoy_python):
    import os
    import shutil
    import subprocess
    folder = tmp_path / "install"
    folder.mkdir()
    shutil.copy(ROOT / name, folder / name)
    log = tmp_path / "decoy.log"
    system32 = os.path.join(os.environ["SystemRoot"], "System32")
    env = {**os.environ, "PATH": f"{decoy_python};{system32}", "STUB_LOG": str(log),
           "LOCALAPPDATA": str(tmp_path / "lad")}
    r = subprocess.run([os.path.join(system32, "cmd.exe"), "/d", "/c", str(folder / name), "doctor"],
                       cwd=str(tmp_path), env=env, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert said in r.stdout, r.stdout + r.stderr
    assert r.returncode == 1, r.stdout + r.stderr
    assert not log.exists(), log.read_text(encoding="utf-8")
