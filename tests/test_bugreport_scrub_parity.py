# SPDX-License-Identifier: AGPL-3.0-or-later
"""The three bug-report scrubbers agree: localm/bugreport/scrub.py
``_scrub_secrets`` (in-app), scripts/report_issue.py ``scrub`` and
scripts/report_issue.ps1 ``Scrub`` (the standalone reporters, which file public
issues when localm itself cannot start).

One corpus runs through all three; every output must equal the in-app output,
and the in-app output must carry none of the corpus's account names or secrets.
The Python copies of each pattern must also be byte-identical to the in-app
ones.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from localm import pathscrub
from localm.bugreport import scrub as br_scrub

_REPO = Path(__file__).resolve().parents[1]
_RI_PATH = _REPO / "scripts" / "report_issue.py"
_PS1_PATH = _REPO / "scripts" / "report_issue.ps1"

_spec = importlib.util.spec_from_file_location("report_issue_parity", _RI_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

# An account name that is never the test machine's own, so no implementation's
# Path.home() replacement fires and all three go through the same rules.
USER = "zqalice"
SECRET = "PARITYCANARY7Q4M"

_CORPUS = [
    # Doubled-backslash home paths: repr() / json.dumps() of a Windows path.
    repr("C:\\Users\\" + USER + "\\localm\\x.gguf"),
    "C:\\\\Users\\\\" + USER + "\\\\localm",
    '{"model_path": "C:\\\\Users\\\\' + USER + '\\\\m.gguf"}',
    "FileNotFoundError: [Errno 2] No such file or directory: 'C:\\\\Users\\\\"
    + USER + "\\\\localm\\\\home\\\\config.json'",
    # Single-backslash, forward-slash and POSIX forms.
    "C:\\Users\\" + USER + "\\localm\\x.gguf",
    "C:/Users/" + USER + "/models",
    "/home/" + USER + "/models and /Users/" + USER + "/Library",
    # JSON-key credentials.
    '{"api_key": "' + SECRET + '"}',
    '{"password":"' + SECRET + '", "n": 1}',
    "{'access_token': '" + SECRET + "'}",
    '{"OPENAI_API_KEY": "' + SECRET + '"}',
    '{"x_auth": "' + SECRET + '"}',
    # JSON flags and ordinary keys that must survive.
    '{"has_token": false, "require_auth": true, "key": "gpt-4"}',
    # Query parameters, header lines, bearer and API-key shapes.
    "https://searx.example.org/?q=a&api_key=" + SECRET + "&x=1",
    "X-Api-Key: " + SECRET,
    "Authorization: Basic " + SECRET,
    "curl -H 'Bearer " + SECRET + "'",
    "python main.py --api-key sk-proj-" + SECRET,
    "localm-sk-" + SECRET + " in a log",
    "OPENAI_API_KEY=" + SECRET + " require_auth=1",
    # URL credentials and email.
    "https://u:" + SECRET + "@comfy.example.org/api",
    "contact bob.builder@example.com about it",
    # Everything at once, in one line.
    repr("C:\\Users\\" + USER + "\\a") + ' {"token": "' + SECRET
    + '"} https://u:p@h.example.org/?sig=' + SECRET + " bob@example.com",
]

_IDS = [f"case{i}" for i in range(len(_CORPUS))]


@pytest.mark.parametrize("text", _CORPUS, ids=_IDS)
def test_in_app_scrub_removes_every_name_and_secret(text):
    out = br_scrub._scrub_secrets(text)
    assert USER not in out.lower()
    assert SECRET not in out
    assert "bob.builder" not in out


@pytest.mark.parametrize("text", _CORPUS, ids=_IDS)
def test_python_standalone_reporter_matches_in_app(text):
    assert ri.scrub(text) == br_scrub._scrub_secrets(text)


def test_flags_and_ordinary_keys_survive_every_scrubber():
    text = '{"has_token": false, "require_auth": true, "key": "gpt-4"}'
    assert br_scrub._scrub_secrets(text) == text
    assert ri.scrub(text) == text


def test_doubled_backslash_path_keeps_its_structure():
    text = "C:\\\\Users\\\\" + USER + "\\\\localm"
    assert ri.scrub(text) == "C:\\\\Users\\\\<redacted>\\\\localm"


def test_python_standalone_patterns_are_byte_identical_to_in_app():
    assert ri._USER_ROOT_PATTERN == pathscrub._USER_ROOT_PATTERN
    for name in ("_QUERY_SECRET_RE", "_HEADER_SECRET_RE", "_JSON_SECRET_RE",
                 "_BEARER_RE", "_APIKEY_RE", "_EMAIL_RE"):
        assert getattr(ri, name).pattern == getattr(br_scrub, name).pattern, name


_PS_DRIVER = (
    "param([string]$Target, [string]$InFile)\n"
    "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
    "$Target, [ref]$null, [ref]$null)\n"
    "$fn = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
    "$n.Name -eq 'Scrub' }, $true)\n"
    "$asg = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.AssignmentStatementAst] -and "
    "$n.Left.Extent.Text -eq '$email' }, $false)\n"
    "if ($fn.Count -ne 1 -or $asg.Count -ne 1) { Write-Output 'SCRUB-NOT-FOUND'; exit 1 }\n"
    "Invoke-Expression $asg[0].Extent.Text\n"
    "Invoke-Expression $fn[0].Extent.Text\n"
    "$text = [System.IO.File]::ReadAllText($InFile, [System.Text.Encoding]::UTF8)\n"
    "$out = New-Object System.Collections.Generic.List[string]\n"
    "foreach ($line in ($text -split \"`n\")) { $out.Add((Scrub $line)) }\n"
    "[System.IO.File]::WriteAllText($InFile + '.out', ($out -join \"`n\"), "
    "(New-Object System.Text.UTF8Encoding($false)))\n"
)


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_standalone_reporter_matches_in_app(tmp_path):
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    driver = tmp_path / "drive_scrub.ps1"
    driver.write_text(_PS_DRIVER, encoding="utf-8")
    infile = tmp_path / "in.txt"
    infile.write_text("\n".join(_CORPUS), encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver), "-Target", str(_PS1_PATH), "-InFile", str(infile)],
        capture_output=True, text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "SCRUB-NOT-FOUND" not in out, out
    assert proc.returncode == 0, f"driver failed ({proc.returncode}): {out}"
    got = (tmp_path / "in.txt.out").read_text(encoding="utf-8").split("\n")
    want = [br_scrub._scrub_secrets(t) for t in _CORPUS]
    assert len(got) == len(want), got
    mismatches = [(t, g, w) for t, g, w in zip(_CORPUS, got, want, strict=True) if g != w]
    assert not mismatches, mismatches


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_standalone_reporter_replaces_the_home_folder(tmp_path):
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    home = str(Path.home())
    sep = "\\"
    lines = [home + sep + "localm" + sep + "x.gguf", repr(home + sep + "localm"),
             home.replace(sep, "/") + "/models"]
    driver = tmp_path / "drive_scrub.ps1"
    driver.write_text(_PS_DRIVER, encoding="utf-8")
    infile = tmp_path / "in.txt"
    infile.write_text("\n".join(lines), encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver), "-Target", str(_PS1_PATH), "-InFile", str(infile)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = (tmp_path / "in.txt.out").read_text(encoding="utf-8").split("\n")
    assert got == [br_scrub._scrub_secrets(t) for t in lines]
    assert all(t.startswith(("~", "'~")) for t in got), got


_SH_PATH = _REPO / "report-issue.sh"


def _sh_scrub_function() -> str:
    text = _SH_PATH.read_text(encoding="utf-8")
    start = text.index("\nscrub() {\n") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _bash_with_perl():
    bash = shutil.which("bash")
    if not bash or "system32" in bash.lower() or not shutil.which("perl"):
        pytest.skip("no bash with perl on PATH")
    return bash


def _run_sh_scrub(tmp_path, lines, home):
    bash = _bash_with_perl()
    driver = tmp_path / "drive_sh_scrub.sh"
    driver.write_text(_sh_scrub_function() + 'scrub < "$1" > "$2"\n',
                      encoding="utf-8", newline="\n")
    infile = tmp_path / "sh_in.txt"
    infile.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    outfile = tmp_path / "sh_out.txt"
    env = dict(os.environ, HOME=home, RI_EMAIL=ri.MAINTAINER_EMAIL)
    proc = subprocess.run([bash, driver.as_posix(), infile.as_posix(), outfile.as_posix()],
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    return outfile.read_text(encoding="utf-8").split("\n")[:-1]


def test_shell_standalone_reporter_matches_in_app(tmp_path):
    got = _run_sh_scrub(tmp_path, _CORPUS, (tmp_path / "nohome").as_posix())
    want = [br_scrub._scrub_secrets(t) for t in _CORPUS]
    assert len(got) == len(want), got
    mismatches = [(t, g, w) for t, g, w in zip(_CORPUS, got, want, strict=True) if g != w]
    assert not mismatches, mismatches


def test_shell_standalone_reporter_replaces_its_home(tmp_path):
    home = "/home/" + USER
    got = _run_sh_scrub(tmp_path, [home + "/localm/x.gguf", "at " + home], home)
    assert got == ["~/localm/x.gguf", "at ~"]


def _run_curl_fallback(tmp_path, bash, *, drop_terminal_check):
    text = _SH_PATH.read_text(encoding="utf-8")
    fallback_text = text[text.index("# --- No Python at all"):]
    if drop_terminal_check:
        assert fallback_text.count("if [ ! -t 0 ]; then") == 1
        fallback_text = fallback_text.replace("if [ ! -t 0 ]; then", "if false; then")
    work = tmp_path / ("ungated" if drop_terminal_check else "gated")
    fake = work / "bin"
    fake.mkdir(parents=True)
    (work / "fallback.sh").write_text(fallback_text, encoding="utf-8", newline="\n")
    sink = work / "curl_called"
    (fake / "curl").write_text('#!/bin/sh\ntouch "' + sink.as_posix() + '"\necho 201\n',
                               encoding="utf-8", newline="\n")
    (fake / "curl").chmod(0o755)
    env = dict(os.environ, LOCALM_BUGREPORT_URL="http://127.0.0.1:9/report")
    proc = subprocess.run(
        [bash, "-c",
         'p="$(cygpath -u "$1" 2>/dev/null || printf %s "$1")"; '
         'PATH="$p:$PATH"; cd "$2" && bash fallback.sh',
         "_", fake.as_posix(), work.as_posix()],
        input="it broke\ny\n", capture_output=True, text=True, timeout=120, env=env)
    return proc, sink


def test_shell_reporter_sends_nothing_without_a_terminal(tmp_path):
    bash = _bash_with_perl()
    control, control_sink = _run_curl_fallback(tmp_path, bash, drop_terminal_check=True)
    assert control_sink.exists(), (
        "control: without the terminal check the fake curl must be reached\n"
        + control.stdout + control.stderr)
    proc, sink = _run_curl_fallback(tmp_path, bash, drop_terminal_check=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "nothing was sent" in out, out
    assert not sink.exists()
