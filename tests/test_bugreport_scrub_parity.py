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
