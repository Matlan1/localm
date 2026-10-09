# SPDX-License-Identifier: AGPL-3.0-or-later
"""Email redaction in the bug-report scrubbers: localm/bugreport/scrub.py
``_scrub_secrets``, scripts/report_issue.py ``scrub`` and scripts/report_issue.ps1
``Scrub``. Every token that contains an address becomes ``<redacted-email>``,
except the maintainer's address.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from localm.bugreport import _common, diagnostics
from localm.bugreport import scrub as br_scrub

_REPO = Path(__file__).resolve().parents[1]
_RI_PATH = _REPO / "scripts" / "report_issue.py"
_PS1_PATH = _REPO / "scripts" / "report_issue.ps1"

_spec = importlib.util.spec_from_file_location("report_issue_email", _RI_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

MAINTAINER = _common.MAINTAINER_EMAIL
OTHER = "bob.builder@example.com"
E_ACUTE, U_UML, A_UML = chr(0xE9), chr(0xFC), chr(0xE4)

_SCRUBBERS = [
    pytest.param(br_scrub._scrub_secrets, id="bugreport"),
    pytest.param(ri.scrub, id="report_issue_py"),
]

# (input, expected output) pairs every scrubber must produce exactly.
_CASES = [
    (f"reviewer {OTHER}, cc Jane.Doe+tag@mail.example.co.uk; "
     "<carol_99@sub-domain.example.org> mailto:dave@example.io",
     "reviewer <redacted-email>, cc <redacted-email>; "
     "<<redacted-email>> mailto:<redacted-email>"),
    ("bob@example.com-alice@evil.org", "<redacted-email>"),
    ("bob@example.com_alice@evil.org+carol@x.io", "<redacted-email>"),
    (f"{MAINTAINER}-bob@evil.org", "<redacted-email>"),
    (f"jos{E_ACUTE}@example.com", "<redacted-email>"),
    (f"bob@mail.b{U_UML}cher.de", "<redacted-email>"),
    (f"{A_UML}{MAINTAINER}", "<redacted-email>"),
    ("x bob%40example.com y", "x <redacted-email> y"),
    ("https://h.example.com/cb?email=bob%40example.com&x=1",
     "https://h.example.com/cb?email=<redacted-email>&x=1"),
    ('{"email":"bob@example.com","n":1}', '{"email":"<redacted-email>","n":1}'),
    ("ask bob@example.com.", "ask <redacted-email>."),
    ("bob@x.xn--p1ai", "<redacted-email>"),
    ("sean.o'brien@example.com", "<redacted-email>"),
    ("bob!x@example.com ~a@x.io {b}c@x.io", "<redacted-email> <redacted-email> <redacted-email>"),
    ("'bob@example.com', 'alice@x.org'.", "'<redacted-email>', '<redacted-email>'."),
    ("**bob@example.com** `carol@x.io`", "**<redacted-email>** `<redacted-email>`"),
    ("bob@my_host.example.com", "<redacted-email>"),
    ("http://bob@host.example.com:8188/ and https://u:pw@h.example.org/x",
     "http://<redacted>@host.example.com:8188/ and https://<redacted>@h.example.org/x"),
    ("user@host a@b x@localhost npm @scope/pkg@1.2.3 v1@2.0",
     "user@host a@b x@localhost npm @scope/pkg@1.2.3 v1@2.0"),
]


@pytest.mark.parametrize("scrub", _SCRUBBERS)
@pytest.mark.parametrize("text,expected", _CASES)
def test_scrub_output(scrub, text, expected):
    assert scrub(text) == expected


@pytest.mark.parametrize("scrub", _SCRUBBERS)
@pytest.mark.parametrize("text,expected", _CASES)
def test_scrub_is_idempotent(scrub, text, expected):
    once = scrub(text)
    assert scrub(once) == once


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_maintainer_address_is_kept_as_written(scrub):
    upper = MAINTAINER.upper()
    encoded = MAINTAINER.replace("@", "%40")
    text = f"write to {MAINTAINER}. or {upper}, or {encoded} or '{MAINTAINER}'."
    assert scrub(text) == text


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_maintainer_lookalikes_are_redacted(scrub):
    local, domain = MAINTAINER.split("@")
    lookalikes = [f"x{MAINTAINER}", f"{MAINTAINER}.evil.example",
                  f"{local}@{domain.split('.')[0]}.example", f"{MAINTAINER}-x"]
    out = scrub(" ".join(lookalikes))
    assert out == " ".join(["<redacted-email>"] * len(lookalikes))


def test_config_subset_redacts_emails(monkeypatch):
    import localm.config as config
    monkeypatch.setattr(config, "load_config", lambda: {
        "net_search_url": f"https://searx.example.org/search?contact={OTHER}",
        "comfy_launch_cmd": f"run.bat --notify {OTHER}",
        "coder_reviewer": OTHER,
        "cors_origins": [OTHER, "http://localhost:3000"],
    })
    out = diagnostics._safe_config_subset()
    assert out == {
        "net_search_url": "https://searx.example.org/search?contact=<redacted-email>",
        "comfy_launch_cmd": "run.bat --notify <redacted-email>",
        "coder_reviewer": "<redacted-email>",
        "cors_origins": ["<redacted-email>", "http://localhost:3000"],
    }


def test_all_three_scrubbers_share_one_pattern_and_address():
    assert ri._EMAIL_RE.pattern == br_scrub._EMAIL_RE.pattern
    assert br_scrub._EMAIL_RE.pattern.isascii()
    assert ri.MAINTAINER_EMAIL == MAINTAINER
    ps1 = _PS1_PATH.read_text(encoding="utf-8")
    assert f"'{br_scrub._EMAIL_RE.pattern}'" in ps1
    assert f"$email = '{MAINTAINER}'" in ps1


_PS_DRIVER = (
    "param([string]$Target, [string]$InFile, [switch]$NoEmail)\n"
    "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
    "$Target, [ref]$null, [ref]$null)\n"
    "$fn = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
    "$n.Name -eq 'Scrub' }, $true)\n"
    "$asg = $ast.FindAll({ param($n) $n -is "
    "[System.Management.Automation.Language.AssignmentStatementAst] -and "
    "$n.Left.Extent.Text -eq '$email' }, $false)\n"
    "if ($fn.Count -ne 1 -or $asg.Count -ne 1) { Write-Output 'SCRUB-NOT-FOUND'; exit 1 }\n"
    "if (-not $NoEmail) { Invoke-Expression $asg[0].Extent.Text }\n"
    "Invoke-Expression $fn[0].Extent.Text\n"
    "$text = [System.IO.File]::ReadAllText($InFile, [System.Text.Encoding]::UTF8)\n"
    "$out = New-Object System.Collections.Generic.List[string]\n"
    "foreach ($line in ($text -split \"`n\")) {\n"
    "  $once = Scrub $line\n"
    "  $out.Add($once); $out.Add((Scrub $once))\n"
    "}\n"
    "[System.IO.File]::WriteAllText($InFile + '.out', ($out -join \"`n\"), "
    "(New-Object System.Text.UTF8Encoding($false)))\n"
)


def _run_ps_scrub(tmp_path: Path, lines: list[str], no_email: bool = False) -> list[str]:
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    driver = tmp_path / "drive_email_scrub.ps1"
    driver.write_text(_PS_DRIVER, encoding="utf-8")
    infile = tmp_path / "in.txt"
    infile.write_text("\n".join(lines), encoding="utf-8")
    args = [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(driver), "-Target", str(_PS1_PATH), "-InFile", str(infile)]
    if no_email:
        args.append("-NoEmail")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "SCRUB-NOT-FOUND" not in out, out
    assert proc.returncode == 0, f"driver failed ({proc.returncode}): {out}"
    return (tmp_path / "in.txt.out").read_text(encoding="utf-8").split("\n")


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_scrub_matches_python_when_actually_executed(tmp_path):
    keep = f"keep {MAINTAINER.upper()}. and {MAINTAINER.replace('@', '%40')} '{MAINTAINER}'"
    inputs = [text for text, _ in _CASES] + [keep]
    expected = [want for _, want in _CASES] + [keep]
    got = _run_ps_scrub(tmp_path, inputs)
    once, twice = got[0::2], got[1::2]
    assert once == expected
    assert twice == expected


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_scrub_without_the_maintainer_address_redacts_everything(tmp_path):
    got = _run_ps_scrub(tmp_path, [f"{OTHER} {MAINTAINER}"], no_email=True)
    assert got[0] == "<redacted-email> <redacted-email>"


def _best(text: str, samples: int = 5) -> float:
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        br_scrub._scrub_emails(text)
        times.append(time.perf_counter() - start)
    return min(times)


_HOSTILE = [
    pytest.param(lambda n: "a" * n, id="token-run"),
    pytest.param(lambda n: "a@" * n, id="many-at-signs"),
    pytest.param(lambda n: "a@" + "b1." * n, id="many-labels-no-tld"),
    pytest.param(lambda n: "x@" + "a-" * n + ".", id="one-long-label"),
    pytest.param(lambda n: "a.b@c" * n, id="chained-addresses"),
    pytest.param(lambda n: "a%40" * n, id="many-encoded-at-signs"),
    pytest.param(lambda n: (E_ACUTE + "@") * n, id="non-ascii-at-signs"),
    pytest.param(lambda n: "'a@" * n, id="quoted-at-signs"),
]


@pytest.mark.parametrize("make", _HOSTILE)
def test_email_pattern_does_not_grow_superlinearly(make):
    """10x input: linear costs ~10x, quadratic ~100x; the bound is 30x."""
    base = _best(make(10_000))
    ten_x = _best(make(100_000))
    assert ten_x < max(30 * base, 0.005), (
        f"{base * 1000:.2f}ms -> {ten_x * 1000:.2f}ms for a 10x input")
