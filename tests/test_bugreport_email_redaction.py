# SPDX-License-Identifier: AGPL-3.0-or-later
"""Email redaction in the bug-report scrubbers: localm/bugreport/scrub.py
``_scrub_secrets``, scripts/report_issue.py ``scrub`` and scripts/report_issue.ps1
``Scrub``. Every address except the maintainer's becomes ``<redacted-email>``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from localm.bugreport import _common
from localm.bugreport import scrub as br_scrub

_REPO = Path(__file__).resolve().parents[1]
_RI_PATH = _REPO / "scripts" / "report_issue.py"
_PS1_PATH = _REPO / "scripts" / "report_issue.ps1"

_spec = importlib.util.spec_from_file_location("report_issue_email", _RI_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)

MAINTAINER = _common.MAINTAINER_EMAIL
OTHER = "bob.builder@example.com"

_SCRUBBERS = [
    pytest.param(br_scrub._scrub_secrets, id="bugreport"),
    pytest.param(ri.scrub, id="report_issue_py"),
]


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_other_addresses_are_redacted(scrub):
    text = (f"reviewer {OTHER}, cc Jane.Doe+tag@mail.example.co.uk; "
            f"<carol_99@sub-domain.example.org> mailto:dave@example.io")
    out = scrub(text)
    assert OTHER not in out
    assert "Jane.Doe+tag@mail.example.co.uk" not in out
    assert "carol_99@sub-domain.example.org" not in out
    assert "dave@example.io" not in out
    assert out == ("reviewer <redacted-email>, cc <redacted-email>; "
                   "<<redacted-email>> mailto:<redacted-email>")


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_maintainer_address_is_kept_as_written(scrub):
    upper = MAINTAINER.upper()
    out = scrub(f"write to {MAINTAINER} or {upper}.")
    assert out == f"write to {MAINTAINER} or {upper}."


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_maintainer_lookalikes_are_redacted(scrub):
    local, domain = MAINTAINER.split("@")
    lookalikes = [f"x{MAINTAINER}", f"{MAINTAINER}.evil.example",
                  f"{local}@{domain.split('.')[0]}.example"]
    out = scrub(" ".join(lookalikes))
    assert out == " ".join(["<redacted-email>"] * len(lookalikes))


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_url_credentials_are_not_redacted_twice(scrub):
    out = scrub("http://bob@host.example.com:8188/ and https://u:pw@h.example.org/x")
    assert out == "http://<redacted>@host.example.com:8188/ and https://<redacted>@h.example.org/x"


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_redaction_is_idempotent(scrub):
    text = (f"{OTHER} {MAINTAINER} http://a@b.example.com/ api_key={OTHER} "
            f'{{"email": "{OTHER}"}} ')
    once = scrub(text)
    assert OTHER not in once
    assert scrub(once) == once


@pytest.mark.parametrize("scrub", _SCRUBBERS)
def test_non_addresses_are_left_alone(scrub):
    text = "user@host a@b x@localhost npm @scope/pkg@1.2.3 v1@2.0"
    assert scrub(text) == text


def test_all_three_scrubbers_share_one_pattern_and_address():
    assert ri._EMAIL_RE.pattern == br_scrub._EMAIL_RE.pattern
    assert ri.MAINTAINER_EMAIL == MAINTAINER
    ps1 = _PS1_PATH.read_text(encoding="utf-8")
    assert f"'{br_scrub._EMAIL_RE.pattern}'" in ps1
    assert f"$email = '{MAINTAINER}'" in ps1


_PS_DRIVER = (
    "param([string]$Target, [string]$Text, [switch]$NoEmail)\n"
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
    "$once = Scrub $Text\n"
    "Write-Output \"ONCE=$once\"\n"
    "Write-Output \"TWICE=$(Scrub $once)\"\n"
)


def _run_ps_scrub(tmp_path: Path, text: str, no_email: bool = False) -> dict:
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    driver = tmp_path / "drive_email_scrub.ps1"
    driver.write_text(_PS_DRIVER, encoding="utf-8")
    args = [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(driver), "-Target", str(_PS1_PATH), "-Text", text]
    if no_email:
        args.append("-NoEmail")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "SCRUB-NOT-FOUND" not in out, out
    assert proc.returncode == 0, f"driver failed ({proc.returncode}): {out}"
    lines = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                 if line.startswith(("ONCE=", "TWICE=")))
    return lines


_PS_SAMPLE = (f"reviewer {OTHER} keep {MAINTAINER.upper()} "
              f"x{MAINTAINER} http://bob@host.example.com/")


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_scrub_redacts_emails_when_actually_executed(tmp_path):
    lines = _run_ps_scrub(tmp_path, _PS_SAMPLE)
    once = lines["ONCE"]
    assert OTHER not in once, once
    assert once == ri.scrub(_PS_SAMPLE)
    assert once == (f"reviewer <redacted-email> keep {MAINTAINER.upper()} "
                    "<redacted-email> http://<redacted>@host.example.com/")
    assert lines["TWICE"] == once


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_scrub_without_the_maintainer_address_redacts_everything(tmp_path):
    lines = _run_ps_scrub(tmp_path, f"{OTHER} {MAINTAINER}", no_email=True)
    assert lines["ONCE"] == "<redacted-email> <redacted-email>"


def _best(text: str, samples: int = 5) -> float:
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        br_scrub._scrub_emails(text)
        times.append(time.perf_counter() - start)
    return min(times)


_HOSTILE = [
    pytest.param(lambda n: "a" * n, id="local-part-run"),
    pytest.param(lambda n: "a@" * n, id="many-at-signs"),
    pytest.param(lambda n: "a@" + "b1." * n, id="many-labels-no-tld"),
    pytest.param(lambda n: "x@" + "a-" * n + ".", id="one-long-label"),
    pytest.param(lambda n: "a.b@c" * n, id="chained-addresses"),
]


@pytest.mark.parametrize("make", _HOSTILE)
def test_email_pattern_does_not_grow_superlinearly(make):
    """10x input: linear costs ~10x, quadratic ~100x; the bound is 30x."""
    base = _best(make(10_000))
    ten_x = _best(make(100_000))
    assert ten_x < max(30 * base, 0.005), (
        f"{base * 1000:.2f}ms -> {ten_x * 1000:.2f}ms for a 10x input")
