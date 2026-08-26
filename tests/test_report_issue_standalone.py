# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/report_issue.py - the standalone reporter used when localm
will not start. Focus: the privacy scrub (never leak a username/token), honest
failure (a failed/declined send is never reported as success), reading the proxy
config from source, and the report body."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Load the standalone script by path (scripts/ is not an importable package).
_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_issue.py"
_spec = importlib.util.spec_from_file_location("report_issue_standalone", _MOD_PATH)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)


# ----------------------------- read_proxy --------------------------------- #

def test_read_proxy_extracts_url_and_token(tmp_path):
    cfg = tmp_path / "config.py"
    cfg.write_text(
        'DEFAULT_CONFIG = {\n'
        '    "bugreport_upload_url": "https://proxy.example.workers.dev",\n'
        '    "bugreport_upload_token": "tok_ABC-123",\n'
        '}\n', encoding="utf-8")
    url, token = ri.read_proxy(cfg)
    assert url == "https://proxy.example.workers.dev"
    assert token == "tok_ABC-123"


def test_read_proxy_missing_file_is_none(tmp_path):
    assert ri.read_proxy(tmp_path / "nope.py") == (None, None)


def test_read_proxy_reads_the_real_shipped_config():
    # The real localm/config.py must carry a usable proxy URL, so the standalone
    # reporter works with no install.
    url, token = ri.read_proxy()
    assert url and url.startswith("https://")


# ------------------------------- scrub ------------------------------------ #

def test_scrub_redacts_windows_username():
    out = ri.scrub(r"error at Z:\Users\alice\localm\home\logs\x.log")
    assert "alice" not in out
    assert "<redacted>" in out or "~" in out


def test_scrub_redacts_posix_username():
    out = ri.scrub("Traceback: /home/bob/localm/setup.sh line 3")
    assert "bob" not in out
    assert "<redacted>" in out or "~" in out


def test_scrub_redacts_macos_username():
    out = ri.scrub("/Users/carol/localm crashed")
    assert "carol" not in out


def test_scrub_strips_bearer_and_api_keys():
    out = ri.scrub("auth Bearer abcdef1234567890 and sk-abcdef0123456789xyz")
    assert "abcdef1234567890" not in out
    assert "sk-abcdef0123456789xyz" not in out
    assert "<redacted>" in out


def test_scrub_strips_url_credentials():
    """The standalone scrub() must mirror localm/bugreport.py _scrub_secrets, which
    strips user:pass@ from URLs - otherwise a credentialed URL pasted into --summary
    or --detail is scrubbed by the in-app reporter but leaks through this fallback."""
    out = ri.scrub("POST http://admin:SECRETPASS@api.corp.local/v1 failed")
    assert "SECRETPASS" not in out
    assert "<redacted>" in out


def test_scrub_strips_query_string_and_header_credentials():
    """Mirrors localm/bugreport.py's _scrub_query_and_header_secrets: a
    credential is at least as often carried as a URL query parameter or a
    pasted header line as via user:pass@ syntax. Same assertion block as the
    user:pass@ case, deliberately - if a regression deletes the query/header
    redaction here, a user:pass@-only test would not catch it, and vice
    versa."""
    out = ri.scrub(
        "https://x.example/s?api_key=CANARY1&q=hello "
        "and X-Api-Key: CANARY2 "
        "and http://admin:CANARY3@api.corp.local/v1")
    assert "CANARY1" not in out
    assert "CANARY2" not in out
    assert "CANARY3" not in out
    assert "api_key=<redacted>" in out
    assert "q=hello" in out  # non-credential param left intact


def test_scrub_strips_bare_and_prefixed_credential_assignments():
    """ri.scrub redacts a credential written as a plain name=value line (a .env
    fragment, a shell line) or behind a prefix (OPENAI_API_KEY=, pull_token=),
    matching the bare-name widening of _QUERY_SECRET_RE in
    localm/bugreport.py.

    Both directions are asserted in one block: the credential assignments are
    redacted, and ordinary config text (n_gpu_layers=35, key=value, monkey=13)
    is left intact."""
    out = ri.scrub(
        "pasted from my .env:\napi_key=CANARYBARE7Q4M\n"
        "OPENAI_API_KEY=CANARYBARE1AAA\nSECRET_KEY=CANARYBARE2BBB\n"
        "n_gpu_layers=35 key=value monkey=13\n")
    assert "CANARYBARE7Q4M" not in out
    assert "CANARYBARE1AAA" not in out
    assert "CANARYBARE2BBB" not in out
    assert "api_key=<redacted>" in out
    assert "OPENAI_API_KEY=<redacted>" in out
    assert "SECRET_KEY=<redacted>" in out
    assert "n_gpu_layers=35" in out
    assert "key=value" in out
    assert "monkey=13" in out


def test_scrub_keeps_assignments_whose_value_cannot_be_a_secret():
    """Mirrors the non-secret-literal suppressor in localm/bugreport.py. Asserted
    in the same block as a real redaction so a regression that deletes the
    suppressor and one that deletes the scrub are both caught here."""
    out = ri.scrub(
        "LOCALM_REQUIRE_AUTH=1 require_auth=true has_token=false "
        "digital_signature=True api_key=CANARYLIT9 api_key=trueCANARYLIT8")
    assert "CANARYLIT9" not in out
    assert "CANARYLIT8" not in out
    assert "LOCALM_REQUIRE_AUTH=1" in out
    assert "require_auth=true" in out
    assert "has_token=false" in out
    assert "digital_signature=True" in out


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
def test_powershell_scrub_behaves_when_actually_executed(tmp_path):
    """The other ps1 test is a STATIC text guard: it asserts the file CONTAINS
    the patterns. That would pass on a .ps1 whose regex is syntactically invalid
    or whose replacement string is wrong, which is a real gap because nothing
    else in either suite ever runs this file.

    This one executes the shipped Scrub for real. It extracts the function by
    AST so what runs is the file's own bytes, not a retyped copy, and asserts
    BOTH directions in one call so a regression in either is caught."""
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("no PowerShell interpreter on PATH")
    ps1 = _MOD_PATH.parent / "report_issue.ps1"
    driver = tmp_path / "drive_scrub.ps1"
    driver.write_text(
        "param([string]$Target)\n"
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        "$Target, [ref]$null, [ref]$null)\n"
        "$fn = $ast.FindAll({ param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
        "$n.Name -eq 'Scrub' }, $true)\n"
        "if ($fn.Count -ne 1) { Write-Output 'SCRUB-NOT-FOUND'; exit 1 }\n"
        "Invoke-Expression $fn[0].Extent.Text\n"
        "Write-Output (Scrub 'api_key=PSCANARY7Q4M require_auth=true "
        "n_gpu_layers=35')\n",
        encoding="utf-8")
    proc = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver), "-Target", str(ps1)],
        capture_output=True, text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "SCRUB-NOT-FOUND" not in out, out
    assert proc.returncode == 0, f"driver failed ({proc.returncode}): {out}"
    # DATA first: the canary must be gone.
    assert "PSCANARY7Q4M" not in out, f"the PowerShell scrub shipped a secret: {out}"
    assert "api_key=<redacted>" in out, out
    # And the other direction: a non-credential value survives.
    assert "require_auth=true" in out, f"the PowerShell scrub ate a flag: {out}"
    assert "n_gpu_layers=35" in out, f"the PowerShell scrub ate a setting: {out}"


def test_scrub_strips_a_quoted_credential_value():
    """Mirrors the quoted-value fix in localm/bugreport.py. A .env writes the
    value quoted more often than bare, and stopping at the opening quote left
    the secret in the report next to a marker claiming it had gone."""
    out = ri.scrub(
        'my .env has API_KEY="RIQUOTED1" and SECRET_KEY=\'RIQUOTED2\' '
        'and require_auth="true" and n_gpu_layers=35')
    assert "RIQUOTED1" not in out
    assert "RIQUOTED2" not in out
    assert '<redacted>"' not in out
    assert 'require_auth="true"' in out
    assert "n_gpu_layers=35" in out


def test_scrub_empty_is_safe():
    assert ri.scrub("") == ""
    assert ri.scrub(None) is None


# ----------------------------- build_body --------------------------------- #

def test_build_body_includes_description_and_scrubbed_log():
    diag = {"platform": "Windows-11", "python": "3.12.1", "venv_present": False,
            "localm_version": "0.1.1"}
    body = ri.build_body("setup failed", "uv would not install", diag,
                         Path("home/logs/x.log"), "some tail line")
    assert "setup failed" in body
    assert "uv would not install" in body
    assert "some tail line" in body
    assert "Venv present: no" in body


def test_build_body_handles_empty_description():
    body = ri.build_body("x", "", {}, None, "")
    assert "(no description given)" in body


# ----------------------------- post_report -------------------------------- #

def test_post_report_success_returns_json():
    def opener(u, data, hdrs, to):
        assert hdrs.get("X-Localm-Token") == "tok"
        return 201, '{"ok": true, "url": "https://github.com/o/r/issues/7"}'
    res = ri.post_report("https://proxy", "tok", "title", "body", opener=opener)
    assert res["url"].endswith("/issues/7")


def test_post_report_non_2xx_raises():
    def opener(u, data, hdrs, to):
        return 500, "boom"
    with pytest.raises(RuntimeError):
        ri.post_report("https://proxy", "tok", "t", "b", opener=opener)


def test_post_report_network_error_raises():
    def opener(u, data, hdrs, to):
        raise RuntimeError("could not reach the server: timed out")
    with pytest.raises(RuntimeError):
        ri.post_report("https://proxy", None, "t", "b", opener=opener)


def test_post_report_omits_token_header_when_none():
    seen = {}
    def opener(u, data, hdrs, to):
        seen.update(hdrs)
        return 201, "{}"
    ri.post_report("https://proxy", None, "t", "b", opener=opener)
    assert "X-Localm-Token" not in seen


# ------------------------------- save_report ------------------------------ #

def test_save_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    path = ri.save_report("hello report", "20260101-000000")
    assert path is not None and path.exists()
    assert path.read_text(encoding="utf-8") == "hello report"


# --------------------------- main honest failure -------------------------- #

def _no_tty(monkeypatch):
    monkeypatch.setattr(ri.sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())


def test_main_noninteractive_without_yes_saves_and_never_sends(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy", "tok"))
    called = {"sent": False}
    def _boom(*a, **k):
        called["sent"] = True
        raise AssertionError("must not send without confirmation")
    monkeypatch.setattr(ri, "post_report", _boom)
    rc = ri.main(["--summary", "x"])
    assert rc == 0
    assert called["sent"] is False
    out = capsys.readouterr().out
    assert "Sent to the maintainer" not in out


def test_main_yes_send_failure_returns_1_and_does_not_claim_success(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy", "tok"))
    def _fail(*a, **k):
        raise RuntimeError("HTTP 502: bad gateway")
    monkeypatch.setattr(ri, "post_report", _fail)
    rc = ri.main(["--yes", "--summary", "x"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Sent to the maintainer" not in out
    assert "Could not send" in out


def test_main_yes_send_success_returns_0(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy", "tok"))
    monkeypatch.setattr(ri, "post_report",
                        lambda *a, **k: {"url": "https://github.com/o/r/issues/9"})
    rc = ri.main(["--yes", "--summary", "x"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sent to the maintainer" in out
    assert "issues/9" in out


def test_main_yes_no_endpoint_configured_saves_and_returns_1(tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: (None, None))
    rc = ri.main(["--yes", "--summary", "x"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "No bug-report endpoint" in out


def test_yes_help_text_says_it_sends_not_preview_only(capsys):
    """--yes genuinely SENDS (see test_main_yes_send_success_returns_0 above),
    and the help text must say so plainly. Wording like "skip the confirm prompt
    (still previews)" reads as "preview only, does not send" and gets run as a
    dry run that files a real GitHub issue with test content."""
    with pytest.raises(SystemExit):
        ri.main(["--help"])
    out = capsys.readouterr().out
    assert "--yes" in out
    # argparse re-flows the whole docstring and can wrap a line anywhere
    # (including between two words of the flag's own description), so compare
    # against whitespace-collapsed text rather than a literal substring.
    flat = " ".join(out.split())
    assert "skip the confirm prompt (still previews)" not in flat
    assert "SEND immediately" in flat


def test_main_scrubs_home_path_in_uploaded_title(tmp_path, monkeypatch, capsys):
    """The issue TITLE lands on a PUBLIC GitHub issue, so it must be scrubbed
    like the body/preview. A home path (username) in --summary must not reach the
    title unredacted - the tool's banner claims the preview is 'exactly what will
    be sent'."""
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy", "tok"))
    captured = {}

    def _capture(url, token, title, body, **k):
        captured["title"] = title
        captured["body"] = body
        return {"url": "https://github.com/o/r/issues/1"}

    monkeypatch.setattr(ri, "post_report", _capture)
    rc = ri.main(["--yes", "--summary", r"crash at Z:\Users\bob\localm\home\x.log"])
    assert rc == 0
    # What post_report RECEIVES as the title (what actually gets filed) is scrubbed.
    assert "bob" not in captured["title"]
    assert "<redacted>" in captured["title"]
    assert "bob" not in captured["body"]
    # And the "exactly what will be sent" preview never showed the raw username.
    assert "bob" not in capsys.readouterr().out


def test_main_scrubs_url_credential_in_title_and_body(tmp_path, monkeypatch):
    """A credentialed URL typed into --summary must not reach the PUBLIC issue title
    or body (end-to-end through main() to what post_report is handed)."""
    _no_tty(monkeypatch)
    monkeypatch.setattr(ri, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ri, "read_proxy", lambda: ("https://proxy", "tok"))
    captured = {}

    def _capture(url, token, title, body, **k):
        captured["title"] = title
        captured["body"] = body
        return {"url": "https://github.com/o/r/issues/2"}

    monkeypatch.setattr(ri, "post_report", _capture)
    rc = ri.main(["--yes", "--summary",
                  "login fails via http://admin:SECRETPASS@api.corp.local/v1"])
    assert rc == 0
    assert "SECRETPASS" not in captured["title"]
    assert "SECRETPASS" not in captured["body"]


def test_powershell_reporter_scrubs_title_and_credentials():
    """The PowerShell no-Python fallback reporter (report_issue.ps1) must mirror the
    Python fix: scrub the uploaded TITLE (not the raw $summary), and its Scrub must
    strip URL credentials + API keys, not just home paths + bearer tokens. Static
    guard - the .ps1 is not exercised by this Python suite - to catch a revert."""
    ps1 = _MOD_PATH.parent / "report_issue.ps1"
    text = ps1.read_text(encoding="utf-8")
    # The uploaded title is scrubbed, never the raw $summary.
    assert "title = (Scrub $summary)" in text
    assert "title = $summary" not in text
    # Scrub covers URL user:pass@ credentials and sk-/localm-sk API keys.
    assert r"(://)[^/@\s]+@" in text
    assert "localm[_-]sk" in text
def _copy_ps1_reporter_to_isolated_dir(tmp_path: Path) -> Path:
    """A byte-identical copy of report_issue.ps1, two directories deep under
    tmp_path with no localm/config.py anywhere above it. report_issue.ps1
    resolves its own repo root as two levels up from its own path and reads
    localm\\config.py from there for the proxy URL/token - so this isolation
    makes that lookup structurally unreachable (combined with clearing the
    LOCALM_BUGREPORT_URL/TOKEN env vars below), which is what makes it SAFE
    to actually execute this reporter's full top-level flow in a test: it
    cannot discover a real endpoint to send to, network or no network."""
    real_ps1 = _MOD_PATH.parent / "report_issue.ps1"
    scripts_dir = tmp_path / "isolated" / "scripts"
    scripts_dir.mkdir(parents=True)
    dest = scripts_dir / "report_issue.ps1"
    dest.write_text(real_ps1.read_text(encoding="utf-8"), encoding="utf-8")
    assert not (scripts_dir.parent / "localm").exists()
    return dest


@pytest.mark.skipif(os.name != "nt",
                    reason="the PowerShell fallback reporter only runs on Windows")
class TestPowerShellReporterFailsSafeUnattended:
    """report_issue.ps1's three Read-Host prompts ran unguarded. Under
    -NonInteractive, Read-Host THROWS and $ErrorActionPreference='Stop'
    aborts the whole script before anything is saved - the report is lost
    with no trace. Against stdin that is merely closed/redirected (no
    -NonInteractive flag), Read-Host does not throw at all: it silently
    returns an empty string, which the confirm prompt's own regex reads as
    "yes" - so the report was SENT with nobody having reviewed or confirmed
    it. Both must now fail safe: save locally, never send, never lose the
    report to an unhandled exception."""

    def _pwsh(self):
        pwsh = shutil.which("powershell") or shutil.which("pwsh")
        if not pwsh:
            pytest.skip("no PowerShell interpreter on PATH")
        return pwsh

    def _clean_env(self):
        env = dict(os.environ)
        env.pop("LOCALM_BUGREPORT_URL", None)
        env.pop("LOCALM_BUGREPORT_TOKEN", None)
        assert "LOCALM_BUGREPORT_URL" not in env
        assert "LOCALM_BUGREPORT_TOKEN" not in env
        return env

    def test_noninteractive_flag_saves_and_never_sends(self, tmp_path):
        ps1 = _copy_ps1_reporter_to_isolated_dir(tmp_path)
        proc = subprocess.run(
            [self._pwsh(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(ps1), "--summary", "unattended test", "--detail", "no one is here"],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
            env=self._clean_env())
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, f"script exited {proc.returncode}: {out}"
        assert "Sent to the maintainer" not in out, out
        assert "Not a terminal" in out, out
        saved = list((ps1.parent.parent / "home" / "bug-reports").glob("bug-*.md"))
        assert saved, f"nothing was saved locally: {out}"
        assert "unattended test" in saved[0].read_text(encoding="utf-8")

    def test_closed_stdin_without_noninteractive_flag_saves_and_never_sends(self, tmp_path):
        """The OTHER failure shape: stdin is closed/redirected but the
        -NonInteractive flag was never passed. Read-Host does not throw here -
        it silently returns an empty string, exactly the input that used to
        read as "yes" at the confirm prompt."""
        ps1 = _copy_ps1_reporter_to_isolated_dir(tmp_path)
        proc = subprocess.run(
            [self._pwsh(), "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(ps1), "--summary", "eof test", "--detail", "no one is here"],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
            env=self._clean_env())
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, f"script exited {proc.returncode}: {out}"
        assert "Sent to the maintainer" not in out, out
        # The specific message only the fixed early-exit path prints - without
        # this, the confirm's empty-string answer falls through and is read
        # as consent, which happens to still fail safe in this test only
        # because no endpoint is reachable here (exit 1, different message).
        assert "Not a terminal" in out, out
        saved = list((ps1.parent.parent / "home" / "bug-reports").glob("bug-*.md"))
        assert saved, f"nothing was saved locally: {out}"

    def test_yes_flag_with_no_endpoint_configured_saves_and_reports_failure(self, tmp_path):
        """The pre-existing --yes bypass must still work unchanged: it skips
        the confirm prompt entirely (this fix never touches that branch), and
        with no proxy reachable it must say so and save, never claim
        success - the send path is exercised up to the point of discovering
        there is nowhere to send to, and no further."""
        ps1 = _copy_ps1_reporter_to_isolated_dir(tmp_path)
        proc = subprocess.run(
            [self._pwsh(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(ps1), "--summary", "yes flag test", "--yes"],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
            env=self._clean_env())
        out = (proc.stdout or "") + (proc.stderr or "")
        assert "Sent to the maintainer" not in out, out
        assert "No bug-report endpoint is configured" in out, out
        saved = list((ps1.parent.parent / "home" / "bug-reports").glob("bug-*.md"))
        assert saved, f"nothing was saved locally: {out}"


def test_powershell_reporter_scrubs_credential_named_assignments():
    """report_issue.ps1's Scrub function carries the header-line pattern and
    all three branches of the query pattern, not the ?/&-anchored branch
    alone. Static guard - the .ps1 is not exercised by this Python suite."""
    ps1 = _MOD_PATH.parent / "report_issue.ps1"
    text = ps1.read_text(encoding="utf-8")
    # The header-line port.
    assert r"(?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token)" in text
    # All three branches of the query port.
    assert r"(?<=[?&])" in text
    assert r"(?<![A-Za-z0-9])" in text
    assert r"(?<=[A-Za-z0-9])[_-](?:api[_-]?key" in text
