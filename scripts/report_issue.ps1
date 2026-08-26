# SPDX-License-Identifier: AGPL-3.0-or-later
# Pure-PowerShell bug reporter: the no-Python fallback that report-issue.bat uses
# when localm will not start AND no Python is available (e.g. setup failed before
# uv provisioned one). It reads the proxy URL+token from localm/config.py, shows
# exactly what will be sent, and files an account-less GitHub issue via the proxy
# after you confirm. A failed or declined send is NEVER reported as success
# (AGENTS.md rule 5): it saves the report and points at the maintainer email.
#
# Args (passed through by report-issue.bat): --summary --detail --log --yes
# No param()/[CmdletBinding()] on purpose: the --foo pass-through args must land in
# $args, which advanced-function binding would instead reject as unknown parameters.

$ErrorActionPreference = 'Stop'
# scripts/report_issue.ps1 -> repo root is two levels up.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$email = 'theilige@gmail.com'
$fence = '```'

# ---- parse the pass-through args ($args, not PowerShell -params) -----------
$summary = ''; $detail = ''; $logPath = ''; $yes = $false
for ($i = 0; $i -lt $args.Count; $i++) {
  switch ($args[$i]) {
    '--summary' { $i++; $summary = [string]$args[$i] }
    '--detail'  { $i++; $detail  = [string]$args[$i] }
    '--log'     { $i++; $logPath = [string]$args[$i] }
    '--yes'     { $yes = $true }
  }
}

function Read-Proxy {
  $u = $env:LOCALM_BUGREPORT_URL; $t = $env:LOCALM_BUGREPORT_TOKEN
  $cfg = Join-Path $root 'localm\config.py'
  if (Test-Path -LiteralPath $cfg) {
    $text = Get-Content -Raw -LiteralPath $cfg
    if (-not $u -and $text -match '"bugreport_upload_url"\s*:\s*"([^"]+)"')   { $u = $Matches[1] }
    if (-not $t -and $text -match '"bugreport_upload_token"\s*:\s*"([^"]+)"') { $t = $Matches[1] }
  }
  return @($u, $t)
}

function Scrub([string]$t) {
  if (-not $t) { return $t }
  # Mirror scripts/report_issue.py scrub() / localm/bugreport.py _scrub_secrets: strip
  # the account name from any home path AND obvious credentials, so nothing this
  # reporter files carries a username or a pasted secret. Every strip the Python
  # reporters apply is applied here too, in the same order: a fallback reporter that
  # scrubs LESS than the in-app one is the shape that leaks.
  $ic = [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
  # Home-path username strip (Windows C:\Users\<name>, plus /home/ and /Users/ forms).
  $t = [regex]::Replace($t, '([A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)[^\\/\r\n]+', '${1}<redacted>', $ic)
  # user:pass@ credentials in any URL-ish value (a comfy/searx/remote-server URL).
  $t = [regex]::Replace($t, '(://)[^/@\s]+@', '${1}<redacted>@')
  # Credential-named assignments (a URL query parameter, a .env line, a shell line)
  # and pasted HTTP header lines. .NET ports of _QUERY_SECRET_RE / _HEADER_SECRET_RE
  # in localm/bugreport.py; the three-branch reasoning is documented there. Applied
  # in the same order as scripts/report_issue.py, so all three reporters agree.
  $t = [regex]::Replace($t, '(?i)((?:(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth|access[_-]?token|sig|signature)|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature|key|auth|sig))=)(?![\"'']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[`\"''\)\]\}]{0,4}(?:[\s&#]|$))(?:\"[^\"\r\n]*\"?|''[^''\r\n]*''?|[^&\s#\"''\)\]\}]*)', '${1}<redacted>')
  $t = [regex]::Replace($t, '(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token|authorization)\s*:\s*)(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+', '${1}<redacted>')
  # Bearer tokens and OpenAI-style / localm API keys.
  $t = [regex]::Replace($t, '(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}', '${1}<redacted>')
  $t = [regex]::Replace($t, '(?i)\b(?:sk|localm[_-]sk)-[A-Za-z0-9._\-]{12,}', '<redacted>')
  return $t
}

Write-Host ''
Write-Host '  LocaLM - report a problem (PowerShell fallback; no Python found)'
Write-Host ''

# Mirrors report_issue.py's `interactive = sys.stdin.isatty()` gate. Against
# redirected/closed stdin, Read-Host either throws (-NonInteractive) or
# returns an empty string that is indistinguishable from a real Enter
# keypress - and an empty confirm answer means "yes" for a real user at the
# console. So an unattended run must never reach Read-Host at all, or it
# sends the report with no one having reviewed or confirmed it.
$interactive = -not [Console]::IsInputRedirected

if (-not $summary -and $interactive) {
  try { $summary = Read-Host '  One line - what went wrong?' } catch { $summary = '' }
}
if (-not $detail -and $interactive) {
  try { $detail = Read-Host '  Any more detail (optional, Enter to skip)' } catch { $detail = '' }
}
if (-not $summary) { $summary = 'localm would not start' }

$os = [System.Environment]::OSVersion.VersionString
$body = "# localm bug report: $summary`n`n" +
        "## What happened`n$detail`n`n" +
        "## How it was filed`n- Filed via the standalone PowerShell reporter (no Python on this machine).`n`n" +
        "## Environment`n- OS: $os`n- PowerShell: $($PSVersionTable.PSVersion)"
if ($logPath -and (Test-Path -LiteralPath $logPath)) {
  try {
    $tail = (Get-Content -LiteralPath $logPath -Tail 120 -ErrorAction Stop) -join "`n"
    if ($tail) { $body += "`n`n## Recent log tail`n$fence`n" + (Scrub $tail) + "`n$fence" }
  } catch { }
}
$body = Scrub $body

Write-Host ''
Write-Host '  ----- this is exactly what will be sent -----'
foreach ($ln in ($body -split "`n")) { Write-Host ('  ' + $ln) }
Write-Host '  ---------------------------------------------'
Write-Host ''

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
function Save-Local {
  try {
    $dir = Join-Path $root 'home\bug-reports'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $p = Join-Path $dir "bug-$stamp.md"
    Set-Content -LiteralPath $p -Value $body -Encoding utf8
    return $p
  } catch { return $null }
}

$send = $yes
if (-not $send) {
  if (-not $interactive) {
    # No one is here to confirm it: fail safe, same as a "no" answer, and
    # say so rather than sending a report nobody reviewed.
    $p = Save-Local
    Write-Host "  Not a terminal, so nothing was sent. Saved at $p."
    Write-Host "  Send it by running report-issue again, or email $email."
    exit 0
  }
  try {
    $ans = Read-Host '  Send this to the maintainer now? [Y/n]'
    $send = -not ($ans -match '^[Nn]')
  } catch {
    $send = $false
  }
}
if (-not $send) {
  $p = Save-Local
  Write-Host "  Not sent. Saved at $p - email it to $email if you change your mind."
  exit 0
}

$pair = Read-Proxy; $url = $pair[0]; $tok = $pair[1]
if (-not $url) {
  $p = Save-Local
  Write-Host "  No bug-report endpoint is configured. Saved at $p; email it to $email."
  exit 1
}

$headers = @{ 'Content-Type' = 'application/json'; 'User-Agent' = 'localm-report-issue' }
if ($tok) { $headers['X-Localm-Token'] = $tok }
# Scrub the title too: it becomes a PUBLIC GitHub issue title. The body is already
# scrubbed (above), so a raw title would leak the username/credential the preview
# banner claims is "exactly what will be sent" (HON-03, matching report_issue.py).
$payload = @{ title = (Scrub $summary); body = $body } | ConvertTo-Json -Depth 4
try {
  $resp = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $payload -TimeoutSec 20
  Save-Local | Out-Null
  $link = $null; if ($resp -and $resp.url) { $link = $resp.url }
  if ($link) { Write-Host "  Sent to the maintainer. Thank you! Tracking issue: $link" }
  else       { Write-Host '  Sent to the maintainer. Thank you!' }
  exit 0
} catch {
  $p = Save-Local
  Write-Host "  Could not send it ($($_.Exception.Message)). Saved at $p - email it to $email instead."
  exit 1
}
