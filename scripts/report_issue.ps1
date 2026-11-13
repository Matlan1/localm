# SPDX-License-Identifier: AGPL-3.0-or-later
# Pure-PowerShell bug reporter: the no-Python fallback that report-issue.bat uses
# when localm will not start AND no Python is available (e.g. setup failed before
# uv provisioned one). It reads the proxy URL+token from localm/config.py, shows
# exactly what will be sent, and files an account-less GitHub issue via the proxy
# after you confirm. A failed or declined send is NEVER reported as success:
# it saves the report and points at the maintainer email. With localm's Network
# access off (LOCALM_NET_MODE, or net_mode in the data dir's config.json) nothing
# is sent; when that setting cannot be read you are asked before anything is sent.
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

# A copy of localm.config.NET_OFF_REMEDY. See test_the_reporters_copy_the_net_off_remedy.
$netOffRemedy = 'Turn it on in Settings > Server & network > Network access, or run: localm config net_mode ask'

# A path with a leading ~ resolved against the user's home, as Python's expanduser does.
function Expand-Home([string]$p) {
  if ($p -eq '~' -or $p.StartsWith('~/') -or $p.StartsWith('~\')) { return $HOME + $p.Substring(1) }
  return $p
}

# The data dir localm uses: LOCALM_HOME, else (in a clone) the first line of
# localm-home.cfg, else the clone's home folder.
function Get-LocalmHome {
  if ($env:LOCALM_HOME -and $env:LOCALM_HOME.Trim()) { return (Expand-Home $env:LOCALM_HOME.Trim()) }
  $marker = Join-Path $root 'localm-home.cfg'
  if ((Test-Path -LiteralPath (Join-Path $root 'pyproject.toml') -PathType Leaf) -and
      (Test-Path -LiteralPath $marker -PathType Leaf)) {
    try {
      $line = Get-Content -LiteralPath $marker -Encoding UTF8 | Where-Object { $_.Trim() } | Select-Object -First 1
      if ($line) { return (Expand-Home $line.Trim()) }
    } catch { }
  }
  return (Join-Path $root 'home')
}

# True when $t is one JSON value in strict syntax (what Python's json module
# accepts without NaN): no comments, single quotes, bare names or trailing commas.
function Test-StrictJson([string]$t) {
  if ($t -match '[\x00-\x08\x0b\x0c\x0e-\x1f]') { return $false }
  $s = [regex]::Replace($t, '"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"', [string][char]1)
  if ($s.Contains('"')) { return $false }
  $s = [regex]::Replace($s, '-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null', [string][char]2)
  $s = [regex]::Replace($s, '[ \t\r\n]', '')
  $v = '[\x01\x02\x03]'
  do {
    $prev = $s
    $s = [regex]::Replace($s, '\[(?:' + $v + '(?:,' + $v + ')*)?\]', [string][char]3)
    $s = [regex]::Replace($s, '\{(?:\x01:' + $v + '(?:,\x01:' + $v + ')*)?\}', [string][char]3)
  } while ($s -ne $prev)
  return ($s.Length -eq 1 -and [int][char]$s[0] -le 3 -and [int][char]$s[0] -ge 1)
}

# @(mode, why): mode is off, ask or allow as localm reads it (LOCALM_NET_MODE, else
# net_mode in the data dir's config.json, or its .bak when config.json is absent;
# an unset or unknown value, or a file that is not a JSON object, means ask), or
# unknown when neither file exists, or one is present and cannot be read the way
# localm reads it (strict UTF-8 JSON), where localm itself turns network access off.
function Get-NetMode {
  $modes = @('off', 'ask', 'allow')
  $e = "$env:LOCALM_NET_MODE".Trim().ToLowerInvariant()
  if ($modes -contains $e) { return @($e, '') }
  try {
    $main = Join-Path (Get-LocalmHome) 'config.json'
    $p = $null
    foreach ($c in @($main, "$main.bak")) {
      if (Test-Path -LiteralPath $c -PathType Leaf) { $p = $c; break }
    }
    if (-not $p) { return @('unknown', "$main does not exist") }
    $bytes = [System.IO.File]::ReadAllBytes($p)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
      return @('unknown', "$p could not be read (it starts with a byte order mark)")
    }
    $raw = (New-Object System.Text.UTF8Encoding($false, $true)).GetString($bytes)
    if (-not (Test-StrictJson $raw)) { return @('unknown', "$p could not be read (not valid JSON)") }
    $cfg = $raw | ConvertFrom-Json
  } catch {
    return @('unknown', "the setting could not be read ($($_.Exception.Message))")
  }
  $m = 'ask'
  if ($cfg -is [System.Management.Automation.PSCustomObject]) {
    $prop = $cfg.PSObject.Properties | Where-Object { $_.Name -ceq 'net_mode' } | Select-Object -First 1
    if ($prop) { $m = "$($prop.Value)".Trim().ToLowerInvariant() }
  }
  if ($modes -notcontains $m) { $m = 'ask' }
  return @($m, '')
}

function Scrub([string]$t) {
  if (-not $t) { return $t }
  # Mirror scripts/report_issue.py scrub() / localm/bugreport/scrub.py _scrub_secrets: strip
  # the account name from any home path AND obvious credentials, so nothing this
  # reporter files carries a username or a pasted secret. Every strip the Python
  # reporters apply is applied here too, in the same order: a fallback reporter that
  # scrubs LESS than the in-app one is the shape that leaks.
  $ic = [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
  # The home folder itself (as given, forward-slash and doubled-backslash forms) becomes ~.
  $h = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
  if ($h) {
    foreach ($v in @($h, $h.Replace('\', '/'), $h.Replace('\', '\\'))) {
      $t = [regex]::Replace($t, [regex]::Escape($v), '~', $ic)
    }
  }
  # Home-path username strip (Windows C:\Users\<name>, plus /home/ and /Users/ forms).
  # The Windows separator is one OR two backslashes, so a repr()'d or JSON-encoded path is matched too.
  $t = [regex]::Replace($t, '([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)|/home/|/Users/)[^\\/\r\n]+', '${1}<redacted>', $ic)
  # user:pass@ credentials in any URL-ish value (a comfy/searx/remote-server URL).
  $t = [regex]::Replace($t, '(://)[^/@\s]+@', '${1}<redacted>@')
  # Bearer tokens.
  $t = [regex]::Replace($t, '(?i)(bearer\s+)(?![A-Za-z0-9_-]+(?:\s*:|[ \t]*=[ \t]*[^=\s]))[A-Za-z0-9._~+/=\-]{8,}', '${1}<redacted>')
  # Credential-named assignments (a URL query parameter, a .env line, a shell line)
  # and pasted HTTP header lines. .NET ports of _QUERY_SECRET_RE / _AUTH_HEADER_RE /
  # _HEADER_SECRET_RE in localm/bugreport/scrub.py; the reasoning is documented there. Applied
  # in the same order as scripts/report_issue.py, so all three reporters agree.
  $t = [regex]::Replace($t, '(?i)((?:(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth|access[_-]?token|sig|signature)|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature|key|auth|sig))[ \t]*=[ \t]*(?![ \t]))(?![\"'']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[`\"''\)\]\}]{0,4}(?:[\s&#]|$))(?:\"[^\"\r\n]*\"?|''[^''\r\n]*''?|[^&\s#\"''\)\]\}]*)', '${1}<redacted>')
  $t = [regex]::Replace($t, '(?i)(authorization[\"'']?\s*:\s*)(?=\S)(?!<redacted>)(?![\"'']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[\"'']?\s*(?:[,}\]]|$))(?:[A-Za-z]?\"(?:[^\"\\\r\n]|\\.)*\"?|[A-Za-z]?''(?:[^''\\\r\n]|\\.)*''?|\[[^\]\r\n]*\]?|(?:(?:bearer|basic|digest|negotiate|ntlm|token|apikey)\s+|[A-Za-z][A-Za-z0-9_-]*[ \t]+(?=[^\s,]))?(?![A-Za-z0-9_-]+[\"'']?:(?:\s|$))(?:[A-Za-z][A-Za-z0-9_-]*[ \t]*=[ \t]*(?:\"(?:[^\"\\\r\n]|\\.)*\"?|\\\"(?:[^\"\\\r\n]|\\[^\"\r\n])*(?:\\\")?|[^\s,]*)|[^\s,]+)(?:,[ \t]*[A-Za-z][A-Za-z0-9_-]*[ \t]*=[ \t]*(?:\"(?:[^\"\\\r\n]|\\.)*\"?|\\\"(?:[^\"\\\r\n]|\\[^\"\r\n])*(?:\\\")?|[^\s,]*))*)', '${1}<redacted>')
  $t = [regex]::Replace($t, '(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token)\s*:\s*)(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+', '${1}<redacted>')
  # Credential-named JSON object keys ("api_key": "..."), .NET port of _JSON_SECRET_RE.
  $t = [regex]::Replace($t, '(?i)([\"''](?:(?:[A-Za-z0-9_-]*[_-])?(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|[A-Za-z0-9_-]*[A-Za-z0-9][_-](?:key|auth|sig))[\"'']\s*:\s*)(?=\S)(?!<redacted>)(?!\s*[\"'']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[\"'']?\s*(?:[,}\]]|$))(?:[A-Za-z]?\"(?:[^\"\\\r\n]|\\.)*\"?|[A-Za-z]?''(?:[^''\\\r\n]|\\.)*''?|\[[^\]\r\n]*\]?|(?:(?:bearer|basic|digest|negotiate|ntlm)[ \t]+)?[^,}\]\s]*)', '${1}<redacted>')
  # OpenAI-style / localm API keys.
  $t = [regex]::Replace($t, '(?i)\b(?:sk|localm[_-]sk)-[A-Za-z0-9._\-]{12,}', '<redacted>')
  # Tokens that contain an email address, except the maintainer's ($email: ASCII, case-insensitive,
  # %40 read as @, edge single quotes and trailing periods set aside). Same pattern as _EMAIL_RE in localm/bugreport/scrub.py.
  $t = [regex]::Replace($t, '(?<![^\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*])[^\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*]*?[^@\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*](?:@|%40)[^@.%+\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*]+(?:\.[^@.%+\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*]+)*\.[^\x00-\x40\x5b-\x60\x7b-\xa0]{2,}[^\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\x60=?&*]*', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $v = $m.Value; $c = $v.TrimStart([char]39); $lead = $v.Substring(0, $v.Length - $c.Length); $c = $c.TrimEnd([char[]]@('.', [char]39)); $trail = $v.Substring($lead.Length + $c.Length); $a = $c.Replace('%40', '@'); if ($email -and $a -cmatch '^[\x00-\x7f]*$' -and [string]::Equals($a, $email, [System.StringComparison]::OrdinalIgnoreCase)) { $v } else { $lead + '<redacted-email>' + $trail } })
  # Machine names: a UNC path's machine name, a host under a local-network suffix (suffix kept), and
  # this machine's own names. Same patterns as _UNC_HOST_RE / _LAN_HOST_RE in localm/bugreport/scrub.py.
  $t = [regex]::Replace($t, '(?:(?<![^\s\x22\x27(\[{=,;|\x60<*])|(?<=[A-Za-z0-9_]{2}:))(\\{1,2}\\{1,2}(?:\?\\{1,2}UNC\\{1,2})?)(?!(?:wsl\$|wsl\.localhost|localhost)(?![A-Za-z0-9._$-]))[A-Za-z0-9][A-Za-z0-9._$-]*', '${1}<host>', $ic)
  $t = [regex]::Replace($t, '(?<![A-Za-z0-9._-])(?!(?:localm\.local|(?:host|gateway|kubernetes)\.docker\.internal)(?![A-Za-z0-9-]|\.[A-Za-z0-9]))[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(local|lan|home\.arpa|internal|localdomain)(?![A-Za-z0-9(-]|\.[A-Za-z0-9])', '<host>.${1}', $ic)
  $hostNames = @()
  try { $hostNames += [System.Net.Dns]::GetHostName() } catch { }
  if ($env:COMPUTERNAME) { $hostNames += $env:COMPUTERNAME }
  $own = @{}
  foreach ($n in $hostNames) {
    $n = "$n".Trim().TrimEnd('.')
    if ($n) { $own[$n] = 1; $own[$n.Split('.')[0]] = 1 }
  }
  # Every name is replaced where it stands as a machine name (after // or @, before :<digit>, or as the
  # value of a host-like key); a name with a non-letter also everywhere else, as _scrub_own_names does.
  $keep = @($own.Keys | Where-Object { $_.Length -ge 3 -and $_.ToLower() -ne 'localhost' } | Sort-Object -Property @{ Expression = { $_.Length }; Descending = $true }, @{ Expression = { $_ } })
  $marked = @($keep | Where-Object { $_ -cnotmatch '^[A-Za-z]+$' })
  $words = $keep
  if ($marked.Count -gt 0) {
    $alt = ($marked | ForEach-Object { [regex]::Escape($_) }) -join '|'
    $t = [regex]::Replace($t, '(?<![A-Za-z0-9_./\\-])(?:' + $alt + ')(?![A-Za-z0-9_-]|[./\\][A-Za-z0-9_])', '<host>', $ic)
  }
  if ($words.Count -gt 0) {
    $alt = '(?:' + (($words | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')'
    $end = '(?![A-Za-z0-9_-]|\.[A-Za-z0-9])'
    $t = [regex]::Replace($t, '(?:(?<=//)|(?<=@))' + $alt + $end, '<host>', $ic)
    $t = [regex]::Replace($t, '(?<![A-Za-z0-9_./\\@<-])' + $alt + '(?=:\d)', '<host>', $ic)
    $t = [regex]::Replace($t, '(?<![A-Za-z0-9_])((?:host|computer|machine|node|device|pc)(?:[ _-]?name)?["'']?[ \t]*[:=][ \t]*["'']?)' + $alt + $end, '${1}<host>', $ic)
  }
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
# Headers of the debug-log records that carry chat content: the same markers as
# _CONTENT_MARKER_RES in localm/_log_digest.py. See
# test_powershell_content_marker_matches_every_log_digest_content_sample.
$contentMarker = '(?m)(?:raw model output:\s*$|\bmemory embed_one failed for |\bjobs web tool: \S+ \{|assembled chat prompt \(\d+ message\(s\)\):\s*$|raw completion prompt \(\d+ character\(s\), no chat template\):\s*$|chat completion reply \(finish_reason=\S+\):\s*$|\bmemory (?:auto-)?consolidate (?:prompt|response):\s*$|\bmemory consolidation: (?:evaluating candidate \[\d+/\d+\]: |decision for )|\brag index: |\brag extract identity: )'

# The last $maxBytes of a file as text.
function Read-LogEnd([string]$path, [int]$maxBytes) {
  $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
  try {
    if ($fs.Length -gt $maxBytes) { [void]$fs.Seek(-$maxBytes, 'End') }
    $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
    try { return $sr.ReadToEnd() } finally { $sr.Dispose() }
  } finally { $fs.Dispose() }
}

if ($logPath -and (Test-Path -LiteralPath $logPath)) {
  try {
    $logEnd = Read-LogEnd (Resolve-Path -LiteralPath $logPath).ProviderPath 2000000
    if ($logEnd -match $contentMarker) {
      $body += "`n`n## Log`nNo log was attached: it holds chat content, which a bug report never includes."
    } else {
      $tail = (($logEnd -split "\r?\n") | Select-Object -Last 120) -join "`n"
      if ($tail.Trim()) { $body += "`n`n## Recent log tail`n$fence`n" + (Scrub $tail) + "`n$fence" }
    }
  } catch {
    $body += "`n`n## Log`nNo log was attached: the log file could not be read."
  }
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

$pair = Read-Proxy; $url = $pair[0]; $tok = $pair[1]
$netMode = 'ask'; $netWhy = ''
if ($url) { $r = Get-NetMode; $netMode = $r[0]; $netWhy = $r[1] }
if ($netMode -eq 'off') {
  $p = Save-Local
  Write-Host "  Network access is off (net_mode=off), so the report was not sent. $netOffRemedy."
  Write-Host "  The report is saved at $p; email it to $email if you prefer."
  exit 1
}
$send = $yes
if ($netMode -eq 'unknown') {
  Write-Host "  localm's Network access setting could not be read: $netWhy. Sending connects to the bug-report server."
  if (-not $interactive) {
    $p = Save-Local
    Write-Host "  No one can confirm that here, so nothing was sent. The report is saved at $p; email it to $email."
    exit 1
  }
  try { $ans = Read-Host '  Send it anyway? [y/N]' } catch { $ans = '' }
  if ($ans -notmatch '^\s*(y|yes)\s*$') {
    $p = Save-Local
    Write-Host "  Not sent. Saved at $p - email it to $email if you change your mind."
    exit 0
  }
  $send = $true
}
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
  if ($resp -and $resp.filed -eq $false) { throw 'the server did not file the report' }
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
