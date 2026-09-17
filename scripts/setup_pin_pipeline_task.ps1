# SPDX-License-Identifier: AGPL-3.0-or-later
# Builds the Windows Task Scheduler entry that runs scripts/pin_pipeline.py
# --pin llama daily. Without -Register, this only PRINTS the task it would
# create - registering a standing, indefinitely-recurring, unattended process
# that pushes and merges to master is a persistent-configuration change, and
# it needs a separate, explicit go-ahead rather than being a silent last step.
#
# Usage:
#   pwsh scripts/setup_pin_pipeline_task.ps1                 # dry run: print only
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Register        # actually register it
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Unregister      # remove a registered task

[CmdletBinding()]
param(
    [switch]$Register,
    [switch]$Unregister,
    [string]$TaskName = 'localm-pin-pipeline-llama',
    [string]$Time = '03:00'
)

$ErrorActionPreference = 'Stop'

# scripts/setup_pin_pipeline_task.ps1 -> repo root is one level up.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$PipelineScript = Join-Path $RepoRoot 'scripts\pin_pipeline.py'
$LogDir = Join-Path $RepoRoot 'dev-notes\pin-pipeline'
$LogFile = Join-Path $LogDir 'task-log.txt'

if ($Unregister) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "No scheduled task named '$TaskName' is registered; nothing to remove."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found at $PythonExe - is this the localm repo's own .venv (run setup.sh/setup.bat first)?"
}
if (-not (Test-Path -LiteralPath $PipelineScript)) {
    throw "scripts/pin_pipeline.py not found at $PipelineScript"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# A direct exe+args action cannot redirect its own output, so this wraps the
# real invocation in one layer of powershell.exe to append stdout+stderr to a
# log file - the only record of an unattended run nobody is watching live.
$innerCommand = "& '$PythonExe' '$PipelineScript' --pin llama *>> '$LogFile'"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$innerCommand`"" `
    -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Write-Host "Task name:    $TaskName"
Write-Host "Runs as:      $env:USERNAME (interactive logon - needs a real desktop session for GPU work)"
Write-Host "Schedule:     daily at $Time"
Write-Host "Working dir:  $RepoRoot"
Write-Host "Command:      powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `"$innerCommand`""
Write-Host "Log file:     $LogFile"
Write-Host "Max runtime:  2 hours; a still-running instance skips the next trigger (IgnoreNew)"
Write-Host ''

if (-not $Register) {
    Write-Host 'Dry run only - nothing was registered. Re-run with -Register to activate it.' -ForegroundColor Yellow
    exit 0
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
