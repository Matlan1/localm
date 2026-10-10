# SPDX-License-Identifier: AGPL-3.0-or-later
# Builds the Windows Task Scheduler entry that runs scripts/pin_weekly.py once a
# week (the default, -Pin weekly: every runtime localm pins), or one pin's
# scripts/pin_pipeline.py --pin <Pin> daily (-Pin llama or -Pin comfyui). Without -Register, this only PRINTS the task it would
# create - registering a standing, indefinitely-recurring, unattended process
# that pushes and merges to master is a persistent-configuration change, and
# it needs a separate, explicit go-ahead rather than being a silent last step.
#
# Usage:
#   pwsh scripts/setup_pin_pipeline_task.ps1                            # dry run: weekly, print only
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Register                  # register the weekly run
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Pin llama                 # dry run: llama only, daily
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Pin comfyui                # dry run: comfyui
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Pin comfyui -Register      # actually register it
#   pwsh scripts/setup_pin_pipeline_task.ps1 -Pin comfyui -Unregister    # remove a registered task

[CmdletBinding()]
param(
    [switch]$Register,
    [switch]$Unregister,
    [ValidateSet('weekly', 'llama', 'comfyui')]
    [string]$Pin = 'weekly',
    [string]$TaskName,
    [string]$Time = '03:00',
    [ValidateSet('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')]
    [string]$Day = 'Sunday',
    # 0 (unset) picks the per-pin default below: llama's confirm is a single
    # cpu+vulkan run (~minutes); comfyui's provision phase alone can run past
    # an hour on a cold cache (several GB of torch/ComfyUI wheels), so it
    # gets a longer budget.
    [int]$TimeLimitHours = 0
)

$ErrorActionPreference = 'Stop'

if (-not $TaskName) {
    $TaskName = "localm-pin-pipeline-$Pin"
}
if ($TimeLimitHours -le 0) {
    $TimeLimitHours = switch ($Pin) { 'weekly' { 12 } 'comfyui' { 4 } default { 2 } }
}

# scripts/setup_pin_pipeline_task.ps1 -> repo root is one level up.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$PipelineScript = if ($Pin -eq 'weekly') { Join-Path $RepoRoot 'scripts\pin_weekly.py' } else { Join-Path $RepoRoot 'scripts\pin_pipeline.py' }
$PipelineArgs = if ($Pin -eq 'weekly') { '' } else { " --pin $Pin" }
$LogDir = Join-Path $RepoRoot 'dev-notes\pin-pipeline'
# Per-pin log file: two independently-scheduled tasks sharing one file would
# interleave their output with nothing to tell the runs apart.
$LogFile = Join-Path $LogDir "task-log-$Pin.txt"

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
    throw "$PipelineScript not found"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# A direct exe+args action cannot redirect its own output, so this wraps the
# real invocation in one layer of powershell.exe to append stdout+stderr to a
# log file - the only record of an unattended run nobody is watching live.
$innerCommand = "& '$PythonExe' '$PipelineScript'$PipelineArgs *>> '$LogFile'"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$innerCommand`"" `
    -WorkingDirectory $RepoRoot
$trigger = if ($Pin -eq 'weekly') { New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $Time } else { New-ScheduledTaskTrigger -Daily -At $Time }
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $TimeLimitHours)

Write-Host "Pin:          $Pin"
Write-Host "Task name:    $TaskName"
Write-Host "Runs as:      $env:USERNAME (interactive logon - needs a real desktop session for GPU work)"
Write-Host "Schedule:     $(if ($Pin -eq 'weekly') { "every $Day at $Time" } else { "daily at $Time" })"
Write-Host "Working dir:  $RepoRoot"
Write-Host "Command:      powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `"$innerCommand`""
Write-Host "Log file:     $LogFile"
Write-Host "Max runtime:  $TimeLimitHours hours; a still-running instance skips the next trigger (IgnoreNew)"
Write-Host ''

if (-not $Register) {
    Write-Host 'Dry run only - nothing was registered. Re-run with -Register to activate it.' -ForegroundColor Yellow
    exit 0
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
