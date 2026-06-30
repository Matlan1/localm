# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Package a localm build zip for the self-updater (Windows). Uses `git archive` of
# HEAD, so the zip contains exactly the TRACKED source (no .venv, no data dir, no
# .git) with files at the root - which is what the updater's verify/extract expects.
#
# Usage (from anywhere in the repo):
#   1. bump VERSION (+ pyproject version), commit
#   2. tools\bugreport-proxy\make-release.ps1
#   3. git tag v<VERSION>; gh release create v<VERSION> dist\localm-<VERSION>.zip --notes "..."
$ErrorActionPreference = "Stop"
Set-Location (git rev-parse --show-toplevel)

$version = (Get-Content VERSION -Raw).Trim()
if (-not $version) { Write-Error "VERSION is empty"; exit 1 }
$tag = "v$version"

git rev-parse $tag 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Error "tag $tag already exists - bump VERSION before packaging a new build"
    exit 1
}

New-Item -ItemType Directory -Force dist | Out-Null
$out = "dist/localm-$version.zip"
git archive --format=zip -o $out HEAD
Write-Host "wrote $out"
Write-Host "next: git tag $tag; gh release create $tag $out --notes ""..."""
