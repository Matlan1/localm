#!/usr/bin/env sh
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Package a localm build zip for the self-updater. Uses `git archive` of HEAD, so the
# zip contains exactly the TRACKED source (no .venv, no data dir, no .git) with files
# at the root - which is what the updater's verify/extract expects.
#
# Usage (from anywhere in the repo):
#   1. bump VERSION (+ pyproject version), commit
#   2. tools/bugreport-proxy/make-release.sh
#   3. git tag v$(cat VERSION) && gh release create v$(cat VERSION) dist/localm-*.zip --notes "..."
set -e
cd "$(git rev-parse --show-toplevel)"

version="$(tr -d ' \t\r\n' < VERSION)"
[ -n "$version" ] || { echo "VERSION is empty" >&2; exit 1; }
tag="v$version"

if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "tag $tag already exists - bump VERSION before packaging a new build" >&2
  exit 1
fi

mkdir -p dist
out="dist/localm-$version.zip"
git archive --format=zip -o "$out" HEAD
echo "wrote $out"
echo "next: git tag $tag && gh release create $tag $out --notes \"...\""
