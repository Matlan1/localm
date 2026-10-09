#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Decides what the docker workflow does. Run from the repository root.
#
# Environment: EVENT_NAME, REF, RELEASE_TAG, PRERELEASE, DRY_RUN, GITHUB_REPOSITORY,
# GITHUB_OUTPUT (a file; version, push and floating are appended to it) and, when a
# push is decided, GH_TOKEN for `gh api`.
#
#   version   the contents of VERSION
#   push      true for a published release, and for a dispatch with dry_run false
#             that runs on the tag v<version>
#   floating  true when the pushed version is the repository's latest release
#             (a pre-release never is): then latest, cpu and vulkan move too
set -eu

fail() {
  echo "::error::$1"
  exit 1
}

version="$(tr -d '[:space:]' < VERSION)"
case "$version" in
  ""|*[!0-9A-Za-z._-]*) fail "VERSION is not usable as an image tag: '$version'" ;;
esac

push=false
floating=false

is_latest_release() {
  latest="$(gh api "repos/${GITHUB_REPOSITORY}/releases/latest" --jq .tag_name)" \
    || fail "could not read the latest release of ${GITHUB_REPOSITORY}"
  [ "$latest" = "v$version" ]
}

if [ "$EVENT_NAME" = "release" ]; then
  if [ "${RELEASE_TAG#v}" != "$version" ]; then
    fail "release tag '$RELEASE_TAG' does not match VERSION '$version'"
  fi
  push=true
  if [ "$PRERELEASE" != "true" ] && is_latest_release; then
    floating=true
  fi
elif [ "$EVENT_NAME" = "workflow_dispatch" ] && [ "$DRY_RUN" = "false" ]; then
  if [ "$REF" != "refs/tags/v$version" ]; then
    fail "publishing from a dispatch requires the release tag ref refs/tags/v$version, got '$REF'"
  fi
  push=true
  if is_latest_release; then
    floating=true
  fi
fi

{
  echo "version=$version"
  echo "push=$push"
  echo "floating=$floating"
} >> "$GITHUB_OUTPUT"
