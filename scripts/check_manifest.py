#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Release-file manifest verification (NEW-RELEASE-FILEMANIFEST, see AGENTS.md).

Diffs the real ``git ls-files`` tree against ``release-manifest.toml`` and fails
closed on ANY drift, so the manifest cannot silently rot:

  - a tracked file matching NEITHER release.include nor release.exclude  (unclassified)
  - a tracked file matching BOTH                                         (ambiguous)
  - a tracked file matching a local_only pattern                         (a leak)
  - an include/exclude pattern matching NO tracked file                  (stale/renamed)
  - a local_only pattern git no longer ignores                           (drift vs .gitignore)

The point is that classifying a new top-level file/dir is a conscious act: a new
file under an already-classified tree (localm/, tests/, ...) needs no change; a new
top-level entry is flagged until it is placed in the manifest.

This runs as part of the ONE ``scripts/check_hygiene.py`` gate (the CI "Hygiene
gate" step and the --install-hook pre-commit hook both invoke check_hygiene, which
folds in :func:`check_manifest`), so drift is caught on every commit/PR without a
separate step to remember.

Importable: :func:`check_manifest` returns ``list[str]`` of problems (empty ==
clean). The classification core (:func:`classify_problems`) is pure - it takes the
tracked list and the three pattern lists and returns problems - so drift cases are
unit-tested without touching the real tree. Stdlib only (tomllib is 3.11+; the repo
targets 3.12), so it runs anywhere without installing anything.

Run:                python scripts/check_manifest.py
As a library:       from check_manifest import check_manifest
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "release-manifest.toml"


# --------------------------------------------------------------------------- #
#  pattern matching                                                           #
# --------------------------------------------------------------------------- #

def _match(pattern: str, rel: str) -> bool:
    """True if repo-relative POSIX path *rel* matches manifest *pattern*.

    - ``"foo/"``   directory prefix: ``foo`` itself or anything under it.
    - ``"*.pem"``  glob (contains ``* ? [``): fnmatch against the full path OR the
      basename, so ``*_signing_key.pem`` matches ``a/b/x_signing_key.pem`` too.
    - otherwise    an exact path match.
    """
    if pattern.endswith("/"):
        prefix = pattern[:-1]
        return rel == prefix or rel.startswith(pattern)
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)
    return rel == pattern


# --------------------------------------------------------------------------- #
#  pure classification core (unit-testable without git)                       #
# --------------------------------------------------------------------------- #

def classify_problems(tracked: list[str], include: list[str],
                      exclude: list[str], local_only: list[str]) -> list[str]:
    """The manifest's classification rules over an explicit *tracked* list.

    Pure and git-free so tests can feed synthetic trees (an unclassified file, a
    leaked local-only path, a stale pattern) and assert each is caught. Covers every
    check EXCEPT the git-check-ignore cross-check, which needs a real repo (see
    :func:`_local_ignore_problems`)."""
    problems: list[str] = []
    inc_hits = {p: 0 for p in include}
    exc_hits = {p: 0 for p in exclude}

    for rel in tracked:
        mi = [p for p in include if _match(p, rel)]
        me = [p for p in exclude if _match(p, rel)]
        for p in mi:
            inc_hits[p] += 1
        for p in me:
            exc_hits[p] += 1
        if mi and me:
            problems.append(
                f"{rel}: classified as BOTH release-include ('{mi[0]}') and "
                f"release-exclude ('{me[0]}') - fix release-manifest.toml so exactly "
                "one matches")
        elif not mi and not me:
            problems.append(
                f"{rel}: unclassified - add it (or its directory) to release.include "
                "or release.exclude in release-manifest.toml")

    # leak: a tracked file that a local_only pattern says must never be committed.
    for rel in tracked:
        ml = [p for p in local_only if _match(p, rel)]
        if ml:
            problems.append(
                f"{rel}: is tracked in git but matches local-only pattern '{ml[0]}' - "
                "it must not be committed (see .gitignore / AGENTS.md)")

    # stale: an include/exclude pattern that matches nothing tracked. This is the
    # rename tripwire - if tests/ became test/, the tests/ pattern goes to zero here
    # AND the new test/ files show up unclassified above, so both ends fire.
    for p, n in inc_hits.items():
        if n == 0:
            problems.append(
                f"release.include pattern '{p}' matches no tracked file - stale "
                "(renamed or removed?); update release-manifest.toml")
    for p, n in exc_hits.items():
        if n == 0:
            problems.append(
                f"release.exclude pattern '{p}' matches no tracked file - stale "
                "(renamed or removed?); update release-manifest.toml")
    return problems


def release_files(tracked: list[str], include: list[str], exclude: list[str]) -> list[str]:
    """The subset of *tracked* that ships in a release: matches an include pattern
    and no exclude pattern. Used by scripts/build_release.py to assemble build.zip
    from the same source of truth the checker enforces. Sorted for determinism."""
    return sorted(
        rel for rel in tracked
        if any(_match(p, rel) for p in include) and not any(_match(p, rel) for p in exclude)
    )


# --------------------------------------------------------------------------- #
#  git-backed wrapper                                                         #
# --------------------------------------------------------------------------- #

def load_manifest(path: Path = MANIFEST) -> dict:
    """Parse release-manifest.toml into ``{include, exclude, local_only}`` lists.

    Raises FileNotFoundError if the manifest is missing and ValueError if a required
    table/key is absent - a broken manifest must fail loudly, never degrade into a
    silent pass (AGENTS.md rule 5)."""
    try:
        import tomllib
    except ModuleNotFoundError as e:   # Python < 3.11; the repo targets 3.12
        raise RuntimeError(
            "release manifest check needs tomllib (Python 3.11+); this interpreter "
            f"is too old ({sys.version.split()[0]})") from e
    if not path.is_file():
        raise FileNotFoundError(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        return {
            "include": list(data["release"]["include"]["patterns"]),
            "exclude": list(data["release"]["exclude"]["patterns"]),
            "local_only": list(data["local_only"]["patterns"]),
        }
    except (KeyError, TypeError) as e:
        raise ValueError(f"release-manifest.toml is missing a required table/key: {e}") from e


def tracked_files(repo: Path = REPO) -> list[str] | None:
    """Repo-relative POSIX paths of every git-tracked file, or None when git is
    unavailable / this is not a checkout (the gate then simply does not apply)."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [line for line in out.splitlines() if line]


def _local_ignore_problems(local_only: list[str], repo: Path = REPO) -> list[str]:
    """Confirm git actually ignores each local_only pattern, via `git check-ignore`.

    Ties the manifest to git's real behavior instead of parsing .gitignore text: if a
    rule is renamed or dropped so a "local-only" path would now be committable, this
    fires. Probes a representative path per pattern and checks it is ignored."""
    probes = {p: _probe_path(p) for p in local_only}
    # NUL-delimited I/O (-z), binary, NOT text mode: on Windows text mode translates
    # the separator \n to \r\n, so git would read each probe with a trailing CR as
    # part of the pathname - directory patterns still prefix-match but exact-file
    # patterns silently do not. NUL separators sidestep newline translation entirely.
    stdin = "".join(probe + "\0" for probe in probes.values()).encode("utf-8")
    try:
        res = subprocess.run(["git", "check-ignore", "-z", "--stdin"], cwd=repo,
                            input=stdin, capture_output=True)
    except FileNotFoundError:
        return []   # no git -> the gate does not apply (tracked_files() already None)
    if res.returncode not in (0, 1):   # 0 = some ignored, 1 = none ignored, else error
        return [f"git check-ignore failed (exit {res.returncode}): "
                f"{res.stderr.decode('utf-8', 'replace').strip() or 'unknown error'}"]
    ignored = {x for x in res.stdout.decode("utf-8", "replace").split("\0") if x}
    problems = []
    for pat, probe in probes.items():
        if probe not in ignored:
            problems.append(
                f"local-only pattern '{pat}' is NOT ignored by git (probed '{probe}') "
                "- release-manifest.toml and .gitignore have drifted; restore the "
                "ignore rule or drop the pattern")
    return problems


def _probe_path(pattern: str) -> str:
    """A concrete path that should match *pattern*, for `git check-ignore`.
    Directory -> a child; glob -> substitute a token for wildcards; else the path."""
    if pattern.endswith("/"):
        return pattern + "__manifest_probe__"
    if any(c in pattern for c in "*?["):
        out = []
        for ch in pattern:
            if ch in "*?":
                out.append("x")
            elif ch == "[":
                out.append("x")   # crude but fine for the simple globs we use
            elif ch == "]":
                continue
            else:
                out.append(ch)
        return "".join(out)
    return pattern


def check_manifest(repo: Path = REPO, manifest: Path = MANIFEST) -> list[str]:
    """Run the full manifest gate against the real repo. Returns problems ([]==clean).

    Returns [] (the gate does not apply) when git is unavailable / this is not a
    checkout; that is the only silent case and it is documented, not a masked
    failure. A missing or malformed manifest still raises (fail loud)."""
    man = load_manifest(manifest)
    tracked = tracked_files(repo)
    if tracked is None:
        return []
    problems = classify_problems(tracked, man["include"], man["exclude"], man["local_only"])
    problems.extend(_local_ignore_problems(man["local_only"], repo))
    return problems


def main(argv: list[str]) -> int:
    try:
        problems = check_manifest()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Release manifest check could not run: {e}", file=sys.stderr)
        return 1
    if problems:
        print("Release manifest check FAILED (see release-manifest.toml / AGENTS.md):\n",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} issue(s).", file=sys.stderr)
        return 1
    print("Release manifest check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
