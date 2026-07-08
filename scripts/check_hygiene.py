#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repository hygiene check (see AGENTS.md).

Scans tracked files and fails on:
  1. The em-dash (U+2014) or en-dash (U+2013) in any text file.
  2. Personal or machine-specific disclosure: a local username used as a path
     component, a leaked secret (token / key / private key), or a known private
     external path. NOTE: the maintainer's contact email is intentionally
     published for bug reports (see localm/bugreport.py) and is NOT flagged.
  3. An absolute or machine-specific path used in code/config (not docs), which
     a default must never assume.
  4. A CHANGELOG.md that is not append-only: a shipped entry line removed or
     rewritten (vs the published-record baseline) instead of new entries added on
     top. The changelog is the permanent public record of what shipped (AGENTS.md).

It also runs the release-file manifest gate (scripts/check_manifest.py,
NEW-RELEASE-FILEMANIFEST): every tracked file must be classified release-include
or release-exclude, nothing local-only may be committed, and no manifest pattern
may go stale. Folding it in here means the ONE CI "Hygiene gate" step and the
--install-hook pre-commit hook cover both without a separate step to remember.

Run before committing:   python scripts/check_hygiene.py
Install as a git hook:    python scripts/check_hygiene.py --install-hook

A line that genuinely needs an absolute-looking example (help text, a doc
sample) can carry a trailing  hygiene-ok  marker to be skipped by check 3.
Checks 1 and 2 have no escape: those are never acceptable.

Stdlib only, so it runs in any environment without installing anything.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- check 1: dashes -------------------------------------------------------
# Em-dash and en-dash, referenced by codepoint so this file stays clean.
# Plain ASCII hyphen-minus only.
_EM_DASH = chr(0x2014)
_EN_DASH = chr(0x2013)
_DASHES = (_EM_DASH, _EN_DASH)

# ---- check 2: disclosure (no escape) ---------------------------------------
# The maintainer username is assembled from fragments so this file does not
# itself contain the plaintext identifier it forbids in path components (rule 2).
# The maintainer's CONTACT EMAIL is deliberately NOT scanned for: the maintainer
# opted to publish it for bug reports (localm/bugreport.py). What stays forbidden
# is a local username used as a filesystem path (a machine-specific bug) and any
# real secret (API token, access key, private key).
_MAINT_USER = "Mat" + "lan"
_DISCLOSURE = [
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]" + _MAINT_USER + r"\b", re.I),  # user dir
    re.compile(r"[\\/]Users[\\/]" + _MAINT_USER + r"\b"),                 # unix-style
    # actual secrets
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),                 # classic GitHub PAT
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),         # fine-grained GitHub PAT
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]

# ---- check 3: absolute paths in code/config (escapable) --------------------
_CODE_EXTS = {".py", ".pyw", ".bat", ".cmd", ".ps1", ".sh", ".toml",
              ".json", ".cfg", ".ini", ".html"}
_ABS_PATH = re.compile(
    r"""(["'(]|\br['"])\s*          # opening quote / r-string
        (?:[A-Za-z]:[\\/]           # Windows drive path
         | /home/ | /Users/ | /mnt/[a-z]/ | /opt/[A-Za-z] )""",
    re.X,
)

# directories never scanned
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "vendor",
              "lib"}  # runtime binaries live in lib/ and are gitignored anyway

_BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
                ".dll", ".exe", ".so", ".dylib", ".bin", ".gguf", ".woff",
                ".woff2", ".ttf"}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    files = []
    for rel in out.splitlines():
        p = REPO / rel
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in _BINARY_EXTS:
            continue
        files.append(p)
    return files


def _scan(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []   # binary or unreadable: not our concern here
    rel = path.relative_to(REPO).as_posix()
    # Tests legitimately use synthetic absolute paths as fixtures (fake drive
    # letters, nonexistent dirs, file URLs). The dash and disclosure checks
    # still apply to them; only the absolute-path heuristic is skipped.
    is_test = rel.startswith("tests/") or "/test_" in "/" + rel or Path(rel).name.startswith("test_")
    is_code = path.suffix.lower() in _CODE_EXTS and not is_test
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        for d in _DASHES:
            if d in line:
                name = "em-dash" if d == _EM_DASH else "en-dash"
                problems.append(f"{rel}:{i}: {name} (U+{ord(d):04X}) not allowed")
        for rx in _DISCLOSURE:
            if rx.search(line):
                problems.append(f"{rel}:{i}: disclosure: {rx.pattern}")
        if is_code and "hygiene-ok" not in line and _ABS_PATH.search(line):
            problems.append(f"{rel}:{i}: absolute/machine path in code "
                            "(use a relative path or user config, or mark "
                            "hygiene-ok if it is a documented example)")
    return problems


# ---- check 4: CHANGELOG is append-only -------------------------------------
# The release changelog is the permanent public record of what shipped: each
# release ADDS its section on top; existing entries are never deleted or rewritten
# (typo/formatting fixes aside - see AGENTS.md). Enforced by diffing the working
# CHANGELOG against the published-record baseline (the merge-base with
# origin/master, else the last commit) and failing if any shipped ENTRY line
# disappeared. Markdown HEADERS ("# ...") and link-reference definitions
# ("[label]: url") are exempt: cutting a release legitimately renames the Unreleased
# header to a version and rewrites the compare link without touching an entry.
# Compared as a multiset, so MOVING entries under a new version header is fine -
# only an actual deletion or rewrite of an entry line is caught.
_CHANGELOG = "CHANGELOG.md"
_CHANGELOG_LINKREF = re.compile(r"\[[^\]]+\]:\s")


def _changelog_protected_lines(text: str) -> list[str]:
    """Changelog lines whose loss would rewrite history: non-blank lines that are
    not a markdown header and not a link-reference definition. rstrip()'d so a
    CRLF/LF or trailing-space difference is not mistaken for a real change."""
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or _CHANGELOG_LINKREF.match(stripped):
            continue
        out.append(line)
    return out


def _changelog_removed_lines(old_text: str, new_text: str) -> list[str]:
    """Protected content lines present in *old_text* but no longer present (with
    multiplicity) in *new_text*: shipped changelog entries that were DELETED or
    REWRITTEN rather than left intact with new entries added above them."""
    from collections import Counter
    old = Counter(_changelog_protected_lines(old_text))
    new = Counter(_changelog_protected_lines(new_text))
    removed = []
    for line, count in old.items():
        removed.extend([line] * (count - new.get(line, 0)))
    return removed


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """Run a git subcommand under REPO; None if git is unavailable at all."""
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return None


def _changelog_baseline_ref() -> str | None:
    """The commit whose CHANGELOG the working tree must not delete entries from.

    Prefer the merge-base with ``origin/master`` - the published record at THIS
    branch's point. Comparing against it means the guard bites in BOTH places:
    pre-commit (working tree vs the base) AND in CI on a clean checkout (the
    committed HEAD vs the base, so a deletion sneaked past the hook / committed via
    a web edit is still caught). It also never false-positives on new releases that
    landed on master AFTER this branch (those are not in the merge-base). Falls back
    to HEAD when origin/master is unavailable (offline, a fresh clone), which is the
    plain "vs the last commit" pre-commit check. None => no git at all."""
    mb = _git("merge-base", "HEAD", "origin/master")
    if mb is None:
        return None
    if mb.returncode == 0 and mb.stdout.strip():
        return mb.stdout.strip()
    head = _git("rev-parse", "HEAD")            # no origin/master: last commit
    return head.stdout.strip() if head and head.returncode == 0 else None


def _changelog_append_only() -> list[str]:
    """CHANGELOG.md must be APPEND-ONLY (AGENTS.md): report every shipped entry line
    removed or rewritten relative to the baseline (see _changelog_baseline_ref). A
    CHANGELOG not yet in the baseline (never committed) or a repo without git has
    nothing to compare against, so it passes - the guard catches deletions from an
    established record, it does not block the first commit."""
    ref = _changelog_baseline_ref()
    if ref is None:
        return []                       # no git available: nothing to diff against
    base = _git("show", f"{ref}:{_CHANGELOG}")
    if base is None or base.returncode != 0:
        return []                       # CHANGELOG not in the baseline yet: no record
    try:
        working = (REPO / _CHANGELOG).read_text(encoding="utf-8")
    except OSError:
        working = ""                    # deleted from the tree: every entry is gone
    removed = _changelog_removed_lines(base.stdout, working)
    if not removed:
        return []
    shown = "; ".join(repr(x.strip()) for x in removed[:4])
    more = f" (+{len(removed) - 4} more)" if len(removed) > 4 else ""
    return [f"{_CHANGELOG}: append-only violation - {len(removed)} shipped entry "
            "line(s) removed or rewritten. The changelog is the permanent public "
            "record: add new entries ABOVE, never delete or rewrite existing ones. "
            f"Removed: {shown}{more}"]


def _install_hook() -> int:
    hook = REPO / ".git" / "hooks" / "pre-commit"
    if not hook.parent.is_dir():
        print("No .git/hooks directory found.", file=sys.stderr)
        return 1
    hook.write_text(
        "#!/bin/sh\n"
        'exec python "$(git rev-parse --show-toplevel)/scripts/check_hygiene.py"\n',
        encoding="utf-8",
    )
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    print(f"Installed pre-commit hook at {hook}")
    return 0


def _manifest_problems() -> list[str]:
    """The release-file manifest gate's findings (NEW-RELEASE-FILEMANIFEST), run as
    part of this one hygiene pass. check_manifest lives beside this file; a missing or
    malformed manifest is reported as a problem, never a silent skip (rule 5)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import check_manifest
        return check_manifest.check_manifest()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return [f"release manifest check could not run: {e}"]


def main(argv: list[str]) -> int:
    if "--install-hook" in argv:
        return _install_hook()
    problems: list[str] = []
    for f in _tracked_files():
        problems.extend(_scan(f))
    problems.extend(_changelog_append_only())
    manifest = _manifest_problems()
    if problems or manifest:
        if problems:
            print("Hygiene check FAILED (see AGENTS.md):\n", file=sys.stderr)
            for p in problems:
                print("  " + p, file=sys.stderr)
            print(f"\n{len(problems)} hygiene issue(s).", file=sys.stderr)
        if manifest:
            print("\nRelease manifest check FAILED (see release-manifest.toml):\n",
                  file=sys.stderr)
            for p in manifest:
                print("  " + p, file=sys.stderr)
            print(f"\n{len(manifest)} manifest issue(s).", file=sys.stderr)
        return 1
    print("Hygiene check passed (content + release manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
