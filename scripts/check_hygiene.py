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
     The [Unreleased] draft stays exempt from that hard gate, but a draft line
     that existed at the baseline and is gone from the working copy is reported
     as a WARNING (check 4b below): a textually clean rebase has silently dropped
     a sibling branch's draft bullet before. --strict (or LOCALM_HYGIENE_STRICT=1)
     escalates warnings to failures for CI-style use.
  5. A raw call to a single-resource accessor from outside its designated
     aggregate-capacity wrapper (see _RAW_ACCESSOR_GUARDS below). When a feature's
     whole value is "combine capacity across N resources" (multi-GPU VRAM split is
     the first case; the same shape applies to any future multi-disk,
     multi-model-instance, or multi-connection-pool feature), every "does this
     fit" decision must go through the wrapper - not just the one call site a PR
     happened to update. This is deliberately an ENFORCED check, not a written
     review note: a note relies on a human remembering to re-grep every call site
     next time, which is exactly the discipline that already failed once (see
     dev-notes/gpu-split-capacity-fix/ for the incident this check was written
     for - vram_info() was single-GPU-only and 8 call sites read it as if it
     were the aggregate ceiling before discover.vram_capacity() existed).
  6. sw.js's SHELL precache array names a file that doesn't exist, or misses an
     app/*.js or pages/*.js module it promises to precache; or sw.js's CACHE
     constant line is no longer in the shape localm/plugins/gui/web.py's GET
     /sw.js route expects to substitute a computed value into on every request
     (that route is what actually keeps an installed PWA from serving stale
     assets now - see check_hygiene's own "check 6" block comment for why a
     hand-bumped version string was retired in favor of it).
  7. A module-level import CYCLE between top-level units under localm/ (e.g.
     inference <-> plugins). Acyclicity is checked instead of a hand-written tier
     map because a map is a maintained artifact encoding opinions - two readers
     auditing the same tree derived different "upward edge" counts purely from
     inventing different maps - whereas a cycle is a property of the graph. The
     two shapes any layer map has to carve out are legal here BY CONSTRUCTION: an
     entry point (__main__ -> cli) is a source node, and peers (image_gen /
     music_gen / video_gen -> media) are unordered, so neither can form a cycle
     and neither needs an allowlist. Function-local imports are ignored: deferring
     an import is the standard, deliberate way to break a cycle in Python and this
     codebase does it on purpose, so only eager module-level edges count.

It also runs the release-file manifest gate (scripts/check_manifest.py,
NEW-RELEASE-FILEMANIFEST): every tracked file must be classified release-include
or release-exclude, nothing local-only may be committed, and no manifest pattern
may go stale. Folding it in here means the ONE CI "Hygiene gate" step and the
--install-hook pre-commit hook cover both without a separate step to remember -
WHEN scripts/check_manifest.py is present. That file is itself gitignored
(AGENTS.md rule 6), so it is absent from a fresh CI checkout and from most
external contributors' clones by design; when it cannot be imported this reports
a WARNING (escalated to a failure only under --strict / LOCALM_HYGIENE_STRICT=1),
never a silent pass - see _release_manifest_gate().

Run before committing:   python scripts/check_hygiene.py
Install as a git hook:    python scripts/check_hygiene.py --install-hook
Warnings as failures:     python scripts/check_hygiene.py --strict
                          (equivalently: set LOCALM_HYGIENE_STRICT=1)

A line that genuinely needs an absolute-looking example (help text, a doc
sample) can carry a trailing  hygiene-ok  marker to be skipped by check 3.
Checks 1 and 2 have no escape: those are never acceptable.

Stdlib only, so it runs in any environment without installing anything.
"""

from __future__ import annotations

import ast
import os
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
              ".json", ".cfg", ".ini", ".html", ".js", ".mjs", ".yaml", ".yml"}
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
        # encoding="utf-8" for the same reason as _git: git emits paths as
        # UTF-8, but a bare text=True decodes with the locale codepage (cp1252
        # on Windows), which would mangle - or, on a byte outside the codepage,
        # raise and abort the whole scan for - a non-ASCII tracked filename.
        out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout
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
    # The frontend test suites don't follow the Python tests/test_*.py
    # convention: they live under tests-js/ and tests-e2e/ and are named
    # *.test.mjs / *.spec.mjs, so those conventions are recognized too now
    # that .js/.mjs are in _CODE_EXTS.
    fname = Path(rel).name
    is_test = (rel.startswith("tests/") or rel.startswith("tests-js/")
               or rel.startswith("tests-e2e/") or "/test_" in "/" + rel
               or fname.startswith("test_")
               or fname.endswith((".test.js", ".test.mjs", ".spec.js", ".spec.mjs")))
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
# "Published" means RELEASED, not merely "carries a version header": the pending
# release (VERSION's own section, before its tag exists) is still a cuttable draft -
# see _pending_release_version. Every other version section is frozen unconditionally.
# The release changelog is the permanent public record of what shipped. Only the
# PUBLISHED (versioned "## [x.y.z]") sections are frozen: a release ADDS its section
# on top and its entries (INCLUDING its own version header and any "### " subsection
# headers within it, e.g. "### Added") are then never deleted or rewritten (typo/
# formatting fixes aside - see AGENTS.md). The "## [Unreleased]" draft's own header
# (and any intro text before the first version header) is the in-progress record and
# is FREELY rewritable until it is cut into a version. Enforced by diffing the working
# CHANGELOG against the published-record baseline (the merge-base with origin/master,
# else the last commit) and failing if any PUBLISHED entry line disappeared. Only the
# "## [Unreleased]" header itself and link-reference definitions ("[label]: url") are
# exempt: cutting a release legitimately renames the Unreleased header to a version
# and rewrites the compare link. Compared as a multiset, so MOVING an entry from
# [Unreleased] under a new version header is fine - only a deletion or rewrite of an
# already-PUBLISHED entry line (body OR header) is caught.
_CHANGELOG = "CHANGELOG.md"
_CHANGELOG_LINKREF = re.compile(r"\[[^\]]+\]:\s")
# An H2 section header opening a version section, e.g. "## [0.1.1] - date"; group 1 is
# the version. "## [Unreleased]" does not match (no leading digit), so its section
# stays editable.
_CHANGELOG_VERSION_HEADER = re.compile(r"^##\s+\[(\d[^\]]*)\]")


def _pending_release_version() -> str | None:
    """The one version whose changelog section is still an editable DRAFT: the version
    the VERSION file names, when no release tag exists for it yet. None => freeze every
    version section (the old, unconditional behavior).

    A ``## [x.y.z]`` header alone does NOT mean x.y.z shipped. The release ritual bumps
    VERSION and cuts the section BEFORE the tag and GitHub release exist, so between the
    cut and the publish the section is a versioned-but-unshipped draft. Freezing it
    protects nothing (nobody can have downloaded a release that does not exist) while
    blocking the legitimate final cut, e.g. re-dating the section on the day it actually
    ships, or folding newer [Unreleased] work into a prep that was never published.

    Deliberately narrow, so a MISSING tag can never unfreeze real history:
      - a section whose version != VERSION is ALWAYS frozen, even in a clone with no
        tags at all, and
      - the VERSION-matching section re-freezes the moment its tag exists.
    So at most ONE section - the pending release - is ever editable, and every genuinely
    published section stays frozen regardless of local tag state. Any uncertainty (no
    VERSION file, git unavailable, git error) fails SAFE by freezing everything."""
    try:
        version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return None                     # no VERSION file: freeze every version section
    if not version:
        return None
    tag = _git("tag", "--list", f"v{version}")
    if tag is None or tag.returncode != 0:
        return None                     # cannot tell: fail safe, freeze everything
    if tag.stdout.strip():
        return None                     # already tagged => really shipped => frozen
    return version


def _changelog_protected_lines(text: str, pending_version: str | None = None) -> list[str]:
    """Lines of the PUBLISHED (versioned) changelog sections whose loss would rewrite
    history: the ``## [x.y.z]`` header line itself, plus every non-blank, non-link-
    reference line sitting under it - INCLUDING a ``### Added``-style subsection
    header, not just its bullet entries. Lines under ``## [Unreleased]`` (or before
    the first version header) are the in-progress draft and are NOT protected - they
    may be rewritten freely until the release is cut (AGENTS.md). rstrip()'d so a
    CRLF/LF or trailing-space difference is not mistaken for a real change.

    Both the version header line AND subsection headers within a published section
    are protected, not just bullet entries: a version header carries the version
    number and ship date, and a subsection header carries WHICH CATEGORY an entry
    shipped under (e.g. distinguishing "Added" from "Removed" for the same bullet
    text) - silently rewriting either is exactly the kind of history rewrite this
    guard exists to catch, the same as editing a bullet's wording.

    *pending_version* (see _pending_release_version) names the ONE version that is cut
    but not yet released; its section is still a draft and is NOT protected. None (the
    default) protects every version section, so the pure-function callers and tests keep
    the original semantics."""
    out = []
    published = False   # intro + [Unreleased] (before the first version header) are editable
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        # An H2 section header ("## ...") switches zones: a versioned header opens the
        # protected published record; [Unreleased] (or any other H2) closes it. Once
        # published, a DEEPER header ("### Added") does NOT change the zone, but is
        # still protected content (handled by the generic append below).
        if stripped.startswith("## "):
            m = _CHANGELOG_VERSION_HEADER.match(stripped)
            # The pending (cut but unreleased) version's section is still a draft.
            published = bool(m) and not (
                pending_version is not None and m.group(1).strip() == pending_version
            )
            if published:
                out.append(line)   # the header itself is part of the published record
            continue
        if not published:
            continue
        if not stripped or _CHANGELOG_LINKREF.match(stripped):
            continue
        out.append(line)
    return out


def _changelog_removed_lines(old_text: str, new_text: str,
                             pending_version: str | None = None) -> list[str]:
    """Protected content lines present in *old_text* but no longer present (with
    multiplicity) in *new_text*: shipped changelog entries that were DELETED or
    REWRITTEN rather than left intact with new entries added above them.

    *pending_version* exempts the cut-but-unreleased section (see
    _changelog_protected_lines); None protects every version section."""
    from collections import Counter
    old = Counter(_changelog_protected_lines(old_text, pending_version))
    new = Counter(_changelog_protected_lines(new_text, pending_version))
    removed = []
    for line, count in old.items():
        removed.extend([line] * (count - new.get(line, 0)))
    return removed


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """Run a git subcommand under REPO; None if git is unavailable at all.

    Decode git's output as UTF-8 explicitly, NOT the platform default: with a
    bare text=True, Python decodes with locale.getpreferredencoding() (cp1252 on
    Windows), which mangles UTF-8 file content from ``git show`` into mojibake.
    The working tree is read with encoding="utf-8" (see the CHANGELOG check), so
    the baseline read here must match it or identical non-ASCII lines (an emoji,
    an accented char) look "changed" and trip a false append-only violation.
    """
    try:
        return subprocess.run(["git", *args], cwd=REPO,
                              capture_output=True, text=True, encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


# The baseline sha, resolved AT MOST ONCE per process and per REPO (see
# _changelog_baseline_ref). Keyed on REPO so a caller that repoints REPO (the tests
# do, one throwaway repo per test) still resolves afresh for the new tree.
_BASELINE_REF_CACHE: dict[Path, str | None] = {}


def _changelog_baseline_ref() -> str | None:
    """The commit whose CHANGELOG the working tree must not delete entries from.

    Prefer the merge-base with ``origin/master`` - the published record at THIS
    branch's point. Comparing against it means the guard bites in BOTH places:
    pre-commit (working tree vs the base) AND in CI on a clean checkout (the
    committed HEAD vs the base, so a deletion sneaked past the hook / committed via
    a web edit is still caught). It also never false-positives on new releases that
    landed on master AFTER this branch (those are not in the merge-base). Falls back
    to HEAD when origin/master is unavailable (offline, a fresh clone), which is the
    plain "vs the last commit" pre-commit check. None => no git at all.

    PINNED (resolved once per process): three separate checks call this - the
    append-only gate, the [Unreleased] warn-only checks, and the service-worker
    cache-bump gate - and each call otherwise shells out to `git merge-base HEAD
    origin/master` again. ``origin/master`` is a MOVING ref: worktrees share one
    ref store, so a sibling session's fetch or a merge landing mid-run can advance
    it between two of those calls, and the checks would then silently compare
    against DIFFERENT baselines within a single run. That is not hypothetical - the
    manual version of this check reported a phantom missing bullet exactly that way
    (guard passed, a sibling lane merged seconds later, a later command re-read the
    ref). Resolving to an immutable sha once and reusing it makes every check in a
    run agree on one baseline; the tell that this matters is two detectors
    disagreeing, which must always mean "one is stale", never "restore the
    difference"."""
    if REPO in _BASELINE_REF_CACHE:
        return _BASELINE_REF_CACHE[REPO]
    mb = _git("merge-base", "HEAD", "origin/master")
    if mb is None:
        ref = None
    elif mb.returncode == 0 and mb.stdout.strip():
        ref = mb.stdout.strip()
    else:
        head = _git("rev-parse", "HEAD")        # no origin/master: last commit
        ref = head.stdout.strip() if head and head.returncode == 0 else None
    _BASELINE_REF_CACHE[REPO] = ref
    return ref


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
    removed = _changelog_removed_lines(base.stdout, working, _pending_release_version())
    if not removed:
        return []
    shown = "; ".join(repr(x.strip()) for x in removed[:4])
    more = f" (+{len(removed) - 4} more)" if len(removed) > 4 else ""
    return [f"{_CHANGELOG}: append-only violation - {len(removed)} shipped entry "
            "line(s) removed or rewritten. The changelog is the permanent public "
            "record: add new entries ABOVE, never delete or rewrite existing ones. "
            f"Removed: {shown}{more}"]


# ---- check 4b: [Unreleased] draft corruption (warn-only) --------------------
# The append-only gate above deliberately exempts the [Unreleased] draft: it is
# freely rewritable until cut. That exemption has a blind spot: when parallel
# branches all add draft bullets, a sibling branch's bullet can disappear around
# a rebase while every mechanical check still reports clean. It happened twice in
# one 12-PR fan-out day (2026-07-22): a landed PR's entry simply vanished from
# the release notes. So two warnings, not one, matching the two failure shapes
# actually observed that day:
#   DROP      - a baseline [Unreleased] content line missing from the working copy.
#   DUPLICATE - a draft bullet that now appears more than once. This is the
#               artifact of the REMEDY going wrong: the naive drop check (diff
#               against the moving origin/master ref) false-positives on every
#               sibling bullet merged after your branch point, and "restore
#               anything you did not author" then re-imports a bullet that was
#               never lost. Two sessions did exactly that.
# Both are WARNINGS, never failures by default: rewording or deleting YOUR OWN
# draft lines is legitimate and must stay frictionless. --strict /
# LOCALM_HYGIENE_STRICT=1 escalates them where a hard gate is wanted.
#
# On the CAUSE, stated carefully because a wrong story sends people hunting the
# wrong thing: "it happens on every rebase that replays an [Unreleased]
# insertion" was FALSIFIED by direct comparison (a matching branch showed zero
# drops). What is actually evidenced is a CONFLICTED rebase resolved
# bulk-take-mine (reproduced end to end while building this check: the sibling
# bullet is gone, the tree is clean, and the hard gate above returns nothing);
# a squash or amend across the section is suspected but unproven. So the warning
# text below points at the resolution step, not at rebases in general.
#
# Design decisions, recorded so they are not re-litigated:
#   - Matching is EXACT (rstrip'd) line equality, so a REWORDED draft line warns
#     too. Accepted, documented cost: a similarity heuristic that suppressed
#     near-matches could suppress exactly the incident case (a sibling's bullet
#     eaten while similar sibling bullets remain). For a warn-only check a false
#     positive costs one glance; a false negative defeats the backstop.
#   - The working side is the WHOLE file, not just its [Unreleased] section, so
#     cutting a release (which MOVES the draft lines under a new version header)
#     does not read as a mass drop.
#   - Occurrences are counted on both sides (Counter), so a draft line whose
#     text also appears in a published section is still reported when the DRAFT
#     copy is the one deleted - the surviving published copy cannot satisfy its
#     count. The converse is the accepted cost of counting: when a line is
#     duplicated EXACTLY across sections, deleting the OTHER copy warns as well,
#     because counting alone cannot tell two identical lines apart. Warning is
#     the safe direction (the alternative attribution silently misses the real
#     incident case), and such a deletion is either already a hard append-only
#     FAILURE or an edit inside the pending cut section, so the extra line is
#     noise on a run that is already asking for a human look.
#   - Only bullet/continuation CONTENT lines are watched. Headers ("### Added"),
#     blank lines and link-reference definitions are scaffolding a draft may
#     freely reorganize; warning on those would be noise that trains people to
#     ignore the one warning that matters.
#   - Scope is [Unreleased] only, not the pending cut-but-untagged section
#     (_pending_release_version): parallel branches land bullets in
#     [Unreleased], while a cut section is edited by exactly one release
#     ritual, so the rebase-collision hazard this backstops does not arise
#     there.
#   - The baseline is the MERGE-BASE with origin/master (_changelog_baseline_ref),
#     never the origin/master ref itself. This is load-bearing, not incidental:
#     worktrees share one ref store, so any sibling session's fetch advances
#     origin/master under you, and a tip-relative comparison then reports every
#     bullet merged after your branch point as a line YOU dropped. That false
#     positive is what produced the duplicate-import artifact above.
#   - DUPLICATES are reported only when NEW relative to the baseline. A duplicate
#     already present at the baseline is master's defect, not this branch's, and
#     nagging every session about it on every run is how a warning gets trained
#     into background noise.
_CHANGELOG_UNRELEASED_HEADER = re.compile(r"^##\s+\[unreleased\]", re.I)


def _changelog_unreleased_lines(text: str) -> list[str]:
    """Content lines of the ``## [Unreleased]`` draft section: bullets and their
    wrapped continuation lines, rstrip()'d. Headers, blank lines and
    link-reference definitions are excluded (see the block comment above)."""
    out = []
    in_draft = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("## "):
            in_draft = bool(_CHANGELOG_UNRELEASED_HEADER.match(stripped))
            continue
        if not in_draft:
            continue
        if not stripped or stripped.startswith("#") or _CHANGELOG_LINKREF.match(stripped):
            continue
        out.append(line)
    return out


def _changelog_dropped_unreleased_lines(old_text: str, new_text: str) -> list[str]:
    """Baseline [Unreleased] content lines missing (with multiplicity) from
    *new_text* AS A WHOLE, in baseline order. Whole-file counting on both sides
    is what keeps a release cut (a move) clean while still catching a deleted
    draft copy of a line whose text is duplicated in a published section."""
    from collections import Counter
    draft = Counter(_changelog_unreleased_lines(old_text))
    if not draft:
        return []
    old_all = Counter(line.rstrip() for line in old_text.splitlines())
    new_all = Counter(line.rstrip() for line in new_text.splitlines())
    dropped = []
    for line, in_draft in draft.items():
        lost = min(in_draft, old_all[line] - new_all.get(line, 0))
        if lost > 0:
            dropped.extend([line] * lost)
    return dropped


def _changelog_new_duplicate_unreleased_bullets(
        old_text: str, new_text: str) -> list[tuple[str, int]]:
    """Top-level [Unreleased] bullets that occur MORE often in *new_text* than in
    *old_text* and now occur at least twice: (line, count-in-working-copy) pairs,
    in working-copy order.

    Only top-level bullets (a raw line starting with "- ") are counted, not their
    wrapped continuations: two different bullets can legitimately wrap to the same
    trailing words, whereas two byte-identical bullet lines are a mistake every
    time. Baseline-relative so a duplicate that already exists on master does not
    nag every branch (see the block comment above)."""
    from collections import Counter
    old = Counter(line for line in _changelog_unreleased_lines(old_text)
                  if line.startswith("- "))
    new = Counter(line for line in _changelog_unreleased_lines(new_text)
                  if line.startswith("- "))
    seen = set()
    out = []
    for line in _changelog_unreleased_lines(new_text):
        if not line.startswith("- ") or line in seen:
            continue
        seen.add(line)
        count = new[line]
        if count >= 2 and count > old.get(line, 0):
            out.append((line, count))
    return out


# Baseline CHANGELOG text, cached per (REPO, ref). Safe to cache without any
# staleness risk: the key is an immutable SHA, so its blob content cannot change.
# Caching it turns three `git show` spawns per run into one - the three warn-only
# checks each need the same baseline, and this gate runs as a pre-commit hook.
_BASELINE_TEXT_CACHE: dict[tuple[Path, str], str | None] = {}


def _changelog_baseline_pair() -> tuple[str, str, str] | None:
    """(ref, baseline CHANGELOG text, working CHANGELOG text), or None when there
    is nothing to compare against (no git, or no CHANGELOG in the baseline yet).
    Shared by the warn-only checks so they cannot drift apart on which baseline
    they read - the merge-base choice is the whole correctness story here (see the
    block comment above).

    The BASELINE side is cached (immutable sha => immutable content); the WORKING
    side is deliberately re-read every call, because it is the thing under test and
    is cheap to read (no subprocess)."""
    ref = _changelog_baseline_ref()
    if ref is None:
        return None                     # no git available: nothing to diff against
    key = (REPO, ref)
    if key not in _BASELINE_TEXT_CACHE:
        base = _git("show", f"{ref}:{_CHANGELOG}")
        _BASELINE_TEXT_CACHE[key] = (
            base.stdout if base is not None and base.returncode == 0 else None)
    base_text = _BASELINE_TEXT_CACHE[key]
    if base_text is None:
        return None                     # CHANGELOG not in the baseline yet: no record
    try:
        working = (REPO / _CHANGELOG).read_text(encoding="utf-8")
    except OSError:
        working = ""                    # deleted from the tree: every draft line is gone
    return ref, base_text, working


def _changelog_unreleased_drops() -> list[str]:
    """Warn-only companion to _changelog_append_only: [Unreleased] draft lines
    present at the baseline but gone from the working copy."""
    pair = _changelog_baseline_pair()
    if pair is None:
        return []
    ref, base_text, working = pair
    dropped = _changelog_dropped_unreleased_lines(base_text, working)
    if not dropped:
        return []
    listing = "\n".join(f"    lost: {x!r}" for x in dropped)
    return [
        f"{_CHANGELOG}: {len(dropped)} [Unreleased] draft line(s) present at the "
        f"baseline ({ref[:8]}) are missing from the working copy:\n{listing}\n"
        "    Rewording or removing your OWN draft entries is fine. What this "
        "catches is the other case: a SIBLING branch's bullet lost around a "
        "rebase (observed after a conflicted rebase resolved bulk-take-mine; "
        "resolve those additively, keeping BOTH sides' bullets).\n"
        "    Attribute a line before acting on it - `git log -S \"<line>\" -- "
        f"{_CHANGELOG}` - then restore only what you actually dropped. Do NOT "
        "reset the section to master and do NOT hand-copy a bullet back in "
        "blind: re-importing a bullet that was never lost is how duplicates get "
        "created."
    ]


def _changelog_added_unreleased_bullets(old_text: str, new_text: str) -> list[str]:
    """Top-level [Unreleased] bullets in *new_text* that were not in *old_text* -
    this branch's own additions, as far as the text can tell."""
    from collections import Counter
    old = Counter(line for line in _changelog_unreleased_lines(old_text)
                  if line.startswith("- "))
    seen = Counter()
    out = []
    for line in _changelog_unreleased_lines(new_text):
        if not line.startswith("- "):
            continue
        seen[line] += 1
        if seen[line] > old.get(line, 0):
            out.append(line)
    return out


def _changelog_unreleased_duplicates() -> list[str]:
    """Warn-only: an [Unreleased] bullet that is newly duplicated in the working
    copy - the artifact left behind when a bullet is hand-restored after a
    false-positive drop report."""
    pair = _changelog_baseline_pair()
    if pair is None:
        return []
    _ref, base_text, working = pair
    dupes = _changelog_new_duplicate_unreleased_bullets(base_text, working)
    if not dupes:
        return []
    listing = "\n".join(f"    x{count} now: {line!r}" for line, count in dupes)
    return [
        f"{_CHANGELOG}: {len(dupes)} [Unreleased] bullet(s) appear MORE OFTEN than "
        f"at the baseline, and more than once:\n{listing}\n"
        "    A duplicate is usually a bullet that was restored but never actually "
        "lost (a stale-ref drop report false-positives on every sibling bullet "
        "merged after your branch point). Delete the EXTRA copies so only the one "
        "you meant to have is left - deleting every copy is how the entry "
        "disappears for real."
    ]


def _changelog_unreleased_added_note() -> list[str]:
    """REPORT-ONLY context: the [Unreleased] bullets this branch adds relative to
    the baseline. Not a warning and never escalated by --strict (a branch adding
    changelog entries is the whole point), so it is emitted ONLY alongside a real
    warning, where it answers the question the warning immediately raises: "which
    of these bullets are mine?"

    It is the import detector's readable half. A hand-restored sibling bullet that
    is NOT a duplicate is textually indistinguishable from one you authored, so no
    check can flag it outright; listing the additions lets the human reading the
    warning spot one. Deliberately not printed on clean runs: this gate is a
    pre-commit hook, and a note on every commit is how a signal becomes noise."""
    pair = _changelog_baseline_pair()
    if pair is None:
        return []
    _ref, base_text, working = pair
    added = _changelog_added_unreleased_bullets(base_text, working)
    if not added:
        return [f"    for context: this branch adds no new {_CHANGELOG} "
                "[Unreleased] bullets."]
    shown = "\n".join(f"    added: {x!r}" for x in added[:6])
    more = f"\n    (+{len(added) - 6} more)" if len(added) > 6 else ""
    return [f"    for context, this branch adds {len(added)} [Unreleased] "
            f"bullet(s) - confirm they are all yours:\n{shown}{more}"]


# ---- check 5: raw single-resource accessor guard ----------------------------
# Each entry maps a raw single-resource function name to the wrapper that must
# be used instead, and the file(s) allowed to still call the raw function
# directly - only the function's own home module (definition + the wrapper's
# own fallback call) by default. `tests/` is exempt everywhere: tests
# legitimately call/mock the raw function directly to test IT, not just its
# consumers. Every OTHER consumer must go through the wrapper - no single-
# device exception is granted by default; add one here only with a
# documented reason that survives "every relevant function should be
# multi/split-aware" as the bar (a maintainer explicitly said so once every
# remaining single-GPU call site turned out to have no real reason to stay
# that way - see dev-notes/gpu-split-capacity-fix/).
#
# Add a new entry here whenever a similar "single -> combined N resources"
# capability ships (see dev-notes/gpu-split-capacity-fix/ for the multi-GPU
# VRAM case this was written for) - do not just write a review note.
_RAW_ACCESSOR_GUARDS = {
    "vram_info": {
        "wrapper": "localm/discover.py's vram_capacity()",
        "allowed": {
            # home module: the definition, plus vram_capacity()'s own
            # documented fallback to the single-GPU number.
            "localm/discover.py",
        },
    },
}


# A test may not allocate this much real disk in one call. 100 MB is far above
# any legitimate fixture (the biggest honest one in the tree is a 2 MB split-part
# pair) and far below the GB-scale sizes that caused the incident.
_MAX_TEST_FILE_BYTES = 100_000_000


def _const_bytes(node: ast.AST) -> "int | None":
    r"""Byte size of a literal size expression, or None when not resolvable.

    A deliberately tiny constant-folder rather than ast.literal_eval: it needs to
    understand only the shapes a size argument actually takes - `9_000_000_000`,
    `b"\0" * 4096`, `2 * 1024 ** 3` - and folding them explicitly keeps this
    guard free of any eval-family call.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return node.value
        if isinstance(node.value, (bytes, str)):
            return len(node.value)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        left = _const_bytes(node.left)
        right = _const_bytes(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Mult):
                return left * right
            if right > 64 or left > 1 << 20:
                return None          # refuse a silly exponent, never hang the gate
            return left ** right
        except (OverflowError, ValueError):
            return None
    return None


def _big_test_write_violations(files: list[Path]) -> list[str]:
    r"""Fail when a TEST allocates >= _MAX_TEST_FILE_BYTES of real disk.

    Why this is a gate and not a comment: truncate() is NOT sparse on Windows/NTFS
    (measured: one truncate(2GB) consumes 1.61 GB for real), and pytest gives each
    test its own tmp_path, keeps the last 3 basetemps, and xdist multiplies that by
    the worker count. Two test files doing this quietly allocated ~17.5 GB per pass
    -> ~315 GB across a real -n 6 run, filled D: to 99.5% (9 GB free of 1863), and
    crashed the box (2026-07-15). test_auto_gpu_layers.py had carried a "NEVER
    truncate() to GB sizes here" comment since an earlier incident; the comment did
    not stop two OTHER files from doing exactly that, which is the whole argument
    for enforcing it mechanically (AGENTS.md "enforce, don't just document").

    The fix a violation wants is never "write fewer bytes" - it is to stop writing
    them at all: create a tiny real file and FAKE the size the code reads back
    (`b._model_bytes = lambda: size_bytes`), the pattern test_auto_gpu_layers.py,
    test_vram_preflight.py and test_kv_bytes_offload.py all use now.

    Two shapes are caught:
      1. a direct literal:            fh.truncate(9_000_000_000)
      2. a helper truncating a NAME (`fh.truncate(size_bytes)`) that is CALLED
         with a big literal anywhere in the same module - the exact shape that hid
         both real offenders behind an innocent-looking helper.
    """
    problems = []
    for path in files:
        if path.suffix != ".py":
            continue
        rel = path.relative_to(REPO).as_posix()
        if not (rel.startswith("tests/") or "/tests/" in rel):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (UnicodeDecodeError, OSError, SyntaxError) as e:
            # Not silently skipped: an unscanned file must never read as clean.
            problems.append(f"{rel}: could not parse for the big-write guard ({e})")
            continue

        truncates_a_name = False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            fname = node.func.attr
            if fname not in ("truncate", "write_bytes") or not node.args:
                continue
            size = _const_bytes(node.args[0])
            if size is not None and size >= _MAX_TEST_FILE_BYTES:
                problems.append(
                    f"{rel}:{node.lineno}: {fname}() allocates {size/1e9:.2f} GB of "
                    f"REAL disk (truncate/write is NOT sparse on NTFS). Write a tiny "
                    f"real file and fake the size the code reads back instead - see "
                    f"test_auto_gpu_layers.py's _model()."
                )
            elif size is None and fname == "truncate" and isinstance(node.args[0], ast.Name):
                truncates_a_name = True

        if not truncates_a_name:
            continue
        # A helper truncates a variable: any big literal handed to a size-ish
        # kwarg or default in this module reaches real disk through it.
        for node in ast.walk(tree):
            for kw in getattr(node, "keywords", None) or []:
                if kw.arg not in ("size", "size_bytes", "n_bytes"):
                    continue
                size = _const_bytes(kw.value)
                if size is not None and size >= _MAX_TEST_FILE_BYTES:
                    problems.append(
                        f"{rel}:{kw.value.lineno}: {kw.arg}={size/1e9:.2f} GB is passed "
                        f"to a helper that truncate()s it to real disk. Fake the size "
                        f"instead of allocating it."
                    )
            if isinstance(node, ast.arguments):
                for d in node.defaults:
                    size = _const_bytes(d)
                    if size is not None and size >= _MAX_TEST_FILE_BYTES:
                        problems.append(
                            f"{rel}:{d.lineno}: a default size of {size/1e9:.2f} GB is "
                            f"truncate()d to real disk. Fake the size instead."
                        )
    return problems


def _raw_accessor_violations(files: list[Path]) -> list[str]:
    problems = []
    for name, spec in _RAW_ACCESSOR_GUARDS.items():
        allowed = spec["allowed"]
        for path in files:
            if path.suffix != ".py":
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel.startswith("tests/") or rel in allowed:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text, filename=rel)
            except (UnicodeDecodeError, OSError, SyntaxError) as e:
                # A tracked .py the guard cannot read or parse was NOT checked - report
                # it rather than `continue`-ing past it, or a file that silently evades
                # the raw-accessor guard reads as "clean" when it was never scanned
                # (AGENTS.md rule 5). Genuine .py files in the tree all parse under the
                # 3.12 interpreter this runs on, so this only fires on a real anomaly.
                problems.append(
                    f"{rel}: could not read/parse to check the raw-accessor guard "
                    f"({type(e).__name__}: {e}) - not checked. A .py the guard cannot "
                    "parse must not silently pass; fix the file or exclude it explicitly.")
                continue
            # Local names this module binds the raw accessor to: the literal
            # name itself, PLUS any `from ... import <name> as <alias>` -
            # an import alias is a demonstrated, real bypass of a literal-name
            # match (verified with `... import vram_info as vi; vi()`), not
            # a hypothetical one. A bare module-attribute call (`disc.vram_info()`)
            # is already caught regardless of what the MODULE is aliased to,
            # since ast.Attribute.attr is still the literal accessor name.
            bound_names = {name}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for imp_alias in node.names:
                        if imp_alias.name == name:
                            bound_names.add(imp_alias.asname or imp_alias.name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = (func.id if isinstance(func, ast.Name) else
                          func.attr if isinstance(func, ast.Attribute) else None)
                if called in bound_names:
                    via = f" (imported as {called!r})" if called != name else ""
                    problems.append(
                        f"{rel}:{node.lineno}: calls {name}(){via} directly - use "
                        f"{spec['wrapper']} instead (or add {rel!r} to "
                        f"_RAW_ACCESSOR_GUARDS in check_hygiene.py with a "
                        f"documented reason if this really is a legitimate "
                        f"single-resource exception)")
    return problems


# ---- check 6: PWA service-worker cache derivation --------------------------
# The GUI's service worker (sw.js) serves static assets CACHE-FIRST, and a browser
# only re-runs a service worker's install (which re-fetches the assets) when sw.js's
# OWN bytes change. sw.js's CACHE constant used to be a hand-typed version string
# bumped whenever a cached asset changed - a value whose only requirement was "differ
# from last time", carrying no meaning of its own.
#
# That shipped stale THREE times undetected by human review before a bump gate was
# added here (v49 for #621's settings.js fix; again for the managed_comfy_enabled
# checkbox removal; PR #640's models.js + knowledge.js, saved only by luck when a
# later unrelated PR happened to bump it). The gate then caught real misses for a
# while - but a hand-maintained "must differ" line is ALSO a guaranteed merge
# conflict between any two concurrent GUI PRs: it went v88 -> v89 -> v90 -> v91 in
# one afternoon across three unrelated PRs, and a fourth session spent two rebase
# rounds on that single line - during which its PR ran with NO CI at all, because a
# conflicted PR gets no checks computed against it. One contested line silently cost
# a session its entire verification signal.
#
# The fix is to stop hand-maintaining the value at all: localm/plugins/gui/web.py's
# GET /sw.js route now computes CACHE fresh on every request, from a content digest
# of the actual static assets being served (see web.py's _compute_sw_cache_value).
# It changes exactly when a cacheable asset's bytes change, for every asset the
# service worker's fetch handler can runtime-cache - not just the SHELL precache
# list, which alone left 22 shipped files (including all 20 KaTeX fonts and
# /vendor/jsQR.min.js) permanently unwatched. Nothing about that value is checked
# into git, so two branches touching different assets can never conflict over it.
#
# What THIS check still enforces, now that "was CACHE bumped" is no longer a
# meaningful question: SHELL precache coverage (every listed entry names a real
# file, every app/*.js and pages/*.js module is listed - unrelated to staleness,
# still a real drift risk on its own) and that the CACHE placeholder line web.py's
# route substitutes into is still in the expected shape, so an edit to sw.js cannot
# silently break that substitution without this check catching it.
_SW_STATIC = "localm/plugins/gui/static"
_SW_JS = f"{_SW_STATIC}/sw.js"

# sw.js's SHELL comment promises to precache "every app/* and pages/* module (the
# import graph)". Globs, NOT a copied file list: duplicating SHELL here would be the
# very drift bug this gate exists to stop.
_SW_SHELL_MODULE_GLOBS = ("app/*.js", "pages/*.js")


def _sw_shell_files(sw_js_text: str) -> set[str]:
    """The SHELL precache array's asset URL paths, parsed out of sw.js's own
    source (a plain regex over the fixed, hand-maintained JS array literal -
    not a real JS parser, but sufficient for this one array).

    An EMPTY result means the array could not be parsed - the real SHELL is never
    empty - so callers must treat empty as a failure, never as "nothing is
    precached" (rule 5). See _sw_cache_derivation_violations."""
    m = re.search(r"const SHELL = \[(.*?)\];", sw_js_text, re.S)
    if not m:
        return set()
    return set(re.findall(r'"(/[^"]+)"', m.group(1)))


def _sw_cache_version(sw_js_text: str) -> str | None:
    """sw.js's CACHE constant (a placeholder that web.py's route substitutes at
    request time - see check 6's block comment). None => unparseable; callers
    must fail loud, not skip."""
    m = re.search(r'const CACHE = "([^"]+)"', sw_js_text)
    return m.group(1) if m else None


def _sw_cache_derivation_violations() -> list[str]:
    """SHELL precache coverage, plus a sanity check that sw.js's CACHE placeholder
    line is still in the shape localm/plugins/gui/web.py's GET /sw.js route expects
    to substitute into (see check 6's block comment for what this replaced and why).

    Every failure path here is LOUD. A silent `return []` when this check cannot do
    its job is how it would rot into decoration: the regexes below match one exact
    hand-maintained format, so a benign reformat of sw.js would otherwise disable the
    gate permanently and invisibly. That is the same standard main() already applies
    to _tracked_files() - a gate that reports clean without actually checking
    anything is the silent pass AGENTS.md rule 5 forbids."""
    sw_path = REPO / _SW_JS
    try:
        working_sw = sw_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Absent vs unreadable are different (rule 5): a checkout with no GUI service
        # worker at all has genuinely nothing to gate, but one where the rest of the
        # static tree is present means sw.js MOVED and the gate is now pointed at
        # nothing - which must not pass silently.
        still_ships = _git("ls-files", "-z", "--", _SW_STATIC)
        if still_ships is not None and still_ships.returncode == 0 and still_ships.stdout:
            return [f"{_SW_JS}: missing, but {_SW_STATIC}/ still ships assets - the PWA "
                    "cache-derivation gate is pointed at a file that no longer exists "
                    "and just checked NOTHING. Update _SW_JS in this script to the new "
                    "path (and web.py's route, which reads the same file)."]
        # Falls through on no output (a checkout with no GUI) and on a failed git call
        # (REPO is not a git tree). Neither is hidden: a tree git cannot enumerate
        # ALREADY fails loud in main(), which refuses to report clean without scanning
        # anything, so duplicating that alarm here would only fire on non-git callers
        # that have nothing to gate in the first place.
        return []
    except OSError as e:
        return [f"{_SW_JS}: exists but could not be read ({e}) - the PWA "
                "cache-derivation gate could not run."]

    shell = _sw_shell_files(working_sw)
    working_cache = _sw_cache_version(working_sw)
    if not shell:
        return [f"{_SW_JS}: could not parse the SHELL precache array (expected "
                '`const SHELL = [ "/asset", ... ];`). The PWA cache-derivation gate '
                "reads it to check precache coverage, so it just checked NOTHING. "
                "Restore that format or update _sw_shell_files in this script."]
    if working_cache is None:
        return [f"{_SW_JS}: could not parse the CACHE constant (expected "
                '`const CACHE = "...";`). localm/plugins/gui/web.py\'s GET /sw.js '
                "route substitutes this line's value on every request - if it cannot "
                "find it either, every client gets a 500 instead of the service "
                "worker. Restore that format or update _sw_cache_version in this "
                "script AND SW_CACHE_LINE_RE in web.py together."]

    return _sw_shell_coverage_problems(shell, sw_path.parent)


def _sw_shell_coverage_problems(shell: set[str], static_root: Path) -> list[str]:
    """SHELL must name real files, and must name every shell module - sw.js promises
    "every app/* and pages/* module". A SHELL entry with no file behind it is silently
    dropped at install time (Promise.allSettled), and a module missing from SHELL is not
    precached at all, so the installed PWA cannot open it offline. Enforced rather than
    documented, per the "N consumers, only some updated" rule."""
    problems = []
    for url_path in sorted(shell):
        if not (static_root / url_path.lstrip("/")).is_file():
            problems.append(
                f"{_SW_JS}: SHELL precaches {url_path!r}, but no such file exists under "
                f"{_SW_STATIC}/ - install() drops it silently (Promise.allSettled). Fix "
                "the path or remove the entry.")
    for glob in _SW_SHELL_MODULE_GLOBS:
        for mod in sorted(static_root.glob(glob)):
            url = "/" + mod.relative_to(static_root).as_posix()
            if url not in shell:
                problems.append(
                    f"{_SW_STATIC}{url}: ships but is NOT listed in {_SW_JS}'s SHELL, "
                    "which promises every app/* and pages/* module - so it is never "
                    "precached and the installed PWA cannot open it offline. Add it "
                    "to SHELL.")
    return problems


def _install_hook() -> int:
    # Resolve the hooks dir through git rather than assuming REPO/.git is a
    # directory: in a WORKTREE, .git is a FILE pointing at the main checkout, so
    # REPO/".git"/"hooks" does not exist and installing from a worktree failed
    # outright with "No .git/hooks directory found". --git-common-dir gives the
    # shared .git for both a normal checkout and a worktree.
    base = REPO / ".git"
    try:
        # encoding="utf-8" for the same reason as _git: a non-ASCII checkout path
        # must not be locale-decoded (cp1252 on Windows) into a wrong hooks dir.
        out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8",
                             timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            base = Path(out.stdout.strip())
            if not base.is_absolute():
                base = (REPO / base).resolve()
    except (OSError, subprocess.SubprocessError):
        pass        # fall back to REPO/.git below; the is_dir() check reports it
    hook = base / "hooks" / "pre-commit"
    if not hook.parent.is_dir():
        print(f"No hooks directory found at {hook.parent}.", file=sys.stderr)
        return 1
    # Run THIS project's own interpreter, never a bare PATH `python`. On Windows a
    # bare `python` hits the Store app-execution alias, which happily resolves to an
    # interpreter that has since been uninstalled and fails the hook with a cryptic
    # "[ERROR] Failed to launch '...python.exe' (0x80070003)" - blocking every commit
    # in the repo, with nothing pointing at the cause (seen live, 2026-07-15). The
    # venv is also the interpreter that actually has this project's deps.
    #
    # Resolved from the git COMMON dir's parent, not --show-toplevel: in a worktree
    # --show-toplevel is the worktree (which has no .venv of its own), while the venv
    # lives beside the main checkout's .git. No absolute path is baked in (rule 1);
    # every path is derived at hook run time.
    hook.write_text(
        "#!/bin/sh\n"
        'root=$(git rev-parse --show-toplevel)\n'
        'main=$(cd "$(git rev-parse --git-common-dir)/.." 2>/dev/null && pwd) || main=$root\n'
        'for py in "$main/.venv/Scripts/python.exe" "$main/.venv/bin/python" \\\n'
        '          "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python"; do\n'
        '    [ -x "$py" ] && exec "$py" "$root/scripts/check_hygiene.py"\n'
        'done\n'
        '# No venv yet (a fresh clone before setup): fall back to PATH.\n'
        'for py in python3 python; do\n'
        '    command -v "$py" >/dev/null 2>&1 && exec "$py" "$root/scripts/check_hygiene.py"\n'
        'done\n'
        'echo "pre-commit: no usable Python found (tried this project\'s .venv, then'
        ' python3/python on PATH)." >&2\n'
        'echo "Run setup, or reinstall the hook with:'
        '  <venv-python> scripts/check_hygiene.py --install-hook" >&2\n'
        'exit 1\n',
        encoding="utf-8",
    )
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    print(f"Installed pre-commit hook at {hook}")
    return 0


# ---- check 7: no module-level import cycles between packages ---------------
#
# WHY A CYCLE CHECK AND NOT A DECLARED LAYER MAP. localm has no declared layering,
# and the cost of that is measurable: two readers auditing the same tree derived
# DIFFERENT "upward edge" counts (2 and 7) purely because each invented a tier map,
# and four of the seven were artifacts of one map omitting modules rather than real
# inversions. A hand-written map is a maintained artifact that encodes opinions;
# every disagreement with it becomes an allowlist entry, which is documentation
# pretending to be enforcement.
#
# Acyclicity needs no map and no allowlist. It is a property of the graph, so the
# two shapes that any tier map has to carve out are legal BY CONSTRUCTION:
#   * an ENTRY POINT (localm/__main__.py importing localm.cli) is a source node;
#   * PEERS (image_gen/music_gen/video_gen all importing media) are unordered.
# Neither can create a cycle, so neither needs an exception. What it does catch is
# a genuine mutual dependency, which is the thing that actually hurts: it makes
# import order load-bearing and forces the deferred-import workarounds below.
#
# MODULE-LEVEL ONLY, deliberately. A function-local import is the standard,
# intentional way to break a cycle in Python and this codebase uses it that way on
# purpose (73% of intra-package imports are function-local). Flagging those would
# fire on dozens of documented design choices and need exactly the allowlist this
# check exists to avoid. Only eager, module-level edges count.
#
# "MODULE-LEVEL" MEANS "RUNS DURING IMPORT", NOT "UNINDENTED". A ``def``/``async
# def``/``class`` body is genuinely deferred (never touched until called or
# instantiated), but a module-level ``try:``/``if:`` body is NOT deferred - it
# executes unconditionally-during-import (``try``) or whenever its guard is true
# (``if``), same as a bare top-level statement, just indented. The common
# "optional dependency" (``try: import X except ImportError: X = None``) and
# "platform fallback" (``if sys.platform == ...``) idioms are exactly this shape,
# so excluding them left real eager coupling invisible to the graph. Both branches
# of an ``if``/``try`` are walked (either can execute depending on the runtime
# condition/exception, and either can be the one that creates a real edge), while
# ``def``/``class`` bodies nested inside a ``try``/``if`` still are not - the
# import inside THEM stays deferred regardless of what encloses them. The one
# carve-out is ``if TYPE_CHECKING:``: that guard is False at runtime by
# definition (True only for a static type checker), so an import inside it never
# actually executes - it is the standard way to add a type-only import without
# creating a REAL cycle, and counting it would flag that idiom as if it did.
#
# RELATIVE IMPORTS (``from . import x``, ``from ..config import y``, ...) are
# resolved to their absolute ``localm.x.y`` target the same way Python itself
# does at runtime (anchored on the importing module's ``__package__``), not
# skipped. A relative import is exactly as eager and exactly as capable of
# closing a cycle as an absolute one; the only difference is spelling.
#
# Granularity is the top-level unit under localm/ (the package name, or the module
# name for a top-level .py), because that is the level a layering claim is made at.
# Intra-package cycles are a separate, much noisier question and are not in scope.


def _import_unit(module: str) -> str:
    """The top-level unit under ``localm`` that *module* belongs to."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "<root>"


def _module_name(path: Path, pkg_root: Path) -> str:
    rel = path.relative_to(pkg_root).as_posix()[: -len(".py")]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return "localm" + ("." + rel.replace("/", ".") if rel else "")


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:`` - the
    conventional name either way, never True at runtime, so the body never
    actually executes (see the check-7 block comment above)."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _eager_module_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Statements in *body* that run EAGERLY at module-import time: direct
    statements plus, recursively, anything inside a module-level ``try``/
    ``except``/``else``/``finally`` or ``if``/``elif``/``else`` block. Both
    branches of an ``if``/``try`` are included - either can run depending on
    the runtime condition or exception, and either can be the one that creates
    a real edge. ``def``/``async def``/``class`` bodies are never recursed
    into (deferred / separately-scoped, regardless of what encloses them), and
    neither is an ``if TYPE_CHECKING:`` body (never True at runtime). See the
    check-7 block comment above for why."""
    out: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            out.append(stmt)
        elif isinstance(stmt, ast.If):
            if _is_type_checking_guard(stmt.test):
                continue
            out.extend(_eager_module_statements(stmt.body))
            out.extend(_eager_module_statements(stmt.orelse))
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            out.extend(_eager_module_statements(stmt.body))
            for handler in stmt.handlers:
                out.extend(_eager_module_statements(handler.body))
            out.extend(_eager_module_statements(stmt.orelse))
            out.extend(_eager_module_statements(stmt.finalbody))
        # def/async def/class: deferred or separately-scoped; do not recurse.
    return out


def _resolve_relative_import(node: ast.ImportFrom, own_module: str,
                              is_package: bool) -> "str | None":
    """The absolute ``localm...`` module a relative ``ImportFrom`` resolves to,
    mirroring ``importlib._bootstrap._resolve_name``: a level-N import is
    anchored at the importing module's ``__package__``, then walked up (N-1)
    more components. ``__package__`` for a package's ``__init__.py`` IS the
    package's own name; for a plain module it is the module's PARENT - one
    relative level shallower inside an ``__init__.py`` than in a sibling
    module of that same package, which is real Python import semantics, not
    an approximation.

    None when the level walks past *own_module*'s own components (an invalid
    relative import that would fail at runtime with "attempted relative
    import beyond top-level package") - there is nothing to resolve, and this
    never silently mis-resolves to the wrong package instead of reporting
    that."""
    package = own_module if is_package else own_module.rsplit(".", 1)[0]
    parts = package.split(".")
    if node.level > len(parts):
        return None
    base = ".".join(parts[: len(parts) - node.level + 1])
    if not base:
        return None
    return f"{base}.{node.module}" if node.module else base


def _module_level_import_edges(pkg_root: Path) -> dict[str, dict[str, str]]:
    """unit -> {imported unit: "file:line of one eager import that creates it"}.

    A parse failure is REPORTED by the caller rather than skipped: a file this gate
    cannot read is a file it cannot vouch for, and silently treating it as
    edge-free would let a cycle hide behind a syntax error (rule 5)."""
    edges: dict[str, dict[str, str]] = {}
    for path in sorted(pkg_root.rglob("*.py")):
        own_module = _module_name(path, pkg_root)
        unit = _import_unit(own_module)
        is_package = path.name == "__init__.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            edges.setdefault(unit, {})["<unparseable>"] = f"{path}: {e}"
            continue
        for node in _eager_module_statements(tree.body):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    if node.module and node.module.startswith("localm"):
                        targets = [node.module]
                else:
                    resolved = _resolve_relative_import(node, own_module, is_package)
                    if resolved is not None and resolved.startswith("localm"):
                        targets = [resolved]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names if a.name.startswith("localm")]
            for target in targets:
                other = _import_unit(target)
                if other != unit:
                    rel = path.relative_to(REPO).as_posix()
                    edges.setdefault(unit, {}).setdefault(
                        other, f"{rel}:{node.lineno}")
    return edges


def _strongly_connected(edges: dict[str, dict[str, str]]) -> list[list[str]]:
    """Tarjan's SCC, iterative so a deep graph cannot blow the recursion limit."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0
    nodes = sorted(set(edges) | {v for d in edges.values() for v in d})
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, list[str]]] = [(root, sorted(edges.get(root, {})))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            if pending:
                nxt = pending.pop(0)
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, sorted(edges.get(nxt, {}))))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node:
                        break
                result.append(component)
    return result


def _import_cycle_violations() -> list[str]:
    """Module-level import cycles between top-level units under localm/."""
    pkg_root = REPO / "localm"
    if not pkg_root.is_dir():
        return []          # not a localm checkout; other gates report that
    edges = _module_level_import_edges(pkg_root)
    problems = [f"could not parse {detail}"
                for targets in edges.values()
                for other, detail in targets.items() if other == "<unparseable>"]
    for component in _strongly_connected(edges):
        if len(component) < 2:
            continue
        members = set(component)
        involved = sorted(
            f"{u} -> {v} ({edges[u][v]})"
            for u in component for v in edges.get(u, {}) if v in members)
        problems.append(
            "module-level import cycle between localm packages: "
            + " <-> ".join(sorted(component))
            + "\n      closed by: " + "\n                 ".join(involved)
            + "\n      Break it by moving the shared code to a lower unit both can "
              "import (a dependency-free leaf usually belongs at the root), NOT by "
              "deferring the import into a function - that hides the cycle rather "
              "than removing it.")
    return problems



def _strict_env() -> bool:
    """CI-style escalation knob: LOCALM_HYGIENE_STRICT set to anything but a
    recognized OFF value behaves like passing --strict (an env knob because a CI
    step or a hook cannot always edit the command line it invokes).

    "off" is in the off-set alongside 0/false/no: a reviewer pointed out that
    LOCALM_HYGIENE_STRICT=off silently turning strictness ON is the kind of
    surprise that gets a knob mistrusted. Anything unrecognized still means ON,
    so a typo fails toward MORE checking, never less."""
    return os.environ.get("LOCALM_HYGIENE_STRICT", "").strip().lower() not in (
        "", "0", "false", "no", "off")


# Files and directories that are DELIBERATELY NOT PUBLISHED. They live on a
# contributor's disk and are listed in .gitignore, but .gitignore only stops a
# file being added while it is UNTRACKED - it does nothing once something is in
# the index, and nothing at all in a worktree whose .gitignore predates the
# entry. Both of those happened: after the internals were removed, every stale
# worktree still had the old .gitignore, so these showed as ordinary untracked
# files and a single `git add -A` staged all 21 of them for re-publication.
#
# Maintainer, 2026-08-04: "NO MORE DISCLOSING OF ANY INTERNALS ON GH THAT DO NOT
# NEED TO BE ON THERE" and "agents and CLAUDE.md do NOT belong there".
#
# So this is a TRACKED-state check, not a content scan: it asks git what is in
# the index, which is the only question that decides what reaches GitHub.
_NEVER_TRACKED = (
    "AGENTS.md",
    "CLAUDE.md",
    "RELEASE.md",
    "release-manifest.toml",
    "scripts/codeql/",
    "scripts/tier2_gpu_split/",
)


def _never_tracked_violations() -> list[str]:
    """Fail if anything in _NEVER_TRACKED is tracked by git.

    Uses `git ls-files` directly rather than the filtered _tracked_files() list,
    because that one drops binaries and _SKIP_DIRS - a filter that is right for a
    content scan and wrong here, where the question is only "is it in the index".
    """
    r = _git("ls-files", "-z")
    if r is None or r.returncode != 0:
        # Distinguish "nothing tracked" (benign) from "could not ask" (not
        # benign): a publication gate that cannot see the index must not pass.
        return ["could not run 'git ls-files' to verify no internal file is "
                "tracked - this gate cannot be assumed clean, so it fails"]
    tracked = [p for p in r.stdout.split("\0") if p]
    out = []
    for entry in _NEVER_TRACKED:
        if entry.endswith("/"):
            hits = [p for p in tracked if p.startswith(entry)]
        else:
            hits = [p for p in tracked if p == entry]
        for h in sorted(hits):
            out.append(
                f"{h}: internal file is TRACKED and would be published to "
                f"GitHub. It is in .gitignore, but .gitignore does not untrack "
                f"something already in the index. Run: git rm --cached '{h}'")
    return out


def _release_manifest_gate() -> tuple[list[str], list[str]]:
    """Run the full release-manifest gate (check_manifest.check_manifest()) and
    return (failures, warnings).

    scripts/check_manifest.py is itself gitignored (AGENTS.md rule 6 - it is
    release-tooling metadata, not part of the public product, same as
    release-manifest.toml), so it is ABSENT from a fresh CI checkout and from most
    external contributors' clones BY DESIGN - that is expected, not a bug. When it
    cannot be imported this returns a WARNING (escalated to a failure only under
    --strict/LOCALM_HYGIENE_STRICT=1), never a silent skip: the module docstring
    above used to claim this gate was unconditionally "folded in" here when in fact
    nothing ever called it - release-manifest.toml drifted 13 entries stale on
    master with this reporting a clean pass throughout (see
    dev-notes/release-manifest-gate-wiring.md for the incident). Reporting nothing
    when the check cannot run would be that exact failure shape again, just
    relocated."""
    try:
        # Deliberately Path(__file__).parent, NOT the module-level REPO global:
        # check_manifest.py always lives next to THIS file on disk regardless of
        # what tree is under test, whereas REPO is legitimately monkeypatched to a
        # scratch tmp_path by tests exercising the file-scanning checks in
        # isolation. Keying this lookup off REPO would make it silently look for
        # check_manifest.py inside that scratch tree instead of next to the real
        # script, breaking the gate under every such test.
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import check_manifest as cm
    except ImportError:
        return [], [
            "release-manifest gate SKIPPED: scripts/check_manifest.py is not present "
            "in this checkout. Expected on CI and most external clones - it is "
            "intentionally gitignored (AGENTS.md rule 6). This checkout's hygiene "
            "pass does NOT mean the release-manifest classification is clean; run "
            "'python scripts/check_manifest.py' by hand on a checkout that has it "
            "(e.g. the maintainer's own) before cutting a release."]
    return list(cm.check_manifest()), []


def main(argv: list[str]) -> int:
    if "--install-hook" in argv:
        return _install_hook()
    strict = "--strict" in argv or _strict_env()
    tracked = _tracked_files()
    if not tracked:
        # `git ls-files` failed or returned nothing, so the dash/disclosure/abs-path
        # scan and the changelog gate below would run over ZERO files and this gate
        # would print "passed" having checked nothing. A disclosure/privacy gate that
        # reports clean without scanning anything is a silent false pass, so fail loud.
        print("Hygiene check FAILED: could not enumerate tracked files via 'git ls-files' "
              "- nothing was scanned. Run this from a git checkout.",
              file=sys.stderr)
        return 1
    problems: list[str] = []
    problems.extend(_never_tracked_violations())
    for f in tracked:
        problems.extend(_scan(f))
    problems.extend(_changelog_append_only())
    problems.extend(_raw_accessor_violations(tracked))
    problems.extend(_big_test_write_violations(tracked))
    problems.extend(_sw_cache_derivation_violations())
    problems.extend(_import_cycle_violations())
    manifest_failures, manifest_warnings = _release_manifest_gate()
    problems.extend(manifest_failures)
    # Check 4b is warn-only by default (rewording your own [Unreleased] draft is
    # legitimate); --strict / LOCALM_HYGIENE_STRICT=1 folds the warnings into the
    # failures for CI-style use.
    changelog_warnings = _changelog_unreleased_drops() + _changelog_unreleased_duplicates()
    if changelog_warnings:
        # Report-only context, FOLDED INTO the last CHANGELOG warning rather than
        # appended as its own entry. Appending it made --strict print the note inside
        # the FAILED list and count it as a hygiene issue ("2 issue(s)" for one real
        # warning), which is exactly what this note must never be: it is context, and
        # a branch adding changelog bullets is the point of the file. Gated on
        # changelog_warnings specifically (not the combined warnings list below) so
        # it is never misattached to an unrelated warning, e.g. manifest_warnings,
        # when a changelog warning did not actually fire.
        note = _changelog_unreleased_added_note()
        if note:
            changelog_warnings[-1] = changelog_warnings[-1] + "\n" + "\n".join(note)
    # manifest_warnings joins the same escalation path: on the one machine that
    # actually has check_manifest.py (the maintainer's), a --strict run (as used
    # before cutting a release) should fail loud rather than silently accept a
    # checkout that is somehow missing it.
    warnings = changelog_warnings + manifest_warnings
    if strict and warnings:
        problems.extend(warnings)
        warnings = []
    if warnings:
        print("Hygiene WARNING(S) - not failures; pass --strict or set "
              "LOCALM_HYGIENE_STRICT=1 to escalate them:\n", file=sys.stderr)
        for w in warnings:
            print("  " + w, file=sys.stderr)
        print(file=sys.stderr)
    if problems:
        print("Hygiene check FAILED:\n", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} hygiene issue(s).", file=sys.stderr)
        return 1
    print("Hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
