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
  6. A PWA-precached GUI static file (listed in sw.js's SHELL array) changed
     without sw.js's own CACHE version constant also being bumped - an already-
     installed PWA would keep serving the stale cached file forever, since a
     service worker only re-checks its precached files when its own bytes
     change. Shipped twice undetected by a human before this check existed.

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

import ast
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
# An H2 section header opening a PUBLISHED version section, e.g. "## [0.1.1] - date".
# "## [Unreleased]" does not match (no leading digit), so its section stays editable.
_CHANGELOG_VERSION_HEADER = re.compile(r"^##\s+\[\d")


def _changelog_protected_lines(text: str) -> list[str]:
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
    guard exists to catch, the same as editing a bullet's wording."""
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
            published = bool(_CHANGELOG_VERSION_HEADER.match(stripped))
            if published:
                out.append(line)   # the header itself is part of the published record
            continue
        if not published:
            continue
        if not stripped or _CHANGELOG_LINKREF.match(stripped):
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


# ---- check 6: PWA service-worker cache-version bump gate -------------------
# The GUI's service worker (sw.js) serves static assets CACHE-FIRST, and a browser
# only re-runs a service worker's install (which re-fetches the assets) when sw.js's
# OWN bytes change - its CACHE version constant is what we bump to force that. So a
# cached asset that changes without a matching CACHE bump is invisible to anyone who
# already has the PWA installed: they keep the old copy forever.
#
# This has shipped THREE times undetected by human review (v49 for #621's settings.js
# fix; again for the managed_comfy_enabled checkbox removal; and PR #640's models.js +
# knowledge.js, which was a live field bug only by luck - a later unrelated PR bumped
# the cache). Hence a gate, not a fourth "please remember" note.
#
# SCOPE - why this watches EVERY static asset and not just sw.js's SHELL list:
# SHELL is only the PRE-cache (what install() fetches up front). sw.js's fetch handler
# ALSO runtime-caches every same-origin, non-API, non-navigate GET into the SAME
# versioned cache (`caches.open(CACHE).then((c) => c.put(req, copy))`) and serves it
# cache-first forever after, with no revalidation and no expiry - the only eviction is
# activate() dropping caches whose name != CACHE, i.e. a bump. So a NON-SHELL asset
# goes stale exactly as hard as a SHELL one. Watching SHELL alone left 22 shipped files
# permanently unwatched, including /vendor/jsQR.min.js (lazily loaded by the QR pairing
# scanner) and all 20 KaTeX fonts. The honest watch set is "everything the worker can
# cache", so this over-approximates to all tracked assets under the static root and
# subtracts only what sw.js's fetch handler provably never caches (_SW_UNCACHED). An
# over-approximation errs toward the gate firing, which is the right direction here:
# the cost of a spurious fire is one cheap CACHE bump, the cost of a miss is a fix no
# installed client ever receives.
_SW_STATIC = "localm/plugins/gui/static"
_SW_JS = f"{_SW_STATIC}/sw.js"

# Paths (relative to the static root) that sw.js's fetch handler returns early on, so
# they are never cache-first and a bump cannot matter for them. Kept in sync with the
# regex in sw.js's fetch listener by hand; nothing under static/ matches today, so this
# is future-proofing, not dead weight.
_SW_UNCACHED = re.compile(r"^(api|v1|plugins|localm-ca\.crt)(/|$)")

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
    precached" (rule 5). See _sw_cache_bump_violations."""
    m = re.search(r"const SHELL = \[(.*?)\];", sw_js_text, re.S)
    if not m:
        return set()
    return set(re.findall(r'"(/[^"]+)"', m.group(1)))


def _sw_cache_version(sw_js_text: str) -> str | None:
    """sw.js's CACHE constant. None => unparseable; callers must fail loud, not skip."""
    m = re.search(r'const CACHE = "([^"]+)"', sw_js_text)
    return m.group(1) if m else None


def _sw_cacheable_assets() -> list[str]:
    """Repo-relative paths of every tracked static asset the service worker can cache.

    Deliberately re-enumerates via git instead of reusing main()'s _tracked_files():
    that list drops _BINARY_EXTS (.woff2, .png) and skips the whole vendor/ directory,
    which is correct for the text-scanning checks but would blind THIS check to exactly
    the assets it most needs to watch (the KaTeX fonts and jsQR both live there)."""
    out = _git("ls-files", "-z", "--", _SW_STATIC)
    if out is None or out.returncode != 0:
        return []
    assets = []
    for rel in out.stdout.split("\0"):
        if not rel or rel == _SW_JS:
            continue                     # sw.js gates the others; it cannot gate itself
        if _SW_UNCACHED.match(rel[len(_SW_STATIC) + 1:]):
            continue
        assets.append(rel)
    return assets


def _sw_cache_bump_violations() -> list[str]:
    """A cacheable static asset must never change without sw.js's CACHE constant
    changing in the same diff (see the block comment above for the incident record).

    Every failure path here is LOUD. A silent `return []` when this check cannot do its
    job is how it would rot into decoration: the regexes below match one exact
    hand-maintained format, so a benign reformat of sw.js would otherwise disable the
    gate permanently and invisibly. That is the same standard main() already applies to
    _tracked_files() - a gate that reports clean without actually checking anything is
    the silent pass AGENTS.md rule 5 forbids."""
    sw_path = REPO / _SW_JS
    try:
        working_sw = sw_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Absent vs unreadable are different (rule 5): a checkout with no GUI service
        # worker at all has genuinely nothing to gate, but one where the rest of the
        # static tree is present means sw.js MOVED and the gate is now pointed at
        # nothing - which must not pass silently.
        if _sw_cacheable_assets():
            return [f"{_SW_JS}: missing, but {_SW_STATIC}/ still ships assets - the PWA "
                    "cache-bump gate is pointed at a file that no longer exists and just "
                    "checked NOTHING. Update _SW_JS in this script to the new path."]
        return []
    except OSError as e:
        return [f"{_SW_JS}: exists but could not be read ({e}) - the PWA cache-bump gate "
                "could not run."]

    shell = _sw_shell_files(working_sw)
    working_cache = _sw_cache_version(working_sw)
    if not shell:
        return [f"{_SW_JS}: could not parse the SHELL precache array (expected "
                '`const SHELL = [ "/asset", ... ];`). The PWA cache-bump gate reads it '
                "to check precache coverage, so it just checked NOTHING. Restore that "
                "format or update _sw_shell_files in this script."]
    if working_cache is None:
        return [f"{_SW_JS}: could not parse the CACHE constant (expected "
                '`const CACHE = "localm-shell-vN";`). The PWA cache-bump gate compares '
                "it against the baseline, so it just checked NOTHING. Restore that "
                "format or update _sw_cache_version in this script."]

    problems = _sw_shell_coverage_problems(shell, sw_path.parent)

    ref = _changelog_baseline_ref()
    if ref is None:
        return problems                  # no git at all: the diff below cannot run
    base_result = _git("show", f"{ref}:{_SW_JS}")
    if base_result is None or base_result.returncode != 0:
        return problems                  # sw.js not in the baseline yet: nothing to diff
    base_cache = _sw_cache_version(base_result.stdout)
    if base_cache is None:
        problems.append(
            f"{_SW_JS}: could not parse the CACHE constant from the baseline ({ref[:8]}), "
            "so the PWA cache-bump gate could not tell whether it was bumped.")
        return problems
    if working_cache != base_cache:
        return problems                  # bumped: exactly what this gate asks for

    # One diff over the whole static tree, so an asset that was DELETED or ADDED counts
    # too - a per-file is_file() walk silently skipped deletions.
    changed = _git("diff", "--name-only", "-z", ref, "--", _SW_STATIC)
    if changed is None or changed.returncode != 0:
        return problems
    watched = set(_sw_cacheable_assets())
    stale = sorted(p for p in changed.stdout.split("\0") if p in watched)
    for rel in stale:
        problems.append(
            f"{rel}: changed since {ref[:8]} but {_SW_JS}'s CACHE version was not bumped "
            f"(still {working_cache!r}) - an already-installed PWA will keep serving the "
            "OLD cached copy forever, since the browser only re-installs a service worker "
            "when ITS OWN bytes change. Bump CACHE in sw.js and say why in the comment "
            "above it.")
    return problems


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
                    "precached and the installed PWA cannot open it offline. Add it to "
                    "SHELL and bump CACHE.")
    return problems


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
    tracked = _tracked_files()
    if not tracked:
        # `git ls-files` failed or returned nothing, so the dash/disclosure/abs-path
        # scan and the changelog gate below would run over ZERO files and this gate
        # would print "passed" having checked nothing. A disclosure/privacy gate that
        # reports clean without scanning anything is exactly the silent pass AGENTS.md
        # rule 5 forbids, so fail loud instead. (check_manifest keeps its documented
        # not-a-checkout silence as a LIBRARY; here, at the top-level gate, a checkout
        # with no enumerable tracked files is an error, not a benign no-op.)
        print("Hygiene check FAILED: could not enumerate tracked files via 'git ls-files' "
              "- nothing was scanned. Run this from a git checkout; a hygiene gate must "
              "not report clean without actually scanning the tree (AGENTS.md rule 5).",
              file=sys.stderr)
        return 1
    problems: list[str] = []
    for f in tracked:
        problems.extend(_scan(f))
    problems.extend(_changelog_append_only())
    problems.extend(_raw_accessor_violations(tracked))
    problems.extend(_big_test_write_violations(tracked))
    problems.extend(_sw_cache_bump_violations())
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
