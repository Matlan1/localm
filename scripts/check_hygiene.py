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
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]

# ---- check 3: absolute paths in code/config (escapable) --------------------
_CODE_EXTS = {".py", ".pyw", ".bat", ".cmd", ".ps1", ".sh", ".toml",
              ".json", ".cfg", ".ini"}
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


def main(argv: list[str]) -> int:
    if "--install-hook" in argv:
        return _install_hook()
    problems: list[str] = []
    for f in _tracked_files():
        problems.extend(_scan(f))
    if problems:
        print("Hygiene check FAILED (see AGENTS.md):\n", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} issue(s).", file=sys.stderr)
        return 1
    print("Hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
