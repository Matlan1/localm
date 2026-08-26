# SPDX-License-Identifier: AGPL-3.0-or-later
"""No tracked runtime string literal names a gitignored internal directory.

An exception, a log line, a CLI print, a pytest skip reason, or an assert
message that reads "(see dev-notes/foo.md)" or "(issues/bar.txt)" points a
reader at a file that exists only on the maintainer's machine (AGENTS.md
rule 6) - this repo is public. Five such sites shipped: a
click.ClickException in setup_llama.py, a logger.debug call in
gpu_usage.py, a print() in check_llama_abi.py, a pytest skip reason in
conftest.py, and an assert message in test_gpu_split_native_vulkan.py.

Scoped to non-docstring string constants. A module/class/function docstring
is documentation - never printed, logged, raised, or asserted at runtime -
so a dangling internal-path reference there is a different, separately
tracked problem from a string that actually ships in output a user or a
public CI log can show a stranger.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_PATTERNS = {
    "dev-notes/": re.compile(r"dev-notes/"),
    "qa/": re.compile(r"qa/"),
    # A GitHub issue URL/API path names issues/<number> (e.g.
    # "https://github.com/x/y/issues/9"); a pointer into the local,
    # gitignored issues/ directory never does - it names a filename or a tag.
    "issues/": re.compile(r"issues/(?!\d)"),
}


def _docstring_constant_ids(tree: ast.AST) -> set:
    """id() of every Constant node the interpreter treats as a docstring."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def find_leaks(source: str, filename: str = "<string>") -> list:
    """(lineno, pattern, value) for every non-docstring string constant in
    *source* that names a gitignored local directory."""
    tree = ast.parse(source, filename=filename)
    skip = _docstring_constant_ids(tree)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in skip:
            continue
        for name, pattern in _PATTERNS.items():
            if pattern.search(node.value):
                hits.append((node.lineno, name, node.value))
    return hits


_SELF = Path(__file__).resolve().relative_to(REPO).as_posix()


def _tracked_python_files() -> list:
    """Every tracked *.py file except this one."""
    out = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return [rel for rel in out.splitlines() if rel.strip() and rel != _SELF]


# --------------------------------------------------------------------------- #
#  Fires-control for the instrument, run BEFORE trusting its clean result      #
# --------------------------------------------------------------------------- #

def test_scan_flags_a_planted_runtime_string():
    hits = find_leaks('raise Exception("boom (dev-notes/ADR-0010)")\n')
    assert hits == [(1, "dev-notes/", "boom (dev-notes/ADR-0010)")]


def test_scan_flags_a_planted_fstring_segment():
    """The real setup_llama.py site is an f-string, not a plain str literal -
    the scan must see literal text INSIDE a JoinedStr, not only bare
    Constant strings, or it proves nothing about that site."""
    hits = find_leaks('raise Exception(f"boom {x!r} (dev-notes/ADR-0010)")\n')
    assert any(name == "dev-notes/" for _, name, _ in hits)


def test_scan_ignores_a_function_docstring():
    hits = find_leaks(
        'def f():\n'
        '    """See dev-notes/foo.md for why."""\n'
        '    return 1\n'
    )
    assert hits == []


def test_scan_ignores_a_module_docstring():
    hits = find_leaks('"""See issues/foo.txt."""\n\nx = 1\n')
    assert hits == []


def test_scan_ignores_a_class_docstring():
    hits = find_leaks(
        'class C:\n'
        '    """See qa/plan.md."""\n'
    )
    assert hits == []


def test_scan_ignores_bare_directory_name_without_trailing_slash():
    """Mirrors localm/_apply_update.py's NEVER_TOUCH frozenset: {"dev-notes",
    "issues", "qa"} name the directories an update swap must exclude, with no
    trailing slash and no path - not a dangling doc pointer."""
    hits = find_leaks('NEVER_TOUCH = frozenset({"dev-notes", "issues", "qa"})\n')
    assert hits == []


def test_scan_ignores_a_github_issue_url():
    """Mirrors the many test fixtures mocking a bug-report response, e.g.
    "https://github.com/Matlan1/localm/issues/999" - a real, public URL, not
    a pointer into the local, gitignored issues/ directory."""
    hits = find_leaks('URL = "https://github.com/Matlan1/localm/issues/999"\n')
    assert hits == []


def test_scan_still_flags_a_non_numeric_issues_reference():
    """The github-URL carve-out must not swallow a genuine local pointer that
    merely happens to share the "issues/" prefix."""
    hits = find_leaks('MSG = "see issues/abi-verification-design.md"\n')
    assert hits == [(1, "issues/", "see issues/abi-verification-design.md")]


# --------------------------------------------------------------------------- #
#  The two carve-out sites named in review, confirmed clean under this scan   #
# --------------------------------------------------------------------------- #

def test_apply_update_never_touch_list_is_not_flagged():
    path = REPO / "localm" / "_apply_update.py"
    hits = find_leaks(path.read_text(encoding="utf-8"), filename=str(path))
    assert hits == []


def test_bugreport_upload_fixture_urls_are_not_flagged():
    path = REPO / "tests" / "test_bugreport_upload.py"
    hits = find_leaks(path.read_text(encoding="utf-8"), filename=str(path))
    assert hits == []


# --------------------------------------------------------------------------- #
#  The real assertion                                                         #
# --------------------------------------------------------------------------- #

def test_no_tracked_python_file_leaks_a_gitignored_path_in_a_runtime_string():
    all_hits = []
    for rel in _tracked_python_files():
        path = REPO / rel
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            hits = find_leaks(source, filename=rel)
        except SyntaxError:
            continue
        for lineno, name, value in hits:
            all_hits.append(f"{rel}:{lineno}  [{name}]  {value!r}")
    assert not all_hits, (
        "a tracked Python file's runtime string (not a docstring) names a "
        "gitignored internal path that does not exist outside the "
        "maintainer's machine (AGENTS.md rule 6):\n" + "\n".join(all_hits))
