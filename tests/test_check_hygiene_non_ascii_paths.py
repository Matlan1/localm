# SPDX-License-Identifier: AGPL-3.0-or-later
"""A tracked file with a non-ASCII name must still be scanned.

`git ls-files` renders such a path as an escaped, quoted string under the
default core.quotePath, so reading its output as text yields a name that
matches no file on disk. `_scan` treats a missing file as unreadable and
returns no problems, so checks 1-3 and 5 report clean on a file they never
opened, the disclosure and machine-path checks included, on a public repo.

Both special characters below are built with chr() rather than written as
literals: this file is itself scanned by the gate under test, which forbids
the em-dash outright.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NON_ASCII_NAME = "caf" + chr(0xE9) + ".py"
EM_DASH = chr(0x2014)


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene_nonascii", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=True)


@pytest.fixture()
def repo_with_non_ascii_file(tmp_path):
    """A git repo holding one ASCII and one non-ASCII tracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", ".")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "plain.py").write_text("x = 1\n", encoding="utf-8")
    (repo / NON_ASCII_NAME).write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    return repo


def test_git_ls_files_quotes_a_non_ascii_path_without_z(repo_with_non_ascii_file):
    """The precondition. Without -z git escapes the name, so the string it
    returns names no file. If this stops being true the tests below are no
    longer exercising anything."""
    out = subprocess.run(["git", "ls-files"], cwd=repo_with_non_ascii_file,
                         capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    quoted = [r for r in out.splitlines() if r.startswith('"')]
    assert quoted, f"expected a quoted path, got {out.splitlines()}"
    assert not (repo_with_non_ascii_file / quoted[0]).exists()


def test_non_ascii_tracked_path_is_scanned(monkeypatch, repo_with_non_ascii_file):
    """_tracked_files must return a path that RESOLVES, for every tracked file."""
    mod = _load_check_hygiene()
    monkeypatch.setattr(mod, "REPO", repo_with_non_ascii_file)

    files = mod._tracked_files()
    names = sorted(p.name for p in files)

    # Assert on the WORLD first: a path that does not resolve is a file no
    # check ever opens, which is the whole defect.
    unreadable = [str(p) for p in files if not p.exists()]
    assert unreadable == [], f"_tracked_files returned unresolvable paths: {unreadable}"
    assert NON_ASCII_NAME in names, (
        f"the non-ASCII tracked file was dropped; got {names}")
    assert "plain.py" in names


def test_a_violation_in_a_non_ascii_file_is_reported(monkeypatch,
                                                     repo_with_non_ascii_file):
    """End to end: an em-dash inside the non-ASCII file must be found."""
    mod = _load_check_hygiene()
    monkeypatch.setattr(mod, "REPO", repo_with_non_ascii_file)
    (repo_with_non_ascii_file / NON_ASCII_NAME).write_text(
        "# a " + EM_DASH + " dash\n", encoding="utf-8")

    problems = []
    for path in mod._tracked_files():
        problems.extend(mod._scan(path))

    assert any("em-dash" in p for p in problems), (
        f"the em-dash in the non-ASCII file was not reported: {problems}")
