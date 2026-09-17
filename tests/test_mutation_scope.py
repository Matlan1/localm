# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/mutation_scope.py - the merge-base diff check that
auto-activates the mutation-test CI job when a change touches a
[tool.mutmut] only_mutate module or the mutation gate itself.

The pure ``touched`` core is tested on lists; ``main`` is driven against a
throwaway git repository built in tmp_path, so the git plumbing (merge base,
diff, show at a ref) is exercised for real rather than mocked.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("mutation_scope")

MODULES = ["localm/scopes.py", "localm/auth.py"]
PYPROJECT = '[tool.mutmut]\nonly_mutate = ["localm/scopes.py", "localm/auth.py"]\n'


class TestTouched:
    def test_a_mutated_module_is_in_scope(self):
        assert ms.touched(["localm/auth.py", "README.md"], MODULES, False) == ["localm/auth.py"]

    def test_backslash_paths_match_the_forward_slash_list(self):
        assert ms.touched(["localm\\scopes.py"], MODULES, False) == ["localm/scopes.py"]

    def test_unrelated_changes_are_out_of_scope(self):
        assert ms.touched(["localm/inference/http_server.py", "tests/test_auth.py"], MODULES, False) == []

    @pytest.mark.parametrize("gate", ms.GATE_FILES)
    def test_every_gate_file_is_in_scope(self, gate):
        assert ms.touched([gate], MODULES, False) == [gate]

    def test_pyproject_counts_only_when_the_mutmut_table_changed(self):
        assert ms.touched(["pyproject.toml"], MODULES, False) == []
        assert ms.touched(["pyproject.toml"], MODULES, True) == ["pyproject.toml ([tool.mutmut] changed)"]

    def test_mutmut_table_flag_without_a_pyproject_change_is_ignored(self):
        assert ms.touched(["README.md"], MODULES, True) == []

    def test_mutmut_section_parses_and_tolerates_garbage(self):
        assert ms.mutmut_section(PYPROJECT) == {"only_mutate": MODULES}
        assert ms.mutmut_section("not = [toml") == {}
        assert ms.mutmut_section("[tool.other]\nx = 1\n") == {}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


@pytest.fixture
def repo(tmp_path):
    """A git repository with a `master` branch holding pyproject.toml and the
    two mutated modules, and HEAD on a `topic` branch one commit ahead."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "master")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "localm").mkdir()
    (r / "pyproject.toml").write_text(PYPROJECT + '[tool.other]\nx = 1\n', encoding="utf-8")
    for m in MODULES:
        (r / m).write_text("def f():\n    return 1\n", encoding="utf-8")
    (r / "README.md").write_text("hi\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    _git(r, "switch", "-q", "-c", "topic")
    return r


def _commit(repo: Path, rel: str, text: str) -> None:
    (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"touch {rel}")


class TestMain:
    def test_module_change_reports_touched_true(self, repo, capsys):
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(["--base", "master", "--repo", str(repo)]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "touched=true" and "localm/auth.py" in out

    def test_unrelated_change_reports_touched_false(self, repo, capsys):
        _commit(repo, "README.md", "changed\n")
        assert ms.main(["--base", "master", "--repo", str(repo)]) == 0
        assert capsys.readouterr().out.strip() == "touched=false"

    def test_unrelated_pyproject_edit_is_out_of_scope(self, repo, capsys):
        _commit(repo, "pyproject.toml", PYPROJECT + '[tool.other]\nx = 2\n')
        assert ms.main(["--base", "master", "--repo", str(repo)]) == 0
        assert capsys.readouterr().out.strip() == "touched=false"

    def test_mutmut_table_edit_is_in_scope(self, repo, capsys):
        _commit(repo, "pyproject.toml",
                '[tool.mutmut]\nonly_mutate = ["localm/scopes.py"]\n[tool.other]\nx = 1\n')
        assert ms.main(["--base", "master", "--repo", str(repo)]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "touched=true" and "[tool.mutmut] changed" in out

    def test_github_output_gets_the_verdict_line(self, repo, tmp_path, monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(["--base", "master", "--repo", str(repo), "--github-output"]) == 0
        assert out_file.read_text(encoding="utf-8") == "touched=true\n"

    def test_no_github_output_file_without_the_flag(self, repo, tmp_path, monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(["--base", "master", "--repo", str(repo)]) == 0
        assert not out_file.exists()

    def test_notice_is_printed_only_on_a_hit(self, repo, capsys):
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(["--base", "master", "--repo", str(repo), "--notice"]) == 0
        out = capsys.readouterr().out
        assert "::notice title=Mutation gate did not run::" in out and "localm/auth.py" in out
        assert "mutation-test label" in out
        _commit(repo, "README.md", "changed again\n")
        _git(repo, "reset", "-q", "--hard", "master")
        _commit(repo, "README.md", "docs only\n")
        assert ms.main(["--base", "master", "--repo", str(repo), "--notice"]) == 0
        assert "::notice" not in capsys.readouterr().out

    def test_unresolvable_base_exits_one_and_never_says_false(self, repo, capsys):
        """NEGATIVE: an unknown answer must not read as 'nothing touched'."""
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(["--base", "no-such-ref", "--repo", str(repo)]) == 1
        captured = capsys.readouterr()
        assert "touched=false" not in captured.out
        assert "could not diff" in captured.err
