# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for scripts/mutation_scope.py - which mutation shards a run needs
(the mutation-scope CI job) and the test-file map that decision reads.

The pure ``select`` / ``build_map`` / ``check_map`` cores are tested on lists
and dicts; ``main`` is driven against a throwaway git repository built in
tmp_path, so the git plumbing (merge base, diff, show at a ref) is exercised
for real rather than mocked. The committed map is checked against the real
pyproject.toml at the end.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
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
SELECTION = ["tests/test_scopes.py", "tests/test_auth.py", "tests/sub/test_deep.py"]
TEST_MAP = {"tests/test_scopes.py": ["localm/scopes.py"],
            "tests/test_auth.py": ["localm/scopes.py", "localm/auth.py"],
            "tests/sub/test_deep.py": ["localm/auth.py"]}
SOURCES = {"tests/test_scopes.py": "import localm.scopes\n",
           "tests/test_auth.py": "from tests._helper import thing\nPath('fixtures/keys')\n",
           "tests/sub/test_deep.py": "x = 1\n",
           "tests/conftest.py": "from tests._hooks import pytest_runtest_makereport  # noqa\n",
           "tests/_hooks.py": "def pytest_runtest_makereport():\n    from tests import _deeper\n",
           "tests/_deeper.py": "x = 1\n"}
WORKFLOW = ("on: push\njobs:\n  lint:\n    runs-on: ubuntu-latest\n"
            "  mutation-run:\n    runs-on: ubuntu-latest\n    timeout-minutes: 180\n"
            "  docs:\n    runs-on: ubuntu-latest\n")
PYPROJECT = ('[tool.mutmut]\nonly_mutate = ["localm/scopes.py", "localm/auth.py"]\n'
             'pytest_add_cli_args_test_selection = ["tests/test_scopes.py", '
             '"tests/test_auth.py", "tests/sub/test_deep.py"]\n')


def _select(changed, config_changed=False, test_map=None):
    return ms.select(changed, MODULES, SELECTION, TEST_MAP if test_map is None else test_map,
                     config_changed, lambda rel: SOURCES.get(rel, ""))


class TestSelect:
    def test_a_mutated_module_selects_only_itself(self):
        picked, reasons = _select(["localm/auth.py", "README.md"])
        assert picked == ["localm/auth.py"]
        assert reasons == ["auth: localm/auth.py changed"]

    def test_backslash_paths_match_the_forward_slash_lists(self):
        assert _select(["localm\\scopes.py"])[0] == ["localm/scopes.py"]

    def test_unrelated_changes_select_nothing(self):
        assert _select(["localm/inference/http_server.py", "README.md"]) == ([], [])

    def test_a_test_outside_the_selection_selects_nothing(self):
        assert _select(["tests/test_unrelated.py", "tests/plugins/test_x.py"]) == ([], [])

    def test_a_selection_test_selects_the_modules_the_map_gives_it(self):
        assert _select(["tests/test_scopes.py"])[0] == ["localm/scopes.py"]
        assert _select(["tests/sub/test_deep.py"])[0] == ["localm/auth.py"]

    def test_the_selection_is_in_only_mutate_order(self):
        assert _select(["localm/auth.py", "tests/test_scopes.py"])[0] == MODULES

    @pytest.mark.parametrize("gate", ms.GATE_FILES)
    def test_every_gate_file_selects_every_module(self, gate):
        picked, reasons = _select([gate])
        assert picked == MODULES
        assert reasons == [f"every module: {gate} changed"]

    def test_the_map_file_is_a_gate_file(self):
        assert ms.MAP_FILE in ms.GATE_FILES

    def test_a_mutmut_table_change_selects_every_module(self):
        assert _select(["pyproject.toml"], config_changed=True)[0] == MODULES
        assert _select(["pyproject.toml"], config_changed=False) == ([], [])

    def test_the_root_conftest_selects_every_module(self):
        picked, reasons = _select(["tests/conftest.py"])
        assert picked == MODULES and reasons == ["every module: tests/conftest.py changed"]

    def test_a_nested_conftest_selects_the_modules_of_the_selection_under_it(self):
        assert _select(["tests/sub/conftest.py"])[0] == ["localm/auth.py"]
        assert _select(["tests/other/conftest.py"]) == ([], [])

    def test_a_helper_selects_the_modules_of_the_selection_files_naming_it(self):
        picked, reasons = _select(["tests/_helper.py"])
        assert picked == MODULES
        assert reasons == ["scopes, auth: tests/_helper.py changed (reaches tests/test_auth.py)"]
        assert _select(["tests/_unused_helper.py"]) == ([], [])

    def test_a_fixture_selects_by_its_first_directory(self):
        assert _select(["tests/fixtures/keys/a.pem"])[0] == MODULES
        assert _select(["tests/fixtures/media/a.png"]) == ([], [])

    def test_a_name_must_match_whole_not_as_a_substring(self):
        assert _select(["tests/_help.py"]) == ([], [])

    def test_a_selection_file_missing_from_the_map_selects_every_module(self):
        """NEGATIVE: an unknown edge set is never read as 'no module'."""
        partial = {k: v for k, v in TEST_MAP.items() if k != "tests/sub/test_deep.py"}
        picked, reasons = _select(["tests/sub/test_deep.py"], test_map=partial)
        assert picked == MODULES
        assert "not in scripts/mutation_test_map.json" in reasons[0]

    def test_a_helper_reaching_an_unmapped_selection_file_selects_every_module(self):
        partial = {k: v for k, v in TEST_MAP.items() if k != "tests/test_auth.py"}
        picked, reasons = _select(["tests/_helper.py"], test_map=partial)
        assert picked == MODULES and "reaches tests/test_auth.py" in reasons[0]


class TestEveryModuleTriggers:
    def test_a_module_the_root_conftest_imports_selects_every_module(self):
        picked, reasons = _select(["tests/_hooks.py"])
        assert picked == MODULES and reasons == ["every module: tests/_hooks.py changed"]

    def test_the_conftest_import_chain_is_followed(self):
        """NEGATIVE: a helper the conftest reaches only through another helper
        still runs on every test, so it is never 'no module'."""
        assert ms.conftest_imports(lambda rel: SOURCES.get(rel, "")) == [
            "tests/_deeper.py", "tests/_hooks.py", "tests/conftest.py"]
        assert _select(["tests/_deeper.py"])[0] == MODULES

    def test_a_workflow_change_selects_every_module_only_in_a_mutation_job(self):
        rt = lambda rel: SOURCES.get(rel, "")  # noqa: E731
        assert ms.select([ms.WORKFLOW], MODULES, SELECTION, TEST_MAP, False, rt, True)[0] == MODULES
        assert ms.select([ms.WORKFLOW], MODULES, SELECTION, TEST_MAP, False, rt, False) == ([], [])

    def test_mutation_jobs_text_is_only_the_mutation_jobs(self):
        text = ms.mutation_jobs_text(WORKFLOW)
        assert text == "  mutation-run:\n    runs-on: ubuntu-latest\n    timeout-minutes: 180"
        assert ms.mutation_jobs_text(WORKFLOW.replace("docs:", "docs2:")) == text
        assert ms.mutation_jobs_text(WORKFLOW.replace("180", "170")) != text

    def test_the_mutmut_pin_is_a_gate_file(self):
        assert ".github/requirements/mutmut.txt" in ms.GATE_FILES


STATS = {"tests_by_mangled_function_name": {
    "localm.scopes.x_grants": ["tests/test_scopes.py::test_a", "tests/test_auth.py::test_b"],
    "localm.auth.x_verify": ["tests/test_auth.py::TestX::test_c[1]"],
    "localm.other.x_f": ["tests/test_scopes.py::test_a"],
    "localm.netpolicy.xǁ_HTMLStripperǁhandle_endtag": ["tests/test_auth.py::test_b"],
}}


class TestMap:
    def test_build_map_lists_every_selection_file_and_only_mutated_modules(self):
        built = ms.build_map([STATS], SELECTION, MODULES)
        assert built == {"schema": 1, "test_files": {
            "tests/sub/test_deep.py": [],
            "tests/test_auth.py": ["localm/scopes.py", "localm/auth.py"],
            "tests/test_scopes.py": ["localm/scopes.py"]}}

    def test_a_class_method_maps_to_its_module(self):
        assert ms.module_of_function("localm.netpolicy.xǁ_HTMLStripperǁhandle_endtag") \
            == "localm/netpolicy.py"

    def test_build_map_is_the_union_of_every_stats_file(self):
        other = {"tests_by_mangled_function_name": {
            "localm.auth.x_verify": ["tests/sub/test_deep.py::test_d"]}}
        built = ms.build_map([STATS, other], SELECTION, MODULES)
        assert built["test_files"]["tests/sub/test_deep.py"] == ["localm/auth.py"]

    def test_check_passes_on_its_own_build_and_on_a_superset(self):
        built = ms.build_map([STATS], SELECTION, MODULES)
        assert ms.check_map(built, built) == []
        wider = json.loads(json.dumps(built))
        wider["test_files"]["tests/sub/test_deep.py"] = ["localm/auth.py"]
        assert ms.check_map(wider, built) == []

    def test_check_names_a_missing_edge(self):
        """NEGATIVE: a recorded edge the map lacks is a failure."""
        built = ms.build_map([STATS], SELECTION, MODULES)
        stale = json.loads(json.dumps(built))
        stale["test_files"]["tests/test_auth.py"] = ["localm/scopes.py"]
        assert ms.check_map(stale, built) == [
            "scripts/mutation_test_map.json: tests/test_auth.py exercises localm/auth.py, "
            "which the map does not list"]

    def test_check_names_files_missing_or_extra(self):
        built = ms.build_map([STATS], SELECTION, MODULES)
        odd = json.loads(json.dumps(built))
        del odd["test_files"]["tests/sub/test_deep.py"]
        odd["test_files"]["tests/test_gone.py"] = []
        problems = ms.check_map(odd, built)
        assert any("tests/sub/test_deep.py is in the mutmut test selection but not in the map" in p
                   for p in problems)
        assert any("tests/test_gone.py is in the map but not in the mutmut test selection" in p
                   for p in problems)

    @pytest.mark.parametrize("committed", [None, {}, {"schema": 2, "test_files": {}},
                                           {"schema": 1, "test_files": []}])
    def test_check_fails_closed_on_a_malformed_map(self, committed):
        assert ms.check_map(committed, ms.build_map([STATS], SELECTION, MODULES))

    def test_merge_keeps_committed_edges_of_files_still_selected(self):
        built = ms.build_map([STATS], SELECTION, MODULES)
        kept = {"tests/sub/test_deep.py": ["localm/scopes.py"], "tests/test_gone.py": ["localm/auth.py"]}
        merged = ms.merge_maps(built, kept, MODULES)
        assert merged["test_files"]["tests/sub/test_deep.py"] == ["localm/scopes.py"]
        assert "tests/test_gone.py" not in merged["test_files"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


@pytest.fixture
def repo(tmp_path):
    """A git repository with a `master` branch holding pyproject.toml, the two
    mutated modules, the selection tests and the map, and HEAD on a `topic`
    branch."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "master")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "localm").mkdir()
    (r / "scripts").mkdir()
    (r / "tests" / "sub").mkdir(parents=True)
    (r / "pyproject.toml").write_text(PYPROJECT + '[tool.other]\nx = 1\n', encoding="utf-8")
    for m in MODULES:
        (r / m).write_text("def f():\n    return 1\n", encoding="utf-8")
    for rel, text in SOURCES.items():
        (r / rel).write_text(text, encoding="utf-8")
    (r / ms.MAP_FILE).write_text(json.dumps({"schema": 1, "test_files": TEST_MAP}), encoding="utf-8")
    (r / ".github" / "workflows").mkdir(parents=True)
    (r / ms.WORKFLOW).write_text(WORKFLOW, encoding="utf-8")
    (r / "README.md").write_text("hi\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    _git(r, "switch", "-q", "-c", "topic")
    return r


def _commit(repo: Path, rel: str, text: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"touch {rel}")


def _run(repo: Path, *extra: str) -> list[str]:
    return ["--base", "master", "--repo", str(repo), *extra]


class TestMain:
    def test_module_change_selects_that_module(self, repo, capsys):
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(_run(repo)) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == 'modules=["auth"]'
        assert json.loads(out[1].removeprefix("shards=")) == ms.mutation_parts.shards(["auth"])
        assert out[2].strip() == "auth: localm/auth.py changed"

    def test_unrelated_change_selects_nothing(self, repo, capsys):
        _commit(repo, "README.md", "changed\n")
        assert ms.main(_run(repo)) == 0
        assert capsys.readouterr().out.splitlines() == ["modules=[]", "shards=[]"]

    def test_a_selection_test_change_selects_its_mapped_modules(self, repo, capsys):
        _commit(repo, "tests/test_scopes.py", "import localm.scopes\n# more\n")
        assert ms.main(_run(repo)) == 0
        assert capsys.readouterr().out.splitlines()[0] == 'modules=["scopes"]'

    def test_a_helper_change_is_read_against_the_selection_at_head(self, repo, capsys):
        _commit(repo, "tests/_helper.py", "thing = 2\n")
        assert ms.main(_run(repo)) == 0
        assert capsys.readouterr().out.splitlines()[0] == 'modules=["scopes","auth"]'

    def test_unrelated_pyproject_edit_selects_nothing(self, repo, capsys):
        _commit(repo, "pyproject.toml", PYPROJECT + '[tool.other]\nx = 2\n')
        assert ms.main(_run(repo)) == 0
        assert capsys.readouterr().out.splitlines() == ["modules=[]", "shards=[]"]

    def test_mutmut_table_edit_selects_every_module(self, repo, capsys):
        _commit(repo, "pyproject.toml", PYPROJECT.replace('"localm/auth.py"]', '"localm/auth.py"]\nx = 1')
                + '[tool.other]\nx = 1\n')
        assert ms.main(_run(repo)) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == 'modules=["scopes","auth"]' and "[tool.mutmut] changed" in out

    def test_a_mutation_job_edit_selects_every_module_and_another_job_edit_nothing(self, repo, capsys):
        _commit(repo, ms.WORKFLOW, WORKFLOW.replace("lint:", "lint2:"))
        assert ms.main(_run(repo)) == 0
        assert capsys.readouterr().out.splitlines()[0] == "modules=[]"
        _commit(repo, ms.WORKFLOW, WORKFLOW.replace("lint:", "lint2:").replace("180", "170"))
        assert ms.main(_run(repo)) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == 'modules=["scopes","auth"]' and "a mutation-* job" in out

    def test_without_the_label_a_touched_module_runs_no_shard(self, repo, capsys):
        """NEGATIVE: an unlabelled pull request never starts a mutation shard."""
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(_run(repo, "--run", "false", "--notice")) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[:2] == ["modules=[]", "shards=[]"]
        notice = [ln for ln in out if ln.startswith("::notice")]
        assert len(notice) == 1
        assert notice[0].startswith("::notice title=Mutation shards did not run::")
        assert "auth" in notice[0] and "mutation-test label" in notice[0]

    def test_without_the_label_github_output_is_empty(self, repo, tmp_path, monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(_run(repo, "--run", "false", "--github-output")) == 0
        assert out_file.read_text(encoding="utf-8") == "modules=[]\nshards=[]\n"

    def test_the_label_runs_only_the_touched_modules(self, repo, capsys):
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(_run(repo, "--run", "true")) == 0
        assert capsys.readouterr().out.splitlines()[0] == 'modules=["scopes"]'

    def test_all_true_runs_every_module_even_without_the_label(self, repo, capsys):
        assert ms.main(["--repo", str(repo), "--all", "true", "--run", "false"]) == 0
        assert capsys.readouterr().out.splitlines()[0] == 'modules=["scopes","auth"]'

    def test_all_true_selects_every_module_without_a_diff(self, repo, capsys):
        assert ms.main(["--base", "no-such-ref", "--repo", str(repo), "--all", "true"]) == 0
        assert capsys.readouterr().out.splitlines()[0] == 'modules=["scopes","auth"]'

    def test_github_output_gets_the_modules_line(self, repo, tmp_path, monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(_run(repo, "--github-output")) == 0
        assert out_file.read_text(encoding="utf-8") == (
            'modules=["scopes"]\n'
            'shards=[{"module":"scopes","part":1,"parts":1,"pattern":"localm.scopes.*"}]\n')

    def test_github_output_gets_an_empty_list_when_nothing_is_needed(self, repo, tmp_path,
                                                                     monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "README.md", "docs\n")
        assert ms.main(_run(repo, "--github-output")) == 0
        assert out_file.read_text(encoding="utf-8") == "modules=[]\nshards=[]\n"

    def test_no_github_output_file_without_the_flag(self, repo, tmp_path, monkeypatch, capsys):
        out_file = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(_run(repo)) == 0
        assert not out_file.exists()

    def test_notice_names_the_shards_only_when_some_run(self, repo, capsys):
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(_run(repo, "--notice")) == 0
        notice = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("::notice")]
        assert len(notice) == 1
        assert notice[0].startswith("::notice title=Mutation shards run on this change::auth (")
        _git(repo, "reset", "-q", "--hard", "master")
        _commit(repo, "README.md", "docs only\n")
        assert ms.main(_run(repo, "--notice")) == 0
        assert "::notice" not in capsys.readouterr().out

    def test_unresolvable_base_exits_one_and_never_says_empty(self, repo, capsys):
        """NEGATIVE: an unknown answer must not read as 'nothing needed'."""
        _commit(repo, "localm/scopes.py", "def f():\n    return 3\n")
        assert ms.main(["--base", "no-such-ref", "--repo", str(repo)]) == 1
        captured = capsys.readouterr()
        assert "modules=" not in captured.out
        assert "could not diff" in captured.err

    def test_missing_map_exits_one_and_never_says_empty(self, repo, capsys):
        """NEGATIVE: without the map a test-only change cannot be judged."""
        (repo / ms.MAP_FILE).unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "drop map")
        assert ms.main(_run(repo)) == 1
        captured = capsys.readouterr()
        assert "modules=" not in captured.out
        assert ms.MAP_FILE in captured.err

    def test_missing_mutmut_table_exits_one_and_never_says_empty(self, repo, capsys):
        _commit(repo, "pyproject.toml", "[tool.other]\nx = 1\n")
        _commit(repo, "localm/auth.py", "def f():\n    return 2\n")
        assert ms.main(_run(repo)) == 1
        captured = capsys.readouterr()
        assert "modules=" not in captured.out
        assert "only_mutate" in captured.err

    def test_empty_only_mutate_exits_one_and_never_says_empty(self, repo, capsys):
        _commit(repo, "pyproject.toml", "[tool.mutmut]\nonly_mutate = []\n[tool.other]\nx = 1\n")
        assert ms.main(_run(repo, "--all", "true")) == 1
        captured = capsys.readouterr()
        assert "modules=" not in captured.out
        assert "only_mutate" in captured.err

    def test_write_map_then_check_map_round_trip(self, repo, tmp_path, capsys):
        stats = tmp_path / "stats.json"
        stats.write_text(json.dumps(STATS), encoding="utf-8")
        out = tmp_path / "proposed.json"
        assert ms.main(["--repo", str(repo), "--write-map", str(out), "--stats", str(stats)]) == 0
        proposed = json.loads(out.read_text(encoding="utf-8"))
        assert proposed["test_files"]["tests/sub/test_deep.py"] == ["localm/auth.py"], (
            "the committed map's edge is kept")
        assert ms.main(["--repo", str(repo), "--check-map", "--stats", str(stats)]) == 0

    def test_check_map_exits_one_on_a_missing_edge(self, repo, tmp_path, capsys):
        stats = tmp_path / "stats.json"
        wider = json.loads(json.dumps(STATS))
        wider["tests_by_mangled_function_name"]["localm.auth.x_verify"].append(
            "tests/test_scopes.py::test_new")
        stats.write_text(json.dumps(wider), encoding="utf-8")
        assert ms.main(["--repo", str(repo), "--check-map", "--stats", str(stats)]) == 1
        assert "tests/test_scopes.py exercises localm/auth.py" in capsys.readouterr().err

    def test_map_commands_need_stats(self, repo, capsys):
        with pytest.raises(SystemExit) as exc:
            ms.main(["--repo", str(repo), "--check-map"])
        assert exc.value.code == 2


class TestCommittedMap:
    """scripts/mutation_test_map.json against the real pyproject.toml."""

    def test_lists_exactly_the_selection_and_only_mutated_modules(self):
        section = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mutmut"]
        data = json.loads((REPO_ROOT / ms.MAP_FILE).read_text(encoding="utf-8"))
        assert data["schema"] == ms.MAP_SCHEMA
        assert set(data["test_files"]) == set(section["pytest_add_cli_args_test_selection"])
        modules = set(section["only_mutate"])
        for f, mods in data["test_files"].items():
            assert set(mods) <= modules, f
            assert len(mods) == len(set(mods)), f
