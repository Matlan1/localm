# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/affected_tests.py: the test files a change affects, computed from
the import graph and from the names (route paths, module names, file names)
a test spells out.

These tests pin each selection rule against a throwaway git checkout: a
changed test file, a direct import, an importer N hops away (only with
--depth), a route called by URL, a module named as a string, a script or a
non-Python file named by file name, a route that only the committed version
still registers, and the files that affect everything. The command line's
output shape and its wide-selection exit status are pinned too. The last
section binds the selector to the real tree: a change to the route whose
dropped field motivated it selects the test that reads that route.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "affected_tests.py"


def _load():
    spec = importlib.util.spec_from_file_location("affected_tests", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, env={**os.environ, **_GIT_ENV}).stdout


_FILES = {
    "localm/__init__.py": "",
    "localm/a.py": "from localm import b\n",
    "localm/b.py": "def func():\n    return 1\n",
    "localm/routes.py": (
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n"
        "@app.get('/api/thing/status')\n"
        "async def status():\n    return {'ok': True}\n\n\n"
        "@app.get('/api/things/{name}')\n"
        "async def one(name):\n    return {'name': name}\n"),
    "localm/static/index.html": "<html></html>\n",
    "scripts/tool.py": "print('tool')\n",
    "tests/conftest.py": "",
    "tests/_helper.py": "VALUE = 1\n",
    "tests/sub/conftest.py": "",
    "tests/sub/test_sub.py": "def test_sub():\n    assert True\n",
    "tests/test_helper_user.py": (
        "from _helper import VALUE\n\n\ndef test_helper():\n    assert VALUE == 1\n"),
    "tests/test_a.py": "import localm.a\n\n\ndef test_a():\n    assert localm.a\n",
    "tests/test_b.py": "from localm.b import func\n\n\ndef test_b():\n    assert func() == 1\n",
    "tests/test_url.py": (
        "def test_url(client):\n"
        "    assert client.get('/api/thing/status').json()['ok']\n"),
    "tests/test_param.py": (
        "def test_param(client):\n"
        "    assert client.get('/api/things/x').json()['name'] == 'x'\n"),
    "tests/test_patch.py": (
        "def test_patch(monkeypatch):\n"
        "    monkeypatch.setattr('localm.b.func', lambda: 2)\n"),
    "tests/test_html.py": (
        "def test_html():\n"
        "    assert 'index.html'\n"),
    "tests/test_tool.py": (
        "def test_tool():\n"
        "    assert 'tool.py'\n"),
    "tests/test_none.py": "def test_none():\n    assert True\n",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A committed throwaway checkout with the files above; the module's REPO
    points at it."""
    for rel, src in _FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    mod = _load()
    monkeypatch.setattr(mod, "REPO", tmp_path)
    return mod, tmp_path


def _select(mod, changed, depth=0, base_ref="HEAD"):
    return mod.select(changed, mod.Graph(), depth=depth, base_ref=base_ref)


# --------------------------------------------------------------------------- #
#  The selection rules                                                        #
# --------------------------------------------------------------------------- #

def test_a_changed_test_file_selects_itself(repo):
    mod, _ = repo
    assert _select(mod, ["tests/test_none.py"]) == {"tests/test_none.py": ["changed"]}


def test_a_direct_importer_is_selected_and_a_transitive_one_only_with_depth(repo):
    mod, _ = repo
    at_zero = _select(mod, ["localm/b.py"])
    assert "tests/test_b.py" in at_zero and "imports localm.b" in at_zero["tests/test_b.py"]
    assert "tests/test_a.py" not in at_zero
    at_one = _select(mod, ["localm/b.py"], depth=1)
    assert at_one["tests/test_a.py"] == ["imports localm.a (1 hop(s) from a change)"]


def test_a_route_called_by_url_selects_the_test_without_an_import(repo):
    mod, _ = repo
    selected = _select(mod, ["localm/routes.py"])
    assert selected["tests/test_url.py"] == ["names route /api/thing/status"]
    assert selected["tests/test_param.py"] == ["names route /api/things/"]
    assert "tests/test_none.py" not in selected


def test_a_module_named_as_a_string_is_selected(repo):
    mod, _ = repo
    selected = _select(mod, ["localm/b.py"])
    assert selected["tests/test_patch.py"] == ["names localm.b"]


def test_a_tests_helper_module_selects_its_importers_and_a_sub_conftest_its_folder(repo):
    mod, _ = repo
    assert _select(mod, ["tests/_helper.py"]) == {
        "tests/test_helper_user.py": ["imports tests._helper"]}
    assert _select(mod, ["tests/sub/conftest.py"]) == {
        "tests/sub/test_sub.py": ["every test file under tests/sub/: its conftest.py changed"]}


def test_a_script_and_a_non_python_file_select_by_file_name(repo):
    mod, _ = repo
    assert _select(mod, ["scripts/tool.py"]) == {"tests/test_tool.py": ["names tool.py"]}
    assert _select(mod, ["localm/static/index.html"]) == {
        "tests/test_html.py": ["names index.html"]}


def test_a_route_only_the_committed_version_registers_still_matches(repo):
    mod, root = repo
    (root / "localm" / "routes.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8")
    selected = _select(mod, ["localm/routes.py"], base_ref="HEAD")
    assert "tests/test_url.py" in selected and "tests/test_param.py" in selected


def test_conftest_pyproject_and_lock_affect_every_test(repo):
    mod, _ = repo
    for rel in ("tests/conftest.py", "pyproject.toml", "uv.lock"):
        selected = _select(mod, [rel])
        assert set(selected) == set(mod.Graph().test_files) and len(selected) == 10, rel
        assert all(r == [f"every test file: {rel} changed"] for r in selected.values())


def test_changed_files_reads_committed_staged_unstaged_and_untracked(repo):
    mod, root = repo
    _git(root, "checkout", "-qb", "topic")
    (root / "localm" / "b.py").write_text("def func():\n    return 2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "change b")
    (root / "localm" / "a.py").write_text("from localm import b  # touched\n", encoding="utf-8")
    (root / "tests" / "test_new.py").write_text("def test_new():\n    pass\n", encoding="utf-8")
    _git(root, "branch", "-q", "main-copy", "HEAD~1")
    changed, base = mod.changed_files("main-copy")
    assert changed == ["localm/a.py", "localm/b.py", "tests/test_new.py"]
    assert base == _git(root, "rev-parse", "HEAD~1").strip()


def test_changed_files_falls_back_to_head_when_the_base_is_unknown(repo):
    mod, root = repo
    (root / "localm" / "a.py").write_text("# touched\n", encoding="utf-8")
    changed, base = mod.changed_files("no/such/ref")
    assert changed == ["localm/a.py"] and base == "HEAD"


# --------------------------------------------------------------------------- #
#  The command line                                                           #
# --------------------------------------------------------------------------- #

def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    """The script's main() in a fresh interpreter with REPO pointed at *root*."""
    driver = ("import importlib.util, sys\n"
              f"spec = importlib.util.spec_from_file_location('m', {str(_SCRIPT)!r})\n"
              "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
              "from pathlib import Path\n"
              f"m.REPO = Path({str(root)!r})\n"
              "raise SystemExit(m.main(sys.argv[1:]))")
    return subprocess.run([sys.executable, "-c", driver, *args], capture_output=True,
                          text=True, env={**os.environ, **_GIT_ENV})


def test_cli_lists_paths_one_per_line_and_reasons_on_request(repo):
    _, root = repo
    out = _run_cli(root, "--files", "localm/routes.py", "--why")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "tests/test_param.py  # names route /api/things/",
        "tests/test_url.py  # names route /api/thing/status",
    ]
    assert "2 of 10 test files affected by 1 changed file(s)" in out.stderr


def test_cli_output_is_lf_terminated_and_never_empty(repo):
    _, root = repo
    out = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util, sys\n"
         f"spec = importlib.util.spec_from_file_location('m', {str(_SCRIPT)!r})\n"
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
         "from pathlib import Path\n"
         f"m.REPO = Path({str(root)!r})\n"
         "raise SystemExit(m.main(sys.argv[1:]))",
         "--files", "localm/routes.py"],
        capture_output=True, env={**os.environ, **_GIT_ENV})
    assert out.returncode == 0
    assert out.stdout == b"tests/test_param.py\ntests/test_url.py\n"
    nothing = _run_cli(root, "--files", "README.md")
    assert nothing.returncode == 0
    assert nothing.stdout == "tests/NO_TEST_FILE_IS_AFFECTED\n"
    assert not (root / "tests" / "NO_TEST_FILE_IS_AFFECTED").exists()
    assert "0 of 10 test files affected" in nothing.stderr


def test_cli_prints_a_refusing_sentinel_and_exits_3_on_a_wide_selection(repo):
    _, root = repo
    out = _run_cli(root, "--files", "tests/conftest.py")
    assert out.returncode == 3, out.stderr
    assert out.stdout == "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR\n"
    assert "WIDE: 100% of the suite exceeds --max-share 25%" in out.stderr
    assert "--list-wide prints the selection" in out.stderr
    listed = _run_cli(root, "--files", "tests/conftest.py", "--list-wide")
    assert listed.returncode == 3
    assert len(listed.stdout.splitlines()) == 10
    assert "tests/NO_TEST_FILE_IS_AFFECTED" not in listed.stdout


def test_cli_prints_a_refusing_sentinel_and_exits_1_when_it_cannot_run(tmp_path):
    out = _run_cli(tmp_path / "not-a-checkout", "--files", "x.py")
    assert out.returncode == 1
    assert out.stdout == "tests/AFFECTED_TESTS_FAILED_SEE_STDERR\n"
    assert "Traceback" in out.stderr


def test_an_untracked_new_test_file_selects_itself(repo):
    mod, root = repo
    (root / "tests" / "test_new.py").write_text("def test_new():\n    pass\n", encoding="utf-8")
    changed, _ = mod.changed_files("no/such/ref")
    assert "tests/test_new.py" in changed
    assert _select(mod, changed)["tests/test_new.py"] == ["changed"]


# --------------------------------------------------------------------------- #
#  The real tree                                                              #
# --------------------------------------------------------------------------- #

def test_real_tree_change_to_the_motivating_route_selects_its_test():
    mod = _load()
    selected = mod.select(["localm/plugins/gui/routes/comfy.py"], mod.Graph())
    assert "names route /api/comfy/managed-status" in selected["tests/test_managed_comfy_s5_gui.py"]


def test_real_tree_changed_files_answers():
    mod = _load()
    changed, base = mod.changed_files("origin/master")
    assert isinstance(changed, list) and base
