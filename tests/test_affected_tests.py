# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/affected_tests.py: the test files a change affects, computed from
the import graph and from the names (route paths, module names, file names)
a test spells out.

These tests pin each selection rule against a throwaway git checkout: a
changed test file, a direct import, an importer N hops away (only with
--depth), a route called by URL, a module named as a string, a script or a
non-Python file named by file name, a route that only the committed version
still registers, and the files that affect everything. A dependency change in
pyproject.toml and uv.lock is pinned against a small lock of its own: a locked
bump, a transitive bump, a requirement change, a reordered table, the changes
that still affect every test, a dev-only tool, and an installed distribution
imported under another name. The command line's output shape and its
wide-selection exit status are pinned too. The last section binds the selector
to the real tree: a change to the route whose dropped field motivated it
selects the test that reads that route, the real dependency files parse, and a
locked uvicorn bump selects the tests that reach it and stays targeted.
"""

import importlib.util
import os
import re
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


def test_conftest_and_a_dependency_file_missing_on_either_side_affect_every_test(repo):
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


def test_read_stays_inside_the_repository(repo, tmp_path):
    mod, root = repo
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        assert mod._read(f"../{outside.name}") == ""
        assert mod._read(str(outside)) == ""
        assert mod._read("localm/b.py") == "def func():\n    return 1\n"
        assert mod._read("localm/missing.py") == ""
    finally:
        outside.unlink()


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
#  Dependency changes: pyproject.toml and uv.lock                             #
# --------------------------------------------------------------------------- #

_PYPROJECT = """\
[project]
name = "demo"
version = "1.0"
dependencies = ["webkit-srv>=1.0", "runtimeonly>=1", "Fancy-Dist>=1"]

[project.optional-dependencies]
pdf = ["pdfkit2>=2"]
dev = ["devtool>=1"]

[tool.pytest.ini_options]
addopts = "-q"
"""

_LOCK = """\
version = 1
revision = 3
requires-python = ">=3.12"

[options.exclude-newer-package]
alpha = false
beta = false

[[package]]
name = "demo"
version = "1.0"
source = { editable = "." }
dependencies = [
    { name = "fancy-dist" },
    { name = "runtimeonly" },
    { name = "webkit-srv" },
]

[package.optional-dependencies]
dev = [
    { name = "devtool" },
]
pdf = [
    { name = "pdfkit2" },
]

[package.metadata]
requires-dist = [
    { name = "devtool", marker = "extra == 'dev'", specifier = ">=1" },
    { name = "fancy-dist", specifier = ">=1" },
    { name = "pdfkit2", marker = "extra == 'pdf'", specifier = ">=2" },
    { name = "runtimeonly", specifier = ">=1" },
    { name = "webkit-srv", specifier = ">=1.0" },
]
provides-extras = ["pdf", "dev"]

[[package]]
name = "devtool"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "fancy-dist"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "lowlevel"
version = "0.5.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pdfkit2"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "runtimeonly"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "webkit-srv"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "lowlevel" },
]
"""

_DEP_FILES = {
    "pyproject.toml": _PYPROJECT,
    "uv.lock": _LOCK,
    "localm/server.py": "import webkit_srv\n",
    "localm/app.py": "from localm import server\n",
    "localm/fancy.py": "import fancymod\n",
    "tests/test_server.py": "import localm.server\n\n\ndef test_server():\n    assert localm.server\n",
    "tests/test_app.py": "import localm.app\n\n\ndef test_app():\n    assert localm.app\n",
    "tests/test_fancy.py": "import localm.fancy\n\n\ndef test_fancy():\n    assert localm.fancy\n",
    "tests/test_pdf.py": "import pdfkit2\n\n\ndef test_pdf():\n    assert pdfkit2\n",
    "tests/test_pdf_skip.py": (
        "import pytest\n\n\ndef test_pdf_skip():\n    pytest.importorskip(\"pdfkit2\")\n"),
    "tests/test_reads_pyproject.py": "def test_reads():\n    assert 'pyproject.toml'\n",
}


@pytest.fixture
def dep_repo(repo, monkeypatch):
    """The throwaway checkout plus a pyproject.toml, a uv.lock, modules that
    import (fake) third-party packages and tests that reach them, committed;
    no installed distribution maps to an import name."""
    mod, root = repo
    for rel, src in _DEP_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "dependencies")
    monkeypatch.setattr(mod, "_installed", lambda: {})
    return mod, root


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (rel, old)
    p.write_text(text.replace(old, new), encoding="utf-8")


def _bump(root: Path, dist: str, old: str, new: str) -> None:
    _edit(root, "uv.lock", f'name = "{dist}"\nversion = "{old}"', f'name = "{dist}"\nversion = "{new}"')


def _everything(mod, selected, rel):
    return (len(selected) == len(mod.Graph().test_files) == 16
            and all(r == [f"every test file: {rel} changed"] for r in selected.values()))


def test_a_locked_version_bump_selects_the_tests_reaching_that_dependency(dep_repo):
    mod, root = dep_repo
    _bump(root, "webkit-srv", "1.0.0", "1.1.0")
    assert _select(mod, ["uv.lock"]) == {"tests/test_server.py": [
        "imports localm.server, which imports webkit-srv: changed dependency webkit-srv"]}
    deeper = _select(mod, ["uv.lock"], depth=1)
    assert deeper["tests/test_app.py"] == [
        "imports localm.app (1 hop(s) from a module importing webkit-srv): "
        "changed dependency webkit-srv"]
    assert set(deeper) == {"tests/test_server.py", "tests/test_app.py"}


def test_a_transitive_bump_reaches_the_dependency_that_pulls_it_in(dep_repo):
    mod, root = dep_repo
    _bump(root, "lowlevel", "0.5.0", "0.6.0")
    assert _select(mod, ["uv.lock"]) == {"tests/test_server.py": [
        "imports localm.server, which imports webkit-srv: "
        "dependency webkit-srv (it depends on a changed package)"]}


def test_a_requirement_change_selects_its_dependency_and_the_tests_naming_pyproject(dep_repo):
    mod, root = dep_repo
    _edit(root, "pyproject.toml", 'pdf = ["pdfkit2>=2"]', 'pdf = ["pdfkit2>=2,<4"]')
    _edit(root, "uv.lock", 'specifier = ">=2" }', 'specifier = ">=2,<4" }')
    assert _select(mod, ["pyproject.toml", "uv.lock"]) == {
        "tests/test_pdf.py": ["imports pdfkit2: changed dependency pdfkit2"],
        "tests/test_pdf_skip.py": ["names pdfkit2: changed dependency pdfkit2"],
        "tests/test_reads_pyproject.py": ["names pyproject.toml"],
    }


def test_reordering_a_lock_table_changes_no_dependency(dep_repo):
    mod, root = dep_repo
    _edit(root, "uv.lock", "alpha = false\nbeta = false", "beta = false\nalpha = false")
    assert _select(mod, ["uv.lock"]) == {}


def test_a_pyproject_change_outside_the_requirement_lists_affects_every_test(dep_repo):
    mod, root = dep_repo
    _edit(root, "pyproject.toml", 'addopts = "-q"', 'addopts = "-q -x"')
    assert _everything(mod, _select(mod, ["pyproject.toml"]), "pyproject.toml")


def test_a_lock_change_outside_the_packages_affects_every_test(dep_repo):
    mod, root = dep_repo
    _edit(root, "uv.lock", 'requires-python = ">=3.12"', 'requires-python = ">=3.13"')
    assert _everything(mod, _select(mod, ["uv.lock"]), "uv.lock")


def test_the_projects_own_locked_entry_outside_its_declarations_affects_every_test(dep_repo):
    mod, root = dep_repo
    _bump(root, "demo", "1.0", "1.1")
    assert _everything(mod, _select(mod, ["uv.lock"]), "uv.lock")


def test_an_unparsable_lock_affects_every_test(dep_repo):
    mod, root = dep_repo
    (root / "uv.lock").write_text("[[package\n", encoding="utf-8")
    assert _everything(mod, _select(mod, ["uv.lock"]), "uv.lock")


def test_a_runtime_dependency_nothing_imports_or_names_affects_every_test(dep_repo):
    mod, root = dep_repo
    _bump(root, "runtimeonly", "1.0.0", "1.0.1")
    assert _everything(mod, _select(mod, ["uv.lock"]), "uv.lock")


def test_a_dev_only_dependency_nothing_imports_or_names_selects_no_test(dep_repo):
    mod, root = dep_repo
    _bump(root, "devtool", "1.0.0", "1.0.1")
    assert _select(mod, ["uv.lock"]) == {}


def test_an_installed_distributions_import_name_finds_its_importers(dep_repo, monkeypatch):
    mod, root = dep_repo
    _bump(root, "fancy-dist", "1.0.0", "2.0.0")
    assert _everything(mod, _select(mod, ["uv.lock"]), "uv.lock")
    monkeypatch.setattr(mod, "_installed", lambda: {"fancymod": ["Fancy-Dist"], "other": ["x"]})
    assert _select(mod, ["uv.lock"]) == {"tests/test_fancy.py": [
        "imports localm.fancy, which imports fancy-dist: changed dependency fancy-dist"]}


def test_cli_a_locked_bump_is_a_targeted_selection(dep_repo):
    _, root = dep_repo
    _bump(root, "webkit-srv", "1.0.0", "1.1.0")
    out = _run_cli(root, "--files", "uv.lock", "--why")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "tests/test_server.py  # imports localm.server, which imports webkit-srv: "
        "changed dependency webkit-srv"]
    assert "1 of 16 test files affected by 1 changed file(s)" in out.stderr


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


def test_real_tree_dependency_files_parse_and_the_dev_extra_is_not_runtime():
    mod = _load()
    py = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    changed, affected, runtime = mod.dependency_change(py, py, lock, lock)
    assert changed == set() and affected == set()
    assert {"uvicorn", "fastapi", "pypdf", "playwright"} <= runtime
    assert "ruff" not in runtime and "zizmor" not in runtime


def test_real_tree_a_locked_uvicorn_bump_is_a_targeted_selection(monkeypatch):
    mod = _load()
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = re.search(r'name = "uvicorn"\nversion = "[^"]+"', lock).group(0)
    bumped = lock.replace(entry, 'name = "uvicorn"\nversion = "999.0.0"')
    real_read = mod._read
    monkeypatch.setattr(mod, "_read", lambda rel: bumped if rel == "uv.lock" else real_read(rel))
    monkeypatch.setattr(mod, "_read_at", lambda ref, rel: real_read(rel))
    graph = mod.Graph()
    selected = mod.select(["uv.lock"], graph)
    assert "imports localm.portmux, which imports uvicorn: changed dependency uvicorn" in \
        selected["tests/test_portmux_redirect.py"]
    assert len(selected) <= 0.25 * len(graph.test_files)
