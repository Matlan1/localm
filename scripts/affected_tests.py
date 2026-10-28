#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Select the test files a change affects, so a targeted run is computed rather
than guessed.

A test file is selected when any of these holds:
  - it changed itself;
  - it imports a changed module directly (with --depth N, also a module that
    imports the changed one, up to N hops);
  - it names a route path a changed module registers (the static prefix of
    the path, so a test that calls the route by URL is found even when it
    never imports the module);
  - it names a changed module by its dotted name (a monkeypatch target, an
    import written as a string), a changed script by file name, or a changed
    non-Python file by file name;
  - a change to a module inside a package listed in _REEXPORT_FACADES also
    counts as a change to that package, and a change to a module inside a
    package mapped in _REEXPORT_MODULE_FACADES also counts as a change to the
    module it is mapped from, for both rules above, and that module counts as
    importing every module the package's modules import;
  - it is listed in _TREE_SCANNERS for the folder a changed .py file sits in
    (it reads every file under that folder instead of importing the changed
    one);
  - it imports a changed dependency, names it in a string literal, imports a
    module that imports it (with --depth N, also that module's importers, up
    to N hops), or sits under the folder of a conftest.py that imports it. A
    changed dependency is a distribution whose requirement in pyproject.toml
    or whose entry in uv.lock differs, and every package in uv.lock that
    depends on one, transitively.

A version bump selects the tests that read the version instead of every test
file: the files that spell the project version (VERSION, localm/__init__.py
when only its __version__ string differs, the [project] version of
pyproject.toml, the version of the project's own editable entry in uv.lock)
select the test files that reference __version__, name the VERSION, pyproject.toml,
uv.lock or CHANGELOG.md files, or spell the old or the new version.

A change to tests/conftest.py affects every test file. So does a change to
pyproject.toml or uv.lock outside the [project] requirement lists, the
[tool.ruff], [tool.basedpyright], [tool.mutmut] and [tool.uv] tables (a
change to [tool.mutmut] selects the tests listed in _TOOL_TABLE_READERS), the lock's
resolution-markers and supported-markers and the locked packages, either file
missing or unparsable on either side, a listed
pyproject.toml or uv.lock that does not differ from the base (a committed
change named with --files), and a changed dependency the project declares
outside the dev extra that the rules above reach no test file from. A changed
dependency declared only in the dev extra selects only the test files those
rules reach, possibly none.

Changes are read from git: the diff from the merge base with --base (default
origin/master) to HEAD, plus staged, unstaged and untracked work. --files takes
an explicit list instead.

    python scripts/affected_tests.py                   list the selection
    python scripts/affected_tests.py --why             list with the reasons
    python scripts/affected_tests.py --files a.py b.py
    pytest $(python scripts/affected_tests.py) -m "not integration"

The selection is printed one path per line so it can be substituted into a
pytest command line; the summary goes to stderr. Stdout never leaves pytest
with an empty or a whole-suite argument list: when nothing is affected the one
line printed is tests/NO_TEST_FILE_IS_AFFECTED; when the selection is wider
than --max-share of all test files (the change touches a module most tests
import, so a run that wide is not a targeted run) the one line printed is
tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR and the exit status is
3, with --list-wide printing the selection instead; when the script itself
fails the one line printed is tests/AFFECTED_TESTS_FAILED_SEE_STDERR and the
exit status is 1. None of those paths exists, so pytest refuses each one.

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import re
import subprocess
import sys
import tomllib
import traceback
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "websocket", "api_route"}
_EVERYTHING = {"tests/conftest.py"}
# Packages whose callers import and patch the package itself, which re-exports
# its submodules' names. Each counts as importing every module its own modules
# import.
_REEXPORT_FACADES = ("localm.setup_llama", "localm.bugreport")
_DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
_UNREAD_TOOL_TABLES = ("ruff", "basedpyright")
_RESOLVER_TOOL_TABLES = ("uv",)
# Tool tables of pyproject.toml that only the listed test files read. A change
# confined to one selects those tests instead of every test file.
_TOOL_TABLE_READERS = {
    "mutmut": ("tests/test_check_mutation_floors.py", "tests/test_mutation_scope.py"),
}
_LOCK_FORK_KEYS = ("resolution-markers", "supported-markers")
_VERSION_FILE = "VERSION"
_PACKAGE_INIT = "localm/__init__.py"
_PROJECT_LOCK_SOURCE = {"editable": "."}
_VERSION_FILE_NAMES = ("pyproject.toml", "uv.lock", "CHANGELOG.md")
_VERSION_FILE_WORD = re.compile(r"(?<![A-Za-z0-9_])VERSION(?![A-Za-z0-9_])")
_DEV_EXTRA = "dev"
_REQUIREMENT_KEYS = ("dependencies", "optional-dependencies")
# Modules whose callers import and patch the module itself, which re-exports the
# names of the modules in the package mapped to it.
_REEXPORT_MODULE_FACADES = {"localm.rag.store": "localm.rag._store"}
# Test files that read a whole source tree instead of importing the module that
# changed, so the import graph never reaches them. Each maps to the path
# prefixes whose .py files it reads and what it does with them. A .py file added,
# changed or deleted under a prefix selects the tests listed for it.
_LOCALM, _TESTS, _SCRIPTS = "localm/", "tests/", "scripts/"
_ALL_SOURCES = (_LOCALM, _TESTS, _SCRIPTS)
_TREE_SCANNERS: dict[str, tuple[tuple[str, ...], str]] = {
    "tests/test_check_pins.py": ((_LOCALM,), "greps every tracked localm/*.py for pin-shaped constants"),
    "tests/test_no_gitignored_path_leak.py": (_ALL_SOURCES, "scans the string literals of every tracked .py"),
    "tests/test_no_stray_encoding_cookie.py": (_ALL_SOURCES, "reads the first lines of every .py"),
    "tests/test_check_hygiene.py": (_ALL_SOURCES, "runs the hygiene checks over every tracked file"),
    "tests/test_check_hygiene_layering.py": (_ALL_SOURCES, "checks import direction over every tracked localm file"),
    "tests/test_check_hygiene_response_keys.py": (_ALL_SOURCES, "checks the JSON keys of every tracked route"),
    "tests/test_bump_uv_pin.py": ((_LOCALM,), "lists every tracked file outside tests/ and scripts/ that pins uv"),
    "tests/test_native_stderr_redirect_scope.py": ((_LOCALM,), "walks the AST of every localm .py for stderr redirects"),
    "tests/test_terminal_control_safety.py": ((_LOCALM,), "walks the AST of every localm .py for rich Console imports"),
    "tests/test_model_name_gate.py": ((_LOCALM,), "reads every localm .py for trust_remote_code=True"),
    "tests/test_no_unguarded_json_parse.py": ((_LOCALM,), "walks the AST of every localm .py for unguarded JSON parses"),
    "tests/test_owner_predicate_single_owner.py": ((_LOCALM,), "walks the AST of every localm .py for inline owner checks"),
    "tests/test_no_bare_secret_chmod.py": ((_LOCALM,), "walks the AST of every localm .py for chmod 0o600 calls"),
    "tests/test_no_untracked_authorization_header_writes.py": (
        (_LOCALM,), "walks the AST of every localm .py for Authorization header writes"),
    "tests/test_hwdetect_recommended_is_legacy.py": ((_LOCALM,), "walks the AST of every localm .py for .recommended reads"),
    "tests/test_install_manifest.py": ((_LOCALM,), "walks the AST of every localm .py for data-dir children"),
    "tests/test_gui_js_route_contract.py": ((_LOCALM,), "reads every localm .py for declared routes"),
    "tests/test_ctrlc_clean_stop.py": ((_LOCALM,), "reads every localm .py for parent-death watchdog installs"),
    "tests/test_residency.py": ((_LOCALM,), "greps every localm .py for active_requests mutations"),
    "tests/test_discover.py": ((_LOCALM,), "walks the AST of every localm .py for calls to the discovery helpers"),
    "tests/test_native_fault_debug_log_hint.py": (("localm/inference/",), "scans every localm/inference .py"),
    "tests/test_debug_scrub.py": ((_TESTS,), "walks the AST of every test file for delenv of LOCALM_DEBUG"),
    "tests/test_real_gguf_gate.py": ((_TESTS,), "scans every test file for broad skips around model loads"),
    "tests/test_tk_root.py": ((_TESTS,), "scans every test file that builds Tk windows"),
    "tests/test_bugreport_live_lookups.py": ((_TESTS,), "scans every test file that mentions bugreport"),
    "tests/test_ci_runner_files.py": ((_SCRIPTS,), "reads every scripts/*.py for runner-file variable reads"),
    "tests/test_pin_weekly.py": ((_SCRIPTS,), "reads the pin, bump and confirm scripts for gitignored paths"),
}
_ROOT_DECLARATIONS = ("dependencies", "optional-dependencies", "dev-dependencies")
_ROOT_METADATA_DECLARATIONS = ("requires-dist", "requires-dev", "provides-extras")
_SOURCE_ROOTS = ("localm", "scripts", "tests")
_TESTS_ROOT = "tests"
_TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")
_WIDE_EXIT = 3
_NOTHING_AFFECTED = "tests/NO_TEST_FILE_IS_AFFECTED"
_TOO_WIDE = "tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR"
_FAILED = "tests/AFFECTED_TESTS_FAILED_SEE_STDERR"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=True).stdout


def _tracked(prefix: str) -> list[str]:
    """Tracked plus untracked, not ignored, files under *prefix*."""
    files: list[str] = []
    for extra in ((), ("--others", "--exclude-standard")):
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", *extra, "--", prefix],
                             capture_output=True, check=True).stdout
        files.extend(p for p in out.decode("utf-8").split("\0") if p)
    return files


def module_name(rel: str) -> str:
    """The dotted module name of a repository-relative .py path."""
    parts = list(Path(rel).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_names(source: str, own: str, is_package: bool) -> set[str]:
    """Every dotted name *source* imports: plain and from-imports (each
    imported attribute also as `module.attr`, so a submodule import resolves),
    relative imports made absolute against *own*, and a literal
    `import_module("x")` target."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = own.split(".") if is_package else own.split(".")[:-1]
                base = base[:len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "import_module" and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


def _top_levels(names: set[str]) -> set[str]:
    """The first component of each dotted name in *names*."""
    return {n.split(".", 1)[0] for n in names if n}


def route_prefixes(source: str) -> set[str]:
    """The static prefix of every literal or f-string route path a decorated
    handler in *source* registers."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in _ROUTE_METHODS and dec.args):
                continue
            arg = dec.args[0]
            text = None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                text = arg.value
            elif isinstance(arg, ast.JoinedStr) and arg.values \
                    and isinstance(arg.values[0], ast.Constant):
                text = str(arg.values[0].value)
            if text and text.startswith("/"):
                prefix = text.split("{", 1)[0]
                if len(prefix) > 1:
                    out.add(prefix)
    return out


class Graph:
    """Import edges over the tracked source files (localm/, scripts/, and the
    helper modules under tests/) and the test files (tests/**/test_*.py)."""

    def __init__(self) -> None:
        self.sources: dict[str, str] = {}          # module name -> relative path
        self.aliases: dict[str, str] = {}          # importable spelling -> module name
        self.reverse: dict[str, set[str]] = {}     # module -> modules importing it
        self.test_files: list[str] = []
        self.test_imports: dict[str, set[str]] = {}
        self.source_top: dict[str, set[str]] = {}  # module name -> top-level names it imports
        self.test_top: dict[str, set[str]] = {}    # test file -> top-level names it imports
        for root in _SOURCE_ROOTS:
            for rel in _tracked(root):
                if not rel.endswith(".py"):
                    continue
                if _TEST_FILE.search(rel):
                    self.test_files.append(rel)
                    continue
                mod = module_name(rel)
                self.sources[mod] = rel
                self.aliases[mod] = mod
                if rel.startswith(_TESTS_ROOT + "/") and "." in mod:
                    self.aliases[mod.split(".", 1)[1]] = mod
        self.test_files.sort()
        for mod, rel in self.sources.items():
            names = imported_names(_read(rel), mod, rel.endswith("__init__.py"))
            self.source_top[mod] = _top_levels(names)
            for dep in self.resolve(names):
                if dep != mod:
                    self.reverse.setdefault(dep, set()).add(mod)
        for facade, pkg in _REEXPORT_MODULE_FACADES.items():
            if facade not in self.sources:
                continue
            inside = {m for m in self.sources if m == pkg or m.startswith(pkg + ".")}
            for dep, users in self.reverse.items():
                if dep != facade and dep not in inside and users & inside:
                    users.add(facade)
        for rel in self.test_files:
            names = imported_names(_read(rel), module_name(rel), False)
            self.test_top[rel] = _top_levels(names)
            self.test_imports[rel] = self.resolve(names)
        for facade in _REEXPORT_FACADES:
            if facade not in self.sources:
                continue
            for dep, users in self.reverse.items():
                inside = dep == facade or dep.startswith(facade + ".")
                if not inside and any(u.startswith(facade + ".") for u in users):
                    users.add(facade)

    def resolve(self, names: set[str]) -> set[str]:
        """The known modules the dotted *names* refer to, by longest prefix."""
        found: set[str] = set()
        for name in names:
            parts = name.split(".")
            while parts:
                candidate = self.aliases.get(".".join(parts))
                if candidate:
                    found.add(candidate)
                    break
                parts.pop()
        return found

    def importers(self, modules: set[str], depth: int) -> dict[str, int]:
        """{module: hops} for *modules* (0 hops) and everything that imports
        them within *depth* hops."""
        hops = {m: 0 for m in modules}
        frontier = set(modules)
        for hop in range(1, depth + 1):
            nxt: set[str] = set()
            for m in frontier:
                for importer in self.reverse.get(m, ()):
                    if importer not in hops:
                        hops[importer] = hop
                        nxt.add(importer)
            frontier = nxt
        return hops


def _read(rel: str) -> str:
    """The text of the repository file *rel*; empty for a path outside the
    repository, an unreadable file, or non-UTF-8 content."""
    root = os.path.normpath(str(REPO))
    full = os.path.normpath(os.path.join(root, rel))
    if not full.startswith(root + os.sep):
        return ""
    try:
        with open(full, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _read_at(ref: str, rel: str) -> str:
    try:
        return _git("show", f"{ref}:{rel}")
    except subprocess.CalledProcessError:
        return ""


def dist_key(name: str) -> str:
    """The normalized (PEP 503) form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(pyproject: dict) -> dict[str, list[tuple[str, str]]] | None:
    """{distribution: sorted (extra, requirement) pairs} over [project]
    dependencies (extra "") and optional-dependencies; None when a
    requirement does not start with a distribution name."""
    project = pyproject.get("project", {})
    sections = [("", project.get("dependencies", []))]
    sections += sorted(project.get("optional-dependencies", {}).items())
    out: dict[str, list[tuple[str, str]]] = {}
    for extra, reqs in sections:
        for req in reqs:
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req) if isinstance(req, str) else None
            if not m:
                return None
            out.setdefault(dist_key(m.group(1)), []).append((extra, req))
    return {name: sorted(pairs) for name, pairs in out.items()}


def _pyproject_rest(pyproject: dict) -> dict:
    """*pyproject* without the [project] requirement lists, the tool tables no
    test reads, the tool tables in _TOOL_TABLE_READERS, which
    changed_tool_table_readers() handles, the [project] version, which
    project_versions() handles, and the [tool.uv] table, whose effect on the locked packages
    dependency_change() reads from the lock instead."""
    project = {k: v for k, v in pyproject.get("project", {}).items()
               if k not in _REQUIREMENT_KEYS and k != "version"}
    skipped = _UNREAD_TOOL_TABLES + _RESOLVER_TOOL_TABLES + tuple(_TOOL_TABLE_READERS)
    tool = {k: v for k, v in pyproject.get("tool", {}).items() if k not in skipped}
    return {**pyproject, "project": project, "tool": tool}


def changed_tool_table_readers(old_pyproject: str, new_pyproject: str) -> dict[str, list[str]]:
    """{test file: reasons} for the test files that read a [tool.*] table of
    _TOOL_TABLE_READERS that differs between the two pyproject.toml texts.
    Empty when either side is empty or does not parse."""
    try:
        old, new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
    except tomllib.TOMLDecodeError:
        return {}
    out: dict[str, list[str]] = {}
    for table, readers in _TOOL_TABLE_READERS.items():
        if old.get("tool", {}).get(table) != new.get("tool", {}).get(table):
            for t in readers:
                out.setdefault(t, []).append(f"reads pyproject.toml [tool.{table}]: it changed")
    return out


def _lock_packages(lock: dict, project: str) -> dict[str, list[str]]:
    """{distribution: the canonical JSON of each of its uv.lock entries}, with
    the dependency declarations left out of *project*'s own entry, and the
    version too when that entry is the project's editable one."""
    out: dict[str, list[str]] = {}
    for entry in lock.get("package", []):
        name = dist_key(str(entry.get("name", "")))
        if name == project:
            meta = {k: v for k, v in entry.get("metadata", {}).items()
                    if k not in _ROOT_METADATA_DECLARATIONS}
            skipped = _ROOT_DECLARATIONS + (("version",) if entry.get("source") == _PROJECT_LOCK_SOURCE
                                            else ())
            entry = {**{k: v for k, v in entry.items() if k not in skipped},
                     "metadata": meta}
        out.setdefault(name, []).append(json.dumps(entry, sort_keys=True))
    return {name: sorted(entries) for name, entries in out.items()}


def _lock_dependents(lock: dict) -> dict[str, set[str]]:
    """{distribution: the uv.lock packages that depend on it}."""
    out: dict[str, set[str]] = {}
    for entry in lock.get("package", []):
        groups = [entry.get("dependencies", [])]
        for key in ("optional-dependencies", "dev-dependencies"):
            groups.extend(entry.get(key, {}).values())
        for group in groups:
            for dep in group:
                out.setdefault(dist_key(str(dep.get("name", ""))), set()).add(
                    dist_key(str(entry.get("name", ""))))
    return out


def _project_version_pair(pyproject: dict, lock: dict, project: str) -> tuple[object, object, bool]:
    """(the [project] version, the version of the project's editable entry in
    *lock*, whether such an entry exists)."""
    for entry in lock.get("package", []):
        if dist_key(str(entry.get("name", ""))) == project and entry.get("source") == _PROJECT_LOCK_SOURCE:
            return pyproject.get("project", {}).get("version"), entry.get("version"), True
    return pyproject.get("project", {}).get("version"), None, False


def _differing_project_versions(py_old: dict, py_new: dict, lock_old: dict, lock_new: dict,
                                project: str) -> set[str] | None:
    """The project version strings, old and new, of each of pyproject.toml and
    the lock's editable entry whose version differs between the two sides; None
    when a version is not a string on a side or the entry exists on one side
    only."""
    pv_old, lv_old, has_old = _project_version_pair(py_old, lock_old, project)
    pv_new, lv_new, has_new = _project_version_pair(py_new, lock_new, project)
    pairs = [(pv_old, pv_new)] + ([(lv_old, lv_new)] if has_old and has_new else [])
    if has_old != has_new or not all(isinstance(v, str) for pair in pairs for v in pair):
        return None
    return {v for old, new in pairs if old != new for v in (old, new)}


def project_versions(old_pyproject: str, new_pyproject: str, old_lock: str,
                     new_lock: str) -> set[str]:
    """The project version strings, old and new, that pyproject.toml and the
    lock's editable entry for the project change. Empty when neither changes
    its version, a side is empty or does not parse, or a version is not a
    string."""
    if not (old_pyproject and new_pyproject and old_lock and new_lock):
        return set()
    try:
        py_old, py_new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
        lock_old, lock_new = tomllib.loads(old_lock), tomllib.loads(new_lock)
    except tomllib.TOMLDecodeError:
        return set()
    project = dist_key(str(py_new.get("project", {}).get("name", "")))
    return _differing_project_versions(py_old, py_new, lock_old, lock_new, project) or set()


def init_version_change(old_source: str, new_source: str) -> tuple[str, str] | None:
    """(old, new) version when the two texts of localm/__init__.py differ only
    in the string assigned to the module-level `__version__`; None otherwise."""
    versions: list[str] = []
    dumps: list[str] = []
    for source in (old_source, new_source):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        found = [node for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__"
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
        if len(found) != 1:
            return None
        versions.append(found[0].value.value)
        found[0].value = ast.Constant("")
        dumps.append(ast.dump(tree))
    if versions[0] == versions[1] or dumps[0] != dumps[1]:
        return None
    return versions[0], versions[1]


def version_bump(changed: list[str], base_ref: str) -> tuple[set[str], set[str]]:
    """(the changed files that are only a version bump, the old and new project
    versions): VERSION when its text differs from *base_ref*'s, and
    localm/__init__.py when only its __version__ string differs."""
    files: set[str] = set()
    versions: set[str] = set()
    if _VERSION_FILE in changed:
        old, new = _read_at(base_ref, _VERSION_FILE).strip(), _read(_VERSION_FILE).strip()
        if old and new and old != new:
            files.add(_VERSION_FILE)
            versions.update((old, new))
    if _PACKAGE_INIT in changed:
        pair = init_version_change(_read_at(base_ref, _PACKAGE_INIT), _read(_PACKAGE_INIT))
        if pair:
            files.add(_PACKAGE_INIT)
            versions.update(pair)
    return files, versions


def _references_version(source: str) -> bool:
    """Whether *source* reads or names `__version__`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return any((isinstance(n, ast.Name) and n.id == "__version__")
               or (isinstance(n, ast.Attribute) and n.attr == "__version__")
               or (isinstance(n, ast.ImportFrom) and any(a.name == "__version__" for a in n.names))
               or (isinstance(n, ast.Constant) and n.value == "__version__")
               for n in ast.walk(tree))


def version_readers(graph: Graph, versions: set[str], test_text) -> dict[str, list[str]]:
    """{test file: reasons} for the test files that read the project version: one
    that references `__version__`, names the VERSION, pyproject.toml, uv.lock or
    CHANGELOG.md file, or spells one of *versions*. *test_text* reads a test
    file."""
    spelled = [(v, re.compile(r"(?<![\w.])v?" + re.escape(v) + r"(?!\w|\.\d)")) for v in sorted(versions)]
    out: dict[str, list[str]] = {}
    for t in graph.test_files:
        text = test_text(t)
        why: list[str] = []
        if _references_version(text):
            why.append("reads __version__: the project version changed")
        if _VERSION_FILE_WORD.search(text):
            why.append("names VERSION: the project version changed")
        why += [f"names {name}: the project version changed"
                for name in _VERSION_FILE_NAMES if name in text]
        why += [f"spells {v}: the project version changed" for v, rx in spelled if rx.search(text)]
        if why:
            out[t] = why
    return out


def dependency_change(old_pyproject: str, new_pyproject: str, old_lock: str,
                      new_lock: str) -> tuple[set[str], set[str], set[str]] | None:
    """(changed, affected, runtime) for a change to pyproject.toml and uv.lock,
    as normalized distribution names: the distributions whose requirement or
    locked entry differs; those plus every locked package that depends on one
    of them, transitively; and the distributions the project requires outside
    the dev extra, on either side. None when either file differs in anything
    besides the [project] requirement lists and the locked packages (the
    project's own locked entry counts only for its dependency declarations),
    or when a side is empty or does not parse."""
    if not (old_pyproject and new_pyproject and old_lock and new_lock):
        return None
    try:
        py_old, py_new = tomllib.loads(old_pyproject), tomllib.loads(new_pyproject)
        lock_old, lock_new = tomllib.loads(old_lock), tomllib.loads(new_lock)
    except tomllib.TOMLDecodeError:
        return None
    req_old, req_new = _requirements(py_old), _requirements(py_new)
    if req_old is None or req_new is None or _pyproject_rest(py_old) != _pyproject_rest(py_new):
        return None
    project = dist_key(str(py_new.get("project", {}).get("name", "")))
    if _differing_project_versions(py_old, py_new, lock_old, lock_new, project) is None:
        return None
    if ({k: v for k, v in lock_old.items() if k != "package" and k not in _LOCK_FORK_KEYS}
            != {k: v for k, v in lock_new.items() if k != "package" and k not in _LOCK_FORK_KEYS}):
        return None
    pkg_old, pkg_new = _lock_packages(lock_old, project), _lock_packages(lock_new, project)
    if pkg_old.get(project) != pkg_new.get(project):
        return None
    changed = {n for n in req_old.keys() | req_new.keys() if req_old.get(n) != req_new.get(n)}
    changed |= {n for n in pkg_old.keys() | pkg_new.keys() if pkg_old.get(n) != pkg_new.get(n)}
    dependents = _lock_dependents(lock_old)
    for name, parents in _lock_dependents(lock_new).items():
        dependents.setdefault(name, set()).update(parents)
    affected, frontier = set(changed), list(changed)
    while frontier:
        for parent in dependents.get(frontier.pop(), ()):
            if parent != project and parent not in affected:
                affected.add(parent)
                frontier.append(parent)
    runtime = {n for reqs in (req_old, req_new) for n, pairs in reqs.items()
               if any(extra != _DEV_EXTRA for extra, _ in pairs)}
    return changed, affected, runtime


@functools.cache
def _installed() -> dict[str, list[str]]:
    """{top-level import name: the installed distributions providing it}."""
    return metadata.packages_distributions()


def import_names(dist: str) -> set[str]:
    """The top-level names the distribution *dist* can be imported under: its
    name with separators as underscores, and the names its installed files
    provide."""
    names = {dist.replace("-", "_")}
    names.update(top for top, dists in _installed().items()
                 if any(d and dist_key(d) == dist for d in dists))
    return names


def _names_in_a_string(names: set[str]) -> re.Pattern[str]:
    """Matches any of *names* opening a string literal, as a whole name."""
    alternatives = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return re.compile(r"""["'](?:""" + alternatives + r""")(?=["'.\[<>=!~;\s])""", re.IGNORECASE)


def _dependency_reasons(graph: Graph, base_ref: str, listed: list[str], depth: int,
                        test_text) -> dict[str, list[str]] | None:
    """{test file: reasons} for the test files a change to the *listed*
    dependency files reaches: a test that imports an affected distribution,
    names it in a string literal, or imports (within *depth* hops) a module
    that imports it, and every test file under the folder of a conftest.py
    that imports it. None when every test file is affected:
    dependency_change() returned None, a listed file does not differ from
    *base_ref*, or an affected distribution the project requires outside the
    dev extra reaches no test file. *test_text* reads a test file."""
    texts = {rel: (_read_at(base_ref, rel), _read(rel)) for rel in _DEPENDENCY_FILES}
    if any(texts[rel][0] == texts[rel][1] for rel in listed):
        return None
    change = dependency_change(*texts["pyproject.toml"], *texts["uv.lock"])
    if change is None:
        return None
    changed, affected, runtime = change
    reasons: dict[str, list[str]] = {}
    for dist in sorted(affected):
        names = import_names(dist)
        users = {m for m, top in graph.source_top.items() if top & names}
        named = _names_in_a_string(names | {dist})
        hops = graph.importers(users, depth)
        folders = sorted(graph.sources[m].rsplit("/", 1)[0] + "/" for m in users
                         if graph.sources[m].endswith("/conftest.py"))
        label = f"changed dependency {dist}" if dist in changed else \
            f"dependency {dist} (it depends on a changed package)"
        hits: dict[str, list[str]] = {}
        for t in graph.test_files:
            why = [f"every test file under {folder}: its conftest.py imports {dist}: {label}"
                   for folder in folders if t.startswith(folder)]
            why += [f"imports {name}: {label}" for name in sorted(graph.test_top[t] & names)]
            for mod in sorted(graph.test_imports[t] & set(hops), key=lambda m: (hops[m], m)):
                why.append(f"imports {mod}, which imports {dist}: {label}" if hops[mod] == 0
                           else f"imports {mod} ({hops[mod]} hop(s) from a module importing "
                                f"{dist}): {label}")
            if named.search(test_text(t)):
                why.append(f"names {dist}: {label}")
            if why:
                hits[t] = why
        if not hits and dist in runtime:
            return None
        for t, why in hits.items():
            reasons.setdefault(t, []).extend(why)
    return reasons


def changed_files(base: str) -> tuple[list[str], str]:
    """Repository-relative paths changed since the merge base with *base*,
    plus staged, unstaged and untracked files; and the merge-base ref the
    committed half was diffed against (HEAD when *base* cannot be resolved)."""
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
    except subprocess.CalledProcessError:
        merge_base = "HEAD"
    files: set[str] = set()
    files.update(_git("diff", "--name-only", merge_base, "HEAD").splitlines())
    files.update(_git("diff", "--name-only", "HEAD").splitlines())
    files.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    return sorted(f.replace("\\", "/") for f in files if f), merge_base


def select(changed: list[str], graph: Graph, depth: int = 0,
           base_ref: str = "HEAD") -> dict[str, list[str]]:
    """{test file: reasons} for the test files *changed* affects."""
    reasons: dict[str, list[str]] = {}

    def add(test: str, why: str) -> None:
        reasons.setdefault(test, []).append(why)

    test_text = functools.cache(_read)
    whole = [c for c in changed if c in _EVERYTHING]
    listed = [c for c in changed if c in _DEPENDENCY_FILES]
    dependency_reasons: dict[str, list[str]] = {}
    if listed:
        found = _dependency_reasons(graph, base_ref, listed, depth, test_text)
        if found is None:
            whole += listed
        else:
            dependency_reasons = found
    if whole:
        which = ", ".join(sorted(whole))
        for t in graph.test_files:
            add(t, f"every test file: {which} changed")
        return reasons

    bump_files, versions = version_bump(changed, base_ref)
    if listed:
        versions |= project_versions(_read_at(base_ref, "pyproject.toml"), _read("pyproject.toml"),
                                     _read_at(base_ref, "uv.lock"), _read("uv.lock"))
    if versions:
        for t, whys in version_readers(graph, versions, test_text).items():
            for why in whys:
                add(t, why)

    if "pyproject.toml" in listed:
        readers = changed_tool_table_readers(_read_at(base_ref, "pyproject.toml"),
                                             _read("pyproject.toml"))
        for t, whys in readers.items():
            if t in graph.test_imports:
                for why in whys:
                    add(t, why)

    for rel in changed:
        if not rel.endswith(".py"):
            continue
        for t, (prefixes, why) in _TREE_SCANNERS.items():
            under = [p for p in prefixes if rel.startswith(p)]
            if under and t in graph.test_imports:
                add(t, f"{why} under {under[0]}: {rel} changed")

    changed_modules: set[str] = set()
    facades: set[str] = set()
    needles: list[tuple[str, re.Pattern[str]]] = []
    for rel in changed:
        if rel in graph.test_imports:
            add(rel, "changed")
            continue
        if rel in bump_files:
            continue
        if rel.endswith("/conftest.py"):
            folder = rel.rsplit("/", 1)[0] + "/"
            for t in graph.test_files:
                if t.startswith(folder):
                    add(t, f"every test file under {folder}: its conftest.py changed")
            continue
        under_source = rel.split("/", 1)[0] in _SOURCE_ROOTS
        if rel.endswith(".py") and under_source:
            mod = module_name(rel)
            changed_modules.add(mod)
            facades.update(f for f in _REEXPORT_FACADES
                           if mod.startswith(f + ".") and f in graph.sources)
            facades.update(f for f, pkg in _REEXPORT_MODULE_FACADES.items()
                           if (mod == pkg or mod.startswith(pkg + "."))
                           and f in graph.sources)
            text = _read(rel) + "\n" + _read_at(base_ref, rel)
            for prefix in sorted(route_prefixes(text)):
                needles.append((f"names route {prefix}", re.compile(re.escape(prefix))))
            if rel.startswith("localm/"):
                needles.append((f"names {mod}", re.compile(re.escape(mod) + r"\b")))
            else:
                needles.append((f"names {Path(rel).name}", re.compile(re.escape(Path(rel).name))))
        else:
            name = Path(rel).name
            if name:
                needles.append((f"names {name}", re.compile(re.escape(name))))
    for facade in sorted(facades - changed_modules):
        changed_modules.add(facade)
        needles.append((f"names {facade}", re.compile(re.escape(facade) + r"\b")))

    hops = graph.importers(changed_modules, depth)
    for t in graph.test_files:
        hit = graph.test_imports[t] & set(hops)
        for mod in sorted(hit, key=lambda m: (hops[m], m)):
            add(t, f"imports {mod}" if hops[mod] == 0 else f"imports {mod} ({hops[mod]} hop(s) from a change)")
        if needles:
            for label, rx in needles:
                if rx.search(test_text(t)):
                    add(t, label)
        for why in dependency_reasons.get(t, []):
            add(t, why)
    return reasons


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--base", default="origin/master",
                    help="ref the committed changes are diffed from (via merge-base)")
    ap.add_argument("--files", nargs="*", help="use these changed paths instead of git")
    ap.add_argument("--depth", type=int, default=0,
                    help="follow importers of a changed module this many hops (default 0)")
    ap.add_argument("--why", action="store_true", help="print the reason next to each file")
    ap.add_argument("--max-share", type=float, default=0.25,
                    help="exit 3 when the selection exceeds this share of all test files")
    ap.add_argument("--list-wide", action="store_true",
                    help="print a selection wider than --max-share instead of the sentinel")
    args = ap.parse_args(argv)

    # Every line is LF-terminated on every platform.
    sys.stdout.reconfigure(newline="\n")
    try:
        return _run(args)
    except Exception:
        print(_FAILED)
        traceback.print_exc()
        return 1


def _run(args) -> int:
    graph = Graph()
    if args.files is not None:
        changed, base_ref = sorted(f.replace("\\", "/") for f in args.files), "HEAD"
    else:
        changed, base_ref = changed_files(args.base)
    reasons = select(changed, graph, depth=args.depth, base_ref=base_ref)
    selected = sorted(reasons)
    total = len(graph.test_files)
    share = len(selected) / total if total else 0.0

    wide = share > args.max_share
    if wide and not args.list_wide:
        print(_TOO_WIDE)
    elif selected:
        for t in selected:
            print(f"{t}  # {'; '.join(reasons[t])}" if args.why else t)
    else:
        print(_NOTHING_AFFECTED)
    print(f"{len(selected)} of {total} test files affected by {len(changed)} changed file(s)",
          file=sys.stderr)
    if wide:
        print(f"WIDE: {share:.0%} of the suite exceeds --max-share {args.max_share:.0%}; "
              "the change touches a module most tests import, so a targeted run "
              "cannot stand in for the suite" + ("" if args.list_wide else
                                                  "; --list-wide prints the selection"),
              file=sys.stderr)
        return _WIDE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
