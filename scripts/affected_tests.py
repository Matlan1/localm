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
  - it imports a changed dependency, names it in a string literal, or imports
    a module that imports it (with --depth N, also that module's importers,
    up to N hops). A changed dependency is a distribution whose requirement in
    pyproject.toml or whose entry in uv.lock differs, and every package in
    uv.lock that depends on one, transitively.

A change to tests/conftest.py affects every test file. So does a change to
pyproject.toml or uv.lock outside the [project] requirement lists and the
locked packages, either file missing or unparsable on either side, and a
changed dependency the project declares outside the dev extra that nothing in
the tree imports or names. A changed dependency declared only in the dev extra
that nothing imports or names selects no test file.

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
_DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
_DEV_EXTRA = "dev"
_REQUIREMENT_KEYS = ("dependencies", "optional-dependencies")
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
        for rel in self.test_files:
            names = imported_names(_read(rel), module_name(rel), False)
            self.test_top[rel] = _top_levels(names)
            self.test_imports[rel] = self.resolve(names)

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
    """*pyproject* without the [project] requirement lists."""
    project = {k: v for k, v in pyproject.get("project", {}).items() if k not in _REQUIREMENT_KEYS}
    return {**pyproject, "project": project}


def _lock_packages(lock: dict, project: str) -> dict[str, list[str]]:
    """{distribution: the canonical JSON of each of its uv.lock entries}, with
    the dependency declarations left out of *project*'s own entry."""
    out: dict[str, list[str]] = {}
    for entry in lock.get("package", []):
        name = dist_key(str(entry.get("name", "")))
        if name == project:
            meta = {k: v for k, v in entry.get("metadata", {}).items()
                    if k not in _ROOT_METADATA_DECLARATIONS}
            entry = {**{k: v for k, v in entry.items() if k not in _ROOT_DECLARATIONS},
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
    if ({k: v for k, v in lock_old.items() if k != "package"}
            != {k: v for k, v in lock_new.items() if k != "package"}):
        return None
    project = dist_key(str(py_new.get("project", {}).get("name", "")))
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


def _dependency_uses(graph: Graph, base_ref: str, test_text) -> list[tuple] | None:
    """(distribution, label, import names, importing modules, name pattern)
    for each affected distribution the tree imports or names; None when every
    test file is affected: dependency_change() returned None, or an affected
    distribution the project requires outside the dev extra is imported and
    named nowhere. *test_text* reads a test file."""
    change = dependency_change(_read_at(base_ref, "pyproject.toml"), _read("pyproject.toml"),
                               _read_at(base_ref, "uv.lock"), _read("uv.lock"))
    if change is None:
        return None
    changed, affected, runtime = change
    uses = []
    for dist in sorted(affected):
        names = import_names(dist)
        users = {m for m, top in graph.source_top.items() if top & names}
        named = _names_in_a_string(names | {dist})
        if not users and not any(graph.test_top[t] & names or named.search(test_text(t))
                                 for t in graph.test_files):
            if dist in runtime:
                return None
            continue
        label = f"changed dependency {dist}" if dist in changed else \
            f"dependency {dist} (it depends on a changed package)"
        uses.append((dist, label, names, users, named))
    return uses


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
    uses: list[tuple] = []
    if any(c in _DEPENDENCY_FILES for c in changed):
        found = _dependency_uses(graph, base_ref, test_text)
        if found is None:
            whole += [c for c in changed if c in _DEPENDENCY_FILES]
        else:
            uses = found
    if whole:
        which = ", ".join(sorted(whole))
        for t in graph.test_files:
            add(t, f"every test file: {which} changed")
        return reasons

    changed_modules: set[str] = set()
    needles: list[tuple[str, re.Pattern[str]]] = []
    for rel in changed:
        if rel in graph.test_imports:
            add(rel, "changed")
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

    hops = graph.importers(changed_modules, depth)
    dependency_hops = [(dist, label, names, graph.importers(users, depth), named)
                       for dist, label, names, users, named in uses]
    for t in graph.test_files:
        hit = graph.test_imports[t] & set(hops)
        for mod in sorted(hit, key=lambda m: (hops[m], m)):
            add(t, f"imports {mod}" if hops[mod] == 0 else f"imports {mod} ({hops[mod]} hop(s) from a change)")
        if needles:
            for label, rx in needles:
                if rx.search(test_text(t)):
                    add(t, label)
        for dist, label, names, dep_hops, named in dependency_hops:
            for name in sorted(graph.test_top[t] & names):
                add(t, f"imports {name}: {label}")
            for mod in sorted(graph.test_imports[t] & set(dep_hops), key=lambda m: (dep_hops[m], m)):
                add(t, f"imports {mod}, which imports {dist}: {label}" if dep_hops[mod] == 0
                    else f"imports {mod} ({dep_hops[mod]} hop(s) from a module importing {dist}): {label}")
            if named.search(test_text(t)):
                add(t, f"names {dist}: {label}")
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
