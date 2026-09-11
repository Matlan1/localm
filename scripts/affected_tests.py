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
    non-Python file by file name.

A change to tests/conftest.py, pyproject.toml or uv.lock affects every test
file.

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
import re
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "websocket", "api_route"}
_EVERYTHING = {"tests/conftest.py", "pyproject.toml", "uv.lock"}
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
            for dep in self.resolve(imported_names(_read(rel), mod, rel.endswith("__init__.py"))):
                if dep != mod:
                    self.reverse.setdefault(dep, set()).add(mod)
        for rel in self.test_files:
            self.test_imports[rel] = self.resolve(imported_names(_read(rel), module_name(rel), False))

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
    try:
        return (REPO / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _read_at(ref: str, rel: str) -> str:
    try:
        return _git("show", f"{ref}:{rel}")
    except subprocess.CalledProcessError:
        return ""


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

    if any(c in _EVERYTHING for c in changed):
        which = ", ".join(c for c in changed if c in _EVERYTHING)
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
    for t in graph.test_files:
        hit = graph.test_imports[t] & set(hops)
        for mod in sorted(hit, key=lambda m: (hops[m], m)):
            add(t, f"imports {mod}" if hops[mod] == 0 else f"imports {mod} ({hops[mod]} hop(s) from a change)")
        if needles:
            text = _read(t)
            for label, rx in needles:
                if rx.search(text):
                    add(t, label)
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
