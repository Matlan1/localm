#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decides which mutation shards a run needs, for the `mutation-scope` job in
.github/workflows/ci.yml, and keeps the test-file map that decision reads.

SELECTION. ``--all true`` (the weekly schedule, a dispatch, a pull request
carrying the ``mutation-test`` label) selects every ``[tool.mutmut]
only_mutate`` module. Otherwise the diff from the merge base with ``--base``
(default origin/master) to HEAD selects:

* every module, when the change touches the mutation gate itself
  (GATE_FILES), the ``[tool.mutmut]`` table of pyproject.toml (compared
  parsed, so an unrelated pyproject edit does not count), the text of a
  ``mutation-*`` job in .github/workflows/ci.yml, tests/conftest.py, or a
  ``tests.*`` module tests/conftest.py imports (directly or through another
  such module);
* a module whose source changed;
* the modules a changed test file of the mutmut test selection exercises,
  read from scripts/mutation_test_map.json; a selection file the map does not
  list selects every module;
* the modules of the selection files a changed test support file reaches: a
  nested conftest.py reaches the selection files under its directory, any
  other file under tests/ reaches the selection files whose source names it
  (a ``.py`` file by its stem, a file under tests/fixtures/ by its first
  directory, any other file by its name).

A changed ``test_*.py`` outside the selection selects nothing: mutmut never
runs it.

Prints ``modules=<JSON list of module stems>`` in only_mutate order and
``shards=<JSON list>``, the CI jobs those modules run as
(scripts/mutation_parts.py), then one line per reason. ``--github-output``
appends both lines to the file ``$GITHUB_OUTPUT`` names; ``--notice`` prints a
``::notice`` annotation when
any module is selected. Exits 1 when the diff cannot be computed, when
``only_mutate`` is empty or missing, or when the map is missing or malformed:
an unknown answer is never reported as an empty selection.

THE MAP. scripts/mutation_test_map.json lists, for every file of the mutmut
test selection, the only_mutate modules whose mutated functions that file's
tests execute, as mutmut's own stats run records it
(``tests_by_mangled_function_name`` in mutants/mutmut-stats.json): exactly
the tests mutmut runs against a module's mutants. ``--write-map OUT --stats
FILE...`` writes it from one or more stats files, keeping every edge the
committed map already records for a file still in the selection; ``--check-map --stats
FILE...`` exits 1 when a stats file records an edge the committed map lacks,
or when the map's files differ from the selection.

Run:  python scripts/mutation_scope.py [--base REF] [--all true|false]
                                       [--github-output] [--notice]
      python scripts/mutation_scope.py --write-map OUT --stats FILE...
      python scripts/mutation_scope.py --check-map --stats FILE...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402
import mutation_parts  # noqa: E402

MAP_FILE = "scripts/mutation_test_map.json"
MAP_SCHEMA = 1

GATE_FILES = (
    "scripts/mutation_baseline.json",
    "scripts/check_mutation_floors.py",
    "scripts/mutmut_run.py",
    "scripts/write_mutation_summary.py",
    "scripts/mutation_scope.py",
    "scripts/mutation_parts.py",
    ".github/requirements/mutmut.txt",
    MAP_FILE,
)

ROOT_CONFTEST = "tests/conftest.py"
WORKFLOW = ".github/workflows/ci.yml"

_JOB = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
_TESTS_IMPORT = re.compile(r"^\s*(?:from\s+tests\.([A-Za-z0-9_]+)\s+import|import\s+tests\.([A-Za-z0-9_]+)"
                           r"|from\s+tests\s+import\s+([A-Za-z0-9_, ]+))", re.M)


def mutation_jobs_text(workflow_text: str) -> str:
    """The lines of every job under ``jobs:`` whose id starts with
    ``mutation-``, from its id line up to the next job id."""
    out, keep, in_jobs = [], False, False
    for line in workflow_text.splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if in_jobs and line and not line.startswith(" ") and not line.startswith("#"):
            in_jobs, keep = False, False
        m = _JOB.match(line) if in_jobs else None
        if m:
            keep = m.group(1).startswith("mutation-")
        if keep:
            out.append(line.rstrip())
    return "\n".join(out)


def conftest_imports(read_text) -> list[str]:
    """tests/conftest.py and every ``tests/<name>.py`` it imports as
    ``tests.<name>``, followed through the imported files."""
    seen, todo = [], [ROOT_CONFTEST]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.append(rel)
        for a, b, c in _TESTS_IMPORT.findall(read_text(rel)):
            names = [a or b] if (a or b) else [n.strip() for n in c.split(",")]
            todo += [f"tests/{n}.py" for n in names if n and f"tests/{n}.py" not in seen]
    return sorted(seen)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def mutmut_section(pyproject_text: str) -> dict:
    """The parsed ``[tool.mutmut]`` table, ``{}`` when absent or unparsable."""
    try:
        return tomllib.loads(pyproject_text).get("tool", {}).get("mutmut", {}) or {}
    except (tomllib.TOMLDecodeError, ValueError):
        return {}


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def stem(module: str) -> str:
    """``localm/auth.py`` -> ``auth``."""
    return PurePosixPath(_norm(module)).stem


def touched(changed: list[str], modules: list[str], mutmut_config_changed: bool) -> list[str]:
    """The changed paths that put every module in scope or one module in
    scope: a mutated module, a gate file, or ``pyproject.toml`` when its
    ``[tool.mutmut]`` table changed."""
    norm = {_norm(c) for c in changed}
    watched = {_norm(m) for m in modules} | set(GATE_FILES)
    hits = sorted(norm & watched)
    if mutmut_config_changed and "pyproject.toml" in norm:
        hits.append("pyproject.toml ([tool.mutmut] changed)")
    return hits


def module_of_function(mangled: str) -> str:
    """``localm.auth.x_verify`` -> ``localm/auth.py``."""
    return mangled.rsplit(".", 1)[0].replace(".", "/") + ".py"


def build_map(stats_list: list[dict], selection: list[str], modules: list[str]) -> dict:
    """The map from one or more mutmut stats files: every selection file,
    with the only_mutate modules its tests execute (sorted, only_mutate
    order), the union over all *stats_list*."""
    order = {_norm(m): i for i, m in enumerate(modules)}
    edges: dict[str, set[str]] = {_norm(f): set() for f in selection}
    for stats in stats_list:
        for fn, tests in (stats.get("tests_by_mangled_function_name") or {}).items():
            module = module_of_function(fn)
            if module not in order:
                continue
            for test in tests:
                edges.setdefault(_norm(test.split("::", 1)[0]), set()).add(module)
    return {"schema": MAP_SCHEMA,
            "test_files": {f: sorted(ms, key=order.__getitem__) for f, ms in sorted(edges.items())}}


def merge_maps(built: dict, kept: dict[str, list[str]], modules: list[str]) -> dict:
    """*built* plus every edge *kept* (the committed map) records for a file
    still in *built*: a regenerated map never drops an edge."""
    order = {_norm(m): i for i, m in enumerate(modules)}
    out = {}
    for f, ms in built["test_files"].items():
        union = set(ms) | {m for m in kept.get(f, []) if m in order}
        out[f] = sorted(union, key=order.__getitem__)
    return {"schema": MAP_SCHEMA, "test_files": out}


def check_map(committed: dict, built: dict) -> list[str]:
    """Problems with *committed* against a map *built* from this run's stats:
    a file listed by one and not the other, a missing edge, a bad schema."""
    if not isinstance(committed, dict) or committed.get("schema") != MAP_SCHEMA \
            or not isinstance(committed.get("test_files"), dict):
        return [f"{MAP_FILE}: missing, malformed or not schema {MAP_SCHEMA}"]
    have = committed["test_files"]
    want = built["test_files"]
    problems = []
    for f in sorted(set(want) - set(have)):
        problems.append(f"{MAP_FILE}: {f} is in the mutmut test selection but not in the map")
    for f in sorted(set(have) - set(want)):
        problems.append(f"{MAP_FILE}: {f} is in the map but not in the mutmut test selection")
    for f in sorted(set(want) & set(have)):
        missing = [m for m in want[f] if m not in (have[f] or [])]
        if missing:
            problems.append(f"{MAP_FILE}: {f} exercises {', '.join(missing)}, which the map "
                            "does not list")
    return problems


def _mentions(text: str, key: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(key) + r"(?![\w])", text) is not None


def select(changed: list[str], modules: list[str], selection: list[str],
           test_map: dict[str, list[str]], config_changed: bool,
           read_text, workflow_changed: bool = False) -> tuple[list[str], list[str]]:
    """``(module paths to run, reasons)``. *read_text* returns a repository
    file's source at HEAD (``""`` when absent); *workflow_changed* says a
    ``mutation-*`` job of the workflow changed."""
    modules = [_norm(m) for m in modules]
    norm = sorted({_norm(c) for c in changed})
    sel = [_norm(s) for s in selection]
    picked: set[str] = set()
    reasons: list[str] = []

    every = [h for h in touched(norm, modules, config_changed) if h not in modules]
    every += [h for h in conftest_imports(read_text) if h in norm]
    if workflow_changed and WORKFLOW in norm:
        every.append(f"{WORKFLOW} (a mutation-* job)")
    if every:
        return list(modules), [f"every module: {h} changed" for h in every]

    def add(found: list[str], why: str) -> None:
        new = [m for m in found if m in modules]
        picked.update(new)
        if new:
            reasons.append(f"{', '.join(stem(m) for m in new)}: {why}")

    for path in norm:
        if path in modules:
            add([path], f"{path} changed")
        elif path in sel:
            if path not in test_map:
                return list(modules), [f"every module: {path} is in the test selection but "
                                       f"not in {MAP_FILE}"]
            add(test_map[path], f"{path} changed")
        elif path.startswith("tests/") and not PurePosixPath(path).name.startswith("test_"):
            p = PurePosixPath(path)
            if p.name == "conftest.py":
                under = str(p.parent) + "/"
                reached = [s for s in sel if s.startswith(under)]
            else:
                parts = p.parts
                if p.suffix == ".py":
                    key = p.stem
                elif len(parts) > 2 and parts[1] == "fixtures":
                    key = parts[2]
                else:
                    key = p.name
                reached = [s for s in sel if _mentions(read_text(s), key)]
            for s in reached:
                if s not in test_map:
                    return list(modules), [f"every module: {path} reaches {s}, which is not "
                                           f"in {MAP_FILE}"]
                add(test_map[s], f"{path} changed (reaches {s})")
    return [m for m in modules if m in picked], reasons


def _load_map(text: str) -> dict[str, list[str]]:
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("schema") != MAP_SCHEMA \
            or not isinstance(data.get("test_files"), dict):
        raise ValueError(f"not a schema {MAP_SCHEMA} map")
    return {_norm(k): [_norm(m) for m in v] for k, v in data["test_files"].items()}


def _read_stats(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/master",
                    help="ref the change is diffed from (via merge-base)")
    ap.add_argument("--all", choices=("true", "false"), default="false",
                    help="select every only_mutate module without diffing")
    ap.add_argument("--github-output", action="store_true",
                    help="also append modules=... to $GITHUB_OUTPUT")
    ap.add_argument("--notice", action="store_true",
                    help="print a ::notice annotation naming the selected modules")
    ap.add_argument("--write-map", type=Path, metavar="OUT",
                    help="write the test-file map built from --stats to OUT")
    ap.add_argument("--check-map", action="store_true",
                    help="check the committed map against --stats")
    ap.add_argument("--stats", type=Path, nargs="+", default=[],
                    help="mutants/mutmut-stats.json file(s) for --write-map / --check-map")
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.write_map or args.check_map:
        if not args.stats:
            ap.error("--write-map and --check-map need --stats")
        try:
            section = mutmut_section((args.repo / "pyproject.toml").read_text(encoding="utf-8"))
            built = build_map(_read_stats(args.stats),
                              [str(s) for s in section.get("pytest_add_cli_args_test_selection", [])],
                              [str(m) for m in section.get("only_mutate", [])])
        except (OSError, ValueError, AttributeError, TypeError) as e:
            print(f"mutation scope: could not read the stats: {e}", file=sys.stderr)
            return 1
        if args.write_map:
            try:
                kept = _load_map((args.repo / MAP_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError, AttributeError):
                kept = {}
            proposed = merge_maps(built, kept, [str(m) for m in section.get("only_mutate", [])])
            args.write_map.parent.mkdir(parents=True, exist_ok=True)
            args.write_map.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.write_map}")
        if args.check_map:
            try:
                committed = json.loads((args.repo / MAP_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                committed = None
            problems = check_map(committed, built)
            for p in problems:
                print(p, file=sys.stderr)
            if problems:
                print(f"\nRegenerate it: python scripts/mutation_scope.py --write-map {MAP_FILE} "
                      "--stats <every shard's mutmut-stats.json>", file=sys.stderr)
                return 1
            print(f"{MAP_FILE} lists every test-file edge this run recorded.")
        return 0

    try:
        head_pyproject = _git(args.repo, "show", "HEAD:pyproject.toml")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"mutation scope: could not read pyproject.toml at HEAD: "
              f"{(getattr(e, 'stderr', '') or str(e)).strip()}", file=sys.stderr)
        return 1
    section = mutmut_section(head_pyproject)
    modules = [_norm(str(m)) for m in section.get("only_mutate", [])]
    if not modules:
        print("mutation scope: could not resolve scope: [tool.mutmut] only_mutate "
              "is empty or missing", file=sys.stderr)
        return 1

    if args.all == "true":
        picked, reasons = list(modules), ["every module: a full mutation run was requested"]
    else:
        try:
            merge_base = _git(args.repo, "merge-base", args.base, "HEAD").strip()
            changed = [c for c in _git(args.repo, "diff", "--name-only", merge_base, "HEAD").splitlines() if c]
            base_pyproject = _git(args.repo, "show", f"{merge_base}:pyproject.toml")
        except (subprocess.CalledProcessError, OSError) as e:
            detail = getattr(e, "stderr", "") or str(e)
            print(f"mutation scope: could not diff against {args.base}: {detail.strip()}",
                  file=sys.stderr)
            return 1
        try:
            test_map = _load_map(_git(args.repo, "show", f"HEAD:{MAP_FILE}"))
        except (subprocess.CalledProcessError, OSError, ValueError, AttributeError) as e:
            print(f"mutation scope: could not read {MAP_FILE} at HEAD: {e}", file=sys.stderr)
            return 1
        selection = [str(s) for s in section.get("pytest_add_cli_args_test_selection", [])]

        def read_text(rel: str) -> str:
            try:
                return _git(args.repo, "show", f"HEAD:{rel}")
            except (subprocess.CalledProcessError, OSError):
                return ""

        def read_at(ref: str, rel: str) -> str:
            try:
                return _git(args.repo, "show", f"{ref}:{rel}")
            except (subprocess.CalledProcessError, OSError):
                return ""

        workflow_changed = (mutation_jobs_text(read_at(merge_base, WORKFLOW))
                            != mutation_jobs_text(read_at("HEAD", WORKFLOW)))
        picked, reasons = select(changed, modules, selection, test_map,
                                 section != mutmut_section(base_pyproject), read_text,
                                 workflow_changed)

    stems = [stem(m) for m in picked]
    line = "modules=" + json.dumps(stems, separators=(",", ":"))
    jobs = "shards=" + json.dumps(mutation_parts.shards(stems), separators=(",", ":"))
    print(line)
    print(jobs)
    for r in reasons:
        print(f"  {r}")
    if args.github_output:
        ci_runner_files.append(ci_runner_files.OUTPUT, line + "\n" + jobs + "\n")
    if args.notice and stems:
        print("::notice title=Mutation shards run on this change::"
              f"{', '.join(stems)} ({'; '.join(reasons)}). Every other module is judged "
              "from the committed baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
