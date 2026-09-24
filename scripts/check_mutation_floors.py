#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-module mutation-score floors and per-mutant dispositions for the trust
boundary: the ``[tool.mutmut] only_mutate`` modules in pyproject.toml, the same
eight scripts/check_coverage_floors.py holds line-coverage floors for.

WHAT THIS GATES. A mutmut run (see .github/workflows/ci.yml, the mutation-test
job) writes one ``mutants/<module path>.meta`` file per mutated module with
every generated mutant's exit code and a hash of every mutated function's
source. This script compares those results against the committed baseline
``scripts/mutation_baseline.json`` and exits 1 when any of these holds:

* a module's mutation score is below its ``score_floor``;
* a mutant the baseline records as ``killed`` is no longer killed (a test
  that used to catch it was weakened or removed, or the mutant's outcome is
  nondeterministic);
* a mutant has no disposition: it is new, or the source of its function
  changed since the baseline (mutant ids are numbered per function, so a
  changed function's dispositions are stale), or its disposition is not one
  of ``killed`` / ``survived`` / ``{"equivalent": "<reason>"}`` /
  ``{"unstable": "<reason>"}``;
* a control mutant (the ``controls`` section: one concrete mutant per
  security-decision class that must stay killed) is missing, stale, recorded
  as unstable or not killed;
* the run is incomplete (a mutant with no outcome), the baseline is missing
  or malformed, or an ``only_mutate`` module has no results or no baseline.

DISPOSITIONS. ``killed`` and ``survived`` are strings; ``survived`` is a known
test gap that counts against the score, and the floor ratchet stops the count
of them growing. An equivalent mutant (one no test can ever distinguish from
the original) is written as ``{"equivalent": "<reason>"}`` with a non-empty
reason and is excluded from the score. An unstable mutant (one whose outcome
differs between runs of the same source and tests, e.g. with the hash seed)
is written as ``{"unstable": "<reason>"}`` with a non-empty reason; it is
excluded from the score and never counts as a regression. Every mutant mutmut
generates is either scored or carries a written reason.

SCORE. ``killed-like / (all mutants - equivalent - unstable)``, in percent,
where killed-like is ``killed``, ``timeout`` and ``caught by type check`` (the
mutant was detected: the tests did not pass under it) and everything else -
``survived``, ``no tests``, ``suspicious``, ``segfault``, ``skipped`` -
counts as not detected. A baseline mutant that mutmut no longer generates
although its function is unchanged (a ``pragma: no mutate``) fails the check
until its entry is classified equivalent or deleted by hand. Recording a
killed mutant as unstable removes a detected mutant from the score, which can
take the score below its floor; that floor is then lowered by hand in the same
PR. ``score_floor`` is the score measured when the baseline was
written, rounded DOWN to two decimals, so it can only be raised by ``--update``
and never lowered by it; lowering one is a hand edit in the same PR that
explains why.

RUN. After ``mutmut run`` (Linux/WSL only; mutmut refuses to run on Windows):

    python scripts/check_mutation_floors.py [--mutants DIR] [--baseline FILE]

``--update`` writes a baseline from the current results: floors ratchet up,
``equivalent`` and ``unstable`` entries and their reasons are kept, every
other mutant gets ``killed`` or ``survived`` from its outcome, and the mutants
of a changed or removed function are pruned. Without ``--out`` it replaces the ``--baseline``
file and the check then runs against the new file; with ``--out FILE`` the
proposal goes to FILE and the check still runs against the committed
baseline, so the CI job can publish the proposal as an artifact on every run
without it ever standing in for the file under review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402
DEFAULT_BASELINE = REPO / "scripts" / "mutation_baseline.json"
DEFAULT_MUTANTS_DIR = REPO / "mutants"
SCHEMA_VERSION = 1

# mutmut 3.7.0's status_by_exit_code (mutmut/__main__.py). Any exit code not
# listed is "suspicious", as in mutmut.
_STATUS_BY_EXIT_CODE: dict[int | None, str] = {
    None: "not checked",
    0: "survived",
    1: "killed",
    2: "check was interrupted by user",
    3: "killed",
    5: "no tests",
    24: "timeout",
    -24: "timeout",
    33: "no tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: "caught by type check",
    152: "timeout",
    255: "timeout",
    -9: "segfault",
    -11: "segfault",
}

KILLED_LIKE = frozenset({"killed", "timeout", "caught by type check"})
NOT_DETECTED = frozenset({"survived", "no tests", "suspicious", "segfault", "skipped"})
INCOMPLETE = frozenset({"not checked", "check was interrupted by user"})


def status_of(exit_code: int | None) -> str:
    """mutmut's outcome name for a recorded exit code."""
    return _STATUS_BY_EXIT_CODE.get(exit_code, "suspicious")


def function_of(mutant: str) -> str:
    """The mangled function name a mutant id belongs to:
    ``localm.scopes.x_grants__mutmut_2`` -> ``x_grants``,
    ``localm.auth.xǁFooǁbar__mutmut_3`` -> ``xǁFooǁbar``."""
    for part in reversed(mutant.split(".")):
        if part.startswith(("x_", "xǁ")):
            return part.split("__mutmut_", 1)[0]
    raise ValueError(f"not a mutmut mutant id: {mutant!r}")


def only_mutate_modules(pyproject_text: str) -> list[str]:
    """The ``[tool.mutmut] only_mutate`` list, forward-slash paths."""
    import tomllib
    data = tomllib.loads(pyproject_text)
    modules = data.get("tool", {}).get("mutmut", {}).get("only_mutate", [])
    return [str(m).replace("\\", "/") for m in modules]


def load_results(mutants_dir: Path, modules: list[str]) -> tuple[dict, list[str]]:
    """``{module: {"mutants": {id: status}, "function_hashes": {...}}}`` read
    from each module's ``.meta`` file, plus the problems found on the way (a
    module with no results is a problem, not an empty entry)."""
    results: dict[str, dict] = {}
    problems: list[str] = []
    for module in modules:
        meta_path = mutants_dir / (module + ".meta")
        if not meta_path.is_file():
            problems.append(
                f"{module}: no mutation results at {meta_path} - mutmut did not "
                "mutate this module (a failed clean-test run, a config change, or "
                "the run never happened); a missing result is not a passing one")
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            exit_codes = meta["exit_code_by_key"]
            hashes = meta.get("hash_by_function_name", {})
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, KeyError, TypeError) as e:
            problems.append(f"{module}: could not read {meta_path.name}: {e}")
            continue
        results[module] = {
            "mutants": {k: status_of(v) for k, v in exit_codes.items()},
            "function_hashes": dict(hashes),
        }
    return results, problems


EXCLUDED_KINDS = frozenset({"equivalent", "unstable"})


def _disposition(value) -> tuple[str, str | None]:
    """``(kind, reason)`` for a baseline disposition value, or ``("invalid",
    None)``. An equivalent or unstable entry needs a non-empty reason."""
    if value in ("killed", "survived"):
        return value, None
    if isinstance(value, dict) and len(value) == 1:
        ((kind, reason),) = value.items()
        if kind in EXCLUDED_KINDS and isinstance(reason, str) and reason.strip():
            return kind, reason
    return "invalid", None


def module_score(statuses: dict[str, str], equivalents: set[str]) -> tuple[float, int, int]:
    """``(score percent, detected, scored)`` over one module's outcomes.
    A module with nothing to score reports 100.0."""
    scored = 0
    detected = 0
    for mutant, status in statuses.items():
        if mutant in equivalents or status in INCOMPLETE:
            continue
        scored += 1
        if status in KILLED_LIKE:
            detected += 1
    score = 100.0 * detected / scored if scored else 100.0
    return score, detected, scored


def floor_two_decimals(score: float) -> float:
    """A score rounded DOWN to two decimals: the floor a baseline records."""
    return math.floor(score * 100 + 1e-9) / 100


def check(results: dict, baseline: dict, modules: list[str]) -> tuple[list[str], list[str], list[dict]]:
    """Pure check of parsed results against a parsed baseline. Returns
    ``(problems, warnings, rows)``: a non-empty ``problems`` is a failure;
    ``rows`` is one summary dict per module for the report."""
    problems: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []

    if baseline.get("schema") != SCHEMA_VERSION:
        problems.append(
            f"baseline schema is {baseline.get('schema')!r}, this script reads "
            f"{SCHEMA_VERSION}")
        return problems, warnings, rows
    base_modules = baseline.get("modules")
    if not isinstance(base_modules, dict):
        problems.append("baseline has no 'modules' table")
        return problems, warnings, rows
    for module in base_modules:
        if module not in modules:
            warnings.append(
                f"{module}: in the baseline but not in [tool.mutmut] only_mutate "
                "- stale entry, --update drops it")

    for module in modules:
        res = results.get(module)
        if res is None:
            continue
        base = base_modules.get(module)
        if not isinstance(base, dict):
            problems.append(
                f"{module}: no baseline entry - add one with --update and "
                "classify every survivor before committing it")
            continue
        base_mutants = base.get("mutants") or {}
        base_hashes = base.get("function_hashes") or {}
        statuses: dict[str, str] = res["mutants"]
        hashes: dict[str, str] = res["function_hashes"]

        incomplete = sorted(m for m, s in statuses.items() if s in INCOMPLETE)
        if incomplete:
            problems.append(
                f"{module}: {len(incomplete)} mutant(s) have no outcome (the run "
                f"did not complete), e.g. {incomplete[0]}")

        changed_functions = {
            fn for fn, h in hashes.items() if base_hashes.get(fn) != h}
        equivalents: set[str] = set()
        unstables: set[str] = set()
        undispositioned: list[str] = []
        regressions: list[str] = []
        promotable: list[str] = []
        invalid: list[str] = []
        for mutant, status in statuses.items():
            fn = function_of(mutant)
            value = base_mutants.get(mutant)
            if value is None or fn in changed_functions:
                if status not in INCOMPLETE:
                    undispositioned.append(f"{mutant} ({status})")
                continue
            kind, _reason = _disposition(value)
            if kind == "invalid":
                invalid.append(mutant)
                continue
            if kind == "unstable":
                unstables.add(mutant)
                continue
            if kind == "equivalent":
                equivalents.add(mutant)
                if status in KILLED_LIKE:
                    warnings.append(
                        f"{module}: {mutant} is classified equivalent but was "
                        f"{status} - a test does distinguish it; reclassify it "
                        "as killed")
                continue
            if kind == "killed" and status not in KILLED_LIKE and status not in INCOMPLETE:
                regressions.append(f"{mutant} ({status})")
            elif kind == "survived" and status in KILLED_LIKE:
                promotable.append(mutant)

        vanished = sorted(m for m in base_mutants if m not in statuses)
        unexplained = [m for m in vanished
                       if hashes.get(function_of(m)) == base_hashes.get(function_of(m))]
        if unexplained:
            problems.append(
                f"{module}: {len(unexplained)} baseline mutant(s) were not generated "
                f"although their function is unchanged (e.g. {unexplained[0]}) - a "
                "`pragma: no mutate` or a mutmut version change; an excluded mutant "
                "needs {\"equivalent\": \"<reason>\"} in the baseline, or delete "
                "the entry by hand in the same PR")
        elif vanished:
            warnings.append(
                f"{module}: {len(vanished)} baseline mutant(s) belong to functions "
                f"that changed or were removed (e.g. {vanished[0]}) - --update prunes them")
        if promotable:
            warnings.append(
                f"{module}: {len(promotable)} mutant(s) recorded as survived are "
                f"now killed (e.g. {promotable[0]}) - --update records the "
                "improvement and raises the floor")
        if invalid:
            problems.append(
                f"{module}: {len(invalid)} mutant(s) carry an invalid disposition "
                f"(e.g. {invalid[0]}); allowed: \"killed\", \"survived\", "
                "{\"equivalent\": \"<non-empty reason>\"}, "
                "{\"unstable\": \"<non-empty reason>\"}")
        if undispositioned:
            fns = sorted({function_of(m.split(" ", 1)[0]) for m in undispositioned})
            problems.append(
                f"{module}: {len(undispositioned)} mutant(s) have no disposition "
                f"in the baseline (new, or their function changed: {', '.join(fns)}"
                f"), e.g. {undispositioned[0]} - run --update, then classify every "
                "survivor among them (kill it with a test, or record "
                "{\"equivalent\": \"<reason>\"}) before committing the baseline")
        if regressions:
            problems.append(
                f"{module}: {len(regressions)} mutant(s) recorded as killed are no "
                f"longer detected: {', '.join(regressions[:5])}"
                f"{' ...' if len(regressions) > 5 else ''} - either a test that "
                "caught them was weakened or removed, or their outcome is "
                "nondeterministic (it can depend on the hash seed, timing or test "
                "order). Re-run each one with `PYTHONHASHSEED=<n> python "
                "scripts/mutmut_run.py run <mutant id>`, for the seed the "
                "mutation-run step printed and a few others: if a re-run kills "
                "it, make the test that catches it deterministic or record it as "
                "{\"unstable\": \"<reason>\"}; if every re-run survives, restore "
                "the test")

        score, detected, scored = module_score(statuses, equivalents | unstables)
        floor = base.get("score_floor")
        if not isinstance(floor, (int, float)) or isinstance(floor, bool):
            problems.append(f"{module}: baseline score_floor is not a number")
            floor = None
        elif score < floor - 1e-9:
            problems.append(
                f"{module}: mutation score {score:.2f}% is BELOW its floor of "
                f"{floor:.2f}% ({detected}/{scored} detected) - kill the new "
                "survivors, classify a genuinely equivalent one with a reason, or "
                "lower the floor by hand in the SAME PR that explains why")
        counts = {s: 0 for s in sorted(set(_STATUS_BY_EXIT_CODE.values()))}
        for status in statuses.values():
            counts[status] = counts.get(status, 0) + 1
        rows.append({
            "module": module, "total": len(statuses), "detected": detected,
            "scored": scored, "equivalent": len(equivalents),
            "unstable": len(unstables), "score": score,
            "floor": floor, "counts": counts,
        })

    controls = baseline.get("controls") or {}
    if not isinstance(controls, dict):
        problems.append("baseline 'controls' is not a table")
        controls = {}
    for name, ctl in controls.items():
        if not isinstance(ctl, dict) or not isinstance(ctl.get("mutant"), str) \
                or not isinstance(ctl.get("module"), str):
            problems.append(f"control {name!r}: needs 'module' and 'mutant'")
            continue
        module, mutant = ctl["module"], ctl["mutant"]
        res = results.get(module)
        if res is None:
            if module in modules:
                continue  # already reported as missing results
            problems.append(f"control {name!r}: module {module} is not mutated")
            continue
        fn = function_of(mutant)
        base_hash = ((base_modules.get(module) or {}).get("function_hashes") or {}).get(fn)
        if res["function_hashes"].get(fn) != base_hash:
            problems.append(
                f"control {name!r}: {fn} in {module} changed since the baseline, so "
                f"{mutant} may no longer be the mutation it was pinned for - "
                "re-verify which mutant now expresses this class and re-pin it")
            continue
        base_value = ((base_modules.get(module) or {}).get("mutants") or {}).get(mutant)
        if _disposition(base_value)[0] == "unstable":
            problems.append(
                f"control {name!r}: {mutant} is recorded as unstable in the "
                "baseline, and a control must be killed on every run - pin a "
                "mutant of the same class that the tests kill deterministically")
            continue
        status = res["mutants"].get(mutant)
        if status is None:
            problems.append(f"control {name!r}: {mutant} was not generated")
        elif status not in KILLED_LIKE:
            problems.append(
                f"control {name!r}: {mutant} is {status}, must be killed - "
                f"{ctl.get('describes', 'a security decision')} no longer has a "
                "test that catches it")
    return problems, warnings, rows


def propose_baseline(results: dict, baseline: dict, modules: list[str]) -> dict:
    """A baseline built from ``results``: floors ratchet up from ``baseline``,
    equivalent and unstable classifications survive, everything else follows
    the outcome, and the mutants of a changed or removed function are dropped.
    A vanished mutant of an unchanged function keeps its old entry."""
    old_modules = baseline.get("modules") or {}
    new_modules: dict[str, dict] = {}
    for module in modules:
        res = results.get(module)
        if res is None:
            if module in old_modules:
                new_modules[module] = old_modules[module]
            continue
        old = old_modules.get(module) or {}
        old_mutants = old.get("mutants") or {}
        old_hashes = old.get("function_hashes") or {}
        changed = {fn for fn, h in res["function_hashes"].items()
                   if old_hashes.get(fn) != h}
        mutants: dict = {}
        excluded: set[str] = set()
        for mutant, status in sorted(res["mutants"].items()):
            old_value = old_mutants.get(mutant)
            kind, _ = _disposition(old_value) if old_value is not None else ("invalid", None)
            if kind in EXCLUDED_KINDS and function_of(mutant) not in changed:
                mutants[mutant] = old_value
                excluded.add(mutant)
            elif status in KILLED_LIKE:
                mutants[mutant] = "killed"
            elif status in INCOMPLETE:
                continue
            else:
                mutants[mutant] = "survived"
        for mutant, old_value in sorted(old_mutants.items()):
            fn = function_of(mutant)
            if mutant not in res["mutants"] and fn not in changed                     and fn in res["function_hashes"]:
                mutants[mutant] = old_value
        score, _, _ = module_score(res["mutants"], excluded)
        old_floor = old.get("score_floor")
        floor = floor_two_decimals(score)
        if isinstance(old_floor, (int, float)) and not isinstance(old_floor, bool):
            floor = max(floor, float(old_floor))
        new_modules[module] = {
            "score_floor": floor,
            "function_hashes": dict(sorted(res["function_hashes"].items())),
            "mutants": mutants,
        }
    return {
        "schema": SCHEMA_VERSION,
        "mutmut": baseline.get("mutmut", ""),
        "modules": new_modules,
        "controls": baseline.get("controls") or {},
    }


def render_table(rows: list[dict]) -> str:
    """Plain-text per-module table for the log."""
    lines = [f"{'module':<22} {'mutants':>7} {'detected':>8} {'scored':>6} "
             f"{'equiv':>5} {'unst':>4} {'surv':>5} {'notest':>6} {'susp':>4} {'score':>8} {'floor':>8}"]
    for r in rows:
        c = r["counts"]
        floor = f"{r['floor']:.2f}%" if isinstance(r["floor"], (int, float)) else "?"
        lines.append(
            f"{r['module']:<22} {r['total']:>7} {r['detected']:>8} {r['scored']:>6} "
            f"{r['equivalent']:>5} {r['unstable']:>4} {c.get('survived', 0):>5} {c.get('no tests', 0):>6} "
            f"{c.get('suspicious', 0) + c.get('segfault', 0):>4} {r['score']:>7.2f}% {floor:>8}")
    return "\n".join(lines)


def render_summary(rows: list[dict], problems: list[str], warnings: list[str]) -> str:
    """Markdown for the GitHub Actions step summary."""
    lines = ["### Mutation floors (trust boundary)", ""]
    if rows:
        lines += ["| module | mutants | detected | score | floor |", "|---|---:|---:|---:|---:|"]
        for r in rows:
            floor = f"{r['floor']:.2f}%" if isinstance(r["floor"], (int, float)) else "?"
            lines.append(f"| `{r['module']}` | {r['total']} | {r['detected']}/{r['scored']} "
                         f"| {r['score']:.2f}% | {floor} |")
        lines.append("")
    if problems:
        lines.append(f"**FAILED** - {len(problems)} problem(s):")
        lines += [f"- {p}" for p in problems]
    else:
        lines.append("**Passed**: every floor holds, every mutant has a disposition, every control is killed.")
    if warnings:
        lines += ["", "<details><summary>Warnings</summary>", ""]
        lines += [f"- {w}" for w in warnings]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _publish(text: str) -> None:
    ci_runner_files.append(ci_runner_files.STEP_SUMMARY, text)


def _read_json(path: Path, what: str) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"{what} not found at {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        return None, f"{what} at {path} is not valid JSON: {e}"
    if not isinstance(data, dict):
        return None, f"{what} at {path} is not a JSON object"
    return data, None


def main(argv: list[str]) -> int:
    # Mutant ids carry mutmut's class separator (U+01C1); a console that
    # cannot encode it prints a replacement character instead of failing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mutants", type=Path, default=DEFAULT_MUTANTS_DIR,
                    help="mutmut output directory (default: <repo>/mutants)")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                    help="baseline JSON (default: scripts/mutation_baseline.json)")
    ap.add_argument("--pyproject", type=Path, default=REPO / "pyproject.toml",
                    help=argparse.SUPPRESS)
    ap.add_argument("--update", action="store_true",
                    help="write a baseline from the current results, then check it")
    ap.add_argument("--out", type=Path, default=None,
                    help="with --update: where to write (default: the --baseline path)")
    args = ap.parse_args(argv)

    try:
        modules = only_mutate_modules(args.pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Mutation floor check could not run: {args.pyproject}: {e}", file=sys.stderr)
        return 1
    if not modules:
        print("Mutation floor check could not run: [tool.mutmut] only_mutate is empty",
              file=sys.stderr)
        return 1

    results, problems = load_results(args.mutants, modules)
    if not results:
        problems.append(
            f"no mutation results under {args.mutants} at all - run "
            "`python scripts/mutmut_run.py run` first (Linux/WSL)")
        for p in problems:
            print("  " + p, file=sys.stderr)
        _publish(render_summary([], problems, []))
        return 1

    baseline, err = _read_json(args.baseline, "mutation baseline")
    if baseline is None:
        if args.update:
            baseline = {"schema": SCHEMA_VERSION, "mutmut": "", "modules": {}, "controls": {}}
        else:
            print(f"Mutation floor check could not run: {err}. Generate one with "
                  "--update and classify every survivor in it.", file=sys.stderr)
            return 1

    if args.update:
        proposed = propose_baseline(results, baseline, modules)
        out = args.out or args.baseline
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(proposed, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"Wrote {out}")
        if out.resolve() == args.baseline.resolve():
            baseline = proposed

    more, warnings, rows = check(results, baseline, modules)
    problems.extend(more)

    print(render_table(rows))
    for w in warnings:
        print("WARNING: " + w)
    sys.stdout.flush()
    _publish(render_summary(rows, problems, warnings))
    if problems:
        print("\nMutation floor check FAILED (see scripts/check_mutation_floors.py):\n",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"\n{len(problems)} problem(s).", file=sys.stderr)
        return 1
    print(f"\nMutation floor check passed ({len(rows)} trust-boundary modules, "
          f"{len(baseline.get('controls') or {})} controls killed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
