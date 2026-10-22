#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Splits a mutation-tested module's mutants over parallel CI jobs and merges
the parts' results back, for the mutation-run and mutation-test jobs in
.github/workflows/ci.yml.

A module listed in PARTS runs as that many jobs; every other module as one.
Part k of n tests the mutants whose id ends in the k-th group of the digits
0-9 (``localm.auth.*__mutmut_*[0-3]``), so every function's mutants spread
over all the parts. ``shards(stems)`` is the CI matrix: one
``{"module", "part", "pattern"}`` entry per job.

Each part's ``mutants/localm/<module>.py.meta`` lists every mutant of the
module, with an exit code only for its own. ``merge`` combines the parts of
each module into one ``.meta`` (and copies the mutated module and the stats
beside it): an exit code from exactly one part, the function hashes every
part agrees on. Exits 1 when a mutant has an exit code in two parts, when the
parts disagree on a function hash, or when a part is missing or unreadable.

Run:  python scripts/mutation_parts.py merge --parts DIR --out DIR

DIR (``--parts``) holds one directory per downloaded artifact,
``mutation-meta-<module>-<part>``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PARTS = {"auth": 3, "config": 3}

_ARTIFACT = re.compile(r"^mutation-meta-(?P<module>[A-Za-z0-9_]+)-(?P<part>\d+)$")


def digit_groups(n: int) -> list[str]:
    """The ten digits in *n* contiguous groups, as fnmatch character sets:
    ``digit_groups(3) == ["0-3", "4-6", "7-9"]``."""
    if not 1 <= n <= 10:
        raise ValueError(f"parts must be 1-10, got {n}")
    base, extra = divmod(10, n)
    groups, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        end = start + size - 1
        groups.append(f"{start}-{end}" if size > 1 else str(start))
        start = end + 1
    return groups


def shards(stems: list[str]) -> list[dict]:
    """One CI job per module part, in *stems* order."""
    out = []
    for stem in stems:
        n = PARTS.get(stem, 1)
        if n == 1:
            out.append({"module": stem, "part": 1, "parts": 1, "pattern": f"localm.{stem}.*"})
            continue
        for k, group in enumerate(digit_groups(n), start=1):
            out.append({"module": stem, "part": k, "parts": n,
                        "pattern": f"localm.{stem}.*__mutmut_*[{group}]"})
    return out


def merge_metas(metas: list[dict]) -> tuple[dict, list[str]]:
    """``(merged meta, problems)`` for the parts of one module."""
    problems: list[str] = []
    merged = {"exit_code_by_key": {}, "hash_by_function_name": {}, "type_check_error_by_key": {},
              "durations_by_key": {}, "estimated_durations_by_key": {}}
    owner: dict[str, int] = {}
    for i, meta in enumerate(metas):
        for key, code in (meta.get("exit_code_by_key") or {}).items():
            if code is None:
                merged["exit_code_by_key"].setdefault(key, None)
                continue
            if key in owner:
                problems.append(f"{key}: an exit code in part {owner[key] + 1} and part {i + 1}")
                continue
            owner[key] = i
            merged["exit_code_by_key"][key] = code
        for fn, h in (meta.get("hash_by_function_name") or {}).items():
            seen = merged["hash_by_function_name"].setdefault(fn, h)
            if seen != h:
                problems.append(f"{fn}: the parts disagree on the function hash")
        for field in ("type_check_error_by_key", "durations_by_key", "estimated_durations_by_key"):
            for key, value in (meta.get(field) or {}).items():
                if field == "estimated_durations_by_key" or key in owner and owner[key] == i:
                    merged[field][key] = value
    return merged, problems


def merge(parts_dir: Path, out_dir: Path) -> list[str]:
    """Merge every ``mutation-meta-<module>-<part>`` directory under
    *parts_dir* into ``<out_dir>/<module>.py.meta`` and copy the mutated
    module and the stats files beside it. Returns the problems."""
    by_module: dict[str, dict[int, Path]] = {}
    problems: list[str] = []
    for d in sorted(p for p in parts_dir.iterdir() if p.is_dir()) if parts_dir.is_dir() else []:
        m = _ARTIFACT.match(d.name)
        if m:
            by_module.setdefault(m["module"], {})[int(m["part"])] = d
    if not by_module:
        return [f"no mutation-meta-<module>-<part> directory under {parts_dir}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, parts in sorted(by_module.items()):
        n = PARTS.get(module, 1)
        missing = [k for k in range(1, n + 1) if k not in parts]
        if missing:
            problems.append(f"{module}: part(s) {', '.join(map(str, missing))} of {n} have no results")
        metas = []
        for k, d in sorted(parts.items()):
            meta_path = d / f"{module}.py.meta"
            try:
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                problems.append(f"{module} part {k}: could not read {meta_path.name}: {e}")
                continue
            for name in (f"{module}.py",):
                src = d / name
                if src.is_file() and not (out_dir / name).is_file():
                    shutil.copyfile(src, out_dir / name)
            stats = d / f"{module}.stats.json"
            if stats.is_file():
                shutil.copyfile(stats, out_dir / f"{module}.{k}.stats.json")
        if not metas:
            continue
        merged, more = merge_metas(metas)
        problems += [f"{module}: {p}" for p in more]
        (out_dir / f"{module}.py.meta").write_text(json.dumps(merged), encoding="utf-8")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("merge", help="merge the downloaded part artifacts")
    mp.add_argument("--parts", type=Path, required=True)
    mp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    problems = merge(args.parts, args.out)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print(f"Merged the mutation parts under {args.parts} into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
