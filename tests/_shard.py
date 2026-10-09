# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split the selected tests into duration-balanced shards that run in parallel CI jobs.

Load it with ``python -m pytest -p tests._shard``. Options:

* ``--shard I/N``                 run shard I (1-based) of N. Every selected test is in
                                  exactly one shard.
* ``--shard-durations PATH``      per-test seconds used to balance the shards (default:
                                  ``tests/shard_durations_<platform>.json``). A missing file
                                  weighs every test the same.
* ``--shard-ids-out PATH``        write this shard's selected node ids and a digest of the
                                  whole selection as JSON.
* ``--shard-durations-out PATH``  write the measured seconds (setup + call + teardown) of
                                  every test this run executed as JSON.

The split is a greedy longest-first assignment over the selection that remains after
``-m`` / ``-k`` deselection, so it depends only on the selection and the durations file.
Each xdist worker computes it on its own; xdist refuses to run when the workers disagree.

Command line, stdlib only:

    python -m tests._shard verify DIR --count N     DIR holds the shards' --shard-ids-out files
    python -m tests._shard merge DIR... --out FILE  fold --shard-durations-out files into a
                                                    durations file
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import pytest

TESTS_DIR = Path(__file__).resolve().parent
UNKNOWN_WEIGHT = 1.0
KEEP_AT_LEAST = 0.1


def default_durations_path() -> Path:
    return TESTS_DIR / f"shard_durations_{sys.platform}.json"


def parse_shard(spec: str) -> tuple[int, int]:
    """``"2/4"`` -> ``(2, 4)``. Raises ValueError unless ``1 <= I <= N``."""
    index_text, sep, count_text = spec.partition("/")
    if not sep:
        raise ValueError(f"--shard takes I/N, got {spec!r}")
    index, count = int(index_text), int(count_text)
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"--shard needs 1 <= I <= N, got {spec!r}")
    return index, count


def load_durations(path: Optional[Path]) -> tuple[float, dict[str, float]]:
    """``(weight of a test the file does not list, {node id: seconds})``.
    A missing file gives ``(UNKNOWN_WEIGHT, {})``, which balances by test count."""
    if path is None or not path.is_file():
        return UNKNOWN_WEIGHT, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["default"]), {k: float(v) for k, v in data["tests"].items()}


def assign(node_ids: Iterable[str], default: float, durations: dict[str, float],
           count: int) -> dict[str, int]:
    """Map each node id to a shard index in ``range(count)``: heaviest first onto the
    currently lightest shard, ties broken by node id and shard index."""
    ordered = sorted(set(node_ids), key=lambda n: (-durations.get(n, default), n))
    heap = [(0.0, shard) for shard in range(count)]
    heapq.heapify(heap)
    out: dict[str, int] = {}
    for node_id in ordered:
        load, shard = heapq.heappop(heap)
        out[node_id] = shard
        heapq.heappush(heap, (load + durations.get(node_id, default), shard))
    return out


def digest(node_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(node_ids)).encode("utf-8")).hexdigest()


def verify_partition(parts: list[dict], count: int) -> list[str]:
    """Problems with ``parts`` (loaded ``--shard-ids-out`` files) as a partition of one
    selection into ``count`` shards. Empty when every selected test ran in exactly one."""
    problems: list[str] = []
    seen_indexes = sorted(p["index"] for p in parts)
    if seen_indexes != list(range(1, count + 1)):
        problems.append(f"expected shards 1..{count}, got {seen_indexes}")
    if {p["count"] for p in parts} != {count}:
        problems.append(f"shards disagree about the shard count: {sorted({p['count'] for p in parts})}")
    if len({(p["all_sha256"], p["all_total"]) for p in parts}) != 1:
        problems.append("shards collected different selections")
    owner: dict[str, int] = {}
    for part in parts:
        for node_id in part["selected"]:
            if node_id in owner:
                problems.append(f"{node_id} ran in shards {owner[node_id]} and {part['index']}")
            owner[node_id] = part["index"]
    if parts:
        expected = parts[0]
        if len(owner) != expected["all_total"] or digest(owner) != expected["all_sha256"]:
            problems.append(f"the shards ran {len(owner)} distinct tests, the selection has "
                            f"{expected['all_total']}")
    return problems


def merge_durations(measured: dict[str, float]) -> dict:
    """The durations file for ``measured``: tests under ``KEEP_AT_LEAST`` seconds are folded
    into the ``default`` weight, the rest are listed."""
    kept = {k: round(v, 2) for k, v in sorted(measured.items()) if v >= KEEP_AT_LEAST}
    small = [v for k, v in measured.items() if v < KEEP_AT_LEAST]
    default = round(sum(small) / len(small), 4) if small else UNKNOWN_WEIGHT
    return {"default": default, "tests": kept}


def pytest_addoption(parser):
    group = parser.getgroup("shard", "duration-balanced test shards")
    group.addoption("--shard", default=None, metavar="I/N", help="run shard I of N")
    group.addoption("--shard-durations", default=None, metavar="PATH")
    group.addoption("--shard-ids-out", default=None, metavar="PATH")
    group.addoption("--shard-durations-out", default=None, metavar="PATH")


def _is_worker(config) -> bool:
    return hasattr(config, "workerinput")


def _is_first_process(config) -> bool:
    return getattr(config, "workerinput", {}).get("workerid", "gw0") == "gw0"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    spec = config.getoption("--shard")
    if spec is None:
        return
    index, count = parse_shard(spec)
    durations_option = config.getoption("--shard-durations")
    path = Path(durations_option) if durations_option else default_durations_path()
    default, durations = load_durations(path)
    node_ids = [item.nodeid for item in items]
    owner = assign(node_ids, default, durations, count)
    kept = [item for item in items if owner[item.nodeid] == index - 1]
    dropped = [item for item in items if owner[item.nodeid] != index - 1]
    items[:] = kept
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    out = config.getoption("--shard-ids-out")
    if out and _is_first_process(config):
        payload = {"index": index, "count": count, "all_total": len(set(node_ids)),
                   "all_sha256": digest(node_ids), "selected": sorted(item.nodeid for item in kept)}
        Path(out).write_text(json.dumps(payload), encoding="utf-8")


_measured: dict[str, float] = {}


def pytest_runtest_logreport(report):
    _measured[report.nodeid] = _measured.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    out = session.config.getoption("--shard-durations-out")
    if out and not _is_worker(session.config):
        Path(out).write_text(json.dumps(_measured, sort_keys=True), encoding="utf-8")


def _load_dir(directory: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests._shard")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="check the shards partition the selection")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--count", type=int, required=True)
    merge = sub.add_parser("merge", help="fold measured durations into a durations file")
    merge.add_argument("directories", type=Path, nargs="+")
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        parts = _load_dir(args.directory)
        problems = verify_partition(parts, args.count)
        for problem in problems:
            print(f"::error::{problem}")
        if problems:
            return 1
        print(f"{args.count} shards ran {parts[0]['all_total']} tests, each exactly once.")
        return 0
    measured: dict[str, float] = {}
    for directory in args.directories:
        for table in _load_dir(directory):
            measured.update(table)
    args.out.write_text(json.dumps(merge_durations(measured), indent=0, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"{args.out}: {len(measured)} tests measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
