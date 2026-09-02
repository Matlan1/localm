# SPDX-License-Identifier: AGPL-3.0-or-later
"""snapshot() and list_machine_peers() must probe registry entries
CONCURRENTLY, not one at a time - GET /api/instances calls both in sequence,
so N sequential loopback probes at up to 0.7s each make the whole listing
take up to N*0.7s. These tests prove wall-clock time for N fake probes stays
far below the sequential total, and that concurrency does not disturb the
existing per-entry exception isolation or sort order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from localm import gpu_registry, instances

# One fake probe/fetch sleeps this long; N probes running SEQUENTIALLY would
# take N * _SLEEP. A concurrent implementation finishes in roughly one
# _SLEEP, well under this threshold even accounting for scheduling noise.
_SLEEP = 0.2
_N = 6
_THRESHOLD = (_N * _SLEEP) / 2


def _register(home, *, port, iid, started=None):
    return instances.register_instance(
        home, instance_id=iid, port=port, host="127.0.0.1",
        root_dir=f"/proj/{iid}", mode="full", token="tok-" + iid,
        scheme="http", started=started)


class TestSnapshotProbesConcurrently:
    def test_wall_clock_is_far_below_sequential(self, tmp_path):
        ids = [f"snap{i:012d}" for i in range(_N)]
        for i, iid in enumerate(ids):
            _register(tmp_path, port=9000 + i, iid=iid)

        def slow_probe(entry):
            time.sleep(_SLEEP)
            return True

        t0 = time.monotonic()
        rows = instances.snapshot(tmp_path, probe=slow_probe, reap=False)
        elapsed = time.monotonic() - t0

        assert len(rows) == _N
        assert all(r["alive"] is True for r in rows)
        assert elapsed < _THRESHOLD, (
            f"snapshot() took {elapsed:.3f}s probing {_N} entries at "
            f"{_SLEEP}s each ({_N * _SLEEP:.3f}s if sequential) - "
            "the probes are not running concurrently")

    def test_exception_isolation_and_sort_order_survive_concurrency(self, tmp_path):
        # Registered in REVERSE start-time order, so a correct sort must
        # reorder them - a test already in sorted order would not catch a
        # broken or dropped sort.
        ids = [f"iso{i:013d}" for i in range(_N)]
        for i, iid in enumerate(ids):
            _register(tmp_path, port=9100 + i, iid=iid,
                      started=f"2026-01-01T00:00:{(_N - i):02d}Z")
        boom_port = 9100 + 2   # one arbitrary entry's probe raises

        def flaky_probe(entry):
            if entry["port"] == boom_port:
                raise RuntimeError("simulated probe failure")
            return entry["port"] % 2 == 0

        rows = instances.snapshot(tmp_path, probe=flaky_probe, reap=False)

        assert len(rows) == _N, "one bad probe must not drop or crash the listing"
        by_port = {r["port"]: r for r in rows}
        assert by_port[boom_port]["alive"] is False, (
            "a probe exception must read as not-alive, matching the "
            "pre-existing sequential behavior")
        for port, row in by_port.items():
            if port != boom_port:
                assert row["alive"] is (port % 2 == 0)
        assert all("token" not in r for r in rows)
        started = [r["started"] for r in rows]
        assert started == sorted(started)


@pytest.fixture
def gpu_dir(tmp_path, monkeypatch):
    d = tmp_path / "machine-registry"
    d.mkdir()
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    return d


@pytest.fixture
def foreign_pid():
    """A REAL live process that is not this one, shared across every fake
    peer entry a test writes - one subprocess is enough since pid liveness is
    checked per entry but no entry is tied to a unique pid."""
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(120)"])
    try:
        assert proc.pid != os.getpid()
        yield proc.pid
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def _write_gpu_entry(gpu_dir, *, instance_id, port, pid,
                     host="127.0.0.1", scheme="http"):
    gpu_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "instance_id": instance_id, "pid": pid, "port": port, "host": host,
        "scheme": scheme, "model": None, "vram_estimate_bytes": None,
        "gpu_index": 0, "updated_at": "2026-09-02T00:00:00+00:00",
        "coordination_token": "irrelevant-for-this-test",
    }
    (gpu_dir / f"{instance_id}.json").write_text(json.dumps(entry),
                                                  encoding="utf-8")


class TestListMachinePeersProbesConcurrently:
    def test_wall_clock_is_far_below_sequential(self, tmp_path, gpu_dir,
                                                foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        ids = [f"peer{i:013d}" for i in range(_N)]
        for i, iid in enumerate(ids):
            _write_gpu_entry(gpu_dir, instance_id=iid, port=9200 + i,
                             pid=foreign_pid)

        def slow_fetch_whoami(scheme, port, iid, timeout, host=None):
            time.sleep(_SLEEP)
            return {"app": "localm", "instance_id": iid, "root_dir": "/proj/x",
                    "mode": "full", "version": "9.9.9"}

        with patch.object(instances, "fetch_whoami", slow_fetch_whoami):
            t0 = time.monotonic()
            peers = instances.list_machine_peers(home)
            elapsed = time.monotonic() - t0

        assert {p["instance_id"] for p in peers} == set(ids)
        assert elapsed < _THRESHOLD, (
            f"list_machine_peers() took {elapsed:.3f}s probing {_N} peers at "
            f"{_SLEEP}s each ({_N * _SLEEP:.3f}s if sequential) - "
            "the probes are not running concurrently")

    def test_exception_isolation_and_sort_order_survive_concurrency(
            self, tmp_path, gpu_dir, foreign_pid):
        home = tmp_path / "homeA"
        home.mkdir()
        # Written in DESCENDING id order, so a correct sort must reverse them.
        ids = [f"peer{n:013d}" for n in reversed(range(_N))]
        for i, iid in enumerate(ids):
            _write_gpu_entry(gpu_dir, instance_id=iid, port=9300 + i,
                             pid=foreign_pid)
        boom_id = ids[2]

        def flaky_fetch_whoami(scheme, port, iid, timeout, host=None):
            if iid == boom_id:
                raise RuntimeError("simulated whoami failure")
            return {"app": "localm", "instance_id": iid, "root_dir": "/proj/x",
                    "mode": "full", "version": "9.9.9"}

        with patch.object(instances, "fetch_whoami", flaky_fetch_whoami):
            peers = instances.list_machine_peers(home)

        peer_ids = {p["instance_id"] for p in peers}
        assert boom_id not in peer_ids, (
            "a raised fetch_whoami must skip that one entry, matching the "
            "pre-existing sequential continue-on-exception behavior")
        assert peer_ids == set(ids) - {boom_id}
        returned_order = [p["instance_id"] for p in peers]
        assert returned_order == sorted(returned_order)
