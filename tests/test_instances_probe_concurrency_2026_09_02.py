# SPDX-License-Identifier: AGPL-3.0-or-later
"""snapshot() and list_machine_peers() must probe their candidates
CONCURRENTLY, not one at a time - GET /api/instances calls both in sequence,
so N sequential loopback probes at up to 0.7s each make the whole listing
take up to N*0.7s. These tests prove wall-clock time for N fake probes stays
far below the sequential total, and that concurrency does not disturb the
existing per-entry exception isolation or sort order.
"""

from __future__ import annotations

import time


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


def _fake_endpoints(count, base_port):
    return [("127.0.0.1", base_port + i) for i in range(count)]


class TestListMachinePeersProbesConcurrently:
    """Detection probes the candidate ports CONCURRENTLY, so N slow peers do not
    cost N sequential timeouts. The seams faked are the three network calls
    (/whoami, /v1/instances/status, and the candidate-port list); the threading
    around them is what is under test."""

    def _patch(self, monkeypatch, endpoints, whoami):
        monkeypatch.setattr(gpu_registry, "candidate_endpoints", lambda: endpoints)
        monkeypatch.setattr(gpu_registry, "fetch_any_whoami", whoami)
        monkeypatch.setattr(
            gpu_registry, "fetch_status",
            lambda scheme, port, timeout, dial="127.0.0.1": {
                "instance_id": f"peer{port:013d}", "pid": 1, "model": None})

    def test_wall_clock_is_far_below_sequential(self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        endpoints = _fake_endpoints(_N, 9200)

        def slow_whoami(scheme, port, timeout, bind_host=None):
            time.sleep(_SLEEP)
            return {"app": "localm", "instance_id": f"peer{port:013d}",
                    "root_dir": "/proj/x", "mode": "full", "version": "9.9.9"}

        self._patch(monkeypatch, endpoints, slow_whoami)
        t0 = time.monotonic()
        peers = instances.list_machine_peers(home)
        elapsed = time.monotonic() - t0

        assert {p["instance_id"] for p in peers} == {f"peer{p:013d}" for _a, p in endpoints}
        assert elapsed < _THRESHOLD, (
            f"list_machine_peers() took {elapsed:.3f}s probing {_N} peers at "
            f"{_SLEEP}s each ({_N * _SLEEP:.3f}s if sequential) - "
            "the probes are not running concurrently")

    def test_exception_isolation_and_sort_order_survive_concurrency(
            self, tmp_path, monkeypatch):
        home = tmp_path / "homeA"
        home.mkdir()
        # Listed in DESCENDING port order, so a correct sort must reverse them.
        endpoints = list(reversed(_fake_endpoints(_N, 9300)))
        boom_port = endpoints[2][1]

        def flaky_whoami(scheme, port, timeout, bind_host=None):
            if port == boom_port:
                raise RuntimeError("simulated whoami failure")
            return {"app": "localm", "instance_id": f"peer{port:013d}",
                    "root_dir": "/proj/x", "mode": "full", "version": "9.9.9"}

        self._patch(monkeypatch, endpoints, flaky_whoami)
        peers = instances.list_machine_peers(home)

        peer_ids = {p["instance_id"] for p in peers}
        assert f"peer{boom_port:013d}" not in peer_ids, (
            "a raised /whoami must skip that one endpoint, matching the "
            "pre-existing sequential continue-on-exception behavior")
        assert peer_ids == {f"peer{p:013d}" for _a, p in endpoints} - {f"peer{boom_port:013d}"}
        returned_order = [p["instance_id"] for p in peers]
        assert returned_order == sorted(returned_order)
