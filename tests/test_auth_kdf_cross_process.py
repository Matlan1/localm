# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two localm processes deriving the owner key's identity at the same moment
agree on it: the read-derive-write of ``auth.kdf.json`` holds a cross-process
lock, so only one salt is minted and no record is lost. Real interpreters, one
shared data dir."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import localm

_REPO = Path(localm.__file__).resolve().parents[1]

_CHILD = r"""
import pathlib, sys, time
from localm import auth
go = pathlib.Path(sys.argv[1])
ready = pathlib.Path(sys.argv[2])
ready.write_text("1")
deadline = time.monotonic() + 60
while not go.exists():
    if time.monotonic() > deadline:
        raise SystemExit("no start signal")
    time.sleep(0.002)
print(auth._owner_digest(sys.argv[3]))
"""

_KEY = "owner-key-for-the-cross-process-kdf-test-0123456789"
_PROCS = 4
_ROUNDS = 3


def _round(tmp: Path) -> tuple:
    home = tmp / "home"
    home.mkdir()
    env = dict(os.environ, LOCALM_HOME=str(home),
               PYTHONPATH=str(_REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("LOCALM_API_KEY", None)
    go = tmp / "go"
    procs, readies = [], []
    for i in range(_PROCS):
        ready = tmp / f"ready{i}"
        readies.append(ready)
        procs.append(subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(go), str(ready), _KEY],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    try:
        deadline = time.monotonic() + 120
        while not all(r.exists() for r in readies):
            assert time.monotonic() < deadline, "children did not start"
            time.sleep(0.01)
        go.write_text("go")
        outs = []
        for p in procs:
            out, err = p.communicate(timeout=120)
            assert p.returncode == 0, err
            outs.append(out.strip())
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    records = json.loads((home / "auth.kdf.json").read_text(encoding="utf-8"))["records"]
    return outs, records


def test_concurrent_processes_mint_one_owner_identity(tmp_path):
    for n in range(_ROUNDS):
        tmp = tmp_path / f"round{n}"
        tmp.mkdir()
        digests, records = _round(tmp)
        assert len(set(digests)) == 1, digests
        assert len(records) == 1, records
        assert records[0]["digest"] == digests[0]
