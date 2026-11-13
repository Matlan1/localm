# SPDX-License-Identifier: AGPL-3.0-or-later
"""A private store file written while another handle holds it open is still
written: every ``atomic_write_private`` caller serialises its writers and rides
out the Windows sharing violation a reader's open handle causes."""

from __future__ import annotations

import ast
import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_WRITERS = {"atomic_write_private", "_atomic_write_private"}
_PASS_THROUGH = {("localm/auth.py", "_atomic_write_private"),
                 ("localm/auth.py", "_save_owner_kdf")}


def _writer_calls():
    for path in sorted((REPO_ROOT / "localm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                name = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else None)
                if name in _WRITERS:
                    kw = {k.arg: k.value for k in n.keywords}
                    yield rel, fn.name, n.lineno, kw.get("retrying")


def test_every_private_store_write_rides_out_a_reader():
    calls = list(_writer_calls())
    assert len(calls) >= 9, calls
    bad = []
    for rel, fname, line, value in calls:
        if isinstance(value, ast.Constant) and value.value is True:
            continue
        if isinstance(value, ast.Name) and (rel, fname) in _PASS_THROUGH:
            continue
        bad.append(f"{rel}:{line} {fname}")
    assert not bad, ("a private store write uses the bare replace, so a reader "
                     "holding the file loses the write on Windows: " + ", ".join(bad))


def _held_for(path: Path, seconds: float) -> threading.Thread:
    """Open *path* for reading (no delete sharing on Windows) and close it after
    *seconds*, on a thread. Returns once the handle is open."""
    opened = threading.Event()

    def hold():
        with open(path, "rb"):
            opened.set()
            threading.Event().wait(seconds)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert opened.wait(5)
    return t


windows_only = pytest.mark.skipif(sys.platform != "win32",
                                  reason="the sharing violation exists on Windows only")


@windows_only
def test_activity_record_is_written_while_a_reader_holds_it(tmp_path):
    from localm.jobmanager import _ActivityStore
    store = _ActivityStore(root=tmp_path, pid=4242, run="r1")
    assert store.write([{"op": "old"}])
    holder = _held_for(store.path, 0.15)
    assert store.write([{"op": "new"}]) is True
    holder.join(5)
    assert json.loads(store.path.read_text(encoding="utf-8"))["operations"] == [{"op": "new"}]


@windows_only
def test_instance_entry_mode_is_written_while_a_reader_holds_it(tmp_path):
    from localm import instances
    path = instances.register_instance(tmp_path, instance_id="abc123", port=1,
                                       host="127.0.0.1", root_dir=str(tmp_path),
                                       mode="api", token="t")
    holder = _held_for(path, 0.15)
    assert instances.set_mode(tmp_path, "abc123", "full") is True
    holder.join(5)
    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "full"


@windows_only
def test_session_store_is_written_while_a_reader_holds_it():
    from localm import sessions
    sessions._save([{"id": "one"}])
    path = sessions.sessions_file()
    holder = _held_for(path, 0.15)
    sessions._save([{"id": "two"}])
    holder.join(5)
    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": "two"}]


@windows_only
def test_filed_issue_record_is_written_while_a_reader_holds_it(monkeypatch):
    from localm import issue_tracker
    monkeypatch.setattr(issue_tracker, "_privacy_mode", lambda: False)
    assert issue_tracker.record_filed(11, "first")
    path = issue_tracker._record_path()
    holder = _held_for(path, 0.15)
    assert issue_tracker.record_filed(12, "second") is True
    holder.join(5)
    numbers = [e["number"] for e in json.loads(path.read_text(encoding="utf-8"))["issues"]]
    assert numbers == [11, 12]


@windows_only
def test_owner_key_record_is_written_while_a_reader_holds_it():
    from localm import auth
    auth._save_owner_kdf([{"digest": "a"}], retrying=True)
    path = auth.owner_kdf_file()
    holder = _held_for(path, 0.15)
    auth._save_owner_kdf([{"digest": "b"}], retrying=True)
    holder.join(5)
    assert json.loads(path.read_text(encoding="utf-8"))["records"] == [{"digest": "b"}]
