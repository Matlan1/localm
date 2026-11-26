# SPDX-License-Identifier: AGPL-3.0-or-later
"""The program a build ships to carry its update out (``update_mechanism.applier``) and
the ``localm update`` that waits for it (``localm._update_parent``), on a small install
of their own: the real gate, process record and update mechanism, a Python environment,
and a ``localm`` that serves ``/whoami`` for the check of its start."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm.update_mechanism import applier, steps_update
from tests import _update_mechanism as um
from tests.test_update_gate import HIDDEN, ID, NEW, OLD, REPO, Install, _statement

_MAIN = """\
import json, os, subprocess, sys, threading
import localm
from localm import part
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, 'VERSION')) as f:
    version = f.read()
args = sys.argv[1:]
if args[:1] == ['_update-probe']:
    if os.path.exists(os.path.join(root, 'FAILS')):
        print('it fails on purpose', flush=True)
        sys.exit(1)
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Answer(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({'version': version}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_a):
            pass
    server = HTTPServer(('127.0.0.1', 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({'port': server.server_address[1]}), flush=True)
    sys.stdin.read()
    sys.exit(0)
if args[:1] == ['_update-record']:
    sys.exit(0)
if args[:1] == ['update']:
    from localm import _update_parent, procs
    if os.environ.get(_update_parent.WORK) != '1':
        sys.exit(_update_parent.run(['localm', *args]))
    me = procs.current()
    program = os.path.join(root, '.localm-update', args[1], 'applier', 'update_mechanism')
    flags = {'creationflags': 0x200} if os.name == 'nt' else {'start_new_session': True}
    child = subprocess.Popen([sys.executable, '-I', '-S', program, 'apply', root, args[1]],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, **flags)
    child.stdin.write(json.dumps({'mode': 'command', 'starter': f'{me.pid}-{me.start}',
                                  'confirmed': [], 'home': os.environ['LOCALM_HOME'],
                                  'old': '1', 'new': '2', 'watch': None,
                                  'relaunch': None}) + '\\n')
    child.stdin.close()
    sys.exit(_update_parent.HANDED_OFF if child.stdout.readline().strip() == 'ready' else 1)
said = {'localm': localm.__version__, 'part': part.V, 'VERSION': version, 'argv': args}
print('STARTED ' + json.dumps(said), flush=True)
if args[:1] == ['stay']:
    sys.stdin.read()
"""
_STEPS = {"format": steps_update.FORMAT, "sync": None, "torch": [], "uv": None, "parts": [],
          "kept": [], "notes": []}


def _with_main(files: dict) -> dict:
    return {**files, "localm/__main__.py": _MAIN,
            "localm/_update_parent.py": (REPO / "localm" / "_update_parent.py").read_text(
                encoding="utf-8")}


class Update(Install):
    """A small localm 1 with a Python environment and an update to 2 prepared; with
    *fails*, the new version does not start."""

    def __init__(self, root: Path, *, fails: bool = False) -> None:
        super().__init__(root)
        old = _with_main(OLD)
        new = _with_main({**NEW, **({"FAILS": "x"} if fails else {})})
        um.write(root, {**old, "update-package.json": _statement(old, "1")})
        work = root / ".localm-update" / ID
        um.write(work / "new", {**new, "update-package.json": _statement(new, "2")})
        (work / steps_update.STEPS_FILE).write_text(json.dumps(_STEPS), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")],
                       check=True, **HIDDEN)
        self.old = self.tree()

    def tree(self) -> dict:
        """The files of the install, the data folder excepted."""
        return um.tree(self.root, skip=(".localm-update", ".localm-run", "home"))

    def is_old(self) -> bool:
        return self.tree() == self.old and not self.record.exists()

    def version(self) -> str:
        return (self.root / "VERSION").read_text(encoding="utf-8")

    def apply(self, *, confirmed=(), starter: str = "0-0") -> subprocess.Popen:
        """Start the applier with its hand-off and return it once it holds the update."""
        program = self.root / ".localm-update" / ID / "applier" / "update_mechanism"
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(program), "apply", str(self.root), ID],
            cwd=str(self.root), env=self.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, **HIDDEN)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({
            "mode": applier.COMMAND, "starter": starter, "confirmed": list(confirmed),
            "home": str(self.root / "home"), "old": "1", "new": "2", "watch": None,
            "relaunch": None}) + "\n")
        proc.stdin.close()
        assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
        return proc

    def waiter(self) -> subprocess.Popen:
        """A start of localm that goes on with the update for the user."""
        return self.popen("-m", "localm", LOCALM_UPDATE_PRIMARY=ID)

    def result(self) -> dict:
        return json.loads((self.root / "home" / "updates" / applier.RESULT).read_text())


def _end(*procs: subprocess.Popen) -> list:
    out = []
    for proc in procs:
        try:
            out.append(proc.communicate(timeout=180))
        except subprocess.TimeoutExpired:
            proc.kill()
            out.append(proc.communicate())
    return out


@pytest.fixture
def update(tmp_path):
    return Update(tmp_path / "inst")


def test_an_update_runs_through_and_the_new_version_starts(update):
    proc = update.apply()
    waiting = update.waiter()
    (_, err), (out, _) = _end(proc, waiting)
    assert proc.returncode == 0, err
    assert update.version() == "2"
    assert not update.record.exists()
    assert not [n for n in os.listdir(update.root / ".localm-update") if len(n) == 32]
    started = next(json.loads(ln[8:]) for ln in out.splitlines() if ln.startswith("STARTED "))
    assert {started[k] for k in ("localm", "part", "VERSION")} == {"2"}
    assert update.result()["state"] == "new"


def test_a_new_version_that_does_not_start_is_undone_and_says_why(tmp_path):
    update = Update(tmp_path / "inst", fails=True)
    proc = update.apply()
    waiting = update.waiter()
    (_, err), (out, _) = _end(proc, waiting)
    assert proc.returncode == 1
    assert update.is_old(), err
    started = next(json.loads(ln[8:]) for ln in out.splitlines() if ln.startswith("STARTED "))
    assert {started[k] for k in ("localm", "part", "VERSION")} == {"1"}
    result = update.result()
    assert result["state"] == "old" and result["version"] == "1"
    assert "fails on purpose" in result["note"] and "was undone" in result["note"]
    assert "fails on purpose" in err


@pytest.mark.parametrize("agreed", [False, True])
def test_a_process_nobody_agreed_to_end_stops_the_update_before_anything_changes(update, agreed):
    other = subprocess.Popen([sys.executable, "-m", "localm", "stay"], cwd=str(update.root),
                             env=update.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert other.stdout is not None and other.stdin is not None
    assert other.stdout.readline().startswith("STARTED ")
    from localm import procs
    entry = next(e for e in procs.list(str(update.root))
                 if e.recorded and (e.pid == other.pid or (e.parent and e.parent[0] == other.pid)))
    proc = update.apply(confirmed=[f"{entry.pid}-{entry.start}"] if agreed else [])
    waiting = update.waiter()
    (_, err), _ = _end(proc, waiting)
    try:
        if agreed:
            assert proc.returncode == 0, err
            assert other.wait(timeout=30) is not None
            assert update.version() == "2"
        else:
            assert proc.returncode == 1
            assert procs.alive(entry), "a process nobody agreed to end was ended"
            assert update.is_old()
            assert f"pid {entry.pid}" in update.result()["note"]
    finally:
        other.stdin.close()
        _end(other)


@pytest.mark.parametrize("fails", [False, True])
def test_localm_update_exits_0_only_once_the_new_version_is_in_place(tmp_path, fails):
    update = Update(tmp_path / "inst", fails=fails)
    proc = update.popen("-m", "localm", "update", ID)
    out, err = proc.communicate(timeout=300)
    old, version, record = update.is_old(), update.version(), update.record.exists()
    if fails:
        assert proc.returncode == 1, err
        assert old
    else:
        assert proc.returncode == 0, err
        assert version == "2" and not record
    assert not list((update.root / ".localm-update").glob("result-*.json"))


def test_the_hand_off_must_be_complete():
    import io
    with pytest.raises(ValueError):
        applier.read_handoff(io.StringIO(json.dumps({"mode": "command"}) + "\n"))
