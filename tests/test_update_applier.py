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
import time
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
if args[:1] == ['_update-probe']:
    if os.path.exists(os.path.join(root, 'FAILS')):
        print('it fails on purpose', flush=True)
        sys.exit(1)
    server = HTTPServer(('127.0.0.1', 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({'port': server.server_address[1]}), flush=True)
    sys.stdin.read()
    sys.exit(0)
if args[:1] == ['_update-record']:
    __import__('time').sleep(3)
    sys.exit(0)
if args[:1] == ['update']:
    from localm import _update_parent, updater_handoff
    if os.environ.get(_update_parent.WORK) != '1':
        sys.exit(_update_parent.run(['localm', *args]))
    from pathlib import Path
    prepared = updater_handoff.Prepared(Path(root), args[1], '1', '2', None, True, None, False,
                                        None)
    updater_handoff.hand_off(prepared, mode='command', confirmed=args[2:],
                             say=lambda t: print(t, file=sys.stderr))
    sys.exit(_update_parent.HANDED_OFF)
said = {'localm': localm.__version__, 'part': part.V, 'VERSION': version, 'argv': args,
        'during': os.path.exists(os.path.join(root, '.localm-update', 'job.json'))}
print('STARTED ' + json.dumps(said), flush=True)
if args[:1] == ['serve'] and os.path.exists(os.path.join(root, 'SERVEFAILS')):
    sys.exit(1)
if args[:1] == ['serve']:
    server = HTTPServer(('127.0.0.1', int(args[1])), Answer)
    if args[2:]:
        import ssl
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args[2], args[3])
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sys.stdin.read()
if args[:1] == ['stay']:
    sys.stdin.read()
"""
_STEPS = {"format": steps_update.FORMAT, "sync": None, "torch": [], "uv": None, "parts": [],
          "kept": [], "notes": []}


#: Stand-ins for the parts of localm that ``updater_handoff`` imports and this small
#: install has not.
_STUBS = {
    "localm/config.py": "import os\nfrom pathlib import Path\n"
                        "HOME_DIR = Path(os.environ['LOCALM_HOME'])\n",
    "localm/updater.py": "",
    "localm/bugreport.py": "class LocalmError(Exception):\n"
                           "    def __init__(self, message, reason=None):\n"
                           "        super().__init__(f'{message}: {reason}')\n"
                           "        self.reason = reason\n",
}


def _with_main(files: dict) -> dict:
    return {**files, **_STUBS, "localm/__main__.py": _MAIN,
            **{f"localm/{n}.py": (REPO / "localm" / f"{n}.py").read_text(encoding="utf-8")
               for n in ("_update_parent", "updater_handoff")}}


class Update(Install):
    """A small localm 1 with a Python environment and an update to 2 prepared; with
    *fails*, the new version does not start."""

    def __init__(self, root: Path, *, fails: bool = False, serve_fails: bool = False) -> None:
        super().__init__(root)
        old = _with_main(OLD)
        new = _with_main({**NEW, **({"FAILS": "x"} if fails else {}),
                          **({"SERVEFAILS": "x"} if serve_fails else {})})
        um.write(root, {**old, "update-package.json": _statement(old, "1")})
        work = root / ".localm-update" / ID
        um.write(work / "new", {**new, "update-package.json": _statement(new, "2")})
        (work / steps_update.STEPS_FILE).write_text(json.dumps(_STEPS), encoding="utf-8")
        later = time.time() + 10
        for path in (work / "new").rglob("*"):
            os.utime(path, (later, later))   # extracted after the install was written
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

    def apply(self, *, confirmed=(), starter: str = "0-0", restarting=(), ancestors=(),
              mode: str = applier.COMMAND, watch=None, relaunch=None,
              **env) -> subprocess.Popen:
        """Start the applier with its hand-off and return it once it holds the update."""
        program = self.root / ".localm-update" / ID / "applier" / "update_mechanism"
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(program), "apply", str(self.root), ID,
             *(relaunch or ())],
            cwd=str(self.root), env=self.env(**env), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **HIDDEN)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({
            "mode": mode, "starter": starter, "confirmed": list(confirmed),
            "home": str(self.root / "home"), "old": "1", "new": "2", "watch": watch,
            "restarting": list(restarting),
            "ancestors": list(ancestors)}) + "\n")
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


@pytest.mark.parametrize("waits", [False, True])
def test_a_result_nobody_waits_for_any_more_is_deleted_by_the_next_cleanup(update, waits):
    from localm import procs
    from localm.update_mechanism import job, steps
    proc = update.apply()
    _end(proc, update.waiter())
    left = update.root / ".localm-update" / job.result_name(ID)
    assert json.loads(left.read_text(encoding="utf-8"))[job.RESULT_FOR]
    other = subprocess.Popen([sys.executable, "-m", "localm", "stay"], cwd=str(update.root),
                             env=update.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert other.stdout is not None and other.stdin is not None
    try:
        assert other.stdout.readline().startswith("STARTED ")
        if waits:
            entry = next(e for e in procs.list(str(update.root)) if e.recorded and (
                e.pid == other.pid or (e.parent and e.parent[0] == other.pid)))
            said = json.loads(left.read_text(encoding="utf-8"))
            left.write_text(json.dumps({**said, job.RESULT_FOR: f"{entry.pid}-{entry.start}"}),
                            encoding="utf-8")
        steps.sweep(update.root)
        assert left.exists() is waits
    finally:
        other.stdin.close()
        _end(other)


def _stay(update, *more):
    """A start of localm in *update* that runs until its input ends, and its entry."""
    from localm import procs
    proc = subprocess.Popen([sys.executable, "-m", "localm", *(more or ("stay",))],
                            cwd=str(update.root), env=update.env(), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert proc.stdout is not None and proc.stdin is not None
    if more:
        return proc, None
    assert proc.stdout.readline().startswith("STARTED ")
    entry = next(e for e in procs.list(str(update.root)) if e.recorded and (
        e.pid == proc.pid or (e.parent and e.parent[0] == proc.pid)))
    return proc, entry


def _waits(update, *, count: int = 1, timeout: float = 60) -> None:
    """Wait until *count* starts of localm wait at the gate of *update*."""
    deadline = time.monotonic() + timeout
    folder = update.root / ".localm-update"
    while len(list(folder.glob("waiting-*"))) < count:
        assert time.monotonic() < deadline, "nothing waits at the gate"
        time.sleep(0.1)


def _started(out: str) -> dict:
    return next(json.loads(ln[8:]) for ln in out.splitlines() if ln.startswith("STARTED "))


def test_a_server_asked_to_restart_is_waited_for_and_runs_the_new_version_after_it(update):
    """The other server (here a start that the test restarts by hand, 3 s after the
    update began) is neither ended nor counted as a process nobody agreed to end: its
    successor waits at the gate and runs the new version."""
    server, entry = _stay(update)
    key = f"{entry.pid}-{entry.start}"
    proc = update.apply(confirmed=[key], restarting=[[key, entry.purpose]])
    waiting = update.waiter()
    try:
        time.sleep(3)
        assert server.poll() is None, "a server asked to restart was ended at once"
        assert json.loads(update.record.read_text(encoding="utf-8"))["phase"] == "ending"
        assert update.version() == "1"
        again, _ = _stay(update, "again")
        _waits(update, count=2)
    finally:
        server.stdin.close()
        _end(server)
    (_, err), _, (out, _) = _end(proc, waiting, again)
    assert proc.returncode == 0, err
    assert update.version() == "2" and "did not restart" not in err
    assert {_started(out)[k] for k in ("localm", "part", "VERSION")} == {"2"}, (out, err)


def test_a_start_during_the_update_waits_and_runs_only_the_new_version(update):
    """MATRIX-COMBO-162: nothing of localm runs between the hand-off and the end of the
    update, so a ``localm update --repair`` (any start) started then neither races the
    update's steps nor sees the old process answer with the new version: it waits at the
    gate, is not ended, and then runs the new version."""
    proc = update.apply()
    repair, _ = _stay(update, "repair")
    _waits(update)
    waiting = update.waiter()
    (_, err), _, (out, _) = _end(proc, waiting, repair)
    assert proc.returncode == 0, err
    assert update.version() == "2"
    said = _started(out)
    assert said["argv"] == ["repair"] and said["during"] is False
    assert {said[k] for k in ("localm", "part", "VERSION")} == {"2"}


def test_the_other_servers_of_this_install_are_asked_to_restart(tmp_path, monkeypatch):
    """``servers_of`` finds a server of this data folder's registry by its process, and
    ``restart_servers`` sends it ``POST /v1/server/restart``; a process the registry does
    not list, or that it lists for another process, is not one."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from localm import config, instances, updater_handoff
    asked: list = []

    class Answer(BaseHTTPRequestHandler):
        def do_POST(self):
            asked.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_a):
            pass

    http = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(config, "HOME_DIR", tmp_path)
        path = instances.register_instance(
            tmp_path, instance_id="a" * 32, port=http.server_address[1], host="127.0.0.1",
            root_dir=str(tmp_path), mode="full", token="t" * 32)
        processes = [{"key": "me", "pid": os.getpid(), "purpose": "server"},
                     {"key": "chat", "pid": os.getpid() + 1, "purpose": "chat"}]
        servers = updater_handoff.servers_of(processes)
        assert list(servers) == ["me"]
        assert servers["me"]["url"].endswith(f":{http.server_address[1]}/v1")
        prepared = updater_handoff.Prepared(tmp_path, "b" * 32, "1", "2", None, True, None,
                                            False, None, processes=processes,
                                            servers=servers)
        assert prepared.restarting() == [["me", "server"]]
        assert [p["key"] for p in prepared.ended()] == ["chat"]
        updater_handoff.restart_servers(prepared, say=lambda _t: None)
        assert asked == ["/v1/server/restart"]
        entry = json.loads(Path(path).read_text(encoding="utf-8"))
        Path(path).write_text(json.dumps({**entry, "start_identity": {"x": 1}}),
                              encoding="utf-8")
        assert updater_handoff.servers_of(processes) == {}
    finally:
        http.shutdown()


@pytest.mark.parametrize("backup_is_older", [True, False])
def test_going_back_reinstalls_what_the_last_update_replaced_not_an_older_backup(
        updates_dir, monkeypatch, backup_is_older):
    """A backup an earlier update of the old updater left (of 0.3.3) does not win over
    the version the later update replaced; a backup made after that update does."""
    from localm import _apply_update as au
    from localm import updater
    monkeypatch.setattr(updater._version, "read_version", lambda: "2")
    monkeypatch.setattr(updater, "_backup_is_restorable", lambda _d: True)
    monkeypatch.setattr(au, "backup_problem", lambda *_a: None)
    backup = updates_dir / "backup"
    backup.mkdir()
    (backup / "VERSION").write_text("0.3.3", encoding="utf-8")
    (backup / au.BACKUP_RECORD).write_text("{}", encoding="utf-8")
    made = time.time() - (100 if backup_is_older else -100)
    os.utime(backup / au.BACKUP_RECORD, (made, made))
    _ended(updates_dir, "new")
    info = updater.rollback_info()
    assert info["reinstall"] is backup_is_older
    assert info["version"] == ("1" if backup_is_older else "0.3.3")


def _free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_a_server_restarts_into_the_update_and_answers_with_the_new_version(
        tmp_path, in_place, scheme):
    """Server mode: the server that started the update waits at the gate after its
    restart, is let through at STARTING and must answer ``/whoami`` with the new version.
    *in_place*: it restarted as the same process (POSIX execv keeps its pid and start
    time, so it waits under the starter's own key). Over https it uses localm's own
    certificate; a proxy in the environment is not used for this own loopback address."""
    from localm import tls
    update = Update(tmp_path / "inst")
    port = _free_port()
    serve = ["serve", str(port)]
    if scheme == "https":
        serve += list(tls.ensure_cert(tmp_path / "tls"))
    # In place: a recorded server of the old version enters the gate again in the same
    # process, as the image a POSIX execv starts does, then runs the new version.
    code = "\n".join([
        "import os, runpy, sys",
        *(["import localm", "from localm import procs", "me = procs.current()",
           "print(f'{me.pid}-{me.start}', flush=True)"] if in_place else []),
        "sys.stdin.readline()",
        *([f"os.environ['LOCALM_UPDATE_PRIMARY'] = {ID!r}", "localm._gate(me)",
           "for name in [m for m in sys.modules if m.split('.')[0] == 'localm']:",
           "    del sys.modules[name]"] if in_place else []),
        f"sys.argv[1:] = {serve!r}",
        "runpy.run_module('localm', run_name='__main__', alter_sys=True)"])
    server = subprocess.Popen([sys.executable, "-c", code], cwd=str(update.root),
                              env=update.env(**({} if in_place else {
                                  "LOCALM_UPDATE_PRIMARY": ID})),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert server.stdin is not None and server.stdout is not None
    starter = server.stdout.readline().strip() if in_place else "0-0"
    proxy = "http://192.0.2.1:9"
    proc = update.apply(starter=starter,
                        mode=applier.SERVER,
                        watch={"host": "127.0.0.1", "port": port, "scheme": scheme},
                        HTTP_PROXY=proxy, HTTPS_PROXY=proxy, NO_PROXY="")
    server.stdin.write("go\n")
    server.stdin.flush()
    try:
        _, err = proc.communicate(timeout=300)
    finally:
        server.stdin.close()
        out, _ = _end(server)[0]
    assert proc.returncode == 0, err
    assert update.version() == "2" and update.result()["state"] == "new"
    assert {_started(out)[k] for k in ("localm", "part", "VERSION")} == {"2"}


def test_a_server_that_is_gone_before_it_is_started_leaves_the_new_version(update):
    """The server that waited for the update ended before STARTING: the new version,
    which passed its own check, stays, and the old one is not started again."""
    marker = update.root / "home" / "relaunched"
    port = _free_port()
    proc = update.apply(mode=applier.SERVER,
                        watch={"host": "127.0.0.1", "port": port, "scheme": "http"},
                        relaunch=[sys.executable, "-c", f"open({str(marker)!r}, 'w')"])
    server = update.popen("-m", "localm", "serve", str(port), LOCALM_UPDATE_PRIMARY=ID)
    _waits(update)
    server.kill()
    server.communicate()
    _, err = proc.communicate(timeout=300)
    time.sleep(1)
    assert proc.returncode == 0, err
    assert update.version() == "2" and not marker.exists()
    assert "the next start of localm runs it" in err


def test_a_process_above_the_starter_goes_on_and_does_not_stop_the_update(update):
    """A launcher above the process that started the update (it shows the window of the
    app) is neither ended nor a process nobody agreed to end."""
    launcher, entry = _stay(update)
    proc = update.apply(ancestors=[f"{entry.pid}-{entry.start}"])
    waiting = update.waiter()
    try:
        (_, err), _ = _end(proc, waiting)
        assert proc.returncode == 0, err
        assert update.version() == "2" and launcher.poll() is None
    finally:
        launcher.stdin.close()
        _end(launcher)


def test_only_an_owner_confirms_and_the_second_press_uses_what_the_first_prepared(
        monkeypatch):
    """``Update now`` while a chat runs: a press without an owner is refused and arms
    nothing; the owner's first press names the processes and keeps the prepared update;
    the owner's second press hands that same update over, without preparing it again."""
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    import localm.inference.http_server as hs
    from localm import updater, updater_handoff
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    owner, made, gone, handed = [False], [], [], []
    chat = [{"key": "9-9", "pid": 9, "purpose": "chat", "restarts": False}]

    def prepare(*_a, **_k):
        p = updater_handoff.Prepared(Path("."), "0" * 32, "1", "2", None, True, None, False,
                                     None, processes=list(chat))
        p.discard = lambda: gone.append(p)
        made.append(p)
        return p

    def look_around(p):
        p.processes = list(chat)
        return p

    monkeypatch.setattr(hs, "_request_restart", lambda **k: handed.append(k.get("update")))
    monkeypatch.setattr(hs, "caller_is_operator", lambda _r: owner[0])
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "1", "latest": "v2", "newer": True, "notes": "", "asset": {"id": 3}})
    monkeypatch.setattr(updater_handoff, "prepare", prepare)
    monkeypatch.setattr(updater_handoff, "look_around", look_around)
    engine = MagicMock()
    engine.display_name, engine.loaded = "m", True
    app = hs.create_app(engine)
    replies = []
    with TestClient(app) as client:
        for who in (False, True, True, True):
            owner[0] = who
            replies.append(client.post("/api/update/apply", headers={
                "Authorization": f"Bearer {app.state.shell_token}"}))
    assert replies[0].status_code == 403 and gone == [made[0]]
    assert replies[1].json()["needs_confirmation"] == "processes"
    assert replies[2].json()["applied"] is True
    assert len(made) == 2 and handed == [made[1]] and gone == [made[0]]
    # One update at a time: a press after the hand-off prepares nothing.
    assert replies[3].status_code == 409 and len(made) == 2


def test_a_scoped_caller_sees_the_reply_of_an_update_without_host_paths(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    import localm.inference.http_server as hs
    from localm import config, updater, updater_handoff
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    secret = str(config.HOME_DIR / "models" / "mine.txt")

    def prepare(*_a, **_k):
        return updater_handoff.Prepared(Path("."), "0" * 32, "1", "2", None, True, None, False,
                                        None, notes=[f"kept {secret}"], edited=[secret])

    monkeypatch.setattr(hs, "_request_restart", lambda **_k: None)
    monkeypatch.setattr(hs, "caller_is_operator", lambda _r: False)
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "1", "latest": "v2", "newer": True, "notes": "", "asset": {"id": 3}})
    monkeypatch.setattr(updater_handoff, "prepare", prepare)
    monkeypatch.setattr(updater_handoff, "look_around", lambda p: p)
    engine = MagicMock()
    engine.display_name, engine.loaded = "m", True
    app = hs.create_app(engine)
    with TestClient(app) as client:
        reply = client.post("/api/update/apply", headers={
            "Authorization": f"Bearer {app.state.shell_token}"}).json()
    assert reply["applied"] is True
    assert str(config.HOME_DIR) not in json.dumps(reply) and "mine.txt" in json.dumps(reply)


def test_the_hand_off_must_be_complete():
    import io
    with pytest.raises(ValueError):
        applier.read_handoff(io.StringIO(json.dumps({"mode": "command"}) + "\n"))


# ------------------------------------------------- the result, and going back

@pytest.fixture
def updates_dir(tmp_path, monkeypatch):
    from localm import updater
    folder = tmp_path / "updates"
    folder.mkdir()
    monkeypatch.setattr(updater, "_updates_dir", lambda create=True: folder)
    monkeypatch.setattr(updater._version, "read_version", lambda: "1")
    return folder


def _ended(folder: Path, state: str, **more) -> None:
    (folder / applier.RESULT).write_text(json.dumps({
        "state": state, "from": "1", "to": "2", "version": "1" if state == "old" else "2",
        "note": f"ended {state}", "notes": ["a note"], "lines": [], "at": time.time(),
        **more}), encoding="utf-8")


def test_a_rolled_back_update_says_why_in_settings(updates_dir):
    from localm import updater_parts
    _ended(updates_dir, "old", note="The update to 2 failed and was undone: it fails.")
    steps = updater_parts._last_steps("1")
    assert [s["part"] for s in steps] == ["update"]
    assert steps[0]["state"] == "old" and "it fails" in steps[0]["note"]
    assert "a note" in steps[0]["note"]
    assert updater_parts._last_steps("2") == []


@pytest.mark.parametrize("state, running, back", [
    ("new", "2", "1"), ("old", "1", None), ("new", "3", None)])
def test_going_back_after_an_update_installs_the_version_it_replaced(
        updates_dir, monkeypatch, state, running, back):
    from localm import updater
    monkeypatch.setattr(updater._version, "read_version", lambda: running)
    _ended(updates_dir, state)
    info = updater.rollback_info()
    assert info["reinstall"] is (back is not None)
    assert info["available"] is (back is not None)
    assert info["version"] == back and info["backup"] is None


# ------------------------------------- the applier run in this process, for its asks

def _in_process(update, monkeypatch, *, ask=None, **handoff):
    """Carry the update of *update* out with an :class:`applier.Applier` of this process
    (so that its ``ask`` and its parts can be replaced), while a start of localm waits
    for it. Returns ``(result, the waiter's output)``."""
    from localm.update_mechanism import job
    monkeypatch.setenv("LOCALM_HOME", str(update.root / "home"))
    h = {"mode": applier.COMMAND, "starter": "0-0", "confirmed": [],
         "home": str(update.root / "home"), "old": "1", "new": "2", "watch": None, **handoff}
    waiting = None
    try:
        with job.Lock(update.root) as lock:
            job.start(lock, ID, old="1", new="2")
            waiting = update.waiter()
            run = applier.Applier(update.root, ID, h)
            run.ask = ask
            result = run.run(lock)
    finally:
        out = _end(waiting)[0][0] if waiting is not None else ""
    return result, out


@pytest.mark.parametrize("again", [True, False])
def test_a_process_in_the_way_is_named_with_try_again_or_cancel(update, monkeypatch, again):
    """A process nobody agreed to end: the user at the terminal is asked; once it is
    closed, Try again goes on with the update, Cancel leaves the old version as it is."""
    other, entry = _stay(update)
    asked: list = []

    def ask(text):
        asked.append(text)
        if again:
            other.stdin.close()
            other.wait(timeout=30)
        return again

    try:
        result, _ = _in_process(update, monkeypatch, ask=ask)
    finally:
        if not other.stdin.closed:
            other.stdin.close()
        _end(other)
    assert len(asked) == 1 and f"pid {entry.pid}" in asked[0]
    if again:
        assert result["state"] == "new" and update.version() == "2"
    else:
        assert result["state"] == "old" and update.is_old()


def test_try_again_reads_the_answer_and_counts_the_end_as_cancel():
    import io
    said: list = []
    assert applier.try_again("held", io.StringIO("x\nT\n"), said.append) is True
    assert applier.try_again("held", io.StringIO("c\n"), said.append) is False
    assert applier.try_again("held", io.StringIO(""), said.append) is False
    assert said[0] == "held" and "[T] Try again" in said[1]


@pytest.mark.skipif(os.name != "nt", reason="an open file holds it only on Windows")
@pytest.mark.parametrize("again", [True, False])
def test_a_held_file_is_named_with_try_again_or_cancel(update, monkeypatch, again):
    """A file of the install another program holds while the files are replaced: after
    the wait the user is asked; Try again once it is free installs the new version,
    Cancel puts the old one back."""
    from localm.update_mechanism import disk
    monkeypatch.setattr(disk, "HELD_WAIT_S", 1.0)
    holder = open(update.root / "localm" / "part.py", encoding="utf-8")
    asked: list = []

    def ask(text):
        asked.append(text)
        if again:
            holder.close()
        return again

    try:
        result, _ = _in_process(update, monkeypatch, ask=ask)
    finally:
        holder.close()
    assert asked and "part.py" in asked[0]
    if again:
        assert result["state"] == "new" and update.version() == "2"
    else:
        assert result["state"] == "old" and update.is_old()


def test_only_a_terminal_command_mode_hand_off_asks(update, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(update.root / "home"))
    monkeypatch.setattr(applier, "terminal_ask", lambda: "asks")
    h = {"mode": applier.COMMAND, "starter": "0-0", "confirmed": [], "home": "", "old": "1",
         "new": "2", "watch": None}
    assert applier.Applier(update.root, ID, {**h, "ask": True}).ask == "asks"
    assert applier.Applier(update.root, ID, h).ask is None
    assert applier.Applier(update.root, ID, {**h, "ask": True, "mode": applier.SERVER}).ask is None


def test_a_server_that_does_not_restart_within_its_time_is_ended(update, monkeypatch):
    """Another server asked to restart for the update that never waits at the gate:
    after :data:`applier.RESTART_S` it is named and ended, and the update goes on."""
    from localm import procs
    monkeypatch.setattr(applier, "RESTART_S", 1.0)
    # Started through a process that ends at once: the applier of this test process
    # spares what this process started, the applier of a real update did not start it.
    mid = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys\n"
         "subprocess.Popen([sys.executable, '-m', 'localm', 'stay'], "
         "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"],
        cwd=str(update.root), env=update.env(), stdin=subprocess.PIPE, **HIDDEN)
    try:
        mid.wait(timeout=60)
        deadline = time.monotonic() + 60
        entry = None
        while entry is None:
            assert time.monotonic() < deadline, "the server did not start"
            entry = next((e for e in procs.list(str(update.root)) if e.recorded), None)
            time.sleep(0.1)
        key = f"{entry.pid}-{entry.start}"
        said: list = []
        monkeypatch.setattr(applier.restoring, "say", said.append)
        result, _ = _in_process(update, monkeypatch, confirmed=[key],
                                restarting=[[key, entry.purpose]])
        ended = not procs.alive(entry)
    finally:
        assert mid.stdin is not None
        mid.stdin.close()
    assert ended, "the server that did not restart was not ended"
    assert result["state"] == "new" and update.version() == "2"
    assert any("did not restart within 1 s" in t and f"pid {entry.pid}" in t for t in said)


def test_a_record_that_raises_leaves_the_new_version_with_a_note(update, monkeypatch):
    def broken(*_a, **_k):
        raise OSError("the disk is gone")

    monkeypatch.setattr(applier.steps_update, "record", broken)
    result, _ = _in_process(update, monkeypatch)
    assert result["state"] == "new" and update.version() == "2"
    assert any("could not be written: the disk is gone" in n for n in result["notes"])


def test_a_way_back_that_finds_no_update_is_never_called_undone(update, monkeypatch):
    from localm.update_mechanism import job
    monkeypatch.setattr(applier.steps_update, "put_back",
                        lambda *_a, **_k: job.Result("none", []))
    monkeypatch.setattr(applier.steps_probe, "probe", lambda *_a, **_k: "it fails")
    result, _ = _in_process(update, monkeypatch)
    assert result["state"] == "failed"
    assert "was undone" not in result["note"] and "'none'" in result["note"]


# ------------------------------------------------------ the hand-off, for real

def test_the_hand_off_starts_the_applier_and_localm_update_waits_for_it(update):
    """``localm update`` hands the update over through ``updater_handoff.hand_off`` and
    the applier it starts (``-I -S``, the base Python) carries it out after the waiting
    parent arrives at the gate: the parent exits 0 with the new version in place."""
    proc = update.popen("-m", "localm", "update", ID)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err
    assert update.version() == "2" and update.result()["state"] == "new"
    assert "localm 2 is being installed." in err


def test_a_hand_off_whose_program_does_not_take_over_discards_the_update(
        update, monkeypatch):
    import shutil

    from localm import config, procs, updater_handoff
    from localm.bugreport import LocalmError
    if procs.current() is None:
        pytest.skip(f"this test process has no process entry: {procs.problem()}")
    monkeypatch.setattr(config, "HOME_DIR", update.root / "home")
    shutil.rmtree(update.root / ".localm-update" / ID / "applier")
    gone: list = []

    class Work:
        def __exit__(self, *_a):
            gone.append(True)

    prepared = updater_handoff.Prepared(update.root, ID, "1", "2", None, True, None, False,
                                        None, _work=Work())
    with pytest.raises(LocalmError) as refused:
        updater_handoff.hand_off(prepared, mode="command", confirmed=[])
    assert "did not take it over" in refused.value.reason
    assert gone == [True] and update.is_old()


def test_other_processes_lists_what_the_update_ends_and_spares_above_and_below(update):
    """From a process of the install started by a launcher of it, which itself started a
    child: the update lists another start of localm, never the launcher above
    (``ancestors``) or the child below."""
    starter = update.root.parent / "starter.py"
    starter.write_text("\n".join([
        "import json, subprocess, sys",
        "import localm",
        "from localm import updater_handoff as uh",
        "root = sys.argv[1]",
        "child = subprocess.Popen([sys.executable, '-m', 'localm', 'stay'], cwd=root,",
        "                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)",
        "child.stdout.readline()",
        "print(json.dumps({'others': uh.other_processes(root), 'above': uh.ancestors(root)}),",
        "      flush=True)",
        "child.stdin.close()",
        "child.wait()"]), encoding="utf-8")
    launcher = update.root.parent / "launcher.py"
    launcher.write_text("import subprocess, sys\nimport localm\n"
                        "sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))\n",
                        encoding="utf-8")
    other, entry = _stay(update)
    try:
        top = subprocess.Popen([sys.executable, str(launcher), str(starter), str(update.root)],
                               cwd=str(update.root), env=update.env(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               **HIDDEN)
        (out, err), = _end(top)
    finally:
        other.stdin.close()
        _end(other)
    said = json.loads(out.strip().splitlines()[-1])
    assert [p["key"] for p in said["others"]] == [f"{entry.pid}-{entry.start}"], err
    assert said["others"][0]["purpose"] == entry.purpose
    assert said["above"], "the launcher above was not found"
    assert all(k != f"{entry.pid}-{entry.start}" for k in said["above"])


# --------------------------------------------- localm update: -y and the terminal

class _Prepared:
    def __init__(self, processes, servers=()):
        self.processes = [{"key": k, "pid": i, "purpose": p, "restarts": k in servers}
                          for i, (k, p) in enumerate(processes, 1)]
        self.servers = {k: {} for k in servers}
        self.legacy, self.edited, self.discarded = False, [], 0

    def ended(self):
        return [p for p in self.processes if p["key"] not in self.servers]

    def discard(self):
        self.discarded += 1


@pytest.mark.parametrize("processes, servers, tty, code", [
    ([("a-1", "chat")], (), False, 1),
    ([("s-1", "server")], ("s-1",), False, 75),
    ([("s-1", "server"), ("a-1", "chat")], ("s-1",), False, 1),
    ([], (), True, 75)])
def test_localm_update_yes_restarts_the_servers_and_refuses_when_anything_else_runs(
        monkeypatch, processes, servers, tty, code):
    from localm import _update_parent, updater_handoff
    from localm.cli import maintenance
    monkeypatch.setenv(_update_parent.WORK, "1")
    prepared = _Prepared(processes, servers)
    handed: list = []
    monkeypatch.setattr(updater_handoff, "prepare", lambda *_a, **_k: prepared)
    monkeypatch.setattr(updater_handoff, "hand_off", lambda p, **k: handed.append(k))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: tty, raising=False)
    with pytest.raises(SystemExit) as stop:
        maintenance._install_with_applier(3, None, yes=True)
    assert stop.value.code == code
    if code == 75:
        assert handed and handed[0]["confirmed"] == [p["key"] for p in prepared.processes]
        assert handed[0]["ask"] is False   # -y: nobody is asked
        assert prepared.discarded == 0
    else:
        assert not handed and prepared.discarded >= 1


def test_localm_update_at_a_terminal_asks_about_what_is_in_the_way(monkeypatch):
    from localm import _update_parent, updater_handoff
    from localm.cli import maintenance
    monkeypatch.setenv(_update_parent.WORK, "1")
    handed: list = []
    monkeypatch.setattr(updater_handoff, "prepare", lambda *_a, **_k: _Prepared([]))
    monkeypatch.setattr(updater_handoff, "hand_off", lambda p, **k: handed.append(k))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    with pytest.raises(SystemExit):
        maintenance._install_with_applier(3, None, yes=False)
    assert handed[0]["ask"] is True


def test_localm_update_stopped_by_ctrl_c_says_the_update_goes_on(monkeypatch, capsys):
    from localm import _update_parent
    monkeypatch.setattr(_update_parent.subprocess, "call",
                        lambda *_a, **_k: _update_parent.HANDED_OFF)

    def stop():
        raise KeyboardInterrupt

    monkeypatch.setattr(_update_parent, "wait", stop)
    assert _update_parent.run(["localm", "update"]) == 130
    assert "The update goes on" in capsys.readouterr().err


def test_localm_update_says_when_the_update_still_runs(update):
    """The update's lock is still held when the parent's time is up: it says that the
    update still runs, never that it ended without saying how."""
    from localm.update_mechanism import job
    code = ("import sys, localm\nfrom localm import _update_parent as p\n"
            "p.LOCK_S = 1.0\nlocalm._gate = lambda me: False\nprint('READY', flush=True)\n"
            "sys.stdin.readline()\nsys.exit(p.wait())\n")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(update.root),
                            env=update.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "READY"
    with job.Lock(update.root) as lock:
        job.start(lock, ID, old="1", new="2")
        _, err = proc.communicate("go\n", timeout=60)
        job.restore(lock)
    assert proc.returncode == 1
    assert "still runs after 1 s" in err and "without saying how" not in err


# --------------------------------------------------------- the hand-off's values

@pytest.mark.parametrize("change", [
    {"old": "-x"}, {"new": 2}, {"confirmed": [1]}, {"starter": None}, {"ask": "yes"},
    {"watch": {"port": "80"}}, {"restarting": [["a"]]}, {"home": "elsewhere"},
    {"relaunch": ["x"]}])
def test_a_hand_off_with_a_wrong_value_is_refused(tmp_path, monkeypatch, change):
    import io
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    good = {"mode": "command", "starter": "1-2", "confirmed": [], "home": str(tmp_path),
            "old": "1", "new": "2", "watch": None}
    assert applier.read_handoff(io.StringIO(json.dumps(good) + "\n")) == good
    with pytest.raises(ValueError):
        applier.read_handoff(io.StringIO(json.dumps({**good, **change}) + "\n"))


def test_a_record_of_the_update_gets_its_values_from_the_environment(monkeypatch):
    from click.testing import CliRunner

    from localm.cli import main, update_prepare
    from localm.update_mechanism import steps_update
    got: list = []
    monkeypatch.setattr(update_prepare, "record", lambda *a: got.append(a) or [])
    result = CliRunner().invoke(main, ["_update-record"], env={
        steps_update.RECORD_FROM: "1", steps_update.RECORD_TO: "2",
        steps_update.RECORD_KEPT: '["k"]'})
    assert result.exit_code == 0, result.output
    assert got == [("1", "2", ["k"])]


def test_a_server_let_through_that_fails_is_replaced_by_the_old_version(tmp_path):
    """The new version passed its probe but ends as the real server: the old version is
    put back and started again with the command line that follows the id."""
    update = Update(tmp_path / "inst", serve_fails=True)
    marker = update.root / "home" / "relaunched"
    port = _free_port()
    proc = update.apply(mode=applier.SERVER,
                        watch={"host": "127.0.0.1", "port": port, "scheme": "http"},
                        relaunch=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
    server = update.popen("-m", "localm", "serve", str(port), LOCALM_UPDATE_PRIMARY=ID)
    try:
        _, err = proc.communicate(timeout=300)
    finally:
        _end(server)
    deadline = time.monotonic() + 30
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert proc.returncode == 1, err
    assert update.is_old() and marker.exists()
    assert update.result()["state"] == "old" and "ended right after it started" in err


# --------------------------------------------------------------- the check's time

def test_a_check_that_does_not_end_in_its_time_is_stopped(tmp_path, monkeypatch):
    from localm import config, updater_handoff
    monkeypatch.setattr(updater_handoff, "CHECK_S", 1.0)
    monkeypatch.setattr(config, "HOME_DIR", tmp_path)
    program = tmp_path / "check.py"
    program.write_text("import sys, time\nprint('checking', file=sys.stderr, flush=True)\n"
                       "time.sleep(60)\n", encoding="utf-8")
    said: list = []
    begun = time.monotonic()
    assert updater_handoff._run_check(program, tmp_path, ID, said.append) == -1
    assert time.monotonic() - begun < 30
    assert "checking" in said and any("did not end within 1 s" in t for t in said)


def test_the_applier_of_a_server_logs_where_diagnostics_go(tmp_path, monkeypatch):
    from localm import audit, debuglog, updater_handoff
    monkeypatch.setattr(debuglog, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(audit, "diagnostics_allowed", lambda **_k: True)
    log = updater_handoff._applier_log()
    assert log is not None
    log.close()
    assert (tmp_path / "logs" / updater_handoff.APPLIER_LOG).exists()
    monkeypatch.setattr(audit, "diagnostics_allowed", lambda **_k: False)
    assert updater_handoff._applier_log() is None


# ------------------------------------------------- the modules a build's applier uses

def test_the_applier_is_copied_with_the_modules_its_own_build_names(tmp_path):
    from localm.update_mechanism import shared
    build = tmp_path / "localm"
    (build / "update_mechanism").mkdir(parents=True)
    for name in ("_apply_update", "update_package", "later", "procs", "other"):
        (build / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    (build / "update_mechanism" / "shared.py").write_text(
        'MODULES = ("_apply_update", "update_package", "later")\nALSO = ("procs", "gone")\n',
        encoding="utf-8")
    shared.copy_applier(build, tmp_path / "a")
    copied = sorted(p.name for p in (tmp_path / "a").glob("*.py"))
    assert copied == ["_apply_update.py", "later.py", "procs.py", "update_package.py"]
    (build / "later.py").unlink()
    with pytest.raises(OSError, match="later.py"):
        shared.copy_applier(build, tmp_path / "b")
    (build / "update_mechanism" / "shared.py").unlink()
    (build / "updater_components.py").write_text("", encoding="utf-8")
    shared.copy_applier(build, tmp_path / "c")
    assert sorted(p.name for p in (tmp_path / "c").glob("*.py")) == [
        "_apply_update.py", "procs.py", "update_package.py", "updater_components.py"]


@pytest.mark.skipif(os.name == "nt", reason="the update window of a desktop start is POSIX only")
@pytest.mark.parametrize("primary", [True, False])
def test_a_desktop_gui_that_goes_on_with_the_update_never_opens_the_update_window(
        update, primary):
    """H1b: a GUI started from a desktop menu (a display, no terminal) that waits at the
    gate as the process that goes on with the update is never sent to the update
    window, which would not let it through; another such start is."""
    from localm.update_mechanism import job
    code = "\n".join([
        "import os, sys, types",
        "import localm",
        "from localm import procs",
        "me = procs.current()",
        "gui = types.SimpleNamespace(pid=me.pid, start=me.start, purpose='gui', parent=me.parent)",
        "print('READY', flush=True)",
        "sys.stdin.readline()",
        "localm._NAME_S = 0.3",
        "def put_back(root, folder, update, window):",
        "    print('WINDOW', window, flush=True)",
        "    return 0",
        "localm._put_back = put_back",
        *([f"os.environ['LOCALM_UPDATE_PRIMARY'] = {ID!r}"] if primary else []),
        "localm._gate(gui)"])
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(update.root),
                            env=update.env(DISPLAY=":0"), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "READY"
    with job.Lock(update.root) as lock:
        job.start(lock, ID, old="1", new="2")
        try:
            out, err = proc.communicate("go\n", timeout=60)
        finally:
            job.restore(lock)
    assert f"WINDOW {not primary}" in out, (out, err)
