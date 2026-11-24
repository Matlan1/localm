# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record of this install's running processes (localm.procs), with real
processes of a small install: its ``localm`` package is ``__init__.py`` and
``procs.py`` of this one."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from localm import procs

REPO = Path(__file__).resolve().parent.parent
HIDDEN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
SLEEP = [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(120)"]

WORKER = """
import localm, multiprocessing as mp, sys, time
def work(q):
    import localm
    q.put(localm._procs.current().purpose)
    time.sleep(120)
if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=work, args=(q,), daemon=True)
    p.start()
    print(q.get(timeout=60), p.pid, flush=True)
    time.sleep(120)
"""


class Install:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "localm").mkdir(parents=True)
        for name in ("__init__.py", "procs.py"):
            shutil.copyfile(REPO / "localm" / name, root / "localm" / name)
        (root / "pyproject.toml").write_text("", encoding="utf-8")
        self.children: list = []

    @property
    def folder(self) -> Path:
        return self.root / procs.FOLDER

    def env(self, **extra) -> dict:
        env = dict(os.environ, PYTHONPATH=str(self.root), PYTHONUTF8="1", **extra)
        env.pop("LOCALM_CONTAINER_IMAGE", None)
        env.update(extra)
        return env

    def run(self, code: str, **extra) -> subprocess.Popen:
        """A python of this install, after its first line of output (``.first``)."""
        return self.popen([sys.executable, "-c", textwrap.dedent(code)], **extra)

    def popen(self, argv, **extra) -> subprocess.Popen:
        p = subprocess.Popen(argv, env=self.env(**extra), cwd=self.root, text=True,
                             encoding="utf-8", stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, **HIDDEN)
        self.children.append(p)
        p.first = p.stdout.readline().strip()
        return p

    def importer(self, then: str = "") -> subprocess.Popen:
        return self.run(f"import localm, os, time, sys, subprocess\n{then}\n"
                        "print(os.getpid(), flush=True)\ntime.sleep(120)")

    def list(self) -> list:
        return procs.list(self.root)

    def pids(self) -> set:
        return {e.pid for e in self.list()}

    def files(self) -> list:
        return sorted(p.name for p in self.folder.iterdir())


def _kill(pid: int) -> None:
    import signal
    try:
        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
    except OSError:
        pass


def _until(check, seconds: float = 20.0):
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end
        time.sleep(0.05)


@pytest.fixture
def make(tmp_path):
    made: list = []

    def build(name: str = "install") -> Install:
        made.append(Install(tmp_path / name))
        return made[-1]
    yield build
    for inst in made:
        for e in inst.list():
            if e.pid != os.getpid():
                _kill(e.pid)
        for p in inst.children:
            p.kill()
            p.communicate()


@pytest.mark.parametrize("name", ["plain", "with a space and ä ö ü ß"])
def test_a_process_that_imports_localm_is_listed_while_it_runs(make, name):
    inst = make(name)
    p = inst.importer()
    pid = int(p.first)
    assert pid in inst.pids()
    entry = next(e for e in inst.list() if e.pid == pid)
    assert (entry.purpose, entry.recorded, procs.alive(entry)) == ("script", True, True)
    _kill(pid)
    p.wait(20)
    _until(lambda: pid not in inst.pids(), 1.0)
    assert not procs.alive(entry)


def test_a_normal_end_leaves_no_file(make):
    inst = make()
    p = inst.run("import localm, os; print(os.getpid(), flush=True); input()")
    assert int(p.first) in inst.pids()
    p.communicate("\n", timeout=20)
    assert p.returncode == 0
    assert inst.files() == []


def test_a_hard_kill_leaves_one_file_that_the_next_process_deletes(make):
    inst = make()
    p = inst.importer()
    _kill(int(p.first))
    p.wait(20)
    (left,) = inst.files()
    _until(lambda: procs._process(int(p.first)) is None)
    q = inst.importer()
    assert left not in inst.files() and len(inst.files()) == 1
    assert int(q.first) in inst.pids()


def test_the_server_stop_leaves_no_file(monkeypatch):
    from localm.inference import http_server
    seen: list = []
    monkeypatch.setattr(http_server, "_shutdown_teardown", lambda **k: seen.append("teardown"))
    monkeypatch.setattr(procs, "leave", lambda: seen.append("leave"))
    monkeypatch.setattr(os, "_exit", lambda code: seen.append("exit"))
    http_server._do_shutdown()
    assert seen == ["teardown", "leave", "exit"]


def test_a_hard_kill_leaves_one_file_that_the_next_reader_deletes(make):
    inst = make()
    p = inst.importer()
    _kill(int(p.first))
    p.wait(20)
    assert len(inst.files()) == 1
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def _file(inst: Install, pid: int, start: str, host: str | None = None, text: str = "") -> Path:
    inst.folder.mkdir(exist_ok=True)
    name = procs._name(pid, start)
    if host is not None:
        name = host + name[name.index("-"):]
    path = inst.folder / name
    path.write_text(text or json.dumps(
        {"pid": pid, "start": start, "purpose": "gui", "parent": None}), encoding="ascii")
    return path


def test_a_pid_that_belongs_to_a_program_that_started_at_another_time_is_not_listed(make):
    inst = make()
    row = procs._process(os.getpid())
    path = _file(inst, os.getpid(), procs._text(row[1] - 1))
    assert inst.list() == []
    assert not path.exists()


def test_a_pid_reused_after_the_table_was_read_is_looked_at_again(make):
    inst = make()
    row = procs._process(os.getpid())
    path = _file(inst, os.getpid(), procs._text(row[1]))
    old_table = {os.getpid(): (row[0], row[1] - 5, "")}
    assert list(procs._recorded(str(inst.root), old_table)) == [os.getpid()]
    assert path.exists()


def test_a_process_that_exited_while_its_handle_is_held_is_not_alive():
    p = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE, **HIDDEN)
    entry = procs.Entry(p.pid, procs._text(procs._process(p.pid)[1]), "tool")
    assert procs.alive(entry)
    p.communicate(b"\n", timeout=20)
    assert not procs.alive(entry)
    assert p.pid not in procs._table()


def test_an_entry_of_another_machine_is_neither_listed_nor_deleted(make):
    inst = make()
    path = _file(inst, os.getpid(), "12345", host="0" * 16)
    assert inst.list() == []
    assert path.exists()
    assert procs.elsewhere(inst.root) == 1


def test_a_half_written_file_still_names_its_process(make):
    inst = make()
    _file(inst, os.getpid(), procs._text(procs._process(os.getpid())[1]), text='{"pi')
    assert [(e.pid, e.purpose) for e in inst.list() if e.recorded] == [(os.getpid(), "localm")]


def test_two_installs_do_not_see_each_other(make):
    a, b = make("a"), make("b")
    pa, pb = int(a.importer().first), int(b.importer().first)
    assert pa in a.pids() and pb not in a.pids()
    assert pb in b.pids() and pa not in b.pids()


def test_a_python_without_localm_is_never_listed(make):
    inst = make()
    inst.importer()
    other = inst.popen(SLEEP)
    assert other.pid not in inst.pids()


def test_a_tool_is_listed_below_the_process_that_runs_it(make):
    inst = make()
    p = inst.importer("t = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(t.pid, flush=True)")
    tool, pid = int(p.first), int(p.stdout.readline())
    entries = {e.pid: e for e in inst.list()}
    assert (entries[tool].purpose, entries[tool].recorded) == ("tool", False)
    assert entries[tool].parent == entries[pid].key
    assert tool in {e.pid for e in procs.descendants(entries[pid], inst.root)}
    _kill(tool)
    _until(lambda: tool not in inst.pids(), 1.0)


def test_start_records_a_program_that_outlives_its_starter(make):
    inst = make()
    p = inst.run("""
        import localm, sys
        from localm import procs
        t = procs.start([sys.executable, '-c', 'import time; time.sleep(120)'],
                        purpose='crash watchdog')
        print(t.pid, flush=True)
        """)
    child = int(p.first)
    p.wait(20)
    entry = next(e for e in inst.list() if e.pid == child)
    assert (entry.purpose, entry.recorded) == ("crash watchdog", True)
    _kill(child)
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def test_the_starter_deletes_the_file_of_a_program_that_has_ended(make):
    inst = make()
    p = inst.run("""
        import localm, sys
        from localm import procs
        t = procs.start([sys.executable, '-c', 'input()'], purpose='tool',
                        stdin=__import__('subprocess').PIPE)
        print(t.pid, flush=True)
        input()
        t.communicate(b'x')
        """)
    assert len(inst.files()) == 2
    p.communicate("\n", timeout=20)
    assert inst.files() == []


def test_start_records_nothing_for_a_pid_that_is_not_its_child(make, monkeypatch):
    inst = make()
    other = inst.popen(SLEEP)

    class Fake:
        pid = other.pid
    me = procs.Entry(os.getpid(), procs._text(procs._process(os.getpid())[1]), "gui")
    inst.folder.mkdir()
    monkeypatch.setattr(procs, "_CURRENT", procs.Entry(os.getpid() + 1, "1", "gui"))
    monkeypatch.setattr(procs, "current", lambda: procs.Entry(me.pid + 1, "1", "gui"))
    monkeypatch.setattr(procs, "_CURRENT_PATH", str(inst.folder / "x.json"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Fake())
    assert procs.start(["x"], purpose="tool").pid == other.pid
    assert inst.files() == []
    monkeypatch.setattr(procs, "current", lambda: me)
    procs.start(["x"], purpose="tool")
    assert len(inst.files()) == 1


def test_a_multiprocessing_worker_is_recorded(make):
    inst = make()
    (inst.root / "main.py").write_text(WORKER, encoding="utf-8")
    p = inst.popen([sys.executable, str(inst.root / "main.py")])
    purpose, pid = p.first.split()
    assert purpose == "worker"
    assert next(e for e in inst.list() if e.pid == int(pid)).purpose == "worker"


@pytest.mark.skipif(sys.platform == "win32", reason="fork")
def test_a_forked_child_does_not_remove_the_file_of_its_parent(make):
    inst = make()
    p = inst.run("""
        import localm, os, sys, time
        pid = os.fork()
        if pid == 0:
            sys.exit(0)
        os.waitpid(pid, 0)
        print(os.getpid(), flush=True)
        time.sleep(120)
        """)
    assert int(p.first) in inst.pids()


def test_the_entry_is_written_before_anything_else_of_localm(make):
    inst = make()
    p = inst.run("""
        import sys, localm
        own = sorted(m for m in sys.modules if m.startswith('localm'))
        print(own, localm._procs.current() is not None, flush=True)
        """)
    assert p.first == "['localm', 'localm.procs'] True"


def test_the_file_holds_four_fields_and_no_command_line(make):
    inst = make()
    secret = "a-model-name-nobody-may-read"
    p = inst.popen([sys.executable, "-c", "import localm, time; print(1, flush=True); time.sleep(120)",
                    secret, "gui"])
    assert p.first == "1"
    (name,) = inst.files()
    text = (inst.folder / name).read_text(encoding="ascii")
    assert secret not in text and "import" not in text and str(inst.root) not in text
    data = json.loads(text)
    assert sorted(data) == ["parent", "pid", "purpose", "start"]
    assert data["purpose"] == "script"


@pytest.mark.parametrize("argv, purpose", [
    (["python", "-m", "localm", "gui", "--port", "1"], "gui"),
    (["python", "-X", "utf8", "-m", "localm", "pull", "gui"], "pull"),
    (["python", "-m", "localm", "--version"], "command"),
    (["python", "-u", "-m", "localm._torch_gpu_probe"], "localm._torch_gpu_probe"),
    (["python", "-m", "some.other", "gui"], "script"),
    (["python", "-m", "localm.x\"y"], "script"),
    (["python", r"D:\x\.venv\Scripts\localm.exe".replace("\\", os.sep), "run", "m"], "run"),
    (["python", "/opt/localm/.venv/bin/localcoder", "do it"], "coder"),
    (["python", "/opt/localm/launcher.pyw"], "launcher"),
    (["python", "-c", "gui"], "script"),
    (["python", "-c", "from multiprocessing.spawn import spawn_main", "--multiprocessing-fork"],
     "worker"),
    (["python"], "command"),
])
def test_what_a_process_calls_itself(argv, purpose):
    assert procs._purpose(argv) == purpose


def test_every_named_command_exists():
    import click

    from localm.cli import main
    assert procs.COMMANDS <= set(main.list_commands(click.Context(main)))


def test_a_pip_install_keeps_no_record(make):
    inst = make("site-packages")
    p = inst.run("import localm; print(localm._procs.current(), flush=True)")
    assert p.first == "None"
    assert not inst.folder.exists()
    assert p.communicate(timeout=20)[1] == ""


def test_the_container_image_keeps_no_record(make):
    inst = make()
    p = inst.run("import localm; print(localm._procs.current(), flush=True)",
                 LOCALM_CONTAINER_IMAGE="1")
    assert p.first == "None"
    assert not inst.folder.exists()


def test_an_install_that_cannot_be_written_keeps_no_record_and_says_nothing(make):
    inst = make()
    inst.folder.write_text("in the way", encoding="utf-8")
    p = inst.run("import localm; from localm import procs, __version__\n"
                 "import sys; print(procs.current(), flush=True)\n"
                 "procs.start([sys.executable, '-c', ''], purpose='tool').wait()")
    assert p.first == "None"
    assert p.communicate(timeout=20) == ("", "")
    assert p.returncode == 0


def test_a_parent_that_started_after_its_child_is_not_its_parent():
    table = {1: (0, 10, ""), 2: (1, 20, ""), 3: (4, 30, ""), 4: (1, 40, ""), 5: (9, 50, "")}
    assert procs._children(table) == {1: [2, 4]}


@pytest.mark.skipif(sys.platform != "win32", reason="windows of a program")
def test_a_program_that_shows_a_window_is_the_users(make, monkeypatch):
    inst = make()
    p = inst.importer("t = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(t.pid, flush=True)")
    tool = int(p.first)
    assert tool in inst.pids()
    monkeypatch.setattr(procs, "_win_windows", lambda: {tool})
    assert tool not in inst.pids()


@pytest.mark.skipif(sys.platform == "win32", reason="sessions")
def test_a_program_in_a_session_of_its_own_is_the_users_unless_recorded(make):
    inst = make()
    p = inst.importer("""
from localm import procs
a = [sys.executable, '-c', 'import time; time.sleep(120)']
t = subprocess.Popen(a, start_new_session=True)
r = procs.start(a, purpose='comfyui', start_new_session=True)
print(t.pid, r.pid, flush=True)""")
    theirs, ours = (int(x) for x in p.first.split())
    try:
        assert theirs not in inst.pids()
        assert ours in inst.pids()
    finally:
        _kill(theirs)


def test_end_others_ends_this_install_and_nothing_else(make):
    inst, other = make("mine"), make("other")
    a = inst.importer("t = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(t.pid, flush=True)")
    tool, pid = int(a.first), int(a.stdout.readline())
    foreign = int(other.importer().first)
    left = inst.run("""
        import localm, os, subprocess, sys
        from localm import procs
        mine = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        left = procs.end_others(timeout=30)
        print(left, mine.poll(), procs.current().pid == os.getpid(), flush=True)
        mine.kill()
        """)
    assert left.first == "[] None True"
    left.wait(20)
    assert not {pid, tool} & inst.pids()
    assert a.wait(20) is not None
    assert foreign in other.pids()


def test_end_others_names_what_still_runs(make):
    inst = make()
    pid = int(inst.importer().first)
    p = inst.run("""
        import localm
        from localm import procs
        procs._end = lambda entry, force: None
        print([e.pid for e in procs.end_others(timeout=0.3)], flush=True)
        """)
    assert str(pid) in p.first


def test_end_others_ends_watchdogs_first(make):
    inst = make()
    inst.importer()
    inst.run("""
        import localm, sys, time
        from localm import procs
        procs.start([sys.executable, '-c', 'import time; time.sleep(120)'],
                    purpose='crash watchdog')
        print(1, flush=True)
        time.sleep(120)
        """)
    p = inst.run("""
        import localm
        from localm import procs
        asked = []
        procs._end = lambda entry, force: asked.append(entry.purpose)
        procs.end_others(timeout=0.3)
        print(asked, flush=True)
        """)
    assert p.first.startswith("['crash watchdog', ") and "'script'" in p.first


def test_end_others_spares_what_this_process_started(make):
    inst = make()
    pid = int(inst.importer().first)
    assert procs.end_others(inst.root, timeout=5) == []
    assert pid in inst.pids()


def test_the_end_of_a_process_is_not_the_end_of_the_one_that_has_its_pid():
    p = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE, **HIDDEN)
    row = procs._process(p.pid)
    procs._end(procs.Entry(p.pid, procs._text(row[1] - 1), "gui"), True)
    time.sleep(0.3)
    assert p.poll() is None
    procs._end(procs.Entry(p.pid, procs._text(row[1]), "gui"), True)
    assert p.wait(20) is not None
    p.stdin.close()


def test_uninstall_removes_the_folder(make):
    from localm import install_manifest as im
    inst = make()
    inst.folder.mkdir()
    items, _ = im._plan(inst.root, None, purge_data=False, defer_runtime=True, old_root="")
    assert [i.status for i in items if i.path == str(inst.folder)] == ["remove"]
