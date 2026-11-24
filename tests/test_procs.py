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
        return sorted(p.name for p in self.folder.iterdir() if p.suffix != ".mark")

    def place(self) -> tuple:
        """Where an entry of this install is written: folder, folder id, host id."""
        self.folder.mkdir(exist_ok=True)
        names = os.listdir(self.folder)
        mark = (procs._marks(names) or [procs._make_mark(str(self.folder))])[0]
        return str(self.folder), procs._folder_id(str(self.root), mark), procs._host(mark)


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


def test_the_file_of_an_ended_process_goes_with_the_next_reader(make):
    inst = make()
    p = inst.run("import localm, os; print(os.getpid(), flush=True); input()")
    pid = int(p.first)
    assert pid in inst.pids()
    p.communicate("\n", timeout=20)
    assert p.returncode == 0
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def test_a_process_is_listed_until_it_has_really_ended(make):
    inst = make()
    p = inst.run("""
        import multiprocessing.util as u, time
        u.Finalize(None, time.sleep, args=(3,), exitpriority=-100)
        import localm, os
        print(os.getpid(), flush=True)
        """)
    pid, seen = int(p.first), 0
    end = time.monotonic() + 60
    while p.poll() is None and time.monotonic() < end:
        running = procs._process(pid)
        listed = pid in inst.pids()
        if running and procs._process(pid):
            assert listed
            seen += 1
        time.sleep(0.2)
    assert p.returncode == 0 and seen > 5


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


def test_a_hard_kill_leaves_one_file_that_the_next_reader_deletes(make):
    inst = make()
    p = inst.importer()
    _kill(int(p.first))
    p.wait(20)
    assert len(inst.files()) == 1
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def _file(inst: Install, pid: int, start: str, host: str = "", at: str = "",
          text: str = "") -> Path:
    place = inst.place()
    path = inst.folder / f"{at or place[1]}-{host or place[2]}-{pid}-{start}.json"
    path.write_text(text or json.dumps(
        {"pid": pid, "start": start, "purpose": "gui", "parent": None}), encoding="utf-8")
    return path


def _me() -> str:
    return procs._text(procs._process(os.getpid())[1])


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


@pytest.mark.skipif(sys.platform == "win32", reason="processes nobody waited for")
def test_an_ended_process_nobody_waited_for_is_not_alive(make):
    inst = make()
    p = inst.importer()
    pid = int(p.first)
    entry = next(e for e in inst.list() if e.pid == pid)
    _kill(pid)
    _until(lambda: not procs.alive(entry), 5.0)
    assert procs._process(pid) is None
    os.kill(pid, 0)
    assert pid not in inst.pids()
    p.wait(20)


def test_an_entry_of_another_machine_is_neither_listed_nor_deleted_for_a_week(make):
    inst = make()
    path = _file(inst, os.getpid(), _me(), host="0" * 16)
    assert inst.list() == []
    assert path.exists() and procs.elsewhere(inst.root) == 1
    assert procs.end_others(inst.root, timeout=0) == [procs.UNSEEN]
    old = time.time() - 8 * 24 * 3600
    os.utime(path, (old, old))
    assert procs.elsewhere(inst.root) == 0
    assert inst.list() == []
    assert not path.exists()


def test_the_files_of_a_copied_install_are_stale_and_the_original_is_left_alone(make, tmp_path):
    a = make("original")
    pid = int(a.importer().first)
    shutil.copytree(a.root, tmp_path / "copy")
    b = Install.__new__(Install)
    b.root, b.children = tmp_path / "copy", []
    assert len(b.files()) == 1
    p = b.run("""
        import localm
        from localm import procs
        left = procs.end_others(timeout=10)
        print([e.purpose for e in left], flush=True)
        """)
    assert p.first == "[]"
    p.communicate(timeout=20)
    assert pid in a.pids()
    assert b.list() == [] and b.files() == []


@pytest.mark.parametrize("name, text", [
    ("{at}-{host}-\u00b2-5.json", "{}"),
    ("{at}-{host}-5-\u00b2.json", "{}"),
    ("{at}-{host}-99999999999999999999-5.json", "{}"),
    ("junk", "x"),
    ("{at}-{host}-{pid}-{start}.json", '{"pid": Infinity, "start": "1"}'),
    ("{at}-{host}-{pid}-{start}.json", "[" * 20000),
    ("{at}-{host}-{pid}-{start}.json", '{"pi'),
    ("{at}-{host}-{pid}-{start}.json", '{"pid": 1, "start": "1", "purpose": "x", "parent": null}'),
    ("{at}-{host}-{pid}-{start}.json",
     '{"pid": {pid}, "start": "{start}", "purpose": "x", "parent": {"pid": "x"}}'),
    ("{at}-{host}-{pid}-{start}.json", "\x00" * 70000),
])
def test_a_malformed_file_is_skipped_and_deleted(make, name, text):
    inst = make()
    _, at, host = inst.place()
    keys = {"{at}": at, "{host}": host, "{pid}": str(os.getpid()), "{start}": _me()}
    for k, v in keys.items():
        name, text = name.replace(k, v), text.replace(k, v)
    (inst.folder / name).write_text(text, encoding="utf-8")
    other = int(inst.importer().first)
    assert inst.pids() >= {other} and os.getpid() not in inst.pids()
    assert len(inst.files()) == 1


@pytest.mark.parametrize("part", range(4))
@pytest.mark.parametrize("value", [
    "../x", "..\\x", "/etc/x", "C:\\x", "", ".", "..", "x", "1/2", "1\x002", " 5", "+5", "-5", "1_0",
    "5\n", "\u0665", "0x5", "f" * 15, "f" * 17, "F" * 16, "f" * 31 + ".5", "f" * 32 + "./5",
    "f" * 32 + ".", ".5/", "9" * 30, -1, 0, 1 << 32, True, 5.0, None, b"5"])
def test_nothing_from_outside_shapes_the_name_of_a_file(make, part, value):
    inst = make()
    folder, at, host = inst.place()
    parts = [at, host, 5, "5"]
    assert procs._name(*parts) == f"{at}-{host}-5-5.json"
    parts[part] = value
    with pytest.raises((ValueError, TypeError, AttributeError)):
        procs._name(*parts)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        procs._write((folder, parts[0], parts[1]), procs.Entry(parts[2], parts[3], "gui"))
    assert inst.files() == []


def test_a_folder_or_a_pipe_in_the_record_harms_nothing(make):
    inst = make()
    _, at, host = inst.place()
    (inst.folder / f"{at}-{host}-{os.getpid()}-{_me()}.json").mkdir()
    if hasattr(os, "mkfifo"):
        os.mkfifo(inst.folder / f"{at}-{host}-{os.getppid()}-"
                  f"{procs._text(procs._process(os.getppid())[1])}.json")
    other = int(inst.importer().first)
    assert inst.pids() == {other}


@pytest.mark.parametrize("count", [0, 20, 500])
def test_a_start_looks_at_few_files_and_reads_the_table_once(make, monkeypatch, count):
    inst = make()
    for i in range(count):
        _file(inst, 4294967000 - 4 * i if sys.platform == "win32" else 4200000 + i, f"{i + 1}"
              if not procs._boot() else f"{procs._boot()}.{i + 1}")
    tables: list = []
    real = procs._table
    monkeypatch.setattr(procs, "_table", lambda: tables.append(1) or real())
    if sys.platform == "win32":
        win = procs._win_table
        monkeypatch.setattr(procs, "_win_table", lambda: tables.append(1) or win())
    names = os.listdir(inst.folder) if count else []
    procs._sweep(str(inst.root), names, procs._marks(names))
    assert len(tables) <= 2 and (count > 0 or not tables)
    if count:
        assert len(names) - len(os.listdir(inst.folder)) in (
            min(procs._SWEEP, count), min(procs._SWEEP, count) - 1)


def test_a_running_process_that_cannot_be_opened_costs_one_table(make, monkeypatch):
    inst = make()
    for i in range(6):
        _file(inst, os.getpid(), f"{procs._boot()}.{i + 1}" if procs._boot() else f"{i + 1}")
    tables: list = []
    real = procs._table
    monkeypatch.setattr(procs, "_table", lambda: tables.append(1) or real())
    monkeypatch.setattr(procs, "_process", lambda pid, table=None: (table or {}).get(pid))
    monkeypatch.setattr(procs, "_gone", lambda pid: False)
    names = os.listdir(inst.folder)
    procs._sweep(str(inst.root), names, procs._marks(names))
    assert len(tables) == 1
    assert inst.files() == []


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


def test_start_records_nothing_for_a_pid_that_is_not_its_child(make, monkeypatch):
    inst = make()
    other = inst.popen(SLEEP)

    class Fake:
        pid = other.pid
    monkeypatch.setattr(procs, "_PLACE", inst.place())
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Fake())
    monkeypatch.setattr(procs, "current", lambda: procs.Entry(os.getpid() + 1, "1", "gui"))
    assert procs.start(["x"], purpose="tool").pid == other.pid
    assert inst.files() == []
    monkeypatch.setattr(procs, "current", lambda: procs.Entry(os.getpid(), _me(), "gui"))
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
    p = inst.run("import localm; from localm import procs\n"
                 "print(procs.current(), bool(procs.problem()), procs.end_others(), flush=True)")
    assert p.first == "None True [<others may run unseen>]"
    assert not inst.folder.exists()
    assert p.communicate(timeout=20)[1] == ""


def test_the_container_image_keeps_no_record(make):
    inst = make()
    p = inst.run("import localm; print(localm._procs.current(), localm._procs.problem(), flush=True)",
                 LOCALM_CONTAINER_IMAGE="1")
    assert p.first == "None the container image keeps no record"
    assert not inst.folder.exists()


def test_an_install_that_cannot_be_written_keeps_no_record_and_says_nothing(make):
    inst = make()
    inst.folder.write_text("in the way", encoding="utf-8")
    p = inst.run("import localm; from localm import procs, __version__\n"
                 "import sys; print(procs.current(), bool(procs.problem()), flush=True)\n"
                 "procs.start([sys.executable, '-c', ''], purpose='tool').wait()")
    assert p.first == "None True"
    assert p.communicate(timeout=20) == ("", "")
    assert p.returncode == 0


def test_the_record_lies_in_the_install_wherever_the_data_folder_is(make, tmp_path):
    inst = make()
    data = tmp_path / "data"
    data.mkdir()
    p = inst.popen([sys.executable, "-c", "import localm; print(1, flush=True); input()"],
                   LOCALM_HOME=str(data))
    assert len(inst.files()) == 1 and list(data.iterdir()) == []
    p.communicate("\n", timeout=20)


def test_a_read_only_install_keeps_no_record_and_says_nothing(make):
    inst = make()
    if sys.platform == "win32":
        _icacls(inst.root, "/deny", f"*{procs._win_account(os.getpid())}:(WD,AD)")
    else:
        if os.getuid() == 0:
            pytest.skip("root may write anywhere")
        os.chmod(inst.root, 0o500)
    try:
        p = inst.run("import localm; from localm import procs\n"
                     "print(procs.current(), bool(procs.problem()), flush=True)")
        assert p.first == "None True"
        assert p.communicate(timeout=20) == ("", "")
        assert not inst.folder.exists()
    finally:
        if sys.platform == "win32":
            _icacls(inst.root, "/remove:d", f"*{procs._win_account(os.getpid())}")
        else:
            os.chmod(inst.root, 0o700)


def test_many_processes_at_once_are_all_listed(make):
    inst = make()
    started = [subprocess.Popen(
        [sys.executable, "-c", "import localm, os; print(os.getpid(), flush=True); input()"],
        env=inst.env(), cwd=inst.root, text=True, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **HIDDEN) for _ in range(8)]
    inst.children += started
    pids = {int(p.stdout.readline()) for p in started}
    assert pids <= inst.pids() and len(inst.files()) == 8
    assert len([n for n in os.listdir(inst.folder) if n.endswith(".mark")]) >= 1
    for p in started:
        p.communicate("\n", timeout=30)
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def test_a_number_made_after_the_files_were_listed_is_still_this_folders(make):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    record = procs._Record(str(inst.root), [])
    assert record.kind(path.name)[0] == "entry"
    assert record.kind("0" * 16 + path.name[16:])[0] == "stale"


def test_git_does_not_see_the_record():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert procs.FOLDER + "/" in lines


@pytest.mark.skipif(sys.platform == "win32", reason="file modes")
@pytest.mark.parametrize("install, record", [(0o755, 0o1777), (0o700, 0o700), (0o750, 0o1770)])
def test_whoever_may_run_the_install_may_write_the_record(make, install, record):
    inst = make()
    os.chmod(inst.root, install)
    inst.importer()
    assert os.stat(inst.folder).st_mode & 0o7777 == record
    assert not procs._others_kept_out(str(inst.root))
    os.chmod(inst.folder, 0o700)
    assert procs._others_kept_out(str(inst.root)) is (install != 0o700)


def _icacls(path, *args) -> None:
    subprocess.run([os.path.join(os.environ["SystemRoot"], "System32", "icacls.exe"),
                    str(path), *args], check=True, capture_output=True, **HIDDEN)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows rights")
def test_whoever_may_run_the_install_may_write_the_record_on_windows(make):
    inst = make()
    me = procs._win_account(os.getpid())
    _icacls(inst.root, "/inheritance:r", "/grant", f"*{me}:(OI)(CI)F",
            "/grant", "*S-1-5-32-545:(OI)(CI)RX")
    inst.importer()
    assert procs._win_kept_out(str(inst.root), str(inst.folder)) == []
    users = procs._WinSecurity(str(inst.folder))
    assert users.mask("S-1-5-32-545") & 0x43 == 0x3
    users.close()
    _icacls(inst.folder, "/remove:g", "*S-1-5-32-545", "*S-1-5-11")
    _icacls(inst.folder, "/inheritance:r", "/grant", f"*{me}:(OI)(CI)F")
    assert "S-1-5-32-545" in procs._win_kept_out(str(inst.root), str(inst.folder))
    assert procs.end_others(inst.root, timeout=0)[-1] is procs.UNSEEN


def test_an_entry_another_account_wrote_is_not_believed(make, monkeypatch):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    assert os.getpid() in inst.pids()
    if sys.platform == "win32":
        class Other:
            def __init__(self, at):
                self.owner = "S-1-5-21-1" if str(at).endswith(".json") else "S-1-5-21-2"

            def close(self):
                pass
        monkeypatch.setattr(procs, "_WinSecurity", Other)
    else:
        real = os.stat
        monkeypatch.setattr(procs, "_account", lambda pid: os.getuid() + 1)
        monkeypatch.setattr(procs.os, "stat", lambda p, **k: os.stat_result(
            [0] * 4 + [os.getuid() + 2] + [0] * 5) if str(p) == str(inst.folder) else real(p, **k))
        if os.getuid() == 0:
            pytest.skip("root may say what any process is")
    assert os.getpid() not in inst.pids()
    assert path.exists()


def test_a_child_that_the_clock_says_started_first_is_still_a_child(monkeypatch):
    now = {1: 5, 2: 6}
    monkeypatch.setattr(procs, "_order", lambda pid, start: now[pid])
    assert procs._proven(1, 2, {1: (0, 1000, ""), 2: (1, 10, "")})
    assert not procs._proven(2, 1, {1: (2, 1000, ""), 2: (0, 10, "")})


def test_the_order_of_creation_is_read_from_the_system(make):
    inst = make()
    p = inst.importer()
    pid = int(p.first)
    table = procs._table()
    mine = procs._order(os.getpid(), table[os.getpid()][1])
    theirs = procs._order(pid, table[pid][1])
    assert mine is not None and theirs is not None and mine <= theirs
    assert procs._later((pid, table[pid][1]), (os.getpid(), table[os.getpid()][1])) or mine == theirs


def test_a_parent_that_started_after_its_child_is_not_its_parent(monkeypatch):
    monkeypatch.setattr(procs, "_order", lambda pid, start: None)
    table = {1: (0, 10, ""), 2: (1, 20, ""), 3: (4, 30, ""), 4: (1, 40, ""), 5: (9, 50, "")}
    assert procs._children(table) == {1: [2, 4], 4: [3]}
    assert procs._proven(1, 2, table) and not procs._proven(4, 3, table)


def test_a_program_opened_for_the_user_is_neither_listed_nor_ended(make):
    inst = make()
    p = inst.importer("""
from localm import procs
a = [sys.executable, '-c', 'import time; time.sleep(60)']
tool = subprocess.Popen(a)
users = procs.open_for_user(subprocess.Popen, a)
later = subprocess.Popen(a)
print(tool.pid, users.pid, later.pid, flush=True)""")
    tool, users, later = (int(x) for x in p.first.split())
    try:
        assert {tool, later} <= inst.pids() and users not in inst.pids()
        q = inst.run("import localm; from localm import procs\nprint(procs.end_others(timeout=20), flush=True)")
        assert q.first == "[]"
        assert procs._process(users) is not None
    finally:
        _kill(users)


@pytest.mark.skipif(sys.platform != "win32", reason="programs with windows")
def test_a_program_that_is_not_for_the_console_is_the_users(make):
    inst = make()
    gui = os.path.join(sys.base_prefix, "pythonw.exe")
    p = inst.importer(f"t = subprocess.Popen([{gui!r}, '-c', 'import time; time.sleep(60)'])\n"
                      "print(t.pid, flush=True)")
    users = int(p.first)
    try:
        assert users not in inst.pids()
    finally:
        _kill(users)


@pytest.mark.skipif(sys.platform != "win32", reason="windows of the user")
def test_windows_that_cannot_be_seen_make_a_tool_of_a_tool_the_users(make, monkeypatch):
    inst = make()
    (inst.root / "nest.py").write_text(
        "import subprocess, sys, time\n"
        "print(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']).pid,"
        " flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    p = inst.importer("""
t = subprocess.Popen([sys.executable, 'nest.py'], stdout=subprocess.PIPE, text=True)
print(t.pid, t.stdout.readline(), flush=True)""")
    tool, below = (int(x) for x in p.first.split())
    try:
        assert {tool, below} <= inst.pids()
        monkeypatch.setattr(procs, "_win_windows", lambda: None)
        assert tool in inst.pids() and below not in inst.pids()
    finally:
        _kill(below)


@pytest.mark.skipif(sys.platform != "win32", reason="desktops")
def test_the_windows_of_another_desktop_are_not_taken_for_the_users(make):
    inst = make()
    p = inst.run("""
        import ctypes, localm
        from localm import procs
        u = ctypes.WinDLL('user32')
        u.CreateDesktopW.restype = ctypes.c_void_p
        u.SetThreadDesktop.argtypes = [ctypes.c_void_p]
        here = procs._win_windows()
        d = u.CreateDesktopW('localm-test', None, None, 0, 0x10000000, None)
        print(bool(d), bool(u.SetThreadDesktop(d)), procs._win_windows(), flush=True)
        """)
    assert p.first == "True True None"


@pytest.mark.skipif(sys.platform != "win32", reason="windows of a program")
def test_a_program_that_shows_a_window_is_the_users(make, monkeypatch):
    inst = make()
    p = inst.importer("t = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                      "print(t.pid, flush=True)")
    tool = int(p.first)
    assert tool in inst.pids()
    monkeypatch.setattr(procs, "_win_windows", lambda: {tool})
    assert tool not in inst.pids()
    _kill(tool)


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


CHAIN = """
import localm, os, subprocess, sys
from localm import procs
depth = int(sys.argv[1])
if depth:
    print(os.getpid(), flush=True)
    sys.exit(subprocess.call([sys.executable, __file__, str(depth - 1)]))
print(procs.end_others(timeout=30), flush=True)
sys.exit(7)
"""


def test_end_others_spares_every_localm_process_that_waits_for_it(make):
    inst = make()
    other = int(inst.importer().first)
    (inst.root / "chain.py").write_text(CHAIN, encoding="utf-8")
    p = inst.popen([sys.executable, str(inst.root / "chain.py"), "2"])
    out = p.communicate(timeout=90)[0]
    assert p.returncode == 7 and "[]" in out
    assert other not in inst.pids()


def test_end_others_spares_what_the_caller_names(make):
    inst = make()
    a, b = int(inst.importer().first), int(inst.importer().first)
    p = inst.run(f"""
        import localm
        from localm import procs
        keep = [e for e in procs.list() if e.pid == {a}]
        print(procs.end_others(timeout=30, spare=keep), flush=True)
        """)
    assert p.first == "[]"
    assert a in inst.pids() and b not in inst.pids()


def test_one_round_of_end_others_uses_one_table(make):
    inst = make()
    p = inst.run("""
        import localm
        from localm import procs
        tables, real = [], procs._table
        procs._table = lambda: tables.append(1) or real()
        procs.end_others(timeout=0)
        print(len(tables), flush=True)
        """)
    assert p.first == "1"


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
