# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record of this install's running processes (localm.procs), with real
processes of a small install: its ``localm`` package is ``__init__.py`` and
``procs.py`` of this one."""

from __future__ import annotations

import ast
import errno
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
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

    def place(self) -> tuple:
        """Where an entry of this install is written: the install and its id."""
        self.folder.mkdir(exist_ok=True)
        return str(self.root), procs._here(str(self.root))


def _kill(pid: int) -> None:
    import signal
    try:
        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
    except OSError:
        pass


def _is_link(path: Path) -> bool:
    return procs._is_link(os.lstat(path))


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
        for e in inst.list() if inst.folder.is_dir() and not _is_link(inst.folder) else ():
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


def _file(inst: Install, pid: int, start: str, here: str = "", text: str = "") -> Path:
    own = inst.place()[1]
    path = inst.folder / f"{here or own}-{pid}-{start}.proc"
    path.write_text(text or f"{path.name[:-5]}\ngui\n\n", encoding="utf-8", newline="")
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


def test_a_file_of_another_machine_or_folder_is_neither_listed_nor_deleted(make):
    inst = make()
    live = _file(inst, os.getpid(), _me(), here="0" * 16)
    gone = _file(inst, os.getpid(), f"{_me()}1", here="0" * 15 + "1")
    old = time.time() - 400 * 24 * 3600
    for path in (live, gone):
        os.utime(path, (old, old))
    assert inst.list() == [] and procs.end_others(inst.root, timeout=0) == []
    inst.importer()
    assert live.exists() and gone.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the folder of a process is renamed")
def test_a_renamed_install_still_lists_its_processes(make, tmp_path):
    inst = make("before")
    pid = int(inst.importer().first)
    inst.root.rename(tmp_path / "after")
    inst.root = tmp_path / "after"
    assert pid in inst.pids()


def test_a_copied_install_ignores_the_files_of_the_original(make, tmp_path):
    a = make("original")
    pid = int(a.importer().first)
    shutil.copytree(a.root, tmp_path / "copy")
    b = Install.__new__(Install)
    b.root, b.children = tmp_path / "copy", []
    copied = b.files()
    assert len(copied) == 1
    p = b.run("""
        import localm
        from localm import procs
        left = procs.end_others(timeout=10)
        print([e.purpose for e in left], flush=True)
        """)
    assert p.first == "[]"
    p.communicate(timeout=20)
    assert pid in a.pids()
    assert b.list() == [] and b.files() == copied


@pytest.mark.parametrize("name, text", [
    ("{here}-\u00b2-5.proc", "{name}\ngui\n\n"),
    ("{here}-5-\u00b2.proc", "{name}\ngui\n\n"),
    ("{here}-99999999999999999999-5.proc", "{name}\ngui\n\n"),
    ("{here}-0{pid}-{start}.proc", "{name}\ngui\n\n"),
    ("{here}-{pid}-.{start}.proc", "{name}\ngui\n\n"),
    ("{here}-{pid}-{start}.0000.new", "{name}\ngui\n\n"),
    ("{here}-{pid}-{start}.PROC2", "{name}\ngui\n\n"),
    ("junk", "x"),
    ("thesis.odt", "x"),
    ("{here}-{pid}-{start}.proc", ""),
    ("{here}-{pid}-{start}.proc", "{name}"),
    ("{here}-{pid}-{start}.proc", "{name}\ngui"),
    ("{here}-{pid}-{start}.proc", "{name}\ngui\n"),
    ("{here}-{pid}-{start}.proc", "{name}\ngui\n\nx\n"),
    ("{here}-{pid}-{start}.proc", "{name}\ngui\nx\n"),
    ("{here}-{pid}-{start}.proc", "{name}\ngu\x00i\n\n"),
    ("{here}-{pid}-{start}.proc", "{name}\ngu\u00fci\n\n"),
    ("{here}-{pid}-{start}.proc", "{here}-1-1\ngui\n\n"),
    ("{here}-{pid}-{start}.proc", "0000000000000000-{pid}-{start}\ngui\n\n"),
    ("{here}-{pid}-{start}.proc", "{name}\ngui\n\n" + "1-1\n" * 20000),
])
def test_a_malformed_file_is_neither_listed_nor_deleted(make, name, text):
    inst = make()
    keys = {"{here}": inst.place()[1], "{pid}": str(os.getpid()), "{start}": _me()}
    keys["{name}"] = "{here}-{pid}-{start}"
    for text_of_key in ("{name}", "{here}", "{pid}", "{start}"):
        name = name.replace(text_of_key, keys[text_of_key])
        text = text.replace(text_of_key, keys[text_of_key])
    (inst.folder / name).write_text(text, encoding="utf-8", newline="")
    assert inst.list() == []
    procs._sweep(*inst.place())
    assert (inst.folder / name).exists() and len(inst.files()) == 1


def test_the_control_of_the_malformed_files_is_believed(make):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    assert os.getpid() in inst.pids()
    path.write_text(f"{path.name[:-5]}\ngui\n1-1\n2-2\n", newline="")
    entry = next(e for e in inst.list() if e.recorded)
    assert entry.parent == (1, "1") and entry.users == {(2, "2")}


@pytest.mark.parametrize("part", range(3))
@pytest.mark.parametrize("value", [
    "../x", "..\\x", "/etc/x", "C:\\x", "", ".", "..", "x", "1/2", "1\x002", " 5", "+5", "-5", "1_0",
    "5\n", "\u0665", "0x5", "f" * 15, "f" * 17, "F" * 16, "f" * 31 + ".5", "f" * 32 + "./5",
    "f" * 32 + ".", ".5/", "9" * 30, -1, 0, 1 << 32, True, 5.0, None, b"5"])
def test_nothing_from_outside_shapes_the_name_of_a_file(make, part, value):
    inst = make()
    install, here = inst.place()
    parts = [here, 5, "5"]
    assert procs._name(*parts) == f"{here}-5-5.proc"
    parts[part] = value
    with pytest.raises((ValueError, TypeError, AttributeError)):
        procs._name(*parts)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        procs._write((install, parts[0]), procs.Entry(parts[1], parts[2], "gui"))
    assert inst.files() == []


def test_a_folder_or_a_pipe_in_the_record_harms_nothing(make):
    inst = make()
    here = inst.place()[1]
    (inst.folder / f"{here}-{os.getpid()}-{_me()}.proc").mkdir()
    if hasattr(os, "mkfifo"):
        os.mkfifo(inst.folder / f"{here}-{os.getppid()}-"
                  f"{procs._text(procs._process(os.getppid())[1])}.proc")
    other = int(inst.importer().first)
    assert inst.pids() == {other}


@pytest.mark.parametrize("count", [0, 20, 500])
@pytest.mark.parametrize("here", ["", "0" * 16])
def test_a_start_looks_at_few_files_and_reads_no_table(make, monkeypatch, count, here):
    inst = make()
    place = inst.place()
    for i in range(count):
        _file(inst, 4294967000 - 4 * i if sys.platform == "win32" else 4200000 + i,
              procs._text(i + 1), here=here)
    looked: list = []
    real = procs._believed
    with monkeypatch.context() as m:
        m.setattr(procs, "_believed", lambda *a: looked.append(1) or real(*a))
        for name in ("_table", "_win_table"):
            m.setattr(procs, name, lambda: pytest.fail("the table was read"))
        procs._sweep(*place)
    few = 0 if here else min(procs._SWEEP, count)
    assert len(looked) == few and len(inst.files()) == count - few


def test_a_start_leaves_the_file_of_a_process_that_cannot_be_opened(make, monkeypatch):
    inst = make()
    _file(inst, os.getpid(), procs._text(1))
    with monkeypatch.context() as m:
        m.setattr(procs, "_process", lambda pid, table=None: procs._UNKNOWN)
        procs._sweep(*inst.place())
    assert len(inst.files()) == 1


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


def test_the_file_holds_four_lines_and_no_command_line(make):
    inst = make()
    secret = "a-model-name-nobody-may-read"
    p = inst.popen([sys.executable, "-c", "import localm, time; print(1, flush=True); time.sleep(120)",
                    secret, "gui"])
    assert p.first == "1"
    (name,) = inst.files()
    text = (inst.folder / name).read_bytes().decode("ascii")
    entry = next(e for e in inst.list() if e.recorded)
    parent = "%d-%s" % entry.parent if entry.parent else ""
    assert text == f"{name[:-5]}\nscript\n{parent}\n"
    assert name == f"{inst.place()[1]}-{entry.pid}-{entry.start}.proc"


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
    assert p.first == "None True []"
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
        _icacls(inst.root, "/deny", "*S-1-1-0:(WD,AD)")
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
            _icacls(inst.root, "/remove:d", "*S-1-1-0")
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
    for p in started:
        p.communicate("\n", timeout=30)
    _until(lambda: inst.list() == [])
    assert inst.files() == []


def test_git_does_not_see_the_record():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert procs.FOLDER + "/" in lines


@pytest.mark.skipif(sys.platform == "win32", reason="file modes")
@pytest.mark.parametrize("mode", [0o755, 0o700])
def test_the_record_is_the_users_own_whatever_the_install_is(make, mode):
    inst = make()
    os.chmod(inst.root, mode)
    inst.importer()
    assert os.stat(inst.folder).st_mode & 0o7777 == 0o700
    (name,) = inst.files()
    assert os.stat(inst.folder / name).st_mode & 0o7777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="file modes")
@pytest.mark.parametrize("mode", [0o755, 0o775, 0o777, 0o1777, 0o700])
def test_a_record_with_other_rights_works(make, mode):
    inst = make()
    inst.folder.mkdir()
    os.chmod(inst.folder, mode)
    pid = int(inst.importer().first)
    assert inst.pids() == {pid}
    left = inst.run("import localm\nfrom localm import procs\n"
                    "print(procs.end_others(timeout=30), procs.problem() == '', flush=True)")
    assert left.first == "[] True"
    assert pid not in inst.pids()
    assert os.stat(inst.folder).st_mode & 0o7777 == mode


def _icacls(path, *args) -> str:
    done = subprocess.run([os.path.join(os.environ["SystemRoot"], "System32", "icacls.exe"),
                           str(path), *args], check=True, capture_output=True, **HIDDEN)
    return done.stdout.decode("utf-8", "replace")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows rights")
def test_the_record_inherits_the_rights_of_the_install(make):
    inst = make()
    _icacls(inst.root, "/grant", "*S-1-5-32-546:(OI)(CI)RX")
    plain = inst.root / "plain"
    plain.mkdir()
    inst.importer()
    record, beside = (re.findall(r"\S+:(?:\([A-Z,]+\))+", _icacls(p)) for p in (inst.folder, plain))
    assert record == beside and any(r.endswith("(RX)") for r in record)


def _planted(inst: Install) -> subprocess.Popen:
    """A process that is not localm's."""
    return inst.popen(SLEEP)


def _key_of(pid: int) -> tuple:
    return pid, procs._text(procs._process(pid)[1])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows rights")
def test_a_file_that_cannot_be_read_still_names_its_process_on_windows(make):
    inst = make()
    other = _planted(inst)
    path = _file(inst, *_key_of(other.pid))
    assert other.pid in inst.pids()
    _icacls(path, "/deny", "*S-1-1-0:(RD)")
    _cannot_be_read(inst, other.pid)
    _icacls(path, "/remove:d", "*S-1-1-0")


END = """
import localm
from localm import procs
print(procs.end_others(timeout={timeout}), flush=True)"""


def _cannot_be_read(inst: Install, pid: int) -> None:
    """The process with the file that cannot be read is listed, its children are not; a
    start works; end_others ends it."""
    begun = time.monotonic()
    (entry,) = inst.list()
    assert time.monotonic() - begun < 5
    assert (entry.pid, entry.purpose, entry.recorded, entry.users) == (pid, "unknown", True, None)
    assert inst.run(END.format(timeout=5)).first == "[]"
    _until(lambda: procs._process(pid) is None)


def _unreadable(path: Path) -> None:
    if sys.platform == "win32":
        _icacls(path, "/deny", "*S-1-1-0:(RD)")
    else:
        os.chmod(path, 0)


@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root may read anything")
def test_below_a_process_whose_file_cannot_be_read_nothing_is_listed_or_ended(make):
    inst = make()
    p = inst.importer("""
from localm import procs
a = [sys.executable, '-c', 'import time; time.sleep(60)']
users = procs.open_for_user(subprocess.Popen, a)
tool = subprocess.Popen(a)
print(users.pid, tool.pid, flush=True)""")
    users, tool = (int(x) for x in p.first.split())
    pid = int(p.stdout.readline())
    try:
        assert {pid, tool} <= inst.pids() and users not in inst.pids()
        (name,) = inst.files()
        _unreadable(inst.folder / name)
        assert inst.pids() & {pid, tool, users} == {pid}
        assert inst.run(END.format(timeout=3)).first == "[]"
        _until(lambda: procs._process(pid) is None)
        assert procs._process(users) is not None and procs._process(tool) is not None
    finally:
        _kill(users)
        _kill(tool)


@pytest.mark.skipif(sys.platform != "win32", reason="files another program has open")
def test_end_others_waits_for_a_file_another_program_holds(make):
    import _winapi
    inst = make()
    p = inst.importer("t = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                      "print(t.pid, flush=True)")
    tool, pid = int(p.first), int(p.stdout.readline())
    (name,) = inst.files()
    handle = _winapi.CreateFile(str(inst.folder / name), 0x80000000, 0, 0, 3, 0, 0)
    closing = threading.Timer(3, _winapi.CloseHandle, [handle])
    closing.start()
    assert inst.run(END.format(timeout=20)).first == "[]"
    closing.join()
    assert procs._process(pid) is None and procs._process(tool) is None


def test_what_is_in_the_way_is_named(make):
    inst = make()
    p = inst.importer()
    pid = int(p.first)
    q = inst.run("""
        import localm
        from localm import procs
        procs._end = lambda entry, force: None
        left = procs.end_others(timeout=0.3)
        print([(e.pid, e.in_the_way) for e in left], flush=True)
        """)
    left = dict(ast.literal_eval(q.first))
    (way,) = left[pid]
    assert way["pid"] == pid and os.path.isfile(way["path"])
    assert way["name"] == os.path.basename(way["path"]) and "python" in way["name"].lower()


@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root may read anything")
def test_what_holds_a_file_of_the_record_is_named(make):
    inst = make()
    pid = int(inst.importer().first)
    (name,) = inst.files()
    path = str(inst.folder / name)
    _unreadable(inst.folder / name)
    asked: list = []

    def holders(files):
        asked.append(files)
        return [{"pid": 77, "name": "Scanner"}]
    way = next(e for e in procs.list(inst.root, holders=holders) if e.pid == pid).in_the_way
    assert way == ({"path": path, "pid": 77, "name": "Scanner"},) and asked == [[path]]
    for silent in (None, lambda files: [], lambda files: 1 / 0):
        way = next(e for e in procs.list(inst.root, holders=silent) if e.pid == pid).in_the_way
        assert way == ({"path": path, "pid": None, "name": ""},)


@pytest.mark.skipif(sys.platform != "win32", reason="the Restart Manager")
def test_the_program_that_holds_a_file_is_the_one_windows_names(make):
    import _winapi

    from localm._apply_update import rm_holders
    inst = make()
    other = _planted(inst)
    path = _file(inst, *_key_of(other.pid))
    handle = _winapi.CreateFile(str(path), 0x80000000, 0, 0, 3, 0, 0)
    try:
        way = next(e for e in procs.list(inst.root, holders=rm_holders)
                   if e.pid == other.pid).in_the_way
    finally:
        _winapi.CloseHandle(handle)
    assert [w["pid"] for w in way] == [os.getpid()] and way[0]["name"]


def test_a_file_of_which_nothing_can_be_told_is_an_error(make, monkeypatch):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    table = {os.getpid(): procs._process(os.getpid())}

    def fail(*a, **k):
        raise PermissionError(errno.EACCES, "denied")
    real = os.lstat
    with monkeypatch.context() as m:
        m.setattr(os, "lstat", lambda p, **k: fail() if str(p) == str(path) else real(p, **k))
        with pytest.raises(procs.CannotTell, match=path.name) as raised:
            procs._recorded(str(inst.root), table, holders=lambda f: [{"pid": 7, "name": "x"}])
        assert raised.value.in_the_way == ({"path": str(path), "pid": 7, "name": "x"},)
    with monkeypatch.context() as m:
        m.setattr(procs, "_believed", fail)
        m.setattr(procs, "_state", lambda *a: None)
        with pytest.raises(OSError, match=path.name):
            procs._recorded(str(inst.root), table)
        m.setattr(procs, "_state", lambda pid, start, table=None: table is not None)
        assert procs._recorded(str(inst.root), table) == {}


@pytest.mark.skipif(sys.platform != "win32", reason="files another program has open")
def test_a_file_another_program_holds_is_not_skipped(make):
    import _winapi
    inst = make()
    other = _planted(inst)
    path = _file(inst, *_key_of(other.pid))
    handle = _winapi.CreateFile(str(path), 0x80000000, 0, 0, 3, 0, 0)
    try:
        _cannot_be_read(inst, other.pid)
    finally:
        _winapi.CloseHandle(handle)


def test_a_file_the_system_cannot_read_now_is_not_skipped(make, monkeypatch):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    for name, number in (("open", errno.EMFILE), ("read", errno.EIO)):
        def fail(*a, number=number):
            raise OSError(number, os.strerror(number))
        with monkeypatch.context() as m:
            m.setattr(os, name, fail)
            found = procs._recorded(str(inst.root), {os.getpid(): procs._process(os.getpid())})
        assert [e.purpose for e in found.values()] == ["unknown"] and path.exists()


def test_a_file_of_an_ended_process_that_cannot_be_read_is_left(make, monkeypatch):
    inst = make()
    path = _file(inst, os.getpid(), procs._text(1))

    def fail(*a):
        raise OSError(errno.EIO, "x")
    with monkeypatch.context() as m:
        m.setattr(procs, "_believed", fail)
        procs._sweep(*inst.place())
        assert inst.list() == []
    assert path.exists()


def test_what_a_cut_off_write_left_goes_with_its_process(make):
    inst = make()
    here = inst.place()[1]
    live = inst.folder / f"{here}-{os.getpid()}-{_me()}.0123abcd.new"
    gone = inst.folder / f"{here}-{os.getpid()}-{procs._text(1)}.0123abcd.new"
    odd = inst.folder / f"{here}-{os.getpid()}-{procs._text(1)}.0123abcx.new"
    for look in (lambda: procs._sweep(*inst.place()), inst.list):
        for path in (live, gone, odd):
            path.write_text("", encoding="utf-8")
        look()
        assert live.exists() and odd.exists() and not gone.exists()
    assert inst.list() == []


@pytest.mark.skipif(sys.platform != "win32", reason="files another program has open")
def test_a_file_another_program_has_for_a_moment_is_still_believed(make):
    import _winapi
    inst = make()
    path = _file(inst, os.getpid(), _me())
    handle = _winapi.CreateFile(str(path), 0x80000000, 0, 0, 3, 0, 0)
    closing = threading.Timer(0.1, _winapi.CloseHandle, [handle])
    closing.start()
    assert os.getpid() in inst.pids()
    closing.join()


@pytest.mark.skipif(sys.platform == "win32", reason="file modes")
def test_a_file_that_cannot_be_read_still_names_its_process(make):
    if os.geteuid() == 0:
        pytest.skip("root may read anything")
    inst = make()
    other = _planted(inst)
    path = _file(inst, *_key_of(other.pid))
    assert inst.pids() == {other.pid}
    os.chmod(path, 0)
    _cannot_be_read(inst, other.pid)


@pytest.mark.skipif(sys.platform == "win32", reason="owners")
def test_a_file_of_another_user_is_neither_believed_nor_deleted(make, monkeypatch):
    inst = make()
    live = _file(inst, os.getpid(), _me())
    gone = _file(inst, os.getpid(), procs._text(1))
    os.chmod(live, 0)
    place = inst.place()
    with monkeypatch.context() as m:
        m.setattr(procs, "_folder", lambda install: (str(inst.folder), {os.getuid() + 1}))
        assert inst.list() == []
        procs._sweep(*place)
    os.chmod(live, 0o600)
    assert live.exists() and gone.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="owners")
def test_a_record_of_another_user_is_not_used(make, monkeypatch):
    inst = make()
    path = _file(inst, os.getpid(), procs._text(1))
    place = inst.place()
    with monkeypatch.context() as m:
        m.setattr(os, "geteuid", lambda: os.getuid() + 1)
        with pytest.raises(OSError, match="another user"):
            inst.list()
        with pytest.raises(OSError, match="another user"):
            procs._write(place, procs.Entry(os.getpid(), _me(), "gui"))
    assert inst.files() == [path.name]


@pytest.mark.skipif(sys.platform == "win32", reason="owners")
def test_root_uses_the_record_of_the_owner(make, monkeypatch):
    inst = make()
    _file(inst, os.getpid(), _me())
    with monkeypatch.context() as m:
        m.setattr(os, "geteuid", lambda: 0)
        assert procs._folder(str(inst.root))[1] == {0, os.stat(inst.root).st_uid}
        assert os.getpid() in inst.pids()


def _link(to: Path, at: Path) -> None:
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(to), str(at))
    else:
        at.symlink_to(to, target_is_directory=True)


@pytest.mark.parametrize("kind", ["link", "broken link", "file"])
def test_a_record_that_is_not_a_folder_is_not_used(make, tmp_path, kind):
    inst = make()
    here = inst.place()[1]
    inst.folder.rmdir()
    to = tmp_path / "documents"
    to.mkdir()
    for name in ("letter.txt", "thesis.odt", f"{here}-{os.getpid()}-{procs._text(1)}.proc"):
        (to / name).write_text(f"{name[:-5]}\ngui\n\n", newline="")
    before = {p.name: p.read_bytes() for p in to.iterdir()}
    if kind == "file":
        inst.folder.write_text("in the way", encoding="utf-8")
    else:
        _link(to, inst.folder)
    if kind == "broken link":
        to.rename(tmp_path / "moved")
    p = inst.run("import localm; from localm import procs\n"
                 "print(procs.current(), procs.problem(), flush=True)\n"
                 "import sys; procs.start([sys.executable, '-c', ''], purpose='tool').wait()")
    assert p.first == "None .localm-run is a link or not a folder"
    assert p.communicate(timeout=20) == ("", "")
    for look in (inst.list, lambda: procs.end_others(inst.root, timeout=0)):
        with pytest.raises(OSError, match="is a link or not a folder"):
            look()
    if kind == "broken link":
        (tmp_path / "moved").rename(to)
    assert {p.name: p.read_bytes() for p in to.iterdir()} == before
    if kind != "file":
        (os.rmdir if sys.platform == "win32" else os.unlink)(inst.folder)


def test_a_file_that_is_a_link_is_not_believed(make, tmp_path):
    inst = make()
    path = _file(inst, os.getpid(), _me())
    assert os.getpid() in inst.pids()
    path.rename(tmp_path / "elsewhere")
    try:
        path.symlink_to(tmp_path / "elsewhere")
    except OSError:
        pytest.skip("this account may not make symbolic links")
    gone = inst.folder / procs._name(inst.place()[1], os.getpid(), procs._text(1))
    gone.symlink_to(tmp_path / "elsewhere")
    assert inst.list() == []
    procs._sweep(*inst.place())
    assert path.is_symlink() and gone.is_symlink() and (tmp_path / "elsewhere").exists()


def test_a_hard_link_to_the_file_of_another_install_is_not_believed(make):
    a, b = make("a"), make("b")
    pid = int(a.importer().first)
    (name,) = a.files()
    os.link(a.folder / name, b.folder.parent / "x")
    here = b.place()[1]
    (b.root / "x").rename(b.folder / (here + name[16:]))
    assert b.list() == [] and len(b.files()) == 1
    assert pid in a.pids()


def test_start_does_not_replace_the_file_a_process_wrote_itself(make):
    inst = make()
    path = _file(inst, os.getpid(), _me(), text="")
    procs._write(inst.place(), procs.Entry(os.getpid(), _me(), "launcher window"), replace=False)
    assert [e.purpose for e in inst.list() if e.recorded] == ["gui"]
    procs._write(inst.place(), procs.Entry(os.getpid(), _me(), "run"))
    assert [e.purpose for e in inst.list() if e.recorded] == ["run"] and inst.files() == [path.name]


def test_a_file_that_cannot_be_written_leaves_nothing(make, monkeypatch):
    inst = make()
    place = inst.place()

    def full(*a):
        raise OSError(28, "no space left")
    with monkeypatch.context() as m:
        m.setattr(os, "write", full)
        with pytest.raises(OSError):
            procs._write(place, procs.Entry(os.getpid(), _me(), "gui"))
    assert inst.files() == []


def test_an_install_at_the_top_of_a_drive_has_not_every_program_in_it(tmp_path):
    top = tmp_path.anchor
    assert not procs._within(os.path.join(top, "Programs", "browser.exe"), top)
    assert procs._within(str(tmp_path / "a" / "b"), str(tmp_path / "a"))
    assert not procs._within(str(tmp_path / "ab"), str(tmp_path / "a"))


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
left = procs.end_others(timeout=30)
print(left, flush=True)
print("listed", *(e.pid for e in procs.list()), flush=True)
sys.exit(7)
"""


def test_end_others_spares_every_localm_process_that_waits_for_it(make):
    inst = make()
    other = int(inst.importer().first)
    (inst.root / "chain.py").write_text(CHAIN, encoding="utf-8")
    p = inst.popen([sys.executable, str(inst.root / "chain.py"), "2"])
    out = p.communicate(timeout=90)[0].splitlines()
    assert p.returncode == 7 and out[1] == "[]"
    assert {p.first, out[0]} <= set(out[2].split()) and str(other) not in out[2].split()
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
