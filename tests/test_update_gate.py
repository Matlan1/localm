# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every way to start localm while an update of it is not finished: the old version is put
back, or a running update is waited for, before anything else of localm runs.

Each test uses a small install of its own: the real gate, process record, update mechanism
and launch scripts, and a ``localm`` that prints what it was started as."""

from __future__ import annotations

import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import localm
from localm import install_manifest, procs
from localm.update_mechanism import disk, job, mover, restoring, shared
from tests import _update_mechanism as um
from tests._fake_clone import CREATE_NO_WINDOW, make_windows_clone, venv_launcher

REPO = Path(__file__).resolve().parents[1]
ID = um.UPDATE_ID
nt_only = pytest.mark.skipif(os.name != "nt", reason="cmd.exe only")
posix_only = pytest.mark.skipif(os.name == "nt", reason="bash launch scripts")
HIDDEN = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}

_MODULES = ["__init__", "procs", "_home_cfg", *shared.MODULES]
_LAUNCHERS = ("localm.bat", "localm.sh", "localm-launcher.bat", "localm-launcher.sh", "rollback.bat",
              "rollback.sh", "scripts/rollback_update.py", "scripts/update_gate.bat",
              "scripts/update_gate.sh", "scripts/update_gate.pyw")
_MAIN = """\
import json, os, sys
import localm
from localm import part
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION')) as f:
    version = f.read()
said = {'localm': localm.__version__, 'part': part.V, 'VERSION': version, 'argv': sys.argv[1:],
        'subprocess': 'subprocess' in sys.modules}
if __name__ == '__main__':
    print('STARTED ' + json.dumps(said), flush=True)
"""


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _release(version: str) -> dict:
    """The files of a small localm of *version*."""
    files = {f"localm/{n}.py": _text(f"localm/{n}.py") for n in _MODULES}
    files.update({f"localm/update_mechanism/{p.name}": p.read_text(encoding="utf-8")
                  for p in (REPO / "localm" / "update_mechanism").glob("*.py")})
    files.update({rel: _text(rel) for rel in _LAUNCHERS})
    files["localm/__init__.py"] = re.sub(r'__version__ = ".*"', f'__version__ = "{version}"',
                                         files["localm/__init__.py"])
    files["localm/procs.py"] += f"\nBUILD = {version!r}\n"
    for name in ("localm.bat", "localm.sh"):
        files[name] += f"\nrem {version}\n" if name.endswith(".bat") else f"\n# {version}\n"
    files.update({"localm/__main__.py": _MAIN, "localm/part.py": f"V = {version!r}\n",
                  f"localm/only{version}.py": "", "VERSION": version, "pyproject.toml": ""})
    if os.name == "nt":
        files = {rel: text.replace("\n", "\r\n") if rel.endswith(".bat") else text
                 for rel, text in files.items()}
    return files


OLD, NEW = _release("1"), _release("2")


def _statement(files: dict, version: str) -> str:
    return json.dumps({"format": shared.update_package.FORMAT, "version": version,
                       "files": um.stated(files), "components": {}})


class Install:
    """A small localm 1 at *root* with an update to 2 prepared."""

    def __init__(self, root: Path, *, applier: bool = True) -> None:
        self.root = root
        root.mkdir(exist_ok=True)
        um.write(root, {**OLD, "update-package.json": _statement(OLD, "1"), "home/mine.txt": "m"})
        for name in ("localm.sh", "rollback.sh"):
            (root / name).chmod(0o755)
        self.old = um.tree(root)
        um.write(root / ".localm-update" / ID / "new",
                 {**NEW, "update-package.json": _statement(NEW, "2")})
        if applier:
            um.no_window(shared.copy_applier(REPO / "localm",
                                             root / ".localm-update" / ID / "applier"))

    @property
    def record(self) -> Path:
        return self.root / ".localm-update" / "job.json"

    def go(self, phase: str) -> None:
        """Make the update as far as *phase*, and leave it there."""
        with job.Lock(self.root) as lock:
            self.start(lock, phase)

    def start(self, lock, phase: str) -> None:
        job.start(lock, ID, old="1", new="2")
        if phase == job.ENDING:
            return
        job.advance(lock, job.MOVING)
        plan = mover.plan_files(self.root, ID, um.stated(NEW), um.stated(OLD), protect=["home"])
        mover.write_plan(self.root, ID, plan)
        mover.forward(self.root, ID, plan)
        for later in (job.STEPS, job.STARTING):
            if phase in (job.STEPS, job.STARTING, job.DONE) and not (
                    phase == job.STEPS and later == job.STARTING):
                job.advance(lock, later)
        if phase == job.DONE:
            record = job.current(self.root)
            job._set_over(lock, record, job.DONE)
            job._set_phase(lock, record, job.DONE)

    def damage(self) -> None:
        """Make it an update that cannot be told, over files of two versions."""
        self.go(job.STEPS)
        work = self.root / ".localm-update" / ID
        for path in (self.record, work / "state.json", *(work / "plans").iterdir()):
            path.write_text("{")
        (self.root / "localm" / "part.py").write_text("V = 'mixed'")

    def env(self, **more) -> dict:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LOCALM_") and k not in (
                   "DISPLAY", "WAYLAND_DISPLAY", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE")}
        return {**env, "PYTHONPATH": str(self.root), "PYTHONNOUSERSITE": "1",
                "LOCALM_HOME": str(self.root / "home"), **more}

    def popen(self, *args, **more) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, *args], cwd=str(self.root), text=True,
                                env=self.env(**more), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **HIDDEN)

    def run(self, *args, **more):
        proc = self.popen(*args, **more)
        out, err = proc.communicate(timeout=120)
        return types.SimpleNamespace(code=proc.returncode, out=out, err=err,
                                     said=said(out))

    def localm(self, *args, **more):
        return self.run("-m", "localm", *args, **more)

    def gate(self, **more):
        """The gate as the launchers and setup run it."""
        return self.run("-I", "-S", str(self.root / "localm" / "__init__.py"), **more)

    def phase(self) -> str:
        return json.loads(self.record.read_text())["phase"] if self.record.exists() else ""

    def lose(self, rel: str) -> None:
        """Delete the old file *rel* the update kept."""
        (self.root / ".localm-update" / ID / "old" / "01" / rel).unlink()

    def is_old(self) -> bool:
        return um.tree(self.root) == self.old and not self.record.exists()


def said(out: str):
    line = next((ln for ln in out.splitlines() if ln.startswith("STARTED ")), None)
    return json.loads(line[8:]) if line else None


def ran(done, version: str) -> bool:
    """Whether localm ran, all of it of *version*."""
    return bool(done.said) and {done.said[k] for k in ("localm", "part", "VERSION")} == {version}


@pytest.fixture
def inst(tmp_path):
    return Install(tmp_path / "inst")


# ------------------------------------------------------------------ undone

def test_without_a_record_the_gate_does_nothing(inst):
    done = inst.localm()
    assert ran(done, "1") and done.err == "" and not done.said["subprocess"]


def test_a_process_record_that_cannot_be_imported_stops_the_start(inst):
    """Only a ``localm/procs.py`` that is not there, or one without ``enter`` and
    ``current`` (an older release's, put in place by an update), lets the gate go on
    without the process record; any other failure to import it stops the start, loudly."""
    procs = inst.root / "localm" / "procs.py"
    procs.write_text("import localm_no_such_module_for_the_gate\n" + procs.read_text(encoding="utf-8"),
                     encoding="utf-8")
    done = inst.localm()
    assert done.code != 0 and not done.said, done.out
    assert "ModuleNotFoundError" in done.err and "localm_no_such_module_for_the_gate" in done.err


@pytest.mark.parametrize("phase", [job.ENDING, job.MOVING, job.STEPS, job.STARTING])
def test_an_interrupted_update_is_undone_before_localm_runs(inst, phase):
    inst.go(phase)
    done = inst.localm("status")
    assert (done.code, done.said["argv"]) == (0, ["status"]) and ran(done, "1"), done.err
    assert inst.is_old()
    assert "Putting the previous version back" in done.err
    assert ("The update to 2 was undone" in done.err) == (phase != job.ENDING)


def test_an_update_that_reached_done_stays(inst):
    inst.go(job.DONE)
    done = inst.localm()
    assert done.code == 0 and ran(done, "2"), done.err
    assert not inst.record.exists() and um.tree(inst.root) != inst.old


def test_killed_at_many_points_the_next_start_runs_one_whole_version(tmp_path):
    def spec(root):
        return {"do": "update", "root": str(root), "id": ID, "swap": list(mover.GATE_FILES),
                "new_files": um.stated(NEW), "old_files": um.stated(OLD)}
    whole = Install(tmp_path / "count")
    run_dir = whole.root / ".localm-update" / ID / "applier" / "update_mechanism"
    counted = um.drive(shared.copy_applier(REPO / "localm", tmp_path / "applier"), spec(whole.root), -1)
    points = int(re.search(r"POINTS (\d+)", counted.stdout).group(1))
    new = um.tree(whole.root)
    assert points > 60
    for n in range(0, points, points // 9):
        one = Install(tmp_path / f"k{n}")
        run_dir = one.root / ".localm-update" / ID / "applier" / "update_mechanism"
        assert um.drive(run_dir, spec(one.root), n).returncode != 0
        done = one.localm()
        assert done.code == 0, (n, done.err)
        assert (ran(done, "1") and um.tree(one.root) == one.old) or (
            ran(done, "2") and um.tree(one.root) == new), (n, done.out, done.err)
        assert not one.record.exists(), n


def test_the_process_that_undid_it_runs_the_old_gate_and_process_record(inst):
    """Also when the files of both versions have the same size and time."""
    inst.go(job.STARTING)
    for path in (inst.root / ".localm-update").rglob("*.py"):
        os.utime(path, (1.7e9, 1.7e9))
    for path in (inst.root / "localm").glob("*.py"):
        os.utime(path, (1.7e9, 1.7e9))
    done = inst.run("-c", "import localm, localm.procs as p; print(localm.__version__, p.BUILD,"
                          " p.current().pid > 0)")
    assert done.out.split() == ["1", "1", "True"], (done.out, done.err)


# ---------------------------------------------------------------- ways in

def test_a_script_that_imports_localm(inst):
    inst.go(job.STEPS)
    done = inst.run("-c", "import localm.part; print(localm.part.V)")
    assert done.out.strip() == "1" and inst.is_old(), done.err


def test_a_spawn_worker(inst):
    inst.go(job.STEPS)
    (inst.root.parent / "w.py").write_text(
        "import multiprocessing as mp, importlib\n"
        "if __name__ == '__main__':\n"
        "    with mp.get_context('spawn').Pool(1) as pool:\n"
        "        print(pool.map(importlib.import_module, [])),\n"
        "    p = mp.get_context('spawn').Process(target=importlib.import_module, args=('localm',))\n"
        "    p.start(); p.join(); print('WORKER', p.exitcode)\n", encoding="utf-8")
    done = inst.run(str(inst.root.parent / "w.py"))
    assert "WORKER 0" in done.out and inst.is_old(), (done.out, done.err)


def test_an_mcp_server_says_everything_on_stderr(inst):
    inst.go(job.STARTING)
    done = inst.localm("mcp")
    assert ran(done, "1") and len(done.out.splitlines()) == 1 and "undone" in done.err


def test_an_mcp_server_does_not_wait_without_end(inst, monkeypatch):
    path = inst.root / "localm" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    inst.go(job.ENDING)
    path.write_text(text.replace("_MCP_WAIT_S = 3600.0, 60.0", "_MCP_WAIT_S = 3600.0, 1.0"),
                    encoding="utf-8")
    with job.Lock(inst.root):
        done = inst.localm("mcp")
    assert done.code == 1 and done.said is None
    assert "another process is updating" in done.err


def test_a_start_outside_the_install_folder(inst, tmp_path):
    inst.go(job.STEPS)
    proc = subprocess.run([sys.executable, "-m", "localm"], cwd=str(tmp_path), text=True,
                          env=inst.env(), stdin=subprocess.DEVNULL, capture_output=True, **HIDDEN)
    assert said(proc.stdout)["part"] == "1" and inst.is_old(), proc.stderr


def test_a_moved_install(inst, tmp_path):
    inst.go(job.STARTING)
    moved = tmp_path / "moved to ünd elsewhere"
    inst.root.rename(moved)
    inst.root = moved
    done = inst.localm()
    assert ran(done, "1") and inst.is_old(), done.err


def test_the_data_folder_of_localm_home_cfg_is_found(inst, tmp_path, monkeypatch):
    home = tmp_path / "elsewhere"
    home.mkdir()
    (home / "config.json").write_text('{"net_mode": "off"}', encoding="utf-8")
    for name in ("LOCALM_HOME", "LOCALM_NET_MODE", "LOCALM_SETUP"):
        monkeypatch.delenv(name, raising=False)
    assert restoring.network(inst.root)
    (inst.root / "localm-home.cfg").write_text(str(home), encoding="utf-8")
    assert restoring.data_folder(inst.root) == home and not restoring.network(inst.root)
    monkeypatch.setenv("LOCALM_SETUP", "1")
    assert restoring.network(inst.root)


@nt_only
@pytest.mark.parametrize("folder", ["install dir", "a&b'c!d%e^f (g)", "pct%y%PATH%"])
def test_localm_bat(tmp_path, folder):
    clone = make_windows_clone(tmp_path, "localm.bat")
    inst = Install(clone.rename(clone.with_name(folder)))
    inst.go(job.STEPS)
    done = subprocess.run(["cmd", "/d", "/c", r".\localm.bat", "run", "a (b) & c", "!x!"],
                          env=inst.env(), cwd=str(inst.root), text=True, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=120, **HIDDEN)
    assert said(done.stdout)["argv"] == ["run", "a (b) & c", "!x!"], (done.stdout, done.stderr)
    assert done.returncode == 0 and said(done.stdout)["localm"] == "1" and inst.is_old()


@nt_only
def test_localm_bat_says_why_it_does_not_start(tmp_path):
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    inst.damage()
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "localm.bat"), "status"],
                          env=inst.env(), cwd=str(tmp_path), text=True, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=120, **HIDDEN)
    assert done.returncode == 1 and said(done.stdout) is None
    assert "could not be undone" in done.stderr and "press any key" in done.stdout.lower()


@nt_only
@pytest.mark.parametrize("where", [".python/cpython-3.12-x", "home of pyvenv.cfg"])
def test_update_gate_bat_with_a_venv_that_cannot_start(tmp_path, where):
    inst = Install(tmp_path / "inst")
    inst.go(job.STEPS)
    base = Path(sys.base_prefix)
    venv = inst.root / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    if where.startswith("."):
        shutil.copyfile(venv_launcher(), venv / "python.exe")
        link = inst.root / ".python" / "cpython-3.12-x"
        link.parent.mkdir()
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(base)], check=True,
                       capture_output=True, **HIDDEN)
    else:
        (inst.root / ".venv" / "pyvenv.cfg").write_text(f"home = {base}\n", encoding="utf-8")
    env = inst.env(PATH=os.environ["SystemRoot"] + "\\System32")
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "scripts" / "update_gate.bat")], env=env,
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)
    if where.startswith("."):
        os.rmdir(link)
    assert done.returncode == 0 and not inst.record.exists(), done.stderr
    assert (inst.root / "VERSION").read_text() == "1"


@nt_only
def test_update_gate_bat_takes_no_python_from_path(tmp_path):
    inst = Install(tmp_path / "inst")
    inst.go(job.STEPS)
    env = inst.env(PATH=os.pathsep.join([sys.base_prefix, os.environ["PATH"]]))
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "scripts" / "update_gate.bat")], env=env,
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)
    assert done.returncode == 4 and inst.record.exists() and "run setup" in done.stderr


@nt_only
def test_update_gate_bat_says_it_in_a_window_for_a_program_without_a_console(tmp_path):
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    (inst.root / "scripts" / "update_gate.pyw").write_text("print('WINDOW'); raise SystemExit(1)")
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "scripts" / "update_gate.bat"),
                           "window"], env=inst.env(), text=True, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=120, **HIDDEN)
    assert (done.returncode, done.stdout.strip()) == (1, "WINDOW")
    assert "if \"%USEBOX%\"==\"1\" set \"GATEIN=window\"" in _text("setup-gui.bat")


@nt_only
@pytest.mark.parametrize("args,held", [(["--yes"], False), ([], True)])
def test_rollback_bat_undoes_an_unfinished_update(tmp_path, args, held):
    inst = Install(make_windows_clone(tmp_path, "rollback.bat"))
    inst.go(job.STEPS)
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "rollback.bat"), *args], env=inst.env(),
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)
    assert done.returncode == 0 and inst.is_old(), (done.stdout, done.stderr)
    assert ("press any key" in done.stdout.lower()) == held


@nt_only
def test_rollback_bat_with_yes_ends_with_the_real_exit_code_and_no_key_press(tmp_path):
    inst = Install(make_windows_clone(tmp_path, "rollback.bat"))
    done = subprocess.run(["cmd", "/d", "/c", "call", str(inst.root / "rollback.bat"), "--yes"], env=inst.env(),
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)
    assert done.returncode == 1 and "No update backup" in done.stderr
    assert "press any key" not in done.stdout.lower()


def test_rollback_update_py(inst):
    inst.go(job.STARTING)
    done = inst.run(str(inst.root / "scripts" / "rollback_update.py"), "--yes")
    assert done.code == 0 and inst.is_old(), done.err


@nt_only
@pytest.mark.parametrize("script,until", [
    ("setup.bat", "setlocal EnableDelayedExpansion"),
    ("setup-gui.bat", 'set "LOCALM_SETUP_GUI_HIDDEN="'),
    ("localm-launcher.bat", 'if exist ".venv\\Scripts\\pythonw.exe"')])
def test_a_batch_script_undoes_it_first_and_starts_anew(tmp_path, script, until):
    """The script's own head, up to where its work begins."""
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    head = _text(script).split(until)[0] + "echo BEGINS %LOCALM_GATE%.\n"
    (inst.root / script).write_bytes(head.replace("\n", "\r\n").encode())
    inst.old = um.tree(inst.root)
    inst.go(job.STEPS)
    cmd = ["cmd", "/d", "/c", "call", str(inst.root / script), "an arg"]
    kw = {"env": inst.env(LOCALM_SETUP_GUI_INPLACE="1"), "text": True, "stdin": subprocess.DEVNULL,
          "capture_output": True, "timeout": 120, **HIDDEN}
    done = subprocess.run(cmd, **kw)
    assert inst.is_old() and done.stdout.count("BEGINS .") == 1, (done.stdout, done.stderr)
    um.write(inst.root / ".localm-update" / ID / "new", NEW)
    inst.damage()
    done = subprocess.run(cmd, **kw)
    assert done.returncode == 1 and "BEGINS" not in done.stdout, (done.stdout, done.stderr)


@posix_only
@pytest.mark.parametrize("script,until", [
    ("setup.sh", "say() {"), ("setup-gui.sh", "# The uv release"),
    ("localm.sh", "if [ ! -x"), ("localm-launcher.sh", "if [ ! -x"), ("rollback.sh", "# Prefer the")])
def test_a_shell_script_undoes_it_first(tmp_path, script, until):
    inst = Install(tmp_path / "inst")
    binaries = inst.root / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "python").symlink_to(Path(sys.base_prefix) / "bin" / "python3")
    (inst.root / script).write_text(_text(script).split(until)[0] + "echo BEGINS ${LOCALM_GATE:-}.\n")
    inst.old = um.tree(inst.root)
    inst.go(job.STEPS)
    cmd = ["bash", str(inst.root / script)]
    kw = {"env": inst.env(), "text": True, "stdin": subprocess.DEVNULL, "capture_output": True,
          "timeout": 120}
    done = subprocess.run(cmd, **kw)
    assert inst.is_old(), (done.stdout, done.stderr)
    assert done.stdout.count("BEGINS .") == (script != "rollback.sh")
    um.write(inst.root / ".localm-update" / ID / "new", NEW)
    inst.damage()
    done = subprocess.run(cmd, **kw)
    assert done.returncode == 1 and "BEGINS" not in done.stdout, (done.stdout, done.stderr)


@pytest.mark.parametrize("script", ["launcher.pyw", "installer/gui.py"])
def test_a_program_with_windows_undoes_it_before_it_imports_localm(inst, script, monkeypatch):
    """The program's own head; the window is a fake one (update_gate.pyw is tested below)."""
    head = _text(script)
    head = head[:head.index("    _update_gate()\n") + len("    _update_gate()\n")] + (
        "open('BEGINS', 'a').write('x')\n")
    path = inst.root / script
    path.parent.mkdir(exist_ok=True)
    path.write_text(head, encoding="utf-8")
    begins = inst.root / "BEGINS"
    done = inst.run(str(path))
    assert done.code == 0 and begins.read_text() == "x", done.err
    inst.go(job.STEPS)
    window = inst.root / "scripts" / "update_gate.pyw"
    window.write_text(
        "import subprocess, sys\nsys.exit(subprocess.call([sys.executable, '-I', '-S', "
        "sys.argv[0].replace('scripts', 'localm').replace('update_gate.pyw', '__init__.py')]))\n")
    done = inst.run(str(path))
    assert done.code == 0 and begins.read_text() == "xx", done.err
    assert not inst.record.exists() and (inst.root / "VERSION").read_text() == "1"
    window.write_text("import sys; sys.exit(1)")
    inst.record.write_text("{")
    done = inst.run(str(path))
    assert done.code == 1 and begins.read_text() == "xx"


def test_the_update_window_shows_why_and_tries_again(inst, monkeypatch):
    shown = []

    class Window:
        def __init__(self, *_a, **_k): pass
        def __getattr__(self, _name): return lambda *a, **k: None

    box = types.ModuleType("tkinter.messagebox")
    box.showinfo = lambda title, text, **k: shown.append(("info", text))

    def ask(_title, text, **_k):
        shown.append(("ask", text))
        um.write(inst.root, {"localm/part.py": NEW["localm/part.py"]})
        return True
    box.askretrycancel = ask
    tk = types.ModuleType("tkinter")
    tk.Tk = tk.StringVar = tk.Label = Window
    tk.messagebox = box
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", box)
    inst.damage()
    main = runpy.run_path(str(inst.root / "scripts" / "update_gate.pyw"))["main"]
    assert main() == 0
    assert [kind for kind, _ in shown] == ["ask", "info"], shown
    assert "could not be undone" in shown[0][1] and "put aside" in shown[1][1]


def test_the_update_window_can_be_closed(inst, monkeypatch):
    close: list = []

    class Window:
        def __init__(self, *_a, **_k): pass
        def protocol(self, _name, call): close.append(call)
        def update(self): [call() for call in close]
        def __getattr__(self, _name): return lambda *a, **k: None

    tk = types.ModuleType("tkinter")
    tk.Tk = tk.StringVar = tk.Label = Window
    tk.messagebox = types.ModuleType("tkinter.messagebox")
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk.messagebox)
    inst.go(job.ENDING)
    with job.Lock(inst.root):
        begun = time.monotonic()
        assert runpy.run_path(str(inst.root / "scripts" / "update_gate.pyw"))["main"]() == 1
        assert time.monotonic() - begun < 30


@posix_only
@pytest.mark.parametrize("display", [True, False])
@pytest.mark.parametrize("way", ["localm gui", "update_gate.sh window"])
def test_a_start_from_a_menu_says_it_in_a_window(inst, way, display):
    binaries = inst.root / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "python").symlink_to(Path(sys.base_prefix) / "bin" / "python3")
    inst.go(job.STEPS)
    (inst.root / "scripts" / "update_gate.pyw").write_text(
        "import os, pathlib, sys\np = pathlib.Path(sys.argv[0]).parents[1] / '.localm-update'\n"
        "(p / 'WINDOW').write_text(' '.join(n.name for n in p.glob('waiting-*')))\nsys.exit(1)")
    cmd = ([sys.executable, "-m", "localm", "gui"] if way == "localm gui" else
           ["bash", str(inst.root / "scripts" / "update_gate.sh"), "window"])
    done = subprocess.run(cmd, cwd=str(inst.root), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=120, env=inst.env(**({"DISPLAY": ":9"} if display else {})))
    window = inst.root / ".localm-update" / "WINDOW"
    assert (done.returncode, window.exists()) == ((1, True) if display else (0, False)), done.stderr
    if display and way == "localm gui":
        assert window.read_text().startswith("waiting-")
    for script in ("localm-launcher.sh", "setup-gui.sh"):
        assert 'scripts/update_gate.sh" window' in _text(script)


# ------------------------------------------------------- a running update

def _waits(proc, inst) -> str:
    until = time.monotonic() + 30
    while time.monotonic() < until and proc.poll() is None:
        marks = list((inst.root / ".localm-update").glob("waiting-*"))
        if marks:
            return marks[0].name
        time.sleep(0.05)
    raise AssertionError(proc.communicate(timeout=30))


def test_a_start_waits_for_a_running_update_and_then_runs_its_result(inst):
    with job.Lock(inst.root) as lock:
        inst.start(lock, job.STARTING)
        proc = inst.popen("-m", "localm")
        mark = _waits(proc, inst)
        time.sleep(0.5)
        assert proc.poll() is None
        job.finish(lock)
    out, err = proc.communicate(timeout=60)
    assert said(out) and {said(out)[k] for k in ("localm", "part", "VERSION")} == {"2"}, err
    assert err.count("Waiting") == 1 and not (inst.root / ".localm-update" / mark).exists()


def test_a_start_that_waits_undoes_the_update_when_its_process_ends(inst):
    with job.Lock(inst.root) as lock:
        inst.start(lock, job.STEPS)
        procs_ = [inst.popen("-m", "localm") for _ in range(2)]
        for proc in procs_:
            _waits(proc, inst)
    for proc in procs_:
        out, err = proc.communicate(timeout=60)
        assert said(out) and said(out)["localm"] == "1", err
    assert inst.is_old()


@pytest.mark.parametrize("phase,through", [(job.ENDING, False), (job.MOVING, False), (job.STEPS, True),
                                           (job.STARTING, True)])
def test_a_child_of_the_update_passes_in_its_steps_only(inst, phase, through):
    with job.Lock(inst.root) as lock:
        inst.start(lock, phase)
        proc = inst.popen("-m", "localm", LOCALM_UPDATE_ID=ID)
        if through:
            out, err = proc.communicate(timeout=60)
            assert said(out)["part"] == "2", err
        else:
            _waits(proc, inst)
        stranger = inst.popen("-m", "localm", LOCALM_UPDATE_ID="f" * 32)
        _waits(stranger, inst)
    for p in (proc, stranger):
        p.communicate(timeout=60)


def test_a_child_of_an_update_whose_process_ended_does_not_run_the_mix(inst):
    inst.go(job.STEPS)
    done = inst.localm(LOCALM_UPDATE_ID=ID)
    assert ran(done, "1") and inst.is_old()


def test_the_chosen_process_and_what_it_starts_pass_once_the_update_starts_it(inst):
    with job.Lock(inst.root) as lock:
        inst.start(lock, job.STEPS)
        proc = inst.popen("-c", "import localm, os, subprocess, sys\n"
                                "print(localm.__version__, flush=True)\n"
                                "print(subprocess.run([sys.executable, '-c', 'import localm; "
                                "print(localm.__version__)'], capture_output=True, text=True).stdout)")
        mark = _waits(proc, inst)
        other = inst.popen("-m", "localm")
        (inst.root / ".localm-update" / "started").write_text(f"{ID} {mark.removeprefix('waiting-')}")
        time.sleep(0.6)
        assert proc.poll() is None
        job.advance(lock, job.STARTING)
        out, err = proc.communicate(timeout=60)
        assert out.split() == ["2", "2"], (out, err)
        assert other.poll() is None
        job.finish(lock)
    assert said(other.communicate(timeout=60)[0])["part"] == "2"


# ---------------------------------------------------- what still runs

@pytest.mark.parametrize("phase,ended", [(job.ENDING, False), (job.STARTING, True)])
def test_what_an_interrupted_update_started_is_ended_and_a_waiting_start_is_not(inst, phase, ended):
    left = inst.popen("-c", "import localm, time; print('UP', flush=True); time.sleep(60)")
    assert left.stdout.readline().strip() == "UP"
    try:
        with job.Lock(inst.root) as lock:
            inst.start(lock, phase)
            waiting = inst.popen("-m", "localm")
            _waits(waiting, inst)
        out, err = waiting.communicate(timeout=60)
        assert said(out) and said(out)["part"] == "1", err
        assert (left.poll() is not None) == ended
    finally:
        left.kill()
        left.communicate()


#: The way back's ending of what still runs (``restoring._end_running``) in a process of
#: its own, beside the one it ends, on a clock that moves 1.1 s at each reading: a process
#: first seen as arriving is past its arrival time at the next reading.
_SLOW_PASS = """\
import json, sys, time, types
from localm.update_mechanism import job, restoring
now = [time.monotonic()]

def monotonic():
    now[0] += 1.1
    return now[0]

restoring.time = types.SimpleNamespace(monotonic=monotonic, sleep=time.sleep)
with job.Lock(sys.argv[1]) as lock:
    print(json.dumps(restoring._end_running(lock)), flush=True)
"""


def test_a_process_spared_as_arriving_is_ended_however_long_the_pass_took(inst):
    """A process of localm first seen while the way back runs is spared for its arrival
    time; when one pass of ending and listing takes longer than that, it is still ended
    in the next pass, not reported as one that still runs."""
    left = inst.popen("-c", "import localm, time; print('UP', flush=True); time.sleep(60)")
    assert left.stdout.readline().strip() == "UP"
    try:
        with job.Lock(inst.root) as lock:
            inst.start(lock, job.STARTING)
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALM_")}
        done = subprocess.run(
            [sys.executable, "-c", _SLOW_PASS, str(inst.root)], cwd=str(REPO), text=True,
            capture_output=True, timeout=120, stdin=subprocess.DEVNULL, **HIDDEN,
            env={**env, "PYTHONPATH": str(REPO), "LOCALM_HOME": str(inst.root / "home")})
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout.splitlines()[-1]) == [], done.stdout
        left.wait(timeout=30)
    finally:
        left.kill()
        left.communicate()


def test_what_still_runs_is_named(inst, monkeypatch):
    inst.go(job.STEPS)
    entry = procs.Entry(4711, "1", "gui")
    entry.in_the_way = ({"path": "x.exe", "pid": 4711, "name": "x"},)
    fake = types.SimpleNamespace(list=lambda *a, **k: [], end_others=lambda *a, **k: [entry])
    monkeypatch.setattr(shared, "also", lambda name: fake)
    code, lines = restoring._once(inst.root)
    assert code == restoring.FAILED and "pid 4711" in lines[0] and inst.record.exists()


def test_a_process_above_that_runs_the_new_version_is_named(inst):
    """Nothing is put back below a process of localm that runs on."""
    with job.Lock(inst.root) as lock:
        inst.start(lock, job.STARTING)
        gui = subprocess.Popen(
            [sys.executable, "-c", "import localm, subprocess, sys\nprint(localm.__version__, flush=True)\n"
             "sys.stdin.readline()\nprint(subprocess.run([sys.executable, '-c', 'import localm'], "
             "stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True))"],
            cwd=str(inst.root), text=True, env=inst.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, **HIDDEN)
        mark = _waits(gui, inst).removeprefix("waiting-")
        (inst.root / ".localm-update" / "started").write_text(f"{ID} {mark}")
        assert gui.stdout.readline().strip() == "2"
    before = um.tree(inst.root)
    out, err = gui.communicate("go\n", timeout=120)
    assert "returncode=1" in out and f"pid {mark.split('-')[0]}" in out, (out, err)
    assert um.tree(inst.root) == before and inst.record.exists()
    assert ran(inst.localm(), "1") and inst.is_old()


def test_a_program_the_waiting_process_started_is_the_users(inst):
    inst.go(job.STEPS)
    done = inst.run("-c", "import subprocess, sys\n"
                    "p = subprocess.Popen([sys.executable, '-I', '-c', 'input()'], stdin=subprocess.PIPE)\n"
                    "import localm, time\ntime.sleep(0.3)\nprint(localm.__version__, p.poll())\n"
                    "p.communicate(b'x')")
    assert done.out.split() == ["1", "None"] and inst.is_old(), (done.out, done.err)


# ------------------------------------------- a way back that cannot be gone

THREE = {**_release("3"), "localm/update_mechanism/new_in_3.py": ""}


@pytest.mark.parametrize("phase", [job.MOVING, job.STEPS, job.STARTING])
def test_a_release_put_over_an_interrupted_update_is_the_install(inst, phase):
    inst.go(phase)
    um.write(inst.root, {**THREE, "update-package.json": _statement(THREE, "3")})
    before = um.tree(inst.root)
    done = inst.localm()
    assert done.code == 0 and ran(done, "3") and "put aside" in done.err, done.err
    assert um.tree(inst.root) == before and not inst.record.exists()
    with job.Lock(inst.root) as lock:
        assert job.restore(lock).state == "none"


@pytest.mark.parametrize("setup", [True, False])
@pytest.mark.parametrize("listed", [True, False])
def test_files_of_another_release_that_are_not_all_of_it(inst, listed, setup):
    inst.go(job.STEPS)
    um.write(inst.root, THREE)
    if listed:
        um.write(inst.root, {"update-package.json": _statement(THREE, "3"), "localm/part.py": "V = 'x'"})
    before = um.tree(inst.root)
    done = inst.gate(**({"LOCALM_SETUP": "1"} if setup else {}))
    assert done.code == 1 and um.tree(inst.root) == before and inst.phase() == job.STEPS
    assert "unzip it over" in done.err and "was changed since its update was interrupted" in done.err
    assert "another release" not in done.err


@pytest.mark.parametrize("setup", [True, False])
def test_an_edited_gate_file_is_called_changed_not_another_release(inst, setup):
    inst.go(job.STEPS)
    path = inst.root / "localm" / "__init__.py"
    path.write_text(path.read_text(encoding="utf-8") + "# mine\n", encoding="utf-8")
    before = um.tree(inst.root)
    done = inst.gate(**({"LOCALM_SETUP": "1"} if setup else {}))
    assert done.code == 1 and um.tree(inst.root) == before and inst.phase() == job.STEPS, done.err
    assert "localm/__init__.py of LocaLM was changed since its update was interrupted" in done.err
    assert "another release" not in done.err and "unzip it over" in done.err


@pytest.mark.parametrize("version,over", [("2", {}), ("2", {**NEW, "update-package.json": _statement(NEW, "2")}),
                                          ("3", {**THREE, "update-package.json": _statement(THREE, "3")})])
def test_setup_goes_on_when_an_old_file_is_lost(inst, version, over):
    inst.go(job.STEPS)
    inst.lose("localm/part.py")
    before = um.tree(inst.root)
    done = inst.localm()
    assert done.code == 1 and um.tree(inst.root) == before and inst.phase() != restoring.SETUP
    for text in ("part.py: its old version is lost", "run setup in", "unzip it over"):
        assert text in done.err, done.err
    um.write(inst.root, over)
    shutil.rmtree(inst.root / "localm" / "__pycache__", ignore_errors=True)
    before = um.tree(inst.root)
    done = inst.gate(LOCALM_SETUP="1")
    assert done.code == 0 and f"LocaLM {version}" in done.err and inst.phase() == restoring.SETUP, done.err
    assert um.tree(inst.root) == before
    again = inst.gate(LOCALM_SETUP="1")
    assert again.code == 0 and inst.phase() == restoring.SETUP, again.err
    cancelled = inst.localm()
    assert cancelled.code == 1 and "setup did not finish" in cancelled.err
    done = inst.localm(LOCALM_SETUP="1")
    assert ran(done, version) and done.err == "", done.err
    inst.record.unlink()
    done = inst.localm()
    assert ran(done, version) and done.err == ""


def test_the_lost_file_put_back_in_its_place_heals_it(inst):
    inst.go(job.STEPS)
    inst.lose("localm/part.py")
    assert inst.localm().code == 1
    um.write(inst.root, {"localm/part.py": OLD["localm/part.py"]})
    assert ran(inst.localm(), "1") and inst.is_old()


def test_a_mix_is_refused_by_setup_until_a_release_is_put_over_it(inst):
    inst.go(job.STEPS)
    inst.lose("localm/part.py")
    (inst.root / "VERSION").write_text("1")
    done = inst.gate(LOCALM_SETUP="1")
    assert done.code == 1 and "unzip it over" in done.err and inst.phase() != restoring.SETUP
    assert ran(inst.localm(LOCALM_UNINSTALL="1"), "1") is False
    assert inst.localm(LOCALM_UNINSTALL="1").said["part"] == "2" and inst.phase() != restoring.SETUP
    um.write(inst.root, NEW)
    assert inst.gate(LOCALM_SETUP="1").code == 0


@pytest.mark.parametrize("result,out", [
    (job.Result("failed", ["why"]), True), (job.Result("failed", ["why"], held=[disk.Held("f", "r")]), False),
    (OSError("why"), True), (disk.Held("why", "r"), False), (PermissionError("why"), False),
    (job.Busy("why"), False)])
def test_the_ways_out_are_for_what_waiting_and_closing_cannot_help(inst, monkeypatch, result, out):
    def put_back(*_a, **_k):
        if isinstance(result, Exception):
            raise result
        return result
    inst.go(job.STEPS)
    monkeypatch.setattr(restoring.steps_update, "put_back", put_back)
    monkeypatch.delenv("LOCALM_SETUP", raising=False)
    code, lines = restoring._once(inst.root)
    said_ = "\n".join(lines)
    assert code == (restoring.BUSY if isinstance(result, job.Busy) else restoring.FAILED)
    assert "why" in said_ and ("run setup" in said_) == out
    assert ("account that installed it" in said_) == isinstance(result, PermissionError)


def test_a_part_that_another_process_changes_is_waited_for(inst, monkeypatch):
    part = inst.root / "runtime"
    part.mkdir()
    with job.Lock(inst.root) as lock:
        inst.start(lock, job.STEPS)
        job.add_part(lock, part)
    monkeypatch.delenv("LOCALM_SETUP", raising=False)
    with job.Lock(part, wait_s=0):
        code, lines = restoring._once(inst.root)
    assert code == restoring.FAILED and "another process" in lines[-1] and "run setup" not in lines[-1]
    assert ran(inst.localm(), "1") and not inst.record.exists()


def _cmd(inst, *lines, **env):
    script = inst.root.parent / "do.bat"
    script.write_text("@echo off\r\n" + "\r\n".join(lines) + "\r\n")
    return subprocess.run(["cmd", "/d", "/c", str(script)], env=inst.env(**env), cwd=str(inst.root.parent),
                          text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)


@nt_only
@pytest.mark.parametrize("script,until", [
    ("setup.bat", "setlocal EnableDelayedExpansion"), ("localm.bat", 'set "PY=%~dp0.venv'),
    ("setup-gui.bat", 'set "LOCALM_SETUP_GUI_HIDDEN="'),
    ("localm-launcher.bat", 'if exist ".venv\\Scripts\\pythonw.exe"')])
def test_nothing_of_the_gate_stays_in_the_console(tmp_path, script, until):
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    head = _text(script).split(until)[0] + "echo BEGINS\n"
    (inst.root / script).write_bytes(head.replace("\n", "\r\n").encode())
    inst.damage()
    call = f'call "{inst.root / script}"'
    done = _cmd(inst, call, call, "echo AFTER [%LOCALM_SETUP%%LOCALM_GATE%%LOCALM_UNINSTALL%]",
                LOCALM_SETUP_GUI_INPLACE="1")
    assert "BEGINS" not in done.stdout and "AFTER []" in done.stdout, (done.stdout, done.stderr)
    assert done.stderr.count("What helps") == 2


@nt_only
def test_update_gate_bat_over_the_network(tmp_path):
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    drive, rest = os.path.splitdrive(inst.root)
    unc = f"\\\\127.0.0.1\\{drive[0]}${rest}"
    if not os.path.isdir(unc):
        pytest.skip("this account cannot reach the drive over the network")
    inst.go(job.STEPS)
    done = _cmd(inst, f'call "{unc}\\scripts\\update_gate.bat"')
    assert done.returncode == 0 and inst.is_old(), (done.stdout, done.stderr)


@nt_only
@pytest.mark.parametrize("script,until", [
    ("setup.bat", "setlocal EnableDelayedExpansion"),
    ("setup-gui.bat", "echo   LocaLM graphical setup"),
    ("localm-launcher.bat", 'if exist ".venv\\Scripts\\pythonw.exe"'),
    ("rollback.bat", "rem Prefer the clone's venv Python")])
def test_a_batch_script_undoes_it_over_the_network(tmp_path, script, until):
    inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    drive, rest = os.path.splitdrive(inst.root)
    unc = f"\\\\127.0.0.1\\{drive[0]}${rest}"
    if not os.path.isdir(unc):
        pytest.skip("this account cannot reach the drive over the network")
    head = _text(script).split(until)[0] + "echo BEGINS %LOCALM_GATE%.\n"
    (inst.root / script).write_bytes(head.replace("\n", "\r\n").encode())
    inst.old = um.tree(inst.root)
    inst.go(job.STEPS)
    done = _cmd(inst, f'call "{unc}\\{script}" --yes', LOCALM_SETUP_GUI_INPLACE="1")
    assert inst.is_old(), (done.stdout, done.stderr)
    if script in ("setup.bat", "setup-gui.bat", "localm-launcher.bat"):
        assert "cannot run from a network folder" in done.stdout and "BEGINS" not in done.stdout, done.stdout
    if script == "setup.bat":
        assert done.stdout.count("Press any key") <= 1, done.stdout


_SETUPS = {"setup.bat": ("setlocal EnableDelayedExpansion", "[%LEFTUPDATE%]"),
           "setup.sh": ("say() {", "[$LEFTUPDATE]")}


@pytest.mark.parametrize("script", [pytest.param("setup.bat", marks=nt_only),
                                    pytest.param("setup.sh", marks=posix_only)])
def test_setup_goes_on_with_a_new_environment_and_always_uninstalls(tmp_path, script):
    if os.name == "nt":
        inst = Install(make_windows_clone(tmp_path, "localm.bat"))
    else:
        inst = Install(tmp_path / "inst")
        (inst.root / ".venv" / "bin").mkdir(parents=True)
        (inst.root / ".venv" / "bin" / "python").symlink_to(Path(sys.base_prefix) / "bin" / "python3")
    until, left = _SETUPS[script]
    head = _text(script).split(until)[0] + f"echo BEGINS {left}\n"
    (inst.root / script).write_bytes((head.replace("\n", "\r\n") if os.name == "nt" else head).encode())
    inst.go(job.STEPS)
    inst.lose("localm/part.py")
    part = inst.root / "localm" / "part.py"

    def setup(*args):
        cmd = ["cmd", "/d", "/c", "call"] if os.name == "nt" else ["bash"]
        return subprocess.run([*cmd, str(inst.root / script), *args], env=inst.env(), cwd=str(inst.root),
                              text=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, **HIDDEN)
    part.write_text("V = 'mixed'")
    assert "BEGINS" not in setup().stdout
    for arg in ("--uninstall", "uninstall", "--rollback", "finish-uninstall"):
        done = setup(arg)
        assert re.search(r"BEGINS \[0?\]", done.stdout), (done.stdout, done.stderr)
    assert inst.phase() != restoring.SETUP
    um.write(inst.root, {"localm/part.py": NEW["localm/part.py"]})
    for _ in range(2):
        done = setup()
        assert "BEGINS [1]" in done.stdout and inst.phase() == restoring.SETUP, (done.stdout, done.stderr)


def test_the_new_environment_ends_it_in_every_setup(inst, monkeypatch):
    bat, sh = _text("setup.bat"), _text("setup.sh")
    assert bat.index("if defined LEFTUPDATE call :jr begin venv & goto venv_create") < bat.index(":venv_create\n")
    done = 'localm\\update_mechanism setup-done . || echo  [^^!] Setup goes on.'
    assert bat.index(":install_ok") < bat.index(
        'if defined LEFTUPDATE ".venv\\Scripts\\python.exe" -I -S ' + done) < bat.index("call :step_done install-localm")
    assert sh.index('elif [ "$LEFTUPDATE" = 1 ]') < sh.index("install_localm_packages || {") < sh.index(
        'if [ "${LEFTUPDATE:-0}" = 1 ]; then\n  .venv/bin/python -I -S localm/update_mechanism setup-done . \\\n'
        '    || say "  [!] Setup goes on.') < sh.index("step_done install-localm")
    gui = runpy.run_path(str(REPO / "installer" / "gui.py"), run_name="gui")
    made: list = []
    scope = gui["build_steps"].__globals__

    def run(cmd, emit, plan, **k):
        if "setup-done" in cmd:
            return subprocess.run([sys.executable, *cmd[1:]], text=True, capture_output=True,
                                  timeout=60, **HIDDEN).returncode
        made.append(inst.phase())
        return 0
    scope.update(ROOT=inst.root, uv_argv=lambda *a: list(a), create_venv=lambda plan, emit: made.append((inst.root / ".venv").mkdir(
        exist_ok=True)), _run=run)
    (inst.root / ".venv").mkdir()
    inst.record.write_text('{"id": "", "phase": "setup"}')
    steps = {step.key: step.run for step in gui["build_steps"](gui["Plan"]())}
    said_: list = []
    steps["venv"](said_.append)
    assert made == [None] and "made again" in said_[0] and inst.phase() == "setup"
    steps["install-localm"](said_.append)
    assert made[1:] == ["setup"] and not inst.record.exists(), said_


@pytest.mark.parametrize("phase", [restoring.SETUP, job.STEPS, job.ENDING, "damaged"])
def test_setup_deletes_only_the_record_it_kept_itself(inst, phase):
    if phase == restoring.SETUP:
        inst.record.parent.mkdir(exist_ok=True)
        inst.record.write_text('{"id": "' + "0" * 32 + '", "phase": "setup"}')
    else:
        inst.go(job.STEPS if phase == "damaged" else phase)
        if phase == "damaged":
            inst.record.write_bytes(b"\0" * 40)
    before = inst.record.read_bytes()
    done = inst.run("-I", "-S", str(inst.root / "localm" / "update_mechanism"), "setup-done", ".")
    assert done.code == 0, done.err
    if phase == restoring.SETUP:
        assert not inst.record.exists() and done.err == ""
    else:
        assert inst.record.read_bytes() == before and "it stays" in done.err
    with job.Lock(inst.root):
        held = inst.run("-I", "-S", str(inst.root / "localm" / "update_mechanism"), "setup-done", ".")
    assert held.code == restoring.BUSY and "stays" in held.err


# ------------------------------------------------- what the gate cannot do

def test_a_broken_stderr_does_not_stop_it(inst):
    for closed in ("sys.stderr.close()", "os.close(2)"):
        inst.go(job.STEPS) if not inst.record.exists() else None
        done = inst.run("-c", f"import os, sys; {closed}; import localm; print(localm.__version__)")
        assert done.out.strip() == "1" and inst.is_old(), (done.out, done.err)
        um.write(inst.root / ".localm-update" / ID / "new", {
            **NEW, "update-package.json": _statement(NEW, "2")})
        shared.copy_applier(REPO / "localm", inst.root / ".localm-update" / ID / "applier")
    inst.go(job.STEPS)
    read, write = os.pipe()
    os.close(read)
    proc = subprocess.run([sys.executable, "-c", "import localm; print(localm.__version__)"], env=inst.env(),
                          cwd=str(inst.root), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=write,
                          text=True, timeout=120, **HIDDEN)
    os.close(write)
    assert (proc.returncode, proc.stdout.strip()) == (0, "1") and inst.is_old()


@pytest.mark.skipif(os.name != "nt" and os.getuid() == 0, reason="root may write everything")
def test_an_account_that_may_not_write_the_install_waits_for_a_running_update(inst, monkeypatch):
    inst.go(job.STEPS)
    lock = inst.root / ".localm-update" / "lock"
    folder = str(lock.parent)
    with job.Lock(inst.root):
        lock.chmod(0o444)
        assert localm._held(folder)
    assert not localm._held(folder)
    done = inst.localm()
    assert done.code == 1 and "account that installed it" in done.err and inst.record.exists(), done.err
    lock.chmod(0o666)
    assert ran(inst.localm(), "1")
    monkeypatch.setattr(localm.os, "open", lambda *a: (_ for _ in ()).throw(PermissionError()))
    assert localm._held(folder)


def test_at_a_terminal_a_lock_this_account_may_not_open_ends_the_wait(inst, monkeypatch, capsys):
    inst.go(job.STEPS)
    real_open, ran_ = os.open, []

    def open_(path, *a, **k):
        if os.path.basename(path) == "lock":
            raise PermissionError(13, "denied", path)
        return real_open(path, *a, **k)

    def sleep(_s):
        ran_.append("slept")
        if ran_.count("slept") > 20:
            raise KeyboardInterrupt
    monkeypatch.setattr(localm, "__file__", str(inst.root / "localm" / "__init__.py"))
    monkeypatch.setattr(localm, "_NAME_S", 0.0)
    monkeypatch.setattr(localm, "_tty", lambda stream: True)
    monkeypatch.setattr(localm, "_put_back", lambda *a: ran_.append("restore") or 1)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(localm.os, "open", open_)
    with pytest.raises(SystemExit) as end:
        localm._gate()
    monkeypatch.undo()
    err = capsys.readouterr().err
    assert end.value.code == 1 and ran_ == ["restore"], (ran_, err)
    assert "may not open" in err and "account that installed it" in err


def test_setup_puts_nothing_aside_while_a_file_an_update_added_is_left(inst, monkeypatch):
    inst.go(job.STEPS)
    um.write(inst.root, {**OLD, "update-package.json": _statement(OLD, "1")})
    assert (inst.root / "localm" / "only2.py").exists()
    monkeypatch.setattr(restoring.steps_update, "put_back", lambda *a, **k: job.Result("failed", ["why"]))
    monkeypatch.setenv("LOCALM_SETUP", "1")
    code, lines = restoring._once(inst.root)
    assert code == restoring.FAILED and inst.phase() == job.STEPS, lines
    assert "unzip it over" in "\n".join(lines) and (inst.root / ".localm-update" / ID).is_dir()
    (inst.root / "localm" / "only2.py").unlink()
    code, lines = restoring._once(inst.root)
    assert code == restoring.OK and inst.phase() == restoring.SETUP and "LocaLM 1" in "\n".join(lines), lines


def test_what_cannot_be_written_is_said_once(inst, monkeypatch):
    inst.go(job.STEPS)
    made: list = []
    monkeypatch.setattr(localm, "__file__", str(inst.root / "localm" / "__init__.py"))
    monkeypatch.setattr(localm, "_touch", lambda mark, remove=False: made.append(mark))

    def sleep(_s):
        made.append("slept")
        if made.count("slept") > 3:
            raise KeyboardInterrupt
    monkeypatch.setattr(time, "sleep", sleep)
    with job.Lock(inst.root), pytest.raises(SystemExit):
        localm._gate(procs.Entry(os.getpid(), "1", "gui"))
    assert len(made) == 5 and made[1:] == ["slept"] * 4


def test_a_record_that_cannot_be_looked_at_is_refused(inst, monkeypatch, capsys):
    def lstat(path, *a):
        raise PermissionError(13, "denied")
    monkeypatch.setattr(localm, "__file__", str(inst.root / "localm" / "__init__.py"))
    monkeypatch.setattr(localm.os, "lstat", lstat)
    with pytest.raises(SystemExit) as end:
        localm._gate()
    monkeypatch.undo()
    assert end.value.code == 1 and "account that installed it" in capsys.readouterr().err


def test_the_program_that_updates_is_named(inst):
    inst.go(job.ENDING)
    path = inst.root / "localm" / "__init__.py"
    path.write_text(path.read_text(encoding="utf-8").replace("3600.0, 60.0, 60.0", "3600.0, 60.0, 0.5"),
                    encoding="utf-8")
    with job.Lock(inst.root):
        proc = inst.popen("-m", "localm")
        lines: list = []
        while not any("another process is updating" in line for line in lines):
            lines.append(proc.stderr.readline())
            assert lines[-1], lines
        if sys.platform in ("win32", "linux"):
            assert f"(pid {os.getpid()})" in lines[-1], lines
        assert proc.poll() is None and not any("interrupted" in line for line in lines)
    out, err = proc.communicate(timeout=60)
    assert ran(types.SimpleNamespace(said=said(out)), "1"), err


def test_no_message_says_that_kept_may_be_deleted():
    for rel in ("localm/__init__.py", "scripts/update_gate.bat", "scripts/update_gate.sh"):
        text = re.sub(r'["\s]+(?:f"|1>&2|>&2|echo)*\s*', " ", _text(rel))
        assert "delete everything in" in text and "except kept" in text and "delete the folder" not in text, rel


# --------------------------------------------------------- planted and damaged

@pytest.mark.parametrize("planted", ["id", "program", "link"])
def test_nothing_planted_runs(inst, tmp_path, planted):
    evil = tmp_path / "evil"
    for at in (evil / "update_mechanism", evil / "applier" / "update_mechanism"):
        at.mkdir(parents=True)
        (at / "__main__.py").write_text("print('EVIL'); import sys; sys.exit(0)")
    inst.go(job.STEPS)
    raw = json.loads(inst.record.read_text())
    if planted == "id":
        raw["id"] = "../../evil"
    elif planted == "program":
        raw["applier"] = raw["program"] = str(evil / "update_mechanism")
    else:
        own = inst.root / ".localm-update" / ID / "applier"
        shutil.rmtree(own)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(own), str(evil / "applier")], check=True,
                           capture_output=True, **HIDDEN)
        else:
            own.symlink_to(evil / "applier")
    inst.record.write_text(json.dumps(raw))
    (evil / "ran").mkdir()
    done = inst.localm()
    assert "EVIL" not in done.out + done.err
    assert ran(done, "1") and (inst.root / "VERSION").read_text() == "1", done.err


@pytest.mark.parametrize("gone", ["applier", "applier and new", "all"])
def test_the_program_that_is_left_is_used(inst, gone):
    inst.go(job.STEPS)
    work = inst.root / ".localm-update" / ID
    shutil.rmtree(work / "applier")
    if gone == "applier":
        um.write(work / "new", {f"localm/{n}.py": OLD[f"localm/{n}.py"] for n in _MODULES})
        um.write(work / "new", {r: t for r, t in OLD.items() if "update_mechanism" in r})
        shutil.rmtree(inst.root / "localm" / "update_mechanism")
    if gone == "all":
        shutil.rmtree(inst.root / "localm" / "update_mechanism")
        shutil.rmtree(work / "new")
        done = inst.localm()
        assert done.code == 1 and done.said is None and "run setup" in done.err
        return
    done = inst.localm()
    assert done.code == 0 and ran(done, "1"), done.err


def test_a_record_of_another_install_is_put_aside_when_the_files_are_one_release(inst):
    inst.go(job.STEPS)
    record = inst.record.read_text()
    other = Install(inst.root.parent / "other")
    shutil.rmtree(other.root / ".localm-update" / ID)
    other.record.write_text(record)
    done = other.localm()
    assert done.code == 0 and ran(done, "1") and um.tree(other.root) == other.old, done.err
    assert not other.record.exists()


@pytest.mark.parametrize("whole", [True, False])
def test_an_update_that_cannot_be_told(inst, whole):
    """Record and state unreadable and the plans damaged."""
    inst.damage()
    if whole:
        um.write(inst.root, {"localm/part.py": NEW["localm/part.py"]})
    before = um.tree(inst.root)
    done = inst.localm()
    assert um.tree(inst.root) == before
    if whole:
        assert done.code == 0 and ran(done, "2") and "put aside" in done.err, done.err
        with job.Lock(inst.root) as lock:
            assert job.restore(lock).state == "none"
        return
    assert done.code == 1 and done.said is None and inst.record.exists()
    for text in (str(inst.root / ".localm-update"), "releases", "run setup"):
        assert text in done.err
    um.write(inst.root, NEW)
    assert ran(inst.localm(), "2")


def test_with_packages_half_replaced_only_setup_puts_it_aside(inst, monkeypatch):
    inst.damage()
    um.write(inst.root, {"localm/part.py": NEW["localm/part.py"]})
    um.write(inst.root / ".venv" / ".localm-old" / ID / "old" / "01", {"click.py": "old"})
    assert inst.localm().code == 1
    um.write(inst.root, {".venv/.localm-kept/y/mine.py": "m"})
    assert inst.gate(LOCALM_SETUP="1").code == 0 and inst.phase() == restoring.SETUP
    done = inst.localm()
    assert done.code == 1 and "setup did not finish" in done.err
    assert ran(inst.localm(LOCALM_SETUP="1"), "2")
    inst.record.unlink()
    assert ran(inst.localm(), "2")
    with job.Lock(inst.root) as lock:
        assert job.restore(lock).state == "none"
    assert list((inst.root / ".localm-update" / "kept").rglob("mine.py"))


def test_an_interpreter_outside_the_versions_localm_runs_on_is_refused(inst):
    inst.go(job.STEPS)
    path = inst.root / "localm" / "__init__.py"
    before = path.read_text(encoding="utf-8")
    path.write_text(before.replace("_PYTHONS = ((3, 12),)", "_PYTHONS = ((3, 11),)"), encoding="utf-8")
    done = inst.localm()
    assert done.code == 1 and "cannot run LocaLM" in done.err and inst.record.exists()


def test_the_python_versions_are_those_of_pyproject():
    text = _text("pyproject.toml")
    assert 'requires-python = ">=3.12,<3.13"' in text
    assert "_PYTHONS = ((3, 12),)" in _text("localm/__init__.py")
    for script in ("scripts/update_gate.bat", "scripts/update_gate.sh"):
        assert "sys.version_info[:2] != (3, 12)" in _text(script)


def test_the_gate_loads_the_standard_library_only():
    text = _text("localm/__init__.py")
    assert "importlib.metadata" not in text
    names = set(re.findall(r"^\s*(?:import|from) (\w+)", text, re.M))
    assert names <= set(sys.stdlib_module_names) | {"localm"}


def test_the_files_of_a_start_are_never_absent_while_an_update_moves_them():
    loaded = [a or b for a, b in re.findall(r"^\s+(?:from localm import (\w+)|import localm\.(\w+))",
                                            _text("localm/__init__.py"), re.M)]
    assert loaded and {"localm/__init__.py", *(f"localm/{n}.py" for n in loaded)} <= set(mover.GATE_FILES)
    callers = {p for p in (*REPO.glob("*.bat"), *REPO.glob("*.sh"), REPO / "launcher.pyw",
                           REPO / "installer" / "gui.py", REPO / "scripts" / "rollback_update.py")
               if ".localm-update" in p.read_text(encoding="utf-8")}
    assert {p.relative_to(REPO).as_posix() for p in callers} <= set(mover.GATE_FILES)
    git = shutil.which("git")
    if not git or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run([git, "ls-files", *mover.GATE_FILES], cwd=REPO, text=True,
                             capture_output=True, check=True, **HIDDEN).stdout.split()
    assert set(mover.GATE_FILES) == set(tracked)


# ---------------------------------------------------------- held files, asking

@nt_only
def test_without_a_terminal_a_held_file_is_waited_for_then_named(inst):
    inst.go(job.STEPS)
    own = inst.root / ".localm-update" / ID / "applier" / "update_mechanism" / "disk.py"
    own.write_text(own.read_text(encoding="utf-8").replace("HELD_WAIT_S = 30.0", "HELD_WAIT_S = 0.5"),
                   encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; f = open(sys.argv[1], 'rb'); print('H', flush=True);"
         "sys.stdin.read()", str(inst.root / "localm" / "part.py")], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, **HIDDEN)
    holder.stdout.readline()
    done = inst.localm()
    assert done.code == 1 and done.said is None and inst.record.exists()
    assert "releases" not in done.err
    assert "part.py" in done.err and re.search(r"held by \w+ .*\(pid \d+\)|held by \w+ \(pid", done.err)
    holder.communicate("")
    assert ran(inst.localm(), "1") and inst.is_old()


@pytest.mark.parametrize("answers,code,tries", [(["c\n"], 1, 1), (["x\n", "T\n", "c\n"], 1, 2), ([], 1, 1),
                                                (["t\n", "t\n"], 0, 3)])
def test_at_a_terminal_it_asks(monkeypatch, answers, code, tries):
    results = [(1, ["why"]), (1, ["why"]), (0, [])]
    made, said_ = [], []
    monkeypatch.setattr(restoring, "_once", lambda root: made.append(1) or results[len(made) - 1])
    monkeypatch.setattr(restoring, "at_terminal", lambda: True)
    monkeypatch.setattr(restoring, "say", said_.append)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(
        readline=lambda: answers.pop(0) if answers else ""))
    assert (restoring.run("x"), len(made)) == (code, tries)
    assert "why" in said_


def test_the_uv_is_the_one_of_the_install(inst, monkeypatch, tmp_path):
    name = "uv.exe" if os.name == "nt" else "uv"
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "user"))
    on_path = tmp_path / "path"
    places = [tmp_path / "user" / ".local" / "bin", inst.root / ".venv" / (
        "Scripts" if os.name == "nt" else "bin"), tmp_path / "shared", inst.root / ".uv"]
    um.write(inst.root, {".localm-install.json": json.dumps(
        {"uv_exe": str(tmp_path / "shared" / name), "runtime_contained": True})})
    um.write(on_path, {name: ""})
    (on_path / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(on_path))
    assert restoring.find_uv(inst.root) is None
    for place in places:
        um.write(place, {name: ""})
        (place / name).chmod(0o755)
        assert restoring.find_uv(inst.root) == str(place / name)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    assert restoring.environment(inst.root)["UV_CACHE_DIR"] == str(inst.root / ".cache")


# ------------------------------------------------------------------ uninstall

@pytest.mark.parametrize("purge", [True, False])
def test_uninstall_removes_what_updates_left_and_keeps_the_users_files(inst, purge):
    package = inst.root / "runtime" / "localm_llama_runtime"
    um.write(inst.root, {".localm-update/kept/x/mine.py": "m", ".venv/.localm-kept/y/mine.py": "m",
                         ".localm-run/a": "", "runtime/localm_llama_runtime/.localm-update/lock": "",
                         ".venv/.localm-venv": ""})
    report = install_manifest.uninstall(inst.root, purge_data=purge, force=True, defer_runtime=True,
                                        log=lambda *_: None)
    kept = inst.root / ".localm-update" / "kept" / "x" / "mine.py"
    assert kept.exists() != purge and not (package / ".localm-update").exists()
    assert not (inst.root / ".localm-update" / ID).exists()
    assert not (inst.root / ".localm-run").exists()
    assert any(".localm-kept" in str(p) for p, _why in report["notes"])
