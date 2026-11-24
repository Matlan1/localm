# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update mechanism's package step: a real uv, real venvs and local wheels."""

from __future__ import annotations

import base64
import csv
import ctypes.util
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import zipfile
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover, packages
from tests import _update_mechanism as um
from tests._held_file import held

uv = pytest.importorskip("uv")

ID = um.UPDATE_ID
WINDOWS = os.name == "nt"
windows_only = pytest.mark.skipif(not WINDOWS, reason="Windows share modes")
SCRIPTS = "Scripts" if WINDOWS else "bin"
EXE = ".exe" if WINDOWS else ""
LIBRARY = "alpha/_uuid.pyd" if WINDOWS else "alpha/_lib.so"
SYNC = ("sync", "--frozen", "--inexact", "--no-config")


# -------------------------------------------------------------- wheels and venvs

def library() -> bytes:
    """A native library this interpreter can load."""
    if WINDOWS:
        return (Path(sys.base_prefix) / "DLLs" / "_uuid.pyd").read_bytes()
    for folder in (Path(sysconfig.get_path("platstdlib")) / "lib-dynload",):
        for path in sorted(folder.glob("*.so")):
            return path.read_bytes()
    name = ctypes.util.find_library("m") or pytest.skip("no native library to copy")
    for folder in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib64", "/usr/lib64",
                   "/lib/aarch64-linux-gnu", "/usr/lib"):
        if (Path(folder) / name).is_file():
            return (Path(folder) / name).read_bytes()
    pytest.skip("no native library to copy")


def wheel(folder: Path, name: str, version: str, files: dict, scripts: dict | None = None) -> None:
    info = f"{name}-{version}.dist-info"
    entries = {**files,
               f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
               f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                                "Tag: py3-none-any\n"}
    for script, content in (scripts or {}).items():
        entries[f"{name}-{version}.data/scripts/{script}"] = content
    rows = io.StringIO()
    out = csv.writer(rows, lineterminator="\n")
    with zipfile.ZipFile(folder / f"{name}-{version}-py3-none-any.whl", "w") as z:
        for path, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            z.writestr(path, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            out.writerow([path, f"sha256={digest}", len(data)])
        out.writerow([f"{info}/RECORD", "", ""])
        z.writestr(f"{info}/RECORD", rows.getvalue())


class World:
    """Local wheels, a uv with its own cache and the projects made from them."""

    def __init__(self, base: Path) -> None:
        self.base, self.wheels = base, base / "wheels"
        self.wheels.mkdir(parents=True)
        self.uv = uv.find_uv_bin()
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("UV_") and k != "VIRTUAL_ENV"}
        self.env.update(UV_CACHE_DIR=str(base / "cache"), UV_PYTHON_DOWNLOADS="never",
                        UV_NO_PROGRESS="1")

    def run(self, root: Path, *argv: str) -> str:
        done = subprocess.run([self.uv, *argv], cwd=root, env=self.env, capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=120, **um.NO_WINDOW)
        assert done.returncode == 0, (argv, done.stderr)
        return done.stderr

    def lock(self, root: Path, deps: list, *, name: str = "app", extra: str = "") -> None:
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\nversion = "0"\nrequires-python = ">=3.12"\n'
            f"dependencies = {json.dumps(deps)}\n{extra}\n"
            f'[tool.uv]\nno-index = true\nfind-links = [{json.dumps(self.wheels.as_posix())}]\n',
            encoding="utf-8")
        self.run(root, "lock", "--offline", "--python", sys.executable)

    def install(self, root: Path, deps: list, **kw) -> Path:
        root.mkdir(parents=True)
        (root / "VERSION").write_text("1", encoding="utf-8")
        self.lock(root, deps, **kw)
        self.run(root, "venv", "--python", sys.executable)
        self.run(root, *SYNC, "--offline", "--no-install-project")
        return root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("world"))
    native = library()
    wheel(w.wheels, "alpha", "1.0", {
        "alpha/__init__.py": "V = 1\n", "alpha/old.py": "x = 1\n", "alpha/olddir/x.py": "x = 1\n",
        "alpha/data.txt": "one\n", LIBRARY: native, "shared_ns/a.py": "a = 1\n"},
        {"alpha-run": "#!python\nprint(1)\n"})
    wheel(w.wheels, "alpha", "2.0", {
        "alpha/__init__.py": "V = 2\n", "alpha/new.py": "x = 2\n", "alpha/newdir/x.py": "x = 2\n",
        "alpha/data.txt": "two\n", LIBRARY: native + b"\0" * 64, "shared_ns/a.py": "a = 2\n"},
        {"alpha-run": "#!python\nprint(2)\n"})
    wheel(w.wheels, "beta", "1.0", {"beta.py": "V = 1\n", "shared_ns/b.py": "b = 1\n"})
    wheel(w.wheels, "gamma", "1.0", {"gamma/__init__.py": "V = 1\n"})
    wheel(w.wheels, "delta", "1.0", {"delta/__init__.py": "V = 1\n"})
    wheel(w.wheels, "big", "1.0", {"big/__init__.py": ""})
    wheel(w.wheels, "big", "2.0", {"big/__init__.py": "",
                                   **{f"big/d{n % 40}/f{n}.py": f"n = {n}\n" for n in range(3000)}})
    return w


@pytest.fixture(scope="module")
def template(world, tmp_path_factory):
    """alpha 1.0, beta and delta installed, with the lock of alpha 2.0, beta and
    gamma."""
    root = world.install(tmp_path_factory.mktemp("template") / "root",
                         ["alpha==1.0", "beta==1.0", "delta==1.0"])
    world.lock(root, ["alpha==2.0", "beta==1.0", "gamma==1.0"])
    return root


@pytest.fixture
def root(template, tmp_path):
    return copy(template, tmp_path / "root")


def copy(template: Path, dest: Path) -> Path:
    shutil.copytree(template, dest, symlinks=True)
    return dest


def tree(root: Path) -> dict:
    return um.tree(root, skip=(".localm-update", ".venv/.lock", ".venv/.localm-old"))


def site(root: Path) -> Path:
    found = packages.shared.updater_components.site_packages(root / ".venv")
    assert found is not None
    return found


def versions(root: Path) -> dict:
    return {n: v for n, v, _f in packages.shared.updater_components.dist_infos(site(root))}


def argv(world, *more: str) -> list:
    return [world.uv, *SYNC, *more]


def begin(root: Path) -> None:
    (root / ".localm-update" / ID / "new").mkdir(parents=True)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)


def plan(root: Path, world, command: list | None = None, **kw):
    if um.phase(root) is None:
        begin(root)
    made = packages.plan_packages(root, ID, command or argv(world), env=world.env, **kw)
    if made is not None:
        mover.write_plan(root, ID, made)
    return made


def forward(root: Path, world, made, command: list | None = None, **kw) -> None:
    packages.forward(root, ID, made, command or argv(world), env=world.env, **kw)


def undo(root: Path, **kw) -> job.Result:
    with job.Lock(root) as lock:
        return job.restore(lock, **kw)


def end(root: Path) -> job.Result:
    with job.Lock(root) as lock:
        job.advance(lock, job.STARTING)
        return job.finish(lock)


def imported(root: Path, code: str) -> str:
    done = subprocess.run([str(root / ".venv" / SCRIPTS / f"python{EXE}"), "-I", "-B", "-c", code],
                          capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                          **um.NO_WINDOW)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


# ------------------------------------------------------------------ there and back

def test_exactly_the_planned_packages_are_replaced_and_put_back(root, world):
    before = tree(root)
    made = plan(root, world)
    assert made is not None
    inner = site(root).relative_to(root / ".venv").as_posix()
    moved = {op.path for op in made.ops}
    assert f"{inner}/alpha/old.py" in moved and f"{SCRIPTS}/alpha-run" in moved
    assert f"{inner}/alpha-1.0.dist-info/RECORD" in moved
    assert not any(part in path for path in moved for part in ("beta", "delta", "shared_ns/b"))
    forward(root, world, made)
    assert versions(root) == {"alpha": "2.0", "beta": "1.0", "gamma": "1.0", "delta": "1.0"}
    assert not (site(root) / "alpha" / "olddir").exists()
    assert imported(root, "import alpha, alpha.new, gamma; print(alpha.V)") == "2"
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert tree(root) == before
    assert not (root / ".venv" / mover.VENV_FOLDER).exists() and um.left(root) == []


def test_a_finished_step_keeps_the_new_packages_and_nothing_of_the_old(root, world):
    forward(root, world, plan(root, world))
    result = end(root)
    assert (result.state, result.errors) == ("new", [])
    assert versions(root)["alpha"] == "2.0"
    assert not (root / ".venv" / mover.VENV_FOLDER).exists() and um.left(root) == []
    assert world.run(root, "pip", "check", "--python", str(root / ".venv")).count("compatible")


def test_nothing_to_do_makes_no_plan(root, world):
    forward(root, world, plan(root, world))
    end(root)
    assert plan(root, world) is None


def test_a_package_uv_only_removes_comes_back(root, world):
    before = tree(root)
    command = [world.uv, "sync", "--frozen", "--no-config"]
    made = plan(root, world, command)
    forward(root, world, made, command)
    assert "delta" not in versions(root)
    assert undo(root).state == "old" and tree(root) == before


def test_the_way_back_needs_no_uv_no_cache_and_no_wheel(template, tmp_path):
    own = World(tmp_path / "own")
    root = tmp_path / "root"
    wheel(own.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 1\n"})
    wheel(own.wheels, "alpha", "2.0", {"alpha/__init__.py": "V = 2\n"})
    own.install(root, ["alpha==1.0"])
    own.lock(root, ["alpha==2.0"])
    before = tree(root)
    made = plan(root, own)
    forward(root, own, made)
    shutil.rmtree(own.base)
    assert undo(root).state == "old"
    assert tree(root) == before and imported(root, "import alpha; print(alpha.V)") == "1"


def test_a_wheel_that_cannot_be_had_offline_leaves_everything_as_it_was(tmp_path):
    own = World(tmp_path / "own")
    root = tmp_path / "root"
    wheel(own.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 1\n"})
    wheel(own.wheels, "alpha", "2.0", {"alpha/__init__.py": "V = 2\n"})
    own.install(root, ["alpha==1.0"])
    own.lock(root, ["alpha==2.0"])
    (own.wheels / "alpha-2.0-py3-none-any.whl").unlink()
    before = tree(root)
    made = plan(root, own)
    with pytest.raises(mover.StepFailed, match="uv could not install") as e:
        forward(root, own, made)
    assert e.value.errors == [] and tree(root) == before


def test_a_loaded_library_is_replaced_and_put_back_while_it_stays_loaded(root, world):
    before = tree(root)
    code = ("import alpha._uuid; print('loaded', flush=True); input()" if WINDOWS else
            "import alpha, ctypes, os; ctypes.CDLL(os.path.join(os.path.dirname(alpha.__file__), "
            "'_lib.so')); print('loaded', flush=True); input()")
    user = subprocess.Popen([str(root / ".venv" / SCRIPTS / f"python{EXE}"), "-I", "-B", "-c", code],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, **um.NO_WINDOW)
    assert user.stdin is not None and user.stdout is not None
    try:
        assert user.stdout.readline().strip() == b"loaded"
        made = plan(root, world)
        forward(root, world, made)
        assert site(root).joinpath(*LIBRARY.split("/")).read_bytes().endswith(b"\0" * 64)
        assert imported(root, "import alpha; print(alpha.V)") == "2"
        result = undo(root)
        assert (result.state, result.errors) == ("old", [])
        assert tree(root) == before and user.poll() is None
        forward(root, world, plan(root, world))
        assert end(root).state == "new"
        assert um.left(root) == ([ID] if WINDOWS else [])
    finally:
        user.communicate(b"\n", timeout=30)
    with job.Lock(root) as lock:
        assert job.restore(lock).state == "none"
    assert um.left(root) == [] and not (root / ".venv" / mover.VENV_FOLDER).exists()


def test_a_uv_that_is_itself_replaced_runs_from_where_it_was_moved(tmp_path):
    own = World(tmp_path / "own")
    root = tmp_path / "root"
    wheel(own.wheels, "tooluv", "1.0", {"tooluv.py": ""},
          {f"uvcopy{EXE}": Path(own.uv).read_bytes()})
    wheel(own.wheels, "tooluv", "2.0", {"tooluv.py": ""}, {f"uvcopy{EXE}": "#!python\n"})
    own.install(root, ["tooluv==1.0"])
    own.lock(root, ["tooluv==2.0"])
    before = tree(root)
    command = [str(root / ".venv" / SCRIPTS / f"uvcopy{EXE}"), *SYNC]
    made = plan(root, own, command)
    forward(root, own, made, command)
    assert versions(root) == {"tooluv": "2.0"}
    assert undo(root).state == "old" and tree(root) == before


# ------------------------------------------------------------------- localm itself

BACKEND = '''
import base64, csv, hashlib, io, zipfile
def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    import tomllib
    scripts = tomllib.load(open("pyproject.toml", "rb"))["project"]["scripts"]
    info = "localm-0.dist-info"
    files = {"_localm.pth": ".\\n", info + "/METADATA": "Metadata-Version: 2.1\\nName: localm\\nVersion: 0\\n",
             info + "/WHEEL": "Wheel-Version: 1.0\\nGenerator: t\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
             info + "/entry_points.txt": "[console_scripts]\\n" + "".join(
                 f"{k} = {v}\\n" for k, v in scripts.items())}
    rows = io.StringIO()
    out = csv.writer(rows, lineterminator="\\n")
    name = "localm-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_directory + "/" + name, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
            digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
            out.writerow([path, "sha256=" + digest, len(text)])
        out.writerow([info + "/RECORD", "", ""])
        z.writestr(info + "/RECORD", rows.getvalue())
    return name
def get_requires_for_build_editable(config_settings=None):
    return []
'''


def project(script: str) -> str:
    return (f'[project.scripts]\n{script} = "m:f"\n[build-system]\nrequires = []\n'
            'build-backend = "backend"\nbackend-path = ["."]\n')


def test_localm_itself_is_replaced_only_when_asked(world, tmp_path):
    root = world.install(tmp_path / "root", ["alpha==1.0"], name="localm", extra=project("lm-one"))
    (root / "backend.py").write_text(BACKEND, encoding="utf-8")
    world.run(root, *SYNC, "--offline")
    assert (root / ".venv" / SCRIPTS / f"lm-one{EXE}").is_file()
    time.sleep(1.1)
    world.lock(root, ["alpha==2.0"], name="localm", extra=project("lm-two"))
    before = tree(root)

    made = plan(root, world)
    assert made is not None and not any("localm" in op.path or "lm-" in op.path for op in made.ops)
    forward(root, world, made)
    assert (root / ".venv" / SCRIPTS / f"lm-one{EXE}").is_file()
    assert undo(root).state == "old" and tree(root) == before

    made = plan(root, world, project=True)
    assert made is not None
    assert f"{SCRIPTS}/lm-one{EXE}" in {op.path for op in made.ops}
    forward(root, world, made, project=True)
    assert (root / ".venv" / SCRIPTS / f"lm-two{EXE}").is_file()
    assert not (root / ".venv" / SCRIPTS / f"lm-one{EXE}").exists()
    assert undo(root).state == "old" and tree(root) == before


def test_a_command_that_would_replace_localm_unasked_is_refused(world, tmp_path):
    wheel(world.wheels, "localm", "1.0", {"localm_x.py": ""})
    root = world.install(tmp_path / "root", ["beta==1.0"])
    command = [world.uv, "pip", "install", "--no-config", "--no-index", "--find-links",
               str(world.wheels), "--python", str(root / ".venv"), "localm==1.0"]
    with pytest.raises(mover.Refused, match="localm itself"):
        plan(root, world, command)


# -------------------------------------------------------------------- refused plans

def test_a_package_without_a_record_is_refused_by_name(root, world):
    (site(root) / "alpha-1.0.dist-info" / "RECORD").unlink()
    before = tree(root)
    with pytest.raises(mover.Refused, match="alpha has no list of its files"):
        plan(root, world)
    assert tree(root) == before


def test_a_package_with_an_empty_record_is_refused_by_name(root, world):
    (site(root) / "alpha-1.0.dist-info" / "RECORD").write_bytes(b"")
    with pytest.raises(mover.Refused, match="alpha has an empty list"):
        plan(root, world)


@pytest.mark.parametrize("path", ["../../../../outside.txt", "C:/outside.txt", "/outside.txt",
                                  "..\\..\\..\\..\\outside.txt"])
def test_a_record_that_names_a_file_outside_the_venv_is_refused(root, world, path):
    if "\\" in path and not WINDOWS:
        pytest.skip("a backslash is a plain character here")
    outside = root.parent / "outside.txt"
    outside.write_bytes(b"not mine")
    depth = "../" * len(site(root).relative_to(root).parts)
    for text in {path, depth + "../outside.txt" if path.startswith("../") else path}:
        with open(site(root) / "alpha-1.0.dist-info" / "RECORD", "a", encoding="utf-8") as f:
            f.write(f"{text},,\n")
    if path == "C:/outside.txt" and not WINDOWS:
        pytest.skip("not a drive here")
    with pytest.raises(mover.Refused, match="outside the Python environment"):
        plan(root, world)
    assert outside.read_bytes() == b"not mine"


def test_a_link_in_the_venv_is_not_followed(root, world):
    outside = root.parent / "outside"
    um.write(outside, {"old.py": "x = 1\n", "__init__.py": "V = 1\n"})
    there = um.tree(outside)
    shutil.rmtree(site(root) / "alpha")
    if WINDOWS:
        import _winapi
        _winapi.CreateJunction(str(outside), str(site(root) / "alpha"))
    else:
        (site(root) / "alpha").symlink_to(outside, target_is_directory=True)
    made = plan(root, world)
    with pytest.raises(mover.StepFailed):
        forward(root, world, made)
    assert um.tree(outside) == there


def test_a_venv_uv_would_replace_is_refused(root, world):
    cfg = root / ".venv" / "pyvenv.cfg"
    lines = [line for line in cfg.read_text(encoding="utf-8").splitlines()
             if not line.startswith("version")]
    cfg.write_text("\n".join([*lines, "version_info = 3.11.1"]) + "\n", encoding="utf-8")
    before = tree(root)
    with pytest.raises(mover.Refused, match="environment|cannot prepare"):
        plan(root, world)
    assert tree(root) == before


def test_an_install_without_a_venv_is_refused(root, world):
    shutil.rmtree(root / ".venv")
    with pytest.raises(mover.Refused, match="not a Python environment"):
        plan(root, world)


@pytest.mark.parametrize("text", [
    "Would uninstall 1 package\nWould install 1 package\n + alpha==2.0\n",
    "Would uninstall 2 packages\n - alpha==1.0\n",
    "Would install 1 package\n",
    " + alpha==2.0\n",
])
def test_a_dry_run_whose_counts_and_lines_differ_is_refused(text):
    with pytest.raises(mover.Refused, match="could not be read"):
        packages.changes(text)


@pytest.mark.parametrize("text", [
    "Would replace project environment at: .venv\n",
    "Would create project environment at: .venv\n",
    "Would replace existing virtual environment at: .venv\n",
])
def test_a_dry_run_that_makes_another_venv_is_refused(text):
    with pytest.raises(mover.Refused, match="would not use"):
        packages.changes(text)


def test_the_lines_of_a_dry_run_are_read():
    text = ("Would use project environment at: .venv\nWould download 1 package\n"
            "Would uninstall 2 packages\nWould install 3 packages\n - Alpha_X==1.0\n"
            " + alpha-x==2.0 (from file:///w)\n ~ beta==1.0\n + localm @ file:///a b/c\n")
    assert packages.changes(text) == (
        [("alpha-x", "1.0"), ("beta", "1.0")],
        [("alpha-x", "2.0"), ("beta", "1.0"), ("localm", "")])


# ------------------------------------------------------- uv does something else

def tampering(world, change):
    """Run uv for real, then *change* the venv as if uv had done it."""
    def run(command, cwd, env):
        done = packages.run_hidden(command, cwd, env)
        change()
        return done
    return run


def test_a_file_added_that_nobody_planned_takes_everything_back(root, world):
    before = tree(root)
    made = plan(root, world)
    run = tampering(world, lambda: (site(root) / "stray.py").write_bytes(b"x"))
    with pytest.raises(mover.StepFailed, match="stray.py was added") as e:
        forward(root, world, made, run=run)
    assert e.value.errors == [] and tree(root) == before


def test_a_changed_file_that_nobody_planned_is_a_failure_that_is_named(root, world):
    made = plan(root, world)
    run = tampering(world, lambda: (site(root) / "beta.py").write_bytes(b"changed!"))
    with pytest.raises(mover.StepFailed, match="beta.py was changed") as e:
        forward(root, world, made, run=run)
    assert [x for x in e.value.errors if "beta.py" in x]
    assert versions(root)["alpha"] == "1.0"


def test_a_file_no_package_lists_does_not_stop_the_way_back(root, world):
    """A program may write its own files into the venv while the new version runs."""
    own = site(root) / "beta_cache.txt"
    own.write_bytes(b"cache")
    made = plan(root, world)
    forward(root, world, made)
    own.write_bytes(b"cache written again")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert own.read_bytes() == b"cache written again" and versions(root)["alpha"] == "1.0"


def test_a_removed_file_that_nobody_planned_is_a_failure_that_is_named(root, world):
    made = plan(root, world)
    run = tampering(world, lambda: (site(root) / "beta.py").unlink())
    with pytest.raises(mover.StepFailed, match="beta.py was removed") as e:
        forward(root, world, made, run=run)
    assert [x for x in e.value.errors if "beta.py" in x]


def test_a_new_package_that_is_incomplete_takes_everything_back(root, world):
    before = tree(root)
    made = plan(root, world)
    run = tampering(world, lambda: (site(root) / "alpha" / "new.py").unlink())
    with pytest.raises(mover.StepFailed, match="alpha/new.py is missing") as e:
        forward(root, world, made, run=run)
    assert e.value.errors == [] and tree(root) == before


def test_a_new_file_that_is_cut_short_takes_everything_back(root, world):
    before = tree(root)
    made = plan(root, world)

    def cut():
        path = site(root) / "alpha" / "newdir" / "x.py"
        path.unlink()
        path.write_bytes(b"x")
    with pytest.raises(mover.StepFailed, match="newdir/x.py is incomplete") as e:
        forward(root, world, made, run=tampering(world, cut))
    assert e.value.errors == [] and tree(root) == before


def test_a_uv_killed_half_way_takes_everything_back(template, world, tmp_path):
    root = world.install(tmp_path / "root", ["big==1.0"])
    world.lock(root, ["big==2.0"])
    before = tree(root)
    made = plan(root, world)
    killed = []

    def run(command, cwd, env):
        p = subprocess.Popen([*command, "--link-mode", "copy"], cwd=cwd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **um.NO_WINDOW)
        while p.poll() is None and not (site(root) / "big" / "d1").exists():
            time.sleep(0.002)
        killed.append(p.poll() is None)
        p.kill()
        p.wait(timeout=30)
        return 1, "killed"
    with pytest.raises(mover.StepFailed, match="killed") as e:
        forward(root, world, made, run=run)
    assert killed == [True]
    assert e.value.errors == [] and tree(root) == before


@windows_only
@pytest.mark.parametrize("share", [0, 1, 3], ids=["no-sharing", "read", "read-write"])
def test_a_held_package_file_takes_everything_back(root, world, share):
    before = tree(root)
    made = plan(root, world)
    with held(site(root) / "alpha" / "olddir" / "x.py", share=share):
        with pytest.raises(mover.StepFailed, match=r"x\.py.*pid") as e:
            forward(root, world, made, wait_s=0.3)
    assert e.value.errors == [] and tree(root) == before


# ------------------------------------------------------------------------- kills

def test_a_kill_at_every_operation_of_the_package_step_leads_back(template, world, tmp_path):
    run_dir = um.applier(tmp_path / "applier")

    def spec(root: Path) -> dict:
        (root / ".localm-update" / ID / "new").mkdir(parents=True)
        return {"do": "update", "root": str(root), "id": ID, "uv": argv(world), "env": world.env}
    whole = copy(template, tmp_path / "count" / "root")
    old = tree(whole)
    done = um.drive(run_dir, spec(whole), -1)
    assert done.returncode == 0, done.stderr
    points = int(done.stdout.split("POINTS")[1])
    new = tree(whole)
    assert points > 30 and new != old
    reached_phases = set()
    for n in range(points):
        root = copy(template, tmp_path / f"k{n}" / "root")
        killed = um.drive(run_dir, spec(root), n)
        assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
        reached = um.phase(root)
        ended = "FINISH" in killed.stdout and reached in (job.DONE, None)
        again = um.restore(run_dir, root)
        assert again.returncode == 0, (n, reached, again.stderr)
        assert tree(root) == (new if ended else old), (n, reached)
        assert um.phase(root) is None
        if reached or ended:
            assert not (root / ".venv" / mover.VENV_FOLDER).exists(), n
        reached_phases.add(reached)
        shutil.rmtree(root.parent)
    assert {job.STEPS, job.STARTING, job.DONE} <= reached_phases


def test_a_kill_at_every_operation_of_the_way_back_of_packages_leads_back(template, world,
                                                                          tmp_path):
    run_dir = um.applier(tmp_path / "applier")
    first = copy(template, tmp_path / "first" / "root")
    old = tree(first)
    forward(first, world, plan(first, world))
    count = copy(first, tmp_path / "count" / "root")
    done = um.drive(run_dir, {"do": "restore", "root": str(count)}, -1)
    assert done.returncode == 0, done.stderr
    points = int(done.stdout.split("POINTS")[1])
    assert points > 20 and tree(count) == old
    for n in range(points):
        root = copy(first, tmp_path / f"r{n}" / "root")
        killed = um.drive(run_dir, {"do": "restore", "root": str(root)}, n)
        assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
        again = um.restore(run_dir, root)
        assert again.returncode == 0, (n, again.stderr)
        assert tree(root) == old, n
        shutil.rmtree(root.parent)


def test_the_way_back_waits_for_a_uv_that_outlived_the_update(world, tmp_path):
    """The applier is killed while uv still installs; uv goes on."""
    run_dir = um.applier(tmp_path / "applier")
    root = world.install(tmp_path / "x" / "root", ["big==1.0"])
    world.lock(root, ["big==2.0"])
    before = tree(root)
    (root / ".localm-update" / ID / "new").mkdir(parents=True)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"do": "update", "root": str(root), "id": ID,
                                "uv": [*argv(world), "--link-mode", "copy"], "env": world.env}),
                    encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-I", "-S", str(um.DRIVER), str(run_dir), str(spec), "-1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **um.NO_WINDOW)
    deadline = time.monotonic() + 60
    while not (site(root) / "big" / "d1").exists():
        assert p.poll() is None and time.monotonic() < deadline
        time.sleep(0.002)
    p.kill()
    p.wait(timeout=30)
    again = um.restore(run_dir, root)
    assert again.returncode == 0, again.stderr
    time.sleep(1)
    assert tree(root) == before
    assert um.phase(root) is None


def test_the_way_back_gives_up_while_uv_still_works_in_the_venv(root, world):
    made = plan(root, world)
    forward(root, world, made)
    now = tree(root)
    fd = disk.open_lock(root / ".venv" / ".lock")
    disk.lock_file(fd)
    try:
        errors = packages.back(root, ID, made, uv_wait_s=0.3)
    finally:
        os.close(fd)
    assert len(errors) == 1 and "still running" in errors[0]
    assert tree(root) == now
    assert packages.back(root, ID, made) == []


def test_uv_really_takes_the_lock_the_way_back_waits_for(root, world):
    """A uv that finds the lock taken waits and changes nothing until it is free."""
    fd = disk.open_lock(root / ".venv" / ".lock")
    disk.lock_file(fd)
    before = tree(root)
    p = subprocess.Popen(packages.command(argv(world)), cwd=root, env=world.env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **um.NO_WINDOW)
    try:
        time.sleep(2)
        assert p.poll() is None and tree(root) == before
    finally:
        os.close(fd)
        assert p.wait(timeout=60) == 0
    assert versions(root)["alpha"] == "2.0"


def test_packages_are_put_back_before_the_files_of_the_release(template, world, tmp_path):
    """A whole update: files, then packages, then a restore of both."""
    root = copy(template, tmp_path / "root")
    um.write(root, um.OLD)
    old_lock = (root / "uv.lock").read_bytes()
    before = tree(root)
    new_files = um.stage(root, um.NEW)
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        files = mover.plan_files(root, ID, new_files, um.stated(um.OLD), swap=um.SWAP,
                                 protect=())
        mover.write_plan(root, ID, files)
        mover.forward(root, ID, files)
        job.advance(lock, job.STEPS)
        made = plan(root, world)
        assert made is not None and (files.number, made.number) == (1, 2)
        forward(root, world, made)
        assert (root / "VERSION").read_text(encoding="utf-8") == "2"
        assert versions(root)["alpha"] == "2.0"
        result = job.restore(lock)
    assert (result.state, result.errors) == ("old", [])
    assert tree(root) == before and (root / "uv.lock").read_bytes() == old_lock
