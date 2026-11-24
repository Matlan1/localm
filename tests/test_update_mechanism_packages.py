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

from localm.update_mechanism import job, mover, packages
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
    return um.tree(root, skip=(".localm-update", ".venv/.lock", ".venv/.localm-old",
                               ".venv/.localm-kept"))


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
    if not packages.stage_packages(root, ID, command or argv(world), env=world.env, **kw):
        return None
    made = packages.plan_packages(root, ID)
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


def test_a_wheel_that_cannot_be_had_offline_is_refused_before_anything_moves(tmp_path):
    own = World(tmp_path / "own")
    root = tmp_path / "root"
    wheel(own.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 1\n"})
    wheel(own.wheels, "alpha", "2.0", {"alpha/__init__.py": "V = 2\n"})
    own.install(root, ["alpha==1.0"])
    own.lock(root, ["alpha==2.0"])
    (own.wheels / "alpha-2.0-py3-none-any.whl").unlink()
    before = tree(root)
    with pytest.raises(mover.Refused, match="uv could not install"):
        plan(root, own)
    assert tree(root) == before


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


def test_a_uv_that_is_itself_replaced_is_replaced_and_put_back(tmp_path):
    own = World(tmp_path / "own")
    root = tmp_path / "root"
    wheel(own.wheels, "tooluv", "1.0", {"tooluv.py": ""},
          {f"uvcopy{EXE}": Path(own.uv).read_bytes()})
    wheel(own.wheels, "tooluv", "2.0", {"tooluv.py": ""},
          {f"uvcopy{EXE}": Path(own.uv).read_bytes() + b"\0"})
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
    with pytest.raises(mover.Refused, match="alpha cannot be replaced safely.*is a link"):
        plan(root, world)
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


def test_uv_runs_offline_and_leaves_the_project_out_unless_asked():
    assert packages.command(["uv", "sync", "--frozen"]) == [
        "uv", "sync", "--frozen", "--offline", "--no-install-project"]
    assert packages.command(["uv", "sync"], project=True) == ["uv", "sync", "--offline"]
    assert packages.command(["uv", "sync"], network=True) == ["uv", "sync", "--no-install-project"]
    assert packages.command(["uv", "pip", "install", "torch"]) == [
        "uv", "pip", "install", "torch", "--offline"]


def test_the_lines_of_a_dry_run_are_read():
    text = ("Would use project environment at: .venv\nWould download 1 package\n"
            "Would uninstall 2 packages\nWould install 3 packages\n - Alpha_X==1.0\n"
            " + alpha-x==2.0 (from file:///w)\n ~ beta==1.0\n + localm @ file:///a b/c\n")
    assert packages.changes(text) == (
        [("alpha-x", "1.0", False), ("beta", "1.0", False)],
        [("alpha-x", "2.0", True), ("beta", "1.0", False), ("localm", "", True)])


# ---------------------------------------------------------- files of somebody else

def test_wheels_that_share_a_file_are_replaced_and_put_back(tmp_path):
    """beta, which stays, and the new alpha both ship tests/__init__.py."""
    own = World(tmp_path / "own")
    wheel(own.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 1\n"})
    wheel(own.wheels, "alpha", "2.0", {"alpha/__init__.py": "V = 2\n",
                                       "tests/__init__.py": "# of alpha\n"})
    wheel(own.wheels, "beta", "1.0", {"beta.py": "V = 1\n", "tests/__init__.py": "# of beta\n"})
    root = own.install(tmp_path / "root", ["alpha==1.0", "beta==1.0"])
    own.lock(root, ["alpha==2.0", "beta==1.0"])
    before = tree(root)
    forward(root, own, plan(root, own))
    assert (site(root) / "tests" / "__init__.py").read_text(encoding="utf-8") == "# of alpha\n"
    result = undo(root)
    assert (result.state, result.errors, tree(root) == before) == ("old", [], True)
    forward(root, own, plan(root, own))
    assert end(root).notes == []
    assert versions(root) == {"alpha": "2.0", "beta": "1.0"}


def test_a_file_of_the_user_in_the_place_of_a_new_one_is_kept(tmp_path):
    own = World(tmp_path / "own")
    wheel(own.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 1\n"})
    wheel(own.wheels, "alpha", "3.0", {"alpha/__init__.py": "V = 3\n", "helper.py": "# of alpha\n"})
    root = own.install(tmp_path / "root", ["alpha==1.0"])
    mine = site(root) / "helper.py"
    mine.write_bytes(b"# mine")
    own.lock(root, ["alpha==3.0"])
    before = tree(root)
    forward(root, own, plan(root, own))
    assert mine.read_bytes() == b"# of alpha\n"
    result = undo(root)
    assert (result.state, result.errors, tree(root) == before) == ("old", [], True)
    forward(root, own, plan(root, own))
    result = end(root)
    kept = root / ".venv" / mover.VENV_KEPT / ID / site(root).relative_to(root / ".venv")
    assert (kept / "helper.py").read_bytes() == b"# mine"
    assert result.state == "new" and len(result.notes) == 1 and ID in result.notes[0]


def test_an_edited_file_of_a_package_is_kept(root, world):
    edited = site(root) / "alpha" / "old.py"
    time.sleep(0.05)
    edited.write_bytes(b"x = 9\n")
    same_size = site(root) / "alpha" / "data.txt"
    same_size.write_bytes(b"ONE\n")
    before = tree(root)
    made = plan(root, world)
    assert {op.path.rpartition("/")[2] for op in made.ops if op.edited} == {"old.py", "data.txt"}
    forward(root, world, made)
    assert undo(root).state == "old" and tree(root) == before
    forward(root, world, plan(root, world))
    result = end(root)
    kept = root / ".venv" / mover.VENV_KEPT / ID / site(root).relative_to(root / ".venv") / "alpha"
    assert (kept / "old.py").read_bytes() == b"x = 9\n"
    assert (kept / "data.txt").read_bytes() == b"ONE\n"
    assert "2 file(s)" in result.notes[0]


def everything(root: Path) -> set:
    return {p.read_bytes() for p in (root / ".venv").rglob("*") if p.is_file()}


def test_what_the_user_puts_into_the_venv_meanwhile_stays(root, world):
    wheel(world.wheels, "extra", "1.0", {"extra.py": "EXTRA = 1\n"})
    forward(root, world, plan(root, world))
    world.run(root, "pip", "install", "--no-config", "--no-index", "--offline", "--find-links",
              str(world.wheels), "--python", str(root / ".venv"), "extra==1.0")
    (site(root) / "note.txt").write_bytes(b"a note")
    (root / ".venv" / "mydata").write_bytes(b"my data")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert versions(root)["alpha"] == "1.0" and versions(root)["extra"] == "1.0"
    assert (site(root) / "note.txt").read_bytes() == b"a note"
    assert (root / ".venv" / "mydata").read_bytes() == b"my data"
    assert imported(root, "import extra; print(extra.EXTRA)") == "1"


def test_another_package_upgraded_meanwhile_is_left_alone(root, world):
    wheel(world.wheels, "beta", "2.0", {"beta.py": "V = 2\n", "beta2.py": "", "shared_ns/b.py": ""})
    forward(root, world, plan(root, world))
    world.run(root, "pip", "install", "--no-config", "--no-index", "--offline", "--find-links",
              str(world.wheels), "--python", str(root / ".venv"), "beta==2.0")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert versions(root)["alpha"] == "1.0" and versions(root)["beta"] == "2.0"
    assert (site(root) / "beta2.py").exists()
    assert imported(root, "import beta, beta2; print(beta.V)") == "2"


@pytest.mark.parametrize("version", ["1.0", "2.0", "3.0"])
def test_a_package_of_the_update_installed_anew_meanwhile_is_decided_as_a_whole(root, world, version):
    wheel(world.wheels, "alpha", "3.0", {"alpha/__init__.py": "V = 3\n", "alpha/three.py": "3\n",
                                         "alpha3_top.py": "top = 3\n"})
    before = tree(root)
    forward(root, world, plan(root, world))
    world.run(root, "pip", "install", "--no-config", "--no-index", "--offline", "--reinstall",
              "--find-links", str(world.wheels), "--python", str(root / ".venv"),
              f"alpha=={version}")
    theirs = {p.read_bytes() for p in (site(root) / "alpha").rglob("*.py")}
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert tree(root) == before
    if version != "2.0":
        assert theirs <= everything(root)
        assert any("kept" in note for note in result.notes)
    if version == "3.0":
        assert any("alpha-3.0" in note for note in result.notes)
        kept = root / ".venv" / mover.VENV_KEPT / ID / site(root).relative_to(root / ".venv")
        assert (kept / "alpha-3.0.dist-info" / "RECORD").exists()
        assert (kept / "alpha3_top.py").exists() and (kept / "alpha" / "three.py").exists()


def test_a_package_removed_by_the_user_meanwhile_comes_back(root, world):
    before = tree(root)
    forward(root, world, plan(root, world))
    world.run(root, "pip", "uninstall", "--no-config", "--python", str(root / ".venv"), "alpha")
    result = undo(root)
    assert (result.state, result.errors) == ("old", [])
    assert tree(root) == before


# ------------------------------------------------------------------ the staging venv

def test_staging_changes_nothing_in_the_venv_and_holds_only_what_is_new(root, world):
    before = tree(root)
    begin(root)
    assert packages.stage_packages(root, ID, argv(world), env=world.env) is True
    assert tree(root) == before
    stage = packages.stage_folder(root, ID)
    assert os.lstat(stage).st_dev == os.lstat(site(root)).st_dev
    staged = stage / site(root).relative_to(root / ".venv")
    assert not any((staged / n).exists() for n in ("beta.py", "delta", "shared_ns/b.py"))
    assert sorted(p.name for p in (staged / "beta-1.0.dist-info").iterdir()) == sorted(
        n for n in packages._METADATA if (site(root) / "beta-1.0.dist-info" / n).exists())


def test_a_later_command_is_staged_on_top_of_an_earlier_one(root, world):
    """The second command sees what the first staged, replaces a part of it, and
    takes another package of the venv along."""
    wheel(world.wheels, "alpha", "2.5", {"alpha/__init__.py": "V = 2.5\n"})
    wheel(world.wheels, "delta", "2.0", {"delta/__init__.py": "V = 2\n"})
    before = tree(root)
    begin(root)
    pip = [world.uv, "pip", "install", "--no-config", "--no-index", "--find-links",
           str(world.wheels), "--python", str(root / ".venv")]
    nothing = [*pip, "beta==1.0"]
    last = [*pip, "alpha==2.5", "delta==2.0"]
    assert packages.stage_packages(root, ID, nothing, env=world.env, more=True) is False
    assert packages.stage_packages(root, ID, argv(world), env=world.env, more=True)
    assert packages.stage_packages(root, ID, last, env=world.env, more=True)
    assert packages.stage_packages(root, ID, nothing, env=world.env, more=True) is False
    assert tree(root) == before
    made = packages.plan_packages(root, ID)
    mover.write_plan(root, ID, made)
    assert made.dists == ["alpha", "delta", "gamma"]
    forward(root, world, made, last)
    assert versions(root) == {"alpha": "2.5", "beta": "1.0", "gamma": "1.0", "delta": "2.0"}
    assert not (site(root) / "alpha" / "new.py").exists()
    assert undo(root).state == "old" and tree(root) == before


def test_a_package_that_changed_after_staging_is_refused(root, world):
    begin(root)
    packages.stage_packages(root, ID, argv(world), env=world.env)
    (site(root) / "alpha-1.0.dist-info").rename(site(root) / "alpha-1.1.dist-info")
    with pytest.raises(mover.Refused, match="changed since the update was prepared"):
        packages.plan_packages(root, ID)


def test_nothing_staged_is_refused(root):
    begin(root)
    with pytest.raises(mover.Refused):
        packages.plan_packages(root, ID)


def put(path: Path, data: bytes) -> None:
    """Write *path* anew; a staged file can be one file with uv's cache."""
    path.unlink(missing_ok=True)
    path.write_bytes(data)


def staging(change):
    """Run uv for real, then *change* the staging venv as if uv had done it."""
    def run(command, cwd, env):
        done = packages.run_hidden(command, cwd, env)
        if "--dry-run" not in command and command[1] != "venv":
            change(Path(env["UV_PROJECT_ENVIRONMENT"]))
        return done
    return run


@pytest.mark.parametrize(("change", "why"), [
    (lambda s: (s / "stray.py").write_bytes(b"x"), "other files"),
    (lambda s: next(s.rglob("alpha/new.py")).unlink(), "other files|not staged"),
    (lambda s: put(next(s.rglob("alpha/new.py")), b"cut"), "not staged completely"),
    (lambda s: shutil.rmtree(next(s.rglob("gamma-1.0.dist-info"))), "other packages"),
    (lambda s: put(next(s.rglob("alpha-2.0.dist-info/INSTALLER")), b"u"), "not staged completely"),
    (lambda s: put(next(s.rglob("alpha-2.0.dist-info/RECORD")), b"\0" * 200000), "RECORD"),
], ids=["stray", "missing", "cut", "package", "installer", "zeroed-record"])
def test_a_staging_venv_that_is_not_the_plan_is_refused(root, world, change, why):
    before = tree(root)
    with pytest.raises(mover.Refused, match=why):
        plan(root, world, run=staging(change))
    assert tree(root) == before


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
def test_a_staged_file_that_names_the_staging_folder_is_refused(root, world, encoding):
    def change(stage: Path) -> None:
        info = next(stage.rglob("alpha-2.0.dist-info"))
        put(info / "direct_url.json", stage.as_posix().replace("/", os.sep).encode(encoding))
    with pytest.raises(mover.Refused, match="direct_url.json names the staging folder"):
        plan(root, world, run=staging(change))


def test_a_zeroed_record_in_the_venv_is_refused_by_name(root, world):
    (site(root) / "alpha-1.0.dist-info" / "RECORD").write_bytes(b"\0" * 200000)
    with pytest.raises(mover.Refused, match="alpha has no"):
        plan(root, world)


def test_a_package_from_a_folder_of_this_machine_is_refused_by_name(root, world, tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    wheel(local, "omega", "1.0", {"omega.py": ""})
    command = [world.uv, "pip", "install", "--no-config", "--no-index", "--python",
               str(root / ".venv"), str(local / "omega-1.0-py3-none-any.whl")]
    before = tree(root)
    with pytest.raises(mover.Refused, match="omega comes from a folder or file"):
        plan(root, world, command)
    assert tree(root) == before


def test_uv_pip_install_is_staged_like_a_sync(root, world):
    before = tree(root)
    command = [world.uv, "pip", "install", "--no-config", "--no-index", "--find-links",
               str(world.wheels), "--python", str(root / ".venv"), "alpha==2.0", "beta"]
    forward(root, world, plan(root, world, command), command)
    assert versions(root) == {"alpha": "2.0", "beta": "1.0", "delta": "1.0"}
    assert undo(root).state == "old" and tree(root) == before


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("uv", 1), KeyboardInterrupt(),
                                   RuntimeError("the runner broke")], ids=["timeout", "ctrl-c", "error"])
def test_an_exception_of_the_runner_goes_through_the_way_back(root, world, error):
    before = tree(root)
    made = plan(root, world)

    def run(command, cwd, env):
        assert versions(root)["alpha"] == "2.0"
        raise error
    with pytest.raises(mover.StepFailed) as e:
        forward(root, world, made, run=run)
    assert e.value.errors == [] and tree(root) == before


def test_packages_uv_would_still_change_take_everything_back(root, world):
    before = tree(root)
    made = plan(root, world)
    command = [world.uv, "pip", "install", "--no-config", "--no-index", "--find-links",
               str(world.wheels), "--python", str(root / ".venv"), "alpha==1.0"]
    with pytest.raises(mover.StepFailed, match="uv would still change packages: alpha") as e:
        forward(root, world, made, command)
    assert e.value.errors == [] and tree(root) == before


# ----------------------------------------------------- the same as a direct install

KITCHEN = {"kitchen/__init__.py": "def main():\n    import sys; print('kitchen', sys.prefix)\n"
                                  "def gui():\n    import sys, pathlib; "
                                  "pathlib.Path(sys.argv[1]).write_text('gui ' + sys.prefix)\n",
           "kitchen_top.py": "", "kitchen.pth": "import sys; sys.kitchen_pth = 1\n"}


def kitchen(folder: Path, version: str, uv_binary: bytes) -> None:
    info = f"kitchen-{version}.dist-info"
    data = f"kitchen-{version}.data"
    wheel(folder, "kitchen", version, {
        **KITCHEN, "kitchen/version.txt": version,
        f"{info}/entry_points.txt": "[console_scripts]\nlocalm = kitchen:main\n"
                                    "localcoder = kitchen:main\n[gui_scripts]\nwin = kitchen:gui\n",
        f"{data}/data/share/kitchen/data.txt": version,
        f"{data}/headers/kitchen.h": "#define K 1\n",
        f"{data}/platlib/kitchen_plat.py": "P = 1\n"},
        {"kitchen-run": f"#!python\nimport sys; print('run {version}', sys.prefix)\n",
         f"uvtool{EXE}": uv_binary})


def behaviour(root: Path) -> dict:
    """What each script of the venv does, its own place left out."""
    bin_dir = root / ".venv" / SCRIPTS
    prefix = str((root / ".venv").resolve())
    out = {}
    for name in ("localm", "localcoder", "kitchen-run"):
        argv_ = ([str(bin_dir / f"python{EXE}"), str(bin_dir / name)] if name == "kitchen-run" and WINDOWS
                 else [str(bin_dir / f"{name}{'' if name == 'kitchen-run' else EXE}")])
        done = subprocess.run(argv_, capture_output=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL,
                              cwd=root, env={**os.environ, "PYTHONUTF8": "1"}, **um.NO_WINDOW)
        assert done.returncode == 0, (name, done.stderr)
        out[name] = done.stdout.strip().replace(prefix, "PREFIX")
    result = root / "gui.txt"
    subprocess.run([str(bin_dir / f"win{EXE}"), str(result)], timeout=60, check=True, cwd=root,
                   env={**os.environ, "PYTHONUTF8": "1"}, **um.NO_WINDOW)
    deadline = time.monotonic() + 30
    while not (result.exists() and result.read_text(encoding="utf-8")) and time.monotonic() < deadline:
        time.sleep(0.05)
    out["win"] = result.read_text(encoding="utf-8").replace(prefix, "PREFIX")
    result.unlink()
    out["uvtool"] = subprocess.run([str(bin_dir / f"uvtool{EXE}"), "--version"], capture_output=True,
                                   text=True, timeout=60, check=True, **um.NO_WINDOW).stdout.strip()
    return out


def test_the_staged_install_is_the_direct_install(tmp_path):
    """File by file in a path with spaces and umlauts, the scripts by what they do,
    also after the whole install was moved."""
    place = tmp_path / "Meine Dateien \u00e4\u00f6\u00fc"
    place.mkdir()
    own = World(place / "own")
    binary = Path(own.uv).read_bytes()
    kitchen(own.wheels, "1.0", binary)
    kitchen(own.wheels, "2.0", binary)
    wheel(own.wheels, "stays", "1.0", {"stays.py": ""}, {"stays-run": "#!python\nprint('stays')\n"})
    roots = {}
    for name in ("staged", "direct"):
        roots[name] = own.install(place / name, ["kitchen==1.0", "stays==1.0"])
        own.lock(roots[name], ["kitchen==2.0", "stays==1.0"])
    staged, direct = roots["staged"], roots["direct"]
    old_stays = (staged / ".venv" / SCRIPTS / "stays-run").read_bytes()
    forward(staged, own, plan(staged, own))
    assert end(staged).state == "new"
    own.run(direct, *SYNC, "--offline", "--no-install-project")
    a = tree(staged)
    b = tree(direct)
    assert set(a) == set(b)
    scripts = {f".venv/{SCRIPTS}/{n}" for n in (f"localm{EXE}", f"localcoder{EXE}", f"win{EXE}",
                                                "kitchen-run")}
    other = {k for k in a if a[k] != b[k]}
    record = f".venv/{(site(staged).relative_to(staged / '.venv')).as_posix()}/kitchen-2.0.dist-info/RECORD"
    venv_own = {k for k in other if k.startswith((".venv/pyvenv.cfg", f".venv/{SCRIPTS}/activate",
                                                  ".venv/CACHEDIR", ".venv/.gitignore"))}
    stays = {f".venv/{SCRIPTS}/stays-run"}
    cache = {k for k in other if k.endswith("uv_cache.json") or k in ("uv.lock", "pyproject.toml")}
    stays_record = {k for k in other if "stays-1.0.dist-info" in k}
    assert other - venv_own - cache - stays - stays_record <= scripts | {record}
    assert (staged / ".venv" / SCRIPTS / "stays-run").read_bytes() == old_stays

    def rows(root):
        return sorted(tuple(r[:1]) for r in packages.shared.updater_components.record_rows(
            site(root), "kitchen-2.0.dist-info"))
    assert rows(staged) == rows(direct)
    for row in packages.shared.updater_components.record_rows(site(staged), "kitchen-2.0.dist-info"):
        path = site(staged) / row[0]
        if len(row) > 2 and row[1]:
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            assert row[1] == "sha256=" + digest.rstrip(b"=").decode(), row
    assert behaviour(staged) == behaviour(direct)
    assert all("2.0" in v or "PREFIX" in v for k, v in behaviour(staged).items() if k != "uvtool")
    assert "compatible" in own.run(staged, "pip", "check", "--python", str(staged / ".venv"))
    said = own.run(staged, *SYNC, "--offline", "--no-install-project", "--dry-run")
    assert "Would make no changes" in said
    expected = behaviour(staged)
    moved = place / "anderswo \u00df"
    staged.rename(moved)
    assert behaviour(moved) == expected
    assert (moved / ".venv" / "share" / "kitchen" / "data.txt").read_text(encoding="utf-8") == "2.0"


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

    def one(n: int):
        root = copy(template, tmp_path / f"k{n}" / "root")
        killed = um.drive(run_dir, spec(root), n)
        assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
        reached = um.phase(root)
        ended = "FINISH" in killed.stdout and (um.done(root) or reached in (job.DONE, None))
        again = um.restore(run_dir, root)
        assert again.returncode == 0, (n, reached, again.stderr)
        if ended:
            assert set(tree(root)) == set(new) and versions(root)["alpha"] == "2.0", (n, reached)
        else:
            assert tree(root) == old, (n, reached)
        assert um.phase(root) is None
        if reached or ended:
            assert not (root / ".venv" / mover.VENV_FOLDER).exists(), n
        shutil.rmtree(root.parent)
        return reached
    assert {job.STEPS, job.STARTING, job.DONE} <= set(um.each(points, one))


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

    def one(n: int) -> None:
        root = copy(first, tmp_path / f"r{n}" / "root")
        killed = um.drive(run_dir, {"do": "restore", "root": str(root)}, n)
        assert killed.returncode != 0 and "POINTS" not in killed.stdout, (n, killed.stderr)
        again = um.restore(run_dir, root)
        assert again.returncode == 0, (n, again.stderr)
        assert tree(root) == old, n
        shutil.rmtree(root.parent)
    um.each(points, one)


def test_a_uv_that_is_killed_leaves_nothing_in_the_venv(template, world, tmp_path):
    root = world.install(tmp_path / "root", ["big==1.0"])
    world.lock(root, ["big==2.0"])
    before = tree(root)
    begin(root)
    killed = []

    def run(command, cwd, env):
        if "--dry-run" in command or command[1] == "venv":
            return packages.run_hidden(command, cwd, env)
        stage = Path(env["UV_PROJECT_ENVIRONMENT"])
        p = subprocess.Popen([*command, "--link-mode", "copy"], cwd=cwd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **um.NO_WINDOW)
        while p.poll() is None and not next(stage.glob("[Ll]ib/**/big/d1"), None):
            time.sleep(0.002)
        killed.append(p.poll() is None)
        p.kill()
        p.wait(timeout=30)
        return 1, "killed"
    with pytest.raises(mover.Refused, match="killed"):
        packages.stage_packages(root, ID, argv(world), env=world.env, run=run)
    assert killed == [True] and tree(root) == before
    assert undo(root).state == "old"
    assert tree(root) == before and not (root / ".venv" / mover.VENV_FOLDER).exists()


def test_an_update_that_was_only_prepared_is_discarded(root, world):
    before = tree(root)
    (root / ".localm-update" / ID / "new").mkdir(parents=True)
    packages.stage_packages(root, ID, argv(world), env=world.env)
    with job.Lock(root) as lock:
        job.cleanup(lock)
        assert packages.stage_folder(root, ID).exists()
        assert job.discard(lock, ID) is True
    assert tree(root) == before and um.left(root) == []
    assert not (root / ".venv" / mover.VENV_FOLDER / ID).exists()


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
