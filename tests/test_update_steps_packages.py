# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Python packages of an update are fetched before anything changes and installed
without a network: a real uv, a real venv and an index on this machine that is shut
down in between."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover, steps, steps_packages, steps_update
from localm.update_mechanism.mover import Refused
from tests import _update_mechanism as um
from tests.test_update_mechanism_packages import World, wheel

pytest.importorskip("uv")
SITE = ("Lib/site-packages" if os.name == "nt"
        else f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages")
ID = um.UPDATE_ID
SKIP = (".localm-update", ".venv/.localm-old", "venv/.localm-old", ".venv/.lock", "venv/.lock")


class Index:
    """A package index on 127.0.0.1 that serves the wheels of a folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        handler = partial(_Pages, folder)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/simple"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Pages(http.server.BaseHTTPRequestHandler):
    def __init__(self, folder: Path, *args) -> None:
        self.folder = folder
        super().__init__(*args)

    def do_GET(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        name = parts[-1] if parts else ""
        if name.endswith(".whl"):
            body, kind = (self.folder / name).read_bytes(), "application/octet-stream"
        else:
            found = sorted(p.name for p in self.folder.glob("*.whl")
                           if p.name.split("-")[0].replace("_", "-") == name)
            sums = {f: hashlib.sha256((self.folder / f).read_bytes()).hexdigest() for f in found}
            links = "".join(f'<a href="/files/{f}#sha256={sums[f]}">{f}</a>' for f in found)
            body, kind = f"<html><body>{links}</body></html>".encode(), "text/html"
            if not found:
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path / "world")
    for name in ("alpha", "beta", "extra"):
        for version in ("1.0", "2.0"):
            wheel(w.wheels, name, version, {f"{name}/__init__.py": f"V = {version!r}\n",
                                            f"{name}/v{version[0]}.py": "x = 1\n"})
    index = Index(w.wheels)
    w.index = index
    yield w
    if index.server.socket.fileno() != -1:
        index.stop()


def _project(world, folder: Path, version: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "VERSION").write_text(version, encoding="utf-8")
    v = f"=={version}.0"
    (folder / "pyproject.toml").write_text(
        f'[project]\nname = "localm"\nversion = "{version}"\nrequires-python = ">=3.12"\n'
        f'dependencies = ["alpha{v}", "beta==1.0"]\n'
        f'[project.optional-dependencies]\nused = ["extra{v}"]\n'
        'coder = []\nvoice = []\nmonitor = []\n'
        f'[[tool.uv.index]]\nurl = {json.dumps(world.index.url)}\ndefault = true\n',
        encoding="utf-8")
    world.run(folder, "lock", "--python", sys.executable)


def _install(world, tmp_path) -> Path:
    root = tmp_path / "root"
    _project(world, root, "1")
    world.run(root, "venv", "--python", sys.executable)
    world.run(root, "sync", "--frozen", "--inexact", "--no-config", "--no-install-project",
              "--extra", "used")
    new = root / ".localm-update" / ID / "new"
    _project(world, new, "2")
    (root / ".localm-update" / ID / "plans").mkdir()
    return root


def _future(world, root, said=None, id=ID):
    return steps_packages.Future(root, id, world.uv, env=world.env,
                                 tree=root / ".localm-update" / ID / "new",
                                 say=(said.append if said is not None else (lambda _t: None)))


def _versions(root: Path) -> dict:
    site = steps_packages.shared.updater_components.site_packages(root / ".venv")
    return {n: v for n, v, _f in steps_packages.shared.updater_components.dist_infos(site)}


def test_the_sync_command_is_the_one_localm_uses():
    from localm import pyinstall
    assert steps_packages.sync_argv("uv", ["a", "b"]) == pyinstall.sync_command("uv", ["a", "b"])


def test_the_extras_in_use_are_the_ones_whose_packages_are_there(world, tmp_path):
    root = _install(world, tmp_path)
    new = root / ".localm-update" / ID / "new"
    used = steps_packages.used_extras(root, new, world.uv, env=world.env)
    assert used[:3] == ["coder", "voice", "monitor"] and "used" in used
    site = steps_packages.shared.updater_components.site_packages(root / ".venv")
    shutil.rmtree(site / "extra")
    for info in (root / ".venv").rglob("extra-*.dist-info"):
        shutil.rmtree(info)
    assert "used" not in steps_packages.used_extras(root, new, world.uv, env=world.env)


def _steps(lock):
    job.start(lock, ID, old="1", new="2")
    job.advance(lock, job.MOVING)
    job.advance(lock, job.STEPS)


def test_staged_packages_are_moved_in_without_a_network_and_put_back(world, tmp_path):
    root = _install(world, tmp_path)
    before = um.tree(root, skip=SKIP)
    said: list = []
    future = _future(world, root, said)
    step = future.sync(["used"])
    assert step == {"extras": ["used"], "project": False}
    assert um.tree(root, skip=SKIP) == before
    assert any("alpha 2.0" in line and "beta" not in line for line in said)
    world.index.stop()
    release = ("pyproject.toml", "uv.lock", "VERSION")
    installed = {name: (root / name).read_bytes() for name in release}
    with job.Lock(root) as lock:
        _steps(lock)
        for name in release:
            shutil.copyfile(future.tree / name, root / name)
        steps_packages.run_packages(root, ID, step, [], world.uv, env=world.env)
        assert _versions(root) == {"alpha": "2.0", "beta": "1.0", "extra": "2.0"}
        for name, data in installed.items():
            (root / name).write_bytes(data)
        assert job.restore(lock).errors == []
    assert um.tree(root, skip=SKIP) == before


def test_a_later_command_sees_the_packages_staged_before_it(world, tmp_path):
    root = _install(world, tmp_path)
    future = _future(world, root)
    step = future.sync(["used"])
    more = ["pip", "install", "--python", str(root / ".venv"), "gamma==1.0", "alpha", "--default-index",
            world.index.url]
    wheel(world.wheels, "gamma", "1.0", {"gamma/__init__.py": "V = 1\n"})
    assert future.add([world.uv, *more])
    assert not future.add([world.uv, *more])
    world.index.stop()
    with job.Lock(root) as lock:
        _steps(lock)
        steps_packages.run_packages(root, ID, step, [more], world.uv, env=world.env)
        assert _versions(root) == {"alpha": "2.0", "beta": "1.0", "extra": "2.0", "gamma": "1.0"}


def test_packages_that_were_not_staged_change_nothing(world, tmp_path):
    root = _install(world, tmp_path)
    before = um.tree(root)
    with job.Lock(root) as lock, pytest.raises(Refused, match="no packages are staged"):
        _steps(lock)
        steps_packages.run_packages(root, ID, {"extras": [], "project": False}, [], world.uv,
                                    env=world.env)
    assert um.tree(root) == before


def test_a_release_with_the_same_packages_has_no_package_step(world, tmp_path):
    root = _install(world, tmp_path)
    new = root / ".localm-update" / ID / "new"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(root / name, new / name)
    world.index.stop()
    future = _future(world, root)
    assert future.sync(["used"]) is None
    assert not future.staged and future.python == steps_packages.python_in(root / ".venv")


def _runs(future, tmp_path):
    code = "import alpha, beta, extra; print(alpha.V, beta.V, extra.V)"
    status, out = steps_packages.packages.run_hidden([str(future.python), "-c", code],
                                                     str(tmp_path), dict(os.environ))
    return status, out.split()


def test_the_new_code_runs_with_the_new_packages_and_every_other_one(world, tmp_path):
    root = _install(world, tmp_path)
    future = _future(world, root)
    assert _runs(future, tmp_path) == (0, ["1.0", "1.0", "1.0"])
    future.sync([])
    assert _runs(future, tmp_path) == (0, ["2.0", "1.0", "1.0"])
    future.sync(["used"])
    assert _runs(future, tmp_path) == (0, ["2.0", "1.0", "2.0"])


def test_packages_that_do_not_fit_on_the_disk_are_refused_before_they_are_fetched(
        world, tmp_path, monkeypatch):
    root = _install(world, tmp_path)
    before = um.tree(root)
    monkeypatch.setattr(steps, "SPARE_BYTES", 1 << 60)
    with pytest.raises(Refused, match="needs .* on the disk"):
        _future(world, root).sync(["used"])
    assert um.tree(root) == before


def test_a_changed_console_script_is_noticed(world, tmp_path):
    root = _install(world, tmp_path)
    new = root / ".localm-update" / ID / "new"
    text = (new / "pyproject.toml").read_text(encoding="utf-8")
    assert steps_packages.scripts(root) == steps_packages.scripts(new)
    (new / "pyproject.toml").write_text(
        text.replace("[project.optional", '[project.scripts]\nx = "alpha:main"\n[project.optional'),
        encoding="utf-8")
    assert steps_packages.scripts(root) != steps_packages.scripts(new)


# ------------------------------------------------- the old versions once more

def _again(world, root, wanted, **kw):
    steps_packages.reinstall(root, ID, mover.PACKAGES, wanted, world.uv,
                             env=kw.pop("env", world.env), **kw)


def test_the_old_versions_are_installed_from_the_lock_with_its_checksums(world, tmp_path):
    root = _install(world, tmp_path)
    alpha = root / ".venv" / SITE / "alpha"
    (alpha / "v1.py").unlink()
    (alpha.parent / "alpha-1.0.dist-info" / "METADATA").unlink()
    _again(world, root, [("alpha", "1.0"), ("beta", "1.0")])
    assert (alpha / "v1.py").is_file() and _versions(root)["alpha"] == "1.0"
    world.index.stop()
    (alpha / "v1.py").unlink()
    _again(world, root, [("alpha", "1.0")], network=False)
    assert (alpha / "v1.py").is_file()


def test_a_package_of_the_lock_with_another_content_is_refused(world, tmp_path):
    root = _install(world, tmp_path)
    wheel(world.wheels, "alpha", "1.0", {"alpha/__init__.py": "V = 'another'\n"})
    env = {**world.env, "UV_CACHE_DIR": str(tmp_path / "another-cache")}
    with pytest.raises(Refused, match="alpha 1.0 could not be installed again"):
        _again(world, root, [("alpha", "1.0")], env=env)


def test_a_package_the_lock_does_not_name_comes_from_the_given_index(world, tmp_path):
    root = _install(world, tmp_path)
    wheel(world.wheels, "gamma", "1.0", {"gamma/__init__.py": "V = 1\n"})
    indexes = steps_packages.index_args([["uv", "pip", "install", "gamma", "--index-url",
                                          world.index.url]])
    assert indexes == ["--extra-index-url", world.index.url]
    _again(world, root, [("gamma", "1.0")], indexes=indexes)
    assert _versions(root)["gamma"] == "1.0"


@pytest.mark.parametrize("wanted", [("alpha", "1.0\nbeta"), ("-r x", "1.0"), ("alpha", "")])
def test_a_name_that_is_no_package_is_refused(world, tmp_path, wanted):
    root = _install(world, tmp_path)
    with pytest.raises(Refused, match="no packages"):
        _again(world, root, [wanted])


def test_a_venv_that_is_gone_is_made_from_the_lock(world, tmp_path):
    root = _install(world, tmp_path)
    shutil.rmtree(root / ".venv")
    _again(world, root, [("alpha", "1.0")], extras=["used"])
    assert _versions(root) == {"alpha": "1.0", "beta": "1.0", "extra": "1.0"}


def test_a_library_that_is_loaded_does_not_stop_the_old_version(world, tmp_path):
    source = next((p for p in sorted(Path(sys.base_prefix).rglob("select*"))
                   if p.suffix in (".pyd", ".so")), None)
    if source is None:
        pytest.skip("no extension module to load")
    for version in ("1.0", "2.0"):
        wheel(world.wheels, "alpha", version, {"alpha/__init__.py": f"V = {version!r}\n",
                                               f"alpha/{source.name}": source.read_bytes()})
    root = _install(world, tmp_path)
    alpha = root / ".venv" / SITE / "alpha"
    held = subprocess.Popen(
        [sys.executable, "-c", "import importlib.util as u, sys\n"
         "s = u.spec_from_file_location('select', sys.argv[1])\n"
         "m = u.module_from_spec(s); s.loader.exec_module(m)\n"
         "print('up', flush=True); sys.stdin.read()", str(alpha / source.name)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, **um.NO_WINDOW)
    try:
        assert held.stdout.readline().strip() == "up"
        (alpha / "__init__.py").unlink()
        _again(world, root, [("alpha", "1.0")])
        assert (alpha / "__init__.py").read_text() == "V = '1.0'\n"
        assert (alpha / source.name).read_bytes() == source.read_bytes()
    finally:
        held.stdin.close()
        held.wait(timeout=30)


def _updated(world, root) -> dict:
    """Update the packages of *root* and leave the update in its steps. Returns the
    release files as they were."""
    step = _future(world, root).sync(["used"])
    release = {n: (root / n).read_bytes() for n in ("pyproject.toml", "uv.lock", "VERSION")}
    with job.Lock(root) as lock:
        _steps(lock)
        for name in release:
            shutil.copyfile(root / ".localm-update" / ID / "new" / name, root / name)
        steps_packages.run_packages(root, ID, step, [], world.uv, env=world.env)
    assert _versions(root)["alpha"] == "2.0"
    return release


def _lose_of(venv: Path, package: str) -> None:
    """Delete the moved away ``v1.py`` of *package*."""
    lost = [p for p in (venv / ".localm-old" / ID / "old").rglob("v1.py")
            if p.parent.name == package]
    assert len(lost) == 1, lost
    lost[0].unlink()


def _put_back(world, root, **kw):
    said: list = []
    with job.Lock(root) as lock:
        return steps_update.put_back(lock, uv=world.uv, env=world.env, say=said.append,
                                     **kw), said


def test_a_package_that_is_lost_on_the_way_back_is_installed_again(world, tmp_path):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    _lose_of(root / ".venv", "alpha")
    result, said = _put_back(world, root)
    assert (result.state, result.errors) == ("old", [])
    assert said == ["Installing alpha 1.0 again..."]
    assert _versions(root) == {"alpha": "1.0", "beta": "1.0", "extra": "1.0"}
    assert (root / ".venv" / SITE / "alpha" / "v1.py").is_file()
    assert job.read(root) is None


def test_another_version_somebody_installed_meanwhile_is_replaced_by_the_old_one(world, tmp_path):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    wheel(world.wheels, "alpha", "3.0", {"alpha/__init__.py": "V = '3.0'\n"})
    world.run(root, "pip", "install", "--python", str(root / ".venv"), "alpha==3.0", "--index-url",
              world.index.url)
    result, _said = _put_back(world, root)
    assert (result.state, result.errors) == ("old", [])
    assert _versions(root) == {"alpha": "1.0", "beta": "1.0", "extra": "1.0"}


def test_a_package_that_cannot_be_fetched_again_stays_named(world, tmp_path):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    _lose_of(root / ".venv", "alpha")
    world.index.stop()
    world.env["UV_CACHE_DIR"] = str(tmp_path / "another-cache")
    result, _said = _put_back(world, root)
    assert result.state == "failed" and job.read(root) is not None
    assert [(p["name"], p["version"]) for p in result.packages] == [("alpha", "1.0")]
    assert "could not be installed again" in result.errors[-2]
    assert result.errors[-1] == ("These Python packages are not complete: alpha 1.0. Run localm's "
                                 "setup again, which repairs that.")


def test_a_package_the_cache_does_not_have_needs_the_network(world, tmp_path):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    _lose_of(root / ".venv", "alpha")
    world.env["UV_CACHE_DIR"] = str(tmp_path / "another-cache")
    result, _said = _put_back(world, root, network=False)
    assert result.state == "failed" and "may not use the network" in result.errors[-1]
    result, _said = _put_back(world, root)
    assert (result.state, result.errors) == ("old", [])


def test_the_version_that_was_installed_comes_back_not_that_of_the_lock(world, tmp_path):
    root = _install(world, tmp_path)
    wheel(world.wheels, "alpha", "1.5", {"alpha/__init__.py": "V = '1.5'\n",
                                         "alpha/v1.py": "x = 1\n"})
    world.run(root, "pip", "install", "--python", str(root / ".venv"), "alpha==1.5",
              "--index-url", world.index.url)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    _lose_of(root / ".venv", "alpha")
    result, said = _put_back(world, root)
    assert (result.state, result.errors) == ("old", [])
    assert said == ["Installing alpha 1.5 again..."] and _versions(root)["alpha"] == "1.5"


def test_the_rounds_of_reinstalling_end(world, tmp_path, monkeypatch):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    _lose_of(root / ".venv", "alpha")
    rounds: list = []
    monkeypatch.setattr(steps_packages, "reinstall", lambda *a, **k: rounds.append(1))
    result, _said = _put_back(world, root)
    assert len(rounds) == 3 and result.state == "failed"
    assert "These Python packages are not complete: alpha 1.0." in result.errors[-1]


def test_an_edit_in_a_package_that_is_installed_again_is_kept(world, tmp_path):
    root = _install(world, tmp_path)
    for name, data in _updated(world, root).items():
        (root / name).write_bytes(data)
    (root / ".venv" / SITE / "extra" / "__init__.py").write_text("edited by the user")
    with job.Lock(root):
        steps_packages.reinstall(root, ID, mover.PACKAGES, [("extra", "1.0")], world.uv,
                                 env=world.env)
    kept = [p.read_text() for p in (root / ".venv" / ".localm-kept" / ID).rglob("*")
            if p.is_file()]
    assert kept == ["edited by the user"]
    assert (root / ".venv" / SITE / "extra" / "__init__.py").read_text() == "V = '1.0'\n"


def test_a_pytorch_build_comes_from_its_own_index():
    found = steps_packages._own_indexes(["torch==2.9.0+cu126", "torchvision==0.24.0+cu126",
                                         "numpy==2.0", "torch==2.9.0+rocm6.2"])
    assert found == ["--extra-index-url", "https://download.pytorch.org/whl/cu126",
                     "--extra-index-url", "https://download.pytorch.org/whl/rocm6.2"]


def test_a_package_of_a_part_is_installed_again_into_its_venv(world, tmp_path):
    root, part = tmp_path / "root", tmp_path / "comfyui"
    um.install(root)
    part.mkdir()
    world.run(part, "venv", "venv", "--python", sys.executable)
    world.run(part, "pip", "install", "--python", str(part / "venv"), "--no-index",
              "--find-links", str(world.wheels), "alpha==1.0")
    with job.Lock(root) as lock:
        (mover.work_folder(root, ID) / "plans").mkdir(parents=True)
        _steps(lock)
        disk.write_json(mover.work_folder(root, ID) / steps_update.STEPS_FILE, {
            "format": steps_update.FORMAT, "sync": None, "torch": [], "uv": None, "kept": [],
            "notes": [], "parts": [{"kind": "comfyui", "label": "x", "removed": [],
                                    "packages": True, "root": job.part_path(root, part),
                                    "indexes": ["--index-url", world.index.url]}]})
        job.add_part(lock, part)
        stage = steps_packages.packages.stage_folder(part, ID, step=mover.VENV)
        world.run(part, "pip", "install", "--python", str(part / "venv"), "--prefix", str(stage),
                  "--no-index", "--find-links", str(world.wheels), "alpha==2.0")
        steps_packages.packages.note_staged(part, ID, mover.VENV, ["alpha"])
        steps_packages.run_pip(part, ID)
    _lose_of(part / "venv", "alpha")
    result, _said = _put_back(world, root)
    assert (result.state, result.errors) == ("old", [])
    assert (part / "venv" / SITE / "alpha" / "v1.py").is_file()
    assert not (part / "venv" / SITE / "alpha" / "v2.py").exists()
    assert job.read(part) is None


@pytest.mark.skipif(os.name != "nt", reason="an open file can be replaced")
def test_somebody_is_asked_about_a_file_that_is_held(world, tmp_path):
    root = _install(world, tmp_path)
    path = root / ".venv" / SITE / "alpha" / "__init__.py"
    with open(path, "rb") as held:
        with pytest.raises(disk.Held):
            _again(world, root, [("alpha", "1.0")], wait_s=0.1)
        assert path.read_text() == "V = '1.0'\n"
        asked: list = []

        def ask(e: disk.Held) -> bool:
            asked.append(Path(e.path).name)
            held.close()
            return True
        _again(world, root, [("alpha", "1.0")], wait_s=0.1, ask=ask)
    assert asked == ["__init__.py"]


# ---------------------------------------------------------------- pip (ComfyUI)

def test_packages_pip_staged_are_moved_in_and_put_back(world, tmp_path):
    root = tmp_path / "comfyui"
    root.mkdir()
    made = subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], text=True,
                          capture_output=True, stdin=subprocess.DEVNULL, **um.NO_WINDOW)
    assert made.returncode == 0, made.stderr
    world.run(root, "pip", "install", "--python", str(root / "venv"), "--no-index", "--find-links",
              str(world.wheels), "alpha==1.0")
    before = um.tree(root, skip=SKIP)
    with steps.Work(root) as work:
        stage = steps_packages.packages.stage_folder(root, work.id, step=mover.VENV)
        stage.mkdir(parents=True)
        python = disk.short_path(steps_packages.python_in(root / "venv"))
        status, out = steps_packages.packages.run_hidden(
            [python, "-m", "pip", "install", "--no-index", "--no-deps", "--ignore-installed",
             "--no-compile", "--find-links", str(world.wheels), "--prefix",
             disk.short_path(stage), "alpha==2.0", "extra==1.0"], str(root), dict(os.environ))
        assert status == 0, out
        steps_packages.packages.note_staged(root, work.id, mover.VENV, ["alpha", "extra"])
        assert um.tree(root, skip=SKIP) == before
        with pytest.raises(steps.Failed, match="no good"):
            steps.change(work, [lambda: steps_packages.run_pip(root, work.id)],
                         prove=lambda: "no good")
    assert um.tree(root, skip=SKIP) == before


def test_staged_packages_outlive_the_end_of_an_interrupted_change(world, tmp_path):
    root = _install(world, tmp_path)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(root / ".localm-update" / ID / "new" / name, root / name)
    before = um.tree(root, skip=SKIP)
    with steps.Work(root) as work:
        other = steps.Place(root, "cd" * 16)
        other.new(mover.UV)
        with job.Lock(root) as lock:
            job.start(lock, other.id, old="", new="")
            job.advance(lock, job.MOVING)
        assert _future(world, root, id=work.id).sync(["used"])
        step = {"extras": ["used"], "project": False}
        with pytest.raises(steps.Failed, match="no good"):
            steps.change(work, [lambda: steps_packages.run_packages(
                root, work.id, step, [], world.uv, env=world.env)], prove=lambda: (
                    "no good" if _versions(root)["alpha"] == "2.0" else "not moved in"))
    assert um.tree(root, skip=SKIP) == before


def test_a_package_pip_did_not_stage_whole_is_refused(world, tmp_path):
    root = tmp_path / "comfyui"
    root.mkdir()
    world.run(root, "venv", "venv", "--python", sys.executable)
    with steps.Work(root) as work, pytest.raises(Refused):
        stage = steps_packages.packages.stage_folder(root, work.id, step=mover.VENV)
        site = stage / ("Lib/site-packages" if os.name == "nt" else "lib/python3.12/site-packages")
        um.write(site, {"alpha-2.0.dist-info/METADATA": "Name: alpha\nVersion: 2.0\n",
                        "alpha-2.0.dist-info/RECORD": "alpha/__init__.py,,3\n",
                        "alpha/__init__.py": "longer"})
        steps_packages.packages.note_staged(root, work.id, mover.VENV, ["alpha"])
