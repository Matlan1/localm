# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every step of an update, and every change of a component on its own, on a small tree
with one file of each kind: the process ends in each phase of the job, and around each
move of each kind of file on the way there and back. The way back then gives the old
tree, also when it runs twice, and an update that may be finished gives the new one."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from localm.update_mechanism import disk, job, mover, steps, steps_packages, steps_update
from tests import _update_mechanism as um

ID = um.UPDATE_ID
DRIVER = um.REPO / "tests" / "_update_steps_driver.py"
NT = os.name == "nt"
SITE = ("Lib/site-packages" if NT
        else f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages")
UV = "uv.exe" if NT else "uv"
_OURS = {".localm-update", ".localm-old", ".localm-kept", "lib.incoming", ".lock", "__pycache__"}


LINKED = "data/comfyui/nodes"


def tree(top: Path) -> dict:
    """The files and links below *top*, also in :class:`Linked`'s folder, without those
    of the mechanism itself."""
    found = um.tree(top, skip=())
    if (top / LINKED).exists():
        found |= {f"{LINKED}/{rel}": what for rel, what in um.tree(top / "theirs").items()}
    return {rel: what for rel, what in found.items()
            if not _OURS & set(rel.split("/")) and what != ("folder",)}


class Step:
    """A step on a small tree. *kinds* names the file of each kind, relative to the
    tree; a kind the step cannot meet is not among them. *left* is what stays of an update
    that is done."""

    left: list = []

    def __init__(self, name: str, *, alone: bool = False, links: bool = True) -> None:
        self.name, self.alone, self.links = name, alone, links
        self.kinds: dict = {}
        self.root = self.part = ""     # relative to the tree
        self.whole, self.packages, self.remove = True, False, []

    def link(self, folder: Path, target: str) -> None:
        if self.links:
            os.symlink(target, folder / "link")

    def old(self, folder: Path) -> None:
        um.write(folder, {"changed": "old", "removed": "old", "edited": "edited by the user",
                          "same": "same"})
        self.link(folder, "changed")

    def new(self, folder: Path) -> None:
        um.write(folder, {"changed": "new", "added": "new", "edited": "new", "same": "same"})
        self.link(folder, "added")

    def names(self) -> list:
        return ["changed", "removed", "edited", "same", *(["link"] if self.links else [])]

    def kinds_in(self, where: str) -> dict:
        return {k: f"{where}/{k}" for k in ("added", *self.names()) if k != "same"}

    def build(self, top: Path) -> None:
        raise NotImplementedError

    def spec(self, top: Path, **kw) -> None:
        """Write the steps of the update."""
        work = mover.work_folder(top / self.root, ID)
        (work / "plans").mkdir(parents=True, exist_ok=True)
        disk.write_json(work / steps_update.STEPS_FILE,
                        {"format": steps_update.FORMAT, "sync": None, "torch": [], "uv": None,
                         "parts": [], "kept": [], "notes": [], **kw})

    def own(self, top: Path) -> Path:
        folder = top / self.part
        if self.alone:
            self.root = self.part
            (mover.work_folder(folder, ID) / "plans").mkdir(parents=True, exist_ok=True)
        else:
            job.own_part(top / self.root, ID, folder)
        return folder

    def part_spec(self, top: Path, kind: str, **kw) -> None:
        if not self.alone:
            self.spec(top, parts=[{"kind": kind, "label": self.name,
                                   "root": job.part_path(top / self.root, top / self.part), **kw}])


class Runtime(Step):
    def build(self, top: Path) -> None:
        self.root, self.part = "root", "root/runtime/localm_llama_runtime"
        um.write(top / self.part, {"lib/mine/notes.txt": "the user's", "lib/.gitkeep": "",
                                   "lib/rocblas/old.dat": "old"})
        self.old(top / self.part / "lib")
        place = steps.Place(self.own(top), ID)
        staged = place.new(mover.WHOLE) / "lib" if self.alone else place.beside("lib.incoming")
        self.new(staged)
        self.kinds = {**self.kinds_in(f"{self.part}/lib"),
                      "own": f"{self.part}/lib/mine/notes.txt"}
        self.remove = [f"lib/{n}" for n in (*self.names(), "rocblas/old.dat")]
        self.part_spec(top, steps_update.RUNTIME, name="lib", beside="lib.incoming",
                       backend="cpu", to="b2", unproven=True, keep=[".gitkeep"],
                       folders=["rocblas"], **{"from": "b1"})


class Folders(Step):
    """A native media runtime: a new tag folder beside the old one, or (alone) the same
    folder replaced."""

    def build(self, top: Path) -> None:
        self.root, self.part = "root", f"data/runtimes/{self.name}"
        um.write(top, {"root/VERSION": "1", f"{self.part}/mine.txt": "the user's"})
        self.old(top / self.part / "v1")
        folder = self.own(top)
        self.kinds = {"own": f"{self.part}/mine.txt"}
        new = steps.Place(folder, ID).new(mover.WHOLE)
        if self.alone:
            self.new(new / "v1")
            self.kinds |= self.kinds_in(f"{self.part}/v1")
            self.remove = steps.folder_files(folder, "v1")
        else:
            self.new(new / "v2")
            self.kinds |= {"added": f"{self.part}/v2/added"}
            if self.links:
                self.kinds["link"] = f"{self.part}/v2/link"
        self.part_spec(top, steps_update.FOLDERS)


class Comfyui(Step):
    """The source of the managed ComfyUI, or its packages."""

    def build(self, top: Path) -> None:
        self.root, self.part, self.whole = "root", "data/comfyui", False
        self.packages = self.name == "packages"
        um.write(top, {"root/VERSION": "1", f"{self.part}/custom_nodes/mine/n.py": "the user's"})
        self.old(top / self.part)
        _venv(top / self.part / "venv")
        folder = self.own(top)
        new = steps.Place(folder, ID).new(mover.TREE)
        if self.packages:
            self.kinds = _staged_packages(folder, mover.VENV, f"{self.part}/venv")
        else:
            self.remove = ["removed"]
            self.new(new)
            self.kinds = {**self.kinds_in(self.part),
                          "own": f"{self.part}/custom_nodes/mine/n.py"}
        self.part_spec(top, steps_update.COMFYUI, removed=self.remove, packages=self.packages)


class Linked(Step):
    """A folder of the managed ComfyUI that is a link to one of the user's, whose replaced
    files are kept."""

    left = ["kept"]

    def build(self, top: Path) -> None:
        self.root, self.part, self.whole = "root", "data/comfyui", False
        um.write(top, {"root/VERSION": "1", f"{self.part}/main.py": "same",
                       "theirs/mine.txt": "the user's"})
        _venv(top / self.part / "venv")
        self.old(top / "theirs")
        if NT:
            import _winapi
            _winapi.CreateJunction(str(top / "theirs"), str(top / LINKED))
        else:
            os.symlink("../../theirs", top / LINKED)
        self.new(steps.Place(self.own(top), ID).new(mover.TREE) / "nodes")
        self.remove = ["nodes/removed"]
        self.kinds = {**self.kinds_in(LINKED), "own": "theirs/mine.txt"}
        self.part_spec(top, steps_update.COMFYUI, removed=self.remove, packages=False)


class Uv(Step):
    def build(self, top: Path) -> None:
        self.root = self.part = "root"
        um.write(top, {"root/VERSION": "1", f"root/.uv/{UV}": "old"})
        self.old(top / "root" / ".uv")
        new = steps.Place(top / "root", ID).new(mover.UV)
        self.new(new)
        um.write(new, {UV: "new"})
        self.kinds = {**self.kinds_in("root/.uv"), "own": "root/VERSION"}
        self.spec(top, uv="9.9")


class Packages(Step):
    def build(self, top: Path) -> None:
        self.root = self.part = "root"
        um.write(top, {"root/VERSION": "1"})
        _venv(top / "root" / ".venv")
        self.spec(top, sync={"extras": [], "project": False})
        self.kinds = _staged_packages(top / "root", mover.PACKAGES, "root/.venv")


def _package(site: Path, version: str, files: dict) -> None:
    info = f"alpha-{version}.dist-info"
    entries = {**files, f"{info}/METADATA": f"Metadata-Version: 2.1\nName: alpha\n"
                                            f"Version: {version}\n"}
    rows = io.StringIO()
    out = csv.writer(rows, lineterminator="\n")
    for path, text in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())
        out.writerow([path, f"sha256={digest.rstrip(b'=').decode()}", len(text)])
    out.writerow([f"{info}/RECORD", "", ""])
    um.write(site, {**entries, f"{info}/RECORD": rows.getvalue()})


def _venv(venv: Path) -> None:
    um.write(venv, {"pyvenv.cfg": "home = x\n"})
    site = venv / SITE
    _package(site, "1.0", {"alpha/changed": "old", "alpha/removed": "old",
                           "alpha/edited": "old"})
    um.write(site, {"alpha/edited": "edited by the user", "alpha/mine": "the user's",
                    "beta/__init__.py": "another package"})


def _staged_packages(root: Path, step: str, venv: str) -> dict:
    stage = steps_packages.packages.stage_folder(root, ID, step=step)
    _package(stage / SITE, "2.0", {"alpha/changed": "new", "alpha/added": "new",
                                   "alpha/edited": "new"})
    steps_packages.packages.note_staged(root, ID, step, ["alpha"])
    return {**{k: f"{venv}/{SITE}/alpha/{k}" for k in ("changed", "added", "removed", "edited")},
            "own": f"{venv}/{SITE}/alpha/mine",
            "package": f"{venv}/{SITE}/alpha-1.0.dist-info/RECORD"}


STEPS = {"runtime": (Runtime, "runtime"), "sdcpp": (Folders, "sdcpp"),
         "koboldcpp": (Folders, "koboldcpp"), "comfyui-tree": (Comfyui, "tree"),
         "comfyui-packages": (Comfyui, "packages"), "comfyui-linked": (Linked, "linked"), "uv": (Uv, "uv"),
         "packages": (Packages, "packages")}
NAMES = [*STEPS, *(f"{n}-alone" for n in STEPS if n not in ("uv", "packages"))]


class Bench:
    """Runs one step, again and again, each time on a tree of its own."""

    def __init__(self, name: str, folder: Path, links: bool) -> None:
        alone = name.endswith("-alone")
        self.kind, self.called = STEPS[name.removesuffix("-alone")]
        self.alone, self.folder, self.links = alone, folder, links

    def build(self, label) -> tuple:
        top = self.folder / f"t{label}"
        step = self.kind(self.called, alone=self.alone, links=self.links)
        step.build(top)
        return step, top

    def run(self, step: Step, top: Path, kill=None, fail: bool = False):
        spec = top.with_name(top.name + ".json")
        spec.write_text(json.dumps({
            "repo": str(um.REPO), "top": str(top), "root": str(top / step.root), "id": ID,
            "alone": step.alone, "whole": step.whole, "packages": step.packages,
            "remove": step.remove, "fail": fail, "kill": kill}), encoding="utf-8")
        return subprocess.run([sys.executable, "-X", "utf8", str(DRIVER), str(spec)],
                              capture_output=True, text=True, encoding="utf-8",
                              stdin=subprocess.DEVNULL, timeout=120, **um.NO_WINDOW)

    def settle(self, step: Step, top: Path):
        """The way back, as the next use of the component, or of the install, runs it."""
        if step.alone:
            return steps.settle(top / step.root, wait_s=1) or job.Result("none")
        if step.part != step.root:
            steps.settle(top / step.part, wait_s=1)   # used before the update ended
        with job.Lock(top / step.root) as lock:
            return job.restore(lock, wait_s=0.3)


@pytest.fixture
def bench(request, tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "HELD_WAIT_S", 0.3)
    links = True
    try:
        os.symlink("target", tmp_path / "link")
    except OSError:
        links = False
    return Bench(request.param, tmp_path, links)


def _result(done: subprocess.CompletedProcess) -> dict:
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.split("RESULT ")[1])


def _places(step: Step, old: dict, new: dict, went: dict, failed: dict) -> list:
    """``(where the process ends, whether the update fails by itself, what comes of
    it)`` for every phase, and around every move of one file of each kind."""
    comes = {("whole", "starting"): "either", ("whole", "done"): "new", ("part", "done"): "new"}
    places: list = [({"record": w, "phase": p}, [w, p] in failed["records"],
                     comes.get((w, p), "old"))
                    for w, p in [*failed["records"], *went["records"]]]
    for whose in ("whole",) if step.part == step.root else ("whole", "part"):
        assert {w[0]["phase"] for w in places if w[0]["record"] == whose} == {
            "ending", "moving", "steps", "starting", "done", "restoring"}, whose
    places = [p for n, p in enumerate(places) if p not in places[:n]]
    for kind, path in step.kinds.items():
        if kind == "own":
            continue
        for back, result in ((False, went), (True, failed)):
            count = result["moves"].count([back, path])
            assert count == (path in old) + (path in new), (kind, back, result["moves"])
            at = {"back": back, "path": path}
            places.append(({**at, "move": 0, "after": False}, True, "old"))
            places += [({**at, "move": n, "after": True}, True, "old") for n in range(count)]
    return places


@pytest.mark.parametrize("bench", NAMES, indirect=True)
def test_a_step_ended_at_any_place_is_old_or_finished(bench):
    step, top = bench.build("new")
    old = tree(top)
    went = _result(bench.run(step, top))
    new = tree(top)
    assert went["done"] and new != old and um.left(top / step.part) == step.left
    assert um.left(top / step.root) == (step.left if step.root == step.part else [])
    assert new[step.kinds["own"]] == old[step.kinds["own"]]
    failing, failed_top = bench.build("failed")
    failed = _result(bench.run(failing, failed_top, fail=True))
    assert not failed["done"] and tree(failed_top) == old

    def ended(label, place) -> tuple:
        step, top = bench.build(label)
        killed = bench.run(step, top, kill=place[0], fail=place[1])
        assert killed.returncode == 9, (place, killed.returncode, killed.stdout, killed.stderr)
        return step, top

    def one(numbered) -> None:
        n, place = numbered
        step, top = ended(n, place)
        for _again in range(2):
            result = bench.settle(step, top)
            assert result.errors == [], (place, result)
            assert tree(top) == (new if place[2] == "new" else old), place
        if place[2] == "either":
            step, top = ended(f"{n}f", place)
            with job.Lock(top / step.root) as lock:
                assert job.finish(lock).errors == [], place
            assert tree(top) == new, place
        shutil.rmtree(disk.long_path(top), ignore_errors=True)

    places = _places(step, old, new, went, failed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, enumerate(places)))


def _aside(top: Path, path: str) -> Path:
    """The old file moved away from *path*."""
    found = [p for p in top.rglob(Path(path).name)
             if "old" in p.relative_to(top).parts and p.is_file()]
    assert len(found) == 1, found
    return found[0]


@pytest.mark.skipif(not NT, reason="an open file can be moved")
@pytest.mark.parametrize("bench", NAMES, indirect=True)
def test_a_held_file_stops_a_step_and_its_way_back_until_it_is_free(bench):
    step, top = bench.build("held")
    old = tree(top)
    changed = "changed" in step.kinds
    path = step.kinds["changed" if changed else "added"]
    if changed:
        with open(top / path, "rb"):
            assert not _result(bench.run(step, top))["done"]
        assert bench.settle(step, top).errors == []
        assert tree(top) == old
    for n in range(2 if changed else 1):
        step, top = bench.build(n)
        killed = bench.run(step, top, kill={"back": False, "path": path, "move": n, "after": True})
        assert killed.returncode == 9, killed.stderr
        with open(_aside(top, path) if changed and n == 0 else top / path, "rb"):
            assert bench.settle(step, top).errors != []
        assert bench.settle(step, top).errors == []
        assert tree(top) == old


@pytest.mark.parametrize("bench", [n for n in NAMES if "packages" in n], indirect=True)
def test_a_package_file_the_user_edited_is_kept_when_the_update_ends(bench):
    step, top = bench.build("kept")
    assert _result(bench.run(step, top))["done"]
    kept = [p for p in top.rglob("edited") if ".localm-kept" in p.parts]
    assert [p.read_text() for p in kept] == ["edited by the user"]
