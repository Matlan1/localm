# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installs, releases and processes for the tests of ``localm.update_mechanism``."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from localm.update_mechanism import shared

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "tests" / "_update_mechanism_driver.py"
UPDATE_ID = "0123456789abcdef0123456789abcdef"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

#: A release before and after, file by file: changed, unchanged, added, removed, a
#: folder that goes, a folder that comes, a file that becomes a folder and back, a
#: name that changes only its case, and two files that are never absent.
OLD = {
    "VERSION": "1",
    "Readme.md": "read me 1",
    "localm/__init__.py": "gate = 1",
    "localm.bat": "start 1",
    "localm/same.py": "same",
    "localm/changed.py": "old",
    "localm/gone.py": "gone",
    "localm/olddir/x.py": "x",
    "localm/olddir/sub/y.py": "y",
    "localm/thing": "a file",
    "localm/other/z.py": "a folder",
}
NEW = {
    "VERSION": "2",
    "README.md": "read me 2",
    "localm/__init__.py": "gate = 2",
    "localm.bat": "start 2",
    "localm/same.py": "same",
    "localm/changed.py": "new",
    "localm/added.py": "added",
    "localm/newdir/deep/n.py": "n",
    "localm/thing/inner.py": "now a folder",
    "localm/other": "now a file",
}
SWAP = ("localm/__init__.py", "localm.bat")
#: What an update never changes, with the same names in the new release.
USER = {
    "home/models/m.gguf": "weights",
    "home/chats/c.json": "a chat",
    ".git/HEAD": "ref",
    "mydata/notes.txt": "mine",
    "localm/user.txt": "mine too",
    "runtime/localm_llama_runtime/lib/llama.dll": "the runtime",
}
INTRUDING = {
    "home/models/m.gguf": "release weights",
    ".git/HEAD": "release ref",
    "mydata/notes.txt": "release notes",
    "runtime/localm_llama_runtime/lib/llama.dll": "release runtime",
}


def write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def stated(files: dict) -> dict:
    return {rel: {"sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(), "size": len(t)}
            for rel, t in files.items()}


def statement(files: dict) -> str:
    return json.dumps({"files": sorted(set(files) - set(USER))})


def install(root: Path, old: dict = OLD) -> Path:
    """An install of the release *old* with the user's files in it."""
    write(root, {**old, **USER, shared.update_package.NAME: statement(old)})
    if os.name != "nt":
        (root / "localm.bat").chmod(0o755)
    return root


def stage(root: Path, new: dict | None = None, update_id: str = UPDATE_ID) -> dict:
    """Extract the release *new* for an update of *root*; returns its file list."""
    new = {**NEW, **INTRUDING} if new is None else new
    write(root / ".localm-update" / update_id / "new",
          {**new, shared.update_package.NAME: statement(new)})
    return stated(new)


def tree(root: Path, skip=(".localm-update",)) -> dict:
    """Every file, link and folder of *root* with its content, target and mode,
    compiled Python files and the top-level names *skip* excepted."""
    found: dict = {}
    for folder, dirs, names in os.walk(root):
        here = Path(folder)
        rel = here.relative_to(root)
        dirs[:] = [d for d in dirs if d != "__pycache__"
                   and (rel / d).as_posix() not in skip]
        for name in [*dirs, *names]:
            path = here / name
            key = (rel / name).as_posix()
            if key in skip:
                continue
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                found[key] = ("link", os.readlink(path))
            elif stat.S_ISDIR(st.st_mode):
                found[key] = ("folder",)
            else:
                found[key] = ("file", hashlib.sha256(path.read_bytes()).hexdigest(),
                              stat.S_IMODE(st.st_mode))
    return found


def new_tree(root: Path, new: dict = NEW) -> dict:
    """What :func:`tree` of a fresh install of *new* with the user's files is."""
    install(root, new)
    return tree(root)


def applier(dest: Path) -> Path:
    """A copy of the update mechanism of this build at *dest*."""
    return shared.copy_applier(REPO / "localm", dest)


def run(argv: list, **kw) -> subprocess.CompletedProcess:
    """Run python with *argv*, isolated, hidden, output captured."""
    return subprocess.run([sys.executable, "-I", "-S", *map(str, argv)], capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=120, **NO_WINDOW, **kw)


def drive(run_dir: Path, job: dict, kill_at: int) -> subprocess.CompletedProcess:
    """Run the driver with the applier at *run_dir* on *job*."""
    path = Path(job["root"]).with_name("job-spec.json")
    path.write_text(json.dumps(job), encoding="utf-8")
    return run([DRIVER, run_dir, path, kill_at])


def restore(run_dir: Path, root: Path) -> subprocess.CompletedProcess:
    """Run the applier's own ``restore`` command."""
    return run([run_dir, "restore", root])


def phase(root: Path):
    """The phase in the record of *root*, None without one."""
    path = root / ".localm-update" / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))["phase"] if path.exists() else None


def left(root: Path) -> list:
    """What is left of updates in *root* besides the lock."""
    folder = root / ".localm-update"
    return sorted(p.name for p in folder.iterdir() if p.name != "lock") if folder.exists() else []
