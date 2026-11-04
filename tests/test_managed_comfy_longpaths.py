# SPDX-License-Identifier: AGPL-3.0-or-later
"""A managed-ComfyUI source fetch must not fail with Windows' legacy 260-char
MAX_PATH limit.

A custom node under a sufficiently nested LOCALM_HOME puts some of its files past
260 characters. The archive is unpacked and applied through extended-length
paths, so that works whatever the OS-level LongPathsEnabled policy is. The same
holds for the app updater's kept copy of the tree, its restore, and the copy of
a user's ComfyUI checkout; the ``max_path`` fixture refuses an over-long path
without the extended-length prefix the way Windows does without that policy.
"""

from __future__ import annotations

import builtins
import io
import os
import shutil
import subprocess
import sys

import pytest

from localm import updater_components as uc
from localm.media import managed_comfy_fresh as fresh
from localm.media import managed_comfy_provision as provision
from localm.media import source_archive
from tests._source_archives import FakeGitHub

_DEEP_REL = "/".join(["a_rather_long_folder_name_for_testing"] * 6) + "/leaf.py"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.fixture
def max_path(monkeypatch):
    """Refuse every file system call on a path of 260 or more characters that
    lacks the extended-length prefix, as Windows does without the
    LongPathsEnabled policy. Returns the refused paths."""
    refused = []

    def too_long(arg) -> bool:
        if not isinstance(arg, (str, bytes, os.PathLike)):
            return False
        text = os.fsdecode(os.fspath(arg))
        return len(text) >= 260 and not text.startswith("\\\\?\\")

    def guard(real):
        def call(*args, **kwargs):
            for arg in args[:2]:
                if too_long(arg):
                    text = os.fsdecode(os.fspath(arg))
                    refused.append(text)
                    raise FileNotFoundError(2, "The system cannot find the path specified",
                                            text)
            return real(*args, **kwargs)
        return call

    for name in ("stat", "lstat", "mkdir", "open", "unlink", "remove", "rmdir", "scandir",
                 "listdir", "replace", "rename", "utime", "chmod"):
        monkeypatch.setattr(os, name, guard(getattr(os, name)))
    for name in ("copy2", "copyfile", "copystat"):
        monkeypatch.setattr(shutil, name, guard(getattr(shutil, name)))
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    return refused


def _read(path) -> str:
    with open(source_archive.long_path(path), encoding="utf-8") as f:
        return f.read()


def test_a_tree_deeper_than_max_path_is_fetched_and_replaced(monkeypatch, tmp_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "Node", {"__init__.py": "x\n", _DEEP_REL: "deep\n"})
    dest = tmp_path / ("n" * 40) / "custom_nodes" / "Node"
    assert len(str(dest / _DEEP_REL)) > 260
    ok, out = fresh._fetch_at_commit(repo, dest, commit, on_progress=None)
    assert ok, out
    assert os.path.isfile(source_archive.long_path(dest / _DEEP_REL))

    repo2, commit2 = hub.add("o", "Node", {"__init__.py": "y\n", _DEEP_REL: "deeper\n"})
    new = tmp_path / ("m" * 40) / "new"
    source_archive.fetch_tree(repo2, commit2, new)
    change = source_archive.apply_tree(new, dest, source_archive.tree_files(dest),
                                       tmp_path / ("b" * 40) / "backup")
    assert _read(dest / _DEEP_REL) == "deeper\n"
    assert change.rollback() == []
    assert _read(dest / _DEEP_REL) == "deep\n"


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows MAX_PATH limit")
def test_a_deep_tree_is_kept_and_restored_by_the_app_updater(monkeypatch, tmp_path, max_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "Node", {"__init__.py": "x\n", _DEEP_REL: "deep\n"})
    root = tmp_path / ("n" * 40) / "custom_nodes" / "Node"
    assert len(str(root / _DEEP_REL)) > 260
    source_archive.fetch_tree(repo, commit, root)
    dest = tmp_path / ("k" * 40) / "comfy-src"

    info = uc._snapshot_tree(root, dest)
    assert _DEEP_REL in info["files"]
    assert _read(dest / _DEEP_REL) == "deep\n"

    with open(source_archive.long_path(root / _DEEP_REL), "w", encoding="utf-8") as f:
        f.write("changed\n")
    stale = root.parent / f".{root.name}-restore-{os.getpid()}"
    os.makedirs(source_archive.long_path(stale / _DEEP_REL).parent)
    with open(source_archive.long_path(stale / _DEEP_REL), "w", encoding="utf-8") as f:
        f.write("stale\n")
    assert uc._restore_tree(dest, root) == []
    assert _read(root / _DEEP_REL) == "deep\n"
    assert not os.path.lexists(source_archive.long_path(stale))
    assert max_path == []


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows MAX_PATH limit")
@pytest.mark.skipif(shutil.which("git") is None, reason="git writes the index")
def test_copying_a_deep_user_comfyui_keeps_every_tracked_file(tmp_path, max_path):
    user = tmp_path / ("u" * 40) / "ComfyUI"
    deep = source_archive.long_path(user / _DEEP_REL)
    os.makedirs(deep.parent)
    deep.write_text("deep\n", encoding="utf-8")
    (user / "main.py").write_text("main\n", encoding="utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    for args in (["init", "-q"], ["-c", "core.longpaths=true", "add", "-A"]):
        subprocess.run(["git", *args], cwd=str(user), env=env, check=True,
                       capture_output=True, creationflags=_NO_WINDOW)
    managed = tmp_path / ("m" * 40) / "comfyui"
    managed.parent.mkdir()

    ok, out = provision._clone_source(user, managed, "a" * 40, None)
    assert ok, out
    assert _read(managed / _DEEP_REL) == "deep\n"
    assert sorted(source_archive.tree_files(managed)) == sorted([_DEEP_REL, "main.py"])
    assert not os.path.lexists(managed.with_name("comfyui.copying"))
    assert max_path == []
