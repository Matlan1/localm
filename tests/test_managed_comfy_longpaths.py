# SPDX-License-Identifier: AGPL-3.0-or-later
"""The managed ComfyUI must not fail with Windows' legacy 260-char MAX_PATH limit.

A custom node under a sufficiently nested LOCALM_HOME puts some of its files past
260 characters. Fetching, replacing, keeping, restoring and removing such a
source tree go through extended-length paths, so they work whatever the OS-level
LongPathsEnabled policy is. The ``max_path`` fixture makes Windows apply the limit
in the test process.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess

import pytest

from localm import updater_components as uc
from localm.media import managed_comfy as mc
from localm.media import managed_comfy_fresh as fresh
from localm.media import managed_comfy_provision as provision
from localm.media import source_archive
from tests._max_path import max_path  # noqa: F401
from tests._source_archives import FakeGitHub

_DEEP_REL = "/".join(["a_rather_long_folder_name_for_testing"] * 6) + "/leaf.py"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

pytestmark = pytest.mark.usefixtures("max_path")


def _long(path):
    return source_archive.long_path(path)


def _read(path) -> str:
    with open(_long(path), encoding="utf-8") as f:
        return f.read()


def _write(path, text) -> None:
    os.makedirs(_long(path).parent, exist_ok=True)
    with open(_long(path), "w", encoding="utf-8") as f:
        f.write(text)


def _deep_tree(tmp_path, name="Node"):
    """A source tree with a record whose file ``_DEEP_REL`` lies past 260 characters."""
    root = tmp_path / ("n" * 40) / "custom_nodes" / name
    assert len(str(root / _DEEP_REL)) > 260
    _write(root / _DEEP_REL, "deep\n")
    _write(root / "__init__.py", "x\n")
    source_archive.write_record(root, "https://github.com/o/Node.git", "1" * 40,
                                ["__init__.py", _DEEP_REL])
    return root


def test_a_tree_deeper_than_max_path_is_fetched_and_replaced(monkeypatch, tmp_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "Node", {"__init__.py": "x\n", _DEEP_REL: "deep\n"})
    dest = tmp_path / ("n" * 40) / "custom_nodes" / "Node"
    assert len(str(dest / _DEEP_REL)) > 260
    ok, out = fresh._fetch_at_commit(repo, dest, commit, on_progress=None)
    assert ok, out
    assert os.path.isfile(_long(dest / _DEEP_REL))

    repo2, commit2 = hub.add("o", "Node", {"__init__.py": "y\n", _DEEP_REL: "deeper\n"})
    new = tmp_path / ("m" * 40) / "new"
    source_archive.fetch_tree(repo2, commit2, new)
    change = source_archive.apply_tree(new, dest, source_archive.tree_files(dest),
                                       tmp_path / ("b" * 40) / "backup")
    assert _read(dest / _DEEP_REL) == "deeper\n"
    assert change.rollback() == []
    assert _read(dest / _DEEP_REL) == "deep\n"


def test_a_deep_tree_is_kept_restored_and_its_copy_removed_by_the_app_updater(tmp_path):
    root = _deep_tree(tmp_path)
    src = tmp_path / ("k" * 40)
    dest = src / "comfy-src"

    info = uc._snapshot_tree(root, dest)
    assert _DEEP_REL in info["files"]
    assert _read(dest / _DEEP_REL) == "deep\n"

    _write(root / _DEEP_REL, "changed\n")
    assert uc._restore_tree(dest, root) == []
    assert _read(root / _DEEP_REL) == "deep\n"

    (src / uc.STATE_FILE).write_text(json.dumps({"components": {"comfyui": {}}}), encoding="utf-8")
    uc.discard(src)
    assert not os.path.lexists(_long(src))


def test_a_restore_leaves_a_deep_backup_an_earlier_restore_kept(tmp_path):
    root = _deep_tree(tmp_path)
    dest = tmp_path / ("k" * 40) / "comfy-src"
    uc._snapshot_tree(root, dest)
    earlier = source_archive.new_backup_path(root, "restore")
    _write(earlier / _DEEP_REL, "the originals\n")

    assert uc._restore_tree(dest, root) == []
    assert _read(earlier / _DEEP_REL) == "the originals\n"


def test_rmtree_robust_removes_a_deep_read_only_file(tmp_path):
    root = _deep_tree(tmp_path)
    os.chmod(_long(root / _DEEP_REL), stat.S_IREAD)
    mc.rmtree_robust(root)
    assert not os.path.lexists(_long(root))


def _fake_venv(path, version, file_text):
    site = path / "Lib" / "site-packages"
    info = site / f"pkg-{version}.dist-info"
    _write(info / "METADATA", f"Name: pkg\nVersion: {version}\n\n")
    rel = _DEEP_REL.replace("leaf.py", "pkg.py")
    _write(info / "RECORD", f"{rel},,\npkg-{version}.dist-info/METADATA,,\n")
    _write(site / rel, file_text)
    return site, rel


def test_a_deep_venv_file_is_kept_and_put_back(tmp_path):
    venv = tmp_path / ("v" * 40) / "venv"
    site, rel = _fake_venv(venv, "1.0", "old\n")
    assert len(str(site / rel)) > 260
    snap = tmp_path / "snap"
    snap.mkdir()
    copy = uc._venv_copy_path(venv, snap)

    uc._clone_venv(venv, copy, snap)
    assert _read(copy / "Lib" / "site-packages" / rel) == "old\n"

    shutil.rmtree(_long(venv))
    _fake_venv(venv, "2.0", "new\n")
    assert uc._restore_venv(copy, venv) == []
    assert _read(site / rel) == "old\n"
    assert uc.dists(_long(site))["pkg"]["version"] == "1.0"


def _user_comfyui(tmp_path):
    user = tmp_path / ("u" * 40) / "ComfyUI"
    _write(user / _DEEP_REL, "deep\n")
    _write(user / "main.py", "main\n")
    return user


@pytest.mark.skipif(shutil.which("git") is None, reason="git writes the index")
def test_copying_a_deep_user_comfyui_keeps_every_tracked_file(tmp_path):
    user = _user_comfyui(tmp_path)
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
