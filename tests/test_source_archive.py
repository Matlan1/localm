# SPDX-License-Identifier: AGPL-3.0-or-later
"""Source trees at a pinned commit without a git program (localm.media.source_archive).

Archives are checked by their zip comment (the commit git writes there) and, for
the shipped pins, by sha256; unsafe paths are refused; a checkout made by git is
read through its .git folder (HEAD, refs, packed-refs, index v2/v3/v4), checked
against real git where it is installed; apply_tree replaces exactly the tree's
files and can put them back.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import zipfile

import pytest

from localm.media import source_archive as sa
from tests._source_archives import FakeGitHub, fake_commit, make_archive

_GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
_IDENT = ["-c", "user.email=t@localhost", "-c", "user.name=t",
          "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]
_HAS_GIT = shutil.which("git") is not None


def _git(cwd, *args) -> str:
    return subprocess.run(["git", *_IDENT, *args], cwd=str(cwd), env=_GIT_ENV, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("repo, expected", [
    ("https://github.com/comfyanonymous/ComfyUI.git",
     "https://codeload.github.com/comfyanonymous/ComfyUI/zip/"),
    ("https://github.com/city96/ComfyUI-GGUF", "https://codeload.github.com/city96/ComfyUI-GGUF/zip/"),
])
def test_archive_url(repo, expected):
    assert sa.archive_url(repo, "a" * 40) == expected + "a" * 40


@pytest.mark.parametrize("repo, commit", [
    ("https://example.com/o/r.git", "a" * 40),
    ("http://github.com/o/r.git", "a" * 40),
    ("https://github.com/o/r.git", "abc123"),
    ("https://github.com/o/r.git", "main"),
    ("D:/local/repo", "a" * 40),
])
def test_archive_url_refuses_anything_else(repo, commit):
    with pytest.raises(sa.ArchiveError):
        sa.archive_url(repo, commit)


def test_fetch_writes_the_tree_and_its_record(monkeypatch, tmp_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "R", {"a.py": "1\n", "sub/b.txt": "2\n"})
    files = sa.fetch_tree(repo, commit, tmp_path / "dest")
    assert files == ["a.py", "sub/b.txt"]
    assert (tmp_path / "dest" / "sub" / "b.txt").read_text(encoding="utf-8") == "2\n"
    assert sa.tree_commit(tmp_path / "dest") == commit
    assert sa.tree_files(tmp_path / "dest") == files
    assert [p.name for p in tmp_path.iterdir()] == sorted(["hub", "dest"])


def test_an_archive_of_another_commit_is_refused(monkeypatch, tmp_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "R", {"a.py": "1\n"}, comment="f" * 40)
    with pytest.raises(sa.ArchiveError, match="not " + commit):
        sa.fetch_tree(repo, commit, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_a_pinned_commit_with_the_wrong_sha256_is_refused(monkeypatch, tmp_path):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "R", {"a.py": "1\n"})
    monkeypatch.setitem(sa.ARCHIVE_SHA256, commit, "0" * 64)
    with pytest.raises(sa.ArchiveError, match="sha256"):
        sa.fetch_tree(repo, commit, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()
    good = hashlib.sha256(next(iter(hub.archives.values())).read_bytes()).hexdigest()
    monkeypatch.setitem(sa.ARCHIVE_SHA256, commit, good)
    sa.fetch_tree(repo, commit, tmp_path / "dest")


def test_the_shipped_pins_carry_a_sha256():
    from localm.media import managed_comfy_fresh as fresh
    assert fresh.COMFYUI_PINNED_COMMIT in sa.ARCHIVE_SHA256
    for node in fresh.required_custom_nodes().nodes:
        assert node.commit in sa.ARCHIVE_SHA256, node.name


@pytest.mark.parametrize("bad", ["../evil.py", "/abs.py", "a/../../evil.py", "C:/x.py"])
def test_an_unsafe_path_is_refused(monkeypatch, tmp_path, bad):
    commit = fake_commit(bad)
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"R-{commit}/ok.py", "x")
        zf.writestr(f"R-{commit}/{bad}", "x")
        zf.comment = commit.encode()

    monkeypatch.setattr(sa, "download", lambda url, dest: shutil.copyfile(zpath, dest))
    with pytest.raises(sa.ArchiveError, match="unsafe"):
        sa.fetch_tree("https://github.com/o/R.git", commit, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()
    assert not (tmp_path / "evil.py").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_an_executable_keeps_its_mode(monkeypatch, tmp_path):
    commit = fake_commit("x")
    zpath = make_archive(tmp_path / "a.zip", "R", commit, {"run.sh": "#!/bin/sh\n", "a": "1"},
                         executable=("run.sh",))
    monkeypatch.setattr(sa, "download", lambda url, dest: shutil.copyfile(zpath, dest))
    sa.fetch_tree("https://github.com/o/R.git", commit, tmp_path / "dest")
    assert os.access(tmp_path / "dest" / "run.sh", os.X_OK)
    assert not os.access(tmp_path / "dest" / "a", os.X_OK)


def test_head_from_a_detached_head_a_ref_and_packed_refs(tmp_path):
    git = tmp_path / "c" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    assert sa.git_head(tmp_path / "c") == "a" * 40
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert sa.git_head(tmp_path / "c") is None
    (git / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/main\n",
                                     encoding="utf-8")
    assert sa.git_head(tmp_path / "c") == "b" * 40
    (git / "refs" / "heads" / "main").write_text("c" * 40 + "\n", encoding="utf-8")
    assert sa.git_head(tmp_path / "c") == "c" * 40
    assert sa.git_head(tmp_path) is None


@pytest.mark.skipif(not _HAS_GIT, reason="real git is the reference")
@pytest.mark.parametrize("index_version", ["2", "3", "4"])
def test_head_and_index_match_real_git(tmp_path, index_version):
    repo = tmp_path / "r"
    repo.mkdir()
    for rel in ("a.py", "deep/er/b.txt", "deep/er/c.txt", "z" * 30 + ".md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(rel, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    if index_version == "3":
        _git(repo, "update-index", "--intent-to-add", "--", "a.py")
    _git(repo, "update-index", "--index-version", index_version)
    _git(repo, "commit", "-qm", "x")
    _git(repo, "pack-refs", "--all")
    assert sa.git_head(repo) == _git(repo, "rev-parse", "HEAD")
    assert sa.git_tracked_files(repo) == sorted(_git(repo, "ls-files").splitlines())


def _tree(tmp_path, name, files, commit):
    d = tmp_path / name
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    sa.write_record(d, "https://github.com/o/R.git", commit, list(files))
    return d


def test_apply_tree_replaces_only_the_trees_files_and_puts_them_back(tmp_path):
    root = _tree(tmp_path, "root", {"keep.py": "old", "gone.py": "old", "same.py": "s"}, "1" * 40)
    (root / "venv").mkdir()
    (root / "venv" / "user.txt").write_text("mine", encoding="utf-8")
    new = _tree(tmp_path, "new", {"keep.py": "new", "same.py": "s", "added/x.py": "x"}, "2" * 40)
    change = sa.apply_tree(new, root, sa.tree_files(root), tmp_path / "backup")
    assert (root / "keep.py").read_text(encoding="utf-8") == "new"
    assert not (root / "gone.py").exists()
    assert (root / "added" / "x.py").is_file()
    assert (root / "venv" / "user.txt").read_text(encoding="utf-8") == "mine"
    assert sa.tree_commit(root) == "2" * 40
    assert change.rollback() == []
    assert (root / "keep.py").read_text(encoding="utf-8") == "old"
    assert (root / "gone.py").read_text(encoding="utf-8") == "old"
    assert not (root / "added" / "x.py").exists()
    assert sa.tree_commit(root) == "1" * 40
    change.discard()
    assert not (tmp_path / "backup").exists()


def test_apply_tree_puts_everything_back_when_a_write_fails(tmp_path, monkeypatch):
    root = _tree(tmp_path, "root", {"a.py": "old", "b.py": "old"}, "1" * 40)
    new = _tree(tmp_path, "new", {"a.py": "new", "b.py": "new"}, "2" * 40)
    real = shutil.copy2
    seen = []

    def flaky(src, dst, *a, **k):
        if str(src).startswith(str(sa._long(new))) and str(dst).endswith("b.py"):
            seen.append(dst)
            raise OSError("disk full")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(sa.shutil, "copy2", flaky)
    with pytest.raises(OSError, match="disk full"):
        sa.apply_tree(new, root, sa.tree_files(root), tmp_path / "backup")
    assert seen
    assert (root / "a.py").read_text(encoding="utf-8") == "old"
    assert (root / "b.py").read_text(encoding="utf-8") == "old"
    assert sa.tree_commit(root) == "1" * 40
