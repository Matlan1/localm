# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update apply ENGINE (localm/_apply_update): verify, extract, and the
swap/backup/rollback file primitives on a fake install tree. The live re-exec
orchestration is integration-level and verified on a real install; these pin the
correctness + safety of the file ops (deletions applied, never-touch preserved,
rollback restores the exact pre-apply state)."""

import zipfile

import pytest

from localm import _apply_update as au


def _make_zip(path, files, wrap=None):
    """Write a zip at *path* with {relpath: content}; optionally under a wrapper dir."""
    with zipfile.ZipFile(path, "w") as z:
        for rel, content in files.items():
            arc = f"{wrap}/{rel}" if wrap else rel
            z.writestr(arc, content)


# ------------------------------ verify ----------------------------------

def test_verify_accepts_flat_build(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "[project]", "localm/__init__.py": ""})
    au.verify_zip(zp)  # no raise


def test_verify_accepts_wrapped_build(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "[project]"}, wrap="localm-0.2.0")
    au.verify_zip(zp)  # no raise


def test_verify_rejects_non_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError):
        au.verify_zip(p)


def test_verify_rejects_zip_without_version(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"readme.md": "hi", "localm/__init__.py": ""})
    with pytest.raises(ValueError):
        au.verify_zip(zp)


def test_verify_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        au.verify_zip(tmp_path / "nope.zip")


# ------------------------------ extract ---------------------------------

def test_extract_flat(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "localm/a.py": "1"})
    root = au.extract(zp, tmp_path / "stg")
    assert (root / "VERSION").read_text().strip() == "0.2.0"
    assert (root / "localm" / "a.py").exists()


def test_extract_descends_into_wrapper(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x"}, wrap="localm-0.2.0")
    root = au.extract(zp, tmp_path / "stg")
    assert root.name == "localm-0.2.0"
    assert (root / "VERSION").exists()


# --------------------------- swap entries -------------------------------

def test_swap_entries_excludes_never_touch(tmp_path):
    staged = tmp_path / "s"
    (staged / "localm").mkdir(parents=True)
    (staged / ".venv").mkdir()
    (staged / "home").mkdir()
    (staged / "VERSION").write_text("0.2.0")
    names = au.swap_entries(staged)
    assert "localm" in names and "VERSION" in names
    assert ".venv" not in names and "home" not in names


# --------------------- swap + backup + rollback -------------------------

def _fake_install(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("old", encoding="utf-8")
    (root / "localm" / "gone.py").write_text("removed upstream", encoding="utf-8")
    (root / "VERSION").write_text("0.1.0", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "marker").write_text("keep", encoding="utf-8")
    (root / "home").mkdir()
    (root / "home" / "config.json").write_text("user data", encoding="utf-8")


def _staged_build(root):
    (root / "localm").mkdir(parents=True)
    (root / "localm" / "__init__.py").write_text("new", encoding="utf-8")
    (root / "localm" / "added.py").write_text("added", encoding="utf-8")
    (root / "VERSION").write_text("0.2.0", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")


def test_swap_applies_changes_deletions_and_preserves_never_touch(tmp_path):
    inst, staged, bdir = tmp_path / "inst", tmp_path / "staged", tmp_path / "bak"
    inst.mkdir(); staged.mkdir()
    _fake_install(inst)
    _staged_build(staged)

    names = au.swap_with_backup(staged, inst, bdir)

    # New content swapped in.
    assert (inst / "localm" / "__init__.py").read_text() == "new"
    assert (inst / "localm" / "added.py").exists()
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    assert (inst / "pyproject.toml").exists()
    # Upstream deletion applied (whole-tree replace of localm/).
    assert not (inst / "localm" / "gone.py").exists()
    # Never-touch trees untouched.
    assert (inst / ".venv" / "marker").read_text() == "keep"
    assert (inst / "home" / "config.json").read_text() == "user data"
    assert "VERSION" in names and ".venv" not in names


def test_rollback_restores_exact_pre_apply_state(tmp_path):
    inst, staged, bdir = tmp_path / "inst", tmp_path / "staged", tmp_path / "bak"
    inst.mkdir(); staged.mkdir()
    _fake_install(inst)
    _staged_build(staged)

    names = au.swap_with_backup(staged, inst, bdir)
    au.rollback(bdir, inst, names)

    assert (inst / "localm" / "__init__.py").read_text() == "old"
    assert (inst / "localm" / "gone.py").read_text() == "removed upstream"
    assert not (inst / "localm" / "added.py").exists()
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    # pyproject was NEW (absent before) -> removed by rollback, not restored.
    assert not (inst / "pyproject.toml").exists()
    # Never-touch survived throughout.
    assert (inst / ".venv" / "marker").read_text() == "keep"
    assert (inst / "home" / "config.json").read_text() == "user data"


# --------------------------- post-swap cmd ------------------------------

def test_post_swap_command_per_class():
    assert au.post_swap_command("reboot") is None
    assert au.post_swap_command("deps")[:3] == ["uv", "pip", "install"]
    assert au.post_swap_command("runtime", backend="cuda")[-1] == "cuda"
    assert au.post_swap_command("setup") is None


# ---------------------- safe extraction (zip slip) ----------------------

def test_extract_rejects_traversal_member(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "../evil.txt": "pwn"})
    with pytest.raises(ValueError):
        au.extract(zp, tmp_path / "stg")
    # Nothing escaped the staging dir.
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_absolute_member(tmp_path):
    zp = tmp_path / "b.zip"
    _make_zip(zp, {"VERSION": "0.2.0", "pyproject.toml": "x", "/abs/evil.txt": "pwn"})
    with pytest.raises(ValueError):
        au.extract(zp, tmp_path / "stg")


def test_unsafe_member_detection():
    assert au._unsafe_member("../evil") is True
    assert au._unsafe_member("a/../../evil") is True
    assert au._unsafe_member("/etc/passwd") is True
    assert au._unsafe_member("C:/windows") is True
    assert au._unsafe_member("localm/foo.py") is False
    assert au._unsafe_member("VERSION") is False
    assert au._unsafe_member("a..b/c") is False   # '..' only as a path component counts


# ----------------- rollback surfaces restore failures -------------------

def test_rollback_raises_when_a_restore_fails(tmp_path, monkeypatch):
    inst, bdir = tmp_path / "inst", tmp_path / "bak"
    inst.mkdir(); bdir.mkdir()
    (bdir / "localm").mkdir()
    (bdir / "localm" / "x.py").write_text("orig", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(au.shutil, "copytree", boom)   # restore fails
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        au.rollback(bdir, inst, ["localm"])


def test_swap_with_backup_surfaces_double_failure(tmp_path, monkeypatch):
    staged, inst, bdir = tmp_path / "s", tmp_path / "i", tmp_path / "b"
    staged.mkdir(); inst.mkdir()
    (staged / "VERSION").write_text("0.2.0", encoding="utf-8")

    def raise_os(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(au, "apply_files", raise_os)   # swap fails
    monkeypatch.setattr(au, "rollback", raise_os)      # AND rollback fails
    with pytest.raises(RuntimeError, match="rollback also failed"):
        au.swap_with_backup(staged, inst, bdir)
