"""The install provenance ledger: uninstall removes ONLY what record() wrote,
never an unrecorded path, and the data-dir rm -rf is guarded against the
Steam/Pop!_OS class of disaster (root, $HOME, repo, ancestors, symlinks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from localm import install_manifest as im


def _fake_install(root: Path, *, data_created=True):
    """Build a recorded install layout under *root* and return key paths."""
    lib = root / "runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_text("x", encoding="utf-8")
    (lib / "ggml.dll").write_text("y", encoding="utf-8")
    unknown = lib / "somebody-elses.dll"          # present but NOT recorded below
    cfg = root / "localm-home.cfg"
    cfg.write_text(str(root / "data"), encoding="utf-8")
    shortcut = root / "localm.lnk"
    shortcut.write_text("lnk", encoding="utf-8")
    data = root / "data"
    data.mkdir()
    (data / "model.gguf").write_text("z", encoding="utf-8")
    venv = root / ".venv"
    venv.mkdir()

    # Record BEFORE creating the unknown file, so it is not in the binary snapshot.
    im.record(root, venv=str(venv), lib_dir=str(lib), home_cfg=str(cfg),
              data_dir=str(data), data_created=data_created, shortcut=str(shortcut))
    unknown.write_text("not ours", encoding="utf-8")
    return {"lib": lib, "cfg": cfg, "shortcut": shortcut, "data": data,
            "venv": venv, "unknown": unknown}


def test_record_and_load_roundtrip(tmp_path):
    p = _fake_install(tmp_path)
    m = im.load(tmp_path)
    assert m["schema"] == im.SCHEMA_VERSION
    assert sorted(m["binaries"]) == ["ggml.dll", "llama.dll"]    # snapshot, no unknown
    assert m["data_created"] is True
    assert Path(m["venv"]) == p["venv"].resolve()


def test_uninstall_removes_only_recorded_items(tmp_path):
    p = _fake_install(tmp_path)
    rep = im.uninstall(tmp_path, purge_data=False)
    assert rep["ok"]
    # recorded binaries + cfg + shortcut gone
    assert not (p["lib"] / "llama.dll").exists()
    assert not (p["lib"] / "ggml.dll").exists()
    assert not p["cfg"].exists()
    assert not p["shortcut"].exists()
    # the UNRECORDED file is left strictly alone
    assert p["unknown"].exists()
    # data kept (no --purge-data), venv reported not removed, manifest gone
    assert p["data"].exists()
    assert p["venv"].exists()
    assert Path(rep["venv"]) == p["venv"].resolve()
    assert im.load(tmp_path) is None


def test_uninstall_purge_data_removes_data_when_we_created_it(tmp_path):
    p = _fake_install(tmp_path, data_created=True)
    rep = im.uninstall(tmp_path, purge_data=True)
    assert not p["data"].exists()
    assert str(p["data"].resolve()) in rep["removed"]


def test_purge_data_not_ours_is_warned_not_removed_without_force(tmp_path):
    p = _fake_install(tmp_path, data_created=False)
    rep = im.uninstall(tmp_path, purge_data=True)     # no force
    assert p["data"].exists()                         # kept - not removed by default
    assert any("not recorded as created" in why for _, why in rep["warned"])


def test_purge_data_not_ours_removed_with_force(tmp_path):
    p = _fake_install(tmp_path, data_created=False)
    im.uninstall(tmp_path, purge_data=True, force=True)   # user accepted the risk
    assert not p["data"].exists()                     # safe nested path -> removed


def test_no_manifest_warns_then_force_removes_known_binaries(tmp_path):
    lib = tmp_path / "runtime" / "localm_llama_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_text("x", encoding="utf-8")
    # No record: warned, but nothing removed without force.
    rep = im.uninstall(tmp_path)
    assert rep["no_manifest"] is True
    assert rep["removed"] == []
    assert any("no install record" in why for _, why in rep["warned"])
    assert (lib / "llama.dll").exists()
    # With force (the dragons warning was shown and accepted): removed.
    im.uninstall(tmp_path, force=True)
    assert not (lib / "llama.dll").exists()


def test_bad_schema_aborts(tmp_path):
    im.manifest_path(tmp_path).write_text('{"schema": 999}', encoding="utf-8")
    rep = im.uninstall(tmp_path, purge_data=True)
    assert rep["ok"] is False
    assert rep["removed"] == []


def test_dry_run_touches_nothing(tmp_path):
    p = _fake_install(tmp_path)
    rep = im.uninstall(tmp_path, purge_data=True, dry_run=True)
    assert (p["lib"] / "llama.dll").exists()      # still there
    assert p["data"].exists()
    assert im.load(tmp_path) is not None          # manifest preserved
    assert rep["removed"]                         # but the plan is reported


# --------------------------- the rm -rf guard ----------------------------- #

def test_unsafe_data_dir_refuses_dangerous_targets(tmp_path):
    repo = tmp_path
    root_anchor = Path(tmp_path.anchor)           # "/" or "C:\\"
    assert im._unsafe_data_dir("", repo)
    assert im._unsafe_data_dir("relative/dir", repo)
    assert im._unsafe_data_dir(str(root_anchor), repo)
    assert im._unsafe_data_dir(str(Path.home()), repo)
    assert im._unsafe_data_dir(str(repo), repo)              # repo root
    assert im._unsafe_data_dir(str(repo.parent), repo)       # ancestor of repo
    # a genuine nested data dir is allowed
    assert im._unsafe_data_dir(str(repo / "data"), repo) is None


@pytest.mark.skipif(sys.platform == "win32",
                    reason="symlink creation usually needs privilege on Windows")
def test_unsafe_data_dir_refuses_symlink(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert im._unsafe_data_dir(str(link), tmp_path)           # symlink refused


def test_uninstall_refuses_when_data_dir_is_unsafe(tmp_path):
    # Record a manifest whose data_dir is the repo root itself, then purge.
    (tmp_path / "runtime" / "lib").mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"),
              lib_dir=str(tmp_path / "runtime" / "lib"),
              data_dir=str(tmp_path), data_created=True)
    (tmp_path / "keepme.txt").write_text("important", encoding="utf-8")
    rep = im.uninstall(tmp_path, purge_data=True, force=True)  # NOT overridable
    assert (tmp_path / "keepme.txt").exists()                 # repo not nuked even forced
    assert any("repository root" in why for _, why in rep["refused"])
