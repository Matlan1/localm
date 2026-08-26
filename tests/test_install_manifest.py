# SPDX-License-Identifier: AGPL-3.0-or-later
"""The install provenance ledger: uninstall removes ONLY what record() wrote,
never an unrecorded path, and the data-dir rm -rf is guarded against the
Steam/Pop!_OS class of disaster (root, $HOME, repo, ancestors, symlinks).
"""

from __future__ import annotations

import json
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
    root_anchor = Path(tmp_path.anchor)           # "/" or "Z:\\"
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


def test_unsafe_data_dir_refuses_when_home_cannot_be_resolved(tmp_path, monkeypatch):
    # A broken Path.home() must make the guard REFUSE the path, not silently
    # treat it as safe just because the $HOME comparison itself blew up.
    def _broken_home():
        raise RuntimeError("could not determine home directory")
    monkeypatch.setattr(Path, "home", staticmethod(_broken_home))
    assert im._unsafe_data_dir(str(tmp_path / "data"), tmp_path) is not None


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


# ------------------------------ schema v2 --------------------------------- #

def _v2_lib(root: Path) -> Path:
    lib = root / "runtime" / "lib"
    lib.mkdir(parents=True)
    return lib


def test_record_v2_fields_roundtrip(tmp_path):
    lib = _v2_lib(tmp_path)
    pydir = tmp_path / ".python"; pydir.mkdir()
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=True, python_dir=str(pydir),
              cache_dir=str(tmp_path / ".cache"),
              path_dir=str(tmp_path / "bin"),
              command_shim=str(tmp_path / "bin" / "localm.cmd"),
              path_modified=True)
    m = im.load(tmp_path)
    assert m["schema"] == 2
    assert m["runtime_contained"] is True
    assert Path(m["python_dir"]) == pydir.resolve()
    assert m["path_modified"] is True


def test_uninstall_removes_contained_runtime(tmp_path):
    lib = _v2_lib(tmp_path)
    pydir = tmp_path / ".python"; pydir.mkdir()
    (pydir / "python").write_text("x", encoding="utf-8")
    cachedir = tmp_path / ".cache"; cachedir.mkdir()
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=True, python_dir=str(pydir), cache_dir=str(cachedir))
    rep = im.uninstall(tmp_path)
    assert not pydir.exists()                          # contained -> removed
    assert not cachedir.exists()
    assert str(pydir.resolve()) in rep["removed"]


def test_uninstall_keeps_shared_runtime(tmp_path):
    # A SHARED runtime lives outside the clone and is reused by other clones: it
    # must be reported, NEVER deleted.
    lib = _v2_lib(tmp_path)
    shared = tmp_path / "elsewhere" / "uvpython"; shared.mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=False, python_dir=str(shared))
    rep = im.uninstall(tmp_path)
    assert shared.exists()                             # never deleted
    assert any("shared runtime kept" in why for _, why in rep["skipped"])


def test_uv_dir_roundtrips_and_is_removed_when_contained(tmp_path):
    # A Portable install that had to bootstrap uv itself confines uv's OWN binary
    # dir to the clone too (not just the Python runtime it manages) - this is the
    # v2 "uv_dir" field. Removed on uninstall exactly like python_dir/cache_dir.
    lib = _v2_lib(tmp_path)
    uvdir = tmp_path / ".uv"; uvdir.mkdir()
    (uvdir / "uv.exe").write_text("x", encoding="utf-8")
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=True, uv_dir=str(uvdir))
    m = im.load(tmp_path)
    assert Path(m["uv_dir"]) == uvdir.resolve()
    rep = im.uninstall(tmp_path)
    assert not uvdir.exists()
    assert str(uvdir.resolve()) in rep["removed"]


def test_uv_dir_kept_when_shared(tmp_path):
    # uv was already on PATH (a pre-existing / Shared install) - setup never
    # installed it into this clone, so uv_dir is blank and there is nothing to
    # remove; a NON-blank uv_dir recorded without runtime_contained (e.g. a
    # user-edited manifest) must still be reported, never deleted.
    lib = _v2_lib(tmp_path)
    shared_uv = tmp_path / "elsewhere" / "uv"; shared_uv.mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=False, uv_dir=str(shared_uv))
    rep = im.uninstall(tmp_path)
    assert shared_uv.exists()
    assert any("shared runtime kept" in why for _, why in rep["skipped"])


def test_uninstall_reverses_global_command(tmp_path, monkeypatch):
    import localm.globalcmd as gc
    calls = {}

    def fake_uninstall_command(path_dir, shim):
        calls["args"] = (path_dir, shim)
        return {"removed": [shim, f"PATH entry {path_dir}"], "notes": []}

    monkeypatch.setattr(gc, "uninstall_command", fake_uninstall_command)
    lib = _v2_lib(tmp_path)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              path_dir=str(tmp_path / "bin"),
              command_shim=str(tmp_path / "bin" / "localm.cmd"), path_modified=True)
    rep = im.uninstall(tmp_path)
    assert calls["args"][0] == str((tmp_path / "bin").resolve())
    assert any("PATH entry" in x for x in rep["removed"])


def test_v1_manifest_still_uninstalls(tmp_path):
    # An OLD schema-1 manifest (no v2 keys) must still uninstall cleanly (missing
    # keys read as absent).
    lib = _v2_lib(tmp_path)
    (lib / "llama.dll").write_text("x", encoding="utf-8")
    im.manifest_path(tmp_path).write_text(json.dumps({
        "schema": 1, "venv": str(tmp_path / ".venv"), "lib_dir": str(lib),
        "binaries": ["llama.dll"], "home_cfg": "", "data_dir": "",
        "data_created": False, "shortcut": "",
    }), encoding="utf-8")
    rep = im.uninstall(tmp_path)
    assert rep["ok"]
    assert not (lib / "llama.dll").exists()


def test_uninstall_skips_path_when_not_modified(tmp_path, monkeypatch):
    # path_modified=False (we installed the shim but PATH was already set): uninstall
    # must remove our shim but NOT strip a PATH dir we did not add ourselves.
    import localm.globalcmd as gc
    seen = {}

    def fake_uninstall_command(path_dir, shim):
        seen["path_dir"] = path_dir
        return {"removed": [shim] if shim else [], "notes": []}

    monkeypatch.setattr(gc, "uninstall_command", fake_uninstall_command)
    lib = _v2_lib(tmp_path)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              path_dir=str(tmp_path / "bin"),
              command_shim=str(tmp_path / "bin" / "localm.cmd"), path_modified=False)
    im.uninstall(tmp_path)
    assert seen["path_dir"] == ""            # PATH we did not change is left alone
