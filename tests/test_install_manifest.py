# SPDX-License-Identifier: AGPL-3.0-or-later
"""The install provenance ledger and the uninstaller built on it.

Uninstall removes what setup recorded plus LocaLM's own fixed in-clone
locations, never an unrecorded path elsewhere, keeps the saved data unless
asked, deletes only LocaLM's own entries from a data folder that existed
before setup, and never deletes the Python runtime it is running on (that is
deferred to the calling script). The data-dir rm -rf stays guarded against
root, $HOME, the repo, its ancestors and symlinks.

The machine is never touched: the user PATH is an in-memory fake, the home
folder and app-data folders point into tmp_path, and the only processes that
are stopped are ones these tests started.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import globalcmd as gc
from localm import install_manifest as im

LOCALM = Path(im.__file__).resolve().parent
REAL_LIST_PROCESSES = im.list_processes


class _FakePath:
    """In-memory HKCU\\Environment\\Path."""

    def __init__(self, value=""):
        self.value = value

    def read(self):
        return self.value, 2

    def write(self, value, regtype):
        self.value = value


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Home, app data and the user PATH all point at tmp_path; process listing
    returns nothing unless a test asks for the real one."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "USERPROFILE"):
        d = tmp_path / ("env-" + var.lower())
        d.mkdir()
        monkeypatch.setenv(var, str(d))
    fp = _FakePath()
    monkeypatch.setattr(gc, "_win_read_user_path", fp.read)
    monkeypatch.setattr(gc, "_win_write_user_path", fp.write)
    monkeypatch.setattr(im, "list_processes", lambda: [])
    return {"home": home, "path": fp}


def _clone(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime" / "localm_llama_runtime" / "lib").mkdir(parents=True)
    return root


def _fake_install(root: Path, *, data_created=True):
    """A recorded install under *root*; returns the key paths."""
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

    im.record(root, venv=str(venv), lib_dir=str(lib), home_cfg=str(cfg),
              data_dir=str(data), data_created=data_created, shortcut=str(shortcut))
    unknown.write_text("not ours", encoding="utf-8")
    return {"lib": lib, "cfg": cfg, "shortcut": shortcut, "data": data,
            "venv": venv, "unknown": unknown}


# ------------------------------ recording --------------------------------- #

def test_record_and_load_roundtrip(tmp_path):
    p = _fake_install(tmp_path)
    m = im.load(tmp_path)
    assert m["schema"] == im.SCHEMA_VERSION
    assert sorted(m["lib_entries"]) == ["ggml.dll", "llama.dll"]  # snapshot, no unknown
    assert m["data_created"] is True
    assert Path(m["venv"]) == p["venv"].resolve()


def test_lib_snapshot_takes_every_provisioned_entry_but_the_git_sentinels(tmp_path):
    lib = _clone(tmp_path) / "runtime" / "localm_llama_runtime" / "lib"
    for name in ("llama.dll", "LICENSE.llama-cpp", ".localm-backend", ".gitkeep", ".gitignore"):
        (lib / name).write_text("x", encoding="utf-8")
    (lib / "rocblas" / "library").mkdir(parents=True)
    im.record(tmp_path, lib_dir=str(lib))
    assert im.load(tmp_path)["lib_entries"] == sorted(
        [".localm-backend", "LICENSE.llama-cpp", "llama.dll", "rocblas"])


def test_record_merges_instead_of_overwriting(tmp_path):
    im.record(tmp_path, venv=str(tmp_path / ".venv"), runtime_contained=True,
              python_dir=str(tmp_path / ".python"), path_modified=True,
              files=[str(tmp_path / "LocaLM.desktop")])
    im.record(tmp_path, shortcut=str(tmp_path / "LocaLM.lnk"), stamp="S",
              files=[str(tmp_path / "LocaLM.desktop"), str(tmp_path / "other.txt")])
    m = im.load(tmp_path)
    assert Path(m["venv"]) == (tmp_path / ".venv").resolve()
    assert Path(m["python_dir"]) == (tmp_path / ".python").resolve()
    assert m["runtime_contained"] is True and m["path_modified"] is True
    assert Path(m["shortcut"]) == (tmp_path / "LocaLM.lnk").resolve()
    assert m["stamp"] == "S"
    assert [Path(f).name for f in m["files"]] == ["LocaLM.desktop", "other.txt"]


def test_record_refuses_a_newer_schema(tmp_path):
    im.manifest_path(tmp_path).write_text('{"schema": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        im.record(tmp_path, venv=str(tmp_path / ".venv"))
    assert im.main(["record", "--root", str(tmp_path), "--venv", "x"]) == 1
    assert json.loads(im.manifest_path(tmp_path).read_text())["schema"] == 999


def test_v2_manifest_is_upgraded_and_its_custom_data_claim_dropped(tmp_path):
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    shared = tmp_path / "shared-data"
    im.manifest_path(tmp_path).write_text(json.dumps({
        "schema": 2, "venv": str(tmp_path / ".venv"), "lib_dir": str(lib),
        "binaries": ["llama.dll"], "data_dir": str(shared), "data_created": True,
    }), encoding="utf-8")
    im.record(tmp_path, stamp="later")
    m = im.load(tmp_path)
    assert m["schema"] == 3
    assert m["lib_entries"] == ["llama.dll"] and "binaries" not in m
    assert m["data_created"] is False and m["data_preexisting"] is None


# ------------------------------ data folder ------------------------------- #

def test_prepare_portable_creates_home_and_drops_the_pointer(tmp_path):
    (tmp_path / "localm-home.cfg").write_text("X:\\old", encoding="utf-8")
    target = im.prepare_data(tmp_path, portable=True)
    assert target == tmp_path.resolve() / "home"
    assert (target / im.DATA_MARKER).is_file()
    assert not (tmp_path / "localm-home.cfg").exists()
    m = im.load(tmp_path)
    assert Path(m["data_dir"]) == target and m["data_created"] is True


def test_prepare_custom_new_folder_records_created_parents_and_utf8_pointer(tmp_path):
    target = tmp_path / "Jösé 数据" / "a" / "LocaLM"
    im.prepare_data(tmp_path, data_dir=str(target))
    assert target.is_dir() and (target / im.DATA_MARKER).is_file()
    cfg = (tmp_path / "localm-home.cfg").read_bytes().decode("utf-8")
    assert cfg.strip() == str(target)
    m = im.load(tmp_path)
    assert m["data_created"] is True and m["data_preexisting"] == []
    assert [Path(p) for p in m["data_parents_created"]] == [
        tmp_path / "Jösé 数据", tmp_path / "Jösé 数据" / "a"]
    assert Path(m["home_cfg"]) == (tmp_path / "localm-home.cfg").resolve()


def test_prepare_existing_folder_remembers_what_was_already_there(tmp_path):
    shared = tmp_path / "AI"
    (shared / "models").mkdir(parents=True)
    (shared / "ComfyUI").mkdir()
    (shared / "notes.txt").write_text("mine", encoding="utf-8")
    im.prepare_data(tmp_path, data_dir=str(shared))
    m = im.load(tmp_path)
    assert m["data_created"] is False
    assert m["data_preexisting"] == ["ComfyUI", "models", "notes.txt"]


def test_prepare_earlier_localm_folder_counts_its_localm_entries_as_localm(tmp_path):
    old = tmp_path / "kept-data"
    (old / "models").mkdir(parents=True)
    (old / "config.json").write_text("{}", encoding="utf-8")
    (old / im.DATA_MARKER).write_text("{}", encoding="utf-8")
    (old / "holiday.jpg").write_text("mine", encoding="utf-8")
    im.prepare_data(tmp_path, data_dir=str(old))
    assert im.load(tmp_path)["data_preexisting"] == ["holiday.jpg"]


@pytest.mark.parametrize("bad", ["relative\\dir", "relative/dir", ""])
def test_prepare_refuses_a_relative_or_empty_path(tmp_path, bad, capsys):
    with pytest.raises(ValueError):
        im.prepare_data(tmp_path, data_dir=bad)
    assert not (tmp_path / "localm-home.cfg").exists()
    assert im.main(["prepare-data", "--root", str(tmp_path), "--data-dir", bad or " "]) == 1
    assert "Cannot use that data folder" in capsys.readouterr().out


def test_prepare_refuses_a_file(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        im.prepare_data(tmp_path, data_dir=str(f))


def test_prepare_again_on_its_own_folder_keeps_it_owned(tmp_path):
    target = tmp_path / "LocaLM-data"
    im.prepare_data(tmp_path, data_dir=str(target))
    (target / "models").mkdir()
    im.prepare_data(tmp_path, data_dir=str(target))
    assert im.load(tmp_path)["data_created"] is True


# ------------------------------ uninstall --------------------------------- #

def test_uninstall_removes_only_recorded_items(tmp_path):
    p = _fake_install(tmp_path)
    rep = im.uninstall(tmp_path, purge_data=False)
    assert rep["ok"] and rep["exit"] == im.EXIT_OK
    assert not (p["lib"] / "llama.dll").exists()
    assert not (p["lib"] / "ggml.dll").exists()
    assert not p["cfg"].exists()
    assert not p["shortcut"].exists()
    assert p["unknown"].exists()                      # unrecorded: warned, kept
    assert any("not in the install record" in why for _, why in rep["warned"])
    assert p["data"].exists()
    assert not p["venv"].exists()
    assert im.load(tmp_path) is None


def test_uninstall_purge_removes_a_folder_setup_created(tmp_path):
    target = tmp_path / "made" / "LocaLM-data"
    im.prepare_data(tmp_path, data_dir=str(target))
    (target / "models").mkdir()
    (target / "models" / "m.gguf").write_text("z", encoding="utf-8")
    rep = im.uninstall(tmp_path, purge_data=True)
    assert not target.exists()
    assert not (tmp_path / "made").exists()           # the parent setup created, now empty
    assert str(target) in rep["removed"]


def test_created_claim_without_the_marker_is_not_trusted(tmp_path):
    p = _fake_install(tmp_path, data_created=True)    # recorded created, no marker
    (p["data"] / "config.json").write_text("{}", encoding="utf-8")
    im.uninstall(tmp_path, purge_data=True, force=True)
    assert p["data"].is_dir()
    assert (p["data"] / "model.gguf").exists()        # not LocaLM's name: kept
    assert not (p["data"] / "config.json").exists()   # LocaLM's own entry: deleted


def test_purge_in_a_folder_that_already_existed_deletes_only_localm_entries(tmp_path):
    shared = tmp_path / "AI"
    (shared / "models").mkdir(parents=True)           # the user's own models folder
    (shared / "models" / "theirs.safetensors").write_text("x", encoding="utf-8")
    (shared / "notes.txt").write_text("mine", encoding="utf-8")
    im.prepare_data(tmp_path, data_dir=str(shared))
    (shared / "config.json").write_text("{}", encoding="utf-8")
    (shared / "sessions").mkdir()
    (shared / "config.json.lock").write_text("", encoding="utf-8")
    (shared / "config.json.123.tmp").write_text("", encoding="utf-8")
    (shared / "comfy-launch-abcdef123456.log").write_text("", encoding="utf-8")
    (shared / "later-app-output").mkdir()             # another program, after setup

    rep = im.uninstall(tmp_path, purge_data=True, force=True)

    assert shared.is_dir()
    assert (shared / "models" / "theirs.safetensors").exists()
    assert (shared / "notes.txt").exists()
    assert (shared / "later-app-output").is_dir()
    for gone in ("config.json", "sessions", im.DATA_MARKER, "config.json.lock",
                 "config.json.123.tmp", "comfy-launch-abcdef123456.log"):
        assert not (shared / gone).exists(), gone
    data = next(d for d in rep["data"] if Path(d["path"]) == shared)
    assert "models" in data["kept_entries"] and "notes.txt" in data["kept_entries"]


def test_legacy_custom_folder_keeps_everything_not_localm(tmp_path):
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "config.json").write_text("{}", encoding="utf-8")
    (shared / "photos").mkdir()
    im.manifest_path(tmp_path).write_text(json.dumps({
        "schema": 2, "venv": "", "lib_dir": str(lib), "binaries": [],
        "data_dir": str(shared), "data_created": True}), encoding="utf-8")
    im.uninstall(tmp_path, purge_data=True, force=True)
    assert shared.is_dir() and (shared / "photos").is_dir()
    assert not (shared / "config.json").exists()


def test_portable_home_is_localms_even_without_a_record(tmp_path):
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    (home / "anything").write_text("x", encoding="utf-8")
    im.uninstall(tmp_path, purge_data=False)
    assert home.is_dir()
    im.uninstall(tmp_path, purge_data=True)
    assert not home.exists()


def test_pointer_file_names_a_data_folder_the_record_does_not(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "registry.json").write_text("{}", encoding="utf-8")
    (other / "family.mp4").write_text("x", encoding="utf-8")
    (tmp_path / "localm-home.cfg").write_text(str(other) + "\n", encoding="utf-8")
    im.uninstall(tmp_path, purge_data=True, force=True)
    assert (other / "family.mp4").exists()
    assert not (other / "registry.json").exists()


def test_kept_data_is_reported_with_its_size(tmp_path):
    im.prepare_data(tmp_path, portable=True)
    (tmp_path / "home" / "big.bin").write_bytes(b"x" * 2048)
    rep = im.uninstall(tmp_path, dry_run=True)
    d = rep["data"][0]
    assert d["delete"] is False and d["bytes"] >= 2048 and d["files"] >= 2
    text = "\n".join(im.format_report(rep))
    assert "KEPT:" in text and "KB" in text


def test_data_folder_link_entries_lose_the_link_not_the_target(tmp_path):
    target = tmp_path / "real-models"
    target.mkdir()
    (target / "keep.gguf").write_text("x", encoding="utf-8")
    shared = tmp_path / "AI"
    shared.mkdir()
    im.prepare_data(tmp_path, data_dir=str(shared))
    link = shared / "models"
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)
    im.uninstall(tmp_path, purge_data=True, force=True)
    assert not os.path.lexists(link)
    assert (target / "keep.gguf").exists()


def test_purge_data_refused_for_unsafe_folder(tmp_path):
    (tmp_path / "runtime" / "lib").mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"),
              lib_dir=str(tmp_path / "runtime" / "lib"),
              data_dir=str(tmp_path), data_created=True)
    (tmp_path / "keepme.txt").write_text("important", encoding="utf-8")
    rep = im.uninstall(tmp_path, purge_data=True, force=True)
    assert (tmp_path / "keepme.txt").exists()
    assert any("repository root" in why for _, why in rep["refused"])
    assert rep["exit"] == im.EXIT_PARTIAL


def test_no_manifest_warns_then_force_removes_known_binaries(tmp_path):
    lib = tmp_path / "runtime" / "localm_llama_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_text("x", encoding="utf-8")
    rep = im.uninstall(tmp_path)
    assert rep["no_manifest"] is True
    assert rep["removed"] == []
    assert any("no install record" in why for _, why in rep["warned"])
    assert (lib / "llama.dll").exists()
    im.uninstall(tmp_path, force=True)
    assert not (lib / "llama.dll").exists()


def test_bad_schema_aborts(tmp_path):
    im.manifest_path(tmp_path).write_text('{"schema": 999}', encoding="utf-8")
    rep = im.uninstall(tmp_path, purge_data=True)
    assert rep["ok"] is False and rep["exit"] == im.EXIT_ABORTED
    assert rep["removed"] == []


def test_dry_run_touches_nothing(tmp_path):
    p = _fake_install(tmp_path)
    rep = im.uninstall(tmp_path, purge_data=True, dry_run=True)
    assert (p["lib"] / "llama.dll").exists()
    assert p["data"].exists() and p["venv"].exists()
    assert im.load(tmp_path) is not None
    assert rep["removed"]


def test_poisoned_record_cannot_reach_outside_the_clone(tmp_path):
    outside = tmp_path / "system"
    outside.mkdir()
    victim = outside / "kernel32.dll"
    victim.write_text("x", encoding="utf-8")
    note = outside / "resume.docx"
    note.write_text("x", encoding="utf-8")
    clone = tmp_path / "clone"
    clone.mkdir()
    im.manifest_path(clone).write_text(json.dumps({
        "schema": 3, "venv": "", "lib_dir": str(outside), "lib_entries": ["kernel32.dll"],
        "shortcut": str(note), "files": [str(note)],
        "home_cfg": str(outside / "localm-home.cfg"), "data_dir": "",
    }), encoding="utf-8")
    rep = im.uninstall(clone, force=True)
    assert victim.exists() and note.exists()
    reasons = " | ".join(why for _, why in rep["refused"])
    assert "runtime folder recorded outside this folder" in reasons
    assert "not a LocaLM shortcut" in reasons


# ------------------------- the runtime it runs on ------------------------- #

def _runtime_layout(root: Path) -> None:
    for name in (".venv", ".python", ".cache", ".uv"):
        (root / name).mkdir(parents=True)
        (root / name / "f").write_text("x", encoding="utf-8")
    (root / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
    im.record(root, venv=str(root / ".venv"), runtime_contained=True,
              python_dir=str(root / ".python"), cache_dir=str(root / ".cache"),
              uv_dir=str(root / ".uv"))


def test_defer_runtime_leaves_the_runtime_for_the_script(tmp_path):
    _runtime_layout(tmp_path)
    rep = im.uninstall(tmp_path, defer_runtime=True)
    assert rep["exit"] == im.EXIT_OK
    for name in (".venv", ".python", ".cache", ".uv"):
        assert (tmp_path / name).is_dir(), name
    assert sorted(im.pending_path(tmp_path).read_text(encoding="ascii").split()) == \
        [".cache", ".python", ".uv", ".venv"]
    assert im.load(tmp_path) is not None                # kept until the script finishes
    removed, left = im.finish_pending(tmp_path)
    assert not left and len(removed) == 4
    assert not im.pending_path(tmp_path).exists()
    assert im.load(tmp_path) is None


def test_finish_pending_ignores_names_it_does_not_own(tmp_path):
    (tmp_path / "src").mkdir()
    im.pending_path(tmp_path).write_text("src\n..\n.venv\n", encoding="ascii")
    im.finish_pending(tmp_path)
    assert (tmp_path / "src").is_dir()


def test_the_running_interpreters_folder_is_deferred_not_deleted(tmp_path, monkeypatch):
    _runtime_layout(tmp_path)
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / ".python" / "cpython"))
    rep = im.uninstall(tmp_path)
    assert (tmp_path / ".python").is_dir()
    assert str(tmp_path.resolve() / ".python") in rep["deferred"]
    assert not (tmp_path / ".cache").exists()           # not in use: removed in-process


def test_contained_runtime_removed_in_process_and_shared_kept(tmp_path):
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    pydir = tmp_path / ".python"
    pydir.mkdir()
    shared = tmp_path / "elsewhere" / "uvpython"
    shared.mkdir(parents=True)
    im.record(tmp_path, lib_dir=str(lib), runtime_contained=True, python_dir=str(pydir))
    im.uninstall(tmp_path)
    assert not pydir.exists()
    im.record(tmp_path, lib_dir=str(lib), runtime_contained=False, python_dir=str(shared))
    rep = im.uninstall(tmp_path)
    assert shared.exists()
    assert any("shared runtime kept" in why for _, why in rep["skipped"])


def test_a_failure_keeps_the_record_and_the_runtime(tmp_path):
    _runtime_layout(tmp_path)
    lib = tmp_path / "runtime" / "localm_llama_runtime" / "lib"
    lib.mkdir(parents=True)
    locked = lib / "llama.dll"
    locked.write_text("x", encoding="utf-8")
    im.record(tmp_path, lib_dir=str(lib))
    if sys.platform == "win32":
        holder = open(locked, "rb")                     # no delete sharing on Windows
    else:
        os.chmod(lib, 0o500)
        holder = None
    try:
        rep = im.uninstall(tmp_path, defer_runtime=True)
    finally:
        if holder:
            holder.close()
        else:
            os.chmod(lib, 0o700)
    if not locked.exists():
        pytest.skip("this account can delete from a read-only folder")
    assert rep["exit"] == im.EXIT_FAILED
    assert rep["deferred"] == []
    assert not im.pending_path(tmp_path).exists()
    assert im.load(tmp_path) is not None
    assert (tmp_path / ".python").is_dir() and (tmp_path / ".venv").is_dir()


# --------------------- entries that point into the clone ------------------ #

def test_user_path_entries_inside_the_clone_are_removed(tmp_path, isolated, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    clone = tmp_path / "clone"
    clone.mkdir()
    other = str(tmp_path / "tools")
    isolated["path"].value = os.pathsep.join([other, str(clone / ".uv"), str(clone / "bin")])
    rep = im.uninstall(clone)
    assert isolated["path"].value == other
    assert any(x.startswith("PATH entry") for x in rep["removed"])


def test_uv_rc_lines_for_this_folder_are_removed_byte_exactly(tmp_path, isolated, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    home = isolated["home"]
    clone = home / "localm"
    (clone / ".uv").mkdir(parents=True)
    im.record(clone, runtime_contained=True, uv_dir=str(clone / ".uv"))
    bashrc = home / ".bashrc"
    bashrc.write_bytes(b"alias ll='ls -l'\r\n\r\n. \"$HOME/localm/.uv/env\"\r\nexport X=1\r\n")
    profile = home / ".profile"
    profile.write_text(f'. "{(clone / ".uv").as_posix()}/env"\n. "$HOME/other/.uv/env"\n',
                       encoding="utf-8")
    fish = home / ".config" / "fish" / "conf.d" / "uv.env.fish"
    fish.parent.mkdir(parents=True)
    fish.write_text('\nsource "$HOME/localm/.uv/env.fish"\n', encoding="utf-8")
    rep = im.uninstall(clone)
    assert bashrc.read_bytes() == b"alias ll='ls -l'\r\n\r\nexport X=1\r\n"
    assert profile.read_text(encoding="utf-8") == '. "$HOME/other/.uv/env"\n'
    assert not fish.exists()
    assert sum(x.startswith("uv line in") for x in rep["removed"]) == 3


def test_uv_rc_lines_follow_the_uv_folder_decision(tmp_path, isolated, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    home = isolated["home"]
    clone = home / "localm"
    (clone / ".uv").mkdir(parents=True)                 # unrecorded: only a warning
    bashrc = home / ".bashrc"
    bashrc.write_text('. "$HOME/localm/.uv/env"\n', encoding="utf-8")
    im.uninstall(clone)                                 # no force
    assert (clone / ".uv").is_dir()
    assert bashrc.read_text(encoding="utf-8") == '. "$HOME/localm/.uv/env"\n'


def test_uv_receipt_is_removed_only_when_it_names_this_folder(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    clone.mkdir()
    receipt = Path(os.environ["LOCALAPPDATA"]) / "uv" / "uv-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"install_prefix": str(clone / ".uv")}), encoding="utf-8")
    im.uninstall(clone)
    assert not receipt.exists()
    receipt.write_text(json.dumps({"install_prefix": str(tmp_path / "global")}),
                       encoding="utf-8")
    im.uninstall(clone)
    assert receipt.exists()


# ------------------------------ global command ---------------------------- #

def test_uninstall_reverses_global_command(tmp_path, monkeypatch):
    calls = {}

    def fake_uninstall_command(path_dir, shim):
        calls["args"] = (path_dir, shim)
        return {"removed": [shim, f"PATH entry {path_dir}"], "notes": []}

    monkeypatch.setattr(gc, "uninstall_command", fake_uninstall_command)
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              path_dir=str(tmp_path / "bin"),
              command_shim=str(tmp_path / "bin" / "localm.cmd"), path_modified=True)
    rep = im.uninstall(tmp_path)
    assert calls["args"][0] == str((tmp_path / "bin").resolve())
    assert any("PATH entry" in x for x in rep["removed"])


def test_uninstall_skips_path_when_not_modified(tmp_path, monkeypatch):
    seen = {}

    def fake_uninstall_command(path_dir, shim):
        seen["path_dir"] = path_dir
        return {"removed": [shim] if shim else [], "notes": []}

    monkeypatch.setattr(gc, "uninstall_command", fake_uninstall_command)
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              path_dir=str(tmp_path / "bin"),
              command_shim=str(tmp_path / "bin" / "localm.cmd"), path_modified=False)
    im.uninstall(tmp_path)
    assert seen["path_dir"] == ""


def test_a_failing_global_command_is_a_failure_not_a_note(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "uninstall_command", lambda p, s: {
        "removed": [], "notes": [f"could not remove shim {s}: denied"]})
    im.record(tmp_path, command_shim=str(tmp_path / "bin" / "localm.cmd"))
    rep = im.uninstall(tmp_path)
    assert rep["exit"] == im.EXIT_FAILED
    assert im.load(tmp_path) is not None


def test_v1_manifest_still_uninstalls(tmp_path):
    lib = tmp_path / "runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_text("x", encoding="utf-8")
    im.manifest_path(tmp_path).write_text(json.dumps({
        "schema": 1, "venv": str(tmp_path / ".venv"), "lib_dir": str(lib),
        "binaries": ["llama.dll"], "home_cfg": "", "data_dir": "",
        "data_created": False, "shortcut": "",
    }), encoding="utf-8")
    rep = im.uninstall(tmp_path)
    assert rep["ok"]
    assert not (lib / "llama.dll").exists()


# ------------------------------- the rm guard ----------------------------- #

def test_unsafe_data_dir_refuses_dangerous_targets(tmp_path):
    repo = tmp_path
    root_anchor = Path(tmp_path.anchor)
    assert im._unsafe_data_dir("", repo)
    assert im._unsafe_data_dir("relative/dir", repo)
    assert im._unsafe_data_dir(str(root_anchor), repo)
    assert im._unsafe_data_dir(str(Path.home()), repo)
    assert im._unsafe_data_dir(str(repo), repo)
    assert im._unsafe_data_dir(str(repo.parent), repo)
    assert im._unsafe_data_dir(str(repo / "data"), repo) is None


@pytest.mark.skipif(sys.platform == "win32",
                    reason="symlink creation usually needs privilege on Windows")
def test_unsafe_data_dir_refuses_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert im._unsafe_data_dir(str(link), tmp_path)


def test_unsafe_data_dir_refuses_a_junction_on_windows(tmp_path):
    if sys.platform != "win32":
        pytest.skip("junctions are Windows-only")
    import _winapi
    real = tmp_path / "real"
    real.mkdir()
    _winapi.CreateJunction(str(real), str(tmp_path / "j"))
    assert im._unsafe_data_dir(str(tmp_path / "j"), tmp_path) == "is a symlink"


def test_unsafe_data_dir_refuses_when_home_cannot_be_resolved(tmp_path, monkeypatch):
    def _broken_home():
        raise RuntimeError("could not determine home directory")
    monkeypatch.setattr(Path, "home", staticmethod(_broken_home))
    assert im._unsafe_data_dir(str(tmp_path / "data"), tmp_path) is not None


# ------------------------------- processes -------------------------------- #

def _abs_path(*parts) -> str:
    return os.path.join("C:\\" if os.name == "nt" else os.sep, *parts)


def test_running_from_follows_children_but_not_reused_pids():
    venv, py = _abs_path("clone", ".venv"), _abs_path("clone", ".python")
    procs = [
        (1, 0, _abs_path("Windows", "explorer.exe"), 5),
        (10, 1, os.path.join(venv, "Scripts", "localm.exe"), 100),
        (11, 10, _abs_path("Users", "u", "uv", "python.exe"), 110),  # child, started later
        (12, 10, _abs_path("Windows", "conhost.exe"), 50),          # older: a reused pid
        (13, 11, None, 120),                                        # grandchild, exe unknown
        (20, 1, os.path.join(py, "python.exe"), 200),               # this process
        (21, 20, os.path.join(venv, "Scripts", "python.exe"), 210), # its child, not an ancestor
        (30, 1, _abs_path("usr", "bin", "python3"), 300,            # started as the venv's
         os.path.join(venv, "bin", "python")),                      # symlinked python
    ]
    got = [pid for pid, _ in im.running_from([venv, py], procs=procs, self_pid=20)]
    assert got[0] == 10
    assert set(got) == {10, 11, 13, 21, 30}


def test_running_from_excludes_this_process_and_its_ancestors():
    procs = [(1, 0, os.path.join(os.sep, "c", ".venv", "x"), 1),
             (2, 1, os.path.join(os.sep, "c", ".python", "p"), 2)]
    got = im.running_from([os.path.join(os.sep, "c")], procs=procs, self_pid=2)
    assert got == []


def _spawn_from(clone: Path) -> subprocess.Popen:
    """A real process whose program lives in clone/.venv."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(clone / ".venv")],
                   check=True, capture_output=True, timeout=180)
    py = clone / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / \
        ("python.exe" if sys.platform == "win32" else "python")
    (clone / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
    return subprocess.Popen([str(py), "-c", "import time; time.sleep(120)"])


def test_a_running_localm_blocks_uninstall_until_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(im, "list_processes", REAL_LIST_PROCESSES)
    clone = _clone(tmp_path / "clone")
    proc = _spawn_from(clone)
    try:
        deadline = time.monotonic() + 30
        found = []
        while time.monotonic() < deadline and proc.pid not in found:
            found = [pid for pid, _ in im.running_from([clone / ".venv"]) or []]
            time.sleep(0.2)
        assert proc.pid in found, found

        dry = im.uninstall(clone, dry_run=True)
        assert dry["running"], dry
        assert "will be stopped first" in "\n".join(im.format_report(dry))

        blocked = im.uninstall(clone)
        assert blocked["exit"] == im.EXIT_RUNNING
        assert (clone / ".venv").is_dir()

        done = im.uninstall(clone, stop_running=True, defer_runtime=True)
        assert done["exit"] == im.EXIT_OK, done
        assert done["stopped"]
        proc.wait(timeout=20)
        assert proc.poll() is not None
        removed, left = im.finish_pending(clone)
        assert not left and not (clone / ".venv").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=20)


# ------------------------------------ CLI --------------------------------- #

def test_cli_dry_run_prints_sections_and_exit_codes(tmp_path, capsys):
    _fake_install(tmp_path)
    assert im.main(["uninstall", "--root", str(tmp_path), "--dry-run"]) == im.EXIT_OK
    out = capsys.readouterr().out
    assert "Will be removed:" in out and "KEPT:" in out
    assert im.main(["uninstall", "--root", str(tmp_path), "--dry-run", "--purge-data"]) in (
        im.EXIT_OK, im.EXIT_PARTIAL)
    out = capsys.readouterr().out
    assert "In " in out or "WILL BE DELETED" in out


def test_cli_prepare_and_finish(tmp_path, capsys):
    assert im.main(["prepare-data", "--root", str(tmp_path), "--portable"]) == 0
    assert "Data directory:" in capsys.readouterr().out
    assert im.main(["finish", "--root", str(tmp_path)]) == 0


# ------------------------------ the data layout --------------------------- #

def _is_data_dir_expr(node) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "HOME_DIR"
    if isinstance(node, ast.Attribute):
        return node.attr == "HOME_DIR"
    if isinstance(node, ast.Call):
        f = node.func
        return (isinstance(f, ast.Name) and f.id == "home_dir") or \
            (isinstance(f, ast.Attribute) and f.attr == "home_dir")
    return False


def _data_dir_children() -> dict:
    """Every ``<data dir> / <name>`` in the package, where the data dir is
    ``HOME_DIR``, ``config.HOME_DIR``, ``home_dir()`` or ``config.home_dir()``
    and the name is a string literal, a module-level string constant, or an
    f-string (its placeholders become ``*``). Paths built any other way (a
    function parameter such as ``tls_dir(home)``) are not seen."""
    found = {}
    for f in LOCALM.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and _is_data_dir_expr(node.left)):
                continue
            r = node.right
            if isinstance(r, ast.Constant) and isinstance(r.value, str):
                name = r.value
            elif isinstance(r, ast.Name) and r.id in consts:
                name = consts[r.id]
            elif isinstance(r, ast.JoinedStr):
                name = "".join(v.value if isinstance(v, ast.Constant) else "*"
                               for v in r.values)
            else:
                continue
            found.setdefault(name.split("/")[0], f"{f.relative_to(LOCALM.parent)}:{node.lineno}")
    return found


def test_data_entries_cover_every_data_dir_child():
    found = _data_dir_children()
    assert len(found) >= 20, found                       # the scan itself works
    missing = {n: where for n, where in found.items() if not im.is_data_entry(n)}
    assert not missing, (
        "LocaLM writes these into its data folder but install_manifest does not "
        f"list them, so 'delete saved data' would leave them behind: {missing}")
