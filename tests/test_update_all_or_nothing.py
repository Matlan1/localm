# SPDX-License-Identifier: AGPL-3.0-or-later
"""An update and its rollback move every part of the install together: the code,
the venv (every dependency and localm's own install record), the native runtime
and the components. A failure at any step after the swap, an update rolled back
later, and a rollback during the server's restart all put every part back. The
process that applied the update keeps importing its own code, and the update
steps of a restart are not held up by this install's own PyTorch probe."""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _apply_update as au
from localm import _version, updater, updater_components as uc, updater_runtime, uv_tool
from localm.bugreport import LocalmError
from tests._net_config import set_config
from tests._update_signing import envelope, keyset

_PRIV = Ed25519PrivateKey.generate()
_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]


def _pyproject(deps):
    return ('[project]\nname = "localm"\ndependencies = ' + json.dumps(deps) + '\n'
            '[project.optional-dependencies]\nhf = []\n')


def _sd_pins(tag):
    from localm.media.sdcpp.runtime import platform_key
    plat = platform_key() or "windows"
    return (f'REPO = "leejet/stable-diffusion.cpp"\nTAG = "{tag}"\nCOMMIT = "c"\n'
            f'_BASE_URL = f"https://x/{{TAG}}/"\n'
            f'ASSETS: dict = {{("{plat}", "vulkan"): ("sd.zip", "{"0" * 64}")}}\n')


def _files(version, deps, sd=None):
    files = {"VERSION": version, "localm/__init__.py": f"# {version}",
             "localm/setup_llama/pins.py": '_ROCM_TAG = "b1355"\n_PINNED_TAG = "b11928"\n',
             "pyproject.toml": _pyproject(deps)}
    if sd:
        files["localm/media/sdcpp/pins.py"] = _sd_pins(sd)
    return files


def _write_tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def _release(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    data = buf.getvalue()
    return data, envelope(data, [_PRIV])


def _download(data):
    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener


def _dist(site: Path, name: str, version: str, files: dict) -> None:
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n",
                                   encoding="utf-8")
    rows = []
    for rel, data in files.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append(f"{rel},,")
    rows += [f"{info.name}/METADATA,,", f"{info.name}/RECORD,,"]
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _uninstall(site: Path, name: str) -> None:
    import shutil
    for info in site.glob(f"{name}-*.dist-info"):
        for row in (info / "RECORD").read_text(encoding="utf-8").splitlines():
            p = site / row.split(",")[0]
            if p.is_file():
                p.unlink()
        shutil.rmtree(info)


def _localm_dist(site: Path, version: str, exe: bytes = b"trampoline") -> None:
    _dist(site, "localm", version, {"_editable_impl_localm.pth": f"# {version}".encode(),
                                    "../../Scripts/localm.exe": exe})


def _venv(root: Path) -> Path:
    site = root / ".venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    _localm_dist(site, "0.3.1")
    _dist(site, "click", "8.0", {"click/__init__.py": b"click 8.0"})
    return site


def _fake_uv(tmp_path: Path) -> Path:
    exe = tmp_path / "fake-uv-bin" / uv_tool.UV_NAME
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    exe.chmod(0o755)
    return exe


def _deps_step(site: Path, *, rc=0):
    """A runner standing in for `uv pip install -e .` of the 0.3.2 tree: it
    replaces the install record, upgrades click and adds rich, then exits *rc*."""
    def run(cmd):
        if Path(str(cmd[0])).parent.name != "fake-uv-bin":
            return 0
        _uninstall(site, "localm")
        _localm_dist(site, "0.3.2")
        _uninstall(site, "click")
        _dist(site, "click", "8.1", {"click/__init__.py": b"click 8.1"})
        _dist(site, "rich", "13.0", {"rich/__init__.py": b"rich"})
        return rc
    return run


def _sd_runtime(home: Path, tag: str, *, failed=False) -> Path:
    d = home / "runtimes" / "sdcpp" / f"{tag}-vulkan"
    d.mkdir(parents=True, exist_ok=True)
    name = "localm-sdcpp-failed.json" if failed else "localm-sdcpp.json"
    (d / name).write_text(json.dumps({"tag": tag, "backend": "vulkan"}), encoding="utf-8")
    return d


def _children(home: Path, site: Path, *, sd_rc=0, deps_rc=0):
    deps = _deps_step(site, rc=deps_rc)

    def run(cmd):
        if "setup-sdcpp" in cmd:
            _sd_runtime(home, "master-2000-aaaaaaa", failed=bool(sd_rc))
            return sd_rc
        return deps(cmd)
    return run


@pytest.fixture
def home(tmp_path, monkeypatch):
    conf = {"bugreport_upload_url": "https://w"}
    set_config(monkeypatch, lambda: conf)
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("localm.config.home_dir", lambda: h)
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [_PRIV], _RECOVERY))
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.1")
    monkeypatch.setattr(uv_tool, "find_uv", lambda root: None)
    return h


@pytest.fixture
def install(tmp_path, home, monkeypatch):
    """An install at 0.3.1 whose venv holds localm 0.3.1 and click 8.0, with
    sd.cpp master-1999 installed."""
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", ["click"], sd="master-1999-bbbbbbb"))
    site = _venv(inst)
    _sd_runtime(home, "master-1999-bbbbbbb")
    monkeypatch.setattr(updater, "_uv_for_deps", lambda t: str(_fake_uv(tmp_path)))
    return inst, site


def _release_032(sd="master-2000-aaaaaaa"):
    return _release(_files("0.3.2", ["click", "rich"], sd=sd))


def _state(site: Path) -> dict:
    return {n: d["version"] for n, d in uc.dists(site).items()}


def _assert_venv_is_031(site: Path) -> None:
    assert _state(site) == {"localm": "0.3.1", "click": "8.0"}
    assert (site / "_editable_impl_localm.pth").read_bytes() == b"# 0.3.1"
    assert (site / "click" / "__init__.py").read_bytes() == b"click 8.0"
    assert not (site / "rich").exists()


# --------------------------- failures inside apply ---------------------------

def test_a_failed_deps_step_puts_the_venv_and_install_record_back(install):
    inst, site = install
    data, sig = _release_032(sd=None)
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_deps_step(site, rc=1))
    assert _state(site) == {"localm": "0.3.1", "click": "8.0"}, "venv rolled back"
    _assert_venv_is_031(site)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.1"
    assert ei.value.summary == "the post-update step failed; rolled back"


def test_a_failed_component_after_the_deps_step_puts_the_venv_back(install, home):
    inst, site = install
    data, sig = _release_032()
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_children(home, site, sd_rc=1))
    _assert_venv_is_031(site)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.1"
    assert "a component could not be updated, so the update was rolled back" == ei.value.summary


def test_a_failed_runtime_after_the_deps_step_puts_the_venv_back(install, home, monkeypatch):
    inst, site = install
    data, sig = _release_032(sd=None)
    monkeypatch.setattr(updater_runtime, "plan", lambda staged, target: {
        "reprovision": True, "backend": "vulkan", "current": "b1", "target": "b2",
        "source": "release", "pins": {}, "note": "re-provisions the runtime"})
    monkeypatch.setattr(updater_runtime, "stop_own_holders", lambda *a, **k: [])
    monkeypatch.setattr(updater_runtime, "finish_pending",
                        lambda **kw: {"state": "failed", "note": "setup-llama exited 1"})
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_deps_step(site))
    _assert_venv_is_031(site)
    assert "native runtime could not be updated" in ei.value.summary


# ------------------------------- rollback_last -------------------------------

def test_rollback_last_puts_the_venv_back_and_records_the_rollback(install, home):
    inst, site = install
    data, sig = _release_032()
    res = updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                        runner=_children(home, site))
    assert res["components"]["state"] == "updated"
    assert _state(site) == {"localm": "0.3.2", "click": "8.1", "rich": "13.0"}
    updater.rollback_last(installed=inst)
    _assert_venv_is_031(site)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.1"
    assert uc.last_status()["state"] == "rolled_back"
    assert updater_runtime.last_status()["state"] == "rolled_back"


def test_the_restart_rollback_puts_every_part_back(install, home, monkeypatch):
    inst, site = install
    data, sig = _release_032()
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site), defer_runtime=True)
    assert _state(site)["localm"] == "0.3.2"
    import localm.inference.http_server as hs
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    run = _children(home, site, sd_rc=1)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, run(cmd)))
    hs._finish_pending_runtime_update()
    _assert_venv_is_031(site)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.1"
    assert uc.read_pending() is None
    status = uc.last_status()
    assert status["state"] == "rolled_back"
    assert "setup-sdcpp exited 1" in status["note"], "the reason stays named"


def test_a_failure_inside_apply_keeps_the_failed_status(install, home):
    inst, site = install
    data, sig = _release_032()
    with pytest.raises(LocalmError):
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_children(home, site, sd_rc=1))
    assert uc.last_status()["state"] == "failed"


def test_a_rollback_undoes_the_pytorch_step_and_the_deps_step_in_order(install, tmp_path):
    inst, site = install
    snap = tmp_path / "snap"
    _dist(site, "transformers", "5.15", {"transformers/__init__.py": b"t 5.15"})
    uc.snapshot([uc.DEPS], snap, root=inst)
    _deps_step(site)([str(_fake_uv(tmp_path))])
    uc.record_changes(snap, uc.DEPS, inst)
    uc.snapshot(["pytorch"], snap, root=inst, add=True)
    _uninstall(site, "transformers")
    _dist(site, "transformers", "5.20", {"transformers/__init__.py": b"t 5.20"})
    assert uc.restore(snap, root=inst) == []
    assert _state(site) == {"localm": "0.3.1", "click": "8.0", "transformers": "5.15"}
    assert (site / "transformers" / "__init__.py").read_bytes() == b"t 5.15"


def test_a_deps_snapshot_does_not_share_its_copy_with_the_pytorch_one(install):
    inst, site = install
    snap = uc.snapshot_dir(updater._backup_dir(create=True))
    uc.snapshot([uc.DEPS], snap, root=inst)
    _uninstall(site, "click")
    _dist(site, "click", "8.1", {"click/__init__.py": b"click 8.1"})
    uc.snapshot(["pytorch"], snap, root=inst, add=True)
    state = json.loads((snap / uc.STATE_FILE).read_text(encoding="utf-8"))["components"]
    assert state[uc.DEPS]["venv_copy"] != state["pytorch"]["venv_copy"]
    uc.record_changes(snap, uc.DEPS, inst)
    assert uc.installed_names(uc.snapshot_names(snap), inst) == ["pytorch", uc.DEPS]
    assert uc.restore(snap, root=inst) == []
    assert _state(site)["click"] == "8.0", "the older (pre-deps) copy wins"
    uc.discard(snap)
    assert not Path(state[uc.DEPS]["venv_copy"]).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="an open file blocks a move on Windows only")
def test_a_file_in_use_refuses_the_venv_restore_with_nothing_changed(install, tmp_path):
    inst, site = install
    snap = tmp_path / "snap"
    uc.snapshot([uc.DEPS], snap, root=inst)
    _deps_step(site)([str(_fake_uv(tmp_path))])
    before = sorted(str(f.relative_to(inst)) for f in (inst / ".venv").rglob("*"))
    with open(site / "rich" / "__init__.py", "rb"):
        errors = uc.restore(snap, root=inst)
    assert errors and "in use" in errors[0] and "nothing was changed" in errors[0]
    assert _state(site) == {"localm": "0.3.2", "click": "8.1", "rich": "13.0"}
    assert sorted(str(f.relative_to(inst)) for f in (inst / ".venv").rglob("*")) == before


@pytest.mark.skipif(sys.platform != "win32", reason="an open file blocks a move on Windows only")
def test_a_file_both_versions_hold_unchanged_is_left_in_place(install, tmp_path):
    inst, site = install
    snap = tmp_path / "snap"
    uc.snapshot([uc.DEPS], snap, root=inst)
    _deps_step(site)([str(_fake_uv(tmp_path))])
    with open(inst / ".venv" / "Scripts" / "localm.exe", "rb"):
        assert uc.restore(snap, root=inst) == []
    _assert_venv_is_031(site)


@pytest.mark.skipif(sys.platform != "win32", reason="a loaded module cannot be deleted on Windows")
def test_a_module_this_process_loaded_is_replaced(install, tmp_path):
    import ctypes
    import shutil
    inst, site = install
    snap = tmp_path / "snap"
    uc.snapshot([uc.DEPS], snap, root=inst)
    _deps_step(site)([str(_fake_uv(tmp_path))])
    dll = site / "rich" / "_native.dll"
    shutil.copy2(Path(sys.base_prefix) / "DLLs" / "sqlite3.dll", dll)
    record = next(site.glob("rich-*.dist-info")) / "RECORD"
    record.write_text(record.read_text(encoding="utf-8") + "rich/_native.dll,,\n",
                      encoding="utf-8")
    ctypes.WinDLL(str(dll))
    with pytest.raises(OSError):
        dll.unlink()
    assert uc.restore(snap, root=inst) == []
    _assert_venv_is_031(site)
    assert not dll.exists()


def test_a_file_two_changed_distributions_share_is_restored(install, tmp_path):
    inst, site = install
    for name in ("alpha", "beta"):
        _dist(site, name, "1.0", {f"{name}/__init__.py": b"1", "shared/x.txt": b"same"})
    snap = tmp_path / "snap"
    uc.snapshot([uc.DEPS], snap, root=inst)
    for name in ("alpha", "beta"):
        _uninstall(site, name)
    for name in ("alpha", "beta"):
        _dist(site, name, "2.0", {f"{name}/__init__.py": b"2", "shared/x.txt": b"same"})
    assert uc.restore(snap, root=inst) == []
    assert _state(site)["alpha"] == _state(site)["beta"] == "1.0"
    assert (site / "shared" / "x.txt").read_bytes() == b"same"


def test_a_rollback_keeps_packages_installed_after_the_update(install, home):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    state = json.loads((uc.snapshot_dir(updater._backup_dir()) / uc.STATE_FILE)
                       .read_text(encoding="utf-8"))
    assert sorted(state["components"][uc.DEPS]["dists"]) == ["click", "localm", "rich"]
    _dist(site, "plugin-dep", "1.0", {"plugin_dep/__init__.py": b"p"})
    updater.rollback_last(installed=inst)
    assert _state(site) == {"localm": "0.3.1", "click": "8.0", "plugin-dep": "1.0"}


def test_a_backup_an_older_build_replaced_does_not_restore_what_was_kept(install, home):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    backup = updater._backup_dir()
    (backup / "VERSION").write_text("0.1.5", encoding="utf-8")
    updater.rollback_last(installed=inst)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.1.5", "the code is restored"
    assert _state(site)["localm"] == "0.3.2", "the venv copy belongs to another backup"


def test_the_standalone_rollback_skips_a_venv_kept_for_another_backup(install, home, monkeypatch):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    (updater._backup_dir() / "VERSION").write_text("0.1.5", encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(home))
    rb = _load_script("rollback_update")
    assert rb.main(["--yes"], install_root=inst, code_root=Path(__file__).resolve().parents[1]) == 0
    assert _state(site)["localm"] == "0.3.2"


def test_a_corrupt_snapshot_refuses_the_rollback_with_nothing_changed(install, home):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    (uc.snapshot_dir(updater._backup_dir()) / uc.STATE_FILE).write_text("{not json",
                                                                         encoding="utf-8")
    with pytest.raises(LocalmError) as ei:
        updater.rollback_last(installed=inst)
    assert "cannot be read, so nothing was rolled back" in ei.value.summary
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.2"
    assert _state(site)["localm"] == "0.3.2"


def test_an_update_that_keeps_the_dependencies_leaves_the_parts_consistent(
        install, home, monkeypatch):
    from localm import updater_parts
    inst, site = install
    data, sig = _release(_files("0.3.2", ["click"], sd="master-1999-bbbbbbb"))
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=lambda c: pytest.fail(f"no step runs: {c}"))
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    parts = updater_parts.installed_parts(inst)
    assert parts["install_record"] == "0.3.1" and parts["install_record_matches"] is False
    assert not [p for p in parts["problems"] if "record" in p or "dependencies" in p]
    updater.rollback_last(installed=inst)
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.1")
    assert not [p for p in updater_parts.installed_parts(inst)["problems"] if "record" in p]


def test_a_kept_record_survives_a_later_update_that_is_rolled_back(install, home, monkeypatch):
    from localm import updater_parts
    inst, site = install
    updater_parts.note_kept_record(inst, "0.3.1", "0.3.2")
    assert updater_parts._kept_records() == {"0.3.2": "0.3.1"}
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    data, sig = _release(_files("0.3.3", ["click", "rich"]))
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    updater.rollback_last(installed=inst)
    assert _state(site)["localm"] == "0.3.1"
    assert not [p for p in updater_parts.installed_parts(inst)["problems"] if "record" in p]


def test_a_real_mismatch_is_not_hidden_by_a_later_update_that_keeps_the_dependencies(install):
    from localm import updater_parts
    inst, site = install
    updater_parts.note_kept_record(inst, "0.1.5", "0.3.2")
    assert updater_parts._kept_records() == {}, "the record was already not 0.1.5's"


def test_a_stale_snapshot_is_replaced_by_the_component_step(install, home, monkeypatch):
    inst, site = install
    snap = uc.snapshot_dir(updater._backup_dir(create=True))
    updater._backup_dir().mkdir(parents=True, exist_ok=True)
    (updater._backup_dir() / "VERSION").write_text("0.1.5", encoding="utf-8")
    uc.snapshot([uc.DEPS], snap, root=inst, for_backup="0.3.2")
    assert not uc.snapshot_is_for(snap, "0.1.5") and uc.snapshot_is_for(snap, "0.3.2")
    assert uc.snapshot_is_for(snap, None)
    (inst / "VERSION").write_text("0.3.1", encoding="utf-8")
    uc.write_pending([{"name": "sdcpp", "update": True}], root=inst, version="0.3.1")
    uc._write_status({"state": "x"})
    monkeypatch.setattr(uc, "plan", lambda root, inst_, home=None, deep=True: [
        {"name": "sdcpp", "label": "sd", "update": True, "current": "a", "target": "b",
         "items": [], "note": None, "kept": False, "installed": True}])
    monkeypatch.setattr(uc, "_commands", lambda p, root: [])
    monkeypatch.setattr(uc, "_verify", lambda p, root, home_: None)
    assert uc.finish_pending(root=inst, runner=lambda c: 0)["state"] == "updated"
    state = json.loads((snap / uc.STATE_FILE).read_text(encoding="utf-8"))
    assert state["backup_version"] == "0.1.5" and list(state["components"]) == ["sdcpp"]


def test_a_rollback_keeps_later_packages_with_a_pytorch_entry_too(install, tmp_path, monkeypatch):
    inst, site = install
    _dist(site, "transformers", "5.15", {"transformers/__init__.py": b"t"})
    snap = tmp_path / "snap"
    uc.snapshot(["pytorch"], snap, root=inst)
    _uninstall(site, "transformers")
    _dist(site, "transformers", "5.20", {"transformers/__init__.py": b"t2"})
    uc.record_changes(snap, "pytorch", inst)
    _dist(site, "plugin-dep", "1.0", {"plugin_dep/__init__.py": b"p"})
    assert uc.restore(snap, root=inst) == []
    assert _state(site)["transformers"] == "5.15" and _state(site)["plugin-dep"] == "1.0"


def test_a_pytorch_copy_that_matches_needs_no_holder_stopped(install, tmp_path, monkeypatch):
    inst, site = install
    snap = tmp_path / "snap"
    uc.snapshot(["pytorch"], snap, root=inst)
    monkeypatch.setattr(uc, "_stop_torch_holders",
                        lambda *a: pytest.fail("nothing differs, nothing to stop"))
    assert uc.restore(snap, root=inst) == []


def test_a_folder_with_only_bytecode_left_is_removed(install, tmp_path):
    inst, site = install
    snap = tmp_path / "snap"
    uc.snapshot([uc.DEPS], snap, root=inst)
    _deps_step(site)([str(_fake_uv(tmp_path))])
    (site / "rich" / "__pycache__").mkdir()
    (site / "rich" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"x")
    assert uc.restore(snap, root=inst) == []
    assert not (site / "rich").exists()


def test_a_backup_version_that_is_not_text_counts_as_unknown(install, home):
    (updater._backup_dir(create=True)).mkdir(parents=True, exist_ok=True)
    (updater._backup_dir() / "VERSION").write_bytes(b"\xff\xfe")
    assert uc.backup_version(updater._backup_dir()) is None
    assert updater.rollback_info()["version"] is None


def test_an_old_failure_is_not_named_as_the_reason(install, home):
    inst, site = install
    data, sig = _release_032(sd=None)
    uc._write_status({"state": "failed", "note": "an old failure", "components": []})
    status = json.loads((updater._updates_dir() / uc.STATUS_FILE).read_text(encoding="utf-8"))
    status["at"] = 1.0
    (updater._updates_dir() / uc.STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    updater.rollback_last(installed=inst)
    assert "an old failure" not in uc.last_status()["note"]


def test_a_runtime_that_was_undone_does_not_stay_reported_as_updated(install, home, monkeypatch):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater_runtime._write_status({"state": "updated", "note": "re-provisioned a -> b"})
    with pytest.raises(LocalmError):
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_deps_step(site, rc=1))
    assert updater_runtime.last_status()["state"] == "rolled_back"


@pytest.mark.skipif(sys.platform != "win32", reason="an open file blocks a move on Windows only")
def test_a_rollback_refused_by_a_held_file_changes_nothing(install, home):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    with open(site / "rich" / "__init__.py", "rb"):
        with pytest.raises(LocalmError) as ei:
            updater.rollback_last(installed=inst)
    assert "nothing was rolled back" in ei.value.summary
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.2"
    assert _state(site) == {"localm": "0.3.2", "click": "8.1", "rich": "13.0"}


def test_the_update_routes_really_hold_the_probes_while_they_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from localm import discover
    from localm.inference.http_server import create_app
    home = tmp_path / "h"
    (home / "updates" / "backup").mkdir(parents=True)
    (home / "updates" / "backup" / "VERSION").write_text("0.1.0", encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_request_restart", lambda **kw: None)
    seen = []
    monkeypatch.setattr(updater, "rollback_last",
                        lambda: seen.append(discover._torch_probe_holds) or {"rolled_back": True})
    app = create_app(None)
    with TestClient(app) as c:
        r = c.post("/api/update/rollback",
                   headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    assert seen == [1] and discover._torch_probe_holds == 0


def test_the_update_routes_hold_the_torch_probes(monkeypatch):
    import ast
    src = (Path(__file__).resolve().parents[1] / "localm" / "inference" / "routes"
           / "admin.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "to_thread"]
    held = {ast.unparse(c.args[1]) for c in calls
            if ast.unparse(c.args[0]) == "_probes_held"}
    assert held == {"updater.apply", "updater_parts.repair", "updater.rollback_last"}
    assert not [c for c in calls if ast.unparse(c.args[0]) in (
        "updater.apply", "updater_parts.repair", "updater.rollback_last")]


def test_a_venv_without_localm_is_a_problem_and_repair_reinstalls(install):
    from localm import updater_parts
    inst, site = install
    _uninstall(site, "localm")
    parts = updater_parts.installed_parts(inst)
    assert parts["install_record_matches"] is False
    assert any("no install record of localm" in p for p in parts["problems"])
    ran = []
    updater_parts.repair(installed=inst, runner=lambda c: ran.append(c) or _localm_dist(
        site, "0.3.1") or 0)
    assert ran and updater_parts.installed_parts(inst)["install_record_matches"] is True


# ----------------------- the running process's own code ----------------------

def test_pinned_imports_load_the_running_code_not_the_swapped_in_files(tmp_path):
    import localm
    import localm.plugins
    live = Path(list(localm.__path__)[0])
    code = tmp_path / "code" / "localm"
    (code / "plugins").mkdir(parents=True)
    (code / "_pin_probe_top.py").write_text("WHO = 'running'\n", encoding="utf-8")
    (code / "plugins" / "_pin_probe_sub.py").write_text("WHO = 'running'\n", encoding="utf-8")
    unpin = au.pin_imports(code, live)
    try:
        import localm._pin_probe_top as top
        import localm.plugins._pin_probe_sub as sub
        assert top.WHO == sub.WHO == "running"
        assert Path(sub.__file__).parent == code / "plugins"
    finally:
        unpin()
        sys.modules.pop("localm._pin_probe_top", None)
        sys.modules.pop("localm.plugins._pin_probe_sub", None)
    assert Path(list(localm.plugins.__path__)[0]) == live / "plugins"


def test_apply_pins_imports_to_the_running_code(install, monkeypatch):
    inst, site = install
    calls, undone = [], []
    monkeypatch.setattr(au, "pin_imports",
                        lambda code, live: calls.append((code, live)) or (lambda: undone.append(1)))
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    assert calls == [(updater._backup_dir() / "localm", inst / "localm")]
    assert undone == [], "a successful update stays pinned until the restart"
    data2, sig2 = _release(_files("0.3.3", ["click", "attrs"]))
    with pytest.raises(LocalmError):
        updater.apply(1, signature=sig2, installed=inst, download_opener=_download(data2),
                      runner=_deps_step(site, rc=1))
    assert len(calls) == 2 and undone == [1], "a rolled-back update unpins"


def test_keep_running_code_copies_and_pins_only_when_this_process_runs_from_there(
        tmp_path, monkeypatch):
    inst = _write_tree(tmp_path / "inst", {"localm/__init__.py": "# x", "localm/a/b.py": "B"})
    pinned = []
    monkeypatch.setattr(au, "pin_imports", lambda code, live: pinned.append((code, live))
                        or (lambda: None))
    monkeypatch.setattr(au, "running_code_dir", lambda: tmp_path / "elsewhere" / "localm")
    au.keep_running_code(inst, tmp_path / "keep")
    assert pinned == [] and not (tmp_path / "keep").exists()
    monkeypatch.setattr(au, "running_code_dir", lambda: inst / "localm")
    au.keep_running_code(inst, tmp_path / "keep")
    assert (tmp_path / "keep" / "localm" / "a" / "b.py").read_text(encoding="utf-8") == "B"
    assert pinned == [(tmp_path / "keep" / "localm", inst / "localm")]


# --------------------------- holders during the restart -----------------------

def test_an_unnamed_holder_is_waited_out_before_it_refuses(tmp_path, monkeypatch):
    seen = [[{"pid": None, "name": "an unidentified process (x.dll)"}]] * 2 + [[]]
    monkeypatch.setattr(updater_runtime, "holders", lambda lib: seen.pop(0) if seen else [])
    assert updater_runtime.stop_own_holders(tmp_path, tmp_path, wait=5.0) == []


def test_an_unnamed_holder_that_stays_is_reported(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(updater_runtime, "holders", lambda lib: calls.append(1) or [
        {"pid": None, "name": "an unidentified process (x.dll)"}])
    left = updater_runtime.stop_own_holders(tmp_path, tmp_path, wait=0.5)
    assert left and left[0]["pid"] is None
    assert len(calls) > 1, "it looked again before refusing"


class _Proc:
    def __init__(self, polls):
        self.polls = list(polls)

    def poll(self):
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]


def test_held_torch_probes_spawn_nothing_and_wait_for_running_ones(monkeypatch):
    from localm import discover
    spawned = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or pytest.fail("spawned"))
    running = _Proc([None, None, 0])
    monkeypatch.setattr(discover, "_torch_probe_procs", {running})
    for latch in ("_isolated_torch_unavailable", "_isolated_torch_broken_warned",
                  "_isolated_torch_unanswered"):
        monkeypatch.setattr(discover, latch, False)
    monkeypatch.setattr(discover, "_isolated_torch_retry_at", 0.0)
    monkeypatch.setattr(discover, "_torch_gpus_resident",
                        lambda: pytest.fail("torch imported in this process"))
    with discover.torch_probes_held(wait=5.0):
        assert running.poll() == 0, "entry waited for the running probe to exit"
        assert discover._torch_gpus_isolated() is None
        assert discover._torch_gpus_isolated_once() == []
    assert spawned == []
    assert discover._torch_probe_holds == 0
    assert discover._isolated_torch_unanswered is True
    assert discover._isolated_torch_broken_warned is False, "no false 'please report' warning"


def test_the_restart_holds_torch_probes_during_the_update_steps(monkeypatch):
    import localm.inference.http_server as hs
    from localm import discover
    seen = []
    monkeypatch.setattr(hs, "_finish_pending_update_steps",
                        lambda: seen.append(discover._torch_probe_holds))
    hs._finish_pending_runtime_update()
    assert seen == [1] and discover._torch_probe_holds == 0


# ------------------------------- upgrade again -------------------------------

def test_a_restart_by_an_older_build_leaves_the_steps_to_one_restart(monkeypatch):
    from localm.cli import _core
    ran = []
    monkeypatch.setattr(_core, "_finish_pending_runtime_update", lambda c: ran.append(c))
    monkeypatch.setattr(_core, "_finish_pending_component_update", lambda c: ran.append(c))
    monkeypatch.setattr(updater_runtime, "read_pending", lambda: None)
    monkeypatch.setattr(uc, "read_pending", lambda: None)
    monkeypatch.setattr(updater_runtime, "plan", lambda s, t: {"reprovision": True})
    monkeypatch.setattr(uc, "plan", lambda s, t, deep=True: [])
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.delenv("LOCALM_UPDATE_STEPS_DONE", raising=False)
    monkeypatch.setenv("LOCALM_FINISH_UPDATE_AFTER_START", "")
    _core._update_steps_at_start("gui")
    assert ran == [] and os.environ["LOCALM_FINISH_UPDATE_AFTER_START"] == "1"
    monkeypatch.setattr(updater_runtime, "plan", lambda s, t: {"reprovision": False})
    monkeypatch.setenv("LOCALM_FINISH_UPDATE_AFTER_START", "")
    _core._update_steps_at_start("gui")
    assert os.environ["LOCALM_FINISH_UPDATE_AFTER_START"] == "", "nothing to install: no restart"
    monkeypatch.setenv("LOCALM_UPDATE_STEPS_DONE", "1")
    _core._update_steps_at_start("gui")
    assert ran == [] and "LOCALM_UPDATE_STEPS_DONE" not in os.environ
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS")
    _core._update_steps_at_start("gui")
    assert ran == ["gui", "gui"], "a fresh start runs the steps itself"


def test_the_server_restarts_once_for_the_deferred_steps(monkeypatch):
    import localm.inference.http_server as hs
    asked = []
    monkeypatch.setattr(hs, "_request_restart", lambda **kw: asked.append(kw))

    class _App:
        class state:
            instance_port, instance_id = 8123, "abc"
    monkeypatch.setenv("LOCALM_FINISH_UPDATE_AFTER_START", "1")
    hs._restart_for_pending_update_steps(_App)
    assert asked == [{"delay": hs.UPDATE_STEPS_RESTART_DELAY_S, "port": 8123,
                      "instance_id": "abc"}]
    assert "LOCALM_FINISH_UPDATE_AFTER_START" not in os.environ
    hs._restart_for_pending_update_steps(_App)
    assert len(asked) == 1


def test_the_restart_catches_up_a_runtime_nothing_left_pending(install, monkeypatch):
    import localm.inference.http_server as hs
    import localm.inference.backends.llamacpp._loader as loader
    inst, _site = install
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    monkeypatch.setattr(loader, "stop_probe_daemon", lambda *a, **k: None)
    monkeypatch.setattr(updater_runtime, "plan", lambda s, t: {
        "reprovision": True, "backend": "vulkan", "current": "b1", "target": "b2",
        "source": "release", "pins": {}, "note": "re-provisions"})
    done = []
    monkeypatch.setattr(updater_runtime, "finish_pending", lambda **kw: done.append(
        updater_runtime.read_pending()) or {"state": "failed", "note": "offline"})
    monkeypatch.setattr(updater, "rollback_last", lambda **kw: pytest.fail("rolled back"))
    hs._finish_pending_runtime_update()
    assert done and done[0]["repair"] is True and done[0]["target"] == "b2"


def test_installed_parts_shows_the_last_update_steps(install, monkeypatch):
    from localm import updater_parts
    inst, _site = install
    updater_runtime._write_status({"state": "updated", "note": "re-provisioned the cpu runtime"})
    steps = updater_parts.installed_parts(inst)["last_steps"]
    assert [s["note"] for s in steps] == ["re-provisioned the cpu runtime"]


def test_this_versions_restart_marks_the_update_steps_done_with_its_version(monkeypatch):
    import localm.inference.http_server as hs
    from localm import instances
    monkeypatch.setenv("LOCALM_UPDATE_STEPS_DONE", "")
    monkeypatch.delenv("LOCALM_UPDATE_STEPS_DONE")
    monkeypatch.setattr(instances, "unregister_own", lambda iid: None)
    monkeypatch.setattr(instances, "restore_own", lambda removed: None)
    seen = []

    def execv(exe, argv):
        seen.append(os.environ.get("LOCALM_UPDATE_STEPS_DONE"))
        raise OSError("not in a test")
    monkeypatch.setattr(os, "execv", execv)
    with pytest.raises(OSError):
        hs._execv_restart(None, None, update_steps_done=True)
    with pytest.raises(OSError):
        hs._execv_restart(None, None)
    assert seen == [_version.read_version(), None]
    assert "LOCALM_UPDATE_STEPS_DONE" not in os.environ


def test_a_marker_carried_through_an_older_build_does_not_skip_the_steps(monkeypatch):
    from localm.cli import _core
    deferred = []
    monkeypatch.setattr(_core, "_defer_update_steps_to_a_restart", lambda: deferred.append(1))
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.setenv("LOCALM_UPDATE_STEPS_DONE", "0.1.5")
    _core._update_steps_at_start("gui")
    assert deferred == [1] and "LOCALM_UPDATE_STEPS_DONE" not in os.environ


# ------------------------------- installed parts ------------------------------

def test_installed_parts_names_an_install_record_of_another_version(install, monkeypatch):
    from localm import updater_parts
    inst, site = install
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    parts = updater_parts.installed_parts(inst)
    assert parts["install_record"] == "0.3.1" and parts["install_record_matches"] is True
    _uninstall(site, "localm")
    _localm_dist(site, "0.1.5")
    parts = updater_parts.installed_parts(inst)
    assert parts["consistent"] is False and parts["install_record_matches"] is False
    assert any("install record is localm 0.1.5, but localm 0.3.1" in p for p in parts["problems"])
    _uninstall(site, "click")
    parts = updater_parts.installed_parts(inst)
    assert any("click missing" in p for p in parts["problems"])


def test_installed_parts_names_a_component_at_another_pin(install, home):
    from localm import updater_parts
    inst, _site = install
    import shutil
    shutil.rmtree(home / "runtimes" / "sdcpp" / "master-1999-bbbbbbb-vulkan")
    _sd_runtime(home, "master-1000-0000000")
    (inst / "localm" / "media" / "sdcpp" / "pins.py").write_text(
        _sd_pins("master-1999-bbbbbbb"), encoding="utf-8")
    parts = updater_parts.installed_parts(inst)
    sd = [c for c in parts["components"] if c["name"] == "sdcpp"]
    assert sd and sd[0]["matches"] is False
    assert any("master-1000-0000000, but localm 0.3.1 pins master-1999-bbbbbbb" in p
               for p in parts["problems"])


def test_repair_reinstalls_the_dependencies_of_the_installed_version(install, tmp_path):
    from localm import updater_parts
    inst, site = install
    _uninstall(site, "localm")
    _localm_dist(site, "0.1.5")
    calls = []

    def run(cmd):
        calls.append(cmd)
        _uninstall(site, "localm")
        _localm_dist(site, "0.3.1")
        return 0
    res = updater_parts.repair(installed=inst, runner=run)
    assert calls and Path(str(calls[0][0])).parent.name == "fake-uv-bin"
    assert _state(site)["localm"] == "0.3.1"
    assert res["restart_needed"] is True
    assert updater_parts.installed_parts(inst)["install_record_matches"] is True


def test_a_failed_repair_puts_the_packages_back(install):
    from localm import updater_parts
    inst, site = install
    _uninstall(site, "localm")
    _localm_dist(site, "0.1.5")

    def run(cmd):
        _uninstall(site, "click")
        return 3
    with pytest.raises(LocalmError) as ei:
        updater_parts.repair(installed=inst, runner=run)
    assert "previous ones were put back" in ei.value.summary
    assert _state(site) == {"localm": "0.1.5", "click": "8.0"}


def test_a_repairs_runtime_failure_during_the_restart_rolls_nothing_back(install, monkeypatch):
    import localm.inference.http_server as hs
    inst, _site = install
    updater_runtime.write_pending({"backend": "vulkan", "current": "b1", "target": "b2",
                                   "pins": {}}, root=inst, version="0.3.1", repair=True)
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    monkeypatch.setattr(updater_runtime, "finish_pending",
                        lambda **kw: {"state": "failed", "note": "setup-llama exited 1"})
    monkeypatch.setattr(updater, "rollback_last", lambda **kw: pytest.fail("rolled back"))
    import localm.inference.backends.llamacpp._loader as loader
    monkeypatch.setattr(loader, "stop_probe_daemon", lambda *a, **k: None)
    hs._finish_pending_runtime_update()


# --------------------------- the standalone rollback --------------------------

def _load_script(name):
    import importlib.util
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(f"_t_{name}", repo / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_standalone_rollback_puts_the_venv_back_too(install, home, monkeypatch):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    assert _state(site)["localm"] == "0.3.2"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    repo = Path(__file__).resolve().parents[1]
    rb = _load_script("rollback_update")
    assert rb.main(["--yes"], install_root=inst, code_root=repo) == 0
    _assert_venv_is_031(site)
    assert (inst / "VERSION").read_text(encoding="utf-8") == "0.3.1"


def test_the_standalone_rollback_puts_the_packages_back_when_the_code_fails(
        install, home, monkeypatch, capsys):
    inst, site = install
    data, sig = _release_032(sd=None)
    updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                  runner=_deps_step(site))
    manifest = updater._updates_dir() / "applied_names.json"
    manifest.write_text(json.dumps(json.loads(manifest.read_text(encoding="utf-8"))
                                   + ["../outside"]), encoding="utf-8")
    monkeypatch.setenv("LOCALM_HOME", str(home))
    repo = Path(__file__).resolve().parents[1]
    rb = _load_script("rollback_update")
    assert rb.main(["--yes"], install_root=inst, code_root=repo) == 2
    assert "The packages were put back as they were" in capsys.readouterr().err
    assert _state(site) == {"localm": "0.3.2", "click": "8.1", "rich": "13.0"}


def test_the_watchdog_runs_the_rollback_of_the_code_root(tmp_path, monkeypatch):
    wd = _load_script("update_watchdog")
    code = tmp_path / "code"
    (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "rollback_update.py").write_text(
        "def main(argv, install_root, code_root):\n"
        "    (install_root / 'ran').write_text(str(code_root))\n    return 0\n",
        encoding="utf-8")
    inst = tmp_path / "inst"
    inst.mkdir()
    assert wd._invoke_rollback(inst, None, code) == wd.EXIT_ROLLED_BACK
    assert (inst / "ran").read_text() == str(code)


def test_the_watchdog_of_a_pinned_process_is_the_backups(install, monkeypatch, tmp_path):
    inst, _site = install
    backup = tmp_path / "bk"
    (backup / "scripts").mkdir(parents=True)
    (backup / "scripts" / "update_watchdog.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(au, "running_code_dir", lambda: backup / "localm")
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    seen = []
    assert updater.spawn_health_watchdog(host="127.0.0.1", port=1, scheme="http",
                                         expect_version="0.3.2",
                                         popen=lambda argv, **kw: seen.append(argv))
    argv = seen[0]
    assert argv[1] == str(backup / "scripts" / "update_watchdog.py")
    assert argv[argv.index("--code-root") + 1] == str(backup)


def test_doctor_lists_the_parts_that_differ(monkeypatch):
    import importlib
    from rich.console import Console
    from localm import updater_parts
    doctor = importlib.import_module("localm.cli.doctor")
    monkeypatch.setattr(updater_parts, "installed_parts", lambda: {
        "app": "0.3.2", "install_record": "0.1.5", "install_record_matches": False,
        "install_record_problem": True,
        "deps_problems": [], "runtime": {}, "components": [], "consistent": False,
        "problems": ["the venv's install record is localm 0.1.5, but localm 0.3.2 is installed"]})
    out = io.StringIO()
    monkeypatch.setattr(doctor, "console", Console(file=out, width=300))
    doctor._check_installed_parts()
    text = out.getvalue()
    assert "install record is localm 0.1.5" in text and "localm update --repair" in text
    assert "install record: localm" not in text, "no OK line over the same record"
