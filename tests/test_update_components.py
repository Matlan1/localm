# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applying an update brings every other provisioned component to the release's
pin: the PyTorch stack, the sd.cpp and KoboldCpp runtimes, the managed ComfyUI
(checkout, pinned PyTorch, pinned custom nodes), the uv inside the install, and
the Linux CUDA libraries of the llama.cpp runtime. A component that is not
installed is never installed; one the user chose is kept and named; a failure
puts every component back and rolls the code back; a rollback restores them."""

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
from localm.media.sdcpp.runtime import platform_key
from tests._net_config import set_config
from tests._update_signing import envelope, keyset

_PRIV = Ed25519PrivateKey.generate()
_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]
_PLAT = platform_key() or "windows"

_LLAMA_PINS = '_ROCM_TAG = "b1355"\n_PINNED_TAG = "b11928"\n'


def _sd_pins(tag, backends=("vulkan", "cpu")):
    assets = ", ".join(f'("{_PLAT}", "{b}"): ("sd-{b}.zip", "{"0" * 64}")' for b in backends)
    return (f'REPO = "leejet/stable-diffusion.cpp"\nTAG = "{tag}"\nCOMMIT = "c"\n'
            f'_BASE_URL = f"https://x/{{TAG}}/"\nASSETS: dict = {{{assets}}}\n')


def _kobold_pins(tag, builds=("nocuda",)):
    assets = ", ".join(f'("{_PLAT}", "{b}"): ("k-{b}", 1, "{"0" * 64}")' for b in builds)
    return f'TAG = "{tag}"\nVERSION = "{tag[1:]}"\nASSETS = {{{assets}}}\n'


def _comfy_pins(commit, version):
    return f'COMFYUI_PINNED_COMMIT = "{commit}"\nCOMFYUI_PINNED_VERSION = "{version}"\n'


_PYPROJECT_031 = ('[project]\nname = "localm"\ndependencies = ["click"]\n'
                  '[project.optional-dependencies]\n'
                  'gpu = ["transformers~=5.15.1", "torch==2.11.0+rocm7.13.0"]\n'
                  'hf = ["transformers~=5.15.1"]\naudio = []\n')
_PYPROJECT_032 = _PYPROJECT_031.replace("5.15.1", "5.20.2")


def _files(version, *, sd=None, kobold=None, comfy=None, uv=None, pyproject=None):
    files = {"VERSION": version, "localm/__init__.py": f"# {version}",
             "localm/setup_llama/pins.py": _LLAMA_PINS,
             "localm/setup_llama/cli.py": 'opt = "--keep-choice"\n',
             "localm/cli/comfy.py": 'opt = "--reinstall-requirements"\n',
             "pyproject.toml": pyproject or _PYPROJECT_031}
    if sd:
        files["localm/media/sdcpp/pins.py"] = _sd_pins(sd)
    if kobold:
        files["localm/media/koboldcpp/pins.py"] = _kobold_pins(kobold)
    if comfy:
        files["localm/media/managed_comfy_fresh.py"] = _comfy_pins(*comfy)
    if uv:
        pin = f'set "UV_INSTALLER_VERSION={uv}"\nset "UV_INSTALLER_SHA256={"a" * 64}"\n'
        files["setup.bat"] = pin
        files["setup.sh"] = f'UV_INSTALLER_VERSION="{uv}"\nUV_INSTALLER_SHA256="{"a" * 64}"\n'
    return files


def _write_tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def _release(files):
    data = _zip(files)
    return data, envelope(data, [_PRIV])


def _download(data):
    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener


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


def _sd_runtime(home: Path, tag: str, backend: str, *, failed=False) -> Path:
    d = home / "runtimes" / "sdcpp" / f"{tag}-{backend}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stable-diffusion.dll").write_bytes(b"lib-" + tag.encode())
    name = "localm-sdcpp-failed.json" if failed else "localm-sdcpp.json"
    (d / name).write_text(json.dumps({"tag": tag, "backend": backend,
                                      "devices": [["Vulkan0", "gpu"]]}), encoding="utf-8")
    return d


def _kobold_runtime(home: Path, tag: str, build: str) -> Path:
    d = home / "runtimes" / "koboldcpp" / f"{tag}-{build}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "localm-koboldcpp.json").write_text(json.dumps({"tag": tag, "build": build}),
                                             encoding="utf-8")
    return d


def _fake_children(home: Path, *, rc=0, calls=None, also=None):
    """A runner standing in for the children an update starts: `setup-sdcpp`
    and `setup-music` install their runtime directory at the tag the command's
    tree pins (read from the call's *also* map), the rest exit 0."""
    def run(cmd):
        if calls is not None:
            calls.append(list(cmd))
        if also and also(cmd) is not None:
            return also(cmd)
        if rc:
            if "setup-sdcpp" in cmd:
                b = cmd[cmd.index("--backend") + 1]
                _sd_runtime(home, run.sd_tag, b, failed=True)
            return rc
        if "setup-sdcpp" in cmd:
            _sd_runtime(home, run.sd_tag, cmd[cmd.index("--backend") + 1])
        if "setup-music" in cmd:
            be = cmd[cmd.index("--backend") + 1]
            _kobold_runtime(home, run.kobold_tag, {"cpu": "nocuda"}.get(be, be))
        return 0
    run.sd_tag, run.kobold_tag = "master-1017-be48649", "v1.123.1"
    return run


@pytest.fixture
def install(tmp_path, home):
    """An install at 0.3.1 pinning sd.cpp master-982 and KoboldCpp v1.123."""
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", sd="master-982-0518b34",
                                                 kobold="v1.123"))
    return inst


def _v(root):
    return (root / "VERSION").read_text(encoding="utf-8")


# ---------------------------------- reading pins ------------------------------

def test_staged_pins_read_every_component_as_text(tmp_path):
    root = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649",
                                              kobold="v1.123.1", comfy=("abc", "v0.43.0"),
                                              uv="0.13.6"))
    pins = uc.staged_pins(root)
    assert pins["sdcpp"]["tag"] == "master-1017-be48649"
    assert (_PLAT, "vulkan") in pins["sdcpp"]["assets"]
    assert pins["koboldcpp"]["tag"] == "v1.123.1" and pins["koboldcpp"]["version"] == "1.123.1"
    assert pins["comfyui"] == {"commit": "abc", "version": "v0.43.0"}
    assert pins["uv"] == "0.13.6"


def test_the_shipped_pin_files_parse(tmp_path):
    root = Path(__file__).resolve().parents[1]
    from localm.media import managed_comfy_fresh as f
    from localm.media.koboldcpp import pins as kp
    from localm.media.sdcpp import pins as sp
    pins = uc.staged_pins(root)
    assert pins["sdcpp"] == {"tag": sp.TAG, "assets": set(sp.ASSETS)}
    assert pins["koboldcpp"]["tag"] == kp.TAG and pins["koboldcpp"]["assets"] == set(kp.ASSETS)
    assert pins["comfyui"] == {"commit": f.COMFYUI_PINNED_COMMIT,
                               "version": f.COMFYUI_PINNED_VERSION}
    assert pins["uv"] == uv_tool.read_pin(root)[0]


def test_the_comfy_marker_name_is_the_one_provisioning_writes():
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    from localm.media.sdcpp import runtime as sd
    from localm.media.koboldcpp import runtime as kb
    assert uc._COMFY_MARKER == MARKER_FILENAME
    assert (uc._SDCPP_MARKER, uc._SDCPP_FAILED, uc._KOBOLD_MARKER) == (
        sd.MARKER, sd.FAILED_MARKER, kb.MARKER)


# ---------------------------------- sd.cpp / KoboldCpp ------------------------

def _plans(staged, inst, **kw):
    return {p["name"]: p for p in uc.plan(staged, inst, **kw)}


def test_an_installed_sdcpp_backend_is_installed_at_the_new_tag(tmp_path, install, home):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649"))
    p = _plans(staged, install)["sdcpp"]
    assert (p["update"], p["items"], p["current"], p["target"]) == (
        True, ["vulkan"], "master-982-0518b34", "master-1017-be48649")
    assert "from master-982-0518b34 to master-1017-be48649" in p["note"]


def test_a_component_that_is_not_installed_is_never_installed(tmp_path, install):
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649",
                                                kobold="v1.123.1", comfy=("abc", "v1"),
                                                uv="0.13.6"))
    plans = uc.plan(staged, install)
    assert [p["name"] for p in plans if p["installed"] or p["update"]] == []
    assert not uc.needs_finish(plans)


def test_a_backend_already_at_the_new_tag_needs_nothing(tmp_path, install, home):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    _sd_runtime(home, "master-1017-be48649", "vulkan")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649"))
    assert _plans(staged, install)["sdcpp"]["update"] is False


def test_a_backend_that_failed_at_the_new_tag_or_has_no_build_is_named_not_installed(
        tmp_path, install, home):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    _sd_runtime(home, "master-982-0518b34", "cuda")
    _sd_runtime(home, "master-1017-be48649", "vulkan", failed=True)
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649"))
    p = _plans(staged, install)["sdcpp"]
    assert p["update"] is False
    assert "vulkan: it did not load on this machine before" in p["note"]
    assert "cuda: this release ships no cuda build" in p["note"]


def test_a_marker_that_does_not_match_its_directory_counts_as_not_installed(
        tmp_path, install, home):
    d = _sd_runtime(home, "master-982-0518b34", "vulkan")
    d.rename(d.with_name("copy-of-it"))
    (home / "runtimes" / "sdcpp" / "broken-cpu").mkdir()
    (home / "runtimes" / "sdcpp" / "broken-cpu" / "localm-sdcpp.json").write_text(
        "{not json", encoding="utf-8")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649"))
    assert _plans(staged, install)["sdcpp"]["installed"] is False


def test_the_tag_in_use_is_the_installed_trees_pin_not_an_older_leftover(
        tmp_path, install, home):
    _sd_runtime(home, "master-964-1d0a58a", "cpu")
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    os.utime(home / "runtimes" / "sdcpp" / "master-964-1d0a58a-cpu" / "localm-sdcpp.json",
             (2e9, 2e9))
    staged = _write_tree(tmp_path / "s", _files("0.3.2", sd="master-1017-be48649"))
    assert _plans(staged, install)["sdcpp"]["items"] == ["vulkan"]


def test_kobold_installs_each_installed_build_with_its_setup_backend(tmp_path, install, home):
    _kobold_runtime(home, "v1.123", "nocuda")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", kobold="v1.123.1"))
    p = _plans(staged, install)["koboldcpp"]
    assert p["update"] and p["items"] == ["nocuda"]
    assert uc._commands(p, staged) == [[sys.executable, "-m", "localm", "setup-music",
                                        "--backend", "cpu", "--no-models", "--no-test"]]


# --------------------------------- apply end to end ---------------------------

def test_update_installs_sdcpp_and_kobold_at_the_new_pins(tmp_path, install, home):
    old_sd = _sd_runtime(home, "master-982-0518b34", "vulkan")
    _kobold_runtime(home, "v1.123", "nocuda")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649", kobold="v1.123.1"))
    calls = []
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=_fake_children(home, calls=calls))
    assert [c[3] for c in calls] == ["setup-sdcpp", "setup-music"]
    assert calls[0] == [sys.executable, "-m", "localm", "setup-sdcpp", "--backend", "vulkan"]
    assert res["components"]["state"] == "updated"
    assert "stable-diffusion.cpp image runtime: master-982-0518b34 -> master-1017-be48649" \
        in res["components_note"]
    assert (home / "runtimes" / "sdcpp" / "master-1017-be48649-vulkan").is_dir()
    assert old_sd.is_dir(), "the old runtime directory stays for a rollback"
    assert uc.read_pending() is None
    updater.rollback_last(installed=install)
    assert _v(install) == "0.3.1" and old_sd.is_dir()


def test_a_failed_component_rolls_back_the_code_and_removes_what_it_created(
        tmp_path, install, home):
    old_sd = _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                      runner=_fake_children(home, rc=1))
    assert "rolled back" in ei.value.summary
    assert "setup-sdcpp" in ei.value.reason and "put back" in ei.value.reason
    assert _v(install) == "0.3.1"
    assert old_sd.is_dir()
    assert not (home / "runtimes" / "sdcpp" / "master-1017-be48649-vulkan").exists(), \
        "a failed attempt leaves nothing that would mark the backend as failed here"
    assert uc.last_status()["state"] == "failed"
    assert uc.read_pending() is None


def test_a_child_that_exits_0_but_installs_nothing_counts_as_failed(tmp_path, install, home):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                      runner=lambda c: 0)
    assert "vulkan not installed at master-1017-be48649" in ei.value.reason
    assert _v(install) == "0.3.1"


def test_the_runtime_and_the_components_are_rolled_back_together(tmp_path, home):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", sd="master-982-0518b34") | {
        "localm/setup_llama/pins.py": '_ROCM_TAG = "b1"\n_PINNED_TAG = "b11757"\n'})
    lib = inst / au.RUNTIME_LIB
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_bytes(b"old" * 30000)
    (lib / ".localm-backend").write_text("vulkan b11757", encoding="utf-8")
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    children = _fake_children(home, rc=0)

    def run(cmd):
        if "setup-llama" in cmd:
            (lib / "llama.dll").unlink()
            (lib / "llama.dll").write_bytes(b"new" * 30000)
            (lib / ".localm-backend").write_text("vulkan b11928", encoding="utf-8")
            return 0
        if "setup-sdcpp" in cmd:
            return 1
        return children(cmd)
    with pytest.raises(LocalmError):
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=run)
    assert (lib / ".localm-backend").read_text(encoding="utf-8") == "vulkan b11757"
    assert (lib / "llama.dll").read_bytes().startswith(b"old")
    assert _v(inst) == "0.3.1"


def test_the_server_defers_the_components_to_its_restart(tmp_path, install, home,
                                                         monkeypatch):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    res = updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                        runner=lambda c: pytest.fail(f"nothing runs before the restart: {c}"),
                        defer_runtime=True)
    assert res["components"] is None and res["components_note"].endswith("while localm restarts")
    assert uc.read_pending()["components"] == ["sdcpp"]
    import localm.inference.http_server as hs
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    fake = _fake_children(home)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    hs._finish_pending_runtime_update()
    assert uc.last_status()["state"] == "updated"
    assert (home / "runtimes" / "sdcpp" / "master-1017-be48649-vulkan").is_dir()


def test_a_failure_during_the_restart_rolls_the_update_back(tmp_path, install, home,
                                                            monkeypatch):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  defer_runtime=True)
    import localm.inference.http_server as hs
    monkeypatch.setattr(updater, "repo_root", lambda: install)
    monkeypatch.setattr(updater, "rollback_last",
                        lambda **kw: rolled.append(1) or {"rolled_back": True})
    rolled = []
    fake = _fake_children(home, rc=1)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    hs._finish_pending_runtime_update()
    assert rolled == [1]
    assert uc.last_status()["state"] == "failed"


def test_first_start_catches_up_a_component_an_old_updater_left_behind(tmp_path, home,
                                                                       monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    inst = _write_tree(tmp_path / "inst", _files("0.3.2", sd="master-1017-be48649"))
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    from localm.cli import _core
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
    calls = []
    fake = _fake_children(home, calls=calls)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    _core._finish_pending_component_update("ps")
    assert calls == [] and uc.read_pending() is None, "only commands that use components"
    _core._finish_pending_component_update("gui")
    assert [c[3] for c in calls] == ["setup-sdcpp"]
    calls.clear()
    _core._finish_pending_component_update("gui")
    assert calls == [], "a current component is left alone"


def test_a_pending_update_of_a_rolled_back_version_is_dropped(tmp_path, install, home):
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    data, sig = _release(_files("0.3.2", sd="master-1017-be48649"))
    updater.apply(1, signature=sig, installed=install, download_opener=_download(data),
                  defer_runtime=True)
    updater.rollback_last(installed=install)
    assert uc.read_pending() is None
    uc.write_pending([{"name": "sdcpp", "update": True}], root=install, version="0.3.2")
    assert uc.finish_pending(root=install, runner=lambda c: pytest.fail("no")) is None


# ------------------------------------ uv --------------------------------------

def _contained_uv(root: Path, version: str) -> Path:
    d = root / ".uv"
    d.mkdir(parents=True, exist_ok=True)
    exe = d / uv_tool.UV_NAME
    exe.write_text(version, encoding="utf-8")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def fake_uv_version(monkeypatch):
    monkeypatch.setattr(uv_tool, "uv_version",
                        lambda p: Path(p).read_text(encoding="utf-8").strip())


def _bootstrap(version="0.13.6", fail=False):
    def boot(root, *, emit, dest):
        if fail:
            raise uv_tool.UvBootstrapError("the download failed")
        Path(dest).mkdir(parents=True)
        (Path(dest) / uv_tool.UV_NAME).write_text(version, encoding="utf-8")
        (Path(dest) / uv_tool.UV_NAME).chmod(0o755)
        return Path(dest) / uv_tool.UV_NAME
    return boot


def test_the_uv_inside_the_install_is_updated_and_rolled_back(tmp_path, home,
                                                              fake_uv_version, monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", uv="0.13.4"))
    _contained_uv(inst, "0.13.4")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", uv="0.13.6"))
    p = _plans(staged, inst)["uv"]
    assert (p["update"], p["current"], p["target"]) == (True, "0.13.4", "0.13.6")
    monkeypatch.setattr(uv_tool, "bootstrap_uv", _bootstrap())
    data, sig = _release(_files("0.3.2", uv="0.13.6"))
    res = updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                        runner=lambda c: 0)
    assert res["components"]["state"] == "updated"
    assert (inst / ".uv" / uv_tool.UV_NAME).read_text(encoding="utf-8") == "0.13.6"
    assert not (inst / ".uv-update").exists() and not (inst / ".uv-previous").exists()
    updater.rollback_last(installed=inst)
    assert (inst / ".uv" / uv_tool.UV_NAME).read_text(encoding="utf-8") == "0.13.4"


def test_a_uv_that_cannot_be_installed_leaves_the_old_one_and_the_code(tmp_path, home,
                                                                       fake_uv_version,
                                                                       monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", uv="0.13.4"))
    _contained_uv(inst, "0.13.4")
    monkeypatch.setattr(uv_tool, "bootstrap_uv", _bootstrap(fail=True))
    data, sig = _release(_files("0.3.2", uv="0.13.6"))
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=lambda c: 0)
    assert "the download failed" in ei.value.reason
    assert (inst / ".uv" / uv_tool.UV_NAME).read_text(encoding="utf-8") == "0.13.4"
    assert _v(inst) == "0.3.1"


def test_a_running_uv_refuses_the_update_with_nothing_changed(tmp_path, home,
                                                              fake_uv_version, monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", uv="0.13.4"))
    _contained_uv(inst, "0.13.4")
    monkeypatch.setattr(uc, "_running_in",
                        lambda folder: [(4242, str(folder / "uv.exe"))]
                        if folder.name == ".uv" else [])
    data, sig = _release(_files("0.3.2", uv="0.13.6"))
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=lambda c: pytest.fail(f"nothing runs: {c}"))
    assert "nothing was updated" in ei.value.summary and "pid 4242" in ei.value.reason
    assert _v(inst) == "0.3.1"


def test_a_uv_the_user_installed_elsewhere_is_kept_and_named(tmp_path, home, monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", uv="0.13.4"))
    mine = tmp_path / "mine" / uv_tool.UV_NAME
    mine.parent.mkdir()
    mine.write_text("0.12.0", encoding="utf-8")
    monkeypatch.setattr(uv_tool, "find_uv", lambda root: str(mine))
    monkeypatch.setattr(uv_tool, "uv_version", lambda p: "0.12.0")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", uv="0.13.6"))
    p = _plans(staged, inst)["uv"]
    assert p["update"] is False and p["kept"] is True
    assert "kept your own uv 0.12.0" in p["note"]


# ------------------------------- the PyTorch stack ---------------------------

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
    for info in site.glob(f"{name}-*.dist-info"):
        for row in (info / "RECORD").read_text(encoding="utf-8").splitlines():
            p = site / row.split(",")[0]
            if p.is_file():
                p.unlink()
        import shutil
        shutil.rmtree(info)


def _venv(root: Path, *, torch="2.11.0+rocm7.13.0", transformers="5.15.1") -> Path:
    site = root / ".venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    if torch:
        _dist(site, "torch", torch, {"torch/__init__.py": b"torch " + torch.encode()})
    _dist(site, "transformers", transformers,
          {"transformers/__init__.py": b"tf " + transformers.encode()})
    return site


@pytest.fixture
def torch_install(tmp_path, home, monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", pyproject=_PYPROJECT_031))
    lib = inst / au.RUNTIME_LIB
    lib.mkdir(parents=True)
    (lib / ".localm-backend").write_text("vulkan b11928", encoding="utf-8")
    site = _venv(inst)
    monkeypatch.setattr(uc, "torch_args", lambda root, backend: "-e .[gpu]")
    monkeypatch.setattr(updater, "_uv_for_deps", lambda t: "/x/uv")
    return inst, site


def _fake_uv_pip(site: Path, transformers="5.20.2", rc=0):
    def run(cmd):
        if cmd[0] != "/x/uv" or ".[gpu,audio]" not in cmd:
            return 0
        if rc:
            _uninstall(site, "transformers")
            return rc
        _uninstall(site, "transformers")
        _dist(site, "transformers", transformers,
              {"transformers/__init__.py": b"tf " + transformers.encode()})
        return 0
    return run


def test_pytorch_stack_is_re_installed_when_the_release_pins_another_one(tmp_path,
                                                                        torch_install):
    inst, site = torch_install
    staged = _write_tree(tmp_path / "s", _files("0.3.2", pyproject=_PYPROJECT_032))
    p = _plans(staged, inst)["pytorch"]
    assert p["update"] is True and "transformers 5.15.1 (wants ~=5.20.2)" in p["note"]
    assert p["items"] == [["pip", "install", "-p", ".venv", "-e", ".[gpu,audio]"]]
    shallow = _plans(staged, inst, deep=False)["pytorch"]
    assert shallow["check"] is True and shallow["update"] is False


def test_pytorch_update_and_rollback_through_apply(tmp_path, torch_install):
    inst, site = torch_install
    data, sig = _release(_files("0.3.2", pyproject=_PYPROJECT_032))
    calls = []
    fake = _fake_uv_pip(site)
    res = updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                        runner=lambda c: calls.append(c) or fake(c))
    assert ["/x/uv", "pip", "install", "-p", ".venv", "-e", ".[gpu,audio]"] in calls
    assert res["components"]["state"] == "updated"
    assert uc.dists(site)["transformers"]["version"] == "5.20.2"
    updater.rollback_last(installed=inst)
    assert uc.dists(site)["transformers"]["version"] == "5.15.1"
    assert (site / "transformers" / "__init__.py").read_bytes() == b"tf 5.15.1"
    assert not list(site.glob("transformers-5.20.2.dist-info"))


def test_a_failed_pytorch_step_puts_the_old_packages_back(tmp_path, torch_install):
    inst, site = torch_install
    data, sig = _release(_files("0.3.2", pyproject=_PYPROJECT_032))
    with pytest.raises(LocalmError) as ei:
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_fake_uv_pip(site, rc=2))
    assert "PyTorch stack was not updated" in ei.value.reason
    assert uc.dists(site)["transformers"]["version"] == "5.15.1"
    assert (site / "transformers" / "__init__.py").read_bytes() == b"tf 5.15.1"
    assert _v(inst) == "0.3.1"


def test_no_pytorch_is_ever_installed_where_there_was_none(tmp_path, home, monkeypatch):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1"))
    _venv(inst, torch=None)
    monkeypatch.setattr(uc, "torch_args", lambda root, backend: pytest.fail("not asked"))
    staged = _write_tree(tmp_path / "s", _files("0.3.2", pyproject=_PYPROJECT_032))
    assert _plans(staged, inst)["pytorch"]["installed"] is False


_HWDETECT = ('_TORCH_BACKEND = {"cuda": "cu126", "cuda-blackwell": "cu130", '
             '"rocm-linux": "rocm6.2", "cpu": "cpu"}\n'
             '_AMD_ROCM_WIN_TORCH = "torch==2.9.1+rocm7.2.1"\n')


def _cuda_torch_install(tmp_path, torch_version):
    files = _files("0.3.1", pyproject=_PYPROJECT_032) | {"localm/hwdetect.py": _HWDETECT}
    inst = _write_tree(tmp_path / "inst", files)
    lib = inst / au.RUNTIME_LIB
    lib.mkdir(parents=True)
    (lib / ".localm-backend").write_text("cuda b11928", encoding="utf-8")
    _venv(inst, torch=torch_version, transformers="5.20.2")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", pyproject=_PYPROJECT_032)
                         | {"localm/hwdetect.py": _HWDETECT})
    return inst, staged


def test_pytorch_wheel_line_setup_installs_is_moved_to_the_releases_line(
        tmp_path, home, monkeypatch):
    inst, staged = _cuda_torch_install(tmp_path, "2.9.0+cu126")
    monkeypatch.setattr(uc, "torch_args",
                        lambda root, backend: "torch torchvision --torch-backend=cu130")
    p = _plans(staged, inst)["pytorch"]
    assert p["update"] and "wants the cu130 build" in p["note"]
    assert p["items"][0] == ["pip", "install", "-p", ".venv", "torch", "torchvision",
                             "--torch-backend=cu130"]


@pytest.mark.parametrize("have,spec", [
    ("2.9.0+cu128", "torch torchvision --torch-backend=cu126"),
    ("2.9.0+cu126", "torch torchvision --torch-backend=rocm6.2"),
    ("2.9.0", "torch torchvision --torch-backend=cu126"),
    ("2.9.0+cpu", "torch==2.9.1+rocm7.2.1 torchvision==0.24.1+rocm7.2.1 --find-links https://x/"),
])
def test_a_pytorch_build_setup_does_not_install_is_the_users_and_kept(tmp_path, home,
                                                                      monkeypatch, have, spec):
    inst, staged = _cuda_torch_install(tmp_path, have)
    monkeypatch.setattr(uc, "torch_args", lambda root, backend: spec)
    p = _plans(staged, inst)["pytorch"]
    assert p["update"] is False and p["kept"] is True
    assert f"kept your PyTorch {have}" in p["note"]


def test_an_older_exact_pin_of_the_same_build_family_is_updated(tmp_path, home, monkeypatch):
    inst, staged = _cuda_torch_install(tmp_path, "2.9.0+rocm7.1.0")
    monkeypatch.setattr(uc, "torch_args", lambda root, backend: (
        "torch==2.9.1+rocm7.2.1 torchvision==0.24.1+rocm7.2.1 --find-links https://x/"))
    p = _plans(staged, inst)["pytorch"]
    assert p["update"] is True and "torch 2.9.0+rocm7.1.0 (wants ==2.9.1+rocm7.2.1)" in p["note"]


def test_torch_args_asks_the_staged_tree_itself(tmp_path):
    root = Path(__file__).resolve().parents[1]
    from localm import hwdetect
    assert uc.torch_args(root, "vulkan") == hwdetect.torch_pip_args("vulkan")


# ------------------------------- the managed ComfyUI --------------------------

def _git(cwd, *args):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _comfy(home: Path, *, version_label="v0.42.1"):
    """A managed ComfyUI: a git checkout with two commits (the old and the new
    pin), localm's patch edit on the old one, a marker, and a venv."""
    root = home / "comfyui"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "main.py").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "one")
    old = _git(root, "rev-parse", "HEAD")
    (root / "main.py").write_text("v2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "two")
    new = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", old)
    (root / "main.py").write_text("v1 patched\n", encoding="utf-8")
    marker = {"source": "fresh", "commit": old, "comfyui_version": version_label}
    (root / ".localm-comfy.json").write_text(json.dumps(marker), encoding="utf-8")
    py = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    site = root / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    _dist(site, "torch", "2.9.1+rocm7.13.0", {"torch/__init__.py": b"t291"})
    return root, old, new, site


def _comfy_update_child(root: Path, new: str, site: Path, rc=0):
    def run(cmd):
        if "comfy" not in cmd:
            return 0
        _git(root, "checkout", "-q", "--force", new)
        (root / "main.py").write_text("v2 patched\n", encoding="utf-8")
        marker = json.loads((root / ".localm-comfy.json").read_text(encoding="utf-8"))
        marker.update(commit=new, comfyui_version="v0.43.0")
        (root / ".localm-comfy.json").write_text(json.dumps(marker), encoding="utf-8")
        _uninstall(site, "torch")
        _dist(site, "torch", "2.11.0+rocm7.13.0", {"torch/__init__.py": b"t211"})
        return rc
    return run


def test_managed_comfy_is_updated_and_rolled_back_with_its_edits_and_venv(
        tmp_path, home, monkeypatch):
    root, old, new, site = _comfy(home)
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", comfy=(old, "v0.42.1")))
    monkeypatch.setattr(uc, "comfy_info", lambda tree: {"torch": [], "nodes": []})
    stopped = []
    monkeypatch.setattr(uc, "_stop_comfy", lambda r, say: stopped.append(r) or [])
    data, sig = _release(_files("0.3.2", comfy=(new, "v0.43.0")))
    calls = []
    child = _comfy_update_child(root, new, site)
    res = updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                        runner=lambda c: calls.append(c) or child(c))
    assert [sys.executable, "-m", "localm", "comfy", "update",
            "--reinstall-requirements"] in calls
    assert stopped, "the managed ComfyUI is stopped before it is replaced"
    assert res["components"]["state"] == "updated", res["components"]
    assert _git(root, "rev-parse", "HEAD") == new
    updater.rollback_last(installed=inst)
    assert _git(root, "rev-parse", "HEAD") == old
    assert (root / "main.py").read_text(encoding="utf-8") == "v1 patched\n"
    marker = json.loads((root / ".localm-comfy.json").read_text(encoding="utf-8"))
    assert marker["comfyui_version"] == "v0.42.1"
    assert uc.dists(site)["torch"]["version"] == "2.9.1+rocm7.13.0"
    assert (site / "torch" / "__init__.py").read_bytes() == b"t291"


def test_a_failed_comfy_update_puts_it_back_and_rolls_back_the_code(tmp_path, home,
                                                                    monkeypatch):
    root, old, new, site = _comfy(home)
    inst = _write_tree(tmp_path / "inst", _files("0.3.1", comfy=(old, "v0.42.1")))
    monkeypatch.setattr(uc, "comfy_info", lambda tree: {"torch": [], "nodes": []})
    data, sig = _release(_files("0.3.2", comfy=(new, "v0.43.0")))
    with pytest.raises(LocalmError):
        updater.apply(1, signature=sig, installed=inst, download_opener=_download(data),
                      runner=_comfy_update_child(root, new, site, rc=1))
    assert _git(root, "rev-parse", "HEAD") == old
    assert (root / "main.py").read_text(encoding="utf-8") == "v1 patched\n"
    assert uc.dists(site)["torch"]["version"] == "2.9.1+rocm7.13.0"
    assert _v(inst) == "0.3.1"


def test_a_comfy_on_a_commit_the_user_chose_is_kept_and_named(tmp_path, home):
    root, old, new, _site = _comfy(home, version_label=None)
    staged = _write_tree(tmp_path / "s", _files("0.3.2", comfy=(new, "v0.43.0")))
    p = _plans(staged, staged)["comfyui"]
    assert p["update"] is False and p["kept"] is True
    assert "kept the managed ComfyUI on the version you chose" in p["note"]


def test_comfy_at_the_pin_still_updates_a_changed_torch_or_custom_node(tmp_path, home,
                                                                      monkeypatch):
    root, old, _new, _site = _comfy(home)
    node = root / "custom_nodes" / "ComfyUI-GGUF"
    node.mkdir(parents=True)
    _git(node, "init", "-q")
    _git(node, "config", "user.email", "t@example.invalid")
    _git(node, "config", "user.name", "t")
    (node / "n.py").write_text("1", encoding="utf-8")
    _git(node, "add", "-A")
    _git(node, "commit", "-qm", "n")
    staged = _write_tree(tmp_path / "s", _files("0.3.2", comfy=(old, "v0.42.1")))
    monkeypatch.setattr(uc, "comfy_info", lambda tree: {
        "torch": ["torch==2.11.0+rocm7.13.0", "--extra-index-url", "https://x/"],
        "nodes": [["ComfyUI-GGUF", "https://x/g.git", "f" * 40]]})
    p = _plans(staged, staged)["comfyui"]
    assert p["update"] is True
    assert any(i.startswith("PyTorch: torch 2.9.1+rocm7.13.0") for i in p["items"])
    assert any(i.startswith("custom node ComfyUI-GGUF") for i in p["items"])


def test_comfy_update_moves_installed_custom_nodes_to_their_pin(tmp_path, home, monkeypatch):
    from localm.media import managed_comfy_fresh as f
    from localm.media import managed_comfy_update as upd
    src = tmp_path / "node-src"
    src.mkdir()
    _git(src, "init", "-q")
    _git(src, "config", "user.email", "t@example.invalid")
    _git(src, "config", "user.name", "t")
    (src / "n.py").write_text("1", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "-qm", "a")
    first = _git(src, "rev-parse", "HEAD")
    (src / "n.py").write_text("2", encoding="utf-8")
    _git(src, "commit", "-qam", "b")
    pinned = _git(src, "rev-parse", "HEAD")
    root = tmp_path / "comfy"
    (root / "custom_nodes").mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(src), str(root / "custom_nodes" / "N")],
                   check=True)
    _git(root / "custom_nodes" / "N", "checkout", "-q", first)
    pin = f.CustomNodePin(name="N", repo=str(src), commit=pinned)
    monkeypatch.setattr(f, "required_custom_nodes", lambda: f.RequiredNodes((pin,), ()))
    assert upd._node_heads(root) == {"N": first}
    ok, detail = upd._sync_custom_nodes(root, None, lambda m: None)
    assert ok, detail
    assert _git(root / "custom_nodes" / "N", "rev-parse", "HEAD") == pinned


def test_comfy_update_installs_the_pinned_torch_only_when_it_differs(tmp_path, monkeypatch):
    from localm.media import managed_comfy_fresh as f
    from localm.media import managed_comfy_update as upd
    venv = tmp_path / "venv"
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True)
    _dist(site, "torch", "2.9.1+rocm7.13.0", {"torch/__init__.py": b"x"})
    py = venv / "Scripts" / "python.exe"
    spec = f.ComfyTorchSpec("rocm", ("torch==2.9.1+rocm7.13.0",), extra_index_url="https://x/")
    monkeypatch.setattr(f, "comfy_torch_spec", lambda det=None: spec)
    ran = []
    monkeypatch.setattr(upd, "_run", lambda argv, **kw: ran.append(argv) or (True, ""))
    assert upd._sync_pinned_torch(py, None, lambda m: None) == (True, "")
    assert ran == []
    spec = f.ComfyTorchSpec("rocm", ("torch==2.11.0+rocm7.13.0",), extra_index_url="https://x/")
    assert upd._sync_pinned_torch(py, None, lambda m: None) == (True, "")
    assert ran == [[str(py), "-m", "pip", "install", "--disable-pip-version-check",
                    "torch==2.11.0+rocm7.13.0", "--extra-index-url", "https://x/"]]


# ------------------------------ the Linux CUDA libraries ----------------------

_CUDA_032 = ('_CUDA_RUNTIME_PYPI_PACKAGES = {"cuda-13": ("nvidia-cublas",)}\n'
             '_CUDA_RUNTIME_PIN = {"nvidia-cublas": ("13.8.2.12", "x")}\n')
_CUDA_031 = _CUDA_032.replace("13.8.2.12", "13.8.1.7")


def _cuda_tree(root: Path, cuda: str, version="0.3.2") -> Path:
    return _write_tree(root, _files(version) | {"localm/setup_llama/cuda.py": cuda})


def test_cuda_libraries_behind_the_pin_reprovision_the_same_build(tmp_path, home):
    inst = _cuda_tree(tmp_path / "inst", _CUDA_031, "0.3.1")
    lib = inst / au.RUNTIME_LIB
    lib.mkdir(parents=True)
    (lib / ".localm-backend").write_text("cuda b11928", encoding="utf-8")
    (lib / "libcublas.so.13").write_bytes(b"x")
    p = updater_runtime.plan(_cuda_tree(tmp_path / "s", _CUDA_032), inst)
    assert p["reprovision"] is True
    assert "nvidia-cublas 13.8.1.7 -> 13.8.2.12" in p["note"]
    from localm.setup_llama.cuda import CUDA_RUNTIME_RECORD
    (lib / CUDA_RUNTIME_RECORD).write_text(json.dumps(
        {"line": "cuda-13", "packages": {"nvidia-cublas": "13.8.2.12"}}), encoding="utf-8")
    assert updater_runtime.plan(_cuda_tree(tmp_path / "s", _CUDA_032), inst)[
        "reprovision"] is False


def test_the_cuda_record_is_written_with_the_fetched_libraries(tmp_path, monkeypatch):
    import localm.setup_llama as sl
    from localm.setup_llama import cuda
    monkeypatch.setattr(sl, "_fetch_pypi_runtime_lib", lambda pkg, target: 1)
    assert cuda._fetch_cuda_runtime_libs("cuda-13", tmp_path) == 2
    rec = cuda.read_cuda_runtime(tmp_path)
    assert rec["line"] == "cuda-13"
    assert rec["packages"] == {p: cuda._CUDA_RUNTIME_PIN[p][0]
                               for p in cuda._CUDA_RUNTIME_PYPI_PACKAGES["cuda-13"]}
    assert updater_runtime.cuda_pins(Path(__file__).resolve().parents[1])["cuda-13"] == \
        rec["packages"]


# ----------------------------------- doctor ----------------------------------

def test_doctor_shows_each_component_against_the_pin(tmp_path, home, monkeypatch, capsys):
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    inst = _write_tree(tmp_path / "inst", _files("0.3.2", sd="master-1017-be48649",
                                                 kobold="v1.123.1"))
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    _kobold_runtime(home, "v1.123.1", "nocuda")
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    import importlib
    d = importlib.import_module("localm.cli.doctor")
    d._check_components()
    out = " ".join(capsys.readouterr().out.split())
    assert ("stable-diffusion.cpp image runtime: this release pins master-1017-be48649, but "
            "master-982-0518b34 is installed") in out
    assert "KoboldCpp music runtime: v1.123.1 (this release's pin)" in out


# ------------------------------ snapshot safety ---------------------------------

def _foreign_copy_snapshot(tmp_path):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1"))
    _venv(inst)
    victim = tmp_path / ".victim-backup-components"
    (victim / "Lib" / "site-packages").mkdir(parents=True)
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    snap = tmp_path / "updates" / "backup-components"
    snap.mkdir(parents=True)
    (snap / uc.STATE_FILE).write_text(json.dumps(
        {"root": str(inst), "components": {"pytorch": {"venv_copy": str(victim)}}}),
        encoding="utf-8")
    return inst, victim, snap


def test_restore_refuses_a_venv_copy_the_snapshot_did_not_make(tmp_path, home):
    inst, _victim, snap = _foreign_copy_snapshot(tmp_path)
    before = sorted(p.name for p in (inst / ".venv" / "Lib" / "site-packages").iterdir())
    errors = uc.restore(snap, root=inst)
    assert errors and "is not a venv copy" in errors[0]
    assert sorted(p.name for p in (inst / ".venv" / "Lib" / "site-packages").iterdir()) == before


def test_discard_never_removes_a_folder_the_snapshot_did_not_make(tmp_path, home):
    _inst, victim, snap = _foreign_copy_snapshot(tmp_path)
    uc.discard(snap)
    assert (victim / "keep.txt").is_file()
    assert not snap.exists()


def test_a_snapshot_venv_copy_is_marked_and_removed_with_its_snapshot(tmp_path, home):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1"))
    _venv(inst)
    snap = tmp_path / "updates" / "backup-components"
    state = uc.snapshot(["pytorch"], snap, root=inst)
    copy = Path(state["components"]["pytorch"]["venv_copy"])
    assert uc._copy_owner(copy) == str(snap.resolve())
    uc.discard(snap)
    assert not copy.exists() and not snap.exists()


def test_a_restore_keeps_a_file_two_distributions_share(tmp_path, home):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1"))
    site = _venv(inst)
    _dist(site, "opencv-python", "4.9.0", {"cv2/__init__.py": b"cv2 full"})
    snap = tmp_path / "updates" / "backup-components"
    uc.snapshot(["pytorch"], snap, root=inst)
    _uninstall(site, "opencv-python")
    _dist(site, "opencv-python-headless", "4.10.0", {"cv2/__init__.py": b"cv2 headless"})
    assert uc.restore(snap, root=inst) == []
    assert (site / "cv2" / "__init__.py").read_bytes() == b"cv2 full"
    assert "opencv-python-headless" not in uc.dists(site)


def test_a_later_snapshot_adds_to_the_updates_own_and_keeps_it(tmp_path, home,
                                                               fake_uv_version):
    inst = _write_tree(tmp_path / "inst", _files("0.3.1"))
    _venv(inst)
    _contained_uv(inst, "0.13.4")
    snap = tmp_path / "updates" / "backup-components"
    uc.snapshot(["uv"], snap, root=inst)
    (inst / ".uv" / uv_tool.UV_NAME).unlink()
    (inst / ".uv" / uv_tool.UV_NAME).write_text("0.13.6", encoding="utf-8")
    state = uc.snapshot(["uv", "pytorch"], snap, root=inst, add=True)
    assert set(state["components"]) == {"uv", "pytorch"}
    assert (snap / "uv" / uv_tool.UV_NAME).read_text(encoding="utf-8") == "0.13.4", \
        "the older copy, the one a rollback needs, is kept"


# --------------------------- first-start check cadence --------------------------

def _start_components(inst, home, monkeypatch, command="gui", rc=0, calls=None):
    from localm.cli import _core
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
    fake = _fake_children(home, rc=rc, calls=calls)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, fake(cmd)))
    _core._finish_pending_component_update(command)


def test_a_failed_first_start_waits_before_trying_again(tmp_path, home, monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    inst = _write_tree(tmp_path / "inst", _files("0.3.2", sd="master-1017-be48649"))
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    calls = []
    _start_components(inst, home, monkeypatch, rc=1, calls=calls)
    assert len(calls) == 1 and uc.last_status()["state"] == "failed"
    _start_components(inst, home, monkeypatch, calls=calls)
    assert len(calls) == 1, "the next start does not retry at once"
    status = uc.last_status()
    status["at"] -= uc.RETRY_AFTER_S + 1
    (updater._updates_dir() / uc.STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    _start_components(inst, home, monkeypatch, calls=calls)
    assert len(calls) == 2 and uc.last_status()["state"] == "updated"


def test_a_pending_update_of_another_install_is_dropped_and_this_one_checked(
        tmp_path, home, monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2")
    inst = _write_tree(tmp_path / "inst", _files("0.3.2", sd="master-1017-be48649"))
    _sd_runtime(home, "master-982-0518b34", "vulkan")
    uc.write_pending([{"name": "sdcpp", "update": True}], root=tmp_path / "elsewhere",
                     version="0.3.2")
    calls = []
    _start_components(inst, home, monkeypatch, calls=calls)
    assert [c[3] for c in calls] == ["setup-sdcpp"]
    assert uc.read_pending() is None
