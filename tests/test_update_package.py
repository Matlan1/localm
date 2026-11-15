# SPDX-License-Identifier: AGPL-3.0-or-later
"""The update package statement (``update-package.json``): stamping a build, the
check at update time, the exact set of entries an update replaces, and the
refusal with nothing changed of a package that is cut off, damaged, unsigned or
does not fit on the disk, and the rollback of a swap the disk fills during."""

import errno
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _apply_update as au
from localm import _version, update_package, updater, updater_components
from localm.bugreport import LocalmError
from tests._net_config import set_config
from tests._update_signing import envelope, keyset

_PRIV = Ed25519PrivateKey.generate()
_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]

_SHA = {c: c * 64 for c in "abcdef0123"}
_PINS = updater_components.stated_components

_SDCPP = (
    'TAG = "master-9-abc"\n'
    "ASSETS: dict = {\n"
    f'    ("windows", "cuda"): ("sd-win-cuda.zip", "{_SHA["a"]}"),\n'
    f'    ("linux", "vulkan"): ("sd-linux-vulkan.zip", "{_SHA["b"]}"),\n'
    "}\n"
    "EXTRA_ASSETS: dict = {\n"
    f'    ("windows", "cuda"): [("cudart-sd.zip", "{_SHA["c"]}")],\n'
    "}\n"
)
_KOBOLD = (
    'TAG = "v1.2.3"\nVERSION = "1.2.3"\n'
    f'ASSETS: dict = {{("linux", "nocuda"): ("kobold-linux", 10, "{_SHA["d"]}")}}\n'
)
_LLAMA = (
    'DEFAULT_URL = (\n    "https://example.invalid/dl/"\n    "llama-b1-windows-rocm-gfx103X-x64.zip"\n)\n'
    f'DEFAULT_URL_SHA256 = (\n    "{_SHA["0"]}"\n)\n'
    '_ROCM_TAG = "b1"\n_ROCM_CPU_TAG = "b2"\n_ROCM_BUILD = f"{_ROCM_TAG}-cpu-{_ROCM_CPU_TAG}"\n'
    '_PINNED_TAG = "b3"\n'
    f'_CUDA_LINUX_PINNED_SHA256 = {{"llama-b3-bin-ubuntu-cuda-x64.tar.gz": "{_SHA["1"]}"}}\n'
    f'_PINNED_FALLBACK_SHA256 = {{"llama-b3-bin-win-vulkan-x64.zip": "{_SHA["e"]}"}}\n'
)
_CUDA = (
    '_CUDA_RUNTIME_PYPI_PACKAGES = {"cuda-12": ("nvidia-cuda-runtime-cu12",)}\n'
    f'_CUDA_RUNTIME_PIN = {{"nvidia-cuda-runtime-cu12": ("12.9.79", "{_SHA["2"]}")}}\n'
)
_SETUP_BAT = f'set "UV_INSTALLER_VERSION=0.9.1"\nset "UV_INSTALLER_SHA256={_SHA["f"]}"\n'
_SETUP_SH = f'UV_INSTALLER_VERSION="0.9.1"\nUV_INSTALLER_SHA256="{_SHA["3"]}"\n'
_PYPROJECT = ('[project]\nname = "localm"\ndependencies = ["click"]\n'
              '[project.optional-dependencies]\ngpu = ["torch>=2.9", "transformers"]\n')


def _tree(version="0.4.0", **extra):
    files = {
        "VERSION": version,
        "pyproject.toml": _PYPROJECT,
        "localm/__init__.py": f"# {version}",
        "localm/media/sdcpp/pins.py": _SDCPP,
        "localm/media/koboldcpp/pins.py": _KOBOLD,
        "localm/setup_llama/pins.py": _LLAMA,
        "localm/setup_llama/cuda.py": _CUDA,
        "localm/media/managed_comfy_fresh.py": 'COMFYUI_PINNED_COMMIT = "' + "1" * 40 + '"\n',
        "setup.bat": _SETUP_BAT,
        "setup.sh": _SETUP_SH,
        "docs/readme.txt": "hello",
    }
    files.update(extra)
    return files


def _zip(files) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def _statement(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return json.loads(z.read(update_package.NAME))


def _with_statement(data: bytes, statement: dict) -> bytes:
    """*data* (a zip without a statement) with *statement* added as written."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        zout.writestr(update_package.NAME, json.dumps(statement))
    return out.getvalue()


def _extract(data: bytes, tmp_path: Path) -> Path:
    zp = tmp_path / "b.zip"
    zp.write_bytes(data)
    return au.extract(zp, tmp_path / "staging")


# ------------------------------- stamping -------------------------------------

def test_stamp_states_every_file_and_every_component_per_platform():
    files = _tree()
    data = update_package.stamped(_zip(files), _PINS)
    st = _statement(data)
    assert st["format"] == update_package.FORMAT and st["version"] == "0.4.0"
    assert set(st["files"]) == set(files)
    assert st["files"]["docs/readme.txt"]["size"] == 5
    comps = st["components"]
    assert set(comps) == set(updater_components.STATED)
    assert comps["sdcpp"] == {"version": "master-9-abc", "assets": {
        "linux/vulkan": _SHA["b"], "windows/cuda": _SHA["a"],
        "windows/cuda/cudart-sd.zip": _SHA["c"]}}
    assert comps["koboldcpp"] == {"version": "v1.2.3", "assets": {"linux/nocuda": _SHA["d"]}}
    assert comps["llama.cpp"] == {"version": "b3 amd-rocm b1-cpu-b2", "assets": {
        "llama-b1-windows-rocm-gfx103X-x64.zip": _SHA["0"],
        "llama-b3-bin-ubuntu-cuda-x64.tar.gz": _SHA["1"],
        "llama-b3-bin-win-vulkan-x64.zip": _SHA["e"]}}
    assert comps["cuda-runtime"] == {
        "version": "cuda-12 nvidia-cuda-runtime-cu12==12.9.79",
        "assets": {"linux-x86_64/nvidia-cuda-runtime-cu12==12.9.79": _SHA["2"]}}
    assert comps["uv"] == {"version": "0.9.1", "assets": {
        "posix/installer": _SHA["3"], "windows/installer": _SHA["f"]}}
    assert comps["comfyui"] == {"version": "1" * 40, "assets": {}}
    assert comps["pytorch"] == {"version": "torch>=2.9; transformers", "assets": {}}
    assert update_package.parse(json.dumps(st)) == st


def test_the_shipped_tree_states_every_component(tmp_path):
    """The real pins of this checkout: every component an update can carry is
    stated, each machine-specific download with a checksum."""
    root = Path(updater.__file__).resolve().parent.parent
    comps = updater_components.stated_components(root)
    assert set(comps) == set(updater_components.STATED)
    for name in ("llama.cpp", "cuda-runtime", "sdcpp", "koboldcpp", "uv"):
        assert comps[name]["assets"], name


def test_stamp_refuses_a_second_statement():
    data = update_package.stamped(_zip(_tree()), _PINS)
    with pytest.raises(update_package.PackageError, match="already"):
        update_package.stamped(data, _PINS)


def test_stamp_refuses_an_unsafe_member():
    with pytest.raises(update_package.PackageError, match="unsafe"):
        update_package.stamped(_zip(_tree(**{"../evil.py": "x"})), _PINS)


def test_stamp_refuses_a_build_whose_statement_does_not_read_back():
    """The release build cannot finish a package whose statement does not match it:
    a component reader that answers differently on the read-back is refused."""
    answers = iter([{}, {"uv": {"version": "1", "assets": {}}}])
    with pytest.raises(update_package.PackageError, match="uv"):
        update_package.stamped(_zip(_tree()), lambda root: next(answers))


def test_a_tree_without_pins_states_no_components():
    st = _statement(update_package.stamped(_zip({"VERSION": "0.4.0", "localm/a.py": "x"}),
                                           _PINS))
    assert st["components"] == {}


def test_a_component_without_a_reader_fails_every_statement(monkeypatch):
    monkeypatch.setattr(updater_components, "STATED", updater_components.STATED + ("new",))
    with pytest.raises(KeyError):
        update_package.stamped(_zip(_tree()), _PINS)


# ------------------------------- the check ------------------------------------

def test_check_accepts_the_stamped_build(tmp_path):
    root = _extract(update_package.stamped(_zip(_tree()), _PINS), tmp_path)
    st = update_package.check(root, "0.4.0", _PINS)
    assert st is not None and "localm/__init__.py" in st["files"]


def test_check_refuses_a_cut_off_file(tmp_path):
    root = _extract(update_package.stamped(_zip(_tree()), _PINS), tmp_path)
    p = root / "localm" / "__init__.py"
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(update_package.PackageError) as ei:
        update_package.check(root, "0.4.0", _PINS)
    assert "localm/__init__.py" in str(ei.value) and "differ" in str(ei.value)


def test_check_refuses_a_missing_file(tmp_path):
    root = _extract(update_package.stamped(_zip(_tree()), _PINS), tmp_path)
    (root / "docs" / "readme.txt").unlink()
    with pytest.raises(update_package.PackageError) as ei:
        update_package.check(root, "0.4.0", _PINS)
    assert "incomplete" in str(ei.value) and "docs/readme.txt" in str(ei.value)


def test_check_refuses_a_file_it_does_not_state(tmp_path):
    root = _extract(update_package.stamped(_zip(_tree()), _PINS), tmp_path)
    (root / "extra.py").write_text("x", encoding="utf-8")
    with pytest.raises(update_package.PackageError) as ei:
        update_package.check(root, "0.4.0", _PINS)
    assert "does not state" in str(ei.value) and "extra.py" in str(ei.value)


def test_check_refuses_a_component_stated_differently_from_the_pins(tmp_path):
    data = _zip(_tree())
    st = _statement(update_package.stamped(data, _PINS))
    st["components"]["sdcpp"]["assets"]["windows/cuda"] = _SHA["b"]
    root = _extract(_with_statement(data, st), tmp_path)
    with pytest.raises(update_package.PackageError) as ei:
        update_package.check(root, "0.4.0", _PINS)
    assert "sdcpp" in str(ei.value)


def test_check_refuses_a_component_the_build_pins_but_does_not_state(tmp_path):
    data = _zip(_tree())
    st = _statement(update_package.stamped(data, _PINS))
    del st["components"]["koboldcpp"]
    root = _extract(_with_statement(data, st), tmp_path)
    with pytest.raises(update_package.PackageError) as ei:
        update_package.check(root, "0.4.0", _PINS)
    assert "koboldcpp" in str(ei.value)


def test_check_refuses_a_component_the_build_does_not_pin(tmp_path):
    data = _zip(_tree())
    st = _statement(update_package.stamped(data, _PINS))
    st["components"]["made-up"] = {"version": "1", "assets": {}}
    root = _extract(_with_statement(data, st), tmp_path)
    with pytest.raises(update_package.PackageError, match="made-up"):
        update_package.check(root, "0.4.0", _PINS)


def test_a_build_newer_than_the_last_without_one_must_carry_a_statement(tmp_path):
    root = _extract(_zip(_tree(version="0.3.4")), tmp_path)
    with pytest.raises(update_package.MissingStatement):
        update_package.check(root, "0.3.4", _PINS)


@pytest.mark.parametrize("version", [update_package.LAST_WITHOUT, "0.3.3"])
def test_an_older_build_without_a_statement_is_applied_from_its_tree(tmp_path, version):
    root = _extract(_zip(_tree(version=version)), tmp_path)
    assert update_package.check(root, version, _PINS) is None


def test_check_refuses_a_statement_for_another_version(tmp_path):
    root = _extract(update_package.stamped(_zip(_tree()), _PINS), tmp_path)
    (root / "VERSION").write_text("0.4.1", encoding="utf-8")
    with pytest.raises(update_package.PackageError, match="0.4.1"):
        update_package.check(root, "0.4.1", _PINS)


@pytest.mark.parametrize("bad", [
    {"format": "localm-update-package/9"},
    {"extra": 1},
    {"files": {}},
    {"files": {"../evil": {"sha256": "a" * 64, "size": 1}}},
    {"files": {"C:/evil": {"sha256": "a" * 64, "size": 1}}},
    {"files": {"a": {"sha256": "nothex", "size": 1}}},
    {"files": {"a": {"sha256": "a" * 64, "size": -1}}},
    {"files": {"a": {"sha256": "a" * 64, "size": True}}},
    {"components": {"uv": {"version": "1", "assets": {"x": "short"}}}},
    {"components": {"uv": {"version": "", "assets": {}}}},
])
def test_parse_refuses_a_malformed_statement(bad):
    st = {"format": update_package.FORMAT, "version": "0.4.0",
          "files": {"a": {"sha256": "a" * 64, "size": 1}}, "components": {}}
    st.update(bad)
    with pytest.raises(update_package.PackageError):
        update_package.parse(json.dumps(st))


def test_parse_refuses_duplicate_fields():
    text = ('{"format": "%s", "format": "%s", "version": "1", "files": {}, "components": {}}'
            % (update_package.FORMAT, update_package.FORMAT))
    with pytest.raises(update_package.PackageError, match="duplicate"):
        update_package.parse(text)


# ------------------------------- disk space -----------------------------------

def test_space_counts_new_files_and_backup_on_one_drive(tmp_path):
    root, target, backup = tmp_path / "r", tmp_path / "t", tmp_path / "b"
    for d in (root / "localm", target / "localm", backup):
        d.mkdir(parents=True)
    (root / "localm" / "a.bin").write_bytes(b"x" * 3000)
    (target / "localm" / "a.bin").write_bytes(b"x" * 2000)
    assert au.space_needed(root, target, ["localm"]) == (3000, 2000)
    exact = au.SPACE_MARGIN + 5000
    au.check_space(root, target, ["localm"], backup, free=lambda p: exact)
    with pytest.raises(au.NoSpace):
        au.check_space(root, target, ["localm"], backup, free=lambda p: exact - 1)


# ------------------------------- apply ----------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [_PRIV], _RECOVERY))
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.4b2")
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.3.4b2", encoding="utf-8")
    (inst / "localm" / "old_only.py").write_text("old", encoding="utf-8")
    (inst / "VERSION").write_text("0.3.4b2", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
    (inst / "mine").mkdir()
    (inst / "mine" / "notes.txt").write_text("kept", encoding="utf-8")
    return home, inst


def _build(version="0.4.0", extra=None):
    files = {
        "VERSION": version,
        "pyproject.toml": '[project]\nname = "localm"\ndependencies = ["click"]\n',
        "localm/__init__.py": f"# {version}",
        "localm/new.py": "new",
    }
    files.update(extra or {})
    return update_package.stamped(_zip(files), _PINS)


def _download(data):
    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _apply(inst, data):
    return updater.apply(4, signature=envelope(data, [_PRIV]), installed=inst,
                         download_opener=_download(data))


def test_apply_replaces_exactly_the_stated_entries(env):
    _home, inst = env
    data = _build(extra={".github/workflow.yml": "x"})
    res = _apply(inst, data)
    assert res["applied"] is True and res["version"] == "0.4.0"
    assert (inst / "localm" / "new.py").read_text(encoding="utf-8") == "new"
    assert not (inst / "localm" / "old_only.py").exists()
    assert (inst / "mine" / "notes.txt").read_text(encoding="utf-8") == "kept"
    assert not (inst / ".github").exists()
    assert json.loads((inst / update_package.NAME).read_text(encoding="utf-8")) == _statement(data)


def test_apply_refuses_a_build_without_a_statement_and_changes_nothing(env):
    _home, inst = env
    before = _snapshot(inst)
    data = _zip({"VERSION": "0.4.0", "pyproject.toml": '[project]\nname = "localm"\n',
                 "localm/__init__.py": "# 0.4.0"})
    with pytest.raises(LocalmError) as ei:
        _apply(inst, data)
    assert "does not state" in ei.value.summary
    assert _snapshot(inst) == before


def test_apply_refuses_a_dropped_download_and_changes_nothing(env):
    _home, inst = env
    before = _snapshot(inst)
    data = _build()
    with pytest.raises(LocalmError):
        updater.apply(4, signature=envelope(data, [_PRIV]), installed=inst,
                      download_opener=_download(data[: len(data) // 2]))
    assert _snapshot(inst) == before


def test_apply_refuses_a_signature_mismatch_and_changes_nothing(env):
    _home, inst = env
    before = _snapshot(inst)
    data = _build()
    with pytest.raises(LocalmError):
        updater.apply(4, signature=envelope(_build("0.4.1"), [_PRIV]), installed=inst,
                      download_opener=_download(data))
    assert _snapshot(inst) == before


def test_apply_refuses_a_signed_package_with_a_cut_off_file_and_changes_nothing(env):
    _home, inst = env
    before = _snapshot(inst)
    damaged = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(_build())) as zin, zipfile.ZipFile(damaged, "w") as zout:
        for item in zin.infolist():
            body = zin.read(item.filename)
            if item.filename == "localm/new.py":
                body = body[:1]
            zout.writestr(item, body)
    with pytest.raises(LocalmError) as ei:
        _apply(inst, damaged.getvalue())
    assert "localm/new.py" in ei.value.reason
    assert _snapshot(inst) == before


def test_apply_refuses_an_update_the_disk_cannot_hold_and_changes_nothing(env, monkeypatch):
    _home, inst = env
    before = _snapshot(inst)
    real = shutil.disk_usage

    def tiny(path):
        u = real(path)
        return type(u)(u.total, u.total - 1024, 1024)
    monkeypatch.setattr(shutil, "disk_usage", tiny)
    with pytest.raises(LocalmError) as ei:
        _apply(inst, _build())
    assert "not enough free disk space" in ei.value.summary
    monkeypatch.setattr(shutil, "disk_usage", real)
    assert _snapshot(inst) == before


def test_a_disk_full_during_the_swap_rolls_everything_back(env, monkeypatch):
    _home, inst = env
    before = _snapshot(inst)
    real = shutil.copytree
    fired = []

    def full(src, dst, *a, **k):
        if Path(dst).parent == inst and not fired:
            fired.append(Path(dst))
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(shutil, "copytree", full)
    with pytest.raises(LocalmError) as ei:
        _apply(inst, _build())
    assert ei.value.summary == ("the update could not replace the installed files, so it "
                                "was rolled back")
    assert f"[Errno {errno.ENOSPC}] No space left on device" in ei.value.reason
    assert fired == [inst / "localm"]
    monkeypatch.setattr(shutil, "copytree", real)
    assert _snapshot(inst) == before
    assert not (inst / update_package.NAME).exists()
