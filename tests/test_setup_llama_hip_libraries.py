# SPDX-License-Identifier: AGPL-3.0-or-later
"""AMD's ROCm libraries for the upstream Windows hip build
(localm/setup_llama/hip_libraries.py): when they are added, which files of
which wheels, where they go, every way the supply refuses, the pin table, the
runtime folder handling the added folder, and the loader finding them.

The DLLs are real PE images and the wheels real zip files, built here; only the
network download and the GPU question are replaced."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from localm import pe_deps
from localm import setup_llama as sl
from localm.inference.backends.llamacpp import _loader
from localm.setup_llama import _hip_probe, hip_libraries

ROCM = sl.HIP_LIBRARIES_DIR


def _pe(imports: list[str], version: tuple | None = None) -> bytes:
    """A minimal PE32+ image importing *imports*, with a VS_FIXEDFILEINFO of
    *version* after its import section when given."""
    opt_size, dirs_at, va, raw = 240, 112, 0x1000, 0x400
    names, name_rva = b"", {}
    strings_at = 20 * (len(imports) + 1)
    for n in imports:
        name_rva[n] = va + strings_at + len(names)
        names += n.encode() + b"\0"
    body = b"".join(struct.pack("<IIIII", 0, 0, 0, name_rva[n], 0) for n in imports)
    body += b"\0" * 20 + names
    opt = bytearray(opt_size)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, dirs_at - 4, 16)
    struct.pack_into("<II", opt, dirs_at + 8, va, 20 * (len(imports) + 1))
    section = b".idata\0\0" + struct.pack("<IIII", len(body), va, len(body), raw) + b"\0" * 16
    head = bytearray(0x40)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    image = bytes(head) + b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, opt_size, 0)
    image += bytes(opt) + section
    image = image + b"\0" * (raw - len(image)) + body
    if version is not None:
        a, b, c, d = version
        image += b"\0" * 8 + b"\xbd\x04\xef\xfe" + struct.pack(
            "<IIII", 0x10000, (a << 16) | b, (c << 16) | d, 0)
    return image


# The upstream b12676 Windows hip archive's import graph, as read from its DLLs.
_BUILD = {
    "ggml-hip.dll": ["ggml-base.dll", "hipblas.dll", "amdhip64_7.dll", "MSVCP140.dll",
                     "KERNEL32.dll", "api-ms-win-crt-heap-l1-1-0.dll"],
    "ggml-base.dll": ["KERNEL32.dll"],
    "rocm_kpack.dll": ["KERNEL32.dll"],
    "amd_comgr.dll": ["KERNEL32.dll"],
    "llama.dll": ["ggml.dll", "ggml-base.dll"],
    "ggml.dll": ["ggml-base.dll"],
}
# rocm-sdk-libraries' bin DLLs: the five ggml-hip.dll needs, and others it does not.
_LIBRARIES = {
    "hipblas.dll": ["rocsolver.dll", "rocblas.dll", "KERNEL32.dll"],
    "rocblas.dll": ["libhipblaslt.dll", "amdhip64_7.dll"],
    "rocsolver.dll": ["rocblas.dll", "amdhip64_7.dll"],
    "libhipblaslt.dll": ["amdhip64_7.dll", "origami.dll"],
    "origami.dll": ["amdhip64_7.dll"],
    "MIOpen.dll": ["amdhip64_7.dll", "rocblas.dll"],
    "hipfft.dll": ["amdhip64_7.dll"],
}
_NEEDED = ["hipblas.dll", "libhipblaslt.dll", "origami.dll", "rocblas.dll", "rocsolver.dll"]


def _build(tmp_path: Path, hip_version=(10, 0, 3581, 0), name="lib") -> Path:
    target = tmp_path / name
    target.mkdir()
    for dll, imports in _BUILD.items():
        (target / dll).write_bytes(_pe(imports))
    (target / "amdhip64_7.dll").write_bytes(_pe(["rocm_kpack.dll", "amd_comgr.dll"],
                                                version=hip_version))
    return target


def _zip(path: Path, members: dict) -> str:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Wheels:
    """The two wheels of a ROCm release as local files, the pin table naming
    them, and a download stand-in serving them by URL."""

    def __init__(self, tmp_path: Path, monkeypatch, archs=("gfx1030",), device_extra=None):
        self.dir = tmp_path / "wheels"
        self.dir.mkdir()
        self.lib_name = "rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl"
        lib_members = {f"_rocm_sdk_libraries/bin/{n}": _pe(i) for n, i in _LIBRARIES.items()}
        lib_members["_rocm_sdk_libraries/__init__.py"] = b""
        lib_members["_rocm_sdk_libraries/bin/hipdnn_plugins/x.dll"] = _pe([])
        lib_members["rocm_sdk_libraries-10.0.0.dist-info/METADATA"] = b"Name: rocm-sdk-libraries\n"
        lib_sha = _zip(self.dir / self.lib_name, lib_members)
        devices = {}
        for arch in archs:
            name = f"rocm_sdk_device_{arch}-10.0.0-py3-none-win_amd64.whl"
            members = {
                f"_rocm_sdk_libraries/.kpack/blas_lib_{arch}.kpack": b"KPACK" + arch.encode(),
                f"_rocm_sdk_libraries/bin/rocblas/library/TensileLibrary_lazy_{arch}.dat": b"T",
                f"_rocm_sdk_libraries/bin/rocblas/library/Kernels.so-000-{arch}.hsaco": b"H",
                "_rocm_sdk_libraries/__init__.py": b"",
                f"_rocm_sdk_libraries/.devel_links/{arch}.json": b"{}",
                f"rocm_sdk_device_{arch}-10.0.0.dist-info/METADATA": b"Name: x\n",
                **(device_extra or {}),
            }
            sha = _zip(self.dir / name, members)
            devices[arch] = ((self.dir / name).stat().st_size, sha)
        self.table = {"10.0": {"version": "10.0.0",
                               "libraries": (self.lib_name,
                                             (self.dir / self.lib_name).stat().st_size, lib_sha),
                               "devices": devices}}
        self.urls: list[str] = []
        monkeypatch.setattr(sl, "_HIP_SDK_WHEELS", self.table)
        monkeypatch.setattr(sl, "_download", self._download)
        monkeypatch.setattr(sl, "hip_gpu_archs", lambda target: list(archs))

    def _download(self, url: str, dest: Path):
        self.urls.append(url)
        shutil.copy2(self.dir / url.rsplit("/", 1)[-1], dest)
        return None


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
#  The supply                                                                 #
# --------------------------------------------------------------------------- #

def test_adds_exactly_the_libraries_ggml_hip_needs_and_the_gpu_kernels(tmp_path, monkeypatch):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    assert sl.needs_hip_libraries(target)

    added = sl.supply_hip_libraries(target)

    assert added == _NEEDED
    assert _tree(target / ROCM) == sorted(
        [f"bin/{n}" for n in _NEEDED]
        + [".kpack/blas_lib_gfx1030.kpack", sl.HIP_LIBRARIES_MARKER,
           "bin/rocblas/library/Kernels.so-000-gfx1030.hsaco",
           "bin/rocblas/library/TensileLibrary_lazy_gfx1030.dat"])
    assert (target / ROCM / "bin" / "hipblas.dll").read_bytes() == _pe(_LIBRARIES["hipblas.dll"])
    record = json.loads((target / ROCM / sl.HIP_LIBRARIES_MARKER).read_text(encoding="utf-8"))
    assert record["rocm"] == "10.0.0" and record["hip_runtime"] == "10.0.3581.0"
    assert record["archs"] == ["gfx1030"]
    assert set(record["wheels"]) == {wheels.lib_name,
                                     "rocm_sdk_device_gfx1030-10.0.0-py3-none-win_amd64.whl"}
    assert wheels.urls == [
        f"{sl._HIP_SDK_INDEX}rocm-sdk-libraries/{wheels.lib_name}",
        f"{sl._HIP_SDK_INDEX}rocm-sdk-device-gfx1030/"
        "rocm_sdk_device_gfx1030-10.0.0-py3-none-win_amd64.whl"]
    assert not sl.needs_hip_libraries(target)
    # The archive's own files are left as they were.
    assert (target / "amdhip64_7.dll").is_file() and not (target / "hipblas.dll").exists()


def test_every_gpu_gets_its_own_kernels(tmp_path, monkeypatch):
    target = _build(tmp_path)
    _Wheels(tmp_path, monkeypatch, archs=("gfx1030", "gfx1100"))
    assert sl.supply_hip_libraries(target) == _NEEDED
    packs = sorted(p.name for p in (target / ROCM / ".kpack").iterdir())
    assert packs == ["blas_lib_gfx1030.kpack", "blas_lib_gfx1100.kpack"]


def test_a_gpu_without_pinned_kernels_is_named_and_the_others_still_get_theirs(
        tmp_path, monkeypatch, capsys):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    monkeypatch.setattr(sl, "hip_gpu_archs", lambda t: ["gfx906", "gfx1030"])
    assert sl.supply_hip_libraries(target) == _NEEDED
    out = capsys.readouterr().out
    assert "gfx906" in out and "runs on gfx1030 only" in out
    assert not any("gfx906" in u for u in wheels.urls)


@pytest.mark.parametrize("archs, said", [
    (["gfx906"], "localm pins no ROCm 10.0 kernels for gfx906"),
    ([], "found no AMD GPU"),
])
def test_no_usable_gpu_adds_nothing(tmp_path, monkeypatch, capsys, archs, said):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    monkeypatch.setattr(sl, "hip_gpu_archs", lambda t: archs)
    assert sl.supply_hip_libraries(target) is None
    assert said in " ".join(capsys.readouterr().out.split())
    assert wheels.urls == [] and not (target / ROCM).exists()


def test_a_hip_runtime_of_a_release_without_pinned_libraries_is_named(
        tmp_path, monkeypatch, capsys):
    target = _build(tmp_path, hip_version=(10, 1, 3700, 0))
    wheels = _Wheels(tmp_path, monkeypatch)
    assert sl.supply_hip_libraries(target) is None
    out = " ".join(capsys.readouterr().out.split())
    assert "HIP runtime of ROCm 10.1" in out and "10.1.3700.0" in out
    assert "for ROCm 10.0 only" in out and "--tag default" in out
    assert wheels.urls == [] and not (target / ROCM).exists()


def test_a_build_without_a_readable_hip_runtime_version_adds_nothing(tmp_path, monkeypatch,
                                                                    capsys):
    target = _build(tmp_path)
    (target / "amdhip64_7.dll").write_bytes(_pe(["KERNEL32.dll"]))
    wheels = _Wheels(tmp_path, monkeypatch)
    assert sl.supply_hip_libraries(target) is None
    assert "matching ROCm release is unknown" in " ".join(capsys.readouterr().out.split())
    assert wheels.urls == []


@pytest.mark.parametrize("which", ["libraries", "device"])
def test_a_wheel_that_does_not_match_its_pin_leaves_no_partial_set(
        tmp_path, monkeypatch, capsys, which):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    spec = wheels.table["10.0"]
    if which == "libraries":
        name, size, _sha = spec["libraries"]
        spec["libraries"] = (name, size, "0" * 64)
    else:
        size, _sha = spec["devices"]["gfx1030"]
        spec["devices"]["gfx1030"] = (size, "0" * 64)
    assert sl.supply_hip_libraries(target) is None
    out = " ".join(capsys.readouterr().out.split())
    assert "were not added" in out and "sha256" in out.lower()
    assert not (target / ROCM).exists()


def test_a_failed_download_leaves_no_partial_set(tmp_path, monkeypatch, capsys):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)

    def refuse(url, dest):
        wheels.urls.append(url)
        if "device" in url:
            raise OSError("connection reset")
        shutil.copy2(wheels.dir / url.rsplit("/", 1)[-1], dest)

    monkeypatch.setattr(sl, "_download", refuse)
    assert sl.supply_hip_libraries(target) is None
    assert "connection reset" in capsys.readouterr().out
    assert not (target / ROCM).exists()


def test_a_wheel_member_that_escapes_is_refused(tmp_path, monkeypatch, capsys):
    target = _build(tmp_path)
    _Wheels(tmp_path, monkeypatch, device_extra={"_rocm_sdk_libraries/../../evil.dll": b"x"})
    assert sl.supply_hip_libraries(target) is None
    assert "unsafe path" in capsys.readouterr().out
    assert not (tmp_path / "evil.dll").exists() and not (target / ROCM).exists()


@pytest.mark.parametrize("has", ["hipblas.dll", f"{ROCM}/bin/hipblas.dll", "no ggml-hip"])
def test_a_build_that_needs_nothing_downloads_nothing(tmp_path, monkeypatch, has):
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    if has == "no ggml-hip":
        (target / "ggml-hip.dll").unlink()
    else:
        (target / has).parent.mkdir(parents=True, exist_ok=True)
        (target / has).write_bytes(_pe([]))
    assert not sl.needs_hip_libraries(target)
    assert sl.supply_hip_libraries(target) is None
    assert wheels.urls == []


# --------------------------------------------------------------------------- #
#  The GPU question                                                           #
# --------------------------------------------------------------------------- #

def _props(name: bytes, arch: bytes) -> bytes:
    buf = bytearray(_hip_probe._PROPS_BYTES)
    buf[:len(name)] = name
    buf[1100:1100 + len(arch)] = arch
    return bytes(buf)


@pytest.mark.parametrize("name, arch, want", [
    (b"AMD Radeon RX 6900 XT", b"gfx1030", "gfx1030"),
    (b"AMD Instinct MI210", b"gfx90a:sramecc+:xnack-", "gfx90a"),
    (b"gfx1036 named card", b"gfx1100", "gfx1100"),
    (b"AMD Radeon RX 6900 XT", b"", None),
])
def test_the_architecture_is_read_after_the_name_field(name, arch, want):
    assert _hip_probe.arch_from_properties(_props(name, arch)) == want


def _probe_stand_in(tmp_path: Path, monkeypatch, body: str) -> None:
    script = tmp_path / "probe.py"
    script.write_text("import sys\n" + body, encoding="utf-8")
    monkeypatch.setattr(hip_libraries, "_PROBE", script)


def test_the_gpu_question_runs_in_a_child_and_lists_each_architecture_once(tmp_path,
                                                                         monkeypatch):
    _probe_stand_in(tmp_path, monkeypatch,
                    "assert sys.argv[1].endswith('lib')\n"
                    "print('gfx1030'); print('gfx1030'); print('gfx1100')\n")
    assert hip_libraries.hip_gpu_archs(tmp_path / "lib") == ["gfx1030", "gfx1100"]


def test_a_gpu_question_that_fails_says_why(tmp_path, monkeypatch, capsys):
    _probe_stand_in(tmp_path, monkeypatch,
                    "sys.stderr.write('amdhip64_7.dll did not load: [WinError 126]')\n"
                    "sys.exit(2)\n")
    with pytest.raises(sl.HipLibrariesError, match="amdhip64_7.dll did not load"):
        hip_libraries.hip_gpu_archs(tmp_path)
    target = _build(tmp_path)
    wheels = _Wheels(tmp_path, monkeypatch)
    monkeypatch.setattr(sl, "hip_gpu_archs", hip_libraries.hip_gpu_archs)
    assert sl.supply_hip_libraries(target) is None
    assert "amdhip64_7.dll did not load" in capsys.readouterr().out
    assert wheels.urls == []


# --------------------------------------------------------------------------- #
#  The pin table                                                              #
# --------------------------------------------------------------------------- #

def test_the_pinned_hip_build_has_its_rocm_libraries_pinned():
    pat = re.compile(rf"^llama-{re.escape(sl._PINNED_TAG)}-bin-win-rocm-(\d+\.\d+)-x64\.zip$")
    releases = {m.group(1) for n in sl._PINNED_FALLBACK_SHA256 if (m := pat.match(n))}
    assert releases, "the pinned tag has no Windows hip asset"
    assert releases <= set(sl._HIP_SDK_WHEELS)


def test_every_pinned_wheel_names_its_release_and_has_a_checksum():
    for release, spec in sl._HIP_SDK_WHEELS.items():
        assert spec["version"].startswith(release + ".")
        name, size, sha = spec["libraries"]
        assert name == f"rocm_sdk_libraries-{spec['version']}-py3-none-win_amd64.whl"
        assert size > 0 and re.fullmatch(r"[0-9a-f]{64}", sha)
        assert spec["devices"]
        for arch, (size, sha) in spec["devices"].items():
            assert re.fullmatch(r"gfx[0-9a-f]+", arch)
            assert size > 0 and re.fullmatch(r"[0-9a-f]{64}", sha)


# --------------------------------------------------------------------------- #
#  The runtime folder                                                         #
# --------------------------------------------------------------------------- #

def test_the_added_folder_is_cleared_and_checked_with_the_runtime(tmp_path, monkeypatch):
    target = _build(tmp_path)
    _Wheels(tmp_path, monkeypatch)
    sl.supply_hip_libraries(target)
    added = {p for p in (target / ROCM).rglob("*") if p.is_file()}
    assert added <= set(sl._clearable_files(target))
    assert ROCM in sl.RUNTIME_SUBDIRS
    assert sl._clear_target(target) == []
    assert not (target / ROCM).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the hip libraries are added on Windows")
def test_a_placed_hip_build_gets_its_libraries_through_the_one_placement_path(
        tmp_path, monkeypatch):
    source = _build(tmp_path, name="archive")
    wheels = _Wheels(tmp_path, monkeypatch)
    monkeypatch.setattr(sl, "offer_vc_runtime", lambda target: True)
    target = tmp_path / "runtime"
    target.mkdir()

    sl._replace_target(target, lambda staging: sl._copy_binaries(source, staging))

    assert (target / ROCM / "bin" / "hipblas.dll").is_file()
    assert (target / ROCM / ".kpack" / "blas_lib_gfx1030.kpack").is_file()
    assert (target / "ggml-hip.dll").is_file()
    assert len(wheels.urls) == 2


def test_kernel_data_checks_cover_the_added_folder(tmp_path, monkeypatch):
    target = _build(tmp_path)
    _Wheels(tmp_path, monkeypatch)
    sl.supply_hip_libraries(target)
    assert sl.blas_kernel_problems(target) == []
    shutil.rmtree(target / ROCM / ".kpack")
    assert sl.blas_kernel_problems(target) == [
        f"rocm_kpack is installed but its {ROCM}/.kpack/ kernel pack directory is missing "
        "or empty"]
    shutil.rmtree(target / ROCM / "bin" / "rocblas")
    assert f"rocblas is installed but its {ROCM}/bin/rocblas/ kernel directory is missing " \
           "entirely" in sl.blas_kernel_problems(target)


def _confirm_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "confirm_llama_runtime.py"
    spec = importlib.util.spec_from_file_location("confirm_llama_runtime_hip", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows hip build holds llama.dll")
def test_the_runtime_confirmation_places_the_hip_build_as_an_install_does(tmp_path,
                                                                         monkeypatch):
    confirm = _confirm_script()
    source = _build(tmp_path, name="archive")
    wheels = _Wheels(tmp_path, monkeypatch)

    def provision(backend, dest, sha256, with_cudart, cuda_line, tag):
        for f in source.iterdir():
            shutil.copy2(f, dest / f.name)

    monkeypatch.setattr(sl, "_provision_backend", provision)
    lib, fetched, err, kind = confirm._place_build("hip", "b1", tmp_path / "d")
    assert err == "" and kind == "" and Path(lib).name == "llama.dll"
    assert (tmp_path / "d" / ROCM / "bin" / "hipblas.dll").is_file()
    spec = wheels.table["10.0"]
    assert fetched == [
        {"asset": "rocm_sdk_device_gfx1030-10.0.0-py3-none-win_amd64.whl",
         "sha256": spec["devices"]["gfx1030"][1]},
        {"asset": spec["libraries"][0], "sha256": spec["libraries"][2]}]


# --------------------------------------------------------------------------- #
#  The loader                                                                 #
# --------------------------------------------------------------------------- #

def test_the_loader_finds_the_added_folder(tmp_path):
    target = _build(tmp_path)
    assert _loader.hip_library_dirs(target) == []
    (target / ROCM / "bin").mkdir(parents=True)
    assert _loader.hip_library_dirs(target) == [target / ROCM / "bin"]
    assert _loader.hip_library_dirs(None) == []


def test_the_loader_loads_the_build_s_hip_runtime_before_hipblas(tmp_path, monkeypatch):
    target = _build(tmp_path)
    hip_bin = target / ROCM / "bin"
    hip_bin.mkdir(parents=True)
    (hip_bin / "hipblas.dll").write_bytes(_pe([]))
    loaded: list[Path] = []
    monkeypatch.setattr(_loader, "_preload", loaded.append)
    _loader._preload_hip_libraries(target, [hip_bin])
    assert loaded == [target / "amdhip64_7.dll", hip_bin / "hipblas.dll"]
    loaded.clear()
    _loader._preload_hip_libraries(target, [])
    assert loaded == []


@pytest.mark.parametrize("own_data, expect_set", [(True, False), (False, True)])
def test_a_runtime_with_its_own_rocblas_data_keeps_it(tmp_path, monkeypatch, own_data,
                                                      expect_set):
    venv_bin = tmp_path / "venv" / "_rocm_sdk_libraries" / "bin"
    (venv_bin / "rocblas" / "library").mkdir(parents=True)
    monkeypatch.setattr(_loader, "rocm_runtime_dirs", lambda: [venv_bin])
    monkeypatch.delenv("ROCBLAS_TENSILE_LIBPATH", raising=False)
    hip_bin = tmp_path / "lib" / ROCM / "bin"
    hip_bin.mkdir(parents=True)
    (hip_bin / "rocblas.dll").write_bytes(_pe([]))
    if own_data:
        (hip_bin / "rocblas" / "library").mkdir(parents=True)
    _loader._ensure_rocblas_tensile_path([tmp_path / "lib", hip_bin])
    got = os.environ.pop("ROCBLAS_TENSILE_LIBPATH", None)
    assert (got == str(venv_bin / "rocblas" / "library")) is expect_set
    assert (got is None) is not expect_set


@pytest.mark.skipif(sys.platform != "win32", reason="the missing-DLL clause is Windows only")
def test_a_load_failure_looks_in_the_added_folder_too(tmp_path, monkeypatch):
    from localm import vc_runtime
    target = _build(tmp_path)
    system = tmp_path / "system32"
    system.mkdir()
    for name in ("KERNEL32.dll", "MSVCP140.dll"):
        (system / name).write_bytes(_pe([]))
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: system)
    monkeypatch.setattr(_loader, "rocm_runtime_dirs", lambda: [])
    # A DLL of that name loaded into this process by another test counts as found.
    monkeypatch.setattr(pe_deps, "_is_loaded", lambda name: False)
    clause = _loader.missing_dll_clause(target, [target / "ggml-hip.dll"])
    assert clause is not None and clause.startswith("hipblas.dll is missing")
    hip_bin = target / ROCM / "bin"
    hip_bin.mkdir(parents=True)
    for name, imports in _LIBRARIES.items():
        (hip_bin / name).write_bytes(_pe(imports))
    assert _loader.missing_dll_clause(target, [target / "ggml-hip.dll"]) is None
    assert pe_deps.imported_dlls(target / "ggml-hip.dll")[1] == "hipblas.dll"
