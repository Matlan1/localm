# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Microsoft Visual C++ runtime placed beside a native runtime on Windows
(localm.vc_runtime): unpacking the redistributable's WiX bundle and its
cabinets, choosing which DLLs a runtime folder needs, the pinned checksum, the
system copy that is new enough, and the load error that names a missing DLL.

The bundle, cabinets and DLLs are real files in their real formats, built here."""

from __future__ import annotations

import ast
import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from localm import pe_deps, vc_runtime
from localm import setup_llama as sl
from localm.setup_llama import load_probe

VERSION = vc_runtime.REDIST_VERSION


# --------------------------------------------------------------------------- #
#  Builders                                                                   #
# --------------------------------------------------------------------------- #

def _pe(imports: list[str], version: tuple | None = None, machine: int = 0x8664) -> bytes:
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
    image = bytes(head) + b"PE\0\0" + struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, opt_size, 0)
    image += bytes(opt) + section
    image = image + b"\0" * (raw - len(image)) + body
    if version is not None:
        a, b, c, d = version
        image += b"\0" * 8 + b"\xbd\x04\xef\xfe" + struct.pack(
            "<IIII", 0x10000, (a << 16) | b, (c << 16) | d, 0)
    return image


def _cab(files: dict[str, bytes], *, mszip: bool = True, block: int = 700) -> bytes:
    """A single-folder cabinet holding *files*, MSZIP blocks of *block* bytes
    (each block's deflate stream primed with the output before it) or stored."""
    blob = b"".join(files.values())
    data, out = b"", b""
    for i in range(0, len(blob), block):
        chunk = blob[i:i + block]
        if mszip:
            c = zlib.compressobj(9, zlib.DEFLATED, -15, zdict=out[-32768:] or None) \
                if out else zlib.compressobj(9, zlib.DEFLATED, -15)
            payload = b"CK" + c.compress(chunk) + c.flush()
        else:
            payload = chunk
        data += struct.pack("<IHH", 0, len(payload), len(chunk)) + payload
        out += chunk
    table, offset = b"", 0
    for name, content in files.items():
        table += struct.pack("<IIHHHH", len(content), offset, 0, 0, 0, 0x20) + name.encode() + b"\0"
        offset += len(content)
    n_blocks = -(-len(blob) // block)
    start = 44 + len(table)
    total = start + len(data)
    head = (b"MSCF" + struct.pack("<IIIII", 0, total, 0, 44, 0)
            + struct.pack("<BBHHHHH", 3, 1, 1, len(files), 0, 0, 0))
    folder = struct.pack("<IHH", start, n_blocks, 1 if mszip else 0)
    return head + folder + table + data


def _bundle(containers: list[bytes], signature: bytes = b"S" * 40) -> bytes:
    """A WiX Burn bundle: a stub with a .wixburn section, the UX container, the
    signature, then the attached containers."""
    stub_size = 0x400
    head = bytearray(0x40)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    image = bytes(head) + b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0, 0)
    image += b".wixburn" + struct.pack("<IIII", 64, 0x1000, 0x200, 0x200) + b"\0" * 16
    image += b"\0" * (0x200 - len(image))
    sig_offset = stub_size + len(containers[0])
    header = struct.pack("<II", 0x00F14300, 2) + b"\0" * 16 + struct.pack(
        "<IIIIII", stub_size, 0, sig_offset, len(signature), 1, len(containers))
    header += struct.pack("<" + "I" * len(containers), *(len(c) for c in containers))
    image += header + b"\0" * (stub_size - len(image) - len(header))
    return image + containers[0] + signature + b"".join(containers[1:])


def _redist(suffix: str = "amd64", version: tuple = VERSION) -> tuple[bytes, dict]:
    dlls = {
        "msvcp140.dll": _pe(["VCRUNTIME140.dll", "KERNEL32.dll"], version),
        "vcruntime140.dll": _pe(["KERNEL32.dll"], version),
        "vcruntime140_1.dll": _pe(["VCRUNTIME140.dll"], version),
        "vcomp140.dll": _pe(["KERNEL32.dll"], version),
    }
    runtime_cab = _cab({f"{n}_{suffix}": b for n, b in dlls.items()})
    other_cab = _cab({"readme_amd64": b"not a dll"}, mszip=False)
    attached = _cab({"a0": b"\xd0\xcf\x11\xe0 an msi", "a1": other_cab, "a2": runtime_cab},
                    mszip=False)
    ux = _cab({"0": b"<BurnManifest/>"})
    return _bundle([ux, attached]), dlls


@pytest.fixture
def pinned(monkeypatch):
    """Pin the amd64 package to a bundle built here and serve it by copy."""
    exe, dlls = _redist()
    monkeypatch.setitem(vc_runtime._REDIST, "amd64", (
        "https://example.invalid/VC_redist.x64.exe", hashlib.sha256(exe).hexdigest(),
        len(exe), "amd64"))
    calls: list[str] = []

    def download(url, dest):
        calls.append(url)
        Path(dest).write_bytes(exe)
    return download, calls, dlls


def _runtime_folder(tmp_path: Path, imports: list[str]) -> Path:
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "llama.dll").write_bytes(_pe(imports))
    (folder / "ggml-base.dll").write_bytes(_pe(["MSVCP140.dll", "KERNEL32.dll"]))
    return folder


# --------------------------------------------------------------------------- #
#  Unpacking                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mszip", [True, False])
def test_cabinet_files_round_trip_across_blocks(mszip):
    files = {"a.dll_amd64": bytes(range(256)) * 9, "b.txt": b"hello" * 300}
    assert vc_runtime.cabinet_files(_cab(files, mszip=mszip, block=500)) == files


def test_cabinet_with_another_compression_is_refused():
    cab = bytearray(_cab({"x": b"y" * 10}))
    struct.pack_into("<H", cab, 36 + 6, 3)
    with pytest.raises(vc_runtime.RedistError, match="compression type 3"):
        vc_runtime.cabinet_files(bytes(cab))


def test_runtime_files_finds_the_runtime_cabinet_for_the_suffix():
    exe, dlls = _redist()
    assert vc_runtime.runtime_files(exe, "amd64") == dlls
    with pytest.raises(vc_runtime.RedistError, match="arm64"):
        vc_runtime.runtime_files(exe, "arm64")


def test_arm64_takes_system_named_files_and_skips_other_machines():
    arm = 0xAA64
    files = {
        "msvcp140.dll_arm64": _pe(["VCRUNTIME140.dll"], VERSION, arm),
        "vcomp140.dll_system_arm64": _pe(["KERNEL32.dll"], VERSION, arm),
        "vcruntime140_1.dll_arm64": _pe(["KERNEL32.dll"], VERSION, 0x8664),
    }
    exe = _bundle([_cab({"0": b"m"}), _cab({"a0": _cab(files)}, mszip=False)])
    found = vc_runtime.runtime_files(exe, "arm64")
    assert sorted(found) == ["msvcp140.dll", "vcomp140.dll"]
    assert found["vcomp140.dll"] == files["vcomp140.dll_system_arm64"]


def test_a_file_that_is_not_a_bundle_is_refused():
    with pytest.raises(vc_runtime.RedistError):
        vc_runtime.runtime_files(_pe(["KERNEL32.dll"]), "amd64")


# --------------------------------------------------------------------------- #
#  Placing                                                                    #
# --------------------------------------------------------------------------- #

def test_places_what_the_folder_imports_and_its_runtime_dependencies(tmp_path, pinned):
    download, calls, dlls = pinned
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140_1.dll", "KERNEL32.dll"])
    placed = vc_runtime.place_app_local(folder, download, system_dir=None, on_windows=True)
    # vcruntime140.dll is not imported by the folder; msvcp140.dll needs it.
    assert placed == ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
    assert calls == ["https://example.invalid/VC_redist.x64.exe"]
    for name in placed:
        assert (folder / name).read_bytes() == dlls[name]
    assert not (folder / "vcomp140.dll").exists()
    notice = (folder / vc_runtime.LICENSE_NAME).read_text(encoding="utf-8")
    assert "msvcp140.dll" in notice and "aka.ms/VCRedistLicense" in notice
    assert vc_runtime.needed(folder) == set()
    assert sorted(p.name for p in folder.iterdir() if p.name.startswith(".vcredist")) == []
    assert vc_runtime.place_app_local(folder, download, system_dir=None, on_windows=True) == []
    assert len(calls) == 1


def test_nothing_is_fetched_when_nothing_imports_the_runtime(tmp_path, pinned):
    download, calls, _ = pinned
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "llama.dll").write_bytes(_pe(["KERNEL32.dll", "vulkan-1.dll"]))
    (folder / "notes.txt").write_text("x")
    assert vc_runtime.place_app_local(folder, download, system_dir=None, on_windows=True) == []
    assert calls == []


def test_a_system_copy_at_the_pinned_version_is_used_instead(tmp_path, pinned):
    download, calls, _ = pinned
    system = tmp_path / "system"
    system.mkdir()
    for name in ("msvcp140.dll", "vcruntime140.dll"):
        (system / name).write_bytes(_pe(["KERNEL32.dll"], VERSION))
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140.dll"])
    assert vc_runtime.place_app_local(folder, download, system_dir=system, on_windows=True) == []
    assert calls == []


def test_an_older_system_copy_gets_an_app_local_one(tmp_path, pinned):
    download, calls, _ = pinned
    system = tmp_path / "system"
    system.mkdir()
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 29, 30133, 0)))
    (system / "vcruntime140.dll").write_bytes(_pe(["KERNEL32.dll"], VERSION))
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140.dll"])
    placed = vc_runtime.place_app_local(folder, download, system_dir=system, on_windows=True)
    assert placed == ["msvcp140.dll"]


def test_a_package_that_does_not_match_its_pin_changes_nothing(tmp_path, pinned, monkeypatch):
    download, _, _ = pinned
    url, sha, size, suffix = vc_runtime._REDIST["amd64"]
    monkeypatch.setitem(vc_runtime._REDIST, "amd64", (url, "0" * 64, size, suffix))
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    before = sorted(p.name for p in folder.iterdir())
    with pytest.raises(vc_runtime.RedistError, match="sha256"):
        vc_runtime.place_app_local(folder, download, system_dir=None, on_windows=True)
    assert sorted(p.name for p in folder.iterdir()) == before


def test_a_dll_of_another_version_is_refused(tmp_path, monkeypatch):
    exe, _ = _redist(version=(14, 40, 1, 0))
    monkeypatch.setitem(vc_runtime._REDIST, "amd64", (
        "https://example.invalid/x.exe", hashlib.sha256(exe).hexdigest(), len(exe), "amd64"))
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    with pytest.raises(vc_runtime.RedistError, match="version"):
        vc_runtime.place_app_local(folder, lambda u, d: Path(d).write_bytes(exe),
                                   system_dir=None, on_windows=True)
    assert not (folder / "msvcp140.dll").exists()


def test_off_windows_it_does_nothing(tmp_path, pinned):
    download, calls, _ = pinned
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    assert vc_runtime.place_app_local(folder, download, on_windows=False) == []
    assert calls == []


@pytest.mark.parametrize("machine,key", [("AMD64", "amd64"), ("x86_64", "amd64"),
                                         ("ARM64", "arm64"), ("x86", None)])
def test_redist_machine(machine, key):
    assert vc_runtime.redist_machine(machine) == key


def test_every_pin_is_microsofts_versioned_download():
    for url, sha, size, _suffix in vc_runtime._REDIST.values():
        assert url.startswith("https://download.visualstudio.microsoft.com/download/pr/")
        assert sha.upper() in url and len(sha) == 64 and size > 1_000_000


# --------------------------------------------------------------------------- #
#  setup-llama: one placement path                                            #
# --------------------------------------------------------------------------- #

def test_bundling_on_windows_places_the_runtime_through_setup_llamas_download(
        tmp_path, pinned, monkeypatch):
    download, calls, _ = pinned
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)
    monkeypatch.setattr(sl, "_download", download)
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    sl._bundle_missing_native_deps(folder)
    assert (folder / "msvcp140.dll").is_file() and calls


def test_bundling_on_windows_never_raises(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)

    def broken(url, dest):
        raise OSError("network down")
    monkeypatch.setattr(sl, "_download", broken)
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    sl._bundle_missing_native_deps(folder)
    assert not (folder / "msvcp140.dll").exists()
    assert any("Visual C++ runtime" in r.message for r in caplog.records)


def test_replace_target_bundles_into_staging_before_the_swap(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    seen = []
    monkeypatch.setattr(sl, "_bundle_missing_native_deps",
                        lambda d: seen.append((d.name, sorted(p.name for p in d.iterdir()))))

    def fill(staging):
        (staging / "llama.dll").write_bytes(b"x")
        return "tag"
    assert sl._replace_target(target, fill) == "tag"
    assert seen == [(sl._staging_path(target).name, ["llama.dll"])]
    assert (target / "llama.dll").is_file()


def test_only_replace_target_calls_the_bundler():
    """Every way of placing a runtime goes through _replace_target, so the
    bundler is called there and nowhere else in the package."""
    root = Path(sl.__file__).parent
    callers = []
    for f in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_bundle_missing_native_deps"):
                callers.append(f.name)
    assert callers == ["runtime_dir.py"]


# --------------------------------------------------------------------------- #
#  Naming a missing DLL                                                       #
# --------------------------------------------------------------------------- #

def test_unresolved_imports_follows_dependencies_in_the_search_dirs(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "llama.dll").write_bytes(_pe(["ggml.dll", "MSVCP140.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]))
    (lib / "ggml.dll").write_bytes(_pe(["vulkan-1.dll", "KERNEL32.dll"]))
    system = tmp_path / "sys"
    system.mkdir()
    (system / "kernel32.dll").write_bytes(b"")
    missing = pe_deps.unresolved_imports([lib / "llama.dll"], [lib, system],
                                         loaded=lambda n: False)
    assert missing == {"msvcp140.dll": ["llama.dll"], "vulkan-1.dll": ["ggml.dll"]}
    assert pe_deps.unresolved_imports([lib / "llama.dll"], [lib, system],
                                      loaded=lambda n: True) == {}


def test_the_clause_names_the_runtime_and_setup_reads_it_back():
    clause = vc_runtime.describe_missing(
        {"msvcp140.dll": ["llama.dll", "ggml.dll"], "vulkan-1.dll": ["ggml-vulkan.dll"]})
    assert clause == ("msvcp140.dll is missing (the Microsoft Visual C++ runtime; needed by "
                      "ggml.dll, llama.dll); vulkan-1.dll is missing (needed by ggml-vulkan.dll)")
    line = f"RuntimeError: Failed to load llama.dll from X: {clause}. [WinError 126] ..."
    named = load_probe._name_missing_shared_lib(line) or ""
    assert named.startswith("msvcp140.dll (the Microsoft Visual C++ runtime)")
    assert vc_runtime.redist_url() in named
    other = load_probe._name_missing_shared_lib(
        "RuntimeError: Failed to load llama.dll from X: vulkan-1.dll is missing "
        "(needed by ggml-vulkan.dll). [WinError 126]") or ""
    assert other.startswith("vulkan-1.dll (needed by ggml-vulkan.dll)")
    assert vc_runtime.describe_missing({}) is None


def test_loader_names_the_missing_runtime_dll(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    lib = _runtime_folder(tmp_path, ["MSVCP140.dll", "ggml-base.dll"])
    monkeypatch.setattr(_loader.sys, "platform", "win32")
    monkeypatch.setattr(_loader, "rocm_runtime_dirs", lambda: [])
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)
    monkeypatch.setattr(pe_deps, "_is_loaded", lambda name: name.lower() == "kernel32.dll")
    monkeypatch.setattr(_loader.sys, "base_prefix", str(tmp_path / "py"))
    monkeypatch.setattr(_loader.sys, "executable", str(tmp_path / "py" / "python.exe"))
    clause = _loader.missing_dll_clause(lib, [lib / "llama.dll"])
    assert clause == ("msvcp140.dll is missing (the Microsoft Visual C++ runtime; needed by "
                      "ggml-base.dll, llama.dll)")
    (lib / "msvcp140.dll").write_bytes(_pe(["VCRUNTIME140.dll", "KERNEL32.dll"]))
    assert _loader.missing_dll_clause(lib, [lib / "llama.dll"]) == (
        "vcruntime140.dll is missing (the Microsoft Visual C++ runtime; needed by "
        "msvcp140.dll)")
    (lib / "vcruntime140.dll").write_bytes(_pe(["KERNEL32.dll"]))
    assert _loader.missing_dll_clause(lib, [lib / "llama.dll"]) is None


def test_sdcpp_load_error_names_the_missing_runtime_dll(tmp_path, monkeypatch):
    from localm.media.sdcpp import _binding, _child
    rt = tmp_path / "sd"
    rt.mkdir()
    monkeypatch.setattr(vc_runtime.sys, "platform", "win32")
    (rt / _binding.lib_filename()).write_bytes(_pe(["MSVCP140.dll", "KERNEL32.dll"]))
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)
    monkeypatch.setattr(pe_deps, "_is_loaded", lambda name: name.lower() == "kernel32.dll")
    monkeypatch.setattr(vc_runtime.sys, "base_prefix", str(tmp_path / "py"))
    monkeypatch.setattr(vc_runtime.sys, "executable", str(tmp_path / "py" / "python.exe"))
    monkeypatch.setattr(_child, "_prepare_environment", lambda d: None)

    def refuse(runtime_dir, extra):
        raise OSError("[WinError 126] The specified module could not be found")
    monkeypatch.setattr(_binding, "load_library", refuse)
    with pytest.raises(RuntimeError) as e:
        _child._load_lib({"runtime_dir": str(rt), "extra_dirs": []})
    message = str(e.value)
    assert "msvcp140.dll is missing (the Microsoft Visual C++ runtime" in message
    assert vc_runtime.redist_url() in message and "setup-sdcpp --force" in message

