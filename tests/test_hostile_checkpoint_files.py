# SPDX-License-Identifier: AGPL-3.0-or-later
"""A crafted ``.pth`` / ``.pt`` / zip model file cannot crash, hang or swell the
process reading it, and a real ``torch.save`` file still reads.

Every hostile file runs in a child process with a timeout and a peak memory
reading: a native stack overflow or a hang inside one C call would otherwise
take the test process with it."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import localm
from localm.media.sdcpp import families
from localm.media.sdcpp.families import UNKNOWN, classify_tensors, inspect_file
from localm.model_manager import ckpt_tensors, registry, sam
from localm.model_manager.ckpt_tensors import read_checkpoint_tensors
from localm.model_manager.gguf import gguf_media_component_type

REPO = Path(localm.__file__).resolve().parent.parent
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
CHILD_TIMEOUT = 60
GROWTH_LIMIT_MB = 100
BOMB_BYTES = 200 << 20

ESRGAN = {"conv_first.weight": (4, 3, 3, 3), "conv_first.bias": (4,),
          "body.0.rdb1.conv1.weight": (4, 4, 3, 3), "conv_last.weight": (3, 4, 3, 3)}
VAE = {"decoder.conv_in.weight": (8, 4, 3, 3), "encoder.conv_in.weight": (8, 3, 3, 3)}
HUGE = 3 * 10 ** 400
SRVGG_OVERFLOW = {"body.0.weight": (4, 3, 3, 3), "body.2.weight": (4, 4, 3, 3),
                  "body.4.weight": (HUGE, 64, 3, 3)}
LONG_DIGITS = {"body." + "1" * 5000 + ".weight": (4, 3, 3, 3)}

_CHILD = r"""
import json, sys
from pathlib import Path

def peak_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("peak", ctypes.c_size_t), ("rest", ctypes.c_size_t * 7)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            raise OSError("no memory counters")
        return counters.peak
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024

from localm.media.sdcpp.families import inspect_file
from localm.model_manager import add_local
from localm.model_manager.ckpt_tensors import read_checkpoint_tensors
from localm.model_manager.sam import is_sam_checkpoint
from localm.rag import extract

def read_docx(p):
    try:
        return extract._extract_docx(p.read_bytes(), p.name)
    except extract.ExtractError as e:
        return "ExtractError"

READERS = {
    "ckpt": read_checkpoint_tensors, "sam": is_sam_checkpoint,
    "inspect": lambda p: inspect_file(p).kind,
    "add": lambda p: add_local(str(p), None, on_duplicate="skip", fast=True),
    "docx": read_docx, "zip": lambda p: extract._extract_zip(p.read_bytes(), p.name),
}
before = peak_bytes()
result = READERS[sys.argv[1]](Path(sys.argv[2]))
print("CHILD-RESULT " + json.dumps({"result": repr(result),
                                    "grew_mb": (peak_bytes() - before) >> 20}))
"""


def _child_env(tmp_path: Path) -> dict:
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(
        json.dumps({"update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(REPO), "LOCALM_HOME": str(home),
            "PYTHONIOENCODING": "utf-8", "COLUMNS": "200"}


def _read_in_child(reader: str, path: Path, tmp_path: Path) -> dict:
    """What *reader* returns for *path* in a child process, and how many MB its peak
    memory grew during the read. Fails when the child dies or runs past the
    timeout."""
    run = subprocess.run([sys.executable, "-c", _CHILD, reader, str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=CHILD_TIMEOUT, env=_child_env(tmp_path),
                         stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    assert run.returncode == 0, f"exit {run.returncode & 0xFFFFFFFF:#x}: {run.stderr[-500:]}"
    line = next(x for x in run.stdout.splitlines() if x.startswith("CHILD-RESULT "))
    return json.loads(line.removeprefix("CHILD-RESULT "))


def _zip(path: Path, blob: bytes, method: int = zipfile.ZIP_STORED,
         member: str = "archive/data.pkl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", method) as zf:
        zf.writestr(member, blob, compress_type=method)
    return path


def _declare_size(path: Path, declared: int) -> Path:
    """Overwrite the uncompressed size that the only member's headers declare."""
    raw = bytearray(path.read_bytes())
    struct.pack_into("<I", raw, raw.rfind(b"PK\x01\x02") + 24, declared)
    struct.pack_into("<I", raw, raw.find(b"PK\x03\x04") + 22, declared)
    path.write_bytes(bytes(raw))
    return path


def _long1(value: int) -> bytes:
    data = value.to_bytes(value.bit_length() // 8 + 1, "little", signed=True)
    return b"\x8a" + bytes([len(data)]) + data


def _tensor_pickle(tensors: dict, *, wrapper: str | None = None) -> bytes:
    """The pickle of a state dict of *tensors*, in the opcodes ``torch.save`` uses:
    a ``_rebuild_tensor_v2`` global called on a storage, an offset and a shape."""
    out = b"\x80\x02"
    if wrapper:
        out += b"}" + b"X" + struct.pack("<I", len(wrapper)) + wrapper.encode()
    out += b"}"
    for name, shape in tensors.items():
        raw = name.encode()
        out += b"X" + struct.pack("<I", len(raw)) + raw
        out += b"ctorch._utils\n_rebuild_tensor_v2\n(NK\x00("
        out += b"".join(_long1(d) for d in shape) + b"tt" + b"Rs"
    if wrapper:
        out += b"s"
    return out + b"."


def _pth(path: Path, tensors: dict, **kw) -> Path:
    return _zip(path, _tensor_pickle(tensors, **kw))


def _safetensors(path: Path, tensors: dict) -> Path:
    header = {name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in tensors.items()}
    raw = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _gguf(path: Path, tensors: dict) -> Path:
    """A GGUF with no metadata at all and *tensors* (ggml order is reversed)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    for name, shape in tensors.items():
        raw = name.encode()
        out += struct.pack("<Q", len(raw)) + raw + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in reversed(shape))
        out += struct.pack("<IQ", 0, 0)
    path.write_bytes(out)
    return path


def _central_directory_bomb(path: Path, entries: int) -> Path:
    entry = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014b50, 20, 20, 0, 0, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 0) + b"a"
    with open(path, "wb") as f:
        for _ in range(entries // 1000):
            f.write(entry * 1000)
        f.write(struct.pack("<IHHHHIIH", 0x06054b50, 0, 0, 0xFFFF, 0xFFFF,
                            len(entry) * entries, 0, 0))
    return path


DEEP_KEY = b"\x80\x02}" + b")" + b"\x85" * 1_000_000 + b"Ns."
DEEP_VALUE = b"\x80\x02}" + b"X\x01\x00\x00\x00k" + b")" + b"\x85" * 1_000_000 + b"s."
DEEP_LIST = b"\x80\x02}" + b"X\x01\x00\x00\x00k" + b"]" + b"]" * 1_000_000 + b"a" * 1_000_000 + b"s."
SHARED_KEY = b"\x80\x02}" + b")" + b"22\x87" * 40 + b"Ns."
SHARED_GLOBAL_16 = b"\x80\x02" + b")" + b"22\x87" * 16 + b"X\x01\x00\x00\x00a\x93."
DEEP_GLOBAL = b"\x80\x02" + b")" + b"\x85" * 1_000_000 + b"X\x01\x00\x00\x00a\x93."


@pytest.fixture(scope="module")
def bombs(tmp_path_factory):
    """One small ``.pth`` per zip method, each declaring 200 bytes and holding
    ``BOMB_BYTES`` of zeros."""
    root = tmp_path_factory.mktemp("bombs")
    blob = b"\0" * BOMB_BYTES
    return {name: _declare_size(_zip(root / f"{name}.pth", blob, method), 200)
            for name, method in (("bzip2", zipfile.ZIP_BZIP2), ("lzma", zipfile.ZIP_LZMA),
                                 ("deflate", zipfile.ZIP_DEFLATED))}


class TestHostilePickle:
    @pytest.mark.parametrize("blob", [DEEP_KEY, DEEP_VALUE, DEEP_LIST, SHARED_KEY,
                                      SHARED_GLOBAL_16, DEEP_GLOBAL],
                             ids=["deep_key", "deep_value", "deep_list", "shared_key",
                                  "shared_global_16", "deep_global"])
    def test_hostile_pickle_returns_none_in_a_child(self, blob, tmp_path):
        path = _zip(tmp_path / "x.pth", blob)
        got = _read_in_child("ckpt", path, tmp_path)
        assert got["result"] == "None"
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    def test_a_hostile_pickle_is_an_unknown_model_in_a_child(self, tmp_path):
        got = _read_in_child("inspect", _zip(tmp_path / "x.pth", DEEP_KEY), tmp_path)
        assert got["result"] == "'unknown'"

    def test_a_non_string_key_is_dropped_and_the_tensors_beside_it_read(self, tmp_path):
        blob = _tensor_pickle(ESRGAN)
        assert blob.endswith(b".")
        with_int_key = blob[:-1] + b"K\x07Ns."
        assert read_checkpoint_tensors(_zip(tmp_path / "x.pth", with_int_key)) == ESRGAN


class TestDeclaredSizes:
    @pytest.mark.parametrize("reader", ["ckpt", "sam"])
    @pytest.mark.parametrize("method", ["bzip2", "lzma", "deflate"])
    def test_a_member_lying_about_its_size_stays_small(self, reader, method, bombs, tmp_path):
        got = _read_in_child(reader, bombs[method], tmp_path)
        assert got["result"] == ("None" if reader == "ckpt" else "False")
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    @pytest.mark.parametrize("reader", ["ckpt", "sam"])
    def test_a_stored_member_lying_about_its_compressed_size_stays_small(self, reader, tmp_path):
        path = _zip(tmp_path / "x.pth", b"\0" * (90 << 20))
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, raw.rfind(b"PK\x01\x02") + 24, 200)
        path.write_bytes(bytes(raw))
        got = _read_in_child(reader, path, tmp_path)
        assert got["result"] == ("None" if reader == "ckpt" else "False")
        assert got["grew_mb"] < 50

    @pytest.mark.parametrize("reader", ["ckpt", "sam"])
    def test_a_huge_central_directory_stays_small(self, reader, tmp_path):
        path = _central_directory_bomb(tmp_path / "x.pth", 1_000_000)
        got = _read_in_child(reader, path, tmp_path)
        assert got["result"] == ("None" if reader == "ckpt" else "False")
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    def test_a_member_declaring_more_than_the_limit_is_not_read(self, tmp_path):
        path = _zip(tmp_path / "x.pth", b"\0" * (9 << 20), zipfile.ZIP_DEFLATED)
        assert read_checkpoint_tensors(path) is None

    def test_a_deflated_checkpoint_still_reads(self, tmp_path):
        path = _zip(tmp_path / "x.pth", _tensor_pickle(ESRGAN), zipfile.ZIP_DEFLATED)
        assert read_checkpoint_tensors(path) == ESRGAN

    @pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
    def test_a_checkpoint_packed_with_another_method_is_unknown(self, method, tmp_path):
        path = _zip(tmp_path / "x.pth", _tensor_pickle(ESRGAN), method)
        assert read_checkpoint_tensors(path) is None
        assert inspect_file(path).kind == "unknown"

    def test_the_read_cap_is_above_every_read_the_readers_ask_for(self):
        assert ckpt_tensors._MAX_ZIP_READ > ckpt_tensors.PICKLE_MAX_BYTES
        assert ckpt_tensors._MAX_ZIP_READ > sam._PICKLE_MAX_BYTES
        assert ckpt_tensors._MAX_ZIP_READ > registry._EXTRACT_CHUNK_BYTES

    @pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED],
                             ids=["stored", "deflate"])
    def test_a_member_of_exactly_the_limit_still_reads(self, method, tmp_path):
        blob = _tensor_pickle(ESRGAN)
        blob += b"\0" * (ckpt_tensors.PICKLE_MAX_BYTES - len(blob))
        assert read_checkpoint_tensors(_zip(tmp_path / "x.pth", blob, method)) == ESRGAN

    def test_many_members_and_tensors_still_read(self, tmp_path):
        names = {f"layer.{i}.weight": (3, 3) for i in range(20000)}
        path = _zip(tmp_path / "x.pth", _tensor_pickle(names))
        with zipfile.ZipFile(path, "a") as zf:
            for i in range(20000):
                zf.writestr(f"archive/data/{i}", b"")
        assert read_checkpoint_tensors(path) == names


class TestLlamafile:
    def test_a_member_packed_with_lzma_is_refused_and_stays_small(self, tmp_path):
        path = _declare_size(
            _zip(tmp_path / "m.llamafile", b"GGUF" + b"\0" * BOMB_BYTES, zipfile.ZIP_LZMA,
                 member="m.gguf"), 200)
        got = _read_in_child("add", path, tmp_path)
        assert got["result"] == "False"
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    def test_a_huge_central_directory_is_not_a_llamafile(self, tmp_path):
        path = _central_directory_bomb(tmp_path / "m.llamafile", 1_000_000)
        got = _read_in_child("add", path, tmp_path)
        assert got["result"] == "False"
        assert got["grew_mb"] < GROWTH_LIMIT_MB


class TestDocumentArchives:
    def test_a_docx_with_an_lzma_body_is_refused_and_stays_small(self, tmp_path):
        path = _zip(tmp_path / "x.docx", b"<a/>" + b" " * BOMB_BYTES, zipfile.ZIP_LZMA,
                    member="word/document.xml")
        got = _read_in_child("docx", path, tmp_path)
        assert got["result"] == "'ExtractError'"
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    def test_a_zip_with_a_bzip2_member_skips_it_and_stays_small(self, tmp_path):
        path = _zip(tmp_path / "x.zip", b"a" * BOMB_BYTES, zipfile.ZIP_BZIP2, member="bomb.txt")
        got = _read_in_child("zip", path, tmp_path)
        assert "aaaa" not in got["result"]
        assert got["grew_mb"] < GROWTH_LIMIT_MB

    def test_a_deflated_zip_member_still_extracts(self, tmp_path):
        from localm.rag import extract
        path = _zip(tmp_path / "x.zip", b"hello there", zipfile.ZIP_DEFLATED, member="note.txt")
        assert "hello there" in extract._extract_zip(path.read_bytes(), path.name)


class TestUnclassifiable:
    @pytest.mark.parametrize("tensors, error", [(SRVGG_OVERFLOW, OverflowError),
                                                (LONG_DIGITS, ValueError)],
                             ids=["overflow", "digits"])
    def test_classify_tensors_turns_the_failure_into_unknown(self, tensors, error):
        with pytest.raises(error):
            families._classify_tensors(tensors, None)
        assert classify_tensors(tensors) is UNKNOWN

    @pytest.mark.parametrize("tensors", [SRVGG_OVERFLOW, LONG_DIGITS],
                             ids=["overflow", "digits"])
    @pytest.mark.parametrize("suffix", ["pth", "safetensors"])
    def test_inspect_file_is_unknown_for_every_container(self, suffix, tensors, tmp_path):
        write = _pth if suffix == "pth" else _safetensors
        assert inspect_file(write(tmp_path / f"x.{suffix}", tensors)) is UNKNOWN

    def test_a_gguf_of_digits_is_unknown_and_not_a_media_part(self, tmp_path):
        path = _gguf(tmp_path / "x.gguf", LONG_DIGITS)
        assert inspect_file(path) is UNKNOWN
        assert gguf_media_component_type(path, meta={"architecture": None, "x": 1}) is None


class TestFolderAdd:
    def _add(self, folder: Path, tmp_path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "localm", "add", str(folder), "--on-duplicate", "skip"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=CHILD_TIMEOUT, env=_child_env(tmp_path), cwd=str(REPO),
            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)

    def test_good_files_register_and_each_bad_file_is_reported(self, tmp_path):
        folder = tmp_path / "in"
        _safetensors(folder / "good_vae.safetensors", VAE)
        _pth(folder / "good_esrgan.pth", ESRGAN, wrapper="params_ema")
        _pth(folder / "bad_overflow.pth", SRVGG_OVERFLOW)
        _pth(folder / "bad_digits.pth", LONG_DIGITS)
        _safetensors(folder / "bad_digits2.safetensors", LONG_DIGITS)
        _zip(folder / "bad_deep_key.pth", DEEP_KEY)
        _zip(folder / "bad_shared_key.pth", SHARED_KEY)
        run = self._add(folder, tmp_path)
        out = " ".join(run.stdout.split())
        assert run.returncode == 0, f"exit {run.returncode & 0xFFFFFFFF:#x}: {run.stderr[-500:]}"
        registry = json.loads((tmp_path / "home" / "registry.json").read_text(encoding="utf-8"))
        assert {Path(e["path"]).name: e["model_type"] for e in registry.values()} == {
            "good_vae.safetensors": "vae", "good_esrgan.pth": "upscaler"}
        for name in ("bad_overflow.pth", "bad_digits.pth", "bad_digits2.safetensors",
                     "bad_deep_key.pth", "bad_shared_key.pth"):
            assert f"Skipped {name}" in out


def _torch_write(script: str, tmp_path: Path, *paths: Path) -> None:
    """Runs *script* with torch imported and *paths* as ``argv[1:]`` in a child, which
    keeps torch's native libraries out of the test process."""
    env = _child_env(tmp_path)
    run = subprocess.run([sys.executable, "-c", "import sys, torch\n" + script,
                          *map(str, paths)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300, env=env, stdin=subprocess.DEVNULL,
                         creationflags=NO_WINDOW)
    assert run.returncode == 0, run.stderr[-2000:]


_STATE = """
sd = {"conv_first.weight": torch.zeros(4, 3, 3, 3), "conv_first.bias": torch.zeros(4),
      "body.0.rdb1.conv1.weight": torch.zeros(4, 4, 3, 3),
      "conv_last.weight": torch.zeros(3, 4, 3, 3)}
"""


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
class TestRealTorchSave:
    @pytest.mark.parametrize("save", [
        "torch.save(sd, sys.argv[1])",
        "torch.save({'params_ema': sd, 'ids': {0: 1}}, sys.argv[1])",
        "torch.save({'state_dict': sd, 'note': 'x' * 100000}, sys.argv[1], pickle_protocol=4)",
    ], ids=["plain", "wrapped_int_key", "protocol4_stack_global"])
    def test_a_real_file_reads_and_classifies(self, save, tmp_path):
        path = tmp_path / "real.pth"
        _torch_write(_STATE + save, tmp_path, path)
        with zipfile.ZipFile(path) as zf:
            assert {i.compress_type for i in zf.infolist()
                    if i.filename.endswith("data.pkl")} == {zipfile.ZIP_STORED}
        assert read_checkpoint_tensors(path) == ESRGAN
        assert inspect_file(path).kind == "upscaler"

    def test_a_legacy_file_is_unknown(self, tmp_path):
        path = tmp_path / "legacy.pth"
        _torch_write(_STATE + "torch.save(sd, sys.argv[1], _use_new_zipfile_serialization=False)",
                     tmp_path, path)
        assert read_checkpoint_tensors(path) is None
        assert inspect_file(path) is UNKNOWN

    def test_a_real_file_rezipped_with_deflate_reads_as_torch_load_does(self, tmp_path):
        path = tmp_path / "real.pth"
        rezipped = tmp_path / "rezipped.pth"
        _torch_write(_STATE + "torch.save(sd, sys.argv[1])", tmp_path, path)
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(rezipped, "w") as dst:
            for info in src.infolist():
                dst.writestr(info.filename, src.read(info), compress_type=zipfile.ZIP_DEFLATED)
        assert read_checkpoint_tensors(rezipped) == ESRGAN
        _torch_write("assert torch.load(sys.argv[1], weights_only=True)['conv_last.weight'].shape"
                     " == (3, 4, 3, 3)", tmp_path, rezipped)

    def test_a_method_torch_cannot_read_is_unknown_here_too(self, tmp_path):
        path = tmp_path / "real.pth"
        rezipped = tmp_path / "rezipped.pth"
        _torch_write(_STATE + "torch.save(sd, sys.argv[1])", tmp_path, path)
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(rezipped, "w") as dst:
            for info in src.infolist():
                dst.writestr(info.filename, src.read(info), compress_type=zipfile.ZIP_BZIP2)
        assert read_checkpoint_tensors(rezipped) is None
        _torch_write("""
try:
    torch.load(sys.argv[1], weights_only=True)
except RuntimeError:
    sys.exit(0)
sys.exit("torch.load read a bzip2 archive")
""", tmp_path, rezipped)
