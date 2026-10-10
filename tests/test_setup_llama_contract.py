# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract of ``localm setup-llama``: the CLI surface, backend choice and
fallback order, build/pin resolution, download verification, and the files and
markers a provision leaves behind.

The command runs end to end. Only the outer seams are replaced: the network
(``verified_urlopen``), child processes (``subprocess.run``: the load probe and
nvidia-smi, and ``shutil.which``, which finds nothing on PATH), GPU detection
(``hwdetect.detect`` and
``hwdetect.recommended_install_backend``), and the ``localm_llama_runtime``
package, which decides the runtime lib dir. Asset resolution, download,
checksum, extraction, copying, the marker, the rollback history and the
provisioning lock all run for real.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tarfile
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import localm.setup_llama as sl
from localm import config, cpu_backend_select, hwdetect
from localm.bugreport import LocalmError
from localm.http_ssl import RedirectDowngradeRefused

PIN = sl._PINNED_TAG
ROCM = sl._ROCM_TAG
API = "https://api.github.com/repos"
UPSTREAM = "ggml-org/llama.cpp"
LEMONADE = "lemonade-sdk/llamacpp-rocm"
HYBRID = "hybridgroup/llama-cpp-builder"


# --------------------------------------------------------------------------- #
#  Fixture builders                                                            #
# --------------------------------------------------------------------------- #

def _noise(n: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(n)


def _zip(files: dict, seed: int = 1) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("pad.bin", _noise(300_000, seed))
    return buf.getvalue()


def _targz(files: dict, seed: int = 2) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in {**files, "pad.bin": _noise(300_000, seed)}.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_elf64_needing(*needed: str) -> bytes:
    strtab = b"\x00"
    offsets = []
    for name in needed:
        offsets.append(len(strtab))
        strtab += name.encode("ascii") + b"\x00"
    dynamic = b"".join(struct.pack("<qQ", 1, o) for o in offsets)
    dynamic += struct.pack("<qQ", 0, 0)
    strtab_off = 64
    dynamic_off = strtab_off + len(strtab)
    shoff = dynamic_off + len(dynamic)
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4], ehdr[5], ehdr[6] = 2, 1, 1
    struct.pack_into("<Q", ehdr, 0x28, shoff)
    struct.pack_into("<H", ehdr, 0x34, 64)
    struct.pack_into("<H", ehdr, 0x3A, 64)
    struct.pack_into("<H", ehdr, 0x3C, 3)
    struct.pack_into("<H", ehdr, 0x3E, 0)

    def shdr(sh_type, off, size, link=0):
        return struct.pack("<IIQQQQIIQQ", 0, sh_type, 0, 0, off, size, link, 0, 0, 0)

    sections = (shdr(0, 0, 0) + shdr(3, strtab_off, len(strtab))
                + shdr(6, dynamic_off, len(dynamic), link=1))
    return bytes(ehdr) + strtab + dynamic + sections


def _fake_libgomp_deb(so_content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("./usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")
        ti.size = len(so_content)
        tf.addfile(ti, io.BytesIO(so_content))
        link = tarfile.TarInfo("./usr/lib/x86_64-linux-gnu/libgomp.so.1")
        link.type = tarfile.SYMTYPE
        link.linkname = "libgomp.so.1.0.0"
        tf.addfile(link)
    data = buf.getvalue()
    out = bytearray(b"!<arch>\n")
    for name, content in (("debian-binary", b"2.0\n"), ("data.tar.gz", data)):
        header = (name.ljust(16) + "0".ljust(12) + "0".ljust(6) + "0".ljust(6)
                  + "100644".ljust(8) + str(len(content)).ljust(10) + "\x60\n")
        out += header.encode("ascii") + content
        if len(content) % 2:
            out += b"\n"
    return bytes(out)


WIN_VULKAN = {"llama/llama.dll": b"llama@vulkan", "llama/ggml.dll": b"ggml@vulkan",
              "llama/ggml-vulkan.dll": b"ggml-vulkan", "llama/llama-cli.exe": b"exe",
              "llama/LICENSE": b"UPSTREAM LICENSE TEXT"}
WIN_VULKAN_FILES = [".gitignore", ".gitkeep", ".localm-backend", "LICENSE.llama-cpp",
                    "ggml-vulkan.dll", "ggml.dll", "llama.dll"]
WIN_CPU = {"llama.dll": b"llama@cpu", "ggml.dll": b"ggml@cpu", "ggml-cpu.dll": b"ggml-cpu"}
WIN_CPU_FILES = [".gitignore", ".gitkeep", ".localm-backend", "LICENSE.llama-cpp",
                 "ggml-cpu.dll", "ggml.dll", "llama.dll"]
WIN_CUDA = {"llama.dll": b"llama@cuda", "ggml.dll": b"ggml@cuda", "ggml-cuda.dll": b"ggml-cuda",
            "LICENSE": b"UPSTREAM LICENSE TEXT"}
WIN_CUDART = {"cudart64_12.dll": b"cudart", "cublas64_12.dll": b"cublas",
              "cublasLt64_12.dll": b"cublasLt"}
LINUX_CPU = {"build/bin/libllama.so": b"libllama", "build/bin/libggml-base.so.0": b"base",
             "build/bin/libggml-cpu.so": b"cpu", "build/bin/llama-cli": b"elf-exe",
             "LICENSE": b"UPSTREAM LICENSE TEXT"}
MAC_METAL = {"build/bin/libllama.dylib": b"libllama", "build/bin/libggml-metal.dylib": b"metal",
             "build/bin/llama-cli": b"macho-exe"}
AMD_ROCM = {"llama.dll": b"llama@rocm", "ggml-hip.dll": b"ggml-hip", "rocblas.dll": b"rocblas",
            "ggml-cpu.dll": b"ggml-cpu@rocm",
            "rocblas/library/TensileLibrary_gfx1030.dat": b"tensile",
            "rocblas/library/Kernels.so-000-gfx1030.hsaco": b"kernels"}


# --------------------------------------------------------------------------- #
#  The fake world                                                              #
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, url: str, body: bytes, ctype: str):
        self._buf = io.BytesIO(body)
        self._url = url
        self.headers = {"Content-Length": str(len(body)), "Content-Type": ctype}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class World:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.mp = monkeypatch
        self.lib = tmp_path / "clone" / "runtime" / "localm_llama_runtime" / "lib"
        mod = types.ModuleType("localm_llama_runtime")
        mod.LIB_DIR = str(self.lib)
        mod.lib_dir = lambda: str(self.lib)
        monkeypatch.setitem(sys.modules, "localm_llama_runtime", mod)
        monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
        monkeypatch.setattr(sl.console, "_width", 10_000)
        self.routes: dict = {}
        self.requests: list = []
        self.probes: list = []
        self.probe_calls = 0
        self.other_cmds: list = []
        self.nvidia = None
        self.vendors: list = []
        self.gpu_names = ""
        self.recommended = "vulkan"
        self.cpu_scores: dict = {}
        monkeypatch.setattr(sl, "verified_urlopen", self._urlopen)
        monkeypatch.setattr(subprocess, "run", self._run)
        monkeypatch.setattr(shutil, "which", lambda cmd, *args, **kwargs: None)
        monkeypatch.setattr(hwdetect, "detect", lambda: SimpleNamespace(
            vendors=list(self.vendors), gpu_names=self.gpu_names))
        monkeypatch.setattr(hwdetect, "recommended_install_backend",
                            lambda det=None: self.recommended)
        self.platform("win32")

    # -- environment ------------------------------------------------------- #

    def platform(self, plat: str) -> None:
        self.mp.setattr(sys, "platform", plat)

    def seed_target(self, *names: str, marker: str | None = None) -> None:
        self.lib.mkdir(parents=True, exist_ok=True)
        for n in names:
            (self.lib / n).write_bytes(b"old:" + n.encode())
        if marker is not None:
            (self.lib / ".localm-backend").write_text(marker + "\n", encoding="utf-8")

    def files(self) -> list:
        if not self.lib.exists():
            return []
        return sorted(p.relative_to(self.lib).as_posix()
                      for p in self.lib.rglob("*") if p.is_file())

    def marker(self) -> str | None:
        p = self.lib / ".localm-backend"
        return p.read_text(encoding="utf-8") if p.exists() else None

    @staticmethod
    def history() -> list:
        return [(e["backend"], e["tag"])
                for e in config.load_config().get("llama_runtime_history") or []]

    @staticmethod
    def pin():
        return config.load_config().get("llama_runtime_pin") or None

    # -- network ----------------------------------------------------------- #

    def _urlopen(self, req, timeout=None, **kw):
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        self.requests.append(url)
        hit = self.routes.get(url)
        if hit is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if isinstance(hit, BaseException):
            raise hit
        body, ctype = hit
        return _Resp(url, body, ctype)

    def serve(self, url: str, body, ctype: str = "application/octet-stream") -> str:
        self.routes[url] = body if isinstance(body, BaseException) else (body, ctype)
        return url

    def serve_json(self, url: str, obj) -> str:
        return self.serve(url, json.dumps(obj).encode(), "application/json")

    def release(self, repo: str, tag: str, assets: list) -> dict:
        """Publish *assets* ``[(name, body, digest)]`` under a release listing.
        *digest* is "ok" (the body's real sha256), None (no digest field), or a
        literal digest string. Returns ``{name: download_url}``."""
        listing, urls = [], {}
        for name, body, digest in assets:
            url = self.serve(f"https://github.com/{repo}/releases/download/{tag}/{name}", body)
            urls[name] = url
            entry = {"name": name, "browser_download_url": url, "size": len(body)}
            if digest == "ok":
                entry["digest"] = "sha256:" + _sha(body)
            elif digest:
                entry["digest"] = digest
            listing.append(entry)
        self.serve_json(f"{API}/{repo}/releases/tags/{tag}", {"tag_name": tag, "assets": listing})
        return urls

    # -- processes ---------------------------------------------------------- #

    def _run(self, cmd, *args, **kwargs):
        cmd = [str(c) for c in cmd]
        if len(cmd) >= 3 and cmd[1] == "-c" and cmd[2] == sl._LOAD_PROBE_CODE:
            self.probe_calls += 1
            rc, err = self.probes.pop(0) if self.probes else (0, "")
            return subprocess.CompletedProcess(cmd, rc, "", err)
        if len(cmd) >= 3 and cmd[1] == "-c" and cmd[2] == cpu_backend_select._SCORE_PROBE:
            name = Path(kwargs["env"]["LOCALM_CPU_TIER_CANDIDATE"]).name
            score = self.cpu_scores.get(name)
            verdict = {"score": score, "error": None if score is not None else "could not load"}
            return subprocess.CompletedProcess(
                cmd, 0, "\n@@VERDICT@@" + json.dumps(verdict) + "\n", "")
        if "nvidia-smi" in Path(cmd[0]).name.lower():
            if self.nvidia is None:
                raise FileNotFoundError(cmd[0])
            driver, cuda, name, cap = self.nvidia
            rest = cmd[1:]
            if "--query-gpu=name" in rest:
                out = name + "\n"
            elif "--query-gpu=compute_cap" in rest:
                out = cap + "\n"
            else:
                out = (f"| NVIDIA-SMI {driver}   Driver Version: {driver}   "
                       f"CUDA Version: {cuda}   |\n")
            return subprocess.CompletedProcess(cmd, 0, out, "")
        self.other_cmds.append(cmd)
        raise FileNotFoundError(cmd[0])

    # -- running ------------------------------------------------------------ #

    def invoke(self, *args: str):
        result = CliRunner().invoke(sl.main, list(args))
        result.text = " ".join(result.output.split())
        return result


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _in_order(text: str, *fragments: str) -> None:
    pos = 0
    for frag in fragments:
        i = text.find(frag, pos)
        assert i >= 0, f"missing (or out of order): {frag!r}\n--- output ---\n{text}"
        pos = i + len(frag)


ROCM_CPU = {"ggml-base.dll": b"base@b10270", "ggml-cpu-x64.dll": b"cpu-x64",
            "ggml-cpu-haswell.dll": b"cpu-haswell", "ggml-cpu-zen4.dll": b"cpu-zen4",
            "libomp.dll": b"openmp"}


def _rocm_cpu_overlay(world, *, scores=None, digest_ok=True, files=None):
    """Serve the pinned upstream CPU archive the amd-rocm provision installs its
    CPU backend from, with this CPU's ggml_backend_score() per variant."""
    body = _zip(ROCM_CPU if files is None else files, 7)
    world.mp.setitem(sl._PINNED_FALLBACK_SHA256, sl._ROCM_CPU_ASSET,
                     _sha(body) if digest_ok else "0" * 64)
    world.cpu_scores = ({"ggml-cpu-x64.dll": 1, "ggml-cpu-haswell.dll": 64}
                        if scores is None else scores)
    return world.serve(sl.rocm_cpu_overlay_url(), body)


def _upstream_win_vulkan(world, tag=PIN, digest="ok", body=None):
    body = _zip(WIN_VULKAN) if body is None else body
    name = f"llama-{tag}-bin-win-vulkan-x64.zip"
    urls = world.release(UPSTREAM, tag, [
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", _zip(WIN_CUDART, 9), "ok"),
        (f"llama-{tag}-bin-win-cpu-x64.zip", _zip(WIN_CPU, 3), "ok"),
        (name, body, digest),
    ])
    return urls[name], body


# --------------------------------------------------------------------------- #
#  CLI surface                                                                 #
# --------------------------------------------------------------------------- #

HELP = """\
Usage: setup-llama [OPTIONS]

  Download or copy the native llama.cpp binaries into localm's own venv.

  The chosen backend is load-tested after provisioning. If it cannot load on this machine, your pick
  is NOT changed silently: for a vendor backend (cuda/hip/sycl/amd-rocm, e.g. CUDA without a new-
  enough driver) setup explains why and (interactively) offers the universal Vulkan build instead,
  or - in a non-interactive install - falls back with a loud warning and tells you how to retry your
  backend once the cause is fixed. vulkan and cpu are themselves the universal builds, so a failure
  there instead names the missing piece and offers a retry, then reports the cause and stops rather
  than silently degrading.

  By default the llama.cpp build this localm release confirmed is installed. --tag <tag> pins
  another exact release, --tag latest tracks upstream's newest (untested here), --tag default
  returns to the confirmed build, and --rollback returns to the previous one. The choice sticks
  across 'localm update'.

    localm setup-llama                        # auto-detect GPU, fetch the right prebuilt
    localm setup-llama --backend vulkan       # universal GPU build (any vendor)
    localm setup-llama --backend cuda         # NVIDIA: checks the driver, fetches a
                                              #   self-contained CUDA runtime (no Toolkit)
    localm setup-llama --backend cpu          # no GPU
    localm setup-llama --backend cuda --cuda-line cuda-12   # image build, no GPU
    localm setup-llama --from /path/to/llama.cpp/build/bin
    localm setup-llama --url https://.../llama-...zip
    localm setup-llama --sha256 <hex>         # pin the expected archive digest
    localm setup-llama --tag b10355           # install exactly b10355 and keep it
    localm setup-llama --tag latest           # track upstream's newest (untested here)
    localm setup-llama --tag default          # back to the build localm confirmed
    localm setup-llama --rollback             # back to the previous build

Options:
  --from DIRECTORY                Copy binaries from a local llama.cpp build directory instead of
                                  downloading.
  --backend [auto|vulkan|cuda|sycl|hip|cpu|metal|amd-rocm]
                                  Which prebuilt to fetch. 'auto' detects your GPU and picks the
                                  best-performing backend it can run out of the box: cuda for NVIDIA
                                  on both Windows and Linux (self-contained, falls back to vulkan if
                                  the driver is too old); the self-contained ROCm build for AMD RX
                                  6000 on Windows; hip for AMD elsewhere when a system ROCm/HIP
                                  toolkit is detected; sycl for Intel on Windows (self-contained);
                                  vulkan for Intel on Linux and for AMD with no toolkit detected;
                                  cpu if no GPU.
  --url TEXT                      Override with an explicit prebuilt archive URL.
  --sha256 TEXT                   Expected sha256 of the downloaded archive. When given, the
                                  download is refused unless its digest matches (opt-in integrity
                                  pin).
  --force                         Re-provision even if binaries are already present.
  --tag TAG                       Install a specific llama.cpp release (e.g. 'b10355') and PIN it,
                                  so later setup-llama runs and 'localm update' keep that exact
                                  build. Two words are special: 'latest' opts in to upstream's
                                  newest release, which localm has not tested, and 'default' returns
                                  to the build localm ships and confirmed.
  --rollback                      Go back to the previous llama.cpp build recorded for this backend
                                  and pin it. For when an upstream release turns out to be broken on
                                  your hardware. See 'localm doctor' for what is installed now.
  --cuda-line [cuda-12|cuda-13]   With --backend cuda on Linux: fetch the CUDA build and runtime
                                  libraries of this line without an NVIDIA GPU present, for building
                                  container images. cuda-12 covers every architecture before
                                  Blackwell, cuda-13 is for Blackwell. The driver check and the load
                                  test are skipped, the runtime is recorded as not load-tested, and
                                  the container's start check tests it on the GPU host.
  -y, --yes                       Non-interactive: accept the recommended action at every prompt
                                  (e.g. fetch the self-contained CUDA runtime). Used by the one-
                                  click installer and for scripted setups.
  -h, --help                      Show this message and exit.
"""


def test_cli_surface_is_unchanged():
    assert sl.main.name == "setup-llama"
    assert sl.BACKENDS == ("auto", "vulkan", "cuda", "sycl", "hip", "cpu", "metal", "amd-rocm")
    shape = [(p.name, tuple(p.opts), type(p.type).__name__,
              tuple(p.type.choices) if hasattr(p.type, "choices") else None,
              getattr(p, "is_flag", False)) for p in sl.main.params]
    assert shape == [
        ("from_dir", ("--from",), "Path", None, False),
        ("backend", ("--backend",), "Choice", sl.BACKENDS, False),
        ("url", ("--url",), "StringParamType", None, False),
        ("sha256", ("--sha256",), "StringParamType", None, False),
        ("force", ("--force",), "BoolParamType", None, True),
        ("tag", ("--tag",), "StringParamType", None, False),
        ("rollback", ("--rollback",), "BoolParamType", None, True),
        ("cuda_line", ("--cuda-line",), "Choice", ("cuda-12", "cuda-13"), False),
        ("assume_yes", ("--yes", "-y"), "BoolParamType", None, True),
    ]
    result = CliRunner().invoke(sl.main, ["--help"], terminal_width=100)
    assert result.exit_code == 0
    assert result.output == HELP
    from localm.cli.maintenance import main as maintenance
    assert maintenance.commands["setup-llama"] is sl.main


def test_an_unknown_backend_is_a_usage_error(world):
    r = world.invoke("--backend", "opencl")
    assert r.exit_code == 2
    assert "Invalid value for '--backend'" in r.text
    assert world.requests == [] and world.probe_calls == 0


@pytest.mark.parametrize("args, message", [
    (["--tag", "b1", "--rollback"],
     "Error: --tag and --rollback both choose a build; pass only one. --rollback goes to the "
     "previous recorded build, --tag names one."),
    (["--tag", "../evil"],
     "Error: '../evil' is not a usable release tag. " + sl.TAG_HELP),
    (["--tag", ""],
     "Error: '' is not a usable release tag. " + sl.TAG_HELP),
    (["--tag", "b1", "--url", "https://example.invalid/x.zip"],
     "Error: --tag selects an upstream llama.cpp release, so it cannot be combined with --url, "
     "which installs a build you supply. Run them separately."),
    (["--rollback"],
     "Error: --rollback needs to know which backend to roll back, and nothing is recorded as "
     "installed on this machine. Name it explicitly, for example: localm setup-llama "
     "--rollback --backend vulkan"),
    (["--rollback", "--backend", "amd-rocm"],
     f"Error: the amd-rocm backend cannot be rolled back: its build is fixed by the localm "
     f"release you are running ({ROCM}, from lemonade-sdk), not chosen from upstream llama.cpp "
     "releases. To try a different llama.cpp build on this machine, switch backend, for "
     "example: localm setup-llama --backend vulkan --tag <tag>"),
    (["--rollback", "--backend", "vulkan"],
     "Error: no earlier llama.cpp build is recorded for the vulkan backend, so there is nothing "
     "to roll back to. Install a specific build instead, for example: localm setup-llama "
     "--tag b10355"),
])
def test_version_requests_that_cannot_be_honoured_are_refused(world, args, message):
    r = world.invoke(*args)
    assert r.exit_code == 1
    assert message in r.text
    assert world.requests == [] and world.probe_calls == 0
    assert world.pin() is None
    assert not world.lib.exists()


def test_rollback_with_from_is_refused(world, tmp_path):
    src = tmp_path / "build"
    src.mkdir()
    r = world.invoke("--rollback", "--from", str(src))
    assert r.exit_code == 1
    assert ("Error: --rollback selects an upstream llama.cpp release, so it cannot be combined "
            "with --from, which installs a build you supply. Run them separately.") in r.text


# --------------------------------------------------------------------------- #
#  Successful provisions, per platform and backend                            #
# --------------------------------------------------------------------------- #

def test_windows_vulkan_provision_places_libraries_marker_and_history(world):
    url, _ = _upstream_win_vulkan(world)
    world.seed_target(".gitignore", ".gitkeep", "stale-old.dll")
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              "Backend: vulkan (universal GPU build (AMD/NVIDIA/Intel via the display driver))",
              f"Downloading {url}",
              "OK - vulkan runtime loads on this machine.",
              f"Native runtime ready -> {world.lib}",
              "Try it: localm run <model>")
    assert "Heads up" not in r.text and "Warning" not in r.text
    assert world.requests == [f"{API}/{UPSTREAM}/releases/tags/{PIN}", url]
    assert world.probe_calls == 1
    assert world.files() == WIN_VULKAN_FILES
    assert (world.lib / "LICENSE.llama-cpp").read_bytes() == b"UPSTREAM LICENSE TEXT"
    assert (world.lib / ".gitignore").read_bytes() == b"old:.gitignore"
    assert world.marker() == f"vulkan {PIN}\n"
    assert world.history() == [("vulkan", PIN)]
    assert world.pin() is None
    assert not (world.lib.parent / "lib.setup.lock").exists()
    assert world.other_cmds == []


def test_auto_resolves_through_the_install_policy(world):
    url, _ = _upstream_win_vulkan(world)
    world.vendors, world.recommended = ["amd"], "vulkan"
    r = world.invoke()
    assert r.exit_code == 0, r.output
    assert "Backend: vulkan (universal GPU build" in r.text
    assert "Heads up" not in r.text
    assert world.marker() == f"vulkan {PIN}\n"


def test_auto_with_detection_failure_defaults_to_cpu(world, monkeypatch):
    def boom():
        raise RuntimeError("wmi exploded")
    monkeypatch.setattr(hwdetect, "detect", boom)
    world.release(UPSTREAM, PIN, [(f"llama-{PIN}-bin-win-cpu-x64.zip", _zip(WIN_CPU), "ok")])
    world.seed_target(".gitignore", ".gitkeep")
    r = world.invoke()
    assert r.exit_code == 0, r.output
    _in_order(r.text, "GPU detection failed (wmi exploded); defaulting to CPU - override with "
                      "--backend.", "Backend: cpu (CPU-only build (no GPU))")
    assert world.files() == WIN_CPU_FILES
    assert world.marker() == f"cpu {PIN}\n"


def test_linux_cpu_provision_copies_shared_objects_only(world):
    world.platform("linux")
    body = _targz(LINUX_CPU)
    urls = world.release(UPSTREAM, PIN, [
        (f"llama-{PIN}-bin-ubuntu-vulkan-x64.tar.gz", _targz({"x/libllama.so": b"v"}, 5), "ok"),
        (f"llama-{PIN}-bin-ubuntu-x64.tar.gz", body, "ok"),
    ])
    r = world.invoke("--backend", "cpu")
    assert r.exit_code == 0, r.output
    url = urls[f"llama-{PIN}-bin-ubuntu-x64.tar.gz"]
    _in_order(r.text, "Backend: cpu (CPU-only build (no GPU))", f"Downloading {url}",
              "OK - cpu runtime loads on this machine.", f"Native runtime ready -> {world.lib}")
    assert world.files() == [".localm-backend", "LICENSE.llama-cpp", "libggml-base.so.0",
                             "libggml-cpu.so", "libllama.so"]
    assert world.marker() == f"cpu {PIN}\n"
    assert world.requests == [f"{API}/{UPSTREAM}/releases/tags/{PIN}", url]


def test_macos_metal_provision(world):
    world.platform("darwin")
    name = f"llama-{PIN}-bin-macos-arm64.tar.gz"
    urls = world.release(UPSTREAM, PIN, [(name, _targz(MAC_METAL), "ok")])
    r = world.invoke("--backend", "metal", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text, "Backend: metal (Apple Silicon (Metal) build)", f"Downloading {urls[name]}",
              "OK - metal runtime loads on this machine.")
    assert world.files() == [".localm-backend", "LICENSE.llama-cpp", "libggml-metal.dylib",
                             "libllama.dylib"]
    assert (world.lib / "LICENSE.llama-cpp").read_text(encoding="utf-8") == sl._LLAMA_CPP_MIT_NOTICE
    assert world.marker() == f"metal {PIN}\n"


@pytest.mark.parametrize("pin, note", [
    (None, None),
    ("b10355", f"Note: the pinned llama.cpp build b10355 does not apply to the amd-rocm backend - "
               f"it ships from lemonade-sdk's own release numbering ({ROCM}), a different tag "
               "series. The pin stays set and applies to every other backend."),
    ("latest", f"Note: '--tag latest' does not apply to the amd-rocm backend - it ships from "
               f"lemonade-sdk's own release numbering ({ROCM}), a different tag series, fixed by "
               "the localm release you are running. The setting stays and applies to every "
               "other backend."),
])
def test_windows_amd_rocm_provision_keeps_blas_kernel_layout(world, pin, note):
    if pin:
        sl.set_pinned_tag(pin)
    world.vendors, world.gpu_names = ["amd"], "amd radeon rx 6900 xt"
    name = f"llama-{ROCM}-windows-rocm-gfx103X-x64.zip"
    urls = world.release(LEMONADE, ROCM, [
        (f"llama-{ROCM}-windows-rocm-gfx110X-x64.zip", _zip({"llama.dll": b"x"}, 4), "ok"),
        (name, _zip(AMD_ROCM), "ok"),
    ])
    cpu_url = _rocm_cpu_overlay(world)
    r = world.invoke("--backend", "amd-rocm")
    assert r.exit_code == 0, r.output
    if note:
        _in_order(r.text, note, "Backend: amd-rocm")
    else:
        assert "Note:" not in r.text
    _in_order(r.text, "Backend: amd-rocm (self-contained AMD ROCm build (gfx103X / RX 6000))",
              f"Downloading {urls[name]}", f"Downloading {cpu_url}",
              f"CPU backend: ggml-cpu-haswell from llama.cpp {sl._ROCM_CPU_TAG} (SIMD), "
              "replacing the amd-rocm build's own", "OK - amd-rocm runtime loads on this machine.")
    assert world.files() == [".localm-backend", ".localm-cpu-overlay", "LICENSE.llama-cpp",
                             "ggml-cpu.dll", "ggml-hip.dll", "libomp.dll", "llama.dll",
                             "rocblas.dll", "rocblas/library/Kernels.so-000-gfx1030.hsaco",
                             "rocblas/library/TensileLibrary_gfx1030.dat"]
    assert (world.lib / "ggml-cpu.dll").read_bytes() == b"cpu-haswell"
    assert (world.lib / "libomp.dll").read_bytes() == b"openmp"
    assert json.loads((world.lib / ".localm-cpu-overlay").read_text(encoding="utf-8")) == {
        "tag": sl._ROCM_CPU_TAG, "variant": "ggml-cpu-haswell.dll"}
    assert (world.lib / "LICENSE.llama-cpp").read_text(encoding="utf-8") == sl._LLAMA_CPP_MIT_NOTICE
    assert world.marker() == f"amd-rocm {sl._ROCM_BUILD}\n"
    assert world.history() == [("amd-rocm", sl._ROCM_BUILD)]
    assert world.requests == [f"{API}/{LEMONADE}/releases/tags/{ROCM}", urls[name], cpu_url]
    assert world.probe_calls == 2
    assert sl.blas_kernel_problems(world.lib) == []
    assert sl.check_runtime_update()["newer"] is False


def _amd_rocm_release(world):
    world.vendors, world.gpu_names = ["amd"], "amd radeon rx 6900 xt"
    world.release(LEMONADE, ROCM, [(f"llama-{ROCM}-windows-rocm-gfx103X-x64.zip",
                                    _zip(AMD_ROCM), "ok")])


def _assert_kept_the_amd_rocm_cpu_backend(world, r, why):
    assert r.exit_code == 0, r.output
    _in_order(r.text, f"Warning: the SIMD CPU backend for the amd-rocm build was not installed "
              f"({why}", "Retry with localm setup-llama --backend amd-rocm --force.",
              "OK - amd-rocm runtime loads on this machine.")
    assert (world.lib / "ggml-cpu.dll").read_bytes() == b"ggml-cpu@rocm"
    assert not (world.lib / "libomp.dll").exists()
    assert not (world.lib / "libomp140.x86_64.dll").exists()
    assert not (world.lib / "ggml-cpu.dll.amd-rocm").exists()
    assert not (world.lib / ".localm-cpu-overlay").is_file()
    assert world.marker() == f"amd-rocm {ROCM}\n"
    assert sl.check_runtime_update()["newer"] is True


def test_amd_rocm_cpu_overlay_copies_the_legacy_openmp_runtime_name(world):
    _amd_rocm_release(world)
    files = {k: v for k, v in ROCM_CPU.items() if k != "libomp.dll"}
    files["libomp140.x86_64.dll"] = b"openmp-legacy"
    _rocm_cpu_overlay(world, files=files)
    r = world.invoke("--backend", "amd-rocm")
    assert r.exit_code == 0, r.output
    assert (world.lib / "ggml-cpu.dll").read_bytes() == b"cpu-haswell"
    assert (world.lib / "libomp140.x86_64.dll").read_bytes() == b"openmp-legacy"
    assert not (world.lib / "libomp.dll").exists()
    assert world.marker() == f"amd-rocm {sl._ROCM_BUILD}\n"


def test_amd_rocm_cpu_overlay_rollback_keeps_an_openmp_runtime_the_build_ships(world):
    world.vendors, world.gpu_names = ["amd"], "amd radeon rx 6900 xt"
    world.release(LEMONADE, ROCM, [(f"llama-{ROCM}-windows-rocm-gfx103X-x64.zip",
                                    _zip({**AMD_ROCM, "libomp.dll": b"omp@rocm"}), "ok")])
    _rocm_cpu_overlay(world)
    world.probes = [(0, ""),
                    (1, "OSError: [WinError 127] The specified procedure could not be found")]
    r = world.invoke("--backend", "amd-rocm")
    assert r.exit_code == 0, r.output
    assert (world.lib / "ggml-cpu.dll").read_bytes() == b"ggml-cpu@rocm"
    assert (world.lib / "libomp.dll").read_bytes() == b"omp@rocm"
    assert not (world.lib / ".localm-cpu-overlay").is_file()
    assert world.marker() == f"amd-rocm {ROCM}\n"


def test_amd_rocm_cpu_overlay_that_does_not_load_is_rolled_back(world):
    _amd_rocm_release(world)
    _rocm_cpu_overlay(world)
    world.probes = [(0, ""),
                    (1, "OSError: [WinError 127] The specified procedure could not be found")]
    r = world.invoke("--backend", "amd-rocm")
    _assert_kept_the_amd_rocm_cpu_backend(
        world, r, "the runtime did not load with it (OSError: [WinError 127] The specified "
                  "procedure could not be found)")
    assert world.probe_calls == 3


def test_amd_rocm_cpu_overlay_whose_marker_cannot_be_written_is_rolled_back(world):
    _amd_rocm_release(world)
    _rocm_cpu_overlay(world)
    world.lib.mkdir(parents=True, exist_ok=True)
    (world.lib / ".localm-cpu-overlay").mkdir()
    r = world.invoke("--backend", "amd-rocm")
    _assert_kept_the_amd_rocm_cpu_backend(world, r, "")
    assert (world.lib / ".localm-cpu-overlay").is_dir()
    assert world.probe_calls == 3


def test_amd_rocm_build_that_does_not_load_gets_no_cpu_overlay(world):
    _amd_rocm_release(world)
    cpu_url = _rocm_cpu_overlay(world)
    _upstream_win_vulkan(world)
    world.probes = [(1, "OSError: [WinError 126] The specified module could not be found"),
                    (0, "")]
    r = world.invoke("--backend", "amd-rocm", "--yes")
    assert cpu_url not in world.requests
    assert "SIMD CPU backend" not in r.text


def test_amd_rocm_cpu_overlay_download_failure_keeps_the_build(world):
    _amd_rocm_release(world)
    r = world.invoke("--backend", "amd-rocm")
    _assert_kept_the_amd_rocm_cpu_backend(world, r, "ArtifactError: the connection was "
                                                    "interrupted after 0 of an unknown number "
                                                    "of bytes (HTTP Error 404: Not Found)")
    assert world.probe_calls == 2


def test_amd_rocm_cpu_overlay_checksum_mismatch_installs_nothing(world):
    _amd_rocm_release(world)
    _rocm_cpu_overlay(world, digest_ok=False)
    r = world.invoke("--backend", "amd-rocm")
    _assert_kept_the_amd_rocm_cpu_backend(world, r, "ArtifactError")
    assert world.probe_calls == 2


def test_amd_rocm_cpu_overlay_with_no_supported_variant_keeps_the_build(world):
    _amd_rocm_release(world)
    _rocm_cpu_overlay(world, scores={"ggml-cpu-x64.dll": 0, "ggml-cpu-haswell.dll": None})
    r = world.invoke("--backend", "amd-rocm")
    _assert_kept_the_amd_rocm_cpu_backend(
        world, r, "no CPU backend variant reports support for this CPU")
    assert world.probe_calls == 2


def test_amd_rocm_cpu_overlay_that_cannot_be_rolled_back_is_not_reported_as_loading(world):
    _amd_rocm_release(world)
    _rocm_cpu_overlay(world)
    _upstream_win_vulkan(world)
    real_replace = os.replace

    def _backup_locked(src, dst):
        if str(src).endswith("ggml-cpu.dll.amd-rocm"):
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_replace(src, dst)

    world.mp.setattr(os, "replace", _backup_locked)
    world.probes = [(0, ""),
                    (1, "OSError: [WinError 127] The specified procedure could not be found"),
                    (1, "OSError: [WinError 127] The specified procedure could not be found"),
                    (0, "")]
    r = world.invoke("--backend", "amd-rocm", "--yes")
    assert world.marker().startswith("vulkan "), r.output
    assert "ggml-cpu.dll.amd-rocm" not in world.files()
    assert "OK - amd-rocm runtime loads" not in r.text
    _in_order(r.text, "Warning: the SIMD CPU backend for the amd-rocm build was not installed",
              "'amd-rocm' backend provisioned but failed to load: OSError: [WinError 127]",
              "OK - vulkan runtime loads.")
    assert world.probe_calls == 4


@pytest.mark.parametrize("cap, cuda, line, ver, need, blackwell", [
    ("8.9", "12.5", "cuda-12", "12.4", "12.4", ""),
    ("12.0", "13.4", "cuda-13", "13.4", "13.4", " (Blackwell)"),
])
def test_windows_cuda_pairs_the_build_with_its_cudart_bundle(world, cap, cuda, line, ver, need,
                                                             blackwell):
    world.vendors = ["nvidia"]
    world.nvidia = ("555.85", cuda, "NVIDIA GeForce RTX 4090", cap)
    build = f"llama-{PIN}-bin-win-{line}-x64.zip".replace(line, f"cuda-{ver}")
    cudart = f"cudart-llama-bin-win-cuda-{ver}-x64.zip"
    other = "13.4" if ver == "12.4" else "12.4"
    urls = world.release(UPSTREAM, PIN, [
        (cudart, _zip(WIN_CUDART, 6), "ok"),
        (f"cudart-llama-bin-win-cuda-{other}-x64.zip", _zip({"z.dll": b"z"}, 7), "ok"),
        (f"llama-{PIN}-bin-win-cuda-{other}-x64.zip", _zip({"llama.dll": b"z"}, 8), "ok"),
        (build, _zip(WIN_CUDA, 5), "ok"),
    ])
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              "CUDA selected (peak NVIDIA performance). Checking your system...",
              "OK NVIDIA GPU: NVIDIA GeForce RTX 4090",
              f"Compute capability {cap}{blackwell} -> {line} line",
              f"OK Driver 555.85 supports CUDA {cuda} (need >= {need} for the {line} line)",
              "Fetching self-contained CUDA runtime bundle. No Toolkit needed.",
              "Backend: cuda (NVIDIA CUDA build + self-contained runtime)",
              f"CUDA build: {build} (0 MB)", f"Downloading {urls[build]}",
              f"CUDA runtime: {cudart} (0 MB) - no Toolkit install needed",
              f"Downloading {urls[cudart]}",
              "OK - cuda runtime loads on this machine.")
    assert world.requests == [f"{API}/{UPSTREAM}/releases/tags/{PIN}", urls[build], urls[cudart]]
    assert world.files() == [".localm-backend", "LICENSE.llama-cpp", "cublas64_12.dll",
                             "cublasLt64_12.dll", "cudart64_12.dll", "ggml-cuda.dll", "ggml.dll",
                             "llama.dll"]
    assert world.marker() == f"cuda {PIN}\n"


def test_linux_cuda_uses_the_third_party_build_and_pypi_runtime_wheels(world):
    world.platform("linux")
    world.vendors = ["nvidia"]
    world.nvidia = ("555.42", "12.5", "NVIDIA RTX A4000", "8.6")
    build = f"llama-{PIN}-bin-ubuntu-cuda-x64.tar.gz"
    urls = world.release(HYBRID, PIN, [
        (f"llama-{PIN}-bin-ubuntu-cuda-13-x64.tar.gz", _targz({"x/libllama.so": b"13"}, 3), "ok"),
        (build, _targz({"b/libllama.so": b"libllama", "b/libggml-cuda.so": b"ggml-cuda"}, 4), "ok"),
    ])
    wheels = {}
    for pkg, libs in (("nvidia-cuda-runtime-cu12", {"nvidia/cuda_runtime/lib/libcudart.so.12": b"rt"}),
                      ("nvidia-cublas-cu12", {"nvidia/cublas/lib/libcublas.so.12": b"b",
                                              "nvidia/cublas/lib/libcublasLt.so.12": b"lt"})):
        body = _zip({**libs, "nvidia/METADATA": b"meta"}, len(pkg))
        wurl = world.serve(f"https://files.pythonhosted.org/packages/{pkg}.whl", body)
        wheels[pkg] = wurl
        world.serve_json(f"https://pypi.org/pypi/{pkg}/json", {
            "info": {"version": "12.9.79"},
            "releases": {"12.9.79": [
                {"filename": f"{pkg}-12.9.79-py3-none-manylinux_2_27_aarch64.whl",
                 "url": "https://files.pythonhosted.org/aarch64.whl", "digests": {"sha256": "0"}},
                {"filename": f"{pkg}-12.9.79-py3-none-manylinux_2_27_x86_64.whl",
                 "url": wurl, "digests": {"sha256": _sha(body)}},
            ]}})
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text, "Compute capability 8.6 -> cuda-12 line",
              "Backend: cuda (NVIDIA CUDA build + self-contained runtime)",
              f"Downloading {urls[build]}",
              "Fetching CUDA runtime library: nvidia-cuda-runtime-cu12",
              "Fetching CUDA runtime library: nvidia-cublas-cu12",
              "CUDA runtime: 3 libraries fetched from PyPI - no CUDA Toolkit install needed",
              "OK - cuda runtime loads on this machine.")
    assert world.requests == [
        f"{API}/{HYBRID}/releases/tags/{PIN}", urls[build],
        "https://pypi.org/pypi/nvidia-cuda-runtime-cu12/json", wheels["nvidia-cuda-runtime-cu12"],
        "https://pypi.org/pypi/nvidia-cublas-cu12/json", wheels["nvidia-cublas-cu12"]]
    assert world.files() == [".localm-backend", "LICENSE.llama-cpp", "libcublas.so.12",
                             "libcublasLt.so.12", "libcudart.so.12", "libggml-cuda.so",
                             "libllama.so"]
    assert world.marker() == f"cuda {PIN}\n"


@pytest.mark.parametrize("sha_matches", [True, False])
def test_linux_bundles_libgomp_only_with_a_verified_package(world, monkeypatch, sha_matches):
    world.platform("linux")
    files = {"b/libllama.so": _minimal_elf64_needing("libggml-base.so.0"),
             "b/libggml-base.so.0": _minimal_elf64_needing("libgomp.so.1", "libc.so.6")}
    name = f"llama-{PIN}-bin-ubuntu-x64.tar.gz"
    urls = world.release(UPSTREAM, PIN, [(name, _targz(files), "ok")])
    so = b"THE-REAL-LIBGOMP" + _noise(30_000, 11)
    deb = _fake_libgomp_deb(so)
    world.serve(sl._LIBGOMP_DEB_URL, deb, "application/vnd.debian.binary-package")
    if sha_matches:
        monkeypatch.setattr(sl, "_LIBGOMP_DEB_SHA256", _sha(deb))
    r = world.invoke("--backend", "cpu")
    assert r.exit_code == 0, r.output
    _in_order(r.text, f"Downloading {urls[name]}",
              "Bundling libgomp.so.1 (OpenMP runtime; upstream's Linux builds link it "
              "dynamically and ship no copy of their own)",
              f"Downloading {sl._LIBGOMP_DEB_URL}", "OK - cpu runtime loads on this machine.")
    assert world.requests[-1] == sl._LIBGOMP_DEB_URL
    if sha_matches:
        assert (world.lib / "libgomp.so.1").read_bytes() == so
        assert (world.lib / "LICENSE.libgomp").read_text(encoding="utf-8") == \
            sl._LIBGOMP_LICENSE_NOTICE
    else:
        assert not (world.lib / "libgomp.so.1").exists()
        assert not (world.lib / "LICENSE.libgomp").exists()
    assert world.marker() == f"cpu {PIN}\n"


# --------------------------------------------------------------------------- #
#  Fallback order                                                              #
# --------------------------------------------------------------------------- #

def _windows_cuda_world(world):
    world.vendors = ["nvidia"]
    world.nvidia = ("555.85", "12.5", "NVIDIA GeForce RTX 4090", "8.9")
    urls = world.release(UPSTREAM, PIN, [
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", _zip(WIN_CUDART, 6), "ok"),
        (f"llama-{PIN}-bin-win-cuda-12.4-x64.zip", _zip(WIN_CUDA, 5), "ok"),
        (f"llama-{PIN}-bin-win-vulkan-x64.zip", _zip(WIN_VULKAN, 1), "ok"),
        (f"llama-{PIN}-bin-win-cpu-x64.zip", _zip(WIN_CPU, 3), "ok"),
    ])
    world.seed_target(".gitignore", ".gitkeep")
    return urls


def test_cuda_that_does_not_load_falls_back_to_vulkan_non_interactively(world):
    urls = _windows_cuda_world(world)
    world.probes = [(1, "Traceback (most recent call last):\n  File \"x\"\n"
                        "OSError: [WinError 126] The specified module could not be found\n"),
                    (0, "")]
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              "'cuda' backend provisioned but failed to load: OSError: [WinError 126] The "
              "specified module could not be found",
              "To retry later: localm setup-llama --backend cuda --force",
              "[!] Non-interactive: falling back to universal build.",
              "Trying vulkan...", "OK - vulkan runtime loads.")
    assert "Trying cpu" not in r.text
    listing = f"{API}/{UPSTREAM}/releases/tags/{PIN}"
    assert world.requests == [listing, urls[f"llama-{PIN}-bin-win-cuda-12.4-x64.zip"],
                              urls["cudart-llama-bin-win-cuda-12.4-x64.zip"], listing,
                              urls[f"llama-{PIN}-bin-win-vulkan-x64.zip"]]
    assert world.files() == WIN_VULKAN_FILES
    assert world.marker() == f"vulkan {PIN}\n"
    assert world.history() == [("vulkan", PIN)]


def test_fallback_continues_to_cpu_when_vulkan_does_not_load(world):
    _windows_cuda_world(world)
    world.probes = [(1, "OSError: cuda broken"), (1, "OSError: vulkan broken"), (0, "")]
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text, "Trying vulkan...",
              "vulkan provisioned but failed to load: OSError: vulkan broken",
              "Trying cpu...", "OK - cpu runtime loads.")
    assert world.files() == WIN_CPU_FILES
    assert world.marker() == f"cpu {PIN}\n"


def test_nothing_loading_raises_a_reportable_error_naming_every_attempt(world):
    _windows_cuda_world(world)
    world.probes = [(1, "OSError: A"), (1, "OSError: B"), (1, "OSError: C")]
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 1
    assert isinstance(r.exception, LocalmError)
    assert r.exception.summary == "no llama.cpp backend could be provisioned and loaded"
    assert r.exception.reason == (
        "none of 3 backends loaded on this machine - cuda: OSError: A; vulkan: OSError: B; "
        "cpu: OSError: C. You can provide a local build with: localm setup-llama --from "
        "<build dir>, or see docs/gpu-setup.md.")
    assert r.exception.context == {"operation": "setup-llama", "requested_backend": "cuda"}
    assert world.marker() is None
    assert world.history() == []


@pytest.mark.parametrize("vendors, lines", [
    (["amd"], ["Heads up: Picked cuda but detected amd. Proceeding. Hardware must be present.",
               "Could not run nvidia-smi - this machine looks like AMD, not NVIDIA.",
               "--yes: using vulkan (AMD detected, not NVIDIA)."]),
    ([], ["Could not run nvidia-smi - no NVIDIA driver detected here (or it is not on PATH).",
          "--yes: using Vulkan (no NVIDIA GPU detected)."]),
])
def test_cuda_without_nvidia_uses_vulkan_under_yes(world, vendors, lines):
    world.vendors, world.recommended = vendors, "vulkan"
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text, *lines, "Backend: vulkan", "OK - vulkan runtime loads on this machine.")
    assert world.marker() == f"vulkan {PIN}\n"


def test_cuda_with_a_too_old_driver_uses_vulkan(world):
    world.vendors = ["nvidia"]
    world.nvidia = ("470.00", "11.8", "NVIDIA GeForce GTX 1080", "6.1")
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "cuda", "--yes")
    assert r.exit_code == 0, r.output
    _in_order(r.text, "no Driver 470.00 supports CUDA 11.8 (need >= 12.4 for the cuda-12 line)",
              "GPU driver update required for CUDA (the cuda-12 line this GPU needs).",
              "To enable later: update driver, reboot, run setup-llama --backend cuda",
              "Using Vulkan now.", "Backend: vulkan")
    assert world.marker() == f"vulkan {PIN}\n"


def test_a_universal_backend_that_does_not_load_names_the_missing_library(world):
    world.platform("linux")
    name = f"llama-{PIN}-bin-ubuntu-vulkan-x64.tar.gz"
    world.release(UPSTREAM, PIN, [(name, _targz({"b/libllama.so": b"x", "b/libggml.so": b"y"}), "ok")])
    world.probes = [(1, "Traceback (most recent call last):\n  File \"x\"\n"
                        "RuntimeError: Failed to load libllama.so from /x: libvulkan.so.1: cannot "
                        "open shared object file: No such file or directory\n"
                        "  localm setup-llama --backend amd-rocm --force  (AMD RX 6000)\n")]
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text,
              "'vulkan' was provisioned but the native library did not load.",
              "Missing OS library: libvulkan.so.1 - on Debian/Ubuntu: sudo apt install libvulkan1 "
              "(or your GPU vendor's Vulkan ICD/driver package)",
              "Fix the cause and retry with: localm setup-llama --backend vulkan --force - or "
              "provide your own build with --from <build dir>. See docs/gpu-setup.md.")
    assert "Trying" not in r.text
    assert world.marker() is None and world.history() == []
    assert world.probe_calls == 1


def test_a_runtime_with_no_compute_backend_is_a_failed_provision(world):
    world.release(UPSTREAM, PIN, [(f"llama-{PIN}-bin-win-cpu-x64.zip", _zip(WIN_CPU), "ok")])
    world.probes = [(sl._PROBE_NO_BACKENDS, "")]
    r = world.invoke("--backend", "cpu")
    assert r.exit_code == 1
    _in_order(r.text, "'cpu' was provisioned but the native library did not load.",
              'Cause: runtime loaded but registered no compute backends ("no backends are '
              'loaded") - this build does not fit this machine')


# --------------------------------------------------------------------------- #
#  Download verification                                                       #
# --------------------------------------------------------------------------- #

ESCAPE_HATCH = ("If your network blocks or filters this download (common on a corporate "
                "network), download the archive yourself through a browser and use --from "
                "<extracted-dir>, or point --url at a mirror your network allows. Retry the same "
                "command once the cause is fixed: localm setup-llama --backend vulkan")


def _download_failure(world, **kw):
    world.seed_target(".gitignore", "stale-old.dll")
    url, body = _upstream_win_vulkan(world, **kw)
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    assert world.probe_calls == 0
    assert world.files() == [".gitignore"]
    assert world.marker() is None and world.history() == []
    return r, url, body


def test_a_digest_mismatch_is_refused(world):
    bad = "sha256:" + "0" * 64
    body = _zip(WIN_VULKAN)
    r, url, _ = _download_failure(world, digest=bad, body=body)
    _in_order(r.text, f"Downloading {url}",
              f"Provisioning vulkan failed: download sha256 does not match the expected pin "
              f"(expected {'0' * 64}, got {_sha(body)}). Refusing to install a possibly tampered "
              "or wrong artifact.", ESCAPE_HATCH)


def test_an_html_block_page_is_diagnosed_as_a_filtered_download(world):
    page = b"<!DOCTYPE html><html><body>Blocked by proxy</body></html>"
    world.seed_target(".gitignore")
    url = world.serve(f"https://github.com/{UPSTREAM}/releases/download/{PIN}/"
                      f"llama-{PIN}-bin-win-vulkan-x64.zip", page, "text/html; charset=utf-8")
    world.serve_json(f"{API}/{UPSTREAM}/releases/tags/{PIN}", {"assets": [
        {"name": f"llama-{PIN}-bin-win-vulkan-x64.zip", "browser_download_url": url,
         "digest": "sha256:" + _sha(page)}]})
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text,
              f"Provisioning vulkan failed: download is too small ({len(page)} bytes < 262144 "
              "minimum): the response is an HTML page, not the archive - almost always a network "
              "that blocks or filters this download (a corporate proxy or security product), not "
              f"a problem with the release itself ({len(page)} bytes received; {len(page)} "
              f"expected (Content-Length); Content-Type: text/html; charset=utf-8; final URL: "
              f"{url}).", ESCAPE_HATCH)


def test_a_non_archive_body_is_refused(world):
    blob = _noise(300_000, 42)
    r, url, _ = _download_failure(world, body=blob)
    assert (f"Provisioning vulkan failed: download is not a valid zip or tar archive: the content "
            "does not clearly indicate the cause - it is neither a recognisable webpage nor a "
            f"valid archive (300000 bytes received; 300000 expected (Content-Length); "
            f"Content-Type: application/octet-stream; final URL: {url}).") in r.text


@pytest.mark.parametrize("exc, message", [
    (ConnectionResetError("reset by peer"),
     "Provisioning vulkan failed: the connection was interrupted after 0 of an unknown number of "
     "bytes (reset by peer) - this looks like a dropped or flaky connection, not a blocked "
     "download. Retry, or provision from a local build with 'localm setup-llama --from "
     "<build-dir>' / '--url <archive-url>'."),
    (TimeoutError("timed out"),
     f"Provisioning vulkan failed: download stalled (no data for {sl._DOWNLOAD_STALL_TIMEOUT}s, "
     "after 0 of an unknown number of bytes) - the connection was interrupted or throttled. Retry "
     "on a stable network, or provision from a local build with 'localm setup-llama --from "
     "<build-dir>' / '--url <archive-url>'."),
    (RedirectDowngradeRefused("redirect to http://mirror.invalid refused"),
     "Provisioning vulkan failed: refused to follow this download off HTTPS (<urlopen error "
     "redirect to http://mirror.invalid refused>) - the archive would have arrived in cleartext, "
     "where anything on the network path can replace it, and its bytes are loaded as a native "
     "library. This is not a transient network fault: check the URL, or provision from a local "
     "build with 'localm setup-llama --from <build-dir>'."),
])
def test_transport_failures_are_reported_by_kind(world, exc, message):
    url, _ = _upstream_win_vulkan(world)
    world.serve(url, exc)
    world.seed_target(".gitignore")
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text, message, ESCAPE_HATCH)


def test_offline_default_install_uses_the_pinned_asset_name_and_checksum(world):
    name = next(n for n in sl._PINNED_FALLBACK_SHA256
                if n.startswith(f"llama-{PIN}-") and "bin-win-vulkan-x64" in n)
    guessed = world.serve(f"https://github.com/{UPSTREAM}/releases/download/{PIN}/{name}",
                          _zip(WIN_VULKAN))
    world.seed_target(".gitignore")
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text,
              f"Could not verify release asset list; using unverified URL: {guessed}",
              "If download fails, pass --from <build dir> or --url <archive>.",
              f"Downloading {guessed}",
              "download sha256 does not match the expected pin (expected "
              f"{sl._PINNED_FALLBACK_SHA256[name]}, got", ESCAPE_HATCH)
    assert world.requests == [f"{API}/{UPSTREAM}/releases/tags/{PIN}", guessed]


def test_an_explicit_sha256_mismatch_never_falls_back(world):
    _upstream_win_vulkan(world)
    world.seed_target(".gitignore")
    r = world.invoke("--backend", "vulkan", "--sha256", "f" * 64)
    assert r.exit_code == 1
    _in_order(r.text, f"download sha256 does not match the expected pin (expected {'f' * 64}",
              "The pinned artifact failed validation. Not falling back (an explicit --sha256 was "
              "set).")
    assert "If your network blocks" not in r.text
    assert len(world.requests) == 2


def test_an_explicit_sha256_is_case_and_whitespace_insensitive(world):
    body = _zip(WIN_VULKAN)
    _upstream_win_vulkan(world, digest=None, body=body)
    r = world.invoke("--backend", "vulkan", "--sha256", f"  {_sha(body).upper()} ")
    assert r.exit_code == 0, r.output
    assert "publishes no checksum" not in r.text
    assert world.marker() == f"vulkan {PIN}\n"


# --------------------------------------------------------------------------- #
#  Build and pin resolution                                                    #
# --------------------------------------------------------------------------- #

def test_tag_pins_an_exact_build_and_warns_when_it_has_no_checksum(world):
    url, _ = _upstream_win_vulkan(world, tag="b10355", digest=None)
    r = world.invoke("--backend", "vulkan", "--tag", "b10355")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              f"Pinned llama.cpp b10355 - setup-llama and localm update will keep this build until "
              f"you run localm setup-llama --tag default (back to localm's confirmed {PIN}).",
              "Warning: this release asset publishes no checksum, so the download's integrity is "
              "not cryptographically verified (its size and archive shape are still checked). "
              "Pass --sha256 <hex> to pin one.", f"Downloading {url}",
              "OK - vulkan runtime loads on this machine.")
    assert world.pin() == "b10355"
    assert world.marker() == "vulkan b10355\n"
    assert world.history() == [("vulkan", "b10355")]


def _serve_recent_releases(world):
    world.serve_json(f"{API}/{UPSTREAM}/releases?per_page=10", [
        {"tag_name": "b99999", "draft": True, "assets": [{"name": "x"}]},
        {"tag_name": "b99998", "assets": []},
        {"tag_name": "nightly", "assets": [{"name": "x"}]},
        {"tag_name": "b99997", "prerelease": True, "assets": [{"name": "x"}]},
        {"tag_name": "b99996", "assets": [{"name": "x"}]},
    ])


def test_tag_latest_tracks_the_newest_uploaded_release(world):
    _serve_recent_releases(world)
    url, _ = _upstream_win_vulkan(world, tag="b99997")
    r = world.invoke("--backend", "vulkan", "--tag", "latest")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              "Now tracking upstream's newest llama.cpp release. That build is whatever ggml-org "
              "published most recently and localm has NOT tested it; upstream has shipped "
              "releases this code cannot load. Go back with: localm setup-llama --tag default "
              f"({PIN}).", f"Downloading {url}")
    assert world.requests == [f"{API}/{UPSTREAM}/releases?per_page=10",
                              f"{API}/{UPSTREAM}/releases/tags/b99997", url]
    assert world.pin() == "latest"
    assert world.marker() == "vulkan b99997\n"


def test_tag_latest_with_no_uploaded_release_installs_the_pin(world):
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "vulkan", "--tag", "latest")
    assert r.exit_code == 0, r.output
    assert ("No ggml-org/llama.cpp release with an uploaded build was found among the most "
            f"recent releases. Installing localm's confirmed build {PIN} instead - rerun later "
            "for upstream's newest.") in r.text
    assert world.marker() == f"vulkan {PIN}\n"


ABI_ERR = "AbiMismatch: llama_context_params size 120 != 128"
ABI_DETAIL = f"the runtime does not match this build's struct layout: {ABI_ERR}"


def test_an_abi_rejected_latest_release_floors_at_the_pin(world):
    _serve_recent_releases(world)
    _upstream_win_vulkan(world, tag="b99997")
    pin_url, _ = _upstream_win_vulkan(world)
    world.probes = [(sl._PROBE_ABI_MISMATCH, ABI_ERR), (0, "")]
    r = world.invoke("--backend", "vulkan", "--tag", "latest")
    assert r.exit_code == 0, r.output
    _in_order(r.text,
              f"llama.cpp b99997 was rejected by localm's own ABI check: {ABI_DETAIL}",
              "That is a mismatch between the release and this build of localm, not a fault of "
              "your machine.",
              f"Falling back to {PIN}, the confirmed build localm ships.",
              f"Downloading {pin_url}",
              f"OK - llama.cpp {PIN} loads on this machine.",
              f"Installed {PIN} rather than b99997: you asked to track upstream's newest (--tag "
              "latest) and that release does not match this build of localm. Update localm and "
              "re-run 'localm setup-llama --force' to move forward again.")
    assert world.marker() == f"vulkan {PIN}\n"
    assert world.history() == [("vulkan", PIN)]
    assert world.pin() == "latest"


def test_an_abi_rejected_pin_has_no_floor_below_it(world):
    _upstream_win_vulkan(world)
    world.probes = [(sl._PROBE_ABI_MISMATCH, ABI_ERR)]
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text,
              f"llama.cpp {PIN} was rejected by localm's own ABI check: {ABI_DETAIL}",
              f"{PIN} is the confirmed build localm ships, so there is no more-tested build to "
              "fall back to.",
              "This means this localm and its own pinned llama.cpp build disagree, which is a bug "
              "in localm rather than in the release. Please report it. To try another build "
              "meanwhile: localm setup-llama --tag <release> (for example --tag b10361).",
              "'vulkan' was provisioned but the native library did not load.",
              f"Cause: {ABI_DETAIL}")
    assert world.probe_calls == 1


def test_an_abi_rejected_user_pin_is_kept(world):
    sl.set_pinned_tag("b10355")
    _upstream_win_vulkan(world, tag="b10355")
    world.probes = [(sl._PROBE_ABI_MISMATCH, ABI_ERR)]
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    _in_order(r.text,
              f"The pinned llama.cpp build b10355 does not load on this machine: {ABI_DETAIL}",
              "Your pin is kept, not changed. Move it with: localm setup-llama --rollback "
              "(previous build), localm setup-llama --tag default (the build localm ships and "
              "confirmed), or localm setup-llama --tag latest (track upstream).")
    assert world.pin() == "b10355"


def test_rollback_installs_and_pins_the_previous_recorded_build(world):
    world.seed_target("llama.dll", ".gitignore", marker=f"vulkan {PIN}")
    config.update_config(lambda c: c.__setitem__("llama_runtime_history", [
        {"backend": "vulkan", "tag": "b10000", "at": 1},
        {"backend": "cpu", "tag": "b9000", "at": 2},
        {"backend": "vulkan", "tag": PIN, "at": 3}]))
    url, _ = _upstream_win_vulkan(world, tag="b10000")
    r = world.invoke("--rollback")
    assert r.exit_code == 0, r.output
    _in_order(r.text, "Rolling back the vulkan runtime to llama.cpp b10000, and pinning it.",
              f"Downloading {url}", "OK - vulkan runtime loads on this machine.")
    assert "Already provisioned" not in r.text
    assert world.pin() == "b10000"
    assert world.marker() == "vulkan b10000\n"
    assert world.history() == [("vulkan", "b10000"), ("cpu", "b9000"), ("vulkan", PIN),
                               ("vulkan", "b10000")]


def test_tag_default_clears_the_pin_and_reinstalls_the_confirmed_build(world):
    sl.set_pinned_tag("b10355")
    world.seed_target("llama.dll", marker="vulkan b10355")
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "vulkan", "--tag", "default")
    assert r.exit_code == 0, r.output
    assert (f"Back to localm's confirmed build ({PIN}) - the one this release was tested with. "
            "Re-run setup-llama --force to install it now.") in r.text
    assert world.pin() is None
    assert world.marker() == f"vulkan {PIN}\n"


# --------------------------------------------------------------------------- #
#  Already provisioned, --from, --url, the provisioning lock                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("marker, args, label", [
    (f"vulkan {PIN}", ["--backend", "vulkan"], f" (vulkan {PIN})"),
    ("cpu", [], " (cpu)"),
    (None, [], ""),
])
def test_an_already_provisioned_runtime_is_left_alone(world, marker, args, label):
    world.seed_target("llama.dll", marker=marker)
    before = world.files()
    r = world.invoke(*args)
    assert r.exit_code == 0, r.output
    assert r.text == f"Already provisioned{label} at {world.lib}"
    assert world.requests == [] and world.probe_calls == 0
    assert world.files() == before


@pytest.mark.parametrize("marker, message", [
    ("cpu b1", "Replacing cpu build with vulkan."),
    (None, "Replacing unrecorded build with vulkan."),
])
def test_an_explicit_backend_replaces_a_different_install(world, marker, message):
    world.seed_target("llama.dll", ".gitignore", ".gitkeep", marker=marker)
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 0, r.output
    _in_order(r.text, message, "Backend: vulkan", "OK - vulkan runtime loads on this machine.")
    assert world.files() == WIN_VULKAN_FILES


@pytest.mark.parametrize("marker, backend, message", [
    ("amd-rocm b1288", "amd-rocm", f"Upgrading the amd-rocm build: b1288 -> {sl._ROCM_BUILD}."),
    (f"amd-rocm {ROCM}", "amd-rocm",
     f"Upgrading the amd-rocm build: {ROCM} -> {sl._ROCM_BUILD}."),
    ("vulkan b10000", "vulkan", "Re-downloading the vulkan build (b10000)."),
    ("vulkan", "vulkan", "Re-downloading the vulkan build."),
])
def test_interactive_replace_names_what_is_being_replaced(world, monkeypatch, capsys,
                                                          marker, backend, message):
    world.seed_target("llama.dll", marker=marker)
    world.vendors, world.gpu_names = ["amd"], "amd radeon rx 6900 xt"
    world.release(LEMONADE, ROCM, [(f"llama-{ROCM}-windows-rocm-gfx103X-x64.zip",
                                    _zip(AMD_ROCM), "ok")])
    _upstream_win_vulkan(world)
    prompts = []
    monkeypatch.setattr(click, "confirm", lambda text, **kw: prompts.append(text) or True)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    sl.main.callback(from_dir=None, backend=backend, url=None, sha256=None, force=False,
                     tag=None, rollback=False, assume_yes=False)
    text = " ".join(capsys.readouterr().out.split())
    _in_order(text, "Already provisioned", "Replacing existing build...", message,
              f"Backend: {backend}", f"OK - {backend} runtime loads on this machine.")
    assert prompts == ["Do you want to re-download/replace them?"]


def test_from_a_local_build_copies_libraries_and_marks_custom(world, tmp_path):
    src = tmp_path / "build" / "bin"
    src.mkdir(parents=True)
    for name, data in (("llama.dll", b"l"), ("ggml.dll", b"g"), ("ggml-vulkan.dll", b"v"),
                       ("llama-server.exe", b"e"), ("README.md", b"r"), ("LICENSE", b"MIT")):
        (src / name).write_bytes(data)
    r = world.invoke("--from", str(src))
    assert r.exit_code == 0, r.output
    _in_order(r.text, f"Copying binaries from {src} ...", f"Copied 3 file(s) into {world.lib}",
              "OK - the provided build loads on this machine.",
              f"Native runtime ready -> {world.lib}")
    assert world.files() == [".localm-backend", "LICENSE.llama-cpp", "ggml-vulkan.dll",
                             "ggml.dll", "llama.dll"]
    assert world.marker() == "custom\n"
    assert world.history() == [] and world.requests == []


def test_from_a_directory_without_the_library_is_refused(world, tmp_path):
    src = tmp_path / "empty-build"
    src.mkdir()
    (src / "ggml.dll").write_bytes(b"g")
    r = world.invoke("--from", str(src))
    assert r.exit_code == 1
    assert ("No llama.dll found in the source directory. Point --from at the build output "
            "containing llama.dll.") in r.text
    assert world.probe_calls == 0


def test_from_a_build_that_does_not_load_is_not_reported_as_success(world, tmp_path):
    src = tmp_path / "b"
    src.mkdir()
    (src / "llama.dll").write_bytes(b"l")
    world.probes = [(1, "OSError: bad image")]
    r = world.invoke("--from", str(src))
    assert r.exit_code == 1
    assert ("Copied, but the library did not load (OSError: bad image) - is it built for this "
            "OS/GPU? See docs/gpu-setup.md.") in r.text
    assert world.marker() is None


def test_url_download_without_sha256_warns_and_marks_custom(world):
    url = world.serve("https://mirror.example.invalid/llama-custom.zip", _zip(WIN_VULKAN))
    r = world.invoke("--url", url)
    assert r.exit_code == 0, r.output
    _in_order(r.text, "Warning: Custom URL download is unverified (no --sha256 provided).",
              f"Fetching: {url}", f"Downloading {url}",
              "OK - the fetched build loads on this machine.")
    assert world.marker() == "custom\n"
    assert world.requests == [url]


def test_url_download_with_matching_sha256_does_not_warn(world):
    body = _zip(WIN_VULKAN)
    url = world.serve("https://mirror.example.invalid/llama-custom.zip", body)
    r = world.invoke("--url", url, "--sha256", _sha(body))
    assert r.exit_code == 0, r.output
    assert "unverified" not in r.text


@pytest.mark.parametrize("serve, message", [
    (None, "Refusing to install: the connection was interrupted after 0 of an unknown number of "
           "bytes (HTTP Error 404: Not Found) - this looks like a dropped or flaky connection, "
           "not a blocked download. Retry, or provision from a local build with 'localm "
           "setup-llama --from <build-dir>' / '--url <archive-url>'. Provide a local build with "
           "--from instead, or a different --url (and --sha256 if you pin one)."),
    ("no-library",
     "The archive did not contain llama.dll. Try a different --url or use --from."),
], ids=["404", "no-library"])
def test_url_failures_exit_non_zero(world, serve, message):
    url = "https://mirror.example.invalid/llama-custom.zip"
    if serve == "no-library":
        world.serve(url, _zip({"ggml.dll": b"g"}))
    r = world.invoke("--url", url)
    assert r.exit_code == 1
    assert message in r.text
    assert world.marker() is None


@pytest.mark.parametrize("owner, message", [
    ("live", "Cannot provision the runtime right now: Another setup-llama run is already "
             "provisioning the runtime (process {pid}). Wait for it to finish, then try again."),
    ("unreadable", "Cannot provision the runtime right now: A provisioning lock exists at {lock} "
                   "but its owner could not be read. If no setup-llama run is in progress, "
                   "remove that folder and try again."),
])
def test_a_held_provisioning_lock_leaves_the_install_untouched(world, owner, message):
    world.seed_target("llama.dll", marker="cpu b1")
    lock = world.lib.parent / "lib.setup.lock"
    lock.mkdir()
    if owner == "live":
        (lock / "owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    _upstream_win_vulkan(world)
    r = world.invoke("--backend", "vulkan")
    assert r.exit_code == 1
    assert message.format(pid=os.getpid(), lock=lock) in r.text
    assert world.files() == [".localm-backend", "llama.dll"]
    assert world.marker() == "cpu b1\n"
    assert world.requests == []
    assert lock.is_dir()


# --------------------------------------------------------------------------- #
#  Asset resolution without a release listing                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("plat, backend, line, fragment", [
    ("win32", "cpu", None, "bin-win-cpu-x64"),
    ("win32", "vulkan", None, "bin-win-vulkan-x64"),
    ("win32", "cuda", "cuda-12", "bin-win-cuda-12"),
    ("win32", "cuda", "cuda-13", "bin-win-cuda-13"),
    ("win32", "sycl", None, "bin-win-sycl-x64"),
    ("win32", "hip", None, "bin-win-rocm"),
    ("linux", "cpu", None, "bin-ubuntu-x64"),
    ("linux", "vulkan", None, "bin-ubuntu-vulkan-x64"),
    ("linux", "sycl", None, "bin-ubuntu-sycl-fp16"),
    ("linux", "hip", None, "bin-ubuntu-rocm"),
    ("darwin", "cpu", None, "bin-macos-arm64"),
    ("darwin", "metal", None, "bin-macos-arm64"),
])
def test_offline_resolution_reads_the_pinned_asset_table(world, plat, backend, line, fragment):
    world.platform(plat)
    url, sha, tag = sl._resolve_backend_asset(backend, line)
    name = url.rsplit("/", 1)[1]
    assert url == f"https://github.com/{UPSTREAM}/releases/download/{PIN}/{name}"
    assert name.startswith(f"llama-{PIN}-") and fragment in name and "cudart" not in name
    assert sha == sl._PINNED_FALLBACK_SHA256[name]
    assert tag == PIN


@pytest.mark.parametrize("plat, backend, message", [
    ("win32", "metal", "backend 'metal' is not available on this platform (win32). Available: "
                       "cpu, cuda, hip, sycl, vulkan."),
    ("linux", "metal", "backend 'metal' is not available on this platform (linux). Available: "
                       "cpu, cuda, hip, sycl, vulkan."),
    ("darwin", "vulkan", "backend 'vulkan' is not available on this platform (darwin). "
                         "Available: cpu, metal."),
    ("linux", "cuda", f"no Linux CUDA build found for llama.cpp tag '{PIN}' on "
                      f"{HYBRID} - falling back to vulkan."),
    ("linux", "amd-rocm", "the self-contained 'amd-rocm' build is Windows-only; on Linux use "
                          "--backend hip (needs ROCm) or build with --from."),
])
def test_unresolvable_backends_are_refused_with_a_reason(world, plat, backend, message):
    world.platform(plat)
    with pytest.raises(click.ClickException) as ei:
        sl._resolve_backend_asset(backend)
    assert ei.value.message == message


@pytest.mark.parametrize("gpu, family", [
    ("amd radeon rx 6900 xt", "gfx103X"),
    ("amd radeon rx 7900 xtx", "gfx110X"),
    ("amd radeon rx 9070 xt", "gfx120X"),
    ("amd radeon vii", "gfx103X"),
])
def test_offline_amd_rocm_resolution_per_gpu_family(world, gpu, family):
    world.gpu_names = gpu
    url, sha, tag = sl._resolve_backend_asset("amd-rocm")
    if family == "gfx103X":
        assert (url, sha) == (sl.DEFAULT_URL, sl.DEFAULT_URL_SHA256)
    else:
        name = f"llama-{ROCM}-windows-rocm-{family}-x64.zip"
        assert url == f"https://github.com/{LEMONADE}/releases/download/{ROCM}/{name}"
        assert sha == sl._PINNED_FALLBACK_SHA256[name]
    assert tag is None
    assert world.requests == [f"{API}/{LEMONADE}/releases/tags/{ROCM}"]


# --------------------------------------------------------------------------- #
#  Read-only API used by the GUI, doctor, the updater and the bug reporter    #
# --------------------------------------------------------------------------- #

def test_check_runtime_update_states(world):
    assert sl.check_runtime_update() == {"installed": False, "backend": None, "current": None,
                                         "target": None, "newer": False, "pinned": None,
                                         "previous": None}
    world.seed_target("llama.dll", marker="vulkan b10000")
    config.update_config(lambda c: c.__setitem__("llama_runtime_history", [
        {"backend": "vulkan", "tag": "b9000", "at": 1}, {"backend": "vulkan", "tag": "..", "at": 2}]))
    assert (sl.installed_backend(), sl.installed_build()) == ("vulkan", "b10000")
    assert sl.check_runtime_update() == {"installed": True, "backend": "vulkan",
                                         "current": "b10000", "target": PIN, "newer": True,
                                         "pinned": None, "previous": "b9000"}
    sl.set_pinned_tag("b10000")
    assert sl.check_runtime_update()["target"] == "b10000"
    assert sl.check_runtime_update()["newer"] is False
    assert sl.check_runtime_update()["pinned"] == "b10000"
    sl.set_pinned_tag("latest")
    _serve_recent_releases(world)
    assert sl.check_runtime_update()["target"] == "b99997"
    assert (sl.pinned_tag(), sl.tracks_latest()) == (None, True)
    (world.lib / ".localm-backend").write_text("amd-rocm b1288\n", encoding="utf-8")
    assert sl.check_runtime_update()["target"] == sl._ROCM_BUILD
    (world.lib / ".localm-backend").write_text(f"amd-rocm {ROCM}\n", encoding="utf-8")
    assert sl.check_runtime_update()["newer"] is True
    (world.lib / ".localm-backend").write_text(f"amd-rocm {sl._ROCM_BUILD}\n",
                                               encoding="utf-8")
    assert sl.check_runtime_update()["newer"] is False


def test_an_unsafe_stored_pin_is_ignored_out_loud(world, capsys):
    config.update_config(lambda c: c.__setitem__("llama_runtime_pin", "../../x"))
    assert sl.pinned_tag() is None
    out = " ".join(capsys.readouterr().out.split())
    assert ("Warning: ignoring the stored llama.cpp pin '../../x' - it is not a usable release "
            "tag. Set one with localm setup-llama --tag <tag>.") in out


def test_tag_safety_predicate():
    assert sl.TAG_HELP == ("Use a tag as upstream publishes it, for example 'b10355' (letters, "
                           "digits, dot, dash and underscore only), or 'default' for the build "
                           "localm ships and confirmed, or 'latest' for upstream's newest.")
    for ok in ("b10355", "v1.2.3", "a_b-c.d", "x" * 64):
        assert sl.is_safe_tag(ok), ok
    for bad in ("", " ", "../x", "a/b", "a..b", "-x", "a?b", "a#b", "x" * 65, None):
        assert not sl.is_safe_tag(bad), bad


def test_blas_kernel_problems(world):
    world.seed_target("llama.dll")
    assert sl.blas_kernel_problems(world.lib) == []
    (world.lib / "rocblas.dll").write_bytes(b"r")
    assert sl.blas_kernel_problems(world.lib) == [
        "rocblas is installed but its rocblas/ kernel directory is missing entirely"]
    (world.lib / "rocblas").mkdir()
    assert sl.blas_kernel_problems(world.lib) == [
        "rocblas is installed but its rocblas/ kernel directory is empty"]
    (world.lib / "rocblas" / "library").mkdir()
    (world.lib / "rocblas" / "library" / "k.dat").write_bytes(b"k")
    assert sl.blas_kernel_problems(world.lib) == []
    assert sl.blas_kernel_problems(world.lib / "absent") == []


def test_runtime_lib_dir_falls_back_to_the_repo_when_the_wheel_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "localm_llama_runtime", None)
    import localm
    repo = Path(localm.__file__).resolve().parent.parent
    assert sl._repo_runtime_lib() == repo / "runtime" / "localm_llama_runtime" / "lib"
    assert sl._runtime_pkg_dir() == repo / "runtime"
