#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a llama.cpp release LOADS AND GENERATES before it is pinned.

``localm setup-llama`` installs ``_PINNED_TAG`` - one release we decided on -
rather than whatever upstream published most recently. This script is what earns
that word "confirmed": a build that merely LOADS is not confirmed, because a
release can load fine for upstream and still be refused by localm's own ABI gate.

WHAT IT DOES, per backend:

  1. places that backend's build for the tag through setup-llama's own
     ``_provision_backend`` (a Windows CUDA build together with its cudart bundle,
     as an install fetches it) into a THROWAWAY directory, and records every
     archive fetched and whether its sha256 was checked against a published or
     pinned digest;
  2. hashes the shared core libraries (the llama library, ggml-base, ggml) so the
     summary states whether they are byte-identical to the cpu build's;
  3. loads it in a fresh interpreter through localm's real loader (which runs the
     ABI gate) and records the compute devices it registered; for a GPU backend it
     also loads the backend's own ggml plugin and, when that fails on Windows,
     names the plugin's imported DLLs that cannot be found;
  4. loads a small real GGUF through the product's own GgufBackend, in the
     isolated worker process a real chat uses, and asks it known-answer questions
     with greedy sampling.

VERDICTS. PASS: loaded, the ABI gate accepted it, every fetched archive matched
its digest, and generation answered every known-answer question, in the build
under test. For a GPU backend PASS additionally needs llama.cpp's own load report
in the worker (``weight_placement``) to put model weights in a buffer of a GPU
device the runtime registered; a ``--backend`` run whose weights stayed in system
RAM is FAIL with ``--require-gpu`` and INCONCLUSIVE without it. FAIL: the build is bad (a
digest mismatch, the ABI gate refused it, no compute backend, a model that does
not load, a wrong answer). INCONCLUSIVE: not measured (no network, no model, an
override that did not apply, an unreadable worker, an unverified archive).
Exit code 0 / 1 / 2 in that order.

``--also-measure BACKEND`` runs the same steps for a backend whose result is
reported but does not have to PASS: its FAIL still makes the run FAIL, while a
backend that registers no GPU device here, or whose archive targets another
platform, is NOT MEASURED with the reason, and one whose weights stayed in system
RAM is INCONCLUSIVE. ``cuda`` measures both CUDA lines.
``metal`` off macOS fetches and digest-checks the macOS archive and reports the
library inside it, which cannot be loaded on this platform.

The build under test is injected with ``LLAMA_CPP_LIB``. The probe asserts
``runtime_binary_dir()`` resolves inside the throwaway dir, and the worker's
mapped modules are read back (psutil) to check the worker loaded the same
library; either check failing is INCONCLUSIVE, never PASS. Nothing here touches
the provisioned runtime.

Usage:
    python scripts/confirm_llama_runtime.py --tag b10375
    python scripts/confirm_llama_runtime.py --tag b10375 --backend cpu --backend vulkan --require-gpu
    python scripts/confirm_llama_runtime.py --backend cpu --also-measure cuda --also-measure hip
    python scripts/confirm_llama_runtime.py            # confirm the current pin on cpu

Needs localm importable (unlike scripts/check_llama_pin.py, which is stdlib-only).
Nothing under localm/ imports this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

PASS, FAIL, INCONCLUSIVE, NOT_MEASURED = "PASS", "FAIL", "INCONCLUSIVE", "NOT MEASURED"

SCHEMA = 1

_ALL_BACKENDS = ("cpu", "vulkan", "cuda", "sycl", "hip", "metal")
_GPU_BACKENDS = frozenset({"vulkan", "cuda", "sycl", "hip", "metal"})
_CUDA_LINES = ("cuda-12", "cuda-13")

# Each question, and the word its answer must contain. Every one is answered
# correctly by the model above under greedy sampling on cpu, vulkan and hip.
KNOWN_ANSWERS = (
    ("What is the capital of France? Answer with one word.", "paris"),
    ("Which planet do we live on? Answer with one word.", "earth"),
)

_SHA_MISMATCH_TEXT = "does not match the expected pin"


def answer_check(answers) -> tuple[bool, str]:
    """(ok, why) for *answers*, the replies to :data:`KNOWN_ANSWERS` in order. ok
    only when there is one reply per question and each contains its expected word
    as a whole word, case-insensitively."""
    if not isinstance(answers, list) or len(answers) != len(KNOWN_ANSWERS):
        return False, f"expected {len(KNOWN_ANSWERS)} replies, got {answers!r}"
    for (question, word), text in zip(KNOWN_ANSWERS, answers, strict=True):
        if not re.search(rf"\b{word}\b", str(text), re.IGNORECASE):
            return False, (f"asked {question!r}, expected {word!r}, got {str(text)[:160]!r}")
    return True, "every known-answer question answered correctly"


# --------------------------------------------------------------------------- #
#  The probe: runs in a fresh interpreter with LLAMA_CPP_LIB pointed at the    #
#  extracted build. Its stdout's last @@VERDICT@@ line is a JSON result whose  #
#  "verdict" is FAIL, INCONCLUSIVE, LOADED (generation not attempted) or       #
#  GENERATED; the parent turns it into the final verdict (judge).             #
# --------------------------------------------------------------------------- #
_PROBE = r'''
import ctypes, json, os, sys
from pathlib import Path

out = {"verdict": "INCONCLUSIVE", "why": "", "abi": None, "compute_backends": None,
       "lib_dir": None, "worker_lib": None, "devices": None, "plugin": None,
       "missing_imports": None, "answers": None, "placement": None, "gpu_layers": None}
expected = Path(os.environ["LOCALM_CONFIRM_EXPECT_DIR"]).resolve()
model = os.environ.get("LOCALM_CONFIRM_MODEL", "")
plugin = os.environ.get("LOCALM_CONFIRM_PLUGIN", "")
imports = json.loads(os.environ.get("LOCALM_CONFIRM_PLUGIN_IMPORTS") or "[]")
gpu_backend = os.environ.get("LOCALM_CONFIRM_GPU_BACKEND") == "1"
prompts = json.loads(os.environ["LOCALM_CONFIRM_PROMPTS"])

def emit(**kw):
    out.update(kw)
    sys.stdout.write("\n@@VERDICT@@" + json.dumps(out) + "\n")
    sys.stdout.flush()
    raise SystemExit(0)

try:
    from localm.inference.backends.llamacpp import _loader
    from localm.inference.backends.llamacpp._abi import AbiMismatch
except Exception as e:
    emit(verdict="INCONCLUSIVE", why="localm is not importable here: %s" % e)

# 1. Load, then check the loaded library is the build under test.
try:
    _loader.load_lib()
except AbiMismatch as e:
    out["lib_dir"] = str(_loader.runtime_binary_dir())
    emit(verdict="FAIL", why="localm's ABI gate refused this build: %s" % e, abi="mismatch")
except Exception as e:
    emit(verdict="FAIL", why="the native library did not load: %s" % e)

got = _loader.runtime_binary_dir()
out["lib_dir"] = str(got) if got else None
try:
    resolved_ok = got is not None and Path(got).resolve() == expected
except Exception:
    resolved_ok = False
if not resolved_ok:
    emit(verdict="INCONCLUSIVE",
         why="LLAMA_CPP_LIB did not take effect - loaded %r, expected %r. Nothing "
             "was measured about the build under test." % (out["lib_dir"], str(expected)))

# 2. The ABI verdict and whether any compute backend registered.
try:
    from localm.inference.backends.llamacpp._abi import abi_report
    out["abi"] = abi_report().status
except Exception as e:
    out["abi"] = "unavailable: %s" % e
try:
    out["compute_backends"] = bool(_loader.compute_backends_available())
except Exception as e:
    emit(verdict="INCONCLUSIVE",
         why="could not tell whether any compute backend registered: %s" % e)
if out["abi"] == "mismatch":
    emit(verdict="FAIL", why="abi_report() reports a mismatch against this build")
if out["compute_backends"] is not True:
    emit(verdict="FAIL",
         why='the runtime registered no compute backends ("no backends are loaded")')

# 3. The devices the runtime registered, as (name, is-GPU).
try:
    out["devices"] = [{"name": str(n), "gpu": int(t) == _loader.GGML_DEV_TYPE_GPU}
                      for n, t in (_loader.compute_devices() or [])]
except Exception as e:
    out["devices"] = "unavailable: %s" % e

# 4. The backend's own ggml plugin, and which of its imports cannot be found.
if plugin:
    path = expected / plugin
    if not path.exists():
        out["plugin"] = "absent: the archive carries no %s" % plugin
    else:
        try:
            ctypes.CDLL(str(path))
            out["plugin"] = "loaded"
        except OSError as e:
            msg = (str(e).splitlines() or [type(e).__name__])[0]
            if "Could not find module" in msg:
                msg = "it or a library it needs is missing"
            out["plugin"] = "did not load: %s" % msg
            missing = []
            for name in imports:
                try:
                    ctypes.CDLL(name)
                except OSError:
                    missing.append(name)
            out["missing_imports"] = missing

gpus = [d["name"] for d in out["devices"] if d.get("gpu")] if isinstance(out["devices"], list) else []
if gpu_backend and not gpus:
    emit(verdict="LOADED", why="no GPU device registered, so generation was not attempted")
if not model:
    emit(verdict="LOADED", why="no model to generate with: %s"
         % os.environ.get("LOCALM_CONFIRM_MODEL_ERR", "not fetched"))

# 5. GENERATE, keeping the worker's load metadata (its weight_placement report).
try:
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.llamacpp import _runner
except Exception as e:
    emit(verdict="INCONCLUSIVE", why="GgufBackend is not importable: %s" % e)

load_meta = {}
_spawn_and_load = _runner.ModelRunner.spawn_and_load

def _recording_spawn_and_load(self, *a, **k):
    meta = _spawn_and_load(self, *a, **k)
    load_meta["meta"] = meta
    return meta

_runner.ModelRunner.spawn_and_load = _recording_spawn_and_load

be = GgufBackend(model, n_ctx=2048)
try:
    be.load()
except Exception as e:
    emit(verdict="FAIL", why="a real GGUF failed to load on this build: %s" % e)

try:
    import psutil
    pid = be._runner._proc.pid
    name = os.environ["LOCALM_CONFIRM_LIB_NAME"].lower()
    mapped = [m.path for m in psutil.Process(pid).memory_maps()
              if Path(m.path).name.lower() == name]
    out["worker_lib"] = mapped
    if not mapped:
        be.unload()
        emit(verdict="INCONCLUSIVE",
             why="could not read which %s the worker process mapped" % name)
    if not all(Path(p).resolve().parent == expected for p in mapped):
        be.unload()
        emit(verdict="INCONCLUSIVE",
             why="the worker mapped a DIFFERENT %s (%r) - the generation below "
                 "would not have been this build's" % (name, mapped))
except SystemExit:
    raise
except Exception as e:
    be.unload()
    emit(verdict="INCONCLUSIVE", why="could not inspect the worker process: %s" % e)

out["gpu_layers"] = getattr(be, "effective_gpu_layers", None)
placement = (load_meta.get("meta") or {}).get("weight_placement")
out["placement"] = placement if isinstance(placement, list) else None

answers = []
try:
    for prompt in prompts:
        answers.append("".join(be.chat_stream(
            [{"role": "user", "content": prompt}],
            max_tokens=40, temperature=0.0, seed=1)).strip())
except Exception as e:
    out["answers"] = answers
    emit(verdict="FAIL", why="generation raised: %s" % e)
finally:
    try:
        be.unload()
    except Exception:
        pass
emit(verdict="GENERATED", why="loaded and generated", answers=answers)
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_model() -> tuple[str, str]:
    """(path, error). Never raises: no model is INCONCLUSIVE, not a failure of
    the build under test."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        return "", f"huggingface_hub is not installed ({e})"
    try:
        return hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE), ""
    except Exception as e:
        return "", f"could not fetch {_MODEL_REPO}/{_MODEL_FILE}: {e}"


# --------------------------------------------------------------------------- #
#  Windows PE import table                                                     #
# --------------------------------------------------------------------------- #

def pe_imports(path: Path) -> list[str]:
    """The DLL names a Windows PE image imports, normal and delay-load, in table
    order without duplicates. [] when *path* is not a readable PE32/PE32+ image."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    try:
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        n_sections = struct.unpack_from("<H", data, pe + 6)[0]
        opt_size = struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", data, opt)[0]
        dirs = opt + (112 if magic == 0x20B else 96 if magic == 0x10B else -1)
        if dirs < opt:
            return []
        n_dirs = struct.unpack_from("<I", data, dirs - 4)[0]
        sections = []
        sec = opt + opt_size
        for i in range(n_sections):
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, sec + 40 * i + 8)
            sections.append((va, max(vsize, rsize), raw))

        def offset(rva: int) -> int:
            for va, size, raw in sections:
                if va <= rva < va + size:
                    return raw + rva - va
            raise ValueError(f"rva {rva:#x} outside every section")

        def cstr(rva: int) -> str:
            start = offset(rva)
            return data[start:data.index(b"\0", start)].decode("ascii", "replace")

        names: list[str] = []
        for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
            if index >= n_dirs:
                continue
            rva = struct.unpack_from("<I", data, dirs + 8 * index)[0]
            if not rva:
                continue
            try:
                at = offset(rva)
            except ValueError:
                continue
            while True:
                desc = data[at:at + size]
                if len(desc) < size or not any(desc):
                    break
                name_rva = struct.unpack_from("<I", desc, name_at)[0]
                try:
                    name = cstr(name_rva) if name_rva else ""
                except ValueError:
                    name = ""
                if name and name not in names:
                    names.append(name)
                at += size
        return names
    except struct.error:
        return []


# --------------------------------------------------------------------------- #
#  Placing a build                                                             #
# --------------------------------------------------------------------------- #

def _place_build(backend: str, tag: str, dest: Path, cuda_line: str | None = None,
                 attempts: int = 3) -> tuple[str, list, str, str]:
    """Place *backend* at *tag* into *dest* through setup-llama's
    ``_provision_backend``. Returns (lib_path, fetched, error, error_kind):
    *fetched* lists ``{"asset", "sha256"}`` for every archive placed (sha256 None
    when no digest was available to check it against); *error_kind* is
    "mismatch" for an archive whose sha256 differs from its digest, "" otherwise.

    The whole placement is retried up to *attempts* times on any other error."""
    from localm import setup_llama as sl
    last, kind = "", ""
    fetched: list = []
    real_fetch = sl._fetch_and_place

    def recording_fetch(url, target, sha256=None):
        fetched.append({"asset": str(url).rsplit("/", 1)[-1], "sha256": sha256 or None})
        return real_fetch(url, target, sha256)

    for attempt in range(1, attempts + 1):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        fetched.clear()
        sl._fetch_and_place = recording_fetch
        try:
            sl._provision_backend(backend, dest, None, with_cudart=True,
                                  cuda_line=cuda_line or sl._CUDA_LINE, tag=tag)
            last = ""
            break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if _SHA_MISMATCH_TEXT in str(e):
                kind = "mismatch"
                break
            if attempt < attempts:
                print(f"  placing {backend} failed (attempt {attempt}/{attempts}: {e}); "
                      "retrying", flush=True)
        finally:
            sl._fetch_and_place = real_fetch
    if last:
        return "", list(fetched), f"could not place the {backend} build for {tag}: {last}", kind
    lib = dest / sl._lib_name()
    if not lib.exists():
        return "", list(fetched), f"the {backend} build for {tag} contained no {sl._lib_name()}", ""
    return str(lib), list(fetched), "", ""


def _place_foreign(backend: str, tag: str, dest: Path) -> dict:
    """Fetch and digest-check *backend*'s archive for another platform (metal, the
    macOS arm64 archive) without installing it. Returns ``{"asset", "sha256",
    "error", "error_kind", "lib", "lib_sha256"}``: *lib* is the llama library
    found inside the archive, "" when none."""
    from localm import setup_llama as sl
    from localm.setup_llama.pins import _ASSET_MATCH, _PINNED_FALLBACK_SHA256
    info = {"asset": "", "sha256": None, "error": "", "error_kind": "", "lib": "",
            "lib_sha256": ""}
    plat = {"metal": "darwin"}.get(backend, "")
    matchers = _ASSET_MATCH.get(plat, {}).get(backend) or []
    lib_name = {"darwin": "libllama.dylib"}.get(plat, "")
    asset = next((a for a in sl._release_assets(tag)
                  if any(m in str(a.get("name", "")).lower() for m in matchers)
                  and a.get("browser_download_url")), None)
    if asset is None:
        name = next((n for n in sorted(_PINNED_FALLBACK_SHA256)
                     if n.startswith(f"llama-{tag}-") and any(m in n for m in matchers)), "")
        if not name:
            info["error"] = f"no {backend} archive found in the {tag} release listing"
            return info
        asset = {"name": name, "browser_download_url":
                 f"https://github.com/{sl._UPSTREAM_REPO}/releases/download/{tag}/{name}"}
    digest = str(asset.get("digest") or "")
    sha = digest.split("sha256:")[-1].strip() if "sha256:" in digest else None
    info["asset"] = asset["name"]
    info["sha256"] = sha or _PINNED_FALLBACK_SHA256.get(asset["name"])
    dest.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if asset["name"].lower().endswith(".zip") else ".tar.gz"
    arc = dest / f"archive{suffix}"
    try:
        dl = sl._download(asset["browser_download_url"], arc)
        sl._validate_archive(arc, expected_sha256=info["sha256"], dl=dl)
        sl._extract_archive(arc, dest / "x")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["error_kind"] = "mismatch" if _SHA_MISMATCH_TEXT in str(e) else ""
        return info
    found = sorted((dest / "x").rglob(lib_name)) if lib_name else []
    if found:
        info["lib"] = str(found[0])
        info["lib_sha256"] = _sha256(found[0])
    return info


_CORE_LIBS = {"win32": ("ggml-base.dll", "ggml.dll"),
              "darwin": ("libggml-base.dylib", "libggml.dylib")}


def _core_hashes(lib: Path) -> dict:
    """{file name: sha256} of the llama library *lib* and the ggml core libraries
    beside it, for those that exist."""
    plat = "win32" if sys.platform == "win32" else "darwin" if sys.platform == "darwin" else "linux"
    names = [lib.name, *_CORE_LIBS.get(plat, ("libggml-base.so", "libggml.so"))]
    return {n: _sha256(lib.parent / n) for n in names if (lib.parent / n).is_file()}


def _plugin_name(backend: str) -> str:
    """The file name of *backend*'s ggml plugin library on this platform, "" for cpu."""
    if backend not in _GPU_BACKENDS:
        return ""
    stem = f"ggml-{backend}"
    if sys.platform == "win32":
        return f"{stem}.dll"
    return f"lib{stem}.dylib" if sys.platform == "darwin" else f"lib{stem}.so"


def _run_probe(lib: Path, model: str, *, backend: str = "cpu", model_err: str = "") -> dict:
    """Run :data:`_PROBE` against *lib*. Returns its result dict; INCONCLUSIVE when
    it could not run or emitted no result."""
    env = dict(os.environ)
    env["LLAMA_CPP_LIB"] = str(lib)
    env["LOCALM_CONFIRM_EXPECT_DIR"] = str(lib.parent.resolve())
    env["LOCALM_CONFIRM_LIB_NAME"] = lib.name
    env["LOCALM_CONFIRM_MODEL"] = model
    env["LOCALM_CONFIRM_MODEL_ERR"] = model_err
    env["LOCALM_CONFIRM_GPU_BACKEND"] = "1" if backend in _GPU_BACKENDS else "0"
    env["LOCALM_CONFIRM_PROMPTS"] = json.dumps([q for q, _ in KNOWN_ANSWERS])
    plugin = _plugin_name(backend)
    env["LOCALM_CONFIRM_PLUGIN"] = plugin
    env["LOCALM_CONFIRM_PLUGIN_IMPORTS"] = json.dumps(
        pe_imports(lib.parent / plugin) if plugin and sys.platform == "win32" else [])
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                           capture_output=True, text=True, timeout=900)
    except Exception as e:
        return {"verdict": INCONCLUSIVE, "why": f"the probe could not be run: {e}"}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@VERDICT@@"):
            try:
                return json.loads(line[len("@@VERDICT@@"):])
            except Exception:
                break
    tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-6:])
    return {"verdict": INCONCLUSIVE,
            "why": f"the probe emitted no verdict (exit {r.returncode}): {tail}"}


# --------------------------------------------------------------------------- #
#  Judging                                                                     #
# --------------------------------------------------------------------------- #

def _gpu_names(res: dict) -> list[str]:
    devices = res.get("devices")
    return [d["name"] for d in devices if d.get("gpu")] if isinstance(devices, list) else []


def _plugin_detail(res: dict) -> str:
    plugin, missing = res.get("plugin"), res.get("missing_imports")
    if not plugin:
        return ""
    text = f"plugin {plugin}"
    if missing:
        text += f"; imports not found on this machine: {', '.join(missing)}"
    return text


def judge(backend: str, res: dict, *, require_gpu: bool = False,
          best_effort: bool = False) -> tuple[str, str]:
    """(verdict, why) for one backend's probe result *res*.

    The probe's own FAIL and INCONCLUSIVE are kept. A GPU backend that registered
    no GPU device is NOT MEASURED when *best_effort*, else FAIL with *require_gpu*
    and INCONCLUSIVE without it. A generation is FAIL when any known-answer reply
    is wrong. A GPU backend's generation is PASS only when llama.cpp's load report
    put weights (more than 0 MiB) in a non-RAM buffer named after a registered GPU
    device; weights only in RAM are FAIL with *require_gpu* unless *best_effort*,
    else INCONCLUSIVE; no load report is INCONCLUSIVE."""
    verdict = res.get("verdict")
    why = str(res.get("why") or "")
    if verdict in (FAIL, INCONCLUSIVE):
        return verdict, why
    if verdict not in ("LOADED", "GENERATED"):
        return INCONCLUSIVE, f"the probe reported {verdict!r}: {why}"
    gpus = _gpu_names(res)
    is_gpu = backend in _GPU_BACKENDS
    devices = res.get("devices")
    seen = ", ".join(d["name"] for d in devices) if isinstance(devices, list) else str(devices)
    if verdict == "LOADED":
        if is_gpu and not gpus:
            reason = (f"the {backend} build registered no GPU device here (devices: "
                      f"{seen or 'none'})" + (f"; {_plugin_detail(res)}" if _plugin_detail(res) else ""))
            if best_effort:
                return NOT_MEASURED, f"generation not measured: {reason}"
            return (FAIL if require_gpu else INCONCLUSIVE), f"no GPU generation: {reason}"
        return INCONCLUSIVE, f"generation not measured: {why}"
    ok, answer_why = answer_check(res.get("answers"))
    if not ok:
        return FAIL, f"wrong answer: {answer_why}"
    if not is_gpu:
        return PASS, f"loaded and generated; {answer_why}"
    placement = res.get("placement")
    if not placement:
        return INCONCLUSIVE, ("generated, but llama.cpp's load report of where the weights "
                              "went was not captured, so GPU use is not shown")
    on_gpu = [b for b in placement if isinstance(b, dict) and not b.get("is_ram")
              and b.get("backend") in gpus and float(b.get("mib") or 0) > 0]
    if not on_gpu:
        held = ", ".join(f"{b.get('backend')} {b.get('mib')} MiB" for b in placement
                         if isinstance(b, dict))
        reason = (f"generated on the CPU: every model buffer was in system RAM ({held}) although "
                  f"the runtime registered {', '.join(gpus)}")
        return (FAIL if require_gpu and not best_effort else INCONCLUSIVE), reason
    where = ", ".join(f"{b['backend']} {b['mib']} MiB" for b in on_gpu)
    return PASS, f"generated on the GPU (weights in {where}); {answer_why}"


def _measure(key: str, backend: str, tag: str, dest: Path, model: str, model_err: str, *,
             cuda_line: str | None, require_gpu: bool, best_effort: bool) -> dict:
    """Place, hash, probe and judge one backend. Returns its result dict with
    ``verdict``, ``why``, ``fetched``, ``core_sha256`` and the probe's fields."""
    from localm import setup_llama as sl
    from localm.setup_llama.pins import _ASSET_MATCH
    print(f"\n=== {key} @ {tag} ===", flush=True)
    plat = sl._platform_key()
    if backend not in _ASSET_MATCH.get(plat, {}):
        if backend == "metal" and best_effort:
            info = _place_foreign(backend, tag, dest)
            res = {"fetched": [{"asset": info["asset"], "sha256": info["sha256"]}] if info["asset"] else [],
                   "foreign": info}
            if info["error"]:
                res["verdict"] = FAIL if info["error_kind"] == "mismatch" else INCONCLUSIVE
                res["why"] = info["error"]
            elif not info["sha256"]:
                res["verdict"], res["why"] = INCONCLUSIVE, f"{info['asset']} has no digest to check"
            elif not info["lib"]:
                res["verdict"], res["why"] = FAIL, f"{info['asset']} contains no llama library"
            else:
                res["verdict"] = NOT_MEASURED
                res["why"] = (f"{info['asset']} matched its digest and carries "
                              f"{Path(info['lib']).name} (sha256 {info['lib_sha256'][:16]}); "
                              f"a macOS library cannot be loaded or run on {plat}")
            return res
        verdict = NOT_MEASURED if best_effort else INCONCLUSIVE
        return {"verdict": verdict, "why": f"no {backend} build exists for {plat}", "fetched": []}
    lib, fetched, err, kind = _place_build(backend, tag, dest, cuda_line)
    res: dict = {"fetched": fetched}
    if err:
        res.update(verdict=FAIL if kind == "mismatch" else INCONCLUSIVE, why=err)
        print(f"  {res['verdict']}: {err}", flush=True)
        return res
    unverified = [f["asset"] for f in fetched if not f.get("sha256")]
    res["core_sha256"] = _core_hashes(Path(lib))
    print(f"  placed {lib}  sha256[:16]={res['core_sha256'].get(Path(lib).name, '')[:16]}", flush=True)
    res.update(_run_probe(Path(lib), model, backend=backend, model_err=model_err))
    res["verdict"], res["why"] = judge(backend, res, require_gpu=require_gpu,
                                       best_effort=best_effort)
    if unverified and res["verdict"] == PASS:
        res["verdict"] = INCONCLUSIVE
        res["why"] = f"no digest to check {', '.join(unverified)} against; {res['why']}"
    print(f"  {res['verdict']}: {res['why']}", flush=True)
    if res.get("answers"):
        print(f"  answers: {res['answers']!r}", flush=True)
    return res


def _also_plan(also: list[str], backends: list[str]) -> list[tuple[str, str, str | None]]:
    """(key, backend, cuda_line) for each --also-measure backend not already in
    *backends*; cuda expands to one entry per CUDA line."""
    plan = []
    for backend in dict.fromkeys(also):
        if backend in backends:
            continue
        if backend == "cuda":
            plan += [(line, "cuda", line) for line in _CUDA_LINES]
        else:
            plan.append((backend, backend, None))
    return plan


def confirm(tag: str, backends: list[str], workdir: Path, *, also: list[str] = (),
            require_gpu: bool = False) -> dict:
    """Measure every backend in *backends* (required) and *also* (best effort).
    Returns ``{"tag", "results", "also", "lib_sha256", "core_sha256"}``."""
    model, model_err = _fetch_model()
    results: dict = {}
    extra: dict = {}
    for backend in backends:
        results[backend] = _measure(backend, backend, tag, workdir / f"build-{backend}", model,
                                    model_err, cuda_line=None, require_gpu=require_gpu,
                                    best_effort=False)
    for key, backend, line in _also_plan(list(also), backends):
        extra[key] = _measure(key, backend, tag, workdir / f"also-{key.replace('.', '')}", model,
                              model_err, cuda_line=line, require_gpu=require_gpu, best_effort=True)
        shutil.rmtree(workdir / f"also-{key.replace('.', '')}", ignore_errors=True)
    core = {k: r["core_sha256"] for k, r in {**results, **extra}.items() if r.get("core_sha256")}
    from localm import setup_llama as sl
    libs = {k: c[sl._lib_name()] for k, c in core.items() if sl._lib_name() in c}
    return {"tag": tag, "results": results, "also": extra, "lib_sha256": libs, "core_sha256": core}


# --------------------------------------------------------------------------- #
#  Checks, report, receipt                                                     #
# --------------------------------------------------------------------------- #

def _identity(core: dict, key: str) -> tuple[str, str]:
    """(status, detail) for whether *key*'s core libraries are byte-identical to cpu's."""
    mine, ref = core.get(key), core.get("cpu")
    if not mine:
        return NOT_MEASURED, "no library was placed"
    if key == "cpu":
        return PASS, "reference build"
    if not ref:
        return NOT_MEASURED, "the cpu build was not placed, so there is nothing to compare with"
    differ = [n for n, digest in mine.items() if ref.get(n) != digest]
    if differ:
        return NOT_MEASURED, f"differs from the cpu build: {', '.join(differ)}"
    return PASS, f"byte-identical to the cpu build: {', '.join(mine)}"


def _archive_check(r: dict) -> tuple[str, str]:
    """(status, detail) for the archives one backend fetched."""
    fetched = r.get("fetched") or []
    detail = "; ".join(f"{f['asset']} sha256 {'checked' if f.get('sha256') else 'NOT checked'}"
                       for f in fetched) or str(r.get("why", ""))
    if _SHA_MISMATCH_TEXT in str(r.get("why", "")):
        return FAIL, f"{detail}; {r.get('why')}"
    if not fetched:
        return NOT_MEASURED, detail
    if any(not f.get("sha256") for f in fetched):
        return INCONCLUSIVE, detail
    return PASS, detail


def build_checks(summary: dict) -> dict:
    """Schema-1 checks for *summary*, each ``{"status", "required", "detail"}``:
    first one ``<backend>:generation`` per backend (its verdict), then per backend
    ``archive``, ``abi``, ``core_identity`` and, for a GPU backend, ``gpu_device``.
    ``required`` is True for the generation, archive and abi checks of the
    backends measured with --backend, False for everything else."""
    groups = ((True, summary["results"]), (False, summary.get("also", {})))
    checks: dict = {}
    for required, group in groups:
        for key, r in group.items():
            checks[f"{key}:generation"] = {"status": r.get("verdict", INCONCLUSIVE),
                                           "required": required, "detail": str(r.get("why", ""))}
    for required, group in groups:
        for key, r in group.items():
            status, detail = _archive_check(r)
            checks[f"{key}:archive"] = {"status": status, "required": required, "detail": detail}
            abi = r.get("abi")
            checks[f"{key}:abi"] = {
                "status": NOT_MEASURED if abi is None else FAIL if abi == "mismatch" else PASS,
                "required": required,
                "detail": f"abi={abi}" if abi is not None else "not loaded here"}
            status, detail = _identity(summary.get("core_sha256", {}), key)
            checks[f"{key}:core_identity"] = {"status": status, "required": False, "detail": detail}
            if ("cuda" if key in _CUDA_LINES else key) in _GPU_BACKENDS:
                gpus = _gpu_names(r)
                checks[f"{key}:gpu_device"] = {
                    "status": PASS if gpus else NOT_MEASURED, "required": False,
                    "detail": (", ".join(gpus) if gpus else "none registered"
                               if isinstance(r.get("devices"), list) else "not loaded here")
                              + (f"; {_plugin_detail(r)}" if _plugin_detail(r) else "")}
    return checks


def exit_code(summary: dict, backends: list[str]) -> int:
    """1 when any backend (required or --also-measure) FAILED, else 2 when a
    required backend is not PASS, else 0."""
    verdicts = [summary["results"].get(b, {}).get("verdict", INCONCLUSIVE) for b in backends]
    extra = [r.get("verdict") for r in summary.get("also", {}).values()]
    if FAIL in verdicts or FAIL in extra:
        return 1
    if any(v != PASS for v in verdicts):
        return 2
    return 0


def _report(summary: dict, backends: list[str]) -> int:
    tag, results = summary["tag"], summary["results"]
    print("\n" + "=" * 72)
    print(f"CONFIRMATION SUMMARY for llama.cpp {tag}")
    print("=" * 72)
    for title, group, keys in (("required", results, backends),
                               ("also measured", summary.get("also", {}),
                                list(summary.get("also", {})))):
        if not keys:
            continue
        print(f"  [{title}]")
        for key in keys:
            r = group.get(key, {})
            print(f"  {key:10s} {r.get('verdict', INCONCLUSIVE):13s} abi={r.get('abi')} "
                  f"backends={r.get('compute_backends')}")
            devices = r.get("devices")
            if isinstance(devices, list):
                print(f"             devices={[d['name'] for d in devices]} "
                      f"gpu={_gpu_names(r) or 'NONE'} offloaded_layers={r.get('gpu_layers')}")
            if r.get("placement"):
                print("             weights: " + ", ".join(
                    f"{b.get('backend')} {b.get('mib')} MiB" for b in r["placement"]))
            if _plugin_detail(r):
                print(f"             {_plugin_detail(r)}")
            print(f"             {r.get('why', '')}")

    core = summary.get("core_sha256", {})
    if "cpu" in core and len(core) > 1:
        print("\n  core libraries against the cpu build:")
        for key in core:
            status, detail = _identity(core, key)
            print(f"    {key:10s} {status:13s} {detail}")
    elif core:
        print("\n  The cpu build was not placed, so core-library identity was NOT measured.")

    untested = sorted(set(_ALL_BACKENDS) - set(backends) - {
        "cuda" if k in _CUDA_LINES else k for k in summary.get("also", {})})
    print("\n  NOT RUN AT ALL: " + (", ".join(untested) or "none"))

    rc = exit_code(summary, backends)
    if rc == 1:
        print(f"\nRESULT: {tag} is NOT confirmed - at least one backend FAILED.")
    elif rc == 2:
        print(f"\nRESULT: {tag} is NOT confirmed - at least one required backend could not "
              "be measured. Nothing here says the build is bad; it says we did not test it.")
    else:
        print(f"\nRESULT: {tag} loaded and GENERATED the known answers on: {', '.join(backends)}.")
    return rc


def _write_receipt(path: Path, summary: dict, backends: list[str], rc: int) -> None:
    """Write the run as JSON: ``schema``, ``tag``, ``exit_code``, ``verdict`` (the
    exit code's verdict), ``why``, per-backend verdict dicts for every requested
    backend under ``backends`` and every --also-measure one under ``also``,
    schema-1 ``checks``, the llama library digest per backend, and the UTC time of
    writing. Written for every outcome, including FAIL and INCONCLUSIVE."""
    checks = build_checks(summary)
    verdict = {0: PASS, 1: FAIL}.get(rc, INCONCLUSIVE)
    failing = [f"{n}: {c['detail']}" for n, c in checks.items() if c["status"] == FAIL]
    open_ = [f"{n}: {c['detail']}" for n, c in checks.items()
             if c["required"] and c["status"] != PASS]
    receipt = {
        "schema": SCHEMA,
        "tag": summary["tag"],
        "exit_code": rc,
        "verdict": verdict,
        "why": "; ".join(failing or open_) or "every required check passed",
        "written_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backends": {b: summary["results"].get(b, {"verdict": INCONCLUSIVE,
                                                   "why": "not run"})
                     for b in backends},
        "also": summary.get("also", {}),
        "checks": checks,
        "lib_sha256": summary["lib_sha256"],
        "core_sha256": summary.get("core_sha256", {}),
    }
    path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    print(f"receipt written: {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=None,
                    help="the llama.cpp release to confirm (default: the pin in "
                         "setup_llama._PINNED_TAG)")
    ap.add_argument("--backend", action="append", dest="backends", default=None,
                    choices=list(_ALL_BACKENDS),
                    help="repeatable; must PASS. Default 'cpu', the only backend that "
                         "needs no GPU and so runs anywhere")
    ap.add_argument("--also-measure", action="append", dest="also", default=None,
                    choices=list(_ALL_BACKENDS),
                    help="repeatable; measured and reported, a FAIL fails the run, a "
                         "backend this machine cannot run is NOT MEASURED")
    ap.add_argument("--require-gpu", action="store_true",
                    help="a GPU backend that did not generate on a GPU device FAILS "
                         "instead of being INCONCLUSIVE")
    ap.add_argument("--keep", action="store_true",
                    help="keep the downloaded builds instead of deleting them")
    ap.add_argument("--workdir", default=None,
                    help="where to place the throwaway builds (default: a temp dir)")
    ap.add_argument("--receipt", default=None,
                    help="write the per-backend verdicts as JSON here; "
                         "scripts/bump_llama_pin.py reads it as the evidence that "
                         "a tag was confirmed")
    args = ap.parse_args(argv)

    try:
        from localm import setup_llama as sl
    except Exception as e:
        print(f"localm is not importable ({e}) - this script needs it installed.")
        return 2
    tag = args.tag or sl._PINNED_TAG
    backends = list(dict.fromkeys(args.backends or ["cpu"]))
    also = list(args.also or [])

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="localm-confirm-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Confirming llama.cpp {tag} on: {', '.join(backends)}"
          + (f"; also measuring: {', '.join(also)}" if also else ""))
    print(f"Throwaway build dir: {workdir}")
    try:
        summary = confirm(tag, backends, workdir, also=also, require_gpu=args.require_gpu)
        rc = _report(summary, backends)
        if args.receipt:
            _write_receipt(Path(args.receipt), summary, backends, rc)
        return rc
    finally:
        if args.keep:
            print(f"\nKept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
