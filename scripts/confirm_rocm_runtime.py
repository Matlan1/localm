#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a lemonade-sdk AMD ROCm llama.cpp build works with localm before it is pinned.

``localm setup-llama --backend amd-rocm`` installs ``_ROCM_TAG`` (a
lemonade-sdk/llamacpp-rocm release) and replaces its scalar CPU backend with the
SIMD variants of the upstream ``_ROCM_CPU_TAG`` release built from the same
commit. This script earns the word "confirmed" for a candidate tag (or for the
tag the repo pins today with ``--current``): the build is installed through
localm's own installer into a throwaway runtime, loaded through localm's own
binding, and made to generate tokens on the GPU and on the CPU.

STAGES, each a separate child process running localm's own code:

  hw       hardware detection; the box must carry an AMD RX 6000 (gfx103X) GPU.
  install  the pins for the candidate are applied exactly as scripts/bump_rocm_pin.py
           would write them, the installer's resolved URL and sha256 are compared
           with the GitHub release listing, then ``setup-llama --backend amd-rocm``
           runs into a runtime directory under ``--workdir``.
  probe    the runtime is loaded in a fresh interpreter: ABI verdict, the ggml
           commit of the loaded binary, the registered GPU device, and a real GGUF
           chat generation with all layers offloaded (video memory use measured,
           the worker's mapped libraries read back).
  probe-cpu  the same load with the GPU hidden from the process: the SIMD CPU
           backend generates tokens on its own.

ISOLATION. LOCALM_HOME, TEMP, TMP, TMPDIR and HF_HOME point under ``--workdir``;
the runtime package the installer fills is a copy of localm_llama_runtime placed
under ``--workdir`` and put first on PYTHONPATH, so the shared runtime of this
machine is never touched; LLAMA_CPP_LIB is removed from the environment. The
stage that resolves those paths reports them and the run is INCONCLUSIVE unless
every one lies under ``--workdir`` (or, for localm itself, under this checkout).
Only processes this script started, recorded by PID, are stopped, as a tree.

The caller wraps the whole run in the GPU lease. The script works without it.

EXIT CODES: 0 PASS, 1 FAIL (the build is bad for localm), 2 INCONCLUSIVE (could not
measure: network, busy or missing hardware, an unreadable result, an uncaught
exception). A required check that did not run is never rounded up to PASS.

Usage:
    python scripts/confirm_rocm_runtime.py --tag b1350 --workdir <dir> --receipt r.json
    python scripts/confirm_rocm_runtime.py --current --workdir <dir> --receipt r.json

Needs localm importable by the interpreter running it. Nothing under localm/
imports this script; it never runs from a user's install.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import bump_rocm_pin as bump  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
INCONCLUSIVE = "INCONCLUSIVE"
EXIT_CODES = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}

CHECK_NAMES = bump.REQUIRED_CHECKS

MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_REVISION = "09816acd5d99df7be770d85ea30822623dab342c"
MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
MODEL_SHA256 = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
MODEL_ARCHITECTURES = ("llama",)
PROMPT = "In one short sentence, what color is a clear daytime sky?"

MIN_FREE_BYTES = 3 * 1024 ** 3
STAGE_TIMEOUTS = {"hw": 300, "model": 300, "install": 1800, "probe": 900, "probe-cpu": 900}
INSTALL_ATTEMPTS = 3
GPU_FAMILY = "gfx103x"
GPU_DEVICE_NAME_RE = re.compile(r"^(ROCm|HIP)", re.I)
MIN_VRAM_DELTA_FRACTION = 0.5
CPU_RUN_MAX_GPU_BYTES = 64 * 2 ** 20

NOT_COVERED = [
    "GPU families other than gfx103X (their assets are pinned by digest only)",
    "the Linux ubuntu assets",
    "models other than one small dense causal LM (no MoE offload, no vision, no audio)",
    "more than one GPU",
    "long contexts and parallel slots",
]

_NET_ERROR_RE = re.compile(
    r"download stalled|Remote end closed|timed out|Connection(?:Reset|Aborted|Error)|"
    r"URLError|getaddrinfo|Name or service|HTTP Error (?:429|5\d\d)|Temporary failure|"
    r"Download failed|could not fetch", re.I)


# --------------------------------------------------------------------------- #
#  Receipt                                                                     #
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_receipt(tag: str, current: bool) -> dict:
    """A receipt in which every check is a required SKIP that has not run."""
    return {"schema": bump.RECEIPT_SCHEMA, "component": bump.COMPONENT, "tag": tag,
            "current": current, "verdict": INCONCLUSIVE, "why": "the run did not finish",
            "written_at": utc_now(), "hardware": {}, "candidate": {},
            "checks": {n: {"status": SKIP, "required": True, "detail": "not run"}
                       for n in CHECK_NAMES},
            "measured": [], "not_measured": [], "not_covered": list(NOT_COVERED)}


def set_check(receipt: dict, name: str, status: str, detail: str, *, required: bool = True) -> None:
    receipt["checks"][name] = {"status": status, "required": required, "detail": detail}


def verdict_of(receipt: dict) -> tuple[str, str]:
    """(verdict, why): FAIL when a required check FAILED, PASS when every required
    check PASSED, else INCONCLUSIVE."""
    required = {n: c for n, c in receipt["checks"].items() if c.get("required")}
    failed = sorted(n for n, c in required.items() if c["status"] == FAIL)
    if failed:
        return FAIL, "; ".join(f"{n}: {required[n]['detail']}" for n in failed)
    unmeasured = sorted(n for n, c in required.items() if c["status"] != PASS)
    if unmeasured:
        return INCONCLUSIVE, "; ".join(f"{n}: {required[n]['detail']}" for n in unmeasured)
    return PASS, "every required check passed"


def finalize(receipt: dict) -> int:
    """Set the verdict and the measured/not-measured lists; returns the exit code."""
    receipt["verdict"], receipt["why"] = verdict_of(receipt)
    checks = receipt["checks"]
    receipt["measured"] = sorted(n for n, c in checks.items() if c["status"] == PASS)
    receipt["not_measured"] = sorted(n for n, c in checks.items() if c["status"] != PASS)
    receipt["written_at"] = utc_now()
    return EXIT_CODES[receipt["verdict"]]


def write_receipt(path: Path, receipt: dict) -> None:
    """Write *receipt* as JSON to *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
#  Pure evaluation of what the stages measured                                 #
# --------------------------------------------------------------------------- #

def under(path: Any, root: Path) -> bool:
    """Whether *path* resolves to *root* or something inside it."""
    try:
        p = Path(str(path)).resolve()
        r = root.resolve()
    except (OSError, ValueError):
        return False
    return p == r or r in p.parents


def isolation_problems(report: dict, workdir: Path, repo: Path) -> list[str]:
    """Paths in the hw stage's *report* that are not where they must be."""
    problems = []
    for key in ("home_dir", "runtime_lib", "tmp_dir"):
        if not report.get(key):
            problems.append(f"{key} was not reported")
        elif not under(report[key], workdir):
            problems.append(f"{key} resolves to {report[key]}, outside the work directory")
    if not report.get("localm_file") or not under(report["localm_file"], repo):
        problems.append(f"localm imports from {report.get('localm_file')}, not from this checkout")
    if report.get("llama_cpp_lib_env"):
        problems.append("LLAMA_CPP_LIB is set in the stage environment")
    return problems


def evaluate_hardware(report: dict) -> tuple[str, str]:
    """(status, detail) for the hardware check from the hw stage's report."""
    if not report.get("probe_ok", True):
        return SKIP, f"hardware detection failed: {report.get('probe_error') or 'unknown'}"
    if "amd" not in (report.get("vendors") or []):
        return SKIP, f"no AMD GPU detected (vendors: {report.get('vendors')}, adapters: {report.get('gpu_names')!r})"
    family = report.get("amd_gfx_family") or ""
    if family != GPU_FAMILY:
        return SKIP, (f"the AMD GPU ({report.get('gpu_names')!r}) is family {family or 'unknown'}, "
                      f"not {GPU_FAMILY}; the pinned gfx103X asset cannot be measured here")
    return PASS, f"AMD {family}: {report.get('gpu_names')!r}"


def classify_install_failure(log: str) -> str:
    """"network" when the installer log shows a transport failure, else "build"."""
    return "network" if _NET_ERROR_RE.search(log or "") else "build"


def evaluate_resolve(resolved: dict, listing_ok: bool, cand: bump.Candidate) -> tuple[str, str]:
    """(status, detail) for what the installer would download versus the release listing."""
    if not listing_ok:
        return SKIP, "the installer's release listing request returned nothing (offline or rate limited)"
    want_url = (f"https://github.com/{bump.LEMONADE_REPO}/releases/download/"
                f"{cand.tag}/{cand.gfx103x_asset}")
    if resolved.get("url") != want_url:
        return FAIL, f"the installer would download {resolved.get('url')!r}, expected {want_url!r}"
    if resolved.get("sha256") != cand.gfx103x_sha256:
        return FAIL, (f"the installer would verify against sha256 {resolved.get('sha256')!r}, "
                      f"the release publishes {cand.gfx103x_sha256}")
    return PASS, f"{cand.gfx103x_asset} sha256 {cand.gfx103x_sha256[:16]}... matches the release listing"


def evaluate_install(out: dict, cand: bump.Candidate) -> tuple[str, str]:
    """(status, detail) for the setup-llama run."""
    exit_code = out.get("exit_code")
    if out.get("exception"):
        kind = classify_install_failure(out.get("log_tail", "") + out["exception"])
        status = SKIP if kind == "network" else FAIL
        return status, f"setup-llama raised {out['exception']}"
    if exit_code not in (None, 0):
        kind = classify_install_failure(out.get("log_tail", ""))
        status = SKIP if kind == "network" else FAIL
        return status, f"setup-llama exited {exit_code}: {out.get('log_tail', '')[-400:]}"
    marker = out.get("marker") or {}
    want_build = f"{cand.tag}-cpu-{cand.cpu_tag}"
    if marker.get("backend") != "amd-rocm":
        return FAIL, (f"the runtime ended up as backend {marker.get('backend')!r}: the installer "
                      "fell back instead of providing amd-rocm")
    if marker.get("build") != want_build:
        return FAIL, (f"the runtime records build {marker.get('build')!r}, expected {want_build!r} "
                      "(the SIMD CPU backend overlay was not installed)")
    overlay = out.get("overlay") or {}
    if overlay.get("tag") != cand.cpu_tag or not overlay.get("variant"):
        return FAIL, f"the CPU overlay record is {overlay!r}, expected tag {cand.cpu_tag}"
    if not out.get("llama_dll"):
        return FAIL, "the runtime directory has no llama.dll"
    return PASS, (f"installed amd-rocm {want_build} into the throwaway runtime; "
                  f"CPU variant {overlay['variant']}")


def coherent(text: str) -> bool:
    """Whether *text* reads as generated prose rather than empty or degenerate output."""
    words = re.findall(r"[A-Za-z]{2,}", text or "")
    if len(words) < 3:
        return False
    letters = sum(c.isalpha() or c.isspace() or c in ".,;:'\"!?-" for c in text)
    if letters / max(len(text), 1) < 0.8:
        return False
    return max(words.count(w) for w in set(words)) / len(words) <= 0.6


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def worker_problems(modules: list, expected_dir: Path, lib_name: str = "llama.dll") -> list[str]:
    """Why the generating worker's mapped libraries do not prove it ran the build
    under test: the llama library must be mapped from *expected_dir*, and no
    llama or ggml library may be mapped from anywhere else."""
    problems = []
    names = {Path(m).name.lower(): m for m in modules if under(m, expected_dir)}
    if lib_name.lower() not in names:
        problems.append(f"the worker did not map {lib_name} from the runtime under test")
    for m in modules:
        base = Path(m).name.lower()
        if (base.startswith(("llama", "ggml")) and base.endswith(".dll")
                and not under(m, expected_dir)):
            problems.append(f"the worker mapped {m} from outside the runtime under test")
    return problems


def evaluate_abi(probe: dict) -> tuple[str, str]:
    load = probe.get("load") or {}
    status = load.get("status")
    if status == "abi_mismatch":
        return FAIL, f"localm's ABI gate refused this build: {load.get('detail')}"
    if status == "load_error":
        return FAIL, f"the native library did not load: {load.get('detail')}"
    if status != "ok":
        return SKIP, f"the library load did not report a result ({status!r})"
    if not probe.get("runtime_dir_ok"):
        return SKIP, (f"the probe loaded {probe.get('runtime_dir')!r}, not the runtime under test; "
                      "nothing was measured about the candidate")
    abi = probe.get("abi") or {}
    if abi.get("status") == "ok":
        return PASS, (f"ABI verdict ok, model params {abi.get('layout')}, "
                      f"context params {abi.get('context_layout')}")
    if abi.get("status") == "mismatch":
        return FAIL, f"abi_report() reports a mismatch: {abi.get('failures')}"
    return SKIP, f"the ABI check did not run ({abi.get('status')}: {abi.get('detail')})"


def evaluate_identity(probe: dict, cand: bump.Candidate) -> tuple[str, str]:
    ident = probe.get("identity") or {}
    commit = (ident.get("ggml_commit") or "").strip().lower()
    if len(commit) < 7:
        return SKIP, f"the loaded binary reports no usable ggml commit ({commit!r})"
    if not cand.cpu_commit.startswith(commit) and not commit.startswith(cand.cpu_commit):
        return FAIL, (f"the loaded binary was built from {commit}, the release notes name "
                      f"{cand.lemonade_commit} and upstream {cand.cpu_tag} is {cand.cpu_commit[:12]}")
    if not commit.startswith(cand.lemonade_commit.lower()):
        return FAIL, f"the loaded binary's commit {commit} does not start with the release's {cand.lemonade_commit}"
    return PASS, f"ggml commit {commit} (ggml {ident.get('ggml_version')}) is upstream {cand.cpu_tag}"


def evaluate_gpu_device(probe: dict) -> tuple[str, str]:
    devices = probe.get("devices")
    if devices is None:
        return SKIP, "the runtime does not expose its device list"
    gpus = [d for d in devices if d.get("type") == probe.get("gpu_type")]
    hip = [d for d in gpus if GPU_DEVICE_NAME_RE.match(str(d.get("name", "")))]
    if not hip:
        return FAIL, (f"no ROCm/HIP GPU device registered (devices: "
                      f"{[(d.get('name'), d.get('description')) for d in devices]})")
    d = hip[0]
    return PASS, (f"{d['name']} {d.get('description')!r}, "
                  f"{d.get('total', 0) // 2 ** 20} MiB total")


def evaluate_model(info: dict) -> tuple[str, str]:
    if info.get("error"):
        return SKIP, info["error"]
    arch = info.get("architecture")
    if info.get("refusal") or arch not in MODEL_ARCHITECTURES:
        return FAIL, f"the test model is not a causal chat model: architecture {arch!r} {info.get('refusal') or ''}"
    return PASS, (f"{MODEL_FILE} ({info.get('bytes')} bytes, sha256 verified, architecture {arch}, "
                  f"{info.get('source')})")


def evaluate_gpu_generate(gpu: dict, ref: dict, model_bytes: int, expected_dir: Path) -> tuple[str, str]:
    """(status, detail) for the offloaded generation *gpu*. *ref* is the same
    load with no layers offloaded and the GPU visible: the worker's dedicated GPU
    memory must exceed the reference's by half the model's size, which a run that
    silently used the CPU cannot do."""
    if not gpu:
        return SKIP, "the GPU generation did not run"
    if gpu.get("skip"):
        return SKIP, gpu["skip"]
    if gpu.get("error"):
        return FAIL, f"a real GGUF failed on the GPU: {gpu['error']}"
    if not (gpu.get("effective_gpu_layers") or 0) > 0:
        return FAIL, f"no layers were offloaded (effective_gpu_layers={gpu.get('effective_gpu_layers')})"
    if not gpu.get("modules"):
        return SKIP, "the worker's mapped libraries could not be read"
    problems = worker_problems(gpu["modules"], expected_dir)
    if not any(Path(m).name.lower().startswith("ggml-hip") for m in gpu["modules"] if under(m, expected_dir)):
        problems.append("the worker did not map ggml-hip from the runtime under test")
    if problems:
        return SKIP, "; ".join(problems)
    used, base = gpu.get("worker_vram"), (ref or {}).get("worker_vram")
    if used is None or base is None:
        return SKIP, ("the per-process GPU memory counter could not be read "
                      f"(offloaded run: {used!r}, reference run: {base!r}); residency on the GPU is unproven")
    delta = used - base
    if delta < MIN_VRAM_DELTA_FRACTION * model_bytes:
        return FAIL, (f"the generating worker holds {delta // 2 ** 20} MiB more GPU memory than the same load "
                      f"with no layers offloaded; a {model_bytes // 2 ** 20} MiB model on the GPU needs at least "
                      f"{int(MIN_VRAM_DELTA_FRACTION * model_bytes) // 2 ** 20} MiB")
    if not coherent(gpu.get("text", "")):
        return FAIL, f"the GPU produced no usable text: {gpu.get('text', '')!r}"
    return PASS, (f"{gpu['effective_gpu_layers']} layers offloaded, the worker holds {used // 2 ** 20} MiB of GPU "
                  f"memory (+{delta // 2 ** 20} MiB over the no-offload reference), mapped the ROCm backend, "
                  f"generated: {gpu['text'][:80]!r}")


def evaluate_cpu_backend(out: dict, cand: bump.Candidate, probe: dict) -> tuple[str, str]:
    overlay = out.get("overlay") or {}
    variant = overlay.get("variant") or ""
    if overlay.get("tag") != cand.cpu_tag or not variant:
        return FAIL, f"the SIMD CPU overlay from upstream {cand.cpu_tag} is not installed ({overlay!r})"
    info = (probe.get("system_info") or "")
    simd = [f for f in ("AVX512", "AVX2", "AVX") if re.search(rf"\b{f} = 1", info)]
    if variant == "ggml-cpu-x64.dll" or not simd:
        return SKIP, (f"this CPU runs the baseline variant {variant} (system info: {info.strip()[:160]!r}); "
                      "the SIMD overlay is not exercised here")
    return PASS, f"CPU variant {variant} active, features {', '.join(simd)}"


def evaluate_cpu_generate(cpu: dict, expected_dir: Path, gpu_text: str = "") -> tuple[str, str]:
    if not cpu:
        return SKIP, "the CPU generation did not run"
    if cpu.get("skip"):
        return SKIP, cpu["skip"]
    if cpu.get("error"):
        return FAIL, f"a real GGUF failed on the CPU backend: {cpu['error']}"
    if cpu.get("gpu_visible"):
        return SKIP, (f"the GPU could not be hidden from the CPU run (devices: {cpu.get('devices')}); "
                      "the run is not a CPU-only measurement")
    if (cpu.get("effective_gpu_layers") or 0) != 0:
        return FAIL, f"layers were offloaded in the CPU run (effective_gpu_layers={cpu.get('effective_gpu_layers')})"
    if (cpu.get("worker_vram") or 0) > CPU_RUN_MAX_GPU_BYTES:
        return SKIP, (f"the CPU run's worker holds {cpu['worker_vram'] // 2 ** 20} MiB of GPU memory; "
                      "the run is not a CPU-only measurement")
    if not cpu.get("modules"):
        return SKIP, "the worker's mapped libraries could not be read"
    problems = worker_problems(cpu["modules"], expected_dir)
    if not any(Path(m).name.lower() == "ggml-cpu.dll" for m in cpu["modules"] if under(m, expected_dir)):
        problems.append("the worker did not map ggml-cpu.dll from the runtime under test")
    if problems:
        return SKIP, "; ".join(problems)
    if not coherent(cpu.get("text", "")):
        return FAIL, f"the CPU backend produced no usable text: {cpu.get('text', '')!r}"
    agree = common_prefix(gpu_text, cpu["text"]) if gpu_text else None
    note = "" if agree is None else f", agrees with the GPU output for {agree} characters"
    return PASS, f"CPU-only generation with no GPU visible: {cpu['text'][:80]!r}{note}"


# --------------------------------------------------------------------------- #
#  Child processes                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class ChildResult:
    returncode: Optional[int]
    out: Optional[dict]
    log_tail: str
    timed_out: bool = False


_LIVE_PIDS: set[int] = set()


def kill_tree(pid: int) -> None:
    """Stop process *pid* and everything it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, 9)
        except OSError:
            pass


def child_env(workdir: Path, extra: Optional[dict] = None) -> dict:
    """The environment of every child: isolated home, temp and cache paths under
    *workdir*, the runtime package placed there first on PYTHONPATH, this
    checkout second, and no LLAMA_CPP_LIB."""
    env = dict(os.environ)
    for name in ("LLAMA_CPP_LIB", "PYTHONHOME", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    tmp = workdir / "tmp"
    env.update({
        "LOCALM_HOME": str(workdir / "home"), "TEMP": str(tmp), "TMP": str(tmp),
        "TMPDIR": str(tmp), "HF_HOME": str(workdir / "hf"),
        "PYTHONPATH": os.pathsep.join([str(workdir / "runtime-pkg"), str(REPO)]),
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "COLUMNS": "200",
    })
    env.update(extra or {})
    return env


def run_child(stage: str, payload: dict, workdir: Path, *, extra_env: Optional[dict] = None,
              timeout: Optional[int] = None) -> ChildResult:
    """Run *stage* in a child interpreter and return what it wrote."""
    stage_dir = workdir / "stages"
    logs = workdir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    in_path = stage_dir / f"{stage}.in.json"
    out_path = stage_dir / f"{stage}.out.json"
    log_path = logs / f"{stage}.log"
    out_path.unlink(missing_ok=True)
    in_path.write_text(json.dumps(payload), encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if timeout is None:
        timeout = STAGE_TIMEOUTS.get(stage, 900)
    timed_out = False
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", stage,
             "--child-in", str(in_path), "--child-out", str(out_path)],
            cwd=str(workdir), env=child_env(workdir, extra_env), stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, start_new_session=(sys.platform != "win32"))
        _LIVE_PIDS.add(proc.pid)
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(proc.pid)
                proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                kill_tree(proc.pid)
            _LIVE_PIDS.discard(proc.pid)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    out = None
    try:
        out = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ChildResult(proc.returncode, out if isinstance(out, dict) else None, tail, timed_out)


def child_failure(stage: str, res: ChildResult) -> Optional[str]:
    """Why *res* is not a usable stage result, or None when it is."""
    if res.timed_out:
        return f"the {stage} stage timed out"
    if res.out is None:
        return f"the {stage} stage wrote no result (exit {res.returncode}): {res.log_tail[-600:]}"
    if res.out.get("error"):
        return f"the {stage} stage raised: {res.out['error'][-800:]}"
    return None


# ---- the stage bodies (run inside the child) ------------------------------ #

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _preload_pins(path: str) -> None:
    """Make `localm.setup_llama.pins` the module at *path* before localm.setup_llama
    is first imported, so every name the package copies from it carries the
    candidate's values."""
    spec = importlib.util.spec_from_file_location("localm.setup_llama.pins", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["localm.setup_llama.pins"] = mod
    spec.loader.exec_module(mod)


def _stage_hw(payload: dict) -> dict:
    import tempfile as _tempfile

    import localm
    from localm import config, hwdetect
    from localm import setup_llama as sl
    det = hwdetect.detect()
    return {"vendors": list(det.vendors), "gpu_names": det.gpu_names.strip(),
            "amd_gfx_family": hwdetect.amd_gfx_family(det.gpu_names),
            "probe_ok": det.probe_ok, "probe_error": det.probe_error,
            "localm_file": localm.__file__, "home_dir": str(config.home_dir()),
            "runtime_lib": str(sl._repo_runtime_lib()), "tmp_dir": _tempfile.gettempdir(),
            "llama_cpp_lib_env": os.environ.get("LLAMA_CPP_LIB", ""),
            "python": sys.version.split()[0], "platform": sys.platform}


def _stage_install(payload: dict) -> dict:
    if payload.get("pins_file"):
        _preload_pins(payload["pins_file"])
    from localm import setup_llama as sl
    from localm.setup_llama.cli import main as cli_main
    out: dict = {"expected_tag": payload["expected_tag"]}
    out["override_ok"] = (sl._ROCM_TAG == payload["expected_tag"]
                          and sl._ROCM_CPU_TAG == payload["expected_cpu_tag"])
    if not out["override_ok"]:
        out["error"] = (f"the candidate pins did not take effect: _ROCM_TAG={sl._ROCM_TAG}, "
                        f"_ROCM_CPU_TAG={sl._ROCM_CPU_TAG}")
        return out
    target = sl._repo_runtime_lib()
    out["target"] = str(target)
    listing = sl._release_assets(sl._ROCM_TAG, repo=bump.LEMONADE_REPO)
    out["listing_ok"] = bool(listing)
    url, sha, _tag = sl._resolve_backend_asset("amd-rocm")
    out["resolved"] = {"url": url, "sha256": sha}
    exit_code, exc = None, None
    try:
        cli_main.main(args=["--backend", "amd-rocm", "--force", "--yes"], standalone_mode=False)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:  # noqa: BLE001 - reported to the orchestrator
        exc = f"{type(e).__name__}: {e}"
    out.update(exit_code=exit_code, exception=exc)
    marker = sl._read_marker(target) or []
    overlay = sl.installed_cpu_overlay(target)
    out["marker"] = {"backend": marker[0] if marker else None,
                     "build": marker[1] if len(marker) > 1 else None}
    out["overlay"] = overlay
    out["llama_dll"] = (target / sl._lib_name()).is_file()
    out["files"] = sorted(p.name for p in target.iterdir() if p.is_file()) if target.is_dir() else []
    cpu_dll = target / "ggml-cpu.dll"
    out["ggml_cpu_sha256"] = _sha256_file(cpu_dll) if cpu_dll.is_file() else None
    return out


def dedicated_gpu_bytes(pid: int) -> Optional[int]:
    """Dedicated GPU memory in use by process *pid*, from the Windows "GPU
    Process Memory" counter set; 0 when the counter set lists no instance of the
    process, None when the counter set cannot be read (not Windows, no typeperf,
    a localized counter name)."""
    if sys.platform != "win32":
        return None
    try:
        r = subprocess.run(["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", "-sc", "1"],
                           capture_output=True, text=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_process_memory(r.stdout, pid)


def parse_gpu_process_memory(csv_text: str, pid: int) -> Optional[int]:
    """The Dedicated Usage bytes of process *pid* summed over its instances in
    typeperf CSV output, 0 when none lists it, None when the output is not the
    GPU Process Memory counter set."""
    rows = [row for row in csv.reader(csv_text.splitlines()) if row]
    if len(rows) < 2 or not any("GPU Process Memory" in c for c in rows[0][1:]):
        return None
    total = 0.0
    for name, value in zip(rows[0][1:], rows[1][1:], strict=False):
        m = re.search(r"\(pid_(\d+)_", name)
        if m and int(m.group(1)) == pid:
            try:
                total += float(value)
            except ValueError:
                return None
    return int(total)


def _generate(model: str, n_gpu_layers: int) -> dict:
    """Load *model* through the product's GgufBackend, generate, and read back
    the worker's mapped libraries and its dedicated GPU memory."""
    import psutil

    from localm.inference.backends.gguf import GgufBackend
    res: dict = {"requested_gpu_layers": n_gpu_layers}
    be = GgufBackend(model, n_ctx=2048, n_gpu_layers=n_gpu_layers)
    try:
        be.load()
        res["effective_gpu_layers"] = be.effective_gpu_layers
        pid = be._runner._proc.pid
        try:
            maps = psutil.Process(pid).memory_maps()
            res["modules"] = sorted({m.path for m in maps if m.path.lower().endswith(".dll")})
        except Exception as e:  # noqa: BLE001 - recorded, judged by the orchestrator
            res["modules_error"] = f"{type(e).__name__}: {e}"
        t0 = time.monotonic()
        text = "".join(be.chat_stream([{"role": "user", "content": PROMPT}], max_tokens=40,
                                      temperature=0.0, seed=1)).strip()
        res["seconds"] = round(time.monotonic() - t0, 2)
        res["text"] = text[:300]
        res["worker_vram"] = dedicated_gpu_bytes(pid)
    except Exception as e:  # noqa: BLE001 - reported as the stage's verdict input
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            be.unload()
        except Exception as e:  # noqa: BLE001
            res.setdefault("unload_error", f"{type(e).__name__}: {e}")
    return res


def _stage_probe(payload: dict, *, cpu_only: bool) -> dict:
    from localm.inference.backends.llamacpp import _api, _loader
    from localm.inference.backends.llamacpp._abi import AbiMismatch, abi_report
    expected = Path(payload["expected_lib_dir"]).resolve()
    out: dict = {"gpu_type": _loader.GGML_DEV_TYPE_GPU}
    try:
        _loader.load_lib()
        out["load"] = {"status": "ok"}
    except AbiMismatch as e:
        out["load"] = {"status": "abi_mismatch", "detail": str(e)}
        return out
    except Exception as e:  # noqa: BLE001
        out["load"] = {"status": "load_error", "detail": f"{type(e).__name__}: {e}"}
        return out
    got = _loader.runtime_binary_dir()
    out["runtime_dir"] = str(got) if got else None
    out["runtime_dir_ok"] = bool(got) and Path(got).resolve() == expected
    verdict = abi_report()
    out["abi"] = {"status": verdict.status, "layout": verdict.layout,
                  "context_layout": verdict.context_layout, "failures": list(verdict.failures),
                  "diagnostics": list(verdict.diagnostics), "detail": verdict.detail}
    handles = _loader._ggml_dev_handles()
    ident: dict = {}
    fn = _loader._ggml_sym(handles, "ggml_commit")
    if fn is not None:
        import ctypes
        fn.restype = ctypes.c_char_p
        fn.argtypes = []
        ident["ggml_commit"] = (fn() or b"").decode(errors="replace")
    from localm.inference.backends.llamacpp._abi import _ggml_version
    ver = _ggml_version(_loader.load_lib())
    ident["ggml_version"] = ".".join(map(str, ver)) if ver else None
    out["identity"] = ident
    try:
        out["system_info"] = _api.llama_print_system_info()
    except Exception as e:  # noqa: BLE001
        out["system_info"] = ""
        out["system_info_error"] = f"{type(e).__name__}: {e}"
    inv = _loader.native_device_inventory()
    out["devices"] = inv
    out["gpu_visible"] = bool(inv) and any(d.get("type") == _loader.GGML_DEV_TYPE_GPU for d in inv)
    try:
        if cpu_only:
            out["cpu"] = _generate(payload["model"], 0)
            out["cpu"]["gpu_visible"] = out["gpu_visible"]
            out["cpu"]["devices"] = [(d.get("name"), d.get("type")) for d in (inv or [])]
        else:
            out["gpu"] = _generate(payload["model"], 99)
            out["gpu_ref"] = _generate(payload["model"], 0)
    finally:
        try:
            _loader.stop_probe_daemon()
        except Exception:  # noqa: BLE001 - nothing left to report
            pass
    return out


def child_main(stage: str, in_path: str, out_path: str) -> int:
    """Entry point of a child process: run *stage* and write its result JSON."""
    try:
        payload = json.loads(Path(in_path).read_text(encoding="utf-8"))
        if stage == "hw":
            out = _stage_hw(payload)
        elif stage == "model":
            out = _stage_model(payload)
        elif stage == "install":
            out = _stage_install(payload)
        elif stage == "probe":
            out = _stage_probe(payload, cpu_only=False)
        elif stage == "probe-cpu":
            out = _stage_probe(payload, cpu_only=True)
        else:
            out = {"error": f"unknown stage {stage!r}"}
    except BaseException:  # noqa: BLE001 - every failure is data for the orchestrator
        out = {"error": traceback.format_exc()}
    Path(out_path).write_text(json.dumps(out, default=str), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

def cache_location(explicit: Optional[str], workdir: Path) -> Path:
    """The directory holding the test model: *explicit*, else ``rocm`` under
    $LOCALM_PIN_CACHE_DIR, else a directory under *workdir*."""
    if explicit:
        return Path(explicit)
    base = os.environ.get("LOCALM_PIN_CACHE_DIR")
    return Path(base) / "rocm" if base else workdir / "model-cache"


def ensure_model(cache_dir: Path, *, url: str = MODEL_URL, sha256: str = MODEL_SHA256,
                 opener: Optional[Callable] = None) -> tuple[Optional[Path], str, str]:
    """(path, source, error): the test model in *cache_dir*, verified against
    *sha256*, downloaded once when absent. Never raises."""
    path = cache_dir / MODEL_FILE
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if _sha256_file(path) == sha256:
                return path, "from the cache", ""
            path.unlink()
        tmp = path.with_name(path.name + ".part")
        if opener is None:
            try:
                from localm.http_ssl import verified_urlopen as opener
            except ImportError:
                opener = urllib.request.urlopen
        h = hashlib.sha256()
        with opener(urllib.request.Request(url, headers={"User-Agent": "localm-confirm-rocm"}),
                    timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha256:
            tmp.unlink(missing_ok=True)
            return None, "", f"the downloaded test model has sha256 {h.hexdigest()}, expected {sha256}"
        os.replace(tmp, path)
        return path, "downloaded", ""
    except Exception as e:  # noqa: BLE001 - an unobtainable model is INCONCLUSIVE
        return None, "", f"could not obtain the test model: {type(e).__name__}: {e}"


def _stage_model(payload: dict) -> dict:
    from localm.model_manager.gguf import gguf_architecture, gguf_chat_refusal
    path = Path(payload["path"])
    arch = gguf_architecture(path)
    return {"architecture": arch, "refusal": gguf_chat_refusal(arch), "bytes": path.stat().st_size}


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #

def _pinned_tag() -> str:
    try:
        return bump.read_pins(bump.read_lf(bump.PINS_PATH)).tag
    except (OSError, bump.Refused):
        return ""


def prepare_workdir(workdir: Path) -> None:
    """Create the isolated layout under *workdir*: the home, temp and cache
    directories and a copy of the repo's localm_llama_runtime package."""
    for name in ("home", "tmp", "hf", "logs", "stages"):
        (workdir / name).mkdir(parents=True, exist_ok=True)
    pkg = workdir / "runtime-pkg" / "localm_llama_runtime"
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "runtime" / "localm_llama_runtime" / "__init__.py", pkg / "__init__.py")


def cleanup_workdir(workdir: Path) -> None:
    for name in ("runtime-pkg", "home", "tmp", "hf"):
        shutil.rmtree(workdir / name, ignore_errors=True)


def _skip_rest(receipt: dict, why: str, names: tuple) -> None:
    for n in names:
        if receipt["checks"][n]["detail"] == "not run":
            set_check(receipt, n, SKIP, f"not run: {why}")


def run(args: argparse.Namespace, receipt: dict, *, fetch: Optional[bump.Fetcher] = None,
        runner: Callable = run_child) -> None:
    """Run every stage, filling *receipt*. Stops a stage chain at the first
    check that cannot go on; later checks stay SKIP with the reason."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir).free
    if free < MIN_FREE_BYTES:
        set_check(receipt, "isolation", SKIP,
                  f"only {free // 2 ** 20} MiB free under the work directory; {MIN_FREE_BYTES // 2 ** 20} MiB needed")
        return
    prepare_workdir(workdir)
    os.environ.update(child_env(workdir))

    pins_text = bump.read_lf(bump.PINS_PATH)
    view = bump.read_pins(pins_text)
    tag = view.tag if args.current else args.tag
    receipt["tag"] = tag

    res = runner("hw", {}, workdir)
    problem = child_failure("hw", res)
    if problem:
        set_check(receipt, "isolation", SKIP, problem)
        return
    hw = res.out
    receipt["hardware"] = hw
    issues = isolation_problems(hw, workdir, REPO)
    if issues:
        set_check(receipt, "isolation", SKIP, "; ".join(issues))
        return
    set_check(receipt, "isolation", PASS,
              f"home, runtime, temp under the work directory; localm imports from {hw['localm_file']}")
    status, detail = evaluate_hardware(hw)
    set_check(receipt, "hardware", status, detail)
    if status != PASS:
        return _skip_rest(receipt, "no measurable AMD gfx103X GPU", CHECK_NAMES)

    try:
        cand = bump.fetch_candidate(tag, fetch)
        problems = bump.pins_problems(view, cand) if args.current else []
        receipt["candidate"] = cand.to_receipt()
        if problems:
            set_check(receipt, "candidate", FAIL, "pins.py disagrees with the upstream release: " + "; ".join(problems))
            return _skip_rest(receipt, "the pins disagree with upstream", CHECK_NAMES)
        set_check(receipt, "candidate", PASS,
                  f"lemonade {tag} is llama.cpp {cand.lemonade_commit} = upstream {cand.cpu_tag}; "
                  f"{len(cand.rocm_assets)} ROCm assets and the CPU archive carry published digests")
    except bump.UpstreamUnreadable as e:
        set_check(receipt, "candidate", SKIP, f"could not read the release metadata: {e}")
        return _skip_rest(receipt, "release metadata unavailable", CHECK_NAMES)
    except bump.Refused as e:
        set_check(receipt, "candidate", FAIL, str(e))
        return _skip_rest(receipt, "the candidate cannot be paired", CHECK_NAMES)

    cache_dir = cache_location(args.cache_dir, workdir)
    model_path, source, err = ensure_model(cache_dir)
    info: dict = {"error": err} if err else {}
    if model_path is not None:
        res = runner("model", {"path": str(model_path)}, workdir)
        problem = child_failure("model", res)
        info = {"error": problem} if problem else {**res.out, "source": source}
    status, detail = evaluate_model(info)
    set_check(receipt, "model", status, detail)
    if status != PASS:
        return _skip_rest(receipt, "no usable test model", CHECK_NAMES)

    payload = {"expected_tag": cand.tag, "expected_cpu_tag": cand.cpu_tag}
    if not args.current:
        try:
            override = bump.rewrite_pins(pins_text, cand)
        except bump.Refused as e:
            set_check(receipt, "install", SKIP, f"the candidate pins could not be generated: {e}")
            return _skip_rest(receipt, "candidate pins unavailable", CHECK_NAMES)
        pins_file = workdir / "candidate_pins.py"
        pins_file.write_bytes(override.encode("utf-8"))
        payload["pins_file"] = str(pins_file)

    out: dict = {}
    for _ in range(INSTALL_ATTEMPTS):
        res = runner("install", payload, workdir)
        problem = child_failure("install", res)
        if problem:
            set_check(receipt, "install", SKIP, problem)
            return _skip_rest(receipt, "the install stage failed", CHECK_NAMES)
        out = res.out
        out["log_tail"] = res.log_tail
        failed = out.get("exception") or out.get("exit_code") not in (None, 0)
        if not failed or classify_install_failure(res.log_tail + str(out.get("exception"))) != "network":
            break
    status, detail = evaluate_resolve(out.get("resolved") or {}, bool(out.get("listing_ok")), cand)
    set_check(receipt, "resolve", status, detail)
    status, detail = evaluate_install(out, cand)
    set_check(receipt, "install", status, detail)
    if status != PASS:
        return _skip_rest(receipt, "the build did not install", CHECK_NAMES)

    expected_dir = Path(out["target"])
    probe_payload = {"expected_lib_dir": str(expected_dir), "model": str(model_path)}
    res = runner("probe", probe_payload, workdir)
    problem = child_failure("probe", res)
    if problem:
        set_check(receipt, "abi", SKIP, problem)
        return _skip_rest(receipt, "the probe stage failed", CHECK_NAMES)
    probe = res.out
    for name, (status, detail) in (("abi", evaluate_abi(probe)),
                                   ("ggml_identity", evaluate_identity(probe, cand)),
                                   ("gpu_device", evaluate_gpu_device(probe))):
        set_check(receipt, name, status, detail)
    status, detail = evaluate_gpu_generate(probe.get("gpu") or {}, probe.get("gpu_ref") or {},
                                           info["bytes"], expected_dir)
    set_check(receipt, "gpu_generate", status, detail)
    status, detail = evaluate_cpu_backend(out, cand, probe)
    set_check(receipt, "cpu_backend", status, detail)

    res = runner("probe-cpu", probe_payload, workdir,
                 extra_env={"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"})
    problem = child_failure("probe-cpu", res)
    if problem:
        set_check(receipt, "cpu_generate", SKIP, problem)
        return
    status, detail = evaluate_cpu_generate(res.out.get("cpu") or {}, expected_dir,
                                           (probe.get("gpu") or {}).get("text", ""))
    set_check(receipt, "cpu_generate", status, detail)


def main(argv=None, *, fetch: Optional[bump.Fetcher] = None, runner: Callable = run_child) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-in", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--tag", default=None, help="the lemonade-sdk release to confirm, e.g. b1350")
    which.add_argument("--current", action="store_true",
                       help="confirm the build the repo pins today instead of a candidate")
    ap.add_argument("--workdir", default=None, help="scratch directory for the whole run")
    ap.add_argument("--receipt", default=None, help="where to write the receipt JSON")
    ap.add_argument("--cache-dir", default=None,
                    help="persistent directory for the test model (default: $LOCALM_PIN_CACHE_DIR/rocm "
                         "when set, else under --workdir)")
    ap.add_argument("--keep", action="store_true", help="keep the runtime and home under --workdir")
    args = ap.parse_args(argv)

    if args.child:
        return child_main(args.child, args.child_in, args.child_out)
    if not (args.tag or args.current) or not args.workdir or not args.receipt:
        ap.error("--workdir and --receipt are required, with one of --tag or --current")
    if args.tag and bump.build_number(args.tag) is None:
        ap.error(f"{args.tag!r} is not a lemonade-sdk build tag (bNNNN)")

    receipt = new_receipt(_pinned_tag() if args.current else args.tag, bool(args.current))
    receipt_path = Path(args.receipt)
    write_receipt(receipt_path, receipt)
    try:
        run(args, receipt, fetch=fetch, runner=runner)
    except BaseException:  # noqa: BLE001 - an uncaught exception is INCONCLUSIVE, never FAIL
        receipt["uncaught"] = traceback.format_exc()[-3000:]
        for c in receipt["checks"].values():
            if c["status"] == SKIP and c["detail"] == "not run":
                c["detail"] = "not run: the confirm script raised an exception"
    finally:
        for pid in list(_LIVE_PIDS):
            kill_tree(pid)
        _LIVE_PIDS.clear()
        code = finalize(receipt)
        if receipt.get("uncaught") and receipt["verdict"] == PASS:
            receipt["verdict"], code = INCONCLUSIVE, EXIT_CODES[INCONCLUSIVE]
            receipt["why"] = "the confirm script raised an exception"
        write_receipt(receipt_path, receipt)
        if not args.keep:
            cleanup_workdir(Path(args.workdir))
    print(f"{receipt['verdict']}: {receipt['why']}")
    for name in CHECK_NAMES:
        c = receipt["checks"][name]
        print(f"  {name:14s} {c['status']:5s} {c['detail'][:200]}")
    print(f"receipt: {receipt_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
