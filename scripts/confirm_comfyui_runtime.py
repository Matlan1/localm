#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a ComfyUI commit LOADS AND RUNS before it is pinned.

``localm/media/managed_comfy_fresh.py`` installs ``COMFYUI_PINNED_COMMIT`` - one
commit we decided on - into every fresh managed ComfyUI. This script is what
earns the word "confirmed": a commit that merely CLONES is not confirmed,
because a real install needs hardware-matched torch, localm's own custom
node, localm's patch set, and a running server that actually executes a
graph, all to line up.

WHAT IT DOES, in three PHASES (each a separate ``--phase`` invocation, so the
caller can hold the shared GPU lease only around the phase that needs it):

  provision  Clone the candidate commit into an ISOLATED scratch LOCALM_HOME
             (never the real one), install hardware-matched torch, ComfyUI's
             own requirements, the derived pinned custom node, and localm's
             patch set - exactly scripts/setup would, via
             localm.media.managed_comfy_fresh.provision_fresh(). No GPU work
             happens here beyond what pip/torch installation itself does, and
             it never launches ComfyUI. NOT run under the GPU lease.
  smoke      Launch the freshly-provisioned instance on an explicit, verified-
             free port (never the default - this box may already be running
             a DIFFERENT ComfyUI on it), confirm its own /system_stats really
             is OUR install (never silently confirm someone else's server),
             then a series of model-free checks culminating in a real GPU
             kernel: EmptyImage -> ImageBlur -> SaveImage, whose blur of a
             solid color is a KNOWN-ANSWER check on a real tensor op. SHOULD
             run under the GPU lease - it is the only phase that touches the
             card.
  teardown   Stop anything left listening, then delete the scratch install
             (keeping its pip/torch download cache so a later run does not
             re-fetch several GB). Never under the lease.
  all        provision, then smoke, then teardown, in one process - for a
             manual/calibration run.

WHY THIS IS MODEL-FREE: no tiny checkpoint model exists anywhere in this
project to build a real-generation smoke test on, and sourcing/bundling one
is a separate, deliberate decision, not part of this pipeline. What this DOES
prove is the actual regression risk unique to a ComfyUI bump: does the
install/patch/custom-node pipeline still work, and does a real GPU kernel
still execute through the node graph. It does NOT prove model loading,
sampling, GGUF dequantization, VAE decode, or the ACE-Step audio-decode patch
branch - see NOT_COVERED in the written receipt.

ISOLATION. LOCALM_HOME (and TEMP/TMP/TMPDIR) are set BEFORE localm is ever
imported - several of localm's own path constants are frozen at import time,
so setting them late would install into scratch PATHS while still reading the
REAL config. After setup, every resolved path is asserted to be under the
scratch dir; if not, nothing is installed and the run is INCONCLUSIVE.

IDENTITY. ensure_comfy() reuses whatever already answers at a URL - so the
smoke phase never uses the ComfyUI default port, and verifies /system_stats
names paths under the scratch checkout before trusting anything else it says.

Three outcomes, kept distinct, same as confirm_llama_runtime.py: PASS, FAIL
(the candidate is bad), INCONCLUSIVE (could not measure - no network, GPU
busy, an environment problem). Exit code 0 / 1 / 2. Every check the script
knows about is recorded independently in the receipt as PASS/FAIL/INCONCLUSIVE
with its own reason; scripts/bump_comfyui_pin.py reads that receipt.

Usage:
    python scripts/confirm_comfyui_runtime.py --tag v0.32.0 --commit <40-hex> \\
        --workdir D:/projects/localm-pin-pipeline-comfy-scratch --receipt r.json \\
        --phase provision
    ... --phase smoke ...
    ... --phase teardown
    python scripts/confirm_comfyui_runtime.py --tag v0.32.0 --commit <40-hex> \\
        --workdir <scratch> --receipt r.json --phase all --require-gpu

Needs localm importable. Nothing under localm/ imports this; it never runs
from a user's install.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

# The canonical check names - MUST match scripts/bump_comfyui_pin.py's
# CHECK_NAMES exactly; that script imports nothing from here (no network
# calls allowed there) so the two lists are kept in sync by hand, verified by
# a test that imports both and compares them.
CHECK_NAMES = ("isolation", "provision", "checkout", "custom_nodes", "localm_patches",
              "torch_device", "identity", "nodes_registered", "shipped_workflows",
              "gpu_roundtrip")

NOT_COVERED = ["model loading and sampling", "GGUF dequantization", "VAE decode",
              "the ACE-Step VAEDecodeAudio patch code path",
              "torch builds other than this box's own",
              "the localm comfy update path for an existing install"]

# A model-free workflow proving a REAL GPU kernel executes (ImageBlur is a
# core V3 node that moves its tensor to get_torch_device() and runs a real
# F.conv2d there - EmptyImage alone never leaves the CPU). Blurring a
# perfectly solid color is a KNOWN-ANSWER check: the normalized kernel must
# return that same color, within +-1 for SaveImage's float->byte truncation.
_PROBE_COLOR = 0x336699  # (51, 102, 153)
_PROBE_RGB = (0x33, 0x66, 0x99)
_PROBE_WORKFLOW = {
    "1": {"class_type": "EmptyImage",
         "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": _PROBE_COLOR}},
    "2": {"class_type": "ImageBlur",
         "inputs": {"image": ["1", 0], "blur_radius": 2, "sigma": 1.0}},
    "3": {"class_type": "SaveImage",
         "inputs": {"images": ["2", 0], "filename_prefix": "localm_pin_confirm"}},
}


# --------------------------------------------------------------------------- #
#  Receipt: read/write, and the one function bump_comfyui_pin.py's contract   #
#  actually depends on for a check's shape                                    #
# --------------------------------------------------------------------------- #

def _new_receipt(tag: str, commit: str) -> dict:
    return {"schema": 1, "tag": tag, "commit": commit, "checks": {}, "baseline": {},
           "server": {}, "launch_log_tail": "", "not_covered": list(NOT_COVERED)}


def _load_receipt(path: Path, tag: str, commit: str) -> "dict | None":
    """The existing receipt at *path* if it is for the SAME tag+commit, else
    None (a stale receipt from a different candidate must never be silently
    reused as if it already covered this one)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("tag") != tag or data.get("commit") != commit:
        return None
    return data


def _save_receipt(path: Path, receipt: dict) -> None:
    receipt["written_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")


def _set_check(receipt: dict, name: str, verdict: str, why: str, **extra) -> None:
    entry = {"verdict": verdict, "why": why}
    entry.update(extra)
    receipt.setdefault("checks", {})[name] = entry


def _check_passed(receipt: dict, name: str) -> bool:
    return receipt.get("checks", {}).get(name, {}).get("verdict") == PASS


# --------------------------------------------------------------------------- #
#  Isolation (H2): LOCALM_HOME must be set BEFORE localm is ever imported,    #
#  and every resolved path re-checked to be under the scratch dir.            #
# --------------------------------------------------------------------------- #

def _prepare_scratch_env(workdir: Path) -> "tuple[Path, Path]":
    """(scratch_home, scratch_tmp), created, and every relevant env var
    pointed at them. MUST run before the first `import localm` anywhere in
    this process - several of localm's path constants are frozen at import
    time (see localm/config.py's module-level HOME_DIR etc.)."""
    home = workdir / "home"
    tmp = workdir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["LOCALM_HOME"] = str(home)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TMPDIR"] = str(tmp)
    return home, tmp


def _verify_isolation(workdir: Path) -> "tuple[bool, str]":
    """(ok, why). Re-checks, AFTER import, that every localm path this script
    is about to touch actually resolved under *workdir* - the only thing that
    stands between a misconfigured environment and installing into the
    maintainer's real LOCALM_HOME (see the module docstring's ISOLATION
    section)."""
    from localm.config import HOME_DIR, home_dir
    from localm.media import managed_comfy as mc
    workdir = workdir.resolve()
    candidates = {"HOME_DIR": Path(HOME_DIR).resolve(),
                 "home_dir()": Path(home_dir()).resolve(),
                 "managed_comfy_paths().root": mc.managed_comfy_paths().root.resolve()}
    bad = {name: str(p) for name, p in candidates.items()
          if workdir not in p.parents and p != workdir}
    if bad:
        return False, f"paths did not resolve under {workdir}: {bad}"
    return True, "every relevant localm path resolves under the scratch dir"


# --------------------------------------------------------------------------- #
#  provision phase                                                            #
# --------------------------------------------------------------------------- #

def _requirements_changed(root: Path, old_commit: str, new_commit: str) -> "bool | None":
    """Whether requirements.txt differs between *old_commit* and *new_commit*
    inside the already-cloned *root* (a full clone, so both commits are
    reachable). None when it could not be determined (never treated as a
    firm "no change" - the bump checklist's --reinstall-requirements advice
    would otherwise silently go missing)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", old_commit, new_commit,
            "--", "requirements.txt"],
            capture_output=True, timeout=60)
    except Exception:
        return None
    if r.returncode not in (0, 1):
        return None
    return r.returncode == 1


def run_provision_phase(tag: str, commit: str, workdir: Path, receipt_path: Path,
                        *, require_gpu: bool) -> int:
    _prepare_scratch_env(workdir)
    receipt = _new_receipt(tag, commit)

    ok, why = _verify_isolation(workdir)
    _set_check(receipt, "isolation", PASS if ok else INCONCLUSIVE, why)
    if not ok:
        _save_receipt(receipt_path, receipt)
        print(f"INCONCLUSIVE: {why}")
        return 2

    from localm import hwdetect
    from localm.media import managed_comfy as mc
    from localm.media.managed_comfy_fresh import (
        COMFYUI_PINNED_COMMIT, comfy_torch_spec, provision_fresh, required_custom_nodes)

    det = hwdetect.detect()
    spec = comfy_torch_spec(det)
    if require_gpu and spec.variant == "cpu":
        why = (f"no verified GPU torch wheel for this hardware (variant={spec.variant!r}, "
              f"note={spec.note!r}); --require-gpu refuses to spend an install on a "
              "CPU-only confirm")
        _set_check(receipt, "provision", INCONCLUSIVE, why)
        _save_receipt(receipt_path, receipt)
        print(f"INCONCLUSIVE: {why}")
        return 2
    receipt["torch_spec"] = {"variant": spec.variant, "note": spec.note,
                            "packages": list(spec.packages)}

    def _log(line):
        print(f"  {line}", flush=True)

    print(f"provisioning ComfyUI {tag} ({commit[:12]}) into {workdir} ...", flush=True)
    result = provision_fresh(comfyui_commit=commit, torch_spec=spec, on_progress=_log)
    if not result.ok:
        verdict = INCONCLUSIVE if result.status == "exists" else FAIL
        _set_check(receipt, "provision", verdict, result.message)
        _save_receipt(receipt_path, receipt)
        print(f"{verdict}: {result.message}")
        return 1 if verdict == FAIL else 2
    _set_check(receipt, "provision", PASS, result.message)

    root = mc.managed_comfy_paths().root
    head_proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=30)
    head = head_proc.stdout.strip() if head_proc.returncode == 0 else None
    tag_proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"refs/tags/{tag}^{{commit}}"],
        capture_output=True, text=True, timeout=30)
    tag_commit = tag_proc.stdout.strip() if tag_proc.returncode == 0 else None
    if head is None:
        _set_check(receipt, "checkout", INCONCLUSIVE,
                  f"could not read HEAD in the clone: {head_proc.stderr.strip()}")
    elif tag_commit is None:
        _set_check(receipt, "checkout", INCONCLUSIVE,
                  f"could not resolve refs/tags/{tag} in the clone: {tag_proc.stderr.strip()}")
    elif head == commit == tag_commit:
        _set_check(receipt, "checkout", PASS,
                  f"HEAD and refs/tags/{tag} both resolve to the requested commit ({head})")
    elif head != commit:
        _set_check(receipt, "checkout", FAIL,
                  f"HEAD is {head!r}, expected {commit!r}")
    else:
        _set_check(receipt, "checkout", FAIL,
                  f"HEAD ({head}) matches, but refs/tags/{tag} resolves to {tag_commit!r} "
                  "in this clone, not the same commit - the tag->commit resolution used to "
                  "pick this candidate does not match a real clone's own tag ref")

    marker_path = root / ".localm-comfy.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _set_check(receipt, "custom_nodes", INCONCLUSIVE, f"could not read the marker: {e}")
        _set_check(receipt, "localm_patches", INCONCLUSIVE, f"could not read the marker: {e}")
        marker = {}
    else:
        failures = marker.get("custom_node_failures") or []
        wanted = len(required_custom_nodes().nodes)
        installed = marker.get("custom_nodes_installed", 0)
        if failures:
            _set_check(receipt, "custom_nodes", FAIL,
                      f"{len(failures)} of {wanted} custom node(s) failed to install: "
                      + "; ".join(failures))
        elif installed != wanted:
            _set_check(receipt, "custom_nodes", FAIL,
                      f"expected {wanted} custom node(s), the marker records {installed}")
        else:
            _set_check(receipt, "custom_nodes", PASS,
                      f"{installed} custom node(s) installed, none failed")

        patches = marker.get("localm_patches") or {}
        not_applied = {n: s for n, s in patches.items() if s != "applied"}
        if not_applied:
            _set_check(receipt, "localm_patches", FAIL,
                      f"not applied: {not_applied}")
        else:
            _set_check(receipt, "localm_patches", PASS,
                      f"{len(patches)} patch(es), all applied: {list(patches)}")

    changed = _requirements_changed(root, COMFYUI_PINNED_COMMIT, commit)
    receipt["baseline"] = {"tag": None, "commit": COMFYUI_PINNED_COMMIT,
                          "requirements_changed": changed}
    receipt["server"]["comfyui_version"] = marker.get("comfyui_version")

    _save_receipt(receipt_path, receipt)
    failed = [n for n in ("checkout", "custom_nodes", "localm_patches")
             if not _check_passed(receipt, n)
             and receipt["checks"][n]["verdict"] == FAIL]
    if failed:
        print(f"FAIL: provision-phase checks failed: {failed}")
        return 1
    print("provision phase: all checks passed")
    return 0


# --------------------------------------------------------------------------- #
#  smoke phase                                                                #
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    """A TCP port nothing is currently listening on, on loopback. There is a
    theoretical race between this check and the actual launch; the identity
    check after launch is the real backstop against ever trusting a server
    this script did not start."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_answers(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _bare_torch_probe(venv_python: Path) -> dict:
    """Ask the SCRATCH VENV's own interpreter (never this process's) what
    torch it has and what device it sees - independent of whatever ComfyUI
    itself later reports, so a "ComfyUI chose CPU" verdict can be told apart
    from "no GPU torch was ever installed"."""
    code = (
        "import json\n"
        "out = {'version': None, 'device_available': None, 'device_name': None}\n"
        "try:\n"
        "    import torch\n"
        "    out['version'] = torch.__version__\n"
        "    if torch.cuda.is_available():\n"
        "        out['device_available'] = True\n"
        "        out['device_name'] = torch.cuda.get_device_name(0)\n"
        "    else:\n"
        "        out['device_available'] = False\n"
        "except Exception as e:\n"
        "    out['error'] = str(e)\n"
        "print(json.dumps(out))\n"
    )
    try:
        r = subprocess.run([str(venv_python), "-c", code],
                           capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def _torch_variant_ok(reported_version: "str | None", spec) -> bool:
    if not reported_version:
        return False
    if "==" in (spec.packages[0] if spec.packages else ""):
        pinned = spec.packages[0].split("==", 1)[1]
        return reported_version == pinned
    suffix = {"cuda": "+cu", "rocm": "+rocm", "xpu": "+xpu", "cpu": "+cpu"}.get(spec.variant)
    return suffix is None or suffix in reported_version


def run_smoke_phase(tag: str, commit: str, workdir: Path, receipt_path: Path) -> int:
    _prepare_scratch_env(workdir)
    receipt = _load_receipt(receipt_path, tag, commit)
    if receipt is None:
        print(f"INCONCLUSIVE: no provision-phase receipt for {tag}@{commit} at {receipt_path}")
        return 2
    if not all(_check_passed(receipt, n) for n in ("provision", "checkout",
                                                   "custom_nodes", "localm_patches")):
        print("INCONCLUSIVE: provision phase did not fully pass; smoke phase refuses to run")
        return 2

    ok, why = _verify_isolation(workdir)
    if not ok:
        _set_check(receipt, "identity", INCONCLUSIVE, why)
        _save_receipt(receipt_path, receipt)
        print(f"INCONCLUSIVE: {why}")
        return 2

    from localm.media import comfy_client as cc
    from localm.media import managed_comfy as mc
    from localm.media.comfy_client import _PLACEMENT_NODES, _MODEL_FILE_EXTS
    from localm.media.managed_comfy_fresh import _shipped_workflow_files, _class_types_in

    paths = mc.managed_comfy_paths()
    root = paths.root

    torch_probe = _bare_torch_probe(paths.venv_python)
    spec_dict = receipt.get("torch_spec") or {}

    class _Spec:
        variant = spec_dict.get("variant", "")
        packages = tuple(spec_dict.get("packages") or ())
    if torch_probe.get("device_available") is None and "error" in torch_probe:
        _set_check(receipt, "torch_device", INCONCLUSIVE,
                  f"the scratch venv's own torch probe failed: {torch_probe['error']}")
    elif spec_dict.get("variant") != "cpu" and not torch_probe.get("device_available"):
        _set_check(receipt, "torch_device", FAIL,
                  f"expected {spec_dict.get('variant')} torch with a GPU device, but "
                  f"the scratch venv reports device_available={torch_probe.get('device_available')}",
                  torch_probe=torch_probe)
    elif not _torch_variant_ok(torch_probe.get("version"), _Spec):
        _set_check(receipt, "torch_device", FAIL,
                  f"installed torch {torch_probe.get('version')!r} does not match the "
                  f"{spec_dict.get('variant')!r} spec - pip may have silently replaced it",
                  torch_probe=torch_probe)
    else:
        _set_check(receipt, "torch_device", PASS,
                  f"torch {torch_probe.get('version')} matches the {spec_dict.get('variant')} spec",
                  torch_probe=torch_probe)

    port = _free_port()
    if _port_answers(port):
        why = f"port {port} answers before we launched anything - refusing to proceed"
        _set_check(receipt, "identity", INCONCLUSIVE, why)
        _save_receipt(receipt_path, receipt)
        print(f"INCONCLUSIVE: {why}")
        return 2
    api_url = f"http://127.0.0.1:{port}"
    launch_cmd = f'"{paths.venv_python}" "{paths.main_py}" --port {port} --disable-auto-launch'

    log_tail = ""
    try:
        launched, msg = cc.ensure_comfy(api_url=api_url, launch_cmd=launch_cmd,
                                        workdir=str(root), wait_seconds=900)
        log_path = cc.comfy_launch_log_path(api_url)
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            pass
        receipt["launch_log_tail"] = log_tail
        if not launched:
            _set_check(receipt, "identity", INCONCLUSIVE, f"launch failed: {msg}")
            _save_receipt(receipt_path, receipt)
            print(f"INCONCLUSIVE: {msg}")
            return 2

        stats = _fetch_system_stats(api_url)
        identity_ok, identity_why = _verify_identity(stats, root)
        _set_check(receipt, "identity", PASS if identity_ok else INCONCLUSIVE, identity_why)
        receipt["server"].update({
            "pytorch_version": (stats or {}).get("system", {}).get("pytorch_version"),
            "comfyui_version": (stats or {}).get("system", {}).get("comfyui_version"),
        })
        if not identity_ok:
            _save_receipt(receipt_path, receipt)
            print(f"INCONCLUSIVE: {identity_why}")
            return 2

        _check_nodes_registered(receipt, api_url, root, _PLACEMENT_NODES,
                                _shipped_workflow_files, _class_types_in)
        _check_shipped_workflows(receipt, api_url, _shipped_workflow_files, _MODEL_FILE_EXTS,
                                 cc)
        _check_gpu_roundtrip(receipt, api_url, workdir, cc)
    finally:
        try:
            cc.stop_comfy(api_url)
        except Exception:
            pass
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and _port_answers(port):
            time.sleep(1)

    _save_receipt(receipt_path, receipt)
    failed = [n for n in CHECK_NAMES if receipt["checks"].get(n, {}).get("verdict") == FAIL]
    inconclusive = [n for n in CHECK_NAMES
                    if receipt["checks"].get(n, {}).get("verdict") == INCONCLUSIVE]
    if failed:
        print(f"FAIL: {failed}")
        return 1
    if inconclusive:
        print(f"INCONCLUSIVE: {inconclusive}")
        return 2
    print("smoke phase: all checks passed")
    return 0


def _fetch_system_stats(api_url: str) -> "dict | None":
    import urllib.request
    try:
        req = urllib.request.Request(f"{api_url}/system_stats")
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - loopback only
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _verify_identity(stats: "dict | None", root: Path) -> "tuple[bool, str]":
    if not stats:
        return False, "/system_stats did not answer or returned unreadable JSON"
    system = stats.get("system") or {}
    argv0 = system.get("argv", [None])[0] if isinstance(system.get("argv"), list) else None
    try:
        resolved = Path(argv0).resolve() if argv0 else None
    except OSError:
        resolved = None
    root = root.resolve()
    if resolved is None or root not in resolved.parents and resolved != root:
        return False, (f"/system_stats argv[0] is {argv0!r}, which does not resolve inside "
                       f"{root} - this may be a DIFFERENT ComfyUI answering on this port")
    return True, f"/system_stats confirms argv[0] ({argv0}) resolves inside {root}"


def _check_nodes_registered(receipt, api_url, root, placement_nodes,
                            shipped_workflow_files, class_types_in) -> None:
    from localm.media import comfy_client as cc
    info = cc.comfy_object_info(api_url)
    if info is None:
        _set_check(receipt, "nodes_registered", INCONCLUSIVE,
                  "/object_info could not be fetched")
        return
    wanted: set = set(placement_nodes)
    for f in shipped_workflow_files():
        wanted |= class_types_in(f)
    missing = sorted(n for n in wanted if n not in info)
    if missing:
        _set_check(receipt, "nodes_registered", FAIL,
                  f"missing from /object_info: {missing}")
    else:
        _set_check(receipt, "nodes_registered", PASS,
                  f"all {len(wanted)} required node type(s) registered")


def _check_shipped_workflows(receipt, api_url, shipped_workflow_files, model_file_exts,
                             comfy_client_mod) -> None:
    files = shipped_workflow_files()
    if not files:
        _set_check(receipt, "shipped_workflows", INCONCLUSIVE, "no shipped workflow files found")
        return
    bad: list = []
    for f in files:
        try:
            workflow = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad.append(f"{f.name}: could not read/parse ({e})")
            continue
        kind, value = comfy_client_mod.comfy_submit_prompt(api_url, workflow, timeout=15)
        if kind == comfy_client_mod.SUBMIT_OK:
            # Accepted outright - fine (a model happened to already resolve, or
            # the workflow needs none). Interrupt so it does not actually run.
            try:
                comfy_client_mod.interrupt_comfy(api_url)
            except Exception:
                pass
            continue
        if kind != comfy_client_mod.SUBMIT_HTTP_ERROR:
            bad.append(f"{f.name}: {kind} ({value})")
            continue
        try:
            body = json.loads(value.read().decode("utf-8", "replace"))
        except Exception as e:
            bad.append(f"{f.name}: HTTP {value.code}, unreadable body ({e})")
            continue
        unacceptable = []
        for node_id, node_info in (body.get("node_errors") or {}).items():
            for err in node_info.get("errors", []):
                if err.get("type") == "value_not_in_list":
                    received = str((err.get("extra_info") or {}).get("received_value") or "")
                    if received.endswith(model_file_exts):
                        continue
                unacceptable.append(f"{node_info.get('class_type', node_id)}: "
                                   f"{err.get('type')} - {err.get('message')}")
        if unacceptable:
            bad.append(f"{f.name}: {'; '.join(unacceptable)}")
    if bad:
        _set_check(receipt, "shipped_workflows", FAIL, "; ".join(bad))
    else:
        _set_check(receipt, "shipped_workflows", PASS,
                  f"{len(files)} shipped workflow(s) validate (only missing-model rejections)")


def _check_gpu_roundtrip(receipt, api_url, workdir: Path, comfy_client_mod) -> None:
    kind, value = comfy_client_mod.comfy_submit_prompt(api_url, _PROBE_WORKFLOW, timeout=15)
    if kind != comfy_client_mod.SUBMIT_OK:
        _set_check(receipt, "gpu_roundtrip", FAIL, f"submit failed: {kind} ({value})")
        return
    status, result = comfy_client_mod.comfy_poll_until_done(
        api_url, value, max_poll_seconds=120)
    if status != comfy_client_mod.POLL_FINISHED:
        _set_check(receipt, "gpu_roundtrip", FAIL, f"did not finish: {status} ({result})")
        return
    info = comfy_client_mod.select_output_info(result, ("images",))
    if info is None:
        _set_check(receipt, "gpu_roundtrip", FAIL, "no image output in the finished history entry")
        return
    out_path = workdir / "gpu_roundtrip_output.png"
    try:
        comfy_client_mod.comfy_fetch_output(api_url, info, out_path, timeout=15)
    except Exception as e:
        _set_check(receipt, "gpu_roundtrip", FAIL, f"could not fetch the output: {e}")
        return
    ok, why = _verify_probe_output(out_path)
    _set_check(receipt, "gpu_roundtrip", PASS if ok else FAIL, why)


def _verify_probe_output(path: Path) -> "tuple[bool, str]":
    """The output is a solid-color PNG blurred by a real GPU kernel: the
    center pixel must still be the input color, within +-1 for float->byte
    truncation. Reads raw PNG bytes - no Pillow dependency in THIS process
    (the scratch venv's own Pillow was used to check dimensions elsewhere if
    ever needed; here we only need one pixel, which a tiny manual PNG decode
    can get for an uncompressed-enough 64x64 solid image via zlib+unfilter)."""
    import struct
    import zlib
    try:
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return False, "output is not a PNG"
        pos, idat = 8, b""
        width = height = bit_depth = color_type = None
        while pos < len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            chunk = data[pos + 8:pos + 8 + length]
            if ctype == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            elif ctype == b"IDAT":
                idat += chunk
            elif ctype == b"IEND":
                break
            pos += 8 + length + 4
        if width is None:
            return False, "no IHDR chunk found"
        if (width, height) != (64, 64):
            return False, f"output is {width}x{height}, expected 64x64"
        if bit_depth != 8 or color_type not in (2, 6):
            return False, f"unexpected PNG shape (bit_depth={bit_depth}, color_type={color_type})"
        channels = 4 if color_type == 6 else 3
        raw = zlib.decompress(idat)
        stride = 1 + width * channels
        cy = height // 2
        row_start = cy * stride
        filter_type = raw[row_start]
        if filter_type != 0:
            return False, f"cannot verify a filtered PNG row (filter type {filter_type})"
        px_start = row_start + 1 + (width // 2) * channels
        r, g, b = raw[px_start], raw[px_start + 1], raw[px_start + 2]
        want = _PROBE_RGB
        if all(abs(a - w) <= 1 for a, w in zip((r, g, b), want)):
            return True, (f"the blurred output's center pixel is {(r, g, b)}, matching the "
                          f"solid input color {want} within rounding - a real GPU kernel ran")
        return False, f"center pixel is {(r, g, b)}, expected {want} (+-1)"
    except Exception as e:
        return False, f"could not verify the output image: {e}"


# --------------------------------------------------------------------------- #
#  teardown phase                                                             #
# --------------------------------------------------------------------------- #

def run_teardown_phase(workdir: Path) -> int:
    home = workdir / "home"
    if home.is_dir():
        for entry in home.iterdir():
            if entry.name == "cache":
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
            except OSError as e:
                print(f"  could not remove {entry}: {e}")
    tmp = workdir / "tmp"
    if tmp.is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"teardown: cleared {home} (kept cache/), removed {tmp}")
    return 0


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"^v\d+(?:\.\d+)+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--workdir", required=True,
                    help="scratch directory holding home/ (the isolated LOCALM_HOME) "
                         "and tmp/ - reused across phase invocations for one run")
    ap.add_argument("--receipt", required=True,
                    help="where the per-check verdicts are read/written as JSON; "
                         "scripts/bump_comfyui_pin.py reads it as the evidence")
    ap.add_argument("--phase", choices=("provision", "smoke", "teardown", "all"),
                    default="all")
    ap.add_argument("--require-gpu", action="store_true",
                    help="refuse (INCONCLUSIVE) rather than spend an install on a "
                         "CPU-only torch spec")
    ap.add_argument("--keep", action="store_true",
                    help="skip teardown even under --phase all")
    args = ap.parse_args(argv)

    tag, commit = args.tag.strip(), args.commit.strip().lower()
    if not _TAG_RE.match(tag):
        print(f"REFUSED: {tag!r} is not an upstream release tag (vX.Y[.Z...])")
        return 1
    if not _COMMIT_RE.match(commit):
        print(f"REFUSED: {commit!r} is not a 40-character hex commit sha")
        return 1
    workdir = Path(args.workdir)
    receipt_path = Path(args.receipt)

    try:
        if args.phase == "provision":
            return run_provision_phase(tag, commit, workdir, receipt_path,
                                       require_gpu=args.require_gpu)
        if args.phase == "smoke":
            return run_smoke_phase(tag, commit, workdir, receipt_path)
        if args.phase == "teardown":
            return run_teardown_phase(workdir)
        # all
        rc = run_provision_phase(tag, commit, workdir, receipt_path,
                                require_gpu=args.require_gpu)
        if rc == 0:
            rc = run_smoke_phase(tag, commit, workdir, receipt_path)
        if not args.keep:
            run_teardown_phase(workdir)
        return rc
    except Exception as e:
        # An uncaught crash is a statement about THIS RUN, never about the
        # candidate - it must read as INCONCLUSIVE (cooldown-retried), not as
        # a permanent FAIL, matching pin_pipeline.py's own unexpected-
        # exception handling for the llama.cpp pin.
        import traceback
        traceback.print_exc()
        print(f"INCONCLUSIVE: unexpected exception: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
