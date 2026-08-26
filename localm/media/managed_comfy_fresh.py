# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: STAGE S3 provisioning (the FRESH hardware-matched path).

S2 (managed_comfy_provision.py) replicates a user's EXISTING ComfyUI. This module
handles the other case (design decisions 4 + 5): the user has NO usable ComfyUI, so
localm installs a fresh one, matched to the machine's GPU, under LOCALM_HOME.

  1. torch per hardware (decision 4): one testable map, ``comfy_torch_spec()``,
     reusing hwdetect (the single hardware source of truth) for GPU detection, the
     AMD gfx family, and the shared PyTorch wheel-index table. AMD gfx1030 -> the
     ROCm gfx103X wheels localm ITSELF runs on (pyproject [gpu]); NVIDIA -> CUDA;
     Intel -> XPU; no GPU -> CPU. ComfyUI cannot run without torch, so a combo with
     no verified GPU wheel degrades to CPU torch (with a note), never to nothing.
  2. custom nodes (decision 5): ``required_custom_nodes()`` derives the nodes a fresh
     ComfyUI needs by scanning localm's OWN shipped workflow JSONs for non-core
     class_types, each mapped to a PINNED repo+commit. Today that is exactly
     {UnetLoaderGGUFAdvanced -> city96/ComfyUI-GGUF}. A shipped-workflow class_type
     that is neither known-core nor mapped is SURFACED, never silently missing.
  3. ``provision_fresh()`` clones a PINNED ComfyUI, makes a fresh localm venv, installs
     the hardware-matched torch + ComfyUI's requirements + the derived pinned nodes,
     shares models via S1's extra_model_paths.yaml, drops a marker, and verifies the
     result. A failed base step rolls the partial install back.
  4. ``setup_managed_comfy()`` is THE public entry point (the CLI command and the GUI
     button both call it): copy when a user ComfyUI is present (S2), fresh when not.

The pinned commit is a CONSTANT here; advancing it, carrying localm's own core
patches, and ``localm comfy update`` are stage S4 and are not built here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm import hwdetect
from localm._mp_spawn import real_base_python
from localm.config import load_config
from localm.debuglog import logger
from localm.media import managed_comfy as mc
from localm.media import managed_comfy_provision as _prov
from localm.media.comfy_patches import apply_patches as _apply_localm_patches
from localm.media.managed_comfy_provision import (
    MARKER_FILENAME, ProgressCb, ProvisionResult, _emit, _probe_pip, _run, _tail)

# --------------------------------------------------------------------------- #
#  Pinned ComfyUI (decision 4/7)                                              #
# --------------------------------------------------------------------------- #
# A tagged, known-version ComfyUI to clone fresh. A fixed CONSTANT here; advancing
# it and carrying localm's own patches on top of it is stage S4's job.
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_PINNED_COMMIT = "fe4195f7f4275f2626cbafc703acc3ddde1e5490"
COMFYUI_PINNED_VERSION = "v0.31.1"

# The FIRST upstream release tag carrying comfy_extras/nodes_multigpu.py, i.e. the
# per-component placement nodes (SelectModelDevice/SelectCLIPDevice/SelectVAEDevice).
# v0.22.2 lacks it; v0.23.0 has it.
#
# It lives next to the pin so a does-this-ComfyUI-offer-placement question is a
# MECHANICAL comparison against COMFYUI_PINNED_VERSION rather than prose.
COMFYUI_PLACEMENT_MIN_VERSION = "v0.23.0"


# --------------------------------------------------------------------------- #
#  torch per hardware (decision 4): ONE testable map                          #
# --------------------------------------------------------------------------- #
# AMD gfx103X (RDNA2 / RX 6000) has no public PyTorch ROCm wheel on Windows, so
# localm runs on AMD's own repo build - the SAME wheels the [gpu] extra in
# pyproject.toml pins. A fresh ComfyUI venv needs the identical stack. Keep these in
# sync with pyproject's [gpu] extra; there is no runtime import of pyproject, so
# this is its ComfyUI-venv mirror.
#
# torchaudio is pinned here too. ComfyUI's own requirements.txt lists
# torch/torchvision/torchaudio UNPINNED, and _install_fresh's later ComfyUI
# requirements step runs plain `pip install -r requirements.txt` with no
# --extra-index-url, so torch and torchvision stay satisfied by the ROCm build
# above while an unpinned torchaudio would resolve from PyPI's default index to a
# build whose compiled C extension does not match this ROCm torch's ABI, disabling
# every audio-dependent ComfyUI node. Pinning the matching ROCm build here makes
# the later bare torchaudio requirement already-satisfied.
_AMD_GFX103X_ROCM_INDEX = "https://repo.amd.com/rocm/whl/gfx103X-all/"
_AMD_GFX103X_TORCH = ("torch==2.9.1+rocm7.13.0", "torchvision==0.24.0+rocm7.13.0",
                      "torchaudio==2.9.0+rocm7.13.0")
_AMD_GFX103X_ROCM_SDK = ("rocm", "rocm-sdk-core", "rocm-sdk-libraries-gfx103x-all")


@dataclass(frozen=True)
class ComfyTorchSpec:
    """The PyTorch install spec for a fresh ComfyUI venv on THIS hardware.
    ``packages`` are pip requirement strings; ``index_url`` is the primary wheel
    index (None = PyPI); ``extra_index_url`` an additional index (the AMD ROCm repo,
    so torch resolves there and everything else from PyPI). ``note`` carries an
    honest reason when the pick is a degraded fallback."""
    variant: str
    packages: tuple
    index_url: Optional[str] = None
    extra_index_url: Optional[str] = None
    note: str = ""


def _cpu_spec(note: str = "") -> ComfyTorchSpec:
    # macOS arm64 PyPI wheels ARE the MPS (Apple GPU) build - no special index.
    if sys.platform == "darwin":
        return ComfyTorchSpec("cpu", ("torch", "torchvision"), note=note)
    return ComfyTorchSpec("cpu", ("torch", "torchvision"),
                          index_url=hwdetect.pytorch_index_url("cpu"), note=note)


def comfy_torch_spec(det=None) -> ComfyTorchSpec:
    """Map detected hardware -> the PyTorch spec for a FRESH ComfyUI venv. THE single
    hardware->torch function (decision 4), reusing hwdetect for GPU detection, the AMD
    gfx family, and the shared wheel-index table.

    Vendor-first, and NOT hwdetect.recommended_torch_variant(): that function's
    never-ROCm-on-a-vendor-neutral-runtime-pick rule is specific to the HF/llama
    path, where a user who chose the vulkan runtime on AMD has opted OUT of ROCm. A
    FRESH ComfyUI has no such prior choice, so AMD maps to ROCm torch directly. And
    unlike the HF path, where "none" means SKIP torch, ComfyUI CANNOT run without
    torch, so any combo with no verified GPU wheel degrades to CPU torch (with a
    note), never to nothing."""
    det = det or hwdetect.detect()
    vendors = det.vendors or []
    if "nvidia" in vendors:
        return ComfyTorchSpec("cuda", ("torch", "torchvision"),
                              index_url=hwdetect.pytorch_index_url("cuda"))
    if "intel" in vendors:
        return ComfyTorchSpec("xpu", ("torch", "torchvision"),
                              index_url=hwdetect.pytorch_index_url("xpu"))
    if "amd" in vendors:
        if sys.platform != "win32":
            # Linux: upstream ROCm wheels cover a broad gfx range.
            return ComfyTorchSpec("rocm", ("torch", "torchvision"),
                                  index_url=hwdetect.pytorch_index_url("rocm-linux"))
        fam = hwdetect.amd_gfx_family(det.gpu_names)
        if fam == "gfx103x":
            return ComfyTorchSpec(
                "rocm", _AMD_GFX103X_TORCH + _AMD_GFX103X_ROCM_SDK,
                extra_index_url=_AMD_GFX103X_ROCM_INDEX)
        if fam in ("gfx110x", "gfx120x"):
            return ComfyTorchSpec("rocm", ("torch", "torchvision"),
                                  index_url=hwdetect.pytorch_index_url("rocm-win"))
        return _cpu_spec(
            note=("no verified ROCm torch wheel for this AMD GPU on Windows; using "
                  "CPU torch so ComfyUI still runs (on CPU) - add a GPU torch by hand "
                  "if a matching wheel exists for your card"))
    # Apple Silicon (MPS via the default macOS wheel) or no GPU -> CPU / PyPI default.
    return _cpu_spec()


def comfy_torch_install_args(spec: ComfyTorchSpec) -> list:
    """The pip arguments (after ``pip install``) for *spec*: the package specs, then
    --index-url / --extra-index-url as set."""
    args = list(spec.packages)
    if spec.index_url:
        args += ["--index-url", spec.index_url]
    if spec.extra_index_url:
        args += ["--extra-index-url", spec.extra_index_url]
    return args


# --------------------------------------------------------------------------- #
#  custom nodes (decision 5): derive from localm's shipped workflows          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CustomNodePin:
    """A pinned custom-node repo a shipped workflow needs."""
    name: str      # dir name under custom_nodes/
    repo: str      # git URL
    commit: str    # pinned commit


# The ONLY non-core node any shipped workflow uses today: city96's GGUF loader
# (GGUF-quantised image models). Music (ACE-Step) and video (Wan) are fully core.
_GGUF_NODE = CustomNodePin(
    name="ComfyUI-GGUF",
    repo="https://github.com/city96/ComfyUI-GGUF.git",
    commit="6ea2651e7df66d7585f6ffee804b20e92fb38b8a")

# class_type (as it appears in a workflow's nodes) -> the custom-node repo providing
# it. Several class_types may map to the same repo (deduped on install). Adding a
# workflow that needs a new node = add its class_type here.
CLASS_TYPE_TO_NODE: dict = {
    "UnetLoaderGGUFAdvanced": _GGUF_NODE,
    # Other city96 GGUF nodes resolve to the same repo once localm uses them:
    # "UnetLoaderGGUF", "DualCLIPLoaderGGUF", "CLIPLoaderGGUF" -> _GGUF_NODE.
}

# Core ComfyUI node class_types localm's shipped workflows use. A class_type a shipped
# workflow references that is neither here nor in CLASS_TYPE_TO_NODE is UNCLASSIFIED
# and surfaced, so a workflow's new node is never silently missing from a fresh
# install; the maintainer maps it (non-core) or adds it here (core).
KNOWN_CORE_NODES = frozenset({
    # flux (image)
    "BasicGuider", "BasicScheduler", "CLIPTextEncode", "DualCLIPLoader",
    "EmptyLatentImage", "FluxGuidance", "KSamplerSelect", "ModelSamplingFlux",
    "RandomNoise", "SamplerCustomAdvanced", "SaveImage", "VAEDecode", "VAELoader",
    # ace-step (music)
    "CheckpointLoaderSimple", "ConditioningZeroOut", "EmptyAceStepLatentAudio",
    "KSampler", "ModelSamplingSD3", "SaveAudio", "TextEncodeAceStepAudio",
    "VAEDecodeAudio",
    # wan (video)
    "CLIPLoader", "CreateVideo", "SaveVideo", "UNETLoader",
    "Wan22ImageToVideoLatent",
})

_GEN_DIRS = ("image_gen", "music_gen", "video_gen")


@dataclass(frozen=True)
class RequiredNodes:
    """The result of scanning shipped workflows: the deduped custom-node repos to
    install, plus any UNCLASSIFIED class_types (non-core and unmapped) to surface."""
    nodes: tuple
    unclassified: tuple


def _shipped_workflow_files(pkg_dir=None) -> list:
    """The COMMITTED workflow JSONs localm ships - never a gitignored personal
    override. Rule: include ``*.example.json``; include a bare ``*.json`` only when it
    is not a ``*_local.json`` AND has no ``<stem>.example.json`` sibling (that sibling
    IS the shipped one, the bare file its personal override). Deterministic regardless
    of a dev's personal overrides on disk."""
    base = Path(pkg_dir) if pkg_dir is not None else Path(__file__).resolve().parent.parent
    out = []
    for d in _GEN_DIRS:
        gd = base / d
        if not gd.is_dir():
            continue
        jsons = sorted(gd.glob("*.json"))
        example_stems = {p.name[:-len(".example.json")] for p in jsons
                         if p.name.endswith(".example.json")}
        for p in jsons:
            name = p.name
            if name.endswith(".example.json"):
                out.append(p)
            elif name.endswith("_local.json"):
                continue
            elif name[:-len(".json")] in example_stems:
                continue
            else:
                out.append(p)
    return out


def _class_types_in(workflow_path) -> set:
    """All node class_type strings in a ComfyUI workflow JSON (API format: a dict of
    node-id -> {class_type, inputs}; the UI-format 'nodes' list is also tolerated).
    Never raises: an unreadable/invalid file contributes nothing."""
    try:
        data = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out = set()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and isinstance(v.get("class_type"), str):
                out.add(v["class_type"])
        nodes = data.get("nodes")
        if isinstance(nodes, list):
            for n in nodes:
                if isinstance(n, dict) and isinstance(n.get("type"), str):
                    out.add(n["type"])
    return out


def required_custom_nodes(workflow_files=None) -> RequiredNodes:
    """Derive the custom-node repos a fresh ComfyUI needs from localm's shipped
    workflows (decision 5). A class_type in CLASS_TYPE_TO_NODE -> its pinned repo; a
    known-core node -> nothing; anything else -> unclassified (surfaced). Deduped by
    repo so each is cloned once, in first-seen order."""
    files = workflow_files if workflow_files is not None else _shipped_workflow_files()
    seen: set = set()
    for f in files:
        seen |= _class_types_in(f)
    by_repo: dict = {}
    unclassified: list = []
    for ct in sorted(seen):
        pin = CLASS_TYPE_TO_NODE.get(ct)
        if pin is not None:
            by_repo.setdefault(pin.repo, pin)
        elif ct not in KNOWN_CORE_NODES:
            unclassified.append(ct)
    return RequiredNodes(nodes=tuple(by_repo.values()),
                         unclassified=tuple(unclassified))


# --------------------------------------------------------------------------- #
#  Fresh provisioning                                                         #
# --------------------------------------------------------------------------- #

def _clone_at_commit(repo: str, dest: Path, commit: str,
                     on_progress: ProgressCb) -> tuple:
    """git clone *repo* into *dest*, then check out *commit*. Returns (ok, output).

    core.longpaths: without it, a deeply-nested LOCALM_HOME (a long username, a
    OneDrive-redirected profile, a nested custom data dir) can push a pack
    object's path past Windows' legacy 260-char MAX_PATH, failing clone with
    "Filename too long" / "invalid index-pack output". A per-invocation -c
    override, not a global git config change."""
    ok, out = _run(["git", "-c", "core.longpaths=true", "clone", "--quiet", repo, str(dest)],
                   on_progress=on_progress, timeout=1800)
    if not ok:
        return False, out
    ok2, out2 = _run(["git", "-C", str(dest), "checkout", "--quiet", commit],
                     on_progress=on_progress, timeout=300)
    return ok2, out + "\n" + out2


def _pip_install(managed_python: Path, args: list, on_progress: ProgressCb,
                 timeout: int) -> tuple:
    return _run([str(managed_python), "-m", "pip", "install",
                 "--disable-pip-version-check", *args],
                on_progress=on_progress, timeout=timeout)


def _install_custom_nodes(managed_root: Path, nodes, managed_python: Path,
                          on_progress: ProgressCb) -> tuple:
    """Clone each pinned custom node into custom_nodes/<name> at its commit and pip
    install its requirements.txt when present. Returns (installed_count, failures),
    failures a list of "name: reason". A node that cannot be installed is SURFACED,
    not silently dropped, but does NOT abort the base install - it only breaks the
    workflow that needs that node, so the rest of the fresh ComfyUI still works."""
    dst = managed_root / "custom_nodes"
    dst.mkdir(parents=True, exist_ok=True)
    installed = 0
    failures: list = []
    for pin in nodes:
        target = dst / pin.name
        ok, out = _clone_at_commit(pin.repo, target, pin.commit, on_progress)
        if not ok:
            failures.append(f"{pin.name}: clone failed ({_tail(out)})")
            continue
        req = target / "requirements.txt"
        if req.is_file():
            ok, out = _pip_install(managed_python, ["-r", str(req)], on_progress, 1800)
            if not ok:
                failures.append(f"{pin.name}: requirements install failed ({_tail(out)})")
                continue
        installed += 1
    return installed, failures


def _write_fresh_marker(managed_root: Path, commit: str, spec: ComfyTorchSpec,
                        n_nodes: int, node_failures: list,
                        patch_outcomes=None) -> None:
    marker = {
        "source": "fresh",
        "stage": "S3",
        "commit": commit,
        # Only claim the pinned version label when it IS the pinned commit (an
        # injected test/override commit gets no false version claim).
        "comfyui_version": COMFYUI_PINNED_VERSION if commit == COMFYUI_PINNED_COMMIT
        else None,
        "torch_variant": spec.variant,
        "torch_note": spec.note,
        "custom_nodes_installed": n_nodes,
        "custom_node_failures": node_failures,
        # The localm patch set applied on top of the pin (S4, decision 7).
        "localm_patches": {o.name: o.status for o in (patch_outcomes or [])},
    }
    try:
        (managed_root / MARKER_FILENAME).write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logger.debug("could not write managed-comfy fresh marker: %s", e)


def provision_fresh(cfg: Optional[dict] = None, *, on_progress: ProgressCb = None,
                    det=None, comfyui_repo: Optional[str] = None,
                    comfyui_commit: Optional[str] = None, custom_nodes=None,
                    torch_spec: Optional[ComfyTorchSpec] = None,
                    install_torch: bool = True) -> ProvisionResult:
    """Install a FRESH, hardware-matched ComfyUI under LOCALM_HOME (decisions 4 + 5).

    Steps: clone ComfyUI at the pinned commit, make a fresh localm venv, install the
    hardware-matched torch FIRST (so ComfyUI's unpinned torch requirement is already
    satisfied and pip does not replace it), then ComfyUI's requirements and the derived
    pinned custom nodes, write extra_model_paths.yaml (S1), drop a marker, and verify
    the result actually reads as installed. A failure at any BASE step rolls the partial
    install back and reports the real reason. The user's own ComfyUI (if any) is never
    touched.

    The repo / commit / nodes / torch_spec / install_torch args are injection points
    for a fast offline test; production callers pass none and get the real pins +
    detected hardware."""
    cfg = cfg if cfg is not None else load_config()
    repo = comfyui_repo or COMFYUI_REPO
    commit = comfyui_commit or COMFYUI_PINNED_COMMIT
    nodes = custom_nodes if custom_nodes is not None else required_custom_nodes().nodes
    spec = torch_spec if torch_spec is not None else comfy_torch_spec(det)

    paths = mc.managed_comfy_paths()
    root = paths.root
    log: list = []

    def _say(line: str) -> None:
        log.append(line)
        _emit(on_progress, line)

    def _fail(message: str) -> ProvisionResult:
        """Roll back a PARTIAL install before returning failure: once cloned and
        venv'd, S1's is_managed_comfy_installed() would read the half-built tree as
        INSTALLED and reroute media to a broken instance, and a re-run would refuse
        with "already exists". So a failure must leave NOTHING; if the tree cannot
        be removed, say so rather than pretend."""
        note = ""
        if root.exists():
            try:
                mc.rmtree_robust(root)
                _emit(on_progress, f"Rolled back the partial install at {root}.")
            except OSError as e:
                note = (f" A partial install remains at {root} and could not be "
                        f"removed automatically ({e}); remove it before retrying.")
        return ProvisionResult(ok=False, status="error", managed_root=root,
                               log="\n".join(log), message=message + note)

    # Never silently clobber an existing managed dir (returns BEFORE creating anything,
    # so it must NOT roll back - the dir is not ours to remove here).
    if root.exists():
        return ProvisionResult(
            ok=False, status="exists", managed_root=root,
            message=(f"A managed ComfyUI already exists at {root}. Remove it first "
                     "(localm comfy remove), then run setup again."))

    try:
        # 1) Clone ComfyUI at the pinned commit.
        _say(f"Cloning ComfyUI ({commit[:12]}) into {root} ...")
        ok, out = _clone_at_commit(repo, root, commit, on_progress)
        if not ok:
            return _fail(f"Could not clone ComfyUI: {_tail(out)}")

        # 2) Fresh localm venv (same base Python as localm; torch wheels are
        #    Python-version specific, and localm's interpreter is the known-good one).
        # Use the real base interpreter, never sys.executable directly: when this
        # process IS the branded LocaLM.exe copy (applaunch.py), sys.executable is a
        # renamed file whose basename never exists in the base install dir. stdlib
        # venv's EnvBuilder (and its ensurepip bootstrap) match on that basename to
        # decide what to copy/invoke inside the new venv, so a LocaLM.exe -m venv
        # creates a venv with no launcher of its own and the mandatory pip bootstrap
        # then fails with WinError 2.
        venv_python = real_base_python() or sys.executable
        _say("Creating a fresh localm venv ...")
        ok, out = _run([str(venv_python), "-m", "venv", str(root / "venv")],
                       on_progress=on_progress, timeout=300)
        if not ok or not paths.venv_python.is_file():
            return _fail(f"Could not create the managed venv: {_tail(out)}")

        # 2b) Guarantee the venv actually has a working pip: `-m venv` can report
        #     success while its own mandatory ensurepip bootstrap silently failed - a
        #     base Python with ensurepip stripped, or (POSIX, real_base_python()
        #     returning None) a pip-less base venv itself created with `uv venv` and
        #     no --seed. Without this probe the failure only surfaces two steps below
        #     as an opaque "No module named pip" buried in the torch install's
        #     pip transcript.
        ok, out = _probe_pip(paths.venv_python, on_progress)
        if not ok:
            return _fail(
                "The venv was created but has no working pip "
                f"({_tail(out)}); managed ComfyUI setup cannot install anything "
                "into it. Run 'localm doctor' to check whether venv creation is "
                "broken on this machine.")

        # 3) Hardware-matched torch FIRST (so ComfyUI's unpinned torch is satisfied and
        #    pip does not overwrite it with a plain PyPI build).
        if install_torch:
            if spec.note:
                _say(f"Note: {spec.note}")
            _say(f"Installing PyTorch ({spec.variant}) for your hardware ...")
            ok, out = _pip_install(paths.venv_python, comfy_torch_install_args(spec),
                                   on_progress, 7200)
            if not ok:
                return _fail(f"Installing PyTorch ({spec.variant}) failed: {_tail(out)}")
        else:
            _say("Skipping the torch install (caller opted out).")

        # 4) ComfyUI's own requirements.
        req = root / "requirements.txt"
        if req.is_file():
            _say("Installing ComfyUI requirements ...")
            ok, out = _pip_install(paths.venv_python, ["-r", str(req)], on_progress, 3600)
            if not ok:
                return _fail(f"Installing ComfyUI requirements failed: {_tail(out)}")

        # 5) The derived pinned custom nodes (decision 5).
        n_nodes, node_failures = _install_custom_nodes(root, nodes, paths.venv_python,
                                                       on_progress)
        _say(f"Installed {n_nodes} custom node(s).")
        for f in node_failures:
            _say(f"WARNING: {f}")

        # 5b) Apply localm's own patch set to the managed core (S4, decision 7): direct
        #     core edits localm carries on top of the pin. A failed compat patch is
        #     SURFACED but non-fatal - like a custom-node failure, it only breaks the
        #     workflow that needs it, not the whole install.
        patch_outcomes = _apply_localm_patches(root)
        for o in patch_outcomes:
            if o.ok:
                _say(f"localm patch {o.name}: {o.status}")
            else:
                _say(f"WARNING: localm patch {o.name} failed: {o.detail}")

        # 6) Models via S1's extra_model_paths.yaml (decision 9). No user comfy_workdir
        #    on the fresh path -> localm's managed models dir (the user can add more).
        mc.write_extra_model_paths(cfg)
        _say("Wrote extra_model_paths.yaml (localm's managed models dir).")

        # 7) Provenance marker for S4, also load-bearing for step 8 below:
        #    is_managed_comfy_installed() requires this file too.
        _write_fresh_marker(root, commit, spec, n_nodes, node_failures, patch_outcomes)

        # 8) Prove it installed (S1's contract), or roll back and say it did not.
        if not mc.is_managed_comfy_installed():
            return _fail("Provisioning finished but the managed ComfyUI does not read "
                         "as installed (main.py or venv missing).")

        msg = f"localm's managed ComfyUI is ready at {root} (fresh, {spec.variant} torch)."
        if node_failures:
            msg += (f" {len(node_failures)} custom node(s) did not install; workflows "
                    "needing them will not work until fixed.")
        return ProvisionResult(
            ok=True, status="fresh", managed_root=root, commit=commit,
            custom_nodes_copied=n_nodes, log="\n".join(log), message=msg)
    except Exception as e:  # last-resort: roll back, never a half-claimed success
        logger.debug("provision_fresh failed", exc_info=True)
        return _fail(f"Provisioning failed: {e}")


# --------------------------------------------------------------------------- #
#  Public entry point (CLI + GUI both call this)                              #
# --------------------------------------------------------------------------- #

def setup_managed_comfy(cfg: Optional[dict] = None, *,
                        copy_custom_nodes: Optional[bool] = None,
                        on_progress: ProgressCb = None,
                        confirm_copy_nodes: Optional[Callable[[int], bool]] = None,
                        interactive: bool = False) -> ProvisionResult:
    """THE public entry point to set up localm's managed ComfyUI - the CLI command and
    the GUI button both call this. Decides copy-vs-fresh:

      * a usable user ComfyUI exists (comfy_workdir + a venv) -> REPLICATE it (S2).
      * otherwise                                             -> FRESH install (S3).

    ``copy_custom_nodes`` (copy path only) plus the ``confirm_copy_nodes`` callback and
    ``interactive`` flag resolve the custom-nodes ASK exactly as S2's CLI did: an
    explicit flag wins; with no flag, nodes present, and interactive, ask via the
    callback (which receives the node count); otherwise default to a clean start. A GUI
    caller passes an explicit bool and interactive=False. Stable signature."""
    cfg = cfg if cfg is not None else load_config()
    stack = _prov.discover_user_comfy(cfg)
    if stack is None:
        return provision_fresh(cfg=cfg, on_progress=on_progress)

    n_nodes = _prov.count_user_custom_nodes(stack.workdir)
    do_copy = _prov.resolve_copy_custom_nodes(
        copy_custom_nodes, n_nodes=n_nodes, interactive=interactive,
        confirm=(lambda: confirm_copy_nodes(n_nodes)) if confirm_copy_nodes else None)
    return _prov.provision_by_copy(stack, cfg, copy_custom_nodes=do_copy,
                                   on_progress=on_progress)
