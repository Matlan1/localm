# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm-managed ComfyUI: STAGE S2 provisioning (the COPY path).

S1 (managed_comfy.py) decided WHERE a managed ComfyUI lives and how routing picks
it; it deliberately provisioned nothing. This module builds one, by REPLICATING a
user's existing ComfyUI stack (design decisions 2 + 3):

  1. Read the user's ComfyUI repo state: its git commit when comfy_workdir is a git
     checkout (else a version marker), plus a pip-freeze of THEIR venv (which pins
     the exact torch build via its local version label).
  2. Provision localm's managed ComfyUI under <LOCALM_HOME>/comfyui: clone that same
     commit, create a FRESH localm venv, and pip-install the SAME versions into it.
     We do NOT byte-copy the user's venv - venvs are not portable (absolute paths in
     pyvenv.cfg / Scripts launchers, platform wheels).
  3. Custom nodes are copied ONLY when the user opts in (decision 3): a clean localm
     ComfyUI is a legitimate want.
  4. Models are shared via S1's extra_model_paths.yaml (decision 9): the managed
     ComfyUI sees the user's models dir AND localm's managed models dir, no copy.

The user's own ComfyUI is READ ONLY here - cloned/frozen, never modified. When no
usable user ComfyUI exists, this module is not used: the setup dispatcher
(managed_comfy_fresh.setup_managed_comfy) runs the FRESH hardware-matched install
(stage S3, managed_comfy_fresh.py) instead.

Not built here (later stages): S4 version pin + localm patch set + `localm comfy
update`, S5 GUI button / bug re-offer (the S3 fresh install lives in
managed_comfy_fresh.py; the S5 doctor hint already shipped).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm.config import load_config
from localm.debuglog import logger
from localm.media import managed_comfy as mc
from localm.media.comfy_patches import apply_patches as _apply_localm_patches

# Progress sink: one human-readable line at a time (a subprocess line, or a note).
# Best-effort - a raising sink never aborts a provision.
ProgressCb = Optional[Callable[[str], None]]

# Provenance marker localm drops in the managed dir recording what was replicated.
# NOT load-bearing for is_managed_comfy_installed() (S1 checks main.py + venv), so a
# missing/edited marker never fakes an install - it is documentation for S4.
MARKER_FILENAME = ".localm-comfy.json"

# Dirs that are the user's DATA / ENV, never part of the ComfyUI source to replicate.
# Used only on the non-git copy fallback; a git clone already excludes these via the
# user's .gitignore.
_NON_SOURCE_DIRS = (
    "venv", ".venv", "models", "custom_nodes", "output", "input", "temp",
    "__pycache__", ".git",
)

# torch/torchvision/torchaudio pinned as e.g. ``torch==2.4.1+rocm6.2``. pip does NOT
# record the index-url a package came from, so we recover the PyTorch wheel index
# from the local version label after the ``+``.
_TORCH_LABEL_RE = re.compile(
    r"^(?:torch|torchvision|torchaudio)==[^+\s]+\+([A-Za-z0-9.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class UserComfyStack:
    """A user's existing ComfyUI, discovered and ready to replicate."""
    workdir: Path
    venv_python: Path
    commit: Optional[str]        # git rev-parse HEAD, or None when not a git checkout
    version_marker: str          # "git:<sha>" | "comfyui_version:<v>" | "unknown"


@dataclass
class ProvisionResult:
    """Outcome of a provision attempt. ``ok`` is True only when the managed ComfyUI
    ended up actually installed (main.py + venv present)."""
    ok: bool
    status: str = ""             # "copied" | "exists" | "error"
    message: str = ""
    managed_root: Optional[Path] = None
    commit: Optional[str] = None
    installed_packages: int = 0
    custom_nodes_copied: int = 0
    log: str = ""


# --------------------------------------------------------------------------- #
#  Small process / output helpers                                             #
# --------------------------------------------------------------------------- #

def _emit(cb: ProgressCb, line: str) -> None:
    if cb is None:
        return
    try:
        cb(line)
    except Exception:  # a broken progress sink must never abort a provision
        logger.debug("progress sink raised (ignored)", exc_info=True)


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _run(cmd: list, *, cwd: Optional[Path] = None, on_progress: ProgressCb = None,
         timeout: Optional[int] = None) -> tuple[bool, str]:
    """Run *cmd*, stream its combined output to *on_progress*, return (ok, output).
    A missing executable or a timeout is a clean (False, reason), never a raise -
    the caller turns it into an honest ProvisionResult."""
    _emit(on_progress, "$ " + " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=(str(cwd) if cwd else None), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        return False, f"{cmd[0]} not found: {e}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as e:
        return False, str(e)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines():
        _emit(on_progress, line)
    return proc.returncode == 0, out


# --------------------------------------------------------------------------- #
#  Discover the user's ComfyUI stack                                          #
# --------------------------------------------------------------------------- #

def _find_user_venv_python(workdir: Path) -> Optional[Path]:
    """The user ComfyUI's own venv interpreter (venv/ or .venv/), if present. Reuses
    comfy_client's resolver so localm agrees with itself on where a ComfyUI venv is."""
    from localm.media.comfy_client import _venv_python
    return _venv_python(workdir)


def read_user_comfy_commit(workdir: Path) -> Optional[str]:
    """The user ComfyUI's HEAD commit, or None when *workdir* is not the root of its
    OWN git checkout. Verifies the repo toplevel IS *workdir* first, so a ComfyUI that
    merely sits inside some other git repo is treated as non-git (we must not clone the
    wrong outer repo)."""
    def _g(*args: str) -> Optional[str]:
        try:
            p = subprocess.run(["git", "-C", str(workdir), *args],
                               capture_output=True, text=True, timeout=15,
                               encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    top = _g("rev-parse", "--show-toplevel")
    if not top:
        return None
    try:
        if Path(top).resolve() != Path(workdir).resolve():
            return None
    except OSError:
        return None
    return _g("rev-parse", "HEAD") or None


def _version_marker(workdir: Path, commit: Optional[str]) -> str:
    """A provenance marker for what we replicated: the git commit when there is one,
    else ComfyUI's own ``comfyui_version.py`` __version__, else "unknown"."""
    if commit:
        return f"git:{commit}"
    vf = workdir / "comfyui_version.py"
    try:
        if vf.is_file():
            m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
                          vf.read_text(encoding="utf-8", errors="replace"))
            if m:
                return f"comfyui_version:{m.group(1)}"
    except OSError:
        pass
    return "unknown"


def discover_user_comfy(cfg: Optional[dict] = None) -> Optional[UserComfyStack]:
    """The user's replicable ComfyUI, or None when there is nothing usable to copy.

    "Usable" means: comfy_workdir is set, is a directory, has ComfyUI's ``main.py``,
    AND a venv interpreter is discoverable under it (we replicate that venv's package
    set, so a ComfyUI with no venv is not copyable). None here is the dispatcher's
    signal to run the fresh hardware-matched install (S3) instead."""
    cfg = cfg if cfg is not None else load_config()
    workdir = cfg.get("comfy_workdir")
    if not workdir:
        return None
    wd = Path(workdir)
    try:
        if not wd.is_dir() or not (wd / "main.py").is_file():
            return None
    except OSError:
        return None
    venv_python = _find_user_venv_python(wd)
    if venv_python is None:
        return None
    commit = read_user_comfy_commit(wd)
    return UserComfyStack(workdir=wd, venv_python=venv_python, commit=commit,
                          version_marker=_version_marker(wd, commit))


def count_user_custom_nodes(workdir) -> int:
    """How many custom nodes the user has (node dirs + standalone .py nodes), so the
    CLI can ask "copy your N custom nodes?". Ignores __pycache__ and *.example."""
    src = Path(workdir) / "custom_nodes"
    if not src.is_dir():
        return 0
    n = 0
    for entry in src.iterdir():
        name = entry.name
        if name == "__pycache__" or name.endswith(".example"):
            continue
        if entry.is_dir() or (entry.is_file() and name.endswith(".py")):
            n += 1
    return n


# --------------------------------------------------------------------------- #
#  Read + interpret the user's venv package set                               #
# --------------------------------------------------------------------------- #

def read_user_freeze(venv_python: Path) -> Optional[list]:
    """``pip freeze`` of the user's venv as a list of requirement lines (blank and
    comment lines dropped), or None when pip freeze FAILED.

    None (failure) is deliberately distinct from an empty list (a venv with no
    packages): a swallowed freeze failure must NOT be replicated as "install nothing"
    and then reported as a ready ComfyUI (rule 5: no hidden problem). The caller fails
    loudly on None, and only faithfully replicates an actually-empty venv on []."""
    try:
        proc = subprocess.run(
            [str(venv_python), "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def torch_index_url_from_freeze(freeze_lines) -> Optional[str]:
    """The PyTorch wheel index-url implied by a torch pin's local version label
    (``torch==2.4.1+rocm6.2`` -> ``https://download.pytorch.org/whl/rocm6.2``), or
    None when torch is a plain PyPI build (no ``+label``) or absent. pip does not
    record the index a package came from, so this label is the only signal we have."""
    for line in freeze_lines:
        m = _TORCH_LABEL_RE.match(line.strip())
        if m:
            return f"https://download.pytorch.org/whl/{m.group(1)}"
    return None


# --------------------------------------------------------------------------- #
#  The COPY provisioning steps                                                #
# --------------------------------------------------------------------------- #

def _clone_source(user_workdir: Path, managed_root: Path, commit: str,
                  on_progress: ProgressCb) -> tuple[bool, str]:
    """Local git clone of the user's ComfyUI into *managed_root*, then pin the exact
    commit. A local clone is offline and brings ONLY tracked source - the user's
    .gitignore keeps venv/models/custom_nodes/outputs out - so the managed checkout
    is a clean copy at the same history, without dragging the un-portable venv."""
    # core.longpaths: without it, a deeply-nested LOCALM_HOME (a long username, a
    # OneDrive-redirected profile, a nested custom data dir) can push a pack
    # object's path past Windows' legacy 260-char MAX_PATH, failing clone with
    # "Filename too long" / "invalid index-pack output" - reproduced live (#621).
    # A per-invocation -c override, not a global git config change.
    ok, out = _run(["git", "-c", "core.longpaths=true", "clone", "--quiet",
                    str(user_workdir), str(managed_root)],
                   on_progress=on_progress, timeout=600)
    if not ok:
        return False, out
    ok2, out2 = _run(["git", "-C", str(managed_root), "checkout", "--quiet", commit],
                     on_progress=on_progress, timeout=120)
    return ok2, out + "\n" + out2


def _copytree_source(user_workdir: Path, managed_root: Path,
                     on_progress: ProgressCb) -> tuple[bool, str]:
    """Fallback when the user's ComfyUI is NOT a git checkout: copy the source tree,
    excluding the venv / models / custom_nodes / outputs (the user's data + env)."""
    try:
        shutil.copytree(str(user_workdir), str(managed_root),
                        ignore=shutil.ignore_patterns(*_NON_SOURCE_DIRS, "*.pyc"))
        return True, f"copied source tree to {managed_root}"
    except Exception as e:
        return False, str(e)


def _create_managed_venv(user_venv_python: Path, managed_venv_dir: Path,
                         on_progress: ProgressCb) -> tuple[bool, str]:
    """Create a FRESH venv for the managed ComfyUI, using the user's venv base Python
    so the managed venv's Python version matches the user's ComfyUI (torch wheels are
    Python-version specific). Never a byte-copy of the user's venv."""
    return _run([str(user_venv_python), "-m", "venv", str(managed_venv_dir)],
                on_progress=on_progress, timeout=300)


def _install_freeze(managed_python: Path, freeze_lines: list, index_url: Optional[str],
                    managed_root: Path, on_progress: ProgressCb) -> tuple[bool, str]:
    """pip-install the user's exact package set into the managed venv. The freeze is
    written to a requirements file (kept as provenance) and installed with -r; the
    derived PyTorch index is added as an extra index so torch resolves to the same
    wheel while everything else still comes from PyPI."""
    req_file = managed_root / "localm-replicated-requirements.txt"
    req_file.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")
    cmd = [str(managed_python), "-m", "pip", "install", "--disable-pip-version-check"]
    if index_url:
        cmd += ["--extra-index-url", index_url]
    cmd += ["-r", str(req_file)]
    return _run(cmd, on_progress=on_progress, timeout=3600)


def _copy_custom_nodes(user_workdir: Path, managed_root: Path) -> tuple[int, list]:
    """Copy the user's custom_nodes into the managed ComfyUI, returning (copied_count,
    warnings). Only when the user opted in. Skips __pycache__ / compiled files and
    *.example. A node that fails to copy is not silently dropped: its warning is RETURNED
    so the caller routes it through the run log (ProvisionResult.log), not only the live
    progress stream (rule 5: a surfaced problem must survive into the result)."""
    src = user_workdir / "custom_nodes"
    if not src.is_dir():
        return 0, []
    dst = managed_root / "custom_nodes"
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    warnings: list = []
    for entry in sorted(src.iterdir()):
        name = entry.name
        if name == "__pycache__" or name.endswith(".example"):
            continue
        target = dst / name
        try:
            if entry.is_dir():
                shutil.copytree(str(entry), str(target),
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                                dirs_exist_ok=True)
                count += 1
            elif entry.is_file() and name.endswith(".py"):
                shutil.copy2(str(entry), str(target))
                count += 1
        except Exception as e:
            warnings.append(f"could not copy custom node {name}: {e}")
    return count, warnings


def _write_marker(managed_root: Path, stack: UserComfyStack,
                  custom_nodes_copied: int, installed_packages: int,
                  patch_outcomes=None) -> None:
    marker = {
        "source": "copy",
        "stage": "S2",
        "commit": stack.commit,
        "version_marker": stack.version_marker,
        "user_workdir": str(stack.workdir),
        "installed_packages": installed_packages,
        "custom_nodes_copied": custom_nodes_copied,
        # The localm patch set applied on top of the replicated source (S4, decision 7).
        "localm_patches": {o.name: o.status for o in (patch_outcomes or [])},
    }
    try:
        (managed_root / MARKER_FILENAME).write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logger.debug("could not write managed-comfy marker: %s", e)


def provision_by_copy(stack: UserComfyStack, cfg: Optional[dict] = None, *,
                      copy_custom_nodes: bool,
                      on_progress: ProgressCb = None) -> ProvisionResult:
    """Replicate *stack* into localm's managed ComfyUI dir. Steps: clone the source
    at the same commit, make a fresh venv, install the same packages, optionally copy
    custom nodes, write extra_model_paths.yaml, drop a provenance marker, and verify
    the result actually reads as installed. Returns an honest ProvisionResult; a
    failure at any step stops and reports the real reason (rule 5: no facade). The
    user's ComfyUI is never modified."""
    cfg = cfg if cfg is not None else load_config()
    paths = mc.managed_comfy_paths()
    root = paths.root
    log: list = []

    def _say(line: str) -> None:
        log.append(line)
        _emit(on_progress, line)

    def _fail(status: str, message: str) -> ProvisionResult:
        """Roll back a PARTIAL install before returning a failure. Once the source is
        cloned and the venv made, S1's is_managed_comfy_installed() would read the
        half-built tree as INSTALLED and reroute media to a broken instance - and a
        re-run would refuse ("already exists"). So a failure must leave NOTHING behind
        (rule 5: no broken-but-reads-as-ready facade). If the tree cannot be removed,
        say so rather than pretend it is gone."""
        note = ""
        if root.exists():
            try:
                mc.rmtree_robust(root)
                _emit(on_progress, f"Rolled back the partial install at {root}.")
            except OSError as e:
                note = (f" A partial install remains at {root} and could not be "
                        f"removed automatically ({e}); remove it before retrying.")
        return ProvisionResult(ok=False, status=status, managed_root=root,
                               log="\n".join(log), message=message + note)

    # Never silently clobber an existing managed dir - the user removes it first. This
    # returns BEFORE creating anything, so it must NOT roll back (the dir is not ours).
    if root.exists():
        return ProvisionResult(
            ok=False, status="exists", managed_root=root,
            message=(f"A managed ComfyUI already exists at {root}. Remove it first "
                     "(localm comfy remove), then run setup again."))

    try:
        # 1) Source at the same commit.
        if stack.commit:
            _say(f"Cloning your ComfyUI ({stack.commit[:12]}) into {root} ...")
            ok, out = _clone_source(stack.workdir, root, stack.commit, on_progress)
        else:
            _say(f"Copying your ComfyUI source ({stack.version_marker}) into {root} ...")
            ok, out = _copytree_source(stack.workdir, root, on_progress)
        if not ok:
            return _fail("error", f"Could not replicate the ComfyUI source: {_tail(out)}")

        # 2) Fresh localm venv (matches the user's Python version; not a byte-copy).
        _say("Creating a fresh localm venv ...")
        ok, out = _create_managed_venv(stack.venv_python, root / "venv", on_progress)
        if not ok or not paths.venv_python.is_file():
            return _fail("error", f"Could not create the managed venv: {_tail(out)}")

        # 3) Replicate the exact package set (incl. the same torch wheel). None means
        #    pip freeze FAILED - fail loudly, never proceed as if the venv were empty.
        freeze = read_user_freeze(stack.venv_python)
        if freeze is None:
            return _fail("error",
                         "Could not read your ComfyUI venv's package set (pip freeze "
                         f"failed for {stack.venv_python}). Nothing was installed.")
        n_pkgs = 0
        if freeze:
            index_url = torch_index_url_from_freeze(freeze)
            _say(f"Replicating {len(freeze)} packages into the localm venv"
                 + (f" (torch index {index_url})" if index_url else "") + " ...")
            ok, out = _install_freeze(paths.venv_python, freeze, index_url, root,
                                      on_progress)
            if not ok:
                return _fail("error",
                             f"Installing the replicated packages failed: {_tail(out)}")
            n_pkgs = len(freeze)
        else:
            _say("Your ComfyUI venv reported no packages - replicating it empty.")

        # 4) Custom nodes - only if the user opted in (decision 3). A per-node copy
        #    failure is non-fatal but is routed through _say so it lands in the run log
        #    (ProvisionResult.log), not only the live stream (rule 5, the #622 class).
        n_nodes = 0
        node_warnings: list = []
        if copy_custom_nodes:
            n_nodes, node_warnings = _copy_custom_nodes(stack.workdir, root)
            _say(f"Copied {n_nodes} custom node(s) from your ComfyUI.")
            for w in node_warnings:
                _say(f"WARNING: {w}")
        else:
            _say("Starting with a clean custom_nodes (your nodes were not copied).")

        # 4b) Apply localm's own patch set to the managed core (S4, decision 7). This
        #     patches localm's OWNED copy, never the user's ComfyUI. A failed compat
        #     patch is SURFACED but non-fatal (it only breaks the workflow needing it).
        patch_outcomes = _apply_localm_patches(root)
        for o in patch_outcomes:
            if o.ok:
                _say(f"localm patch {o.name}: {o.status}")
            else:
                _say(f"WARNING: localm patch {o.name} failed: {o.detail}")

        # 5) Models via S1's extra_model_paths.yaml (decision 9): the user's models
        #    dir + localm's managed models dir, ComfyUI-native, no copy.
        mc.write_extra_model_paths(cfg)
        _say("Wrote extra_model_paths.yaml (your models + localm's managed models).")

        # 6) Provenance marker (documentation for S4; not load-bearing).
        _write_marker(root, stack, n_nodes, n_pkgs, patch_outcomes)

        # 7) Prove it actually installed (S1's contract), or roll back and say it did not.
        if not mc.is_managed_comfy_installed():
            return _fail("error",
                         "Provisioning finished but the managed ComfyUI does not read "
                         "as installed (main.py or venv missing).")

        # Fold any non-fatal shortfalls into the success message so a caller that only
        # reads the result (not the live stream) still learns of them (the #622
        # vanished-log class): custom nodes that failed to copy, and localm patches that
        # did not apply. Both are non-fatal - each only breaks the workflow that needs it
        # - but must not be reported as a clean, complete success (rule 5). This mirrors
        # the fresh path's node-failure fold (managed_comfy_fresh.provision_fresh).
        msg = f"localm's managed ComfyUI is ready at {root}."
        if node_warnings:
            msg += (f" {len(node_warnings)} custom node(s) could not be copied; "
                    "workflows needing them will not work until fixed.")
        patch_failures = [o for o in patch_outcomes if not o.ok]
        if patch_failures:
            msg += (f" {len(patch_failures)} localm patch(es) did not apply; the "
                    "workflow they fix (e.g. ACE-Step music) may not work.")
        return ProvisionResult(
            ok=True, status="copied", managed_root=root, commit=stack.commit,
            installed_packages=n_pkgs, custom_nodes_copied=n_nodes, log="\n".join(log),
            message=msg)
    except Exception as e:  # last-resort: roll back, never a half-claimed success
        logger.debug("provision_by_copy failed", exc_info=True)
        return _fail("error", f"Provisioning failed: {e}")


def resolve_copy_custom_nodes(flag: Optional[bool], n_nodes: int, interactive: bool,
                              confirm: Optional[Callable[[], bool]] = None) -> bool:
    """Resolve whether to copy the user's custom nodes (decision 3's ASK). An explicit
    --copy-custom-nodes / --no-custom-nodes flag wins. With no flag: no nodes -> False;
    nodes present AND interactive -> ask via *confirm*; otherwise (non-interactive)
    default to a clean start (False), never hang on a prompt."""
    if flag is not None:
        return bool(flag)
    if not n_nodes:
        return False
    if interactive and confirm is not None:
        try:
            return bool(confirm())
        except Exception as e:
            # A raising confirm callback (a broken prompt) must not abort setup; default
            # to the safe clean start (do not copy), but record WHY (rule 5).
            logger.debug("custom-nodes confirm callback raised; defaulting to not "
                         "copying the user's custom nodes: %s", e)
            return False
    return False
