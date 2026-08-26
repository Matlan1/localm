# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm-managed ComfyUI: STAGE S2 provisioning (the COPY path).

Builds a managed ComfyUI by REPLICATING a user's existing ComfyUI stack:

  1. Read the user's ComfyUI repo state: its git commit when comfy_workdir is a git
     checkout (else a version marker), plus a pip-freeze of THEIR venv (which pins
     the exact torch build via its local version label).
  2. Provision localm's managed ComfyUI under <LOCALM_HOME>/comfyui: clone that same
     commit, create a FRESH localm venv, and pip-install the SAME versions into it.
     The user's venv is never byte-copied: venvs are not portable (absolute paths in
     pyvenv.cfg / Scripts launchers, platform wheels).
  3. Custom nodes are copied ONLY when the user opts in.
  4. Models are shared via S1's extra_model_paths.yaml: the managed ComfyUI sees the
     user's models dir AND localm's managed models dir, no copy.

The user's own ComfyUI is READ ONLY here - cloned/frozen, never modified. When no
usable user ComfyUI exists, this module is not used: the setup dispatcher
(managed_comfy_fresh.setup_managed_comfy) runs the FRESH hardware-matched install
(stage S3, managed_comfy_fresh.py) instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm import config
from localm.config import load_config
from localm.debuglog import logger
from localm.media import managed_comfy as mc
from localm.media.comfy_patches import apply_patches as _apply_localm_patches


class ProgressEvent(str):
    """A best-effort provisioning progress update.

    IS the human-readable line every sink treats it as: isinstance, equality,
    formatting and a markup-off ``rich`` render all work on it directly. When an
    update corresponds to a discrete, countable step it ALSO carries the
    structured facts that line was built from: ``phase`` (a short machine key,
    e.g. "clone", "custom_nodes"), and, for a counted step,
    ``done``/``total``/``unit``. A consumer that wants state reads these
    directly; it must never regex them back out of the text.

    ``pct`` is NOT carried here. A consumer that wants a percentage derives it
    from ``done``/``total`` itself (e.g. via ``Job.progress``, which already owns
    that arithmetic).
    """

    def __new__(cls, line: str, *, phase: Optional[str] = None,
               done: Optional[int] = None, total: Optional[int] = None,
               unit: Optional[str] = None) -> "ProgressEvent":
        self = super().__new__(cls, line)
        self.phase = phase
        self.done = done
        self.total = total
        self.unit = unit
        return self


# Progress sink: one ProgressEvent at a time (a subprocess line, or a narrated step -
# see ProgressEvent above for the structured facts a step carries alongside its text).
# Best-effort - a raising sink never aborts a provision.
ProgressCb = Optional[Callable[[ProgressEvent], None]]

# Provenance marker localm drops in the managed dir recording what was replicated.
# ALSO load-bearing for is_managed_comfy_installed(), which checks it IN ADDITION
# to main.py + venv: main.py and the venv alone exist while an install is still
# running. Written only once every earlier step has succeeded.
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

def _emit(cb: ProgressCb, event: str | ProgressEvent) -> None:
    """Hand *event* to *cb*. *event* may be a plain ``str`` (a raw subprocess line, or
    any other caller with no structured facts to add - it becomes a bare
    ``ProgressEvent`` with phase/done/total/unit all None) or an already-built
    ``ProgressEvent``. Best-effort per the module contract above: a raising sink must
    never abort the provision it is merely reporting on."""
    if cb is None:
        return
    if not isinstance(event, ProgressEvent):
        event = ProgressEvent(event)
    try:
        cb(event)
    except Exception:  # a broken progress sink must never abort a provision
        logger.debug("progress sink raised (ignored)", exc_info=True)


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def pip_cache_dir() -> Path:
    """localm's OWN pip cache, inside the data dir.

    A cache, not ``--no-cache-dir``: the fresh path installs torch and the copy
    path replicates the user's whole freeze, so the bytes are multi-GB and every
    re-provision would otherwise re-download all of it. The cache lives inside
    the data dir and is removed with it. Delegates to ``config.pip_cache_dir()``
    so the pip-cache location has ONE definition shared with the other
    localm-driven pip subprocesses (plugins/deps.py, setup_llama.py)."""
    return config.pip_cache_dir()


def _isolated_env() -> dict:
    """The environment for a subprocess that runs Python against the USER's venv or
    the freshly-created managed venv - never localm's own ``PYTHONPATH``, and never the
    ambient pip cache.

    PIP_CACHE_DIR: provisioning shells out to pip in the MANAGED ComfyUI's own venv, a
    separate interpreter from localm's, so localm's own process environment does not
    reach it - the setting has to be in the env handed to the child, which is this one.
    Left unset, that child's pip caches to the per-user default OUTSIDE the data dir.
    This is THE choke point: every pip subprocess in both provisioning paths (``_run``
    here and in managed_comfy_fresh, which imports it, plus ``read_user_freeze`` below)
    builds its env from this function. The git/venv subprocesses that also run through
    here ignore it.

    ``subprocess.run`` inherits the full parent environment by default, and
    ``PYTHONPATH`` is a per-process search path that Python's ``venv`` isolation
    does NOT filter out - so a calling localm process with ``PYTHONPATH`` set would
    make a fresh "isolated" venv see whatever localm's OWN ``site-packages`` holds
    on top of its actual package set, and ``pip freeze`` would report a contaminated
    package list."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir())
    return env


def _run(cmd: list, *, cwd: Optional[Path] = None, on_progress: ProgressCb = None,
         timeout: Optional[int] = None) -> tuple[bool, str]:
    """Run *cmd*, streaming its combined output to *on_progress* AS EACH LINE ARRIVES
    (not only after the whole command exits). This is the one shared helper behind
    every git/pip/venv step in both provisioning pipelines and `comfy update`, where a
    single step can be a multi-minute `pip install` of a hardware-matched torch wheel
    or a full `git clone`.

    A missing executable or a timeout is still a clean (False, reason), never a
    raise - the caller turns it into an honest ProvisionResult. Runs with
    ``_isolated_env()`` since every caller of this helper drives the user's or the
    managed venv.

    Streams via a Popen + background reader thread rather than
    ``subprocess.run(capture_output=True)``, which buffers the whole child output and
    only hands it over once the process exits. The reader thread only does line I/O
    and calls the best-effort progress sink (``_emit``, which never raises); the
    timeout is enforced in THIS thread via ``proc.wait(timeout=...)``, so a hung child
    is killed."""
    _emit(on_progress, "$ " + " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.Popen(
            cmd, cwd=(str(cwd) if cwd else None), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=_isolated_env())
    except FileNotFoundError as e:
        return False, f"{cmd[0]} not found: {e}"
    except OSError as e:
        return False, str(e)

    lines: list[str] = []

    def _read_stdout() -> None:
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\r\n")
                lines.append(line)
                _emit(on_progress, line)
        except ValueError:
            pass  # the pipe was closed under us (process killed on timeout)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        return False, f"timed out after {timeout}s"
    reader.join(timeout=5)
    return proc.returncode == 0, "\n".join(lines)


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


def _any_plugin_workdir(cfg: dict) -> Optional[str]:
    """The first non-empty per-plugin ``comfy.workdir`` among image/music/video,
    in that order, or None. `setup_managed_comfy()` is a single, plugin-agnostic
    action (one shared managed instance, not invoked per-plugin), so there is no
    "current plugin" to resolve against here: any plugin's configured folder is
    equally valid evidence of "an existing ComfyUI to copy"."""
    from localm.plugins.media_config import MEDIA_PLUGINS, resolve_config
    for name in MEDIA_PLUGINS:
        block, _warning = resolve_config(name, cfg)
        comfy_blk = block.get("comfy") if isinstance(block.get("comfy"), dict) else {}
        workdir = comfy_blk.get("workdir")
        if workdir:
            return workdir
    return None


def discover_user_comfy(cfg: Optional[dict] = None) -> Optional[UserComfyStack]:
    """The user's replicable ComfyUI, or None when there is nothing usable to copy.

    "Usable" means: comfy_workdir is set, is a directory, has ComfyUI's ``main.py``,
    AND a venv interpreter is discoverable under it (that venv's package set is what
    gets replicated, so a ComfyUI with no venv is not copyable). None here is the
    dispatcher's signal to run the fresh hardware-matched install (S3) instead.

    Checks the legacy GLOBAL comfy_workdir first, then falls back to any plugin's own
    comfy.workdir, which is what that plugin's own launch path already uses."""
    cfg = cfg if cfg is not None else load_config()
    workdir = cfg.get("comfy_workdir") or _any_plugin_workdir(cfg)
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

    None (failure) is distinct from an empty list (a venv with no packages): a
    freeze failure must NOT be replicated as "install nothing" and then reported as
    a ready ComfyUI. The caller fails loudly on None, and only faithfully
    replicates an actually-empty venv on []."""
    try:
        proc = subprocess.run(
            [str(venv_python), "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace", env=_isolated_env())
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


# pip's own wording when a requirement resolves to nothing at all, printed in
# addition to "Could not find a version that satisfies the requirement ...".
# A requirement in the user's freeze that only pip's OWN venv could satisfy - a
# vendor-bundled driver package installed from a local wheel or a private index
# never uploaded anywhere localm's replicated install can reach - hits this,
# distinct from an ordinary transient/network install failure.
_PIP_NO_MATCH_RE = re.compile(r"^ERROR: No matching distribution found for (\S+)",
                              re.MULTILINE)


def _unresolvable_packages(pip_output: str) -> list:
    """Package names pip reported it found NO version for at all (parsed from its
    own ``No matching distribution found for <req>`` lines), deduped and in
    first-seen order. Empty when *pip_output* names none - an ordinary failure
    (network error, build failure, disk full) leaves this empty and the caller
    falls back to the generic message."""
    names: list = []
    for m in _PIP_NO_MATCH_RE.finditer(pip_output or ""):
        # Strip the version/marker suffix so the message names the bare package,
        # e.g. "amd-torch-device-gfx1030==2.12.0+rocm7.14.0" -> the package name.
        name = re.split(r"[=<>!~;\s]", m.group(1), maxsplit=1)[0]
        if name and name not in names:
            names.append(name)
    return names


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
    # "Filename too long" / "invalid index-pack output".
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


def _probe_pip(python: Path, on_progress: ProgressCb) -> tuple[bool, str]:
    """Confirm the venv just created at *python* actually has a working pip.
    ``-m venv`` can report success (return code 0, the interpreter file present)
    while its own mandatory ensurepip bootstrap silently failed - a base Python
    with ensurepip stripped, or (POSIX, ``real_base_python()`` returning None) a
    pip-less base venv itself created with ``uv venv`` and no ``--seed``. Without
    this probe the failure only surfaces two steps later as an opaque "No module
    named pip" deep inside a package install. Mirrors doctor.py's own
    ``_check_venv_creation`` probe, same wording, so the two diagnoses agree."""
    return _run([str(python), "-m", "pip", "--version"],
               on_progress=on_progress, timeout=60)


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


def _copy_custom_nodes(user_workdir: Path, managed_root: Path,
                       on_progress: ProgressCb = None) -> tuple[int, list]:
    """Copy the user's custom_nodes into the managed ComfyUI, returning (copied_count,
    warnings). Only when the user opted in. Skips __pycache__ / compiled files and
    *.example. A node that fails to copy is not silently dropped: its warning is
    RETURNED so the caller routes it through the run log (ProvisionResult.log), not
    only the live progress stream.

    Emits one structured progress event per node, before its own copy attempt
    (phase="custom_nodes", done/total/unit); this loop has no total, no timeout and no
    other output, so without a per-item signal a multi-minute copy of a large node
    collection is indistinguishable from a hang."""
    src = user_workdir / "custom_nodes"
    if not src.is_dir():
        return 0, []
    dst = managed_root / "custom_nodes"
    dst.mkdir(parents=True, exist_ok=True)
    # Same predicate as the inline skip below: a real custom node is a dir or a bare
    # .py file, never __pycache__/*.example - matches count_user_custom_nodes() so the
    # progress total agrees with what the caller already told the user to expect.
    entries = [e for e in sorted(src.iterdir())
              if e.name != "__pycache__" and not e.name.endswith(".example")
              and (e.is_dir() or (e.is_file() and e.name.endswith(".py")))]
    total = len(entries)
    count = 0
    warnings: list = []
    for i, entry in enumerate(entries, start=1):
        name = entry.name
        target = dst / name
        _emit(on_progress, ProgressEvent(
            f"Copying custom node {name} ({i}/{total}) ...",
            phase="custom_nodes", done=i, total=total, unit="nodes"))
        try:
            if entry.is_dir():
                shutil.copytree(str(entry), str(target),
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                                dirs_exist_ok=True)
                count += 1
            elif entry.is_file():
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
    failure at any step stops and reports the real reason. The user's ComfyUI is
    never modified."""
    cfg = cfg if cfg is not None else load_config()
    paths = mc.managed_comfy_paths()
    root = paths.root
    log: list = []

    def _say(line: str, *, phase: Optional[str] = None) -> None:
        log.append(line)
        _emit(on_progress, ProgressEvent(line, phase=phase))

    def _fail(status: str, message: str) -> ProvisionResult:
        """Roll back a PARTIAL install before returning a failure. Once the source is
        cloned and the venv made, S1's is_managed_comfy_installed() would read the
        half-built tree as INSTALLED and reroute media to a broken instance - and a
        re-run would refuse ("already exists"). So a failure leaves NOTHING behind.
        If the tree cannot be removed, say so rather than pretend it is gone."""
        note = ""
        if root.exists():
            try:
                mc.rmtree_robust(root)
                _emit(on_progress, ProgressEvent(
                    f"Rolled back the partial install at {root}.", phase="rollback"))
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
            _say(f"Cloning your ComfyUI ({stack.commit[:12]}) into {root} ...",
                phase="clone")
            ok, out = _clone_source(stack.workdir, root, stack.commit, on_progress)
        else:
            _say(f"Copying your ComfyUI source ({stack.version_marker}) into {root} ...",
                phase="clone")
            ok, out = _copytree_source(stack.workdir, root, on_progress)
        if not ok:
            return _fail("error", f"Could not replicate the ComfyUI source: {_tail(out)}")

        # 2) Fresh localm venv (matches the user's Python version; not a byte-copy).
        _say("Creating a fresh localm venv ...", phase="venv")
        ok, out = _create_managed_venv(stack.venv_python, root / "venv", on_progress)
        if not ok or not paths.venv_python.is_file():
            return _fail("error", f"Could not create the managed venv: {_tail(out)}")

        # 2b) Guarantee the venv actually has a working pip: `-m venv` can report
        #     success while its own mandatory ensurepip bootstrap silently failed,
        #     which would otherwise only surface two steps below as an opaque pip-
        #     transcript failure.
        ok, out = _probe_pip(paths.venv_python, on_progress)
        if not ok:
            return _fail(
                "error",
                "The venv was created but has no working pip "
                f"({_tail(out)}); managed ComfyUI setup cannot install anything "
                "into it. Run 'localm doctor' to check whether venv creation is "
                "broken on this machine.")

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
                 + (f" (torch index {index_url})" if index_url else "") + " ...",
                phase="install")
            ok, out = _install_freeze(paths.venv_python, freeze, index_url, root,
                                      on_progress)
            if not ok:
                unresolvable = _unresolvable_packages(out)
                if unresolvable:
                    return _fail("error",
                                 f"Could not replicate {', '.join(unresolvable)}: "
                                 "pip found no matching version. This usually means "
                                 "the package came from a non-public index or a "
                                 "local wheel on your ComfyUI's machine (e.g. a "
                                 "vendor-bundled driver package) that is not "
                                 "available here. This ComfyUI cannot be replicated - "
                                 "clear your ComfyUI folder setting (Settings -> "
                                 "Media, or comfy_workdir in config) and run 'localm "
                                 "comfy setup' again to install a fresh, "
                                 f"hardware-matched ComfyUI instead. {_tail(out)}")
                return _fail("error",
                             f"Installing the replicated packages failed: {_tail(out)}")
            n_pkgs = len(freeze)
        else:
            _say("Your ComfyUI venv reported no packages - replicating it empty.",
                phase="install")

        # 4) Custom nodes - only if the user opted in. A per-node copy failure is
        #    non-fatal but is routed through _say so it lands in the run log
        #    (ProvisionResult.log), not only the live stream.
        n_nodes = 0
        node_warnings: list = []
        if copy_custom_nodes:
            n_nodes, node_warnings = _copy_custom_nodes(stack.workdir, root,
                                                         on_progress=on_progress)
            _say(f"Copied {n_nodes} custom node(s) from your ComfyUI.",
                phase="custom_nodes")
            for w in node_warnings:
                _say(f"WARNING: {w}", phase="custom_nodes")
        else:
            _say("Starting with a clean custom_nodes (your nodes were not copied).",
                phase="custom_nodes")

        # 4b) Apply localm's own patch set to the managed core (S4). This patches
        #     localm's OWNED copy, never the user's ComfyUI. A failed compat patch
        #     is SURFACED but non-fatal (it only breaks the workflow needing it).
        patch_outcomes = _apply_localm_patches(root)
        for o in patch_outcomes:
            if o.ok:
                _say(f"localm patch {o.name}: {o.status}", phase="patches")
            else:
                _say(f"WARNING: localm patch {o.name} failed: {o.detail}", phase="patches")

        # 5) Models via S1's extra_model_paths.yaml: the user's models dir + localm's
        #    managed models dir, ComfyUI-native, no copy.
        mc.write_extra_model_paths(cfg)
        _say("Wrote extra_model_paths.yaml (your models + localm's managed models).",
            phase="models")

        # 6) Provenance marker, load-bearing for step 7 below -
        #    is_managed_comfy_installed() requires this file too.
        _write_marker(root, stack, n_nodes, n_pkgs, patch_outcomes)

        # 7) Prove it actually installed (S1's contract), or roll back and say it did not.
        if not mc.is_managed_comfy_installed():
            return _fail("error",
                         "Provisioning finished but the managed ComfyUI does not read "
                         "as installed (main.py or venv missing).")

        # Fold any non-fatal shortfalls into the success message so a caller that only
        # reads the result (not the live stream) still learns of them: custom nodes
        # that failed to copy, and localm patches that did not apply. Both are
        # non-fatal - each only breaks the workflow that needs it - but are not
        # reported as a clean, complete success. Mirrors the fresh path's
        # node-failure fold (managed_comfy_fresh.provision_fresh).
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
    """Resolve whether to copy the user's custom nodes. An explicit
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
            # to the safe clean start (do not copy), but record WHY.
            logger.debug("custom-nodes confirm callback raised; defaulting to not "
                         "copying the user's custom nodes: %s", e)
            return False
    return False
