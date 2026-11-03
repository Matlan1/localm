# SPDX-License-Identifier: AGPL-3.0-or-later
"""The provisioned components of a localm update besides the llama.cpp runtime.

Each component a release pins and localm installs on request is compared with
what is installed (:func:`plan`), and an installed one whose pin differs is
brought to the release's pin (:func:`finish_pending`):

- ``pytorch``: the PyTorch and transformers stack in the install's venv, which
  setup installs from its own wheel source (``python -m localm.hwdetect
  torch-args <backend>``: the ``[gpu]`` extra, AMD's Windows ROCm wheels, the
  CUDA, XPU or ROCm wheel lines) outside the dependency step; the same step is
  re-run when what is installed does not satisfy it;
- ``sdcpp``: the stable-diffusion.cpp image runtime, one directory per tag and
  backend under ``<data dir>/runtimes/sdcpp``; each installed backend is
  installed at the new tag (``localm setup-sdcpp``) beside the old one;
- ``koboldcpp``: the KoboldCpp music runtime, one directory per tag and build
  under ``<data dir>/runtimes/koboldcpp`` (``localm setup-music``);
- ``comfyui``: localm's managed ComfyUI, when it is on a commit localm pinned:
  its checkout, its venv's pinned PyTorch and its pinned custom nodes
  (``localm comfy update --reinstall-requirements``); one the user chose
  (``comfy update --commit``, or a copy of their own ComfyUI) is kept;
- ``uv``: the uv copy inside the install (``<root>/.uv``); a uv the user
  installed elsewhere is kept.

A component that is not installed is never installed by an update.

Before a component is replaced, a copy of what it replaces is kept
(:func:`snapshot`): the ComfyUI commit, its edits, its marker and its custom
nodes' commits beside the update backup (:func:`snapshot_dir`); a venv as hard
links in a folder beside the venv; the uv folder. A component that fails
restores every component this run replaced, and a later rollback of the update
restores them too (:func:`restore`). A new runtime directory of sd.cpp or
KoboldCpp never replaces the old one, which stays in place for a rollback.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

PENDING_FILE = "pending-components.json"
STATUS_FILE = "components-update.json"
# The install root and version whose components were last found at its pins.
CHECKED_FILE = "components-checked.json"
# Seconds a failed first-start update waits before the next start tries again.
RETRY_AFTER_S = 3600.0
STATE_FILE = "state.json"
# Marks a venv copy a snapshot made; holds that snapshot folder's path.
COPY_MARKER = ".localm-venv-copy"
# The folder, in a snapshot, that keeps the uv folder of the install.
_KEPT_UV = "kept-uv"

# Commands that use a provisioned component, and so finish a pending update of
# one before they start.
COMPONENT_COMMANDS = frozenset({"run", "serve", "gui", "image", "music", "video"})

# The order components are updated in.
ORDER = ("pytorch", "sdcpp", "koboldcpp", "comfyui", "uv")

LABELS = {"pytorch": "PyTorch stack",
          "sdcpp": "stable-diffusion.cpp image runtime",
          "koboldcpp": "KoboldCpp music runtime",
          "comfyui": "managed ComfyUI",
          "uv": "uv"}

_SDCPP_SOURCE = "localm/media/sdcpp/pins.py"
_KOBOLD_SOURCE = "localm/media/koboldcpp/pins.py"
_COMFY_SOURCE = "localm/media/managed_comfy_fresh.py"
_SDCPP_MARKER = "localm-sdcpp.json"
_SDCPP_FAILED = "localm-sdcpp-failed.json"
_KOBOLD_MARKER = "localm-koboldcpp.json"
_COMFY_MARKER = ".localm-comfy.json"

# The ``localm setup-music --backend`` value that installs each KoboldCpp build.
_KOBOLD_BUILD_BACKEND = {"cuda": "cuda", "nocuda": "cpu", "metal": "metal"}

# Distributions a venv restore never touches: the install's own editable package.
_OWN_DISTS = frozenset({"localm"})

# Folders under the venv's site-packages whose native libraries a running
# process holds once it imported PyTorch.
_TORCH_HELD = ("torch/lib", "tokenizers")
_TORCH_HELD_GLOB = "_rocm_sdk_*"

# Seconds a stopped ComfyUI process may take to exit.
_STOP_WAIT_S = 10.0
_GIT_TIMEOUT_S = 300
_ASK_TIMEOUT_S = 300


def _warn(msg, *args) -> None:
    from localm.debuglog import logger
    logger.warning(msg, *args)


# ------------------------------ reading the pins -----------------------------

def _literals(path: Path, names) -> dict:
    """The literal values of the top-level assignments to *names* in the Python
    file at *path*, parsed (never imported). Empty when the file cannot be read
    or parsed; a name whose value is not a literal is left out."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError, RecursionError):
        return {}
    out = {}
    for node in tree.body:
        targets: list = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    continue
    return out


def _uv_pin(root: Path) -> Optional[str]:
    from localm import uv_tool
    try:
        return uv_tool.read_pin(root)[0]
    except uv_tool.UvBootstrapError:
        return None


def staged_pins(root) -> dict:
    """The component pins of the localm tree at *root*, read as text: ``{"sdcpp":
    {"tag", "assets"}, "koboldcpp": {"tag", "version", "assets"}, "comfyui":
    {"commit", "version"}, "uv": version}``, each None when that tree does not
    pin it. ``assets`` is the set of (platform, backend or build) the tree can
    install. The PyTorch stack's pins depend on the hardware and are asked of
    the tree itself (:func:`_ask_tree`)."""
    root = Path(root)
    out: dict = {"sdcpp": None, "koboldcpp": None, "comfyui": None, "uv": _uv_pin(root)}
    sd = _literals(root / _SDCPP_SOURCE, {"TAG", "ASSETS"})
    if isinstance(sd.get("TAG"), str):
        out["sdcpp"] = {"tag": sd["TAG"], "assets": set((sd.get("ASSETS") or {}).keys())}
    kb = _literals(root / _KOBOLD_SOURCE, {"TAG", "VERSION", "ASSETS"})
    if isinstance(kb.get("TAG"), str):
        out["koboldcpp"] = {"tag": kb["TAG"], "version": kb.get("VERSION"),
                            "assets": set((kb.get("ASSETS") or {}).keys())}
    cf = _literals(root / _COMFY_SOURCE, {"COMFYUI_PINNED_COMMIT", "COMFYUI_PINNED_VERSION"})
    if isinstance(cf.get("COMFYUI_PINNED_COMMIT"), str):
        out["comfyui"] = {"commit": cf["COMFYUI_PINNED_COMMIT"],
                          "version": cf.get("COMFYUI_PINNED_VERSION")}
    return out


def _extras(root: Path, names) -> list:
    """The requirement strings of the optional-dependency groups *names* in the
    ``pyproject.toml`` of the tree at *root*."""
    from localm import updater
    data = updater._read_toml(root / "pyproject.toml")
    groups = ((data.get("project") or {}).get("optional-dependencies") or {}
              if isinstance(data, dict) else {})
    out = []
    for n in names:
        out += [str(x) for x in groups.get(n) or []]
    return out


_COMFY_INFO = (
    "import json\n"
    "from localm.media import managed_comfy_fresh as f\n"
    "spec = f.comfy_torch_spec()\n"
    "nodes = [[n.name, n.repo, n.commit] for n in f.required_custom_nodes().nodes]\n"
    "print(json.dumps({'torch': f.comfy_torch_install_args(spec), 'nodes': nodes}))\n")


def _ask_tree(root: Path, argv: list) -> Optional[str]:
    """Run ``python <argv>`` with the localm tree at *root* first on the import
    path and return its standard output, or None when it fails. Used for what
    a tree decides from the hardware (the PyTorch wheel source)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    try:
        r = subprocess.run([sys.executable, *argv], cwd=str(root), env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=_ASK_TIMEOUT_S,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        _warn("update: could not ask the localm tree at %s: %s", root, e)
        return None
    if r.returncode != 0:
        _warn("update: the localm tree at %s answered %s: %s", root, r.returncode,
              (r.stderr or "").strip()[-300:])
        return None
    return r.stdout


def torch_args(root: Path, backend: str) -> Optional[str]:
    """What the setup of the tree at *root* installs PyTorch with for the
    llama.cpp *backend* on this machine (``python -m localm.hwdetect
    torch-args``): pip arguments, "" for no PyTorch, None when it cannot be asked."""
    out = _ask_tree(root, ["-m", "localm.hwdetect", "torch-args", backend])
    return None if out is None else (out.strip().splitlines() or [""])[-1].strip()


def comfy_info(root: Path) -> Optional[dict]:
    """The managed ComfyUI's PyTorch install arguments and pinned custom nodes
    the tree at *root* chooses on this machine: ``{"torch": [args], "nodes":
    [[name, repo, commit]]}``, or None when it cannot be asked."""
    out = _ask_tree(root, ["-c", _COMFY_INFO])
    if out is None:
        return None
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


# ------------------------------ venv contents --------------------------------

def site_packages(venv: Path) -> Optional[Path]:
    """The site-packages folder of the venv at *venv*, or None."""
    win = venv / "Lib" / "site-packages"
    if win.is_dir():
        return win
    for p in sorted((venv / "lib").glob("python3*/site-packages")):
        return p
    return None


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dists(site: Path) -> dict:
    """``{canonical name: {"dir", "version"}}`` for every distribution installed
    in *site* (its ``*.dist-info`` folders)."""
    out = {}
    try:
        infos = list(site.glob("*.dist-info"))
    except OSError:
        return out
    for d in infos:
        name = version = None
        try:
            with open(d / "METADATA", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        break
                    if line.startswith("Name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
        except OSError:
            continue
        if name and version:
            out[_canon(name)] = {"dir": d.name, "version": version}
    return out


def _packaging():
    try:
        from packaging.requirements import Requirement
        from packaging.version import InvalidVersion
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement  # type: ignore
        from pip._vendor.packaging.version import InvalidVersion  # type: ignore
    return Requirement, InvalidVersion


def unsatisfied(reqs, installed: dict) -> list:
    """The requirement strings of *reqs* (markers evaluated for this machine)
    the distributions *installed* (from :func:`dists`) do not satisfy, each as
    ``"name have (wants spec)"``."""
    Requirement, InvalidVersion = _packaging()
    out = []
    for s in reqs:
        try:
            r = Requirement(s)
        except Exception:
            continue
        if r.marker is not None and not r.marker.evaluate({"extra": ""}):
            continue
        have = (installed.get(_canon(r.name)) or {}).get("version")
        if have is None:
            out.append(f"{r.name} missing (wants {r.specifier or 'any'})")
            continue
        if not r.specifier:
            continue
        try:
            ok = r.specifier.contains(have, prereleases=True)
        except InvalidVersion:
            ok = False
        if not ok:
            out.append(f"{r.name} {have} (wants {r.specifier})")
    return out


def _requirements_in(args) -> list:
    """The requirement strings among pip install *args* (not options or their
    values)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--find-links", "--index-url", "--extra-index-url", "-f", "-i"):
            skip = True
            continue
        if a.startswith("-") or a.startswith("."):
            continue
        out.append(a)
    return out


# ------------------------------ what is installed ----------------------------

def _home(home=None) -> Path:
    if home is not None:
        return Path(home)
    from localm.config import home_dir
    return home_dir()


def _platform() -> Optional[str]:
    from localm.media.sdcpp.runtime import platform_key
    return platform_key()


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _runtime_entries(base: Path, marker: str, key: str) -> list:
    """``[(tag, item, mtime)]`` for every directory under *base* whose *marker*
    names a tag and an *item* (backend or build) matching the directory name
    ``<tag>-<item>``."""
    out = []
    try:
        dirs = [d for d in base.iterdir() if d.is_dir()]
    except OSError:
        return out
    for d in dirs:
        meta = _read_json(d / marker)
        if not meta:
            continue
        tag, item = meta.get("tag"), meta.get(key)
        if not isinstance(tag, str) or not isinstance(item, str) or d.name != f"{tag}-{item}":
            continue
        try:
            mtime = (d / marker).stat().st_mtime
        except OSError:
            continue
        out.append((tag, item, mtime))
    return out


def _current_tag(entries: list, preferred: Optional[str]) -> Optional[str]:
    """The tag in use: *preferred* (the installed tree's pin) when something is
    installed at it, else the tag of the most recently installed entry."""
    if not entries:
        return None
    if preferred and any(e[0] == preferred for e in entries):
        return preferred
    return max(entries, key=lambda e: e[2])[0]


def _new(name: str) -> dict:
    return {"name": name, "label": LABELS[name], "installed": False, "current": None,
            "target": None, "update": False, "check": False, "items": [], "kept": False,
            "note": None}


def _plan_tagged(name: str, staged: Optional[dict], installed_pin: Optional[dict],
                 base: Path, marker: str, key: str, plat: Optional[str],
                 failed_marker: Optional[str] = None) -> dict:
    out = _new(name)
    entries = _runtime_entries(base, marker, key)
    current = _current_tag(entries, (installed_pin or {}).get("tag"))
    if current is None:
        return out
    items = sorted({e[1] for e in entries if e[0] == current})
    out.update(installed=True, current=current, items=items)
    if staged is None:
        out["note"] = f"this version does not use the {LABELS[name]}; it was left as it is"
        return out
    target = staged["tag"]
    out["target"] = target
    have = {e[1] for e in entries if e[0] == target}
    need = [i for i in items if i not in have]
    if not need:
        return out
    skipped = []
    todo = []
    for item in need:
        if (plat, item) not in staged["assets"]:
            skipped.append(f"{item}: this release ships no {item} build for this platform")
        elif failed_marker and (_read_json(base / f"{target}-{item}" / failed_marker)
                                or {}).get("tag") == target:
            skipped.append(f"{item}: it did not load on this machine before")
        else:
            todo.append(item)
    out["items"] = todo
    notes = []
    if todo:
        out["update"] = True
        notes.append(f"updates the {LABELS[name]} ({', '.join(todo)}) from {current} "
                     f"to {target}")
    if skipped:
        notes.append(f"left the {LABELS[name]} {current} as it is for "
                     + "; ".join(skipped))
    out["note"] = "; ".join(notes)
    return out


def _git_exe() -> str:
    """git's full path (localm.sysprog.find); raises FileNotFoundError when git is
    not installed."""
    from localm import sysprog
    exe = sysprog.find("git")
    if not exe:
        raise FileNotFoundError(2, "git was not found in the system folders or on PATH", "git")
    return exe


def _git(root: Path, *args, timeout: float = _GIT_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_git_exe(), "-C", str(root), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          stdin=subprocess.DEVNULL)


def _git_bytes(root: Path, *args) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([_git_exe(), "-C", str(root), *args], capture_output=True,
                          timeout=_GIT_TIMEOUT_S, stdin=subprocess.DEVNULL)


def _git_head(root: Path) -> Optional[str]:
    try:
        r = _git(root, "rev-parse", "HEAD", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def comfy_root(home=None) -> Path:
    """Where the managed ComfyUI of the data dir *home* lives."""
    from localm.media import managed_comfy as mc
    return _home(home) / mc._INSTALL_DIRNAME


def _comfy_venv(root: Path) -> Path:
    return root / "venv"


def _comfy_installed(root: Path) -> bool:
    from localm.media import managed_comfy as mc
    return ((root / "main.py").is_file() and mc._venv_python_path(root).is_file()
            and (root / _COMFY_MARKER).is_file())


def _comfy_extra_changes(tree: Path, root: Path) -> Optional[list]:
    """What of the managed ComfyUI at *root* differs from the pinned PyTorch
    and custom nodes the tree at *tree* chooses: a list of short descriptions,
    or None when the tree cannot be asked."""
    info = comfy_info(tree)
    if info is None:
        return None
    out = []
    site = site_packages(_comfy_venv(root))
    pinned = [r for r in _requirements_in(info.get("torch") or []) if "==" in r]
    if site is not None and pinned:
        out += [f"PyTorch: {u}" for u in unsatisfied(pinned, dists(site))]
    for name, _repo, commit in info.get("nodes") or []:
        node = root / "custom_nodes" / str(name)
        if node.is_dir():
            head = _git_head(node)
            if head is not None and head != commit:
                out.append(f"custom node {name}: {head[:12]} (wants {str(commit)[:12]})")
    return out


def _plan_comfy(staged: Optional[dict], home, tree: Path, deep: bool) -> dict:
    out = _new("comfyui")
    root = comfy_root(home)
    if not _comfy_installed(root):
        return out
    marker = _read_json(root / _COMFY_MARKER) or {}
    head = _git_head(root) or marker.get("commit")
    out.update(installed=True, current=head)
    if staged is None:
        out["note"] = "this version does not manage a ComfyUI; it was left as it is"
        return out
    out["target"] = staged["commit"]
    shown = f"ComfyUI {staged.get('version') or staged['commit'][:12]}"
    if not marker.get("comfyui_version"):
        if head != staged["commit"]:
            out["kept"] = True
            out["note"] = (f"kept the managed ComfyUI on the version you chose "
                           f"({str(head or '?')[:12]}); this release ships {shown}")
        return out
    if _git_head(root) is None:
        if head != staged["commit"]:
            out["note"] = (f"the managed ComfyUI has no git history, so it was left as it "
                           f"is; this release ships {shown} (to install it: localm comfy "
                           "remove, then localm comfy setup)")
        return out
    changes = []
    if head != staged["commit"]:
        changes.append(f"ComfyUI {marker.get('comfyui_version')} ({str(head)[:12]}) -> {shown}")
    if not deep:
        out["check"] = True
    else:
        extra = _comfy_extra_changes(tree, root)
        if extra is None:
            out["note"] = ("could not ask this release which PyTorch and custom nodes the "
                           "managed ComfyUI uses; they were left as they are")
        else:
            changes += extra
    if changes:
        out["update"] = True
        out["items"] = changes
        out["note"] = "updates the managed ComfyUI: " + ", ".join(changes) + (
            "; " + out["note"] if out["note"] else "")
    return out


def _llama_backend(installed_root: Path) -> Optional[str]:
    from localm import _apply_update as au
    from localm.setup_llama.runtime_dir import _provisioned_backend
    return _provisioned_backend(Path(installed_root) / au.RUNTIME_LIB)


def torch_step(tree: Path, spec: str) -> tuple:
    """The requirements and ``uv pip`` argv lists (without the uv executable)
    of the PyTorch step setup runs for the torch-args *spec* in the tree at
    *tree*: ``(requirements, commands, wheel_line)``. ``wheel_line`` is the
    ``--torch-backend`` value (``cu126``, ``rocm6.2``, ...) or None."""
    if spec.strip() == "-e .[gpu]":
        return (_extras(tree, ["gpu", "audio"]),
                [["pip", "install", "-p", ".venv", "-e", ".[gpu,audio]"]], None)
    args = shlex.split(spec, posix=True)
    line = None
    for a in args:
        if a.startswith("--torch-backend="):
            line = a.split("=", 1)[1]
    return (_requirements_in(args) + _extras(tree, ["hf", "audio"]),
            [["pip", "install", "-p", ".venv", *args],
             ["pip", "install", "-p", ".venv", "-e", ".[hf,audio]"]], line)


def _local_label(version: str) -> Optional[str]:
    return version.split("+", 1)[1] if "+" in version else None


def _family(label: Optional[str]) -> Optional[str]:
    m = re.match(r"[a-z]+", label or "")
    return m.group(0) if m else None


def _torch_req_label(reqs) -> Optional[str]:
    """The local version label of an exact ``torch==`` requirement in *reqs*."""
    for r in reqs:
        m = re.match(r"\s*torch\s*==\s*([^\s;]+)", r)
        if m:
            return _local_label(m.group(1))
    return None


def torch_labels(tree: Path) -> set:
    """The local version labels (``cu126``, ``rocm7.13.0``, ...) of the PyTorch
    builds the setup of the tree at *tree* installs from any of its wheel
    sources, read as text."""
    hw = _literals(tree / "localm" / "hwdetect.py", {"_TORCH_BACKEND", "_AMD_ROCM_WIN_TORCH"})
    out = {str(v) for v in (hw.get("_TORCH_BACKEND") or {}).values()}
    for req in [str(hw.get("_AMD_ROCM_WIN_TORCH") or "")] + _extras(tree, ["gpu"]):
        label = _torch_req_label([req])
        if label:
            out.add(label)
    return out


def _describe_torch(installed: dict) -> str:
    parts = [f"{n} {installed[n]['version']}" for n in ("torch", "transformers")
             if n in installed]
    return ", ".join(parts)


def _plan_torch(tree: Path, installed_root: Path, deep: bool) -> dict:
    """The PyTorch stack's plan. The installed torch counts as the one setup
    installed when its build label is one either tree's setup installs from
    its wheel sources (:func:`torch_labels`), or, against an exact pin, a build
    of the pin's family; any other build is the user's and is kept."""
    out = _new("pytorch")
    site = site_packages(Path(installed_root) / ".venv")
    if site is None:
        return out
    installed = dists(site)
    if "torch" not in installed:
        return out
    out.update(installed=True, current=_describe_torch(installed))
    if not deep:
        out["check"] = True
        return out
    backend = _llama_backend(installed_root)
    if not backend:
        return out
    spec = torch_args(tree, backend)
    if spec is None:
        out["note"] = ("could not ask this release which PyTorch it installs here; the "
                       "PyTorch stack was left as it is")
        return out
    if not spec:
        return out
    reqs, commands, line = torch_step(tree, spec)
    have = installed["torch"]["version"]
    label = _local_label(have)
    wanted = line or _torch_req_label(reqs)
    known = torch_labels(tree) | torch_labels(Path(installed_root))
    ours = label is not None and (label in known or (
        line is None and wanted is not None and _family(label) == _family(wanted)))
    if not ours or (wanted and _family(label) != _family(wanted)):
        if wanted and label != wanted:
            out["kept"] = True
            out["note"] = (f"kept your PyTorch {have}; this release installs the "
                           f"{wanted} build for the {backend} runtime")
        return out
    problems = unsatisfied(reqs, installed)
    if line and label != line:
        problems.append(f"torch {have} (wants the {line} build)")
    if not problems:
        return out
    out.update(update=True, target=spec, items=commands)
    out["note"] = "updates the PyTorch stack: " + ", ".join(problems)
    return out


def _plan_uv(staged: Optional[str], installed_root: Path) -> dict:
    from localm import uv_tool
    out = _new("uv")
    contained = uv_tool.contained_uv(installed_root)
    if contained is None:
        found = uv_tool.find_uv(installed_root)
        if found and staged:
            version = uv_tool.uv_version(found)
            if version and version != staged:
                out["kept"] = True
                out["note"] = (f"kept your own uv {version} ({found}); this release pins "
                               f"uv {staged}")
        return out
    version = uv_tool.uv_version(contained)
    out.update(installed=True, current=version, target=staged)
    if not staged or version == staged:
        return out
    if version is None:
        out["note"] = (f"updates the uv in {contained.parent}, which does not report its "
                       f"version, to uv {staged}")
    else:
        out["note"] = f"updates the uv inside localm from {version} to {staged}"
    out["update"] = True
    return out


def plan(staged_root, installed_root, *, home=None, deep: bool = True) -> list:
    """What applying the tree at *staged_root* over the install at
    *installed_root* (data dir *home*, default the configured one) must do to
    each component, in :data:`ORDER`: a list of ``{"name", "label",
    "installed", "current", "target", "update", "check", "items", "kept",
    "note"}``.

    ``update`` is True only for an installed component whose installed version
    differs from the staged tree's pin and that the user did not choose
    themselves (``kept`` True, named in ``note``). ``items`` are the sd.cpp
    backends or KoboldCpp builds to install, the PyTorch step's argv lists, or
    the ComfyUI changes. ``note`` is a one-line, user-facing sentence for the
    update summary, or None.

    The PyTorch stack and the managed ComfyUI's PyTorch and custom nodes are
    decided by running the staged tree's own code (:func:`_ask_tree`); with
    *deep* False that is skipped and ``check`` is True for an installed one,
    meaning only :func:`finish_pending` can tell. Never raises: a component
    that cannot be read is reported in its note and not updated."""
    staged_root, installed_root = Path(staged_root), Path(installed_root)
    staged = staged_pins(staged_root)
    installed = staged_pins(installed_root)
    home = _home(home)
    plat = _platform()
    makers = {
        "pytorch": lambda: _plan_torch(staged_root, installed_root, deep),
        "sdcpp": lambda: _plan_tagged(
            "sdcpp", staged["sdcpp"], installed["sdcpp"], home / "runtimes" / "sdcpp",
            _SDCPP_MARKER, "backend", plat, failed_marker=_SDCPP_FAILED),
        "koboldcpp": lambda: _plan_tagged(
            "koboldcpp", staged["koboldcpp"], installed["koboldcpp"],
            home / "runtimes" / "koboldcpp", _KOBOLD_MARKER, "build", plat),
        "comfyui": lambda: _plan_comfy(staged["comfyui"], home, staged_root, deep),
        "uv": lambda: _plan_uv(staged["uv"], installed_root),
    }
    out = []
    for name in ORDER:
        try:
            out.append(makers[name]())
        except Exception as e:
            _warn("update: could not compare the installed %s with the release's pin: %s",
                  LABELS[name], e)
            p = _new(name)
            p["note"] = (f"could not compare the installed {LABELS[name]} with this "
                         f"release's pin ({e}); it was left as it is")
            out.append(p)
    return out


def summary(plans) -> Optional[str]:
    """The notes of *plans* as one line, or None when none has one."""
    notes = [p["note"] for p in plans or [] if p.get("note")]
    return "; ".join(notes) or None


def needs_finish(plans) -> bool:
    """Whether *plans* (from :func:`plan`) leave anything for
    :func:`finish_pending`: a component to update or one only it can decide."""
    return any(p["update"] or p.get("check") for p in plans)


# ------------------------------ who holds them -------------------------------

def _running_in(folder: Path) -> Optional[list]:
    from localm import install_manifest
    if not folder.is_dir():
        return []
    return install_manifest.running_from([folder])


def blockers(plans, installed_root) -> list:
    """Processes that keep a component of *plans* from being updated and that
    the update does not stop: a uv running from inside the install. ``[{"pid",
    "name"}]``."""
    out = []
    for p in plans:
        if p["name"] == "uv" and p["update"]:
            for pid, exe in _running_in(Path(installed_root) / ".uv") or []:
                out.append({"pid": pid, "name": f"uv ({exe})"})
    return out


def _stop_comfy(root: Path, say: Callable) -> list:
    """Stop every process running the managed ComfyUI at *root* (its venv's
    Python, and whatever that started). Returns the holders left as ``[{"pid",
    "name"}]``."""
    import time

    from localm import install_manifest, updater_runtime
    found = _running_in(root)
    if not found:
        return []
    procs = install_manifest.list_processes() or []
    by_pid = {p[0]: p for p in procs}
    deadline = time.monotonic() + _STOP_WAIT_S
    left = []
    for pid, _exe in found:
        if pid not in by_pid:
            continue
        say(f"Stopping the managed ComfyUI (pid {pid})...")
        left += updater_runtime._stop_tree(updater_runtime._subtree(by_pid, pid), deadline)
    return [{"pid": pid, "name": "the managed ComfyUI"} for pid in left]


def _stop_torch_holders(installed_root: Path, say: Callable) -> list:
    """Stop this install's own processes that hold the PyTorch stack's native
    libraries (``updater_runtime.stop_own_holders``). Returns the holders left."""
    from localm import updater_runtime
    site = site_packages(installed_root / ".venv")
    if site is None:
        return []
    left = []
    for d in [site / rel for rel in _TORCH_HELD] + sorted(site.glob(_TORCH_HELD_GLOB)):
        if d.is_dir():
            left += updater_runtime.stop_own_holders(d, installed_root, say=say)
    return left


# ------------------------------ pending marker -------------------------------

def _updates_dir(create: bool = True) -> Path:
    from localm import updater
    return updater._updates_dir(create=create)


def write_pending(plans, *, root, version: str) -> None:
    """Record that the components of *plans* (:func:`needs_finish`) must be
    finished in the install at *root*, for :func:`finish_pending`. Raises OSError."""
    data = {"root": str(Path(root).resolve()), "version": version,
            "components": [p["name"] for p in plans if p["update"] or p.get("check")]}
    path = _updates_dir() / PENDING_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def read_pending() -> Optional[dict]:
    """The pending component update, or None. A marker that cannot be parsed is
    reported in the log and treated as none."""
    path = _updates_dir(create=False) / PENDING_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as e:
        _warn("update: ignoring an unreadable pending component update %s: %s", path, e)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("components"), list):
        return None
    return data


def clear_pending() -> None:
    try:
        (_updates_dir(create=False) / PENDING_FILE).unlink()
    except FileNotFoundError:
        pass


def _write_status(status: dict, root: Optional[Path] = None,
                  version: Optional[str] = None) -> None:
    import time
    status = dict(status, root=str(Path(root).resolve()) if root else None, version=version,
                  at=time.time())
    try:
        (_updates_dir() / STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    except OSError as e:
        _warn("update: could not record the component update result: %s", e)


def _mark_checked(root: Path, version: Optional[str]) -> None:
    """Record that every component of the install at *root*, at *version*, is
    at its pins (or kept by the user's choice)."""
    try:
        (_updates_dir() / CHECKED_FILE).write_text(json.dumps(
            {"root": str(Path(root).resolve()), "version": version}), encoding="utf-8")
    except OSError as e:
        _warn("update: could not record the component check: %s", e)


def _is_checked(root: Path, version: Optional[str]) -> bool:
    data = _read_json(_updates_dir(create=False) / CHECKED_FILE) or {}
    return bool(version) and data.get("version") == version and data.get("root") == str(
        Path(root).resolve())


def last_status() -> Optional[dict]:
    """The result of the last :func:`finish_pending` that changed or tried to
    change something, or None."""
    return _read_json(_updates_dir(create=False) / STATUS_FILE)


def catch_up(root=None) -> Optional[list]:
    """Record a pending component update when none is pending and a component
    installed for the install at *root* (default the running install) is not
    at the pin that install's own release ships (:func:`plan` of the install
    against itself). Returns the plans when it recorded one, else None. Raises
    OSError when the marker cannot be written.

    Checks once per install root and version: after the components were found
    at their pins (:func:`finish_pending` or this check) it returns None at
    once. After a failed attempt it waits :data:`RETRY_AFTER_S` seconds before
    trying again."""
    import time

    from localm import _version, updater
    root = Path(root) if root else updater.repo_root()
    if read_pending() is not None:
        return None
    version = _version.read_version()
    if _is_checked(root, version):
        return None
    last = last_status() or {}
    if (last.get("state") in ("failed", "waiting") and last.get("version") == version
            and last.get("root") == str(root.resolve())
            and time.time() - float(last.get("at") or 0) < RETRY_AFTER_S):
        return None
    plans = plan(root, root)
    if not any(p["update"] for p in plans):
        _mark_checked(root, version)
        return None
    write_pending(plans, root=root, version=version)
    return plans


# ------------------------------ snapshot / restore ---------------------------

def snapshot_dir(backup_dir) -> Path:
    """Where the update whose backup is *backup_dir* keeps the components it
    replaced: a sibling ``<backup_dir>-components``."""
    backup_dir = Path(backup_dir)
    return backup_dir.with_name(backup_dir.name + "-components")


def _venv_copy_path(venv: Path, dest: Path) -> Path:
    """Where the snapshot *dest* keeps its copy of the venv at *venv*: a folder
    beside the venv (same volume, so its files are hard links)."""
    return venv.parent / f".{venv.name}-{dest.name}"


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_owner(copy: Path) -> Optional[str]:
    """The snapshot folder a venv copy at *copy* belongs to (from its
    ``COPY_MARKER``), or None when *copy* is not a venv copy this module made."""
    try:
        return (copy / COPY_MARKER).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _own_copy(info: Optional[dict], src: Path) -> Optional[Path]:
    """The venv copy named by the snapshot entry *info* of the snapshot *src*,
    only when that folder is one :func:`_clone_venv` made for *src*: its name is
    ``.<venv>-<src name>`` and its marker names *src*. None otherwise."""
    raw = (info or {}).get("venv_copy")
    if not raw:
        return None
    copy = Path(str(raw))
    if not copy.name.startswith(".") or not copy.name.endswith(f"-{src.name}"):
        return None
    try:
        owner = _copy_owner(copy)
        return copy if owner and Path(owner).resolve() == src.resolve() else None
    except OSError:
        return None


def _clone_venv(venv: Path, copy: Path, owner: Path) -> None:
    """Recreate every file of *venv* in *copy* (replaced first) as a hard link,
    or a copy where a link cannot be made, and mark *copy* as belonging to the
    snapshot *owner*. Raises OSError."""
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    (copy / COPY_MARKER).write_text(str(owner.resolve()), encoding="utf-8")
    for dirpath, _dirnames, filenames in os.walk(venv):
        rel = os.path.relpath(dirpath, venv)
        target = copy / rel if rel != "." else copy
        target.mkdir(parents=True, exist_ok=True)
        for f in filenames:
            _link(Path(dirpath) / f, target / f)


def _record(site: Path, dist_dir: str) -> list:
    """The paths, relative to *site*, of the distribution *dist_dir*'s files."""
    import csv
    try:
        with open(site / dist_dir / "RECORD", encoding="utf-8", newline="") as fh:
            return [row[0] for row in csv.reader(fh) if row]
    except OSError:
        return []


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _restore_venv(copy: Path, venv: Path) -> list:
    """Put back, in *venv*, every distribution whose installed version differs
    from the one kept in *copy* (by :func:`_clone_venv`), removing ones added
    since: first every changed distribution's files are removed, then every
    kept one's files are put back, so a file two distributions share ends up
    restored. The install's own package is left alone, and nothing outside
    *venv* is removed. Returns the failures."""
    site_now, site_was = site_packages(venv), site_packages(copy)
    if site_now is None or site_was is None:
        return [f"could not find the packages of {venv} or of its copy {copy}"]
    now, was = dists(site_now), dists(site_was)
    errors = []
    changed = [name for name in sorted(set(now) | set(was))
               if name not in _OWN_DISTS
               and (now.get(name) or {}).get("dir") != (was.get(name) or {}).get("dir")]
    for name in changed:
        if name in now:
            for rel in _record(site_now, now[name]["dir"]):
                p = (site_now / rel)
                if _inside(p, venv) and p.is_file():
                    try:
                        p.unlink()
                    except OSError as e:
                        errors.append(f"remove {rel}: {e}")
            shutil.rmtree(site_now / now[name]["dir"], ignore_errors=True)
    for name in changed:
        if name in was:
            for rel in _record(site_was, was[name]["dir"]):
                src = (site_was / rel)
                if not _inside(src, copy) or not src.is_file():
                    continue
                dst = venv / src.resolve().relative_to(copy.resolve())
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    _link(src, dst)
                except OSError as e:
                    errors.append(f"restore {rel}: {e}")
    return [f"{venv}: " + "; ".join(errors[:5])] if errors else []


def _snapshot_comfy(root: Path, dest: Path) -> dict:
    """Keep the managed ComfyUI's commit, its edits to tracked files (the localm
    patch set), its marker, its custom nodes' commits and its venv. Raises
    OSError when any of them cannot be read or written."""
    head = _git_head(root)
    if head is None:
        raise OSError(f"could not read the commit of the managed ComfyUI at {root}")
    try:
        r = _git_bytes(root, "diff", "--binary", "HEAD")
    except (OSError, subprocess.SubprocessError) as e:
        raise OSError(f"could not read the edits of the managed ComfyUI: {e}") from None
    if r.returncode != 0:
        raise OSError("could not read the edits of the managed ComfyUI: "
                      + r.stderr.decode("utf-8", "replace").strip()[-300:])
    (dest / "comfy.patch").write_bytes(r.stdout)
    shutil.copy2(root / _COMFY_MARKER, dest / "comfy-marker.json")
    nodes = {}
    nodes_dir = root / "custom_nodes"
    if nodes_dir.is_dir():
        for d in sorted(nodes_dir.iterdir()):
            if d.is_dir() and (d / ".git").exists():
                h = _git_head(d)
                if h:
                    nodes[d.name] = h
    venv = _comfy_venv(root)
    copy = _venv_copy_path(venv, dest)
    _clone_venv(venv, copy, dest)
    return {"commit": head, "nodes": nodes, "venv_copy": str(copy)}


def _restore_comfy(root: Path, src: Path, info: dict, say: Callable) -> list:
    """Put the managed ComfyUI kept in *src* back at *root*: its commit, edits,
    marker, custom nodes' commits and venv. Returns the failures."""
    from localm.media import managed_comfy as mc
    errors = []
    left = _stop_comfy(root, say)
    if left:
        return [f"the managed ComfyUI (pid {left[0]['pid']}) did not stop, so it was not "
                "restored"]
    acquired, busy = mc._acquire_update_lock("update")
    if not acquired:
        return [f"the managed ComfyUI was not restored: {busy}"]
    try:
        say(f"Restoring the managed ComfyUI ({info['commit'][:12]})...")
        try:
            r = _git(root, "checkout", "--force", "--quiet", info["commit"])
            if r.returncode != 0:
                return [f"managed ComfyUI: could not check out {info['commit'][:12]}: "
                        + r.stderr.strip()[-300:]]
            patch = src / "comfy.patch"
            if patch.is_file() and patch.stat().st_size:
                r = _git(root, "apply", "--binary", "--whitespace=nowarn", str(patch))
                if r.returncode != 0:
                    errors.append("managed ComfyUI: could not re-apply its edits: "
                                  + r.stderr.strip()[-300:])
            for name, head in (info.get("nodes") or {}).items():
                node = root / "custom_nodes" / name
                if node.is_dir() and _git_head(node) != head:
                    r = _git(node, "checkout", "--force", "--quiet", head)
                    if r.returncode != 0:
                        errors.append(f"managed ComfyUI: custom node {name}: "
                                      + r.stderr.strip()[-300:])
        except (OSError, subprocess.SubprocessError) as e:
            return [f"managed ComfyUI: could not run git: {e}"]
        try:
            shutil.copy2(src / "comfy-marker.json", root / _COMFY_MARKER)
        except OSError as e:
            errors.append(f"managed ComfyUI: could not restore its marker: {e}")
        if info.get("venv_copy"):
            copy = _own_copy(info, src)
            errors += (_restore_venv(copy, _comfy_venv(root)) if copy is not None else
                       [f"managed ComfyUI: {info['venv_copy']} is not a venv copy of {src}"])
    finally:
        mc._release_update_lock()
    return errors


def _snapshot_torch(installed_root: Path, dest: Path) -> dict:
    venv = installed_root / ".venv"
    copy = _venv_copy_path(venv, dest)
    _clone_venv(venv, copy, dest)
    return {"venv_copy": str(copy)}


def _restore_torch(installed_root: Path, src: Path, info: dict, say: Callable) -> list:
    left = _stop_torch_holders(installed_root, say)
    if left:
        from localm import updater_runtime
        return ["the PyTorch stack is in use, so it was not restored: "
                + updater_runtime.describe_holders(left)]
    say("Restoring the PyTorch stack...")
    copy = _own_copy(info, src)
    if copy is None:
        return [f"PyTorch stack: {info.get('venv_copy')} is not a venv copy of {src}"]
    return _restore_venv(copy, installed_root / ".venv")


def _snapshot_uv(installed_root: Path, dest: Path) -> dict:
    shutil.copytree(installed_root / ".uv", dest / _KEPT_UV)
    return {}


def _restore_uv(installed_root: Path, src: Path, say: Callable) -> list:
    live = installed_root / ".uv"
    kept = src / _KEPT_UV
    if not kept.is_dir():
        return [f"uv: the kept copy {kept} is missing"]
    running = _running_in(live)
    if running:
        return [f"uv: {live} is in use (pid {running[0][0]}), so it was not restored"]
    say("Restoring the uv inside localm...")
    try:
        if live.exists():
            shutil.rmtree(live)
        shutil.copytree(kept, live)
    except OSError as e:
        return [f"uv: could not restore {live}: {e}"]
    return []


def discard(src) -> None:
    """Remove the snapshot *src* and the venv copies it keeps elsewhere. A
    folder that cannot be removed is reported in the log."""
    src = Path(src)
    state = _read_json(src / STATE_FILE) or {}
    for name, info in (state.get("components") or {}).items():
        _drop_one(src, name, info)
    if src.exists():
        try:
            shutil.rmtree(src)
        except OSError as e:
            _warn("update: could not remove the kept components %s: %s", src, e)


def _drop_one(src: Path, name: str, info: Optional[dict]) -> None:
    """Remove what the snapshot *src* keeps for component *name* (entry *info*)
    outside its own state file: a venv copy it made, the kept uv folder, the
    kept ComfyUI files. A venv copy that is not one it made is left alone."""
    copy = _own_copy(info, src)
    if copy is not None:
        try:
            shutil.rmtree(copy)
        except OSError as e:
            _warn("update: could not remove the kept venv copy %s: %s", copy, e)
    elif (info or {}).get("venv_copy"):
        _warn("update: not removing %s: it is not a venv copy of %s",
              (info or {}).get("venv_copy"), src)
    extra = {"uv": [_KEPT_UV], "comfyui": ["comfy.patch", "comfy-marker.json"]}.get(name, [])
    for rel in extra:
        p = src / rel
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except OSError as e:
            _warn("update: could not remove the kept %s: %s", p, e)


def snapshot(names, dest, *, root, home=None, add: bool = False) -> dict:
    """Keep the components *names* of the install at *root* in *dest* and
    return the state :func:`restore` reads, also written to
    ``<dest>/state.json``. An earlier snapshot in *dest* is discarded first;
    with *add* it is kept instead and only the components it does not hold yet
    are added (what it already holds is the older state, the one a rollback
    needs). Raises OSError; what this call kept is removed again when the
    snapshot cannot be completed."""
    dest = Path(dest)
    root = Path(root)
    loaded = _read_json(dest / STATE_FILE) if add else None
    fresh = not loaded or not isinstance(loaded.get("components"), dict)
    state: dict = {"root": str(root.resolve()), "components": {}}
    if fresh:
        discard(dest)
    elif loaded is not None:
        state = loaded
    dest.mkdir(parents=True, exist_ok=True)
    added: list = []
    try:
        for name in names:
            if name in state["components"]:
                continue
            added.append(name)
            if name == "comfyui":
                state["components"][name] = _snapshot_comfy(comfy_root(home), dest)
            elif name == "pytorch":
                state["components"][name] = _snapshot_torch(root, dest)
            elif name == "uv":
                state["components"][name] = _snapshot_uv(root, dest)
            else:
                state["components"][name] = {}
            (dest / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    except (OSError, shutil.Error) as e:
        if fresh:
            discard(dest)
        else:
            for name in added:
                _drop_one(dest, name, state["components"].pop(name, None)
                          or {"venv_copy": str(_venv_copy_path(
                              (comfy_root(home) / "venv") if name == "comfyui"
                              else root / ".venv", dest))})
            try:
                (dest / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
            except OSError as w:
                _warn("update: could not rewrite the kept components state %s: %s", dest, w)
        raise OSError(str(e)) from None
    return state


def restore(src, *, root, home=None, say: Optional[Callable] = None,
            names=None) -> list:
    """Put the components kept in *src* (by :func:`snapshot`), or only those of
    them in *names*, back in the install at *root*. Returns the failures as
    strings (empty on success, or when *src* holds nothing). sd.cpp and
    KoboldCpp need nothing: their earlier runtime directories are never
    replaced."""
    say = say or (lambda _m: None)
    src = Path(src)
    state = _read_json(src / STATE_FILE)
    if not state:
        return []
    errors = []
    comps = {k: v for k, v in (state.get("components") or {}).items()
             if names is None or k in names}
    if "pytorch" in comps:
        errors += _restore_torch(Path(root), src, comps["pytorch"], say)
    if "comfyui" in comps:
        errors += _restore_comfy(comfy_root(home), src, comps["comfyui"], say)
    if "uv" in comps:
        errors += _restore_uv(Path(root), src, say)
    return errors


def snapshot_names(src) -> list:
    """The components a snapshot at *src* holds."""
    state = _read_json(Path(src) / STATE_FILE) or {}
    return [n for n in ORDER if n in (state.get("components") or {})]


def installed_names(names, root, *, home=None) -> list:
    """Those of the components *names* that are installed now for the install
    at *root* (a ComfyUI, PyTorch or uv removed since is not restored)."""
    from localm import uv_tool
    out = []
    for n in names:
        if n == "comfyui" and not _comfy_installed(comfy_root(home)):
            continue
        if n == "uv" and uv_tool.contained_uv(root) is None:
            continue
        if n == "pytorch" and site_packages(Path(root) / ".venv") is None:
            continue
        out.append(n)
    return out


# ------------------------------ the update itself ----------------------------

def _tree_text(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def _commands(p: dict, root: Path) -> list:
    """The argv lists that update component *p* with the localm tree at *root*."""
    if p["name"] == "pytorch":
        from localm import updater
        uv = updater._uv_for_deps(root)
        return [[uv, *c] for c in p["items"]]
    if p["name"] == "sdcpp":
        return [[sys.executable, "-m", "localm", "setup-sdcpp", "--backend", b]
                for b in p["items"]]
    if p["name"] == "koboldcpp":
        return [[sys.executable, "-m", "localm", "setup-music", "--backend",
                 _KOBOLD_BUILD_BACKEND.get(b, b), "--no-models", "--no-test"]
                for b in p["items"]]
    if p["name"] == "comfyui":
        cmd = [sys.executable, "-m", "localm", "comfy", "update"]
        if '"--reinstall-requirements"' in _tree_text(root, "localm/cli/comfy.py"):
            cmd.append("--reinstall-requirements")
        return [cmd]
    return []


def _verify(p: dict, root: Path, home: Path) -> Optional[str]:
    """None when component *p* is now at its target, else what is wrong."""
    target = p["target"]
    if p["name"] in ("sdcpp", "koboldcpp"):
        marker, key = ((_SDCPP_MARKER, "backend") if p["name"] == "sdcpp"
                       else (_KOBOLD_MARKER, "build"))
        base = home / "runtimes" / p["name"]
        missing = []
        for i in p["items"]:
            meta = _read_json(base / f"{target}-{i}" / marker) or {}
            if meta.get("tag") != target or meta.get(key) != i:
                missing.append(i)
        return f"{', '.join(missing)} not installed at {target}" if missing else None
    if p["name"] == "pytorch":
        again = _plan_torch(root, root, True)
        return f"it still differs: {again['note']}" if again["update"] else None
    if p["name"] == "comfyui":
        again = _plan_comfy(staged_pins(root)["comfyui"], home, root, True)
        return ("it still differs: " + ", ".join(again["items"])) if again["update"] else None
    if p["name"] == "uv":
        from localm import uv_tool
        found = uv_tool.contained_uv(root)
        version = uv_tool.uv_version(found) if found else None
        return None if version == target else f"it reports {version or 'no version'}"
    return None


def _update_uv(p: dict, root: Path, say: Callable, bootstrap: Optional[Callable]) -> Optional[str]:
    """Install the pinned uv into a staging folder beside ``<root>/.uv``, then
    swap it in. None on success, else what failed (``.uv`` unchanged)."""
    from localm import uv_tool
    staging = root / ".uv-update"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        (bootstrap or uv_tool.bootstrap_uv)(root, emit=say, dest=staging)
    except uv_tool.UvBootstrapError as e:
        shutil.rmtree(staging, ignore_errors=True)
        return str(e)
    found = uv_tool._uv_in(staging)
    got = uv_tool.uv_version(found) if found else None
    if got != p["target"]:
        shutil.rmtree(staging, ignore_errors=True)
        return f"the installed uv reports {got or 'no version'}, not {p['target']}"
    live = root / ".uv"
    old = root / ".uv-previous"
    try:
        shutil.rmtree(old, ignore_errors=True)
        os.replace(live, old)
        try:
            os.replace(staging, live)
        except OSError:
            os.replace(old, live)
            raise
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        return f"could not swap in the new uv: {e}"
    shutil.rmtree(old, ignore_errors=True)
    return None


def _remove_created(p: dict, home: Path, existed: set) -> None:
    """Remove the runtime directories component *p* created in this run (none of
    which existed before), so a later attempt starts clean."""
    if p["name"] not in ("sdcpp", "koboldcpp"):
        return
    base = home / "runtimes" / p["name"]
    for item in p["items"]:
        d = base / f"{p['target']}-{item}"
        if d.exists() and str(d) not in existed:
            try:
                shutil.rmtree(d)
            except OSError as e:
                _warn("update: could not remove the incomplete %s %s: %s", p["label"], d, e)


def finish_pending(*, root=None, home=None, runner: Optional[Callable] = None,
                   env: Optional[dict] = None, say: Optional[Callable] = None,
                   bootstrap: Optional[Callable] = None) -> Optional[dict]:
    """Carry out the pending component update, if there is one for the install
    at *root* (default the running install). One recorded for another install
    root, or for another version than the one installed at *root*, is dropped
    and None returned.

    The components are compared with *root*'s own pins again (:func:`plan`), so
    one the user changed since is left as it is. Returns None when nothing is
    pending, else ``{"state", "components", "note", "error"}`` (also recorded
    for ``localm doctor`` when something was attempted), ``components`` the
    plans with each updated one's ``"result"``. ``state`` is:

    - ``"updated"``: every component that needed it is at the release's pin;
    - ``"current"``: none needed it;
    - ``"waiting"``: a process that is not this install's, or did not stop,
      holds a component; nothing was changed;
    - ``"failed"``: a component could not be updated; every component this run
      changed was put back (``error`` says what failed, including a restore
      that failed).

    Before anything is replaced, this install's processes that hold a
    component are stopped (the managed ComfyUI; the processes holding the
    PyTorch stack, by ``updater_runtime.stop_own_holders``) and a copy of each
    component is kept (:func:`snapshot`) beside the update backup. *say*
    receives each step as one line. *runner* runs an argv and returns its exit
    code (tests); *env* is its environment; *bootstrap* replaces
    ``uv_tool.bootstrap_uv`` (tests)."""
    from localm import updater, updater_runtime
    pending = read_pending()
    if pending is None:
        return None
    root = Path(root) if root else updater.repo_root()
    if Path(pending.get("root") or "").resolve() != root.resolve():
        clear_pending()
        return None
    try:
        installed_version = (root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        installed_version = None
    if pending.get("version") and installed_version and pending["version"] != installed_version:
        clear_pending()
        return None
    say = say or (lambda _m: None)
    home = _home(home)
    plans = plan(root, root, home=home)
    todo = [p for p in plans if p["update"]]
    status = {"state": None, "components": plans, "note": summary(plans), "error": None}
    if not todo:
        clear_pending()
        status["state"] = "current"
        _mark_checked(root, installed_version)
        return status

    held = blockers(todo, root)
    names = [p["name"] for p in todo]
    if not held and "pytorch" in names:
        held += _stop_torch_holders(root, say)
    if not held and "comfyui" in names:
        held += _stop_comfy(comfy_root(home), say)
    if held:
        status.update(state="waiting", note=(
            "the components were not updated: "
            + updater_runtime.in_use_message(held, "start localm again")))
        _write_status(status, root, installed_version)
        return status

    snap = snapshot_dir(updater._backup_dir(create=True))
    try:
        snapshot(names, snap, root=root, home=home, add=True)
    except OSError as e:
        clear_pending()
        status.update(state="failed", error=f"could not keep a copy of the components: {e}",
                      note=(f"the components were not updated because a copy of the current "
                            f"ones could not be kept ({e})"))
        _write_status(status, root, installed_version)
        return status

    if env is None:
        env = updater._deps_env(root)
    run = runner or (lambda c: subprocess.run(c, cwd=str(root), env=env).returncode)
    existed = set()
    for n in ("sdcpp", "koboldcpp"):
        base = home / "runtimes" / n
        if base.is_dir():
            existed |= {str(d) for d in base.iterdir() if d.is_dir()}
    done = []
    problem = None
    for p in todo:
        done.append(p)
        say(f"Updating the {p['label']}...")
        try:
            if p["name"] == "uv":
                problem = _update_uv(p, root, say, bootstrap)
            else:
                for cmd in _commands(p, root):
                    rc = int(run(cmd))
                    if rc != 0:
                        what = (" ".join(cmd[2:4]) if cmd[1:2] == ["-m"]
                                else f"{Path(str(cmd[0])).name} {' '.join(cmd[1:3])}")
                        problem = f"{what} exited {rc}"
                        break
        except Exception as e:
            problem = f"could not run its update: {e}"
        if problem is None:
            problem = _verify(p, root, home)
        if problem is not None:
            problem = f"the {p['label']} was not updated ({problem})"
            p["result"] = "failed"
            break
        p["result"] = "updated"
    clear_pending()
    if problem is None:
        rest = summary([p for p in plans if not p["update"]])
        status.update(state="updated", note="; ".join(
            "updated the " + p["label"] + (f": {p['current']} -> {p['target']}"
                                           if p["name"] in ("sdcpp", "koboldcpp", "uv")
                                           else "")
            for p in todo) + (f"; {rest}" if rest else ""))
        _write_status(status, root, installed_version)
        _mark_checked(root, installed_version)
        return status
    for p in done:
        _remove_created(p, home, existed)
    errors = restore(snap, root=root, home=home, say=say, names=names)
    kept = ("the previous versions were put back" if not errors
            else "putting the previous versions back FAILED: " + "; ".join(errors))
    status.update(state="failed", error=problem + ("; " + "; ".join(errors) if errors else ""),
                  note=f"{problem}; {kept}")
    _warn("update: component update failed: %s", status["error"])
    _write_status(status, root, installed_version)
    return status
