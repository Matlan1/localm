# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adding, on the user's request, the Python packages a trusted model's own code needs.

Only modules listed in :data:`ON_DEMAND` are ever installed, each from the PyPI
project that list names; a project name is never derived from a module name a
model's code imports. :data:`REFUSED` lists the modules localm has reviewed and
cannot install, with the reason shown to the user.

An install has three steps:

1. :func:`plan` resolves the requested projects with ``uv pip compile``, every
   installed distribution held at its installed version, wheels only, with
   hashes. It fails when the resolution would change an installed package or
   needs a source archive. The plan lists the distributions it would add.
2. The caller shows :attr:`Plan.adds` and asks the user.
3. :func:`install` installs exactly those pins (no dependencies, hash-checked,
   wheels only) with the install's own uv, imports every requested module in a
   fresh interpreter, removes what it added when one does not import, and
   records the install in the install record.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

PYPI_INDEX = "https://pypi.org/simple"
_PYPI_FILES = "https://files.pythonhosted.org/"
_TORCH_INDEX = "https://download.pytorch.org/whl/"
_TORCH_FAMILY = frozenset({"torch", "torchvision", "torchaudio"})
_NEVER_ADDED = frozenset({"torch", "transformers", "localm"})
_TORCH_LABEL = re.compile(r"^(?:cpu|cu\d+|rocm\d+(?:\.\d+)*|xpu)$")
_PLAN_TTL_S = 15 * 60
_IMPORT_TIMEOUT_S = 600
_INSTALL_TIMEOUT_S = 1800
_PLAN_TIMEOUT_S = 300
_SCAN_CACHE_MAX = 256

ProgressCb = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class OnDemand:
    """One reviewed package localm installs on request."""
    project: str
    floor: str
    platforms: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()


# Importable module -> the PyPI project localm installs for it. Each needs the
# installed torch, so torch-built parts are taken from torch's own index.
ON_DEMAND: dict[str, OnDemand] = {
    "timm": OnDemand("timm", "1.0.30"),
    "open_clip": OnDemand("open-clip-torch", "3.3.0"),
    "torchaudio": OnDemand("torchaudio", "2.0"),
    "fla": OnDemand("flash-linear-attention", "0.6.0", platforms=("linux",), needs=("triton",)),
}

_KERNELS = "a CUDA kernel package that PyPI publishes only as source code to compile"
_SOURCE_ONLY = "PyPI publishes it only as source code to build, which localm does not do"

# Importable module -> why localm cannot add it.
REFUSED: dict[str, str] = {
    "flash_attn": _KERNELS, "mamba_ssm": _KERNELS, "causal_conv1d": _KERNELS, "apex": _KERNELS,
    "deepspeed": _KERNELS,
    "xformers": "its wheels match one PyTorch build only",
    "vllm": "it is a separate inference engine",
    "whisper": _SOURCE_ONLY, "minicpmo": _SOURCE_ONLY,
    "funasr": "it depends on packages PyPI publishes only as source code to build",
    "smirk": "it requires transformers 4",
    "ebfm": "the PyPI package of that name is an unrelated project",
    "fairseq_signals_backbone": "it is not published on PyPI",
    "configuration_robobrain_audio": "the model's repository does not contain that module",
    "distutils": "it was removed from Python 3.12",
}


class PlanError(Exception):
    """No install can be planned; the message says why, in words for a user."""


@dataclass
class Plan:
    """What an install would add. ``adds`` is ``[(distribution, version)]``."""
    modules: list
    adds: list
    requirements: str
    torch_backend: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.monotonic)

    def source(self, name: str) -> str:
        """The index *name* is downloaded from."""
        if self.torch_backend and _canon(name) in _TORCH_FAMILY:
            return f"download.pytorch.org ({self.torch_backend})"
        return "pypi.org"

    def as_dict(self) -> dict:
        return {"plan_id": self.id, "modules": list(self.modules),
                "packages": [package_label(m) for m in self.modules],
                "adds": [{"name": n, "version": v, "source": self.source(n)}
                         for n, v in self.adds]}


@dataclass
class InstallResult:
    """Outcome of :func:`install`. ``ok`` only when every requested module imports."""
    ok: bool
    installed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    error: str = ""


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_label(module: str) -> str:
    """The project name a user knows the package for *module* by."""
    entry = ON_DEMAND.get(module)
    if entry is not None:
        return entry.project
    from localm.model_manager.missing_packages import package_label as _label
    return _label(module)


def unavailable_reason(module: str) -> Optional[str]:
    """Why localm cannot add *module* on this machine, or None when it can.
    Never touches the network."""
    from localm.model_manager.missing_packages import (
        SHIPPED_MODULES,
        platform_label,
        shipped_here,
    )
    entry = ON_DEMAND.get(module)
    if entry is None:
        if module in SHIPPED_MODULES and not shipped_here(module):
            return f"it is not available for this platform ({platform_label()})"
        reason = REFUSED.get(module)
        return reason or "it is not on the list of packages localm has reviewed for installing"
    if entry.platforms and not any(sys.platform.startswith(p) for p in entry.platforms):
        return "it is not available for this operating system"
    for needed in entry.needs:
        if not _importable(needed):
            return f"it needs {needed}, which this install does not have"
    return None


def installable(module: str) -> bool:
    """True when localm can add *module* on this machine."""
    return unavailable_reason(module) is None


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_distributions() -> dict:
    """``{canonical name: version}`` of every distribution in this interpreter's
    environment except localm itself."""
    out: dict = {}
    for dist in md.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if not name or not dist.version:
            continue
        canon = _canon(name)
        if canon != "localm":
            out.setdefault(canon, dist.version)
    return out


def torch_backend(installed: dict) -> Optional[str]:
    """The ``--torch-backend`` value matching the installed torch (its local
    version label: ``cpu``, ``cu126``, ``rocm6.4``, ``xpu``), or None for a torch
    from PyPI, no torch, or a label of another shape."""
    version = installed.get("torch")
    if not version or "+" not in version:
        return None
    label = version.split("+", 1)[1]
    return label if _TORCH_LABEL.match(label) else None


def torch_from_elsewhere() -> Optional[str]:
    """The installed torch's version when it was installed from a file or a URL
    rather than an index (its ``direct_url.json``), else None. Packages built
    against such a torch cannot be matched to it from PyPI or PyTorch's index."""
    try:
        dist = md.distribution("torch")
    except md.PackageNotFoundError:
        return None
    try:
        direct = dist.read_text("direct_url.json")
    except OSError:
        return None
    return dist.version if direct else None


def _unmatched_torch(version: str) -> PlanError:
    return PlanError(f"this install's PyTorch ({version}) did not come from PyPI or PyTorch's "
                     "own index, so localm cannot find packages built to match it.")


def check_network(backend: Optional[str]) -> None:
    """Raise PlanError when the network policy refuses the indexes an install uses."""
    from localm.netpolicy import NetworkPolicyError, check_url
    urls = [PYPI_INDEX + "/", _PYPI_FILES] + ([_TORCH_INDEX] if backend else [])
    for url in urls:
        try:
            check_url(url)
        except NetworkPolicyError as e:
            raise PlanError(f"localm cannot download the packages: {e}") from e


def _uv() -> str:
    from localm import uv_tool
    uv = uv_tool.bundled_uv()
    if not uv:
        raise PlanError("localm cannot find the uv its install uses, so it cannot add packages. "
                        "Run localm's setup again to restore it.")
    return uv


def _run_kwargs() -> dict:
    from localm import config
    env = {k: v for k, v in config.contained_pip_env().items()
           if not k.upper().startswith("UV_") or k.upper() == "UV_CACHE_DIR"}
    kw: dict = {"env": env, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def _index_args(backend: Optional[str]) -> list:
    args = ["--no-config", "--python", sys.executable, "--default-index", PYPI_INDEX,
            "--only-binary", ":all:"]
    return args + (["--torch-backend", backend] if backend else [])


def parse_pins(text: str) -> dict:
    """``{canonical name: (name, version, [hashes])}`` from ``uv pip compile
    --generate-hashes`` output. A requirement that is not an exact ``==`` pin
    raises PlanError."""
    joined = re.sub(r"\\\s*\n", " ", text)
    pins: dict = {}
    for line in joined.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        head, *opts = line.split()
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", head)
        if not m:
            raise PlanError(f"uv returned a requirement localm cannot read: {head}")
        hashes = [o.split("=", 1)[1] for o in opts if o.startswith("--hash=")]
        pins[_canon(m.group(1))] = (m.group(1), m.group(2), hashes)
    return pins


def plan(modules: list, *, run: Callable = subprocess.run) -> Plan:
    """Resolve what installing the packages for *modules* would add.

    Raises PlanError when a module is not one localm can add here, the network
    policy refuses the indexes, uv is missing, or the resolution fails (it would
    change an installed package, or a needed package has no wheel)."""
    modules = sorted(dict.fromkeys(modules))
    if not modules:
        raise PlanError("no packages to install")
    for m in modules:
        reason = unavailable_reason(m)
        if reason is not None:
            raise PlanError(f"localm cannot add {package_label(m)}: {reason}.")
    installed = installed_distributions()
    elsewhere = torch_from_elsewhere()
    if elsewhere is not None:
        raise _unmatched_torch(elsewhere)
    backend = torch_backend(installed)
    check_network(backend)
    uv = _uv()
    requested = "".join(f"{ON_DEMAND[m].project}>={ON_DEMAND[m].floor}\n" for m in modules)
    with tempfile.TemporaryDirectory(prefix="localm-packages-") as tmp:
        constraints = Path(tmp) / "installed.txt"
        constraints.write_text("".join(f"{n}=={v}\n" for n, v in sorted(installed.items())),
                               encoding="utf-8")
        cmd = [uv, "pip", "compile", "-", "-c", str(constraints), "--generate-hashes",
               "--no-header", "--no-annotate", "--quiet", *_index_args(backend)]
        try:
            proc = run(cmd, input=requested, capture_output=True, encoding="utf-8",
                       errors="replace",
                       timeout=_PLAN_TIMEOUT_S, **{k: v for k, v in _run_kwargs().items()
                                                      if k != "stdin"})
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PlanError(f"uv could not resolve the packages: {e}") from e
    if proc.returncode != 0:
        raise _compile_failure(proc.stderr or proc.stdout, installed)
    pins = parse_pins(proc.stdout)
    adds = []
    lines = []
    for canon, (name, version, hashes) in sorted(pins.items()):
        have = installed.get(canon)
        if have is not None:
            if have != version:
                raise PlanError(f"installing would change {name} {have} to {version}")
            continue
        if canon in _NEVER_ADDED:
            raise PlanError(f"installing would add {name}, which localm's own setup installs")
        if not hashes:
            raise PlanError(f"uv gave no hash for {name} {version}")
        adds.append((name, version))
        lines.append(f"{name}=={version} " + " ".join(f"--hash={h}" for h in hashes))
    return Plan(modules=modules, adds=adds, requirements="\n".join(lines) + "\n",
                torch_backend=backend)


_NETWORK_FAILURE = re.compile(r"failed to fetch|error sending request|dns error|"
                              r"failed to connect|connection refused|timed out|offline",
                              re.IGNORECASE)


def _compile_failure(text: Optional[str], installed: dict) -> PlanError:
    """The PlanError for a failed ``uv pip compile`` whose output is *text*."""
    detail = _tail(text)
    if "--torch-backend" in (text or ""):
        return _unmatched_torch(installed.get("torch", "unknown"))
    if _NETWORK_FAILURE.search(text or ""):
        return PlanError(f"localm could not reach the package index. uv said: {detail}")
    return PlanError("localm found no way to add these packages as ready-built packages "
                     f"without changing a package it already has. uv said: {detail}")


def _tail(text: Optional[str], limit: int = 600) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else "..." + text[-limit:]


def _emit(cb: ProgressCb, line: str) -> None:
    if cb is None:
        return
    try:
        cb(line)
    except Exception:  # a broken progress sink never aborts an install
        from localm.debuglog import logger
        logger.debug("package install progress sink raised", exc_info=True)


_VERIFY = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
_INSTALL_LOCK = threading.Lock()


def _stream(cmd: list, on_progress: ProgressCb,
            timeout: float = _INSTALL_TIMEOUT_S) -> tuple[int, str]:
    """Run *cmd*, passing each output line to *on_progress*. Returns ``(exit
    code, output)``; a run still going after *timeout* seconds is killed and
    reported with a nonzero code and a closing line saying so."""
    kw = _run_kwargs()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", bufsize=1, **kw)
    expired = threading.Event()

    def kill() -> None:
        expired.set()
        proc.kill()

    timer = threading.Timer(timeout, kill)
    timer.daemon = True
    timer.start()
    lines = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            _emit(on_progress, line.rstrip())
        rc = proc.wait()
    finally:
        timer.cancel()
    if expired.is_set():
        lines.append(f"stopped after {int(timeout)} seconds without finishing\n")
        rc = rc or 1
    return rc, "".join(lines)


def _imports_fail(modules: list) -> Optional[str]:
    """None when every module in *modules* imports in a fresh interpreter that does not
    put the current folder on its path, else its error."""
    kw = _run_kwargs()
    try:
        proc = subprocess.run([sys.executable, "-P", "-c", _VERIFY, *modules], capture_output=True,
                              encoding="utf-8", errors="replace", timeout=_IMPORT_TIMEOUT_S,
                              **{k: v for k, v in kw.items() if k != "stdin"})
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    return None if proc.returncode == 0 else _tail(proc.stderr or proc.stdout)


def install(p: Plan, *, on_progress: ProgressCb = None) -> InstallResult:
    """Install exactly what plan *p* lists, then check every requested module,
    and torchvision or torchaudio when the plan adds them, imports. When one
    does not, or the install fails, the distributions this call added are
    removed again. One install runs at a time per process; a second is refused."""
    if not _INSTALL_LOCK.acquire(blocking=False):
        return InstallResult(ok=False, error="another package install is running")
    try:
        return _install(p, on_progress)
    finally:
        _INSTALL_LOCK.release()


def _install(p: Plan, on_progress: ProgressCb) -> InstallResult:
    present = installed_distributions()
    already = [n for n, _v in p.adds if _canon(n) in present]
    if already:
        return InstallResult(ok=False, error=(
            f"This install list is out of date: {', '.join(already)} "
            f"{'is' if len(already) == 1 else 'are'} installed now. Open the list again."))
    try:
        check_network(p.torch_backend)
        uv = _uv()
    except PlanError as e:
        return InstallResult(ok=False, error=str(e))
    names = [n for n, _v in p.adds]
    if names:
        _emit(on_progress, "Installing: " + ", ".join(f"{n} {v}" for n, v in p.adds))
        with tempfile.TemporaryDirectory(prefix="localm-packages-") as tmp:
            reqs = Path(tmp) / "plan.txt"
            reqs.write_text(p.requirements, encoding="utf-8")
            cmd = [uv, "pip", "install", "--no-deps", "--require-hashes",
                   *_index_args(p.torch_backend), "-r", str(reqs)]
            try:
                rc, out = _stream(cmd, on_progress)
            except OSError as e:
                return InstallResult(ok=False, error=f"uv could not be started: {e}"
                                     + _remove_added(uv, p, on_progress))
            except Exception as e:
                return InstallResult(ok=False, error=f"the install stopped unexpectedly: {e}"
                                     + _remove_added(uv, p, on_progress))
        if rc != 0:
            error = "uv could not install the packages: " + _tail(out)
            return InstallResult(ok=False, error=error + _remove_added(uv, p, on_progress))
    importlib.invalidate_caches()
    check = list(p.modules) + [m for m in ("torchvision", "torchaudio")
                               if m in {_canon(n) for n in names} and m not in p.modules]
    _emit(on_progress, "Checking that " + ", ".join(check) + " import...")
    failed = _imports_fail(check)
    if failed is not None:
        error = f"the packages were installed but do not import: {failed}"
        return InstallResult(ok=False, error=error + _remove_added(uv, p, on_progress))
    _record(p)
    _emit(on_progress, "Done.")
    return InstallResult(ok=True, installed=[f"{n}=={v}" for n, v in p.adds])


def _remove_added(uv: str, p: Plan, on_progress: ProgressCb) -> str:
    """Uninstall the distributions of plan *p* that are installed now (none
    were before the plan). Returns the clause the error message ends with."""
    present = installed_distributions()
    names = [n for n, _v in p.adds if _canon(n) in present]
    if not names:
        return ""
    try:
        rc, out = _stream([uv, "pip", "uninstall", "--no-config", "--python", sys.executable,
                           *names], on_progress)
    except OSError as e:
        return f"; removing the packages added failed too: {e}"
    return ("; the packages added were removed again" if rc == 0
            else "; removing the packages added failed too: " + _tail(out))


def _record(p: Plan) -> None:
    """Add the distributions *p* installed to the install record, when this
    install has one. A record that cannot be written is logged at WARNING; the
    packages stay installed and work."""
    if not p.adds:
        return
    from localm import install_manifest, uv_tool
    from localm.debuglog import logger
    day = time.strftime("%Y-%m-%d")
    entries = [{"project": n, "version": v, "for_modules": list(p.modules), "installed": day}
               for n, v in p.adds]
    try:
        install_manifest.record_model_packages(uv_tool.install_root(), entries)
    except (OSError, ValueError) as e:
        logger.warning("the install record could not list the packages added (%s): %s",
                       ", ".join(n for n, _v in p.adds), e)


_PLANS: dict = {}
_PLANS_LOCK = threading.Lock()


def remember(p: Plan) -> None:
    """Keep *p* so a later :func:`take` with its id returns it."""
    with _PLANS_LOCK:
        now = time.monotonic()
        for key in [k for k, v in _PLANS.items() if now - v.created > _PLAN_TTL_S]:
            del _PLANS[key]
        _PLANS[p.id] = p


def take(plan_id: str) -> Optional[Plan]:
    """The remembered plan *plan_id*, removed from the store, or None when it is
    unknown or older than the plan lifetime."""
    with _PLANS_LOCK:
        p = _PLANS.pop(plan_id, None) if isinstance(plan_id, str) else None
    if p is None or time.monotonic() - p.created > _PLAN_TTL_S:
        return None
    return p


def model_folder(name: str) -> Optional[Path]:
    """The Hugging Face folder of the registered model *name*, of a registered
    model whose folder is called *name* (when exactly one is), or the folder at
    the path *name*; None when none of these is a folder."""
    from localm.config import load_registry
    from localm.model_manager.registry import _entry_path, get_model_path
    path = get_model_path(name)
    if path is not None:
        return path if path.is_dir() else None
    matches = []
    for entry in load_registry().values():
        ep = _entry_path(entry)
        if ep and Path(ep).name == name and Path(ep).is_dir():
            matches.append(Path(ep))
    if len(set(matches)) == 1:
        return matches[0]
    candidate = Path(name).expanduser()
    return candidate if candidate.is_dir() else None


@dataclass
class FolderState:
    """What the bundled code of a model folder needs that this install lacks.
    ``trusted`` is False when ``hf_trust_remote_code`` is off (the code is not
    run, so nothing is checked)."""
    trusted: bool
    missing: list
    shipped: list
    blocked: list
    addable: list

    def as_dict(self) -> dict:
        return {"missing": [package_label(m) for m in self.missing],
                "addable": [package_label(m) for m in self.addable],
                "installable": bool(self.addable) and not self.shipped and not self.blocked}


def folder_state(folder: Path) -> FolderState:
    """The packages the bundled code of the model *folder* needs and this
    install lacks, sorted into ones localm ships here (the install is incomplete),
    ones it cannot add here (including a shipped module this platform does not
    get), and ones it can add. Never touches the network."""
    from localm.inference.backends.hf import (
        _declares_custom_code,
        _trust_remote_code_enabled,
        uses_transformers_classes,
    )
    from localm.model_manager.missing_packages import is_importable, shipped_here
    if not folder.is_dir() or not _declares_custom_code(str(folder)):
        return FolderState(True, [], [], [], [])
    if not _trust_remote_code_enabled():
        return FolderState(False, [], [], [], [])
    if uses_transformers_classes(str(folder)):
        return FolderState(True, [], [], [], [])
    missing = sorted(m for m in _folder_imports(folder) if not is_importable(m))
    shipped = [m for m in missing if shipped_here(m)]
    others = [m for m in missing if not shipped_here(m)]
    blocked = [m for m in others if not installable(m)]
    addable = [m for m in others if installable(m)]
    return FolderState(True, missing, shipped, blocked, addable)


_scan_cache: dict = {}
_scan_lock = threading.Lock()


def _scan_key(folder: Path) -> tuple:
    from localm.model_manager.missing_packages import CONFIG_FILES
    parts = []
    for f in sorted(folder.glob("*.py")) + [folder / n for n in CONFIG_FILES]:
        try:
            st = f.stat()
            parts.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            parts.append((f.name, None, None))
    return tuple(parts)


def _folder_imports(folder: Path) -> frozenset:
    """The third-party modules the bundled code of *folder* imports, scanned
    again only when one of its ``.py`` or config files changed. Empty when the
    folder cannot be read."""
    from localm.model_manager.missing_packages import folder_third_party_imports
    try:
        key = _scan_key(folder)
    except OSError:
        return frozenset()
    name = str(folder)
    with _scan_lock:
        hit = _scan_cache.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        found = frozenset(folder_third_party_imports(folder))
    except (OSError, ValueError, RecursionError):
        return frozenset()
    with _scan_lock:
        if len(_scan_cache) >= _SCAN_CACHE_MAX:
            _scan_cache.clear()
        _scan_cache[name] = (key, found)
    return found


def recorded() -> list:
    """The packages the install record lists as added for models
    (``{"project", "version", "for_modules", "installed"}``), oldest first;
    empty without an install record."""
    from localm import install_manifest, uv_tool
    data = install_manifest.load(uv_tool.install_root()) or {}
    return [e for e in data.get("model_packages") or [] if isinstance(e, dict)]
