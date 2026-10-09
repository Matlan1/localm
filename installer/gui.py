#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LocaLM graphical installer - the windowed alternative to setup.bat/setup.sh.

Run by `setup-gui.bat` (Windows) or `setup-gui.sh` via:

    uv run --no-project --python 3.12 python installer/gui.py

SEPARATE FROM THE INSTALL IT CREATES. This script is not part of the `localm`
package and never imports the installed distribution: it runs on a managed
CPython that uv provides, before `.venv` exists, and its only import from this
source tree is `localm.hwdetect`, which is stdlib-only by design so GPU
detection can happen BEFORE anything is installed. That is what lets the whole
install be described on one page up front, the way an installer should be,
rather than interrogating the user between steps.

DEPENDENCIES: none. tkinter ships with the managed CPython uv installs, so the
window costs nothing to provision. If tkinter is genuinely unavailable the
launcher scripts fall back to the console installer and say why.

WHAT IT DOES is exactly what setup.bat's prompts decide, in setup.bat's own
order and with its own commands, so the two installers cannot drift: create
the venv, install localm and the native-runtime wheel, install the PyTorch
stack that matches the chosen backend, provision llama.cpp, record where data
lives, build the launcher, optionally create a desktop shortcut, and
optionally put `localm` on PATH.

When setup has run in the folder before, it first offers to repair the
install or uninstall it. Uninstall runs localm.install_manifest in the window
and leaves the Python runtime the window runs on to setup-gui.bat /
setup-gui.sh, which remove it after the window closes (exit code 42).
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
_SRC = Path(__file__).resolve().parents[1]
APP_NAME = "LocaLM"
PYVER = "3.12"

# Exit codes telling setup-gui.bat / setup-gui.sh how an uninstall in the window
# ended. 42 and 43: the Python runtime folders named in
# .localm-uninstall-pending are theirs to remove now that the window has closed;
# 43 and 44: something the user asked to delete was kept; 45: the uninstall did
# not finish.
EXIT_FINISH_UNINSTALL = 42
EXIT_FINISH_UNINSTALL_PARTIAL = 43
EXIT_UNINSTALL_PARTIAL = 44
EXIT_UNINSTALL_FAILED = 45

# The extras setup.bat installs. `desktop` is added only when the user asks for
# an app window, because it pulls pythonnet in and no install should take on a
# dependency nobody asked for.
BASE_EXTRAS = "coder,voice,monitor"

# Mirrors setup.bat's backend menu. The recommendation is computed at runtime
# and shown as the default; "own" skips the download for people who build
# llama.cpp themselves.
_BACKEND_CHOICES = [
    ("vulkan", "Vulkan - any GPU (AMD/NVIDIA/Intel), no vendor toolkit"),
    ("cuda", "CUDA - NVIDIA, peak performance"),
    ("hip", "ROCm/HIP - AMD, peak performance (needs the ROCm runtime)"),
    ("amd-rocm", "ROCm - AMD RX 6000 (gfx103X), self-contained"),
    ("sycl", "SYCL - Intel GPU (incl. integrated), often faster than Vulkan"),
    ("metal", "Metal - Apple Silicon, native GPU acceleration"),
    ("cpu", "CPU only - no GPU"),
    ("own", "I will provide my own llama.cpp build (skip the download)"),
]


def backend_choices() -> List[tuple]:
    """The runtime menu for this platform. metal exists only on macOS and
    amd-rocm only on Windows, matching the console installer's menu."""
    out = []
    for key, desc in _BACKEND_CHOICES:
        if key == "metal" and sys.platform != "darwin":
            continue
        if key == "amd-rocm" and not IS_WINDOWS:
            continue
        out.append((key, desc))
    return out


# The plugins the console installer preselects.
RECOMMENDED_PLUGINS = ("coder", "rag", "web", "tts")


def plugin_choices() -> List[tuple]:
    """(name, description) for every plugin a user can choose, read from the
    package's own catalog so this menu cannot drift from it. Returns an empty
    list when the catalog cannot be read, and the Features page says so."""
    try:
        sys.path.insert(0, str(ROOT))
        from localm.plugins import catalog
        return [(n, catalog.get(n).description)
                for n in catalog.names() if n not in catalog.preinstalled()]
    except Exception:
        return []

IS_WINDOWS = sys.platform == "win32"


def venv_bin(root: Path) -> Path:
    return root / ".venv" / ("Scripts" if IS_WINDOWS else "bin")


def venv_python(root: Path) -> Path:
    return venv_bin(root) / ("python.exe" if IS_WINDOWS else "python")


def uv_dirs(root: Path) -> List[Path]:
    """Every directory uv may live in, most preferred first: the portable copy
    the launcher puts inside the clone, then Astral's own default install
    locations, which a shell started before the installer ran does not
    necessarily have on PATH yet."""
    home = Path.home()
    return [root / ".uv", root / ".uv" / "bin",
            home / ".local" / "bin", home / ".cargo" / "bin"]


def find_uv(root: Path) -> Optional[str]:
    """The uv to run, or None if there is none. Returns a full path for a
    portable uv and a bare name for one resolved on PATH.

    The launcher scripts put the uv they used on PATH for this process, so a
    uv that is not in one of the directories above is still reachable here.

    Every uv invocation and the entry check below both go through this, so a
    uv that starts the installer is always a uv the steps can run. See
    tests/test_installer_gui.py TestUvResolution."""
    exe = "uv.exe" if IS_WINDOWS else "uv"
    for d in uv_dirs(root):
        candidate = d / exe
        if candidate.is_file():
            return str(candidate)
    return shutil.which("uv")


def install_manifest():
    """localm.install_manifest from this source tree (stdlib only, so it
    imports on the window's own interpreter)."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from localm import install_manifest as im
    return im


def existing_install(root: Path) -> bool:
    """Whether setup has run in *root* before: an install record, or an
    environment carrying setup's marker."""
    return ((root / ".localm-install.json").is_file()
            or (root / ".venv" / ".localm-venv").is_file())


def _tooling_fields(root: Path) -> dict:
    """install_manifest.record() keywords for the tooling folders inside
    *root*; empty when there are none."""
    fields = {}
    for key, name in (("python_dir", ".python"), ("cache_dir", ".cache"),
                      ("uv_dir", ".uv")):
        if (root / name).exists():
            fields[key] = str(root / name)
    if fields:
        fields["runtime_contained"] = True
    return fields


def detect_recommendation() -> tuple:
    """(vendor, recommended_backend) from localm.hwdetect, imported straight out
    of this source tree. Never raises: an undetectable machine offers the same
    menu with vulkan preselected, which is the universal fallback."""
    try:
        sys.path.insert(0, str(ROOT))
        from localm import hwdetect
        det = hwdetect.detect()
        vendor = det.vendors[0] if det.vendors else None
        return vendor, hwdetect.recommended_install_backend(det)
    except Exception:
        return None, "vulkan"


def torch_spec_for(backend: str) -> tuple:
    """The PyTorch install arguments for *backend*, from the SAME policy
    setup.bat consults (`python -m localm.hwdetect torch-args`).

    Returns (spec, problem). A spec of None with an empty problem means this
    machine needs no torch stack. A non-empty problem means the policy could
    not be asked, which is never reported as "no torch needed"."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "localm.hwdetect", "torch-args", backend],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except Exception as e:
        return None, f"could not ask which PyTorch build this machine needs: {e}"
    if out.returncode != 0:
        detail = (out.stderr or "").strip().splitlines()
        return None, ("could not ask which PyTorch build this machine needs "
                      f"(exited {out.returncode}"
                      + (f": {detail[-1]}" if detail else "") + ")")
    return ((out.stdout or "").strip() or None), ""


@dataclass
class Plan:
    """Every decision the installer needs, all collected before any work runs."""
    backend: str = "vulkan"
    app_window: bool = False          # installs the `desktop` extra
    portable_data: bool = True        # ./home  vs  a custom directory
    data_path: str = ""
    shortcut: str = "launcher"        # launcher | gui | none
    add_to_path: bool = False
    portable_store: bool = True       # keep uv's python + cache inside the clone
    plugins: tuple = ()               # optional features to install
    plugin_deps: bool = True          # install the pip extras those need

    @property
    def extras(self) -> str:
        return BASE_EXTRAS + (",desktop" if self.app_window else "")


class StepFailed(Exception):
    """A step that must not be reported as success."""


def uv_argv(*args: str) -> List[str]:
    """A uv command line, resolved when the step runs. Raises StepFailed when
    uv cannot be found, so a missing uv is reported as the step it broke."""
    exe = find_uv(ROOT)
    if exe is None:
        raise StepFailed(
            "uv was not found in this folder or on PATH. Close this window "
            "and run setup.bat / setup.sh instead.")
    return [exe, *args]


@dataclass
class Step:
    label: str
    run: Callable[[Callable[[str], None]], None]
    fatal: bool = True
    key: str = ""             # the name this step goes by in the setup journal


def journal(emit: Callable[[str], None], event: str, name: str = "", path: str = "") -> None:
    """Append *event* to the setup journal, saying so once if it cannot be written."""
    try:
        install_manifest().journal_event(ROOT, event, name, path)
    except Exception as e:
        _warn_journal(emit, e)


def _warn_journal(emit: Callable[[str], None], error: Exception) -> None:
    if not getattr(journal, "warned", False):
        journal.warned = True
        emit(f"[!] Could not write the setup journal ({error}); if this setup is "
             "interrupted it cannot say where it stopped.")


def begin_journal(emit: Callable[[str], None]) -> dict:
    """Start (or pick up) the journal for this run and return the state it found.

    A journal that ends in ``complete`` belongs to a finished setup and is replaced.
    One that does not is the trace of an interrupted setup: it is reported, a
    ``resume`` is recorded, and the returned state lists the steps it left open."""
    journal.warned = False
    try:
        im = install_manifest()
        state = im.journal_state(ROOT)
    except Exception as e:
        _warn_journal(emit, e)
        return {"exists": False, "complete": False, "done": [], "started": []}
    if state["exists"] and state["complete"]:
        try:
            im.journal_reset(ROOT)
        except OSError as e:
            _warn_journal(emit, e)
            return {"exists": False, "complete": False, "done": [], "started": []}
        return im.journal_state(ROOT)
    if state["exists"]:
        emit(f"A previous setup in this folder was interrupted: {im.describe_journal(state)}.")
        emit("Picking it up: every step is run again, and an interrupted download is redone.")
        journal(emit, "resume")
    return state


def _env_for(plan: Plan) -> dict:
    """uv's environment. Portable keeps the managed interpreter AND the wheel
    cache inside the clone, so nothing is written to the user profile - the
    same containment setup.bat's Portable option gives."""
    env = dict(os.environ)
    env["LOCALM_SETUP"] = "1"
    env["UV_SYSTEM_CERTS"] = "1"
    # Put the uv being used on PATH for everything the steps run. localm's own
    # plugin dependency installer shells out to a bare uv, and a portable copy
    # inside the folder is not on PATH; its fallback cannot help either,
    # because a uv-created environment has no pip.
    exe = find_uv(ROOT)
    uv_bin = os.path.dirname(exe) if exe else ""
    if uv_bin:
        env["PATH"] = uv_bin + os.pathsep + env.get("PATH", "")
    ours = {"UV_PYTHON_INSTALL_DIR": str(ROOT / ".python"),
            "UV_CACHE_DIR": str(ROOT / ".cache")}
    for key, value in ours.items():
        if plan.portable_store:
            env[key] = value
        elif env.get(key) == value:
            # Inherited from the launcher, not chosen by the user.
            env.pop(key)
    return env


def _run(cmd: List[str], emit: Callable[[str], None], plan: Plan,
         *, allow_fail: bool = False) -> int:
    """Run a command, streaming its output into the log a line at a time.

    Returns the exit code. Raises StepFailed on a non-zero exit unless
    *allow_fail*, so a step can never be silently skipped and still reported as
    done."""
    emit("$ " + " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(ROOT), env=_env_for(plan),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
    except OSError as e:
        if allow_fail:
            emit(f"[!] could not start: {e}")
            return 1
        raise StepFailed(f"could not start {cmd[0]}: {e}")
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(line.rstrip())
    code = proc.wait()
    if code != 0 and not allow_fail:
        raise StepFailed(f"{cmd[0]} exited {code}")
    if code != 0:
        emit(f"[!] {cmd[0]} exited {code} - continuing (this step is optional)")
    return code


def _query(args: List[str]) -> str:
    """The first line the installed localm prints for *args*, or empty."""
    try:
        out = subprocess.run([str(venv_python(ROOT)), *args], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=60)
        lines = (out.stdout or "").strip().splitlines()
        return lines[0] if lines else ""
    except Exception:
        return ""


def _record_now(emit: Callable[[str], None], **fields) -> None:
    """Record what a step just created in the install record; the manifest step
    at the end records everything again."""
    try:
        install_manifest().record(ROOT, **fields)
    except Exception as e:
        emit(f"[!] could not record this step yet: {e}")


def run_steps(steps: List[Step], emit: Callable[[str], None],
              on_step: Callable[[int, int, str], None] = lambda i, n, label: None,
              ) -> tuple:
    """Run *steps* in order, journaling each by its key.

    Returns ``(failures, fatal)``: the messages of the optional steps that failed,
    and the message of the required step that stopped the run (None when it ran to
    the end). A step that fails stays open in the journal; ``complete`` is written
    only when the run reaches the end."""
    failures: List[str] = []
    for i, step in enumerate(steps):
        on_step(i, len(steps), step.label)
        if step.key:
            journal(emit, "begin", step.key)
        try:
            step.run(emit)
        except Exception as e:                # never leave the UI hanging
            if step.fatal:
                return failures, f"{step.label}: {e}"
            failures.append(f"{step.label}: {e}")
            emit(f"[!] {step.label} did not finish: {e}")
        else:
            if step.key:
                journal(emit, "done", step.key)
    journal(emit, "complete")
    return failures, None


def build_steps(plan: Plan, resume: Optional[dict] = None) -> List[Step]:
    """The install, as setup.bat performs it, in setup.bat's order. *resume* is the
    journal state of an interrupted earlier run (see begin_journal)."""
    steps: List[Step] = []
    cut_short = set((resume or {}).get("started") or [])
    # What earlier steps created, so the manifest records exactly that.
    state: dict = {}

    def venv(emit):
        args = ["venv", "--python", PYVER]
        if plan.portable_store:
            args += ["--python-preference", "only-managed"]
        args += ["--clear", ".venv"]
        _run(uv_argv(*args), emit, plan)
        # uninstall removes .venv only when this marker says setup created it.
        try:
            (ROOT / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
        except OSError as e:
            raise StepFailed(f"the environment was not created where it was "
                             f"expected: {e}")
        # Recorded now; the manifest step at the end records again.
        try:
            install_manifest().record(ROOT, venv=str(ROOT / ".venv"),
                                      **_tooling_fields(ROOT))
        except Exception as e:
            emit(f"[!] could not record the environment yet: {e}")
    steps.append(Step("Creating the Python environment", venv, key="venv"))

    # Runs before anything writes data: setup-llama records its builds in this
    # folder. See test_data_folder_is_chosen_before_the_runtime_is_provisioned.
    def data_dir(emit):
        # prepare_data creates the folder before writing localm-home.cfg, and
        # records it for uninstall with what was already in it.
        try:
            if plan.portable_data:
                target = install_manifest().prepare_data(ROOT, portable=True)
            else:
                target = install_manifest().prepare_data(ROOT, data_dir=plan.data_path)
        except (OSError, ValueError) as e:
            raise StepFailed(f"could not use that data folder: {e}")
        state["data_dir"] = str(target)
        emit(f"Data directory: {target}" + (" (portable)" if plan.portable_data else ""))
    steps.append(Step("Recording where data lives", data_dir, key="data-folder"))

    def install_localm(emit):
        _run(uv_argv("pip", "install", "-p", ".venv", "-e", f".[{plan.extras}]"),
             emit, plan)
    steps.append(Step("Installing LocaLM", install_localm, key="install-localm"))

    def install_runtime_pkg(emit):
        # Carries llama.dll + ggml inside the venv; setup-llama fills it below.
        _run(uv_argv("pip", "install", "-p", ".venv", "-e", "./runtime"), emit, plan)
    steps.append(Step("Installing the native runtime package", install_runtime_pkg,
                      key="runtime-package"))

    def install_torch(emit):
        spec, problem = torch_spec_for(plan.backend)
        if problem:
            raise StepFailed(problem)
        if not spec:
            emit("No PyTorch stack needed for this backend (GGUF chat does not use it).")
            return
        if spec == "-e .[gpu]":
            codes = [_run(uv_argv("pip", "install", "-p", ".venv", "-e", ".[gpu,audio]"),
                          emit, plan, allow_fail=True)]
        else:
            codes = [
                _run(uv_argv("pip", "install", "-p", ".venv", *spec.split()),
                     emit, plan, allow_fail=True),
                _run(uv_argv("pip", "install", "-p", ".venv", "-e", ".[hf,audio]"),
                     emit, plan, allow_fail=True),
            ]
        if any(codes):
            raise StepFailed("the PyTorch stack did not install; GGUF chat still "
                             "works, models that need PyTorch will not")
    # Not fatal: a failed torch stack still leaves a working GGUF chat install,
    # which is what setup.bat also says at this point.
    steps.append(Step("Installing PyTorch and transformers", install_torch, fatal=False,
                      key="torch"))

    if plan.backend != "own":
        def provision(emit):
            force = ["--force"] if "native-runtime" in cut_short else []
            _run([str(venv_bin(ROOT) / "localm"), "setup-llama",
                  "--backend", plan.backend, "--yes", *force], emit, plan)
        steps.append(Step(f"Provisioning the {plan.backend} inference runtime", provision,
                          key="native-runtime"))

    def launcher(emit):
        _run([str(venv_python(ROOT)), "-m", "localm", "make-launcher",
              "--force", "--quiet"], emit, plan, allow_fail=True)
    steps.append(Step("Building the launcher", launcher, fatal=False, key="launcher"))

    if plan.shortcut != "none":
        def shortcut(emit):
            intended = intended_shortcut_path()
            if intended:
                journal(emit, "intend", "shortcut", intended)
            state["shortcut"] = make_shortcut(plan, emit) or ""
            if state["shortcut"]:
                _record_now(emit, shortcut=state["shortcut"])
        steps.append(Step("Creating the desktop shortcut", shortcut, fatal=False,
                          key="menu-entry"))

    if plan.add_to_path:
        def global_cmd(emit):
            journal(emit, "intend", "command", str(intended_command_path()))
            # --yes: a conflict prompt has no console to answer it here.
            code = _run([str(venv_python(ROOT)), "-m", "localm.globalcmd",
                         "install", "--root", ".", "--yes"], emit, plan,
                        allow_fail=True)
            # 20 = the command was created and its directory was already on
            # PATH. Only 0 also changed PATH.
            if code not in (0, 20):
                raise StepFailed("the global localm command was not added")
            state["path_dir"] = _query(["-m", "localm.globalcmd",
                                        "path-dir", "--root", "."])
            state["command_shim"] = _query(["-m", "localm.globalcmd",
                                            "shim", "--root", "."])
            state["path_modified"] = code == 0
            _record_now(emit, path_dir=state["path_dir"], command_shim=state["command_shim"],
                        path_modified=state["path_modified"])
        steps.append(Step("Adding 'localm' to your PATH", global_cmd,
                          fatal=False, key="global-command"))

    if plan.plugins:
        def plugins(emit):
            # An explicit deps flag: the default asks, and nothing can answer.
            _run([str(venv_bin(ROOT) / "localm"), "plugin", "setup",
                  "--plugins", ",".join(plan.plugins),
                  "--with-deps" if plan.plugin_deps else "--no-deps"],
                 emit, plan)
        steps.append(Step("Installing the optional features you chose",
                          plugins, fatal=False, key="plugins"))

    def manifest(emit):
        # The data folder was recorded by the data step (prepare_data).
        args = [str(venv_python(ROOT)), "-m", "localm.install_manifest",
                "record", "--root", ".",
                "--venv", str(ROOT / ".venv"),
                "--lib-dir", str(ROOT / "runtime" / "localm_llama_runtime" / "lib"),
                "--shortcut", state.get("shortcut", ""),
                "--path-dir", state.get("path_dir", ""),
                "--command-shim", state.get("command_shim", ""),
                "--stamp",
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")]
        if (ROOT / "LocaLM.desktop").is_file():
            args += ["--file", str(ROOT / "LocaLM.desktop")]
        if state.get("path_modified"):
            args.append("--path-modified")
        # These sit inside the folder whichever way the tooling question was
        # answered: the launcher puts the window's own interpreter there.
        contained = [("--python-dir", ROOT / ".python"),
                     ("--cache-dir", ROOT / ".cache"),
                     ("--uv-dir", ROOT / ".uv")]
        present = [(flag, d) for flag, d in contained if d.exists()]
        if present:
            args.append("--runtime-contained")
            for flag, d in present:
                args += [flag, str(d)]
        _run(args, emit, plan)
        emit("Recorded what this install created, so uninstall removes only that.")
    steps.append(Step("Writing the install record", manifest, fatal=False, key="record"))

    return steps


def intended_shortcut_path() -> str:
    """Where make_shortcut will write the shortcut, or empty when that cannot be
    worked out. Journaled before the file exists so an interruption between
    creating and recording it still lets uninstall remove it."""
    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetFolderPath('Desktop')"],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return ""
        lines = (out.stdout or "").strip().splitlines()
        return str(Path(lines[-1]) / "LocaLM.lnk") if lines and lines[-1].strip() else ""
    return str(Path.home() / ".local/share/applications" / "LocaLM.desktop")


def intended_command_path() -> Path:
    """The file the global command step creates for this folder."""
    if IS_WINDOWS:
        return ROOT / "bin" / "localm.cmd"
    return Path.home() / ".local" / "bin" / "localm"


def make_shortcut(plan: Plan, emit: Callable[[str], None]) -> str:
    """A desktop shortcut, by the same means setup.bat uses on each platform:
    a WScript.Shell .lnk on Windows, a freedesktop .desktop file elsewhere.

    Returns the path written, which the install manifest records so uninstall
    removes this shortcut and no other."""
    if IS_WINDOWS:
        exe = ROOT / ".venv" / "localm-app" / "LocaLM.exe"
        if plan.shortcut == "gui" and exe.exists():
            target, args = str(exe), "-m localm gui"
        elif plan.shortcut == "gui":
            target, args = str(venv_bin(ROOT) / "localm.exe"), "gui"
        else:
            target, args = str(ROOT / "localm-launcher.bat"), ""
        ico = ROOT / "assets" / "localm.ico"
        # A single quote inside a PowerShell single-quoted string is doubled.
        # Paths under a name like O'Brien reach here.
        def q(value) -> str:
            return str(value).replace("'", "''")
        ps = (
            "$p = [Environment]::GetFolderPath('Desktop') + '\\LocaLM.lnk';"
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($p);"
            f"$s.TargetPath = '{q(target)}';"
            + (f"$s.Arguments = '{q(args)}';" if args else "")
            + f"$s.WorkingDirectory = '{q(ROOT)}';"
            + (f"$s.IconLocation = '{q(ico)}';" if ico.exists() else "")
            + "$s.Description = 'LocaLM';$s.Save();Write-Output $p"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             check=True, capture_output=True, text=True)
        written = (out.stdout or "").strip().splitlines()
        emit("Shortcut created on your Desktop.")
        return written[-1] if written else ""

    if plan.shortcut == "gui":
        exec_line = f"{venv_bin(ROOT) / 'localm'} gui"
        comment = "Local AI, offline"
    else:
        exec_line = str(ROOT / "localm-launcher.sh")
        comment = "LocaLM launcher: GUI, chat, server or coder"
    svg = ROOT / "assets" / "localm.svg"
    ico = ROOT / "assets" / "localm.ico"
    icon = svg if svg.exists() else (ico if ico.exists() else None)
    text = ("[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            f"Comment={comment}\n"
            f"Exec={exec_line}\n"
            + (f"Icon={icon}\n" if icon else "")
            + f"Path={ROOT}\n"
            "Terminal=false\n"
            "Categories=Utility;Development;Science;\n")
    d = Path.home() / ".local/share/applications"
    try:
        d.mkdir(parents=True, exist_ok=True)
        f = d / "LocaLM.desktop"
        f.write_text(text, encoding="utf-8")
        f.chmod(0o755)
    except OSError as e:
        raise StepFailed(f"no desktop entry could be written: {e}")
    emit(f"Wrote {f}")
    return str(f)


# --------------------------------------------------------------------------- #
#  The window                                                                  #
# --------------------------------------------------------------------------- #

def _is_custom_folder_text(text: str) -> bool:
    """Whether text typed into the data-folder box means the custom option."""
    return bool(text.strip())


# The environment variable that forces the window's theme: "dark" or "light".
THEME_ENV = "LOCALM_THEME"

# The window's colours for each theme.
PALETTES = {
    "dark": {
        "bg": "#0e1014",
        "surface": "#171c26",
        "field": "#1e242f",
        "border": "#2c3341",
        "outline": "#5c6473",
        "text": "#dce2ec",
        "dim": "#98a2b4",
        "accent": "#5aa2fb",
        "on_accent": "#0e1014",
        "warn": "#e25d5d",
    },
    "light": {
        "bg": "#f5f6f8",
        "surface": "#ffffff",
        "field": "#eef0f4",
        "border": "#d8dce4",
        "outline": "#858c99",
        "text": "#23272f",
        "dim": "#5f6675",
        "accent": "#2563eb",
        "on_accent": "#ffffff",
        "warn": "#c2410c",
    },
}

_PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19


def _apps_use_light_theme():
    """The current user's AppsUseLightTheme setting (0 means dark apps).
    Raises OSError when Windows has no such setting."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE_KEY) as key:
        return winreg.QueryValueEx(key, "AppsUseLightTheme")[0]


def _command_output(argv: List[str], timeout: float = 2.0) -> Optional[str]:
    """stdout of argv, or None when it cannot start, exits non-zero or runs
    longer than timeout seconds."""
    try:
        proc = subprocess.run(argv, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def detect_theme(env=None, platform: Optional[str] = None,
                 read_windows: Optional[Callable[[], object]] = None,
                 run: Optional[Callable[[List[str]], Optional[str]]] = None) -> str:
    """Return "dark" or "light" for the setup window.

    LOCALM_THEME wins when it is "dark" or "light" (any case). Otherwise:
    Windows reads AppsUseLightTheme (0 is dark), macOS reads
    AppleInterfaceStyle ("Dark" is dark), and every other system reads GNOME's
    color-scheme ("prefer-dark" or "prefer-light"), then whether the GTK theme
    name contains "dark". A setting that is missing or cannot be read means
    light.

    env replaces os.environ, platform replaces sys.platform, read_windows
    replaces the registry read and run replaces the command runner, which
    returns a command's stdout or None."""
    env = os.environ if env is None else env
    forced = str(env.get(THEME_ENV) or "").strip().lower()
    if forced in PALETTES:
        return forced
    platform = sys.platform if platform is None else platform
    run = _command_output if run is None else run
    # Any failure leaves the light theme. See
    # test_a_reader_that_fails_means_light.
    try:
        if platform == "win32":
            read = _apps_use_light_theme if read_windows is None else read_windows
            return "dark" if read() == 0 else "light"
        if platform == "darwin":
            style = run(["defaults", "read", "-g", "AppleInterfaceStyle"]) or ""
            return "dark" if style.strip().lower() == "dark" else "light"
        scheme = run(["gsettings", "get", "org.gnome.desktop.interface",
                      "color-scheme"]) or ""
        scheme = scheme.strip().strip("'\"").lower()
        if scheme in ("prefer-dark", "prefer-light"):
            return scheme[len("prefer-"):]
        gtk = run(["gsettings", "get", "org.gnome.desktop.interface",
                   "gtk-theme"]) or ""
        return "dark" if "dark" in gtk.lower() else "light"
    except Exception:
        return "light"


def apply_theme(root, ttk, palette: dict) -> None:
    """Switch ttk to the "clam" theme and colour the root window and every ttk
    widget class the wizard uses from palette. Adds the Dim.TLabel and
    Warn.TLabel label styles."""
    p = palette
    root.configure(background=p["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=p["bg"], foreground=p["text"],
                    bordercolor=p["border"], lightcolor=p["bg"],
                    darkcolor=p["bg"], troughcolor=p["field"],
                    fieldbackground=p["field"], insertcolor=p["text"],
                    selectbackground=p["accent"],
                    selectforeground=p["on_accent"],
                    focuscolor=p["accent"], arrowcolor=p["dim"])
    style.map(".",
              background=[("disabled", p["bg"]), ("active", p["bg"])],
              foreground=[("disabled", p["outline"])],
              selectbackground=[("!focus", p["border"])],
              selectforeground=[("!focus", p["text"])])

    style.configure("Dim.TLabel", foreground=p["dim"])
    style.configure("Warn.TLabel", foreground=p["warn"])

    style.configure("TButton", background=p["surface"], foreground=p["text"],
                    bordercolor=p["outline"], lightcolor=p["surface"],
                    darkcolor=p["surface"])
    style.map("TButton",
              background=[("disabled", p["bg"]), ("pressed", p["border"]),
                          ("active", p["field"])],
              lightcolor=[("disabled", p["bg"]), ("pressed", p["border"]),
                          ("active", p["field"])],
              darkcolor=[("disabled", p["bg"]), ("pressed", p["border"]),
                         ("active", p["field"])],
              bordercolor=[("disabled", p["border"]), ("focus", p["accent"])],
              foreground=[("disabled", p["outline"])])

    for cls in ("TCheckbutton", "TRadiobutton"):
        style.configure(cls, background=p["bg"], foreground=p["text"],
                        indicatorbackground=p["field"],
                        indicatorforeground=p["accent"],
                        upperbordercolor=p["outline"],
                        lowerbordercolor=p["outline"])
        style.map(cls,
                  background=[("active", p["bg"])],
                  foreground=[("disabled", p["outline"])],
                  indicatorbackground=[("pressed", p["surface"]),
                                       ("disabled", p["bg"]),
                                       ("alternate", p["accent"])])

    style.configure("TEntry", fieldbackground=p["field"], foreground=p["text"],
                    insertcolor=p["text"], bordercolor=p["outline"],
                    lightcolor=p["field"], darkcolor=p["field"])
    style.map("TEntry",
              background=[("readonly", p["bg"])],
              fieldbackground=[("disabled", p["bg"]), ("readonly", p["bg"])],
              foreground=[("disabled", p["outline"])],
              bordercolor=[("focus", p["accent"])],
              lightcolor=[("focus", p["accent"])],
              darkcolor=[("focus", p["accent"])])

    style.configure("TProgressbar", background=p["accent"],
                    troughcolor=p["field"], bordercolor=p["border"],
                    lightcolor=p["accent"], darkcolor=p["accent"])

    style.configure("TScrollbar", background=p["surface"],
                    troughcolor=p["bg"], bordercolor=p["border"],
                    lightcolor=p["surface"], darkcolor=p["surface"],
                    arrowcolor=p["dim"], gripcount=0)
    style.map("TScrollbar",
              background=[("disabled", p["bg"]), ("active", p["field"])],
              lightcolor=[("disabled", p["bg"]), ("active", p["field"])],
              darkcolor=[("disabled", p["bg"]), ("active", p["field"])],
              arrowcolor=[("disabled", p["outline"])])


def text_options(palette: dict) -> dict:
    """Colour and border options for a plain tk.Text widget in palette."""
    p = palette
    return {
        "background": p["surface"],
        "foreground": p["text"],
        "insertbackground": p["text"],
        "selectbackground": p["accent"],
        "selectforeground": p["on_accent"],
        "inactiveselectbackground": p["border"],
        "relief": "flat",
        "borderwidth": 0,
        "highlightthickness": 1,
        "highlightbackground": p["border"],
        "highlightcolor": p["accent"],
        "padx": 4,
        "pady": 4,
    }


def set_title_bar_theme(root, dark: bool, windll=None) -> bool:
    """On Windows, ask the window manager to draw root's title bar dark (dark
    is True) or light. Returns True when it accepted the setting and False on
    any other system or when any call fails. Never raises.

    windll replaces ctypes.windll."""
    # A failure leaves the system's default title bar. See
    # test_a_title_bar_that_cannot_be_set_is_reported_not_raised.
    try:
        import ctypes
        if windll is None:
            if not IS_WINDOWS:
                return False
            windll = ctypes.windll
        root.update_idletasks()
        hwnd = windll.user32.GetParent(root.winfo_id())
        if not hwnd:
            return False
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (DWMWA_USE_IMMERSIVE_DARK_MODE,
                          DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1):
            result = windll.dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd), attribute, ctypes.byref(value),
                ctypes.sizeof(value))
            if result == 0:
                return True
    except Exception:
        return False
    return False


class Wizard:
    """The setup dialogue: one page per group of questions, then the install.

    Every page is built up front and shown one at a time, so Back never has to
    rebuild anything and an answer survives moving away from its page.

    next_page/prev_page and current_plan are the whole navigation surface, so a
    test can drive the dialogue without a person clicking. See
    tests/test_installer_gui.py TestWizard.

    When setup has run in this folder before, the dialogue opens on a choice
    between repairing the install and uninstalling it. The uninstall page
    shows exactly what will be removed, with an option to delete the saved
    data too; exit_code is EXIT_FINISH_UNINSTALL (or
    EXIT_FINISH_UNINSTALL_PARTIAL) when the runtime folders are left for the
    launcher script to remove after the window closes, EXIT_UNINSTALL_PARTIAL
    when something asked for was kept and nothing is left to remove, and
    EXIT_UNINSTALL_FAILED when the uninstall did not finish. While an
    uninstall runs, closing the window is refused.

    theme is "dark" or "light"; None follows detect_theme()."""

    def __init__(self, root, tk, ttk, filedialog, *, existing=None,
                 messagebox=None, theme: Optional[str] = None):
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.theme = detect_theme() if theme is None else theme
        self.palette = PALETTES[self.theme]
        self.vendor, self.recommended = detect_recommendation()
        self.plugin_rows = plugin_choices()
        self.index = 0
        self.installing = False
        self.existing = existing_install(ROOT) if existing is None else bool(existing)
        self.mode = "choose" if self.existing else "install"
        self.exit_code = 0

        root.title(f"{APP_NAME} Setup")
        root.geometry("660x600")
        root.minsize(580, 520)
        apply_theme(root, ttk, self.palette)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.container = ttk.Frame(root, padding=18)
        self.container.pack(fill="both", expand=True)

        self.backend_var = tk.StringVar(value=self.recommended)
        self.portable_var = tk.BooleanVar(value=True)
        self.path_var = tk.StringVar(value=str(ROOT / "home"))
        self.path_var.trace_add("write", self._on_path_typed)
        self.missing_data = ""
        if self.existing:
            current, exists = install_manifest().configured_data_dir(ROOT)
            if current and os.path.normcase(current) != os.path.normcase(str(ROOT / "home")):
                self.path_var.set(current)
                self.portable_var.set(False)
                if not exists:
                    self.missing_data = current
        self.store_var = tk.BooleanVar(value=True)
        self.appwin_var = tk.BooleanVar(value=False)
        self.path_cmd_var = tk.BooleanVar(value=False)
        self.shortcut_var = tk.StringVar(value="launcher")
        self.deps_var = tk.BooleanVar(value=True)
        self.plugin_vars = {
            name: tk.BooleanVar(value=name in RECOMMENDED_PLUGINS)
            for name, _ in self.plugin_rows
        }
        self.choice_var = tk.StringVar(value="repair")
        self.purge_var = tk.BooleanVar(value=False)

        self.pages = []
        self._build_runtime_page()
        self._build_location_page()
        self._build_features_page()
        self._build_options_page()
        self._build_install_page()
        self._build_choice_frame()
        self._build_uninstall_frame()
        self._build_footer()
        if self.existing:
            self._show_choice()
        else:
            self._show(0)
        set_title_bar_theme(root, self.theme == "dark")

    # -- pages --------------------------------------------------------------

    def _page(self, title):
        frame = self.ttk.Frame(self.container)
        self.pages.append((title, frame))
        return frame

    def _heading(self, parent, text, sub=""):
        self.ttk.Label(parent, text=text,
                       font=("Segoe UI", 15, "bold")).pack(anchor="w")
        if sub:
            self.ttk.Label(parent, text=sub, wraplength=590,
                           style="Dim.TLabel").pack(anchor="w", pady=(2, 12))

    def _build_runtime_page(self):
        ttk = self.ttk
        page = self._page("Inference runtime")
        detected = (f"Detected: {self.vendor.upper()} graphics" if self.vendor
                    else "No GPU detected")
        self._heading(page, f"Install {APP_NAME}",
                      f"{detected}. Every answer has a sensible default, so you "
                      "can click through this.")
        ttk.Label(page, text="Which inference runtime should LocaLM use?",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for key, desc in backend_choices():
            suffix = ("   (recommended for your hardware)"
                      if key == self.recommended else "")
            ttk.Radiobutton(page, text=desc + suffix, value=key,
                            variable=self.backend_var).pack(anchor="w")

    def _build_location_page(self):
        ttk = self.ttk
        page = self._page("Where things live")
        self._heading(page, "Where things live",
                      "Both of these can stay inside this folder, which keeps "
                      "the install self-contained.")

        ttk.Label(page, text="Models and data",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Radiobutton(page, text="Inside this folder - delete it and "
                                   "everything is gone",
                        value=True, variable=self.portable_var).pack(anchor="w")
        ttk.Label(page, text=str(ROOT / "home"), style="Dim.TLabel",
                  wraplength=560).pack(anchor="w", padx=(22, 0))
        row = ttk.Frame(page)
        ttk.Radiobutton(row, text="A folder I choose:", value=False,
                        variable=self.portable_var).pack(side="left")
        ttk.Entry(row, textvariable=self.path_var, width=30).pack(side="left", padx=6)
        ttk.Button(row, text="Browse...", command=self._browse).pack(side="left")
        row.pack(anchor="w", pady=(2, 0))
        if self.missing_data:
            ttk.Label(page, text=f"[!] The data folder this install uses, "
                                 f"{self.missing_data}, is not available right now "
                                 "(a drive that is not connected, or a network folder "
                                 "that is offline). Connect it before you continue, "
                                 "or choose another folder.",
                      wraplength=560, style="Warn.TLabel").pack(anchor="w", pady=(4, 0))

        ttk.Label(page, text="Python tooling",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(16, 0))
        ttk.Radiobutton(page, text="Keep it inside this folder - nothing is "
                                   "written to your user profile",
                        value=True, variable=self.store_var).pack(anchor="w")
        ttk.Radiobutton(page, text="Share it with other installs - saves disk if "
                                   "you have more than one",
                        value=False, variable=self.store_var).pack(anchor="w")

    def _build_features_page(self):
        ttk = self.ttk
        page = self._page("Optional features")
        self._heading(page, "Optional features",
                      "Chat is always installed. Pick anything else you want; "
                      "you can add or remove these later in Settings.")
        if not self.plugin_rows:
            ttk.Label(page, text="The feature list could not be read, so none "
                                 "are preselected. Choose them after setup "
                                 "with:  localm plugin setup",
                      wraplength=590, style="Warn.TLabel").pack(anchor="w")
            return
        for name, desc in self.plugin_rows:
            ttk.Checkbutton(page, text=f"{name} - {desc}",
                            variable=self.plugin_vars[name]).pack(anchor="w")
        ttk.Checkbutton(page, text="Also install what these features need "
                                   "(downloads more)",
                        variable=self.deps_var).pack(anchor="w", pady=(14, 0))

    def _build_options_page(self):
        ttk = self.ttk
        page = self._page("Options")
        self._heading(page, "Options", "The last few. None of these is required.")
        ttk.Checkbutton(page, text="Open LocaLM in its own app window instead of "
                                   "a browser tab",
                        variable=self.appwin_var).pack(anchor="w")
        ttk.Checkbutton(page, text="Make 'localm' runnable from any terminal",
                        variable=self.path_cmd_var).pack(anchor="w")
        ttk.Label(page, text="Desktop shortcut:" if IS_WINDOWS else "Shortcut:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 0))
        for key, text in (("launcher", "Launcher menu (GUI / chat / server / coder)"),
                          ("gui", "Straight to the GUI"),
                          ("none", "No shortcut")):
            ttk.Radiobutton(page, text=text, value=key,
                            variable=self.shortcut_var).pack(anchor="w")

    def _build_install_page(self):
        ttk = self.ttk
        tk = self.tk
        page = self._page("Installing")
        self.step_label = ttk.Label(page, text="", font=("Segoe UI", 11, "bold"))
        self.bar = ttk.Progressbar(page, mode="determinate")
        self.log = tk.Text(page, height=18, wrap="none", font=("Consolas", 9),
                           **text_options(self.palette))
        self.log_scroll = ttk.Scrollbar(page, command=self.log.yview)
        self.log.configure(yscrollcommand=self.log_scroll.set, state="disabled")

    def _build_choice_frame(self):
        ttk = self.ttk
        frame = self.choice_frame = ttk.Frame(self.container)
        self._heading(frame, f"{APP_NAME} is already set up here",
                      f"{ROOT}\nWhat would you like to do?")
        ttk.Radiobutton(frame, text="Repair or reinstall - your chats, settings "
                                    "and models are kept",
                        value="repair", variable=self.choice_var).pack(anchor="w")
        ttk.Radiobutton(frame, text=f"Uninstall {APP_NAME} from this computer",
                        value="uninstall", variable=self.choice_var).pack(anchor="w")

    def _build_uninstall_frame(self):
        ttk = self.ttk
        tk = self.tk
        frame = self.uninstall_frame = ttk.Frame(self.container)
        self._heading(frame, f"Uninstall {APP_NAME}",
                      "This removes what setup installed in this folder and "
                      "anything it added elsewhere (shortcut, PATH entry).")
        ttk.Checkbutton(frame, text="Also delete my saved data - chats, settings, "
                                    "downloaded models, generated images",
                        variable=self.purge_var,
                        command=self._refresh_plan).pack(anchor="w", pady=(0, 8))
        self.plan_text = tk.Text(frame, height=16, wrap="none", font=("Consolas", 9),
                                 **text_options(self.palette))
        self.plan_text.pack(fill="both", expand=True)
        self.plan_text.configure(state="disabled")

    def _refresh_plan(self) -> None:
        """Show what uninstall would do with the current choices."""
        try:
            im = install_manifest()
            rep = im.uninstall(ROOT, purge_data=bool(self.purge_var.get()),
                               dry_run=True, defer_runtime=True)
            lines = im.format_report(rep)
        except Exception as e:
            lines = [f"[!] Could not work out what to remove: {e}"]
        self.plan_text.configure(state="normal")
        self.plan_text.delete("1.0", "end")
        self.plan_text.insert("end", "\n".join(lines) + "\n")
        self.plan_text.configure(state="disabled")

    def _build_footer(self):
        ttk = self.ttk
        footer = ttk.Frame(self.container)
        footer.pack(fill="x", side="bottom", pady=(12, 0))
        self.status = ttk.Label(footer, text="")
        self.status.pack(side="left")
        self.action = ttk.Button(footer, text="Next", command=self.next_page)
        self.action.pack(side="right")
        self.back = ttk.Button(footer, text="Back", command=self.prev_page)
        self.back.pack(side="right", padx=(0, 8))

    # -- navigation ---------------------------------------------------------

    @property
    def last_question_page(self) -> int:
        return len(self.pages) - 2

    def _hide_all(self) -> None:
        for _, frame in self.pages:
            frame.pack_forget()
        self.choice_frame.pack_forget()
        self.uninstall_frame.pack_forget()

    def _show(self, i: int) -> None:
        self._hide_all()
        self.pages[i][1].pack(fill="both", expand=True)
        self.index = i
        self.status.configure(text="")
        self.back.configure(state="disabled" if i == 0 and not self.existing
                            else "normal")
        self.action.configure(
            text="Install" if i == self.last_question_page else "Next")

    def _show_choice(self) -> None:
        self._hide_all()
        self.mode = "choose"
        self.choice_frame.pack(fill="both", expand=True)
        self.status.configure(text="")
        self.back.configure(state="disabled")
        self.action.configure(text="Next")

    def _show_uninstall(self) -> None:
        self._hide_all()
        self.mode = "uninstall"
        self.uninstall_frame.pack(fill="both", expand=True)
        self.status.configure(text="")
        self.back.configure(state="normal")
        self.action.configure(text="Uninstall")
        self._refresh_plan()

    def _problem(self) -> str:
        """Why the current page cannot be left, or empty."""
        if self.pages[self.index][0] == "Where things live":
            if not self.portable_var.get() and not self.path_var.get().strip():
                return "Choose a data folder, or pick the one inside this folder."
        return ""

    def next_page(self) -> None:
        if self.installing:
            return
        if self.mode == "choose":
            if self.choice_var.get() == "uninstall":
                self._show_uninstall()
            else:
                self.mode = "install"
                self._show(0)
            return
        if self.mode == "uninstall":
            self.start_uninstall()
            return
        problem = self._problem()
        if problem:
            self.status.configure(text=problem)
            return
        if self.index == self.last_question_page:
            self.start_install()
            return
        self._show(self.index + 1)

    def prev_page(self) -> None:
        if self.installing:
            return
        if self.mode == "uninstall" or (self.mode == "install" and self.index == 0
                                        and self.existing):
            self._show_choice()
            return
        if self.mode != "install" or self.index == 0:
            return
        self._show(self.index - 1)

    def current_plan(self) -> Plan:
        """Everything the pages have collected so far."""
        return Plan(
            backend=self.backend_var.get(),
            app_window=bool(self.appwin_var.get()),
            portable_data=bool(self.portable_var.get()),
            data_path=self.path_var.get().strip(),
            shortcut=self.shortcut_var.get(),
            add_to_path=bool(self.path_cmd_var.get()),
            portable_store=bool(self.store_var.get()),
            plugins=tuple(n for n, _ in self.plugin_rows
                          if self.plugin_vars[n].get()),
            plugin_deps=bool(self.deps_var.get()),
        )

    def _browse(self) -> None:
        chosen = self.filedialog.askdirectory(title="Choose a data folder")
        if chosen:
            self.path_var.set(chosen)
            self.portable_var.set(False)

    def _on_path_typed(self, *_args) -> None:
        if _is_custom_folder_text(self.path_var.get()):
            self.portable_var.set(False)

    # -- running the install ------------------------------------------------

    def start_install(self) -> None:
        plan = self.current_plan()
        self.installing = True
        self._show(len(self.pages) - 1)
        self.step_label.pack(anchor="w")
        self.bar.pack(fill="x", pady=(8, 10))
        self.log.pack(side="left", fill="both", expand=True)
        self.log_scroll.pack(side="right", fill="y")
        self.back.configure(state="disabled")
        self.action.configure(state="disabled", text="Installing...")

        self.lines = queue.Queue()
        resume = begin_journal(self._emit)
        steps = build_steps(plan, resume)
        threading.Thread(target=self._worker, args=(steps,), daemon=True).start()
        self.root.after(80, self._pump)

    def _emit(self, text: str) -> None:
        self.lines.put(("log", text))

    def _worker(self, steps: List[Step]) -> None:
        def on_step(i: int, total: int, label: str) -> None:
            self.lines.put(("step", (i, total, label)))

        failures, fatal = run_steps(steps, self._emit, on_step)
        if fatal:
            self.lines.put(("done", fatal))
            return
        self.lines.put(("done", None if not failures
                        else "PARTIAL:" + "; ".join(failures)))

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.lines.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", str(payload) + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "step":
                    i, total, label = payload
                    self.step_label.configure(
                        text=f"Step {i + 1} of {total}: {label}")
                    self.bar.configure(maximum=total, value=i)
                elif kind == "done":
                    self._finish(payload)
                    return
                elif kind == "uninstalled":
                    self._finish_uninstall(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _on_close(self) -> None:
        if self.installing and self.mode == "uninstalling":
            if self.messagebox is not None:
                self.messagebox.showinfo(
                    f"Uninstall {APP_NAME}",
                    "The uninstall is still running. Close this window when it "
                    "has finished.")
            return
        self.root.destroy()

    # -- running the uninstall ----------------------------------------------

    def start_uninstall(self) -> None:
        purge = bool(self.purge_var.get())
        if self.messagebox is not None:
            question = (f"Remove {APP_NAME} from this folder?\n\n"
                        + ("Your saved data will be DELETED too."
                           if purge else "Your saved data is kept."))
            if not self.messagebox.askyesno(f"Uninstall {APP_NAME}", question):
                return
        self.installing = True
        self.mode = "uninstalling"
        self._hide_all()
        self.pages[-1][1].pack(fill="both", expand=True)
        self.step_label.configure(text=f"Uninstalling {APP_NAME} ...")
        self.step_label.pack(anchor="w")
        self.log.pack(side="left", fill="both", expand=True)
        self.log_scroll.pack(side="right", fill="y")
        self.back.configure(state="disabled")
        self.action.configure(state="disabled", text="Uninstalling...")
        self.lines = queue.Queue()
        threading.Thread(target=self._uninstall_worker, args=(purge,),
                         daemon=True).start()
        self.root.after(80, self._pump)

    def _uninstall_worker(self, purge: bool) -> None:
        try:
            im = install_manifest()
            rep = im.uninstall(ROOT, purge_data=purge, force=True,
                               stop_running=True, defer_runtime=True)
            for line in im.format_report(rep):
                self.lines.put(("log", line))
        except Exception as e:                  # never leave the UI hanging
            rep = {"exit": 1, "error": str(e)}
            self.lines.put(("log", f"[!] Uninstall failed: {e}"))
        self.lines.put(("uninstalled", rep))

    def _finish_uninstall(self, rep: dict) -> None:
        self.installing = False
        self.action.configure(text="Close", state="normal", command=self.root.destroy)
        if rep.get("exit") not in (0, 2):
            self.exit_code = EXIT_UNINSTALL_FAILED
            self.step_label.configure(text="Uninstall could not finish.")
            self.status.configure(text=(rep.get("error") or
                                        f"See above. Close any {APP_NAME} window "
                                        "and try again.")[:90])
            return
        pending = (ROOT / ".localm-uninstall-pending").is_file()
        partial = rep.get("exit") == 2
        if pending:
            self.exit_code = EXIT_FINISH_UNINSTALL_PARTIAL if partial else EXIT_FINISH_UNINSTALL
        else:
            self.exit_code = EXIT_UNINSTALL_PARTIAL if partial else 0
        self.step_label.configure(
            text=f"{APP_NAME} is removed"
                 + (", but some things you asked to delete were not deleted."
                    if partial else ".")
                 + (" Close this window to finish." if pending else ""))
        if partial:
            self.status.configure(text="See REFUSED in the list above for what was kept and why.")
        else:
            self.status.configure(
                text="The Python runtime this window runs on is removed after it "
                     "closes." if pending else "Done.")

    def _finish(self, error: Optional[str]) -> None:
        self.bar.configure(value=self.bar["maximum"])
        if error is None:
            self.step_label.configure(text=f"{APP_NAME} is installed.")
            self.status.configure(text="Done.")
            self.action.configure(text=f"Start {APP_NAME}", state="normal",
                                  command=self._launch)
        elif str(error).startswith("PARTIAL:"):
            detail = str(error)[len("PARTIAL:"):]
            self.step_label.configure(
                text=f"{APP_NAME} is installed, but some optional steps did "
                     "not finish.")
            self.status.configure(text=detail[:90])
            self.action.configure(text=f"Start {APP_NAME}", state="normal",
                                  command=self._launch)
        else:
            self.step_label.configure(text="Setup could not finish.")
            self.status.configure(text=str(error)[:90])
            self.action.configure(text="Close", state="normal",
                                  command=self.root.destroy)

    def _launch(self) -> None:
        exe = venv_bin(ROOT) / ("localm.exe" if IS_WINDOWS else "localm")
        try:
            subprocess.Popen([str(exe), "gui"], cwd=str(ROOT))
        except OSError:
            pass
        self.root.destroy()


def main() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    wizard = Wizard(root, tk, ttk, filedialog, messagebox=messagebox)
    root.mainloop()
    return wizard.exit_code


if __name__ == "__main__":
    if find_uv(ROOT) is None:
        print("uv is required and was not found. Run setup.bat / setup.sh instead.",
              file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
