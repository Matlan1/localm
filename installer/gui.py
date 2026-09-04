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
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "LocaLM"
PYVER = "3.12"

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
    """A step that must not be reported as success (AGENTS.md rule 5)."""


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


def _env_for(plan: Plan) -> dict:
    """uv's environment. Portable keeps the managed interpreter AND the wheel
    cache inside the clone, so nothing is written to the user profile - the
    same containment setup.bat's Portable option gives."""
    env = dict(os.environ)
    env["LOCALM_SETUP"] = "1"
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


def build_steps(plan: Plan) -> List[Step]:
    """The install, as setup.bat performs it, in setup.bat's order."""
    steps: List[Step] = []
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
    steps.append(Step("Creating the Python environment", venv))

    def install_localm(emit):
        _run(uv_argv("pip", "install", "-p", ".venv", "-e", f".[{plan.extras}]"),
             emit, plan)
    steps.append(Step("Installing LocaLM", install_localm))

    def install_runtime_pkg(emit):
        # Carries llama.dll + ggml inside the venv; setup-llama fills it below.
        _run(uv_argv("pip", "install", "-p", ".venv", "-e", "./runtime"), emit, plan)
    steps.append(Step("Installing the native runtime package", install_runtime_pkg))

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
                _run(uv_argv("pip", "install", "-p", ".venv",
                             "transformers[kernels]~=5.12", "tokenizers==0.22.2",
                             "accelerate>=1.0", "pillow>=10.0", "soundfile>=0.12"),
                     emit, plan, allow_fail=True),
            ]
        if any(codes):
            raise StepFailed("the PyTorch stack did not install; GGUF chat still "
                             "works, models that need PyTorch will not")
    # Not fatal: a failed torch stack still leaves a working GGUF chat install,
    # which is what setup.bat also says at this point.
    steps.append(Step("Installing PyTorch and transformers", install_torch, fatal=False))

    if plan.backend != "own":
        def provision(emit):
            _run([str(venv_bin(ROOT) / "localm"), "setup-llama",
                  "--backend", plan.backend, "--yes"], emit, plan)
        steps.append(Step(f"Provisioning the {plan.backend} inference runtime", provision))

    def data_dir(emit):
        marker = ROOT / "localm-home.cfg"
        if plan.portable_data:
            (ROOT / "home").mkdir(parents=True, exist_ok=True)
            if marker.exists():
                marker.unlink()
            state["data_dir"] = str(ROOT / "home")
            state["data_created"] = True
            emit(f"Data directory: {ROOT / 'home'} (portable)")
            return
        target = Path(plan.data_path).expanduser()
        if not target.is_absolute():
            raise StepFailed(f"{target} is not an absolute path")
        # Directory first, marker second: a marker must never point at
        # something that could not be created.
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StepFailed(f"could not use {target}: {e}")
        try:
            marker.write_text(str(target), encoding="utf-8")
        except OSError as e:
            raise StepFailed(f"could not record the data directory: {e}")
        state["data_dir"] = str(target)
        state["data_created"] = True
        state["home_cfg"] = str(marker)
        emit(f"Data directory: {target}")
    steps.append(Step("Recording where data lives", data_dir))

    def launcher(emit):
        _run([str(venv_python(ROOT)), "-m", "localm", "make-launcher",
              "--force", "--quiet"], emit, plan, allow_fail=True)
    steps.append(Step("Building the launcher", launcher, fatal=False))

    if plan.shortcut != "none":
        def shortcut(emit):
            state["shortcut"] = make_shortcut(plan, emit) or ""
        steps.append(Step("Creating the desktop shortcut", shortcut, fatal=False))

    if plan.add_to_path:
        def global_cmd(emit):
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
        steps.append(Step("Adding 'localm' to your PATH", global_cmd,
                          fatal=False))

    if plan.plugins:
        def plugins(emit):
            # An explicit deps flag: the default asks, and nothing can answer.
            _run([str(venv_bin(ROOT) / "localm"), "plugin", "setup",
                  "--plugins", ",".join(plan.plugins),
                  "--with-deps" if plan.plugin_deps else "--no-deps"],
                 emit, plan)
        steps.append(Step("Installing the optional features you chose",
                          plugins, fatal=False))

    def manifest(emit):
        args = [str(venv_python(ROOT)), "-m", "localm.install_manifest",
                "record", "--root", ".",
                "--venv", str(ROOT / ".venv"),
                "--lib-dir", str(ROOT / "runtime" / "localm_llama_runtime" / "lib"),
                "--data-dir", state.get("data_dir", ""),
                "--shortcut", state.get("shortcut", ""),
                "--home-cfg", state.get("home_cfg", ""),
                "--path-dir", state.get("path_dir", ""),
                "--command-shim", state.get("command_shim", ""),
                "--stamp",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")]
        if state.get("data_created"):
            args.append("--data-created")
        if state.get("path_modified"):
            args.append("--path-modified")
        if plan.portable_store:
            args += ["--runtime-contained",
                     "--python-dir", str(ROOT / ".python"),
                     "--cache-dir", str(ROOT / ".cache")]
            if (ROOT / ".uv").exists():
                args += ["--uv-dir", str(ROOT / ".uv")]
        _run(args, emit, plan)
        emit("Recorded what this install created, so uninstall removes only that.")
    steps.append(Step("Writing the install record", manifest, fatal=False))

    return steps


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
    text = ("[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            f"Comment={comment}\n"
            f"Exec={exec_line}\n"
            f"Path={ROOT}\n"
            "Terminal=false\n"
            "Categories=Utility;Development;\n")
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

class Wizard:
    """The setup dialogue: one page per group of questions, then the install.

    Every page is built up front and shown one at a time, so Back never has to
    rebuild anything and an answer survives moving away from its page.

    next_page/prev_page and current_plan are the whole navigation surface, so a
    test can drive the dialogue without a person clicking. See
    tests/test_installer_gui.py TestWizard."""

    def __init__(self, root, tk, ttk, filedialog):
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.vendor, self.recommended = detect_recommendation()
        self.plugin_rows = plugin_choices()
        self.index = 0
        self.installing = False

        root.title(f"{APP_NAME} Setup")
        root.geometry("660x600")
        root.minsize(580, 520)

        self.container = ttk.Frame(root, padding=18)
        self.container.pack(fill="both", expand=True)

        self.backend_var = tk.StringVar(value=self.recommended)
        self.portable_var = tk.BooleanVar(value=True)
        self.path_var = tk.StringVar(value=str(ROOT / "home"))
        self.store_var = tk.BooleanVar(value=True)
        self.appwin_var = tk.BooleanVar(value=False)
        self.path_cmd_var = tk.BooleanVar(value=False)
        self.shortcut_var = tk.StringVar(value="launcher")
        self.deps_var = tk.BooleanVar(value=True)
        self.plugin_vars = {
            name: tk.BooleanVar(value=name in RECOMMENDED_PLUGINS)
            for name, _ in self.plugin_rows
        }

        self.pages = []
        self._build_runtime_page()
        self._build_location_page()
        self._build_features_page()
        self._build_options_page()
        self._build_install_page()
        self._build_footer()
        self._show(0)

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
                           foreground="#555").pack(anchor="w", pady=(2, 12))

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
        ttk.Label(page, text=str(ROOT / "home"), foreground="#555",
                  wraplength=560).pack(anchor="w", padx=(22, 0))
        row = ttk.Frame(page)
        ttk.Radiobutton(row, text="A folder I choose:", value=False,
                        variable=self.portable_var).pack(side="left")
        ttk.Entry(row, textvariable=self.path_var, width=30).pack(side="left", padx=6)
        ttk.Button(row, text="Browse...", command=self._browse).pack(side="left")
        row.pack(anchor="w", pady=(2, 0))

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
                      wraplength=590, foreground="#a33").pack(anchor="w")
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
        self.log = tk.Text(page, height=18, wrap="none", font=("Consolas", 9))
        self.log_scroll = ttk.Scrollbar(page, command=self.log.yview)
        self.log.configure(yscrollcommand=self.log_scroll.set, state="disabled")

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

    def _show(self, i: int) -> None:
        for _, frame in self.pages:
            frame.pack_forget()
        self.pages[i][1].pack(fill="both", expand=True)
        self.index = i
        self.status.configure(text="")
        self.back.configure(state="disabled" if i == 0 else "normal")
        self.action.configure(
            text="Install" if i == self.last_question_page else "Next")

    def _problem(self) -> str:
        """Why the current page cannot be left, or empty."""
        if self.pages[self.index][0] == "Where things live":
            if not self.portable_var.get() and not self.path_var.get().strip():
                return "Choose a data folder, or pick the one inside this folder."
        return ""

    def next_page(self) -> None:
        if self.installing:
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
        if self.installing or self.index == 0:
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
        steps = build_steps(plan)
        threading.Thread(target=self._worker, args=(steps,), daemon=True).start()
        self.root.after(80, self._pump)

    def _emit(self, text: str) -> None:
        self.lines.put(("log", text))

    def _worker(self, steps: List[Step]) -> None:
        failures: List[str] = []
        for i, step in enumerate(steps):
            self.lines.put(("step", (i, len(steps), step.label)))
            try:
                step.run(self._emit)
            except StepFailed as e:
                if step.fatal:
                    self.lines.put(("done", f"{step.label}: {e}"))
                    return
                failures.append(f"{step.label}: {e}")
                self.lines.put(("log", f"[!] {step.label} did not finish: {e}"))
            except Exception as e:            # never leave the UI hanging
                if step.fatal:
                    self.lines.put(("done", f"{step.label}: {e}"))
                    return
                failures.append(f"{step.label}: {e}")
                self.lines.put(("log", f"[!] {step.label} did not finish: {e}"))
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
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

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
    from tkinter import filedialog, ttk

    root = tk.Tk()
    Wizard(root, tk, ttk, filedialog)
    root.mainloop()
    return 0


if __name__ == "__main__":
    if find_uv(ROOT) is None:
        print("uv is required and was not found. Run setup.bat / setup.sh instead.",
              file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
