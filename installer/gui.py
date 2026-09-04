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
BACKEND_CHOICES = [
    ("vulkan", "Vulkan - any GPU (AMD/NVIDIA/Intel), no vendor toolkit"),
    ("cuda", "CUDA - NVIDIA, peak performance"),
    ("amd-rocm", "ROCm - AMD RX 6000 (gfx103X), self-contained"),
    ("cpu", "CPU only - no GPU"),
    ("own", "I will provide my own llama.cpp build (skip the download)"),
]

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


def torch_spec_for(backend: str) -> Optional[str]:
    """The PyTorch install arguments for *backend*, from the SAME policy
    setup.bat consults (`python -m localm.hwdetect torch-args`). None means
    this machine needs no torch stack (GGUF chat does not use it)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "localm.hwdetect", "torch-args", backend],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        spec = (out.stdout or "").strip()
        return spec or None
    except Exception:
        return None


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
    if plan.portable_store:
        env["UV_PYTHON_INSTALL_DIR"] = str(ROOT / ".python")
        env["UV_CACHE_DIR"] = str(ROOT / ".cache")
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


def build_steps(plan: Plan) -> List[Step]:
    """The install, as setup.bat performs it, in setup.bat's order."""
    steps: List[Step] = []

    def venv(emit):
        _run(uv_argv("venv", "--python", PYVER, "--python-preference",
                     "only-managed", "--clear", ".venv"), emit, plan)
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
        spec = torch_spec_for(plan.backend)
        if not spec:
            emit("No PyTorch stack needed for this backend (GGUF chat does not use it).")
            return
        if spec == "-e .[gpu]":
            _run(uv_argv("pip", "install", "-p", ".venv", "-e", ".[gpu,audio]"),
                 emit, plan, allow_fail=True)
            return
        _run(uv_argv("pip", "install", "-p", ".venv", *spec.split()),
             emit, plan, allow_fail=True)
        _run(uv_argv("pip", "install", "-p", ".venv",
                     "transformers[kernels]~=5.12", "tokenizers==0.22.2",
                     "accelerate>=1.0", "pillow>=10.0", "soundfile>=0.12"),
             emit, plan, allow_fail=True)
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
        emit(f"Data directory: {target}")
    steps.append(Step("Recording where data lives", data_dir))

    def launcher(emit):
        _run([str(venv_python(ROOT)), "-m", "localm", "make-launcher",
              "--force", "--quiet"], emit, plan, allow_fail=True)
    steps.append(Step("Building the launcher", launcher, fatal=False))

    if plan.shortcut != "none":
        def shortcut(emit):
            make_shortcut(plan, emit)
        steps.append(Step("Creating the desktop shortcut", shortcut, fatal=False))

    if plan.add_to_path:
        def global_cmd(emit):
            _run([str(venv_python(ROOT)), "-m", "localm.globalcmd",
                  "install", "--root", "."], emit, plan, allow_fail=True)
        steps.append(Step("Adding 'localm' to your PATH", global_cmd, fatal=False))

    return steps


def make_shortcut(plan: Plan, emit: Callable[[str], None]) -> None:
    """A desktop shortcut, by the same means setup.bat uses on each platform:
    a WScript.Shell .lnk on Windows, a freedesktop .desktop file elsewhere."""
    if IS_WINDOWS:
        exe = ROOT / ".venv" / "localm-app" / "LocaLM.exe"
        if plan.shortcut == "gui" and exe.exists():
            target, args = str(exe), "-m localm gui"
        elif plan.shortcut == "gui":
            target, args = str(venv_bin(ROOT) / "localm.exe"), "gui"
        else:
            target, args = str(ROOT / "localm-launcher.bat"), ""
        ico = ROOT / "assets" / "localm.ico"
        ps = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            "[Environment]::GetFolderPath('Desktop') + '\\LocaLM.lnk');"
            f"$s.TargetPath = '{target}';"
            + (f"$s.Arguments = '{args}';" if args else "")
            + f"$s.WorkingDirectory = '{ROOT}';"
            + (f"$s.IconLocation = '{ico}';" if ico.exists() else "")
            + "$s.Description = 'LocaLM';$s.Save()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       check=True, capture_output=True, text=True)
        emit("Shortcut created on your Desktop.")
        return

    exec_path = venv_bin(ROOT) / "localm"
    text = ("[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            "Comment=Local AI, offline\n"
            f"Exec={exec_path} gui\n"
            f"Path={ROOT}\n"
            "Terminal=false\n"
            "Categories=Utility;Development;\n")
    wrote = []
    for d in (Path.home() / "Desktop", Path.home() / ".local/share/applications"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / "LocaLM.desktop"
            p.write_text(text, encoding="utf-8")
            p.chmod(0o755)
            wrote.append(str(p))
        except OSError as e:
            emit(f"[!] could not write {d}: {e}")
    if not wrote:
        raise StepFailed("no desktop entry could be written")
    for w in wrote:
        emit(f"Wrote {w}")


# --------------------------------------------------------------------------- #
#  The window                                                                  #
# --------------------------------------------------------------------------- #

def main() -> int:
    import tkinter as tk
    from tkinter import filedialog, ttk

    vendor, recommended = detect_recommendation()
    plan = Plan(backend=recommended)

    root = tk.Tk()
    root.title(f"{APP_NAME} Setup")
    root.geometry("640x560")
    root.minsize(560, 480)

    container = ttk.Frame(root, padding=18)
    container.pack(fill="both", expand=True)

    # ---- page 1: the choices ------------------------------------------------
    options = ttk.Frame(container)
    options.pack(fill="both", expand=True)

    ttk.Label(options, text=f"Install {APP_NAME}",
              font=("Segoe UI", 16, "bold")).pack(anchor="w")
    detected = (f"Detected: {vendor.upper()} graphics" if vendor
                else "No GPU detected")
    ttk.Label(options, text=f"{detected}. Everything below has a sensible "
                            "default; change what you like.",
              wraplength=580, foreground="#555").pack(anchor="w", pady=(2, 14))

    # Backend
    ttk.Label(options, text="Inference runtime",
              font=("Segoe UI", 10, "bold")).pack(anchor="w")
    backend_var = tk.StringVar(value=recommended)
    labels = {}
    for key, desc in BACKEND_CHOICES:
        suffix = "   (recommended for your hardware)" if key == recommended else ""
        labels[key] = desc + suffix
        ttk.Radiobutton(options, text=labels[key], value=key,
                        variable=backend_var).pack(anchor="w")

    # Data location
    ttk.Label(options, text="Where should LocaLM keep models and data?",
              font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 0))
    portable_var = tk.BooleanVar(value=True)
    path_var = tk.StringVar(value=str(ROOT / "home"))
    ttk.Radiobutton(options, text=f"Inside this folder ({ROOT / 'home'}) - "
                                  "portable, delete it and everything is gone",
                    value=True, variable=portable_var).pack(anchor="w")
    row = ttk.Frame(options)
    ttk.Radiobutton(row, text="A folder I choose:", value=False,
                    variable=portable_var).pack(side="left")
    entry = ttk.Entry(row, textvariable=path_var, width=34)
    entry.pack(side="left", padx=6)

    def browse():
        chosen = filedialog.askdirectory(title="Choose a data folder")
        if chosen:
            path_var.set(chosen)
            portable_var.set(False)
    ttk.Button(row, text="Browse...", command=browse).pack(side="left")
    row.pack(anchor="w", pady=(2, 0))

    # Extras
    ttk.Label(options, text="Options", font=("Segoe UI", 10, "bold")).pack(
        anchor="w", pady=(14, 0))
    appwin_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="Open LocaLM in its own app window instead of "
                                  "a browser tab", variable=appwin_var).pack(anchor="w")
    path_cmd_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="Make 'localm' runnable from any terminal",
                    variable=path_cmd_var).pack(anchor="w")
    shortcut_var = tk.StringVar(value="launcher")
    ttk.Label(options, text="Desktop shortcut:").pack(anchor="w", pady=(8, 0))
    for key, text in (("launcher", "Launcher menu (GUI / chat / server / coder)"),
                      ("gui", "Straight to the GUI"),
                      ("none", "No shortcut")):
        ttk.Radiobutton(options, text=text, value=key,
                        variable=shortcut_var).pack(anchor="w")

    # ---- page 2: progress ---------------------------------------------------
    progress = ttk.Frame(container)
    step_label = ttk.Label(progress, text="", font=("Segoe UI", 11, "bold"))
    bar = ttk.Progressbar(progress, mode="determinate")
    log = tk.Text(progress, height=18, wrap="none", font=("Consolas", 9))
    log_scroll = ttk.Scrollbar(progress, command=log.yview)
    log.configure(yscrollcommand=log_scroll.set, state="disabled")

    # ---- shared footer ------------------------------------------------------
    footer = ttk.Frame(container)
    footer.pack(fill="x", side="bottom", pady=(12, 0))
    status = ttk.Label(footer, text="")
    status.pack(side="left")
    action = ttk.Button(footer, text="Install")
    action.pack(side="right")

    lines: "queue.Queue[object]" = queue.Queue()

    def emit(text: str) -> None:
        lines.put(("log", text))

    def worker(steps: List[Step]) -> None:
        failures: List[str] = []
        for i, step in enumerate(steps):
            lines.put(("step", (i, len(steps), step.label)))
            try:
                step.run(emit)
            except StepFailed as e:
                if step.fatal:
                    lines.put(("done", f"{step.label}: {e}"))
                    return
                failures.append(f"{step.label}: {e}")
                lines.put(("log", f"[!] {step.label} did not finish: {e}"))
            except Exception as e:            # never leave the UI hanging
                if step.fatal:
                    lines.put(("done", f"{step.label}: {e}"))
                    return
                failures.append(f"{step.label}: {e}")
                lines.put(("log", f"[!] {step.label} did not finish: {e}"))
        lines.put(("done", None if not failures else "PARTIAL:" + "; ".join(failures)))

    def pump() -> None:
        try:
            while True:
                kind, payload = lines.get_nowait()
                if kind == "log":
                    log.configure(state="normal")
                    log.insert("end", str(payload) + "\n")
                    log.see("end")
                    log.configure(state="disabled")
                elif kind == "step":
                    i, total, label = payload
                    step_label.configure(text=f"Step {i + 1} of {total}: {label}")
                    bar.configure(maximum=total, value=i)
                elif kind == "done":
                    finish(payload)
                    return
        except queue.Empty:
            pass
        root.after(80, pump)

    def finish(error: Optional[str]) -> None:
        bar.configure(value=bar["maximum"])
        if error is None:
            step_label.configure(text=f"{APP_NAME} is installed.")
            status.configure(text="Done.")
            action.configure(text=f"Start {APP_NAME}", state="normal",
                             command=launch)
        elif str(error).startswith("PARTIAL:"):
            detail = str(error)[len("PARTIAL:"):]
            step_label.configure(
                text=f"{APP_NAME} is installed, but some optional steps did not finish.")
            status.configure(text=detail[:90])
            action.configure(text=f"Start {APP_NAME}", state="normal",
                             command=launch)
        else:
            step_label.configure(text="Setup could not finish.")
            status.configure(text=str(error)[:90])
            action.configure(text="Close", state="normal", command=root.destroy)

    def launch() -> None:
        exe = venv_bin(ROOT) / ("localm.exe" if IS_WINDOWS else "localm")
        try:
            subprocess.Popen([str(exe), "gui"], cwd=str(ROOT))
        except OSError:
            pass
        root.destroy()

    def start_install() -> None:
        plan.backend = backend_var.get()
        plan.app_window = bool(appwin_var.get())
        plan.portable_data = bool(portable_var.get())
        plan.data_path = path_var.get().strip()
        plan.shortcut = shortcut_var.get()
        plan.add_to_path = bool(path_cmd_var.get())

        if not plan.portable_data and not plan.data_path:
            status.configure(text="Choose a data folder, or pick the portable one.")
            return

        options.pack_forget()
        progress.pack(fill="both", expand=True)
        step_label.pack(anchor="w")
        bar.pack(fill="x", pady=(8, 10))
        log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        action.configure(state="disabled", text="Installing...")
        status.configure(text="")

        steps = build_steps(plan)
        threading.Thread(target=worker, args=(steps,), daemon=True).start()
        root.after(80, pump)

    action.configure(command=start_install)
    root.mainloop()
    return 0


if __name__ == "__main__":
    if find_uv(ROOT) is None:
        print("uv is required and was not found. Run setup.bat / setup.sh instead.",
              file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
