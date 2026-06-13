"""
localm launcher - double-click to configure and start localm.

Pick a mode (Web GUI, terminal chat, API server, coder agent), a model,
and options like debug logging or the context window, then Launch. Each
mode opens in its own console window; the launcher remembers your choices
in ~/.localm/launcher.json.

Pure tkinter - no extra dependencies, works offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog, ttk

REPO_DIR = Path(__file__).resolve().parent


def _settings_file() -> Path:
    """Launcher settings live in the localm data dir (LOCALM_HOME / portable
    home/ folder / ~/.localm) so each checkout stays self-contained."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.config import HOME_DIR
        return HOME_DIR / "launcher.json"
    except Exception:
        return Path.home() / ".localm" / "launcher.json"


SETTINGS_FILE = _settings_file()

# ----- palette (matches the web GUI's dark theme) -----
BG = "#0f1115"
BG_RAISED = "#161a21"
BG_INPUT = "#1c212b"
BORDER = "#262c38"
TEXT = "#d7dde7"
TEXT_DIM = "#8b94a5"
ACCENT = "#4f9cf9"
GREEN = "#3fb68b"

MODES = [
    ("gui", "Web GUI", "Chat, coder agent, models, and images in your browser"),
    ("chat", "Chat (terminal)", "Interactive chat in a console window"),
    ("serve", "API server", "OpenAI-compatible server only, no UI"),
    ("coder", "Coder agent", "AI coding agent in a project folder"),
]

# Session persistence (privacy) modes - see localm/audit.py
PRIVACY_MODES = ["privacy", "log", "full"]
USE_GLOBAL = "(use global)"


def python_exe() -> str:
    """Prefer the repo venv; fall back to the interpreter running this script."""
    for candidate in (REPO_DIR / ".venv" / "Scripts" / "python.exe",):
        if candidate.is_file():
            return str(candidate)
    return sys.executable.replace("pythonw.exe", "python.exe")


def load_models() -> list:
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.config import load_registry
        # Pick up models added to (or gone missing from) the models folder since
        # last refresh. Guarded so an older localm without sync still lists fine.
        try:
            from localm.model_manager import sync_models_dir
            sync_models_dir()
        except Exception:
            pass
        return sorted(load_registry())
    except Exception:
        return []


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("localm launcher")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._style()

        saved = load_settings()
        self.mode = tk.StringVar(value=saved.get("mode", "gui"))
        self.model = tk.StringVar(value=saved.get("model", ""))
        self.debug = tk.BooleanVar(value=saved.get("debug", False))
        self.port = tk.StringVar(value=saved.get("port", ""))
        self.ctx = tk.StringVar(value=saved.get("ctx", ""))
        self.gpu_layers = tk.StringVar(value=saved.get("gpu_layers", ""))
        self.host_lan = tk.BooleanVar(value=False)   # deliberately not persisted
        self.no_browser = tk.BooleanVar(value=saved.get("no_browser", False))
        self.coder_dir = tk.StringVar(value=saved.get("coder_dir", ""))
        self.coder_yes = tk.BooleanVar(value=saved.get("coder_yes", False))
        self.keep_open = tk.BooleanVar(value=saved.get("keep_open", False))
        self.privacy_global = tk.StringVar(value=saved.get("privacy_global", "privacy"))
        self.privacy_chat = tk.StringVar(value=saved.get("privacy_chat", USE_GLOBAL))
        self.privacy_coder = tk.StringVar(value=saved.get("privacy_coder", USE_GLOBAL))

        self._build()
        self._on_mode_change()

    # ------------------------------------------------------------- #

    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Card.TFrame", background=BG_RAISED)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Card.TLabel", background=BG_RAISED, foreground=TEXT)
        s.configure("Dim.TLabel", background=BG_RAISED, foreground=TEXT_DIM,
                    font=("Segoe UI", 9))
        s.configure("Title.TLabel", background=BG, foreground=TEXT,
                    font=("Segoe UI", 16, "bold"))
        s.configure("TCheckbutton", background=BG_RAISED, foreground=TEXT,
                    focuscolor=BG_RAISED)
        s.map("TCheckbutton",
              background=[("active", BG_RAISED)],
              foreground=[("active", TEXT)])
        s.configure("TRadiobutton", background=BG_RAISED, foreground=TEXT,
                    focuscolor=BG_RAISED, font=("Segoe UI", 10, "bold"))
        s.map("TRadiobutton",
              background=[("active", BG_RAISED)],
              foreground=[("selected", ACCENT), ("active", TEXT)])
        s.configure("TEntry", fieldbackground=BG_INPUT, foreground=TEXT,
                    insertcolor=TEXT, bordercolor=BORDER)
        s.configure("TCombobox", fieldbackground=BG_INPUT, foreground=TEXT,
                    background=BG_INPUT, arrowcolor=TEXT_DIM,
                    bordercolor=BORDER)
        s.map("TCombobox",
              fieldbackground=[("readonly", BG_INPUT)],
              foreground=[("readonly", TEXT)])
        s.configure("Launch.TButton", background=ACCENT, foreground="#ffffff",
                    font=("Segoe UI", 11, "bold"), padding=(24, 8),
                    borderwidth=0)
        s.map("Launch.TButton", background=[("active", "#3d86e0")])
        s.configure("Quiet.TButton", background=BG_INPUT, foreground=TEXT_DIM,
                    borderwidth=0, padding=(10, 4))
        s.map("Quiet.TButton", background=[("active", BORDER)],
              foreground=[("active", TEXT)])
        self.option_add("*TCombobox*Listbox.background", BG_INPUT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _card(self, parent) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        inner = ttk.Frame(outer, style="Card.TFrame", padding=14)
        inner.pack(fill="both", expand=True)
        outer.pack(fill="x", pady=(0, 12))
        return inner

    def _build(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 14))
        title = tk.Label(header, text="local", bg=BG, fg=TEXT,
                         font=("Segoe UI", 18, "bold"))
        title.pack(side="left")
        tk.Label(header, text="m", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(header, text="  launcher", bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 12)).pack(side="left", pady=(5, 0))

        # ----- mode -----
        mode_card = self._card(root)
        for value, label, hint in MODES:
            row = ttk.Frame(mode_card, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Radiobutton(row, text=label, value=value, variable=self.mode,
                            command=self._on_mode_change).pack(side="left")
            ttk.Label(row, text="  " + hint, style="Dim.TLabel").pack(side="left")

        # ----- model -----
        model_card = self._card(root)
        ttk.Label(model_card, text="Model", style="Card.TLabel").grid(
            row=0, column=0, sticky="w")
        self.model_box = ttk.Combobox(model_card, textvariable=self.model,
                                      state="readonly", width=46)
        self.model_box.grid(row=1, column=0, sticky="we", pady=(4, 0))
        ttk.Button(model_card, text="refresh", style="Quiet.TButton",
                   command=self._refresh_models).grid(
            row=1, column=1, padx=(8, 0), pady=(4, 0))
        model_card.columnconfigure(0, weight=1)

        # ----- import a model (for empty registries / new models) -----
        imp = ttk.Frame(model_card, style="Card.TFrame")
        imp.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(imp, text="Import:", style="Dim.TLabel").pack(side="left")
        self.import_btns = []
        for text, cmd in (("from file…", self._import_from_file),
                          ("from folder…", self._import_from_folder),
                          ("from URL…", self._import_from_url)):
            b = ttk.Button(imp, text=text, style="Quiet.TButton", command=cmd)
            b.pack(side="left", padx=(8, 0))
            self.import_btns.append(b)
        # NOTE: the initial _refresh_models() runs at the END of _build -
        # its "no models" message needs the footer status label to exist.

        # ----- options -----
        opt_card = self._card(root)
        self.opt_card = opt_card
        ttk.Checkbutton(opt_card, text="Debug mode  (log file, native stderr "
                        "capture, raw model output)",
                        variable=self.debug).grid(
            row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(opt_card, text="Port", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        self.port_entry = ttk.Entry(opt_card, textvariable=self.port, width=8)
        self.port_entry.grid(row=2, column=0, sticky="w")
        ttk.Label(opt_card, text="Context", style="Card.TLabel").grid(
            row=1, column=1, sticky="w", pady=(10, 0), padx=(14, 0))
        ttk.Entry(opt_card, textvariable=self.ctx, width=8).grid(
            row=2, column=1, sticky="w", padx=(14, 0))
        ttk.Label(opt_card, text="GPU layers", style="Card.TLabel").grid(
            row=1, column=2, sticky="w", pady=(10, 0), padx=(14, 0))
        ttk.Entry(opt_card, textvariable=self.gpu_layers, width=8).grid(
            row=2, column=2, sticky="w", padx=(14, 0))
        ttk.Label(opt_card, text="(blank = automatic / config default)",
                  style="Dim.TLabel").grid(
            row=2, column=3, sticky="w", padx=(14, 0))

        self.gui_opts = ttk.Frame(opt_card, style="Card.TFrame")
        ttk.Checkbutton(self.gui_opts, text="Don't open the browser",
                        variable=self.no_browser).pack(side="left")
        self.gui_opts.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.serve_opts = ttk.Frame(opt_card, style="Card.TFrame")
        ttk.Checkbutton(self.serve_opts,
                        text="Expose on the network (0.0.0.0 - set "
                             "LOCALM_API_KEY first!)",
                        variable=self.host_lan).pack(side="left")
        self.serve_opts.grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.coder_opts = ttk.Frame(opt_card, style="Card.TFrame")
        ttk.Label(self.coder_opts, text="Project folder",
                  style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.coder_opts, textvariable=self.coder_dir,
                  width=42).grid(row=1, column=0, sticky="we")
        ttk.Button(self.coder_opts, text="browse…", style="Quiet.TButton",
                   command=self._pick_dir).grid(row=1, column=1, padx=(8, 0))
        ttk.Checkbutton(self.coder_opts,
                        text="Auto-approve destructive tools (--yes)",
                        variable=self.coder_yes).grid(
            row=2, column=0, sticky="w", pady=(8, 0))
        self.coder_opts.columnconfigure(0, weight=1)
        self.coder_opts.grid(row=5, column=0, columnspan=4, sticky="we", pady=(10, 0))

        # ----- privacy / session persistence -----
        priv_card = self._card(root)
        ttk.Label(priv_card, text="Privacy (session persistence)",
                  style="Card.TLabel").grid(row=0, column=0, columnspan=3,
                                            sticky="w")
        ttk.Label(priv_card,
                  text="privacy = no traces saved  ·  log = JSONL audit trail "
                       "(~/.localm/sessions)  ·  full = log + transcript",
                  style="Dim.TLabel").grid(row=1, column=0, columnspan=3,
                                           sticky="w", pady=(0, 6))

        ttk.Label(priv_card, text="Global", style="Card.TLabel").grid(
            row=2, column=0, sticky="w")
        ttk.Combobox(priv_card, textvariable=self.privacy_global,
                     values=PRIVACY_MODES, state="readonly", width=12).grid(
            row=3, column=0, sticky="w")
        ttk.Label(priv_card, text="Chat", style="Card.TLabel").grid(
            row=2, column=1, sticky="w", padx=(14, 0))
        ttk.Combobox(priv_card, textvariable=self.privacy_chat,
                     values=[USE_GLOBAL] + PRIVACY_MODES, state="readonly",
                     width=12).grid(row=3, column=1, sticky="w", padx=(14, 0))
        ttk.Label(priv_card, text="Coder", style="Card.TLabel").grid(
            row=2, column=2, sticky="w", padx=(14, 0))
        ttk.Combobox(priv_card, textvariable=self.privacy_coder,
                     values=[USE_GLOBAL] + PRIVACY_MODES, state="readonly",
                     width=12).grid(row=3, column=2, sticky="w", padx=(14, 0))

        # ----- footer -----
        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(footer, text="Keep launcher open",
                        variable=self.keep_open,
                        style="TCheckbutton").pack(side="left")
        self.status = tk.Label(footer, text="", bg=BG, fg=GREEN,
                               font=("Segoe UI", 9))
        self.status.pack(side="left", padx=12)
        ttk.Button(footer, text="Launch", style="Launch.TButton",
                   command=self._launch).pack(side="right")

        # Populate the model list last - on a fresh install with an empty
        # registry this shows a hint in the status label built just above.
        self._refresh_models()

    # ------------------------------------------------------------- #

    def _refresh_models(self) -> None:
        models = load_models()
        self.model_box["values"] = models
        if models and self.model.get() not in models:
            self.model.set(models[0])
        if not models:
            self.model.set("")
            self.status_msg("No models yet - Import one, or launch the Web GUI "
                            "to add one there", error=True)

    # ------------------------- model import ----------------------- #

    def _set_import_enabled(self, enabled: bool) -> None:
        for b in self.import_btns:
            b.configure(state="normal" if enabled else "disabled")

    def _import_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a GGUF model file",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if path:
            self._register_path(path)

    def _import_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select a HuggingFace model directory")
        if path:
            self._register_path(path)

    def _register_path(self, path: str) -> None:
        """Register a local file/dir via `localm add` off the UI thread.
        SHA256 hashing of a multi-GB file can take a few seconds."""
        before = set(load_models())
        self._set_import_enabled(False)
        self.status_msg("Importing… (hashing may take a moment)")

        def work():
            ok, msg = False, ""
            try:
                proc = subprocess.run(
                    [python_exe(), "-m", "localm", "add", path,
                     "--on-duplicate", "alias"],
                    cwd=str(REPO_DIR), capture_output=True, text=True,
                    timeout=900)
                ok = proc.returncode == 0
                out = (proc.stdout + proc.stderr).strip().splitlines()
                msg = out[-1] if out else ""
            except Exception as e:
                msg = str(e)
            self.after(0, lambda: self._register_done(ok, msg, before))

        threading.Thread(target=work, daemon=True).start()

    def _register_done(self, ok: bool, msg: str, before: set) -> None:
        self._set_import_enabled(True)
        self._refresh_models()
        new = sorted(set(load_models()) - before)
        if ok and new:
            self.model.set(new[0])
            self.status_msg(f"Imported {new[0]} ✓")
        elif ok:
            self.status_msg("Imported ✓ (already registered)")
        else:
            self.status_msg(f"Import failed: {msg[:70]}", error=True)

    def _import_from_url(self) -> None:
        """Download from a URL/HuggingFace spec inside the Web GUI, where the
        Models page shows a live progress bar. Opens a model-less GUI that
        starts the pull immediately."""
        spec = simpledialog.askstring(
            "Import from URL",
            "HuggingFace repo, repo:file.gguf, or https URL:",
            parent=self)
        if not spec or not spec.strip():
            return
        spec = spec.strip()
        cmd = [python_exe(), "-m", "localm", "gui", "--pull", spec]
        port = self.port.get().strip()
        if port:
            cmd += ["-p", port]
        cmd += ["--mode", self.privacy_global.get() or "privacy"]
        try:
            subprocess.Popen(cmd, cwd=str(REPO_DIR),
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            self.status_msg(f"Launch failed: {e}", error=True)
            return
        self.status_msg("Downloading in the Web GUI - watch the Models page ✓")
        if not self.keep_open.get():
            self.after(900, self.destroy)

    def _pick_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.coder_dir.get() or str(Path.home()))
        if chosen:
            self.coder_dir.set(chosen)

    def _on_mode_change(self) -> None:
        mode = self.mode.get()
        self.gui_opts.grid_remove()
        self.serve_opts.grid_remove()
        self.coder_opts.grid_remove()
        if mode == "gui":
            self.gui_opts.grid()
        elif mode == "serve":
            self.serve_opts.grid()
        elif mode == "coder":
            self.coder_opts.grid()
        self.port_entry.configure(
            state="normal" if mode in ("gui", "serve") else "disabled")

    def status_msg(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, fg="#e25d5d" if error else GREEN)

    # ------------------------------------------------------------- #

    def _build_command(self) -> list | None:
        mode = self.mode.get()
        model = self.model.get().strip()
        # The Web GUI can open with no model (you add one on the Models page);
        # chat / serve / coder need a model to run.
        if not model and mode != "gui":
            self.status_msg("Pick or import a model first", error=True)
            return None

        cmd = [python_exe(), "-m", "localm"]
        ctx = self.ctx.get().strip()
        gpu = self.gpu_layers.get().strip()
        port = self.port.get().strip()

        # Effective privacy mode for the surface being launched
        glob_mode = self.privacy_global.get() or "privacy"
        chat_mode = self.privacy_chat.get()
        coder_mode = self.privacy_coder.get()
        surface_mode = {
            "gui": glob_mode,
            "serve": glob_mode,
            "chat": chat_mode if chat_mode != USE_GLOBAL else glob_mode,
            "coder": coder_mode if coder_mode != USE_GLOBAL else glob_mode,
        }[mode]

        if mode == "gui":
            cmd += ["gui"]
            if model:
                cmd += [model]
            if port:
                cmd += ["-p", port]
            if ctx:
                cmd += ["-c", ctx]
            if gpu:
                cmd += ["-g", gpu]
            if self.no_browser.get():
                cmd += ["--no-browser"]
            if self.debug.get():
                cmd += ["--debug"]
        elif mode == "chat":
            cmd += ["run", model]
            if ctx:
                cmd += ["-c", ctx]
            if gpu:
                cmd += ["-g", gpu]
            if self.debug.get():
                cmd += ["--debug"]
        elif mode == "serve":
            cmd += ["serve", model]
            if self.host_lan.get():
                cmd += ["-H", "0.0.0.0"]
            if port:
                cmd += ["-p", port]
            if ctx:
                cmd += ["-c", ctx]
            if gpu:
                cmd += ["-g", gpu]
            if self.debug.get():
                cmd += ["--debug"]
        elif mode == "coder":
            cmd += ["coder", "--model", model]
            folder = self.coder_dir.get().strip()
            if folder:
                cmd += ["--cwd", folder]
            if self.coder_yes.get():
                cmd += ["--yes"]
            # coder has no --debug flag; the env var below covers it
        cmd += ["--mode", surface_mode]
        return cmd

    def _save_privacy_config(self) -> None:
        """Persist the privacy choices as durable localm config so non-launcher
        invocations (plain CLI, GUI restarts) see the same modes."""
        try:
            sys.path.insert(0, str(REPO_DIR))
            from localm.config import load_config, save_config
            cfg = load_config()
            cfg["mode"] = self.privacy_global.get() or "privacy"
            chat = self.privacy_chat.get()
            coder = self.privacy_coder.get()
            cfg["chat_mode"] = None if chat == USE_GLOBAL else chat
            cfg["coder_mode"] = None if coder == USE_GLOBAL else coder
            save_config(cfg)
        except Exception:
            pass  # config write is best-effort; the --mode flag still applies

    def _launch(self) -> None:
        cmd = self._build_command()
        if cmd is None:
            return

        env = os.environ.copy()
        if self.debug.get() and "--debug" not in cmd:
            env["LOCALM_DEBUG"] = "1"

        save_settings({
            "mode": self.mode.get(),
            "model": self.model.get(),
            "debug": self.debug.get(),
            "port": self.port.get(),
            "ctx": self.ctx.get(),
            "gpu_layers": self.gpu_layers.get(),
            "no_browser": self.no_browser.get(),
            "coder_dir": self.coder_dir.get(),
            "coder_yes": self.coder_yes.get(),
            "keep_open": self.keep_open.get(),
            "privacy_global": self.privacy_global.get(),
            "privacy_chat": self.privacy_chat.get(),
            "privacy_coder": self.privacy_coder.get(),
        })
        self._save_privacy_config()

        try:
            subprocess.Popen(
                cmd,
                cwd=str(REPO_DIR),
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except Exception as e:
            self.status_msg(f"Launch failed: {e}", error=True)
            return

        self.status_msg("Launched ✓")
        if not self.keep_open.get():
            self.after(700, self.destroy)


if __name__ == "__main__":
    Launcher().mainloop()
