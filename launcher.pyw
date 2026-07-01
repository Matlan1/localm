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

# Dropdown sentinel: open the Web GUI with nothing loaded (pick a model in the
# GUI). Only valid for the Web GUI mode; chat/serve/coder need a real model.
NO_MODEL_LABEL = "(no model - choose later in the GUI)"

# Wordmark treatments, shared with the web GUI through the logo_style config key
# (see localm/config.py). The blue half is the accent colour; the rest is normal
# text. Mirrors LOGO_STYLES in localm/plugins/gui/static/app.js. The console
# command, app icon, and desktop shortcut are fixed regardless of this.
LOGO_STYLES = {
    "local-m": ("LocaL", "M"),    # default: single blue M, matches the icon
    "loca-lm": ("Loca", "LM"),
    "localm": ("local", "m"),
}
LOGO_DEFAULT = "local-m"


# The launcher window must keep a FIXED width: neither a long status message
# nor a long model name in the dropdown may grow it. Status text is ellipsized
# to this many characters; the window width is also hard-pinned after build.
_STATUS_MAX_CHARS = 46


def _ellipsize(text: str, limit: int = _STATUS_MAX_CHARS) -> str:
    """Clip *text* to *limit* characters with a trailing ellipsis, so a long
    status line cannot widen the launcher window."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def logo_parts() -> tuple:
    """(white_part, blue_part) of the wordmark for the configured logo style."""
    style = LOGO_DEFAULT
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.config import load_config
        style = load_config().get("logo_style", LOGO_DEFAULT)
    except Exception:
        pass
    return LOGO_STYLES.get(style, LOGO_STYLES[LOGO_DEFAULT])


def _spawn_detached(cmd: list, *, cwd: str, env: dict | None = None):
    """Start a child mode process detached from the launcher, cross-platform.

    On Windows we open the child in its own console window via
    ``creationflags=CREATE_NEW_CONSOLE`` so the user sees its output. That flag
    (and the ``CREATE_NEW_CONSOLE`` attribute itself) is Windows-only:
    referencing it on Linux/macOS raises ``AttributeError``, which the callers'
    bare ``except`` would swallow into a silent "Launch failed" - making the
    setup.sh ``.desktop`` entry a dead button (BUG-15). On POSIX we instead use
    ``start_new_session=True`` so the child survives the launcher closing,
    without touching the Windows-only flag.
    """
    kwargs: dict = {"cwd": cwd}
    if env is not None:
        kwargs["env"] = env
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _invalid_numeric_fields(port: str, ctx: str, gpu: str):
    """Return a human-readable message if any numeric launcher field is invalid,
    else None. A stray letter or an out-of-range port used to crash the spawned
    server in pick_port ("connect_ex: port must be 0-65535") before the bug
    reporter could catch it - validate here and refuse with a clear message."""
    if port:
        try:
            p = int(port)
        except ValueError:
            return f"Port must be a whole number (got '{port}')."
        if not (1 <= p <= 65535):
            return f"Port must be between 1 and 65535 (got {p})."
    if ctx:
        try:
            if int(ctx) <= 0:
                return f"Context must be a positive whole number (got {ctx})."
        except ValueError:
            return f"Context must be a whole number (got '{ctx}')."
    if gpu:
        try:
            int(gpu)
        except ValueError:
            return f"GPU layers must be a whole number (got '{gpu}')."
    return None


def python_exe() -> str:
    """Prefer the repo venv's interpreter; fall back to the one running us.
    Handles both the Windows (Scripts/) and POSIX (bin/) venv layouts."""
    if sys.platform == "win32":
        candidates = (REPO_DIR / ".venv" / "Scripts" / "python.exe",)
    else:
        candidates = (REPO_DIR / ".venv" / "bin" / "python3",
                      REPO_DIR / ".venv" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable.replace("pythonw.exe", "python.exe")


def _auth():
    """The localm.auth module (file-backed API key), or None if unavailable."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm import auth
        return auth
    except Exception:
        return None


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


def _anchor_home(p: str) -> str:
    """Store a path home-anchored ('~/...') so launcher.json does not persist the
    absolute machine layout / OS username into a file that could be shared
    (REC-CODER-CWD-LEAK). A path outside home keeps its form (no username in it)."""
    if not p:
        return p
    try:
        rel = Path(p).resolve().relative_to(Path.home().resolve())
        return "~/" + rel.as_posix()
    except Exception:
        return p


def _expand_home(p: str) -> str:
    """Inverse of _anchor_home: expand a stored '~/...' back to an absolute path."""
    if not p:
        return p
    try:
        return str(Path(p).expanduser())
    except Exception:
        return p


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
        self.title("LocaLM launcher")
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
        self.coder_dir = tk.StringVar(value=_expand_home(saved.get("coder_dir", "")))
        self.coder_yes = tk.BooleanVar(value=saved.get("coder_yes", False))
        self.keep_open = tk.BooleanVar(value=saved.get("keep_open", False))
        self.privacy_global = tk.StringVar(value=saved.get("privacy_global", "privacy"))
        self.privacy_chat = tk.StringVar(value=saved.get("privacy_chat", USE_GLOBAL))
        self.privacy_coder = tk.StringVar(value=saved.get("privacy_coder", USE_GLOBAL))

        _a = _auth()
        self.api_key = tk.StringVar(value=(_a.get_api_key() if _a else "") or "")
        self.show_key = tk.BooleanVar(value=False)
        self.require_auth = tk.BooleanVar(
            value=bool(_a.require_auth_enabled()) if _a else False)

        self._build()
        self._on_mode_change()
        self._pin_width()

    def _pin_width(self) -> None:
        """Lock the window to its built width so nothing can widen it later.

        tkinter auto-fits a non-resizable window to its content, so a long model
        name in the dropdown or a long status line would otherwise stretch it.
        We compute the natural width once (after the layout settles) and clamp
        min == max width; height stays free so mode-switch rows can show/hide.
        The model combobox is a fixed character width and the status label is
        ellipsized, so the content never needs more than this width anyway."""
        try:
            self.update_idletasks()
            w = self.winfo_reqwidth()
            self.minsize(w, self.winfo_reqheight())
            self.maxsize(w, self.winfo_screenheight())
        except Exception:
            pass

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
        # Two labels (white + accent half) with the wordmark drawn as one word:
        # zero each label's internal padx/border so the halves butt together with
        # no gap (a tk.Label defaults to padx=1 + a 1px border, which otherwise
        # shows as a space between, e.g., "LocaL" and "M").
        white, blue = logo_parts()
        tk.Label(header, text=white, bg=BG, fg=TEXT, padx=0, bd=0,
                 highlightthickness=0, font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(header, text=blue, bg=BG, fg=ACCENT, padx=0, bd=0,
                 highlightthickness=0, font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(header, text="  launcher", bg=BG, fg=TEXT_DIM, padx=0, bd=0,
                 highlightthickness=0, font=("Segoe UI", 12)).pack(side="left", pady=(5, 0))

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
        # The Web GUI is the phone/LAN surface, so it gets the same expose toggle
        # as the API server. A network bind serves HTTPS automatically (built-in
        # local-CA TLS), so no extra option is needed for encryption.
        ttk.Checkbutton(self.gui_opts,
                        text="Expose on the network (0.0.0.0, HTTPS - set "
                             "LOCALM_API_KEY first!)",
                        variable=self.host_lan).pack(side="left", padx=(16, 0))
        self.gui_opts.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.serve_opts = ttk.Frame(opt_card, style="Card.TFrame")
        ttk.Checkbutton(self.serve_opts,
                        text="Expose on the network (0.0.0.0, HTTPS - set "
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

        # ----- authentication -----
        auth_card = self._card(root)
        ttk.Label(auth_card, text="Authentication", style="Card.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(auth_card,
                  text="API key clients must send (Bearer token). Blank = open "
                       "(local only); a key is required to expose on the network.",
                  style="Dim.TLabel").grid(row=1, column=0, columnspan=4,
                                           sticky="w", pady=(0, 6))
        self.key_entry = ttk.Entry(auth_card, textvariable=self.api_key,
                                   width=40, show="•")
        self.key_entry.grid(row=2, column=0, sticky="we", pady=(2, 0))
        self.gen_btn = ttk.Button(auth_card, text="Generate", style="Quiet.TButton",
                                  command=self._gen_key)
        self.gen_btn.grid(row=2, column=1, padx=(8, 0), pady=(2, 0))
        ttk.Button(auth_card, text="Clear", style="Quiet.TButton",
                   command=lambda: self.api_key.set("")).grid(
            row=2, column=2, padx=(8, 0), pady=(2, 0))
        ttk.Checkbutton(auth_card, text="Show", variable=self.show_key,
                        command=self._toggle_key).grid(
            row=2, column=3, padx=(8, 0), pady=(2, 0))
        ttk.Checkbutton(auth_card,
                        text="Require a key (server refuses to run without one)",
                        variable=self.require_auth).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        auth_card.columnconfigure(0, weight=1)

        # ----- footer -----
        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(footer, text="Keep launcher open",
                        variable=self.keep_open,
                        style="TCheckbutton").pack(side="left")
        # Fixed-character-width + left anchor so the status label keeps a stable
        # footprint: short messages do not stretch it and long ones are clipped
        # (status_msg ellipsizes), never widening the window.
        self.status = tk.Label(footer, text="", bg=BG, fg=GREEN,
                               font=("Segoe UI", 9), width=_STATUS_MAX_CHARS,
                               anchor="w")
        self.status.pack(side="left", padx=12)
        self.launch_btn = ttk.Button(footer, text="Launch", style="Launch.TButton",
                                     command=self._launch)
        self.launch_btn.pack(side="right")

        # Populate the model list last - on a fresh install with an empty
        # registry this shows a hint in the status label built just above.
        self._refresh_models()

    # ------------------------------------------------------------- #

    def _refresh_models(self) -> None:
        models = load_models()
        # The "no model" sentinel always leads the list so the Web GUI can be
        # launched with nothing loaded even when usable models are registered.
        values = [NO_MODEL_LABEL] + models
        self.model_box["values"] = values
        if self.model.get() not in values:
            self.model.set(models[0] if models else NO_MODEL_LABEL)
        if not models:
            self.status_msg("No models yet - Import one, or just launch the Web "
                            "GUI and add one on the Models page", error=False)

    # ------------------------- model import ----------------------- #

    def _set_busy(self, busy: bool) -> None:
        """Disable the controls that must not run mid-import: a multi-GB hash can
        take many seconds, and Launching (R48) or generating a key (R49) during it
        would race the registry write or stomp the 'hashing' status text. Re-enabled
        when the import finishes."""
        state = "disabled" if busy else "normal"
        for b in (*self.import_btns, self.launch_btn, self.gen_btn):
            try:
                b.configure(state=state)
            except tk.TclError:
                # The only realistic failure is configuring a widget that is being
                # destroyed during teardown (e.g. the window closed mid-import) -
                # harmless. We deliberately do NOT swallow other exceptions.
                pass

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
        self._set_busy(True)
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
        self._set_busy(False)
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
            _spawn_detached(cmd, cwd=str(REPO_DIR))
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
        # Ellipsize so a long line cannot widen the window (the label is fixed
        # width; the full text is still available as a tooltip-style title).
        self.status.configure(text=_ellipsize(text),
                              fg="#e25d5d" if error else GREEN)

    def _gen_key(self) -> None:
        a = _auth()
        if not a:
            self.status_msg("auth module unavailable", error=True)
            return
        self.api_key.set(a.generate_key())
        self.show_key.set(True)
        self._toggle_key()
        self.status_msg("New key generated - click Launch to save & apply it")

    def _toggle_key(self) -> None:
        self.key_entry.configure(show="" if self.show_key.get() else "•")

    # ------------------------------------------------------------- #

    def _build_command(self) -> list | None:
        mode = self.mode.get()
        model = self.model.get().strip()
        # The Web GUI can open with no model (you add or switch on the Models
        # page); chat / serve / coder need a real model to run.
        no_model = (not model) or (model == NO_MODEL_LABEL)
        if no_model and mode != "gui":
            self.status_msg("Pick or import a model first", error=True)
            return None

        cmd = [python_exe(), "-m", "localm"]
        ctx = self.ctx.get().strip()
        gpu = self.gpu_layers.get().strip()
        port = self.port.get().strip()

        # Validate the numeric fields before spawning - a letter or an
        # out-of-range port used to crash the child in pick_port (socket
        # connect_ex: "port must be 0-65535") before the bug reporter could
        # catch it. Fail here with a clear message instead.
        bad = _invalid_numeric_fields(port, ctx, gpu)
        if bad:
            self.status_msg(bad, error=True)
            return None

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
            if no_model:
                cmd += ["--no-model"]
            else:
                cmd += [model]
            # Expose the GUI on the LAN (phone access). A network bind turns on
            # built-in HTTPS automatically in the CLI - no TLS flag needed here.
            if self.host_lan.get():
                cmd += ["-H", "0.0.0.0"]
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

        # --- authentication: persist the key + require flag, inject into env ---
        key = self.api_key.get().strip()
        exposing = self.mode.get() in ("gui", "serve") and self.host_lan.get()
        if exposing and not key:
            self.status_msg("Set or Generate an API key before exposing on the "
                            "network", error=True)
            return
        if key and len(key) < 8:
            self.status_msg("API key must be at least 8 characters long.", error=True)
            return
        a = _auth()
        if a:
            try:
                a.set_api_key(key)            # writes/clears <data dir>/auth.key
                from localm.config import load_config, save_config
                cfg = load_config()
                cfg["require_auth"] = bool(self.require_auth.get())
                save_config(cfg)
            except Exception:
                pass
        if key:
            env["LOCALM_API_KEY"] = key
        if self.require_auth.get():
            env["LOCALM_REQUIRE_AUTH"] = "1"

        save_settings({
            "mode": self.mode.get(),
            "model": self.model.get(),
            "debug": self.debug.get(),
            "port": self.port.get(),
            "ctx": self.ctx.get(),
            "gpu_layers": self.gpu_layers.get(),
            "no_browser": self.no_browser.get(),
            "coder_dir": _anchor_home(self.coder_dir.get()),
            "coder_yes": self.coder_yes.get(),
            "keep_open": self.keep_open.get(),
            "privacy_global": self.privacy_global.get(),
            "privacy_chat": self.privacy_chat.get(),
            "privacy_coder": self.privacy_coder.get(),
        })
        self._save_privacy_config()

        try:
            _spawn_detached(cmd, cwd=str(REPO_DIR), env=env)
        except Exception as e:
            self.status_msg(f"Launch failed: {e}", error=True)
            return

        self.status_msg("Launched ✓")
        if not self.keep_open.get():
            self.after(700, self.destroy)


if __name__ == "__main__":
    Launcher().mainloop()
