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
import time
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


def _console_hold(cmd: list, env: dict | None) -> str:
    """The cmd.exe tail that decides whether the child's console window stays
    open after the child exits.

    Debug mode holds it whatever happened. Otherwise it is held for any
    non-zero exit, testing both signs: "errorlevel 1" alone is a >= test on a
    signed value and misses a native fault's negative code.

    Debug is read from BOTH signals the launcher uses - the --debug flag, and
    LOCALM_DEBUG for the coder mode, which has no such flag.
    See tests/test_launcher_console_hold.py."""
    source = env if env is not None else os.environ
    if "--debug" in cmd or str(source.get("LOCALM_DEBUG", "")) == "1":
        return " & pause"
    return " || pause"


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

    On Windows the child runs through ``cmd.exe /c`` with a trailing hold
    rather than directly: a console window closes the instant its owning
    process exits, so a crash closed the window before its last lines -
    including the error itself - could be read.

    The hold is ``|| pause``, which runs on any non-zero exit whatever its
    sign. ``if errorlevel 1`` is a >= test against a signed value and never
    matched a native fault's negative exit code (an access violation exits
    -1073741819), and chaining a second ``if`` after ``&`` does not help: cmd
    folds it into the first one's branch, so neither test runs.

    In debug mode the window is held unconditionally: the log is the reason
    the console is open, and a clean exit would otherwise take it away.
    Outside debug mode a clean exit still closes normally.
    """
    kwargs: dict = {"cwd": cwd}
    if env is not None:
        kwargs["env"] = env
    if sys.platform == "win32":
        cmd = ["cmd.exe", "/c", subprocess.list2cmdline(cmd) + _console_hold(cmd, env)]
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


def launch_exe() -> str:
    """The interpreter to SPAWN modes with. Prefer the branded LocaLM launcher
    (built by ``localm make-launcher`` into the venv) so the running mode shows as
    LocaLM in Task Manager / a process monitor, not python. Falls back to the plain
    venv interpreter when the launcher was not built."""
    if sys.platform == "win32":
        cand = REPO_DIR / ".venv" / "localm-app" / "LocaLM.exe"
    else:
        cand = REPO_DIR / ".venv" / "bin" / "LocaLM"
    try:
        if cand.is_file():
            return str(cand)
    except OSError:
        pass
    return python_exe()


def _auth():
    """The localm.auth module (file-backed API key), or None if unavailable."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm import auth
        return auth
    except Exception:
        return None


def load_models() -> list:
    """The launcher's chat-model dropdown entries: registered LLM names, deduped
    and sorted. A plain registry read - call sync_models_dir_safe() first to
    pick up files added to (or gone missing from) the models folder."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.config import load_registry
        reg = load_registry()
        # Only LLM (chat) models belong in the launcher's selector: this dropdown
        # launches a chat model, so an embedding / LoRA / VAE / diffusion component
        # or an unclassified 'unknown' entry must not appear here. Mirror the GUI's
        # own isLlm rule (model_type 'llm', or a legacy entry with no model_type).
        try:
            from localm.model_manager import is_llm
            names = [n for n, e in reg.items() if is_llm(e)]
        except Exception:
            # Older localm without is_llm (e.g. a mixed tree mid self-update): inline
            # the same rule so the selector still filters, not shows everything.
            names = [n for n, e in reg.items()
                     if isinstance(e, dict) and e.get("model_type", "llm") == "llm"]
        return sorted(set(names))
    except Exception:
        return []


def sync_models_dir_safe():
    """Reconcile the models folder with the registry - the same local, no-network
    scan `localm gui`/`serve` runs at startup. Returns the sync_models_dir()
    result (a ModelSyncResult), or None if it could not run."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.model_manager import sync_models_dir
        return sync_models_dir(backfill_mmproj=False)
    except Exception:
        return None


def is_models_root(path: str) -> bool:
    """True when *path* is the data directory or its models folder, not a
    specific model to import."""
    try:
        sys.path.insert(0, str(REPO_DIR))
        from localm.config import HOME_DIR, MODELS_DIR
        p = Path(path).resolve()
        return p in {HOME_DIR.resolve(), MODELS_DIR.resolve()}
    except Exception:
        return False


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
        try:
            _ico = REPO_DIR / "assets" / "localm.ico"
            if _ico.is_file():
                self.iconbitmap(str(_ico))
        except Exception:
            pass  # cosmetic; a missing/blocked icon must not stop the launcher
        self.configure(bg=BG)
        self.resizable(False, False)
        self._style()

        saved = load_settings()
        self.mode = tk.StringVar(value=saved.get("mode", "gui"))
        self.model = tk.StringVar(value=saved.get("model", ""))
        self.debug = tk.BooleanVar(value=saved.get("debug", False))
        self.keep_diagnostics = tk.BooleanVar(
            value=saved.get("keep_diagnostics", False))
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

        self._ticker_active = False

        self._build()
        self._on_mode_change()
        self._pin_width()
        self._center()

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

    def _center(self) -> None:
        """Open centered on the screen (upper third) so the launcher does not land
        in a random corner. Best-effort; never blocks startup."""
        try:
            self.update_idletasks()
            w, h = self.winfo_reqwidth(), self.winfo_reqheight()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x, y = max(0, (sw - w) // 2), max(0, (sh - h) // 3)
            self.geometry(f"+{x}+{y}")
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
        self.refresh_btn = ttk.Button(model_card, text="rescan",
                                      style="Quiet.TButton",
                                      command=self._refresh_models)
        self.refresh_btn.grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
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
                        text="Expose on the network (0.0.0.0, HTTPS)",
                        variable=self.host_lan).pack(side="left", padx=(16, 0))
        self.gui_opts.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.serve_opts = ttk.Frame(opt_card, style="Card.TFrame")
        ttk.Checkbutton(self.serve_opts,
                        text="Expose on the network (0.0.0.0, HTTPS)",
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
        # Privacy mode saves nothing - including the diagnostics a bug report
        # needs. This keeps hang/crash diagnostics + a debug log even in privacy
        # mode (never chat content). Passed as --keep-diagnostics at launch.
        ttk.Checkbutton(priv_card, text="Keep diagnostics for bug reports "
                        "(no chat content)",
                        variable=self.keep_diagnostics).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

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
        # Always-available escape hatch: file a bug report even when a mode fails to
        # launch (or localm will not start at all). Opens the standalone reporter,
        # which needs no running server.
        ttk.Button(footer, text="Report a problem", style="Quiet.TButton",
                   command=self._report_problem).pack(side="right", padx=(0, 8))

        # Populate the model list last - on a fresh install with an empty
        # registry this shows a hint in the status label built just above.
        self._refresh_models()

    # ------------------------------------------------------------- #

    def _refresh_models(self, sync: bool = True) -> None:
        """Update the model dropdown. *sync* (the default - startup and the
        rescan button) reconciles the models folder first, off the UI thread,
        so a large registry never freezes the window; pass sync=False for a
        plain reload of what's already registered (e.g. right after an import
        that just wrote the registry itself - no folder scan needed to see it)."""
        if not sync:
            self._apply_models(load_models())
            return
        self._set_busy(True)
        self._start_ticker("Checking models folder")

        def work():
            result = sync_models_dir_safe()
            models = load_models()
            # The scan can outlive the window: closing the launcher while a
            # large models folder is being read destroys the widget this would
            # schedule onto, and Tk raises from a thread that is no longer the
            # one running its loop. Nothing is owed to a window that is gone.
            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._refresh_done(result, models))
            except (tk.TclError, RuntimeError):
                pass

        threading.Thread(target=work, daemon=True).start()

    def _apply_models(self, models: list) -> None:
        # The "no model" sentinel always leads the list so the Web GUI can be
        # launched with nothing loaded even when usable models are registered.
        # Deduplicates [NO_MODEL_LABEL] + models before it reaches the widget.
        seen = set()
        values = []
        for v in [NO_MODEL_LABEL] + models:
            if v not in seen:
                seen.add(v)
                values.append(v)
        self.model_box["values"] = values
        if self.model.get() not in values:
            self.model.set(models[0] if models else NO_MODEL_LABEL)

    def _refresh_done(self, result, models: list) -> None:
        self._stop_ticker()
        self._set_busy(False)
        self._apply_models(models)
        if result is not None and result.changed:
            bits = []
            if result.added:
                bits.append(f"{result.added} new")
            if result.flagged:
                bits.append(f"{result.flagged} missing")
            if result.restored:
                bits.append(f"{result.restored} restored")
            if result.pruned:
                bits.append(f"{result.pruned} pruned")
            if result.backfilled:
                bits.append(f"{result.backfilled} metadata backfilled")
            if bits:
                self.status_msg(f"Models folder synced: {', '.join(bits)}.")
                return
        if not models:
            self.status_msg("No models yet - Import one, or just launch the Web "
                            "GUI and add one on the Models page", error=False)
        else:
            plural = "" if len(models) == 1 else "s"
            self.status_msg(f"Up to date ({len(models)} model{plural}).")

    # ------------------------- progress ticker ---------------------- #

    def _start_ticker(self, label: str) -> None:
        """Begin a live, changing status line ('label... (Ns)') that keeps
        updating until _stop_ticker() - so a background scan or import never
        leaves the window showing a static, unchanging message."""
        self._ticker_label = label
        self._ticker_start = time.monotonic()
        self._ticker_frame = 0
        self._ticker_active = True
        self._tick()

    def _tick(self) -> None:
        if not self._ticker_active:
            return
        elapsed = int(time.monotonic() - self._ticker_start)
        # Dot count keys off the frame counter, not the whole elapsed second.
        dots = "." * ((self._ticker_frame % 3) + 1)
        self._ticker_frame += 1
        self.status_msg(f"{self._ticker_label}{dots} ({elapsed}s)")
        self.after(400, self._tick)

    def _stop_ticker(self) -> None:
        self._ticker_active = False

    # ------------------------- model import ----------------------- #

    def _set_busy(self, busy: bool) -> None:
        """Disable the controls that must not run mid-import or mid-scan: a
        multi-GB hash or a folder sync can take a while, and Launching (R48),
        generating a key (R49), or starting a second scan/import during it
        would race the registry write or stomp the live status line.
        Re-enabled when the work finishes."""
        state = "disabled" if busy else "normal"
        for b in (*self.import_btns, self.launch_btn, self.gen_btn, self.refresh_btn):
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
        if not path:
            return
        if is_models_root(path):
            # The data dir / models folder is not a model - sync it instead
            # of calling `localm add`, which only ever refuses this path.
            self._refresh_models(sync=True)
            return
        self._register_path(path)

    def _register_path(self, path: str) -> None:
        """Register a local file/dir via `localm add` off the UI thread.
        SHA256 hashing of a multi-GB file can take a few seconds."""
        before = set(load_models())
        self._set_busy(True)
        self._start_ticker("Importing")

        def work():
            ok, msg = False, ""
            try:
                proc = subprocess.run(
                    [python_exe(), "-m", "localm", "add", path,
                     "--on-duplicate", "alias", "--fast"],
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
        self._stop_ticker()
        self._set_busy(False)
        current = load_models()
        self._apply_models(current)
        new = sorted(set(current) - before)
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
        cmd = [launch_exe(), "-m", "localm", "gui", "--pull", spec]
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

    def _report_problem(self) -> None:
        """Open the standalone bug reporter in its own console. Works even when a
        mode just failed to launch or localm will not start - it needs no running
        server (it files an account-less GitHub issue via the localm proxy after
        showing you what will be sent)."""
        script = REPO_DIR / "scripts" / "report_issue.py"
        if not script.is_file():
            self.status_msg("Reporter not found (scripts/report_issue.py)", error=True)
            return
        try:
            _spawn_detached([python_exe(), str(script)], cwd=str(REPO_DIR))
            self.status_msg("Opened the problem reporter in a new window")
        except Exception as e:
            self.status_msg(f"Could not open the reporter: {e}", error=True)

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

        cmd = [launch_exe(), "-m", "localm"]
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
            if self.keep_diagnostics.get():
                cmd += ["--keep-diagnostics"]
        elif mode == "chat":
            cmd += ["run", model]
            if ctx:
                cmd += ["-c", ctx]
            if gpu:
                cmd += ["-g", gpu]
            if self.debug.get():
                cmd += ["--debug"]
        elif mode == "serve":
            cmd += ["gui", "--api-mode", "--no-browser"]
            if no_model:
                cmd += ["--no-model"]
            else:
                cmd += [model]
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
            if self.keep_diagnostics.get():
                cmd += ["--keep-diagnostics"]
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

    @staticmethod
    def _expose_key_plan(mode, host_lan, current_key, auth_mod):
        """Decide the API key for this launch. Exposing on the network requires
        one (a loopback-only launch stays open and needs none); if the field is
        empty we MINT one rather than making the user pre-set LOCALM_API_KEY - the
        launcher already persists + injects the key, so instructing the user to do
        it by hand was busywork. Returns ``(key, auto_generated, error)``:
        *auto_generated* is True when we minted one just now (the caller reveals it
        and keeps the window open so the user can copy it / pair a phone)."""
        key = (current_key or "").strip()
        exposing = mode in ("gui", "serve") and bool(host_lan)
        if exposing and not key:
            if auth_mod is None:
                return "", False, ("Cannot expose on the network: the auth module "
                                   "is unavailable to mint a key.")
            return auth_mod.generate_key(), True, None
        return key, False, None

    def _launch(self) -> None:
        cmd = self._build_command()
        if cmd is None:
            return

        env = os.environ.copy()
        if self.debug.get() and "--debug" not in cmd:
            env["LOCALM_DEBUG"] = "1"
        # Web GUI mode shows a tray + status window, so the console we open for it
        # is ours to hide once the server is up (it then runs like a background
        # app). Debug mode keeps the console visible so its live log stays readable.
        if self.mode.get() == "gui" and not self.debug.get():
            env["LOCALM_OWN_CONSOLE"] = "1"

        # --- authentication: persist the key + require flag, inject into env ---
        a = _auth()
        key, auto_keyed, err = self._expose_key_plan(
            self.mode.get(), self.host_lan.get(), self.api_key.get(), a)
        if err:
            self.status_msg(err, error=True)
            return
        if auto_keyed:
            # Surface the freshly minted key: reveal it in the field so the user
            # can copy it / pair a phone (Companion QR), and keep the window open
            # below even if "keep open" is off, so it is not lost behind a close.
            self.api_key.set(key)
            self.show_key.set(True)
            self._toggle_key()
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
            "keep_diagnostics": self.keep_diagnostics.get(),
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

        if auto_keyed:
            self.status_msg("Launched ✓ - minted a network API key (shown above); "
                            "clients need it (pair a phone via the Companion QR).")
        else:
            self.status_msg("Launched ✓")
        # A just-minted key must stay visible: keep the window open even if "keep
        # open" is off, so the user can copy the key / pair a phone.
        if not self.keep_open.get() and not auto_keyed:
            self.after(700, self.destroy)


if __name__ == "__main__":
    Launcher().mainloop()
