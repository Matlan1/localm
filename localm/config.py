import json
import os
import socket
from pathlib import Path
from typing import Optional


def _detect_home() -> Path:
    """
    Resolve the localm data directory (config, registry, models, logs,
    sessions, generated images/music).

    Priority — each install is self-contained and agnostic of others:
      1. LOCALM_HOME env var (explicit override / custom install location)
      2. Portable mode: a checkout carrying its own data — when this package
         sits in a source tree (pyproject.toml next to it) that contains a
         ``localm-home.cfg`` (one line: the data path) or a ``home``
         directory, that wins.  Created by setup.bat's "keep data in this
         folder" option.
      3. Shared per-user default: ~/.localm
    """
    env = os.environ.get("LOCALM_HOME", "").strip()
    if env:
        return Path(env).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "pyproject.toml").is_file():       # source checkout only
        cfg_marker = repo_root / "localm-home.cfg"
        if cfg_marker.is_file():
            try:
                line = cfg_marker.read_text(encoding="utf-8").strip()
                if line:
                    return Path(line).expanduser()
            except OSError:
                pass
        portable = repo_root / "home"
        if portable.is_dir():
            return portable

    return Path.home() / ".localm"


def home_dir() -> Path:
    """Lazy variant of HOME_DIR — resolves the data dir at call time."""
    return _detect_home()


HOME_DIR = _detect_home()
MODELS_DIR = HOME_DIR / "models"
REGISTRY_FILE = HOME_DIR / "registry.json"
CONFIG_FILE = HOME_DIR / "config.json"

# Prefer the newer prebuilt, fall back to the locally compiled build
_BINARY_CANDIDATES = [
    Path(r"D:\projects\llama-gfx1030-prebuilt"),
    Path(r"D:\projects\llama.cpp\build\bin"),
]

DEFAULT_CONFIG: dict = {
    "binary_dir": None,    # None = auto-detect from _BINARY_CANDIDATES
    "n_ctx": 4096,         # initial context window (grows on demand)
    "n_ctx_max": 16384,    # ceiling the window may grow to (0 = unlimited)
    "n_ctx_grow": 4096,    # growth step — window expands in multiples of this
    "ctx_auto": False,     # True = derive n_ctx_max from free VRAM at load
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 1024,
    "confirm_remove": True,   # ask before localm rm deletes files
    "port": 8642,             # default inference server port (auto-bumps if busy)
    "cors_origins": None,     # None = localhost only; list of origins; or "*"
    # Command that starts your ComfyUI install (e.g. a launch .bat). When set,
    # image generation can start ComfyUI automatically if it is not running.
    "comfy_launch_cmd": None,
    # ComfyUI's own output directory (e.g. StabilityMatrix's Images folder).
    # When set, the duplicate ComfyUI keeps after generation is deleted so the
    # only copy is the one localm saved.
    "comfy_output_dir": None,
    # Session persistence mode for ALL surfaces (chat, server, GUI, coder):
    #   privacy = no traces written automatically (default)
    #   log     = JSONL audit trail in ~/.localm/sessions/
    #   full    = log + markdown transcript
    # chat_mode / coder_mode override the global mode per surface (None =
    # inherit). CLI --mode flags override everything.
    "mode": "privacy",
    "chat_mode": None,
    "coder_mode": None,
    # Seconds a GUI coder approval card may sit unanswered before it is
    # auto-rejected and the agent moves on.
    "coder_confirm_timeout": 600,
    # After an image is generated, ask ComfyUI to release its VRAM and reload
    # the chat model so the next reply is instant. Turn off when generating
    # many images in a row — the chat model then reloads lazily on the next
    # chat message instead.
    "reload_llm_after_imagine": True,
    # Network policy for model-initiated requests (coder fetch_url/web_search,
    # chat web access). See localm/netpolicy.py and docs/network.md.
    #   off   = all policy-routed network access fails fast
    #   ask   = allowed; the coder asks for approval per request (default)
    #   allow = no confirmation
    "net_mode": "ask",
    "net_allow": [],            # domains; empty = any. "x.com" covers *.x.com
    "net_deny": [],             # domains always refused (wins over allow)
    "net_allow_private": False, # True = permit loopback/private targets (SSRF guard off)
    "net_search_url": None,     # SearXNG base URL; None = DuckDuckGo (no key)
    # Speech-to-text (GUI mic button; needs the [voice] extra).
    # Model sizes: tiny / base / small / medium — bigger = better + slower.
    "voice_stt_model": "base",
    "voice_stt_language": None,  # None = auto-detect; or "en", "de", …
}

# localm claims 8642-8741 — far from ComfyUI (8188), A1111 (7860),
# Ollama (11434), and the 8000/8080/8888 dev-server crowd.
PORT_RANGE = (8642, 8741)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is already listening on host:port."""
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def pick_port(requested: Optional[int] = None, host: str = "127.0.0.1"):
    """
    Resolve the port to serve on.

    Returns (port, requested_port_was_busy). Tries the requested port (or the
    configured default), then walks the localm range for a free one, and as a
    last resort lets the OS assign any free port.
    """
    start = requested if requested is not None else load_config().get("port", PORT_RANGE[0])
    if not port_in_use(start, host):
        return start, False
    for candidate in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
        if candidate != start and not port_in_use(candidate, host):
            return candidate, True
    return get_free_port(), True


def ensure_dirs() -> None:
    HOME_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    ensure_dirs()
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: dict) -> None:
    ensure_dirs()
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_registry() -> dict:
    ensure_dirs()
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}


def save_registry(reg: dict) -> None:
    ensure_dirs()
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


def find_binary_dir() -> Optional[Path]:
    """Return the directory containing llama-server.exe, or None."""
    cfg = load_config()
    if cfg.get("binary_dir"):
        p = Path(cfg["binary_dir"])
        if (p / "llama-server.exe").exists():
            return p
    for candidate in _BINARY_CANDIDATES:
        if (candidate / "llama-server.exe").exists():
            return candidate
    return None


def get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
