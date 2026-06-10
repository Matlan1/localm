import json
import socket
from pathlib import Path
from typing import Optional

HOME_DIR = Path.home() / ".localm"
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
    "n_ctx": 4096,
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 1024,
    "confirm_remove": True,   # ask before localm rm deletes files
    "port": 8642,             # default inference server port (auto-bumps if busy)
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
