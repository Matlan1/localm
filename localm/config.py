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
}


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
