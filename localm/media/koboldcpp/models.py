# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ACE-Step 1.5 model files the native music backend loads, and pulling the
default set on first use.

Each component is configured as a localm registry model name or a file path.
Unset components use the smallest published set from ``DEFAULT_REPO``, pulled
through ``localm pull`` (so the download goes through the same network policy,
verification and registry as any other model) with its progress relayed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from .server import Cancelled, ModelSet

DEFAULT_REPO = "Serveurperso/ACE-Step-1.5-GGUF"

# component -> default file in DEFAULT_REPO.
DEFAULT_FILES = {
    "lm": "acestep-5Hz-lm-0.6B-Q8_0.gguf",
    "text_encoder": "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "dit": "acestep-v15-turbo-Q8_0.gguf",
    "vae": "vae-BF16.gguf",
}

COMPONENTS = ("text_encoder", "dit", "vae", "lm")

Progress = Callable[[str], None]
CancelCheck = Callable[[], bool]


class ModelError(RuntimeError):
    """A configured model could not be found, or the default could not be pulled."""


def default_name(component: str) -> str:
    """The registry name ``localm pull`` gives the default file of *component*."""
    return Path(DEFAULT_FILES[component]).stem


def resolve_path(value: str) -> Optional[Path]:
    """The file for a configured *value*: an existing file path, else a registry
    model name. None when it is neither."""
    value = (value or "").strip()
    if not value:
        return None
    p = Path(value).expanduser()
    if p.is_file():
        return p
    from localm.model_manager import get_model_path
    found = get_model_path(value)
    if found is not None and Path(found).is_file():
        return Path(found)
    return None


def _pull(spec: str, on_progress: Progress, cancel_check: CancelCheck) -> None:
    from localm.model_manager._shared import PROGRESS_SENTINEL
    env = dict(os.environ, LOCALM_PROGRESS_JSON="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "localm", "pull", "--", spec],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env)
    tail: list = []
    last_pct = [-10.0]

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if PROGRESS_SENTINEL in line:
                try:
                    data = json.loads(line.partition(PROGRESS_SENTINEL)[2])
                except (ValueError, RecursionError):
                    continue
                pct = data.get("pct") if isinstance(data, dict) else None
                total = data.get("total") if isinstance(data, dict) else None
                if isinstance(pct, (int, float)) and pct >= last_pct[0] + 10:
                    last_pct[0] = pct
                    size = f" of {total / 1024 ** 3:.2f} GB" if total else ""
                    on_progress(f"Downloading {spec.rsplit(':', 1)[-1]}: {pct:.0f}%{size}")
                continue
            tail.append(line)
            del tail[:-6]

    reader = threading.Thread(target=_read, name="music-model-pull", daemon=True)
    reader.start()
    while proc.poll() is None:
        reader.join(0.25)
        if cancel_check():
            proc.kill()
            proc.wait()
            raise Cancelled()
    reader.join(5)
    if proc.returncode != 0:
        raise ModelError(f"could not download {spec}: " + (" / ".join(tail) or
                                                            f"exit code {proc.returncode}"))


def resolve_models(native_cfg: dict, *, use_lm: bool = True, pull_missing: bool = True,
                   on_progress: Optional[Progress] = None,
                   cancel_check: Optional[CancelCheck] = None) -> ModelSet:
    """The model set for *native_cfg* (``plugins.music.native``).

    A configured component must resolve, or :class:`ModelError` names it; it is
    never replaced by the default. An unset component uses the default, pulled
    first when *pull_missing* is true. The LM is skipped when *use_lm* is false."""
    say = on_progress or (lambda _m: None)
    cancelled = cancel_check or (lambda: False)
    paths: dict = {}
    for comp in COMPONENTS:
        if comp == "lm" and not use_lm:
            continue
        configured = str(native_cfg.get(comp) or "").strip()
        if configured:
            p = resolve_path(configured)
            if p is None:
                raise ModelError(
                    f"the configured music {comp.replace('_', ' ')} model '{configured}' "
                    "is not a file or a model in the library")
            paths[comp] = p
            continue
        p = resolve_path(default_name(comp))
        if p is None:
            if not pull_missing:
                raise ModelError(
                    f"the default music {comp.replace('_', ' ')} model is not downloaded; "
                    "run 'localm setup-music' first")
            spec = f"{DEFAULT_REPO}:{DEFAULT_FILES[comp]}"
            say(f"Downloading the default music {comp.replace('_', ' ')} model "
                f"({DEFAULT_FILES[comp]})...")
            _pull(spec, say, cancelled)
            p = resolve_path(default_name(comp))
            if p is None:
                raise ModelError(f"{spec} downloaded but is not in the model library")
        paths[comp] = p
    return ModelSet(text_encoder=str(paths["text_encoder"]), dit=str(paths["dit"]),
                    vae=str(paths["vae"]),
                    lm=str(paths["lm"]) if "lm" in paths else None)


def estimate_bytes(models: ModelSet) -> int:
    """Approximate VRAM the model set needs: the files plus the planner's KV
    cache and working buffers."""
    total = 0
    for p in (models.text_encoder, models.dit, models.vae, models.lm):
        if p:
            try:
                total += Path(p).stat().st_size
            except OSError:
                pass
    overhead = 2 * 1024 ** 3 if models.lm else 1024 ** 3
    return total + overhead
