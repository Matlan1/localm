# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ACE-Step 1.5 model files the native music backend loads, and pulling the
default set on first use.

Each component is configured as a localm registry model name or a file path,
and must carry that component's GGUF architecture. Unset components use the
smallest published set from ``DEFAULT_REPO``, pulled through ``localm pull`` (so
the download goes through the same network policy, verification and registry as
any other model) with its progress relayed. When a different model already holds
a default file's registry name, the default is pulled under ``DISTINCT_PREFIX``
plus that name.
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

# component -> the general.architecture its GGUF must carry.
ARCHITECTURES = {
    "lm": "acestep-lm",
    "text_encoder": "acestep-text-enc",
    "dit": "acestep-dit",
    "vae": "acestep-vae",
}

DISTINCT_PREFIX = "ace-step-"

COMPONENTS = ("text_encoder", "dit", "vae", "lm")

_OVERHEAD_WITH_LM = int(3.5 * 1024 ** 3)
_OVERHEAD_WITHOUT_LM = int(1.75 * 1024 ** 3)

Progress = Callable[[str], None]
CancelCheck = Callable[[], bool]


class ModelError(RuntimeError):
    """A configured model could not be found or is not that component, or the
    default could not be pulled."""


def _label(component: str) -> str:
    return component.replace("_", " ")


def default_names(component: str) -> list[str]:
    """Registry names the default file of *component* may be registered under:
    the name ``localm pull`` gives it, then the distinct name."""
    stem = Path(DEFAULT_FILES[component]).stem
    return [stem, DISTINCT_PREFIX + stem]


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


def architecture_of(path: Path) -> Optional[str]:
    from localm.model_manager import gguf_architecture
    try:
        return gguf_architecture(path)
    except Exception:  # noqa: BLE001
        return None


def find_default(component: str) -> Optional[Path]:
    """The downloaded default file of *component*, or None. A registry entry
    under a default name whose architecture is not this component's is skipped."""
    for name in default_names(component):
        p = resolve_path(name)
        if p is not None and architecture_of(p) == ARCHITECTURES[component]:
            return p
    return None


def default_pull(component: str) -> tuple[str, Optional[str]]:
    """``(spec, name)`` for pulling the default file of *component*: *name* is
    None when its usual registry name is free, else the distinct name."""
    spec = f"{DEFAULT_REPO}:{DEFAULT_FILES[component]}"
    plain = default_names(component)[0]
    return spec, (None if resolve_path(plain) is None else default_names(component)[1])


def _pull(spec: str, name: Optional[str], on_progress: Progress,
          cancel_check: CancelCheck) -> None:
    from localm.model_manager._shared import PROGRESS_SENTINEL
    env = dict(os.environ, LOCALM_PROGRESS_JSON="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    argv = [sys.executable, "-X", "utf8", "-m", "localm", "pull"]
    if name:
        argv += ["--name", name]
    proc = subprocess.Popen(
        argv + ["--", spec],
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

    A configured component must resolve to a file with that component's
    architecture, or :class:`ModelError` says why; it is never replaced by the
    default. An unset component uses the default, pulled first when
    *pull_missing* is true. The LM is skipped when *use_lm* is false."""
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
                    f"the configured music {_label(comp)} model '{configured}' "
                    "is not a file or a model in the library")
            arch = architecture_of(p)
            if arch != ARCHITECTURES[comp]:
                raise ModelError(
                    f"the configured music {_label(comp)} model '{configured}' is not an "
                    f"ACE-Step {_label(comp)} (its architecture is {arch or 'unreadable'}, "
                    f"expected {ARCHITECTURES[comp]})")
            paths[comp] = p
            continue
        p = find_default(comp)
        if p is None:
            if not pull_missing:
                raise ModelError(
                    f"the default music {_label(comp)} model is not downloaded; "
                    "run 'localm setup-music' first")
            spec, name = default_pull(comp)
            say(f"Downloading the default music {_label(comp)} model "
                f"({DEFAULT_FILES[comp]})...")
            _pull(spec, name, say, cancelled)
            p = find_default(comp)
            if p is None:
                raise ModelError(f"{spec} downloaded but is not in the model library as an "
                                 f"ACE-Step {_label(comp)}")
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
    return total + (_OVERHEAD_WITH_LM if models.lm else _OVERHEAD_WITHOUT_LM)
