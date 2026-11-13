# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, TypeVar, cast

from localm import _home_cfg, mgmt_events
from localm.storekit import (  # noqa: F401  (re-exported: callers use config.<name>)
    READ_ABSENT,
    READ_CURRENT,
    READ_FROM_BACKUP,
    READ_UNREADABLE,
    ConfigLockLost,
    ConfigLockTimeout,
    ConfigUnreadable,
    assert_store_lock_held,
    atomic_write_private,
    restrict_file_perms,
)
from localm.storekit import read_json as _read_json
from localm.storekit import read_json_checked as _read_json_checked
from localm.storekit import read_json_state as _read_json_state
from localm.storekit import (_REPLACE_RETRIES, _is_transient_permission_error,
                             _transient_backoff)
from localm.storekit import replace_atomic as _replace_atomic
from localm.storekit import store_lock as _cross_process_lock


_warned_unconfigured_home = False
_warned_unreadable_home_cfg: set = set()
_unusable_temp_dirs: set = set()

# config files already warned about (present but not a JSON object), so a
# non-dict config.json is surfaced once per process, not on every load_config
# call (see _merge_stored_config).
_warned_bad_config: set = set()
_warned_stray_credentials: set = set()

# The three states of gui_proxy_remote_images, least to most permissive.
#   off  the route refuses; a reply's remote image never leaves this machine.
#   ask  the route refuses until the GUI has asked the reader about that
#        ORIGIN, and re-asks it with the reader's answer. The only state that
#        closes the channel for an arbitrary host while still showing images.
#   on   fetch it, no question.
REMOTE_IMAGE_OFF = "off"
REMOTE_IMAGE_ASK = "ask"
REMOTE_IMAGE_ON = "on"
REMOTE_IMAGE_MODES = (REMOTE_IMAGE_OFF, REMOTE_IMAGE_ASK, REMOTE_IMAGE_ON)
# config.json can still hold a bare boolean for this key: load_config() is
# defaults + the stored delta, with no migration step anywhere in this file.
# (false_option, true_option) is what such a value maps to, and it is the ONE
# definition of that mapping - settings_schema's SettingField.legacy_bool reads
# the same pair, so what the file can hold and what PATCH /v1/config accepts
# cannot drift apart.
REMOTE_IMAGE_LEGACY_BOOL = (REMOTE_IMAGE_OFF, REMOTE_IMAGE_ON)

# gui_proxy_remote_images values already warned about, so an unreadable one is
# surfaced once per process rather than on every load_config (see
# _normalize_remote_image_mode).
_warned_bad_remote_image_mode: set = set()

# keep_diagnostics values already reported as unrecognised, once per process.
_warned_bad_keep_diagnostics: set = set()

_KEEP_DIAGNOSTICS_ON = ("1", "true", "on", "yes")


def coerce_remote_image_mode(val):
    """*val* as one of REMOTE_IMAGE_MODES, or None if it is not a shape this
    key has ever legitimately held.

    Accepts the current strings and the pre-3-state boolean (including the
    `1`/`0` and `"true"`/`"false"` spellings a JSON or CLI client may send),
    per REMOTE_IMAGE_LEGACY_BOOL. Returning None rather than a default keeps
    "this is unreadable" distinguishable from "this says off", which is what
    lets the caller decide whether to warn."""
    false_opt, true_opt = REMOTE_IMAGE_LEGACY_BOOL
    if isinstance(val, bool):
        return true_opt if val else false_opt
    if isinstance(val, int) and val in (0, 1):
        return true_opt if val else false_opt
    if isinstance(val, str):
        low = val.strip().lower()
        # An exact mode wins first: "off"/"on" are also boolean spellings, and
        # both readings agree, but the mode is what the value IS.
        if low in REMOTE_IMAGE_MODES:
            return low
        if low in ("1", "true", "yes"):
            return true_opt
        if low in ("0", "false", "no"):
            return false_opt
    return None


def _normalize_remote_image_mode(cfg: dict) -> None:
    """Canonicalise gui_proxy_remote_images in *cfg*, in place.

    Runs on every load so a config.json written before the key became a
    three-state setting reads as "on"/"off" everywhere - the route, the settings
    schema, the bug report and `localm config` all go through load_config(), so
    normalising here is the whole migration. save_config() writes the user delta
    from the loaded dict, so the file heals itself on the next save.

    An UNREADABLE value falls back to "off" and says so once. Failing closed is
    the safe direction for a key that decides whether rendering a reply makes an
    outbound request, and saying so is the difference between a fallback and a
    hidden problem: without the warning a typo in config.json would silently
    turn a feature the owner had switched on back off."""
    raw = cfg.get("gui_proxy_remote_images")
    mode = coerce_remote_image_mode(raw)
    if mode is not None:
        cfg["gui_proxy_remote_images"] = mode
        return
    cfg["gui_proxy_remote_images"] = REMOTE_IMAGE_OFF
    token = repr(raw)
    if token not in _warned_bad_remote_image_mode:
        _warned_bad_remote_image_mode.add(token)
        print(f"[localm] config.json has gui_proxy_remote_images={token}, which "
              f"is not one of {list(REMOTE_IMAGE_MODES)}; treating it as "
              f"'{REMOTE_IMAGE_OFF}' (remote images in replies stay blocked).",
              file=sys.stderr)


def _warn_unconfigured_home(path: Path) -> None:
    """Warn once that no data dir was configured. stderr, not the logger: this
    runs at import time before logging is wired."""
    # setup.bat / setup.sh run setup-llama BEFORE the data-dir is chosen and set
    # LOCALM_SETUP=1 to suppress this one warning for that phase. It is the ONLY
    # suppressor, and it must never be set outside the setup scripts: left in a
    # shell environment it masks a real lost-config warning at runtime.
    if os.environ.get("LOCALM_SETUP") == "1":
        return
    global _warned_unconfigured_home
    if _warned_unconfigured_home:
        return
    _warned_unconfigured_home = True
    print("[localm] WARNING: no data directory is configured (no LOCALM_HOME, no "
          "localm-home.cfg, no ./home). Setup may not have run, or the config was "
          f"lost. Using a contained default inside this install: {path}. Run setup "
          "or set LOCALM_HOME to choose where data lives.", file=sys.stderr)


def _detect_home() -> Path:
    """
    Resolve the localm data directory (config, registry, models, logs,
    sessions, generated images/music).

    Priority - each install is self-contained and agnostic of others:
      1. LOCALM_HOME env var (explicit override / custom install location)
      2. Portable mode: a checkout carrying its own data - when this package
         sits in a source tree (pyproject.toml next to it) that contains a
         ``localm-home.cfg`` (one line: the data path) or a ``home``
         directory, that wins.  Created by setup.bat's "keep data in this
         folder" option. A relative path in ``localm-home.cfg`` is relative
         to the checkout; a relative LOCALM_HOME is made absolute against the
         working folder at the first read and stored back into the
         environment.
      3. Contained fallback: ``./home`` INSIDE the install, with a stderr
         warning, when NOTHING is configured. Reaching here is a special case
         (setup never ran, or the config/marker was lost), not a normal default -
         so it is surfaced, and it NEVER silently falls back to a shared
         ~/.localm outside the install.
    """
    env = os.environ.get("LOCALM_HOME", "").strip()
    if env:
        home = Path(env).expanduser()
        if not home.is_absolute():
            # A relative LOCALM_HOME is made absolute against the working folder
            # at the first read and written back to the environment.
            home = home.absolute()
            os.environ["LOCALM_HOME"] = str(home)
        return home

    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "pyproject.toml").is_file():       # source checkout only
        cfg_marker = repo_root / _home_cfg.NAME
        if cfg_marker.is_file():
            try:
                line = _home_cfg.read_line(cfg_marker)
                if line:
                    return _home_cfg.data_dir(line, repo_root)
            except (OSError, ValueError, LookupError) as e:
                # The marker EXISTS but could not be read: warn and fall back to
                # the default data dir rather than silently switching. stderr
                # rather than the logger - this runs at import time, before
                # logging is wired, and importing debuglog here would be
                # circular.
                warning = (f"[localm] WARNING: cannot read {cfg_marker} ({e}); "
                           "falling back to the default data dir.")
                if warning not in _warned_unreadable_home_cfg:
                    _warned_unreadable_home_cfg.add(warning)
                    print(warning, file=sys.stderr)
        portable = repo_root / "home"
        if portable.is_dir():
            return portable

    # No data dir configured: no LOCALM_HOME, no localm-home.cfg, no ./home -
    # setup was never run, or the config/marker was lost. Default to the
    # instance's OWN root (a contained ./home) and SAY SO on stderr, never a
    # silent shared ~/.localm outside the install.
    fallback = repo_root / "home"
    _warn_unconfigured_home(fallback)
    return fallback


def home_dir() -> Path:
    """Lazy variant of HOME_DIR - resolves the data dir at call time."""
    return _detect_home()


def cache_dir() -> Path:
    """Root for caches localm's OWN subprocesses write, inside the data dir.

    Anything localm downloads on the way to installing something - pip's wheel/http
    cache while provisioning the managed ComfyUI venv, the Whisper STT model - lands
    here rather than in the user's home profile, where those tools cache when left
    to their defaults (under ``%LOCALAPPDATA%`` on Windows, ``~/.cache`` on POSIX).

    Derived from ``home_dir()``, never a hardcoded path, so the cache follows
    LOCALM_HOME. NOT conditional on an ambient ``PIP_CACHE_DIR``: LOCALM_HOME is
    the only knob that moves it. The Hugging Face caches under it are the one
    exception: ``HF_HOME`` / ``HF_HUB_CACHE`` / ``HF_XET_CACHE`` set by the user
    are honoured (see ``contained_hf_env``)."""
    return home_dir() / "cache"


def pip_cache_dir() -> Path:
    """localm's OWN pip cache, inside the data dir.

    Shared by every pip subprocess localm drives itself - plugin-extra installs
    (plugins/deps.py), the native runtime wheel (setup_llama/), and managed-ComfyUI
    provisioning (media/managed_comfy_provision.py delegates here) - so wheels are
    cached once, contained, and removed with the data dir. Left unset, pip caches to a
    per-user location OUTSIDE the data dir (``%LOCALAPPDATA%\\pip\\cache`` on Windows,
    ``~/.cache/pip`` on POSIX). A cache rather than ``--no-cache-dir``, so a re-run
    does not re-download multi-GB wheels; the disk cost lands inside the data dir and
    is reclaimed with it."""
    return cache_dir() / "pip"


def uv_cache_dir() -> Path:
    """localm's OWN uv cache, inside the data dir.

    uv keeps a SEPARATE cache from pip (its own ``UV_CACHE_DIR``, default
    ``%LOCALAPPDATA%\\uv\\cache`` on Windows / ``~/.cache/uv`` on POSIX). localm's
    installers try ``uv pip install`` BEFORE falling back to ``python -m pip``, so
    uv's cache is pinned here too. Same location root as pip's (``cache_dir()``), a
    sibling subdir since the two tools' cache formats differ."""
    return cache_dir() / "uv"


def contained_pip_env(base: Optional[dict] = None) -> dict:
    """A subprocess environment with pip's AND uv's caches pinned inside the data dir.

    *base* defaults to a copy of the current process environment. localm's package
    installers (plugins/deps.py, setup_llama/) shell out to ``uv pip install`` first
    and ``python -m pip install`` second; BOTH tools cache to a per-user location
    outside the data dir when left to their defaults, so BOTH ``PIP_CACHE_DIR`` and
    ``UV_CACHE_DIR`` are set here - pinning only one still leaks via the other. Both
    OVERRIDE any ambient value (see ``cache_dir()``). Callers pass the returned dict
    as ``subprocess``'s ``env=``; a subprocess that runs neither tool simply ignores
    the two extra vars."""
    env = dict(os.environ if base is None else base)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir())
    env["UV_CACHE_DIR"] = str(uv_cache_dir())
    return env


# Env var recording the HF_HOME value localm itself pinned.
_HF_PIN_MARKER = "LOCALM_PINNED_HF_HOME"
_HF_USER_PLACEMENT_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
                           "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE")


def hf_home_dir() -> Path:
    """localm's OWN Hugging Face home, inside the data dir.

    huggingface_hub and its xet transfer layer write under ``HF_HOME`` (default
    ``~/.cache/huggingface``, or ``$XDG_CACHE_HOME/huggingface``): xet logs and
    staging, the chunk cache, hub cache refs, update-check stamps. Left unset
    that is a per-user location OUTSIDE the data dir that "delete saved data"
    never reaches. Sibling of the pip / uv / whisper caches under ``cache_dir()``."""
    return cache_dir() / "huggingface"


def _default_hf_token_path(env: dict) -> str:
    """Where huggingface_hub looks for a saved login when ``HF_HOME`` is not set,
    for the ``XDG_CACHE_HOME`` in *env* and this process's home directory."""
    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    hf_home = os.path.expandvars(os.path.expanduser(
        os.path.join(env.get("XDG_CACHE_HOME", default_home), "huggingface")))
    return os.path.join(hf_home, "token")


def hf_cache_user_placement(base: Optional[dict] = None) -> dict:
    """The Hugging Face cache locations the USER chose, ``{var: value}``.

    ``HF_HUB_CACHE``, ``HF_XET_CACHE`` and the asset / legacy cache variables always
    count when set; ``HF_HOME``
    counts unless it is the value localm itself pinned. Empty when localm's
    own contained home is in effect."""
    env = os.environ if base is None else base
    out = {}
    for var in _HF_USER_PLACEMENT_VARS:
        val = env.get(var)
        if val is not None and not (var == "HF_HOME" and val == env.get(_HF_PIN_MARKER)):
            out[var] = val
    return out


def contained_hf_env(base: Optional[dict] = None) -> dict:
    """An environment with ``HF_HOME`` pinned inside the data dir.

    *base* defaults to a copy of the current process environment. ``HF_HOME``
    moves the xet cache and logs, the hub cache and every other Hugging Face
    cache that localm's downloads write; ``HF_HUB_CACHE`` / ``HF_XET_CACHE``
    follow it unless set separately. A location the user set (``HF_HOME`` other
    than localm's own earlier pin) is left exactly as set, as are an explicit
    ``HF_HUB_CACHE`` / ``HF_XET_CACHE``.

    When localm pins ``HF_HOME``, ``HF_TOKEN_PATH`` is pinned to where the saved
    Hugging Face login lives by default (unless the user set it), so a
    ``huggingface-cli login`` made outside localm keeps authenticating
    downloads. The login is read from there and never moved or copied."""
    env = dict(os.environ if base is None else base)
    pinned = env.get(_HF_PIN_MARKER)
    if env.get("HF_HOME") not in (None, pinned):
        return env
    if "HF_TOKEN_PATH" not in env:
        env["HF_TOKEN_PATH"] = _default_hf_token_path(env)
    home = os.path.abspath(str(hf_home_dir()))
    env["HF_HOME"] = home
    env[_HF_PIN_MARKER] = home
    return env


def contain_hf_cache() -> None:
    """Apply ``contained_hf_env`` to THIS process, so in-process Hugging Face use
    and every child process inherit it.

    huggingface_hub reads ``HF_HOME`` once, when it is first imported, so this
    runs when ``localm.config`` is imported, ahead of any localm import of it."""
    env = contained_hf_env()
    for key in ("HF_HOME", "HF_TOKEN_PATH", _HF_PIN_MARKER):
        if key in env and os.environ.get(key) != env[key]:
            os.environ[key] = env[key]
    pinned = os.environ.get(_HF_PIN_MARKER)
    if pinned and os.environ.get("HF_HOME") == pinned:
        loaded = getattr(sys.modules.get("huggingface_hub.constants"), "HF_HOME", None)
        if loaded is not None and (os.path.normcase(os.path.normpath(loaded))
                                   != os.path.normcase(os.path.normpath(pinned))):
            print(f"[localm] WARNING: huggingface_hub was imported before localm, so "
                  f"its caches stay at {loaded} instead of the data folder ({pinned}).",
                  file=sys.stderr)


# GPU driver caches that default to the user profile, as (env var, subfolder of
# cache_dir()). Each is pinned only when the user has not set the variable.
_GPU_CACHE_PINS = (
    ("AMD_COMGR_CACHE_DIR", "comgr"),
    ("MIOPEN_USER_DB_PATH", "miopen/db"),
    ("MIOPEN_CUSTOM_CACHE_DIR", "miopen/cache"),
    ("MESA_SHADER_CACHE_DIR", "mesa"),
    ("__GL_SHADER_DISK_CACHE_PATH", "glcache"),
    ("CUDA_CACHE_PATH", "cuda"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
)


def _gpu_pin_marker(var: str) -> str:
    return "LOCALM_PINNED_" + var


def gpu_cache_dirs() -> dict:
    """localm's OWN GPU driver caches, ``{env var: path}``, inside the data dir.

    comgr (the HIP runtime's kernel compiler) keeps its LLVM object cache under
    ``%LOCALAPPDATA%\\comgr`` and MIOpen (torch convolutions) keeps its kernel cache and
    tuning database under ``~/.miopen`` when these variables are unset. Mesa
    (``MESA_SHADER_CACHE_DIR``) and the NVIDIA Linux driver
    (``__GL_SHADER_DISK_CACHE_PATH``) keep compiled OpenGL shaders under ``~/.cache``,
    and the NVIDIA CUDA driver (``CUDA_CACHE_PATH``) keeps JIT-compiled kernels under
    ``~/.nv/ComputeCache`` (``%APPDATA%\\NVIDIA\\ComputeCache`` on Windows).
    PyTorch (``TORCH_EXTENSIONS_DIR``) keeps the C++/CUDA extensions it compiles on
    first import (ExLlama's, for one) under the user's cache folder.
    Those are per-user locations OUTSIDE the data dir that "delete saved data" never
    reaches. Siblings of the pip / uv / whisper / Hugging Face caches under
    ``cache_dir()``."""
    return {var: cache_dir().joinpath(*sub.split("/")) for var, sub in _GPU_CACHE_PINS}


def contained_gpu_cache_env(base: Optional[dict] = None) -> dict:
    """An environment with the GPU driver cache variables pinned inside the data dir.

    *base* defaults to a copy of the current process environment. A variable the user
    set (other than localm's own earlier pin of it) is left exactly as set;
    ``AMD_COMGR_CACHE`` (the on/off switch) is never touched."""
    env = dict(os.environ if base is None else base)
    for var, path in gpu_cache_dirs().items():
        marker = _gpu_pin_marker(var)
        if env.get(var) not in (None, env.get(marker)):
            continue
        value = os.path.abspath(str(path))
        env[var] = value
        env[marker] = value
    return env


def contain_gpu_caches() -> None:
    """Apply ``contained_gpu_cache_env`` to THIS process, so in-process HIP use (the
    llama.cpp ROCm runtime) and every child process (the torch ROCm worker) inherit it."""
    env = contained_gpu_cache_env()
    for var, _ in _GPU_CACHE_PINS:
        for key in (var, _gpu_pin_marker(var)):
            if key in env and os.environ.get(key) != env[key]:
                os.environ[key] = env[key]


def user_command_env(base: Optional[dict] = None) -> dict:
    """*base* (default: a copy of the process environment) without the cache
    locations localm pinned for itself, for a command the USER asked to run.

    Drops ``HF_HOME`` and ``HF_TOKEN_PATH`` when localm set them (``HF_HOME``
    equal to the pin marker, ``HF_TOKEN_PATH`` equal to the default login
    location localm pinned it to), each GPU cache variable equal to its pin
    marker, and the markers. A value the user set is left exactly as set, so a
    command sees the caches and login it would see outside localm."""
    env = dict(os.environ if base is None else base)
    hf_pin = env.pop(_HF_PIN_MARKER, None)
    if hf_pin is not None and env.get("HF_HOME") == hf_pin:
        del env["HF_HOME"]
        if env.get("HF_TOKEN_PATH") == _default_hf_token_path(env):
            del env["HF_TOKEN_PATH"]
    for var, _ in _GPU_CACHE_PINS:
        pin = env.pop(_gpu_pin_marker(var), None)
        if pin is not None and env.get(var) == pin:
            del env[var]
    return env


TEMP_DIR_ENV = "LOCALM_TMPDIR"
TEMP_LOCATION_KEY = "temp_location"


def _is_portable_home(home: Path) -> bool:
    """Whether *home* is the data folder of a self-contained checkout: inside a
    source checkout (``pyproject.toml`` beside the package) and under it."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        return False
    try:
        Path(home).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _stored_temp_location(home: Path) -> str:
    """The ``temp_location`` value in ``<home>/config.json``, or ``""`` when the
    file or the key is absent or unreadable. Read straight from the file: this
    runs while ``localm.config`` is still being imported."""
    try:
        data = json.loads((Path(home) / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return ""
    value = data.get(TEMP_LOCATION_KEY) if isinstance(data, dict) else None
    return value.strip() if isinstance(value, str) else ""


def _absolute_folder(text: str) -> Optional[Path]:
    """*text* as an absolute folder path (``~`` expanded), or None when it is
    relative or its ``~user`` cannot be resolved."""
    try:
        candidate = Path(text).expanduser()
    except (RuntimeError, OSError):
        return None
    return candidate if candidate.is_absolute() else None


def _is_unc_or_device(text: str) -> bool:
    """True for a UNC or device-namespace path: a leading two backslashes on any
    host, or ``//`` and ``\\??\\`` on Windows. Lexical only, no filesystem
    call."""
    if text.startswith("\\\\"):
        return True
    return os.name == "nt" and (text.startswith("//") or text.startswith("\\??\\"))


def temp_dir(home: Optional[Path] = None, warn: bool = True) -> Optional[Path]:
    """Where localm's temporary files go, or None to leave the system temp folder.

    In order: the ``LOCALM_TMPDIR`` environment variable; the ``temp_location``
    setting (``data`` = ``<data dir>/tmp``, ``system`` = the system temp folder, or
    an absolute folder path); then ``auto`` (the default), which keeps temp files
    in ``<data dir>/tmp`` for a self-contained checkout and leaves the system temp
    folder for any other install. A ``temp_location`` that is neither of those words
    nor an absolute path is reported on stderr (unless *warn* is False) and treated as
    ``auto``."""
    home = Path(home) if home is not None else home_dir()
    override = os.environ.get(TEMP_DIR_ENV, "").strip()
    if override:
        folder = _absolute_folder(override)
        if folder is not None:
            return folder
        if warn:
            print(f"[localm] WARNING: {TEMP_DIR_ENV}={override!r} is not an absolute "
                  "folder; ignoring it.", file=sys.stderr)
    choice = _stored_temp_location(home)
    if choice.lower() == "data":
        return home / "tmp"
    if choice.lower() == "system":
        return None
    if choice and choice.lower() != "auto":
        folder = None if _is_unc_or_device(choice) else _absolute_folder(choice)
        if folder is not None:
            return folder
        if warn:
            print(f"[localm] WARNING: temp_location {choice!r} in config.json is not "
                  "'auto', 'data', 'system' or a local absolute folder; using 'auto'.",
                  file=sys.stderr)
    return home / "tmp" if _is_portable_home(home) else None


def contain_temp_dir() -> None:
    """Point ``tempfile`` and TMP/TEMP/TMPDIR of THIS process at :func:`temp_dir`,
    so in-process temp use and every child process inherit it. Runs when
    ``localm.config`` is imported; does nothing when :func:`temp_dir` is None. When
    the directory cannot be created, says so on stderr and leaves the temp location
    as it was."""
    target = temp_dir()
    if target is None:
        return
    path = os.path.abspath(str(target))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        _unusable_temp_dirs.add(path)
        print(f"[localm] WARNING: cannot create the temp folder {path} ({e}); "
              "temporary files will go to the system temp folder instead.",
              file=sys.stderr)
        return
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = path
    tempfile.tempdir = path


def temp_dir_in_use(home: Optional[Path] = None) -> Optional[Path]:
    """:func:`temp_dir`, or None when that folder could not be created at start-up
    (:func:`contain_temp_dir` then left the system temp folder in use)."""
    target = temp_dir(home, warn=False)
    if target is not None and os.path.abspath(str(target)) in _unusable_temp_dirs:
        return None
    return target


HOME_DIR = _detect_home()
MODELS_DIR = HOME_DIR / "models"
REGISTRY_FILE = HOME_DIR / "registry.json"
CONFIG_FILE = HOME_DIR / "config.json"
contain_hf_cache()
contain_gpu_caches()
contain_temp_dir()


# Only the keys the user actually changed are persisted to config.json; a key
# absent on disk follows this table at load time, so editing a default value
# here reaches every existing install (see save_config / _user_delta).
DEFAULT_CONFIG: dict = {
    "binary_dir": None,    # None = auto-detect (runtime wheel, then legacy dirs)
    # Which llama.cpp release `setup-llama` provisions. Empty = the build localm
    # ships and confirmed (setup_llama._PINNED_TAG); the literal "latest" opts in
    # to whatever upstream published most recently, which localm has not tested;
    # a tag such as "b10355" pins that exact build until the user changes it.
    # Read by setup_llama._tag_for()/tracks_latest(), so `localm update`'s
    # re-provision inherits the choice.
    "llama_runtime_pin": "",
    # Append-only log of the runtime builds actually provisioned on this box,
    # newest LAST: [{"backend": ..., "tag": ..., "at": <epoch>}, ...]. It is what
    # makes `setup-llama --rollback` possible without the user having to remember
    # a tag number. Capped (see setup_llama._RUNTIME_HISTORY_MAX); state rather
    # than preference, but kept here because LOCALM_HOME outlives the venv the
    # runtime itself is installed into.
    "llama_runtime_history": [],
    "n_ctx": 4096,         # initial context window (grows on demand)
    "n_ctx_max": 16384,    # ceiling the window may grow to (0 = unlimited)
    "n_ctx_grow": 4096,    # growth step - window expands in multiples of this
    # Context window every load keeps usable (tokens); a Mixture-of-Experts
    # model keeps more experts in system RAM to hold it on the GPU.
    "ctx_reserve": 16384,
    # Size ctx ceiling from free VRAM/RAM at load (clamped 4k-64k); window still
    # starts at n_ctx and grows on demand. False = use fixed n_ctx_max.
    "ctx_auto": True,
    "n_gpu_layers": 99,    # 99 = offload everything to GPU
    # MoE expert placement: keep the EXPERT weights of the first N layers in
    # system RAM while the rest of the model follows the normal layer
    # assignment (llama.cpp's --n-cpu-moe). 0 = automatic: with auto-sized GPU
    # layers, only as many layers as the model needs to fit, before a whole
    # layer moves to the CPU. Only affects Mixture-of-Experts models; a dense
    # model has no expert tensors to move, so any value is a no-op there.
    "n_cpu_moe": 0,
    # Whether a model switch unloads other models to make room: "always"
    # (this instance's idle models, without asking), "ask" or "never" (load
    # beside them with the VRAM that is left).
    "unload_for_room": "always",
    "moe_cache_mib": 0,    # MB of VRAM caching MoE experts kept in system RAM; 0 = off
    # Memory-map the model file: "auto" (the loader decides), "on" (only the
    # pages a run touches are resident, so a model that may not fit in
    # available RAM can run from disk-backed memory, first tokens slower) or
    # "off" (the whole model is read into memory first).
    "use_mmap": "auto",
    # Element type of the KV cache: "f16" (llama.cpp's default) or "q8_0"
    # (1.0625 bytes per element instead of 2, so about 1.88x the context fits
    # in the same VRAM; needs flash attention). Applies at the next model load.
    "kv_cache_type": "f16",
    # When n_gpu_layers is left at its "everything" default, auto-size how many
    # layers actually go on the GPU from free VRAM at load: a model too big for
    # full GPU offload runs some layers on CPU (slower) and LOADS, instead of
    # being refused. An explicit n_gpu_layers (any value other than 99, e.g.
    # `-g 24`) is always honoured verbatim. Off => request full offload as-is.
    "n_gpu_layers_auto": True,
    # Enable Multi-Token Prediction (MTP) speculative decoding for models with
    # native MTP/next-n prediction heads (e.g. DeepSeek-V3/R1, Qwen MTP models).
    # True = active when model supports it; False = force standard autoregressive.
    # Off by default: speculation only pays when verifying two tokens costs about
    # what verifying one costs, which holds while the whole model is resident on
    # the GPU and stops holding once layers run on the CPU. `localm bench-mtp`
    # measures it per model rather than predicting it from the model's size.
    # See test_mtp_default_is_off_until_speculation_is_measured_to_pay.
    "mtp_enabled": False,
    # Draft tokens one MTP speculation step proposes (1-3). Each step costs one
    # small draft decode per token plus one verification decode of all of them.
    "mtp_draft_tokens": 1,
    # Where speculative drafts come from: "off", "mtp" (the model's own MTP
    # head), "ngram" (tokens that followed the same few tokens earlier in the
    # conversation; no second model, no extra VRAM) or "draft" (a smaller
    # model sharing the vocabulary, spec_draft_model). None follows
    # mtp_enabled: true means "mtp", false means "off". `localm bench-spec`
    # measures it.
    "spec_source": None,
    # Draft tokens one ngram or draft step may propose (1-16). None uses the
    # default (8; a draft model picks shorter lengths from its measured step
    # costs); a model with recurrent layers is
    # capped at 4.
    "spec_draft_tokens": None,
    # The draft model for spec_source "draft": a registered model name or a
    # GGUF path. `localm spec-drafts MODEL` lists downloaded models that fit.
    "spec_draft_model": "",
    # How many chat requests one loaded GGUF model answers at the same time,
    # decoding them together in one batch per step: "auto" or 1-16. Every
    # request shares the one context window. auto uses 4, fewer when the extra
    # recurrent state of a hybrid model would not fit in VRAM, and 1 when a
    # draft source is on (spec_source / mtp_enabled). 1 answers one at a time.
    "parallel_slots": "auto",
    # Diffusion language models (Dream, LLaDA, LLaDA-MoE, RND1) write a whole
    # reply at once over a number of denoising steps. diffusion_steps: steps
    # per reply, None for 128 (or the reply length when shorter).
    # diffusion_max_tokens: the reply length in tokens; a request's max_tokens
    # caps it further.
    "diffusion_steps": None,
    "diffusion_max_tokens": 256,
    # VRAM (MB) that n_gpu_layers_auto/ctx_auto/_check_vram always reserve beyond
    # model weights for the GGUF backend, before deciding how many layers fit or
    # refusing outright. This is NOT a discardable safety margin - it funds the
    # KV cache's compute buffers and llama.cpp's graph/scratch allocations, real
    # memory the native loader actually needs. Lowering it risks a native crash
    # or a GPU driver hang (TDR) instead of a slower but working load; only
    # lower it if you have confirmed your model/context combination needs less.
    "vram_overhead_mb": 1500,
    # Ceiling (seconds) for a GGUF model load in its isolated worker process
    # (see llamacpp/_runner.py) before it is treated as hung and killed. A
    # stalled load has no safe "unmeasurable" fallback (unlike a VRAM probe),
    # so raise this only if a genuinely huge model on slow storage needs
    # longer than the generous default.
    "gguf_load_timeout_s": 900.0,
    # How long a reply may take to produce its FIRST token before the worker is
    # treated as hung. This covers prompt PREFILL, not one token's decode, so it
    # is sized like the load timeout rather than the per-token ceiling: on CPU,
    # under heavy partial offload, or with a very long prompt, prefill can
    # legitimately take minutes. Raise this only if a genuinely slow box needs
    # longer than the generous default.
    "gguf_first_token_timeout_s": 900.0,
    # Ceiling (seconds) for an HF (HuggingFace-transformers) model load in its
    # isolated worker process (see backends/_hf_runner.py) before it is
    # treated as hung and killed. Same rationale as gguf_load_timeout_s: a
    # stalled load has no safe "unmeasurable" fallback. HF loads read
    # full-precision safetensors from disk (no quantized mmap fast path), so
    # this can legitimately need longer than a GGUF load of similar size.
    "hf_load_timeout_s": 900.0,
    # How long an HF reply may take to produce its first token before the
    # worker is treated as hung. Same rationale as gguf_first_token_timeout_s
    # (covers prompt prefill, not one token's decode). HF's CPU path is
    # always dense full-precision (never quantized the way GGUF's ggml
    # kernels are), so this is unvalidated for very slow CPU-only HF
    # inference - raise it if a genuinely slow box needs longer.
    "hf_first_token_timeout_s": 900.0,
    # Bounded wait for one HF embed() RPC. Separate from and more generous
    # than the dedicated GGUF-based embedder's own timeout: HFBackend.embed()
    # loops over texts one at a time with no batching, and an HF load never
    # quantizes - so a large batch against a full-precision CPU-fallback
    # model can plausibly run far longer than the dedicated small-model
    # embedder ever needs to. hf_embed_max_texts/hf_embed_max_chars below
    # reject an oversized request outright; this timeout bounds whatever is
    # allowed through.
    "hf_embed_timeout_s": 600.0,
    # Per-request caps on an HF-backed /v1/embeddings request, enforced by
    # HFBackend.embed() before a batch ever reaches the isolated worker.
    # Two independent axes: many texts means many one-at-a-time forward
    # passes, while a huge individual or aggregate text can be slow to even
    # tokenize (the sentence-transformer `.encode()` path truncates nothing
    # at all). Not applicable to GGUF (can_embed is a fixed False there) or
    # the dedicated on-device embedder (a separate, purpose-built path).
    "hf_embed_max_texts": 256,
    "hf_embed_max_chars": 200_000,
    # Primary GPU device on a multi-GPU box, and the device VRAM is read from
    # unless a 1-entry gpu_split_indices names one, or the only discrete GPU
    # beside integrated ones is used (discover.resolve_load_gpu_index).
    # None = no explicit selection (device 0). For a GGUF model it does not
    # confine the load to that device: without gpu_split_indices llama.cpp
    # still spreads the layers over every GPU. A stale index falls back to
    # device 0 with a logged warning, not a wrong/out-of-range GPU (see
    # discover.resolve_main_gpu_index).
    "main_gpu_index": None,
    # Which GPUs a model may use (GGUF: llama.cpp layer split; HF: accelerate
    # device_map restricted to these devices). 2+ entries split the model
    # across them. GGUF chat model: 1 entry loads it on that one GPU only, when
    # the device can be matched to llama.cpp's own device list
    # (discover.single_gpu_load_slot); None/empty spreads a GGUF model over
    # every discrete GPU. A GGUF load over 2+ GPUs is placed per card
    # (llamacpp/_gpu_plan.py, discover.gguf_split_devices). A GGUF split that names an
    # integrated GPU llama.cpp leaves out is refused. HF: 1 entry loads the
    # model on that GPU, overflowing to CPU; with no entries main_gpu_index
    # applies when set. A device no longer detected
    # at load time is dropped with a logged warning, not trusted blindly (see
    # discover.resolve_gpu_split).
    "gpu_split_indices": None,
    # Optional relative weight per entry in gpu_split_indices (same length,
    # any positive numbers, read as proportions). None means AUTOMATIC: a GGUF
    # load gives each card the layers that fit its free VRAM
    # (llamacpp/_gpu_plan.py). A length mismatch is ignored (warned). Set
    # explicit ratios (e.g. [1, 1]) to pin the shares.
    "gpu_split_ratios": None,
    # How many chat models may stay loaded at once. None (default) = no cap:
    # free-VRAM arithmetic alone decides, so a second model loads alongside the
    # first only when it provably fits (see inference/residency.py). Set an
    # integer to bound it regardless of headroom - 1 restores strict
    # single-resident. Applies to the HTTP server and the MCP server alike.
    "max_resident_models": None,
    # Model display names that are never chosen as an eviction victim, so a
    # user can say "keep these resident" instead of relying on LRU order.
    # Pinning only protects an already-loaded model; it never loads one. If
    # pins leave nothing evictable, the load proceeds and the miss is logged
    # rather than silently exceeding or silently enforcing the cap.
    "pinned_models": None,
    # Default system prompt for chat. The GUI's per-chat System prompt field
    # OVERRIDES this when set; a blank field inherits this. Empty by default.
    "chat_system_prompt": "",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    # Generation budget per reply. Thinking models (qwen3, deepseek-r1) spend
    # most of it reasoning, so 1024 cut answers mid-thought; this is a runaway
    # guard, not a cost control.
    "max_tokens": 4096,
    "confirm_remove": True,   # ask before localm rm deletes files
    # Sidebar wordmark, shared by web GUI and desktop launcher (picker writes,
    # launcher reads). One of: local-m (LocaL white + M blue, default),
    # loca-lm (Loca white + LM blue), localm (lowercase, m blue). Console
    # command, app icon, and shortcut are fixed regardless.
    "logo_style": "local-m",
    # Interface language for the web GUI, as a LANGUAGE_IDS id ("en", "de", "es", "fr").
    # The GUI reads it at boot and loads static/i18n/<id>.json; English is
    # built in and is the fallback for anything a catalog does not carry.
    "language": "en",
    # A background image behind chat messages: "" or a data:image URI. See
    # settings_schema.py's _validate_background_value / _AVATAR_DATA_URI_RE.
    "chat_background": "",
    # Standalone app window (localm[desktop] extra): its own close button hides
    # it to the tray by default, leaving the server running; Stop quits. True
    # makes the close button quit the whole app and stop the server instead.
    # No effect when localm opens in a browser tab.
    "desktop_window_quit_on_close": False,
    # "auto" (default): use the standalone app window (localm[desktop]) when
    # it is installed, otherwise the browser tab. "browser": always use the
    # browser tab even when the extra IS installed. There is no "always
    # window" value: a setting must never defeat run_native_window's
    # fallback-on-failure.
    "desktop_window_mode": "auto",
    # Where localm's temporary files go. "auto" (default): <data dir>/tmp for a
    # self-contained checkout, the system temp folder otherwise. "data": always
    # <data dir>/tmp. "system": the system temp folder. Or an absolute folder.
    # Applies on restart; the LOCALM_TMPDIR environment variable overrides it.
    "temp_location": "auto",
    "import_max_depth": 3,    # `localm add <dir>` recurses up to this many levels
    "port": 8642,             # default inference server port (auto-bumps if busy; an explicit --port does not)
    # Bind address for a fresh server start when -H/--host is not given on the
    # command line: "" = loopback only (127.0.0.1), "0.0.0.0" = every interface,
    # or one specific interface IP. GUI-settable (Settings > Server & network >
    # Bind address) so a browser-only user can enable the phone/Companion
    # feature without a terminal; applies on restart. An explicit -H always wins for that process
    # (and survives an in-place restart, which re-execs the same argv). Binding
    # past loopback still requires a strong API key: without one the server
    # IGNORES this key and stays on loopback (see plugins/gui/cli.py) - the
    # --insecure override has NO config form, so an unauthenticated network
    # bind can only ever be caused by an operator typing it in a terminal,
    # never by a config write.
    "bind_host": "",
    # Built-in TLS on network binds. tls_enabled False is the persistent
    # form of --no-tls (plain HTTP past loopback - the API key then crosses the
    # network in cleartext); tls_cert/tls_key are the persistent form of the
    # --tls-cert/--tls-key override pair (blank = localm's own local-CA cert).
    # CLI flags win over all three; loopback binds never use TLS either way.
    "tls_enabled": True,
    "tls_cert": "",
    "tls_key": "",
    "cors_origins": None,     # None = localhost only; list of origins; or "*"
    "allowed_hosts": None,    # None = own names only; list of host names; or "*"
    # Require a configured API key on protected endpoints: true refuses requests
    # until a key is set (see localm/auth.py); env override LOCALM_REQUIRE_AUTH.
    # False (default) = open in local/dev mode on loopback.
    "require_auth": False,
    # Let transformers IMPORT AND RUN a HuggingFace model directory's own bundled
    # Python (its `auto_map` custom code) when loading it. That is arbitrary code
    # execution as the server user, and a pulled repo's .py lands on disk like any
    # other file, so it is OFF by default and owner-only to turn on. With it off, a
    # model that needs custom code is REFUSED with an explanatory error rather than
    # silently executed (see inference/backends/hf.py).
    "hf_trust_remote_code": False,
    # Whether localm may treat a Windows drive letter mapped to a network share
    # (`net use Z: \\host\share`) as a normal local folder - the GUI folder
    # picker/create-folder/rename/log-export routes (pathsafe.
    # reject_unsafe_path_string) and RAG document indexing (rag/store.py
    # confine_index_path) both check it before touching such a path. True
    # (default) preserves the existing behaviour: a mapped drive already works
    # exactly like a local one today (pathsafe.is_unc_or_device_path never
    # flagged it, since "Z:" is syntactically an ordinary local drive) and this
    # setting does not change that unless turned off. Not a security floor - an
    # already-mapped drive is already connected, not a fresh SMB dial - so
    # turning it off is a preference (avoid remote-dependency latency/hangs,
    # keep localm strictly local-disk), not a vulnerability fix.
    "allow_network_drives": True,
    # Quick-select scope bundles for the "Keys & devices" manager (Settings),
    # each {name, scopes}, offered as one-tap presets when minting a key.
    # Re-seeded only when ABSENT (an emptied list stays empty). Privileged
    # scopes (coder:full / admin) apply only when an OWNER mints the key.
    "key_presets": [
        {"name": "Minimal", "scopes": ["chat"]},
        {"name": "Companion", "scopes": ["chat", "image", "music", "video",
                                         "voice", "rag", "web", "models:read"]},
        {"name": "Full", "scopes": ["chat", "coder", "image", "music", "video",
                                    "rag", "web", "voice", "mcp",
                                    "models:read", "models:write", "config:read"]},
        {"name": "Admin", "scopes": ["admin"]},
    ],
    # Command that starts your ComfyUI install (e.g. a launch .bat). When set,
    # image/music/video generators auto-start ComfyUI if not running (GUI, CLI,
    # or the coder's generate_image tool).
    "comfy_launch_cmd": None,
    # Working directory for comfy_launch_cmd (launchers assuming their own folder,
    # e.g. "python main.py" in a ComfyUI checkout). Blank + a launcher-file cmd =
    # run from that file's folder (the ComfyUI / ZLUDA .bat convention).
    "comfy_workdir": None,
    # Seconds to wait for ComfyUI to answer after launch. A ZLUDA / ROCm cold
    # start compiles GPU kernels and can take minutes, so the default is generous.
    "comfy_launch_timeout": 300,
    # ComfyUI's own output directory (e.g. StabilityMatrix's Images folder).
    # Only needed with comfy_delete_outputs, to find and delete ComfyUI's
    # duplicate copy. Blank = derived from the ComfyUI folder on demand.
    "comfy_output_dir": None,
    # Delete ComfyUI's OWN copy (and /history entry) after localm saves its own.
    # False (default) KEEPs them (a user may want ComfyUI's gallery). Privacy
    # mode forces deletion (no traces). Per-plugin config can override.
    "comfy_delete_outputs": False,
    # ComfyUI base URL localm talks to. None/blank uses the FLUX_API_URL env
    # override when set, else http://127.0.0.1:8188 (the ComfyUI default).
    "comfy_api_url": None,
    # Rewrite a slow `dequant_dtype: "float32"` in a Flux GGUF UNet loader to the
    # fast default (fp16/bf16) on submit: float32 doubles unpacked model size and
    # forces CPU offload on a VRAM-limited card (~36 s/it vs ~6-7 s/it). True
    # (default) auto-corrects; False submits your dequant choice verbatim.
    "comfy_fast_dequant": True,
    # Shared fallback for the per-plugin "Model weight dtype" Media setting
    # (music/video backends read config["plugins"][<name>]["comfy"]["float_type"],
    # else this). None = inherit the workflow default. The per-plugin values are
    # what the GUI's Media section edits; this global is the documented fallback
    # they inherit (settable via `localm config` / PATCH like any key).
    "comfy_float_type": None,
    # Suppress ComfyUI opening its own web page when localm auto-launches it.
    # False (default) keeps ComfyUI's tab. True appends --disable-auto-launch so
    # it starts headless; stock run_*.bat / comfyui.* / bare "python main.py"
    # forward the flag, a launcher that drops extra args just ignores it
    # (non-breaking). Applies to image/music/video (shared ensure_comfy()).
    "comfy_disable_auto_launch": False,
    # Reactive, opt-in in-memory shim for the upstream ComfyUI __func__
    # regression. Off by default (touches nothing).
    # When on, a ComfyUI localm SPAWNS gets a localm-owned shim dir on its child
    # PYTHONPATH to patch the regression in memory; localm never writes into the
    # user's install nor shims a ComfyUI it did not start. Set by the reactive
    # offer or `localm config comfy_func_shim on`; self-expires once Comfy is fixed.
    "comfy_func_shim": False,
    # Which ComfyUI localm targets (coexistence). "own" (default) = use a
    # localm-managed instance once one is installed under <LOCALM_HOME>/comfyui
    # (`localm comfy setup`); inert until then, so a fresh install behaves
    # identically to today. "user" = always use the user's own ComfyUI
    # (comfy_workdir / comfy_api_url), even if a managed instance exists. With
    # no managed instance, both behave identically. This key only routes
    # (provisioning is stages S2/S3) and never modifies the user's own ComfyUI.
    # See localm/media/managed_comfy.py.
    "comfy_target": "own",
    # EXPERIMENTAL, default OFF: per-component GPU placement for media generation.
    # When on, and the running ComfyUI offers the multigpu Select*Device nodes
    # AND a 2+ card split is configured, localm injects those nodes so the text
    # encoder + VAE load on a second card while the heavy diffusion model stays
    # on the preferred one. OFF by default even on a multi-GPU box: ComfyUI's
    # gpu:N is a POSITION in a reordered visible list, so an off-by-one lands a
    # component on the wrong card and STILL RENDERS (a silent wrong result, not
    # a crash). On one card, or an older ComfyUI, or with the toggle off, media
    # generation runs on the single preferred card.
    "comfy_gpu_placement": False,
    # Session persistence mode for ALL surfaces (chat, server, GUI, coder):
    #   privacy = no traces written automatically (default)
    #   log     = JSONL audit trail in <data dir>/sessions/
    #   full    = log + markdown transcript
    # chat_mode / coder_mode override per surface (None = inherit); CLI --mode
    # overrides everything.
    "mode": "privacy",
    "chat_mode": None,
    "coder_mode": None,
    # Long-term chat memory: recall durable facts/preferences and inject them
    # (server-side) into the system prompt each turn. Recall is free (BM25 over a
    # small store). False stops injecting (existing memories kept, just unused).
    "memory_enabled": True,
    # Grow memory automatically: after a chat turn (log/full mode, never privacy)
    # distil durable facts into the store in the background, debounced to once per
    # MEMORY_AUTO_MIN_INTERVAL (see the chat plugin). Skipped in privacy (no new
    # traces). False requires the manual "Synthesize now" button / jobs "memory"
    # task / POST /api/memory/consolidate.
    "memory_auto_consolidate": True,
    # Privacy mode normally disables memory ENTIRELY (no recall + no writes). On =
    # allow READING existing memories into the prompt in privacy mode (writing
    # stays off - privacy never creates a trace). Off by default. Per-surface:
    "memory_recall_in_privacy": False,
    "memory_recall_in_privacy_chat": True,      # applies only when the master is on
    "memory_recall_in_privacy_coder": True,     # applies only when the master is on
    # Keep diagnostics for bug reports even in privacy mode. Off by default:
    # privacy mode writes NO automatic trace (the hang watchdog trace, the
    # crash-restart breadcrumbs, and the debug log are all suppressed). On =
    # keep those diagnostics regardless of mode, so an intermittent freeze/crash
    # leaves something to attach to a report. Never chat content - code stacks
    # and operational logs only.
    "keep_diagnostics": False,
    # On-device embedding model for semantic search (RAG hybrid retrieval + agent
    # memory): a small dedicated GGUF, loaded separately from the chat model.
    # Value = a known key (bge-small-en-v1.5, nomic-embed-text-v1.5), a registered
    # model name, or a GGUF path. A known model is fetched into
    # <home>/models/embeddings on first use (automatic under net_mode ask and allow;
    # under off run 'localm setup-embeddings'). Until present, memory/RAG fall back to BM25.
    "embedding_model": "bge-small-en-v1.5",
    # GPU layers for the embedding model. None (default) = automatic placement:
    # full GPU offload when free VRAM demonstrably holds the model (file size
    # + 20% slop), else CPU - so a resident chat model is not thrashed by
    # WDDM oversubscription when a LARGE embedder (e.g. a 4 GB Qwen3-Embedding
    # Q8) shares one card (measured 34 -> 5 tok/s on 16 GB). 0 forces CPU;
    # 99 forces full GPU offload regardless. When unset, an explicit global
    # n_gpu_layers other than 99 is still inherited, as it always was.
    "embedding_gpu_layers": None,
    # How the embedding model's token states are pooled into one vector. None
    # (nothing explicitly chosen) is NOT the same as "mean": embedder.py's
    # resolve_pooling_setting/_effective_pooling apply the measured-safe
    # default per model - MEAN for the bundled bge/nomic choices (matches every
    # existing index they built), but LAST for a model that DECLARES last-token
    # pooling (a decoder-based embedder such as Qwen3-Embedding, gte-Qwen2),
    # since forcing mean on those measurably degrades them and there is no
    # existing mean-built index of that shape to protect. An explicit choice
    # here ("mean", "last", "cls", "none", or "auto" to always follow whatever
    # the GGUF declares) always overrides that default. Changing the EFFECTIVE
    # pooling invalidates already-embedded RAG collections and memory vectors
    # (same dimensions, different meaning), so re-index after changing it.
    "embedding_pooling": None,
    # Which host folders the document-indexing (RAG) API may READ. All three keys
    # are OWNER-ONLY: a non-owner config:write key can neither see nor set them
    # (enforced at PATCH /v1/config; see settings_schema.admin_only). The localm
    # data dir and credential folders (.ssh, .aws, ...) stay denied in EVERY mode
    # (hard floor; rag/store.py confine_index_path). Read by indexing_policy().
    #   whitelist (default) = only home, the working dir, and rag_allowed_roots.
    #   blacklist           = anywhere EXCEPT rag_denied_roots.
    "rag_indexing_mode": "whitelist",
    "rag_allowed_roots": [],   # extra folders allowed in whitelist mode
    "rag_denied_roots": [],    # folders refused in blacklist mode
    # A document's format label (json/yaml/python/...) is derived heuristic-FIRST:
    # known extension wins, else a structural sniff (rag/extract.classify_format).
    # This toggle only governs the LLM TIE-BREAK: when both are inconclusive AND a
    # chat model is loaded, prompt-guess from a snippet (cached per extension for
    # the process). Off -> labeled "text"; never fired during embedding-only index.
    "rag_classify_unknown_files": True,
    # Re-score the best Knowledge hits with a reranker model before they are
    # injected into a chat. Takes effect only while a reranker is registered
    # (rag_rerank_model names one; empty = the only registered reranker); without
    # one retrieval is unchanged. rag_rerank_candidates is how many of the best
    # blended hits the reranker sees.
    "rag_rerank": True,
    "rag_rerank_model": "",
    "rag_rerank_candidates": 20,
    # The reranker score a hit needs to count as relevant when chat asks for
    # relevant hits only, in the reranker's own scale. None = the measured value
    # for the reranker files in localm.rag.rerank.CALIBRATED_MIN_SCORES, and the
    # cosine/keyword floor for any other reranker.
    "rag_rerank_min_score": None,
    # Seconds a GUI coder approval card may sit unanswered before it is
    # auto-rejected and the agent moves on.
    "coder_confirm_timeout": 600,
    # Wall-clock cap on the coder's startup project-map scan. <= 0 disables
    # the deadline (scan to completion however long it takes).
    "coder_index_timeout": 20,
    # Caps on the coder's grep tool. Each is overridable per call; 0 = no cap.
    # Matches shown per file (the rest are still counted and reported), output
    # lines before the sweep stops, and the per-file size above which a file is
    # skipped (and the skip reported) rather than read.
    "coder_grep_max_per_file": 20,
    "coder_grep_max_output_lines": 300,
    "coder_grep_max_file_bytes": 4194304,
    # Episodic memory: the coder recalls lessons from past sessions on a project
    # (BM25, free) and at session close distils the session into a new lesson (one
    # model call per session that changed files). Writes skipped in privacy mode
    # and for restricted (shareable-key) sessions, stored under the home dir not
    # the project tree. False disables both halves.
    "coder_episodic_memory": True,
    # Provenance tagging: re-frame coder tool results from untrusted (network /
    # MCP) tools as data-not-instructions, so a fetched page or external server
    # cannot inject into the model loop (indirect prompt injection). Defense in
    # depth (blocks nothing, only labels). Leave ON absent a specific reason.
    "coder_untrusted_provenance": True,
    # Whether an unattended coder run (no one to ask) may use network tools
    # while net_mode is "ask": "deny" or "allow". net_mode "off" still refuses them.
    "coder_unattended_network": "deny",
    # Pre-done self-review: before the coder declares done, a reviewer model reads
    # the diff and feeds blocking issues back for one more fix pass. Off by default
    # (adds a model round-trip per task that changed files).
    "coder_review": False,
    # Reviewer target: "" = the agent's own model (local); "local" = a different
    # small model on CPU in the coder's process (set coder_reviewer_model; adds CPU
    # latency); "openai"/"anthropic" = cloud; an http(s) URL = a 2nd
    # OpenAI-compatible endpoint. A NETWORK reviewer (cloud / non-loopback URL) is
    # skipped in privacy mode and for shared keys (would send the diff off-machine)
    # and falls back to the local model; the "local" CPU reviewer stays on-machine.
    "coder_reviewer": "",
    # Model name (or path) for a heterogeneous reviewer ("local"/cloud/URL); blank
    # uses a sensible provider default or the agent's own model name.
    "coder_reviewer_model": "",
    # Constrain coder tool calls with a LAZY GBNF grammar: thinking/prose flow
    # free, but a started <tool_call> is forced to valid JSON (no malformed calls
    # to repair). ON by default for grammar-capable local backends. External API
    # / grammar-less builds unaffected (supports_grammar gate + runtime
    # soft-degrade). NOTE: a config.json that already stores False keeps it
    # (saved wins) - flip it in Settings.
    "coder_tool_grammar": True,
    # Same constraint as coder_tool_grammar, for chat's own tool caller (the
    # web_search/fetch_url loop, scheduled jobs and the interactive GUI alike):
    # once the model starts a <tool_call>, force it to be valid tool-call JSON.
    "chat_tool_grammar": True,
    # Most web lookups (searches and page reads) the chat runs for one message
    # before it stops looking things up and answers from what it found. 0 = no
    # ceiling. Repeated and no-progress lookups end the loop sooner either way.
    "chat_web_max_lookups": 20,
    # The GUI chat and `localm run` add a short style note to the system prompt
    # (answer directly, speak of yourself as "I"). API callers never get it.
    "chat_style_note": True,
    # Chat avatars: an emoji or a small data:image URI, never a URL. See
    # settings_schema.py's _validate_avatar_value / _AVATAR_DATA_URI_RE.
    "user_avatar": "",
    "user_name": "",
    "model_avatar_default": "",
    "model_avatar_overrides": {},
    # After an image is generated, ask ComfyUI to release VRAM and reload the chat
    # model so the next reply is instant. Off = the chat model reloads lazily on
    # the next message instead (better for many images in a row).
    "reload_llm_after_imagine": True,
    # VRAM-aware media model swap: before an image/music/video gen the chat LLM is
    # unloaded so the media model gets the GPU (on a big card both fit).
    #   auto   = keep chat loaded when the media model fits alongside it (free VRAM
    #            >= estimate + headroom), else swap (default)
    #   always = always unload the chat model
    #   never  = never unload; keep chat hot (media may OOM on a small card)
    # reload_llm_after_imagine is a SEPARATE axis (eager-vs-lazy reload AFTER a
    # gen, not this unload-before decision).
    "model_swap_policy": "auto",
    # When a chat request that does not pin its model may be answered by another
    # installed model (capability_routing.AUTOSWITCH_MODES):
    #   off    = never; the selected or loaded model always answers
    #   image  = only for an image or audio clip the current model cannot read
    #            (default)
    #   ask    = never on its own; the reply offers the model it would switch to
    #   loaded = only to a model that is already loaded (nothing is loaded or evicted)
    #   auto   = to an installed model when the current one is confirmed to lack
    #            a capability the request needs
    #   eager  = as auto, and also when the current model's capability is unknown
    "model_autoswitch": "image",
    # Free the loaded model from VRAM after this many idle seconds, so a running
    # server stops holding the GPU; the next request reloads it lazily. 0 =
    # disabled (default): resident until an explicit unload or swap. Measured from
    # the last request, like Ollama's keep_alive.
    "idle_unload_seconds": 0,
    # Network policy for model-initiated requests (coder fetch_url/web_search,
    # chat web access). See localm/netpolicy.py and docs/network.md.
    #   off   = all policy-routed network access fails fast
    #   ask   = allowed; the coder asks for approval per request, and the chat
    #           GUI advertises the web tools with an approval card per
    #           model-initiated request (default)
    #   allow = no confirmation
    "net_mode": "ask",
    "net_allow": [],            # domains; empty = any. "x.com" covers *.x.com
    "net_deny": [],             # domains always refused (wins over allow)
    "net_allow_private": False, # True = permit loopback/private targets (SSRF guard off)
    "net_allow_model_downloads": False,  # True = explicit downloads bypass net_mode=off
    "net_search_url": None,     # SearXNG base URL; None = DuckDuckGo, then Brave (no key)
    # Display a remote image a model links in a reply, by fetching it SERVER-side
    # and streaming it back, so the browser never contacts the remote origin.
    # OFF by default and that is deliberate: a rendered remote image is the
    # standard model-driven exfiltration channel (the URL itself carries the
    # payload), and turning this on does not close that channel, it only makes
    # the request come from this machine's server rather than from the browser.
    # What it does buy, when a user wants images to work: the remote host never
    # sees the browser's IP, User-Agent or referrer, and the fetch is subject to
    # the same SSRF guard and domain lists as every other outbound request.
    # "ask" is the state that DOES close the channel for an arbitrary host: the
    # route refuses until the GUI has asked the reader about that origin.
    "gui_proxy_remote_images": REMOTE_IMAGE_OFF,
    # Render a self-contained HTML/SVG reply block in the sandboxed artifact
    # canvas. Off hides the canvas button everywhere; owner_only hides it from
    # every caller that is not the owner.
    # Browser automation. Off by default even for a key holding the browser
    # capability: the scope grants eligibility, this switch grants use.
    "browser_enabled": False,
    "browser_headless": True,
    "browser_engine": "bundled",
    "browser_custom_domain_rules": False,
    "browser_allow": [],
    "browser_deny": [],
    "browser_inline_live_view": False,
    "gui_preview_enabled": True,
    "gui_preview_owner_only": False,
    # Let a canvas artifact load another page in its frame. Off keeps the
    # shell's frame-src 'none' and closes the canvas when the frame navigates.
    "canvas_allow_navigation": False,
    "coder_rail_side": "right",
    "coder_remember_projects": True,
    "coder_projects_remembered": 20,
    # Reach localm by NAME, not just IP, on a network bind (see localm/netname.py
    # and docs/naming.md). mDNS/Bonjour advertises "<mdns_name>.local" so a phone
    # opens https://localm.local:PORT with no IP; the name is folded into the TLS
    # cert and the Tailscale MagicDNS name is detected + certified automatically.
    # Loopback binds never advertise.
    "mdns_name": "localm",      # the .local name; sanitized to a DNS label on use
    "mdns_enabled": True,       # advertise the name over mDNS on network binds
    "metrics_enabled": False,   # serve Prometheus GET /metrics (admin key; restart to apply)
    # Speech-to-text (GUI mic button; needs the [voice] extra).
    # Model sizes: tiny / base / small / medium - bigger = better + slower.
    "voice_stt_model": "base",
    "voice_stt_language": None,  # None = auto-detect; or "en", "de", ...
    # When a registered model's file has gone missing, False (default) flags the
    # entry as "missing" (kept, shown in the list); True deletes the entry.
    # Only files under the models folder are ever auto-deleted.
    "autoprune_missing_models": False,
    # When a command belongs to a known first-party plugin that is not
    # installed/enabled (e.g. /generate-image with image off), suggest installing
    # it instead of "unknown command". False silences the hint; a truly unknown
    # command always errors.
    "suggest_plugins": True,
    # When a plugin declaring pip extras (requires_extras) is installed/enabled by
    # the local operator, auto-install those extras on the HOST. A remote client
    # never triggers a server-side pip (only CLI or a loopback GUI request does).
    # `localm plugin setup` records the choice here. Default True, read via
    # .get(..., True), so a config.json without this key gets True.
    "auto_install_plugin_deps": True,
    # Bug reports, the read-only Issues view, and the self-updater all talk to ONE
    # small Cloudflare Worker (the localm proxy; see tools/bugreport-proxy/) that
    # holds the GitHub tokens SERVER-SIDE. Shipped as DEFAULTS so a fresh download
    # works with ZERO setup (update_url/token below fall back to these). No GitHub
    # token is in the app - only the public Worker URL and a low-value client
    # token that is PUBLIC, NOT a secret: it only gates against drive-by spam
    # (Cloudflare rate limiting is the real control),
    # can ONLY file an issue (never read the repo), and is rotatable at the Worker.
    # Set either to "" (or null) to opt a build out of the hosted channel.
    "bugreport_upload_url": "https://localm-bugreport-proxy.localm.workers.dev",
    "bugreport_upload_token": "3x_HA2UXbwNDnNfdDmpFBvvfcl2S-I-9t7XLQRAShM4",
    # Update channel + read-only issues tracker. One Worker hosts report + issues +
    # update, so these default to the proxy above; set update_url/token ONLY to
    # point updates at a different Worker (needs a Contents:read token, separate
    # from Issues, plus the shared secret). None = no update channel (banner +
    # `localm update` hidden). See tools/bugreport-proxy/.
    "update_url": None,
    "update_token": None,
    # Opt-in only: the update check stays stable-only unless a local admin turns
    # this on (settings_schema.py's update_allow_prerelease, admin_only). A
    # prerelease build is signed and anti-rollback checked exactly like a stable
    # one; this only widens WHICH candidate the proxy considers offering.
    "update_allow_prerelease": False,
    # Net-policy carve-out for the update channel ONLY (see updater.py's check()
    # and settings_schema.py's update_ignore_net_policy, admin_only). net_mode
    # is meant to be a real kill switch - explicit user actions still respect
    # net_mode=off (see netpolicy.py's module docstring) - so this defaults to
    # False: the update check obeys it like everything else unless an admin
    # opts this one channel out.
    "update_ignore_net_policy": False,
    # Names of enabled engine plugins (WordPress-style). Managed by the plugin
    # engine (plugin enable/disable, GUI Plugins page) via update_config, NOT the
    # settings form. Declared here for a documented home + default (else the
    # settings-save endpoint rejects it as unknown). A plugin is active only when
    # installed on disk AND in this list; see docs/plugins.md.
    "plugins_enabled": [],
    # Per-plugin config namespace (e.g. plugins["image"]["comfy"]["output_dir"]).
    # Written by the plugin engine and media backends via update_config, NOT the
    # flat settings form. Declared here so settings-save accepts it and the
    # per-plugin media-containment knob survives a full-config round-trip.
    "plugins": {},
}

# localm claims 8642-8741, clear of ComfyUI (8188), A1111 (7860),
# Ollama (11434), and the 8000/8080/8888 dev-server crowd.
PORT_RANGE = (8642, 8741)

# pick_port's restart_grace_window poll interval.
_RESTART_PORT_POLL_S = 0.15


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is already listening on host:port.

    Resolves *host* first instead of assuming a family, so an IPv6 host such as
    ``::1`` is probed rather than raising ``socket.gaierror``.

    A name that resolves to several addresses (``localhost`` on a dual-stack box
    is both ``::1`` and ``127.0.0.1``) is in use when ANY of them answers.

    An unresolvable host returns False rather than raising, and the resolution
    failure is logged rather than dropped; the caller goes on to bind, and the
    bind produces the accurate error. A config-driven bind has already been
    screened by ``cli._bind_preflight_error`` before reaching here.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        from localm.debuglog import logger
        logger.debug("port_in_use: cannot resolve %r (%s); treating the port as "
                     "free and leaving the real error to the bind", host, e)
        return False
    for family, socktype, proto, _canon, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.settimeout(0.2)
                if s.connect_ex(sockaddr) == 0:
                    return True
        except OSError:
            # This family is unusable on this box (an IPv6 address with the
            # stack disabled): keep probing the remaining addresses.
            continue
    return False


class PortInUseError(RuntimeError):
    """An explicitly requested port is already in use.

    An explicit ``--port`` is honored exactly or not at all: if it is busy,
    ``pick_port`` raises this instead of quietly binding a different port. The
    default (no ``--port``) still auto-bumps through the range; only an explicit
    request refuses.
    """

    def __init__(self, port: int):
        self.port = port
        super().__init__(f"Port {port} is already in use.")


def pick_port(requested: Optional[int] = None, host: str = "127.0.0.1", *,
              restart_grace_window: float = 0.0):
    """
    Resolve the port to serve on. Returns ``(port, default_port_was_busy)``.

    An explicit ``requested`` port is honored exactly: returned if free, otherwise
    :class:`PortInUseError` is raised. It is never silently relocated to a
    different port (see that class for why).

    ``restart_grace_window`` (seconds, default 0 - no grace): when the
    explicit port is busy, poll for up to this long before raising, instead
    of failing on the first check. For a self-restart resuming on the port it
    just held, the prior process's listening socket can take a brief moment
    to release. ``PortInUseError`` is still raised, unchanged, once the
    window elapses with the port still busy - this delays the same refusal
    by a bounded amount, it never relocates an explicit port.

    With no explicit port, uses the configured default (``config['port']``, 8642)
    and, if that is busy, walks localm's range (8642-8741) for the next free port,
    falling back to an OS-assigned port. The returned flag is True only in that
    auto-bump case; an explicit request never returns True (it raises instead).
    """
    if requested is not None:
        if not port_in_use(requested, host):
            return requested, False
        if restart_grace_window > 0:
            import time
            deadline = time.monotonic() + restart_grace_window
            while time.monotonic() < deadline:
                time.sleep(min(_RESTART_PORT_POLL_S, max(0.0, deadline - time.monotonic())))
                if not port_in_use(requested, host):
                    return requested, False
        raise PortInUseError(requested)
    start = load_config().get("port", PORT_RANGE[0])
    if not port_in_use(start, host):
        return start, False
    for candidate in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
        if candidate != start and not port_in_use(candidate, host):
            return candidate, True
    return get_free_port(), True


def _mkdir_or_explain(path: Path, *, is_home: bool) -> None:
    """``path.mkdir(parents=True, exist_ok=True)`` with one user-error case turned
    into a clean message instead of a crash.

    ``parents=True`` so an explicit ``LOCALM_HOME`` a level or two below a
    not-yet-existing folder (a fresh ``D:\\localm\\data``) is created like
    ``mkdir -p`` rather than crashing with WinError 3.

    ``exist_ok=True`` already swallows "exists AND is a directory", so a
    ``FileExistsError`` from mkdir means exactly "exists but is NOT a directory"
    (a regular file / symlink; WinError 183 on Windows, EEXIST on POSIX),
    typically ``LOCALM_HOME`` set to a file. That is raised as a
    ``click.ClickException``: the CLI's cross-cutting handler passes those
    straight through (a clean "Error: ..." line, exit 1), never routing them to
    the generic "unexpected error" + bug-report path. Other OSErrors (permission
    denied) are left to propagate unchanged."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        import click  # lazy: keep the CLI framework out of config's import graph
        env = os.environ.get("LOCALM_HOME", "").strip()
        if is_home and env and Path(env).expanduser() == path:
            msg = (f"LOCALM_HOME points at a file, not a directory: {path}. "
                   "Set it to a directory (or remove/rename that file).")
        else:
            msg = (f"localm's data path is a file, not a directory: {path}. "
                   "localm needs it to be a directory; remove or rename that "
                   "file, or point LOCALM_HOME at a directory.")
        raise click.ClickException(msg) from None


def ensure_dirs() -> None:
    _mkdir_or_explain(HOME_DIR, is_home=True)
    _mkdir_or_explain(MODELS_DIR, is_home=False)
    _dirs_made.add((HOME_DIR, MODELS_DIR))


_dirs_made: set = set()


def _ensure_dirs_once() -> None:
    """``ensure_dirs()`` unless it already succeeded in this process for the
    current HOME_DIR and MODELS_DIR. For readers; writers call ensure_dirs()."""
    if (HOME_DIR, MODELS_DIR) not in _dirs_made:
        ensure_dirs()


def _logger():
    from localm.debuglog import logger
    return logger


# Registry and config are mutated from several places at once - the GUI server
# threads, the `localm pull` subprocess the GUI spawns, and sync_models_dir on
# every launch. The helpers below make every write atomic
# (write a temp file in the same dir, fsync, then os.replace - readers see only
# the old or the new complete file, never a torn one) and make every read
# crash-proof (fall back to the .bak snapshot, then to the default).
# LOCK ORDER: _rmw_lock (outer) -> <name>.lock file -> _io_lock (inner). _io_lock
# is the leaf: nothing is acquired while it is held. Readers take only _io_lock.
_io_lock = threading.RLock()

# Serializes update_config()/update_registry() read-modify-write callers within
# this process; held across the cross-process lock wait, which _io_lock is not.
# Re-entrant: a same-thread nested call passes through and fails at
# _cross_process_lock's token check. LOCK ORDER: _rmw_lock (outer) ->
# <name>.lock file -> _io_lock (inner).
_rmw_lock = threading.RLock()


# --- Reads made on a running asyncio event loop ---------------------------- #
# A read of config.json or registry.json called on an event-loop thread runs on
# that file's background reader thread. The loop waits for it at most
# LOOP_READ_WAIT_S; a read that takes longer is answered from the last complete
# read of the same file (the kept read), and the file is then marked slow: every
# later call on the loop gets the kept read at once, with a fresh read queued
# behind, until a read finishes within LOOP_READ_WAIT_S again. A kept read is
# served only while it is at most LOOP_KEPT_MAX_AGE_S old (counted from when
# that read completed); past that the loop waits for the fresh read. The reader
# threads share _io_lock, so a read stuck on the disk holds the other files'
# reads too. A write made by
# this process through save_config, update_config, save_registry or
# update_registry reads every kept file again on the writing thread before it
# returns, so a call on the loop that starts after the write returned never gets
# a value older than that write. A call with no kept read yet, or made while
# this thread holds _io_lock or _rmw_lock, reads on the calling thread. Calls
# off the loop always read on the calling thread. LOCK ORDER: _loop_reads_lock
# is a leaf: nothing is acquired and no file is read while it is held.
LOOP_READ_WAIT_S = 0.5
LOOP_KEPT_MAX_AGE_S = 2.0

_loop_reads_lock = threading.Lock()
_loop_reads_epoch = 0
_loop_kept: dict = {}
_loop_inflight: dict = {}
_loop_slow: dict = {}
_loop_readers: dict = {}
_loop_refilling = threading.local()


class _LoopRead:
    """One read handed to a reader thread."""

    __slots__ = ("name", "key", "epoch", "fn", "started", "done", "value", "error",
                 "ran")

    def __init__(self, name: str, key: tuple, epoch: int, fn: Callable) -> None:
        self.name = name
        self.key = key
        self.epoch = epoch
        self.fn = fn
        self.started = time.monotonic()
        self.done = threading.Event()
        self.value = None
        self.error: Optional[BaseException] = None
        self.ran = False


def _on_event_loop_thread() -> bool:
    """True when the calling thread is running an asyncio event loop."""
    aio = sys.modules.get("asyncio")
    if aio is None:
        return False
    try:
        aio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _holds_store_lock() -> bool:
    """True when the calling thread holds _io_lock or _rmw_lock."""
    return any(getattr(lk, "_is_owned", lambda: False)()
               for lk in (_io_lock, _rmw_lock))


def _store_paths() -> tuple:
    return (HOME_DIR, MODELS_DIR, CONFIG_FILE, REGISTRY_FILE)


def _run_loop_read(job: _LoopRead) -> None:
    """Run *job* on its reader thread and publish the result."""
    snapshot = None
    if job.key == _store_paths():
        job.ran = True
        try:
            job.value = job.fn()
            snapshot = copy.deepcopy(job.value)
        except Exception as e:  # handed to every waiting caller
            job.error = e
    took = time.monotonic() - job.started
    recovered = None
    with _loop_reads_lock:
        if _loop_inflight.get(job.name) is job:
            del _loop_inflight[job.name]
        if job.ran and job.error is None and job.epoch == _loop_reads_epoch:
            _loop_kept[job.name] = (job.key, snapshot, job.fn, time.monotonic())
            if took <= LOOP_READ_WAIT_S and job.name in _loop_slow:
                recovered = time.monotonic() - _loop_slow.pop(job.name)
    job.done.set()
    if recovered is not None:
        _logger().info("reads of %s are fast again after %.1f s; requests on the "
                       "server's event loop read it fresh again.", job.name, recovered)


def _loop_reader_main(q) -> None:
    while True:
        job = q.get()
        try:
            _run_loop_read(job)
        except BaseException:  # noqa: BLE001
            job.done.set()
            _logger().exception("the reader thread for %s failed a read", job.name)


def _submit_loop_read(job: _LoopRead) -> None:
    """Queue *job* on its file's reader thread, starting the thread if needed.
    Caller holds _loop_reads_lock."""
    entry = _loop_readers.get(job.name)
    if entry is None or not entry[1].is_alive():
        import queue
        q = queue.SimpleQueue()
        t = threading.Thread(target=_loop_reader_main, args=(q,),
                             name=f"localm-reader-{job.name}", daemon=True)
        t.start()
        entry = (q, t)
        _loop_readers[job.name] = entry
    entry[0].put(job)


def _loop_reads_after_fork() -> None:
    """In a forked child: forget the parent's reader threads and kept reads."""
    global _loop_reads_lock
    _loop_reads_lock = threading.Lock()
    _loop_readers.clear()
    _loop_kept.clear()
    _loop_inflight.clear()
    _loop_slow.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_loop_reads_after_fork)


def _loop_reads_invalidate() -> None:
    """Drop every kept read and every slow mark, and stop later callers from
    joining a read that started before now."""
    global _loop_reads_epoch
    with _loop_reads_lock:
        _loop_reads_epoch += 1
        _loop_kept.clear()
        _loop_inflight.clear()
        _loop_slow.clear()


def _refill_after_write() -> None:
    """Read every kept file again on the calling thread, after a write by this
    process. A read already running is not joined by later callers and its
    result is not kept. A file whose read fails here is dropped from the kept
    reads. Does nothing when called from inside another refill."""
    global _loop_reads_epoch
    if getattr(_loop_refilling, "active", False):
        return
    with _loop_reads_lock:
        _loop_reads_epoch += 1
        epoch = _loop_reads_epoch
        _loop_inflight.clear()
        todo = [(name, kept[0], kept[2]) for name, kept in _loop_kept.items()]
    _loop_refilling.active = True
    try:
        for name, key, fn in todo:
            try:
                value = copy.deepcopy(fn()) if key == _store_paths() else None
            except Exception:  # dropped below; the next read reports it
                value = None
            with _loop_reads_lock:
                if epoch != _loop_reads_epoch:
                    return
                if value is None:
                    _loop_kept.pop(name, None)
                else:
                    _loop_kept[name] = (key, value, fn, time.monotonic())
    finally:
        _loop_refilling.active = False


_T = TypeVar("_T")


def _loop_safe_read(name: str, fn: Callable[[], _T]) -> _T:
    """``fn()`` (a read of *name*). On a running event loop with a kept read of
    *name* at most LOOP_KEPT_MAX_AGE_S old, the loop waits at most
    LOOP_READ_WAIT_S and is otherwise answered from that kept read; with an
    older kept read it waits for the fresh read. Off the loop, or with no kept
    read yet, it is ``fn()`` on the calling thread. Every caller gets its own
    copy. Raises what ``fn`` raises."""
    if not _on_event_loop_thread() or _holds_store_lock():
        return fn()
    key = _store_paths()
    with _loop_reads_lock:
        kept = _loop_kept.get(name)
        if kept is not None and kept[0] != key:
            kept = None
        job: Optional[_LoopRead] = _loop_inflight.get(name)
        if kept is not None and (job is None or job.key != key
                                 or job.epoch != _loop_reads_epoch):
            job = _LoopRead(name, key, _loop_reads_epoch, fn)
            _loop_inflight[name] = job
            _submit_loop_read(job)
        usable = kept is not None and time.monotonic() - kept[3] <= LOOP_KEPT_MAX_AGE_S
        if kept is not None and usable and name in _loop_slow:
            return copy.deepcopy(kept[1])
        epoch = _loop_reads_epoch
    if kept is None:
        value = fn()
        snapshot = copy.deepcopy(value)
        with _loop_reads_lock:
            if epoch == _loop_reads_epoch:
                _loop_kept[name] = (key, snapshot, fn, time.monotonic())
        return value
    job = cast(_LoopRead, job)
    if usable:
        remaining = LOOP_READ_WAIT_S - (time.monotonic() - job.started)
        job.done.wait(max(0.0, remaining))
        if not job.done.is_set() and time.monotonic() - kept[3] <= LOOP_KEPT_MAX_AGE_S:
            with _loop_reads_lock:
                first = name not in _loop_slow
                if first:
                    _loop_slow[name] = job.started
            if first:
                _logger().warning(
                    "reading %s has taken over %.1f s (a busy disk?); requests on "
                    "the server's event loop are answered from the last complete "
                    "read, at most %.0f s old, until a read is fast again.",
                    name, LOOP_READ_WAIT_S, LOOP_KEPT_MAX_AGE_S)
            return copy.deepcopy(kept[1])
    job.done.wait()
    if not job.ran:
        return fn()
    if job.error is not None:
        raise job.error
    return cast(_T, copy.deepcopy(job.value))


def _holds_unparseable_json(path: Path) -> bool:
    """True when *path* can be opened but its content is not valid JSON (corrupt,
    truncated, not UTF-8, too deep). False when it parses, and False when it
    cannot be opened at all (locked or denied), which says nothing about its
    content."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
    except (ValueError, RecursionError):
        return True
    except OSError:
        return False
    return False


def _copy_to_backup(path: Path, bak: Path) -> None:
    """Copy *path* over *bak* through a unique temp file in the same folder,
    leaving *path* in place. A transient Windows sharing violation on the read
    is retried like ``_replace_atomic``. Raises OSError when the copy fails;
    the temp file is removed."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=bak.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        for attempt in range(_REPLACE_RETRIES):
            try:
                shutil.copyfile(path, tmp)
                break
            except PermissionError as e:
                if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                    raise
                _transient_backoff(attempt)
        _replace_atomic(tmp, bak)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, data) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace).

    Keeps a one-step .bak of the previous good file so a corrupt read can
    recover. The .bak is a copy, so *path* exists at every moment of the write
    and a reader in another process never finds it missing. os.replace is atomic on Windows and POSIX when src/dst share a
    filesystem, which they do (same directory); _replace_atomic additionally
    rides out the transient Windows sharing violation a concurrent reader causes.

    The temp file gets a UNIQUE per-write name (mkstemp), NOT a fixed ``<name>.tmp``.
    Two localm processes writing the same file concurrently (a CLI ``pull``/``config``
    alongside the running GUI, or two CLI invocations) would otherwise both open the
    SAME ``<name>.tmp`` and collide - one save then crashes when the other's replace
    has already consumed the shared temp (Windows WinError 2/32), or on POSIX their
    writes interleave into a torn temp. ``_io_lock`` only serialises writers WITHIN
    one process; _replace_atomic's retry rides out a locked DESTINATION but not two
    writers sharing one temp SOURCE. A unique temp per writer removes the collision
    at the source, so cross-process contention degrades to the documented
    last-writer-wins on the final file, with no crash or torn write."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            bak = path.with_name(path.name + ".bak")
            try:
                if _holds_unparseable_json(path):
                    # Never rotate an unreadable file over the .bak it may have been
                    # recovered from: keep it under its own name instead.
                    kept = path.with_name(
                        f"{path.name}.unreadable-{time.strftime('%Y%m%d-%H%M%S')}")
                    _replace_atomic(path, kept)
                    print(f"[localm] {path.name} could not be read; kept it as "
                          f"{kept.name} and left {bak.name} unchanged.", file=sys.stderr)
                else:
                    _copy_to_backup(path, bak)
            except OSError as e:
                # The .bak snapshot is best-effort: a failed one does NOT fail the
                # primary write (which still proceeds below). _replace_atomic already
                # rode out the transient sharing violation, so reaching here means a
                # PERSISTENT problem (a lock that outlasted the retries, disk full, a
                # real permission error), noted on stderr rather than raised.
                print(f"[localm] note: could not refresh {path.name}.bak ({e}); "
                      "the main write still succeeded.", file=sys.stderr)
        _replace_atomic(tmp, path)
    except BaseException:
        # _replace_atomic consumes tmp on success; on any failure BEFORE that,
        # remove our unique temp so a failed write leaves no orphan behind.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def instance_id() -> str:
    """Stable, unguessable identifier for THIS install's data directory. Minted
    once (uuid4 hex) on first call and reused for the life of the install by
    persisting it to a small file under the data dir; never regenerated on a
    normal restart. References the bare ``HOME_DIR`` global (like
    ``ensure_dirs``), not a path frozen at import, so it always tracks whichever
    data directory is actually configured, including one repointed at runtime.

    Served to the GUI over ``/v1/config``, which compares it against the id it
    last saw for this browser origin to tell a restart of the same install
    apart from a different install sharing that origin's localStorage. The
    value is kept in memory per data folder after the first call, so a file
    that cannot be read or written still gives one id for the whole process.
    A file that exists but cannot be read is never overwritten."""
    home = HOME_DIR
    val = _instance_ids.get(home)
    if val is None:
        val = _instance_ids[home] = _instance_id_now()
    return val


_instance_ids: dict = {}


def _instance_id_now() -> str:
    """``instance_id()`` read (or minted) on the calling thread."""
    ensure_dirs()
    path = HOME_DIR / "instance_id.txt"
    with _io_lock:
        if path.is_file():
            try:
                val = path.read_text(encoding="utf-8").strip()
                if val:
                    return val
            except OSError as e:
                # The file exists but cannot be read: it is left as it is and
                # a fresh id is used in memory.
                print(f"[localm] WARNING: cannot read {path} ({e}); using an "
                      "in-memory id for this run; the file is left unchanged.",
                      file=sys.stderr)
                return uuid.uuid4().hex
        val = uuid.uuid4().hex
        try:
            path.write_text(val, encoding="utf-8")
        except OSError as e:
            print(f"[localm] WARNING: cannot persist instance id to {path} "
                  f"({e}); using an in-memory-only id for this run (it will "
                  "change on the next start).", file=sys.stderr)
        return val


def _drop_stray_credentials(stored: dict) -> None:
    """Remove model-source credential keys (``hf_token``, ``civitai_api_key``)
    from *stored* in place, so a copy in config.json is never read or served
    and is dropped from the file on the next config save. Warns once per
    process per key on stderr, naming the key, never the value."""
    from localm.model_source_credentials import CREDENTIAL_KEYS
    for key in sorted(CREDENTIAL_KEYS & set(stored)):
        stored.pop(key, None)
        if key not in _warned_stray_credentials:
            _warned_stray_credentials.add(key)
            print(f"[localm] {CONFIG_FILE.name} holds {key}, which localm does not "
                  f"read from there; it is ignored and removed on the next settings "
                  f"save. Set it with `localm config {key} <value>` or in "
                  "Settings instead.", file=sys.stderr)


CONFIG_VERSION_KEY = "config_version"

# Stored keys that are legitimate although DEFAULT_CONFIG does not list them.
_EXTRA_KNOWN_CONFIG_KEYS = frozenset({CONFIG_VERSION_KEY, "plugins_first_use_done"})

_warned_unknown_stored_keys: set = set()
_warned_newer_config_version: set = set()


def _config_rename(old: tuple, new: tuple, value_map: Optional[Callable] = None) -> dict:
    return {"kind": "rename", "old": old, "new": new, "value_map": value_map}


def _config_remove(old: tuple) -> dict:
    return {"kind": "remove", "old": old}


def _map_remote_image_value(value):
    mapped = coerce_remote_image_mode(value)
    return value if mapped is None else mapped


# Ordered migration steps; step N (1-based) takes a stored config from version
# N-1 to N. Append only; never reorder or edit a released step. A step is a
# rename (key path -> key path, optional value map) or a removal. A key path is
# a tuple of keys, so a nested key is addressed as ("plugins", "x", "key").
_CONFIG_STEPS: tuple = (
    _config_rename(("gui_proxy_remote_images",), ("gui_proxy_remote_images",),
                   _map_remote_image_value),
)
CURRENT_CONFIG_VERSION = len(_CONFIG_STEPS)


def _stored_config_version(stored: dict) -> int:
    """The ``config_version`` in *stored*; 0 when absent or not a plain
    non-negative integer."""
    raw = stored.get(CONFIG_VERSION_KEY, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _config_path_parent(root: dict, path: tuple):
    node = root
    for part in path[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return None
    return node


def _apply_config_step(stored: dict, step: dict) -> None:
    old = step["old"]
    src = _config_path_parent(stored, old)
    if src is None or old[-1] not in src:
        return
    value = src.pop(old[-1])
    if step["kind"] == "remove":
        return
    new = step["new"]
    value_map = step["value_map"]
    if value_map is not None:
        value = value_map(value)
    node = stored
    for part in new[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    node.setdefault(new[-1], value)


def _apply_config_migrations(stored: dict) -> None:
    """Bring *stored* (the persisted delta) in place up to
    ``CURRENT_CONFIG_VERSION`` by applying the steps above its
    ``config_version``, then stamp the version. A stored version newer than this
    build is left untouched and warned about once per process. An empty *stored*
    is left empty."""
    if not stored:
        return
    version = _stored_config_version(stored)
    if version > CURRENT_CONFIG_VERSION:
        if version not in _warned_newer_config_version:
            _warned_newer_config_version.add(version)
            _logger().warning(
                "%s has config_version %d, newer than this build understands (%d); "
                "it is read as is and not rewritten.",
                CONFIG_FILE.name, version, CURRENT_CONFIG_VERSION)
        return
    for step in _CONFIG_STEPS[version:]:
        _apply_config_step(stored, step)
    stored[CONFIG_VERSION_KEY] = CURRENT_CONFIG_VERSION


def _config_needs_migration(stored) -> bool:
    """True when a migration step would change *stored*'s settings. A file that
    only lacks the version stamp is not rewritten on load; the stamp is written
    with the next save."""
    if not (isinstance(stored, dict) and stored
            and _stored_config_version(stored) < CURRENT_CONFIG_VERSION):
        return False
    migrated = copy.deepcopy(stored)
    _apply_config_migrations(migrated)
    migrated.pop(CONFIG_VERSION_KEY, None)
    original = {k: v for k, v in stored.items() if k != CONFIG_VERSION_KEY}
    return migrated != original


def _warn_unknown_stored_keys(stored: dict) -> None:
    """One WARNING per process per stored top-level key that DEFAULT_CONFIG does
    not define. Names the key, never the value; the key stays in the file."""
    for key in sorted(k for k in stored if isinstance(k, str)):
        if key in DEFAULT_CONFIG or key in _EXTRA_KNOWN_CONFIG_KEYS:
            continue
        if key in _warned_unknown_stored_keys:
            continue
        _warned_unknown_stored_keys.add(key)
        _logger().warning(
            "%s holds the key %.80r, which this build does not use; its value is "
            "ignored and the key is kept in the file.", CONFIG_FILE.name, key)


def _persist_migrated_config() -> None:
    """Rewrite config.json in the migrated form under the store lock. A failure
    is logged and leaves the file as it was; the in-memory config is already
    migrated."""
    try:
        update_config(lambda _cfg: None)
    except Exception as e:
        _logger().warning("%s could not be rewritten after its settings were "
                          "migrated (%s); it is migrated again on the next load.",
                          CONFIG_FILE.name, e)


def _merge_stored_config(cfg: dict, stored) -> None:
    """Overlay the persisted config delta *stored* onto *cfg* (the defaults),
    without any model-source credential key (see _drop_stray_credentials).

    A present-but-non-dict config.json - valid JSON that is a list / string /
    number / null, or any non-object - is ignored so it cannot corrupt the merge,
    but that discard is SURFACED once per process. A MISSING file arrives here
    as the ``{}`` default (a dict) and is a normal no-op, not a warning; a
    genuinely unparseable file already warned in _read_json and also arrives as
    ``{}``. stderr, not the logger (see _detect_home)."""
    if isinstance(stored, dict):
        _drop_stray_credentials(stored)
        _apply_config_migrations(stored)
        _warn_unknown_stored_keys(stored)
        cfg.update(stored)
        return
    key = str(CONFIG_FILE)
    if key not in _warned_bad_config:
        _warned_bad_config.add(key)
        print(f"[localm] config.json is not a JSON object (got "
              f"{type(stored).__name__}); ignoring it and using defaults. The file "
              "is left untouched so you can recover it by hand.", file=sys.stderr)


def load_config_checked() -> tuple:
    """``(config, read_ok)``. ``read_ok`` is False ONLY when config.json (or its
    ``.bak``) was PRESENT and could not be read, so *config* is DEFAULT_CONFIG
    with nothing merged in - the same shape a genuinely absent config.json
    produces (see _read_json_checked). A caller that needs to tell "no config
    yet" apart from "a config exists but is corrupt" (e.g. bug-report
    diagnostics) uses this instead of load_config().

    On a running event loop the read follows ``_loop_safe_read``: at most
    LOOP_READ_WAIT_S of waiting, then the last complete read."""
    return _loop_safe_read("config.json", _load_config_checked_now)


def _load_config_checked_now() -> tuple:
    """``load_config_checked()`` read on the calling thread."""
    from localm import model_ids
    _ensure_dirs_once()
    # DEEP copy: a shallow .copy() shares the nested mutable defaults (e.g. the
    # "plugins" dict) with DEFAULT_CONFIG, so a caller mutating cfg["plugins"][x]
    # (per-plugin media config, workflow selection) would silently corrupt the
    # module-level DEFAULT_CONFIG for the rest of the process.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with _io_lock:
        stored, read_ok = _read_json_checked(CONFIG_FILE, {})
    refs = stored.pop(model_ids.REFS_KEY, None) if isinstance(stored, dict) else None
    needs_migration = read_ok and _config_needs_migration(stored)
    _merge_stored_config(cfg, stored)
    model_ids.config_to_current(cfg, refs, model_index())
    if needs_migration:
        _persist_migrated_config()
    # gui_proxy_remote_images is off/ask/on but config.json may hold a boolean.
    # There is no migration step: this normalises on every read, so a stored
    # `true` reads as "on" for the route, the schema, the CLI and the bug report
    # alike.
    _normalize_remote_image_mode(cfg)
    if "keep_diagnostics" in cfg:
        cfg["keep_diagnostics"] = coerce_keep_diagnostics(cfg["keep_diagnostics"])
    return cfg, read_ok


def load_config() -> dict:
    return load_config_checked()[0]


def remote_image_mode(cfg: Optional[dict] = None) -> str:
    """The current gui_proxy_remote_images mode, ALWAYS one of
    REMOTE_IMAGE_MODES.

    load_config() already normalises the key, so for the ordinary caller this
    only reads it. It coerces again anyway because it is the read a security
    decision is made on, and a *cfg* built some other way (a hand-assembled
    dict, a test double) has been through no normalisation at all. Same
    function both times, so there is one definition of what a value means;
    unreadable resolves to "off", which is the direction that refuses."""
    if cfg is None:
        cfg = load_config()
    mode = coerce_remote_image_mode(cfg.get("gui_proxy_remote_images"))
    return mode if mode is not None else REMOTE_IMAGE_OFF


def coerce_keep_diagnostics(raw) -> bool:
    """*raw* (a stored ``keep_diagnostics`` value) as a bool: True only for
    ``True`` or one of the strings ``1``, ``true``, ``on``, ``yes`` (any case,
    surrounding spaces ignored). Every other value is False; one that is not
    False, None, 0 or one of the strings ``0``, ``false``, ``off``, ``no`` or
    empty is also reported once per process on stderr."""
    if raw is True:
        return True
    if isinstance(raw, str) and raw.strip().lower() in _KEEP_DIAGNOSTICS_ON:
        return True
    if raw in (False, None) or (isinstance(raw, str) and raw.strip().lower() in (
            "", "0", "false", "off", "no")):
        return False
    token = repr(raw)
    if token not in _warned_bad_keep_diagnostics:
        _warned_bad_keep_diagnostics.add(token)
        print(f"[localm] config.json has keep_diagnostics={token}, which is not "
              "true or false; treating it as false (diagnostics stay off in "
              "privacy mode).", file=sys.stderr)
    return False


def keep_diagnostics_enabled() -> bool:
    """Whether to keep diagnostic traces/logs even in privacy mode. Resolved from
    the ``LOCALM_KEEP_DIAGNOSTICS`` env (set by the launcher checkbox / the
    ``--keep-diagnostics`` flag, a per-run override) OR the persistent
    ``keep_diagnostics`` config key (the WebUI Settings > Privacy & data > Keep
    diagnostics for bug reports toggle). So the launcher and the in-app toggle
    both take effect. The config value is read through
    :func:`coerce_keep_diagnostics`. Never raises."""
    if os.environ.get("LOCALM_KEEP_DIAGNOSTICS", "").strip().lower() in (
            _KEEP_DIAGNOSTICS_ON):
        return True
    try:
        return coerce_keep_diagnostics(load_config().get("keep_diagnostics"))
    except Exception:
        return False


def _user_delta(cfg: dict) -> dict:
    """Reduce *cfg* to the keys that need persisting: values that differ from
    the CURRENT DEFAULT_CONFIG, plus keys DEFAULT_CONFIG does not know (a
    config written by a newer version, or version-scoped state such as
    plugins_first_use_done).

    An absent key means "follow the default" and load_config() reconstructs it
    at read time, so a later change to a DEFAULT_CONFIG value reaches existing
    installs.

    Two documented consequences of delta persistence:
      - A value set EQUAL to the current default is indistinguishable from
        "follow the default" and is dropped, so it tracks future default
        changes (the usual user-settings-file semantics).
      - In a config.json written under the older full-dump scheme, a stored key
        equal to the CURRENT default is dropped on the next save, while a
        stored value that DIFFERS (for example a frozen old default such as
        max_tokens 1024) cannot be told apart from a deliberate user choice and
        is KEPT.
    """
    return {k: v for k, v in cfg.items()
            if k not in DEFAULT_CONFIG or v != DEFAULT_CONFIG[k]}


def _config_for_disk(cfg: dict) -> dict:
    """The dict config.json stores for *cfg*: its user-set delta (_user_delta)
    plus the ``model_refs`` of the model names in that delta."""
    from localm import model_ids
    delta = _user_delta(cfg)
    model_ids.attach_refs(delta, model_ids.config_names(delta), model_index())
    return delta


def save_config(cfg: dict) -> None:
    """Persist *cfg* atomically, writing ONLY the user-set delta.

    Callers keep passing the full defaults-merged dict (usually from
    load_config()); keys whose value equals the current default are not
    written to disk and are reconstructed by load_config(), so a shipped
    default-value fix applies to existing installs. See _user_delta for the
    exact rules and the migration of old full-dump files."""
    ensure_dirs()
    with _io_lock:
        delta = _config_for_disk(cfg)
        previous = _read_json(CONFIG_FILE, {})
        _atomic_write_json(CONFIG_FILE, delta)
    _refill_after_write()
    _record_config_saved(previous if isinstance(previous, dict) else {}, delta)


def _changed_config_keys(before: dict, after: dict) -> list:
    """Names of the top-level keys whose value differs between *before* and
    *after*, leaving out the ``config_version`` stamp."""
    return sorted(k for k in set(before) | set(after)
                  if k != CONFIG_VERSION_KEY
                  and before.get(k, _MISSING) != after.get(k, _MISSING))


_MISSING = object()


def _record_config_saved(before: dict, after: dict) -> None:
    """Record a config save in the management log (key NAMES only); nothing when no key changed."""
    changed = _changed_config_keys(before, after)
    if changed:
        mgmt_events.record("config_saved", home=HOME_DIR, keys=changed)


# update_config()/update_registry() are read-modify-write: read the file, let a
# mutator edit the in-memory dict, then write it back. _io_lock only serializes
# that whole cycle within THIS process - a genuinely separate localm OS process
# (the CLI `localm config` racing a running server's PATCH /v1/config, or two CLI
# invocations) has its OWN _io_lock and can interleave its own read-modify-write
# entirely inside this process's window, silently losing whichever change gets
# read-before-written-back last. A lock FILE closes that gap across processes:
# os.open(..., O_CREAT | O_EXCL) is an atomic create-only-if-absent op on both
# Windows and POSIX, so exactly one process at a time can hold the lock.
def update_config(mutator: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write the config.

    Holds _rmw_lock (serializes writers within this process) AND a cross-process
    lock file (serializes separate localm OS processes - e.g. the CLI
    `localm config` racing a running server's PATCH /v1/config, or two CLI
    invocations) across the WHOLE read-modify-write, so no writer, in this
    process or another, can read a stale copy and silently clobber a concurrent
    change. _io_lock is taken only around the read-mutate-write itself, so a
    reader never waits out the cross-process lock wait. *mutator* receives the
    loaded config dict (defaults merged) and edits it in place; the result is
    persisted with a single atomic write. Use this instead of a bare
    load_config()/save_config() pair wherever a lost update would matter."""
    from localm import model_ids
    ensure_dirs()
    # LOCK ORDER: _rmw_lock -> config.json.lock -> _io_lock. See
    # test_reader_is_not_blocked_by_a_writer_waiting_on_the_cross_process_lock.
    with _rmw_lock, _cross_process_lock(CONFIG_FILE) as lock, _io_lock:
        cfg = copy.deepcopy(DEFAULT_CONFIG)   # deep: see load_config (nested dicts)
        stored, read_ok = _read_json_checked(CONFIG_FILE, {})
        if not read_ok:
            # The file EXISTS and could not be read, so `stored` is {} - the same
            # value a genuinely absent config produces. Merging that leaves cfg ==
            # DEFAULT_CONFIG, and _user_delta would then reduce the write to ONLY
            # the key this mutator set, replacing every setting the user has with
            # defaults while the caller reports success. Two of those are silent
            # security downgrades rather than lost preferences: net_mode reverts
            # to "ask" (outbound prompting returns for someone who chose "off")
            # and llama_runtime_pin reverts to "" (a deliberate pin away from a
            # bad upstream build, plus the llama_runtime_history that
            # `setup-llama --rollback` reads). Refusing is recoverable; the
            # overwrite is not. Names the FILE, never the path: this can surface
            # in an HTTP error body (inference/routes/config.py).
            raise ConfigUnreadable(
                f"{CONFIG_FILE.name} exists but could not be read, so saving "
                f"would replace its settings with an older copy or with "
                f"defaults; refused. Fix or remove {CONFIG_FILE.name} (the "
                f"{CONFIG_FILE.name}.bak next to it may hold a good copy).")
        refs = stored.pop(model_ids.REFS_KEY, None) if isinstance(stored, dict) else None
        _merge_stored_config(cfg, stored)
        model_ids.config_to_current(cfg, refs, model_index())
        before = copy.deepcopy(cfg)
        mutator(cfg)
        # The mutator and the return value see the full merged dict; only the
        # user-set delta hits the disk (see _user_delta / save_config).
        lock.assert_held()
        _atomic_write_json(CONFIG_FILE, _config_for_disk(cfg))
    _refill_after_write()
    _record_config_saved(before, cfg)
    return cfg


_model_index_cache: Optional[tuple] = None


def model_index():
    """The :class:`~localm.model_ids.ModelIndex` of registry.json as it is on
    disk now. Cached on the file's path, mtime and size, so an unchanged registry
    costs one stat. Never raises and never writes; an absent or unreadable registry
    gives an empty index. On a running event loop the read follows
    ``_loop_safe_read``."""
    return _loop_safe_read("registry.json index", _model_index_now)


def _model_index_now():
    """``model_index()`` read on the calling thread."""
    from localm import model_ids
    global _model_index_cache
    path = REGISTRY_FILE
    try:
        st = path.stat()
    except OSError:
        return model_ids.EMPTY_INDEX
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _model_index_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    with _io_lock:
        reg = _read_json(path, {})
    index = model_ids.build_index(reg)
    _model_index_cache = (key, index)
    return index


def load_registry() -> dict:
    """The registry, each entry's ``base`` holding its base model's current name.
    Entries without their own ID get one on this read and it is written back. A
    failed write-back is logged and retried by the next read; the entries are then
    returned without an ID. On a running event loop the read follows
    ``_loop_safe_read``."""
    return _loop_safe_read("registry.json", _load_registry_now)


def _load_registry_now() -> dict:
    """``load_registry()`` read on the calling thread."""
    from localm import model_ids
    _ensure_dirs_once()
    with _io_lock:
        reg = _read_json(REGISTRY_FILE, {})
    if not isinstance(reg, dict):
        return {}
    if model_ids.needs_ids(reg):
        try:
            return update_registry(lambda _r: None)
        except (ConfigUnreadable, ConfigLockTimeout, OSError) as e:
            print(f"[localm] could not give models their IDs yet ({e}); "
                  "retrying on the next read.", file=sys.stderr)
    model_ids.follow_bases(reg, model_ids.build_index(reg))
    return reg


def peek_registry() -> dict:
    """The registry as ``load_registry`` returns it, read without creating the
    data dir or the models dir and without writing anything, so an entry may have
    no ID yet. ``{}`` when the file is absent, unreadable or not a JSON object.
    On a running event loop the read follows ``_loop_safe_read``."""
    return _loop_safe_read("registry.json (peek)", _peek_registry_now)


def _peek_registry_now() -> dict:
    """``peek_registry()`` read on the calling thread."""
    from localm import model_ids
    with _io_lock:
        reg = _read_json(REGISTRY_FILE, {})
    if not isinstance(reg, dict):
        return {}
    model_ids.follow_bases(reg, model_ids.build_index(reg))
    return reg


def save_registry(reg: dict) -> None:
    from localm import model_ids
    ensure_dirs()
    model_ids.ensure_ids(reg)
    model_ids.follow_bases(reg, model_ids.build_index(reg))
    with _io_lock:
        _atomic_write_json(REGISTRY_FILE, reg)
    _refill_after_write()


def registry_lock_busy(wait: float = 0.5) -> bool:
    """True when another localm process holds the registry's cross-process lock
    for longer than *wait* seconds. Takes and at once releases the lock and
    writes nothing. A lock that cannot be attempted at all (an OSError other
    than contention) reads as not busy, so the caller's own registry access
    reports the real error."""
    try:
        with _cross_process_lock(REGISTRY_FILE, timeout=wait):
            return False
    except ConfigLockTimeout:
        return True
    except OSError:
        return False


def update_registry(mutator: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write the registry.

    Holds _rmw_lock (serializes writers within this process) AND a cross-process
    lock file (serializes separate localm OS processes - e.g. a CLI `pull`
    running alongside the GUI, or two CLI invocations) across the WHOLE
    read-modify-write, so no writer, in this process or another, can read a
    stale copy and silently clobber a concurrent change (the same closed gap as
    update_config(), see its docstring / _cross_process_lock). _io_lock is taken
    only around the read-mutate-write itself, so a reader never waits out the
    cross-process lock wait. *mutator* receives the registry dict and edits it
    in place; the result is persisted with a single atomic write. Use this
    instead of a bare load_registry()/save_registry() pair wherever a lost
    update would matter. (save_registry() itself remains a blind overwrite -
    last-writer-wins by design, not a read-modify-write, so it needs no lock
    beyond the atomic write it already has.)"""
    from localm import model_ids
    ensure_dirs()
    # LOCK ORDER: _rmw_lock -> registry.json.lock -> _io_lock. See
    # test_reader_is_not_blocked_by_a_writer_waiting_on_the_cross_process_lock.
    with _rmw_lock, _cross_process_lock(REGISTRY_FILE) as lock, _io_lock:
        reg, state = _read_json_state(REGISTRY_FILE, {})
        if state == READ_UNREADABLE:
            # Same refusal as update_config, and the loss here is worse: this
            # writes the WHOLE dict rather than a delta, so an unreadable
            # registry plus one registration leaves a registry.json holding that
            # single model and nothing else. The registry is not reconstructible
            # from anything else on disk.
            raise ConfigUnreadable(
                f"{REGISTRY_FILE.name} exists but could not be read, so saving "
                f"would drop every model registered in it; refused. Fix or "
                f"remove {REGISTRY_FILE.name} (a .bak may hold a good copy).")
        if not isinstance(reg, dict):
            reg = {}
        model_ids.ensure_ids(reg)
        model_ids.follow_bases(reg, model_ids.build_index(reg))
        previous = {name: {k: entry.get(k) for k in ("id", model_ids.FORMER_KEY)}
                    for name, entry in reg.items() if isinstance(entry, dict)}
        mutator(reg)
        model_ids.ensure_ids(reg, previous)
        model_ids.follow_bases(reg, model_ids.build_index(reg))
        lock.assert_held()
        _atomic_write_json(REGISTRY_FILE, reg)
    _refill_after_write()
    return reg


def _loadable_lib_names() -> tuple:
    """Loadable native llama library filename(s) for this platform."""
    if sys.platform == "win32":
        return ("llama.dll",)
    if sys.platform == "darwin":
        return ("libllama.dylib",)
    return ("libllama.so",)


def find_binary_dir() -> Optional[Path]:
    """Return the directory holding the native llama.cpp binaries (llama.dll,
    plus optional llama-cli/llama-server exes), used by `localm info` and
    `localm doctor`, or None when unprovisioned.

    Project-local resolution only: the user-configured binary_dir, then the
    localm-llama-runtime wheel bundled in this venv. No absolute path is ever
    assumed as a default; an unprovisioned install resolves to None and the
    user runs `localm setup-llama` (or sets binary_dir)."""
    cfg = load_config()
    candidates = []
    if cfg.get("binary_dir"):
        candidates.append(Path(cfg["binary_dir"]))
    try:
        import localm_llama_runtime
        d = localm_llama_runtime.lib_dir()
        if d:
            candidates.append(Path(d))
    except Exception:
        pass
    names = _loadable_lib_names()
    for p in candidates:
        try:
            if any((p / n).exists() for n in names):
                return p
        except OSError:
            continue
    return None


def get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
#  The Network access switch (net_mode). netpolicy builds the full policy on   #
#  it; units below netpolicy in the layering (bugreport) read it here.         #
# --------------------------------------------------------------------------- #

NET_MODES = ("off", "ask", "allow")
NET_MODE_ENV_VAR = "LOCALM_NET_MODE"

#: How to turn network access back on, worded for every surface (GUI, CLI,
#: model): the Settings field and the command.
NET_OFF_REMEDY = ("Turn it on in Settings > Server & network > Network access, "
                  "or run: localm config net_mode ask")


def net_off_lead(what: str = "") -> str:
    """The first part of a net_mode=off refusal, without a remedy: that network
    access is off and, when given, *what* was refused."""
    return (f"Network access is off (net_mode=off), so {what}" if what
            else "Network access is off (net_mode=off)")


def net_off_refusal(what: str = "") -> str:
    """The text of a net_mode=off refusal: *what* was refused (optional) and
    :data:`NET_OFF_REMEDY`."""
    return f"{net_off_lead(what)}. {NET_OFF_REMEDY}."


#: How to let a download through while network access is off.
NET_OFF_DOWNLOAD_REMEDY = (
    "Turn it on, or turn on \"Allow model downloads while network access is off\", "
    "in Settings > Server & network; or run: localm config net_mode ask (or: "
    "localm config net_allow_model_downloads true)")


def net_off_download_refusal(what: str) -> str:
    """The text of a net_mode=off refusal of a download: *what* was refused and
    :data:`NET_OFF_DOWNLOAD_REMEDY`."""
    return f"{net_off_lead(what)}. {NET_OFF_DOWNLOAD_REMEDY}."


#: How to let the update channel through while network access is off.
NET_OFF_UPDATE_REMEDY = (
    "Turn it on in Settings > Server & network > Network access, or turn on "
    "\"Check for updates even when network access is off\" in Settings > System > "
    "Updates; or run: localm config net_mode ask (or: localm config "
    "update_ignore_net_policy true)")


def net_off_update_refusal(what: str) -> str:
    """The text of a net_mode=off refusal on the update channel: *what* was
    refused and :data:`NET_OFF_UPDATE_REMEDY`."""
    return f"{net_off_lead(what)}. {NET_OFF_UPDATE_REMEDY}."


def network_mode() -> str:
    """Resolve the active mode: LOCALM_NET_MODE env > config > "ask".

    A config-read failure resolves to "off", not "ask", and logs a warning.
    That covers both a load that raises and a config.json (with its .bak) that
    is present but cannot be read, which load_config_checked reports as
    read_ok False. An unset or unrecognised config value resolves to "ask"."""
    return network_mode_checked()[0]


def network_mode_checked() -> tuple:
    """``(mode, unreadable)``: *mode* as :func:`network_mode` resolves it, and
    *unreadable* True when it is ``off`` only because the config file could
    not be read (no ``LOCALM_NET_MODE`` override)."""
    import logging
    logger = logging.getLogger("localm")
    env = os.environ.get(NET_MODE_ENV_VAR, "").strip().lower()
    if env in NET_MODES:
        return env, False
    try:
        cfg, read_ok = load_config_checked()
    except Exception as exc:
        logger.warning("netpolicy: could not load config (%s); resolving "
                       "net_mode to 'off' (fail-safe) so an unreadable config "
                       "cannot silently re-enable network access", exc)
        return "off", True
    if not read_ok:
        logger.warning("netpolicy: config.json is present but unreadable; "
                       "resolving net_mode to 'off' (fail-safe) so an "
                       "unreadable config cannot silently re-enable network "
                       "access")
        return "off", True
    mode = str(cfg.get("net_mode", "ask")).strip().lower()
    return (mode if mode in NET_MODES else "ask"), False


def net_off_unreadable_config_refusal(what: str = "") -> str:
    """The text of a refusal made because the config file could not be read,
    which turns network access off: the file, *what* was refused (optional),
    and what to do."""
    lead = f"{CONFIG_FILE} could not be read, so network access is treated as off"
    return (f"{lead} and {what}. " if what else f"{lead}. ") + (
        "Correct that file (or set LOCALM_NET_MODE for this run) and try again.")
