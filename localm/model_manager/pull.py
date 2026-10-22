# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model download/transport: pull routing, the HF + URL + Ollama backends,
resumable downloads, hashing-on-the-wire, and GUI progress streaming."""

import localm.model_manager as _mm  # read package-patchable names at call time
from localm import instances

import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn
from ..debuglog import logger
from ._shared import _emit_progress
from ._shared import _verify_digest
from ._shared import console
from ._transfer import METER
from ._transfer import RateWindow
from ._transfer import describe as _describe_progress
from ._transfer import eta_seconds
from ._transfer import format_duration
from ._transfer import format_rate
from ._transfer import format_size
from ._transfer import meter_tqdm_class
from .gguf import _safe_models_filename
from .gguf import first_split_part
from .gguf import gguf_n_embd
from .gguf import split_gguf_parts
from .registry import _detect_local_model_type, _media_component_type, _sanitize_name
from .registry import alias_model
from .registry import find_aliases_by_path

# HF_ENDPOINT / HF_HUB_ENDPOINT are ambient env vars localm never exposes as a
# setting anywhere (no settings_schema.py key, no CLI flag, no docs). Pinned
# explicitly at every huggingface_hub call site below so a stray env var in
# the user's shell can never silently redirect a model pull elsewhere.
_HF_ENDPOINT = "https://huggingface.co"


def _hf_token() -> Optional[str]:
    """The Hugging Face token saved in localm's settings, else the HF_TOKEN
    environment variable, else None (huggingface_hub then falls back to its own
    saved login)."""
    from ..model_source_credentials import get_hf_token
    return get_hf_token()


def _hf_api():
    """A ``huggingface_hub.HfApi`` on the pinned endpoint that sends the
    configured token."""
    from huggingface_hub import HfApi
    return HfApi(endpoint=_HF_ENDPOINT, token=_hf_token())


_PARTIAL_OWNER_SUFFIX = ".owner"
_INFLIGHT_PARTIALS: "set[str]" = set()
_INFLIGHT_LOCK = threading.Lock()


def _partial_owner_path(partial: Path) -> Path:
    """The sidecar file recording which process owns *partial*."""
    return partial.with_name(partial.name + _PARTIAL_OWNER_SUFFIX)


def _pid_space_id() -> str:
    """An opaque id for the pid table this process's pids belong to; see
    :func:`localm.instances.pid_space_id`."""
    return instances.pid_space_id()


def _same_pid_space(rec: dict, where: Path) -> bool:
    """True when owner record *rec*, stored in directory *where*, was written
    in this process's pid space: its id is this one (see
    :func:`_pid_space_id`); or it is the id an earlier localm wrote on this
    host (see :func:`localm.instances.legacy_pid_space_id` and
    :func:`localm.instances.legacy_record_is_this_host`); or its host id
    equals this one (see :func:`localm.instances.host_id`, which does not
    depend on the host name), its ``local`` field is True (its writer saw
    *where* on a disk attached to the writer's machine) and *where* is on a
    disk attached to this machine (see :func:`localm.instances.path_is_local`).
    A record whose ``local`` field is missing or not True is never matched by
    host id. A record naming no pid space is not judged here."""
    space = rec.get("space")
    if space == _pid_space_id():
        return True
    if (space is not None and space == instances.legacy_pid_space_id()
            and instances.legacy_record_is_this_host(rec)):
        return True
    host = rec.get("host")
    return (space is not None and isinstance(host, str)
            and host == instances.host_id() and rec.get("local") is True
            and instances.path_is_local(where))


def _write_partial_owner(partial: Path) -> None:
    """Record this process (pid, pid space, host id, start identity, and
    whether the folder of *partial* is on a disk attached to this machine) as
    the owner of *partial*."""
    rec = {"pid": os.getpid(), "space": _pid_space_id(),
           "host": instances.host_id(),
           "local": instances.path_is_local(partial.parent),
           "start": instances.process_start_identity(os.getpid())}
    _partial_owner_path(partial).write_text(json.dumps(rec), encoding="utf-8")


def _read_partial_owner(partial: Path) -> "dict | None":
    """The owner record of *partial*, or None when absent or unreadable."""
    try:
        rec = json.loads(_partial_owner_path(partial).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return rec if isinstance(rec, dict) else None


def _unlink_quiet(p: Path) -> None:
    """Remove *p* if present; a failure to remove is not an error here."""
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _partial_in_flight(partial: Path) -> bool:
    """True while a download in this process holds *partial* open."""
    with _INFLIGHT_LOCK:
        return str(partial) in _INFLIGHT_PARTIALS


def _partial_owner_is_gone(partial: Path, *, lock_held: bool = False) -> bool:
    """True only when the recorded owner of *partial* is PROVEN dead.

    Returns False for an unparsable pid, a record written in another pid space
    (see :func:`_same_pid_space`), a live pid whose start identity matches the
    record or cannot be compared with it (see
    :func:`localm.instances.start_identity_differs`), a live pid in a record
    that names no pid space, and any probe failure. A pid that is this
    process's own is gone when no download in this process currently holds
    the file.

    A missing or unreadable sidecar (a partial written by huggingface_hub
    itself or by a localm that wrote none) counts as gone only when
    *lock_held* is True: the caller holds huggingface_hub's per-file download
    lock for this blob, which every huggingface_hub writer of the file holds
    for the whole transfer, so no other writer is appending to it.
    """
    rec = _read_partial_owner(partial)
    if rec is None:
        return lock_held
    try:
        pid = int(rec.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    space = rec.get("space")
    if space is not None and not _same_pid_space(rec, partial.parent):
        return False
    if pid == os.getpid():
        return not _partial_in_flight(partial)
    if not instances.pid_alive(pid):
        return True
    return space is not None and instances.start_identity_differs(
        rec.get("start"), instances.process_start_identity(pid))


def _partial_candidates(incomplete_path: Path) -> "list[Path]":
    """Every temp file for the blob *incomplete_path* names: the per-process
    ``<stem>.<id>.incomplete`` files and a plain ``<stem>.incomplete``."""
    stem = incomplete_path.stem
    out = []
    try:
        for p in incomplete_path.parent.iterdir():
            n = p.name
            if not n.endswith(".incomplete"):
                continue
            if (n == incomplete_path.name or n.startswith(stem + ".")) and p.is_file():
                out.append(p)
    except OSError:
        pass
    return out


def _partial_size(p: Path) -> int:
    """Size of *p* in bytes, or -1 when it cannot be read."""
    try:
        return p.stat().st_size
    except OSError:
        return -1


def _reusable_partial_bytes(incomplete_path: Path,
                            expected_size: "int | None" = None, *,
                            lock_free: bool = False) -> int:
    """Bytes a download of this blob would resume from: the size of the largest
    candidate that this process holds or whose owner is proven gone (see
    :func:`_partial_owner_is_gone`; *lock_free* True means huggingface_hub's
    per-file download lock for this blob is free right now, so a partial with
    no owner record has no writer). Candidates larger than a known
    *expected_size* are not counted."""
    best = 0
    for c in _partial_candidates(incomplete_path):
        if not (_partial_in_flight(c)
                or _partial_owner_is_gone(c, lock_held=lock_free)):
            continue
        n = _partial_size(c)
        if n < 0:
            continue
        if expected_size is not None and expected_size > 0 and n > expected_size:
            continue
        best = max(best, n)
    return best


def _adopt_partial(incomplete_path: Path, tmp_path: Path,
                   expected_size: "int | None", adopt: bool = True) -> int:
    """Rename the largest usable proven-orphan candidate for this blob onto
    *tmp_path* and delete every other proven-orphan candidate; with *adopt*
    False every proven-orphan candidate is deleted instead. Candidates whose
    owner is not proven gone are left untouched. The caller holds
    huggingface_hub's download lock for this blob, so a candidate with no
    owner record is adoptable. Returns the adopted size, 0 when nothing was
    adopted."""
    adoptable = []
    for c in _partial_candidates(incomplete_path):
        if c == tmp_path or not _partial_owner_is_gone(c, lock_held=True):
            continue
        n = _partial_size(c)
        if n >= 0:
            adoptable.append((n, c))
    adoptable.sort(key=lambda t: t[0], reverse=True)
    adopted = 0
    for n, c in adoptable:
        usable = adopt and n > 0 and (expected_size is None or expected_size <= 0
                                      or n <= expected_size)
        if adopted == 0 and usable:
            try:
                c.rename(tmp_path)
            except OSError:
                continue
            adopted = n
        else:
            try:
                c.unlink()
            except OSError:
                continue
        _unlink_quiet(_partial_owner_path(c))
    return adopted


def _in_local_dir_layout(path: Path) -> bool:
    """True when *path* sits in a ``<local_dir>/.cache/huggingface/download``
    folder, where huggingface_hub's download lock is per destination file
    rather than per etag."""
    parts = path.parent.parts[-3:]
    return len(parts) == 3 and parts == (".cache", "huggingface", "download")


def _reap_stale_etag_partials(incomplete_path: Path) -> None:
    """Delete proven-orphan temp files of the same destination file but a
    different etag. Files whose owner is not proven gone are left alone. In a
    local-dir folder the caller holds the destination file's download lock, so
    a temp file with no owner record is deleted too."""
    lock_held = _in_local_dir_layout(incomplete_path)
    name = incomplete_path.name
    short = name.split(".")[0]
    stem = incomplete_path.stem
    try:
        entries = list(incomplete_path.parent.iterdir())
    except OSError:
        return
    for p in entries:
        n = p.name
        if not n.endswith(".incomplete") or not n.startswith(short + "."):
            continue
        if n == name or n.startswith(stem + "."):
            continue
        if not _partial_owner_is_gone(p, lock_held=lock_held):
            continue
        try:
            p.unlink()
        except OSError:
            continue
        _unlink_quiet(_partial_owner_path(p))


def _resumable_download_to_tmp_and_move(
    incomplete_path: Path,
    destination_path: Path,
    url_to_download: str,
    headers: dict,
    expected_size: Optional[int],
    filename: str,
    force_download: bool,
    etag: Optional[str],
    xet_file_data: Any,
    tqdm_class: Any = None,
) -> None:
    """Download into a process-owned temp file, resuming from a partial whose
    owner is proven gone, then move the result into place.

    The temp file is ``<stem>.<id>.incomplete`` beside *incomplete_path* with
    a ``.owner`` sidecar naming this process. A partial whose owner is not
    proven gone (live, unknown, or no sidecar) is never read, renamed or
    deleted. Every transfer goes over HTTP, a resume with a Range request, so
    the temp file holds every byte received when the transfer stops; a file
    stored with Xet is fetched over HTTP too, except a first transfer larger
    than huggingface_hub's HTTP limit (``constants.MAX_HTTP_DOWNLOAD_SIZE``),
    which goes through Xet and keeps only what Xet has written when it stops.
    See test_a_stopped_first_download_of_a_xet_file_keeps_the_bytes_received. On
    a failed transfer the temp file and its sidecar are kept for the next
    attempt, and a failure before the temp file exists removes the sidecar; a
    failed size check deletes both.
    """
    import huggingface_hub.file_download as fd
    import uuid

    if destination_path.exists() and not force_download:
        return

    incomplete_path = Path(incomplete_path)
    incomplete_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = incomplete_path.with_name(
        f"{incomplete_path.stem}.{uuid.uuid4().hex[:8]}.incomplete")

    with _INFLIGHT_LOCK:
        _INFLIGHT_PARTIALS.add(str(tmp_path))
    try:
        _write_partial_owner(tmp_path)
        resume_size = _adopt_partial(incomplete_path, tmp_path, expected_size,
                                     adopt=not force_download)
        _reap_stale_etag_partials(incomplete_path)

        needed = max(0, (expected_size or 0) - resume_size)
        if needed > 0:
            fd._check_disk_space(needed, tmp_path.parent)
            fd._check_disk_space(needed, destination_path.parent)

        if (resume_size == 0 and xet_file_data is not None and expected_size
                and expected_size > fd.constants.MAX_HTTP_DOWNLOAD_SIZE
                and fd.is_xet_available()):
            fd.xet_get(
                incomplete_path=tmp_path,
                xet_file_data=xet_file_data,
                headers=headers,
                expected_size=expected_size,
                displayed_filename=filename,
                tqdm_class=tqdm_class,
            )
        else:
            mode = "r+b" if resume_size > 0 else "wb"
            try:
                with tmp_path.open(mode) as f:
                    if resume_size > 0:
                        f.seek(resume_size)
                    fd.http_get(
                        url_to_download,
                        f,
                        resume_size=resume_size,
                        headers=headers,
                        expected_size=expected_size,
                        displayed_filename=filename,
                        tqdm_class=tqdm_class,
                    )
            except OSError as e:
                if "Consistency check failed" in str(e):
                    _unlink_quiet(tmp_path)
                    _unlink_quiet(_partial_owner_path(tmp_path))
                raise

        fd._chmod_and_move(tmp_path, destination_path)
        _unlink_quiet(_partial_owner_path(tmp_path))
    finally:
        if not os.path.lexists(tmp_path):
            _unlink_quiet(_partial_owner_path(tmp_path))
        with _INFLIGHT_LOCK:
            _INFLIGHT_PARTIALS.discard(str(tmp_path))


_HF_PLACEMENT_NOTED = False


def _note_user_hf_placement() -> None:
    """Log once per process, at INFO, which Hugging Face cache locations come
    from the user's environment (and so sit outside the data folder)."""
    global _HF_PLACEMENT_NOTED
    if _HF_PLACEMENT_NOTED:
        return
    _HF_PLACEMENT_NOTED = True
    from ..config import hf_cache_user_placement
    placed = hf_cache_user_placement()
    if placed:
        logger.info(
            "Hugging Face caches use the location set in the environment, "
            "outside the data folder: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(placed.items())))


def _ensure_hf_resumable_download() -> None:
    """Wire _resumable_download_to_tmp_and_move into huggingface_hub.file_download.

    Called from the pull entry points, never at import, so a process that
    never pulls keeps huggingface_hub's own download behaviour.
    """
    _note_user_hf_placement()
    try:
        import huggingface_hub.file_download as fd
        if getattr(fd, "_localm_resumable_patched", False):
            return
        fd._orig_download_to_tmp_and_move = fd._download_to_tmp_and_move
        fd._download_to_tmp_and_move = _resumable_download_to_tmp_and_move
        fd._localm_resumable_patched = True
    except (ImportError, AttributeError):
        pass


def _hf_incomplete_path(base_dir: Path, rel: str, etag: str) -> "Path | None":
    """The temp-file path huggingface_hub uses for *rel* under *base_dir* at
    *etag*, or None when that layout cannot be computed."""
    try:
        from huggingface_hub._local_folder import get_local_download_paths
        return get_local_download_paths(Path(base_dir), rel).incomplete_path(etag)
    except Exception as e:
        logger.debug("cannot compute the .incomplete path for %s: %s", rel, e)
        return None


def _lock_file_is_free(lock_path: Path) -> bool:
    """True when nobody holds the file lock at *lock_path* right now. False when
    it is held, or when the probe cannot tell."""
    try:
        from filelock import FileLock, Timeout
        lock = FileLock(str(lock_path), timeout=0)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return False
        lock.release()
        return True
    except Exception as e:
        logger.debug("cannot probe the download lock %s: %s", lock_path, e)
        return False


def _hf_blob_lock_is_free(base_dir: Path, rel: str) -> bool:
    """True when huggingface_hub's per-file download lock for *rel* under
    *base_dir* is free right now: no download of that file is running, so a
    temp file of it with no owner record has no writer."""
    try:
        from huggingface_hub._local_folder import get_local_download_paths
        return _lock_file_is_free(get_local_download_paths(Path(base_dir), rel).lock_path)
    except Exception as e:
        logger.debug("cannot compute the download lock path for %s: %s", rel, e)
        return False


_REPO_PATH_PART_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")


# The folder under a destination that a download of a file inside a repo
# folder is staged in before it is moved into the destination.
_REPO_FOLDER_STAGING = Path(".cache") / "localm-staging"


def _safe_repo_subdir(remote_dir: str, base_dir: Path) -> bool:
    """True when *remote_dir* (the folder part of an ``owner/repo:dir/file``
    spec) is a plain relative path whose every component matches
    ``_REPO_PATH_PART_RE`` and is not ``.``/``..``, and which stays inside
    *base_dir* when joined to it and resolved."""
    comps = remote_dir.split("/")
    if not comps or any(c in ("", ".", "..") or not _REPO_PATH_PART_RE.match(c)
                        for c in comps):
        return False
    try:
        base = Path(base_dir).resolve()
        return (base / remote_dir).resolve().is_relative_to(base)
    except (OSError, ValueError):
        return False


def _etag_from_head(resp: Any) -> "str | None":
    """The etag huggingface_hub will name a download by, read from the first
    response of a redirect-following HEAD (X-Linked-Etag, then ETag)."""
    try:
        import huggingface_hub.file_download as fd
        history = list(getattr(resp, "history", None) or [])
        first = history[0] if history else resp
        h = getattr(first, "headers", None) or {}
        raw = h.get("X-Linked-Etag") or h.get("ETag")
        return fd._normalize_etag(raw) if raw else None
    except Exception:
        return None




def _progress_file_info(target_parts: list[Path]) -> "tuple[str | None, int, int]":
    """(current-file name, 1-based index, count) for a multi-part download, derived
    from which parts have already landed at their final path - the first one not yet
    present is the file currently downloading. Cheap existence checks only. Returns
    (None, 0, 0) for a single-file download (nothing to disambiguate)."""
    n = len(target_parts)
    if n <= 1:
        return (None, 0, 0)
    done = sum(1 for p in target_parts if p.exists())
    cur = next((p for p in target_parts if not p.exists()), target_parts[-1])
    return (cur.name, min(done + 1, n), n)




class _ProgressOutcome:
    """Explicit success signal for the progress context managers.

    A context manager cannot infer success from the absence of an exception:
    ``_pull_gguf_file`` reports a failed part with ``return False`` from INSIDE
    its ``with`` block, which unwinds cleanly. The body has to SAY it finished,
    and silence means it did not.
    """

    __slots__ = ("succeeded",)

    def __init__(self) -> None:
        self.succeeded = False

    def ok(self) -> None:
        self.succeeded = True


def _incomplete_prefixes(base_dir: Path, rel_parts: list[str],
                         etags: "dict[str, str | None] | None" = None,
                         ) -> "dict[str, str | None] | None":
    """Filename prefixes of the ``.incomplete`` temp files for *rel_parts*,
    mapped to the etag each part is expected at (None when unknown).

    huggingface_hub names a local-dir temp file
    ``<short_hash(<name>.metadata)>.<etag>.<id>.incomplete`` under
    ``<local_dir>/.cache/huggingface/download/<subpath>/``. The hash prefix is
    what separates OUR parts from a concurrent pull's; the etag, when *etags*
    supplies it, separates the current upload from a stale one.

    Returns None when the layout cannot be computed - this reaches into
    huggingface_hub internals, so a version that moves them must degrade rather
    than break. The caller then falls back to an unfiltered scan of this
    destination, which is coarser but never counts another DESTINATION's bytes.
    """
    try:
        from huggingface_hub._local_folder import _short_hash
        from huggingface_hub._local_folder import get_local_download_paths
        out: dict[str, str | None] = {}
        for rel in rel_parts:
            paths = get_local_download_paths(Path(base_dir), rel)
            out[_short_hash(paths.metadata_path.name)] = (etags or {}).get(rel)
        return out or None
    except Exception as e:
        # Not fatal and not silenced: progress simply gets coarser, and the
        # cause is logged at debug.
        logger.debug("cannot compute .incomplete prefixes (progress will be "
                     "coarser, not wrong): %s", e)
        return None


def _download_lock_paths(base_dir: Path, rel_parts: list[str]) -> "dict[str, Path]":
    """The download-lock path of each of *rel_parts*, keyed by the same
    short-hash prefix as :func:`_incomplete_prefixes`; empty when the layout
    cannot be computed."""
    try:
        from huggingface_hub._local_folder import _short_hash
        from huggingface_hub._local_folder import get_local_download_paths
        out: dict[str, Path] = {}
        for rel in rel_parts:
            paths = get_local_download_paths(Path(base_dir), rel)
            out[_short_hash(paths.metadata_path.name)] = paths.lock_path
        return out
    except Exception as e:
        logger.debug("cannot compute the download lock paths: %s", e)
        return {}


_SAVED_LAG_MIN = 16_000_000
_PROGRESS_POLL_S = 0.7

# Seconds between the progress lines handed to a progress listener.
_LISTENER_EVERY_S = 2.0

# Seconds between the plain progress lines written to stderr when stdout is not
# a terminal.
_PLAIN_EVERY_S = 5.0

_PROGRESS_LISTENER: "Optional[Callable[[str], None]]" = None


@contextlib.contextmanager
def progress_listener(callback: "Callable[[str], None]"):
    """For the duration of the block, hand a one-line description of the running
    download's progress to *callback*, about every two seconds. The MCP server
    uses it to pass pull progress to its client."""
    global _PROGRESS_LISTENER
    previous = _PROGRESS_LISTENER
    _PROGRESS_LISTENER = callback
    try:
        yield
    finally:
        _PROGRESS_LISTENER = previous


def _progress_wanted() -> bool:
    """True when something shows download progress: the GUI job runner, a
    terminal, a progress listener, or stderr (plain progress lines)."""
    return (os.environ.get("LOCALM_PROGRESS_JSON") == "1"
            or _PROGRESS_LISTENER is not None
            or console.is_terminal
            or sys.stderr is not None)


class _TerminalBar:
    """A one-line progress bar on the terminal showing percent, bytes, speed and
    time left. It is removed when the download ends."""

    def __init__(self, label: str) -> None:
        from rich.markup import escape
        self._label = escape(label)
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[detail]}"),
            console=console, transient=True, auto_refresh=False)
        self._task = None

    def start(self) -> None:
        self._progress.start()
        self._task = self._progress.add_task(self._label, total=None, detail="")

    def update(self, downloaded: int, total: int, rate: "float | None",
               eta: "int | None", saved: "int | None" = None) -> None:
        detail = format_size(downloaded)
        if total:
            detail += f" / {format_size(total)}"
        if rate is not None:
            detail += f"  {format_rate(rate)}"
        if eta is not None:
            detail += f"  ETA {format_duration(eta)}"
        if saved is not None:
            detail += f"  {format_size(saved)} saved"
        task = self._task
        if task is None:
            return
        if total:
            self._progress.update(task, total=total)
        self._progress.update(task, completed=downloaded, detail=detail, refresh=True)

    def stop(self) -> None:
        self._progress.stop()


class _ByteTracker:
    """Combines the bytes on disk with the bytes the transfer meter counted into
    one downloaded figure, a transfer rate and the time left.

    The figure is the larger of the bytes on disk and the bytes already on disk
    when the download started plus the bytes received since, so it keeps moving
    while a Xet download holds data back from the file. The rate is the network
    rate; a download that reports no network bytes gets the rate its disk size
    grows at.

    ``saved`` is the bytes on disk from the last sample when the figure is at
    least ``_SAVED_LAG_MIN`` ahead of them, else None."""

    def __init__(self, total: int, base: int) -> None:
        self._total = total
        self._base = base
        self._disk_rate = RateWindow()
        self.saved: int | None = None

    def sample(self, on_disk: int) -> "tuple[int, float | None, int | None]":
        self._base = min(self._base, on_disk)
        downloaded = max(on_disk, self._base + METER.transferred)
        if self._total:
            downloaded = min(downloaded, self._total)
        self.saved = on_disk if downloaded - on_disk >= _SAVED_LAG_MIN else None
        rate = METER.rate()
        if rate is None:
            self._disk_rate.add(downloaded)
            rate = self._disk_rate.rate()
        return downloaded, rate, eta_seconds(downloaded, self._total, rate)


class _ProgressRun:
    """Reports one download's progress while it runs: frames for the GUI job
    runner (LOCALM_PROGRESS_JSON=1), a bar on a terminal, lines for a progress
    listener, and otherwise (stdout not a terminal) a plain line on stderr every
    few seconds. Does nothing when none of them is wanted."""

    def __init__(self, measure, total: int, *, parts: "list[Path] | None" = None,
                 label: str = "") -> None:
        self._measure = measure
        self._total = total
        self._parts = parts or []
        self._label = label
        self._gui = os.environ.get("LOCALM_PROGRESS_JSON") == "1"
        self._listener = _PROGRESS_LISTENER
        self._bar = (_TerminalBar(label)
                     if not self._gui and console.is_terminal else None)
        self._plain = (not self._gui and self._listener is None and self._bar is None
                       and sys.stderr is not None)
        self.active = (self._gui or self._listener is not None or self._bar is not None
                       or self._plain)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker: _ByteTracker | None = None
        self._last_key: tuple | None = None
        self._last_listener_at = 0.0
        self._last_plain_at = 0.0
        self._failed = False

    def start(self) -> None:
        if not self.active:
            return
        METER.reset()
        self._tracker = _ByteTracker(self._total, self._measure())
        if self._bar is not None:
            self._bar.start()
        try:
            self._tick()
        except Exception as e:
            self._failed = True
            logger.warning("download progress reporting stopped: %s", e)
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(_PROGRESS_POLL_S):
            try:
                self._tick()
            except Exception as e:
                if not self._failed:
                    self._failed = True
                    logger.warning("download progress reporting stopped: %s", e)
                return

    def _tick(self) -> None:
        tracker = self._tracker
        if tracker is None:
            return
        downloaded, rate, eta = tracker.sample(self._measure())
        if self._gui:
            key = (downloaded, None if rate is None else int(rate // 1000))
            if key != self._last_key:
                self._last_key = key
                fn, fi, fc = _progress_file_info(self._parts)
                _emit_progress(downloaded, self._total, name=fn, index=fi, count=fc,
                               zero_is_unknown=True, rate=rate, eta=eta,
                               saved=tracker.saved)
        if self._bar is not None:
            self._bar.update(downloaded, self._total, rate, eta, tracker.saved)
        if self._listener is not None:
            now = time.monotonic()
            if now - self._last_listener_at >= _LISTENER_EVERY_S:
                self._last_listener_at = now
                self._listener(_describe_progress(downloaded, self._total, rate,
                                                  self._label, tracker.saved))
        if self._plain:
            now = time.monotonic()
            if now - self._last_plain_at >= _PLAIN_EVERY_S:
                self._last_plain_at = now
                print(_describe_progress(downloaded, self._total, rate, self._label,
                                         tracker.saved),
                      file=sys.stderr, flush=True)

    def finish(self, succeeded: bool) -> None:
        if not self.active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._bar is not None:
            self._bar.stop()
        if self._gui:
            fn, fi, fc = _progress_file_info(self._parts)
            final = self._total if (succeeded and self._total) else self._measure()
            _emit_progress(final, self._total, name=fn, index=fi, count=fc)


_STOPPED: "tuple[int, int] | None" = None


def stopped_download() -> "tuple[int, int] | None":
    """``(bytes saved, total bytes)`` of the download in this process that a
    KeyboardInterrupt stopped most recently, or None when no download was
    stopped or its saved bytes could not be measured. The total is 0 when it
    is unknown. The saved bytes are what a rerun of the same pull reuses."""
    return _STOPPED


def forget_stopped_download() -> None:
    """Clear the record :func:`stopped_download` returns."""
    global _STOPPED
    _STOPPED = None


@contextlib.contextmanager
def _record_stop(measure, total_size: int):
    """On a KeyboardInterrupt in the body, record ``(measure(), total_size)``
    for :func:`stopped_download` and re-raise."""
    global _STOPPED
    _STOPPED = None
    try:
        yield
    except KeyboardInterrupt:
        try:
            _STOPPED = (int(measure()), int(total_size or 0))
        except Exception as e:
            logger.warning("could not measure the saved part of the stopped "
                           "download: %s", e)
        raise


def _job_bytes_measure(target_parts: list[Path], total_size: int, *,
                       base_dir: "Path | None" = None,
                       rel_parts: "list[str] | None" = None,
                       etags: "dict[str, str | None] | None" = None):
    """A function returning the bytes of a download job on disk: the finished
    *target_parts* plus the ``.incomplete`` temp files under *base_dir* that
    :func:`_download_progress` counts, clamped to *total_size* when known."""
    cache_root = (Path(base_dir) if base_dir is not None else _mm.MODELS_DIR) / ".cache"
    prefixes = _incomplete_prefixes(Path(base_dir), rel_parts, etags) \
        if base_dir is not None and rel_parts else None
    lock_paths = _download_lock_paths(Path(base_dir), rel_parts) \
        if base_dir is not None and rel_parts else {}

    def _downloaded_bytes() -> int:
        done = 0
        for p in target_parts:
            try:
                done += p.stat().st_size
            except OSError:
                pass  # not finished yet
        largest: dict[str, int] = {}
        if cache_root.is_dir():
            try:
                for f in cache_root.rglob("*.incomplete"):
                    # Scoped to THIS job, so a concurrent pull's temp file is
                    # never added to this job's numerator.
                    segs = f.name.split(".")
                    if prefixes is not None:
                        if segs[0] not in prefixes:
                            continue
                        want = prefixes[segs[0]]
                        if want is not None and (len(segs) < 2 or segs[1] != want):
                            continue
                    lock_path = lock_paths.get(segs[0])
                    no_writer = (lock_path is not None
                                 and _read_partial_owner(f) is None
                                 and _lock_file_is_free(lock_path))
                    if not (_partial_in_flight(f)
                            or _partial_owner_is_gone(f, lock_held=no_writer)):
                        continue
                    n = _partial_size(f)
                    if n < 0:
                        continue
                    key = ".".join(segs[:2])
                    largest[key] = max(largest.get(key, 0), n)
            except OSError:
                pass
        active = sum(largest.values())
        total_now = done + active
        # Only clamp against a total that is actually known: min(x, 0) is 0,
        # which would pin an indeterminate download's byte count at zero.
        return min(total_now, total_size) if total_size else total_now

    return _downloaded_bytes


@contextlib.contextmanager
def _download_progress(target_parts: list[Path], total_size: int, *,
                       base_dir: "Path | None" = None,
                       rel_parts: "list[str] | None" = None,
                       etags: "dict[str, str | None] | None" = None):
    """Report progress while files land under *base_dir*: JSON frames in GUI
    mode (LOCALM_PROGRESS_JSON=1), a bar on a terminal, and lines for a
    progress listener. A total of 0 means the size could not be determined:
    progress still streams with ``pct: null`` so the GUI shows a busy bar with
    a running byte count, matching _snapshot_progress.

    In-flight bytes are counted from the ``.incomplete`` temp files of THIS
    job's parts (and, when *etags* names them, of the current etag only) that
    this process holds or whose owner is proven gone: the largest such file
    per part, which is the one a resume reuses. The download's transfer rate
    comes from the bytes received (see :mod:`._transfer`).

    Yields a _ProgressOutcome; call ``.ok()`` on the success path or the closing
    event reports the measured partial instead of 100%.
    """
    outcome = _ProgressOutcome()
    _downloaded_bytes = _job_bytes_measure(target_parts, total_size, base_dir=base_dir,
                                           rel_parts=rel_parts, etags=etags)

    if not _progress_wanted():
        with _record_stop(_downloaded_bytes, total_size):
            yield outcome
        return

    label = target_parts[0].name if target_parts else ""
    if len(target_parts) > 1:
        label += f" (+{len(target_parts) - 1} more)"
    run = _ProgressRun(_downloaded_bytes, total_size, parts=target_parts, label=label)
    run.start()
    try:
        with _record_stop(_downloaded_bytes, total_size):
            yield outcome
    finally:
        run.finish(outcome.succeeded)


@contextlib.contextmanager
def _snapshot_progress(disk_bytes_fn, total_size: int, label: str = ""):
    """Like _download_progress but for snapshot_download (many files): byte
    count comes from a caller-supplied directory-size function. Indeterminate
    (total_size == 0) still streams a 'downloading' phase so the GUI can show
    a busy bar; no-op when nothing shows progress.

    Yields a _ProgressOutcome; call ``.ok()`` on the success path or the closing
    event reports the measured partial instead of 100%.
    """
    outcome = _ProgressOutcome()

    def _measured() -> int:
        dl = disk_bytes_fn()
        return min(dl, total_size) if total_size else dl

    if not _progress_wanted():
        with _record_stop(_measured, total_size):
            yield outcome
        return

    run = _ProgressRun(_measured, total_size, label=label)
    run.start()
    try:
        with _record_stop(_measured, total_size):
            yield outcome
    finally:
        run.finish(outcome.succeeded)


def _report_success(rich_msg: str, plain_msg: str) -> None:
    """Announce a completed pull without letting a DISPLAY failure read as an
    OPERATION failure.

    PRECONDITION: every call site must reach this only after the download,
    checksum verification and registry write are fully done, so there is no
    remaining work a swallowed failure here could mask.

    Catches ``Exception`` rather than an enumerated set: printing the checkmark
    glyph has crashed with a ``ModuleNotFoundError`` from rich's cell-width
    lookup (``rich._unicode_data``) and with a ``UnicodeEncodeError`` from a
    legacy Windows console write path. The GUI runs pull as a subprocess and
    treats a non-zero exit as a failed pull, so an uncaught exception here
    would report a successful download as failed.
    """
    try:
        console.print(rich_msg)
    except Exception as e:
        logger.warning("could not render the pull success message (%s); "
                        "falling back to a plain-ASCII version", e)
        try:
            console.print(plain_msg)
        except Exception as e2:
            logger.warning("plain-text fallback also failed to print: %s", e2)



def pull_model(
    model_spec: str,
    name: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    mmproj_spec: Optional[str] = None,
    model_type: str = "auto",
    store: Optional[str] = None,
    dest_dir: Optional[Path] = None,
    register: bool = True,
) -> bool:
    """Download a model from HuggingFace or a URL.

    Returns True on success or a benign no-op (already present / aliased /
    user-skipped), False on a real error, so callers can set a non-zero exit
    code and the GUI can mark the job failed instead of reporting "finished".

    *store* ("copy" / "move" / None) only applies to the local-path branch
    below - a remote HF/URL download already lands in MODELS_DIR on its own.

    *dest_dir*, when given, routes the download to that directory instead of
    MODELS_DIR (e.g. a ComfyUI models subfolder) and skips localm's own
    registry when *register* is False. Only supported for a single-file HF
    spec (``owner/repo:file`` or ``owner/repo/file.gguf``) - a bare-repo
    snapshot or a direct URL pull with *dest_dir* set is refused rather than
    silently downloading to MODELS_DIR anyway.
    """
    spec = _mm.resolve_spec(model_spec)
    type_is_auto = (model_type == "auto")

    # A local filesystem path is not a remote spec: register it in place rather
    # than mis-parsing a Windows drive-colon as an owner/repo:file spec. This is
    # checked BEFORE any network auto-detect, so registering a local file never
    # leaks its path (and model filename) to huggingface.co - a POSIX absolute
    # path or a forward-slash relative path both contain "/" and would otherwise
    # be probed against HF. Only an absolute path or an existing file counts, so
    # a bare HF "owner/repo" is never shadowed by a same-named local directory.
    # add_local does the validation and dedup.
    #
    # model_spec (and so `local`) is the raw CLI/GUI-supplied spec string, and
    # expected_sha256 (`want`) is the raw --sha256 value; neither is
    # charset-validated anywhere upstream, so both are escaped. `actual` is a
    # hashlib hexdigest and is escaped too.
    from rich.markup import escape
    try:
        local = Path(model_spec).expanduser()
        is_local_path = local.exists() and (local.is_absolute() or local.is_file())
    except OSError:
        is_local_path = False
    if is_local_path:
        # A user-supplied --sha256 is verified against the real bytes here and
        # the pull is refused on a mismatch, never registered as a success. A
        # full HF repo refuses --sha256 outright instead.
        if expected_sha256:
            if local.is_dir():
                console.print(
                    "[red]--sha256 is not supported for a local directory[/red] "
                    f"({escape(str(local))}): a folder has many files and no single "
                    "digest to verify. Drop --sha256, or point at a single .gguf file.")
                return False
            want = expected_sha256.strip().lower()
            actual = _verify_digest(local).lower()
            if actual != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] {escape(str(local))} is "
                    f"{escape(actual[:16])}…, not --sha256 {escape(want[:16])}…. "
                    "Refusing to register.")
                return False
            _report_success(f"[green]✓[/green] SHA256 verified: {escape(actual[:16])}…",
                            f"[green]OK[/green] SHA256 verified: {escape(actual[:16])}…")
        # A local file gets no remote type probe: honour an explicit --type,
        # else let add_local detect it (GGUF -> llm, HF dir -> config.json,
        # otherwise the 'unknown' sentinel).
        local_type = None if model_type == "auto" else model_type
        return _mm.add_local(str(local), name=name, model_type=local_type, store=store)

    # owner/repo@revision[:file] names a branch, tag or commit of a HuggingFace
    # repo; the revision is split off here so every later step sees a plain spec.
    ref: Optional[str] = None
    if not (spec.startswith("http://") or spec.startswith("https://")):
        try:
            spec, ref = split_hf_revision(spec)
            if mmproj_spec and split_hf_revision(mmproj_spec)[1] is not None:
                raise ValueError("--mmproj does not take a revision; name the "
                                 "projector as owner/repo:file.gguf")
        except ValueError as e:
            console.print(f"[red]Invalid model spec:[/red] {escape(str(e))}")
            return False

    # net_mode=off means "no network at all", so it stops a remote model
    # download and the type auto-detect probe below - unless the owner has
    # exempted explicit downloads via net_allow_model_downloads.
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        console.print(
            "[red]Network access is disabled (net_mode=off).[/red] A model pull "
            "needs the network; enable it with: localm config net_mode ask - or "
            "allow just downloads: localm config net_allow_model_downloads true")
        return False

    # civitai:<versionId> or civitai:<versionId>:<fileId>: a distinct provider,
    # dispatched before the HF-oriented type auto-detect below - its type comes
    # from CivitAI's own model type (sources.CIVITAI_TYPE_MAP), never from an
    # HF pipeline-tag probe, and it always routes to the ComfyUI-subfoldered
    # tree rather than an explicit dest_dir.
    if spec.startswith("civitai:"):
        if dest_dir is not None:
            console.print(
                "[red]dest_dir is not supported for a civitai: spec[/red] - "
                "the destination is resolved automatically from the file's "
                "own CivitAI type."
            )
            return False
        if mmproj_spec:
            console.print(
                "[yellow]--mmproj is ignored for a civitai: spec[/yellow] "
                "(vision projectors are an HF/GGUF concept).")
        version_id, _, file_id = spec[len("civitai:"):].partition(":")
        return _mm._pull_civitai_file(
            version_id, name, file_id=file_id or None,
            expected_sha256=expected_sha256, redownload=redownload,
            register=register, model_type=model_type)

    # A Hugging Face spec: the hub endpoint must pass the network policy
    # (domain lists, private-address guard) before any metadata or download call.
    if not (spec.startswith("http://") or spec.startswith("https://")):
        refusal = _hf_endpoint_refusal()
        if refusal:
            console.print(
                f"[red]Network policy refused the Hugging Face pull:[/red] "
                f"{escape(refusal)}")
            return False

    # A named revision is checked once, before any other Hugging Face call or
    # download, and every later call is pinned to the commit it resolves to.
    revision: Optional[HfRevision] = None
    if ref is not None:
        try:
            revision = resolve_hf_revision(spec.split(":", 1)[0], ref)
        except HfRevisionError as e:
            console.print(f"[red]Cannot pull {escape(spec.split(':', 1)[0])}@{escape(ref)}:"
                          f"[/red] {escape(str(e))}. Nothing was downloaded.")
            return False
        console.print(f"Revision [bold]{escape(ref)}[/bold] is commit "
                      f"[dim]{revision.commit[:12]}[/dim]")
    rev_kw: dict[str, Any] = {"revision": revision} if revision is not None else {}

    # Remote spec: resolve the model type (a network probe against HF for a bare
    # owner/repo). Only reached for confirmed-remote specs, after the local-path
    # and net_mode gates above.
    detected_type = "llm"
    if model_type == "auto":
        if "/" in spec and not (spec.startswith("http://") or spec.startswith("https://")):
            repo_id = spec.split(":")[0] if ":" in spec else spec
            # A named .gguf file is a hard LLM signal, so type it 'llm' and do
            # NOT fall back to the repo's HF pipeline_tag, which a GGUF-quant
            # repo often lacks. Both file forms the dispatch below accepts
            # (owner/repo:file.gguf and owner/repo/file.gguf) are caught via the
            # last path segment; the pipeline tag is only probed for a bare repo.
            if spec.rsplit("/", 1)[-1].lower().endswith(".gguf"):
                detected_type = "llm"
            else:
                detected_type = _hf_pipeline_tag_to_type(repo_id)
                logger.info("Auto-detected model type for %s: %s", repo_id, detected_type)
                # A bare owner/repo pull is a full snapshot, and
                # _pull_hf_snapshot takes a second, harder look at the
                # config.json it downloads and announces the outcome itself.
                # A single-file spec is announced by _content_media_type once the
                # file's own tensors have been read.
        else:
            detected_type = "llm"
    else:
        detected_type = model_type

    is_url_spec = spec.startswith("http://") or spec.startswith("https://")
    is_single_file_spec = not is_url_spec and "/" in spec and (
        ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"))
    if dest_dir is not None and not is_single_file_spec:
        # dest_dir routing is only wired through _pull_gguf_file, so any other
        # spec shape is refused rather than downloaded to MODELS_DIR.
        console.print(
            "[red]dest_dir is only supported for a single-file spec[/red] "
            "(owner/repo:file or owner/repo/file.gguf) - "
            # Quoted by hand rather than via !r/repr(): escape() after repr()
            # re-escapes repr()'s own backslash, and repr() after escape() is
            # wrong in the other direction.
            f"'{escape(model_spec)}' would pull a full snapshot or direct "
            "URL instead."
        )
        return False

    if spec.startswith("http://") or spec.startswith("https://"):
        res = _pull_url(spec, _sanitize_name(name or _stem_from_url(spec)),
                         expected_sha256=expected_sha256, redownload=redownload,
                         model_type=detected_type)
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            starter_pin = (None if expected_sha256 or revision is not None
                           else _mm.starter_model_sha256(spec))
            res = _mm._pull_gguf_file(spec, name, expected_sha256=expected_sha256 or starter_pin,
                                   redownload=redownload, model_type=detected_type,
                                   dest_dir=dest_dir, register=register,
                                   type_is_auto=type_is_auto, mmproj_spec=mmproj_spec,
                                   pinned_starter=bool(starter_pin), **rev_kw)
        else:
            # owner/repo  (no filename) -> full HuggingFace snapshot
            res = _mm._pull_hf_snapshot(spec, name, expected_sha256=expected_sha256,
                                     redownload=redownload, model_type=detected_type,
                                     **rev_kw)
    else:
        console.print(f"[red]Unknown spec:[/red] {escape(model_spec)}")
        console.print("Formats:")
        console.print("  [bold]owner/repo[/bold]              full HF model directory")
        console.print("  [bold]owner/repo:file.gguf[/bold]   single GGUF file")
        console.print("  [bold]owner/repo@revision[/bold]     a branch, tag or commit "
                      "(also owner/repo@revision:file)")
        console.print("  [bold]https://...[/bold]             direct URL")
        console.print("Run [bold]localm models[/bold] for GGUF shortcuts.")
        res = False

    # A single-file GGUF spec already threaded mmproj_spec into
    # _pull_gguf_file above, which fetched it AND recorded it on the registry
    # entry. This tail covers the other dispatch shapes (a direct URL, a full
    # HF snapshot), where there is no llama.cpp GGUF registration to attach a
    # projector to, so the file is only downloaded.
    if res and mmproj_spec and not is_single_file_spec:
        console.print(f"Pulling mmproj: {escape(mmproj_spec)}")
        # "/" must be present (an owner/repo), the same precondition
        # _fetch_explicit_mmproj enforces: without it a bare "file.gguf" passes
        # the .endswith(".gguf") half of this check and then crashes
        # _pull_gguf_file's own identical split with an IndexError.
        if "/" in mmproj_spec and (
                ":" in mmproj_spec or mmproj_spec.rsplit("/", 1)[-1].endswith(".gguf")):
            _mm._pull_gguf_file(mmproj_spec, name=None, register=False)
        else:
            console.print("[red]mmproj spec must be a specific file (owner/repo:file.gguf)[/red]")

    return res




def _stem_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0].removesuffix(".gguf")




def _check_disk_space(dest_dir: Path, required_bytes: int) -> bool:
    """
    Verify there is at least *required_bytes* of free space on the volume that
    holds *dest_dir*.  Prints a warning and returns False when space is
    insufficient; returns True when fine or when the check is skipped
    (e.g. ``required_bytes == 0``).

    If the free-space check itself cannot be measured (offline models dir,
    permission denied, etc.) it is treated as OK by design: a WARNING is
    logged and the download proceeds rather than blocking a working setup.
    """
    if not required_bytes:
        return True
    try:
        usage = shutil.disk_usage(dest_dir)
        if usage.free < required_bytes:
            need_gb  = required_bytes / 1024**3
            free_gb  = usage.free / 1024**3
            # dest_dir can be caller-supplied (e.g. --comfy-dest-dir), so it is
            # escaped; need_gb/free_gb are floats formatted with :.1f.
            from rich.markup import escape
            console.print(
                f"[red]Not enough disk space.[/red] "
                f"Need {need_gb:.1f} GB, have {free_gb:.1f} GB free on "
                f"{escape(str(dest_dir))}"
            )
            return False
    except Exception as e:
        # An unmeasurable check is surfaced but non-fatal: an unknown
        # free-space value is treated as OK and the download proceeds.
        logger.warning("could not check free space on %s (%s); proceeding", dest_dir, e)
    return True




_FULL_GIT_ID_RE = re.compile(r"^[0-9a-f]{40}$")


def _valid_commit(value: object) -> Optional[str]:
    """*value* when it is a full 40-hex commit id, else None."""
    return value if isinstance(value, str) and _FULL_GIT_ID_RE.match(value) else None


class _PublishedDigest(str):
    """A sha256 hex digest that also carries, in ``commit``, the repository
    commit that last changed the file it describes (None when unknown). The
    file at that commit is the file the digest describes, so a download made at
    it cannot disagree with the digest because the repository changed
    meanwhile."""

    commit: Optional[str]

    def __new__(cls, digest: str, commit: Optional[str] = None) -> "_PublishedDigest":
        obj = super().__new__(cls, digest)
        obj.commit = _valid_commit(commit)
        return obj


def _revision_kwargs(commit: Optional[str]) -> dict[str, Any]:
    """The ``revision`` keyword for a hub call at *commit*; nothing when no
    commit is known, which leaves the call on the default branch."""
    return {"revision": commit} if commit else {}


def _file_revision(published: Optional[str],
                   revision: "Optional[HfRevision]") -> dict[str, Any]:
    """The ``revision`` keyword for a hub download of the file *published*
    describes, when the pull is at *revision*: the commit that last changed the
    file, else the revision's own commit, so a file with no published digest is
    still read at the revision and never from the default branch."""
    return _at_commit(published) or _revision_kwargs(_rev_commit(revision))


def _at_commit(published: Optional[str]) -> dict[str, Any]:
    """The ``revision`` keyword for a hub download of the file *published*
    describes."""
    return _revision_kwargs(getattr(published, "commit", None))


_HF_NAME_PART = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_HF_REVISION_SPEC_RE = re.compile(
    rf"^(?P<repo>{_HF_NAME_PART}/{_HF_NAME_PART})@(?P<rest>.*)$", re.DOTALL)
_HF_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,254}$")


def split_hf_revision(spec: str) -> "tuple[str, Optional[str]]":
    """Split ``owner/repo@<revision>[:file]`` into ``(owner/repo[:file], revision)``.

    A spec with no revision comes back unchanged with ``None``. The revision is
    a branch, tag or commit and may itself contain ``/`` (``refs/pr/1``); it
    ends at the first ``:``. Only the exact shape ``owner/repo@`` counts: a URL
    (``https://user@host/m.gguf``), a drive path (``D:/a@b/m.gguf``), a path
    starting ``~``, ``.`` or ``..``, a ``civitai:`` spec and a file name holding
    an ``@`` (``owner/repo:m@2.gguf``, ``owner/repo/m@2.gguf``) are all
    returned untouched.

    Raises :class:`ValueError` with a user-facing message for a revision that
    is empty or malformed, and for ``owner/repo@rev/file.gguf``, whose revision
    would swallow the file name (``owner/repo@rev:file.gguf`` is the form).
    Does no I/O."""
    m = _HF_REVISION_SPEC_RE.match(spec)
    if m is None:
        return spec, None
    repo, rest = m.group("repo"), m.group("rest")
    ref, colon, remote = rest.partition(":")
    if not ref:
        raise ValueError(f"empty revision after '@' in {repo}@ - name a branch, "
                         "tag or commit, e.g. owner/repo@main")
    if (not _HF_REVISION_RE.match(ref) or ".." in ref or "//" in ref
            or ref.endswith(("/", ".", ".lock"))):
        raise ValueError(f"'{ref[:60]}' is not a valid branch, tag or commit name")
    if colon and not remote:
        raise ValueError(f"no file name after ':' in {repo}@{ref}:")
    if not colon and ref.lower().endswith(".gguf"):
        raise ValueError(f"'{ref}' looks like a file name. To pull one file at a "
                         f"revision write {repo}@<revision>:<file>")
    return (f"{repo}:{remote}" if colon else repo), ref


class HfRevision(NamedTuple):
    """A resolved Hugging Face revision: *ref* as the user typed it and the full
    *commit* it pointed at when the pull started."""

    ref: str
    commit: str


class HfRevisionError(Exception):
    """A revision could not be resolved. *kind* is ``missing`` (the repo exists
    and has no such revision), ``repo`` (no such repo, or one this account cannot
    see), ``denied`` (gated or unauthorised) or ``unreachable`` (the hub could
    not be asked at all)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def resolve_hf_revision(repo_id: str, ref: str) -> HfRevision:
    """Pin *ref* of *repo_id* to its commit with one metadata call, before any
    download. Raises :class:`HfRevisionError`; a network or permission failure
    is never reported as a missing revision."""
    from huggingface_hub.errors import (
        GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError)
    try:
        info = _hf_api().model_info(repo_id, revision=ref)
    except GatedRepoError as e:
        raise HfRevisionError(
            "denied", f"{repo_id} is gated: accept its terms on huggingface.co and "
                      "set a Hugging Face token to pull it") from e
    except RevisionNotFoundError as e:
        raise HfRevisionError(
            "missing", f"{repo_id} has no branch, tag or commit named '{ref}'") from e
    except RepositoryNotFoundError as e:
        raise HfRevisionError(
            "repo", f"{repo_id} was not found on Hugging Face, or needs a token "
                    "that can see it") from e
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise HfRevisionError(
                "denied", f"Hugging Face refused access to {repo_id} (HTTP {status})") from e
        raise HfRevisionError(
            "unreachable", f"could not ask Hugging Face about {repo_id}@{ref}: "
                           f"HTTP {status or 'error'}") from e
    except Exception as e:
        raise HfRevisionError(
            "unreachable", f"could not reach Hugging Face to check {repo_id}@{ref}: "
                           f"{type(e).__name__}") from e
    commit = _valid_commit(getattr(info, "sha", None))
    if commit is None:
        raise HfRevisionError(
            "unreachable", f"Hugging Face did not report a commit for {repo_id}@{ref}")
    return HfRevision(ref, commit)


def _rev_commit(revision: "Optional[HfRevision]") -> Optional[str]:
    """The commit a revision pull is pinned to, or None for a pull of the default branch."""
    return revision.commit if revision is not None else None


def _published_digest(repo_id: str, filename: str,
                      revision: "Optional[HfRevision]") -> Optional[str]:
    """``_mm._hf_file_sha256`` for *filename* as it is at *revision* (the default
    branch when None)."""
    if revision is None:
        return _mm._hf_file_sha256(repo_id, filename)
    return _mm._hf_file_sha256(repo_id, filename, revision=revision.commit)


def _register_revision(revision: "Optional[HfRevision]") -> dict[str, Any]:
    """The registry keywords recording *revision* on a pulled model's entry;
    nothing for a pull of the default branch."""
    if revision is None:
        return {}
    return {"revision": revision.ref, "commit": revision.commit}


def _revision_record_differs(model_name: str, revision: "Optional[HfRevision]") -> bool:
    """True when registry entry *model_name* records a different revision than
    *revision* (none when None), so registering the file again must rewrite it."""
    entry = _mm.load_registry().get(model_name)
    return isinstance(entry, dict) and (
        (entry.get("revision") or None) != (revision.ref if revision else None))


_SNAPSHOT_CLAIM = ".localm-revision.json"


def _snapshot_claim_held_by_other(dest: Path, repo_id: str,
                                  revision: "Optional[HfRevision]"
                                  ) -> "Optional[tuple[str, Optional[str]]]":
    """``(repo, revision)`` of the unfinished download that holds *dest* when it
    is not the pull of *repo_id* at *revision* (the default branch when None),
    else None. Every snapshot pull leaves a marker in its folder until the
    folder is registered, because an unregistered folder has no registry entry
    to say whose files it holds. A registered folder is described by its entry,
    so a marker there is stale and counts as no claim, as does an unreadable
    one. Repo ids compare in any letter case (Hugging Face resolves either).
    An unregistered folder that holds files but no marker (an unfinished pull
    by a localm that wrote none) is held by an unknown default-branch download:
    ``("", None)`` for a pull at a revision, no claim for a default-branch
    pull."""
    try:
        claim = json.loads((dest / _SNAPSHOT_CLAIM).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if revision is None or find_aliases_by_path(dest, _mm.load_registry()):
            return None
        try:
            has_files = any(p.name != ".cache" for p in dest.iterdir())
        except OSError:
            has_files = True
        return ("", None) if has_files else None
    except (OSError, ValueError, RecursionError) as e:
        logger.warning("could not read %s in %s (%s); treating the folder as unclaimed",
                       _SNAPSHOT_CLAIM, dest, e)
        return None
    if not isinstance(claim, dict) or not isinstance(claim.get("repo"), str):
        logger.warning("%s in %s is not a claim; treating the folder as unclaimed",
                       _SNAPSHOT_CLAIM, dest)
        return None
    held = (claim["repo"], claim.get("revision") or None)
    if (held[0].casefold(), held[1]) == (repo_id.casefold(), revision.ref if revision else None):
        return None
    if find_aliases_by_path(dest, _mm.load_registry()):
        return None
    return held


def _clear_snapshot_claim(dest: Path) -> None:
    """Delete the marker of *dest* once the folder is registered. A marker that
    cannot be deleted is logged; a registered folder ignores it."""
    try:
        (dest / _SNAPSHOT_CLAIM).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove %s from %s: %s", _SNAPSHOT_CLAIM, dest, e)


def _holds_other_revision(dest: Path, repo_id: str,
                          revision: "Optional[HfRevision]") -> bool:
    """True when *dest* is registered as *repo_id* at a different revision than
    *revision* (the default branch when None): its files are another
    revision's, so they never count as this pull's finished download."""
    reg = _mm.load_registry()
    wanted = revision.ref if revision else None
    return any(
        reg[n].get("source") == f"hf:{repo_id}" and (reg[n].get("revision") or None) != wanted
        for n in find_aliases_by_path(dest, reg))


def _entry_is_hf_source(info: dict, repo_id: str, revision: "Optional[HfRevision]") -> bool:
    """True when registry entry *info* was pulled from *repo_id* at the same
    revision ref as *revision* (the default branch when None)."""
    return (info.get("source") == f"hf:{repo_id}"
            and (info.get("revision") or None) == (revision.ref if revision else None))


def _hf_file_sha256(repo_id: str, filename: str,
                     token: Optional[str] = None,
                     revision: Optional[str] = None) -> Optional[str]:
    """
    Ask the HuggingFace API for a file's LFS sha256 without downloading it.
    Returns a :class:`_PublishedDigest`, or None when offline, on any API error
    (logged at debug), or for non-LFS files.

    *token*: optional HF API token - needed for a gated repo's metadata. The
    configured token is used when none is given. *revision*: the branch, tag or
    commit to read the file at; the default branch when None.
    """
    try:
        info = _hf_api().get_paths_info(
            repo_id, [filename], token=token or _hf_token(), expand=True,
            **({"revision": revision} if revision else {}))
        if info:
            lfs = getattr(info[0], "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else None
            if digest:
                last = getattr(info[0], "last_commit", None)
                return _PublishedDigest(digest.lower(), getattr(last, "oid", None))
    except Exception as e:
        logger.debug("could not read the sha256 of %s from %s: %s", filename, repo_id, e)
    return None


def _note_unverified(repo_id: str, names: list[str]) -> None:
    """Say that *names* are not checked against a published sha256 because none
    could be read."""
    from rich.markup import escape
    shown = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
    console.print(
        f"[yellow]Note:[/yellow] no sha256 could be read for {escape(shown)} from "
        f"{escape(repo_id)} (the file is not stored with a digest, or the lookup "
        "failed), so it is not checked against one.")
    logger.warning("no published sha256 for %s in %s; the download is not integrity-checked",
                   shown, repo_id)




def _pick_best_of_same_repo_mmprojs(cands: list[str], *, speech: bool = False) -> str:
    """Deterministic pick among several mmproj filenames found in the SAME
    repo, once stem-matching (``_pick_mmproj_candidate``) could not narrow them
    to one.

    PRECONDITION: every candidate must already come from the ONE repo the
    caller is pulling from, so they are quantised variants of the same
    projector rather than projectors for different models. Prefers the
    conventional highest-precision f16 build; falls back to a sorted-first pick
    for determinism. With *speech* (a text-to-speech model's generation
    stages) the order is Q8_0, then bf16, then any other non-f16 build, then
    f16. See test_a_speech_model_takes_the_q8_projector_over_f16."""
    # Matches f16 as a whole token, not a bf16 substring. See
    # test_bartowski_gemma3_triple_picks_f16_not_bf16.
    def _has(c: str, token: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", c.lower()) is not None

    f16 = [c for c in cands if _has(c, "f16")]
    if speech:
        for group in ([c for c in cands if _has(c, "q8_0")],
                      [c for c in cands if _has(c, "bf16")],
                      [c for c in cands if c not in f16]):
            if group:
                return sorted(group)[0]
    return f16[0] if f16 else sorted(cands)[0]


def _hf_repo_files(repo_id: str) -> Optional[list[str]]:
    """*repo_id*'s file listing, or None when it could not be fetched at all
    (offline, API error, rate limit).

    None is kept distinct from an empty listing: a listing FAILURE is never
    read as "this repo has no projector"."""
    try:
        return _hf_api().list_repo_files(repo_id)
    except Exception as e:
        logger.debug("could not list files for %s to look for an mmproj "
                     "sibling: %s", repo_id, e)
        return None


def _pick_mmproj_from_listing(
    files: list[str], model_filename: str, base_dir: Path, *, speech: bool = False,
) -> Optional[str]:
    """The mmproj (vision projector) filename among *files* (a repo's file
    listing) that pairs with *model_filename*, or None when none qualify.

    *files* comes from a REMOTE HF repo listing, so every candidate is
    confined through ``_safe_models_filename`` - the same guard an explicit
    --mmproj filename gets - BEFORE it is considered for picking, not after it
    is chosen. A single-path-component check is not enough: on Windows a value
    with no forward slash can still be a drive-qualified or backslash-relative
    path, and ``_safe_models_filename`` rejects those and confines the result
    inside *base_dir*.

    The pick is ``_pick_mmproj_candidate``'s, the policy the folder scan
    (``find_sibling_mmproj``) uses, given the listing's other model files and no
    GGUF headers. When several candidates share the model's leading name token,
    ``_pick_best_of_same_repo_mmprojs`` picks among them (with *speech* for a
    text-to-speech model)."""
    cands = [f for f in files
             if f != model_filename and "mmproj" in f.lower()
             and f.lower().endswith(".gguf")
             and _mm._safe_models_filename(f, base_dir) is not None]
    if not cands:
        return None
    model_name = Path(model_filename).name
    others = sorted({first_split_part(Path(f).name) for f in files
                     if f.lower().endswith(".gguf") and "mmproj" not in f.lower()}
                    - {first_split_part(model_name)})
    picked = _mm._pick_mmproj_candidate(model_name, cands, others=others)
    if picked:
        return picked
    # _pick_mmproj_candidate gave up. Only candidates that share the model's
    # leading token go to _pick_best_of_same_repo_mmprojs, and only two or more.
    stem = Path(model_filename).stem.lower().replace("mmproj", "").split("-")[0].split(".")[0]
    stem_matches = [c for c in cands if stem and stem in c.lower()]
    if len(stem_matches) >= 2:
        return _pick_best_of_same_repo_mmprojs(stem_matches, speech=speech)
    return None


def _hf_repo_mmproj_filename(
    repo_id: str, model_filename: str, base_dir: Path, *, speech: bool = False,
) -> Optional[str]:
    """The mmproj (vision projector) filename in *repo_id*'s OWN file listing
    that pairs with *model_filename*, or None when the repo ships none (or its
    listing could not be fetched at all). A HuggingFace metadata call only: the
    repo file listing, no download."""
    files = _hf_repo_files(repo_id)
    if files is None:
        return None
    return _pick_mmproj_from_listing(files, model_filename, base_dir, speech=speech)


# Highest -<n> suffix tried before giving up on a free local name for a
# same-named projector from a different repo. Mirrors
# registry._MAX_PROJECTOR_NAME_SUFFIX.
_MAX_MMPROJ_COLLISION_SUFFIX = 999


def _mmproj_reuse_confirmed(dest: Path, repo_id: str, candidate: str,
                            model_path: "Path | None") -> Optional[bool]:
    """Whether the file already at *dest* is confirmed to be *repo_id*'s own
    *candidate* vision projector.

    Checks the repo's published LFS sha256 first (``_hf_file_sha256``); when
    that is not published (a non-LFS file, offline, an API error), falls back
    to comparing *dest*'s GGUF embedding width against *model_path*'s, when
    both are known - the same signal ``find_sibling_mmproj`` (registry.py)
    already rejects a mismatch on. A vision projector's filename is not
    unique across HF repos (several vendors' vision releases each ship a lone
    ``mmproj-model-f16.gguf``), so a same-named file already at *dest* is
    never assumed to be this repo's own without one of these two checks
    confirming it.

    True: confirmed the same file. False: confirmed a DIFFERENT file - never
    reused or overwritten (see ``_mmproj_download_dest``). None: neither the
    digest nor the width could be compared, so the caller falls back to
    reusing *dest* as-is."""
    remote_digest = _mm._hf_file_sha256(repo_id, candidate)
    if remote_digest:
        try:
            local_digest = _verify_digest(
                dest, purpose="to confirm it is this repo's own vision projector")
        except OSError:
            return None
        return local_digest.lower() == remote_digest.lower()
    if model_path is not None:
        model_w = gguf_n_embd(model_path)
        dest_w = gguf_n_embd(dest)
        if model_w and dest_w:
            return model_w == dest_w
    logger.debug(
        "cannot confirm the existing %s is %s's own vision projector (no "
        "published digest, no comparable embedding width) - reusing it as-is",
        dest, repo_id)
    return None


def _mmproj_origin_path(dest: Path) -> Path:
    """The file beside the projector *dest* that records where it was
    downloaded from."""
    return dest.with_name(dest.name + ".origin.json")


def _record_mmproj_origin(dest: Path, repo_id: str, remote_name: str,
                          sha256: Optional[str]) -> None:
    """Record that *dest* was downloaded from ``hf:<repo_id>`` as *remote_name*.
    A failed write is logged and leaves *dest* without a recorded origin, which
    only means a differing copy is never replaced automatically. See
    test_a_projector_without_a_recorded_origin_is_never_replaced."""
    from ..storekit import atomic_write
    try:
        atomic_write(_mmproj_origin_path(dest), json.dumps(
            {"source": f"hf:{repo_id}", "file": remote_name, "sha256": sha256}))
    except OSError as e:
        logger.warning("could not record where %s came from: %s", dest, e)


def _mmproj_origin_is(dest: Path, repo_id: str, remote_name: str) -> bool:
    """True when the origin recorded beside *dest* says it was downloaded from
    ``hf:<repo_id>`` as *remote_name*: the file is that repo's own, so a digest
    that differs from the repo's published one means it is damaged or outdated,
    not another repo's projector of the same name."""
    try:
        recorded = json.loads(_mmproj_origin_path(dest).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    return (isinstance(recorded, dict) and recorded.get("source") == f"hf:{repo_id}"
            and recorded.get("file") == remote_name)


def _mmproj_download_dest(base_dir: Path, candidate: str, repo_id: str,
                          model_path: "Path | None") -> "tuple[Path | None, bool]":
    """Where to place *candidate* (a vision projector from *repo_id*) under
    *base_dir*, and whether a download can be skipped because that location
    already holds it.

    The plain *base_dir* / *candidate* name, unless that name is occupied by a
    file ``_mmproj_reuse_confirmed`` rules out as a different repo's
    projector - in which case the first free ``<candidate stem>-<n>.gguf`` is
    returned instead, and the occupied file is never read from beyond that
    one comparison or written to; it may be another entry's own correctly
    recorded projector. When the occupied file differs from the digest the repo
    publishes and the origin recorded beside it says it came from *repo_id* as
    *candidate*, it is that repo's own damaged or outdated copy: it is deleted and the plain name is
    returned for a fresh download. Returns ``(None, False)`` when every
    numbered name up to ``_MAX_MMPROJ_COLLISION_SUFFIX`` is also taken."""
    dest = base_dir / candidate
    if not dest.exists():
        return dest, False
    confirmed = _mmproj_reuse_confirmed(dest, repo_id, candidate, model_path)
    if confirmed is not False:
        return dest, True
    if _mm._hf_file_sha256(repo_id, candidate) and _mmproj_origin_is(dest, repo_id, candidate):
        from rich.markup import escape
        try:
            dest.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("could not delete the damaged %s: %s", dest, e)
        else:
            console.print(
                f"[yellow]{escape(dest.name)} does not match the digest {escape(repo_id)} "
                "publishes - deleted, fetching it again.[/yellow]")
            return dest, False
    stem, suffix = Path(candidate).stem, Path(candidate).suffix
    for n in range(2, _MAX_MMPROJ_COLLISION_SUFFIX + 1):
        free = base_dir / f"{stem}-{n}{suffix}"
        if not free.exists():
            return free, False
    return None, False


def _download_mmproj_file(repo_id: str, remote_name: str, dest: Path, base_dir: Path) -> None:
    """Download *remote_name* from *repo_id* to *dest*. May raise; every call
    site wraps this in its own try/except.

    When *dest* is the plain *base_dir* / *remote_name*, downloads straight
    there. Otherwise - the plain name is occupied by a confirmed-different
    file - downloads into a private scratch directory under *base_dir* first
    and moves the result to *dest*: huggingface_hub's local-dir download mode
    names the downloaded file after *remote_name* alone, so downloading
    straight into *base_dir* would land on the occupied name instead of
    *dest*."""
    from huggingface_hub import hf_hub_download
    _ensure_hf_resumable_download()
    expected = _mm._hf_file_sha256(repo_id, remote_name)
    extra = _at_commit(expected)
    plain = base_dir / remote_name
    token = _hf_token()
    if dest == plain:
        with _record_stop(_job_bytes_measure([dest], 0, base_dir=base_dir,
                                             rel_parts=[remote_name]), 0):
            local = hf_hub_download(repo_id=repo_id, filename=remote_name,
                                     local_dir=str(base_dir), endpoint=_HF_ENDPOINT,
                                     token=token, tqdm_class=meter_tqdm_class(), **extra)
        if Path(local) != dest:
            shutil.move(local, dest)
    else:
        scratches: list[Path] = []
        with _record_stop(lambda: sum(_job_bytes_measure([], 0, base_dir=d)()
                                      for d in scratches if d.is_dir()), 0):
            with tempfile.TemporaryDirectory(dir=base_dir, prefix=".mmproj-") as scratch:
                scratches.append(Path(scratch))
                local = hf_hub_download(repo_id=repo_id, filename=remote_name,
                                         local_dir=scratch, endpoint=_HF_ENDPOINT,
                                         token=token, tqdm_class=meter_tqdm_class(),
                                         **extra)
                shutil.move(local, dest)
    if expected is None:
        _note_unverified(repo_id, [remote_name])
        _record_mmproj_origin(dest, repo_id, remote_name, None)
        return
    actual = _verify_digest(dest, purpose="to check the vision projector").lower()
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {remote_name}: expected {expected[:16]}..., "
            f"got {actual[:16]}... - the file was deleted")
    _record_mmproj_origin(dest, repo_id, remote_name, actual)


def _maybe_fetch_repo_mmproj(repo_id: str, filename: str, base_dir: Path, *,
                             speech: bool = False) -> Optional[Path]:
    """Auto-attach companion: look for a vision projector shipped in the SAME
    HF repo as *filename* and fetch it too.

    Returns the local Path of a verified projector to record on the model's
    registry entry, or None. When the listing could not be fetched at all this
    stays silent (see ``_hf_repo_files``); it prints an informational note only
    when the listing WAS read, *filename* looks like a vision-language release
    by name, and no usable projector is among the files.
    """
    # repo_id/filename/candidate/e all come from a REMOTE repo and none is
    # charset-restricted, so all are escaped. `candidate` sits directly inside
    # an OPEN [yellow]...[/yellow] tag below, where an unescaped value is
    # markup-TAG-INJECTION rather than bracket-drop.
    from rich.markup import escape
    files = _hf_repo_files(repo_id)
    if files is None:
        return None
    candidate = _pick_mmproj_from_listing(files, filename, base_dir, speech=speech)
    if candidate is None:
        if _mm._looks_like_vision_gguf_name(repo_id, filename):
            console.print(
                "[yellow]Note:[/yellow] this looks like a vision-language "
                f"model, but no vision projector (mmproj) file was found in "
                f"{escape(repo_id)}. It may not be able to see images - if the "
                "projector lives in a different repo, pull it explicitly "
                "with [bold]--mmproj <repo>:<file>[/bold]."
            )
        return None

    dest, reuse_ok = _mmproj_download_dest(base_dir, candidate, repo_id, base_dir / filename)
    if dest is None:
        console.print(
            f"[yellow]Found a vision projector ({escape(candidate)}) in "
            f"{escape(repo_id)}, but every local name for it is already taken "
            "by a different file - not attaching it.[/yellow]"
        )
        return None
    if not reuse_ok:
        console.print(f"Pulling projector (mmproj): {escape(candidate)}")
        try:
            _download_mmproj_file(repo_id, candidate, dest, base_dir)
        except Exception as e:
            console.print(
                f"[yellow]Found a vision projector ({escape(candidate)}) in "
                f"{escape(repo_id)} but could not download it: {escape(str(e))}. "
                "Vision may not work for this model.[/yellow]"
            )
            return None

    if not _mm.gguf_is_mmproj(dest):
        # Hard-verify what was just fetched: the filename match is only a
        # heuristic, so a file that fails the real GGUF-metadata check is
        # reported as "none found" rather than attached.
        console.print(
            f"[yellow]{escape(dest.name)} does not look like a valid vision "
            "projector (GGUF metadata check failed) - not attaching it.[/yellow]"
        )
        return None
    return dest


def mmproj_backfill_candidate(entry: dict, path: Path) -> bool:
    """True when *entry* (a registry entry whose file is *path*) is a
    plausible target for the mmproj backfill: pulled from the default branch of
    an HF repo (a pull at a named revision is not backfilled from it), a plain
    LLM or text-to-speech registration, and not already carrying a recorded
    ``mmproj``.

    Pure, with no I/O; the network decision lives in
    ``backfill_mmproj_for_entry`` below, so a caller can filter cheaply
    first."""
    source = str(entry.get("source", ""))
    if not source.startswith("hf:") or entry.get("revision"):
        return False
    if entry.get("model_type") not in ("llm", "tts"):
        return False
    if entry.get("mmproj"):
        return False
    if "mmproj" in path.name.lower():
        return False
    return True


def backfill_mmproj_for_entry(entry: dict, path: Path) -> Optional[Path]:
    """Retroactively attach a vision projector to an already-pulled LLM entry.

    Runs the same-repo lookup a fresh pull does (``_maybe_fetch_repo_mmproj``),
    driven by the ``source`` the entry already recorded (``hf:<repo_id>``), so
    an existing registry entry gets the same auto-attach and hard-verify
    treatment. The model itself is not re-downloaded.

    Returns the fetched and verified projector Path for the caller to record,
    or None when *entry* is not a candidate, is blocked by policy, or nothing
    was found. NEVER RAISES, matching ``_maybe_fetch_repo_mmproj``'s contract,
    so one bad entry cannot take down a sync pass.

    Network policy: gated on ``network_mode() != "off"`` (or
    ``downloads_allowed_when_off()``), the same bar ``pull_model``'s net_mode
    gate uses for this identical HF-listing-plus-download operation, not the
    stricter ``== "allow"`` bar ``embedder.py`` applies to its background
    fetch, and on ``_hf_endpoint_refusal()`` being None."""
    if not mmproj_backfill_candidate(entry, path):
        return None
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        return None
    if _hf_endpoint_refusal():
        return None
    repo_id = str(entry["source"])[len("hf:"):]
    if not repo_id:
        return None
    from localm.model_manager.gguf import gguf_architecture, gguf_is_tts_architecture
    return _maybe_fetch_repo_mmproj(
        repo_id, path.name, path.parent,
        speech=gguf_is_tts_architecture(gguf_architecture(path)))


def _fetch_explicit_mmproj(mmproj_spec: str, base_dir: Path,
                           model_filename: Optional[str] = None) -> Optional[Path]:
    """Download the user-named --mmproj file (owner/repo:file.gguf) into
    *base_dir* and return its local path, verified as a real vision projector
    - or None on a bad spec, failed download, or failed verification, each of
    which is printed. An explicit --mmproj always wins over the same-repo
    auto-detection in ``_maybe_fetch_repo_mmproj``; the caller only reaches
    here when the user named one.

    *model_filename*, when given, is the model (also under *base_dir*) this
    projector is being attached to - used only as a fallback identity check
    when a same-named file already exists at the destination and the repo
    publishes no digest for it; see ``_mmproj_reuse_confirmed``."""
    # "/" must be present (an owner/repo) as well as one of the two file
    # markers, matching pull_model's own is_single_file_spec check. Without the
    # "/" precondition a bare "file.gguf" passes this guard and then crashes
    # the else branch below with an IndexError on parts[1], since rsplit("/", 1)
    # on a "/"-free string returns a ONE-element list.
    #
    # mmproj_spec/m_file/safe/e are all attacker/user/MCP-supplied and NOT
    # charset-restricted (_safe_models_filename rejects path traversal and
    # Windows-reserved characters, not '['/']'), so all are escaped. `safe` sits
    # directly inside an OPEN [yellow]...[/yellow] tag below, and it names a
    # filename that just failed the safety check.
    from rich.markup import escape
    if not ("/" in mmproj_spec
            and (":" in mmproj_spec or mmproj_spec.rsplit("/", 1)[-1].endswith(".gguf"))):
        console.print("[red]mmproj spec must be a specific file (owner/repo:file.gguf)[/red]")
        return None
    if ":" in mmproj_spec:
        m_repo, m_file = mmproj_spec.rsplit(":", 1)
    else:
        parts = mmproj_spec.rsplit("/", 1)
        m_repo, m_file = parts[0], parts[1]

    # Same traversal guard as the main file: m_file comes from a spec, which
    # may be user- or client-supplied over MCP.
    safe = _mm._safe_models_filename(m_file, base_dir)
    if safe is None:
        console.print(f"[red]Unsafe mmproj filename:[/red] {escape(m_file)}")
        return None

    model_path = (base_dir / model_filename) if model_filename else None
    dest, reuse_ok = _mmproj_download_dest(base_dir, safe, m_repo, model_path)
    if dest is None:
        console.print(
            f"[red]Every local name for {escape(safe)} is already taken by a "
            "different file[/red] - refusing to overwrite one.")
        return None
    if not reuse_ok:
        console.print(f"Pulling mmproj: {escape(mmproj_spec)}")
        try:
            _download_mmproj_file(m_repo, safe, dest, base_dir)
        except Exception as e:
            console.print(f"[red]mmproj download failed:[/red] {escape(str(e))}")
            return None

    if not _mm.gguf_is_mmproj(dest):
        console.print(
            f"[yellow]{escape(dest.name)} does not look like a valid vision projector "
            "(GGUF metadata check failed) - not attaching it.[/yellow]"
        )
        return None
    return dest


def _embedding_detection_note(path) -> str:
    """The console line a pull prints for a GGUF typed as an embedding model:
    names it a reranker when its own metadata marks it as one."""
    if _mm.gguf_reranker_signal(path):
        return "[dim]Detected as a reranker (GGUF metadata).[/dim]"
    return "[dim]Detected as an embedding model (GGUF metadata).[/dim]"


def _non_chat_detection_note(architecture) -> str:
    """The console line a pull prints for a GGUF typed from its non-chat
    ``general.architecture``."""
    from rich.markup import escape

    from localm.model_manager.gguf import gguf_is_tts_architecture
    if gguf_is_tts_architecture(architecture):
        return ("[dim]Detected as a text-to-speech model (GGUF metadata). Speak "
                "with it using 'localm speak'.[/dim]")
    return f"[yellow]{escape(_mm.gguf_chat_refusal(architecture) or '')}[/yellow]"


_MEDIA_TYPE_WORDS = {"lora": "a LoRA", "diffusion-unet": "an image or video diffusion model",
                     "vae": "a VAE", "text-encoder": "a text encoder",
                     "controlnet": "a ControlNet", "textual-inversion": "a textual-inversion embedding",
                     "upscaler": "an upscaler"}


def _media_type_phrase(media_type: str) -> str:
    """The console wording for registry media type *media_type*; a type with no
    entry in ``_MEDIA_TYPE_WORDS`` is named by its own string."""
    return _MEDIA_TYPE_WORDS.get(media_type) or f"a {media_type} model"


def _content_media_type(path: Path, reg_type: Optional[str]) -> Optional[str]:
    """The registry type for the downloaded non-GGUF file *path*: the
    diffusion model, text encoder, VAE or LoRA its own tensor layout says it is
    (as ``localm add`` types it), else *reg_type*. Prints a note when the
    file's layout changes the type."""
    media_type = _media_component_type(path, None)
    if media_type is None or media_type == reg_type:
        if reg_type == "unknown":
            console.print(
                "[yellow]Could not determine this model's type[/yellow] - "
                "registering it as 'unknown'. Run it by name, or set its type "
                "later: [bold]localm set-type <name> <type>[/bold]")
        return reg_type
    console.print(f"[dim]Detected as {_media_type_phrase(media_type)} (tensor names in "
                  "the file).[/dim]")
    return media_type


def _mmproj_for_registration(
    reg_type: str,
    repo_id: str,
    filename: str,
    base_dir: Path,
    dest_dir: Optional[Path],
    mmproj_spec: Optional[str],
    pinned_revision: bool = False,
) -> Optional[Path]:
    """The projector Path to record on this pull's registry entry, or None: the
    vision projector of an 'llm', or the speech generation stages of a 'tts'
    model. An explicit --mmproj wins when given, else the same-repo listing is
    auto-checked. Skipped entirely for a foreign destination (an explicit
    dest_dir), for any other registration type, and for a *filename* that
    already looks like a projector by its own name. A pull at a pinned revision
    (*pinned_revision*) attaches only an explicit --mmproj: the same-repo lookup
    reads the default branch, whose projector may not pair with that revision."""
    if dest_dir is not None or reg_type not in ("llm", "tts") or "mmproj" in filename.lower():
        return None
    if mmproj_spec:
        return _fetch_explicit_mmproj(mmproj_spec, base_dir, model_filename=filename)
    if pinned_revision:
        return None
    return _maybe_fetch_repo_mmproj(repo_id, filename, base_dir, speech=reg_type == "tts")


def _pull_gguf_file(
    spec: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    register: bool = True,
    model_type: str = "llm",
    dest_dir: Optional[Path] = None,
    type_is_auto: bool = False,
    mmproj_spec: Optional[str] = None,
    pinned_starter: bool = False,
    revision: Optional[HfRevision] = None,
) -> bool:
    """Download a single file from a HuggingFace repo (despite the name, not
    restricted to .gguf - any single-file ``owner/repo:filename`` spec dispatches
    here, see ``pull_model``'s docstring).

    ``expected_sha256`` is the user-supplied ``--sha256`` digest. When given it
    is reconciled with HuggingFace's own LFS metadata up front, and the
    downloaded first part is verified against it before the model is
    registered.

    ``dest_dir``, when given, routes the download to that directory instead of
    ``MODELS_DIR`` (e.g. a ComfyUI models subfolder) and is created via
    ``_mkdir_or_explain`` instead of ``ensure_dirs()``. ``register`` still
    controls whether the download is added to localm's own model registry -
    a file routed elsewhere (e.g. for ComfyUI, not for localm's own chat-model
    catalog) should normally pass ``register=False``.

    ``pinned_starter`` marks *expected_sha256* as the digest localm pins for a
    starter model rather than one the user typed, which changes only the
    message printed when the hub now serves a different file.

    ``revision``, when given, is a branch, tag or commit already resolved by
    ``resolve_hf_revision``: the digests, the size checks and the download are
    all taken at its commit, the registry entry records it, and the automatic
    same-repo projector lookup (which reads the default branch) is skipped.

    ``mmproj_spec``, when given, is the user's explicit ``--mmproj
    owner/repo:file.gguf`` choice and always wins over the automatic same-repo
    projector lookup below. When it is None and the pulled file registers as a
    plain 'llm', the HF repo's own file listing is checked for a
    vision-projector (mmproj) sibling and, if found, fetched and recorded on
    the registry entry.
    """
    # spec (and repo_id/filename/parts derived from it), the resolved
    # `part`/`want`/`expected` values below, and every exception's text are all
    # attacker/user/MCP-supplied or remote-sourced and none is
    # charset-restricted, so all are escaped. repo_id/filename sit directly
    # inside OPEN [bold cyan]/[bold] tags at several sites below, where an
    # unescaped value is markup-TAG-INJECTION.
    from rich.markup import escape
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False
    _ensure_hf_resumable_download()

    base_dir = dest_dir if dest_dir is not None else _mm.MODELS_DIR

    if ":" in spec:
        repo_id, remote = spec.split(":", 1)
    else:
        parts = spec.rsplit("/", 1)
        repo_id, remote = parts[0], parts[1]
    # "owner/repo:sub/dir/file" names a file inside a repo folder. huggingface_hub
    # writes it to <stage_dir>/sub/dir/file, which is then moved to
    # base_dir/file; stage_dir is under base_dir's hidden .cache so no folder of
    # the repo's layout appears in base_dir. A root-level file stages in
    # base_dir itself.
    remote_dir, in_folder, filename = remote.rpartition("/")
    stage_dir = base_dir / _REPO_FOLDER_STAGING if in_folder else base_dir
    if in_folder and not _safe_repo_subdir(remote_dir, stage_dir):
        console.print(
            f"[red]Unsafe repository path:[/red] {escape(remote)}\n"
            "A folder inside the repository may only use letters, digits and "
            "'._+-', with no '.' or '..' parts."
        )
        return False

    def _remote(part: str) -> str:
        return f"{remote_dir}/{part}" if remote_dir else part

    # Split GGUF: normalise to the full ordered part list. llama.cpp loads
    # the model from the first part, so that's what gets registered. A
    # non-split, non-gguf file (e.g. a .safetensors) is just a one-element list.
    all_parts = split_gguf_parts(filename) or [filename]
    filename  = all_parts[0]

    # Traversal guard: the filename comes from an untrusted spec
    # (owner/repo:../../evil.gguf), so every part is confined to base_dir before
    # it is used as a destination, and the whole pull is rejected on any unsafe
    # part. `part` is escaped: it names a filename that just failed this check.
    for part in all_parts:
        if _safe_models_filename(part, base_dir) is None:
            console.print(
                f"[red]Unsafe model filename:[/red] {escape(part)}\n"
                "A model filename must be a single name inside the models folder "
                "(no '/', '\\', or '..')."
            )
            return False

    model_name = _sanitize_name(name or filename.removesuffix(".gguf"))
    dest = base_dir / filename

    if register and _refuse_name_held_by_external_file(model_name, dest):
        return False

    # Expected digest from HF metadata - free, no download needed. Each file is
    # downloaded at the commit its digest was read at.
    expected = _published_digest(repo_id, _remote(filename), revision)
    part_digests = {}
    for part in all_parts[1:]:
        part_digest = _published_digest(repo_id, _remote(part), revision)
        if part_digest:
            part_digests[part] = part_digest
    published = {filename: expected, **part_digests}

    # Honour a user-supplied --sha256: when HF's own metadata digest is known
    # and disagrees with it, the bytes can never match, so refuse up front
    # rather than spending a download to discover the mismatch.
    want = expected_sha256.lower() if expected_sha256 else None
    if want and expected and want != expected.lower():
        if pinned_starter:
            console.print(
                f"[red]{escape(filename)} has changed on Hugging Face.[/red] This release "
                f"of localm pins it at sha256 {escape(want)}, and the hub now serves "
                f"{escape(expected)}. Refusing to download it. Update localm, or accept "
                f"the file the hub serves now with [bold]--sha256 {escape(expected)}[/bold]."
            )
            return False
        console.print(
            f"[red]SHA256 mismatch (before download):[/red] --sha256 "
            f"{escape(want[:16])}… does not match HuggingFace's metadata for "
            f"{escape(filename)} ({escape(expected[:16])}…). Refusing to download."
        )
        return False
    # The digest we will verify against / store: prefer HF metadata, else the
    # user-supplied value.
    verify_digest = expected or want
    digests = dict(part_digests)
    if verify_digest:
        digests[filename] = verify_digest
    unchecked = [p for p in all_parts if p not in digests]
    if unchecked:
        _note_unverified(repo_id, unchecked)

    missing = [p for p in all_parts
               if not (base_dir / p).exists()
               or (revision is not None and p not in digests)]
    for part in all_parts:
        if part in digests or not (base_dir / part).exists():
            continue
        taken, own = _undigested_file_held_by_other_pull(
            base_dir / part, repo_id, revision, model_name)
        if own:
            missing = [p for p in all_parts if p in missing or p not in digests]
        if taken:
            console.print(
                f"[red]File name already in use:[/red] {escape(part)} in the "
                "models folder is registered as "
                f"{_civitai_registered_as(taken)}, not as "
                f"{escape(repo_id)}{'@' + escape(revision.ref) if revision else ''}, "
                "and was left as it is.\n"
                "A file name holds one file, and --name only sets the registry "
                "name. To keep both, pull a file of this repository that has "
                "another name. To use this one instead, remove the other first, "
                "which deletes its file "
                f"({' and '.join('localm rm ' + escape(n) for n, _ in taken)}), "
                "then pull again."
            )
            return False
    if missing:
        for part, expected_digest in digests.items():
            if part in missing or (part == filename and want):
                continue
            taken = _hf_file_name_taken_by(base_dir / part, repo_id)
            if not taken:
                continue
            on_disk = _verify_digest(base_dir / part,
                                     purpose="to check the file already here")
            if on_disk.lower() != expected_digest.lower():
                _print_digest_name_in_use(part, on_disk, expected_digest, repo_id, taken)
                return False
    if not missing:
        # If the user asserted a hash, verify the file actually on disk before
        # treating it as the requested model.
        if want:
            on_disk = _verify_digest(dest, purpose="to check the file already here")
            if on_disk.lower() != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] The file already at "
                    f"{escape(filename)} ({escape(on_disk[:16])}…) does not "
                    + (f"match the digest localm pins for this starter model ({escape(want[:16])}…). "
                       "Delete it and pull again."
                       if pinned_starter else
                       f"match --sha256 ({escape(want[:16])}…).")
                )
                return False
        # A digest read from Hugging Face is registered as the file's own, so
        # the file on disk has to match it first.
        for part, published in digests.items():
            if part == filename and want:
                continue
            on_disk = _verify_digest(base_dir / part,
                                     purpose="to check the file already here")
            if on_disk.lower() != published.lower():
                taken = _hf_file_name_taken_by(base_dir / part, repo_id)
                if taken:
                    _print_digest_name_in_use(part, on_disk, published, repo_id, taken)
                    return False
                console.print(
                    f"[red]SHA256 mismatch![/red] The file already at "
                    f"{escape(part)} ({escape(on_disk[:16])}…) does not match the "
                    f"digest Hugging Face publishes ({escape(published[:16])}…): it is "
                    "incomplete, damaged or another version. Delete it and pull again."
                )
                return False
        console.print(f"[yellow]Already downloaded:[/yellow] {escape(filename)}")
        if register:
            unusable = _mm.gguf_unusable_reason(dest)
            if unusable is not None:
                console.print(f"[red]Not registered:[/red] {escape(filename)}\n{escape(unusable)}")
                return False
            reg_type = model_type
            # One shared header probe backs the mmproj/embedding refinement
            # below AND the persisted architecture/expert_count. Captured
            # regardless of type_is_auto: it is a fact about the file, not
            # about which type label this call chooses.
            gguf_meta = _mm.gguf_registry_metadata(dest)
            if type_is_auto and reg_type == "llm":
                if _mm.gguf_adapter_kind(dest) is not None:
                    console.print("[dim]Detected as a LoRA adapter (GGUF metadata). "
                                  "Attach it with 'localm adapter attach'.[/dim]")
                    reg_type = "lora"
                elif _mm.gguf_is_mmproj(dest):
                    console.print("[dim]Detected as a vision projector (GGUF metadata).[/dim]")
                    reg_type = "mmproj"
                elif _mm.gguf_embedding_signal(dest):
                    console.print(_embedding_detection_note(dest))
                    reg_type = "embedding"
                elif _mm.gguf_non_chat_model_type(gguf_meta.get("architecture")):
                    console.print(_non_chat_detection_note(gguf_meta.get("architecture")))
                    reg_type = _mm.gguf_non_chat_model_type(gguf_meta.get("architecture"))
                elif (media_type := _mm.gguf_media_component_type(dest)) is not None:
                    reg_type = media_type
                    console.print(f"[dim]Detected as {_media_type_phrase(reg_type)} (tensor names "
                                  "in the GGUF).[/dim]")
            if type_is_auto and not filename.lower().endswith(".gguf"):
                reg_type = _content_media_type(dest, reg_type)
            mmproj_path = _mmproj_for_registration(
                reg_type, repo_id, filename, base_dir, dest_dir, mmproj_spec,
                pinned_revision=revision is not None)
            _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}",
                                 digest=verify_digest, model_type=reg_type,
                                 mmproj=mmproj_path, architecture=gguf_meta.get("architecture"),
                                 expert_count=gguf_meta.get("expert_count"),
                                 refresh=_revision_record_differs(model_name, revision),
                                 **_register_revision(revision))
        return True

    # Pre-download duplicate check: are the same bytes already on disk
    # elsewhere? Only meaningful when the destination IS localm's own models
    # dir, because find_by_sha256 answers "is it in localm's REGISTRY", not "is
    # it at the destination". With an explicit dest_dir (a ComfyUI models
    # folder) the file is wanted THERE, so the download is not skipped; and a
    # dest_dir pull is register=False, so it does not alias either.
    if verify_digest and not redownload and dest_dir is None:
        dups = _mm.find_by_sha256(verify_digest)
        if dups:
            action = _mm._prompt_predownload_dup(dups, model_name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], model_name)
                return True
            # "download" falls through

    if dest_dir is not None:
        from ..config import _mkdir_or_explain
        _mkdir_or_explain(base_dir, is_home=False)
    else:
        _mm.ensure_dirs()

    # Disk space pre-flight - HEAD each missing part's CDN URL for its
    # Content-Length and the etag huggingface_hub will name its temp file by.
    from localm.netpolicy import NetworkPolicyError, UnresolvableHost
    part_sizes: dict[str, int] = {}
    etags: dict[str, str | None] = {}
    try:
        total_size = 0
        for part in missing:
            cdn_url = hf_hub_url(repo_id, _remote(part), endpoint=_HF_ENDPOINT,
                                 **_file_revision(published.get(part), revision))
            head    = _policy_head(cdn_url)
            size = int(head.headers.get("content-length", 0))
            part_sizes[part] = size
            etags[part] = _etag_from_head(head)
            total_size += size
    except UnresolvableHost:
        total_size = 0
    except NetworkPolicyError as e:
        console.print(f"[red]Network policy refused the download:[/red] "
                      f"{escape(str(e))}")
        return False
    except Exception:
        total_size = 0

    # Same-repo vision projector this pull would auto-attach (mirrors
    # _mmproj_for_registration's own gating): HEAD it and charge its size to
    # the disk-space preflight only, unless a file of its name and size is
    # already in base_dir. It is fetched after the model, outside the model's
    # progress.
    projector_size = 0
    if (register and mmproj_spec is None and dest_dir is None and revision is None
            and model_type not in ("mmproj", "embedding")
            and "mmproj" not in filename.lower()):
        try:
            mmproj_name = _mm._hf_repo_mmproj_filename(
                repo_id, filename, base_dir, speech=model_type == "tts")
            if mmproj_name:
                mmproj_head = _policy_head(
                    hf_hub_url(repo_id, mmproj_name, endpoint=_HF_ENDPOINT))
                projector_size = int(mmproj_head.headers.get("content-length", 0))
                try:
                    if (base_dir / mmproj_name).stat().st_size == projector_size:
                        projector_size = 0
                except OSError:
                    pass
        except UnresolvableHost:
            pass
        except NetworkPolicyError as e:
            console.print(f"[red]Network policy refused the download:[/red] "
                          f"{escape(str(e))}")
            return False
        except Exception:
            pass

    # Only bytes a resume will actually reuse count: a partial of the current
    # etag whose owner is proven gone. An unknown etag counts nothing.
    already_have = 0
    for part in missing:
        etag = etags.get(part)
        if not etag:
            continue
        inc = _hf_incomplete_path(stage_dir, _remote(part), etag)
        if inc is None:
            continue
        already_have += _reusable_partial_bytes(
            inc, part_sizes.get(part),
            lock_free=_hf_blob_lock_is_free(stage_dir, _remote(part)))

    if not _mm._check_disk_space(
            base_dir, max(0, total_size + projector_size - already_have)):
        return False

    # TAG-INJECTION site: repo_id/filename sit directly inside OPEN
    # [bold cyan]/[bold] tags, so they are escaped. len(all_parts)/len(missing)
    # are ints.
    if already_have > 0:
        console.print(
            f"Resuming [bold cyan]{escape(repo_id)}[/bold cyan] / "
            f"[bold]{escape(filename)}[/bold] "
            f"[dim](skipping first {format_size(already_have)})[/dim]"
        )
    elif len(all_parts) > 1:
        console.print(
            f"Pulling [bold cyan]{escape(repo_id)}[/bold cyan] / "
            f"[bold]{escape(filename)}[/bold] "
            f"[dim](split GGUF, {len(all_parts)} parts, "
            f"{len(missing)} to download)[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]{escape(repo_id)}[/bold cyan]"
                      + (f"@[bold]{escape(revision.ref)}[/bold]" if revision else "")
                      + f" / [bold]{escape(filename)}[/bold]")

    with _download_progress([base_dir / p for p in missing], total_size,
                            base_dir=stage_dir,
                            rel_parts=[_remote(p) for p in missing],
                            etags={_remote(p): e for p, e in etags.items()}) as _prog:
        for part in missing:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=_remote(part),
                    local_dir=str(stage_dir),
                    endpoint=_HF_ENDPOINT,
                    token=_hf_token(),
                    tqdm_class=meter_tqdm_class(),
                    **_file_revision(published.get(part), revision),
                )
                final = base_dir / part
                if Path(local) != final:
                    shutil.move(local, final)
            except Exception as e:
                console.print(f"[red]Download failed[/red] ({escape(part)}): "
                              f"{escape(str(e))}")
                # WITHOUT _prog.ok(): this return unwinds the context manager
                # cleanly, and only the absence of ok() stops it announcing 100%
                # for a download that failed.
                return False
        _prog.ok()

    # Verify the downloaded first part against the digest we will register it
    # under: HF's LFS metadata when known, else the user's --sha256. On
    # mismatch, delete the part(s) and fail.
    if digests:
        actual_digests = {}
        for part in all_parts:
            if part not in digests:
                continue
            actual = _verify_digest(base_dir / part).lower()
            actual_digests[part] = actual
            if actual != digests[part].lower():
                console.print(
                    f"[red]SHA256 mismatch![/red] Expected "
                    f"{escape(digests[part][:16])}…, got "
                    f"{escape(actual[:16])}… - deleting downloaded file(s)"
                    + (f" ({escape(part)})" if part != filename else "")
                )
                for each in all_parts:
                    p = base_dir / each
                    if p.exists():
                        p.unlink()
                return False
        shown = actual_digests.get(filename) or next(iter(actual_digests.values()))
        _report_success(f"[green]✓[/green] SHA256 verified: {escape(shown[:16])}…",
                        f"[green]OK[/green] SHA256 verified: {escape(shown[:16])}…")

    if register:
        unusable = _mm.gguf_unusable_reason(base_dir / filename)
        if unusable is not None:
            console.print(f"[red]Not registered:[/red] {escape(filename)}\n{escape(unusable)}")
            return False
        reg_type = model_type
        # One shared header probe, as in the "already downloaded" branch above,
        # for the freshly-downloaded case.
        gguf_meta = _mm.gguf_registry_metadata(base_dir / filename)
        if type_is_auto and reg_type == "llm":
            if _mm.gguf_adapter_kind(base_dir / filename) is not None:
                console.print("[dim]Detected as a LoRA adapter (GGUF metadata). "
                              "Attach it with 'localm adapter attach'.[/dim]")
                reg_type = "lora"
            elif _mm.gguf_is_mmproj(base_dir / filename):
                console.print("[dim]Detected as a vision projector (GGUF metadata).[/dim]")
                reg_type = "mmproj"
            elif _mm.gguf_embedding_signal(base_dir / filename):
                console.print(_embedding_detection_note(base_dir / filename))
                reg_type = "embedding"
            elif _mm.gguf_non_chat_model_type(gguf_meta.get("architecture")):
                console.print(_non_chat_detection_note(gguf_meta.get("architecture")))
                reg_type = _mm.gguf_non_chat_model_type(gguf_meta.get("architecture"))
            elif (media_type := _mm.gguf_media_component_type(base_dir / filename)) is not None:
                reg_type = media_type
                console.print(f"[dim]Detected as {_media_type_phrase(reg_type)} (tensor names "
                              "in the GGUF).[/dim]")
        if type_is_auto and not filename.lower().endswith(".gguf"):
            reg_type = _content_media_type(base_dir / filename, reg_type)
        mmproj_path = _mmproj_for_registration(
            reg_type, repo_id, filename, base_dir, dest_dir, mmproj_spec,
            pinned_revision=revision is not None)
        _mm._register(model_name, base_dir / filename, f"hf:{repo_id}",
                  sha256=verify_digest, model_type=reg_type, mmproj=mmproj_path,
                  architecture=gguf_meta.get("architecture"),
                  expert_count=gguf_meta.get("expert_count"),
                  **_register_revision(revision))
        # model_name is _sanitize_name()-derived (A-Za-z0-9._- only) and is
        # escaped anyway.
        _report_success(f"[green]✓[/green] [bold]{escape(model_name)}[/bold] is ready",
                        f"[green]OK[/green] [bold]{escape(model_name)}[/bold] is ready")
    else:
        # filename here is NOT charset-restricted (register=False routes here,
        # e.g. a ComfyUI dest_dir pull), and it sits in a tag-injection
        # position.
        _report_success(f"[green]✓[/green] [bold]{escape(filename)}[/bold] downloaded",
                        f"[green]OK[/green] [bold]{escape(filename)}[/bold] downloaded")
    return True




def _snapshot_files_failing_digest(dest: Path, repo_siblings, repo_id: str) -> list[str]:
    """Names of the files of the snapshot at *dest* that do not hash to the LFS
    sha256 *repo_siblings* (a ``model_info(files_metadata=True)`` listing) gives
    them. A listed file with a digest that is absent from disk counts as failing.
    Each failing file is deleted, so a second pull fetches it again; one that
    cannot be deleted is logged and still counts as failing."""
    from localm import pathsafe
    failing = []
    for sib in repo_siblings:
        lfs = getattr(sib, "lfs", None)
        published = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))             if lfs else None
        if not published:
            continue
        name = str(sib.rfilename)
        try:
            fp = pathsafe.confined_under(dest, name)
        except ValueError as e:
            logger.warning("repo %s lists an out-of-bounds filename (%s); not checking it",
                           repo_id, e)
            failing.append(name)
            continue
        if not fp.is_file():
            failing.append(name)
            continue
        actual = _verify_digest(fp, purpose=f"to check {name}").lower()
        if actual != str(published).lower():
            try:
                fp.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("could not delete the damaged %s: %s", fp, e)
            failing.append(name)
    return failing


def _snapshot_files_damaged(dest: Path, repo_siblings, repo_id: str) -> list[str]:
    """Names of the files of the snapshot at *dest* that are present but do not
    have the size, or the LFS sha256, *repo_siblings* gives them. A file that is
    absent is not listed (a download fetches it). Deletes nothing."""
    from localm import pathsafe
    damaged = []
    for sib in repo_siblings:
        name = str(sib.rfilename)
        try:
            fp = pathsafe.confined_under(dest, name)
        except ValueError:
            continue
        if not fp.is_file():
            continue
        size = getattr(sib, "size", None)
        if size is not None and fp.stat().st_size != size:
            damaged.append(name)
            continue
        lfs = getattr(sib, "lfs", None)
        published = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if published and _verify_digest(fp, purpose=f"to check {name}").lower() != str(published).lower():
            damaged.append(name)
    return damaged


def _git_blob_id(path: Path) -> str:
    """The git object id (SHA-1 of ``blob <size>\\0`` plus the content) of the
    file at *path*, the id a Hugging Face listing gives a file stored outside
    LFS."""
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SNAPSHOT_SIDE_FILES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


def _snapshot_matches_listing(dest: Path, repo_siblings) -> bool:
    """True when *dest* holds exactly the files *repo_siblings* lists: each at
    its stated size, an LFS file with a published sha256 (which
    :func:`_snapshot_files_damaged` compares), every other file with the git
    blob id the listing gives it, and no file the listing does not name outside
    ``.cache``, a top-level ``.git``, the claim marker and the operating
    system's folder files (:data:`_SNAPSHOT_SIDE_FILES`). False when a file
    carries no digest to compare, a subfolder is a link, or any part of the
    folder cannot be read. Reads every non-LFS file; deletes nothing."""
    from localm import pathsafe
    listed: set[str] = set()
    try:
        for sib in repo_siblings:
            name = str(sib.rfilename)
            fp = pathsafe.confined_under(dest, name)
            if not fp.is_file():
                return False
            size = getattr(sib, "size", None)
            if size is not None and fp.stat().st_size != size:
                return False
            lfs = getattr(sib, "lfs", None)
            if lfs:
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                if not sha:
                    return False
            else:
                blob = getattr(sib, "blob_id", None)
                if not blob or _git_blob_id(fp) != str(blob).lower():
                    return False
            listed.add(name.replace("\\", "/"))
        def _unreadable(error: OSError) -> None:
            raise error

        for root, dirs, files in os.walk(dest, onerror=_unreadable):
            rel_root = Path(root).relative_to(dest).as_posix()
            if rel_root == ".":
                dirs[:] = [d for d in dirs if d not in (".cache", ".git")]
            for d in dirs:
                sub = os.path.join(root, d)
                if os.path.islink(sub) or os.path.isjunction(sub):
                    return False
            for f in files:
                rel = f if rel_root == "." else f"{rel_root}/{f}"
                if (rel != _SNAPSHOT_CLAIM and rel not in listed
                        and f.lower() not in _SNAPSHOT_SIDE_FILES):
                    return False
    except (OSError, ValueError):
        return False
    return True


def _delete_snapshot_files(dest: Path, names: list[str]) -> list[str]:
    """Delete the files *names* of the snapshot at *dest*; the names of those
    that could not be deleted (in use, no permission) are returned."""
    from localm import pathsafe
    stuck = []
    for name in names:
        try:
            pathsafe.confined_under(dest, name).unlink(missing_ok=True)
        except (OSError, ValueError) as e:
            logger.warning("could not delete the damaged %s: %s", name, e)
            stuck.append(name)
    return stuck


def _snapshot_bytes_on_disk(dest: Path) -> int:
    """Bytes of *dest* that a snapshot resume would not re-fetch: the finished
    files, plus the temp files of the files still downloading. Excludes the
    rest of huggingface_hub's own .cache scratch, which is not part of the
    model."""
    total = 0
    try:
        files = list(dest.rglob("*"))
    except OSError:
        return 0
    pending = dest / ".cache" / "huggingface" / "download"
    for f in files:
        in_cache = ".cache" in f.parts
        if in_cache and not (pending in f.parents and f.name.endswith(".incomplete")):
            continue
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _delete_hf_temp_files(base_dir: Path, rels: "list[str] | None") -> int:
    """Delete the temp files (and their owner records) of downloads under
    *base_dir*: those of the files *rels*, or every one when *rels* is None.
    Returns the bytes freed.

    Raises :class:`PullInFlight` when a download of one of *rels* holds its
    download lock. With *rels* None only temp files whose owner is proven gone
    are deleted."""
    root = Path(base_dir) / ".cache" / "huggingface" / "download"
    if rels is None:
        if not root.is_dir():
            return 0
        return _unlink_temp_files(
            [f for f in root.rglob("*.incomplete")
             if _partial_owner_is_gone(f, lock_held=False)])
    from huggingface_hub._local_folder import _short_hash
    from huggingface_hub._local_folder import get_local_download_paths
    with contextlib.ExitStack() as held:
        targets: list[tuple[Path, str]] = []
        for rel in rels:
            paths = get_local_download_paths(Path(base_dir), rel)
            if not paths.lock_path.parent.is_dir():
                continue
            held.enter_context(_hold_download_lock(
                paths.lock_path, wait=_SNAPSHOT_LOCK_WAIT_SECONDS))
            targets.append((paths.metadata_path.parent,
                            _short_hash(paths.metadata_path.name) + "."))
        files = []
        for folder, prefix in targets:
            if folder.is_dir():
                files += [f for f in folder.glob("*.incomplete")
                          if f.name.startswith(prefix)]
        return _unlink_temp_files(files)


@contextlib.contextmanager
def _hold_download_lock(lock_path: Path, busy: str = (
        "a download of this file is running; stop it before deleting "
        "its partial data"), wait: float = 0.0):
    """Hold the file lock at *lock_path* for the block, trying for up to *wait*
    seconds; raises :class:`PullInFlight` with the text *busy* when a download
    still holds it then."""
    from filelock import FileLock, Timeout
    lock = FileLock(str(lock_path), timeout=wait)
    try:
        lock.acquire(timeout=wait, poll_interval=0.02)
    except Timeout as e:
        raise PullInFlight(busy) from e
    try:
        yield
    finally:
        lock.release()


def _snapshot_lock_path(model_name: str) -> Path:
    """The lock file serialising full-repo pulls into the models folder
    *model_name*: beside the models dir, in ``locks/``, so deleting the folder
    never touches it. The name is lower-cased because a case-insensitive
    filesystem resolves two spellings of a folder to one."""
    return _mm.MODELS_DIR.parent / "locks" / ("snapshot-" + model_name.lower() + ".lock")


@contextlib.contextmanager
def _snapshot_dest_lock(model_name: str):
    """Hold the cross-process lock on the models folder *model_name* for the
    block, trying for up to :data:`_SNAPSHOT_LOCK_WAIT_SECONDS`. The OS
    releases it when the holder exits, however it exits (on Windows a moment
    after the process is reaped, which the wait covers). See
    test_discard_right_after_the_holder_is_killed_deletes.

    Raises :class:`PullInFlight` when a download into the folder holds it, and
    :class:`PullLockUnavailable` when the lock cannot be taken at all."""
    path = _snapshot_lock_path(model_name)
    with contextlib.ExitStack() as held:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            held.enter_context(_hold_download_lock(path, busy=(
                f"a download into the folder '{model_name}' is already "
                "running."), wait=_SNAPSHOT_LOCK_WAIT_SECONDS))
        except OSError as e:
            raise PullLockUnavailable(
                f"could not take the download lock for '{model_name}' "
                f"({path}): {e}") from e
        yield


def _unlink_temp_files(files: "list[Path]") -> int:
    """Delete the temp files *files* and their owner records; returns the bytes
    freed."""
    freed = 0
    for f in files:
        size = _partial_size(f)
        try:
            f.unlink()
        except OSError as e:
            logger.warning("could not delete the partial download %s: %s", f, e)
            continue
        _unlink_quiet(_partial_owner_path(f))
        freed += max(0, size)
    return freed


def partial_download_kept(spec: str) -> Optional[str]:
    """What :func:`discard_partial_download` leaves on disk for *spec*, by the
    spec's shape: ``"civitai"`` (a ``civitai:`` spec is never discarded),
    ``"repo"`` (a full-repo download keeps every file that finished), else
    None. Does no filesystem or network access."""
    spec = _mm.resolve_spec(spec.strip())
    with contextlib.suppress(ValueError):
        spec = split_hf_revision(spec)[0]
    if spec.startswith("civitai:"):
        return "civitai"
    if spec.startswith(("http://", "https://")) or "/" not in spec:
        return None
    if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
        return None
    return "repo"


def discard_partial_download(spec: str, name: Optional[str] = None) -> int:
    """Delete the partial data a stopped pull of *spec* left behind: the temp
    files of its file or files, or the ``.part`` file of a direct URL. Finished
    files are never touched. *name* is the ``--name`` the pull was given
    (it names a full-repo download's folder). Returns the bytes freed.

    Raises :class:`PullInFlight` when a download of it is running."""
    spec = _mm.resolve_spec(spec.strip())
    ref = None
    if not spec.startswith(("http://", "https://")):
        try:
            spec, ref = split_hf_revision(spec)
        except ValueError:
            return 0
    if spec.startswith(("http://", "https://")):
        stem = _stem_from_url(spec)
        filename = _safe_models_filename(stem + ".gguf") if stem else None
        if filename is None:
            return 0
        part_file = _mm.MODELS_DIR / (filename + ".part")
        with _part_lock(filename):
            freed = max(0, _partial_size(part_file))
            _discard_part(part_file)
        return freed
    if "/" not in spec or spec.startswith("civitai:"):
        return 0
    if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
        if ":" in spec:
            remote = spec.split(":", 1)[1]
        else:
            remote = spec.rsplit("/", 1)[1]
        remote_dir, in_folder, filename = remote.rpartition("/")
        base_dir = _mm.MODELS_DIR / _REPO_FOLDER_STAGING if in_folder else _mm.MODELS_DIR
        if in_folder and not _safe_repo_subdir(remote_dir, base_dir):
            return 0
        parts = split_gguf_parts(filename) or [filename]
        if any(_safe_models_filename(p, base_dir) is None for p in parts):
            return 0
        rels = [f"{remote_dir}/{p}" if remote_dir else p for p in parts]
        return _delete_hf_temp_files(base_dir, rels)
    model_name = _sanitize_name(
        name or spec.split("/")[-1] + (f"@{ref}" if ref else ""))
    with _snapshot_dest_lock(model_name):
        return _delete_hf_temp_files(_mm.MODELS_DIR / model_name, None)


def _warn_if_repo_ships_code(dest: Path, repo_id: str) -> None:
    """Say plainly when a downloaded repo contains Python.

    ``snapshot_download`` fetches the WHOLE repo, so a HuggingFace repo's own
    .py lands on disk like any other file. It is inert on disk:
    ``hf_trust_remote_code`` defaults to False and a model that needs custom
    code is refused with an explanation instead of being run (see
    inference/backends/hf.py).

    This is a warning rather than an ``allow_patterns`` allowlist: a model needs
    more than weights plus a tokenizer - chat templates (.jinja), merges.txt,
    shard index files, per-component subdirectories for multimodal repos - and a
    pattern list that misses one silently produces a broken, half-downloaded
    model. Everything is fetched, and the presence of code is made VISIBLE as it
    arrives.
    """
    # repo_id/e/file names below all come from a REMOTE repo and are
    # interpolated into a Rich markup string, so they are escaped. Unescaped, a
    # file named '[/b]evil.py' raises MarkupError, which would abort the pull
    # between the download and the _register call and leave the model on disk
    # and unregistered; one named '[red]x.py' is parsed as a style tag and
    # vanishes from the notice. Imported up front so BOTH branches of this
    # function, including the early return, get the same protection.
    from rich.markup import escape
    try:
        py = sorted(p for p in dest.rglob("*.py") if p.is_file())
    except OSError as e:
        # A failed scan does not fail an otherwise-good download, and is
        # reported as "not checked" rather than as "no code found".
        console.print(f"[yellow]Could not check {escape(repo_id)} for bundled "
                      f"code: {escape(str(e))}[/yellow]")
        return
    if not py:
        return
    shown = ", ".join(escape(p.name) for p in py[:5])
    if len(py) > 5:
        shown += f", and {len(py) - 5} more"
    console.print(
        f"[yellow]Note:[/yellow] {escape(str(repo_id))} ships {len(py)} "
        f"Python file(s) ({shown}).")
    console.print(
        "[dim]  localm will NOT run them: model-bundled custom code is disabled "
        "by default. A model that requires it is refused with an explanation "
        "unless you enable 'hf_trust_remote_code'.[/dim]")


def _resolve_snapshot_type(dest: Path, model_type: str) -> str:
    """The type to register a downloaded HF snapshot under.

    The pipeline_tag probe runs BEFORE the download and answers 'unknown' for
    any repo without an exact tag (common for base and older repos). The files
    now on disk are a HARDER signal than that API record, so when the probe
    could not resolve, the real config.json is classified with the same
    deterministic reader ``add_local`` uses.

    A probe that DID resolve (lora/vae/embedding/...) is authoritative and is
    never overridden here, and an unresolvable config.json stays 'unknown':
    there is no 'llm' fallback.
    """
    if model_type != "unknown":
        return model_type
    detected, _gmeta = _detect_local_model_type(dest, is_gguf=False, is_hf=True)
    logger.info("Snapshot %s typed %r from its downloaded config.json (the HF "
                "pipeline_tag probe could not resolve it)", dest.name, detected)
    if detected != "unknown":
        # detected comes from _detect_local_model_type's fixed type-label
        # vocabulary and is escaped anyway; it sits in a tag-injection position.
        from rich.markup import escape
        console.print(f"[green]Determined model type from config.json:[/green] "
                      f"[bold]{escape(detected)}[/bold]")
    else:
        # Still no hard signal even from the real files. The model stays
        # runnable by name, it is just not auto-loaded for chat.
        console.print(
            "[yellow]Could not determine this model's type[/yellow] - "
            "registering it as 'unknown'. Run it by name, or set its type "
            "later: [bold]localm set-type <name> <type>[/bold]")
    return detected


def _snapshot_is_complete(dest: Path, repo_siblings, repo_id: str) -> bool:
    """True when every file the remote repo listing names is present under *dest*
    at its stated size.

    A disk-full mid-download can leave config.json - usually one of the smallest,
    earliest files - on disk while weight shards are still missing, so an
    existence check on config.json alone would register a broken snapshot as a
    ready model on the very next retry. *repo_siblings* is None when the listing
    could not be fetched (offline / API error), which degrades to exactly that
    weaker check."""
    # Imported inside the function, not at module scope, so the CLI download
    # path does not pull fastapi (which pathsafe imports for confined_name's
    # HTTPException) just to validate a filename.
    from localm import pathsafe
    if not (dest / "config.json").exists():
        return False
    if repo_siblings is None:
        return True
    for sib in repo_siblings:
        # rfilename comes from the remote model_info response and is not a
        # trusted path component: pathlib lets an absolute or drive-qualified
        # value REPLACE dest entirely, and a '..' walks out of it. Unconfined,
        # this check would stat files anywhere on disk, so a repo listing names
        # that exist off-tree could make an EMPTY download look complete.
        # Confine before the stat, never after.
        try:
            fp = pathsafe.confined_under(dest, str(sib.rfilename))
        except ValueError as e:
            # An out-of-bounds name is reported, and the snapshot counts as
            # INCOMPLETE, so it re-downloads rather than registering a
            # half-present tree.
            logger.warning("repo %s lists an out-of-bounds filename (%s); "
                           "treating the local snapshot as incomplete", repo_id, e)
            return False
        if not fp.is_file():
            return False
        if sib.size is not None and fp.stat().st_size != sib.size:
            return False
    return True


def _pull_hf_snapshot(
    repo_id: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
    revision: Optional[HfRevision] = None,
) -> bool:
    """Download a complete HuggingFace model repo (for transformers/HF format models).

    *revision*, when given, is the branch, tag or commit already resolved by
    ``resolve_hf_revision``: the listing, the download and the digest check all
    use its commit, the registry entry records it, and the default folder name
    carries it."""
    # repo_id is the raw caller-supplied spec (CLI/GUI/MCP), dest is derived
    # from it via _sanitize_name, and every exception's text is remote-sourced;
    # all are escaped. Several sites below interpolate repo_id/dest directly
    # inside an OPEN Rich tag, a markup-TAG-INJECTION position.
    from rich.markup import escape
    # A full-repo snapshot is many files and has no single digest to check
    # --sha256 against, so the flag is refused rather than ignored.
    if expected_sha256:
        console.print(
            "[red]--sha256 is not supported for a full HuggingFace repo[/red] "
            f"({escape(repo_id)}): a snapshot has many files and no single "
            "digest to verify. Drop --sha256, or pull a single file with "
            "[bold]owner/repo:file.gguf --sha256 <hash>[/bold]."
        )
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

    default_name = repo_id.split("/")[-1] + (f"@{revision.ref}" if revision else "")
    model_name = _sanitize_name(name or default_name)

    try:
        with _snapshot_dest_lock(model_name):
            return _pull_hf_snapshot_locked(
                repo_id, model_name, redownload, model_type, revision,
                snapshot_download)
    except PullLockUnavailable as e:
        console.print(f"[red]Could not lock the download folder:[/red] {escape(str(e))}")
        return False
    except PullInFlight as e:
        console.print(f"[red]Download already in progress:[/red] {escape(str(e))} "
                      "Wait for it to finish, or pull with -n <a-different-name>.")
        return False


def _pull_hf_snapshot_locked(
    repo_id: str,
    model_name: str,
    redownload: bool,
    model_type: str,
    revision: Optional[HfRevision],
    snapshot_download,
) -> bool:
    """The body of :func:`_pull_hf_snapshot`, run while holding the lock on the
    folder *model_name* (see :func:`_snapshot_dest_lock`); *snapshot_download*
    is ``huggingface_hub.snapshot_download``."""
    from rich.markup import escape
    dest = _mm.MODELS_DIR / model_name

    # Fetch the repo's file listing once, used both to verify an existing
    # download is genuinely complete (every file present with a matching size,
    # not just config.json) and to size the disk-space preflight and progress
    # display below. A disk-full mid-download can leave config.json, one of the
    # smallest and earliest files, on disk while weight shards are missing.
    repo_siblings = None
    commit = _rev_commit(revision)
    total_size = 0
    try:
        info = _hf_api().model_info(repo_id, files_metadata=True, **_revision_kwargs(commit))
        repo_siblings = info.siblings
        total_size = sum(getattr(s, "size", None) or 0 for s in repo_siblings)
        info_sha = getattr(info, "sha", None)
        commit = _valid_commit(info_sha) or commit
    except Exception as e:
        # Offline or API error: fall back to a config.json-only completeness
        # check below and an indeterminate (0) progress total.
        logger.debug("could not fetch file listing for %s (%s); falling back "
                     "to a config.json-only completeness check", repo_id, e)

    damaged: list[str] = []
    if dest.exists() and repo_siblings is not None:
        damaged = _snapshot_files_damaged(dest, repo_siblings, repo_id)
    claimed = _snapshot_claim_held_by_other(dest, repo_id, revision) if dest.exists() else None
    adopted = (claimed == ("", None) and not damaged and repo_siblings is not None
               and _snapshot_matches_listing(dest, repo_siblings))
    if adopted:
        claimed = None
    if claimed is not None:
        claim_repo, claim_revision = claimed
        what = ("files localm cannot match to a repo or revision" if not claim_repo
                else "another revision of it" if claim_repo.casefold() == repo_id.casefold()
                else f"{escape(claim_repo)}"
                + (f"@{escape(claim_revision)}" if claim_revision else ""))
        console.print(
            f"[red]Refusing to pull {escape(repo_id)}"
            + (f"@{escape(revision.ref)}" if revision else "")
            + f" into {escape(str(dest))}:[/red] this folder holds "
            + ("" if not claim_repo else "an unfinished download of ")
            + f"{what}. Pull with "
            "[bold]-n <a-different-name>[/bold], or delete the folder first.")
        return False
    if (dest.exists() and not damaged
            and not _holds_other_revision(dest, repo_id, revision)
            and (adopted or _snapshot_is_complete(dest, repo_siblings, repo_id))):
        console.print(f"[yellow]Already downloaded:[/yellow] {escape(model_name)}")
        registered = _mm._register_with_dedup(
            model_name, dest, f"hf:{repo_id}",
            model_type=_resolve_snapshot_type(dest, model_type),
            **_register_revision(revision))
        if not registered:
            console.print(
                f"[yellow]{escape(repo_id)} is in {escape(str(dest))}, but could not "
                f"be registered as '{escape(model_name)}'[/yellow] (see message above). "
                "Retry with a different -n name, or 'localm alias' it in.")
            return False
        _clear_snapshot_claim(dest)
        return True
    repairing = bool(damaged) or (
        dest.exists() and any(
            _entry_is_hf_source(_mm.load_registry()[n], repo_id, revision)
            for n in find_aliases_by_path(dest, _mm.load_registry())))

    # Same repo already pulled under a different name?
    if not redownload and not repairing:
        reg = _mm.load_registry()

        def _is_same_repo(info) -> bool:
            # Skip a malformed sibling entry (non-dict, or a null, non-string
            # or empty path) so one corrupt entry cannot crash the pull-dedup
            # scan: a str entry's .get or Path(None) would. Every entry goes
            # through _entry_path.
            epath = _mm._entry_path(info)
            if epath is None:
                return False
            return _entry_is_hf_source(info, repo_id, revision) and Path(epath).is_dir()

        same_source = sorted(n for n, info in reg.items() if _is_same_repo(info))
        if same_source:
            # Registry keys are _sanitize_name()-derived and escaped anyway;
            # quoted by hand rather than via !r, because escape() after repr()
            # re-escapes repr()'s own backslash.
            names = ", ".join(f"'{escape(n)}'" for n in same_source)
            console.print(
                f"[yellow]This repo is already downloaded - registered as "
                f"{names}[/yellow]"
            )
            if not sys.stdin.isatty():
                console.print("[dim]Non-interactive session - skipping. "
                              "Use --redownload to force.[/dim]")
                return True
            import click
            choice = click.prompt(
                f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
                type=click.Choice(["a", "d", "s"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "s":
                return True
            if choice.lower() == "a":
                alias_model(same_source[0], model_name)
                return True

    # COLLISION CHECK: model_name comes from _sanitize_name, a LOSSY coercion
    # (any run of disallowed characters collapses to a single '-', consecutive
    # dots collapse to one) used directly as both the MODELS_DIR subdirectory
    # name and the registry key, with no uniqueness check upstream. Two
    # different repos, or a --name reused across two pulls, can compute the same
    # dest, and snapshot_download's local_dir=dest MERGES into whatever is
    # already there rather than clearing it first.
    #
    # The signal: is there ALREADY a registry entry pointing at this exact dest,
    # for a DIFFERENT source? A resumable partial of the SAME repo has no
    # registry entry yet, so it does not false-positive; a redownload of the
    # model already registered here has a matching source, so it is excluded.
    if dest.exists():
        reg_now = _mm.load_registry()
        foreign = [n for n in find_aliases_by_path(dest, reg_now)
                  if not _entry_is_hf_source(reg_now[n], repo_id, revision)]
        if foreign:
            # repo_id/dest sit directly inside the OPEN [red]...[/red] tag, a
            # tag-injection position. `foreign` entries are registry keys,
            # quoted by hand rather than via !r.
            foreign_names = ", ".join(f"'{escape(n)}'" for n in foreign)
            console.print(
                f"[red]Refusing to pull {escape(repo_id)} into "
                f"{escape(str(dest))}:[/red] this "
                f"folder already holds a DIFFERENT model, registered as "
                f"{foreign_names}. Pulling here would "
                f"silently mix the two repos' files together. Pull with "
                f"[bold]-n <a-different-name>[/bold] to give this repo its "
                "own directory."
            )
            return False

    if damaged:
        shown = ", ".join(escape(n) for n in damaged[:3]) + (" ..." if len(damaged) > 3 else "")
        stuck = _delete_snapshot_files(dest, damaged)
        if stuck:
            console.print(
                f"[red]Could not delete {len(stuck)} damaged file(s) of {escape(model_name)}"
                f"[/red] ({escape(', '.join(stuck[:3]))}): a running model or another program "
                "may be using them. Unload the model and pull again.")
            return False
        console.print(
            f"[yellow]{len(damaged)} file(s) of {escape(model_name)} do not match the size or "
            f"digest Hugging Face publishes ({shown}) - deleted, fetching them again."
            "[/yellow]")

    def _disk_bytes() -> int:
        return _snapshot_bytes_on_disk(dest)

    # Only the bytes still MISSING need room: snapshot_download resumes on top
    # of whatever already landed in dest. _pull_gguf_file sums only the
    # `missing` parts and _pull_url uses (total - already_have); this matches
    # them.
    if not _mm._check_disk_space(_mm.MODELS_DIR,
                                 max(0, total_size - _disk_bytes())):
        return False

    _mm.ensure_dirs()
    from ..storekit import atomic_write
    dest.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write(dest / _SNAPSHOT_CLAIM,
                     json.dumps({"repo": repo_id,
                                 "revision": revision.ref if revision else None}))
    except OSError as e:
        console.print(
            f"[red]Could not start the download into {escape(str(dest))}:[/red] "
            f"localm could not write {_SNAPSHOT_CLAIM} there ({escape(str(e))}).")
        return False
    # TAG-INJECTION site: repo_id/dest sit directly inside OPEN
    # [bold cyan]/[bold] tags.
    console.print(
        f"Downloading full model [bold cyan]{escape(repo_id)}[/bold cyan]"
        + (f" at [bold]{escape(revision.ref)}[/bold]" if revision else "")
        + f" -> [bold]{escape(str(dest))}[/bold]"
    )
    console.print("[dim]This may take a while for large models...[/dim]")

    try:
        _ensure_hf_resumable_download()
        with _snapshot_progress(_disk_bytes, total_size, label=repo_id) as _prog:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                endpoint=_HF_ENDPOINT,
                token=_hf_token(),
                tqdm_class=meter_tqdm_class(),
                **_revision_kwargs(commit),
            )
            _prog.ok()
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {escape(str(e))}")
        if damaged:
            console.print("The damaged files were deleted; pull again to fetch them.")
        return False

    if repo_siblings is None:
        console.print(
            f"[yellow]Note:[/yellow] the file digests of {escape(repo_id)} could not be "
            "read, so the downloaded files are not checked against them.")
        logger.warning("no file digests for %s; the snapshot is not integrity-checked", repo_id)
    else:
        damaged = _snapshot_files_failing_digest(dest, repo_siblings, repo_id)
        if damaged:
            shown = ", ".join(escape(n) for n in damaged[:3]) + (" ..." if len(damaged) > 3 else "")
            console.print(
                f"[red]SHA256 mismatch![/red] {len(damaged)} downloaded file(s) of "
                f"{escape(repo_id)} do not match the digests Hugging Face publishes "
                f"({shown}) - deleted. Pull again to fetch them; the model was not registered.")
            return False

    _warn_if_repo_ships_code(dest, repo_id)
    # _register_with_dedup, the same dedup-aware registration _pull_gguf_file
    # routes through.
    #
    # Its bool return MUST be checked, not assumed True: the dest-collision
    # block above only guards a foreign occupant already sitting at THIS path,
    # and says nothing about model_name itself already being taken by a
    # DIFFERENT path (dest may not have existed, so that block never ran).
    # _register_with_dedup can decline in that case, non-interactively, and by
    # then the download has written real bytes to disk.
    registered = _mm._register_with_dedup(
        model_name, dest, f"hf:{repo_id}",
        model_type=_resolve_snapshot_type(dest, model_type), refresh=True,
        **_register_revision(revision))
    if not registered:
        # TAG-INJECTION site: repo_id/dest sit directly inside the OPEN
        # [yellow]...[/yellow] tag. model_name is _sanitize_name()-derived and
        # escaped anyway.
        console.print(
            f"[yellow]{escape(repo_id)} was downloaded to {escape(str(dest))}, "
            f"but could not be registered as '{escape(model_name)}'[/yellow] "
            "(see message above) - the files are on disk. Retry with a "
            "different -n name, or 'localm alias' it in."
        )
        return False
    _clear_snapshot_claim(dest)
    _report_success(
        f"[green]✓[/green] [bold]{escape(model_name)}[/bold] downloaded to "
        f"{escape(str(dest))}",
        f"[green]OK[/green] [bold]{escape(model_name)}[/bold] downloaded to "
        f"{escape(str(dest))}")
    return True




def _ssrf_resolve_final_url(url: str) -> str:
    """Follow the redirect chain HEAD-only, re-validating EVERY hop against the
    netpolicy SSRF guard, and return the final URL. Model pulls legitimately
    redirect (HuggingFace -> CDN), so the chain is followed, but every hop is
    checked rather than left to requests' automatic, UNCHECKED redirect
    following, which a public URL could use to bounce the download into
    127.0.0.1, 169.254.169.254 or an RFC1918 service. Each HEAD is IP-pinned to
    the validated address so the connect cannot rebind off the checked host.
    net_allow_model_downloads exempts this from the net_mode=off floor, same
    as pull_model's own top-level gate; the SSRF/private-address checks apply
    regardless. Raises NetworkPolicyError if any hop resolves to a non-public
    host or cannot be resolved to a validated address."""
    import urllib.parse

    from localm import netpolicy
    allow_off = netpolicy.downloads_allowed_when_off()
    current = url
    for _ in range(6):
        netpolicy.check_url(current, allow_when_off=allow_off)
        try:
            resp = netpolicy.pinned_request(
                "HEAD", current, allow_redirects=False, timeout=10)
        except netpolicy.NetworkPolicyError:
            raise                       # a policy refusal / rebind must NOT be swallowed into a silent GET
        except Exception:
            break                       # unreachable HEAD -> let the GET report it
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                break
            current = urllib.parse.urljoin(current, loc)
            continue
        break
    netpolicy.check_url(current, allow_when_off=allow_off)   # final target, revalidated
    return current


def _hf_endpoint_refusal() -> Optional[str]:
    """None when the network policy allows a request to the Hugging Face
    endpoint (``check_url`` on ``_HF_ENDPOINT``, with
    net_allow_model_downloads applied to the net_mode=off floor), else the
    refusal's message."""
    from localm import netpolicy
    try:
        netpolicy.check_url(
            _HF_ENDPOINT, allow_when_off=netpolicy.downloads_allowed_when_off())
    except netpolicy.NetworkPolicyError as e:
        return str(e)
    return None


def _policy_head(url: str):
    """HEAD *url*, following at most 5 redirects. Every hop passes
    ``check_url`` (net_allow_model_downloads applied to the net_mode=off
    floor) and is sent over the pinned transport. Returns the last response
    with ``history`` set to the earlier hops' responses, first hop first.
    Raises NetworkPolicyError when a hop is refused, ``UnresolvableHost`` (a
    NetworkPolicyError) when a hop's host cannot be resolved; transport
    errors propagate. The configured Hugging Face token is sent only to the
    hub endpoint's own host, never to a redirect target."""
    import urllib.parse

    from localm import netpolicy
    from localm.discover import _hf_auth_headers
    allow_off = netpolicy.downloads_allowed_when_off()
    hub_host = urllib.parse.urlparse(_HF_ENDPOINT).hostname
    token = _hf_token()
    history: list = []
    current = url
    while True:
        netpolicy.check_url(current, allow_when_off=allow_off)
        headers = (_hf_auth_headers(token)
                   if urllib.parse.urlparse(current).hostname == hub_host
                   else None)
        resp = netpolicy.pinned_request(
            "HEAD", current, allow_redirects=False, timeout=10, headers=headers)
        location = (resp.headers.get("Location")
                    if resp.status_code in (301, 302, 303, 307, 308) else None)
        if not location or len(history) == 5:
            resp.history = history
            return resp
        history.append(resp)
        current = urllib.parse.urljoin(current, location)


class PullInFlight(Exception):
    """Another process is already downloading to this destination."""


class PullLockUnavailable(PullInFlight):
    """The lock on a download destination cannot be taken at all (the lock
    file cannot be created or opened), whether or not anything is running."""


_SNAPSHOT_LOCK_WAIT_SECONDS = 1.0


def _part_lock_dir(filename: str) -> Path:
    """Where the lock for ``<filename>.part`` lives.

    BESIDE the models dir, never inside it: a lock directory under ``models/``
    would sit in the path of the tree ``sync_models_dir`` walks. The filename
    has already been through ``_safe_models_filename``, so it is a single path
    component and cannot escape this directory.
    """
    return _mm.MODELS_DIR.parent / "locks" / ("pull-" + filename + ".lock")


_LOCK_STAGING_PREFIX = ".pull-acq-"
_LOCK_TOMBSTONE_PREFIX = ".pull-tomb-"
_LOCK_ATTEMPTS = 50
_RENAME_RETRIES = 50
_RENAME_RETRY_DELAY = 0.02


def _read_lock_record(d: Path):
    """The owner record in directory *d*, or None when it cannot be read."""
    try:
        rec = json.loads((d / "owner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return rec if isinstance(rec, dict) else None


def _part_lock_owner(d: Path):
    """The recorded holder of lock *d*, or None when that cannot be read."""
    return _read_lock_record(d)


def _record_holder_is_gone(rec, where: Path) -> bool:
    """:func:`_part_lock_holder_is_gone` for the owner record *rec*, which
    may be None, stored in directory *where*."""
    if rec is None:
        return False
    try:
        pid = int(rec.get("pid", -1))
    except (TypeError, ValueError):
        return False
    space = rec.get("space")
    if space is not None and not _same_pid_space(rec, where):
        return False
    if pid == os.getpid():
        return False
    if not instances.pid_alive(pid):
        return True
    return space is not None and instances.start_identity_differs(
        rec.get("start"), instances.process_start_identity(pid))


def _part_lock_holder_is_gone(d: Path) -> bool:
    """True only when the recorded holder is PROVEN gone: its pid is dead, or
    the record names this pid space (see :func:`_same_pid_space`) and the live
    pid's start identity differs from the one the holder recorded (see
    :func:`localm.instances.process_start_identity`). A record from another
    pid space is never gone; one that names no pid space is judged by pid
    liveness alone.

    Staleness is decided by PID LIVENESS and that identity, never by elapsed
    time or the wall clock. Any fixed timeout eventually reclaims a live
    holder's lock and recreates the exact corruption this exists to prevent,
    and a large model on a slow link is precisely the download that outlives a
    generous timeout.

    Every uncertainty KEEPS the lock: an owner record that cannot be read (a
    half-written lock, a permission error) is not evidence its owner died, nor
    is a start identity that is missing, unreadable or not comparable.
    ``instances.pid_alive`` is conservative in the same direction - it reports
    True when it genuinely cannot tell - so this composes rather than
    re-deciding.
    """
    return _record_holder_is_gone(_part_lock_owner(d), d.parent)


def _is_empty_dir(d: Path) -> bool:
    """True when *d* is a directory with no entries."""
    try:
        with os.scandir(d) as entries:
            return next(entries, None) is None
    except OSError:
        return False


def _lock_is_reclaimable(d: Path) -> bool:
    """True when lock *d* may be taken over: its recorded holder is proven
    gone (see :func:`_part_lock_holder_is_gone`), or it is an empty directory
    with no record."""
    if _part_lock_owner(d) is None:
        return _is_empty_dir(d)
    return _part_lock_holder_is_gone(d)


def _rename_dir(src: Path, dst: Path) -> None:
    """``os.rename(src, dst)``. On Windows a PermissionError, which a file
    open inside *src* causes, is retried up to ``_RENAME_RETRIES`` times,
    ``_RENAME_RETRY_DELAY`` seconds apart."""
    for attempt in range(_RENAME_RETRIES):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == _RENAME_RETRIES - 1:
                raise
            time.sleep(_RENAME_RETRY_DELAY)


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Rename directory *src* to *dst*, raising FileExistsError when *dst*
    exists.

    On Windows the rename itself refuses an existing *dst*. Elsewhere *dst* is
    checked first and the rename refuses a non-empty *dst*; an empty directory
    created at *dst* between the check and the rename is replaced.
    """
    if sys.platform == "win32":
        _rename_dir(src, dst)
        return
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST),
                                  str(dst)) from e
        raise


def _tombstone_path(locks: Path) -> Path:
    """A new, unused name in *locks* for a lock being removed."""
    return locks / (_LOCK_TOMBSTONE_PREFIX + uuid.uuid4().hex)


def _remove_tombstone(tomb: Path) -> None:
    """Remove *tomb*; what cannot be removed now is left for
    :func:`_sweep_lock_leftovers`."""
    shutil.rmtree(tomb, ignore_errors=True)
    if os.path.lexists(tomb):
        logger.debug("left %s in place; the next pull removes it", tomb)


def _sweep_lock_leftovers(locks: Path) -> None:
    """Remove from *locks* every tombstone, and every staging directory that
    is empty or whose record names a holder proven gone (see
    :func:`_part_lock_holder_is_gone`)."""
    try:
        with os.scandir(locks) as entries:
            paths = [Path(e.path) for e in entries]
    except OSError as e:
        logger.debug("could not list %s for download lock leftovers: %s",
                     locks, e)
        return
    for p in paths:
        if p.name.startswith(_LOCK_TOMBSTONE_PREFIX):
            _remove_tombstone(p)
        elif p.name.startswith(_LOCK_STAGING_PREFIX):
            rec = _read_lock_record(p)
            if rec is None:
                try:
                    os.rmdir(p)
                except OSError as e:
                    logger.debug("left %s in place: %s", p, e)
            elif _record_holder_is_gone(rec, locks):
                shutil.rmtree(p, ignore_errors=True)


_GUARD_BUSY_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK,
                                errno.EDEADLK})


def _lock_held_error(d: Path, filename: str) -> PullInFlight:
    """The refusal for lock *d*, naming its recorded holder."""
    rec = _part_lock_owner(d) or {}
    who = rec.get("pid", "unknown")
    return PullInFlight(
        f"{filename} is already being downloaded by process {who}. "
        f"Wait for it to finish, or - if you are certain no download "
        f"is running - remove {d}.")


def _lock_guard_file(f) -> None:
    """Take a non-blocking, exclusive OS advisory lock on the open file *f*;
    raises OSError when it cannot."""
    if sys.platform == "win32":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_guard_file(f) -> None:
    """Release the lock :func:`_lock_guard_file` took on *f*."""
    if sys.platform == "win32":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _reclaim_guard_path(d: Path, filename: str) -> Path:
    """The reclaim guard file for *filename*'s lock *d*: beside it, named
    ``pull-<filename>.grd``, one character shorter than the lock's own name.
    Two spellings of *filename* that the filesystem resolves to one lock
    resolve to one guard file."""
    return d.parent / ("pull-" + filename + ".grd")


@contextlib.contextmanager
def _reclaim_guard(d: Path, filename: str):
    """Hold an exclusive OS advisory lock on the reclaim guard file of lock
    *d* (see :func:`_reclaim_guard_path`) for the block.

    The guard file is created when missing and never deleted, and the OS
    releases the lock when its holder exits, however it exits. Raises
    :class:`PullInFlight` when another process holds the guard, or when the
    lock cannot be taken at all; that message names the error and *d*.
    """
    guard = _reclaim_guard_path(d, filename)
    try:
        f = open(guard, "a+b")
    except OSError as e:
        raise PullInFlight(
            f"could not open {guard} to take over the stale download lock for "
            f"{filename}: {e}. If you are certain no download is running, "
            f"remove {d}.") from e
    with f:
        try:
            _lock_guard_file(f)
        except OSError as e:
            if e.errno in _GUARD_BUSY_ERRNOS:
                raise PullInFlight(
                    f"{filename} is already being taken over by another "
                    f"download.") from e
            raise PullInFlight(
                f"could not lock {guard} to take over the stale download lock "
                f"for {filename}: {e}. If you are certain no download is "
                f"running, remove {d}.") from e
        try:
            yield
        finally:
            try:
                _unlock_guard_file(f)
            except OSError as e:
                logger.debug("could not unlock %s (%s); closing the file "
                             "releases it", guard, e)


def _write_lock_record(staging: Path, payload: str) -> None:
    """Write *payload* to ``owner.json`` in *staging* and flush it to disk."""
    with open(staging / "owner.json", "x", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _stage_lock_record(locks: Path, filename: str, token: str) -> Path:
    """A new staging directory in *locks* holding this process's owner record
    for *filename*'s lock, carrying *token*. A staging directory removed
    before its record is written is replaced by a new one. Raises
    :class:`PullInFlight` when the record cannot be written."""
    # `space`, `host`, `local` and `start` are what _part_lock_holder_is_gone
    # compares and `token` is what the release checks; `started` is only read
    # by a human inspecting the lock.
    payload = json.dumps({"pid": os.getpid(), "filename": filename,
                          "space": _pid_space_id(),
                          "host": instances.host_id(),
                          "local": instances.path_is_local(locks),
                          "start": instances.process_start_identity(os.getpid()),
                          "token": token, "started": time.time()})
    for _ in range(_LOCK_ATTEMPTS):
        staging = locks / (_LOCK_STAGING_PREFIX + uuid.uuid4().hex)
        try:
            staging.mkdir()
            _write_lock_record(staging, payload)
            return staging
        except FileNotFoundError:
            continue
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise PullInFlight(
                f"could not record ownership of the download lock for "
                f"{filename}: {e}") from e
    raise PullInFlight(
        f"could not record ownership of the download lock for {filename}")


def _take_over_stale_lock(d: Path, filename: str, staging: Path) -> None:
    """Take lock *d* while holding its reclaim guard, by renaming *staging*
    onto it: directly when it is already gone, or after moving it to a
    tombstone when :func:`_lock_is_reclaimable` holds for it now. Raises
    :class:`PullInFlight` otherwise."""
    for _ in range(_LOCK_ATTEMPTS):
        try:
            _rename_noreplace(staging, d)
            return
        except FileExistsError:
            pass
        except OSError as e:
            raise PullInFlight(
                f"could not take the download lock for {filename}: {e}") from e
        if not _lock_is_reclaimable(d):
            if os.path.lexists(d):
                raise _lock_held_error(d, filename)
            continue
        tomb = _tombstone_path(d.parent)
        try:
            _rename_dir(d, tomb)
        except FileNotFoundError:
            continue
        except OSError as e:
            raise PullInFlight(
                f"could not take over the stale download lock for {filename}: "
                f"{e}. If you are certain no download is running, remove {d}.") from e
        try:
            _rename_noreplace(staging, d)
        except FileExistsError as exc:
            raise PullInFlight(
                f"{filename} is already being downloaded by another process.") from exc
        except OSError as e:
            raise PullInFlight(
                f"could not take the download lock for {filename}: {e}") from e
        finally:
            _remove_tombstone(tomb)
        return
    raise PullInFlight(
        f"{filename} is already being downloaded by another process.")


def _acquire_part_lock(d: Path, filename: str, staging: Path) -> None:
    """Take lock *d* by renaming *staging* onto it. A lock that is released
    while this looks at it is retried; a reclaimable one (see
    :func:`_lock_is_reclaimable`) is taken over under its reclaim guard.
    Raises :class:`PullInFlight` when someone else holds it."""
    for _ in range(_LOCK_ATTEMPTS):
        try:
            _rename_noreplace(staging, d)
            return
        except FileExistsError:
            pass
        except OSError as e:
            raise PullInFlight(
                f"could not take the download lock for {filename}: {e}") from e
        if _lock_is_reclaimable(d):
            with _reclaim_guard(d, filename):
                _take_over_stale_lock(d, filename, staging)
            return
        if os.path.lexists(d):
            raise _lock_held_error(d, filename)
    raise PullInFlight(
        f"{filename} is already being downloaded by another process.")


def _release_part_lock(d: Path, filename: str, token: str) -> None:
    """Move lock *d* to a tombstone and remove the tombstone, when its record
    still carries *token*; otherwise leave it in place and log a warning."""
    if not d.exists():
        logger.warning("the download lock at %s for %s disappeared while it "
                       "was held", d, filename)
        return
    rec = _part_lock_owner(d)
    if rec is None or rec.get("token") != token:
        logger.warning(
            "left the download lock at %s in place: its record is no longer "
            "this download of %s", d, filename)
        return
    tomb = _tombstone_path(d.parent)
    try:
        _rename_dir(d, tomb)
    except OSError as e:
        logger.warning(
            "could not release the download lock at %s (%s) - a later pull "
            "of %s will refuse while this process runs", d, e, filename)
        return
    _remove_tombstone(tomb)


@contextlib.contextmanager
def _part_lock(filename: str):
    """Serialise everything writing ``<filename>.part``, ACROSS PROCESSES.

    The contenders are separate processes, not threads: the GUI starts a pull
    by spawning ``localm pull`` as a child, and a user can run the same command
    in a terminal at the same time, so a ``threading.Lock`` would serialise
    nothing.

    The lock is taken by renaming a staging directory that already holds the
    owner record onto the lock path (see :func:`_rename_noreplace`), so the
    lock path never exists without its record, and the rename refuses when
    the lock path exists.

    Unserialised, two pulls of the same URL open one ``.part`` concurrently and
    the second one's ``already_have`` read decides append-or-truncate from a
    size the first is still changing, interleaving writes into a single file.

    A stale lock, or an empty directory at the lock path, is taken over only
    while holding its reclaim guard (see :func:`_reclaim_guard`), after
    re-checking it at that moment. Release renames the lock to a tombstone,
    and only while its record still carries this acquisition's token, then
    removes the tombstone. Tombstones and staging directories a crash leaves
    behind are removed by the next pull (see :func:`_sweep_lock_leftovers`).

    Raises :class:`PullInFlight` when someone else holds it.
    """
    d = _part_lock_dir(filename)
    d.parent.mkdir(parents=True, exist_ok=True)
    _sweep_lock_leftovers(d.parent)
    token = uuid.uuid4().hex
    staging = _stage_lock_record(d.parent, filename, token)
    try:
        _acquire_part_lock(d, filename, staging)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)
    try:
        yield
    finally:
        _release_part_lock(d, filename, token)


def _pull_url(
    url: str,
    name: str,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
) -> bool:
    """Download a model from a direct URL with resumable .part file support.

    Derives the destination name, then hands the whole download to
    :func:`_pull_url_locked` under a cross-process lock on that name. The
    derivation stays here so the lock is keyed on exactly the name the download
    will write, rather than on a second derivation that could disagree with it.
    """
    # url is the raw caller-supplied spec (CLI/GUI/MCP) and filename/e are
    # derived from it, so all are escaped: _safe_models_filename rejects path
    # traversal and Windows-reserved characters, not '['/']', so a URL like
    # "https://host/x][red]FAKE[/red]y.gguf" survives with brackets intact.
    from rich.markup import escape
    stem = _stem_from_url(url)
    if not stem:
        console.print(
            f"[red]Invalid URL - no file name in the path:[/red] {escape(url)}\n"
            "A direct URL must point at a file, e.g. https://host/model.gguf"
        )
        return False

    filename = stem + ".gguf"

    # Traversal guard: the filename is derived from an untrusted URL path
    # segment, so it is confined to MODELS_DIR before being used as a dest.
    # `filename` is escaped: it names a value that just failed this check.
    safe = _safe_models_filename(filename)
    if safe is None:
        console.print(
            f"[red]Unsafe model filename derived from URL:[/red] "
            f"{escape(filename)}\n"
            "The download destination must be a single name inside the models "
            "folder (no '/', '\\', or '..')."
        )
        return False
    filename = safe

    try:
        with _part_lock(filename):
            return _pull_url_locked(url, name, filename, expected_sha256,
                                    redownload, model_type)
    except PullInFlight as e:
        console.print(f"[red]Download already in progress:[/red] {escape(str(e))}")
        return False


def _url_part_identity(url: str, expected_sha256: Optional[str],
                       total: int) -> dict:
    """The record a direct-URL ``.part`` is resumed against: a SHA256 of *url*
    as the caller gave it (the record never holds the URL itself), the expected
    SHA256 lower-cased, and the size the size HEAD reported (None when it
    reported none)."""
    return {"source": "url",
            "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "sha256": expected_sha256.lower() if expected_sha256 else None,
            "size": total or None}


def _pull_url_locked(
    url: str,
    name: str,
    filename: str,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
) -> bool:
    """The body of :func:`_pull_url`, run holding the lock on *filename*.

    Split out so the lock can wrap every exit path through a single ``with``
    instead of a ``try/finally`` wrapped around the whole download.

    The download goes to ``<filename>.part``; ``<filename>.part.json`` beside it
    records which URL, digest and size that partial holds (see
    :func:`_url_part_identity`). A partial is appended to only when its record
    matches this pull, and is truncated otherwise. A transfer that stops early
    keeps both for the next pull; the record is removed once the partial is
    moved into place or discarded.
    """
    import requests
    from localm.netpolicy import NetworkPolicyError
    # url is the raw caller-supplied spec, filename/dl_url/e are derived from
    # it or remote-sourced, and expected_sha256 is the raw --sha256 value with
    # no format validation upstream; all are escaped. `url` sits directly
    # inside OPEN [bold cyan]/[red] tags at several sites below, a
    # markup-TAG-INJECTION position on a fully caller-controlled value.
    from rich.markup import escape

    dest      = _mm.MODELS_DIR / filename
    part_file = _mm.MODELS_DIR / (filename + ".part")

    if dest.exists():
        # A file with this derived name is already here. It only counts as the
        # requested model when the caller's --sha256, if given, matches its
        # bytes, so a new name is never aliased onto unrelated existing bytes.
        if expected_sha256:
            on_disk = _verify_digest(dest, purpose="to check the file already here")
            if on_disk.lower() != expected_sha256.lower():
                console.print(
                    f"[red]SHA256 mismatch![/red] A different file already "
                    f"occupies {escape(filename)} ({escape(on_disk[:16])}…); "
                    f"it does not match --sha256 "
                    f"({escape(expected_sha256.lower()[:16])}…). Refusing to "
                    "alias onto unrelated bytes - use --redownload or a "
                    "different -n name."
                )
                return False
            console.print(
                f"[yellow]Already downloaded:[/yellow] {escape(filename)} "
                f"[dim](sha256 verified)[/dim]"
            )
        else:
            console.print(f"[yellow]Already downloaded:[/yellow] {escape(filename)}")
        unusable = _mm.gguf_unusable_reason(dest)
        if unusable is not None:
            console.print(f"[red]Not registered:[/red] {escape(filename)}\n{escape(unusable)}")
            return False
        _mm._register_with_dedup(name, dest, url, model_type=model_type)
        return True

    if _refuse_name_held_by_external_file(name, dest):
        return False

    # Pre-download check by user-supplied hash (URL servers can't tell us one)
    if expected_sha256 and not redownload:
        dups = _mm.find_by_sha256(expected_sha256)
        if dups:
            action = _mm._prompt_predownload_dup(dups, name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], name)
                return True

    _mm.ensure_dirs()

    # Resolve the redirect chain with each hop validated, then use the final
    # CHECKED URL for both the size HEAD and the streaming GET with redirects
    # OFF, so no unchecked hop can bounce the download into an internal host.
    from localm import netpolicy
    try:
        dl_url = _ssrf_resolve_final_url(url)
    except NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {escape(str(e))}")
        return False

    # HEAD the final URL for the total file size the disk-space check needs.
    # Pinned to the validated IP like the GET below. A connect error here is
    # non-fatal; the GET still fails closed.
    try:
        head  = netpolicy.pinned_request("HEAD", dl_url, allow_redirects=False, timeout=10)
        total = int(head.headers.get("content-length", 0))
    except NetworkPolicyError as e:
        # A policy refusal is not the benign case: it is surfaced and fails
        # closed, like the GET below, rather than collapsing into total=0.
        console.print(f"[red]Refused by network policy:[/red] {escape(str(e))}")
        return False
    except Exception as e:
        # Benign: a HEAD connect error only costs the disk-space pre-check, and
        # the GET still fails closed. Logged at debug.
        logger.debug("size HEAD failed for %s (non-fatal, size unknown): %s", dl_url, e)
        total = 0

    # Bytes already on disk from an earlier pull of this same URL.
    identity = _url_part_identity(url, expected_sha256, total)
    already_have = (part_file.stat().st_size
                    if _part_is_resumable(part_file, identity) else 0)

    remaining = max(0, total - already_have)
    if not _mm._check_disk_space(_mm.MODELS_DIR, remaining):
        return False

    # Build request - try to resume from where we left off
    headers: dict = {}
    if already_have:
        headers["Range"] = f"bytes={already_have}-"
        # TAG-INJECTION site: url is fully caller-controlled (the raw spec
        # string) and sits directly inside an OPEN [bold cyan]...[/bold cyan]
        # tag.
        console.print(
            f"Resuming [bold cyan]{escape(url)}[/bold cyan] "
            f"[dim](skipping first {format_size(already_have)})[/dim]"
        )
    else:
        console.print(f"Downloading [bold cyan]{escape(url)}[/bold cyan]")

    try:
        # revalidate immediately before the connect
        netpolicy.check_url(dl_url, allow_when_off=netpolicy.downloads_allowed_when_off())
        r = netpolicy.pinned_request("GET", dl_url, headers=headers, stream=True,
                                     timeout=30, allow_redirects=False)
        if already_have and r.status_code == 416:
            already_have = 0
            headers.pop("Range", None)
            _discard_part(part_file)
            r = netpolicy.pinned_request("GET", dl_url, headers=headers, stream=True,
                                         timeout=30, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            console.print(f"[red]Refused:[/red] unexpected redirect from "
                          f"{escape(dl_url)}")
            return False
        r.raise_for_status()
    except NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {escape(str(e))}")
        return False
    except requests.HTTPError as e:
        # code is always an int (a real HTTP status_code) or the literal "?"
        # fallback - never markup.
        code = getattr(e.response, "status_code", "?")
        console.print(f"[red]Download failed[/red] (HTTP {code}): {escape(url)}")
        return False
    except requests.RequestException as e:
        console.print(f"[red]Could not reach[/red] {escape(url)}: {escape(str(e))}")
        return False

    # Server may ignore the Range header - detect and reset if needed
    if already_have and r.status_code == 200:
        # Server returned the full file despite Range request
        already_have = 0

    content_length = int(r.headers.get("content-length", 0))
    # No content-length means the size is unknown, and it stays unknown even
    # when bytes are already on disk: on a resumed chunked download, adding
    # already_have to a zero content-length would make the total equal to
    # already_have and report a stuck 100% for the whole transfer.
    total_display = (already_have + content_length) if content_length else None

    # Opened, and truncated on a restart, before either progress reporter below
    # reads the partial's size. See
    # test_a_restart_over_another_urls_partial_reports_progress_from_zero.
    f = open(part_file, "ab") if already_have else _start_part(part_file, identity)

    stream_error: list = []

    def _write_chunks(on_chunk=None):
        with f:
            try:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
                    if on_chunk is not None:
                        on_chunk(len(chunk))
            except requests.RequestException as e:
                stream_error.append(e)

    def _part_bytes() -> int:
        try:
            return part_file.stat().st_size
        except OSError:
            return 0

    if os.environ.get("LOCALM_PROGRESS_JSON") == "1":
        # GUI mode: stream JSON progress polled from the .part file on disk,
        # the same mechanism the HuggingFace path uses (_download_progress).
        # No Rich bar here: there is no terminal, and its ANSI would clutter the
        # captured stdout the GUI parses.
        with _snapshot_progress(_part_bytes, total_display or 0, label=filename) as _prog:
            _write_chunks()
            if not stream_error:
                _prog.ok()
    else:
        with _record_stop(_part_bytes, total_display or 0), Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as prog:
            from rich.markup import escape
            task = prog.add_task(escape(filename), total=total_display,
                                 completed=already_have)
            _write_chunks(lambda n: prog.update(task, advance=n))

    if stream_error:
        console.print(
            f"[red]Download interrupted:[/red] {escape(str(stream_error[0]))}\n"
            "The partial download is kept; pull the same URL again to continue.")
        return False

    # Atomically rename on successful completion
    part_file.rename(dest)
    _unlink_quiet(_part_record_path(part_file))

    # SHA256 verification. `actual` is a hashlib hexdigest; expected_sha256 is
    # the raw --sha256 value, with no charset validation upstream. Both are
    # escaped.
    actual = _verify_digest(dest)
    if expected_sha256:
        if actual.lower() == expected_sha256.lower():
            _report_success(f"[green]✓[/green] SHA256 verified: {escape(actual[:16])}…",
                            f"[green]OK[/green] SHA256 verified: {escape(actual[:16])}…")
        else:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected "
                f"{escape(expected_sha256[:16])}…, got {escape(actual[:16])}… "
                "- deleting corrupted file"
            )
            dest.unlink()
            return False
    else:
        console.print(f"[dim]SHA256: {escape(actual)}[/dim]")

    # Post-download identity check: is this a byte-identical copy of something
    # already registered? A URL download cannot know the hash up front, so this
    # is the earliest detection point.
    dups = [n for n in _mm.find_by_sha256(actual) if n != name]
    if dups and not redownload:
        # dups are registry keys (_sanitize_name-derived) and are escaped
        # anyway; they sit in a TAG-INJECTION position.
        names = ", ".join(f"'{escape(n)}'" for n in dups)
        console.print(
            f"[yellow]Downloaded file is byte-identical to {names}[/yellow]"
        )
        if sys.stdin.isatty():
            import click
            choice = click.prompt(
                f"  [a]lias as '{name}' and delete the duplicate file  "
                "[k]eep both copies",
                type=click.Choice(["a", "k"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "a":
                # Route through _entry_path like every other registry consumer:
                # a raw ``["path"]`` read raises TypeError on a null/int path and
                # skips _entry_path's ``..`` rejection, and this value decides an
                # unlink().
                epath = _mm._entry_path(_mm.load_registry().get(dups[0]))
                if epath is None:
                    # An unreadable sibling entry is reported rather than
                    # silently kept, and never licenses deleting a file. Quoted
                    # by hand rather than via !r.
                    console.print(
                        f"[yellow]Registry entry for '{escape(dups[0])}' is "
                        "malformed - keeping both copies rather than deleting "
                        "a file on the strength of an unreadable path.[/yellow]"
                    )
                else:
                    existing_path = Path(epath)
                    if dest.resolve() != existing_path.resolve():
                        dest.unlink()
                    alias_model(dups[0], name)
                    return True

    unusable = _mm.gguf_unusable_reason(dest)
    if unusable is not None:
        console.print(f"[red]Not registered:[/red] {escape(filename)}\n{escape(unusable)}")
        return False
    _mm._register(name, dest, url, sha256=actual, model_type=model_type)
    # name is always _sanitize_name()-derived by pull_model() before reaching
    # here, and is escaped anyway; it sits in a TAG-INJECTION position.
    _report_success(f"[green]✓[/green] [bold]{escape(name)}[/bold] is ready",
                    f"[green]OK[/green] [bold]{escape(name)}[/bold] is ready")
    return True


def _part_record_path(part_file: Path) -> Path:
    """The record of which file *part_file* holds: ``<name>.part.json`` beside
    it."""
    return part_file.with_name(part_file.name + ".json")


def _part_is_resumable(part_file: Path, identity: dict) -> bool:
    """True when *part_file* exists and the record beside it equals
    *identity*. A missing, unreadable or malformed record never matches."""
    if not part_file.is_file():
        return False
    try:
        rec = json.loads(_part_record_path(part_file).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    return rec == identity


def _discard_part(part_file: Path) -> None:
    """Remove the record beside *part_file*, then *part_file* itself."""
    _unlink_quiet(_part_record_path(part_file))
    _unlink_quiet(part_file)


def _start_part(part_file: Path, identity: dict):
    """Open *part_file* empty for writing and record *identity* beside it.

    The earlier record is removed before the file is truncated and the new one
    is written after it (see
    test_a_restart_that_cannot_truncate_leaves_no_record_beside_the_old_bytes).
    Raises OSError when the earlier record cannot be removed or the file cannot
    be opened. A new record that cannot be written is logged and left absent,
    which makes the partial unresumable. Returns the open file.
    """
    rec = _part_record_path(part_file)
    rec.unlink(missing_ok=True)
    f = open(part_file, "wb")
    try:
        rec.write_text(json.dumps(identity), encoding="utf-8")
    except OSError as e:
        _unlink_quiet(rec)
        logger.warning("could not record which file %s holds (%s); if this "
                       "download is interrupted it restarts from the beginning",
                       part_file, e)
    except BaseException:
        f.close()
        raise
    return f


def _civitai_part_identity(version_id: object, resolved: Any,
                           digest: Optional[str]) -> dict:
    """The record a CivitAI ``.part`` is resumed against: the model version,
    the file CivitAI resolved, its expected SHA256 and its size."""
    return {"source": "civitai", "version_id": str(version_id),
            "file_id": resolved.file_id,
            "sha256": digest.lower() if digest else None,
            "size": resolved.size_bytes}


def _pull_civitai_file_locked(
    version_id: str,
    resolved: Any,
    dest: Path,
    *,
    verify_digest: Optional[str],
    register: bool,
    model_name: str,
    reg_type: str,
) -> bool:
    """The transfer half of :func:`_pull_civitai_file`, run holding the part
    lock on ``dest.name``.

    The download goes to ``<dest>.part``; ``<dest>.part.json`` beside it
    records which file that partial holds (see :func:`_civitai_part_identity`).
    A partial is appended to only when its record matches this pull, and is
    truncated otherwise. The finished partial is verified against
    *verify_digest* before it replaces *dest*; one that fails is deleted with
    its record and *dest* is left as it was. A transfer that stops early keeps
    both for the next pull.
    """
    import requests
    from rich.markup import escape

    from localm import netpolicy

    filename = dest.name
    dest_dir = dest.parent
    part_file = dest_dir / (filename + ".part")
    identity = _civitai_part_identity(version_id, resolved, verify_digest)
    # Read under the part lock. See
    # test_a_partial_that_grew_before_the_lock_resumes_from_its_new_end.
    already_have = (part_file.stat().st_size
                    if _part_is_resumable(part_file, identity) else 0)

    remaining = max(0, (resolved.size_bytes or 0) - already_have)
    if not _mm._check_disk_space(dest_dir, remaining):
        return False

    try:
        dl_url = _ssrf_resolve_final_url(resolved.url)
    except netpolicy.NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {escape(str(e))}")
        return False

    headers: dict = {}
    if already_have:
        headers["Range"] = f"bytes={already_have}-"
        console.print(
            f"Resuming [bold cyan]civitai:{escape(str(version_id))}[/bold cyan] / "
            f"[bold]{escape(filename)}[/bold] "
            f"[dim](skipping first {format_size(already_have)})[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]civitai:{escape(str(version_id))}[/bold cyan] / "
                      f"[bold]{escape(filename)}[/bold]")

    try:
        netpolicy.check_url(dl_url, allow_when_off=netpolicy.downloads_allowed_when_off())
        r = netpolicy.pinned_request("GET", dl_url, headers=headers, stream=True,
                                     timeout=30, allow_redirects=False)
        if already_have and r.status_code == 416:
            already_have = 0
            headers.pop("Range", None)
            _discard_part(part_file)
            r = netpolicy.pinned_request("GET", dl_url, headers=headers, stream=True,
                                         timeout=30, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            console.print(f"[red]Refused:[/red] unexpected redirect from "
                          f"{escape(dl_url)}")
            return False
        r.raise_for_status()
    except netpolicy.NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {escape(str(e))}")
        return False
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        console.print(f"[red]Download failed[/red] (HTTP {code}): "
                      f"{escape(dl_url)}")
        return False
    except requests.RequestException as e:
        console.print(f"[red]Could not reach[/red] {escape(dl_url)}: "
                      f"{escape(str(e))}")
        return False

    if already_have and r.status_code == 200:
        already_have = 0

    content_length = int(r.headers.get("content-length", 0))
    total_display = (already_have + content_length) if content_length else resolved.size_bytes or 0

    def _part_bytes() -> int:
        try:
            return part_file.stat().st_size
        except OSError:
            return 0

    try:
        f = open(part_file, "ab") if already_have else _start_part(part_file, identity)
    except OSError as e:
        console.print(f"[red]Download failed:[/red] {escape(str(e))}")
        return False
    with _snapshot_progress(_part_bytes, total_display) as _prog:
        try:
            with f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {escape(str(e))}")
            return False
        _prog.ok()

    actual = _verify_digest(part_file)
    if verify_digest and actual.lower() != verify_digest.lower():
        console.print(
            f"[red]SHA256 mismatch![/red] Expected {escape(verify_digest[:16])}…, "
            f"got {escape(actual[:16])}… - deleting corrupted file"
        )
        _discard_part(part_file)
        return False

    try:
        os.replace(part_file, dest)
    except OSError as e:
        console.print(f"[red]Could not move the download into place:[/red] "
                      f"{escape(str(e))}")
        return False
    # A record left with no partial beside it is never resumed. See
    # test_a_record_with_no_partial_beside_it_is_ignored.
    _unlink_quiet(_part_record_path(part_file))

    if verify_digest:
        _report_success(f"[green]✓[/green] SHA256 verified: {escape(actual[:16])}…",
                        f"[green]OK[/green] SHA256 verified: {escape(actual[:16])}…")
    else:
        console.print(f"[dim]SHA256: {escape(actual)}[/dim]")

    if register:
        registered = _mm._register_with_dedup(
            model_name, dest, resolved.source_tag, digest=actual,
            model_type=reg_type, refresh=True)
        if not registered:
            console.print(
                f"[yellow]{escape(filename)} was downloaded to {escape(str(dest))}, "
                f"but could not be registered as '{escape(model_name)}'[/yellow] "
                "(see message above) - the file is on disk. Retry with a "
                "different -n name, or 'localm alias' it in.")
            return False
    _report_success(
        f"[green]✓[/green] [bold]{escape(model_name)}[/bold] downloaded to "
        f"{escape(str(dest))}",
        f"[green]OK[/green] [bold]{escape(model_name)}[/bold] downloaded to "
        f"{escape(str(dest))}")
    return True


def _civitai_owners(dest: Path) -> "list[tuple[str, str]]":
    """``(name, source)`` of every registry entry whose path is *dest*."""
    reg = _mm.load_registry()
    return [(n, str(reg[n].get("source") or ""))
            for n in find_aliases_by_path(dest, reg)]


def _name_held_by_external_file(model_name: str, dest: Path) -> "str | None":
    """The path *model_name* is registered to when that is an existing file
    outside the models folder other than *dest*, else None. A registration
    whose file is gone, or that lives in the models folder, is not held."""
    reg = _mm.load_registry()
    entry = reg.get(model_name)
    if not isinstance(entry, dict):
        return None
    held = _mm._entry_path(entry)
    if not held:
        return None
    if model_name in _mm.find_aliases_by_path(dest, reg):
        return None
    try:
        held_path = Path(held)
        if not held_path.exists() or _mm.is_owned_model_path(held_path):
            return None
    except OSError:
        return None
    return str(held)


def _refuse_name_held_by_external_file(model_name: str, dest: Path) -> bool:
    """Print why the pull stops and return True when *model_name* is registered
    to an existing file outside the models folder other than *dest*
    (``_name_held_by_external_file``)."""
    from rich.markup import escape
    held_at = _name_held_by_external_file(model_name, dest)
    if held_at is None:
        return False
    console.print(
        f"[red]Name already in use:[/red] {escape(model_name)} is registered "
        f"to a file outside the models folder ({escape(held_at)}), which "
        f"this pull would replace in the registry. Pull with another name "
        f"([bold]--name[/bold]), or remove the registration first "
        f"([bold]localm rm {escape(model_name)}[/bold], which keeps that file)."
    )
    return True


def _hf_file_name_taken_by(dest: Path, repo_id: str) -> "list[tuple[str, str]]":
    """``(name, source)`` of every registry entry whose path is *dest* and whose
    source is not ``hf:<repo_id>``."""
    return [(n, s) for n, s in _civitai_owners(dest) if s != f"hf:{repo_id}"]


def _print_digest_name_in_use(part: str, on_disk: str, published: str,
                              repo_id: str, taken: "list[tuple[str, str]]") -> None:
    """Print why a pull stops: the file *part* in the models folder hashes to
    *on_disk*, not to the *published* digest of *repo_id*, and the registry
    entries *taken* (``(name, source)``) own it."""
    from rich.markup import escape
    console.print(
        f"[red]File name already in use:[/red] {escape(part)} in "
        f"the models folder is a different file "
        f"({escape(on_disk[:16])}…) from the one {escape(repo_id)} "
        f"publishes ({escape(published[:16])}…). It is registered "
        f"as {_civitai_registered_as(taken)} and was left as it is.\n"
        "A file name holds one file, and --name only sets the "
        "registry name. To keep both, pull a file of this "
        "repository that has another name. To use this one "
        "instead, remove the other first, which deletes its file "
        f"({' and '.join('localm rm ' + escape(n) for n, _ in taken)}), "
        "then pull again."
    )


def _undigested_file_held_by_other_pull(
        dest: Path, repo_id: str, revision: "Optional[HfRevision]",
        model_name: str) -> "tuple[list[tuple[str, str]], bool]":
    """``(blocking, own)`` for a file at *dest* that Hugging Face publishes no
    digest for, so its bytes cannot be told from the pull's own.

    Of the registry entries whose path is *dest*, those not pulled from
    *repo_id* at *revision* hold a different pull's bytes (a pull on the default
    branch, *revision* None, counts only an entry that records a revision, and
    an unregistered file counts as nobody's). *blocking* lists ``(name, source)``
    of those the pull must not replace: every one but an entry named
    *model_name* of the same repo, whose own record the pull rewrites. *own* is
    True when the file is held by such an entry alone and so is fetched again.
    The source carries ``@revision`` when the entry records one."""
    reg = _mm.load_registry()
    wanted = revision.ref if revision else None
    blocking: list[tuple[str, str]] = []
    own = False
    for name in find_aliases_by_path(dest, reg):
        source = str(reg[name].get("source") or "")
        recorded = reg[name].get("revision") or None
        if source == f"hf:{repo_id}" and recorded == wanted:
            continue
        if wanted is None and recorded is None:
            continue
        if name == model_name and source == f"hf:{repo_id}":
            own = True
        else:
            blocking.append((name, source + (f"@{recorded}" if recorded else "")))
    return blocking, own and not blocking


def _civitai_registry_unreadable() -> bool:
    """True when registry.json exists but neither it nor its backup can be read,
    which ``load_registry`` reports as an empty registry."""
    from .. import config
    return not config._read_json_checked(config.REGISTRY_FILE, {})[1]


_CIVITAI_REGISTRY_STEP = "Fix or remove registry.json, then pull again."


def _civitai_registered_as(owners: "list[tuple[str, str]]") -> str:
    """*owners* as ``'name' (source)``, escaped for console markup."""
    from rich.markup import escape
    return ", ".join(f"'{escape(n)}' ({escape(s or 'no source recorded')})"
                     for n, s in owners)


def _civitai_next_step(others: "list[tuple[str, str]]") -> str:
    """How to replace a file that no registry entry points at (empty *others*)
    or that the entries *others*, recorded from another source, point at."""
    from rich.markup import escape
    if not others:
        return "Pull again with --redownload to replace it."
    removals = " and ".join(f"localm rm {escape(n)}" for n, _ in others)
    return (f"Remove it from the registry first ({removals}), then pull again "
            "with --redownload to replace the file.")


def _civitai_unproven_reason(dest: Path, resolved: Any,
                             owners: "list[tuple[str, str]]") -> Optional[str]:
    """Why the file at *dest* cannot be taken as *resolved*'s when there is no
    SHA256 to check it against, or None when the registry records it as that
    version and, if CivitAI lists a size, its size equals it."""
    from rich.markup import escape
    if _civitai_registry_unreadable():
        return ("localm's model registry (registry.json) could not be read, so "
                "it cannot tell which model owns this file")
    if not any(source == resolved.source_tag for _, source in owners):
        if owners:
            return (f"it is registered as {_civitai_registered_as(owners)}, "
                    f"not as {escape(resolved.source_tag)}")
        return ("it is not in the model registry, so nothing shows which "
                "version it belongs to")
    if resolved.size_bytes:
        try:
            on_disk = dest.stat().st_size
        except OSError as e:
            return f"its size could not be read ({escape(str(e))})"
        if on_disk != resolved.size_bytes:
            return (f"it is {on_disk} bytes and CivitAI lists "
                    f"{resolved.size_bytes}")
    return None


def _civitai_refuse_replacing(dest: Path, resolved: Any) -> bool:
    """True, after saying why, when *dest* exists and a registry entry recorded
    from another source than *resolved*'s points at it, so that replacing it
    would change what that entry loads."""
    from rich.markup import escape
    if not dest.exists():
        return False
    if _civitai_registry_unreadable():
        console.print(
            f"[red]Refusing to replace {escape(str(dest))}:[/red] localm's model "
            "registry (registry.json) could not be read, so it cannot tell "
            "which model owns this file. Nothing was changed.")
        console.print(f"[dim]{_CIVITAI_REGISTRY_STEP}[/dim]")
        return True
    others = [o for o in _civitai_owners(dest) if o[1] != resolved.source_tag]
    if not others:
        return False
    console.print(
        f"[red]Refusing to replace {escape(str(dest))}:[/red] it is registered "
        f"as {_civitai_registered_as(others)}, and {escape(resolved.source_tag)} "
        "would change what that entry loads. Nothing was changed.")
    console.print(f"[dim]{_civitai_next_step(others)}[/dim]")
    return True


def _pull_civitai_file(
    version_id: str,
    name: Optional[str],
    file_id: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    register: bool = True,
    model_type: str = "auto",
    include_legacy_formats: bool = False,
) -> bool:
    """Download one file from a CivitAI model VERSION id into the
    ComfyUI-subfoldered tree for whichever ComfyUI instance is currently
    active (managed or external - see media.managed_comfy.comfy_models_dest_dir),
    and register it into localm's own registry.json eagerly.

    *file_id*, when given, selects one file among the version's files (see
    sources.CivitAISource.resolve_download); otherwise the primary file (or
    the first remaining after the safe-format filter) is used. *model_type*
    "auto" takes the registry type CivitAI's own model type maps to
    (sources.CIVITAI_TYPE_MAP); an explicit value overrides only the registry
    label, never the ComfyUI subfolder the file is physically routed to,
    which always follows CivitAI's own type.

    The download redirect (a time-boxed pre-signed URL) is resolved and
    fetched through the same per-hop SSRF-guarded path _pull_url uses
    (_ssrf_resolve_final_url), never a hardcoded-trusted-host shortcut.

    A file already at the destination is used as this version's only when its
    SHA256 matches (CivitAI's, else *expected_sha256*) or, with no SHA256 to
    compare, when the registry records it as this version and its size equals
    the size CivitAI lists, when it lists one; any other file is left as it is and the pull is
    refused. *redownload* replaces the file, but is refused while a registry
    entry recorded from another source points at it. Both checks are repeated
    after the part lock on the file's name is taken; the transfer itself runs
    in :func:`_pull_civitai_file_locked` under that lock.
    """
    from rich.markup import escape

    from . import sources as _sources
    from localm.media.managed_comfy import comfy_models_dest_dir

    try:
        resolved = _sources.CivitAISource().resolve_download(
            version_id, file_id, include_legacy_formats=include_legacy_formats)
    except _sources.ModelSourceError as e:
        console.print(f"[red]CivitAI:[/red] {escape(str(e))}")
        return False

    reg_type = resolved.model_type if model_type == "auto" else model_type

    dest_dir = comfy_models_dest_dir(resolved.comfy_subfolder)
    if dest_dir is None:
        console.print(
            "[red]No ComfyUI folder is configured to download into.[/red] Set "
            "a ComfyUI workdir (localm config comfy_workdir <path>), or enable "
            "the managed ComfyUI instance, then retry."
        )
        return False

    safe = _safe_models_filename(resolved.filename, dest_dir)
    if safe is None:
        console.print(
            f"[red]Unsafe model filename:[/red] {escape(resolved.filename)}")
        return False
    filename = safe
    dest = dest_dir / filename

    want = expected_sha256.lower() if expected_sha256 else None
    if want and resolved.sha256 and want != resolved.sha256.lower():
        console.print(
            f"[red]SHA256 mismatch (before download):[/red] --sha256 "
            f"{escape(want[:16])}… does not match CivitAI's metadata for "
            f"{escape(filename)} ({escape(resolved.sha256[:16])}…). "
            "Refusing to download."
        )
        return False
    verify_digest = resolved.sha256 or want

    model_name = _sanitize_name(name or Path(filename).stem)

    def _already_here() -> bool:
        if not verify_digest:
            owners = _civitai_owners(dest)
            reason = _civitai_unproven_reason(dest, resolved, owners)
            if reason:
                others = [o for o in owners if o[1] != resolved.source_tag]
                console.print(
                    f"[red]Refusing to use the file already at "
                    f"{escape(str(dest))} as {escape(resolved.source_tag)}:"
                    f"[/red] {reason}. CivitAI lists no SHA256 to check it "
                    "against, so it was left as it is.")
                if _civitai_registry_unreadable():
                    step = _CIVITAI_REGISTRY_STEP
                else:
                    step = _civitai_next_step(others)
                    if not others:
                        step += (" Or pass --sha256 with this file's expected "
                                 "digest to check the one already here.")
                console.print(f"[dim]{step}[/dim]")
                return False
        console.print(f"[yellow]Already downloaded:[/yellow] {escape(filename)}")
        if verify_digest:
            on_disk = _verify_digest(dest, purpose="to check the file already here")
            if on_disk.lower() != verify_digest.lower():
                console.print(
                    f"[red]SHA256 mismatch![/red] The file already at "
                    f"{escape(filename)} ({escape(on_disk[:16])}…) does not "
                    f"match the expected digest ({escape(verify_digest[:16])}…)."
                )
                owners = _civitai_owners(dest)
                others = [o for o in owners if o[1] != resolved.source_tag]
                held = (f"It is registered as {_civitai_registered_as(owners)}. "
                        if owners else "")
                console.print(f"[dim]{held}{_civitai_next_step(others)}[/dim]")
                return False
        if register:
            _mm._register_with_dedup(model_name, dest, resolved.source_tag,
                                     digest=verify_digest, model_type=reg_type)
        return True

    if dest.exists() and not redownload:
        return _already_here()

    from ..config import _mkdir_or_explain
    _mkdir_or_explain(dest_dir, is_home=False)

    try:
        with _part_lock(filename):
            # Checked again under the lock. See
            # test_a_file_another_pull_finished_first_is_kept.
            if dest.exists() and not redownload:
                return _already_here()
            if redownload and _civitai_refuse_replacing(dest, resolved):
                return False
            return _pull_civitai_file_locked(
                version_id, resolved, dest, verify_digest=verify_digest,
                register=register, model_name=model_name, reg_type=reg_type)
    except PullInFlight as e:
        console.print(f"[red]Download already in progress:[/red] {escape(str(e))}")
        return False


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type from HARD metadata. The real
    implementation lives in localm.discover (shared with search-result
    classification there); lazy import matches this module's existing
    convention of not importing localm.discover at module scope."""
    from localm.discover import _hf_pipeline_tag_to_type as _classify
    return _classify(repo_id)

