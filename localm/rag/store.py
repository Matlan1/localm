# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Persistent document collections ("knowledge bases").

Layout - one directory per collection under ``<data dir>/rag/``:

    rag/<name>/meta.json      {"name", "created",
                               "docs": {path: {mtime, size, chunks}},
                               "roots": {folder: {"added": ts}}}
    rag/<name>/chunks.jsonl   one chunk per line: {"source", "pos", "text"}
    rag/<name>/vectors.json   optional: {"dim", "vectors": [[...]|null, ...]}
                              aligned with chunks.jsonl line order

``roots`` records the FOLDERS that were indexed, alongside the per-file ``docs``
entries the folder walk produced. ``resync()`` re-walks these roots through the
ordinary incremental path.

Collections are explicit user data (like generated images): indexing writes
to disk in every session mode. Rewrites are whole-file + atomic rename.

Retrieval is hybrid: BM25 always; when vectors exist for (almost) all chunks
and the caller can embed the query, scores become an equal blend of
max-normalised BM25 and cosine similarity.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import stat as _stat
import time
from pathlib import Path
from typing import Callable, Optional

from localm.debuglog import logger as _log
from localm.jsonl import dumps_lines, split_jsonl
from localm.pathsafe import is_mapped_network_drive, is_unc_or_device_path

# numpy is optional: absent -> None, and every caller degrades to pure Python.
# Bound ONCE at module import rather than per call.
try:
    import numpy as _numpy
except ImportError:      # optional dependency - every caller degrades to pure Python
    _numpy = None

#: True when numpy imported but is a namespace stub / attribute-less object
#: rather than a real install (``__file__`` is None for a PEP 420 namespace
#: package).
_NUMPY_IS_STUB = _numpy is not None and getattr(_numpy, "__file__", None) is None


def _warn_numpy_degrade(exc: Exception, operation: str) -> None:
    """Announce the pure-Python fallback ONCE per process, saying which case it is.

    Three states, three branches:

    1. numpy ABSENT      - the default install has no numpy. Debug level, no
                           warning.
    2. numpy is a STUB   - the install is BROKEN. Something put a bare 'numpy'
                           directory on sys.path. Warns, naming the artefact so
                           it can be deleted.
    3. numpy present but
       otherwise UNUSABLE - warns as unexpected.
    """
    if _NUMPY_DEGRADE_LOGGED:
        return
    _NUMPY_DEGRADE_LOGGED.add(True)
    if _numpy is None:
        # Absent: logged at debug, never as a warning. Branches on the MODULE
        # STATE, never on the exception's text.
        _log.debug("numpy is not installed; using the pure-Python %s (%s: %s).",
                   operation, type(exc).__name__, exc)
        return
    if _NUMPY_IS_STUB:
        _log.warning(
            "numpy imported as an EMPTY NAMESPACE PACKAGE from %s - it is not a real "
            "install, and this will break anything else here that imports numpy. Most "
            "likely a bare 'numpy' directory left on sys.path by a failed or "
            "partially-removed install; find and remove it. Falling back to "
            "pure-Python %s (%s: %s).",
            getattr(_numpy, "__path__", None) or "an unknown path",
            operation, type(exc).__name__, exc)
    else:
        _log.warning(
            "numpy is present but unusable (%s: %s); falling back to pure-Python %s. "
            "Results are identical, it is slower on large collections.",
            type(exc).__name__, exc, operation)
from localm.storekit import NamespaceLockRegistry, atomic_write as _storekit_atomic_write
from .bm25 import BM25, ENGLISH_STOP_WORDS
from .collection_lock import (CollectionLockedError, collection_write_lock,
                              lock_path_for, wait_budget)
from .chunk import chunk_text
from .extract import (BLACKLISTED_SUFFIXES, SECRET_SUFFIXES,
                      UNINDEXABLE_SUFFIXES, ExtractError, classify_format,
                      extract_bytes, extract_text, is_secret_index_name)

ClassifyFn = Callable[[str], Optional[str]]
DescribeImageFn = Callable[[bytes, str], Optional[str]]

# Printable, 1-64 characters, no control chars, no path separators, no
# Windows-reserved punctuation, and no ".". See
# test_dot_rejected_would_collide_with_lock_sibling.
_NAME_RE = re.compile(r'\A[^\x00-\x1f\x7f./\\:*?"<>|]{1,64}\Z')

# Windows reserved device names: they match _NAME_RE but mkdir raises on them.
_RESERVED_NAMES = {"con", "prn", "aux", "nul",
                   *(f"com{i}" for i in range(1, 10)),
                   *(f"lpt{i}" for i in range(1, 10))}


def _first_dim(vectors: list) -> Optional[int]:
    """Dimensionality of the first non-empty vector, or None."""
    for v in vectors:
        if v:
            return len(v)
    return None


def _well_formed_vectors(vectors) -> bool:
    """Cheap (O(n)) structural check that *vectors* is what ``_save`` writes: a
    list whose entries are each a null placeholder (a missing embedding) or a
    list/tuple."""
    return isinstance(vectors, list) and all(
        (not v) or isinstance(v, (list, tuple)) for v in vectors)


def _vectors_finite(vectors) -> bool:
    """True when every component of every present vector is a FINITE number.

    Structure is already validated by ``_well_formed_vectors``; this checks the
    values."""
    try:
        np = _numpy
        if np is None:
            raise ImportError("numpy is not installed")
        for v in vectors:
            if not v:
                continue
            try:
                arr = np.asarray(v, dtype="float64")
            except (ValueError, TypeError):
                return False                       # non-numeric component
            if not np.isfinite(arr).all():
                return False
        return True
    # AttributeError as well as ImportError: an attribute-less numpy raises
    # AttributeError rather than failing to import.
    except (ImportError, AttributeError) as e:
        _warn_numpy_degrade(e, "vector validation")
        for v in vectors:
            if not v:
                continue
            for x in v:
                try:
                    if not math.isfinite(x):
                        return False
                except TypeError:
                    return False
        return True

# Directories never worth indexing when a folder is added
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
              ".pytest_cache", ".mypy_cache", "dist", "build", ".idea",
              ".vscode"}

EmbedFn = Callable[[list[str]], list[list[float]]]
# Called with a human-readable message as the sole positional argument. A call
# site with an exact numerator/denominator additionally passes phase/done/total/
# unit as keywords; any sink such a site can reach must accept and ignore them
# (**_).
ProgressFn = Callable[..., None]


def rag_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "rag"


# Third-party credential/secret folders that are never indexed, even when they
# sit inside an allowed root. Does NOT include ".localm".
_SENSITIVE_HOME_SUBDIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure",
)
# Lower-cased, and matched against a path component ANYWHERE in a resolved
# path, so a nested ~/proj/.ssh and a ".SSH" component both match.
_SENSITIVE_NAMES = frozenset(s.lower() for s in _SENSITIVE_HOME_SUBDIRS)


def _path_within(child: Path, parent: Path) -> bool:
    """True when *child* is *parent* or lives underneath it (both resolved)."""
    try:
        child, parent = child.resolve(), parent.resolve()
    except (OSError, ValueError):
        return False
    if child == parent:
        return True
    if hasattr(child, "is_relative_to"):
        return child.is_relative_to(parent)
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# Cap the folder-walk recursion depth, bounding a pathological directory cycle.
_MAX_WALK_DEPTH = 50


def _walk_files(root: Path, *, max_depth: int = _MAX_WALK_DEPTH):
    """Yield files under *root* without following linked DIRECTORIES, bounded by
    depth and a visited-realpath set.

    Never descends into a linked directory (junction OR bind-mount OR symlink)
    and refuses to revisit a resolved directory, so no directory cycle can hang
    indexing. ``_SKIP_DIRS`` are pruned during descent.

    A linked FILE **is** yielded. Confinement (a link escaping an allowed root)
    is enforced by ``_expand``'s confine loop, not here."""
    reparse_flag = getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    seen: set = set()
    stack: list = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            real = os.path.realpath(d)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                attrs = getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0)
                if e.is_symlink() or (attrs & reparse_flag):
                    # Branch on the RESOLVED type: a linked DIRECTORY is not
                    # followed, a linked FILE is yielded.
                    try:
                        if e.is_dir(follow_symlinks=True):
                            _log.debug("rag: not following linked directory during "
                                       "index walk: %s", e.path)
                            continue
                        if e.is_file(follow_symlinks=True):
                            yield Path(e.path)
                            continue
                        # Neither: a dangling or unresolvable link.
                        _log.debug("rag: skipping unresolvable link during index "
                                   "walk: %s", e.path)
                    except OSError as exc:
                        _log.debug("rag: could not resolve link during index walk: "
                                   "%s (%s)", e.path, exc)
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in _SKIP_DIRS:
                        continue                   # prune .git/node_modules/etc.
                    stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False):
                    yield Path(e.path)
            except OSError:
                continue


class ConfinementError(ValueError):
    """A path may not be indexed. ``reason`` distinguishes a fixable whitelist
    miss (``outside_allowed``) from a hard refusal (``credential`` /
    ``secret_file`` / ``denied`` / ``invalid`` / ``unc_or_device``). Subclasses
    ``ValueError``."""

    def __init__(self, message: str, *, path: Path, reason: str):
        super().__init__(message)
        self.path = path
        self.reason = reason


_INDEX_MODES = ("whitelist", "blacklist")


def indexing_policy(cfg: Optional[dict] = None,
                    key_roots: Optional[list] = None) -> dict:
    """The current RAG indexing confinement policy, read from config.

    ``mode`` is ``whitelist`` (index only your home folder, the working directory,
    and the ``rag_allowed_roots`` you added) or ``blacklist`` (index anywhere
    EXCEPT the ``rag_denied_roots`` you listed). In BOTH modes credential folders
    and UNC/device paths are still refused - a hard floor that
    ``confine_index_path`` enforces separately and no mode can turn off. The
    localm data directory is not part of that floor. Returns resolved ``Path``
    lists.

    *key_roots* is an optional PER-KEY folder allowlist (``auth.rag_roots_for`` /
    ``http_server.effective_rag_roots`` - empty/None for the owner or a key that
    never had one set). When non-empty it OVERRIDES the config-driven policy
    entirely: the returned policy is forced to ``whitelist`` with ``allowed`` set
    to exactly the resolved *key_roots* and a ``key_scoped`` flag set, so
    ``confine_index_path`` does NOT also imply the home directory, the working
    directory, or the global ``rag_allowed_roots`` on top of it. The hard floor
    (credential folders, secret files, UNC/device paths) still applies underneath
    this exactly as it does for the global policy; only the whitelist SET changes.
    """
    # Loaded once, before the key_roots branch, so a key-scoped caller also
    # reads the owner's allow_network_drives setting.
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception as e:
            # A config we cannot load falls back to an EMPTY policy, which
            # confine_index_path treats as whitelist-with-no-extra-roots, and the
            # failure is logged.
            from localm.debuglog import logger as _dbg
            _dbg.debug("rag indexing_policy: could not load config, using an empty "
                       "fail-closed policy: %s", e)
            cfg = {}
    if key_roots:
        resolved: list[Path] = []
        for r in key_roots:
            try:
                resolved.append(Path(r).expanduser().resolve())
            except (OSError, ValueError):
                continue
        return {"mode": "whitelist", "allowed": resolved, "denied": [],
                "key_scoped": True,
                "allow_network_drives": bool(cfg.get("allow_network_drives", True))}
    mode = cfg.get("rag_indexing_mode", "whitelist")
    if mode not in _INDEX_MODES:
        mode = "whitelist"

    def _resolve(key: str) -> list[Path]:
        out: list[Path] = []
        for r in cfg.get(key, []) or []:
            try:
                out.append(Path(r).expanduser().resolve())
            except (OSError, ValueError) as e:
                # A configured root we cannot resolve is DROPPED and logged; a
                # dropped DENIED root warns that a path inside it may now be
                # indexable.
                from localm.debuglog import logger as _dbg
                if key == "rag_denied_roots":
                    _dbg.warning(
                        "rag: denied root %r could not be resolved and is NOT being "
                        "enforced - a path inside it may now be indexable: %s", r, e)
                else:
                    _dbg.warning(
                        "rag: configured %s entry %r could not be resolved and is "
                        "being ignored: %s", key, r, e)
                continue
        return out

    return {"mode": mode,
            "allowed": _resolve("rag_allowed_roots"),
            "denied": _resolve("rag_denied_roots"),
            # confine_index_path applies this regardless of mode.
            "allow_network_drives": bool(cfg.get("allow_network_drives", True))}


def _network_drives_allowed_fresh() -> bool:
    """One-off config read for confine_index_path's ``policy=None`` callers
    (settings_schema.py's PATHLIST save-time validation, and the bare CLI),
    which have no ``indexing_policy()`` dict to read the value off. A config
    that cannot be loaded resolves to the True default."""
    try:
        from localm.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    return bool(cfg.get("allow_network_drives", True))


def confine_index_path(p, policy: Optional[dict] = None) -> Path:
    """Resolve *p* and verify it may be indexed, raising ``ConfinementError`` (a
    ``ValueError``) otherwise.

    The HARD FLOOR is enforced ALWAYS, even when *policy* is None: well-known
    credential folders (``.ssh``, ``.aws``, ...) are never indexable - wherever
    they appear in the resolved path, so a nested ``~/proj/.ssh`` or a symlink
    into one is caught too. A UNC/device path is refused unconditionally too
    (see the ``is_unc_or_device_path`` check below). A mapped Windows network
    drive (``Z:\\...``) is refused the same way, ALSO unconditionally by
    caller kind, but only when the ``allow_network_drives`` config setting is
    off (default on).

    The localm data directory (LOCALM_HOME) is NOT refused, at all.

    With a *policy* (the HTTP API passes ``indexing_policy()``):
      - ``whitelist``: *p* must be within your home folder, the working directory,
        or a ``rag_allowed_roots`` entry, else ``reason='outside_allowed'`` - this
        applies to LOCALM_HOME exactly like any other folder outside the
        defaults, not as a special case;
      - ``blacklist``: *p* is allowed unless it is within a ``rag_denied_roots``
        entry, then ``reason='denied'``;
      - a KEY-SCOPED policy (``indexing_policy(key_roots=...)``, marked
        ``policy["key_scoped"]``) replaces the whitelist SET entirely: *p* must
        be within one of the key's own explicit roots, and the home
        directory/working directory/global ``rag_allowed_roots`` are NOT also
        allowed on top of it.

    ``policy=None`` means hard-floor only; the caller is otherwise unconfined.
    """
    try:
        rp = Path(p).expanduser()
    except (OSError, ValueError):
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid")
    # Refuse UNC/device syntax unconditionally, BEFORE the .resolve() below ever
    # runs, and on the EXPANDED string. Raised OUTSIDE the try/except above.
    if is_unc_or_device_path(str(rp)):
        raise ConfinementError(f"Refusing to index a UNC or device path: {p}",
                               path=Path(str(p)), reason="unc_or_device")
    try:
        rp = rp.resolve()
    except (OSError, ValueError):
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid")

    # Credential folders are denied wherever they appear in the resolved path,
    # not only at the home root. rp is already resolved, so a symlink pointing
    # into a credential dir is caught too.
    if any(part.lower() in _SENSITIVE_NAMES for part in rp.parts):
        raise ConfinementError(f"Refusing to index a credential directory: {p}",
                               path=rp, reason="credential")

    # Checked unconditionally, BEFORE the policy=None return below. Read off
    # *policy* when one is given, else read fresh here; .get(), not [], so a
    # hand-built policy dict without the key defaults to True.
    if policy is not None:
        allow_net = bool(policy.get("allow_network_drives", True))
    else:
        allow_net = _network_drives_allowed_fresh()
    if not allow_net and is_mapped_network_drive(str(rp)):
        raise ConfinementError(f"Refusing to index a network drive: {p}",
                               path=rp, reason="network_drive_denied")

    if policy is None:
        return rp

    # --- API floor: refuse model-weight / binary / credential FILES (policy set) ---
    # The same suffix + secret-name filter _expand applies to a folder walk,
    # applied to explicit picks whenever a policy is present, and BEFORE the mode
    # branches. Guarded on "not a directory" rather than is_file(): a directory
    # merely NAMED like a secret stays walkable, and a path that does not exist
    # is refused here too. The CLI (policy=None, returned above) stays unconfined.
    if not rp.is_dir() and (rp.suffix.lower() in SECRET_SUFFIXES
                            or is_secret_index_name(rp.name)):
        raise ConfinementError(
            f"Refusing to index {rp.name}: key/credential material is not "
            f"indexed through the API. Use the local CLI (`localm rag add`) if "
            f"you really intend to.",
            path=rp, reason="secret_file")
    # A non-secret binary/media file (UNINDEXABLE_SUFFIXES: .mp4, .db, .7z, model
    # weights, ...) does NOT raise here. _add_paths_locked reports it as an
    # individual per-file failure instead, still BEFORE reading the bytes.

    if policy.get("mode") == "blacklist":
        # Allow anything not explicitly denied (the hard floor above still holds).
        # Path(d) coerces a policy hand-built with str entries.
        for d in policy.get("denied", []):
            if _path_within(rp, Path(d)):
                raise ConfinementError(
                    f"This folder is on your denied list, so it is not indexed: {p}",
                    path=rp, reason="denied")
        return rp

    # whitelist: home and the working dir are always allowed, plus the roots the
    # owner added. A KEY-SCOPED policy (indexing_policy(key_roots=...)) does NOT
    # imply home/cwd/the global rag_allowed_roots: only the key's own explicit
    # roots count. The hard floor above runs ahead of this branch either way.
    if policy.get("key_scoped"):
        roots: list[Path] = []
        for r in policy.get("allowed", []):
            try:
                roots.append(Path(r).resolve())
            except (OSError, ValueError):
                continue
    else:
        roots = []
        for r in [Path.home(), Path.cwd(), *policy.get("allowed", [])]:
            try:
                roots.append(Path(r).resolve())   # coerce str entries, then resolve
            except (OSError, ValueError):
                continue
    if any(_path_within(rp, r) for r in roots):
        return rp
    raise ConfinementError(
        f"This folder is outside the folders localm may index. Add it to your "
        f"allowed folders in Settings to index it: {p}",
        path=rp, reason="outside_allowed")


def check_collection_name(name: str) -> str:
    """Validate a collection name, returning it, or raise ``ValueError``."""
    name = name or ""
    if not _NAME_RE.match(name):
        raise ValueError(
            'Collection names must be 1-64 characters and cannot contain '
            '. / \\ : * ? " < > | or control characters')
    if name != name.strip():
        raise ValueError("Collection names cannot start or end with whitespace")
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(f"'{name}' is a reserved device name and cannot be used")
    return name


# Internal alias for the in-module call sites.
_check_name = check_collection_name


def collection_names(base: Optional[Path] = None) -> list[str]:
    base = base or rag_dir()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / "meta.json").is_file())


def delete_collection(name: str, base: Optional[Path] = None,
                      on_wait: Optional[Callable[[str], None]] = None) -> bool:
    """Delete a collection, waiting for any in-flight write to finish first.

    Deleting takes the same locks a write does. Raises ``CollectionLockedError``
    if another process's run does not finish in time, rather than deleting
    underneath it."""
    import shutil
    base = base or rag_dir()
    path = base / _check_name(name)
    if not (path / "meta.json").is_file():
        return False
    # Bounds the in-process half too: refuses after the same budget instead of
    # queueing behind a re-sync.
    budget = wait_budget()
    local = _collection_lock(name)
    if not local.acquire(timeout=budget):
        raise CollectionLockedError(name, None, budget, same_process=True)
    try:
        with collection_write_lock(lock_path_for(path), collection=name,
                                   op="a delete", on_wait=on_wait):
            if not (path / "meta.json").is_file():
                return False      # someone else deleted it while we waited
            shutil.rmtree(path)
    finally:
        local.release()
    return True


# Per-collection-NAME locks: concurrent writes to one collection serialise
# process-wide across separate Collection instances. Keyed by name, so the map
# is bounded by the number of collections. RLock, so a locked method may call
# another.
#
# PER PROCESS: it does not reach a CLI invocation. That half is covered by
# collection_lock.collection_write_lock, a lock FILE beside the collection
# directory, which is always held INSIDE this lock.
_COLLECTION_LOCKS = NamespaceLockRegistry()


def _collection_lock(name: str):
    # Keyed case-INSENSITIVELY: Collection("Docs") and Collection("docs") are
    # two names but the same directory and the same lock file on Windows and
    # macOS, so folding them means two threads meet here rather than at the
    # lock file.
    return _COLLECTION_LOCKS.get(name.casefold())


# Where a vectors.json that _load() REFUSED is set aside when the chunks it was
# (mis)aligned with get rewritten. Preserved, never deleted.
_REJECTED_VECTORS = "vectors.json.rejected"

#: How many set-aside vector indexes to KEEP per collection. Older ones are
#: deleted with a warning.
_MAX_REJECTED_KEPT = 3

#: Set once when numpy has been found present-but-unusable; the pure-Python
#: cosine fallback then announces itself exactly once per process.
_NUMPY_DEGRADE_LOGGED: set = set()

#: Warn-once keys already logged in THIS process: vector degrades
#: (_note_vector_degrade) and the chunks.jsonl malformed-line warning in _load().
#: Never consulted for state, only for whether to LOG. Every key starts with the
#: collection dir plus a distinguishing tag (a literal string, or the degrade
#: text).
_WARNED_DEGRADES: set = set()

#: meta.json key for the derived-stats cache _save() writes and peek_stats() /
#: peek_detail() read. Internal/derived, never user data; a meta.json without it
#: is simply "no cache yet".
_STATS_CACHE_KEY = "_stats_cache"


class Collection:
    def __init__(self, name: str, base: Optional[Path] = None) -> None:
        self.name = _check_name(name)
        self.dir = (base or rag_dir()) / self.name
        self._meta: dict = {}
        self._chunks: list[dict] = []
        self._vectors: Optional[list] = None     # aligned with _chunks, or None
        self._vec_dim: Optional[int] = None       # dimensionality of stored vectors
        self._bm25: Optional[BM25] = None
        self.corrupt: bool = False
        # How many lines of chunks.jsonl _load() had to skip as unparseable or
        # wrong-shape; 0 whenever the file is clean or absent. Exposed via
        # stats().
        self.chunks_bad_lines: int = 0
        # Why semantic (vector) scoring is unavailable when it should be present.
        # None = vectors are used, or legitimately absent (no embeddings indexed).
        # A non-None string means a corrupt/stale/mismatched vectors index was
        # detected and scoring fell back to BM25 lexical. Exposed via stats() and
        # logged once.
        self.vector_degrade_reason: Optional[str] = None
        # True when _load() found a vectors.json on disk and REFUSED to use it;
        # _save() then sets that file aside instead of deleting it. Distinct from
        # vector_degrade_reason, which is also set by query-time degrades (a
        # failed query embedding, partial coverage).
        self._vectors_file_rejected: bool = False
        if self.exists():
            self._load()

    # ------------------------------------------------------------- #
    #  Lifecycle / IO                                                #
    # ------------------------------------------------------------- #

    def exists(self) -> bool:
        return (self.dir / "meta.json").is_file()

    def _write_lock(self, op: str, on_progress: Optional[ProgressFn] = None):
        """The CROSS-PROCESS write lock for this collection.

        Every read-modify-write entry point takes it INSIDE the per-process
        ``_collection_lock``, never the other way round, so at most one thread
        of this process is ever at the lock file.

        The two halves have DIFFERENT waiting rules. Writers inside one process
        QUEUE for as long as it takes. A writer in ANOTHER process is bounded
        and ends in a refusal. ``delete_collection`` is the one caller that
        bounds both (see its docstring).

        The wait is reported through the caller's existing progress channel.
        """
        return collection_write_lock(
            lock_path_for(self.dir), collection=self.name, op=op,
            on_wait=on_progress)

    def create(self) -> "Collection":
        """Create the collection if it does not exist yet.

        Takes the write lock and re-checks existence inside it. The fast path
        (already exists) takes no lock at all."""
        if self.exists():
            return self
        with _collection_lock(self.name), self._write_lock("a create"):
            if self.exists():
                return self       # somebody else created it while we waited
            self.dir.mkdir(parents=True, exist_ok=True)
            self._meta = {"name": self.name, "created": time.time(), "docs": {}}
            self._save()
        return self

    def _load(self) -> None:
        # A corrupt meta.json is flagged, not fatal, and does not discard the
        # INDEPENDENT chunks.jsonl / vectors.json files. Execution falls through
        # to load the chunks and then rebuild a minimal docs map from their
        # sources.
        self.corrupt = False
        self.chunks_bad_lines = 0
        meta_corrupt = False
        try:
            meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("meta.json is not an object")
            self._meta = meta
        except (json.JSONDecodeError, ValueError, OSError):
            self.corrupt = True
            meta_corrupt = True
            self._meta = {"name": self.name, "docs": {}}
        self._chunks = []
        chunks_file = self.dir / "chunks.jsonl"
        if chunks_file.is_file():
            bad_lines = 0
            # split_jsonl, NOT str.splitlines(): JSONL is delimited by LINE FEED
            # and nothing else.
            for line in split_jsonl(chunks_file.read_text(encoding="utf-8")):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                # A chunk MUST be a dict carrying a str "text". A
                # valid-JSON-but-wrong-shape line (a scalar or array, or a dict
                # missing "text") is skipped and counted as corruption instead.
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    bad_lines += 1
                    continue
                self._chunks.append(obj)
            self.chunks_bad_lines = bad_lines
            if bad_lines:
                self.corrupt = True
                # Warn-once, using the same process-scoped set as
                # _note_vector_degrade below; self.corrupt is still set
                # unconditionally above. Keyed on the bad_lines COUNT as well as
                # the dir, so a fault that changes shape warns again.
                key = ("chunks_malformed", str(self.dir), bad_lines)
                if key not in _WARNED_DEGRADES:
                    _WARNED_DEGRADES.add(key)
                    _log.warning("RAG collection %r: skipped %d malformed line(s) in "
                                 "chunks.jsonl; run 'localm rag repair'",
                                 self.name, bad_lines)
        self._vectors = None
        self._vec_dim = None
        self.vector_degrade_reason = None
        self._vectors_file_rejected = False
        vec_file = self.dir / "vectors.json"
        if vec_file.is_file():
            # vectors.json PRESENT but unusable is handled on its own path, never
            # collapsed into "simply absent".
            try:
                data = json.loads(vec_file.read_text(encoding="utf-8"))
                vectors = data.get("vectors", [])
            except (json.JSONDecodeError, OSError) as e:
                data, vectors = None, None
                self._note_vector_degrade(
                    f"vectors.json is unreadable ({type(e).__name__}); "
                    f"using BM25 lexical retrieval only", warn=True)
            if vectors is not None:
                if not _well_formed_vectors(vectors):
                    # Valid JSON but the entries are not vectors (scalars or
                    # strings from a hand-edit or truncation): treated as corrupt.
                    self._note_vector_degrade(
                        "vectors.json is malformed (entries are not vectors); "
                        "using BM25 lexical retrieval only", warn=True)
                elif len(vectors) == len(self._chunks):
                    if not _vectors_finite(vectors):
                        # Structurally a vector list, but a component is NaN/inf
                        # or non-numeric.
                        self._note_vector_degrade(
                            "vectors.json has non-finite (NaN/inf) or non-numeric "
                            "values; using BM25 lexical retrieval only", warn=True)
                    else:
                        self._vectors = vectors
                        self._vec_dim = data.get("dim") or _first_dim(vectors)
                elif vectors:
                    # A non-empty vectors list that does not line up with the
                    # chunks. FEWER vectors than chunks is a partial embed; MORE
                    # means orphaned entries from a prior, larger chunk set.
                    kind = ("a partial embed" if len(vectors) < len(self._chunks)
                            else "orphaned entries from a prior, larger index")
                    self._note_vector_degrade(
                        f"vectors.json has {len(vectors)} vectors for "
                        f"{len(self._chunks)} chunks ({kind}); "
                        f"using BM25 lexical retrieval only", warn=True)
            # Any reason recorded in this block means the file IS there and was
            # refused. Remembered for _save().
            self._vectors_file_rejected = self.vector_degrade_reason is not None
        # A sidecar an earlier write set aside (_quarantine_rejected_vectors)
        # keeps semantic search degraded until the index is REBUILT, so the
        # reason is restated here. Checked independently of vectors.json, and
        # gated on the current index being COMPLETE rather than on that file
        # merely existing.
        if (self.vector_degrade_reason is None       # keep a more specific reason
                and not self._vector_index_complete()
                and self._rejected_vector_files()):
            self._note_vector_degrade(
                f"an earlier vector index was unusable and was set aside as "
                f"{_REJECTED_VECTORS} (nothing was deleted), and the current one "
                f"does not cover every chunk; using BM25 lexical retrieval only - "
                f"rebuild it with 'localm rag repair <name> --embed'",
                warn=True)
        self._bm25 = None
        # If meta.json was corrupt but chunks survived, rebuild a minimal docs
        # map from the chunk sources. The rebuilt entries lack mtime/size/hash,
        # so a later add/repair re-reads the file. Gated on META corruption: a
        # valid meta whose chunks.jsonl merely had a bad line keeps its real
        # docs map.
        if meta_corrupt and self._chunks:
            rebuilt: dict = {}
            for c in self._chunks:
                src = c.get("source")
                if not src:
                    continue
                entry = rebuilt.setdefault(src, {"chunks": 0})
                entry["chunks"] += 1
                if str(src).startswith("upload:"):
                    entry["uploaded"] = True
            self._meta["docs"] = rebuilt

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))
        # dumps_lines escapes the line-break-alikes json.dumps(ensure_ascii=False)
        # would otherwise emit raw (U+0085/U+2028/U+2029), so a record cannot be
        # split in half by a line-oriented reader.
        self._atomic_write("chunks.jsonl", dumps_lines(self._chunks))
        # meta.json and chunks.jsonl were just rewritten from this instance's own
        # in-memory state, which _load() only ever fills with well-formed
        # records, so both corruption flags are cleared here as well as in
        # _load().
        self.corrupt = False
        self.chunks_bad_lines = 0
        # The fate of a REJECTED vectors.json is decided FIRST, before anything
        # below writes or unlinks that filename.
        if self._vectors_file_rejected:
            if self._chunks:
                self._quarantine_rejected_vectors()
            self._vectors_file_rejected = False
        # "Complete" means every chunk has a usable vector. Partial coverage
        # does not clear a set-aside sidecar's degrade.
        complete = self._vector_index_complete()
        if self._vectors is not None and any(v for v in self._vectors):
            self._vec_dim = _first_dim(self._vectors)
            self._atomic_write("vectors.json", json.dumps(
                {"dim": self._vec_dim, "vectors": self._vectors}))
        else:
            # Nothing usable to write. A REJECTED file was already moved out of
            # the way above.
            (self.dir / "vectors.json").unlink(missing_ok=True)
            self._vec_dim = None
        if not self._chunks:
            # Every document is gone. Stored vectors are positional against
            # chunks, so nothing is left to realign a set-aside sidecar to.
            self._discard_rejected_vectors("the collection no longer has any "
                                           "documents to realign them to")
            self.vector_degrade_reason = None
        elif complete:
            # Every chunk has a vector, so the degrade clears. The sidecar file
            # itself is KEPT.
            self.vector_degrade_reason = None
        elif self._rejected_vector_files():
            self.vector_degrade_reason = (
                f"an earlier vector index was unusable and was set aside as "
                f"{_REJECTED_VECTORS} (nothing was deleted), and the current one "
                f"does not cover every chunk; using BM25 lexical retrieval only - "
                f"rebuild it with 'localm rag repair <name> --embed'")
        else:
            self.vector_degrade_reason = None
        self._bm25 = None
        # Cache the LISTING-relevant fields that are NOT otherwise persisted -
        # vector_degrade_reason and the vector-coverage math above - so a listing
        # can answer from meta.json alone, without reconstructing this Collection
        # at all (see peek_stats() / peek_detail() below).
        #
        # The cache carries a cheap (mtime_ns, size) fingerprint of chunks.jsonl
        # and vectors.json, taken AFTER they were written: peek_stats() /
        # peek_detail() stat (never read) both files again and refuse the cache
        # the instant either no longer matches, and fall back to a full load
        # whenever the cache is missing entirely.
        #
        # A second, small atomic write: the meta.json write at the top of this
        # method happens BEFORE vector_degrade_reason is finalised above.
        self._meta[_STATS_CACHE_KEY] = self._stats_cache_block()
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))

    def _stats_cache_block(self) -> dict:
        """The ``_stats_cache`` block for meta.json, computed from THIS
        instance's current in-memory state and a FRESH fingerprint of
        chunks.jsonl/vectors.json taken right now (see ``_file_fingerprint``).

        Caller must hold this collection's write lock; without that, the
        fingerprint could describe files a concurrent writer is mid-way
        through replacing."""
        return {
            "n_chunks": len(self._chunks),
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "vector_degrade_reason": self.vector_degrade_reason,
            "corrupt": self.corrupt,
            "chunks_bad_lines": self.chunks_bad_lines,
            "fingerprint": self._file_fingerprint(),
            # Lets a listing-time caller compare against the currently active
            # embedding model WITHOUT loading anything.
            "vector_dim": self._vec_dim,
        }

    def _vector_index_complete(self) -> bool:
        """True when every chunk currently has a usable vector.

        Stricter than "a vectors.json exists": partial coverage is not complete,
        and neither is an empty chunk list."""
        return bool(self._chunks) and (
            self._vectors is not None
            and len(self._vectors) == len(self._chunks)
            and all(v for v in self._vectors))

    def _rejected_vector_files(self) -> list:
        """Every set-aside vectors sidecar, oldest name first."""
        try:
            return sorted(p for p in self.dir.glob(_REJECTED_VECTORS + "*")
                          if p.is_file())
        except OSError:
            return []

    def _discard_rejected_vectors(self, why: str) -> None:
        """Delete set-aside sidecars, saying why.

        The ONLY place they are ever removed. Announced at warning level."""
        for p in self._rejected_vector_files():
            try:
                p.unlink()
            except OSError as e:
                _log.warning("RAG collection %r: could not remove %s (%s); it is "
                             "left in place", self.name, p.name, e)
                continue
            _log.warning("RAG collection %r: removed the set-aside vector index "
                         "%s - %s.", self.name, p.name, why)

    def _quarantine_rejected_vectors(self) -> None:
        """Set a rejected vectors.json aside as ``vectors.json.rejected``.

        The bytes stay on disk for recovery, and no loader will ever pair them
        with chunks again. ``_load`` reports the set-aside file as a degrade for
        as long as it exists."""
        src = self.dir / "vectors.json"
        if not src.is_file():
            return
        dest = self._free_rejected_name()
        if dest is None:
            _log.warning(
                "RAG collection %r: an unusable vectors.json could not be set "
                "aside because %s and its numbered siblings all exist; it is "
                "left in place so nothing is overwritten. Rebuild the index "
                "('localm rag repair %s --embed') or clear the old .rejected "
                "files by hand.", self.name, _REJECTED_VECTORS, self.name)
            return
        try:
            os.replace(src, dest)
        except OSError as e:
            # Best-effort; the failure is logged and the file is left in place.
            _log.warning("RAG collection %r: could not set the unusable "
                         "vectors.json aside as %s (%s); it is left in place",
                         self.name, dest.name, e)
            return
        _log.warning("RAG collection %r: the unusable vectors.json was set aside "
                     "as %s (%s). Nothing was deleted; re-embed with "
                     "'localm rag reembed %s' (no source files needed) or rebuild "
                     "from source with 'localm rag repair %s --embed'.",
                     self.name, dest.name,
                     self.vector_degrade_reason or "unusable",
                     self.name, self.name)
        self._prune_rejected_vectors()

    def _prune_rejected_vectors(self) -> None:
        """Keep only the newest ``_MAX_REJECTED_KEPT`` set-aside indexes.

        Each set-aside file is a full copy of the vector index.

        Ordered by MTIME, not by name: ``_rejected_vector_files`` sorts
        lexicographically, which puts ``.rejected.20`` before ``.rejected.3``.
        Deletion is announced at WARNING level."""
        files = self._rejected_vector_files()
        if len(files) <= _MAX_REJECTED_KEPT:
            return
        try:
            by_age = sorted(files, key=lambda p: p.stat().st_mtime)
        except OSError as e:
            _log.warning("RAG collection %r: could not order the set-aside vector "
                         "indexes to prune them (%s); all are left in place",
                         self.name, e)
            return
        for p in by_age[:-_MAX_REJECTED_KEPT]:
            try:
                size = p.stat().st_size
                p.unlink()
            except OSError as e:
                _log.warning("RAG collection %r: could not prune %s (%s); it is "
                             "left in place", self.name, p.name, e)
                continue
            _log.warning("RAG collection %r: pruned the oldest set-aside vector "
                         "index %s (%.1f MB) - keeping the newest %d. The current "
                         "index is unaffected.",
                         self.name, p.name, size / 1048576.0, _MAX_REJECTED_KEPT)

    def _free_rejected_name(self):
        """An unused ``vectors.json.rejected[.N]`` path, or None if there is none.

        ``os.replace`` overwrites its destination, so the name is numbered and
        every preserved copy is kept. Past the cap (20) None is returned and the
        caller leaves the file where it is."""
        first = self.dir / _REJECTED_VECTORS
        if not first.exists():
            return first
        for n in range(2, 21):
            candidate = self.dir / f"{_REJECTED_VECTORS}.{n}"
            if not candidate.exists():
                return candidate
        return None

    def _save_meta(self) -> None:
        """Persist meta.json ONLY, leaving chunks.jsonl / vectors.json alone.

        For a change that touches nothing but metadata (a newly recorded root, a
        missing/restored flag). Chunks are untouched, so the cached BM25 index
        stays valid too."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))

    def _atomic_write(self, filename: str, content: str) -> None:
        # storekit.atomic_write: unique temp name plus a Windows PermissionError
        # retry.
        _storekit_atomic_write(self.dir / filename, content)

    # ------------------------------------------------------------- #
    #  Indexing                                                      #
    # ------------------------------------------------------------- #

    @staticmethod
    def _expand(paths: list,
                policy: Optional[dict] = None) -> list[Path]:
        """Resolve files + recursive folder contents to indexable files.

        When *policy* is given, files that fail confinement (system paths,
        credential dirs, denied roots, symlinks escaping an allowed folder, or a
        model-weight / binary / credential FILE) are dropped by the confine loop
        below. With no policy (the CLI) explicit picks are unfiltered."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_file():
                out.append(p.resolve())
            elif p.is_dir():
                # _walk_files (NOT rglob): it bounds directory-link loops, branches
                # on the resolved type of a link, and prunes _SKIP_DIRS during
                # descent.
                for f in sorted(_walk_files(p)):
                    if (f.suffix.lower() not in BLACKLISTED_SUFFIXES
                            and not is_secret_index_name(f.name)
                            and not any(part in _SKIP_DIRS for part in f.parts)):
                        out.append(f.resolve())
        # de-dup, keep order
        seen: set = set()
        deduped = [p for p in out if not (p in seen or seen.add(p))]
        if policy is None:
            return deduped
        kept: list[Path] = []
        for p in deduped:
            try:
                confine_index_path(p, policy)
            except ValueError:
                continue   # nested escape (symlink / credential / denied) -> skip
            kept.append(p)
        return kept

    def _record_roots(self, paths: list) -> bool:
        """Persist the FOLDER roots among *paths*. Returns True if anything new
        was recorded.

        Only directories are recorded. An individually added FILE is already
        tracked by its own ``docs`` entry, which ``resync`` re-checks directly.

        Called from ``_add_paths_locked`` AFTER the confinement check.
        """
        roots = self._meta.setdefault("roots", {})
        if not isinstance(roots, dict):
            # An externally written meta.json could hold anything here. The bad
            # value is replaced and the collection is flagged corrupt.
            _log.warning("RAG collection %r: meta.json 'roots' was not an object "
                         "(%s); starting a fresh roots map", self.name,
                         type(roots).__name__)
            roots = {}
            self._meta["roots"] = roots
            self.corrupt = True
        changed = False
        for p in paths:
            try:
                rp = Path(p).expanduser()
                if not rp.is_dir():
                    continue
                key = str(rp.resolve())
            except (OSError, ValueError) as e:
                # Already skipped by _expand; logged rather than dropped silently.
                _log.debug("rag: could not record %s as an index root: %s", p, e)
                continue
            if key not in roots:
                roots[key] = {"added": time.time()}
                changed = True
        return changed

    def roots(self) -> list:
        """The folders indexed into this collection, resolved and sorted.

        These are what ``resync`` re-walks. Empty for a collection built only
        from individually named files or uploads, and for one whose meta.json was
        corrupt; re-add the folder to restore them."""
        return self._roots_from_meta(self._meta)

    @staticmethod
    def _roots_from_meta(meta: dict) -> list:
        roots = meta.get("roots")
        return sorted(roots) if isinstance(roots, dict) else []

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  classify_fn: Optional[ClassifyFn] = None,
                  describe_image_fn: Optional[DescribeImageFn] = None,
                  on_progress: Optional[ProgressFn] = None,
                  policy: Optional[dict] = None,
                  force: bool = False,
                  model_name: Optional[str] = None) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size+content hash) are
        skipped; changed ones are re-indexed in place. Pass ``force=True`` to
        re-index every file regardless (``localm rag add --force`` / repair).
        Returns counters plus per-file failures. embed_fn failures degrade to
        lexical-only, never abort.

        When *policy* is given (the HTTP API passes ``indexing_policy()``), an
        out-of-bounds top-level path raises ``ValueError`` and nested escapes are
        dropped. CLI callers omit it and stay unconfined. Indexing with an
        embedding model whose dimensionality differs from the collection's also
        raises ``ValueError``.

        *model_name*, like ``reembed()``'s, is the EMBEDDING model's name, only
        recorded (as ``embedding_model()``) the first time this collection is
        actually embedded - passing it when *embed_fn* is None is harmless, it is
        simply never reached.
        """
        # Serialise the whole read-modify-write per collection, re-reading the
        # latest committed state under the lock. The _load() must happen INSIDE
        # both locks.
        with _collection_lock(self.name), self._write_lock("an index", on_progress):
            self._load()
            return self._add_paths_locked(
                paths, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, policy=policy, force=force,
                model_name=model_name)

    def _add_paths_locked(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                          classify_fn: Optional[ClassifyFn] = None,
                          describe_image_fn: Optional[DescribeImageFn] = None,
                          on_progress: Optional[ProgressFn] = None,
                          policy: Optional[dict] = None,
                          force: bool = False,
                          model_name: Optional[str] = None) -> dict:
        """The add_paths read-modify-write body. MUST run under
        _collection_lock(self.name) after a fresh _load() (see add_paths)."""
        say = on_progress or (lambda _t: None)
        if policy is not None:
            # confine_index_path returns the RESOLVED path it validated, and
            # that is what the walk below starts from, rather than the caller's
            # original string.
            paths = [confine_index_path(p, policy) for p in paths]  # raises ValueError
        # Persist the FOLDER roots now that confinement has accepted them, and
        # before the expand, so an add that finds no indexable file still records
        # the folder.
        roots_changed = self._record_roots(paths)
        files = self._expand(paths, policy)
        if not files:
            if roots_changed:
                self._save_meta()   # metadata only: this add indexed nothing
            return {"added": 0, "updated": 0, "skipped": 0, "failed": [],
                    "chunks": len(self._chunks)}

        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None

        for f in files:
            key = str(f)
            # An EXPLICITLY-NAMED non-secret binary; a folder walk already filters
            # these out in _expand, so only a direct pick reaches here. Reported as
            # an ordinary per-file failure, the same shape an ExtractError below
            # produces, and BEFORE stat/read_bytes.
            if f.suffix.lower() in UNINDEXABLE_SUFFIXES:
                msg = (f"{f.name}: no extractable text (binary, media, or model "
                       f"weights)")
                failed.append({"path": key, "error": msg})
                say(f"skip {msg}")
                continue
            try:
                stat = f.stat()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            known = self._meta["docs"].get(key)
            # Content hash as well as (mtime, size), so a same-size edit whose
            # mtime is unchanged is still re-indexed. Legacy entries lacking
            # "hash" compare unequal and self-heal on the next add.
            try:
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            if not force and known \
                    and known.get("mtime") == stat.st_mtime \
                    and known.get("size") == stat.st_size \
                    and known.get("hash") == digest:
                skipped += 1
                continue
            try:
                text = extract_text(f, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {f.name}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Heuristic-first format label; the LLM tie-break is consulted only
            # for an unknown extension whose structure is unclear AND a chat model
            # loaded (classify_fn short-circuits otherwise).
            fmt = classify_format(text, f.name, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

            vectors: list = [None] * len(new_chunks)
            if not embed_broken and new_chunks:
                try:
                    vecs = embed_fn([c["text"] for c in new_chunks])
                except Exception as e:
                    embed_broken = True
                    say(f"embeddings unavailable ({e}) - indexing lexical-only")
                else:
                    if len(vecs) == len(new_chunks):
                        if not _vectors_finite(vecs):
                            # A NaN/inf component: no vectors are stored for this
                            # doc (lexical-only) and the degrade is reported.
                            say(f"embeddings had non-finite (NaN/inf) values for "
                                f"{f.name} - indexing it lexical-only")
                        else:
                            new_dim = _first_dim(vecs)
                            # A different embedding dimensionality means a
                            # different model: refused rather than stored as
                            # mixed-dim vectors.
                            if (self._vec_dim is not None and new_dim is not None
                                    and new_dim != self._vec_dim):
                                # Persist every file this call already finished
                                # before raising: the exception HALTS the batch,
                                # and add_paths()/add_uploads() otherwise _save()
                                # only once at the very end.
                                self._save()
                                raise ValueError(self._dim_mismatch_message(new_dim))
                            vectors = vecs
                            if self._vec_dim is None and new_dim is not None:
                                self._vec_dim = new_dim
                                # Record which model built this index, the same
                                # key reembed() writes.
                                if model_name:
                                    self._meta["embedding_model"] = str(model_name)

            # Replace any previous chunks (and vectors) for this document
            if known:
                keep = [i for i, c in enumerate(self._chunks)
                        if c.get("source") != key]
                self._chunks = [self._chunks[i] for i in keep]
                if self._vectors is not None:
                    self._vectors = [self._vectors[i] for i in keep]
                updated += 1
            else:
                added += 1
            if self._vectors is None:
                self._vectors = [None] * len(self._chunks)
            self._chunks.extend(new_chunks)
            self._vectors.extend(vectors)
            self._meta["docs"][key] = {
                "mtime": stat.st_mtime, "size": stat.st_size,
                "hash": digest,
                "chunks": len(new_chunks),
            }
            say(f"indexed {f.name} ({len(new_chunks)} chunks)")

        if added or updated:
            self._save()
        elif roots_changed or self.corrupt:
            # Nothing was indexed (every file skipped as unchanged, or every one
            # failed), so chunks and vectors are exactly as _load() read them.
            # Persist metadata only, and only when there is something to persist:
            # a newly recorded root, or a meta.json that _load() flagged corrupt
            # and rebuilt a docs map for. Mirrors the no-indexable-files early
            # return above.
            self._save_meta()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    # ------------------------------------------------------------- #
    #  Folder re-sync                                                #
    # ------------------------------------------------------------- #

    def resync(self, *, embed_fn: Optional[EmbedFn] = None,
               classify_fn: Optional[ClassifyFn] = None,
               describe_image_fn: Optional[DescribeImageFn] = None,
               on_progress: Optional[ProgressFn] = None,
               policy: Optional[dict] = None,
               force: bool = False,
               prune_missing: bool = False,
               model_name: Optional[str] = None) -> dict:
        """Bring the index back in line with the folders it was built from.

        Re-walks every persisted root through the ORDINARY incremental path
        (``add_paths``): a file ADDED to an indexed folder since the last run is
        picked up, a CHANGED file is re-indexed, an unchanged file is skipped by
        content hash. Individually indexed files are re-checked too. This is what
        a scheduled ``rag`` job calls; it is also ``localm rag resync``.

        DELETION SEMANTICS. A document whose file has VANISHED is FLAGGED
        (``missing: True`` + ``missing_since``), not dropped: its chunks stay
        indexed and stay searchable. The flag CLEARS by itself when the file
        comes back. Actual deletion happens only when the caller passes
        ``prune_missing=True``.

        A root that is not currently an available directory (deleted, unmounted,
        unreadable, or replaced by a file) is REPORTED and skipped whole, and
        every document underneath it is left completely untouched - not indexed,
        not flagged, not pruned. The same holds for a root the current
        ``policy`` refuses.

        *policy* is applied exactly as in ``add_paths``, including to a root that
        was legal when it was added but is outside the owner's allowed folders
        now. Callers that run unattended (the jobs runner) always pass one.

        *model_name*: see ``add_paths()`` - forwarded to the same first-embed
        recording.

        Returns the ``add_paths`` counters plus ``missing`` (newly flagged),
        ``missing_total``, ``restored``, ``pruned``, ``roots``,
        ``unavailable_roots`` and ``blocked_roots`` (each ``{root, reason}``), and
        ``vector_degrade_reason`` (why semantic search is degraded after this run,
        None when it is fine).
        """
        with _collection_lock(self.name), self._write_lock("a re-sync", on_progress):
            self._load()
            return self._resync_locked(
                embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force, prune_missing=prune_missing,
                model_name=model_name)

    def _resync_locked(self, *, embed_fn, classify_fn, describe_image_fn,
                       on_progress, policy, force, prune_missing,
                       model_name=None) -> dict:
        """The resync body. MUST run under _collection_lock after _load()."""
        say = on_progress or (lambda _t: None)
        available, unavailable, blocked = self._partition_roots(policy, say)
        # Roots we could not judge: nothing under them is indexed, flagged, or
        # pruned this run.
        skipped_roots = [Path(r["root"]) for r in (unavailable + blocked)]

        targets: list = list(available)
        targets.extend(self._resyncable_files(skipped_roots, policy, say))

        # Snapshot the flagged-missing set BEFORE indexing: re-indexing a document
        # REPLACES its docs entry wholesale (_add_paths_locked), dropping the
        # flag.
        docs_before = self._meta.get("docs", {})
        was_missing = {k for k, e in docs_before.items()
                       if isinstance(e, dict) and e.get("missing")}

        if targets:
            result = self._add_paths_locked(
                targets, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force, model_name=model_name)
        else:
            result = {"added": 0, "updated": 0, "skipped": 0, "failed": [],
                      "chunks": len(self._chunks)}

        missing, restored, pruned = self._reconcile_missing(
            skipped_roots, was_missing=was_missing,
            prune_missing=prune_missing, say=say)
        if pruned:
            self._save()            # chunks and vectors changed
        elif missing or restored:
            self._save_meta()       # only flags changed - see _save_meta

        docs = self._meta.get("docs", {})
        result.update({
            # Re-read AFTER the reconcile: pruning drops chunks, so the count
            # _add_paths_locked returned is stale by then.
            "chunks": len(self._chunks),
            "roots": self.roots(),
            "unavailable_roots": unavailable,
            "blocked_roots": blocked,
            "missing": missing,
            "restored": restored,
            "pruned": pruned,
            "missing_total": sum(
                1 for e in docs.values()
                if isinstance(e, dict) and e.get("missing")),
            # Why semantic search is degraded, AFTER this run (None when it is
            # fine).
            "vector_degrade_reason": self.vector_degrade_reason,
        })
        return result

    def _partition_roots(self, policy: Optional[dict], say: ProgressFn):
        """Split the persisted roots into (available, unavailable, blocked).

        Availability is checked FIRST and reported, never assumed. ``is_dir()``
        answers most of that; see ``_unmounted_reason`` for the case it cannot
        see."""
        available: list = []
        unavailable: list = []
        blocked: list = []
        for raw in self.roots():
            root = Path(raw)
            if not root.is_dir():
                # is_dir() is False for gone, unreadable, AND replaced-by-a-file.
                # The response is the same for all three (skip whole, touch
                # nothing); this branches only to report the right reason.
                reason = ("the indexed folder is now a file, not a directory"
                          if root.exists() else
                          "the indexed folder is not available (deleted, "
                          "unmounted, or unreadable)")
                unavailable.append({"root": raw, "reason": reason})
                say(f"skipping {raw}: {reason} - nothing under it was changed")
                continue
            reason = self._unmounted_reason(root)
            if reason:
                unavailable.append({"root": raw, "reason": reason})
                say(f"skipping {raw}: {reason} - nothing under it was changed")
                continue
            if policy is not None:
                try:
                    confine_index_path(root, policy)
                except ValueError as e:
                    blocked.append({"root": raw, "reason": str(e)})
                    say(f"skipping {raw}: {e}")
                    continue
            available.append(root)
        return available, unavailable, blocked

    def _unmounted_reason(self, root: Path) -> Optional[str]:
        """Why *root* looks like an UNMOUNTED mount point, or None if it is fine.

        ``is_dir()`` cannot see this on POSIX: unmounting leaves the mount point
        behind as an ordinary, existing, EMPTY directory.

        All three conditions are required and none is sufficient alone: *root*
        is a mount point, it is empty, and this collection holds documents
        indexed under it.
        """
        try:
            if not os.path.ismount(root):
                return None
            if next(root.iterdir(), None) is not None:
                return None
        except OSError:
            # Could not look inside at all: reported as unavailable, so nothing
            # under this root is touched.
            return ("the indexed folder could not be read (disconnected, or "
                    "permission denied)")
        if not self._has_docs_under(root):
            return None
        return ("the indexed folder is an empty mount point, so its drive or "
                "share appears to be unmounted")

    def _has_docs_under(self, root: Path) -> bool:
        """True when at least one indexed document's source lives under *root*.

        Uploads are excluded: an ``upload:`` key is not a filesystem path, so it
        can neither be under a root nor be evidence that one lost its contents."""
        return any(
            not str(key).startswith("upload:") and _path_within(Path(key), root)
            for key in self._meta.get("docs", {})
        )

    def _resyncable_files(self, skipped_roots: list, policy: Optional[dict],
                          say: ProgressFn) -> list:
        """Existing document source files that are safe to re-index this run.

        Uploads have no source file (``upload:`` keys) and are skipped. So is
        anything under a skipped root, and anything the current policy refuses;
        the last of those is filtered HERE rather than handed to
        ``_add_paths_locked``, whose top-level confinement check raises."""
        out: list = []
        for key in sorted(self._meta.get("docs", {})):
            if str(key).startswith("upload:"):
                continue
            p = Path(key)
            if any(_path_within(p, r) for r in skipped_roots):
                continue
            try:
                if not p.is_file():
                    continue        # gone: the missing pass decides what to do
            except OSError:
                continue
            if policy is not None:
                try:
                    confine_index_path(p, policy)
                except ValueError as e:
                    say(f"skipping {key}: {e}")
                    continue
            out.append(p)
        return out

    def _reconcile_missing(self, skipped_roots: list, *, was_missing: set,
                           prune_missing: bool, say: ProgressFn):
        """Flag documents whose file has vanished, clear the flag on ones that
        came back, and prune only when explicitly asked. Returns
        (newly_missing, restored, pruned) as lists of doc keys.

        *was_missing* is the flagged set as it stood BEFORE this run indexed
        anything; a document that came back CHANGED has already been re-indexed,
        which rewrites its entry and drops the flag.

        Mutates ``self._meta`` / ``self._chunks`` in place; the caller saves."""
        docs = self._meta.get("docs", {})
        newly_missing: list = []
        restored: list = []
        pruned: list = []
        for key in sorted(docs):
            entry = docs.get(key)
            if not isinstance(entry, dict) or str(key).startswith("upload:"):
                continue
            p = Path(key)
            if any(_path_within(p, r) for r in skipped_roots):
                continue        # unreachable root: no verdict, no change
            try:
                present = p.exists()
            except (OSError, ValueError):
                # Could not ask at all: treated as present, so an unanswerable
                # question is never resolved in the destructive direction.
                present = True
            if present:
                entry.pop("missing", None)
                entry.pop("missing_since", None)
                if key in was_missing:
                    restored.append(key)
                    say(f"back: {key}")
                continue
            if prune_missing:
                pruned.append(key)
            elif not entry.get("missing"):
                entry["missing"] = True
                entry["missing_since"] = time.time()
                newly_missing.append(key)
                say(f"missing: {key} (kept in the index, flagged)")
        if pruned:
            drop = set(pruned)
            keep = [i for i, c in enumerate(self._chunks)
                    if c.get("source") not in drop]
            self._chunks = [self._chunks[i] for i in keep]
            if self._vectors is not None:
                self._vectors = [self._vectors[i] for i in keep]
            for key in pruned:
                docs.pop(key, None)
                say(f"pruned: {key} (file is gone)")
        return newly_missing, restored, pruned

    def add_uploads(self, uploads: list, *, embed_fn: Optional[EmbedFn] = None,
                    classify_fn: Optional[ClassifyFn] = None,
                    describe_image_fn: Optional[DescribeImageFn] = None,
                    on_progress: Optional[ProgressFn] = None,
                    force: bool = False,
                    model_name: Optional[str] = None) -> dict:
        """Index documents UPLOADED from the caller's own device (the per-device
        path for a client that cannot browse the server disk).

        Each item is ``{"filename": str, "data": bytes}``. Extraction runs in
        memory (``extract_bytes`` - same zip-bomb / encoding guards as chat
        attachments); the resulting chunks and optional vectors ARE persisted.
        There is NO filesystem confinement here: nothing on the server disk is
        read, so the whitelist/blacklist policy does not apply. Docs are keyed
        ``upload:<filename>`` and deduped by content hash, so re-uploading an
        unchanged file is skipped. Returns the same counters as ``add_paths``.

        The uploaded BYTES are not retained (only the extracted chunks/vectors), so
        an ``upload:<name>`` doc cannot be re-read from disk: ``localm rag repair``
        simply skips these keys (Path('upload:x') is not a file) - their chunks
        persist, they just cannot be re-embedded from source.

        *model_name*: see ``add_paths()`` - the embedding model's name, recorded
        the first time this collection is actually embedded.
        """
        with _collection_lock(self.name), self._write_lock("an upload", on_progress):
            self._load()
            return self._add_uploads_locked(
                uploads, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, force=force, model_name=model_name)

    def _add_uploads_locked(self, uploads: list, *,
                            embed_fn: Optional[EmbedFn] = None,
                            classify_fn: Optional[ClassifyFn] = None,
                            describe_image_fn: Optional[DescribeImageFn] = None,
                            on_progress: Optional[ProgressFn] = None,
                            force: bool = False,
                            model_name: Optional[str] = None) -> dict:
        """The add_uploads body. MUST run under _collection_lock after _load().

        Mirrors the per-document body of _add_paths_locked (chunk -> embed ->
        replace-prior-chunks -> record meta), but sourced from in-memory bytes with
        a hash-only dedup (no fs mtime/size)."""
        # The no-op accepts **_ because _finished below passes the structured
        # keywords ProgressFn carries.
        say = on_progress or (lambda _t, **_: None)
        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None
        n_total = len(uploads)

        def _finished(n: int, text: str) -> None:
            """Report one item DONE, whatever its outcome.

            Called on every exit from the loop body, including the two that
            `continue`, so the done-SEQUENCE is exactly 1..n_total. Progress is
            about how far the LOOP got, not how many succeeded.
            """
            say(text, phase="indexing uploads", done=n, total=n_total,
                unit="files")

        for n_done, up in enumerate(uploads, start=1):
            filename = (str(up.get("filename") or "").strip() or "upload")
            data = up.get("data") or b""
            key = f"upload:{filename}"
            digest = hashlib.sha256(data).hexdigest()
            known = self._meta["docs"].get(key)
            if not force and known and known.get("hash") == digest:
                skipped += 1
                _finished(n_done, f"skip {filename} (unchanged)")
                continue
            try:
                text = extract_bytes(data, filename, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                _finished(n_done, f"skip {filename}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Same heuristic-first labeling as add_paths (see there).
            fmt = classify_format(text, filename, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

            vectors: list = [None] * len(new_chunks)
            if not embed_broken and new_chunks:
                try:
                    vecs = embed_fn([c["text"] for c in new_chunks])
                except Exception as e:
                    embed_broken = True
                    say(f"embeddings unavailable ({e}) - indexing lexical-only")
                else:
                    if len(vecs) == len(new_chunks):
                        if not _vectors_finite(vecs):
                            # See add_paths: a NaN/inf component means this doc is
                            # indexed lexical-only, with the degrade reported.
                            say(f"embeddings had non-finite (NaN/inf) values for "
                                f"{filename} - indexing it lexical-only")
                        else:
                            new_dim = _first_dim(vecs)
                            if (self._vec_dim is not None and new_dim is not None
                                    and new_dim != self._vec_dim):
                                # See add_paths: persist this call's completed
                                # uploads before halting.
                                self._save()
                                raise ValueError(self._dim_mismatch_message(new_dim))
                            vectors = vecs
                            if self._vec_dim is None and new_dim is not None:
                                self._vec_dim = new_dim
                                # See add_paths: record the model that built this
                                # index, the same key reembed() writes.
                                if model_name:
                                    self._meta["embedding_model"] = str(model_name)

            if known:
                keep = [i for i, c in enumerate(self._chunks)
                        if c.get("source") != key]
                self._chunks = [self._chunks[i] for i in keep]
                if self._vectors is not None:
                    self._vectors = [self._vectors[i] for i in keep]
                updated += 1
            else:
                added += 1
            if self._vectors is None:
                self._vectors = [None] * len(self._chunks)
            self._chunks.extend(new_chunks)
            self._vectors.extend(vectors)
            self._meta["docs"][key] = {
                "size": len(data), "hash": digest,
                "chunks": len(new_chunks), "uploaded": True,
            }
            _finished(n_done, f"indexed {filename} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def _dim_mismatch_message(self, new_dim: int) -> str:
        """The refusal a user sees when they change embedding model.

        Names the collection, both models where known, and the command that
        re-embeds in place without touching the source files.
        """
        was = self.embedding_model()
        built = f" with {was}" if was else ""
        return (
            f"Embedding dimension changed ({self._vec_dim} -> {new_dim}): "
            f"collection {self.name!r} was built{built} and its stored vectors "
            f"cannot be mixed with a different model's. Re-embed it in place from "
            f"the text already stored (no source files needed, nothing deleted):\n"
            f"    localm rag reembed {self.name}\n"
            f"or use 'Re-embed' on the Knowledge page. To keep the existing index "
            f"instead, switch the embedding model back{built}.")

    def reembed(self, *, embed_fn: EmbedFn, model_name: Optional[str] = None,
                on_progress: Optional[ProgressFn] = None,
                batch: int = 32) -> dict:
        """Recompute EVERY vector from the stored chunk text, with a new model.

        The chunk text is already on disk in chunks.jsonl, so nothing needs
        re-reading, re-chunking, or even to still exist: a collection whose
        sources moved, were deleted, or arrived as uploads re-embeds exactly the
        same as one whose files are all present. ``rag repair --embed`` differs
        in that it re-indexes FROM THE ORIGINAL SOURCE FILES.

        Every vector is computed into a LOCAL list first, and ``self._vectors``
        is only replaced once the whole set is in hand and validated, so a
        failing embedder leaves the previous index exactly as it was. One full
        vector set is held in memory during the run.
        """
        with _collection_lock(self.name), self._write_lock("reembed", on_progress):
            self._load()
            if not self._chunks:
                return {"chunks": 0, "dim": None, "model": model_name,
                        "note": "collection has no chunks; nothing to re-embed"}

            texts = [c.get("text") or "" for c in self._chunks]
            total = len(texts)
            fresh: list = []
            for i in range(0, total, max(1, batch)):
                part = embed_fn(texts[i:i + max(1, batch)])
                if part is None:
                    raise RuntimeError(
                        "the embedding function returned nothing for chunks "
                        f"{i}-{i + len(texts[i:i + batch])} of {total}")
                fresh.extend(part)
                if on_progress:
                    done = min(i + batch, total)
                    on_progress(f"re-embedding {done}/{total}", phase="re-embedding",
                                done=done, total=total, unit="chunks")

            # Validate BEFORE touching the live index: a short or ragged result is
            # refused here, leaving the previous index untouched.
            if len(fresh) != total:
                raise RuntimeError(
                    f"embedder returned {len(fresh)} vectors for {total} chunks; "
                    "the previous index has been left untouched")
            dims = {len(v) for v in fresh if v is not None}
            if len(dims) != 1 or not dims or next(iter(dims)) <= 0:
                raise RuntimeError(
                    f"embedder returned inconsistent vector sizes {sorted(dims)}; "
                    "the previous index has been left untouched")

            self._vectors = fresh
            self._vec_dim = next(iter(dims))
            # Record the model NAME as well as its dimension, so a later mismatch
            # can say which model built this index.
            if model_name:
                self._meta["embedding_model"] = str(model_name)
            self._meta["embedding_dim"] = self._vec_dim
            # A rebuilt full-coverage index clears the degrade state and makes any
            # set-aside sidecar moot.
            self._vectors_file_rejected = False
            self.vector_degrade_reason = None
            self.corrupt = False
            self._save()
            self._discard_rejected_vectors(
                f"re-embedded to {self._vec_dim} dimensions"
                + (f" with {model_name}" if model_name else ""))
            return {"chunks": total, "dim": self._vec_dim, "model": model_name}

    def embedding_model(self) -> Optional[str]:
        """The model NAME this collection's vectors were built with, if recorded."""
        v = self._meta.get("embedding_model")
        return str(v) if v else None

    def documents(self) -> list:
        """The source paths currently indexed in this collection (for repair)."""
        return list(self._meta.get("docs", {}).keys())

    def remove_doc(self, source: str) -> bool:
        # Same per-collection lock and re-load as add_paths, and the same
        # cross-process lock.
        with _collection_lock(self.name), self._write_lock("a document removal"):
            self._load()
            if source not in self._meta.get("docs", {}):
                return False
            keep = [i for i, c in enumerate(self._chunks)
                    if c.get("source") != source]
            self._chunks = [self._chunks[i] for i in keep]
            if self._vectors is not None:
                self._vectors = [self._vectors[i] for i in keep]
            del self._meta["docs"][source]
            self._save()
            return True

    # ------------------------------------------------------------- #
    #  Retrieval                                                     #
    # ------------------------------------------------------------- #

    def query(self, text: str, k: int = 4,
              embed_fn: Optional[EmbedFn] = None) -> list[dict]:
        """Top-*k* chunks for *text*: max-normalised BM25, blended 50/50 with
        cosine similarity when vectors cover the corpus and the query can be
        embedded."""
        if not text.strip() or not self._chunks:
            return []
        if self._bm25 is None:
            # Filter English stopwords from the lexical index, so a query and a
            # chunk that overlap ONLY on a stopword cannot win the BM25 half.
            self._bm25 = BM25([c["text"] for c in self._chunks],
                              stop_words=ENGLISH_STOP_WORDS)
        scores = self._bm25.scores(text)
        top = max(scores) if scores else 0.0
        if top > 0:
            scores = [s / top for s in scores]

        vec_scores = self._vector_scores(text, embed_fn)
        if vec_scores is not None:
            scores = [0.5 * lex + 0.5 * vec
                      for lex, vec in zip(scores, vec_scores)]

        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)[:max(1, k)]
        return [
            {**self._chunks[i], "score": round(scores[i], 4)}
            for i in order if scores[i] > 0
        ]

    def _note_vector_degrade(self, reason: str, *, warn: bool) -> None:
        """Record WHY semantic (vector) scoring is unavailable and surface it once.

        A corrupt, stale, or dimensionally mismatched vectors index does not
        silently vanish into BM25-only: it is recorded (exposed via
        ``stats()``) and genuine corruption is logged.

        "Once" means once per PROCESS, not once per instance: ``_load`` runs
        from ``__init__`` and every request builds a FRESH Collection, so an
        instance-scoped guard would log the same sentence on every /api/rag
        call. The instance field is still set unconditionally, so ``stats()``
        and the GUI's "needs repair" state are unaffected; only the duplicate
        LOG LINE is suppressed. A genuinely NEW reason for the same collection
        still warns."""
        self.vector_degrade_reason = reason
        if not warn:
            return
        key = (str(self.dir), reason)
        if key in _WARNED_DEGRADES:
            return
        _WARNED_DEGRADES.add(key)
        _log.warning("RAG collection %r: %s", self.name, reason)

    def _vector_scores(self, text: str,
                       embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        if embed_fn is None or self._vectors is None:
            return None
        present = [v for v in self._vectors if v]
        if not self._chunks or len(present) / len(self._chunks) < 0.8:
            # Partial coverage while a collection is still embedding: recorded
            # (visible in stats) but not warned about.
            self._note_vector_degrade(
                "vector coverage below 80% (index not fully embedded); "
                "using BM25 lexical retrieval only", warn=False)
            return None
        # Stored vectors must share one dimensionality. A legacy collection with
        # mixed-dim vectors is ambiguous, so vector scoring is skipped and the
        # answer is lexical.
        dims = {len(v) for v in present}
        if len(dims) > 1:
            self._note_vector_degrade(
                "stored vectors have mixed dimensionality (legacy index); "
                "using BM25 lexical retrieval only", warn=True)
            return None
        stored_dim = next(iter(dims))
        try:
            qvec = embed_fn([text])[0]
        except Exception as e:
            # The embedder raised: a real failure (backend down, model unloaded),
            # not an expected transient like partial coverage. The note is
            # idempotent, so this warns once per distinct error, not per query.
            self._note_vector_degrade(
                f"query embedding failed ({type(e).__name__}); "
                f"using BM25 lexical retrieval only", warn=True)
            return None
        # A query vector of a different dimensionality than the stored ones (a
        # switched embedding model) falls back to lexical-only.
        if len(qvec) != stored_dim:
            self._note_vector_degrade(
                f"embedding model changed (query dim {len(qvec)} != stored "
                f"{stored_dim}); using BM25 lexical retrieval only - rebuild the "
                f"collection to restore semantic search", warn=True)
            return None
        if not _vectors_finite([qvec]):
            # A non-finite query embedding would make every cosine nan and empty
            # the result set.
            self._note_vector_degrade(
                "query embedding has non-finite (NaN/inf) values; using BM25 "
                "lexical retrieval only", warn=True)
            return None
        # Vectors are usable: clear any stale query-time degrade note.
        self.vector_degrade_reason = None
        out = []
        for v in self._vectors:
            out.append(_cosine(qvec, v) if v else 0.0)
        # normalise to [0, 1] like the lexical side
        top = max(out, default=0.0)
        return [s / top for s in out] if top > 0 else out

    # ------------------------------------------------------------- #
    #  Introspection                                                 #
    # ------------------------------------------------------------- #

    @staticmethod
    def _has_vectors(chunks: list, vectors: Optional[list]) -> bool:
        """"Has vectors" = whether query() will actually blend embeddings: the
        same >=80% coverage threshold _vector_scores uses, NOT "every chunk
        embedded". A partially-embedded collection (80-99%) still does hybrid
        retrieval and is reported as having vectors.

        Shared with _save(), which caches the same value into meta.json."""
        present = [v for v in (vectors or []) if v]
        return bool(present) and len(present) >= 0.8 * len(chunks)

    def stats(self) -> dict:
        docs = self._meta.get("docs", {})
        return {
            "name": self.name,
            "created": self._meta.get("created"),
            "n_docs": len(docs),
            # Indexed documents whose source file was gone at the last resync.
            # Still counted in n_docs and still searchable: the flag says the
            # index is ahead of the disk, it does not remove anything.
            "n_missing": sum(1 for e in docs.values()
                             if isinstance(e, dict) and e.get("missing")),
            "n_roots": len(self.roots()),
            "n_chunks": len(self._chunks),
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "corrupt": self.corrupt,
            # Count of chunks.jsonl lines _load() had to skip (0 if none), so a
            # caller can name a count instead of a generic "index damaged".
            "chunks_bad_lines": self.chunks_bad_lines,
            # Why semantic search fell back to BM25 (None when vectors are used or
            # legitimately absent).
            "vector_degrade_reason": self.vector_degrade_reason,
            "vector_dim": self._vec_dim,
        }

    def vector_dim(self) -> Optional[int]:
        """The dimensionality of THIS collection's currently stored vectors, or
        None when it cannot be established: no usable vectors are stored at all
        (see ``stats()["has_vectors"]``), or ``_load()`` found the file present
        but unusable for a reason ``vector_degrade_reason`` names.

        Reads the same ``_vec_dim`` the add-time consistency guard trusts.
        ``_load()`` computes it as ``data.get("dim") or _first_dim(vectors)``,
        so a legacy vectors.json without a "dim" field still resolves it from
        the first stored vector; only an unusable/empty index gives None."""
        return self._vec_dim

    def docs(self) -> list[dict]:
        return self._docs_from_meta(self._meta)

    @staticmethod
    def _docs_from_meta(meta: dict) -> list[dict]:
        return [
            {"path": path, **info}
            for path, info in sorted(meta.get("docs", {}).items())
        ]

    # ------------------------------------------------------------- #
    #  Lazy stats: answer from meta.json alone, no chunks/vectors    #
    # ------------------------------------------------------------- #

    def _file_fingerprint(self) -> dict:
        """(mtime_ns, size) for chunks.jsonl and vectors.json (None for
        either that does not exist) - a cheap (stat only, no content read)
        signature of the two files the ``_stats_cache`` block is derived
        from. Written by _save() right after both files were rewritten, and
        checked by ``_fingerprint_matches`` before the cache is ever trusted, so
        a file that changed WITHOUT going through this class is detected and the
        cache refused."""
        def _stat(name: str) -> "list[int] | None":
            try:
                st = (self.dir / name).stat()
            except OSError:
                return None
            return [st.st_mtime_ns, st.st_size]
        return {"chunks": _stat("chunks.jsonl"), "vectors": _stat("vectors.json")}

    @staticmethod
    def _fingerprint_matches(coll_dir: Path, recorded) -> bool:
        """True when *recorded* (the cache's "fingerprint" value) still
        matches chunks.jsonl and vectors.json on disk RIGHT NOW. Duplicates
        the same two os.stat() calls as _file_fingerprint(); this one runs from
        peek_stats() BEFORE any Collection exists."""
        if not isinstance(recorded, dict):
            return False
        def _stat(name: str) -> "list[int] | None":
            try:
                st = (coll_dir / name).stat()
            except OSError:
                return None
            return [st.st_mtime_ns, st.st_size]
        return (_stat("chunks.jsonl") == recorded.get("chunks")
                and _stat("vectors.json") == recorded.get("vectors"))

    @classmethod
    def _peek_meta(cls, name: str, base: Optional[Path] = None
                    ) -> "tuple[str, Path, dict] | None":
        """(checked name, collection dir, parsed meta.json), or None when
        there is nothing here the lazy path can trust enough to skip the full
        load: an invalid name, no meta.json, or one that fails to parse. On None
        the caller falls back to the real ``Collection(name)``, which carries
        the corrupt-meta.json recovery."""
        try:
            checked_name = _check_name(name)
        except ValueError:
            return None
        coll_dir = (base or rag_dir()) / checked_name
        meta_path = coll_dir / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(meta, dict):
            return None
        return checked_name, coll_dir, meta

    @classmethod
    def _stats_from_meta(cls, checked_name: str, coll_dir: Path,
                         meta: dict) -> Optional[dict]:
        """stats()-shaped dict from an already-parsed meta.json, using ONLY the
        cache _save() writes - never a re-derivation of
        has_vectors/vector_degrade_reason from a partial read of chunks.jsonl /
        vectors.json. None when the cache is absent, its recorded file
        fingerprint no longer matches chunks.jsonl / vectors.json on disk (see
        ``_fingerprint_matches``), or it does not look like this class wrote it;
        the caller must then fall back to the full, authoritative
        Collection(name).stats()."""
        cache = meta.get(_STATS_CACHE_KEY)
        if not isinstance(cache, dict) or not isinstance(cache.get("n_chunks"), int):
            return None
        if not cls._fingerprint_matches(coll_dir, cache.get("fingerprint")):
            return None
        docs = meta.get("docs", {})
        if not isinstance(docs, dict):
            return None
        return {
            "name": checked_name,
            "created": meta.get("created"),
            "n_docs": len(docs),
            "n_missing": sum(1 for e in docs.values()
                             if isinstance(e, dict) and e.get("missing")),
            "n_roots": len(cls._roots_from_meta(meta)),
            "n_chunks": cache["n_chunks"],
            "has_vectors": bool(cache.get("has_vectors")),
            "corrupt": bool(cache.get("corrupt")),
            # 0 when the cache does not carry this field; the caller then
            # falls back to generic "index damaged" wording instead of a count.
            "chunks_bad_lines": cache.get("chunks_bad_lines", 0),
            "vector_degrade_reason": cache.get("vector_degrade_reason"),
            # Absent when the cache does not carry this field; the collection
            # then falls back to the cold load-and-backfill path below once.
            "vector_dim": cache.get("vector_dim"),
        }

    @classmethod
    def peek_stats(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``stats()`` without constructing a full ``Collection`` - reads
        meta.json alone and trusts its cached derived fields, never
        chunks.jsonl or vectors.json. None means "cannot answer cheaply and
        correctly" (see ``_stats_from_meta``); the caller MUST fall back to
        the real ``Collection(name).stats()`` in that case."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        return cls._stats_from_meta(checked_name, coll_dir, meta)

    @classmethod
    def peek_detail(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``peek_stats()`` plus the docs list, for the collection-detail route -
        both read meta.json exactly once. Same None contract as peek_stats()."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        stats = cls._stats_from_meta(checked_name, coll_dir, meta)
        if stats is None:
            return None
        return {**stats, "docs": cls._docs_from_meta(meta)}

    @staticmethod
    def _doc_is_host_path(doc_key: str) -> bool:
        """False for a doc key ``add_uploads`` records (``upload:<filename>``,
        no host filesystem path behind it); True for a doc key ``add_paths``
        records (an absolute, resolved host path)."""
        return not doc_key.startswith("upload:")

    @classmethod
    def _docs_within_roots(cls, doc_keys, key_roots: list) -> bool:
        """True when every host-filesystem doc key in *doc_keys* resolves
        under one of *key_roots* (``_path_within``, both sides resolved).
        Upload-recorded keys (see ``_doc_is_host_path``) are skipped. An
        empty *key_roots*, or one whose entries all fail to resolve to a
        real path, returns True and False respectively."""
        if not key_roots:
            return True
        roots: list[Path] = []
        for r in key_roots:
            try:
                roots.append(Path(r).expanduser().resolve())
            except (OSError, ValueError):
                continue
        if not roots:
            return False
        for key in doc_keys:
            if not cls._doc_is_host_path(key):
                continue
            if not any(_path_within(Path(key), r) for r in roots):
                return False
        return True

    @classmethod
    def confined_to(cls, name: str, key_roots: list, base: Optional[Path] = None
                    ) -> Optional[bool]:
        """Whether every host-filesystem document indexed into collection
        *name* resolves under one of *key_roots*. Reads meta.json only, the
        same cheap path ``peek_stats``/``peek_detail`` use.

        An empty *key_roots* always returns True. Otherwise: True/False from
        ``_docs_within_roots`` over the collection's recorded doc keys, or
        None when meta.json cannot be read or parsed (missing, invalid JSON,
        or a "docs" field that is not an object). A caller enforcing
        confinement treats None the same as False."""
        if not key_roots:
            return True
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        _checked_name, _coll_dir, meta = found
        docs = meta.get("docs", {})
        if not isinstance(docs, dict):
            return None
        return cls._docs_within_roots(docs.keys(), key_roots)

    @classmethod
    def load_and_maybe_backfill(cls, name: str, base: Optional[Path] = None
                                ) -> "Collection":
        """The COLD-fallback path for ``peek_stats()``/``peek_detail()``: a
        full, authoritative load of *name* (identical to plain ``Collection(
        name, base)``), with an opportunistic attempt to backfill its
        ``_stats_cache`` so future listings of this SAME collection stop
        paying the full-load cost.

        ORDERING: the write lock is acquired FIRST and the load happens INSIDE
        it - never load-then-lock. Nothing else can write to this collection
        while the lock is held (every real writer takes the SAME file lock
        before touching disk), so what ``_load()`` reads is current and the
        fingerprint taken from that same held state describes exactly what was
        read.

        Takes ONLY the cross-process file lock, not the in-process
        ``_collection_lock``: this method never mutates anything, it only
        re-derives a cache from bytes it read under the file lock.

        ``timeout=0``: an opportunistic path serving a READ. On a busy lock this
        returns a fully loaded, fully correct ``Collection`` without writing a
        cache."""
        base = base or rag_dir()
        checked_name = _check_name(name)
        coll_dir = base / checked_name
        try:
            with collection_write_lock(
                    lock_path_for(coll_dir), collection=checked_name,
                    op="a stats-cache backfill", timeout=0):
                coll = cls(checked_name, base)   # _load() runs INSIDE the lock
                if coll.exists():
                    coll._meta[_STATS_CACHE_KEY] = coll._stats_cache_block()
                    coll._atomic_write("meta.json", json.dumps(coll._meta, indent=2))
                return coll
        except CollectionLockedError:
            return cls(checked_name, base)       # busy: full load, no cache write


def collection_provenance_report() -> list:
    """Every collection that currently has vectors, with its recorded 'built
    with' model (``Collection.embedding_model()``, None if never recorded)
    and chunk count - the pre-switch, new-model-dimension-free report used by
    every writer of the ``embedding_model`` config key (the RAG picker's
    ``POST /api/rag/embedding``, ``PATCH /v1/config``, and
    ``localm setup-embeddings``) to warn what an embedding-model switch is
    about to invalidate, before it happens.

    Does NOT assert whether a given collection's dimension will actually
    change: that would need the CANDIDATE model's own dimension, which means
    resolving and loading it. This reports only what can be read from disk:
    which collections have semantic search today, and what they were built
    with.

    Best-effort per collection: one that fails to construct is still named,
    with the failure NOTED (not silently dropped from the count). The
    exception's own text is logged server-side only, never placed in the field
    this function returns."""
    out: list = []
    for name in collection_names():
        try:
            coll = Collection(name)
            stats = coll.stats()
        except Exception as e:
            _log.warning("rag: %r could not be read for the embedding-switch "
                        "impact preview (%s: %s)", name, type(e).__name__, e)
            out.append({"name": name, "built_with": None, "n_chunks": None,
                        "reason": "could not be read"})
            continue
        if not stats.get("has_vectors"):
            continue
        out.append({"name": name, "built_with": coll.embedding_model(),
                    "n_chunks": stats["n_chunks"]})
    return out


def collection_provenance_note(model: str, affected: list) -> str:
    """The human-readable note accompanying a ``collection_provenance_report()``
    result, shared by every writer of ``embedding_model`` (the RAG picker,
    ``PATCH /v1/config``, ``localm setup-embeddings``) so the wording a user
    sees does not drift between which surface they switched from."""
    if affected:
        return (
            f"Switching to '{model}' may invalidate the semantic search of "
            f"{len(affected)} existing collection(s) until they are "
            "re-embedded. The exact impact cannot be confirmed until the "
            "new model is loaded and tested - re-embed after switching if "
            "any of them drop to BM25/lexical-only.")
    return (f"No existing collection currently has embeddings, so "
            f"switching to '{model}' has nothing to invalidate.")


def _cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        # Callers (_vector_scores) guarantee equal lengths; a mismatch raises
        # rather than being scored as a real (zero) similarity.
        raise ValueError(
            f"cosine similarity needs equal-length vectors "
            f"(got {len(a)} and {len(b)})")
    try:
        np = _numpy
        if np is None:
            raise ImportError("numpy is not installed")
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        sim = float(va @ vb) / denom if denom else 0.0
    except (ImportError, AttributeError) as e:
        # ImportError is the ordinary "numpy not installed" case; AttributeError
        # is an importable but attribute-less numpy, where np.asarray is missing.
        # Not a bare except: a real numerical error from a usable numpy still
        # propagates. The degrade is announced once per process.
        _warn_numpy_degrade(e, "cosine similarity")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        sim = dot / (na * nb) if na and nb else 0.0
    # A NaN/inf component makes the similarity non-finite; non-finite is
    # returned as a miss (0.0) and never leaves this function.
    return sim if math.isfinite(sim) else 0.0
