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
entries the folder walk produced. Without it an index can only ever be refreshed
over the files it already knows, so a file ADDED to (or DELETED from) an indexed
folder after the fact is invisible - which is exactly the drift a folder re-sync
exists to catch. ``resync()`` re-walks these roots through the ordinary
incremental path.

Collections are explicit user data (like generated images): indexing writes
to disk in every session mode. Rewrites are whole-file + atomic rename -
corpora here are home-scale (thousands of chunks, not millions).

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

# numpy is bound ONCE, HERE, at module import - deliberately not lazily inside the
# functions that use it. It is an optional dependency (not pinned in
# pyproject.toml; it arrives transitively), hence the None fallback.
#
# The lazy imports this replaces were the SOURCE of the partial-initialisation
# fault, not merely a victim of it. CPython's import lock has been PER-MODULE
# since 3.3, so a thread that finds numpy already in sys.modules does not wait for
# it to finish initialising - it gets the module as it currently stands. Both
# users of numpy here run on the shared plugin ThreadPoolExecutor (rag/plug.py:357
# and :612), so two requests could race the FIRST import: thread A starts numpy's
# long top-level init, thread B sees `import numpy` succeed and finds np.asarray
# missing. Binding at module import is single-threaded by construction and closes
# the window entirely.
#
# It also explains what a state-handling fix alone could not: why the failure is
# NONDETERMINISTIC on identical trees, why CI reddens where local passes
# (contention widens the window), and why one OS stays green (different
# scheduling). Diagnosis credit: local_f7754072 (CodeQL triage lane).
#
# The (ImportError, AttributeError) catches at the call sites STAY, as defence for
# any partial-init shape this binding does not prevent.
try:
    import numpy as _numpy
except ImportError:      # optional dependency - every caller degrades to pure Python
    _numpy = None
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

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

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
    list/tuple. A hand-corrupted or truncated vectors.json can hold scalars or
    strings that pass JSON parsing but crash cosine scoring at QUERY time with an
    opaque error (``object of type 'int' has no len()``); catch that at load and
    degrade to BM25 with a reason instead of shipping a delayed crash."""
    return isinstance(vectors, list) and all(
        (not v) or isinstance(v, (list, tuple)) for v in vectors)


def _vectors_finite(vectors) -> bool:
    """True when every component of every present vector is a FINITE number.

    A single NaN/inf component makes ``_cosine`` return ``nan``; the blended query
    score is then ``nan``, and ``nan > 0`` is False, so the chunk is SILENTLY
    dropped from results (a query for a word that IS indexed returns everything
    except the matching chunk) with no error and no degrade reason. A non-finite
    or non-numeric vector store must therefore degrade to BM25 with a surfaced
    reason, never be trusted (AGENTS rule 5). Structure is already validated by
    ``_well_formed_vectors``; this checks the values."""
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
    # AttributeError as well as ImportError: this is the site where an unusable
    # numpy did the MOST damage. _cosine returning a wrong score mis-ranks; THIS
    # function returns the boolean that decides whether the vector store is
    # trusted at all, and an escaping AttributeError does not degrade to BM25 with
    # a surfaced reason (what the docstring promises) - it errors the whole query.
    # CI only ever pointed at _cosine because it was reached first.
    except (ImportError, AttributeError) as e:
        if not _NUMPY_DEGRADE_LOGGED:
            _NUMPY_DEGRADE_LOGGED.add(True)
            _log.warning(
                "numpy is present but unusable (%s: %s); falling back to pure-Python "
                "vector validation. Results are identical, checking is slower on "
                "large collections.", type(e).__name__, e)
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
ProgressFn = Callable[[str], None]


def rag_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "rag"


# Well-known THIRD-PARTY credential/secret folders under the user's home that
# must never be indexed even though they sit inside an allowed root - these hold
# OTHER services' secrets (SSH, cloud CLIs) that a folder walk could sweep in
# unnoticed, unrelated to localm's own data. Deliberately does NOT include
# ".localm" (the conventional default LOCALM_HOME name): localm does not block
# the owner from indexing their own data directory at all (see
# confine_index_path's docstring) - it is their machine and their choice.
_SENSITIVE_HOME_SUBDIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure",
)
# Lower-cased for matching a path component ANYWHERE in a resolved path, so a
# nested ~/proj/.ssh is caught too, and case-insensitively so a ".SSH"
# component cannot slip past on a case-insensitive filesystem.
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


# Cap the folder-walk recursion depth. Real document trees are nowhere near
# this deep; the cap is a backstop against a pathological directory cycle.
_MAX_WALK_DEPTH = 50


def _walk_files(root: Path, *, max_depth: int = _MAX_WALK_DEPTH):
    """Yield files under *root* without following linked DIRECTORIES, bounded by
    depth and a visited-realpath set.

    ``rglob('*')`` follows NTFS junctions - which report ``is_symlink() == False``,
    so pathlib's symlink-loop guard misses them - and a self-referential junction
    makes the walk spin until the path length overflows (a folder-index DoS, B3).
    This manual walk never descends into a linked directory (junction OR
    bind-mount OR symlink) and refuses to revisit a resolved directory, so no
    directory cycle can hang indexing. ``_SKIP_DIRS`` are pruned during descent.

    A linked FILE **is** yielded (REG-569): only a directory can cycle, so a link
    to a file is a terminal node the guard does not need to exclude, and dropping
    those silently lost real documents that ``rglob`` used to index. Confinement
    (a link escaping an allowed root) is enforced by ``_expand``'s confine loop,
    not here."""
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
                    # Resolve what the link POINTS AT and branch on it, rather
                    # than skipping every link (REG-569).
                    #
                    # A linked DIRECTORY is still not followed: that is what the
                    # loop + escape guard is for, and it is the only shape that
                    # can cycle. A linked FILE is a TERMINAL node - the walk never
                    # recurses into it - so following one cannot loop, and
                    # skipping it silently dropped legitimate documents from the
                    # index (a docs folder of links to files elsewhere is a very
                    # common layout; rglob+is_file indexed them before this walk
                    # replaced it). Escape stays handled a layer up: _expand's
                    # confine loop rejects symlinks escaping an allowed folder
                    # when a policy is given, and the policy-less CLI operator is
                    # unconfined by design.
                    #
                    # On Windows a file symlink sets the reparse-point attribute
                    # too (so does e.g. a cloud-sync placeholder), which is why
                    # this branches on the RESOLVED type instead of the attribute.
                    try:
                        if e.is_dir(follow_symlinks=True):
                            # Log at debug so a user who expected a symlinked docs
                            # folder to be indexed can discover why (rule 5).
                            _log.debug("rag: not following linked directory during "
                                       "index walk: %s", e.path)
                            continue
                        if e.is_file(follow_symlinks=True):
                            yield Path(e.path)
                            continue
                        # Neither: a dangling/unresolvable link. Nothing to index,
                        # but say so rather than dropping it without a trace.
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
    """A path may not be indexed. ``reason`` tells the caller WHY, so the API can
    offer 'add and continue' for a fixable whitelist miss (``outside_allowed``)
    but hard-refuse the rest (``credential`` / ``secret_file`` / ``denied`` /
    ``invalid``). Subclasses ``ValueError`` so existing ``except ValueError``
    sites keep catching it."""

    def __init__(self, message: str, *, path: Path, reason: str):
        super().__init__(message)
        self.path = path
        self.reason = reason


_INDEX_MODES = ("whitelist", "blacklist")


def indexing_policy(cfg: Optional[dict] = None) -> dict:
    """The current RAG indexing confinement policy, read from config.

    ``mode`` is ``whitelist`` (index only your home folder, the working directory,
    and the ``rag_allowed_roots`` you added) or ``blacklist`` (index anywhere
    EXCEPT the ``rag_denied_roots`` you listed). In BOTH modes the localm data dir
    and credential folders are still refused - a hard floor that
    ``confine_index_path`` enforces separately and no mode can turn off. Returns
    resolved ``Path`` lists so callers compare like-for-like.
    """
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception as e:
            # A config we cannot load falls back to an EMPTY policy, which
            # confine_index_path treats as whitelist-with-no-extra-roots: the safe,
            # fail-CLOSED direction (it refuses more, never less). Benign default, but
            # not hidden - a genuinely corrupt config should be discoverable. Rule 5.
            from localm.debuglog import logger as _dbg
            _dbg.debug("rag indexing_policy: could not load config, using an empty "
                       "fail-closed policy: %s", e)
            cfg = {}
    mode = cfg.get("rag_indexing_mode", "whitelist")
    if mode not in _INDEX_MODES:
        mode = "whitelist"

    def _resolve(key: str) -> list[Path]:
        out: list[Path] = []
        for r in cfg.get(key, []) or []:
            try:
                out.append(Path(r).expanduser().resolve())
            except (OSError, ValueError) as e:
                # A configured root we cannot resolve is DROPPED, but the two lists
                # fail in OPPOSITE directions: dropping an ALLOWED root only fails
                # CLOSED (a would-be-indexable path is then refused), while dropping a
                # DENIED root fails OPEN - it silently removes a privacy control, so
                # confine_index_path can no longer refuse a path inside a folder the
                # user explicitly denied even though the UI still shows the deny. Rule
                # 5: never silence this. Warn, naming the root, loudly for the denied
                # list. (Cross-note: localm-privacy-review.)
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
            "denied": _resolve("rag_denied_roots")}


def confine_index_path(p, policy: Optional[dict] = None) -> Path:
    """Resolve *p* and verify it may be indexed, raising ``ConfinementError`` (a
    ``ValueError``) otherwise.

    The HARD FLOOR is enforced ALWAYS, even when *policy* is None: well-known
    credential folders (``.ssh``, ``.aws``, ...) are never indexable - wherever
    they appear in the resolved path, so a nested ``~/proj/.ssh`` or a symlink
    into one is caught too.

    The localm data directory (LOCALM_HOME) is NOT refused, at all: localm is a
    local, single-user tool, and it is not localm's place to block the owner
    from indexing their own files, including config.json/registry.json/auth.json
    if they explicitly choose to - they already have direct filesystem access to
    every one of them. Keeping documents in (or portably alongside, as one
    self-contained folder) the data directory is a legitimate choice, not
    something to guard against.

    With a *policy* (the HTTP API passes ``indexing_policy()``):
      - ``whitelist``: *p* must be within your home folder, the working directory,
        or a ``rag_allowed_roots`` entry, else ``reason='outside_allowed'`` (the
        route may offer the owner to add it and continue) - this applies to
        LOCALM_HOME exactly like any other folder outside the defaults, not as a
        special case;
      - ``blacklist``: *p* is allowed unless it is within a ``rag_denied_roots``
        entry, then ``reason='denied'``.

    ``policy=None`` means hard-floor only: the local CLI operator, who can already
    read their own files, is otherwise unconfined.
    """
    try:
        rp = Path(p).expanduser().resolve()
    except (OSError, ValueError):
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid")

    # Credential folders are denied wherever they appear in the resolved path,
    # not only at the home root: ~/proj/.ssh and <cwd>/sub/.aws are as sensitive
    # as ~/.ssh. rp is already resolved, so this also catches a symlink that
    # points into a credential dir. Tradeoff: a folder literally named one of
    # these (a real ./.docker you wanted to index) is refused too - acceptable
    # for a credential denylist.
    if any(part.lower() in _SENSITIVE_NAMES for part in rp.parts):
        raise ConfinementError(f"Refusing to index a credential directory: {p}",
                               path=rp, reason="credential")

    if policy is None:
        return rp

    # --- API floor: refuse model-weight / binary / credential FILES (policy set) ---
    # The recursive folder walk (_expand) already skips these by suffix + secret
    # name, but an EXPLICITLY-named top-level file used to bypass that filter, so a
    # `rag`-scoped HTTP caller could POST paths=["<home>/deploy.pem"] and read the
    # key back via /query (C2). Apply the SAME filter to explicit picks whenever a
    # policy is present (every API caller, owner and non-owner alike - a loopback
    # page or remote client is untrusted). This runs BEFORE the mode branches so a
    # secret is never offered through the whitelist "add and continue" consent flow.
    # Guarded on is_file() so a directory merely NAMED like a secret (a real
    # ./credentials or ./.env folder) is still walkable, not over-blocked. The CLI
    # (policy=None, returned above) stays unconfined: the local operator can already
    # read their own files, so an explicit single-file pick is still honoured there.
    if rp.is_file() and (rp.suffix.lower() in SECRET_SUFFIXES
                         or is_secret_index_name(rp.name)):
        raise ConfinementError(
            f"Refusing to index {rp.name}: key/credential material is not "
            f"indexed through the API. Use the local CLI (`localm rag add`) if "
            f"you really intend to.",
            path=rp, reason="secret_file")
    # NOTE: a non-secret binary/media file (UNINDEXABLE_SUFFIXES: .mp4, .db, .7z,
    # model weights, ...) deliberately does NOT raise here. Confinement is a
    # SECURITY boundary, and "this file has no text in it" is not a security
    # question - refusing it here made the caller's WHOLE request fail, so one
    # video in a 30-file pick indexed nothing (REG-567). It is instead reported as
    # an individual per-file failure by _add_paths_locked, which still refuses it
    # BEFORE reading the bytes, so the multi-GB-model perf guard is preserved.

    if policy.get("mode") == "blacklist":
        # Allow anything not explicitly denied (the hard floor above still holds).
        # Path(d) coerces in case a caller hand-built the policy with str entries
        # (indexing_policy() already returns Paths; this just never trusts that).
        for d in policy.get("denied", []):
            if _path_within(rp, Path(d)):
                raise ConfinementError(
                    f"This folder is on your denied list, so it is not indexed: {p}",
                    path=rp, reason="denied")
        return rp

    # whitelist: your home folder + the working dir are always allowed, plus the
    # roots you added. Anything else is a fixable miss (the owner can widen).
    roots: list[Path] = []
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
    """Validate a collection name, returning it, or raise ``ValueError``.

    Public because callers OUTSIDE this package need the same rule before a
    collection is touched: the jobs store validates a scheduled re-sync job's
    ``collection`` at definition time, so a typo is rejected when the job is
    created rather than failing silently on every unattended tick.
    """
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            "Collection names must be 1-64 letters, digits, '-' or '_'")
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(f"'{name}' is a reserved device name and cannot be used")
    return name


# Internal alias kept for the in-module call sites.
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

    Deleting IS a write, so it takes the same locks as one. Without them a
    delete could land in the middle of another process's indexing run and leave
    the collection half-rebuilt by that run's final _save() - a collection the
    user believes they deleted, holding a subset of its documents. Raises
    ``CollectionLockedError`` if that other run does not finish in time, rather
    than deleting underneath it."""
    import shutil
    base = base or rag_dir()
    path = base / _check_name(name)
    if not (path / "meta.json").is_file():
        return False
    # The ONE caller that bounds the in-process half too (Collection._write_lock
    # explains why writers normally queue there). A delete is a foreground action
    # someone is waiting on, and it can land in an HTTP request thread, so
    # queueing behind a re-sync that legitimately runs for hours would be a hang
    # with no way to report itself. Refusing after the same budget is honest and
    # actionable; nothing is lost, the collection is still there to delete.
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


# Per-collection-NAME locks. The mutation race (CHK-RAG-LOCK) is across Collection
# INSTANCES: two requests each construct Collection(name), each _load()s the same
# on-disk state, each add a different doc, and each _save()s - last writer wins and
# one update is silently lost. A per-instance lock cannot help (different objects);
# the lock must be keyed by the collection name so writes to one collection serialise
# process-wide. Keyed by name, so the map is bounded by the number of collections
# (small, stable), not per-event. RLock so a locked method may call another safely.
# Shared registry implementation (storekit.NamespaceLockRegistry) - see CF-9/CF-10:
# memory/store.py independently re-implements the identical lazy-RLock-per-key
# pattern, keyed by namespace hash instead of collection name.
#
# SCOPE, stated rather than implied: this lock is PER PROCESS. It serialises the
# server's own concurrent writers (API adds, a scheduled re-sync job), which is
# what CHK-RAG-LOCK was about. It does NOT reach a `localm rag add|resync` CLI
# invocation, which opens the collection directly in its own process with its own
# registry. That second half is closed by collection_lock.collection_write_lock,
# a lock FILE beside the collection directory, held INSIDE this one (see
# Collection._write_lock for why that nesting order, and collection_lock's module
# docstring for why config._cross_process_lock could not simply be reused).
_COLLECTION_LOCKS = NamespaceLockRegistry()


def _collection_lock(name: str):
    # Keyed case-INSENSITIVELY. Collection names are not normalised, so
    # Collection("Docs") and Collection("docs") are two different keys - but on
    # Windows and macOS they are the SAME directory and the same lock file. Two
    # threads spelling the name differently would then sail past each other here
    # and meet at the lock file, which cannot tell one thread from another and
    # would report a thread of this very process as "another localm process".
    # Folding costs only some needless mutual exclusion on a case-sensitive
    # filesystem, where the two really are separate collections.
    return _COLLECTION_LOCKS.get(name.casefold())


# Where a vectors.json that _load() REFUSED is set aside when the chunks it was
# (mis)aligned with get rewritten. Preserved, never deleted - see
# Collection._quarantine_rejected_vectors.
_REJECTED_VECTORS = "vectors.json.rejected"

#: How many set-aside vector indexes to KEEP per collection. Each one is a full
#: copy of the index (10 MB is ordinary, 68 MB was observed), and they accumulate
#: with no upper bound in practice - a real install had three totalling 88 MB.
#: Preserving the evidence is right (AGENTS rule 5); preserving EVERY generation
#: of it forever is just an unbounded disk leak, and the oldest copies are the
#: least useful ones. Keep the newest few, delete the rest LOUDLY.
_MAX_REJECTED_KEPT = 3

#: Set once when numpy has been found present-but-unusable, so the pure-Python
#: cosine fallback announces itself exactly once per process instead of on every
#: scored chunk. A set rather than a bool because rebinding a module-level bool
#: from inside a function needs `global`, and this is read from a hot path.
_NUMPY_DEGRADE_LOGGED: set = set()

#: (collection dir, reason) pairs already logged in THIS process. _load() runs
#: from __init__ and every request builds a fresh Collection, so an instance-level
#: guard re-armed constantly and the same warning was emitted 25+ times a session.
#: Process-scoped so the first occurrence is still loud and the rest are quiet.
#: Never consulted for state - only for whether to LOG (see _note_vector_degrade).
_WARNED_DEGRADES: set = set()


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
        # Why semantic (vector) scoring is unavailable when it should be present.
        # None = vectors are used, or legitimately absent (no embeddings indexed).
        # A non-None string means a corrupt/stale/mismatched vectors index was
        # DETECTED and we fell back to BM25 lexical - surfaced, not silently
        # swallowed (AGENTS rule 5). Exposed via stats() and logged once.
        self.vector_degrade_reason: Optional[str] = None
        # True when _load() found a vectors.json on disk and REFUSED to use it.
        # That file is then both the only remaining copy of those vectors and the
        # only evidence of the fault, so _save() must not delete it (see _save).
        # Distinct from vector_degrade_reason, which is ALSO set by query-time
        # degrades (a failed query embedding, partial coverage) that say nothing
        # about the file on disk.
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
        ``_collection_lock``, never the other way round, for two reasons. It is
        one consistent order everywhere, so the pair cannot deadlock; and it
        means at most one thread of this process is ever at the lock file, so a
        second thread of the same process waits on the in-process lock instead
        of meeting its own process's lock file (which the file lock, like
        config's, can only read as a nested call - a bug - since a file lock
        cannot tell one thread from another).

        The two halves have DIFFERENT waiting rules, on purpose. Writers inside
        one process QUEUE for as long as it takes: the holder is a thread of
        this process that is demonstrably making progress, so waiting always
        ends and always does the work - refusing there would break something
        that works today (a second GUI index of a collection whose first index
        runs for ten minutes). A writer in ANOTHER process cannot be trusted
        that way, since it may be hung or gone, so that half is bounded and
        ends in a refusal. ``delete_collection`` is the one caller that bounds
        both (see its docstring).

        The wait is reported through the caller's existing progress channel, so
        a CLI or a job stream says why it is waiting instead of looking hung.
        """
        return collection_write_lock(
            lock_path_for(self.dir), collection=self.name, op=op,
            on_wait=on_progress)

    def create(self) -> "Collection":
        """Create the collection if it does not exist yet.

        Under the write lock, and re-checking existence inside it: creating is a
        write too. Without it, a `rag add` that finds no collection can drop a
        fresh meta.json into a directory another process is part-way through
        deleting - which both resurrects a collection the user deleted and
        fails that delete's final rmdir on a directory that grew entries after
        it was listed. The fast path (already exists) takes no lock at all, so
        the usual add pays nothing."""
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
        # A corrupt meta.json must not crash construction (and thus the whole
        # collections listing). Flag it - but do NOT discard the INDEPENDENT
        # chunks.jsonl / vectors.json files: meta.json holds only
        # {name, created, docs}, none of which retrieval needs, so intact chunks
        # stay fully queryable. Collapsing "meta corrupt" into "empty collection"
        # silently loses recoverable data (AGENTS rule 5: a "missing" and a
        # "corrupt" input must not share one silent path). We fall through to load
        # the chunks and then reconstruct a minimal docs map from their sources so
        # `rag repair` can rebuild and the next _save() self-heals meta.json.
        self.corrupt = False
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
            # and nothing else, but splitlines() also breaks on U+0085/U+2028/
            # U+2029, which json.dumps(ensure_ascii=False) writes RAW. That tore
            # one record into two unparseable fragments, and the damage did not
            # stop at a warning: the dropped records made the vector sidecar
            # look stale (silent degrade to lexical), got it quarantined, and
            # then _save() rewrote this file from the survivors - deleting the
            # user's chunks. Measured on a real collection: 36 raw U+0085 turned
            # 1192 records into 1228 lines, 62 unparseable, 26 chunks lost per
            # load/save cycle. See localm/jsonl.py.
            for line in split_jsonl(chunks_file.read_text(encoding="utf-8")):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                # A chunk MUST be a dict carrying a str "text": query(),
                # remove_doc() and add_paths() all assume that shape. A
                # valid-JSON-but-wrong-shape line (an externally appended scalar or
                # array, or a dict missing "text") would otherwise crash every one
                # of those with TypeError/AttributeError/KeyError and brick the
                # collection while stats() reported it healthy. Skip it and surface
                # the corruption - symmetric with the meta.json and vectors.json
                # guards (AGENTS rule 5: validate shape, do not silently trust).
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    bad_lines += 1
                    continue
                self._chunks.append(obj)
            if bad_lines:
                self.corrupt = True
                _log.warning("RAG collection %r: skipped %d malformed line(s) in "
                             "chunks.jsonl; run 'localm rag repair'",
                             self.name, bad_lines)
        self._vectors = None
        self._vec_dim = None
        self.vector_degrade_reason = None
        self._vectors_file_rejected = False
        vec_file = self.dir / "vectors.json"
        if vec_file.is_file():
            # vectors.json PRESENT but unusable is the unexpected case: do NOT
            # collapse it with "simply absent" into one silent path (AGENTS rule 5).
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
                    # Valid JSON but the entries are not vectors (scalars/strings
                    # from a hand-edit or truncation): would crash cosine at query
                    # time, so treat it as corrupt here rather than later.
                    self._note_vector_degrade(
                        "vectors.json is malformed (entries are not vectors); "
                        "using BM25 lexical retrieval only", warn=True)
                elif len(vectors) == len(self._chunks):
                    if not _vectors_finite(vectors):
                        # Structurally a vector list, but a component is NaN/inf or
                        # non-numeric - would silently drop chunks at query time
                        # (nan cosine, nan !> 0). Degrade + surface, do not trust.
                        self._note_vector_degrade(
                            "vectors.json has non-finite (NaN/inf) or non-numeric "
                            "values; using BM25 lexical retrieval only", warn=True)
                    else:
                        self._vectors = vectors
                        self._vec_dim = data.get("dim") or _first_dim(vectors)
                elif vectors:
                    # A non-empty vectors list that does not line up with the
                    # chunks is a stale/partial index, not "no embeddings yet".
                    # Distinguish the two real causes instead of one blanket
                    # "stale or partial" phrase: FEWER vectors than chunks is a
                    # genuinely partial embed (e.g. interrupted mid-run, or a
                    # doc added while embed_fn was broken); MORE vectors than
                    # chunks means leftover/orphaned entries from a prior,
                    # larger chunk set (e.g. docs removed or re-chunked without
                    # the vector list being pruned to match) - not "in
                    # progress". Both are fixed the same way (a full reindex
                    # rebuilds vectors in lockstep with chunks - see
                    # _add_paths_locked), but the diagnosis differs.
                    kind = ("a partial embed" if len(vectors) < len(self._chunks)
                            else "orphaned entries from a prior, larger index")
                    self._note_vector_degrade(
                        f"vectors.json has {len(vectors)} vectors for "
                        f"{len(self._chunks)} chunks ({kind}); "
                        f"using BM25 lexical retrieval only", warn=True)
            # Any reason recorded in this block means the file IS there and we
            # refused it (the reason was reset to None immediately above, so
            # nothing else can have set it). Remember that for _save().
            self._vectors_file_rejected = self.vector_degrade_reason is not None
        # An earlier write set an unusable sidecar aside rather than deleting it
        # (_quarantine_rejected_vectors). Nothing was lost, but semantic search is
        # still degraded until the index is actually REBUILT - so keep saying so.
        # Tidying a fault out of the way must not also tidy away the fact that it
        # happened (AGENTS rule 5).
        #
        # Checked independently of vectors.json, and gated on COMPLETENESS rather
        # than on that file's mere presence. As an `elif` on "no vectors.json" this
        # went silent the moment anything wrote one - which the commonest path
        # does: re-embedding a single changed document writes real vectors for it
        # and null placeholders for every other chunk, a structurally valid file
        # that loads clean. The collection then reported no degrade at all while
        # most of its chunks had silently lost their vectors. A COMPLETE index is
        # the honest all-clear, because it is exactly what the prescribed remedy
        # ('rag repair --embed') produces; anything less still needs saying.
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
        # map from the chunk sources. This makes stats()/documents() reflect the
        # recoverable data (not a false "empty"), lets `rag repair` re-index the
        # real source files WITHOUT duplicating chunks (add_paths keys its
        # replace-in-place on a known doc), and self-heals meta.json on the next
        # _save(). The reconstructed entries lack mtime/size/hash, so a later
        # add/repair re-reads the file - correct. Gated on META corruption
        # specifically: a valid meta whose chunks.jsonl merely had a bad line must
        # keep its real docs map (with mtime/size/hash), not have it overwritten.
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
        # would otherwise emit raw (U+0085/U+2028/U+2029), so a record can never
        # again be split in half by a line-oriented reader - ours or anyone's.
        # The reader above is fixed independently, so files written before this
        # still load; this stops NEW ones being produced. See localm/jsonl.py.
        self._atomic_write("chunks.jsonl", dumps_lines(self._chunks))
        # The fate of a REJECTED vectors.json is decided FIRST, before anything
        # below writes or unlinks that filename. Setting it aside from inside only
        # one branch loses the bytes on every other one, and the branch it lived in
        # was not the common case: with an embedder available, re-indexing a single
        # changed document produces some real vectors, which took the WRITE branch
        # and overwrote the rejected file (and the evidence) at the same time.
        if self._vectors_file_rejected:
            if self._chunks:
                self._quarantine_rejected_vectors()
            self._vectors_file_rejected = False
        # "Complete" means every chunk has a usable vector: the state the
        # prescribed remedy ('rag repair --embed') produces, and the only honest
        # all-clear. Partial coverage is a legitimate, supported state - it just is
        # not a rebuild, so it does not clear a set-aside sidecar's degrade.
        complete = self._vector_index_complete()
        if self._vectors is not None and any(v for v in self._vectors):
            self._vec_dim = _first_dim(self._vectors)
            self._atomic_write("vectors.json", json.dumps(
                {"dim": self._vec_dim, "vectors": self._vectors}))
        else:
            # Nothing usable to write. Whatever is at this filename now is either
            # ours from a previous save or nothing at all - a REJECTED file was
            # already moved out of the way above, so this can no longer delete the
            # only copy of a fault's evidence, which is what it used to do.
            (self.dir / "vectors.json").unlink(missing_ok=True)
            self._vec_dim = None
        if not self._chunks:
            # Every document is gone. Stored vectors are positional against
            # chunks, so with nothing left to realign them to they are
            # unrecoverable rather than recoverable, and a sidecar kept here would
            # pin a degrade on an empty collection that no rebuild could ever
            # clear ('rag repair' returns early with no documents to re-index).
            # This deletion is not hiding a fault: there is no longer a collection
            # for the fault to be about.
            self._discard_rejected_vectors("the collection no longer has any "
                                           "documents to realign them to")
            self.vector_degrade_reason = None
        elif complete:
            # Every chunk has a vector: exactly what 'rag repair --embed'
            # produces, so the fault the sidecar recorded is genuinely fixed and
            # the degrade must clear, or the remedy we print would never work.
            # The sidecar itself is KEPT - the bytes cost little and are the only
            # record of what went wrong - it just stops meaning "still broken".
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

    def _vector_index_complete(self) -> bool:
        """True when every chunk currently has a usable vector.

        The all-clear condition for a set-aside sidecar, and deliberately
        stricter than "a vectors.json exists": partial coverage is a legitimate
        state (a collection mid-embed), but it is not a REBUILD, so it must not
        clear a recorded fault. Empty chunks is not "complete" - there is nothing
        to be complete about, and that case is handled on its own."""
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

        The ONLY place they are ever removed, and only when they have become
        unrecoverable rather than merely inconvenient. Announced at warning level
        because deleting preserved evidence is exactly the kind of thing that
        must never happen quietly (AGENTS rule 5)."""
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

        Preserving the file in place would be enough to keep the data and the
        evidence, but not enough to keep it SAFE: the caller has just rewritten
        chunks.jsonl, and ``_load`` decides a vectors sidecar is usable partly by
        comparing its length to the chunk count. A rejected file left in place can
        therefore be silently RE-ADOPTED once an unrelated change happens to make
        the counts agree again (index 2 documents, truncate vectors.json to the
        second document's vector, remove the first document: one vector, one
        chunk, structurally valid, and every semantic score from then on is
        computed against the wrong chunk). Trading one silent fault for another is
        not a fix (AGENTS rule 5).

        Renaming solves both at once: the bytes are still on disk for recovery and
        for anyone diagnosing what happened, and no loader will ever pair them with
        chunks again. ``_load`` reports the set-aside file as a degrade for as long
        as it exists, so the fault stays visible rather than becoming folklore."""
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
            # Best-effort, and the failure is NOT silent. Leaving the file where
            # it is still preserves the data and the evidence (the property that
            # actually matters); only the re-adoption guard is lost.
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

        Each set-aside file is a full copy of the vector index, so an unbounded
        pile is a disk leak: one real install had vectors.json.rejected,
        .rejected.2 and .rejected.3 totalling 88 MB, and the allocator would have
        gone on to twenty. Preserving the evidence is the rule; preserving every
        generation of it forever is not what that rule asks for, and the OLDEST
        copies are the least diagnostic.

        Ordered by MTIME, deliberately not by name: ``_rejected_vector_files``
        sorts lexicographically, which puts ``.rejected.20`` before ``.rejected.3``
        and would make "oldest" wrong the moment a collection passed nine
        rejections. Deletion is announced at WARNING level for the same reason
        ``_discard_rejected_vectors`` announces its own - removing preserved
        evidence must never happen quietly."""
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

        A collection can degrade more than once (it can be repaired back to
        health and break again; and _save writes meta, chunks and vectors as
        three independent atomic writes, so an ill-timed crash is a repeatable
        route to a second rejection). ``os.replace`` overwrites its destination
        without a word, so a fixed name meant the SECOND incident destroyed the
        first preserved copy while this very function logged "Nothing was
        deleted" - a false safety statement on top of real data loss. Numbering
        keeps every copy. The cap is not a limit on preservation but a limit on
        silently filling a disk: past it we keep the file where it is and say so,
        which loses only the re-adoption guard, never the bytes."""
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
        missing/restored flag), this is both sufficient and strictly safer than a
        full ``_save()``: ``_save`` rewrites chunks.jsonl and decides the fate of
        vectors.json from ``self._vectors``, which ``_load`` sets to None on
        purpose when it finds a corrupt or stale vector sidecar
        (``vector_degrade_reason``). A metadata-only write must not turn that
        recoverable state into real data loss. ``_save`` now refuses that
        particular deletion itself (it sets a rejected file aside before any
        branch can write or unlink that filename), but the two guards are
        deliberately independent: this one keeps a metadata write from touching
        chunks or vectors AT ALL, which is the property callers here actually
        want. Chunks are untouched, so the cached
        BM25 index stays valid too."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))

    def _atomic_write(self, filename: str, content: str) -> None:
        # storekit.atomic_write: unique temp name + Windows PermissionError
        # retry (an AV real-time scan / Search Indexer can transiently hold a
        # handle to the target, which would otherwise fail a good write).
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
        below - add_paths already validated the top-level inputs, so this catches
        nested escapes and the per-file secret filter. With no policy (the CLI)
        explicit picks are unfiltered: the local operator is unconfined."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_file():
                out.append(p.resolve())
            elif p.is_dir():
                # _walk_files (NOT rglob) so a Windows junction loop cannot hang
                # the index walk (B3); it also skips symlinks/reparse points and
                # prunes _SKIP_DIRS during descent.
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
        tracked by its own ``docs`` entry, which ``resync`` re-checks directly;
        recording its parent folder would silently widen the index to every
        sibling file the user never asked for.

        Called from ``_add_paths_locked`` AFTER the confinement check, so a folder
        the policy refuses never becomes a persisted root that a later unattended
        re-sync would walk.
        """
        roots = self._meta.setdefault("roots", {})
        if not isinstance(roots, dict):
            # A hand-edited / externally written meta.json could hold anything
            # here. Do not crash and do not silently keep using the bad value:
            # replace it and say so, the same shape as the other meta guards.
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
                # An unresolvable path was already skipped by _expand; note why
                # rather than dropping it without a trace (AGENTS rule 5).
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
        corrupt (the roots cannot be reconstructed from chunk sources the way the
        docs map can - re-add the folder to restore them)."""
        roots = self._meta.get("roots")
        return sorted(roots) if isinstance(roots, dict) else []

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  classify_fn: Optional[ClassifyFn] = None,
                  describe_image_fn: Optional[DescribeImageFn] = None,
                  on_progress: Optional[ProgressFn] = None,
                  policy: Optional[dict] = None,
                  force: bool = False) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size+content hash) are
        skipped; changed ones are re-indexed in place. Pass ``force=True`` to
        re-index every file regardless (``localm rag add --force`` / repair).
        Returns counters plus per-file failures. embed_fn failures degrade to
        lexical-only, never abort.

        When *policy* is given (the HTTP API passes ``indexing_policy()``), an
        out-of-bounds top-level path raises ``ValueError`` and nested escapes are
        dropped (C2). CLI callers omit it and stay unconfined. Indexing with an
        embedding model whose dimensionality differs from the collection's also
        raises ``ValueError`` rather than corrupting the vectors with mixed
        dimensions (C3).
        """
        # Serialise the whole read-modify-write per collection AND re-sync with the
        # latest committed state under the lock, so a concurrent add_paths() that
        # finished first is not read-stale-then-overwritten (CHK-RAG-LOCK). The
        # _load() must happen INSIDE both locks: state read before the lock is
        # exactly the stale copy that overwrites someone else's committed work.
        with _collection_lock(self.name), self._write_lock("an index", on_progress):
            self._load()
            return self._add_paths_locked(
                paths, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, policy=policy, force=force)

    def _add_paths_locked(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                          classify_fn: Optional[ClassifyFn] = None,
                          describe_image_fn: Optional[DescribeImageFn] = None,
                          on_progress: Optional[ProgressFn] = None,
                          policy: Optional[dict] = None,
                          force: bool = False) -> dict:
        """The add_paths read-modify-write body. MUST run under
        _collection_lock(self.name) after a fresh _load() (see add_paths)."""
        say = on_progress or (lambda _t: None)
        if policy is not None:
            for p in paths:
                confine_index_path(p, policy)   # raises ValueError
        # Persist the FOLDER roots now that confinement has accepted them, so a
        # later re-sync can re-walk them (module docstring). Done before the
        # expand so an add that finds no indexable file still records the folder -
        # an empty folder that gets its first document tomorrow must be picked up.
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
            # An EXPLICITLY-NAMED non-secret binary (a folder walk already filtered
            # these out in _expand, so only a direct pick reaches here). Report it
            # as an ordinary per-file failure - the same shape an ExtractError
            # below produces - so the rest of the batch still indexes (REG-567).
            # BEFORE stat/read_bytes: a named .gguf/.safetensors must never be
            # pulled into RAM and hashed just to be rejected.
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
            # Content hash, not just (mtime, size): a same-size edit whose mtime
            # is unchanged (coarse-mtime FS, cp -p / rsync --times, git restore,
            # restore-from-backup, mtime-preserving editors) would otherwise be
            # silently skipped and the index left stale. Legacy entries lacking
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
            # Label the document's format heuristic-first (free); the LLM tie-break
            # is only consulted for an unknown extension whose structure is unclear
            # AND a chat model is loaded (classify_fn short-circuits otherwise).
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
                            # A NaN/inf component would silently drop this doc's
                            # chunks from every query (nan cosine !> 0). Store no
                            # vectors for it (lexical-only) and surface why, rather
                            # than persist a poison vector (AGENTS rule 5).
                            say(f"embeddings had non-finite (NaN/inf) values for "
                                f"{f.name} - indexing it lexical-only")
                        else:
                            new_dim = _first_dim(vecs)
                            # A different embedding dimensionality means a different
                            # model: refuse rather than store mixed-dim vectors that
                            # would silently mis-score every query (C3).
                            if (self._vec_dim is not None and new_dim is not None
                                    and new_dim != self._vec_dim):
                                # Persist every file this call already finished before
                                # raising: this exception intentionally HALTS the
                                # batch (the file(s) after this one, including this
                                # one, are not indexed), but a mid-batch model switch
                                # must not also silently discard the WORK ALREADY DONE
                                # on earlier files in the same call - add_paths()/
                                # add_uploads() only _save() once at the very end, so
                                # without this the caller's except ValueError (plug.py)
                                # would report just an error line while N-1 already-
                                # embedded files vanish with no trace (AGENTS rule 5).
                                self._save()
                                raise ValueError(self._dim_mismatch_message(new_dim))
                            vectors = vecs
                            if self._vec_dim is None and new_dim is not None:
                                self._vec_dim = new_dim

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
            # failed), so chunks and vectors are exactly as _load() read them. A
            # full _save() here would rewrite chunks.jsonl for no reason and
            # rewrite or DELETE vectors.json off in-memory state that no longer
            # reflects a real change - which is how a scheduled re-sync tick, whose
            # normal outcome is "nothing changed", used to erase a vectors.json
            # that _load() had deliberately kept as evidence of a degraded index.
            # Persist metadata only, and only when there is something to persist:
            # a newly recorded root, or a meta.json that _load() flagged corrupt
            # and rebuilt a docs map for (that self-heal happens on the next
            # metadata write). Mirrors the no-indexable-files early return above.
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
               prune_missing: bool = False) -> dict:
        """Bring the index back in line with the folders it was built from.

        Re-walks every persisted root through the ORDINARY incremental path
        (``add_paths``): a file ADDED to an indexed folder since the last run is
        picked up, a CHANGED file is re-indexed, an unchanged file is skipped by
        content hash. Individually indexed files are re-checked too. This is what
        a scheduled ``rag`` job calls; it is also ``localm rag resync``.

        DELETION SEMANTICS (deliberate, and the reason this is not just
        ``add_paths(roots)``). A document whose file has VANISHED is FLAGGED
        (``missing: True`` + ``missing_since``), not dropped: its chunks stay
        indexed and stay searchable. This mirrors how the model registry treats a
        model file that disappears (``model_manager/registry.py`` sync_models_dir:
        "a moved file, unplugged drive, sync hiccup is not silently forgotten") -
        an unattended job that ran while a network share was mounting must not be
        able to destroy an index. The flag CLEARS by itself when the file comes
        back. Actual deletion happens only when the caller passes
        ``prune_missing=True``.

        Two guards keep a transient condition from being read as deletion at all:
        a root that is not currently an available directory (deleted, unmounted,
        unreadable, or replaced by a file) is REPORTED and skipped whole, and
        every document underneath it is left completely untouched - not indexed,
        not flagged, not pruned. The same holds for a root the current
        ``policy`` refuses.

        *policy* is applied exactly as in ``add_paths``: a scheduled re-sync must
        never index a path an interactive add would refuse, including a root that
        was legal when it was added but is outside the owner's allowed folders
        now. Callers that run unattended (the jobs runner) always pass one.

        Returns the ``add_paths`` counters plus ``missing`` (newly flagged),
        ``missing_total``, ``restored``, ``pruned``, ``roots``,
        ``unavailable_roots`` and ``blocked_roots`` (each ``{root, reason}``), and
        ``vector_degrade_reason`` (why semantic search is degraded after this run,
        None when it is fine), so the caller can report honestly what the run did
        and did NOT do, and over what state.
        """
        with _collection_lock(self.name), self._write_lock("a re-sync", on_progress):
            self._load()
            return self._resync_locked(
                embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force, prune_missing=prune_missing)

    def _resync_locked(self, *, embed_fn, classify_fn, describe_image_fn,
                       on_progress, policy, force, prune_missing) -> dict:
        """The resync body. MUST run under _collection_lock after _load()."""
        say = on_progress or (lambda _t: None)
        available, unavailable, blocked = self._partition_roots(policy, say)
        # Roots we could not judge. Nothing under them is indexed, flagged, or
        # pruned this run - that is the transient-condition guard.
        skipped_roots = [Path(r["root"]) for r in (unavailable + blocked)]

        targets: list = list(available)
        targets.extend(self._resyncable_files(skipped_roots, policy, say))

        # Snapshot the flagged-missing set BEFORE indexing. Re-indexing a
        # document REPLACES its docs entry wholesale (_add_paths_locked), which
        # drops the flag as a side effect - so a file that came back CHANGED
        # would be silently un-flagged and never reported as restored. The
        # snapshot is what makes "this came back" observable either way.
        docs_before = self._meta.get("docs", {})
        was_missing = {k for k, e in docs_before.items()
                       if isinstance(e, dict) and e.get("missing")}

        if targets:
            result = self._add_paths_locked(
                targets, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force)
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
            # fine). _load() only logs it, and a scheduled job's result is the
            # single place anyone looks at an unattended run - so a corrupt or
            # stale vectors.json has to be reported here too, or the job reads as
            # a clean success over a knowingly broken index (AGENTS rule 5).
            "vector_degrade_reason": self.vector_degrade_reason,
        })
        return result

    def _partition_roots(self, policy: Optional[dict], say: ProgressFn):
        """Split the persisted roots into (available, unavailable, blocked).

        Availability is checked FIRST and reported, never assumed: an
        unreachable root is the single most likely reason a re-sync would
        otherwise conclude that every file under it was deleted. ``is_dir()``
        answers most of that, but not all of it - see ``_unmounted_reason`` for
        the case it cannot see."""
        available: list = []
        unavailable: list = []
        blocked: list = []
        for raw in self.roots():
            root = Path(raw)
            if not root.is_dir():
                # is_dir() is False for gone, unreadable, AND replaced-by-a-file.
                # They differ in cause but not in the safe response (skip whole,
                # touch nothing), so branch only to report the right reason.
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
        behind as an ordinary, existing, EMPTY directory. The root then passes the
        availability check above, every document under it fails ``p.exists()`` in
        the missing pass, and an explicit ``resync --prune-missing`` run during
        the unmount window deletes the entire index for that folder - the exact
        outcome the "an unplugged drive cannot destroy the index" promise rules
        out. (The scheduled path never prunes, so only a hand-run prune could
        reach it.)

        All three conditions are required and none is sufficient alone.
        ``os.path.ismount`` by itself would skip a volume that is mounted and was
        legitimately indexed (a NAS share, a second drive); "empty" by itself
        would break the user who really did empty an indexed folder and wants
        --prune-missing to act on it. A mount point that is empty WHILE we hold
        documents indexed under it is the specific shape of a drive that went
        away, and skipping it is recoverable (remove the entries with
        ``localm rag rm`` if the folder really is empty for good), where pruning
        a mounted-away drive is not.
        """
        try:
            if not os.path.ismount(root):
                return None
            if next(root.iterdir(), None) is not None:
                return None
        except OSError:
            # We could not even look inside. Same reasoning as is_dir() being
            # False: an unanswerable question is never resolved destructively.
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
        anything under a skipped root, and anything the current policy refuses -
        the latter must be filtered HERE rather than handed to
        ``_add_paths_locked``, whose top-level confinement check raises and would
        abort the entire re-sync over one now-out-of-bounds file."""
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
        anything: a document that came back CHANGED has already been re-indexed
        (which rewrites its entry and drops the flag), so the live entry can no
        longer tell us it was ever missing.

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
                # We could not even ask. Treat it as present: the whole point of
                # this pass is that an unanswerable question must never be
                # resolved in the destructive direction.
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
                    force: bool = False) -> dict:
        """Index documents UPLOADED from the caller's own device (the per-device
        path for a client that cannot browse the server disk).

        Each item is ``{"filename": str, "data": bytes}``. Extraction runs in
        memory (``extract_bytes`` - same zip-bomb / encoding guards as chat
        attachments); the resulting chunks and optional vectors ARE persisted,
        because a knowledge base is explicit user data like ``add_paths``. There is
        deliberately NO filesystem confinement here: nothing on the server disk is
        read, so the whitelist/blacklist policy does not apply - the bytes are the
        caller's own device content. Docs are keyed ``upload:<filename>`` and
        deduped by content hash, so re-uploading an unchanged file is skipped.
        Returns the same counters as ``add_paths``.

        The uploaded BYTES are not retained (only the extracted chunks/vectors), so
        an ``upload:<name>`` doc cannot be re-read from disk: ``localm rag repair``
        simply skips these keys (Path('upload:x') is not a file) - their chunks
        persist and are never lost, they just cannot be re-embedded from source.
        """
        with _collection_lock(self.name), self._write_lock("an upload", on_progress):
            self._load()
            return self._add_uploads_locked(
                uploads, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, force=force)

    def _add_uploads_locked(self, uploads: list, *,
                            embed_fn: Optional[EmbedFn] = None,
                            classify_fn: Optional[ClassifyFn] = None,
                            describe_image_fn: Optional[DescribeImageFn] = None,
                            on_progress: Optional[ProgressFn] = None,
                            force: bool = False) -> dict:
        """The add_uploads body. MUST run under _collection_lock after _load().

        Mirrors the per-document body of _add_paths_locked (chunk -> embed ->
        replace-prior-chunks -> record meta), but sourced from in-memory bytes with
        a hash-only dedup (no fs mtime/size). Kept as a separate loop rather than
        refactoring the battle-tested path loop, so the server-disk path is
        untouched."""
        say = on_progress or (lambda _t: None)
        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None

        for up in uploads:
            filename = (str(up.get("filename") or "").strip() or "upload")
            data = up.get("data") or b""
            key = f"upload:{filename}"
            digest = hashlib.sha256(data).hexdigest()
            known = self._meta["docs"].get(key)
            if not force and known and known.get("hash") == digest:
                skipped += 1
                continue
            try:
                text = extract_bytes(data, filename, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {filename}: {e}")
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
                            # See add_paths: a NaN/inf component silently drops this
                            # doc from every query, so index it lexical-only and say why.
                            say(f"embeddings had non-finite (NaN/inf) values for "
                                f"{filename} - indexing it lexical-only")
                        else:
                            new_dim = _first_dim(vecs)
                            if (self._vec_dim is not None and new_dim is not None
                                    and new_dim != self._vec_dim):
                                # See add_paths: persist this call's already-completed
                                # uploads before halting, so a mid-batch model switch
                                # does not silently discard their work (AGENTS rule 5).
                                self._save()
                                raise ValueError(self._dim_mismatch_message(new_dim))
                            vectors = vecs
                            if self._vec_dim is None and new_dim is not None:
                                self._vec_dim = new_dim

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
            say(f"indexed {filename} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def _dim_mismatch_message(self, new_dim: int) -> str:
        """The refusal a user actually sees when they change embedding model.

        It used to say "Rebuild it (delete and re-add)" - telling someone to DELETE
        their collection, while naming neither the collection nor a command that
        works. Worse, the two remedies it implied both failed: `rag repair --embed`
        and the GUI reindex button hit this very guard, because neither reset
        _vec_dim. Now it names the collection, both models where known, and the one
        command that does the job without touching the source files.
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

        This is the answer to "I changed the embedding model, now my collection
        refuses everything". The chunk text is already on disk in chunks.jsonl, so
        nothing needs re-reading, re-chunking, or even to still exist: a collection
        whose sources moved, were deleted, or arrived as uploads re-embeds exactly
        the same as one whose files are all present. That is the difference from
        ``rag repair --embed``, which re-indexes FROM THE ORIGINAL SOURCE FILES and
        therefore cannot help when they are gone - and which could not help anyway,
        because it never reset ``_vec_dim`` and so tripped the very dimension guard
        it was supposed to resolve.

        Crash-safe by construction: every vector is computed into a LOCAL list
        first, and ``self._vectors`` is only replaced once the whole set is in hand
        and validated. A failing embedder (model unloaded, VRAM gone, a bad batch)
        therefore leaves the previous index exactly as it was, rather than a
        half-dimension index that would mis-score every later query. The cost is
        holding one full vector set in memory during the run, which is the same
        order as the file it is about to write.
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
                    on_progress(f"re-embedding {min(i + batch, total)}/{total}")

            # Validate BEFORE touching the live index. A short or ragged result is
            # the failure mode that produces a silently mis-scoring collection, so
            # it is refused loudly here rather than saved and discovered at query
            # time (AGENTS rule 5).
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
            # Record the model NAME, not only its dimension, so a later mismatch
            # can say which model built this index instead of inferring it from a
            # dimension count.
            if model_name:
                self._meta["embedding_model"] = str(model_name)
            self._meta["embedding_dim"] = self._vec_dim
            # A rebuilt full-coverage index makes any set-aside sidecar moot: this
            # IS the rebuild those files were kept as evidence for.
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
        # Same per-collection lock + re-sync as add_paths so a concurrent add and
        # remove on one collection cannot lose each other's write (CHK-RAG-LOCK),
        # and the same cross-process lock: removing a document while another
        # process re-indexes the collection would otherwise put it straight back.
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
            # Filter English stopwords from the lexical index so a query and a
            # chunk that overlap ONLY on a stopword (e.g. "and") cannot let that
            # chunk win the BM25 half and, via the 50/50 blend, outrank the true
            # semantic match - a real failure mode on small/narrow home-scale
            # corpora, where a stopword can earn a spuriously high IDF.
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

        We do not hide problems (AGENTS rule 5): a corrupt, stale, or dimensionally
        mismatched vectors index must not silently vanish into BM25-only. Recording
        the reason (exposed via ``stats()``) and logging genuine corruption once is
        the right altitude: the lexical fallback still works, but the fault stays
        discoverable.

        "Once" has to mean once per PROCESS, not once per instance. This method was
        already idempotent against ``self.vector_degrade_reason``, but ``_load``
        runs from ``__init__`` and every request builds a FRESH Collection, so the
        guard reset on each one and the same sentence was logged on essentially
        every /api/rag call - measured at 25+ identical WARNING lines in one
        session, several twice within a single request. The instance field is still
        set unconditionally, so ``stats()`` and the GUI's "needs repair" state stay
        exactly as accurate as before; only the duplicate LOG LINE is suppressed.
        A genuinely NEW reason for the same collection still warns, so a fault that
        changes shape is never hidden behind an earlier one."""
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
            # Partial coverage is expected while a collection is still embedding,
            # so record it (visible in stats) but do not warn - not a corruption.
            self._note_vector_degrade(
                "vector coverage below 80% (index not fully embedded); "
                "using BM25 lexical retrieval only", warn=False)
            return None
        # Stored vectors must share one dimensionality. A legacy collection with
        # mixed-dim vectors (built before the C3 add-time guard) is ambiguous -
        # skip vector scoring and answer lexically rather than mis-score with
        # zeros for the odd-dim chunks.
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
            # not an expected transient like partial coverage - surface it. The
            # note is idempotent, so this warns once per distinct error, not per query.
            self._note_vector_degrade(
                f"query embedding failed ({type(e).__name__}); "
                f"using BM25 lexical retrieval only", warn=True)
            return None
        # A switched embedding model yields query vectors of a different
        # dimensionality than the stored ones: fall back to lexical-only rather
        # than crash or return wrong scores.
        if len(qvec) != stored_dim:
            self._note_vector_degrade(
                f"embedding model changed (query dim {len(qvec)} != stored "
                f"{stored_dim}); using BM25 lexical retrieval only - rebuild the "
                f"collection to restore semantic search", warn=True)
            return None
        if not _vectors_finite([qvec]):
            # A non-finite query embedding would make every cosine nan and empty
            # the result set (nan !> 0), defeating the BM25-always promise.
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

    def stats(self) -> dict:
        present = [v for v in (self._vectors or []) if v]
        docs = self._meta.get("docs", {})
        return {
            "name": self.name,
            "created": self._meta.get("created"),
            "n_docs": len(docs),
            # Indexed documents whose source file was gone at the last resync.
            # They are still counted in n_docs and still searchable - the flag
            # says the index is ahead of the disk, it does not remove anything
            # (see resync's deletion semantics). Reported so a stale index is
            # visible instead of quietly drifting (AGENTS rule 5).
            "n_missing": sum(1 for e in docs.values()
                             if isinstance(e, dict) and e.get("missing")),
            "n_roots": len(self.roots()),
            "n_chunks": len(self._chunks),
            # "has vectors" = whether query() will actually blend embeddings: the
            # same >=80% coverage threshold _vector_scores uses, NOT "every chunk
            # embedded". A partially-embedded collection (80-99%) still does hybrid
            # retrieval, so the retrieval-mode label must not under-report it (rule 5).
            "has_vectors": bool(present) and len(present) >= 0.8 * len(self._chunks),
            "corrupt": self.corrupt,
            # Why semantic search fell back to BM25 (None when vectors are used or
            # legitimately absent); surfaced instead of silently swallowed.
            "vector_degrade_reason": self.vector_degrade_reason,
        }

    def docs(self) -> list[dict]:
        return [
            {"path": path, **info}
            for path, info in sorted(self._meta.get("docs", {}).items())
        ]


def _cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        # Mismatched dims must never be scored as a real (zero) similarity - that
        # silently mis-ranks. Callers (_vector_scores) guarantee equal lengths;
        # reaching here is a bug or unrepaired mixed-dim data (C3).
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
        # ImportError is the ordinary "numpy not installed" case. AttributeError is
        # the one that broke Windows CI across several lanes: numpy is PARTIALLY
        # INITIALISED on those runners, so `import numpy` SUCCEEDS while np.asarray
        # is absent - the fallback below was never reached and the AttributeError
        # escaped into every caller of a vector query.
        #
        # Deliberately NOT a bare `except`: that would also swallow a genuinely
        # broken numpy and silently halve retrieval quality forever with nobody
        # the wiser (AGENTS rule 5). These two are the exact shapes of "numpy is
        # unusable HERE", and a usable-numpy failure (a real numerical error) still
        # propagates. And the degrade is ANNOUNCED once per process rather than
        # taken silently, because "semantic search got slower and nobody knows why"
        # is exactly the invisible fault that rule exists to prevent.
        if not _NUMPY_DEGRADE_LOGGED:
            _NUMPY_DEGRADE_LOGGED.add(True)
            _log.warning(
                "numpy is present but unusable (%s: %s); falling back to pure-Python "
                "cosine similarity. Results are identical, scoring is slower on large "
                "collections. This usually means a partially-initialised or broken "
                "numpy install - reinstall it to restore the fast path.",
                type(e).__name__, e)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        sim = dot / (na * nb) if na and nb else 0.0
    # A NaN/inf component (corrupt/degenerate vector) makes the similarity
    # non-finite; nan silently drops the chunk (nan !> 0) and inf mis-ranks it to
    # the top. Treat non-finite as a miss (0.0), never let it leave this function.
    return sim if math.isfinite(sim) else 0.0
