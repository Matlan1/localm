# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent document collections ('knowledge bases')."""

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

# numpy is optional here: it is NOT in pyproject.toml, and CI installs from
# pyproject rather than uv.lock, so on those runners it is simply absent and the
# normal, correct state is a clean ModuleNotFoundError into the pure-Python path.
# Bound ONCE at module import so this is decided at a single known point instead
# of per call; absent -> None -> every caller degrades.
#
# THE REAL ROBUSTNESS IS THE (ImportError, AttributeError) CATCH AT EACH CALL
# SITE, not this binding. An attribute-less `numpy` can appear with no import
# error at all: a bare DIRECTORY named `numpy` on sys.path resolves as a PEP 420
# implicit namespace package, so `import numpy` SUCCEEDS and yields a module with
# no attributes and ``__file__`` of None. An injected stub module looks the same.
# That is the state CI hit, reproduced with a control (no stray dir ->
# ModuleNotFoundError; stray dir -> character-identical AttributeError).
#
# CORRECTION, kept deliberately because this comment previously asserted it and a
# reader would otherwise copy it: an earlier diagnosis blamed a per-module
# import-lock race across the plugin thread pool. That is FALSE and was falsified
# from the CI logs - numpy is not installed there at all (you cannot
# half-initialise a package that is not installed), the failing traceback has no
# thread pool on it (memory/store.py, a plain synchronous call), and every failure
# landed on gw0 and only gw0, which a first-import race cannot produce. Two CPython
# details that diagnosis got backwards: a failed import does NOT leave a partial
# module behind (``_bootstrap._load`` deletes ``sys.modules[name]`` on any
# BaseException), and plain ``import x`` BLOCKS on the module lock rather than
# handing a second thread a partial module - it accepts a partial one only on
# _DeadlockError, i.e. a genuine circular import.
#
# So this binding NARROWS a lazy-import window that is a latent hazard regardless;
# it does not close the hole that actually bit us, because it binds a namespace
# stub just as readily as a lazy import would. What creates that stray directory
# is an ENVIRONMENT defect, under separate investigation, and is not claimed here.
try:
    import numpy as _numpy
except ImportError:      # optional dependency - every caller degrades to pure Python
    _numpy = None

#: True when numpy imported but is a namespace stub / attribute-less object rather
#: than a real install. ``__file__`` is None for a PEP 420 namespace package, which
#: is an EXACT discriminator - so the fallback can say WHICH case it hit instead of
#: collapsing "numpy is legitimately absent" (routine) and "something has put a
#: fake numpy on your path" (the install is broken) into one benign-sounding
#: message. Same rule-5 branch as missing-vs-corrupt, applied to the message.
_NUMPY_IS_STUB = _numpy is not None and getattr(_numpy, "__file__", None) is None


def _warn_numpy_degrade(exc: Exception, operation: str) -> None:
    """Announce the pure-Python fallback ONCE per process, saying which case it is."""
    if _NUMPY_DEGRADE_LOGGED:
        return
    _NUMPY_DEGRADE_LOGGED.add(True)
    if _numpy is None:
        # ABSENT, which is the ORDINARY case and not a fault: numpy is not a
        # declared dependency (it appears nowhere in pyproject.toml), so a default
        # install simply does not have it and the pure-Python path is the intended
        # behaviour. Debug, never a warning.
        #
        # This branch is the point of this function. Without it, absence fell into
        # the "present but unusable" arm below - because the callers signal it by
        # raising ImportError("numpy is not installed"), which that arm then
        # reported as a broken install. Every numpy-less user got
        #
        #     numpy is present but unusable (ImportError: numpy is not installed)
        #
        # on their first query: a message that contradicts itself in one line, and
        # sends whoever debugs it hunting a machine fault that does not exist.
        #
        # Note WHERE the test is. It branches on the MODULE STATE, never on the
        # exception's text, so no reworded sentinel can silently re-collapse the
        # cases. That is the same reason _NUMPY_IS_STUB keys on ``__file__``.
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
    """Cheap (O(n)) structural check that *vectors* is what ``_save`` writes: a list whose entries are each a null placeholder (a missing embedding) or a list/tuple."""
    return isinstance(vectors, list) and all(
        (not v) or isinstance(v, (list, tuple)) for v in vectors)


def _vectors_finite(vectors) -> bool:
    """True when every component of every present vector is a FINITE number."""
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
# Called with a human-readable message as the sole positional argument. The one
# call site with an exact numerator/denominator (reembed's batch loop)
# additionally passes phase/done/total/unit as keywords, carrying the same
# numbers the message was built from - so a job-aware sink can forward them to
# Job.progress verbatim instead of re-deriving them from the string. Any sink
# reused for reembed (currently _job_progress and the CLI's reembed callback)
# must accept and ignore these keywords (**_); every other on_progress use
# (add_paths, add_uploads, resync, _write_lock) never passes them.
ProgressFn = Callable[..., None]


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
    """Yield files under *root* without following linked DIRECTORIES, bounded by depth and a visited-realpath set."""
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
    """A path may not be indexed. ``reason`` tells the caller WHY, so the API can offer 'add and continue' for a fixable whitelist miss (``outside_allowed``) but hard-refuse the rest (``credential`` / ``secret_file`` / ``denied`` / ``invalid`` / ``unc_or_device``)."""

    def __init__(self, message: str, *, path: Path, reason: str):
        super().__init__(message)
        self.path = path
        self.reason = reason


_INDEX_MODES = ("whitelist", "blacklist")


def indexing_policy(cfg: Optional[dict] = None,
                    key_roots: Optional[list] = None) -> dict:
    """The current RAG indexing confinement policy, read from config."""
    # Loaded ONCE, before the key_roots branch too: allow_network_drives is a
    # WHOLE-MACHINE preference (see the comment on the return below), not
    # part of the per-key folder scoping key_roots exists for, so a key-scoped
    # caller must still see the owner's real setting rather than silently
    # defaulting to "allowed" because this branch never looked.
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
            "denied": _resolve("rag_denied_roots"),
            # NOT a hard-floor-fail-closed key like mode/allowed/denied above:
            # allow_network_drives is a preference, not a security boundary
            # (see config.py's DEFAULT_CONFIG comment), so a config we could
            # not load resolves it to the same True default as a normal read,
            # not to False. confine_index_path applies it regardless of mode.
            "allow_network_drives": bool(cfg.get("allow_network_drives", True))}


def _network_drives_allowed_fresh() -> bool:
    """One-off config read for confine_index_path's ``policy=None`` callers (settings_schema.py's PATHLIST save-time validation, and the bare CLI), which have no ``indexing_policy()`` dict to read the value off."""
    try:
        from localm.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    return bool(cfg.get("allow_network_drives", True))


def confine_index_path(p, policy: Optional[dict] = None) -> Path:
    """Resolve *p* and verify it may be indexed, raising ``ConfinementError`` (a ``ValueError``) otherwise."""
    try:
        rp = Path(p).expanduser()
    except (OSError, ValueError):
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid")
    # `p` is HTTP-API-reachable (the whitelist/blacklist policy branches below
    # are only meaningful for that caller; CLI/policy=None call sites are the
    # local operator). Refuse UNC/device syntax unconditionally, BEFORE the
    # .resolve() below - the actual filesystem syscall - ever runs: a UNC
    # target dials SMB and auto-authenticates before any of the checks below
    # get a chance to refuse it. Checked on the EXPANDED string (expanduser()
    # is pure string/env-var work, no syscall, so it is safe to call first) -
    # not the raw one, so a `~` whose configured home is itself a UNC path
    # cannot slip past a pre-expansion check. Raised OUTSIDE the try/except
    # above: ConfinementError is a ValueError, so raising it INSIDE that
    # block would be caught by the same `except (OSError, ValueError)` and
    # relabelled "invalid", losing this reason.
    if is_unc_or_device_path(str(rp)):
        raise ConfinementError(f"Refusing to index a UNC or device path: {p}",
                               path=Path(str(p)), reason="unc_or_device")
    try:
        rp = rp.resolve()
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

    # allow_network_drives is a PREFERENCE, not part of the SMB-dial-safety
    # hard floor above (a mapped drive is already connected, not a fresh UNC
    # dial - see pathsafe.is_mapped_network_drive's docstring), but it is
    # still checked here unconditionally, BEFORE the policy=None return below:
    # it is a whole-machine setting, not a per-caller policy like
    # whitelist/blacklist mode, so turning it off means "never index a
    # network share" for the CLI operator too, not only the HTTP API.
    # *policy* already carries the resolved value when given - indexing_policy()
    # reads it once per request/resync, so reading it off *policy* here avoids
    # a config load per file in the hot indexing-walk loops (_add_paths_locked,
    # resync). `.get(...)`, not `[...]`: several tests build a minimal policy
    # dict by hand without this key, and it must default exactly like a fresh
    # config read would (True). With policy=None (settings_schema.py's own
    # save-time validation call, and the bare CLI) it is read fresh here
    # instead - a rare, non-hot-loop call, per the grep at the time this was
    # added: every hot-loop confine_index_path call already guards on
    # `policy is not None` before it is ever reached.
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
    # The recursive folder walk (_expand) already skips these by suffix + secret
    # name, but an EXPLICITLY-named top-level file used to bypass that filter, so a
    # `rag`-scoped HTTP caller could POST paths=["<home>/deploy.pem"] and read the
    # key back via /query (C2). Apply the SAME filter to explicit picks whenever a
    # policy is present (every API caller, owner and non-owner alike - a loopback
    # page or remote client is untrusted). This runs BEFORE the mode branches so a
    # secret is never offered through the whitelist "add and continue" consent flow.
    # Guarded on "not a directory" rather than is_file() so a directory merely
    # NAMED like a secret (a real ./credentials or ./.env folder) is still
    # walkable, not over-blocked, while a path that does NOT EXIST is still
    # refused here. That difference matters: with is_file(), an existing
    # secret-named path raised secret_file (400) and a missing one fell through
    # to the whitelist branch (409/403), so the two answers differed and the
    # response was still an existence oracle for exactly the interesting targets
    # (a .pem, a .key, an id_rsa). Both now get the same refusal. Compare
    # _SENSITIVE_NAMES above, which already never consulted the filesystem.
    # The CLI (policy=None, returned above) stays unconfined: the local operator
    # can already read their own files, so an explicit single-file pick is still
    # honoured there.
    if not rp.is_dir() and (rp.suffix.lower() in SECRET_SUFFIXES
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
    #
    # UNLESS this is a KEY-SCOPED policy (indexing_policy(key_roots=...)) - a
    # per-key allowlist exists specifically to confine a credential to LESS than
    # the owner's own reach, so home/cwd/the global rag_allowed_roots are
    # deliberately NOT implied on top of it; only the key's own explicit roots
    # count. The hard floor above (credential dirs, secret files, UNC/device
    # paths) still applies either way - that guard runs before this branch and
    # cannot be widened by any policy.
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
    """Delete a collection, waiting for any in-flight write to finish first."""
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

#: Warn-once keys already logged in THIS process - not just vector degrades
#: (_note_vector_degrade) despite the name, also the chunks.jsonl malformed-line
#: warning in _load() below. _load() runs from __init__ and every request builds
#: a fresh Collection, so an instance-level guard re-armed constantly and the
#: same warning was emitted 25+ times a session. Process-scoped so the first
#: occurrence is still loud and the rest are quiet. Never consulted for state -
#: only for whether to LOG. Every key starts with the collection dir so two
#: DIFFERENT collections' warnings never collapse into one; a distinguishing
#: tag (a literal string, or the reason text) as the second element keeps this
#: shared set's entries from colliding across unrelated warning sites.
_WARNED_DEGRADES: set = set()

#: meta.json key for the derived-stats cache _save() writes and peek_stats() /
#: peek_detail() read - see the block comment at the end of _save(). Leading
#: underscore marks it as internal/derived (like _REJECTED_VECTORS), never
#: user data, so a hand-edited meta.json without it is simply "no cache yet".
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
        # How many lines of chunks.jsonl _load() had to skip as unparseable /
        # wrong-shape (NEW-RAG-INDEX-WARN-SPAM residual B). 0 whenever the file
        # is clean or absent - distinct from self.corrupt, which also covers a
        # bad meta.json or roots map and carries no count of its own. Exposed
        # via stats() so a caller can say "62 malformed chunk lines" instead of
        # a generic "index damaged".
        self.chunks_bad_lines: int = 0
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
        """The CROSS-PROCESS write lock for this collection."""
        return collection_write_lock(
            lock_path_for(self.dir), collection=self.name, op=op,
            on_wait=on_progress)

    def create(self) -> "Collection":
        """Create the collection if it does not exist yet."""
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
            self.chunks_bad_lines = bad_lines
            if bad_lines:
                self.corrupt = True
                # Warn-once, same pattern and same process-scoped set as
                # _note_vector_degrade below: _load() runs from __init__, and
                # every /api/rag request builds a FRESH Collection, so an
                # instance-level guard re-arms on every single request - this
                # fired on essentially every call for a collection with any
                # corrupt lines at all. self.corrupt is still set
                # unconditionally above (stats()/the GUI's "needs repair"
                # state stay exactly as accurate as before); only the
                # duplicate LOG LINE is suppressed. Keyed on the bad_lines
                # COUNT (not just the dir), so a fault that changes shape -
                # more lines corrupt after further damage, or fewer after a
                # partial repair - still warns again instead of hiding
                # behind an earlier, now-stale count.
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
        # This call just rewrote BOTH meta.json (above, from self._meta) and
        # chunks.jsonl (from self._chunks) verbatim from this instance's own
        # in-memory state, which _load()'s three self.corrupt sites already
        # require to be well-formed before it is ever placed in self._meta /
        # self._chunks: a corrupt meta.json is replaced with a clean
        # {name, docs} dict (and 'roots' is reset the same way in
        # _record_roots) before it can be re-written, and a malformed
        # chunks.jsonl line is dropped, never appended to self._chunks, so
        # dumps_lines(self._chunks) can never reproduce one. So whatever
        # on-disk corruption made THIS load flag self.corrupt / count
        # self.chunks_bad_lines is resolved by construction the moment this
        # write lands - clear both here, not just where _load() sets them.
        # Without this, a repair (add_paths(force=True) -> this _save()) left
        # the CACHED corrupt/chunks_bad_lines stale at their pre-repair values
        # forever: peek_stats()'s fingerprint check still matches (it is taken
        # from what THIS call just wrote), so the GUI's "needs repair" badge
        # and the CLI's corrupt marker never cleared even though the fault was
        # actually fixed (NEW-RAG-INDEX-WARN-SPAM, caught in manual GUI
        # verification - click Repair, and the badge did not go away).
        self.corrupt = False
        self.chunks_bad_lines = 0
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
        # Cache the LISTING-relevant fields that are NOT otherwise persisted -
        # vector_degrade_reason and the vector-coverage math above live only on
        # this Python object - so a listing can answer from meta.json alone,
        # without reconstructing this Collection at all (see peek_stats() /
        # peek_detail() below and rag/plug.py's rag_collections/rag_detail,
        # which used to pay a full _load() - parsing every chunk AND every
        # embedding vector - just to report counts; measured at 2.7-3.8s on the
        # single-worker event loop, freezing every other in-flight request for
        # that long).
        #
        # Computed from the SAME state stats() itself would read, and written
        # right after that state was saved to disk, so it cannot drift from
        # what a fresh stats() would say right now via THIS class. It CAN go
        # stale if something else edits chunks.jsonl or vectors.json without
        # going through this class - and that is not just a hypothetical
        # "external tampering" case: a crash mid-write between this method's
        # OWN chunks.jsonl and vectors.json writes above is exactly the shape
        # test_rag_degraded_vectors_preserved.py already simulates (a
        # vectors.json that no longer matches the chunks it should describe),
        # and _load() is trusted to catch that on the NEXT load. A cache that
        # ignored this would silently keep reporting the collection healthy
        # in a LISTING while a real open (query/detail without the cache, or
        # any write) correctly reports it degraded - precisely the "answers
        # wrong instead of not at all" failure this whole design exists to
        # avoid, and it was caught exactly this way in review (see
        # tests/test_rag_peek_stats.py, which hand-edits vectors.json after a
        # save and asserts the cache is NOT trusted).
        #
        # The fix is a cheap (mtime_ns, size) fingerprint of both files, taken
        # AFTER they were written above so it reflects what this save just
        # put on disk. peek_stats()/peek_detail() stat() (never read) both
        # files again and refuse the cache the instant either one no longer
        # matches - two os.stat() calls, not a re-parse, so this stays cheap.
        # peek_stats()/peek_detail() also fall back to a full load whenever
        # this cache is missing entirely (an old collection never resaved
        # under this code, e.g.), so a caller always gets a correct answer -
        # just not always a free one.
        #
        # A second, small atomic write rather than folding this into the
        # meta.json write at the top of this method: that write happens BEFORE
        # vector_degrade_reason is finalised above, and reordering this
        # carefully-sequenced method (the vectors_file_rejected quarantine
        # above is deliberately decided before anything writes or unlinks that
        # filename) is a bigger risk than one extra write on what is already an
        # infrequent, multi-write path.
        self._meta[_STATS_CACHE_KEY] = self._stats_cache_block()
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))

    def _stats_cache_block(self) -> dict:
        """The ``_stats_cache`` block for meta.json, computed from THIS instance's current in-memory state and a FRESH fingerprint of chunks.jsonl/vectors.json taken right now (see ``_file_fingerprint``)."""
        return {
            "n_chunks": len(self._chunks),
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "vector_degrade_reason": self.vector_degrade_reason,
            "corrupt": self.corrupt,
            "chunks_bad_lines": self.chunks_bad_lines,
            "fingerprint": self._file_fingerprint(),
            # Cheap to include (already in memory - see vector_dim()) and lets
            # a listing-time caller compare against the currently active
            # embedding model WITHOUT loading anything (rag_collections()
            # does this against embedder.loaded_dim()) - closing the gap
            # named in that route's own docstring for the common case where
            # an embedder happens to already be resident.
            "vector_dim": self._vec_dim,
        }

    def _vector_index_complete(self) -> bool:
        """True when every chunk currently has a usable vector."""
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
        """Delete set-aside sidecars, saying why."""
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
        """Set a rejected vectors.json aside as ``vectors.json.rejected``."""
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
        """Keep only the newest ``_MAX_REJECTED_KEPT`` set-aside indexes."""
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
        """An unused ``vectors.json.rejected[.N]`` path, or None if there is none."""
        first = self.dir / _REJECTED_VECTORS
        if not first.exists():
            return first
        for n in range(2, 21):
            candidate = self.dir / f"{_REJECTED_VECTORS}.{n}"
            if not candidate.exists():
                return candidate
        return None

    def _save_meta(self) -> None:
        """Persist meta.json ONLY, leaving chunks.jsonl / vectors.json alone."""
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
        """Resolve files + recursive folder contents to indexable files."""
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
        """Persist the FOLDER roots among *paths*."""
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
        """The folders indexed into this collection, resolved and sorted."""
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
        """Index files/folders."""
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
                on_progress=on_progress, policy=policy, force=force,
                model_name=model_name)

    def _add_paths_locked(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                          classify_fn: Optional[ClassifyFn] = None,
                          describe_image_fn: Optional[DescribeImageFn] = None,
                          on_progress: Optional[ProgressFn] = None,
                          policy: Optional[dict] = None,
                          force: bool = False,
                          model_name: Optional[str] = None) -> dict:
        """The add_paths read-modify-write body."""
        say = on_progress or (lambda _t: None)
        if policy is not None:
            # ACT ON WHAT WAS CHECKED. confine_index_path returns the RESOLVED
            # path it validated; calling it for its exception alone and then
            # walking the caller's original string is the decide-on-one/act-on-
            # the-other shape registry.py documents as an escape at its own
            # remove_model gate.
            #
            # HONEST SCOPE: this is defence in depth, NOT a live hole that was
            # open. _expand re-confines every file it emits, on that file's own
            # resolved path, so an out-of-policy file was never indexed even
            # before this line changed; and _walk_files skips reparse points, so
            # the obvious symlinked-root case was already covered. What it buys
            # is that the property is now local - the top-level walk root is the
            # value confinement returned, rather than a different value that a
            # second gate downstream happens to catch. Do not read it as a
            # vulnerability fix.
            paths = [confine_index_path(p, policy) for p in paths]  # raises ValueError
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
                                # Record which model built this index, same as
                                # reembed() - the ONLY other writer of this key -
                                # so a later dimension-mismatch message can name
                                # it instead of leaving the "built with" clause
                                # empty for every collection made the ordinary
                                # way (FIX4).
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
               prune_missing: bool = False,
               model_name: Optional[str] = None) -> dict:
        """Bring the index back in line with the folders it was built from."""
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
        """The resync body."""
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
            # fine). _load() only logs it, and a scheduled job's result is the
            # single place anyone looks at an unattended run - so a corrupt or
            # stale vectors.json has to be reported here too, or the job reads as
            # a clean success over a knowingly broken index (AGENTS rule 5).
            "vector_degrade_reason": self.vector_degrade_reason,
        })
        return result

    def _partition_roots(self, policy: Optional[dict], say: ProgressFn):
        """Split the persisted roots into (available, unavailable, blocked)."""
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
        """Why *root* looks like an UNMOUNTED mount point, or None if it is fine."""
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
        """True when at least one indexed document's source lives under *root*."""
        return any(
            not str(key).startswith("upload:") and _path_within(Path(key), root)
            for key in self._meta.get("docs", {})
        )

    def _resyncable_files(self, skipped_roots: list, policy: Optional[dict],
                          say: ProgressFn) -> list:
        """Existing document source files that are safe to re-index this run."""
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
        """Flag documents whose file has vanished, clear the flag on ones that came back, and prune only when explicitly asked."""
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
                    force: bool = False,
                    model_name: Optional[str] = None) -> dict:
        """Index documents UPLOADED from the caller's own device (the per-device path for a client that cannot browse the server disk)."""
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
        """The add_uploads body."""
        # The no-op accepts **_ because `_finished` below passes the structured
        # keywords ProgressFn now carries. A one-positional lambda would raise
        # TypeError on every caller that passes no callback at all, which is most
        # of them.
        say = on_progress or (lambda _t, **_: None)
        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None
        # Known before the first byte is extracted: the caller handed us the whole
        # list, already decoded, in memory.
        n_total = len(uploads)

        def _finished(n: int, text: str) -> None:
            """Report one item DONE, whatever its outcome."""
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
                # Previously silent. A skipped file was indistinguishable from
                # one that was never seen, which is its own small dishonesty.
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
                                # See add_paths: record the model that built this
                                # index, same key reembed() writes (FIX4).
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
        """The refusal a user actually sees when they change embedding model."""
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
        """Recompute EVERY vector from the stored chunk text, with a new model."""
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
        """Top-*k* chunks for *text*: max-normalised BM25, blended 50/50 with cosine similarity when vectors cover the corpus and the query can be embedded."""
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
        """Record WHY semantic (vector) scoring is unavailable and surface it once."""
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

    @staticmethod
    def _has_vectors(chunks: list, vectors: Optional[list]) -> bool:
        """'Has vectors' = whether query() will actually blend embeddings: the same >=80% coverage threshold _vector_scores uses, NOT 'every chunk embedded'."""
        present = [v for v in (vectors or []) if v]
        return bool(present) and len(present) >= 0.8 * len(chunks)

    def stats(self) -> dict:
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
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "corrupt": self.corrupt,
            # Count of chunks.jsonl lines _load() had to skip (0 if none) - lets
            # a caller say "62 malformed chunk lines" instead of a generic
            # "index damaged" (NEW-RAG-INDEX-WARN-SPAM residual B).
            "chunks_bad_lines": self.chunks_bad_lines,
            # Why semantic search fell back to BM25 (None when vectors are used or
            # legitimately absent); surfaced instead of silently swallowed.
            "vector_degrade_reason": self.vector_degrade_reason,
            "vector_dim": self._vec_dim,
        }

    def vector_dim(self) -> Optional[int]:
        """The dimensionality of THIS collection's currently stored vectors, or None when it cannot be established: no usable vectors are stored at all (see ``stats()['has_vectors']``), or ``_load()`` found the file present but unusable for a reason ``vector_degrade_reason`` names."""
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
        """(mtime_ns, size) for chunks.jsonl and vectors.json (None for either that does not exist) - a cheap (stat only, no content read) signature of the two files the ``_stats_cache`` block is derived from."""
        def _stat(name: str) -> "list[int] | None":
            try:
                st = (self.dir / name).stat()
            except OSError:
                return None
            return [st.st_mtime_ns, st.st_size]
        return {"chunks": _stat("chunks.jsonl"), "vectors": _stat("vectors.json")}

    @staticmethod
    def _fingerprint_matches(coll_dir: Path, recorded) -> bool:
        """True when *recorded* (the cache's 'fingerprint' value) still matches chunks.jsonl and vectors.json on disk RIGHT NOW."""
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
        """(checked name, collection dir, parsed meta.json), or None when there is nothing here the lazy path can trust enough to skip the full load: an invalid name, no meta.json, or one that fails to parse."""
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
        """stats()-shaped dict from an already-parsed meta.json, using ONLY the cache _save() writes (see its block comment) - never a re-derivation of has_vectors/vector_degrade_reason from a partial read of chunks.jsonl / vectors.json, which would be exactly the 'answers wrong instead of not at all' failure..."""
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
            # 0 on a cache written before this field existed - same
            # graceful-degrade as vector_dim below, not a migration. A stale
            # 0 alongside corrupt=True just means the caller falls back to
            # generic "index damaged" wording instead of naming a count.
            "chunks_bad_lines": cache.get("chunks_bad_lines", 0),
            "vector_degrade_reason": cache.get("vector_degrade_reason"),
            # Absent on a cache written before this field existed - the
            # collection falls back to the cold load-and-backfill path below
            # exactly once (same graceful-degrade this whole cache design
            # already relies on for any newly added field), not a migration.
            "vector_dim": cache.get("vector_dim"),
        }

    @classmethod
    def peek_stats(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``stats()`` without constructing a full ``Collection`` - reads meta.json alone and trusts its cached derived fields, never chunks.jsonl or vectors.json."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        return cls._stats_from_meta(checked_name, coll_dir, meta)

    @classmethod
    def peek_detail(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``peek_stats()`` plus the docs list, for the collection-detail route - both read meta.json exactly once."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        stats = cls._stats_from_meta(checked_name, coll_dir, meta)
        if stats is None:
            return None
        return {**stats, "docs": cls._docs_from_meta(meta)}

    @classmethod
    def load_and_maybe_backfill(cls, name: str, base: Optional[Path] = None
                                ) -> "Collection":
        """The COLD-fallback path for ``peek_stats()``/``peek_detail()``: a full, authoritative load of *name* (identical to plain ``Collection( name, base)``), with an opportunistic attempt to backfill its ``_stats_cache`` so future listings of this SAME collection stop paying the full-load cost - closing the..."""
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
            return cls(checked_name, base)       # busy: today's exact fallback


def collection_provenance_report() -> list:
    """Every collection that currently has vectors, with its recorded 'built with' model (``Collection.embedding_model()``, None if never recorded) and chunk count - the pre-switch, new-model-dimension-free report used by every writer of the ``embedding_model`` config key (the RAG picker's ``POST /api/rag/..."""
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
    """The human-readable note accompanying a ``collection_provenance_report()`` result, shared by every writer of ``embedding_model`` (the RAG picker, ``PATCH /v1/config``, ``localm setup-embeddings``) so the wording a user sees does not drift between which surface they switched from."""
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
        _warn_numpy_degrade(e, "cosine similarity")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        sim = dot / (na * nb) if na and nb else 0.0
    # A NaN/inf component (corrupt/degenerate vector) makes the similarity
    # non-finite; nan silently drops the chunk (nan !> 0) and inf mis-ranks it to
    # the top. Treat non-finite as a miss (0.0), never let it leave this function.
    return sim if math.isfinite(sim) else 0.0
