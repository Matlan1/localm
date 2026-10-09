# SPDX-License-Identifier: AGPL-3.0-or-later
"""Indexing confinement: which paths a collection may index."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from localm.pathsafe import is_mapped_network_drive, is_unc_or_device_path

from ..extract import SECRET_SUFFIXES, is_secret_index_name


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
    except (OSError, ValueError) as e:
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid") from e
    # Refuse UNC/device syntax unconditionally, BEFORE the .resolve() below ever
    # runs, and on the EXPANDED string. Raised OUTSIDE the try/except above.
    if is_unc_or_device_path(str(rp)):
        raise ConfinementError(f"Refusing to index a UNC or device path: {p}",
                               path=Path(str(p)), reason="unc_or_device")
    try:
        rp = rp.resolve()
    except (OSError, ValueError) as e:
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid") from e

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
