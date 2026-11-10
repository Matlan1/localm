# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which project tool hooks and MCP servers the user has trusted.

A project's ``.localcoder/config.toml`` can declare commands that run around
every tool call (hooks.py) and MCP servers the coder starts (mcp.py). Those
hooks run and those servers start only once the user has trusted them. Trust is
recorded per config file as a digest of the hooks and servers it declares
(commands_digest): editing, adding or removing a hook or a server changes the
digest, and the changed config needs trusting again.

The store is ``<data dir>/coder-hook-trust.json``, a JSON object mapping a
config file's normalised absolute path to ``{"sha256": <digest>,
"trusted_at": <unix time>}``. It is written only when the user asks for trust
to be remembered.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from localm.debuglog import logger

_FILENAME = "coder-hook-trust.json"

# Held, outside the store's cross-process lock, across every read-modify-write
# of the store. LOCK ORDER: _LOCK (outer) -> the store's .lock file.
_LOCK = threading.Lock()

# Unreadable-store messages already logged at WARNING since the store was last
# read successfully.
_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _store() -> Path:
    from localm.config import home_dir
    return home_dir() / _FILENAME


def config_key(config_path: Path) -> str:
    """The store key for *config_path*: its absolute path, case-folded where the
    filesystem is case-insensitive."""
    return os.path.normcase(str(Path(config_path).resolve()))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")


def hooks_digest(hooks_table: Any) -> str:
    """SHA-256 of the ``[hooks]`` table exactly as parsed, keys sorted, so any
    change to a command, matcher, timeout or the list of hooks changes it."""
    return hashlib.sha256(_canonical(hooks_table)).hexdigest()


_COMMANDS_PREFIX = b"localm-coder-commands-v1\0"


def commands_digest(raw: dict) -> str:
    """The digest trust is recorded against for a parsed project config *raw*:
    everything in it that runs commands, its ``[hooks]`` and ``[mcp]`` tables.

    Without an ``mcp`` key it is ``hooks_digest`` of the ``[hooks]`` table. With
    one it is the SHA-256 of a fixed prefix and both tables as parsed, so any
    change to an MCP server's command, args, env or flags changes it, and it
    never equals a hooks-only digest."""
    if "mcp" not in raw:
        return hooks_digest(raw.get("hooks"))
    tables = {"mcp": raw["mcp"]}
    if "hooks" in raw:
        tables["hooks"] = raw["hooks"]
    return hashlib.sha256(_COMMANDS_PREFIX + _canonical(tables)).hexdigest()


_PROJECT_PREFIX = b"localm-coder-project-v1\0"

# Config keys that take effect only once the project is trusted.
GATED_KEYS: tuple[str, ...] = ("auto_approve", "verify", "model", "mode")

# Files under a project's .localcoder folder that take effect only once the
# project is trusted, relative to that folder.
CONTENT_FILES: tuple[str, ...] = ("system.md", "memory.md")
SKILLS_DIR = "skills"

_MAX_SKILL_ENTRIES = 10_000


def _is_link(path: Path) -> bool:
    """Whether *path* is a symlink or an NTFS junction (not followed)."""
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _file_fingerprint(path: Path) -> str:
    """``sha256:<hex>`` of *path*'s bytes, ``link:<target>`` for a link (not
    followed), ``unreadable:<error>`` when it cannot be read."""
    if _is_link(path):
        try:
            return "link:" + os.readlink(path)
        except OSError as e:
            return f"link-unreadable:{type(e).__name__}"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
    except OSError as e:
        return f"unreadable:{type(e).__name__}"
    return "sha256:" + h.hexdigest()


def project_folder(cwd: Path) -> Path:
    """The ``.localcoder`` folder whose instruction, memory and skill files a
    session in *cwd* reads."""
    return Path(cwd).resolve() / ".localcoder"


def project_content(cwd: Path) -> list[tuple[str, str]]:
    """The trust-gated files in *cwd*'s own ``.localcoder`` folder
    (folder_content)."""
    return folder_content(project_folder(cwd))


def folder_content(folder: Path) -> list[tuple[str, str]]:
    """The trust-gated files of the ``.localcoder`` folder *folder* that exist,
    as sorted ``(path relative to the folder, fingerprint)`` pairs.

    Covers ``system.md``, ``memory.md`` and every entry under ``skills/``. A
    link is recorded with its target and not followed; a folder that cannot be
    listed is recorded as unreadable. At most ``_MAX_SKILL_ENTRIES`` skill
    entries are listed, and one more entry records that the list was cut."""
    folder = Path(folder)
    out: list[tuple[str, str]] = []
    if not folder.is_dir():
        return out
    for name in CONTENT_FILES:
        p = folder / name
        if _is_link(p) or p.exists():
            out.append((name, _file_fingerprint(p)))
    skills = folder / SKILLS_DIR
    if _is_link(skills):
        out.append((SKILLS_DIR, _file_fingerprint(skills)))
    elif skills.is_dir():
        count = 0
        stack = [skills]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda e: e.name)
            except OSError as e:
                rel = Path(current).relative_to(folder).as_posix()
                out.append((rel + "/", f"unreadable:{type(e).__name__}"))
                continue
            for entry in entries:
                count += 1
                if count > _MAX_SKILL_ENTRIES:
                    out.append((SKILLS_DIR + "/", "truncated"))
                    stack.clear()
                    break
                p = Path(entry.path)
                rel = p.relative_to(folder).as_posix()
                if _is_link(p):
                    out.append((rel, _file_fingerprint(p)))
                elif entry.is_dir(follow_symlinks=False):
                    out.append((rel + "/", "dir"))
                    stack.append(p)
                else:
                    out.append((rel, _file_fingerprint(p)))
    return sorted(out)


def project_digest(raw: dict, content: list[tuple[str, str]]) -> str:
    """The digest trust is recorded against for a project whose parsed config
    is *raw* and whose trust-gated files are *content* (folder_content).

    Equals ``commands_digest(raw)`` when the config sets none of
    ``GATED_KEYS`` and there are no gated files. Otherwise it is the SHA-256 of
    a fixed prefix, the ``[hooks]`` and ``[mcp]`` tables, every ``GATED_KEYS``
    value and every gated file's fingerprint, so a change to any of them
    changes it. Other keys (``max_turns``, ``temperature``, ...) are not part
    of it."""
    gated = {k: raw[k] for k in GATED_KEYS if k in raw}
    if not content and not gated:
        return commands_digest(raw)
    payload = {"hooks": raw.get("hooks"), "mcp": raw.get("mcp"), "keys": gated,
               "files": [list(item) for item in content]}
    return hashlib.sha256(_PROJECT_PREFIX + _canonical(payload)).hexdigest()


def trust_subject(cwd: Path):
    """``(config path or None, parsed config, trust path, gated files)`` for a
    session in *cwd*.

    The trust is recorded under, and the gated files are read from, *cwd*'s
    own ``.localcoder`` folder when the project config lives there or that
    folder holds gated files; otherwise under the project config found by
    walking up from *cwd* and its folder, so every folder below one project
    shares its trust. With neither, the trust path is
    ``<cwd>/.localcoder/config.toml``. Raises ProjectConfigUnreadable when the
    project config exists but cannot be read."""
    from .project_config import find_project_config, read_project_config_file
    config = find_project_config(Path(cwd))
    raw = read_project_config_file(config) if config is not None else {}
    local = project_folder(cwd)
    local_content = folder_content(local)
    if config is not None:
        config_folder = Path(config).resolve().parent
        if config_folder == local or not local_content:
            content = local_content if config_folder == local else folder_content(config_folder)
            return config, raw, Path(config), content
    return config, raw, local / "config.toml", local_content


def trust_path(cwd: Path, config_path: Path | None = None) -> Path:
    """The path trust for a session in *cwd* is recorded under
    (trust_subject). *config_path* is ignored."""
    return trust_subject(cwd)[2]


def gated_items(raw: dict, content: list[tuple[str, str]]) -> list[str]:
    """One line per setting or file that takes effect only once the project is
    trusted, for display to the user. Values are shown through ``json.dumps``
    with ASCII escapes."""
    lines: list[str] = []
    if raw.get("auto_approve"):
        lines.append("auto_approve = true: tool calls that change files or run "
                     "commands are not confirmed")
    if raw.get("verify"):
        lines.append(f"verify = {json.dumps(str(raw['verify']), ensure_ascii=True)}: "
                     "run as a shell command after every turn that changed files")
    if raw.get("model") is not None:
        lines.append(f"model = {json.dumps(str(raw['model']), ensure_ascii=True)}")
    if raw.get("mode") is not None and _mode_widens(raw.get("mode")):
        lines.append(f"mode = {json.dumps(str(raw['mode']), ensure_ascii=True)}: "
                     "records more of the session than your own coder mode")
    names = {rel for rel, _ in content}
    if "system.md" in names:
        lines.append(".localcoder/system.md: instructions the agent follows")
    if "memory.md" in names:
        lines.append(".localcoder/memory.md: project memory the agent reads")
    skills = sorted({rel.split("/")[1] for rel, _ in content
                     if rel.startswith(SKILLS_DIR + "/") and rel.count("/") >= 1
                     and rel.split("/")[1]})
    if skills or SKILLS_DIR in names:
        lines.append(".localcoder/skills: agent skills "
                     + (json.dumps(skills, ensure_ascii=True) if skills else "(a link)"))
    return lines


def has_gated(raw: dict, content: list[tuple[str, str]]) -> bool:
    """Whether trusting the project would change anything beyond its hooks and
    MCP servers."""
    if content:
        return True
    if any(raw.get(k) not in (None, False, "") for k in GATED_KEYS if k != "mode"):
        return True
    return raw.get("mode") is not None and _mode_widens(raw.get("mode"))


def _mode_widens(value) -> bool:
    """Whether a project ``mode`` of *value* would record more than the coder
    mode resolved without the project. An invalid value does not: callers
    refuse it on their own."""
    from .project_config import mode_widens
    return mode_widens(value)


# Trust decisions made for this process only ("this session" at the prompt, or
# a refusal), keyed like the store: config_key -> (digest, trusted).
_SESSION: dict[str, tuple[str, bool]] = {}
_SESSION_LOCK = threading.Lock()


def set_session_decision(path: Path, digest: str, trusted: bool) -> None:
    """Record, for this process only, that the project with *digest* at *path*
    was trusted or refused."""
    with _SESSION_LOCK:
        _SESSION[config_key(path)] = (digest, bool(trusted))


def clear_session_decision(path: Path) -> None:
    """Drop this process's decision for *path*."""
    with _SESSION_LOCK:
        _SESSION.pop(config_key(path), None)


def session_decision(path: Path, digest: str) -> bool | None:
    """This process's decision for the project with *digest* at *path*: True,
    False, or None when there is none for that digest."""
    with _SESSION_LOCK:
        entry = _SESSION.get(config_key(path))
    if entry is None or entry[0] != digest:
        return None
    return entry[1]


def project_trusted(cwd: Path) -> bool:
    """Whether the trust-gated settings and files of the project at *cwd* may
    take effect: True when it has none, else this process's decision for its
    current digest, else the trust store. Raises ProjectConfigUnreadable when
    the project config exists but cannot be read."""
    _, raw, path, content = trust_subject(Path(cwd))
    if not has_gated(raw, content):
        return True
    digest = project_digest(raw, content)
    decided = session_decision(path, digest)
    if decided is not None:
        return decided
    return is_trusted(path, digest)


def _current(cwd: Path):
    """``(trust path, digest)`` of the project at *cwd*, or None when it has
    nothing trust-gated or its config cannot be read."""
    from .project_config import ProjectConfigUnreadable
    try:
        _, raw, path, content = trust_subject(Path(cwd))
    except ProjectConfigUnreadable:
        return None
    if not has_gated(raw, content) and not set(raw) & {"hooks", "mcp"}:
        return None
    return path, project_digest(raw, content)


# Digests keep_trust moved trust from: (config_key, old digest) -> new digest.
_RENEWED: dict[tuple[str, str], str] = {}


def renewed_digest(path: Path, digest: str) -> str:
    """*digest*, or the digest keep_trust moved the trust for *path* to from
    it in this process."""
    with _SESSION_LOCK:
        return _RENEWED.get((config_key(path), digest), digest)


@contextmanager
def keep_trust(cwd: Path):
    """Keep the project at *cwd* trusted across an edit the user makes through
    localm (``/remember``, ``/forget``): when it was trusted before the edit,
    the digest after it is trusted the same way, for this process only or in
    the store, and a ToolHooks still holding the old digest takes the new one
    at its next refresh_trust (renewed_digest). A store that cannot be written is logged at WARNING and the
    project stays trusted for this process."""
    before = _current(cwd)
    how = None
    old_digest = ""
    if before is not None:
        path, digest = before
        old_digest = digest
        decided = session_decision(path, digest)
        if decided:
            how = "session"
        elif decided is None and is_trusted(path, digest):
            how = "store"
    yield
    if how is None:
        return
    after = _current(cwd)
    if after is None or after == before:
        return
    path, digest = after
    set_session_decision(path, digest, True)
    key = config_key(path)
    with _SESSION_LOCK:
        for entry, new in list(_RENEWED.items()):
            if entry[0] == key and new == old_digest:
                _RENEWED[entry] = digest
        _RENEWED[(key, old_digest)] = digest
    if how == "store":
        try:
            remember_trust(path, digest)
        except (OSError, TrustStoreUnreadable) as e:
            logger.warning("coder project trust: the edit to %s could not be "
                           "recorded as trusted (%s); it is trusted for this "
                           "session only", path, e)


class TrustStoreUnreadable(Exception):
    """The trust store exists but could not be read or is not a JSON object."""


def _load_strict() -> dict:
    """The store's contents. Returns {} when there is no store; raises
    TrustStoreUnreadable when one exists but is unreadable or malformed."""
    try:
        raw = json.loads(_store().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise TrustStoreUnreadable(f"{_store()}: {e}") from e
    if not isinstance(raw, dict):
        raise TrustStoreUnreadable(f"{_store()}: not a JSON object")
    return raw


def is_trusted(config_path: Path, digest: str) -> bool:
    """Whether the hooks with *digest* in *config_path* were trusted.

    An unreadable store trusts nothing and False is returned. The failure is
    logged at WARNING the first time it is seen, and at DEBUG while it
    repeats, until the store is read successfully again. Never raises."""
    try:
        entries = _load_strict()
    except TrustStoreUnreadable as e:
        msg = str(e)
        with _WARNED_LOCK:
            first = msg not in _WARNED
            _WARNED.add(msg)
        log = logger.warning if first else logger.debug
        log("coder hook trust: %s; no project hooks are trusted until it is "
            "fixed or removed", msg)
        return False
    with _WARNED_LOCK:
        _WARNED.clear()
    entry = entries.get(config_key(config_path))
    return isinstance(entry, dict) and entry.get("sha256") == digest


@contextmanager
def _locked():
    from localm.config import _cross_process_lock
    store = _store()
    store.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _cross_process_lock(store):
        yield


def _write(entries: dict) -> None:
    """Replace the store with *entries* through a temporary file in the same
    folder. Raises OSError when it could not be written; the old file is then
    unchanged."""
    from localm.config import _replace_atomic
    store = _store()
    fd, tmp = tempfile.mkstemp(prefix=store.name + ".", suffix=".tmp",
                               dir=str(store.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(entries, indent=2, sort_keys=True))
        _replace_atomic(Path(tmp), store)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remember_trust(config_path: Path, digest: str) -> None:
    """Record that the hooks with *digest* in *config_path* are trusted,
    replacing any earlier record for that file.

    Raises OSError when the store cannot be written, and TrustStoreUnreadable
    when an existing store cannot be read (it is left untouched rather than
    overwritten)."""
    with _locked():
        entries = _load_strict()
        entries[config_key(config_path)] = {
            "sha256": digest, "trusted_at": int(time.time())}
        _write(entries)


def forget_trust(config_path: Path) -> bool:
    """Remove the record for *config_path*. Returns True when one was removed.

    Raises OSError when the store cannot be written, and TrustStoreUnreadable
    when an existing store cannot be read."""
    with _locked():
        entries = _load_strict()
        if entries.pop(config_key(config_path), None) is None:
            return False
        _write(entries)
        return True
