# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder tool calls that reach localm's own credentials and settings.

:func:`check` names the first protected target a call reaches, or returns None:

* a file in localm's data folder (``config.home_dir()``) that holds a credential
  or the settings: the owner key, the scoped key store, the owner key records,
  the browser session store, ``config.json`` and its ``.bak``, the model-source
  credentials, and the running-instance registry (attach tokens);
* a shell command naming one of those paths, deleting, moving or changing the
  permissions of a folder that contains one, or running ``localm key <sub>``
  (any subcommand except ``list``) or ``localm config <key> <value>``.

The agent requires an explicit approval of the exact call for a hit, under
``auto_approve`` too, and denies it when no one can be asked. A best-effort
lexical check (plus ``os.path.realpath`` for file-tool paths), not a sandbox:
variables, ``eval`` and encodings defeat the shell half.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from . import shell_guard as _sg

_FILE_LABELS: tuple[tuple[str, str], ...] = (
    ("auth.key", "localm's owner API key"),
    ("auth.json", "localm's API key store"),
    ("auth.kdf.json", "localm's owner key records"),
    ("sessions.json", "localm's browser session store"),
    ("config.json", "localm's settings"),
    ("config.json.bak", "localm's settings backup"),
    ("model_source_credentials.json", "localm's model-source credentials"),
    ("run", "localm's running-instance registry"),
)

_TREE_VERBS = _sg._SECRET_DELETERS | _sg._SECRET_DEST_WRITERS | _sg._SECRET_PERM_WRITERS
_PYTHON_NAMES = frozenset({"python", "python3", "py", "pythonw"})


def _norm(text: str) -> str:
    return text.replace("\\", "/").rstrip("/").casefold()


def protected_targets(home: Optional[Path] = None) -> list[tuple[str, str]]:
    """``(normalised path, label)`` for every protected entry under *home*
    (default ``config.home_dir()``). Each entry appears in canonical absolute
    form and, when *home* is under the user's home directory, also in the
    shell guard's ``<home>/...`` form. Normalised: forward slashes, casefolded,
    no trailing slash."""
    if home is None:
        from localm.config import home_dir
        home = home_dir()
    bases = []
    canon = _sg._canon(str(home))
    if canon:
        bases.append(canon)
    try:
        rel = Path(os.path.abspath(home)).relative_to(Path.home())
        bases.append(_sg._HOME + "/" + rel.as_posix() if rel.parts else _sg._HOME)
    except (ValueError, RuntimeError, OSError):
        pass
    out = []
    for base in bases:
        for name, label in _FILE_LABELS:
            out.append((_norm(base.rstrip("/") + "/" + name), label))
    return out


def _match(norm_path: Optional[str], targets, *, tree: bool) -> Optional[str]:
    """The label of the target *norm_path* equals or lies inside, or, with
    *tree*, also of a target lying inside *norm_path*."""
    if not norm_path:
        return None
    p = _norm(norm_path)
    for target, label in targets:
        if p == target or p.startswith(target + "/"):
            return label
        if tree and target.startswith(p + "/"):
            return label
    return None


def _file_forms(value: str, cwd: Optional[Path]) -> Iterable[str]:
    """Normalised forms of a file-tool path: lexical, then symlink-resolved."""
    raw = os.path.expanduser(str(value).strip())
    if not raw:
        return []
    p = Path(raw)
    if not p.is_absolute() and cwd is not None:
        p = Path(cwd) / p
    forms = [os.path.abspath(p)]
    try:
        forms.append(os.path.realpath(p))
    except (OSError, ValueError):
        pass
    return [f for f in (_sg._canon(x) for x in forms) if f]


def _localm_command(name: str, args: list[str]) -> Optional[str]:
    """A label when *name* (a shell_guard ``_argv`` base name) with *args* runs
    ``localm key <sub>`` (not ``list``) or ``localm config <key> <value>``,
    also through ``python -m localm`` and ``uv run``."""
    argv = list(args)
    if name == "uv" and argv[:1] == ["run"] and len(argv) > 1:
        return _localm_command(_sg._base(argv[1]), argv[2:])
    if name in _PYTHON_NAMES:
        if len(argv) >= 2 and argv[0] == "-m" and argv[1].casefold() == "localm":
            argv = argv[2:]
        else:
            return None
    elif name != "localm":
        return None
    ops = [a for a in argv if not a.startswith("-")]
    if not ops:
        return None
    if ops[0].casefold() == "key" and (len(ops) < 2 or ops[1].casefold() != "list"):
        return "localm's API keys (localm key)"
    if ops[0].casefold() == "config" and len(ops) >= 3:
        return "localm's settings (localm config)"
    return None


def _check_shell(command: str, cwd: Optional[Path], targets) -> Optional[str]:
    cwd_s = _sg._canon(str(cwd)) if cwd is not None else None
    segments, _ = _sg._collect(command, 0, 0)
    current = cwd_s
    for seg in segments:
        name, args = _sg._argv(seg)
        if not name:
            continue
        label = _localm_command(name, list(args))
        if label:
            return label
        tree = name in _TREE_VERBS
        for tok in args:
            if _sg._is_flag(tok):
                continue
            label = _match(_sg._norm_target(tok, current), targets, tree=tree)
            if label:
                return label
        for _op, target in seg.redirects:
            label = _match(_sg._norm_target(target, current), targets, tree=False)
            if label:
                return label
        if name == "cd":
            current = _sg._cd_destination(args, current)
    return None


def _patch_paths(diff: str) -> list[str]:
    out = []
    for line in str(diff or "").splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split("\t", 1)[0].strip()
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path and path != "/dev/null":
                out.append(path)
    return out


def check(call_name: str, args: dict, cwd: Optional[Path], *,
          shell_args: Iterable[str] = (), path_args: Iterable[str] = (),
          paths: Iterable[str] = (), home: Optional[Path] = None) -> Optional[str]:
    """The label of the first protected target the call reaches, or None.

    *shell_args* name the args holding a command line, *path_args* the args
    holding a file path or glob (for a glob, the part before its first wildcard
    is checked), and *paths* are further target paths (nested or list args).
    A ``search_replace`` glob also hits when its fixed part names a folder that
    contains a protected entry; a glob with no fixed part (the working folder)
    does not.
    ``patch_file``'s diff headers are checked as paths too. Raises
    on an internal error; the caller must then treat the call as a hit."""
    targets = protected_targets(home)
    for arg in shell_args:
        command = args.get(arg)
        if isinstance(command, str) and command.strip():
            label = _check_shell(command, cwd, targets)
            if label:
                return label
    values = [args.get(a) for a in path_args]
    values.extend(paths)
    if call_name == "patch_file":
        values.extend(_patch_paths(args.get("diff", "")))
    for value in values:
        if not value or not isinstance(value, str):
            continue
        cut = min((i for i in (value.find(c) for c in "*?[") if i >= 0), default=-1)
        stem = value if cut < 0 else value[:cut]
        tree = (call_name == "search_replace"
                and stem.strip().rstrip("/\\") not in ("", "."))
        for form in _file_forms(stem or ".", cwd):
            label = _match(form, targets, tree=tree)
            if label:
                return label
    return None
