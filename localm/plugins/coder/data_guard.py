# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder tool calls that reach localm's own credentials and settings.

:func:`check` names the first protected target a call reaches, or returns None:

* a file in localm's data folder (``config.home_dir()``) that holds a credential
  or the settings: the owner key, the scoped key store, the owner key records,
  the browser session store, ``config.json`` and its ``.bak``, the model-source
  credentials, the running-instance registry (attach tokens), the TLS
  private keys and the coder's hook and MCP server trust store;
* the data folder itself (listing, reading or archiving it whole);
* a shell command naming one of those paths (also as ``$LOCALM_HOME/...`` and
  in a ``--opt=value`` flag), deleting, moving or changing the permissions of a
  folder that contains one, running ``localm key <sub>`` (any subcommand except
  ``list``) or ``localm config <key> <value>``, or running a nested shell
  (``sh -c``, ``cmd /c``, ``powershell -Command``) whose command line does.

:func:`protected_file` answers the same question for one resolved path; the
sweep tools (grep, search_replace) skip such files.

The agent requires an explicit approval of the exact call for a hit, under
``auto_approve`` too, and denies it when no one can be asked. A best-effort
lexical check (plus ``os.path.realpath`` for file-tool paths and for the path
operands and redirect targets of a shell command, never for a UNC or device
path), not a
sandbox: other variables, ``eval``, interpreter one-liners and encodings defeat
the shell half.
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
    ("tls/ca.key", "localm's local CA private key"),
    ("tls/server.key", "localm's TLS server private key"),
    ("coder-hook-trust.json", "localm's coder hook and MCP server trust store"),
)
_FOLDER_LABEL = "localm's data folder"

_TREE_VERBS = _sg._SECRET_DELETERS | _sg._SECRET_DEST_WRITERS | _sg._SECRET_PERM_WRITERS
_PYTHON_NAMES = frozenset({"python", "python3", "py", "pythonw"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "ash", "fish", "cmd",
                     "powershell", "pwsh"})
_SHELL_COMMAND_FLAGS = frozenset({"-c", "/c", "/k", "-command"})
_HOME_VARS = ("${localm_home}", "$localm_home", "%localm_home%", "$env:localm_home")
_MAX_NESTING = 3


def _norm(text: str) -> str:
    """Forward slashes, casefolded, no trailing slash, and each segment without
    the trailing dots and spaces and the ``:stream`` suffix Windows ignores."""
    parts = text.replace("\\", "/").casefold().split("/")
    out = []
    for i, seg in enumerate(parts):
        drive = i == 0 and len(seg) == 2 and seg[1] == ":"
        if not drive and seg not in (".", ".."):
            seg = seg.split(":", 1)[0].rstrip(" .")
        out.append(seg)
    return "/".join(out).rstrip("/")


class Targets(list):
    """``(normalised path, label)`` entries; ``folders`` holds the normalised
    forms of the data folder itself."""

    folders: list[str]


def protected_targets(home: Optional[Path] = None) -> Targets:
    """The protected entries under *home* (default ``config.home_dir()``), for
    the path as given and its symlink-resolved form, each in canonical
    absolute form and, when under the user's home directory, also in the shell
    guard's ``<home>/...`` form. Normalised by :func:`_norm`."""
    if home is None:
        from localm.config import home_dir
        home = home_dir()
    bases: list[str] = []
    for form in (os.path.abspath(home), os.path.realpath(home)):
        canon = _sg._canon(form)
        if canon and canon not in bases:
            bases.append(canon)
        try:
            rel = Path(form).relative_to(Path.home())
        except (ValueError, RuntimeError, OSError):
            continue
        sentinel = _sg._HOME + "/" + rel.as_posix() if rel.parts else _sg._HOME
        if sentinel not in bases:
            bases.append(sentinel)
    out = Targets()
    out.folders = [_norm(b) for b in bases]
    for base in bases:
        for name, label in _FILE_LABELS:
            out.append((_norm(base.rstrip("/") + "/" + name), label))
    return out


def _match(norm_path: Optional[str], targets: Targets, *, tree: bool) -> Optional[str]:
    """The label of the target *norm_path* equals or lies inside, the data
    folder label when *norm_path* is the data folder itself, or, with *tree*,
    the label of a target lying inside *norm_path*."""
    if not norm_path:
        return None
    p = _norm(norm_path)
    for target, label in targets:
        if p == target or p.startswith(target + "/"):
            return label
    if p in targets.folders:
        return _FOLDER_LABEL
    if tree:
        for target, label in targets:
            if target.startswith(p + "/"):
                return label
    return None


_MAX_LINK_HOPS = 32


def _is_link(path: str) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(isjunction and isjunction(path))


def _first_link_target(path: str) -> Optional[str]:
    """*path* with its first symlink or junction component replaced by that
    link's target (read with ``os.readlink``, which does not open the target),
    or None when no existing component is a link."""
    parts = Path(path).parts
    if not parts:
        return None
    current = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            return None
        if _is_link(current):
            target = os.readlink(current)
            if target.startswith("\\\\?\\") and target[5:6] == ":":
                target = target[4:]
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(current), target)
            return os.path.abspath(os.path.join(target, *parts[i + 1:]))
    return None


def _local_realpath(path: str) -> Optional[str]:
    """``os.path.realpath`` of *path* when *path* and every link on the way to
    its target are local, else None: a UNC or device path, a mapped network
    drive, a link whose target is one of those, or more than
    :data:`_MAX_LINK_HOPS` links. Never opens a network path."""
    from localm.pathsafe import is_mapped_network_drive, is_unc_or_device_path
    pending = os.path.abspath(path)
    for _ in range(_MAX_LINK_HOPS):
        if is_unc_or_device_path(pending) or is_mapped_network_drive(pending):
            return None
        target = _first_link_target(pending)
        if target is None:
            return os.path.realpath(pending)
        pending = target
    return None


def _is_network_path(path: str) -> bool:
    """True for a UNC or device path or a path on a mapped network drive. No
    file system call is made on *path*."""
    from localm.pathsafe import is_mapped_network_drive, is_unc_or_device_path
    return is_unc_or_device_path(path) or is_mapped_network_drive(path)


def _posix_resolve(path: str) -> Optional[str]:
    """*path* resolved the way a POSIX tool does: a ``..`` after a link steps
    back from the link's target, where :func:`os.path.realpath` and
    ``ntpath.realpath`` collapse ``..`` first. None when *path* is not absolute,
    when it, a link on the way or a link's target is a UNC or device path or a
    mapped network drive, or when more than :data:`_MAX_LINK_HOPS` links are
    followed. Never opens a network path."""
    from localm.pathsafe import is_mapped_network_drive, is_unc_or_device_path
    if _is_network_path(path) or not os.path.isabs(path):
        return None
    drive, rest = os.path.splitdrive(path)
    pending = [p for p in rest.replace("\\", "/").split("/") if p not in ("", ".")]
    pending.reverse()
    current = drive + os.sep
    hops = 0
    while pending:
        part = pending.pop()
        if part == "..":
            current = os.path.dirname(current)
            continue
        candidate = os.path.join(current, part)
        if not (os.path.lexists(candidate) and _is_link(candidate)):
            current = candidate
            continue
        hops += 1
        if hops > _MAX_LINK_HOPS:
            return None
        target = os.readlink(candidate)
        if target.startswith("\\\\?\\") and target[5:6] == ":":
            target = target[4:]
        t_drive, t_rest = os.path.splitdrive(target)
        if is_unc_or_device_path(target) or (
                t_drive and is_mapped_network_drive(t_drive + os.sep)):
            return None
        parts = [p for p in t_rest.replace("\\", "/").split("/") if p not in ("", ".")]
        if os.path.isabs(target):
            current = (t_drive or drive) + os.sep
        pending.extend(reversed(parts))
    return _local_realpath(current)


_HOME_MARKERS = ("${home}", "$home", "%userprofile%", "%homepath%",
                 "$env:userprofile", "$env:home", "~")


def _expand_home(text: str) -> Optional[str]:
    """*text* with a leading home marker (``~``, ``$HOME``, ``%USERPROFILE%``)
    or the shell guard's ``<home>`` sentinel replaced by the user's home
    directory, or *text* itself when it has none. None when the home directory
    is unknown."""
    low = text.lower()
    prefix = None
    for marker in _HOME_MARKERS:
        if low == marker or (low.startswith(marker) and text[len(marker)] in "/\\"):
            prefix = len(marker)
            break
    if prefix is None and (text == _sg._HOME or text.startswith(_sg._HOME + "/")):
        prefix = len(_sg._HOME)
    if prefix is None:
        return text
    try:
        home = Path.home().as_posix().rstrip("/")
    except (RuntimeError, OSError):
        return None
    return home + text[prefix:].replace("\\", "/")


def _dotdot_form(token: str, current: Optional[str]) -> Optional[str]:
    """The canonical :func:`_posix_resolve` form of the shell operand *token*
    (relative to the canonical directory *current*) when it has a ``..`` after
    a link, which :func:`shell_guard._norm_target` has already collapsed, else
    None. A ``~``, ``$HOME`` or ``<home>`` prefix is expanded first. No file
    system call is made on a UNC, device or mapped network path."""
    tok = token.strip()
    while tok.endswith("*"):
        tok = tok[:-1]
    if ".." not in tok.replace("\\", "/").split("/"):
        return None
    tok = _expand_home(tok)
    base = _expand_home(current) if current else None
    if tok is None or (current and base is None):
        return None
    if _is_network_path(tok) or (base and _is_network_path(base)):
        return None
    if _sg._canon(tok) is not None:
        raw = tok
    elif base:
        raw = base.rstrip("/") + "/" + tok
    else:
        return None
    if _is_network_path(raw):
        return None
    try:
        if not _sg._dotdot_after_link(raw, base or raw):
            return None
        resolved = _posix_resolve(raw)
    except (OSError, ValueError):
        return None
    return _sg._canon(resolved) if resolved else None


def _file_forms(value: str, cwd: Optional[Path]) -> list[str]:
    """Canonical forms of a file-tool path: lexical, then symlink-resolved
    (:func:`_local_realpath`, so never through a network path)."""
    raw = os.path.expanduser(str(value).strip())
    if not raw:
        return []
    p = Path(raw)
    if not p.is_absolute() and cwd is not None:
        p = Path(cwd) / p
    forms = [os.path.abspath(p)]
    from localm.pathsafe import is_unc_or_device_path
    if not is_unc_or_device_path(raw):
        try:
            real = _local_realpath(forms[0])
        except (OSError, ValueError):
            real = None
        if real:
            forms.append(real)
        if os.name == "posix":
            try:
                through = _posix_resolve(str(p))
            except (OSError, ValueError):
                through = None
            if through:
                forms.append(through)
    return [f for f in (_sg._canon(x) for x in forms) if f]


def protected_file(path: Path, home: Optional[Path] = None, *,
                   targets: Optional[Targets] = None) -> Optional[str]:
    """The label when *path* is, or lies inside, a protected entry of localm's
    data folder (compared lexically and symlink-resolved), else None. Pass
    *targets* (``protected_targets()``) to reuse them across many paths."""
    if targets is None:
        targets = protected_targets(home)
    for form in _file_forms(str(path), None):
        p = _norm(form)
        for target, label in targets:
            if p == target or p.startswith(target + "/"):
                return label
    return None


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


def _expand_home_vars(token: str, home: Path) -> str:
    """*token* with a leading LOCALM_HOME variable replaced by *home*."""
    low = token.casefold()
    for var in _HOME_VARS:
        if low.startswith(var) and (len(token) == len(var) or token[len(var)] in "/\\"):
            return str(home) + token[len(var):]
    return token


def _operands(args: Iterable[str]) -> Iterable[str]:
    """Each non-flag arg, and the value of a ``--opt=value`` flag."""
    for tok in args:
        if _sg._is_flag(tok):
            if "=" in tok:
                yield tok.split("=", 1)[1]
            continue
        yield tok


def _nested_command(name: str, args: Iterable[str]) -> Optional[str]:
    """The command line a shell is told to run (``sh -c``, ``cmd /c``,
    ``powershell -Command``), or None."""
    if name not in _SHELLS:
        return None
    argv = list(args)
    for i, tok in enumerate(argv):
        if tok.casefold() in _SHELL_COMMAND_FLAGS and i + 1 < len(argv):
            rest = argv[i + 1:]
            rest[0] = rest[0].strip("'\"")
            return " ".join(rest)
    return None


def _check_shell(command: str, cwd: Optional[Path], targets: Targets, home: Path,
                 depth: int = 0) -> Optional[str]:
    """The first protected target *command* reaches in any of the shell
    guard's quoting dialects, or None."""
    for segments in _sg._parses(command):
        label = _check_segments(segments, cwd, targets, home, depth)
        if label:
            return label
    return None


def _check_segments(segments, cwd: Optional[Path], targets: Targets, home: Path,
                    depth: int) -> Optional[str]:
    current = _sg._canon(str(cwd)) if cwd is not None else None
    for seg in segments:
        name, args = _sg._argv(seg)
        if not name:
            continue
        label = _localm_command(name, list(args))
        if label:
            return label
        nested = _nested_command(name, args)
        if nested and depth < _MAX_NESTING:
            label = _check_shell(nested, cwd, targets, home, depth + 1)
            if label:
                return label
        tree = name in _TREE_VERBS
        for tok in _operands(args):
            tok = _expand_home_vars(tok, home)
            label = _match_shell_path(_sg._norm_target(tok, current), targets, tree=tree)
            if label:
                return label
            label = _match(_dotdot_form(tok, current), targets, tree=tree)
            if label:
                return label
        for _op, target in seg.redirects:
            target = _expand_home_vars(target, home)
            label = _match_shell_path(_sg._norm_target(target, current), targets, tree=False)
            if label:
                return label
            label = _match(_dotdot_form(target, current), targets, tree=False)
            if label:
                return label
        if name == "cd":
            current = _sg._cd_destination(args, current)
    return None


def _resolved_form(norm_path: Optional[str]) -> Optional[str]:
    """The symlink- and junction-resolved canonical form of the canonical path
    *norm_path*, or None when it is not a local absolute path (relative, UNC)
    or cannot be resolved. A ``<home>`` form is resolved under the user's home
    directory. Never touches a UNC or device path (``pathsafe.is_unc_or_device_path``)."""
    from localm.pathsafe import is_unc_or_device_path
    if not norm_path or is_unc_or_device_path(norm_path):
        return None
    if norm_path == _sg._HOME or norm_path.startswith(_sg._HOME + "/"):
        try:
            norm_path = Path.home().as_posix().rstrip("/") + norm_path[len(_sg._HOME):]
        except (RuntimeError, OSError):
            return None
    if _sg._canon(norm_path) is None:
        return None
    try:
        real = _local_realpath(norm_path)
    except (OSError, ValueError):
        return None
    return _sg._canon(real) if real else None


def _match_shell_path(norm_path: Optional[str], targets: Targets, *,
                      tree: bool) -> Optional[str]:
    """:func:`_match` of a shell operand, as written and then resolved through
    symlinks and junctions (see :func:`_resolved_form`)."""
    label = _match(norm_path, targets, tree=tree)
    if label:
        return label
    return _match(_resolved_form(norm_path), targets, tree=tree)


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
    does not. ``patch_file``'s diff headers are checked as paths too. Raises
    on an internal error; the caller must then treat the call as a hit."""
    if home is None:
        from localm.config import home_dir
        home = home_dir()
    targets = protected_targets(home)
    for arg in shell_args:
        command = args.get(arg)
        if isinstance(command, str) and command.strip():
            label = _check_shell(command, cwd, targets, home)
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
