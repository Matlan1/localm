# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic reject-list for model-issued shell commands.

:func:`classify` inspects a command string and returns a :class:`ShellRefusal`
for a small, fixed set of catastrophic shapes, or ``None`` for everything else.
It is checked unconditionally in ``agent/execution.py`` before the confirmation
gate, so it holds under ``auto_approve``, an absent confirm handler, ``lenient``
calls and sub-agents alike.

The function is pure: given the same ``(command, cwd)`` it returns the same
answer. It reads no environment variables, touches no filesystem, and starts no
process. Relative paths are resolved against ``cwd`` LEXICALLY
(``posixpath.normpath``), never with ``Path.resolve()``, which does I/O.

WHAT IT BLOCKS (rule ids are stable and are named in the refusal message):

``fs-root-wipe``      recursive delete whose target is a filesystem root, a
                      drive root, a home directory or a top-level system dir
``fs-device-wipe``    filesystem creation or raw writes to a block device
``secrets-write``     a write, delete or permission change on a credential path
``remote-exec-pipe``  downloaded content fed straight into an interpreter
``git-force-push``    a force push at a protected branch, at every ref, or a
                      remote delete of a protected branch, whether written as a
                      flag or as a ``+ref`` / ``:ref`` refspec
``git-hard-reset``    ``git reset --hard``, which discards uncommitted work

WHAT IT IS NOT. This is a floor against the accidental and the model-error
case, not a sandbox. It is a reject-list over a lexical parse, so it does not
resist a determined adversary: ``eval``, base64, variable indirection and
unusual quoting all defeat it, and it makes no claim otherwise. Confinement
belongs to the operating system, the disabled-tool gate and ``--scope``.

Ambiguous shapes are deliberately ALLOWED here and fall through to the
confirmation gate. A false positive that blocks ordinary work is the failure
mode that gets a guard removed, so a rule that cannot be stated exactly is not
a rule in this module.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_HOME = "<home>"
BSLASH = chr(92)

# The characters a backslash escapes inside a double-quoted shell string.
_DQ_ESCAPABLE = '"\\$`'

# How deep command substitution is followed before it is left alone.
_MAX_SUB_DEPTH = 3

_POSIX_SYSTEM_DIRS = frozenset({
    "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
    "/libx32", "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin",
    "/srv", "/sys", "/usr", "/var",
    "/Applications", "/Library", "/System", "/Users", "/Volumes",
})

# Directories whose immediate children are user home directories.
_HOME_PARENTS = ("/home", "/Users", "/users")

_WIN_SYSTEM_RESTS = frozenset({
    "", "windows", "winnt", "users", "programdata",
    "program files", "program files (x86)",
})

_RECURSIVE_LONG_FLAGS = frozenset({"--recursive", "--recurse", "-recurse", "/s"})
_UNIX_DELETERS = frozenset({"rm", "shred"})
_WIN_DELETERS = frozenset({"del", "erase", "rd", "rmdir", "remove-item"})

_MKFS_PREFIX = "mkfs"
_DEVICE_WIPERS = frozenset({"wipefs", "sgdisk", "sfdisk", "diskpart", "format"})

# Character devices that are ordinary sinks or sources, not storage.
_SAFE_DEV_NODES = frozenset({
    "/dev/null", "/dev/zero", "/dev/full", "/dev/tty", "/dev/stdin",
    "/dev/stdout", "/dev/stderr", "/dev/random", "/dev/urandom",
})

_DOWNLOADERS = frozenset({
    "curl", "wget", "fetch", "aria2c", "httpie", "http", "https",
    "iwr", "invoke-webrequest", "irm", "invoke-restmethod",
})
_INTERPRETERS = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "ash", "csh", "tcsh", "fish",
    "python", "python2", "python3", "perl", "ruby", "node", "deno", "bun",
    "php", "lua", "osascript", "cmd", "powershell", "pwsh",
    "iex", "invoke-expression",
})

# Command prefixes that wrap another command without changing what it does.
_WRAPPERS = frozenset({
    "sudo", "doas", "nohup", "time", "nice", "ionice", "stdbuf", "command",
    "builtin", "exec", "setsid", "env", "xargs", "script", "unbuffer",
})
# Shell keywords and grouping tokens that stand before a real command.
_SHELL_KEYWORDS = frozenset({
    "if", "then", "else", "elif", "fi", "while", "until", "for", "do",
    "done", "case", "esac", "select", "function", "{", "}", "!",
})
# Wrapper flags that consume the token after them.
_WRAPPER_FLAGS_WITH_VALUE = frozenset({
    "-u", "--user", "-g", "--group", "-I", "-i", "--replace", "-n",
    "--max-args", "-P", "--max-procs", "-d", "--delimiter",
})

_SECRET_DELETERS = frozenset({
    "rm", "shred", "unlink", "truncate", "del", "erase", "rd", "rmdir",
    "remove-item", "clear-content",
})
_SECRET_DEST_WRITERS = frozenset({
    "mv", "cp", "install", "ln", "move", "copy", "xcopy", "robocopy",
    "move-item", "copy-item", "rename-item", "out-file", "set-content",
    "add-content",
})
_SECRET_PERM_WRITERS = frozenset({
    "chmod", "chown", "chgrp", "icacls", "cacls", "takeown", "attrib",
    "set-acl",
})
_SECRET_INPLACE_EDITORS = frozenset({"sed", "perl", "ruby", "gawk", "awk"})
_SECRET_TEE = frozenset({"tee"})

_PROTECTED_BRANCHES = frozenset({
    "master", "main", "trunk", "develop", "development", "production", "release",
})
_GIT_FORCE_FLAGS = frozenset({"-f", "--force"})
_GIT_GLOBAL_FLAGS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree",
                                          "--namespace", "--exec-path"})


@dataclass(frozen=True)
class ShellRefusal:
    """A refused command, with the rule that refused it.

    Attributes
    ----------
    rule:
        Stable rule id, e.g. ``"fs-root-wipe"``. Named in the message and used
        by tests and the audit trail.
    matched:
        The fragment of the command that triggered the rule.
    reason:
        One sentence naming what the command would do.
    guidance:
        The narrower operation to use instead. Always non-empty, so a refusal
        is a redirection rather than a dead end.
    """

    rule: str
    matched: str
    reason: str
    guidance: str

    def message(self) -> str:
        """The text shown to the caller and returned to the model."""
        return (
            f"Blocked by the shell safety gate [{self.rule}]: {self.reason} "
            f"Matched: {self.matched!r}. {self.guidance} "
            "This check is unconditional and cannot be approved away."
        )


@dataclass(frozen=True)
class _Segment:
    """One simple command from a parsed command line."""

    tokens: tuple[str, ...]
    redirects: tuple[tuple[str, str], ...]
    subs: tuple[str, ...]
    group: int


# --------------------------------------------------------------------------- #
#  Lexing                                                                      #
# --------------------------------------------------------------------------- #

def _read_balanced(text: str, start: int, opener: str, closer: str) -> tuple[str, int]:
    """Read from *start* to the matching *closer*, tracking nesting and quotes.

    Returns the body and the index just past the closer. An unterminated span
    yields everything remaining.
    """
    depth = 1
    i = start
    n = len(text)
    in_single = False
    in_double = False
    while i < n:
        ch = text[i]
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double = False
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:], n


def _read_backtick(text: str, start: int) -> tuple[str, int]:
    """Read from *start* to the next unescaped backtick."""
    i = start
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if text[i] == "`":
            return text[start:i], i + 1
        i += 1
    return text[start:], n


def _lex(text: str) -> list[_Segment]:
    """Split *text* into simple commands, quote-aware, without evaluating it.

    Segments joined by ``|`` share a ``group``; every other separator
    (``;``, ``&&``, ``||``, ``&``, newline) starts a new one. ``>`` and ``>>``
    targets are recorded as redirects instead of ordinary tokens. Command and
    process substitutions are kept verbatim inside their token AND collected in
    ``subs`` for the caller to parse separately.
    """
    segments: list[_Segment] = []
    tokens: list[str] = []
    redirects: list[tuple[str, str]] = []
    subs: list[str] = []
    cur: list[str] = []
    pending: Optional[str] = None
    group = 0
    i = 0
    n = len(text)
    in_single = False
    in_double = False

    def flush_token() -> None:
        nonlocal pending
        if not cur:
            return
        tok = "".join(cur)
        cur.clear()
        if pending is not None:
            redirects.append((pending, tok))
            pending = None
        else:
            tokens.append(tok)

    def flush_segment(new_group: bool) -> None:
        nonlocal tokens, redirects, subs, group, pending
        flush_token()
        pending = None
        if tokens or redirects:
            segments.append(_Segment(tuple(tokens), tuple(redirects),
                                     tuple(subs), group))
        tokens = []
        redirects = []
        subs = []
        if new_group:
            group += 1

    while i < n:
        ch = text[i]

        if in_single:
            if ch == "'":
                in_single = False
            else:
                cur.append(ch)
            i += 1
            continue

        if in_double:
            if ch == "\\" and i + 1 < n and text[i + 1] in _DQ_ESCAPABLE:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_double = False
                i += 1
                continue
            if ch == "$" and i + 1 < n and text[i + 1] == "(":
                body, i = _read_balanced(text, i + 2, "(", ")")
                subs.append(body)
                cur.append("$(" + body + ")")
                continue
            if ch == "`":
                body, i = _read_backtick(text, i + 1)
                subs.append(body)
                cur.append("`" + body + "`")
                continue
            cur.append(ch)
            i += 1
            continue

        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "(":
            body, i = _read_balanced(text, i + 2, "(", ")")
            subs.append(body)
            cur.append("$(" + body + ")")
            continue
        if ch == "`":
            body, i = _read_backtick(text, i + 1)
            subs.append(body)
            cur.append("`" + body + "`")
            continue
        if ch in "<>" and i + 1 < n and text[i + 1] == "(":
            body, i = _read_balanced(text, i + 2, "(", ")")
            subs.append(body)
            cur.append(ch + "(" + body + ")")
            continue
        if ch.isspace():
            flush_token()
            i += 1
            continue
        if ch == "|":
            if pending is not None and not cur:
                i += 1
                continue
            if i + 1 < n and text[i + 1] == "|":
                flush_segment(True)
                i += 2
            elif i + 1 < n and text[i + 1] == "&":
                flush_segment(False)
                i += 2
            else:
                flush_segment(False)
                i += 1
            continue
        if ch == "&":
            if pending is not None and not cur:
                cur.append(ch)
                i += 1
                continue
            if i + 1 < n and text[i + 1] == "&":
                flush_segment(True)
                i += 2
                continue
            if i + 1 < n and text[i + 1] == ">":
                flush_token()
                if i + 2 < n and text[i + 2] == ">":
                    pending = ">>"
                    i += 3
                else:
                    pending = ">"
                    i += 2
                continue
            flush_segment(True)
            i += 1
            continue
        if ch in ";\n()":
            flush_segment(True)
            i += 1
            continue
        if ch == ">":
            flush_token()
            if i + 1 < n and text[i + 1] == ">":
                pending = ">>"
                i += 2
            else:
                pending = ">"
                i += 1
            continue
        if ch == "<":
            flush_token()
            pending = "<"
            i += 1
            continue
        cur.append(ch)
        i += 1

    flush_segment(False)
    return segments


# --------------------------------------------------------------------------- #
#  Command-name extraction                                                     #
# --------------------------------------------------------------------------- #

def _base(token: str) -> str:
    """The lowercased command name of *token*, without directory or extension."""
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in (".exe", ".cmd", ".bat", ".com", ".ps1"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _is_assignment(token: str) -> bool:
    """True for a leading VAR=value environment assignment."""
    head, sep, _ = token.partition("=")
    if not sep or not head or head[0].isdigit():
        return False
    return all(c.isalnum() or c == "_" for c in head)


def _is_flag(token: str) -> bool:
    """True for an option rather than an operand, in both flag styles.

    A DOS-style switch is at most two characters after the slash and wholly
    alphanumeric, so /s and /q are options while the root, /etc and /usr stay
    operands.
    """
    if token.startswith("-"):
        return True
    if token.startswith("/") and 1 < len(token) <= 3 and token[1:].isalnum():
        return True
    return False


def _peel(tokens: Sequence[str]) -> int:
    """Index of the real command in *tokens*, past assignments and wrappers."""
    i = 0
    n = len(tokens)
    while i < n:
        if _is_assignment(tokens[i]):
            i += 1
            continue
        wrapper = _base(tokens[i])
        if wrapper in _SHELL_KEYWORDS:
            i += 1
            continue
        if wrapper in _WRAPPERS:
            i += 1
            while i < n:
                tok = tokens[i]
                if tok.startswith("-"):
                    i += 2 if tok in _WRAPPER_FLAGS_WITH_VALUE else 1
                    continue
                if wrapper == "env" and _is_assignment(tok):
                    i += 1
                    continue
                break
            continue
        break
    return i


def _argv(segment: _Segment) -> tuple[str, tuple[str, ...]]:
    """The command name and arguments of *segment*, wrappers peeled off."""
    start = _peel(segment.tokens)
    if start >= len(segment.tokens):
        return "", ()
    return _base(segment.tokens[start]), tuple(segment.tokens[start + 1:])


# --------------------------------------------------------------------------- #
#  Path normalisation (lexical only - never touches the filesystem)            #
# --------------------------------------------------------------------------- #

def _canon(raw: str) -> Optional[str]:
    """Canonical lexical form of an absolute path, or None when it is relative.

    Windows paths become c:/x/y with a lowercased drive, a bare C: becomes
    c:/, and a UNC path keeps its two leading slashes. No parent reference is
    resolved against the filesystem and no symlink is followed.
    """
    if not raw:
        return None
    text = raw.replace("\\", "/")
    if text.startswith("//"):
        return posixpath.normpath(text)
    if len(text) >= 2 and text[1] == ":" and text[0].isascii() and text[0].isalpha():
        drive = text[0].lower()
        rest = text[2:]
        if rest == "":
            return drive + ":/"
        if not rest.startswith("/"):
            return None
        normed = posixpath.normpath(rest)
        return drive + ":/" if normed == "/" else drive + ":" + normed
    if text.startswith("/"):
        return posixpath.normpath(text)
    return None


def _join(base: Optional[str], tail: str) -> Optional[str]:
    """Join *tail* onto *base* lexically, or None when *base* cannot resolve it.

    Handles the <home> sentinel, which no canonical absolute form recognises,
    and strips a trailing separator off *base*, so a base of "/" does not
    produce the doubled leading slash that POSIX reserves for a UNC root.
    """
    if base is None:
        return None
    text = tail.replace(BSLASH, "/")
    stem = base.rstrip("/")
    if base == _HOME or base.startswith(_HOME + "/"):
        joined = posixpath.normpath(stem + "/" + text)
        return None if joined.startswith("..") else joined
    return _canon(posixpath.normpath(stem + "/" + text))


def _norm_target(token: str, cwd: Optional[str]) -> Optional[str]:
    """Canonical form of a path-like *token*, or None when it is not one.

    A trailing glob and trailing separators are stripped first, so /* and /
    normalise alike; a token that is nothing but a glob normalises to *cwd*.
    A tilde, $HOME and the Windows home variables become the <home> sentinel.
    A relative token is joined onto *cwd* lexically; without a *cwd* it is not
    resolvable and None is returned.
    """
    tok = token.strip()
    if not tok or _is_flag(tok):
        return None
    while tok.endswith("*"):
        tok = tok[:-1]
    while len(tok) > 1 and tok[-1] in "/\\":
        tok = tok[:-1]
    if not tok:
        return _join(cwd, ".")

    lowered = tok.lower()
    for marker in ("${home}", "$home", "%userprofile%", "%homepath%",
                   "$env:userprofile", "$env:home", "~"):
        if lowered == marker:
            return _HOME
        if lowered.startswith(marker) and tok[len(marker)] in "/\\":
            rest = tok[len(marker) + 1:].replace("\\", "/")
            joined = posixpath.normpath(_HOME + "/" + rest)
            return None if joined.startswith("..") else joined

    direct = _canon(tok)
    if direct is not None:
        return direct
    return _join(cwd, tok)


def _target_kind(norm: Optional[str]) -> Optional[str]:
    """The label root, home or system for a wipe-worthy path, else None."""
    if not norm:
        return None
    if norm == "/":
        return "root"
    if norm == _HOME:
        return "home"
    if len(norm) == 3 and norm[1] == ":" and norm[2] == "/":
        return "root"
    if norm.startswith("//"):
        return "root" if len([p for p in norm.split("/") if p]) <= 2 else None
    if norm in _POSIX_SYSTEM_DIRS:
        return "system"
    for parent in _HOME_PARENTS:
        if norm.startswith(parent + "/") and norm.count("/") == parent.count("/") + 1:
            return "home"
    if len(norm) > 3 and norm[1] == ":" and norm[2] == "/":
        rest = norm[3:].lower()
        if rest in _WIN_SYSTEM_RESTS:
            return "system"
        if rest.startswith("users/") and rest.count("/") == 1:
            return "home"
    return None


def _home_relative(norm: str) -> Optional[str]:
    """The part of *norm* below a home directory, or None when it is elsewhere."""
    if norm == _HOME:
        return ""
    if norm.startswith(_HOME + "/"):
        return norm[len(_HOME) + 1:]
    if norm == "/root":
        return ""
    if norm.startswith("/root/"):
        return norm[len("/root/"):]
    for parent in _HOME_PARENTS:
        prefix = parent + "/"
        if norm.startswith(prefix):
            tail = norm[len(prefix):]
            return tail.partition("/")[2] if "/" in tail else ""
    if len(norm) > 3 and norm[1] == ":" and norm[2] == "/":
        rest = norm[3:]
        if rest.lower().startswith("users/"):
            tail = rest[len("users/"):]
            return tail.partition("/")[2] if "/" in tail else ""
    return None


def _secret_kind(norm: Optional[str]) -> Optional[str]:
    """A short label for a credential path, or None.

    Only a path anchored at a home directory or at /etc counts, so a project
    fixture such as tests/data/.ssh is not a credential path.
    """
    if not norm:
        return None
    lowered = norm.lower()
    for etc_path, label in (("/etc/shadow", "the shadow password file"),
                            ("/etc/gshadow", "the shadow group file"),
                            ("/etc/passwd", "the system password file"),
                            ("/etc/sudoers", "the sudoers policy")):
        if lowered == etc_path or lowered.startswith(etc_path + "/"):
            return label

    rest = _home_relative(norm)
    if rest is None:
        return None
    parts = [p.lower() for p in rest.split("/") if p]
    if not parts:
        return None
    head = parts[0]
    if head == ".ssh":
        return "the SSH key directory"
    if head == ".gnupg":
        return "the GnuPG keyring"
    if head in (".netrc", "_netrc"):
        return "the netrc credentials file"
    if head == ".aws" and len(parts) > 1 and parts[1] == "credentials":
        return "the AWS credentials file"
    if head == ".docker" and len(parts) > 1 and parts[1] == "config.json":
        return "the Docker registry credentials"
    if head == ".kube" and len(parts) > 1 and parts[1] == "config":
        return "the Kubernetes cluster credentials"
    return None


def _is_block_device(norm: Optional[str], raw: str) -> bool:
    """True for a raw disk target: a /dev node that is not an ordinary sink."""
    if raw.replace("\\", "/").lower().startswith("//./physicaldrive"):
        return True
    if not norm or not norm.startswith("/dev/"):
        return False
    return norm.lower() not in _SAFE_DEV_NODES


# --------------------------------------------------------------------------- #
#  Rules                                                                       #
# --------------------------------------------------------------------------- #

def _has_recursive_flag(args: Sequence[str]) -> bool:
    """True when *args* carry a recursion switch in any of its spellings."""
    for arg in args:
        low = arg.lower()
        if low in _RECURSIVE_LONG_FLAGS:
            return True
        if low.startswith("--"):
            continue
        if low.startswith("-") and "r" in low[1:]:
            return True
    return False


def _rule_root_wipe(name: str, args: Sequence[str],
                    cwd: Optional[str]) -> Optional[ShellRefusal]:
    if name not in _UNIX_DELETERS and name not in _WIN_DELETERS:
        return None
    if not _has_recursive_flag(args):
        return None
    for arg in args:
        kind = _target_kind(_norm_target(arg, cwd))
        if kind is None:
            continue
        where = {"root": "the filesystem root",
                 "home": "a home directory",
                 "system": "a top-level system directory"}[kind]
        return ShellRefusal(
            rule="fs-root-wipe",
            matched=name + " " + arg,
            reason="it recursively deletes " + where + ".",
            guidance="Delete a specific subdirectory of the project instead.")
    return None


def _rule_device_wipe(name: str, args: Sequence[str],
                      redirects: Sequence[tuple[str, str]],
                      cwd: Optional[str]) -> Optional[ShellRefusal]:
    if name.startswith(_MKFS_PREFIX):
        return ShellRefusal(
            rule="fs-device-wipe", matched=name,
            reason="it creates a filesystem, destroying everything on the target.",
            guidance="Filesystem creation is never part of a coding task.")
    if name in _DEVICE_WIPERS:
        targeted = any(_target_kind(_norm_target(a, cwd)) == "root"
                       or _is_block_device(_norm_target(a, cwd), a) for a in args)
        if name != "format" or targeted:
            return ShellRefusal(
                rule="fs-device-wipe", matched=" ".join([name, *args])[:120],
                reason="it repartitions or wipes a storage device.",
                guidance="Disk administration is never part of a coding task.")
    if name == "dd":
        for arg in args:
            if arg.lower().startswith("of="):
                target = arg[3:]
                norm = _norm_target(target, cwd)
                if _is_block_device(norm, target) or _target_kind(norm) == "root":
                    return ShellRefusal(
                        rule="fs-device-wipe", matched=arg,
                        reason="it writes raw bytes over a storage device.",
                        guidance="Write to a file inside the project instead.")
    if name == "shred":
        for arg in args:
            if _is_block_device(_norm_target(arg, cwd), arg):
                return ShellRefusal(
                    rule="fs-device-wipe", matched="shred " + arg,
                    reason="it overwrites a storage device.",
                    guidance="Shred a specific file inside the project instead.")
    for op, target in redirects:
        if op in (">", ">>") and _is_block_device(_norm_target(target, cwd), target):
            return ShellRefusal(
                rule="fs-device-wipe", matched=op + " " + target,
                reason="it redirects output onto a storage device.",
                guidance="Redirect to a file inside the project instead.")
    return None


def _rule_secrets_write(name: str, args: Sequence[str],
                        redirects: Sequence[tuple[str, str]],
                        cwd: Optional[str]) -> Optional[ShellRefusal]:
    def refuse(matched: str, label: str, verb: str) -> ShellRefusal:
        return ShellRefusal(
            rule="secrets-write", matched=matched,
            reason="it " + verb + " " + label + ".",
            guidance="Credential stores are out of bounds for an automated edit.")

    for op, target in redirects:
        if op in (">", ">>"):
            label = _secret_kind(_norm_target(target, cwd))
            if label:
                return refuse(op + " " + target, label, "writes over")

    operands = [a for a in args if not _is_flag(a)]

    if name in _SECRET_DELETERS or name in _SECRET_PERM_WRITERS or name in _SECRET_TEE:
        if name in _SECRET_DELETERS:
            verb = "deletes"
        elif name in _SECRET_PERM_WRITERS:
            verb = "changes the permissions of"
        else:
            verb = "writes over"
        for arg in operands:
            label = _secret_kind(_norm_target(arg, cwd))
            if label:
                return refuse(name + " " + arg, label, verb)

    if name in _SECRET_DEST_WRITERS and operands:
        dest = operands[-1]
        label = _secret_kind(_norm_target(dest, cwd))
        if label:
            return refuse(name + " ... " + dest, label, "writes into")

    if name in _SECRET_INPLACE_EDITORS and any(
            a.startswith("-i") and not a.startswith("--") for a in args):
        for arg in operands:
            label = _secret_kind(_norm_target(arg, cwd))
            if label:
                return refuse(name + " -i ... " + arg, label, "edits in place")

    if name == "dd":
        for arg in args:
            if arg.lower().startswith("of="):
                label = _secret_kind(_norm_target(arg[3:], cwd))
                if label:
                    return refuse(arg, label, "writes over")
    return None


def _is_force_refspec(arg: str) -> bool:
    """True for a +ref refspec, which forces the push with no flag present."""
    return arg.startswith("+") and len(arg) > 1


def _is_delete_refspec(arg: str) -> bool:
    """True for a :ref refspec, which deletes the ref on the remote."""
    return arg.startswith(":") and len(arg) > 1


def _protected_refspecs(args: Sequence[str]) -> list[str]:
    """Protected branch names appearing as the destination of a refspec."""
    found = []
    for arg in args:
        if arg.startswith("-"):
            continue
        spec = arg[1:] if arg.startswith("+") else arg
        dest = spec.rpartition(":")[2] if ":" in spec else spec
        name = dest[len("refs/heads/"):] if dest.startswith("refs/heads/") else dest
        if name.lower() in _PROTECTED_BRANCHES:
            found.append(name)
    return found


def _rule_git(name: str, args: Sequence[str]) -> Optional[ShellRefusal]:
    if name != "git":
        return None
    rest = list(args)
    while rest:
        head = rest[0]
        if head in _GIT_GLOBAL_FLAGS_WITH_VALUE:
            rest = rest[2:]
            continue
        if head.startswith("-"):
            rest = rest[1:]
            continue
        break
    if not rest:
        return None
    sub, sub_args = rest[0], rest[1:]

    if sub == "reset" and "--hard" in sub_args:
        return ShellRefusal(
            rule="git-hard-reset", matched="git reset --hard",
            reason="it discards every uncommitted change in the working tree.",
            guidance="Commit or branch first, or restore one path with "
                     "git checkout <ref> -- <path>.")

    if sub == "push":
        operands = [a for a in sub_args if not a.startswith("-")]
        forcing = (any(a in _GIT_FORCE_FLAGS for a in sub_args)
                   or any(_is_force_refspec(a) for a in operands))
        deleting = (any(a in ("--delete", "-d") for a in sub_args)
                    or any(_is_delete_refspec(a) for a in operands))
        if "--mirror" in sub_args:
            return ShellRefusal(
                rule="git-force-push", matched="git push --mirror",
                reason="it overwrites and can delete every branch on the remote.",
                guidance="Push one named branch instead.")
        if forcing and "--all" in sub_args:
            return ShellRefusal(
                rule="git-force-push", matched="git push --force --all",
                reason="it force-overwrites every branch on the remote.",
                guidance="Push one named branch instead.")
        if forcing or deleting:
            for branch in _protected_refspecs(sub_args):
                verb = ("deletes the protected branch" if deleting and not forcing
                        else "force-overwrites the protected branch")
                return ShellRefusal(
                    rule="git-force-push", matched="git push ... " + branch,
                    reason="it " + verb + " " + branch + " on the remote.",
                    guidance="Push to a feature branch and open a pull request.")
    return None


def _cd_destination(args: Sequence[str], current: Optional[str]) -> Optional[str]:
    """Where a cd moves to, or None when it cannot be resolved lexically.

    A bare cd goes to the home sentinel. "cd -" returns to a previous directory
    this module does not track, so it resolves to None, which makes every later
    relative target unclassifiable rather than wrongly attributed.
    """
    operands = [a for a in args if not _is_flag(a)]
    if not operands:
        return _HOME
    if operands[0] == "-":
        return None
    return _norm_target(operands[0], current)


def _first_command(text: str) -> str:
    """The command name that *text* starts with, wrappers peeled off."""
    segments = _lex(text)
    return _argv(segments[0])[0] if segments else ""


def _rule_remote_exec(segments: Sequence[_Segment]) -> Optional[ShellRefusal]:
    by_group: dict[int, list[_Segment]] = {}
    for seg in segments:
        by_group.setdefault(seg.group, []).append(seg)

    for pipeline in by_group.values():
        names = [_argv(seg)[0] for seg in pipeline]
        for idx, name in enumerate(names):
            if name not in _DOWNLOADERS:
                continue
            for later in names[idx + 1:]:
                if later in _INTERPRETERS:
                    return ShellRefusal(
                        rule="remote-exec-pipe",
                        matched=name + " ... | " + later,
                        reason="it runs downloaded content without review.",
                        guidance="Download to a file, read it, then run it as "
                                 "a separate step.")

    for seg in segments:
        name = _argv(seg)[0]
        if name not in _INTERPRETERS:
            continue
        for sub in seg.subs:
            downloader = _first_command(sub)
            if downloader in _DOWNLOADERS:
                return ShellRefusal(
                    rule="remote-exec-pipe",
                    matched=name + " <(" + downloader + " ...)",
                    reason="it runs downloaded content without review.",
                    guidance="Download to a file, read it, then run it as "
                             "a separate step.")
    return None


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _collect(text: str, depth: int, base: int) -> tuple[list[_Segment], int]:
    """Segments of *text* and of any substitution inside it, with unique groups.

    Each nested parse is given a group range disjoint from every other, so two
    unrelated commands can never be read as one pipeline.
    """
    segments = _lex(text)
    out: list[_Segment] = []
    highest = base
    for seg in segments:
        group = base + seg.group
        highest = max(highest, group)
        out.append(_Segment(seg.tokens, seg.redirects, seg.subs, group))
    nxt = highest + 1
    if depth < _MAX_SUB_DEPTH:
        for seg in segments:
            for sub in seg.subs:
                nested, nxt = _collect(sub, depth + 1, nxt)
                out.extend(nested)
    return out, nxt


def classify_git_push(remote: str, branch: str) -> Optional[ShellRefusal]:
    """Refuse a git_push TOOL call that would force or delete a protected ref.

    The tool takes argv parts rather than a command line, and appends *branch*
    to the git argv verbatim, so a refspec such as "+main" or ":main" reaches
    git as a force or a delete without any flag being present. The same rule
    that governs a shell "git push" is applied here.

    Returns the refusal, or None when nothing matched.
    """
    args = ["push"]
    if remote:
        args.append(remote)
    if branch:
        args.append(branch)
    return _rule_git("git", args)


def classify(command: str, cwd: Optional[Path | str] = None) -> Optional[ShellRefusal]:
    """Refuse *command* when it matches a reject-list rule, else return None.

    Parameters
    ----------
    command:
        The shell command line as the model wrote it.
    cwd:
        Directory that relative paths are resolved against, lexically. Optional;
        a relative path is not classifiable without it and is allowed through.

    Returns
    -------
    ShellRefusal | None
        The first matching rule, or None when nothing matched.

    Notes
    -----
    Exceptions are NOT swallowed. A caller must treat a raised exception as
    "this command was never checked" and route it to confirmation rather than
    running it. See test_a_classifier_failure_denies_rather_than_allows.
    """
    if not command or not command.strip():
        return None
    cwd_s = _canon(str(cwd)) if cwd is not None else None
    segments, _ = _collect(command, 0, 0)

    pipe_refusal = _rule_remote_exec(segments)
    if pipe_refusal is not None:
        return pipe_refusal

    current = cwd_s
    for seg in segments:
        name, args = _argv(seg)
        if not name:
            continue
        for refusal in (
            _rule_root_wipe(name, args, current),
            _rule_device_wipe(name, args, seg.redirects, current),
            _rule_secrets_write(name, args, seg.redirects, current),
            _rule_git(name, args),
        ):
            if refusal is not None:
                return refusal
        if name == "cd":
            current = _cd_destination(args, current)
    return None
