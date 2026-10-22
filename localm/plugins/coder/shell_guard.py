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

A command line is parsed in three quoting dialects (:data:`_DIALECTS`): the
POSIX ``sh`` rules, the ``cmd.exe`` rules, and a mixed form of the two. A
command is refused when any parse matches a rule. :func:`is_read_only` is the
exception to purity: it reads the filesystem (see its docstring).

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

import glob
import hashlib
import os
import posixpath
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

_HOME = "<home>"
BSLASH = chr(92)

# The characters a backslash escapes inside a double-quoted shell string.
_DQ_ESCAPABLE = '"\\$`'

# How deep command substitution is followed before it is left alone.
_MAX_SUB_DEPTH = 3

# Quoting dialects a command line is parsed in:
#   "mixed": ' and " quote, \ escapes inside "...", an unquoted \ is literal.
#   "sh":    POSIX sh: an unquoted \ escapes the next character, and \ before a
#            line break joins the two lines into one command.
#   "cmd":   cmd.exe: only " quotes, \ is always literal (so \" ends a quoted
#            span), ^ escapes the next character outside quotes, and there is
#            no command substitution.
_DIALECTS = ("mixed", "sh", "cmd")

# Characters after an unquoted backslash that the "sh-nt" parse keeps the
# backslash before, besides letters and digits.
_PATH_CHARS = frozenset("._-/\\:+,=@")

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


def _lex(text: str, dialect: str = "mixed") -> list[_Segment]:
    """Split *text* into simple commands, quote-aware, without evaluating it.

    Segments joined by ``|`` share a ``group``; every other separator
    (``;``, ``&&``, ``||``, ``&``, newline) starts a new one. ``>`` and ``>>``
    targets are recorded as redirects instead of ordinary tokens. Command and
    process substitutions are kept verbatim inside their token AND collected in
    ``subs`` for the caller to parse separately. *dialect* is one of
    :data:`_DIALECTS`, or ``"sh-nt"``: the sh rules, except that an unquoted
    backslash before a letter, a digit or one of :data:`_PATH_CHARS` is kept,
    as cmd.exe passes it to the program (a Windows path separator).
    """
    sh = dialect in ("sh", "sh-nt")
    nt_paths = dialect == "sh-nt"
    cmd = dialect == "cmd"
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
            if sh and ch == "\\" and text[i + 1:i + 2] == "\n":
                i += 2
                continue
            if not cmd and ch == "\\" and i + 1 < n and text[i + 1] in _DQ_ESCAPABLE:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_double = False
                i += 1
                continue
            if not cmd and ch == "$" and i + 1 < n and text[i + 1] == "(":
                body, i = _read_balanced(text, i + 2, "(", ")")
                subs.append(body)
                cur.append("$(" + body + ")")
                continue
            if not cmd and ch == "`":
                body, i = _read_backtick(text, i + 1)
                subs.append(body)
                cur.append("`" + body + "`")
                continue
            cur.append(ch)
            i += 1
            continue

        if sh and ch == "\\":
            if text[i + 1:i + 2] == "\n":
                i += 2
            elif text[i + 1:i + 3] == "\r\n":
                i += 3
            elif i + 1 < n:
                nxt = text[i + 1]
                if nt_paths and (nxt.isalnum() or nxt in _PATH_CHARS):
                    cur.append(ch)
                cur.append(nxt)
                i += 2
            else:
                cur.append(ch)
                i += 1
            continue
        if cmd and ch == "^":
            if i + 1 < n:
                cur.append(text[i + 1])
            i += 2
            continue
        if ch == "'" and not cmd:
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if not cmd and ch == "$" and i + 1 < n and text[i + 1] == "(":
            body, i = _read_balanced(text, i + 2, "(", ")")
            subs.append(body)
            cur.append("$(" + body + ")")
            continue
        if not cmd and ch == "`":
            body, i = _read_backtick(text, i + 1)
            subs.append(body)
            cur.append("`" + body + "`")
            continue
        if not cmd and ch in "<>" and i + 1 < n and text[i + 1] == "(":
            body, i = _read_balanced(text, i + 2, "(", ")")
            subs.append(body)
            cur.append(ch + "(" + body + ")")
            continue
        # A line break ends a command, so it is checked before the generic
        # whitespace branch. See test_a_newline_separates_commands.
        if ch in "\r\n":
            flush_segment(True)
            i += 1
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

def _collect(text: str, depth: int, base: int,
             dialect: str = "mixed") -> tuple[list[_Segment], int]:
    """Segments of *text* and of any substitution inside it, with unique groups,
    parsed in *dialect*.

    Each nested parse is given a group range disjoint from every other, so two
    unrelated commands can never be read as one pipeline.
    """
    segments = _lex(text, dialect)
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
                nested, nxt = _collect(sub, depth + 1, nxt, dialect)
                out.extend(nested)
    return out, nxt


def _parses(text: str, dialects: Sequence[str] = _DIALECTS) -> list[list[_Segment]]:
    """The segments of *text* (substitutions included) in each of *dialects*,
    in that order."""
    return [_collect(text, 0, 0, d)[0] for d in dialects]


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
        The first matching rule in any of :data:`_DIALECTS`, or None when
        nothing matched in any of them.

    Notes
    -----
    Exceptions are NOT swallowed. A caller must treat a raised exception as
    "this command was never checked" and route it to confirmation rather than
    running it. See test_a_classifier_failure_denies_rather_than_allows.
    """
    if not command or not command.strip():
        return None
    cwd_s = _canon(str(cwd)) if cwd is not None else None
    for segments in _parses(command):
        refusal = _classify_segments(segments, cwd_s)
        if refusal is not None:
            return refusal
    return None


def _classify_segments(segments: Sequence[_Segment],
                       cwd_s: Optional[str]) -> Optional[ShellRefusal]:
    """The first rule *segments* (one parse of a command) match, or None."""
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


# --------------------------------------------------------------------------- #
#  Read-only allowlist                                                         #
# --------------------------------------------------------------------------- #
#
# The opposite question to classify(): not "is this catastrophic" but "is this
# provably read-only and confined to the project", so the confirmation prompt
# can be skipped. It is an ALLOWLIST: anything it cannot prove (a wrapper, a
# substitution, a variable, a redirect into a file, an operand outside the
# project, an unknown command, subcommand or option) answers False and the
# command goes to the normal prompt.

# Commands that only read, when every operand stays inside the project. The
# command must be written as one of these bare names.
_READ_ONLY_COMMANDS = frozenset({
    "ls", "dir", "pwd", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep",
    "rg", "tree", "file", "stat", "du", "diff", "cmp", "basename", "dirname",
    "realpath", "echo", "true", "nl", "cut", "tr", "type", "findstr", "where",
    "which", "cd", "get-childitem", "get-content", "select-string",
    "get-location", "test-path", "find", "sort", "uniq",
})

# Read-only git subcommands.
_READ_ONLY_GIT = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame",
    "describe", "shortlog", "branch", "tag", "remote", "grep",
})

# Options that make an otherwise read-only command write a file, run a
# program, read a file the option names, or follow symlinks while recursing:
# (short option letters, matched anywhere in a cluster such as -ro; long
# options, matched as written or abbreviated, with or without a value). git
# is keyed by subcommand ("git branch"); every git subcommand also gets
# _GIT_RISKY_LONG.
_GIT_RISKY_LONG = ("--output", "--ext-diff", "--textconv", "--show-signature")
_GIT_REF_WRITING = ("dDmMcCuf", (
    "--delete", "--move", "--copy", "--set-upstream-to", "--unset-upstream",
    "--edit-description", "--force", "--track", "--no-track", "--create-reflog"))
_RISKY_OPTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "git branch": _GIT_REF_WRITING,
    "git tag": _GIT_REF_WRITING,
    "git grep": ("Of", ("--open-files-in-pager", "--file")),
    "git blame": ("S", ()),
    "git ls-files": ("X", ("--exclude-from",)),
    "git diff": ("O", ()),
    "git log": ("O", ()),
    "git show": ("O", ()),
    "rg": ("Lfz", ("--pre", "--hostname-bin", "--follow", "--search-zip",
                   "--file", "--ignore-file")),
    "grep": ("fR", ("--file", "--dereference-recursive", "--exclude-from")),
    "egrep": ("fR", ("--file", "--dereference-recursive", "--exclude-from")),
    "fgrep": ("fR", ("--file", "--dereference-recursive", "--exclude-from")),
    "sort": ("oT", ("--output", "--compress-program", "--files0-from",
                    "--temporary-directory", "--random-source")),
    "tree": ("oRl", ("--output", "--fromfile", "--infofile")),
    "du": ("LX", ("--dereference", "--files0-from", "--exclude-from")),
    "wc": ("", ("--files0-from",)),
    "diff": ("rX", ("--recursive", "--exclude-from", "--from-file", "--to-file")),
    "file": ("fmCzZ", ("--files-from", "--magic-file", "--compile",
                        "--uncompress", "--uncompress-noreport")),
    "ls": ("L", ("--dereference",)),
}

# Real options of a command that are also the start of one of its risky long
# options: written in full they are that option, not an abbreviation.
_EXACT_OPTIONS: dict[str, frozenset[str]] = {
    "grep": frozenset({"--exclude"}),
    "egrep": frozenset({"--exclude"}),
    "fgrep": frozenset({"--exclude"}),
    "du": frozenset({"--exclude"}),
    "diff": frozenset({"--exclude"}),
    "rg": frozenset({"--ignore"}),
    "git diff": frozenset({"--text"}),
    "git log": frozenset({"--text"}),
    "git show": frozenset({"--text"}),
    "git grep": frozenset({"--text"}),
}

# git remote subcommand words that write.
_GIT_WRITING_WORDS = frozenset({"add", "rm", "rename", "set-url", "prune",
                                "update", "set-head"})

# cmd.exe commands, whose short /x switches are options rather than paths, and
# the switches that recurse (through junctions too).
_DOS_COMMANDS = frozenset({"dir", "findstr", "where", "type", "cd"})
_DOS_RECURSE = frozenset({"/s"})

_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete",
                           "-fprint", "-fprint0", "-fprintf", "-fls"})
_FIND_RISKY = frozenset({"-L", "-H", "-follow", "-files0-from"})
_UNSAFE_CHARS = ("$", "`", "<", ">", "%", "!")

# git configuration that makes a read command run a program, or pulls in
# another file: whole sections, and keys within a section. core.fsmonitor
# names a program unless it is a boolean (true selects git's built-in
# monitor); log.showSignature runs the signature program when true;
# core.hooksPath points git at hook scripts (see _hooks_path_ok); a hook
# section defines hooks in the configuration.
_GIT_RISKY_SECTIONS = frozenset({"include", "includeif", "filter", "pager", "hook"})
_GIT_RISKY_KEYS = {
    "core": frozenset({"fsmonitor", "pager", "hookspath"}),
    "diff": frozenset({"external", "textconv", "command"}),
    "log": frozenset({"showsignature"}),
    "gpg": frozenset({"program"}),
}
_GIT_FALSE = frozenset({"false", "no", "off", "0"})
_GIT_BOOL = _GIT_FALSE | {"true", "yes", "on", "1", ""}


def _plain_operand(tok: str) -> bool:
    """False for operand shapes the lexical path check cannot place: a tilde
    form (~user), a drive-relative path (C:foo), an NTFS stream (a.txt:s) or
    any other colon past a drive letter, and a UNC or device path."""
    if tok.startswith("~") or tok.startswith("\\\\") or tok.startswith("//"):
        return False
    # Brace expansion can produce any path; a glob in a component that starts
    # with a dot (".*") can match "..".
    if any(ch in tok for ch in "{}["):
        return False
    for part in tok.replace("\\", "/").split("/"):
        if part.startswith(".") and any(ch in part for ch in "*?"):
            return False
    colon = tok.find(":")
    if colon == -1:
        return True
    is_drive_absolute = (colon == 1 and tok[0].isalpha() and len(tok) > 2
                         and tok[2] in "/\\")
    return is_drive_absolute and ":" not in tok[2:]


def _inside(norm: Optional[str], root: str) -> bool:
    return norm is not None and (norm == root or norm.startswith(root.rstrip("/") + "/"))


def _real(path: str) -> str:
    """*path* with every symlink and junction resolved, case-normalised."""
    return os.path.normcase(os.path.realpath(path))


_REPARSE_POINT = 0x400
_SYSTEM_FILE = 0x4


def _dotdot_after_link(path: str, root: str) -> bool:
    """True when absolute *path* cannot be judged by resolving it: a ``..``
    comes after a component that is a symlink or junction, or after a wildcard
    component that could match one (a POSIX-style tool applies that ``..`` to
    the link's target, while :func:`_real` collapses it first on Windows); or a
    component below the project *root* (as written) is a link Python does not
    resolve but an MSYS or Cygwin tool follows (a reparse point that is neither
    a symlink nor a junction, such as a WSL symlink; a file with the system
    attribute, such as a Cygwin symlink file); or a component below *root*
    before the last exists and is not a directory; or a ``..`` steps back over
    such a link at or outside *root*. One ``lstat`` per component. Raises
    OSError or ValueError when a component cannot be examined."""
    root_norm = os.path.normcase(os.path.normpath(root))
    drive, rest = os.path.splitdrive(path)
    current = drive + (os.sep if rest[:1] in ("/", "\\") else "")
    parts = [p for p in rest.replace("\\", "/").split("/") if p not in ("", ".")]
    linked = False
    opaque_outside: list[bool] = []
    for i, part in enumerate(parts):
        if part == "..":
            if linked or (opaque_outside and opaque_outside.pop()):
                return True
            current = os.path.dirname(current)
            continue
        current = os.path.join(current, part)
        opaque_outside.append(False)
        if any(ch in part for ch in "*?["):
            linked = True
            continue
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            continue
        attrs = getattr(st, "st_file_attributes", 0)
        if stat.S_ISLNK(st.st_mode) or os.path.isjunction(current):
            linked = True
            continue
        opaque_link = bool(attrs & _REPARSE_POINT) or (
            stat.S_ISREG(st.st_mode) and bool(attrs & _SYSTEM_FILE))
        here = os.path.normcase(os.path.normpath(current))
        if here == root_norm or not _real_within(here, root_norm):
            opaque_outside[-1] = opaque_link
            continue
        file_in_the_middle = i < len(parts) - 1 and not stat.S_ISDIR(st.st_mode)
        if opaque_link or file_in_the_middle:
            return True
    return False


def _real_inside(path: str, root: str, root_real: str) -> bool:
    """Whether absolute *path* stays in *root_real* (the resolved form of
    *root*) with links resolved, and has no ``..`` after a link (see
    :func:`_dotdot_after_link`). Raises OSError or ValueError when the path
    cannot be resolved."""
    return not _dotdot_after_link(path, root) and _real_within(_real(path), root_real)


def _real_within(real: str, root_real: str) -> bool:
    return real == root_real or real.startswith(root_real.rstrip(os.sep) + os.sep)


def _risky_option(tok: str, short: str, long: Sequence[str],
                  exact: frozenset[str] = frozenset()) -> bool:
    """True when option *tok* is, or contains, one of the *short* letters or
    *long* options. A long option written in part counts as the risky options
    it starts (GNU getopt and git accept any unambiguous abbreviation), unless
    it is written exactly as one of the command's real options in *exact*."""
    if tok.startswith("--"):
        stem = tok.split("=", 1)[0]
        return (len(stem) > 2 and stem not in exact
                and any(opt.startswith(stem) for opt in long))
    return tok.startswith("-") and any(ch in short for ch in tok[1:])


def _split_args(name: str, args: Sequence[str]) -> tuple[list[str], list[str]]:
    """(options, operands) of *args*. Everything after ``--`` is an operand,
    and a short /x switch is an option only for a cmd.exe command."""
    options: list[str] = []
    operands: list[str] = []
    ended = False
    for tok in args:
        if ended:
            operands.append(tok)
        elif tok == "--":
            ended = True
        elif tok.startswith("-") or (name in _DOS_COMMANDS and _is_flag(tok)):
            options.append(tok)
        else:
            operands.append(tok)
    return options, operands


def _operand_ok(tok: str, current: str, root: str, root_real: str) -> bool:
    """Whether path operand *tok*, relative to *current*, stays inside the
    project, lexically and with symlinks resolved (every match of a glob)."""
    if not _plain_operand(tok):
        return False
    norm = _norm_target(tok, current)
    if norm is None or not _inside(norm, root) or _secret_kind(norm):
        return False
    path = tok if _canon(tok) is not None else os.path.join(current, tok)
    try:
        if _dotdot_after_link(path, root):
            return False
        hits = glob.glob(path) if any(ch in tok for ch in "*?") else [path]
        return all(_real_inside(p, root, root_real) for p in hits)
    except (OSError, ValueError, RecursionError):
        return False


def _search_path(root: str, root_real: str) -> tuple[list[str], list[str]]:
    """The PATH entries that matter to :func:`_program_shadowed`: the absolute
    ones whose path, as written, lies in the project (*root* or its resolved
    form *root_real*), and the relative or empty ones. Nothing outside the
    project is touched."""
    inside: list[str] = []
    relative: list[str] = []
    roots = {os.path.normcase(os.path.abspath(root)), root_real}
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            relative.append(entry)
            continue
        lexical = os.path.normcase(os.path.abspath(entry))
        if any(_real_within(lexical, r) for r in roots):
            inside.append(entry)
    return inside, relative


def _program_shadowed(name: str, current: str, root: str, root_real: str,
                      search: tuple[list[str], list[str]]) -> bool:
    """True when a file inside the project could be what command *name* runs.

    Looks for a file (not a folder) named *name* (on Windows also *name* plus
    each PATHEXT extension) in every directory the lookup may search that lies
    in the project (*root* or *root_real*, compared as written): on Windows the
    directory the command runs in and the process directory first, then each
    PATH entry (*search*, from :func:`_search_path`), a relative or empty
    entry taken relative to *current*. Returns True when a candidate cannot
    be checked."""
    inside, relative = search
    dirs = list(inside)
    exts = [""]
    if os.name == "nt":
        dirs += [current, os.getcwd()]
        exts += [e for e in os.environ.get(
            "PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
    dirs += [os.path.join(current, entry) for entry in relative]
    roots = {os.path.normcase(os.path.abspath(root)), root_real}
    try:
        for d in dirs:
            lexical = os.path.normcase(os.path.abspath(d))
            if not any(_real_within(lexical, r) for r in roots):
                continue
            if any(os.path.isfile(os.path.join(d, name + ext)) for ext in exts):
                return True
    except (OSError, ValueError):
        return True
    return False


def _git_config_risky(text: str,
                      hooks_path_ok: Optional[Callable[[str], bool]] = None) -> bool:
    """True when git config *text* sets something that makes a read command
    run a program (a filter, a diff driver or external diff, a pager, an
    fsmonitor hook, a hook defined in the configuration, signature checking)
    or includes another file. A core.hooksPath value is risky unless
    *hooks_path_ok* is given and returns True for it (the value as written,
    surrounding whitespace removed)."""
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            end = line.find("]")
            if end == -1:
                return True
            header = line[1:end].strip().lower()
            section = header.replace('"', " ").replace(".", " ").split(" ", 1)[0]
            if section in _GIT_RISKY_SECTIONS:
                return True
            line = line[end + 1:].strip()
        if not line or line[0] in "#;":
            continue
        key, eq, value = line.partition("=")
        key = key.strip().lower()
        if key not in _GIT_RISKY_KEYS.get(section, ()):
            continue
        if (section, key) == ("core", "hookspath"):
            if eq and hooks_path_ok is not None and hooks_path_ok(value.strip()):
                continue
            return True
        value = value.strip().strip('"').strip().lower() if eq else "true"
        if (section, key) == ("core", "fsmonitor") and value in _GIT_BOOL:
            continue
        if (section, key) == ("log", "showsignature") and value in _GIT_FALSE:
            continue
        return True
    return False


_INDEX_HOOK = "post-index-change"
_GITLINK = 0o160000
_MAX_SUBMODULE_DEPTH = 8


def _has_hook(hooks_dir: str) -> bool:
    """True when *hooks_dir* holds git's post-index-change hook: an entry
    named ``post-index-change`` or ``post-index-change.exe``, in any letter
    case. See test_only_the_index_hook_turns_the_git_shortcut_off."""
    try:
        names = os.listdir(hooks_dir)
    except FileNotFoundError:
        return False
    return any(n.lower() in (_INDEX_HOOK, _INDEX_HOOK + ".exe") for n in names)


def _hooks_path_ok(value: str, base: str, root: str, root_real: str) -> bool:
    """Whether core.hooksPath *value* names a folder inside the project, taken
    relative to the work-tree root *base*, that holds no hook :func:`_has_hook`
    reports. False for a value this cannot place (empty, quoted, escaped,
    commented, absolute, starting with ``~``, holding a colon or a ``%``
    interpolation) and for a
    folder that cannot be examined."""
    if (not value or any(ch in value for ch in '"\\#;~:%')
            or os.path.isabs(value) or value.startswith("/")):
        return False
    path = os.path.join(base, value)
    try:
        return _real_inside(path, root, root_real) and not _has_hook(path)
    except (OSError, ValueError):
        return False


def _varint(data: bytes, pos: int, end: int) -> tuple[int, int]:
    """The index-format-4 offset varint at *pos* and the position after it."""
    if pos >= end:
        raise ValueError("truncated index")
    c = data[pos]
    pos += 1
    value = c & 0x7F
    while c & 0x80:
        if pos >= end:
            raise ValueError("truncated index")
        c = data[pos]
        pos += 1
        value = ((value + 1) << 7) | (c & 0x7F)
    return value, pos


def _parse_index(data: bytes, hash_len: int) -> tuple[list[str], Optional[str]]:
    """The gitlink (submodule) paths in index file content *data* whose object
    ids are *hash_len* bytes, and the hex id of its shared index (split index)
    or None. Raises ValueError when *data* is not such an index."""
    end = len(data) - hash_len
    if end < 12 or data[:4] != b"DIRC":
        raise ValueError("not a git index")
    version, count = struct.unpack_from(">II", data, 4)
    if version not in (2, 3, 4):
        raise ValueError(f"index version {version}")
    fixed = 40 + hash_len
    pos = 12
    prev = b""
    paths: list[str] = []
    for _ in range(count):
        if pos + fixed + 2 > end:
            raise ValueError("truncated index")
        mode = struct.unpack_from(">I", data, pos + 24)[0]
        flags = struct.unpack_from(">H", data, pos + fixed)[0]
        name_at = pos + fixed + 2
        if flags & 0x4000:
            if version < 3:
                raise ValueError("extended flags in a version 2 index")
            name_at += 2
        if version == 4:
            strip, name_at = _varint(data, name_at, end)
            nul = data.index(b"\0", name_at, end)
            if strip > len(prev):
                raise ValueError("bad path compression")
            name = prev[:len(prev) - strip] + data[name_at:nul]
            pos = nul + 1
        else:
            nul = data.index(b"\0", name_at, end)
            name = data[name_at:nul]
            pos += (nul - pos + 8) & ~7
        prev = name
        if mode & 0o170000 == _GITLINK:
            paths.append(name.decode("utf-8", "surrogateescape"))
    shared = None
    while pos < end:
        if pos + 8 > end:
            raise ValueError("truncated index extension")
        signature = data[pos:pos + 4]
        size = struct.unpack_from(">I", data, pos + 4)[0]
        body = pos + 8
        if body + size > end:
            raise ValueError("truncated index extension")
        if signature == b"link":
            if size < hash_len:
                raise ValueError("bad link extension")
            oid = data[body:body + hash_len]
            if any(oid):
                shared = oid.hex()
        pos = body + size
    if pos != end:
        raise ValueError("index entries overrun")
    return paths, shared


def _read_index(path: str) -> tuple[list[str], Optional[str], int]:
    """:func:`_parse_index` of the index file at *path*, with the object id
    length found from its trailing checksum (SHA-1 or SHA-256), or by trying
    both when the checksum is zero (index.skipHash). Raises OSError or
    ValueError."""
    with open(path, "rb") as fh:
        data = fh.read()
    lengths = []
    for hash_len, algo in ((20, "sha1"), (32, "sha256")):
        if len(data) > hash_len and hashlib.new(algo, data[:-hash_len]).digest() == data[-hash_len:]:
            lengths = [hash_len]
            break
    else:
        if len(data) > 32 and not any(data[-20:]):
            lengths = [20, 32]
    if not lengths:
        raise ValueError("index checksum does not match")
    error: Optional[Exception] = None
    for hash_len in lengths:
        try:
            paths, shared = _parse_index(data, hash_len)
            return paths, shared, hash_len
        except (ValueError, struct.error) as exc:
            error = exc
    raise ValueError(f"unreadable index: {error}")


def _index_gitlinks(gitdir: str) -> list[str]:
    """The paths, relative to the work tree and ``/``-separated, of the
    submodule entries in the index of git directory *gitdir*, its shared index
    included; empty when there is no index. Raises OSError or ValueError when
    an index file cannot be read."""
    index = os.path.join(gitdir, "index")
    if not os.path.lexists(index):
        return []
    paths, shared, hash_len = _read_index(index)
    if shared is not None:
        more, _shared, more_len = _read_index(os.path.join(gitdir, "sharedindex." + shared))
        if more_len != hash_len:
            raise ValueError("shared index uses another object id length")
        paths = paths + more
    return paths


def _network_path(path: str) -> bool:
    """True for a UNC, device or other double-slash path."""
    return path.replace("\\", "/").startswith("//")


def _repo_config_risky(dotgit: str, base: str, root: str, depth: int = 0) -> bool:
    """True when the repository whose git directory is *dotgit* (a ``.git``
    entry in work-tree root *base*, or *base* itself for a bare repository)
    can make a read command run a program: a local config file passes
    :func:`_git_config_risky` (a core.hooksPath in the repository's own config
    files passes when :func:`_hooks_path_ok` accepts it, one in a submodule's
    config under ``modules/`` never does), or a ``hooks`` folder of it or of a
    submodule under ``modules/`` holds a hook (:func:`_has_hook`), or the same
    holds for the repository of a submodule entry in its index whose folder
    has a ``.git`` entry, recursively. A ``gitdir:`` file and a ``commondir``
    are followed. Only files whose resolved path lies in the project *root*
    are read: a target outside it is not looked at, and a network or device
    target makes the answer True without being touched. A gitdir or commondir
    with a ``..`` after a link (see :func:`_dotdot_after_link`), a submodule
    path that is not a plain relative path, and submodules nested deeper than
    :data:`_MAX_SUBMODULE_DEPTH` also make the answer True. Raises OSError or
    ValueError when a file cannot be read."""
    root_real = _real(root)
    prefix = os.path.realpath(root).rstrip(os.sep) + os.sep
    real = os.path.realpath(dotgit)
    if not real.startswith(prefix):
        return False
    gitdirs = [real]
    if os.path.isfile(real):
        with open(real, encoding="utf-8", errors="replace") as fh:
            first = fh.read().strip()
        if not first.lower().startswith("gitdir:"):
            return True
        target = first[len("gitdir:"):].strip()
        if _network_path(target):
            return True
        gitdirs = [target if os.path.isabs(target) else os.path.join(base, target)]
    configs: list[tuple[str, bool]] = []
    seen: set[str] = set()
    primary: Optional[str] = None
    while gitdirs:
        nxt = gitdirs.pop()
        if _dotdot_after_link(nxt, root):
            return True
        real = os.path.realpath(nxt)
        if not real.startswith(prefix) or real in seen:
            continue
        seen.add(real)
        if primary is None:
            primary = real
        configs += [(os.path.join(real, "config"), True),
                    (os.path.join(real, "config.worktree"), True)]
        if _has_hook(os.path.join(real, "hooks")):
            return True
        for top, _dirs, files in os.walk(os.path.join(real, "modules")):
            configs += [(os.path.join(top, f), False) for f in files
                        if f in ("config", "config.worktree")]
            if os.path.basename(top) == "hooks" and _has_hook(top):
                return True
        common = os.path.join(real, "commondir")
        if os.path.isfile(common):
            with open(common, encoding="utf-8", errors="replace") as fh:
                target = fh.read().strip()
            if _network_path(target):
                return True
            gitdirs.append(target if os.path.isabs(target) else os.path.join(real, target))

    def hooks_path_ok(value: str) -> bool:
        return _hooks_path_ok(value, base, root, root_real)

    for path, own in configs:
        real = os.path.realpath(path)
        if not real.startswith(prefix) or not os.path.isfile(real):
            continue
        with open(real, encoding="utf-8", errors="replace") as fh:
            if _git_config_risky(fh.read(), hooks_path_ok if own else None):
                return True
    if primary is None or os.path.normcase(dotgit) == os.path.normcase(base):
        return False
    for rel in _index_gitlinks(primary):
        parts = rel.split("/")
        if (os.path.isabs(rel) or "\\" in rel or ":" in rel
                or any(p in ("", ".", "..") for p in parts)):
            return True
        sub = os.path.join(base, *parts)
        sub_git = os.path.join(sub, ".git")
        if not os.path.lexists(sub_git):
            continue
        if depth >= _MAX_SUBMODULE_DEPTH or _repo_config_risky(sub_git, sub, root, depth + 1):
            return True
    return False


def _repo_at(d: str) -> Optional[str]:
    """The git directory git would use at folder *d* itself: its ``.git``
    entry, else *d* when *d* is a git directory (``HEAD`` file plus ``objects``
    and ``refs`` folders), else None. Raises OSError."""
    dotgit = os.path.join(d, ".git")
    if os.path.lexists(dotgit):
        return dotgit
    if (os.path.isfile(os.path.join(d, "HEAD")) and os.path.isdir(os.path.join(d, "objects"))
            and os.path.isdir(os.path.join(d, "refs"))):
        return d
    return None


def _git_runs_programs(start: str, root: str) -> bool:
    """True when the repository git uses for a command run in *start* is
    inside the project and its local configuration or hook scripts can make a
    read command run a program, or cannot be read. The repository is looked up
    from *start* as written and from *start* with its links resolved (git uses
    the resolved one); either being risky makes the answer True. At each folder
    a ``.git`` entry is used first, then the folder itself when it is a git
    directory (a bare repository). A repository above the project is not looked
    at."""
    d = start
    while _inside(d, root):
        try:
            dotgit = _repo_at(d)
            if dotgit is not None:
                if _repo_config_risky(dotgit, d, root):
                    return True
                break
        except (OSError, ValueError):
            return True
        parent = _canon(posixpath.dirname(d))
        if parent is None or parent == d:
            break
        d = parent
    try:
        d = os.path.realpath(start)
        root_real = _real(root)
    except (OSError, ValueError):
        return True
    while _real_within(os.path.normcase(d), root_real):
        try:
            dotgit = _repo_at(d)
            if dotgit is not None:
                return _repo_config_risky(dotgit, d, root)
        except (OSError, ValueError):
            return True
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return False


def _read_only_segments(segments: Sequence[_Segment], root: str, root_real: str,
                        search: tuple[list[str], list[str]]) -> bool:
    """Whether every segment of one parse provably only reads inside *root*."""
    current = root
    for seg in segments:
        if seg.subs:
            return False
        for _op, target in seg.redirects:
            if _canon(target) not in ("/dev/null",) and target.lower() != "nul":
                return False
        if not seg.tokens or _peel(seg.tokens) != 0:
            return False
        if any(ch in tok for tok in seg.tokens for ch in _UNSAFE_CHARS):
            return False
        prog = seg.tokens[0]
        name = prog.lower() if os.name == "nt" else prog
        if name != "git" and name not in _READ_ONLY_COMMANDS:
            return False
        if _program_shadowed(name, current, root, root_real, search):
            return False
        args = seg.tokens[1:]
        key = name
        if name == "git":
            if not args or args[0].startswith("-") or args[0].lower() not in _READ_ONLY_GIT:
                return False
            sub = args[0].lower()
            key = f"git {sub}"
            options, operands = _split_args(name, args[1:])
            if sub in ("branch", "tag", "remote") and operands:
                return False
            if any(op.lower() in _GIT_WRITING_WORDS for op in operands):
                return False
            if _git_runs_programs(current, root):
                return False
        else:
            if name == "find" and any(a.lower() in _FIND_ACTIONS or a in _FIND_RISKY
                                      for a in args):
                return False
            options, operands = _split_args(name, args)
            if name in ("dir", "findstr") and any(
                    o.lower() in _DOS_RECURSE for o in options):
                return False
            if name == "uniq" and len(operands) > 1:
                return False
        short, long = _RISKY_OPTIONS.get(key, ("", ()))
        if name == "git":
            long = long + _GIT_RISKY_LONG
        exact = _EXACT_OPTIONS.get(key, frozenset())
        if any(_risky_option(o, short, long, exact) for o in options):
            return False
        if any(o.startswith("-") and ch in o.split("=", 1)[0]
               for o in options for ch in "/\\"):
            return False
        values = [o.split("=", 1)[1] for o in options
                  if o.startswith("--") and "=" in o]
        for tok in operands + values:
            if not _operand_ok(tok, current, root, root_real):
                return False
        if name == "cd":
            current = _cd_destination(args, current)
            if current is None or not _inside(current, root):
                return False
            try:
                if not _real_inside(current, root, root_real):
                    return False
            except (OSError, ValueError):
                return False
    return True


def is_read_only(command: str, cwd: Optional[Path | str]) -> bool:
    """True when *command* provably only reads, and only inside *cwd*.

    The command must be read-only in every one of :data:`_DIALECTS` (on
    Windows the sh parse is ``"sh-nt"``, see :func:`_lex`). Unlike
    :func:`classify` it reads the filesystem: operands are resolved with
    symlinks followed, a project file that a command name could run (a PATH
    entry inside the project, or on Windows the working directory) makes it
    False, and a git command is False when the project's git configuration
    can run a program or its repository holds a hook script. A False answer is not a verdict that the command is
    unsafe, only that it could not be proven safe, so the caller asks the
    user as before. See test_read_only_shell_commands.
    """
    if not command or not command.strip() or cwd is None:
        return False
    if "\n" in command or "\r" in command:
        return False
    root = _canon(str(cwd))
    if root is None:
        return False
    try:
        root_real = _real(root)
        search = _search_path(root, root_real)
    except (OSError, ValueError):
        return False
    dialects = ("mixed", "sh-nt", "cmd") if os.name == "nt" else _DIALECTS
    for segments in _parses(command, dialects):
        if not segments or not _read_only_segments(segments, root, root_real, search):
            return False
    return True
