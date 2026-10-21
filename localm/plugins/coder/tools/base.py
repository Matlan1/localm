# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared tool primitives: the ``ToolResult`` value type, the cwd-confinement
and output-truncation helpers every tool builds on, and the canonical
subprocess-execution primitive (``run_subprocess``, CODER-2)."""

from __future__ import annotations

import logging
import os
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from localm import pathsafe

log = logging.getLogger(__name__)

@dataclass
class ToolResult:
    ok:      bool
    output:  str
    summary: str = ""       # one-line display shown in the console
    truncated: bool = False
    # True when this output is external content even though the tool that
    # returned it is not an untrusted tool (read_output reading back a spilled
    # fetch_url page). provenance frames it as untrusted either way.
    untrusted_output: bool = False
    # For a tool whose target files are discovered at RUNTIME rather than
    # named in its call args (search_replace's glob+regex sweep has no
    # `path` arg to read ahead of time), the (relative_path, old_bytes,
    # new_text) of every file it touched - or would touch, when called with
    # its own dry_run=True. None for every other tool, and None here too
    # when nothing matched. Both a caller that must NOT touch disk (patch
    # mode, previewing via dry_run) and a caller recording what a REAL write
    # just changed (the changed-files/undo tracker) read this one field, fed
    # by the same matching pass.
    changes: list[tuple[str, bytes, str]] | None = None

    @classmethod
    def success(cls, output: str, summary: str = "",
                changes: list[tuple[str, bytes, str]] | None = None) -> ToolResult:
        return cls(ok=True, output=output, summary=summary, changes=changes)

    @classmethod
    def error(cls, message: str,
              changes: list[tuple[str, bytes, str]] | None = None) -> ToolResult:
        # summary is a ONE-LINE console display (see field comment). Keep the
        # full message in output and only a capped first line here, so a long
        # diagnostic (a timeout's partial output, git stderr) cannot flood the
        # interactive console or the audit/event summary field.
        head = message.splitlines()[0] if message else ""
        if len(head) > 200:
            head = head[:200] + "..."
        # changes: a PARTIAL apply (some files written before a mid-batch
        # failure) still tracks those files as changed/undoable - they really
        # did change on disk, and reporting the call as an error must not make
        # that mutation invisible.
        return cls(ok=False, output=message, summary=f"ERROR: {head}", changes=changes)

    def to_xml(self, tool_name: str):
        """The ``<tool_result>`` frame for this result.

        When ``output`` carries untrusted character ranges, the frame is built
        with ``compose()`` so those ranges survive into the returned text,
        shifted past the opening tag. Output with no ranges returns the plain
        string this has always returned.
        """
        status = "ok" if self.ok else "error"
        trunc  = ' truncated="true"' if self.truncated else ""
        head = f'<tool_result name="{tool_name}" status="{status}"{trunc}>\n'
        tail = "\n</tool_result>"
        from localm.textguard import compose, untrusted_spans_of
        if untrusted_spans_of(self.output):
            return compose(head, self.output, tail)
        return f"{head}{self.output}{tail}"


_MAX_OUTPUT = 8_000   # chars - truncate large outputs to spare context


def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> tuple[str, bool]:
    """Head and tail of *text* within *max_chars*. Inside a running tool call
    the whole text is also kept in memory (see ..spill) and the marker names
    the id read_output reads it back by."""
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    from ..spill import spill
    sid = spill(text)
    if sid is None:
        marker = f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n"
    else:
        lines = text.count("\n") + 1
        marker = (
            f"\n\n... [{len(text) - max_chars} chars not shown. The full output "
            f"({lines} lines) is kept as {sid}: read it with "
            f'read_output(id="{sid}", offset=N, limit=N) or search it with '
            f'read_output(id="{sid}", pattern="...")] ...\n\n')
    return text[:half] + marker + text[-half:], True


def _partial_on_timeout(exc) -> str:
    """Format any output a timed-out subprocess produced before it was killed.

    ``subprocess.TimeoutExpired`` carries the stdout/stderr captured up to the
    kill - the test progress or last log line before the hang. On POSIX the
    attributes can be bytes even in text mode (a CPython quirk), so they are
    decoded defensively. Returns '' when nothing was captured.
    """
    parts = []
    for label, data in (("", getattr(exc, "stdout", None)),
                        ("STDERR:\n", getattr(exc, "stderr", None))):
        if not data:
            continue
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", errors="replace")
        data = data.strip()
        if data:
            parts.append(label + data)
    if not parts:
        return ""
    text, _ = _truncate("\n".join(parts))
    return "\n[partial output captured before the timeout]\n" + text


def _confine(cwd: Path, path: str) -> Path:
    """
    Resolve *path* against *cwd* and verify it stays inside *cwd*.

    Raises ``PermissionError`` with a clear message if the resolved path
    escapes the working directory (path traversal attempt or accidental
    absolute path outside the project root).

    Delegates to ``pathsafe.confined_absolute_or_under``, the shared
    absolute-or-relative confinement primitive. That refuses a UNC/device
    *path* before it can reach ``Path(path).resolve()`` (the syscall dials SMB
    and can hang for minutes), and rejects an NTFS Alternate Data Stream or
    short-name alias in *path* on a character and identity check.

    ``"."``/``""`` naming *cwd* itself is a supported contract here (a tool
    listing "the project root" is a legitimate request). The shared primitive
    treats a self-referential result as invalid, matching its delete-oriented
    callers, so that one case is handled in this function rather than by
    relaxing the primitive for every caller.
    """
    if path in (".", ""):
        return cwd.resolve()
    try:
        return pathsafe.confined_absolute_or_under(cwd, path)
    except ValueError as e:
        raise PermissionError(
            f"'{path}' resolves outside the working directory '{cwd.resolve()}'. "
            f"All file operations must stay within the project root. ({e})"
        ) from e


# Directory names a restricted (scoped-key) coder session may not create, or
# create or change anything inside, at any depth of the project.
RESTRICTED_UNWRITABLE_DIRS: frozenset[str] = frozenset({".localcoder", ".git"})

RESTRICTED_UNWRITABLE_MESSAGE = (
    "a .localcoder or .git directory holds this project's coder or repository "
    "configuration, and a restricted session cannot create or change anything "
    "inside one, through a link to one, or through a hard link to any file")

# Largest .git or commondir file followed; git accepts the same size.
_GIT_POINTER_MAX_BYTES = 1 << 20


def unwritable_component(parts) -> Optional[str]:
    """The first of *parts* naming a ``RESTRICTED_UNWRITABLE_DIRS`` entry, or None.

    Matches case-insensitively and ignoring trailing dots and spaces, the
    spellings Windows folds onto the same directory name."""
    for part in parts:
        if str(part).rstrip(" .").casefold() in RESTRICTED_UNWRITABLE_DIRS:
            return str(part)
    return None


def _is_link(path) -> bool:
    """Whether *path* is a symlink or an NTFS junction (not followed)."""
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _local_target(base: Path, named: str) -> Optional[Path]:
    """*named* joined onto *base* when relative, then resolved.

    None when *named* is empty or a UNC, device or mapped network drive path;
    such a path is never resolved. A ``\\\\?\\X:`` prefix is read as the plain
    drive path ``X:``."""
    if named[:4] in ("\\\\?\\", "//?/") and named[5:6] == ":":
        named = named[4:]
    if (not named or pathsafe.is_unc_or_device_path(named)
            or pathsafe.is_mapped_network_drive(named)):
        return None
    return (base / named).resolve()


def _read_pointer(pointer_file: Path, prefix: str) -> Optional[str]:
    """The path *pointer_file* names after *prefix* (``"gitdir:"`` for a
    ``.git`` file, ``""`` for a ``commondir`` file), read the way git reads it:
    the whole file, trailing whitespace removed.

    None when *pointer_file* is not a regular file, is larger than
    ``_GIT_POINTER_MAX_BYTES`` or does not start with *prefix*. Raises
    ``OSError`` when it is a file that cannot be read."""
    if not pointer_file.is_file():
        return None
    with open(pointer_file, "rb") as fh:
        data = fh.read(_GIT_POINTER_MAX_BYTES + 1)
    if len(data) > _GIT_POINTER_MAX_BYTES:
        return None
    text = data.decode("utf-8", errors="replace").rstrip()
    if not text.startswith(prefix):
        return None
    return text[len(prefix):].strip()


def _protected_entry_targets(entry: Path) -> list[Path]:
    """The folders *entry*, a path named like a ``RESTRICTED_UNWRITABLE_DIRS``
    entry, redirects to.

    For a link: what it points at. For a ``.git`` file: the folder its
    ``gitdir:`` line names. Then the folder named by the ``commondir`` file of
    that folder, or of *entry* itself when it is a directory. UNC, device and
    mapped network drive targets are left out (see :func:`_local_target`).
    Raises ``OSError`` when *entry* cannot be inspected."""
    found: list[Path] = []
    gitdir: Optional[Path] = None
    if _is_link(entry):
        gitdir = _local_target(entry.parent, os.readlink(entry))
        if gitdir is not None:
            found.append(gitdir)
    elif entry.is_file():
        named = _read_pointer(entry, "gitdir:")
        gitdir = _local_target(entry.parent, named) if named else None
        if gitdir is not None:
            found.append(gitdir)
    elif entry.is_dir():
        gitdir = entry
    if gitdir is not None:
        named = _read_pointer(gitdir / "commondir", "")
        common = _local_target(gitdir, named) if named else None
        if common is not None:
            found.append(common)
    return found


def _reachable_by_name(directory: Path) -> bool:
    """Whether an entry of *directory* can be looked up by name (the directory
    can be traversed even when it cannot be listed)."""
    for name in sorted(RESTRICTED_UNWRITABLE_DIRS):
        try:
            os.lstat(directory / name)
        except FileNotFoundError:
            continue
        except OSError:
            return False
        return True
    return True


def unwritable_redirects(cwd_resolved: Path) -> tuple[Path, ...]:
    """Every folder a ``RESTRICTED_UNWRITABLE_DIRS`` entry of the project at
    *cwd_resolved* redirects to, at any depth.

    Walks the project without following links and without entering a matching
    entry, collecting :func:`_protected_entry_targets` of each matching entry.
    A subdirectory that cannot be listed is skipped, and logged at debug
    level, only when its entries cannot be looked up by name either.

    Raises ``OSError`` when the project root cannot be listed, a matching entry
    cannot be inspected, or a subdirectory cannot be listed but its entries can
    still be looked up by name.
    """
    isjunction = getattr(os.path, "isjunction", None)
    found: list[Path] = []
    stack = [cwd_resolved]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError as e:
            if current == cwd_resolved or _reachable_by_name(current):
                raise
            log.debug("restricted write check skipped unreachable %s: %s", current, e)
            continue
        for entry in entries:
            if entry.name.rstrip(" .").casefold() in RESTRICTED_UNWRITABLE_DIRS:
                found.extend(_protected_entry_targets(Path(entry.path)))
            elif (entry.is_dir(follow_symlinks=False) and not entry.is_symlink()
                    and not (isjunction and isjunction(entry.path))):
                stack.append(Path(entry.path))
    return tuple(found)


def inside_unwritable_dir(cwd_resolved: Path, resolved: Path,
                          redirects: Optional[tuple[Path, ...]] = None) -> bool:
    """Whether the already-resolved *resolved* lies inside a
    ``RESTRICTED_UNWRITABLE_DIRS`` directory of the project at *cwd_resolved*.

    True when a component below the project root names one, or when *resolved*
    lies inside a folder such an entry anywhere in the project redirects to
    (:func:`unwritable_redirects`; pass *redirects* to reuse one walk across
    many checks). An ``OSError`` from the walk counts as inside. False for a
    path outside the project.
    """
    try:
        rel = resolved.relative_to(cwd_resolved).parts
    except ValueError:
        return False
    if unwritable_component(rel) is not None:
        return True
    if redirects is None:
        try:
            redirects = unwritable_redirects(cwd_resolved)
        except OSError:
            return True
    return any(resolved.is_relative_to(target) for target in redirects)


def has_other_hard_link(resolved: Path) -> bool:
    """Whether *resolved* is an existing file that has another name (a hard
    link). False when it does not exist; an ``OSError`` while checking counts
    as True."""
    try:
        st = os.stat(resolved, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return stat.S_ISREG(st.st_mode) and st.st_nlink > 1


def restricted_write_refusal(cwd: Path, path: str, *,
                             walk_memo: Optional[dict] = None) -> Optional[str]:
    """Why a restricted session must refuse to write the model-named *path*,
    or None when it may.

    Refused: a path whose own components relative to *cwd* name a
    ``RESTRICTED_UNWRITABLE_DIRS`` entry (checked before anything collapses
    ``a/../.localcoder/x`` or a backslash-separated spelling); a location
    :func:`_confine` resolves inside one, or inside a folder such an entry
    anywhere in the project redirects to; an existing file with another hard
    link; and any path when the project walk itself fails. A path ``_confine``
    refuses is not reported here: the write tool refuses it on its own.

    *walk_memo*, when given, holds the outcome of the project walk, so the
    paths of one tool call share a single walk.
    """
    raw = Path(str(path).replace("\\", "/"))
    lexical: tuple = raw.parts
    if raw.is_absolute():
        lexical = ()
        for anchor in (cwd, cwd.resolve()):
            try:
                lexical = raw.relative_to(anchor).parts
                break
            except ValueError:
                continue
    if unwritable_component(lexical) is not None:
        return RESTRICTED_UNWRITABLE_MESSAGE
    try:
        resolved = _confine(cwd, str(path))
    except PermissionError:
        return None
    cwd_resolved = cwd.resolve()
    if walk_memo is not None and "outcome" in walk_memo:
        outcome = walk_memo["outcome"]
    else:
        try:
            outcome = unwritable_redirects(cwd_resolved)
        except OSError as e:
            outcome = e
        if walk_memo is not None:
            walk_memo["outcome"] = outcome
    if isinstance(outcome, OSError):
        return ("the project could not be checked for .localcoder or .git links "
                f"({outcome}), and a restricted session cannot write until it can")
    redirects = outcome
    if (inside_unwritable_dir(cwd_resolved, resolved, redirects)
            or has_other_hard_link(resolved)):
        return RESTRICTED_UNWRITABLE_MESSAGE
    return None


def restricted_unwritable(cwd: Path, path: str) -> bool:
    """Whether a restricted session must refuse to write the model-named *path*
    (:func:`restricted_write_refusal` returns a reason)."""
    return restricted_write_refusal(cwd, path) is not None


@dataclass
class SubprocessResult:
    """Outcome of one :func:`run_subprocess` call.

    A completed process sets *returncode*/*stdout*/*stderr* (``ok`` is
    ``returncode == 0``). A timeout sets *timed_out* and carries whatever the
    process produced before the kill in *stdout*/*stderr* - possibly bytes even
    in text mode (a CPython ``TimeoutExpired`` quirk) - pass the result straight
    to :func:`_partial_on_timeout` to format it. A launch failure sets
    *not_found* (missing executable) or *error* (any other exception); *ok* is
    always False for either.
    """
    ok: bool
    returncode: Optional[int] = None
    stdout: object = ""
    stderr: object = ""
    timed_out: bool = False
    not_found: bool = False
    error: Optional[str] = None
    cancelled: bool = False


def platform_shell(command: str) -> Union[list, str]:
    """The launchable form of *command* run through the platform shell.

    Returns a STRING on Windows and an argument LIST on POSIX, because that is
    what each platform needs to receive the command line UNCHANGED - and this is
    the one definition of that, so no caller has to rediscover it.

    On POSIX an argv list is handed to ``execv`` verbatim, so ``/bin/sh -c`` plus
    the command as one element already arrives intact.

    On Windows it must NOT be ``["cmd", "/C", command]``. ``subprocess`` renders
    an argv list with :func:`subprocess.list2cmdline`, which escapes every
    embedded quote MSVCRT-style as ``\\"`` - syntax ``cmd.exe`` does not speak -
    so a quoted path, the normal way to pass a path containing spaces, reaches
    cmd mangled and cannot be opened. Pre-compensating is not possible:
    list2cmdline turns EVERY quote into ``\\"``, so no list element can put a
    bare quote on the command line. A command STRING is passed to
    ``CreateProcess`` verbatim instead, so cmd applies its own quoting rules to
    exactly what was written.

    The string is ``cmd /S /C "<command>"``. Without ``/S``, ``cmd /C`` strips
    the first and last quote of the line whenever it starts with a quote and
    holds more than two, so a command beginning with a quoted program path
    reaches cmd with its quotes eaten. With ``/S`` cmd removes exactly the one
    outer pair added here, and *command* reaches cmd byte for byte.
    """
    if sys.platform == "win32":
        return 'cmd /S /C "' + command + '"'
    return ["/bin/sh", "-c", command]


# How often a running subprocess is checked against its cancel token.
_CANCEL_POLL_SECONDS = 0.25

# Longest a finished or killed process's output pipes are waited on.
_DRAIN_SECONDS = 1.0


def _kill_tree(proc) -> None:
    """Kill *proc* and every process it started, so a shell-routed command's
    real process dies with its shell instead of running on and holding the
    output pipes open. Best-effort at every step; never raises."""
    children = []
    try:
        import psutil
        children = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        except Exception:
            pass
    for child in children:
        try:
            child.kill()
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def run_subprocess(
    argv_or_cmd: Union[list, str],
    cwd: Path,
    *,
    timeout: float,
    shell_wrap: bool = False,
    env: Optional[dict] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> SubprocessResult:
    """
    Run a subprocess, capturing stdout+stderr as text with a timeout.

    *argv_or_cmd* is an argument list, run directly, unless *shell_wrap* is
    true - then it must be a command STRING, routed through the platform
    shell by :func:`platform_shell` so shell operators (pipes, redirects,
    ``&&``) work. An already-routed caller (``tools/shell.py:_shell_argv``) may
    also pass the string that function returns for the Windows shell route; a
    bare string is a raw Windows command line, never a POSIX one.

    On a timeout, the process's captured stdout/stderr up to the kill is
    preserved on the result, not dropped - format it for display with
    :func:`_partial_on_timeout`. *cancel*, when given, is polled while the
    process runs; once it answers True the process is killed and the result
    carries ``cancelled=True`` plus the output captured so far. This is the
    canonical subprocess-execution primitive for the coder's tools/shell.py,
    tools/git.py, and cli/goal.py.

    Output collection never outlives the process by more than
    :data:`_DRAIN_SECONDS`: a descendant that left the process tree and kept
    the output pipes open cannot stretch a timeout, a cancel or a normal exit;
    the result then carries the output read so far.
    """
    argv = platform_shell(argv_or_cmd) if shell_wrap else argv_or_cmd
    from localm.config import user_command_env
    env = user_command_env(env)
    return _run_cancellable(argv, cwd, timeout=timeout, env=env, cancel=cancel)


def _read_pipe(stream, sink: bytearray) -> None:
    """Append everything *stream* yields to *sink* until EOF or a read error."""
    try:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                return
            sink.extend(chunk)
    except (OSError, ValueError):
        return


def _decode_output(raw: bytearray) -> str:
    """Decode captured pipe bytes as UTF-8 text with universal newlines."""
    return bytes(raw).decode("utf-8", errors="replace").replace(
        "\r\n", "\n").replace("\r", "\n")


def _drain(proc, readers, sinks) -> tuple[str, str]:
    """Collect the output of an exited or killed *proc* within _DRAIN_SECONDS.

    Each pipe is closed only once its reader thread has finished; a reader
    still blocked on a pipe held open by an escaped descendant is left to end
    on its own (it is a daemon thread) and the output read so far is returned.
    """
    deadline = time.monotonic() + _DRAIN_SECONDS
    for t in readers:
        t.join(max(0.0, deadline - time.monotonic()))
    for stream, t in zip((proc.stdout, proc.stderr), readers, strict=True):
        if not t.is_alive():
            try:
                stream.close()
            except Exception:
                pass
    return _decode_output(sinks[0]), _decode_output(sinks[1])


def _reap(proc) -> None:
    """Wait for a killed *proc* to be collected, bounded by _DRAIN_SECONDS."""
    try:
        proc.wait(timeout=_DRAIN_SECONDS)
    except Exception:
        pass


def _run_cancellable(argv, cwd: Path, *, timeout: float, env: Optional[dict],
                     cancel: Optional[Callable[[], bool]]) -> SubprocessResult:
    """The body of :func:`run_subprocess`: the process is polled every
    _CANCEL_POLL_SECONDS against *cancel* (when given) and killed the moment it
    answers True, or at *timeout*, keeping the output captured so far.

    stdout and stderr are read by daemon threads, so the wait on the process
    never depends on the pipes reaching EOF. stdin is the null device: the
    process never reads the coder's terminal."""
    try:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
    except FileNotFoundError as e:
        return SubprocessResult(ok=False, not_found=True, error=str(e))
    except Exception as e:
        return SubprocessResult(ok=False, error=str(e))

    sinks = (bytearray(), bytearray())
    readers = [
        threading.Thread(target=_read_pipe, args=(stream, sink), daemon=True)
        for stream, sink in zip((proc.stdout, proc.stderr), sinks, strict=True)
    ]
    for t in readers:
        t.start()

    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_tree(proc)
                _reap(proc)
                stdout, stderr = _drain(proc, readers, sinks)
                return SubprocessResult(
                    ok=False, timed_out=True, stdout=stdout, stderr=stderr)
            if cancel is not None and cancel():
                _kill_tree(proc)
                _reap(proc)
                stdout, stderr = _drain(proc, readers, sinks)
                return SubprocessResult(
                    ok=False, cancelled=True, stdout=stdout, stderr=stderr)
            try:
                proc.wait(timeout=min(remaining, _CANCEL_POLL_SECONDS))
            except subprocess.TimeoutExpired:
                continue
            break
    except Exception as e:
        _kill_tree(proc)
        _reap(proc)
        _drain(proc, readers, sinks)
        return SubprocessResult(ok=False, error=str(e))
    except BaseException:
        _kill_tree(proc)
        _reap(proc)
        _drain(proc, readers, sinks)
        raise

    stdout, stderr = _drain(proc, readers, sinks)
    return SubprocessResult(
        ok=(proc.returncode == 0), returncode=proc.returncode,
        stdout=stdout, stderr=stderr,
    )
