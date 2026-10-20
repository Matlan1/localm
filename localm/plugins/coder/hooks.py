# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project tool hooks: user commands run before and after a coder tool call.

Declared in ``.localcoder/config.toml``::

    [[hooks.pre_tool_use]]
    matcher = "write_file|edit_file"     # regex matched against the whole tool
                                         # name; "*" or absent matches every tool
    command = "python scripts/guard.py"  # a string runs through the platform
                                         # shell; a list runs as an argv
    timeout = 30                         # seconds (default 60, at most 600)

    [[hooks.post_tool_use]]
    matcher = "write_file|edit_file|patch_file"
    command = ["ruff", "check", "--quiet", "."]

Each hook gets one JSON object on stdin, ASCII-only (other characters are
``\\uXXXX`` escapes)::

    {"event": "pre_tool_use", "tool": "write_file", "args": {...},
     "cwd": "<session folder>", "project_dir": "<folder holding .localcoder>"}

and a ``post_tool_use`` hook also gets ``"ok"`` (whether the tool succeeded),
``"output"`` (the tool's output, at most _PAYLOAD_OUTPUT_CAP characters) and
``"output_truncated"``. The command runs in the project folder with
``LOCALM_CODER_HOOK_EVENT``, ``LOCALM_CODER_TOOL`` and
``LOCALM_CODER_PROJECT_DIR`` set.

Exit codes:

* ``pre_tool_use``: 0 lets the call proceed. 2 blocks it, and the hook's stderr
  is the reason the model is given. Any other outcome (another exit code, a
  timeout, a command that cannot start) also blocks the call: a gate that could
  not give a verdict does not pass. An invalid ``pre_tool_use`` entry blocks
  every tool call until it is fixed.
* Untrusted hooks never run. A call an untrusted ``pre_tool_use`` hook matches
  is refused, because its check cannot run; untrusted ``post_tool_use`` hooks
  are skipped.
* ``post_tool_use``: 0 does nothing. 2 adds the hook's stderr to the tool result
  the model sees. Any other outcome is reported to the user as a warning and
  leaves the result unchanged. An invalid entry is reported and skipped.

Hooks of every matching entry run in the order declared; the first
``pre_tool_use`` hook that blocks stops the rest. Hooks never run in a
restricted session, and project hooks run only once trusted (hook_trust.py).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

EVENTS = ("pre_tool_use", "post_tool_use")
PRE, POST = EVENTS

BLOCK_EXIT = 2
DEFAULT_TIMEOUT = 60.0
MAX_TIMEOUT = 600.0

_FEEDBACK_CAP = 4000
_PAYLOAD_OUTPUT_CAP = 100_000
_POLL_SECONDS = 0.25
_DRAIN_SECONDS = 2.0
_OUTPUT_CAP = 1_000_000
_ENTRY_KEYS = frozenset({"matcher", "command", "timeout"})


@dataclass(frozen=True)
class HookSpec:
    """One declared hook. *error* is set when the entry is invalid; an invalid
    entry matches every tool."""
    event: str
    index: int
    matcher: str = "*"
    command: Union[str, tuple[str, ...]] = ""
    timeout: float = DEFAULT_TIMEOUT
    error: Optional[str] = None
    regex: Optional[re.Pattern] = field(default=None, compare=False)

    @property
    def label(self) -> str:
        if self.index == 0:
            return "hooks table"
        return f"{self.event} hook #{self.index}"

    def command_text(self, limit: Optional[int] = 120) -> str:
        """The command as one line; cut to *limit* characters unless None."""
        text = (self.command if isinstance(self.command, str)
                else " ".join(self.command))
        if limit is None or len(text) <= limit:
            return text
        return text[:limit - 3] + "..."

    def matches(self, tool: str) -> bool:
        if self.error is not None or self.regex is None:
            return True
        return self.regex.fullmatch(tool) is not None


@dataclass
class HookOutcome:
    """What running one hook produced. *returncode* is None when the process
    did not finish (timed out, cancelled, or never started)."""
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    start_error: Optional[str] = None

    def describe_failure(self, timeout: float) -> str:
        """A short phrase for an outcome that was not exit 0 or exit 2."""
        if self.start_error is not None:
            return f"could not start ({self.start_error})"
        if self.timed_out:
            return f"timed out after {timeout:g}s"
        if self.cancelled:
            return "was stopped because the run was cancelled"
        tail = _tail(self.stderr or self.stdout)
        return f"exited {self.returncode}" + (f": {tail}" if tail else "")


def _tail(text: str, cap: int = _FEEDBACK_CAP) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap else "..." + text[-cap:]


def _parse_entry(event: str, index: int, entry: Any) -> HookSpec:
    def broken(msg: str) -> HookSpec:
        return HookSpec(event=event, index=index, error=msg)

    if not isinstance(entry, dict):
        return broken("an entry must be a table with a 'command'")
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        return broken("unknown key(s) " + ", ".join(repr(k) for k in unknown))

    command = entry.get("command")
    if isinstance(command, str):
        if not command.strip():
            return broken("'command' is empty")
        cmd: Union[str, tuple[str, ...]] = command
    elif (isinstance(command, list) and command
          and all(isinstance(a, str) for a in command) and command[0].strip()):
        cmd = tuple(command)
    else:
        return broken("'command' must be a non-empty string or list of strings")

    matcher = entry.get("matcher", "*")
    if not isinstance(matcher, str):
        return broken("'matcher' must be a string")
    matcher = matcher.strip()
    regex = None
    if matcher not in ("", "*"):
        try:
            regex = re.compile(matcher)
        except re.error as e:
            return broken(f"'matcher' is not a valid regular expression ({e})")

    timeout = entry.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return broken("'timeout' must be a number of seconds")
    if not 0 < timeout <= MAX_TIMEOUT:
        return broken(f"'timeout' must be above 0 and at most {MAX_TIMEOUT:g}")

    return HookSpec(event=event, index=index, matcher=matcher, command=cmd,
                    timeout=float(timeout), regex=regex)


def parse_hooks(table: Any) -> tuple[list[HookSpec], list[HookSpec], list[str]]:
    """Parse a ``[hooks]`` table into (pre hooks, post hooks, warnings).

    An invalid ``pre_tool_use`` entry is kept, with its error set, so it blocks
    every call; an invalid ``post_tool_use`` entry is dropped. Each invalid
    entry produces one warning. A ``[hooks]`` value that is not a table, and
    each unknown event name in it, becomes one invalid ``pre_tool_use`` entry."""
    pre: list[HookSpec] = []
    post: list[HookSpec] = []
    warnings: list[str] = []
    if not isinstance(table, dict):
        spec = HookSpec(event=PRE, index=0,
                        error="'hooks' must be a table of pre_tool_use / post_tool_use lists")
        warnings.append(f"Tool hooks: {spec.error}; every tool call is blocked until it is fixed.")
        return [spec], post, warnings

    unknown = [HookSpec(event=PRE, index=0,
                        error=f"unknown event '{key}' (known: {', '.join(EVENTS)})")
               for key in sorted(set(table) - set(EVENTS), key=str)]
    for spec in unknown:
        warnings.append(f"Tool hooks: {spec.error}; every tool call is blocked "
                        "until it is fixed.")
    pre.extend(unknown)

    for event, out in ((PRE, pre), (POST, post)):
        raw = table.get(event)
        if raw is None:
            continue
        entries = [raw] if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            entries = [raw]
        for i, entry in enumerate(entries, start=1):
            spec = _parse_entry(event, i, entry)
            if spec.error is None:
                out.append(spec)
            elif event == PRE:
                out.append(spec)
                warnings.append(f"Tool hooks: {spec.label} is invalid ({spec.error}); "
                                "every tool call is blocked until it is fixed.")
            else:
                warnings.append(f"Tool hooks: {spec.label} is invalid ({spec.error}) "
                                "and will not run.")
    return pre, post, warnings


def _json_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _drain(stream, sink: list, cap: int) -> None:
    """Read *stream* to EOF into *sink*, keeping at most *cap* bytes."""
    kept = 0
    try:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                break
            if kept < cap:
                sink.append(chunk[:cap - kept])
                kept += len(sink[-1])
    except (OSError, ValueError):
        pass


def _shell_argv(command: str) -> Union[list, str]:
    """*command* run through the platform shell. On Windows the whole command
    is wrapped in one more pair of quotes for ``cmd /S /C``, which removes
    exactly that pair, so a command that starts with a quoted program path
    reaches cmd unchanged."""
    if sys.platform == "win32":
        return 'cmd /S /C "' + command + '"'
    return ["/bin/sh", "-c", command]


def _feed(stream, data: bytes) -> None:
    """Write *data* to *stream* and close it; a hook that exits or closes its
    stdin early ends the write."""
    try:
        stream.write(data)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def run_hook(spec: HookSpec, payload: dict, project_dir: Path,
             cancel: Optional[Callable[[], bool]] = None) -> HookOutcome:
    """Run *spec*'s command with *payload* as JSON on stdin, in *project_dir*.

    Stdin is written and stdout/stderr are read on their own threads, so the
    spec's timeout and *cancel* are checked every _POLL_SECONDS whatever the
    hook does with its pipes. At the timeout or on cancel the process and
    everything it started are killed. After the hook process ends, output is
    collected for at most _DRAIN_SECONDS more, so a process that escaped the
    kill and still holds a pipe cannot hold the call. Each stream keeps at most
    _OUTPUT_CAP bytes. Never raises for a failure of the command."""
    from .tools.base import _kill_tree
    from localm.config import user_command_env

    argv = (_shell_argv(spec.command) if isinstance(spec.command, str)
            else list(spec.command))
    env = user_command_env()
    env["LOCALM_CODER_HOOK_EVENT"] = spec.event
    env["LOCALM_CODER_TOOL"] = str(payload.get("tool", ""))
    env["LOCALM_CODER_PROJECT_DIR"] = str(project_dir)
    try:
        proc = subprocess.Popen(
            argv, cwd=str(project_dir), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except Exception as e:
        return HookOutcome(start_error=str(e) or type(e).__name__)

    out: list = []
    err: list = []
    threads = [
        threading.Thread(target=_feed, args=(proc.stdin, _json_payload(payload).encode("utf-8")),
                         daemon=True),
        threading.Thread(target=_drain, args=(proc.stdout, out, _OUTPUT_CAP), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err, _OUTPUT_CAP), daemon=True),
    ]
    for t in threads:
        t.start()

    def _collect(**flags) -> HookOutcome:
        drain_deadline = time.monotonic() + _DRAIN_SECONDS
        for t, stream in zip(threads[1:], (proc.stdout, proc.stderr), strict=True):
            t.join(max(0.0, drain_deadline - time.monotonic()))
            if stream is not None and not t.is_alive():
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        if flags.get("timed_out") or flags.get("cancelled"):
            try:
                proc.wait(timeout=_DRAIN_SECONDS)
            except subprocess.TimeoutExpired:
                pass

        def _text(chunks: list) -> str:
            return b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
        return HookOutcome(stdout=_text(out), stderr=_text(err), **flags)

    deadline = time.monotonic() + spec.timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_tree(proc)
                return _collect(timed_out=True)
            if cancel is not None and cancel():
                _kill_tree(proc)
                return _collect(cancelled=True)
            try:
                proc.wait(timeout=min(remaining, _POLL_SECONDS))
            except subprocess.TimeoutExpired:
                continue
            break
    except BaseException:
        _kill_tree(proc)
        raise
    return _collect(returncode=proc.returncode)


@dataclass
class PostHookReport:
    """Text a post hook gave the model (*feedback*) and the problems to show the
    user (*warnings*)."""
    feedback: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ToolHooks:
    """The hooks one project config declares, and whether they may run."""

    def __init__(self, config_path: Path, table: Any, *, trusted: bool = False):
        from .hook_trust import hooks_digest
        self.config_path = Path(config_path)
        self.project_dir = self.config_path.parent.parent
        self.digest = hooks_digest(table)
        self.pre, self.post, self.warnings = parse_hooks(table)
        self.trusted = trusted
        self.trust_source: Optional[str] = "store" if trusted else None
        self.revoked = False
        self.loaded_cwd: Optional[Path] = None
        self._lock = threading.Lock()
        self._untrusted_reported = False

    @classmethod
    def load(cls, cwd: Path) -> Optional[ToolHooks]:
        """The hooks in the project config nearest *cwd*, with ``trusted`` read
        from the trust store. None when there is no config or it has no
        ``hooks`` key. Raises ProjectConfigUnreadable when the config exists but
        cannot be read."""
        from .project_config import find_project_config, read_project_config_file
        from .hook_trust import is_trusted
        path = find_project_config(cwd)
        if path is None:
            return None
        raw = read_project_config_file(path)
        if "hooks" not in raw:
            return None
        hooks = cls(path, raw["hooks"])
        hooks.loaded_cwd = Path(cwd).resolve()
        hooks.trusted = hooks.has_hooks() and is_trusted(path, hooks.digest)
        hooks.trust_source = "store" if hooks.trusted else None
        return hooks

    @classmethod
    def unreadable(cls, config_path: Path, error: str) -> ToolHooks:
        """Hooks for a project config that exists but cannot be parsed and
        mentions hooks: one invalid ``pre_tool_use`` entry, so every tool call
        is refused until the file is fixed."""
        hooks = cls(config_path, {})
        hooks.pre = [HookSpec(event=PRE, index=0,
                              error=f"the project config could not be read ({error})")]
        hooks.warnings = [f"Tool hooks: {config_path} could not be read and mentions "
                          "hooks, so every tool call is blocked until it is fixed."]
        hooks.trusted = True
        hooks.trust_source = "fixed"
        return hooks

    def has_hooks(self) -> bool:
        return bool(self.pre or self.post)

    def applies(self, event: str, tool: str) -> bool:
        specs = self.pre if event == PRE else self.post
        return any(s.matches(tool) for s in specs)

    def trust_for_session(self) -> None:
        self.trusted = True
        self.trust_source = "session"
        self.revoked = False

    def revoke(self) -> None:
        """Stop these hooks for the rest of the session, whatever the trust
        store says."""
        self.trusted = False
        self.trust_source = None
        self.revoked = True

    def refresh_trust(self) -> bool:
        """Re-read the trust store, so trust granted or withdrawn elsewhere
        (another session, the GUI) applies. Hooks trusted for this session only
        stay trusted; revoked hooks stay off. Returns ``trusted``."""
        if self.revoked or not self.has_hooks() or self.trust_source in ("session", "fixed"):
            return self.trusted
        from .hook_trust import is_trusted
        stored = is_trusted(self.config_path, self.digest)
        self.trusted = stored
        self.trust_source = "store" if stored else None
        return self.trusted

    def remember_trust(self) -> None:
        """Trust these hooks now and record it for later sessions. Raises
        OSError or TrustStoreUnreadable when the record cannot be written; the
        hooks are trusted for this session either way."""
        from .hook_trust import remember_trust
        self.trusted = True
        self.trust_source = "session"
        self.revoked = False
        remember_trust(self.config_path, self.digest)
        self.trust_source = "store"

    def claim_untrusted_report(self) -> bool:
        """True exactly once per session: the caller reports that these hooks
        are not trusted and were not run."""
        with self._lock:
            if self._untrusted_reported:
                return False
            self._untrusted_reported = True
            return True

    def untrusted_message(self) -> str:
        return (f"Tool hooks in {self.config_path} are not trusted, so none of them "
                "run: a tool call a pre_tool_use hook would check is refused, and "
                "post_tool_use hooks are skipped. Review them and trust them with "
                "/hooks trust or --trust-hooks, or remove them from the file.")

    def untrusted_block_reason(self, tool: str) -> str:
        return (f"{tool} was not run: the project's pre_tool_use hooks in "
                f"{self.config_path} would check it, and they are not trusted, so "
                "the check cannot run. The user can trust them (/hooks trust or "
                "--trust-hooks) or remove them.")

    def describe(self) -> list[str]:
        """One line per hook, for display to the user."""
        lines = []
        for spec in (*self.pre, *self.post):
            if spec.error is not None:
                lines.append(f"{spec.label}: INVALID ({spec.error})")
            else:
                lines.append(f"{spec.label}: [{spec.matcher}] {spec.command_text(None)} "
                             f"(timeout {spec.timeout:g}s)")
        return lines

    def _payload(self, event: str, tool: str, args: dict, cwd: Path,
                 project_dir: Path) -> dict:
        return {"event": event, "tool": tool, "args": args, "cwd": str(cwd),
                "project_dir": str(project_dir)}

    def run_pre(self, tool: str, args: dict, cwd: Path,
                cancel: Optional[Callable[[], bool]] = None,
                on_start: Optional[Callable[[HookSpec], None]] = None,
                permit: Optional[Callable[[HookSpec], Optional[str]]] = None,
                project_dir: Optional[Path] = None,
                ) -> Optional[str]:
        """Run the matching pre hooks. Returns None when the call may proceed,
        else the reason it is blocked. Only for trusted hooks: the caller checks
        ``trusted`` first.

        *permit*, when given, is asked before each hook runs and returns None
        to run it or the reason it may not; a hook that may not run blocks the
        call. *project_dir* overrides the folder the hooks run in."""
        run_dir = Path(project_dir) if project_dir is not None else self.project_dir
        payload = self._payload(PRE, tool, args, cwd, run_dir)
        for spec in self.pre:
            if not spec.matches(tool):
                continue
            if spec.error is not None:
                return (f"{tool} was not run: the project's {spec.label} in "
                        f"{self.config_path} is invalid ({spec.error}), and every "
                        "tool call is blocked until it is fixed.")
            if permit is not None:
                refusal = permit(spec)
                if refusal is not None:
                    return (f"{tool} was not run: the project's {spec.label} "
                            f"({spec.command_text()}) did not run because {refusal}.")
            if on_start is not None:
                on_start(spec)
            outcome = run_hook(spec, payload, run_dir, cancel)
            if outcome.returncode == 0:
                continue
            if outcome.returncode == BLOCK_EXIT:
                reason = _tail(outcome.stderr) or "(the hook gave no reason on stderr)"
                return (f"{tool} was blocked by the project's {spec.label} "
                        f"({spec.command_text()}): {reason}")
            return (f"{tool} was not run: the project's {spec.label} "
                    f"({spec.command_text()}) {outcome.describe_failure(spec.timeout)}, "
                    "and a pre_tool_use hook that does not exit 0 blocks the call.")
        return None

    def run_post(self, tool: str, args: dict, cwd: Path, ok: bool, output: str,
                 cancel: Optional[Callable[[], bool]] = None,
                 on_start: Optional[Callable[[HookSpec], None]] = None,
                 permit: Optional[Callable[[HookSpec], Optional[str]]] = None,
                 project_dir: Optional[Path] = None,
                 ) -> PostHookReport:
        """Run the matching post hooks. Only for trusted hooks: the caller
        checks ``trusted`` first. *permit* and *project_dir* as for run_pre; a
        hook *permit* refuses is skipped with a warning."""
        report = PostHookReport()
        text = str(output or "")
        run_dir = Path(project_dir) if project_dir is not None else self.project_dir
        payload = self._payload(POST, tool, args, cwd, run_dir)
        payload["ok"] = bool(ok)
        payload["output"] = text[:_PAYLOAD_OUTPUT_CAP]
        payload["output_truncated"] = len(text) > _PAYLOAD_OUTPUT_CAP
        for spec in self.post:
            if not spec.matches(tool):
                continue
            if permit is not None:
                refusal = permit(spec)
                if refusal is not None:
                    report.warnings.append(
                        f"The project's {spec.label} ({spec.command_text()}) after {tool} "
                        f"did not run because {refusal}.")
                    continue
            if on_start is not None:
                on_start(spec)
            outcome = run_hook(spec, payload, run_dir, cancel)
            if outcome.returncode == 0:
                continue
            if outcome.returncode == BLOCK_EXIT:
                report.feedback.append(
                    _tail(outcome.stderr) or "(the hook exited 2 with no message on stderr)")
                continue
            report.warnings.append(
                f"The project's {spec.label} ({spec.command_text()}) after {tool} "
                f"{outcome.describe_failure(spec.timeout)}.")
        return report
