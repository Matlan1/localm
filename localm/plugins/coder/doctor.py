# SPDX-License-Identifier: AGPL-3.0-or-later
"""Environment checks for ``localm coder doctor`` and the REPL ``/doctor``.

Each check answers one question about the session the coder is about to run and
returns a :class:`Check`. A check that raises is reported as a ``fail`` carrying
the exception, never skipped, so one broken probe cannot hide the others.

Statuses: ``ok`` (healthy), ``info`` (nothing wrong, worth knowing), ``warn``
(works, but degraded), ``fail`` (the session will not work as configured),
``skip`` (nothing to check). Only ``fail`` makes :func:`report_ok` false.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

OK, INFO, WARN, FAIL, SKIP = "ok", "info", "warn", "fail", "skip"

# Repo-map share of the context window above which the map is reported.
MAP_WARN_FRACTION = 0.25
MAP_FAIL_FRACTION = 0.50

_GIT_TIMEOUT_S = 15

# Longest transport error quoted in a report line.
_ERROR_CHARS = 120


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _guard(name: str, fn: Callable[[], Check | list[Check]]) -> list[Check]:
    """Run one check; an exception becomes a ``fail`` that names it."""
    try:
        out = fn()
    except Exception as exc:                                   # noqa: BLE001
        return [Check(name, FAIL, f"check raised {type(exc).__name__}: {exc}")]
    return out if isinstance(out, list) else [out]


# ------------------------------------------------------------------ #
#  Individual checks                                                   #
# ------------------------------------------------------------------ #

def check_model(backend) -> Check:
    """Does the model server answer, accept the key, and list the model?"""
    probe = getattr(backend, "probe", None)
    model = getattr(backend, "model_id", "(unknown)")
    if not callable(probe):
        return Check("model", SKIP,
                     f"{model}: backend has no HTTP endpoint to probe")
    res = probe()
    status = res.get("status")
    if status is None:
        return Check("model", FAIL,
                     f"{model}: server not reachable at {res.get('url')} "
                     f"({str(res.get('error'))[:_ERROR_CHARS]})")
    if status in (401, 403):
        return Check("model", FAIL,
                     f"{model}: server rejected the API key (HTTP {status})")
    if status >= 500:
        return Check("model", FAIL, f"{model}: server error (HTTP {status})")
    if not 200 <= status < 300:
        return Check("model", WARN,
                     f"{model}: server answered HTTP {status} for /models, so "
                     "its model list could not be read")
    models = res.get("models")
    if models is None:
        return Check("model", OK, f"{model}: server reachable")
    if model not in models and model not in ("", "localm"):
        return Check("model", WARN,
                     f"{model}: server reachable but does not list this model "
                     f"({len(models)} listed)")
    return Check("model", OK, f"{model}: server reachable")


def check_tools(backend) -> Check:
    """Which tool-call protocol will this session speak?"""
    native = (bool(getattr(backend, "native_tools", False))
              and bool(getattr(backend, "supports_native_tools", True)))
    grammar = bool(getattr(backend, "supports_grammar", False))
    if native:
        return Check("tool-calls", OK, "native tool calls (OpenAI tools API)")
    if grammar:
        return Check("tool-calls", OK,
                     "XML tool calls, grammar-constrained by the localm server")
    return Check("tool-calls", WARN,
                 "XML tool calls taught in the prompt and not grammar-"
                 "constrained: a small model may emit malformed calls. "
                 "--native-tools (with --url) uses the server's tools API "
                 "where it has one")


def check_context(backend, work_dir: Path, project_map=None) -> Check:
    """Context window against the size of the repo map injected into the prompt."""
    cap = None
    get_cap = getattr(backend, "context_capacity", None)
    if callable(get_cap):
        cap = get_cap()
    if project_map is None:
        from .indexer import ProjectMap
        project_map = ProjectMap.build(work_dir)
    map_tokens = len(project_map.to_context_string()) // 4
    files = project_map.file_count()
    if not isinstance(cap, int) or cap <= 0:
        return Check("context", INFO,
                     f"window not reported by this server; repo map is "
                     f"~{map_tokens} tokens ({files} files)")
    share = map_tokens / cap
    text = (f"window {cap} tokens; repo map ~{map_tokens} tokens "
            f"({files} files, {share:.0%} of the window)")
    if share >= MAP_FAIL_FRACTION:
        return Check("context", FAIL, text + ": too little room left for the "
                     "conversation; load the model with a larger context")
    if share >= MAP_WARN_FRACTION:
        return Check("context", WARN, text + ": leaves less room for the "
                     "conversation")
    return Check("context", OK, text)


def _git(work_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(work_dir), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=_GIT_TIMEOUT_S)


def check_git(work_dir: Path) -> Check:
    """Is the working directory a git work tree, on which branch, how dirty?"""
    try:
        return _check_git(work_dir)
    except FileNotFoundError:
        return Check("git", WARN, "git was not found on PATH")
    except subprocess.TimeoutExpired:
        return Check("git", WARN,
                     f"git did not answer within {_GIT_TIMEOUT_S} s")


def _check_git(work_dir: Path) -> Check:
    inside = _git(work_dir, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return Check("git", WARN, "not a git work tree")
    branch = _git(work_dir, "symbolic-ref", "--short", "-q", "HEAD")
    label = (branch.stdout.strip() if branch.returncode == 0
             else "detached HEAD")
    status = _git(work_dir, "status", "--porcelain")
    if status.returncode != 0:
        return Check("git", WARN,
                     f"branch {label}; git status failed: "
                     f"{status.stderr.strip()[:200]}")
    changed = len([ln for ln in status.stdout.splitlines() if ln.strip()])
    return Check("git", OK if not changed else INFO,
                 f"branch {label}, "
                 + ("clean" if not changed else f"{changed} changed file(s)"))


def _first_word(command) -> str:
    if isinstance(command, (list, tuple)):
        return str(command[0]) if command else ""
    try:
        words = shlex.split(str(command))
    except ValueError:
        words = str(command).split()
    return words[0] if words else ""


def check_verify(work_dir: Path, verify_cmd=None, *,
                 verify_off: bool = False) -> Check:
    """Is there a verify command, and does its program resolve? ``verify_off``
    reports a session that has verification turned off."""
    from .verify import command_text, detect_verify_command
    if verify_off:
        return Check("verify", INFO,
                     "verification is off for this session (/verify auto "
                     "turns it back on)")
    command = verify_cmd or detect_verify_command(work_dir)
    if not command:
        return Check("verify", INFO,
                     "no verify command: a turn that changes files is not "
                     "gated on an exit-code check (set `verify` in "
                     ".localcoder/config.toml or pass --verify)")
    shown = command_text(command)
    word = _first_word(command)
    if word and shutil.which(word) is None:
        return Check("verify", WARN,
                     f"`{shown}`: `{word}` not found on PATH (fine only if "
                     "it is a shell builtin)")
    return Check("verify", OK, f"`{shown}` resolves")


def check_project_config(work_dir: Path) -> Check:
    """Does the project config parse? An unreadable one refuses the session."""
    from .project_config import (
        ProjectConfigUnreadable,
        find_project_config,
        load_project_config,
    )
    path = find_project_config(work_dir)
    if path is None:
        return Check("config", INFO, "no .localcoder/config.toml")
    try:
        load_project_config(work_dir)
    except ProjectConfigUnreadable as exc:
        return Check("config", FAIL, f"{path.name} unreadable: {exc}")
    return Check("config", OK, f"{path.name} readable")


def check_mcp(work_dir: Path) -> list[Check]:
    """Start each configured MCP server on its own throwaway process, count its
    tools, and stop it. Never touches the session's pooled servers."""
    from .mcp import MCPError, MCPServer, load_mcp_config
    from .project_config import ProjectConfigUnreadable
    try:
        configured = load_mcp_config(work_dir)
    except ProjectConfigUnreadable as exc:
        return [Check("mcp", FAIL, f"project config unreadable: {exc}")]
    if not configured:
        return [Check("mcp", SKIP, "no MCP servers configured")]
    out: list[Check] = []
    for name, spec in configured.items():
        label = f"mcp:{name}"
        server = MCPServer(name=name, command=spec["command"],
                           args=spec.get("args", []), env=spec.get("env"),
                           trusted=bool(spec.get("trusted", False)))
        try:
            server.start()
            out.append(Check(label, OK, f"{len(server.tools)} tool(s)"))
        except MCPError as exc:
            out.append(Check(label, FAIL, str(exc)))
        except Exception as exc:                               # noqa: BLE001
            out.append(Check(label, FAIL,
                             f"failed to start: {type(exc).__name__}: {exc}"))
        finally:
            server.stop()
    return out


# ------------------------------------------------------------------ #
#  Runner and formatting                                               #
# ------------------------------------------------------------------ #

def run_doctor(work_dir: Path, backend, *, verify_cmd=None,
               verify_off: bool = False, project_map=None,
               check_mcp_servers: bool = True) -> list[Check]:
    """Run every check and return the results in a fixed order."""
    checks: list[Check] = []
    checks += _guard("model", lambda: check_model(backend))
    checks += _guard("tool-calls", lambda: check_tools(backend))
    checks += _guard("context",
                     lambda: check_context(backend, work_dir, project_map))
    checks += _guard("git", lambda: check_git(work_dir))
    checks += _guard("verify", lambda: check_verify(work_dir, verify_cmd,
                                                 verify_off=verify_off))
    checks += _guard("config", lambda: check_project_config(work_dir))
    if check_mcp_servers:
        checks += _guard("mcp", lambda: check_mcp(work_dir))
    return checks


def report_ok(checks: list[Check]) -> bool:
    """True when no check failed."""
    return not any(c.status == FAIL for c in checks)


_MARK = {OK: "ok  ", INFO: "info", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}


def format_text(checks: list[Check]) -> str:
    """Plain-text report: one ``[mark] name  detail`` line per check."""
    from localm.console import strip_controls
    width = max((len(c.name) for c in checks), default=0)
    lines = [f"[{_MARK[c.status]}] {c.name.ljust(width)}  "
             f"{strip_controls(c.detail)}" for c in checks]
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append(f"{len(checks)} checks: {failed} failed, {warned} warning(s)")
    return "\n".join(lines)


def format_json(checks: list[Check]) -> str:
    """One JSON object: ``{"ok": bool, "checks": [...]}``."""
    return json.dumps({"ok": report_ok(checks),
                       "checks": [c.as_dict() for c in checks]}, indent=2)


def format_jsonl(checks: list[Check]) -> str:
    """One ``check`` record per line, then a closing ``summary`` record."""
    rows = [{"type": "check", **c.as_dict()} for c in checks]
    rows.append({"type": "summary", "ok": report_ok(checks),
                 "failed": sum(c.status == FAIL for c in checks),
                 "warnings": sum(c.status == WARN for c in checks)})
    return "\n".join(json.dumps(r) for r in rows)


def render(checks: list[Check], output_format: str = "text") -> str:
    fmt = (output_format or "text").lower()
    if fmt == "json":
        return format_json(checks)
    if fmt == "jsonl":
        return format_jsonl(checks)
    return format_text(checks)


__all__ = [
    "Check", "run_doctor", "report_ok", "render",
    "format_text", "format_json", "format_jsonl",
    "OK", "INFO", "WARN", "FAIL", "SKIP",
]
