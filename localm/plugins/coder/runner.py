# SPDX-License-Identifier: AGPL-3.0-or-later
"""
One-shot coder runs, shared by the CLI and the in-process MCP tool.

``resolve_task_config`` turns a project directory plus caller overrides into
the settings a run needs: the project's ``.localcoder/config.toml`` first, then
the same defaults the CLI applies. ``build_agent`` constructs the Agent from
them, ``run_single_task`` runs one task and reports the outcome, and
``finish_agent`` closes it. The CLI wires its flags into these; the MCP server
calls them with an engine it already holds.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agent import Agent
from .agent.constants import _SHELL_EXEC_TOOLS
from .audit import SessionMode, parse_mode
from .display import print_warning
from .project_config import load_project_config

DEFAULT_MAX_TURNS = 40

# Grace after a timed-out run is cancelled before it is reported as abandoned
# (still winding down: the tool call in flight is killed or refused, the loop
# stops at its next check, then the agent closes).
STOP_GRACE_SECONDS = 30.0


class InvalidSessionMode(ValueError):
    """The requested session mode is not one of privacy, log or full."""


@dataclass(frozen=True)
class TaskConfig:
    """Resolved settings for one coder run."""
    model: Optional[str]
    max_turns: int
    auto_approve: bool
    always_confirm: frozenset
    session_mode: SessionMode
    gen_kw: dict


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one task: the agent's final text plus its counters.

    ``success`` is True only when the run completed normally AND no tool call
    was denied for want of a confirmation (see ``denied``). ``denied`` lists
    those calls as ``(tool name, reason)``, reason ``"lenient"`` (the call was
    not in the <tool_call> format) or ``"unconfirmable"`` (the tool always
    needs a confirmation this run could not give); a denied call that later
    ran with identical arguments is not listed. A sub-agent's denied calls
    are listed with their tool name prefixed ``sub-agent:``."""
    success: bool
    response: str
    turns: int
    total_tokens: int
    timed_out: bool = False
    denied: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "response": self.response,
            "turns": self.turns,
            "total_tokens": self.total_tokens,
            "denied": [{"tool": name, "reason": reason} for name, reason in self.denied],
        }


_DENIED_REASONS = {
    "lenient": ("the call was not written in the <tool_call> format, and a "
                "loosely formatted call needs a confirmation an unattended "
                "run cannot give"),
    "unconfirmable": ("the tool needs a confirmation an unattended run cannot "
                      "give"),
}


def describe_denied(denied) -> str:
    """One paragraph naming every denied call and why, or "" when none.

    Each entry of *denied* is ``(tool name, reason)`` as in
    ``TaskResult.denied``. Nothing a denied call would have done happened,
    and the text says so."""
    if not denied:
        return ""
    counts: dict[tuple[str, str], int] = {}
    for name, reason in denied:
        counts[(name, reason)] = counts.get((name, reason), 0) + 1
    parts = []
    for (name, reason), n in counts.items():
        times = f" x{n}" if n > 1 else ""
        parts.append(f"{name}{times}: {_DENIED_REASONS.get(reason, reason)}")
    total = sum(counts.values())
    calls = "call was" if total == 1 else "calls were"
    return (f"{total} tool {calls} denied and did not run, so nothing "
            f"{'it' if total == 1 else 'they'} would have done happened: "
            + "; ".join(parts) + ".")


def resolve_task_config(work_dir: Path, *, model: Optional[str] = None,
                        max_turns: Optional[int] = None,
                        max_tokens: Optional[int] = None,
                        temperature: Optional[float] = None,
                        seed: Optional[int] = None, yes: bool = False,
                        interactive_confirm: bool = False,
                        mode: Optional[str] = None) -> TaskConfig:
    """Resolve a run's settings: explicit arguments win, then the project's
    ``.localcoder/config.toml``, then the defaults.

    Raises ``ProjectConfigUnreadable`` when the project config exists but
    cannot be read, and ``InvalidSessionMode`` for an unknown mode."""
    proj_cfg = load_project_config(work_dir)
    if model is None:
        model = proj_cfg.get("model")
    if max_turns is None:
        max_turns = int(proj_cfg.get("max_turns", DEFAULT_MAX_TURNS))
    if max_tokens is None:
        cfg_max_tokens = proj_cfg.get("max_tokens")
        if cfg_max_tokens is not None:
            max_tokens = int(cfg_max_tokens)
        else:
            from .harness_profiles import cli_max_tokens
            max_tokens = cli_max_tokens(model)
    if temperature is None and "temperature" in proj_cfg:
        temperature = float(proj_cfg["temperature"])
    if seed is None and "seed" in proj_cfg:
        seed = int(proj_cfg["seed"])
    if not yes and proj_cfg.get("auto_approve"):
        yes = True
    always_confirm: set = set()
    if interactive_confirm:
        always_confirm.update(_SHELL_EXEC_TOOLS)
    cfg_confirm = proj_cfg.get("always_confirm", [])
    if isinstance(cfg_confirm, list):
        always_confirm.update(cfg_confirm)
    if mode is None:
        mode = proj_cfg.get("mode")
    if mode is None:
        from localm.audit import effective_mode
        mode = effective_mode("coder").value
    try:
        session_mode = parse_mode(mode)
    except ValueError as exc:
        raise InvalidSessionMode(str(exc)) from exc
    gen_kw = {k: v for k, v in [
        ("temperature", temperature),
        ("max_tokens", max_tokens),
        ("seed", seed),
    ] if v is not None}
    return TaskConfig(model=model, max_turns=max_turns, auto_approve=yes,
                      always_confirm=frozenset(always_confirm),
                      session_mode=session_mode, gen_kw=gen_kw)


def unattended_shell_gated(task: str, auto_approve: bool) -> bool:
    """True when a one-shot task runs without ``--yes``: the shell tools then
    need a confirmation the run cannot give, so they are denied."""
    return bool(task) and not auto_approve


def build_agent(backend, work_dir: Path, *, task: str, max_turns: int,
                auto_approve: bool, always_confirm, session_mode: SessionMode,
                gen_kw: Optional[dict] = None, verbose: bool = False,
                dry_run: bool = False, scope: Optional[str] = None,
                custom_instructions: Optional[str] = None, verify_cmd=None,
                browser_enabled: bool = False, on_event=None) -> Agent:
    """Construct the Agent for a session the way the CLI does.

    A one-shot task auto-approves file writes; without ``auto_approve`` the
    shell tools are added to ``always_confirm`` so an unattended run denies
    them instead of executing unconfirmed."""
    always_confirm = set(always_confirm or ())
    if unattended_shell_gated(task, auto_approve):
        always_confirm = set(always_confirm) | set(_SHELL_EXEC_TOOLS)
    return Agent(
        backend=backend,
        cwd=work_dir,
        name="localcoder",
        max_turns=max_turns,
        verbose=verbose,
        auto_approve=auto_approve or (task != ""),
        always_confirm=always_confirm,
        dry_run=dry_run,
        mode=session_mode,
        scope=scope,
        custom_instructions=custom_instructions,
        verify_cmd=verify_cmd,
        browser_enabled=browser_enabled,
        on_event=on_event,
        **(gen_kw or {}),
    )


def warn_unfinished_background(agent) -> None:
    """Report background sub-agents this one-shot run is leaving behind.

    The turn-boundary drain only fires at the START of a turn, so a child that is
    still running (or that finished after the final turn) is never folded in.
    Ending silently would drop work the user explicitly asked for. What survives
    is stated exactly: a committed branch does, a running child does not. Only
    this run's own jobs are reported and drained: the registry is process-wide
    and another run's completions belong to that run.
    """
    try:
        from .background import get_registry
        registry = get_registry()
        owner = getattr(agent, "job_owner", None)
        running = [j for j in registry.list_status(kind="agent", owner=owner)
                   if j["state"] == "running"]
        pending = registry.drain_finished(kind="agent", owner=owner)
    except Exception:
        return

    # Completions evicted before any drain saw them. Its OWN try, after the drain:
    # drain_finished CONSUMES, so a failure folded into the same try would discard
    # completions already handed over.
    try:
        lost = registry.take_dropped_undrained("agent", owner=owner)
    except Exception:
        lost = 0

    if lost:
        print_warning(
            f"{lost} background sub-agent completion(s) were discarded before "
            "they could be collected, so their results are lost.")
    for st in pending:
        branch = (st.get("result") or {}).get("branch")
        where = (f" Its work is committed on branch '{branch}'."
                 if branch else "")
        print_warning(
            f"background sub-agent '{st.get('label')}' ({st.get('id')}) finished "
            f"after the last turn, so its result was not folded into this run."
            f"{where}")
    for st in running:
        print_warning(
            f"background sub-agent '{st.get('label')}' ({st.get('id')}) is STILL "
            "RUNNING: this one-shot run has ended, so it is being stopped and "
            "its result will not be folded in. Use an interactive session for "
            "background delegation, or spawn_agent (synchronous) for a one-shot.")


def stop_unfinished_background(agent, reason: str = "the run that started it ended") -> int:
    """Cancel this run's background sub-agents that are still running, so a
    child never keeps writing after the run that asked for it has reported.
    Returns how many were cancelled. Best-effort, never raises."""
    stopped = 0
    try:
        from .background import get_registry
        registry = get_registry()
        owner = getattr(agent, "job_owner", None)
        for st in registry.list_status(kind="agent", owner=owner):
            if st.get("state") != "running":
                continue
            job = registry.get(st["id"])
            child = getattr(job, "child", None)
            cancel = getattr(child, "cancel", None)
            if callable(cancel):
                cancel(reason)
                stopped += 1
    except Exception:                                       # noqa: BLE001
        return stopped
    return stopped


def warn_sensitive_changes(agent) -> None:
    """Surface test / CI-config edits so a green check over rewritten tests is
    reviewed, not trusted. Best-effort: never let this advisory break the
    session."""
    try:
        from .review_guard import classify_sensitive_changes, render_warning
        message = render_warning(classify_sensitive_changes(agent.changed_files()))
        if message:
            print_warning(message)
    except Exception:                                       # noqa: BLE001
        pass


def browser_enabled() -> bool:
    """Whether a session that holds every capability may drive the browser:
    only the ``browser_enabled`` setting is left to check. An unreadable
    config answers False."""
    try:
        from localm.config import load_config
        return bool(load_config().get("browser_enabled", False))
    except Exception:
        return False


def run_single_task(agent: Agent, task: str) -> TaskResult:
    """Run one task to completion and report the outcome. A run in which a
    tool call was denied for want of a confirmation is not a success. A
    background sub-agent the run leaves behind is reported and then
    cancelled."""
    response = agent.run_task(task)
    denied = tuple(agent.denied_unconfirmed)
    success = agent.last_run_ok and not denied
    warn_unfinished_background(agent)
    stop_unfinished_background(agent)
    return TaskResult(success=success, response=response, turns=agent.turns,
                      total_tokens=agent.total_tokens, denied=denied)


def finish_agent(agent: Agent) -> Optional[Path]:
    """Close the agent: surface sensitive edits, then finalise the session.
    Returns the transcript path when the mode writes one."""
    warn_sensitive_changes(agent)
    return agent.close()


def run_task_with_timeout(agent: Agent, task: str, timeout: Optional[float],
                          *, on_finished=None) -> TaskResult:
    """Run one task on a worker thread, then close the agent there.

    ``on_finished`` runs on the worker thread once the agent is closed, whether
    or not the caller is still waiting. When ``timeout`` elapses the run is
    CANCELLED (``Agent.cancel``): no further tool call runs, the tool call in
    flight is killed or refused, the generation in flight is aborted, and every
    child of the run is cancelled with it. If the worker has not closed within
    STOP_GRACE_SECONDS the run is reported as timed out and left to wind down
    and close on its own; an error it raises after that is logged. The host is
    long-lived, so the close-time reflection runs on its own thread."""
    box: dict = {}
    agent.reflect_in_background = True

    def _work():
        try:
            box["result"] = run_single_task(agent, task)
        except BaseException as e:     # noqa: BLE001
            box["error"] = e
        finally:
            try:
                finish_agent(agent)
            except Exception as e:     # noqa: BLE001
                box["close_error"] = e
            if on_finished is not None:
                on_finished()
            if box.get("cancelled"):
                _report_late_failure(box)

    worker = threading.Thread(target=_work, name="coder-task", daemon=True)
    try:
        worker.start()
    except BaseException:
        try:
            finish_agent(agent)
        finally:
            if on_finished is not None:
                on_finished()
        raise
    worker.join(timeout)
    if not worker.is_alive():
        if "error" in box:
            raise box["error"]
        if "close_error" in box:
            raise box["close_error"]
        return box["result"]
    box["cancelled"] = True
    agent.cancel(f"timed out after {timeout:g}s")
    worker.join(STOP_GRACE_SECONDS)
    note = f"coder task timed out after {timeout:g}s and was cancelled"
    if worker.is_alive():
        response = (f"{note}: no further tool call will run; it is being "
                    "wound down")
    else:
        tails = []
        if "result" in box and box["result"].response:
            tails.append(box["result"].response)
        for key in ("error", "close_error"):
            if key in box:
                tails.append(f"{key} while winding down: {box[key]}")
        response = "\n".join([note, *tails])
    return TaskResult(success=False, response=response, turns=agent.turns,
                      total_tokens=agent.total_tokens, timed_out=True,
                      denied=tuple(agent.denied_unconfirmed))


def _report_late_failure(box: dict) -> None:
    """Log a failure raised by a cancelled run while it wound down, so it is
    never silent even when the caller has already stopped listening."""
    from localm.debuglog import logger
    for key in ("error", "close_error"):
        if key in box:
            logger.warning("coder task (cancelled after its timeout): %s: %s",
                           key, box[key])
            print_warning(f"cancelled coder task: {key}: {box[key]}")
